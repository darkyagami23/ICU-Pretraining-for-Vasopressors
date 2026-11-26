"""
finetune.py -- one script for every fine-tuning experiment.

It is your downstream_code.ipynb (same model, loss, sampler, augmentation,
optimiser, scheduler, checkpoint selection, temperature scaling, Youden
threshold and 5-fold ensemble) with these changes:

  * CV folds come from splits/windows.csv (patient-level, same patient -> same
    fold at every horizon). --cv_mode window reproduces the thesis
    StratifiedKFold over windows, to check you can still get the old numbers.
  * --init scratch           : same architecture, random init         (A1 / E4)
  * --freeze_encoder         : encoder frozen, heads trained          (A2 / E5)
  * --label_frac 0.1 --seed s: train on 10% of training PATIENTS      (E7)
  * --modalities ppg,ecg     : any subset of PPG/ABP/ECG              (E9, pairs)
  * --encoder_path X         : any encoder checkpoint (loss ablations A3-A6,
                               in-domain A7, your pair checkpoints)
  * Every run writes per-window predictions WITH case ids, so all statistics
    (patient bootstrap, DeLong, AUPRC, tables) are computed afterwards by
    evaluate.py without retraining.
  * --resume: skips folds/horizons already finished (Kaggle 12 h limit).

Example (Kaggle):
  !python finetune.py --memmap_dir /kaggle/input/<...>/physio_memmap \
      --windows_csv /kaggle/working/splits/windows.csv \
      --init pretrained --encoder_path /kaggle/input/<...>/hybrid_v4_encoders.pth \
      --run_name trimodal_ssl --out_dir /kaggle/working/runs --resume
"""
import argparse
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from common import (CANONICAL, PhysiologicalDatasetMM, mask_edges, build_encoder_bundle, describe_checkpoint,
                    load_encoder_checkpoint, parse_filename, patient_subset, read_state_dict,
                    save_json, seed_everything)

WINDOWS = ['offset_3m', 'offset_5m', 'offset_7m', 'offset_10m', 'offset_12m', 'offset_15m']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MASK = (0, 0)   # (mask_start, mask_end), set from the command line


# =============================================================================
# Augmentation (verbatim from the notebook)
# =============================================================================
def _smooth_curve(L, n_knots, sigma, rng, center=1.0):
    knots = rng.normal(center, sigma, size=n_knots)
    return np.interp(np.arange(L), np.linspace(0, L - 1, n_knots), knots).astype(np.float32)


def augment_waveform(wave_t, rng, p_jitter=0.6, jitter_std=0.05, p_scale=0.5, scale_std=0.05,
                     p_magwarp=0.5, magwarp_sigma=0.10, p_timewarp=0.4, timewarp_sigma=0.10,
                     p_baseline=0.4, baseline_amp=0.10):
    w = wave_t.numpy().copy()
    C, L = w.shape
    for c in range(C):
        sig = w[c]
        if rng.random() < p_jitter:
            sig = sig + rng.normal(0.0, jitter_std, size=L).astype(np.float32)
        if rng.random() < p_scale:
            sig = sig * np.float32(rng.normal(1.0, scale_std))
        if rng.random() < p_magwarp:
            sig = sig * _smooth_curve(L, 4, magwarp_sigma, rng, center=1.0)
        if rng.random() < p_timewarp:
            curve = np.clip(_smooth_curve(L, 4, timewarp_sigma, rng, center=1.0), 0.1, None)
            cum = np.cumsum(curve)
            cum = (cum - cum.min()) / (cum.max() - cum.min() + 1e-8) * (L - 1)
            sig = np.interp(np.arange(L), cum, sig).astype(np.float32)
        if rng.random() < p_baseline:
            sig = sig + _smooth_curve(L, 3, baseline_amp, rng, center=0.0)
        w[c] = sig
    return torch.from_numpy(w.astype(np.float32))


class AugmentingSubset(Dataset):
    def __init__(self, base, indices, augment=False, aug_prob=0.5, seed=42):
        self.base, self.indices = base, list(indices)
        self.augment, self.aug_prob = augment, aug_prob
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        (wave, scal), (label, m, s, d) = self.base[self.indices[i]]
        if self.augment and label.item() > 0.5 and self.rng.random() < self.aug_prob:
            wave = augment_waveform(wave, self.rng)
            wave = mask_edges(wave, *MASK)   # re-mask: augmentation must not put signal back into the masked edges
        return (wave, scal), (label, m, s, d)


# =============================================================================
# Model: same heads as PretrainedMultiTaskHybrid, any modality subset
# =============================================================================
class MultiTaskHybrid(nn.Module):
    CH = {'ppg': 0, 'abp': 1, 'ecg': 2}

    def __init__(self, bundle, modalities, d_model=128, num_scalars=3, n_bp_steps=15, dropout=0.1):
        super().__init__()
        self.encoder = bundle
        self.mods = list(modalities)
        fused_dim = len(self.mods) * d_model + num_scalars
        self.proj = nn.Sequential(nn.Linear(fused_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(256, 1)

        def _reg():
            return nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout / 2),
                                 nn.Linear(128, n_bp_steps))
        self.map_regressor, self.sbp_regressor, self.dbp_regressor = _reg(), _reg(), _reg()
        for name, m in self.named_modules():
            if name.startswith('encoder'):
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.map_regressor[-1].bias.fill_(0.70)
            self.sbp_regressor[-1].bias.fill_(1.20)
            self.dbp_regressor[-1].bias.fill_(0.70)

    def forward(self, waveforms, scalars):
        xs = [waveforms[:, self.CH[m], :] for m in self.mods]
        zs = self.encoder.encode_subset(xs, self.mods)
        fused = torch.cat([z.mean(dim=1) for z in zs] + [scalars], dim=1)
        h = self.proj(fused)
        return (self.classifier(h).squeeze(-1), self.map_regressor(h),
                self.sbp_regressor(h), self.dbp_regressor(h))


def build_model(args):
    bundle = build_encoder_bundle(patch_len=args.patch_len, ecg_encoder=args.ecg_encoder)
    if args.init == 'pretrained':
        pair = tuple(args.pair_order.split(',')) if args.pair_order else None
        missing, _ = load_encoder_checkpoint(bundle, args.encoder_path, pair_order=pair,
                                             verbose=not getattr(args, '_quiet', False))
        args._quiet = True
        # Hard failure if a branch we USE got no pretrained weights at all
        for m in args.mods:
            n_param = sum(1 for k, _ in bundle.named_parameters() if k.startswith(f'enc_{m}.'))
            n_miss = sum(1 for k in missing if k.startswith(f'enc_{m}.'))
            if n_miss and not args.allow_partial_load:
                raise RuntimeError(
                    f'{n_miss}/{n_param} parameters of enc_{m} are NOT in the checkpoint '
                    f'(e.g. {[k for k in missing if k.startswith("enc_" + m)][:3]}). '
                    f'Check --ecg_encoder / --patch_len, or pass --allow_partial_load if intended.')
        if any(k.startswith('shared.') for k in missing) and not args.allow_partial_load:
            raise RuntimeError('Shared encoder weights missing from checkpoint.')
    model = MultiTaskHybrid(bundle, args.mods)
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    return model.to(DEVICE)


# =============================================================================
# Loss / metrics (verbatim logic from the notebook)
# =============================================================================
class FocalLossBCE(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        a_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (a_t * (1 - p_t) ** self.gamma * bce).mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, cls_criterion, alpha=0.5, beta=2.0, huber_delta=0.05):
        super().__init__()
        self.cls_loss, self.huber = cls_criterion, nn.HuberLoss(delta=huber_delta)
        self.alpha, self.beta = alpha, beta

    def forward(self, cls_logit, mp, sp, dp, cls_t, mt, st, dt):
        return (self.alpha * self.cls_loss(cls_logit, cls_t)
                + self.beta * (self.huber(mp, mt) + self.huber(sp, st) + self.huber(dp, dt)))


def pearson_r(pred, target):
    p, t = pred.flatten().float(), target.flatten().float()
    vp, vt = p - p.mean(), t - t.mean()
    return ((vp * vt).sum() / (torch.sqrt((vp ** 2).sum() * (vt ** 2).sum()) + 1e-8)).item()


def pick_threshold_youden(labels, probs):
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, probs)
    best = int(np.argmax(tpr - fpr))
    t = float(thr[best])
    return t if np.isfinite(t) else float(np.max(probs) + 1e-6)


def fit_temperature(logits, labels, max_iter=200):
    logits = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.05, max_iter=max_iter)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(logits / torch.exp(log_T), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(log_T).detach().clamp(0.05, 20.0).item())


@torch.no_grad()
def forward_all(model, loader):
    model.eval()
    out = defaultdict(list)
    for (wave, scal), (lab, mt, st, dt) in loader:
        logit, mp, sp, dp = model(wave.to(DEVICE), scal.to(DEVICE))
        out['logit'].append(logit.float().cpu())
        out['label'].append(lab)
        for k, v in [('mp', mp), ('sp', sp), ('dp', dp)]:
            out[k].append(v.float().cpu())
        for k, v in [('mt', mt), ('st', st), ('dt', dt)]:
            out[k].append(v)
    return {k: torch.cat(v) for k, v in out.items()}


def make_train_loader(pooled, train_idx, train_labels, args):
    subset = AugmentingSubset(pooled, train_idx, augment=not args.no_augment, aug_prob=args.aug_prob,
                              seed=args.seed)
    kw = dict(batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    if args.no_sampler:
        return DataLoader(subset, shuffle=True, **kw)
    lab = np.asarray(train_labels)
    n_pos, n_neg = max(int((lab == 1).sum()), 1), max(int((lab == 0).sum()), 1)
    f = args.sampler_pos_frac
    w_pos = f * n_neg / ((1 - f) * n_pos)
    sw = np.where(lab == 1, w_pos, 1.0).astype(np.float64)
    sampler = WeightedRandomSampler(torch.as_tensor(sw), num_samples=len(sw), replacement=True)
    return DataLoader(subset, sampler=sampler, **kw)


def train_one_fold(model, train_loader, val_loader, args, save_path):
    criterion = MultiTaskLoss(FocalLossBCE(args.focal_alpha, args.focal_gamma), args.alpha, args.beta,
                              args.huber_delta)
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids and p.requires_grad]
    groups = [{'params': head_params, 'lr': args.lr_heads}]
    if enc_params:
        groups.insert(0, {'params': enc_params, 'lr': args.lr_encoder})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)

    best, no_imp, best_ep = -1.0, 0, 0
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()
        for b, ((wave, scal), (lab, mt, st, dt)) in enumerate(train_loader):
            if args.max_train_batches and b >= args.max_train_batches:
                break
            wave, scal, lab = wave.to(DEVICE), scal.to(DEVICE), lab.to(DEVICE)
            mt, st, dt = mt.to(DEVICE), st.to(DEVICE), dt.to(DEVICE)
            opt.zero_grad()
            loss = criterion(*model(wave, scal), lab, mt, st, dt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        o = forward_all(model, val_loader)
        y, p = o['label'].numpy(), torch.sigmoid(o['logit']).numpy()
        if not np.all(np.isfinite(p)):
            if os.path.exists(save_path):
                model.load_state_dict(torch.load(save_path, map_location=DEVICE))
            no_imp += 1
            if no_imp >= args.patience:
                break
            continue
        two = len(np.unique(y)) > 1
        auprc = average_precision_score(y, p) if two else 0.5
        auroc = roc_auc_score(y, p) if two else 0.5
        r = [pearson_r(o['mp'], o['mt']), pearson_r(o['sp'], o['st']), pearson_r(o['dp'], o['dt'])]
        comb = 0.5 * auprc + 0.5 * ((np.mean(r) + 1) / 2)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'    ep {epoch + 1:>3}  comb={comb:.4f}  AUPRC={auprc:.4f}  AUROC={auroc:.4f}', flush=True)
        if comb > best:
            best, no_imp, best_ep = comb, 0, epoch + 1
            torch.save(model.state_dict(), save_path)
        else:
            no_imp += 1
            if no_imp >= args.patience:
                break
    if not os.path.exists(save_path):          # every epoch gave non-finite predictions
        print('    [!] no finite validation epoch; saving last state', flush=True)
        torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    print(f'    [best ep {best_ep}] comb={best:.4f}', flush=True)
    return best_ep


class DeadlineReached(Exception):
    pass


# =============================================================================
# One horizon
# =============================================================================
def run_window(window, args, wtab, run_dir):
    print(f'\n{"=" * 70}\n  {args.run_name}  |  {"+".join(args.mods)}  |  {window}\n{"=" * 70}', flush=True)
    t0 = time.time()
    tr = PhysiologicalDatasetMM(args.memmap_dir, 'train', window, mask_start=args.mask_start, mask_end=args.mask_end)
    va = PhysiologicalDatasetMM(args.memmap_dir, 'val', window, mask_start=args.mask_start, mask_end=args.mask_end)
    te = PhysiologicalDatasetMM(args.memmap_dir, 'test', window, mask_start=args.mask_start, mask_end=args.mask_end)
    pooled = ConcatDataset([tr, va])

    # metadata for pooled + test rows
    def meta(ds, split):
        fn = ds.filenames if ds.filenames is not None else np.array([f'Case_{i:04d}_control' for i in range(ds.N)])
        rows = []
        for i in range(ds.N):
            cid, dose = parse_filename(fn[i])
            rows.append(dict(window=window, split=split, idx=i, filename=fn[i], caseid=cid, dose=dose,
                             label=int(ds.labels[i])))
        return pd.DataFrame(rows)
    pm = pd.concat([meta(tr, 'train'), meta(va, 'val')], ignore_index=True)
    tm = meta(te, 'test')

    if args.cv_mode == 'grouped':
        w = wtab[wtab.window == window]
        pm = pm.merge(w[['split', 'idx', 'filename', 'subjectid', 'fold', 'exclude']],
                      on=['split', 'idx'], how='left', suffixes=('', '_w'))
        assert pm.fold.notna().all(), 'windows.csv does not match the memmaps (re-run make_splits.py)'
        assert (pm.filename == pm.filename_w).all(), 'filename mismatch between memmaps and windows.csv'
        tm = tm.merge(w[['split', 'idx', 'subjectid', 'exclude']], on=['split', 'idx'], how='left')
        folds = []
        for k in range(args.n_folds):
            val_idx = np.where(pm.fold.values == k)[0]
            tr_mask = (pm.fold.values >= 0) & (pm.fold.values != k)
            if args.label_frac < 1.0:
                tr_mask &= pm.subjectid.isin(args._subset).values
            folds.append((np.where(tr_mask)[0], val_idx))
        test_keep = np.where(tm.exclude.fillna(0).values == 0)[0]
    else:  # thesis protocol: StratifiedKFold over windows
        pm['subjectid'], tm['subjectid'], pm['fold'] = pm.caseid, tm.caseid, -1
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        folds = list(skf.split(np.zeros(len(pm)), pm.label.values))
        for k, (_, v) in enumerate(folds):
            pm.loc[v, 'fold'] = k
        test_keep = np.arange(len(tm))

    labels = pm.label.values
    n_folds = min(args.n_folds, args.max_folds) if args.max_folds else args.n_folds
    oof = np.full(len(pm), np.nan)
    ckpts = []
    ckdir = os.path.join(run_dir, 'ckpts')
    os.makedirs(ckdir, exist_ok=True)
    for k in range(n_folds):
        tr_idx, va_idx = folds[k]
        ck = os.path.join(ckdir, f'{window}_fold{k}.pth')
        oof_path = ck.replace('.pth', '_oof.npy')
        ckpts.append(ck)
        if args.resume and os.path.exists(ck) and os.path.exists(oof_path):
            oof[va_idx] = np.load(oof_path)
            print(f'  -- fold {k}: resumed', flush=True)
            continue
        est = np.mean(args._fold_times) if args._fold_times else 0.0
        if args.deadline and time.time() + 1.2 * est > args.deadline:
            raise DeadlineReached(f'{window} fold {k}')
        t_fold = time.time()
        print(f'  -- fold {k}: train={len(tr_idx)} (pos {int(labels[tr_idx].sum())})  '
              f'val={len(va_idx)} (pos {int(labels[va_idx].sum())})', flush=True)
        model = build_model(args)
        tl = make_train_loader(pooled, tr_idx, labels[tr_idx], args)
        vl = DataLoader(Subset(pooled, va_idx), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
        train_one_fold(model, tl, vl, args, ck)
        o = forward_all(model, vl)
        oof[va_idx] = o['logit'].numpy()
        np.save(oof_path, oof[va_idx])
        args._fold_times.append(time.time() - t_fold)
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    valid = ~np.isnan(oof)
    T = fit_temperature(oof[valid], labels[valid]) if not args.no_calibrate else 1.0
    oof_p = 1 / (1 + np.exp(-oof[valid] / T))
    thr = pick_threshold_youden(labels[valid], oof_p)

    # ---- test ensemble ----
    tl = DataLoader(Subset(te, test_keep), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    fold_probs, mp, sp, dp = [], 0, 0, 0
    for ck in ckpts:
        model = build_model(args)
        model.load_state_dict(torch.load(ck, map_location=DEVICE))
        o = forward_all(model, tl)
        fold_probs.append(torch.sigmoid(o['logit'] / T).numpy())
        mp, sp, dp = mp + o['mp'].numpy(), sp + o['sp'].numpy(), dp + o['dp'].numpy()
        del model
    K = len(ckpts)
    prob = np.mean(fold_probs, axis=0)
    mp, sp, dp = mp / K * 100, sp / K * 100, dp / K * 100
    mt, st, dt = o['mt'].numpy() * 100, o['st'].numpy() * 100, o['dt'].numpy() * 100

    tdf = tm.iloc[test_keep].reset_index(drop=True)[['window', 'split', 'idx', 'filename', 'caseid',
                                                      'subjectid', 'dose', 'label']].copy()
    tdf['prob'] = prob
    for k, fp in enumerate(fold_probs):
        tdf[f'prob_fold{k}'] = fp
    for name, arr in [('map_pred', mp), ('map_true', mt), ('sbp_pred', sp), ('sbp_true', st),
                      ('dbp_pred', dp), ('dbp_true', dt)]:
        for j in range(arr.shape[1]):
            tdf[f'{name}_{j + 1}'] = arr[:, j]
    odf = pm[['window', 'split', 'idx', 'filename', 'caseid', 'subjectid', 'dose', 'label', 'fold']].copy()
    odf['logit'] = oof
    odf['prob_cal'] = 1 / (1 + np.exp(-oof / T))
    odf = odf[valid]

    y = tdf.label.values
    two = len(np.unique(y)) > 1
    summ = dict(run=args.run_name, window=window, temperature=T, threshold=thr,
                oof_auroc=roc_auc_score(labels[valid], oof_p) if len(np.unique(labels[valid])) > 1 else np.nan,
                oof_auprc=average_precision_score(labels[valid], oof_p) if len(np.unique(labels[valid])) > 1 else np.nan,
                test_n=len(y), test_pos=int(y.sum()),
                test_auroc=roc_auc_score(y, prob) if two else np.nan,
                test_auprc=average_precision_score(y, prob) if two else np.nan,
                test_brier=brier_score_loss(y, prob),
                test_mae_map=float(np.abs(mp - mt).mean()), test_mae_sbp=float(np.abs(sp - st).mean()),
                test_mae_dbp=float(np.abs(dp - dt).mean()), elapsed_min=(time.time() - t0) / 60)
    tdf.to_csv(os.path.join(run_dir, f'preds_test_{window}.csv'), index=False)
    odf.to_csv(os.path.join(run_dir, f'preds_oof_{window}.csv'), index=False)
    save_json(summ, os.path.join(run_dir, f'summary_{window}.json'))
    print(f'  TEST [{window}] AUROC={summ["test_auroc"]:.4f}  AUPRC={summ["test_auprc"]:.4f}  '
          f'Brier={summ["test_brier"]:.4f}  T={T:.3f}  thr={thr:.4f}  '
          f'MAE MAP/SBP/DBP={summ["test_mae_map"]:.2f}/{summ["test_mae_sbp"]:.2f}/{summ["test_mae_dbp"]:.2f}',
          flush=True)
    if not args.keep_ckpts:
        for ck in ckpts:
            if os.path.exists(ck):
                os.remove(ck)
    return summ


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--memmap_dir', required=True)
    ap.add_argument('--windows_csv', default=None, help='splits/windows.csv from make_splits.py')
    ap.add_argument('--cv_mode', choices=['grouped', 'window'], default='grouped',
                    help='grouped = patient-level folds from windows.csv; window = thesis protocol')
    ap.add_argument('--init', choices=['pretrained', 'scratch'], default='pretrained')
    ap.add_argument('--encoder_path', default=None)
    ap.add_argument('--pair_order', default=None, help='e.g. abp,ecg for a pair checkpoint')
    ap.add_argument('--allow_partial_load', action='store_true')
    ap.add_argument('--modalities', default='ppg,abp,ecg')
    ap.add_argument('--ecg_encoder', choices=['auto', 'wavelet', 'raw'], default='auto',
                    help='auto = whatever the checkpoint contains (wavelet for --init scratch)')
    ap.add_argument('--patch_len', type=int, default=0, help='0 = read from checkpoint (25 for scratch, to match your pretrained encoders)')
    ap.add_argument('--freeze_encoder', action='store_true')
    ap.add_argument('--label_frac', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--windows', nargs='+', default=WINDOWS)
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--max_folds', type=int, default=0, help='debug: train only the first k folds')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr_encoder', type=float, default=1e-5)
    ap.add_argument('--lr_heads', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--beta', type=float, default=2.0)
    ap.add_argument('--huber_delta', type=float, default=0.05)
    ap.add_argument('--focal_alpha', type=float, default=0.5)
    ap.add_argument('--focal_gamma', type=float, default=2.0)
    ap.add_argument('--sampler_pos_frac', type=float, default=0.30)
    ap.add_argument('--no_sampler', action='store_true')
    ap.add_argument('--no_augment', action='store_true')
    ap.add_argument('--aug_prob', type=float, default=0.5)
    ap.add_argument('--no_calibrate', action='store_true')
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--max_train_batches', type=int, default=0, help='debug: cap batches per epoch')
    ap.add_argument('--out_dir', default='/kaggle/working/runs')
    ap.add_argument('--run_name', required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--deadline', type=float, default=0,
                    help='unix time; stop cleanly before starting a fold that would end after it (Kaggle 12 h)')
    ap.add_argument('--mask_start', type=int, default=0, help='zero the first N samples of every channel (train and test)')
    ap.add_argument('--mask_end', type=int, default=0, help='zero the last N samples of every channel (train and test)')
    ap.add_argument('--keep_ckpts', action='store_true', help='keep fold checkpoints (needed for E10)')
    args = ap.parse_args()
    global MASK
    MASK = (args.mask_start, args.mask_end)

    args.mods = [m.strip() for m in args.modalities.split(',')]
    assert all(m in CANONICAL for m in args.mods), args.mods
    args.mods = [m for m in CANONICAL if m in args.mods]          # canonical order
    if args.init == 'pretrained':
        assert args.encoder_path and os.path.exists(args.encoder_path), \
            f'--encoder_path not found: {args.encoder_path}'
        info = describe_checkpoint(read_state_dict(args.encoder_path))
        print('[ckpt]', info)
        if info['format'] == 'trimodal':
            branches = {m: info[f'enc_{m}'] for m in CANONICAL}
        else:
            order = (args.pair_order.split(',') if args.pair_order else
                     re.search(r'ablation_(ppg|abp|ecg)_(ppg|abp|ecg)_', os.path.basename(args.encoder_path)).groups())
            branches = {order[0]: info['enc_a'], order[1]: info['enc_b']}
        pls = {b['patch_len'] for m, b in branches.items() if m in args.mods and b['patch_len']}
        if args.patch_len == 0:
            assert len(pls) == 1, f'cannot infer patch_len from checkpoint: {pls}'
            args.patch_len = pls.pop()
            print(f'[*] patch_len={args.patch_len} (from checkpoint)')
        if args.ecg_encoder == 'auto':
            t = branches.get('ecg', {}).get('type', 'wavelet')
            args.ecg_encoder = 'raw' if t == 'raw-patch' else 'wavelet'
            print(f'[*] ecg_encoder={args.ecg_encoder} (from checkpoint)')
    else:
        args.patch_len = args.patch_len or 25
        args.ecg_encoder = 'wavelet' if args.ecg_encoder == 'auto' else args.ecg_encoder
    if args.cv_mode == 'grouped':
        assert args.windows_csv, '--windows_csv is required for --cv_mode grouped'
    if args.label_frac < 1.0 and args.cv_mode != 'grouped':
        raise SystemExit('--label_frac needs --cv_mode grouped (patient-level subsets)')

    seed_everything(args.seed)
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_json({k: v for k, v in vars(args).items() if not k.startswith('_')},
              os.path.join(run_dir, 'config.json'))
    print(f'[*] device={DEVICE}  run_dir={run_dir}')

    wtab = None
    if args.cv_mode == 'grouped':
        wtab = pd.read_csv(args.windows_csv)
        if args.label_frac < 1.0:
            dev = wtab[(wtab.fold >= 0)]
            pts = dev.groupby('subjectid').label.max().rename('plabel').reset_index()
            args._subset = patient_subset(pts, args.label_frac, args.seed)
            print(f'[*] label_frac={args.label_frac}: {len(args._subset)}/{len(pts)} training patients '
                  f'(seed {args.seed})')

    args._fold_times = []
    summaries = []
    for w in args.windows:
        sp = os.path.join(run_dir, f'summary_{w}.json')
        if args.resume and os.path.exists(sp) and os.path.exists(os.path.join(run_dir, f'preds_test_{w}.csv')):
            print(f'[*] {w}: already done, skipping')
            summaries.append(pd.read_json(sp, typ='series').to_dict())
            continue
        try:
            summaries.append(run_window(w, args, wtab, run_dir))
        except DeadlineReached as e:
            print(f'\n[deadline] stopping before {e}. Finished folds are saved. Commit, attach this output, '
                  f'and run again with --resume.', flush=True)
            return
    pd.DataFrame(summaries).to_csv(os.path.join(run_dir, 'summary.csv'), index=False)
    print(f'\n[OK] {run_dir}/summary.csv')
    print(pd.DataFrame(summaries)[['window', 'test_auroc', 'test_auprc', 'test_brier', 'threshold']].to_string())


if __name__ == '__main__':
    main()
