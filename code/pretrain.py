"""
pretrain.py -- self-supervised pretraining with ablation switches (E6 / A3-A6, A7).

Model and loss are your pretraining-code.ipynb (HybridPretrainerV4 +
HybridLossV6, Cells 2-6) with the bugs in the uploaded Cell 6 fixed:
  * the ECG branch is the wavelet PrivateEncoderECG (the uploaded cell built
    PrivateEncoderRaw for ECG, then called enc_ecg.wavelet, which crashes);
  * HybridLossV6 no longer receives lam_var / lam_cov (not accepted -> crash);
  * lam_wavelet is actually passed through (the uploaded cell hard-coded 0.0).

New:
  --variant full | recon_only | no_contrastive | no_morph
  --ecg_encoder wavelet | raw           (raw = A6, raw-patch ECG branch)
  --compiled_dirs any list of memmaps   (A7: a VitalDB unlabelled memmap)
  --resume                              continues after a Kaggle time-out
  --retrieval_only <full.pth>           cross-modal retrieval on the WHOLE
                                        validation pool, joint and isolated
  Terms with weight 0 are not computed (the PPG fiducial term runs scipy per
  sample, so recon_only / no_morph are also faster).

IMPORTANT: compare A3-A6 against a 'full' model pretrained with THIS script
and the same seed/epochs, not against an older checkpoint made by different
code. Run --variant full once (or confirm with check_checkpoint.py that your
existing checkpoint has the same architecture and patch length).

Usage (Kaggle, GPU):
  !python pretrain.py --mimic_dir /kaggle/input/<...>/mimic-bp/dataverse_files \
       --variant no_contrastive --tag no_contrastive --save_dir /kaggle/working --resume
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

from common import build_encoder_bundle, robust_z, seed_everything

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    from sdtw_cuda_loss import SoftDTW
    _HAS_SOFTDTW = True
except ImportError:
    _HAS_SOFTDTW = False


# =============================================================================
# Data (Cells 4-5)
# =============================================================================
def compile_mimic_to_memmap(data_root_dir, output_dir, seq_len=3750, legacy_order=False):
    """Same as Cell 4, plus segment_subject.npy (patient index of every
    segment) so a subject-level train/val split is possible."""
    os.makedirs(output_dir, exist_ok=True)
    flag = os.path.join(output_dir, 'n_samples.npy')
    if os.path.exists(flag):
        print(f'[*] memmap exists ({int(np.load(flag)[0])} segments): {output_dir}')
        return
    dirs = {m: os.path.join(data_root_dir, m) for m in ['ecg', 'ppg', 'abp']}
    valid = []
    # legacy_order=True keeps the unsorted os.listdir order of the original notebook, so that
    # random_split(seed 42) gives the SAME validation segments as when the thesis model was trained
    names = os.listdir(dirs['ecg']) if legacy_order else sorted(os.listdir(dirs['ecg']))
    for f in names:
        if not f.endswith('.npy'):
            continue
        base = f.replace('_ecg.npy', '').replace('.npy', '')
        p, a = f'{base}_ppg.npy', f'{base}_abp.npy'
        if os.path.exists(os.path.join(dirs['ppg'], p)) and os.path.exists(os.path.join(dirs['abp'], a)):
            valid.append((f, p, a))
    counts = []
    for e, _, _ in valid:
        arr = np.load(os.path.join(dirs['ecg'], e), mmap_mode='r')
        counts.append(len(arr) // seq_len if arr.ndim == 1 else (arr.shape[0] if arr.shape[1] == seq_len else 0))
    N = int(sum(counts))
    print(f'[*] {len(valid)} patients, {N} segments')
    mm = {m: np.memmap(os.path.join(output_dir, f'{m}.dat'), dtype='float32', mode='w+', shape=(N, seq_len))
          for m in ['ecg', 'ppg', 'abp']}
    subj = np.zeros(N, dtype=np.int32)

    def load(path, n):
        arr = np.load(path).astype(np.float32)
        arr = arr[:n * seq_len].reshape(n, seq_len) if arr.ndim == 1 else arr[:n, :seq_len]
        for i in range(arr.shape[0]):
            seg = arr[i]
            m = np.nanmean(seg) if not np.all(np.isnan(seg)) else 0.0
            arr[i] = np.nan_to_num(seg, nan=m)
        return arr
    w = 0
    for pid, ((e, p, a), n) in enumerate(zip(valid, counts)):
        if n == 0:
            continue
        mm['ecg'][w:w + n] = load(os.path.join(dirs['ecg'], e), n)
        mm['ppg'][w:w + n] = load(os.path.join(dirs['ppg'], p), n)
        mm['abp'][w:w + n] = load(os.path.join(dirs['abp'], a), n)
        subj[w:w + n] = pid
        w += n
    for v in mm.values():
        v.flush()
    np.save(os.path.join(output_dir, 'segment_subject.npy'), subj)
    np.save(flag, np.array([N]))
    print(f'[OK] memmap -> {output_dir}')


class TripletPretrainDataset(Dataset):
    def __init__(self, compiled_dir, seq_len=3750):
        N = int(np.load(os.path.join(compiled_dir, 'n_samples.npy'))[0])
        self.N = N
        self.data = {ch: np.memmap(os.path.join(compiled_dir, f'{ch}.dat'), dtype='float32', mode='r',
                                   shape=(N, seq_len)) for ch in ['ppg', 'abp', 'ecg']}
        sp = os.path.join(compiled_dir, 'segment_subject.npy')
        self.subject = np.load(sp) if os.path.exists(sp) else None
        print(f'[*] {compiled_dir}: {N} triplets')

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return tuple(torch.from_numpy(robust_z(np.array(self.data[ch][idx]))) for ch in ['ppg', 'abp', 'ecg'])


# =============================================================================
# Pretrainer (Cell 2, unchanged)
# =============================================================================
def _mlp(d, out=None):
    return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, out or d))


class HybridPretrainerV4(nn.Module):
    def __init__(self, enc_ppg, enc_abp, enc_ecg, shared, d_model=128, patch_len=30, num_patches=125,
                 M=3, tau=0.2, mask_ratio=0.25, block_patches=6):
        super().__init__()
        self.enc_ppg, self.enc_abp, self.enc_ecg, self.shared = enc_ppg, enc_abp, enc_ecg, shared
        self.d_model, self.patch_len, self.num_patches = d_model, patch_len, num_patches
        self.M, self.tau, self.mask_ratio, self.block_patches = M, tau, mask_ratio, block_patches
        self.intra_pool_ppg, self.intra_pool_abp, self.intra_pool_ecg = _mlp(d_model), _mlp(d_model), _mlp(d_model)
        self.proj_ppg, self.proj_abp, self.proj_ecg = _mlp(d_model), _mlp(d_model), _mlp(d_model)
        self.dec_ppg, self.dec_abp, self.dec_ecg = (_mlp(d_model, patch_len) for _ in range(3))
        self.dec_clean_ppg, self.dec_clean_abp, self.dec_clean_ecg = (_mlp(d_model, patch_len) for _ in range(3))

    def _block_mask(self, x):
        B, L = x.shape
        block_len = self.patch_len * self.block_patches
        n_blocks = max(1, int((self.mask_ratio * L) / block_len))
        masked = x.clone()
        for b in range(B):
            for s in torch.randint(0, L - block_len + 1, (n_blocks,), device=x.device):
                masked[b, s:s + block_len] = 0.0
        return masked

    def _intra_rep(self, z, m):
        return getattr(self, f'intra_pool_{m}')(z.mean(dim=1))

    def _cross_rep(self, z, m):
        return getattr(self, f'proj_{m}')(z.mean(dim=1))

    def _encode_full(self, x_ppg, x_abp, x_ecg):
        z_p, z_a, z_e = self.enc_ppg(x_ppg), self.enc_abp(x_abp), self.enc_ecg(x_ecg)
        return (z_p, z_a, z_e), self.shared(z_p, z_a, z_e)

    def _simmtm_aggregate(self, z_orig_s, z_masked_s_list, s_orig, s_masked_list, decoder, B):
        z_all = torch.cat([z_orig_s] + z_masked_s_list, dim=0)
        s_all = torch.cat([s_orig] + s_masked_list, dim=0)
        s_norm = F.normalize(s_all, p=2, dim=-1)
        R = torch.mm(s_norm, s_norm.T)
        D_total = (self.M + 1) * B
        out = []
        for i in range(B):
            idx = torch.tensor([j for j in range(D_total) if j != i], device=z_orig_s.device)
            w_i = F.softmax(R[i, idx] / self.tau, dim=0)
            out.append((w_i.view(-1, 1, 1) * z_all[idx]).sum(0))
        return decoder(torch.stack(out, 0)).reshape(B, -1), s_all

    def forward(self, x_ppg, x_abp, x_ecg):
        B = x_ppg.shape[0]
        (zp, za, ze), (zps, zas, zes) = self._encode_full(x_ppg, x_abp, x_ecg)
        s_o = {'ppg': self._intra_rep(zp, 'ppg'), 'abp': self._intra_rep(za, 'abp'), 'ecg': self._intra_rep(ze, 'ecg')}
        c = {'ppg': self._cross_rep(zps, 'ppg'), 'abp': self._cross_rep(zas, 'abp'), 'ecg': self._cross_rep(zes, 'ecg')}
        x_clean = {'ppg': self.dec_clean_ppg(zps).reshape(B, -1), 'abp': self.dec_clean_abp(zas).reshape(B, -1),
                   'ecg': self.dec_clean_ecg(zes).reshape(B, -1)}
        zsm = {'ppg': [], 'abp': [], 'ecg': []}
        sm = {'ppg': [], 'abp': [], 'ecg': []}
        for _ in range(self.M):
            (zpm, zam, zem), (zpsm, zasm, zesm) = self._encode_full(
                self._block_mask(x_ppg), self._block_mask(x_abp), self._block_mask(x_ecg))
            for m, zs_, zpv in [('ppg', zpsm, zpm), ('abp', zasm, zam), ('ecg', zesm, zem)]:
                zsm[m].append(zs_)
                sm[m].append(self._intra_rep(zpv, m))
        out = dict(B=B)
        for m, zs_ in [('ppg', zps), ('abp', zas), ('ecg', zes)]:
            x_hat, s_all = self._simmtm_aggregate(zs_, zsm[m], s_o[m], sm[m], getattr(self, f'dec_{m}'), B)
            out[f'x_hat_{m}'], out[f's_all_{m}'] = x_hat, s_all
            out[f'x_clean_{m}'], out[f'c_{m}'] = x_clean[m], c[m]
        return out


# =============================================================================
# Loss (Cell 3, unchanged maths; zero-weight terms skipped)
# =============================================================================
def _detect_ppg_fiducials(x, fs=125, max_hr_bpm=180):
    from scipy.signal import find_peaks
    x_np = x.detach().cpu().numpy()
    sys_idx, _ = find_peaks(x_np, distance=int(60 * fs / max_hr_bpm))
    empty = torch.empty(0, dtype=torch.long, device=x.device)
    if len(sys_idx) < 2:
        return empty, empty, empty
    d2 = np.diff(x_np, n=2, prepend=x_np[0], append=x_np[-1])
    notch, dia = [], []
    for i in range(len(sys_idx) - 1):
        s0, s1 = sys_idx[i], sys_idx[i + 1]
        win = -d2[s0:s0 + (s1 - s0) // 2]
        if len(win) > 2:
            nl, _ = find_peaks(win)
            if len(nl):
                npos = s0 + int(nl[0])
                notch.append(npos)
                dw = x_np[npos:s1]
                if len(dw) > 2:
                    dl, _ = find_peaks(dw)
                    if len(dl):
                        dia.append(npos + int(dl[0]))
    t = lambda a: torch.tensor(a, dtype=torch.long, device=x.device)
    return t(sys_idx), t(notch), t(dia)


class HybridLossV6(nn.Module):
    def __init__(self, tau=0.2, M=3, lam_mse=0.8, lam_pcc=0.2, lam_mrstft=0.5, lam_peak=0.3, lam_clean=1.0,
                 lam_d1=0.3, lam_d2=0.5, lam_fid=0.05, lam_sdtw=0.01, fid_w_sys=1.0, fid_w_notch=2.0,
                 fid_w_dia=1.0, lam_intra=1.0, lam_cross=0.1, lam_wavelet=0.01, peak_quantile=0.85, fs=125,
                 stft_ffts=(128, 256, 512, 1024)):
        super().__init__()
        for k, v in list(locals().items()):
            if k not in ('self', '__class__'):
                setattr(self, k, v)
        self.softdtw = (SoftDTW(use_cuda=torch.cuda.is_available(), gamma=0.1, normalize=False)
                        if _HAS_SOFTDTW and lam_sdtw > 0 else None)

    def _mr_stft_loss(self, x, x_hat):
        loss = 0.0
        for n_fft in self.stft_ffts:
            win = torch.hann_window(n_fft, device=x.device)
            X = torch.stft(x, n_fft=n_fft, hop_length=n_fft // 4, window=win, return_complex=True).abs()
            Xh = torch.stft(x_hat, n_fft=n_fft, hop_length=n_fft // 4, window=win, return_complex=True).abs()
            loss = loss + F.l1_loss(torch.log1p(Xh), torch.log1p(X)) + (X - Xh).norm(p='fro') / (X.norm(p='fro') + 1e-8)
        return loss / len(self.stft_ffts)

    @staticmethod
    def _pcc_loss(x, x_hat):
        xm, xh = x - x.mean(-1, keepdim=True), x_hat - x_hat.mean(-1, keepdim=True)
        pcc = (xm * xh).sum(-1) / (torch.sqrt((xm ** 2).sum(-1) * (xh ** 2).sum(-1)) + 1e-8)
        return (1.0 - pcc).mean()

    @staticmethod
    def _peak_loss(x, x_hat, q=0.85):
        thr = torch.quantile(x.abs(), q, dim=-1, keepdim=True)
        mask = (x.abs() >= thr).float()
        pm = (((x - x_hat) ** 2) * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        return (pm / x.abs().mean(-1).clamp(min=1e-6) ** 2).mean()

    @staticmethod
    def _d1_loss(x, xh):
        return F.mse_loss(xh[..., 1:] - xh[..., :-1], x[..., 1:] - x[..., :-1])

    @staticmethod
    def _d2_loss(x, xh):
        return F.mse_loss(xh[..., 2:] - 2 * xh[..., 1:-1] + xh[..., :-2], x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2])

    def _fiducial_loss(self, x, xh):
        terms = []
        for b in range(x.shape[0]):
            parts = []
            for idx, w in zip(_detect_ppg_fiducials(x[b], fs=self.fs), (self.fid_w_sys, self.fid_w_notch, self.fid_w_dia)):
                if len(idx):
                    parts.append(w * (xh[b, idx] - x[b, idx]).abs().mean())
            if parts:
                terms.append(sum(parts) / len(parts))
        return torch.stack(terms).mean() if terms else torch.tensor(0.0, device=x.device)

    def _intra_contrastive(self, s_all, B):
        s = F.normalize(s_all, p=2, dim=-1)
        R = torch.mm(s, s.T) / self.tau
        loss = 0.0
        for i in range(B):
            neg = torch.ones(R.shape[0], dtype=torch.bool, device=R.device)
            neg[i] = False
            lse = torch.logsumexp(R[i][neg], dim=0)
            for m in range(1, self.M + 1):
                loss += -(R[i, i + m * B] - lse)
        return loss / (B * self.M)

    def _info_nce(self, a, b):
        sim = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T / self.tau
        lab = torch.arange(sim.shape[0], device=sim.device)
        return 0.5 * (F.cross_entropy(sim, lab) + F.cross_entropy(sim.T, lab))

    def forward(self, x, out, ecg_wavelet=None):
        z = torch.tensor(0.0, device=x['ppg'].device)
        B = out['B']
        tot, parts = 0.0, {}
        for m in ['ppg', 'abp', 'ecg']:
            xm, xh, xc = x[m], out[f'x_hat_{m}'], out[f'x_clean_{m}']
            l = dict(mse=F.mse_loss(xh, xm), pcc=self._pcc_loss(xm, xh), clean=F.mse_loss(xc, xm),
                     mrstft=self._mr_stft_loss(xm, xh) if self.lam_mrstft else z,
                     peak=self._peak_loss(xm, xh, self.peak_quantile) if self.lam_peak else z)
            for k, v in l.items():
                parts[k] = parts.get(k, 0.0) + v
            parts[f'{m}_mse'] = l['mse'].item()
        tot = (self.lam_mse * parts['mse'] + self.lam_pcc * parts['pcc'] + self.lam_clean * parts['clean']
               + self.lam_mrstft * parts['mrstft'] + self.lam_peak * parts['peak'])
        if self.lam_d1 or self.lam_d2:
            d1 = self._d1_loss(x['ppg'], out['x_clean_ppg']) + self._d1_loss(x['abp'], out['x_clean_abp'])
            d2 = self._d2_loss(x['ppg'], out['x_clean_ppg']) + self._d2_loss(x['abp'], out['x_clean_abp'])
            tot = tot + self.lam_d1 * d1 + self.lam_d2 * d2
            parts['d1'], parts['d2'] = d1, d2
        if self.lam_fid:
            parts['fid'] = self._fiducial_loss(x['ppg'], out['x_clean_ppg'])
            tot = tot + self.lam_fid * parts['fid']
        if self.softdtw is not None and self.lam_sdtw:
            parts['sdtw'] = self.softdtw(x['ppg'].unsqueeze(-1), out['x_clean_ppg'].unsqueeze(-1)).mean() / x['ppg'].shape[-1]
            tot = tot + self.lam_sdtw * parts['sdtw']
        if self.lam_intra:
            parts['intra'] = sum(self._intra_contrastive(out[f's_all_{m}'], B) for m in ['ppg', 'abp', 'ecg'])
            tot = tot + self.lam_intra * parts['intra']
        if self.lam_cross:
            parts['cross'] = (self._info_nce(out['c_ppg'], out['c_abp']) + self._info_nce(out['c_ppg'], out['c_ecg'])
                              + self._info_nce(out['c_abp'], out['c_ecg'])) / 3.0
            tot = tot + self.lam_cross * parts['cross']
        if ecg_wavelet is not None and self.lam_wavelet:
            parts['wavelet'] = ecg_wavelet.qmf_penalty()
            tot = tot + self.lam_wavelet * parts['wavelet']
        return tot, {k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in parts.items()}


VARIANTS = {
    'full': {},
    'recon_only': dict(lam_mrstft=0.0, lam_peak=0.0, lam_d1=0.0, lam_d2=0.0, lam_fid=0.0, lam_sdtw=0.0,
                       lam_intra=0.0, lam_cross=0.0),
    'no_contrastive': dict(lam_intra=0.0, lam_cross=0.0),
    'no_morph': dict(lam_peak=0.0, lam_d1=0.0, lam_d2=0.0, lam_fid=0.0, lam_sdtw=0.0),
}


# =============================================================================
# Cross-modal retrieval on the full validation pool
# =============================================================================
@torch.no_grad()
def retrieval_eval(model, loader):
    """Embeds every validation window and ranks each query against ALL
    windows of the other modality.
    joint    : shared encoder sees all three modalities of the window (as in
               training) -- the PPG embedding already contains ABP/ECG
               information of the same window, so this is optimistic.
    isolated : each modality passes through the shared encoder alone, so a
               match must come from the signal itself. Report this one."""
    model.eval()
    emb = {mode: {m: [] for m in ['ppg', 'abp', 'ecg']} for mode in ['joint', 'isolated']}
    for x_ppg, x_abp, x_ecg in loader:
        xs = {'ppg': x_ppg.to(DEVICE), 'abp': x_abp.to(DEVICE), 'ecg': x_ecg.to(DEVICE)}
        _, (zps, zas, zes) = model._encode_full(xs['ppg'], xs['abp'], xs['ecg'])
        for m, zs in [('ppg', zps), ('abp', zas), ('ecg', zes)]:
            emb['joint'][m].append(F.normalize(model._cross_rep(zs, m), dim=-1).cpu())
            zp = getattr(model, f'enc_{m}')(xs[m])
            zi = model.shared.forward_subset([zp], [m])[0]
            emb['isolated'][m].append(F.normalize(model._cross_rep(zi, m), dim=-1).cpu())
    res = {}
    for mode in emb:
        E = {m: torch.cat(v) for m, v in emb[mode].items()}
        N = E['ppg'].shape[0]
        for a in ['ppg', 'abp', 'ecg']:
            for b in ['ppg', 'abp', 'ecg']:
                if a == b:
                    continue
                ranks = []
                for s in range(0, N, 1024):
                    sim = E[a][s:s + 1024] @ E[b].T
                    true = sim[torch.arange(sim.shape[0]), torch.arange(s, s + sim.shape[0])]
                    ranks.append((sim > true[:, None]).sum(1) + 1)
                r = torch.cat(ranks).float()
                res[f'{mode}_{a}->{b}'] = dict(R1=float((r <= 1).float().mean()), R5=float((r <= 5).float().mean()),
                                              MRR=float((1 / r).mean()), pool=N)
    for k, v in res.items():
        print(f'  {k:18s} R@1={v["R1"]:.3f}  R@5={v["R5"]:.3f}  MRR={v["MRR"]:.3f}  (pool {v["pool"]})')
    return res


# =============================================================================
# Main
# =============================================================================
def make_split(full_ds, mode, seed=42):
    n_val = int(0.1 * len(full_ds))
    if mode == 'segment':          # as in the notebook
        return random_split(full_ds, [len(full_ds) - n_val, n_val], generator=torch.Generator().manual_seed(seed))
    subs = full_ds.datasets if isinstance(full_ds, ConcatDataset) else [full_ds]
    subj, off = [], 0
    for i, d in enumerate(subs):
        if d.subject is None:
            raise SystemExit('subject split needs segment_subject.npy: recompile the memmap with this script')
        subj.append(d.subject.astype(np.int64) + i * 10_000_000)
    subj = np.concatenate(subj)
    rng = np.random.RandomState(seed)
    u = np.unique(subj)
    rng.shuffle(u)
    val_s = set(u[:int(round(0.1 * len(u)))])
    is_val = np.array([s in val_s for s in subj])
    return Subset(full_ds, np.where(~is_val)[0]), Subset(full_ds, np.where(is_val)[0])


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--mimic_dir', default=None, help='MIMIC-BP dataverse_files (compiled once)')
    ap.add_argument('--compiled_dirs', nargs='*', default=None, help='already-compiled memmap dirs')
    ap.add_argument('--compile_to', default='/kaggle/working/mimic_memmap')
    ap.add_argument('--variant', choices=list(VARIANTS), default='full')
    ap.add_argument('--ecg_encoder', choices=['wavelet', 'raw'], default='wavelet')
    ap.add_argument('--patch_len', type=int, default=25, help='25 = same as your hybrid_v4 and pair encoders')
    ap.add_argument('--split', choices=['segment', 'subject'], default='segment')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--mask_ratio', type=float, default=0.25)
    ap.add_argument('--block_patches', type=int, default=6)
    ap.add_argument('--lam_wavelet', type=float, default=0.01)
    ap.add_argument('--lam_sdtw', type=float, default=0.01, help='only active if sdtw_cuda_loss is installed')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--max_batches', type=int, default=0, help='debug: cap batches per epoch')
    ap.add_argument('--save_dir', default='/kaggle/working')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--legacy_order', action='store_true',
                    help='compile MIMIC in the original notebook file order (for --retrieval_only on old models)')
    ap.add_argument('--time_limit_h', type=float, default=0,
                    help='stop cleanly after this many hours (e.g. 11 on Kaggle); continue with --resume')
    ap.add_argument('--retrieval_only', default=None, help='path to a *_full.pth: only run retrieval eval')
    args = ap.parse_args()
    tag = args.tag or f'{args.variant}_{args.ecg_encoder}'
    t_start = time.time()
    seed_everything(args.seed)

    if args.compiled_dirs:
        dirs = args.compiled_dirs
    else:
        assert args.mimic_dir, 'give --mimic_dir or --compiled_dirs'
        compile_mimic_to_memmap(args.mimic_dir, args.compile_to, legacy_order=args.legacy_order)
        dirs = [args.compile_to]
    subs = [TripletPretrainDataset(d) for d in dirs]
    full_ds = ConcatDataset(subs) if len(subs) > 1 else subs[0]
    train_ds, val_ds = make_split(full_ds, args.split, args.seed)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    print(f'[*] train {len(train_ds)} | val {len(val_ds)} | split={args.split}')

    bundle = build_encoder_bundle(patch_len=args.patch_len, ecg_encoder=args.ecg_encoder).to(DEVICE)
    model = HybridPretrainerV4(bundle.enc_ppg, bundle.enc_abp, bundle.enc_ecg, bundle.shared,
                               patch_len=args.patch_len, num_patches=3750 // args.patch_len,
                               mask_ratio=args.mask_ratio, block_patches=args.block_patches).to(DEVICE)
    ecg_wavelet = bundle.enc_ecg.wavelet if args.ecg_encoder == 'wavelet' else None

    if args.retrieval_only:
        os.makedirs(args.save_dir, exist_ok=True)
        model.load_state_dict(torch.load(args.retrieval_only, map_location=DEVICE))
        res = retrieval_eval(model, val_loader)
        json.dump(res, open(os.path.join(args.save_dir, f'retrieval_{os.path.basename(args.retrieval_only)}.json'), 'w'),
                  indent=2)
        return

    lam = dict(lam_wavelet=args.lam_wavelet, lam_sdtw=args.lam_sdtw)
    lam.update(VARIANTS[args.variant])
    crit = HybridLossV6(**lam)
    active = {k: v for k, v in vars(crit).items() if k.startswith('lam_')}
    print(f'[*] variant={args.variant}  ecg={args.ecg_encoder}  patch_len={args.patch_len}')
    print(f'[*] loss weights: {active}')
    print(f'[*] soft-DTW term active: {crit.softdtw is not None}  (paper must say whether it was used)')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    E = args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / 5 if ep < 5 else 0.5 * (1 + math.cos(math.pi * (ep - 5) / max(1, E - 5))))

    os.makedirs(args.save_dir, exist_ok=True)
    p_enc = os.path.join(args.save_dir, f'pretrain_{tag}_encoders.pth')
    p_full = os.path.join(args.save_dir, f'pretrain_{tag}_full.pth')
    p_state = os.path.join(args.save_dir, f'pretrain_{tag}_state.pth')
    p_hist = os.path.join(args.save_dir, f'pretrain_{tag}_history.json')
    start, best, no_imp, hist, done = 0, float('inf'), 0, [], False
    if args.resume and os.path.exists(p_state):
        st = torch.load(p_state, map_location=DEVICE, weights_only=False)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        start, best, no_imp, hist = st['epoch'] + 1, st['best'], st['no_imp'], st['hist']
        done = st.get('done', False)
        print(f'[*] resumed at epoch {start} (best val {best:.4f}){"  -- already finished" if done else ""}')

    stopped_for_time = False
    for ep in range(0 if done else start, 0 if done else E):
        t0 = time.time()
        model.train()
        tr, n = 0.0, 0
        for b, (xp, xa, xe) in enumerate(train_loader):
            if args.max_batches and b >= args.max_batches:
                break
            x = {'ppg': xp.to(DEVICE, non_blocking=True), 'abp': xa.to(DEVICE, non_blocking=True),
                 'ecg': xe.to(DEVICE, non_blocking=True)}
            opt.zero_grad()
            loss, _ = crit(x, model(x['ppg'], x['abp'], x['ecg']), ecg_wavelet)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr, n = tr + loss.item(), n + 1
        sched.step()
        model.eval()
        vl, vn, vparts = 0.0, 0, {}
        with torch.no_grad():
            for b, (xp, xa, xe) in enumerate(val_loader):
                if args.max_batches and b >= args.max_batches:
                    break
                x = {'ppg': xp.to(DEVICE), 'abp': xa.to(DEVICE), 'ecg': xe.to(DEVICE)}
                loss, parts = crit(x, model(x['ppg'], x['abp'], x['ecg']), ecg_wavelet)
                vl, vn = vl + loss.item(), vn + 1
                for k, v in parts.items():
                    vparts[k] = vparts.get(k, 0.0) + v
        vl /= max(vn, 1)
        vparts = {k: v / max(vn, 1) for k, v in vparts.items()}
        hist.append(dict(epoch=ep + 1, train=tr / max(n, 1), val=vl, **vparts))
        print(f'  ep {ep + 1:03d}/{E}  train={tr / max(n, 1):.4f}  val={vl:.4f}  '
              f'mse ppg/abp/ecg={vparts.get("ppg_mse", 0):.4f}/{vparts.get("abp_mse", 0):.4f}/'
              f'{vparts.get("ecg_mse", 0):.4f}  t={(time.time() - t0) / 60:.1f}m', flush=True)
        if vl < best:
            best, no_imp = vl, 0
            torch.save(bundle.state_dict(), p_enc)
            torch.save(model.state_dict(), p_full)
            print(f'    [saved] {p_enc}')
        else:
            no_imp += 1
        finished = no_imp >= args.patience or ep + 1 == E
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep,
                        best=best, no_imp=no_imp, hist=hist, done=finished), p_state)
        json.dump(hist, open(p_hist, 'w'), indent=1)
        if no_imp >= args.patience:
            print(f'[*] early stop at epoch {ep + 1}')
            break
        hours = (time.time() - t_start) / 3600
        per_epoch = hours / max(1, ep + 1 - start)
        if args.time_limit_h and hours + per_epoch > args.time_limit_h and not finished:
            print(f'[*] time limit: stopping after epoch {ep + 1} ({hours:.1f} h). '
                  f'Save this version, attach its output, and re-run with --resume.')
            stopped_for_time = True
            break

    if stopped_for_time:
        return

    print(f'[*] best val {best:.4f}; encoders -> {p_enc}')
    model.load_state_dict(torch.load(p_full, map_location=DEVICE))
    res = retrieval_eval(model, val_loader)
    json.dump(res, open(os.path.join(args.save_dir, f'pretrain_{tag}_retrieval.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
