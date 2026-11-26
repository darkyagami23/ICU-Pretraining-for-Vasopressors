"""
baselines_tabular.py -- naive baselines on the SAME windows, folds and test set
(CPU only, a few minutes). Covers E8 (part) and E12.

The waveforms in the memmaps are z-scored per window (no absolute pressure),
so the baselines use the monitor's numeric blood-pressure values stored in
the per-window *_abp_based.csv files of Master_Training_Data_Expanded_New_2
(art_sbp_0..14, art_dbp_0..14, art_mbp_0..14: 15 samples over the 30-s
input window, in mmHg).

  baseline_lastmap     logistic regression on the most recent MAP
                       (art_mbp at the end of the input window)
  baseline_bpfeat_gbm  gradient-boosted trees on MAP mean/min/max/SD/slope,
                       last MAP/SBP/DBP, pulse pressure, HR and SpO2 of the
                       30-s input window
  baseline_persistence BP forecast = last observed MAP/SBP/DBP carried forward
                       for all 15 steps (no classifier)

Each baseline uses the same 5 patient-level folds as finetune.py: 5 fold
models, Youden threshold on out-of-fold probabilities, test probability =
mean of the 5 fold models. Outputs use the finetune.py format, so evaluate.py
treats them like any other run.

Also writes drug_counts.csv: the vasopressor_name of every positive window
(for the outcome definition in the paper).

NOTE (put this in the paper): the original plan said "summary statistics of
the preceding 10 min of MAP". These features use the 30-s input window only;
a 10-min-context version needs the VitalDB numeric tracks.

Usage:
  !python baselines_tabular.py --memmap_dir <physio_memmap> \
      --csv_dir <...>/Master_Training_Data_Expanded_New_2 \
      --windows_csv /kaggle/working/splits/windows.csv --out_dir /kaggle/working/runs
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import dev_test_tables, save_json

WINDOWS = ['offset_3m', 'offset_5m', 'offset_7m', 'offset_10m', 'offset_12m', 'offset_15m']
FEATS = ['last_map', 'map_mean', 'map_min', 'map_max', 'map_sd', 'map_slope', 'last_sbp', 'last_dbp',
         'pulse_pressure', 'hr', 'spo2']


def _series(row, prefix):
    cols = sorted([c for c in row.index if c.startswith(prefix + '_') and c.rsplit('_', 1)[-1].isdigit()],
                  key=lambda c: int(c.rsplit('_', 1)[-1]))
    return pd.to_numeric(row[cols], errors='coerce').values.astype(float)


def _last(x):
    x = x[np.isfinite(x)]
    return float(np.mean(x[-3:])) if len(x) else np.nan        # mean of the last 3 samples (~last 6 s)


def bp_features(row):
    """row: first row of an *_abp_based.csv (pandas Series)."""
    mbp, sbp, dbp = _series(row, 'art_mbp'), _series(row, 'art_sbp'), _series(row, 'art_dbp')
    ok = np.isfinite(mbp)
    t_min = np.arange(len(mbp)) * (30.0 / max(len(mbp), 1)) / 60.0
    slope = np.polyfit(t_min[ok], mbp[ok], 1)[0] if ok.sum() > 2 else np.nan
    f = dict(last_map=_last(mbp), last_sbp=_last(sbp), last_dbp=_last(dbp),
             map_mean=np.nanmean(mbp) if ok.any() else np.nan, map_min=np.nanmin(mbp) if ok.any() else np.nan,
             map_max=np.nanmax(mbp) if ok.any() else np.nan, map_sd=np.nanstd(mbp) if ok.any() else np.nan,
             map_slope=slope)
    f['pulse_pressure'] = f['last_sbp'] - f['last_dbp']
    for name, pre in [('hr', 'pleth_hr'), ('spo2', 'pleth_spo2')]:
        v = pd.to_numeric(row[[c for c in row.index if c.startswith(pre)]], errors='coerce').values.astype(float)
        f[name] = np.nanmean(v) if np.isfinite(v).any() else np.nan
    f['vasopressor_name'] = row.get('vasopressor_name', np.nan)
    return f


def load_split_features(memmap_dir, csv_dir, split, window):
    d = os.path.join(memmap_dir, split, window)
    N = int(np.load(os.path.join(d, 'n_samples.npy'))[0])
    fn = np.load(os.path.join(d, 'filenames.npy'), allow_pickle=True).astype(str)
    sc = np.memmap(os.path.join(d, 'scalars.dat'), dtype='float32', mode='r', shape=(N, 3))
    rows, missing = [], 0
    for i in range(N):
        path = os.path.join(csv_dir, split, 'abp', window, fn[i].replace('_wave.csv', '_based.csv'))
        try:
            f = bp_features(pd.read_csv(path).iloc[0])
        except Exception:  # noqa: BLE001
            missing += 1
            f = {k: np.nan for k in FEATS}
        if not np.isfinite(f.get('hr', np.nan)):
            f['hr'] = float(sc[i, 1]) * 100
        if not np.isfinite(f.get('spo2', np.nan)):
            f['spo2'] = float(sc[i, 0]) * 100
        rows.append(f)
    if missing:
        print(f'  [!] {split}/{window}: {missing}/{N} _based.csv files missing or unreadable')
    feats = pd.DataFrame(rows)
    tg = {k: np.memmap(os.path.join(d, f'{k}_targets.dat'), dtype='float32', mode='r', shape=(N, 15)) * 100
          for k in ['map', 'sbp', 'dbp']}
    return feats, tg


def youden(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    t = float(thr[int(np.argmax(tpr - fpr))])
    return t if np.isfinite(t) else float(np.max(p) + 1e-6)


def make_model(name, seed):
    if name == 'baseline_lastmap':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), ['last_map']
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          l2_regularization=1.0, random_state=seed), FEATS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memmap_dir', required=True)
    ap.add_argument('--csv_dir', required=True,
                    help='Master_Training_Data_Expanded_New_2 (has <split>/abp/<window>/*_abp_based.csv)')
    ap.add_argument('--windows_csv', default=None)
    ap.add_argument('--cv_mode', choices=['grouped', 'window'], default='grouped')
    ap.add_argument('--windows', nargs='+', default=WINDOWS)
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default='/kaggle/working/runs')
    args = ap.parse_args()
    wtab = pd.read_csv(args.windows_csv) if args.windows_csv else None
    if args.cv_mode == 'grouped' and wtab is None:
        raise SystemExit('--windows_csv required for grouped CV')

    for w in args.windows:
        print(f'\n=== {w} ===')
        pm, tm, folds, keep = dev_test_tables(args.memmap_dir, w, wtab, args.cv_mode, args.n_folds, args.seed)
        ftr, tgtr = load_split_features(args.memmap_dir, args.csv_dir, 'train', w)
        fva, tgva = load_split_features(args.memmap_dir, args.csv_dir, 'val', w)
        fte, tgte = load_split_features(args.memmap_dir, args.csv_dir, 'test', w)
        Xdev = pd.concat([ftr, fva], ignore_index=True)
        Xte = fte.iloc[keep].reset_index(drop=True)
        tte = {k: v[keep] for k, v in tgte.items()}
        y = pm.label.values
        yte = tm.label.values[keep]

        allx = pd.concat([Xdev.assign(label=pm.label.values, split=pm.split.values),
                          fte.assign(label=tm.label.values, split='test')], ignore_index=True)
        drugs = allx[allx.label == 1].groupby(['split', 'vasopressor_name']).size().rename('n_windows')
        os.makedirs(args.out_dir, exist_ok=True)
        drugs.to_csv(os.path.join(args.out_dir, f'drug_counts_{w}.csv'))
        print('  positives by vasopressor_name:', drugs.groupby(level=1).sum().to_dict())
        med = np.nanmedian(Xdev.last_map)
        print(f'  median last MAP across dev windows: {med:.1f} mmHg')
        if not 30 <= med <= 200:
            print('  [!] MAP does not look like mmHg -- check --csv_dir. Baselines NOT valid.')

        base_cols = ['window', 'split', 'idx', 'filename', 'caseid', 'subjectid', 'dose', 'label']
        for name in ['baseline_lastmap', 'baseline_bpfeat_gbm']:
            oof = np.full(len(pm), np.nan)
            fold_probs = []
            for k, (tr_idx, va_idx) in enumerate(folds):
                model, cols = make_model(name, args.seed)
                Xtr = Xdev.iloc[tr_idx][cols].fillna(Xdev[cols].median())
                model.fit(Xtr, y[tr_idx])
                oof[va_idx] = model.predict_proba(Xdev.iloc[va_idx][cols].fillna(Xdev[cols].median()))[:, 1]
                fold_probs.append(model.predict_proba(Xte[cols].fillna(Xdev[cols].median()))[:, 1])
            valid = ~np.isnan(oof)
            thr = youden(y[valid], oof[valid])
            prob = np.mean(fold_probs, axis=0)
            run_dir = os.path.join(args.out_dir, name)
            os.makedirs(run_dir, exist_ok=True)
            tdf = tm.iloc[keep].reset_index(drop=True)[base_cols].copy()
            tdf['prob'] = prob
            for k, fp in enumerate(fold_probs):
                tdf[f'prob_fold{k}'] = fp
            tdf.to_csv(os.path.join(run_dir, f'preds_test_{w}.csv'), index=False)
            odf = pm[base_cols + ['fold']].copy()
            odf['prob_cal'] = oof
            odf[valid].to_csv(os.path.join(run_dir, f'preds_oof_{w}.csv'), index=False)
            two = len(np.unique(yte)) > 1
            summ = dict(run=name, window=w, threshold=thr,
                        test_auroc=roc_auc_score(yte, prob) if two else np.nan,
                        test_auprc=average_precision_score(yte, prob) if two else np.nan,
                        test_brier=brier_score_loss(yte, prob), test_n=len(yte), test_pos=int(yte.sum()))
            save_json(summ, os.path.join(run_dir, f'summary_{w}.json'))
            save_json(dict(run_name=name, kind='baseline', features=make_model(name, 0)[1]),
                      os.path.join(run_dir, 'config.json'))
            print(f'  {name:22s} AUROC={summ["test_auroc"]:.4f}  AUPRC={summ["test_auprc"]:.4f}')

        # persistence forecast
        run_dir = os.path.join(args.out_dir, 'baseline_persistence')
        os.makedirs(run_dir, exist_ok=True)
        tdf = tm.iloc[keep].reset_index(drop=True)[base_cols].copy()
        tdf['prob'] = np.nan
        last = {'map': Xte.last_map.values, 'sbp': Xte.last_sbp.values, 'dbp': Xte.last_dbp.values}
        summ = dict(run='baseline_persistence', window=w)
        for k in ['map', 'sbp', 'dbp']:
            pred = np.repeat(last[k][:, None], 15, axis=1)
            for j in range(15):
                tdf[f'{k}_pred_{j + 1}'] = pred[:, j]
                tdf[f'{k}_true_{j + 1}'] = tte[k][:, j]
            summ[f'test_mae_{k}'] = float(np.abs(pred - tte[k]).mean())
        tdf.to_csv(os.path.join(run_dir, f'preds_test_{w}.csv'), index=False)
        save_json(summ, os.path.join(run_dir, f'summary_{w}.json'))
        save_json(dict(run_name='baseline_persistence', kind='baseline'), os.path.join(run_dir, 'config.json'))
        print(f'  persistence MAE MAP/SBP/DBP = {summ["test_mae_map"]:.2f}/{summ["test_mae_sbp"]:.2f}/'
              f'{summ["test_mae_dbp"]:.2f} mmHg')


if __name__ == '__main__':
    main()
