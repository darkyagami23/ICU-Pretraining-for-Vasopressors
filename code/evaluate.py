"""
evaluate.py -- every number for the paper tables, from saved predictions (CPU, minutes).
Covers E2 (patient-level bootstrap CIs + DeLong), E3 (AUPRC) and the tables /
label-efficiency figure.

For every run folder in --runs_dir (finetune.py or baselines_tabular.py output)
and every horizon:
  * AUROC, AUPRC, Brier with 95% CIs from --n_boot patient-level bootstrap
    resamples (whole patients are resampled, not windows)
  * at the run's own threshold (Youden on OOF): recall, specificity,
    precision, F1, accuracy, TP/FP/FN/TN
  * vs the --reference run on the same test windows: AUROC difference,
    DeLong p-value, and a patient-bootstrap CI of the difference
  * BP: MAE and RMSE (mmHg) for MAP/SBP/DBP when present

Outputs in --out_dir:
  results_long.csv        one row per (run, horizon) with everything above
  table_auroc.csv / .tex  AUROC [95% CI] per run x horizon (+ mean)
  table_auprc.csv / .tex
  table_bp_mae.csv        MAE per run x horizon for MAP/SBP/DBP
  label_efficiency.csv + fig4_label_efficiency.pdf   (if label_frac runs exist)

Usage:
  !python evaluate.py --runs_dir /kaggle/working/runs --reference trimodal_ssl
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ORDER = ['offset_3m', 'offset_5m', 'offset_7m', 'offset_10m', 'offset_12m', 'offset_15m']
HLAB = {w: w.replace('offset_', '').replace('m', ' min') for w in ORDER}


# ---------------------------------------------------------------- DeLong (Sun & Xu 2014)
def _midrank(x):
    J = np.argsort(x, kind='mergesort')
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N)
    out[J] = T
    return out


def _fast_delong(preds, m):
    n = preds.shape[1] - m
    k = preds.shape[0]
    tx = np.array([_midrank(preds[r, :m]) for r in range(k)])
    ty = np.array([_midrank(preds[r, m:]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_test(y, p1, p2):
    """Two-sided p-value for H0: AUC(p1) == AUC(p2), paired on the same windows."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind='mergesort')
    m = int(y.sum())
    aucs, cov = _fast_delong(np.vstack([p1, p2])[:, order].astype(float), m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs[0] - aucs[1], 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0] - aucs[1], float(2 * norm.sf(abs(z)))


# ---------------------------------------------------------------- bootstrap
def patient_groups(subj):
    subj = np.asarray(subj)
    uniq, inv = np.unique(subj, return_inverse=True)
    return [np.where(inv == i)[0] for i in range(len(uniq))]


def boot_indices(groups, n_boot, seed):
    rng = np.random.RandomState(seed)
    G = len(groups)
    for _ in range(n_boot):
        pick = rng.randint(0, G, size=G)
        yield np.concatenate([groups[g] for g in pick])


def ci(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if len(vals) < 10:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- loading
_FLAT_CACHE = {}


def flat_test_windows(memmap_dir, window, max_flat_frac):
    """Test-set file names in which any channel is flat (|diff| < 1e-6) for more
    than max_flat_frac of the window -- i.e. no usable signal."""
    key = (window, max_flat_frac)
    if key not in _FLAT_CACHE:
        d = os.path.join(memmap_dir, 'test', window)
        N = int(np.load(os.path.join(d, 'n_samples.npy'))[0])
        fn = np.load(os.path.join(d, 'filenames.npy'), allow_pickle=True).astype(str)
        worst = np.zeros(N)
        for c in ['ppg', 'abp', 'ecg']:
            a = np.memmap(os.path.join(d, f'{c}.dat'), dtype='float32', mode='r', shape=(N, 3750))
            worst = np.maximum(worst, (np.abs(np.diff(a, axis=1)) < 1e-6).mean(axis=1))
        _FLAT_CACHE[key] = set(fn[worst > max_flat_frac])
        if _FLAT_CACHE[key]:
            print(f'  [quality] {window}: excluding {len(_FLAT_CACHE[key])} test window(s) with >'
                  f'{max_flat_frac:.0%} flat signal: {sorted(str(x) for x in _FLAT_CACHE[key])}')
    return _FLAT_CACHE[key]


def load_runs(runs_dir, exclude_cases=None, memmap_dir=None, max_flat_frac=0.0):
    runs = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, '*'))):
        if not os.path.isdir(d) or os.path.basename(d).startswith('_'):
            continue
        files = glob.glob(os.path.join(d, 'preds_test_*.csv'))
        if not files:
            continue
        name = os.path.basename(d)
        cfg = {}
        if os.path.exists(os.path.join(d, 'config.json')):
            cfg = json.load(open(os.path.join(d, 'config.json')))
        per_w = {}
        for f in files:
            w = os.path.basename(f)[len('preds_test_'):-4]
            s = os.path.join(d, f'summary_{w}.json')
            df = pd.read_csv(f)
            if exclude_cases:
                df = df[~df.caseid.isin(exclude_cases)].reset_index(drop=True)
            if memmap_dir and max_flat_frac > 0:
                df = df[~df.filename.isin(flat_test_windows(memmap_dir, w, max_flat_frac))].reset_index(drop=True)
            per_w[w] = (df, json.load(open(s)) if os.path.exists(s) else {})
        runs[name] = (cfg, per_w)
    return runs


def use_masked_15(runs, window='offset_15m'):
    """Paper tables: the 15-min result of run X is taken from X_m15 (start of window masked,
    see edge_check.py). Neural runs without a masked 15-min counterpart lose their 15-min row;
    baselines (numeric BP features, unaffected by the artifact) keep it."""
    for name in [n for n in runs if n.endswith('_m15')]:
        base = name[:-4]
        cfg, per_w = runs.pop(name)
        if window not in per_w:
            continue
        df, summ = per_w[window]
        summ = dict(summ, _masked=True)
        if base in runs:
            runs[base][1][window] = (df, summ)
        else:
            runs[base] = (cfg, {window: (df, summ)})
        print(f'  [masked 15] {base}: 15-min result from {name}')
    for name, (cfg, per_w) in runs.items():
        if window in per_w and not name.startswith('baseline_') and not cfg.get('mask_start'):
            if per_w[window][1].get('_masked'):
                continue
            del per_w[window]
            print(f'  [masked 15] {name}: unmasked 15-min result dropped')
    return {k: v for k, v in runs.items() if v[1]}


def threshold_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(recall=rec, specificity=spec, precision=prec, f1=f1, accuracy=(tp + tn) / len(y),
                tp=tp, fp=fp, fn=fn, tn=tn)


def fmt_ci(v, lo, hi, d=3):
    if not np.isfinite(v):
        return ''
    return f'{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]' if np.isfinite(lo) else f'{v:.{d}f}'


def pooled_comparison(runs, reference, n_boot, seed):
    """Mean AUROC over the horizons both runs share, compared with the reference
    by resampling whole test patients (the same patients appear at every
    horizon, so the comparison is paired). Two-sided bootstrap p-value."""
    ref_w = {w: v[0] for w, v in runs[reference][1].items() if 'prob' in v[0] and v[0].prob.notna().all()}
    rows = []
    for name, (cfg, per_w) in runs.items():
        if name == reference:
            continue
        ws = [w for w in ORDER if w in per_w and w in ref_w and 'prob' in per_w[w][0]
              and per_w[w][0].prob.notna().all()]
        if len(ws) < 2:
            continue
        data = {}
        for w in ws:
            mm = per_w[w][0][['filename', 'label', 'prob', 'subjectid']].merge(
                ref_w[w][['filename', 'prob']], on='filename', suffixes=('', '_ref'))
            groups = {}
            for i, sid in enumerate(mm.subjectid.values):
                groups.setdefault(sid, []).append(i)
            data[w] = (mm.label.values, mm.prob.values, mm.prob_ref.values,
                       {k: np.array(v) for k, v in groups.items()})
        subjects = sorted(set().union(*[set(d[3]) for d in data.values()]))
        point_this = np.mean([roc_auc_score(d[0], d[1]) for d in data.values()])
        point_ref = np.mean([roc_auc_score(d[0], d[2]) for d in data.values()])
        rng = np.random.RandomState(seed)
        diffs = []
        for _ in range(n_boot):
            pick = rng.choice(subjects, size=len(subjects), replace=True)
            a, b, ok = [], [], True
            for y, p, pr, groups in data.values():
                idx = np.concatenate([groups[s] for s in pick if s in groups])
                if len(np.unique(y[idx])) < 2:
                    ok = False
                    break
                a.append(roc_auc_score(y[idx], pr[idx]))
                b.append(roc_auc_score(y[idx], p[idx]))
            if ok:
                diffs.append(np.mean(a) - np.mean(b))
        diffs = np.array(diffs)
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) if len(diffs) else np.nan
        rows.append(dict(run=name, horizons=len(ws), mean_auroc=point_this, ref_mean_auroc=point_ref,
                         ref_minus_this=point_ref - point_this,
                         diff_lo=np.percentile(diffs, 2.5) if len(diffs) else np.nan,
                         diff_hi=np.percentile(diffs, 97.5) if len(diffs) else np.nan,
                         p_bootstrap=min(p, 1.0), n_boot=len(diffs)))
    return pd.DataFrame(rows).sort_values('mean_auroc', ascending=False) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', default='/kaggle/working/runs')
    ap.add_argument('--reference', default='trimodal_ssl', help='run every other run is compared with')
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--le_prefix', default='le_',
                    help='runs whose name starts with this form the label-efficiency curve')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--exclude_cases', default=None,
                    help='CSV with a caseid column; those test windows are left out (sensitivity analysis, '
                         'e.g. splits/controls_with_bolus.csv). Use with a different --out_dir.')
    ap.add_argument('--memmap_dir', default=None, help='needed for --max_flat_frac')
    ap.add_argument('--max_flat_frac', type=float, default=0.0,
                    help='leave out test windows where any channel is flat for more than this fraction (e.g. 0.1)')
    ap.add_argument('--masked_15', action='store_true',
                    help='paper tables: 15-min results from the *_m15 runs; unmasked neural 15-min results dropped')
    ap.add_argument('--pooled_boot', type=int, default=1000,
                    help='bootstrap resamples for the pooled all-horizon comparison (0 = skip)')
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.runs_dir, '_eval')
    os.makedirs(out_dir, exist_ok=True)

    excl = set(pd.read_csv(args.exclude_cases).caseid.astype(int)) if args.exclude_cases else None
    if excl:
        print(f'Sensitivity analysis: excluding test windows of {len(excl)} cases listed in {args.exclude_cases}')
    runs = load_runs(args.runs_dir, excl, args.memmap_dir, args.max_flat_frac)
    if args.masked_15:
        runs = use_masked_15(runs)
    print(f'Runs found: {list(runs)}')
    ref = runs.get(args.reference)
    if ref is None:
        print(f'[!] reference run "{args.reference}" not found; no DeLong comparisons')

    rows = []
    for name, (cfg, per_w) in runs.items():
        for w, (df, summ) in sorted(per_w.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99):
            r = dict(run=name, window=w, n=len(df), n_pos=int(df.label.sum()),
                     n_patients=df.subjectid.nunique(),
                     label_frac=cfg.get('label_frac', 1.0), seed=cfg.get('seed'), init=cfg.get('init'),
                     freeze=cfg.get('freeze_encoder', False), modalities=cfg.get('modalities'))
            y = df.label.values.astype(int)
            groups = patient_groups(df.subjectid.values)
            has_prob = 'prob' in df and df.prob.notna().all() and len(np.unique(y)) > 1
            if has_prob:
                p = df.prob.values
                r.update(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                         brier=brier_score_loss(y, p), prevalence=y.mean())
                ba, bp, bb = [], [], []
                for idx in boot_indices(groups, args.n_boot, args.seed):
                    yb, pb = y[idx], p[idx]
                    if len(np.unique(yb)) < 2:
                        continue
                    ba.append(roc_auc_score(yb, pb))
                    bp.append(average_precision_score(yb, pb))
                    bb.append(brier_score_loss(yb, pb))
                r['auroc_lo'], r['auroc_hi'] = ci(ba)
                r['auprc_lo'], r['auprc_hi'] = ci(bp)
                r['brier_lo'], r['brier_hi'] = ci(bb)
                thr = summ.get('threshold', np.nan)
                r['threshold'] = thr
                if np.isfinite(thr):
                    r.update(threshold_metrics(y, p, thr))
                # comparison with the reference run (same windows, matched by file name)
                if ref is not None and name != args.reference and w in ref[1]:
                    rdf = ref[1][w][0]
                    mm = df[['filename', 'label', 'prob', 'subjectid']].merge(
                        rdf[['filename', 'prob']], on='filename', suffixes=('', '_ref'))
                    if len(mm) >= 0.95 * len(df) and len(np.unique(mm.label)) > 1:
                        diff, pval = delong_test(mm.label.values, mm.prob_ref.values, mm.prob.values)
                        r['ref_minus_this_auroc'] = diff
                        r['delong_p'] = pval
                        g2 = patient_groups(mm.subjectid.values)
                        ds = []
                        for idx in boot_indices(g2, args.n_boot, args.seed + 1):
                            yb = mm.label.values[idx]
                            if len(np.unique(yb)) < 2:
                                continue
                            ds.append(roc_auc_score(yb, mm.prob_ref.values[idx]) - roc_auc_score(yb, mm.prob.values[idx]))
                        r['diff_lo'], r['diff_hi'] = ci(ds)
                        r['n_matched'] = len(mm)
            # windows whose BP target file was missing were default-filled with
            # 65/120/80 mmHg by the compile step -> not real targets, drop them
            bp_ok = np.ones(len(df), dtype=bool)
            if 'map_true_1' in df:
                mt = df[[f'map_true_{j}' for j in range(1, 16)]].values
                st = df[[f'sbp_true_{j}' for j in range(1, 16)]].values
                bp_ok = ~(np.all(np.isclose(mt, 65.0, atol=1e-3), 1) & np.all(np.isclose(st, 120.0, atol=1e-3), 1))
                r['bp_n_default_filled'] = int((~bp_ok).sum())
            for k in ['map', 'sbp', 'dbp']:
                pc = [c for c in df.columns if c.startswith(f'{k}_pred_')]
                tc = [c.replace('_pred_', '_true_') for c in pc]
                if pc and all(c in df for c in tc):
                    e = df.loc[bp_ok, pc].values - df.loc[bp_ok, tc].values
                    r[f'mae_{k}'] = float(np.abs(e).mean())
                    r[f'rmse_{k}'] = float(np.sqrt((e ** 2).mean()))
            rows.append(r)
            print(f'  {name:28s} {w:11s} AUROC={r.get("auroc", np.nan):.4f} '
                  f'[{r.get("auroc_lo", np.nan):.3f},{r.get("auroc_hi", np.nan):.3f}]  '
                  f'AUPRC={r.get("auprc", np.nan):.4f}  DeLong p={r.get("delong_p", np.nan):.3g}')

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, 'results_long.csv'), index=False)

    # ---- wide tables ----
    for metric in ['auroc', 'auprc']:
        if metric not in res:
            continue
        sub = res[res[metric].notna()].copy()
        sub['cell'] = [fmt_ci(v, lo, hi) for v, lo, hi in zip(sub[metric], sub[f'{metric}_lo'], sub[f'{metric}_hi'])]
        wide = sub.pivot(index='run', columns='window', values='cell')
        wide = wide[[c for c in ORDER if c in wide.columns]]
        means = sub.pivot(index='run', columns='window', values=metric)
        wide['mean'] = means[[c for c in ORDER if c in means.columns]].mean(axis=1).map(lambda v: f'{v:.3f}')
        wide.to_csv(os.path.join(out_dir, f'table_{metric}.csv'))
        with open(os.path.join(out_dir, f'table_{metric}.tex'), 'w') as f:
            f.write('% ' + ' & '.join(['Run'] + [HLAB.get(c, c) for c in wide.columns]) + ' \\\\\n')
            for run, row in wide.iterrows():
                cells = [str(v).replace(' [', ' {\\scriptsize[').replace(']', ']}') if '[' in str(v) else str(v)
                         for v in row.values]
                f.write(run.replace('_', '\\_') + ' & ' + ' & '.join(cells) + ' \\\\\n')
    bp = [c for c in ['mae_map', 'mae_sbp', 'mae_dbp'] if c in res]
    if bp:
        res.pivot_table(index='run', columns='window', values=bp).to_csv(os.path.join(out_dir, 'table_bp_mae.csv'))

    # ---- pooled comparison over all horizons (paired patient bootstrap) ----
    if ref is not None and args.pooled_boot:
        pooled = pooled_comparison(runs, args.reference, args.pooled_boot, args.seed + 2)
        if len(pooled):
            pooled.to_csv(os.path.join(out_dir, 'pooled_vs_reference.csv'), index=False)
            print('\nPooled over horizons (mean AUROC; paired patient bootstrap vs reference):')
            print(pooled.round(4).to_string(index=False))

    # ---- label efficiency ----
    le = (res[res.auroc.notna() & res.run.str.startswith(args.le_prefix)]
          if 'auroc' in res else pd.DataFrame())
    if len(le) and le.label_frac.nunique() > 1:
        le = le.copy()
        le['variant'] = np.where(le.init == 'scratch', 'scratch', 'pretrained')
        agg = (le.groupby(['window', 'variant', 'label_frac']).auroc
               .agg(['mean', 'std', 'count']).reset_index())
        agg.to_csv(os.path.join(out_dir, 'label_efficiency.csv'), index=False)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            for w, g in agg.groupby('window'):
                fig, ax = plt.subplots(figsize=(3.5, 2.6))
                for v, mk in [('pretrained', 'o-'), ('scratch', 's--')]:
                    gg = g[g.variant == v].sort_values('label_frac')
                    if len(gg):
                        ax.errorbar(gg.label_frac * 100, gg['mean'], yerr=gg['std'].fillna(0), fmt=mk,
                                    capsize=3, lw=1.5, ms=4, label='Pretrained' if v == 'pretrained' else 'From scratch')
                ax.set_xlabel('Labelled training patients (%)')
                ax.set_ylabel('Test AUROC')
                ax.set_xticks([10, 25, 50, 100])
                ax.grid(alpha=0.3)
                ax.legend(frameon=False, fontsize=8)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f'fig4_label_efficiency_{w}.pdf'))
                plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f'[warn] plot skipped: {e}')

    print(f'\n[OK] {out_dir}')


if __name__ == '__main__':
    main()
