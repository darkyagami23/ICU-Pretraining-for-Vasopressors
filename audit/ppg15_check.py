# PPG check: why do PPG models reach 0.93-0.99 AUROC at 15 min but not at 12 min?
# Paste as one cell in Notebook_2 (no GPU). Needs the memmap dataset and the newest Notebook_2 output as inputs.
import numpy as np, pandas as pd, glob, os
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 100)

MEM = sorted(glob.glob('/kaggle/input/**/physio_memmap/', recursive=True))[0]
best = max(glob.glob('/kaggle/input/**/runs/_eval/results_long.csv', recursive=True), key=lambda f: len(pd.read_csv(f)))
BASE = os.path.dirname(os.path.dirname(best))
print('memmaps:', MEM, '\nruns   :', BASE)

def load(split, w, c):
    d = os.path.join(MEM, split, w)
    N = int(np.load(os.path.join(d, 'n_samples.npy'))[0])
    fn = np.load(os.path.join(d, 'filenames.npy'), allow_pickle=True).astype(str)
    lab = np.array(np.memmap(os.path.join(d, 'labels.dat'), dtype='int32', mode='r', shape=(N,)))
    x = np.array(np.memmap(os.path.join(d, f'{c}.dat'), dtype='float32', mode='r', shape=(N, 3750)), dtype=np.float64)
    return fn, lab, x

def feats(a):
    q = np.percentile(a, [1, 25, 50, 75, 99], axis=1)
    iqr = q[3] - q[1]
    z = np.clip((a - q[2][:, None]) / (iqr[:, None] + 1e-6), -5, 5)
    zc = z - z.mean(1, keepdims=True)
    mx, mn = a.max(1, keepdims=True), a.min(1, keepdims=True)
    spec = np.abs(np.fft.rfft(a - a.mean(1, keepdims=True), axis=1)); spec[:, :5] = 0
    return pd.DataFrame(dict(
        mean=a.mean(1), std=a.std(1), maxabs=np.abs(a).max(1), iqr=iqr, p99_p1=q[4] - q[0],
        flat_frac=(np.abs(np.diff(a, axis=1)) < 1e-6).mean(1),
        clip_frac=(np.isclose(a, mx) | np.isclose(a, mn)).mean(1),
        z_sat_frac=(np.abs(z) >= 5).mean(1),
        z_skew=(zc ** 3).mean(1) / (zc.std(1) ** 3 + 1e-9),
        dom_freq_bin=spec.argmax(1), spec_peak_share=spec.max(1) / (spec.sum(1) + 1e-9)))

def flat_any(split, w):
    worst = 0
    for c in ['ppg', 'abp', 'ecg']:
        _, _, x = load(split, w, c)
        worst = np.maximum(worst, (np.abs(np.diff(x, axis=1)) < 1e-6).mean(1))
    return worst > 0.1

tables, aucs = {}, []
for w in ['offset_5m', 'offset_12m', 'offset_15m']:
    for split in ['test', 'dev']:
        parts = [('train', 'val') if split == 'dev' else ('test',)][0]
        F, Y, FN = [], [], []
        for sp in parts:
            fn, lab, x = load(sp, w, 'ppg')
            keep = ~flat_any(sp, w)
            F.append(feats(x[keep])); Y.append(lab[keep]); FN.append(fn[keep])
        F = pd.concat(F, ignore_index=True); Y = np.concatenate(Y); F['filename'] = np.concatenate(FN); F['label'] = Y
        tables[(w, split)] = F
        for col in F.columns.drop(['filename', 'label']):
            aucs.append(dict(window=w, split=split, feature=col, auroc=roc_auc_score(Y, F[col])))

A = pd.DataFrame(aucs).pivot_table(index='feature', columns=['split', 'window'], values='auroc').round(3)
print('\n===== single PPG features: AUROC for predicting the label (0.5 = useless; far from 0.5 either way = informative)')
print(A.to_string())

# per positive window at 15 min: features + where each model ranks it among all test windows
T = tables[('offset_15m', 'test')].copy()
for run in ['single_ppg_s1', 'pair_ppg_ecg', 'drop_ecg', 'trimodal_ssl', 'single_abp']:
    f = f'{BASE}/{run}/preds_test_offset_15m.csv'
    if os.path.exists(f):
        p = pd.read_csv(f)
        if 'filename' in p.columns:
            p = p[p.filename.isin(T.filename)]
            p['pct'] = p.prob.rank(pct=True)
            T = T.merge(p[['filename', 'pct']].rename(columns={'pct': f'rank_{run}'}), on='filename', how='left')
cols = ['filename', 'std', 'maxabs', 'iqr', 'clip_frac', 'z_sat_frac', 'z_skew', 'dom_freq_bin'] + [c for c in T.columns if c.startswith('rank_')]
print('\n===== 15-min TEST positives (rank_* = percentile of the model score among all test windows; 1.0 = highest)')
print(T[T.label == 1][cols].round(3).to_string(index=False))
print('\n===== 15-min TEST controls: median of the same columns')
print(T[T.label == 0][cols[1:]].median().round(3).to_string())
P12 = tables[('offset_12m', 'test')]
print('\n===== same positives at 12 min (for comparison)')
print(P12[P12.label == 1][cols[:8]].round(3).to_string(index=False))
