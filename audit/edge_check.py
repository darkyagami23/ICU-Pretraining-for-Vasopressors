# Edge check: is there a start-of-window transient in positive PPG windows (e.g. a filter edge effect)?
# Paste as one cell in Notebook_2 (no GPU). Needs the memmap dataset as input.
import numpy as np, pandas as pd, glob, os
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 200)
MEM = sorted(glob.glob('/kaggle/input/**/physio_memmap/', recursive=True))[0]
E = 250   # samples at each edge

def load(split, w, c):
    d = os.path.join(MEM, split, w)
    N = int(np.load(os.path.join(d, 'n_samples.npy'))[0])
    lab = np.array(np.memmap(os.path.join(d, 'labels.dat'), dtype='int32', mode='r', shape=(N,)))
    x = np.array(np.memmap(os.path.join(d, f'{c}.dat'), dtype='float32', mode='r', shape=(N, 3750)), dtype=np.float64)
    return lab, x

def rz(x):   # the model's robust z-score
    q25, med, q75 = np.percentile(x, [25, 50, 75], axis=1)
    return np.clip((x - med[:, None]) / ((q75 - q25)[:, None] + 1e-6), -5, 5)

aucs, prof = [], []
for c in ['ppg', 'abp', 'ecg']:
    for w in ['offset_3m', 'offset_5m', 'offset_7m', 'offset_10m', 'offset_12m', 'offset_15m']:
        for split, parts in [('dev', ['train', 'val']), ('test', ['test'])]:
            L, X = zip(*[load(sp, w, c) for sp in parts]); y = np.concatenate(L); x = np.concatenate(X)
            ok = (np.abs(np.diff(x, axis=1)) < 1e-6).mean(1) <= 0.1
            y, z = y[ok], rz(x[ok])
            F = dict(start_absmax=np.abs(z[:, :E]).max(1), end_absmax=np.abs(z[:, -E:]).max(1),
                     start_mean=z[:, :E].mean(1), end_mean=z[:, -E:].mean(1),
                     start_minus_middle=np.abs(z[:, :E]).mean(1) - np.abs(z[:, E:-E]).mean(1))
            for k, v in F.items():
                aucs.append(dict(ch=c, window=w, split=split, feature=k, auroc=roc_auc_score(y, v)))
            if split == 'dev':
                bins = np.array_split(np.arange(3750), 15)
                for lab in [0, 1]:
                    prof.append(dict(ch=c, window=w, label=lab,
                                     **{f'b{i:02d}': np.abs(z[y == lab][:, b]).mean() for i, b in enumerate(bins)}))

A = pd.DataFrame(aucs).pivot_table(index=['ch', 'feature'], columns=['split', 'window'], values='auroc').round(3)
print('===== AUROC of edge features after the model\'s robust z-score (0.5 = no signal)')
print(A.to_string())
print('\n===== dev windows: mean |z| in 15 time bins across the window (b00 = first 250 samples), controls (0) vs positives (1)')
print(pd.DataFrame(prof).round(3).to_string(index=False))
# look for: b00 (start) or b14 (end) of positives clearly above the controls and the other bins
