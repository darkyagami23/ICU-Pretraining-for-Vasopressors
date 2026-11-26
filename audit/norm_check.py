# Normalisation check: were positive and control windows normalised differently?
# Paste as one cell in Notebook_2 (no GPU). Needs the memmap dataset as input.
import numpy as np, pandas as pd, glob, os
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 200)
MEM = sorted(glob.glob('/kaggle/input/**/physio_memmap/', recursive=True))[0]

def load(split, w, c):
    d = os.path.join(MEM, split, w)
    N = int(np.load(os.path.join(d, 'n_samples.npy'))[0])
    lab = np.array(np.memmap(os.path.join(d, 'labels.dat'), dtype='int32', mode='r', shape=(N,)))
    x = np.array(np.memmap(os.path.join(d, f'{c}.dat'), dtype='float32', mode='r', shape=(N, 3750)), dtype=np.float64)
    return lab, x

stats, aucs = [], []
for c in ['ppg', 'abp', 'ecg']:
    for w in ['offset_5m', 'offset_12m', 'offset_15m']:
        for split, parts in [('dev', ['train', 'val']), ('test', ['test'])]:
            L, X = zip(*[load(sp, w, c) for sp in parts]); y = np.concatenate(L); x = np.concatenate(X)
            ok = (np.abs(np.diff(x, axis=1)) < 1e-6).mean(1) <= 0.1          # same quality rule as scoring
            y, x = y[ok], x[ok]
            sd, mu = x.std(1), x.mean(1)
            for lab in [0, 1]:
                m = y == lab
                stats.append(dict(ch=c, window=w, split=split, label=lab, n=int(m.sum()),
                                  std_median=np.median(sd[m]), std_exactly_1=np.mean(np.abs(sd[m] - 1) < 1e-4),
                                  mean_median=np.median(mu[m]), mean_exactly_0=np.mean(np.abs(mu[m]) < 1e-4)))
            # what the MODEL sees: robust z-score per window (median / IQR, clipped at +-5)
            q25, med, q75 = np.percentile(x, [25, 50, 75], axis=1)
            z = np.clip((x - med[:, None]) / ((q75 - q25)[:, None] + 1e-6), -5, 5)
            zc = z - z.mean(1, keepdims=True)
            F = dict(raw_mean=mu, raw_std=sd, z_mean=z.mean(1), z_std=z.std(1), z_sat=(np.abs(z) >= 5).mean(1),
                     z_skew=(zc ** 3).mean(1) / (zc.std(1) ** 3 + 1e-9))
            for k, v in F.items():
                aucs.append(dict(ch=c, window=w, split=split, feature=k, auroc=roc_auc_score(y, v)))

S = pd.DataFrame(stats)
print('===== stored values per window: is every window z-scored the same way? (std_exactly_1 / mean_exactly_0 = share of windows)')
print(S.round(6).to_string(index=False))
A = pd.DataFrame(aucs).pivot_table(index=['ch', 'feature'], columns=['split', 'window'], values='auroc').round(3)
print('\n===== AUROC of single features: raw_* = stored values, z_* = after the model\'s own robust z-score')
print(A.to_string())
