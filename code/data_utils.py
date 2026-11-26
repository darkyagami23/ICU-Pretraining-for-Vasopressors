"""
data_utils.py -- torch-free helpers (window metadata, patient subsets, json).
Imported by common.py and by the CPU-only scripts (make_splits, baselines, evaluate).
"""
import os
import re
import json

import numpy as np
import pandas as pd

SEQ_LEN = 3750
FS = 125
CANONICAL = ['ppg', 'abp', 'ecg']


FNAME_RE = re.compile(r'Case_(\d+)_(?:dose_(\d+)|control)')


def parse_filename(fname):
    m = FNAME_RE.search(str(fname))
    if not m:
        raise ValueError(f'Cannot parse case id from {fname}')
    caseid = int(m.group(1))
    dose = int(m.group(2)) if m.group(2) is not None else -1
    return caseid, dose


def robust_z(x, clip=5.0):
    med = np.median(x)
    q75, q25 = np.percentile(x, [75, 25])
    return np.clip((x - med) / ((q75 - q25) + 1e-6), -clip, clip).astype(np.float32)



def window_table(memmap_dir, window, splits=('train', 'val', 'test')):
    """One row per window in the memmaps: split, idx, filename, caseid, dose, label."""
    rows = []
    for split in splits:
        d = os.path.join(memmap_dir, split, window)
        N = int(np.load(os.path.join(d, 'n_samples.npy'))[0])
        fn = np.load(os.path.join(d, 'filenames.npy'), allow_pickle=True).astype(str)
        lab = np.memmap(os.path.join(d, 'labels.dat'), dtype='int32', mode='r', shape=(N,))
        assert len(fn) == N, f'{d}: filenames.npy has {len(fn)} rows, n_samples={N}'
        for i in range(N):
            cid, dose = parse_filename(fn[i])
            rows.append(dict(window=window, split=split, idx=i, filename=fn[i],
                             caseid=cid, dose=dose, label=int(lab[i])))
    return pd.DataFrame(rows)


def patient_subset(patients_df, frac, seed):
    """Stratified (by patient label) random subset of patients, nested across
    fractions for a fixed seed: the 10% set is inside the 25% set, etc.
    patients_df: columns subjectid, plabel."""
    if frac >= 1.0:
        return set(patients_df.subjectid)
    rng = np.random.RandomState(seed)
    keep = set()
    for _, g in patients_df.sort_values('subjectid').groupby('plabel', sort=True):
        ids = g.subjectid.values.copy()
        rng.shuffle(ids)
        n = max(1, int(round(frac * len(ids))))
        keep.update(ids[:n].tolist())
    return keep


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def dev_test_tables(memmap_dir, window, wtab=None, cv_mode='grouped', n_folds=5, seed=42):
    """Same fold logic as finetune.py, without torch.
    Returns pm (pooled train+val rows, in ConcatDataset order), tm (test rows),
    folds [(train_idx, val_idx)], test_keep (row indices of tm to evaluate)."""
    from sklearn.model_selection import StratifiedKFold
    t = window_table(memmap_dir, window)
    pm = t[t.split.isin(['train', 'val'])].copy()
    pm['split_order'] = pm.split.map({'train': 0, 'val': 1})
    pm = pm.sort_values(['split_order', 'idx']).drop(columns='split_order').reset_index(drop=True)
    tm = t[t.split == 'test'].sort_values('idx').reset_index(drop=True)
    if cv_mode == 'grouped':
        w = wtab[wtab.window == window]
        pm = pm.merge(w[['split', 'idx', 'subjectid', 'fold', 'exclude']], on=['split', 'idx'], how='left')
        tm = tm.merge(w[['split', 'idx', 'subjectid', 'exclude']], on=['split', 'idx'], how='left')
        assert pm.fold.notna().all(), 'windows.csv does not match the memmaps'
        folds = [(np.where((pm.fold.values >= 0) & (pm.fold.values != k))[0],
                  np.where(pm.fold.values == k)[0]) for k in range(n_folds)]
        test_keep = np.where(tm.exclude.fillna(0).values == 0)[0]
    else:
        pm['subjectid'], tm['subjectid'], pm['fold'] = pm.caseid, tm.caseid, -1
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = list(skf.split(np.zeros(len(pm)), pm.label.values))
        for k, (_, v) in enumerate(folds):
            pm.loc[v, 'fold'] = k
        test_keep = np.arange(len(tm))
    return pm, tm, folds, test_keep
