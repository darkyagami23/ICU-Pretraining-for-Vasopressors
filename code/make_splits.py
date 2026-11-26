"""
make_splits.py -- STEP 1 (CPU only, minutes). Run ONCE; every later script reads its output.

What it does
  1. Reads filenames.npy + labels.dat from your compiled physio_memmap for every
     horizon, and recovers case id and dose number for every window.
  2. Maps case id -> VitalDB subjectid (one patient can have several surgeries)
     using the VitalDB clinical table, if you give it (--vitaldb_cases).
  3. Checks for leakage: the same PATIENT in test and in train/val. By default
     those train/val windows are excluded (test set stays exactly as before).
  4. Assigns 5 CV folds at PATIENT level, stratified by whether the patient has
     any positive window, and uses the SAME fold for a patient at every horizon.
     (This replaces StratifiedKFold over windows in downstream_code.ipynb.)
  5. Writes numbers the paper needs: test positives and the number of distinct
     test patients behind them, cases with several doses, Table 1 (if the
     clinical table is available), and controls that received bolus pressors.

Outputs (in --out_dir):
  windows.csv   one row per window: window, split, idx, filename, caseid, dose,
                label, subjectid, fold (-1 for test), exclude (0/1)
  report.txt    the checks above
  cohort_table.csv  (only with --vitaldb_cases)

Usage (Kaggle; internet ON lets it fetch the clinical table directly):
  !python make_splits.py --memmap_dir /kaggle/input/<...>/physio_memmap \
        --vitaldb_cases https://api.vitaldb.net/cases --out_dir /kaggle/working/splits
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data_utils import window_table

WINDOWS = ['offset_3m', 'offset_5m', 'offset_7m', 'offset_10m', 'offset_12m', 'offset_15m']


class Tee:
    def __init__(self, path):
        self.f = open(path, 'w')

    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        print(s)
        self.f.write(s + '\n')
        self.f.flush()


def iqr_str(x):
    x = pd.to_numeric(x, errors='coerce').dropna()
    if len(x) == 0:
        return 'n/a'
    return f'{np.median(x):.0f} [{np.percentile(x, 25):.0f}-{np.percentile(x, 75):.0f}]'


def pct_str(mask):
    mask = mask.dropna()
    return 'n/a' if len(mask) == 0 else f'{100 * mask.mean():.1f}'


def cohort_table(cases, clin):
    """cases: DataFrame caseid, group ('Vasopressor'/'Control'), split."""
    df = cases.merge(clin, on='caseid', how='left')
    out = {}
    groups = [('Vasopressor cases', df[df.group == 'Vasopressor']),
              ('Control cases', df[df.group == 'Control']),
              ('Total', df)]
    for name, g in groups:
        col = {}
        col['Cases, n'] = len(g)
        col['Patients, n'] = g['subjectid'].nunique() if 'subjectid' in g else 'n/a'
        if 'age' in g:
            col['Age, years'] = iqr_str(g['age'])
        if 'sex' in g:
            col['Female sex, %'] = pct_str(g['sex'].astype(str).str.upper().str.startswith('F'))
        if 'asa' in g:
            col['ASA >= III, %'] = pct_str(pd.to_numeric(g['asa'], errors='coerce') >= 3)
        if 'ane_type' in g:
            col['General anaesthesia, %'] = pct_str(g['ane_type'].astype(str).str.lower().str.contains('general'))
        if 'opstart' in g and 'opend' in g:
            dur = (pd.to_numeric(g['opend'], errors='coerce') - pd.to_numeric(g['opstart'], errors='coerce')) / 60
            col['Surgery duration, min'] = iqr_str(dur)
        if 'department' in g:
            vc = g['department'].value_counts(normalize=True) * 100
            col['Department, %'] = '; '.join(f'{k} {v:.0f}' for k, v in vc.items())
        for s in ['train', 'val', 'test']:
            col[f'{s} cases'] = int((g['split'] == s).sum())
        out[name] = col
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memmap_dir', required=True)
    ap.add_argument('--windows', nargs='+', default=WINDOWS)
    ap.add_argument('--vitaldb_cases', default=None,
                    help='Path or URL of the VitalDB clinical table (https://api.vitaldb.net/cases)')
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--keep_leaks', action='store_true',
                    help='Do NOT exclude train/val windows of patients who are also in test')
    ap.add_argument('--drop_cases', default=None,
                    help='CSV with a caseid column: cases to exclude everywhere (incl. test), '
                         'e.g. controls_with_bolus.csv from a first run')
    ap.add_argument('--out_dir', default='splits')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log = Tee(os.path.join(args.out_dir, 'report.txt'))

    df = pd.concat([window_table(args.memmap_dir, w) for w in args.windows], ignore_index=True)
    log(f'Windows read: {len(df)} across {df.window.nunique()} horizons, {df.caseid.nunique()} cases')

    # ---- subject mapping --------------------------------------------------------
    clin = None
    if args.vitaldb_cases:
        try:
            clin = pd.read_csv(args.vitaldb_cases)
            clin.columns = [c.strip() for c in clin.columns]
            log(f'Clinical table: {len(clin)} rows, columns include: '
                f'{", ".join(c for c in ["caseid", "subjectid", "age", "sex", "asa", "ane_type", "department", "intraop_eph", "intraop_phe", "intraop_epi"] if c in clin.columns)}')
        except Exception as e:  # noqa: BLE001
            log(f'[!] Could not read {args.vitaldb_cases}: {e}')
            clin = None
    if clin is not None and 'subjectid' in clin.columns:
        smap = dict(zip(clin.caseid.astype(int), clin.subjectid.astype(int)))
        df['subjectid'] = df.caseid.map(smap)
        n_unmapped = df.subjectid.isna().sum()
        if n_unmapped:
            log(f'[!] {n_unmapped} windows have no subjectid; using caseid for them')
        df['subjectid'] = df.subjectid.fillna(df.caseid).astype(int)
    else:
        log('[!] No subject map: grouping by caseid. Patients with several surgeries can still leak. '
            'Re-run with --vitaldb_cases https://api.vitaldb.net/cases')
        df['subjectid'] = df.caseid

    # ---- sanity checks -------------------------------------------------------
    both = df.groupby('caseid').label.nunique()
    both = both[both > 1].index.tolist()
    log(f'Cases used both as vasopressor case and as control: {len(both)} {both[:20]}')

    case_split = df.groupby('caseid').split.nunique()
    if (case_split > 1).any():
        log(f'[!] {int((case_split > 1).sum())} cases appear in more than one split (case-level leak!)')

    subj_splits = df.groupby('subjectid').split.agg(lambda s: set(s))
    leak_subj = [s for s, ss in subj_splits.items() if 'test' in ss and (ss & {'train', 'val'})]
    log(f'Patients in BOTH test and train/val: {len(leak_subj)}')
    df['exclude'] = 0
    if args.drop_cases:
        drop = set(pd.read_csv(args.drop_cases).caseid.astype(int))
        m = df.caseid.isin(drop)
        df.loc[m, 'exclude'] = 1
        log(f'Dropped {int(m.sum())} windows from {len(drop)} cases listed in {args.drop_cases} (all splits)')
    if leak_subj and not args.keep_leaks:
        m = df.subjectid.isin(leak_subj) & df.split.isin(['train', 'val'])
        df.loc[m, 'exclude'] = 1
        log(f'  -> excluded {int(m.sum())} train/val windows of those patients (test unchanged)')

    # ---- patient-level folds (same for every horizon) ------------------------
    dev = df[df.split.isin(['train', 'val']) & (df.exclude == 0)]
    pt = dev.groupby('subjectid').label.max().rename('plabel').reset_index().sort_values('subjectid')
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    pt['fold'] = -1
    for k, (_, va) in enumerate(skf.split(pt, pt.plabel)):
        pt.iloc[va, pt.columns.get_loc('fold')] = k
    df = df.merge(pt[['subjectid', 'fold']], on='subjectid', how='left')
    df['fold'] = df.fold.fillna(-1).astype(int)
    df.loc[df.split == 'test', 'fold'] = -1
    df.loc[df.exclude == 1, 'fold'] = -1
    log(f'Dev patients: {len(pt)} ({int(pt.plabel.sum())} with >=1 positive window)')

    # ---- per-horizon report ----------------------------------------------------
    log('\nPer horizon:')
    for w in args.windows:
        d = df[df.window == w]
        dv = d[(d.fold >= 0)]
        te = d[(d.split == 'test') & (d.exclude == 0)]
        pos_te = te[te.label == 1]
        multi = d[d.label == 1].groupby('caseid').size()
        log(f'  {w:11s} dev={len(dv)} (pos {int(dv.label.sum())}) | test={len(te)} (pos {len(pos_te)}, '
            f'prev {100 * te.label.mean():.1f}%) | test positives come from {pos_te.caseid.nunique()} cases / '
            f'{pos_te.subjectid.nunique()} patients | cases with >1 positive window: {int((multi > 1).sum())}')
        fc = dv.groupby('fold').label.agg(['size', 'sum'])
        log('     folds (n, pos): ' + ', '.join(f'{k}:({int(r["size"])},{int(r["sum"])})' for k, r in fc.iterrows()))

    # ---- clinical extras -------------------------------------------------------
    if clin is not None:
        cases = df.groupby('caseid').agg(label=('label', 'max'), split=('split', 'first')).reset_index()
        cases['group'] = np.where(cases.label == 1, 'Vasopressor', 'Control')
        clin2 = clin.copy()
        clin2['caseid'] = clin2.caseid.astype(int)
        if 'subjectid' not in clin2.columns:
            clin2['subjectid'] = clin2.caseid
        tab = cohort_table(cases, clin2)
        tab.to_csv(os.path.join(args.out_dir, 'cohort_table.csv'))
        log('\nCohort table (Table 1):')
        log(tab.to_string())

        drug_cols = [c for c in ['intraop_eph', 'intraop_phe', 'intraop_epi'] if c in clin2.columns]
        if drug_cols:
            ctrl = cases[cases.group == 'Control'].merge(clin2[['caseid'] + drug_cols], on='caseid', how='left')
            got = (ctrl[drug_cols].apply(pd.to_numeric, errors='coerce').fillna(0) > 0).any(axis=1)
            log(f'\n[!] Control cases with a recorded intra-op bolus of {"/".join(drug_cols)}: '
                f'{int(got.sum())} of {len(ctrl)}')
            if got.any():
                bad = ctrl.loc[got, 'caseid'].tolist()
                pd.Series(bad, name='caseid').to_csv(os.path.join(args.out_dir, 'controls_with_bolus.csv'),
                                                     index=False)
                log('    These "controls" received a vasopressor bolus (not visible in infusion tracks). '
                    'List saved to controls_with_bolus.csv. Decide whether to drop them (state it in the paper).')

    out = os.path.join(args.out_dir, 'windows.csv')
    df.to_csv(out, index=False)
    log(f'\n[OK] wrote {out}')


if __name__ == '__main__':
    sys.exit(main())
