"""
predict_thesis.py -- re-score the ORIGINAL thesis fold models on the test set
(no training; a few minutes on GPU, ~10-20 min on CPU).

The thesis notebook saved 30 fold checkpoints (hybrid_ckpts/offset_*_fold1..5.pth)
and hybrid_results.csv (temperature and threshold per horizon), but not the
per-window predictions. This script rebuilds the fold ensemble exactly as the
notebook did (mean of temperature-scaled probabilities over the 5 folds) and
saves the predictions in the finetune.py format, so evaluate.py can give the
thesis model patient-bootstrap CIs, AUPRC and DeLong tests against the
baselines today, before any retraining. It also checks that the recomputed
AUROC matches hybrid_results.csv (proves the models were loaded correctly).

Note: these models were trained with the thesis protocol (window-level folds);
the paper's final numbers come from the patient-level reruns (finetune.py).

Usage:
  !python predict_thesis.py --memmap_dir <physio_memmap> --windows_csv <splits/windows.csv> \
      --ckpt_dir <...>/hybrid_ckpts --results_csv <...>/hybrid_results.csv --out_dir /kaggle/working/runs
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader

from common import PhysiologicalDatasetMM, build_encoder_bundle, save_json
from data_utils import dev_test_tables
from finetune import DEVICE, MultiTaskHybrid, forward_all

TOP_LEVEL_DUPLICATES = ('enc_ppg.', 'enc_abp.', 'enc_ecg.', 'shared.')   # thesis model kept these twice


def load_fold(path, patch_len):
    model = MultiTaskHybrid(build_encoder_bundle(patch_len=patch_len, ecg_encoder='wavelet'),
                            ['ppg', 'abp', 'ecg']).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    state = {k: v for k, v in state.items() if not k.startswith(TOP_LEVEL_DUPLICATES)}
    model.load_state_dict(state, strict=True)      # fails loudly if anything does not match
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memmap_dir', required=True)
    ap.add_argument('--windows_csv', required=True)
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--results_csv', required=True)
    ap.add_argument('--patch_len', type=int, default=25)
    ap.add_argument('--run_name', default='thesis_original')
    ap.add_argument('--out_dir', default='/kaggle/working/runs')
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()

    res = pd.read_csv(args.results_csv)
    wtab = pd.read_csv(args.windows_csv)
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_json(dict(run_name=args.run_name, kind='thesis_original', init='pretrained', modalities='ppg,abp,ecg',
                   cv_mode='window (thesis protocol)', label_frac=1.0, seed=42), os.path.join(run_dir, 'config.json'))
    rows = []
    for _, r in res.iterrows():
        w = r['window']
        cks = sorted(glob.glob(os.path.join(args.ckpt_dir, '**', f'{w}_fold*.pth'), recursive=True))
        if len(cks) == 0:
            print(f'[!] no checkpoints for {w}')
            continue
        T = float(r['temperature'])
        thr = float(r['test_threshold']) if 'test_threshold' in r and pd.notna(r['test_threshold']) else np.nan
        _, tm, _, keep = dev_test_tables(args.memmap_dir, w, wtab, 'grouped')
        te = PhysiologicalDatasetMM(args.memmap_dir, 'test', w, verbose=False)
        loader = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=2)
        probs, mp, sp, dp = [], 0, 0, 0
        for ck in cks:
            o = forward_all(load_fold(ck, args.patch_len), loader)
            probs.append(torch.sigmoid(o['logit'] / T).numpy())
            mp, sp, dp = mp + o['mp'].numpy(), sp + o['sp'].numpy(), dp + o['dp'].numpy()
        K = len(cks)
        prob = np.mean(probs, axis=0)
        y = tm.label.values
        auroc = roc_auc_score(y, prob)
        tdf = tm[['window', 'split', 'idx', 'filename', 'caseid', 'subjectid', 'dose', 'label']].copy()
        tdf['prob'] = prob
        for k, p in enumerate(probs):
            tdf[f'prob_fold{k}'] = p
        for name, arr in [('map_pred', mp / K * 100), ('map_true', o['mt'].numpy() * 100),
                          ('sbp_pred', sp / K * 100), ('sbp_true', o['st'].numpy() * 100),
                          ('dbp_pred', dp / K * 100), ('dbp_true', o['dt'].numpy() * 100)]:
            for j in range(arr.shape[1]):
                tdf[f'{name}_{j + 1}'] = arr[:, j]
        tdf = tdf.iloc[keep].reset_index(drop=True)
        tdf.to_csv(os.path.join(run_dir, f'preds_test_{w}.csv'), index=False)
        summ = dict(run=args.run_name, window=w, temperature=T, threshold=thr, n_folds=K,
                    test_auroc=auroc, thesis_auroc=float(r.get('test_auroc', np.nan)),
                    test_auprc=average_precision_score(y, prob), test_brier=brier_score_loss(y, prob))
        save_json(summ, os.path.join(run_dir, f'summary_{w}.json'))
        rows.append(summ)
        flag = 'OK' if abs(auroc - summ['thesis_auroc']) < 0.005 else '!! DIFFERENT'
        print(f'{w:11s} folds={K}  AUROC recomputed={auroc:.4f}  thesis={summ["thesis_auroc"]:.4f}  [{flag}]  '
              f'AUPRC={summ["test_auprc"]:.4f}  thr={thr:.3f}', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, 'summary.csv'), index=False)


if __name__ == '__main__':
    main()
