# Results

- `SUPPLEMENTARY_NUMBERS.md`: every number in the paper's tables and supplementary tables.
- `runs/`: one folder per run, copied from the final Kaggle notebook output (`runs/`). Each holds `preds_test_<horizon>.csv` (calibrated test probability per window and per fold model, BP predictions and targets), `preds_oof_<horizon>.csv`, `summary_<horizon>.json`, `summary.csv` and `config.json` (the exact flags).
- `eval_paper/`: the evaluation output used for the paper (`results_long.csv`, `table_auroc.*`, `pooled_vs_reference.csv`, `label_efficiency.csv`).

Fold checkpoints (`*.pth`) are not stored here; see the GitHub release.
