# Results

- `SUPPLEMENTARY_NUMBERS.md`: every number in the paper's tables and supplementary tables.
- `eval_paper/`: the evaluation output the paper was written from (`results_long.csv`, `table_auroc.*`, `pooled_vs_reference.csv`, `label_efficiency.csv` and related tables), produced by `code/evaluate.py` with `--max_flat_frac 0.1 --masked_15`.
- `eval_paper_nobolus/`: the same evaluation with the test windows of the 1,232 controls that received an intra-operative bolus left out (`--exclude_cases ../splits/controls_with_bolus.csv`); the sensitivity analysis reported in the paper.

The per-run prediction folders and fold checkpoints are not included because of their size. The pretrained encoder is in [release v1.0](https://github.com/darkyagami23/ICU-Pretraining-for-Vasopressors/releases/tag/v1.0).
