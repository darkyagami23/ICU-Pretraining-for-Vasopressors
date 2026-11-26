# Splits

Copy these files from the output of `make_splits.py` (`splits/`):

- `windows.csv`: one row per window (split, index, file name, case id, dose, label, VitalDB `subjectid`, fold, exclude flag). Test set: 232 windows per horizon (18 positive, 12 patients). Folds: 5, patient-level, identical at every horizon. Rows with `exclude = 1` are the 114 training/validation windows of the 9 patients who also have test cases.
- `cohort_table.csv`: cohort characteristics (Table I).
- `controls_with_bolus.csv`: controls that received an intraoperative bolus (sensitivity analysis).
- `report.txt`: split and leakage report.

Case and subject identifiers are public VitalDB identifiers.
