# Data extraction (VitalDB → windows)

| File | Purpose |
|---|---|
| `vasopressor_label_extraction.py` | Detects infusion starts on `Orchestra/{PHEN,NEPI,EPI,VASO,DOPA}_RATE` (rate > 1 mL/h, 0.5 mL/h for vasopressin; ≥ 10 s; gaps ≤ 60 s merged) |
| `vasopressor_data_extraction_v2.py` | Positive windows: the 15 min ending 30 s before infusion start, cut into 30-s windows at 125 Hz |
| `control_data_extraction_v3.py` | Control windows, with the same filtering, gap filling and quality rule |
| `verify_extraction.py` | Checks stored windows against the original recordings (horizon, waveforms, normalisation, targets, control placement, dose rule) |
| `build_master_dataset.py` | Selects the horizon windows (`-3m00s` … `-15m00s` for positives, the matching positions for controls) and assigns splits |

**Version note.** The training windows were produced by an earlier revision of these scripts, which is not available. `verify_extraction.py` locates stored windows in the original VitalDB recordings and compares them with what these scripts produce. On 40 cases (120 windows) the released filters reproduced the stored ABP, ECG and PPG waveforms up to offset and scale (median R² 0.985–1.000), and every well-located window ended *h* min before a pump-recorded infusion start. The earlier revision differed in three ways: it centred each control segment on the midpoint of the recording (these scripts place it differently), its new-dose rule was stricter (143 stored infusion starts against 214 found by the rule here), and it normalised positive and control PPG differently (see `../audit/` and Section S-II of the paper's supplement). The scripts here store raw, un-normalised waveforms.

Windows were compiled into memmaps (`physio_memmap/`) with the compile step of `../original_thesis_code/downstream_code.ipynb` (cell "Compile downstream physio data into memmap"). Blood-pressure targets come from a separate `BP_target/` folder: one fixed target per case (the last 30-s segment before infusion start, or the last 30 s of the control segment), reused at every horizon.
