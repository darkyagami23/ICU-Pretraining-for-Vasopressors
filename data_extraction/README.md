# Data extraction (VitalDB → windows)

| File | Purpose |
|---|---|
| `vasopressor_label_extraction.py` | Detects infusion starts on `Orchestra/{PHEN,NEPI,EPI,VASO,DOPA}_RATE` (rate > 1 mL/h, 0.5 mL/h for vasopressin; ≥ 10 s; gaps ≤ 60 s merged) |
| `vasopressor_data_extraction_v2.py` | Positive windows: the 15 min ending 30 s before infusion start, cut into 30-s windows at 125 Hz |
| `control_data_extraction_v3.py` | Control windows, with the same filtering, gap filling and quality rule |
| `build_master_dataset.py` | Selects the horizon windows (`-3m00s` … `-15m00s` for positives, the matching positions for controls) and assigns splits |

**Version note.** The extraction scripts here are later versions than the ones that produced the paper's data: they store raw (un-normalised) waveforms and select controls differently. The original versions are not available. As described in the paper, the original pipeline processed positive and control PPG differently; see `../audit/` and the paper's Methods.

Windows were compiled into memmaps (`physio_memmap/`) with the compile step of `../original_thesis_code/downstream_code.ipynb` (cell "Compile downstream physio data into memmap"). Blood-pressure targets come from a separate `BP_target/` folder: one fixed target per case (the last 30-s segment before infusion start), reused at every horizon.
