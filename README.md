# Self-supervised multimodal waveform pretraining for multi-horizon prediction of intraoperative vasopressor requirement

Code and results for:

> M. M. Ahasan, J. Briti, M. Rahman, S. B. Shuvo and J. Ferdous, "Self-Supervised Multimodal Waveform Pretraining for Multi-Horizon Prediction of Intraoperative Vasopressor Requirement," *IEEE Journal of Biomedical and Health Informatics*, 2026 (submitted).

A hybrid transformer encoder (private branches for PPG and ABP, a learnable-wavelet branch for ECG, and a shared cross-modal transformer) is pretrained on unlabelled ICU waveforms (MIMIC-BP) and fine-tuned on VitalDB to predict the start of a vasopressor or inotrope infusion 3–15 min ahead.

## Main result (test AUROC, patient-level evaluation)

| Model | 3 | 5 | 7 | 10 | 12 | 15* | Mean |
|---|---|---|---|---|---|---|---|
| Tri-modal, pretrained | 0.864 | 0.883 | 0.876 | 0.885 | 0.841 | 0.861 | 0.868 |
| Tri-modal, from scratch | 0.874 | 0.885 | 0.873 | 0.858 | 0.808 | 0.815 | 0.852 |
| Gradient boosting on numeric BP | 0.881 | 0.937 | 0.909 | 0.839 | 0.726 | 0.761 | 0.842 |
| Logistic regression on last MAP | 0.747 | 0.851 | 0.869 | 0.796 | 0.725 | 0.680 | 0.778 |

Horizon in minutes. \*15-min neural results use the window-start mask (see *Data artefact* below). None of the differences from the pretrained model is significant (paired patient bootstrap). All tables of the paper are in [`results/SUPPLEMENTARY_NUMBERS.md`](results/SUPPLEMENTARY_NUMBERS.md).

## Repository layout

| Folder | Contents |
|---|---|
| `code/` | Model, pretraining, fine-tuning, baselines and evaluation (`common.py`, `data_utils.py`, `pretrain.py`, `finetune.py`, `baselines_tabular.py`, `evaluate.py`, `make_splits.py`, `predict_thesis.py`, `check_checkpoint.py`) |
| `audit/` | The data-artefact analysis: `norm_check.py`, `ppg15_check.py`, `edge_check.py` |
| `data_extraction/` | VitalDB label and window extraction (see its README for version notes) |
| `notebooks/` | The Kaggle and Colab notebooks used to run every experiment |
| `splits/` | Patient-level split and fold assignment, cohort table |
| `results/` | Test predictions of every run and the evaluation tables; `evaluate.py` rebuilds all tables from them without a GPU |
| `paper/` | Figure scripts |
| `original_thesis_code/` | The original thesis notebooks the code was derived from (for provenance) |

## Data access

The data cannot be redistributed and must be obtained from the sources:

- **VitalDB** (intraoperative cohort): https://vitaldb.net/dataset (open access).
- **MIMIC-BP** (pretraining corpus): Harvard Dataverse, doi:10.7910/DVN/DBM1NF, under the MIMIC-III data use agreement (PhysioNet credentialing required).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The experiments were run on Kaggle (2× NVIDIA T4, 4× NVIDIA L4) and Google Colab (T4). Evaluation runs on CPU.

## Reproducing the results

### Option A: inspect or rebuild the tables (CPU)

`results/eval_paper/` holds the evaluation output the paper was written from, and `results/SUPPLEMENTARY_NUMBERS.md` lists every number in the paper and supplement. The per-run prediction folders (`runs/`) are not included because of their size. Given them, this command rebuilds every table:

```bash
cd code
python evaluate.py --runs_dir <runs> --reference trimodal_ssl \
    --memmap_dir <physio_memmap> --max_flat_frac 0.1 --masked_15 --out_dir ../results/eval_paper
```

`--memmap_dir` is needed only for the signal-quality rule (flat-window exclusion). `--masked_15` takes all 15-min neural results from the masked runs, as in the paper. Add `--exclude_cases ../splits/controls_with_bolus.csv` for the sensitivity analysis without bolus controls.

### Option B: rerun the experiments (GPU)

1. **Extract windows** from VitalDB (`data_extraction/`) and compile them into per-horizon memmaps (`physio_memmap/{train,val,test}/offset_{3,5,7,10,12,15}m/`).
2. **Splits:**
   ```bash
   python make_splits.py --memmap_dir <physio_memmap> --vitaldb_cases https://api.vitaldb.net/cases --out_dir ../splits
   ```
3. **Pretraining** on MIMIC-BP (the main encoder used 100 epochs; the ablation variants 30):
   ```bash
   python pretrain.py --mimic_dir <MIMIC-BP/dataverse_files> --variant full --epochs 100 --tag full
   # ablations: --variant {full,no_contrastive,no_morph,recon_only} --epochs 30, and --ecg_encoder raw
   ```
4. **Fine-tuning** (one run = 5 patient-level folds per horizon):
   ```bash
   python finetune.py --memmap_dir <physio_memmap> --windows_csv ../splits/windows.csv \
       --init pretrained --encoder_path <hybrid_v4_encoders.pth> --run_name trimodal_ssl --out_dir ../results/runs
   ```
5. **Baselines:**
   ```bash
   python baselines_tabular.py --memmap_dir <physio_memmap> --csv_dir <Master_Training_Data> \
       --windows_csv ../splits/windows.csv --out_dir ../results/runs
   ```
6. **Evaluate** as in option A.

`notebooks/02_finetune.ipynb` contains every fine-tuning configuration as named groups. The table below maps each part of the paper to its group.

| Paper | Runs | `finetune.py` flags (beyond the common ones) |
|---|---|---|
| Table II, main model | `trimodal_ssl` | `--init pretrained --encoder_path <main encoder>` |
| Table II, from scratch | `scratch` | `--init scratch` |
| Table II, pairs | `pair_*` | `--encoder_path <pair encoder> --modalities a,b` |
| Table II, thesis models | `thesis_original` | `predict_thesis.py` (re-scores the original fold models) |
| Table III, modalities | `single_*`, `drop_*` | `--modalities ...` `--windows offset_5m offset_15m` |
| Table III, frozen encoder | `frozen` | `--freeze_encoder` |
| All 15-min results | `*_m15` | `--windows offset_15m --mask_start 375` |
| Fig. 2, label efficiency | `le_{ssl,scratch}_f{0.1,0.25,0.5,1.0}_s{0..4}` | `--windows offset_5m --label_frac f --seed s` |
| Table IV, pretraining ablations | `abl_*`, `abq_*` | `--encoder_path <variant encoder>`; `abq`: `--label_frac 0.25 --seed s` |

## Data artefact and the 15-min mask

Positive (vasopressor) PPG windows were extracted with a different processing step from control windows. Per-window robust z-scoring at the network input removes the resulting offset and scale difference, but positive PPG windows at the 15-min horizon also carry a transient in the first ~2 s. It inflated 15-min AUROC of PPG-containing models to as high as 0.99. All reported 15-min results therefore zero the first 375 samples (3 s) of every channel (`--mask_start 375`) in training and testing. The analysis is reproducible with the scripts in `audit/`.

## Pretrained weights

Release [v1.0](https://github.com/darkyagami23/ICU-Pretraining-for-Vasopressors/releases/tag/v1.0) contains the encoder pretrained on MIMIC-BP (100-epoch schedule) that all main experiments use:

- `hybrid_v4_encoders.pth`: encoder weights; pass to `finetune.py --encoder_path`.
- `hybrid_v4_full.pth`: the full pretraining model (encoders, decoders, projection heads); used for the reconstruction and retrieval results (`pretrain.py --retrieval_only`).

Fine-tuned VitalDB fold models are not released.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Code: MIT (see [`LICENSE`](LICENSE)). The datasets keep their own licences and data use agreements.
