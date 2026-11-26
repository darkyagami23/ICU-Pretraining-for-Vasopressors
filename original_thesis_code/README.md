# Original thesis code

The notebooks from the original thesis study, from which `../code/` was derived. Kept unchanged for provenance:

- `pretraining-code.ipynb`: hybrid encoder and SimMTM pretraining on MIMIC-BP.
- `downstream_code.ipynb`: memmap compile step and the original window-level fine-tuning.
- `ablation-code_for_dual-modalities.ipynb`: two-modality encoder pretraining.

The paper's experiments use `../code/`, which fixes the patch length (25), uses patient-level folds, and adds the experiment switches.
