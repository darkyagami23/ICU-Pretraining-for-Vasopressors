# Supplementary numbers (final)

Source: Notebook_2 cell 5, commit of 25 Nov 2026 (`runs/_eval` and `runs/_eval_m15`, the same rows `runs/_eval_paper` uses). Generated from that output; every value is a test-set value.

**Conventions**
- Test set: 232 windows per horizon (18 positives, 12 patients, prevalence 0.078). At 3, 5 and 7 min one fully flat positive window is excluded by the quality rule: 231 windows, 17 positives (prevalence 0.074). AUPRC of a random classifier equals the prevalence.
- 95 % CIs: patient-level bootstrap. DeLong p: paired test against the pretrained tri-modal model on the same windows.
- **15 min = masked runs** (first 375 samples of every channel zeroed); DeLong at 15 min is against the masked tri-modal model. The thesis fold models cannot be re-run with the mask, so their 15-min cell is empty here (see Table S6).
- Recall and specificity at the threshold chosen by Youden's J on the out-of-fold predictions. The thesis fold models use the test threshold stored with them (`hybrid_results.csv`).
- Modality runs and ablations were trained at 5 and 15 min only; the low-label ablation (25 % labels, 3 seeds) is reported as mean ± SD in the main text and in `RESULTS_LOG.md`.
- BP error = mean absolute error over the 15-step trajectory in mmHg; windows whose target was a default fill value (65/120) are excluded.

## Table S1. Test AUROC [95 % CI]
| Model | 3 min | 5 min | 7 min | 10 min | 12 min | 15 min |
|---|---|---|---|---|---|---|
| Tri-modal, pretrained (thesis encoder) | 0.864 [0.753, 0.946] | 0.883 [0.751, 0.958] | 0.876 [0.743, 0.960] | 0.885 [0.791, 0.957] | 0.841 [0.714, 0.930] | 0.861 [0.781, 0.927] |
| Tri-modal, from scratch | 0.874 [0.748, 0.958] | 0.885 [0.765, 0.962] | 0.873 [0.749, 0.953] | 0.858 [0.756, 0.931] | 0.808 [0.663, 0.926] | 0.815 [0.652, 0.924] |
| Thesis fold models (re-scored) | 0.881 [0.786, 0.950] | 0.924 [0.856, 0.972] | 0.885 [0.784, 0.951] | 0.872 [0.775, 0.946] | 0.845 [0.729, 0.936] | — |
| Frozen encoder | — | 0.855 [0.704, 0.945] | — | — | — | 0.844 [0.753, 0.917] |
| PPG only | — | 0.894 [0.838, 0.945] | — | — | — | 0.879 [0.818, 0.935] |
| ABP only | — | 0.855 [0.755, 0.919] | — | — | — | 0.765 [0.642, 0.892] |
| ECG only | — | 0.868 [0.784, 0.928] | — | — | — | 0.818 [0.732, 0.902] |
| ABP+ECG | — | 0.883 [0.798, 0.939] | — | — | — | 0.869 [0.786, 0.933] |
| PPG+ECG | — | 0.902 [0.843, 0.949] | — | — | — | 0.894 [0.831, 0.949] |
| PPG+ABP | — | 0.880 [0.783, 0.944] | — | — | — | 0.873 [0.793, 0.936] |
| Pair encoder ABP+PPG | 0.906 [0.843, 0.962] | 0.893 [0.826, 0.945] | 0.905 [0.842, 0.954] | 0.854 [0.734, 0.941] | 0.807 [0.658, 0.920] | 0.809 [0.685, 0.902] |
| Pair encoder PPG+ECG | 0.860 [0.758, 0.941] | 0.876 [0.791, 0.940] | 0.851 [0.710, 0.944] | 0.843 [0.728, 0.925] | 0.854 [0.733, 0.944] | 0.879 [0.798, 0.942] |
| Pair encoder ABP+ECG | 0.869 [0.772, 0.937] | 0.874 [0.770, 0.939] | 0.843 [0.704, 0.934] | 0.857 [0.747, 0.937] | 0.776 [0.646, 0.888] | 0.787 [0.671, 0.892] |
| Ablation: full loss (30 ep) | — | 0.856 [0.689, 0.951] | — | — | — | 0.852 [0.737, 0.933] |
| Ablation: no contrastive | — | 0.920 [0.859, 0.963] | — | — | — | 0.889 [0.820, 0.941] |
| Ablation: no morphology | — | 0.880 [0.743, 0.962] | — | — | — | 0.887 [0.791, 0.951] |
| Ablation: raw-patch ECG | — | 0.875 [0.769, 0.945] | — | — | — | 0.882 [0.808, 0.937] |
| Ablation: reconstruction only | — | 0.875 [0.784, 0.935] | — | — | — | 0.823 [0.708, 0.907] |
| GBM on numeric BP features | 0.881 [0.785, 0.964] | 0.937 [0.881, 0.973] | 0.909 [0.790, 0.971] | 0.839 [0.698, 0.951] | 0.726 [0.561, 0.853] | 0.761 [0.603, 0.907] |
| Logistic regression on last MAP | 0.747 [0.506, 0.953] | 0.851 [0.694, 0.984] | 0.869 [0.734, 0.969] | 0.796 [0.624, 0.916] | 0.725 [0.571, 0.859] | 0.680 [0.479, 0.896] |

## Table S2. Test AUPRC
| Model | 3 min | 5 min | 7 min | 10 min | 12 min | 15 min |
|---|---|---|---|---|---|---|
| Tri-modal, pretrained (thesis encoder) | 0.479 | 0.420 | 0.508 | 0.525 | 0.425 | 0.391 |
| Tri-modal, from scratch | 0.498 | 0.507 | 0.515 | 0.326 | 0.458 | 0.432 |
| Thesis fold models (re-scored) | 0.538 | 0.549 | 0.545 | 0.424 | 0.449 | — |
| Frozen encoder | — | 0.347 | — | — | — | 0.391 |
| PPG only | — | 0.313 | — | — | — | 0.294 |
| ABP only | — | 0.370 | — | — | — | 0.322 |
| ECG only | — | 0.328 | — | — | — | 0.308 |
| ABP+ECG | — | 0.427 | — | — | — | 0.345 |
| PPG+ECG | — | 0.323 | — | — | — | 0.349 |
| PPG+ABP | — | 0.388 | — | — | — | 0.422 |
| Pair encoder ABP+PPG | 0.439 | 0.337 | 0.415 | 0.513 | 0.391 | 0.334 |
| Pair encoder PPG+ECG | 0.405 | 0.311 | 0.399 | 0.324 | 0.416 | 0.385 |
| Pair encoder ABP+ECG | 0.462 | 0.371 | 0.455 | 0.491 | 0.299 | 0.351 |
| Ablation: full loss (30 ep) | — | 0.345 | — | — | — | 0.371 |
| Ablation: no contrastive | — | 0.413 | — | — | — | 0.382 |
| Ablation: no morphology | — | 0.466 | — | — | — | 0.424 |
| Ablation: raw-patch ECG | — | 0.361 | — | — | — | 0.364 |
| Ablation: reconstruction only | — | 0.347 | — | — | — | 0.350 |
| GBM on numeric BP features | 0.569 | 0.525 | 0.545 | 0.548 | 0.364 | 0.389 |
| Logistic regression on last MAP | 0.410 | 0.507 | 0.405 | 0.411 | 0.374 | 0.408 |

## Table S3. DeLong p-value vs the pretrained tri-modal model
| Model | 3 min | 5 min | 7 min | 10 min | 12 min | 15 min |
|---|---|---|---|---|---|---|
| Tri-modal, from scratch | 0.800 | 0.978 | 0.942 | 0.451 | 0.445 | 0.414 |
| Thesis fold models (re-scored) | 0.604 | 0.341 | 0.855 | 0.460 | 0.952 | — |
| Frozen encoder | — | 0.042 | — | — | — | 0.241 |
| PPG only | — | 0.824 | — | — | — | 0.650 |
| ABP only | — | 0.491 | — | — | — | 0.054 |
| ECG only | — | 0.792 | — | — | — | 0.262 |
| ABP+ECG | — | 0.994 | — | — | — | 0.731 |
| PPG+ECG | — | 0.602 | — | — | — | 0.237 |
| PPG+ABP | — | 0.908 | — | — | — | 0.676 |
| Pair encoder ABP+PPG | 0.332 | 0.852 | 0.467 | 0.281 | 0.546 | 0.187 |
| Pair encoder PPG+ECG | 0.917 | 0.874 | 0.484 | 0.032 | 0.632 | 0.575 |
| Pair encoder ABP+ECG | 0.832 | 0.824 | 0.439 | 0.268 | 0.174 | 0.029 |
| Ablation: full loss (30 ep) | — | 0.126 | — | — | — | 0.687 |
| Ablation: no contrastive | — | 0.322 | — | — | — | 0.215 |
| Ablation: no morphology | — | 0.871 | — | — | — | 0.299 |
| Ablation: raw-patch ECG | — | 0.850 | — | — | — | 0.447 |
| Ablation: reconstruction only | — | 0.831 | — | — | — | 0.382 |
| GBM on numeric BP features | 0.770 | 0.248 | 0.597 | 0.293 | 0.122 | 0.216 |
| Logistic regression on last MAP | 0.243 | 0.705 | 0.927 | 0.201 | 0.210 | 0.068 |

## Table S4. Recall / specificity at the Youden threshold
| Model | 3 min | 5 min | 7 min | 10 min | 12 min | 15 min |
|---|---|---|---|---|---|---|
| Tri-modal, pretrained (thesis encoder) | 0.65 / 0.84 | 0.77 / 0.86 | 0.77 / 0.82 | 0.61 / 0.85 | 0.83 / 0.63 | 1.00 / 0.63 |
| Tri-modal, from scratch | 0.82 / 0.79 | 0.82 / 0.85 | 0.77 / 0.80 | 0.83 / 0.67 | 0.56 / 0.88 | 0.78 / 0.75 |
| Thesis fold models (re-scored) | 0.53 / 0.93 | 0.88 / 0.78 | 0.94 / 0.65 | 0.61 / 0.89 | 0.78 / 0.71 | — |
| Frozen encoder | — | 0.77 / 0.86 | — | — | — | 0.83 / 0.72 |
| PPG only | — | 0.71 / 0.89 | — | — | — | 0.89 / 0.67 |
| ABP only | — | 0.65 / 0.80 | — | — | — | 0.28 / 0.90 |
| ECG only | — | 0.94 / 0.62 | — | — | — | 1.00 / 0.44 |
| ABP+ECG | — | 0.94 / 0.68 | — | — | — | 0.72 / 0.83 |
| PPG+ECG | — | 0.82 / 0.83 | — | — | — | 0.72 / 0.81 |
| PPG+ABP | — | 0.71 / 0.88 | — | — | — | 0.83 / 0.75 |
| Pair encoder ABP+PPG | 0.53 / 0.94 | 0.71 / 0.88 | 0.71 / 0.91 | 0.67 / 0.88 | 0.72 / 0.82 | 0.50 / 0.86 |
| Pair encoder PPG+ECG | 0.65 / 0.87 | 0.71 / 0.85 | 0.82 / 0.79 | 0.72 / 0.76 | 0.72 / 0.78 | 0.94 / 0.65 |
| Pair encoder ABP+ECG | 0.59 / 0.90 | 0.77 / 0.82 | 0.77 / 0.77 | 0.61 / 0.82 | 0.61 / 0.81 | 0.50 / 0.85 |
| Ablation: full loss (30 ep) | — | 0.82 / 0.84 | — | — | — | 0.56 / 0.89 |
| Ablation: no contrastive | — | 0.82 / 0.87 | — | — | — | 0.56 / 0.90 |
| Ablation: no morphology | — | 0.77 / 0.82 | — | — | — | 0.50 / 0.90 |
| Ablation: raw-patch ECG | — | 0.88 / 0.76 | — | — | — | 0.61 / 0.86 |
| Ablation: reconstruction only | — | 0.77 / 0.84 | — | — | — | 0.56 / 0.87 |
| GBM on numeric BP features | 0.82 / 0.76 | 0.88 / 0.83 | 0.94 / 0.79 | 0.72 / 0.83 | 0.44 / 0.88 | 0.67 / 0.70 |
| Logistic regression on last MAP | 0.65 / 0.86 | 0.82 / 0.84 | 0.82 / 0.82 | 0.72 / 0.78 | 0.44 / 0.86 | 0.56 / 0.86 |

## Table S5. Blood-pressure trajectory MAE (mmHg)
### S5a. MAP
| Model | 3 min | 5 min | 7 min | 10 min | 12 min | 15 min |
|---|---|---|---|---|---|---|
| Tri-modal, pretrained (thesis encoder) | 10.0 | 10.1 | 10.1 | 10.6 | 10.7 | 10.8 |
| Tri-modal, from scratch | 10.2 | 10.6 | 10.5 | 11.0 | 11.2 | 10.9 |
| Thesis fold models (re-scored) | 11.6 | 12.0 | 11.4 | 11.7 | 12.3 | — |
| Frozen encoder | — | 10.2 | — | — | — | 10.9 |
| PPG only | — | 10.5 | — | — | — | 11.2 |
| ABP only | — | 10.0 | — | — | — | 10.9 |
| ECG only | — | 11.3 | — | — | — | 11.3 |
| ABP+ECG | — | 9.9 | — | — | — | 10.8 |
| PPG+ECG | — | 10.4 | — | — | — | 10.9 |
| PPG+ABP | — | 10.0 | — | — | — | 11.1 |
| Pair encoder ABP+PPG | 10.1 | 10.0 | 10.2 | 10.7 | 10.8 | 11.0 |
| Pair encoder PPG+ECG | 10.3 | 10.5 | 10.6 | 11.0 | 11.0 | 10.9 |
| Pair encoder ABP+ECG | 10.2 | 10.0 | 10.0 | 10.5 | 10.7 | 10.7 |
| Ablation: full loss (30 ep) | — | 10.1 | — | — | — | 10.9 |
| Ablation: no contrastive | — | 10.1 | — | — | — | 10.7 |
| Ablation: no morphology | — | 10.1 | — | — | — | 10.9 |
| Ablation: raw-patch ECG | — | 10.3 | — | — | — | 10.9 |
| Ablation: reconstruction only | — | 10.1 | — | — | — | 10.9 |
| Persistence (BP) | 5.7 | 7.1 | 8.3 | 10.0 | 10.1 | 10.7 |

### S5b. SBP
| Model | 3 min | 5 min | 7 min | 10 min | 12 min | 15 min |
|---|---|---|---|---|---|---|
| Tri-modal, pretrained (thesis encoder) | 13.2 | 13.4 | 13.3 | 14.1 | 14.0 | 14.4 |
| Tri-modal, from scratch | 13.5 | 14.1 | 14.0 | 14.8 | 14.6 | 14.6 |
| Thesis fold models (re-scored) | 14.8 | 15.3 | 14.7 | 15.1 | 15.5 | — |
| Frozen encoder | — | 13.7 | — | — | — | 14.4 |
| PPG only | — | 14.2 | — | — | — | 14.9 |
| ABP only | — | 13.5 | — | — | — | 14.6 |
| ECG only | — | 14.9 | — | — | — | 15.0 |
| ABP+ECG | — | 13.3 | — | — | — | 14.3 |
| PPG+ECG | — | 14.2 | — | — | — | 14.7 |
| PPG+ABP | — | 13.4 | — | — | — | 14.8 |
| Pair encoder ABP+PPG | 13.4 | 13.6 | 13.6 | 14.3 | 14.3 | 14.5 |
| Pair encoder PPG+ECG | 13.9 | 14.4 | 14.2 | 14.8 | 14.6 | 14.6 |
| Pair encoder ABP+ECG | 13.6 | 13.4 | 13.5 | 14.2 | 14.2 | 14.3 |
| Ablation: full loss (30 ep) | — | 13.7 | — | — | — | 14.5 |
| Ablation: no contrastive | — | 13.7 | — | — | — | 14.1 |
| Ablation: no morphology | — | 13.5 | — | — | — | 14.6 |
| Ablation: raw-patch ECG | — | 13.8 | — | — | — | 14.5 |
| Ablation: reconstruction only | — | 13.6 | — | — | — | 14.6 |
| Persistence (BP) | 7.9 | 9.7 | 11.5 | 12.6 | 13.5 | 14.1 |

### S5c. DBP
| Model | 3 min | 5 min | 7 min | 10 min | 12 min | 15 min |
|---|---|---|---|---|---|---|
| Tri-modal, pretrained (thesis encoder) | 8.4 | 8.4 | 8.5 | 8.8 | 8.8 | 8.9 |
| Tri-modal, from scratch | 8.6 | 8.8 | 8.9 | 9.1 | 8.9 | 9.1 |
| Thesis fold models (re-scored) | 8.9 | 9.6 | 9.0 | 9.2 | 9.6 | — |
| Frozen encoder | — | 8.6 | — | — | — | 9.0 |
| PPG only | — | 8.9 | — | — | — | 9.3 |
| ABP only | — | 8.3 | — | — | — | 9.0 |
| ECG only | — | 9.2 | — | — | — | 9.1 |
| ABP+ECG | — | 8.4 | — | — | — | 8.9 |
| PPG+ECG | — | 8.8 | — | — | — | 9.1 |
| PPG+ABP | — | 8.3 | — | — | — | 9.2 |
| Pair encoder ABP+PPG | 8.3 | 8.4 | 8.5 | 8.8 | 8.8 | 9.1 |
| Pair encoder PPG+ECG | 8.6 | 8.9 | 8.8 | 9.1 | 9.0 | 9.0 |
| Pair encoder ABP+ECG | 8.6 | 8.5 | 8.5 | 8.8 | 9.0 | 8.9 |
| Ablation: full loss (30 ep) | — | 8.5 | — | — | — | 9.0 |
| Ablation: no contrastive | — | 8.4 | — | — | — | 8.9 |
| Ablation: no morphology | — | 8.6 | — | — | — | 9.0 |
| Ablation: raw-patch ECG | — | 8.6 | — | — | — | 9.0 |
| Ablation: reconstruction only | — | 8.5 | — | — | — | 9.0 |
| Persistence (BP) | 4.1 | 5.2 | 6.5 | 6.8 | 7.4 | 7.8 |

## Table S6. 15-min results without and with the start-of-window mask
| Model | Unmasked AUROC [95 % CI] | Unmasked AUPRC | Masked AUROC [95 % CI] | Masked AUPRC |
|---|---|---|---|---|
| PPG only | 0.923 [0.859, 0.972] (seeds 1, 2: 0.947, 0.926) | 0.441 | 0.879 [0.818, 0.935] | 0.294 |
| PPG+ABP | 0.984 [0.952, 1.000] (seeds 1, 2: 0.969, 0.853) | 0.900 | 0.873 [0.793, 0.936] | 0.422 |
| PPG+ECG | 0.877 [0.807, 0.938] | 0.359 | 0.894 [0.831, 0.949] | 0.349 |
| Pair ABP+PPG | 0.967 [0.929, 0.992] | 0.663 | 0.809 [0.685, 0.902] | 0.334 |
| Pair PPG+ECG | 0.992 [0.980, 1.000] | 0.828 | 0.879 [0.798, 0.942] | 0.385 |
| Tri-modal, pretrained | 0.862 [0.788, 0.920] | 0.374 | 0.861 [0.781, 0.927] | 0.391 |
| Tri-modal, from scratch | 0.858 [0.758, 0.929] | 0.448 | 0.815 [0.652, 0.924] | 0.432 |
| Frozen encoder | 0.812 [0.664, 0.922] | 0.410 | 0.844 [0.753, 0.917] | 0.391 |
| ABP only | 0.818 [0.720, 0.913] | 0.321 | 0.765 [0.642, 0.892] | 0.322 |
| ECG only | 0.789 [0.661, 0.903] | 0.288 | 0.818 [0.732, 0.902] | 0.308 |
| ABP+ECG | 0.786 [0.674, 0.894] | 0.358 | 0.869 [0.786, 0.933] | 0.345 |
| Pair ABP+ECG | 0.784 [0.685, 0.885] | 0.276 | 0.787 [0.671, 0.892] | 0.351 |
| Thesis fold models | 0.922 [0.851, 0.971] | 0.568 | not re-runnable | — |
