# Study 2: Model Development and Fine-Tuning

Architecture comparison and fine-tuning strategy for explainable diabetic retinopathy severity grading on APTOS 2019.

Individual study by **Peter Bello**, part of the Group 5 DR grading pipeline (MAI204 Computer Vision, Milestone 3). Two other studies run in parallel: data and preprocessing (Ritika Lal), evaluation and explainability (Alain Dika). This repo covers the model side only: which backbone to use, and how much of it to fine-tune.

## Summary

Six full training runs, one shared protocol, two questions:

1. **Which architecture?** Custom CNN vs. ResNet-50 vs. EfficientNet-B0, all fully fine-tuned, everything else held fixed.
2. **How much of it should be trainable?** Frozen backbone vs. partial unfreezing vs. full fine-tuning, on the Stage 1 winner.

EfficientNet-B0 won Stage 1 (val QWK 0.8414) with a sixth of ResNet-50's parameters. Full fine-tuning won Stage 2 (val QWK 0.8493), beating the frozen baseline by 4.7 points. **Selected configuration: EfficientNet-B0, fully fine-tuned.**

## Results

### Stage 1, architecture comparison (validation split, n=448)

| Architecture | Params | Best epoch | Val QWK | Val Macro-F1 | Val Acc. |
|---|---|---|---|---|---|
| custom_cnn | 1.21M | 16 | 0.7215 | 0.3634 | 0.7165 |
| resnet50 | 23.5M | 4 | 0.8310 | 0.6119 | 0.8013 |
| **efficientnet_b0** | **4.0M** | **4** | **0.8414** | **0.6362** | **0.7924** |

### Stage 2, fine-tuning depth on EfficientNet-B0 (validation split, n=448)

| Strategy | Trainable params | Best epoch | Val QWK | Val Macro-F1 | Val Acc. |
|---|---|---|---|---|---|
| frozen | 6,405 | 18 | 0.8027 | 0.5079 | 0.7545 |
| partial | 418,565 | 10 | 0.8224 | 0.5342 | 0.7545 |
| **full** | **4,013,953** | **7** | **0.8493** | **0.6393** | **0.7946** |

### Selected model, per-class recall

| Class | Recall |
|---|---|
| No DR | 0.978 |
| Mild | 0.488 |
| Moderate | 0.710 |
| Severe | 0.571 |
| Proliferative | 0.447 |

No DR and Moderate, the two best-represented grades, are classified reliably. Mild, Severe, and Proliferative lag behind, which is expected: this study runs an unweighted loss on purpose, since class-imbalance correction is owned by a parallel study. Full metrics, confusion matrix, and training curves are in the [report](Study2_Report_Peter_Bello.pdf).

## Repo contents

```
.
├── model_study_2.ipynb           Full notebook: Stage 1 + Stage 2 + selected-model analysis
├── Study2_Report_Peter_Bello.md  Write-up, same content as the PDF/docx below
├── Study2_Report_Peter_Bello.pdf Formatted report
├── images/                       Figures referenced by the markdown report
│   ├── fig1_architecture_curves.png
│   ├── fig2_finetuning_curves.png
│   ├── fig3_confusion_matrix.png
│   └── fig4_selected_model_curves.png
├── configs/
│   └── selected_model.yaml       Architecture, freeze mode, training recipe, validation metrics
└── artifacts/
    ├── checkpoints/               Saved model weights per run
    └── results/                   architecture_comparison.csv, finetuning_comparison.csv
```

## Reproducing

The notebook expects the shared project split (`aptos_seed42.csv`, seed 42, 70/15/15) and the shared preprocessing module (`training_utils.py`, `models.py`) from the group repo. On a fresh Kaggle or Colab GPU session:

```bash
pip install -q pyyaml
```

Then run the notebook top to bottom. It downloads ImageNet weights for ResNet-50 and EfficientNet-B0 on first use, trains all six configurations in sequence, and writes results to `artifacts/`. A single run through both stages takes roughly 2.5 hours on a T4.

**Note on dataset size.** This notebook was run against a Kaggle mirror of APTOS 2019 containing 2,930 of the full 3,662 images. After filtering the shared split to what was available, the working numbers were 2,051 train / 448 validation / 431 test, smaller than the project's canonical 2,563/549/550 split used in the final integrated pipeline. Architecture and fine-tuning conclusions are expected to transfer but were not re-verified on the full dataset here.

## Shared training protocol

All six runs use identical settings so that each stage isolates exactly one variable:

| Component | Setting |
|---|---|
| Optimizer | Adam, lr = 1e-3, weight decay = 1e-4 |
| Schedule | ReduceLROnPlateau (patience 2, factor 0.1) |
| Batch size | 32 |
| Max epochs | 30, early stop patience 5 on validation QWK |
| Loss | Unweighted cross-entropy |
| Input | 224x224 RGB, retinal-field crop + ImageNet normalize |
| Discriminative LR | Unfrozen backbone layers train at 0.1x the head's rate |
| Selection metric | Validation QWK (primary), macro-F1 (tiebreaker) |

## Scope

**In scope:** architecture selection, fine-tuning depth, model-side hyperparameters.

**Out of scope**, owned by other studies in the group project:
- CLAHE, gamma correction, and other preprocessing variants (Ritika Lal)
- Weighted losses, oversampling, class-targeted augmentation, Grad-CAM (Alain Dika)
- Test-set evaluation, held out until the final integrated notebook

## Handoff artifacts

Three files feed into the group's final integrated pipeline:

- `configs/selected_model.yaml`, architecture, freeze mode, and full training recipe
- `artifacts/checkpoints/stage2_efficientnet_b0_full.pth`, the selected model weights
- `models.py`, shared model-construction code

## References

1. V. Gulshan et al., "Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs," *JAMA*, 2016.
2. J. Krause et al., "Grader variability and the importance of reference standards for evaluating machine learning models for diabetic retinopathy," *Ophthalmology*, 2018.
3. K. He et al., "Deep residual learning for image recognition," *CVPR*, 2016.
4. M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," *ICML*, 2019.
5. J. Cohen, "Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial credit," *Psychological Bulletin*, 1968.
6. Asia Pacific Tele-Ophthalmology Society, "APTOS 2019 Blindness Detection," Kaggle, 2019.

## Author

Peter Bello, Group 5, MAI204 Computer Vision, Seneca Polytechnic.
