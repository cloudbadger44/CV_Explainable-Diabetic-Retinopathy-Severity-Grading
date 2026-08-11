# Study 2: Model Development and Fine-Tuning

*Architecture Comparison and Fine-Tuning Strategy for Explainable Diabetic Retinopathy Severity Grading on APTOS 2019*

**Peter Bello**
MAI204 Computer Vision · Milestone 3 · Seneca Polytechnic

---

## Abstract

Two decisions shape how well a diabetic retinopathy classifier performs before any class-imbalance fix or preprocessing tweak enters the picture: which backbone to use, and how much of it to unfreeze. This study settles both, under a protocol held fixed across six full training runs on APTOS 2019 fundus images.

First, three architectures were run head to head: a custom CNN trained from nothing, ResNet-50, and EfficientNet-B0, the latter two pretrained on ImageNet. EfficientNet-B0 won on validation QWK (0.8414 against 0.8310 for ResNet-50 and 0.7215 for the custom network) while carrying roughly a sixth of ResNet-50's parameters. With the architecture fixed, three fine-tuning depths were then compared: frozen backbone, partial unfreezing, full fine-tuning. Full fine-tuning pulled ahead clearly, QWK 0.8493 against 0.8224 for partial and 0.8027 for frozen, and reached its best epoch faster than either alternative.

The resulting configuration, EfficientNet-B0 fully fine-tuned, reached 79.5% validation accuracy and macro-F1 0.639. No DR and Moderate, the two best-represented grades, were classified reliably; Mild, Severe, and Proliferative were not, which tracks with how little training data each holds, since imbalance correction was deliberately left outside this study. The checkpoint, configuration file, and training recipe from this run are what carry forward as the architecture decision for the group's final pipeline.

## 1. Introduction

Diabetic retinopathy is one of the more preventable causes of blindness, provided it is caught early, which is where automated grading from fundus photographs can help, particularly where specialists are scarce. Group 5 split that problem into three parallel pieces: data and preprocessing (Ritika Lal), model development (this study, Peter Bello), and evaluation with explainability (Alain Dika). Keeping the pieces separable meant each study had to hold the other two variables still.

This study asks one question: given the project's shared preprocessing and training protocol, which architecture and which fine-tuning depth reach the best validation QWK on APTOS 2019? Answering it took two stages: an architecture comparison (custom CNN, ResNet-50, EfficientNet-B0), followed by a fine-tuning-depth comparison on whichever architecture won.

What sits outside this study matters as much as what's inside it. Preprocessing variants such as CLAHE and gamma correction belong to Ritika's work; weighted losses, oversampling, and Grad-CAM belong to Alain's. Because imbalance handling is out of scope here, the per-class recall numbers in Section 5 undersell the rarer grades. That's expected, not a flaw, and exactly what the parallel study exists to fix.

## 2. Related Work

Gulshan et al. [1] showed CNNs could grade DR at specialist level, though on datasets many times larger than APTOS's 3,662 images, which is the practical reason this study leans on transfer learning rather than training from scratch. Krause et al.'s [2] work on grader variability is part of why QWK, rather than accuracy, is the metric used here: it penalizes a distant misgrade more heavily than an adjacent one, matching how the five DR grades are actually ordered.

The two pretrained backbones represent different bets. ResNet-50's residual connections [3] made very deep networks trainable, and it remains a dependable baseline for medical-imaging transfer learning. EfficientNet [4] instead scales depth, width, and resolution together under one search-derived ratio, trading fewer parameters for comparable or better accuracy. EfficientNet-B0 was included specifically to test whether that trade holds on a dataset this small.

How much of a pretrained network to unfreeze has no universal answer; it depends on how close the source and target domains sit. Rather than assume full fine-tuning wins by default, Stage 2 treats that as an open question and tests it directly.

## 3. Data

APTOS 2019 [7] provides fundus photographs graded 0 through 4. The project fixes one stratified 70/15/15 split, seed 42, shared across every study to avoid leakage. This notebook's environment mounted a smaller mirror of the dataset, 2,930 of the full 3,662 images, so after filtering the shared split down to what was actually available, the working numbers were 2,051 training images, 448 validation, 431 test. The test set was never touched here; per the project's rule, it gets scored exactly once, later, in the final integrated notebook.

Every run, all three architectures and all three fine-tuning depths, used identical preprocessing: a retinal-field crop that removes the black borders common in APTOS photos, a resize to 224x224, and ImageNet normalization. No class weighting or oversampling was applied; the loss is unweighted cross-entropy, on purpose, since imbalance correction belongs to a different study.

## 4. Methods

Six runs, one shared protocol: Adam at lr 1e-3, weight decay 1e-4, ReduceLROnPlateau (patience 2, factor 0.1), batch size 32, up to 30 epochs, early stopping on validation QWK with patience 5.

| Component | Setting |
|---|---|
| Optimizer | Adam, lr = 1e-3, weight decay = 1e-4 |
| Schedule | ReduceLROnPlateau (patience 2, factor 0.1) |
| Batch / epochs | 32, up to 30, early stop patience 5 on QWK |
| Loss | Unweighted cross-entropy |
| Input | 224x224 RGB, shared crop + normalize pipeline |
| Selection | Validation QWK, macro-F1 as tiebreaker |

Stage 1 compared three architectures, each fully fine-tuned so the only variable was the architecture itself: a from-scratch CNN, ResNet-50, EfficientNet-B0. Stage 2 took EfficientNet-B0, the Stage 1 winner, and tested three fine-tuning depths: frozen backbone with only the classifier head trained, partial unfreezing of the head plus the last MBConv block, and full fine-tuning of every layer. Unfrozen backbone layers trained at a tenth of the head's learning rate, standard practice, fixed at that ratio rather than swept so the depth comparison stayed uncontaminated.

The winner was then reloaded and scored on the validation set with a full metrics suite plus a confusion matrix, to see not just how accurate it was but where it broke down.

## 5. Experiments and Results

### 5.1 Architecture comparison

| Architecture | Params | Best epoch | Val QWK | Val Macro-F1 | Val Acc. | Train time (s) |
|---|---|---|---|---|---|---|
| custom_cnn | 1.21M | 16 | 0.7215 | 0.3634 | 0.7165 | 3,338 |
| resnet50 | 23.5M | 4 | 0.8310 | 0.6119 | 0.8013 | 1,406 |
| **efficientnet_b0** | **4.0M** | **4** | **0.8414** | **0.6362** | **0.7924** | **1,440** |

*Table 2. Architecture comparison, validation split (n=448). EfficientNet-B0 wins on QWK.*

![Figure 1: Validation loss and QWK by architecture](images/fig1_architecture_curves.png)

*Figure 1. Validation loss (left) and QWK (right) by architecture.*

EfficientNet-B0 came out ahead on every axis that mattered: highest QWK, highest macro-F1, and a sixth of ResNet-50's parameter count. Both pretrained models converged within four epochs and then started overfitting, validation loss climbing while training QWK kept heading toward 1.0, which triggered early stopping well short of the 30-epoch ceiling. The custom CNN took sixteen epochs just to reach its best checkpoint and never got close to either pretrained model's QWK, about as direct a case for transfer learning as this dataset offers.

### 5.2 Fine-tuning depth

| Strategy | Trainable params | Best epoch | Val QWK | Val Macro-F1 | Val Acc. | Train time (s) |
|---|---|---|---|---|---|---|
| frozen | 6,405 | 18 | 0.8027 | 0.5079 | 0.7545 | 3,641 |
| partial | 418,565 | 10 | 0.8224 | 0.5342 | 0.7545 | 2,523 |
| **full** | **4,013,953** | **7** | **0.8493** | **0.6393** | **0.7946** | **2,157** |

*Table 3. Fine-tuning depth on EfficientNet-B0, validation split (n=448). Full fine-tuning wins.*

![Figure 2: Validation loss and QWK by fine-tuning depth](images/fig2_finetuning_curves.png)

*Figure 2. Validation loss (left) and QWK (right) by fine-tuning depth.*

Full fine-tuning won by a clear margin, a 4.7-point QWK gap over the frozen baseline, and got there fastest, best epoch 7 versus 10 for partial and 18 for frozen. Frozen's plateau makes the point plainly: ImageNet features alone, untouched, leave real accuracy on the table for fundus images. Partial fine-tuning helps but doesn't close the gap, so whatever adaptation this dataset needs isn't confined to the last block.

### 5.3 Selected model

| Metric | Score |
|---|---|
| Validation QWK | 0.8493 |
| Accuracy | 0.7946 |
| Macro-F1 | 0.6393 |
| Balanced accuracy | 0.6388 |

| Class | Recall |
|---|---|
| No DR | 0.978 |
| *Mild* | *0.488* |
| Moderate | 0.710 |
| *Severe* | *0.571* |
| *Proliferative* | *0.447* |

*Table 4-5. Selected-model metrics and per-class recall, validation split.*

![Figure 3: Confusion matrix](images/fig3_confusion_matrix.png)

*Figure 3. Confusion matrix, selected model.*

No DR is essentially solved (219 of 224 correct), and Moderate follows at 0.710, the two grades with the most training data. Mild, Severe, and Proliferative lag well behind, and the confusion matrix shows why: their errors aren't scattered, they cluster on neighboring, visually similar, equally under-represented grades. That's the fingerprint of an unweighted loss on an imbalanced dataset, and it's the exact gap the imbalance study downstream is meant to close.

![Figure 4: Training vs validation loss and QWK](images/fig4_selected_model_curves.png)

*Figure 4. Training vs. validation loss and QWK, selected model.*

Training and validation QWK track closely up to epoch 7, training at 0.92, validation at 0.849, then pull apart as training keeps climbing toward 0.98 while validation flattens out. Early stopping caught it at the right point.

## 6. Conclusion

EfficientNet-B0, fully fine-tuned, is the architecture and depth this study hands off to the rest of the group. It beat ResNet-50 with a sixth of the parameters, and beat both shallower fine-tuning strategies while training fastest: best validation QWK 0.8493, accuracy 0.7946, macro-F1 0.6393. No DR and Moderate are reliable; Mild, Severe, and Proliferative are not, for reasons already covered.

A few things this study didn't do: no test-set numbers, since that split stays locked until final integration; no preprocessing search, that's Ritika's territory; no imbalance correction, that's Alain's; and no sweep of the 1:10 discriminative learning-rate ratio, fixed going in to keep the fine-tuning comparison clean. This notebook's dataset mirror also held 732 fewer images than the project's canonical split, so the architecture conclusions weren't re-verified against the full dataset.

Three things carry forward: the configuration file, the checkpoint, and the shared model-construction code. Combined with Ritika's preprocessing choice and Alain's imbalance strategy, they fix the architecture half of the group's final model.

## References

[1] V. Gulshan et al., "Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs," JAMA, vol. 316, no. 22, pp. 2402-2410, 2016.

[2] J. Krause et al., "Grader variability and the importance of reference standards for evaluating machine learning models for diabetic retinopathy," Ophthalmology, vol. 125, no. 8, pp. 1264-1272, 2018.

[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE CVPR, 2016, pp. 770-778.

[4] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in Proc. ICML, 2019, pp. 6105-6114.

[5] J. Cohen, "Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial credit," Psychological Bulletin, vol. 70, no. 4, pp. 213-220, 1968.

[6] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in Advances in NeurIPS, 2019, pp. 8024-8035.

[7] Asia Pacific Tele-Ophthalmology Society, "APTOS 2019 Blindness Detection," Kaggle, 2019.
