# Literature Survey Analysis (Phase 2)

Source: `Literarture Survey - Sheet1.pdf`, 48 papers, compiled by Delna, Anantha, Ashik.
All claims below are directly attributable to specific rows in that survey (cited by
title). Nothing here is invented.

## 1. Research domain
Non-destructive, image/spectral-based automated seed quality, defect and variety
classification for maize (and related grains), using classical ML, CNNs, attention
mechanisms, transformers, contrastive/self-supervised learning, and object detection.

## 2. Core problem (as framed across the survey)
Manual visual seed inspection and grading is slow, subjective, labour-intensive and
error-prone (repeated across rows 1, 2, 12, 15, 16, 20, 36). Traditional handcrafted
feature methods (color/shape/texture) don't scale or generalize well (rows 16, 33, 37).

## 3. Existing approaches identified in the survey
- **Spectral/imaging modalities:** FT-NIR spectroscopy, X-ray/CT imaging, thermal
  imaging, hyperspectral imaging (VIS-NIR), multispectral imaging, RGB machine vision
  (rows 1, 4, 5, 6, 9, 10, 13, 17, 24, 25, 32, 38, 41).
- **Classical ML:** SVM, Random Forest, KNN, Naive Bayes, Decision Tree, LSSVM,
  Logistic Regression (rows 6, 7, 8, 20, 21, 22, 30).
- **Deep learning backbones:** CNN, GoogLeNet, ResNet-50, VGG16/19, InceptionV3,
  MobileNetV2/V3, AlexNet, EfficientNet (rows 2, 4, 9, 16, 18, 22, 27, 36).
- **Attention mechanisms:** CBAM (channel + spatial attention) applied to
  MobileNetV3-Large (row 43) and MobileNetV2 (row 46, "I_CBAM" parallel variant),
  Slot Attention (row 47), Vision Transformer self-attention (SeedViT, row 42), dual
  attention with transformers (row 48).
- **Contrastive / self-supervised learning:** row 45 explicitly uses SimCLR and NNCLR
  to pretrain on unlabeled maize kernel images, fine-tuning for classification and
  embryo segmentation — the closest prior-art precedent for this project's contrastive
  pretraining plan. It reports SSL-pretrained segmentation beating ImageNet-pretrained
  and random-init baselines, and competitive results from as little as 1% labeled data.
- **Object detection:** YOLOv5/YOLOv8 for maize seed defect identification (row 23),
  Fast R-CNN for seed classification/quality testing (row 12).
- **Explainability:** Grad-CAM / feature maps used in rows 9, 16, 42, 46, 47 to
  visualize discriminative regions — consistently described as an interpretability aid,
  never as a segmentation mask.

## 4. Limitations of existing approaches (as stated in the survey)
- Small/narrow datasets, single cultivar or single growing condition (rows 35, 41, 42,
  43, 46, 48).
- Poor generalization test accuracy despite high training accuracy (row 14: 97% train
  vs 64% test).
- Accuracy drops sharply for defective/abnormal classes vs normal ones (row 37: 95.6%
  normal vs 80.6% defective).
- Most defect-classification work evaluates only a handful of defect categories under
  controlled lab imaging (rows 41, 43) and does not test cross-variety or
  cross-environment robustness.
- Row 31 (systematic review of 30 papers) flags that automated **size-based** grain
  quality grading remains underserved despite its practical importance.
- Row 45 explicitly calls out that self-supervised contrastive pretraining has *not*
  yet been combined with seed quality evaluation or defect detection tasks — stated
  directly as future work.

## 5. Identified research gaps (directly stated in the survey, not inferred)
1. Contrastive/self-supervised pretraining has been validated for maize kernel
   classification/segmentation (row 45) but not combined with attention-refined
   feature learning for downstream variety recognition at scale.
2. Attention modules (CBAM, Slot Attention, ViT self-attention) have each been applied
   separately to maize seed classification (rows 42, 43, 46, 47, 48), but no row
   combines contrastive pretraining *and* multi-scale cognitive attention in one
   pipeline.
3. Several papers (41, 43, 45, 46, 48) explicitly list "integrating advanced
   self-supervised, transformer-based, or multimodal attention architectures" as
   future work.
4. Grad-CAM/attention visualization is consistently used for interpretability but
   never conflated with true segmentation in the well-conducted studies — reinforcing
   this project's own rule to keep that distinction explicit.

## 6. This project's honest position relative to the survey
- **Directly supported by the survey:** attention-augmented CNN classification for
  maize seed variety (rows 42, 43, 46, 47, 48) and contrastive/self-supervised
  pretraining for maize kernel representation learning (row 45) are both established,
  working techniques with reported numbers this project can compare against.
- **Engineering judgement (this project's design choice, not a literature claim):**
  using SimCLR-style contrastive pretraining specifically (row 45 used SimCLR/NNCLR,
  giving direct precedent) combined with a CBAM-style channel+spatial attention module
  (rows 43/46 precedent) on top of an EfficientNet/ResNet backbone, sized for a single
  RTX 4060 laptop GPU.
- **New contribution of this project (not claimed as already proven in the
  literature):** combining contrastive pretraining + cognitive (channel+spatial,
  multi-scale) attention in one pipeline for maize variety recognition across *two
  independent variety datasets*, paired with a real single-class YOLO detection stage
  and a Gemini-based natural-language explanation layer over verified model outputs.
  This composition is motivated by gap #2/#3 above but its performance is an
  experimental question this project answers empirically (Phase 7 ablation:
  baseline vs +attention vs +contrastive vs +contrastive+attention), not a claim
  copied from any single paper.
- **What the survey does NOT license this project to claim:** none of the datasets in
  hand contain defect, disease, crack, mold, or foreign-object labels, so despite the
  survey being full of defect-classification precedent (rows 16, 23, 37, 41, 43),
  this project cannot train or claim a defect/quality model without a genuinely
  labeled defect dataset (see `01_RESOURCE_INVENTORY.md`).
