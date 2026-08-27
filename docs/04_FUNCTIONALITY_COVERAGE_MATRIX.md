# Functionality Coverage Matrix (Phase 6)

Statuses used: SUPPORTED / CONDITIONALLY SUPPORTED / NOT SUPPORTED / REQUIRES
ADDITIONAL DATA / REQUIRES CUSTOM ANNOTATION.

| # | Functionality | Status | Supporting dataset | Exact supporting label/annotation | Implementation method | Limitations |
|---|---|---|---|---|---|---|
| 1 | Seed classification (generic) | SUPPORTED | A, B, 4 | Folder-per-class labels | Unified two-head model | Single model since Aug 26; see 09 section 6.10 |
| 2 | Variety classification | SUPPORTED | A, B | 6 classes in one head | EfficientNet-B0 + cognitive attention | 88.95% acc / 93.16% macro-F1 on the honest split. SanzalSima recall 60.6% is the weak class |
| 3 | Quality classification (Good/Bad) | SUPPORTED | 4 | Expert-assigned Good/Bad kernel labels | Second head on the unified model | 97.25% acc. Validated only on kernel close-ups; out-of-distribution inputs are flagged, not graded |
| 4 | Healthy vs defective classification | SUPPORTED | 4 | as above | as above | Bad-class precision 99.1%, recall 94.9% — roughly 1 defective kernel in 20 is missed |
| 5 | Defect type classification | NOT SUPPORTED | — | Dataset 4 is binary Good/Bad only | — | A binary grade is not a defect taxonomy |
| 6 | Crack classification | NOT SUPPORTED | — | none exist | — | — |
| 7 | Fungal classification | NOT SUPPORTED | — | none exist | — | Filenames in C mention "fusarium" but it is not an annotated class. Published kernel-level fungal detection relies on NIR/hyperspectral bands (715 nm, 965 nm) an RGB camera cannot observe |
| 8 | Discoloration classification | NOT SUPPORTED | — | none exist | — | — |
| 9 | Contrastive learning | SUPPORTED | A, B (unlabeled use) | raw images only, no labels needed | SimCLR-style two-view augmentation + NT-Xent loss | Literature precedent: survey row 45 (SimCLR/NNCLR on maize kernels) |
| 10 | Self-supervised pretraining | SUPPORTED | A, B | same as above | same encoder pretraining, then fine-tune head | — |
| 11 | Cognitive attention (general) | SUPPORTED | A, B | N/A — architectural | CBAM-style channel+spatial attention module | Ablation-tested (Phase 7 experiments) |
| 12 | Channel attention | SUPPORTED | A, B | N/A | Squeeze-and-Excitation block | — |
| 13 | Spatial attention | SUPPORTED | A, B | N/A | CBAM spatial gate | — |
| 14 | Multi-scale attention | CONDITIONALLY SUPPORTED | A, B | N/A | Multi-scale feature fusion across backbone stages | Feasible but adds training/compute cost; scoped to time/GPU budget |
| 15 | Feature embeddings | SUPPORTED | A, B | Encoder output vector | Post-classifier penultimate layer or projection head | — |
| 16 | Similarity search | SUPPORTED | A, B | Embeddings above | FAISS index, cosine/L2 | Must be labeled "visual/feature similarity," not certification |
| 17 | Explainable AI | SUPPORTED | A, B | N/A | Grad-CAM on classifier | — |
| 18 | Grad-CAM | SUPPORTED | A, B | N/A | pytorch-grad-cam or custom hook | Must never be called a segmentation mask |
| 19 | Attention visualization | CONDITIONALLY SUPPORTED | A, B | N/A | Visualize CBAM attention maps | Valid only where attention module is actually used in the active model variant |
| 20 | Object detection | SUPPORTED | C | 42,602 verified bounding boxes | YOLOv8/v11 (Ultralytics), 1 class | Detects "Corn" only |
| 21 | Bounding boxes | SUPPORTED | C | as above | — | — |
| 22 | Seed localization | SUPPORTED | C | as above | — | — |
| 23 | Multiple seed detection | SUPPORTED | C | avg 34 objects/image, max 125 | — | — |
| 24 | Seed counting | SUPPORTED | C | count of predicted boxes above confidence threshold | — | Counting accuracy bounded by detector recall/precision |
| 25 | Batch analysis | SUPPORTED | A, B, C (application-level) | N/A | Backend loop over uploaded images | Upload-based, not seed-lot metadata |
| 26 | Defect localization | NOT SUPPORTED | — | none exist | — | Requires labeled defect boxes/masks |
| 27 | Defect severity classification | NOT SUPPORTED | — | no ordinal labels; no defect masks | — | The quality head outputs confidence, not severity. A kernel graded Bad at 0.95 is more *recognisably* damaged than one at 0.70, not more damaged. Requires ordinal labels or defect-area masks |
| 28 | Pixel-level seed segmentation | NOT SUPPORTED | — | no masks in any dataset | — | Only bounding boxes exist (C) |
| 29 | Pixel-level defect segmentation | NOT SUPPORTED | — | no masks | — | — |
| 30 | Exact defect area calculation | NOT SUPPORTED | — | no masks | — | Cannot compute area without segmentation |
| 31 | Foreign-object detection | NOT SUPPORTED | — | C has only 1 class | — | Would need annotated non-kernel classes (chaff, stones, cob). Open-set rejection via embedding distance could flag *unrecognised* objects but cannot name them |
| 32 | Purity estimation | CONDITIONALLY SUPPORTED | A, B, 4 (application-level) | Aggregated per-seed predictions | /api/lot/report — variety composition, off-type rate, soundness | Computed over predictions, not ground truth. Explicitly NOT a certification, which requires an accredited lab and a prescribed sampling protocol |
| 33 | True seed-lot metadata analysis | NOT SUPPORTED | — | none provided | — | Only image-level data exists. Physical purity by weight, moisture and germination cannot be derived from photographs |
| 34 | Application-generated analysis history | SUPPORTED | N/A (app-level) | N/A | SQLite persistence layer | — |

## Net effect on scope

**Revised August 26, 2026.** The original matrix marked every quality and defect capability (items 3-8, 26-32) NOT SUPPORTED, on the correct finding that Datasets A, B and C carry no quality labels. That finding still holds for those three datasets. The addition of Dataset 4 (Mendeley EfficientMaize), which carries real expert-assigned Good/Bad kernel labels with zero byte-overlap against the existing data, moves items 3 and 4 to SUPPORTED and item 32 to CONDITIONALLY SUPPORTED.

What has **not** changed: items 5-8 and 26-31 remain NOT SUPPORTED, because a binary Good/Bad grade is not a defect taxonomy, a confidence is not a severity, and no dataset in this project contains pixel masks, ordinal severity labels, or non-kernel object classes. These are listed as unsupported in the application UI rather than approximated.

The synthetic defect classifier described in `06_SYNTHETIC_DEFECT_POLICY.md` is **superseded** by the real quality head and is off by default in every served path. It is retained in the repository as the documented record of how the capability gap was handled before real labels were available.

One further limit applies specifically to the new capability: the quality head is validated on kernel close-ups resembling its training distribution and extrapolates confidently outside it, calling 71% of Dataset A defective at 89% mean confidence. The platform gates on representation distance rather than confidence and marks such grades unverified. See `09_PROJECT_STATUS_REPORT.md` sections 4.3 and 6.11.
