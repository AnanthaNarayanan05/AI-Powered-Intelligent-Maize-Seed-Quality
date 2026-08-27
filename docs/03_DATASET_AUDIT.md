# Dataset Audit (Phases 3-5)

All numbers below were computed directly from the files, not estimated.

## Dataset A — `Corn_3_Classes_Image_Dataset` (from `seed_dataset1.zip`)
- **Location (target):** `datasets/dataset_variety_a_corntype/`
- **License/provenance:** academic, citation-required (3 papers listed in
  `Explanation_Citation_Request.txt`), no explicit open license text given.
- **Total images:** 1050 (verified count)
- **Classes (3, exact):** `Zea_mays_Chulpi_Cancha` (350), `Zea_mays_Indurata` (350),
  `Zea_mays_Rugosa` (350) — perfectly balanced, 33.3% each.
- **Format/resolution:** JPG, uniform 400x400px, file sizes 5.5–13.1 KB (very
  compressed/simple images — likely single-seed crops on plain background).
- **Corrupted files:** 1 unreadable file found (the citation `.txt` mis-scanned as an
  image by the initial pass; 0 corrupted JPGs on re-check).
- **Duplicates:** 4 exact-duplicate pairs (by MD5), all within-class — no cross-class
  leakage risk, but must dedupe before train/val/test split to avoid leakage across
  splits.
- **Annotation format:** folder-per-class only. No bounding boxes, no masks, no
  severity/defect labels.
- **Splits:** none provided — must be created (recommend stratified 70/15/15).
- **Limitations:** small dataset (1050 images), single fixed low resolution, no
  metadata, no defect/quality signal of any kind, provenance tied to a specific
  published study (must cite per its terms in the final report).

## Dataset B — `MaizeData` (from `seed_dataset2.zip`)
- **Location (target):** `datasets/dataset_2_variety/`
- **Total images:** 17,724 (verified exact count, matches master prompt's Dataset 2
  description)
- **Classes (3, exact):** `Bhihilifa` (6480, 36.6%), `WangDataa` (6144, 34.7%),
  `SanzalSima` (5100, 28.8%) — mild imbalance (≈1.27:1 max:min ratio), manageable with
  class-weighted loss, no resampling required.
- **Format/resolution:** JPG, uniform 224x224px.
- **Corrupted files:** 0 found in a 150-image stratified sample (50/class); full-corpus
  verification will run as part of the Phase-3 preprocessing script before training.
- **Duplicates:** 11 exact-duplicate pairs (by MD5) found via full-corpus hash pass,
  all within-class.
- **Annotation format:** folder-per-class only, no accompanying README/metadata file
  in the archive.
- **Splits:** none provided — must be created (recommend stratified 70/15/15,
  deduping first).
- **Limitations:** no documented provenance/citation for this file (unlike Dataset A);
  no metadata beyond folder name; uniform 224x224 suggests pre-cropped single-seed
  images, consistent with Dataset A's format but not confirmed identical acquisition
  pipeline — cross-dataset merging is NOT assumed safe (see below).

## Dataset C — Detection dataset (`Grain and Objects Detection.v1i.yolov11.zip`)
- **Location (target):** `datasets/dataset_3_detection/`
- **Source:** Roboflow, "grain-and-objects-detection" project v1, CC BY 4.0.
- **Format:** YOLOv11 `.txt` labels, `data.yaml` present and consistent
  (`nc: 1`, `names: ['Corn']`).
- **Splits (pre-existing, preserved as-is):** train 913 images / valid 256 / test 82
  images = 1251 total. Every image has a matching, non-empty label file (0 missing,
  0 empty).
- **Instances:** 42,602 total bounding boxes. Average 34.05 objects/image (multi-seed
  images), min 1, max 125 objects in a single image.
- **Box validity:** 0 invalid boxes — all normalized coordinates in (0,1], all class
  ids = 0.
- **Preprocessing already applied by Roboflow:** auto-orientation (EXIF-stripped),
  resize to 640x640 (stretch). No augmentation baked in.
- **Classes:** exactly 1 — `Corn`. No sub-classes for healthy/defective/foreign-object.
  Some source filenames (visible in the raw filenames) reference words like
  "fusarium" or "dal," but these are just names of the original photo batches — they
  are **not** encoded as annotation classes, so they confer no defect-detection
  capability whatsoever.
- **Limitations:** single-class only — supports localization/counting/cropping, not
  any defect or quality categorization.

## Cross-dataset compatibility analysis (Phase 5)

| Pair | Compatible for... | NOT compatible for... | Reasoning |
|---|---|---|---|
| A ↔ B | Nothing directly merged | Blind merging as one variety classifier | Different class taxonomies (Chulpi Cancha/Indurata/Rugosa vs Bhihilifa/SanzalSima/WangDataa are different corn types from different studies/acquisitions), different resolutions (400x400 vs 224x224), different backgrounds/lighting pipelines (unconfirmed but visually distinct file-size profiles). Treating them as one 6-class problem would silently assume they're drawn from a comparable domain, which is not verified. |
| A, B → contrastive pretraining | Yes, separately or as unlabeled pretraining corpora (SimCLR does not need consistent class taxonomy) | Do not use both as one labeled fine-tuning target | Contrastive pretraining only needs images, not a shared label space, so both datasets can jointly enrich the encoder's visual prior; the downstream classification head is still trained and evaluated separately per dataset. |
| A or B ↔ C | Yes, sequential pipeline (C detects/crops → A/B model classifies the crop) | Not a joint end-to-end trained model | C provides bounding boxes on raw multi-seed images; A/B provide single, pre-cropped seed images at fixed resolution. Feeding C's cropped seeds into an A/B-trained classifier is a valid **inference-time pipeline composition**, not a claim that the datasets share a training distribution. This will need visual validation once training crops are generated (Phase 13/14), since C's crops come from real field/tray photos while A/B are pre-cropped and possibly cleaner. |

**Decision:** Dataset A and Dataset B are each trained as their own independent
variety classifier (two separate experiments/checkpoints), both benefiting from
contrastive pretraining + cognitive attention per the proposed architecture. Dataset C
trains one detection model. The unified inference pipeline chains detection (C) →
crop → variety classification (A-model or B-model, selectable) → embedding →
similarity search. No blind merging occurs anywhere.

## Amendment (validated by src/data/validate_detection_dataset.py, run on real data)
- Full validator run confirms manual audit: 42,602 total instances, splits 913/256/82, 0 missing/empty labels.
- Found 1 label line (in `valid/labels/DSC_0069_JPG.rf.c01fe06d42740f8afa1ad84db1fcf8bc.txt`) that uses YOLO **segmentation polygon** format (class + variable-length x,y point list) instead of standard bbox format (class + 4 floats). This single stray line is excluded/skipped by the training dataloader (see `src/training/train_detection.py`) rather than silently mis-parsed as a box.

---

## Dataset 4 — `EfficientMaize` (Mendeley `doi:10.17632/r6vvm5jkh6`)

Added August 26, 2026. This is the first dataset in the project carrying **real quality labels**, and it is the reason the synthetic defect classifier could be retired.

The raw archive (`EfficientMaize.rar`, 4,846 images) was used rather than the pre-augmented release (`Augmented EfficientMaize.rar`, ~29k images). That was a deliberate choice: the augmented release contains many derived copies of each source image, and splitting over those files would have reproduced exactly the leakage documented for Dataset B below. Augmentation belongs in the training loop, where it cannot straddle a split boundary.

| Property | Value |
|---|---|
| Images | 4,846 |
| Classes | `Good` 2,635 / `Bad` 2,211 |
| Corrupt or unreadable | 0 |
| Colour mode | 4,846/4,846 RGB |
| Typical resolution | ~62–68 px — already single-kernel crops |
| Exact duplicates | 240 groups, 251 redundant copies (5.2%), 238 of them in `Bad` |
| Duplicates straddling Good/Bad | 0 — no label conflicts |
| Near-duplicate pairs (dHash ≤ 5/64) | 494 |
| …of which span Good/Bad | 33 — retained, never relabelled |
| Byte-overlap vs A / B / C | **0 / 0 / 0** across all 20,025 existing images |

Three consequences for how it is used:

1. **Duplicates are contained, not deleted.** The 251 exact copies and 494 near-duplicate pairs are grouped by content cluster so that every copy of an image lands in the same split. No data is discarded and no image can appear in both train and test.
2. **The 33 label-conflicting near-duplicate pairs are left exactly as the authors labelled them.** Near-identical kernels with opposite grades are either genuine borderline cases or label noise; deciding which would require re-annotating someone else's expert labels, which this project does not do. Group-aware splitting keeps each conflicting pair intact within one split, so they cannot leak.
3. **The ~64 px resolution is an advantage, not a defect.** These are already single-kernel crops, which matches the serving path (YOLO crop → classify) far better than Datasets A and B, whose whole-image framing caused the 22-point train/serve gap documented in `09_PROJECT_STATUS_REPORT.md` §6.6.

Split (near-duplicate-clustered, 4,416 groups): train 3,401 / val 719 / test 726.

---

## Amendment (August 26, 2026) — split integrity across all datasets

A defect was found in how Dataset B was split that invalidated its published results. It is recorded here because it is a property of the *dataset*, not of the training code.

**Dataset B's 17,713 files are derived from 127 physical seeds.** The filenames say so: `aug_4_1632053314_Bihilifa40.jpg` and `77542039_Bihilifa30.jpg` both carry a trailing source token. Collapsing on it:

```
17,713 image files  ->  127 DISTINCT SOURCE SEEDS
copies per source: max 294, median 132, min 66
per class: Bhihilifa 46, SanzalSima 33, WangDataa 48
sources whose copies appear in >1 split: 127 / 127 = 100.0%
TEST images whose source ALSO appears in TRAIN: 2655 / 2655 = 100.0%
```

The original split was performed over files, and the files were not independent samples. Full analysis and the corrected results are in `09_PROJECT_STATUS_REPORT.md` §6.9.

**Dataset A was checked for the same defect and is clean.** 1,046 byte-distinct images, no augmentation naming, and zero test images with a ≥0.99-correlation twin in training. An intermediate measurement using a 64-bit perceptual hash suggested 83% near-duplication; that figure was an artefact of insufficient hash resolution on a homogeneous dataset and is retracted. Against the dataset's own similarity distribution — mean pairwise similarity 0.697, 95th percentile 0.925 — a 0.95 threshold selects the top 5% of *all* pairs rather than duplicates.

**The invariant is now enforced by tests.** `tests/test_unified_model.py::test_no_group_spans_more_than_one_split` asserts it for every grouped manifest, so a regenerated manifest cannot silently reintroduce the problem.
