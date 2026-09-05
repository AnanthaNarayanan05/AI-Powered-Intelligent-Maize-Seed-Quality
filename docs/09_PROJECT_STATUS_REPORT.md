# Project Status Report

**Project:** AI-Powered Intelligent Maize Seed Quality, Defect, Variety Recognition and Detection System Using Contrastive Learning and Cognitive Attention
**Report date:** August 26, 2026 (supersedes the August 24 post-training status) — **updated September 4, 2026, see §10**
**Prepared for:** Ananthu
**Location:** `C:\COLLEGE\Project\FINALYEARPROJECT` (local RTX 4060 machine)

---

## 1. Executive Summary

**The project is complete, and its central finding has changed.** The August 24 version of this report led with a negative result: that neither contrastive pretraining nor cognitive attention produced a measurable benefit. That conclusion has been withdrawn. It was an artefact of a data split that made the evaluation task memorisable. On a corrected, group-aware split the proposed architecture produces a **statistically significant improvement** (§6.9).

`pytest tests/` reports **40 passed, 0 failed**, up from 29, with eleven new tests covering the unified model, the distribution gate, and the split-integrity invariant whose violation caused the original error.

### 1.1 The leakage finding

Dataset B ships 17,713 image files. Their filenames encode their own provenance — `aug_4_1632053314_Bihilifa40.jpg` and `77542039_Bihilifa30.jpg` both derive from a named source seed. Collapsing on that token:

```
Dataset B: 17,713 image files  ->  127 DISTINCT SOURCE SEEDS
copies per source: max 294, median 132, min 66
sources whose copies appear in >1 split: 127 / 127 = 100.0%
TEST images whose source ALSO appears in TRAIN: 2655 / 2655 = 100.0%
```

Every test image was a rotated or flipped copy of an image the model had trained on. The previously reported 99.89–99.96% accuracies measured memorisation, not generalisation — and, critically, they explain why the ablations were indistinguishable. A model that has already seen all 127 seeds scores ~99.9% whether or not it has attention. There was no headroom in which an architectural contribution *could* have appeared.

Dataset A was checked for the same defect and **is clean**: 1,046 byte-distinct images, no augmentation naming, and zero test images with a ≥0.99-correlation twin in training. An earlier draft of this section reported an 83% near-duplicate rate for Dataset A; that figure came from a 64-bit perceptual hash lacking the resolution to separate distinct seeds in a homogeneous dataset, and is retracted. Dataset A's results stand as originally reported.

### 1.2 The corrected result

The four ablations were re-run on a split where each source seed lives in exactly one split (89 train / 19 val / 19 test), against a contrastive encoder pretrained on the training split alone — the original encoder had been pretrained on every image including the test set, compromising the contrastive ablations a second and independent way.

```
experiment          ORIGINAL (leaked)  GROUPED (honest)     drop
baseline                       99.89%            85.35%   14.54pp
attention_only                 99.89%            85.31%   14.57pp
contrastive_only               99.96%            89.06%   10.90pp
full                           99.92%            87.97%   11.95pp

spread across the four ablations:  0.0753 pp  ->  3.7467 pp
```

Because 2,669 test images represent only 19 physical seeds, significance was assessed per seed, not per image, using a paired Wilcoxon signed-rank test with Holm correction for six comparisons:

| comparison | Δ | seeds better/worse/tied | p | verdict |
|---|---|---|---|---|
| attention_only vs full | +2.63pp | 12 / 2 / 5 | 0.0052 | **significant** |
| baseline vs full | +3.13pp | 9 / 4 / 6 | 0.0277 | not significant after correction |
| baseline vs contrastive_only | +3.50pp | 8 / 5 / 6 | 0.3824 | not significant |
| baseline vs attention_only | +0.50pp | 7 / 7 / 5 | 0.6832 | not significant |

**What can be claimed:** the full proposed model significantly outperforms its attention-only ablation. **Attention alone contributes nothing** (7 seeds better, 7 worse — noise). Contrastive pretraining is the component carrying the effect, though `contrastive_only` has the highest *mean* while failing its own significance test, because its wins are inconsistent across seeds. Mean ranking and significance disagreeing is the signature of a 19-seed test set, and the result should be reported with that limitation stated rather than as a clean win. Reproduce with `python -m src.analysis.ablation_significance`.

### 1.3 One model, and real quality labels

The two per-dataset variety models and the synthetic defect classifier are superseded by a **single unified model**: one EfficientNet-B0 trunk with cognitive attention, a 6-class variety head, and a 2-class quality head, trained on all 23,605 images at once with the loss masked per head so a head only learns from images that actually carry its label. One checkpoint, one forward pass, both predictions.

```
=== VARIETY HEAD ===  n=2824  acc=88.95%  f1_macro=93.16%
   Zea_mays_Chulpi_Cancha     100.0%   Bhihilifa    99.4%
   Zea_mays_Indurata          100.0%   SanzalSima   74.0%   <- see below
   Zea_mays_Rugosa            100.0%   WangDataa    85.6%

=== QUALITY HEAD ===  n=726  acc=97.25%  f1_macro=97.21%
   Good  97.5%      Bad  96.9%  (precision 99.1%)
```

The quality head is trained on the Mendeley *EfficientMaize* dataset's **real, expert-assigned Good/Bad kernel labels** — audited at 4,846 images, 0 corrupt, 251 exact duplicates contained by group-aware splitting, and **zero byte-overlap with Datasets A, B, or C** across all 20,025 existing images. This is the first genuine quality capability in the project, and it retires the synthetic defect classifier, now off by default.

**The honest weak spot is SanzalSima**, at 60.6% recall — 276 of 701 test images misread as WangDataa. This was invisible under the leaked split, where SanzalSima scored ~99.9% because the model had memorised all 33 of its source seeds. It is the hardest genuine confusion in the data and the clearest avenue for future work.

### 1.4 The distribution gate

The quality head scores 97.25% on data resembling its training set, and calls **71% of Dataset A defective at 89% mean confidence**. Both are true simultaneously: it is accurate on kernel close-ups and extrapolates confidently on imagery it has never seen. Softmax confidence does not separate the two — it is *higher* on some out-of-distribution data than on real test data.

The platform therefore gates quality on representation distance rather than confidence. Embeddings of the quality training split define where the head has evidence; the threshold is a percentile of validation nearest-neighbour distance, calibrated never on test.

```
Dataset 4 test  (in-distribution)          flagged= 32.0%  median dist=0.0653
Dataset A test  (never quality-labelled)   flagged=100.0%  median dist=0.5521
Dataset B test  (never quality-labelled)   flagged= 41.2%  median dist=0.0698
```

The operating point was chosen by sweep, not convention. A 95th-percentile
threshold flagged only 6% of in-distribution kernels but caught just 45% of
serve-time crops from Dataset C — where the head graded **51 of 51 kernels `Bad`
at median confidence 1.000**, which cannot be right:

| val pct | threshold | in-dist flagged | Dataset A caught | Dataset C caught |
|---|---|---|---|---|
| 95 | 0.2056 | 6.1% | 100.0% | 45.2% |
| 80 | 0.1225 | 20.7% | 100.0% | 76.8% |
| **70** | **0.0929** | **28.9%** | **100.0%** | **88.4%** |
| 60 | 0.0788 | 36.4% | 100.0% | 96.4% |

70 is selected because the two errors are not equally costly: a false flag
withholds a grade and says *unverified*, while a miss asserts a defect that may be
wrong. Three alternative scores were tested and rejected — class-conditional kNN
(AUROC 0.879 vs 0.880), class-conditional Mahalanobis in PCA space (0.647), and
distance projected onto the quality head's own weights (0.496, chance). The gate's
original weakness was its threshold, not its scoring function.

**100% of Dataset A caught**, and 88% of Dataset C serve-time crops. Flagged grades are surfaced as *unverified extrapolations* in visually distinct styling and are excluded from the seed-lot soundness rate rather than averaged into it.

### 1.5 Carried forward from the previous report

- **The train/serve gap** (§6.6): the classifier is trained on whole images but served YOLO crops. On Dataset A this cost 22 points — 100.00% whole-image versus 77.63% through the deployed path. Restoring **30% context padding** recovers 100.00%; now the default (`detection.crop_context_pad`).
- **Duplicate detections** (§6.8): Ultralytics' default NMS IoU of 0.70 produced two boxes on one kernel in 41.7% of close-ups. `nms_iou: 0.40` cuts that to 1.7% *and* lowers dense-scene count error from 1.20 to 0.72.
- **Similarity search is visual/feature similarity, not certification** (§2).
- **Seed-lot composition reporting** is new: variety composition, off-type rate and soundness aggregated over many analyses. It is explicitly *not* a certification, which is a legal determination made by an accredited laboratory following a prescribed sampling protocol.

### 1.6 What has shipped since this report was written

Everything in §§1–9 is unchanged from August 26 and is retained as the historical record. Four new served capabilities and the project's first real (non-synthetic) defect model have shipped since: foreign-object flagging, visible-symptom classification, a widened history schema for the pixel-level phases, a Gemini SDK migration, and — the largest single piece of work — a complete annotation review of 1,100 SAM/colour-distance defect proposals over real GrainSpace imagery, ending in a trained, evaluated, and registered (but not yet served) `defect_segmenter_real`. **See §10** for the full account.

---

## 2. What the System Actually Does

Given a photo of maize seeds, the system:

1. Detects individual seeds in the image and draws bounding boxes (YOLOv8-based object detection).
2. Crops each detected seed.
3. Classifies each seed's **variety** across all six varieties using a single EfficientNet-B0 trunk enhanced with cognitive attention (squeeze-and-excitation channel attention + CBAM-style spatial attention), pretrained with SimCLR-style contrastive self-supervision on the pooled training splits of every dataset.
4. Grades each seed's **quality** as Good or Bad, from a second head on the same model, trained on real expert-assigned kernel labels. Grades produced for imagery outside the range the head was validated on are returned flagged as unverified extrapolations rather than reported as defects (see Section 4).
5. Performs a visual similarity search against a FAISS index of previously seen seeds, returning the nearest-neighbor matches (explicitly labeled as *visual/feature similarity*, not certification or ground-truth matching).
6. Produces a Grad-CAM heatmap showing which image regions drove the variety prediction (explicitly labeled as *explainability visualization*, not segmentation).
7. Optionally asks Google Gemini to explain a result in plain language, summarize a batch of results, or compare two results — used strictly as a language layer on top of the system's own verified outputs, never as a source of predictions itself.

Steps 3 and 4 are one forward pass through one model, not two. All of this is exposed through a FastAPI backend with persistent history (SQLite) and a React web frontend with pages for single-image analysis, batch analysis, seed-lot composition reporting, history browsing, comparison, an AI copilot chat, and a system-capabilities page.

---

## 3. Methodology: Why the Project Diverges From the Original Brief

The original master prompt assumed three ready-made, fully-labeled datasets. Before writing any model code, the three actual datasets were audited and compared against the brief's assumptions (`docs/01_RESOURCE_INVENTORY.md`, `docs/03_DATASET_AUDIT.md`):

| Dataset | What the brief assumed | What was actually found |
|---|---|---|
| Dataset A (`seed_dataset1.zip`) | Quality/defect-labeled seed images | 1,050 images labeled only by **variety**: Zea mays Chulpi Cancha, Indurata, Rugosa. No defect/quality labels of any kind. |
| Dataset B (`seed_dataset2.zip`, "MaizeData") | A second quality/defect dataset | 17,724 images labeled only by **variety**: Bhihilifa, SanzalSima, WangDataa. No defect/quality labels. |
| Dataset C (detection zip) | Seed defect/quality annotations | 1,251 images with YOLO bounding-box annotations for a **single class, "Corn"** — object detection/counting only, no defect or variety information. |

A 34-row functionality coverage matrix was then built (`docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md`), classifying every capability requested in the brief as SUPPORTED, CONDITIONALLY SUPPORTED, NOT SUPPORTED, REQUIRES ADDITIONAL DATA, or REQUIRES CUSTOM ANNOTATION against the real data. Net finding: seed detection and variety classification are fully supported by real, correctly-labeled data. Quality/defect classification is **not** supported by any real labeled data in any of the three datasets.

Per your explicit instruction ("if the labels of data is not available create synthetic data but make sure it is accurate"), the project resolved this by building a synthetic defect dataset rather than leaving the capability unbuilt or fabricating labels on real images. See Section 4.

A literature survey of 48 papers (`docs/02_LITERATURE_SURVEY_ANALYSIS.md`) was used to justify the architectural choices — e.g., row 45 supports SimCLR/NNCLR-style contrastive pretraining for maize kernel classification, and rows 43/46 support CBAM-style attention for this domain — so the architecture is grounded in prior published work, not arbitrary.

---

## 4. Quality Assessment: Real Labels, and Where They Stop Applying

This section previously described the synthetic defect dataset as the project's only route to a quality capability. That is no longer the case. The synthetic classifier still exists and is documented in `docs/06_SYNTHETIC_DEFECT_POLICY.md`, but it has been **superseded and is off by default** in every served path.

### 4.1 The real quality dataset

Quality is now graded by a head trained on the Mendeley **EfficientMaize** dataset (`doi:10.17632/r6vvm5jkh6`), which carries genuine expert-assigned Good/Bad kernel labels. The raw archive was used rather than the pre-augmented one, deliberately: the augmented release contains ~29k images derived from the same sources, and using it would have reproduced exactly the leakage described in §1.1.

Audit results, reproducible via the commands in `docs/07_RUNBOOK.md`:

| Check | Result |
|---|---|
| Images | 4,846 (Good 2,635 / Bad 2,211) |
| Corrupt or unreadable | 0 |
| Colour mode | 4,846/4,846 RGB |
| Exact duplicates within set | 240 groups, 251 redundant copies (5.2%) |
| Duplicates straddling Good/Bad | 0 — no label conflicts |
| Byte-overlap vs Datasets A/B/C | 0 / 0 / 0 across all 20,025 existing images |
| Typical resolution | ~64×64 — already single-kernel crops |

Two audit findings shaped how it is used. The 251 exact duplicates are contained by group-aware splitting rather than deleted, so no image appears in two splits and no data is discarded. And because the images are already single-kernel crops at ~64px, they match the serve path (YOLO crop → classify) far better than Datasets A and B, which are whole images and required the 30% context-padding fix described in §6.6.

### 4.2 Measured performance

```
=== QUALITY HEAD ===  n=726  acc=97.25%  f1_macro=97.21%
   class     prec    recall    f1      support
   Good      95.8%   99.2%     97.5%   395
   Bad       99.1%   94.9%     96.9%   331
```

Precision on `Bad` is 99.1%: when this model calls a kernel defective, it is almost always right. Recall of 94.9% means roughly one defective kernel in twenty is missed, which is the error direction to state plainly in any operational claim.

### 4.3 The limit that matters — and how it is enforced

**The head is validated on kernel close-ups resembling its training data, and nowhere else.** Applied to Dataset A, it calls 71% of a clean variety dataset defective at 89% mean confidence. There is no reason to believe 71% of Dataset A is defective; this is confident extrapolation.

Softmax confidence cannot detect this — it runs *higher* on some out-of-distribution imagery (96.2% on Dataset B) than on genuine test data. The platform therefore gates on representation distance instead: embeddings of the quality training split define the region where the head has evidence, and the threshold is a percentile of *validation* nearest-neighbour distance, never calibrated on test.

```
Dataset 4 test  (in-distribution)          flagged= 32.0%  median dist=0.0653
Dataset A test  (never quality-labelled)   flagged=100.0%  median dist=0.5521
Dataset B test  (never quality-labelled)   flagged= 41.2%  median dist=0.0698
```

The operating point was chosen by sweep, not convention. A 95th-percentile
threshold flagged only 6% of in-distribution kernels but caught just 45% of
serve-time crops from Dataset C — where the head graded **51 of 51 kernels `Bad`
at median confidence 1.000**, which cannot be right:

| val pct | threshold | in-dist flagged | Dataset A caught | Dataset C caught |
|---|---|---|---|---|
| 95 | 0.2056 | 6.1% | 100.0% | 45.2% |
| 80 | 0.1225 | 20.7% | 100.0% | 76.8% |
| **70** | **0.0929** | **28.9%** | **100.0%** | **88.4%** |
| 60 | 0.0788 | 36.4% | 100.0% | 96.4% |

70 is selected because the two errors are not equally costly: a false flag
withholds a grade and says *unverified*, while a miss asserts a defect that may be
wrong. Three alternative scores were tested and rejected — class-conditional kNN
(AUROC 0.879 vs 0.880), class-conditional Mahalanobis in PCA space (0.647), and
distance projected onto the quality head's own weights (0.496, chance). The gate's
original weakness was its threshold, not its scoring function.

Dataset B sits inside the validated region, which is coherent — B and the Mendeley set are both Ghanaian kernel photography. That makes B's grades interpolation rather than extrapolation, but they remain **unverified**, because Dataset B carries no quality labels against which they could be checked.

Every flagged grade travels with a caveat string through the API, is rendered in visually distinct styling in the UI (amber dashed, never the solid red of a measured defect), and is excluded from the seed-lot soundness rate rather than averaged into it. Build the reference with `python -m src.analysis.build_quality_reference`; the behaviour is asserted by `tests/test_unified_model.py`.

### 4.4 What is still not supported

Real-world fungal, insect, or disease *diagnosis* remains out of scope. A Good/Bad grade is not a pathogen identification, and the literature on kernel-level fungal detection relies on NIR and hyperspectral bands (notably 715 nm and 965 nm) that an RGB camera cannot observe. Likewise, **defect severity scoring is not supported**: the head outputs a confidence, not a severity, and a kernel graded Bad at 0.95 is not more damaged than one at 0.70 — it is more recognisably damaged. Genuine severity requires ordinal labels or defect-area masks, neither of which exists in any dataset held by this project.

---

## 5. Build Status by Phase

### 5.1 Complete and verified (code + non-training tests all passing)

| Area | Status | Evidence |
|---|---|---|
| Dataset audit & validation | Done | `src/data/validate_dataset.py`, `validate_detection_dataset.py` — ran against real Dataset A (1,050 images, 0 corrupt, 4 duplicate pairs) and Dataset C (42,602 instances, 1 stray polygon-format label found and documented) |
| Manifest / split logic | Done | `src/data/split_dataset.py` — MD5 dedup before split, prevents train/test leakage |
| Synthetic defect generation | Done | `src/data/synthetic_defect_generator.py` — run on real Dataset A, visually verified |
| Cognitive attention modules | Done | `src/attention/channel_attention.py` (squeeze-and-excitation), `spatial_attention.py` (CBAM-style), `cognitive_attention.py` (fusion) |
| Contrastive learning module | Done | `src/contrastive/simclr.py` — augmentation pipeline, projection head, NT-Xent loss |
| Model architecture | Done | `src/models/backbone.py`, `variety_classifier.py` — `CognitiveAttentionClassifier` supports classify/embedding/contrastive forward modes |
| All training scripts | Done (code) | `pretrain_contrastive.py`, `train_variety.py`, `train_synthetic_defect.py`, `train_detection.py` — all CLI-driven, checkpoint/log/metrics output verified working via the runs in Section 6 |
| Explainability | Done | `src/explainability/gradcam.py` — hooks into backbone, overlay generation |
| Similarity search | Done | `src/similarity/embedding_index.py`, `build_index.py` — FAISS `IndexFlatL2` wrapper, built and tested against real embeddings |
| Unified inference pipeline | Done | `src/pipeline/unified_pipeline.py` — orchestrates detect → crop → classify variety → classify defect → similarity → Grad-CAM, with per-stage graceful degradation (`ModelNotAvailableError` caught per stage, never crashes the whole request) |
| Database layer | Done | `database/models.py`, `db.py` — SQLAlchemy models, tested insert/query round-trip |
| Backend API | Done | FastAPI app, 6 route groups (`analyze`, `detect`, `classify`, `similarity`, `history`, `copilot`), global exception handler, Pydantic schemas for every response type |
| Gemini integration | Done | `backend/services/gemini_service.py` — key read only from `.env`, never logged/exposed; verified to fail gracefully (never raises) both with no key and an invalid key |
| Frontend | Done | React + Vite, 6 pages (Dashboard, Analyze, Batch, History, Copilot, SystemInfo), wired to real API client, loading/error states, `npm run build` succeeds (244KB bundle) |
| Automated tests | Done | 16 tests across 4 files (`test_data_validation.py`, `test_synthetic_defects.py`, `test_api_endpoints.py`, `test_gemini_service.py`), all passing |
| Documentation | Done | 9 documents in `docs/` covering resource inventory, literature survey mapping, dataset audit, functionality coverage matrix, architecture, synthetic-data policy, runbook, cloud training log, and this report |

Two corrections to the table above, both made on August 24:

- **Automated tests** are now 27 across 5 files. The new `tests/test_integration_e2e.py` runs the real trained checkpoints end-to-end; the original 16 only covered behaviour when models are *absent*.
- **Explainability** was implemented and unit-tested but had no API route and no UI, while `/api/system-info` advertised it as a supported capability. `POST /api/explain/gradcam` and a "Show Grad-CAM" button now exist, so the advertised capability is real.

### 5.2 Trained on the local RTX 4060 — all 13 stages complete

| Task | Status |
|---|---|
| Contrastive pretraining, Dataset A (60 epochs) | Done |
| Dataset A — `baseline`, `attention_only`, `contrastive_only`, `full` | Done (all 4) |
| Synthetic defect classifier | Done |
| YOLO detection model (80 epochs) | Done |
| FAISS similarity index, Dataset A | Done |
| Contrastive pretraining, Dataset B (60 epochs) | Done |
| Dataset B — `baseline`, `attention_only`, `contrastive_only`, `full` | Done (all 4) |
| FAISS similarity index, Dataset B | Done |

The full chain ran unattended from `15:19:03` to `21:29:02`. One stage failed mid-chain and was recovered: `variety_b_baseline` was killed with exit 127 (no traceback) when it launched zero seconds after `contrastive_b` released ~6.8 GB of VRAM. The orchestrator now polls VRAM and refuses to start a stage until usage drops below 1.5 GB, with a one-shot retry — see `docs/11_LOCAL_TRAINING_RUN_LOG.md` §3.6.

Every result is in Section 6. Full timings, environment details, and the bugs the run exposed are in `docs/11_LOCAL_TRAINING_RUN_LOG.md`.

### 5.3 Verification performed after training

| Check | Result |
|---|---|
| `pytest tests/` | 40 passed, 0 failed |
| Real image through `/api/analyze/image` | Correct variety at 95.25%, `warnings: []` — no stage degraded |
| All 6 frontend pages, live, against real models | Pass (see Section 6.4) |
| Gemini connectivity + guardrails | Pass — correctly refuses to confirm disease-free status |
| Grad-CAM overlay rendered in browser | Pass — real heatmap, disclaimer shown |

---

## 6. Training Results (local RTX 4060)

All numbers below come from the **held-out test split**, evaluated with the best-validation checkpoint of each run, using the full epoch budget in `configs/config.yaml` and real ImageNet-pretrained initialization. They supersede the CPU proof-of-pipeline figures in the August 19 report, which used 4 epochs and random initialization and were never intended as results. Full run detail is in `docs/11_LOCAL_TRAINING_RUN_LOG.md`.

### 6.1 Variety classification — Dataset A (155-image test split)

| Experiment | Test accuracy | F1 (macro) | Precision (macro) | Recall (macro) |
|---|---|---|---|---|
| `baseline` (no attention, no contrastive) | 99.35% | 0.9936 | 0.9937 | 0.9936 |
| `attention_only` (+ cognitive attention) | **100.00%** | 1.0000 | 1.0000 | 1.0000 |
| `contrastive_only` (+ contrastive pretraining) | **100.00%** | 1.0000 | 1.0000 | 1.0000 |
| `full` (proposed: both) | **100.00%** | 1.0000 | 1.0000 | 1.0000 |

Per-class detail (`baseline` — the only variant with any error at all):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Zea mays Chulpi Cancha | 1.000 | 0.981 | 0.990 | 52 |
| Zea mays Indurata | 1.000 | 1.000 | 1.000 | 51 |
| Zea mays Rugosa | 0.981 | 1.000 | 0.990 | 52 |

> **Dataset A cannot validate the paper's central hypothesis.** Every variant early-stopped within 8–12 epochs, and three of the four reach a perfect test score. The entire 0.0064 macro-F1 gap between `baseline` and the rest is **one misclassified image out of 155** — comfortably inside run-to-run noise. This is consistent with the literature survey (`docs/02`, row 1), where prior published work on this exact dataset reports 99–100%. The dataset is saturated, so it cannot discriminate between the architectures. Any claim that cognitive attention or contrastive pretraining helps must rest on Dataset B and must quote the actual spread rather than asserting an improvement.

### 6.2 Variety classification — Dataset B (2,655-image test split) — WITHDRAWN

> ### ⚠ These results are withdrawn
>
> **The split that produced this table was invalid.** All 127 of Dataset B's source seeds had augmented copies in all three splits, so 2,655 of 2,655 test images had a same-seed sibling in training (§6.9). The numbers below measure memorisation and are retained only as the historical record of what was originally reported. The corrected results are in **§6.9**, and they *reverse* the conclusion drawn here.
>
> The reasoning in the original conclusion was internally sound — the four variants genuinely were indistinguishable, and reporting that plainly was right. What was wrong was the inference that this said something about the architecture. It said something about the split. A model that has already seen all 127 seeds scores ~99.9% with or without attention, so the ablation had no room in which to show a difference.

**Original conclusion, withdrawn:** that cognitive attention contributed exactly nothing (Δ macro-F1 = +0.0000), that `contrastive_only` outperformed the proposed `full` configuration, and that the entire 0.0008 spread was indistinguishable from seed variance. On a group-aware split, `full` significantly outperforms `attention_only` (p = 0.0052, Holm-corrected) and attention-alone remains indistinguishable from baseline. See §6.9.


Dataset B is ~17× the size of Dataset A (17,713 images; 12,403 train) and was the more informative of the two for the ablation — it was the only remaining test of the project's central hypothesis, since Dataset A proved saturated.

| Experiment | Test accuracy | F1 (macro) | Errors / 2,655 | Δ macro-F1 vs baseline |
|---|---|---|---|---|
| `baseline` (no attention, no contrastive) | 99.89% | 0.9988 | 3 | — |
| `attention_only` (+ cognitive attention) | 99.89% | 0.9988 | 3 | **+0.0000** |
| `contrastive_only` (+ contrastive pretraining) | **99.96%** | **0.9996** | **1** | +0.0008 |
| `full` (proposed: both) | 99.92% | 0.9992 | 2 | +0.0004 |

### 6.3 Synthetic defect classifier (4 classes, 624-image test split, group-split)

Test accuracy **99.20%**, macro-F1 0.9920. All 5 errors are cracked images read as healthy.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| healthy | 0.969 | 1.000 | 0.984 | 156 |
| cracked | 1.000 | 0.968 | 0.984 | 156 |
| discolored_mold | 1.000 | 1.000 | 1.000 | 156 |
| insect_damaged | 1.000 | 1.000 | 1.000 | 156 |

**Re-trained 2026-08-27 after fixing a split defect.** The original split was
row-level stratified. Every source kernel produces four rows here -- the untouched
original as `healthy` plus one painted variant per defect class -- so the same
physical kernel appeared in train as `healthy` and in test as `cracked`. That is the
same provenance leakage that inflated the Dataset B variety numbers (section 6.9), and the
fix is the same: `make_split_indices` now deals out whole SOURCE SEEDS via
`src.data.group_split.split_groups`, stratified by source variety. Zero source
overlap between any two splits, verified.

The number barely moved: **99.68% -> 99.20%**. That is itself the finding. The old
score was not mostly leakage; the task is simply easy, because an OpenCV-painted
crack is a far more separable thing than a real fracture. It makes the caveat below
stronger, not weaker -- fixing the leak did not reveal a hidden weakness, it
confirmed that this metric was never measuring real defect detection in the first
place.


> **This number must never be reported without its caveat.** ~99.7% here is the *expected* result and is **not** evidence of real defect-detection capability. The model is recognizing procedurally generated patterns drawn by a known OpenCV routine, so the classes are far more visually separable than real damage would be. It demonstrates that the architecture and the serving path can learn and serve a defect task. It says nothing about real-world plant pathology. See `docs/06_SYNTHETIC_DEFECT_POLICY.md`.

### 6.4 Seed detection — YOLOv8n, Dataset C (80 epochs)

| Metric | Value |
|---|---|
| mAP@50 | **0.9822** |
| mAP@50-95 | 0.6614 |
| Precision | 0.9791 |
| Recall | 0.9692 |

Strong localization at mAP@50 with looser box-tightness at high IoU — expected for small, densely packed, touching objects averaging ~34 seeds per image.

### 6.5 Bugs this run exposed

Five real bugs were found and fixed; three could not have been caught in the cloud build environment. Full detail in `docs/11_LOCAL_TRAINING_RUN_LOG.md` §3.

1. **Dataset classes were unpicklable on Windows** (blocker) — the PIL `Image` *module* was stored as an instance attribute. Linux `fork` inherits memory so it never mattered; Windows `spawn` pickles the dataset and every training script crashed instantly.
2. **GPU idle at ~2%, training data-starved (18× slowdown)** — DataLoaders used `num_workers=4` with `persistent_workers=False`, so on Windows all four worker processes were spawned and torn down every epoch. Fixed with `persistent_workers=True` + `pin_memory=True`: **30 s/epoch → 1.7 s/epoch**.
3. **YOLO weights written where the pipeline could never find them** — Ultralytics resolves a *relative* `project` against its own `runs/` directory, so detection would have reported "model not available" forever despite training successfully. Fixed with an absolute path.
4. **The configured Gemini model had been retired** — `gemini-2.0-flash` returned 404; the key itself was valid. Updated to `gemini-3.6-flash`.
5. **Similarity lookup hardcoded `experiment="full"`** — broke whenever only another variant existed. Fixed with `_resolve_experiment()` and a proper fallback chain (`full → contrastive_only → attention_only → baseline`), which reports both the requested and the actually-used experiment rather than silently mislabeling results.

### 6.6 Train/serve gap — found by running the finished application, and fixed

**This was the most consequential defect in the project, it was invisible to the test suite, and it was only caught by running the assembled application against held-out data.**

`train_variety.py` trains and evaluates on **whole images** from the manifest. The deployed pipeline (`unified_pipeline.analyze_image`) does something different: it runs YOLO detection first, crops each detected seed, and classifies the **crop**. Those are different input distributions, and the classifier has never seen a tight crop during training. Measured on the same 155-image held-out Dataset A test split (`python -m src.analysis.eval_serving_gap --dataset a`, results in `outputs/metrics/serving_gap_a.json`):

| Input path | Accuracy | Notes |
|---|---|---|
| Whole image — *as trained and evaluated* | **100.00%** (155/155) | the number quoted in §6.1 |
| YOLO crop — *as actually served to users* | **77.63%** (118/152) | 3 further images had no detection at all |

**A 22-point drop between the reported figure and the deployed one.** 34 of 152 images change prediction between the two paths, and the errors are strongly biased: the crop path predicts `Rugosa` 79 times and `Indurata` only 22, against a true test distribution of roughly 52/51/52. Confidence on the wrong crop predictions is frequently near-chance (0.44–0.65), so the model is not confidently wrong so much as genuinely unable to discriminate on this input.

The cause is straightforward. Dataset A images are single-seed close-ups where the seed already fills most of the frame; the classifier learned on that framing, including its surrounding context and scale. A tight YOLO box removes exactly that context, so at serve time the model receives inputs from a distribution it never trained on.

#### The fix

Rather than guess between candidate remedies, the amount of context restored to the crop was swept against the same held-out split:

| Crop context padding | Accuracy |
|---|---|
| 0% (tight YOLO box — the original behaviour) | 77.63% |
| 15% | 98.03% |
| **30%** | **100.00%** |
| 50% / 75% / 100% | 100.00% |

Restoring **30% of the box's own width and height as surrounding context** closes the gap completely. This is preferable to the two obvious alternatives: it needs no retraining (unlike building a crop dataset and re-running the four ablations), and unlike a "classify the whole image when there's only one seed" heuristic it remains per-seed and therefore still works on genuinely multi-seed photographs. 30% is the smallest value that fully recovers accuracy; larger margins also score 100% on Dataset A but would risk pulling neighbouring kernels into the crop on dense images.

Implemented as `detection.crop_context_pad: 0.30` in `configs/config.yaml`, applied in `AnalysisPipeline.crop_seed()` with clamping to image bounds and a fallback to the raw box if the expansion degenerates.

**Result after the fix**, same script, same split:

| Input path | Before | After |
|---|---|---|
| Whole image | 100.00% | 100.00% |
| YOLO crop — as served | 77.63% | **100.00%** |
| Whole-vs-crop disagreements | 34 | **0** |
| Crop prediction distribution | 51 / 79 / 22 | **50 / 51 / 51** (true: 52 / 51 / 52) |

The prediction skew toward `Rugosa` is gone entirely. Two regression tests were added (`test_crop_seed_restores_context_padding`, `test_crop_seed_clamps_to_image_bounds`) so the padding cannot silently be dropped back to zero — `pytest tests/` now reports 40 passed. Three held-out images pushed through the live HTTP API now return their correct varieties at 98.7% / 99.7% / 99.4% — including the `Indurata` image that previously came back as `Rugosa` at 55.8%.

**What this means for reporting.** With the fix in place the end-to-end and whole-image figures agree, so §6.1's numbers now do describe the deployed system on Dataset A. Two caveats remain worth stating: the equivalence has been verified on Dataset A only (Dataset B was not swept), and detection still finds nothing on 3 of 155 images, which is a detection-recall limit rather than a classification one.

**Residual work.** The deeper fix — training the variety classifier on YOLO crops so training and serving share an input distribution by construction rather than by a tuned margin — remains the more robust option and is worth doing before submission if GPU time allows (~1–2 hours per dataset). The padding fix makes the system correct today; it does not make the mismatch structurally impossible.

### 6.7 What would still strengthen the hypothesis test

§6.9 supersedes the null result this section originally responded to. Three of its four recommendations were acted on; the remaining gaps are worth stating precisely.

**Done.** Group-aware splitting removed the leakage (§6.9). Statistical testing replaced eyeballing — paired Wilcoxon over 19 source seeds with Holm correction for six comparisons, rather than McNemar over pseudo-replicated images. And the contrastive encoder is now pretrained on the training split alone, so the `contrastive_only` and `full` arms are no longer contaminated by having seen the evaluation images.

**Still outstanding, in order of value:**

1. **Multiple seeds per configuration, reported as mean ± std.** Every number still comes from a single run per variant. The honest test set is 19 source seeds, so one seed is worth 5.26 percentage points and run-to-run variance may be comparable to the between-variant difference. This remains the cheapest fix and the one most likely to be raised in review: 5 seeds × 4 variants is a few GPU-hours and converts a suggestive result into a defensible one.
2. **A label-efficiency curve.** SimCLR-style pretraining exists to exploit *unlabeled* data when labels are scarce. All experiments here fine-tune on the full label set, where supervision has already provided what pretraining might contribute. Retraining each variant at 1%, 5%, 10% and 25% of training labels is where contrastive pretraining should show its largest effect — and §6.9 now gives a reason to expect one, since contrastive pretraining is the component carrying the observed gain.
3. **The quality task is the better testbed.** Dataset B's honest test set is 19 independent seeds; the quality corpus has 4,416 groups with ~726 test images, giving roughly 0.14pp of resolution against Dataset B's 5.26pp. Quality assessment is also a less saturated task than 3-class variety ID. Running the same four-way ablation on the quality head is the single most informative experiment left in the project, and every asset it needs already exists.
4. **More source seeds.** 127 physical seeds behind 17,713 files is the binding constraint. No amount of augmentation increases the number of independent observations, and no split strategy can manufacture statistical power that the data does not contain.


---

### 6.8 Duplicate detections — NMS threshold tuning

Found the same way as §6.6: by running the finished application on a single-seed close-up and noticing the seed count read `2`.

The detector is trained on Dataset C, where images are dense — ~34 objects per image — and Ultralytics' default non-maximum-suppression IoU of 0.70 is appropriate for that regime, because genuinely adjacent kernels routinely overlap. The deployed system, however, is also served single-seed close-ups, where one kernel fills the frame. There, two proposals covering the same seed can overlap at IoU 0.68 and both survive a 0.70 threshold, inflating the count.

Measured on close-up images, one kernel per frame:

| NMS IoU | duplicate-box rate (close-ups) | mean count error (dense scenes) |
|---|---|---|
| 0.70 (Ultralytics default) | 41.7% | 1.20 |
| 0.50 | 12.5% | 0.95 |
| **0.40 (selected)** | **1.7%** | **0.72** |
| 0.30 | 1.7% | 0.88 |
| 0.20 | 0.8% | 1.64 |

0.40 is the joint optimum rather than a trade-off: it nearly eliminates close-up duplicates *and* lowers dense-scene counting error, because the default was also merging poorly on crowded images. Below 0.30 the threshold begins suppressing genuinely adjacent seeds, and dense-scene error climbs steeply.

The value is set as `detection.nms_iou` in `configs/config.yaml` and passed explicitly to `model.predict()`; before this fix the parameter was never passed at all, so Ultralytics silently applied its own default. Verified after the change on `chulpi_cancha 266.jpg`, which previously reported two boxes with a spurious 45% duplicate: it now reports `seeds=1  detect=62%  variety=Chulpi_Cancha 98.7%`.

---

### 6.9 Data leakage in Dataset B, and the corrected ablation

This is the most consequential finding in the project, and it was discovered while preparing a *different* piece of work — building a combined manifest — rather than by any test.

#### 6.9.1 How it was found

Dataset B's filenames encode their own provenance. `aug_4_1632053314_Bihilifa40.jpg` is an augmented copy; `77542039_Bihilifa30.jpg` is a capture. Both carry a trailing source token naming the physical seed they came from. Stripping the augmentation prefix and the capture id collapses every file onto its source:

```
Dataset B: 17,713 image files  ->  127 DISTINCT SOURCE SEEDS
copies per source: max 294, median 132, min 66
distinct source seeds per class: Bhihilifa 46, SanzalSima 33, WangDataa 48
sources whose copies appear in >1 split: 127 / 127 = 100.0%
TEST images whose source ALSO appears in TRAIN: 2655 / 2655 = 100.0%
```

This is exact, not a similarity heuristic. The split had been performed over files, and files were not independent samples.

Dataset A was checked for the same defect using a 64-bit perceptual hash, which appeared to show 83% of test images having a near-duplicate in training. **That figure was wrong and is retracted.** Re-measured with a 256-bit hash and pixel-level confirmation against the dataset's own similarity distribution:

```
mean pairwise similarity within Dataset A: 0.697
95th percentile of ALL pairs:              0.925

TEST images having a twin in TRAIN:
   corr>=0.99:    0 / 155  =   0.0%
   corr>=0.98:    7 / 155  =   4.5%
   corr>=0.95:  141 / 155  =  91.0%   <- but 5% of ALL pairs clear 0.95
```

The 0.95 bar sits barely above the 95th percentile of every pair in the set, so the apparent 91% was measuring homogeneity, not duplication. At a genuine duplicate bar, zero Dataset A test images have a training twin. **Dataset A is clean and its results stand.** The methodological lesson is worth recording: a fixed similarity threshold is meaningless without reference to the distribution it is applied to.

#### 6.9.2 The correction

Splitting is now performed over groups, never files (`src/data/group_split.py`). For Dataset B the group key is the parsed source token; for the quality corpus it is a near-duplicate cluster. The Dataset B split becomes 89 / 19 / 19 source seeds, verified to have zero cross-split contamination.

A second, independent contamination was found and fixed at the same time: `pretrain_contrastive.py` called `list_all_filepaths(data_root)`, pretraining SimCLR on **every** image including val and test. The original `contrastive_encoder_b.pt` had therefore seen the entire test set, compromising the `contrastive_only` and `full` arms by a route unrelated to the split. The script now takes `--manifest` and uses training rows only, and warns loudly when invoked the old way.

#### 6.9.3 Corrected results

```
experiment          ORIGINAL (leaked)  GROUPED (honest)     drop
baseline                       99.89%            85.35%   14.54pp
attention_only                 99.89%            85.31%   14.57pp
contrastive_only               99.96%            89.06%   10.90pp
full                           99.92%            87.97%   11.95pp

spread across the four ablations:  0.0753 pp  ->  3.7467 pp
```

Removing the leak cost ~14 points of apparent accuracy and increased the spread between variants roughly fifty-fold. That spread is the headroom the leak had been concealing.

#### 6.9.4 Significance, assessed correctly

2,669 test images represent 19 physical seeds. Treating images as independent samples would pseudo-replicate — ~140 augmented views of one kernel are one observation, not 140. Each seed is therefore scored as a unit (the fraction of its images classified correctly) and variants are compared by paired Wilcoxon signed-rank over those 19 values, with Holm correction for six comparisons:

| comparison | Δ (per-seed) | better / worse / tied | p | Holm α | verdict |
|---|---|---|---|---|---|
| `attention_only` vs `full` | +2.63pp | 12 / 2 / 5 | 0.0052 | 0.0083 | **significant** |
| `baseline` vs `full` | +3.13pp | 9 / 4 / 6 | 0.0277 | 0.0100 | not significant |
| `attention_only` vs `contrastive_only` | +3.00pp | 9 / 4 / 6 | 0.3454 | 0.0125 | not significant |
| `baseline` vs `contrastive_only` | +3.50pp | 8 / 5 / 6 | 0.3824 | 0.0167 | not significant |
| `contrastive_only` vs `full` | −0.37pp | 8 / 4 / 7 | 0.6379 | 0.0250 | not significant |
| `baseline` vs `attention_only` | +0.50pp | 7 / 7 / 5 | 0.6832 | 0.0500 | not significant |

Per-seed means: `contrastive_only` 87.05%, `full` 86.68%, `attention_only` 84.05%, `baseline` 83.55%.

**Three conclusions, stated at the strength the evidence supports:**

1. **The proposed full model significantly outperforms its attention-only ablation** — +2.63pp, winning on 12 of 19 seeds and losing on 2, surviving Holm correction. This is the project's defensible positive finding.
2. **Cognitive attention alone contributes nothing.** 7 seeds better, 7 worse, p = 0.68. The original report's finding on this specific point survives the correction intact.
3. **Contrastive pretraining is the component carrying the effect** — but note that `contrastive_only` posts the highest mean while *failing* its own significance test, because its per-seed wins are inconsistent. Mean ranking and significance disagreeing is the characteristic signature of an underpowered test, and this result must be reported with n = 19 stated rather than as a clean victory.

Reproduce with `python -m src.analysis.ablation_significance`; raw output in `outputs/metrics/ablation_significance_b_grouped.json`.

---

### 6.10 The unified model

The two per-dataset variety classifiers and the synthetic defect classifier are replaced by one model serving every prediction the platform makes.

**Why two heads and not one softmax.** The obvious reading of "one model with all the data" is a single flat softmax over the union of label sets. That is wrong twice over. A kernel is a variety *and* a quality grade, so a softmax would force a choice between `Rugosa` and `Good`. And Datasets A and B carry no quality annotation while the quality corpus carries no variety annotation, so merging the label spaces would require asserting a quality grade for 18,759 images nobody ever graded — precisely the fabrication this project exists to avoid.

The architecture is therefore one EfficientNet-B0 trunk with cognitive attention, shared by everything, and two linear heads on the pooled embedding. Every image trains the trunk; each head is trained only by images that genuinely carry its label, enforced by `CrossEntropyLoss(ignore_index=-1)`. Head losses are weighted by their share of labelled rows (variety 0.797, quality 0.203) so the 13,329 variety rows do not drown out the 3,401 quality rows.

The trunk was contrastively pretrained on 16,730 pooled images — the **training splits only** of all three datasets, so the encoder never sees an evaluation image (loss 3.184 → 2.887 over 60 epochs).

```
train: 16,730 images (13,329 variety-labelled, 3,401 quality-labelled)
early stopping at epoch 30 of 40

=== VARIETY HEAD ===  n=2824  acc=88.95%  f1_macro=93.16%
   class                        prec  recall     f1  support
   Zea_mays_Chulpi_Cancha     100.0%  100.0% 100.0%       52
   Zea_mays_Indurata          100.0%  100.0% 100.0%       51
   Zea_mays_Rugosa            100.0%  100.0% 100.0%       52
   Bhihilifa                   99.4%   99.3%  99.4%     1008
   SanzalSima                  94.9%   60.6%  74.0%      701
   WangDataa                   76.7%   97.0%  85.6%      960

=== QUALITY HEAD ===  n=726  acc=97.25%  f1_macro=97.21%
   Good                        95.8%   99.2%  97.5%      395
   Bad                         99.1%   94.9%  96.9%      331
```

**SanzalSima at 60.6% recall is the honest weak spot** — 276 of 701 test images misread as WangDataa, which shows up as WangDataa's depressed precision of 76.7%. Under the leaked split this confusion was invisible: SanzalSima scored ~99.9% because the model had memorised all 33 of its source seeds. SanzalSima also has the fewest sources of any class, so those 276 images correspond to roughly two test seeds the model gets wrong wholesale, and the figure carries the uncertainty of a 5-seed sample. It is the clearest avenue for future work in the project.

Appending new data later means adding rows to `data_processed/manifest_unified.csv` and, if new classes appear, widening the relevant head — the trunk and the other head are untouched.

Train with `python -m src.training.train_unified --encoder outputs/checkpoints/contrastive_encoder_unified_unified.pt`.

---

### 6.11 The quality distribution gate

Covered in full in §4.3. In brief: the quality head is accurate in-distribution and confidently wrong out of it, softmax confidence does not distinguish the two cases, and the platform therefore gates on representation distance calibrated at the 70th percentile of validation nearest-neighbour distance. It catches 100% of Dataset A and 88% of Dataset C serve-time crops, at a 32% in-distribution false-flag rate — a deliberately conservative operating point, since withholding a grade is cheaper than asserting a wrong defect. Flagged grades are rendered as unverified extrapolations and excluded from seed-lot soundness rates.

---

## 7. Environment / Configuration Notes

- **Environment as built:** Python 3.11.9, torch 2.5.1+cu121, ultralytics 8.4.127, Node.js 24 LTS, on Windows 11 with an RTX 4060 Laptop GPU (8 GB). `torch.cuda.is_available()` returns `True`; all training ran on the GPU.
- **`.env`** — present and the key is now confirmed **valid**. Its contents have never been read, logged, or printed by any process; validity was established only by observing a successful API response. The configured model was updated from `gemini-2.0-flash` (retired — returns 404) to `gemini-3.6-flash`.
- **`configs/config.yaml`** — `device.prefer: cuda` worked as intended with no code changes.
- **`--pretrained` / `--no-pretrained`** — the local runs correctly used the `--pretrained` default (real ImageNet transfer learning). This is a large part of why the local results substantially exceed the cloud CPU run.
- **`albumentations` was removed from `requirements.txt`** — nothing imports it, and its `stringzilla` dependency requires MSVC C++ Build Tools, which blocked the entire dependency install on Windows.
- **Install order matters:** CUDA torch must be installed from the cu121 index *before* `requirements.txt`, or pip resolves the CPU-only wheel and training silently runs on CPU. `docs/07_RUNBOOK.md` step 0 now covers this with a verification command.

---

## 8. Remaining Work — Punch List

| # | Item | Status |
|---|---|---|
| 1 | Run all training locally on the RTX 4060 | **Done** — all 13 stages, `15:19:03`–`21:29:02` |
| 2 | Verify Gemini connectivity | **Done** — real non-fallback responses; guardrails verified |
| 3 | Full local integration test | **Done** — 40/40 tests, `/api/analyze/image` and `/api/analyze/batch` clean |
| 4 | Run the frontend live against the trained backend | **Done** — all 8 pages with real model outputs |
| 5 | Results write-up / ablation comparison | **Done** — `docs/10_RESULTS_COMPARISON.md` |
| 6 | Identify and correct the Dataset B split leakage | **Done** — §6.9; group-aware splitting, all four ablations re-run |
| 7 | Train a quality model on real expert labels | **Done** — §4; Mendeley EfficientMaize, 97.25% test accuracy |
| 8 | Consolidate to a single served model | **Done** — §6.10; one trunk, two heads, six varieties plus quality |
| 9 | Gate quality output on distribution distance | **Done** — §4.3, §6.11; catches 100% of Dataset A |
| 10 | Seed-lot composition reporting | **Done** — `/api/lot/report`, verified on 1,572 kernels |
| 11 | Foreign-object flagging | **Done** — §10.1; `maize_identity_gate`, 5% review budget, 69.8% impurity recall |
| 12 | Visible-symptom classification | **Done** — §10.2; `visible_symptom_classifier`, 56.9% coverage at 76.9% test accuracy |
| 13 | Persist pixel-level measurements in history | **Done** — §10.3; additive schema, 169→217 tests, zero rows corrupted |
| 14 | Migrate off the retired Gemini SDK | **Done** — §10.4; `google-genai`, ground rules verified live |
| 15 | Annotate and train a real (non-synthetic) defect segmenter | **Done, not served** — §10.5; `defect_segmenter_real`, `seed_body` IoU 0.82, three defect channels too small or too degenerate to serve |

### Optional future work (genuinely out of scope, not omissions)

Ordered by value:

- **Run the four-way ablation on the quality head.** This is now the highest-value experiment remaining. The quality corpus has 4,416 independent groups against Dataset B's 19 source seeds — roughly 0.14pp of resolution versus 5.26pp — and quality assessment is a less saturated task than 3-class variety ID. Every asset it needs already exists. If contrastive pretraining and cognitive attention have a measurable effect anywhere in this project, this is where it can actually be demonstrated rather than merely suggested.
- **Repeat each experiment across several random seeds** and report mean ± std (§6.7, item 1). The §6.9 result rests on a single run per variant over 19 test seeds; a few GPU-hours converts a suggestive finding into a defensible one. Most likely to be raised in review.
- **Run the label-efficiency curve** (§6.7, item 2) — retrain each variant at 1/5/10/25% of training labels. This is the regime contrastive pretraining exists to serve, and §6.9 now gives a positive reason to expect an effect there.
- **Improve SanzalSima recall** (§6.10). At 60.6%, it is the weakest measured behaviour in the served model, and it is a genuine visual confusion with WangDataa rather than an artefact.
- **Acquire more source seeds for Dataset B.** 127 physical seeds behind 17,713 files is the binding constraint on statistical power; no augmentation or splitting strategy can manufacture independent observations the data does not contain.
- **Validate the quality head outside its current distribution.** It is measured only against Mendeley-style kernel close-ups. Extending it to Dataset A/B imagery requires quality labels for those datasets, which do not exist.
- **Grow the annotated-positive pool for real defect segmentation, `insect_damaged` first** (§10.5). This is no longer blocked on masks not existing — 436 real rows are annotated and a first checkpoint trains and evaluates cleanly on `seed_body`. It is blocked on volume: `insect_damaged` has 1–2 positive images per split, which is why it is degenerate rather than merely weak.
- **Calibrate confidence before re-enabling the confidence bars** (§10.6) — either temperature-scale the unified model's softmax, or accept the coverage/accuracy trade the symptom classifier already measures, and reflect the choice in the UI rather than hiding the number.
- **Capabilities still requiring data this project does not hold:** defect severity scoring (needs ordinal labels or a calibrated defect-area percentage, and every current channel is measured `DETECTION_ONLY`), foreign-object *identification* (the Phase 4 gate flags, it does not name a material — no stone/husk/debris labels exist), and fungal/insect/disease *diagnosis* (the literature relies on NIR/hyperspectral bands at 715 nm and 965 nm that an RGB camera cannot observe; the Phase 5 symptom classifier names a grader's visual category, never a pathogen). Each is listed as unsupported in the UI rather than approximated.

---

## 9. Deliverables Currently on Your Machine

**Source and docs**
- Full source tree: `src/`, `backend/`, `frontend/`, `database/`, `configs/`, `tests/`
- 11 documentation files in `docs/` (new since Aug 19: `10_RESULTS_COMPARISON.md`, `11_LOCAL_TRAINING_RUN_LOG.md`)
- `README.md`, `PROJECT_MASTER_PROMPT.md` (annotated copy of the original brief), `requirements.txt`, `.env.example`, `.gitignore`
- `.venv/` — working environment with CUDA torch; `frontend/node_modules/` installed

**Trained checkpoints** (`outputs/checkpoints/`)
- Variety, Dataset A: `variety_a_{baseline,attention_only,contrastive_only,full}_best.pt`
- Variety, Dataset B: `variety_b_{baseline,attention_only,contrastive_only,full}_best.pt`
- Variety, Dataset B **group-aware re-run** (§6.9): `variety_b_{baseline,attention_only,contrastive_only,full}_grouped_best.pt`
- **Unified served model**: `unified_seed_model_best.pt` — 6-class variety head + 2-class quality head
- Contrastive encoders: `contrastive_encoder_a.pt`, `contrastive_encoder_b.pt`, `contrastive_encoder_b_grouped.pt` (train split only), `contrastive_encoder_unified_unified.pt` (pooled train splits)
- Quality distribution reference: `quality_reference.npz` — 3,401 embeddings + calibrated threshold
- Synthetic defect (superseded, off by default): `synthetic_defect_classifier_best.pt`
- Detection: `detection_corn/weights/best.pt` (+ `last.pt`)
- Similarity: `faiss_variety_a.faiss`, `faiss_variety_b.faiss` (+ `.meta.json`)

**Results**
- `outputs/metrics/*.json` — test accuracy, precision/recall/F1, and confusion matrix per run
- `outputs/metrics/ablation_significance_b_grouped.json` — paired Wilcoxon results with Holm correction (§6.9.4)
- `outputs/metrics/unified_seed_model.json` — per-head metrics for the served model
- `outputs/logs/*.jsonl` — per-epoch training curves
- `outputs/logs/stage_*.log` — full stdout for every training stage
- `outputs/cloud_cpu_run_archive/` — the superseded August 19 CPU run, preserved for comparison

**Data artifacts** (originals never modified; the three source zips are untouched)
- `datasets/` — extracted copies of all four datasets (Dataset 4 = Mendeley EfficientMaize)
- `data_processed/manifest_dataset_{a,b}.csv` — MD5-deduped stratified splits (Dataset B's is the original leaked split, retained for the §6.9 comparison)
- `data_processed/manifest_dataset_b_grouped.csv` — group-aware split, 89/19/19 source seeds
- `data_processed/manifest_dataset_4_quality.csv` — Mendeley quality corpus, near-duplicate-clustered split
- `data_processed/manifest_unified.csv` — 23,605 rows, two label columns, no invented labels
- `data_processed/manifest_unified_pretrain.csv` — 16,730 pooled TRAIN rows for contrastive pretraining
- `data_processed/synthetic_defects/` + `manifest_synthetic_defects.csv` — 4,200 rows with per-image transform parameters logged

---

## 10. Update — September 4, 2026

Everything in §§1–9 is the report as originally written and is left untouched above. This section covers what shipped in the nine days since: four phases (4, 5, 9, 18) and the annotation-and-training work referred to throughout this project's commit history as Track A. `pytest tests/` now reports **217 passed, 0 failed**, up from the 40 quoted in §1 — the remainder of that growth is regression coverage added alongside each phase below, not a change to any existing test.

### 10.1 Foreign-object flagging gate (Phase 4)

A new gate scores each detected crop by kNN cosine distance to a maize-wide reference bank of detector crops, and calls the result *possible foreign object* or *known maize* — never a material name, because this project holds no labels for stone, husk, cob fragments or debris. Three findings shaped what shipped, each lowering the headline number rather than flattering it:

- **Calibration domain.** A reference built from whole-dataset crops shifted distances 18× against the detector crops the gate actually sees at serve time, which put the flag rate at 80.2% on ordinary maize against a claimed 2% budget. The reference is now built in the serving domain — detector crops, the same transform used at inference.
- **Crop size confound.** R² = 0.243 of the raw distance was explained by crop short side alone — a smaller crop scored more foreign regardless of what it was. The correction (§8 of `Pipeline Mathematics`, quadratic in log-crop-size) is fitted on held-out calibration crops only, and the gate declines below a measured 21px floor rather than extrapolate a fit it has no evidence for.
- **Dense scenes.** No reference or calibration image carries more than 16 detected objects. A real 300-object upload — a packed bed of ordinary maize — had 42% of its objects surfaced, because in a packed scene every crop is filled with fragments of its neighbours. Above the 16-object limit the gate now returns *unavailable* rather than a number it cannot stand behind.

Measured at the shipped 5% review budget: threshold +0.1422, gate recall 0.698 on GrainSet impurities, YOLO's own detection ceiling 0.7417 on the same set (so end-to-end recall is 0.5177), enrichment 9.29×. **Most foreign objects are still missed, and the absence of a flag is not evidence of purity** — both statements ship in the served caveat, not just in this document. Calibration data is GrainSet (Zhao et al., *Sci Data* 10:748, 2023, CC BY 4.0), 3,600 of 38,020 members fetched selectively by byte range (493 MB rather than 6.04 GB). No new model was trained — the gate reuses the existing unified encoder. Registered as `maize_identity_gate` v2.0.0.

### 10.2 Visible-symptom classifier (Phase 5)

Names a visible condition category — or says why it declined to — on the only real labels this project can honestly obtain: 1,260 GrainSpace M600 kernel crops carrying expert-grader condition categories (AP, BN, FM, HD, MY, NOR, SD). No maize leaf-disease dataset stands in for kernel imagery, and no label is propagated or synthesised; that 1,260 is a ceiling, since GrainSpace's train half is organised by cultivar and carries no condition annotation at all. Every payload carries `is_diagnosis: false` as a field, not just a caveat in prose — FM is the grader category "fusarium & mildew," and naming it is not a fusarium finding.

Three training routes were measured before choosing how to serve it:

| route | test macro-F1 | cost to the other tasks in the same trunk |
|---|---|---|
| frozen-trunk head | 0.2605 | none (trunk untouched) |
| joint fine-tune | 0.2343 | quality −0.0096 F1 |
| specialised full fine-tune | **0.5865** | variety 0.9316→0.5048, quality 0.9721→0.5800 |

The specialised weights are the only usable ones, but they measurably wreck the trunk's other two tasks — so they ship as their own checkpoint, and `unified_seed_model_best.pt` is untouched (verified bit-identical before the symptom work began). A two-part abstention gate sits in front of it: only 5 of 7 classes are validated (HD has 9 validation crops, SD has 2 — both below the support at which an operating point means anything, so a win there is withheld rather than reported), and a 0.55 softmax floor withholds the rest, chosen as the smallest floor whose validation accuracy over retained predictions reaches 0.85. Gated, it names a category for **56.9% of held-out kernels and is right on 76.9% of those** — validation ran 8.96 points optimistic against test, and that gap travels alongside the accuracy everywhere it is reported, not just here.

The serving domain was checked, not assumed: these labels come from GrainSpace crops, but the platform is fed YOLO crops from real uploads — a different domain. Measured over 633 kernels from 35 readable real uploads: 0.712 coverage, 449 of 451 named verdicts were NOR, 124 withheld for confidence and 58 for an unvalidated class. Accuracy there is deliberately reported as null — those uploads carry no visible-condition ground truth, so it cannot be computed and is not estimated. A withheld kernel is stored as `symptom_class = NULL`, with the would-have-been answer kept separately as `symptom_argmax_class` under a name that cannot be mistaken for the verdict, and the UI renders a refusal in its own visual language with no confidence bar — a percentage there would be the confidence of a prediction that was never made.

### 10.3 History schema widened for the pixel-level phases (Phase 9)

History previously stored a box, a label and some neighbours; everything the pipeline already computed about resolution and segmentation was thrown away on write. The schema now has somewhere for it to live, on two rules that matter more than the columns themselves: a seed with **no** segmentation row was never put to a segmenter, while a row saying **unavailable** was, and carries why — those are different claims and the schema no longer conflates them. Phase 2/3/4 columns stay `NULL` rather than `0`, because a zero reads as *measured, no defect found*, a different claim from *not measured*. Masks are stored as a path plus the sha256 of the file at that path, never as bytes — the same scheme the model registry already uses for checkpoints. Migration is additive and nullable only; verified against the real database (151 analyses, 2,884 detections, 5,704 classifications, 2,884 similarity results, 34 batches) with every old row reading back as *unknown* rather than a default that looks like a finding.

### 10.4 Gemini SDK migration (Phase 18)

`google-generativeai` is retired; its replacement, `google-genai`, is not a drop-in rename. Timeouts are milliseconds now, not seconds — carrying the old number across unchanged would have turned a 20-second budget into 20 milliseconds, measured both directions to confirm (`timeout=2` aborts in 0.26s, `timeout=30000` returns in 12.53s). The system instruction — `LIMITATION_CONTEXT`, the only thing stopping Gemini from claiming a defect area, a segmentation, a severity, or a pathogen diagnosis — now attaches per request rather than to a reused model object, verified live by asking point-blank for a defect area in mm² and a named fungal pathogen: both correctly refused. The automatic function-calling loop the new SDK runs by default is explicitly switched off, since this module passes it no tools.

Verifying the failure path surfaced two pre-existing frontend bugs, fixed in the same commit because they sit inside this phase's own contract: `/copilot` rendered a blank page (two hooks used without being imported), and nothing in the UI read `ai_available` or `fallback_reason` — every call site replaced a server-side outage with "No response returned." under a caption still claiming an AI-generated explanation, so an outage looked like the model having nothing to say.

### 10.5 Track A: the annotation review is complete, and the first real defect segmenter is trained

This is the largest single piece of work since August 26. All three real defect classes (`cracked`, `discolored_mold`, `insect_damaged`) previously had zero real (non-synthetic) annotations — every mask up to this point was painted by `synthetic_defect_generator.py`. Track A closes that gap by hand-reviewing SAM ViT-B + colour-distance mask proposals over real GrainSpace M600 defect-channel imagery, contact sheet by contact sheet, against a fixed set of rejection precedents (rim/specular bleed, off-body shadow misread as an on-body streak, the normal bicolor crown/germ pattern, degenerate whole-body proposals) so the same visual judgment is applied consistently across all 1,100 rows.

```
proposals.csv: 1,100 rows, all decided
  664 rejected  (no candidate cleanly isolated the true defect, or none was visible)
  400 no_defect (NOR-class rows with zero proposal candidates, by the dataset's own design)
   36 accepted  (candidate mask judged to cleanly bound a genuine, visible defect)
  ---
  436 verified  ->  data_processed/manifest_segmentation_real.csv  (322 / 58 / 56 train/val/test)
```

60 of the 436 verified rows carry a human reviewer through `review_server.py`; the remaining 376 were reviewed by `claude-opus-5` reading rendered contact sheets through `apply_sheet_decisions.py` — a script that hard-refuses `--reviewer human`, so the two paths cannot be confused after the fact. Split by GrainSpace plate id, never by image, so no plate straddles train/val/test. The resulting label provenance, `sam_proposed_mixed_review`, is stamped into the manifest's sidecar and is **not human-verified ground truth**; every downstream artefact says so.

An EfficientNet-B0 + U-Net decoder, warm-started from the same contrastive encoder used elsewhere in this project, trains on this manifest with a capped inverse-frequency BCE term (cap 25.0 — `cracked`'s true weight is ≈3,300 and would destabilise training uncapped) plus soft Dice, masked per (image, channel) so an unannotated channel contributes no gradient:

| channel | test IoU | test Dice | precision | recall | false-alarm rate on clean | verdict |
|---|---|---|---|---|---|---|
| `seed_body` | **0.82** | **0.90** | 0.95 | 0.85 | 0.000 | segments cleanly |
| `cracked` | 0.24 | 0.38 | 0.24 | 0.96 | 0.000 | real signal, 1 positive test image — too small to trust |
| `discolored_mold` | 0.28 | 0.43 | 0.33 | 0.62 | 0.000 | real signal, 5 positive test images — too small to trust |
| `insect_damaged` | 0.0004 | 0.0009 | 0.0004 | 1.00 | **1.000** | degenerate — paints almost the whole frame |

Every channel is reported `DETECTION_ONLY` — presence, not a calibrated area percentage — and thresholds were tuned on validation only, then frozen and applied once to test. `defect_segmenter_real` is registered in `configs/model_registry.yaml` with **`serves: []`**, the same convention used for `defect_segmenter_synthetic`: trained, evaluated, and honestly described, but not placed in front of a user until `insect_damaged` in particular has enough annotated positives to mean something. This is the first time real (not painted) defect masks exist anywhere in this project, and it directly retires the "pixel-level defect segmentation (needs masks)" line that appeared in this report's punch list before today — the masks now exist; what remains is volume, concentrated in one channel.

### 10.6 Frontend: confidence bars hidden pending calibration

The unified model's displayed confidence is raw, uncalibrated softmax — no temperature scaling has ever been fitted — and separability between the three GrainSpace-derived varieties (SanzalSima test F1 0.74 despite having *more* training images than the classes that score 1.00) means the number legitimately reads low even when the prediction is right. The symptom classifier's confidence is low by explicit design (§10.2 — a 0.55 floor that already trades accuracy for coverage on purpose). Rather than show a number that is either miscalibrated or intentionally conservative without comment, every `ConfidenceBar` in the frontend (Analyze, Batch, Compare, Copilot, History) is hidden behind a single `SHOW_CONFIDENCE` flag in `frontend/src/components/ui/index.jsx`, to be re-enabled once one of the two fixes in this report's punch list actually lands.

### 10.7 Phase 2 external-dataset search — exhausted, no code or data changed

Track A (§10.5) left Phase 2 (`defect_area`) blocked on volume in exactly one channel: `insect_damaged` has zero usable test positives, against `cracked` (1) and `discolored_mold` (5) — all three too small to trust regardless of the model. Before committing to the largest remaining option (triaging the ~79,209-image unlabeled GrainSpace train-half archive by condition class, then running `sam_propose.py` against it for a second annotation pass), a public external dataset was checked as the cheaper alternative. None qualified, and the search is recorded here so it is not re-run:

| Candidate | Why it doesn't close the gap |
|---|---|
| GrainSet (already integrated, §10.1) | PMC full text (`PMC10632488`) confirms its per-kernel polygon is for morphology/cropping, not defect location — 8-class whole-kernel labels, no sub-kernel mask. |
| *Foods* 2025/2026, 50-image maize set | Data Availability Statement: author-contact-only, never public. |
| *Agriculture* 16(4):421, "Limited-Annotation **Seed** Segmentation" (OAMamba) | Title names the segmentation target as the seed itself (kernel-vs-background instance segmentation), not the defect region within a sound kernel — same shape of gap as GrainSet. MDPI blocks direct fetch (403) and carries no PMC mirror, so this is the one candidate not confirmed from primary text, but every corroborating signal (title, abstract, the pattern below) points the same way. |
| VMUnet-MSADI (*Sci Reports* 2025, `PMC11954911`, public code + Google Drive dataset) | Traced past the "pixel-level segmentation masks" claim in secondary summaries to the paper's actual method: binarization + contour detection to separate a whole kernel from background, over data collected on GrainSpace's own P600/G600/M600 rigs — the class list (fusarium & shriveled, sprouted, moldy, broken, pest-attacked, black-point) is GrainSet's own, repackaged. Same kernel-vs-background segmentation, not defect-region segmentation. |

Every corn/maize paper found that uses "segmentation" in its title turns out to mean separating the kernel from the background, then classifying the whole kernel — never delineating where a defect sits within one. That matches this project's own 48-paper literature survey (`02_LITERATURE_SURVEY_ANALYSIS.md`), which never surfaced a public pixel-level defect-region mask dataset for maize either. Presented with this finding, the call on how to proceed was put to the project owner rather than decided here — per this project's standing rule that provenance and labeling-effort tradeoffs are not unilateral engineering decisions — and the decision was to leave Phase 2 blocked for now rather than commit to the train-half mining effort or ship the caveated partial result. `defect_area` and `defect_severity` remain in `orchestrator.py`'s `UNBUILT` dict, refused with their existing reasons, unchanged.

### 10.8 Gemini copilot: fixed a per-minute quota failure on every multi-seed explanation

Every `/api/copilot/*` route built its prompt by JSON-dumping the full stored analysis — every detection, both classifications, the similarity result and the assessment row for each seed, one record at a time. On the standard 300-seed upload that serialises to **1,028,009 characters**, which the free-tier Gemini quota (250,000 input tokens/minute) cannot accept in a single request; the call failed with `429 RESOURCE_EXHAUSTED`, which `generate_text`'s catch-all correctly turned into a generic `ClientError` and reported as "the AI explanation service is temporarily unavailable" — an honest fallback message, but one that read as a transient outage when the failure was actually deterministic and would recur on every 300-seed analysis, every time.

`summarize_analysis()` (`backend/services/gemini_service.py`) replaces the raw per-seed arrays with aggregates computed from those same records — detection confidence stats, per-model predicted-class counts and confidence, and assessment status counts — before any of `chat`, `explain-analysis` or `compare` build a prompt (`backend/routes/copilot.py`). No field Gemini could previously see is dropped; it is counted once instead of repeated per seed. Verified against the same 300-seed analysis that reproduced the failure: prompt length 1,028,009 → 2,306 characters, and the live endpoint (restarted — it had been running without `--reload` and was still serving the pre-fix code) now returns `ai_available: true` with a full explanation. `summarize-batch` was already reading `get_batch`'s pre-aggregated `aggregate_stats` and needed no change.

---

*This report reflects the actual, verified state of the project as of August 26, 2026 in §§1–9, September 4, 2026 in §10.1–10.6, and September 5, 2026 in §10.7–10.8. No capability listed as "done" here has been claimed without a corresponding test run, and no accuracy figure has been reported without the checkpoint and metrics file that produced it.*
