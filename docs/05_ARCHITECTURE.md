# Final Verified System Architecture (Phases 7-9)

## Project title (kept from master prompt, scope adjusted per evidence)
AI-Powered Maize Seed Variety Recognition and Detection System Using Contrastive
Learning and Cognitive Attention — with Gemini-Powered Analysis Intelligence.
("Quality/Defect" dropped from the title's implied scope since no such dataset
exists; see `04_FUNCTIONALITY_COVERAGE_MATRIX.md`.)

## Pipeline

```
DATASET A (Corn_3_Classes: Chulpi Cancha / Indurata / Rugosa, 1050 img)
DATASET B (MaizeData: Bhihilifa / SanzalSima / WangDataa, 17,724 img)
        |
        v
CONTRASTIVE PRETRAINING (SimCLR-style, shared or per-dataset encoder)
        |
        v
COGNITIVE ATTENTION (SE channel attention + CBAM spatial attention)
        |
        v
VARIETY CLASSIFICATION HEAD  --->  per-dataset model: variety_a.pt, variety_b.pt
        |
        v
FEATURE EMBEDDING  --->  FAISS similarity index (per dataset)
        |
        v
GRAD-CAM EXPLAINABILITY

DATASET C (Grain and Objects Detection, YOLO, 1 class "Corn", 1251 img)
        |
        v
YOLOv8/v11 OBJECT DETECTION --->  bounding boxes, seed count
        |
        v
INDIVIDUAL SEED CROPPING
        |
        v
   (each crop routed into the variety pipeline above, model selectable A or B)
        |
        v
AGGREGATE RESULTS --> SQLite (analysis, detection, classification, similarity, batch)
        |
        v
FASTAPI BACKEND --> GEMINI API (explanation/copilot/summary/compare, verified-context only)
        |
        v
FRONTEND (React + Vite)
```

## Experiments (Phase 7, per dataset A and B independently)
1. Baseline backbone (EfficientNet-B0 or ResNet50, plain transfer learning)
2. Baseline + cognitive attention (SE + CBAM)
3. Baseline + contrastive pretraining (SimCLR), linear-probe then fine-tune
4. Baseline + contrastive pretraining + cognitive attention (full proposed model)

Metrics: accuracy, precision/recall/F1 (macro + weighted), per-class confusion
matrix. Compared honestly — no numbers invented before training runs.

## What is explicitly out of scope (and why)
- Quality/defect classification, defect localization, severity, segmentation, defect
  area, foreign-object/purity — all NOT SUPPORTED per the coverage matrix. The backend
  schema and frontend will not expose endpoints/fields implying these exist.
- Cross-dataset (A+B) merged variety classifier — datasets kept separate per the
  cross-dataset compatibility analysis; may revisit only after a documented domain-
  shift check (e.g. embedding-space visualization) shows they're compatible.

## Module → dataset map
| Module | Feeds from |
|---|---|
| `src/data` | A, B, C loaders + validators |
| `src/contrastive` | A, B (unlabeled) |
| `src/attention` | shared architecture, used inside variety model |
| `src/models` (variety head) | A, B (2 independent trained heads) |
| `src/models` (detection) | C (Ultralytics YOLO) |
| `src/similarity` | embeddings from variety encoder(s) |
| `src/explainability` | variety model (Grad-CAM), NOT a segmentation tool |
| `src/pipeline` | unified: C detect → crop → A/B classify → embed → similarity |
| `backend/` | FastAPI wrapping all of the above + SQLite + Gemini |
| `frontend/` | consumes backend only, never calls Gemini directly |

## Project directory (created this phase)
Matches the master prompt's Phase 27 structure, with dataset folder names corrected
to match verified contents:
```
datasets/dataset_variety_a_corntype/   (was described as "dataset_1_quality_defect")
datasets/dataset_2_variety/
datasets/dataset_3_detection/
```
All other top-level folders (`data_processed/`, `models/`, `src/`, `backend/`,
`frontend/`, `database/`, `configs/`, `outputs/`, `tests/`, `docs/`) created as
specified.

## GPU / execution note
This planning and code-authoring work happens in a cloud sandbox with no GPU. Actual
training on the RTX 4060 must run on your machine. Default plan: I write complete,
config-driven, ready-to-run training scripts (with CUDA auto-detect + CPU fallback,
AMP, checkpointing) that you run locally; you share back logs/metrics and I evaluate
and iterate. This can change if you'd rather I drive execution directly on your laptop
through the device bridge.

## Status update
Phase 10 (quality/defect) is no longer simply skipped — see
`docs/06_SYNTHETIC_DEFECT_POLICY.md` for the synthetic-data resolution adopted at the
user's request. All of Phases 1-23 have working, tested code as of this update (data
audit/validation/split, synthetic defect generation, contrastive pretraining,
cognitive attention, variety + synthetic-defect training scripts, YOLO detection
training wrapper, Grad-CAM, FAISS similarity, the unified pipeline, SQLite history,
the full FastAPI backend, the Gemini copilot layer, and a React+Vite frontend). See
`docs/07_RUNBOOK.md` for exactly what has been executed/verified in the cloud
sandbox (everything except GPU training) versus what still needs to run locally.

---

## Amendment (August 26, 2026) — the served architecture is now a single two-head model

The pipeline diagram above describes the *experimental* architecture, which remains accurate for the ablation study in `09_PROJECT_STATUS_REPORT.md` §6.1–6.2 and §6.9. It no longer describes what the application serves.

### What changed

Two per-dataset variety classifiers plus a separate synthetic defect classifier have been consolidated into **one model**:

```
DATASET A (3 varieties, 1,046 img)  ─┐
DATASET B (3 varieties, 17,713 img) ─┼─→ pooled TRAIN splits only (16,730 img)
DATASET 4 (Good/Bad,     4,846 img) ─┘         │
                                               v
                        CONTRASTIVE PRETRAINING (SimCLR, NT-Xent)
                                               │
                                               v
                         EfficientNet-B0 TRUNK (shared by both tasks)
                                               │
                                               v
                    COGNITIVE ATTENTION (SE channel + CBAM spatial)
                                               │
                                    ┌──────────┴──────────┐
                                    v                     v
                        VARIETY HEAD (6 classes)   QUALITY HEAD (2 classes)
                                    │                     │
                                    │                     v
                                    │           DISTRIBUTION GATE
                                    │      (flags extrapolated grades)
                                    v
                        FEATURE EMBEDDING → FAISS · GRAD-CAM
```

`unified_seed_model_best.pt`. One checkpoint, one forward pass, both predictions.

### Why two heads rather than one softmax

A flat softmax over the union of label sets would be wrong twice over:

1. **The labels are not mutually exclusive.** A kernel is a variety *and* a quality grade. An 8-way softmax would force the model to choose between `Rugosa` and `Good`.
2. **It would require inventing labels.** Datasets A and B carry no quality annotation; Dataset 4 carries no variety annotation. Merging the label spaces means asserting a quality grade for 18,759 images nobody ever graded.

Instead, every image trains the shared trunk, and each head is trained only by images that genuinely carry its label — enforced by `CrossEntropyLoss(ignore_index=-1)`, with `-1` meaning *this label was never collected*, never a stand-in for an assumed value. Head losses are weighted by their share of labelled rows (variety 0.797, quality 0.203) so the 13,329 variety rows do not drown out the 3,401 quality rows.

Model selection uses the **mean of both heads' macro-F1**, because optimising one head alone would let the other quietly regress.

### Contrastive pretraining is now split-aware

`pretrain_contrastive.py` previously enumerated the dataset root, pretraining SimCLR on every image including val and test. The unified encoder is pretrained on the pooled **training splits only** (`--manifest`), so it never sees an evaluation image. See `09_PROJECT_STATUS_REPORT.md` §6.9.2.

### Extending the model with new data

Add rows to `data_processed/manifest_unified.csv`. If the new data introduces classes, widen the relevant head; the trunk and the other head are untouched. If it introduces a new *task*, add a third head and a third mask — the training loop generalises without structural change.

### Files

| Concern | Module |
|---|---|
| Model | `src/models/unified_model.py` |
| Dataset | `src/data/datasets.py::UnifiedSeedDataset` |
| Training | `src/training/train_unified.py` |
| Group-aware splitting | `src/data/group_split.py` |
| Distribution gate calibration | `src/analysis/build_quality_reference.py` |
| Significance testing | `src/analysis/ablation_significance.py` |
| Serving | `src/pipeline/unified_pipeline.py::classify_unified` |

The per-dataset models (`variety_a_*.pt`, `variety_b_*.pt`) and the synthetic defect classifier remain on disk and reachable via `variety_dataset="a"|"b"`, so every ablation result in the documentation stays reproducible. They are not what the platform serves.
