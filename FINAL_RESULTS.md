# Final Results

Every number below is read from a metrics JSON under `outputs/metrics/` or a
checkpoint under `outputs/checkpoints/` — none is retyped from memory or estimated.
The file that produced each table is named in its heading so any figure here can be
traced back to source. Consistent with the project's own rule (§0 of
[`docs/09_PROJECT_STATUS_REPORT.md`](docs/09_PROJECT_STATUS_REPORT.md)), there is no
single blended "accuracy" for the system: each subsystem is reported on its own
task, on its own test set, with its own honest caveats.

Test sets were touched once per model. All model-selection decisions (which
checkpoint to keep, which of two remedies to prefer) were made on validation data;
test numbers below are reported for the record, never used to pick between variants.

---

## 1. Detection — YOLOv8n, Dataset C (`outputs/metrics/detection.json`)

| Metric | Value |
|---|---|
| mAP@50 | 98.22% |
| mAP@50-95 | 66.14% |
| Precision | 97.91% |
| Recall | 96.92% |

Single class (`Corn`). Locates and counts kernels; carries no variety, quality, or
condition information of its own.

## 2. Variety classification — per-dataset ablations

### 2.1 Dataset A (155-image test split, `outputs/metrics/variety_a_full.json`)

Clean split — zero test images share a ≥0.99-correlation twin with training. Test
accuracy and macro-F1 both **100.0%** on all three varieties (Chulpi Cancha,
Indurata, Rugosa); the task is not close to saturating this dataset's difficulty.

### 2.2 Dataset B — leaked vs. corrected, group-aware ablation (`outputs/metrics/ablation_significance_b_grouped.json`)

17,713 images are augmented copies of only 127 physical seeds. The original split
let copies of the same seed land in train and test; corrected splitting groups all
copies of a seed into one split only (89/19/19 seeds).

| Comparison | Δ (pp) | Seeds better/worse/tied (of 19) | Wilcoxon p | Survives Holm? |
|---|---|---|---|---|
| baseline → attention_only | +0.50 | 7 / 7 / 5 | 0.683 | No |
| baseline → contrastive_only | +3.50 | 8 / 5 / 6 | 0.382 | No |
| baseline → full | +3.13 | 9 / 4 / 6 | 0.028 | No (fails correction) |
| attention_only → contrastive_only | +3.00 | 9 / 4 / 6 | 0.345 | No |
| **attention_only → full** | **+2.63** | **12 / 2 / 5** | **0.0052** | **Yes** |
| contrastive_only → full | −0.37 | 8 / 4 / 7 | 0.638 | No |

**Only one comparison survives Holm correction: `full` significantly beats
`attention_only` alone.** Contrastive pretraining, not cognitive attention, carries
the effect — attention alone contributes nothing measurable on this dataset. n = 19
seeds, one run per variant; see `docs/09` §6.9 for the full derivation.

### 2.3 Quality dataset (EfficientMaize) — the same ablation, repeated on a second task (`outputs/metrics/ablation_significance_quality_grouped.json`)

662 test groups (near-duplicate kernels merged), far more statistical resolution
than Dataset B's 19 seeds.

| Comparison | Δ (pp) | Groups better/worse/tied (of 662) | Wilcoxon p |
|---|---|---|---|
| baseline → attention_only | +1.01 | 12 / 6 / 644 | 0.094 |
| baseline → contrastive_only | +0.93 | 16 / 11 / 635 | 0.174 |
| baseline → full | +0.03 | 14 / 15 / 633 | 0.876 |
| attention_only → contrastive_only | −0.08 | 11 / 12 / 639 | 0.986 |
| attention_only → full | −0.98 | 7 / 14 / 641 | 0.174 |
| contrastive_only → full | −0.91 | 9 / 15 / 638 | 0.221 |

**No comparison is significant.** The served architecture (`full`) is statistically
indistinguishable from every other variant on this task. Read two ways in `docs/09`
§10.11: either the quality task lacks the fine spatial structure the Dataset B
effect depends on, or that effect is itself task-specific. No served checkpoint
changed as a result of this ablation — it does not produce a deployable quality-only
model, only a significance test on the architecture choice.

## 3. The served unified model — variety + quality, one trunk, two heads

Checkpoint `unified_seed_model_best.pt`, `outputs/metrics/unified_seed_model.json`.
6 varieties (3 from Dataset A + 3 from Dataset B) and Good/Bad quality, trained
jointly with contrastive pretraining + cognitive attention (the `full` recipe from
§2.2 above, extended across datasets).

### 3.1 Variety head (test n = 2,824)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Zea_mays_Chulpi_Cancha | 100.0% | 100.0% | 100.0% | 52 |
| Zea_mays_Indurata | 100.0% | 100.0% | 100.0% | 51 |
| Zea_mays_Rugosa | 100.0% | 100.0% | 100.0% | 52 |
| Bhihilifa | 99.40% | 99.31% | 99.35% | 1,008 |
| **SanzalSima** | 94.87% | **60.63%** | 73.98% | 701 |
| WangDataa | 76.69% | 96.98% | 85.65% | 960 |

Overall: **accuracy 88.95%**, macro-F1 93.16%. SanzalSima is the one weak class,
mostly mistaken for WangDataa (276 of its 701 test images) — see §5 below for the
investigation into why, and why it was left as is.

### 3.2 Quality head (test n = 726)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Good | 95.84% | 99.24% | 97.51% | 395 |
| Bad | 99.05% | 94.86% | 96.91% | 331 |

Overall: **accuracy 97.25%**, macro-F1 97.21%.

## 4. Confidence calibration (`outputs/metrics/unified_calibration.json`, full write-up [`docs/12_CONFIDENCE_CALIBRATION.md`](docs/12_CONFIDENCE_CALIBRATION.md))

Temperature scaling (Guo et al. 2017), one scalar per head, fit on validation
logits by NLL minimisation, evaluated on test once. Accuracy is unchanged by
construction — temperature scaling only reshapes confidence, never the argmax.

| Head | Temperature | Test ECE before | Test ECE after | Test Brier before | Test Brier after |
|---|---|---|---|---|---|
| Variety | 2.101 | 0.0924 | **0.0630** | 0.1975 | 0.1811 |
| Quality | 1.682 | 0.0211 | **0.0112** | 0.0461 | 0.0431 |

Both pass the pre-declared acceptance bar (test ECE ≤ 0.08) and are live in serving
as `confidence_calibrated`, verified end-to-end from checkpoint to API response to
frontend display. Checksum of the calibrated checkpoint matches the one still
served today (`80a7c541f1ad…`) — calibration did not require retraining or change
any prediction.

## 5. SanzalSima recall — investigated, not resolved (`outputs/metrics/unified_varweighted_seed_model.json`, `unified_focal2_seed_model.json`)

Two standard remedies for a low-recall class were tried against the served
baseline. Model selection was validation-only; both experimental checkpoints failed
that bar, so neither was ever a candidate to replace the served model regardless of
its test numbers (reported below purely for the record).

| Variant | Val mean F1 (selection metric) | Test variety acc | Test variety F1 (macro) | SanzalSima test recall |
|---|---|---|---|---|
| **baseline (served)** | **0.9733** | 88.95% | 93.16% | **60.63%** |
| class-weighted CE | 0.9704 | 88.17% | 92.72% | 59.63% |
| focal loss (γ=2) | 0.9689 | 87.08% | 92.12% | 59.20% |

Both remedies made recall *worse*, not better, and cost precision on
Bhihilifa/WangDataa along the way. This corroborates rather than resolves the read
in `docs/09` §6.10: the SanzalSima/WangDataa confusion looks like genuine visual
similarity in the source imagery, not a class-imbalance or hard-example artefact a
loss-function change can buy back. The served checkpoint is unchanged. Untried and
not ruled out: stronger augmentation, a different crop strategy, and more physical
source seeds (docs/09 §10.12).

## 6. Quality distribution gate (`outputs/metrics/quality_gate_comparison.json`)

kNN cosine distance to the training feature bank, catching quality predictions made
on imagery unlike anything the head trained on, instead of trusting softmax
confidence (which is measurably *higher* on some out-of-distribution data than on
real test data).

| Method | In-distribution flag rate | Dataset A catch rate | Dense-scene crop catch rate | AUROC (Dataset A) |
|---|---|---|---|---|
| **Pooled kNN (shipped)** | 6.06% | **100.0%** | 45.2% | 0.9999 |
| Class-wise kNN | 5.92% | 100.0% | 44.4% | 0.9996 |
| Mahalanobis | 3.03% | 65.8% | 22.0% | 0.9652 |
| Quality-projection distance | 4.41% | 86.5% | 28.8% | 0.9687 |

The shipped pooled-kNN gate catches every one of Dataset A's clean-but-unrelated
images while withholding only ~6% of genuinely in-distribution grades as
unverified.

## 7. Foreign-object review gate (`outputs/metrics/maize_gate_comparison.json`)

A ranked review aid, not a classifier — it scores how unlike known maize a detected
object looks; it never names what the object is. Fitted entirely on held-out maize
(no foreign-object labels used in fitting); GrainSet impurities are used only to
*measure* the resulting catch rate.

At the shipped 5% review budget (threshold 0.1422, size-corrected distance):

| Quantity | Value |
|---|---|
| Maize surfaced rate (unified test set) | 7.98% |
| Maize surfaced rate (GrainSpace damaged, held out) | 6.49% |
| Foreign-object catch rate (GrainSet impurities) | 69.8% |
| Enrichment over chance | 9.29× |
| Detector fires on impurity images | 74.2% |
| **End-to-end recall (detect × flag)** | **51.8%** |

The absence of a flag is explicitly not evidence an object is maize — end-to-end
recall near 52% means roughly half of impurities are missed somewhere in the
detect-then-flag chain.

## 8. Visible symptom classifier (`outputs/metrics/symptom_classifier_finetune.json`, `symptom_gate.json`)

Fine-tuned on real GrainSpace M600 expert-grader condition labels (7 classes, 1,260
crops total — the entire labelled supply available). Gated: only 5 of 7 classes are
ever asserted (HD and SD are withheld — 9 and 2 validation crops respectively, below
the threshold at which an operating point means anything).

| Quantity | Value |
|---|---|
| Raw test accuracy, all 7 classes, argmax (no gate) | 65.0% |
| Raw test macro-F1, all 7 classes | 58.6% |
| **Gated coverage** (fraction of test kernels given a named category) | **56.9%** |
| **Gated accuracy** (accuracy over what's named) | **76.9%** |
| Validation accuracy at the same threshold (for comparison) | 85.9% |

The gap between validation (85.9%) and test (76.9%) accuracy at the fixed threshold
is the honest optimism gap of choosing a threshold on validation data — reported,
not hidden. `is_diagnosis: false`: this reproduces a grader's *visual category*
label, never a pathogen finding.

## 9. Real-world defect segmentation — synthetic ceiling vs. real signal

### 9.1 Resolution floor (`outputs/metrics/segmentation_resolution_floor.json`)

Measured via a true-mask downscale/upscale round trip (an information-loss
ceiling, not a model result): below **24px of kernel width**, no segmenter — this
one or any other — can produce a mask meeting the tolerance bar (median IoU ≥ 0.85,
p90 area error ≤ 5%). The platform refuses to segment below this floor rather than
return a low-confidence mask.

### 9.2 Synthetic segmenter (`outputs/metrics/defect_segmenter_synthetic.json`) — establishes what's achievable, not a real-world result

| Channel | Test IoU | Test Dice | False-alarm rate on clean |
|---|---|---|---|
| seed_body | 0.991 | 0.995 | 0.0% |
| discolored_mold | 0.957 | 0.978 | 0.6% |
| insect_damaged | 0.932 | 0.965 | 0.0% |
| cracked | 0.423 | 0.595 | 0.0% |

Labels here are exact by construction (this project painted the defects and
reconstructed the mask from its own transform parameters) — this measures the
architecture's ceiling, never real-world defect performance.

### 9.3 Real segmenter, first pass (`outputs/metrics/defect_segmenter_real.json`) — **not served**

Trained on SAM-proposed / colour-distance masks over real GrainSpace defect images,
reviewed row-by-row (60/436 by a person, 376/436 by claude-opus-5 reading contact
sheets — model-verified, not human-verified, per its own sidecar).

| Channel | Test IoU | Recall | False-alarm rate on clean | Test images with this defect |
|---|---|---|---|---|
| **seed_body** | **0.817** | 85.5% | 0.0% | 56 |
| cracked | 0.236 | 96.2% | 0.0% | 1 |
| discolored_mold | 0.277 | 62.1% | 0.0% | 5 |
| insect_damaged | 0.0004 | 99.7% | **100.0%** | 2 |

`seed_body` segments cleanly. `cracked` and `discolored_mold` show real signal —
zero false alarms on clean kernels — but on test sets of 1 and 5 positive images,
too small for the number to mean anything beyond "not obviously broken."
`insect_damaged` is degenerate: it paints almost the entire frame and must not be
trusted for that channel at all. Every channel here is `DETECTION_ONLY` (presence,
not a calibrated area percentage) — not served pending more annotated positives,
particularly for `insect_damaged`.

## 10. Dataset integrity audit (`outputs/metrics/data_audit.json`, `src/data/audit_manifests.py`)

Machine-readable, rerunnable version of the manual checks in
[`docs/03_DATASET_AUDIT.md`](docs/03_DATASET_AUDIT.md): content-hash duplicate
detection, group/split-leakage detection, and class-balance reporting, across every
manifest in the project.

| Manifest | Rows | Group-split leakage | Content-hash leakage |
|---|---|---|---|
| manifest_dataset_a.csv | 1,046 | n/a (no groups) | 0 offending hashes |
| manifest_dataset_b.csv | 17,713 | n/a (ungrouped — the withdrawn split) | 0 offending hashes |
| manifest_dataset_b_grouped.csv | — | **0 offending groups** | 0 offending hashes |
| manifest_dataset_4_quality.csv | 4,846 | **0 offending groups** (4,416 groups) | 0 offending hashes |

Zero leakage detected on every manifest actually used to train or evaluate a served
model — the tool that would have caught Dataset B's original leak now runs
automatically rather than depending on a one-time manual audit.

## 11. Test suite

`pytest tests/ -q` → **244 passed, 0 failed** across 13 files. `npm run build` in
`frontend/` is clean. See [`docs/09_PROJECT_STATUS_REPORT.md`](docs/09_PROJECT_STATUS_REPORT.md)
§10.14 for what the suite covers and the one HTTP-layer gap it closed this pass.

---

*Every table above was generated from a metrics file that exists on disk under
`outputs/metrics/` as of September 6, 2026, or from a live `pytest`/checkpoint-hash
run on that date. Nothing here is a projection, an estimate, or a number carried
over from an earlier, superseded run.*
