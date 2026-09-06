# Final System Status

**As of September 6, 2026.** This is the single point-in-time snapshot of what the
Maize Seed Intelligence platform does, does not do, and how each claim was verified
— written at the close of the completion pass tracked in
[`docs/09_PROJECT_STATUS_REPORT.md`](docs/09_PROJECT_STATUS_REPORT.md). For results
and numbers, see [`FINAL_RESULTS.md`](FINAL_RESULTS.md). For the model inventory,
see [`FINAL_MODEL_REGISTRY.md`](FINAL_MODEL_REGISTRY.md). For the test suite, see
[`FINAL_TEST_REPORT.md`](FINAL_TEST_REPORT.md).

## What it does

A field photo of one or more maize kernels goes in. The pipeline (a YOLOv8n
detector feeding a single shared EfficientNet-B0 trunk with SE/CBAM attention,
warm-started by SimCLR contrastive pretraining) produces, per kernel:

1. **A location and count** — the detector crop.
2. **A variety name**, one of six, with a calibrated confidence.
3. **A Good/Bad quality grade**, with a calibrated confidence — or **"unverified"**
   when the kernel's representation falls outside the region the quality head has
   evidence for, rather than a guessed grade.
4. **A foreign-object review flag**, when the crop ranks among the most unlike known
   maize the platform has seen — framed as a ranking for human review, never as an
   identification of what the object is.
5. **A visible symptom category**, one of five validated GrainSpace grader
   categories, when the model's confidence clears its gate — or "withheld" when it
   doesn't. Never framed as a pathogen diagnosis.
6. **A similarity search** against a FAISS index of the training population, for
   provenance/lookalike queries — never used to certify a result.
7. **A Grad-CAM saliency map** on request, labelled as an attention visualisation,
   never as a segmentation mask.

Seed-lot composition reports, batch analysis, a Gemini-backed copilot (for
explanation and free-text Q&A only — it never independently supplies a
variety/quality/defect verdict), an orchestrator that plans which of the above
stages a natural-language question actually needs, and a history log with CSV
export and delete sit on top of this pipeline, all reachable from the live React
frontend.

## What it deliberately does not do

Reported here exactly as the platform itself reports it — as an explicit
`unavailable`/`not_implemented` response, never a lowered-confidence guess:

- **Defect *type* classification or severity scoring** on real imagery. A synthetic
  4-class defect classifier exists and is disabled in the serving path
  (`serves: []`) — it was trained on defects this project painted onto real
  kernels, and is kept only as a demonstration, never presented as a real-world
  result.
- **Pixel-level defect segmentation**, with one narrow exception: `seed_body`
  (kernel-vs-background) segments cleanly (test IoU 0.82) on real defect imagery
  and is the one channel with enough annotated signal to trust. The other three
  channels (cracked, discolored_mold, insect_damaged) are trained but not served —
  either too few positive test images to mean anything, or (insect_damaged)
  outright degenerate. Below 24px of kernel width, segmentation is refused outright
  rather than attempted — a measured information-loss floor, not a guess.
- **Foreign-object identification.** The review gate ranks and flags; it has never
  seen a stone/husk/debris label and cannot name what an object is.
- **Fungal/insect/disease diagnosis.** No dataset exists — after two independent,
  primary-source-verified searches (`docs/09` §10.7, §10.13) — of real maize
  *kernel* images paired with a lab- or expert-confirmed pathogen label at RGB
  resolution. Published kernel-level fungal detection relies on NIR bands (715 nm,
  965 nm) a consumer RGB camera cannot observe. The visible-symptom classifier
  reproduces a grader's *visual category*, explicitly not a pathogen finding
  (`is_diagnosis: false`).

## What changed in this completion pass

Starting point: a working prototype with the core pipeline built, but confidence
bars hidden pending calibration, one known weak class un-investigated, two
externally-sourced capability gaps unexplored, a "built but invisible" orchestrator,
and a materially stale README/test count. In order:

1. Confidence calibration shipped (temperature scaling, both heads, `docs/12`)
   — the confidence bars are back on, showing a number now proven honest.
2. The architecture choice (attention + contrastive pretraining) was re-tested on a
   second task (the quality head) — an honest null result, reported as one.
3. The one known weak class (SanzalSima, 60.6% recall) was investigated with two
   standard remedies; both failed validation selection and neither shipped —
   corroborating rather than resolving the "genuine visual confusion" read.
4. Two dataset searches (defect-segmentation masks, kernel disease labels) came back
   empty against primary-source verification — the honest ceiling was reported, not
   papered over with a lower verification bar.
5. The orchestrator (already built) was surfaced on the Copilot page as a real
   "ask a question" control, separate from the Gemini chat panel.
6. History export and delete were built as genuinely new backend routes and wired
   into the Settings page, replacing a "not offered here" placeholder.
7. A machine-readable, rerunnable data-quality audit (`src/data/audit_manifests.py`)
   replaced the one-off manual checks in `docs/03`, catching the same class of
   split-leakage bug Dataset B once had, automatically, on every manifest.
8. A genuine HTTP-layer test-coverage gap (the two new history routes) was found by
   systematic grep, not assumption, and closed.
9. README.md and this set of final documents were brought current — test badge from
   "40 passed" to the actual 244, and every capability above added to the record.

Nothing above changed a served model checkpoint except by explicit, reported
choice — the calibration temperatures are the only numbers threaded into serving
this pass; every ablation and remedy that didn't clear its own bar was left
un-promoted, checksum-verified against the running `unified_seed_model_best.pt`.

## Where to look next

| Question | Document |
|---|---|
| What resources were actually available vs. assumed? | [`docs/01_RESOURCE_INVENTORY.md`](docs/01_RESOURCE_INVENTORY.md) |
| What does the research literature say, and where does this diverge? | [`docs/02_LITERATURE_SURVEY_ANALYSIS.md`](docs/02_LITERATURE_SURVEY_ANALYSIS.md) |
| Is any dataset leaked, duplicated, or imbalanced? | [`docs/03_DATASET_AUDIT.md`](docs/03_DATASET_AUDIT.md) |
| What is/isn't genuinely supported, functionality by functionality? | [`docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md`](docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md) |
| How is the system built, end to end? | [`docs/05_ARCHITECTURE.md`](docs/05_ARCHITECTURE.md) |
| Full narrative history, every phase, every caveat | [`docs/09_PROJECT_STATUS_REPORT.md`](docs/09_PROJECT_STATUS_REPORT.md) |
| How was confidence calibrated? | [`docs/12_CONFIDENCE_CALIBRATION.md`](docs/12_CONFIDENCE_CALIBRATION.md) |
| Every number, one place, cited to its source file | [`FINAL_RESULTS.md`](FINAL_RESULTS.md) |
| Which model serves which task, and how do I reproduce it? | [`FINAL_MODEL_REGISTRY.md`](FINAL_MODEL_REGISTRY.md) |
| Does the test suite actually pass? | [`FINAL_TEST_REPORT.md`](FINAL_TEST_REPORT.md) |
| How do I reproduce any of this from a clean clone? | [`docs/07_RUNBOOK.md`](docs/07_RUNBOOK.md) |
| What was the project originally asked to build? | [`PROJECT_MASTER_PROMPT.md`](PROJECT_MASTER_PROMPT.md) |
