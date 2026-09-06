# 12. Confidence Calibration

Status: **done**. `SHOW_CONFIDENCE` (`frontend/src/components/ui/index.jsx`) is back on,
gated per-prediction on a real, measured `confidence_calibrated` flag rather than
flipped globally.

## Why this existed as an open item

Accuracy and calibration are different claims. A classifier can be genuinely
89% accurate on held-out data and still be lying about its own certainty --
reporting 95% confidence on predictions that are actually right 80% of the
time, say. Before this phase nobody had measured which of those the unified
model's variety and quality heads do, so `SHOW_CONFIDENCE` was hard-disabled
project-wide (a blanket, unconditional `false`) rather than risk showing a
number nobody had checked.

## Method

Implemented in [`src/analysis/calibrate_unified.py`](../src/analysis/calibrate_unified.py),
run as:

```
python -m src.analysis.calibrate_unified
```

1. **Measure the baseline.** For each head, compute the raw softmax confidence
   and correctness of every validation and test prediction, bin by confidence
   into 10 bins, and report:
   - **Reliability diagram** -- accuracy vs. mean confidence per bin.
   - **Expected Calibration Error (ECE)** -- the population-weighted gap
     between confidence and accuracy across bins.
   - **Brier score** -- mean squared distance between the predicted
     distribution and the one-hot true label.
2. **Fit temperature scaling** (Guo et al., 2017): one scalar `T` per head,
   fit on **validation logits only** by minimising NLL with LBFGS. Dividing
   every logit by the same positive `T` before softmax is a monotonic
   rescaling -- it changes confidence, never which class wins, so accuracy is
   provably unchanged by construction, not just unchanged in practice.
3. **Evaluate on test exactly once**, after `T` was already fixed on
   validation. A temperature tuned against the split it is graded on is not a
   calibration, it is an overfit -- this project's standing rule for every
   threshold/parameter choice, applied here too.
4. **Accept for display only if it earns it**: test ECE after calibration
   must clear a bar (`ECE <= 0.08`) fixed in the script before this run, not
   read off the result.

## Results

Checkpoint: `unified_seed_model_best.pt`
(`sha256:80a7c541f1ad61378f33e9d5f2e93981dcdcc0d512df983b872493f46e530a7b`).
Manifest: `data_processed/manifest_unified.csv` (val n=3325, test n=3550, per
`outputs/metrics/unified_seed_model.json`). Full per-bin numbers and both
reliability diagrams are in `outputs/metrics/unified_calibration.json` and
`outputs/metrics/reliability_diagrams/{variety,quality}.png`.

| Head | Split | Accuracy | ECE (raw) | ECE (calibrated) | Brier (raw) | Brier (calibrated) | T |
|---|---|---|---|---|---|---|---|
| Variety | val  | 0.944 | 0.046 | 0.024 | 0.102 | 0.095 | 2.101 |
| Variety | test | 0.890 | 0.092 | **0.063** | 0.198 | 0.181 | 2.101 |
| Quality | val  | 0.979 | 0.012 | 0.007 | 0.037 | 0.036 | 1.682 |
| Quality | test | 0.972 | 0.021 | **0.011** | 0.046 | 0.043 | 1.682 |

Accuracy is identical before/after by construction (temperature scaling cannot
move it). Both heads' temperatures are `> 1`, meaning the raw model was
**overconfident** in both cases -- softening the softmax brought its stated
confidence closer to its actual hit rate. Both heads clear the 0.08 test-ECE
bar after calibration (variety 0.063, quality 0.011) and are therefore
**accepted for display**.

## What was wired, not just measured

A report alone would have left `SHOW_CONFIDENCE` exactly as disabled as
before. The calibration is live in the serving path:

- `outputs/checkpoints/unified_calibration.json` is the file the pipeline
  actually reads: per head, the fitted temperature, whether it was accepted
  for display, and the checksum of the checkpoint it was measured against.
- `src/pipeline/unified_pipeline.py::AnalysisPipeline._get_calibration()`
  loads that file and **verifies the checksum matches the checkpoint currently
  loaded** before applying a temperature. A mismatch (the model was retrained
  since this calibration ran) or a missing/unreadable file makes that head
  serve raw softmax, unchanged -- never a stale or unmeasured `T` applied
  silently. `accepted_for_display: false` is honoured the same way even when
  the hash matches.
- `classify_unified_full` and `classify_unified_full_batch` (the single-crop
  and Phase 21 batched paths -- covering both the legacy `/api/analyze/image`
  route and the orchestrator) both divide logits by the verified temperature
  before softmax and attach `confidence_calibrated: bool` to every
  `variety_prediction`/`quality_prediction` dict they return.
- The database's `classifications` table gained a nullable
  `confidence_calibrated` column (`database/models.py`, applied via the
  existing additive-migration mechanism in `database/db.py`) so the flag
  survives being persisted and re-read from history. Rows written before this
  phase read back as `NULL` -- "nobody recorded whether this one was
  calibrated" -- never coerced to `False` ("measured and found uncalibrated"),
  the same distinction every other additive column in this schema already
  keeps.
- The frontend's `ConfidenceBar` (`frontend/src/components/ui/index.jsx`) now
  renders only when a `calibrated` prop is explicitly `true`, threaded from
  each prediction's own `confidence_calibrated` field at every call site
  (Analyze, Batch, Compare, Copilot, History). The one confidence value in the
  UI that was never covered by this measurement -- the visible-symptom
  classifier's raw softmax, gated separately by
  `src/analysis/calibrate_symptom.py`'s abstention threshold -- passes no
  `calibrated` prop and stays hidden, exactly as before this phase.

## Tests

- `tests/test_unified_model.py`: `_get_calibration()`'s decision logic (no
  file present, checksum mismatch, `accepted_for_display: false`, and the
  positive case), plus a test that temperature scaling changes confidence
  without ever moving the predicted class.
- `tests/test_database_schema.py`: `confidence_calibrated` round-trips `True`
  and `False` independently per head through `save_analysis_result` /
  `get_analysis`, and reads back as `None` (not `False`) for a result shaped
  like every analysis written before this phase.

## What this does not do

- It does not touch the visible-symptom classifier's calibration -- that
  model already has its own, different mechanism (an abstention threshold
  from `calibrate_symptom.py`), and measuring its ECE was out of this phase's
  stated scope (mega-prompt §17 names the unified model's two heads).
- It does not change SanzalSima's 60.6% test recall (variety head's weakest
  class, tracked separately -- see `docs/09_PROJECT_STATUS_REPORT.md` and the
  quality-head ablation item in the execution plan). Calibration makes the
  model honest about its confidence; it does not make a weak class strong.
- Re-running `train_unified.py` retrains the checkpoint and invalidates this
  calibration by construction (the hash check will simply stop matching, and
  both heads will serve raw softmax again until `calibrate_unified.py` is
  re-run against the new checkpoint).
