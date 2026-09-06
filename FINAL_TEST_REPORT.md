# Final Test Report

Run fresh on September 6, 2026, from a clean invocation of each suite — not carried
over from an earlier run.

## Backend — `pytest tests/`

```
244 passed, 0 failed, 29 warnings in 47.29s
```

All 29 warnings are `torch.load(weights_only=False)` `FutureWarning`s from
PyTorch's own deprecation notice (checkpoints are loaded from files this project
wrote and controls; this is a forward-compatibility notice, not a test failure).
No test was skipped, xfailed, or excluded to reach this count.

### Per-file breakdown

| File | Passed | What it covers |
|---|---|---|
| `test_orchestrator.py` | 43 | Intent→stage planning, per-stage `ran/reused/skipped/unavailable/failed` states, the image-digest cache, and the `/api/analyze/ask` + `/api/analyze/capabilities` routes now surfaced on the Copilot page. |
| `test_database_schema.py` | 35 | Schema-level NULL-vs-measured discipline, history persistence, and (new this pass) the HTTP-layer tests for `DELETE /api/history/{id}` and `GET /api/history/export` against a throwaway sqlite fixture. |
| `test_integration_e2e.py` | 26 | Full pipeline chains end to end — detect → variety → quality → gate → history — through the actual FastAPI app, not mocks. |
| `test_unified_model.py` | 24 | The unified model's dataset/split integrity, including `test_no_group_spans_more_than_one_split` — the exact invariant that would have caught the Dataset B leak before it shipped. |
| `test_visible_symptom.py` | 21 | The symptom classifier's abstention gate, withheld-class logic, and the validation→test optimism gap. |
| `test_model_registry.py` | 20 | Registry hydration from disk (fingerprint, classes, metrics binding), and that `serves: []` models are genuinely unreachable. |
| `test_resolution_gate.py` | 19 | The 24px segmentation floor and per-model `requires_resolution` enforcement. |
| `test_gemini_service.py` | 15 | Copilot Gemini integration, including the per-minute quota-failure fix for large multi-seed analyses. |
| `test_api_endpoints.py` | 15 | Route-level contracts across `analyze`, `classify`, `detect`, `similarity`, `lot`, `stats`. |
| `test_foreign_object_gate.py` | 14 | The `maize_identity_gate` review-aid framing, size correction, and resolution/scene-count limits. |
| `test_synthetic_defects.py` | 6 | The synthetic defect classifier stays labelled synthetic and off the serving path. |
| `test_data_validation.py` | 3 | Manifest schema validation. |
| `test_data_audit.py` | 3 | The new machine-readable audit tool (`src/data/audit_manifests.py`) — duplicate/leakage/balance checks, rerunnable on any manifest. |

**244** total. No file has zero coverage of the capability it names.

### What closed this pass

The one confirmed, previously-undiscovered gap: `backend/routes/history.py`'s HTTP
layer (`DELETE /api/history/{id}`, `GET /api/history/export`) had no route-level
test even though the service functions underneath (`history_service.py`) were
already covered. Found by grepping every test file for "delete"/"export" rather
than assuming coverage existed, and closed with three tests against the existing
`temp_db` fixture — never the real `database/app.db`:

- delete an existing analysis → 200, then repeat → 404
- delete an id that was never stored → 404
- export serves a real `text/csv` body with a `Content-Disposition` header, and is
  not swallowed by the dynamic `/{analysis_id}` route declared after it in the
  router (a genuine FastAPI route-ordering hazard, now guarded directly).

### What is intentionally not unit-tested

Standalone GPU training/analysis scripts — `train_variety.py`,
`pretrain_contrastive.py`, `ablation_significance.py`,
`ablation_significance_quality.py`, `calibrate_unified.py`, and the loss code added
to `train_unified.py` this pass (`FocalLoss`, class-weighted CE) — are validated by
running them against real data and inspecting real results (see
[`FINAL_RESULTS.md`](FINAL_RESULTS.md) §5), the established convention for this
project, confirmed by grep before this pass added anything new. `FocalLoss`'s
correctness is evidenced by its real training curve and its algebraic reduction to
plain cross-entropy at `gamma=0`, not a synthetic-tensor unit test.

## Frontend — `npm run build`

```
✓ built in 595ms
```

Clean production build, zero errors or warnings. `dist/assets/index-*.js` bundles
at 477.84 kB (146.62 kB gzipped).

## Historical trend

| Point in the project | Backend tests passing |
|---|---|
| README's carried-over figure (stale, pre-Phase 2) | 40 |
| Baseline at the start of this completion pass | 218 |
| After the widened history schema, foreign-object gate, symptom classifier | 241 |
| **Current** | **244** |

Every increase came from new coverage for a capability that shipped in the same
pass — none from loosening an existing assertion. See
[`docs/09_PROJECT_STATUS_REPORT.md`](docs/09_PROJECT_STATUS_REPORT.md) §10.14 for
the narrative version of this closure.
