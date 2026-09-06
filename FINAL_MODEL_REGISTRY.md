# Final Model Registry — Live Snapshot

This is a rendered snapshot of `configs/model_registry.yaml` as **hydrated from
disk** by `src.registry.model_registry.get_registry()` on September 6, 2026 — the
same function the running backend calls to decide what it is allowed to serve.
Nothing below is hand-typed from memory: every fingerprint, size, and training
timestamp came from the actual checkpoint file on this machine at snapshot time. If
a model is ever retrained, this file goes stale exactly where a hand-maintained
copy would — which is the reason the registry itself hydrates live rather than
storing these values in the YAML (see the file's own header comment).

`serves: []` is a deliberate, explicit refusal, not an oversight — see
[`FINAL_SYSTEM_STATUS.md`](FINAL_SYSTEM_STATUS.md) for what each unserved model is
trained on and why it stays off the serving path.

## Served models (`serves` non-empty — reachable from the live API)

| Key | Name | Task(s) served | Version | SHA-256 (first 16) | Trained at (UTC) |
|---|---|---|---|---|---|
| `detection` | Seed detection | detect | 1.0.0 | `22b3c2d0234edb10` | 2026-08-24T10:11:31 |
| `unified_seed_model` | Variety + kernel quality | variety, quality, embedding | 1.0.0 | `80a7c541f1ad6137` | 2026-08-25T18:39:31 |
| `quality_gate` | Quality distribution gate | distribution_gate | 1.0.0 | `79856e98bdb83b6b` | 2026-08-25T20:45:32 |
| `maize_identity_gate` | Foreign object flagging gate | foreign_object_flag | 2.0.0 | `9ea8004d8596202c` | 2026-08-29T14:35:30 |
| `visible_symptom_classifier` | Visible symptom classifier | visible_symptom | 1.0.0 | `7af51c2e2cefad85` | 2026-08-29T18:27:18 |

- **`unified_seed_model`**'s checksum (`80a7c541f1ad6137…`) is the same checkpoint
  recorded in `outputs/metrics/unified_calibration.json` — confirming the
  confidence-calibration work this pass, and every ablation/remedy experiment
  since, changed nothing about the model actually being served.
- **`visible_symptom_classifier`** withholds 2 of its 7 trained classes at serve
  time (`HD`, `SD`) — 9 and 2 validation crops respectively, below the minimum for
  a measurable operating point. See `metrics_binding: verified` above: its
  `outputs/metrics/symptom_classifier_finetune.json` was checked against the
  checkpoint's own weights, not merely assumed current.
- **`maize_identity_gate`** and **`visible_symptom_classifier`** both carry
  `metrics_binding: verified` — the strongest anti-drift status the registry
  assigns, meaning the metrics file's own fingerprint matches this checkpoint
  exactly, not just its filename.

## Trained, present on disk, deliberately not served

| Key | Name | Task | Version | SHA-256 (first 16) | Why it's not served |
|---|---|---|---|---|---|
| `contrastive_encoder` | Contrastive pretraining | embedding | 1.0.0 | `1095dede88dc1e9f` | Backbone initialiser only — folded into `unified_seed_model` at train time, never queried on its own. |
| `variety_a_full` | Variety model A (ablation) | variety | 1.0.0 | `69c0c839199e9f21` | Superseded by the unified model; kept so the §2.1/§2.2 ablation stays reproducible. |
| `variety_b_full_grouped` | Variety model B (ablation) | variety | 1.0.0 | `7d6835ca3e31d89e` | Same — superseded, kept for the group-aware ablation record. |
| `synthetic_defect_classifier` | Synthetic defect classifier | defect_pattern | 1.0.0 | `7e8c4d79208c29a1` | Trained on defects this project painted onto real kernels — demonstration only, never a real-world result. |
| `defect_segmenter_synthetic` | Defect segmenter (synthetic) | defect_segmentation | 1.0.0 | `ccf37b5d2f796db0` | Establishes the achievable ceiling and the 24px resolution floor only; not evidence about real defects. |
| `defect_segmenter_real` | Defect segmenter (real, first pass) | defect_segmentation | 0.1.0 | `0f76d46a30261c78` | `seed_body` works (IoU 0.82); the other three channels have too few positive test images or are degenerate (`insect_damaged`). Not served pending more annotated positives. |

None of the six above are reachable from any API route — `serves: []` in the
YAML is enforced at the registry level, so a caller cannot invoke them by
guessing a key.

## Task coverage — what the registry can and cannot answer

| Task | Serving model | Status |
|---|---|---|
| `detect` | `detection` | Served |
| `variety` | `unified_seed_model` | Served |
| `quality` | `unified_seed_model` | Served |
| `distribution_gate` | `quality_gate` | Served |
| `embedding` | `unified_seed_model` | Served |
| `foreign_object_flag` | `maize_identity_gate` | Served |
| `visible_symptom` | `visible_symptom_classifier` | Served (2 of 7 classes withheld) |
| `defect_segmentation` | — | **Unavailable** — no model has `serves: [defect_segmentation]`. `defect_area`/`defect_severity` in the orchestrator report `not_implemented` for exactly this reason. |
| `defect_pattern` | — | **Unavailable** — `synthetic_defect_classifier` exists but is deliberately unserved. |

A task absent from the "serving model" column returns an explicit `unavailable` or
`not_implemented` response from the API — never a value borrowed from an adjacent
model. This is the single mechanism behind every "still not supported" line in
[`FINAL_SYSTEM_STATUS.md`](FINAL_SYSTEM_STATUS.md).

---

*Regenerate this file at any time with:*
```python
from src.registry.model_registry import get_registry
reg = get_registry(refresh=True)
```
*`reg.models[key].as_dict()` returns the full hydrated record — fingerprint, size,
training timestamp, and metrics-binding status — for any key above.*
