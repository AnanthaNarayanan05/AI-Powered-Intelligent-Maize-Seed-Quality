# Synthetic Defect/Quality Data Policy (Phase 10, revised)

## Why this exists
You asked for the quality/defect module to be completed even though no real
defect-labeled dataset was provided, using synthetic data "made accurate." The
project's own mandatory rules (from `PROJECT_MASTER_PROMPT.md`) say: never invent
labels, never fabricate ground truth, never claim severity without real severity
labels. Those two instructions only reconcile one way: synthetic data is used, but
the ground truth is not invented — it is **generated and therefore exactly known**,
and the system is honest everywhere about the data being synthetic/simulated, never
presented as real diseased-seed pathology.

## What "accurate" means here
Accuracy for a synthetic dataset means: the transformation that creates a "defect"
image is deterministic, controlled, and logged, so the label is ground truth **by
construction**, not guessed. It does not mean the synthetic images are claimed to be
photographs of genuinely diseased/fungal/cracked seeds — they are procedurally
generated visual approximations of damage classes described in the literature survey
(rows 16, 23, 37, 41, 43: crack, mold/fungal discoloration, insect/worm damage,
breakage), built by applying OpenCV image operations to the real corn images already
in Datasets A/B.

## Classes generated (chosen from literature-survey-documented defect types)
`healthy` (unmodified real image), `cracked` (procedural fracture-line overlay +
local contrast/texture disruption), `discolored_mold` (procedural blotchy
color/darkness patches mimicking fungal discoloration, survey rows 5, 41, 43),
`insect_damaged` (procedural small dark punched-hole regions, survey row 43
"pest-attacked").

## Hard constraints enforced in the generator and everywhere downstream
1. The synthetic generator only ever runs on the real corn images already audited in
   `03_DATASET_AUDIT.md` (Datasets A/B) — no unrelated stock imagery.
2. Every generated image's transformation parameters (type, position, intensity,
   seed) are written to a manifest CSV alongside the image, so every label is
   traceable and reproducible.
3. The word "synthetic" appears in: the model name (`synthetic_defect_classifier`),
   every API response field that returns its predictions, every database row that
   stores its predictions, every frontend label, and every Gemini prompt/response
   touching it.
4. Nowhere does the system claim this model detects real fungal infection, real
   insect infestation, or real mechanical damage — the UI/API copy is
   "simulated/synthetic defect-pattern classifier for demonstration," not a
   certified plant-pathology tool. `docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md` is
   updated accordingly (see the amended row below).
5. The Gemini limitation-aware copilot is given this exact caveat as context so it
   cannot describe synthetic predictions as real-world defect detection.

## Updated functionality coverage (supersedes the original row 3-8, 26-32 in
`04_FUNCTIONALITY_COVERAGE_MATRIX.md`)
| Functionality | Status | Note |
|---|---|---|
| Quality/defect classification | CONDITIONALLY SUPPORTED (synthetic) | Trained on procedurally generated synthetic damage patterns over real seed images; ground truth exact by construction; explicitly not validated against real diseased seeds |
| Defect type classification (cracked/discolored/insect) | CONDITIONALLY SUPPORTED (synthetic) | same as above |
| Defect severity | NOT SUPPORTED | still no real severity scale; synthetic intensity parameter is not exposed as a clinical severity score |
| Real-world fungal/foreign-object/purity detection | NOT SUPPORTED | unchanged — still no real-world labeled data for these |

This document is the audit trail for that decision — if a genuine defect dataset is
supplied later, the synthetic model is swapped out and this file is updated to say so.
