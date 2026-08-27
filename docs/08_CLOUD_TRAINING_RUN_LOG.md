# Cloud Training Run Log — what was actually executed here, and its limits

You asked me to run training myself. I don't have a way to execute commands on your
laptop — the device bridge only transfers files, it has no remote-shell tool. So
what I actually did was install PyTorch/torchvision/faiss-cpu in this cloud sandbox
and run REAL training against your REAL datasets here, on CPU. This section is the
honest record of what that produced and — just as importantly — what it did NOT
produce, so you know exactly what still needs your RTX 4060.

## Environment constraints hit (and how they were handled)
- **No GPU.** `torch.cuda.is_available()` is `False` here; everything ran on CPU.
- **Only 2 vCPUs.** A single classification epoch on Dataset A (736 train images)
  took ~95-145s. Contrastive pretraining (which forwards two augmented views per
  batch) was tested and did not finish even one epoch inside a 5-minute bound —
  genuinely too slow on this hardware, not a bug. Dataset B (17,724 images, ~24x
  Dataset A) and YOLO detection training were not attempted here for the same
  reason — they would take many hours to days on 2 CPU cores, versus minutes on
  your RTX 4060.
- **ImageNet-pretrained weight downloads are blocked** from this sandbox (same
  network restriction that blocked `download.pytorch.org` earlier for pip). Added a
  `--no-pretrained` flag to `train_variety.py`, `pretrain_contrastive.py`, and
  `train_synthetic_defect.py` so training could proceed with random-initialized
  backbones instead. Your local runs should omit this flag (the default is
  `--pretrained`, i.e. real ImageNet transfer learning) since your machine can reach
  the weights host normally.
- **A real bug found and fixed while testing this:** `src/pipeline/unified_pipeline.py`
  hardcoded `experiment="full"` for similarity/embedding lookups, so it broke as
  soon as a dataset had only, say, an `attention_only` checkpoint trained. Fixed
  with `_resolve_experiment()`, which falls back through
  `full → contrastive_only → attention_only → baseline` (whichever actually exists
  on disk) and reports which one it used in the `model` field rather than silently
  mislabeling the result. `classify_variety()`'s response now includes both
  `model` (what was actually used) and `requested_experiment`.

## What was actually trained (real data, real gradient updates, real held-out test set)
Command run in this sandbox:
```
python -m src.training.train_variety --dataset a --experiment baseline --epochs 4 --no-pretrained
python -m src.training.train_variety --dataset a --experiment attention_only --epochs 4 --no-pretrained
python -m src.similarity.build_index --dataset a --experiment attention_only
```

| Experiment | Epochs | Test accuracy | Test F1 (macro) | Checkpoint |
|---|---|---|---|---|
| `baseline` (no attention, no contrastive, no ImageNet init) | 4 | 97.4% | 0.974 | `outputs/checkpoints/variety_a_baseline_best.pt` |
| `attention_only` (+ cognitive attention, no ImageNet init) | 4 | 96.8% | 0.968 | `outputs/checkpoints/variety_a_attention_only_best.pt` |

Both are genuinely trained and were verified end-to-end through the actual FastAPI
pipeline (`AnalysisPipeline.classify_variety` and `.embed_and_search`) — not just the
training script in isolation. A held-out Indurata test image was correctly classified
at 99.95% confidence, and similarity search against the FAISS index returned other
real Indurata images as nearest neighbors.

**Read the accuracy honestly:** ~97% after only 4 epochs, no pretrained weights, is
plausible mainly because Dataset A is small (1050 images), low-resolution, and
per the literature survey (`docs/02_LITERATURE_SURVEY_ANALYSIS.md` row 1: prior work
on this exact published dataset reports 99-100% accuracy) this is a comparatively
easy 3-class task. This result does NOT mean the harder Dataset B (real-world
variation, 3 different classes, more images) will reach similar accuracy this
quickly, and it does NOT include the contrastive-pretraining or ImageNet-transfer
components the full architecture calls for — see below.

## What still needs to run on your RTX 4060 (not done here, and why)
| Task | Why not run in the cloud sandbox |
|---|---|
| Contrastive pretraining (`pretrain_contrastive.py`), any dataset | Confirmed too slow on 2 CPU cores — didn't complete 1 epoch in 5 minutes |
| `contrastive_only` / `full` experiments, Dataset A | Depend on the contrastive checkpoint above |
| Any training on Dataset B (17,724 images) | ~24x the compute of Dataset A per epoch; infeasible on CPU here |
| YOLO detection training (Dataset C) | Object detection on CPU is very slow; needs `ultralytics` + GPU |
| Synthetic defect classifier training | Deferred to prioritize proving the core variety pipeline first; same CPU constraint applies |
| ImageNet-pretrained transfer learning (any model) | Weights host blocked from this sandbox; use `--pretrained` (default) locally |

Run `docs/07_RUNBOOK.md` steps 4-8 on your laptop to fill these in — they should each
take minutes rather than the many-minutes-per-epoch seen here, and will also get the
accuracy benefit of real ImageNet-pretrained weights and contrastive pretraining that
this cloud run deliberately skipped.

## Files delivered from this run
`outputs/checkpoints/variety_a_baseline_best.pt`,
`outputs/checkpoints/variety_a_attention_only_best.pt`,
`outputs/checkpoints/faiss_variety_a.faiss` + `.meta.json`,
`outputs/metrics/variety_a_baseline.json`, `outputs/metrics/variety_a_attention_only.json`,
`outputs/logs/variety_a_baseline.jsonl`, `outputs/logs/variety_a_attention_only.jsonl`.

You can start the backend right now and get real (if not-yet-fully-trained)
`/api/classify/variety` and `/api/similarity` results for Dataset A. `/api/detect`,
Dataset B classification, and the synthetic defect endpoint will correctly return
503 "model not available" until you run the remaining local training steps.
