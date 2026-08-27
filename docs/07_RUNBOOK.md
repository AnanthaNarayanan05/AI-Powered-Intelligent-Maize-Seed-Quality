# Runbook — how to actually run this project locally (RTX 4060 laptop)

Every step below has now been executed end-to-end on the target machine (Windows 11,
RTX 4060 Laptop 8 GB, CUDA 13.1 driver). See `docs/11_LOCAL_TRAINING_RUN_LOG.md` for
the actual results and the bugs that run surfaced.

## 0. One-time setup
```
cd FINALYEARPROJECT
python -m venv .venv
.venv\Scripts\activate                                 # Windows
```

Install a CUDA build of torch FIRST — plain `pip install torch` gets the CPU-only
wheel and training silently falls back to CPU (~18x slower here):
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
Confirm the GPU is actually visible before training anything:
```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
This must print `True NVIDIA GeForce RTX 4060 Laptop GPU`. If it prints `False`, the
CPU wheel got installed — uninstall torch/torchvision and redo the cu121 line above.

Then set the Gemini key:
```
copy .env.example .env
# edit .env and put your real key: GEMINI_API_KEY=xxxxx
```

### Windows notes (these cost real debugging time — don't skip)
- **Python** must be a real python.org/winget install. The Microsoft Store `python`
  stub on a clean Windows box is not a working interpreter.
- **`albumentations` was removed from requirements.txt.** Nothing in the codebase
  imports it, and its `stringzilla` dependency needs MSVC C++ Build Tools to compile
  on Windows — it blocked the entire install for no benefit.
- **Node.js** is required for the frontend only (`winget install OpenJS.NodeJS.LTS`).

## 1. Extract the datasets (never overwrite the original zips)
```
Expand-Archive seed_dataset1.zip -DestinationPath datasets\dataset_variety_a_corntype
Expand-Archive seed_dataset2.zip -DestinationPath datasets\dataset_2_variety
Expand-Archive "Grain and Objects Detection.v1i.yolov11.zip" -DestinationPath datasets\dataset_3_detection
```
After this, `configs/config.yaml`'s `dataset_a`/`dataset_b`/`dataset_c` paths should
resolve. If your folder names differ slightly, edit `configs/config.yaml`'s `paths`
section to match.

## 2. Audit (fast, already verified logic — re-run to confirm your local copy)
```
python -m src.data.validate_dataset --root datasets/dataset_variety_a_corntype/Corn_3_Classes_Image_Dataset --out data_processed/audit_a.json
python -m src.data.validate_dataset --root datasets/dataset_2_variety/MaizeData --out data_processed/audit_b.json
python -m src.data.validate_detection_dataset --root datasets/dataset_3_detection --out data_processed/audit_c.json
```

## 3. Generate the synthetic defect dataset (see docs/06_SYNTHETIC_DEFECT_POLICY.md)
```
python -m src.data.synthetic_defect_generator ^
  --source-root datasets/dataset_variety_a_corntype/Corn_3_Classes_Image_Dataset ^
  --out-root data_processed/synthetic_defects ^
  --manifest data_processed/manifest_synthetic_defects.csv
```

## 4. Contrastive pretraining (Phase 8), per dataset
```
python -m src.training.pretrain_contrastive --dataset a
python -m src.training.pretrain_contrastive --dataset b
```

## 5. Variety classifier training — run all 4 ablation experiments per dataset
```
python -m src.training.train_variety --dataset a --experiment baseline
python -m src.training.train_variety --dataset a --experiment attention_only
python -m src.training.train_variety --dataset a --experiment contrastive_only
python -m src.training.train_variety --dataset a --experiment full
# repeat with --dataset b
```
Each run writes: `outputs/checkpoints/variety_<dataset>_<experiment>_best.pt`,
`outputs/logs/variety_<dataset>_<experiment>.jsonl`,
`outputs/metrics/variety_<dataset>_<experiment>.json` (accuracy, precision/recall/F1,
confusion matrix).

## 6. Synthetic defect classifier
```
python -m src.training.train_synthetic_defect
```

## 7. Detection model
```
pip install ultralytics
python -m src.training.train_detection
```

## 8. Similarity index (after step 5's "full" experiment finishes for a dataset)
```
python -m src.similarity.build_index --dataset a
python -m src.similarity.build_index --dataset b
```

## 9. Backend
```
uvicorn backend.main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for interactive Swagger docs of every endpoint.

## 10. Frontend
```
cd frontend
npm install
npm run dev
```
Visit the printed local URL (default `http://localhost:5173`).

## 11. Ablation comparison (after step 5 finishes for both datasets)
```
python -m src.analysis.compare_experiments
```
Writes `docs/10_RESULTS_COMPARISON.md` — the baseline / +attention / +contrastive /
+both table used in the final report and viva.

## 12. Verify the whole thing
```
python -m pytest tests/ -v
```
`tests/test_integration_e2e.py` runs the real pipeline against the real checkpoints:
detection boxes, calibrated variety probabilities, the predicted variety matching the
source folder, the synthetic disclaimer being present, nearest-first similarity
ordering, Grad-CAM range, and the full `/api/analyze/image` chain persisting to
history. Those tests **skip** (not fail) if a checkpoint is missing, so a green run
before training does not mean the pipeline works — check for `skipped` in the output.

## Verified on the local GPU machine
All of the above has been run to completion. Results, timings, and the four bugs the
local run exposed are recorded in `docs/11_LOCAL_TRAINING_RUN_LOG.md`.
