# Resource Inventory (Phase 1)

Verified by direct inspection on 2026-08-19. This document is the source of truth;
it supersedes any dataset description in the master prompt where they conflict.

## Files provided

| File | Size | Verified contents |
|---|---|---|
| `seed_dataset1.zip` | 8.3 MB | `Corn_3_Classes_Image_Dataset/` — 3 folders: `Zea_mays_Chulpi_Cancha` (350), `Zea_mays_Indurata` (350), `Zea_mays_Rugosa` (350). 1050 JPGs, 400x400px. Includes `Explanation_Citation_Request.txt` citing 3 published papers (Avuçlu & Köklü 2025 in *Journal of Food Science*; Avuçlu, Taşdemir & Köklü 2023; Yasin et al. 2025). This is a **corn variety** image set, not a defect/quality set. |
| `seed_dataset2.zip` | 58.0 MB | `MaizeData/` — 3 folders: `Bhihilifa` (6480), `SanzalSima` (5100), `WangDataa` (6144) = **17,724 images exactly**, 224x224px JPG. No README/citation file included. Matches the master prompt's "Dataset 2" description (class names, exact count). |
| `Grain and Objects Detection.v1i.yolov11.zip` | 39.6 MB | Roboflow export, YOLOv11 format. `data.yaml`: `nc: 1`, `names: ['Corn']`. Pre-split: train 913 / valid 256 / test 82 images, all with matching label files. Single-class seed/grain bounding-box detection only. |
| `Literarture Survey - Sheet1.pdf` | 186 KB | Spreadsheet of 48 summarized research papers on seed/maize quality, variety and defect classification, compiled by 3 student authors (Delna, Anantha, Ashik). Real, citable, contains links. |

## Critical discrepancy vs. the master prompt

The master prompt assumes:
- Dataset 1 = quality/defect dataset (healthy/defective, crack, fungal, etc.)
- Dataset 2 = variety dataset, ~17,724 images, classes Bhihilifa/SanzalSima/WangDataa
- Dataset 3 = YOLO detection dataset

**Actual evidence:**
- No dataset among the 3 provided contains healthy/defective, crack, fungal, mold,
  discoloration, or severity labels. **A genuine quality/defect dataset does not exist
  in the provided resources.**
- `seed_dataset1.zip` is a second, independent **variety** dataset (Chulpi Cancha /
  Indurata / Rugosa), not quality/defect.
- `seed_dataset2.zip` is exactly the variety dataset the master prompt describes as
  "Dataset 2" (Bhihilifa/SanzalSima/WangDataa, 17,724 images).
- The detection dataset has exactly 1 class, `Corn` — matches the prompt's fallback
  scenario ("if the dataset contains only one class such as Corn...").

**Resolution adopted (pending your correction):** the project is built around the
data that actually exists — two independent variety-classification datasets and one
single-class detection dataset. No quality/defect classifier is trained, and none of
the UI/API/Gemini layers claim defect detection, since no genuine labels exist for it.
This is documented, not hidden, per the project's own mandatory rule "never invent
labels." If a real defect/quality dataset is supplied later, Phase 10 can be added
without disrupting the rest of the architecture (see `03_ARCHITECTURE.md`).
