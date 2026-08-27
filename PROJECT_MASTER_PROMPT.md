# Project Master Prompt (verbatim, as given by the user)

> Preserved for reference. See `docs/01_RESOURCE_INVENTORY.md` through
> `docs/05_ARCHITECTURE.md` for how this prompt's assumptions were verified/corrected
> against the actual provided resources, and `docs/05_ARCHITECTURE.md` for the final
> adopted scope.

## Title
AI-Powered Intelligent Maize Seed Quality, Defect, Variety Recognition and Detection
System Using Contrastive Learning and Cognitive Attention

## Goal
Build a complete end-to-end AI-powered maize seed analysis system using the provided
datasets to train real ML/CV models, integrated into a functional application that
only exposes capabilities genuinely supported by the trained models and datasets.

## Mandatory rules (kept in force for this project)
1. Do not fake results. 2. Do not invent labels. 3. Do not invent annotations.
4. Do not modify original datasets. 5. Use processed copies when required.
6. Do not blindly merge datasets. 7. Do not claim segmentation without masks.
8. Do not claim exact defect area without genuine masks. 9. Do not use Grad-CAM as a
defect mask. 10. Do not claim exact defect localization from image-level labels.
11. Do not claim severity without severity labels. 12. Do not claim foreign-object
detection without labelled data. 13. Do not claim purity estimation without
sufficient valid data. 14. Do not claim true seed-lot analysis without seed-lot
metadata. 15. Clearly identify limitations. 16. Keep models modular. 17. Use
reproducible seeds. 18. Save checkpoints. 19. Save training metrics. 20. Save
evaluation results. 21. Use real inference in the final application. 22. Do not
hard-code fake UI values. 23. Do not hard-code GEMINI_API_KEY. 24. Never expose API
keys to the frontend. 25. Never log secrets. 26. Core ML functionality must work
without Gemini. 27. Gemini must explain verified ML results, not replace them.
28. Gemini must not hallucinate unsupported project capabilities. 29. Handle errors
gracefully. 30. Test modules before declaring completion. 31. Prefer working
functionality over unnecessary complexity. 32. Do not stop after generating code.
33. Run, test, debug and integrate the project. 34. Preserve all original datasets.

## Gemini security rules
- Read only from `GEMINI_API_KEY` in a local `.env` file (never hardcode, print,
  log, return, or expose to the frontend).
- `.env` in `.gitignore`; `.env.example` contains only `GEMINI_API_KEY=`.
- All Gemini calls go Frontend -> Backend -> Gemini API; frontend never calls Gemini
  directly.
- If Gemini fails (missing key, timeout, rate limit, invalid response): do not crash;
  preserve and return ML results normally; clearly show "AI explanation temporarily
  unavailable"; log only safe diagnostic info; allow retry.

## Layers
1. Computer Vision and ML Models
2. Unified Analysis Pipeline
3. Database and Analysis History
4. Backend REST API
5. Gemini AI Intelligence Layer
6. Frontend

## Suggested API surface (adapted to actual final architecture)
`POST /api/analyze/image`, `POST /api/analyze/batch`, `POST /api/detect`,
`POST /api/classify/quality` *(dropped — no data)*, `POST /api/classify/variety`,
`POST /api/similarity`, `GET /api/history`, `GET /api/history/{analysis_id}`,
`POST /api/copilot/chat`, `POST /api/copilot/explain-analysis`,
`POST /api/copilot/summarize-batch`, `POST /api/copilot/compare`.

## Gemini functionalities to implement
A. AI Analysis Explanation — B. Interactive Seed Analysis Copilot —
C. Batch Analysis Intelligence — D. Analysis Comparison — E. Result Interpretation —
F. Limitation-Aware Copilot (must respect the functionality coverage matrix and never
claim unsupported capabilities like defect detection, segmentation, or purity).

## Frontend requirement
React + Vite (or Streamlit if significantly more reliable given time), communicating
only with the FastAPI backend; no hard-coded predictions; Gemini output always
labeled "AI-generated explanation based on model analysis."

## Original dataset descriptions given (see docs/01 for what was actually verified)
- Dataset 1 was described as maize seed quality/defect. **Verified as a corn-variety
  dataset instead (Chulpi Cancha/Indurata/Rugosa, 1050 images) — no defect labels
  exist.**
- Dataset 2 was described as maize variety (~17,724 images; Bhihilifa/SanzalSima/
  WangDataa). **Verified as exactly this.**
- Dataset 3 was described as YOLO object detection. **Verified: 1 class ("Corn"),
  1251 images, 42,602 boxes, pre-split by Roboflow.**

## Implementation order (as given; adapted where Dataset 1 quality/defect steps do
not apply — see docs/05_ARCHITECTURE.md "Next phase")
Inspect resources -> analyze literature -> audit each dataset -> validate -> 
cross-dataset compatibility -> functionality coverage matrix -> finalize
architecture -> project structure -> preprocessing -> [quality/defect training —
SKIPPED, no data] -> contrastive pipeline -> pretrain encoder -> cognitive attention
-> fine-tune variety classifier(s) -> evaluate -> embeddings -> FAISS index -> YOLO
training pipeline -> train/evaluate detection -> individual seed cropping -> unified
inference pipeline -> Grad-CAM/explainability -> batch analysis -> database/history
-> FastAPI backend -> Gemini integration -> test endpoints -> frontend -> integrate
-> end-to-end test -> debug -> verify real inference everywhere -> verify core ML
works without Gemini -> verify Gemini only interprets verified results.
