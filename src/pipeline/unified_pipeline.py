"""Phase 14: unified end-to-end analysis pipeline.

    input image -> validate -> YOLO detect -> (no seeds? early return) ->
    bounding boxes + count -> crop each seed -> for each crop:
        variety classification (dataset A or B model, selectable)
        synthetic defect classification (clearly labeled synthetic)
        embedding + similarity search
        Grad-CAM (best-effort, non-fatal if it fails)
    -> aggregate -> return structured, JSON-serializable result

Every model is loaded lazily and cached, and every stage degrades gracefully:
missing checkpoints raise ModelNotAvailableError which the backend turns into a
clear API error rather than a crash (Phase 14/29 requirement).
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field

import numpy as np

from src.utils.config import load_config, get_device
from src.utils.logging_utils import get_logger

logger = get_logger("unified_pipeline")


class ModelNotAvailableError(Exception):
    """Raised when a required checkpoint has not been trained/placed yet."""


class InvalidImageError(Exception):
    pass


@dataclass
class SeedResult:
    seed_index: int
    bbox: list  # [x1, y1, x2, y2] in original image pixel coords
    detection_confidence: float
    variety_prediction: dict | None = None
    synthetic_defect_prediction: dict | None = None
    similarity_results: list = field(default_factory=list)
    gradcam_available: bool = False


class AnalysisPipeline:
    def __init__(self, config_path: str | None = None):
        self.cfg = load_config(config_path)
        self._device = None  # resolved lazily — torch may not be installed until model load time
        self._detection_model = None
        self._variety_models: dict[str, object] = {}
        self._unified_model = None
        self._quality_reference = None
        self._synthetic_model = None
        self._faiss_indices: dict[str, object] = {}
        self._eval_transform = None

    @property
    def device(self):
        if self._device is None:
            self._device = get_device(self.cfg["device"]["prefer"])
        return self._device

    # ---------- lazy loaders ----------
    def _get_eval_transform(self):
        if self._eval_transform is None:
            from src.contrastive.simclr import build_eval_transform

            self._eval_transform = build_eval_transform(self.cfg["training"]["image_size"])
        return self._eval_transform

    def _get_detection_model(self):
        if self._detection_model is None:
            weights_path = os.path.join(self.cfg["paths"]["checkpoints"], "detection_corn", "weights", "best.pt")
            if not os.path.exists(weights_path):
                raise ModelNotAvailableError(
                    f"Detection model not found at {weights_path}. Run src/training/train_detection.py locally first."
                )
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise ModelNotAvailableError(
                    "ultralytics is not installed in this environment. Run: pip install ultralytics"
                ) from e

            self._detection_model = YOLO(weights_path)
        return self._detection_model

    EXPERIMENT_PREFERENCE = ["full", "contrastive_only", "attention_only", "baseline"]

    def _resolve_experiment(self, dataset: str, requested: str = "full") -> str:
        """Not every experiment variant is necessarily trained yet (e.g. a partial
        local run, or the reduced-scale verification run done in the cloud sandbox
        without GPU access). Rather than hard-fail whenever the specific
        'full' checkpoint is missing, fall back through the other trained
        experiments in order of research quality, and say clearly which one was
        actually used instead of mislabeling results."""
        candidates = [requested] + [e for e in self.EXPERIMENT_PREFERENCE if e != requested]
        for exp in candidates:
            ckpt_path = os.path.join(self.cfg["paths"]["checkpoints"], f"variety_{dataset}_{exp}_best.pt")
            if os.path.exists(ckpt_path):
                return exp
        raise ModelNotAvailableError(
            f"No trained variety model found for dataset '{dataset}' (checked {candidates}). "
            f"Run src/training/train_variety.py --dataset {dataset} locally first."
        )

    def _get_variety_model(self, dataset: str, experiment: str = "full"):
        experiment = self._resolve_experiment(dataset, experiment)
        key = f"{dataset}_{experiment}"
        if key not in self._variety_models:
            try:
                import torch
                from src.models.variety_classifier import CognitiveAttentionClassifier
            except ImportError as e:
                raise ModelNotAvailableError(
                    "torch/torchvision is not installed in this environment. Run: pip install torch torchvision"
                ) from e

            ckpt_path = os.path.join(self.cfg["paths"]["checkpoints"], f"variety_{dataset}_{experiment}_best.pt")
            if not os.path.exists(ckpt_path):
                raise ModelNotAvailableError(
                    f"Variety model for dataset '{dataset}' not found at {ckpt_path}. "
                    f"Run src/training/train_variety.py --dataset {dataset} locally first."
                )
            state = torch.load(ckpt_path, map_location=self.device)
            model = CognitiveAttentionClassifier(
                num_classes=len(state["classes"]), backbone_name=state["backbone"],
                pretrained=False, use_attention=state["use_attention"],
            ).to(self.device)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            self._variety_models[key] = (model, state["classes"])
        return self._variety_models[key]

    def _get_unified_model(self):
        """The single model that serves the platform: one trunk, a 6-class variety
        head and a 2-class quality head. Replaces the former pair of per-dataset
        variety models plus the synthetic defect classifier."""
        if self._unified_model is None:
            try:
                import torch
                from src.models.unified_model import UnifiedSeedModel
            except ImportError as e:
                raise ModelNotAvailableError(
                    "torch/torchvision is not installed in this environment. "
                    "Run: pip install torch torchvision"
                ) from e

            ckpt_path = os.path.join(self.cfg["paths"]["checkpoints"], "unified_seed_model_best.pt")
            if not os.path.exists(ckpt_path):
                raise ModelNotAvailableError(
                    f"Unified seed model not found at {ckpt_path}. "
                    f"Run: python -m src.training.train_unified "
                    f"--encoder outputs/checkpoints/contrastive_encoder_unified_unified.pt"
                )
            state = torch.load(ckpt_path, map_location=self.device)
            model = UnifiedSeedModel(
                state["variety_classes"], state["quality_classes"],
                backbone_name=self.cfg["backbone"], pretrained=False,
                use_attention=state.get("use_attention", True),
            ).to(self.device)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            self._unified_model = (model, state["variety_classes"], state["quality_classes"])
        return self._unified_model

    def _get_quality_reference(self):
        """Embeddings of the Mendeley training split plus the calibrated distance
        threshold, used to tell interpolation from extrapolation. Returns None if
        the reference was never built, in which case quality is still returned but
        carries no in-distribution claim."""
        if self._quality_reference is None:
            path = os.path.join(self.cfg["paths"]["checkpoints"], "quality_reference.npz")
            if not os.path.exists(path):
                self._quality_reference = False
            else:
                z = np.load(path)
                self._quality_reference = (z["reference"], float(z["threshold"]), int(z["k"]))
        return self._quality_reference or None

    def classify_unified(self, crop_image):
        """One forward pass, both predictions. Returns (variety, quality) dicts."""
        import torch

        model, v_classes, q_classes = self._get_unified_model()
        tensor = self._get_eval_transform()(crop_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = model(tensor, mode="embedding")
            v_logits = model.variety_head(embedding)
            q_logits = model.quality_head(embedding)
            v_probs = torch.softmax(v_logits, dim=1).cpu().numpy()[0]
            q_probs = torch.softmax(q_logits, dim=1).cpu().numpy()[0]
            emb = embedding.cpu().numpy()[0]

        vi, qi = int(np.argmax(v_probs)), int(np.argmax(q_probs))
        variety = {
            "predicted_class": v_classes[vi],
            "confidence": round(float(v_probs[vi]), 4),
            "class_probabilities": {c: round(float(p), 4) for c, p in zip(v_classes, v_probs)},
            "model": "unified_seed_model",
            "is_synthetic_model": False,
        }
        quality = {
            "predicted_class": q_classes[qi],
            "confidence": round(float(q_probs[qi]), 4),
            "class_probabilities": {c: round(float(p), 4) for c, p in zip(q_classes, q_probs)},
            "model": "unified_seed_model_quality",
            "is_synthetic_model": False,
            "label_source": "Mendeley EfficientMaize expert-assigned Good/Bad kernel labels",
        }

        # Softmax confidence does not detect extrapolation -- it runs HIGHER on some
        # out-of-distribution imagery than on real test data. Distance to the
        # training representation does. See src/analysis/build_quality_reference.py.
        ref = self._get_quality_reference()
        if ref:
            reference, threshold, k = ref
            e = emb / (np.linalg.norm(emb) + 1e-8)
            sims = reference @ e
            dist = float(1.0 - np.partition(sims, -k)[-k:].mean())
            quality["distribution_distance"] = round(dist, 4)
            quality["distribution_threshold"] = round(threshold, 4)
            quality["out_of_distribution"] = bool(dist > threshold)
            if quality["out_of_distribution"]:
                quality["caveat"] = (
                    "This kernel does not resemble the images the quality model was "
                    "trained and validated on, so its grade is an extrapolation rather "
                    "than a measurement. Treat it as unverified."
                )
        return variety, quality

    def _get_synthetic_model(self):
        if self._synthetic_model is None:
            try:
                import torch
                from src.models.variety_classifier import CognitiveAttentionClassifier
            except ImportError as e:
                raise ModelNotAvailableError(
                    "torch/torchvision is not installed in this environment. Run: pip install torch torchvision"
                ) from e
            from src.data.synthetic_defect_generator import SYNTHETIC_CLASSES

            ckpt_path = os.path.join(self.cfg["paths"]["checkpoints"], "synthetic_defect_classifier_best.pt")
            if not os.path.exists(ckpt_path):
                raise ModelNotAvailableError(
                    f"Synthetic defect model not found at {ckpt_path}. "
                    f"Run src/training/train_synthetic_defect.py locally first."
                )
            state = torch.load(ckpt_path, map_location=self.device)
            model = CognitiveAttentionClassifier(
                num_classes=len(SYNTHETIC_CLASSES), backbone_name=state["backbone"],
                pretrained=False, use_attention=state["use_attention"],
            ).to(self.device)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            self._synthetic_model = (model, SYNTHETIC_CLASSES)
        return self._synthetic_model

    def _get_faiss_index(self, dataset: str):
        if dataset not in self._faiss_indices:
            from src.similarity.embedding_index import EmbeddingIndex

            path = self.cfg["paths"][f"faiss_index_{dataset}"].rsplit(".index", 1)[0]
            if not os.path.exists(path + ".faiss"):
                raise ModelNotAvailableError(
                    f"Similarity index for dataset '{dataset}' not found. Run src/similarity/build_index.py locally first."
                )
            model, _classes = self._get_variety_model(dataset)
            self._faiss_indices[dataset] = EmbeddingIndex.load(path, dim=model.feature_dim)
        return self._faiss_indices[dataset]

    # ---------- stages ----------
    def load_and_validate_image(self, image_path: str):
        from PIL import Image

        if not os.path.exists(image_path):
            raise InvalidImageError(f"File not found: {image_path}")
        try:
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            raise InvalidImageError(f"Unreadable or unsupported image: {e}") from e
        return img

    def detect_seeds(self, image_path: str, conf_threshold: float | None = None):
        model = self._get_detection_model()
        conf = conf_threshold or self.cfg["detection"]["conf_threshold"]
        # Explicit NMS IoU rather than Ultralytics' 0.70 default — see the
        # detection.nms_iou note in configs/config.yaml for the measurements.
        iou = self.cfg["detection"].get("nms_iou", 0.40)
        results = model.predict(image_path, conf=conf, iou=iou, verbose=False)
        r = results[0]
        boxes = []
        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            boxes.append({"bbox": [round(v, 1) for v in xyxy], "confidence": round(confidence, 4)})
        return boxes

    def crop_seed(self, image, bbox: list, context_pad: float | None = None):
        """Crop a detected seed, expanding the box by a fraction of its own size.

        The variety classifier is trained on whole dataset images, where the seed sits
        in surrounding context. A tight YOLO box discards that context and shifts the
        input distribution away from anything the model saw in training — measured on
        the Dataset A held-out test split, a zero-padding crop scores 77.63% against
        100.00% for the whole image. Restoring ~30% context closes the gap entirely.
        See docs/09_PROJECT_STATUS_REPORT.md section 6.6.
        """
        if context_pad is None:
            context_pad = self.cfg.get("detection", {}).get("crop_context_pad", 0.0)

        x1, y1, x2, y2 = [float(v) for v in bbox]
        if context_pad > 0:
            dx = (x2 - x1) * context_pad / 2.0
            dy = (y2 - y1) * context_pad / 2.0
            x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy

        width, height = image.size
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(width, int(round(x2)))
        y2 = min(height, int(round(y2)))
        if x2 <= x1 or y2 <= y1:  # degenerate box — fall back to the raw detection
            x1, y1, x2, y2 = [int(v) for v in bbox]
        return image.crop((x1, y1, x2, y2))

    def classify_variety(self, crop_image, dataset: str, experiment: str = "full"):
        import torch

        resolved_experiment = self._resolve_experiment(dataset, experiment)
        model, classes = self._get_variety_model(dataset, resolved_experiment)
        tf = self._get_eval_transform()
        tensor = tf(crop_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(tensor, mode="classify")
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        return {
            "predicted_class": classes[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "class_probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
            "model": f"variety_{dataset}_{resolved_experiment}",
            "requested_experiment": experiment,
        }

    def classify_synthetic_defect(self, crop_image):
        import torch

        model, classes = self._get_synthetic_model()
        tf = self._get_eval_transform()
        tensor = tf(crop_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(tensor, mode="classify")
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        return {
            "predicted_class": classes[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "class_probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
            "model": "synthetic_defect_classifier",
            "is_synthetic_model": True,
            "disclaimer": "Trained on procedurally generated synthetic damage patterns, "
                           "not verified real-world plant pathology data. See docs/06_SYNTHETIC_DEFECT_POLICY.md.",
        }

    def embed_and_search(self, crop_image, dataset: str, top_k: int = 5):
        import torch

        model, _classes = self._get_variety_model(dataset)
        tf = self._get_eval_transform()
        tensor = tf(crop_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = model(tensor, mode="embedding").cpu().numpy()[0]
        index = self._get_faiss_index(dataset)
        return index.search(embedding, top_k)

    GRADCAM_HEADS = ("variety", "quality")

    def generate_gradcam(
        self,
        crop_image,
        dataset: str = "unified",
        experiment: str = "full",
        head: str = "variety",
        out_size: tuple[int, int] | None = None,
    ):
        """Grad-CAM for the model that actually produced the served prediction.

        Returns (cam, meta). `dataset` defaults to "unified" for the same reason
        analyze_image does: the unified model is what the platform serves, and a
        heatmap taken from variety_a_full_best.pt would be explaining a *different*
        model's decision than the one displayed beside it. "a"/"b" stay reachable so
        the ablation figures in docs/09 remain reproducible.

        The quality head is only available on the unified model — the per-dataset
        checkpoints have no quality head to differentiate, so asking for one is an
        error rather than a silent fall back to variety.
        """
        from src.explainability.gradcam import GradCAM, target_layer_name

        if head not in self.GRADCAM_HEADS:
            raise ValueError(f"head must be one of {self.GRADCAM_HEADS}, got {head!r}")

        if dataset == "unified":
            model, classes, quality_classes = self._get_unified_model()
            classes = classes if head == "variety" else quality_classes
            model_name = "unified_seed_model" if head == "variety" else "unified_seed_model_quality"
            experiment_used = None
        else:
            if head != "variety":
                raise ValueError(
                    f"the '{dataset}' variety checkpoint has only a variety head; "
                    f"head={head!r} is not available on it. Use dataset='unified'."
                )
            experiment_used = self._resolve_experiment(dataset, experiment)
            model, classes = self._get_variety_model(dataset, experiment_used)
            model_name = f"variety_{dataset}_{experiment_used}"

        tensor = self._get_eval_transform()(crop_image).unsqueeze(0).to(self.device)
        cam, class_idx = GradCAM(model, head=head).generate(tensor, out_size=out_size)

        meta = {
            "model": model_name,
            "head": head,
            "target_layer": target_layer_name(model),
            "predicted_class": classes[class_idx],
            "class_index": class_idx,
            "experiment": experiment_used,
            "is_segmentation_mask": False,
            "note": (
                "Grad-CAM attention heatmap over the post-attention feature map. It "
                "shows which regions influenced this prediction. It is NOT a "
                "segmentation mask and NOT a defect area measurement."
            ),
        }
        return cam, meta

    # ---------- orchestration ----------
    def analyze_image(self, image_path: str, variety_dataset: str = "unified", run_similarity: bool = True, run_synthetic_defect: bool = False, top_k: int = 5) -> dict:
        """variety_dataset defaults to "unified": one model, six varieties, plus a
        real quality head. "a"/"b" still select the older single-dataset models so
        the ablation results in docs/09 remain reproducible, but they are not what
        the platform serves."""
        image = self.load_and_validate_image(image_path)
        detections = self.detect_seeds(image_path)

        result = {
            "analysis_id": str(uuid.uuid4()),
            "image_path": image_path,
            "seed_count": len(detections),
            "seeds": [],
            "warnings": [],
        }

        if len(detections) == 0:
            result["warnings"].append("No seeds detected above the confidence threshold.")
            return result

        for i, det in enumerate(detections):
            crop = self.crop_seed(image, det["bbox"])
            seed_entry = {
                "seed_index": i,
                "bbox": det["bbox"],
                "detection_confidence": det["confidence"],
            }
            if variety_dataset == "unified":
                try:
                    variety, quality = self.classify_unified(crop)
                    seed_entry["variety_prediction"] = variety
                    seed_entry["quality_prediction"] = quality
                except ModelNotAvailableError as e:
                    seed_entry["variety_prediction"] = None
                    seed_entry["quality_prediction"] = None
                    result["warnings"].append(str(e))
            else:
                try:
                    seed_entry["variety_prediction"] = self.classify_variety(crop, variety_dataset)
                except ModelNotAvailableError as e:
                    seed_entry["variety_prediction"] = None
                    result["warnings"].append(str(e))

            if run_synthetic_defect:
                try:
                    seed_entry["synthetic_defect_prediction"] = self.classify_synthetic_defect(crop)
                except ModelNotAvailableError as e:
                    seed_entry["synthetic_defect_prediction"] = None
                    result["warnings"].append(str(e))

            if run_similarity:
                try:
                    # FAISS indices were built per source dataset; the unified model
                    # has no index of its own yet, so similarity falls back to A.
                    sim_ds = "a" if variety_dataset == "unified" else variety_dataset
                    seed_entry["similarity_results"] = self.embed_and_search(crop, sim_ds, top_k)
                except ModelNotAvailableError as e:
                    seed_entry["similarity_results"] = []
                    result["warnings"].append(str(e))

            result["seeds"].append(seed_entry)

        result["warnings"] = list(dict.fromkeys(result["warnings"]))  # dedupe, preserve order
        return result
