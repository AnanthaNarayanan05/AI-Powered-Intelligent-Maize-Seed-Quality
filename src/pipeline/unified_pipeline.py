"""Phase 14: unified end-to-end analysis pipeline.

    input image -> validate -> YOLO detect -> (no seeds? early return) ->
    bounding boxes + count -> crop each seed -> for each crop:
        variety + quality classification (unified model; a/b kept for ablation)
        synthetic defect classification (clearly labeled synthetic)
        embedding + similarity search (same model's gallery index)
        Grad-CAM (best-effort, non-fatal if it fails)
    -> aggregate -> return structured, JSON-serializable result

Every model is loaded lazily and cached, and every stage degrades gracefully:
missing checkpoints raise ModelNotAvailableError which the backend turns into a
clear API error rather than a crash (Phase 14/29 requirement).
"""
from __future__ import annotations

import os
import statistics
import uuid
from dataclasses import dataclass, field

import numpy as np

from src.pipeline import resolution_gate
from src.registry import TaskUnavailableError, get_registry
from src.similarity.embedding_index import IndexEncoderMismatch
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
        self._registry = None
        self._eval_transform = None
        self._segmenters = {}

    @property
    def device(self):
        if self._device is None:
            self._device = get_device(self.cfg["device"]["prefer"])
        return self._device

    @property
    def registry(self):
        """configs/model_registry.yaml — which model answers which task."""
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    def _record_for(self, task: str):
        """The registry record that serves `task`, or a refusal.

        The refusal is the point. A task no model is allowed to serve arrives
        here as TaskUnavailableError and leaves as ModelNotAvailableError, which
        the backend already turns into a clear API error rather than a crash.
        That is a CAPABILITY refusal, and it is not the same fact as the
        resolution refusal in resolution_gate: one says the platform cannot
        answer this question at all, the other says it cannot answer it about
        this particular image. They are reported separately everywhere.
        """
        try:
            return self.registry.resolve(task)
        except TaskUnavailableError as e:
            raise ModelNotAvailableError(str(e)) from e

    def _checkpoint_for(self, task: str) -> str:
        """The checkpoint that serves `task`, resolved through the registry.

        Model choice does not live in this file. A path spelled out here would be
        a second answer to "which model serves this?" that drifts from the first
        the moment a model is renamed or retrained -- and, worse, would happily
        serve a task the registry deliberately refuses (the synthetic-defect
        models declare `serves: []` precisely so they cannot answer a
        real-world defect question). A task no model serves arrives here as
        TaskUnavailableError and leaves as ModelNotAvailableError, which the
        backend already turns into a clear API error rather than a crash.
        """
        return self._record_for(task).checkpoint

    # ---------- lazy loaders ----------
    def _get_eval_transform(self):
        if self._eval_transform is None:
            from src.contrastive.simclr import build_eval_transform

            self._eval_transform = build_eval_transform(self.cfg["training"]["image_size"])
        return self._eval_transform

    def _get_detection_model(self):
        if self._detection_model is None:
            weights_path = self._checkpoint_for("detect")
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
        # Deliberately a filesystem scan rather than a registry lookup. The
        # registry names the models the platform serves; these are ablation
        # variants (baseline / attention_only / contrastive_only) that exist only
        # to reproduce the comparison table, are never used to answer a user's
        # question, and would clutter the registry with entries whose whole point
        # is that they must not be served.
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

            # one record serves all three tasks; resolving any of them is the
            # same assertion that this model is the platform's declared answer
            ckpt_path = self._checkpoint_for("variety")
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
            # absence is a supported state here, not an error, so the record is
            # inspected rather than resolved -- resolve() raises on a missing
            # checkpoint and quality grading is designed to continue without the gate
            record = self.registry.get("quality_gate")
            if not record.available:
                self._quality_reference = False
            else:
                path = record.checkpoint
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

            # Fetched by key, never resolved by task: this model declares
            # serves: [] in the registry, so it can only be reached by a caller
            # that explicitly asked for it (run_synthetic_defect=True) and gets a
            # prediction stamped is_synthetic_model. registry.resolve() will not
            # hand it to anyone asking about real defects.
            record = self.registry.get("synthetic_defect_classifier")
            if not record.present:
                raise ModelNotAvailableError(
                    f"Synthetic defect model not found at {record.checkpoint}. "
                    f"Train it with: {record.trained_by}"
                )
            state = torch.load(record.checkpoint, map_location=self.device)
            model = CognitiveAttentionClassifier(
                num_classes=len(SYNTHETIC_CLASSES), backbone_name=state["backbone"],
                pretrained=False, use_attention=state["use_attention"],
            ).to(self.device)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            self._synthetic_model = (model, SYNTHETIC_CLASSES)
        return self._synthetic_model

    def _get_similarity_encoder(self, dataset: str):
        """The model whose feature space a similarity query lives in.

        This must be the same model that produced the prediction shown next to the
        neighbours, otherwise the UI puts two unrelated feature spaces side by side.
        """
        from src.similarity.embedding_index import encoder_id

        if dataset == "unified":
            model, _v, _q = self._get_unified_model()
            return model, encoder_id("unified")
        experiment = self._resolve_experiment(dataset, "full")
        model, _classes = self._get_variety_model(dataset, experiment)
        return model, encoder_id(dataset, experiment)

    def _get_faiss_index(self, dataset: str):
        if dataset not in self._faiss_indices:
            from src.similarity.embedding_index import EmbeddingIndex, index_path

            path = index_path(self.cfg, dataset)
            if not os.path.exists(path + ".faiss"):
                raise ModelNotAvailableError(
                    f"Similarity index for dataset '{dataset}' not found. "
                    f"Run: python -m src.similarity.build_index --dataset {dataset}"
                )
            model, encoder = self._get_similarity_encoder(dataset)
            # dim + encoder name are both checked: two different models can emit
            # embeddings of the same width, so width alone proves nothing.
            self._faiss_indices[dataset] = EmbeddingIndex.load(
                path, dim=model.feature_dim, encoder=encoder
            )
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

    # ---------- segmentation, resolution-gated ----------
    def _get_segmenter(self, key: str):
        """Load a segmentation checkpoint by key. Cached per key, like every other
        model here. Channel names, input size and the per-channel thresholds tuned
        on VAL all come out of the checkpoint -- none of them are assumed, because
        a segmenter retrained with a different channel set would otherwise be read
        with the previous run's meanings."""
        if key not in self._segmenters:
            import torch

            from src.segmentation.model import DefectSegmenter

            record = self.registry.get(key)
            if not record.present:
                raise ModelNotAvailableError(
                    f"Segmentation model not found at {record.checkpoint}. "
                    f"Train it with: {record.trained_by}"
                )
            state = torch.load(record.checkpoint, map_location=self.device, weights_only=False)
            model = DefectSegmenter(
                channel_names=state["channel_names"],
                backbone_name=state["backbone"],
                pretrained=False,
                use_attention=state["use_attention"],
            ).to(self.device)
            model.load_state_dict(state["model_state"])
            model.eval()
            self._segmenters[key] = (model, state)
        return self._segmenters[key]

    def segment_defects(self, crop_image, bbox=None, model_key: str | None = None) -> dict:
        """Pixel masks for one kernel -- or a documented refusal, never a guess.

        Two gates stand in front of the model, in this order:

          1. CAPABILITY. Without `model_key` the model is resolved by task, so a
             task the registry refuses to serve stops here. Passing `model_key`
             is the explicit opt-in used for ablation and tests; the result then
             carries `is_synthetic_model` and the checkpoint's own label note, so
             a synthetic-label mask can never be mistaken for a real one.
          2. RESOLUTION. Below the measured floor no mask is produced at all --
             not a low-confidence one, not a smaller one. The floor is an
             information-loss ceiling (see resolution_gate), so a mask under it
             would not be a worse measurement, it would be an unmeasured one.

        `masks` holds numpy uint8 planes at the CROP's own pixel size, keyed by
        channel name. They are arrays, not JSON: whatever serialises this response
        must summarise or encode them rather than pass them through.
        """
        if model_key is None:
            record = self._record_for("defect_segmentation")
        else:
            record = self.registry.get(model_key)

        kernel_px = (
            resolution_gate.kernel_px_from_bbox(bbox)
            if bbox is not None
            else resolution_gate.kernel_px_from_size(*crop_image.size)
        )
        verdict = resolution_gate.check(record, kernel_px, task="defect_segmentation")
        if not verdict.sufficient:
            return {
                "available": False,
                "reason": "insufficient_resolution",
                "message": verdict.message,
                "model": record.key,
                "resolution": verdict.as_dict(),
            }

        import cv2
        import torch

        from src.segmentation.dataset import IMAGENET_MEAN, IMAGENET_STD

        model, state = self._get_segmenter(record.key)
        size = int(state["image_size"])
        channels = list(state["channel_names"])
        # Thresholds were tuned on VAL per channel; 0.5 is only a fallback for a
        # checkpoint saved before that sweep existed.
        # float() rather than the stored numpy scalars: this dict travels out to
        # callers that serialise it, and np.float64 is not JSON.
        thresholds = {k: float(v) for k, v in (state.get("thresholds") or {}).items()}

        source = np.array(crop_image.convert("RGB"))
        h, w = source.shape[:2]
        resized = cv2.resize(source, (size, size), interpolation=cv2.INTER_AREA)
        x = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        x = x.unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(model(x))[0].cpu().numpy()

        masks = {}
        for ci, channel in enumerate(channels):
            plane = (probs[ci] >= float(thresholds.get(channel, 0.5))).astype(np.uint8)
            # Back to the crop's own pixels so the mask is measurable against the
            # image it was cut from; NEAREST because a resized mask must stay binary.
            masks[channel] = cv2.resize(plane, (w, h), interpolation=cv2.INTER_NEAREST)

        return {
            "available": True,
            "reason": None,
            "message": None,
            "model": record.key,
            "is_synthetic_model": record.label_provenance == "synthetic",
            "label_note": record.note,
            "channels": channels,
            "thresholds": thresholds,
            "resolution": verdict.as_dict(),
            "masks": masks,
        }

    def segmentation_status(self, kernel_px_values) -> dict:
        """What this image allows to be said about defect segmentation.

        Reported even when nothing ran, and especially then: two independent
        things can stop segmentation and a reader needs to know which. "No model
        is allowed to serve this task" is not fixable with a better photograph;
        "your kernels are 12 px across" is. Collapsing them into one "unavailable"
        would send people out to buy a camera for a capability that does not exist.

        The numbers here are measurements of the submitted image, not claims about
        a mask -- no mask is produced anywhere in this method.
        """
        floor = resolution_gate.declared_floor(self.registry, "defect_segmentation")
        minimum = floor["min_kernel_px"] if floor else None
        measured = sorted(int(k) for k in kernel_px_values)
        below = [k for k in measured if minimum is not None and k < minimum]

        resolution = {
            "min_kernel_px": minimum,
            "source": floor["source"] if floor else None,
            "seeds_measured": len(measured),
            "seeds_below_floor": len(below),
            "kernel_px_median": int(statistics.median(measured)) if measured else None,
            "kernel_px_min": measured[0] if measured else None,
            "kernel_px_max": measured[-1] if measured else None,
        }

        try:
            record = self._record_for("defect_segmentation")
        except ModelNotAvailableError as e:
            return {
                "status": "unavailable",
                "reason": "no_served_model",
                "message": str(e),
                "model": None,
                "resolution": resolution,
            }

        if measured and len(below) == len(measured):
            status, reason = "unavailable", "insufficient_resolution"
        elif below:
            status, reason = "partial", "insufficient_resolution"
        else:
            status, reason = "available", None

        return {
            "status": status,
            "reason": reason,
            "message": None if reason is None else resolution_gate.SEGMENTATION_UNAVAILABLE,
            "model": record.key,
            "resolution": resolution,
        }

    def embed_and_search(self, crop_image, dataset: str, top_k: int = 5):
        """Nearest gallery images in the encoder's feature space.

        VISUAL SIMILARITY only: a neighbour's label describes that neighbour, and is
        never merged into the query's own prediction.
        """
        import torch

        model, _encoder = self._get_similarity_encoder(dataset)
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
            result["segmentation"] = self.segmentation_status([])
            return result

        for i, det in enumerate(detections):
            crop = self.crop_seed(image, det["bbox"])
            seed_entry = {
                "seed_index": i,
                "bbox": det["bbox"],
                "detection_confidence": det["confidence"],
                # Measured, not requested: how many pixels across this kernel
                # actually is, by the same short-side rule the resolution floor
                # was measured with. Carried on every seed because it decides
                # per seed which pixel-level answers are available for it.
                "kernel_px": resolution_gate.kernel_px_from_bbox(det["bbox"]),
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
                # Search the gallery belonging to the model that just made the
                # prediction. There is deliberately no fallback to another dataset's
                # index: neighbours drawn from a 3-variety gallery cannot describe a
                # kernel the 6-variety model just classified.
                try:
                    seed_entry["similarity_results"] = self.embed_and_search(
                        crop, variety_dataset, top_k
                    )
                except (ModelNotAvailableError, IndexEncoderMismatch, KeyError) as e:
                    seed_entry["similarity_results"] = []
                    result["warnings"].append(f"Similarity unavailable: {e}")

            result["seeds"].append(seed_entry)

        # Stated whether or not anything pixel-level ran, and separating the two
        # reasons it might not have: no model is allowed to serve the task, or the
        # kernels in this image are below the measured floor. No mask is produced
        # here in either case.
        result["segmentation"] = self.segmentation_status(
            [s["kernel_px"] for s in result["seeds"]]
        )

        result["warnings"] = list(dict.fromkeys(result["warnings"]))  # dedupe, preserve order
        return result
