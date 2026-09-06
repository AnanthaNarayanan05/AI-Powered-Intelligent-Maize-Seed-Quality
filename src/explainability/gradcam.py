"""Grad-CAM over the POST-ATTENTION feature map, for whichever head is asked for.

IMPORTANT (per docs/05_ARCHITECTURE.md Phase 15 and the master prompt's mandatory
rules): this is an interpretability visualization, NOT a segmentation mask. It must
never be used to compute "defect area" or treated as ground-truth localization — see
docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md items 18/28/30. Every caller (backend
schemas, frontend labels) must keep this distinction in field names
(`gradcam_heatmap`, never `defect_mask`). Grad-CAM is a coarse 7x7 map upsampled to
image size; its blobs have no defect boundary in them.

WHICH FEATURE MAP. The map used is the output of `model.attention`, not of
`model.backbone`. The two differ exactly by the cognitive-attention block, which is
the part of this architecture the project is actually making a claim about — a CAM
taken before it explains a representation the classifier never sees. When attention
is disabled (the ablation baselines) the block is nn.Identity and the backbone output
is the post-attention map, so the target falls back to the backbone; hooking an
Identity module instead would register a backward hook on a module whose output IS
its input, which is not reliable.

WHICH HEAD. A CAM is only defined with respect to one scalar. The unified model has
two heads — variety and quality — and they do not attend to the same pixels, so the
head is a required part of the request and is reported back with the result. Asking
`UnifiedSeedModel` for `mode="classify"` returns a *tuple* of both heads' logits,
which is why this module never calls plain `forward()`.

NO HOOKS. Gradients come from torch.autograd.grad against the feature map returned by
the model's own `forward_with_features`. The previous implementation registered
forward/backward hooks and never removed them; because the pipeline caches model
instances, every Grad-CAM request left two more permanent hooks on a model that then
kept firing them during ordinary classification.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def target_layer_name(model) -> str:
    """The module whose activations the CAM is computed over, for reporting.

    Reported to the caller (and out through the API) rather than assumed, because
    "which layer" is the first question asked of any CAM figure and the answer
    differs between the attention and no-attention ablations.
    """
    return "backbone" if isinstance(getattr(model, "attention", None), nn.Identity) else "attention"


class GradCAM:
    """Grad-CAM for CognitiveAttentionClassifier and UnifiedSeedModel.

    Stateless with respect to the model: nothing is registered, nothing needs
    releasing, and the same instance can be reused or thrown away freely.
    """

    def __init__(self, model, head: str = "variety"):
        if not hasattr(model, "forward_with_features"):
            raise TypeError(
                f"{type(model).__name__} has no forward_with_features(); Grad-CAM needs "
                "the post-attention feature map alongside the logits."
            )
        self.model = model
        self.head = head
        self.target_layer = target_layer_name(model)

    def _forward(self, x: torch.Tensor):
        """Logits and feature map, tolerating both model signatures.

        UnifiedSeedModel takes a head argument; the single-task classifier has only
        one head and takes none. Inspecting the signature rather than the class keeps
        this working for any future model that follows the same contract.
        """
        import inspect

        fn = self.model.forward_with_features
        if "head" in inspect.signature(fn).parameters:
            return fn(x, head=self.head)
        if self.head not in ("variety", "classify", "default"):
            raise ValueError(
                f"{type(self.model).__name__} has a single head; head={self.head!r} "
                "cannot be explained by it."
            )
        return fn(x)

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
        out_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Return (cam in [0,1] at out_size or input size, class index explained).

        out_size is (height, width). Upsampling here rather than in the caller keeps
        the single bilinear resize on the low-resolution CAM, where it belongs: the
        alternative — resizing the *photograph* down to the CAM's grid — threw away
        the image detail that makes the overlay readable.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            logits, feats = self._forward(input_tensor)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            score = logits[:, class_idx].sum()
            grads = torch.autograd.grad(score, feats, retain_graph=False)[0]

            weights = grads.mean(dim=(2, 3), keepdim=True)   # gradient global-avg-pool
            cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
            size = out_size or tuple(input_tensor.shape[-2:])
            cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
        finally:
            self.model.train(was_training)

        span = float(cam.max() - cam.min())
        if span < 1e-8:
            # Every gradient cancelled: a genuinely flat map. Returning zeros is
            # correct; the old code divided by (span + 1e-8) and turned float noise
            # into a full-contrast heatmap of nothing.
            return np.zeros_like(cam), class_idx
        return (cam - cam.min()) / span, class_idx


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a Grad-CAM heatmap onto an RGB image (0-255 uint8) for display.

    The CAM is resized to the image rather than the other way round, so the overlay
    comes back at the resolution the user uploaded.
    """
    import cv2

    if cam.shape[:2] != image_rgb.shape[:2]:
        cam = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (image_rgb * (1 - alpha) + heatmap * alpha).astype(np.uint8)
