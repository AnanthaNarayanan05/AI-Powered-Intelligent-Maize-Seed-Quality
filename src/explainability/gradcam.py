"""Grad-CAM for CognitiveAttentionClassifier. Produces a heatmap over the LAST
CONV FEATURE MAP (after attention) showing which regions influenced the prediction.

IMPORTANT (per docs/05_ARCHITECTURE.md Phase 15 and the master prompt's mandatory
rules): this is an interpretability visualization, NOT a segmentation mask. It must
never be used to compute "defect area" or treated as ground-truth localization — see
docs/04_FUNCTIONALITY_COVERAGE_MATRIX.md items 18/28/30. Every caller (backend
schemas, frontend labels) must keep this distinction in field names
(`gradcam_heatmap`, never `defect_mask`).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer_name: str = "backbone"):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Hook the backbone's final conv output (before attention/pool)
        target = self.model.backbone
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        self.model.eval()
        logits = self.model(input_tensor, mode="classify")
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients  # (B, C, H, W)
        activations = self.activations  # (B, C, H, W)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # global-avg-pool gradients
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlays a Grad-CAM heatmap onto an RGB image (0-255 uint8) for display."""
    import cv2

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlaid = (image_rgb * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return overlaid
