"""Phase 17: is the unified model's confidence number telling the truth?

Accuracy and calibration are different claims. A head can be 89% accurate and
still be wrong about its own certainty -- reporting 95% confidence on
predictions that are actually right 80% of the time, say. `SHOW_CONFIDENCE` has
been hard-disabled project-wide (frontend/src/components/ui/index.jsx) because
nobody had measured which of those this project's variety and quality heads do.
This script measures it, on the two splits that make the measurement honest:

  1. Reliability diagrams, Expected Calibration Error (ECE) and Brier score for
     the RAW softmax output of both heads, on validation AND on test. This is
     the baseline claim: "is confidence == accuracy already?" Answered without
     touching test more than this one read.
  2. Temperature scaling (Guo et al. 2017): one scalar T per head, fit on
     validation logits only by minimising NLL. T>1 softens over-confident
     probabilities; T<1 sharpens under-confident ones. It cannot change which
     class wins -- dividing every logit by the same positive T is a monotonic
     rescaling, so argmax is invariant and accuracy cannot move. Only the
     number attached to the prediction can.
  3. The calibrated probabilities are then evaluated on test EXACTLY ONCE,
     because a temperature tuned to look good on the split it's tested against
     is not a calibration, it's an overfit.

The verdict this script writes to outputs/checkpoints/unified_calibration.json
is read by src/pipeline/unified_pipeline.py at serve time if present: it is the
only thing that may divide a served logit by a number before softmax, and it is
never invented -- if this file does not exist, or does not name a task, that
task serves raw softmax and says so, it does not silently assume T=1 as if that
had been measured.

    python -m src.analysis.calibrate_unified
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.contrastive.simclr import build_eval_transform  # noqa: E402
from src.data.datasets import UnifiedSeedDataset  # noqa: E402
from src.models.unified_model import UnifiedSeedModel  # noqa: E402
from src.registry.model_registry import checkpoint_sha256  # noqa: E402
from src.utils.config import get_device, load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("calibrate_unified")

MANIFEST = "data_processed/manifest_unified.csv"
CHECKPOINT = "outputs/checkpoints/unified_seed_model_best.pt"
CALIBRATION_PATH = "outputs/checkpoints/unified_calibration.json"
METRICS_PATH = "outputs/metrics/unified_calibration.json"
RELIABILITY_DIR = "outputs/metrics/reliability_diagrams"
IGNORE = -1
N_BINS = 10

# A head is only handed back to the frontend once calibrated -- not merely
# measured. Below this, a displayed number would still be misleading often
# enough that withholding it remains the honest choice.
ECE_ACCEPTABLE = 0.08


def load_model(checkpoint: str, cfg, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    v_classes, q_classes = state["variety_classes"], state["quality_classes"]
    model = UnifiedSeedModel(
        v_classes, q_classes,
        backbone_name=cfg["backbone"], pretrained=False,
        use_attention=state.get("use_attention", True),
        use_channel=cfg["attention"]["use_channel_attention"],
        use_spatial=cfg["attention"]["use_spatial_attention"],
        projection_dim=cfg["contrastive"]["projection_dim"],
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, v_classes, q_classes


def collect_logits(model, split: str, v_classes, q_classes, cfg, device):
    """One forward pass per split; returns raw logits (pre-softmax) and labels for
    each head, masked to only the images that carry that head's label -- exactly
    the masking train_unified.py's own evaluate() uses, so these numbers describe
    the same population the accuracy figures in outputs/metrics/unified_seed_model.json
    already report on."""
    ds = UnifiedSeedDataset(MANIFEST, split, v_classes, q_classes,
                            build_eval_transform(cfg["training"]["image_size"]))
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    v_logits, v_labels, q_logits, q_labels = [], [], [], []
    with torch.no_grad():
        for imgs, v, q in loader:
            vlog, qlog = model(imgs.to(device), mode="classify")
            vm, qm = v != IGNORE, q != IGNORE
            if vm.any():
                v_logits.append(vlog[vm].cpu()); v_labels.append(v[vm])
            if qm.any():
                q_logits.append(qlog[qm].cpu()); q_labels.append(q[qm])
    return (
        torch.cat(v_logits), torch.cat(v_labels),
        torch.cat(q_logits), torch.cat(q_labels),
    )


def reliability_bins(probs: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """One row per confidence bin: how many predictions landed there, their mean
    confidence, and their actual accuracy. A perfectly calibrated head has
    accuracy == confidence in every populated bin -- that equality, plotted, is
    the reliability diagram."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # The top bin is closed on both ends so confidence == 1.0 lands somewhere.
        in_bin = (probs > lo) & (probs <= hi) if hi < 1.0 else (probs > lo) & (probs <= hi + 1e-9)
        n = int(in_bin.sum())
        rows.append({
            "bin_lo": round(float(lo), 2), "bin_hi": round(float(hi), 2),
            "n": n,
            "mean_confidence": round(float(probs[in_bin].mean()), 4) if n else None,
            "accuracy": round(float(correct[in_bin].mean()), 4) if n else None,
        })
    return rows


def expected_calibration_error(bins: list[dict], n_total: int) -> float:
    if n_total == 0:
        return 0.0
    return float(sum(
        (b["n"] / n_total) * abs(b["accuracy"] - b["mean_confidence"])
        for b in bins if b["n"]
    ))


def brier_score(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Multiclass Brier score: mean squared distance between the predicted
    distribution and the one-hot true label, averaged over samples. Reduces to
    the familiar binary Brier score when n_classes == 2."""
    onehot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """The single free parameter of Guo et al.'s temperature scaling, fit by
    minimising NLL with LBFGS on the split passed in. Call this on validation
    logits only -- fitting on test would let the calibration see the split it is
    later graded against."""
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature.clamp(min=1e-2), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(min=1e-2).item())


def evaluate_head(logits: torch.Tensor, labels: torch.Tensor, n_classes: int,
                   temperature: float | None = None) -> dict:
    t = temperature if temperature is not None else 1.0
    probs = F.softmax(logits / t, dim=1).numpy()
    labels_np = labels.numpy()
    pred = probs.argmax(1)
    confidence = probs.max(1)
    correct = (pred == labels_np).astype(float)

    bins = reliability_bins(confidence, correct)
    ece = expected_calibration_error(bins, len(labels_np))
    brier = brier_score(probs, labels_np, n_classes)
    nll = float(F.cross_entropy(torch.log(torch.from_numpy(probs).clamp(min=1e-12)), labels,
                                 reduction="mean").item())
    return {
        "n": len(labels_np),
        "accuracy": float((pred == labels_np).mean()),
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "nll": round(nll, 4),
        "mean_confidence": round(float(confidence.mean()), 4),
        "reliability_bins": bins,
    }


def plot_reliability(before: dict, after: dict, title: str, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, data, label in ((axes[0], before, "before (raw softmax)"),
                            (axes[1], after, "after (temperature scaled)")):
        bins = [b for b in data["reliability_bins"] if b["n"]]
        centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
        conf = [b["mean_confidence"] for b in bins]
        acc = [b["accuracy"] for b in bins]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        ax.bar(centers, acc, width=0.08, alpha=0.7, label="accuracy", color="#2c7fb8")
        ax.scatter(conf, acc, color="#d95f0e", zorder=5, label="confidence vs accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_title(f"{label}\nECE={data['ece']:.3f}  Brier={data['brier_score']:.3f}")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_head(name: str, v_logits, v_labels, t_logits, t_labels, n_classes: int) -> dict:
    logger.info(f"=== {name} head ===")
    val_before = evaluate_head(v_logits, v_labels, n_classes)
    test_before = evaluate_head(t_logits, t_labels, n_classes)
    logger.info(f"  raw softmax   val: acc={val_before['accuracy']:.3f} ece={val_before['ece']:.3f} "
                f"brier={val_before['brier_score']:.3f}")
    logger.info(f"  raw softmax  test: acc={test_before['accuracy']:.3f} ece={test_before['ece']:.3f} "
                f"brier={test_before['brier_score']:.3f}")

    temperature = fit_temperature(v_logits, v_labels)
    val_after = evaluate_head(v_logits, v_labels, n_classes, temperature)
    test_after = evaluate_head(t_logits, t_labels, n_classes, temperature)
    logger.info(f"  T={temperature:.3f}")
    logger.info(f"  calibrated    val: acc={val_after['accuracy']:.3f} ece={val_after['ece']:.3f} "
                f"brier={val_after['brier_score']:.3f}")
    logger.info(f"  calibrated   test: acc={test_after['accuracy']:.3f} ece={test_after['ece']:.3f} "
                f"brier={test_after['brier_score']:.3f}")

    plot_reliability(test_before, test_after, f"{name} head -- test split",
                      os.path.join(RELIABILITY_DIR, f"{name}.png"))

    accepted = test_after["ece"] <= ECE_ACCEPTABLE
    return {
        "temperature": round(temperature, 4),
        "validation": {"before": val_before, "after": val_after},
        "test": {"before": test_before, "after": test_after},
        "accepted_for_display": accepted,
        "acceptance_rule": (
            f"test ECE after calibration <= {ECE_ACCEPTABLE:.2f}, chosen before "
            f"this run as the bar a displayed confidence number must clear"
        ),
    }


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"]["prefer"])
    model, v_classes, q_classes = load_model(CHECKPOINT, cfg, device)

    v_val_logits, v_val_labels, q_val_logits, q_val_labels = collect_logits(
        model, "val", v_classes, q_classes, cfg, device)
    v_test_logits, v_test_labels, q_test_logits, q_test_labels = collect_logits(
        model, "test", v_classes, q_classes, cfg, device)

    variety_result = run_head("variety", v_val_logits, v_val_labels,
                               v_test_logits, v_test_labels, len(v_classes))
    quality_result = run_head("quality", q_val_logits, q_val_labels,
                               q_test_logits, q_test_labels, len(q_classes))

    payload = {
        "checkpoint": os.path.basename(CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha256(CHECKPOINT),
        "manifest": MANIFEST,
        "method": "temperature_scaling (Guo et al. 2017), one scalar per head, "
                  "fit on validation logits by NLL minimisation, evaluated on test once",
        "variety": variety_result,
        "quality": quality_result,
    }
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # The file the serving pipeline actually reads is deliberately narrow: just
    # the temperature each head may serve calibrated with, and whether it earned
    # display. Everything else above is the evidence for that decision, kept in
    # METRICS_PATH for the record but not consulted at serve time.
    serve_payload = {
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "variety": {
            "temperature": variety_result["temperature"],
            "accepted_for_display": variety_result["accepted_for_display"],
            "test_ece": variety_result["test"]["after"]["ece"],
        },
        "quality": {
            "temperature": quality_result["temperature"],
            "accepted_for_display": quality_result["accepted_for_display"],
            "test_ece": quality_result["test"]["after"]["ece"],
        },
    }
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(serve_payload, f, indent=2)
    logger.info(f"wrote {METRICS_PATH} and {CALIBRATION_PATH}")
    logger.info(f"reliability diagrams written under {RELIABILITY_DIR}/")


if __name__ == "__main__":
    main()
