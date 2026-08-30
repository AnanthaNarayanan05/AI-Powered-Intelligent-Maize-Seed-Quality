"""Calibrates what the visible-symptom classifier is allowed to say.

A 7-class model at 0.5865 macro-F1 is not a model that should answer every
question put to it. Two separate things have to be decided before it can serve,
and both are decided here on the validation split, never on test:

WHICH CLASSES ARE VALIDATED AT ALL
----------------------------------
The GrainSpace condition classes are not equally evidenced. Training support runs
from 298 NOR crops down to 26 SD, and validation support -- the only honest basis
for judging a class, since test is held out -- runs from 54 down to 2. A class
with 2 validation crops has no measurable operating characteristics; an F1
computed on it is noise wearing a number's clothes.

So each class earns the right to be asserted. It needs enough validation support
for the estimate to mean anything, and it needs to actually score on that
support. A class that fails is not deleted from the model -- its logit still
competes, and it still absorbs inputs that would otherwise be mislabelled as
something else -- but when it wins, the prediction is withheld rather than
reported. The user is told the input does not match any validated category, which
is true, rather than being handed a category the evidence does not support.

WHERE THE CONFIDENCE FLOOR SITS
-------------------------------
Even within validated classes the model is wrong about a third of the time. A
prediction near the decision boundary is worth less than one made confidently,
and the project rule is to withhold rather than assert on insufficient evidence.
The floor is the smallest softmax probability at which validation accuracy over
the retained predictions clears a stated target -- so the threshold buys accuracy
with coverage, and both halves of that trade are reported. Test coverage and test
accuracy at the chosen floor are then measured once, as a held-out check that the
calibration transfers.

WHAT THIS CANNOT FIX
--------------------
Abstention does not make a weak class strong; it only stops a weak class from
speaking. HD in particular is heat damage, which presents as whole-kernel
discolouration and is the class the model confuses with NOR in both directions.
No threshold recovers a class the representation does not separate, and none of
this converts a visual grading category into a pathogen diagnosis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.contrastive.simclr import build_eval_transform  # noqa: E402
from src.data.build_symptom_manifest import SYMPTOM_CLASSES, SYMPTOM_DESCRIPTIONS  # noqa: E402
from src.data.datasets import SymptomDataset  # noqa: E402
from src.models.unified_model import UnifiedSeedModel  # noqa: E402
from src.registry.model_registry import checkpoint_sha256  # noqa: E402
from src.utils.config import get_device, load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("calibrate_symptom")

MANIFEST = "data_processed/manifest_symptom.csv"
CHECKPOINT = "outputs/checkpoints/symptom_classifier_finetune_best.pt"
GATE_PATH = "outputs/checkpoints/symptom_gate.json"
METRICS_PATH = "outputs/metrics/symptom_gate.json"

# A class needs this many validation crops before any statistic computed on it is
# reported as evidence, and this much macro-F1 on them before it may be asserted.
MIN_VAL_SUPPORT = 10
MIN_VAL_F1 = 0.40
# The accuracy the retained predictions must reach on validation. This is a stated
# requirement on the product, fixed before the test split was consulted, not a
# number read off a curve: a reported symptom that is wrong more than about one
# time in seven is worse for a defect report than reporting nothing, because a
# wrong category is acted on while an absent one is investigated.
#
# 0.75 was tried first and rejected on principle, not on its test score: class
# eligibility alone already cleared it, so the rule selected a floor of 0.00 and
# the "gate" gated nothing. A target that any threshold satisfies does not
# express a requirement.
TARGET_VAL_ACCURACY = 0.85


def load_model(checkpoint: str, cfg, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    classes = state["symptom_classes"]
    model = UnifiedSeedModel(
        state["variety_classes"], state["quality_classes"],
        backbone_name=cfg["backbone"], pretrained=False,
        use_attention=state.get("use_attention", True),
        use_channel=cfg["attention"]["use_channel_attention"],
        use_spatial=cfg["attention"]["use_spatial_attention"],
        projection_dim=cfg["contrastive"]["projection_dim"],
        symptom_classes=classes,
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, classes, state


def probabilities(model, split: str, classes: list[str], cfg, device):
    ds = SymptomDataset(MANIFEST, split, classes, build_eval_transform(cfg["training"]["image_size"]))
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    probs, labels = [], []
    with torch.no_grad():
        for imgs, y in loader:
            logits = model.forward_heads(imgs.to(device))["symptom"]
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
            labels.extend(y.tolist())
    return np.concatenate(probs), np.array(labels)


def per_class(probs: np.ndarray, labels: np.ndarray, classes: list[str]) -> dict:
    pred = probs.argmax(1)
    out = {}
    for i, c in enumerate(classes):
        tp = int(((pred == i) & (labels == i)).sum())
        fp = int(((pred == i) & (labels != i)).sum())
        fn = int(((pred != i) & (labels == i)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {
            "support": int((labels == i).sum()),
            "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
        }
    return out


def sweep(probs: np.ndarray, labels: np.ndarray, eligible: np.ndarray) -> list[dict]:
    """Coverage / accuracy at every candidate confidence floor.

    A prediction is retained only if it names an eligible class AND clears the
    floor, because those are the two independent reasons to withhold and the
    operating point has to be read off the combination that will actually serve.
    """
    pred, conf = probs.argmax(1), probs.max(1)
    rows = []
    for tau in np.arange(0.0, 0.96, 0.05):
        keep = (conf >= tau) & eligible[pred]
        n = int(keep.sum())
        rows.append({
            "threshold": round(float(tau), 2),
            "coverage": n / len(labels),
            "n_retained": n,
            "accuracy": float((pred[keep] == labels[keep]).mean()) if n else 0.0,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--target-accuracy", type=float, default=TARGET_VAL_ACCURACY)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"]["prefer"])
    model, classes, _ = load_model(args.checkpoint, cfg, device)

    val_p, val_y = probabilities(model, "val", classes, cfg, device)
    test_p, test_y = probabilities(model, "test", classes, cfg, device)
    val_class = per_class(val_p, val_y, classes)

    validated, reasons = [], {}
    for c in classes:
        m = val_class[c]
        if m["support"] < MIN_VAL_SUPPORT:
            reasons[c] = (f"only {m['support']} validation crops "
                          f"(minimum {MIN_VAL_SUPPORT}); no measurable operating point")
        elif m["f1"] < MIN_VAL_F1:
            reasons[c] = (f"validation F1 {m['f1']:.3f} below the {MIN_VAL_F1:.2f} floor "
                          f"on {m['support']} crops")
        else:
            validated.append(c)
    eligible = np.array([c in validated for c in classes])

    logger.info("validation per class:")
    for c in classes:
        m = val_class[c]
        mark = "SERVED " if c in validated else "WITHHELD"
        logger.info(f"  {c:<4} {mark} n={m['support']:<3} P={m['precision']:.3f} "
                    f"R={m['recall']:.3f} F1={m['f1']:.3f}")
    for c, why in reasons.items():
        logger.info(f"  {c} withheld: {why}")

    val_sweep = sweep(val_p, val_y, eligible)
    chosen = next((r for r in val_sweep if r["accuracy"] >= args.target_accuracy
                   and r["n_retained"] > 0), None)
    if chosen is None:
        chosen = max(val_sweep, key=lambda r: r["accuracy"])
        logger.info(f"no threshold reaches val accuracy {args.target_accuracy:.2f}; "
                    f"reporting the best available at {chosen['threshold']:.2f}")
    tau = chosen["threshold"]

    logger.info("val threshold sweep (eligible classes only):")
    for r in val_sweep:
        mark = " <--" if r["threshold"] == tau else ""
        logger.info(f"  tau={r['threshold']:.2f} coverage={r['coverage']:.3f} "
                    f"n={r['n_retained']:<3} acc={r['accuracy']:.3f}{mark}")

    test_sweep = sweep(test_p, test_y, eligible)
    test_at = next(r for r in test_sweep if r["threshold"] == tau)
    logger.info(f"HELD-OUT at tau={tau:.2f}: coverage={test_at['coverage']:.3f} "
                f"n={test_at['n_retained']} accuracy={test_at['accuracy']:.3f}")

    payload = {
        "checkpoint": os.path.basename(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "classes": classes,
        "class_descriptions": {c: SYMPTOM_DESCRIPTIONS[c] for c in classes},
        "validated_classes": validated,
        "withheld_classes": reasons,
        "confidence_threshold": tau,
        "selection_rule": (
            f"smallest softmax floor whose validation accuracy over retained "
            f"predictions reaches {args.target_accuracy:.2f}; classes need "
            f"{MIN_VAL_SUPPORT}+ validation crops and F1 >= {MIN_VAL_F1:.2f} to be asserted"
        ),
        "validation": {"per_class": val_class, "sweep": val_sweep, "at_threshold": chosen},
        "test_at_threshold": test_at,
        # Carried explicitly rather than left to be derived, because it is the
        # number most likely to be dropped when these results are quoted: the
        # threshold was chosen on validation and validation flatters it.
        "calibration_transfer": {
            "val_accuracy": chosen["accuracy"],
            "test_accuracy": test_at["accuracy"],
            "optimism": chosen["accuracy"] - test_at["accuracy"],
            "note": ("Accuracy over retained predictions on the split the threshold "
                     "was chosen on, versus the held-out split. The held-out figure "
                     "is the one that describes serving."),
        },
        "test_sweep": test_sweep,
    }
    os.makedirs(os.path.dirname(GATE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    for path in (GATE_PATH, METRICS_PATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    logger.info(f"wrote {GATE_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    main()
