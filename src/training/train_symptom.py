"""Trains the visible-symptom head on real GrainSpace M600 condition labels.

    # stage 1 -- head only, existing predictions provably untouched
    python -m src.training.train_symptom --stage head

    # stage 2 -- joint fine-tune, shipped only if it does not regress
    python -m src.training.train_symptom --stage joint

WHAT THIS TRAINS ON
-------------------
data_processed/manifest_symptom.csv: 1,260 real maize kernel crops, each carrying
a condition label assigned by a GrainSpace grader. No synthetic damage, no
propagated labels, no maize LEAF disease imagery standing in for kernels. That is
the entire supply of real symptom labels available to this project -- GrainSpace
train/ is organised by cultivar and carries no condition annotation.

WHY TWO STAGES
--------------
The project rule is that new functionality must not silently break what already
works. Stage 1 makes that a property of the arithmetic rather than a hope: the
backbone, the attention block and both existing heads are frozen and run in eval
mode, so the variety and quality logits for any input are bit-identical to the
shipped model. Only the new linear head learns. If stage 1 alone gives a usable
symptom head, nothing else needs to be risked.

Stage 2 unfreezes the last backbone block and the attention module at a low
learning rate and trains all three heads together on the union manifest, with the
same ignore_index=-1 masking that keeps a head from being trained on labels
nobody assigned. It is strictly a candidate: it is compared against the shipped
variety and quality test scores, and the comparison, not the symptom score,
decides whether it may be served.

CLASS WEIGHTING
---------------
The training split runs from 298 NOR crops down to 26 SD. Unweighted
cross-entropy on that distribution buys accuracy by ignoring the rare classes,
which is exactly backwards for a defect detector. Loss is therefore weighted by
inverse class frequency and model selection is on macro-F1, so a class with 26
examples counts as much as one with 298. Weighting cannot manufacture evidence:
SD holds 2 validation crops and HD holds 5 test crops, and those supports are
reported with the metrics because no amount of weighting makes an F1 computed on
2 samples informative.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report,
)

from src.contrastive.simclr import build_eval_transform, build_simclr_augmentation
from src.data.build_symptom_manifest import SYMPTOM_CLASSES, SYMPTOM_DESCRIPTIONS
from src.data.datasets import SymptomDataset, TriTaskSeedDataset, UnifiedSeedDataset
from src.models.unified_model import UnifiedSeedModel
from src.registry.model_registry import checkpoint_sha256
from src.training.train_unified import QUALITY_CLASSES, VARIETY_CLASSES
from src.utils.config import get_device, load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger("train_symptom")

SYMPTOM_MANIFEST = "data_processed/manifest_symptom.csv"
TRITASK_MANIFEST = "data_processed/manifest_tritask.csv"
UNIFIED_MANIFEST = "data_processed/manifest_unified.csv"
BASE_CHECKPOINT = "outputs/checkpoints/unified_seed_model_best.pt"
IGNORE = -1

# The shipped two-head test scores, from outputs/metrics/unified_seed_model.json.
# Stage 2 may only be served if it matches or beats both. Read from the metrics
# file at runtime rather than hard-coded, so retraining the base model cannot
# leave a stale bar behind.
SHIPPED_METRICS = "outputs/metrics/unified_seed_model.json"


def _metrics(preds, labels, classes):
    if not labels:
        return None
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    present = sorted(set(labels) | set(preds))
    return {
        "n": len(labels),
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": p, "recall_macro": r, "f1_macro": f1,
        "support": {classes[i]: int(n) for i, n in Counter(labels).items()},
        "confusion_matrix": confusion_matrix(
            labels, preds, labels=list(range(len(classes)))).tolist(),
        "per_class_report": classification_report(
            labels, preds, labels=present, target_names=[classes[i] for i in present],
            output_dict=True, zero_division=0),
    }


def _load_base(cfg, device, symptom_classes):
    """The shipped unified model, loaded strictly, then grown a symptom head."""
    if not os.path.exists(BASE_CHECKPOINT):
        raise SystemExit(f"{BASE_CHECKPOINT} not found -- train the unified model first")
    state = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=False)
    model = UnifiedSeedModel(
        state["variety_classes"], state["quality_classes"],
        backbone_name=cfg["backbone"], pretrained=False,
        use_attention=state.get("use_attention", True),
        use_channel=cfg["attention"]["use_channel_attention"],
        use_spatial=cfg["attention"]["use_spatial_attention"],
        projection_dim=cfg["contrastive"]["projection_dim"],
    ).to(device)
    model.load_state_dict(state["model_state_dict"])  # strict: nothing missing, nothing extra
    model.attach_symptom_head(symptom_classes)
    return model, state


# --------------------------------------------------------------------------- #
# stage 1: symptom head only, trunk frozen
# --------------------------------------------------------------------------- #

def evaluate_symptom(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, s in loader:
            logits = model.forward_heads(imgs.to(device))["symptom"]
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(s.tolist())
    return _metrics(preds, labels, SYMPTOM_CLASSES)


def run_head_stage(args, cfg, device):
    img_size = cfg["training"]["image_size"]
    train_ds = SymptomDataset(SYMPTOM_MANIFEST, "train", SYMPTOM_CLASSES,
                              build_simclr_augmentation(img_size))
    val_ds = SymptomDataset(SYMPTOM_MANIFEST, "val", SYMPTOM_CLASSES,
                            build_eval_transform(img_size))
    test_ds = SymptomDataset(SYMPTOM_MANIFEST, "test", SYMPTOM_CLASSES,
                             build_eval_transform(img_size))
    logger.info(f"symptom split sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    nw = cfg["training"]["num_workers"]
    lk = {"num_workers": nw, "pin_memory": device.type == "cuda", "persistent_workers": nw > 0}
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **lk)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **lk)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, **lk)

    model, base_state = _load_base(cfg, device, SYMPTOM_CLASSES)

    # stage=head freezes everything the shipped model already does. eval() on the
    # trunk as well as requires_grad_(False): BatchNorm running statistics update
    # in train() mode regardless of requires_grad, and that alone would shift the
    # variety and quality predictions.
    #
    # stage=finetune deliberately abandons that guarantee. It is a DIAGNOSTIC, not
    # a candidate for the unified checkpoint: it unfreezes the whole trunk on the
    # symptom set alone to measure how well this task can be learned when nothing
    # is held back for the other two. Its answer separates "the frozen trunk is
    # the bottleneck" from "1,260 crops cannot support this task", and those two
    # findings call for opposite responses. Its variety and quality heads are
    # expected to degrade and it is never served for them.
    finetune = args.stage == "finetune"
    for name, p in model.named_parameters():
        p.requires_grad_(True if finetune else name.startswith("symptom_head"))
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"stage={args.stage} trainable tensors: {len(trainable)} "
                f"({sum(p.numel() for p in trainable)} parameters)")

    counts = Counter(r["symptom_label"] for r in train_ds.rows)
    weights = torch.tensor(
        [len(train_ds) / (len(SYMPTOM_CLASSES) * counts[c]) for c in SYMPTOM_CLASSES],
        dtype=torch.float32, device=device)
    logger.info("class weights: " + ", ".join(
        f"{c}={w:.2f}" for c, w in zip(SYMPTOM_CLASSES, weights.tolist())))
    crit = nn.CrossEntropyLoss(weight=weights)

    epochs = args.epochs or cfg["training"]["epochs"]
    opt = torch.optim.AdamW(trainable, lr=args.lr or cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_path = os.path.join(cfg["paths"]["checkpoints"], f"{args.tag}_best.pt")
    log_path = os.path.join(cfg["paths"]["logs"], f"{args.tag}.jsonl")
    os.makedirs(cfg["paths"]["logs"], exist_ok=True)
    os.makedirs(cfg["paths"]["metrics"], exist_ok=True)
    best, stale, patience = -1.0, 0, cfg["training"]["early_stopping_patience"]

    for ep in range(epochs):
        if finetune:
            model.train()
        else:
            model.eval()         # trunk frozen, including BatchNorm statistics
            model.symptom_head.train()
        t0, tot = time.time(), 0.0
        for imgs, s in train_loader:
            imgs, s = imgs.to(device), s.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model.forward_heads(imgs)["symptom"], s)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()

        val = evaluate_symptom(model, val_loader, device)
        logger.info(f"epoch {ep+1}/{epochs} loss={tot/max(len(train_loader),1):.4f} "
                    f"val_f1_macro={val['f1_macro']:.4f} val_acc={val['accuracy']:.4f} "
                    f"({time.time()-t0:.1f}s)")
        with open(log_path, "a") as f:
            f.write(json.dumps({"stage": args.stage, "epoch": ep + 1,
                                "loss": tot / max(len(train_loader), 1),
                                "val_f1_macro": val["f1_macro"]}) + "\n")

        if val["f1_macro"] > best:
            best, stale = val["f1_macro"], 0
            torch.save({"model_state_dict": model.state_dict(),
                        "variety_classes": base_state["variety_classes"],
                        "quality_classes": base_state["quality_classes"],
                        "symptom_classes": SYMPTOM_CLASSES,
                        "use_attention": base_state.get("use_attention", True),
                        "stage": args.stage,
                        "base_checkpoint_sha256": checkpoint_sha256(BASE_CHECKPOINT)},
                       best_path)
        else:
            stale += 1
            if stale >= patience:
                logger.info(f"early stopping at epoch {ep+1}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device,
                                     weights_only=False)["model_state_dict"])
    test = evaluate_symptom(model, test_loader, device)
    if finetune:
        # Measured, not assumed: the diagnostic reports what full fine-tuning did
        # to the shipped heads so the cost of this route is on the record even
        # though the route is not taken.
        tri = evaluate_tritask(model, DataLoader(
            TriTaskSeedDataset(UNIFIED_MANIFEST, "test", VARIETY_CLASSES,
                               QUALITY_CLASSES, SYMPTOM_CLASSES,
                               build_eval_transform(img_size)),
            batch_size=bs, shuffle=False, **lk), device)
        regression = compare_to_shipped(tri["variety"], tri["quality"])
        regression["note"] = (
            "Diagnostic run. The trunk was fine-tuned on symptom labels alone, so "
            "the variety and quality heads are expected to move; this checkpoint is "
            "not a candidate to serve them."
        )
    else:
        regression = verify_no_regression(model, cfg, device)
    write_metrics(args, cfg, {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
                  best, test, regression, best_path, stage=args.stage)
    return model


# --------------------------------------------------------------------------- #
# stage 2: joint fine-tune on the union manifest
# --------------------------------------------------------------------------- #

def evaluate_tritask(model, loader, device):
    model.eval()
    acc = {"variety": ([], []), "quality": ([], []), "symptom": ([], [])}
    with torch.no_grad():
        for imgs, v, q, s in loader:
            out = model.forward_heads(imgs.to(device))
            for head, lab in (("variety", v), ("quality", q), ("symptom", s)):
                mask = lab != IGNORE
                if mask.any():
                    acc[head][0].extend(out[head][mask].argmax(1).cpu().tolist())
                    acc[head][1].extend(lab[mask].tolist())
    classes = {"variety": VARIETY_CLASSES, "quality": QUALITY_CLASSES,
               "symptom": SYMPTOM_CLASSES}
    return {h: _metrics(p, l, classes[h]) for h, (p, l) in acc.items()}


def run_joint_stage(args, cfg, device):
    img_size = cfg["training"]["image_size"]
    mk = lambda split, tf: TriTaskSeedDataset(  # noqa: E731
        TRITASK_MANIFEST, split, VARIETY_CLASSES, QUALITY_CLASSES, SYMPTOM_CLASSES, tf)
    train_ds = mk("train", build_simclr_augmentation(img_size))
    val_ds = mk("val", build_eval_transform(img_size))
    test_ds = mk("test", build_eval_transform(img_size))

    nw = cfg["training"]["num_workers"]
    lk = {"num_workers": nw, "pin_memory": device.type == "cuda", "persistent_workers": nw > 0}
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **lk)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **lk)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, **lk)

    model, base_state = _load_base(cfg, device, SYMPTOM_CLASSES)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=device,
                                         weights_only=False)["model_state_dict"])
        logger.info(f"stage=joint initialised from {args.init_from}")

    # Unfreeze only the last backbone block plus attention and the heads. The
    # early layers hold generic edge and texture filters that 931 new crops
    # cannot improve but can easily damage, taking the variety head down with
    # them.
    last_block = _last_backbone_block_prefix(model)
    for name, p in model.named_parameters():
        p.requires_grad_(
            name.startswith(("attention", "variety_head", "quality_head", "symptom_head"))
            or (last_block is not None and name.startswith(last_block))
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"stage=joint unfrozen: {last_block or 'attention+heads only'} "
                f"({sum(p.numel() for p in trainable)} parameters)")

    n = {h: sum(1 for r in train_ds.rows if r.get(c))
         for h, c in (("variety", "variety_label"), ("quality", "quality_label"),
                      ("symptom", "symptom_label"))}
    # train_unified weights heads by their raw share of labelled rows, which works
    # when the two tasks are 80/20. At three tasks the symptom rows are 5.3% of the
    # corpus, and the raw rule would hand the new head a gradient so small that the
    # stage could not answer the question it exists to ask. Weights are therefore
    # proportional to sqrt(n): still ordered by data volume, so neither shipped head
    # is displaced, but the smallest task is not effectively switched off.
    root = {h: v ** 0.5 for h, v in n.items()}
    total = sum(root.values())
    w = {h: v / total for h, v in root.items()}
    if args.symptom_weight is not None:
        w["symptom"] = args.symptom_weight
    logger.info("labelled train rows: " + ", ".join(f"{h}={v}" for h, v in n.items()))
    logger.info("head loss weights: " + ", ".join(f"{h}={x:.3f}" for h, x in w.items()))
    args._head_weights = w

    s_counts = Counter(r["symptom_label"] for r in train_ds.rows if r.get("symptom_label"))
    s_weights = torch.tensor(
        [n["symptom"] / (len(SYMPTOM_CLASSES) * s_counts[c]) for c in SYMPTOM_CLASSES],
        dtype=torch.float32, device=device)
    crit = {
        "variety": nn.CrossEntropyLoss(ignore_index=IGNORE),
        "quality": nn.CrossEntropyLoss(ignore_index=IGNORE),
        "symptom": nn.CrossEntropyLoss(ignore_index=IGNORE, weight=s_weights),
    }

    epochs = args.epochs or 12
    lr = args.lr or cfg["training"]["learning_rate"] / 10  # low: this is a fine-tune
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=cfg["training"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = cfg["device"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_path = os.path.join(cfg["paths"]["checkpoints"], f"{args.tag}_best.pt")
    log_path = os.path.join(cfg["paths"]["logs"], f"{args.tag}.jsonl")
    best, stale, patience = -1.0, 0, cfg["training"]["early_stopping_patience"]

    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for imgs, v, q, s in train_loader:
            imgs = imgs.to(device)
            lab = {"variety": v.to(device), "quality": q.to(device), "symptom": s.to(device)}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model.forward_heads(imgs)
                loss = 0.0
                for h in ("variety", "quality", "symptom"):
                    if (lab[h] != IGNORE).any():
                        loss = loss + w[h] * crit[h](out[h], lab[h])
                    else:
                        loss = loss + out[h].sum() * 0.0
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += float(loss)
        sched.step()

        val = evaluate_tritask(model, val_loader, device)
        # Mean of all three head F1s, for the same reason train_unified uses the
        # mean of two: optimising the new head alone would let the shipped ones
        # quietly regress, which is precisely what this stage risks.
        parts = [m["f1_macro"] for m in val.values() if m]
        score = sum(parts) / len(parts)
        logger.info(f"epoch {ep+1}/{epochs} loss={tot/max(len(train_loader),1):.4f} " +
                    " ".join(f"{h}={val[h]['f1_macro']:.4f}" for h in val if val[h]) +
                    f" mean={score:.4f} ({time.time()-t0:.1f}s)")
        with open(log_path, "a") as f:
            f.write(json.dumps({"stage": "joint", "epoch": ep + 1,
                                "loss": tot / max(len(train_loader), 1),
                                "val": {h: (m["f1_macro"] if m else None)
                                        for h, m in val.items()}}) + "\n")

        if score > best:
            best, stale = score, 0
            torch.save({"model_state_dict": model.state_dict(),
                        "variety_classes": base_state["variety_classes"],
                        "quality_classes": base_state["quality_classes"],
                        "symptom_classes": SYMPTOM_CLASSES,
                        "use_attention": base_state.get("use_attention", True),
                        "stage": "joint",
                        "base_checkpoint_sha256": checkpoint_sha256(BASE_CHECKPOINT)},
                       best_path)
        else:
            stale += 1
            if stale >= patience:
                logger.info(f"early stopping at epoch {ep+1}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device,
                                     weights_only=False)["model_state_dict"])
    test = evaluate_tritask(model, test_loader, device)
    regression = compare_to_shipped(test["variety"], test["quality"])
    write_metrics(args, cfg, {k: len(d) for k, d in (("train", train_ds), ("val", val_ds),
                                                    ("test", test_ds))},
                  best, test["symptom"], regression, best_path, stage="joint",
                  extra={"variety_test": test["variety"], "quality_test": test["quality"]})
    return model


def _last_backbone_block_prefix(model) -> str | None:
    """Parameter-name prefix of the final backbone stage, discovered rather than
    assumed, so a backbone swap does not silently unfreeze nothing."""
    names = [n for n, _ in model.backbone.named_parameters()]
    tops = []
    for n in names:
        head = n.split(".")[0]
        if head not in tops:
            tops.append(head)
    if not tops:
        return None
    # torchvision EfficientNet exposes one Sequential named "features"; go one
    # level deeper so this is the last block, not the whole trunk.
    if tops == ["features"]:
        idx = sorted({n.split(".")[1] for n in names if n.startswith("features.")},
                     key=lambda s: int(s) if s.isdigit() else -1)
        return f"backbone.features.{idx[-1]}."
    return f"backbone.{tops[-1]}."


# --------------------------------------------------------------------------- #
# regression checks
# --------------------------------------------------------------------------- #

def _shipped_bar():
    with open(SHIPPED_METRICS) as f:
        m = json.load(f)["test_metrics"]
    return {"variety": m["variety"]["f1_macro"], "quality": m["quality"]["f1_macro"]}


def compare_to_shipped(variety_m, quality_m) -> dict:
    bar = _shipped_bar()
    got = {"variety": variety_m["f1_macro"] if variety_m else None,
           "quality": quality_m["f1_macro"] if quality_m else None}
    ok = all(got[h] is not None and got[h] >= bar[h] for h in bar)
    return {"method": "unified test split, macro-F1 per head",
            "shipped": bar, "candidate": got,
            "delta": {h: (None if got[h] is None else got[h] - bar[h]) for h in bar},
            "no_regression": ok}


def verify_no_regression(model, cfg, device) -> dict:
    """Stage-1 check: the frozen-trunk model must reproduce the shipped variety
    and quality predictions EXACTLY, not merely score as well.

    Run over the unified test split against a freshly loaded copy of the shipped
    checkpoint, comparing argmax per image. Anything other than zero disagreement
    means the freeze leaked -- most likely a BatchNorm left in train mode.
    """
    img_size = cfg["training"]["image_size"]
    ds = UnifiedSeedDataset(UNIFIED_MANIFEST, "test", VARIETY_CLASSES, QUALITY_CLASSES,
                            build_eval_transform(img_size))
    nw = cfg["training"]["num_workers"]
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                        num_workers=nw, pin_memory=device.type == "cuda",
                        persistent_workers=nw > 0)

    state = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=False)
    ref = UnifiedSeedModel(state["variety_classes"], state["quality_classes"],
                           backbone_name=cfg["backbone"], pretrained=False,
                           use_attention=state.get("use_attention", True),
                           use_channel=cfg["attention"]["use_channel_attention"],
                           use_spatial=cfg["attention"]["use_spatial_attention"],
                           projection_dim=cfg["contrastive"]["projection_dim"]).to(device)
    ref.load_state_dict(state["model_state_dict"])
    ref.eval(); model.eval()

    n = 0
    disagree = {"variety": 0, "quality": 0}
    max_abs = {"variety": 0.0, "quality": 0.0}
    with torch.no_grad():
        for imgs, _, _ in loader:
            imgs = imgs.to(device)
            rv, rq = ref(imgs, mode="classify")
            out = model.forward_heads(imgs)
            n += imgs.size(0)
            for head, r in (("variety", rv), ("quality", rq)):
                c = out[head]
                disagree[head] += int((r.argmax(1) != c.argmax(1)).sum())
                max_abs[head] = max(max_abs[head], float((r - c).abs().max()))
    return {"method": "argmax agreement with the shipped checkpoint, unified test split",
            "n_images": n, "prediction_disagreements": disagree,
            "max_abs_logit_difference": max_abs,
            "no_regression": disagree["variety"] == 0 and disagree["quality"] == 0}


def write_metrics(args, cfg, split_sizes, best_val, test, regression, ckpt_path,
                  stage, extra=None):
    dst = os.path.join(cfg["paths"]["metrics"], f"{args.tag}.json")
    payload = {
        "stage": stage,
        "manifest": SYMPTOM_MANIFEST if stage == "head" else TRITASK_MANIFEST,
        "base_checkpoint": BASE_CHECKPOINT,
        "symptom_classes": SYMPTOM_CLASSES,
        "symptom_class_descriptions": SYMPTOM_DESCRIPTIONS,
        "label_provenance": (
            "GrainSpace M600 val split -- expert grader condition categories. "
            "Real maize kernel imagery; no synthetic damage and no leaf-disease "
            "substitution. Reproduces the grader category, not a pathogen "
            "identification."
        ),
        "split_unit": "plate (src.annotation.splits.plate_id + assign_split)",
        "split_sizes": split_sizes,
        "head_loss_weights": getattr(args, "_head_weights", None),
        "best_val_symptom_f1_macro" if stage == "head" else "best_val_mean_f1": best_val,
        "symptom_test_metrics": test,
        "regression_check": regression,
        "checkpoint_sha256": checkpoint_sha256(ckpt_path),
    }
    if extra:
        payload.update(extra)
    with open(dst, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"metrics: {dst}")
    logger.info(f"checkpoint: {ckpt_path}")
    if test:
        logger.info(f"TEST symptom: n={test['n']} acc={test['accuracy']:.4f} "
                    f"f1_macro={test['f1_macro']:.4f}")
        for c in SYMPTOM_CLASSES:
            r = test["per_class_report"].get(c)
            if r:
                logger.info(f"  {c:<4} P={r['precision']:.3f} R={r['recall']:.3f} "
                            f"F1={r['f1-score']:.3f} n={int(r['support'])}")
    logger.info(f"regression: no_regression={regression['no_regression']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["head", "joint", "finetune"], default="head")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--symptom-weight", type=float, default=None,
                    help="Override the joint-stage symptom head loss weight.")
    ap.add_argument("--init-from", default=None,
                    help="Checkpoint to warm-start the joint stage from (e.g. the stage-1 result).")
    args = ap.parse_args()
    if args.tag is None:
        args.tag = f"symptom_classifier_{args.stage}"

    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"]["prefer"])
    logger.info(f"device={device} stage={args.stage} tag={args.tag}")

    if args.stage == "joint":
        run_joint_stage(args, cfg, device)
    else:
        run_head_stage(args, cfg, device)


if __name__ == "__main__":
    main()
