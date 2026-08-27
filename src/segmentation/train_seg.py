"""Train the pixel-level defect segmenter.

    python -m src.segmentation.train_seg                      # synthetic masks
    python -m src.segmentation.train_seg         --manifest data_processed/manifest_segmentation_real.csv         --out-name defect_segmenter_real                      # verified real masks

WHAT THE LABELS ARE is not decided here. It is read from the manifest's own
`<manifest>.provenance.json` sidecar and copied into both the checkpoint and the
metrics file, so a number can never travel further than its provenance. With no
sidecar the default is the synthetic one: every defect mask re-derived from the
transformation parameters this project logged when it PAINTED the defect. Those
are exact, and they are synthetic -- a model trained on them has learned to
segment procedurally generated crack lines, mould blobs and bore holes, and has
NOT been shown to segment real fungal damage.

Samples carry a per-channel `valid` vector (see src.segmentation.dataset). A
channel that was never annotated for an image is excluded from the loss AND from
every count in the metrics, because scoring a prediction against a label that
does not exist is not evaluation. The synthetic manifest yields all-ones there,
so its arithmetic is unchanged.

Two things about the loss are deliberate:

  * Dice alongside BCE. The `cracked` channel is ~0.03% of the frame (see
    src.segmentation.dataset.describe). Plain BCE scores 99.97% by predicting an
    empty mask, and will happily converge there. Dice is computed on the overlap, so
    an empty prediction scores zero no matter how rare the class is.
  * pos_weight, capped. The exact inverse frequency for `cracked` is ~3300, which
    makes the gradient explode and the model paint everything. The cap is a stability
    choice and is recorded in the metrics file rather than buried here.

Thresholds are swept on VALIDATION and then applied unchanged to test. Sweeping on
test and reporting the best number is how a segmenter gets a score it cannot
reproduce in production.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.segmentation.area_stability import area_tier
from src.segmentation.build_synthetic_manifest import CHANNELS
from src.segmentation.dataset import SegmentationDataset
from src.segmentation.model import DefectSegmenter
from src.utils.config import load_config, get_device
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger("train_seg")

POS_WEIGHT_CAP = 25.0
SWEEP_THRESHOLDS = [round(t, 2) for t in np.arange(0.10, 0.91, 0.05)]
# A prediction below this many pixels on a kernel with no defect is noise, not a
# false alarm worth counting. 8px is the minimum measurable region established
# empirically in outputs/metrics/segmentation_resolution_floor.json.
MIN_REGION_PX = 8


def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              valid: torch.Tensor | None = None, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice, averaged over channels. eps=1.0 in BOTH numerator and denominator so
    an empty prediction on an empty target scores 1.0 rather than 0/0.

    `valid` is (B, C): 1 where that channel is annotated for that image. Masked
    samples contribute nothing to either the numerator or the denominator, so an
    unannotated channel produces no gradient rather than a gradient toward empty.
    """
    probs = torch.sigmoid(logits)
    if valid is not None:
        m = valid.view(*valid.shape, 1, 1)
        probs, target = probs * m, target * m
    dims = (0, 2, 3)
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def masked_bce(per_px: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    """Mean BCE over annotated (image, channel) pairs only."""
    if valid is None:
        return per_px.mean()
    m = valid.view(*valid.shape, 1, 1).expand_as(per_px)
    return (per_px * m).sum() / m.sum().clamp(min=1.0)


def compute_pos_weight(ds: SegmentationDataset, n_sample: int = 400) -> torch.Tensor:
    """Inverse positive-pixel frequency per channel, capped.

    Averaged over ANNOTATED images only. Counting an unknown channel as all-zero
    would inflate its apparent rarity and hand the loss a pos_weight derived from
    images that never carried a label for it.
    """
    rng = np.random.default_rng(0)
    picks = rng.permutation(len(ds))[: min(len(ds), n_sample)]
    acc = np.zeros(len(ds.channels))
    seen = np.zeros(len(ds.channels))
    for i in picks:
        _, y, v = ds[int(i)]
        acc += y.numpy().reshape(len(ds.channels), -1).mean(axis=1) * v.numpy()
        seen += v.numpy()
    frac = np.clip(acc / np.maximum(seen, 1), 1e-6, 1 - 1e-6)
    return torch.tensor(np.minimum((1 - frac) / frac, POS_WEIGHT_CAP), dtype=torch.float32)


@torch.no_grad()
def collect(model, loader, device, channels) -> dict:
    """One pass, keeping only what the metrics need: per-threshold tp/fp/fn per channel
    plus per-image predicted and true areas. Storing raw probability maps for 624
    images x 4 channels would be 125M floats; storing counts is 4 numbers.

    Every count is masked by the per-image `valid` flags. A channel that was never
    annotated for an image contributes no tp, no fp and no fn there: counting its
    prediction as a false positive would be scoring the model against a label that
    does not exist, and would make the real-data numbers look worse than the
    evidence supports for exactly the same reason that counting it as a true
    negative would make them look better."""
    C, T = len(channels), len(SWEEP_THRESHOLDS)
    tp = np.zeros((T, C), dtype=np.int64)
    fp = np.zeros((T, C), dtype=np.int64)
    fn = np.zeros((T, C), dtype=np.int64)
    img_iou_sum = np.zeros((T, C))
    img_iou_n = np.zeros((T, C), dtype=np.int64)
    empty_fp = np.zeros((T, C), dtype=np.int64)
    empty_n = np.zeros((T, C), dtype=np.int64)
    pred_area, true_area, valid_all = [[] for _ in range(T)], [], []

    model.eval()
    body_idx = channels.index("seed_body") if "seed_body" in channels else None
    for x, y, v in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        v = v.to(device, non_blocking=True)                # (B, C) 1 = annotated
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            probs = torch.sigmoid(model(x)).float()
        vb = v.bool()
        vl = v.long()
        yb = y.bool()
        true_px = y.sum(dim=(2, 3))                       # (B, C)
        true_area.append(true_px.cpu().numpy())
        valid_all.append(v.cpu().numpy())
        for ti, th in enumerate(SWEEP_THRESHOLDS):
            p = probs > th
            inter = (p & yb).sum(dim=(2, 3))
            pcount = p.sum(dim=(2, 3))
            union = pcount + true_px.long() - inter
            tp[ti] += (inter * vl).sum(0).cpu().numpy()
            fp[ti] += ((pcount - inter) * vl).sum(0).cpu().numpy()
            fn[ti] += ((true_px.long() - inter) * vl).sum(0).cpu().numpy()
            has = (true_px > 0) & vb
            iou = torch.where(union > 0, inter.float() / union.clamp(min=1), torch.ones_like(inter, dtype=torch.float))
            img_iou_sum[ti] += (iou * has).sum(0).cpu().numpy()
            img_iou_n[ti] += has.sum(0).cpu().numpy()
            clean = (true_px == 0) & vb
            empty_n[ti] += clean.sum(0).cpu().numpy()
            empty_fp[ti] += (clean & (pcount > MIN_REGION_PX)).sum(0).cpu().numpy()
            pred_area[ti].append(pcount.cpu().numpy())

    true_area = np.concatenate(true_area, axis=0)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "img_iou": img_iou_sum / np.maximum(img_iou_n, 1),
        "img_iou_n": img_iou_n,
        "empty_fp_rate": empty_fp / np.maximum(empty_n, 1),
        "empty_n": empty_n,
        "pred_area": [np.concatenate(a, axis=0) for a in pred_area],
        "true_area": true_area,
        "valid": np.concatenate(valid_all, axis=0),
        "body_idx": body_idx,
    }


def metrics_at(stats: dict, ti: int, channels: list[str]) -> dict:
    tp, fp, fn = stats["tp"][ti], stats["fp"][ti], stats["fn"][ti]
    out = {}
    for ci, ch in enumerate(channels):
        prec = tp[ci] / max(tp[ci] + fp[ci], 1)
        rec = tp[ci] / max(tp[ci] + fn[ci], 1)
        out[ch] = {
            "threshold": SWEEP_THRESHOLDS[ti],
            "iou_micro": float(tp[ci] / max(tp[ci] + fp[ci] + fn[ci], 1)),
            "dice_micro": float(2 * tp[ci] / max(2 * tp[ci] + fp[ci] + fn[ci], 1)),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
            "iou_per_image_mean": float(stats["img_iou"][ti][ci]),
            "images_with_this_defect": int(stats["img_iou_n"][ti][ci]),
            "false_alarm_rate_on_clean": float(stats["empty_fp_rate"][ti][ci]),
            "clean_images": int(stats["empty_n"][ti][ci]),
        }
    return out


def coverage_error(stats: dict, thresholds: dict, channels: list[str]) -> dict:
    """VISIBLE DEFECT AREA % error -- the number this pipeline actually reports.

    coverage% = defect pixels / valid visible seed pixels x 100. Computed two ways:
    against the TRUE body (isolates defect-mask error) and against the PREDICTED body
    (what production will do, since production has no ground-truth body). Reporting
    only the first would overstate what the deployed number is worth.

    Both an ABSOLUTE (percentage-point) and a RELATIVE error are recorded, because on
    this data the absolute figure alone is actively misleading: the `cracked` channel
    scores 0.16pp at p90, which reads like precision until you notice the quantity
    being estimated averages 0.15% -- the error is larger than the value. Relative
    error is taken per image over the images that actually carry the defect; a ratio
    of two p90s is not a p90 of ratios, and dividing the summary figures after the
    fact would quietly invent a statistic nobody computed.
    """
    bi = stats["body_idx"]
    if bi is None:
        return {}
    true_body = stats["true_area"][:, bi]
    valid = stats["valid"]
    out = {}
    for ci, ch in enumerate(channels):
        if ch == "seed_body":
            continue
        ti = SWEEP_THRESHOLDS.index(thresholds[ch])
        pred = stats["pred_area"][ti][:, ci]
        pred_body = stats["pred_area"][SWEEP_THRESHOLDS.index(thresholds["seed_body"])][:, bi]
        # Both the numerator's channel and the denominator's body must be
        # annotated: a coverage % computed from an unknown defect mask is not a
        # measurement with error, it is a number with no referent.
        ok = (true_body > 0) & (valid[:, ci] > 0) & (valid[:, bi] > 0)
        if not ok.any():
            out[ch] = {"n_images_with_defect": 0, "area_tier": "UNKNOWN",
                       "note": "no image in this split carries an annotation for "
                               "both this channel and the seed body"}
            continue
        true_cov = np.where(ok, stats["true_area"][:, ci] / np.maximum(true_body, 1) * 100, 0.0)
        cov_true_den = np.where(ok, pred[:] / np.maximum(true_body, 1) * 100, 0.0)
        cov_pred_den = np.where(pred_body > 0, pred[:] / np.maximum(pred_body, 1) * 100, 0.0)
        err_t = np.abs(cov_true_den - true_cov)[ok]
        err_p = np.abs(cov_pred_den - true_cov)[ok]
        # relative error only where the defect is actually present -- on a clean kernel
        # true coverage is 0 and the ratio is undefined, not zero
        has = ok & (true_cov > 0)
        rel = (np.abs(cov_pred_den - true_cov)[has] / true_cov[has] * 100.0
               if has.any() else np.array([]))
        rel_med = float(np.median(rel)) if rel.size else None
        out[ch] = {
            "true_coverage_pct_mean": float(true_cov[ok].mean()),
            "abs_err_pp_median_true_body": float(np.median(err_t)),
            "abs_err_pp_p90_true_body": float(np.percentile(err_t, 90)),
            "abs_err_pp_median_pred_body": float(np.median(err_p)),
            "abs_err_pp_p90_pred_body": float(np.percentile(err_p, 90)),
            "n_images_with_defect": int(has.sum()),
            "rel_err_pct_median_pred_body": rel_med,
            "rel_err_pct_p90_pred_body": float(np.percentile(rel, 90)) if rel.size else None,
            "area_tier": area_tier(rel_med) if rel_med is not None else "UNKNOWN",
        }
    return out


SYNTHETIC_PROVENANCE = {
    "label_provenance": "synthetic",
    "label_note": ("Every defect mask was re-derived from the transformation "
                   "parameters logged when this project painted the defect "
                   "(src/data/synthetic_defect_generator.reconstruct_defect_mask). "
                   "Exact by construction, and synthetic. seed_body is an "
                   "algorithmic threshold label, not a human annotation. These "
                   "numbers do NOT establish performance on real defects."),
}


def read_provenance(manifest_path: str) -> dict:
    """Label provenance for a manifest, taken from its sidecar.

    Provenance travels WITH the data rather than as a command-line flag. A flag
    can be forgotten on a rerun, and the failure mode of forgetting it is a
    checkpoint that claims real human annotation for synthetic masks -- the one
    claim this project must never make by accident.
    """
    sidecar = os.path.splitext(manifest_path)[0] + ".provenance.json"
    if os.path.exists(sidecar):
        with open(sidecar) as fh:
            prov = json.load(fh)
        missing = {"label_provenance", "label_note"} - set(prov)
        if missing:
            raise SystemExit(f"{sidecar} is missing {sorted(missing)}")
        return prov
    return dict(SYNTHETIC_PROVENANCE)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data_processed/manifest_segmentation_synthetic.csv")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-attention", action="store_true")
    ap.add_argument("--warm-start", default="outputs/checkpoints/contrastive_encoder_a.pt",
                    help="SimCLR encoder to initialise from; '' to use ImageNet weights only")
    ap.add_argument("--out-name", default="defect_segmenter_synthetic")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training; re-run the val threshold sweep and the test "
                         "pass against the existing checkpoint. For recomputing metrics "
                         "after a change to the METRIC, so the reported numbers come "
                         "from the same weights rather than from a fresh run that "
                         "would differ for unrelated reasons.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"]["prefer"])
    # Logged because a silent CPU fallback looks identical to a GPU run in the
    # metrics file, and only shows up as an epoch time 30x too long.
    logger.info(f"device={device} (config prefers {cfg['device']['prefer']!r})")
    tr = cfg["training"]
    epochs = args.epochs or tr["epochs"]
    batch_size = args.batch_size or tr["batch_size"]
    lr = args.lr or tr["learning_rate"]
    size = tr["image_size"]

    prov = read_provenance(args.manifest)
    logger.info(f"label provenance: {prov['label_provenance']}")

    train_ds = SegmentationDataset(args.manifest, "train", image_size=size, augment=True, seed=cfg["seed"])
    val_ds = SegmentationDataset(args.manifest, "val", image_size=size, augment=False)
    test_ds = SegmentationDataset(args.manifest, "test", image_size=size, augment=False)
    channels = train_ds.channels
    # Same loader settings as every other trainer here (train_variety, train_unified):
    # workers matter more for this task than for classification, because each sample
    # decodes a source photo AND up to two mask PNGs.
    nw = tr["num_workers"]
    lk = {"num_workers": nw, "pin_memory": device.type == "cuda", "persistent_workers": nw > 0}
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **lk)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **lk)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **lk)

    model = DefectSegmenter(channels, cfg["backbone"], pretrained=True,
                            use_attention=not args.no_attention).to(device)
    warm = {"path": None, "matched": 0, "total": 0}
    if args.warm_start and os.path.exists(args.warm_start):
        state = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        # src/training/pretrain_contrastive.py writes {"backbone_state_dict", "config"};
        # the other names are here so a raw state_dict also works.
        sd = state
        if isinstance(state, dict):
            for key in ("backbone_state_dict", "model_state", "state_dict"):
                if key in state:
                    sd = state[key]
                    break
        m, t = model.load_contrastive_encoder(sd)
        warm = {"path": args.warm_start, "matched": m, "total": t}
        logger.info(f"warm start from {args.warm_start}: {m}/{t} encoder tensors matched")
        if m == 0:
            raise SystemExit(
                f"--warm-start {args.warm_start} matched 0 of {t} encoder tensors. "
                "Training would silently fall back to ImageNet weights and the metrics "
                "would be attributed to a contrastive warm start that never happened. "
                "Fix the key mapping, or pass --warm-start '' to train from ImageNet "
                "on purpose."
            )
    elif args.warm_start:
        logger.warning(f"{args.warm_start} not found; ImageNet weights only")

    if args.eval_only:
        pos_weight = torch.ones(len(channels))
        bce = None
    else:
        pos_weight = compute_pos_weight(SegmentationDataset(args.manifest, "train", image_size=size, augment=False))
        logger.info("pos_weight " + ", ".join(f"{c}={w:.1f}" for c, w in zip(channels, pos_weight.tolist())))
        # reduction="none": the mean has to be taken over ANNOTATED elements only,
        # which the loss function cannot know about.
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(1, -1, 1, 1).to(device),
                                   reduction="none")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["device"]["mixed_precision"] and device.type == "cuda")

    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.out_name}_best.pt")
    best_score, best_epoch, patience = -1.0, -1, tr["early_stopping_patience"]

    defect_idx = [i for i, c in enumerate(channels) if c != "seed_body"]
    history = []

    if args.eval_only and not os.path.exists(ckpt_path):
        raise SystemExit(f"--eval-only needs {ckpt_path}, which does not exist. Train first.")

    for epoch in ([] if args.eval_only else range(1, epochs + 1)):
        model.train()
        t0, running = time.time(), 0.0
        for x, y, v in train_ld:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            v = v.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(x)
                loss = masked_bce(bce(logits, y), v) + dice_loss(logits, y, v)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / len(train_ds)

        stats = collect(model, val_ld, device, channels)
        # model selection on mean DEFECT dice at the best per-channel threshold; the
        # body channel is easy and would otherwise dominate the score
        per_ch_best = []
        for ci in defect_idx:
            d = [2 * stats["tp"][ti][ci] / max(2 * stats["tp"][ti][ci] + stats["fp"][ti][ci] + stats["fn"][ti][ci], 1)
                 for ti in range(len(SWEEP_THRESHOLDS))]
            per_ch_best.append(max(d))
        score = float(np.mean(per_ch_best))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_defect_dice": score})
        logger.info(f"[seg] epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
                    f"val_defect_dice={score:.4f} ({time.time() - t0:.1f}s)")

        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({
                "model_state": model.state_dict(),
                "channel_names": channels,
                "image_size": size,
                "backbone": cfg["backbone"],
                "use_attention": not args.no_attention,
                "label_provenance": prov["label_provenance"],
                "label_note": prov["label_note"],
                "epoch": epoch,
                "val_defect_dice": score,
            }, ckpt_path)
        elif epoch - best_epoch >= patience:
            logger.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state"])

    val_stats = collect(model, val_ld, device, channels)
    thresholds = {}
    for ci, ch in enumerate(channels):
        d = [2 * val_stats["tp"][ti][ci] / max(2 * val_stats["tp"][ti][ci] + val_stats["fp"][ti][ci] + val_stats["fn"][ti][ci], 1)
             for ti in range(len(SWEEP_THRESHOLDS))]
        thresholds[ch] = SWEEP_THRESHOLDS[int(np.argmax(d))]
    logger.info("thresholds tuned on VAL: " + json.dumps(thresholds))

    test_stats = collect(model, test_ld, device, channels)
    test_metrics = {ch: metrics_at(test_stats, SWEEP_THRESHOLDS.index(thresholds[ch]), channels)[ch]
                    for ch in channels}
    val_metrics = {ch: metrics_at(val_stats, SWEEP_THRESHOLDS.index(thresholds[ch]), channels)[ch]
                   for ch in channels}
    cov = coverage_error(test_stats, thresholds, channels)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state["thresholds"] = thresholds
    state["test_metrics"] = test_metrics
    torch.save(state, ckpt_path)

    out = {
        "model": args.out_name,
        "manifest": args.manifest,
        "label_provenance": prov["label_provenance"],
        "label_note": prov["label_note"],
        "channels": channels,
        "split_unit": "source seed image (no kernel appears in two splits)",
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "warm_start": warm,
        "pos_weight": {c: float(w) for c, w in zip(channels, pos_weight.tolist())},
        "pos_weight_cap": POS_WEIGHT_CAP,
        "best_epoch": best_epoch,
        "best_val_defect_dice": best_score,
        "thresholds_tuned_on": "val",
        "thresholds": thresholds,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_coverage_error_pp": cov,
        "min_region_px_for_false_alarm": MIN_REGION_PX,
        "history": history,
    }
    os.makedirs(cfg["paths"]["metrics"], exist_ok=True)
    mpath = os.path.join(cfg["paths"]["metrics"], f"{args.out_name}.json")
    with open(mpath, "w") as fh:
        json.dump(out, fh, indent=2)

    logger.info(f"[SEG] test per-channel (threshold tuned on val):")
    for ch in channels:
        m = test_metrics[ch]
        logger.info(f"  {ch:18s} th={m['threshold']:.2f} IoU={m['iou_micro']:.4f} "
                    f"Dice={m['dice_micro']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
                    f"per-image IoU={m['iou_per_image_mean']:.4f} "
                    f"false-alarm on clean={m['false_alarm_rate_on_clean']:.3f}")
    for ch, c in cov.items():
        logger.info(f"  {ch:18s} coverage% err (pred body) median={c['abs_err_pp_median_pred_body']:.3f}pp "
                    f"p90={c['abs_err_pp_p90_pred_body']:.3f}pp  (true mean coverage "
                    f"{c['true_coverage_pct_mean']:.2f}%)")
        # The relative figure is the one that decides whether the % may be quoted.
        logger.info(f"  {ch:18s} coverage% REL err median={c['rel_err_pct_median_pred_body']:.1f}% "
                    f"p90={c['rel_err_pct_p90_pred_body']:.1f}% -> area tier {c['area_tier']}")
    logger.info(f"metrics -> {mpath}   checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
