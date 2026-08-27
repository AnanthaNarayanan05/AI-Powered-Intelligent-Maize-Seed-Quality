"""Paired significance test across the Dataset B group-aware ablations.

The test set is 2,669 images but only 19 physical seeds. Treating images as
independent samples would pseudo-replicate: ~140 augmented views of one kernel are
one observation, not 140. Each seed is therefore scored as a unit (fraction of its
images classified correctly) and the four ablations are compared with a paired
Wilcoxon signed-rank test over those 19 paired values.

Writes outputs/metrics/ablation_significance_b_grouped.json
"""
from __future__ import annotations

import collections, csv, itertools, json, os

import torch
from scipy.stats import wilcoxon

from src.contrastive.simclr import build_eval_transform
from src.data.datasets import VarietyImageDataset
from src.models.variety_classifier import CognitiveAttentionClassifier
from src.utils.config import load_config, get_device
from torch.utils.data import DataLoader

MANIFEST = "data_processed/manifest_dataset_b_grouped.csv"
RUNS = {
    "baseline":         ("variety_b_baseline_grouped_best.pt",         False),
    "attention_only":   ("variety_b_attention_only_grouped_best.pt",   True),
    "contrastive_only": ("variety_b_contrastive_only_grouped_best.pt", False),
    "full":             ("variety_b_full_grouped_best.pt",             True),
}


def per_seed_accuracy(ckpt, use_attention, cfg, device, classes):
    model = CognitiveAttentionClassifier(
        num_classes=len(classes), backbone_name=cfg["backbone"], pretrained=False,
        use_attention=use_attention,
        use_channel=cfg["attention"]["use_channel_attention"],
        use_spatial=cfg["attention"]["use_spatial_attention"],
        projection_dim=cfg["contrastive"]["projection_dim"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
    model.eval()

    ds = VarietyImageDataset(MANIFEST, "test", classes, build_eval_transform(cfg["training"]["image_size"]))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    preds = []
    with torch.no_grad():
        for imgs, _ in loader:
            preds.extend(model(imgs.to(device), mode="classify").argmax(1).cpu().tolist())

    hit, tot = collections.Counter(), collections.Counter()
    for row, p in zip(ds.rows, preds):
        g = row["group"]
        tot[g] += 1
        hit[g] += int(p == ds.class_to_idx[row["label"]])
    return {g: hit[g] / tot[g] for g in tot}


def main():
    cfg = load_config()
    device = get_device(cfg["device"]["prefer"])
    classes = cfg["dataset_b"]["classes"]
    ck = cfg["paths"]["checkpoints"]

    acc = {}
    for name, (fn, att) in RUNS.items():
        path = os.path.join(ck, fn)
        acc[name] = per_seed_accuracy(path, att, cfg, device, classes)
        print(f"{name}: {len(acc[name])} seeds, mean per-seed acc "
              f"{sum(acc[name].values())/len(acc[name])*100:.2f}%")

    seeds = sorted(next(iter(acc.values())).keys())
    out = {"n_seeds": len(seeds),
           "per_seed_mean": {k: sum(v.values())/len(v) for k, v in acc.items()},
           "pairwise": {}}

    print(f"\nPaired Wilcoxon signed-rank over {len(seeds)} seeds:")
    for a, b in itertools.combinations(RUNS, 2):
        xa = [acc[a][s] for s in seeds]
        xb = [acc[b][s] for s in seeds]
        diff = [p - q for p, q in zip(xb, xa)]
        if all(d == 0 for d in diff):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(xa, xb, zero_method="wilcox")
        better = sum(1 for d in diff if d > 0)
        worse = sum(1 for d in diff if d < 0)
        out["pairwise"][f"{a}_vs_{b}"] = {
            "mean_delta_pp": (sum(xb)/len(xb) - sum(xa)/len(xa)) * 100,
            "seeds_better_for_second": better, "seeds_worse": worse,
            "seeds_tied": len(seeds) - better - worse,
            "wilcoxon_p": float(p),
            "significant_at_0.05": bool(p < 0.05),
        }
        print(f"  {a:<17} vs {b:<17} delta={out['pairwise'][f'{a}_vs_{b}']['mean_delta_pp']:+6.2f}pp  "
              f"better/worse/tied={better}/{worse}/{len(seeds)-better-worse}  p={p:.4f}"
              f"{'  *' if p < 0.05 else ''}")

    os.makedirs(cfg["paths"]["metrics"], exist_ok=True)
    dst = os.path.join(cfg["paths"]["metrics"], "ablation_significance_b_grouped.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"\nwritten: {dst}")


if __name__ == "__main__":
    main()
