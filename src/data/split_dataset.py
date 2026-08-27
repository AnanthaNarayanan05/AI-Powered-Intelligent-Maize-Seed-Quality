"""Creates stratified train/val/test manifests (CSV: filepath,label) for a
folder-per-class dataset, deduping exact-hash duplicates first so leakage across
splits can't happen. Never copies or moves the original images — writes only a
manifest to data_processed/, per the "do not modify original datasets" rule.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
from pathlib import Path


def build_manifest(root: str, val_split: float, test_split: float, seed: int = 42):
    root_path = Path(root)
    classes = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
    rng = random.Random(seed)

    seen_hashes = set()
    rows = []  # (path, label)
    for cls in classes:
        cls_dir = root_path / cls
        files = sorted([f for f in cls_dir.iterdir() if f.is_file()])
        kept = []
        for f in files:
            digest = hashlib.md5(f.read_bytes()).hexdigest()
            if digest in seen_hashes:
                continue  # drop duplicate to prevent leakage
            seen_hashes.add(digest)
            kept.append(f)
        rng.shuffle(kept)

        n = len(kept)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        n_train = n - n_val - n_test

        splits = (
            [("train", p) for p in kept[:n_train]]
            + [("val", p) for p in kept[n_train:n_train + n_val]]
            + [("test", p) for p in kept[n_train + n_val:]]
        )
        for split, p in splits:
            rows.append((str(p), cls, split))

    return classes, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True, help="Output manifest CSV path")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    classes, rows = build_manifest(args.root, args.val_split, args.test_split, args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "split"])
        writer.writerows(rows)

    counts = {}
    for _, label, split in rows:
        counts[split] = counts.get(split, 0) + 1

    print(f"Classes ({len(classes)}): {classes}")
    print(f"Total (post-dedup): {len(rows)}  Split counts: {counts}")
    print(f"Manifest written to {args.out}")


if __name__ == "__main__":
    main()
