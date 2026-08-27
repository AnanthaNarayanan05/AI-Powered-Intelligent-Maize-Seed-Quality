"""Image + mask dataset for defect segmentation.

Reads data_processed/manifest_segmentation_synthetic.csv (built by
src.segmentation.build_synthetic_manifest) and returns, per sample:

    image  float32 (3, H, W)   ImageNet-normalised, same convention as every other
                               model in this project (src/contrastive/simclr.py)
    target float32 (C, H, W)   one binary plane per channel in CHANNELS order

The target is a STACK, not a label map, because the planes overlap: `seed_body`
contains every defect pixel by definition, and a real kernel can be cracked and
mouldy in the same spot. Downstream this is read with independent sigmoids.

Augmentation is the part worth being careful about. Geometric transforms are applied
to the image and to every mask plane with the SAME parameters, and only exact
transforms are used -- flips and 90-degree rotations. Nothing here resamples a mask:
a rotation by 17 degrees would have to interpolate mask pixels and re-threshold them,
which silently changes the mask's area, and defect AREA is the number this whole
pipeline exists to produce. Photometric jitter touches the image only.
"""
from __future__ import annotations

import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.segmentation.build_synthetic_manifest import CHANNELS, DEFECT_CHANNELS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# A mask PNG is written as 0/255 by the generator. Anything above this counts as
# defect. It is a re-read of an exact binary file, not a soft-alpha decision, so the
# midpoint is safe; it exists as a constant only so the value is greppable.
MASK_BINARIZE_THRESHOLD = 127


def _joint_geometric(img: np.ndarray, target: np.ndarray, rng: np.random.Generator):
    """Flip/rotate image and all mask planes identically. Exact, no interpolation."""
    if rng.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
        target = np.ascontiguousarray(target[:, :, ::-1])
    if rng.random() < 0.5:
        img = np.ascontiguousarray(img[::-1, :])
        target = np.ascontiguousarray(target[:, ::-1, :])
    k = int(rng.integers(0, 4))
    if k:
        img = np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))
        target = np.ascontiguousarray(np.rot90(target, k, axes=(1, 2)))
    return img, target


def _photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brightness / contrast / saturation / hue jitter on the IMAGE only.

    Deliberately milder than the SimCLR augmentation (0.2/0.2/0.2/0.05 there): this
    model has to decide whether a brown patch is mould, and mould is defined here by
    a colour shift. Jitter hard enough to turn a healthy kernel the colour of a mouldy
    one and the labels stop matching the pixels.
    """
    out = img.astype(np.float32)
    out *= 1.0 + float(rng.uniform(-0.15, 0.15))            # brightness
    mean = out.mean()
    out = (out - mean) * (1.0 + float(rng.uniform(-0.15, 0.15))) + mean   # contrast
    out = np.clip(out, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + float(rng.uniform(-4, 4))) % 180          # hue
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + float(rng.uniform(-0.12, 0.12))), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
        image_size: int = 224,
        augment: bool = False,
        channels: list[str] | None = None,
        seed: int = 42,
    ):
        self.channels = list(channels) if channels else list(CHANNELS)
        self.image_size = image_size
        self.augment = augment
        self.seed = seed

        with open(manifest_path) as fh:
            rows = [r for r in csv.DictReader(fh) if r["split"] == split]
        if not rows:
            raise SystemExit(
                f"no rows with split={split!r} in {manifest_path}; run "
                "python -m src.segmentation.build_synthetic_manifest first"
            )
        self.rows = rows
        self.split = split

    def __len__(self) -> int:
        return len(self.rows)

    def _read_mask(self, path: str, shape: tuple[int, int]) -> np.ndarray:
        if not path:
            return np.zeros(shape, dtype=np.uint8)
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise SystemExit(f"mask file listed in the manifest is unreadable: {path}")
        return (m > MASK_BINARIZE_THRESHOLD).astype(np.uint8)

    def __getitem__(self, i: int):
        r = self.rows[i]
        bgr = cv2.imread(r["image_path"])
        if bgr is None:
            raise SystemExit(f"image listed in the manifest is unreadable: {r['image_path']}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        defect = self._read_mask(r["defect_mask_path"], (h, w))
        body = self._read_mask(r["body_mask_path"], (h, w))

        planes = []
        for ch in self.channels:
            if ch == "seed_body":
                planes.append(body)
            elif ch == r["label"]:
                planes.append(defect)
            else:
                # Not "unknown" -- the generator painted exactly one defect class onto
                # this kernel, so every other defect plane is a true negative.
                planes.append(np.zeros((h, w), dtype=np.uint8))
        target = np.stack(planes, axis=0)

        s = self.image_size
        if (h, w) != (s, s):
            # INTER_AREA for the photo (correct downsampling), NEAREST for the masks
            # (a mask must stay binary; averaging mask pixels invents partial defects)
            img = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
            target = np.stack(
                [cv2.resize(p, (s, s), interpolation=cv2.INTER_NEAREST) for p in target],
                axis=0,
            )

        if self.augment:
            # Seeded per-sample rather than globally so a worker process cannot hand
            # out the same augmentation to every item in its batch.
            rng = np.random.default_rng((self.seed * 1_000_003 + i * 7919 +
                                         int(torch.randint(0, 2**31 - 1, (1,)).item())) % (2**63))
            img, target = _joint_geometric(img, target, rng)
            img = _photometric(img, rng)

        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(target)).float()
        return x, y


def describe(manifest_path: str) -> None:
    """Print what is actually in the manifest, per split and per channel.

    Positive-pixel fractions matter for reading the loss later: a channel that is
    0.3% of pixels will look 99.7% correct while predicting nothing at all.
    """
    for split in ("train", "val", "test"):
        ds = SegmentationDataset(manifest_path, split, augment=False)
        n = len(ds)
        sums = np.zeros(len(ds.channels))
        total = 0
        # RANDOM subsample, not a stride. The manifest is written in source order and
        # every source contributes its four classes consecutively, so any fixed stride
        # whose value shares a factor with 4 samples only some of the classes and
        # reports 0.00% for the rest -- a fake statistic produced by the sampler.
        picks = np.random.default_rng(0).permutation(n)[:min(n, 400)]
        for i in picks:
            _, y = ds[int(i)]
            sums += y.numpy().reshape(len(ds.channels), -1).mean(axis=1)
            total += 1
        frac = sums / max(total, 1)
        print(f"{split:5s} n={n:5d} sampled={total:4d}")
        for ch, f in zip(ds.channels, frac):
            print(f"        {ch:18s} mean positive pixels {f * 100:6.2f}%")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data_processed/manifest_segmentation_synthetic.csv")
    args = ap.parse_args()
    print("channels:", CHANNELS, " (defects:", DEFECT_CHANNELS, ")")
    describe(args.manifest)
