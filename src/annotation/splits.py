"""Deterministic plate-level train/val/test assignment.

GrainSpace crops are cut from multi-kernel plate photographs: 1,260 val crops
come from 362 plates, median 3 kernels per plate. Two kernels off the same plate
share illumination, focus, white balance and grain lot, so splitting by IMAGE
would put near-siblings on both sides of the split and report a generalisation
number that is really a memorisation number. The split unit is the plate.

The assignment is a hash of the plate id rather than a shuffle, so it is stable
across runs, stable as annotations accumulate over several sessions, and
reproducible from the plate id alone -- there is no split file to keep in sync.
"""
from __future__ import annotations

import hashlib
import os

# 70/15/15, matching configs/config.yaml (val_split 0.15, test_split 0.15).
_TRAIN_END, _VAL_END = 70, 85


def plate_id(image_path: str) -> str:
    """Plate id from a GrainSpace crop filename.

    Names look like
      20-UW-M600-0010_5_20211019105331_9_00_06_UD_13_1918-1458-2153-1664_...png
    where the first three underscore-separated fields are device, channel and
    capture timestamp -- together, one plate photograph. Everything after that is
    the kernel's index and bounding box within that plate.
    """
    stem = os.path.basename(image_path)
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    # Unrecognised naming: the file becomes its own group. That is the
    # conservative direction -- it can only make the split stricter, never leakier.
    return stem


def assign_split(group: str) -> str:
    """Map a group key to 'train', 'val' or 'test'.

    blake2b rather than the builtin hash(): Python salts string hashing per
    process, so the same plate would land in a different split on every run and
    the leakage guarantee would evaporate between the annotation session and the
    training session.
    """
    digest = hashlib.blake2b(group.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 100
    if bucket < _TRAIN_END:
        return "train"
    if bucket < _VAL_END:
        return "val"
    return "test"
