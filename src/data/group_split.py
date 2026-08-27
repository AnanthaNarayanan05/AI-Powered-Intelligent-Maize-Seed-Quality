"""Group-aware train/val/test splitting.

Motivation (docs/09 section 6.9): Dataset B ships 17,713 image files that are
augmented copies of only 127 distinct source seeds, and the original random split
placed copies of all 127 sources into all three splits. Every one of the 2,655
test images was therefore a rotated/flipped view of an image the model had
trained on, and the reported 99.89-99.96% accuracies measured memorisation.

The fix is to split on GROUPS, never on individual files: whatever unit the
augmentation was derived from must live entirely inside one split.

Group keys:
  * dataset B  -- parsed from the filename, which encodes its own provenance:
                    aug_<n>_<captureid>_<Source>.jpg  and  <captureid>_<Source>.jpg
                  both belong to source <Source>. This is exact, not a heuristic.
  * dataset 4  -- exact-content clusters (md5), since the Mendeley set carries no
                  provenance in filenames but does contain 251 byte-identical copies.

Dataset A is deliberately NOT regrouped: it has 1,046 byte-distinct images, no
augmentation naming, and zero test images with a >=0.99-correlation twin in train.
Its existing split is sound.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import os
import random
import re

# aug_4_1632053314_Bihilifa40.jpg -> Bihilifa40 ; 77542039_Bihilifa30.jpg -> Bihilifa30
_AUG_PREFIX = re.compile(r"^aug_\d+_", re.I)
_CAPTURE_ID = re.compile(r"^\d+_")


def source_key_b(filepath: str) -> str:
    """Provenance token shared by every copy derived from one physical seed."""
    stem = os.path.splitext(os.path.basename(filepath))[0]
    stem = _AUG_PREFIX.sub("", stem)
    stem = _CAPTURE_ID.sub("", stem)
    return stem.lower()


def content_key(filepath: str) -> str:
    with open(filepath, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def split_groups(items, group_of, label_of, val_frac=0.15, test_frac=0.15, seed=42):
    """Assign whole groups to splits, balancing each class independently.

    Groups are shuffled per class and dealt out by group count, not image count, so
    a source contributing 294 copies carries no more weight in the split decision
    than one contributing 66. Returns {group_key: split}.
    """
    rng = random.Random(seed)
    by_class = collections.defaultdict(set)
    group_label = {}
    for it in items:
        g = group_of(it)
        by_class[label_of(it)].add(g)
        group_label[g] = label_of(it)

    assignment = {}
    for label, groups in sorted(by_class.items()):
        gs = sorted(groups)
        rng.shuffle(gs)
        n = len(gs)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        if n_test + n_val >= n:                    # tiny classes: keep at least one train group
            n_test = n_val = max(1, (n - 1) // 3)
        for g in gs[:n_test]:
            assignment[g] = "test"
        for g in gs[n_test:n_test + n_val]:
            assignment[g] = "val"
        for g in gs[n_test + n_val:]:
            assignment[g] = "train"
    return assignment


def write_manifest(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filepath", "label", "split", "group"])
        w.writeheader()
        w.writerows(rows)
