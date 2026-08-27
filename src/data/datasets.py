"""PyTorch Dataset classes built on top of the manifests produced by
split_dataset.py / synthetic_defect_generator.py. Torch is imported lazily at call
time so the rest of src/data stays usable without torch installed (as in this dev
sandbox)."""
from __future__ import annotations

import csv
from pathlib import Path

# Module-level, not an instance attribute: Windows DataLoader workers use spawn,
# which pickles the dataset, and module objects are not picklable.
from PIL import Image


def _read_manifest(path: str, split: str | None = None):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is None or row.get("split") == split:
                rows.append(row)
    return rows


class VarietyImageDataset:
    """Folder-per-class variety dataset (A or B), driven by a manifest CSV with
    columns filepath,label,split."""

    def __init__(self, manifest_path: str, split: str, classes: list[str], transform=None):
        self.rows = _read_manifest(manifest_path, split)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        label = self.class_to_idx[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label


class ContrastiveImageDataset:
    """Wraps an image folder for SimCLR-style pretraining: returns two independently
    augmented views of the same image, no labels required."""

    def __init__(self, filepaths: list[str], two_view_transform):
        self.filepaths = filepaths
        self.two_view_transform = two_view_transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        view1 = self.two_view_transform(img)
        view2 = self.two_view_transform(img)
        return view1, view2


class SyntheticDefectDataset:
    """Driven by the manifest written by synthetic_defect_generator.py:
    synthetic_image_path,synthetic_label,source_image,source_variety_class,transform_params
    """

    def __init__(self, manifest_path: str, classes: list[str], split_indices=None, transform=None):
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)
        if split_indices is not None:
            self.rows = [self.rows[i] for i in split_indices]
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["synthetic_image_path"]).convert("RGB")
        label = self.class_to_idx[row["synthetic_label"]]
        if self.transform:
            img = self.transform(img)
        return img, label


def list_all_filepaths(root: str) -> list[str]:
    """All image filepaths under a folder-per-class root, for contrastive pretraining
    (which needs no labels)."""
    root_path = Path(root)
    paths = []
    for cls_dir in root_path.iterdir():
        if not cls_dir.is_dir():
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                paths.append(str(f))
    return paths

class UnifiedSeedDataset:
    """Multi-task dataset over manifest_unified.csv.

    Returns (image, variety_idx, quality_idx). An index of -1 means the label was
    never collected for that image -- Datasets A and B have no quality annotation,
    the Mendeley quality set has no variety annotation. -1 is the ignore_index the
    training loop's CrossEntropyLoss is configured with, so a head receives gradient
    only from images that genuinely carry its label. It is never a stand-in for an
    unknown or assumed value.
    """

    def __init__(self, manifest_path: str, split: str,
                 variety_classes: list[str], quality_classes: list[str], transform=None):
        self.rows = _read_manifest(manifest_path, split)
        self.variety_classes = variety_classes
        self.quality_classes = quality_classes
        self.variety_to_idx = {c: i for i, c in enumerate(variety_classes)}
        self.quality_to_idx = {c: i for i, c in enumerate(quality_classes)}
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        v = self.variety_to_idx.get(row.get("variety_label") or "", -1)
        q = self.quality_to_idx.get(row.get("quality_label") or "", -1)
        return img, v, q
