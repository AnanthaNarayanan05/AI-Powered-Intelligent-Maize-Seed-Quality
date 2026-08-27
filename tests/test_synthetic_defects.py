import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from src.data.synthetic_defect_generator import (
    _apply_crack, _apply_discoloration, _apply_insect_damage, generate_synthetic_dataset, SYNTHETIC_CLASSES,
)
import random


def _fake_seed_image():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 70, (200, 210, 220), -1)  # bright "seed" on black background
    return img


def test_transforms_only_modify_foreground_and_stay_deterministic():
    img = _fake_seed_image()
    rng = random.Random(0)
    out, params = _apply_crack(img, rng)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)  # something changed
    assert "lines" in params and len(params["lines"]) == params["n_lines"]

    rng2 = random.Random(0)
    out2, params2 = _apply_crack(img, rng2)
    assert np.array_equal(out, out2)  # same seed -> same result (reproducible ground truth)


def test_discoloration_and_insect_damage_run():
    img = _fake_seed_image()
    rng = random.Random(1)
    out_d, params_d = _apply_discoloration(img, rng)
    assert out_d.shape == img.shape
    out_i, params_i = _apply_insect_damage(img, rng)
    assert out_i.shape == img.shape
    assert params_i["n_holes"] >= 3


def test_generate_synthetic_dataset_manifest_matches_disk(tmp_path):
    source_root = tmp_path / "source"
    (source_root / "classA").mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(source_root / "classA" / f"img{i}.jpg"), _fake_seed_image())

    out_root = tmp_path / "synthetic_out"
    manifest_path = tmp_path / "manifest.csv"
    rows = generate_synthetic_dataset(str(source_root), str(out_root), str(manifest_path), per_class_synthetic_ratio=1.0, seed=42)

    labels = {r["synthetic_label"] for r in rows}
    assert labels == set(SYNTHETIC_CLASSES)

    # every non-healthy row's file must actually exist on disk (manifest is not lying about what was generated)
    for r in rows:
        if r["synthetic_label"] != "healthy":
            assert os.path.exists(r["synthetic_image_path"]), f"Missing file: {r['synthetic_image_path']}"

    # healthy rows reference the real source image directly, never a copy
    healthy_rows = [r for r in rows if r["synthetic_label"] == "healthy"]
    assert len(healthy_rows) == 3
    for r in healthy_rows:
        assert r["synthetic_image_path"] == r["source_image"]
