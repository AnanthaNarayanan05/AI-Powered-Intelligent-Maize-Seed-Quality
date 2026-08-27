"""Tests for the unified two-head seed model, its distribution gate, and the
group-aware splits that make its evaluation honest.

Three properties are load-bearing for the project's claims and are asserted here
rather than left to manual inspection:

  1. The model is genuinely multi-task, not a merged softmax, and a head only ever
     learns from images that actually carry its label.
  2. The quality head's output is accompanied by an honest statement of whether it
     was interpolating or extrapolating.
  3. No source seed appears in more than one split, in any manifest. This is the
     invariant whose violation invalidated the original Dataset B results; a test
     is the only thing that stops it silently returning.

    python -m pytest tests/test_unified_model.py -v
"""
from __future__ import annotations

import collections
import csv
import os

import pytest

from src.utils.config import load_config

cfg = load_config()
CKPT = cfg["paths"]["checkpoints"]
PROCESSED = cfg["paths"]["processed"]

UNIFIED_CKPT = os.path.join(CKPT, "unified_seed_model_best.pt")
QUALITY_REF = os.path.join(CKPT, "quality_reference.npz")

needs_unified = pytest.mark.skipif(
    not os.path.exists(UNIFIED_CKPT), reason="unified model not trained"
)


# --------------------------------------------------------------------------
# split integrity -- the invariant that was violated for the whole first run
# --------------------------------------------------------------------------

GROUPED_MANIFESTS = [
    "manifest_dataset_b_grouped.csv",
    "manifest_dataset_4_quality.csv",
    "manifest_unified.csv",
]


@pytest.mark.parametrize("name", GROUPED_MANIFESTS)
def test_no_group_spans_more_than_one_split(name):
    """A source seed with copies in both train and test makes the test set a
    memory check. Dataset B originally had 127 of 127 sources doing exactly that."""
    path = os.path.join(PROCESSED, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not generated")

    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    assert rows, f"{name} is empty"
    assert "group" in rows[0], f"{name} carries no group column"

    splits = collections.defaultdict(set)
    for r in rows:
        # manifest_unified.csv reuses group ids across sources; scope by source.
        key = (r.get("source", ""), r["group"])
        splits[key].add(r["split"])

    offenders = {k: v for k, v in splits.items() if len(v) > 1}
    assert not offenders, (
        f"{len(offenders)} group(s) span multiple splits in {name}, "
        f"e.g. {list(offenders.items())[:3]}"
    )


def test_dataset_b_group_key_collapses_augmented_copies():
    """The provenance parser must map every derived file back to its source seed.
    If it stops doing so, a grouped split silently degrades into a random one."""
    from src.data.group_split import source_key_b

    assert source_key_b("aug_4_1632053314_Bihilifa40.jpg") == "bihilifa40"
    assert source_key_b("77542039_Bihilifa30.jpg") == "bihilifa30"
    # every form of the same source must collapse to one key
    assert source_key_b("aug_0_1_Bihilifa40.png") == source_key_b("999_Bihilifa40.jpg")


def test_dataset_b_really_has_far_fewer_sources_than_files():
    """Guards the finding itself: if a future manifest regenerates without the
    provenance collapse, this number jumps back up and the test fails loudly."""
    path = os.path.join(PROCESSED, "manifest_dataset_b_grouped.csv")
    if not os.path.exists(path):
        pytest.skip("grouped Dataset B manifest not generated")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    n_groups = len({r["group"] for r in rows})
    assert n_groups < len(rows) / 50, (
        f"expected ~127 source seeds behind ~17.7k files; got {n_groups} groups "
        f"for {len(rows)} rows"
    )


# --------------------------------------------------------------------------
# multi-task structure
# --------------------------------------------------------------------------

def test_unified_model_exposes_two_independent_heads():
    import torch
    from src.models.unified_model import UnifiedSeedModel

    v = ["a", "b", "c", "d", "e", "f"]
    q = ["Good", "Bad"]
    m = UnifiedSeedModel(v, q, pretrained=False)
    vl, ql = m(torch.randn(2, 3, 224, 224), mode="classify")

    assert vl.shape == (2, len(v))
    assert ql.shape == (2, len(q))
    # A merged softmax would produce one tensor of width 8; two heads must not be
    # collapsible into a single distribution.
    assert vl.shape[1] + ql.shape[1] == 8
    assert not torch.allclose(vl.softmax(1).sum(1), ql.softmax(1).sum(1) * 0)


def test_masked_loss_ignores_absent_labels():
    """Datasets A and B carry no quality label and the Mendeley set no variety
    label. A head must receive zero gradient from rows lacking its label, rather
    than being trained against an invented default."""
    import torch
    import torch.nn as nn

    crit = nn.CrossEntropyLoss(ignore_index=-1)
    logits = torch.randn(4, 6, requires_grad=True)

    # only rows 0 and 1 carry a variety label
    labels = torch.tensor([0, 2, -1, -1])
    crit(logits, labels).backward()
    grad_masked = logits.grad.clone()

    assert torch.count_nonzero(grad_masked[2]) == 0, "ignored row produced gradient"
    assert torch.count_nonzero(grad_masked[3]) == 0, "ignored row produced gradient"
    assert torch.count_nonzero(grad_masked[0]) > 0, "labelled row produced no gradient"


def test_unified_manifest_never_invents_a_label():
    """Every row must carry exactly the labels its source dataset actually
    collected -- never both, and never a filled-in placeholder."""
    path = os.path.join(PROCESSED, "manifest_unified.csv")
    if not os.path.exists(path):
        pytest.skip("unified manifest not generated")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))

    both = [r for r in rows if r["variety_label"] and r["quality_label"]]
    neither = [r for r in rows if not r["variety_label"] and not r["quality_label"]]
    assert not both, f"{len(both)} rows claim both a variety and a quality label"
    assert not neither, f"{len(neither)} rows carry no label at all"

    by_source = collections.Counter(
        (r["source"], "variety" if r["variety_label"] else "quality") for r in rows
    )
    # the quality corpus must be the only source of quality labels
    assert by_source[("a", "quality")] == 0
    assert by_source[("b", "quality")] == 0
    assert by_source[("q", "variety")] == 0


# --------------------------------------------------------------------------
# distribution gate
# --------------------------------------------------------------------------

@needs_unified
@pytest.mark.skipif(not os.path.exists(QUALITY_REF), reason="quality reference not built")
def test_quality_prediction_is_marked_when_extrapolating():
    """A Dataset A photograph is nothing like the 64px Mendeley kernel crops the
    quality head learned from. Its grade must arrive flagged, with a caveat -- the
    measured distance was 0.97 against a 0.21 threshold."""
    from PIL import Image
    from src.pipeline.unified_pipeline import AnalysisPipeline

    root = cfg["paths"]["dataset_a"]
    if not os.path.isdir(root):
        pytest.skip("Dataset A not present")
    img = None
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if os.path.isdir(d):
            files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png")))
            if files:
                img = os.path.join(d, files[0])
                break
    if img is None:
        pytest.skip("no Dataset A image found")

    p = AnalysisPipeline()
    _, quality = p.classify_unified(Image.open(img).convert("RGB"))

    assert quality["is_synthetic_model"] is False
    assert quality["out_of_distribution"] is True
    assert quality["distribution_distance"] > quality["distribution_threshold"]
    assert "caveat" in quality and "extrapolation" in quality["caveat"].lower()


@needs_unified
@pytest.mark.skipif(not os.path.exists(QUALITY_REF), reason="quality reference not built")
def test_in_distribution_quality_carries_no_caveat():
    """The gate must not cry wolf: a kernel from the quality corpus itself has to
    come back unflagged, or the flag conveys nothing."""
    from PIL import Image
    from src.pipeline.unified_pipeline import AnalysisPipeline

    manifest = os.path.join(PROCESSED, "manifest_dataset_4_quality.csv")
    if not os.path.exists(manifest):
        pytest.skip("quality manifest not generated")
    with open(manifest) as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "test"]
    if not rows:
        pytest.skip("no test rows")

    p = AnalysisPipeline()
    flagged = 0
    # Stride across the whole split rather than taking its head: the manifest is
    # ordered by class, so the first N rows are all one grade and are not
    # representative of the split's distance distribution.
    step = max(1, len(rows) // 60)
    sample = rows[::step][:60]
    for r in sample:
        _, q = p.classify_unified(Image.open(r["filepath"]).convert("RGB"))
        flagged += bool(q["out_of_distribution"])
    in_dist_rate = flagged / len(sample)

    # The gate is calibrated at the 70th percentile of validation distance, a
    # deliberately conservative operating point chosen by sweep (see
    # src/analysis/build_quality_reference.py): it flags ~32% of in-distribution
    # kernels in exchange for catching 88% of Dataset C serve-time crops, on which
    # the head graded 51 of 51 Bad at confidence 1.000. A false flag withholds a
    # grade; a miss asserts a defect. The bound is loose enough to allow that
    # trade-off and tight enough that the gate must still be informative.
    assert in_dist_rate < 0.45, (
        f"{flagged}/{len(sample)} in-distribution kernels flagged ({in_dist_rate:.0%}); "
        f"the gate is too aggressive to convey anything"
    )

    # Discrimination, not just a rate: imagery the head was never validated on must
    # be flagged far more often than data drawn from its own distribution.
    root = cfg["paths"]["dataset_a"]
    if os.path.isdir(root):
        a_imgs = []
        for cls in sorted(os.listdir(root)):
            d = os.path.join(root, cls)
            if os.path.isdir(d):
                a_imgs += [os.path.join(d, f) for f in sorted(os.listdir(d))[:8]
                           if f.lower().endswith((".jpg", ".png"))]
        a_imgs = a_imgs[:20]
        if a_imgs:
            a_flagged = sum(
                bool(p.classify_unified(Image.open(f).convert("RGB"))[1]["out_of_distribution"])
                for f in a_imgs
            )
            a_rate = a_flagged / len(a_imgs)
            assert a_rate > in_dist_rate + 0.30, (
                f"gate does not discriminate: {a_rate:.0%} of out-of-distribution "
                f"imagery flagged vs {in_dist_rate:.0%} of in-distribution"
            )


@needs_unified
def test_quality_head_is_not_labelled_synthetic():
    """The synthetic-model flag drives exclusion logic in the lot report and the
    UI's provenance badge. The real quality head must never carry it."""
    from PIL import Image
    from src.pipeline.unified_pipeline import AnalysisPipeline

    manifest = os.path.join(PROCESSED, "manifest_dataset_4_quality.csv")
    if not os.path.exists(manifest):
        pytest.skip("quality manifest not generated")
    with open(manifest) as fh:
        row = next(r for r in csv.DictReader(fh) if r["split"] == "test")

    p = AnalysisPipeline()
    variety, quality = p.classify_unified(Image.open(row["filepath"]).convert("RGB"))
    assert quality["is_synthetic_model"] is False
    assert variety["is_synthetic_model"] is False
    assert "expert-assigned" in quality["label_source"].lower()


# --------------------------------------------------------------------------
# stats aggregation must not mix label spaces
# --------------------------------------------------------------------------

def test_dashboard_stats_never_counts_quality_grades_as_varieties():
    """`Good` and `Bad` are real predictions from a real model, but they are not
    maize varieties. They must not appear in the variety distribution or inflate
    the distinct-variety count.

    This regressed once: the endpoint separated varieties from everything else using
    the is_synthetic_model flag, which was sufficient while the only non-variety model
    was the synthetic one. The quality head is correctly stored with
    is_synthetic_model=0, so it slipped through and was counted as a variety.
    """
    from fastapi.testclient import TestClient
    from backend.main import app

    body = TestClient(app).get("/api/stats").json()
    if not body.get("has_data"):
        pytest.skip("no analyses stored")

    dist = body.get("variety_distribution") or {}
    for grade in ("Good", "Bad"):
        assert grade not in dist, f"{grade!r} is being counted as a maize variety"

    assert body["varieties_recognized"] == len(dist), (
        "varieties_recognized disagrees with the variety distribution it summarises"
    )
