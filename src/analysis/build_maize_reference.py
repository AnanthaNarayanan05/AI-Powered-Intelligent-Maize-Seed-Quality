"""Builds the foreign-object review aid shipped by Phase 4.

What this ships, and what it refuses to claim
---------------------------------------------
One question is answered: how unlike the maize kernels the system knows is this
detected object? The answer is a percentile against held-out maize -- an
atypicality score, not an identity. There is no stone, husk, cob-fragment or
debris label anywhere in this project's data, so no such word is ever produced.
This is FOREIGN OBJECT FLAGGING, not FOREIGN OBJECT CLASSIFICATION.

It is deliberately shipped as a RANKED REVIEW AID rather than as a detector, and
that is a conclusion from the measurement below rather than a hedge. As a ranking
the score is good; as a binary alarm it is useful only to the extent that a human
looks at what it surfaces. Reporting it as a detector would overstate what was
measured.

Calibrated where it is served
-----------------------------
Every embedding here -- reference, calibration and evaluation alike -- is a YOLO
detector crop, because a crop is what the gate is handed at inference. An earlier
build embedded whole dataset photographs. It honoured its stated 2% false-flag
rate on the images it was shown and produced an 80% flag rate on real uploads:
the same kernels sat eighteen times further from a whole-image reference once cut
out of a bounding box (median distance 0.0058 against 0.1026). A guarantee
measured in the wrong domain is not a weak guarantee, it is not a guarantee.

Corrected for crop size, because a quarter of the raw score was crop size
------------------------------------------------------------------------
The raw kNN distance falls steadily as crops get larger. Regressed on crop short
side alone it gives R2 = 0.243: nearly a quarter of what the "maize identity"
score measured was image resolution. That is not an academic objection. GrainSet
impurity crops have a median short side of 92px against 274-290px for same-rig
maize, so a size-sensitive score is partly rewarded for a difference that has
nothing to do with what the object is -- and on real uploads, where one 740x493
image yielded 300 crops at 24-48px, the same sensitivity marked 63.5% of objects
against a nominal 10% budget.

The fix is a parametric correction, fitted by least squares on the held-out
CALIBRATION crops only and never on the evaluation sets or the impurities:

    expected(px) = a + b*log(px) + c*log(px)^2
    score        = knn_distance - expected(short_side_px)

After it, the residual's correlation with log crop size is +0.000 (from -0.72
pooled on same-rig data). A binned size-conditional threshold was tried as the
alternative and was worse and noisier -- same-rig maize surfaced 37.5% against
79.3% caught -- so the parametric residual is what ships.

A size-matched control was also run and had to be thrown out, which is worth
recording because its first reading looked fatal. Restricting to overlapping size
bands collapsed AUROC to 0.5946, but reconstructing each crop's detection rank
from the cached per-image counts showed why: 100% of same-rig sound maize crops
in the impurity size band, and 82% of damaged ones, are YOLO's SECOND or later
detection on a single-object image -- spurious fragments, median 113px against
296px for the first detection. Impurity images yield 1.02 crops each and only 3%
come from multi-detection images. The control was comparing impurities against
the detector's own mistakes rather than against maize.

The size-corrected AUROC against impurities, first detection per image only:

    unified_test                 0.896  [0.884, 0.909]
    grainspace_damaged_held_out  0.937  [0.923, 0.950]
    grainset_sound_same_rig      0.860  [0.845, 0.875]
    grainset_damaged_same_rig    0.888  [0.872, 0.903]

The same-rig figures are the honest ones: those kernels were photographed on the
impurities' own rig, so imaging cannot explain the separation. They are lower
after correction than before it (0.901 and 0.923 raw), which is the expected
direction -- part of the raw margin really was resolution.

Why one gate and not two
------------------------
Phase 4 requires the existing distribution gating to be tried first. It was.
``eval_maize_gate.py`` scores it on identical data: at its shipped threshold of
0.0929 the quality gate catches 99.7% of impurities and false-flags 77% to 95% of
real maize. Routed to a foreign-object flag it would call almost every kernel
suspect. Recalibrated it is a real but weaker discriminator, and an earlier
whole-image measurement suggested that requiring BOTH gates to fire beat either
alone. In the serving domain that reverses, so the AND rule was dropped. The
quality distance is still reported alongside the verdict as context; it no longer
votes.

The measured sweep
------------------
The threshold is the residual at which the given fraction of CALIBRATION crops
sits above -- held-out maize, fixed before any evaluation set is touched. The
surfaced rates below are therefore measurements of how that calibration
generalises, not fits. Nothing is calibrated on impurities, so the catch rate is
also a measurement. "Enrichment" is how much richer in foreign objects the
surfaced slice is than the pool it came from.

    budget  threshold   maize surfaced (4 sets)      caught  enrichment
      2%     +0.1999    3.4  3.1  3.4  3.7           43.4%     13.0x
 -->  5%     +0.1422    8.0  6.5  9.0  6.8           69.8%      9.3x
     10%     +0.0929   13.7 10.7 18.0 15.4           80.1%      5.9x
     20%     +0.0464   24.1 18.3 40.8 29.3           86.4%      3.4x

5% is shipped. It finds 69.8% of the impurities the detector found while
surfacing under 10% of maize on every evaluation set, at 9.3x the hit rate of
reviewing at random. 10% buys a tenth more recall for roughly double the
reviewing and lets same-rig surfaced rates drift to 18%. The previous build shipped
10% because at 5% the uncorrected score caught only 35.1%; the correction is what
makes the tighter operating point affordable.

Note that the budget is a calibration-set guarantee and the evaluation sets
exceed it by up to 1.8x at 5%. The overshoot is real and is reported rather than
smoothed: held-out maize from corpora with different size distributions is
surfaced somewhat more often than calibration maize was.

Two limits of applicability, both derived rather than chosen
------------------------------------------------------------
The score answers for one object at a time and says nothing about whether that
object was photographed in a scene the gate has ever seen. Two conditions put it
outside the range it was measured in, and in both the platform returns UNAVAILABLE
rather than a number -- the same principle as Phase 8's segmentation floor.

RESOLUTION FLOOR. Below the smallest crops the correction was fitted on,
``expected(px)`` is an extrapolation, and even just above them the residual keeps
some size-dependent spread. So the floor is walked up from the 1st percentile of
calibration crop short sides: any band whose surfaced rate exceeds twice the
budget is ruled out, bands widened until they hold at least twenty crops. On this
build the 13-21px band surfaces at 2.9x the budget and is rejected; 21-26px
surfaces at 1.1x and is accepted, so the floor lands at 21px -- above 4.1% of
calibration crops.

SCENE LIMIT. A densely packed upload -- 300 confident detections of 21-47px
kernels, every one of them ordinary maize by inspection -- had 42% of its objects
surfaced against a 5% budget, while calibration crops of the same size behave at
6-8%. Crop size does not explain it and the size correction cannot fix it: in a
packed scene each crop is filled with fragments of neighbouring kernels rather
than background, which is a different image statistic entirely. Nothing in the
reference or calibration corpora looks like that -- the densest scene in either
yields sixteen objects. Above that count the gate is extrapolating about the scene
rather than the object, so it declines. This is a real coverage gap, not a fixed
problem: closing it needs densely packed labelled imagery this project does not
have.

Honest limitations, all of which belong in any report of these numbers
----------------------------------------------------------------------
* This is not a detector and its output is not a purity figure. At a realistic 1%
  contamination rate the great majority of surfaced objects are ordinary maize.
  The score ranks; a human decides.
* The catch rate is measured among objects YOLO found. The detector fires on only
  74.2% of impurity images, and nothing it misses is ever scored, so end-to-end
  recall at the shipped point is about 51.8%. Absence of a flag means very little.
* The correction removes the mean size trend but not all of the size-dependent
  spread. Calibration crops under 45px are still surfaced at about 15% at a 10%
  budget against 3.5% for crops over 180px, so small objects remain
  over-represented in the review list.
* The impurities are GrainSet's, from one rig and one region. Generalisation to
  other contaminants and other imaging is untested.
* Densely packed scenes are refused rather than served, so on bulk-sample
  photographs -- arguably the most useful case -- this feature currently returns
  nothing at all.
* YOLO fires on only 32.1% of ``unified_test`` images -- they are largely tight
  single-kernel photographs, unlike the multi-kernel scenes it was trained on. The
  maize evaluation sets are therefore biased toward images the detector fires on.

    python -m src.analysis.build_maize_reference

Writes outputs/checkpoints/maize_reference.npz
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.analysis import maize_identity as mi
from src.pipeline.unified_pipeline import AnalysisPipeline
from src.registry import checkpoint_sha256

QUALITY_REFERENCE = "outputs/checkpoints/quality_reference.npz"
METRICS = "outputs/metrics/maize_gate_comparison.json"
OUT = "outputs/checkpoints/maize_reference.npz"

# The fraction of held-out CALIBRATION crops the threshold allows above it. See
# the sweep above for why 5% rather than the 10% the previous build shipped.
REVIEW_BUDGET = 0.05

# The resolution floor is derived, not chosen: walking up from the smallest
# calibration crops, any size band whose surfaced rate exceeds TOLERANCE times the
# budget is ruled outside the range the score was calibrated on, and the floor is
# where that stops. Bands are widened until they hold at least MIN_BAND_N crops so
# the decision is never made on a handful of them.
FLOOR_START_PERCENTILE = 1.0
FLOOR_BAND_RATIO = 1.25
FLOOR_TOLERANCE = 2.0
FLOOR_MIN_BAND_N = 20


def _design(sizes: np.ndarray) -> np.ndarray:
    """Quadratic in log crop short side. Quadratic rather than linear because the
    distance flattens at large sizes; higher orders bought no R2 worth the
    extrapolation risk below the floor."""
    log = np.log(np.asarray(sizes, dtype=np.float64))
    return np.column_stack([np.ones_like(log), log, log * log])


def _fit_size_correction(distances: np.ndarray, sizes: np.ndarray):
    """Least squares on calibration only. Returns (coef, r2, residual_size_corr).

    Fitting this anywhere else would be circular: a correction fitted on the
    evaluation maize would flatten exactly the sets whose surfaced rates are then
    reported as evidence the gate behaves.
    """
    design = _design(sizes)
    coef, *_ = np.linalg.lstsq(design, distances, rcond=None)
    residual = distances - design @ coef
    r2 = float(1.0 - residual.var() / distances.var())
    corr = float(np.corrcoef(np.log(sizes), residual)[0, 1])
    return coef, r2, corr


def _score(distances: np.ndarray, sizes: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """The shipped score: distance with the size trend subtracted."""
    return distances - _design(sizes) @ coef


def _resolution_floor(scores: np.ndarray, sizes: np.ndarray, threshold: float,
                      budget: float, log=print) -> int:
    """The smallest crop size at which calibration still honours its own budget.

    The size correction removes the mean trend but not all of the size-dependent
    spread, so the very smallest crops stay over-surfaced even after it. Rather
    than let them through at a rate the budget does not describe, or pick a round
    number, the floor is walked up from the bottom of the fitted range until a band
    behaves.
    """
    floor = float(np.percentile(sizes, FLOOR_START_PERCENTILE))
    log(f"  deriving the resolution floor (tolerance {FLOOR_TOLERANCE:g}x "
        f"of a {budget:.0%} budget)")
    while floor < np.median(sizes):
        top = floor * FLOOR_BAND_RATIO
        band = (sizes >= floor) & (sizes < top)
        # Widen rather than judge a band on too few crops.
        while band.sum() < FLOOR_MIN_BAND_N and top < np.median(sizes):
            top *= FLOOR_BAND_RATIO
            band = (sizes >= floor) & (sizes < top)
        if band.sum() < FLOOR_MIN_BAND_N:
            break
        rate = float((scores[band] > threshold).mean())
        ok = rate <= FLOOR_TOLERANCE * budget
        log(f"    {floor:5.1f}-{top:5.1f}px  n={band.sum():<4d} surfaced "
            f"{rate:5.1%} ({rate / budget:4.1f}x)  {'ok' if ok else 'rejected'}")
        if ok:
            break
        floor = top
    return int(math.ceil(floor))


def _scene_limit(log=print) -> int | None:
    """The most objects any single calibration or reference image contributed.

    A per-crop score says nothing about the scene it came from, and it turns out
    the scene matters. On a densely packed upload -- 300 confident detections of
    21-47px kernels, every one of them maize by inspection -- 42% of objects were
    surfaced against a 5% budget, while calibration crops of the same size behave
    at 6-8%. Crop size does not explain it; packing does. Each crop in such a scene
    is filled with fragments of its neighbours rather than background, and nothing
    in the corpus this gate was calibrated on looks like that: the densest image
    anywhere in it yields sixteen objects.

    So this is reported as a limit rather than corrected. Beyond it the gate is
    outside the domain it was measured in and serving returns unavailable, on the
    same principle as the resolution floor.
    """
    peak = 0
    for key, paths, _loader in mi.calibration_spec():
        counts = mi.cached_counts(paths, key)
        if counts is not None and len(counts):
            peak = max(peak, int(counts.max()))
    for source in ("a", "b", "q"):
        rows = mi.unified_rows("train")
        paths = [r["filepath"] for r in rows if r["source"] == source]
        counts = mi.cached_counts(paths, f"ref_unified_{source}")
        if counts is not None and len(counts):
            peak = max(peak, int(counts.max()))
    counts = mi.cached_counts(mi.grainspace_split()["reference"], "ref_grainspace")
    if counts is not None and len(counts):
        peak = max(peak, int(counts.max()))
    if peak:
        log(f"  scene limit       {peak} objects per image (the densest scene the "
            "gate was calibrated on)")
    return peak or None


def main():
    pipe = AnalysisPipeline()
    model, _, _ = pipe._get_unified_model()
    tf = pipe._get_eval_transform()

    print("building the maize-identity reference (serving domain: detector crops)...")
    reference, provenance = mi.build_reference(pipe, model, tf)
    for p in provenance:
        print(f"  {p['count']:>6} crops from {p['images']:>6} images  {p['source']}")

    print("calibrating on held-out maize...")
    calibration, n_cal = mi.calibration_embeddings(pipe, model, tf)
    cal_sizes = mi.calibration_sizes()
    if cal_sizes is None or len(cal_sizes) != n_cal:
        raise SystemExit(
            "calibration crop sizes are missing or misaligned. Delete "
            f"{mi.CACHE_DIR} and rebuild so embeddings and sizes come from one pass."
        )
    cal_raw = mi.knn_distance(calibration, reference)
    coef, r2, resid_corr = _fit_size_correction(cal_raw, cal_sizes)
    cal_scores = np.sort(_score(cal_raw, cal_sizes, coef))
    print(f"  {n_cal} crops")
    print(f"  size correction   R2={r2:.3f} of the raw distance was crop size; "
          f"residual size-corr {resid_corr:+.3f}")

    threshold = float(np.quantile(cal_scores, 1.0 - REVIEW_BUDGET))
    floor = _resolution_floor(_score(cal_raw, cal_sizes, coef), cal_sizes,
                              threshold, REVIEW_BUDGET)
    print(f"  resolution floor  {floor}px "
          f"({(cal_sizes < floor).mean():.1%} of calibration below it)")
    scene_limit = _scene_limit()

    # Scored for the first time here. The threshold above is already fixed.
    maize_emb, foreign_emb = mi.evaluation_sets(pipe, model, tf)
    ev_sizes = mi.evaluation_crop_sizes()
    missing = [k for k in {**maize_emb, **foreign_emb} if k not in ev_sizes]
    if missing:
        raise SystemExit(f"no cached crop sizes for {missing}; rebuild the cache")
    maize = {k: _score(mi.knn_distance(v, reference), ev_sizes[k], coef)
             for k, v in maize_emb.items()}
    foreign = {k: _score(mi.knn_distance(v, reference), ev_sizes[k], coef)
               for k, v in foreign_emb.items()}

    catch = {k: float((v > threshold).mean()) for k, v in foreign.items()}
    recall = float(np.mean(list(catch.values())))
    surfaced_by_set = {k: float((v > threshold).mean()) for k, v in maize.items()}

    # How much richer the surfaced slice is than the pool it came from. This is
    # the figure that justifies shipping a ranking at all, and it is the one a
    # grader deciding whether to use the review list actually needs.
    pooled = np.concatenate(list(maize.values()))
    surfaced = float((pooled > threshold).mean())
    enrichment = recall / surfaced if surfaced else None

    # The detector is the first filter and nothing it misses is ever scored, so
    # the recall a user experiences is the product of the two. Read from the counts
    # embed_crops cached during this same pass -- no second detector run.
    impurities = mi.grainset_paths("train/7_IM") + mi.grainset_paths("test/7_IM")
    counts = mi.cached_counts(impurities, "ev_grainset_im")
    detection = float((counts > 0).mean()) if counts is not None and len(counts) else None

    print(f"  review budget     {REVIEW_BUDGET:.0%} of calibration crops")
    print(f"  threshold         {threshold:+.4f} (size-corrected score)")
    for k, v in surfaced_by_set.items():
        print(f"    surfaced  {k:<32} {v:6.1%}")
    print(f"  impurities caught {recall:.1%} (of those the detector found)")
    if enrichment:
        print(f"  enrichment        {enrichment:.1f}x over reviewing at random")
    if detection is not None:
        print(f"  detection rate    {detection:.1%}")
        print(f"  end-to-end recall {recall * detection:.1%}")

    # float16 halves an 89 MB artifact and was measured to change none of 7,465
    # decisions -- the largest distance change was 2.3e-5, against thresholds
    # separated by tenths. Re-normalised on load, so the rounding does not
    # accumulate into the cosine. The calibration stays float32: it is 8 KB and it
    # is what every reported percentile is read off.
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT,
        reference=reference.astype(np.float16),
        calibration=cal_scores.astype(np.float32),
        size_coef=coef.astype(np.float64),
        threshold=np.float32(threshold),
        review_budget=np.float32(REVIEW_BUDGET),
        resolution_floor_px=np.int32(floor),
        **({"scene_limit_objects": np.int32(scene_limit)} if scene_limit else {}),
        k=np.int32(mi.K),
        calibration_crops=np.int32(n_cal),
        size_r2=np.float32(r2),
        # Carried inside the artifact so serving quotes what this build measured.
        # Nothing downstream hard-codes a performance figure; a rebuild that moves
        # these numbers moves what the user is told.
        measured_recall=np.float32(recall),
        **({"enrichment": np.float32(enrichment)} if enrichment else {}),
        **({"detection_rate": np.float32(detection)} if detection is not None else {}),
        provenance=json.dumps(provenance),
    )
    print(f"written: {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")

    # The measurement was made before this checkpoint existed, so bind them here:
    # the registry recomputes this hash and reports the metrics as unverified if
    # the artifact was rebuilt without re-measuring.
    if os.path.exists(METRICS):
        with open(METRICS) as f:
            metrics = json.load(f)
        metrics["checkpoint_sha256"] = checkpoint_sha256(OUT)
        metrics["shipped"] = {
            "rule": ("rank every detected object by its size-corrected distance "
                     "from the maize reference; surface those above the review "
                     "threshold; return unavailable below the resolution floor"),
            "is_classification": False,
            "is_detector": False,
            "framing": ("ranked review aid -- as a binary alarm this score is weak; "
                        "as a ranking it concentrates foreign objects about "
                        "ninefold"),
            "review_budget": REVIEW_BUDGET,
            "threshold": round(threshold, 4),
            "size_correction": {
                "form": "distance - (a + b*log(px) + c*log(px)^2)",
                "fitted_on": "held-out calibration crops only",
                "coef": [round(float(c), 6) for c in coef],
                "raw_distance_r2_on_crop_size": round(r2, 4),
                "residual_size_correlation": round(resid_corr, 4),
                "resolution_floor_px": floor,
                "scene_limit_objects": scene_limit,
            },
            "measured": {
                "maize_surfaced_rate": {k: round(v, 4)
                                        for k, v in surfaced_by_set.items()},
                "foreign_catch_rate": {k: round(v, 4) for k, v in catch.items()},
                "enrichment": round(enrichment, 2) if enrichment else None,
                "detection_rate": (round(detection, 4)
                                   if detection is not None else None),
                "end_to_end_recall": (round(recall * detection, 4)
                                      if detection is not None else None),
            },
        }
        metrics["label_provenance"] = (
            "No foreign-object labels were used to fit anything. Both the size "
            "correction and the threshold are fitted on held-out maize alone; "
            "GrainSet impurities are used only to measure the resulting catch rate."
        )
        with open(METRICS, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"stamped: {METRICS}")
    else:
        print(f"WARNING: {METRICS} missing -- run "
              "python -m src.analysis.eval_maize_gate first, then rebuild, so the "
              "registry can bind these metrics to this checkpoint.")


if __name__ == "__main__":
    main()
