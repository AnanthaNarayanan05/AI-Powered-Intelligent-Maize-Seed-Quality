"""Measures whether a maize-identity gate can honestly flag foreign objects (Phase 4).

Phase 4 requires the existing distribution gating to be tried first, so this runs
both gates side by side on identical data:

  quality_gate    the shipped gate -- reference is the Mendeley quality training
                  split only (outputs/checkpoints/quality_reference.npz)
  maize_identity  the same pooled k-NN cosine score over a reference spanning
                  every maize corpus the project holds

and reports, at a sweep of operating points calibrated on held-out maize:

  * the false-flag rate on each maize evaluation set, kept separate rather than
    pooled -- an average hides a gate that is fine on clean kernels and useless on
    damaged ones, and damaged kernels are most of what a grader inspects
  * the catch rate on GrainSet impurities, which are real non-grain objects
  * AUROC per maize set, raw and after the crop-size correction the platform
    actually serves -- R2 = 0.243 of the raw distance was crop short side alone,
    so an uncorrected reading of these tables credits resolution as identity
  * the same AUROC restricted to each image's FIRST detection, because later
    detections on single-kernel photographs are fragments rather than kernels
    and land in exactly the size range the impurities occupy
  * the densest scene the gate was calibrated on, which is a limit of
    applicability rather than a statistic: past it serving declines

Nothing here is calibrated on impurities. The threshold comes from held-out maize
alone, so the catch rate is a measurement rather than a fit.

    python -m src.analysis.eval_maize_gate

Writes outputs/metrics/maize_gate_comparison.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.analysis import maize_identity as mi
# The size correction is imported rather than re-derived: this file has to
# measure the transform that actually ships, not a second copy of it.
from src.analysis.build_maize_reference import (_fit_size_correction, _score,
                                                _scene_limit)
from src.pipeline.unified_pipeline import AnalysisPipeline

PERCENTILES = [90, 95, 97.5, 99, 99.5, 99.9]

# Operating points are also chosen the other way round: fix the worst tolerable
# false-flag rate across every maize set and read off what that costs in recall.
# A percentile is a statement about the calibration set; this is a statement about
# the guarantee a grader actually cares about.
MAX_FALSE_FLAG = [0.01, 0.02, 0.05]
OUT = "outputs/metrics/maize_gate_comparison.json"


def _rank(scores, calibration):
    """A score restated as "what fraction of held-out maize it exceeds".

    The two gates live on different distance scales -- their thresholds differ by
    a factor of three -- so a raw min() would just always return the smaller one.
    Mapping each through its own calibration distribution puts them on a common
    axis where the minimum genuinely means "the more forgiving of the two gates".
    """
    return np.searchsorted(np.sort(calibration), scores, side="right") / len(calibration)


def _worst_case_points(maize_scores, foreign_scores, targets):
    """Thresholds chosen by the guarantee rather than by a percentile.

    For each tolerated false-flag rate, the lowest threshold at which NO maize
    evaluation set exceeds it -- the worst set governs, because a gate that is
    clean on catalogue photographs and noisy on same-rig kernels is noisy.
    """
    grid = np.unique(np.concatenate(list(maize_scores.values())))
    rows = []
    for target in targets:
        # Per set, the smallest threshold meeting the target; the max over sets
        # is the smallest threshold meeting it everywhere.
        need = max(float(np.quantile(s, 1.0 - target)) for s in maize_scores.values())
        thr = float(grid[np.searchsorted(grid, need)]) if need <= grid[-1] else float(need)
        rows.append({
            "max_false_flag_target": target,
            "threshold": round(thr, 4),
            "maize_false_flag_rate": {k: round(float((s > thr).mean()), 4)
                                      for k, s in maize_scores.items()},
            "foreign_catch_rate": {k: round(float((s > thr).mean()), 4)
                                   for k, s in foreign_scores.items()},
        })
    return rows


def _summarise(maize_scores, foreign_scores, cal, percentiles):
    rows = []
    for pct in percentiles:
        thr = float(np.percentile(cal, pct))
        rows.append({
            "calibration_percentile": pct,
            "threshold": round(thr, 4),
            "maize_false_flag_rate": {k: round(float((s > thr).mean()), 4)
                                      for k, s in maize_scores.items()},
            "foreign_catch_rate": {k: round(float((s > thr).mean()), 4)
                                   for k, s in foreign_scores.items()},
        })

    auroc = {
        f"{fname}_vs_{mname}": round(mi.auroc(fs, ms), 4)
        for fname, fs in foreign_scores.items()
        for mname, ms in maize_scores.items()
    }
    medians = {
        "maize": {k: round(float(np.median(s)), 4) for k, s in maize_scores.items()},
        "foreign": {k: round(float(np.median(s)), 4) for k, s in foreign_scores.items()},
    }
    return {
        "operating_points": rows,
        "worst_case_operating_points": _worst_case_points(
            maize_scores, foreign_scores, MAX_FALSE_FLAG),
        "auroc": auroc,
        "median_distance": medians,
    }


def _score_gate(reference, calibration, maize_sets, foreign_sets, percentiles):
    cal = mi.knn_distance(calibration, reference)
    maize_scores = {k: mi.knn_distance(v, reference) for k, v in maize_sets.items()}
    foreign_scores = {k: mi.knn_distance(v, reference) for k, v in foreign_sets.items()}
    result = _summarise(maize_scores, foreign_scores, cal, percentiles)
    return result, cal, maize_scores, foreign_scores


def _print(name, result, maize_names, foreign_names):
    print(f"\n=== {name} ===")
    head = f"{'cal pct':>8}{'thresh':>9}"
    for m in maize_names:
        head += f"{('FF ' + m)[:20]:>21}"
    for f in foreign_names:
        head += f"{('CATCH ' + f)[:20]:>21}"
    print(head)
    print("-" * len(head))
    for row in result["operating_points"]:
        line = f"{row['calibration_percentile']:>8}{row['threshold']:>9.4f}"
        for m in maize_names:
            line += f"{row['maize_false_flag_rate'][m] * 100:20.1f}%"
        for f in foreign_names:
            line += f"{row['foreign_catch_rate'][f] * 100:20.1f}%"
        print(line)
    print(f"{'max FF':>8}{'thresh':>9}"
          + "".join(f"{('FF ' + m)[:20]:>21}" for m in maize_names)
          + "".join(f"{('CATCH ' + f)[:20]:>21}" for f in foreign_names))
    for row in result["worst_case_operating_points"]:
        line = f"{row['max_false_flag_target'] * 100:>7.0f}%{row['threshold']:>9.4f}"
        for m in maize_names:
            line += f"{row['maize_false_flag_rate'][m] * 100:20.1f}%"
        for f in foreign_names:
            line += f"{row['foreign_catch_rate'][f] * 100:20.1f}%"
        print(line)
    print("  AUROC: " + "  ".join(f"{k}={v}" for k, v in result["auroc"].items()))


def _first_detection_mask(counts):
    """True for the crop YOLO found first in each image.

    Needed because a size-matched control on this data reads as fatal until you
    look at what sits in the overlapping band. Same-rig maize photographs hold
    one kernel each, so every crop after the first is a fragment of it -- small,
    and small in exactly the size range the impurities occupy. Comparing all
    crops pits impurities against the detector's own mistakes rather than
    against maize.
    """
    mask = []
    for n in counts:
        mask.extend(j == 0 for j in range(int(n)))
    return np.asarray(mask, bool)


def _size_corrected(pipe, model, tf, reference, calibration, cal_sizes,
                    maize_sets, foreign_sets, results):
    """The gate as it actually ships, and the finding that shaped it.

    The raw kNN distance falls steadily as crops get larger, so part of what an
    earlier build reported as maize identity was image resolution. The
    correction is fitted here by the same code that fits it at build time, on
    the same held-out calibration crops, so this file measures the transform
    that serves rather than a re-derivation of it.
    """
    sizes = mi.evaluation_crop_sizes()
    missing = [k for k in list(maize_sets) + list(foreign_sets)
               if sizes.get(k) is None]
    if cal_sizes is None or missing:
        print("  crop sizes unavailable; skipping the size-corrected section")
        return

    cal_raw = mi.knn_distance(calibration, reference)
    coef, r2, resid_corr = _fit_size_correction(cal_raw, cal_sizes)
    cal_corr = _score(cal_raw, cal_sizes, coef)
    raw = {k: mi.knn_distance(v, reference)
           for k, v in {**maize_sets, **foreign_sets}.items()}
    corr = {k: _score(v, sizes[k], coef) for k, v in raw.items()}

    m_corr = {k: corr[k] for k in maize_sets}
    f_corr = {k: corr[k] for k in foreign_sets}
    summary = _summarise(m_corr, f_corr, cal_corr, PERCENTILES)
    summary["size_correction"] = {
        "form": "knn_distance - (a + b*log(px) + c*log(px)^2)",
        "coefficients": [round(float(c), 6) for c in coef],
        "fitted_on": "held-out calibration crops only, never on evaluation sets",
        "r2_of_raw_distance_on_crop_size": round(float(r2), 4),
        "residual_correlation_with_log_size": round(float(resid_corr), 4),
        "finding": (
            "R2 = %.3f of the raw distance was explained by crop short side "
            "alone. GrainSet impurity crops are far smaller than same-rig maize "
            "crops, so an uncorrected score is rewarded for a difference that "
            "has nothing to do with what the object is." % r2
        ),
    }
    results["gates"]["maize_identity_size_corrected"] = summary
    _print("maize_identity, size-corrected (SHIPPED)", summary,
           list(maize_sets), list(foreign_sets))

    # AUROC before and after, over all crops and over first detections only. The
    # corrected numbers are lower, which is the expected direction: part of the
    # raw margin really was resolution rather than identity.
    counts = {}
    for _kind, name, key, paths, _loader in mi.evaluation_spec():
        c = mi.cached_counts(paths, key)
        if c is not None:
            counts[name] = c
    comparison = {}
    for fname in foreign_sets:
        for mname in maize_sets:
            entry = {
                "auroc_raw": round(mi.auroc(raw[fname], raw[mname]), 4),
                "auroc_size_corrected": round(mi.auroc(corr[fname], corr[mname]), 4),
            }
            fm = _first_detection_mask(counts.get(fname, []))
            mm = _first_detection_mask(counts.get(mname, []))
            if len(fm) == len(raw[fname]) and len(mm) == len(raw[mname]):
                entry["auroc_raw_first_detection_only"] = round(
                    mi.auroc(raw[fname][fm], raw[mname][mm]), 4)
                entry["auroc_size_corrected_first_detection_only"] = round(
                    mi.auroc(corr[fname][fm], corr[mname][mm]), 4)
                entry["first_detection_fraction"] = {
                    fname: round(float(fm.mean()), 4),
                    mname: round(float(mm.mean()), 4),
                }
            comparison["%s_vs_%s" % (fname, mname)] = entry
    results["auroc_raw_vs_size_corrected"] = comparison
    print("\n  AUROC, raw -> size-corrected (first detection only in brackets):")
    for pair, e in comparison.items():
        extra = e.get("auroc_size_corrected_first_detection_only")
        print("    %-52s %.4f -> %.4f%s"
              % (pair, e["auroc_raw"], e["auroc_size_corrected"],
                 "  [%.4f]" % extra if extra is not None else ""))

    # Median crop size per set, which is the whole reason the correction exists.
    results["crop_size_px"] = {
        k: {"median": int(np.median(v)),
            "p05": int(np.percentile(v, 5)),
            "p95": int(np.percentile(v, 95))}
        for k, v in sizes.items() if v is not None and len(v)
    }


def _scene_coverage(results):
    """The densest scene the gate was calibrated on, and why that is a hard limit.

    Recorded as a measured coverage gap rather than a solved problem. On a real
    upload of 300 detections -- a packed bed of ordinary maize, verified by
    inspection -- 42% of objects were surfaced against a 5% budget, while
    calibration crops of the same size behave at 6-8%. Crop size does not
    explain it and the size correction does not touch it: in a packed scene
    every crop is filled with fragments of its neighbours instead of background.
    Serving declines above this count.
    """
    # The limit itself comes from the build script rather than being recomputed,
    # so the number reported here is the number the checkpoint ships.
    peak = _scene_limit(log=lambda *a, **k: None) or 0
    per_set = {}
    for key, paths, _loader in mi.calibration_spec():
        c = mi.cached_counts(paths, key)
        if c is not None and len(c):
            per_set[key] = int(c.max())
    for _kind, name, key, paths, _loader in mi.evaluation_spec():
        c = mi.cached_counts(paths, key)
        if c is not None and len(c):
            per_set[name] = int(c.max())
    results["scene_coverage"] = {
        "max_objects_per_image_by_set": per_set,
        "calibration_scene_limit": peak,
        "finding": (
            "No image in the reference or calibration corpora yields more than "
            "%d detected "
            "objects. A real upload yielding 300 had 42%% of its objects "
            "surfaced at a 5%% budget with no foreign object present. Serving "
            "returns unavailable above the calibrated limit rather than scoring "
            "a scene the gate has never been measured on." % peak
        ),
    }
    print("\n  scene coverage: at most %d objects per image anywhere in "
          "reference or calibration" % peak)


def main():
    pipe = AnalysisPipeline()
    model, _, _ = pipe._get_unified_model()
    tf = pipe._get_eval_transform()

    print("building the maize-identity reference...")
    reference, provenance = mi.build_reference(pipe, model, tf)
    for p in provenance:
        print(f"  {p['count']:>6}  {p['source']}")
    print(f"  total: {reference.shape[0]} embeddings, dim {reference.shape[1]}")

    print("embedding held-out maize for calibration...")
    calibration, n_cal = mi.calibration_embeddings(pipe, model, tf)
    cal_sizes = mi.calibration_sizes()
    print(f"  {n_cal} crops")

    print("embedding evaluation sets...")
    maize_sets, foreign_sets = mi.evaluation_sets(pipe, model, tf)
    for k, v in list(maize_sets.items()) + list(foreign_sets.items()):
        print(f"  {v.shape[0]:>6}  {k}")

    maize_names = list(maize_sets)
    foreign_names = list(foreign_sets)

    # The detector is the first filter, and nothing it misses is ever scored.
    # Reported next to the catch rate because the two multiply: end-to-end recall
    # is at most detection_rate x catch_rate, and quoting the catch rate alone
    # overstates the flag by about a fifth.
    detection = {}
    for name, (paths, key) in {
        "grainset_impurities": (mi.grainset_paths("train/7_IM")
                                + mi.grainset_paths("test/7_IM"), "ev_grainset_im"),
        "grainset_sound_same_rig": (mi.grainset_paths("test/0_NOR"), "ev_grainset_nor"),
        "unified_test": ([r["filepath"] for r in mi.unified_rows("test")],
                         "ev_unified_test"),
    }.items():
        counts = mi.cached_counts(paths, key)
        if counts is not None and len(counts):
            detection[name] = {
                "images": int(len(counts)),
                "detected": int((counts > 0).sum()),
                "detection_rate": round(float((counts > 0).mean()), 4),
                "objects_per_image": round(float(counts.mean()), 3),
            }
    for name, d in detection.items():
        print(f"  detection {name:<32} {d['detection_rate'] * 100:5.1f}% "
              f"({d['detected']}/{d['images']} images)")

    results = {
        "reference_provenance": provenance,
        "domain": (
            "serving domain -- every embedding is a YOLO detector crop, matching "
            "what the gate is handed at inference. An earlier build embedded whole "
            "dataset images and its calibrated guarantee did not transfer."
        ),
        "detection": detection,
        "calibration_crops": n_cal,
        "evaluation_counts": {**{k: int(v.shape[0]) for k, v in maize_sets.items()},
                              **{k: int(v.shape[0]) for k, v in foreign_sets.items()}},
        "gates": {},
    }

    m_res, m_cal, m_maize, m_foreign = _score_gate(
        reference, calibration, maize_sets, foreign_sets, PERCENTILES)
    results["gates"]["maize_identity"] = m_res
    _print("maize_identity (reference = all maize corpora)",
           m_res, maize_names, foreign_names)

    # The shipped gate on the same data, so "the existing gate was tried first" is
    # a number in a file rather than an assertion in a report.
    qref_path = "outputs/checkpoints/quality_reference.npz"
    if os.path.exists(qref_path):
        z = np.load(qref_path)
        qref = z["reference"]
        print(f"\nscoring the existing quality gate ({qref.shape[0]} reference "
              f"embeddings, shipped threshold {float(z['threshold']):.4f})...")
        q, q_cal, q_maize, q_foreign = _score_gate(
            qref, calibration, maize_sets, foreign_sets, PERCENTILES)
        # Its shipped threshold is the operating point that actually serves today.
        shipped = float(z["threshold"])
        q["shipped_threshold"] = round(shipped, 4)
        q["at_shipped_threshold"] = {
            "maize_false_flag_rate": {k: round(float((v > shipped).mean()), 4)
                                      for k, v in q_maize.items()},
            "foreign_catch_rate": {k: round(float((v > shipped).mean()), 4)
                                   for k, v in q_foreign.items()},
        }
        results["gates"]["quality_gate"] = q
        _print("quality_gate (reference = Mendeley quality train only)",
               q, maize_names, foreign_names)
        print(f"  at its shipped threshold {shipped:.4f}:")
        for k, v in q["at_shipped_threshold"]["maize_false_flag_rate"].items():
            print(f"    maize false flag  {k:<32} {v * 100:5.1f}%")
        for k, v in q["at_shipped_threshold"]["foreign_catch_rate"].items():
            print(f"    foreign caught    {k:<32} {v * 100:5.1f}%")

        # Two references, two independent questions, one AND. The references
        # overlap only in dataset q, so agreement is not a foregone conclusion --
        # and a flag both gates raise is the one least likely to be domain shift.
        both_maize = {k: np.minimum(_rank(m_maize[k], m_cal), _rank(q_maize[k], q_cal))
                      for k in m_maize}
        both_foreign = {k: np.minimum(_rank(m_foreign[k], m_cal), _rank(q_foreign[k], q_cal))
                        for k in m_foreign}
        both_cal = np.minimum(_rank(m_cal, m_cal), _rank(q_cal, q_cal))
        both = _summarise(both_maize, both_foreign, both_cal, PERCENTILES)
        results["gates"]["both_agree"] = both
        _print("both_agree (each score as its own calibration percentile, "
               "then the lower of the two)", both, maize_names, foreign_names)

    # The shipped configuration last, so a reader who stops early has still seen
    # the two gates Phase 4 required be compared before it.
    print("\n=== size-corrected scoring (what the platform serves) ===")
    _size_corrected(pipe, model, tf, reference, calibration, cal_sizes,
                    maize_sets, foreign_sets, results)
    _scene_coverage(results)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
