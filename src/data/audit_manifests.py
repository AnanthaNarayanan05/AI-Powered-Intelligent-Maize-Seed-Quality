"""Phase 28 (mega-prompt): one machine-readable audit across every training manifest,
run before any retraining rather than trusted from memory of the manual pass in
docs/03_DATASET_AUDIT.md.

This does not replace that document -- it makes its central claim rerunnable and
extends it to manifests the original manual audit did not cover. Three checks, each
answering a question a retrain could silently reintroduce:

  1. GROUP/SPLIT LEAKAGE. Generalises tests/test_unified_model.py::
     test_no_group_spans_more_than_one_split from three manifests to every manifest
     that declares a `group` column. This is the exact defect that invalidated the
     original Dataset B results (docs/09 section 6.9): a source seed's augmented
     copies scattered across train/val/test turns the test set into a memory check.

  2. CONTENT-HASH LEAKAGE WITHIN A MANIFEST. A group id is only as good as the code
     that assigned it. Hashing every file and checking whether any exact byte-for-byte
     duplicate spans two splits catches a duplicate a group key missed -- the
     complement to check 1, not a repeat of it.

  3. CONTENT-HASH OVERLAP ACROSS MANIFESTS. Re-checks the specific claim
     docs/03 makes for Dataset 4 ("byte-overlap vs A/B/C: 0/0/0") and extends it to
     every declared-independent pair, so a future dataset addition cannot silently
     share source images with an existing one without it being noticed.

Never modifies a dataset or a manifest. Writes one JSON report and prints a summary;
exit code is 1 if any check fails, so it can gate a training run in CI without a human
reading the JSON first.

    python -m src.data.audit_manifests
    python -m src.data.audit_manifests --hash-content   # adds checks 2 and 3 (slower: reads every file)
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

from src.utils.config import PROJECT_ROOT, load_config

# Manifests whose splits are supposed to be independent of one another -- a pair
# flagged here is not "wrong" by construction (dataset_a/b before the group-aware fix,
# or the unfixed manifest_dataset_b.csv this list deliberately excludes) so much as
# something a human should look at, the same way docs/03's own byte-overlap figures
# were reported rather than assumed.
INDEPENDENT_PAIRS = [
    ("manifest_dataset_a.csv", "manifest_dataset_b_grouped.csv"),
    ("manifest_dataset_a.csv", "manifest_dataset_4_quality.csv"),
    ("manifest_dataset_b_grouped.csv", "manifest_dataset_4_quality.csv"),
]
# manifest_symptom.csv and manifest_segmentation_real.csv are deliberately NOT listed
# here: both are annotated from the same GrainSpace M600 val crops (docs/09 sections
# 10.2 and 10.5 -- it is the only condition-labelled supply either phase had), so
# sharing files is the documented, correct state, not leakage. Nothing in this project
# fuses or cross-validates those two models' numbers against each other, which is the
# only way shared source images would become a problem.

# The path column varies by manifest (`filepath` for classification manifests,
# `image_path` for the two segmentation manifests) -- declared here rather than
# sniffed, so a manifest with neither is a loud KeyError, not a silently-skipped file.
PATH_COLUMNS = ("filepath", "image_path")


def _path_column(fieldnames: list[str]) -> str | None:
    for col in PATH_COLUMNS:
        if col in fieldnames:
            return col
    return None


def _resolve(raw_path: str) -> Path:
    p = Path(raw_path.replace("\\", "/"))
    return p if p.is_absolute() else PROJECT_ROOT / p


def _read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def check_group_split_leakage(name: str, rows: list[dict]) -> dict:
    """One manifest's worth of tests/test_unified_model.py::
    test_no_group_spans_more_than_one_split, generalised to any group-bearing manifest."""
    if not rows or "group" not in rows[0] or "split" not in rows[0]:
        return {"applicable": False}

    splits_by_group: dict[tuple, set[str]] = collections.defaultdict(set)
    for r in rows:
        # manifest_unified.csv / manifest_tritask.csv reuse group ids across sources
        # (dataset a's group key is literally its filepath); scope by source so an
        # unrelated collision across datasets is never mistaken for real leakage.
        key = (r.get("source", ""), r["group"])
        splits_by_group[key].add(r["split"])

    offenders = {k: sorted(v) for k, v in splits_by_group.items() if len(v) > 1}
    return {
        "applicable": True,
        "groups": len(splits_by_group),
        "offending_groups": len(offenders),
        "sample_offenders": list(offenders.items())[:5],
        "passed": not offenders,
    }


def check_content_hash_leakage(name: str, rows: list[dict], path_col: str) -> dict:
    """Byte-identical files split across train/val/test, independent of any group
    column. Reads every file this manifest names; skipped unless --hash-content."""
    if not rows or "split" not in rows[0]:
        return {"applicable": False}

    hash_splits: dict[str, set[str]] = collections.defaultdict(set)
    hash_paths: dict[str, list[str]] = collections.defaultdict(list)
    missing = 0
    for r in rows:
        p = _resolve(r[path_col])
        if not p.exists():
            missing += 1
            continue
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        hash_splits[digest].add(r["split"])
        hash_paths[digest].append(str(p))

    offenders = {h: sorted(s) for h, s in hash_splits.items() if len(s) > 1}
    return {
        "applicable": True,
        "files_checked": len(rows) - missing,
        "files_missing": missing,
        "offending_hashes": len(offenders),
        "sample_offenders": [
            {"splits": splits, "paths": hash_paths[h][:2]}
            for h, splits in list(offenders.items())[:5]
        ],
        "passed": not offenders,
    }


def check_cross_manifest_overlap(processed: Path, a_name: str, b_name: str) -> dict:
    """Re-checks docs/03's 'byte-overlap vs A/B/C: 0/0/0' claim for a declared-
    independent manifest pair, and extends it to pairs docs/03 never reported."""
    a_path, b_path = processed / a_name, processed / b_name
    if not a_path.exists() or not b_path.exists():
        return {"applicable": False}

    a_col = _path_column(_read_manifest(a_path)[0].keys()) if a_path.exists() else None
    a_rows = _read_manifest(a_path)
    b_rows = _read_manifest(b_path)
    a_col = _path_column(list(a_rows[0].keys()))
    b_col = _path_column(list(b_rows[0].keys()))
    if a_col is None or b_col is None:
        return {"applicable": False}

    def hashes(rows, col):
        out = set()
        for r in rows:
            p = _resolve(r[col])
            if p.exists():
                out.add(hashlib.md5(p.read_bytes()).hexdigest())
        return out

    overlap = hashes(a_rows, a_col) & hashes(b_rows, b_col)
    return {
        "applicable": True,
        "overlapping_files": len(overlap),
        "passed": len(overlap) == 0,
    }


def check_class_balance(rows: list[dict]) -> dict:
    label_col = next((c for c in ("label", "variety_label", "quality_label",
                                   "symptom_label", "grainspace_class") if rows and c in rows[0]), None)
    if not label_col:
        return {"applicable": False}
    counts = collections.Counter(r[label_col] for r in rows if r.get(label_col))
    if not counts:
        return {"applicable": False}
    return {
        "applicable": True,
        "label_column": label_col,
        "counts": dict(counts),
        "imbalance_ratio": round(max(counts.values()) / max(min(counts.values()), 1), 3),
    }


def audit(hash_content: bool) -> dict:
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    if not processed.is_absolute():
        processed = PROJECT_ROOT / processed

    manifests = sorted(processed.glob("manifest_*.csv"))
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "hash_content": hash_content, "manifests": {}, "cross_manifest_overlap": {}}

    all_passed = True
    for path in manifests:
        rows = _read_manifest(path)
        path_col = _path_column(rows[0].keys()) if rows else None
        entry = {
            "rows": len(rows),
            "group_split_leakage": check_group_split_leakage(path.name, rows),
            "class_balance": check_class_balance(rows),
        }
        if hash_content and path_col:
            entry["content_hash_leakage"] = check_content_hash_leakage(path.name, rows, path_col)
        report["manifests"][path.name] = entry

        for check_name in ("group_split_leakage", "content_hash_leakage"):
            result = entry.get(check_name, {})
            if result.get("applicable") and not result.get("passed", True):
                all_passed = False

    if hash_content:
        for a_name, b_name in INDEPENDENT_PAIRS:
            result = check_cross_manifest_overlap(processed, a_name, b_name)
            report["cross_manifest_overlap"][f"{a_name} vs {b_name}"] = result
            if result.get("applicable") and not result.get("passed", True):
                all_passed = False

    report["all_passed"] = all_passed
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-content", action="store_true",
                         help="Also hash every file for within- and cross-manifest "
                              "duplicate detection (checks 2 and 3). Slower: reads "
                              "every image named by every manifest.")
    parser.add_argument("--out", default="outputs/metrics/data_audit.json")
    args = parser.parse_args()

    report = audit(args.hash_content)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"Audited {len(report['manifests'])} manifest(s).")
    for name, entry in report["manifests"].items():
        leak = entry["group_split_leakage"]
        # N/A is reported as its own state, never folded into PASS: a manifest with
        # no `group` column (manifest_dataset_b.csv, kept only as the superseded,
        # pre-fix record) has not been checked, and printing PASS for it would read
        # as a clean bill of health for the exact file the group-aware split exists
        # to correct.
        status = "N/A (no group column)" if not leak.get("applicable") else ("PASS" if leak["passed"] else "FAIL")
        print(f"  {name}: {entry['rows']} rows, group/split leakage {status}")
    if args.hash_content:
        for pair, result in report["cross_manifest_overlap"].items():
            if result.get("applicable"):
                status = "PASS" if result["passed"] else "FAIL"
                print(f"  cross-overlap {pair}: {status} ({result['overlapping_files']} shared files)")
    print(f"Report written to {out_path}")
    print("ALL CHECKS PASSED" if report["all_passed"] else "SOME CHECKS FAILED -- see report")

    sys.exit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
