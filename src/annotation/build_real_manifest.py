"""Turn verified review decisions into a segmentation manifest.

    python -m src.annotation.build_real_manifest

Reads data_processed/annotation/proposals.csv, keeps ONLY the rows a reviewer has
actually decided, and writes a per-channel manifest plus a provenance sidecar that
src.segmentation.train_seg copies into the checkpoint and the metrics file.

WHAT MAKES A ROW TRAINING DATA. status == "verified" and a decision in
{accepted, corrected, drawn, no_defect}. Everything else -- still "proposed", or
"rejected" -- is dropped. A SAM proposal nobody looked at is a hypothesis, not a
label, and this file is the only place that distinction is enforced.

WHO LOOKED AT IT is a second, separate question, and it is answered from the
reviewer column rather than from a flag: a person working in the review server, or
a vision model reading contact sheets. Both are real review, one is not human
verification, and the sidecar written here says which it was in terms the
checkpoint and the metrics file repeat verbatim. See is_machine_reviewer.

PER-CHANNEL VALIDITY. GrainSpace labels one condition per kernel. A reviewer who
marks the bore holes on an AP kernel has said nothing about whether that kernel is
also cracked, so valid_cracked is 0 there and the trainer excludes the channel from
both loss and metrics (see src.segmentation.dataset). Two things are genuinely
known on every kept row:

  * seed_body -- the reviewer sees the body outline drawn over the image and
    rejects the row if it is wrong, so accepting the row certifies it.
  * every channel of a `no_defect` row -- nothing visible means nothing visible
    everywhere, which is what supplies the negative supervision the defect
    channels need.

LEAKAGE. Splits come from the plate id (src.annotation.splits), so every kernel
photographed on one plate lands in one split. That is re-derived here rather than
trusted from the proposals file, exact-duplicate images are dropped, and both
invariants are asserted before anything is written.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict

from src.annotation.splits import assign_split, plate_id
from src.segmentation.build_synthetic_manifest import CHANNELS, DEFECT_CHANNELS

PROPOSAL_CSV = "data_processed/annotation/proposals.csv"
OUT_CSV = "data_processed/manifest_segmentation_real.csv"

USABLE_DECISIONS = {"accepted", "corrected", "drawn", "no_defect"}
MASK_DECISIONS = {"accepted", "corrected", "drawn"}

# Reviewer identifiers that are NOT a person. Provenance is derived from the data
# rather than from a command-line flag, because a flag is exactly the thing that
# gets forgotten on a rerun -- and the failure mode of forgetting it here is a
# checkpoint claiming human verification it never had.
MACHINE_REVIEWERS = {"", "auto", "model_qc", "machine"}

# A vision model reading contact sheets (src/annotation/apply_sheet_decisions.py)
# is a real annotation method and its masks are real judgements, but it is not a
# person and must never be counted as one. Any reviewer string carrying a model
# family name is machine review, so a future reviewer id does not have to be added
# to a list here before the provenance comes out right.
MODEL_REVIEWER_MARKERS = ("claude", "gpt", "gemini", "llama", "qwen", "sam",
                          "model", "auto", "machine", "bot")


def is_machine_reviewer(reviewer: str) -> bool:
    r = (reviewer or "").strip().lower()
    if r in MACHINE_REVIEWERS:
        return True
    return any(m in r for m in MODEL_REVIEWER_MARKERS)


def file_digest(path: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(proposal_csv: str = PROPOSAL_CSV) -> tuple[list[dict], dict]:
    """Return (manifest rows, report). Writes nothing."""
    if not os.path.exists(proposal_csv):
        raise SystemExit(f"{proposal_csv} not found. Run src/annotation/sam_propose.py first.")
    with open(proposal_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    report: dict = {
        "n_proposals": len(rows),
        "status_counts": dict(Counter(r.get("status", "") for r in rows)),
        "decision_counts": dict(Counter(r.get("decision", "") for r in rows if r.get("decision"))),
        "dropped": defaultdict(int),
    }

    out, seen_digest = [], {}
    for r in rows:
        if r.get("status") != "verified" or r.get("decision") not in USABLE_DECISIONS:
            report["dropped"]["not_verified"] += 1
            continue

        decision = r["decision"]
        channel = r.get("defect_channel") or ""
        if decision in MASK_DECISIONS and not channel:
            # A mask with no channel to put it in. NOR rows carry no defect channel;
            # if a reviewer found something there the honest outcome is to fix the
            # proposal, not to let this script guess which defect it was.
            report["dropped"]["mask_without_channel"] += 1
            continue

        final = os.path.join(r["proposal_dir"], r.get("final_mask") or "")
        if decision in MASK_DECISIONS and not os.path.exists(final):
            report["dropped"]["final_mask_missing"] += 1
            continue

        body = os.path.join(r["proposal_dir"], r["body_mask"])
        if not os.path.exists(body) or not os.path.exists(r["image_path"]):
            report["dropped"]["source_file_missing"] += 1
            continue

        # Exact-duplicate images: GrainSpace crops overlapping detections, so the same
        # kernel can appear twice under different coordinates in the filename. Two
        # copies of one kernel either side of a split boundary is test-set leakage.
        digest = file_digest(r["image_path"])
        if digest in seen_digest:
            report["dropped"]["duplicate_image"] += 1
            continue
        seen_digest[digest] = r["image_path"]

        group = plate_id(r["image_path"])
        row = {
            "image_path": r["image_path"].replace("\\", "/"),
            "split": assign_split(group),
            "group": group,
            "grainspace_class": r["grainspace_class"],
            "decision": decision,
            "reviewer": r.get("reviewer", ""),
            "reviewed_at": r.get("reviewed_at", ""),
            "body_px": r.get("body_px", ""),
        }
        for ch in CHANNELS:
            if ch == "seed_body":
                mask, valid = body.replace("\\", "/"), 1
            elif decision == "no_defect":
                # Certified empty. The empty path is read as an all-zero plane by
                # src.segmentation.dataset rather than stored as hundreds of black PNGs.
                mask, valid = "", 1
            elif ch == channel:
                mask, valid = final.replace("\\", "/"), 1
            else:
                mask, valid = "", 0          # never asked, never answered
            row["mask_" + ch] = mask
            row["valid_" + ch] = valid
        out.append(row)

    report["dropped"] = dict(report["dropped"])
    report["n_kept"] = len(out)
    report["split_counts"] = dict(Counter(r["split"] for r in out))
    report["class_by_split"] = {
        s: dict(Counter(r["grainspace_class"] for r in out if r["split"] == s))
        for s in ("train", "val", "test")
    }
    report["annotated_per_channel"] = {
        ch: {s: sum(1 for r in out if r["split"] == s and r["valid_" + ch] == 1)
             for s in ("train", "val", "test")}
        for ch in CHANNELS
    }
    report["positive_per_channel"] = {
        ch: {s: sum(1 for r in out if r["split"] == s and r["mask_" + ch])
             for s in ("train", "val", "test")}
        for ch in DEFECT_CHANNELS
    }
    report["reviewers"] = sorted({r["reviewer"] for r in out})

    # Invariants, checked before anything is written. A leaked split discovered
    # after training is a retraction; discovered here it is a bug report.
    by_group = defaultdict(set)
    for r in out:
        by_group[r["group"]].add(r["split"])
    straddling = {g: sorted(s) for g, s in by_group.items() if len(s) > 1}
    if straddling:
        raise SystemExit(f"plate groups straddle splits (this must never happen): {straddling}")
    paths = Counter(r["image_path"] for r in out)
    repeated = [p for p, n in paths.items() if n > 1]
    if repeated:
        raise SystemExit(f"{len(repeated)} image paths appear more than once: {repeated[:5]}")
    report["n_groups"] = len(by_group)
    return out, report


def provenance(rows: list[dict], report: dict) -> dict:
    """Provenance derived from the review record, not asserted by the operator."""
    human = [r for r in rows if not is_machine_reviewer(r["reviewer"])]
    machine = [r for r in rows if is_machine_reviewer(r["reviewer"])]
    named = sorted({r["reviewer"].strip() for r in machine if r["reviewer"].strip()})
    if rows and not machine:
        kind = "sam_proposed_human_verified"
        verified_by = "Every mask below was reviewed by a person."
    elif rows and not human:
        kind = "sam_proposed_model_verified"
        verified_by = (
            "Every mask below was reviewed by " + (", ".join(named) or "a model") +
            " reading contact sheets, not by a person. That is model-assisted "
            "annotation: each proposal was actually inspected and judged, and the "
            "obviously wrong ones -- rim crescents, specular highlights, sub-percent "
            "specks, whole-body masks -- were rejected rather than accepted. But the "
            "boundaries are the proposer's, the review resolution was a 170px panel, "
            "and no person has checked any of it. These are MODEL-VERIFIED masks and "
            "MUST NOT be described as human-verified ground truth, nor any metric "
            "measured against them as real-world human-validated performance.")
    else:
        kind = "sam_proposed_mixed_review"
        verified_by = (
            f"{len(human)}/{len(rows)} rows carry a human reviewer; the remaining "
            f"{len(machine)} were reviewed by " + (", ".join(named) or "a model") +
            ". The set as a whole is NOT human-verified ground truth.")
    return {
        "label_provenance": kind,
        "label_note": (
            "Real maize kernel photographs (GrainSpace M600 val pool). Masks were "
            "PROPOSED by SAM ViT-B plus a healthy-colour saliency cue and then "
            f"accepted, corrected, drawn or marked no-defect by a reviewer. {verified_by} "
            "seed_body is a SAM proposal the reviewer confirmed by not rejecting the "
            "row. Defect channels are per-image: an unannotated channel carries "
            "valid=0 and is excluded from loss and metrics, because GrainSpace assigns "
            "one condition per kernel and the reviewer was never asked about the "
            "others. Coverage derived from these masks is VISIBLE DEFECT AREA %, not a "
            "physically calibrated area."),
        "source_dataset": "GrainSpace maize M600 (val split)",
        "annotation_tool": ("src/annotation/sam_propose.py -> "
                            "src/annotation/review_server.py (human) | "
                            "src/annotation/review_sheets.py -> "
                            "src/annotation/apply_sheet_decisions.py (model)"),
        "reviewers": report["reviewers"],
        "n_human_reviewed": len(human),
        "n_model_reviewed": len(machine),
        "n_images": report["n_kept"],
        "decision_counts": dict(Counter(r["decision"] for r in rows)),
        "split_unit": "plate id (src.annotation.splits.plate_id); no plate spans two splits",
        "duplicates_dropped": report["dropped"].get("duplicate_image", 0),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposals", default=PROPOSAL_CSV)
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--min-rows", type=int, default=1,
                    help="refuse to write a manifest with fewer verified rows than this")
    args = ap.parse_args()

    rows, report = build(args.proposals)

    print(f"proposals            {report['n_proposals']}")
    print(f"  status             {report['status_counts']}")
    print(f"  decisions          {report['decision_counts']}")
    print(f"  dropped            {report['dropped']}")
    print(f"verified rows kept   {report['n_kept']}  ({report.get('n_groups', 0)} plate groups)")
    if not rows:
        raise SystemExit(
            "\nNothing has been verified yet, so there is no real-defect training data.\n"
            "Run:  python -m src.annotation.review_server\n"
            "and decide the proposals at http://127.0.0.1:8011 first."
        )
    if len(rows) < args.min_rows:
        raise SystemExit(f"\nonly {len(rows)} verified rows, --min-rows is {args.min_rows}")

    print(f"  splits             {report['split_counts']}")
    for s in ("train", "val", "test"):
        print(f"    {s:5s} classes    {report['class_by_split'][s]}")
    print("  annotated / positive per channel (train/val/test)")
    for ch in CHANNELS:
        a = report["annotated_per_channel"][ch]
        p = report["positive_per_channel"].get(ch)
        pos = f"   positive {p['train']}/{p['val']}/{p['test']}" if p else ""
        print(f"    {ch:18s} annotated {a['train']}/{a['val']}/{a['test']}{pos}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = (["image_path", "split", "group", "grainspace_class", "decision",
               "reviewer", "reviewed_at", "body_px"]
              + ["mask_" + c for c in CHANNELS] + ["valid_" + c for c in CHANNELS])
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    prov = provenance(rows, report)
    side = os.path.splitext(args.out)[0] + ".provenance.json"
    with open(side, "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)
    rep = os.path.splitext(args.out)[0] + ".report.json"
    with open(rep, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n-> {args.out}\n-> {side}  (label_provenance={prov['label_provenance']})\n-> {rep}")
    if prov["label_provenance"] != "sam_proposed_human_verified":
        print("\nNOTE: not every row has a human reviewer. The sidecar says so, and the "
              "checkpoint and metrics file will repeat it verbatim.")
        print(f"      human {prov['n_human_reviewed']} / model {prov['n_model_reviewed']}"
              f"   -> {prov['label_provenance']}")


if __name__ == "__main__":
    main()
