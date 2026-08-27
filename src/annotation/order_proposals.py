"""Interleave the proposal order so a PARTIAL review is still a usable dataset.

    python -m src.annotation.order_proposals

sam_propose walks the class directories in turn, so proposals.csv arrives in
blocks: 100 AP, then 200 BN, then 100 FM, 300 MY, 400 NOR. Reviewing 1,100
kernels is a long sitting, and a reviewer who stops a third of the way through a
block-ordered file has annotated insect damage and cracks and nothing else --
no mould, and, worse, no NOR kernels, which are the only rows that certify a
channel is EMPTY. The result trains a model with no negative supervision.

Round-robin over (class, split) fixes that. Every prefix of the reordered file is
approximately stratified, so the reviewer can stop whenever they like and
build_real_manifest still sees every channel in every split.

Refuses to run once any row has been reviewed: the review server addresses rows
by index, so reordering mid-session would reassign decisions to other kernels.
"""
from __future__ import annotations

import csv
import os
import shutil
from collections import Counter, defaultdict

PROPOSAL_CSV = "data_processed/annotation/proposals.csv"


def interleave(rows: list[dict]) -> list[dict]:
    """Spread every (grainspace_class, split) stratum evenly over the whole file.

    Each row gets the fractional position (i + 0.5) / n within its own stratum and
    everything is sorted on that. A stratum of 298 and a stratum of 13 both end up
    spanning the full length, so ANY prefix is approximately proportional to the
    full distribution -- which is the property that makes stopping early safe.

    Dealing from the largest remaining stratum instead does not work: with 298 NOR
    against 226 MY it empties the difference first and opens with twenty
    consecutive NOR kernels, which is the block problem in miniature.

    Within a stratum the original order is kept, so kernels from one plate stay
    adjacent -- reviewing a plate at a time is faster than jumping between lighting
    conditions on consecutive kernels.
    """
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r["grainspace_class"], r["split"])].append(r)

    keyed = []
    for k, items in strata.items():
        n = len(items)
        for i, r in enumerate(items):
            # `k` breaks ties deterministically; without it, equal positions would be
            # ordered by dict insertion and the file would differ between runs.
            keyed.append(((i + 0.5) / n, k, r))
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in keyed]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=PROPOSAL_CSV)
    ap.add_argument("--head", type=int, default=20, help="rows to print as a preview")
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    decided = [r for r in rows if r.get("status") and r["status"] != "proposed"]
    if decided:
        raise SystemExit(
            f"{len(decided)} of {len(rows)} rows are already reviewed. Reordering now "
            "would reassign their decisions to different kernels, so this refuses to "
            "run. Finish the current review, or reorder a fresh proposals.csv."
        )

    out = interleave(rows)
    backup = args.csv + ".block_order.bak"
    if not os.path.exists(backup):
        shutil.copy2(args.csv, backup)
    tmp = args.csv + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    os.replace(tmp, args.csv)

    print(f"reordered {len(out)} proposals -> {args.csv}  (original kept at {backup})")
    print(f"first {args.head}: " + " ".join(
        f"{r['grainspace_class']}/{r['split'][:2]}" for r in out[:args.head]))
    for cut in (100, 200, 300, 400, len(out)):
        pref = out[:cut]
        cls = Counter(r["grainspace_class"] for r in pref)
        spl = Counter(r["split"] for r in pref)
        print(f"  first {cut:4d} reviewed -> classes {dict(sorted(cls.items()))}  "
              f"splits {dict(sorted(spl.items()))}")


if __name__ == "__main__":
    main()
