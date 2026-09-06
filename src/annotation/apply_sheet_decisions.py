"""Record contact-sheet review decisions into the proposal CSV.

    python -m src.annotation.apply_sheet_decisions --decisions d.json

This is the only writer to data_processed/annotation/proposals.csv besides the
interactive review server, and the two must never run at once: the server holds
every row in memory and rewrites the whole file on each decision, so an edit made
underneath it is silently reverted. This script refuses to start while port 8011
is listening.

WHO THE REVIEWER IS. --reviewer is stamped verbatim into the row and travels from
here into the manifest, the provenance sidecar, the checkpoint and the metrics
file. A vision model reading contact sheets is a real annotation method, but it is
NOT human verification, and the string recorded here is what keeps the difference
legible downstream. The default is a model identifier, and "human" is refused
outright -- a person who reviews a kernel does it in the review server, which
stamps itself.

DECISION FORMAT. A JSON object keyed by CSV row index. Long form:

    {"12": {"d": "accept", "c": 3, "note": "bore hole, clear of the rim"}}

Short form, for bulk passes -- an integer accepts that candidate index, "x"
rejects every candidate on the row, "0" records no visible defect:

    {"12": 3, "15": "x", "18": "0"}

The candidate index is the index in candidates_json, which is what the sheet
captions print in brackets. It is deliberately not the sheet position, so a
decision stays valid if the sheet is re-rendered in another order.

WHAT EACH DECISION MEANS. accept -- that candidate mask is the defect, copied to
final_mask.png. reject -- a defect may well be present but none of these masks is
it, so the row is dropped from training rather than guessed at. no_defect --
nothing visibly wrong with this kernel, which is a positive statement and supplies
the negative supervision the defect channels need. Anything not listed stays
pending.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
from collections import Counter
from datetime import datetime, timezone

PROPOSAL_CSV = "data_processed/annotation/proposals.csv"
SERVER_PORT = 8011

ACCEPT = "accepted"
REJECT = "rejected"
NO_DEFECT = "no_defect"

SHORT_REJECT = {"x", "reject", "rejected"}
SHORT_NONE = {"0", "none", "no_defect", "nd"}
ALIASES = {"accept": ACCEPT, "accepted": ACCEPT, "a": ACCEPT,
           "x": REJECT, "reject": REJECT, "rejected": REJECT,
           "0": NO_DEFECT, "none": NO_DEFECT, "no_defect": NO_DEFECT, "nd": NO_DEFECT}


def server_is_up(port: int = SERVER_PORT) -> bool:
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


def normalise(raw) -> dict:
    """Accept either the short or the long decision form; return the long one."""
    if isinstance(raw, bool):
        raise ValueError("a boolean is not a decision")
    if isinstance(raw, int):
        return {"d": ACCEPT, "c": raw, "note": ""}
    if isinstance(raw, str):
        t = raw.strip().lower()
        if t in SHORT_REJECT:
            return {"d": REJECT, "c": None, "note": ""}
        if t in SHORT_NONE:
            return {"d": NO_DEFECT, "c": None, "note": ""}
        if t.isdigit():
            return {"d": ACCEPT, "c": int(t), "note": ""}
        raise ValueError("unrecognised short decision " + repr(raw))
    if isinstance(raw, dict):
        d = str(raw.get("d") or raw.get("decision") or "").strip().lower()
        d = ALIASES.get(d, d)
        if d not in (ACCEPT, REJECT, NO_DEFECT):
            raise ValueError("unrecognised decision " + repr(raw))
        c = raw.get("c", raw.get("candidate"))
        return {"d": d, "c": None if c is None else int(c), "note": raw.get("note", "")}
    raise ValueError("unrecognised decision " + repr(raw))


def plan(rows: list[dict], decisions: dict, force: bool = False) -> list[tuple]:
    """Validate every decision before anything is written. Raises on the first batch
    of problems rather than half-applying a pass."""
    planned, errors = [], []
    for key, raw in decisions.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            errors.append(repr(key) + ": not a row index")
            continue
        if not 0 <= idx < len(rows):
            errors.append("{}: out of range (csv has {} rows)".format(idx, len(rows)))
            continue
        row = rows[idx]
        if row["decision"] and not force:
            errors.append("{}: already decided as {!r}; --force to overwrite".format(
                idx, row["decision"]))
            continue
        try:
            dec = normalise(raw)
        except ValueError as exc:
            errors.append("{}: {}".format(idx, exc))
            continue

        src_mask = None
        if dec["d"] == ACCEPT:
            cands = json.loads(row["candidates_json"] or "[]")
            if dec["c"] is None:
                errors.append("{}: accept with no candidate index".format(idx))
                continue
            if not 0 <= dec["c"] < len(cands):
                errors.append("{}: candidate {} of {}".format(idx, dec["c"], len(cands)))
                continue
            if not row["defect_channel"]:
                # NOR rows carry no channel to file a mask under, and this script is
                # not going to guess which defect was found.
                errors.append("{}: accept on a row with no defect channel".format(idx))
                continue
            src_mask = os.path.join(row["proposal_dir"], cands[dec["c"]]["file"])
            if not os.path.exists(src_mask):
                errors.append("{}: candidate mask missing at {}".format(idx, src_mask))
                continue
        planned.append((idx, dec, src_mask))

    if errors:
        raise SystemExit("refusing to write, {} problem(s):\n  {}".format(
            len(errors), "\n  ".join(errors)))
    return planned


def apply(rows: list[dict], planned: list[tuple], reviewer: str, source: str) -> dict:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    counts: Counter = Counter()
    for idx, dec, src_mask in planned:
        row = rows[idx]
        final = ""
        if src_mask:
            final = "final_mask.png"
            shutil.copyfile(src_mask, os.path.join(row["proposal_dir"], final))
        elif row["final_mask"]:
            # An overwrite that no longer carries a mask must not leave the old one
            # on disk, where the manifest builder would still find it.
            stale = os.path.join(row["proposal_dir"], row["final_mask"])
            if os.path.exists(stale):
                os.remove(stale)
        row["status"] = REJECT if dec["d"] == REJECT else "verified"
        row["reviewer"] = reviewer
        row["reviewed_at"] = stamp
        row["decision"] = dec["d"]
        row["final_mask"] = final
        row["review_note"] = "; ".join(x for x in (source, dec["note"]) if x)
        counts[dec["d"]] += 1
    return {"applied": len(planned), "by_decision": dict(counts), "reviewer": reviewer}


def save(rows: list[dict], fields: list[str], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=PROPOSAL_CSV)
    ap.add_argument("--decisions", required=True, help="JSON file keyed by CSV row index")
    ap.add_argument("--reviewer", default="claude-opus-5")
    ap.add_argument("--source", default="contact-sheet review (src/annotation/review_sheets.py)")
    ap.add_argument("--force", action="store_true", help="overwrite rows already decided")
    ap.add_argument("--dry-run", action="store_true", help="validate only; write nothing")
    args = ap.parse_args()

    if args.reviewer.strip().lower() in ("human", "person", "reviewer", "manual"):
        raise SystemExit(
            "--reviewer 'human' is refused here. This script is driven by a model "
            "reading contact sheets; stamping that as human verification would put a "
            "claim into the provenance record that nothing supports. A person reviews "
            "in src/annotation/review_server.py, which stamps itself.")
    if server_is_up():
        raise SystemExit(
            "the review server is listening on {}. It rewrites the whole CSV from "
            "memory on every decision, so it would revert this edit. Stop it "
            "first.".format(SERVER_PORT))

    with open(args.csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields, rows = list(reader.fieldnames or []), list(reader)
    with open(args.decisions, encoding="utf-8") as fh:
        decisions = json.load(fh)

    planned = plan(rows, decisions, args.force)
    if args.dry_run:
        print(json.dumps({"would_apply": len(planned), "by_decision": dict(
            Counter(d["d"] for _, d, _ in planned))}))
        return

    report = apply(rows, planned, args.reviewer, args.source)
    save(rows, fields, args.csv)

    decided = sum(1 for r in rows if r["decision"])
    print(json.dumps(report))
    print("decided {}/{}   pending {}".format(decided, len(rows), len(rows) - decided))
    print("  decisions " + str(dict(Counter(r["decision"] for r in rows if r["decision"]))))


if __name__ == "__main__":
    main()
