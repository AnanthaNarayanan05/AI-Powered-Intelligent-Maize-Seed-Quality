"""Contact sheets for reviewing SAM defect-mask proposals in bulk.

The review server shows one kernel at a time, which is the right shape for a
person with a mouse. This module renders the same information as a grid so a
vision model can inspect many proposals in one pass and record a decision per
row. It reads the proposal CSV and writes PNG sheets; it never writes a
decision. Deciding is done by src/annotation/apply_sheet_decisions.py, which is
the only thing that touches the CSV.

WHAT A SHEET SHOWS. One kernel per row. The leftmost panel is the crop with the
proposed kernel body outlined, so the body mask itself can be judged -- a body
that has swallowed the plate or clipped half the kernel invalidates every
candidate computed inside it. The remaining panels are the candidate defect
masks, each overlaid on the same crop and captioned with its true candidate
index, its source and its coverage of the body. The index in the caption is the
index in candidates_json, not the position on the sheet, so a decision recorded
against it survives any change to display order.

WHY THE DISPLAY ORDER IS NOT THE RANKING. Candidates are ordered for legibility
-- SAM-refined regions first, then raw colour unions, then the whole body -- but
that order carries no claim about which is correct. It exists so that panels of
the same kind sit next to each other and differences between them are visible.
The reviewer is expected to reject every candidate on a kernel where none of
them is a defect, and the sheets are built to make that easy: a kernel whose
candidates are all rim crescents or specular highlights should look obviously
wrong at this size.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROPOSAL_CSV = "data_processed/annotation/proposals.csv"

CELL = 170          # px per panel, before the caption strip
CAPTION_H = 26      # px of caption under each panel
PAD = 6
BG = (18, 20, 24)
FG = (222, 226, 232)
MUTED = (140, 148, 160)


def _source_rank(source: str) -> tuple:
    """Legibility order only: SAM regions, then colour unions, then whole body."""
    if source.startswith("sam@"):
        return (0, source)
    if source.startswith("colour@"):
        return (1, source)
    return (2, source)


def _fit(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    out = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                     interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:] = BG
    y, x = (size - out.shape[0]) // 2, (size - out.shape[1]) // 2
    canvas[y:y + out.shape[0], x:x + out.shape[1]] = out
    return canvas


def _overlay(crop: np.ndarray, mask: np.ndarray, colour, alpha: float = 0.45) -> np.ndarray:
    out = crop.copy()
    if mask is not None and mask.any():
        tint = np.zeros_like(out)
        tint[:] = colour
        m = mask.astype(bool)
        out[m] = (out[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)
        edges = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                                 np.ones((3, 3), np.uint8)).astype(bool)
        out[edges] = colour
    return out


def _load_row_panels(row: dict, max_cands: int) -> tuple[list, list]:
    """Return (panels, captions) for one kernel, or ([], []) if unreadable."""
    crop = cv2.imread(row["image_path"])
    if crop is None:
        return [], []
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    body = None
    if row["body_mask"]:
        bp = os.path.join(row["proposal_dir"], row["body_mask"])
        b = cv2.imread(bp, cv2.IMREAD_GRAYSCALE)
        if b is not None:
            body = b > 127

    first = crop.copy()
    if body is not None:
        edges = cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_GRADIENT,
                                 np.ones((3, 3), np.uint8)).astype(bool)
        first[edges] = (90, 200, 120)
    panels = [first]
    captions = [f"#{row['_idx']} {row['grainspace_class']} {row['split']}"
                f" | body {'ok' if row['body_plausible'] == '1' else 'IMPLAUSIBLE'}"]

    cands = json.loads(row["candidates_json"] or "[]")
    indexed = list(enumerate(cands))
    indexed.sort(key=lambda t: _source_rank(t[1]["source"]))
    for true_idx, cand in indexed[:max_cands]:
        m = cv2.imread(os.path.join(row["proposal_dir"], cand["file"]), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        panels.append(_overlay(crop, m > 127, (235, 70, 70)))
        captions.append(f"[{true_idx}] {cand['source']} {cand['coverage_pct']:.1f}%")
    return panels, captions


def build_sheet(rows: list[dict], out_path: str, max_cands: int = 5) -> dict:
    """Render one sheet. Returns a manifest of what is on it."""
    built, manifest = [], []
    for row in rows:
        panels, captions = _load_row_panels(row, max_cands)
        if not panels:
            continue
        built.append((panels, captions))
        manifest.append({"idx": int(row["_idx"]), "class": row["grainspace_class"],
                         "channel": row["defect_channel"], "split": row["split"],
                         "n_candidates": int(row["n_candidates"]),
                         "shown": [c for c in captions[1:]]})
    if not built:
        return {"sheet": out_path, "rows": []}

    cols = max(len(p) for p, _ in built)
    cell_h = CELL + CAPTION_H
    W = PAD + cols * (CELL + PAD)
    H = PAD + len(built) * (cell_h + PAD)
    sheet = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(sheet)

    for r, (panels, captions) in enumerate(built):
        y = PAD + r * (cell_h + PAD)
        for c, (panel, cap) in enumerate(zip(panels, captions)):
            x = PAD + c * (CELL + PAD)
            sheet.paste(Image.fromarray(_fit(panel, CELL)), (x, y))
            draw.text((x + 2, y + CELL + 4), cap[:34], fill=FG if c == 0 else MUTED)
    sheet.save(out_path)
    return {"sheet": out_path, "size": [W, H], "rows": manifest}


def pending_rows(csv_path: str, with_candidates: bool = True) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for i, r in enumerate(rows):
        r["_idx"] = i
    out = [r for r in rows if not r["decision"]]
    if with_candidates:
        out = [r for r in out if int(r["n_candidates"]) > 0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=PROPOSAL_CSV)
    ap.add_argument("--out-dir", default="data_processed/annotation/sheets")
    ap.add_argument("--per-sheet", type=int, default=10)
    ap.add_argument("--max-cands", type=int, default=5)
    ap.add_argument("--start", type=int, default=0, help="offset into the pending list")
    ap.add_argument("--sheets", type=int, default=1, help="how many sheets to build")
    ap.add_argument("--indices", default="", help="comma-separated CSV row indices instead")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.indices:
        wanted = {int(x) for x in args.indices.split(",") if x.strip()}
        with open(args.csv, newline="", encoding="utf-8") as fh:
            allrows = list(csv.DictReader(fh))
        for i, r in enumerate(allrows):
            r["_idx"] = i
        pending = [r for r in allrows if r["_idx"] in wanted]
    else:
        pending = pending_rows(args.csv)[args.start:]

    manifests = []
    for s in range(args.sheets):
        chunk = pending[s * args.per_sheet:(s + 1) * args.per_sheet]
        if not chunk:
            break
        name = f"sheet_{args.start + s * args.per_sheet:04d}.png"
        manifests.append(build_sheet(chunk, os.path.join(args.out_dir, name), args.max_cands))

    print(json.dumps(manifests, indent=1))


if __name__ == "__main__":
    main()
