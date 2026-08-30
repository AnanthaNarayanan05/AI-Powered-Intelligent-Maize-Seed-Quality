"""Build the visible-symptom manifest from GrainSpace M600.

WHERE THE LABELS COME FROM
--------------------------
Every label in this manifest is a directory name in the GrainSpace M600 release
-- an expert grader assigned each kernel crop to one of seven condition
categories. Nothing here is inferred, propagated or synthesised. If a crop is
not already filed under one of those seven directories it does not enter the
manifest at all.

Only the `val/` half of the release carries condition labels. `train/` is
organised by cultivar and region (545, HF, HN, JZ, Malis, SQ, SY, WC) with no
condition annotation whatsoever, confirmed by listing the archive directly, so
its 79,208 crops cannot supply a single supervised symptom label. 1,260 is
therefore the hard ceiling on real labelled symptom data available to this
project, and the class supports below reflect that -- SD in particular has 40
crops in the entire world of this dataset.

WHY THE SPLIT IS BY PLATE
-------------------------
The crops are cut out of multi-kernel plate photographs. Two kernels off one
plate share illumination, focus, white balance and grain lot, so splitting by
IMAGE would put near-siblings on both sides and turn a memorisation number into
a reported generalisation number. This module reuses `src.annotation.splits`
unchanged rather than defining its own scheme: the Phase 1 segmentation work
already assigns GrainSpace plates with that hash, and the symptom head shares a
backbone lineage with it. One global plate->split map across everything derived
from GrainSpace is the only way a plate cannot be train in one phase and test in
another.

A consequence worth stating plainly: the plates are class-pure -- none of the
362 spans more than one condition -- so plate-level nuisance factors correlate
perfectly with the label inside any split. Plate-level splitting is not a nicety
here, it is the only thing that makes the test number mean anything.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.annotation.splits import assign_split, plate_id  # noqa: E402

SOURCE_ROOT = os.path.join("datasets", "grainspace_maize_m600", "val")
OUT_PATH = os.path.join("data_processed", "manifest_symptom.csv")
UNIFIED_PATH = os.path.join("data_processed", "manifest_unified.csv")
COMBINED_PATH = os.path.join("data_processed", "manifest_tritask.csv")
COMBINED_FIELDS = ["filepath", "variety_label", "quality_label", "symptom_label",
                   "split", "source", "group"]

# The seven GrainSpace condition directories, kept under their source codes.
# Renaming them to something friendlier is where overclaiming starts: "FM" is
# the grader category "fusarium & mildew", and calling the class `fusarium`
# would quietly promote a visual grading category into a pathogen
# identification. The codes stay; docs/serving carry the descriptions.
SYMPTOM_CLASSES = ["AP", "BN", "FM", "HD", "MY", "NOR", "SD"]

# Descriptions of what the grader saw, for display. Deliberately phrased as
# appearance, not aetiology.
SYMPTOM_DESCRIPTIONS = {
    "AP": "Pest / insect damage (bore holes, galleries)",
    "BN": "Broken kernel (fracture faces exposed)",
    "FM": "Fusarium- or mildew-type discolouration",
    "HD": "Heat damage (whole-kernel discolouration)",
    "MY": "Mouldy appearance",
    "NOR": "No visible symptom",
    "SD": "Sprouted (germination visible)",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_rows(source_root: str = SOURCE_ROOT) -> list[dict]:
    rows: list[dict] = []
    for label in SYMPTOM_CLASSES:
        class_dir = os.path.join(source_root, label)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"expected GrainSpace condition directory {class_dir!r}. The manifest "
                f"is built only from directories that already carry a grader label."
            )
        for name in sorted(os.listdir(class_dir)):
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            path = os.path.join(class_dir, name)
            group = plate_id(name)
            rows.append({
                "filepath": path,
                "symptom_label": label,
                "split": assign_split(group),
                "source": "grainspace_m600_val",
                "group": group,
            })
    return rows


def summarise(rows: list[dict]) -> dict:
    by_split = Counter(r["split"] for r in rows)
    matrix: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        matrix[r["split"]][r["symptom_label"]] += 1
    plates_per_split: dict[str, set] = defaultdict(set)
    plate_classes: dict[str, set] = defaultdict(set)
    for r in rows:
        plates_per_split[r["split"]].add(r["group"])
        plate_classes[r["group"]].add(r["symptom_label"])
    straddling = [p for p, cs in plate_classes.items() if len(cs) > 1]
    return {
        "n": len(rows),
        "by_split": dict(by_split),
        "matrix": {s: dict(c) for s, c in matrix.items()},
        "plates": len(plate_classes),
        "plates_per_split": {s: len(p) for s, p in plates_per_split.items()},
        "plates_spanning_multiple_classes": len(straddling),
    }


def write_manifest(rows: list[dict], out_path: str = OUT_PATH) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filepath", "symptom_label", "split", "source", "group"])
        w.writeheader()
        w.writerows(rows)


def write_combined(symptom_rows: list[dict], unified_path: str = UNIFIED_PATH,
                   out_path: str = COMBINED_PATH) -> dict:
    """Union of the unified manifest and the symptom manifest, one row per image.

    Cells stay EMPTY where a label was never collected -- a GrainSpace crop gets
    no variety, a Mendeley kernel gets no visible-condition grade -- and the
    dataset turns empty into the ignore index so no head is ever trained on a
    label nobody assigned. Each side keeps the split it was already assigned by
    its own group-aware scheme; the two schemes never see the same file, so
    there is nothing to reconcile.
    """
    with open(unified_path, newline="", encoding="utf-8") as f:
        unified = list(csv.DictReader(f))
    seen = {r["filepath"] for r in unified}
    out: list[dict] = []
    for r in unified:
        out.append({
            "filepath": r["filepath"], "variety_label": r.get("variety_label", ""),
            "quality_label": r.get("quality_label", ""), "symptom_label": "",
            "split": r["split"], "source": r.get("source", ""), "group": r.get("group", ""),
        })
    added = 0
    for r in symptom_rows:
        if r["filepath"] in seen:  # would give one image two split assignments
            continue
        out.append({
            "filepath": r["filepath"], "variety_label": "", "quality_label": "",
            "symptom_label": r["symptom_label"], "split": r["split"],
            "source": r["source"], "group": r["group"],
        })
        added += 1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COMBINED_FIELDS)
        w.writeheader()
        w.writerows(out)
    return {"n": len(out), "from_unified": len(unified), "from_symptom": added,
            "by_split": dict(Counter(r["split"] for r in out))}


def main() -> None:
    rows = collect_rows()
    write_manifest(rows)
    s = summarise(rows)
    print(f"wrote {OUT_PATH}: {s['n']} rows, {s['plates']} plates")
    print(f"plates spanning >1 class: {s['plates_spanning_multiple_classes']}")
    header = "split".ljust(7) + "".join(c.rjust(6) for c in SYMPTOM_CLASSES) + "total".rjust(8)
    print(header)
    for split in ("train", "val", "test"):
        counts = s["matrix"].get(split, {})
        line = split.ljust(7) + "".join(str(counts.get(c, 0)).rjust(6) for c in SYMPTOM_CLASSES)
        print(line + str(s["by_split"].get(split, 0)).rjust(8))
    print("plates per split: " + ", ".join(
        f"{k} {v}" for k, v in sorted(s["plates_per_split"].items())))

    c = write_combined(rows)
    print(f"\nwrote {COMBINED_PATH}: {c['n']} rows "
          f"({c['from_unified']} unified + {c['from_symptom']} symptom), {c['by_split']}")


if __name__ == "__main__":
    main()
