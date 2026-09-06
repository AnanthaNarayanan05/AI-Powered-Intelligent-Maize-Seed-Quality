"""Seed-body mask proposals for single-kernel crops.

The body mask is not a decoration: defect coverage % is defect pixels divided by
VALID SEED pixels, so an error in the body mask is an error in the reported
coverage number. That is why this lives in its own module and is proposed by SAM
rather than by a colour threshold -- a threshold that happens to include two
pixels of the white background inflates the denominator and silently deflates
every coverage figure computed from it.

GrainSpace crops are cut from a multi-kernel plate, so neighbouring kernels
intrude at the crop edges. "All maize-coloured pixels" is therefore the WRONG
body: it would merge the centre kernel with its neighbours. Everything here is
built to isolate the CENTRE kernel specifically.

Nothing in this module produces ground truth. It produces proposals that a human
reviewer accepts, corrects, or rejects.
"""
from __future__ import annotations

import cv2
import numpy as np

# Below this the mask is not a kernel -- it is a fragment of one, or the whole
# frame. Measured against the 1,260 GrainSpace val crops, real bodies sit at
# 0.25-0.80 of the frame; the guard bands are deliberately wider than that so a
# genuinely unusual crop is flagged rather than silently discarded.
BODY_FRAC_MIN = 0.08
BODY_FRAC_MAX = 0.95

# SAM's own candidate filter, applied before the centre-containment test.
CAND_FRAC_MIN = 0.10
CAND_FRAC_MAX = 0.92


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes by refilling external contours.

    Deliberately NOT floodFill-from-(0,0): when the kernel touches the top-left
    corner -- which it does in a crop cut tight to a plate -- the seed pixel is
    already foreground, the fill is a no-op, and inverting the result turns the
    ENTIRE BACKGROUND into body. That failure is silent and it poisons every
    downstream coverage number, so the seed-free formulation is used instead.
    """
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def propose_body(image: np.ndarray, predictor) -> tuple[np.ndarray, bool]:
    """Propose the centre kernel's body mask. Returns (mask, plausible).

    `predictor` must be a segment_anything SamPredictor with set_image already
    called on this image -- embedding is the expensive half and the caller
    usually wants to reuse it for the defect prompts too.

    The `plausible` flag is advisory, not a filter: an implausible mask is still
    returned so the reviewer can see and fix it. Dropping it here would hide the
    hard cases, which are exactly the ones worth a human's attention.
    """
    h, w = image.shape[:2]
    # One positive at the centre (the kernel the crop is about) and four
    # negatives in the corners. The corners are where neighbouring kernels
    # intrude, so labelling them background is what stops SAM from returning
    # "all the maize in frame" instead of "this kernel".
    points = np.array([[w // 2, h // 2], [4, 4], [w - 5, 4], [4, h - 5], [w - 5, h - 5]])
    labels = np.array([1, 0, 0, 0, 0])
    masks, scores, _ = predictor.predict(
        point_coords=points, point_labels=labels, multimask_output=True
    )

    # Prefer the LARGEST candidate that still contains the centre. SAM's
    # multimask output is roughly part / subpart / whole; for a kernel the
    # subpart is usually the crown or the germ, and taking the highest-scoring
    # mask would systematically return a piece of the kernel as the whole body.
    best = None
    for mask in masks:
        if not mask[h // 2, w // 2]:
            continue
        frac = float(mask.mean())
        if not (CAND_FRAC_MIN < frac < CAND_FRAC_MAX):
            continue
        if best is None or frac > best[0]:
            best = (frac, mask)
    chosen = best[1] if best is not None else masks[int(np.argmax(scores))]

    m = chosen.astype(np.uint8)
    # OPEN first to shed the dithering speckle the source PNGs carry over
    # neighbouring kernels, CLOSE second to seal the seam along the pedicel.
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = lab[h // 2, w // 2]
    if keep == 0:
        # Morphology ate the centre pixel. Fall back to the largest component
        # rather than returning an empty mask, and let `plausible` carry the doubt.
        if n_lab <= 1:
            return m.astype(bool), False
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    body = fill_holes(lab == keep)
    return body, bool(BODY_FRAC_MIN < body.mean() < BODY_FRAC_MAX)
