/* ============================================================
   Per-seed health flag used to mark defective kernels on the detection canvas.

   Two signals can drive the flag, and which one is used matters for how the
   result may be described:

     1. quality_prediction — a real good/bad kernel-quality classification.
        Present only once a quality model trained on genuine expert labels is
        loaded. This is the signal the flag is meant to express.

     2. detection_confidence — how sure YOLO is that the box contains a seed.
        This is NOT a health measurement. It is the fallback so the canvas still
        marks boxes worth a second look when no quality model is available, and
        `reason` says exactly that so the UI never calls it a defect.

   Anything the pipeline labels synthetic is deliberately ignored: procedural
   pattern classes are a demonstration of the attention architecture, not a
   statement about a real kernel, and must never drive a defect marker.
   ============================================================ */

export const HEALTH_THRESHOLD = 0.65;

// Class names a quality model may use for the defective side of the split.
const UNHEALTHY = /^(bad|defect|defective|damaged|unhealthy|poor|rotten|broken|discoloured|discolored)/i;

export function seedHealth(seed, threshold = HEALTH_THRESHOLD) {
  const q = seed?.quality_prediction;

  if (q && !q.is_synthetic_model && q.predicted_class) {
    const bad = UNHEALTHY.test(q.predicted_class);
    // Score is "probability this kernel is sound": the model's own confidence
    // when it says good, its complement when it says bad.
    const score = bad ? 1 - (q.confidence ?? 0) : q.confidence ?? 0;

    // A grade produced for a kernel unlike anything the quality head was validated
    // on is an extrapolation. It still gets marked, because a possible defect is
    // worth a human look, but it is marked as UNVERIFIED rather than as a defect --
    // the platform must not present a guess in the same visual language as a
    // measurement.
    if (q.out_of_distribution) {
      return {
        flagged: bad,
        score,
        source: "quality-extrapolated",
        verified: false,
        reason:
          `${q.predicted_class} (${((q.confidence ?? 0) * 100).toFixed(0)}%), but this ` +
          `kernel falls outside the images the quality model was validated on. ` +
          `Unverified extrapolation, not a measured defect.`,
      };
    }

    return {
      flagged: bad || score < threshold,
      score,
      source: "quality",
      verified: true,
      reason: bad
        ? `Classified ${q.predicted_class} (${((q.confidence ?? 0) * 100).toFixed(0)}% confidence)`
        : `Sound kernel, but only ${(score * 100).toFixed(0)}% confidence`,
    };
  }

  const det = seed?.detection_confidence ?? 1;
  return {
    flagged: det < threshold,
    score: det,
    source: "detection",
    verified: false,
    reason: `Low detection confidence (${(det * 100).toFixed(0)}%) — no quality model loaded, so this flags an uncertain detection, not a measured defect`,
  };
}

/* Selecting a stored classification by role.

   Rows cannot be split by the is_synthetic_model flag any more: the real quality
   head is correctly stored with is_synthetic_model = 0, so that flag no longer
   distinguishes "quality" from "variety". Match on model name instead, exactly as
   backend/routes/lot.py does, and treat legacy synthetic rows as neither. */
const QUALITY_MODEL = /(quality|defect)/i;

export const isQualityRow = (c) =>
  !!c && !c.is_synthetic_model && QUALITY_MODEL.test(c.model_name || "");

export const isVarietyRow = (c) =>
  !!c && !c.is_synthetic_model && !QUALITY_MODEL.test(c.model_name || "");

export const isLegacySyntheticRow = (c) => !!c && !!c.is_synthetic_model;
