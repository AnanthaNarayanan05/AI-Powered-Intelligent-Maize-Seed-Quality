/**
 * Reads a CopilotResponse the way the backend actually shapes it.
 *
 * Every /api/copilot/* endpoint returns {text, ai_available, fallback_reason}. When
 * Gemini is unreachable — no API key configured, a timeout, a rate limit, an upstream
 * 5xx — `text` is null and `fallback_reason` says which. The pages used to read only
 * `text` and fall back to a hardcoded "No response returned.", which threw away the
 * one piece of information the user needed and, worse, kept rendering the bubble
 * under the label "AI-generated explanation based on model analysis". A server-side
 * outage read as though the model had answered and had nothing to say.
 *
 * `available` is what the caller should branch on for styling and labelling: an
 * unavailable answer is not an explanation, and must never be presented as one.
 * The ML results beside it are unaffected and stay on screen — Gemini explains
 * verified output, it never produces it.
 */
export const UNAVAILABLE_LABEL = "AI explanation temporarily unavailable";

export function readCopilot(response) {
  // The endpoints have never agreed on a key name; keep accepting all of them.
  const text = response?.text ?? response?.answer ?? response?.explanation
    ?? response?.summary ?? response?.comparison ?? null;

  if (text) return { text, available: true };

  return {
    text: response?.fallback_reason || "No response returned.",
    available: false,
  };
}
