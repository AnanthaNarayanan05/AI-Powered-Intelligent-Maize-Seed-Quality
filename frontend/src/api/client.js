const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function handle(res) {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      detail = body.detail || body.error || detail;
    } catch (_) {
      // response wasn't JSON — keep default message, never surface raw stack traces
    }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

/** URL for a server-side image path returned by history or similarity results.
 *  The backend allow-lists which roots may be served and rejects anything else. */
export function mediaUrl(path, size = 0) {
  if (!path) return null;
  const q = new URLSearchParams({ path });
  if (size) q.set("size", String(size));
  return `${BASE_URL}/api/media/image?${q.toString()}`;
}

export const api = {
  health: () => fetch(`${BASE_URL}/api/health`).then(handle),
  systemInfo: () => fetch(`${BASE_URL}/api/system-info`).then(handle),
  stats: () => fetch(`${BASE_URL}/api/stats`).then(handle),
  trainingScale: () => fetch(`${BASE_URL}/api/training-scale`).then(handle),

  analyzeImage: (file, varietyDataset = "a") => {
    const form = new FormData();
    form.append("file", file);
    form.append("variety_dataset", varietyDataset);
    return fetch(`${BASE_URL}/api/analyze/image`, { method: "POST", body: form }).then(handle);
  },

  analyzeBatch: (files, varietyDataset = "a") => {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    form.append("variety_dataset", varietyDataset);
    return fetch(`${BASE_URL}/api/analyze/batch`, { method: "POST", body: form }).then(handle);
  },

  detect: (file) => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BASE_URL}/api/detect`, { method: "POST", body: form }).then(handle);
  },

  similarity: (file, varietyDataset = "a", topK = 5) => {
    const form = new FormData();
    form.append("file", file);
    form.append("variety_dataset", varietyDataset);
    form.append("top_k", topK);
    return fetch(`${BASE_URL}/api/similarity`, { method: "POST", body: form }).then(handle);
  },

  // Returns { url, model, head, targetLayer, predictedClass, region, note } rather
  // than a bare URL: the caller has to be able to say WHICH prediction the heatmap
  // explains, and the backend is the only thing that knows. `bbox` is the selected
  // seed's box in original-image pixels, exactly as /api/analyze returned it.
  gradcam: async (file, varietyDataset = "unified", { head = "variety", bbox = null } = {}) => {
    const form = new FormData();
    form.append("file", file);
    form.append("variety_dataset", varietyDataset);
    form.append("head", head);
    if (bbox) form.append("bbox", bbox.join(","));
    const res = await fetch(`${BASE_URL}/api/explain/gradcam`, { method: "POST", body: form });
    if (!res.ok) {
      let detail = `Request failed (${res.status})`;
      try {
        const body = await res.json();
        detail = body.detail || body.error || detail;
      } catch (_) {
        // non-JSON error body — keep the default message
      }
      throw new Error(detail);
    }
    return {
      url: URL.createObjectURL(await res.blob()),
      model: res.headers.get("X-Gradcam-Model"),
      head: res.headers.get("X-Gradcam-Head"),
      targetLayer: res.headers.get("X-Gradcam-Target-Layer"),
      predictedClass: res.headers.get("X-Gradcam-Predicted-Class"),
      region: res.headers.get("X-Gradcam-Region"),
      note: res.headers.get("X-Explainability-Note"),
    };
  },

  history: (limit = 20, offset = 0) =>
    fetch(`${BASE_URL}/api/history?limit=${limit}&offset=${offset}`).then(handle),

  historyDetail: (analysisId) => fetch(`${BASE_URL}/api/history/${analysisId}`).then(handle),

  batchDetail: (batchId) => fetch(`${BASE_URL}/api/history/batch/${batchId}`).then(handle),

  copilotChat: (analysisId, question) =>
    fetch(`${BASE_URL}/api/copilot/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ analysis_id: analysisId, question }),
    }).then(handle),

  explainAnalysis: (analysisId) =>
    fetch(`${BASE_URL}/api/copilot/explain-analysis`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ analysis_id: analysisId }),
    }).then(handle),

  summarizeBatch: (batchId) =>
    fetch(`${BASE_URL}/api/copilot/summarize-batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ batch_id: batchId }),
    }).then(handle),

  lotReport: (analysisIds, declaredVariety = null, lotReference = null) =>
    fetch(`${BASE_URL}/api/lot/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        analysis_ids: analysisIds,
        declared_variety: declaredVariety,
        lot_reference: lotReference,
      }),
    }).then(handle),

  compareAnalyses: (analysisIds) =>
    fetch(`${BASE_URL}/api/copilot/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ analysis_ids: analysisIds }),
    }).then(handle),
};
