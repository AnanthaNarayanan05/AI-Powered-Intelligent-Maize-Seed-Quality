import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Send, Sparkles, MessageSquare, Sprout, Workflow, CircleHelp } from "lucide-react";
import { api, mediaUrl } from "../api/client";
import { isQualityRow, isVarietyRow } from "../lib/seedHealth";
import { UNAVAILABLE_LABEL, readCopilot } from "../lib/copilotResponse";
import {
  Badge,
  Button,
  ConfidenceBar,
  EmptyState,
  ErrorState,
  GlassCard,
  Page,
  ProvenanceBadge,
  SectionHeader,
  Skeleton,
} from "../components/ui";
import "./copilot.css";

const pretty = (s) => (s ? s.replace(/_/g, " ") : "—");

const SUGGESTIONS = [
  "Explain this result",
  "Why was this variety predicted?",
  "What did the detector find?",
  "What are the model limitations?",
];

// The orchestrator's own question table (src/pipeline/orchestrator.py INTENTS),
// offered as suggestions so a reader who doesn't know the phrasing that routes to
// each capability can still reach it. Wording matches the canonical form the
// backend already returns in `intent.canonical`, not reworded here.
const PIPELINE_SUGGESTIONS = [
  "Find defective seeds",
  "Show me where the defect is",
  "Measure the defect",
  "How severe is the defect?",
  "Check for foreign objects",
  "What visible condition do these kernels show?",
];

const STAGE_TONE = { ran: "ok", reused: "neutral", cached: "neutral", skipped: "neutral", unavailable: "warn", failed: "error" };

/** The orchestrator's stage-by-stage account and its answer to one question, read
 * back exactly as src/pipeline/orchestrator.py.run() returned it. Nothing here is
 * reworded or summarised by a language model -- every field shown is a field the
 * response actually carries, and a field the response omits is simply not shown. */
function PipelineAnswer({ result }) {
  const a = result.answer || {};
  const unavailable = [...(result.unavailable || [])];
  return (
    <div className="co__panswer">
      <div className="co__pintent">
        <span>Resolved to:</span>
        <strong>{result.intent?.canonical}</strong>
        <Badge tone="neutral" title={result.intent?.matched ? `matched "${result.intent.matched}"` : undefined}>
          {result.intent?.resolved_by === "explicit" ? "explicit" :
           result.intent?.resolved_by === "pattern" ? "matched from your question" :
           result.intent?.resolved_by === "default" ? "no exact match — default analysis" :
           "no question given"}
        </Badge>
      </div>

      <div className="co__pstages">
        {(result.stages || []).map((s) => (
          <div className="co__pstage" key={s.stage}>
            <Badge tone={STAGE_TONE[s.status] || "neutral"}>{s.status}</Badge>
            <span>{s.describe}</span>
          </div>
        ))}
      </div>

      <div className="co__pans">
        <div className="co__panshead">
          <strong>{a.label || a.capability || "Answer"}</strong>
          {"available" in a && (
            <Badge tone={a.available ? "ok" : "warn"}>{a.available ? "available" : "unavailable"}</Badge>
          )}
        </div>
        {a.available === false && a.message && <p className="co__pmsg">{a.message}</p>}
        {a.what_it_means && <p>{a.what_it_means}</p>}
        {a.what_it_does_not_mean && <p className="co__pwarn">{a.what_it_does_not_mean}</p>}
        {Array.isArray(a.caveats) && a.caveats.length > 0 && (
          <ul className="co__pcaveats">
            {a.caveats.map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        )}
      </div>

      {unavailable.length > 0 && (
        <div className="co__punavail">
          <span className="co__punavaillabel">Not answered</span>
          {unavailable.map((u, i) => (
            <p key={i}><strong>{u.capability}:</strong> {u.message}</p>
          ))}
        </div>
      )}

      {(result.warnings || []).map((w, i) => (
        <p className="co__pwarn" key={i}>{w}</p>
      ))}
    </div>
  );
}

export default function Copilot() {
  const [analyses, setAnalyses] = useState([]);
  const [listState, setListState] = useState("loading");
  const [listError, setListError] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const scrollRef = useRef(null);

  // The pipeline's own Q&A (POST /api/analyze/ask), kept entirely separate from the
  // Gemini conversation above: no history, no follow-up context beyond the cached
  // analysis_id, and no language model in the loop.
  const [pInput, setPInput] = useState("");
  const [pState, setPState] = useState("idle"); // idle | loading | ready | error
  const [pResult, setPResult] = useState(null);
  const [pError, setPError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api
      .history(20, 0)
      .then((d) => {
        if (cancelled) return;
        const list = d.analyses || [];
        setAnalyses(list);
        setListState("ready");
        if (list.length) setSelectedId(list[0].analysis_id);
      })
      .catch((e) => {
        if (cancelled) return;
        setListError(e.message);
        setListState("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setMessages([]);
    setPState("idle");
    setPResult(null);
    setPError(null);
    setPInput("");
  }, [selectedId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  const selected = analyses.find((a) => a.analysis_id === selectedId) || null;
  const topVariety = (selected?.classifications || []).find(isVarietyRow);
  const defect = (selected?.classifications || []).find(isQualityRow);

  const send = async (text) => {
    const q = (text ?? input).trim();
    if (!q || !selectedId || sending) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setSending(true);
    try {
      const { text: answer, available } = readCopilot(await api.copilotChat(selectedId, q));
      setMessages((m) => [...m, { role: "ai", text: answer, error: !available }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "ai", text: e.message, error: true }]);
    } finally {
      setSending(false);
    }
  };

  const askPipeline = async (text) => {
    const q = (text ?? pInput).trim();
    if (!q || !selectedId || pState === "loading") return;
    setPInput("");
    setPState("loading");
    setPError(null);
    try {
      const result = await api.askOrchestrator(selectedId, { question: q });
      setPResult(result);
      setPState("ready");
    } catch (e) {
      setPError(e.message || "The pipeline could not answer this question.");
      setPState("error");
    }
  };

  return (
    <Page className="co">
      <SectionHeader
        title="AI seed analysis copilot"
        subtitle="Ask questions about verified model results. The copilot explains outputs the models already produced — it never generates predictions itself."
      />

      {/* ---------- the pipeline's own Q&A, separate from the Gemini chat below ---------- */}
      <GlassCard accent="gold" tier="primary" className="co__pipeline">
        <div className="co__pipehead">
          <span className="co__pipeorb" aria-hidden="true">
            <Workflow size={14} />
          </span>
          <div>
            <h3 className="co__chattitle">Ask the pipeline directly</h3>
            <span className="co__chatsub">
              No language model — routes to the exact stages that can answer it, and says so when they can't.
            </span>
          </div>
          <Badge tone="gold">Pipeline</Badge>
        </div>

        {!selectedId && (
          <p className="muted co__pidle">Select an analysis above to ask it a question.</p>
        )}

        {selectedId && (
          <>
            <div className="co__suggest">
              {PIPELINE_SUGGESTIONS.map((s) => (
                <button key={s} className="co__chip" onClick={() => askPipeline(s)} disabled={pState === "loading"}>
                  {s}
                </button>
              ))}
            </div>

            <form
              className="co__composer co__pcomposer"
              onSubmit={(e) => {
                e.preventDefault();
                askPipeline();
              }}
            >
              <input
                value={pInput}
                onChange={(e) => setPInput(e.target.value)}
                placeholder="e.g. how severe is the defect?"
                disabled={pState === "loading"}
                aria-label="Ask the pipeline a question"
              />
              <Button
                type="submit"
                variant="secondary"
                icon={CircleHelp}
                disabled={!pInput.trim() || pState === "loading"}
              >
                Ask
              </Button>
            </form>

            {pState === "loading" && (
              <div className="co__skel co__pskel">
                <Skeleton height="12px" />
                <Skeleton height="12px" width="80%" />
              </div>
            )}
            {pState === "error" && <ErrorState title="The pipeline could not answer" reason={pError} />}
            {pState === "ready" && pResult && <PipelineAnswer result={pResult} />}
          </>
        )}
      </GlassCard>

      <div className="co__layout">
        {/* ---------- context ---------- */}
        <div className="co__context">
          <GlassCard tier="primary">
            <h3 className="co__label">Analysis context</h3>

            {listState === "loading" && (
              <div className="co__skel">
                <Skeleton height="14px" />
                <Skeleton height="14px" width="70%" />
              </div>
            )}

            {listState === "error" && <ErrorState title="Could not load analyses" reason={listError} />}

            {listState === "ready" && analyses.length === 0 && (
              <EmptyState
                icon={Sprout}
                title="No analyses yet"
                message="Run an analysis first — the copilot answers questions about stored results."
              />
            )}

            {listState === "ready" && analyses.length > 0 && (
              <>
                <select
                  className="co__select"
                  value={selectedId || ""}
                  onChange={(e) => setSelectedId(e.target.value)}
                  aria-label="Select an analysis"
                >
                  {analyses.map((a) => (
                    <option key={a.analysis_id} value={a.analysis_id}>
                      {a.image_filename || a.analysis_id.slice(0, 8)} · {a.seed_count} seed
                      {a.seed_count === 1 ? "" : "s"}
                    </option>
                  ))}
                </select>

                {selected && (
                  <div className="co__ctx">
                    {selected.image_path && (
                      <div className="co__ctximg">
                        <img src={mediaUrl(selected.image_path, 300)} alt="Selected analysis" />
                      </div>
                    )}

                    <div className="co__ctxrow">
                      <span>Seeds detected</span>
                      <strong className="mono">{selected.seed_count}</strong>
                    </div>
                    <div className="co__ctxrow">
                      <span>Dataset</span>
                      <strong className="mono">{selected.variety_dataset_used?.toUpperCase()}</strong>
                    </div>

                    {topVariety && (
                      <div className="co__ctxblock">
                        <div className="co__ctxhead">
                          <span>Variety</span>
                          <ProvenanceBadge synthetic={false} />
                        </div>
                        <strong className="co__ctxval">{pretty(topVariety.predicted_class)}</strong>
                        <ConfidenceBar
                          value={topVariety.confidence}
                          calibrated={topVariety.confidence_calibrated}
                          showValue
                          label={null}
                        />
                      </div>
                    )}

                    {defect && (
                      <div className="co__ctxblock">
                        <div className="co__ctxhead">
                          <span>Defect pattern</span>
                          <ProvenanceBadge synthetic={false} />
                        </div>
                        <strong
                          className={`co__ctxval ${
                            /^(bad|defect)/i.test(defect.predicted_class) ? "co__ctxval--warn" : ""
                          }`}
                        >
                          {pretty(defect.predicted_class)}
                        </strong>
                        <ConfidenceBar
                          value={defect.confidence}
                          calibrated={defect.confidence_calibrated}
                          showValue
                          label={null}
                        />
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </GlassCard>
        </div>

        {/* ---------- conversation ---------- */}
        <GlassCard accent="cyan" tier="primary" className="co__chat">
          <div className="co__chathead">
            <span className="co__aiorb" aria-hidden="true">
              <Sparkles size={14} />
            </span>
            <div>
              <h3 className="co__chattitle">Copilot</h3>
              <span className="co__chatsub">Grounded in stored analysis results</span>
            </div>
            <Badge tone="ai">Gemini</Badge>
          </div>

          <div className="co__messages" ref={scrollRef}>
            {messages.length === 0 && !sending && (
              <div className="co__intro">
                <MessageSquare size={22} />
                <p>
                  {selectedId
                    ? "Ask anything about this analysis."
                    : "Select an analysis to begin."}
                </p>
              </div>
            )}

            {messages.map((m, i) => (
              <motion.div
                key={i}
                className={`msg msg--${m.role} ${m.error ? "is-error" : ""}`}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                {m.role === "ai" && (
                  <span className="msg__label">
                    {m.error ? UNAVAILABLE_LABEL : "AI-generated explanation based on model analysis"}
                  </span>
                )}
                <p>{m.text}</p>
              </motion.div>
            ))}

            {sending && (
              <div className="msg msg--ai">
                <span className="msg__label">Thinking…</span>
                <div className="co__skel">
                  <Skeleton height="11px" />
                  <Skeleton height="11px" width="82%" />
                </div>
              </div>
            )}
          </div>

          {selectedId && messages.length === 0 && (
            <div className="co__suggest">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="co__chip" onClick={() => send(s)} disabled={sending}>
                  {s}
                </button>
              ))}
            </div>
          )}

          <form
            className="co__composer"
            onSubmit={(e) => {
              e.preventDefault();
              send();
            }}
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={selectedId ? "Ask about this analysis…" : "Select an analysis first"}
              disabled={!selectedId || sending}
              aria-label="Your question"
            />
            <Button
              type="submit"
              variant="ai"
              icon={Send}
              disabled={!selectedId || !input.trim() || sending}
            >
              Send
            </Button>
          </form>
        </GlassCard>
      </div>
    </Page>
  );
}
