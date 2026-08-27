import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Send, Sparkles, MessageSquare, Sprout } from "lucide-react";
import { api, mediaUrl } from "../api/client";
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

export default function Copilot() {
  const [analyses, setAnalyses] = useState([]);
  const [listState, setListState] = useState("loading");
  const [listError, setListError] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const scrollRef = useRef(null);

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
      const r = await api.copilotChat(selectedId, q);
      setMessages((m) => [
        ...m,
        { role: "ai", text: r.answer || r.explanation || r.text || "No response returned." },
      ]);
    } catch (e) {
      setMessages((m) => [...m, { role: "ai", text: e.message, error: true }]);
    } finally {
      setSending(false);
    }
  };

  return (
    <Page className="co">
      <SectionHeader
        title="AI seed analysis copilot"
        subtitle="Ask questions about verified model results. The copilot explains outputs the models already produced — it never generates predictions itself."
      />

      <div className="co__layout">
        {/* ---------- context ---------- */}
        <div className="co__context">
          <GlassCard>
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
                        <ConfidenceBar value={topVariety.confidence} showValue label={null} />
                      </div>
                    )}

                    {defect && (
                      <div className="co__ctxblock">
                        <div className="co__ctxhead">
                          <span>Defect pattern</span>
                          <ProvenanceBadge synthetic={false} />
                        </div>
                        <strong className="co__ctxval co__ctxval--warn">
                          {pretty(defect.predicted_class)}
                        </strong>
                        <ConfidenceBar value={defect.confidence} showValue label={null} />
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </GlassCard>
        </div>

        {/* ---------- conversation ---------- */}
        <GlassCard accent="cyan" className="co__chat">
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
                  <span className="msg__label">AI-generated explanation based on model analysis</span>
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
