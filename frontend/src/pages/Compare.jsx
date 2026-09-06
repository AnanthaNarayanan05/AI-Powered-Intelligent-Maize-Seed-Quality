import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { GitCompare, Sparkles, Sprout } from "lucide-react";
import { api, mediaUrl } from "../api/client";
import { UNAVAILABLE_LABEL, readCopilot } from "../lib/copilotResponse";
import { isQualityRow, isVarietyRow } from "../lib/seedHealth";
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
import "./compare.css";

const pretty = (s) => (s ? s.replace(/_/g, " ") : "—");
const topVariety = (a) => (a?.classifications || []).find(isVarietyRow);
const topDefect = (a) => (a?.classifications || []).find(isQualityRow);

function Side({ analysis, label }) {
  const v = topVariety(analysis);
  const d = topDefect(analysis);
  return (
    <div className="cmp__side">
      <span className="cmp__sidelabel">{label}</span>
      <div className="cmp__img">
        {analysis?.image_path ? (
          <img src={mediaUrl(analysis.image_path, 420)} alt={analysis.image_filename || "Analysis"} />
        ) : (
          <span className="cmp__noimg">No preview stored</span>
        )}
      </div>
      <span className="cmp__file" title={analysis?.image_filename}>
        {analysis?.image_filename || analysis?.analysis_id?.slice(0, 8) || "—"}
      </span>

      <div className="cmp__block">
        <div className="cmp__blockhead">
          <span>Seeds detected</span>
        </div>
        <strong className="cmp__big mono">{analysis?.seed_count ?? "—"}</strong>
      </div>

      <div className="cmp__block">
        <div className="cmp__blockhead">
          <span>Variety</span>
          {v && <ProvenanceBadge synthetic={false} />}
        </div>
        {v ? (
          <>
            <strong className="cmp__val">{pretty(v.predicted_class)}</strong>
            <ConfidenceBar
              value={v.confidence}
              calibrated={v.confidence_calibrated}
              showValue
              label={null}
            />
          </>
        ) : (
          <span className="faint">No variety prediction</span>
        )}
      </div>

      <div className="cmp__block">
        <div className="cmp__blockhead">
          <span>Kernel quality</span>
          {d && <ProvenanceBadge synthetic={false} />}
        </div>
        {d ? (
          <>
            <strong
              className={`cmp__val ${/^(bad|defect)/i.test(d.predicted_class) ? "cmp__val--warn" : ""}`}
            >
              {pretty(d.predicted_class)}
            </strong>
            <ConfidenceBar
              value={d.confidence}
              calibrated={d.confidence_calibrated}
              showValue
              label={null}
            />
          </>
        ) : (
          <span className="faint">No quality grade</span>
        )}
      </div>
    </div>
  );
}

export default function Compare() {
  const [analyses, setAnalyses] = useState([]);
  const [state, setState] = useState("loading");
  const [error, setError] = useState(null);
  const [idA, setIdA] = useState(null);
  const [idB, setIdB] = useState(null);
  const [ai, setAi] = useState({ state: "idle", text: null });

  useEffect(() => {
    let cancelled = false;
    api
      .history(30, 0)
      .then((d) => {
        if (cancelled) return;
        const list = d.analyses || [];
        setAnalyses(list);
        setState("ready");
        if (list[0]) setIdA(list[0].analysis_id);
        if (list[1]) setIdB(list[1].analysis_id);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e.message);
        setState("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const a = analyses.find((x) => x.analysis_id === idA) || null;
  const b = analyses.find((x) => x.analysis_id === idB) || null;

  // Differences are computed, never asserted — each row states both values.
  const diffs = useMemo(() => {
    if (!a || !b) return [];
    const va = topVariety(a);
    const vb = topVariety(b);
    const da = topDefect(a);
    const db = topDefect(b);
    return [
      {
        label: "Seed count",
        left: a.seed_count,
        right: b.seed_count,
        same: a.seed_count === b.seed_count,
      },
      {
        label: "Variety",
        left: pretty(va?.predicted_class),
        right: pretty(vb?.predicted_class),
        same: va?.predicted_class === vb?.predicted_class,
      },
      {
        label: "Variety confidence",
        left: va ? `${(va.confidence * 100).toFixed(1)}%` : "—",
        right: vb ? `${(vb.confidence * 100).toFixed(1)}%` : "—",
        same:
          va && vb ? Math.abs(va.confidence - vb.confidence) < 0.01 : va === vb,
      },
      {
        label: "Kernel quality",
        left: pretty(da?.predicted_class),
        right: pretty(db?.predicted_class),
        same: da?.predicted_class === db?.predicted_class,
      },
      {
        label: "Dataset",
        left: a.variety_dataset_used?.toUpperCase() || "—",
        right: b.variety_dataset_used?.toUpperCase() || "—",
        same: a.variety_dataset_used === b.variety_dataset_used,
      },
    ];
  }, [a, b]);

  const askAi = async () => {
    if (!idA || !idB) return;
    setAi({ state: "loading", text: null });
    try {
      const { text, available } = readCopilot(await api.compareAnalyses([idA, idB]));
      setAi({ state: available ? "ready" : "error", text });
    } catch (e) {
      setAi({ state: "error", text: e.message });
    }
  };

  return (
    <Page className="cmp">
      <SectionHeader
        title="Compare analyses"
        subtitle="Place two stored analyses side by side. Differences are highlighted; both values are always shown."
      />

      {state === "loading" && (
        <GlassCard>
          <Skeleton height="200px" />
        </GlassCard>
      )}

      {state === "error" && (
        <GlassCard accent="danger">
          <ErrorState title="Could not load analyses" reason={error} />
        </GlassCard>
      )}

      {state === "ready" && analyses.length < 2 && (
        <GlassCard>
          <EmptyState
            icon={Sprout}
            title="At least two analyses are needed"
            message="Run a couple of analyses from the Analyze page, then return here to compare them."
          />
        </GlassCard>
      )}

      {state === "ready" && analyses.length >= 2 && (
        <>
          <GlassCard tier="floating" className="cmp__pickers">
            <label className="cmp__picker">
              <span>Analysis A</span>
              <select value={idA || ""} onChange={(e) => setIdA(e.target.value)}>
                {analyses.map((x) => (
                  <option key={x.analysis_id} value={x.analysis_id}>
                    {x.image_filename || x.analysis_id.slice(0, 8)}
                  </option>
                ))}
              </select>
            </label>
            <span className="cmp__vs">
              <GitCompare size={17} />
            </span>
            <label className="cmp__picker">
              <span>Analysis B</span>
              <select value={idB || ""} onChange={(e) => setIdB(e.target.value)}>
                {analyses.map((x) => (
                  <option key={x.analysis_id} value={x.analysis_id}>
                    {x.image_filename || x.analysis_id.slice(0, 8)}
                  </option>
                ))}
              </select>
            </label>
          </GlassCard>

          <motion.div
            className="cmp__grid"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
          >
            <GlassCard>
              <Side analysis={a} label="A" />
            </GlassCard>
            <GlassCard>
              <Side analysis={b} label="B" />
            </GlassCard>
          </motion.div>

          <GlassCard>
            <SectionHeader title="Differences" level={3} />
            <div className="cmp__diffs">
              {diffs.map((d) => (
                <div className={`cmp__diff ${d.same ? "" : "is-diff"}`} key={d.label}>
                  <span className="cmp__difflabel">{d.label}</span>
                  <span className="cmp__diffval mono">{d.left}</span>
                  <Badge tone={d.same ? "neutral" : "gold"}>{d.same ? "same" : "differs"}</Badge>
                  <span className="cmp__diffval mono">{d.right}</span>
                </div>
              ))}
            </div>
          </GlassCard>

          <GlassCard accent="cyan" tier="primary">
            <SectionHeader
              title="AI comparison"
              subtitle="Gemini interprets the two verified results above."
              level={3}
              right={
                <Button
                  variant="ai"
                  size="sm"
                  icon={Sparkles}
                  onClick={askAi}
                  loading={ai.state === "loading"}
                >
                  Compare with AI
                </Button>
              }
            />
            {ai.state === "idle" && (
              <p className="muted cmp__idle">Ask the copilot to describe how these two analyses differ.</p>
            )}
            {ai.state === "loading" && <Skeleton height="48px" />}
            {(ai.state === "ready" || ai.state === "error") && (
              <div className={`cmp__ai ${ai.state === "error" ? "is-error" : ""}`}>
                <span className="cmp__ailabel">
                  {ai.state === "error"
                    ? UNAVAILABLE_LABEL
                    : "AI-generated comparison based on model analysis"}
                </span>
                <p>{ai.text}</p>
              </div>
            )}
          </GlassCard>
        </>
      )}
    </Page>
  );
}
