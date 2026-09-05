import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Sprout,
  Images,
  Layers3,
  Gauge,
  ScanSearch,
  Leaf,
  Activity,
  Brain,
  Sparkles,
  History as HistoryIcon,
} from "lucide-react";
import { api } from "../api/client";
import { isVarietyRow } from "../lib/seedHealth";
import {
  Badge,
  Button,
  GlassCard,
  MetricCard,
  Page,
  SectionHeader,
  Skeleton,
  StatusPill,
} from "../components/ui";
import { DonutChart } from "../components/Charts";
import "./dashboard.css";

const pretty = (s) => (s ? s.replace(/_/g, " ") : "—");

function timeAgo(iso) {
  if (!iso) return "—";
  const mins = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min${mins === 1 ? "" : "s"} ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs} hour${hrs === 1 ? "" : "s"} ago`;
  const days = Math.round(hrs / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

const QUICK_ACTIONS = [
  { to: "/analyze", label: "Analyze seeds", icon: ScanSearch },
  { to: "/batch", label: "Batch analysis", icon: Layers3 },
  { to: "/copilot", label: "Ask Copilot", icon: Sparkles },
  { to: "/history", label: "View history", icon: HistoryIcon },
];

const MODULES = [
  {
    icon: ScanSearch,
    tone: "gold",
    title: "Detect",
    body: "Locate and count individual maize seeds with a trained YOLO detector.",
    tag: "YOLOv8n",
    metric: { model: "detection", label: "mAP@50" },
  },
  {
    icon: Leaf,
    tone: "green",
    title: "Recognize",
    body: "Identify supported maize varieties from a contrastive-pretrained CNN with cognitive attention.",
    tag: "EfficientNet-B0 + attention",
  },
  {
    icon: Activity,
    tone: "warn",
    title: "Analyze",
    body: "Grade kernel quality against expert-assigned Good/Bad labels, with extrapolated grades marked unverified.",
    tag: "EfficientNet-B0",
    metric: { model: "unified_seed_model", label: "Quality accuracy" },
  },
  {
    icon: Brain,
    tone: "cyan",
    title: "Explain",
    body: "Understand predictions through Grad-CAM attention maps and a Gemini copilot grounded in verified results.",
    tag: "Grad-CAM + Gemini",
  },
];

// A card's headline number is read from the model's own evaluation file via
// /api/system-info, never typed in here — the "97.3% test accuracy" this replaced
// had drifted from the 97.2% the checkpoint actually scored.
function moduleTag(mod, models) {
  if (!mod.metric) return mod.tag;
  const found = models.find((m) => m.key === mod.metric.model);
  const metric = found?.metrics?.find((x) => x.label === mod.metric.label);
  return metric ? `${mod.tag} · ${mod.metric.label} ${metric.value}` : mod.tag;
}

const MODEL_PILL = { ready: "Ready", available: "Not served", missing: "Missing" };
// short strip labels; the API's full model names are too long for a pill
const PILL_LABEL = {
  detection: "Detection model",
  unified_seed_model: "Variety + quality",
  quality_gate: "Distribution gate",
};

export default function Dashboard() {
  const navigate = useNavigate();
  const [stats, setStats] = useState(null);
  const [scale, setScale] = useState(null);
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [recent, setRecent] = useState([]);
  const [recentState, setRecentState] = useState("loading");

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([api.stats(), api.systemInfo(), api.trainingScale()]).then(([s, i, t]) => {
      if (cancelled) return;
      if (s.status === "fulfilled") setStats(s.value);
      if (i.status === "fulfilled") setInfo(i.value);
      if (t.status === "fulfilled") setScale(t.value);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    api
      .history(5, 0)
      .then((d) => {
        if (cancelled) return;
        setRecent(d.analyses || []);
        setRecentState("ready");
      })
      .catch(() => !cancelled && setRecentState("error"));
    return () => {
      cancelled = true;
    };
  }, []);

  const modelStatus = Object.fromEntries((info?.models || []).map((m) => [m.key, m]));
  const hasData = stats?.has_data;
  const fmt = (n) => (n == null ? "—" : n.toLocaleString());

  return (
    <Page className="dash">
      {/* ---------------- hero ---------------- */}
      {/* The real field video is the page's own background now (see Layout.jsx /
         bg-field--video) — it plays fixed and full-bleed behind the whole shell,
         not just this section. The copy card floats over it as translucent glass,
         letting the field show through rather than blocking it. */}
      <section className="hero">
        <GlassCard tier="primary" className="hero__copy">
          <motion.h1
            className="hero__title"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          >
            <span>MAIZE</span>
            <span className="hero__title-accent">INTELLIGENCE</span>
          </motion.h1>

          <motion.p
            className="hero__lede"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.08, ease: [0.22, 1, 0.36, 1] }}
          >
            AI-powered maize seed analysis. Detect, recognize, analyze and explain — with every
            prediction traceable to the model and data that produced it.
          </motion.p>

          <motion.div
            className="hero__actions"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.16, ease: [0.22, 1, 0.36, 1] }}
          >
            <Button size="lg" icon={ScanSearch} onClick={() => navigate("/analyze")}>
              Analyze seeds
            </Button>
            <Button size="lg" variant="secondary" onClick={() => navigate("/batch")}>
              Batch analysis
            </Button>
          </motion.div>
        </GlassCard>
      </section>

      {/* ---------------- live system strip ---------------- */}
      {/* "Ready" used to be printed here unconditionally, so the strip claimed a full
         stack even on a machine with no checkpoints at all. Each pill now reflects
         what /api/system-info found on disk. */}
      <section className="strip" aria-label="System status">
        <StatusPill
          label="Vision engine"
          value={info ? "Online" : loading ? "Checking" : "Unreachable"}
          tone={info ? "ok" : loading ? "idle" : "warn"}
          pulse={!!info}
        />
        {["detection", "unified_seed_model", "quality_gate"].map((key) => {
          const m = modelStatus[key];
          return (
            <StatusPill
              key={key}
              label={PILL_LABEL[key]}
              value={m ? MODEL_PILL[m.status] || m.status : loading ? "Checking" : "Unknown"}
              tone={m?.status === "ready" ? "ok" : m ? "warn" : "idle"}
            />
          );
        })}
        <StatusPill
          label="Gemini"
          value={info?.gemini?.configured ? "Connected" : "Not configured"}
          tone={info?.gemini?.configured ? "ai" : "warn"}
          pulse={!!info?.gemini?.configured}
        />
      </section>

      {/* ---------------- model scale ---------------- */}
      {/* Headline figures describe the MODELS, not app usage. Every value is read
         from a manifest or metrics file by /api/training-scale, so a retrain moves
         them and none can drift into a stale claim. */}
      <section className="dash__metrics">
        <MetricCard
          icon={Images}
          tone="gold"
          label="Training images"
          value={fmt(scale?.training_images)}
          sub="across 3 labelled datasets"
          loading={loading}
        />
        <MetricCard
          icon={ScanSearch}
          tone="green"
          label="Annotated seed boxes"
          value={fmt(scale?.annotated_bounding_boxes)}
          sub={`${fmt(scale?.detection_images)} detection images`}
          loading={loading}
        />
        <MetricCard
          icon={Gauge}
          tone="cyan"
          label="Detection mAP@50"
          value={scale?.detection_map50 != null ? `${(scale.detection_map50 * 100).toFixed(1)}%` : "—"}
          sub="held-out test split"
          loading={loading}
        />
        <MetricCard
          icon={Activity}
          tone="gold"
          label="Quality accuracy"
          value={scale?.quality_accuracy != null ? `${(scale.quality_accuracy * 100).toFixed(1)}%` : "—"}
          sub="expert-labelled test split"
          loading={loading}
        />
      </section>

      {/* ---------------- usage ---------------- */}
      <section className="dash__usage">
        <SectionHeader
          title="This installation"
          subtitle="What has been run through the platform here — separate from the training figures above."
          level={3}
        />
        <div className="dash__metrics">
          <MetricCard
            icon={Sprout}
            tone="gold"
            label="Seeds analyzed"
            value={hasData ? fmt(stats.seeds_analyzed) : "—"}
            sub={hasData ? undefined : "No analyses yet"}
            loading={loading}
          />
          <MetricCard
            icon={Images}
            tone="green"
            label="Images processed"
            value={hasData ? fmt(stats.images_processed) : "—"}
            sub={hasData ? undefined : "No analyses yet"}
            loading={loading}
          />
          <MetricCard
            icon={Layers3}
            tone="cyan"
            label="Varieties seen"
            value={hasData ? fmt(stats.varieties_recognized) : "—"}
            sub={hasData ? `of ${scale?.variety_classes ?? 6} supported` : "No analyses yet"}
            loading={loading}
          />
          <MetricCard
            icon={Gauge}
            tone="gold"
            label="Avg. confidence"
            value={
              stats?.average_confidence != null ? `${(stats.average_confidence * 100).toFixed(1)}%` : "—"
            }
            sub={stats?.average_confidence != null ? "variety predictions" : "No analyses yet"}
            loading={loading}
          />
        </div>
      </section>

      {/* ---------------- modules ---------------- */}
      <section className="dash__modules">
        <SectionHeader
          title="Intelligence modules"
          subtitle="What this system genuinely does — and, where a capability is limited, exactly how."
        />
        <div className="modgrid">
          {MODULES.map((m, i) => (
            <motion.div
              key={m.title}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.05 * i, ease: [0.22, 1, 0.36, 1] }}
            >
              <GlassCard hover className="modcard">
                <div className={`modcard__icon modcard__icon--${m.tone}`}>
                  <m.icon size={19} />
                </div>
                <h3 className="modcard__title">{m.title}</h3>
                <p className="modcard__body">{m.body}</p>
                <span className={`modcard__tag modcard__tag--${m.tone}`}>
                  {moduleTag(m, info?.models || [])}
                </span>
              </GlassCard>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ---------------- variety distribution (real data only) ---------------- */}
      {hasData && Object.keys(stats.variety_distribution || {}).length > 0 && (
        <section className="dash__dist">
          <SectionHeader
            title="Variety distribution"
            subtitle="Across every variety prediction stored in the database."
            right={
              <Link to="/history" className="dash__link">
                Browse history <ArrowRight size={14} />
              </Link>
            }
          />
          <GlassCard>
            <DonutChart data={stats.variety_distribution} />
          </GlassCard>
        </section>
      )}

      {/* ---------------- recent analyses + quick actions ---------------- */}
      <section className="dash__bottom">
        <GlassCard>
          <SectionHeader
            title="Recent analyses"
            subtitle="The latest results saved to history."
            level={3}
            right={
              <Link to="/history" className="dash__link">
                View all <ArrowRight size={14} />
              </Link>
            }
          />

          {recentState === "loading" && (
            <div className="dash__recentlist">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} height="42px" radius="var(--r-md)" />
              ))}
            </div>
          )}

          {recentState === "error" && <p className="muted">Recent analyses are unavailable right now.</p>}

          {recentState === "ready" && recent.length === 0 && (
            <p className="muted">No analyses yet — run one from Analyze and it will show up here.</p>
          )}

          {recentState === "ready" && recent.length > 0 && (
            <div className="dash__recentlist">
              {recent.map((r) => {
                const top = (r.classifications || []).find(isVarietyRow);
                return (
                  <Link to="/history" key={r.analysis_id} className="dash__recentrow">
                    <span className="dash__recentname" title={r.image_filename}>
                      {r.image_filename || r.analysis_id.slice(0, 8)}
                    </span>
                    <span className="dash__recentmeta mono">
                      {r.seed_count} seed{r.seed_count === 1 ? "" : "s"}
                    </span>
                    <span className="dash__recentvariety">{top ? pretty(top.predicted_class) : "—"}</span>
                    <Badge tone={r.status === "completed" ? "ok" : "warn"}>{r.status}</Badge>
                    <span className="dash__recenttime mono faint">{timeAgo(r.created_at)}</span>
                  </Link>
                );
              })}
            </div>
          )}
        </GlassCard>

        <GlassCard>
          <SectionHeader title="Quick actions" level={3} />
          <div className="dash__quickgrid">
            {QUICK_ACTIONS.map(({ to, label, icon: Icon }) => (
              <Link key={to} to={to} className="dash__quick">
                <Icon size={18} aria-hidden="true" />
                <span>{label}</span>
              </Link>
            ))}
          </div>
        </GlassCard>
      </section>
    </Page>
  );
}
