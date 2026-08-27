import { lazy, Suspense, useEffect, useState } from "react";
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
} from "lucide-react";
import { api } from "../api/client";
import { Button, GlassCard, MetricCard, Page, SectionHeader, StatusPill } from "../components/ui";
import "./dashboard.css";

// The 3D scene is the heaviest thing in the bundle — keep it out of the initial chunk.
const SeedHero = lazy(() => import("../components/SeedHero"));

const MODULES = [
  {
    icon: ScanSearch,
    tone: "gold",
    title: "Detect",
    body: "Locate and count individual maize seeds with a trained YOLO detector.",
    tag: "YOLOv8 · mAP@50 98.2%",
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
    tag: "97.3% test accuracy",
  },
  {
    icon: Brain,
    tone: "cyan",
    title: "Explain",
    body: "Understand predictions through Grad-CAM attention maps and a Gemini copilot grounded in verified results.",
    tag: "Grad-CAM + Gemini",
  },
];

export default function Dashboard() {
  const navigate = useNavigate();
  const [stats, setStats] = useState(null);
  const [scale, setScale] = useState(null);
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);

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

  const hasData = stats?.has_data;
  const fmt = (n) => (n == null ? "—" : n.toLocaleString());

  return (
    <Page className="dash">
      {/* ---------------- hero ---------------- */}
      <section className="hero">
        <div className="hero__copy">
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
            <Button size="lg" variant="secondary" onClick={() => navigate("/history")}>
              View history
            </Button>
          </motion.div>
        </div>

        <div className="hero__visual">
          <Suspense fallback={<div className="seed-hero seed-hero--static" aria-hidden="true" />}>
            <SeedHero />
          </Suspense>
        </div>
      </section>

      {/* ---------------- live system strip ---------------- */}
      <section className="strip" aria-label="System status">
        <StatusPill label="Vision engine" value="Online" tone="ok" pulse />
        <StatusPill label="Variety model" value="Ready" tone="ok" />
        <StatusPill label="Detection model" value="Ready" tone="ok" />
        <StatusPill label="Quality model" value="Ready" tone="ok" />
        <StatusPill
          label="Gemini"
          value={info?.gemini_configured ? "Connected" : "Not configured"}
          tone={info?.gemini_configured ? "ai" : "warn"}
          pulse={!!info?.gemini_configured}
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
                <span className={`modcard__tag modcard__tag--${m.tone}`}>{m.tag}</span>
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
            <div className="distlist">
              {Object.entries(stats.variety_distribution).map(([name, n], i) => {
                const total = Object.values(stats.variety_distribution).reduce((a, b) => a + b, 0);
                return (
                  <div className="distrow" key={name}>
                    <span className="distrow__name">{name.replace(/_/g, " ")}</span>
                    <div className="distrow__bar">
                      <motion.span
                        className="distrow__fill"
                        initial={{ width: 0 }}
                        animate={{ width: `${(n / total) * 100}%` }}
                        transition={{ duration: 0.7, delay: 0.06 * i, ease: [0.22, 1, 0.36, 1] }}
                      />
                    </div>
                    <span className="distrow__n mono">{n}</span>
                  </div>
                );
              })}
            </div>
          </GlassCard>
        </section>
      )}
    </Page>
  );
}
