import { useState } from "react";
import { motion } from "framer-motion";
import { Layers, Images, Sprout, Gauge, Trophy, RotateCcw, Sparkles } from "lucide-react";
import { api, mediaUrl } from "../api/client";
import { UNAVAILABLE_LABEL, readCopilot } from "../lib/copilotResponse";
import {
  Badge,
  Button,
  ConfidenceBar,
  EmptyState,
  ErrorState,
  GlassCard,
  MetricCard,
  Page,
  SectionHeader,
  Skeleton,
} from "../components/ui";
import UploadZone from "../components/UploadZone";
import { ConfidenceHistogram, DonutChart } from "../components/Charts";
import "./batch.css";

const pretty = (s) => (s ? s.replace(/_/g, " ") : "—");

export default function Batch() {
  const [files, setFiles] = useState([]);
  const [dataset] = useState("unified");
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [batch, setBatch] = useState(null);
  const [details, setDetails] = useState([]);
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState({ state: "idle", text: null });

  const reset = () => {
    setFiles([]);
    setStatus("idle");
    setBatch(null);
    setDetails([]);
    setError(null);
    setSummary({ state: "idle", text: null });
  };

  const run = async () => {
    if (!files.length) return;
    setStatus("running");
    setError(null);
    setDetails([]);
    try {
      const data = await api.analyzeBatch(files, dataset);
      setBatch(data);
      setStatus("done");

      // The batch response carries aggregate stats and ids only, so the per-image
      // grid is built by fetching each stored analysis.
      const settled = await Promise.allSettled(
        (data.analysis_ids || []).map((id) => api.historyDetail(id))
      );
      setDetails(settled.filter((s) => s.status === "fulfilled").map((s) => s.value));
    } catch (e) {
      setError(e.message || "The batch could not be processed.");
      setStatus("error");
    }
  };

  const askSummary = async () => {
    if (!batch?.batch_id) return;
    setSummary({ state: "loading", text: null });
    try {
      const { text, available } = readCopilot(await api.summarizeBatch(batch.batch_id));
      setSummary({ state: available ? "ready" : "error", text });
    } catch (e) {
      setSummary({ state: "error", text: e.message });
    }
  };

  const stats = batch?.aggregate_stats;
  const confidences = details.flatMap((d) =>
    (d.classifications || []).filter((c) => !c.is_synthetic_model).map((c) => c.confidence)
  );
  const topVariety = stats?.variety_distribution
    ? Object.entries(stats.variety_distribution).sort((a, b) => b[1] - a[1])[0]
    : null;

  return (
    <Page className="ba">
      <SectionHeader
        title="Batch analysis"
        subtitle="Analyze multiple maize seed images in one pass and compare their aggregate distribution."
        right={
          status !== "idle" && (
            <Button variant="ghost" size="sm" icon={RotateCcw} onClick={reset}>
              New batch
            </Button>
          )
        }
      />

      <GlassCard>
        <UploadZone
          files={files}
          onFiles={(f) => {
            setFiles(f);
            setStatus("idle");
            setBatch(null);
          }}
          multiple
          disabled={status === "running"}
          hint="JPG · PNG · WEBP — select several"
        />
        <div className="ba__controls">
          <div className="ba__field">
            <span className="ba__fieldlabel">Model</span>
            <p className="ba__modelnote">Unified seed model — 6 varieties + quality</p>
          </div>
          <Button
            icon={Layers}
            onClick={run}
            disabled={!files.length || status === "running"}
            loading={status === "running"}
          >
            {status === "running" ? `Analyzing ${files.length}…` : `Analyze ${files.length || ""} images`}
          </Button>
        </div>
      </GlassCard>

      {status === "idle" && !files.length && (
        <GlassCard>
          <EmptyState
            icon={Images}
            title="Upload a batch to begin"
            message="Select several seed photographs. Each is analysed independently and the results are aggregated below."
          />
        </GlassCard>
      )}

      {status === "error" && (
        <GlassCard accent="danger">
          <ErrorState
            title="Batch could not be processed"
            reason={error}
            action={
              <Button variant="secondary" size="sm" onClick={run}>
                Try again
              </Button>
            }
          />
        </GlassCard>
      )}

      {status === "running" && (
        <div className="ba__metrics">
          {[0, 1, 2, 3].map((i) => (
            <GlassCard key={i} className="ba__skelcard">
              <Skeleton height="14px" width="60%" />
              <Skeleton height="28px" width="40%" />
            </GlassCard>
          ))}
        </div>
      )}

      {status === "done" && stats && (
        <motion.div
          className="ba__results"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="ba__metrics">
            <MetricCard icon={Images} tone="green" label="Images processed" value={batch.successful_images} sub={batch.failed_images ? `${batch.failed_images} failed` : "all succeeded"} />
            <MetricCard icon={Sprout} tone="gold" label="Seeds detected" value={stats.total_seeds_detected} />
            <MetricCard
              icon={Gauge}
              tone="cyan"
              label="Avg. confidence"
              value={stats.average_confidence != null ? `${(stats.average_confidence * 100).toFixed(1)}%` : "—"}
              sub={stats.low_confidence_count ? `${stats.low_confidence_count} low-confidence` : undefined}
            />
            <MetricCard
              icon={Trophy}
              tone="gold"
              label="Top variety"
              value={topVariety ? pretty(topVariety[0]).split(" ").slice(-1)[0] : "—"}
              sub={topVariety ? `${topVariety[1]} of ${stats.total_seeds_detected}` : undefined}
            />
          </div>

          <div className="ba__charts">
            <GlassCard>
              <SectionHeader title="Variety distribution" level={3} />
              <DonutChart data={stats.variety_distribution} emptyMessage="No variety predictions in this batch." />
            </GlassCard>

            <GlassCard>
              <SectionHeader
                title="Confidence distribution"
                subtitle="Variety predictions, bucketed into 10% bins."
                level={3}
              />
              <ConfidenceHistogram values={confidences} emptyMessage="No confidence values recorded." />
            </GlassCard>
          </div>

          {stats.quality_distribution &&
            Object.keys(stats.quality_distribution).length > 0 && (
              <GlassCard>
                <SectionHeader
                  title="Kernel quality distribution"
                  subtitle="Graded against expert-assigned Good/Bad labels. Kernels outside the model's validated image range are counted separately as unverified."
                  level={3}
                />
                <DonutChart data={stats.quality_distribution} />
              </GlassCard>
            )}

          <GlassCard accent="cyan">
            <SectionHeader
              title="AI batch summary"
              subtitle="Gemini summarises the verified aggregate results above."
              level={3}
              right={
                <Button
                  variant="ai"
                  size="sm"
                  icon={Sparkles}
                  onClick={askSummary}
                  loading={summary.state === "loading"}
                >
                  Summarize batch
                </Button>
              }
            />
            {summary.state === "idle" && (
              <p className="muted ba__idle">Ask the copilot to summarise this batch in plain language.</p>
            )}
            {summary.state === "loading" && (
              <div className="ba__skel">
                <Skeleton height="12px" />
                <Skeleton height="12px" width="88%" />
              </div>
            )}
            {(summary.state === "ready" || summary.state === "error") && (
              <div className={`ba__ai ${summary.state === "error" ? "is-error" : ""}`}>
                <span className="ba__ailabel">
                  {summary.state === "error"
                    ? UNAVAILABLE_LABEL
                    : "AI-generated summary based on model analysis"}
                </span>
                <p>{summary.text}</p>
              </div>
            )}
          </GlassCard>

          {details.length > 0 && (
            <GlassCard>
              <SectionHeader title="Images in this batch" level={3} />
              <div className="ba__grid">
                {details.map((d, i) => {
                  const top = (d.classifications || []).find((c) => !c.is_synthetic_model);
                  return (
                    <motion.div
                      className="bcard"
                      key={d.analysis_id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.32, delay: 0.04 * i }}
                    >
                      <div className="bcard__img">
                        {d.image_path ? (
                          <img src={mediaUrl(d.image_path, 260)} alt={d.image_filename || "Analysed image"} />
                        ) : (
                          <span className="bcard__noimg">No preview</span>
                        )}
                        <span className="bcard__count mono">{d.seed_count}</span>
                      </div>
                      <span className="bcard__name" title={d.image_filename}>
                        {d.image_filename || d.analysis_id.slice(0, 8)}
                      </span>
                      {top ? (
                        <>
                          <span className="bcard__variety">{pretty(top.predicted_class)}</span>
                          <ConfidenceBar value={top.confidence} showValue label={null} />
                        </>
                      ) : (
                        <span className="bcard__variety faint">No variety prediction</span>
                      )}
                    </motion.div>
                  );
                })}
              </div>
            </GlassCard>
          )}
        </motion.div>
      )}
    </Page>
  );
}
