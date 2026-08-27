import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  ScanSearch,
  Sprout,
  Leaf,
  Activity,
  Brain,
  Images,
  Sparkles,
  RotateCcw,
} from "lucide-react";
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
import UploadZone from "../components/UploadZone";
import DetectionCanvas from "../components/DetectionCanvas";
import ProcessingSequence from "../components/ProcessingSequence";
import "./analyze.css";

// One unified model now covers all six varieties and grades kernel quality, so
// there is nothing for the operator to choose between.
const VARIETIES = [
  "Chulpi Cancha", "Indurata", "Rugosa", "Bhihilifa", "SanzalSima", "WangDataa",
];

const pretty = (s) => (s ? s.replace(/_/g, " ") : "—");

export default function Analyze() {
  const [files, setFiles] = useState([]);
  const [dataset] = useState("unified");
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);

  const [gradcam, setGradcam] = useState({ state: "idle", url: null, error: null });
  const [explainTab, setExplainTab] = useState("original");
  const [aiText, setAiText] = useState(null);
  const [aiState, setAiState] = useState("idle");

  const file = files[0] || null;
  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);
  useEffect(() => () => gradcam.url && URL.revokeObjectURL(gradcam.url), [gradcam.url]);

  const reset = () => {
    setFiles([]);
    setStatus("idle");
    setResult(null);
    setError(null);
    setSelected(null);
    setGradcam({ state: "idle", url: null, error: null });
    setExplainTab("original");
    setAiText(null);
    setAiState("idle");
  };

  const run = async () => {
    if (!file) return;
    setStatus("running");
    setError(null);
    setResult(null);
    setSelected(null);
    try {
      const data = await api.analyzeImage(file, dataset);
      setResult(data);
      setStatus("done");
      if (data.seeds?.length) setSelected(0);
    } catch (e) {
      setError(e.message || "The analysis could not be completed.");
      setStatus("error");
    }
  };

  const loadGradcam = async () => {
    if (!file || gradcam.state === "loading") return;
    setGradcam({ state: "loading", url: null, error: null });
    try {
      const url = await api.gradcam(file, dataset);
      setGradcam({ state: "ready", url, error: null });
      setExplainTab("gradcam");
    } catch (e) {
      setGradcam({ state: "error", url: null, error: e.message });
    }
  };

  const askAi = async () => {
    if (!result?.analysis_id) return;
    setAiState("loading");
    try {
      const r = await api.explainAnalysis(result.analysis_id);
      setAiText(r.explanation || r.answer || r.text || "No explanation returned.");
      setAiState("ready");
    } catch (e) {
      setAiText(e.message);
      setAiState("error");
    }
  };

  const seeds = result?.seeds || [];
  const active = selected != null ? seeds[selected] : null;

  return (
    <Page className="an">
      <SectionHeader
        title="Analyze seed image"
        subtitle="Upload an image to run detection, variety recognition, defect classification, similarity search and explainability in a single pass."
        right={
          status !== "idle" && (
            <Button variant="ghost" size="sm" icon={RotateCcw} onClick={reset}>
              New analysis
            </Button>
          )
        }
      />

      <div className="an__top">
        {/* ---------- upload ---------- */}
        <GlassCard className="an__upload">
          <h3 className="an__cardtitle">Upload image</h3>
          <UploadZone
            files={files}
            onFiles={(f) => {
              setFiles(f);
              setStatus("idle");
              setResult(null);
              setError(null);
            }}
            disabled={status === "running"}
          />

          {previewUrl && (
            <div className="an__preview">
              <img src={previewUrl} alt="Selected upload preview" />
            </div>
          )}

          <div className="an__field">
            <span className="an__fieldlabel">Model</span>
            <p className="an__modelnote">
              Unified seed model — {VARIETIES.length} varieties plus kernel quality,
              from a single pass.
            </p>
          </div>

          <Button
            size="lg"
            icon={ScanSearch}
            onClick={run}
            disabled={!file || status === "running"}
            loading={status === "running"}
            className="an__go"
          >
            {status === "running" ? "Analyzing…" : "Analyze image"}
          </Button>
        </GlassCard>

        {/* ---------- pipeline / summary ---------- */}
        <GlassCard className="an__pipeline">
          <h3 className="an__cardtitle">Analysis pipeline</h3>
          {status === "idle" && (
            <EmptyState
              icon={Images}
              title="No image selected"
              message="Choose a seed photograph to begin. Nothing is sent anywhere until you press Analyze."
            />
          )}
          {(status === "running" || status === "done") && (
            <ProcessingSequence done={status === "done"} />
          )}
          {status === "error" && (
            <ErrorState
              title="Analysis could not be completed"
              reason={error}
              action={
                <Button variant="secondary" size="sm" onClick={run}>
                  Try again
                </Button>
              }
            />
          )}
        </GlassCard>
      </div>

      {/* ---------- results ---------- */}
      {status === "done" && (
        <motion.div
          className="an__results"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
        >
          {seeds.length === 0 ? (
            <GlassCard>
              <EmptyState
                icon={Sprout}
                title="No seeds detected"
                message="The detector found no maize seed in this image. Try a clearer photograph, or one where the seeds fill more of the frame."
                action={
                  <Button variant="secondary" size="sm" onClick={reset}>
                    Upload another image
                  </Button>
                }
              />
            </GlassCard>
          ) : (
            <>
              {/* headline */}
              <div className="an__headline">
                <div className="an__count">
                  <span className="an__countnum mono">
                    {String(result.seed_count).padStart(2, "0")}
                  </span>
                  <span className="an__countlabel">
                    {result.seed_count === 1 ? "seed detected" : "seeds detected"}
                  </span>
                </div>
                {result.warnings?.length > 0 ? (
                  <div className="an__warnings">
                    {result.warnings.map((w, i) => (
                      <Badge key={i} tone="warn">
                        {w}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <Badge tone="ok">All pipeline stages completed</Badge>
                )}
              </div>

              <div className="an__grid">
                {/* image + boxes */}
                <GlassCard className="an__viewer">
                  <SectionHeader
                    title="Detection"
                    subtitle="Click a box to inspect that seed."
                    level={3}
                  />
                  <DetectionCanvas
                    src={previewUrl}
                    seeds={seeds}
                    selectedIndex={selected}
                    onSelect={setSelected}
                  />
                </GlassCard>

                {/* per-seed detail */}
                <div className="an__detail">
                  {active ? (
                    <>
                      <GlassCard accent="gold" className="an__resultcard">
                        <div className="an__rowhead">
                          <span className="an__eyebrow">
                            <Leaf size={13} /> Variety recognition
                          </span>
                          <ProvenanceBadge synthetic={false} />
                        </div>
                        <h3 className="an__predict">
                          {pretty(active.variety_prediction?.predicted_class)}
                        </h3>
                        <ConfidenceBar
                          value={active.variety_prediction?.confidence}
                          label="Confidence"
                        />
                        <div className="an__probs">
                          {Object.entries(active.variety_prediction?.class_probabilities || {})
                            .sort((a, b) => b[1] - a[1])
                            .map(([cls, p], i) => (
                              <div className="an__prob" key={cls}>
                                <span className="an__probname">{pretty(cls)}</span>
                                <ConfidenceBar value={p} showValue delay={0.05 * i} label={null} />
                              </div>
                            ))}
                        </div>
                        <p className="an__model faint mono">
                          model: {active.variety_prediction?.model || "—"}
                        </p>
                      </GlassCard>

                      <GlassCard
                        accent={active.quality_prediction?.out_of_distribution ? "warn" : undefined}
                        className="an__resultcard"
                      >
                        <div className="an__rowhead">
                          <span className="an__eyebrow">
                            <Activity size={13} /> Kernel quality
                          </span>
                          {active.quality_prediction?.out_of_distribution ? (
                            <Badge tone="warn">unverified</Badge>
                          ) : (
                            <ProvenanceBadge synthetic={false} />
                          )}
                        </div>
                        {active.quality_prediction ? (
                          <>
                            <h3
                              className={`an__predict ${
                                /^(bad|defect)/i.test(active.quality_prediction.predicted_class)
                                  ? "an__predict--warn"
                                  : ""
                              }`}
                            >
                              {pretty(active.quality_prediction.predicted_class)}
                            </h3>
                            <ConfidenceBar
                              value={active.quality_prediction.confidence}
                              label="Confidence"
                            />
                            <p className="an__disclaimer">
                              {active.quality_prediction.caveat ||
                                "Graded against expert-assigned Good/Bad kernel labels."}
                            </p>
                          </>
                        ) : (
                          <p className="muted">Quality model unavailable for this analysis.</p>
                        )}
                      </GlassCard>
                    </>
                  ) : (
                    <GlassCard>
                      <EmptyState
                        icon={Sprout}
                        title="Select a seed"
                        message="Click any bounding box to see its variety and defect predictions."
                      />
                    </GlassCard>
                  )}
                </div>
              </div>

              {/* explainability */}
              <GlassCard accent="cyan">
                <SectionHeader
                  title="Explainability"
                  subtitle="Model attention visualization — this is not a segmentation mask and does not measure defect area."
                  level={3}
                  right={
                    <div className="an__tabs">
                      <button
                        className={`an__tab ${explainTab === "original" ? "is-on" : ""}`}
                        onClick={() => setExplainTab("original")}
                      >
                        Original
                      </button>
                      <button
                        className={`an__tab ${explainTab === "gradcam" ? "is-on" : ""}`}
                        onClick={() =>
                          gradcam.state === "ready" ? setExplainTab("gradcam") : loadGradcam()
                        }
                      >
                        Grad-CAM
                      </button>
                    </div>
                  }
                />

                <div className="an__explain">
                  {explainTab === "original" && (
                    <img src={previewUrl} alt="Original upload" className="an__explainimg" />
                  )}
                  {explainTab === "gradcam" && gradcam.state === "ready" && (
                    <motion.img
                      key="gc"
                      src={gradcam.url}
                      alt="Grad-CAM model attention heatmap"
                      className="an__explainimg"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.35 }}
                    />
                  )}
                  {gradcam.state === "loading" && <Skeleton height="280px" radius="var(--r-md)" />}
                  {gradcam.state === "error" && (
                    <ErrorState title="Grad-CAM unavailable" reason={gradcam.error} />
                  )}
                  {explainTab === "gradcam" && gradcam.state === "idle" && (
                    <EmptyState
                      icon={Brain}
                      title="Generate attention map"
                      message="Grad-CAM highlights the image regions that drove the variety prediction."
                      action={
                        <Button variant="ai" size="sm" icon={Brain} onClick={loadGradcam}>
                          Generate
                        </Button>
                      }
                    />
                  )}
                </div>
              </GlassCard>

              {/* similarity */}
              {active?.similarity_results?.length > 0 && (
                <GlassCard>
                  <SectionHeader
                    title="Visual memory"
                    subtitle="Nearest matches in feature space. This is visual similarity, not certification or ground-truth matching."
                    level={3}
                  />
                  <div className="simstrip">
                    {active.similarity_results.map((s, i) => (
                      <motion.div
                        className="simcard"
                        key={`${s.path}-${i}`}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.35, delay: 0.06 * i }}
                      >
                        <div className="simcard__imgwrap">
                          {/* Not lazy-loaded: these are part of the result the user
                              just requested and there are only top-k of them. */}
                          <img
                            src={mediaUrl(s.path, 220)}
                            alt={`Similar seed ${i + 1}: ${pretty(s.label)}`}
                          />
                          <span className="simcard__rank mono">{i + 1}</span>
                        </div>
                        <span className="simcard__label">{pretty(s.label)}</span>
                        <span className="simcard__score mono faint">
                          d={s.distance?.toFixed(1)}
                        </span>
                      </motion.div>
                    ))}
                  </div>
                </GlassCard>
              )}

              {/* AI explanation */}
              <GlassCard accent="cyan">
                <SectionHeader
                  title="AI interpretation"
                  subtitle="Gemini explains the verified model output above. It never produces predictions itself."
                  level={3}
                  right={
                    <Button
                      variant="ai"
                      size="sm"
                      icon={Sparkles}
                      onClick={askAi}
                      loading={aiState === "loading"}
                    >
                      Explain this result
                    </Button>
                  }
                />
                {aiState === "idle" && (
                  <p className="muted an__aiidle">
                    Ask the copilot to describe this analysis in plain language.
                  </p>
                )}
                {aiState === "loading" && (
                  <div className="an__aiskel">
                    <Skeleton height="12px" />
                    <Skeleton height="12px" width="92%" />
                    <Skeleton height="12px" width="76%" />
                  </div>
                )}
                {(aiState === "ready" || aiState === "error") && (
                  <div className={`an__ai ${aiState === "error" ? "is-error" : ""}`}>
                    <span className="an__ailabel">AI-generated explanation based on model analysis</span>
                    <p>{aiText}</p>
                  </div>
                )}
              </GlassCard>
            </>
          )}
        </motion.div>
      )}
    </Page>
  );
}
