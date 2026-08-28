import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Check, X, AlertTriangle, Cpu, Database, Server, Sparkles } from "lucide-react";
import { api } from "../api/client";
import {
  Badge,
  ErrorState,
  GlassCard,
  Page,
  SectionHeader,
  Skeleton,
  StatusDot,
} from "../components/ui";
import "./systeminfo.css";

/* Nothing about the model stack is written into this file. It is read from
   /api/system-info, which probes checkpoints, evaluation JSON and index sidecars on
   every request — the previous hard-coded table went on advertising "Datasets A & B
   · 3 classes each" for weeks after one 6-variety model replaced both. */

const STATUS_TONE = { ready: "ok", available: "neutral", missing: "warn" };
const STATUS_LABEL = { ready: "Serving", available: "Trained", missing: "Not on this machine" };

function ModelRow({ model, index }) {
  return (
    <motion.div
      className="si__row"
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: 0.04 * index }}
    >
      <div className="si__rowmain">
        <span className="si__rowname">{model.name}</span>
        <span className="si__rowmodel mono">{model.architecture}</span>
        <span className="si__rowdetail">{model.role}</span>

        {model.metrics.length > 0 && (
          <div className="si__metrics">
            {model.metrics.map((m) => (
              <span className="si__metric" key={m.label} title={m.n ? `n = ${m.n}` : undefined}>
                <span className="si__metriclabel">{m.label}</span>
                <span className="si__metricval mono">{m.value}</span>
              </span>
            ))}
          </div>
        )}

        {/* The training script's own caveat, shown verbatim rather than paraphrased. */}
        {model.label_provenance === "synthetic" && (
          <span className="si__rownote">
            Labels are synthetic. {model.note}
          </span>
        )}

        <span className="si__rowpath mono" title={model.checkpoint}>
          {model.checkpoint}
        </span>
      </div>
      <Badge
        tone={STATUS_TONE[model.status] || "neutral"}
        title={model.trained_at ? `Checkpoint written ${model.trained_at}` : undefined}
      >
        {STATUS_LABEL[model.status] || model.status}
      </Badge>
    </motion.div>
  );
}

export default function SystemInfo() {
  const [info, setInfo] = useState(null);
  const [state, setState] = useState("loading");
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api
      .systemInfo()
      .then((d) => {
        if (cancelled) return;
        setInfo(d);
        setState("ready");
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

  const models = info?.models || [];
  const served = models.filter((m) => m.served);
  const offline = models.filter((m) => !m.served);
  const runtime = info?.runtime;
  const db = info?.database;
  const galleries = (info?.similarity_indices || []).filter((g) => g.status === "built");
  // the banner's headline number comes from the checkpoint's own evaluation file
  const qualityAcc = models
    .find((m) => m.key === "unified_seed_model")
    ?.metrics?.find((x) => x.label === "Quality accuracy")?.value;

  return (
    <Page className="si">
      <SectionHeader
        title="System information"
        subtitle="Read live from this server: the models actually loaded, the numbers they actually scored, and the capabilities the data cannot support."
      />

      {state === "error" && (
        <GlassCard accent="danger">
          <ErrorState title="System information unavailable" reason={error} />
        </GlassCard>
      )}

      {/* ---------- honesty banner ---------- */}
      <GlassCard accent="warn" className="si__banner">
        <span className="si__bannericon">
          <AlertTriangle size={18} />
        </span>
        <div>
          <strong className="si__bannertitle">Quality grading is real, and bounded</strong>
          <p className="si__bannerbody">
            Kernel quality is graded against expert-assigned Good/Bad labels
            {qualityAcc ? `, scoring ${qualityAcc} on a held-out test split` : ""}. The bound matters as much as the number: applied to imagery unlike
            its training data the model extrapolates confidently, so every grade is checked against
            the region where the model actually has evidence. Grades outside it are marked{" "}
            <strong>unverified</strong> rather than reported as defects. A Good/Bad grade is not a
            pathogen diagnosis and not a severity score.
          </p>
        </div>
      </GlassCard>

      <div className="si__cols">
        {/* ---------- model stack ---------- */}
        <GlassCard>
          <SectionHeader
            title="Model stack"
            subtitle="Status, metrics and checkpoint paths are probed on this machine at request time."
            level={3}
          />
          {state === "loading" ? (
            <div className="si__skel">
              <Skeleton height="52px" />
              <Skeleton height="52px" />
              <Skeleton height="52px" />
            </div>
          ) : (
            <>
              <div className="si__stack">
                {served.map((m, i) => (
                  <ModelRow key={m.key} model={m} index={i} />
                ))}
              </div>

              {offline.length > 0 && (
                <>
                  <SectionHeader
                    title="Trained but not served"
                    subtitle="Kept for reproducibility. These do not contribute to any result the platform reports."
                    level={4}
                  />
                  <div className="si__stack si__stack--muted">
                    {offline.map((m, i) => (
                      <ModelRow key={m.key} model={m} index={i} />
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </GlassCard>

        <div className="si__side">
          {/* ---------- live status ---------- */}
          <GlassCard>
            <SectionHeader title="System status" level={3} />
            {state === "loading" ? (
              <div className="si__skel">
                <Skeleton height="14px" />
                <Skeleton height="14px" width="80%" />
                <Skeleton height="14px" width="60%" />
              </div>
            ) : (
              <div className="si__statuslist">
                <div className="si__status">
                  <Server size={15} />
                  <span>Backend API</span>
                  <span className="si__statusval">
                    <StatusDot tone={state === "ready" ? "ok" : "warn"} pulse={state === "ready"} />
                    {state === "ready" ? `Online · v${info.backend.api_version}` : "Unreachable"}
                  </span>
                </div>
                <div className="si__status">
                  <Database size={15} />
                  <span>Database</span>
                  <span className="si__statusval">
                    <StatusDot tone={db?.status === "connected" ? "ok" : "warn"} />
                    {db?.status === "connected"
                      ? `${db.tables.analyses.toLocaleString()} analyses`
                      : db?.status || "unknown"}
                  </span>
                </div>
                <div className="si__status">
                  <Cpu size={15} />
                  <span>Inference device</span>
                  {/* the GPU name only appears when CUDA actually answered */}
                  <span className="si__statusval mono">
                    {runtime?.gpu || (runtime ? "CPU" : "unknown")}
                  </span>
                </div>
                <div className="si__status">
                  <Sparkles size={15} />
                  <span>Gemini copilot</span>
                  <span className="si__statusval">
                    <StatusDot
                      tone={info?.gemini?.configured ? "ai" : "warn"}
                      pulse={!!info?.gemini?.configured}
                    />
                    {info?.gemini?.configured ? info.gemini.model : "Not configured"}
                  </span>
                </div>
                {runtime?.torch && (
                  <div className="si__status">
                    <Cpu size={15} />
                    <span>Runtime</span>
                    <span className="si__statusval mono">
                      torch {runtime.torch} · py {runtime.python}
                    </span>
                  </div>
                )}
              </div>
            )}
          </GlassCard>

          {/* ---------- similarity galleries ---------- */}
          {galleries.length > 0 && (
            <GlassCard>
              <SectionHeader
                title="Similarity galleries"
                subtitle="Each gallery is bound to the encoder that built it; a query from any other model is refused rather than answered."
                level={3}
              />
              <div className="si__statuslist">
                {galleries.map((g) => (
                  <div className="si__status" key={g.dataset}>
                    <span className="mono">{g.encoder}</span>
                    <span className="si__statusval">
                      {g.gallery_size.toLocaleString()} images · {g.gallery_split} split
                    </span>
                  </div>
                ))}
              </div>
              <p className="si__note">
                Neighbours are visually similar images, not a verdict on the query. Their labels
                describe them, never the seed you uploaded.
              </p>
            </GlassCard>
          )}
        </div>
      </div>

      {/* ---------- capability matrix ---------- */}
      <div className="si__cols">
        <GlassCard accent="green">
          <SectionHeader title="Supported" level={3} />
          {state === "loading" ? (
            <Skeleton height="120px" />
          ) : (
            <ul className="si__caps">
              {(info?.supported_capabilities || []).map((c) => (
                <li key={c}>
                  <span className="si__capicon si__capicon--ok">
                    <Check size={12} />
                  </span>
                  {c}
                </li>
              ))}
            </ul>
          )}
        </GlassCard>

        <GlassCard accent="danger">
          <SectionHeader
            title="Explicitly not supported"
            subtitle="Listed deliberately. Absence of a capability is stated, never hidden."
            level={3}
          />
          {state === "loading" ? (
            <Skeleton height="120px" />
          ) : (
            <ul className="si__caps">
              {(info?.explicitly_not_supported || []).map((c) => (
                <li key={c}>
                  <span className="si__capicon si__capicon--no">
                    <X size={12} />
                  </span>
                  {c}
                </li>
              ))}
            </ul>
          )}
        </GlassCard>
      </div>
    </Page>
  );
}
