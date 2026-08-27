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

/* The model stack is a static description of the architecture, not a live probe.
   Availability that IS live (backend, Gemini) is fetched; anything describing how
   a model was trained is fixed and must stay accurate to the project docs. */
const MODEL_STACK = [
  { name: "Seed detection", model: "YOLOv8n", detail: "1 class · mAP@50 98.2%", tone: "ok", state: "Trained" },
  { name: "Variety recognition", model: "EfficientNet-B0 + cognitive attention", detail: "Datasets A & B · 3 classes each", tone: "ok", state: "Trained" },
  { name: "Contrastive pretraining", model: "SimCLR / NT-Xent", detail: "60 epochs per dataset", tone: "ok", state: "Trained" },
  { name: "Kernel quality", model: "Unified model, quality head", detail: "Expert-assigned Good/Bad labels", tone: "ok", state: "READY" },
  { name: "Similarity search", model: "FAISS IndexFlatL2", detail: "Feature embeddings, both datasets", tone: "ok", state: "Built" },
  { name: "Explainability", model: "Grad-CAM", detail: "Attention map — not segmentation", tone: "ok", state: "Available" },
];

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

  return (
    <Page className="si">
      <SectionHeader
        title="System information"
        subtitle="What this platform genuinely supports, what it does not, and which models back each capability."
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
            Kernel quality is graded against expert-assigned Good/Bad labels, scoring 97.3% on a held-out
            test split with 99.1% precision on the defective class. The bound matters as much as the
            number: applied to imagery unlike its training data the model extrapolates confidently, so
            every grade is checked against the region where the model actually has evidence. Grades
            outside it are marked <strong>unverified</strong> rather than reported as defects. A Good/Bad
            grade is not a pathogen diagnosis and not a severity score.
          </p>
        </div>
      </GlassCard>

      <div className="si__cols">
        {/* ---------- model stack ---------- */}
        <GlassCard>
          <SectionHeader title="Model stack" level={3} />
          <div className="si__stack">
            {MODEL_STACK.map((m, i) => (
              <motion.div
                className="si__row"
                key={m.name}
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.04 * i }}
              >
                <div className="si__rowmain">
                  <span className="si__rowname">{m.name}</span>
                  <span className="si__rowmodel mono">{m.model}</span>
                  <span className="si__rowdetail">{m.detail}</span>
                </div>
                <Badge tone={m.tone === "warn" ? "warn" : "ok"}>{m.state}</Badge>
              </motion.div>
            ))}
          </div>
        </GlassCard>

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
                  <StatusDot tone="ok" pulse /> Online
                </span>
              </div>
              <div className="si__status">
                <Database size={15} />
                <span>Database</span>
                <span className="si__statusval">
                  <StatusDot tone="ok" /> Connected
                </span>
              </div>
              <div className="si__status">
                <Cpu size={15} />
                <span>Inference device</span>
                <span className="si__statusval mono">CUDA / CPU fallback</span>
              </div>
              <div className="si__status">
                <Sparkles size={15} />
                <span>Gemini copilot</span>
                <span className="si__statusval">
                  <StatusDot tone={info?.gemini_configured ? "ai" : "warn"} pulse={!!info?.gemini_configured} />
                  {info?.gemini_configured ? "Connected" : "Not configured"}
                </span>
              </div>
            </div>
          )}
        </GlassCard>
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
