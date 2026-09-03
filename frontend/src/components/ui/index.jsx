/* ============================================================
   MAIZE INTELLIGENCE — shared UI primitives.
   Every page composes these; no page defines its own card/button/badge.
   ============================================================ */
import { motion } from "framer-motion";
import { AlertTriangle, Inbox, Loader2 } from "lucide-react";
import "./ui.css";

/* ---------- GlassCard ---------- */
export function GlassCard({ children, className = "", accent, hover = false, ...rest }) {
  return (
    <div
      className={`glass-card ${hover ? "glass-card--hover" : ""} ${
        accent ? `glass-card--${accent}` : ""
      } ${className}`}
      {...rest}
    >
      {children}
    </div>
  );
}

/* ---------- SectionHeader ---------- */
export function SectionHeader({ title, subtitle, right, level = 2 }) {
  const H = `h${level}`;
  return (
    <div className="section-header">
      <div>
        <H className="section-header__title">{title}</H>
        {subtitle && <p className="section-header__sub">{subtitle}</p>}
      </div>
      {right && <div className="section-header__right">{right}</div>}
    </div>
  );
}

/* ---------- Button ---------- */
export function Button({
  children,
  variant = "primary",
  size = "md",
  icon: Icon,
  loading = false,
  className = "",
  ...rest
}) {
  return (
    <button
      className={`btn btn--${variant} btn--${size} ${className}`}
      disabled={loading || rest.disabled}
      {...rest}
    >
      {loading ? (
        <Loader2 size={16} className="btn__spin" aria-hidden="true" />
      ) : (
        Icon && <Icon size={16} aria-hidden="true" />
      )}
      <span>{children}</span>
    </button>
  );
}

/* ---------- StatusDot ----------
   tone: ok | ai | warn | error | idle                                    */
export function StatusDot({ tone = "idle", pulse = false }) {
  return <span className={`status-dot status-dot--${tone} ${pulse ? "is-pulsing" : ""}`} />;
}

export function StatusPill({ label, value, tone = "idle", pulse = false }) {
  return (
    <div className="status-pill">
      <StatusDot tone={tone} pulse={pulse} />
      <span className="status-pill__label">{label}</span>
      <span className={`status-pill__value status-pill__value--${tone}`}>{value}</span>
    </div>
  );
}

/* ---------- Badge ----------
   Provenance badges are load-bearing in this project: a prediction from the
   synthetic defect model must never be visually indistinguishable from one
   backed by real labelled data.                                          */
export function Badge({ children, tone = "neutral", title }) {
  return (
    <span className={`badge badge--${tone}`} title={title}>
      {children}
    </span>
  );
}

export function ProvenanceBadge({ synthetic }) {
  return synthetic ? (
    <Badge
      tone="warn"
      title="Trained on procedurally generated damage patterns, not real plant pathology. Demonstration only."
    >
      SYNTHETIC MODEL
    </Badge>
  ) : (
    <Badge tone="ok" title="Trained on real, human-labelled variety data.">
      REAL DATA
    </Badge>
  );
}

/* ---------- ConfidenceBar ---------- */
// Hidden project-wide for now: displayed confidence is uncalibrated and, for
// some models (visible-symptom classifier), deliberately low by design. Bring
// this back once calibration/coverage is improved -- see model_registry.yaml.
const SHOW_CONFIDENCE = false;

export function ConfidenceBar({ value, tone, label, showValue = true, delay = 0 }) {
  if (!SHOW_CONFIDENCE) return null;
  const pct = Math.max(0, Math.min(1, value ?? 0));
  // Low confidence must look different from high confidence, not just read differently.
  const auto = pct >= 0.85 ? "ok" : pct >= 0.6 ? "warn" : "error";
  const t = tone || auto;
  return (
    <div className="confbar">
      {(label || showValue) && (
        <div className="confbar__head">
          {label && <span className="confbar__label">{label}</span>}
          {showValue && <span className={`confbar__val mono confbar__val--${t}`}>{(pct * 100).toFixed(1)}%</span>}
        </div>
      )}
      <div className="confbar__track">
        <motion.div
          className={`confbar__fill confbar__fill--${t}`}
          initial={{ width: 0 }}
          animate={{ width: `${pct * 100}%` }}
          transition={{ duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] }}
        />
      </div>
    </div>
  );
}

/* ---------- MetricCard ---------- */
export function MetricCard({ icon: Icon, label, value, sub, tone = "gold", loading = false }) {
  return (
    <GlassCard className="metric" hover>
      <div className={`metric__icon metric__icon--${tone}`}>{Icon && <Icon size={18} />}</div>
      <div className="metric__body">
        <span className="metric__label">{label}</span>
        {loading ? (
          <Skeleton width="70px" height="28px" />
        ) : (
          <span className="metric__value mono">{value}</span>
        )}
        {sub && <span className="metric__sub">{sub}</span>}
      </div>
    </GlassCard>
  );
}

/* ---------- Skeleton ---------- */
export function Skeleton({ width = "100%", height = "1rem", radius = "var(--r-sm)", className = "" }) {
  return <span className={`skeleton ${className}`} style={{ width, height, borderRadius: radius }} />;
}

/* ---------- EmptyState ---------- */
export function EmptyState({ icon: Icon = Inbox, title, message, action }) {
  return (
    <div className="state-block">
      <div className="state-block__icon">
        <Icon size={26} />
      </div>
      <h3 className="state-block__title">{title}</h3>
      {message && <p className="state-block__msg">{message}</p>}
      {action}
    </div>
  );
}

/* ---------- ErrorState ----------
   Shows a reason and an action, never a stack trace.                      */
export function ErrorState({ title = "Something went wrong", reason, action }) {
  return (
    <div className="state-block state-block--error">
      <div className="state-block__icon state-block__icon--error">
        <AlertTriangle size={26} />
      </div>
      <h3 className="state-block__title">{title}</h3>
      {reason && (
        <p className="state-block__msg">
          <span className="faint">Reason: </span>
          {reason}
        </p>
      )}
      {action}
    </div>
  );
}

/* ---------- Page wrapper: consistent enter transition ---------- */
export function Page({ children, className = "" }) {
  return (
    <motion.div
      className={`page ${className}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}
