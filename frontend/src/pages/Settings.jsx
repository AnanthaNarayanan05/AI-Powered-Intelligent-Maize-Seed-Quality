import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Check,
  Cpu,
  Database,
  Download,
  Layers,
  Lock,
  RotateCcw,
  Server,
  SlidersHorizontal,
  Sparkles,
  AlertTriangle,
} from "lucide-react";
import { api } from "../api/client";
import { DEFAULTS, useSettings } from "../lib/settings";
import { Badge, Button, ErrorState, GlassCard, Page, SectionHeader, Skeleton, StatusDot } from "../components/ui";
import "./settings.css";

/* Two kinds of thing live on this page and they are kept visually apart on
   purpose:

   PREFERENCES are editable and stored in this browser. Every one of them is read
   by real code — the page names where — and there are no others. A switch that
   changed nothing would be worse than a missing switch, because it would look
   like the platform had a capability it does not have.

   INSTALLATION is read-only and comes from /api/system-info on every visit. None
   of it is written down here: a copy of the GPU name, the model version or the
   Gemini state would describe the machine this file was written on.

   The Gemini API key appears nowhere on this page in any form. It is read from
   the environment by the server, and the only thing the API will say about it is
   whether one is configured. */

const LANDING_PAGES = [
  { value: "/", label: "Overview" },
  { value: "/analyze", label: "Analyze" },
  { value: "/batch", label: "Batch" },
  { value: "/history", label: "History" },
  { value: "/compare", label: "Compare" },
  { value: "/lot", label: "Seed Lot" },
  { value: "/copilot", label: "AI Copilot" },
  { value: "/system", label: "System" },
];

const MODELS = [
  { value: "unified", label: "Unified seed model — served" },
  { value: "a", label: "Dataset A model — ablation" },
  { value: "b", label: "Dataset B model — ablation" },
];

const GLASS = [
  { value: "subtle", label: "Subtle", hint: "More photograph" },
  { value: "balanced", label: "Balanced", hint: "Default" },
  { value: "strong", label: "Strong", hint: "More contrast" },
];

const DENSITY = [
  { value: "comfortable", label: "Comfortable", hint: "Default" },
  { value: "compact", label: "Compact", hint: "Tighter spacing" },
];

const MOTION = [
  { value: "system", label: "Follow system", hint: "Uses prefers-reduced-motion" },
  { value: "reduced", label: "Reduced", hint: "Suppresses animation here" },
];

/* Settings the platform deliberately does not offer. Listed rather than omitted
   silently: an operator looking for one of these deserves to know it is absent by
   decision and why, instead of hunting for a control that was never built. */
const NOT_OFFERED = [
  {
    title: "Theme / light mode",
    reason:
      "The interface is dark by design — every surface is a translucent panel over a field photograph, and there is no light palette for it to fall back to.",
  },
  {
    title: "Language",
    reason: "The application ships one language. There is no translation layer for a selector to switch between.",
  },
  {
    title: "Notifications and alerts",
    reason:
      "Nothing in this platform emits a notification. There is no scheduler, no email path and no push channel, so every switch on such a panel would be decorative.",
  },
  {
    title: "A single \"clear all history\" action",
    reason:
      "Export and delete are both real now (below, and on each analysis in History), but there is deliberately no one button that empties the whole table — a bulk-destructive action with no per-row review is a bigger decision than either of those, and not one this page makes unasked.",
  },
  {
    title: "Gemini API key",
    reason:
      "The key is read from the server environment and never leaves it. This page can only report whether one is configured, which it does above.",
  },
];

function Row({ label, help, htmlFor, children }) {
  return (
    <div className="set__row">
      <div className="set__rowtext">
        <label className="set__rowlabel" htmlFor={htmlFor}>
          {label}
        </label>
        {help && <p className="set__rowhelp">{help}</p>}
      </div>
      <div className="set__rowctl">{children}</div>
    </div>
  );
}

/* Radios rather than buttons: this is a one-of-many choice, and a radio group
   already carries that meaning to a screen reader and to the keyboard. */
function Segmented({ name, options, value, onChange }) {
  return (
    <div className="seg" role="radiogroup" aria-label={name}>
      {options.map((o) => (
        <label key={o.value} className={`seg__opt ${value === o.value ? "is-on" : ""}`}>
          <input
            type="radio"
            name={name}
            value={o.value}
            checked={value === o.value}
            onChange={() => onChange(o.value)}
            className="sr-only"
          />
          <span className="seg__label">{o.label}</span>
          {o.hint && <span className="seg__hint">{o.hint}</span>}
        </label>
      ))}
    </div>
  );
}

function Toggle({ id, checked, onChange, label }) {
  return (
    <label className="tgl" htmlFor={id}>
      <input
        id={id}
        type="checkbox"
        className="sr-only"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className={`tgl__track ${checked ? "is-on" : ""}`} aria-hidden="true">
        <span className="tgl__knob" />
      </span>
      <span className="tgl__text">{checked ? "On" : "Off"}</span>
      <span className="sr-only">{label}</span>
    </label>
  );
}

function Facts({ items }) {
  return (
    <dl className="set__facts">
      {items.map(({ label, value, mono = true }) => (
        <div className="set__fact" key={label}>
          <dt>{label}</dt>
          <dd className={mono ? "mono" : undefined}>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

const bytes = (n) =>
  n == null ? "—" : n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`;

export default function Settings() {
  const { settings, save, reset } = useSettings();
  // The literal outcome of the last write, never an assumption about it.
  const [saved, setSaved] = useState(null); // { ok, at } | { ok: false, error }
  const [info, setInfo] = useState(null);
  const [infoState, setInfoState] = useState("loading");

  useEffect(() => {
    let cancelled = false;
    api
      .systemInfo()
      .then((d) => {
        if (cancelled) return;
        setInfo(d);
        setInfoState("ready");
      })
      .catch((e) => {
        if (cancelled) return;
        setInfo({ error: e.message });
        setInfoState("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Each control writes through immediately. The status line reports what the
  // write actually returned — a storage failure leaves the control where it was
  // and says so, rather than showing a success the browser never granted.
  const put = (key, value) => {
    const res = save({ ...settings, [key]: value });
    setSaved(res.ok ? { ok: true, at: new Date() } : { ok: false, error: res.error });
  };

  const onReset = () => {
    const res = reset();
    setSaved(res.ok ? { ok: true, at: new Date() } : { ok: false, error: res.error });
  };

  const isDefault = Object.keys(DEFAULTS).every((k) => settings[k] === DEFAULTS[k]);
  const runtime = info?.runtime;
  const db = info?.database;
  const det = info?.inference?.detection;

  return (
    <Page className="set">
      <SectionHeader
        title="Settings"
        subtitle="Preferences are stored in this browser and read by the pages named beside them. Everything below the divider is read from this server and cannot be edited here."
        right={
          <Button variant="ghost" size="sm" icon={RotateCcw} onClick={onReset} disabled={isDefault}>
            Reset to defaults
          </Button>
        }
      />

      {/* One status line for the whole page, because one write happens at a time. */}
      <div className="set__status" role="status" aria-live="polite">
        {saved?.ok && (
          <span className="set__saved">
            <Check size={14} aria-hidden="true" />
            Saved to this browser at{" "}
            <span className="mono">{saved.at.toLocaleTimeString()}</span>
          </span>
        )}
        {saved && !saved.ok && (
          <span className="set__savefail">
            <AlertTriangle size={14} aria-hidden="true" />
            Not saved — {saved.error} Your change was not applied.
          </span>
        )}
        {!saved && <span className="faint">Changes are written as you make them.</span>}
      </div>

      <div className="set__grid">
        <GlassCard tier="primary" className="set__panel">
          <h3 className="set__panelhead">
            <SlidersHorizontal size={16} aria-hidden="true" /> General
          </h3>

          <Row
            label="Landing page"
            htmlFor="set-landing"
            help="Where the app opens. Applies when you first load it; navigating to Overview afterwards still goes to Overview."
          >
            <select
              id="set-landing"
              className="set__select"
              value={settings.landingPage}
              onChange={(e) => put("landingPage", e.target.value)}
            >
              {LANDING_PAGES.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.label}
                </option>
              ))}
            </select>
          </Row>
        </GlassCard>

        <GlassCard tier="primary" className="set__panel">
          <h3 className="set__panelhead">
            <Layers size={16} aria-hidden="true" /> Analysis
          </h3>

          <Row
            label="Variety model"
            htmlFor="set-model"
            help="Sent as variety_dataset by the Analyze and Batch pages. The unified model is what the platform serves; the two dataset models are kept reachable so the ablation in the status report stays reproducible, and their predictions cover only their own dataset's varieties."
          >
            <select
              id="set-model"
              className="set__select"
              value={settings.defaultModel}
              onChange={(e) => put("defaultModel", e.target.value)}
            >
              {MODELS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </Row>

          <Row
            label="Similarity search"
            htmlFor="set-sim"
            help="Gallery neighbours for every detected kernel, searched in the feature space of the model that made the prediction. Off makes single-image analysis faster and leaves the Similar kernels panel empty. Batch analysis always runs it — that endpoint takes no such flag."
          >
            <Toggle
              id="set-sim"
              checked={settings.runSimilarity}
              onChange={(v) => put("runSimilarity", v)}
              label="Run similarity search with each analysis"
            />
          </Row>
        </GlassCard>

        <GlassCard tier="primary" className="set__panel set__panel--wide">
          <h3 className="set__panelhead">
            <Sparkles size={16} aria-hidden="true" /> Appearance
          </h3>

          <Row
            label="Panel opacity"
            help="How much of the background photograph shows through the panels. Every surface stays translucent at all three settings."
          >
            <Segmented
              name="Panel opacity"
              options={GLASS}
              value={settings.glass}
              onChange={(v) => put("glass", v)}
            />
          </Row>

          <Row label="Density" help="Spacing between and inside panels across every page.">
            <Segmented
              name="Density"
              options={DENSITY}
              value={settings.density}
              onChange={(v) => put("density", v)}
            />
          </Row>

          <Row
            label="Motion"
            help="Reduced suppresses transitions, the animated status dots and the Overview background video. The system option already does this when your OS asks for reduced motion."
          >
            <Segmented
              name="Motion"
              options={MOTION}
              value={settings.motion}
              onChange={(v) => put("motion", v)}
            />
          </Row>
        </GlassCard>
      </div>

      <SectionHeader
        title="This installation"
        subtitle="Read from /api/system-info each time this page opens. Nothing in this section is editable, and nothing in it is written into the frontend."
        level={2}
      />

      {infoState === "loading" && (
        <GlassCard className="set__panel">
          <Skeleton height="1rem" />
          <Skeleton height="1rem" width="70%" />
          <Skeleton height="1rem" width="45%" />
        </GlassCard>
      )}

      {infoState === "error" && (
        <GlassCard className="set__panel">
          <ErrorState
            title="System information unavailable"
            reason={`${info?.error} — the backend did not answer, so this section is left empty rather than filled in with values from somewhere else.`}
          />
        </GlassCard>
      )}

      {infoState === "ready" && (
        <div className="set__grid">
          <GlassCard className="set__panel">
            <h3 className="set__panelhead">
              <Cpu size={16} aria-hidden="true" /> Runtime
            </h3>
            <Facts
              items={[
                {
                  label: "Compute device",
                  value: runtime.device === "cuda" ? `CUDA — ${runtime.gpu}` : "CPU",
                },
                { label: "PyTorch", value: runtime.torch || "not installed" },
                { label: "Python", value: runtime.python },
                { label: "Host", value: runtime.platform },
                { label: "API version", value: info.backend.api_version },
              ]}
            />
          </GlassCard>

          <GlassCard className="set__panel">
            <h3 className="set__panelhead">
              <Database size={16} aria-hidden="true" /> Database
            </h3>
            <div className="set__inline">
              <StatusDot tone={db.status === "connected" ? "ok" : "warn"} />
              <span>{db.status}</span>
              <span className="faint mono">{db.engine}</span>
            </div>
            <Facts
              items={[
                { label: "Analyses", value: db.tables?.analyses ?? "—" },
                { label: "Detections", value: db.tables?.detections ?? "—" },
                { label: "Batches", value: db.tables?.batches ?? "—" },
                { label: "Similarity results", value: db.tables?.similarity_results ?? "—" },
                { label: "File size", value: bytes(db.size_bytes) },
                { label: "Path", value: db.path },
              ]}
            />
          </GlassCard>

          <GlassCard className="set__panel">
            <h3 className="set__panelhead">
              <Lock size={16} aria-hidden="true" /> Data &amp; privacy
            </h3>
            <p className="set__note">
              Export writes one row per stored analysis — the same summary History's card
              view already shows (filename, seed count, top variety and quality grade), not
              a per-seed dump. Deleting a specific analysis is done from its own card on the{" "}
              <Link to="/history">History</Link> page, where it can be reviewed before it's
              confirmed.
            </p>
            <a
              className="btn btn--secondary btn--md set__export"
              href={api.historyExportUrl()}
              download="maize_analysis_history.csv"
            >
              <Download size={16} aria-hidden="true" /> Export history (CSV)
            </a>
          </GlassCard>

          <GlassCard className="set__panel">
            <h3 className="set__panelhead">
              <Sparkles size={16} aria-hidden="true" /> Gemini
            </h3>
            <div className="set__inline">
              <StatusDot tone={info.gemini.configured ? "ai" : "warn"} />
              <span>{info.gemini.configured ? "Key configured" : "No key configured"}</span>
            </div>
            <Facts items={[{ label: "Model", value: info.gemini.model }]} />
            <p className="set__note">
              {info.gemini.configured
                ? "The key lives in the server environment. It is not readable from this page, and a configured key is not a guarantee that a given request will succeed — quota and timeouts are reported per request."
                : "Set GEMINI_API_KEY in the server environment to enable the copilot. Explanations and summaries return an unavailable notice until then; nothing else on the platform depends on it."}
            </p>
          </GlassCard>

          <GlassCard className="set__panel set__panel--wide" accent="gold">
            <h3 className="set__panelhead">
              <Lock size={16} aria-hidden="true" /> Detection configuration
              <Badge tone="neutral" title="Bound when the pipeline loads; a request cannot change them.">
                READ-ONLY
              </Badge>
            </h3>
            <p className="set__note">
              These are the values this server loaded from <span className="mono">configs/config.yaml</span>. They are
              not editable here, and the choice is deliberate rather than unfinished: the pipeline is a
              process-level singleton, so a per-request override would not reach it — and each threshold was
              picked against a measurement recorded in the project status report, so a control that let it be
              nudged would be a control over whether the counts are correct.
            </p>
            <Facts
              items={[
                { label: "Detector", value: `${det.framework} · ${det.model}` },
                { label: "Inference size", value: `${det.image_size} px` },
                { label: "Confidence threshold", value: det.conf_threshold },
                { label: "NMS IoU", value: det.nms_iou },
                { label: "Crop context padding", value: det.crop_context_pad },
                { label: "Match IoU", value: det.iou_threshold },
              ]}
            />
          </GlassCard>

          <GlassCard className="set__panel set__panel--wide">
            <h3 className="set__panelhead">
              <Server size={16} aria-hidden="true" /> Models on this machine
            </h3>
            <ul className="set__models">
              {info.models.map((m) => (
                <li key={m.key}>
                  <span className="set__modelname">{m.name}</span>
                  <span className="mono faint">v{m.version}</span>
                  <Badge tone={m.status === "ready" ? "ok" : m.status === "missing" ? "warn" : "neutral"}>
                    {m.status}
                  </Badge>
                </li>
              ))}
            </ul>
            <p className="set__note">
              Status, versions and metrics for each of these are on the{" "}
              <Link to="/system">System</Link> page, which reads the same report.
            </p>
          </GlassCard>
        </div>
      )}

      <SectionHeader
        title="Not offered here"
        subtitle="Controls a settings page might be expected to have, and why this one does not have them."
        level={2}
      />
      <GlassCard className="set__panel set__panel--wide">
        <ul className="set__absent">
          {NOT_OFFERED.map((n) => (
            <li key={n.title}>
              <span className="set__absenttitle">{n.title}</span>
              <span className="set__absentreason">{n.reason}</span>
            </li>
          ))}
        </ul>
      </GlassCard>
    </Page>
  );
}
