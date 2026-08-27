import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { ClipboardList, Sprout, ShieldAlert, Info } from "lucide-react";
import { api } from "../api/client";
import {
  Badge,
  Button,
  EmptyState,
  ErrorState,
  GlassCard,
  Page,
  SectionHeader,
  Skeleton,
} from "../components/ui";
import "./lot.css";

const pretty = (s) => (s ? s.replace(/_/g, " ") : "—");

export default function Lot() {
  const [analyses, setAnalyses] = useState([]);
  const [listState, setListState] = useState("loading");
  const [listError, setListError] = useState(null);
  const [selected, setSelected] = useState(() => new Set());
  const [declared, setDeclared] = useState("");
  const [lotRef, setLotRef] = useState("");
  const [report, setReport] = useState({ state: "idle" });

  useEffect(() => {
    let cancelled = false;
    api
      .history(50, 0)
      .then((d) => {
        if (cancelled) return;
        const list = d.analyses || [];
        setAnalyses(list);
        setSelected(new Set(list.map((a) => a.analysis_id)));
        setListState("ready");
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

  // Varieties already seen across the loaded analyses, offered as declared-variety
  // options. Free text stays allowed: the lot may be sold as something the model
  // has never predicted, and that mismatch is exactly what an off-type rate exists
  // to surface.
  const knownVarieties = useMemo(() => {
    const s = new Set();
    for (const a of analyses)
      for (const c of a.classifications || [])
        if (!c.is_synthetic_model && c.predicted_class) s.add(c.predicted_class);
    return [...s].sort();
  }, [analyses]);

  const toggle = (id) =>
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  const run = async () => {
    if (!selected.size) return;
    setReport({ state: "loading" });
    try {
      const r = await api.lotReport(
        [...selected],
        declared.trim() || null,
        lotRef.trim() || null
      );
      setReport({ state: "ready", data: r });
    } catch (e) {
      setReport({ state: "error", message: e.message });
    }
  };

  const d = report.data;

  return (
    <Page className="lot">
      <SectionHeader
        title="Seed lot composition"
        subtitle="Aggregates per-kernel variety and quality predictions across many images into a lot-level breakdown."
      />

      <GlassCard accent="warn" className="lot__disclaimer">
        <ShieldAlert size={18} aria-hidden="true" />
        <p>
          <strong>This is not a seed certification.</strong> Certification is a legal
          determination made by an accredited laboratory following a prescribed sampling
          protocol, including physical purity by weight. This report describes what the
          models predicted about the images you selected — nothing more.
        </p>
      </GlassCard>

      {listState === "loading" && (
        <GlassCard>
          <Skeleton height="180px" />
        </GlassCard>
      )}

      {listState === "error" && (
        <GlassCard accent="danger">
          <ErrorState title="Could not load analyses" reason={listError} />
        </GlassCard>
      )}

      {listState === "ready" && analyses.length === 0 && (
        <GlassCard>
          <EmptyState
            icon={Sprout}
            title="No analyses stored yet"
            message="Analyse some images first — a lot report is built from analyses you have already run."
          />
        </GlassCard>
      )}

      {listState === "ready" && analyses.length > 0 && (
        <>
          <GlassCard>
            <SectionHeader
              title="Build the lot"
              level={3}
              right={
                <div className="lot__selectactions">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelected(new Set(analyses.map((a) => a.analysis_id)))}
                  >
                    Select all
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => setSelected(new Set())}>
                    Clear
                  </Button>
                </div>
              }
            />

            <div className="lot__fields">
              <label className="lot__field">
                <span>Lot reference (optional)</span>
                <input
                  value={lotRef}
                  onChange={(e) => setLotRef(e.target.value)}
                  placeholder="e.g. LOT-2026-014"
                />
              </label>
              <label className="lot__field">
                <span>Declared variety (optional)</span>
                <input
                  value={declared}
                  onChange={(e) => setDeclared(e.target.value)}
                  placeholder="Measure off-type against this instead of the majority"
                  list="lot-varieties"
                />
                <datalist id="lot-varieties">
                  {knownVarieties.map((v) => (
                    <option key={v} value={v} />
                  ))}
                </datalist>
              </label>
            </div>

            <div className="lot__picker">
              {analyses.map((a) => (
                <label
                  key={a.analysis_id}
                  className={`lot__row ${selected.has(a.analysis_id) ? "is-on" : ""}`}
                >
                  <input
                    type="checkbox"
                    checked={selected.has(a.analysis_id)}
                    onChange={() => toggle(a.analysis_id)}
                  />
                  <span className="lot__rowname">
                    {a.image_filename || a.analysis_id.slice(0, 8)}
                  </span>
                  <span className="lot__rowseeds mono">{a.seed_count ?? 0} seeds</span>
                </label>
              ))}
            </div>

            <div className="lot__run">
              <span className="muted">
                {selected.size} of {analyses.length} images selected
              </span>
              <Button
                icon={ClipboardList}
                onClick={run}
                disabled={!selected.size}
                loading={report.state === "loading"}
              >
                Generate report
              </Button>
            </div>
          </GlassCard>

          {report.state === "loading" && (
            <GlassCard>
              <Skeleton height="220px" />
            </GlassCard>
          )}

          {report.state === "error" && (
            <GlassCard accent="danger">
              <ErrorState title="Could not build the report" reason={report.message} />
            </GlassCard>
          )}

          {report.state === "ready" && d && (
            <motion.div
              className="lot__result"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35 }}
            >
              <GlassCard>
                <SectionHeader
                  title={d.lot_reference ? `Lot ${d.lot_reference}` : "Lot summary"}
                  level={3}
                  right={<Badge tone="neutral">not a certification</Badge>}
                />
                <div className="lot__kpis">
                  <div className="lot__kpi">
                    <span>Kernels assessed</span>
                    <strong className="mono">{d.kernels_assessed}</strong>
                    <small>across {d.images_analysed} images</small>
                  </div>
                  <div className="lot__kpi">
                    <span>
                      {d.reference_is_declared ? "Matches declared variety" : "Majority variety share"}
                    </span>
                    <strong className="mono lot__kpi--gold">{d.purity_percent}%</strong>
                    <small>{pretty(d.reference_variety)}</small>
                  </div>
                  <div className="lot__kpi">
                    <span>Off-type</span>
                    <strong className="mono">{d.off_type_percent}%</strong>
                    <small>{d.off_type_count} kernels</small>
                  </div>
                  <div className="lot__kpi">
                    <span>Low confidence</span>
                    <strong className="mono">{d.low_confidence_percent}%</strong>
                    <small>below {Math.round(d.low_confidence_threshold * 100)}%</small>
                  </div>
                </div>
              </GlassCard>

              <GlassCard>
                <SectionHeader title="Variety composition" level={3} />
                <div className="lot__bars">
                  {d.composition.map((c) => (
                    <div className="lot__bar" key={c.variety}>
                      <div className="lot__barhead">
                        <span>{pretty(c.variety)}</span>
                        <span className="mono">
                          {c.percent}% · {c.count}
                        </span>
                      </div>
                      <div className="lot__bartrack">
                        <div className="lot__barfill" style={{ width: `${c.percent}%` }} />
                      </div>
                      <small className="faint">
                        mean confidence {(c.mean_confidence * 100).toFixed(1)}%
                      </small>
                    </div>
                  ))}
                </div>
                {d.evenness !== null && (
                  <p className="lot__evenness muted">
                    Evenness {d.evenness} — 0 means a single variety dominates entirely,
                    1 means the varieties are present in equal proportion.
                  </p>
                )}
              </GlassCard>

              <GlassCard>
                <SectionHeader title="Soundness" level={3} />
                {d.soundness.available ? (
                  <div className="lot__kpis">
                    <div className="lot__kpi">
                      <span>Sound kernels</span>
                      <strong className="mono lot__kpi--gold">{d.soundness.sound_percent}%</strong>
                      <small>of {d.soundness.kernels_graded} graded</small>
                    </div>
                    {Object.entries(d.soundness.distribution || {}).map(([k, v]) => (
                      <div className="lot__kpi" key={k}>
                        <span>{pretty(k)}</span>
                        <strong className="mono">{v}</strong>
                        <small>kernels</small>
                      </div>
                    ))}
                  </div>
                ) : (
                  <EmptyState
                    icon={Info}
                    title="No soundness rate available"
                    message="None of these analyses carry a verified quality grade, so no soundness percentage is shown rather than one being estimated."
                  />
                )}
              </GlassCard>

              <GlassCard accent="warn">
                <SectionHeader title="What this report does not tell you" level={3} />
                <ul className="lot__caveats">
                  {d.caveats.map((c, i) => (
                    <li key={i}>{c}</li>
                  ))}
                </ul>
              </GlassCard>
            </motion.div>
          )}
        </>
      )}
    </Page>
  );
}
