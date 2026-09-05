import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Search, ChevronLeft, ChevronRight, Clock, X } from "lucide-react";
import { api, mediaUrl } from "../api/client";
import { foreignFlag, isQualityRow, isVarietyRow, seedHealth } from "../lib/seedHealth";
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
import DetectionCanvas from "../components/DetectionCanvas";
import "./history.css";

const PAGE_SIZE = 12;
const pretty = (s) => (s ? s.replace(/_/g, " ") : "—");

const fmtDate = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

// A stored analysis keeps detections, classifications and assessments as separate
// flat rows (one write per seed, per table). DetectionCanvas expects the shape a
// live analysis returns instead — one object per seed with its own predictions
// nested inline — so this rebuilds that shape by seed_index. No extra fetch: every
// field here already came back with the row from GET /api/history.
function buildDetectionSeeds(analysis) {
  const classifications = analysis?.classifications || [];
  const assessments = analysis?.assessments || [];
  return (analysis?.detections || []).map((d) => {
    const variety = classifications.find((c) => c.seed_index === d.seed_index && isVarietyRow(c));
    const quality = classifications.find((c) => c.seed_index === d.seed_index && isQualityRow(c));
    const assessment = assessments.find((a) => a.seed_index === d.seed_index);
    return {
      seed_index: d.seed_index,
      bbox: d.bbox,
      kernel_px: d.kernel_px,
      detection_confidence: d.confidence,
      variety_prediction: variety
        ? { predicted_class: variety.predicted_class, confidence: variety.confidence }
        : null,
      quality_prediction: quality
        ? {
            predicted_class: quality.predicted_class,
            confidence: quality.confidence,
            out_of_distribution: !!quality.class_probabilities?._out_of_distribution,
          }
        : null,
      foreign_object: assessment?.foreign_object_status
        ? { status: assessment.foreign_object_status, basis: assessment.foreign_object_basis }
        : null,
    };
  });
}

function SeedDetail({ seed }) {
  if (!seed) return null;
  const foreign = foreignFlag(seed);
  if (foreign) {
    return (
      <div className="hcard__seeddetail">
        <span className="hcard__seeddetaillabel">Seed {seed.seed_index + 1}</span>
        <span className="hcard__seedforeign">
          {foreign.label} — {foreign.reason}
        </span>
      </div>
    );
  }
  const health = seedHealth(seed);
  return (
    <div className="hcard__seeddetail">
      <span className="hcard__seeddetaillabel">Seed {seed.seed_index + 1}</span>
      <div className="hcard__seedrow">
        <span className="hcard__seedval">
          {seed.variety_prediction ? pretty(seed.variety_prediction.predicted_class) : "No variety prediction"}
        </span>
        {seed.variety_prediction && (
          <ConfidenceBar value={seed.variety_prediction.confidence} showValue label={null} />
        )}
      </div>
      {seed.quality_prediction && (
        <div className="hcard__seedrow">
          <span className={`hcard__seedval ${health.flagged ? "hcard__seedval--warn" : ""}`}>
            {pretty(seed.quality_prediction.predicted_class)}
          </span>
          <ConfidenceBar value={seed.quality_prediction.confidence} showValue label={null} />
        </div>
      )}
    </div>
  );
}

export default function History() {
  const [rows, setRows] = useState([]);
  const [state, setState] = useState("loading");
  const [error, setError] = useState(null);
  const [page, setPage] = useState(0);
  const [query, setQuery] = useState("");
  const [variety, setVariety] = useState("all");
  const [selected, setSelected] = useState(null);
  const [seedSel, setSeedSel] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    // Paginated server-side so a large history is never pulled into the browser.
    api
      .history(PAGE_SIZE, page * PAGE_SIZE)
      .then((d) => {
        if (cancelled) return;
        setRows(d.analyses || []);
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
  }, [page]);

  const varieties = useMemo(() => {
    const set = new Set();
    rows.forEach((r) =>
      (r.classifications || [])
        .filter(isVarietyRow)
        .forEach((c) => set.add(c.predicted_class))
    );
    return [...set].sort();
  }, [rows]);

  const filtered = rows.filter((r) => {
    const top = (r.classifications || []).find(isVarietyRow);
    if (variety !== "all" && top?.predicted_class !== variety) return false;
    if (query) {
      const hay = `${r.image_filename || ""} ${top?.predicted_class || ""}`.toLowerCase();
      if (!hay.includes(query.toLowerCase())) return false;
    }
    return true;
  });

  return (
    <Page className="hi">
      <SectionHeader
        title="Analysis history"
        subtitle="Every analysis is persisted with its detections, class probabilities and the model that produced them."
      />

      <GlassCard tier="floating" className="hi__filters">
        <div className="hi__search">
          <Search size={15} />
          <input
            type="text"
            placeholder="Search filename or variety…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search analyses"
          />
          {query && (
            <button onClick={() => setQuery("")} aria-label="Clear search">
              <X size={14} />
            </button>
          )}
        </div>

        <select
          className="hi__select"
          value={variety}
          onChange={(e) => setVariety(e.target.value)}
          aria-label="Filter by variety"
        >
          <option value="all">All varieties</option>
          {varieties.map((v) => (
            <option key={v} value={v}>
              {pretty(v)}
            </option>
          ))}
        </select>

        <div className="hi__pager">
          <Button
            variant="ghost"
            size="sm"
            icon={ChevronLeft}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0 || state === "loading"}
          >
            Prev
          </Button>
          <span className="hi__page mono">Page {page + 1}</span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPage((p) => p + 1)}
            disabled={rows.length < PAGE_SIZE || state === "loading"}
          >
            Next <ChevronRight size={14} />
          </Button>
        </div>
      </GlassCard>

      {state === "loading" && (
        <div className="hi__grid">
          {Array.from({ length: 6 }).map((_, i) => (
            <GlassCard key={i} className="hi__skel">
              <Skeleton height="120px" radius="var(--r-md)" />
              <Skeleton height="12px" width="70%" />
              <Skeleton height="12px" width="45%" />
            </GlassCard>
          ))}
        </div>
      )}

      {state === "error" && (
        <GlassCard accent="danger">
          <ErrorState title="History unavailable" reason={error} />
        </GlassCard>
      )}

      {state === "ready" && filtered.length === 0 && (
        <GlassCard>
          <EmptyState
            icon={Clock}
            title={rows.length ? "No analyses match these filters" : "Your analysis history will appear here"}
            message={
              rows.length
                ? "Try clearing the search or variety filter."
                : "Run an analysis from the Analyze page and it will be saved here automatically."
            }
          />
        </GlassCard>
      )}

      {state === "ready" && filtered.length > 0 && (
        <div className="hi__grid">
          {filtered.map((r, i) => {
            const top = (r.classifications || []).find(isVarietyRow);
            const open = selected === r.analysis_id;
            const detSeeds = open ? buildDetectionSeeds(r) : [];
            return (
              <motion.div
                key={r.analysis_id}
                className={`hi__cell ${open ? "is-open" : ""}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.03 * i }}
              >
                <GlassCard
                  hover
                  className={`hcard ${open ? "is-open" : ""}`}
                  onClick={() => {
                    setSelected(open ? null : r.analysis_id);
                    setSeedSel(null);
                  }}
                >
                  <div className="hcard__img">
                    {r.image_path ? (
                      <img
                        src={mediaUrl(r.image_path, 300)}
                        alt={r.image_filename || "Analysed image"}
                        loading="lazy"
                      />
                    ) : (
                      <span className="hcard__noimg">No preview stored</span>
                    )}
                    <span className="hcard__seeds mono">{r.seed_count}</span>
                  </div>

                  <div className="hcard__head">
                    <span className="hcard__name" title={r.image_filename}>
                      {r.image_filename || r.analysis_id.slice(0, 8)}
                    </span>
                    <Badge tone={r.status === "completed" ? "ok" : "warn"}>{r.status}</Badge>
                  </div>

                  <span className="hcard__date mono faint">{fmtDate(r.created_at)}</span>

                  {top ? (
                    <>
                      <span className="hcard__variety">{pretty(top.predicted_class)}</span>
                      <ConfidenceBar value={top.confidence} showValue label={null} />
                    </>
                  ) : (
                    <span className="hcard__variety faint">No variety prediction</span>
                  )}

                  {open && (
                    <motion.div
                      className="hcard__detail"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.25 }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <span className="hcard__dl">Dataset {r.variety_dataset_used?.toUpperCase()}</span>
                      {(r.classifications || []).map((c, k) => (
                        <div className="hcard__cls" key={k}>
                          <span className="hcard__clsname">{pretty(c.predicted_class)}</span>
                          <ProvenanceBadge synthetic={!!c.is_synthetic_model} />
                        </div>
                      ))}
                      <span className="hcard__id mono faint">{r.analysis_id}</span>

                      {detSeeds.length > 0 && r.image_path && (
                        <div className="hcard__seeds-drilldown">
                          <span className="hcard__dl">Seed detections</span>
                          <DetectionCanvas
                            src={mediaUrl(r.image_path)}
                            seeds={detSeeds}
                            selectedIndex={seedSel}
                            onSelect={setSeedSel}
                          />
                          {seedSel != null && <SeedDetail seed={detSeeds[seedSel]} />}
                        </div>
                      )}
                    </motion.div>
                  )}
                </GlassCard>
              </motion.div>
            );
          })}
        </div>
      )}
    </Page>
  );
}
