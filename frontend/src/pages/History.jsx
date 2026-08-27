import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Search, ChevronLeft, ChevronRight, Clock, X } from "lucide-react";
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

export default function History() {
  const [rows, setRows] = useState([]);
  const [state, setState] = useState("loading");
  const [error, setError] = useState(null);
  const [page, setPage] = useState(0);
  const [query, setQuery] = useState("");
  const [variety, setVariety] = useState("all");
  const [selected, setSelected] = useState(null);

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
        .filter((c) => !c.is_synthetic_model)
        .forEach((c) => set.add(c.predicted_class))
    );
    return [...set].sort();
  }, [rows]);

  const filtered = rows.filter((r) => {
    const top = (r.classifications || []).find((c) => !c.is_synthetic_model);
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

      <GlassCard className="hi__filters">
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
            const top = (r.classifications || []).find((c) => !c.is_synthetic_model);
            const open = selected === r.analysis_id;
            return (
              <motion.div
                key={r.analysis_id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.03 * i }}
              >
                <GlassCard
                  hover
                  className={`hcard ${open ? "is-open" : ""}`}
                  onClick={() => setSelected(open ? null : r.analysis_id)}
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
                    >
                      <span className="hcard__dl">Dataset {r.variety_dataset_used?.toUpperCase()}</span>
                      {(r.classifications || []).map((c, k) => (
                        <div className="hcard__cls" key={k}>
                          <span className="hcard__clsname">{pretty(c.predicted_class)}</span>
                          <ProvenanceBadge synthetic={!!c.is_synthetic_model} />
                        </div>
                      ))}
                      <span className="hcard__id mono faint">{r.analysis_id}</span>
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
