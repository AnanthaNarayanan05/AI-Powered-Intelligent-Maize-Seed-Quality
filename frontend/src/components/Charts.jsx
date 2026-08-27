/* ============================================================
   Small SVG charts.

   Hand-rolled rather than pulling a charting library: the project needs exactly
   two chart types, and a general-purpose library would add hundreds of KB to the
   bundle for features nothing here uses.

   Both render an explicit empty state rather than an empty axis when given no
   data, so a chart never implies "zero" when the truth is "nothing recorded yet".
   ============================================================ */
import { motion } from "framer-motion";
import "./charts.css";

const SERIES = ["#F4C542", "#6FBF73", "#48D6C6", "#E0A340", "#8FA8C8", "#C98A1B"];

export function DonutChart({ data, size = 168, thickness = 22, emptyMessage = "No data yet" }) {
  const entries = Object.entries(data || {}).filter(([, v]) => v > 0);
  const total = entries.reduce((s, [, v]) => s + v, 0);

  if (!total) return <p className="chart-empty">{emptyMessage}</p>;

  const r = (size - thickness) / 2;
  const c = 2 * Math.PI * r;
  let offset = 0;

  return (
    <div className="donut">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Distribution">
        <g transform={`rotate(-90 ${size / 2} ${size / 2})`}>
          {entries.map(([name, v], i) => {
            const frac = v / total;
            const dash = frac * c;
            const el = (
              <motion.circle
                key={name}
                cx={size / 2}
                cy={size / 2}
                r={r}
                fill="none"
                stroke={SERIES[i % SERIES.length]}
                strokeWidth={thickness}
                strokeDasharray={`${dash} ${c - dash}`}
                strokeDashoffset={-offset}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.08 * i }}
              />
            );
            offset += dash;
            return el;
          })}
        </g>
        <text x="50%" y="47%" textAnchor="middle" className="donut__total">
          {total}
        </text>
        <text x="50%" y="60%" textAnchor="middle" className="donut__cap">
          total
        </text>
      </svg>

      <ul className="donut__legend">
        {entries.map(([name, v], i) => (
          <li key={name}>
            <span className="donut__swatch" style={{ background: SERIES[i % SERIES.length] }} />
            <span className="donut__name">{name.replace(/_/g, " ")}</span>
            <span className="donut__val mono">
              {v} · {((v / total) * 100).toFixed(0)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Histogram of confidence values bucketed into ten 10% bins. */
export function ConfidenceHistogram({ values, emptyMessage = "No confidence values yet" }) {
  const vals = (values || []).filter((v) => typeof v === "number");
  if (!vals.length) return <p className="chart-empty">{emptyMessage}</p>;

  const bins = Array.from({ length: 10 }, () => 0);
  vals.forEach((v) => {
    const idx = Math.min(9, Math.floor(v * 10));
    bins[idx] += 1;
  });
  const max = Math.max(...bins);

  return (
    <div className="histo" role="img" aria-label="Confidence distribution">
      <div className="histo__bars">
        {bins.map((n, i) => (
          <div className="histo__col" key={i}>
            <motion.span
              className={`histo__bar ${i >= 8 ? "is-high" : i >= 6 ? "is-mid" : "is-low"}`}
              initial={{ height: 0 }}
              animate={{ height: max ? `${(n / max) * 100}%` : 0 }}
              transition={{ duration: 0.5, delay: 0.03 * i, ease: [0.22, 1, 0.36, 1] }}
              title={`${i * 10}–${i * 10 + 10}%: ${n}`}
            />
          </div>
        ))}
      </div>
      <div className="histo__axis">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </div>
  );
}
