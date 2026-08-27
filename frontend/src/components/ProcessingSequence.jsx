/* ============================================================
   Pipeline activity display shown while an analysis request is in flight.

   HONESTY NOTE: the backend performs the whole analysis in a single request and
   reports no per-stage progress, so this component deliberately shows NO
   percentage and never marks an individual stage "complete" while the request is
   still running. It lists the stages the pipeline actually executes, highlights
   that work is underway with an indeterminate animation, and only marks
   everything done once the response has genuinely arrived. Inventing "72%" here
   would be fabricating telemetry the system does not produce.
   ============================================================ */
import { motion } from "framer-motion";
import { Check, Loader2 } from "lucide-react";
import "./processing.css";

const STAGES = [
  "Initializing vision engine",
  "Detecting seeds",
  "Cropping individual seeds",
  "Classifying variety",
  "Classifying defect pattern",
  "Generating feature embedding",
  "Searching similarity index",
];

export default function ProcessingSequence({ done = false }) {
  return (
    <div className="proc">
      <div className="proc__head">
        {done ? (
          <span className="proc__badge proc__badge--done">
            <Check size={13} /> Analysis complete
          </span>
        ) : (
          <span className="proc__badge">
            <Loader2 size={13} className="proc__spin" /> Pipeline running
          </span>
        )}
        {!done && (
          <span className="proc__note">
            The backend returns one response for the whole pipeline, so no per-stage progress
            is reported.
          </span>
        )}
      </div>

      {!done && (
        <div className="proc__indeterminate" role="progressbar" aria-label="Analysis in progress">
          <span className="proc__indeterminate-bar" />
        </div>
      )}

      <ul className="proc__list">
        {STAGES.map((s, i) => (
          <motion.li
            key={s}
            className={`proc__item ${done ? "is-done" : "is-pending"}`}
            initial={{ opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.04 * i }}
          >
            <span className="proc__dot">{done && <Check size={11} />}</span>
            <span>{s}</span>
          </motion.li>
        ))}
      </ul>
    </div>
  );
}
