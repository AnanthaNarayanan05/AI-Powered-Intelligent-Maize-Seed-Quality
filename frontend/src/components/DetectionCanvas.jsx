/* ============================================================
   Draws YOLO bounding boxes over the analysed image.

   Boxes arrive in the source image's pixel space, so they are positioned as
   percentages of the natural dimensions. That keeps them correct at any rendered
   size without recomputing on resize, and avoids animating layout properties.
   ============================================================ */
import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Eye, EyeOff, Maximize2, ZoomIn, ZoomOut } from "lucide-react";
import { seedHealth, foreignFlag } from "../lib/seedHealth";
import "./detectioncanvas.css";

export default function DetectionCanvas({
  src,
  seeds = [],
  selectedIndex,
  onSelect,
  alt = "Analysed seed image",
}) {
  const [natural, setNatural] = useState(null);
  const [showBoxes, setShowBoxes] = useState(true);
  const [zoom, setZoom] = useState(1);
  const imgRef = useRef(null);

  // If the image is cached the load event can fire before this mounts.
  useEffect(() => {
    const el = imgRef.current;
    if (el?.complete && el.naturalWidth) {
      setNatural({ w: el.naturalWidth, h: el.naturalHeight });
    }
  }, [src]);

  const pct = (seed) => {
    if (!natural) return null;
    const [x1, y1, x2, y2] = seed.bbox;
    return {
      left: `${(x1 / natural.w) * 100}%`,
      top: `${(y1 / natural.h) * 100}%`,
      width: `${((x2 - x1) / natural.w) * 100}%`,
      height: `${((y2 - y1) / natural.h) * 100}%`,
    };
  };

  return (
    <div className="dcanvas">
      <div className="dcanvas__toolbar">
        <button
          className={`dcanvas__tool ${showBoxes ? "is-on" : ""}`}
          onClick={() => setShowBoxes((v) => !v)}
          aria-pressed={showBoxes}
        >
          {showBoxes ? <Eye size={14} /> : <EyeOff size={14} />}
          <span>{showBoxes ? "Boxes on" : "Boxes off"}</span>
        </button>

        <div className="dcanvas__spacer" />

        <button
          className="dcanvas__tool"
          onClick={() => setZoom((z) => Math.max(1, +(z - 0.25).toFixed(2)))}
          aria-label="Zoom out"
          disabled={zoom <= 1}
        >
          <ZoomOut size={14} />
        </button>
        <span className="dcanvas__zoom mono">{Math.round(zoom * 100)}%</span>
        <button
          className="dcanvas__tool"
          onClick={() => setZoom((z) => Math.min(3, +(z + 0.25).toFixed(2)))}
          aria-label="Zoom in"
          disabled={zoom >= 3}
        >
          <ZoomIn size={14} />
        </button>
        <button className="dcanvas__tool" onClick={() => setZoom(1)} aria-label="Reset view">
          <Maximize2 size={14} />
        </button>
      </div>

      <div className="dcanvas__viewport">
        <div
          className="dcanvas__stage"
          style={{ transform: `scale(${zoom})` }}
        >
          <img
            ref={imgRef}
            src={src}
            alt={alt}
            className="dcanvas__img"
            onLoad={(e) =>
              setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })
            }
          />

          {showBoxes &&
            natural &&
            seeds.map((seed, i) => {
              const box = pct(seed);
              if (!box) return null;
              const active = selectedIndex === i;
              const health = seedHealth(seed);
              // A possible foreign object supersedes the health marker: grading
              // the soundness of something that may not be a kernel is not a
              // result, and showing both invites the reader to merge them.
              const foreign = foreignFlag(seed);
              const mark = foreign
                ? { flagged: true, verified: false, reason: foreign.reason }
                : health;
              return (
                <motion.button
                  key={seed.seed_index ?? i}
                  className={`dbox ${active ? "is-active" : ""} ${
                    foreign
                      ? "is-foreign"
                      : health.flagged
                        ? health.verified ? "is-flagged" : "is-unverified"
                        : ""
                  }`}
                  style={box}
                  initial={{ opacity: 0, scale: 1.14 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{
                    duration: 0.4,
                    delay: 0.06 * i,
                    ease: [0.22, 1, 0.36, 1],
                  }}
                  onClick={() => onSelect?.(active ? null : i)}
                  title={
                    foreign
                      ? `Object ${i + 1} — ${foreign.label}. ${foreign.reason}`
                      : mark.flagged
                        ? `Seed ${i + 1} — ${mark.reason}`
                        : `Seed ${i + 1}`
                  }
                  aria-label={
                    foreign
                      ? `Object ${i + 1}, ${foreign.label}, not identified`
                      : mark.flagged
                        ? `Seed ${i + 1}, flagged: ${mark.reason}`
                        : `Seed ${i + 1}`
                  }
                >
                  {mark.flagged && (
                    <span
                      className={`dbox__flag ${
                        foreign
                          ? "dbox__flag--foreign"
                          : mark.verified ? "" : "dbox__flag--unverified"
                      }`}
                      aria-hidden="true"
                    />
                  )}
                </motion.button>
              );
            })}
        </div>
      </div>
    </div>
  );
}
