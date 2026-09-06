import { useEffect, useState } from "react";

/* CSS transitions are handled by the token sheet; this hook covers the motion CSS
   can't reach — the background video, framer-motion, the 3D hero. It reads the
   `data-motion` attribute off <html> rather than the settings context so that
   every existing call site keeps working unchanged and no component that only
   wants to know "should I animate?" has to sit inside a provider. */
function isReduced() {
  if (typeof window === "undefined") return false;
  return (
    document.documentElement.dataset.motion === "reduced" ||
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/** Tracks the OS `prefers-reduced-motion` setting and the in-app override, live. */
export function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(isReduced);

  useEffect(() => {
    const sync = () => setReduced(isReduced());

    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    mq.addEventListener("change", sync);

    const observer = new MutationObserver(sync);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-motion"],
    });

    // The attribute is written by an effect that may land after this one.
    sync();

    return () => {
      mq.removeEventListener("change", sync);
      observer.disconnect();
    };
  }, []);

  return reduced;
}
