import { useEffect, useRef, useState } from "react";
import { NavLink, Navigate, Outlet, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  LayoutGrid,
  ScanSearch,
  Layers,
  History as HistoryIcon,
  GitCompare,
  ClipboardList,
  Sparkles,
  Cpu,
  SlidersHorizontal,
  Menu,
  X,
} from "lucide-react";
import { api } from "../api/client";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";
import { useSettings } from "../lib/settings";
import { StatusPill } from "./ui";
import "./layout.css";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutGrid, end: true },
  { to: "/analyze", label: "Analyze", icon: ScanSearch },
  { to: "/batch", label: "Batch", icon: Layers },
  { to: "/history", label: "History", icon: HistoryIcon },
  { to: "/compare", label: "Compare", icon: GitCompare },
  { to: "/lot", label: "Seed Lot", icon: ClipboardList },
  { to: "/copilot", label: "AI Copilot", icon: Sparkles },
  { to: "/system", label: "System", icon: Cpu },
  { to: "/settings", label: "Settings", icon: SlidersHorizontal },
];

export default function Layout() {
  const location = useLocation();
  const [health, setHealth] = useState({ state: "loading" });
  const [navOpen, setNavOpen] = useState(false);
  const prefersReducedMotion = usePrefersReducedMotion();
  const [videoFailed, setVideoFailed] = useState(false);
  const videoRef = useRef(null);
  const isOverview = location.pathname === "/";

  // The landing-page preference applies to opening the app, not to navigating
  // inside it: clicking "Overview" afterwards must reach Overview, or the nav
  // item is unreachable. So it is decided once, from the URL the browser was
  // actually opened at, and cleared on mount. It is state rather than a ref
  // written during render — a ref mutated mid-render is undone by StrictMode's
  // second pass, which silently swallowed the redirect.
  const { settings } = useSettings();
  const [landingPending, setLandingPending] = useState(
    () => settings.landingPage !== "/" && window.location.pathname === "/"
  );
  useEffect(() => setLandingPending(false), []);
  const redirectTo = landingPending ? settings.landingPage : null;

  useEffect(() => {
    let cancelled = false;
    api
      .health()
      .then((h) => !cancelled && setHealth({ state: "ok", ...h }))
      .catch(() => !cancelled && setHealth({ state: "error" }));
    return () => {
      cancelled = true;
    };
  }, []);

  // Close the mobile drawer on navigation, otherwise it covers the new page.
  useEffect(() => setNavOpen(false), [location.pathname]);

  // The Overview background video mounts only on "/", so this re-runs each time
  // it (re)appears. autoplay alone is unreliable across browsers/policies; retry
  // play() explicitly, then once more on the first interaction anywhere as a
  // last-resort fallback for a stricter autoplay policy.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || videoFailed || !isOverview) return;
    if (prefersReducedMotion) {
      v.pause();
      return;
    }
    const tryPlay = () => v.play().catch(() => {});
    tryPlay();
    document.addEventListener("pointerdown", tryPlay, { once: true });
    return () => document.removeEventListener("pointerdown", tryPlay);
  }, [prefersReducedMotion, videoFailed, isOverview]);

  const backendTone = health.state === "ok" ? "ok" : health.state === "loading" ? "idle" : "error";
  const backendLabel =
    health.state === "ok" ? "Online" : health.state === "loading" ? "Checking" : "Offline";
  const geminiTone = health.gemini_configured ? "ai" : "warn";
  const geminiLabel = health.gemini_configured ? "Connected" : "Not configured";

  if (redirectTo) return <Navigate to={redirectTo} replace />;

  return (
    <div className="shell">
      {/* The real golden-hour field is the Overview page's background, not a boxed
         hero element — it plays fixed and full-bleed behind the glass shell, with
         a dark overlay + vignette layered over it for text contrast. Every other
         page gets the same treatment over the static Background.png instead (see
         bg-field rules in index.css) — one visual language, two source images. A
         missing/failed video falls back to the same still poster the app already
         uses for its first paint. */}
      <div className={`bg-field ${isOverview ? "bg-field--video" : ""}`} aria-hidden="true">
        {isOverview && !videoFailed && (
          <video
            ref={videoRef}
            className="bg-field__video"
            autoPlay={!prefersReducedMotion}
            muted
            loop
            playsInline
            poster="/maize-field-golden-hour-poster.jpg"
            onError={() => setVideoFailed(true)}
          >
            <source src="/maize-field-golden-hour.mp4" type="video/mp4" />
          </video>
        )}
      </div>

      <a href="#main" className="sr-only">
        Skip to main content
      </a>

      <aside className={`sidebar ${navOpen ? "is-open" : ""}`}>
        <div className="sidebar__brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none">
              <path
                d="M12 2c3.2 2.4 4.8 5.6 4.8 9.2 0 4.2-2.1 8-4.8 10.8-2.7-2.8-4.8-6.6-4.8-10.8C7.2 7.6 8.8 4.4 12 2Z"
                fill="url(#kernel)"
              />
              <path d="M12 4.6v15.2" stroke="rgba(36,29,5,.32)" strokeWidth="1.1" />
              <defs>
                <linearGradient id="kernel" x1="7" y1="2" x2="17" y2="22">
                  <stop stopColor="var(--gold-bright)" />
                  <stop offset="1" stopColor="var(--gold-dim)" />
                </linearGradient>
              </defs>
            </svg>
          </span>
          <div className="sidebar__brandtext">
            <strong>MAIZE</strong>
            <span>INTELLIGENCE</span>
          </div>
          <button
            className="sidebar__close"
            onClick={() => setNavOpen(false)}
            aria-label="Close navigation"
          >
            <X size={18} />
          </button>
        </div>

        <nav className="sidebar__nav" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) => `navitem ${isActive ? "is-active" : ""}`}
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <motion.span
                      layoutId="nav-active"
                      className="navitem__bg"
                      transition={{ type: "spring", stiffness: 380, damping: 32 }}
                    />
                  )}
                  <Icon size={17} aria-hidden="true" />
                  <span>{label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar__foot">
          <p className="sidebar__note">
            {/* No metric is quoted here: this component only has /api/health, and a
                number copied into the chrome drifts the moment a model is retrained.
                The live figures live on the System information page. */}
            Kernel quality is graded by a model trained on{" "}
            <strong>expert-assigned Good/Bad labels</strong>. Grades on imagery unlike
            its training set are marked{" "}
            <strong>unverified</strong> rather than reported as defects.
          </p>
        </div>
      </aside>

      {navOpen && <div className="scrim" onClick={() => setNavOpen(false)} aria-hidden="true" />}

      <div className="shell__main">
        <header className="topbar">
          <button
            className="topbar__menu"
            onClick={() => setNavOpen(true)}
            aria-label="Open navigation"
          >
            <Menu size={19} />
          </button>

          <div className="topbar__title">
            <span className="topbar__eyebrow">AI-Powered Maize Seed Analysis Platform</span>
            <span className="topbar__desc">
              Detection · Variety Recognition · Quality Analysis · Explainable AI
            </span>
          </div>

          <div className="topbar__status">
            <StatusPill label="Backend" value={backendLabel} tone={backendTone} pulse={health.state === "ok"} />
            <StatusPill label="Gemini" value={geminiLabel} tone={geminiTone} pulse={!!health.gemini_configured} />
          </div>
        </header>

        <main id="main" className="content">
          <AnimatePresence mode="wait">
            <div key={location.pathname}>
              <Outlet />
            </div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
