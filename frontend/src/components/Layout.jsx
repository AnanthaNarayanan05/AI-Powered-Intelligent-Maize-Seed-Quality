import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
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
  Menu,
  X,
} from "lucide-react";
import { api } from "../api/client";
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
];

export default function Layout() {
  const location = useLocation();
  const [health, setHealth] = useState({ state: "loading" });
  const [navOpen, setNavOpen] = useState(false);

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

  const backendTone = health.state === "ok" ? "ok" : health.state === "loading" ? "idle" : "error";
  const backendLabel =
    health.state === "ok" ? "Online" : health.state === "loading" ? "Checking" : "Offline";
  const geminiTone = health.gemini_configured ? "ai" : "warn";
  const geminiLabel = health.gemini_configured ? "Connected" : "Not configured";

  return (
    <div className="shell">
      <div className="bg-field" aria-hidden="true" />

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
                  <stop stopColor="#F7D97A" />
                  <stop offset="1" stopColor="#C9A233" />
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
            Kernel quality is graded by a model trained on{" "}
            <strong>expert-assigned Good/Bad labels</strong> (97.3% test accuracy).
            Grades on imagery unlike its training set are marked{" "}
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
