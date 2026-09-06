/* User preferences.
 *
 * Every field in DEFAULTS is read by real application code — the Settings page
 * lists where each one lands. Preferences the platform cannot genuinely honour
 * are deliberately absent rather than present and inert: there is no language
 * option because the app ships one language, no theme option because there is
 * one theme, and no notification options because nothing in this project emits
 * a notification.
 *
 * Storage is localStorage. That is the whole persistence layer: these are
 * per-browser display and request preferences, not account state, and the
 * backend has no user model to attach them to.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

const STORAGE_KEY = "maize-intelligence.settings.v1";

export const DEFAULTS = Object.freeze({
  // General
  landingPage: "/",
  // Analysis — both are sent to /api/analyze, which already accepts them.
  //
  // run_synthetic_defect is deliberately NOT here even though the endpoint takes
  // it: the synthetic classifier declares serves: [] in the registry, the Analyze
  // page renders none of its output, and tests/test_api_endpoints.py asserts it is
  // never advertised as a capability. A switch for it would change a request and
  // show the operator nothing.
  defaultModel: "unified",
  runSimilarity: true,
  // Appearance — these drive real token overrides on the document element.
  glass: "balanced",
  density: "comfortable",
  motion: "system",
});

const ALLOWED = {
  landingPage: ["/", "/analyze", "/batch", "/history", "/compare", "/lot", "/copilot", "/system"],
  defaultModel: ["unified", "a", "b"],
  glass: ["subtle", "balanced", "strong"],
  density: ["comfortable", "compact"],
  motion: ["system", "reduced"],
};

/* A stored value that is no longer offered — a route that was renamed, an option
   that was withdrawn — must not quietly configure the app into a state its own UI
   cannot represent. Anything unrecognised falls back to the default. */
function coerce(raw) {
  const out = { ...DEFAULTS };
  if (!raw || typeof raw !== "object") return out;
  for (const key of Object.keys(DEFAULTS)) {
    const value = raw[key];
    if (value === undefined) continue;
    if (typeof DEFAULTS[key] === "boolean") {
      if (typeof value === "boolean") out[key] = value;
    } else if (ALLOWED[key]?.includes(value)) {
      out[key] = value;
    }
  }
  return out;
}

/* Reading can throw outright — Safari private mode, storage disabled by policy —
   so a failure here yields defaults rather than an unmounted app. */
export function readSettings() {
  try {
    return coerce(JSON.parse(window.localStorage.getItem(STORAGE_KEY)));
  } catch {
    return { ...DEFAULTS };
  }
}

const SettingsContext = createContext(null);

export function SettingsProvider({ children }) {
  const [settings, setSettings] = useState(readSettings);

  /* Applied to the document element rather than a React tree, because the tokens
     these swap are consumed by every stylesheet in the app, including chrome that
     renders outside the router. */
  useEffect(() => {
    const root = document.documentElement;
    root.dataset.glass = settings.glass;
    root.dataset.density = settings.density;
    root.dataset.motion = settings.motion;
  }, [settings.glass, settings.density, settings.motion]);

  /* Returns the outcome instead of assuming one: the Settings page must not
     report a successful save that did not happen. */
  const save = useCallback((draft) => {
    const next = coerce(draft);
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch (e) {
      return { ok: false, error: e?.message || "This browser refused to store preferences." };
    }
    setSettings(next);
    return { ok: true };
  }, []);

  const reset = useCallback(() => {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      return { ok: false, error: e?.message || "This browser refused to clear stored preferences." };
    }
    setSettings({ ...DEFAULTS });
    return { ok: true };
  }, []);

  const value = useMemo(() => ({ settings, save, reset }), [settings, save, reset]);
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used inside a SettingsProvider");
  return ctx;
}
