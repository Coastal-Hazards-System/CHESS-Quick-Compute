/* CHESS-QC web, applies WaveMaker theming via html[data-theme] (Light|Dark) and
 * html[data-style] (vibe: Signal|Trail|Forge). Decimals, vibe and the default mode are
 * chosen in the launcher (run_chess_quick_compute.py), which writes _prefs.js as
 *   window.CHESSQC_PREFS = {decimals, theme, vibe, stamp}.
 * The landing page also has a Light/Dark switch: it persists to localStorage and wins
 * for the rest of that launch; a new launch (new `stamp`) resets to the launcher default.
 * Loaded in <head> right after _prefs.js so the attributes are set before first paint. */
"use strict";
(function () {
  const MODES = ["Light", "Dark"];
  const VIBES = ["Signal", "Trail", "Forge"];
  const LS_MODE = "chessqc_mode", LS_STAMP = "chessqc_stamp";
  const prefs = () => window.CHESSQC_PREFS || {};

  const currentVibe = () => (VIBES.includes(prefs().vibe) ? prefs().vibe : "Signal");
  const currentBadge = () => (["Tinted", "Solid"].includes(prefs().badge) ? prefs().badge : "Tinted");

  const currentMode = () => {
    const p = prefs();
    if (String(p.stamp) !== localStorage.getItem(LS_STAMP)) {
      localStorage.setItem(LS_STAMP, String(p.stamp));   // new launch -> launcher default wins
      localStorage.removeItem(LS_MODE);
    }
    const m = localStorage.getItem(LS_MODE);
    if (MODES.includes(m)) return m;
    return MODES.includes(p.theme) ? p.theme : "Light";
  };

  const apply = () => {
    const el = document.documentElement;
    el.dataset.theme = currentMode().toLowerCase();
    el.dataset.style = currentVibe().toLowerCase();
    el.dataset.badge = currentBadge().toLowerCase();
  };
  apply();   // run immediately (before paint)

  const getDecimals = () => {
    const n = prefs().decimals;
    return Number.isFinite(n) ? Math.min(Math.max(n, 0), 8) : 2;
  };
  const toggleMode = () => {
    const next = currentMode() === "Dark" ? "Light" : "Dark";
    localStorage.setItem(LS_MODE, next);   // persists for the rest of this launch
    apply();
    return next;
  };

  window.CHESSQCUI = { getDecimals, currentMode, currentVibe, toggleMode, apply };
})();
