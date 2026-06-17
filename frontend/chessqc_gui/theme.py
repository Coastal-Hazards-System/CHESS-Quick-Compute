"""Theme system for the CHESS-QC desktop GUI.

Uses the authoritative "WaveMaker Qt QSS" deliverable sheets (chessqc_gui/qss/), one per
palette-mode-vibe. Per the design, the two user-facing modes map to:
    Light  ->  vibrant-light   (Vibrant palette)
    Dark   ->  original-dark   (Original palette)
each x three vibes (Signal / Trail / Forge). `load_qss(mode, vibe)` reads the sheet,
resolves its relative icon URLs to this package's absolute /icons folder, and appends
a few overrides for app-specific object names the stock sheet doesn't define.
"""
from __future__ import annotations

import os

# user mode -> the palette-mode QSS prefix
THEMES: dict[str, str] = {"Light": "vibrant-light", "Dark": "original-dark"}
DEFAULT_THEME = next(iter(THEMES))

# vibe -> label rendering hints (Qt QSS can't do text-transform/letter-spacing, so
# the Forge uppercase-mono labels are applied in app_shell via vibe_label_opts).
VIBES: dict[str, dict] = {
    "Signal": {"label_upper": False, "mono_labels": False},
    "Trail":  {"label_upper": False, "mono_labels": False},
    "Forge":  {"label_upper": True,  "mono_labels": True},
}
DEFAULT_VIBE = next(iter(VIBES))

# matplotlib plot colors per mode. Light = Vibrant, Dark = Original.
# Line colors = the mode's brand turquoise + a sand neutral.
PLOT_PALETTES: dict[str, dict] = {
    "Light": {"bg": "#ffffff", "fg": "#16211f", "grid": "#e2dcd0", "axis": "#cfc7b7", "text": "#7c888b",
              "eta": "#008cb0", "u": "#008cb0", "w": "#a86f2f"},
    "Dark":  {"bg": "#121d21", "fg": "#eaf2f3", "grid": "#243035", "axis": "#33444b", "text": "#6e8086",
              "eta": "#2bb8d4", "u": "#2bb8d4", "w": "#cda05f"},
}

# Fidelity-class badge accent (A exact, B standard, C provisional).
SEMANTIC: dict[str, dict] = {
    "Light": {"exact": "#16b048", "standard": "#db8c0a", "provisional": "#e5210a"},
    "Dark":  {"exact": "#9cb57a", "standard": "#dcc8a1", "provisional": "#e5654f"},
}

# Landing-page cards are colored by FUNCTIONAL AREA (distinct per area, consistent
# within), using the WaveMaker ramps. Light = Vibrant shades, Dark = Original shades.
# Keys must match APP_META.area exactly.
AREA_COLORS: dict[str, dict] = {
    "Light": {
        "Wave Prediction": "#008cb0", "Wave Theory": "#16b048", "Wave Transformation": "#f5ae1b",
        "Structural Design": "#8a5f38", "Wave Runup, Transmission, and Overtopping": "#e5210a",
        "Littoral Processes": "#00a8cc", "Inlet Processes": "#db8c0a", "Harbor Design": "#0e9138",
        "Storm Surge": "#6a4ca8",
    },
    "Dark": {
        "Wave Prediction": "#2bb8d4", "Wave Theory": "#9cb57a", "Wave Transformation": "#dcc8a1",
        "Structural Design": "#a9805a", "Wave Runup, Transmission, and Overtopping": "#e5654f",
        "Littoral Processes": "#5bcee2", "Inlet Processes": "#cbb07e", "Harbor Design": "#7de39a",
        "Storm Surge": "#9a86d4",
    },
}
# canonical area order (also the index used by the web --area-N variables)
AREA_ORDER = list(AREA_COLORS["Light"].keys())

# ink colors per mode for the appended overrides (fieldLabel / resultValue)
_INK = {"Light": {"ink": "#16211f", "ink2": "#4c5a5c"},
        "Dark":  {"ink": "#eaf2f3", "ink2": "#a7b7bb"}}
_MONO = '"JetBrains Mono", "Cascadia Mono", "Consolas", monospace'

# class-badge fill style
BADGE_STYLES = ["Tinted", "Solid"]
DEFAULT_BADGE = "Tinted"
_ACCENT = {"Light": "#008cb0", "Dark": "#2bb8d4"}
_BADGE_TEXT = {"Light": "#ffffff", "Dark": "#0b1316"}   # contrast text for solid fills

_DIR = os.path.dirname(os.path.abspath(__file__))
_QSS_DIR = os.path.join(_DIR, "qss")


def get_plot_palette(mode: str = DEFAULT_THEME) -> dict:
    return PLOT_PALETTES.get(mode, PLOT_PALETTES[DEFAULT_THEME])


def class_color(cls: str, mode: str = DEFAULT_THEME) -> str:
    return SEMANTIC.get(mode, SEMANTIC[DEFAULT_THEME]).get(cls, "#888888")


def vibe_label_opts(vibe: str = DEFAULT_VIBE) -> dict:
    return dict(VIBES.get(vibe, VIBES[DEFAULT_VIBE]))


def load_qss(mode: str = DEFAULT_THEME, vibe: str = DEFAULT_VIBE,
             badge: str = DEFAULT_BADGE) -> str:
    prefix = THEMES.get(mode, THEMES[DEFAULT_THEME])
    sheet = f"{prefix}-{vibe.lower()}.qss"
    path = os.path.join(_QSS_DIR, sheet)
    if not os.path.exists(path):
        path = os.path.join(_QSS_DIR, f"{THEMES[DEFAULT_THEME]}-signal.qss")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    # icon urls are relative ("url(icons/...)") -> make absolute to this package
    text = text.replace("url(icons/", f"url({_DIR.replace(chr(92), '/')}/icons/")
    ink = _INK.get(mode, _INK[DEFAULT_THEME])
    icons = _DIR.replace(chr(92), "/") + "/icons/" + ("light" if mode == "Light" else "dark")
    # overrides for object names the stock sheet doesn't define
    text += (f"\n/* CHESS-QC overrides */\n"
             f"QLabel#fieldLabel {{ color: {ink['ink2']}; background: transparent; }}\n"
             f"QLabel#resultValue {{ font-family: {_MONO}; color: {ink['ink']}; background: transparent; }}\n"
             f"QPushButton#AppCard {{ text-align: left; padding: 9px 12px 9px 14px;"
             f" border-left: 4px solid {ink['ink2']}; }}\n"
             # Light/Dark switch: a QCheckBox whose indicator is a switch image (theme-aware)
             f"QCheckBox#ModeSwitch {{ spacing: 0; background: transparent; }}\n"
             f"QCheckBox#ModeSwitch::indicator {{ width: 44px; height: 24px; }}\n"
             f"QCheckBox#ModeSwitch::indicator:unchecked {{ image: url({icons}/switch-off.svg); }}\n"
             f"QCheckBox#ModeSwitch::indicator:checked {{ image: url({icons}/switch-on.svg); }}\n")
    # landing-page cards: border-left colored by functional area
    for area, col in AREA_COLORS.get(mode, AREA_COLORS[DEFAULT_THEME]).items():
        text += f'QPushButton#AppCard[area="{area}"] {{ border-left: 4px solid {col}; }}\n'
    # solid-fill class badges (override the stock sheet's pale tinted style)
    if badge == "Solid":
        sem = SEMANTIC.get(mode, SEMANTIC[DEFAULT_THEME])
        txt = _BADGE_TEXT.get(mode, "#ffffff")
        fills = {"success": sem["exact"], "warning": sem["standard"], "danger": sem["provisional"],
                 "accent": _ACCENT.get(mode, "#008cb0")}
        for name, col in fills.items():
            text += f'QLabel[badge="{name}"] {{ background-color: {col}; color: {txt}; }}\n'
    return text
