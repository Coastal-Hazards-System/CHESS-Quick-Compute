"""CHESS-QC application 4-1 — Breakwater Design Using Hudson and Related Equations.

Sizes the primary armor units of a rubble-mound breakwater/revetment from the Hudson
(1953-61) stability equation, and reports crest width, cover-layer thickness, and
armor-unit placement density (SPM 1984 Ch. 7; EM 1110-2-2904).

Classification: exact (closed-form Hudson stability equation and the related SPM crest-
width / thickness / placement-density formulas; reproduces the User's Guide Example 4-1 to
the digit. K_D, k_delta and porosity are user-supplied table coefficients, as in ACES.)

Self-contained (zero sibling imports): embeds the AppMeta/Field/Out/Result dataclasses.
No wave kinematics -> value-rows-only (no profile plot).

Theory: TR 4-1 (eqs 1-4). Validated against the ACES User's Guide Example 4-1
(see tests/test_manual_oracle.py).

Run:
    python chessqc_4_1_breakwater_hudson.py    # self-tests + tabulate the manual example
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# --- constants ------------------------------------------------------------------
_FT = 0.3048               # ft -> m
_LBF_PER_FT3 = 157.08746   # lb/ft^3 -> N/m^3
_LBF = 4.4482216           # lbf -> N
_TON = 8896.4432           # US short ton-force (2000 lbf) -> N


# --- contract dataclasses -------------------------------------------------------
@dataclass(frozen=True)
class AppMeta:
    aces_id: str
    name: str
    area: str
    classification: str
    cite: str
    default_system: str = "SI"
    status: str = "Current"          # Current | Screening only | Superseded
    superseded_by: str = ""          # newer method, if any (surfaced in the docs)


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: str = "float"
    unit_si: str = ""
    unit_us: str = ""
    default: object = 0.0
    lo: float = -math.inf
    hi: float = math.inf
    choices: tuple = ()
    note: str = ""


@dataclass(frozen=True)
class Out:
    key: str
    label: str
    unit_si: str = ""
    unit_us: str = ""
    kind: str = "scalar"


# armor-unit types (informational; help the user choose K_D, k_delta, P from SPM tables)
_ARMOR_TYPES = (
    "Quarrystone (smooth, rounded)", "Quarrystone (rough, angular)", "Graded riprap",
    "Tribar (trunk, nonbreaking)", "Tribar (trunk, breaking)", "Tetrapod", "Quadripod",
    "Dolos", "Modified cube", "Hexapod", "Toskane", "Other",
)

APP_META = AppMeta(
    aces_id="4-1",
    name="Breakwater Design (Hudson)",
    area="Structural Design",
    classification="exact",
    cite="Hudson (1953-61); SPM (1984) Ch.7; EM 1110-2-2904; TR 4-1",
    default_system="US",     # opens on the User's Guide Example (US units)
    superseded_by="Van der Meer (1988) stability formulae (preferred for many cases)",
)

# Complete input list (ACES User's Guide 4-1). Defaults = User's Guide Example 4-1.
INPUTS = (
    Field("armor_type", "Type of armor unit", "choice", "", "",
          default="Tribar (trunk, nonbreaking)", choices=_ARMOR_TYPES,
          note="optional/informational; pick K_D, k_delta, P from SPM tables accordingly"),
    Field("w_r", "Armor unit weight", "float", "kN/m^3", "lb/ft^3", default=165.0 * _LBF_PER_FT3,
          lo=1.0, hi=1e6, note="unit weight of armor material; must exceed water unit weight"),
    Field("H", "Wave height", "float", "m", "ft", default=11.50 * _FT, lo=1e-6, hi=1e4,
          note="design wave height (H or H_i)"),
    Field("w_w", "Water unit weight", "float", "kN/m^3", "lb/ft^3", default=64.0 * _LBF_PER_FT3,
          lo=1.0, hi=1e6, note="64 lb/ft^3 seawater, 62.4 lb/ft^3 fresh"),
    Field("K_D", "Stability coefficient", "float", "", "", default=10.0, lo=1e-3, hi=1e4,
          note="K_D from SPM Table 7-8 (depends on armor type / slope / wave condition)"),
    Field("k_delta", "Layer coefficient", "float", "", "", default=1.02, lo=1e-3, hi=10.0,
          note="layer coefficient k_delta (SPM Table 7-13)"),
    Field("P", "Average porosity of armor layer", "float", "%", "%", default=54.0, lo=0.0, hi=99.0,
          note="cover-layer porosity, percent (SPM Table 7-13)"),
    Field("cot_theta", "Cotangent of structure slope", "float", "", "", default=2.00, lo=1e-3, hi=1e3,
          note="cot(theta); theta = seaward slope angle"),
    Field("n", "Number of armor units (layer thickness)", "int", "", "", default=2, lo=1, hi=10,
          note="number of armor-unit layers (>= 2 typical)"),
)

# Complete output list (ACES User's Guide 4-1).
OUTPUTS = (
    Out("W",   "Weight of individual armor unit", "kN", "tons", "scalar"),
    Out("B",   "Crest width of breakwater",       "m",  "ft",   "scalar"),
    Out("r",   "Average cover layer thickness",   "m",  "ft",   "scalar"),
    Out("N_r", "Armor units per 1000 ft^2",       "",   "",     "scalar"),
)

_N_CREST = 3      # ACES/SPM: crest width uses a minimum of 3 armor units


@dataclass
class Result:
    W: float; B: float; r: float; N_r: float
    notes: str = ""


def _validate(inp: dict) -> None:
    for f in INPUTS:
        if f.kind in ("float", "int", "angle"):
            v = inp[f.key]
            if not (f.lo <= v <= f.hi):
                raise ValueError(f"{f.label} ({f.key}) = {v} outside [{f.lo}, {f.hi}] ({f.note})")
    if inp["w_r"] <= inp["w_w"]:
        raise ValueError("Armor unit weight must exceed water unit weight (S_r > 1)")


# --- compute (single entry point both front-ends call) --------------------------
def compute(inp: dict) -> Result:
    """Hudson breakwater design for SI inputs
    {armor_type, w_r[N/m^3], H[m], w_w[N/m^3], K_D, k_delta, P[%], cot_theta, n}."""
    _validate(inp)
    w_r = float(inp["w_r"]); H = float(inp["H"]); w_w = float(inp["w_w"])
    K_D = float(inp["K_D"]); k_delta = float(inp["k_delta"]); P = float(inp["P"])
    cot_theta = float(inp["cot_theta"]); n = int(inp["n"])

    S_r = w_r / w_w                                         # specific gravity of armor
    # (1) Hudson armor-unit weight (W in N, since w_r in N/m^3 and H in m)
    W = w_r * H ** 3 / (K_D * (S_r - 1.0) ** 3 * cot_theta)
    cube = (W / w_r) ** (1.0 / 3.0)                         # (W/w_r)^(1/3)  [m]
    # (2) crest width (ACES uses n = 3 armor units for the crest)
    B = _N_CREST * k_delta * cube
    # (3) average cover-layer thickness (uses the input number of layers n)
    r = n * k_delta * cube
    # (4) placement density: number of units per 1000 ft^2 (US-convention count;
    #     (w_r/W)^(2/3) is dimensional, so evaluate on US-unit volume to match ACES)
    w_r_us = w_r / _LBF_PER_FT3
    W_us = W / _LBF
    N_r = 1000.0 * n * k_delta * (1.0 - P / 100.0) * (w_r_us / W_us) ** (2.0 / 3.0)

    notes = f"{inp.get('armor_type', 'armor')}; S_r={S_r:.3f}; n={n} (crest uses {_N_CREST})"
    return Result(W=W, B=B, r=r, N_r=N_r, notes=notes)


# --- self-tests + manual-example tabulation -------------------------------------
def _self_tests() -> None:
    r = compute({f.key: f.default for f in INPUTS})
    assert abs(r.W / _TON - 1.59) < 0.02, r.W / _TON       # tons
    assert abs(r.B / _FT - 8.21) < 0.05, r.B / _FT         # ft
    assert abs(r.r / _FT - 5.47) < 0.05, r.r / _FT         # ft
    assert abs(r.N_r - 130.30) < 0.5, r.N_r
    # sanity: heavier armor (larger K_D) -> lighter unit
    r2 = compute({**{f.key: f.default for f in INPUTS}, "K_D": 20.0})
    assert r2.W < r.W
    print("  self-tests: PASS (matches User's Guide Example 4-1)")


def _tab() -> None:
    r = compute({f.key: f.default for f in INPUTS})
    print(f"\nACES application {APP_META.aces_id} - {APP_META.name}  [{APP_META.classification}]")
    print(f"  cite: {APP_META.cite}")
    print("  (values in US units; matches User's Guide Example 4-1)")
    print(f"  Armor unit weight W   = {r.W / _TON:8.2f} tons   [manual 1.59]")
    print(f"  Crest width B         = {r.B / _FT:8.2f} ft     [manual 8.21]")
    print(f"  Cover layer thickness = {r.r / _FT:8.2f} ft     [manual 5.47]")
    print(f"  Units per 1000 ft^2   = {r.N_r:8.2f}        [manual 130.30]")
    print(f"  notes: {r.notes}")


if __name__ == "__main__":
    print(f"CHESS-QC {APP_META.aces_id} {APP_META.name} - running self-tests...")
    _self_tests()
    _tab()
