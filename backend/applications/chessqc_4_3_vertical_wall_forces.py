"""CHESS-QC application 4-3 — Nonbreaking Wave Forces at Vertical Walls.

Standing-wave (clapotis) forces and overturning moments on a vertical wall, by two
methods: Sainflou (1928) and Miche-Rundgren [Miche (1944), Rundgren (1958)], for the
crest and the trough at the wall. The wall surface elevation (height above bottom) is
the Miche-Rundgren second-order result (used for both methods per the TR); force and
moment come from numerically integrating the Lagrangian pressure over 90 increments,
weighting each at-rest increment by its stretched (elevated) thickness.

Classification: exact (closed-form analytical standing-wave pressure theory -- Sainflou
1928 and Miche-Rundgren second-order; no empirical regression, the reflection coefficient
is a user input; reproduces the User's Guide Example 4-3 to ~0.1%, the residual being only
the dispersion solve and the 90-increment quadrature).

Self-contained (zero sibling imports): embeds the AppMeta/Field/Out/Result dataclasses
and the Hunt (1979) dispersion solver. No wave profile -> value rows only.

Theory: TR 4-3 (eqs 1-18). Validated against the ACES User's Guide Example 4-3
(see tests/test_manual_oracle.py).

Run:
    python chessqc_4_3_vertical_wall_forces.py   # self-tests + tabulate the manual example
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# --- constants ------------------------------------------------------------------
G_SI = 9.80665
_FT = 0.3048               # ft -> m
_LBF_PER_FT3 = 157.08746   # lb/ft^3 -> N/m^3
_LBF_PER_FT = 14.593903    # lb/ft -> N/m   (force per unit length)
_MOM = 4.4482216           # lb-ft/ft -> N-m/m  (moment per unit length)


@dataclass(frozen=True)
class AppMeta:
    aces_id: str
    name: str
    area: str
    classification: str
    cite: str
    default_system: str = "SI"


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


APP_META = AppMeta(
    aces_id="4-3",
    name="Nonbreaking Wave Forces at Vertical Walls",
    area="Structural Design",
    classification="exact",
    cite="Sainflou (1928); Miche (1944); Rundgren (1958); Hunt (1979); TR 4-3",
    default_system="US",
)

# Complete input list (ACES User's Guide 4-3). Defaults = User's Guide Example 4-3.
INPUTS = (
    Field("d", "Depth at SWL", "float", "m", "ft", default=15.0 * _FT, lo=1e-6, hi=1e5,
          note="still-water depth at the wall"),
    Field("H_i", "Incident wave height", "float", "m", "ft", default=8.0 * _FT, lo=1e-6, hi=1e4,
          note="incident wave height (the clapotis reaches ~(1+chi)*H_i at the wall)"),
    Field("T", "Wave period", "float", "s", "s", default=10.0, lo=1e-3, hi=1e4, note="> 0"),
    Field("chi", "Wave reflection coefficient", "float", "", "", default=1.0, lo=0.0, hi=1.0,
          note="1.0 = full reflection (smooth wall); do not use < 0.9 for design"),
    Field("cot_phi", "Cotangent of nearshore slope", "float", "", "", default=100.0, lo=1e-6, hi=1e6,
          note="collected for context (not used in the wall-force calculation)"),
    Field("gamma", "Water unit weight", "float", "kN/m^3", "lb/ft^3", default=64.0 * _LBF_PER_FT3,
          lo=1.0, hi=1e6, note="64 lb/ft^3 seawater, 62.4 lb/ft^3 fresh"),
)

# Complete output list (ACES User's Guide 4-3). Heights are the Miche-Rundgren clapotis
# elevations (shared by both methods); force/moment are per method & case.
OUTPUTS = (
    Out("hgt_crest",  "Crest height above bottom",          "m", "ft", "scalar"),
    Out("hgt_trough", "Trough height above bottom",         "m", "ft", "scalar"),
    Out("mr_F_crest", "Miche-Rundgren crest force",         "N/m", "lb/ft", "scalar"),
    Out("mr_M_crest", "Miche-Rundgren crest moment",        "N-m/m", "lb-ft/ft", "scalar"),
    Out("mr_F_trough","Miche-Rundgren trough force",        "N/m", "lb/ft", "scalar"),
    Out("mr_M_trough","Miche-Rundgren trough moment",       "N-m/m", "lb-ft/ft", "scalar"),
    Out("sf_F_crest", "Sainflou crest force",               "N/m", "lb/ft", "scalar"),
    Out("sf_M_crest", "Sainflou crest moment",              "N-m/m", "lb-ft/ft", "scalar"),
    Out("sf_F_trough","Sainflou trough force",              "N/m", "lb/ft", "scalar"),
    Out("sf_M_trough","Sainflou trough moment",             "N-m/m", "lb-ft/ft", "scalar"),
)


@dataclass
class Result:
    hgt_crest: float; hgt_trough: float
    mr_F_crest: float; mr_M_crest: float; mr_F_trough: float; mr_M_trough: float
    sf_F_crest: float; sf_M_crest: float; sf_F_trough: float; sf_M_trough: float
    notes: str = ""


# --- Hunt (1979) dispersion (identical kernel to 2-1/3-1/4-2) -------------------
_HUNT_D = (0.66667, 0.35550, 0.16084, 0.06320, 0.02174,
           0.00654, 0.00171, 0.00039, 0.00011)


def wave_celerity(T: float, d: float, g: float = G_SI) -> float:
    omega = 2.0 * math.pi / T
    y = omega * omega * d / g
    denom = 1.0 + sum(dn * y ** (n + 1) for n, dn in enumerate(_HUNT_D))
    return math.sqrt(g * d / (y + 1.0 / denom))


def _validate(inp: dict) -> None:
    for f in INPUTS:
        v = inp[f.key]
        if not (f.lo <= v <= f.hi):
            raise ValueError(f"{f.label} ({f.key}) = {v} outside [{f.lo}, {f.hi}] ({f.note})")


def compute(inp: dict, *, g: float = G_SI, n_inc: int = 90) -> Result:
    """Vertical-wall standing-wave forces for SI inputs {d, H_i, T, chi, cot_phi, gamma}."""
    _validate(inp)
    d = float(inp["d"]); H = float(inp["H_i"]); T = float(inp["T"])
    chi = float(inp["chi"]); gam = float(inp["gamma"])

    L = wave_celerity(T, d, g) * T
    k = 2.0 * math.pi / L
    a = k * d
    sh, ch, tanh = math.sinh(a), math.cosh(a), math.tanh(a)
    coth = ch / sh
    th1 = 1.0 + 3.0 / (4.0 * sh * sh) - 1.0 / (4.0 * ch * ch)
    th2 = 3.0 / (4.0 * sh * sh) - 1.0 / (4.0 * ch * ch)
    c1, c2 = (1.0 + chi) ** 2, (1.0 - chi) ** 2

    # --- wall surface elevations (Miche-Rundgren eqs 12/13; used for both methods) ---
    second = (math.pi * H / 4.0) * (H / L) * coth * (c1 * th1 + c2 * th2)
    eta_cr = (H / 2.0) * (1.0 + chi) + second
    eta_tr = -(H / 2.0) * (1.0 + chi) + second

    # --- per-method Lagrangian elevation y(y0) and pressure p(y0) ---
    def y_sain(y0, s):       # eq 1 (crest s=+1) / eq 2 (trough s=-1)
        R = math.sinh(k * (d + y0)) / sh
        Q = math.cosh(k * (d + y0)) / sh
        return y0 + s * H * R + math.pi * H * (H / L) * R * Q

    def p_sain(y0, s):       # eq 6 / eq 7  (returns pressure, Pa-equivalent in SI)
        return gam * (-y0 - s * H * math.sinh(k * y0) / (sh * ch))

    def y_mr(y0, s):         # eq 8 / eq 9
        R = math.sinh(k * (d + y0)) / sh
        Q = math.cosh(k * (d + y0)) / sh
        return y0 + s * (H / 2.0) * (1.0 + chi) * R + (math.pi * H / 4.0) * (H / L) * R * Q * (c1 * th1 + c2 * th2)

    def p_mr(y0, s):         # eq 15 / eq 16 (with theta_3, theta_4 from eqs 17/18)
        th3 = ((1.0 - 1.0 / (4.0 * ch * ch)) * math.cosh(k * (2.0 * d + y0))
               - 2.0 * tanh * math.sinh(k * (2.0 * d + y0))
               + 0.75 * (math.cosh(k * y0) / (sh * sh) - 2.0 * math.cosh(k * (d + y0)) / ch))
        th4 = (math.cosh(k * (2.0 * d + y0)) / (4.0 * ch * ch)
               - 2.0 * tanh * math.sinh(k * (2.0 * d + y0))
               + 0.75 * (math.cosh(k * y0) / (sh * sh) - 2.0 * math.cosh(k * (d + y0)) / ch))
        return gam * (-y0 - s * (H / 2.0) * (1.0 + chi) * math.sinh(k * y0) / (sh * ch)
                      - (math.pi * H / 4.0) * (H / L) * math.sinh(k * y0) / (sh * sh) * (c1 * th3 + c2 * th4))

    def integrate(yfun, pfun, crest):
        """90 at-rest increments over [-d, 0]; force = sum p*dz over the stretched
        (elevated) thickness; moment = sum p*(z+d)*dz about the bottom."""
        s = 1.0 if crest else -1.0
        F = M = 0.0
        prev_z = yfun(-d, s)
        for i in range(1, n_inc + 1):
            y0 = -d + i * (d / n_inc)
            z = yfun(y0, s)
            dz = z - prev_z
            y0m = -d + (i - 0.5) * (d / n_inc)
            p = pfun(y0m, s)
            F += p * dz
            M += p * (0.5 * (prev_z + z) + d) * dz
            prev_z = z
        return F, M

    mr_F_cr, mr_M_cr = integrate(y_mr, p_mr, True)
    mr_F_tr, mr_M_tr = integrate(y_mr, p_mr, False)
    sf_F_cr, sf_M_cr = integrate(y_sain, p_sain, True)
    sf_F_tr, sf_M_tr = integrate(y_sain, p_sain, False)

    notes = (f"chi={chi:g}; d/L={d / L:.3f}; smaller (force) values recommended "
             f"({'Sainflou' if sf_F_cr < mr_F_cr else 'Miche-Rundgren'})")

    return Result(
        hgt_crest=d + eta_cr, hgt_trough=d + eta_tr,
        mr_F_crest=mr_F_cr, mr_M_crest=mr_M_cr, mr_F_trough=mr_F_tr, mr_M_trough=mr_M_tr,
        sf_F_crest=sf_F_cr, sf_M_crest=sf_M_cr, sf_F_trough=sf_F_tr, sf_M_trough=sf_M_tr,
        notes=notes,
    )


# --- self-tests + manual-example tabulation -------------------------------------
def _self_tests() -> None:
    r = compute({f.key: f.default for f in INPUTS})
    assert abs(r.hgt_crest / _FT - 32.95) < 0.1, r.hgt_crest / _FT
    assert abs(r.hgt_trough / _FT - 16.95) < 0.1, r.hgt_trough / _FT
    assert abs(r.sf_F_crest / _LBF_PER_FT - 17724.17) < 30, r.sf_F_crest / _LBF_PER_FT
    assert abs(r.mr_F_crest / _LBF_PER_FT - 28683.39) < 40, r.mr_F_crest / _LBF_PER_FT
    assert abs(r.sf_M_crest / _MOM - 148008.60) < 300, r.sf_M_crest / _MOM
    assert abs(r.mr_M_crest / _MOM - 306958.40) < 600, r.mr_M_crest / _MOM
    print("  self-tests: PASS (matches User's Guide Example 4-3, both methods)")


def _tab() -> None:
    r = compute({f.key: f.default for f in INPUTS})
    print(f"\nACES application {APP_META.aces_id} - {APP_META.name}  [{APP_META.classification}]")
    print(f"  cite: {APP_META.cite}")
    print("  (US units; matches User's Guide Example 4-3)")
    print(f"  {'':22}{'M-R crest':>12}{'M-R trough':>12}{'Sain crest':>12}{'Sain trough':>12}")
    print(f"  {'Hgt above bottom (ft)':22}{r.hgt_crest/_FT:12.2f}{r.hgt_trough/_FT:12.2f}"
          f"{r.hgt_crest/_FT:12.2f}{r.hgt_trough/_FT:12.2f}")
    print(f"  {'Force (lb/ft)':22}{r.mr_F_crest/_LBF_PER_FT:12.2f}{r.mr_F_trough/_LBF_PER_FT:12.2f}"
          f"{r.sf_F_crest/_LBF_PER_FT:12.2f}{r.sf_F_trough/_LBF_PER_FT:12.2f}")
    print(f"  {'Moment (lb-ft/ft)':22}{r.mr_M_crest/_MOM:12.2f}{r.mr_M_trough/_MOM:12.2f}"
          f"{r.sf_M_crest/_MOM:12.2f}{r.sf_M_trough/_MOM:12.2f}")
    print("  manual: 32.95/16.95; F 28683.39/7121.92/17724.17/2323.04; "
          "M 306958.40/38825.47/148008.60/7214.73")
    print(f"  notes: {r.notes}")


if __name__ == "__main__":
    print(f"CHESS-QC {APP_META.aces_id} {APP_META.name} - running self-tests...")
    _self_tests()
    _tab()
