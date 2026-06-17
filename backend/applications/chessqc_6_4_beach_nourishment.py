"""CHESS-QC application 6-4 — Beach Nourishment Overfill Ratio and Volume.

Originating ACES application: 6-4 "Beach Nourishment Overfill Ratio and Volume"
(functional area: Littoral Processes). Tells a nourishment designer how much borrow
sand to place to obtain a given volume of usable beach (the overfill ratio), and how
much faster or slower the borrow sand erodes compared with the native sand (the
renourishment factor).

Classification: exact (closed-form James 1975 formulas).
Theory and references: James (1975), TM 60; Krumbein (1957); SPM (1984). Equations
transcribed in docs/EQUATIONS.md, TR chapter 6-4.

    delta = (M_b - M_n) / sigma_n            (phi-mean difference, scaled by native sorting)
    sigma = sigma_b / sigma_n                (sorting ratio)
    Overfill ratio R_A (James 1975), by category of (delta, sigma):
      1/R_A = 1 - F((th2-delta)/sigma) + F((th1-delta)/sigma)
              + [(F(th2)-F(th1))/sigma] * exp{ 0.5*[ th1^2 - ((th1-delta)/sigma)^2 ] }
      finer (delta>=0):  th1 = max(-1, -delta/(sigma^2-1)),  th2 = +inf
      coarser (delta<0): th1 = -1,  th2 = max(-1, 1 + 2 delta/(1-sigma^2))
    Renourishment factor: R_J = exp[ W*delta - (W^2/2)*(sigma^2 - 1) ],  W = winnowing (=1)
    Design (borrow) volume: VOL_D = VOL_I * R_A

with F the standard-normal CDF, M the phi mean, sigma the phi standard deviation, and
subscripts b (borrow) and n (native).

Self-containment: zero sibling imports; embeds its own contract dataclasses and a normal
CDF (math.erf). Runnable standalone:
    python chessqc_6_4_beach_nourishment.py
which reproduces the ACES User's Guide Example 6-4, then prints it. stdlib only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_YD3 = 0.764554858          # m^3 per yd^3


@dataclass(frozen=True)
class AppMeta:
    aces_id: str
    name: str
    area: str
    classification: str
    cite: str
    default_system: str = "SI"
    status: str = "Current"
    superseded_by: str = ""


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
    aces_id="6-4",
    name="Beach Nourishment Overfill Ratio and Volume",
    area="Littoral Processes",
    classification="exact",
    cite="James (1975) TM-60; SPM (1984); TR 6-4",
    default_system="US",
)

INPUTS = (
    Field("VOL_I", "Initial (usable) volume", "float", "m^3", "yd^3", default=800000.0 * _YD3,
          lo=0.0, hi=1e12, note="target volume of usable beach fill"),
    Field("M_n", "Native mean", "float", "phi", "phi", default=1.800, lo=-5.0, hi=15.0,
          note="phi mean grain size of the native beach sand"),
    Field("sigma_n", "Native standard deviation", "float", "phi", "phi", default=0.450,
          lo=1e-4, hi=10.0, note="phi sorting (standard deviation) of the native sand"),
    Field("M_b", "Borrow mean", "float", "phi", "phi", default=2.250, lo=-5.0, hi=15.0,
          note="phi mean grain size of the borrow sand"),
    Field("sigma_b", "Borrow standard deviation", "float", "phi", "phi", default=0.760,
          lo=1e-4, hi=10.0, note="phi sorting (standard deviation) of the borrow sand"),
    Field("W", "Winnowing function", "float", "", "", default=1.0, lo=0.0, hi=5.0,
          note="James (1975) renourishment winnowing parameter (recommended 1.0)"),
)

OUTPUTS = (
    Out("R_A", "Overfill ratio", "", "", "scalar"),
    Out("R_J", "Renourishment factor", "", "", "scalar"),
    Out("VOL_D", "Design (borrow) volume", "m^3", "yd^3", "scalar"),
    Out("delta", "Phi-mean difference (scaled)", "", "", "scalar"),
    Out("sigma_ratio", "Sorting ratio", "", "", "scalar"),
    Out("category", "James category", "", "", "scalar"),
)


@dataclass
class Result:
    R_A: float; R_J: float; VOL_D: float; delta: float; sigma_ratio: float; category: str
    notes: str = ""


def _ncdf(x: float) -> float:
    """Standard-normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _validate(inp: dict) -> None:
    for f in INPUTS:
        if f.kind not in ("float", "int", "angle"):
            continue
        v = float(inp[f.key])
        if not (f.lo <= v <= f.hi):
            raise ValueError(f"{f.label} ({f.key}) = {v} outside [{f.lo}, {f.hi}] ({f.note})")


def compute(inp: dict) -> Result:
    """Overfill ratio, renourishment factor, and design volume for SI inputs."""
    _validate(inp)
    VOL_I = float(inp["VOL_I"]); M_n = float(inp["M_n"]); s_n = float(inp["sigma_n"])
    M_b = float(inp["M_b"]); s_b = float(inp["sigma_b"]); W = float(inp.get("W", 1.0))

    delta = (M_b - M_n) / s_n
    sig = s_b / s_n

    if delta >= 0.0:                                    # borrow finer than native (I / II)
        denom = sig * sig - 1.0
        th1 = max(-1.0, -delta / denom) if abs(denom) > 1e-12 else -1.0
        th2 = math.inf
        category = "I (finer, more poorly sorted)" if sig > 1.0 else "II (finer, better sorted)"
    else:                                               # borrow coarser than native (III / IV)
        th1 = -1.0
        denom = 1.0 - sig * sig
        th2 = max(-1.0, 1.0 + 2.0 * delta / denom) if abs(denom) > 1e-12 else -1.0
        category = "III (coarser, more poorly sorted)" if sig > 1.0 else "IV (coarser, better sorted)"

    F1 = _ncdf((th1 - delta) / sig)
    expf = math.exp(0.5 * (th1 * th1 - ((th1 - delta) / sig) ** 2))
    if math.isinf(th2):
        inv = F1 + ((1.0 - _ncdf(th1)) / sig) * expf    # F(inf) = 1
    else:
        F2arg = _ncdf((th2 - delta) / sig)
        inv = 1.0 - F2arg + F1 + ((_ncdf(th2) - _ncdf(th1)) / sig) * expf
    R_A = 1.0 / inv if inv > 0 else float("inf")

    R_J = math.exp(W * delta - 0.5 * W * W * (sig * sig - 1.0))
    VOL_D = VOL_I * R_A

    notes = [f"category {category}; delta = {delta:.3f}, sorting ratio = {sig:.3f}"]
    if abs(M_b - M_n) < 1e-9 and abs(sig - 1.0) < 1e-9:
        notes.append("identical borrow and native: R_A = R_J = 1")
    return Result(R_A=R_A, R_J=R_J, VOL_D=VOL_D, delta=delta, sigma_ratio=sig,
                  category=category, notes="; ".join(notes))


# --- self-tests (User's Guide Example 6-4) --------------------------------------
def _approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _self_tests() -> None:
    r = compute(dict(VOL_I=800000.0 * _YD3, M_n=1.800, sigma_n=0.450, M_b=2.250,
                     sigma_b=0.760, W=1.0))
    assert _approx(r.R_A, 2.003, 3e-3), r.R_A
    assert _approx(r.R_J, 1.077, 2e-3), r.R_J
    assert _approx(r.VOL_D / _YD3, 1602521.0, 3000.0), r.VOL_D / _YD3
    assert r.category.startswith("I ")

    # identical borrow and native -> R_A = R_J = 1
    r0 = compute(dict(VOL_I=1.0, M_n=2.0, sigma_n=0.5, M_b=2.0, sigma_b=0.5, W=1.0))
    assert _approx(r0.R_A, 1.0, 1e-6) and _approx(r0.R_J, 1.0, 1e-9)
    # coarser, better-sorted borrow (case IV) is favorable: R_A near or below 1
    r4 = compute(dict(VOL_I=1.0, M_n=2.0, sigma_n=0.6, M_b=1.5, sigma_b=0.4, W=1.0))
    assert r4.category.startswith("IV") and r4.R_A > 0

    print("  self-tests: PASS (User's Guide Example 6-4 R_A/R_J/VOL_D; identity; case IV)")


def _print_default_example() -> None:
    inp = {f.key: f.default for f in INPUTS}
    r = compute(inp)
    print(f"\nACES application {APP_META.aces_id} - {APP_META.name}  [{APP_META.classification}]")
    print(f"  cite: {APP_META.cite}")
    print("  INPUTS:")
    print(f"    VOL_I = {inp['VOL_I'] / _YD3:.0f} yd^3 | native M={inp['M_n']}, s={inp['sigma_n']} | "
          f"borrow M={inp['M_b']}, s={inp['sigma_b']} phi")
    print("  OUTPUTS:")
    print(f"    Overfill ratio        R_A   = {r.R_A:.3f}")
    print(f"    Renourishment factor  R_J   = {r.R_J:.3f}")
    print(f"    Design volume         VOL_D = {r.VOL_D / _YD3:.0f} yd^3")
    print(f"    category: {r.category}")
    print(f"  notes: {r.notes}")


if __name__ == "__main__":
    print(f"CHESS-QC {APP_META.aces_id} {APP_META.name} - running self-tests...")
    _self_tests()
    _print_default_example()
