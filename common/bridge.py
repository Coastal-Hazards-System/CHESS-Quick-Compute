"""Python<->JS bridge for the Pyodide web driver.

Pure Python (runs identically under Pyodide and CPython, so it is unit-testable
without a browser). Serializes any chessqc_* application's contract to JSON and runs
its compute() returning JSON. The JS driver calls `contract(mod)` and `run(mod, json)`.
"""
import dataclasses
import json
import math


def _clean(x):
    if isinstance(x, float) and math.isinf(x):
        return None
    return x


def _san(v):
    """Make a result value valid-JSON for the browser: non-finite floats (inf/nan)
    become string tokens so JSON.parse never sees the Python-only `Infinity`/`NaN`."""
    if isinstance(v, float):
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if math.isnan(v):
            return "nan"
        return v
    if isinstance(v, list):
        return [_san(x) for x in v]
    return v


def contract(mod) -> str:
    """JSON of {meta, inputs, outputs} for the application module `mod`."""
    meta = dataclasses.asdict(mod.APP_META)
    inputs = []
    for f in mod.INPUTS:
        d = dataclasses.asdict(f)
        d["lo"] = _clean(d["lo"])
        d["hi"] = _clean(d["hi"])
        d["choices"] = list(d.get("choices") or [])
        inputs.append(d)
    outputs = [dataclasses.asdict(o) for o in mod.OUTPUTS]
    return json.dumps({"meta": meta, "inputs": inputs, "outputs": outputs})


def run(mod, inp_json: str) -> str:
    """Run mod.compute on SI inputs (JSON dict); return JSON of all outputs.

    Scalars/points -> numbers; profile arrays -> lists; plus `_notes` and, on
    failure, `_error`."""
    try:
        inp = json.loads(inp_json)
        r = mod.compute(inp)
    except Exception as exc:  # validation / numeric
        return json.dumps({"_error": str(exc)})
    out = {}
    for o in mod.OUTPUTS:
        v = getattr(r, o.key, None)
        if v is None:
            continue
        if hasattr(v, "tolist"):      # numpy array / scalar
            v = v.tolist()
        out[o.key] = _san(v)
    out["_notes"] = getattr(r, "notes", "")
    return json.dumps(out)
