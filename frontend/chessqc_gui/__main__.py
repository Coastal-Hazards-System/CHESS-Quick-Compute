"""Launcher:  python -m chessqc_gui <aces_id|name>

Discovers chessqc_* applications under ../applications, matches one by ACES id (e.g.
"2-1") or name substring, and opens its calculator. With no argument, lists the
available applications.
"""
from __future__ import annotations

import importlib.util
import os
import sys

# this file is frontend/chessqc_gui/__main__.py; the apps live in backend/applications/
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APPLICATIONS_DIR = os.path.join(_ROOT, "backend", "applications")


def _load_module(path: str):
    name = "chessqc_app_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses + `from __future__ import annotations`
    # resolve string annotations via sys.modules[cls.__module__].
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def discover() -> dict:
    """Return {aces_id: module} for every application exposing the contract."""
    apps = {}
    if not os.path.isdir(APPLICATIONS_DIR):
        return apps
    for fn in sorted(os.listdir(APPLICATIONS_DIR)):
        if not (fn.startswith("chessqc_") and fn.endswith(".py")):
            continue
        try:
            mod = _load_module(os.path.join(APPLICATIONS_DIR, fn))
            if all(hasattr(mod, a) for a in ("APP_META", "INPUTS", "OUTPUTS", "compute")):
                apps[mod.APP_META.aces_id] = mod
        except Exception as exc:  # don't let one bad file break discovery
            print(f"  (skipped {fn}: {exc})")
    return apps


def resolve(query: str, apps: dict):
    if query in apps:
        return apps[query]
    q = query.lower()
    for mod in apps.values():
        if q in mod.APP_META.name.lower():
            return mod
    return None


from .theme import load_qss, THEMES  # noqa: E402  (re-exported for callers/tests)


def _make_app():
    from .qt import QtWidgets
    from . import settings
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(load_qss(settings.get_theme(), settings.get_vibe(), settings.get_badge()))
    return app


def launch(mod):
    from .app_shell import CalculatorWindow
    app = _make_app()
    win = CalculatorWindow(mod)
    win.show()
    return app.exec()


def launch_hub(apps=None):
    """Open the landing-page hub listing all applications."""
    from .hub import HubWindow
    app = _make_app()
    win = HubWindow(apps if apps is not None else discover())
    win.show()
    return app.exec()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    apps = discover()
    if not apps:
        print(f"No applications found in {APPLICATIONS_DIR}")
        return 1
    if not argv or argv[0] in ("--hub", "hub"):
        return launch_hub(apps)        # landing page
    mod = resolve(argv[0], apps)
    if mod is None:
        print(f"No application matching '{argv[0]}'. Available: {', '.join(apps)}")
        return 1
    return launch(mod)


if __name__ == "__main__":
    sys.exit(main())
