"""Single Qt import surface for the desktop driver.

All Qt access funnels through here so the binding is pinned in one place (and a
future swap is a one-file edit). We target PySide6 (LGPL).
"""
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: F401
from PySide6.QtCore import Qt  # noqa: F401
