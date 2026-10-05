"""Compatibility import for the retained common runtime contracts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agxforge.g17 import runtime as _impl
from agxforge.g17.compat import install as _install

globals().update(_install(__name__, _impl, {
    name: name for name in vars(_impl) if not name.startswith("__")
}))
