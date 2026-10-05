"""Compatibility entry point for the IR; the implementation is agxforge.g17.ir.

The same shape as tools/g17evidence.py: this file keeps the import name its callers already use
and owns no logic. There is ONE IR implementation and it lives in the library.

`_fbits` and `_expand_erf` are re-exported explicitly because `import *` skips underscore names.
Dropping them would not fail here - it would fail in a caller, later, reading as a compiler defect.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agxforge.g17.ir import *                      # noqa: F401,F403
from agxforge.g17.ir import _fbits, _expand_erf    # noqa: F401
