#!/usr/bin/env python3
"""Compatibility entry point for g17cc; the implementation is agxforge.g17.cc.

The compiler itself is in the package now. It could only move once everything it reaches was
already there: the IR, the assembler and its form tables, the six consolidated form encoders, the
control-flow and SSA passes, the object readers, and the production halves of const, formops,
predicateform, tensorlower and uniformpreload. Until then any move would have left the compiler
behind a facade that imported tools/ - the one shape the policy forbids, and the reason this was
the last batch rather than the first.

This file owns no logic. It forwards every attribute, because g17cc moved WHOLE rather than being
split, so unlike the split entries there is no retained probe code here that reads a bare global.
"""
from pathlib import Path
import os as _os
import sys as _sys

# THE REPOSITORY IMPORT ROOT COMES FIRST, ahead of the package import: run by absolute path from
# another directory with PYTHONPATH unset, nothing else puts the checkout on sys.path.
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from agxforge.g17 import cc as _impl
from agxforge.g17 import compat as _compat

_compat.install(__name__, _impl)
