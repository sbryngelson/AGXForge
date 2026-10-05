#!/usr/bin/env python3
"""Compatibility entry point for g17auth; the implementation is agxforge.g17.auth.

A PROXY IN BOTH DIRECTIONS, NOT A COPY. Two weaker shapes were measured and rejected:

  `from agxforge.g17.auth import *`
      binds every public name by VALUE at import, so a global the implementation rebinds later
      stays frozen here. g17auth._LIFE loads lazily under `global _LIFE`; a caller reading
      g17auth._LIFE through a copying shim gets None forever.

  the same plus a module-level `__getattr__`
      does not reach the names `*` already copied - `__getattr__` fires only for attributes
      MISSING from this module - and it does not forward ASSIGNMENT. A test monkeypatching
      g17auth.X would patch this module while the implementation ran unchanged: a mutation
      control that silently stops controlling rather than failing. 22 files in this tree control
      behaviour that way.

Replacing this module's class forwards reads, writes and deletes, so there is exactly ONE binding
of every name. This module stays a real, distinct module loaded from this file - it is not
sys.modules aliasing - so cold source auditing still records the shim that selected the
implementation, not only the implementation it selected.
"""
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agxforge.g17.auth as _impl


class _Forwarding(types.ModuleType):
    def __getattr__(self, name):
        return getattr(_impl, name)

    def __setattr__(self, name, value):
        if name.startswith("__") or name == "_impl":
            return object.__setattr__(self, name, value)
        setattr(_impl, name, value)

    def __delattr__(self, name):
        if name.startswith("__") or name == "_impl":
            return object.__delattr__(self, name)
        delattr(_impl, name)

    def __dir__(self):
        return dir(_impl)


sys.modules[__name__].__class__ = _Forwarding


# THE COMMAND-LINE ENTRY IS PART OF THE COMPATIBILITY SURFACE, and leaving it out is a silent
# failure rather than a loud one: running this file imports the implementation, installs the
# forwarding class and exits ZERO, having done nothing. Root found both CLIs dead that way.
# main() is called on the implementation so argument parsing and exit status stay in one place.
if __name__ == "__main__":
    sys.exit(_impl.main())