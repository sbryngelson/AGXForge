#!/usr/bin/env python3
"""Compatibility entry point for g17opclass; the implementation is agxforge.g17.opclass.

Forwarding proxy: reads, writes and deletes reach the implementation, nothing is copied, and this
stays a real module loaded from this file so cold source auditing records the shim that selected
the implementation. Shebang, mode and CLI are inherited from the replaced file, not chosen.
"""
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agxforge.g17.opclass as _impl


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


if __name__ == "__main__":
    sys.exit(_impl.main())