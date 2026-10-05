#!/usr/bin/env python3
"""Compatibility entry point for agxdis; the implementation is agxforge.g17.agxdis.

The same forwarding proxy as the g17 shims: reads, writes and deletes reach the implementation,
nothing is copied, and this stays a real module loaded from this file so cold source auditing
records the shim that selected the implementation.

This module is NOT named g17*, and that is why it was invisible to the dependency scans that
produced the first migration order. Its consumers are 19 files under tools/ and test/ that
`import agxdis` after putting tools/ on sys.path - including scripts under spike/ that do so with a
hardcoded absolute path - so this file must keep existing at exactly this location.
"""
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agxforge.g17.agxdis as _impl


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