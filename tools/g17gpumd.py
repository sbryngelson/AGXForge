#!/usr/bin/env python3
"""Compatibility entry point for g17gpumd; the implementation is agxforge.g17.gpumd.

A forwarding proxy like tools/g17regs.py, with one addition the package deliberately does not
carry: the `--check` corpus diagnostic walks g17context, which is an experiment module, so the
package takes that stream as a PARAMETER and this legacy entry supplies it. A diagnostic-only
dependency may live outside the package; an authoring dependency may not hide here.
"""
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agxforge.g17.gpumd as _impl


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
    _argv = sys.argv[1:]
    _streams = None
    if "--check" in _argv:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import g17context
        _streams = g17context.walk()
    sys.exit(_impl.main(_argv, streams=_streams))