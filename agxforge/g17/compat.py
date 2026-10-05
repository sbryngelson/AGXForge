"""One forwarding mechanism for the legacy entry points, instead of eleven copies of it.

Each module in tools/ whose production half moved into this package needs the same three things,
and each of them was learned from a defect:

  READS stay bound in the legacy module. A module's `__getattr__` does not fire for that module's
  OWN bare global lookups, and the retained probe code reads these names that way - g17const reads
  HERE at module level, the form modules read encode and LENGTH. Removing the bindings broke five
  modules at import.

  WRITES reach the implementation. `NAME = _impl.NAME` binds a second name to one object; it does
  not connect them, so rebinding the legacy name changed an idle copy while the implementation read
  its own global. That is the mutation control root's rebinding case measures.

  THE LOCAL BINDING IS UPDATED ON WRITE, not deleted. Deleting it made `module.NAME` read through
  correctly and broke the probe code in the same file, because a bare global lookup never consults
  __getattr__: g17const.main() raised NameError on load() after any patch.

The eleven copies of this were 834 lines. This module holds it once; a legacy entry calls install()
and passes the names it forwards. Nothing here imports tools or touches sys.path - establishing the
repository import root is the legacy entry's job, ahead of importing this package at all.
"""
import sys
import types


def install(module_name, implementation, names=None):
    """Make `module_name` forward to `implementation`, and return the values to bind.

    `names` maps the legacy name to the implementation's name for it - they differ where several
    forms share one module, as the six form encoders do (`BASE` -> `_Fadd4_BASE`). Pass a plain
    iterable when the names are the same.

    Pass None for a module that moved WHOLE rather than being split. There is no name list to keep
    then, every attribute forwards, and nothing is bound back into the legacy module - it retains
    no code of its own that could read a bare global, which is the only reason the binding half
    exists for the split modules.
    """
    if names is None:
        return _install_all(module_name, implementation)
    if not isinstance(names, dict):
        names = {name: name for name in names}
    module = sys.modules[module_name]

    class _Delegating(types.ModuleType):
        def __getattr__(self, name):
            target = names.get(name)
            if target is None:
                raise AttributeError(name)
            return getattr(implementation, target)

        def __setattr__(self, name, value):
            target = names.get(name)
            if target is None:
                return object.__setattr__(self, name, value)
            setattr(implementation, target, value)
            self.__dict__[name] = value          # the probe code reads this as a bare global

        def __delattr__(self, name):
            target = names.get(name)
            if target is None:
                return object.__delattr__(self, name)
            delattr(implementation, target)
            self.__dict__.pop(name, None)

    module.__class__ = _Delegating
    _mirror(implementation, module, names)
    return {name: getattr(implementation, target) for name, target in names.items()}


def _mirror(implementation, legacy, names):
    """Make a write to `implementation` update `legacy`'s bindings for the names it forwards.

    THE OTHER DIRECTION, and the one the first version missed. A legacy entry holds real bindings
    so its retained probe code can read them as bare globals - and those bindings are a SNAPSHOT.
    Patching the implementation left `g17const.load` and `main()`'s bare `load` pointing at the old
    callable, which root's PackageToLegacyMutation case reproduces. Forwarding writes one way is
    not forwarding.

    The registration comes from the legacy side and hands over a module object. The implementation
    never imports tools and nothing here touches sys.path: what it gains is a list of mirrors it
    was given, which is the only shape that keeps the dependency pointing the right way.
    """
    inverted = {target: name for name, target in names.items()}
    registry = getattr(implementation, "_compat_mirrors", None)
    if registry is None:
        registry = []
        object.__setattr__(implementation, "_compat_mirrors", registry)

        class _Mirrored(types.ModuleType):
            def __setattr__(self, name, value):
                object.__setattr__(self, name, value)
                for mirror, mapping in registry:
                    legacy_name = mapping.get(name)
                    if legacy_name is not None:
                        mirror.__dict__[legacy_name] = value

            def __delattr__(self, name):
                object.__delattr__(self, name)
                for mirror, mapping in registry:
                    legacy_name = mapping.get(name)
                    if legacy_name is not None:
                        mirror.__dict__.pop(legacy_name, None)

        implementation.__class__ = _Mirrored
    registry.append((legacy, inverted))


def _install_all(module_name, implementation):
    """Forward every attribute, for a module whose implementation moved entirely."""
    module = sys.modules[module_name]

    class _ForwardingAll(types.ModuleType):
        def __getattr__(self, name):
            return getattr(implementation, name)

        def __setattr__(self, name, value):
            if name.startswith("__"):
                return object.__setattr__(self, name, value)
            setattr(implementation, name, value)

        def __delattr__(self, name):
            if name.startswith("__"):
                return object.__delattr__(self, name)
            delattr(implementation, name)

        def __dir__(self):
            return dir(implementation)

    module.__class__ = _ForwardingAll
    return {}


# THE PACKAGE WAS RENAMED on 2026-10-04: `triad` became `agxforge`. Receipts, inventories and reports written
# before then name files under the old directory, and they are records, so they keep that spelling. A reader
# that resolves a recorded path to a file goes through current_path(); one that matches recorded paths
# against a current one also accepts legacy_spellings().
LEGACY_PACKAGE = "tri" "ad/"     # split so a rename sweep over this tree cannot rewrite the old name
PACKAGE = "agxforge/"


def current_path(path):
    """Where a recorded repository path lives now: the old package prefix maps to the new one."""
    text = str(path)
    return PACKAGE + text[len(LEGACY_PACKAGE):] if text.startswith(LEGACY_PACKAGE) else text


def legacy_spellings(path):
    """The spellings a record written before the rename may use for the current `path`."""
    text = str(path)
    return (LEGACY_PACKAGE + text[len(PACKAGE):],) if text.startswith(PACKAGE) else ()


def source_of(path, root=None):
    """The file that actually holds `path`'s implementation, following a forwarding entry.

    A test or an inventory that reads implementation SOURCE through a legacy path reads a
    forwarding module once that implementation moves - and gets nothing. It does not error; the
    pattern simply is not found, so a structural scan returns an empty set and whatever depended on
    it reports "no such form" instead of failing. Three readers hit this: the compiler inventory's
    `contains "add"` check, test_g17fieldmax reading FIELD_MAX, and g17refusedforms scanning g17cc
    for MInst declarations and resolving registry citations.

    `path` is repository-relative. Returns it unchanged when it is not a forwarding entry, so a
    caller can use this unconditionally and a module that has not moved is unaffected.
    """
    import ast
    import os

    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = current_path(path)
    full = os.path.join(root, path)
    if not os.path.isfile(full):
        return path
    try:
        tree = ast.parse(open(full, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return path
    for node in ast.walk(tree):
        # `from agxforge.g17.ir import *` - a star import IS the statement that this file re-exports
        # that module. Recognising only the `_impl` alias let g17ir and g17evidence resolve to
        # THEMSELVES, and a caller cannot tell that from "this module never moved": the function
        # fails open by design, so every spelling it misses is a silent wrong answer.
        if (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("agxforge.g17.")
                and any(alias.name == "*" for alias in node.names)):
            candidate = os.path.join("agxforge", "g17", node.module.split(".")[-1] + ".py")
            if os.path.isfile(os.path.join(root, candidate)):
                return candidate
        if isinstance(node, ast.ImportFrom) and node.module == "agxforge.g17":
            for alias in node.names:
                if alias.asname in ("_impl", "_formenc"):
                    candidate = os.path.join("agxforge", "g17", alias.name + ".py")
                    if os.path.isfile(os.path.join(root, candidate)):
                        return candidate
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname == "_impl" and alias.name.startswith("agxforge.g17."):
                    candidate = os.path.join("agxforge", "g17",
                                             alias.name.split(".")[-1] + ".py")
                    if os.path.isfile(os.path.join(root, candidate)):
                        return candidate
    return path
