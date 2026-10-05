"""One loader for `isa/g17.yaml`, because parsing it is most of several modules' runtime.

The spec is 49 MB.  `yaml.safe_load` takes 29s on it; libyaml's `CSafeLoader` takes 5.7s for a dict
that compares equal - 6,718 keys, checked rather than assumed, by `_verified_equal_once` below and
by a test that loads both ways.

Memoised per path, because a module that reads the spec at three sites was paying three parses.
Collapsing those to one took `test_g17isamap` from 140.4s to 47.4s at an unchanged 119 tests.

**Why this is shared rather than copied.**  `test_g17isamap` had the technique in a local helper,
with a comment saying libyaml is 5.9s against 24.6s - and three call sites in the same file still
went through `yaml.safe_load`.  A fix that lives in one module's private helper stops at that
module's edge, and worse, the comment makes the file look converted to anyone grepping for the
loader name.  One implementation, imported.

**The returned object is SHARED and must be treated as read-only.**  Callers that need to withhold
part of the spec should build a new mapping over the same values rather than mutating this one.
"""
import os

_CACHE = {}


def _loader():
    import yaml
    return getattr(yaml, "CSafeLoader", yaml.SafeLoader)

# There is deliberately NO runtime "verified equal" hook here. The first version of this file had
# one; it set a flag and compared nothing, so it was a control whose failure was impossible - and
# comparing on every process would cost the 29s this module exists to avoid. The check is real and
# it lives in the suite, which calls `equal_under_both_loaders` below.


def load(path, root=None):
    """Parse a YAML or JSON artifact once per process and return the shared object.

    `path` may be absolute or relative to `root` (default: the repository root).
    """
    import json
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # realpath, not the raw string: `isa/g17.yaml` and `<root>/test/../isa/g17.yaml` name one
    # file, and keying on the spelling gave them separate cache entries - two parses of the same
    # 49 MB, which is the exact cost this module exists to remove. Caught by its own test.
    full = os.path.realpath(path if os.path.isabs(path) else os.path.join(root, path))
    if full not in _CACHE:
        if full.endswith((".yaml", ".yml")):
            import yaml
            with open(full) as handle:
                _CACHE[full] = yaml.load(handle, Loader=_loader())
        else:
            with open(full) as handle:
                _CACHE[full] = json.load(handle)
    return _CACHE[full]


def equal_under_both_loaders(path):
    """Load `path` with the reference loader and the fast one and report whether they agree.

    This is the check the docstring's claim rests on, exposed so a test can run it rather than
    trusting the sentence. Returns None when libyaml is unavailable.
    """
    import yaml
    if _loader() is yaml.SafeLoader:
        return None
    with open(path) as handle:
        slow = yaml.load(handle, Loader=yaml.SafeLoader)
    with open(path) as handle:
        fast = yaml.load(handle, Loader=yaml.CSafeLoader)
    return slow == fast
