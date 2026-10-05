#!/usr/bin/env python3
"""One owner for every derived cache in the class pipeline: stamping, locking, atomic writes.

WHY THIS FILE EXISTS. Three caches were added in one session - the decoded instruction facts, the
per-tag Metal source, the contract class tables - and a cache is exactly where a wrong answer
survives a fix. This project has been bitten twice already: a donor table cached with no
invalidation at all, and then one invalidated on the corpus SIZE, which is the half that was never
the problem. The second was worse than the first because it looked correct.

The failure mode is specific and quiet. Every function that decides a contract key lives in a
handful of modules; edit one, and every cached key computed the old way is still on disk and still
looks fresh. The score that comes back is a measurement of a model that no longer exists, and
nothing reports it, because a stale cache does not raise - it answers.

So the stamp is not the corpus. It is the corpus AND the source of every module that can change
what a key means, hashed, and any of them moving invalidates everything.

    stamp()                 what the caches are a function of
    load(name) / save()     stamped, atomically written, lock-protected pickles
    source_text(tag)        the per-tag Metal source, read once per process
    check_fresh(name)       assert a cache on disk agrees with this stamp

WHAT THIS DOES NOT DO. It does not make a cache correct - only current. A cache whose BUILDER is
wrong is wrong at every stamp, which is what the regression suite is for.
"""
import hashlib
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# THE ROOT ANCHOR IS TWO DIRECTORIES UP FROM agxforge/g17/, not one. Every path in KEY_SOURCES is
# joined to this, so getting it wrong does not raise: os.stat fails, each entry hashes as
# "missing", and the stamp becomes a constant that no source change can move - a cache that never
# invalidates. Measured, not reasoned: with the one-step anchor, touching agxforge/g17/facts.py left
# the stamp identical.
ROOT = os.path.dirname(os.path.dirname(HERE))
CACHEDIR = os.path.expanduser("~/.cache/agxforge")

# EVERY MODULE THAT CAN CHANGE WHAT A CONTRACT KEY MEANS. g17classgen holds written_of, read_of,
# signed_stores, barrier_kinds, bound_indices and contract_inputs; g17classbytes holds
# source_signature, which is half the key and was NOT in the stamp until this file existed -
# editing it changed every key in the corpus and invalidated nothing. g17mdgen parses the section
# the key is read from, g17gpumd finds the per-kernel table, and agx3dis.c is the decoder whose
# output becomes the instruction count and the loop flag.
KEY_SOURCES = (
    # THIS FILE IS IN ITS OWN STAMP, and leaving it out cost a silent wrong answer within an hour
    # of writing it. corpus_keys started storing (key, shape) instead of a bare key; the stamp did
    # not move, because only g17classgen and friends were watched; the next run loaded the OLD
    # pickle and read the first element of the key as if it were the key. g17witness.groups()
    # reported 219 groups where there are 1,771 and singletons() reported none where there are 91,
    # and nothing raised - it just answered, which is what a stale cache always does.
    #
    # The module that defines the cache FORMAT belongs in the stamp for the same reason as the
    # modules that define what a key means. A cache layer that cannot invalidate itself is the
    # defect it was written to prevent.
    "agxforge/g17/cache.py",
    # AND THE WITNESS LIST, because it decides CORPUS MEMBERSHIP. in_corpus admits a syn- tag only
    # when the repo file names it, so appending a witness changes which kernels the corpus keys
    # describe - the same reason the corpus size is already stamped. A file that changes the
    # population belongs in the stamp exactly as much as one that changes the key.
    "isa/g17-synth-witnesses.txt",
    "agxforge/g17/classgen.py",
    # AND g17facts, WHICH DEFINES KEY COMPONENTS AND WAS NOT HERE. contract_key calls
    # written_from_md, bound_from_md and resource_records out of it, so a change to any of those
    # readers changes what every cached key MEANS while leaving the stamp untouched - the exact
    # failure the comment above describes, one module over. Found by hitting it: correcting
    # resource_records to count promoted ranges rather than five-field records moves 51.5% of
    # Apple's sections and would have loaded the old pickle and answered from it.
    # AND THE PATHS FOLLOW THE IMPLEMENTATIONS, WHICH THE LIBRARY MIGRATION BROKE. facts, mdgen
    # and gpumd moved into agxforge.g17 and their tools/ paths became forwarding shims, so this stamp
    # went on hashing ~35-line proxies that never change while the readers it exists to track moved
    # underneath it. Measured rather than reasoned: touching agxforge/g17/facts.py left the stamp
    # identical, and touching the shim moved it - exactly backwards. That is the failure the
    # comment above describes, reintroduced by a file move rather than by an edit.
    "agxforge/g17/facts.py",
    "agxforge/g17/classbytes.py",
    "agxforge/g17/mdgen.py",
    "agxforge/g17/gpumd.py",
    "tools/agx3dis.c",
)


def _corpus_dir():
    from . import metal as g17metal
    return g17metal.CACHE


def stamp():
    """What every derived cache here is a function of: the corpus, and the code that reads it."""
    # THE STAMP HASHES CONTENT, NOT stat. It used (int(st_mtime), st_size), and integer-second
    # timestamps plus a size are not an identity: an edit that preserves both - the same number of
    # bytes written back inside the same second, or a timestamp restored after an edit, which is
    # what a checkout, a patch tool and this project's own test helpers all do - left the stamp
    # unchanged and every derived cache answered from a pickle computed by different code. Root
    # pinned it with a control that rewrites a file to the same length and restores its mtime.
    #
    # The cost stays bounded: these are a dozen source files, read once per stamp() call, and a
    # warm cache calls it once. Reading them is cheaper than the corpus listing already below it.
    h = hashlib.blake2b(digest_size=16)
    for rel in KEY_SOURCES:
        p = os.path.join(ROOT, rel)
        try:
            with open(p, "rb") as handle:
                content = handle.read()
            h.update(("%s:%d:" % (rel, len(content))).encode())
            h.update(hashlib.blake2b(content, digest_size=16).digest())
            h.update(b";")
        except OSError:
            h.update(("%s:missing;" % rel).encode())
    try:
        n = len(os.listdir(_corpus_dir()))
    except OSError:
        n = -1
    return (n, h.hexdigest())


def _path(name):
    return os.path.join(CACHEDIR, "g17-%s.pkl" % name)


class _Lock(object):
    """An advisory lock so two runs do not both pay for the same rebuild, and never interleave.

    Best-effort by design: if flock is unavailable or the lock cannot be taken, the work is done
    anyway and the write is still atomic. A lock that can fail closed would turn a slow cache into
    a broken tool.
    """

    def __init__(self, name):
        self.p = os.path.join(CACHEDIR, ".%s.lock" % name)
        self.fh = None

    def __enter__(self):
        try:
            os.makedirs(CACHEDIR, exist_ok=True)
            import fcntl
            self.fh = open(self.p, "w")
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
        except Exception:
            self.fh = None
        return self

    def __exit__(self, *a):
        if self.fh is not None:
            try:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            finally:
                self.fh.close()
                self.fh = None


def load(name, want=None):
    """The cached object for `name` if it was built at this stamp, else None."""
    if os.environ.get("REBUILD_CLASSES") == "1":
        return None
    want = stamp() if want is None else want
    try:
        with open(_path(name), "rb") as fh:
            got = pickle.load(fh)
    except Exception:
        return None
    if isinstance(got, tuple) and len(got) == 2 and got[0] == want:
        return got[1]
    return None


def save(name, obj, want=None):
    """Write atomically, so a reader never sees a partial file and a crash leaves the old one."""
    want = stamp() if want is None else want
    try:
        os.makedirs(CACHEDIR, exist_ok=True)
        tmp = _path(name) + ".%d" % os.getpid()
        with open(tmp, "wb") as fh:
            pickle.dump((want, obj), fh, 2)
        os.replace(tmp, _path(name))
        return True
    except Exception:
        return False


def check_fresh(name):
    """(exists, fresh) for a cache on disk - what a regression asserts rather than assumes."""
    p = _path(name)
    if not os.path.exists(p):
        return False, False
    try:
        with open(p, "rb") as fh:
            got = pickle.load(fh)
    except Exception:
        return True, False
    return True, bool(isinstance(got, tuple) and len(got) == 2 and got[0] == stamp())


_SRCTXT = {}


def source_text(tag):
    """The kernel's Metal source, read once per process.

    NINE OPENS PER KERNEL before this. written_of, read_of, signed_stores, barrier_kinds,
    bound_indices and _arg_types each opened and re-parsed the same file, and source_signature
    opened it again - 113,491 opens across a 12,564-kernel class-table rebuild, 1.1 s of open()
    inside 11.7 s. Read once, shared, because the string is immutable and the file does not change
    while a process runs. The rebuild is 5.5 s.

    Not persisted: the win is per-process and a stale copy on disk would be a new way to be wrong.
    """
    t = _SRCTXT.get(tag)
    if t is None:
        try:
            with open(os.path.join(_corpus_dir(), tag, "s.metal"), errors="replace") as fh:
                t = fh.read()
        except OSError:
            t = ""
        _SRCTXT[tag] = t
    return t


_KEYS = None


def corpus_keys():
    """{tag: contract key} for the whole corpus - a view on corpus_entries()."""
    return {d: v[0] for d, v in corpus_entries().items()}


def corpus_entries():
    """{tag: (contract key, shape)} for the whole corpus, built once and shared by every tool.

    THE SHAPE RIDES ALONG because the witness generator needs it and computing it separately
    means walking the corpus twice. One cache, two views - corpus_keys() for the tools that only
    group by key, corpus_entries() for the ones that also compare structure.

    THE SAME WALK, FOUR TIMES OVER. g17synth builds it to rank synthesis targets, g17classholdout
    to score the holdout, the refusal ledger to categorise refusals, and the surface scorer to
    group kernels - each one opening every object file, decoding every program and recomputing
    every key. The keys are a pure function of the same inputs the stamp already covers, so they
    are computed once and cached like everything else here.

    Held on disk under the same stamp, so an edit to any key-defining module rebuilds it rather
    than handing back keys computed by the previous definition.
    """
    global _KEYS
    if _KEYS is not None:
        return _KEYS
    got = load("corpus-keys")
    if got is not None:
        _KEYS = got
        return _KEYS
    from . import classbytes as CB
    from . import classgen as G
    from . import mdgen as M
    from . import obj as g17obj
    out = {}
    for d in sorted(os.listdir(_corpus_dir())):
        sig = CB.source_signature(d)
        md = CB.metadata(d) if sig else None
        obj = os.path.join(_corpus_dir(), d, "out", "object", "0-0")
        if not md or not os.path.exists(obj):
            continue
        md = bytes(md)
        try:
            desc = M.describe(md)
            raw = open(obj, "rb").read()
            sects, syms = g17obj.sections_of(raw)
            off, size = sects["__TEXT,__text"]
            ni, loop = G.code_facts(bytes(raw[off:off + size]), syms["_agc.main"])
            out[d] = (G.key_for(d, md, desc, sig, ni, loop), G.shape_of(desc))
        except Exception:
            continue
    _KEYS = out
    with _Lock("corpus-keys"):
        save("corpus-keys", out)
    G._save_code_facts()
    return _KEYS


def main():
    n, h = stamp()
    print("stamp: %d corpus entries, %s" % (n, h))
    for name in ("contract-classes", "skeletons", "code-facts", "corpus-keys"):
        exists, fresh = check_fresh(name)
        print("  %-18s %-9s %s" % (name, "present" if exists else "absent",
                                   "FRESH" if fresh else ("STALE" if exists else "-")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
