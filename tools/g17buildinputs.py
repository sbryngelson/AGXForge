#!/usr/bin/env python3
"""WHAT A BUILD ACTUALLY TOUCHES, asked of the running process rather than of its imports.

tools/g17halfimage.py's audit_loaded() hashes every module a build imports against the HEAD of
whichever repo it came from. That is the right question about the wrong half of the closure for the
defect that started this workstream: isa/g17-operand-maps-store.jsonl is DATA, and no module audit
can see it. The compiler owner found the other half by wrapping a real compile, and found their
encoder forking an untracked binary to derive field bases - so the emitted bytes depended on a tool
no HEAD pins.

So this wraps `open` and `subprocess.Popen` around a callable and reports what it read and what it
ran, classified against git rather than against a path prefix:

    TRACKED AND CLEAN    in HEAD of a known repo and identical on disk
    TRACKED, MODIFIED    in HEAD and different
    UNTRACKED            inside a repo, invisible to `git ls-files` - the gap a repo-prefix test
                         misses entirely, because the path looks local and is pinned by nothing
    EXTERNAL             outside every known repo

DETERMINES VERSUS DESCRIBES, kept apart rather than collapsed. A subprocess that decides emitted
bytes breaks reproducibility; one that reads bytes already chosen does not. An artifact checker
that decodes what it just built is not the same finding as a builder that decodes to decide, and
reporting them as one number would make the first look like the second.

    python3 tools/g17buildinputs.py            audit the FP16 build and each gate
"""
import builtins
import contextlib
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The compiler used to live in a second checkout at a fixed path on one machine; it is in this repository now, so the
# search is this checkout only. (The old path no longer exists; keeping it made the tool machine-specific.)
REPOS = (ROOT,)


def source_of(path):
    """A compiled helper is pinned by its SOURCE, not by being committed itself.

    tools/agx3dis is a Mach-O binary, gitignored next to *.o and *.dylib, and rebuilt by
    g17ref.binary() whenever tools/agx3dis.c is newer. Calling that "untracked, pinned by nothing"
    is the wrong reading: it is pinned by a tracked source, and committing the binary would be the
    actual mistake. So a build product resolves to its source and is judged on that.

    What tracking cannot pin is the toolchain: agx3dis.c links Apple's MCDisassembler from the
    local SDK, so the same source on another machine or OS version can decode differently. That is
    a real limit and no amount of git fixes it - it is the same "one instrument" caveat the two
    checkers share, and it belongs in the scope statement rather than in a gate.
    """
    for ext in (".c", ".cc", ".cpp", ".m"):
        if os.path.exists(path + ext):
            return path + ext
    return None


def classify(path):
    path = os.path.abspath(path)
    src = source_of(path)
    if src is not None:
        kind, repo, rel = classify(src)
        if kind == "TRACKED AND CLEAN":
            fresh = os.path.getmtime(path) >= os.path.getmtime(src)
            return (("BUILT FROM TRACKED SOURCE" if fresh else "BUILT, STALE VS SOURCE"), repo, rel)
        return kind, repo, rel
    for repo in REPOS:
        if not path.startswith(os.path.abspath(repo) + os.sep):
            continue
        rel = os.path.relpath(path, repo)
        listed = subprocess.run(["git", "-C", repo, "ls-files", "--error-unmatch", rel],
                                capture_output=True)
        if listed.returncode != 0:
            # An extracted experiment is owned by its committed archive member.
            # Do not trust either the loose cache or a locally modified archive.
            import g17buildaudit as audit
            try:
                expected = audit.committed_hashes(repo, "HEAD", [rel])[rel]
                with open(path, "rb") as stream:
                    same = audit.sha(stream.read()) == expected
                return ("TRACKED AND CLEAN" if same else "TRACKED, MODIFIED"), repo, rel
            except (ValueError, OSError):
                return "UNTRACKED", repo, rel
        blob = subprocess.run(["git", "-C", repo, "cat-file", "-p", "HEAD:" + rel],
                              capture_output=True)
        if blob.returncode != 0:
            return "UNTRACKED", repo, rel
        try:
            disk = open(path, "rb").read()
        except OSError:
            return "TRACKED, UNREADABLE", repo, rel
        same = hashlib.sha256(blob.stdout).digest() == hashlib.sha256(disk).digest()
        return ("TRACKED AND CLEAN" if same else "TRACKED, MODIFIED"), repo, rel
    return "EXTERNAL", None, path


@contextlib.contextmanager
def watch():
    reads, runs = set(), []
    real_open, real_popen = builtins.open, subprocess.Popen

    def spy_open(file, mode="r", *a, **k):
        try:
            if isinstance(file, (str, bytes, os.PathLike)) and "w" not in str(mode) \
               and "a" not in str(mode) and "x" not in str(mode):
                reads.add(os.fspath(file))
        except Exception:
            pass
        return real_open(file, mode, *a, **k)

    class SpyPopen(real_popen):
        def __init__(self, args, *a, **k):
            runs.append(args[0] if isinstance(args, (list, tuple)) and args else args)
            super().__init__(args, *a, **k)

    builtins.open, subprocess.Popen = spy_open, SpyPopen
    try:
        yield reads, runs
    finally:
        builtins.open, subprocess.Popen = real_open, real_popen


def audit(fn, label):
    with watch() as (reads, runs):
        result = fn()
    rows = {}
    for p in sorted(reads):
        if not os.path.isfile(p):
            continue
        rows.setdefault(classify(p)[0], []).append(p)
    procs = {}
    for r in runs:
        r = r if isinstance(r, str) else str(r)
        if r == "git":
            continue
        procs.setdefault(classify(r)[0] if os.path.sep in r else "EXTERNAL", []).append(r)
    print("\n%s" % label)
    for kind in ("TRACKED AND CLEAN", "BUILT FROM TRACKED SOURCE", "TRACKED, MODIFIED",
                 "BUILT, STALE VS SOURCE", "UNTRACKED", "EXTERNAL"):
        if rows.get(kind):
            print("   read  %-20s %d" % (kind, len(rows[kind])))
            for p in rows[kind][:4] if kind != "TRACKED AND CLEAN" else []:
                print("            %s" % os.path.relpath(p, ROOT))
    if not procs:
        print("   forked                         0")
    for kind, items in procs.items():
        print("   forked %-20s %d   %s" % (kind, len(items),
                                           ", ".join(sorted(set(os.path.basename(i) for i in items)))))
    bad = (rows.get("UNTRACKED", []) + rows.get("TRACKED, MODIFIED", [])
           + rows.get("EXTERNAL", []) + rows.get("BUILT, STALE VS SOURCE", []))
    return result, bad, procs


def main():
    import g17halfimage as A
    A.pin_compiler_modules()
    print(__doc__.split("\n\n")[0])
    _r, bad, procs = audit(lambda: A.build(), "CODE GENERATION - what decides the emitted bytes")
    determines = [k for k in procs if k != "TRACKED AND CLEAN"]
    print("\n   verdict: %s" % ("every input tracked and clean, nothing forked"
                                if not bad and not procs else
                                "%d unpinned inputs, %s" % (len(bad), procs or "no subprocess")))
    _r2, bad2, procs2 = audit(lambda: A.gate_code(A.BUNDLE),
                              "THE CODE GATE - what decides the verdict on delivered bytes")
    print("\n   the gate DESCRIBES bytes already chosen, so a fork here does not make the ARTIFACT")
    print("   unreproducible. It makes the VERDICT unreproducible, which is its own finding.")
    return 2 if (bad or procs or bad2 or procs2) else 0


if __name__ == "__main__":
    sys.exit(main())
