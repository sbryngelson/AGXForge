#!/usr/bin/env python3
"""Apple's AGX3 decoder, wrapped as a walk, so tools/g17dis.py can be checked against it.

g17dis walks by rules this project measured; this walks by the rules Apple's compiler encoded
with. On the CORPUS, where they disagree, Apple is the ground truth: the decoder is generated
from the same instruction description that produced those bytes. On bytes this project AUTHORED
it is not ground truth in either direction - see docs/agx3-oracle.md section 12.

The decoder is not exposed by any shipping tool. tools/agx3dis.c explains how it is reached
and what it does not give: no mnemonics, because instruction names are compiled out of that
LLVM, so an instruction is identified by a numeric opcode id out of 17796.

Use it as a check, not as a replacement. g17dis carries the field models, the confidence
classes and the provenance; this carries only lengths, opcode ids and raw operands.

    python3 tools/g17ref.py <applegpu-object>       walk _agc.main and report
    python3 tools/g17ref.py --check                 compare against g17dis over the cache
"""
import functools
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# ANCHORED ON THE CHECKOUT ROOT. Under tools/ this file sat one level below the checkout and
# beside the native helpers; under agxforge/g17/ it is two levels below and the helpers do not move -
# the Makefile keeps building them into tools/.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
# THE BUILD TARGET STAYS IN tools/. This compiles agx3dis on demand, so the anchor decides
# where a BINARY lands; under agxforge/g17/ it would build into the library from a source that
# is not there. Same hazard batch 3 corrected for agx3meta.
SRC = os.path.join(TOOLS, "agx3dis.c")
BIN = os.path.join(TOOLS, "agx3dis")


class Undecodable(Exception):
    """The decoder rejected the bytes at this offset. It does not guess a stride, and neither
    do we: the walk stops where Apple's decoder stops."""
    def __init__(self, off):
        self.off = off
        super().__init__("Apple's decoder rejected the bytes at +0x%03x" % off)


class _DecoderFailure(RuntimeError):
    """A native failure carrying stdout that the historical generator still yields first."""
    def __init__(self, message, stdout):
        super().__init__(message)
        self.stdout = stdout


def binary():
    """Path to the built decoder, compiling it if the source is newer."""
    # the renumbering headers it includes count as source (tools/agx3remap.h, tools/agx3renumber.h)
    newest = max(os.path.getmtime(p) for p in (SRC, os.path.join(TOOLS, "agx3remap.h"),
                                                os.path.join(TOOLS, "agx3renumber.h")))
    if not os.path.exists(BIN) or newest > os.path.getmtime(BIN):
        subprocess.run(["clang", "-O2", "-o", BIN, SRC], check=True)
    return BIN


@functools.lru_cache(maxsize=1024)
def _walk_output(t, start, limit):
    """Run the native decoder once for one immutable stream and walk range.

    The textual output is safe to reuse inside this process; ``walk`` parses it anew for every
    caller and therefore never shares mutable rows.  Only successful native invocations are
    cached: a decoder failure raises before an entry is stored.
    """
    end = len(t) if limit is None else min(len(t), start + limit)
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(t)
        f.flush()
        r = subprocess.run([binary(), f.name, str(start), str(end - start), "--pc", str(start)],
                           capture_output=True, text=True)
    if r.returncode not in (0, 3):
        # Do not return or cache a failed invocation, but preserve the old walk contract: valid
        # rows already printed by the decoder are yielded before the RuntimeError is raised.
        raise _DecoderFailure("agx3dis failed: %s" % r.stderr.strip(), r.stdout)
    return r.stdout


def walk(t, start=0, limit=None):
    """Yield (offset, length, opcode) for each instruction from `start`, offsets relative to t.

    Signature matches tools/g17dis.walk so the two can be swapped in a comparison. The third
    element differs in kind: g17dis reports a hand-named class, this reports Apple's opcode id.
    """
    t = bytes(t)
    failure = None
    try:
        output = _walk_output(t, start, limit)
    except _DecoderFailure as exc:
        output, failure = exc.stdout, exc
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "bad":
            raise Undecodable(int(parts[0], 16))
        if len(parts) >= 3:
            yield int(parts[0], 16), int(parts[1]), int(parts[2])
    if failure is not None:
        raise failure


def walk_many(texts):
    """[(offset, length, opcode), ...] per text, with ONE decoder invocation for all of them.

    `walk` runs the native decoder once per program: a process spawn plus a dlopen of libLLVM
    before a single instruction is decoded. Measured over the 23,028-entry build cache that setup
    is ~90% of the cost - the corpus walk in the atomic regression case spent 87.8s, of which the
    decoding proper was a few seconds. `--batch` pays the setup once and decodes every program
    with the same disassembler.

    Semantics match `walk` for the ordinary whole-program case (start 0, no limit, no stride).
    A program the decoder rejects yields the instructions decoded before the rejection, and its
    index appears in the returned `failed` set - `walk` raises Undecodable there, and the corpus
    callers all catch and skip, so the set is the same information without the control flow.

    -> (rows, failed) where rows[i] is the list for texts[i].
    """
    texts = [bytes(t) for t in texts]
    if not texts:
        return [], set()
    rows = [[] for _ in texts]
    failed = set()
    with tempfile.TemporaryDirectory() as d:
        manifest = os.path.join(d, "manifest")
        with open(manifest, "w") as fh:
            for i, t in enumerate(texts):
                path = os.path.join(d, "p%d.bin" % i)
                with open(path, "wb") as out:
                    out.write(t)
                fh.write("%s 0 %d 0\n" % (path, len(t)))
        proc = subprocess.run([binary(), "--batch", manifest], capture_output=True, text=True)
    index = None
    for line in proc.stdout.splitlines():
        if line.startswith("=== end "):
            parts = line.split()
            if len(parts) >= 4 and parts[3] != "0":
                failed.add(int(parts[2]) - 1)
            index = None
            continue
        if line.startswith("=== "):
            index = int(line.split()[1]) - 1
            continue
        if index is None:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "bad":
            failed.add(index)
            continue
        if len(parts) >= 3:
            rows[index].append((int(parts[0], 16), int(parts[1]), int(parts[2])))
    return rows, failed


def _check():
    """Corpus-wide framing comparison against g17dis. Reports where the two disagree."""
    import glob, collections
    # THE SIBLINGS ARE PACKAGE MODULES NOW. These lines used to put tools/ and spike/accel/re on
    # sys.path to reach them by bare name; 4a moved agxdis and machobj into the package, so the
    # library imports them directly and adds nothing to the import path.
    from agxforge.g17 import dis as g17dis, machobj, agxdis
    objs = mine_n = apple_n = shared = 0
    first = []
    for d in sorted(glob.glob(os.path.expanduser("~/.cache/agxforge/agx/*"))):
        arc, obj = d + "/s.arc.metallib", d + "/out/object/0-0"
        if not (os.path.exists(arc) and os.path.exists(obj)):
            continue
        try:
            loc = machobj.locate(arc, obj)
            f, sz = agxdis.sections(loc["obj"])
            t = loc["obj"][f:f + sz]
            start = loc["syms"]["_agc.main"]
            mine = [(o, l) for o, l, _ in g17dis.walk(t, start, limit=len(t))]
            ref = [(o, l) for o, l, _ in walk(t, start)]
        except Exception:
            continue
        objs += 1
        mine_n += len(mine)
        apple_n += len(ref)
        shared += len(set(o for o, _ in mine) & set(o for o, _ in ref))
        if mine != ref and len(first) < 5:
            for a, b in zip(mine, ref):
                if a != b:
                    first.append((os.path.basename(d), a, b, t[a[0]:a[0] + 16].hex(" ")))
                    break
    print("objects compared     : %d" % objs)
    print("instructions, g17dis : %d" % mine_n)
    print("instructions, Apple  : %d" % apple_n)
    print("boundaries in common : %d (%.1f%% of Apple's)" % (shared, 100.0 * shared / max(apple_n, 1)))
    for name, a, b, hexs in first:
        print("\n%s\n  g17dis %s  apple %s\n  %s" % (name, a, b, hexs))


def main(argv=None):
    """The command-line entry, callable. It was inline under `if __name__`, so after the move
    neither the shim nor the library could invoke it - the compatibility module imports this file
    rather than executing it. Same defect the auth CLI had at 14a13a34.
    """
    argv = list(sys.argv if argv is None else argv)
    if "--check" in argv:
        _check()
    else:
        from agxforge.g17 import machobj, agxdis
        path = argv[1]
        blob = open(path, "rb").read()
        f, sz = agxdis.sections(blob)
        t = blob[f:f + sz]
        for off, ln, op in walk(t, 0):
            print("%08x %2d op=%-6d %s" % (off, ln, op, t[off:off + ln].hex(" ")))


if __name__ == "__main__":
    main()
