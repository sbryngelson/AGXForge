#!/usr/bin/env python3
"""CPU validation of delivered scan bytes using Apple's decoder, never Metal.

The interpreter consumes decoded machine operands, not SSA or compiler layout.
It checks address arithmetic, register dataflow and FP32 arithmetic; it cannot
model silicon scheduling, undocumented register behaviour, or loader acceptance.
"""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import g17ref
import g17scan
import g17scanlink
import g17verify


# Decodes filled by prefetch(), in this process only, keyed by (bytes, decoder path). A corpus walk
# (g17ffma4/6, g17fmullength) decodes ~6,600 programs three or four times per build(); with only the
# 512-entry LRU below every walk re-spawned the decoder per program - 20,000-26,000 processes, about
# 95% of those modules' test time.
_PREFETCHED = {}


def prefetch(codes):
    """Decode many byte streams with ONE decoder process (agx3dis --batch --expr).

    Each entry is decoded independently over its own [off, off+len) at pc 0, exactly as the
    single-program call does (agx3dis decode_one), so the lines per entry are the same. Only entries
    the decoder finished (rc 0) and _parse accepts are kept; anything else is left for decode() to
    fail on in the usual single-program way.
    """
    decoder = str(g17ref.binary())
    todo = [c for c in dict.fromkeys(bytes(c) for c in codes)
            if (c, decoder) not in _PREFETCHED and c]
    if not todo:
        return 0
    with tempfile.TemporaryDirectory() as d:
        blob, manifest, offs = Path(d) / "codes.bin", Path(d) / "manifest.txt", []
        with open(blob, "wb") as f:
            for c in todo:
                offs.append(f.tell())
                f.write(c)
        manifest.write_text("".join("%s %d %d 0\n" % (blob, o, len(c)) for o, c in zip(offs, todo)))
        r = subprocess.run([decoder, "--batch", str(manifest), "--expr"],
                           check=True, capture_output=True, text=True, timeout=3600)
    kept, entry, lines = 0, None, []
    for line in r.stdout.splitlines():
        if line.startswith("=== end "):
            _, _, n, rc = line.split()
            code = todo[int(n) - 1]
            if int(n) == entry and rc == "0":
                # the batch reports offsets relative to pc 0 of each entry, as the single call does
                try:
                    _PREFETCHED[(code, decoder)] = _parse(lines, code)
                    kept += 1
                except ValueError:
                    pass
            entry, lines = None, []
        elif line.startswith("=== "):
            entry, lines = int(line.split()[1]), []
        else:
            lines.append(line)
    return kept


def _parse(lines, code):
    instructions = []
    cursor = 0
    for line in lines:
        words = line.split()
        off, size, opcode = int(words[0], 16), int(words[1]), int(words[2])
        if off != cursor or size <= 0 or off + size > len(code):
            raise ValueError("decoder did not cover consecutive complete instructions")
        instructions.append((off, size, opcode, tuple(words[3:])))
        cursor += size
    if cursor != len(code) or not instructions or instructions[-1][2] != 684:
        raise ValueError("program must be fully decoded and end with end")
    return tuple(instructions)


@functools.lru_cache(maxsize=512)
def _decode_cached(code, decoder):
    """Decode one immutable byte stream, retaining only successful results per process."""
    hit = _PREFETCHED.get((code, decoder))
    if hit is not None:
        return hit
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(code)
        f.flush()
        r = subprocess.run([decoder, f.name, "0", str(len(code)), "--expr"],
                           check=True, capture_output=True, text=True, timeout=30)
    instructions = []
    cursor = 0
    for line in r.stdout.splitlines():
        words = line.split()
        off, size, opcode = int(words[0], 16), int(words[1]), int(words[2])
        if off != cursor or size <= 0 or off + size > len(code):
            raise ValueError("decoder did not cover consecutive complete instructions")
        instructions.append((off, size, opcode, tuple(words[3:])))
        cursor += size
    if cursor != len(code) or not instructions or instructions[-1][2] != 684:
        raise ValueError("program must be fully decoded and end with end")
    # Keep the cached value immutable so callers cannot corrupt later validations.  ``decode``
    # returns a fresh list for compatibility with the historical API.
    return tuple(instructions)


def decode(code):
    """Decode bytes through Apple's decoder, reusing identical requests in this process only."""
    code = bytes(code)
    decoder = str(g17ref.binary())
    # Rebuild the historical row/list shape while keeping cached rows and their token sequences
    # immutable.  A caller can therefore mutate either level without affecting later checks.
    return [(off, size, opcode, list(tokens))
            for off, size, opcode, tokens in _decode_cached(code, decoder)]


def simulate(instructions, matrix, vector, *, rows, row_ids):
    """Vectorized interpretation for selected rows, including full-grid boundaries."""
    a, x = np.asarray(matrix, np.float32), np.asarray(vector, np.float32)
    ids = np.asarray(row_ids, np.uint32)
    if a.shape != (len(ids), len(x)) or not np.all(ids[:-1] < ids[1:]) or np.any(ids >= rows):
        raise ValueError("sample rows must be sorted, unique, and inside the exact grid")
    regs, writes = {}, {}
    width, vector_base = len(x), rows * len(x)

    def read(token):
        if not token.startswith("reg:") or token not in regs:
            raise ValueError("uninitialized or unexpected register " + token)
        return regs[token].copy()

    def assign(token, value):
        if not token.startswith("reg:"):
            raise ValueError("unexpected destination " + token)
        regs[token] = np.broadcast_to(np.asarray(value, np.uint32), ids.shape).copy()

    def immediate(token):
        if not token.startswith("imm:"):
            raise ValueError("expected immediate " + token)
        return int(token[4:])

    def load(indices):
        ix = indices.astype(np.uint64)
        if np.any(ix >= vector_base + width):
            raise ValueError("input read outside packed allocation")
        out = np.empty(len(ids), np.float32)
        q = ix >= vector_base
        out[q] = x[(ix[q] - vector_base).astype(int)]
        rr, kk = ix[~q] // width, ix[~q] % width
        local = np.searchsorted(ids, rr)
        if np.any(local >= len(ids)) or not np.array_equal(ids[local], rr):
            raise ValueError("load addressed another row's matrix data")
        out[~q] = a[local, kk.astype(int)]
        return out.view(np.uint32)

    for off, size, op, t in instructions:
        if op == 14059 and size == 4:
            if t[1:] != ["imm:1048576", "reg:61", "imm:0"]:
                raise ValueError("unexpected special register or width")
            assign(t[0], ids)
        elif op == 11842 and size == 8:
            if t[1] != "imm:68736253952":
                raise ValueError("unexpected constant width")
            assign(t[0], immediate(t[2]) & 0xFFFFFFFF)
        elif op == 10825 and size == 14:
            if t[1] != "imm:0" or t[6] != "imm:0":
                raise ValueError("unexpected multiply modifiers")
            assign(t[0], read(t[2]) * read(t[4]))
        elif op == 10279 and size == 12:
            if t[1] != "imm:0":
                raise ValueError("unexpected add modifiers")
            assign(t[0], read(t[3]) + np.uint32(immediate(t[2])))
        elif op == 10282 and size == 12:
            if t[1] != "imm:0":
                raise ValueError("unexpected register add modifiers")
            assign(t[0], read(t[2]) + read(t[4]))
        elif op == 12682 and size == 14:
            if t[1:5] != ["imm:137464119296", "imm:2066", "expr:bin(op0,const(0),8)", "imm:0"] or t[6:] != ["imm:0", "imm:0", "imm:4"]:
                raise ValueError("unexpected load width, binding, offset, or scale")
            assign(t[0], load(read(t[5])))
        elif (op, size) in ((3290, 14), (998, 12)):
            mode = "imm:139586437120" if op == 3290 else "imm:146028888064"
            if t[1] != mode or t[3] not in ("imm:16", "imm:32") or t[5] not in ("imm:16", "imm:32"):
                raise ValueError("unexpected float width or modifiers")
            av, bv = read(t[2]).view(np.float32), read(t[4]).view(np.float32)
            result = av * bv if op == 3290 else av + bv
            assign(t[0], result.view(np.uint32))
        elif op == 17229 and size == 8:
            if t[1:5] != ["imm:16", "imm:2066", "expr:bin(op0,const(4),8)", "imm:0"] or t[6:] != ["imm:16", "imm:0", "imm:4"]:
                raise ValueError("unexpected store width, binding, offset, or scale")
            for index, value in zip(read(t[5]).tolist(), read(t[0]).tolist()):
                if not 0 <= index < 2 * rows or index in writes:
                    raise ValueError("out-of-bounds or duplicate output store")
                writes[index] = value
        elif op == 684 and size == 4 and t == ["imm:0"]:
            if off != instructions[-1][0]:
                raise ValueError("early end instruction")
        else:
            raise ValueError(f"unsupported decoded instruction at {off}: {size} op{op} {t}")
    expected = set(ids.tolist()) | {int(i) + rows for i in ids}
    if set(writes) != expected or any(writes[int(i) + rows] != 0x5A17C0DE for i in ids):
        raise ValueError("missing scores or completion markers")
    return np.asarray([writes[int(i)] for i in ids], np.uint32).view(np.float32)


def check_bundle(directory):
    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    for name, want in m["sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != want:
            raise ValueError("bundle hash mismatch: " + name)
    arc, lib = (directory / "scan.arc.metallib").read_bytes(), (directory / "scan.lib.metallib").read_bytes()
    obj = g17scanlink.verify_contract(arc, lib, [(1, 0, False), (2, 2, True)])
    text = g17verify._objsect(obj, "__text")
    code = (directory / "scan.bin").read_bytes()
    abi_check = None
    if m.get("profile", {}).get("name") == "scalar-buffer-two-bindings-measured-v2":
        import g17scalarabi
        abi_check = g17scalarabi.verify_object(obj, code)
    if text[m["entry"]:m["entry"] + len(code)] != code:
        raise ValueError("delivered text differs from the code under test")
    instructions = decode(code)
    rows, columns = m["rows"], m["columns"]
    ids = np.asarray(sorted({i for i in (0, 1, 2, 3, 7, 15, 16, 31, 32, rows - 2, rows - 1) if 0 <= i < rows}), np.uint32)
    rng = np.random.default_rng(1729)
    checks = []
    for kind in ("random", "cancellation", "subnormal", "basis", "signed_zero"):
        a = (rng.standard_normal((len(ids), columns)) * 0.2).astype(np.float16)
        x = (rng.standard_normal(columns) * 0.2).astype(np.float16)
        if kind == "cancellation":
            a[:] = np.where(np.arange(columns) % 2, -0.125, 0.125)
            x[:] = 1
        elif kind == "subnormal":
            a[:] = np.nextafter(np.float16(0), np.float16(1))
            x[:] = 0.5
        elif kind == "basis":
            a[:] = 0
            a[:, -1] = 1
        elif kind == "signed_zero":
            a[:] = np.float16(-0.0)
            x[:] = 1
        got = simulate(instructions, a, x, rows=rows, row_ids=ids)
        # A separate column-wise FP32 reference also detects excess tolerance.
        sequential = np.zeros(len(ids), np.float32)
        for k in range(columns):
            sequential = sequential + a[:, k].astype(np.float32) * np.float32(x[k])
        if not np.array_equal(got.view(np.uint32), sequential.view(np.uint32)):
            raise ValueError("machine interpretation differs from sequential FP32 dot")
        result = g17scan.check_output(a, x, got.astype(np.float16))
        if not result["ok"]:
            raise ValueError("machine interpretation violates numerical contract")
        checks.append({"case": kind, **result})
    report = {"status": "passed", "scope": "Apple-decoded bytes interpreted on CPU; no GPU execution",
              "instructions": len(instructions), "row_ids": ids.tolist(), "checks": checks,
              "archive_sha256": m["sha256"]["scan.arc.metallib"]}
    if abi_check is not None:
        report["abi_check"] = abi_check
    (directory / "cpu-check.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    print(json.dumps(check_bundle(p.parse_args().bundle), indent=2))
