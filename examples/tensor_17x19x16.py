"""Workflow 1: compile the technical reference's 17 x 19 x 16 tensor product (chapter 15, "The running example").

    python3 examples/tensor_17x19x16.py OUT_DIR

COMPILES ONLY. Nothing is dispatched and the product is never computed: this shows the IR, the G17 instructions the
compiler emits, the executor contract (ABI) and the image the Metal path would load. Compilation needs Apple's G17
decoder: the compiler's release checks decode the bytes it emits with Apple's decoder (GPUCompiler.framework on macOS,
reached through tools/agx3dis, built by `make native-tools`), so this runs only on macOS with Xcode installed.

The reference states the expected program: 1,232 bytes, 101 instructions, SHA-256 8bb67a3c...c667, 76 registers,
ABI version 5. The script checks those facts and exits non-zero if any differs.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agxforge.g17 import cc, ir, model, scanlink   # noqa: E402

EXPECTED = {"bytes": 1232, "instructions": 101, "sha256": "8bb67a3c65919cb7fa9a6f2e70d152cb2d8086fba0cade7437f18d1d8114c667",
            "abi_version": 5}


def build():
    """The program of reference section 15.2: C = A.B for a 17 x 19 x 16 half-precision product, then C[i] += 1."""
    a = ir.Buffer("A", 1, elem=ir.F16)
    b = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_example", [a, b, c])
    bd = ir.Builder(fn, fn.block("entry"))
    bd.tensor_matmul(a, b, c, M=17, N=19, K=16)
    i = bd.builtin("threadgroup_position_in_grid")
    one = struct.unpack("<I", struct.pack("<f", 1.0))[0]
    bd.store_at(c, i, bd.fadd(bd.load(c, i, type=ir.F32), bd.const(one)))
    bd.ret()
    ir.verify(fn)
    return fn


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    fn = build()
    (out / "ir.txt").write_text(repr(fn) + "\n")

    program = cc.compile_function(fn)
    code = bytes(program.code)
    (out / "program.bin").write_bytes(code)
    glossary = {e["op"]: e["name"] for e in json.loads((ROOT / "isa" / "g17-opcode-glossary.json").read_text())["opcodes"]}
    listing = []
    for inst in model.decode(code):            # Apple's decoder, through tools/agx3dis
        op = inst.opcode.id if inst.opcode else None
        listing.append("%s    ; %s" % (inst, glossary.get(op, "no glossary entry")))
    (out / "instructions.txt").write_text("\n".join(listing) + "\n")

    abi = program.abi()
    (out / "abi.json").write_text(json.dumps(abi, indent=1, default=lambda v: dict(v) if hasattr(v, "keys") else str(v)) + "\n")

    image = scanlink.author(program)           # object, metadata, library and archive for the Metal path
    parts = {}
    for part in ("object", "library", "archive"):
        blob = bytes(getattr(image, part))
        (out / ("image." + part)).write_bytes(blob)
        parts[part] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}

    facts = {"bytes": len(code), "instructions": len(listing), "sha256": hashlib.sha256(code).hexdigest(),
             "abi_version": abi["abi_version"], "register_count": abi.get("register_count"),
             "bindings": [dict(b) for b in abi["bindings"]], "image": parts,
             "executed": False, "note": "compiled and decoded on the CPU; nothing was dispatched"}
    (out / "summary.json").write_text(json.dumps(facts, indent=1) + "\n")
    print(repr(fn))
    print("\n".join(listing[:12] + ["... (%d instructions; full listing in instructions.txt)" % len(listing)]))
    print(json.dumps({k: facts[k] for k in ("bytes", "instructions", "sha256", "abi_version", "register_count")}, indent=1))
    print("image:", json.dumps(parts))
    wrong = {k: (facts[k], v) for k, v in EXPECTED.items() if facts[k] != v}
    if wrong:
        print("DIFFERS FROM THE REFERENCE:", wrong)
        return 1
    print("matches the technical reference, section 15.2 (compiled, not executed)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
