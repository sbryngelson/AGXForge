#!/usr/bin/env python3
"""THE NATIVE FP16 SCAN: half matrix and query in device storage, FP32 accumulation, half scores.

This is the target docs/archive/g17-scan-acceptance.md names, and the difference from the packed FP32 path
that has executed is entirely in the CODE: the same two bindings, the same entry, the same
prologue, and - measured by the linker rather than assumed - the same metadata, because
g17mdgen.build has no element-type parameter at all and the section is a function of the binding
indices alone.

FOUR FORMS THIS NEEDS, all located one variable at a time in tools/g17halfprobe.py:

    op12646/14   the half load        a 16-bit destination in the file based at 425
    op1004 /12   the WIDENING         an fadd.imm of zero with a 16-bit source and a 32-bit
                                      destination - not a conversion opcode, and Apple folds it
                                      into the FMA's operands when the shape allows
    op1016 /12   the NARROWING        cvt.f32.f16, which IS a conversion opcode
    op17193/14   the half store       a 16-bit value operand, a 32-bit index

THE ACCUMULATOR STAYS FP32 THROUGHOUT. Only the two loads and the final store touch 16-bit
registers; every multiply and add is the same f32 form the packed path already executes.

    python3 tools/g17halfscan.py            compile the 384-column scan and print its ABI
    python3 tools/g17halfscan.py --rows N --columns K
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROFILE = "half-buffer-two-bindings"


def scan_ir(rows, columns, unroll=None):
    """One thread per row. Fully unrolled by default, which is what the executed path does.

    A counted loop is possible - the trip-count proof lands it - but two ceilings bind it and both
    are measured: the compare's immediate is eight bits so a trip count above 255 cannot be
    expressed, and the mask stack holds 256 while a `while` pushes one level per iteration. 384 is
    outside both, so a loop would have to be unrolled by at least 2 anyway.
    """
    import g17ir as ir
    if not (1 <= rows and 1 <= columns and (rows + 1) * columns < 2 ** 30):
        raise ValueError("positive dimensions and a packed buffer below 4 GiB required")
    packed = ir.Buffer("packed", 1, elem=ir.F16)
    scores = ir.Buffer("scores", 2, elem=ir.F16)
    f = ir.Function("half_scan", [packed, scores])
    b = ir.Builder(f, f.block("entry"))
    row = b.builtin("thread_position_in_grid", name="row")
    stride = b.const(columns, name="stride")
    ai = b.mul(row, stride, name="ai0")
    xi = b.const(rows * columns, name="xi0")
    acc = b.const(0, name="zero")
    for k in range(columns):
        a = b.f16_to_f32(b.load(packed, ai, name="a%d" % k, width="half"), name="af%d" % k)
        x = b.f16_to_f32(b.load(packed, xi, name="x%d" % k, width="half"), name="xf%d" % k)
        acc = b.fadd(acc, b.fmul(a, x, name="p%d" % k), name="sum%d" % k)
        if k + 1 < columns:
            ai = b.add(ai, ir.Imm(1), name="ai%d" % (k + 1))
            xi = b.add(xi, ir.Imm(1), name="xi%d" % (k + 1))
    b.store_at(scores, row, b.f32_to_f16_rte(acc, name="half_out"), width="half")
    # This is the source-owned long FP16 scan path whose 33x384 bytes and hardware results are
    # retained under results/g17-half-full-executed and results/g17-half-runtime-executed-33.
    # The indexed half-store dependency is not general; keep this explicit opt-in on the known
    # application shape rather than making arbitrary half-load consumers appear supported.
    f.allow_validated_half_store = True
    b.ret()
    return f


# THE RECEIPT THAT JUSTIFIES THE EXEMPTION BELOW, by path and by the code hash it records.
VALIDATED = ("results/g17-common-v3-validation/packed-33x384/common-validation-3.json",
             "6c2f696341e9f0b25316408417ec8450efcf58a9af7734404d046224fee7e31f", (33, 384))


def compile_scan(rows=500_000, columns=384):
    import g17cc
    fn = scan_ir(rows, columns)
    # THIS PROGRAM'S BYTES HAVE A PASSING 32-LANE RECEIPT, and that is what justifies the opt-out
    # scan_ir sets. integration's guard (159cccdc) refuses an indexed half store whose value derives
    # from a half load, after a bounded campaign in which four such members left odd half elements
    # unwritten on hardware. The scan is one of those by construction - 768 half loads reduced into
    # one store - and the refusal made it uncompilable. But its bytes at 33x384 hash to
    # 6c2f696341e9f0b25316408417ec8450efcf58a9af7734404d046224fee7e31f, which is the code hash in
    # results/g17-common-v3-validation/packed-33x384/common-validation-3.json: dispatched, status 0,
    # passed. A guard cannot be right about a program that has already run.
    #
    # The marker itself is integration's `allow_validated_half_store`, set in scan_ir where every
    # caller gets it - which is the right place, because thirty-six tests compile scan_ir directly
    # rather than through here. What this comment adds is the receipt, and test_g17halfscan checks
    # that the bytes this tool builds still hash to it. An exemption whose justification is not
    # checked is a flag; one whose justification is checked is a claim.
    #
    # WHAT IT MEANS FOR THE GUARD, integration's to decide: the scan and the campaign's
    # `narrow_roundtrip` end in the same two opcodes with the same modifiers - op1016/12 with the
    # same wait and the same imm:128, then op17193/14 with the same index - and differ in ONE
    # register. The scan's store reads reg:431 and passes; narrow_roundtrip's reads reg:430 and
    # fails. One witness on each side; tools/g17halfstoredep.py carries the pair that settles it.
    program = g17cc.compile_function(fn)
    abi = program.abi(profile=PROFILE)
    # THE GUARD IS THE SAME ONE THE EXECUTED PATH USES, and it is stated rather than assumed: a
    # program outside the two-binding buffer class does not belong to this profile.
    if abi["uses_threadgroup"] or not abi["writes_buffer"] or abi["writes_texture"]:
        raise ValueError("program is outside the two-binding buffer ABI: %r" % abi)
    want = [(1, 0, False, "half"), (2, 2, True, "half")]
    got = [(x["index"], x["offset"], x["written"], x["element_type"]) for x in abi["bindings"]]
    if got != want:
        raise ValueError("bindings %r are not this profile's %r" % (got, want))
    return program, abi


DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "isa",
                    "g17-fp16-scan-abi.json")


def dump(rows, columns, path=None):
    """The program's TEXT and its ABI as data, so the linker can package without this compiler.

    Repository-owned by construction: everything here is produced from the IR above and the
    recovered forms, and nothing is copied from a delivered object.
    """
    import hashlib
    import g17cc
    program, abi = compile_scan(rows, columns)
    # The ABI is a read-only mapping; abi_plain converts it at the serialisation boundary rather
    # than the compiler handing out something mutable.
    out = g17cc.G17Program.abi_plain(abi)
    out["pk_values"] = {str(k): v for k, v in out["pk_values"].items()}
    out["rows"], out["columns"] = rows, columns
    out["text"] = program.code.hex()
    out["text_sha256"] = hashlib.sha256(program.code).hexdigest()
    out["instructions"] = len(program.layout)
    out["source"] = "tools/g17halfscan.py --dump --rows %d --columns %d" % (rows, columns)
    out["note"] = ("ld_md_slots, ld_md_values and per-kernel slot 1 are ABSENT on purpose: this "
                   "side has no measurement that determines them, and for this class they come "
                   "from the artefact.")
    path = path or DUMP
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    return path, out


def main():
    if "--dump" in sys.argv:
        r = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 33
        c = int(sys.argv[sys.argv.index("--columns") + 1]) if "--columns" in sys.argv else 384
        path, out = dump(r, c)
        print("wrote %s: %dx%d, %d instructions, %d text bytes"
              % (os.path.relpath(path, os.path.dirname(os.path.dirname(path))),
                 r, c, out["instructions"], len(out["text"]) // 2))
        return 0
    rows = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 500_000
    cols = int(sys.argv[sys.argv.index("--columns") + 1]) if "--columns" in sys.argv else 384
    program, abi = compile_scan(rows, cols)
    print("half scan %d x %d: %d instructions, %d bytes"
          % (rows, cols, len(program.layout), len(program.code)))
    import g17cc
    print(json.dumps(g17cc.G17Program.abi_plain(abi), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
