"""Tensor lowering: the stream of instructions a tensor operation becomes.

The production half of tools/g17tensorlower.py - what g17cc asks for when it lowers a tensor op.
The probes that build and check witnesses for those streams stayed behind, because they compile
through g17cc.
"""
import json, os, sys
from pathlib import Path
# IMPORTED FROM THE PACKAGE, NOT THE tools/ SHIMS. `import agxdis, g17asm, g17tensor` reaches
# tools/agxdis.py, tools/g17asm.py and tools/g17tensor.py, which are forwarding proxies onto
# exactly these three modules - so it only resolves when tools/ is already on sys.path. Every
# other module under agxforge/g17/ imports them this way, and this was the last one that did not.
from agxforge.g17 import agxdis, asm as g17asm, tensor as g17tensor
# ANCHORED ON THE CHECKOUT ROOT: parents[2] from agxforge/g17/, where parents[1] sufficed from
# tools/. A Path object rather than a string, which is why the library path sweep - which
# walks module-level strings - did not see it. The abi suite did.
ROOT = Path(__file__).resolve().parents[2]
WITNESSES = {
    "baseline": (ROOT / "results/g17-tensor-common-witness-v1/tensor-common", dict(strideA=128, strideB=64, strideC=128)),
    "b-stride": (ROOT / "results/g17-tensor-common-witness-v1/tensor-b-stride", dict(strideA=128, strideB=96, strideC=128)),
    "c-stride": (ROOT / "results/g17-tensor-variant-witness-v1/tensor-c-stride-48", dict(strideA=128, strideB=64, strideC=192)),
    "a-stride": (ROOT / "results/g17-tensor-variant-witness-v1/tensor-a-stride-96", dict(strideA=192, strideB=64, strideC=128)),
    # a used pad buffer bound before A: the same strides, the bindings at offsets 2/4/6
    "leading-buf": (ROOT / "results/g17-tensor-variant-witness-v1/tensor-leading-buf", dict(strideA=128, strideB=64, strideC=128, bindings=(2, 4, 6))),
}
ADVANCE_WITNESSES = {
    "a-stride-80": (ROOT / "results/g17-tensor-variant-witness-v1/tensor-a-stride-80", dict(strideA=160, strideB=64, strideC=128)),
    "a-stride-128": (ROOT / "results/g17-tensor-variant-witness-v1/tensor-a-stride-128", dict(strideA=256, strideB=64, strideC=128)),
    "a-stride-256": (ROOT / "results/g17-tensor-variant-witness-v1/tensor-a-stride-256", dict(strideA=512, strideB=64, strideC=128)),
}
ADVANCE_UNIT = dict(strideA=32, strideB=32, strideC=64)
ADVANCE_OFFSETS = dict(strideA=(102,), strideB=(286, 726), strideC=(1120,))
ADVANCE_MIN = dict(strideA=4, strideB=2, strideC=2)          # K*2/32, N*2/32, N*4/64 at 32x32x64
SCALE_CODE = {2: 8, 4: 10, 8: 1, 16: 0}                        # g17asm.decode_alu scale_code per x
REG_FORM_WITNESS = dict(strideA="a-stride-80", strideB="b-stride", strideC="c-stride")
CONFIRMED_ROWS = dict(strideA={4, 5, 8, 9, 16, 17}, strideB={2, 3, 4, 5, 8, 9, 16}, strideC={2, 3, 5, 8, 9, 16})
MULTIPLY_IMM_MAX = 63                                          # measured: 15 rows (60) generates, 18 rows (72) does not
MULTIPLY_FAMILY = "a-stride"
MULTIPLY_OFFSETS = (74, 704)
MULTIPLY_IMM_BYTE = 9
MULTIPLY_ROWS_CONFIRMED = {6, 7, 10, 11, 12, 13, 14, 15}     # round 2 confirmed 10, 11, 13, 14, 15 byte for byte
PRED = ROOT / "results/g17-tensor-stride-prediction-v1"
B_MULTIPLY_WITNESS = (PRED / "b-rows-6", dict(strideA=128, strideB=192, strideC=128))
B_MULTIPLY_OFFSETS = (274, 704)
B_MULTIPLY_ROWS_CONFIRMED = {6, 7, 10, 11, 12, 13, 14, 15}    # round 4 confirmed 11, 13, 14, 15
C_MULTIPLY_WITNESS = (PRED / "c-rows-6", dict(strideA=128, strideB=64, strideC=384))
C_MULTIPLY_REPLACES = (1120, 1132)                              # baseline offsets replaced by the one multiply at +1120
C_MULTIPLY_ROWS_CONFIRMED = {6, 7, 10, 11, 12, 13, 14, 15}    # round 4 confirmed 10, 11, 13, 14, 15
K_MIN, K_MAX = 64, 224
K_CONFIRMED = {64, 96, 128, 160, 192, 224}      # round 6: 160 and 224 byte-identical, with A x16 and B x3 at K = 128 too
def _put_k_units(u, value):
    if value % 32 or not 0 <= value // 32 <= 7:
        raise ValueError("K field %d is not a multiple of 32 below 256" % value)
    f = value // 32; u = bytearray(u)
    u[3] = (u[3] & ~0x60) | ((f & 1) << 5) | (((f >> 1) & 1) << 6)
    u[2] = (u[2] & ~0x10) | (((f >> 2) & 1) << 4)
    return bytes(u)
def k_rows(rows, K):
    """The family's rows with the five K-dependent fields written for K: {offset: (opcode, bytes)}."""
    from agxforge.g17 import auth as g17auth
    if K % 32 or not K_MIN <= K <= K_MAX:
        raise ValueError("K = %d: the loop bounds are 3-bit multiples of 32 below 256 (K in 64..224 step 32; Apple's K = 256 program changes forms - op10378, op615 with a pool word - and is not generated)" % K)
    cmps = [r for r in rows if r[2] == 10369]; sel = [r for r in rows if r[2] == 11452]; mins = [r for r in rows if r[2] == 612]
    if len(cmps) != 2 or len(sel) != 1 or len(mins) != 2:
        raise AssertionError("the family's rows do not carry the two compares, the select and the two op612 (%d, %d, %d)" % (len(cmps), len(sel), len(mins)))
    out = {}
    out[cmps[0][0]] = (10369, _put_k_units(cmps[0][4], K - 32))
    out[cmps[1][0]] = (10369, _put_k_units(cmps[1][4], K))
    out[sel[0][0]] = (11452, _put_k_units(sel[0][4], K))
    for r in mins:
        out[r[0]] = (612, bytes(g17auth.encode(612, {5: K}, template=r[4]))[:12])
    return out
def family_rows(prefix):
    """A witness as stream rows (offset, phase, opcode, length, bytes, regs) by the phase rule of
    g17tensorphases; regs = the 32-bit registers the decode names (105 + n), as in
    g17tensorwitnessdata.STREAM (which this reproduces for the baseline, asserted by regression)."""
    from agxforge.g17 import tensorphases as g17tensorphases
    code, ins = witness(prefix); seen = False; out = []
    for o, ln, op, toks in ins:
        ph = g17tensorphases.phase_of(op, seen); seen = seen or op == 5106
        if ph == "end": continue                      # the caller adds `end`, as with STREAM
        regs = [int(t[4:]) - 105 for t in toks if t.startswith("reg:") and 105 <= int(t[4:]) < 425]
        out.append((o, ph, op, ln, code[o:o + ln], regs))
    return out
def _rows(which, stride):
    unit = ADVANCE_UNIT[which]
    return stride // unit if stride > 0 and stride % unit == 0 else None
def family_of(strides, confirmed_only=True):
    """Which witness program a stride triple is generated from: ("shift", baseline),
    ("multiply", the A-stride-192 witness), ("b-multiply", the b-rows-6 witness) or
    ("c-multiply", baseline with the readout chain's two adds replaced); ValueError otherwise."""
    base_strides = WITNESSES["baseline"][1]
    sa = strides["strideA"]
    mb, mc = _rows("strideB", strides["strideB"]), _rows("strideC", strides["strideC"])
    shift_b = strides["strideB"] == base_strides["strideB"] or (mb is not None and (mb in CONFIRMED_ROWS["strideB"] or (not confirmed_only and (mb in SCALE_CODE or (mb - 1) in SCALE_CODE))))
    shift_c = strides["strideC"] == base_strides["strideC"] or (mc is not None and (mc in CONFIRMED_ROWS["strideC"] or (not confirmed_only and (mc in SCALE_CODE or (mc - 1) in SCALE_CODE))))
    if not shift_b and mb is not None and mb >= ADVANCE_MIN["strideB"] and strides["strideB"] // 8 > MULTIPLY_IMM_MAX:
        raise ValueError("strideB of %d bytes = %d rows: the multiply immediate (stride / 8 = %d) exceeds its six-bit field (%d); not generated" % (strides["strideB"], mb, strides["strideB"] // 8, MULTIPLY_IMM_MAX))
    if not shift_c and mc is not None and mc >= ADVANCE_MIN["strideC"] and strides["strideC"] // 16 > MULTIPLY_IMM_MAX:
        raise ValueError("strideC of %d bytes = %d rows: the multiply immediate (stride / 16 = %d) exceeds its six-bit field (%d); not generated" % (strides["strideC"], mc, strides["strideC"] // 16, MULTIPLY_IMM_MAX))
    if not shift_b and mb is not None and mb >= ADVANCE_MIN["strideB"] and (mb in B_MULTIPLY_ROWS_CONFIRMED or not confirmed_only):
        if sa != base_strides["strideA"] or strides["strideC"] != base_strides["strideC"]:
            raise ValueError("strideB of %d bytes needs Apple's multiply program for B, and no witness combines it with another A or C stride (%s)" % (strides["strideB"], strides))
        return "b-multiply", B_MULTIPLY_WITNESS
    if not shift_c and mc is not None and mc >= ADVANCE_MIN["strideC"] and (mc in C_MULTIPLY_ROWS_CONFIRMED or not confirmed_only):
        # C's multiply is a LOCAL substitution, and it COMPOSES with the A and B advances of the
        # shift family: Apple's programs at (A x5, C x6) and (B x3, C x7) are exactly the union
        # of the two single-operand changes (round 4, retained a5-c6 and b3-c7). It does not
        # combine with A's or B's multiply program (rescheduled; no witness).
        ma = _rows("strideA", sa)
        shift_a = sa == base_strides["strideA"] or (ma is not None and (ma in CONFIRMED_ROWS["strideA"] or (not confirmed_only and (ma in SCALE_CODE or (ma - 1) in SCALE_CODE))))
        if not (shift_a and shift_b):
            raise ValueError("strideC of %d bytes needs Apple's multiply for C, which composes only with the shift family's A and B advances, not with their multiply programs (%s)" % (strides["strideC"], strides))
        return "c-multiply", WITNESSES["baseline"]
    if sa != base_strides["strideA"] and sa % 32 == 0 and sa // 32 >= ADVANCE_MIN["strideA"]:
        m = sa // 32
        shift = m in CONFIRMED_ROWS["strideA"] or (not confirmed_only and (m in SCALE_CODE or (m - 1) in SCALE_CODE))
        if not shift:
            if sa // 8 > MULTIPLY_IMM_MAX:
                raise ValueError("strideA of %d bytes = %d rows: the multiply immediate (stride / 8 = %d) exceeds its six-bit field (%d); Apple's program there multiplies by a constant-pool operand (op10828/op10829; a shift instruction at 32 rows, other load forms at 63), which is not generated" % (sa, sa // 32, sa // 8, MULTIPLY_IMM_MAX))
            if m in MULTIPLY_ROWS_CONFIRMED or not confirmed_only:
                if any(strides[k] != base_strides[k] for k in ("strideB", "strideC")):
                    raise ValueError("strideA of %d bytes needs Apple's multiply program, and no witness combines it with another B or C stride (%s)" % (sa, strides))
                return "multiply", WITNESSES[MULTIPLY_FAMILY]
    return "shift", WITNESSES["baseline"]
def advance_instructions(which, stride, confirmed_only=True):
    """{offset: {"bytes", "opcode"}} for the row advance(s) of `which` at `stride` bytes, or
    ValueError. THE OPCODE TRAVELS WITH THE BYTES: the 1 + 2^k form is op10282 where the
    baseline's is op10279, and the stream's instruction record must say so - integration caught
    a B-stride-96 compile declaring 10279 at +286/+726 while its bytes decoded as 10282
    (its shared-core fix, adopted here)."""
    unit = ADVANCE_UNIT[which]
    if stride <= 0 or stride % unit:
        raise ValueError("%s of %d bytes is not a whole number of %d-byte tile rows; no witness advances by a fraction" % (which, stride, unit))
    m = stride // unit
    if m < ADVANCE_MIN[which]:
        raise ValueError("%s of %d bytes is below the operand's own row (%d units of %d bytes)" % (which, stride, ADVANCE_MIN[which], unit))
    if m not in CONFIRMED_ROWS[which] and (confirmed_only or not (m in SCALE_CODE or (m - 1) in SCALE_CODE)):
        if which == "strideC" and m == 4:
            raise ValueError("strideC of 256 bytes = 4 x 64: Apple's program shares the x4 with the A advance and drops the x16 step (results/g17-tensor-stride-prediction-v1/c-rows-4), so the generated chain is not Apple's program and is unmeasured; refused")
        raise ValueError("%s of %d bytes = %d x %d: not a row count Apple's compiler confirmed (%s); the multiply program Apple emits for A at 6, 7 and 12 rows is not generated here" % (which, stride, m, unit, sorted(CONFIRMED_ROWS[which])))
    if m in SCALE_CODE:
        code, _ = witness(WITNESSES["baseline"][0]); scale = SCALE_CODE[m]
        opcode = 10279
    elif (m - 1) in SCALE_CODE:
        code, _ = witness(dict(WITNESSES, **ADVANCE_WITNESSES)[REG_FORM_WITNESS[which]][0]); scale = SCALE_CODE[m - 1]
        opcode = 10282
    else:
        raise AssertionError("a confirmed row count with no form: %s %d" % (which, m))
    out = {}
    for off in ADVANCE_OFFSETS[which]:
        u = code[off:off + 12]; d = g17asm.decode_alu(u)
        if d["opcode"] not in (10279, 10282) if "opcode" in d else False:
            raise AssertionError("the advance template at +%d is not an add" % off)
        raw = bytes(g17asm.encode_alu(d["dest"], d["src1"], d["mode"], u, imm=d.get("imm"), src2=d.get("src2"),
                                     live=d["live"], keep=d["keep"], src2_keep=d["src2_keep"], scale_code=scale))
        assert g17asm.decode_alu(raw)["scale_code"] == scale
        out[off] = dict(bytes=raw, opcode=opcode)
    return out
def witness(prefix):
    obj = (prefix.parent / (prefix.name + ".o")).read_bytes(); off, size = agxdis.sections(obj); code = obj[off:off + size][64:]
    ins = json.load(open(prefix.parent / (prefix.name + ".instructions.json")))
    ins = [(r["offset"], r["length"], r["opcode"], r["fields"]) if isinstance(r, dict) else tuple(r) for r in ins]
    return code, ins
def assignment_from(prefix):
    """Read the allocation (not the offsets) off a witness: per load its dest pair, index reg,
    which buffer (A/B) and tile (rb-or-cb, ks, half), k selector, hi, lifetimes, token; per MAC
    the composer's assignment; per store its value tuple, index, last flag, tile and half."""
    code, ins = witness(prefix)
    loads, stores, macs, zeros = [], [], [], []
    for o, ln, op, t in ins:
        u = code[o:o + ln]
        if op == 554: zeros.append(g17asm.decode_tensor_init(u)["dest"])
        elif op in (12674, 12675):
            f = g17asm.decode_tensor_load(u); f["opcode"] = op; f["length"] = ln
            f["op10"] = ((u[g17asm.TLOAD_OP10[0]] >> g17asm.TLOAD_OP10[1]) & 1) << 4
            f["parity"] = (u[6] >> 3) & 1; f["token_reg"] = (u[5] >> 1) & 0x7F
            loads.append(f)
        elif op == 5106: macs.append(g17asm.decode_tensor_mac(u))
        elif op == 17257:
            f = g17asm.decode_tensor_store(u); f["length"] = ln; stores.append(f)
    return dict(zeros=zeros, loads=loads, macs=macs, stores=stores)
def tile_of_offset(offset, which, strides):
    """Invert the offset rule: which tile and half a witness load/store addresses."""
    s = strides["strideA"] if which == "A" else strides["strideB"] if which == "B" else strides["strideC"]
    if which == "A":
        ks, rem = (1, offset - 32) if (offset % s) == 32 else (0, offset); rows = rem // s; return dict(block=rows // 16, half=(rows % 16) // 8, ks=ks)
    if which == "B":
        cb, rem = (1, offset - 32) if (offset % s) == 32 else (0, offset); rows = rem // s; return dict(block=cb, half=(rows % 16) // 8, ks=rows // 16)
    cb, rem = (1, offset - 64) if (offset % s) == 64 else (0, offset); rows = rem // s; return dict(block=rows // 16, half=(rows % 16) // 8, cb=cb)
def offset_of(tile, which, strides):
    if which == "A": return (tile["block"] * 16 + tile["half"] * 8) * strides["strideA"] + tile["ks"] * 32
    if which == "B": return (tile["ks"] * 16 + tile["half"] * 8) * strides["strideB"] + tile["block"] * 32
    return (tile["block"] * 16 + tile["half"] * 8) * strides["strideC"] + tile["cb"] * 64
def author(strides, bindings, assignment, templates):
    """-> dict(phase -> list of bytes) for the four generated phases."""
    tA, tB, tC = bindings
    out = {"zeroing": [], "loads": [], "macs": [], "readout": []}
    for dest in assignment["zeros"]:
        out["zeroing"].append(g17asm.encode_tensor_init(dest, templates[(554, 4)], 0))
    for f in assignment["loads"]:
        which = f["which"]
        offset = offset_of(f["tile"], which, strides)
        length = 16 if (offset >= 256 or f["hi"]) else 12
        u = bytearray(g17asm.encode_tensor_load(templates[(f["opcode"], length)], dest=f["dest"], index=f["index"], base=(tA if which == "A" else tB),
                                                 k=f["k"], hi=f["hi"], offset=offset, op6=f["op6"], op10=f["op10"]))
        u[6] = (u[6] & ~8) | (f["parity"] << 3)
        if f["opcode"] == 12675: u[5] = (u[5] & 1) | (f["token_reg"] << 1)
        out["loads"].append(bytes(u))
    macs = assignment["macs"]
    out["macs"] = list(g17tensor.compose_mac_bytes(32, 32, macs["acc_regs"], macs["a_regs"], macs["b_regs"], templates[(5106, 10)]))
    for f in assignment["stores"]:
        offset = offset_of(f["tile"], "C", strides); length = 16 if offset >= 256 else 10
        out["readout"].append(g17asm.encode_tensor_store(templates[(17257, length)], value=f["value"], index=f["index"], base=tC, offset=offset, last=f["last"]))
    return out
def templates_of(prefix):
    code, ins = witness(prefix); t = {}
    for o, ln, op, _ in ins:
        t.setdefault((op, ln), code[o:o + ln])
    return t
def assignment_of(prefix, strides, bindings=(0, 2, 4)):
    """A witness's allocation (registers, k selectors, lifetimes, tokens, order) with every
    OFFSET replaced by the tile it addresses and every base by which buffer it names, so that
    authoring it back with the rules from (strides, bindings) tests the RULES, not the copy."""
    a = assignment_from(prefix); tA, tB, tC = bindings
    for f in a["loads"]:
        which = "A" if f["base"] == tA else "B"; f["which"] = which; f["tile"] = tile_of_offset(f["offset"], which, strides); del f["offset"]; del f["base"]
    for f in a["stores"]:
        f["tile"] = tile_of_offset(f["offset"], "C", strides); del f["offset"]; del f["base"]
    law = g17tensor.compose_macs(32, 32, 64); acc, ar, br = {}, {}, {}
    for m, (k, row, col, ks) in zip(a["macs"], law):
        acc.setdefault(k, m["acc"]); ar.setdefault(2 * row + ks, m["a"]); br.setdefault(2 * col + ks, m["b"])
    a["macs"] = dict(acc_regs=[acc[k] for k in sorted(acc)], a_regs=[ar[i] for i in sorted(ar)], b_regs=[br[i] for i in sorted(br)])
    return a
from agxforge.g17 import tensorwitnessdata as WD
AUTHORED_PHASES = ("zeroing", "loads", "macs", "readout")
def stream(M, N, K, strides, bindings, a_dtype="half", b_dtype="half", confirmed_only=True):
    return _stream(M, N, K, strides, bindings, a_dtype, b_dtype, confirmed_only)
def _stream(M, N, K, strides, bindings, a_dtype, b_dtype, confirmed_only):
    """The complete instruction stream for the contract, in the witness's order: the four authored
    phases generated from (strides, bindings, Apple's 32x32x64 allocation) and the setup/address/
    loop phases INHERITED from the witness (g17tensorwitnessdata.STREAM). -> list of dicts
    (phase, opcode, length, bytes, inherited, regs). Refuses any shape/dtype but the one the
    witness measures; the caller adds `end`."""
    if (M, N) != (32, 32) or (a_dtype, b_dtype) != ("half", "half"):
        raise ValueError("the authored tensor stream covers 32x32xK half x half -> float only; %dx%dx%d %s x %s has no witness" % (M, N, K, a_dtype, b_dtype))
    if K != 64 and (K not in K_CONFIRMED and confirmed_only):
        raise ValueError("K = %d: not a depth Apple's compiler confirmed (%s); the loop bounds are generated for K in 64..224 step 32 once a round confirms them" % (K, sorted(K_CONFIRMED)))
    family, (base_prefix, base_strides) = family_of(strides, confirmed_only)
    # THE STRIDES REACH THE SETUP through the row advances, generated by advance_bytes (refused
    # with the reason where no witness gives the rule), or through the multiply program's two
    # immediates; everything else in the setup, address and loop phases is inherited byte for
    # byte from the family's witness, which the aligned witnesses show is right.
    advances = {}
    if family == "shift":
        for key in ("strideA", "strideB", "strideC"):
            if strides.get(key) != base_strides[key]:
                advances.update(advance_instructions(key, strides[key], confirmed_only))
        rows = [(o, ph, op, ln, bytes.fromhex(hx) if hx else b"", regs) for o, ph, op, ln, hx, regs in WD.STREAM]
    elif family == "multiply":
        rows = family_rows(base_prefix)
        for off in MULTIPLY_OFFSETS:
            row = next(r for r in rows if r[0] == off); u = bytearray(row[4]); u[MULTIPLY_IMM_BYTE] = strides["strideA"] // 8; advances[off] = dict(bytes=bytes(u), opcode=row[2])
    elif family == "b-multiply":
        rows = family_rows(base_prefix)
        for off in B_MULTIPLY_OFFSETS:
            row = next(r for r in rows if r[0] == off); u = bytearray(row[4]); u[MULTIPLY_IMM_BYTE] = strides["strideB"] // 8; advances[off] = dict(bytes=bytes(u), opcode=row[2])
    else:   # c-multiply: the baseline rows with +1120 and +1132 replaced by the one multiply from the C witness
        cw = family_rows(C_MULTIPLY_WITNESS[0]); mul = bytearray(next(r[4] for r in cw if r[0] == C_MULTIPLY_REPLACES[0] and r[2] == 10822))
        mul[MULTIPLY_IMM_BYTE] = strides["strideC"] // 16
        for key in ("strideA", "strideB"):                  # the shift family's advances compose with it
            if strides.get(key) != base_strides[key]:
                advances.update(advance_instructions(key, strides[key], confirmed_only))
        rows = []
        for o, ph, op, ln, hx, regs in WD.STREAM:
            if o == C_MULTIPLY_REPLACES[0]: rows.append((o, ph, 10822, 14, bytes(mul), regs)); advances[o] = dict(bytes=bytes(mul), opcode=10822)
            elif o == C_MULTIPLY_REPLACES[1]: continue
            else: rows.append((o, ph, op, ln, bytes.fromhex(hx) if hx else b"", regs))
    if K != 64:
        for off, (op, by) in k_rows(rows, K).items():
            advances[off] = dict(bytes=by, opcode=op)
    templates = templates_of(base_prefix); assignment = assignment_of(base_prefix, base_strides)
    authored = author(strides, bindings, assignment, templates)
    cursor = {ph: 0 for ph in AUTHORED_PHASES}; out = []
    for offset, phase, opcode, length, inherited, regs in rows:
        if phase in AUTHORED_PHASES:
            by = authored[phase][cursor[phase]]; cursor[phase] += 1
            assert len(by) == length, (phase, opcode, length, len(by))
            out.append(dict(phase=phase, opcode=opcode, length=length, bytes=by, inherited=False, regs=regs))
        elif offset in advances:
            advance = advances[offset]; by = advance['bytes']; assert len(by) == length
            out.append(dict(phase=phase, opcode=advance['opcode'], length=length, bytes=by, inherited=False, regs=regs))
        else:
            out.append(dict(phase=phase, opcode=opcode, length=length, bytes=inherited, inherited=True, regs=regs))
    assert all(cursor[ph] == len(authored[ph]) for ph in AUTHORED_PHASES)
    return out
