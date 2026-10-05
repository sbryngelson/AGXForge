"""The phase model of Apple's complete tensor matmul witness, and the round trip that shows the
assembler's measured encoders cover its tensor-path instructions.

    python3 tools/g17tensorphases.py [results/g17-tensor-common-witness-v1/tensor-common]

Every instruction of the witness is assigned to one of six phases (setup, zeroing, loads, MACs,
loop, readout). For each tensor-path form the instruction's fields are DECODED with the
assembler's measured field maps and RE-ENCODED from a canonical template - the first instance of
that opcode and length in the witness - and the result must equal the original bytes. That is
what makes the lowering possible: the zeroing (op554), the operand loads (op12674/op12675), the
MACs (op5106) and the readout (op17257) are functions of their fields, not bytes copied from a
donor. The setup phase (op17016, op14060, op423, op426, op10283, op10286, op586, op612, op11452)
is listed with its tokens and left as the measured unknown it is."""
import json, os, sys
from pathlib import Path
from agxforge.g17 import agxdis, asm as g17asm

# ANCHORED ON THE CHECKOUT ROOT: parents[2] from agxforge/g17/, where parents[1] sufficed from
# tools/.
ROOT = Path(__file__).resolve().parents[2]
SETUP = {17016, 14060, 423, 426, 10283, 10286, 586, 612, 11452}
LOOP = {10369, 579, 582, 458, 462, 577}


def phase_of(op, seen_mac):
    if op == 554: return "zeroing"
    if op in (12674, 12675): return "loads"
    if op == 5106: return "macs"
    if op == 17257: return "readout"
    if op in LOOP: return "loop"
    if op == 684: return "end"
    if op in SETUP or op in (10279, 10282, 11842): return "setup" if not seen_mac else "address"
    return "other"


def witness(prefix):
    prefix = Path(prefix); obj = (prefix.parent / (prefix.name + ".o")).read_bytes()
    off, size = agxdis.sections(obj); code = obj[off:off + size][64:]
    ins = json.load(open(prefix.parent / (prefix.name + ".instructions.json")))
    return code, ins


def model(prefix=ROOT / "results/g17-tensor-common-witness-v1/tensor-common"):
    code, ins = witness(prefix)
    rows, templates, seen_mac = [], {}, False
    for o, ln, op, toks in ins:
        u = code[o:o + ln]; ph = phase_of(op, seen_mac); seen_mac = seen_mac or op == 5106
        key = (op, ln); templates.setdefault(key, u)
        row = dict(offset=o, length=ln, opcode=op, phase=ph, bytes=u.hex(), tokens=toks)
        try:
            if op == 554:
                f = g17asm.decode_tensor_init(u); row["fields"] = f
                row["reencoded"] = g17asm.encode_tensor_init(f["dest"], templates[key], f["flag"]) == u
            elif op in (12674, 12675):
                f = g17asm.decode_tensor_load(u)
                # the two fields the decoder does not return: operand 10 (the index register's
                # lifetime, op12675's second source) and the rendezvous parity b6[3], which on
                # op12675 is the consuming MAC's token parity inverted - the same bit the k field
                # reads as its low bit, so it is re-applied after k
                f["op10"] = ((u[g17asm.TLOAD_OP10[0]] >> g17asm.TLOAD_OP10[1]) & 1) << 4
                f["parity"] = (u[g17asm.TLOAD_TOKEN_PARITY[0]] >> g17asm.TLOAD_TOKEN_PARITY[1]) & 1
                # op12675's operand 10 is a REGISTER - the wait token produced by op612/op11452,
                # a 16-bit half (decoder ids 425+n) - and its index sits in byte5[7:1] as slot bits
                # 1..7: reg:465 (half 40) -> 0x50, reg:429 (half 4) -> 0x08, reg:430 (half 5) ->
                # 0x0a, the exact six bytes the round trip could not reproduce without it.
                f["token_reg"] = (u[5] >> 1) & 0x7F
                row["fields"] = f
                enc = bytearray(g17asm.encode_tensor_load(templates[key], **{k: v for k, v in f.items() if k in ("dest", "index", "base", "k", "hi", "offset", "op6", "op10")}))
                enc[g17asm.TLOAD_TOKEN_PARITY[0]] = (enc[g17asm.TLOAD_TOKEN_PARITY[0]] & ~(1 << g17asm.TLOAD_TOKEN_PARITY[1])) | (f["parity"] << g17asm.TLOAD_TOKEN_PARITY[1])
                if op == 12675: enc[5] = (enc[5] & 1) | (f["token_reg"] << 1)
                row["reencoded"] = bytes(enc) == u
            elif op == 5106:
                f = g17asm.decode_tensor_mac(u); row["fields"] = f
                row["reencoded"] = None   # the MAC's canonical encoder is g17tensor.compose_mac_bytes; checked there
            elif op == 17257:
                f = g17asm.decode_tensor_store(u); row["fields"] = f
                row["reencoded"] = g17asm.encode_tensor_store(templates[key], **{k: v for k, v in f.items() if k in ("value", "index", "base", "offset", "last")}) == u
        except Exception as e:
            row["fields"] = "decode failed: %s: %s" % (type(e).__name__, str(e)[:80]); row["reencoded"] = False
        rows.append(row)
    # THE MACS, through the composer: Apple's register assignment read off the witness, the law's
    # issue order from g17tensor.compose_macs, bytes from g17tensor.compose_mac_bytes.
    from agxforge.g17 import tensor as g17tensor
    macs = [r for r in rows if r["opcode"] == 5106]
    law = g17tensor.compose_macs(32, 32, 64)
    if len(law) == len(macs):
        acc, a, b = {}, {}, {}
        for r, (k, row_, col, ks) in zip(macs, law):
            f = r["fields"]; acc.setdefault(k, f["acc"]); a.setdefault(2 * row_ + ks, f["a"]); b.setdefault(2 * col + ks, f["b"])
        consistent = all(acc[k] == r["fields"]["acc"] and a[2 * row_ + ks] == r["fields"]["a"] and b[2 * col + ks] == r["fields"]["b"] for r, (k, row_, col, ks) in zip(macs, law))
        acc_regs = [acc[k] for k in sorted(acc)]; a_regs = [a[i] for i in sorted(a)]; b_regs = [b[i] for i in sorted(b)]
        try:
            composed = g17tensor.compose_mac_bytes(32, 32, acc_regs, a_regs, b_regs, bytes.fromhex(macs[0]["bytes"]))
            for r, by in zip(macs, composed):
                r["reencoded"] = (by == bytes.fromhex(r["bytes"])); r["composed"] = by.hex()
        except Exception as e:
            for r in macs: r["reencoded"] = False; r["composed"] = "compose failed: %s" % e
        for r in macs: r["assignment_consistent"] = consistent
    return rows


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results/g17-tensor-common-witness-v1/tensor-common")
    rows = model(prefix)
    from collections import Counter
    print("phases:", dict(Counter(r["phase"] for r in rows)))
    rt = [r for r in rows if r.get("reencoded") is not None]
    print("round trips: %d checked, %d equal, %d differ" % (len(rt), sum(1 for r in rt if r["reencoded"]), sum(1 for r in rt if not r["reencoded"])))
    for r in rows:
        if r["phase"] in ("zeroing",) and r["offset"] > 0x10: continue
        print("+%#05x %-8s op%-6d %2dB  %s" % (r["offset"], r["phase"], r["opcode"], r["length"], (str(r.get("fields", ""))[:120] if "fields" in r else str(r["tokens"][:7]))))
        if r.get("reencoded") is False: print("      RE-ENCODE DIFFERS: %s" % r["bytes"])
