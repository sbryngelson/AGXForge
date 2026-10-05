"""Form selection rules: which opcode a named operation lowers to.

The production half of tools/g17formops.py. Its --refresh path harvests these from a
program and needs the compiler; that half stayed behind. This half is what g17cc reads.
"""
import json
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
# ANCHORED ON THE CHECKOUT ROOT. The split copied this line from tools/, where one dirname
# reached the checkout; from agxforge/g17/ it reaches agxforge/. The library path sweep caught it.
ROOT = os.path.dirname(os.path.dirname(HERE))
ISA = os.path.join(ROOT, "isa")
TABLE = os.path.join(ISA, "g17-form-opcodes.json")
SELECTORS = {
    # THE WIDTHS ARE PART OF APPLE'S OPCODE NUMBER, which the conflict guard below found
    # rather than anyone assuming it: op=3 mode=1 is 10279 at dest_w=1 src1_w=1, 10280 at
    # src1_w=0 and 10288 at dest_w=0. That is the same fact the half load taught - a sixteen-bit
    # operand is a different FORM, not a field of one - showing up in the ALU family.
    "alu.12": ("op", "mode", "src1_w", "srcb_w", "dest_w"),
    "read_sr.4": ("half",),
    "load.8": (),
    "load.10": ("half",),
    "load.14": ("half", "narrow", "hi16"),
    # THE COMPONENT COUNT IS PART OF THE STORE'S OPCODE NUMBER: n=2 is 17244, 3 is 17253, 4 is
    # 17262 at either length, and a key without it made one row stand for three opcodes - the
    # rangestore kernel compiled and then raised at abi() (docs/g17-cooperative-integration-
    # feedback.md). `half` still separates the sixteen-bit store, and the conflict guard in
    # harvest() is what proves each key now names exactly one opcode.
    "store.8": ("half", "n"),
    "store.14": ("half", "n"),
    # THE HALF-VECTOR SLOT STORES: the component count is part of Apple's opcode number here exactly as it
    # is for the word slot stores - n=2 is 17208, n=3 is 17217, n=4 is 17226 - and the conflict guard in
    # harvest() is what proves this key names one opcode each rather than standing for three.
    "store.halfvec.14": ("n",),
    "atomic.add.10": ("aop",),
    "atomic.add.12": ("aop",),
}
def key_of(minst, length):
    """(form, length, selector values) - never the registers, immediate or address."""
    sel = SELECTORS.get(minst.form, ())
    vals = [minst.fields.get(k) for k in sel]
    return "|".join([minst.form, str(length)] + ["%s=%s" % (k, v) for k, v in zip(sel, vals)])
def load():
    if not os.path.exists(TABLE):
        return {}
    with open(TABLE) as fh:
        return json.load(fh).get("opcodes") or {}
def opcode_of(minst, length, table=None):
    """Apple's opcode for this emitted instruction, from the compiler's own record.

    fields['opcode'] wins where the emitter already carries it - the auth and per-opcode ALU forms
    name it outright - and the table answers for the rest. A miss raises: see the module docstring.
    """
    got = minst.fields.get("opcode")
    if got is not None:
        return got
    t = load() if table is None else table
    k = key_of(minst, length)
    if k not in t:
        raise KeyError("no opcode recorded for %r; run tools/g17formops.py --refresh" % k)
    return t[k]
