"""The constant-bit inventory: which bits every selectable form leaves fixed.

The production half of tools/g17const.py - what g17cc reads to know a form's invisible bits. The
census that harvests these, and the report that compares them against the corpus, stayed behind
because they build witnesses through the compiler.

Its one edge back into g17cc was BITWISE_REG_TEMPLATE, which is encoding data and now lives in
agxforge.g17.formenc; that is what let this half enter the package at all.
"""
import collections, glob, os, re, subprocess, sys, tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
def inventory():
    """-> [(form, opcode, length, corpus_opcodes, encoder(template) -> bytes)] for every form the
    backend can select. One entry per selectable opcode."""
    from agxforge.g17 import asm as g17asm, formenc as _formenc
    out = []
    # The family forms: one entry each, opcode None, because that is what the emitter passes.
    out.append(("alu.12", None, 12, [10279, 10282],
                lambda t: g17asm.encode_alu(dest=3, src1=2, template=t, op=3, mode=1, imm=9,
                                            src1_w=1, dest_w=1, srcb_w=1, scale_code=2, keep=1,
                                            load_wait=0, hazard=0)))
    out.append(("movimm.8", None, 8, [11842],
                lambda t: g17asm.encode_movimm(imm=0, template=t, dest=3)))
    out.append(("read_sr.4", None, 4, [14059],
                lambda t: g17asm.encode_sr(dest=3, sr=160, seq=0, template=t)))
    out.append(("store.8", None, 8, [17244],
                lambda t: g17asm.encode_store(src=3, n=2, slot=8, template=t)))
    out.append(("store.14", None, 14, [17235, 17244],
                lambda t: g17asm.encode_store(src=3, n=2, slot=64, template=t, wait_load=1)))
    # The per-opcode ALU forms. `add` is selected through the alu.12 family, so it is not here.
    NAME = {"sub": "alu.sub", "mul": "alu.mul", "shl": "alu.shift", "shr": "alu.shift",
            "addsat": "alu.sat", "subsat": "alu.sat", "sar": "alu.sat"}
    for opc, (op_name, ra, rb) in g17asm.ALU_FORM.items():
        if op_name == "add":
            continue
        ln = g17asm.ALU_FORM_SIZE[opc]
        form = ("alu.sat" if op_name in ("addsat", "subsat", "sar")
                else "%s.%s" % (NAME[op_name], "imm" if rb == "imm" or ra == "imm" else "reg"))
        a = ("imm", 5) if ra == "imm" else ("reg", 2, 0)
        b = ("imm", 7) if rb == "imm" else ("reg", 4, 0)
        out.append((form, opc, ln, [opc],
                    (lambda o, aa, bb: lambda t: g17asm.encode_alu_form(
                        o, 3, aa, bb, t, keep_a=True, keep_b=True, hazard=0,
                        addend=0 if o in g17asm.ALU_ADDEND else None))(opc, a, b)))
    for opc in g17asm.BITWISE_FORM:
        out.append(("bitwise.imm", opc, 10, [opc],
                    (lambda o: lambda t: g17asm.encode_bitwise_imm(o, 3, 2, 15, t, hazard=0,
                                                                   keep=True))(opc)))
    # THE FLOAT UNARY FORM. Ten bytes, one entry per operation, and the entry matters more here
    # than elsewhere: nine of these ten opcodes differ from each other ONLY in bits the encoder
    # never writes - byte6[2:0] for the transcendental code and byte8[6:7] for the rounding mode -
    # so a resolver that substituted a same-length template would silently compute log2 where the
    # IR said sqrt. That is exactly the substitution this table was rewritten to make impossible.
    for opc, (nm, _src) in g17asm.TRANS_FORM.items():
        out.append(("float.unary", opc, 10, [opc],
                    (lambda o: lambda t: g17asm.encode_trans(o, 3, 2, t, keep=True))(opc)))
    for opc, (nm, ln, _src) in g17asm.UNARY_FORM.items():
        # THE PROBE DECIDES WHICH BITS COUNT AS WRITTEN, so op11179 gets its own. The shared probe
        # asks for hazard=0 and keep=True; on the int-to-float form that hazard write CLEARS
        # byte4[3], which its witnessed template has SET and whose role is unmeasured. Probing it
        # that way would record the field as encoder-written and freeze the constant with the bit
        # cleared - emitting an instruction one unmeasured bit away from anything Apple compiled.
        # Its lowering passes hazard=None for the same reason, and UNARY_KEEP has None for it.
        if opc == g17asm.CVT_I2F_OPCODE:
            out.append(("unary", opc, ln, [opc],
                        (lambda o: lambda t: g17asm.encode_unary(o, 3, 2, t))(opc)))
            continue
        out.append(("unary", opc, ln, [opc],
                    (lambda o: lambda t: g17asm.encode_unary(o, 3, 2, t, hazard=0, keep=True))(opc)))
    for opc, tmpl in _formenc.BITWISE_REG_TEMPLATE.items():
        out.append(("bitwise.reg", opc, len(tmpl), [opc],
                    (lambda o: lambda t: g17asm.encode_bitwise_reg(o, 3, 2, 4, t))(opc)))
    out.append(("load.14", None, 14, [12682, 12674],
                lambda t: g17asm.encode_load(dest=3, base=0, offset=0, template=t, index_reg=2,
                                             disp2=0, index_scale=1, narrow=0, hi16=0)))
    # THE BRANCH FORMS, and their absence was costing nine bits of reported debt. They do not go
    # through _tmpl - the emitter carries BRANCH_BACK and BRANCH_FWD as module constants - so this
    # inventory, which enumerates what the backend can SELECT, had never been asked about them.
    # The consequence is not a wrong byte: encode_branch10 writes every displacement bit, and it
    # is computed from the branch target. It is that g17debt saw nine written bits with no row to
    # attribute them to and classified them `displacement` debt, which is the census asking about
    # a form that was not in its inventory rather than a bit anybody owes. The loop work put the
    # back edge in front of the scorecard for the first time on 2026-09-08 and it has read that
    # way since.
    #
    # The displacement passed here is arbitrary: _mask_and_probe encodes into an all-zero and an
    # all-ones template and keeps the bits that come out the same, which is exactly the set the
    # encoder writes, whatever value it writes into them.
    # THE ALU WIDTH VARIANTS. Apple's opcode number for this form encodes the OPERAND WIDTHS -
    # op=3 mode=1 is 10279 at dest_w=1 src1_w=1, 10280 at src1_w=0 and 10288 at dest_w=0 - so a
    # kernel doing sixteen-bit arithmetic emits an opcode the family row (which names 10279 and
    # 10282) does not describe. `narrow` and `ushort16` both do. Same law as the half load: a
    # sixteen-bit operand is a different FORM, not a field of one.
    for _opc, _s1w, _dw in ((10280, 0, 1), (10288, 1, 0)):
        out.append(("alu.12", _opc, 12, [_opc],
                    (lambda s1, d: lambda t: g17asm.encode_alu(
                        dest=3, src1=2, template=t, op=3, mode=1, imm=9, src1_w=s1, dest_w=d,
                        srcb_w=1, scale_code=2, keep=1, load_wait=0, hazard=0))(_s1w, _dw)))
    # THE SIXTEEN-BIT FORMS THIS BACKEND EMITS, which the family rows above do not cover. A row
    # keyed (form, None, length) describes the corpus opcodes it names and no others - substituting
    # it for a neighbour "hands one opcode's form-defining bits to another", as the note above says
    # - and the half load op12646 and the sixteen-bit read_sr op14060 are neighbours, not members.
    # Without these rows the compiler emits instructions nothing has characterised, so no bit of
    # them can be called settled or inherited either way.
    out.append(("load.14", 12646, 14, [12646],
                lambda t: g17asm.encode_load(dest=3, base=0, offset=0, template=t, index_reg=2,
                                             disp2=0, index_scale=1, narrow=0, hi16=0)))
    out.append(("read_sr.4", 14060, 4, [14060],
                lambda t: g17asm.encode_sr(dest=3, sr=160, seq=0, template=t, half=1)))
    out.append(("branch.cond.back", 458, 10, [458], lambda t: g17asm.encode_branch10(t, -8)))
    out.append(("branch.cond.fwd", 462, 10, [462], lambda t: g17asm.encode_branch10(t, 8)))
    return out
def hook_from(table):
    """-> a TEMPLATE_HOOK that serves this table and REFUSES anything it does not cover.

    No fallback. A resolver that substitutes a same-length entry of the same form name hands one
    opcode's form-defining bits to another, and since the encoders do not write those bits the
    result is a different instruction that decodes perfectly.
    """
    def hook(form, opcode, length, default):
        key = (form, opcode, length)
        if key not in table:
            raise KeyError("no form constant for form=%r opcode=%r length=%d; a template would "
                           "have been inherited here" % (form, opcode, length))
        return table[key]["value"]
    return hook
# ANCHORED ON THE CHECKOUT ROOT: two levels up from agxforge/g17/ where one sufficed from tools/.
ROOT = os.path.dirname(os.path.dirname(HERE))
TOML = os.path.join(ROOT, "isa", "g17-form-constants.toml")
def load(path=TOML):
    """-> the table, from the checked-in TOML. No Apple object is opened."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    doc = tomllib.load(open(path, "rb"))
    return {(f["name"], None if f["opcode"] < 0 else f["opcode"], f["length"]):
            dict(written=bytes.fromhex(f["written_mask"]), value=bytes.fromhex(f["value"]),
                 unresolved=f.get("unresolved", []), instances=f["corpus_instances"],
                 roles=f.get("roles", []), form=f["name"], opcode=f["opcode"], length=f["length"])
            for f in doc["form"]}
