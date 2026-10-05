#!/usr/bin/env python3
"""The operand class SIGNATURE of an opcode, from Apple's own MCInstrDesc.

tools/g17layout.py answers which BITS an operand occupies. This answers what the operand IS:
Apple declares, per opcode, an ordered list of MCOperandInfo giving a register class id, an
operand type and a flag set. That declaration is the authoring contract - it says how many
operands the instruction takes, which are definitions, which are registers of which class and
which are immediates - and it is ground truth rather than inference, because the disassembler
reads the same table to decide what to print.

    python3 tools/g17opclass.py 10094 16874 16906     signature of named opcodes
    python3 tools/g17opclass.py --corpus 10094        signature plus the corpus forms

TYPE and FLAGS are LLVM's MCOI enums. OPERAND_UNKNOWN(0) on a register-class operand is normal:
the class carries the type. Flags are bit positions - LookupPtrRegClass 0, Predicate 1,
OptionalDef 2, BranchTarget 3.
"""
import collections, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# ANCHORED ON THE CHECKOUT ROOT: two levels up from agxforge/g17/, where one sufficed from
# tools/. The native helpers do not move - the Makefile keeps building them into tools/.
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.path.join(ROOT, "tools")

TYPES = {0: "unknown", 1: "imm", 2: "reg", 3: "mem", 4: "pcrel"}
FLAGS = {0: "lookup_ptr_regclass", 1: "predicate", 2: "optional_def", 3: "branch_target"}


def classes():
    """{class id: (name, bit width, member count)}."""
    out = {}
    for line in subprocess.run([os.path.join(TOOLS, "agx3meta"), "classes"],
                               capture_output=True, text=True).stdout.splitlines():
        if line.startswith("#"):
            continue
        i, name, bits, nregs = line.split()
        out[int(i)] = (name, int(bits), int(nregs))
    return out


# LLVM's generic instruction Flags, the 32-bit word at MCInstrDesc +8. Independent of the
# target-specific word at +16, and the only thing in this project that describes an instruction's
# CONTROL FLOW - which makes it the one instrument that can audit an instruction selector's
# branches, calls and returns from outside the selector.
MCID = ["PreISelOpcode", "Variadic", "HasOptionalDef", "Pseudo", "Meta", "Return", "EHScopeReturn",
        "Call", "Barrier", "Terminator", "Branch", "IndirectBranch", "Compare", "MoveImm",
        "MoveReg", "Bitcast", "Select", "DelaySlot", "FoldableAsLoad", "MayLoad", "MayStore",
        "MayRaiseFPException", "Predicable", "NotDuplicable", "UnmodeledSideEffects", "Commutable",
        "ConvertibleTo3Addr", "UsesCustomInserter", "HasPostISelHook", "Rematerializable",
        "CheapAsAMove", "ExtraSrcRegAllocReq", "ExtraDefRegAllocReq"]


def mcid_names(v):
    return [MCID[i] for i in range(len(MCID)) if v >> i & 1]


def _regs(s):
    """An implicit-operand list. "-" is none; "?" is a pointer that failed agx3meta's validation
    and is a HOLE, not an empty set - the difference matters to anything that would conclude an
    instruction touches nothing."""
    if s == "-":
        return ()
    if s == "?":
        return None
    return tuple(int(x) for x in s.split(","))


def instrs():
    """{opcode: {nops, ndefs, schedclass, tsflags, flags8, operands, implicit_uses/defs}}."""
    out = {}
    for line in subprocess.run([os.path.join(TOOLS, "agx3meta"), "instrs"],
                               capture_output=True, text=True).stdout.splitlines():
        if line.startswith("#"):
            continue
        p = line.split()
        # COLUMNS 5, 6 AND 7 ARE NEW and this parser silently broke when they appeared: it read
        # p[5:] as the operand list, and `flags8` has no colons in it. Every tool that reads an
        # instruction description went down at once. Parse by prefix instead of by position, so
        # the next column agx3meta grows cannot do the same thing.
        flags8, implicit_uses, implicit_defs, ops = 0, (), (), []
        for tok in p[5:]:
            if tok.startswith("uses="):
                implicit_uses = _regs(tok[5:])
            elif tok.startswith("defs="):
                implicit_defs = _regs(tok[5:])
            elif ":" in tok:
                rc, ty, fl = tok.split(":")
                ops.append((int(rc), int(ty), int(fl)))
            else:
                flags8 = int(tok, 16)
        out[int(p[0])] = dict(nops=int(p[1]), ndefs=int(p[2]), schedclass=int(p[3]),
                              tsflags=int(p[4], 16), flags8=flags8, operands=ops,
                              implicit_uses=implicit_uses, implicit_defs=implicit_defs)
    return out


def flag_names(fl):
    return ",".join(n for b, n in FLAGS.items() if fl >> b & 1) or "-"


def signature(op, desc, cls):
    """One line per operand, definitions marked."""
    d = desc[op]
    lines = ["op%-6d nops=%d ndefs=%d sched=%d tsflags=0x%x"
             % (op, d["nops"], d["ndefs"], d["schedclass"], d["tsflags"])]
    for k, (rc, ty, fl) in enumerate(d["operands"]):
        role = "def " if k < d["ndefs"] else "use "
        name = cls[rc][0] if rc in cls else ("-" if rc < 0 else "class%d" % rc)
        width = "%db" % cls[rc][1] if rc in cls else ""
        lines.append("    %d %s %-24s %-5s %-4s %s"
                     % (k, role, name, width, TYPES.get(ty, str(ty)), flag_names(fl)))
    return "\n".join(lines)


def corpus_forms(op):
    """The (length, printed operand kinds) forms the corpus actually contains."""
    from agxforge.g17 import fields as g17fields
    rows, _ = g17fields.instances({op})
    forms = collections.Counter()
    for b, ops in rows.get(op, []):
        forms[(len(b), tuple(k for k, _ in ops))] += 1
    return forms


def main():
    want = [int(a) for a in sys.argv[1:] if a.isdigit()]
    cls, desc = classes(), instrs()
    for op in want:
        if op not in desc:
            print("op%d NOT IN MCInstrDesc (%d opcodes)" % (op, len(desc)))
            continue
        print(signature(op, desc, cls))
        if "--corpus" in sys.argv:
            for (ln, kinds), n in corpus_forms(op).most_common():
                print("    corpus %2dB %-40s n=%d" % (ln, " ".join(kinds), n))
        print()


if __name__ == "__main__":
    main()
