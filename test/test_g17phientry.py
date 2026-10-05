"""A loop's entry value must still be in its register when the loop starts.

A phi that takes its entry value's register has no copy instruction, so the entry value's last visible use
BEFORE the loop was its last use to liveness, and cc released it there (source lifetime 16). A released register
reads 0, so the loop's first trip started from 0: build_rmsnorm_wide's scale loop starts from the thread index t,
and all 1,024 threads stored element 0 (MM 25.141.4). The fix charges each header phi's entry value to the
instruction that falls into the header.

The check reads the EMITTED bytes: every register a loop reads before writing it (a carried value, entry values
included) must not be released by any instruction between the program start and the loop. It fails on the
unfixed compiler at the wide norm's second x-address add. Compile-only."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from agxforge.g17 import model, tensorview as TV


def released_before_loop(code):
    """[(loop start, instruction index, register)]: a register the loop carries, released before the loop."""
    v = TV.view(code)
    ins = list(model.decode(code, 0))
    out = []
    for first, back in TV.loops(v):
        written, carried = set(), set()
        for n in range(first, back + 1):
            for r in v[n].uses:
                if r not in written:
                    carried.add(r)
            written |= set(v[n].defs)
        for n in range(0, first):
            vals = list(ins[n].values) if ins[n].opcode else []
            regs = [(k, x) for k, (kind, x) in enumerate(vals) if kind == "reg"]
            srcs = regs if v[n].kind == "store" else regs[1:]
            for k, x in srcs:
                if k + 1 < len(vals) and vals[k + 1] == ("imm", 16) and TV._regs(x) & carried:
                    # released, and not rewritten before the loop
                    later = [m for m in range(n + 1, first) if TV._regs(x) & set(v[m].defs)]
                    if not later:
                        out.append((first, n, x))
    return out


class PhiEntry(unittest.TestCase):
    def test_the_wide_norm_keeps_its_loop_start(self):
        import g17decodeops as O
        for dt, u in (("half", 16), ("float", 8)):
            lay = O.rmsnorm_loop_layout(2048, dt, groups=32, unroll=u, hoist=True)
            self.assertEqual(released_before_loop(O.build_rmsnorm_wide(lay, 1e-5).code), [], dt)

    def test_the_shipped_qmv_loops_keep_theirs(self):
        import g17qmv as Q
        B = dict(interleave=True, lean=True, coalesced=True, a16=True, x16=True)
        lay = dict(Q.qmv_swiglu_layout(8192, 2048, 8, 2), **B, wpt=2)
        self.assertEqual(released_before_loop(Q.build_qmv2(lay).code), [])


if __name__ == "__main__":
    unittest.main()
