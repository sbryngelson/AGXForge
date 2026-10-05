"""No shipped builder hands cc an operation whose result nobody reads.

cc keeps an unread pure op: it has no dead-code pass, so a value built and never used is still emitted and executed.
That has cost us twice, both times inside the hottest loop:
- MM 25.141.3: an x index formed for every element, 15 unread adds per trip.
- MM 25.141.17: the per-position pre-scaled x formed unconditionally while the a16 path reads only xq16, 12-14 unread
  fmuls per trip in every shipped q4/q8 qmv loop (121 -> 107 instructions per trip for q4 split-K w2).
Neither changed a value, so no bit-exact check could see it. This check reads the IR each builder hands to cc and
fails on any unread result of a side-effect-free op, including chains that feed only dead ops. Compile-only."""
import json, os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))

EFFECTS = ("store", "atomic", "barrier", "br", "ret", "machine")
BASE = dict(interleave=True, lean=True, coalesced=True, a16=True, xvec=True, vload=True, hoist_consts=True)


def dead_ops(fn):
    """[(kind, name)] of ops whose result is never read, iterated to a fixed point. An unread constant counts only
    inside a loop body (a block holding a phi): outside one it is a single instruction per thread, once."""
    ops = [o for b in fn.blocks for o in b.ops]
    in_loop = {id(o) for b in fn.blocks if any(o.kind == "phi" for o in b.ops) for o in b.ops}
    dead = set()
    while True:
        used = {id(a) for o in ops if id(o) not in dead for a in o.args}
        new = {id(o) for o in ops if id(o) not in dead and o.dest is not None and id(o.dest) not in used
               and not o.kind.startswith(EFFECTS)}
        if not new:
            return [(o.kind, getattr(o.dest, "name", "")) for o in ops
                    if id(o) in dead and (o.kind != "const" or id(o) in in_loop)]
        dead |= new


def captured(build):
    """The IR function(s) a build hands to cc.compile_function."""
    from agxforge.g17 import cc
    got, orig = [], cc.compile_function
    def cap(fn, *a, **k):
        got.append(fn)
        return orig(fn, *a, **k)
    cc.compile_function = cap
    try:
        build()
    finally:
        cc.compile_function = orig
    return got


class NoDeadOps(unittest.TestCase):
    def check(self, label, build, known=()):
        fns = captured(build)
        self.assertTrue(fns, label)
        for fn in fns:
            self.assertEqual(sorted(dead_ops(fn)), sorted(known), label)

    def test_the_split_k_projections(self):
        import g17qmv as Q
        for bits in (4, 8):
            for kind, (K, N) in (("qkv", (2048, 4096)), ("wo", (2048, 2048)), ("w2", (8192, 2048))):
                for hi16, ptr in ((False, False), (True, False), (True, True)):
                    wpt = 4 if (bits == 8 and kind == "w2") else 2
                    lay = dict(Q.case(N, K, bits, 1, nocarrier=True)[0], **dict(BASE, wpt=wpt), sgs=4, ksplit=True,
                               coop=True, hi16_scales=hi16, ptr_addr=ptr)
                    if kind == "wo":
                        lay = Q.with_residual(lay, "add16")
                    if kind == "w2":
                        lay = Q.with_residual(lay, "add32_to16")
                    self.check("q%d %s hi16=%s ptr=%s" % (bits, kind, hi16, ptr), lambda lay=lay: Q.build_qmv2(lay))

    def test_the_fused_ffn(self):
        import g17qmv as Q
        for bits in (4, 8):
            lay = dict(Q.qmv_swiglu_layout(8192, 2048, bits, 1), **dict(BASE, wpt=2), act32=True, sgs=2, ksplit=True, coop=True)
            self.check("q%d ffn" % bits, lambda lay=lay: Q.build_qmv2(lay))

    def test_the_wide_norms(self):
        import g17decodeops as O
        lay = O.rmsnorm_loop_layout(2048, "half", groups=32, unroll=16, hoist=True)
        for flags in (dict(rs_once=True, out32=True), dict(rs_seed=True, out32=True)):
            self.check("norm %s" % flags, lambda f=flags: O.build_rmsnorm_wide(dict(lay, **f), 1e-5))

    def test_the_attention(self):
        import g17attn as A
        lay = A.with_attn32(A.with_bfly_merge(A.with_wide(A.with_fused_merge(A.with_rope_tables(A.attn_rope_layout())))))
        # KNOWN, left on purpose: the wide form never reads q0 - 1 (only the register-select form does). It is one
        # instruction per thread outside the key loop, and removing it moves the delivered widebf bytes that
        # test_g17cap2048 pins and Piece A's graph runs; remove it at the next attention redelivery.
        self.check("attn widebf", lambda: A.build_attn_split_rope(lay), known=[("sub", "q0m1")])

    def test_the_check_can_fail(self):
        from agxforge.g17 import ir
        fn = ir.Function("f", [ir.Buffer("C", 0, elem=ir.F32)])
        b = ir.Builder(fn, fn.block("entry"))
        x = b.const(1, name="x")
        b.add(x, x, name="unread")
        b.ret()
        self.assertEqual([k for k, _ in dead_ops(fn)], ["add"])


if __name__ == "__main__":
    unittest.main()
