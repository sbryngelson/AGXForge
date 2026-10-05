"""The q4 decode loop's per-row extras, removed (build_qmv2 lay["chains"], lay["epi_fma"], lay["and16_direct"]; MM 25.144.4).

Against Apple's compile of MLX's qmv the r1 lean loop carried, per 16 weights: 8 fp32 multiplies (op3290) pre-scaling x
for the byte's high field, 2 isolation copies (op10279 add-immediate 0) in front of the field extracts, and a four-op
epilogue (two multiplies, two adds). The flags remove them:

- chains: one fma chain per field position against the RAW x, folded once per trip (t = fma(t_pos, 2^-(bits pos), t)),
  so no pre-scaled x;
- epi_fma: acc = fma(b, sx, fma(s, t, acc));
- and16_direct: a vector-load lane is read in place by the 16-bit AND with immediate (op426) once an earlier consumer of
  that load has waited (cc's first-consumer rule), instead of through a waited copy.

The new order has its own reference: qmv2_ksplit_reference reads chains / epi_fma. Hardware: bit-exact on fp32
outputs at two seeds (25.144.4). Compile-only here: the counts, the reference differing from the old order (so a
bit-exact check can see the order), and the flags defaulting off."""
import collections, os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))

LEAN = dict(interleave=False, lean=True, coalesced=True, a16=True, xvec=True, vload=True, hoist_consts=False, wpt=2,
            hi16_scales=True, ptr_addr=True, sgs=4, ksplit=True, coop=True)


def _lay(bits=4, **kw):
    import g17qmv as Q
    return Q.with_residual(dict(Q.case(2048, 8192, bits, 1, nocarrier=True)[0], **LEAN, **kw), "add32_to16")


def _loop_hist(lay):
    import g17qmv as Q
    from agxforge.g17 import model, tensorview as TV
    code = Q.build_qmv2(lay).code
    first, back = TV.loops(TV.view(code))[0]
    ins = [i for i in model.decode(code, 0) if i.opcode][first:back + 1]
    return collections.Counter(i.opcode.id for i in ins), len(ins)


class Chains(unittest.TestCase):
    def test_the_extras_leave(self):
        h0, n0 = _loop_hist(_lay())
        h1, n1 = _loop_hist(_lay(chains=True, epi_fma=True, and16_direct=True))
        self.assertEqual(n0 - n1, 11, (n0, n1))
        self.assertEqual(h0[3290] - h1[3290], 9)          # 11 -> 2: 8 pre-scales and 2 epilogue multiplies go, a second chain head comes
        self.assertEqual(h0[10279] - h1[10279], 2)        # the two isolation copies
        self.assertEqual((h0[426], h0[11179]), (h1[426], h1[11179]))   # extraction and conversion unchanged

    def test_the_reference_sees_the_order(self):
        import numpy as np, g17qmv as Q
        lay = _lay()
        rng = np.random.default_rng(5)
        Wf = (rng.standard_normal((2048, 8192)) * 0.02).astype(np.float32)
        x = np.asarray(rng.standard_normal(8192), np.float16).astype(np.float32)
        _, s16, b16, q = Q.quantize(Wf, bits=4)
        y0 = Q.qmv2_ksplit_reference(lay, x, q, s16, b16)
        y1 = Q.qmv2_ksplit_reference(dict(lay, chains=True, epi_fma=True), x, q, s16, b16)
        d = int((y0 != y1).sum())
        self.assertGreater(d, 1000)                       # 1615 of 2048 at seed 5
        self.assertLess(float(np.abs(y0 - y1).max()), 1e-5)

    def test_the_flags_default_off(self):
        self.assertEqual(_loop_hist(_lay()), _loop_hist(_lay(chains=False, epi_fma=False, and16_direct=False)))


if __name__ == "__main__":
    unittest.main()
