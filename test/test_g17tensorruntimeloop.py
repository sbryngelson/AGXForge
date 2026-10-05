"""The capped RUNTIME trip count around tensor bodies (MM 25.144.8), compile-only.

cc admits a key-block loop whose latch is `i + 1 < n, cap=K` when n is provably simdgroup-uniform, and proves
the emitted latch from the bytes (tensorlife.counted_loop_check(runtime=True)): the counter rises by exactly 1
per trip and the loop continues only while it is below the cap, so it ends within K trips whatever n is. The
hardware receipt is tools/g17tensorruntimeloop.py.

- the compile-time loop keeps its preregistered bytes (loop_n8, efd83e924bc2b196...);
- the runtime loop compiles, and its latch check proves cap 8 with 0 hazards;
- lane-varying bounds, missing caps and caps past the measured runtime maximum are refused by name;
- the byte check FAILS on a wrong cap and on a patched cap immediate (the check can fail)."""
import hashlib, os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from agxforge.g17 import cc, ir, tensorlife, model, tensorview  # noqa: E402
import g17tensorcommonruntime as R  # noqa: E402

REQ = {"new_blocks": 8, "key_offsets": "loop"}
LOOP_N8 = "efd83e924bc2b196"          # tools/g17keyblockloop.py PREREGISTERED['loop_n8']


def _runtime_program(cap=8):
    R.LOOP_RUNTIME_CAP = cap
    try:
        return R.build_generic_program(R.generic_spec({"attention": dict(REQ)}))
    finally:
        R.LOOP_RUNTIME_CAP = None


def _bound_fn(bound, cap=8):
    """A minimal stream loop whose latch compares against `bound(builder, a, c)`."""
    a = ir.Buffer("A", 1, elem=ir.F16); bb = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("t", [a, bb, c]); b = ir.Builder(fn, fn.block("entry"))
    n = bound(b, a, c)
    b.tensor_index_init("k", 65536)
    i0 = b.const(0, name="i0")
    hdr, ex = fn.block("keys"), fn.block("done")
    b.br(hdr); b.at(hdr)
    i = b.phi(i0, name="i")
    b.tensor_matmul(a, c, c, M=32, N=16, K=128, transB=True, offsetA=0, offsetB=0, offsetC=0,
                    offsetB_register="k", offsetB_step=4096)
    b.tensor_matmul(c, bb, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=True, offsetA=0,
                    offsetB=0, offsetC=2048)
    nxt = b.add(i, ir.Imm(1), name="n")
    ir.Builder.phi_latch(i, nxt)
    b.br_cond(b.cmp(nxt, n, "lt", cap=cap) if cap is not None else b.cmp(nxt, n, "lt"), hdr, ex)
    b.at(ex); b.ret(); ir.verify(fn)
    return fn


class RuntimeTensorLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt = _runtime_program()
        cls.code = bytes(cls.rt.code)

    def test_the_compile_time_loop_keeps_its_bytes(self):
        p = R.build_generic_program(R.generic_spec({"attention": dict(REQ)}))
        self.assertEqual(hashlib.sha256(bytes(p.code)).hexdigest()[:16], LOOP_N8)

    def test_the_runtime_loop_is_proved_from_its_bytes(self):
        got = self.rt._tensor_loop
        self.assertEqual((got["runtime"], got["cap"], got["trips"], got["hazards"], got["latch_releases"]),
                         (True, 8, 8, 0, 0))
        self.assertEqual(got["advances"], {"R125": 1, "R124": 1})

    def test_bounds_that_are_not_provably_uniform_are_refused(self):
        lane = lambda b, a, c: b.load(c, b.builtin("thread_index_in_simdgroup"), type=ir.I32)
        tpos = lambda b, a, c: b.add(b.builtin("thread_position_in_threadgroup"), ir.Imm(1))
        for name, bound in (("lane load", lane), ("thread position", tpos)):
            with self.subTest(name), self.assertRaisesRegex(cc.Unsupported, "not provably simdgroup-uniform"):
                cc.compile_function(_bound_fn(bound))

    def test_a_uniform_bound_from_the_threadgroup_position_is_admitted_to_the_route(self):
        tg = lambda b, a, c: b.add(getattr(b, "and")(b.builtin("threadgroup_position_in_grid"), ir.Imm(7)), ir.Imm(1))
        route = cc.tensor_loop_route(_bound_fn(tg))
        self.assertEqual((route["runtime"], route["trips"]), (True, 8))

    def test_a_threadgroup_position_bound_compiles_with_its_sr_read_admitted(self):
        """The bound is read from a system register (slot 0, unwaited): MM 25.117/25.121 measured that read
        correct at distance 0, so the check admits it and names the count; the same filter refuses any other
        late read (below)."""
        tg = lambda b, a, c: b.add(getattr(b, "and")(b.builtin("threadgroup_position_in_grid"), ir.Imm(7)), ir.Imm(1))
        got = cc.compile_function(_bound_fn(tg))._tensor_loop
        self.assertEqual((got["runtime"], got["hazards"], got["sr_slot0_reads_admitted"]), (True, 0, 2))

    def test_the_sr_filter_refuses_a_read_whose_producer_is_not_an_sr(self):
        tg = lambda b, a, c: b.add(getattr(b, "and")(b.builtin("threadgroup_position_in_grid"), ir.Imm(7)), ir.Imm(1))
        v = tensorview.view(bytes(cc.compile_function(_bound_fn(tg)).code))
        hz = tensorview.hazards(v)
        self.assertTrue(hz and all(tensorlife._sr_slot0_read(v, h) for h in hz))
        idx, what = hz[0]
        self.assertFalse(tensorlife._sr_slot0_read(v, (idx, what.replace("before slot 0", "before slot 1"))))
        self.assertFalse(tensorlife._sr_slot0_read(v, (0, what)), "no producer before instruction 0")
        n = next(n for n in range(len(v)) if v[n].defs and v[n].opcode not in tensorview.SR)
        h = min(v[n].defs)
        fake = (n + 1, "reads r%d%s before slot 0 is waited on" % (h // 2, "H" if h % 2 else "L"))
        self.assertFalse(tensorlife._sr_slot0_read(v, fake), "a non-SR producer (op%d)" % v[n].opcode)

    def test_a_missing_or_oversized_cap_is_refused(self):
        self.assertEqual(tensorlife.TENSOR_LOOP_MAX_RUNTIME_TRIPS, 8388608,
                         "raising this ceiling needs a new bounded GPU count witness")
        tg = lambda b, a, c: b.builtin("threadgroup_position_in_grid")
        with self.assertRaisesRegex(cc.Unsupported, "cap"):
            cc.compile_function(_bound_fn(tg, cap=None))
        with self.assertRaisesRegex(cc.Unsupported, "cap"):
            cc.compile_function(_bound_fn(tg, cap=tensorlife.TENSOR_LOOP_MAX_RUNTIME_TRIPS + 1))

    def test_the_byte_check_refuses_a_wrong_cap(self):
        with self.assertRaisesRegex(ValueError, "cap"):
            tensorlife.counted_loop_check(self.code, 7, carried=cc.TENSOR_STREAM_INDEX_REGISTERS, runtime=True)

    def test_the_byte_check_refuses_a_patched_cap_immediate(self):
        """Rewrite the loop's cap constant (op11842 ... imm 8) to 9 in the bytes: the check must refuse."""
        v = tensorview.view(self.code)
        (first, back), = tensorview.loops(v)
        k = next(n for n in range(back, first, -1) if v[n].opcode == 11842 and
                 list(next(model.decode(v[n].raw, 0)).values)[-1] == ("imm", 8))
        from agxforge.g17 import asm
        # the immediate's bits are scattered across the word: re-encode the WHOLE operand with cc's own encoder,
        # the emitted instruction as template (its destination and every other field kept)
        raw = asm.encode_movimm(9, v[k].raw)
        self.assertEqual(asm.encode_movimm(8, v[k].raw), v[k].raw, "the encoder must reproduce the emitted cap")
        self.assertEqual(list(next(model.decode(raw, 0)).values)[-1], ("imm", 9))
        patched = self.code[:v[k].offset] + raw + self.code[v[k].offset + len(raw):]
        with self.assertRaisesRegex(ValueError, "cap"):
            tensorlife.counted_loop_check(patched, 8, carried=cc.TENSOR_STREAM_INDEX_REGISTERS, runtime=True)


if __name__ == "__main__":
    unittest.main()
