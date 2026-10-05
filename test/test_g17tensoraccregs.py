"""The register accumulator (MM 25.144.8), compile-only. Hardware receipt: tools/g17tensoraccregs.py.

cc keeps a named accumulator's 16x16 fp32 tile in eight fixed registers per lane (below R64, reserved from
every body and from scalar allocation). A body with acc= writes D = A @ B + C back into them (tlower's
c_inplace), and tensor_acc_read / tensor_acc_write move one register to and from a scalar value.

- the lane/register -> (row, col) map is a bijection, and it is tlower's own C-store layout, not transpose.pos;
- the probe's reads name the accumulator's registers in slot order, after the in-place adds;
- the looped accumulator compiles, passes the counted-loop latch check, and nothing but the accumulator's own
  moves and adds writes its registers;
- tlower refuses c_inplace with a store, and the IR and cc refuse the malformed requests by name."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from agxforge.g17 import cc, ir, model, tensorview, tensorlife, tlower  # noqa: E402
import g17tensoraccregs as A  # noqa: E402

MOVE, FADD, OR = 586, 998, 13574          # the plain move, the fp32 add, the bitwise OR with an immediate


def _regs(inst):
    names = model.registers()
    return [names.get(x) for k, x in list(next(model.decode(inst.raw, 0)).values) if k == "reg"]


class Layout(unittest.TestCase):
    def test_the_map_is_a_bijection_onto_the_tile(self):
        got = {ir.tensor_acc_position(l, s) for l in range(32) for s in range(8)}
        self.assertEqual(got, {(r, c) for r in range(16) for c in range(16)})

    def test_it_is_tlowers_store_layout(self):
        """indexgen.prologue: m = 4 lane[4] + lane[2:1], col = 8 lane[3] + 4 lane[0]; registers 0..3 at row m,
        4..7 at row m + 8 (tlower's two C stores)."""
        for l in range(32):
            m, col = 4 * (l >> 4) + ((l >> 1) & 3), (l & 8) + 4 * (l & 1)
            for s in range(8):
                self.assertEqual(ir.tensor_acc_position(l, s), (m + 8 * (s >> 2), col + (s & 3)))

    def test_it_is_not_the_transpose_pos_labeling(self):
        inv = {A.transpose_pos(r, c): (r, c) for r in range(16) for c in range(16)}
        same = sum(inv[(l, s)] == ir.tensor_acc_position(l, s) for l in range(32) for s in range(8))
        self.assertLess(same, 256)


class Programs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = tensorview.view(bytes(A.compile_arm("probe").code))
        cls.probe_regs = dict(cc._TENSOR_ACC_REGS["O"])
        cls.loop_code = bytes(A.compile_arm("loop").code)
        cls.loop_carried = tuple(cc._TENSOR_INDEX_USED)      # the index registers the bodies name (cc's record)
        cls.loop = tensorview.view(cls.loop_code)
        cls.loop_regs = dict(cc._TENSOR_ACC_REGS["O"])

    def test_the_accumulators_sit_at_the_top_and_the_wide_one_spans_both_write_forms(self):
        self.assertEqual(self.probe_regs, {(0, 0): 112})
        self.assertEqual(sorted(self.loop_regs.values()), [56, 64, 72, 80, 88, 96, 104, 112])
        ops = [i.opcode for i in self.loop]
        self.assertEqual((ops.count(OR) > 0, ops.count(MOVE) > 0), (True, True))

    def test_the_probe_reads_the_accumulator_in_slot_order_after_the_adds(self):
        adds = [n for n, i in enumerate(self.probe) if i.opcode == FADD]
        self.assertEqual([_regs(self.probe[n])[0] for n in adds], ["R%d" % (112 + s) for s in range(8)])
        reads = [n for n, i in enumerate(self.probe) if i.opcode == MOVE and n > adds[-1]]
        self.assertEqual([_regs(self.probe[n])[1] for n in reads], ["R%d" % (112 + s) for s in range(8)])
        writes = [n for n, i in enumerate(self.probe) if i.opcode == OR and n < adds[0]]
        self.assertEqual([_regs(self.probe[n])[0] for n in writes], ["R%d" % (112 + s) for s in range(8)])

    def test_the_loop_passes_the_latch_check(self):
        self.assertEqual(self.loop_carried, (125,))
        got = tensorlife.counted_loop_check(self.loop_code, A.TRIPS, carried=self.loop_carried)
        self.assertEqual((got["trips"], got["hazards"], got["latch_releases"]), (A.TRIPS, 0, 0))

    def test_only_the_accumulators_writes_and_adds_write_its_registers(self):
        for v, regs in ((self.probe, self.probe_regs), (self.loop, self.loop_regs)):
            acc = {"R%d" % (g + s) for g in regs.values() for s in range(8)}
            for i in v:
                d = _regs(i)[:1]
                if d and d[0] in acc:
                    self.assertIn(i.opcode, (MOVE, OR, FADD), "op%d writes %s" % (i.opcode, d[0]))

    def test_the_loop_rescales_every_register_inside_the_loop(self):
        (first, back), = tensorview.loops(self.loop)
        body = self.loop[first:back + 1]
        acc = {"R%d" % (g + s) for g in self.loop_regs.values() for s in range(8)}
        writes = [_regs(i)[0] for i in body if i.opcode in (MOVE, OR) and _regs(i)[0] in acc]
        self.assertEqual(sorted(writes), sorted(acc))


class SharedOperand(unittest.TestCase):
    """(4): four simdgroups, each its own 16 rows of A and its own register accumulator, all reading the same B."""
    @classmethod
    def setUpClass(cls):
        cls.code = bytes(A.compile_arm("sg4").code)
        cls.regs = dict(cc._TENSOR_ACC_REGS["O"])
        cls.carried = tuple(cc._TENSOR_INDEX_USED)

    def test_it_compiles_and_passes_the_latch_check(self):
        got = tensorlife.counted_loop_check(self.code, A.TRIPS, carried=self.carried)
        self.assertEqual((got["trips"], got["hazards"], got["latch_releases"], got["advances"]["R125"]),
                         (A.TRIPS, 0, 0, 1))

    def test_the_accumulator_is_per_simdgroup(self):
        self.assertEqual(len(self.regs), A.LOOP_N // 16)       # one simdgroup's 16 rows, not all 64

    def test_it_reads_the_simdgroup_index(self):
        names = model.registers()
        srs = {names.get(list(next(model.decode(i.raw, 0)).values)[2][1]) for i in tensorview.view(self.code)
               if i.opcode in (14059, 14060)}
        self.assertIn("SR_SIMD_GRP", srs)

    def test_the_head_grid_composes_with_the_simdgroup_split(self):
        code = bytes(A.compile_arm("sg4h").code)
        got = tensorlife.counted_loop_check(code, A.TRIPS, carried=tuple(cc._TENSOR_INDEX_USED))
        self.assertEqual((got["trips"], got["hazards"], got["latch_releases"]), (A.TRIPS, 0, 0))
        with self.assertRaisesRegex(ValueError, "head_index"):
            tlower.lower(48, 16, 16, 16, 16, 16, sg=3, head_index=(0, 0, 0))

    def test_a_storing_body_takes_the_b_index_with_a_simdgroup_split(self):
        """M2's QK shape: 64x16x128, transB, B index register, four simdgroups, stored."""
        body, plan = tlower.lower(64, 16, 128, 128, 128, 16, transB=True, sg=4, reserved=tuple(range(16)) + (124, 125),
                                  b_index=(125, 2048), index_init=(125,))
        self.assertEqual(plan["simdgroups"], 4)
        with self.assertRaisesRegex(ValueError, "b_index"):
            tlower.lower(48, 16, 128, 128, 128, 16, transB=True, sg=3, reserved=tuple(range(16)) + (124, 125),
                         b_index=(125, 2048), index_init=(125,))

    def test_simdgroups_must_split_whole_tiles(self):
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        bd = ir.Builder(ir.Function("t", [a, b, c]), None)
        for sg, M in ((3, 48), (4, 32)):
            with self.subTest(sg=sg, M=M), self.assertRaisesRegex(ir.IRError, "simdgroups"):
                bd.tensor_matmul(a, b, c, M=M, N=16, K=16, accumulate=True, acc="O", simdgroups=sg)

    def test_a_stream_mixing_simdgroup_counts_takes_no_route(self):
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("t", [a, b, c]); bd = ir.Builder(fn, fn.block("entry"))
        bd.tensor_matmul(a, b, c, M=64, N=16, K=16, accumulate=True, acc="O", simdgroups=4)
        bd.tensor_matmul(a, b, c, M=64, N=16, K=16, accumulate=True, acc="P")
        bd.ret()
        ops = [o for blk in fn.blocks for o in blk.ops if o.kind == "tensor_matmul"]
        self.assertFalse(cc._memory_stream_group(fn, ops))


def _pressure_fn(n_live=40):
    """QK 16x16x128, then n_live values loaded and all held live at once, summed, stored; then PV into a register O
    of 8 tiles. The tensor union (the QK working set plus 64 accumulator registers) leaves the scalar pool too few
    registers, so it compiles only when values that live across no body take the bodies' working registers."""
    a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("t", [a, b, c]); bd = ir.Builder(fn, fn.block("entry"))
    zero = A._f32_const(bd, 0.0, "zero")
    for t in range(8):
        for s in range(8):
            bd.tensor_acc_write("O", s, zero, tile=(0, t))
    bd.tensor_matmul(a, c, c, M=16, N=16, K=128, transB=True, offsetC=0, offsetB=4096)
    lane = bd.builtin("thread_index_in_simdgroup", name="lane")
    vals = [bd.load(c, bd.add(lane, ir.Imm(32 * k)), type=ir.F32, name="x%d" % k) for k in range(n_live)]
    while len(vals) > 1:
        vals = [bd.fadd(vals[k], vals[k + 1]) if k + 1 < len(vals) else vals[k] for k in range(0, len(vals), 2)]
    bd.store_at(c, lane, vals[0])
    bd.tensor_matmul(c, b, c, M=16, N=128, K=16, a_dtype="float", accumulate=True, acc="O")
    for s in range(8):
        bd.store_at(c, bd.add(lane, ir.Imm(64 + 32 * s)), bd.tensor_acc_read("O", s, tile=(0, 0)))
    bd.ret(); ir.verify(fn)
    return fn


class RegistersBetweenBodies(unittest.TestCase):
    def test_values_between_bodies_take_the_bodies_working_registers(self):
        p = cc.compile_function(_pressure_fn())
        v = tensorview.view(bytes(p.code))
        loads = [i for i in v if i.opcode in (12682, 12646)]                 # the scalar word loads
        dests = {h // 2 for i in loads for h in i.defs}
        self.assertTrue(dests & set(range(16, 56)), "no value between the bodies took a body register")

    def test_nothing_persistent_is_shared(self):
        cc.compile_function(_pressure_fn())
        self.assertTrue({r + i for r in cc._TENSOR_ACC_REGS["O"].values() for i in range(8)} <= cc._TENSOR_PERSISTENT)


def _one_index_loop(P):
    """M6's fuzz case: a counted loop whose body names ONE stream index register ("k", R125), P words held live
    between bodies. At P = 30 the allocator gives R124 - named by no body - to the sum tree."""
    a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("t", [a, b, c]); bd = ir.Builder(fn, fn.block("entry"))
    zero = bd.const(0, type=ir.F32, name="z")
    for t in range(4):
        for s in range(8):
            bd.tensor_acc_write("O", s, zero, tile=(t // 4, t % 4))
    bd.tensor_index_init("k", 0)
    i0 = bd.const(0, name="i0")
    hdr, ex = fn.block("trips"), fn.block("done")
    bd.br(hdr); bd.at(hdr)
    i = bd.phi(i0, name="i")
    bd.tensor_matmul(a, b, c, M=32, N=64, K=16, accumulate=True, acc="O", offsetB_register="k", offsetB_step=2048)
    lane = bd.builtin("thread_index_in_simdgroup", name="lane")
    vals = [bd.load(c, bd.add(lane, ir.Imm(32 * k)), type=ir.I32, name="x%d" % k) for k in range(P)]
    while len(vals) > 1:
        vals = [bd.add(vals[k], vals[k + 1]) if k + 1 < len(vals) else vals[k] for k in range(0, len(vals), 2)]
    bd.store_at(c, lane, vals[0])
    nxt = bd.add(i, ir.Imm(1), name="n")
    ir.Builder.phi_latch(i, nxt)
    bd.br_cond(bd.cmp(nxt, 3, "lt"), hdr, ex)
    bd.at(ex); bd.ret(); ir.verify(fn)
    return fn


class OnlyTheNamedIndexRegistersAreCarried(unittest.TestCase):
    def test_a_loop_naming_one_index_register_may_use_the_other_as_a_scalar(self):
        p = cc.compile_function(_one_index_loop(30))
        self.assertEqual(p._tensor_loop["advances"], {"R125": 1})
        # the premise: R124 really is a scalar value in the body here, so the old check (both registers carried)
        # refuses these same bytes - the change is in what is called carried, not in the check
        with self.assertRaisesRegex(ValueError, "R124"):
            tensorlife.counted_loop_check(bytes(p.code), 3, carried=cc.TENSOR_STREAM_INDEX_REGISTERS)

    def test_the_named_index_register_stays_reserved(self):
        cc.compile_function(_one_index_loop(30))
        self.assertIn(125, cc._TENSOR_PERSISTENT)
        self.assertEqual(cc._TENSOR_INDEX_USED, [125])


class FoldOffsets(unittest.TestCase):
    """fold_offsets: a buffer offset past the 32,767-byte displacement field is added to its index register once,
    instead of re-derived for every tile (tlower based()'s slow path)."""
    def test_it_removes_the_per_tile_base_arithmetic(self):
        far = tensorview.view(bytes(A.compile_arm("far").code))
        fold = tensorview.view(bytes(A.compile_arm("farfold").code))
        (f0, b0), = tensorview.loops(far)
        (f1, b1), = tensorview.loops(fold)
        self.assertLess(b1 - f1, (b0 - f0) // 2)

    def test_the_default_is_unchanged(self):
        a = tlower.lower(16, 16, 16, 16, 16, 16, offsets=(40960, 0, 0))
        b = tlower.lower(16, 16, 16, 16, 16, 16, offsets=(40960, 0, 0), fold_offsets=False)
        self.assertEqual(a[0], b[0])
        c = tlower.lower(16, 16, 16, 16, 16, 16, offsets=(40960, 0, 0), fold_offsets=True)
        self.assertLess(len(c[0]), len(a[0]))

    def test_an_offset_splitting_an_element_is_refused(self):
        with self.assertRaisesRegex(ValueError, "fold_offsets"):
            tlower.lower(16, 16, 16, 16, 16, 16, offsets=(40961, 0, 0), fold_offsets=True)


class ScaleAndHoist(unittest.TestCase):
    """tensor_acc_scale (one multiply on the register) and hoist_prologue (the body's invariant index prologue run once,
    before the loop): the loop body shrinks and the latch still proves."""
    def test_the_scale_is_one_instruction_per_register(self):
        a = tensorview.view(bytes(A.compile_arm("loop").code))
        b = tensorview.view(bytes(A.compile_arm("loopscale").code))
        (fa, ba), = tensorview.loops(a)
        (fb, bb), = tensorview.loops(b)
        self.assertEqual((ba - fa) - (bb - fb), 2 * 8 * (A.LOOP_N // 16))      # read + write gone, per register

    def test_a_loaded_factor_carries_the_load_wait(self):
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("t", [a, b, c]); bd = ir.Builder(fn, fn.block("entry"))
        bd.tensor_matmul(a, b, c, M=16, N=16, K=16, accumulate=True, acc="O")
        lane = bd.builtin("thread_index_in_simdgroup")
        bd.tensor_acc_scale("O", 0, bd.load(c, lane, type=ir.F32))
        bd.store_at(c, lane, bd.tensor_acc_read("O", 0))
        bd.ret(); ir.verify(fn)
        v = tensorview.view(bytes(cc.compile_function(fn).code))
        muls = [i for i in v if i.opcode == 3290]
        self.assertEqual(len(muls), 1)
        self.assertTrue(muls[0].raw[0] & 0x08, "the multiply must wait for the loaded factor (slot 7)")

    def test_the_hoisted_prologue_leaves_the_loop(self):
        far = tensorview.view(bytes(A.compile_arm("farfold").code))
        fast = tensorview.view(bytes(A.compile_arm("fast").code))
        (f0, b0), = tensorview.loops(far)
        (f1, b1), = tensorview.loops(fast)
        # the fast arm also scales in place (-128); what is left over is the hoisted prologue, less its per-trip add
        self.assertGreater((b0 - f0) - (b1 - f1) - 128, 5)
        self.assertEqual(sum(i.opcode in (14059, 14060) for i in fast[f1:b1 + 1]), 0, "no SR read left in the loop")

    def test_hoist_outside_a_loop_is_refused(self):
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("t", [a, b, c]); bd = ir.Builder(fn, fn.block("entry"))
        bd.tensor_matmul(a, b, c, M=16, N=16, K=16, accumulate=True, acc="O", hoist_prologue=True)
        bd.store_at(c, bd.builtin("thread_index_in_simdgroup"), bd.tensor_acc_read("O", 0))
        bd.ret(); ir.verify(fn)
        with self.assertRaisesRegex(cc.Unsupported, "hoist_prologue"):
            cc.compile_function(fn)


class Refusals(unittest.TestCase):
    def test_tlower_refuses_c_inplace_with_a_store(self):
        with self.assertRaisesRegex(ValueError, "c_inplace"):
            tlower.lower(16, 16, 16, 16, 16, 16, accumulate=True, c_regs={(0, 0): 56}, c_inplace=True,
                         reserved=tuple(range(16)) + tuple(range(56, 64)))

    def test_an_acc_body_must_accumulate(self):
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        bd = ir.Builder(ir.Function("t", [a, b, c]), None)
        with self.assertRaisesRegex(ir.IRError, "accumulate"):
            bd.tensor_matmul(a, b, c, M=16, N=16, K=16, acc="O")

    def _fn(self, read_name="O", tile=(0, 0)):
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("t", [a, b, c]); bd = ir.Builder(fn, fn.block("entry"))
        bd.tensor_matmul(a, b, c, M=16, N=16, K=16, accumulate=True, acc="O")
        v = bd.tensor_acc_read(read_name, 0, tile=tile)
        bd.store_at(c, bd.builtin("thread_index_in_simdgroup"), v)
        bd.ret(); ir.verify(fn)
        return fn

    def test_an_unknown_accumulator_or_tile_is_refused(self):
        with self.assertRaisesRegex(cc.Unsupported, "no tensor body accumulates"):
            cc.compile_function(self._fn(read_name="P"))
        with self.assertRaisesRegex(cc.Unsupported, "outside its 1 tiles"):
            cc.compile_function(self._fn(tile=(1, 0)))


if __name__ == "__main__":
    unittest.main()
