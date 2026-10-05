#!/usr/bin/env python3
"""The whole-kernel tensor route: what it emits, what it leaves alone, and what it refuses by name.

The route exists because the general lowering returns a COMPLETE kernel - it ends in END and
allocates its own registers - so its instructions can only be emitted where the GEMM is the whole
program. These cases pin all three halves of that: the registry's shapes must be untouched, a
pure-GEMM function the registry refuses must be authored, and anything else must be declined with
a reason rather than served wrongly.

What would make each fail is stated on the case, because a test whose failure mode is unnamed
tends to be a test that cannot fail.
"""
import collections
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

import g17cc                                                             # noqa: E402
from g17cc import TENSOR_ELEMENT_BYTES                                   # noqa: E402
import g17ir as ir                                                       # noqa: E402
from agxforge.g17 import model, tensor                                      # noqa: E402
from agxforge.g17 import registerdomain

class EveryPinnedRefusalIsPreserved(unittest.TestCase):
    """One case per refusal that g17regress._tensor_contract_delivered pins, each asserting the
    exact substring that function asserts. Fails if the route serves a shape the delivered
    contract refuses, or if a preserved refusal loses the wording the contract checks for.
    """

    def refusal(self, M, N, K, **kw):
        f = ir.Function("a", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
        b = ir.Builder(f, f.block("e"))
        b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=M, N=N, K=K, **kw)
        b.ret()
        ir.verify(f)
        with self.assertRaises(g17cc.Unsupported) as caught:
            g17cc.select(f)
        return str(caught.exception)

    def test_A_at_18_tile_rows_routes_through_measured_fold(self):
        insts = select(32, 32, 64, strideA=576)
        self.assertTrue(any(i.form == "tensor.wholekernel" for i in insts))

    def test_B_at_64_tile_rows_routes_through_measured_fold(self):
        insts = select(32, 32, 64, strideB=2048)
        self.assertTrue(any(i.form == "tensor.wholekernel" for i in insts))

    def test_a_stride_past_the_measured_range_still_refuses(self):
        """The lift is bounded by the sweep (A to 1 MiB, B to 512 KiB): one element past either cap is not served
        by the general lowering, and the refusal names the measured range; at the caps it routes."""
        for kw in (dict(strideA=(1 << 20) + 2), dict(strideB=(1 << 19) + 2)):
            self.assertIn("beyond the measured stride range", self.refusal(32, 32, 64, **kw))
        for kw in (dict(strideA=1 << 20), dict(strideB=1 << 19)):
            self.assertTrue(any(i.form == "tensor.wholekernel" for i in select(32, 32, 64, **kw)), kw)

    def test_16x32x64_routes_now_and_the_pin_still_refuses_by_shape(self):
        """The pin was BY SHAPE, not by message: "no witness" appears in every unwitnessed shape's
        reason, so matching it would have declined the whole route. It is lifted - the general
        lowering's 16x32x64 ran bit-exact on hardware (gemm_generic run 1) - and restoring it must
        still refuse by shape."""
        self.assertTrue(select(16, 32, 64))
        saved = g17cc.TENSOR_PIN_16X32X64
        try:
            g17cc.TENSOR_PIN_16X32X64 = True
            self.assertIn("no witness", self.refusal(16, 32, 64))
        finally:
            g17cc.TENSOR_PIN_16X32X64 = saved

    def test_the_no_witness_wording_does_not_itself_decline_the_route(self):
        """The companion to the case above, and the reason the guard is a shape and not a string:
        17x19x16's reason carries the same "no witness" wording and MUST still route."""
        insts = select(17, 19, 16)
        self.assertTrue(any(i.form == "tensor.wholekernel" for i in insts))


WITNESSED = (32, 32, 64)          # the shape tensorlower's registry serves from Apple's witness
GENERAL = [(17, 19, 16), (50, 37, 80), (16, 16, 16)]


def select(M, N, K, extra=False, **kw):
    f = ir.Function("a", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("e"))
    b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=M, N=N, K=K, **kw)
    if extra:
        t = b.builtin("threadgroup_position_in_grid", name="t")
        b.store_at(f.buffers[2], t, b.add(t, ir.Imm(1), name="v"))
    b.ret()
    ir.verify(f)
    return g17cc.select(f)


class TheRegistrysShapesAreUntouched(unittest.TestCase):
    """Fails if the route is reached for a shape the registry already serves - which would change
    bytes that Apple's witness certifies."""

    # The registry path reads Apple's retained witness objects, which live under results/ and are
    # GITIGNORED. In a fresh checkout they are absent and this case errored with a
    # FileNotFoundError - which would fail root's fresh-main replay on a missing fixture rather
    # than on anything about the compiler. It skips with the path named instead. The skip is
    # deliberately not silent: it states which invariant goes unverified, because a skip that
    # reads like a pass is worse than the error it replaced.
    WITNESS = os.path.join(ROOT, "results", "g17-tensor-common-witness-v1", "tensor-common.o")

    def test_the_witnessed_shape_emits_no_wholekernel_row(self):
        if not os.path.exists(self.WITNESS):
            self.skipTest(
                "the registry's witness object is absent (%s): it is gitignored, so in a fresh "
                "checkout the registry path cannot run and 'the witnessed shape is untouched by "
                "the route' is UNVERIFIED here. Run where results/ is populated to check it."
                % self.WITNESS)
        forms = collections.Counter(i.form for i in select(*WITNESSED))
        self.assertEqual(forms.get("tensor.wholekernel", 0), 0)
        self.assertTrue(forms["tensor.inherited"] > 0, "the witness rows are gone")
        self.assertTrue(forms["tensor.authored"] > 0)


class APureGemmTheRegistryRefusesIsAuthored(unittest.TestCase):
    """Fails if a shape the registry cannot serve goes back to raising Unsupported."""

    def test_each_general_shape_routes_whole_kernel(self):
        for shape in GENERAL:
            with self.subTest(shape=shape):
                insts = select(*shape)
                forms = collections.Counter(i.form for i in insts)
                self.assertTrue(forms.get("tensor.wholekernel", 0) > 0,
                                "%r emitted no whole-kernel row" % (shape,))
                self.assertEqual(forms.get("tensor.inherited", 0), 0,
                                 "%r inherited witness bytes it has no witness for" % (shape,))

    def test_the_rows_are_the_lowerings_body_without_its_END(self):
        """The route drops the lowering's END because this compiler emits its own for `ret`.

        Fails if the END is kept (an END mid-program) or if any instruction is lost or reordered:
        the concatenated row bytes must equal the body exactly, minus that one trailing END.
        """
        for shape in GENERAL:
            with self.subTest(shape=shape):
                body = bytes(tensor.emit_gemm(*shape).body)
                decoded = [i for i in model.decode(body, 0) if i.opcode]
                self.assertEqual(decoded[-1].opcode.id, 684, "the lowering's body must end in END")
                want = body[:len(body) - len(decoded[-1].raw)]
                rows = [i for i in select(*shape) if i.form == "tensor.wholekernel"]
                self.assertEqual(b"".join(i.fields["bytes"] for i in rows), want)
                self.assertEqual(len(rows), len(decoded) - 1)
                self.assertNotIn(684, [i.fields["opcode"] for i in rows])

    def test_the_occupied_register_set_is_published_once_and_is_precise(self):
        """The allocator learns the body's physical registers from `_occupies` on the first row.

        This replaced a single CEILING declaration (plan["registers"] + 1 as a def). A ceiling
        reserves a range and says nothing about which registers inside it are written, so a
        surrounding scalar value could still be coloured into a hole the body uses. Fails if the
        set is missing, published on more than one row, or coarser than the body's own operands.
        """
        rows = [i for i in select(*GENERAL[0]) if i.form == "tensor.wholekernel"]
        publishing = [i for i in rows if "_occupies" in i.fields]
        self.assertEqual(len(publishing), 1, "_occupies must be published exactly once")
        self.assertIs(publishing[0], rows[0], "it must be the first row")
        occupied = set(publishing[0].fields["_occupies"])
        self.assertTrue(occupied)
        names = model.registers()
        actual = set()
        for row in rows:
            for inst in model.decode(row.fields["bytes"], 0):
                if inst.opcode is None:
                    continue
                for kind, value in inst.values:
                    if kind == "reg":
                        actual.update(registerdomain.registers_in_name(names.get(value, "") or ""))
        self.assertEqual(occupied, actual,
                         "the published set must be exactly the body's register operands")


class AnythingElseIsDeclinedByName(unittest.TestCase):
    """Fails if a function the route cannot serve is served anyway, or refused without a reason."""

    def test_scalar_work_beside_the_tensor_op_is_now_COMPOSED_not_declined(self):
        """This case asserted the opposite until the row-splice slice landed.

        It previously required a function with scalar work to be REFUSED, which was correct while
        the allocator could not be told what the tensor body occupies. That is the behaviour the
        slice deliberately changes, so the case is inverted rather than deleted - keeping it
        records that the old refusal was a real limit and not an oversight, and it fails if the
        composition silently stops working and the refusal comes back.
        """
        insts = select(17, 19, 16, extra=True)
        self.assertTrue(any(i.form == "tensor.wholekernel" for i in insts))
        self.assertTrue(any(i.form not in ("tensor.wholekernel", "end") for i in insts),
                        "the scalar work vanished from the stream")

    def test_the_lowerings_own_reason_is_carried_when_it_refuses(self):
        """When the general lowering refuses, the compiler's message must carry ITS reason and not
        only the registry's older one.

        `Builder.tensor_matmul` cannot express simdgroups or int8 partial tiles, so the reachable
        refusal here is a stride the lowering rejects. Fails if the message reports only the
        registry's reason, which would send a reader to the wrong lowering.
        """
        with self.assertRaises(Exception) as caught:
            select(17, 19, 16, strideA=3)          # 3 bytes is not a whole number of halves
        message = str(caught.exception)
        self.assertIn("general lowering", message,
                      "the compiler must say which lowering refused")


class TheRouteReachesTheEmittedProgram(unittest.TestCase):
    """Selection is not emission. These use emit() with NO forms argument, so the templates come
    from the repository's own _const_forms() and no generated or external input is involved."""

    def compile(self, M, N, K, **kw):
        f = ir.Function("a", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
        b = ir.Builder(f, f.block("e"))
        b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=M, N=N, K=K, **kw)
        b.ret()
        ir.verify(f)
        return g17cc.emit(g17cc.Alloc(regs=range(0, 126)).run(g17cc.select(f)))

    def test_the_program_is_the_lowerings_body(self):
        """Fails if emission drops, reorders or re-encodes the route's bytes: the compiled program
        must be the lowering's own body, since the route emits that body plus this compiler's END.
        """
        for shape in [(17, 19, 16), (50, 37, 80)]:
            with self.subTest(shape=shape):
                code, _ = self.compile(*shape)
                self.assertEqual(bytes(code), bytes(tensor.emit_gemm(*shape).body))

    def test_a_form_absent_from_the_passthrough_would_be_refused(self):
        """The route's rows carry their own bytes and must be in _emit's passthrough allowlist.

        Guards the wiring rather than the bytes: if `tensor.wholekernel` were dropped from that
        tuple the rows would reach the template lookup and raise "no canonical template", which is
        exactly how the route failed the first time it was compiled.
        """
        code, layout = self.compile(17, 19, 16)
        rows = [m for _, _, m in layout if m.form == "tensor.wholekernel"]
        self.assertTrue(rows)
        self.assertTrue(all("bytes" in m.fields for m in rows))

    def test_the_ir_strides_reach_the_lowering(self):
        """THE IR CARRIES STRIDES IN BYTES AND THE LOWERING TAKES ELEMENTS.

        Fails if the conversion is dropped - the route would then read a nonexistent at["lda"],
        lower every strided GEMM as contiguous, and two different programs would compile to the
        same bytes. That is the same defect class as ownimage recomputing leading dimensions.
        """
        contiguous, _ = self.compile(17, 19, 16)
        strided, _ = self.compile(17, 19, 16, strideA=64)
        self.assertNotEqual(bytes(contiguous), bytes(strided),
                            "strideA did not change the emitted program")

    def test_the_system_register_the_body_reads_is_declared(self):
        """The body reads a system register, and a program that reads an undeclared one is what
        authorobj refuses by name. Fails if the route stops setting `sr` on the read rows."""
        _, layout = self.compile(17, 19, 16)
        declared = [m.fields["sr"] for _, _, m in layout
                    if m.form == "tensor.wholekernel" and "sr" in m.fields]
        self.assertTrue(declared, "no system register declared by any route row")


class AnUnexpectedExceptionPropagates(unittest.TestCase):
    """The route absorbs the lowering's documented refusal and NOTHING else.

    Converting a defect into a named refusal is worse than crashing: the caller reads "this shape
    is not supported" when the truth is "this compiler is broken", and every workload failure
    becomes untrustworthy. These inject faults the route must not swallow.
    """

    def compile_one(self):
        f = ir.Function("a", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
        b = ir.Builder(f, f.block("e"))
        b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=17, N=19, K=16)
        b.ret()
        ir.verify(f)
        return g17cc.select(f)

    def test_a_keyerror_from_the_lowering_is_not_a_refusal(self):
        real = tensor.emit_gemm
        def boom(*a, **k):
            raise KeyError("injected defect")
        tensor.emit_gemm = boom
        try:
            with self.assertRaises(KeyError):
                self.compile_one()
        finally:
            tensor.emit_gemm = real

    def test_a_bare_valueerror_without_the_refused_prefix_is_not_a_refusal(self):
        """The discriminator is the documented `refused: ` prefix, not the exception type: a
        ValueError raised by a defect must still propagate."""
        real = tensor.emit_gemm
        def boom(*a, **k):
            raise ValueError("index out of range")
        tensor.emit_gemm = boom
        try:
            with self.assertRaises(ValueError) as caught:
                self.compile_one()
            self.assertNotIsInstance(caught.exception, g17cc.Unsupported)
            self.assertIn("index out of range", str(caught.exception))
        finally:
            tensor.emit_gemm = real

    def test_a_documented_refusal_IS_absorbed_and_named(self):
        """The companion, so the two tests above cannot be satisfied by absorbing nothing."""
        real = tensor.emit_gemm
        def refuse(*a, **k):
            raise ValueError("refused: injected refusal")
        tensor.emit_gemm = refuse
        try:
            with self.assertRaises(g17cc.Unsupported) as caught:
                self.compile_one()
            self.assertIn("injected refusal", str(caught.exception))
        finally:
            tensor.emit_gemm = real


class OneTensorBodyComposesWithScalarWork(unittest.TestCase):
    """The row-splice slice: one tensor body plus ordinary scalar/control/memory operations, in
    ONE emitted stream with ONE final END and a single coherent allocation.

    A second tensor operation is still refused - two complete kernels cannot both be the whole
    program - so these cases cover gemm-activation, gemm-residual and tensor-scalar-pressure and
    deliberately not chain-same, mixed-shapes or shared-buffers.
    """

    def residual(self, adds=1):
        f = ir.Function("a", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
        b = ir.Builder(f, f.block("e"))
        b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=17, N=19, K=16)
        t = b.builtin("threadgroup_position_in_grid", name="t")
        v = t
        for i in range(adds):
            v = b.add(v, ir.Imm(1), name="s%d" % i)
        b.store_at(f.buffers[2], t, v)
        b.ret()
        ir.verify(f)
        return f

    def activation(self):
        import struct
        f = ir.Function("a", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
        b = ir.Builder(f, f.block("e"))
        b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=17, N=19, K=16)
        t = b.builtin("threadgroup_position_in_grid", name="t")
        y = b.load(f.buffers[2], t, type="f32", name="y")
        z = b.const(struct.unpack("<I", struct.pack("<f", 0.0))[0], type="f32", name="zero")
        b.store_at(f.buffers[2], t, b.fmax(y, z, name="act"))
        b.ret()
        ir.verify(f)
        return f

    def compile(self, fn):
        insts = g17cc.Alloc(regs=range(0, 126)).run(g17cc.select(fn))
        code, layout = g17cc.emit(insts)
        return insts, bytes(code), layout

    CELLS = None

    def cells(self):
        return (("gemm-residual", self.residual(1)),
                ("gemm-activation", self.activation()),
                ("tensor-scalar-pressure", self.residual(8)))

    def test_exactly_one_END_and_it_is_last(self):
        """Fails if the lowering's own END survives the splice - an END mid-program would retire
        the kernel before the scalar work ran, which is the failure the old refusal prevented."""
        for name, fn in self.cells():
            with self.subTest(cell=name):
                _, code, _ = self.compile(fn)
                ins = [i for i in model.decode(code, 0) if i.opcode]
                ends = [j for j, i in enumerate(ins) if i.opcode.id == 684]
                self.assertEqual(ends, [len(ins) - 1],
                                 "%s: END positions %s of %d" % (name, ends, len(ins)))

    def test_the_tensor_body_and_the_scalar_work_are_both_present(self):
        """Fails if either half is dropped: a stream with no tensor rows, or one where the scalar
        operations were absorbed and lost, would still have one END and pass the case above."""
        for name, fn in self.cells():
            with self.subTest(cell=name):
                _, _, layout = self.compile(fn)
                forms = collections.Counter(m.form for _, _, m in layout)
                self.assertTrue(forms["tensor.wholekernel"] > 50,
                                "%s: tensor body missing (%s)" % (name, dict(forms)))
                scalar = sum(v for k, v in forms.items()
                             if k not in ("tensor.wholekernel", "end"))
                self.assertTrue(scalar >= 2,
                                "%s: scalar work missing (%s)" % (name, dict(forms)))

    def test_the_allocation_is_coherent_scalar_never_lands_in_the_bodys_registers(self):
        """THE POINT OF THE SLICE. Fails if a surrounding value is coloured into a register the
        tensor body writes - two writers, one register, and no diagnostic at all. This is what
        `_occupies` exists to prevent and what made the mixed cells unsafe before it."""
        for name, fn in self.cells():
            with self.subTest(cell=name):
                insts, _, _ = self.compile(fn)
                occupied, scalar = set(), set()
                for m in insts:
                    occupied |= set((m.fields or {}).get("_occupies", ()))
                    if m.form == "tensor.wholekernel":
                        continue
                    for key in ("_defs", "_uses"):
                        for r in (m.fields or {}).get(key, ()) or ():
                            if isinstance(r, int):
                                scalar.add(r)
                self.assertTrue(occupied, "%s: nothing published" % name)
                self.assertTrue(scalar, "%s: no scalar registers to check" % name)
                self.assertEqual(occupied & scalar, set(),
                                 "%s: scalar values collide with the body at %s"
                                 % (name, sorted(occupied & scalar)))

    def test_a_third_tensor_operation_is_still_refused_by_name(self):
        """The slice's boundary. Two measured bodies compose; a third remains refused, and the
        reason must say why rather than reporting a generic failure."""
        f = ir.Function("a", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
        b = ir.Builder(f, f.block("e"))
        b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=17, N=19, K=16)
        b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=17, N=19, K=16)
        b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=17, N=19, K=16)
        b.ret()
        ir.verify(f)
        with self.assertRaises(g17cc.Unsupported) as caught:
            g17cc.select(f)
        message = str(caught.exception)
        self.assertIn("other tensor operation", message)
        self.assertIn("two complete kernels", message)
        self.assertIn("END mid-program", message,
                      "the reason must name the mechanism, not just decline")


class SixBodyMeasuredBOffsets(unittest.TestCase):
    """The six-body route admits only the measured even B-offset transport domain.

    This is a compile/selection guard.  The common-runtime image class remains the separately
    measured two/three-region contract, so these tests do not imply a new six-body dispatch class.
    """

    def function(self, offsets, *, offsetA=None, offsetC=None, first_m=16):
        f = ir.Function("six_body_offsets", [ir.Buffer("A", 1, elem=ir.F16),
                                              ir.Buffer("B", 2, elem=ir.F16),
                                              ir.Buffer("C", 3, elem=ir.F32)])
        b = ir.Builder(f, f.block("e"))
        specs = ((f.buffers[0], 64, "half", False),
                 (f.buffers[2], 32, "float", True),
                 (f.buffers[2], 32, "float", True),
                 (f.buffers[2], 32, "float", False),
                 (f.buffers[2], 32, "float", True),
                 (f.buffers[2], 32, "float", True))
        for index, (source, k, a_dtype, accumulate) in enumerate(specs):
            kwargs = dict(M=first_m, N=32, K=k, a_dtype=a_dtype, b_dtype="half",
                          accumulate=accumulate, offsetB=offsets[index])
            if offsetA is not None:
                kwargs["offsetA"] = offsetA
            if offsetC is not None:
                kwargs["offsetC"] = offsetC
            b.tensor_matmul(source, f.buffers[1], f.buffers[2], **kwargs)
        b.ret()
        ir.verify(f)
        return f

    def test_measured_even_positions_route_including_displacement_boundary(self):
        for offsets in ((0, 0, 0, 4096, 0, 0),
                        (0, 4096, 8192, 12288, 16384, 20480),
                        (0, 0, 0, 47104, 0, 0)):
            with self.subTest(offsets=offsets):
                selected = g17cc.select(self.function(offsets))
                self.assertGreater(sum(m.form == "tensor.wholekernel" for m in selected), 0)

    def test_a_wider_stream_is_an_adjacent_chain_now(self):
        """The M=32 six-body stream was refused as an unmeasured shape. Its bodies are adjacent with
        nothing between them, so cc._adjacent_tensor_chain admits it (Set A item 2): every boundary
        a register feed, the offsets inside the measured transport domain. It compiles; that it
        EXECUTES is not claimed here - the generic runtime class carries that check."""
        fn = self.function((0, 0, 0, 4096, 0, 0), first_m=32)
        ops = [o for b in fn.blocks for o in b.ops if o.kind == "tensor_matmul"]
        self.assertTrue(g17cc._adjacent_tensor_chain(fn, ops))
        self.assertGreater(sum(m.form == "tensor.wholekernel" for m in g17cc.select(fn)), 0)

    def test_odd_large_and_nonmeasured_offsets_remain_refused(self):
        for name, fn in (
                ("odd", self.function((0, 0, 0, 3, 0, 0))),
                ("large", self.function((0, 0, 0, 47106, 0, 0))),
                ("A", self.function((0, 0, 0, 4096, 0, 0), offsetA=2)),
                ("C", self.function((0, 0, 0, 4096, 0, 0), offsetC=2))):
            with self.subTest(name=name):
                with self.assertRaises(g17cc.Unsupported):
                    g17cc.select(fn)


class ThePureCellsAreByteIdentical(unittest.TestCase):
    """Acceptance requires the pure whole-kernel cells to be unchanged by the splice work.

    Hashes are recomputed here rather than frozen as literals: a frozen digest would also pass if
    the compiler stopped emitting anything, and would need editing on every legitimate change. The
    invariant is that a pure function's program equals the lowering's own body - which is what
    "unchanged" means for this route - so that is what is asserted.
    """

    def test_a_pure_function_emits_exactly_the_lowerings_body(self):
        for shape in [(17, 19, 16), (50, 37, 80), (16, 16, 16)]:
            with self.subTest(shape=shape):
                f = ir.Function("a", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
                b = ir.Builder(f, f.block("e"))
                b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2],
                                M=shape[0], N=shape[1], K=shape[2])
                b.ret()
                ir.verify(f)
                code, _ = g17cc.emit(g17cc.Alloc(regs=range(0, 126)).run(g17cc.select(f)))
                self.assertEqual(bytes(code), bytes(tensor.emit_gemm(*shape).body))


class AScalarF32ConsumerOfTheTensorResult(unittest.TestCase):
    """Regression for the GEMM-plus-residual workload root's matrix found failing on 6a63019d.

    The function was: tensor_matmul, a dynamic f32 load of C, a dynamic f32 load of R, an `add`,
    and a store. It reached cc._width and raised KeyError('f32') - an unexplained compiler crash
    several frames from its cause, not a named refusal.

    The classification is that the program was malformed rather than the route being limited: the
    IR keeps `add` (integer) and `fadd` (float) separate, so `add` on two f32 values is the wrong
    operation. Both halves are asserted here, because a refusal is only the right answer if the
    intended workload is actually expressible.
    """

    def workload(self, op):
        f = ir.Function("a", [ir.Buffer("A", 0), ir.Buffer("B", 1),
                              ir.Buffer("C", 2), ir.Buffer("R", 3)])
        b = ir.Builder(f, f.block("e"))
        b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=17, N=19, K=16)
        t = b.builtin("threadgroup_position_in_grid", name="t")
        c = b.load(f.buffers[2], t, type=ir.F32, name="c")
        r = b.load(f.buffers[3], t, type=ir.F32, name="r")
        y = getattr(b, op)(c, r, name="y")
        b.store_at(f.buffers[2], t, y)
        b.ret()
        ir.verify(f)
        return f

    def test_an_integer_add_on_f32_is_a_named_refusal_not_a_KeyError(self):
        """Fails if the KeyError comes back, or if the refusal does not name the float operation
        the caller should have used - a refusal that does not say what to do instead leaves the
        caller exactly where the crash did."""
        fn = self.workload("add")
        with self.assertRaises(g17cc.Unsupported) as caught:
            g17cc.emit(g17cc.Alloc(regs=range(0, 126)).run(g17cc.select(fn)))
        message = str(caught.exception)
        self.assertIn("f32", message)
        self.assertIn("INTEGER-ONLY", message)
        self.assertIn("fadd", message)

    def test_the_same_workload_with_fadd_composes(self):
        """The half that makes the refusal legitimate: the residual workload IS expressible, so
        the refusal is a wrong-operation diagnostic and not a missing capability."""
        fn = self.workload("fadd")
        insts = g17cc.Alloc(regs=range(0, 126)).run(g17cc.select(fn))
        code, layout = g17cc.emit(insts)
        ins = [i for i in model.decode(bytes(code), 0) if i.opcode]
        ends = [j for j, i in enumerate(ins) if i.opcode.id == 684]
        self.assertEqual(ends, [len(ins) - 1], "END positions %s" % ends)
        self.assertTrue(any(m.form == "tensor.wholekernel" for _, _, m in layout))
        occupied, scalar = set(), set()
        for m in insts:
            occupied |= set((m.fields or {}).get("_occupies", ()))
            if m.form == "tensor.wholekernel":
                continue
            for key in ("_defs", "_uses"):
                for r in (m.fields or {}).get(key, ()) or ():
                    if isinstance(r, int):
                        scalar.add(r)
        self.assertEqual(occupied & scalar, set(), "scalar collides with the tensor body")

    def test_an_UNKNOWN_type_still_raises_KeyError_and_is_not_dressed_as_a_refusal(self):
        """THE DISCRIMINATING COMPANION. Only the known float types are refused by name; any other
        unrecognised type is a selector routing something nobody planned for, which is a defect in
        this compiler. Fails if the fix blanket-converted every width lookup into "your program is
        wrong", which would hide exactly that class of bug - the same mistake the broad
        `except Exception` made in the route.
        """
        with self.assertRaises(KeyError):
            g17cc._width("not_a_type")
        for float_type in ("f16", "f32"):
            with self.subTest(type=float_type):
                with self.assertRaises(g17cc.Unsupported):
                    g17cc._width(float_type)


class TheGeneralRoutePublishesCompleteABIFacts(unittest.TestCase):
    """Root's fresh-main reproducer: the general lowering emitted stores and captured none of them.

    17x19x16, 17x19x19 and 50x37x80 compiled, each stream carried op17257/op17258 tensor stores,
    and the captured facts were has_stores=False, writes_buffer=False, register_count=0,
    pk_values={} - so ProgramABI.contract() refused the binding/write inconsistency and the shapes
    could not be authored at all. Two causes, one test class:

      * the write predicate keyed on form name and phase, and the row-spliced rows carry
        phase="general lowering" rather than "readout";
      * register_count read only _defs/_uses, and a spliced row publishes `_occupies`.
    """

    SHAPES = [(17, 19, 16), (17, 19, 19), (50, 37, 80)]

    def program(self, M, N, K):
        f = ir.Function("a", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
        b = ir.Builder(f, f.block("e"))
        b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=M, N=N, K=K)
        b.ret()
        ir.verify(f)
        return g17cc.compile_function(f)

    def test_the_contract_is_authorable(self):
        """The failure root actually hit. Fails if contract() refuses again for any of them."""
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                self.program(*shape).contract()

    def test_the_stores_are_seen_and_the_register_count_is_not_zero(self):
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                facts = self.program(*shape).abi_inputs()
                self.assertTrue(facts["has_stores"], "has_stores false with tensor stores present")
                self.assertTrue(facts["writes_buffer"])
                self.assertEqual(facts["pk_values"].get(15), 1)
                self.assertEqual(facts["pk_values"].get(16), 1)
                self.assertTrue(facts["register_count"] > 0,
                                "register_count 0 for a program that names registers")

    def test_the_register_count_is_the_highest_index_named_plus_one(self):
        """The law the file states, applied to the rows the allocator did not colour. Fails if the
        count is merely non-zero - a ceiling or a constant would pass the case above."""
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                program = self.program(*shape)
                named = set()
                for row in program.layout:
                    m = row[-1]
                    for key in ("_defs", "_uses", "_occupies"):
                        named.update(m.fields.get(key) or [])
                self.assertEqual(program.abi_inputs()["register_count"], max(named) + 1)

    def test_a_tensor_store_is_classified_BY_OPCODE_not_by_phase(self):
        """The generic half of the fix: the row-spliced stores carry phase="general lowering", so
        a predicate keyed on the phase string reports no writes. Fails if the opcode rule is
        dropped and the phase special-case is relied on again."""
        program = self.program(17, 19, 16)
        stores = [row[-1] for row in program.layout
                  if row[-1].form.startswith("tensor.")
                  and row[-1].fields.get("opcode") in g17cc.TENSOR_STORE_OPCODES]
        self.assertTrue(stores, "no tensor store rows in a program that stores")
        self.assertEqual({m.fields.get("phase") for m in stores}, {"general lowering"},
                         "these rows are exactly the ones the phase rule cannot see")
        self.assertNotIn(17258, g17cc.TENSOR_OPCODES,
                         "op17258 is absent from TENSOR_OPCODES, which is why that set was not reused")

    def test_the_registry_control_facts_are_unchanged(self):
        """32x32x64 goes through the registry and must be untouched: root measured
        register_count=72 on it before this fix."""
        facts = self.program(32, 32, 64).abi_inputs()
        self.assertEqual(facts["register_count"], 72)
        self.assertTrue(facts["has_stores"] and facts["writes_buffer"])


class TheAllocatorPoolSurvivesARefusal(unittest.TestCase):
    """Root's c0ad2c32 finding. The occupancy refusal raised after narrowing self.regs/self.wide
    and before the try, so the finally never ran and a reused Alloc carried a shrunken pool - a
    later allocation would then fail for a reason belonging to an earlier program."""

    def insts(self):
        f = ir.Function("a", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
        b = ir.Builder(f, f.block("e"))
        b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=17, N=19, K=16)
        b.ret()
        ir.verify(f)
        return g17cc.select(f)

    def test_the_pool_is_restored_after_the_refusal(self):
        allocator = g17cc.Alloc(regs=range(0, 8))     # too small for the body on purpose
        before = list(allocator.regs)
        with self.assertRaises(g17cc.Unsupported):
            allocator.run(self.insts())
        self.assertEqual(list(allocator.regs), before, "the refusal left the pool narrowed")

    def test_the_pool_is_restored_after_a_SUCCESS_too(self):
        """The companion, so the case above cannot be satisfied by never narrowing at all."""
        allocator = g17cc.Alloc(regs=range(0, 126))
        before = list(allocator.regs)
        allocator.run(self.insts())
        self.assertEqual(list(allocator.regs), before)




class TheDefaultStridesComeFromTheDeclaredElementWidths(unittest.TestCase):
    """The default row strides were `K * 2` and `N * 2` for A and B whatever the operand types.

    Right for half and bfloat, wrong for float and int8/uint8 - and wrong SILENTLY, because a
    smaller-than-correct stride still compiles, still passes every structural check and still
    publishes a valid ABI. The integration replay caught it on hardware: a 32x32x64 int8 matmul
    with no explicit strides disagreed with the reference on 1024 of 1024 output elements over
    three trials, while the same shape with strideA=64, strideB=32 emitted the lowerer's body byte
    for byte.

    What each case here would catch:
      - the identity matrix: any width whose default stride stops agreeing with the contiguous
        strides written out by hand, which is the defect itself for float, int8 and uint8
      - the two unchanged pins: a "fix" that moved half or bfloat, which would mean the correction
        reached shapes that were already right
      - int8's pin on 470ebeda...: the lowerer's own body, so this is checked against a second
        implementation rather than against whatever this compiler now happens to emit
      - the refusal cases: a correction that widened the strides it accepts as well as the ones it
        derives
    """

    SHAPE = dict(M=32, N=32, K=64)
    # bytes per element, and the contiguous strides that follow for this shape
    WIDTHS = (("half", 2), ("bfloat", 2), ("float", 4), ("int8", 1), ("uint8", 1))

    def program(self, **attrs):
        f = ir.Function("t", [ir.Buffer("a", 1, elem=ir.F16), ir.Buffer("b", 2, elem=ir.F16),
                              ir.Buffer("c", 3, elem=ir.F32)])
        b = ir.Builder(f, f.block("entry"))
        b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2],
                        **dict(dict(self.SHAPE), **attrs))
        b.ret()
        return f

    def code_of(self, **attrs):
        import hashlib
        program = g17cc.compile_function(self.program(**attrs))
        return len(program.code), hashlib.sha256(program.code).hexdigest()

    def test_the_default_equals_the_explicit_contiguous_stride_at_every_width(self):
        """The matrix root's acceptance asks for. `strideC` is N * 4 for all of them because C has
        no dtype - it is always float - which is why only A and B move with the width.
        """
        for dtype, width in self.WIDTHS:
            default = self.code_of(a_dtype=dtype, b_dtype=dtype)
            explicit = self.code_of(a_dtype=dtype, b_dtype=dtype,
                                    strideA=self.SHAPE["K"] * width,
                                    strideB=self.SHAPE["N"] * width,
                                    strideC=self.SHAPE["N"] * 4)
            self.assertEqual(default, explicit,
                             "%s: the default stride does not derive from the element width "
                             "(default %s, explicit %s)" % (dtype, default, explicit))

    def test_half_and_bfloat_did_not_move(self):
        """These two were already correct, so the correction must be invisible to them. Their
        hashes are the ones the candidate published before the stride fix.
        """
        self.assertEqual(self.code_of(a_dtype="half", b_dtype="half")[1][:16],
                         "472570faa7d227ee")
        self.assertEqual(self.code_of(a_dtype="bfloat", b_dtype="bfloat")[1][:16],
                         "265241da8b926e18")

    def test_the_int8_default_is_now_the_lowerers_own_body(self):
        """The externally checked value. 470ebeda... is agxforge/g17/tensorlower's body for this
        shape, measured by the integration independently of this compiler, so it pins the fix
        against a second implementation rather than against this one's current output. Before the
        fix the default compiled 33bb8af0..., 28 bytes different.
        """
        size, digest = self.code_of(a_dtype="int8", b_dtype="int8")
        self.assertEqual(size, 1002)
        self.assertTrue(digest.startswith("470ebeda6a7a0b46"), digest)
        self.assertFalse(digest.startswith("33bb8af0"), "the pre-fix body is back")

    def test_an_unknown_dtype_still_raises_rather_than_defaulting_to_two(self):
        """The disposition this IR already had: KeyError on an unrecognised dtype is an invalid
        fixture, not a compiler defect. It must NOT become a silent width of 2, which is what a
        `.get(dtype, 2)` would have made it - the same shape as the bug being fixed.
        """
        with self.assertRaises(KeyError):
            self.code_of(a_dtype="f32")


    def test_the_binding_element_type_is_the_buffers_not_the_operands(self):
        """The subtlety the stride fix creates, pinned so a consumer cannot get it wrong.

        With the defaults derived from `a_dtype`, an int8 operand strides by ONE byte - but its
        binding reports `half` and two bytes, and that is not a mismatch to reconcile. The IR
        refuses a 1-byte buffer for any op at all ("this backend accesses 2- and 4-byte scalars
        only, so a declaration of this type is carried and never touched"), so an int8 tensor
        operand has nowhere to live except a 2- or 4-byte declared buffer. The binding describes
        the BUFFER's declared scalar type; `a_dtype` describes how the tensor unit reads the bytes.

        This case fails if someone "fixes" the binding to echo `a_dtype` - which would make the
        contract claim a 1-byte buffer the backend cannot access - or if the 1-byte refusal is
        dropped, which would let the two declarations disagree silently for real.
        """
        program = g17cc.compile_function(self.program(a_dtype="int8", b_dtype="int8"))
        bindings = program.contract().bindings
        self.assertEqual([b.index for b in bindings], [1, 2, 3],
                         "the binding indices are the IR's declared slots")
        self.assertEqual([(b.element_type, b.element_bytes) for b in bindings[:2]],
                         [("half", 2), ("half", 2)],
                         "A and B report the BUFFER's declared element, not a_dtype")
        self.assertEqual((bindings[2].element_type, bindings[2].element_bytes), ("float", 4))
        # the stride basis really is the other number, or the note above is decoration
        self.assertEqual(TENSOR_ELEMENT_BYTES["int8"], 1)
        # and a 1-byte buffer is refused backend-wide, which is WHY they differ
        with self.assertRaises(ir.IRError) as caught:
            f = ir.Function("t", [ir.Buffer("a", 1, elem="uchar"), ir.Buffer("b", 2, elem="uchar"),
                                  ir.Buffer("c", 3, elem=ir.F32)])
            b = ir.Builder(f, f.block("entry"))
            b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], **dict(self.SHAPE))
        self.assertIn("2- and 4-byte scalars only", str(caught.exception))

    def test_the_stride_refusals_are_untouched(self):
        """A correction that also widened what is ACCEPTED would pass every case above. Explicit
        strides are still bytes and still checked for whole elements. (The two six-bit stride limits are
        lifted within the measured range; EveryPinnedRefusalIsPreserved checks both sides of the bound.)
        """
        for attrs, word in ((dict(strideA=3), "whole"), (dict(strideA=65), "whole")):
            with self.assertRaises(g17cc.Unsupported) as caught:
                self.code_of(**attrs)
            self.assertIn(word, str(caught.exception),
                          "%s no longer refused by name: %s" % (attrs, str(caught.exception)[:120]))

class ThePackagingPathIsRunnableForATensorProgram(unittest.TestCase):
    """`to_image` was twice called the blocker on closing the stride fix end to end - "the
    reference container it wants, which neither of us has manufactured out of guesswork". That
    was wrong on the facts and this case is what settles it: the build cache holds 25
    Apple-compiled tensor containers, and the int8 default-stride program packages into one.

    The first check of this got a FALSE match and nearly became "the packaging is broken": the
    code is spliced at the window, which is relative to the container's __text, while a search of
    the materialised file finds it at the text section's own file offset. Reading the file at
    `probe.entry` indexes a text-relative offset into a whole metallib, so it was the check that
    was wrong. The case therefore asserts the relationship rather than one offset.

    What this does NOT establish, kept here so the record does not drift the other way: nothing is
    dispatched. The bytes are covered transitively - the linker lane has run this exact body 3/3
    against reference - but a runtime result for the packaged image is not claimed by this case.

    Skipped rather than failed when the cache is absent: this reads a gitignored build cache, so a
    fresh checkout has no containers and that is an environment fact, not a regression.
    """

    CONTAINER = os.path.expanduser("~/.cache/agxforge/agx/abi-v1-tensor-common-9e3449b707")

    def test_the_int8_program_packages_into_an_apple_tensor_container(self):
        if not os.path.isdir(self.CONTAINER):
            self.skipTest("build cache container absent: %s" % self.CONTAINER)
        from agxforge.g17 import image as g17image
        f = ir.Function("t", [ir.Buffer("a", 1, elem=ir.F16), ir.Buffer("b", 2, elem=ir.F16),
                              ir.Buffer("c", 3, elem=ir.F32)])
        b = ir.Builder(f, f.block("entry"))
        b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2],
                        M=32, N=32, K=64, a_dtype="int8", b_dtype="int8")
        b.ret()
        program = g17cc.compile_function(f)
        probe = g17image.G17Image(self.CONTAINER, "k")
        image = program.to_image(self.CONTAINER, probe.entry)
        raw = image.materialise()
        self.assertEqual(image.window, (probe.entry, len(program.code)),
                         "the window is not the entry plus this program's length")
        found = raw.find(program.code)
        self.assertNotEqual(found, -1, "the compiled bytes are not in the materialised image")
        # the window is TEXT-relative; the text section sits at found - entry in the file
        self.assertEqual(raw[found:found + len(program.code)], program.code)
        self.assertEqual(raw[found - probe.entry:found - probe.entry + 4], bytes(probe.text[:4]),
                         "the implied text base does not hold the container's own text")
        self.assertGreater(len(raw), len(program.code),
                           "the image is not larger than the program it carries")



class AComposedProgramUsesTheMeasuredTwoRegisterMetadata(unittest.TestCase):
    """The composed route now uses the measured SR130/SR156 vector.

    The cached population supplies 1,039 exact (130,156) objects, each carrying slot-29 [0,52].
    The serializer derives the second entry through the existing measured vector-growth rule;
    the singleton SR130 class remains byte-identical. Unsupported register sets still refuse by
    name, so adding this pair does not turn the class into a general metadata guess.
    """

    def test_a_single_sr130_program_still_authors(self):
        """Ground truth first: the pure case must PASS, or the refusal below proves nothing."""
        from agxforge.g17 import tensormetadata
        tensormetadata.layout(register_count=72, system_registers=(130,),
                              instruction_count=128, has_back_edge=True)

    def test_the_two_entry_layout_really_is_measured(self):
        """The vector is populated from the measured register map, not a tensor-only fill."""
        from agxforge.g17 import mdgen
        self.assertEqual(mdgen.slot29_entries((160, 161)), [80, 81])
        self.assertIn(2, mdgen.MEASURED_SYSTEM_REGISTER_COUNTS)
        self.assertEqual(mdgen.slot29_entries((130,)), [52])
        self.assertEqual(mdgen.slot29_entries((130, 156)), [0, 52])

    def test_the_two_register_layout_is_derived_and_other_sets_still_refuse(self):
        from agxforge.g17 import tensormetadata
        import hashlib
        one = tensormetadata.emit([(1, 0, False), (2, 2, False), (3, 4, True)],
                                  register_count=72, system_registers=(130,),
                                  instruction_count=128, has_back_edge=True)
        two = tensormetadata.emit([(1, 0, False), (2, 2, False), (3, 4, True)],
                                  register_count=72, system_registers=(130, 156),
                                  instruction_count=128, has_back_edge=True)
        self.assertEqual(hashlib.sha256(one).hexdigest(),
                         "7242f32c6af1c892d38170366c8178bb5933bf9c9a4599e16379a87e06e81f72")
        self.assertEqual(len(two), len(one) + 4)
        from agxforge.g17 import gpumd
        self.assertEqual(gpumd.fields(two).get(29), 64)
        import struct
        pk = gpumd.kernel_table(two)
        slots, _ = gpumd.table_at(two, pk)
        field = pk + slots[29]
        vector = field + struct.unpack_from("<I", two, field)[0]
        count = struct.unpack_from("<I", two, vector)[0]
        self.assertEqual((count, [struct.unpack_from("<I", two, vector + 4 + 4 * j)[0]
                                  for j in range(count)]), (2, [0, 52]))
        for registers in ((130, 160), (156,), (), (130, 130)):
            with self.assertRaises(ValueError) as caught:
                tensormetadata.layout(register_count=72, system_registers=registers,
                                      instruction_count=128, has_back_edge=True)
            self.assertIn("requires SR130", str(caught.exception))

    def test_composed_three_buffer_program_is_authorable_with_two_entries(self):
        """A tensor body plus scalar work reaches public authoring with both readings.

        A same-buffer scalar epilogue isolates this metadata lift from an unrelated four-binding
        signature. The compiler route still owns code and register planning; this checks that its
        measured ABI can now be serialized.
        """
        from agxforge.g17 import scanlink, verify
        f = ir.Function("a", [ir.Buffer("A", 1), ir.Buffer("B", 2), ir.Buffer("C", 3)])
        b = ir.Builder(f, f.block("entry"))
        b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=17, N=19, K=16)
        t = b.builtin("threadgroup_position_in_grid", name="t")
        c = b.load(f.buffers[2], t, type=ir.F32, name="c")
        b.store_at(f.buffers[2], t, b.fadd(c, c, name="scaled"))
        b.ret()
        ir.verify(f)
        program = g17cc.compile_function(f)
        image = scanlink.author(program)
        metadata = verify._objsect(image.object, "__GPU_METADATA")
        self.assertTrue(verify.verify_metadata(metadata).ok)
        self.assertEqual(len(metadata), 488)
        self.assertIn("SR130 + SR156", image.field_ledger["metadata class"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
