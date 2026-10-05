import math
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from agxforge.g17 import cc, ir, tensorreduce


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


class TensorReductionReferenceTests(unittest.TestCase):
    def test_row_sum_uses_local_chain_then_measured_row_butterfly(self):
        lanes = [[f32(1.0)] for _ in range(32)]
        lanes[0] = [f32(1.0), f32(2 ** -24)]
        got = tensorreduce.row_sum(lanes)
        # Lane 0's local chain rounds first; every lane then receives the two
        # XOR stages.  All lanes must agree on the same FP32 result.
        self.assertEqual(got, (f32(4.0),) * 32)

    def test_row_max_ignores_nan_and_uses_negative_finite_identity(self):
        lanes = [[float("nan")] for _ in range(32)]
        got = tensorreduce.row_max(lanes)
        self.assertEqual(got, (tensorreduce.FP32_NEG_MAX,) * 32)
        lanes[3] = [float("nan"), 4.0]
        got = tensorreduce.row_max(lanes)
        self.assertEqual(got[3], 4.0)
        self.assertEqual(got[0], tensorreduce.FP32_NEG_MAX)

    def test_columns_use_the_distinct_measured_mask_sequence(self):
        lanes = [[float(i)] for i in range(32)]
        got = tensorreduce.reduce_columns(lanes)
        self.assertEqual(got[0], 88.0)
        self.assertEqual(got[1], 96.0)

    def test_layout_is_16x16_pos_b_tile_blocks(self):
        self.assertEqual(tensorreduce.tile_slot(0, 0, 32), 0)
        self.assertEqual(tensorreduce.tile_slot(0, 16, 32), 8)
        self.assertEqual(tensorreduce.tile_slot(16, 0, 32), 16)
        self.assertEqual(tensorreduce.tile_slot(16, 16, 32), 24)
        self.assertEqual(len(tensorreduce.tile_slots(32, 32)), 4)

    def test_pos_b_and_row_lane_order_are_explicit(self):
        self.assertEqual(tensorreduce.pos_b(0, 0), (0, 0))
        self.assertEqual(tensorreduce.pos_b(15, 15), (31, 7))
        matrix = [list(range(32 * r, 32 * (r + 1))) for r in range(32)]
        lanes = tensorreduce.row_lane_values(matrix, 3)
        nonempty = [v for v in lanes if v]
        self.assertEqual(len(nonempty), 4)
        self.assertEqual(sorted(map(len, nonempty)), [8, 8, 8, 8])
        self.assertEqual(sorted(sum(nonempty, [])), list(range(96, 128)))
        self.assertEqual(nonempty[0][0] > nonempty[0][-1], True)

    def test_unmeasured_domains_refuse_by_name(self):
        cases = [
            dict(accumulator="f16"),
            dict(accumulator="bf16"),
            dict(accumulator="i32"),
            dict(K=32),
            dict(simdgroups=2),
            dict(M=17, N=16),
        ]
        for kw in cases:
            args = dict(M=16, N=16)
            args.update(kw)
            with self.subTest(kw=kw):
                with self.assertRaisesRegex(tensorreduce.UnsupportedReduction, "tensor reduction"):
                    tensorreduce.require_domain(**args)

    def test_ffn_reduction_primitives_accept_only_the_measured_tile(self):
        self.assertEqual(tensorreduce.require_domain(16, 16),
                         dict(M=16, N=16, K=64, accumulator="f32", simdgroups=1))
        for operation in (tensorreduce.emit_row_gelu, tensorreduce.emit_row_layernorm):
            with self.subTest(operation=operation.__name__):
                fn = ir.Function("ffn_domain", [ir.Buffer("scores", 3, elem=ir.F32)])
                block = fn.block("entry")
                builder = ir.Builder(fn, block)
                operation(builder, fn.buffers[0], row=0, M=16, N=16, K=64)
                builder.ret()
                ir.verify(fn)

        for operation in (tensorreduce.emit_row_gelu, tensorreduce.emit_row_layernorm):
            with self.subTest(operation=operation.__name__):
                fn = ir.Function("ffn_bad_domain", [ir.Buffer("scores", 3, elem=ir.F32)])
                block = fn.block("entry")
                builder = ir.Builder(fn, block)
                with self.assertRaisesRegex(tensorreduce.UnsupportedReduction, "16x16"):
                    operation(builder, fn.buffers[0], row=0, M=32, N=16, K=64)


class TensorReductionCompilerTests(unittest.TestCase):
    def _function(self):
        buf = ir.Buffer("values", 1, ir.I32)
        fn = ir.Function("row_reduce_probe", [buf])
        b = ir.Builder(fn, fn.block("entry"))
        lane = b.builtin("thread_index_in_simdgroup")
        value = b.load(buf, lane, type=ir.F32)
        value = b.simd_shuffle_xor(value, 1)
        b.store_at(buf, lane, value)
        b.ret()
        return fn

    def test_shuffle_xor_is_selected_as_authorable_op14169(self):
        selected = cc.select(self._function())
        rows = [m for m in selected if m.form == "auth" and m.fields.get("opcode") == 14169]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].fields["imms"], {1: 32, 3: 32, 4: 1})

    def test_shuffle_xor_compiles_through_the_normal_compiler(self):
        program = cc.compile_function(self._function())
        # THE FORM, NOT ITS REGISTER BYTES: a loaded operand now reaches the shuffle through the
        # waiting copy (ledger/g17-cross-lane-operand-from-a-load-waits.toml), which moves the
        # allocation and so the register fields; op14169/10 itself is what this pins
        import g17ref
        self.assertIn((14169, 10), {(o, n) for _a, n, o in g17ref.walk(bytes(program.code), 0)})

    def test_shuffle_xor_rejects_non_fp32_and_non_butterfly_masks(self):
        fn = ir.Function("bad", [])
        b = ir.Builder(fn, fn.block("entry"))
        value = b.const(0, type=ir.I32)
        with self.assertRaisesRegex(ir.IRError, "FP32"):
            b.simd_shuffle_xor(value, 1, type=ir.I32)
        with self.assertRaisesRegex(ir.IRError, "lane_mask"):
            b.simd_shuffle_xor(value, 3, type=ir.F32)

    def test_row_sum_emits_two_explicit_shuffle_and_fadd_stages(self):
        buf = ir.Buffer("values", 1, ir.I32)
        fn = ir.Function("row_sum_probe", [buf])
        b = ir.Builder(fn, fn.block("entry"))
        lane = b.builtin("thread_index_in_simdgroup")
        value = b.load(buf, lane, type=ir.F32)
        out = tensorreduce.emit_row_sum(b, value)
        b.store_at(buf, lane, out)
        b.ret()
        selected = cc.select(fn)
        shuffles = [m for m in selected if m.form == "auth" and m.fields.get("opcode") == 14169]
        adds = [m for m in selected if m.form == "auth" and m.fields.get("opcode") in (998, 10279, 10282)]
        self.assertEqual([m.fields["imms"][4] for m in shuffles], [1, 8])
        self.assertGreaterEqual(len(adds), 2)

    def test_row_softmax_is_an_ordinary_compiler_program(self):
        a = ir.Buffer("A", 1, ir.F16)
        bbuf = ir.Buffer("B", 2, ir.F16)
        scores = ir.Buffer("scores", 3, ir.F32)
        fn = ir.Function("tensor_row_softmax_probe", [a, bbuf, scores])
        builder = ir.Builder(fn, fn.block("entry"))
        # The reduction itself uses SR_SIMD_ELEM.  This harmless read keeps SR_THREADGROUP_X in
        # the captured contract as well, which is the measured (130,156) common tensor class.
        group = builder.builtin("threadgroup_position_in_grid", name="reduction_group")
        builder.add(group, builder.const(0), name="reduction_group_keep")
        builder.tensor_row_softmax(scores, row=0, M=32, N=16, K=64)
        builder.ret()
        ir.verify(fn)
        program = cc.compile_function(fn)
        plain = program.abi_plain(program.abi())
        self.assertEqual(plain["system_registers"], [130, 156])
        self.assertEqual([(x["index"], x["offset"], x["written"]) for x in plain["bindings"]],
                         [(1, 0, False), (2, 2, False), (3, 4, True)])
        self.assertIn([14169, 10], plain["forms"])
        self.assertIn([684, 4], plain["forms"])

    def test_row_softmax_refuses_unmeasured_shape_and_dtype(self):
        buf = ir.Buffer("scores", 3, ir.F32)
        fn = ir.Function("bad_row_softmax", [buf])
        builder = ir.Builder(fn, fn.block("entry"))
        with self.assertRaisesRegex(tensorreduce.UnsupportedReduction, "M=32, N=16"):
            tensorreduce.emit_row_softmax(builder, buf, M=16, N=16, K=64)

    def test_ffn_scalar_regions_are_explicit_and_compiler_owned(self):
        buf = ir.Buffer("scores", 3, elem=ir.F32)
        fn = ir.Function("ffn_scalar_probe", [buf])
        block = fn.block("entry")
        builder = ir.Builder(fn, block)
        builder.tensor_row_gelu(buf, row=0, M=16, N=16, K=64)
        builder.tensor_row_layernorm(buf, row=0, M=16, N=16, K=64)
        builder.ret()
        selected = cc.select(fn)
        opcodes = {m.fields.get("opcode") for m in selected if m.form == "auth"}
        self.assertIn(14169, opcodes)       # explicit row butterfly
        self.assertIn(11372, opcodes)       # exp2
        self.assertIn(11375, opcodes)       # reciprocal
        self.assertIn(3290, opcodes)        # rsqrt
        self.assertGreaterEqual(sum(m.fields.get("opcode") == 14169
                                    for m in selected if m.form == "auth"), 4)
        program = cc.compile_function(fn)
        self.assertEqual(program.abi_plain(program.abi())["system_registers"], [130])

    def test_all_rows_and_two_tile_reductions_compile_with_shared_lane(self):
        a = ir.Buffer("A", 1, ir.F16)
        bbuf = ir.Buffer("B", 2, ir.F16)
        scores = ir.Buffer("scores", 3, ir.F32)
        fn = ir.Function("ffn_spatial_probe", [a, bbuf, scores])
        builder = ir.Builder(fn, fn.block("entry"))
        builder.tensor_matmul(a, bbuf, scores, M=16, N=16, K=64, a_dtype="half", b_dtype="half")
        builder.tensor_tile_gelu(scores, M=16, N=16, K=64)
        builder.tensor_matmul(scores, bbuf, scores, M=16, N=16, K=16,
                              a_dtype="float", b_dtype="half", accumulate=True)
        builder.tensor_tile_layernorm(scores, M=16, N=16, K=64)
        builder.ret()
        program = cc.compile_function(fn)
        self.assertEqual(program.abi_plain(program.abi())["system_registers"], [130])

        wide = ir.Function("ffn_wide_probe", [a, bbuf, scores])
        wide_builder = ir.Builder(wide, wide.block("entry"))
        wide_builder.tensor_matmul(a, bbuf, scores, M=16, N=32, K=64,
                                   a_dtype="half", b_dtype="half")
        wide_builder.tensor_wide_tile_gelu(scores, M=16, N=32, K=64)
        wide_builder.tensor_matmul(scores, bbuf, scores, M=16, N=32, K=32,
                                   a_dtype="float", b_dtype="half", accumulate=True)
        wide_builder.tensor_wide_tile_layernorm(scores, M=16, N=32, K=64)
        wide_builder.ret()
        wide_program = cc.compile_function(wide)
        self.assertEqual(wide_program.abi_plain(wide_program.abi())["system_registers"], [130])

    def test_spatial_ffn_programs_retain_every_graph_stage(self):
        """The runtime builders must not replace a stage with a host-side or empty shortcut."""
        import hashlib
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import g17tensorcommonruntime as common

        expected_code = {
            common.build_ffn_allrows_program: "28a29784b97377e399323aca245fb70c2e928a00b8754371199666455c8ba616",
            common.build_ffn_wide_program: "8baf075f8d45190106a54b0780bd5d3bc8f241ff24ddec58ec488f6823abb746",
        }
        for build in (common.build_ffn_allrows_program, common.build_ffn_wide_program):
            with self.subTest(build=build.__name__):
                program = build()
                self.assertEqual(hashlib.sha256(program.code).hexdigest(), expected_code[build])
                forms = {opcode for opcode, _length in program.abi_plain(program.abi())["forms"]}
                # First GEMM, contraction GEMM, compiler-owned GELU, residual add, explicit
                # row LayerNorm reduction and the final stores must all survive lowering.
                self.assertGreaterEqual(len(forms & {5100, 5101, 5106, 5107}), 3)
                self.assertIn(9700, forms)       # vector max/nonlinear GELU stage
                self.assertIn(10279, forms)      # compiler-owned residual/fadd path
                self.assertIn(14169, forms)      # measured shuffle-XOR reduction
                self.assertIn(3290, forms)       # LayerNorm rsqrt
                self.assertIn(11375, forms)      # LayerNorm reciprocal
                self.assertIn(17257, forms)      # final tensor store
                self.assertEqual(sum(opcode == 684 for opcode, _ in
                                     program.abi_plain(program.abi())["forms"]), 1)

    def test_transformer_layer_program_retains_attention_ffn_residual_and_layernorm(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import g17tensorcommonruntime as common

        program = common.build_transformer_layer_program()
        plain = program.abi_plain(program.abi())
        forms = {opcode for opcode, _length in plain["forms"]}
        self.assertEqual(plain["system_registers"], [130, 156])
        self.assertEqual(plain["register_count"], 107)
        self.assertGreaterEqual(sum(opcode in (5100, 5101, 5106, 5107)
                                    for opcode in forms), 4)
        for opcode in (1004, 9700, 10279, 11375, 14169, 3290, 17257):
            self.assertIn(opcode, forms)
        self.assertIn(684, forms)
        self.assertIn(12646, forms)  # half residual loads

    def test_two_transformer_layers_retain_both_complete_graphs(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import g17tensorcommonruntime as common

        program = common.build_two_transformer_layers_program()
        plain = program.abi_plain(program.abi())
        forms = {opcode for opcode, _length in plain["forms"]}
        instructions = program.contract().instructions
        tensor_count = sum(i.opcode in (5100, 5101, 5106, 5107) for i in instructions)
        self.assertEqual(plain["system_registers"], [130, 156])
        self.assertEqual(plain["register_count"], 107)
        self.assertGreaterEqual(tensor_count, 6)
        for opcode in (1004, 9700, 10279, 11375, 14169, 3290, 17257, 12646):
            self.assertIn(opcode, forms)
        self.assertEqual(sum(i.opcode == 684 for i in instructions), 1)
        self.assertTrue(all(i.opcode not in (684,) for i in instructions[:-1]))

    def test_spatial_reductions_refuse_unmeasured_tile_domains(self):
        scores = ir.Buffer("scores", 3, ir.F32)
        fn = ir.Function("bad_spatial", [scores])
        builder = ir.Builder(fn, fn.block("entry"))
        for operation in (builder.tensor_tile_gelu, builder.tensor_tile_layernorm):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(tensorreduce.UnsupportedReduction, "16x16"):
                    operation(scores, M=32, N=16, K=64)
        with self.assertRaisesRegex(tensorreduce.UnsupportedReduction, "16x32"):
            tensorreduce.emit_tile_gelu_wide(builder, scores, M=16, N=16, K=64)


if __name__ == "__main__":
    unittest.main()
