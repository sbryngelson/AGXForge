#!/usr/bin/env python3
"""Admission and exact-byte checks for the measured requantization lowering.

The separate measured scalar-stage runtime kind is covered by
test_g17requantcommonruntime.py; these checks keep the compiler primitive, exact measured body
and metadata class, and the old embedded-scale refusal explicit.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

import g17cc as cc
import g17ir as ir
from agxforge.g17 import runtime


def make_program(*, signed=True, scale=1.0 / 512.0, elements=256, groups=8):
    src = ir.Buffer("acc", 3, elem=ir.I32)
    dst = ir.Buffer("quantized_word", 4, elem=ir.I32)
    fn = ir.Function("requant_probe", [src, dst])
    b = ir.Builder(fn, fn.block("entry"))
    b.requantize_int32_to_int8(src, dst, scale=scale, signed=signed,
                               elements=elements, groups=groups)
    b.ret()
    ir.verify(fn)
    return fn


def make_scalar_stage(*, signed=True, scale=1.0 / 512.0, elements=256, groups=8):
    src = ir.Buffer("acc", 0, elem=ir.I32)
    scale_buffer = ir.Buffer("scale_bits", 1, elem=ir.I32)
    dst = ir.Buffer("quantized_word", 2, elem=ir.I32)
    fn = ir.Function("requant_scalar_stage", [src, scale_buffer, dst])
    b = ir.Builder(fn, fn.block("entry"))
    b.requantize_int32_to_int8_stage(src, scale_buffer, dst, scale=scale, signed=signed,
                                     elements=elements, groups=groups)
    b.ret()
    ir.verify(fn)
    return fn


class RequantAdmission(unittest.TestCase):
    def test_int8_tensor_accumulator_is_compile_admitted_but_common_runtime_refuses_uint_output(self):
        """The first quantized stage has a compiler image shape, but not a runtime class yet.

        The tensor lowerer emits an int32 accumulator as the ABI's physical ``uint`` binding.
        That is distinct from the ordinary float-output tensor class used by the common worker.
        Keep the refusal explicit until a measured int8 tensor metadata/runtime contract is
        integrated; accepting this binding by widening the existing class would turn a missing
        measurement into a silent admission.
        """
        a = ir.Buffer("a", 1, elem=ir.F16)
        b = ir.Buffer("b", 2, elem=ir.F16)
        c = ir.Buffer("c", 3, elem=ir.I32)
        fn = ir.Function("int8_tensor_accumulator", [a, b, c])
        builder = ir.Builder(fn, fn.block("entry"))
        builder.tensor_matmul(a, b, c, M=32, N=32, K=64,
                              a_dtype="int8", b_dtype="int8")
        builder.ret()
        program = cc.compile_function(fn)
        self.assertEqual([(x.index, x.element_type, x.written)
                          for x in program.contract().bindings],
                         [(1, "half", False), (2, "half", False), (3, "uint", True)])

        # Convert the compiler's detached ABI view to the runtime model's typed inputs.  The
        # conversion is intentionally local to this test: it exercises the boundary without
        # authoring an image or touching the common worker.
        plain = program.abi_plain(program.abi())
        plain["bindings"] = tuple(runtime.Binding(**x) for x in plain["bindings"])
        plain["constant_pool"] = tuple(plain["constant_pool"])
        plain["forms"] = tuple(tuple(x) for x in plain["forms"])
        plain["pk_extra"] = tuple(plain["pk_extra"])
        plain["system_registers"] = tuple(plain["system_registers"])
        plain["prologue"] = bytes.fromhex(plain["prologue"])
        plain["launch"] = runtime.Launch(**plain["launch"])
        with self.assertRaisesRegex(ValueError, "unsupported tensor runtime binding contract"):
            runtime.CompilerABI(**plain)

    def test_int8_tensor_accumulator_can_author_but_has_no_common_runtime_admission(self):
        """Image serialization is measured separately from worker admission.

        This proves the compiler -> scanlink path does not need an Apple-generated body for the
        int8 tensor stage.  The runtime refusal above remains required: authoring an image is not
        evidence that the common worker has a matching payload and output contract.
        """
        from agxforge.g17 import obj, scanlink
        a = ir.Buffer("a", 1, elem=ir.F16)
        b = ir.Buffer("b", 2, elem=ir.F16)
        c = ir.Buffer("c", 3, elem=ir.I32)
        fn = ir.Function("int8_tensor_accumulator", [a, b, c])
        builder = ir.Builder(fn, fn.block("entry"))
        builder.tensor_matmul(a, b, c, M=32, N=32, K=64,
                              a_dtype="int8", b_dtype="int8")
        builder.ret()
        program = cc.compile_function(fn)
        image = scanlink.author(program)
        self.assertEqual(scanlink.verify_contract(
            image.archive, image.library,
            [(1, 0, False), (2, 2, False), (3, 4, True)]), image.object)
        sections, _ = obj.sections_of(image.object)
        text_offset, _text_size = sections["__TEXT,__text"]
        self.assertEqual(image.object[text_offset + 64:text_offset + 64 + len(program.code)],
                         program.code)

    def test_measured_signed_lowering_is_compile_only_and_has_narrowing_store(self):
        p = cc.compile_function(make_program())
        opcodes = [m.fields.get("opcode") for _off, _size, m in p.layout
                   if m.form in ("unary", "float.unary", "auth")]
        self.assertIn(11179, opcodes)
        self.assertIn(3290, opcodes)
        self.assertIn(3770, opcodes)
        self.assertIn(9320, opcodes)
        self.assertEqual(sum(m.fields.get("opcode") == 17229 for _o, _n, m in p.layout), 1)
        self.assertEqual([(b.index, b.element_type, b.written) for b in p.contract().bindings],
                         [(3, "uint", False), (4, "uint", True)])

    def test_unsigned_measured_scale_is_admitted(self):
        p = cc.compile_function(make_program(signed=False, scale=1.0 / 16.0))
        self.assertEqual(sum(m.fields.get("opcode") == 17229 for _o, _n, m in p.layout), 1)

    def test_measured_stage_main_bytes_match_the_retained_scalar_class(self):
        from agxforge.g17 import requantenc
        steps = ("read_sr", "load", "i32_to_f32", "fmul", "rint", "narrow",
                 "clamp_low", "clamp_high", "store")
        for signed, scale in ((True, 1.0 / 512.0), (False, 1.0 / 16.0)):
            with self.subTest(signed=signed):
                p = cc.compile_function(make_scalar_stage(signed=signed, scale=scale))
                expected = b"".join(requantenc.stage_bytes(signed, step) for step in steps)
                expected += bytes.fromhex("0e000000")
                self.assertEqual(p.code, expected)
                self.assertEqual(len(p.code), 78)

    def test_stage_refuses_unmeasured_scale_signedness_and_extent(self):
        cases = (
            ({"signed": False, "scale": 1.0 / 512.0}, "unsigned"),
            ({"signed": True, "scale": 1.0 / 64.0}, "outside the measured set"),
            ({"signed": True, "elements": 128}, "256 elements"),
            ({"signed": True, "groups": 4}, "256 elements"),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ir.IRError, message):
                    cc.compile_function(make_scalar_stage(**kwargs))

    def test_ordinary_uchar_store_remains_refused(self):
        src = ir.Buffer("src", 1, elem=ir.I32)
        dst = ir.Buffer("dst", 2, elem="uchar")
        fn = ir.Function("ordinary_byte_store", [src, dst])
        b = ir.Builder(fn, fn.block("entry"))
        with self.assertRaisesRegex(ir.IRError, "2- and 4-byte"):
            b.store_at(dst, b.builtin("thread_position_in_grid"), b.const(1), width="byte")

    def test_unmeasured_shape_and_scale_refuse(self):
        for kwargs, message in (
            ({"elements": 128, "groups": 4}, "256 elements"),
            ({"scale": 1.0 / 32.0}, "outside the measured set"),
            ({"signed": False, "scale": 1.0 / 512.0}, "unsigned"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ir.IRError, message):
                    make_program(**kwargs)

    def test_direct_abi_uchar_write_remains_strict(self):
        from agxforge.g17.abi import Binding
        with self.assertRaisesRegex(ValueError, "declaration-only"):
            Binding(index=4, offset=2, written=True, element_type="uchar", element_bytes=1)

    def test_requantization_does_not_create_a_writable_uchar_exception(self):
        from agxforge.g17.abi import Binding, allow_requantized_binding
        with allow_requantized_binding({4}):
            for index in (4, 5):
                with self.subTest(index=index):
                    with self.assertRaisesRegex(ValueError, "declaration-only"):
                        Binding(index=index, offset=2, written=True,
                                element_type="uchar", element_bytes=1)

    def test_public_authoring_names_the_unmeasured_requant_metadata_boundary(self):
        from agxforge.g17 import scanlink
        from agxforge.g17.authorobj import Missing
        with self.assertRaisesRegex(Missing, r"requantization int32_to_int8.*no measured metadata class"):
            scanlink.author(cc.compile_function(make_program()))

    def test_three_buffer_scalar_stage_uses_the_measured_476_class(self):
        from agxforge.g17 import scanlink, verify
        program = cc.compile_function(make_scalar_stage())
        self.assertEqual(program.abi_plain(program.abi())["system_registers"], [160])
        self.assertEqual([(b.index, b.offset, b.element_type, b.written)
                          for b in program.contract().bindings],
                         [(0, 0, "uint", False), (1, 2, "uint", False),
                          (2, 4, "uint", True)])
        self.assertEqual(program.contract().requantization.metadata_class, "scalar_476")
        image = scanlink.author(program)
        metadata = verify._objsect(image.object, "__GPU_METADATA")
        self.assertEqual(len(metadata), 476)
        self.assertEqual(scanlink.binding_records(metadata),
                         [(0, 0, False), (1, 2, False), (2, 4, True)])

    def test_three_buffer_stage_keeps_unmeasured_system_registers_refused(self):
        from agxforge.g17 import scanlink
        from agxforge.g17.authorobj import Missing
        program = cc.compile_function(make_scalar_stage())
        emission = program.abi_plain(program.abi())
        emission["system_registers"] = [130]
        with self.assertRaisesRegex(Missing, "measured only for system registers"):
            scanlink.author(program.code, program.contract(), emission=emission)

    def test_common_runtime_accepts_the_measured_constant_scale_stage(self):
        program = cc.compile_function(make_scalar_stage())
        plain = program.abi_plain(program.abi())
        plain["bindings"] = tuple(plain["bindings"])
        plain["forms"] = tuple(tuple(x) for x in plain["forms"])
        plain["pk_extra"] = tuple(plain["pk_extra"])
        plain["system_registers"] = tuple(plain["system_registers"])
        plain["prologue"] = bytes.fromhex(plain["prologue"])
        plain["launch"] = runtime.Launch(**plain["launch"])
        abi = runtime.CompilerABI(**plain)
        self.assertEqual(abi.prologue.hex(),
                         "248021104701a0821c8a08270f000300804400a04100800000002304070256a0a41a0e000000"
                         "0600060006000600060006000600060006000600060006000600")
        self.assertEqual(abi.constant_program_sha256,
                         "830c032dd18a1b0939d59da15a1c00584d0ac51d083452900b5f077b61a1260d")

    def test_common_runtime_does_not_translate_a_marker_mismatch(self):
        program = cc.compile_function(make_scalar_stage())
        plain = program.abi_plain(program.abi())
        plain["bindings"] = tuple(plain["bindings"])
        plain["forms"] = tuple(tuple(x) for x in plain["forms"])
        plain["pk_extra"] = tuple(plain["pk_extra"])
        plain["system_registers"] = tuple(plain["system_registers"])
        plain["prologue"] = bytes.fromhex(plain["prologue"])
        plain["launch"] = runtime.Launch(**plain["launch"])
        # This test reaches the launch-fact comparison only after explicitly constructing the
        # measured preload marker.  The ordinary stage above is refused before that boundary.
        plain["requantization"] = dict(plain["requantization"],
                                       scale_addressing="constant_program_preload")
        abi = runtime.CompilerABI(**plain)
        with self.assertRaisesRegex(ValueError, "marker and launch facts disagree"):
            runtime.RuntimeContract(
                format="g17-common-pipeline-v4", kind=runtime.REQUANT_KIND,
                name="requant_scalar_stage", shape=runtime.Shape(rows=256, columns=1), abi=abi,
                requantization=runtime.RequantizationSpec(
                    elements=256, groups=8, scale="1/16", saturation="signed_int8",
                    grid=(256, 1, 1), threadgroup=(32, 1, 1), dispatch_boundary="required"))

    def test_requant_marker_survives_typed_and_emission_abi_round_trip(self):
        p = cc.compile_function(make_program())
        contract = p.contract()
        marker = contract.requantization
        self.assertEqual(marker.kind, "int32_to_int8")
        self.assertEqual(marker.source_binding, 3)
        self.assertEqual(marker.destination_binding, 4)
        self.assertEqual(marker.scale, "1/512")
        self.assertEqual(p.abi_plain(p.abi())["requantization"], {
            "kind": "int32_to_int8", "source_binding": 3, "destination_binding": 4,
            "elements": 256, "groups": 8, "scale": "1/512",
            "rounding": "round_half_to_even", "saturation": "signed_int8",
            "dispatch_boundary": "required", "scale_storage": "uint32_bits",
            "output_storage": "int32_word"})
        self.assertEqual(contract.to_dict()["requantization"]["saturation"], "signed_int8")
        from agxforge.g17 import abi as g17abi
        self.assertEqual(g17abi.ProgramABI.from_dict(contract.to_dict()).requantization, marker)

    def test_public_authoring_cannot_drop_the_marker_to_reach_generic_metadata(self):
        from agxforge.g17 import scanlink
        p = cc.compile_function(make_program())
        emission = p.abi_plain(p.abi())
        emission.pop("requantization")
        with self.assertRaisesRegex(ValueError, "typed requantization marker differs"):
            scanlink.author(p.code, p.contract(), emission=emission)


if __name__ == "__main__":
    unittest.main()
