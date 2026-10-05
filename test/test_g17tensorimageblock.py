#!/usr/bin/env python3
"""Set A item 12: a tensor GEMM staging its outputs through an explicit imageblock. The compiler
states the imageblock, the linker declares it, and the image's three metadata sections equal
Apple's own compile of a matmul2d kernel with an explicit imageblock, byte for byte."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import g17tensorcommonruntime as R  # noqa: E402
from agxforge.g17 import scanlink  # noqa: E402

WITNESS = os.path.join(ROOT, "results/g17-tensor-imageblock-witness-v1")
SECTIONS = ("__GPU_METADATA,__compute", "__GPU_LD_MD,__compute", "__GPU_ARCH_LD_MD,__compute")


def sections(raw):
    import g17obj
    s, _ = g17obj.sections_of(raw)
    return {k: raw[o:o + n] for k, (o, n) in s.items()}


def image(mode):
    s = R.generic_spec(dict(M=32, N=32, K=64, imageblock=mode))
    p = R.build_generic_program(s)
    if mode == "undeclared":
        p._abi_imageblock = None
    return p, sections(scanlink.author(p).object)


class TensorImageblock(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(os.path.join(WITNESS, "both.o")):
            self.skipTest("witness not extracted (make evidence)")

    def test_the_compiler_states_the_imageblock_and_the_measured_set(self):
        p, _ = image("stage")
        abi = p.abi_plain(p.abi())
        self.assertEqual(abi["imageblock"], {"layout": "explicit", "element_bytes": 4})
        self.assertEqual(list(abi["system_registers"]), [130, 164, 165])

    def test_the_declared_image_is_apples_metadata_byte_for_byte(self):
        apple = sections(open(os.path.join(WITNESS, "both.o"), "rb").read())
        _, ours = image("stage")
        for k in SECTIONS:
            with self.subTest(section=k):
                self.assertEqual(ours[k], apple[k])

    def test_the_undeclared_control_differs_only_by_the_declaration(self):
        tensor = sections(open(os.path.join(WITNESS, "tensor.o"), "rb").read())
        _, ctl = image("undeclared")
        _, dec = image("stage")
        self.assertEqual(ctl["__GPU_LD_MD,__compute"], tensor["__GPU_LD_MD,__compute"])
        self.assertEqual(ctl["__GPU_ARCH_LD_MD,__compute"], tensor["__GPU_ARCH_LD_MD,__compute"])
        self.assertNotEqual(ctl["__GPU_METADATA,__compute"], dec["__GPU_METADATA,__compute"])

    def test_the_neighbour_arm_reads_x_plus_one_after_an_imageblock_barrier_inside_a_region(self):
        # the round-2 discriminator: a same-lane round trip passed with nothing declared, so the
        # read that matters is a NEIGHBOUR's, fenced so lane 31 never reads outside the 32x1 tile
        from agxforge.g17 import model
        s = R.generic_spec(dict(M=32, N=32, K=64, imageblock="neighbour"))
        ins = [i for i in model.decode(R.build_generic_program(s).code, 0) if i.opcode]
        ops = [i.opcode.id for i in ins]
        w = ops.index(13075)
        # searched from the write onward: the tensor body before it carries its own instances
        bar, cmp, push, rd, pop = (ops.index(o, w) for o in (447, 10369, 582, 12151, 577))
        self.assertTrue(w < bar < cmp < push < rd < pop)
        self.assertEqual(ins[bar].raw[:2], bytes.fromhex("4751"))          # Apple's imageblock scope
        self.assertEqual(dict(enumerate(v for _k, v in ins[rd].values))[7], 1)   # x offset +1
        self.assertEqual(dict(enumerate(v for _k, v in ins[cmp].values))[5], 31)  # lane < 31

    def test_both_arms_keep_the_coordinate_the_read_reuses(self):
        # rounds 1 and 2 ran with op13075 slot 6 = 16 (release) while the read reused the register;
        # the liveness fix must keep it in the tensor programs too, including across the branch region
        from agxforge.g17 import model
        for mode in ("stage", "neighbour"):
            s = R.generic_spec(dict(M=32, N=32, K=64, imageblock=mode))
            ins = [i for i in model.decode(R.build_generic_program(s).code, 0) if i.opcode]
            store = next(i for i in ins if i.opcode.id == 13075)
            read = next(i for i in ins if i.opcode.id == 12151)
            with self.subTest(mode=mode):
                self.assertEqual([v for _k, v in store.values][6], 0)
                self.assertEqual([v for _k, v in store.values][5], [v for _k, v in read.values][5])

    def test_the_open_neighbour_arm_differs_from_the_fenced_one_only_by_the_region(self):
        from agxforge.g17 import model
        reads, rest = {}, {}
        for mode in ("neighbour", "neighbour_open"):
            s = R.generic_spec(dict(M=32, N=32, K=64, imageblock=mode))
            ins = [i for i in model.decode(R.build_generic_program(s).code, 0) if i.opcode]
            reads[mode] = [i.raw for i in ins if i.opcode.id == 12151]
            rest[mode] = [i.opcode.id for i in ins if i.opcode.id not in (10369, 582, 577)]
        self.assertEqual(reads["neighbour"], reads["neighbour_open"])
        self.assertEqual(rest["neighbour"], rest["neighbour_open"])
        self.assertEqual(R.generic_unscored(dict(imageblock="neighbour_open")), [(0, 31)])
        self.assertEqual(R.generic_unscored(dict(imageblock="neighbour")), [])

    def test_an_explicit_column_is_the_reads_coordinate_with_no_offset(self):
        # Apple's own neighbour read, tensor kernel or not: compute the column, read with offset 0
        from agxforge.g17 import model
        s = R.generic_spec(dict(M=32, N=32, K=64, imageblock="x_neighbour"))
        ins = [i for i in model.decode(R.build_generic_program(s).code, 0) if i.opcode]
        read = next(i for i in ins if i.opcode.id == 12151)
        store = next(i for i in ins if i.opcode.id == 13075)
        vals = [v for _k, v in read.values]
        self.assertEqual(vals[7], 0)                                        # no dx offset
        self.assertNotEqual(vals[5], [v for _k, v in store.values][5])       # not the packed SR coordinate
        with self.assertRaises(Exception):
            from agxforge.g17 import ir
            f = ir.Function("k", [ir.Buffer("A", 0)])
            b = ir.Builder(f, f.block("e"))
            b.imageblock_read(x=b.const(1), dx=1)

    def test_the_store_waits_on_what_it_reads(self):
        # op13075 operand 1 bits 24-31 are its wait mask (Piece B, 2026-09-23). The store waits on
        # slot 0 (the coordinate's read_sr) and nothing else; a LOADED stored value reaches it only
        # through the measured waiting copy (alu.12 with load_wait), never straight from the load.
        # This replaces a test of "bit 31 = shared storage", which was a wait on slot 7 misread.
        from agxforge.g17 import cc, model
        for mode in ("stage", "x_own", "x_neighbour", "x_neighbour_early", "neighbour_open"):
            s = R.generic_spec(dict(M=32, N=32, K=64, imageblock=mode))
            ins = [i for i in model.decode(R.build_generic_program(s).code, 0) if i.opcode]
            k = next(j for j, i in enumerate(ins) if i.opcode.id == 13075)
            store = [v for _k, v in ins[k].values]
            with self.subTest(mode=mode):
                self.assertEqual(store[1], cc.IB_STORE_OP1)
                self.assertEqual(store[1] >> 24 & 0xFF, 1)             # slot 0 only
                producer = next(i for i in reversed(ins[:k]) if i.opcode and
                                [v for kk, v in i.values if kk == "reg"][:1] == [store[0]])
                self.assertNotIn(producer.opcode.id, (12682, 12151))    # not a load straight in

    def test_a_non_tensor_imageblock_keeps_the_measured_free_byte(self):
        from agxforge.g17 import imageblock, mdgen, ldmd
        md = mdgen.build([0, 1, 2], layout=mdgen.THREE_COORDINATE,
                         system_registers=list(mdgen.COORDINATE_REGISTERS), register_count=8)
        out, _, _ = imageblock.declare(md, ldmd.build(entry=64), ldmd.build_arch(), 8)
        from agxforge.g17 import gpumd
        t = gpumd.kernel_table(out)
        import struct
        vt = t - struct.unpack_from("<i", out, t)[0]
        self.assertEqual(struct.unpack_from("<H", out, vt + 4 + 2 * imageblock.PK_SLOT)[0], imageblock.PK_OFFSET)


if __name__ == "__main__":
    unittest.main()
