#!/usr/bin/env python3
"""Set A item 10b: a whole GEMM D fragment staged between two tensor bodies through the explicit
imageblock (agxforge.g17.ibstage), and the straight-line tensor metadata tail that declares it
(tensormetadata.layout imageblock=True), checked against Apple's own compiles byte for byte."""
import hashlib
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import g17tensorcommonruntime as R  # noqa: E402
from agxforge.g17 import cc, ibstage, imageblock as IB, model, scanlink, tensorlife  # noqa: E402
from agxforge.g17 import tensormetadata as TM  # noqa: E402

WITNESS = os.path.join(ROOT, "results/g17-tensor-ibfragment-v1/witness")
MD, LD, ARCH = "__GPU_METADATA,__compute", "__GPU_LD_MD,__compute", "__GPU_ARCH_LD_MD,__compute"
# main's plain two-stage chain at M16 (results/g17-tensor-ibfragment-v1/M16_register): staging must
# not move a byte of it
PLAIN_M16 = "1d6ef92937f9ed4c"


def sections(raw):
    import g17obj
    s, _ = g17obj.sections_of(raw)
    return {k: raw[o:o + n] for k, (o, n) in s.items()}


def witness(name):
    return sections(open(os.path.join(WITNESS, name + ".o"), "rb").read())


def spec(M, stage_through=None):
    return R.generic_spec(dict(M=M, N=32, K=32, stages=[[16, 32, "float"]], stage_through=stage_through))


def program(M, stage_through=None):
    return R.build_generic_program(spec(M, stage_through))


def decoded(p):
    return [i for i in model.decode(p.code, 0) if i.opcode]


def vals(i):
    return [v for _k, v in i.values]


class StraightLineTail(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(os.path.join(WITNESS, "k32_both.o")):
            self.skipTest("witness not extracted (make evidence)")

    def test_the_authored_section_is_apples_k32_both_byte_for_byte(self):
        apple = witness("k32_both")[MD]
        rc = struct.unpack_from("<I", apple, 152 + 60)[0]
        ours = TM.emit([(1, 0, False), (2, 2, False), (3, 4, True)], register_count=rc,
                       system_registers=(130, 164, 165), instruction_count=37, has_back_edge=False,
                       imageblock=True)
        self.assertEqual(ours, apple)

    def test_the_control_without_slot_19_is_not_apples_section(self):
        # the comparison can fail: the same inputs without the imageblock tail differ from the witness
        apple = witness("k32_both")[MD]
        rc = struct.unpack_from("<I", apple, 152 + 60)[0]
        ours = TM.emit([(1, 0, False), (2, 2, False), (3, 4, True)], register_count=rc,
                       system_registers=(130, 164, 165), instruction_count=37, has_back_edge=False)
        self.assertNotEqual(ours, apple)

    def test_the_declared_ld_and_arch_halves_are_apples(self):
        tensor, both = witness("k32_tensor"), witness("k32_both")
        self.assertEqual(IB._ld_arch(bytearray(tensor[LD]), tensor[ARCH], 4), (both[LD], both[ARCH]))
        for name, size in (("w16", 64), ("w64", 256)):
            with self.subTest(element=size):
                self.assertEqual(witness(name)[ARCH], IB.arch(size))

    def test_the_tail_order_puts_19_directly_above_44(self):
        self.assertEqual(TM._tail_order((44, 32)), [44, 32])
        self.assertEqual(TM._tail_order((44, 33, 32)), [44, 33, 32])
        self.assertEqual(TM._tail_order((44, 32, 19)), [44, 19, 32])
        # and it reproduces the loop placement imageblock.declare applies (44:54 19:55 33:56 32:57)
        self.assertEqual(TM._tail_order((44, 33, 32, 19)), [44, 19, 33, 32])

    def test_the_loop_table_is_refused_here(self):
        with self.assertRaises(ValueError):
            TM.layout(76, (130, 164, 165), 131, True, imageblock=True)

    def test_declare_without_the_per_kernel_half_requires_slot_19(self):
        md = TM.emit([(1, 0, False), (2, 2, False), (3, 4, True)], register_count=76,
                     system_registers=(130, 164, 165), instruction_count=131, has_back_edge=False)
        with self.assertRaises(IB.Refused):
            IB.declare(md, bytes(IB.LD_SIZE), bytes(32), 64, per_kernel=False)


class FragmentStaging(unittest.TestCase):
    def test_the_plain_chain_is_byte_identical_to_main(self):
        p = program(16)
        self.assertEqual(hashlib.sha256(p.code).hexdigest()[:16], PLAIN_M16)

    def test_the_staging_arm_states_the_measured_set_and_element(self):
        for M, element in ((16, 64), (64, 256)):
            with self.subTest(M=M):
                p = program(M, "imageblock")
                abi = p.abi_plain(p.abi())
                self.assertEqual(list(abi["system_registers"]), [130, 164, 165])
                self.assertEqual(abi["imageblock"], {"layout": "explicit", "element_bytes": element})

    def test_the_image_declares_slot_19_in_the_tail(self):
        md = sections(scanlink.author(program(16, "imageblock")).object)[MD]
        self.assertTrue(IB.pk_declared(md))
        self.assertEqual(len(md), 496)

    def test_positive_and_control_share_every_store_byte(self):
        pos, ctl = decoded(program(16, "imageblock")), decoded(program(16, "imageblock_noread"))
        ps = [i.raw for i in pos if i.opcode.id == 13075]
        cs = [i.raw for i in ctl if i.opcode.id == 13075]
        self.assertEqual(len(ps), 16)
        self.assertEqual(ps, cs)
        # every store carries cc's operand 1, a wait on slot 0 (the coordinate); bits 24-31 are a wait
        # mask, and the earlier "bit 31 = shared" reading is refuted (MM section 25.104)
        self.assertTrue(all(vals(i)[1] == cc.IB_STORE_OP1 for i in pos if i.opcode.id == 13075))
        self.assertEqual(sum(i.opcode.id == 12151 for i in pos), 16)
        self.assertEqual(sum(i.opcode.id == 12151 for i in ctl), 0)

    def test_every_zero_precedes_the_barrier_and_every_read_follows_it(self):
        ins = decoded(program(16, "imageblock"))
        ops = [i.opcode.id for i in ins]
        stores = [n for n, o in enumerate(ops) if o == 13075]
        loads = [n for n, o in enumerate(ops) if o == 12151]
        bar = ops.index(447, stores[-1])
        self.assertEqual(ins[bar].raw[:2], bytes.fromhex("4751"))
        self.assertLess(max(stores), bar)
        self.assertLess(bar, min(loads))

    def test_the_coordinate_is_built_once_before_the_first_store(self):
        # read_sr x (low half) then y (high half), the pair immediately before the first store
        ins = decoded(program(16, "imageblock"))
        ops = [i.opcode.id for i in ins]
        first = ops.index(13075)
        self.assertEqual(ops[first - 2:first], [14060, 14060])
        self.assertEqual(ops.count(13075), 16)

    def test_tensorlife_is_clean_on_both_arms(self):
        for st in ("imageblock", "imageblock_noread"):
            with self.subTest(arm=st):
                self.assertEqual(tensorlife.released_reads(program(16, st).code), [])

    def test_the_coordinate_outside_r0_r15_is_refused(self):
        with self.assertRaises(ValueError):
            ibstage.insts({0: 40}, 16)

    def test_the_staging_frees_no_allocated_register(self):
        # a stated result, not a hope: the composition still reserves the handed registers, so the
        # register count is the plain chain's; the D registers are only DEAD between store and read
        self.assertEqual(program(16, "imageblock").abi()["register_count"],
                         program(16).abi()["register_count"])

    def test_staging_refuses_more_than_one_stage_or_a_split(self):
        with self.assertRaises(ValueError):
            R.generic_spec(dict(M=16, N=32, K=32, stages=[[16, 32, "float"], [16, 32, "float"]],
                                stage_through="imageblock"))
        with self.assertRaises(ValueError):
            R.generic_spec(dict(M=64, N=32, K=32, simdgroups=2, stages=[[64, 32, "float"]],
                                stage_through="imageblock"))


class SelfcheckNamedMember(unittest.TestCase):
    def test_a_named_member_immediate_is_not_the_y_offset(self):
        # op3=#4 on load.ib.32 was read as dy=4 and the staging block failed its own selfcheck
        code, layout = cc.emit([cc.MInst("load.ib.32", 14, dict(member=4, dx=0, dy=0, explicit=True,
                                                                 _defs=[20], _uses=[2]))])
        self.assertEqual(cc.selfcheck(layout), [])


if __name__ == "__main__":
    unittest.main()
