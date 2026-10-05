"""The explicit-imageblock declaration: what executed, what it reproduces, and what it refuses."""
import json
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from agxforge.g17 import imageblock as IB, ldmd, mdgen
import g17gpumd as GM

REC = json.load(open(os.path.join(ROOT, "isa", "g17-imageblock-receipt.json")))


def control_sections():
    md = mdgen.build([0, 1, 2], layout=mdgen.THREE_COORDINATE,
                     system_registers=list(mdgen.COORDINATE_REGISTERS),
                     register_count=REC["apple_register_count"])
    return md, ldmd.build(entry=REC["entry"]), ldmd.build_arch()


class TheReceipt(unittest.TestCase):
    def test_declared_arm_is_exact_and_the_control_is_not(self):
        exp = REC["expected"]
        dec, ctl = REC["arms"]["declared"], REC["arms"]["control"]
        self.assertTrue(dec["agree"])
        self.assertEqual(len(dec["runs"]), 3)
        for run in dec["runs"]:
            self.assertEqual((run["a"], run["b"]), (exp["a"], exp["b"]))
            self.assertEqual(run["imageblock_bytes"], REC["lanes"] * REC["element_bytes"])
        for run in ctl["runs"]:
            self.assertEqual(run["imageblock_bytes"], 0)
            self.assertNotEqual(run["a"], exp["a"])

    def test_a_lane_can_only_pass_by_reading_its_neighbour(self):
        # The expected values are the RIGHT neighbour's, so echoing a lane's own write fails.
        own = [i * 3 + 1 for i in range(REC["lanes"])]
        self.assertNotEqual(own, REC["expected"]["a"])

    def test_the_executed_sections_are_what_declare_builds_today(self):
        got = IB.declare(*control_sections(), REC["element_bytes"])
        for name, g in zip(("metadata", "ld_md", "arch_ld_md"), got):
            self.assertEqual(g.hex(), REC["arms"]["declared"]["sections"][name], name)

    def test_the_control_arm_is_the_same_class_without_the_declaration(self):
        md, ld, arch = control_sections()
        self.assertEqual(md.hex(), REC["arms"]["control"]["sections"]["metadata"])
        self.assertEqual(ld.hex(), REC["arms"]["control"]["sections"]["ld_md"])


class TheDeclaration(unittest.TestCase):
    def test_it_writes_the_three_measured_facts(self):
        md, ld, arch = IB.declare(*control_sections(), 12)
        self.assertEqual(GM.fields(md).get(19), 1)
        self.assertEqual((ld[100], ld[154]), (18, 1))
        self.assertEqual(struct.unpack_from("<I", arch, 32)[0], 12)
        self.assertEqual(len(arch), 40)

    def test_nothing_else_in_the_per_kernel_table_moves(self):
        base = control_sections()[0]
        md = IB.declare(*control_sections(), 4)[0]
        before, after = GM.fields(base), GM.fields(md)
        self.assertEqual({k: v for k, v in after.items() if k != 19}, before)

    def test_it_refuses_what_it_was_not_measured_on(self):
        md, ld, arch = control_sections()
        with self.assertRaises(IB.Refused):
            IB.declare(md, ld, arch, 0)
        with self.assertRaises(IB.Refused):
            IB.declare(md, ldmd.build(entry=REC["entry"], atomic=True, size=224), arch, 4)
        with self.assertRaises(IB.Refused):                   # twice
            IB.declare(*IB.declare(md, ld, arch, 4), 4)
        with self.assertRaises(IB.Refused):                   # slot 23's body byte already set
            bad = bytearray(ld)
            bad[154] = 1
            IB.declare(md, bytes(bad), arch, 4)

    def test_it_refuses_a_table_whose_byte_53_is_a_field(self):
        md = bytearray(control_sections()[0])
        t = GM.kernel_table(bytes(md))
        vt = t - struct.unpack_from("<i", md, t)[0]
        struct.pack_into("<H", md, vt + 4 + 2 * 20, 52)      # a field starting at 52 covers 53
        with self.assertRaises(IB.Refused):
            IB.declare(bytes(md), *control_sections()[1:], 4)


class TheCoordinateClass(unittest.TestCase):
    def test_three_is_unchanged_for_its_witness(self):
        self.assertEqual(mdgen.build([0, 1, 2], layout=mdgen.THREE, system_registers=[156]),
                         mdgen.build([0, 1, 2], layout=mdgen.THREE))

    def test_the_coordinate_class_admits_only_the_coordinate_pair(self):
        with self.assertRaises(ValueError):
            mdgen.build([0, 1, 2], layout=mdgen.THREE_COORDINATE, system_registers=[164])
        with self.assertRaises(ValueError):
            mdgen.build([0, 1, 2], layout=mdgen.THREE_COORDINATE, system_registers=[156, 164])

    def test_layout_for_selects_it_only_for_that_pair(self):
        self.assertIs(mdgen.layout_for([0, 1, 2], system_registers=[164, 165]),
                      mdgen.THREE_COORDINATE)
        self.assertIs(mdgen.layout_for([0, 1, 2], system_registers=[156]), mdgen.THREE)


class TheLinker(unittest.TestCase):
    """scanlink.link applies the declaration when the ABI asks, and refuses an implicit layout."""

    @classmethod
    def setUpClass(cls):
        from agxforge.g17 import cc as g17cc, link as g17link, ir
        f = ir.Function("ibprog", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
        b = ir.Builder(f, f.block("e"))
        tid = b.builtin("thread_position_in_threadgroup", axis="x", name="tid")
        b.store_at(f.buffers[2], b.add(b.const(256, name="base"), tid, name="at"), tid, width="word")
        b.ret(); ir.verify(f)
        p = g17cc.compile_function(f)
        cls.abi = dict(p.abi())
        plain = p.abi_plain(p.abi())
        cls.offsets = [b["offset"] for b in plain["bindings"]]
        cls.kernel = g17link.Kernel(code=p.code, name=p.name, entry=cls.abi["entry"],
                                    prologue=cls.abi["prologue"],
                                    bindings=[g17link.Binding(index=b["index"], readonly=not b["written"],
                                                              element_type=b["element_type"])
                                              for b in plain["bindings"]])
        # THE COORDINATE PAIR IS STATED HERE, not read off this program: an imageblock program's
        # prologue reads both, and this compiler cannot yet describe one through its ABI (the form
        # table has no store.ib.32 opcode). What is under test is the linker's handling of the set.
        cls.abi["system_registers"] = list(mdgen.COORDINATE_REGISTERS)
        cls.abi.setdefault("argument_state", dict(
            pointer_offsets=[[0, 0], [1, 2], [2, 4]], pointer_words=2, block_words=6, block_bytes=24,
            basis="emitted_pointer_offsets", offset_rule="2 * rank", indices_contiguous=True,
            not_stated=["per_kernel_slot_1"]))

    def link(self, imageblock):
        import g17obj
        from agxforge.g17 import scanlink
        abi = dict(self.abi)
        if imageblock is not None:
            abi["imageblock"] = imageblock
        img = scanlink.link(self.kernel, abi, binding_offsets=self.offsets)
        secs, _ = g17obj.sections_of(img.object)
        return {k: img.object[o:o + n] for k, (o, n) in secs.items()}

    def test_without_the_key_nothing_is_declared(self):
        s = self.link(None)
        self.assertIsNone(GM.fields(s["__GPU_METADATA,__compute"]).get(19))
        self.assertEqual(len(s["__GPU_ARCH_LD_MD,__compute"]), 32)

    def test_with_the_key_all_three_facts_are_written(self):
        s = self.link({"layout": "explicit", "element_bytes": 8})
        self.assertEqual(GM.fields(s["__GPU_METADATA,__compute"]).get(19), 1)
        ld = s["__GPU_LD_MD,__compute"]
        self.assertEqual((ld[100], ld[154]), (18, 1))
        self.assertEqual(struct.unpack_from("<I", s["__GPU_ARCH_LD_MD,__compute"], 32)[0], 8)

    def test_an_implicit_layout_is_refused(self):
        with self.assertRaises(Exception) as cm:
            self.link({"layout": "implicit", "element_bytes": 8})
        self.assertIn("EXPLICIT", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
