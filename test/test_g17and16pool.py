"""op428 with its mask in the CONSTANT POOL (MM 25.141.16): ir.and16(pool=True), build_qmv2 lay["pool_masks"].

op428's second operand is a 16-bit UNIFORM register, not a GPR: the byte-6 type 0xa1 is the uniform one and the
index is byte 9 << 2 | byte 8 >> 6. Apple's compiler preloads masks above 8 bits into the constant pool, which
the driver places after the binding pointers - uniform half 4 x (buffers) + h. That is why cc's op428 with a GPR
mask read zero on hardware (MM 25.138): it named uniform 2r.

THE WITNESSES are Apple compiles of the three-binding cooperative LOOP control (tgctl3-u64b-iloop-short, MM
25.140.4) with its final store replaced by field extractions:

    ushort v = (ushort)acc;
    u[400] = (uint)(v & (ushort)0x0f00) * 3u + (uint)(v & (ushort)0xf000) * 5u;          (l3-m2, source 9b653ef65cbf770b)
    ... + (uint)(v & (ushort)0x0ff0) * 7u;                                                (l3-m3, source bb09e7760f3eac5a)

Their sections are 484 bytes, slot 1 = 8, slot 32 = 3, slot 33 = 1 - the empty-pool loop layout - and differ from
the empty witness only in the 8-byte slot-13 vector (halfword 0 zero, the masks after it, ascending). Their
instructions read the masks as uniforms 13, 14, 15. Compile-only: nothing here dispatches."""
import os, struct, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from test_g17phientry import released_before_loop

# Apple's sections, verbatim (metadata sha256 d1e4f10468ea19e0 and 457ea8931f2e0865)
APPLE_L3_M2 = bytes.fromhex(
    "100000000c000e0008000000000004000c000000140000007400000000000a000c000000080004000a000000400000000400"
    "0000080000006167632e6d61696e0000000048004400400030002c00240028000000200000001c0000001800000014001000"
    "00003f003e0000003d0000000000000000000000000000000c00080034000400000000003c003b0048000000440000003c00"
    "00004800000090000000980000009800000098000000980000001800000094000000a0000000080000000001000000000001"
    "03010101060000000000000002000000000000003000000001000000100000000c00140010000f00080004000c0000001000"
    "000001000000000000030800000000000000190000006167632e6d61696e2e636f6e7374616e745f70726f6772616d000000"
    "080000000000000f00f000000000000000000000000000000000000003000000900000007000000050000000020000002c00"
    "00001000000000000a000c000700000008000a00000000000003060000000c0012000b0000000c0004000c00000006000000"
    "000000060200000000000a0012000b0004000c000a00000002000000000000050400000000000a0010000b0004000c000a00"
    "00000100000000000005020000000c00080006000000000007000c00000000000501")
APPLE_L3_M3 = bytes.fromhex(
    "100000000c000e0008000000000004000c000000140000007400000000000a000c000000080004000a000000400000000400"
    "0000080000006167632e6d61696e0000000048004400400030002c00240028000000200000001c0000001800000014001000"
    "00003f003e0000003d0000000000000000000000000000000c00080034000400000000003c003b0048000000440000003c00"
    "00004800000090000000980000009800000098000000980000001800000094000000a0000000080000000001000000000001"
    "03010101060000000000000002000000000000003000000001000000100000000c00140010000f00080004000c0000001000"
    "000001000000000000030800000000000000190000006167632e6d61696e2e636f6e7374616e745f70726f6772616d000000"
    "080000000000000ff00f00f00000000000000000000000000000000003000000900000007000000050000000020000002c00"
    "00001000000000000a000c000700000008000a00000000000003060000000c0012000b0000000c0004000c00000006000000"
    "000000060200000000000a0012000b0004000c000a00000002000000000000050400000000000a0010000b0004000c000a00"
    "00000100000000000005020000000c00080006000000000007000c00000000000501")
BASE = dict(interleave=True, lean=True, coalesced=True, a16=True, xvec=True, wpt=2, vload=True, hoist_consts=True)


def _w2(pool, rows=1, sgs=4):
    import g17qmv as Q
    return Q, Q.with_residual(dict(Q.case(2048, 8192, 4, rows, nocarrier=True)[0], **BASE, sgs=sgs, ksplit=True,
                                   coop=True, pool_masks=pool), "add32_to16")


def _loop(code):
    from agxforge.g17 import model, tensorview as TV
    (first, back), = TV.loops(TV.view(code))
    return [i for i in list(model.decode(code, 0))[first:back + 1] if i.opcode]


class Encoding(unittest.TestCase):
    def test_every_pool_and_names_its_uniform(self):
        Q, lay = _w2(True)
        p = Q.build_qmv2(lay)
        pool = bytes(p.abi()["constant_pool"])
        self.assertEqual(pool.hex(), "0000000f00f00000")
        want = {0x0f00: 13, 0xf000: 14}                        # 4 x 3 bindings + halfword
        loop = _loop(p.code)
        seen = []
        for i in loop:
            if i.opcode.id == 428:
                b = p.code[i.offset:i.offset + i.size]
                self.assertEqual(b[6], 0xA1, "the uniform operand type")
                u = (b[9] << 2) | (b[8] >> 6)
                self.assertIn(u, want.values())
                self.assertEqual(i.values[4][0], "expr", "Apple's decoder reads the pool operand as an expression")
                seen.append(u)
        self.assertEqual(sorted(set(seen)), [13, 14])
        self.assertEqual(len(seen), 8, "4 of the 8 nibbles per 2-word trip need a mask above 8 bits")
        self.assertNotIn(17013, [i.opcode.id for i in loop], "no w >> 8 per word")
        self.assertEqual(released_before_loop(p.code), [])

    def test_the_default_form_is_untouched(self):
        Q, lay = _w2(False)
        p = Q.build_qmv2(lay)
        self.assertEqual(p.abi()["constant_pool"], ())
        self.assertNotIn(428, [i.opcode.id for i in _loop(p.code)])


class Section(unittest.TestCase):
    def _ours(self, pool, register_count):
        import g17mdgen as M
        from agxforge.g17 import cooperativemetadata as CM
        built = M.build([0, 1, 2], layout=M.threadgroup_three(256, 0, instructions=64, back_edge=True),
                        offsets=[0, 2, 4], register_count=register_count, system_registers=(156, 164))
        return CM._place_pool(built, pool, CM.THREE_SLOT1)       # _emit_three's own serialization

    def test_apples_mask_pool_sections_reproduce(self):
        for apple, halves in ((APPLE_L3_M2, (0, 0x0f00, 0xf000, 0)), (APPLE_L3_M3, (0, 0x0f00, 0x0ff0, 0xf000))):
            pool = b"".join(struct.pack("<H", h) for h in halves)
            from agxforge.g17 import gpumd as GM
            pk = GM.kernel_table(apple); slots, _ = GM.table_at(apple, pk)
            regs = struct.unpack_from("<I", apple, pk + slots[0])[0]
            self.assertEqual(self._ours(pool, regs), apple)

    def test_the_authored_program_carries_its_pool(self):
        from agxforge.g17 import scanlink, gpumd as GM
        import g17obj
        Q, lay = _w2(True)
        img = scanlink.author(Q.build_qmv2(lay))
        s, _ = g17obj.sections_of(img.object); o, n = s["__GPU_METADATA,__compute"]; md = img.object[o:o + n]
        pk = GM.kernel_table(md); slots, _ = GM.table_at(md, pk)
        at = pk + slots[13]; vec = at + struct.unpack_from("<I", md, at)[0]
        self.assertEqual(struct.unpack_from("<I", md, vec)[0], 8)
        self.assertEqual(md[vec + 4:vec + 12].hex(), "0000000f00f00000")
        self.assertEqual(struct.unpack_from("<I", md, pk + slots[1])[0], 8)
        self.assertEqual(len(md), len(APPLE_L3_M2))


class Refusals(unittest.TestCase):
    def test_only_the_witnessed_pool_shape(self):
        from agxforge.g17 import cooperativemetadata as CM
        ok = lambda *h: CM.three_mask_pool(b"".join(struct.pack("<H", x) for x in h))
        self.assertTrue(ok(0, 0x0f00, 0, 0) and ok(0, 0x0f00, 0xf000, 0) and ok(0, 0x0f00, 0x0ff0, 0xf000))
        self.assertFalse(ok(0, 0, 0, 0), "empty is the 16-byte form, not this one")
        self.assertFalse(ok(0x0f00, 0, 0, 0), "halfword 0 is zero in every witness")
        self.assertFalse(ok(0, 0, 0x0f00, 0), "masks follow halfword 0 without a gap")
        self.assertFalse(CM.three_mask_pool(bytes(16)))

    def test_a_fourth_mask_and_an_eight_bit_mask_refuse(self):
        from agxforge.g17 import ir
        c, a, bb = ir.Buffer("C", 0, elem=ir.F32), ir.Buffer("A", 1, elem=ir.F16), ir.Buffer("B", 2, elem=ir.F16)
        fn = ir.Function("k", [c, a, bb]); b = ir.Builder(fn, fn.block("entry"))
        v = b.load(a, b.builtin("thread_position_in_threadgroup"), type=ir.I32)
        for m in (0x0f00, 0xf000, 0x0ff0):
            b.and16(v, "L", m, pool=True)
        with self.assertRaises(ir.IRError):
            b.and16(v, "L", 0x1f00, pool=True)
        with self.assertRaises(ir.IRError):
            b.and16(v, "L", 0x00f0, pool=True)

    def test_pool_masks_need_the_cooperative_class(self):
        Q, lay = _w2(True)
        with self.assertRaises(ValueError):
            Q.build_qmv2(dict(lay, coop=False, ksplit=False, sgs=1))


class Cost(unittest.TestCase):
    def test_it_pays_only_with_rows_to_amortise_the_prescale(self):
        """Pool masks remove one w >> 8 per row-word but pre-scale x per HALF position (12 fmuls per q4 trip
        where the byte form needs 8), and the pre-scale is per trip, shared by the rows. At rows 1 (the
        delivered split-K forms) the loop grows 123 -> 125; at rows 4 it shrinks 318 -> 314."""
        for rows, sgs, grows in ((1, 4, True), (4, 2, False)):
            Q, off = _w2(False, rows, sgs)
            _, on = _w2(True, rows, sgs)
            a, b = len(_loop(Q.build_qmv2(off).code)), len(_loop(Q.build_qmv2(on).code))
            self.assertEqual(b > a, grows, (rows, a, b))


if __name__ == "__main__":
    unittest.main()
