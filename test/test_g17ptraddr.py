"""Loop-carried addresses in the qmv loop (build_qmv2 lay["ptr_addr"]; MM 25.141.3 item 4, MM 25.141.17).

Without the flag every trip re-forms its addresses from the counter k: the x vector index (k * 32E, + base, >> 2,
then + j for each later vector load), each row's weight-vector index (k * 32 wpt, + row base, >> wsh) and each
row's scale index ((k * 32 wpt + lane * wpt) >> gsh, + row base). With it, those indices are loop phis whose entry
values - the split-K slice's first trip included - are formed once before the loop, and each advances by one add of
a constant. The trip's later x loads take the vector load's immediate BYTE displacement (16, 32, 48) off the one
carried index, as Apple's compile of MLX's qmv does.

The displacement law was measured on hardware (MM 25.141.17): the field adds F BYTES in the 8- and 14-byte forms
alike, 10 of 10 cases; encode_vec4 takes its disp in eight-byte units, so offset_bytes is passed x8. The four
flagged kernels below were bit-exact on hardware against the unchanged references (two seeds, sentinel).

Compile-only here:
- the flagged loop has no multiply (op10825) and no shift-add index (op17014) left, and is shorter;
- every x vector load in the loop reads ONE index register, at displacements 0, 16, 32, 48 as the decoder reads them;
- the loop carries no register released before it (test_g17phientry's check);
- the flag defaults off, and an unsupported layout is refused by name."""
import collections, os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from test_g17phientry import released_before_loop

BASE = dict(interleave=True, lean=True, coalesced=True, a16=True, xvec=True, wpt=2, vload=True, hoist_consts=True)


def _forms():
    import g17qmv as Q
    return {
        "q4 w2 split-K r1 S4": lambda f: Q.with_residual(dict(Q.case(2048, 8192, 4, 1, nocarrier=True)[0], **BASE, sgs=4,
                                                               ksplit=True, coop=True, hi16_scales=True, **f), "add32_to16"),
        "q4 ffn split-K r1 S2": lambda f: dict(Q.qmv_swiglu_layout(8192, 2048, 4, 1), **BASE, act32=True, sgs=2,
                                                ksplit=True, coop=True, hi16_scales=True, **f),
        "q8 qkv split-K r1 S4": lambda f: dict(Q.case(4096, 2048, 8, 1, nocarrier=True)[0], **BASE, sgs=4, ksplit=True,
                                                coop=True, hi16_scales=True, **f),
        "q4 qkv r4": lambda f: dict(Q.case(4096, 2048, 4, 4, nocarrier=True)[0], **BASE, hi16_scales=True, **f),
    }


def _loop(code):
    from agxforge.g17 import model, tensorview as TV
    v = TV.view(code)
    ins = list(model.decode(code, 0))
    first, back = list(TV.loops(v))[0]
    return ins[first:back + 1]


def _op(i):
    return str(i.opcode).split("(")[0]


class PtrAddr(unittest.TestCase):
    def test_the_loop_forms_no_address_from_k(self):
        import g17qmv as Q
        for name, lay in _forms().items():
            plain = _loop(Q.build_qmv2(lay({})).code)
            flagged = _loop(Q.build_qmv2(lay(dict(ptr_addr=True))).code)
            h = collections.Counter(_op(i) for i in flagged)
            self.assertEqual(h["op10825"], 0, name)
            self.assertEqual(h["op17014"], 0, name)
            self.assertGreater(collections.Counter(_op(i) for i in plain)["op10825"], 0, name)   # the check can fire
            self.assertLessEqual(len(flagged), len(plain) - 10, name)

    def test_x_loads_share_one_index_at_byte_displacements(self):
        import g17qmv as Q
        for name, lay in _forms().items():
            L = [i for i in _loop(Q.build_qmv2(lay(dict(ptr_addr=True))).code) if _op(i) == "op12709"]
            E = lay({})["wpt"] * lay({})["per_word"]
            self.assertEqual(len(L), E // 4, name)
            idx = {i.values[5] for i in L}
            disp = [i.values[7][1] for i in L]
            self.assertEqual(len(idx), 1, name)
            self.assertEqual(disp, [16 * j for j in range(E // 4)], name)

    def test_nothing_is_released_before_the_loop(self):
        import g17qmv as Q
        for name, lay in _forms().items():
            self.assertEqual(released_before_loop(Q.build_qmv2(lay(dict(ptr_addr=True))).code), [], name)

    def test_off_by_default_and_refused_outside_its_shape(self):
        import g17qmv as Q
        from agxforge.g17 import ir
        lay = _forms()["q4 qkv r4"]
        self.assertEqual(Q.build_qmv2(lay({})).code, Q.build_qmv2(lay(dict(ptr_addr=False))).code)
        with self.assertRaises(ValueError):
            Q.build_qmv2(dict(lay(dict(ptr_addr=True)), xvec=False, x16=True))
        fn = ir.Function("f", [ir.Buffer("A", 1, elem=ir.F16)])
        b = ir.Builder(fn, fn.block("entry"))
        for bad in (2, 256, -4):
            with self.assertRaises(ir.IRError):
                b.load_vec_at(fn.buffers[0], b.const(0), 4, offset_bytes=bad)


if __name__ == "__main__":
    unittest.main()
