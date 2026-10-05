"""A device load in a texture kernel: admitted inside root's measured base domain, refused outside.

WHAT WAS OPEN AND WHAT CLOSED IT. The load's base was already the binding RANK here, measured from
Apple's indexed pair at public indices [1,2,3] and [2,4,6]. What was NOT measured was whether the
rank law survives INTERNAL texture bindings - a load resolving to internal 44 reads descriptor
state - so `select` refused a device load in any texture kernel outright. Root measured exactly that
with four Apple-compiled sources, objects and full decoder output retained at
results/g17-source-admission-v1/texture-indexed-load-base-measurement.json, nothing dispatched:

    0 textures, buffer 0   ->  expr:bin(op0,const(0),8)
    1 texture,  buffer 0   ->  expr:bin(op0,const(8),8)
    2 textures, buffer 0   ->  expr:bin(op0,const(8),8)
    2 textures, buffer 7   ->  expr:bin(op0,const(8),8)

const(8) = 4 * 2, TEXTURE_INTERNALS is 2 for one texture as much as for two, and the existing rank
calculation reproduces all four - no new public-index table, which is what root asked for. Two
shapes of that evidence matter and are asserted below: the base does NOT depend on the public index
(0 and 7 agree) and does NOT scale with the texture count (1 and 2 agree).

THE THREE BOUNDARIES STAY REFUSED, each by name: a second user buffer (rank 3 is unmeasured), a
non-word load (the narrow forms carry their own base field), and a constant-slot load (root's
constant-index controls selected op12688 and are retained separately).

NOTHING HERE IS DISPATCHED. Everything is selection, emission and this project's own decoder.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

import g17cc
import g17packedcheck as D
from agxforge.g17 import ir

WORD_LOAD = 12682


def coordinate_program(*, textures=1, slot=0, second_buffer=False, imm_index=False,
                       width=None, element=None):
    """Root's shape as authored IR: one user buffer, N read textures, coordinates from a load."""
    bufs = [ir.Buffer("b", slot, elem=ir.I32)]
    if second_buffer:
        bufs.append(ir.Buffer("c", slot + 1, elem=ir.I32))
    f = ir.Function("k", bufs)
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    kw = dict(width=width) if width else {}
    tkw = dict(type=element) if element else {}
    x = b.load(bufs[0], ir.Imm(4) if imm_index else b.add(t, b.const(4), name="i"), name="x", **kw)
    y = b.load(bufs[0], b.add(t, b.const(5), name="j"), name="y")
    b.store(bufs[0], ir.Imm(16), b.texture_read(x, y, tex=0, name="v0", **tkw))
    if textures == 2:
        x2 = b.load(bufs[0], b.add(t, b.const(6), name="k6"), name="x2")
        y2 = b.load(bufs[0], b.add(t, b.const(7), name="k7"), name="y2")
        b.store(bufs[0], ir.Imm(20), b.texture_read(x2, y2, tex=1, name="v1", **tkw))
    b.ret()
    return f


def load_bases(prog):
    text = "\n".join("%d %s" % (r[2], " ".join(r[3])) for r in D.decode(bytes(prog.code)))
    return sorted({int(m.group(1))
                   for m in re.finditer(r"%d .*?const\((\d+)\)" % WORD_LOAD, text)})


class TheMeasuredDomainIsAdmitted(unittest.TestCase):
    def test_one_texture_at_buffer_zero_loads_from_base_eight(self):
        self.assertEqual(load_bases(g17cc.compile_function(coordinate_program(textures=1))), [8])

    def test_two_textures_at_buffer_zero_load_from_base_eight(self):
        self.assertEqual(load_bases(g17cc.compile_function(coordinate_program(textures=2))), [8])

    def test_the_base_does_not_depend_on_the_public_index(self):
        """Apple's buffer 0 and buffer 7 agree at const(8); so must this."""
        a = load_bases(g17cc.compile_function(coordinate_program(textures=2, slot=0)))
        b = load_bases(g17cc.compile_function(coordinate_program(textures=2, slot=7)))
        self.assertEqual((a, b), ([8], [8]))

    def test_the_base_does_not_scale_with_the_texture_count(self):
        """One texture and two both give const(8) in Apple's objects: the internals are 2 either way."""
        a = load_bases(g17cc.compile_function(coordinate_program(textures=1)))
        b = load_bases(g17cc.compile_function(coordinate_program(textures=2)))
        self.assertEqual(a, b)
        self.assertEqual(g17cc.TEXTURE_INTERNALS, 2)

    def test_a_kernel_with_no_texture_still_loads_from_base_zero(self):
        """The control that makes the eight mean something: without a texture the rank is 0."""
        bufs = [ir.Buffer("b", 0, elem=ir.I32)]
        f = ir.Function("k", bufs)
        b = ir.Builder(f, f.block("entry"))
        t = b.builtin("thread_position_in_grid", name="t")
        x = b.load(bufs[0], b.add(t, b.const(4), name="i"), name="x")
        b.store(bufs[0], ir.Imm(16), x)
        b.ret()
        self.assertEqual(load_bases(g17cc.compile_function(f)), [0])

    def test_the_two_r32float_coordinate_program_compiles(self):
        prog = g17cc.compile_function(coordinate_program(textures=2, element=ir.F32))
        self.assertGreater(len(bytes(prog.code)), 0)
        res = prog.abi()["resources"]
        self.assertEqual(sorted(t["element"] for t in res["textures"]), ["float32", "float32"])
        self.assertEqual(sorted(t["dense_index"] for t in res["textures"]), [0, 1])


class TheReadFactDescribesTheEmittedLoads(unittest.TestCase):
    """Root: "Read access/lifetime/load-wait facts must describe the emitted coordinate loads.\""""

    def setUp(self):
        self.res = g17cc.compile_function(
            coordinate_program(textures=2, element=ir.F32)).abi()["resources"]
        self.user = [a for a in self.res["access"] if a["kind"] == "user"]

    def test_the_user_record_is_recorded_as_read(self):
        self.assertEqual(len(self.user), 1)
        self.assertTrue(self.user[0]["read"])

    def test_the_coordinate_read_is_recorded_DIVERGENT(self):
        """It was recorded uniform: the predicate read a field the load forms do not carry."""
        self.assertFalse(self.user[0]["uniform"])

    def test_the_internal_records_are_still_the_measured_pair(self):
        self.assertEqual([(i["rank"], i["apple_index"]) for i in self.res["internal"]],
                         [(0, 44), (1, 48)])


class TheBoundariesStayRefused(unittest.TestCase):
    def test_a_second_user_buffer_refuses_by_name(self):
        with self.assertRaises(g17cc.Unsupported) as cm:
            g17cc.compile_function(coordinate_program(textures=2, second_buffer=True))
        self.assertIn("2 user buffers", str(cm.exception))
        self.assertIn("unmeasured", str(cm.exception))

    def test_a_constant_slot_load_refuses_by_name(self):
        with self.assertRaises(g17cc.Unsupported) as cm:
            g17cc.compile_function(coordinate_program(textures=1, imm_index=True))
        self.assertIn("constant-index load", str(cm.exception))
        self.assertIn("op12688", str(cm.exception))

    def test_a_narrow_load_refuses_by_name(self):
        with self.assertRaises(g17cc.Unsupported) as cm:
            g17cc.compile_function(coordinate_program(textures=1, width="half"))
        self.assertIn("'half' load", str(cm.exception))

    def test_every_refusal_names_the_retained_measurement(self):
        for kw in (dict(second_buffer=True), dict(imm_index=True), dict(width="half")):
            with self.subTest(**kw):
                with self.assertRaises(g17cc.Unsupported) as cm:
                    g17cc.compile_function(coordinate_program(textures=1, **kw))
                self.assertIn("texture-indexed-load-base-measurement.json", str(cm.exception))

    def test_the_domain_predicate_accepts_the_measured_shape(self):
        """A guard must first accept ground truth."""
        for textures in (1, 2):
            for slot in (0, 7):
                with self.subTest(textures=textures, slot=slot):
                    self.assertIsNone(g17cc._texture_load_domain(
                        coordinate_program(textures=textures, slot=slot)))


class ThereIsNoSecondGuardAndTheReasonIsMeasured(unittest.TestCase):
    """Three guards were tried in _resource_layout and all three were UNREACHABLE. Asserted here so
    nobody adds them back believing they are evidence - and so the limitation that survives is
    written down as a test rather than a comment.
    """

    def seen(self, fn):
        out = {}
        real = g17cc._resource_layout

        def spy(layout, ranks, abi_bindings):
            out.update(layout=layout, ranks=ranks, abi_bindings=abi_bindings)
            return real(layout, ranks, abi_bindings)
        g17cc._resource_layout = spy
        try:
            g17cc.compile_function(fn)
        finally:
            g17cc._resource_layout = real
        return out

    def test_a_record_holding_a_load_is_never_uniform(self):
        """So "is the read uniform?" could not have failed."""
        res = g17cc.compile_function(
            coordinate_program(textures=2, element=ir.F32)).abi()["resources"]
        self.assertFalse([a for a in res["access"] if a["kind"] == "user"][0]["uniform"])

    def test_a_stray_load_base_reads_back_as_NOT_READ_rather_than_refusing(self):
        """The limitation that survives, measured: `read` is defined in terms of the measured base,
        so moving the base makes the record read=False and no guard in this function can see it."""
        seen = self.seen(coordinate_program(textures=2, element=ir.F32))
        for _off, _raw, m in seen["layout"]:
            if m.form.startswith("load"):
                m.fields["base"] = 12                    # rank 3, which nothing measured
        res = g17cc._resource_layout(seen["layout"], seen["ranks"], seen["abi_bindings"])
        self.assertIsNotNone(res)
        self.assertFalse([a for a in res["access"] if a["kind"] == "user"][0]["read"])

    def test_what_actually_stands_between_that_and_an_image_is_select(self):
        """Both surviving gates are in select, and both are exercised above and here."""
        src = open(os.path.join(ROOT, "agxforge", "g17", "cc.py")).read()
        self.assertIn("an internal index has "
                      "collided with a user one", src)
        self.assertIn("outside the measured base domain", src)



class CoordinateLoadWaits(unittest.TestCase):
    def test_disabling_wait_reproduces_retained_hardware_failure(self):
        import hashlib
        from unittest.mock import patch
        from examples.g17_compile import compile_texture_pair
        old = "0cb2cdb45bc7e3ddabea97e8af022d43768393586638aa130368204bc2e6f300"
        with patch.object(g17cc, '_wait_for_load', side_effect=lambda out, value: value):
            broken = compile_texture_pair(coordinate_inputs=True)
        self.assertEqual(hashlib.sha256(broken.code).hexdigest(), old)
        fixed = compile_texture_pair(coordinate_inputs=True)
        self.assertNotEqual(fixed.code, broken.code)
        waits = [m for _, _, m in fixed.layout if m.form == 'alu.12' and m.fields.get('load_wait') == 1]
        self.assertEqual(len(waits), 4)
        publications = [m for _, _, m in fixed.layout if m.form.startswith('publish.coord.')]
        self.assertEqual(len(publications), 4)
        self.assertEqual({m.uses[0] for m in publications}, {m.defs[0] for m in waits})
        # Four extra waited ALUs. No other instruction is introduced.
        self.assertEqual(len(fixed.code), len(broken.code)+4*12)
        self.assertEqual(len(D.decode(fixed.code)), len(D.decode(broken.code))+4)

    def test_nonloaded_coordinates_keep_the_validated_program(self):
        from unittest.mock import patch
        from examples.g17_compile import compile_texture_pair
        original = compile_texture_pair()
        with patch.object(g17cc, '_wait_for_load', side_effect=lambda out, value: value):
            control = compile_texture_pair()
        self.assertEqual(original.code, control.code)

    def test_one_loaded_value_published_twice_is_waited_once(self):
        f = ir.Function('same_xy', [ir.Buffer('b', 0)])
        b = ir.Builder(f, f.block('entry'))
        t = b.builtin('thread_position_in_grid')
        x = b.load(f.buffers[0], t)
        v = b.texture_read(x, x, tex=0, type=ir.F32)
        b.store_fetch(f.buffers[0], ir.Imm(16), v, b.const(0), components=1)
        b.ret()
        p = g17cc.compile_function(f)
        waits = [m for _, _, m in p.layout if m.form == 'alu.12' and m.fields.get('load_wait') == 1]
        self.assertEqual(len(waits), 1)
        pubs = [m for _, _, m in p.layout if m.form.startswith('publish.coord.')]
        self.assertEqual(len(pubs), 2)
        self.assertTrue(all(m.uses[0] is waits[0].defs[0] for m in pubs))

# Keep direct execution and unittest discovery on the same set of tests.
if __name__ == "__main__":
    unittest.main(verbosity=1)
