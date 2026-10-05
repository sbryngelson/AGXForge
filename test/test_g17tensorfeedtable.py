#!/usr/bin/env python3
"""The register-feed TABLE (production row P2, machine model 25.125), compile only.

- Every entry is a FeedKey with a receipt under results/.
- Present keys: the receipted programs compile register-fed, byte for byte as before the table
  (hashes taken on the predicate this table replaced), and the accumulator (C) feed as dispatched.
- Absent keys: a representative enumeration over dtype, conversion, grid, role, transpose, offset,
  simdgroup and threadgroup split and modifiers. Each is a hand-off candidate whose key is not in the
  table, and each compiles EXACTLY as with TENSOR_REGISTER_FEED off: the same bytes, or the same
  refusal where the IR has no memory form (a declared register-only conversion or staging)."""
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "test"))

import g17tensorcommonruntime as R  # noqa: E402
from agxforge.g17 import cc, ir, model, tlower  # noqa: E402


def h(code):
    return hashlib.sha256(code).hexdigest()[:16]


def compiled(fn_or_build, feed=True):
    """(sha16 of the code, None) or (None, the refusal's type and text), with the feed on or off."""
    saved = cc.TENSOR_REGISTER_FEED
    cc.TENSOR_REGISTER_FEED = feed
    try:
        program = fn_or_build() if callable(fn_or_build) else cc.compile_function(fn_or_build)
        return h(program.code), None
    except Exception as why:          # the refusal itself is what must match
        return None, "%s: %s" % (type(why).__name__, why)
    finally:
        cc.TENSOR_REGISTER_FEED = saved


def generic(spec):
    return lambda: R.build_generic_program(spec)


def tensor_pairs(fn):
    ops = [o for b in fn.blocks for o in b.ops if o.kind == "tensor_matmul"]
    return list(zip(ops, ops[1:]))


def two_bodies(first, second, *, slots=(("A", "B", "C"), ("C", "B", "C")), between=False):
    """Two tensor bodies over buffers A(1, half) B(2, half) C(3, float); `first`/`second` are the
    tensor_matmul keyword arguments, `slots` which buffers each reads and writes."""
    a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    by = dict(A=a, B=b, C=c)
    fn = ir.Function("feedtable_probe", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
    bl.tensor_matmul(*(by[x] for x in slots[0]), **first)
    if between:
        t = bl.builtin("threadgroup_position_in_grid", name="t")
        bl.store_at(c, t, bl.load(c, t, type=ir.F32, name="v"))
    bl.tensor_matmul(*(by[x] for x in slots[1]), **second)
    bl.ret(); ir.verify(fn)
    return fn


# THE PRESENT KEYS' PROGRAMS, hashed on main's predicate before the table replaced it (and, for the
# C feed, the bytes dispatched in results/g17-tensor-cfeed-v1).
SQ = dict(M=32, N=32, K=64)
PRESENT = {
    "chain_register": (lambda: R.build_chain_program("chain_register"), "6887f3b7eabf3456"),
    "epilogue_register": (lambda: R.build_chain_program("epilogue_register"), "0e092f1be4115e32"),
    "transformer_layer": (R.build_transformer_layer_program, "43287db39af19c44"),
    "transformer_two_layer": (R.build_two_transformer_layers_program, "94d62248ff7c97c1"),
    "transformer_layer_weight_offset": (R.build_transformer_layer_weight_offset_program, "49ae0337fcd6ccdb"),
    "transformer_continuation_weight_offset": (R.build_transformer_continuation_weight_offset_program,
                                               "719fab01c8d1e9ce"),
    "chain_M32N32K64_3232half_fixed": (generic(dict(SQ, stages=[[32, 32, "half"]])), "97cc3f208d108297"),
    "chain_M16N32K32_4832float1648float": (generic(dict(M=16, N=32, K=32, stages=[[48, 32, "float"], [16, 48, "float"]])),
                                           "4aecdb7cc5a87904"),
    "chain_M16N64K64_3264half": (generic(dict(M=16, N=64, K=64, stages=[[32, 64, "half"]])), "925032241adf3f85"),
    "chain_M32N64K64_1664float": (generic(dict(M=32, N=64, K=64, stages=[[16, 64, "float"]])), "4ee38231ec5ec8f1"),
    "M16_register": (generic(dict(M=16, N=32, K=32, stages=[[16, 32, "float"]])), "1d6ef92937f9ed4c"),
    "M64_register": (generic(dict(M=64, N=32, K=32, stages=[[16, 32, "float"]])), "ad1a1a792dff1f41"),
    "M16_imageblock": (generic(dict(M=16, N=32, K=32, stages=[[16, 32, "float"]], stage_through="imageblock")),
                       "0dbc3d5df732e468"),
    "M64_imageblock": (generic(dict(M=64, N=32, K=32, stages=[[16, 32, "float"]], stage_through="imageblock")),
                       "872e5c3a985ebf52"),
    "M16_imageblock_noread": (generic(dict(M=16, N=32, K=32, stages=[[16, 32, "float"]],
                                           stage_through="imageblock_noread")), "0c7bd7642fb6c25d"),
    "feed_B_float": (generic(dict(SQ, stages=[[32, 32, "float", "B"]])), "f1ed5608eb35af87"),
    "feed_B_half": (generic(dict(SQ, stages=[[32, 32, "half", "B"]])), "320ac87a0f876b6d"),
    "feed_At_float": (generic(dict(SQ, stages=[[32, 32, "float", "At"]])), "21af3311262362d3"),
    "feed_At_half": (generic(dict(SQ, stages=[[32, 32, "half", "At"]])), "65e6a961c4711fac"),
    "feed_Bt_float": (generic(dict(SQ, stages=[[32, 32, "float", "Bt"]])), "ffa1549947e13990"),
    "feed_Bt_half": (generic(dict(SQ, stages=[[32, 32, "half", "Bt"]])), "6d893d4498ce6904"),
    "cfeed_M32": (generic(dict(M=32, N=32, K=128, c_feed=True)), "3fa20239854e840c"),
    "cfeed_M16": (generic(dict(M=16, N=64, K=64, c_feed=True)), "81efd0ef450d9379"),
}


def absent_cases():
    """Hand-off candidates whose keys are not in the table: (name, function)."""
    f = dict(a_dtype="float", b_dtype="half")
    out = [
        # grid
        ("grid 16x16x16", two_bodies(dict(M=16, N=16, K=16), dict(M=16, N=16, K=16, **f))),
        ("grid 32x64x64 -> 32x32x64", two_bodies(dict(M=32, N=64, K=64), dict(M=32, N=32, K=64, **f))),
        ("grid 64x32x64 -> 64x32x32", two_bodies(dict(M=64, N=32, K=64), dict(M=64, N=32, K=32, **f))),
        # consumer form: accumulate at a grid without a receipt
        ("mode A +acc at 32x32", two_bodies(dict(SQ), dict(M=32, N=32, K=32, accumulate=True, **f))),
        # dtype: a bfloat producer, an int8 producer
        ("bfloat producer", two_bodies(dict(SQ, a_dtype="bfloat", b_dtype="bfloat"), dict(M=32, N=32, K=32, **f))),
        ("int8 producer", two_bodies(dict(SQ, a_dtype="int8", b_dtype="int8"), dict(M=32, N=32, K=32, **f))),
        # conversion at a grid without a receipt (no memory form: both must refuse alike)
        ("half conversion at 16x16", two_bodies(dict(M=16, N=16, K=16),
                                               dict(M=16, N=16, K=16, a_dtype="half", b_dtype="half",
                                                    a_converted_from="float"))),
        ("B half conversion at 16x16", two_bodies(dict(M=16, N=16, K=64),
                                                 dict(M=16, N=16, K=16, a_dtype="half", b_dtype="half",
                                                      b_converted_from="float", feed="B"),
                                                 slots=(("A", "B", "C"), ("B", "C", "C")))),
        # role and transpose
        ("mode B fp32 at 16x16", two_bodies(dict(M=16, N=16, K=64), dict(M=16, N=16, K=16, a_dtype="half",
                                                                        b_dtype="float", feed="B"),
                                           slots=(("A", "B", "C"), ("B", "C", "C")))),
        ("mode At fp32 at 16x16", two_bodies(dict(M=16, N=16, K=64), dict(M=16, N=16, K=16, feed="At", **f))),
        ("IR transpose on the consumer", two_bodies(dict(SQ), dict(M=32, N=32, K=32, transB=True, **f))),
        ("C role at 32x32x32", two_bodies(dict(M=32, N=32, K=32), dict(M=32, N=32, K=32, accumulate=True,
                                                                       offsetA=2048, offsetB=2048),
                                         slots=(("A", "B", "C"), ("A", "B", "C")))),
        ("C role on an offset C", two_bodies(dict(M=32, N=32, K=64, offsetC=4096),
                                            dict(M=32, N=32, K=64, accumulate=True, offsetA=4096, offsetB=4096,
                                                 offsetC=4096),
                                            slots=(("A", "B", "C"), ("A", "B", "C")))),
        # locality
        ("two simdgroups", two_bodies(dict(SQ), dict(M=32, N=32, K=32, **f))),
        ("two threadgroups", two_bodies(dict(SQ, threadgroups=2), dict(M=32, N=32, K=32, threadgroups=2, **f))),
        # modifiers
        ("producer epilogue relu", two_bodies(dict(SQ, epilogue=(("relu",),)), dict(M=32, N=32, K=32, **f))),
        ("producer split fp32", two_bodies(dict(SQ, a_dtype="float", b_dtype="float", split_fp32=True),
                                          dict(M=32, N=32, K=32, **f))),
    ]
    sg = out[[n for n, _ in out].index("two simdgroups")][1]
    for op in (o for b in sg.blocks for o in b.ops if o.kind == "tensor_matmul"):
        op.attrs["simdgroups"] = 2
    # a staged A at a grid without a receipt
    out.append(("staged imageblock at 32x32", generic(dict(M=32, N=32, K=32, stages=[[16, 32, "float"]],
                                                           stage_through="imageblock"))))
    return out


class Table(unittest.TestCase):
    def test_every_entry_is_a_key_with_a_receipt(self):
        self.assertGreaterEqual(len(cc.FEED_TABLE), 20)
        for key, entry in cc.FEED_TABLE.items():
            with self.subTest(key=key):
                self.assertIsInstance(key, cc.FeedKey)
                self.assertIn(key.role, ("A", "B", "C"))
                self.assertIn(entry["disposition"], ("keep", "elide"))
                self.assertTrue(entry["receipt"].startswith("results/g17-"), entry["receipt"])
                self.assertEqual(key.locality, ((1, 1), (1, 1)))          # no split has a receipt

    def test_the_c_role_is_receipted_by_its_own_dispatch(self):
        roles = {k.role for k in cc.FEED_TABLE}
        self.assertEqual(roles, {"A", "B", "C"})
        for key in (k for k in cc.FEED_TABLE if k.role == "C"):
            self.assertIn("g17-tensor-cfeed-v1", cc.FEED_TABLE[key]["receipt"])


class PresentKeys(unittest.TestCase):
    def test_present_programs_are_byte_identical_to_before_the_table(self):
        for name, (build, want) in PRESENT.items():
            with self.subTest(name):
                self.assertEqual(compiled(build), (want, None))

    def test_present_programs_are_register_fed(self):
        # the feed changes their bytes: each differs from its own feed-off build
        for name, (build, want) in PRESENT.items():
            if name.startswith("M16_imageblock") or name.startswith("M64_imageblock"):
                continue                        # staging has no feed-off form (a named refusal)
            if name in ("chain_register", "epilogue_register"):
                continue                        # build_chain_program sets the switch from its arm name
            with self.subTest(name):
                off = compiled(build, feed=False)
                self.assertNotEqual(off[0], want)

    def test_the_feedmode_tests_programs_find_their_keys(self):
        import test_g17tensorfeedmodes as FM
        for mode in ("A", "B", "At", "Bt"):
            for conv in (False, True):
                a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16)
                c = ir.Buffer("C", 3, elem=ir.F32)
                fn = ir.Function("k", [a, b, c]); bl = ir.Builder(fn, fn.block("e"))
                bl.tensor_matmul(a, b, c, **SQ)
                t = "half" if conv else "float"
                if mode in ("A", "At"):
                    bl.tensor_matmul(c, b, c, M=32, N=32, K=32, a_dtype=t, b_dtype="half",
                                     a_converted_from="float" if conv else None, feed=mode)
                else:
                    bl.tensor_matmul(b, c, c, M=32, N=32, K=32, a_dtype="half", b_dtype=t,
                                     b_converted_from="float" if conv else None, feed=mode)
                bl.ret()
                (p, q), = tensor_pairs(fn)
                with self.subTest(mode=mode, half=conv):
                    self.assertIn(cc._feed_key(fn, p, q), cc.FEED_TABLE)
                    self.assertEqual(cc._tensor_register_feed(fn, p, q), "elide")
        self.assertTrue(FM.chain("A"))


class AbsentKeys(unittest.TestCase):
    def test_absent_keys_compile_exactly_as_with_the_feed_off(self):
        seen = set()
        for name, fn in absent_cases():
            with self.subTest(name):
                if not callable(fn):
                    keys = [cc._feed_key(fn, p, q) for p, q in tensor_pairs(fn)]
                    self.assertTrue(keys and all(k is not None for k in keys), "not a hand-off candidate")
                    self.assertTrue(all(k not in cc.FEED_TABLE for k in keys), keys)
                    seen.update(keys)
                on, off = compiled(fn), compiled(fn, feed=False)
                self.assertEqual(on, off)
        self.assertGreaterEqual(len(seen), 15)

    def test_the_fallback_is_taken_not_refused_where_a_memory_form_exists(self):
        # most absent keys have a memory bridge and must COMPILE to it
        compiled_ok = [name for name, fn in absent_cases() if compiled(fn)[0] is not None]
        for name in ("grid 16x16x16", "grid 32x64x64 -> 32x32x64", "mode A +acc at 32x32", "mode B fp32 at 16x16",
                     "mode At fp32 at 16x16", "C role at 32x32x32", "producer epilogue relu"):
            self.assertIn(name, compiled_ok)

    def test_a_non_adjacent_pair_is_no_key(self):
        fn = two_bodies(dict(SQ), dict(M=32, N=32, K=32, a_dtype="float", b_dtype="half"), between=True)
        (p, q), = tensor_pairs(fn)
        self.assertIsNone(cc._feed_key(fn, p, q))

    def test_a_disposition_other_than_the_receipted_one_is_absent(self):
        # the receipted mode-A key is "elide"; the same key writing a DIFFERENT C region must keep D,
        # which no receipt covers, so it falls back
        fn = two_bodies(dict(SQ), dict(M=32, N=32, K=32, a_dtype="float", b_dtype="half"))
        (p, q), = tensor_pairs(fn)
        self.assertEqual(cc._tensor_register_feed(fn, p, q), "elide")
        other = ir.Buffer("D", 4, elem=ir.F32)
        q.args[2] = other
        self.assertIsNone(cc._tensor_register_feed(fn, p, q))


class AccumulatorFeed(unittest.TestCase):
    def test_the_c_feed_replaces_the_c_loads_and_the_producer_stores(self):
        from collections import Counter
        on = R.build_generic_program(dict(M=32, N=32, K=128, c_feed=True)).code
        saved = cc.TENSOR_REGISTER_FEED; cc.TENSOR_REGISTER_FEED = False
        try:
            off = R.build_generic_program(dict(M=32, N=32, K=128, c_feed=True)).code
        finally:
            cc.TENSOR_REGISTER_FEED = saved
        con = Counter(i.opcode.id for i in model.decode(on, 0) if i.opcode)
        coff = Counter(i.opcode.id for i in model.decode(off, 0) if i.opcode)
        self.assertEqual(coff[12709], 8); self.assertEqual(con[12709], 0)     # the fp32 C loads
        self.assertEqual(coff[17257], 16); self.assertEqual(con[17257], 8)    # D1 no longer stored
        self.assertEqual(con[998], coff[998])                                 # the same adds

    def test_c_regs_refusals(self):
        D = {(i, j): 40 + 8 * (2 * i + j) for i in range(2) for j in range(2)}
        kw = dict(reserved=tuple(range(40, 72)))
        tlower.lower(32, 32, 32, 32, 32, 32, accumulate=True, c_regs=D, **kw)
        for bad in (dict(accumulate=False), dict(accumulate=True, a_type="float", a_regs=D),
                    dict(accumulate=True, a_type="int8", b_type="int8"),
                    dict(accumulate=True, kloop=True), dict(accumulate=True, grid=2)):
            with self.subTest(bad=sorted(bad)):
                with self.assertRaises(ValueError):
                    tlower.lower(32, 32, 32, 32, 32, 32, c_regs=D, **dict(kw, **bad))
        with self.assertRaises(ValueError):                                 # every tile, exactly
            tlower.lower(32, 32, 32, 32, 32, 32, accumulate=True, c_regs={(0, 0): 40}, **kw)

    def test_the_hardware_receipt(self):
        base = Path(ROOT) / "results/g17-tensor-cfeed-v1"
        if not (base / "M32_register.receipt.json").exists():
            self.skipTest("evidence not extracted (make evidence)")
        for shape in ("M32", "M16"):
            mem = json.loads((base / ("%s_memory.receipt.json" % shape)).read_text())
            reg = json.loads((base / ("%s_register.receipt.json" % shape)).read_text())
            self.assertEqual((mem["status"], reg["status"]), ("passed", "passed"))
            self.assertEqual({q["output_sha256"] for q in mem["queries"]}, {q["output_sha256"] for q in reg["queries"]})
            self.assertEqual(reg["files"]["program.bin"][:16], PRESENT["cfeed_" + shape][1])
        for control in ("M32_swapped", "M32_claims_no_c", "M16_swapped"):
            z = np.load(base / control / "mismatch-q1.npz")
            self.assertEqual(int((z["got_u32"] != z["expected_u32"]).sum()), 1024, control)


class FloatFeedReceipts(unittest.TestCase):
    def test_the_fp32_feed_outputs_match_the_identity_reference(self):
        # the B, At and Bt fp32 keys cite query 1's saved GPU output, checked here offline
        for mode in ("B", "At", "Bt"):
            d = Path(ROOT) / ("results/g17-tensor-feedmodes-v1/feed_%s_float" % mode)
            if not (d / "mismatch-q1.npz").exists():
                self.skipTest("evidence not extracted (make evidence)")
            spec = json.loads((d / "generic.json").read_text()); spec["feed_model"] = "identity"
            want = np.asarray(R.generic_reference(d, R.generic_spec(spec)), dtype="<f4")
            with self.subTest(mode=mode):
                self.assertTrue(np.array_equal(np.load(d / "mismatch-q1.npz")["got"].view("<u4"), want.view("<u4")))


if __name__ == "__main__":
    unittest.main()
