#!/usr/bin/env python3
"""N key blocks of online-softmax attention with register-held K/V offsets (production row P7,
docs/g17-tensorops-machine-model.md 25.114.3): tlower's B index register, the memory-stream route
that names it, the n-block program, and its reference with the frozen-advance control. Offline:
nothing here dispatches (tools/g17keyblocks.py holds the hardware arms)."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import g17tensorcommonruntime as R  # noqa: E402
from agxforge.g17 import cc, ir, model, tensor, tensorlife, tensorview, tlower  # noqa: E402

NAMES = model.registers()
# Programs dispatched in 25.114.3: 2 rows at 2, 4 and 8 blocks, 1 row at 16 blocks (KV 256), and their
# frozen controls (16 rows at 8 blocks is dispatched too and not pinned here: it takes minutes to build). The released two-block immediate program keeps its own pin in
# test_g17tensorstreaming (a31f5327225144c5).
DISPATCHED = {
    (2, 2, "register"): "12f822217860caa4",
    (2, 2, "frozen"): "af0143dde21c4807",
    (2, 4, "register"): "78ed8c28b373b0b3",
    (2, 8, "register"): "07ad54f55880a9c4",
    (2, 8, "frozen"): "410c4d83e69928d8",
    (1, 16, "register"): "e5092bec33cab0e6",       # KV 256, one decode row, head 64
    (1, 16, "frozen"): "0265bc268caf4caa",
}


def spec(n, rows=2, **kw):
    return dict(M=32, N=80, K=R.keyblock_spec_k(n), stream=rows, key_blocks=n, **kw)


def writers(code, reg):
    """(instruction index, opcode) of every instruction whose first register operand is `reg`."""
    out = []
    for n, i in enumerate(x for x in model.decode(bytes(code), 0) if x.opcode):
        regs = [NAMES.get(v, v) for k, v in i.values if k == "reg"]
        if regs and regs[0] in ("R%d" % reg, "R%dL" % reg, "R%dH" % reg):
            out.append((n, i.opcode.id))
    return out


class TheBIndexRegister(unittest.TestCase):
    """tlower's b_index: idxB += reg after the prologue, reg += advance after the body."""
    KW = dict(M=32, N=16, K=64, lda=64, ldb=16, ldc=16, reserved=tuple(range(16)) + (125,), end=False,
              binds=(0, 1, 2), offsets=(0, 0, 0))

    def body(self, **kw):
        kw = dict(self.KW, **kw)
        return tlower.lower(kw.pop("M"), kw.pop("N"), kw.pop("K"), kw.pop("lda"), kw.pop("ldb"), kw.pop("ldc"), **kw)[0]

    def test_the_register_body_is_the_plain_body_plus_three_instructions(self):
        plain = [i.raw for i in model.decode(self.body(), 0) if i.opcode]
        indexed = [i.raw for i in model.decode(self.body(b_index=(125, 1024)), 0) if i.opcode]
        self.assertEqual(len(indexed), len(plain) + 3)       # the add, then the step and the advance
        # removing the idxB add and the two trailing instructions gives the plain body back, byte for byte
        adds = [n for n, i in enumerate(x for x in model.decode(self.body(b_index=(125, 1024)), 0) if x.opcode)
                if i.opcode.id == 10282 and NAMES.get(i.values[4][1]) == "R125"]
        self.assertEqual(len(adds), 1)
        self.assertEqual(indexed[:adds[0]] + indexed[adds[0] + 1:-2], plain)

    def test_the_advance_uses_the_eight_bit_add_when_it_fits(self):
        code = self.body(b_index=(125, 255))
        self.assertEqual([op for _n, op in writers(code, 125)], [10279])
        code = self.body(b_index=(125, 256))
        self.assertEqual([op for _n, op in writers(code, 125)], [10282])
        self.assertEqual(writers(self.body(b_index=(125, 0)), 125), [])     # no advance, no writer

    def test_the_bytes_stay_clean(self):
        code = self.body(b_index=(125, 1024), index_init=(125,))
        self.assertEqual(tensorlife.released_reads(code), [])
        self.assertEqual(tensorview.hazards(tensorview.view(code)), [])

    def test_refusals(self):
        with self.assertRaises(ValueError):                     # the register is not reserved
            self.body(b_index=(124, 16))
        with self.assertRaises(ValueError):
            self.body(b_index=(125, 16), kloop=True)
        with self.assertRaises(ValueError):
            self.body(b_index=(125, 16), index_init=(124,))

    def test_transB_is_the_plain_transposed_body_plus_three_instructions(self):
        # MM 25.114.4: the K-cache read (B^T, keys x head); the register goes into the same idxB
        kw = dict(N=16, K=64, ldb=64, transB=True)
        plain = [i.raw for i in model.decode(self.body(**kw), 0) if i.opcode]
        indexed = [i.raw for i in model.decode(self.body(b_index=(125, 1024), **kw), 0) if i.opcode]
        adds = [n for n, i in enumerate(x for x in model.decode(self.body(b_index=(125, 1024), **kw), 0) if x.opcode)
                if i.opcode.id == 10282 and NAMES.get(i.values[4][1]) == "R125"]
        self.assertEqual(len(adds), 1)
        self.assertEqual(indexed[:adds[0]] + indexed[adds[0] + 1:-2], plain)


class TheStreamRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program = R.build_generic_program(spec(4))
        cls.bodies = [b"".join(bytes(m.fields["bytes"]) for m in rows) for rows in cc._TENSOR_COMPOSED_ROWS.values()]

    def test_no_key_block_address_is_an_immediate(self):
        # bodies in order: S0, PV0, S1, PV1, S2, PV2, S3, PV3. Blocks j and j+2 use the same score
        # region (S1 or S2), and their bodies differ in nothing: the key-block address is the register.
        # S0 alone also zeroes the registers
        b = self.bodies
        self.assertEqual(len(b), 8)
        self.assertEqual(b[2], b[6])
        self.assertEqual(b[3], b[7])
        self.assertNotEqual(b[3], b[5])                        # the control: PV2 reads the other S region
        self.assertNotEqual(b[0], b[4])
        self.assertEqual(len(b[0]) - len(b[4]), 16)            # two op11842 of 8 bytes

    def test_only_the_bodies_write_the_index_registers(self):
        code = bytes(self.program.code)
        for reg in cc.TENSOR_STREAM_INDEX_REGISTERS:
            ops = [op for _n, op in writers(code, reg)]
            # one zeroing, then one advance per block (512 and 2048 bytes are 256 and 1024 halves: op10282)
            self.assertEqual(ops, [11842] + [10282] * 4, "R%d" % reg)

    def test_the_frozen_control_is_the_program_without_its_advances(self):
        frozen = R.build_generic_program(spec(4, key_advance="frozen")).code
        self.assertEqual(len(self.program.code) - len(frozen), 8 * 20)   # op11842 + op10282 per body
        for reg in cc.TENSOR_STREAM_INDEX_REGISTERS:
            self.assertEqual([op for _n, op in writers(frozen, reg)], [11842])

    def test_the_bytes_are_clean_and_the_metadata_set_is_measured(self):
        self.assertEqual(tensorlife.released_reads(bytes(self.program.code)), [])
        self.assertEqual(tuple(self.program.abi()["system_registers"]), (130, 156))

    def test_a_register_offset_outside_the_stream_route_refuses(self):
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("f", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
        bl.tensor_matmul(a, b, c, M=32, N=16, K=64, offsetB_register="k", offsetB_step=2048)
        bl.ret()
        with self.assertRaises(cc.Unsupported):
            cc.compile_function(fn)
        with self.assertRaises(ir.IRError):
            bl.tensor_matmul(a, b, c, M=32, N=16, K=64, offsetB_step=2048)   # a step names a register

    def test_an_odd_byte_step_refuses(self):
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("g", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
        bl.tensor_matmul(a, b, c, M=32, N=16, K=64, offsetC=0, offsetB_register="k", offsetB_step=3)
        bl.tensor_matmul(a, b, c, M=32, N=16, K=64, offsetC=2048, offsetB_register="k", offsetB_step=3)
        bl.ret()
        with self.assertRaises(cc.Unsupported):
            cc.compile_function(fn)

    def test_the_dispatched_programs_keep_their_bytes(self):
        for (rows, n, adv), digest in DISPATCHED.items():
            with self.subTest(rows=rows, n=n, adv=adv):
                code = R.build_generic_program(spec(n, rows, key_advance=adv)).code
                self.assertEqual(hashlib.sha256(code).hexdigest()[:16], digest)


class TheReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.e = {k: v // 4 for k, v in R.STREAM_C.items()}
        cls.bundles = {}
        for n in (2, 4, 8):
            b = Path(cls.tmp.name) / ("kb%d" % n)
            R.author_generic(b, spec(n))
            cls.bundles[n] = (b, R.generic_spec(json.loads((b / "generic.json").read_text())))
        cls.legacy = Path(cls.tmp.name) / "legacy"
        R.author_generic(cls.legacy, dict(M=32, N=80, K=64, stream=2))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def O(self, flat):
        return np.asarray(flat, dtype=np.float64).ravel()[self.e["O"]: self.e["O"] + 512].reshape(32, 16)[:2]

    def test_two_blocks_are_the_released_program_exactly(self):
        b, s = self.bundles[2]
        for f in ("a.f16", "b.f16", "c.f32"):
            self.assertEqual((b / f).read_bytes(), (self.legacy / f).read_bytes(), f)
        legacy = R.stream_reference(self.legacy, R.generic_spec(json.loads((self.legacy / "generic.json").read_text())))
        self.assertTrue(np.array_equal(R.stream_reference(b, s).view("<u4"), legacy.view("<u4")))

    def test_the_exact_attention_is_inside_the_online_enclosure_and_outside_the_frozen_one(self):
        for n in (4, 8):
            b, s = self.bundles[n]
            exact = R.stream_one_shot(b, s)                    # float64 softmax over all 16 n keys
            self.assertEqual(exact.shape, (2, 16))
            for claim, inside in (("online", True), ("frozen", False)):
                lo, hi = R.stream_enclosure(b, s, mode="exact", model=claim)
                with self.subTest(n=n, claim=claim):
                    self.assertEqual(bool(np.all((exact >= self.O(lo)) & (exact <= self.O(hi)))), inside)

    def test_the_frozen_control_fails_for_certain_on_every_softmaxed_output(self):
        for n in (2, 8):
            b, s = self.bundles[n]
            ref = R.stream_reference(b, s).astype(np.float64)
            bound = R.stream_bound(b, s)
            flo, fhi = R.stream_enclosure(b, s, model="frozen")
            certain = (flo > ref + bound) | (fhi < ref - bound)
            with self.subTest(n=n):
                self.assertTrue(np.all(self.O(certain)))

    def test_the_spec_refuses_what_it_does_not_state(self):
        for kw in (dict(key_blocks=1), dict(key_blocks=17), dict(key_blocks=8, K=64),
                   dict(key_advance="frozen"), dict(key_blocks=2, stream_model="no_alpha"),
                   dict(stream_model="frozen")):
            base = dict(M=32, N=80, K=64, stream=2)
            if kw.get("key_blocks") in (1, 17):
                base["K"] = R.keyblock_spec_k(kw["key_blocks"])
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                R.generic_spec(dict(base, **kw))


class TheAttentionClassRegisterRoute(unittest.TestCase):
    """MM 25.114.4: P7's attention class with its K cache (buffer 3, transB) and V cache (buffer 3)
    read at a register-held offset (runtime key_offsets). The immediate programs keep their bytes."""
    # preregistered before dispatch (tools/g17keyblocktransb.py); the 8- and 16-block programs take
    # a minute each to build and are checked by that tool's prepare
    REGISTER = {
        ("register", 2): "fb4355b6ee71ef4f5e6a6fcfce8908bd78e05193642f1b1fc1f3b9c404d22dd7",
        ("frozen", 2): "cb08455fac1cacac1c1d87f3e7d8c31c4a374a0e8f38af7b92f75c0618077e2d",
    }

    def test_the_register_programs_and_the_immediate_one_keep_their_bytes(self):
        for (ko, n), digest in self.REGISTER.items():
            with self.subTest(ko=ko, n=n):
                code = R.build_generic_program(R.generic_spec({"attention": {"new_blocks": n, "key_offsets": ko}})).code
                self.assertEqual(hashlib.sha256(code).hexdigest(), digest)
        code = R.build_generic_program(R.generic_spec({"attention": {"new_blocks": 2}})).code
        self.assertEqual(hashlib.sha256(code).hexdigest(),
                         "918cdfd080c9f2686a40c019410a292c1ca0ec51d07374db34c5eb21ce686b13")   # 25.129 fused_n2

    def test_only_the_register_route_passes_eight_blocks(self):
        from agxforge.g17 import runtime
        n = runtime.ATTENTION_MAX_BLOCKS + 1
        with self.assertRaises(runtime.AttentionRefused):
            runtime.attention_spec({"new_blocks": n})
        self.assertEqual(runtime.attention_spec({"new_blocks": n, "key_offsets": "register"})["blocks"], n)
        with self.assertRaises(runtime.AttentionRefused):
            runtime.attention_spec({"new_blocks": runtime.ATTENTION_MAX_REGISTER_BLOCKS + 1, "key_offsets": "register"})
        with self.assertRaises(runtime.AttentionRefused):
            runtime.attention_spec({"new_blocks": 2, "key_offsets": "sideways"})
        with self.assertRaises(runtime.AttentionRefused):
            runtime.attention_spec({"new_blocks": 2, "key_offsets": "register", "phase": "project"})
        # the immediate spec is unchanged: no key_offsets key off the default
        self.assertNotIn("key_offsets", runtime.attention_spec({"new_blocks": 2}))

    def test_the_block0_claim_is_a_key_offsets_claim(self):
        with self.assertRaises(ValueError):
            R.generic_spec({"attention": {"new_blocks": 2}, "attention_model": "block0_only"})

    def test_a_transposed_register_read_is_only_the_cache_read(self):
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("t", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
        bl.tensor_matmul(a, b, c, M=32, N=16, K=64, transB=True, offsetC=0, offsetB_register="k", offsetB_step=2048)
        bl.tensor_matmul(a, b, c, M=32, N=16, K=64, transB=True, offsetC=2048, offsetB_register="k", offsetB_step=2048)
        bl.ret()
        with self.assertRaises(cc.Unsupported):
            cc.compile_function(fn)


def _loop_fn(trips=2, init="before", runtime=False, blocks_after=False):
    """The smallest counted key-block loop: QK (K cache from buffer 3, transB) and PV accumulating,
    both register-offset, inside `i + 1 < trips`. init: "before" (tensor_index_init before the loop),
    "inside" (in the loop body) or None."""
    a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("t", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
    if init == "before":
        bl.tensor_index_init("k", 8192); bl.tensor_index_init("v", 16384)
    n = bl.load(c, bl.builtin("thread_index_in_simdgroup"), type=ir.I32) if runtime else None
    z = bl.const(0, name="z")
    hdr, ex = fn.block("L"), fn.block("X")
    bl.br(hdr); bl.at(hdr)
    i = bl.phi(z, name="i")
    if init == "inside":
        bl.tensor_index_init("k", 8192); bl.tensor_index_init("v", 16384)
    bl.tensor_matmul(a, c, c, M=32, N=16, K=64, transB=True, offsetB=0, offsetC=0, offsetB_register="k",
                     offsetB_step=2048)
    bl.tensor_matmul(c, c, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=True, offsetA=0,
                     offsetB=0, offsetC=2048, offsetB_register="v", offsetB_step=512)
    nxt = bl.add(i, ir.Imm(1), name="i_next")
    ir.Builder.phi_latch(i, nxt)
    bl.br_cond(bl.cmp(nxt, n, "lt", cap=8) if runtime else bl.cmp(nxt, trips, "lt"), hdr, ex)
    bl.at(ex)
    if blocks_after:
        after = fn.block("Y")
        bl.br_cond(bl.cmp(bl.builtin("thread_index_in_simdgroup"), 4, "lt"), after, after)
        bl.at(after)
    bl.ret()
    return fn


class TheCountedKeyBlockLoop(unittest.TestCase):
    """MM 25.114.5: one QK -> row stage -> PV body in a counted loop, the K/V offsets carried by the
    register route's registers, set once before the loop. A looping kernel has rebooted this machine
    (25.116), so every refusal here is a hang class, and the latch check reads the emitted bytes."""
    # preregistered before dispatch (tools/g17keyblockloop.py and its keyblock-loop-v1 bundle directory)
    LOOP = {"loop_n2": ({"new_blocks": 2, "key_offsets": "loop"},
                        "ea810e0fad87bd58a3c5d57b84fa97f412ac1f924b7570fe92cf6bf7dfa88474")}

    @classmethod
    def setUpClass(cls):
        cls.small = bytes(cc.compile_function(_loop_fn(trips=3)).code)

    def test_the_loop_program_keeps_its_bytes_and_passes_the_latch_check(self):
        for arm, (req, digest) in self.LOOP.items():
            p = R.build_generic_program(R.generic_spec({"attention": req}))
            self.assertEqual(hashlib.sha256(bytes(p.code)).hexdigest(), digest, arm)
            got = p._tensor_loop
            self.assertEqual((got["trips"], got["counter"], got["advances"], got["hazards"], got["latch_releases"]),
                             (2, "R5", {"R125": 1, "R124": 1}, 0, 0))

    def test_the_only_registers_the_loop_carries_are_the_counter_and_the_two_offsets(self):
        v = tensorview.view(self.small)
        (first, back), = tensorview.loops(v)
        written, carried = set(), set()
        for n in range(first, back + 1):
            carried |= v[n].uses - written
            written |= v[n].defs
        chk = tensorlife.counted_loop_check(self.small, 3, carried=cc.TENSOR_STREAM_INDEX_REGISTERS)
        self.assertEqual(sorted({h // 2 for h in carried}), sorted([int(chk["counter"][1:]), 124, 125]))

    def test_the_latch_check_refuses_a_wrong_trip_count_and_a_straight_program(self):
        with self.assertRaisesRegex(ValueError, "refused: the trip compare"):
            tensorlife.counted_loop_check(self.small, 4, carried=(125, 124))
        with self.assertRaisesRegex(ValueError, "refused: 256 trips"):
            tensorlife.counted_loop_check(self.small, 256)
        straight = bytes(R.build_generic_program(R.generic_spec({"attention": {"new_blocks": 2}})).code)
        with self.assertRaisesRegex(ValueError, "refused: 0 back edges"):
            tensorlife.counted_loop_check(straight, 2)

    def test_the_latch_check_refuses_a_carried_register_rewritten_in_the_body(self):
        v = tensorview.view(self.small)
        (first, back), = tensorview.loops(v)
        other = next(min(v[n].defs) // 2 for n in range(first, back) if v[n].opcode == 11842)   # a scratch constant
        with self.assertRaisesRegex(ValueError, "not a self-add"):
            tensorlife.counted_loop_check(self.small, 3, carried=(other,))

    def test_the_released_counter_of_25_116_is_refused(self):
        """The runaway's defect, written into the bytes: the trip compare re-encoded by cc's own encoder
        with keep=False releases the counter (lifetime 16), so the next trip reads 0 and the loop never
        exits. The same encoder with keep=True reproduces the emitted compare exactly, so the patch
        changes that one field and nothing else. (cc's pre-fix liveness does not produce it for a
        one-compare constant latch: 25.116's needed the runtime bound's second, cap compare.)"""
        from agxforge.g17 import asm
        v = tensorview.view(self.small)
        (first, back), = tensorview.loops(v)
        cmp = v[back - 2]
        reg = next(x for k, x in model.decode(cmp.raw, 0).__next__().values if k == "reg" and NAMES.get(x, "").startswith("R"))
        r = int(NAMES[reg][1:])

        def enc(keep):
            return asm.encode_flag(10369, asm.encode_cmp_src(r) + asm.encode_cmp_imm(3, "lt", cc.CMP_IMM, keep=keep),
                                   cc._flag("CMP"))
        self.assertEqual(enc(True), cmp.raw)
        patched = self.small[:cmp.offset] + enc(False) + self.small[cmp.offset + len(cmp.raw):]
        self.assertEqual(len(patched), len(self.small))
        with self.assertRaisesRegex(ValueError, "not a kept `cnt < 3`"):
            tensorlife.counted_loop_check(patched, 3, carried=cc.TENSOR_STREAM_INDEX_REGISTERS)

    def test_every_other_loop_around_a_tensor_body_is_refused_by_name(self):
        # a runtime trip count is admitted when the bound is simdgroup-uniform (MM 25.144.8); this one loads at
        # thread_index_in_simdgroup, lane-varying, so every lane could leave on a different trip
        for kw, why in ((dict(runtime=True), "runtime trip count is not provably simdgroup-uniform"),
                        (dict(init=None), "need tensor_index_init before the loop"),
                        (dict(init="inside"), "inside a loop"),
                        (dict(trips=256), "8 bits|8-bit"),
                        (dict(blocks_after=True), "branches conditionally")):
            with self.subTest(**kw), self.assertRaisesRegex(cc.Unsupported, why):
                cc.compile_function(_loop_fn(**kw))

    def test_bodies_across_blocks_without_a_loop_are_no_stream(self):
        a = ir.Buffer("A", 1, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("t", [a, ir.Buffer("B", 2, elem=ir.F16), c]); bl = ir.Builder(fn, fn.block("entry"))
        bl.tensor_matmul(a, c, c, M=32, N=16, K=64, transB=True, offsetB=0, offsetC=0)
        nxt = fn.block("next"); bl.br(nxt); bl.at(nxt)
        bl.tensor_matmul(c, c, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", offsetC=2048)
        bl.ret()
        self.assertIsInstance(cc.tensor_route(fn), cc.TensorRouteRefusal)

    def test_the_class_admits_the_loop_to_its_measured_bound_and_no_mask(self):
        from agxforge.g17 import runtime
        n = runtime.ATTENTION_MAX_LOOP_BLOCKS
        self.assertEqual(runtime.attention_spec({"cache_blocks": n - 1, "rows": 1, "key_offsets": "loop"})["blocks"], n)
        with self.assertRaises(runtime.AttentionRefused):
            runtime.attention_spec({"cache_blocks": n, "rows": 1, "key_offsets": "loop"})
        with self.assertRaisesRegex(runtime.AttentionRefused, "attention_loop"):
            runtime.attention_spec({"new_blocks": 4, "causal": True, "key_offsets": "loop"})
        # decode: the query row sees every key, so no block carries a mask
        self.assertTrue(runtime.attention_spec({"cache_blocks": 3, "rows": 1, "causal": True, "q0": 63,
                                                "key_offsets": "loop"})["causal"])


if __name__ == "__main__":
    unittest.main()
