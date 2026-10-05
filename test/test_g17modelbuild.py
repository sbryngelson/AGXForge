"""The model-from-source build (MM 25.142.8): the shipped model configs, the deliver-index resolver's
exactly-one-match rule, and the ownership check that flags any bundle outside the deliver root. CPU only."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import g17modelbuild as MB  # noqa: E402


class Configs(unittest.TestCase):
    def test_every_shipped_config_names_every_role(self):
        configs = sorted((ROOT / "tools" / "models").glob("*.json"))
        self.assertTrue(configs, "no model configs shipped")
        for p in configs:
            c = json.loads(p.read_text())
            self.assertIn(c["bits"], (4, 8), p.name)
            # a qsm config's projections are g17qsm (MM 25.172): its qmv names at least the head, or every role
            if c.get("qsm"):
                self.assertTrue({"head"} <= set(c["qmv"]) <= set(MB.ROLES), p.name)
            else:
                self.assertEqual(set(c["qmv"]), set(MB.ROLES), p.name)
            for key in ("norm", "final_norm", "attn"):
                self.assertIsInstance(c[key], dict, p.name)


class Batched(unittest.TestCase):
    """The batched decode graph's configs (MM 25.144.3). Hardware: g17modelbuild --check on each batched config compares
    every sequence's tokens with a single-sequence GPU run and a single-sequence simulator run of its prompt."""

    def test_every_batch_config_is_consistent(self):
        configs = sorted((ROOT / "tools" / "models").glob("*_batch*.json"))
        self.assertTrue(configs)
        for p in configs:
            c = json.loads(p.read_text())
            B = c["batch"]
            self.assertNotIn("prefill", c, p.name)
            if c.get("qsm"):
                # the tensor-unit batched decode (MM 25.172, 25.174, 25.178): B 8, 16 or 32 (its kinds are delivered at each), the
                # norms fp16 out; a qsm head reads the fp16 final norm
                self.assertIn(B, (8, 16, 32), p.name)
                self.assertFalse(c["norm"].get("out32"), p.name)
                if c.get("qsm_head"):
                    self.assertFalse(c["final_norm"].get("out32"), p.name)
                for key in ("norm", "final_norm", "attn", "gen"):
                    self.assertEqual(c[key].get("batch"), B, (p.name, key))
                continue
            self.assertIn(B, (2, 4, 8), p.name)
            for role in ("qkv", "wo_res1", "ffn", "w2_res2"):
                self.assertEqual(c["qmv"][role].get("batch"), B, (p.name, role))
            # the head is B single-vector dispatches, or ONE batched head (MM 25.144.7) reading the out32 final norm
            if "batch" in c["qmv"]["head"]:
                self.assertEqual(c["qmv"]["head"]["batch"], B, p.name)
                self.assertTrue(c["final_norm"].get("out32"), p.name)
                self.assertNotIn("ptr", c["qmv"]["head"], p.name)
            for key in ("norm", "final_norm", "attn", "gen"):
                self.assertEqual(c[key].get("batch"), B, (p.name, key))

    def test_the_single_config_of_a_sequence(self):
        c = json.loads((ROOT / "tools" / "models" / "internlm2_q8_batch8.json").read_text())
        s = MB.single_of(c, 3)
        self.assertNotIn("batch", s)
        self.assertEqual(s["prompt_rotate"], 3)
        for v in list(s["qmv"].values()) + [s["norm"], s["final_norm"], s["attn"]]:
            self.assertFalse({"batch", "pass"} & set(v))
        self.assertEqual(s["qmv"]["w2_res2"], {"ks": 4, "lean": True, "dq": True})    # the single dq kernel
        c2 = MB.single_of(c, 3, "splitk")                                           # the control drops the order
        self.assertEqual(c2["qmv"]["w2_res2"], {"ks": 4, "lean": True})
        self.assertNotEqual(c2["name"], s["name"])

    def test_the_shared_single_run_key(self):
        # batch_check_many runs each distinct single sequence once: the key is the config without name and rotation, the
        # prompt ids and the order - so B 4's sequence 0 is B 8's when the single configs agree, a rotation by the
        # prompt length is sequence 0, and anything the tokens depend on separates keys
        c = json.loads((ROOT / "tools" / "models" / "internlm2_q4_batch8.json").read_text())
        p = [5, 6, 7]
        g = lambda r: dict(prompt_ids=p[r % 3:] + p[:r % 3])
        s0, s3 = MB.single_of(c, 0), MB.single_of(c, 3)
        self.assertNotEqual(s0["name"], s3["name"])
        self.assertEqual(MB._single_key(s0, g(0), "config"), MB._single_key(s3, g(3), "config"))
        self.assertNotEqual(MB._single_key(s0, g(0), "config"), MB._single_key(MB.single_of(c, 1), g(1), "config"))
        self.assertNotEqual(MB._single_key(s0, g(0), "config"), MB._single_key(s0, g(0), "splitk"))
        sk = MB.single_of(c, 0, "splitk")
        self.assertNotEqual(MB._single_key(s0, g(0), "config")[1], MB._single_key(sk, g(0), "config")[1])
        s4 = MB.single_of(json.loads((ROOT / "tools" / "models" / "internlm2_q4_batch4.json").read_text()), 0)
        self.assertEqual(MB._single_key(s0, g(0), "config"), MB._single_key(s4, g(0), "config"))

    def test_batch_and_prefill_refuse_together_and_the_regions_chain(self):
        import g17q4graph as G, g17gen as GEN, g17qmv as Q
        saved = (G.DELIVER_ROOT, dict(G.CONFIG), G.CAP, G.PROMPT_LEN, G.BATCH)
        try:
            with tempfile.TemporaryDirectory() as t:
                p = Path(t) / "c.json"
                p.write_text(json.dumps(dict(bits=8, batch=4, prefill={"M": 128}, qmv={})))
                with self.assertRaises(SystemExit):
                    G.configure(t, p)
                for B in (2, 4, 8):
                    p.write_text(json.dumps(dict(bits=8, batch=B, qmv={})))
                    G.configure(t, p)
                    gl = GEN.gen_batch_layout(B)
                    w2 = Q.with_batch(Q.with_residual(Q.case(2048, 8192, 8, 1, nocarrier=True)[0], "add32_to16"), B)
                    self.assertEqual((G._r_res(), G._r_gen()), (gl["R_X16"], gl["GEN"]))
                    self.assertEqual(G._r_res(), w2["RES"])            # the step writes x where the batched w2 does
        finally:
            G.DELIVER_ROOT, G.CAP, G.PROMPT_LEN, G.BATCH = saved[0], saved[2], saved[3], saved[4]
            G.CONFIG.clear(); G.CONFIG.update(saved[1])


class Resolver(unittest.TestCase):
    def setUp(self):
        import g17q4graph as G
        self.G = G
        self.saved = (G.DELIVER_ROOT, list(G._INDEX))
        self.tmp = tempfile.TemporaryDirectory()
        G.DELIVER_ROOT = Path(self.tmp.name)
        G._INDEX.clear()
        G._INDEX.extend([
            dict(kind="qmv", bits=8, role="qkv", variant={"ks": 4}, bundle="a"),
            dict(kind="qmv", bits=8, role="qkv", variant={"ks": 4, "lean": True}, bundle="b"),
            dict(kind="qmv", bits=8, role="qkv", variant={"ks": 4, "lean": True}, bundle="c")])

    def tearDown(self):
        self.G.DELIVER_ROOT = self.saved[0]
        self.G._INDEX[:] = self.saved[1]
        self.tmp.cleanup()

    def test_one_match_is_returned(self):
        self.assertEqual(self.G._pick("qmv", bits=8, role="qkv", variant={"ks": 4})["bundle"], "a")

    def test_a_variant_matches_as_a_whole_not_as_a_subset(self):
        with self.assertRaises(KeyError):          # {"ks": 4} must not also match {"ks": 4, "lean": True}
            self.G._pick("qmv", bits=8, role="qkv", variant={"ks": 2})

    def test_an_ambiguous_selection_refuses(self):
        with self.assertRaises(KeyError):
            self.G._pick("qmv", bits=8, role="qkv", variant={"ks": 4, "lean": True})


class Ownership(unittest.TestCase):
    def test_a_bundle_reached_through_a_symlink_out_of_the_root_is_flagged(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as elsewhere:
            inside = Path(root) / "bundles" / "k"
            inside.mkdir(parents=True)
            (Path(elsewhere) / "x").mkdir()
            link = Path(root) / "bundles" / "linked"
            os.symlink(Path(elsewhere) / "x", link)
            self.assertEqual(MB.outside_root([str(inside)], root), [])
            self.assertEqual(MB.outside_root([str(inside), str(link)], root), [str(link)])


class CodeOnlyAttention(unittest.TestCase):
    """kvvec and hwexp2 (MM 25.144.5) change the attention's code, not its buffers: the graph takes them through the
    butterfly form's op, and refuses one whose delivered layout differs beyond those flags."""

    def setUp(self):
        import g17q4graph as G, g17attn as A
        self.G = G
        self.saved = (G.DELIVER_ROOT, list(G._INDEX), dict(G.CONFIG), G.CAP, G.BATCH)
        self.tmp = tempfile.TemporaryDirectory()
        G.DELIVER_ROOT, G.CAP, G.BATCH = Path(self.tmp.name), 2048, 1
        lay = A.with_attn32(A.with_bfly_merge(A.with_wide(A.with_fused_merge(A.with_rope_tables(
            A.attn_rope_layout(cap=2048))))))
        lay = json.loads(json.dumps(lay))                  # as the index stores it

        def entry(name, variant, **flags):
            return dict(kind="attn", cap=2048, variant=variant, bundle="bundles/" + name, threadgroups=16,
                        threads_per_group=1024, ATTN=lay["ATTN"], region3_bytes=lay["region3_bytes"],
                        recipe=dict(layout=dict(lay, **flags)))
        base = {"widebf": True, "attn32": True}
        moved = entry("moved", dict(base, kvvec=True, hwexp2=True), kvvec=True, hw_exp2=True)
        moved["recipe"]["layout"]["KOFF"] += 128           # same ATTN and region3, a different K offset
        G._INDEX[:] = [entry("base", base), entry("kvvec", dict(base, kvvec=True), kvvec=True),
                       entry("hwexp2", dict(base, kvvec=True, hwexp2=True), kvvec=True, hw_exp2=True)]
        self.moved = moved

    def tearDown(self):
        G = self.G
        G.DELIVER_ROOT, G._INDEX[:], G.CAP, G.BATCH = self.saved[0], self.saved[1], self.saved[3], self.saved[4]
        G.CONFIG.clear(); G.CONFIG.update(self.saved[2])
        self.tmp.cleanup()

    def _op(self, variant):
        self.G.CONFIG.clear(); self.G.CONFIG.update(attn=variant)
        return self.G._fused_attn_op(0)

    def test_each_variant_selects_its_own_bundle(self):
        for name, extra in (("base", {}), ("kvvec", {"kvvec": True}), ("hwexp2", {"kvvec": True, "hwexp2": True})):
            op = self._op(dict({"widebf": True, "attn32": True}, **extra))
            self.assertEqual(Path(op.bundle).name, name)

    def test_a_layout_that_differs_beyond_the_code_flags_refuses(self):
        self.G._INDEX[2] = self.moved
        with self.assertRaises(ValueError):
            self._op({"widebf": True, "attn32": True, "kvvec": True, "hwexp2": True})

    def test_a_batched_config_refuses_the_single_sequence_forms(self):
        self.G.BATCH = 2
        e = dict(self.G._INDEX[1], variant={"widebf": True, "attn32": True, "kvvec": True, "batch": 2})
        self.G._INDEX[:] = [e]
        with self.assertRaises(ValueError):
            self._op(e["variant"])


if __name__ == "__main__":
    unittest.main()
