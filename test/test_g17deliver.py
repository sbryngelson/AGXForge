"""tools/g17deliver.py: every decode bundle from repository sources (MM 25.141.15). Compile-only.

- The index CONTRACT (g17deliver.SCHEMA) holds every field Piece A's graph selects and asserts by, and every field
  the earlier hand-built index files carried. Those are written out here as literals, so the tool cannot drift from
  what the graph reads without this test saying so.
- Entries the builders actually produce satisfy the contract.
- Every recipe rebuilds deterministically: the same program twice, and the same program again from the recipe's
  JSON layout, which is what `check` rebuilds from.
- `check` refuses an index whose recorded program no longer rebuilds to its sha.
The hardware verification itself (a 0x7f sentinel over every output region, bit-exact against each kind's
reference) runs in `build`, which dispatches; a negative control (a reference off by one ulp) refused delivery."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

# Piece A's selection and assertion fields (the graph's side of the contract)
GRAPH_FIELDS = {
    "qmv": ("kind", "bits", "role", "variant", "bundle", "threadgroups", "threads_per_group", "base", "slot_map",
            "program_sha256", "recipe", "S", "B", "X", "OUT", "RES"),
    "lm_head": ("kind", "bits", "role", "variant", "bundle", "threadgroups", "threads_per_group", "base", "slot_map",
                "program_sha256", "recipe", "S", "B", "X", "OUT"),
    "norm": ("kind", "role", "variant", "bundle", "threadgroups", "threads_per_group", "base", "slot_map",
             "program_sha256", "recipe", "X", "G", "OUT"),
    "attn": ("kind", "cap", "variant", "bundle", "threadgroups", "threads_per_group", "base", "slot_map",
             "program_sha256", "recipe", "ATTN", "region3_bytes", "COST", "SINT", "rope_bytes"),
    "gen_argmax": ("kind", "cap", "bundle", "threadgroups", "threads_per_group", "base", "slot_map", "program_sha256",
                   "recipe", "LOG", "region_bytes"),
    "gen_step": ("kind", "cap", "bundle", "threadgroups", "threads_per_group", "base", "slot_map", "program_sha256",
                 "recipe", "LOG", "region_bytes"),
}
# the fields the hand-built q4/q8 v2, attention and cap-2048 index files carried
LEGACY_FIELDS = {
    "qmv": ("B", "OUT", "RES", "S", "W", "base", "bits", "op", "out_dtype", "program_sha256", "threadgroups",
            "threads_per_group", "x_dtype", "x_offset"),
    "norm": ("G", "OUT", "X", "base", "in_dtype", "op", "out_dtype", "program_sha256", "slot_map", "threadgroups",
             "threads_per_group"),
    "attn": ("ATTN", "COST", "KOFF", "VOFF", "SINT", "base", "program_sha256", "region3_bytes", "rope_bytes",
             "threadgroups", "threads_per_group", "cap"),
    "gen_argmax": ("GEN", "LOG", "PAIRS", "R_X16", "base", "cap", "program_sha256", "region_bytes", "threadgroups",
                   "threads_per_group"),
    "gen_step": ("GEN", "LOG", "PAIRS", "R_X16", "base", "cap", "program_sha256", "region_bytes", "threadgroups",
                 "threads_per_group"),
}


def sha(p):
    return hashlib.sha256(p.code).hexdigest()


class Deliver(unittest.TestCase):
    def test_the_contract_holds_the_graph_and_legacy_fields(self):
        import g17deliver as D
        for kind, fields in list(GRAPH_FIELDS.items()) + list(LEGACY_FIELDS.items()):
            self.assertEqual([f for f in fields if f not in D.SCHEMA[kind]], [], kind)

    def test_built_entries_satisfy_the_contract(self):
        import g17deliver as D
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            jobs = D.build_norm(t, "attn_norm", "half", True, True) + \
                D.build_qmv(t, 4, "qkv", dict(ks=2, lean=True, ptr=True))
            for j in jobs:
                e = D.deliver(j, t, t)
                self.assertEqual(D.validate(e), [], e["name"])
                self.assertTrue((t / e["bundle"] / "program.bin").exists())
                self.assertEqual(hashlib.sha256((t / e["bundle"] / "program.bin").read_bytes()).hexdigest(),
                                 e["program_sha256"])

    def test_recipes_rebuild_deterministically(self):
        import g17attn as A
        import g17decodeops as O
        import g17deliver as D
        import g17gen as G
        import g17qmv as Q
        cases = [("g17qmv.build_qmv2", D.qmv_layout(4, "qkv", dict(ks=2, lean=True, ptr=True))[0]),
                 ("g17qmv.build_qmv2", D.qmv_layout(8, "w2_res2", dict(ks=4))[0]),
                 ("g17qmv.build_qmv2", D.qmv_layout(4, "ffn", {})[0]),
                 ("g17decodeops.build_rmsnorm_wide", D.norm_layout("float", True, True)),
                 ("g17attn.build_attn_split_rope", D.attn_layout(272)),
                 ("g17gen.build_pass1", G.gen_layout(cap=272)),
                 ("g17gen.build_gen", G.gen_layout(cap=2048))]
        direct = {"g17qmv.build_qmv2": Q.build_qmv2, "g17decodeops.build_rmsnorm_wide": lambda l: O.build_rmsnorm_wide(l, D.EPS),
                  "g17attn.build_attn_split_rope": A.build_attn_split_rope, "g17gen.build_pass1": G.build_pass1,
                  "g17gen.build_gen": G.build_gen}
        for builder, lay in cases:
            one, two = sha(direct[builder](lay)), sha(direct[builder](lay))
            self.assertEqual(one, two, builder)
            again = sha(D.rebuild(dict(recipe=dict(builder=builder, layout=D.jsonable(lay)))))
            self.assertEqual(one, again, "%s from its JSON recipe" % builder)

    def test_check_refuses_a_changed_program(self):
        import g17deliver as D
        lay = D.norm_layout("half", True, True)
        good = dict(name="n", bundle="n", program_sha256=sha(D.rebuild(dict(recipe=dict(
            builder="g17decodeops.build_rmsnorm_wide", layout=D.jsonable(lay))))),
            recipe=dict(builder="g17decodeops.build_rmsnorm_wide", layout=D.jsonable(lay)))
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "index.json"
            p.write_text(json.dumps([good]))
            D.main(["check", str(p)])                     # passes
            p.write_text(json.dumps([dict(good, program_sha256="0" * 64)]))
            with self.assertRaises(SystemExit):
                D.main(["check", str(p)])



class TheHwExp2EnclosureCheck(unittest.TestCase):
    """prefill_mma_hwexp2 is checked at each ROW's scale, not per-element ulps (MM 25.144.2): an output near zero is a
    cancellation, where even the bit-exact kernel is several ulps from true-exp2 softmax."""

    def test_the_error_is_scaled_by_the_row_not_the_element(self):
        import numpy as np
        import g17deliver as G
        want = np.array([[[1.0, 0.001]]], np.float16)                      # one row: scale 1.0
        near_zero_off = np.array([[[1.0, 0.0015]]], np.float16).view(np.uint16)
        e = G.row_scaled_error(near_zero_off, want, 1, 1, 2)
        self.assertLess(float(e.max()), G.ENCLOSURE_ROW_BOUND)             # 50 percent of a tiny element, 5e-4 of the row
        big_off = np.array([[[1.002, 0.001]]], np.float16).view(np.uint16)
        self.assertGreater(float(G.row_scaled_error(big_off, want, 1, 1, 2).max()), G.ENCLOSURE_ROW_BOUND)

    def test_the_bound_sits_between_the_measured_kernels_and_the_control(self):
        import g17deliver as G
        self.assertGreater(G.ENCLOSURE_ROW_BOUND, 8.7e-4)                  # hw_exp2's worst, cap-2048 buckets
        self.assertLess(G.ENCLOSURE_CONTROL_MIN, 1.12)                     # the wrong base's smallest max


if __name__ == "__main__":
    unittest.main()
