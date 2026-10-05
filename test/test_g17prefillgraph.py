"""The prefill section's kernel resolution (MM 25.142.10), CPU only: every kernel comes from the deliver index, a
missing kind is refused by name, and decode's attention region grows to exactly the prefill kernels' region."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import g17prefillgraph as PG  # noqa: E402

M, CAP, BITS = 128, 272, 8


def _index(drop=None):
    ix = []
    for role in ("attn_norm", "ffn_norm"):
        ix.append(dict(kind="norm", role=role, variant={"out32": False, "seed": False, "batch": M}))
    ix.append(dict(kind="prefill_attn", role="append", cap=CAP, variant={"scalar": True}))
    ix.append(dict(kind="prefill_attn", role="attn", cap=CAP, variant={"scalar": True, "out16": True},
                   prefill_region3_bytes=4600064))
    for proj in ("qkv", "wo", "w1", "w2"):
        ix.append(dict(kind="qmm_dequant", bits=BITS, role=proj, variant={}))
        ix.append(dict(kind="qmm", bits=BITS, role=proj, variant={"M": M}))
    for kind in ("residual_rows", "swiglu_rows", "fold_rows"):
        ix.append(dict(kind=kind, variant={"rows": M}))
    return [e for e in ix if not (drop and e["kind"] == drop[0] and e.get("role") == drop[1])]


def _pick(ix):
    def pick(kind, **match):
        hits = [e for e in ix if e["kind"] == kind and all(e.get(k) == v for k, v in match.items())]
        if len(hits) != 1:
            raise KeyError("%d entries" % len(hits))
        return hits[0]
    return pick


class Kernels(unittest.TestCase):
    def test_a_complete_index_resolves_every_prefill_kernel(self):
        k = PG.kernels(_pick(_index()), BITS, CAP, M)
        self.assertEqual(len(k), 2 + 2 + 8 + 3)

    def test_a_missing_kind_is_refused_by_name(self):
        with self.assertRaises(PG.Missing) as cm:
            PG.kernels(_pick(_index(drop=("qmm", "w2"))), BITS, CAP, M)
        self.assertIn("qmm", str(cm.exception))
        self.assertIn("w2", str(cm.exception))

    def test_decode_reserves_exactly_the_prefill_kernels_region(self):
        self.assertEqual(PG.region3_bytes(_pick(_index()), CAP), 4600064)



class TheRegisterORoute(unittest.TestCase):
    """{"attn": "mma", "rego": true} (MM 25.142.11): M2's register-O attention with the causal skip, which took a
    1,024-token prefill's attention from 118 to 15 ms at M 512. It pairs only with the register-O append, and each chunk
    start needs its own bucket - M 256 chunks of a 1,024-token prompt start at 0, 256, 512 and 768."""
    RM, RCAP = 256, 2048

    def _index(self, opt_for=(0,)):
        ix = [e for e in _index() if e["kind"] not in ("prefill_attn",)]
        ix = [dict(e, variant=dict(e["variant"], batch=self.RM)) if e["kind"] == "norm" else
              dict(e, variant={"M": self.RM}) if e["kind"] == "qmm" else
              dict(e, variant={"rows": self.RM}) if e["kind"].endswith("_rows") else e for e in ix]
        ix.append(dict(kind="prefill_mma", role="append", cap=self.RCAP, variant={"mma": True, "M": self.RM},
                       mma_region3_bytes=111))
        ix.append(dict(kind="prefill_mma", role="append", cap=self.RCAP, variant={"mma": True, "M": self.RM, "rego": True},
                       mma_region3_bytes=222))
        for c in range(4):
            base = {"mma": True, "M": self.RM, "p0_block": c * self.RM // 16, "out16": True, "skip": True,
                    "rego": True, "sg": 1}
            ix.append(dict(kind="prefill_mma", role="attn", cap=self.RCAP, variant=base, tag="plain%d" % c))
            if c in opt_for:
                ix.append(dict(kind="prefill_mma", role="attn", cap=self.RCAP, variant=dict(base, **PG.REGO_OPT),
                               tag="opt%d" % c))
        return ix

    def test_it_resolves_the_register_o_append_and_a_skipping_attention_per_chunk(self):
        k = PG.kernels(_pick(self._index()), BITS, self.RCAP, self.RM, route="mma", chunks=4, rego=True)
        self.assertTrue(k["append"]["variant"].get("rego"))
        self.assertEqual([a["variant"]["p0_block"] for a in k["attn_chunks"]], [0, 16, 32, 48])
        self.assertTrue(all(a["variant"]["skip"] and a["variant"]["rego"] for a in k["attn_chunks"]))

    def test_it_prefers_the_code_size_variant_and_falls_back_to_the_plain_one(self):
        k = PG.kernels(_pick(self._index(opt_for=(0,))), BITS, self.RCAP, self.RM, route="mma", chunks=4, rego=True)
        self.assertEqual([a["tag"] for a in k["attn_chunks"]], ["opt0", "plain1", "plain2", "plain3"])

    def test_a_missing_chunk_start_is_refused_by_name(self):
        ix = [e for e in self._index() if e.get("tag") not in ("plain3",)]
        with self.assertRaises(PG.Missing) as cm:
            PG.kernels(_pick(ix), BITS, self.RCAP, self.RM, route="mma", chunks=4, rego=True)
        self.assertIn("48", str(cm.exception))

    def test_hw_exp2_selects_the_hardware_exp2_variant_and_refuses_when_it_is_absent(self):
        """{"hw_exp2": true} (MM 25.144.2, PR #289): the enclosure-checked op1272 row stage, built with the code-size
        options. Without those entries in the index the route is refused by name, never silently exact."""
        ix = self._index(opt_for=(0, 1, 2, 3))
        for e in list(ix):
            if e.get("tag", "").startswith("opt"):
                ix.append(dict(e, variant=dict(e["variant"], hw_exp2=True), tag="hw" + e["tag"][3:]))
        k = PG.kernels(_pick(ix), BITS, self.RCAP, self.RM, route="mma", chunks=4, rego=True, hw_exp2=True)
        self.assertEqual([a["tag"] for a in k["attn_chunks"]], ["hw0", "hw1", "hw2", "hw3"])
        with self.assertRaises(PG.Missing):
            PG.kernels(_pick(self._index()), BITS, self.RCAP, self.RM, route="mma", chunks=4, rego=True, hw_exp2=True)

    def test_gemm_slices_take_the_row_kernels_at_gemm_m_and_the_attention_at_m(self):
        """{"M": 1024, "gemm_M": 256} (MM 25.142.11): the norms, projections and row kernels stay at 256 rows (M1's
        2 x 2 qmm exists only there) while the append and the attention see the whole 1,024-row chunk, one dispatch."""
        ix = self._index()
        ix.append(dict(kind="prefill_mma", role="append", cap=self.RCAP, variant={"mma": True, "M": 1024, "rego": True},
                       tag="append1024"))
        ix.append(dict(kind="prefill_mma", role="attn", cap=self.RCAP, tag="attn1024",
                       variant=dict({"mma": True, "M": 1024, "p0_block": 0, "out16": True, "skip": True, "rego": True,
                                     "sg": 1}, **PG.REGO_OPT)))
        k = PG.kernels(_pick(ix), BITS, self.RCAP, 1024, route="mma", chunks=1, rego=True, gemm_M=self.RM)
        self.assertEqual((k["append"]["tag"], [a["tag"] for a in k["attn_chunks"]]), ("append1024", ["attn1024"]))
        self.assertEqual({k["mm_" + p]["variant"]["M"] for p in ("qkv", "wo", "w1", "w2")}, {self.RM})
        self.assertEqual({k[r]["variant"]["rows"] for r in ("residual", "swiglu", "fold")}, {self.RM})
        self.assertEqual({k[r]["variant"]["batch"] for r in ("attn_norm", "ffn_norm")}, {self.RM})
        with self.assertRaises(PG.Missing):                     # without gemm_M the projections are sought at 1,024
            PG.kernels(_pick(ix), BITS, self.RCAP, 1024, route="mma", chunks=1, rego=True)

    def test_a_gemm_m_that_does_not_divide_the_chunk_is_refused(self):
        for gm in (384, 2048):
            with self.assertRaises(ValueError):
                PG.section(None, BITS, self.RCAP, {"M": 1024, "gemm_M": gm, "attn": "mma", "rego": True}, [1] * 8,
                           {}, ".", ".", None, 0)

    def test_region3_is_sized_by_the_register_o_append_not_the_memory_o_one(self):
        pick = _pick(self._index())
        self.assertEqual(PG.region3_bytes(pick, self.RCAP, {"attn": "mma", "M": self.RM, "rego": True}), 222)
        self.assertEqual(PG.region3_bytes(pick, self.RCAP, {"attn": "mma", "M": self.RM}), 111)


if __name__ == "__main__":
    unittest.main()
