"""Long context: the token log and the RoPE tables share the GEN region, and at cap 2048 they must not overlap.

g17gen.gen_layout puts the q0 word at GEN and the token log int32 [cap] at GEN + 4, so the log ends at byte 4 + 4 cap
of the region. g17attn.with_rope_tables put the cos table at a FIXED COST = 2048, which held only for cap <= 511: at
cap 2048 the log ends at 8196 and log[cap - 1] would have overwritten cos[0]. COST is now max(2048, align(4 + 4 cap)),
so every cap <= 511 layout (the delivered cap-272 set) is byte-unchanged.

Checked here by layout arithmetic (compile-only), by the cap-272 program's unchanged bytes, and by the recorded
hardware run (MM 25.141.14): at cap 2048 the gen step wrote log[cap - 1] with a cos[0] sentinel intact past it, and
the attention read cos correctly with the whole log filled with junk, at q0 0/271/1024/2047."""
import hashlib, json, os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from _evidence import require

# the delivered cap-272 wide butterfly-merge attention (attn_widebf_s32_attn32, MM 25.141.14)
WIDEBF_272_SHA256 = "b8b734a3f064c8aa201bc2928aacb6b79b7be66052ce8189e75ec3345abc3080"


def widebf(cap):
    import g17attn as A
    return A.with_attn32(A.with_bfly_merge(A.with_wide(A.with_fused_merge(A.with_rope_tables(A.attn_rope_layout(cap=cap))))))


class TheLogAndTheTablesDoNotOverlap(unittest.TestCase):
    def test_the_log_ends_before_cos_at_every_cap(self):
        import g17gen as G
        for cap in (16, 272, 511, 512, 1024, 2048, 4096):
            lay, gen = widebf(cap), G.gen_layout(cap=cap)
            log_end = gen["LOG"] - gen["GEN"] + 4 * cap              # bytes into the region past log[cap - 1]
            self.assertEqual(gen["region_bytes"] - gen["GEN"], log_end, cap)
            self.assertLessEqual(log_end, lay["COST"], cap)
            self.assertEqual(lay["COST"] % 256, 0, cap)
            self.assertGreaterEqual(lay["SINT"], lay["COST"] + cap * lay["head_dim"] // 2 * 4, cap)

    def test_the_old_fixed_offset_would_have_overlapped(self):
        # the check can fail: at cap 2048 the log reaches past the old COST of 2048
        self.assertGreater(4 + 4 * 2048, 2048)
        self.assertEqual(widebf(2048)["COST"], 8448)

    def test_caps_up_to_511_keep_the_delivered_layout(self):
        for cap in (16, 272, 511):
            self.assertEqual(widebf(cap)["COST"], 2048, cap)
        import g17attn as A
        self.assertEqual(hashlib.sha256(A.build_attn_split_rope(widebf(272)).code).hexdigest(), WIDEBF_272_SHA256)

    def test_the_cap_2048_loop_bound_fits_the_compare(self):
        # 2048 keys over 32 simdgroups: the latch compares against 64, inside the 8-bit compare bound
        self.assertEqual(widebf(2048)["trips_cap"], 64)


class TheHardwareRunRecordedIt(unittest.TestCase):
    def test_neither_region_clobbered_the_other(self):
        path, = require(os.path.join("results", "g17-cap2048-v1", "verify.json"),
                        invariant="the cap-2048 hardware run: log[cap-1] written with cos[0] intact, attention exact over a junk log")
        rec = json.load(open(path))
        self.assertEqual(rec["layout"]["COST"], 8448)
        gen = rec["gen_step_cap2048"]["verified"]
        for q0 in ("0", "271", "1024", "2047"):
            for kind in ("chosen", "forced"):
                self.assertIn('"%s_%s": {' % (q0, kind), gen)
        self.assertNotIn('"cos0_intact": false', gen)
        self.assertNotIn('"log_ok": false', gen)
        attn = rec["attn_widebf_s32_attn32_cap2048"]["verified"]
        self.assertIn("token log filled with 0x7f", attn)
        for q0 in ("0", "271", "1024", "2047"):
            self.assertIn('"%s": {"attn_differing": 0, "new_k_row_differing": 0, "nan_left": 0}' % q0, attn)


if __name__ == "__main__":
    unittest.main()
