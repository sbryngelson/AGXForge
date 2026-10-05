"""The simulator's references, vectorised and threaded, are BIT-IDENTICAL to the forms they replaced.

g17q4graph's simulate (the prompt fed token by token through the exact references) took ~24 s per token at q8 (an M=128
prompt: ~50 min), and g17prefillgraph.reference ~8.7 s per layer. The hot references now do the same fp32 operations in
the same order, but over more of the data at once:

- qmv2_reference_fast forms every trip's x sum and dot chain at once (they restart per trip) and runs row blocks on a
  thread pool; the per-trip form is kept as _qmv2_reference_fast_trips, the oracle here;
- _fma32v's TwoSum is written in place (_fma32v_ref is the old text);
- attn_reference's one-key-per-trip form runs every (head, slice) at once (_attn_wide_fast), falling back to the
  scalar path when a value is not finite; tensorreduce.butterfly_array is butterfly over an array's last axis;
- rmsnorm_wide_rows_reference is rmsnorm_wide_reference over rows;
- _gemm_mma_fast runs its column blocks on a thread pool; g17q4graph._unpack returns uint8 fields.

Each is compared bit for bit with the old form over random and adversarial inputs (midpoint fma cases, signed zeros,
subnormals, overflow, non-finite values that force the fallback), and each comparison has a control that it can fail."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np

F32 = np.float32


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    w = np.uint16 if a.dtype == np.float16 else np.uint32
    return a.shape == b.shape and np.array_equal(a.view(w), b.view(w))


class QmvReference(unittest.TestCase):
    def test_all_trips_at_once_equals_per_trip(self):
        import g17qmv as Q
        rng = np.random.default_rng(0)
        for bits in (4, 8):
            lay0, _, _, s16, b16, q = Q.case(128, 2048, bits, 1, nocarrier=True)
            for coal in (True, False):
                for extra in ({}, dict(chains=True), dict(epi_fma=True), dict(chains=True, epi_fma=True, pool_masks=True)):
                    for wpt in (1, 2, 4):
                        lay = dict(lay0, coalesced=coal, wpt=wpt, **extra)
                        for kind in ("normal", "huge", "tiny", "mixed"):
                            x = rng.standard_normal(2048).astype(F32)
                            if kind == "huge":
                                x *= F32(1e30)
                            elif kind == "tiny":
                                x *= F32(1e-38)
                            elif kind == "mixed":
                                x[::7] *= F32(1e20); x[::5] *= F32(1e-30); x[3] = 0.0; x[4] = -0.0
                            with np.errstate(all="ignore"):
                                a = Q.qmv2_reference_fast(lay, x, q, s16, b16)
                                b = Q._qmv2_reference_fast_trips(lay, x, q, s16, b16)
                            self.assertTrue(same(a, b), (bits, coal, extra, wpt, kind))
        # the control: the check sees one trip's operations out of order
        lay = dict(lay0, coalesced=True, wpt=2)
        x = rng.standard_normal(2048).astype(F32)
        wrong = Q._qmv2_reference_fast_trips(dict(lay, epi_fma=True), x, q, s16, b16)
        self.assertFalse(same(Q.qmv2_reference_fast(lay, x, q, s16, b16), wrong))

    def test_threads_do_not_change_values(self):
        import g17qmv as Q
        lay, x, _, s16, b16, q = Q.case(1024, 2048, 4, 1, nocarrier=True)
        lay = dict(lay, coalesced=True, wpt=2)
        old = os.environ.get("G17_REF_THREADS")
        try:
            os.environ["G17_REF_THREADS"] = "1"
            one = Q.qmv2_reference_fast(lay, x, q, s16, b16)
            os.environ["G17_REF_THREADS"] = "8"
            many = Q.qmv2_reference_fast(lay, x, q, s16, b16)
        finally:
            os.environ.pop("G17_REF_THREADS", None) if old is None else os.environ.__setitem__("G17_REF_THREADS", old)
        self.assertTrue(same(one, many))


class Fma32vInPlace(unittest.TestCase):
    def test_equals_the_old_text(self):
        import g17qmv as Q
        rng = np.random.default_rng(7)
        sets = []
        for _ in range(8):
            n = 100000
            sets.append((rng.integers(0, 256, n).astype(F32),
                         (rng.standard_normal(n) * np.exp2(rng.integers(-140, 120, n))).astype(F32),
                         (rng.standard_normal(n) * np.exp2(rng.integers(-150, 127, n))).astype(F32)))
        a = (rng.integers(1 << 12, 1 << 13, 20000) | 1).astype(F32)            # exact fp32 midpoints, tiny c decides
        b = (rng.integers(1 << 11, 1 << 12, 20000) | 1).astype(F32)
        c = (rng.choice([1, -1], a.size) * 2.0 ** -30).astype(F32)
        sets.append((a, b, c))
        z = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-45, -1e-45, 3.4e38, -3.4e38], F32)
        A, B, C = np.meshgrid(z, z, z)
        sets.append((A.ravel(), B.ravel(), C.ravel()))
        sets.append((F32(3), F32(0.1), F32(0.2)))
        for a, b, c in sets:
            with np.errstate(all="ignore"):
                self.assertTrue(same(Q._fma32v(a, b, c), Q._fma32v_ref(a, b, c)))
        # the control: plain double rounding gets the midpoint set wrong
        a, b, c = sets[8]
        naive = (a.astype(np.float64) * b + c).astype(F32)
        self.assertFalse(same(naive, Q._fma32v(a, b, c)))


class Attention(unittest.TestCase):
    def test_every_head_and_slice_at_once_equals_the_scalar_path(self):
        import g17attn as A
        rng = np.random.default_rng(3)
        for S, bm in ((32, True), (32, False), (8, False)):
            lay = dict(heads=16, kv_heads=8, head_dim=128, cap=100, slices=S, PW=132, wide=True, bfly_merge=bm)
            for q0 in (0, 1, 31, 33, 99, 150):
                for kind in ("normal", "big", "spread", "overflow"):
                    q = (rng.standard_normal((16, 128)) * 0.1).astype(np.float16).astype(F32)
                    K = rng.standard_normal((8, 100, 128)).astype(np.float16).astype(F32)
                    V = rng.standard_normal((8, 100, 128)).astype(np.float16).astype(F32)
                    if kind == "big":
                        q *= F32(30)
                    elif kind == "spread":
                        K[:, ::3] *= F32(50); V[:, ::5] *= F32(1e-3)
                    elif kind == "overflow":                  # a non-finite score: the fast path must hand back
                        q[3] = F32(1e30); K[1, :] = F32(1e20)
                    with np.errstate(all="ignore"):
                        try:
                            b = A.attn_reference(lay, q, K, V, q0, partials=True, _scalar=True)
                        except (OverflowError, ValueError) as e:     # the scalar path refuses: so must the default
                            with self.assertRaises(type(e)):
                                A.attn_reference(lay, q, K, V, q0, partials=True)
                            continue
                        a = A.attn_reference(lay, q, K, V, q0, partials=True)
                    self.assertTrue(same(a[0], b[0]) and same(a[1], b[1]), (S, bm, q0, kind))
        # the control: one different key position changes the output
        q = (rng.standard_normal((16, 128)) * 0.1).astype(np.float16).astype(F32)
        K = rng.standard_normal((8, 100, 128)).astype(np.float16).astype(F32)
        V = rng.standard_normal((8, 100, 128)).astype(np.float16).astype(F32)
        lay = dict(heads=16, kv_heads=8, head_dim=128, cap=100, slices=32, PW=132, wide=True, bfly_merge=True)
        self.assertFalse(same(A.attn_reference(lay, q, K, V, 40), A.attn_reference(lay, q, K, V, 41, _scalar=True)))

    def test_butterfly_array_equals_butterfly(self):
        from agxforge.g17 import tensorreduce as TR
        rng = np.random.default_rng(5)
        v = (rng.standard_normal((400, 32)) * np.exp2(rng.integers(-60, 60, (400, 32)))).astype(F32)
        v[0, :4] = [0.0, -0.0, 1e-45, -1e-45]
        v[1, 7] = np.nan
        for masks in (TR.ROW_BUTTERFLY_MASKS, TR.COLUMN_BUTTERFLY_MASKS):
            for op in ("sum", "max"):
                got = TR.butterfly_array(v, masks, op)
                for r in range(v.shape[0]):
                    want = np.array(TR.butterfly([float(x) for x in v[r]], masks, op), F32)
                    self.assertTrue(same(got[r], want) or (np.isnan(got[r]) == np.isnan(want)).all() and
                                    same(np.where(np.isnan(got[r]), 0, got[r]), np.where(np.isnan(want), 0, want)),
                                    (masks, op, r))
        # the control: the other stage order is a different sum on some rows
        a = TR.butterfly_array(TR.butterfly_array(v[2:], TR.ROW_BUTTERFLY_MASKS), TR.COLUMN_BUTTERFLY_MASKS)
        b = TR.butterfly_array(TR.butterfly_array(v[2:], TR.COLUMN_BUTTERFLY_MASKS), TR.ROW_BUTTERFLY_MASKS)
        self.assertFalse(same(a, b))


class RowsNorm(unittest.TestCase):
    def test_rows_equal_per_row(self):
        import g17decodeops as O, g17realmodel as M
        rng = np.random.default_rng(9)
        spec = M.layer_spec(0)
        V = (rng.standard_normal((40, 2048)) * 3).astype(np.float16).astype(F32)
        V[1] *= F32(1e-20); V[2, ::9] = F32(0.0); V[3, :5] = F32(1e18)             # tiny, zeros, a large sum
        g = rng.standard_normal(2048).astype(np.float16).astype(F32)
        bad = V[:2].copy(); bad[1] = V[0] * F32(1e25)                                  # squares overflow: both forms raise
        with np.errstate(all="ignore"):
            with self.assertRaises(ValueError):
                O.rmsnorm_wide_reference(bad[1], g, spec)
            with self.assertRaises(ValueError):
                O.rmsnorm_wide_rows_reference(bad, g, spec)
            bad[1] = V[0]; bad[1, :5] = F32(1e19)               # finite squares whose lane sum overflows: both raise
            with self.assertRaises(OverflowError):
                O.rmsnorm_wide_reference(bad[1], g, spec)
            with self.assertRaises(OverflowError):
                O.rmsnorm_wide_rows_reference(bad, g, spec)
        for seed in (False, True):
            with np.errstate(all="ignore"):
                rows = O.rmsnorm_wide_rows_reference(V, g, spec, seed=seed)
                for i in range(V.shape[0]):
                    self.assertTrue(same(rows[i], O.rmsnorm_wide_reference(V[i], g, spec, seed=seed)), (seed, i))
        v0 = V[0].copy(); v0[0] += F32(1.0)                                         # the control: one input moved
        self.assertFalse(same(rows[0], O.rmsnorm_wide_reference(v0, g, spec, seed=True)))


class GemmAndFields(unittest.TestCase):
    def test_threaded_gemm_equals_in_order(self):
        import g17tensorcommonruntime as TCR
        rng = np.random.default_rng(2)
        A = rng.standard_normal((64, 256)).astype(np.float16).astype(F32)
        B = rng.standard_normal((256, 1536)).astype(np.float16).astype(F32)
        old = os.environ.get("G17_REF_THREADS")
        try:
            os.environ["G17_REF_THREADS"] = "1"
            one = TCR._gemm_mma_fast(A, B, None, 64, 1536, 256)
            os.environ["G17_REF_THREADS"] = "8"
            many = TCR._gemm_mma_fast(A, B, None, 64, 1536, 256)
        finally:
            os.environ.pop("G17_REF_THREADS", None) if old is None else os.environ.__setitem__("G17_REF_THREADS", old)
        self.assertTrue(same(one, many))
        self.assertFalse(same(one, TCR._gemm_mma_fast(A, B[::-1].copy(), None, 64, 1536, 256)))

    def test_uint8_fields_equal_the_int64_form(self):
        import g17q4graph as G
        rng = np.random.default_rng(1)
        for bits in (4, 8):
            W = rng.integers(0, 2 ** 32, size=64 * 2048 * bits // 32, dtype=np.uint64).astype(np.uint32)
            per = 32 // bits
            w = W.reshape(64, 2048 // per).astype(np.uint64)
            old = ((w[:, :, None] >> (bits * np.arange(per)).astype(np.uint64)) & ((1 << bits) - 1)).reshape(64, 2048)
            new = G._unpack(W, bits, 64, 2048)
            self.assertTrue(np.array_equal(old.astype(np.int64), new.astype(np.int64)))


if __name__ == "__main__":
    unittest.main()
