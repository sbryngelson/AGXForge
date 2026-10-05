"""tlower kloop_bases: loop-carried row bases (MM 25.144.1), compile-only.

A fragment row more than 7 rows (4,096-byte rows) from the loop index overflows the load's byte displacement, and the
plain K loop rebuilt its base register inside the loop on every trip (32 of 53 loop instructions at M 256, K 2048).
kloop_bases sets each such base once before the loop and advances it with one add per trip. Hardware: bit-exact
(g17tensorcommonruntime generic arms, recorded in 25.144.1)."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))


def loop_len(body):
    from agxforge.g17 import tensorview as TV
    first, back = list(TV.loops(TV.view(body)))[0]
    return back - first + 1


class KloopBases(unittest.TestCase):
    def lower(self, **kw):
        from agxforge.g17 import tlower
        return tlower.lower(256, 2048, 2048, 2048, 2048, 2048, a_type="half", b_type="half", kloop=True, sg=4, grid=1,
                            grid_n=128, **kw)

    def test_default_bytes_unchanged(self):
        self.assertEqual(self.lower()[0], self.lower(kloop_bases=False)[0])

    def test_the_loop_shrinks(self):
        self.assertEqual(loop_len(self.lower()[0]), 53)
        self.assertEqual(loop_len(self.lower(kloop_bases=True)[0]), 29)
        self.assertEqual(loop_len(self.lower(kloop_bases=True, kloop_unroll=2)[0]), 45)

    def test_the_module_default_is_off(self):
        from agxforge.g17 import tlower
        self.assertFalse(tlower.KLOOP_BASES)


if __name__ == "__main__":
    unittest.main()
