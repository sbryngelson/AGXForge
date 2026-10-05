"""Piece B's imageblock deform arms: the positive control reproduces and every removed
instruction is load-bearing. Asserts the committed receipt (a GPU run's evidence)."""
import json, os, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = json.load(open(os.path.join(ROOT, "isa", "g17-imageblock-deform-receipt.json")))
ARMS = REC["arms"]
REMOVES = ["remove_op11180", "remove_op11667", "remove_op10289", "remove_op1006", "remove_op435"]


def const(v):
    return v is not None and len(set(v)) == 1


class ImageblockDeform(unittest.TestCase):
    def test_the_observable_is_buf0_buf1_not_buf2(self):
        # BUF2 is never written; comparing it would make every arm match degenerately
        for name in ARMS:
            self.assertTrue(ARMS[name]["buf2_is_sentinel"], name)

    def test_the_positive_control_reproduces(self):
        c = ARMS["apple_unmodified"]
        self.assertTrue(c["agree"])
        self.assertTrue(c["matches_control"])
        # a real program output: both buffers vary across the 32 lanes
        self.assertEqual(len(set(c["buf0"])), 32)
        self.assertEqual(len(set(c["buf1"])), 32)

    def test_every_removed_instruction_is_load_bearing(self):
        for name in REMOVES:
            a = ARMS[name]
            self.assertTrue(a["agree"], name)
            self.assertFalse(a["matches_control"], name)          # the output changed
            self.assertTrue(a["tail_match"], name)                 # framing preserved
            self.assertEqual(a["fillers"], REC_len(name))          # length/2 fillers

    def test_each_effect_is_instruction_specific(self):
        # not uniform corruption: the arms differ from each other, and from the control,
        # in distinct ways (some zero a buffer, some make it constant, some leave it varying)
        signatures = {name: (const(ARMS[name]["buf0"]), set(ARMS[name]["buf0"]) == {0},
                             const(ARMS[name]["buf1"]), set(ARMS[name]["buf1"]) == {0})
                      for name in REMOVES}
        self.assertGreater(len(set(signatures.values())), 1)
        # op11667 zeroes buf0; op1006 zeroes buf1 -- opposite, so not a shared corruption
        self.assertEqual(set(ARMS["remove_op11667"]["buf0"]), {0})
        self.assertEqual(set(ARMS["remove_op1006"]["buf1"]), {0})


def REC_len(name):
    return {"remove_op11180": 5, "remove_op11667": 6, "remove_op10289": 6,
            "remove_op1006": 3, "remove_op435": 5}[name]


if __name__ == "__main__":
    unittest.main()
