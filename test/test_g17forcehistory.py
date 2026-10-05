"""tools/g17forcehistory.py: a shared-history graph is the original with tokens written into the log, nothing else.

CPU only. A synthetic graph directory carries the fields the helper reads (gen_region, arena_init, prompt_ids) laid out
the way tools/g17q4graph.py writes them: R_init.bin loaded at an offset in the activation arena, the generation state
inside it, the token log four bytes into the state, the prompt pre-filled and 0xFFFFFFFF where the model chooses.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import g17forcehistory as F  # noqa: E402

UNSET = 0xFFFFFFFF


def make_graph(d, prompt, cap=16, r0=4096, gen_at=1024, rinit_bytes=2048):
    rinit = bytearray(b"\x5a" * rinit_bytes)                  # recognisable filler outside the log
    log = prompt + [UNSET] * (cap - len(prompt))
    at = gen_at + 4
    rinit[gen_at:gen_at + 4] = (0).to_bytes(4, "little")     # q0
    for i, v in enumerate(log):
        rinit[at + 4 * i:at + 4 * i + 4] = v.to_bytes(4, "little")
    (d / "R_init.bin").write_bytes(bytes(rinit))
    g = dict(arenas={"ACT": 1 << 16, "W": 64},
             arena_init=[dict(arena="W", offset=0, file=str(d / "weights.bin")),
                         dict(arena="ACT", offset=r0, file=str(d / "R_init.bin"))],
             gen_region=dict(arena="ACT", offset=r0 + gen_at, log_offset=4, q0_offset=0, log_entries=cap),
             dispatches=[dict(bundle="b0", binds={})], prompt_ids=prompt, capacity=cap)
    (d / "weights.bin").write_bytes(b"\x01" * 64)
    (d / "graph.json").write_text(json.dumps(g))
    return d / "graph.json", at


class ForcedHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tokens_land_after_the_prompt_and_nothing_else_changes(self):
        graph, at = make_graph(self.d, [1, 918, 11498])
        out_graph, out_bin = F.force(graph, [395, 410, 7], "forced")
        before, after = (self.d / "R_init.bin").read_bytes(), out_bin.read_bytes()
        self.assertEqual(len(before), len(after))
        changed = [i for i in range(len(before)) if before[i] != after[i]]
        self.assertTrue(changed and min(changed) >= at + 12 and max(changed) < at + 24, changed)
        words = [int.from_bytes(after[at + 4 * i:at + 4 * i + 4], "little") for i in range(16)]
        self.assertEqual(words[:6], [1, 918, 11498, 395, 410, 7])
        self.assertTrue(all(w == UNSET for w in words[6:]))
        self.assertEqual(out_bin.name, "R_init.bin", "the executor re-applies R_init.bin by name between passes")
        g0, g1 = json.loads(graph.read_text()), json.loads(out_graph.read_text())
        self.assertEqual(g1.pop("forced_history")["first_position"], 3)
        self.assertEqual(g1["arena_init"][1]["file"], str(out_bin))
        g1["arena_init"][1]["file"] = g0["arena_init"][1]["file"]
        self.assertEqual(g0, g1, "only the R_init reference may differ")
        self.assertEqual((self.d / "R_init.bin").read_bytes(), before, "the original is untouched")

    def test_refusals(self):
        graph, _ = make_graph(self.d, [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "exceed"):
            F.force(graph, list(range(20)), "toolong")
        g = json.loads(graph.read_text())
        g["prompt_ids"] = [1, 2, 4]
        (self.d / "wrongprompt.json").write_text(json.dumps(g))
        with self.assertRaisesRegex(ValueError, "prompt_ids"):
            F.force(self.d / "wrongprompt.json", [5], "x")
        out_graph, _ = F.force(graph, [5], "once")
        with self.assertRaisesRegex(ValueError, "not all unset"):
            F.force(out_graph, [6], "twice")
        g = json.loads(graph.read_text())
        g["gen_region"]["batch"] = 2
        (self.d / "batched.json").write_text(json.dumps(g))
        with self.assertRaisesRegex(ValueError, "single-sequence"):
            F.force(self.d / "batched.json", [5], "b")

    def test_the_study_receipt_feeds_it(self):
        shared = json.loads((ROOT / "evidence/g17-matched-study-v1/decode-shared-history.json").read_text())
        tokens = shared["shared_tokens"]
        self.assertEqual(len(tokens), 128)
        graph, at = make_graph(self.d, [1, 2], cap=200)
        out_graph, out_bin = F.force(graph, tokens, "forced")
        b = out_bin.read_bytes()
        self.assertEqual([int.from_bytes(b[at + 4 * (2 + i):at + 4 * (3 + i)], "little") for i in range(128)], tokens)


if __name__ == "__main__":
    unittest.main()
