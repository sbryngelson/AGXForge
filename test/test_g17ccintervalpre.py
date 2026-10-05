"""cc's interval attempt (MM 25.144.12): scalar code beside a K-looped tensor body may run a counted loop.

Before it, the allocator's narrow pool beside such a body was r4..r15 less the body's registers (5 left), and every
pre-coloured value - a loop phi group, a read_sr destination - held its register for the WHOLE program, so a scalar tail
with a counter and three read_sr values could not allocate (g17swigluqmm's tail had to be straight-line, 49.7 KB). The
interval attempt, tried only after every earlier attempt refused, shares the body's narrow registers after its last row
and holds a pre-coloured register only over its live interval.

- The loop-tail fused kernel compiles, guards refusing and clean, with no load-use hazard and two loops (the body's and
  the tail's), and it is under the 16 KiB instruction-footprint cliff.
- The same program REFUSES with the interval attempt disabled: this test is about the attempt, not a program that would
  have compiled anyway.
- Bytes of programs that compiled before are unchanged: every pinned delivered recipe (tools/tensorops-model/
  cc-bytes-pin.json, taken from the recorded deliver indexes) rebuilds to its recorded sha.
- The rule itself, on the allocator's own output: no two values the allocator gave one register are live at once."""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

PIN = os.path.join(ROOT, "tools", "tensorops-model", "cc-bytes-pin.json")


class LoopTail(unittest.TestCase):
    def test_the_loop_tail_compiles_clean_and_small(self):
        import g17swigluqmm as SW
        from agxforge.g17 import tensorview as TV
        prog = SW.build(SW.layout(256))
        g = prog.guards
        self.assertEqual((g["mode"], g["dead"], g["release"]), ("refuse", [], []))
        v = TV.view(prog.code)
        self.assertEqual(TV.hazards(v), [])
        self.assertEqual(len(TV.loops(v)), 2)
        self.assertLess(len(prog.code), 16384)

    def test_it_refuses_without_the_interval_attempt(self):
        import g17swigluqmm as SW
        from agxforge.g17 import cc
        orig = cc.Alloc._run

        def no_interval(self, insts):
            if getattr(self, "interval_pre", False):
                raise cc.Unsupported("interval attempt disabled for the control")
            return orig(self, insts)
        cc.Alloc._run = no_interval
        try:
            with self.assertRaises(cc.Unsupported):
                SW.build(SW.layout(256))
        finally:
            cc.Alloc._run = orig


class BytesUnchanged(unittest.TestCase):
    def test_pinned_delivered_programs_rebuild_to_their_sha(self):
        import hashlib
        import g17deliver as D
        pins = json.load(open(PIN))["entries"]
        self.assertGreater(len(pins), 20)
        bad = [p["name"] for p in pins if hashlib.sha256(D.rebuild(p).code).hexdigest() != p["program_sha256"]]
        self.assertEqual(bad, [])


class NoTwoLiveValuesShareARegister(unittest.TestCase):
    def test_the_allocation_of_the_loop_tail(self):
        # The allocator's own output (the instruction list it returns): recompute CFG liveness exactly as it does and
        # check that every register holds at most one live value at every index - the property the interval rule must
        # keep when it lends a pre-coloured register.
        import g17swigluqmm as SW
        from agxforge.g17 import cc
        captured = {}
        orig = cc.Alloc._run

        def capture(self, insts):
            out = orig(self, insts)
            if getattr(self, "interval_pre", False):
                captured["insts"] = out
                captured["alloc"] = self
            return out
        cc.Alloc._run = capture
        try:
            SW.build(SW.layout(256))
        finally:
            cc.Alloc._run = orig
        insts, al = captured["insts"], captured["alloc"]
        live_in, live_out = al._cfg_live(insts)
        reg_of = {}
        for m in insts:
            for v, r in zip(m.defs, m.fields.get("_defs", ())):
                reg_of.setdefault(v, r)
        # a phi group's members share one register by design (phi coalescing): one value for this check
        group = {}
        for m in insts:
            g = [x for x in (m.fields.get("phi_group") or ()) if hasattr(x, "name")]
            ids = {id(x) for x in g}
            for x in list(group):
                if ids & group[x]:
                    ids |= group.pop(x)
            for x in ids:
                group[x] = ids
        key = lambda v: min(group.get(id(v), {id(v)}))     # noqa: E731
        clashes = []
        for i in range(len(insts)):
            seen = {}
            for v in live_out[i]:                           # the values that survive instruction i
                r = reg_of.get(v)
                if r is None:
                    continue
                if r in seen and seen[r] != key(v):
                    clashes.append((i, r))
                seen[r] = key(v)
        self.assertEqual(clashes[:5], [])


if __name__ == "__main__":
    unittest.main()
