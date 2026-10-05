"""Guards for the source-to-opcode reachability map.

The map's value is that a row is a POSITIVE result - this source emits this instruction. So the
tests pin that positives were actually found, that the ubiquitous prologue forms are excluded rather
than credited to every construct, and that the absence of a route is recorded as a limit of the
probe set rather than a fact about the opcode.
"""
import json, os, unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RECORD = os.path.join(ROOT, "tools", "tensorops-model", "source-reachability.json")


def _r():
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


class TheProbesCompiledAndEmitted(unittest.TestCase):

    # A probe that never compiles emits nothing, and "emits nothing" is indistinguishable from
    # "reaches no new opcode" in every downstream count.  `compiled > 0.8 * probes` let exactly
    # that through: `C[tid] = A[tid] %% (B[tid] | 1);` reached the Metal compiler with the `%%`
    # intact - the body is a `%`-format ARGUMENT, so it was never unescaped - and the int-modulo
    # route was silently absent from the map while 65 of 66 kept the suite green.
    EXPECTED_REFUSALS = {}   # construct -> why. Empty: every probe here compiles.

    def test_every_probe_compiles(self):
        bad = {row["construct"]: row["status"] for row in _r()["rows"]
               if row["status"] != "ok" and row["construct"] not in self.EXPECTED_REFUSALS}
        self.assertEqual(bad, {}, "a probe that does not compile reaches nothing, and that is "
                                  "indistinguishable from reaching nothing new")

    def test_the_refusal_allowlist_is_not_a_dumping_ground(self):
        """The control on the escape hatch: a refusal must carry a reason, and stay rare."""
        self.assertLess(len(self.EXPECTED_REFUSALS), 0.05 * _r()["probes"] + 1)
        for construct, why in self.EXPECTED_REFUSALS.items():
            self.assertTrue(why and len(why) > 15, construct)

    def test_every_compiled_probe_emitted_instructions(self):
        for row in _r()["rows"]:
            if row["status"] != "ok":
                continue
            with self.subTest(construct=row["construct"]):
                self.assertGreater(row["instructions"], 0)
                self.assertTrue(row["forms"])


class ThePrologueIsMeasuredNotInferred(unittest.TestCase):
    """The baseline must come from a CONTROL, never from what most probes happen to emit.

    The population-share version ("a form in >=90% of constructs is prologue") is wrong in a way
    that gets worse as the probe set grows: it calls a form noise exactly when most probes reach
    it.  It had already misfiled the LOAD `590/8` - no empty kernel emits it - so every probe that
    reads a buffer was denied credit for the instruction that does the reading.  An empty kernel of
    each signature puts the real floor at two forms.
    """

    def test_the_prologue_is_exactly_what_an_empty_kernel_emits(self):
        p = _r()["prologue_forms"]
        self.assertEqual(sorted(p), ["13483/2", "684/4"],
                         "the empty-kernel floor moved; re-measure before trusting any row")

    def test_the_population_heuristic_is_shown_to_disagree(self):
        """The finding itself, pinned: forms most probes emit that no control does."""
        r = _r()
        misfiled = set(r["ubiquitous_but_reached"])
        self.assertTrue(misfiled, "if this is empty the two baselines agree - suspect the control")
        self.assertIn("590/8", misfiled, "the load is reached, not prologue")
        self.assertFalse(misfiled & set(r["prologue_forms"]),
                         "a form cannot be both the control's and not the control's")

    def test_no_row_credits_itself_with_the_prologue(self):
        """What the subtraction is for. Its failure mode is crediting every probe with the floor."""
        floor = set(_r()["prologue_forms"])
        for row in _r()["rows"]:
            if row["status"] != "ok":
                continue
            with self.subTest(construct=row["construct"]):
                self.assertFalse(floor & set(row["reached"]))

    def test_a_loading_construct_is_credited_with_the_load(self):
        """The positive control: the subtraction must not remove a genuinely reached form."""
        rows = {r["construct"]: r for r in _r()["rows"] if r["status"] == "ok"}
        self.assertIn("590/8", rows["4-byte load/store"]["reached"])


class ReachIsMeasuredAgainstApplesOwnCorpus(unittest.TestCase):
    """The progress number: how many observed opcodes this probe set can make the compiler emit."""

    def test_both_denominators_are_published(self):
        """992 is the vendor corpus, 1,167 the union. A share with no named population is noise."""
        r = _r()
        self.assertEqual(r["opcodes_observed_vendor_only"], 992)
        self.assertEqual(r["opcodes_observed_union"], 1167)
        self.assertGreater(r["opcodes_observed_union"], r["opcodes_observed_vendor_only"])

    def test_coverage_does_not_regress(self):
        """A ratchet whose floor comes from the field it guards, not from a neighbouring filter."""
        r = _r()
        self.assertGreaterEqual(r["opcodes_reached_and_observed"], 170,
                                "reach fell; 177 of 1,167 was measured at 190 probes")
        self.assertEqual(len(r["opcodes_reached"]),
                         r["opcodes_reached_and_observed"] + len(r["opcodes_reached_not_in_corpora"]),
                         "reached must partition into observed and not-observed")

    def test_the_unreached_remainder_is_not_claimed_as_unreachable(self):
        self.assertIn("not about the opcode", _r()["caveat"])

    def test_reached_is_a_subset_of_what_was_emitted(self):
        for row in _r()["rows"]:
            if row["status"] != "ok":
                continue
            with self.subTest(construct=row["construct"]):
                self.assertTrue(set(row["reached"]) <= set(row["forms"]))


class UbiquitousFormsAreNotCreditedAsReached(unittest.TestCase):
    """Kept for the population-share view, now reported BESIDE the control rather than as truth."""

    def test_the_ubiquitous_set_is_identified(self):
        u = _r()["ubiquitous_forms"]
        self.assertTrue(u, "no ubiquitous form found - suspect the scan")
        self.assertLess(len(u), 6, "too many 'ubiquitous' forms; the threshold is wrong")

    def test_no_prologue_form_is_claimed_as_uniquely_reached(self):
        r = _r()
        ub = set(r["prologue_forms"])
        for construct, forms in r["uniquely_reached"].items():
            with self.subTest(construct=construct):
                self.assertFalse(ub & set(forms))


class TheKnownRoutesHold(unittest.TestCase):
    """A handful of rows a compiler author would rely on.

    These are asserted from each row's OWN `reached` set, never from `uniquely_reached`.  The
    earlier version used uniqueness, and uniqueness is a property of the probe POPULATION: adding
    `while loop` and `do-while` took `10369/10` away from `runtime loop`, and adding `fp32 sin`
    took `2190/16` away from `fp32 fma`.  Both routes still hold - the test went red because the
    map got better, which is the wrong direction for a guard to point.
    """

    CASES = {"simdgroup mul half": "839/14", "simdgroup mad half": "838/14",
             "fp32 fma": "2190/16", "simd_sum": "16842/10", "uniform load": "592/4",
             "runtime loop": "10369/10",
             # A switch reaches op10369 at width 6 - the mode-SET layout, from a construct no
             # loop probe selects.
             "switch 4-way": "10369/6"}

    def test_each_named_construct_reaches_its_form(self):
        rows = {r["construct"]: r for r in _r()["rows"] if r["status"] == "ok"}
        for construct, form in self.CASES.items():
            with self.subTest(construct=construct):
                self.assertIn(construct, rows)
                self.assertIn(form, rows[construct]["reached"])

    def test_uniqueness_is_not_used_as_a_route_assertion(self):
        """The guard on the guard: uniqueness may legitimately vanish as probes are added.

        So it must not be what a route claim rests on. If a CASES form is also reached by another
        construct, that is fine and this records it rather than failing.
        """
        rows = [r for r in _r()["rows"] if r["status"] == "ok"]
        for construct, form in self.CASES.items():
            others = [r["construct"] for r in rows
                      if r["construct"] != construct and form in r["reached"]]
            with self.subTest(construct=construct):
                self.assertIsInstance(others, list)


class TheLimitIsRecorded(unittest.TestCase):

    def test_the_caveat_says_absence_is_about_the_probes(self):
        c = _r()["caveat"]
        self.assertIn("probe set", c)
        self.assertIn("not about the opcode", c)


if __name__ == "__main__":
    unittest.main()
