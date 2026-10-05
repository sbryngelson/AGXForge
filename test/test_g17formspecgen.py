"""Guards for the form-spec generator and the forms it admits.

The acceptance this exists for: every form the generator appends must reconstruct byte-exact
through the repo's own encoder, from the COMMITTED files - re-verified here, not trusted from the
run that admitted it. And the generator is trusted on new forms only as far as it reproduces the
984 forms the file was written with.
"""
import json, os, sys, unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
SPEC = os.path.join(ROOT, "isa", "g17-form-spec.jsonl")
FREE = os.path.join(ROOT, "isa", "g17-free-bits.jsonl")
GEN = "tools/g17formspecgen.py --extend"


def _rows(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh]


def _appended(path):
    return [r for r in _rows(path) if r.get("generator") == GEN]


class TheRecordedFileIsUntouched(unittest.TestCase):
    """The generator APPENDS. The 984 forms the file was written with must not move."""

    def test_the_first_984_rows_carry_no_generator_tag(self):
        rows = _rows(SPEC)
        self.assertGreaterEqual(len(rows), 984)
        self.assertFalse([r for r in rows[:984] if r.get("generator")],
                         "a recorded row was rewritten")
        self.assertTrue(all(r.get("generator") == GEN for r in rows[984:]),
                        "an appended row has no provenance")

    def test_no_form_is_specified_twice(self):
        keys = [(r["opcode"], r["length"]) for r in _rows(SPEC)]
        self.assertEqual(len(keys), len(set(keys)))


class TheClassifierReproducesTheRecordedFile(unittest.TestCase):
    """Trusted on new forms only as far as it reproduces the old. A subset keeps this fast; the
    full 984 is `python3 tools/g17formspecgen.py --check`."""

    def test_invisible_is_reproduced_on_a_sample(self):
        import g17formspecgen as F
        rep = F.check(limit=30)
        got, tot = rep["by_class"]["invisible"]
        self.assertGreaterEqual(got / tot, 0.99, "the class that governs D2 is not reproduced")
        g, t = rep["operand_first_moved_agrees"]
        self.assertEqual(g, t, "the operand weights the encoder reads disagree")

    def test_a_length_changing_flip_is_not_called_an_operand(self):
        """op458/10 bit 2.0 re-lengths the instruction to four bytes. For authoring the ten-byte
        form it must hold its witness value, so it is NOT an operand bit - whatever the recorded
        file says."""
        import g17formspecgen as F
        r = next(x for x in F.recorded() if x["opcode"] == 458 and x["length"] == 10)
        self.assertNotEqual(F.classify(bytes.fromhex(r["witness"]))["2.0"]["class"], "operand")


class EveryAppendedFormReconstructs(unittest.TestCase):
    """The goal's acceptance, re-verified from the committed files."""

    @classmethod
    def setUpClass(cls):
        from agxforge.g17 import encode as E
        import g17formspecgen as F
        cls.E, cls.F = E, F
        E._FORMS = None; E._FREE = None
        E.forms(); E.freebits()
        cls.added = _appended(SPEC)

    def test_there_are_appended_forms(self):
        self.assertGreater(len(self.added), 0)

    def test_each_appended_witness_reconstructs_byte_exact(self):
        bad = [(r["opcode"], r["length"]) for r in self.added
               if not self.F._reconstructs(self.E, r["opcode"], r["length"],
                                           bytes.fromhex(r["witness"]))]
        self.assertEqual(bad, [], "an admitted form does not reconstruct its own Apple witness")

    def test_the_reconstruction_can_fail(self):
        """Positive control: flip a FORCED bit of an admitted witness and reconstruction must
        refuse it - otherwise the guard above passes on anything."""
        for r in self.added:
            forced = [k for k, v in r["bits"].items() if v["class"] == "forced"]
            if not forced:
                continue
            b, i = map(int, forced[0].split("."))
            w = bytearray.fromhex(r["witness"]); w[b] ^= 1 << i
            self.assertFalse(self.F._reconstructs(self.E, r["opcode"], r["length"], bytes(w)))
            return
        self.skipTest("no admitted form has a forced bit")


class TheFreeBitsAreHonest(unittest.TestCase):

    def test_no_appended_form_has_an_undetermined_bit(self):
        bad = [(r["opcode"], r["length"]) for r in _appended(FREE)
               if any(v.get("verdict") == "undetermined" for v in r["bits"].values())]
        self.assertEqual(bad, [])

    def test_absent_at_this_length_is_only_used_beyond_the_form(self):
        """The verdict is a checked fact about THIS form's bytes, never a way to settle a bit the
        form has."""
        for r in _appended(FREE):
            for k, v in r["bits"].items():
                if v.get("verdict") == "absent at this length":
                    with self.subTest(form=(r["opcode"], r["length"]), bit=k):
                        self.assertGreaterEqual(int(k.split(".")[0]), r["length"])

    def test_constant_needs_two_instances(self):
        for r in _appended(FREE):
            for k, v in r["bits"].items():
                if v.get("verdict") == "constant":
                    self.assertGreaterEqual(v["instances"], 2)

    def test_every_appended_spec_row_has_a_free_bits_row(self):
        a = {(r["opcode"], r["length"]) for r in _appended(SPEC)}
        b = {(r["opcode"], r["length"]) for r in _appended(FREE)}
        self.assertEqual(a, b)


class NoAdmittedFormBreaksTheOperandMapInvariant(unittest.TestCase):
    """g17regress holds every form to "no operand map holds a bit the spec calls an opcode bit";
    a form this tool adds must satisfy the invariant already in force."""

    def test_no_appended_form_conflicts(self):
        import g17formspecgen as F
        bad = [(r["opcode"], r["length"]) for r in _appended(SPEC)
               if F.operand_map_conflicts(r["opcode"], r["length"], r["bits"])]
        self.assertEqual(bad, [])

    def test_the_conflict_check_can_fire(self):
        """Positive control: call a bit some shipped operand map owns an OPCODE bit, and the check
        must report it. Otherwise the test above passes on a check that looks at nothing."""
        import g17as, g17formspecgen as F
        for (op, ln, idx, kind), r in g17as.maps().items():
            pos = [(b, i) for _, b, i, _ in (tuple(p) for p in (r.get("positions") or []))]
            pos = [p for p in pos if (op, ln, p[0], p[1]) not in g17as.SPEC_MEASURED_OPERAND]
            if pos:
                b, i = pos[0]
                bits = {"%d.%d" % (b, i): {"class": "opcode", "value": 0}}
                self.assertTrue(F.operand_map_conflicts(op, ln, bits))
                return
        self.skipTest("no operand map with a position")

    def test_forced_bits_are_not_banned(self):
        """The check is the GUARD'S OWN definition: opcode class, not forced. A first version
        included forced, found nine conflicts where the guard finds two, and would have refused
        forms the invariant allows."""
        import g17as, g17formspecgen as F
        for (op, ln, idx, kind), r in g17as.maps().items():
            pos = [(b, i) for _, b, i, _ in (tuple(p) for p in (r.get("positions") or []))]
            pos = [p for p in pos if (op, ln, p[0], p[1]) not in g17as.SPEC_MEASURED_OPERAND]
            import g17opmap
            pos = [p for p in pos if p not in g17opmap.selects_opcode(op, ln)]
            if pos:
                b, i = pos[0]
                bits = {"%d.%d" % (b, i): {"class": "forced", "value": 0}}
                self.assertFalse(F.operand_map_conflicts(op, ln, bits))
                return
        self.skipTest("no suitable map")


class TheBreadthReportIsConsistent(unittest.TestCase):
    """The split the work is steered by. It must agree with what was actually admitted."""

    REPORT = os.path.join(ROOT, "isa", "g17-encoding-breadth.json")

    @classmethod
    def setUpClass(cls):
        with open(cls.REPORT) as fh:
            cls.doc = json.load(fh)

    def test_admitted_in_the_report_is_exactly_what_was_appended(self):
        admitted = {row["form"] for k in ("specified and lowered", "specified, not lowered")
                    for row in self.doc["ranked"].get(k, [])}
        appended = {"%d/%d" % (r["opcode"], r["length"]) for r in _appended(SPEC)}
        self.assertEqual(admitted, appended)

    def test_the_three_splits_partition_the_population(self):
        seen = [row["form"] for rows in self.doc["ranked"].values() for row in rows]
        self.assertEqual(len(seen), len(set(seen)), "a form appears in two splits")
        for k, v in self.doc["summary"].items():
            self.assertEqual(v["forms"], len(self.doc["ranked"][k]))
            self.assertEqual(v["programs"], sum(r["programs"] for r in self.doc["ranked"][k]))

    def test_each_split_is_ranked_by_programs_gated(self):
        """Effort goes where programs are gated, not where forms are numerous."""
        for k, rows in self.doc["ranked"].items():
            with self.subTest(split=k):
                progs = [r["programs"] for r in rows]
                self.assertEqual(progs, sorted(progs, reverse=True))

    def test_not_recovered_carries_a_reason_for_every_form(self):
        rows = self.doc["ranked"].get("not recovered", [])
        self.assertTrue(all(r["reason"] and r["reason"] != "admit" for r in rows))
        self.assertEqual(sum(self.doc["not_recovered_by_reason"].values()), len(rows))


if __name__ == "__main__":
    unittest.main()
