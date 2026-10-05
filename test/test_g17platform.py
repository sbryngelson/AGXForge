"""Guards for T9's platform pin.

T9 is blocked, not open - it asks about hardware this campaign does not have. What is testable is
that the boundary announces itself: a reader on a different part, OS or toolchain must be TOLD, and
told which claims that bears on.
"""
import json, os, shutil, subprocess, sys, tempfile, unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TOOL = os.path.join(ROOT, "tools", "g17platform.py")
RECORD = os.path.join(ROOT, "isa", "g17-measurement-platform.json")


def _run(*args):
    return subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True, cwd=ROOT)


class ThePlatformIsRecorded(unittest.TestCase):

    def test_the_record_names_the_part_and_the_toolchain(self):
        p = json.load(open(RECORD, encoding="utf-8"))["platform"]
        for k in ("machine_model", "chip", "os_build", "metal_compiler", "xcode"):
            with self.subTest(field=k):
                self.assertTrue(p.get(k), "%s is empty; the pin does not pin it" % k)

    def test_gpu_cores_is_not_taken_from_hw_ncpu(self):
        """hw.ncpu is CPU cores; reading it as GPU cores would put a wrong figure beside
        every per-core claim in chapters 8 and 9."""
        p = json.load(open(RECORD, encoding="utf-8"))["platform"]
        self.assertNotEqual(str(p["gpu_cores"]).split()[0], str(p["cpu_cores"]))
        self.assertIn("20", str(p["gpu_cores"]))

    def test_every_field_says_what_it_bears_on(self):
        d = json.load(open(RECORD, encoding="utf-8"))
        for k in d["platform"]:
            if k in ("cpu_cores",):
                continue
            with self.subTest(field=k):
                self.assertIn(k, d["bears_on"],
                              "%s is recorded but nothing says what it invalidates" % k)


class TheBoundaryAnnouncesItself(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._record_before = open(RECORD, encoding="utf-8").read()

    def test_check_passes_on_the_recorded_platform(self):
        r = _run("--check")
        self.assertEqual(r.returncode, 0, r.stdout)
        if "THE PLATFORM HAS MOVED" in r.stdout:
            # Running on a build the figures were NOT taken on (macOS 27, 26A434, since 2026-10-02).
            # Re-recording would claim they were; the tool reports the move without failing, which
            # is its contract, and the negative control below covers the naming on a copy.
            self.skipTest("the running platform is not the recorded one: " + r.stdout.splitlines()[0])
        self.assertIn("matches", r.stdout)

    def test_a_moved_field_is_named_with_what_it_bears_on(self):
        """The negative control, on a COPY: without this, --check passing proves nothing.

        This used to rewrite the tracked record in place and restore it. Serial that is correct;
        under a parallel runner it races every reader of the file, and the loser of that race
        leaves the tracked record modified on disk. The tool takes --record so the corruption
        happens somewhere that is nobody else's.
        """
        with tempfile.TemporaryDirectory() as t:
            rec = os.path.join(t, "record.json")
            shutil.copy(RECORD, rec)
            self.assertEqual(_run("--check", "--record", rec).returncode, 0,
                             "the copy must start matching, or the corruption below proves nothing")
            d = json.load(open(rec, encoding="utf-8"))
            d["platform"]["os_build"] = "99Z99-not-a-real-build"
            with open(rec, "w", encoding="utf-8") as fh:
                json.dump(d, fh, indent=1, sort_keys=True)
            r = _run("--check", "--record", rec)
            self.assertIn("PLATFORM HAS MOVED", r.stdout)
            self.assertIn("os_build", r.stdout)
            self.assertIn("driver behaviour", r.stdout)
        self.assertEqual(open(RECORD, encoding="utf-8").read(), self._record_before,
                         "the tracked record was modified by this test")

    def test_a_difference_is_reported_but_does_not_fail(self):
        """'You are on a different build' is information, not an error. Also on a copy."""
        with tempfile.TemporaryDirectory() as t:
            rec = os.path.join(t, "record.json")
            shutil.copy(RECORD, rec)
            d = json.load(open(rec, encoding="utf-8"))
            d["platform"]["metal_compiler"] = "Apple metal version 99999"
            with open(rec, "w", encoding="utf-8") as fh:
                json.dump(d, fh, indent=1, sort_keys=True)
            self.assertEqual(_run("--check", "--record", rec).returncode, 0)
        self.assertEqual(open(RECORD, encoding="utf-8").read(), self._record_before,
                         "the tracked record was modified by this test")


if __name__ == "__main__":
    unittest.main()
