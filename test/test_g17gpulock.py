#!/usr/bin/env python3
"""The machine-wide GPU lock (tools/g17gpulock.py and tools/g17gpulock.h).

Every behavioural case runs in subprocesses under a THROWAWAY HOME, and with AGXFORGE_GPU_LOCK_HELD
removed from their environment: a test that took the real lock would stall a peer's dispatch, and
one that inherited the suite's variable would skip the very acquire it means to test.
"""
import ctypes, os, re, subprocess, sys, tempfile, textwrap, time, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
LIBACCEL = os.path.join(ROOT, "spike", "accel", "libaccel.dylib")


def env(home, **extra):
    e = {k: v for k, v in os.environ.items() if k not in ("AGXFORGE_GPU_LOCK_HELD", "AGXFORGE_GPU_LOCK_TIMEOUT")}
    e.update(HOME=home, PYTHONPATH=TOOLS, **extra)
    return e


HOLD = textwrap.dedent("""
    import sys, time, g17gpulock
    with g17gpulock.acquire(sys.argv[1]):
        print("held", flush=True)
        time.sleep(float(sys.argv[2]))
""")
TRY = textwrap.dedent("""
    import os, sys, time, g17gpulock
    t = time.monotonic()
    try:
        with g17gpulock.acquire(sys.argv[1], timeout=float(sys.argv[2])) as h:
            print("acquired %.2f inherited=%s" % (time.monotonic() - t, h.inherited_from), flush=True)
    except TimeoutError as e:
        print("timeout", e, flush=True)
""")


class Holder:
    def __init__(self, home, mode="exclusive", seconds=3.0):
        self.p = subprocess.Popen([sys.executable, "-c", HOLD, mode, str(seconds)], env=env(home),
                                  stdout=subprocess.PIPE, text=True)
        assert self.p.stdout.readline().strip() == "held"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.p.kill()
        self.p.wait()


def attempt(home, mode="exclusive", timeout=0.5, **extra):
    return subprocess.run([sys.executable, "-c", TRY, mode, str(timeout)], env=env(home, **extra),
                          capture_output=True, text=True, timeout=30).stdout.strip()


class ThePythonLock(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()

    def test_a_second_exclusive_waits_then_times_out_naming_the_holder(self):
        with Holder(self.home) as h:
            out = attempt(self.home)
        self.assertTrue(out.startswith("timeout"), out)
        self.assertIn("pid %d" % h.p.pid, out)

    def test_it_acquires_once_the_holder_exits(self):
        with Holder(self.home, seconds=0.6):
            out = attempt(self.home, timeout=10)
        self.assertTrue(out.startswith("acquired"), out)

    def test_a_child_of_a_live_holder_inherits_instead_of_waiting(self):
        with Holder(self.home) as h:
            out = attempt(self.home, AGXFORGE_GPU_LOCK_HELD="exclusive:%d" % h.p.pid)
        self.assertEqual(out.split()[0], "acquired")
        self.assertIn("inherited=%d" % h.p.pid, out)

    def test_a_dead_holder_in_the_variable_is_not_honoured(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"]); dead.wait()
        with Holder(self.home):
            out = attempt(self.home, AGXFORGE_GPU_LOCK_HELD="exclusive:%d" % dead.pid)
        self.assertTrue(out.startswith("timeout"), out)

    def test_shared_holders_coexist_and_exclude_an_exclusive(self):
        with Holder(self.home, mode="shared"):
            self.assertTrue(attempt(self.home, mode="shared").startswith("acquired"))
            self.assertTrue(attempt(self.home, mode="exclusive").startswith("timeout"))


@unittest.skipUnless(os.path.exists(LIBACCEL), "libaccel.dylib is built by `make native-tools`")
class TheNativeLock(unittest.TestCase):
    """The C side, through libaccel's exported ac_gpu_lock - no command buffer, no dispatch."""
    CALL = ("import ctypes, sys; L = ctypes.CDLL(sys.argv[1]); "
            "print(L.ac_gpu_lock(), flush=True)")

    def native(self, home, **extra):
        return subprocess.run([sys.executable, "-c", self.CALL, LIBACCEL], env=env(home, **extra),
                              capture_output=True, text=True, timeout=30)

    def test_native_code_waits_on_a_python_holder_and_times_out(self):
        home = tempfile.mkdtemp()
        with Holder(home) as h:
            r = self.native(home, AGXFORGE_GPU_LOCK_TIMEOUT="0.5")
        self.assertEqual(r.stdout.strip(), "-2")
        self.assertIn("pid %d" % h.p.pid, r.stderr)

    def test_native_code_inherits_from_a_live_python_holder(self):
        home = tempfile.mkdtemp()
        with Holder(home) as h:
            r = self.native(home, AGXFORGE_GPU_LOCK_TIMEOUT="0.5", AGXFORGE_GPU_LOCK_HELD="exclusive:%d" % h.p.pid)
        self.assertEqual(r.stdout.strip(), "0")

    def test_native_code_takes_a_free_lock(self):
        self.assertEqual(self.native(tempfile.mkdtemp(), AGXFORGE_GPU_LOCK_TIMEOUT="0.5").stdout.strip(), "0")


class NoDispatchPathBypassesIt(unittest.TestCase):
    """THE CHECK THAT FIRES ON THE NEXT ENTRY POINT. A new native program, or a new function in
    libaccel, that makes its own command buffer would dispatch unlocked; so would a Python tool
    that opens the lock file with its own non-blocking flock."""

    def test_every_command_buffer_is_made_through_the_header(self):
        bad = []
        for d in ("tools", os.path.join("spike", "accel")):
            for f in sorted(os.listdir(os.path.join(ROOT, d))):
                if f.endswith((".m", ".mm")):
                    text = open(os.path.join(ROOT, d, f)).read()
                    if re.search(r"\[\s*\w+\s+commandBuffer(WithUnretainedReferences)?\s*\]", text):
                        bad.append(os.path.join(d, f))
        self.assertEqual(bad, [], "make command buffers with g17_gpu_cb(queue) from tools/g17gpulock.h")

    def test_no_tool_takes_the_lock_file_by_hand(self):
        bad = [f for f in sorted(os.listdir(TOOLS)) if f.endswith(".py") and f != "g17gpulock.py"
               and "g17-execution.lock" in open(os.path.join(TOOLS, f)).read()
               and "LOCK_NB" in open(os.path.join(TOOLS, f)).read()]
        self.assertEqual(bad, [], "use g17gpulock.acquire(): it waits, reports and is inherited")

    def test_native_command_buffer_refuses_when_lock_file_cannot_open(self):
        # No device is created: the helper must exit before messaging the nil queue.
        with tempfile.TemporaryDirectory() as td:
            source = os.path.join(td, "lock-failure.m")
            binary = os.path.join(td, "lock-failure")
            with open(source, "w") as stream:
                stream.write('#include "g17gpulock.h"\nint main(void) { g17_gpu_cb(nil); return 0; }\n')
            subprocess.run(["xcrun", "clang", "-fobjc-arc", "-I", TOOLS,
                            "-framework", "Foundation", "-framework", "Metal",
                            source, "-o", binary], check=True, capture_output=True)
            result = subprocess.run([binary], env=env("/dev/null"), capture_output=True, text=True)
            self.assertEqual(result.returncode, 75, result.stderr)
            self.assertIn("cannot open", result.stderr)


if __name__ == "__main__":
    unittest.main()
