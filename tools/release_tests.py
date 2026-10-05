#!/usr/bin/env python3
"""Run the release's selected test modules, one process each, the way the research checkout's runner does.

    python3 tools/release_tests.py [-j N]

The release keeps only tests that passed from a fresh export with an empty cache; none of them dispatches GPU work
(the research suite's three GPU modules are not part of the release). Exit 0 only when every module passes.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(mod, timeout):
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-W", "ignore", "-m", "unittest", "discover", "-s", "test", "-p", mod],
                           cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        out, rc = (p.stderr or "") + (p.stdout or ""), p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "timed out after %ds" % timeout, -1
    ran = next((int(l.split()[1]) for l in out.splitlines() if l.startswith("Ran ") and " test" in l), 0)
    return mod, rc, ran, time.time() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    # THE SELECTED TESTS ONLY: RELEASE-MANIFEST.json marks them reason "seed". A test module that is here only because a
    # selected test imports it (reason "import") is a helper in this release, not a test of it.
    manifest = json.loads((ROOT / "RELEASE-MANIFEST.json").read_text())["files"]
    mods = sorted(Path(p).name for p, r in manifest.items()
                  if p.startswith("test/test_") and p.endswith(".py") and r.get("reason") == "seed")
    with ThreadPoolExecutor(a.j) as ex:
        results = list(ex.map(lambda m: run(m, a.timeout), mods))
    failed = [r for r in results if r[1] != 0]
    for mod, rc, ran, sec, out in failed:
        print("FAIL %s (rc %d)\n%s" % (mod, rc, out[-2000:]))
    print("%d modules, %d tests, %d failed" % (len(results), sum(r[2] for r in results), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
