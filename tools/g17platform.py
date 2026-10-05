#!/usr/bin/env python3
"""T9: pin what this document's measurements are true OF, and say so when the platform changes.

T9 records that "other G17 parts, OS/compiler revisions and undocumented loader variants are
unmeasured unless a row says otherwise".  That is not answerable by measurement here - it asks about
hardware this campaign does not have, and no amount of work on one machine settles it.  So it is
**blocked, not open**, and the useful thing is not to answer it but to make the boundary announce
itself.

Every figure in this document was taken on one part, one OS build and one toolchain.  A reader on a
different M5, a later macOS or a newer Metal compiler has no way to know which claims still hold, and
nothing in the repository tells them they have moved.  `--check` compares the running platform
against the recorded one and names each field that differs.

**A difference is not a refutation.** Most of the machine model is encoding and layout, which a point
release will not move; timing, power and occupancy are the parts that plausibly do. `--check` reports
drift and does not fail the build, because "you are on a different build" is information a reader
needs and not an error in the document.

    python3 tools/g17platform.py            # show the platform
    python3 tools/g17platform.py --write    # record it
    python3 tools/g17platform.py --check    # name every field that has moved
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVIDENCE = os.path.join(ROOT, "isa", "g17-measurement-platform.json")

# Which fields would actually invalidate what, stated per field rather than as one verdict.
BEARING = {
    "machine_model": "everything: a different part may have different core counts and limits",
    "chip": "everything",
    "gpu_cores": "peak throughput, occupancy and every per-core figure",
    "os_build": "driver behaviour, loader variants, dispatch and completion semantics",
    "metal_compiler": "lowering, scheduling, spill counts and every 'the compiler emits' claim",
    "xcode": "the libTensorOps archive the accelerator forms resolve out of",
    "os_product": "as os_build, coarser",
    "clang": "the native tools only, not the GPU claims",
}


def _sh(*a):
    try:
        r = subprocess.run(a, capture_output=True, text=True)
        return r.stdout.strip()
    except Exception:
        return ""


def platform():
    metal = _sh("xcrun", "-sdk", "macosx", "metal", "--version").splitlines()
    return dict(
        machine_model=_sh("sysctl", "-n", "hw.model"),
        chip=_sh("sysctl", "-n", "machdep.cpu.brand_string"),
        # hw.ncpu is CPU cores. The GPU core count this document quotes (20) is NOT this number,
        # and reading it as such would put a wrong figure beside every per-core claim.
        cpu_cores=_sh("sysctl", "-n", "hw.ncpu"),
        gpu_cores="20 (from the campaign's own record; not readable from sysctl)",
        os_product=_sh("sw_vers", "-productVersion"),
        os_build=_sh("sw_vers", "-buildVersion"),
        metal_compiler=(metal[0] if metal else ""),
        xcode=_sh("xcodebuild", "-version").replace("\n", " / "),
        clang=(_sh("clang", "--version").splitlines() or [""])[0],
    )


def main(argv=None):
    global EVIDENCE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", default=None, help="measurement-platform record to read or write")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    # ADDRESSABLE SO A TEST NEED NOT MUTATE THE REAL FILE. Proving a check can fail
    # means corrupting its input, and doing that in place races every reader of it
    # under a parallel runner - the loser leaves the tracked file modified on disk.
    if getattr(a, 'record', None):
        EVIDENCE = getattr(a, 'record')
    now = platform()

    if a.check:
        if not os.path.exists(EVIDENCE):
            print("REFUSED: no recorded platform at %s" % os.path.relpath(EVIDENCE, ROOT))
            return 2
        was = json.load(open(EVIDENCE, encoding="utf-8"))["platform"]
        moved = {k: (was.get(k), now.get(k)) for k in now if was.get(k) != now.get(k)}
        if not moved:
            print("platform matches the one every figure in the machine model was taken on.")
            return 0
        print("THE PLATFORM HAS MOVED. The machine model's figures were taken on:\n")
        for k, (b, n) in sorted(moved.items()):
            print("   %-16s recorded %s" % (k, b))
            print("   %-16s running  %s" % ("", n))
            print("   %-16s bears on %s\n" % ("", BEARING.get(k, "unclassified")))
        print("A difference is not a refutation: encoding and layout claims are unlikely to move,")
        print("timing, power and occupancy are the ones that plausibly do. Re-measure before")
        print("quoting a number from chapters 8, 9 or 25.56 on this machine.")
        return 0

    for k, v in now.items():
        print("  %-16s %s" % (k, v))
    if a.write:
        with open(EVIDENCE, "w", encoding="utf-8") as fh:
            json.dump(dict(platform=now, bears_on=BEARING,
                           note="T9 is blocked, not open: it asks about hardware this campaign does "
                                "not have. This file exists so a reader on a different platform is "
                                "told which claims were never measured there."),
                      fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("\nwrote %s" % os.path.relpath(EVIDENCE, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
