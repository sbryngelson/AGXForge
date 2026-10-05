#!/usr/bin/env python3
"""What GPU this toolchain is pointed at, in one place.

Everything else here recovers an instruction set by measurement rather than by knowing it in
advance, which means the method is not specific to one GPU - but the TARGET is, and it was
written into seven files as a literal. Retargeting to another Apple GPU then meant finding all
seven, and finding all seven is exactly the kind of thing nobody does correctly the first time.

    AGXFORGE_ARCH=applegpu_g16p python3 tools/g17corpus.py --build

The Mach-O cpu type and subtype are the same fact in a different form: Apple's own objects carry
them, and a different GPU carries different ones. They are read from the environment for the same
reason.

WHAT IS AND IS NOT PORTABLE. The recovery method is: compile probes with Apple's own toolchain,
decode them with Apple's own disassembler, measure each operand field by FLIPPING its bits rather
than fitting a corpus, certify every map in both directions, and prove the encoder runs with the
disassembler removed. None of that knows anything about G17. What does know: the opcode numbers
this project has named, the control-bit positions it located, and the two constants below.
"""
import os, subprocess, sys, tempfile

# The slice name `xcrun metal-lipo -thin` selects, and the one Apple's fat objects carry.
ARCH = os.environ.get("AGXFORGE_ARCH") or "applegpu_g17s"

# The chip the arch belongs to, for the records this project writes.
CHIP = os.environ.get("AGXFORGE_CHIP") or "H17s"

# From Apple's own objects for that arch. A different GPU has different ones, and reading a
# foreign object with these silently produces nonsense rather than an error.
CPUTYPE = int(os.environ.get("AGXFORGE_CPUTYPE") or 16777235)
CPUSUBTYPE = int(os.environ.get("AGXFORGE_CPUSUBTYPE") or 355)


def discover():
    """Ask this machine what it is, rather than being told.

    The arch name is NOT reported by anything that describes the hardware. `sysctl hw.model` says
    Mac17,8, system_profiler says Apple M5 Pro, and the AGX driver's IORegistry entry names
    neither an architecture nor a generation. The only thing that says `applegpu_g17s` is the
    TOOLCHAIN, and only after it has archived something: a compiled .metallib is AIR
    (air64_v28) and carries no native slice at all, so compiling is not enough.

    So compile the smallest possible kernel, archive it, and read the slice list. Anyone
    retargeting this toolchain to another Apple GPU can run it and be told the name to set.

        python3 tools/g17target.py
    """
    import ctypes
    src = "#include <metal_stdlib>\nusing namespace metal;\n" \
          "kernel void k(device uint *u [[buffer(0)]]) { u[0] = 1u; }\n"
    d = tempfile.mkdtemp(prefix="g17target-")
    open(d + "/s.metal", "w").write(src)
    r = subprocess.run(["xcrun", "metal", "-o", d + "/s.metallib", d + "/s.metal"],
                       capture_output=True, text=True)
    if r.returncode:
        return None, "the Metal compiler refused a trivial kernel: %s" % r.stderr.strip()[:80]
    try:
        from agxforge.g17 import corpus as g17corpus
        L = ctypes.CDLL(g17corpus.LIBACCEL)
        L.ac_init()
        if L.ac_lib_from_data((d + "/s.metallib").encode()) != 0:
            return None, "the library would not load"
        if L.ac_archive(b"k", (d + "/s.arc.metallib").encode()) != 0:
            return None, "the archive step failed"
    except Exception as e:
        return None, "no archiver available here: %r" % (e,)
    info = subprocess.run(["xcrun", "metal-lipo", "-info", d + "/s.arc.metallib"],
                          capture_output=True, text=True).stdout
    slices = info.split(":")[-1].split()
    native = [s for s in slices if not s.startswith("air")]
    return (native[0] if native else None), info.strip()


def main(argv=None):
    """The command-line entry, callable. It was inline under `if __name__`, so after the
    move neither the compatibility module nor the library could reach it: a shim IMPORTS
    this file, it does not execute it, and the result is an entry point that exits zero
    having done nothing. Sixth, seventh and eighth instance of that shape in this
    migration, which is why it is now the first thing checked per module.
    """
    argv = list(sys.argv if argv is None else argv)
    name, note = discover()
    print("hw.model              %s" % subprocess.run(["sysctl", "-n", "hw.model"],
                                                      capture_output=True,
                                                      text=True).stdout.strip())
    print("cpu                   %s" % subprocess.run(["sysctl", "-n",
                                                       "machdep.cpu.brand_string"],
                                                      capture_output=True,
                                                      text=True).stdout.strip())
    print("configured ARCH       %s   (AGXFORGE_ARCH overrides)" % ARCH)
    print("this machine builds   %s" % (name or "could not tell"))
    print("   %s" % note)
    if name and name != ARCH:
        print("\n   MISMATCH: set AGXFORGE_ARCH=%s to point the toolchain at this machine" % name)


if __name__ == "__main__":
    main()
