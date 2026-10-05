#!/usr/bin/env python3
"""G17Image: the explicit program-image model.

The mission asks for "an explicit G17Program model containing code, constant program, descriptors,
launch metadata, tensor setup, and relocations, even if some fields initially come from an
Apple-derived container", and then for those dependencies to be replaced one by one.

The point of the model is the ACCOUNTING. "We patched an Apple metallib" hides how much of the
image is ours; this makes every region either GENERATED or INHERITED, with a byte count, so
"complete native program image" becomes a number that can be driven to 100% instead of a
milestone that is either declared or not.

    region              what it is                              today
    constant_program    _agc.main.constant_program, [0, entry)  inherited
    prologue            [entry, window_start)                   inherited
    code                the compiler's own instruction stream   GENERATED
    tail                [window_end, end of __text)             inherited
    descriptors         argument/buffer tables                  inherited, NOT YET LOCATED
    launch_metadata     threadgroup sizes and entry info        inherited, NOT YET LOCATED
    tensor_setup        the tensor bound/setup preamble         inherited
    relocations         none observed                           n/a

"NOT YET LOCATED" is deliberately distinct from "inherited": a region we take wholesale but could
point at is a smaller problem than one we have never isolated, and collapsing the two would make
the container look better understood than it is.
"""
import os, sys
# NO AMBIENT PATH MUTATION: machobj and agxdis are package modules now. The inserts this carried
# resolved to the repository's spike tree from tools/ and would resolve to agxforge/spike from here -
# a path that does not exist - and a library that edits sys.path on import decides what its callers
# can import.
from . import machobj
from . import agxdis

FILLER = bytes.fromhex("0600")     # the compiler's own NOP filler; a tracked dependency

class G17Image:
    def __init__(self, reference, name="k"):
        arc, obj = reference + "/s.arc.metallib", reference + "/out/object/0-0"
        self.reference = reference; self.name = name
        self.loc = machobj.locate(arc, obj)
        f, sz = agxdis.sections(self.loc["obj"])
        self.text_off_in_obj = f; self.text = bytearray(self.loc["obj"][f:f+sz])
        self.cprog_inherited_head = None
        self.entry = self.loc["syms"]["_agc.main"]
        self.window = None          # (start, length) of the region the compiler owns
        self.code = b""
        self.cprog_generated = False

    # The constant program is EXACTLY `end` followed by filler in 154 of 941 cached kernels, all
    # 64 bytes. Both pieces are known semantics - `end` is causal in isa/g17-scalar-isa.toml and
    # 06 00 is the compiler's own filler - so for those kernels this region is GENERATABLE and
    # need not be inherited at all. The other dominant form (711 kernels) prefixes 8 bytes of
    # real instructions, a 6-byte class-3 and a 2-byte class-0, which are not modelled; asking
    # for a generated constant program on such a container raises rather than guessing them.
    TRIVIAL_CPROG_HEAD = bytes.fromhex("0e000000")

    # The one head that 849 of 1083 catalogued kernels share, byte for byte. It appears exactly
    # when the kernel really loads from a buffer - the 157 containers with an EMPTY constant
    # program are the ones whose load was dead-code-eliminated - so it is buffer-binding setup,
    # invariant across every shape, buffer count and program in the corpus. Unmodelled, and
    # therefore emitted verbatim and COUNTED, not silently inherited.
    COMMON_CPROG_HEAD = bytes.fromhex("2300070242 20a002".replace(" ", ""))

    def generate_constant_program(self, allow_common_head=True):
        """Author the constant program instead of inheriting it.

        Everything from the terminating `end` onwards is authored: that is the whole 64-byte
        region for a container whose constant program is empty, and 56 of 64 for one carrying the
        common 8-byte head. The head itself is unmodelled; with allow_common_head it is emitted
        verbatim and recorded in cprog_inherited_head, and any OTHER head raises rather than being
        copied. ledger/g17-constant-program-authored.toml
        """
        n = self.entry
        have = bytes(self.text[:n])
        head = b""
        if not have.startswith(self.TRIVIAL_CPROG_HEAD):
            if allow_common_head and have.startswith(self.COMMON_CPROG_HEAD):
                head = self.COMMON_CPROG_HEAD
            else:
                raise ValueError("this container's constant program is neither empty nor the "
                                 "common head - it prefixes unmodelled instructions (%s). "
                                 "Generating it would mean inventing them." % have[:8].hex(" "))
        want = head + self.TRIVIAL_CPROG_HEAD
        want += FILLER * ((n - len(want)) // 2)
        if len(want) != n:
            raise ValueError("constant program region %d is not a whole number of fillers" % n)
        if bytes(self.text[:n]) != want:
            raise ValueError("authored constant program differs from this container's: %s vs %s"
                             % (want[:12].hex(" "), have[:12].hex(" ")))
        self.text[:n] = want
        self.cprog_generated = True
        self.cprog_inherited_head = len(head)
        return self

    def place(self, code, at):
        """Install a generated instruction stream at `at` (an offset within __text).

        The window is padded with the compiler's own filler to a whole number of instructions so
        the surrounding stream stays framed; the padding is counted as GENERATED because we chose
        it, but the filler ENCODING is an inherited two-byte form and is declared as such.
        """
        if at < self.entry:
            raise ValueError("window starts at 0x%x, before the entry symbol 0x%x - that is the "
                             "constant program, where NOP is invalid" % (at, self.entry))
        if at + len(code) > len(self.text):
            raise ValueError("window overruns __text")
        self.code = bytes(code); self.window = (at, len(code))
        return self

    def pad_to(self, length):
        if length < len(self.code): raise ValueError("cannot shrink generated code")
        n = length - len(self.code)
        if n % len(FILLER): raise ValueError("pad %d is not a whole number of fillers" % n)
        self.code += FILLER * (n // len(FILLER))
        self.window = (self.window[0], len(self.code))
        return self

    def materialise(self):
        """The full container bytes, with the generated window spliced in."""
        if self.window is None: raise ValueError("no code placed")
        fat = bytearray(self.loc["fat"])
        base = self.loc["base"] + self.text_off_in_obj + self.window[0]
        fat[base:base + len(self.code)] = self.code
        return bytes(fat)

    def regions(self):
        n = len(self.text); ws, wl = self.window if self.window else (n, 0)
        return [("constant_program", 0, self.entry,
                 "GENERATED" if self.cprog_generated else "inherited"),
                ("prologue",         self.entry, ws, "inherited"),
                ("code",             ws, ws + wl, "GENERATED"),
                ("tail",             ws + wl, n, "inherited")]

    def report(self):
        gen = sum(b - a for _, a, b, k in self.regions() if k == "GENERATED")
        tot = len(self.text)
        # PADDING IS NOT CONTENT. Generating a whole fixed-size main program means emitting the
        # real instructions and then filling the rest with NOPs, and counting that fill as
        # "generated" would let the number be inflated by choosing a bigger container. Both are
        # reported: the region the compiler OWNS, and the semantic content inside it.
        fill = self.code.count(FILLER) * len(FILLER) if self.code else 0
        real = gen - fill
        out = ["G17Image %s  reference=%s" % (self.name, os.path.basename(self.reference)),
               "  __text %d bytes" % tot]
        for nm, a, b, kind in self.regions():
            if b > a:
                out.append("    %-16s [0x%04x,0x%04x)  %5d bytes  %s" % (nm, a, b, b - a, kind))
        out.append("  container regions not in __text, inherited and NOT YET LOCATED:")
        out.append("    descriptors, launch_metadata")
        out.append("  __text GENERATED: %d of %d bytes (%.1f%%)  of which %d bytes are NOP "
                   "padding to fit the container" % (gen, tot, 100.0 * gen / tot, fill))
        out.append("  semantic content generated: %d of %d bytes (%.1f%%)"
                   % (real, tot, 100.0 * real / tot))
        out.append("  instructions in the main program inherited from this container: 0")
        out.append("  whole container: %d of %d bytes (%.2f%%)"
                   % (gen, len(self.loc["fat"]), 100.0 * gen / len(self.loc["fat"])))
        return "\n".join(out)
