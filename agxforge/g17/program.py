#!/usr/bin/env python3
"""THE PROGRAM IMAGE AS ONE OBJECT, with every byte's provenance recorded.

Until now the image was assembled by whichever probe needed it: each one built its own object,
picked its own metadata, and spliced or rebuilt an archive around it. That works and it hides the
only number that matters for finishing this - HOW MUCH OF THE DELIVERED IMAGE IS STILL APPLE'S.

A G17Program carries the whole image and can say, byte for byte, where each part came from:

    authored     generated from the kernel by this compiler - the code, the bindings
    generated    structure this file computes from the payload: headers, offsets, symbol tables.
                 Ours, but not kernel-specific, and it would be dishonest to count it as either
                 a borrowing or a semantic achievement.
    class        a constant that EXECUTION shows is per-class rather than per-kernel. Recorded as
                 its own kind because the distinction is a measurement
                 (ledger/g17-image-region-dependence.toml), not a convenience.
    zeroed       proven inert by blanking it and running the kernel, so it is emitted as zeros
    inherited    still Apple's bytes, and the whole of what remains to be eliminated

The census is the progress metric for the image half of this compiler, and it is deliberately
pessimistic: anything not yet proven ours counts as inherited.
"""
import hashlib, os, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))

from . import link as g17link
from . import obj as g17obj
from . import arc as g17arc
from . import container as g17container
from . import mdgen as g17mdgen
from . import ldmd as g17ldmd
from . import mtlb as g17mtlb

AUTHORED, GENERATED, CLASS, ZEROED, INHERITED = (
    "authored", "generated", "class", "zeroed", "inherited")
ORDER = (AUTHORED, GENERATED, CLASS, ZEROED, INHERITED)


class Region:
    __slots__ = ("name", "kind", "size", "note")

    def __init__(self, name, kind, size, note=""):
        self.name, self.kind, self.size, self.note = name, kind, size, note

    def __repr__(self):
        return "%-22s %-10s %6d  %s" % (self.name, self.kind, self.size, self.note)


class G17Program:
    """One kernel's complete native image.

    Constructed from: the machine code this compiler emitted, the kernel's buffer list, the entry
    offset, and a metadata class. Everything else is computed here or named as a borrowing.
    """

    def __init__(self, text, entry, buffers, q=None, stats_md=b"", remarks=b"",
                 constant_program=b"", text_len=None, obj_size=None, host=None,
                 air=None, metallib=None, function_hash=None, name="k", md_patch=None,
                 binding_kinds=None, md_counts=None, md_class=None, memory_opcodes=(),
                 spans=(), declared=None, md_structure=None, md_register_count=None):
        self.text = bytes(text)
        self.entry = entry
        self.buffers = list(buffers)
        self.q = g17mdgen.Q_SCALAR if q is None else q
        self.stats_md = bytes(stats_md)
        self.remarks = bytes(remarks)
        # THE CONSTANT PROGRAM is a second entry point, _agc.main.constant_program, and every
        # kernel this compiler has emitted has an empty one. It is a field here rather than an
        # assumption so that a kernel needing one has somewhere to put it.
        self.constant_program = bytes(constant_program)
        # THE TWO AIR COPIES ARE OURS BY DEFAULT. Every region of Apple's metallib was zeroed one
        # at a time with the kernel still correct - the bitcode, the function hash, the UUID, the
        # reflection and source lists, even the function's name - so the loader wants a
        # metallib-SHAPED file, not that kernel's AIR. g17mtlb.build() writes the smallest thing
        # the format describes; passing `air=`/`metallib=` is how a borrowed one is put back for a
        # comparison (spike/accel/re/airmin.py).
        # THE ONE FIELD THE LOADER DEMANDS is the source library's function hash, and it is a
        # CACHE KEY rather than a field that merely has to be non-zero: the container's AIR_MODULE
        # and AIR_HASHES tables carry it, and a pipeline built from that library finds its compiled
        # form by matching it. A hash of our own choosing there gets no pipeline while the
        # library's own runs, and swapping which side holds which decides it - the container must
        # agree with the library, the metallib copies are free (spike/accel/re/hashkey.py).
        #
        # That made it an identity of the INPUT until the input became ours too: Metal's
        # newLibraryWithData accepts the same 360-byte metallib g17mtlb writes for the archive,
        # hands back an MTLFunction named `k`, and the archive supplies its compiled form. So the
        # hash is chosen here, from the program this compiler emitted, and `function_hash=` is now
        # only for building against a library someone else wrote.
        self.name = name
        # Values the metadata carries that belong to the PROGRAM rather than to its shape - the
        # identity of an atomic reduction is the one measured so far. {offset: bytes}.
        self.md_patch = dict(md_patch or {})
        # The address space of each bound argument, in the order the signature declares them. A
        # `constant` argument changes the shape of the first binding record, so this is part of
        # the metadata class key rather than decoration.
        self.binding_kinds = list(binding_kinds or [])
        # The two counts the first vector's record carries, {slot: value}, from the signature.
        self.md_counts = dict(md_counts or {})
        # The metadata class is structural; slot 0 is per-program and comes from the compiler ABI.
        self.md_register_count = md_register_count
        # A caller that KNOWS which metadata class its kernel needs may name it. The derived key -
        # binding count, starts-at-zero, is anything constant - does not determine the class: that
        # key holds five distinct shapes across the 40 kernels that share it, so picking one modally
        # only moves which kernels are wrong.
        self.md_class = md_class
        # The memory instructions the program uses, as the backend named them. Within one signature
        # key this is what separates the metadata shapes, so it is a derivation from the code
        # rather than a fact the backend has to look up.
        self.memory_opcodes = sorted(memory_opcodes or ())
        # (offset, length, opcode) as the backend assembled them, entry-relative.
        self.spans = list(spans or ())
        # How many resources the SIGNATURE declares, as against how many the code binds. Within an
        # otherwise ambiguous key this is what separates the metadata shapes.
        self.declared = declared
        # (descriptor tables, vectors) the program's metadata needs, when the backend knows. See
        # Kernel.metadata_structure: it exists because a signature-selected class computed a wrong
        # answer for c16-originA32 and nothing refused it.
        self.md_structure = md_structure
        self.function_hash = (hashlib.sha256(self.text + bytes(str(self.buffers), "ascii")).digest()
                              if function_hash is None else bytes(function_hash))
        self.air = g17mtlb.build(name=name, hash=self.function_hash) if air is None else bytes(air)
        self.metallib = (g17mtlb.build(name=name, hash=self.function_hash) if metallib is None
                         else bytes(metallib))
        self.text_len = len(self.text) if text_len is None else text_len
        self.obj_size = obj_size
        self.host = host
        self._obj = None
        self._image = None

    @classmethod
    def from_kernel(cls, k, **kw):
        """Build from the backend's own description (tools/g17link.Kernel).

        This is the interface the mission names: the backend supplies code, a constant program,
        labels, symbolic fixup targets, bindings, resource requirements and the values only
        semantics can give; this layer lays everything out and resolves the references afterwards.
        Every argument this constructor used to take positionally now has a name on that side.
        """
        text = g17link.text_of(k)
        text = g17link.resolve(k, text, {})       # __text is the only placed section so far
        return cls(text=text, entry=k.entry, buffers=k.buffer_indices, name=k.name,
                   constant_program=k.constant_program, stats_md=k.stats_md,
                   md_patch=k.semantics, binding_kinds=[b.kind for b in k.bindings],
                   md_class=k.metadata_class,
                   md_structure=getattr(k, "metadata_structure", None),
                   memory_opcodes=[op for _o, _l, op in k.spans], spans=k.spans,
                   declared=(len(k.resources) or None), **kw)

    # --- the object -----------------------------------------------------------------------
    def metadata(self):
        """__GPU_METADATA for this signature, composed rather than taken from a measured class.

        The two hand-measured layouts have exactly two binding records each, so a kernel binding
        one or three buffers had nowhere to be described. The schema-driven path builds the
        document from the signature - tables, vectors and typed references, placed by
        g17schema - and handles any count.
        """
        # THE MEASURED CLASS LAYOUT IS THE DEFAULT WHERE IT APPLIES, because the corpus says so:
        # packaged with the class's own positions, 12 of 14 cached kernels with an observable
        # compute exactly what Apple's archive computes, and re-laying the binding region drops
        # that to 3. The store-only probe that validated re-laying could not see the difference -
        # one slot, one value - which is what a degenerate observable looks like.
        path = os.environ.get("MD_PATH", "auto")
        if path == "auto":
            # The RECORDED class first, at every binding count. Preferring the hand-written SCALAR
            # layout for two bindings meant the two-binding kernels never saw the recorded variant
            # that carries the extra per-kernel slot, which is what kept b45p_prev_load wrong.
            # A patch this layer can compute from the program beats one it was handed: the atomic
            # reduction's identity is the operation code, and the operation code is in the bytes.
            patch = dict(self.md_patch)
            if self.spans:
                sized = g17mdgen.for_class(self.buffers, kinds=self.binding_kinds,
                                           want=self.md_class,
                                           memory_opcodes=self.memory_opcodes,
                                           declared=self.declared)
                if sized is not None:
                    patch.update(g17mdgen.derived_patch(self.text[self.entry:], self.spans,
                                                        len(sized)))
            recorded = g17mdgen.for_class(self.buffers, patch=patch,
                                          kinds=self.binding_kinds, counts=self.md_counts,
                                          want=self.md_class,
                                          memory_opcodes=self.memory_opcodes,
                                          declared=self.declared,
                                          structure=self.md_structure,
                                          register_count=self.md_register_count)
            if recorded is not None:
                return recorded
            path = "class" if len(self.buffers) == len(g17mdgen.SCALAR["bind"]) else "relaid"
        if path == "graph":
            return g17mdgen.build_graph(self.buffers)
        if path == "class":
            return g17mdgen.build(self.buffers, self.q,
                                  register_count=self.md_register_count)
        return g17mdgen.build(self.buffers, layout=g17mdgen.for_bindings(self.buffers),
                              register_count=self.md_register_count)

    def object(self):
        """The native Mach-O object: header, load commands, six sections, symbol table."""
        if self._obj is None:
            md = self.metadata()
            ld, arch = g17ldmd.build(entry=self.entry), g17ldmd.build_arch()
            if self.obj_size is None:
                self._obj = g17obj.build(self.text, md, ld, arch, self.stats_md,
                                         entry=self.entry, remarks=self.remarks)
            else:
                for pad in range(0, 256, 8):
                    cand = g17obj.build(self.text, md, ld, arch, self.stats_md,
                                        entry=self.entry, remarks=self.remarks, extra_pad=pad)
                    if len(cand) == self.obj_size:
                        self._obj = cand; break
                if self._obj is None:
                    raise ValueError("no padding gives the requested object size %d" % self.obj_size)
        return self._obj

    def library(self):
        """The MTLLibrary this program is compiled for - `newLibraryWithData` takes it directly.

        It declares the function's name and the hash the archive is keyed by, and nothing else: the
        library parser accepts it with no bitcode at all, and the compiled form comes from the
        archive.
        """
        return g17mtlb.build(name=self.name, hash=self.function_hash)

    # --- the image ------------------------------------------------------------------------
    def check(self):
        """Every relation a finished image has to satisfy, read back out of the delivered bytes.

        Independent of the builders: g17verify re-parses the archive rather than asking this file
        where anything is. Malformed metadata segfaults the process that loads it, so this is the
        difference between a message and a crash - it caught a binding vector whose third record
        offset had been written over the second vector's count, which the loader answered with a
        SIGSEGV.
        """
        from . import verify as g17verify
        r = g17verify.verify(self.image(verify=False), self.library())
        r.extend(g17verify.verify_metadata(self.metadata(), self.buffers))
        return r

    def image(self, air=None, metallib=None, verify=True):
        """The whole archive, from payloads alone - no host archive is opened to build it. `air`
        and `metallib` override this program's two metallib copies, which is how their content is
        tested rather than assumed."""
        img = g17arc.emit(self.object(), self.air if air is None else air,
                          self.metallib if metallib is None else metallib,
                          obj_size=len(self.object()), module_hash=self.function_hash)
        if verify:
            from . import verify as g17verify
            r = g17verify.verify(img, self.library())
            if r:
                raise ValueError("this image would not have loaded:\n%s" % r)
        return img

    # --- provenance -----------------------------------------------------------------------
    def regions(self):
        """Every part of the delivered image, with where it came from."""
        obj = self.object()
        md = self.metadata()
        ld, arch = g17ldmd.build(entry=self.entry), g17ldmd.build_arch()
        objhdr = len(obj) - (len(self.text) + len(md) + len(ld) + len(arch)
                             + len(self.stats_md) + len(self.remarks))
        out = [
            Region("__text", AUTHORED, len(self.text),
                   "the kernel's instructions and the filler that pads them"),
            ]
        # __GPU_METADATA, split by who decided each byte. The recorded class is structure measured
        # once and re-emitted; the binding indices and the two symbol names are this program's; and
        # what is left is bytes the walk does not describe and nobody can name, which are Apple's
        # until they are.
        residue = g17mdgen.class_residue(self.buffers, self.binding_kinds, self.md_class)
        names = sum(len(n) for n in g17mdgen.SYMBOLS)
        mine = 4 * len(self.buffers) + names
        out += [
            Region("__GPU_METADATA", AUTHORED, mine,
                   "one buffer index per binding, and the entry and constant-program symbol names"),
            Region("  the class structure", CLASS, len(md) - mine - residue,
                   "tables, vectors and slot maps measured once and re-emitted for this signature"),
            Region("  undescribed bytes", INHERITED, residue,
                   "load-bearing - dropping them makes the corpus disagree - and unexplained"),
            Region("__GPU_LD_MD", GENERATED, len(ld), "computed from the entry PC"),
            Region("__GPU_ARCH_LD_MD", GENERATED, len(arch), "structure around an inert field"),
            Region("__GPU_STATS_MD", ZEROED, len(self.stats_md),
                   "blanking it leaves the kernel correct"),
            Region("__GPU_REMARKS_MD", ZEROED, len(self.remarks), "empty"),
            Region("object header+symtab", GENERATED, objhdr,
                   "mach header, load commands, section table, symbols, padding"),
        ]
        out.append(Region("container __reflection", ZEROED, g17arc.REFLECTION_LEN,
                          "zeroing it leaves the kernel correct"))
        out.append(Region("container __descriptor", ZEROED, g17arc.DESCRIPTOR_LEN,
                          "zeroing it leaves the kernel correct"))
        out.append(Region("container __metallib", GENERATED, len(self.metallib),
                          "built from the format by g17mtlb; no Apple bytes"))
        out.append(Region("slice 0 metallib", GENERATED, len(self.air),
                          "built from the format by g17mtlb; no Apple bytes"))
        # The container's head, split by who decided each byte. g17container writes all of it -
        # it reproduces the host's block byte for byte on 7,468 of the 7,491 archives in the cache
        # - but writing a constant is not the same as choosing it, so the values that are the same
        # in every archive are counted as class rather than claimed.
        head_len = g17container.HEAD_LEN
        # Only the bytes execution shows are READ count as class; the rest were zeroed one at a
        # time with the kernel still correct, so they are emitted as zeros and counted as zeroed.
        CMD_CLASS, CMD_INERT = 16, len(g17container.BUILD_CMD) - 16
        TBL_CLASS, TBL_INERT = 56, 60
        img = len(self.image())
        counted = sum(r.size for r in out)
        out.append(Region("container header+cmds", GENERATED,
                          g17container.TABLES_OFF - CMD_CLASS - CMD_INERT,
                          "mach header, twelve load commands, every offset from the payload sizes"))
        out.append(Region("  the build record", CLASS, CMD_CLASS,
                          "the four words of LC 0x32 that execution shows are read"))
        out.append(Region("  its inert half", ZEROED, CMD_INERT,
                          "four words including the one holding an OS version"))
        out.append(Region("container __AIR_DATA", GENERATED,
                          head_len - g17container.TABLES_OFF - TBL_CLASS - TBL_INERT - 64,
                          "eight table descriptors and their entries, computed from the layout"))
        out.append(Region("  class constants", CLASS, TBL_CLASS,
                          "SHA-256 of empty content, value-checked, and six words that are read"))
        out.append(Region("  constants proven inert", ZEROED, TBL_INERT,
                          "the descriptor key and seven words, each zeroed with the kernel correct"))
        out.append(Region("  library function hash", GENERATED, 64,
                          "twice here; the archive's cache key, and the library it keys is ours"))
        if img > counted + head_len:
            out.append(Region("fat header + alignment", GENERATED, img - counted - head_len,
                              "the magic, the slice table and the padding, computed from the sizes"))
        return out

    def census(self):
        c = dict((k, 0) for k in ORDER)
        for r in self.regions():
            c[r.kind] = c.get(r.kind, 0) + r.size
        return c

    def report(self, out=sys.stdout):
        regs = self.regions()
        total = sum(r.size for r in regs)
        print("G17Program image: %d bytes" % total, file=out)
        for r in regs:
            if r.size:
                print("   %s" % r, file=out)
        c = self.census()
        print("   %s" % ("-" * 66), file=out)
        for k in ORDER:
            if c.get(k):
                print("   %-12s %6d  %5.1f%%" % (k, c[k], 100.0 * c[k] / total), file=out)
        ours = total - c.get(INHERITED, 0)
        print("   OURS %d of %d bytes (%.1f%%); INHERITED %d (%.1f%%)"
              % (ours, total, 100.0 * ours / total, c.get(INHERITED, 0),
                 100.0 * c.get(INHERITED, 0) / total), file=out)
        return c
