#!/usr/bin/env python3
"""THE CONTRACT BETWEEN THE BACKEND AND THE IMAGE, written down as a type.

Until now a program reached the linker as a bag of positional arguments - a `text` blob, a list of
buffer indices, an entry offset - and everything else was implied. That is workable for one kernel
shape and it is the reason a signature dimension can go missing without anyone noticing: there is no
place in the interface where it would have been absent.

A `Kernel` is what the backend hands over and what this layer promises to turn into a running
image. The division is deliberate and it is the mission's:

    THE BACKEND DECIDES                        THIS LAYER DECIDES
    which instructions, in what order          where __text sits and how long it is
    where its instructions begin and end       (it assembled them; this layer has no decoder)
    which registers                            every section offset, size and alignment
    the constant program's contents            the symbol table and the entry symbol's value
    where a label is, symbolically             what a fixup's displacement resolves to
    which buffers are bound, and their kinds   the binding records and their placement
    what resources the program needs           the metadata class and its layout
    values a class cannot carry                every hash, UUID and archive field

`fixups` is the part that is a promise rather than a description: the backend names a site and a
target symbolically, and this layer resolves it AFTER placement, because only this layer knows where
anything landed. Nothing uses it yet - the programs packaged so far have no cross-section references
- and it is here so that the first one that does has a place to say so rather than a reason to
reach into `text`.
"""
import hashlib


class Fixup:
    """A reference whose value is not known until placement.

    `at` is a byte offset into the section named by `where`; `target` is a symbol or label name;
    `kind` says how the resolved address is written. Resolution is this layer's job.

    `kind="branch"` is the G17 exec-mask branch, and it has rules placement can violate:

        the displacement is measured from the BRANCH'S OWN OFFSET, not from the end of the
            instruction - 14,195 of 14,195 corpus-wide
        it must be EVEN: the field has no weight-one bit
        it is signed and 47 bits wide, so -2^47 <= disp < 2^47

    The parity is the one that will bite, because it depends on where this layer puts things and
    nothing warns you. And there is no rounding that helps: the field's bits are scattered across
    all ten bytes in a shuffled order, so a displacement that is off by one is not a branch to a
    nearby instruction, it is a branch to an arbitrary one. resolve() refuses rather than rounds.
    """
    __slots__ = ("where", "at", "target", "kind", "addend")

    def __init__(self, where, at, target, kind="pcrel", addend=0):
        self.where, self.at, self.target = where, at, target
        self.kind, self.addend = kind, addend

    def __repr__(self):
        return "Fixup(%s+%d -> %s, %s%+d)" % (self.where, self.at, self.target, self.kind,
                                              self.addend)


class Binding:
    """One bound resource. `index` is the buffer index the code refers to.

    `element_type` is the type the signature declares - `float`, `half`, `ulong` and so on. The
    section carries type reflection, so two kernels that bind the same COUNT of buffers with
    different element types are different classes; measured over the corpus, knowing which types
    are present takes structural determination from 78.6% to 86.6%, more than every other
    candidate dimension put together. Counting them is worth nothing - it is knowing which.

    `readonly` is the other half of a binding's kind, and it decides structure rather than
    decorating it: the metadata carries one small record per buffer the kernel WRITES. sw-c_f2h
    writes the float buffer and its table 6 has slot 1 with tlen 12; sw-ad_big writes the uint
    buffer and its table 6 lacks slot 1 with tlen 8; sw-base writes all three and has three such
    records. Supplying it is worth 352 kernels on the class model.
    """
    __slots__ = ("index", "kind", "readonly", "element_type")

    def __init__(self, index, kind="device_buffer", readonly=False, element_type=None):
        self.index, self.kind, self.readonly = index, kind, readonly
        self.element_type = element_type

    def __repr__(self):
        return "Binding(%d, %s%s%s)" % (
            self.index, self.kind, ", readonly" if self.readonly else "",
            ", " + self.element_type if self.element_type else "")


class Kernel:
    """A program as the backend describes it, before anything about the image is decided.

    Required:
        code            the instruction bytes, entry-relative, exactly as they will execute
        bindings        the resources the code refers to, in the order the signature declares
    Optional, each with a reason to exist rather than a default that hides a question:
        name            the function name the pipeline is built from
        entry           where the entry symbol points inside __text; the prologue precedes it
        prologue        bytes before `entry` - a constant-program jump or filler
        constant_program  the second entry point's contents; empty in every kernel measured
        labels          {name: offset into code}, for fixups to target
        fixups          references this layer resolves after placement
        resources       the FULL declared signature, not just the buffers the code binds. A
                        `[(index, kind), ...]` list, and its LENGTH is part of the metadata class
                        key: within the key that bindings alone leave ambiguous - two buffers,
                        starting at zero, one of them constant - two declared resources select a
                        456-byte class and five select a 468-byte one. That is what closed the last
                        two kernels, and it needs no instruction decoding.
        threadgroup_memory  bytes of threadgroup memory the program needs
        semantics       values the metadata carries that its class cannot - two atomic kernels of
                        one class differ in a single word and packaging them from that class gets
                        one wrong. {offset: bytes}. What the word MEANS is not established; that
                        it must be reproduced is
        text_len        pad __text to this length; None means "as long as the code is"
        metadata_structure  (descriptor tables, vectors) the program's metadata needs. Exists
                        because of an executed counterexample, not on principle: c16-originA32
                        matches ac2-32x32x64 on binding count, starting index, binding kinds,
                        declared count AND argument types, and needs eight descriptor tables where
                        ac2 needs nine. Handed the nine-table class it loads, dispatches and
                        computes a different answer than Apple's own archive, and nothing refuses
                        it. Supplied, a mismatch raises before Metal sees the image. It is one
                        integer pair a backend has by construction.
    """

    def __init__(self, code, bindings, name="k", entry=0x40, prologue=None,
                 constant_program=b"", labels=None, fixups=(), threadgroup_memory=0,
                 semantics=None, text_len=None, stats_md=b"", resources=(),
                 metadata_class=None, spans=(), metadata_structure=None,
                 instruction_count=None, loops=None, builtins=None,
                 constant_program_slots=None, argument_types=None):
        self.code = bytes(code)
        self.bindings = [b if isinstance(b, Binding) else Binding(b) for b in bindings]
        self.name, self.entry = name, entry
        self.prologue = prologue
        self.constant_program = bytes(constant_program)
        self.labels = dict(labels or {})
        self.fixups = list(fixups)
        self.threadgroup_memory = int(threadgroup_memory)
        self.resources = list(resources)
        # The metadata class by name, when the backend knows it. The signature-derived key does not
        # determine it - one key holds five shapes across the kernels that share it - so a caller
        # that can see its whole resource set may say which, and one that cannot gets the default.
        self.metadata_class = metadata_class
        self.metadata_structure = tuple(metadata_structure) if metadata_structure else None
        # (offset, length, opcode) for every instruction, as the backend assembled them - OPTIONAL.
        # An assembler has this by construction, and it is `code` described rather than a new fact
        # about the program. This layer cannot recover it: walking G17 machine code needs a decoder
        # and building one is the backend's work, not this layer's.
        #
        # WHAT IT BUYS, measured rather than assumed: NOTHING, now. It bought two kernels while
        # the class key was missing the declared resource count; with that in the key the corpus is
        # 50 of 50 either way - atomics 19 of 19, control flow 29 of 29, bp_ 48 of 48, ad- 23 of 23,
        # with spans and without (NO_SPANS=1 runs the comparison without them). It is kept because a
        # future class may need an instruction-level distinction, and because a backend that has it
        # loses nothing by passing it.
        self.spans = list(spans)
        # TWO SCALARS THE METADATA NEEDS AND THE CODE ALONE DOES NOT GIVE THIS LAYER.
        #
        # `instruction_count` is how many instructions the backend emitted, not how many bytes.
        # The per-kernel table's slot 32 is decided by it - 0 at 30 instructions or fewer, 2 above
        # 300, 3 when the program branches backward, 1 otherwise - both thresholds pinned by
        # compiling kernels one instruction apart across them, not fitted to a corpus gap,
        # where the best byte-length thresholds reach only 98.5% and 99.5%. An assembler knows the
        # count by construction; this layer would need a decoder to recover it, and building one
        # is the backend's work.
        #
        # `loops` is whether the program branches BACKWARD - which is also what makes a kernel
        # unsafe to dispatch with foreign buffer contents, so it is worth carrying for two
        # reasons. Slot 33 is present exactly when it is true.
        #
        # Both are derived from `spans` when a backend passes those and omits these, so an
        # assembler that already reports its instruction boundaries need say nothing more.
        self._instruction_count = instruction_count
        self._loops = loops
        # THREE MORE THINGS THE SECTION CARRIES THAT A BACKEND KNOWS AND THIS LAYER CANNOT DERIVE.
        #
        # `builtins` are the ids of the builtins the body consumes, as a tuple:
        # threadgroup_position_in_grid is 0, thread_position_in_grid is 80, a simd builtin adds 58.
        # They are what the CODE uses, not what the signature declares - mp-smoothstep declares
        # threadgroup_position_in_grid and consumes none - and the per-kernel table's tail is
        # 8 + 4 * len(builtins) bytes holding exactly them. They are NOT the read_sr immediates:
        # over 700 decoded kernels the two agree in 53 and differ in 647.
        #
        # `constant_program_slots` is the count at the head of the constant-program tail, a vector
        # of N u32s holding 0..N-1 with N in {0, 4, 8, 12}. It separates shapes that are otherwise
        # identical: two kernels in the (2 bound, at zero, no constant, float+half) group differ
        # only in an 84- against a 100-byte tail, which is four more list entries.
        #
        # `argument_types` is the set of element types the signature declares, for callers that
        # would rather state it once than set it on every Binding.
        self.builtins = tuple(builtins) if builtins is not None else None
        self.constant_program_slots = constant_program_slots
        self._argument_types = tuple(sorted(argument_types)) if argument_types else None
        # Displacements this layer worked out after placement, {(where, at): value}, for the
        # encoder to write. Filled by resolve().
        self.resolved = {}
        self.semantics = dict(semantics or {})
        self.text_len, self.stats_md = text_len, bytes(stats_md)

    @property
    def instruction_count(self):
        """How many instructions the backend emitted. None when it did not say and cannot be
        derived - and None is a refusal to guess, not a zero."""
        if self._instruction_count is not None:
            return int(self._instruction_count)
        return len(self.spans) or None

    @property
    def argument_types(self):
        """The set of element types the signature declares, from the bindings or stated directly."""
        if self._argument_types is not None:
            return self._argument_types
        t = {b.element_type for b in self.bindings if b.element_type}
        return tuple(sorted(t)) if t else None

    @property
    def written(self):
        """The bindings the kernel writes, sorted - the access half of a binding's kind."""
        return tuple(sorted(b.index for b in self.bindings if not b.readonly))

    def contract_key(self):
        """The key the linker looks a metadata class up by, or None if the backend did not say.

        Everything in it is something a backend has by construction, and NONE of it is an
        instruction walk by this layer: the signature, the constant program's LENGTH (its bytes are
        stricter than the shape needs and cost witnesses), the count at the head of its tail, the
        size class - which is slot 32's own value, bucketed from the instruction count - whether
        the program loops, the builtins it consumes, and its threadgroup allocation.

        Returns None rather than a partial key. A key with a field guessed is a class chosen by
        guess, and the whole point of the gate is that it refuses instead.
        """
        from . import classgen as g17classgen
        if self.instruction_count is None or self.loops is None:
            return None
        if self.builtins is None or self.constant_program_slots is None:
            return None
        types = self.argument_types
        if types is None or not self.resources:
            return None
        idx = self.buffer_indices
        sig = (len(idx), min(idx) == 0 if idx else False,
               any(b.kind == "constant" for b in self.bindings), len(self.resources),
               tuple(types), tuple(sorted({b.kind for b in self.bindings})))
        return g17classgen.contract_key(sig, self.constant_program, self.constant_program_slots,
                                        self.instruction_count, self.loops, self.builtins,
                                        self.threadgroup_memory, written=self.written)

    @property
    def loops(self):
        """Whether the program branches backward. None when unknown."""
        if self._loops is not None:
            return bool(self._loops)
        if not self.spans:
            return None
        from . import cf as g17cf
        for off, ln, op in self.spans:
            if op in getattr(g17cf, "BRANCH_OPCODES", (450, 458, 462)):
                try:
                    if g17cf._disp(self.code[off:off + ln]) < 0:
                        return True
                except Exception:
                    return None
        return False

    @property
    def buffer_indices(self):
        return [b.index for b in self.bindings]

    def digest(self):
        """A stable identity for this program, used where the image needs one."""
        h = hashlib.sha256()
        h.update(self.code); h.update(self.name.encode())
        h.update(bytes(str(self.buffer_indices), "ascii"))
        return h.digest()

    def __repr__(self):
        return ("Kernel(%r, %d bytes, bindings %s%s%s)"
                % (self.name, len(self.code), self.buffer_indices,
                   ", %d fixups" % len(self.fixups) if self.fixups else "",
                   ", tgmem %d" % self.threadgroup_memory if self.threadgroup_memory else ""))


FILLER = bytes.fromhex("0600")
PROLOGUE_WORD = bytes.fromhex("0e000000")


def text_of(k):
    """__text for a kernel: the prologue, the code at `entry`, and padding to `text_len`.

    The prologue defaults to an `end` word followed by filler, which is what every object measured
    has before its entry point, and the padding is filler rather than zeros because zeros decode.
    """
    pro = k.prologue
    if pro is None:
        pro = PROLOGUE_WORD + FILLER * ((k.entry - len(PROLOGUE_WORD)) // 2)
    if len(pro) != k.entry:
        raise ValueError("the prologue is %d bytes and the entry is at %d" % (len(pro), k.entry))
    text = pro + k.code
    want = k.text_len if k.text_len is not None else ((len(text) + 15) & ~15)
    if want < len(text):
        raise ValueError("text_len %d is shorter than the %d bytes of program" % (want, len(text)))
    return text + FILLER * ((want - len(text)) // 2)


def resolve(k, text, placement):
    """Write every fixup's value now that placement is known. -> the patched text.

    `placement` maps a section name to its address. A fixup whose target is a label resolves within
    __text; anything else must name a section. Unresolvable is an error, not a zero.
    """
    if not k.fixups:
        return text
    out = bytearray(text)
    for f in k.fixups:
        if f.target in k.labels:
            addr = k.entry + k.labels[f.target]
        elif f.target in placement:
            addr = placement[f.target]
        else:
            raise ValueError("%r targets %r, which is neither a label nor a placed section"
                             % (f, f.target))
        site = k.entry + f.at if f.where == "code" else f.at
        if f.kind not in ("pcrel", "absolute", "branch"):
            raise ValueError("%r has no rule for kind %r" % (f, f.kind))
        value = addr if f.kind == "absolute" else addr - site
        value += f.addend
        if f.kind == "branch":
            if value % 2:
                raise ValueError("%r resolves to an ODD displacement of %d, which the branch field "
                                 "cannot encode - it has no weight-one bit, and there is no nearby "
                                 "value to round to because its bits are shuffled across ten bytes"
                                 % (f, value))
            if not -(1 << 47) <= value < (1 << 47):
                raise ValueError("%r resolves to %d, outside the 47-bit signed displacement"
                                 % (f, value))
            # THE BYTES BELONG TO THE ENCODER, not to this file: the displacement's bits are
            # scattered across ten bytes in a shuffled order and only the specification knows
            # where. This layer's job is to say what the value IS, once placement is known, and to
            # refuse the values the field cannot hold. The resolved displacement is recorded for
            # whoever writes it.
            k.resolved[(f.where, f.at)] = value
            continue
        out[site:site + 4] = (value & 0xFFFFFFFF).to_bytes(4, "little")
    return bytes(out)
