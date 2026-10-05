"""Immutable version-1 compiler/linker ABI, validated and serialized by Pydantic.

Pydantic handles strict types, required/extra fields, JSON and schema generation.
Only cross-field GPU contract invariants are implemented here.
"""
from __future__ import annotations
import hashlib
import json
from contextlib import contextmanager
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass
from .projection import ProjectionCertificate

CONFIG = ConfigDict(strict=True, extra="forbid", ser_json_bytes="hex", val_json_bytes="hex")
U32 = Annotated[int, Field(ge=0, le=2**32-1)]


# THE PHYSICAL TABLE: every element spelling a binding may state, its size, its alignment, and
# whether this backend can ACCESS it.
#
# The four accessible ones are what the compiler loads and stores. The rest are DECLARATION-ONLY:
# a program may declare a buffer of that type and never touch it, and the binding still has to say
# what it actually is. Dropping such a binding, or relabelling a `ulong` as a `uint`, would make
# this document disagree with the signature the program was compiled from - so the spelling is
# carried verbatim and the size with it.
#
# ALIGNMENT IS A PROPERTY, NOT A FIELD, and that is a deliberate contract decision. Adding a key
# inside every binding would make every retained ABI document differ from a fresh one by an
# addition, which is precisely the failure the evolution machinery below exists for (see
# EXPLICIT_SINCE) - and that machinery constructs its expectation from the retained contract's own
# bindings, so a nested added key is not expressible in it today. The alignment is therefore
# derived from the stated type, identically by every consumer, and nothing serialized moves.
#
# This table must agree with ir.DECL_PHYSICAL, which is the compiler-side statement of the same
# facts; test_g17halfconversions cross-checks the two, so neither can drift alone.
# THE LANE COLUMN, and why a whole-element access stays inaccessible.
#
# The compiler now reaches a vector buffer ONE LANE AT A TIME: `<4 x i32>` is four word accesses at
# the lane stride, so the element itself is still not something this backend loads or stores, and
# `accessible` below still says False for it. What changed is that such a binding may be WRITTEN,
# because its lanes can be. The fourth column names the lane's own spelling, and a binding is
# writable when either the element or its lane is accessible.
#
# bfloat2 AND bfloat4 KEEP NO LANE ON PURPOSE. A bfloat lane is not accessible either, and bfloat2
# is indistinguishable from half2 by size and alignment alone - the same reason ir.LANE_ELEM names
# the lane element per spelling rather than deriving it. `uchar`, `long` and `ulong` have no lane
# decomposition at all.
ELEMENT_PHYSICAL = {
    #  spelling     bytes  alignment  accessible        lane
    "ushort":    (      2,         2,       True,       None),
    "half":      (      2,         2,       True,       None),
    "uint":      (      4,         4,       True,       None),
    "float":     (      4,         4,       True,       None),
    # uchar is not an ordinary accessible scalar.  The measured requantization probe carries its
    # logical int8/uint8 result in a four-byte int32 word, so no writable uchar exception exists.
    "uchar":     (      1,         1,      False,       None),
    "bfloat":    (      2,         2,      False,       None),
    "half2":     (      4,         4,      False,     "half"),
    "bfloat2":   (      4,         4,      False,       None),
    "short2":    (      4,         4,      False,   "ushort"),
    "long":      (      8,         8,      False,       None),
    "ulong":     (      8,         8,      False,       None),
    "uint2":     (      8,         8,      False,     "uint"),
    "float2":    (      8,         8,      False,    "float"),
    "half4":     (      8,         8,      False,     "half"),
    "bfloat4":   (      8,         8,      False,       None),
    "uint4":     (     16,        16,      False,     "uint"),
    "int4":      (     16,        16,      False,     "uint"),
    "float4":    (     16,        16,      False,    "float"),
    # the atomic declarations: 4 bytes from the struct's field, and NOT ordinarily accessible
    "atomic_uint":  (   4,         4,      False,       None),
    "atomic_int":   (   4,         4,      False,       None),
    "atomic_float": (   4,         4,      False,       None),
}

# WHAT AN ATOMIC OPERATION REACHES, per spelling. Named separately for the same reason the lane
# column is: ELEMENT_PHYSICAL cannot imply it, because `atomic_uint` and `atomic_float` are
# identical in that table and only one of them has a measured operation. A binding whose element
# is here may be WRITTEN even though it is not ordinarily accessible - an atomic read-modify-write
# is the only access it admits - and one that is absent stays carried and untouched.
ELEMENT_ATOMIC = {"atomic_uint": "uint", "atomic_int": "uint"}

# WHAT A WORD-COMPONENT ACCESS REACHES, per spelling. Named separately for the same reason as the
# lane and atomic maps: ELEMENT_PHYSICAL cannot imply it. A `long` is ONE eight-byte element and
# stays one - this does not make it a vector and does not change its size or alignment - but its
# two 32-bit halves are individually reachable, which is what makes a WRITTEN wide binding legal
# while an ordinary whole-element load or store of it still refuses.
ELEMENT_WORD_COMPONENT = {"long": "uint", "ulong": "uint"}

_ALLOW_REQUANTIZED_BINDING = frozenset()


@contextmanager
def allow_requantized_binding(indices=()):
    """Compatibility context retained for old callers; physical requant bindings are words."""
    global _ALLOW_REQUANTIZED_BINDING
    old = _ALLOW_REQUANTIZED_BINDING
    _ALLOW_REQUANTIZED_BINDING = frozenset(indices)
    try:
        yield
    finally:
        _ALLOW_REQUANTIZED_BINDING = old


@dataclass(frozen=True, config=CONFIG)
class Binding:
    index: Annotated[int, Field(ge=0, le=30)]
    offset: Annotated[int, Field(ge=0, le=60)]
    written: bool
    element_type: Literal["half", "ushort", "float", "uint",
                          "uchar", "bfloat", "half2", "bfloat2", "short2",
                          "atomic_uint", "atomic_int", "atomic_float",
                          "long", "ulong", "uint2", "float2", "half4", "bfloat4",
                          "uint4", "int4", "float4"]
    element_bytes: Annotated[int, Field(ge=1, le=16)]

    def __post_init__(self):
        if self.element_bytes != ELEMENT_PHYSICAL[self.element_type][0]:
            raise ValueError("binding element type and size disagree")
        if self.written and not (self.element_accessible or self.element_lane_accessible
                                 or self.element_atomic_accessible
                                 or self.element_word_component_accessible):
            raise ValueError("binding %d declares %s, which this backend cannot access - not as a "
                             "whole element, not one lane at a time, not one word at a time and "
                             "not atomically - and is marked written: a declaration-only type is "
                             "carried, never touched" % (self.index, self.element_type))

    @property
    def element_alignment(self):
        """The declared alignment of one element, derived from the stated type."""
        return ELEMENT_PHYSICAL[self.element_type][1]

    @property
    def element_accessible(self):
        """Whether this backend can load or store this element WHOLE. A vector never is."""
        return ELEMENT_PHYSICAL[self.element_type][2]

    @property
    def element_lane(self):
        """The spelling of one lane of this element, or None if it has no reachable lane."""
        return ELEMENT_PHYSICAL[self.element_type][3]

    @property
    def element_lane_accessible(self):
        """Whether this backend can load or store ONE LANE of this element.

        This is what makes a written vector binding legal: the declaration stays the vector, its
        size and alignment stay the vector's, and the accesses are lane-wide. A lane that is not
        itself accessible - bfloat - leaves the binding unwritable, as before.
        """
        lane = self.element_lane
        return lane is not None and ELEMENT_PHYSICAL[lane][2]

    @property
    def element_word_component(self):
        """The element one 32-bit word of this declaration is, or None.

        A wide scalar keeps its own spelling, size and alignment; this only says that its halves
        can be named. `None` for everything that is not a declared wide scalar.
        """
        return ELEMENT_WORD_COMPONENT.get(self.element_type)

    @property
    def element_word_component_accessible(self):
        """Whether this backend can load or store ONE WORD of this element."""
        component = self.element_word_component
        return component is not None and ELEMENT_PHYSICAL[component][2]

    @property
    def element_atomic(self):
        """The element an ATOMIC operation on this declaration reaches, or None.

        `metal::_atomic` is a struct and its field gives the width; this names the field's element
        for the spellings that have a MEASURED atomic operation. `atomic_float` has none, so it
        answers None and stays carried-and-untouched like any other declaration-only type.
        """
        return ELEMENT_ATOMIC.get(self.element_type)

    @property
    def element_atomic_accessible(self):
        """Whether this element admits an atomic read-modify-write.

        This is what makes a written ATOMIC binding legal while an ordinary load or store of it
        still refuses: the declaration stays `atomic_uint`, its size and alignment stay the
        struct's, and the only access is the per-lane RMW.
        """
        atomic = self.element_atomic
        return atomic is not None and ELEMENT_PHYSICAL[atomic][2]

    @property
    def element_lanes(self):
        """How many lanes the declared element has: its size divided by the lane's."""
        lane = self.element_lane
        return 1 if lane is None else ELEMENT_PHYSICAL[self.element_type][0] // ELEMENT_PHYSICAL[lane][0]


@dataclass(frozen=True, config=CONFIG)
class Instruction:
    offset: U32
    length: Annotated[int, Field(ge=2, le=32)]
    opcode: Annotated[int, Field(ge=0, le=65535)]

    def __post_init__(self):
        if self.offset % 2 or self.length % 2:
            raise ValueError("instruction offset and length must be even")


@dataclass(frozen=True, config=CONFIG)
class ThreadgroupABI:
    """The threadgroup block of ABI v4: what a program that uses threadgroup memory REQUIRES, as
    its author declared it - the compiler cannot derive a register-indexed scratchpad's size.
    Shape agreed in docs/archive/g17-cooperative-integration-feedback.md; `required_size` is a compiler
    requirement the runtime must launch with exactly; `dynamic_memory` stays empty until a dynamic
    binding is specified and measured."""
    required_size: tuple[Annotated[int, Field(ge=1, le=1024)], Annotated[int, Field(ge=1, le=1024)], Annotated[int, Field(ge=1, le=1024)]]
    static_memory_bytes: Annotated[int, Field(ge=0, le=32768)]
    static_memory_alignment: Literal[4, 8, 16]
    dynamic_memory: tuple[()] = ()

    def __post_init__(self):
        if self.static_memory_bytes % self.static_memory_alignment:
            raise ValueError("static threadgroup bytes are not a multiple of the alignment")


# The tensor forms of the delivered 32x32x64 contract (results/g17-tensor-common-witness-v1):
# operand loads, MACs and the readout. A program executing any of them states its execution
# requirement (ABI v5), and only such a program does.
# op10384/op10385 are the int8 MACs (with and without C, mmaenc). Every int8 GEMM this compiler emitted
# before also carried op17257, so adding them moved no image; an int8 GEMM whose output leaves by the
# requantization epilogue's word store (op17202, MM 25.130) carries no op17257 and was misread as a
# scalar program (ABI v3) without them.
TENSOR_OPCODES = frozenset({5106, 12674, 12675, 17257, 10384, 10385})


@dataclass(frozen=True, config=CONFIG)
class ExecutionABI:
    """ABI v5: what the program's lane layout REQUIRES of the launch, each value a fact of the
    delivered bytes - proposed in docs/archive/g17-architectural-compiler-handoff.md 9e, amended by the
    linker (b846283a): `simd_width` is the lane count the address setup is a function of (the
    SR130 lane read, and op17016 measured on lane inputs 0..31 by integration); `tensor` says the
    program executes tensor forms. The witness source's execution_simdgroups<1> is NOT carried:
    it is a fact of the source, not of the bytes, and an absent field is honest where a present
    one would be read as measured later."""
    simd_width: Literal[32]
    tensor: Literal[True]


# THE TEXTURE FORMS this backend emits (isa/g17-form-opcodes.json, harvested from integration's
# frontier programs): the two coordinate publishes and the thirty-two-bit read. A program emitting
# any of them states its resource layout (ABI v6), and only such a program does.
TEXTURE_OPCODES = frozenset({592, 15813})


@dataclass(frozen=True, config=CONFIG)
class InternalResource:
    """One binding record the section carries that the program never declared: an ordinary texture
    costs two, Apple's 44 and 48, measured constant over twelve controlled probes and confirmed by
    the linker against ten exact Apple witnesses of the frontier's binding shape. Internals rank
    FIRST, ascending by Apple index, and the machine code indexes that list - so the user buffers'
    offsets (2 * rank) start after them. Stated per record with its Apple index, not as a count:
    the linker measured both [44, 48] and [44, 45, 48] in the same class, and they give the user
    buffers different ranks."""
    rank: Annotated[int, Field(ge=0, le=30)]
    apple_index: Annotated[int, Field(ge=0, le=255)]
    kind: Literal["texture_internal"]


@dataclass(frozen=True, config=CONFIG)
class TextureResource:
    """A texture the program reads, as the IR states it. `dense_index` is the compiler's dense index
    over the textures the function uses (g17ir.texture_read: measured to differ from the Metal
    binding index, which is the runtime's fact, not this side's). `rank` is None and only None: the
    ten witnesses hold exactly the four binding records (two internals, two user buffers) and none
    of them is the texture, so giving it a rank would put a record in the vector Apple does not."""
    dense_index: Annotated[int, Field(ge=0, le=30)]
    access: Literal["read"]
    dimension: Literal["2d"]
    # float32 admitted 2026-09-12 (root's queue at 3f22b4a4). The element is a DECLARATION: the
    # linker's three Apple controls show texture2d<uint> and a bitcast texture2d<float> producing
    # identical program and metadata bytes, so nothing in the image carries it and the source
    # saying so is the only fact there is. g17cc derives it from g17ir.texture_read's declared
    # type and refuses an unknown or mixed one by name rather than defaulting.
    element: Literal["uint32", "float32"]
    coordinates: Literal["publish.coord.x/y"]
    rank: None = None


@dataclass(frozen=True, config=CONFIG)
class AccessFact:
    """What the PROGRAM does with one record of the binding list - the compiler's fact, per record,
    from which a measured section rule (slot 27 is one) can be evaluated once it exists. `read` is
    None for an internal record: whether the texture read consumes 44 or 48 is not a compiler fact."""
    record: Annotated[int, Field(ge=0, le=255)]
    kind: Literal["user", "internal"]
    written: bool
    read: bool | None
    # UNIFORM: every access the program makes to this record is lane-invariant (a store to a
    # constant slot; a load at a constant index). The linker's slot-27 rule from the compiled family
    # (H5: read-only records accessed uniformly - M1's constant-index read is in slot 27, M8's
    # divergent read is not) evaluates on this fact, not on `read`. None for an internal record.
    uniform: bool | None


UNSTATED = Literal["slot27_contents", "slot2_resource_record", "slot2_kind9_record", "argument_bytes"]
# The largest constant pool the argument_bytes rule is witnessed at (P3/P4, slot 1 = 24). H3 at 768 and P5 at
# 1024 measure the baseline 8 instead, so the boundary lies in between and is NOT bracketed.
POOL_WITNESSED_MAX = 288


@dataclass(frozen=True, config=CONFIG)
class CoordinatePublication:
    """A coordinate publish as ENCODED: its form, the target constant and the operand code. The byte
    offset is deliberately not a field (handoff 10w): whether const 2 is byte 2 (16-bit components,
    one 4-byte pair) or byte 8 (4-byte units, two slots) is unmeasured, and the two readings give
    slot 38 = 8 and 16 for this program. A consumer converts only when the unit is measured."""
    form: Literal["publish.coord.x", "publish.coord.y"]
    target_constant: Annotated[int, Field(ge=0, le=64)]
    operand_code: Literal[4]
    byte_offset_basis: Literal["unmeasured: byte 2 if coordinate components are 16-bit, byte 8 if the constant is in 4-byte units"]


@dataclass(frozen=True, config=CONFIG)
class PreloadTerm:
    binding: Annotated[int, Field(ge=0, le=30)]
    element: Annotated[int, Field(ge=0, le=0)]          # only element 0 lowers: the prologue load's offset field is not located


@dataclass(frozen=True, config=CONFIG)
class PreloadConsumer:
    """Structured, so a validator can check it against the instruction list: main consumes the published word
    through the block-operand add at this constant (4 x the declared records - the linker derives the same
    number from the binding list and refuses on disagreement)."""
    opcode: Literal[10282]
    form: Literal["alu.block"]
    block_constant: Annotated[int, Field(ge=4, le=508)]


@dataclass(frozen=True, config=CONFIG)
class PreloadABI:
    """ABI v7 (handoff 10aa): a word every lane needs, read once by the constant program before main and
    published into the argument block. `terms` is the SUM the preload carries - Apple folds a[0] + b[0] +
    c[0] into one (S2/S3) - though only one term lowers today. No block offset is stated: the linker derives
    it from the declared bindings. Preloads and uniform access facts are independent in both directions
    (S2/S3: many uniform users, one preload; M4: one preload from the written buffer, no uniform user)."""
    terms: Annotated[tuple[PreloadTerm, ...], Field(min_length=1)]
    element_type: Literal["uint"]
    lifetime: Literal["release"]
    consumer: PreloadConsumer


@dataclass(frozen=True, config=CONFIG)
class ResolvedLayoutABI:
    """THE LINKER'S RESOLUTION, carried whole (integration's 29644d9c two-phase interface; the linker's
    g17teximage.resolve, c46d34da): the canonical form both preload uses were encoded against, with its
    identity. Typed here only; it is CHECKED by the linker's own check_resolved - every canonical field
    against what the delivered binding list resolves to, the digest against the fields, the derived
    preload offset against the recomputation - which ProgramABI calls, so this side keeps no second,
    weaker copy of that check (integration's c54a7466: two checks of unequal strength look doubly
    verified and are not)."""
    declared: tuple[Annotated[int, Field(ge=0, le=30)], ...]
    internal: tuple[Annotated[int, Field(ge=0, le=63)], ...]
    order: tuple[Annotated[int, Field(ge=0, le=63)], ...]
    ranks: dict[str, Annotated[int, Field(ge=0, le=63)]]
    record_bytes: Literal[4]
    block_bytes: Annotated[int, Field(ge=4, le=508)]
    descriptor_offsets: dict[str, Annotated[int, Field(ge=0, le=126)]]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    preload_offset: Annotated[int, Field(ge=4, le=508)] | None = None


@dataclass(frozen=True, config=CONFIG)
class ResourcesABI:
    """ABI v6: the resource layout of a program that reads a texture, agreed with the linker
    (docs/archive/g17-architectural-compiler-handoff.md 10s): internals with rank and Apple index, textures
    with no rank, an EXPLICITLY EMPTY sampler list the type refuses to fill, the spill scalar (zero
    because this allocator does not spill - NOT because no form exists; see spill_basis below),
    per-record access facts, and the section
    facts this side does NOT state - named, so a consumer that needs one refuses by name instead
    of filling it from a witness."""
    internal: Annotated[tuple[InternalResource, ...], Field(min_length=1)]
    textures: Annotated[tuple[TextureResource, ...], Field(min_length=1)]
    samplers: tuple[()]
    spill_bytes: Literal[0]
    # WHY IT IS ZERO, stated beside the value at the linker's request: "0 measured" and "0 because
    # no spill form exists" author the same section today and mean different things the day a spill
    # form exists.
    #
    # THAT DAY HAS COME AND THIS VALUE IS NOW INACCURATE, which is exactly the case the linker
    # asked to be able to distinguish. The justification used to read "this backend has none
    # (g17cc's allocator docstring)" - and that docstring's claim was stale: encode_vec4 addresses
    # through an index register in both directions, and a spill/reload sequence compiles today
    # through the ordinary path (tools/g17spillgap.py, handoff 10bi/10bj). So the FORM exists and
    # the ALLOCATOR does not emit one, which "no_spill_form" conflates into a single value - the
    # very conflation this field was added to prevent.
    #
    # IT IS NOT CHANGED HERE BECAUSE THAT IS A CONTRACT CHANGE AND NOT MINE ALONE. The string is
    # asserted by g17regress and consumed by g17texturecompilerside; a new value is vocabulary for
    # integration to set, and the linker measured the facts and explicitly declined to invent it
    # rather than pick something convenient for whoever asked. Raised in the handoff instead.
    #
    # WHAT THE POPULATION SAYS, so the widening is not designed from nothing (the linker's
    # 8df2c0ef): slot 31 is present on 5,008 of 6,495 sections with six values - 32, 48, 64, 80,
    # 96, 128 - every one a MULTIPLE OF 16, which is the granularity this side derived
    # independently from the instruction, four consecutive registers. And 4,135 of those 5,008
    # carry only internals that non-spilling sections also carry, so a spill needs no scratch
    # BINDING: the type will need a size, not a resource.
    spill_basis: Literal["no_spill_form"]
    access: Annotated[tuple[AccessFact, ...], Field(min_length=1)]
    # THE COORDINATE PUBLICATIONS IN ADDRESS UNITS (integration's f60054e6): each publish's target
    # byte offset and written width, as the emitted forms state them - publish.coord.x writes four
    # bytes at [op4 + 0], publish.coord.y four bytes at [op4 + 8]. The linker derives the storage
    # extent (what per-kernel slot 38 tracks: the compiled family refuted "8 x textures" both ways -
    # M5 carries 16 for one texture at two coordinates, M6 8 for two textures at one) from these;
    # this side does not state a slot count, because the slot size is the section's rule, not the
    # program's, and two components must not silently become two coordinate pairs.
    coordinate_publications: Annotated[tuple[CoordinatePublication, ...], Field(min_length=1)]
    not_stated: tuple[UNSTATED, ...]
    # THE ARGUMENT BUFFER'S BYTE SIZE (per-kernel slot 1), the field the linker's route named as required and
    # refused without (handoff 10af). Measured over 31 Apple members across four preregistered families:
    #     argument_bytes = 8 + 4 * ceil(u / 2) + 4 * (pool // 64)
    # with u the uniformly-read non-output user bindings and pool this program's own constant pool in bytes -
    # both facts this contract already states, which is why the compiler can state this one. Present exactly
    # when the program is inside the witnessed class (pool <= 288); above it slot 1 COLLAPSES to the baseline
    # on two members and the boundary is unbracketed, so the compiler names it in not_stated instead of
    # extrapolating. __post_init__ recomputes it from this contract's own access facts and pool.
    argument_bytes: Annotated[int, Field(ge=8, le=508)] | None = None
    # ABI v7's keys, INSIDE the resources block because that is where the linker's plan reads them
    # (g17teximage.plan: resources.preloads, resources.resolved_layout; a first cut carried them at the
    # top level, and against that shape the linker's check found no layout and no preloads and passed
    # on nothing). Present exactly when the program has a uniform preload.
    preloads: tuple[PreloadABI, ...] | None = None
    resolved_layout: ResolvedLayoutABI | None = None

    def __post_init__(self):
        ranks = [r.rank for r in self.internal]
        if ranks != list(range(len(ranks))):
            raise ValueError("internal records must occupy ranks 0..n-1 in order; got %s" % ranks)
        idx = [r.apple_index for r in self.internal]
        if idx != sorted(set(idx)):
            raise ValueError("internal records rank first ASCENDING by Apple index; got %s" % idx)
        dense = [t.dense_index for t in self.textures]
        if dense != sorted(set(dense)):
            raise ValueError("texture dense indices must be unique and ascending; got %s" % dense)
        recs = [a.record for a in self.access if a.kind == "internal"]
        if sorted(recs) != sorted(idx):
            raise ValueError("access facts must cover exactly the internal records %s; got %s" % (idx, recs))
        if any(a.read is not None for a in self.access if a.kind == "internal"):
            raise ValueError("whether an internal record is read is not a compiler fact; state None")
        if any(a.read is None or a.uniform is None for a in self.access if a.kind == "user"):
            raise ValueError("a user binding's read and uniform facts are the compiler's to state")
        if any(a.uniform is not None for a in self.access if a.kind == "internal"):
            raise ValueError("an internal record's uniformity is not a compiler fact; state None")
        offs = [c.target_constant for c in self.coordinate_publications]
        if offs != sorted(set(offs)):
            raise ValueError("coordinate publications must target distinct ascending constants; got %s" % offs)
        # THE TWO SECTION FACTS WITHOUT A MEASURED RULE STAY NAMED (integration's f60054e6): slot 27's
        # H5 rule is supported on the family and not established over the 6,495 sections (165 carry
        # internal 44); the slot-2 record is absent on the nine members and carried by 3,959 sections.
        # A consumer needing either refuses by name; the compiler supplies the facts a rule would read.
        # the kind-9 record (the linker's 034b117b: its presence is what splits field2 = pool / 4, 8,970 of 8,970
        # either way) is a class fact this side cannot state and must not guess; named so the linker derives it
        required = {"slot27_contents", "slot2_resource_record", "slot2_kind9_record"}
        if not required <= set(self.not_stated):
            raise ValueError("not_stated must name the three section facts until the linker derives them for the chosen class; got %s" % (self.not_stated,))
        if set(self.not_stated) - required - {"argument_bytes"}:
            raise ValueError("not_stated names something this side has no reading for: %s" % (set(self.not_stated) - required - {"argument_bytes"},))
        # ARGUMENT_BYTES IS EITHER STATED OR NAMED, NEVER BOTH AND NEVER NEITHER (handoff 10af)
        if (self.argument_bytes is not None) == ("argument_bytes" in self.not_stated):
            raise ValueError("argument_bytes must be stated or named in not_stated, exactly one")
        if (self.preloads is None) != (self.resolved_layout is None):
            raise ValueError("preloads and the resolved layout travel together: the publication and main's block read are encoded against the layout, so one without the other states an encoding nobody can check")


@dataclass(frozen=True, config=CONFIG)
class PromotedRange:
    """One constant range promoted into the argument buffer, as the RECORD states it.

    ADDED 2026-09-12 for the linker's six-buffer authoring path, which root's queue holds refused
    "until the compiler supplies that ABI contract". The field meanings are the linker's
    measurements (`results/g17-promoted-ranges-v1`, integrated at 38055ec4); nothing here is
    inferred, and the one thing this file adds is somewhere to put them.

    THE COMPILER DOES NOT PRODUCE ONE. This backend has no constant-promotion pass, so `g17cc`
    states None and every existing contract is unchanged. The contract exists so an authored
    section can SAY what it did, rather than a linker-only field saying it somewhere the ABI
    cannot see - which is the arrangement the promoted-range refusal exists to avoid.

    FIELD 0 IS A ONE-BYTE KIND AND THE PACKING I DERIVED WAS AN OVERREAD. Corrected 2026-09-12 from
    docs/archive/g17-six-promoted-class-handoff.md. The first version of this said field 0 packed the length
    and the kind as `(length << 8) | 5`, from a table showing 261, 517 and 773 for lengths 1, 2 and
    3. Those numbers came from a reader that fetched every slot below 3 as four bytes regardless of
    its vtable width: slot 0 sits at body offset 15 and slot 2 - the length - at 16, so the read
    swallowed the length and handed it back as the high bits. The high byte WAS the length, fetched
    by the overrun.

    THE FORMULA FIT EVERY WITNESS BECAUSE THE INSTRUMENT CONSTRUCTED THE FIT. "Holds everywhere" was
    guaranteed by the layout rather than measured against it, and a check of mine asserted it across
    nine records without being able to fail. Three facts settle the real width and none of them is
    that reader's arithmetic: the offset is ODD (15 here, 11 and 19 in the other records) and a
    two- or four-byte FlatBuffers field is aligned to its width; a wider field would OVERLAP slot 2
    at offset 16 and table fields do not overlap; and this repository's own emitter has always
    written a slot-2 record's field 0 as one byte, `{0: ("<B", vals[0])}` in g17mdgen, while
    reproducing four measured classes byte-for-byte.

    So field 0 is the KIND, one byte, value 5 in every record, and the length is a separate
    four-byte field at slot 2. No packed value is derived here any more.
    """
    # in ELEMENTS, at slot 2, a genuine FOUR-BYTE field read at its true width.
    #
    # THE BOUND IS THE FIELD'S ENCODING CAPACITY AND NOTHING MORE, and it has now been wrong twice
    # in opposite directions. First it was `le=3`, the largest value any witness carries - the
    # population maximum mistaken for a limit. Then it became `le=255`, argued from a 16-bit field 0
    # that the length supposedly shared; that field is one byte and holds the kind alone, so there
    # was no 16-bit container to overflow and 256 was refused for a reason that did not exist.
    #
    # What the field can HOLD and what this project can AUTHOR are different statements and are
    # separated here: the capacity is four bytes, and AUTHORING_SUPPORT below records what is
    # actually witnessed. Admitting a value is not a claim that it works.
    length: Annotated[int, Field(ge=1, le=2**32 - 1)]
    # twice the DECLARED buffer count, in 2-byte units - declared, not bound: the promoted buffer
    # still takes a pointer slot. Measured at 6, 12 and Apple's 14 for 3, 6 and 7 declared.
    argument_offset: Annotated[int, Field(ge=0, le=2**16 - 1)]
    # which buffer the range came from. ELIDED AT ZERO in the record, so the default is the value
    # rather than an absence: re_osd_eval_stencils binds 1..6 and its promoted buffer is the
    # missing 0.
    binding_index: Annotated[int, Field(ge=0, le=30)] = 0
    # in ELEMENTS, not bytes, and elided at zero the same way (a[0] absent, a[4] -> 4, a[8] -> 8).
    source_offset: Annotated[int, Field(ge=0, le=2**16 - 1)] = 0
    # FIELD 0 ITSELF: one byte, value 5 in every record. Pinned to its single witnessed value,
    # which is a discriminator and a different kind of bound from `length`'s capacity above.
    kind: Annotated[int, Field(ge=5, le=5)] = 5

    # WHAT IS ACTUALLY AUTHORABLE, kept apart from what the field can hold. From the linker's
    # retained evidence: length 1 has a local witness that authors; length 2's section is retained
    # at 536 bytes and is REFUSED, which is not the same as impossible; length 3 is Apple-only with
    # no local witness at all. A schema that admitted 1..2^32-1 silently would be claiming every
    # length works, which is exactly what this separation refuses to do.
    AUTHORING_SUPPORT = {1: "authors; local witness (sixp-one)",
                         2: "refused; section retained at 536 bytes (sixp-cstruct), not impossible",
                         3: "refused; Apple-only, no local witness"}

    @property
    def authoring_support(self) -> str:
        """What is known about authoring THIS length, as opposed to whether the field holds it."""
        return PromotedRange.AUTHORING_SUPPORT.get(
            self.length, "unwitnessed; the four-byte field holds it and nothing authors it")


@dataclass(frozen=True, config=CONFIG)
class ArgumentState:
    """ABI v9: the argument buffer THIS COMPILER stages, read off the offsets its own code reads.

    THE LINKER'S REQUEST (docs/archive/g17-six-buffer-image-handoff.md): "somewhere a DEVICE-BUFFER
    ProgramABI can state per-kernel slot 1. resources.argument_bytes exists and is the TEXTURE
    route's; a buffer contract has no equivalent." Without a stated value the six-binding class
    inherits FIVE's `pk_extra` and emits 12, which is right for one of seven Apple arms and wrong
    for six, with nothing in the section to notice.

    WHAT THIS IS NOT. It is NOT per-kernel slot 1. Seven Apple arms share a six-buffer binding
    shape and carry 12 or 16, so slot 1 does not follow from the binding count - the linker
    measured that (g17sixpromoted.argument_bytes_is_not_derivable) and this side is not going to
    fit a rule to it. What a compiler can state is what IT stages, and `per_kernel_slot_1` is
    therefore required in `not_stated`: a consumer that needs slot 1 refuses by name rather than
    reading this extent as if it were the answer. Labelling an externally supplied witness value
    compiler-derived is the failure this field exists to avoid, not one to commit inside it.

    THE DERIVATION. Every bound buffer's pointer is read by the emitted code at word offset
    `2 * rank`, rank being the position in the slot-sorted binding list (g17cc's `_abi_bindings`);
    a pointer is two words. So the block this program reads spans `2 * n` words for n bindings, and
    that extent is a fact about the emitted loads.

    AND THE RULE IS NAMED BECAUSE IT IS NOT APPLE'S. The linker measured Apple's pointer offset as
    `2 * (index - lowest bound index)`; this compiler emits `2 * rank`. The two agree exactly when
    the bound indices are CONTIGUOUS, which the six-buffer program at hand is - so they agree here,
    and a program binding 1,2,3,5,6,7 would separate them: this compiler emits 0,2,4,6,8,10 where
    the index rule wants 0,2,4,8,10,12. `indices_contiguous` says which case a contract is in, so
    the agreement is stated per program rather than assumed from the one that happened to be first.
    """
    # (binding index, word offset), ascending by index - the pointers as the emitted code reads them
    pointer_offsets: Annotated[tuple[tuple[Annotated[int, Field(ge=0, le=30)],
                                           Annotated[int, Field(ge=0, le=60)]], ...],
                               Field(min_length=1)]
    pointer_words: Literal[2]
    block_words: Annotated[int, Field(ge=2, le=62)]
    block_bytes: Annotated[int, Field(ge=8, le=248)]
    basis: Literal["emitted_pointer_offsets"]
    offset_rule: Literal["2 * rank"]
    indices_contiguous: bool
    not_stated: tuple[Literal["per_kernel_slot_1"], ...]

    def __post_init__(self):
        indices = [i for i, _ in self.pointer_offsets]
        if indices != sorted(set(indices)):
            raise ValueError("pointer offsets must be unique and ascending by binding index; got %s"
                             % indices)
        offsets = [o for _, o in self.pointer_offsets]
        want = [self.pointer_words * r for r in range(len(offsets))]
        if offsets != want:
            raise ValueError("offset_rule is %r, which gives %s, but the contract carries %s"
                             % (self.offset_rule, want, offsets))
        if self.block_words != self.pointer_words * len(self.pointer_offsets):
            raise ValueError("block_words %d is not %d pointers of %d words"
                             % (self.block_words, len(self.pointer_offsets), self.pointer_words))
        if self.block_bytes != 4 * self.block_words:
            raise ValueError("block_bytes %d is not 4 x %d words"
                             % (self.block_bytes, self.block_words))
        if self.indices_contiguous != (indices == list(range(indices[0], indices[0] + len(indices)))):
            raise ValueError("indices_contiguous says %s of %s"
                             % (self.indices_contiguous, indices))
        # THE ONE FIELD THAT MUST BE NAMED RATHER THAN STATED. Not optional: a contract that quietly
        # dropped it would read as though this extent WERE slot 1.
        if "per_kernel_slot_1" not in self.not_stated:
            raise ValueError("a buffer contract's argument state must name per_kernel_slot_1 in "
                             "not_stated: seven Apple arms share this binding shape and carry 12 "
                             "or 16, so slot 1 does not follow from what this side stages")


@dataclass(frozen=True, config=CONFIG)
class SpillState:
    """ABI v10: the scratch an automatically-spilled program needs, per thread.

    ROOT'S REQUIREMENT, VERBATIM: "Scratch is deliberately not guessed from a buffer name or
    observed stores. The spilled delivery must supply its allocation extent through its ABI."
    Carrying the scratch as a binding says WHERE it is bound and nothing about HOW BIG it must be,
    and a consumer that sized it by counting stores in the emitted code would be deriving a
    requirement from an artifact rather than reading a stated one.

    THE EXTENT IS PER THREAD AND THAT IS THE WHOLE POINT. Each spilled group is four words and
    every thread owns its own slots - the address is `4 * (thread * groups + g)` - so the
    allocation is `4 * groups` words PER THREAD and a caller that allocates the per-thread figure
    once will have threads writing over each other. `words_per_thread` is named to make that hard
    to misread.

    THIS IS NOT THE HARDWARE SPILL REGION, and `ResourcesABI.spill_bytes` is deliberately untouched
    at 0: this scratch is an ordinary declared device buffer that the compiler added to the binding
    list. Calling a declared buffer a spill region is the relabelling
    docs/archive/g17-spill-provenance-handoff.md warns against, and the two must not be conflated just
    because they share a word.
    """
    binding_index: Annotated[int, Field(ge=0, le=30)]
    groups: Annotated[int, Field(ge=1)]
    words_per_group: Literal[4]
    # The per-thread stride, in words and bytes: thread t's slots begin at t * words_per_thread.
    words_per_thread: Annotated[int, Field(ge=4)]
    bytes_per_thread: Annotated[int, Field(ge=16)]
    # THE LAUNCH INDEX THE ADDRESS IS FORMED FROM, stated because "per thread" is meaningless
    # without it: thread_position_in_threadgroup repeats in every group, so a slot addressed from
    # it would be shared by one thread per group. The allocation must cover the whole grid extent
    # along this axis.
    launch_index: Literal["thread_position_in_grid"]
    launch_axis: Literal["x"]
    # The vector store moves four consecutive words, so each group's base is four-word aligned and
    # the allocation's own base must be too. Stated rather than left for a caller to derive from
    # the group size.
    alignment_bytes: Literal[16]
    basis: Literal["emitted_vector_stores"]

    def __post_init__(self):
        if self.words_per_thread != self.words_per_group * self.groups:
            raise ValueError("words_per_thread %d is not %d groups of %d"
                             % (self.words_per_thread, self.groups, self.words_per_group))
        if self.bytes_per_thread != 4 * self.words_per_thread:
            raise ValueError("bytes_per_thread %d is not 4 x %d words"
                             % (self.bytes_per_thread, self.words_per_thread))
        if self.bytes_per_thread % self.alignment_bytes:
            raise ValueError("bytes_per_thread %d is not a multiple of the %d-byte alignment the "
                             "vector store needs" % (self.bytes_per_thread, self.alignment_bytes))


@dataclass(frozen=True, config=CONFIG)
class RequantizationABI:
    """The measured scalar narrowing boundary, separate from image-class selection.

    This marker is deliberately narrow.  The logical result is int8/uint8, but the retained
    scalar probes use 32-bit ``int32_t`` storage for both the scale word and output word.  It is
    not a permission to infer a byte-buffer class from the logical result type.
    """
    kind: Literal["int32_to_int8", "int32_to_uint8"]
    source_binding: Annotated[int, Field(ge=0, le=30)]
    destination_binding: Annotated[int, Field(ge=0, le=30)]
    elements: Literal[256]
    groups: Literal[8]
    scale: Literal["1/512", "1/16"]
    rounding: Literal["round_half_to_even"]
    saturation: Literal["signed_int8", "unsigned_uint8"]
    dispatch_boundary: Literal["required"]
    scale_binding: Annotated[int, Field(ge=0, le=30)] | None = None
    metadata_class: Literal["scalar_476"] | None = None
    scale_storage: Literal["uint32_bits"] | None = None
    scale_addressing: Literal["constant_program_preload", "indexed_replicated_unmeasured"] | None = None
    output_storage: Literal["int32_word"] | None = None

    def __post_init__(self):
        if self.source_binding == self.destination_binding:
            raise ValueError("requantization source and destination bindings must differ")
        if self.scale_binding is not None and self.scale_binding in (self.source_binding, self.destination_binding):
            raise ValueError("requantization scale binding must be distinct from source and destination")
        if self.metadata_class == "scalar_476" and self.scale_binding is None:
            raise ValueError("scalar_476 requantization requires a scale binding")
        if self.kind == "int32_to_int8":
            if self.scale not in ("1/512", "1/16") or self.saturation != "signed_int8":
                raise ValueError("signed requantization ABI has an invalid measured scale or range")
        elif self.scale != "1/16" or self.saturation != "unsigned_uint8":
            raise ValueError("unsigned requantization ABI is measured only at scale 1/16")


@dataclass(frozen=True, config=CONFIG)
class ProgramABI:
    version: Annotated[int, Field(ge=1, le=1)]
    name: Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z_0-9]*$")]
    code_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    code_size: Annotated[int, Field(ge=2, le=2**32-1)]
    entry: Annotated[int, Field(ge=4, le=2**32-1)]
    prologue: bytes
    bindings: Annotated[tuple[Binding, ...], Field(min_length=1)]
    instructions: Annotated[tuple[Instruction, ...], Field(min_length=1)]
    arch_flag: bool
    uses_threadgroup: bool
    writes_buffer: bool
    writes_texture: bool
    exact_grid_required: bool
    # OPTIONAL AT THE SCHEMA, REQUIRED AT THE POINT OF USE. Retained contracts that predate the
    # block still load (none of them uses threadgroup memory); a contract that says
    # uses_threadgroup and carries no block is refused here, and a block without the use likewise.
    threadgroup: ThreadgroupABI | None = None
    # ABI v4's second key: the program's external constant bytes (metadata slot 13), stated by
    # the compiler from its emitted code - an explicit EMPTY tuple for a program with none. It
    # travels with the threadgroup block (both are v4) and is refused without it and vice versa,
    # so a v3 contract never carries it and a v4 contract never omits it.
    constant_pool: tuple[Annotated[int, Field(ge=0, le=255)], ...] | None = None
    # ABI v5's key: present exactly when the program executes tensor forms (symmetric, so the
    # field cannot drift onto a contract that does not need it and become decorative).
    execution: ExecutionABI | None = None
    # ABI v6's key: present exactly when the program emits texture forms (symmetric, as v5's is).
    resources: ResourcesABI | None = None
    # ABI v7's keys: present exactly when the program has a uniform preload; the constant program's
    # identity is stated beside main's (the prologue bytes are already the `prologue` field).
    # (the preloads and the resolved layout they were encoded against are resources.preloads and
    # resources.resolved_layout - the linker reads them there)
    constant_program_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    # ABI v8's key: present exactly when the program promotes a constant range into the argument
    # buffer, symmetric with v5's and v6's. None - not an empty tuple - for a program that promotes
    # nothing, so a contract cannot carry a decorative empty list and a reader cannot mistake
    # "promotes nothing" for "was authored before promotion was stateable".
    #
    # NOT CAPPED AT ONE RECORD. Every arm of the linker's probe family has exactly one, and
    # Apple's arithmetic_binary_gradient_float has TWO in one section - so a one-record cap would
    # have been the probe family's shape mistaken for the format's.
    promoted_ranges: Annotated[tuple[PromotedRange, ...], Field(min_length=1)] | None = None
    # ABI v9's key: the argument state a DEVICE-BUFFER program stages, the linker's named blocker.
    # OPTIONAL AT THE SCHEMA, REQUIRED AT THE POINT OF USE - threadgroup's precedent, chosen for the
    # same reason: every retained buffer contract predates this field, and making it mandatory would
    # turn an additive key into a breaking change for every consumer that loads one. The linker's
    # author requires it where it needs it. Where it IS carried the symmetry is enforced below: the
    # texture route states slot 1 through resources.argument_bytes and must not carry this too.
    argument_state: ArgumentState | None = None
    # ABI v10's key: present exactly when the allocator spilled. None - not a zero record - for a
    # program that did not, so "did not spill" and "was authored before spilling was stateable"
    # stay distinguishable.
    spill_state: SpillState | None = None
    # Explicit compact compilation only. Bindings remain the exact emission map;
    # the certificate carries complete original declarations and their omission proof.
    resource_projection: ProjectionCertificate | None = None
    # The measured int32 -> int8/uint8 scalar bridge. This is carried only by the explicit
    # requantization primitive; it is not a generic writable-uchar escape or a metadata selector.
    requantization: RequantizationABI | None = None

    def __post_init__(self):
        if self.requantization is not None:
            by_index = {b.index: b for b in self.bindings}
            src = by_index.get(self.requantization.source_binding)
            scale = (by_index.get(self.requantization.scale_binding)
                     if self.requantization.scale_binding is not None else None)
            dst = by_index.get(self.requantization.destination_binding)
            if src is None or dst is None:
                raise ValueError("requantization ABI names a binding absent from the contract")
            if (src.written or src.element_type != "uint" or src.element_bytes != 4):
                raise ValueError("requantization source must be a read-only uint binding")
            if self.requantization.scale_binding is not None:
                if (scale is None or scale.written or scale.element_type != "uint" or
                        scale.element_bytes != 4):
                    raise ValueError("requantization scale must be a read-only uint32 bits binding")
            if (not dst.written or dst.element_type != "uint" or dst.element_bytes != 4):
                raise ValueError("requantization destination must be a writable uint word binding")
            if (self.requantization.scale_storage != "uint32_bits" or
                    self.requantization.output_storage != "int32_word"):
                raise ValueError("requantization physical storage is not the measured int32 word class")
            if (self.requantization.scale_binding is not None and
                    self.requantization.scale_addressing not in
                    ("constant_program_preload", "indexed_replicated_unmeasured")):
                raise ValueError("requantization scale addressing is not named")
            if not any(i.opcode == 17229 for i in self.instructions):
                raise ValueError("requantization ABI requires the measured indexed word store")
        if self.resource_projection is not None:
            self.resource_projection.check_bindings(self.bindings)
            if (self.resources is not None or self.spill_state is not None or
                    self.promoted_ranges is not None or self.execution is not None or
                    self.threadgroup is not None or self.uses_threadgroup or self.constant_pool):
                raise ValueError("resource projection requires the ordinary device-buffer domain")
        block_adds = [i for i in self.instructions if i.opcode == 10282 and i.length == 12]
        trivial = self.prologue == bytes.fromhex("0e000000") + bytes.fromhex("0600") * ((self.entry - 4) // 2)
        preloads = self.resources.preloads if self.resources is not None else None
        measured_requant_preload = (self.requantization is not None and
                                    self.requantization.scale_addressing == "constant_program_preload")
        if measured_requant_preload:
            from . import requantpreload
            if (self.entry != requantpreload.ENTRY or self.prologue != requantpreload.PROLOGUE or
                    self.constant_program_sha256 != requantpreload.PROLOGUE_SHA256):
                raise ValueError("measured requantization preload must carry the retained 64-byte constant program")
        if preloads is not None:
            if self.constant_program_sha256 is None or self.constant_program_sha256 != hashlib.sha256(self.prologue).hexdigest():
                raise ValueError("a preload's constant program identity must be stated and must be the sha256 of the prologue carried")
            if trivial: raise ValueError("preloads with a trivial prologue: nobody publishes the word")
            if not block_adds: raise ValueError("preloads without a block-operand add (op10282) in main: nobody consumes the word")
            if any(p.consumer.block_constant != 4 * (len(self.bindings) + len(self.resources.internal)) for p in preloads):
                raise ValueError("a preload's block constant is not 4 x the declared records")
            # THE LAYOUT AGREEMENT IS THE LINKER'S CHECK, CONSUMED: every canonical field of the delivered layout
            # against what this binding list resolves to, the digest against the fields, the derived offset against
            # the recomputation. It refuses with its own message (g17authorobj.Missing), re-raised here as the
            # contract's refusal so a stale pairing cannot load anywhere the contract is read.
            from . import teximage as g17teximage
            try: g17teximage.check_resolved(self.to_dict())
            except Exception as e: raise ValueError("the resolved layout does not pass the linker's agreement check: %s" % e) from e
        else:
            if self.constant_program_sha256 is not None and not measured_requant_preload:
                raise ValueError("a constant program identity without preloads")
            if not trivial and not measured_requant_preload:
                raise ValueError("a non-trivial prologue without preloads: the constant program does something the contract does not state")
        texture = any(i.opcode in TEXTURE_OPCODES for i in self.instructions)
        if texture and self.resources is None:
            raise ValueError("texture forms without a stated resource layout: a program emitting %s must carry ABI v6's resources block"
                             % sorted({i.opcode for i in self.instructions if i.opcode in TEXTURE_OPCODES}))
        if self.resources is not None and not texture:
            raise ValueError("a resource layout on a program with no texture forms: the block is stated only where internal records shift the ranks")
        if self.resources is not None and self.constant_pool is None:
            raise ValueError("an ABI v6 contract without its constant_pool: the section states slot 13, so the program must, an empty tuple when it has none")
        # ARGUMENT_BYTES IS RECOMPUTED FROM THIS CONTRACT'S OWN FACTS (handoff 10af). The measured rule reads
        # the uniformly-read non-output user records and the constant pool, both stated here, so a contract
        # cannot carry a value that disagrees with what it says about itself - and the witnessed class is a
        # pool of at most 288 bytes, above which slot 1 collapses on the two members that reach it.
        if self.resources is not None and self.resources.argument_bytes is not None:
            u = sum(1 for a in self.resources.access if a.kind == "user" and not a.written and a.uniform)
            pool = len(self.constant_pool or ())
            if pool > POOL_WITNESSED_MAX:
                raise ValueError("argument_bytes stated for a %d-byte constant pool: the rule is witnessed to %d bytes and slot 1 collapses to the baseline above it (P5 at 1024, H3 at 768), so name it in not_stated instead" % (pool, POOL_WITNESSED_MAX))
            want = 8 + 4 * -(-u // 2) + 4 * (pool // 64)
            if self.resources.argument_bytes != want:
                raise ValueError("argument_bytes %d disagrees with this contract's own facts: %d uniformly-read non-output records and a %d-byte pool give %d" % (self.resources.argument_bytes, u, pool, want))
        # ABI v9: the two argument-state routes are exclusive. The texture route states per-kernel
        # slot 1 through resources.argument_bytes, from a rule measured on 31 Apple members; the
        # buffer route states the block it stages and names slot 1 unstated. A contract carrying
        # both would be answering the same question two ways.
        if self.argument_state is not None:
            if self.resources is not None:
                raise ValueError("argument_state on a texture contract: that route states per-kernel "
                                 "slot 1 through resources.argument_bytes, and two answers to one "
                                 "question is how a section gets authored from the wrong one")
            stated = [i for i, _ in self.argument_state.pointer_offsets]
            declared = [b.index for b in self.bindings]
            if stated != declared:
                raise ValueError("argument_state covers bindings %s but the contract declares %s"
                                 % (stated, declared))
            for (index, offset), binding in zip(self.argument_state.pointer_offsets, self.bindings):
                if binding.offset != offset:
                    raise ValueError("argument_state puts binding %d at word %d and the binding "
                                     "itself says %d" % (index, offset, binding.offset))
        # ABI v10: the spill state is symmetric with the vector stores that produce it, so it can
        # neither be claimed by a program that never spilled nor omitted by one that did.
        # A VECTOR STORE IS NOT EVIDENCE OF A SPILL. A program may author one itself, and an
        # earlier cut of this check refused such a contract outright - so `spill_state=None` beside
        # an ordinary op17256 is valid and says what it should: this program's stores are its own.
        # What the check does enforce is the other direction, that a stated extent cannot exceed
        # the stores actually emitted.
        vector_stores = sum(1 for i in self.instructions if i.opcode == 17256 and i.length == 8)
        if self.spill_state is not None:
            if not vector_stores:
                raise ValueError("a spill state on a program that emits no vector store")
            if self.spill_state.groups > vector_stores:
                raise ValueError("spill state declares %d spilled groups and the program emits "
                                 "only %d vector store(s)"
                                 % (self.spill_state.groups, vector_stores))
            if self.spill_state.binding_index not in [b.index for b in self.bindings]:
                raise ValueError("spill state names binding %d, which the contract does not declare"
                                 % self.spill_state.binding_index)
        tensor = any(i.opcode in TENSOR_OPCODES for i in self.instructions)
        if tensor and self.execution is None:
            raise ValueError("tensor forms without a stated execution requirement: a program executing %s must carry ABI v5's execution block"
                             % sorted({i.opcode for i in self.instructions if i.opcode in TENSOR_OPCODES}))
        if self.execution is not None and not tensor:
            raise ValueError("an execution requirement on a program with no tensor forms: the field is stated only where the lane layout requires it")
        if self.uses_threadgroup and self.threadgroup is None:
            raise ValueError("uses_threadgroup without a threadgroup declaration: the required group size and static bytes must be stated")
        if self.threadgroup is not None and not self.uses_threadgroup:
            raise ValueError("a threadgroup declaration on a program that does not use threadgroup memory")
        if self.threadgroup is not None and self.constant_pool is None:
            raise ValueError("an ABI v4 contract without its constant_pool: the program's external constant bytes must be stated, an empty tuple when it has none")
        # ABI v5 STATES ITS POOL TOO (integration 614b7635): the tensor metadata class distinguishes an
        # empty pool from a non-empty one (the pooled programs at 18+ A rows carry their multiplier
        # in slot 13), so a tensor contract must say which it is rather than let the linker assume.
        if self.execution is not None and self.constant_pool is None:
            raise ValueError("an ABI v5 contract without its constant_pool: a tensor program must state its external constant bytes, an empty tuple when it reads none")
        if self.threadgroup is None and self.execution is None and self.resources is None and self.constant_pool is not None:
            raise ValueError("constant_pool on a v3 contract: v3 is frozen and does not carry it")
        if len(self.prologue) != self.entry:
            raise ValueError("prologue must contain exactly entry bytes")
        indices = [b.index for b in self.bindings]
        # THE RANK LAW: internals first ascending, then users ascending, offset = 2 * rank, ranks
        # consecutive from zero across both. A contract with no resources block has no internals, so
        # the rule is what it always was - offset == 2 * ordinal - and every executed v3/v4/v5
        # contract is unchanged byte for byte. With internals the user offsets start after them.
        base = len(self.resources.internal) if self.resources is not None else 0
        if indices != sorted(set(indices)) or any(b.offset != 2*(base + i) for i, b in enumerate(self.bindings)):
            raise ValueError("ABI v1 requires unique ascending device bindings and consecutive descriptor offsets"
                             + (" after the %d internal records" % base if base else ""))
        if self.resources is not None:
            users = {a.record: a for a in self.resources.access if a.kind == "user"}
            if sorted(users) != indices:
                raise ValueError("access facts must cover exactly the user bindings %s; got %s" % (indices, sorted(users)))
            if any(users[b.index].written != b.written for b in self.bindings):
                raise ValueError("a user binding's access fact disagrees with its `written`")
        cursor = 0
        for i in self.instructions:
            if i.offset != cursor:
                raise ValueError("instruction boundaries are not consecutive")
            cursor += i.length
        if cursor != self.code_size:
            raise ValueError("instruction boundaries do not cover the code")
        if self.writes_buffer != any(b.written for b in self.bindings):
            raise ValueError("written bindings disagree with writes_buffer")

    @property
    def forms(self):
        return tuple(sorted({(i.opcode, i.length) for i in self.instructions}))

    def check_code(self, code):
        if type(code) is not bytes or len(code) != self.code_size or hashlib.sha256(code).hexdigest() != self.code_sha256:
            raise ValueError("code differs from the compiler ABI")
        if self.resource_projection is not None:
            self.resource_projection.check_emission(code, self.bindings)

    def to_dict(self):
        d = ADAPTER.dump_python(self, mode="json")
        # EVERY CONTRACT WITHOUT A RESOURCES BLOCK SERIALISES AS IT DID BEFORE THE BLOCK EXISTED: the
        # key is carried only when present, so the retained contract.json of every executed
        # buffer-only and tensor program stays byte-identical (integration's encoder test compares
        # those bytes). from_dict reads an absent key as None.
        if d.get("resources") is None:
            d.pop("resources", None)
        if d.get("constant_program_sha256") is None: d.pop("constant_program_sha256", None)   # v7 keys only when a preload exists: v3-v6 files unchanged
        if d.get("resource_projection") is None: d.pop("resource_projection", None)
        if d.get("requantization") is None: d.pop("requantization", None)
        for k in ("preloads", "resolved_layout"):
            if "resources" in d and d["resources"].get(k) is None: d["resources"].pop(k, None)
        return d

    @classmethod
    def from_dict(cls, value):
        # Keep the compatibility context while nested bindings are reconstructed. The measured
        # requantization ABI now uses writable uint words, so it grants no uchar exception.
        marker = value.get("requantization") if isinstance(value, dict) else None
        indices = ()
        if isinstance(marker, dict):
            indices = (marker.get("destination_binding"),)
        with allow_requantized_binding(i for i in indices if isinstance(i, int)):
            return ADAPTER.validate_json(json.dumps(value))


ADAPTER = TypeAdapter(ProgramABI)


def verify_retained_rebuild(root, files):
    """Legacy path adapter; the comparison rules are verify_rebuild below."""
    from pathlib import Path
    root = Path(root)
    return verify_rebuild({name: (root / name).read_bytes() for name in files}, files)


def main(argv=None):
    print(json.dumps(ADAPTER.json_schema(), indent=2))




# ---------------------------------------------------------------------------
# RETAINED-ABI EVOLUTION. This half lived in agxforge/g17/abi.py while ProgramABI itself
# stayed in tools/g17abi.py, so a package module that needed ProgramABI or the opcode
# tables had to import the legacy name - which is what root's new control forbids and
# what scanlink was doing. One module now holds both.
# ---------------------------------------------------------------------------
from copy import deepcopy
# THE FIELDS RETAINED EVIDENCE PREDATES, and what each one is derived from. Evidence frozen before
# 2026-09-12 carries none of them; a contract built from the same sources today carries all four.
# Byte equality cannot tell an ADDED key from a CHANGED value, so a checker that compares retained
# bytes to fresh bytes fails on the addition - which is what the full-FFN release gate did.
#
# A KEY-NAME ALLOWLIST IS THE WRONG FIX and the first version of that gate used one. Permitting any
# added key CALLED spill_state admits a FABRICATED non-null one; permitting argument_state admits an
# extra field inside it. Both pass with every native artifact unchanged, which is the case the gate
# exists for. So nothing is tolerated by name here: the expectation is CONSTRUCTED from the retained
# contract's own bindings and boundaries, and the fresh document must equal it exactly.
EXPLICIT_SINCE = ("main_instruction_count", "argument_state", "promoted_ranges", "spill_state")
import hashlib
import json


def with_instruction_count(abi, *, instructions, code):
    instructions = tuple(instructions)
    if not instructions or not code:
        raise ValueError("instruction count needs a nonempty retained program")
    cursor = 0
    for instruction in instructions:
        offset, length = instruction["offset"], instruction["length"]
        if (type(offset) is not int or type(length) is not int or offset != cursor
                or length <= 0 or length % 2):
            raise ValueError("retained instruction boundaries are incomplete or inconsistent")
        cursor += length
    if cursor != len(code):
        raise ValueError("retained instruction boundaries do not cover the exact code")
    count = len(instructions)
    if "main_instruction_count" in abi:
        stated = abi["main_instruction_count"]
        if type(stated) is not int or stated != count:
            raise ValueError("stated instruction count contradicts retained boundaries")
    result = deepcopy(abi)
    result["main_instruction_count"] = count
    return result


AUTHORING_OPTIONS = 'authoring_options'


def _recorded_authoring_options(name, rebuilt_manifest, rebuilt_requirements):
    """The authoring options a rebuild records for one program, or None if it records none.

    ADDITIVE PROVENANCE, NOT A LICENCE TO DIFFER. A delivery written after root's
    authoring_options field records the options it actually authored with; a retained document
    written before the field predates it entirely, so its absence is an age difference rather than
    a disagreement, and demanding equality there would refuse every historical bundle forever.

    The addition is accepted only when the rebuild's manifest and its requirements record the SAME
    options. That is the invariant that makes "which record carries the complete inputs" answerable
    at all: a producer that writes the options into one document and not the other, or writes two
    different things, is refused here rather than leaving a consumer to infer them. The value must
    also be a non-empty object - an empty one records nothing while looking like a record.

    And the allowance never widens what the bytes may be: verify_rebuild returns only once every
    payload has compared equal, so a bundle whose code, object, library or archive moved still
    fails, whatever its documents say.
    """
    record = (rebuilt_manifest.get('programs') or {}).get(name) or {}
    options = record.get(AUTHORING_OPTIONS)
    if options is None:
        return None
    if not isinstance(options, dict) or not options:
        raise ValueError('recorded authoring options are not a non-empty object: ' + name)
    if rebuilt_requirements is not None:
        mirror = (rebuilt_requirements.get(name) or {}).get(AUTHORING_OPTIONS)
        if mirror != options:
            raise ValueError('manifest and requirements record different authoring options: ' + name)
    return options


def verify_rebuild(retained, rebuilt):
    """Verify native identity and explicitly evolve counts in retained ABI documents.

    Both mappings cover the builder's complete payload. Receipts are separate and
    never rewritten. Only named ABI locations may evolve; every other byte stays
    exact. Counts come from retained boundaries covering hash-verified native code.
    """
    if retained.keys() != rebuilt.keys():
        raise ValueError("rebuild payload membership changed")
    expected = {}
    if 'manifest.json' in retained:
        manifest = json.loads(retained['manifest.json'])
        if manifest.get('format') == 'g17-attention-images-v1':
            manifest = deepcopy(manifest)
            programs = manifest['programs']
            requirements = (json.loads(retained['requirements.json'])
                            if 'requirements.json' in retained else None)
            rebuilt_manifest = json.loads(rebuilt['manifest.json'])
            rebuilt_requirements = (json.loads(rebuilt['requirements.json'])
                                    if 'requirements.json' in rebuilt else None)
            for name, record in programs.items():
                code_path = 'programs/' + name + '/program.bin'
                code = retained[code_path]
                if hashlib.sha256(code).hexdigest() != record['sha256']['program.bin']:
                    raise ValueError('retained manifest code hash differs: ' + name)
                def evolve(abi):
                    return with_instruction_count(abi,
                        instructions=record['instructions'], code=code)
                record['abi'] = evolve(record['abi'])
                # The second explicit evolution, beside the instruction count: a record that
                # predates root's authoring_options field gains what the rebuild records. A record
                # that ALREADY names its options is left alone, so it is compared exactly and a
                # rebuild that changes what it says it authored with still fails.
                options = _recorded_authoring_options(name, rebuilt_manifest, rebuilt_requirements)
                if options is not None and AUTHORING_OPTIONS not in record:
                    record[AUTHORING_OPTIONS] = options
                if requirements is not None and name in requirements:
                    requirements[name]['abi'] = evolve(requirements[name]['abi'])
                    if options is not None and AUTHORING_OPTIONS not in requirements[name]:
                        requirements[name][AUTHORING_OPTIONS] = options
                abi_path = 'programs/' + name + '/abi.json'
                if abi_path in retained:
                    expected[abi_path] = evolve(json.loads(retained[abi_path]))
                if len(programs) == 1 and 'abi.json' in retained:
                    expected['abi.json'] = evolve(json.loads(retained['abi.json']))
            expected['manifest.json'] = manifest
            if requirements is not None:
                expected['requirements.json'] = requirements
    for path, old in retained.items():
        new = rebuilt[path]
        if path in expected:
            if json.loads(new) != expected[path]:
                raise ValueError('rebuilt document differs from explicit ABI evolution: ' + path)
        elif new != old:
            raise ValueError('rebuilt payload differs: ' + path)


def argument_state(contract):
    """The argument block a retained contract's own bindings determine, field by field.

    Derived, never defaulted: the pointer offsets are the bindings' own, and they must already be
    `2 * rank` - a contract whose offsets are anything else is not this shape and raises rather
    than being described by a state it does not have.
    """
    bindings = contract["bindings"]
    if not bindings:
        raise ValueError("argument state needs at least one declared binding")
    offsets = [[b["index"], b["offset"]] for b in bindings]
    if [offset for _index, offset in offsets] != [2 * rank for rank in range(len(bindings))]:
        raise ValueError("retained binding offsets are not the measured 2 * rank block")
    indices = [index for index, _offset in offsets]
    return {"pointer_offsets": offsets,
            "pointer_words": 2,
            "block_words": 2 * len(bindings),
            "block_bytes": 8 * len(bindings),
            "basis": "emitted_pointer_offsets",
            "offset_rule": "2 * rank",
            "indices_contiguous": indices == list(range(indices[0], indices[0] + len(indices))),
            "not_stated": ["per_kernel_slot_1"]}


def with_explicit_fields(contract, *, instructions=None):
    """A retained contract plus exactly the fields that became explicit, with derived values.

    `instructions` is the retained boundary count for this program - the manifest's own list - and
    is required only when the caller wants `main_instruction_count` filled. A value already stated
    that CONTRADICTS the derivation raises, which is the rule `with_instruction_count` set for the
    count and the reason this is an evolution rather than a merge.
    """
    result = deepcopy(contract)
    derived = argument_state(contract)
    stated = result.get("argument_state")
    if stated is not None and stated != derived:
        raise ValueError("stated argument state contradicts the retained bindings")
    result["argument_state"] = derived
    for name in ("promoted_ranges", "spill_state"):
        if result.get(name) is not None:
            raise ValueError("retained contract already states a non-null " + name)
        result[name] = None
    if instructions is not None:
        count = len(instructions) if not isinstance(instructions, int) else instructions
        if type(count) is not int or count < 0:
            raise ValueError("retained instruction count is not a count")
        present = result.get("main_instruction_count")
        if present is not None and present != count:
            raise ValueError("stated instruction count contradicts retained boundaries")
        result["main_instruction_count"] = count
    return result


def evolution_differences(retained, fresh, *, instructions=None):
    """[] when `fresh` is `retained` plus exactly the explicit fields, else what differs.

    The caller gets a list rather than a bool so a failure names its path. Anything else - a
    changed value under an added key, an extra field inside one, a key added at a path it never
    appears at, a removed key, a changed value anywhere - is a difference.
    """
    expected = with_explicit_fields(retained, instructions=instructions)
    return _differences(expected, fresh)


def _scalar_type(value):
    """The JSON type of a scalar, with bool kept apart from the integer it equals in Python."""
    if value is None or isinstance(value, bool):
        return type(value)
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    return type(value)


def _differences(expected, actual, path=""):
    # PYTHON EQUALITY ERASES JSON TYPES AND THIS COMPARISON USED IT. `False == 0` and `1 == 1.0`
    # are true, so a retained `"written": false` that came back as `"written": 0`, an index that
    # became `1.0`, or an offset that became `false` all compared EQUAL and the check returned no
    # differences. Integration caught it with three controls. A comparison that cannot disagree is
    # the defect this whole rule exists to prevent, and it was sitting inside the rule itself.
    #
    # The type is compared before the value, and `bool` is kept apart from `int` rather than folded
    # into it, because JSON has both and a contract that swaps one for the other has changed.
    out = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                out.append(path + "/" + key + " is not an expected addition")
            elif key not in actual:
                out.append(path + "/" + key + " is missing")
            else:
                out += _differences(expected[key], actual[key], path + "/" + key)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(path + " differs in length")
        else:
            for i, (a, b) in enumerate(zip(expected, actual)):
                out += _differences(a, b, path + "/%d" % i)
    elif _scalar_type(expected) is not _scalar_type(actual):
        out.append("%s changed JSON type: %s -> %s"
                   % (path, _scalar_type(expected).__name__, _scalar_type(actual).__name__))
    elif expected != actual:
        out.append(path + " differs")
    return out


# THE ENTRY POINT IS LAST, not beside the function it calls. When the two halves of this
# module were merged the dispatch stayed where tools/g17abi.py had it, with the retained-ABI
# evolution appended after - so running the module as a script would execute it before those
# names were bound. main() happens not to read them today, which is exactly how that defect
# survives until someone extends main().
if __name__ == "__main__":
    main()
