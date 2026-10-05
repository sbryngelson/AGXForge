#!/usr/bin/env python3
"""AUTHOR A COMPLETE OBJECT - all five sections - for a program the compiler hands down.

Reproducing Apple's objects byte-exactly is capped: some of the five sections are the compiler's
inputs, and this side established that by elimination rather than by giving up. So this emits an
object for a NEW program rather than a copy of an old one, which is what a linker does.

THE CAP WAS THREE SECTIONS AND IS NOW TWO.

  __GPU_LD_MD        the format is this side's and the entry PC is named causally, but the main
                     table's SLOT SET and its values are a function of nothing this side holds.
  __GPU_ARCH_LD_MD   one boolean, set in 6,234 of 21,001 objects, that no contract fact and no
                     opcode determines - seven of each tested, the best leaving 3,144 exceptions.
  __GPU_STATS_MD     RETIRED AS AN INPUT. It is 96 bytes with 20,999 distinct patterns over
                     20,999 objects, so a byte census could only ever say that it varies. A
                     dispatch says more: tools/g17statsprobe.py sets it to zeros, to 0xFF, to a
                     BUFFER kernel's own 96 bytes, and removes it entirely, and the authored
                     texture read stays exact through all four - in a harness where the same
                     four mutations of each of the other three sections kill Apple's driver
                     inside newComputePipelineState. The loader does not read it, so this side
                     emits it.

THE COMPILER-OWNED INPUTS ARE PARAMETERS AND THIS REFUSES WITHOUT THEM. Not defaults, not a
constant measured on buffer kernels, and not a donor object - a constant measured on buffer kernels
is exactly what put a 224-byte __GPU_LD_MD into a texture image on 2026-09-08. Anything this side
cannot derive is named in the handover as a copied field, and the list is printed with the object.

AND NOTE WHAT THE SAME PROBE FOUND ABOUT THE OTHER TWO: a foreign but structurally valid
__GPU_LD_MD - the buffer shape in a texture kernel's image, the exact defect shipped on
2026-09-08 - BUILDS A PIPELINE WITHOUT COMPLAINT. There is no load-time error to catch it. That
is why these are refused rather than defaulted.

    python3 tools/g17authorobj.py --demo    author one and report its field ledger
"""
import collections
import json
import os
import struct
import sys


SECTIONS = ("__GPU_METADATA,__compute", "__GPU_LD_MD,__compute", "__GPU_ARCH_LD_MD,__compute",
            "__GPU_STATS_MD,__compute", "__GPU_REMARKS_MD,__compute")


class Missing(KeyError):
    """A compiler-owned input was not supplied. Refusing beats defaulting."""


# THE PER-KERNEL SLOTS THAT ARE POINTERS, NOT SCALARS - MEASURED, not inherited from a docstring.
#
# Over 6,000 cached sections, ten slots hold a value that lands in-bounds as a relative offset AND
# whose target reads as a length: 2, 4, 6, 8, 10, 12, 13, 26, 27 and 29, each at 100% (13 and 2 at
# 99.8%). No other slot reaches 6%. The first version of this guard listed SEVEN of them - the set
# g17facts records for the vector slots - and the three it missed, 13, 27 and 29, are exactly the
# three that kept every authored section a few dozen bytes short of Apple's.
#
# Writing a caller's integer into one of these produces a table whose slot points at an address
# that is not a vector - a DANGLING POINTER in a section the loader walks. That is not a wrong
# number, it is the failure class that hung this GPU on 2026-09-08, and until this guard existed
# `author` accepted it silently: handed slot 6 = 132 out of a real Apple section it emitted 132 as
# a scalar and returned a section 64 bytes short, with no complaint.
#
# THIS SIDE AUTHORS SEVEN OF THE TEN. Slots 6, 8, 10 and 12 are empty vectors (length 0 in 2,828
# of 2,828), and 2, 4 and 26 are built from the binding list. Slots 13, 27 and 29 carry real
# content this side does not model - over 6,000 sections slot 13 is length >= 4 in 79.7%, slot 27
# is length 0 in 81.4% and length 1 in 18.3%, slot 29 is length 1 in 54.5% and length 0 in 33.3%
# - so an authored section REFUSES rather than shipping three fields pointing at nothing.
POINTER_SLOTS = frozenset({2, 4, 6, 8, 10, 12, 13, 26, 27, 29})
VECTOR_SLOTS = POINTER_SLOTS          # the old name, kept because callers pass it


# A BUFFER'S ELEMENT TYPE DOES NOT REACH __GPU_METADATA, and that is measured, not assumed.
#
# THE EVIDENCE. Over 12,426 corpus kernels, 151 of them share ONE byte-identical metadata blob
# while their declared type sets run from ['float'] alone to nineteen distinct types including
# half, bfloat, short, long, ushort and the vector forms - and two further blobs of 167 and 110
# kernels straddle half and float-without-half the same way. The tightest single control is
# atm-i-add-tg-used-1-r0 against atm-u-add-tg-used-1-r0: one variable, atomic_int to atomic_uint,
# and the metadata is byte-identical.
#
# AND THE APPARENT COUNTEREXAMPLE IS CONFOUNDED. Holding the binding structure fixed, half and
# float kernels DO show different per-kernel slot sets - but the slots that differ are 18 and 28,
# uses_threadgroup and its allocation size, because this corpus's half kernels are
# threadgroup-heavy. That is a known non-type fact with its own ABI input, not a type field.
#
# NO BINDING RECORD CARRIES A TYPE FIELD EITHER: across 9,207 records in 6,001 compiled objects,
# every one has a slot set within {0, 1, 2, 3} and not one carries slot 4.
#
# SO THE REQUIRED TYPED-RESOURCE METADATA FOR A SCALAR BUFFER IS NONE. This side consumes
# element_type, records that finding beside the object, and refuses a type the corpus has never
# declared - because "not encoded" is a claim about the 24 types below, not about every type.
NOT_ENCODED_TYPES = frozenset({
    "float", "float2", "float4", "half", "half2", "half4", "bfloat", "bfloat2", "bfloat4",
    "int", "int4", "uint", "uint2", "uint4", "short", "short2", "ushort",
    "long", "ulong", "uchar", "uchar4", "atomic_int", "atomic_uint", "atomic_float"})
# these ARE declared in the corpus and are NOT buffer element types - an imageblock has its own
# per-kernel slot 19, and AB/P/S are argument-buffer and struct resources with a different shape
OTHER_RESOURCES = frozenset({"imageblock", "AB", "P", "S"})
# and the ones this file computes for itself; a caller supplying them is overriding a derivation
DERIVED_SLOTS = frozenset({1, 2, 3, 4, 26})


# ------------------------------------------------------------------------------------------
# MEASURED ARCHITECTURAL CLASSES, represented explicitly rather than approached by default.
#
# The general path above is calibrated on the cached Apple objects and reaches 99.6% byte-exact on
# 3,000 of them. That number did NOT predict the executed scalar class, and the difference is not
# small: measured from the delivered bytes of scalar-buffer-two-bindings-measured-v2,
#
#     per-kernel table declares tlen 0        the general path computes the natural extent
#     slot 12 ABSENT                          the general path emits four q-vectors 6/8/10/12
#     slot 26 present but EMPTY               the general path emits a constant-program record
#     slots 15, 16, 27, 29 absent             the general path emits 27/29 on request
#     slot 1 = 0, present                     the general path requires pk_slot1 and refuses
#     kind-6 f2 + f3 = 24 while slot 1 = 0    REFUTES the sum law measured at 15,777 of 15,777
#                                             over the cache; it was a regularity of that
#                                             population and not a property of the format
#
# and 192 of its 500 metadata bytes are unused padding, which a tightly-packing author cannot
# invent. Three of those differences are exactly the v1 image's profile - no slot 13, no kind-6,
# and a slot-26 record the measured scalar leaves empty - so the general path WOULD have built it.
# No single one of them is blamed for that crash: the ledger records four differences repaired
# together and no causal isolation.
#
# So a measured class is emitted from the serializers that were measured, and this refuses any
# combination it has not been shown. The frozen bytes are the contract, not the shape.
# ABI v5 adds the execution requirement (simd_width, tensor). Accepting the version is not the
# same as being able to author what it carries: see _execution_requirement below.
# v6 adds the resources block a texture program needs: the internal records with their Apple
# indices and ranks, the textures with no rank, an explicitly empty sampler list, the spill basis,
# and a not_stated list naming the section facts the compiler will not guess. Reading a version
# this side does not know is refused by number, which is why adding it is a deliberate edit.
ABI_VERSIONS = frozenset({2, 3, 4, 5, 6, 7})

PK_EMPTY_VECTORS = (6, 8, 10, 12)

PROFILES = {
    "scalar-buffer-two-bindings-measured-v2": {
        "bindings": ((1, 0, False), (2, 2, True)),
        "entry": 64,
        "arch_flag": False,
        "reference": "isa/g17-scalar-abi-v2.json",
        "note": "two device buffers, readonly at pointer-block offset 0 and written at 2",
        "evidence": "NINE hardware runs - 1x1, 3x7 and 33x384, three isolated runs each, every "
                    "FP32 score matching sequential CPU arithmetic bit for bit, through the "
                    "integration owner. isa/g17-packed-scan-validation.json.",
    },
}


def profile_sections(name, bindings, entry, arch_flag=None):
    """The five sections of a measured class, from repository-owned serializers.

    REFUSES anything the class was not measured with. A profile that quietly accepted a different
    binding list or entry would be the borrowed default this whole file exists to prevent, and the
    v1 image is what that looks like when it reaches a GPU.
    """
    from . import ldmd as g17ldmd
    from . import mdgen as M
    prof = PROFILES.get(name)
    if prof is None:
        raise ValueError("no measured profile named %r; known: %s"
                         % (name, ", ".join(sorted(PROFILES))))
    got = tuple(tuple(b[:3]) for b in bindings)
    if got != prof["bindings"]:
        raise ValueError("profile %s is measured for bindings %s; got %s. It is a compatibility "
                         "contract for one class, not a template to re-point."
                         % (name, list(prof["bindings"]), list(got)))
    if entry != prof["entry"]:
        raise ValueError("profile %s pins entry %d; got %d" % (name, prof["entry"], entry))
    if arch_flag is not None and bool(arch_flag) != prof["arch_flag"]:
        raise ValueError(
            "profile %s ships __GPU_ARCH_LD_MD with the flag %s and that section has nine hardware "
            "runs behind it; the caller says %s. The general author and the frozen profile must "
            "not be silently interchangeable - route through one or the other deliberately."
            % (name, "ELIDED" if not prof["arch_flag"] else "SET",
               "SET" if arch_flag else "ELIDED"))
    led = collections.OrderedDict()
    led["profile"] = "%s - %s" % (name, prof["note"])
    led["execution evidence"] = prof.get(
        "evidence", "none recorded for this profile")
    # restore_swept=False is NOT a default. The default differs from the delivered bytes, so a
    # caller who takes it gets a section that was never the one validated.
    # NO FROZEN SECTIONS. Every one of the five is emitted by a serializer from the measured class
    # and the compiler's entry, and then checked byte-for-byte against the delivered object below.
    # Two of them used to be constants copied out of that object - g17imgconst_scalar.GPU_LD_MD and
    # STATS_MD - and returning a copy of the artefact is not authoring it, however well it verifies.
    #
    # Neither needed to be frozen. g17ldmd.build(entry) reproduces the delivered 216 LD bytes
    # exactly, sha 1b14207cdf6df455 both ways, and STATS is 96 zeros on every executed image.
    indices = [b[0] for b in bindings]
    layout = M.layout_for(indices)
    if layout is None:
        raise Missing("no measured metadata layout for the binding indices %s" % (indices,))
    md = M.build(indices, layout=layout, restore_swept=False)
    led["__GPU_METADATA"] = ("MEASURED CLASS: g17mdgen.build(%s) on the layout measured for these "
                             "binding indices" % (indices,))
    led["__GPU_LD_MD"] = ("DERIVED from the compiler's entry: g17ldmd.build(entry=%d), byte-identical "
                          "to the delivered 216 bytes" % entry)
    arch = _arch(prof["arch_flag"], led)
    # THE LEDGER DESCRIBED A TEMPLATE AND THIS PATH EMITS ZEROS. The template belongs to the
    # reproduction path, where an unswept Apple object is the target; the bytes DISPATCHED for
    # this class are ninety-six zeros - what the nine hardware runs ran and what
    # _verify_against_delivered checks against the delivered object. A ledger describing bytes
    # the image does not hold is the record a reader trusts over the artefact. The bytes are
    # unchanged; only the description is.
    led["__GPU_STATS_MD"] = ("EMITTED: 96 zero bytes, which is what this class delivered and "
                             "what the nine hardware runs executed. The 90-byte template "
                             "belongs to the reproduction path.")
    led["__GPU_REMARKS_MD"] = "DERIVED: empty"
    out = {SECTIONS[0]: bytes(md), SECTIONS[1]: bytes(g17ldmd.build(entry)), SECTIONS[2]: bytes(arch),
           SECTIONS[3]: bytes(96), SECTIONS[4]: b""}
    _verify_against_delivered(name, prof, out, led)
    return out, led


def _verify_against_delivered(name, prof, out, led):
    """Check every emitted byte against the DELIVERED object this class was measured from.

    A named measured class is the one place this file returns bytes chosen by a class rather than
    derived from a contract, and the goal it exists under says no per-kernel patches. The honest
    difference between a class and a patch is whether anything checks it: a patch is trusted, and a
    class is verified against the artefact it claims to reproduce, on every author, or it is a
    patch wearing a better name.
    """
    ref = prof.get("reference")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ref or "")
    if not ref or not os.path.exists(path):
        led["measured-class verification"] = ("NOT VERIFIED: %s names no readable reference, so "
                                              "its bytes are trusted rather than checked" % name)
        return
    doc = json.load(open(path))
    sections = doc.get("sections") or {}
    checked = 0
    for key, got in out.items():
        rec = sections.get(key.split(",")[0])
        if rec is None:
            continue
        want = bytes.fromhex(rec["hex"])
        if got != want:
            raise ValueError(
                "measured class %s emitted %d bytes for %s that differ from the DELIVERED object "
                "recorded in %s (%d bytes). This class returns bytes because they were measured; "
                "if they no longer reproduce, the class is wrong and must not be authored."
                % (name, len(got), key.split(",")[0], ref, len(want)))
        checked += 1
    led["measured-class verification"] = (
        "%d of 5 sections checked byte-for-byte against the delivered object in %s"
        % (checked, ref))


def _register_count(abi):
    """The program's own register count, delivered by the compiler and never decoded here.

    Per-kernel slot 0 is the highest 32-bit register index named in _agc.main plus one, excluding
    the constant program: 2,022 of 2,022 corpus objects once the count is taken over main alone.
    Over main plus the constant program it is 1,942, and all 80 exceptions are that difference.

    It is DELIVERED rather than read off the bytes because recovering "the highest register any
    operand names" means resolving operand kinds and register naming, which is instruction
    semantics this layer does not own and must not rediscover. Absence is refused rather than
    defaulted: falling back to a decode would turn a disagreement between two independent sources,
    which is a finding, into silence, and falling back to the class constant would emit a value
    that is correct only for the object the class was measured from.
    """
    value = abi.get("register_count")
    if value is None:
        raise Missing(
            "the compiler must deliver register_count - the highest 32-bit register index named "
            "in _agc.main plus one, excluding the constant program. This layer will not decode it "
            "from the text, and will not fall back to the measured class's witness value, which "
            "is right for that witness and wrong for every other program in its class")
    if type(value) is not int or value < 1:
        raise Missing("register_count must be a positive integer, not %r" % (value,))
    return value


# THE EXECUTION REQUIREMENT IS A LAUNCH FACT, ON THE COMPILER OWNER'S STATEMENT.
#
# This side refused it first, because a contract stating something the section does not carry is
# how the system-register and threadgroup drops happened. The refusal named the two ways to close
# it - a class measured to hold it, or a statement that it belongs at the launch boundary - and
# the compiler owner gave the second in writing (handoff 9e, 673055ae): `execution` states what
# the DISPATCH must satisfy, like `exact_grid_required`, and asserts nothing about any slot.
#
# So it is not encoded here, and it is not silently ignored either: it is recorded in the field
# ledger, so a delivered image says in its own manifest that a launch requirement exists and what
# it is. Enforcing it against an actual group size is the runtime boundary's job, not this one's.
#
# THE BASIS IS A STATEMENT, NOT A MEASUREMENT, and the owner said so. What this side can check is
# only that it is not refutable from the bytes: the tensor section does differ from a non-tensor
# one, at 488 bytes against 456 with slots 32, 33 and 44 present, but those are class facts
# measured on objects, and with a single width (32) in evidence no field could be shown to vary
# with width in either direction. So the bytes neither confirm nor deny it.
MEASURED_EXECUTION_CLASSES = ()


def _execution_requirement(abi, led=None):
    execution = abi.get("execution")
    if execution is None:
        return None
    stated = dict(execution) if not hasattr(execution, "simd_width") else dict(
        simd_width=execution.simd_width, tensor=execution.tensor)
    if led is not None:
        led["execution requirement"] = (
            "LAUNCH BOUNDARY, not encoded in this section: %r. Carried on the compiler owner's "
            "statement (handoff 9e), not on a measurement; the dispatch must satisfy it and the "
            "runtime boundary enforces it." % (stated,))
    return stated


def _general_can_carry(registers, abi=None):
    """True when the general serializer can place this register set's slot-29 vector itself.

    The measured classes below refused a set they were not witnessed for because the general
    serializer used to write no slot-29 vector, so falling through dropped the declaration. It now
    derives the vector from g17mdgen.slot29_entries - the law that predicts Apple's own vector in 37
    of 43 fixed197 sources - so a set the law maps is carried, not dropped, and only a set outside
    the law still has to refuse (MM 25.149)."""
    from . import mdgen as M
    # The general serializer needs the argument block's size (pk_slot1) from the contract; a contract
    # that states none cannot be serialized there, so it keeps the measured class's refusal.
    if abi is not None and "pk_slot1" not in abi:
        return False
    try:
        M.slot29_entries(tuple(registers))
        return True
    except (ValueError, KeyError):
        return False

def author(text, entry, bindings, abi):
    """(sections, ledger) for one program.

    `bindings` is [(index, offset, written)] in the contract's order - the order is load-bearing,
    since a load or store encodes a RANK into this list. `abi` carries the compiler-owned inputs.

    THREE WERE REQUIRED; TWO ARE. __GPU_STATS_MD was the third, and a dispatch retired it: the
    loader does not read that section, so this side emits it. `stats` is still accepted and still
    width-checked when the backend supplies it.
    """
    from . import emit as E
    from . import ldmd as g17ldmd
    from . import mdgen as M
    from . import schema as S
    # Requantization is admitted below only for the explicitly measured three-buffer scalar class.
    # The older embedded-scale two-buffer primitive has no class and still refuses by name; routing
    # it through the generic serializer would drop its SR declaration and create a plausible but
    # unmeasured image.
    if abi.get('execution') is not None:
        from . import tensormetadata as g17tensormetadata
        try:g17tensormetadata.validate(abi,bindings)
        except ValueError as error:raise Missing(str(error)) from error
        abi=dict(abi,pk_slot1=8)
    # Cooperative contracts require their measured resource class. Ordinary FOUR
    # omits slots 18/28 and would silently declare no scratchpad. Validate the
    # complete supported signature before emitting any section.
    if abi.get("uses_threadgroup"):
        from . import cooperativemetadata as g17cooperativemetadata
        try:
            g17cooperativemetadata.validate(abi, bindings)
        except ValueError as error:
            raise Missing(str(error)) from error
        # Derived from this class's kind-6 and pointer-block records, not FOUR's - but only for
        # the 16-byte pool form. A pool-carrying cooperative section carries 40 there, which is
        # not derivable from the pool length (8 + pool/4 fits both points and 154 of 5,685 census
        # sections), so the contract's own value stands and the class refuses without it.
        #
        # AND IT IS THE SELECTED CLASS'S OWN VALUE, not a cooperative constant: the three-binding
        # threadgroup class carries 8 at the same empty pool where the four-binding one carries
        # 12. Filling in 12 here regardless made a contract this side had measured refuse at the
        # emitter for stating the value its own controls show.
        if abi.get("pk_slot1") is None:
            pool = g17cooperativemetadata._pool_bytes(abi)
            if pool is not None and len(pool) == 16:
                abi = dict(abi, pk_slot1=g17cooperativemetadata.empty_pool_slot1(bindings))
    # `profile` IS A LABEL, NOT A ROUTING KEY, agreed with the compiler owner: it attributes an
    # execution claim to an architectural class and must not select an authoring path. While it
    # routed, there were two authoring paths by construction and the compiler decided which one ran.
    #
    # The measured class is still reachable, by an explicit request from THIS side - author the
    # class deliberately or author generally, never by reading a label out of the contract. The
    # short-circuit survives only until the general path reproduces all 844 bytes, and the reason
    # it survives at all is that those bytes carry the only execution evidence either side has.
    if abi.get("measured_class"):
        return profile_sections(abi["measured_class"], bindings, entry, abi.get("arch_flag"))
    if abi.get("profile") and not abi.get("measured_class"):
        # A KEY THAT USED TO ROUTE MUST NOT SILENTLY STOP ROUTING. Ignoring `profile` here would
        # hand a caller who asked for the measured class a general-path section that is valid,
        # different, and never executed - the quietest possible version of the substitution this
        # file exists to prevent. So the transition is a refusal with the fix in it, not a default.
        raise ValueError(
            "abi['profile']=%r is a LABEL and no longer selects an authoring path. If you want the "
            "measured class's bytes, ask for them: abi['measured_class']=%r. If you want the "
            "general path, drop 'profile' or pass measured_class=None explicitly."
            % (abi["profile"], abi["profile"]))
    led = collections.OrderedDict()

    # ELEMENT TYPES, when the caller states them. A binding may be (index, offset, written) or
    # (index, offset, written, element_type); the fourth is consumed rather than ignored.
    etypes = [b[3] for b in bindings if len(b) > 3 and b[3] is not None]
    for t in etypes:
        if t in OTHER_RESOURCES:
            raise ValueError(
                "%r is not a buffer element type - it is a different resource shape with its own "
                "metadata. An imageblock is not a binding: state abi['imageblock'] = {'layout': "
                "'explicit', 'element_bytes': N} and scanlink.link declares it (per-kernel slot 19, "
                "LD_MD t136 slot 23 and the ARCH element size; agxforge.g17.imageblock)" % t)
        if t not in NOT_ENCODED_TYPES:
            raise ValueError(
                "the corpus never declares element type %r, so 'the type is not encoded' has no "
                "witness for it; refusing rather than assuming it behaves like the 24 measured "
                "types" % t)
    bindings = [tuple(b[:3]) for b in bindings]

    # THE REQUIRED SET IS ONE KEY, and it shrank because two of the four were never the backend's.
    #
    # ld_md_slots and ld_md_values: __GPU_LD_MD is a function of the ENTRY PC, which the backend
    # does supply. g17ldmd.build(entry=64) reproduces the executed class's 216 bytes byte for byte,
    # sha 1b14207cdf6df455 both ways. Demanding them was stricter than this side's own evidence,
    # and the compiler was right to say it had no measurement that determines them. They are still
    # honoured when supplied, because a backend that HAS measured its own slot set should be able
    # to state it rather than accept a derivation.
    #
    # pk_slot1: the per-kernel slot 1 is a ONE-byte field at table offset 48 whose presence is
    # decided by the layout for the binding count, not by the program. It comes from the layout.
    #
    # arch_flag stays required and stays the backend's. Nothing here can derive it: the recovered
    # rule needs constructs that emit no instruction at all.
    # THE ABI CARRIES ITS OWN VERSION and an unknown one is refused rather than read optimistically.
    # A contract this side does not know may have moved a key's meaning while keeping its name, and
    # every check below reads keys by name.
    # A MINIMAL CONTRACT DOES NOT DETERMINE THE CLASS, and authoring one anyway is how 291 of 293
    # out-of-sample objects got confidently wrong sections. ab_base and addr_base carry the SAME
    # binding contract as the executed scalar class - (1,0,False),(2,2,True) at entry 64 - and a
    # different class entirely: 384 bytes with fifteen per-kernel slots against 340 with nine. No
    # fact in the bindings separates them.
    #
    # What does separate them is what the PROGRAM does, and a real contract carries that. So the
    # program-property facts are required before this side will author generatively; without them
    # the honest answer is that the class is unknown. An explicit measured_class request bypasses
    # this, because naming a class IS pinning it.
    if not abi.get("measured_class"):
        # ANY sufficient fact set pins the class, not only the program properties. A caller that
        # states ld_md_slots or the per-kernel values has pinned it by a different and equally
        # legitimate route; what is refused is a contract that pins it by NOTHING.
        PINS = ("has_stores", "uses_threadgroup", "writes_buffer", "writes_texture",
                "ld_md_slots", "ld_md_values", "pk_slot1", "pk_values", "pk_vectors", "pk_extra",
                "v0_field3", "slot2_extra", "slot2_kind6", "slot2_kind6_f2", "constant_program")
        stated = [k for k in PINS if k in abi]
        if not stated:
            raise Missing(
                "the class is not determined by bindings and entry alone - objects with identical "
                "binding contracts carry different metadata classes - so authoring from them would "
                "emit a confidently wrong section. State the program properties (has_stores, "
                "uses_threadgroup, writes_buffer, writes_texture), or name a measured_class.")
    version = abi.get("abi_version")
    if version is not None and version not in ABI_VERSIONS:
        raise ValueError("abi_version %r is not one this side knows (%s); refusing to read a "
                         "contract whose key meanings may have moved under their names"
                         % (version, ", ".join(str(v) for v in sorted(ABI_VERSIONS))))
    if version == 7:
        # v7 adds the uniform preload: the terms it sums, the element type, the lifetime, and the
        # consumer in main. What it deliberately does NOT carry is the block offset - that is this
        # side's derivation from the binding list, agreed with the compiler after their own
        # publish_const parameter turned out to be a second copy of it. What it DOES carry is the
        # constant main's op10282 reads, which is a restatement of the code's own bytes and is here
        # to be CHECKED against the derivation rather than trusted.
        block = abi.get("resources")
        if not isinstance(block, dict):
            raise Missing("ABI v7 states a resources block")
        # `entry` is this function's entry PC. A first version named the loop variable `entry`
        # too and overwrote it with a dict, so author() reached `entry % 64` and raised TypeError
        # before it could refuse for the named unresolved storage facts - a crash where a refusal
        # was owed. Integration found it on the positive path.
        for preload in block.get("preloads") or ():
            if not preload.get("terms"):
                raise Missing("a preload states the terms it sums; a single binding cannot express "
                              "S2 and S3, whose preloads fold two and three uniformly read buffers "
                              "into one publication")
            consumer = preload.get("consumer") or {}
            constant = consumer.get("block_constant")
            if constant is None:
                raise Missing("a preload states its consumer's block constant so this side can "
                              "check it against the block layout; without it the check has nothing "
                              "to compare")
    if version in (6, 7):
        # THE BLOCK IS REQUIRED AT THIS VERSION, not optional within it. A v6 contract without it
        # is a texture program whose user bindings have no stated ranks, and a rank guessed from
        # the user ordinal puts the store into an internal - the failure that hung the device once
        # and is recorded in g17texsection's own first paragraph.
        block = abi.get("resources")
        if not isinstance(block, dict):
            raise Missing("ABI v6 states a resources block; without it a texture program's user "
                          "bindings have no ranks, and a rank taken from the user ordinal indexes "
                          "into an internal")
        if block.get("samplers") is None:
            raise Missing("ABI v6 must state samplers explicitly, even as an empty list, so a "
                          "sampler is refused by name rather than ignored")
        for texture in block.get("textures") or ():
            if texture.get("rank") is not None:
                raise ValueError("a texture carries no binding rank: the binding vector of every "
                                 "measured witness holds buffers and internals only")
    if version in (3, 4, 5, 6, 7):
        registers = abi.get("system_registers")
        if not isinstance(registers, (tuple, list)):
            raise Missing("ABI v3 requires system_registers from instruction selection")
        if any(type(r) is not int or not 0 <= r <= 255 for r in registers) or \
                tuple(registers) != tuple(sorted(set(registers))):
            raise ValueError("system_registers must be sorted, unique 8-bit register indices")
    # ABI v2 travels through JSON, where numeric mapping keys become strings.
    # Normalize at this boundary without mutating the compiler-owned mapping or
    # silently dropping two different keys that normalize to the same slot.
    if "pk_values" in abi:
        values = {}
        for key, value in abi["pk_values"].items():
            if type(key) is int:
                slot = key
            elif isinstance(key, str) and key.isdecimal() and str(int(key)) == key:
                slot = int(key)
            else:
                raise ValueError("pk_values requires canonical integer slot keys")
            if slot in values:
                raise ValueError("pk_values keys collide after JSON normalization")
            values[slot] = value
        abi = dict(abi, pk_values=values)
    for need in ("arch_flag",):
        if need not in abi:
            raise Missing("the backend did not supply %r; this side cannot derive it" % need)
    if "pk_slot1" not in abi:
        abi = dict(abi)
        from . import metaclass as g17metaclass
        measured = g17metaclass.select(bindings, abi)
        if measured is not None:
            abi["pk_slot1"] = measured["pk_extra"].get(1, ("<I", 0))[1]
        elif _takes_texture_route(abi):
            # THE TEXTURE ROUTE OWNS SLOT 1 FOR THIS CONTRACT, so nothing is derived here and
            # nothing is refused here either: the route reads it as resources.argument_bytes and,
            # when that is not stated, names it among the facts it will not guess.
            #
            # What this REPLACES is a donor value. _pk_slot1 asks the BUFFER layout generator,
            # keyed on the declared-buffer count alone, so S1 and S2 - one texture, two texture
            # internals, two and three declared buffers - were handed the scalar two- and
            # three-buffer classes' slot 1 and carried on, while S3's four bindings refused with
            # "this class's layout has exactly 2 binding records", a buffer-class sentence about a
            # texture contract. One defect, two faces. Slot 1 is not a resource fact in any case:
            # 326 of 1,341 corpus groups with identical resource structure carry differing values,
            # so no binding count could have determined it.
            led["per-kernel slot 1"] = ("LEFT TO THE TEXTURE ROUTE: a texture contract's slot 1 is "
                                        "resources.argument_bytes, not the buffer layout "
                                        "generator's class of the same declared-buffer count")
        elif _texture_shape(abi):
            # A texture-shaped contract that does NOT take the texture route - samplers with no
            # texture - has no measured class anywhere, and the buffer generator's answer would be
            # another class's number. Refuse by naming the shape.
            raise Missing(
                "this is a %s and no measured metadata class covers it, so slot 1 "
                "(argument_bytes) cannot be derived here. It is NOT the scalar class of the same "
                "declared-buffer count: the buffer layout generator would answer for a different "
                "class, and slot 1 is not a resource fact in any case - 326 of 1,341 corpus groups "
                "with identical resource structure carry differing slot-1 values. The backend must "
                "state pk_slot1, or this shape needs its own measured class."
                % _texture_shape(abi))
        else:
            # A CLASS THAT STATES ITS OWN SLOT 1 ANSWERS FOR ITSELF. `_pk_slot1` derives the value
            # from the layout for a binding COUNT, which is right where one count means one class
            # and wrong where the write mask selects between classes - and it refuses a count it
            # has no layout for, which is how the one-binding contract was refused before this
            # class existed. Asking with the contract's own mask and instruction facts lets the
            # selected class supply the value it was measured with, and a class that carries no
            # slot 1 of its own still falls through to the derivation below.
            _sel = M.layout_for([b[0] for b in bindings],
                                promoted_ranges=abi.get("promoted_ranges"),
                                written=[bool(b[2]) for b in bindings],
                                instructions=abi.get("instruction_count"),
                                back_edge=bool(abi.get("has_back_edge")))
            _own = (_sel or {}).get("pk_extra", {}).get(1)
            if _own is not None and len(bindings) == 1:
                abi["pk_slot1"] = _own[1]
                led["slot 1"] = (
                    "MEASURED for this class: %d. It is NOT a one-binding formula - one binding at "
                    "offset 0 takes 4, 8, 12 and 16 across 21,020 corpus sections - so it is the "
                    "value this class's own controls carry and nothing derives it." % _own[1])
            else:
                abi["pk_slot1"] = _pk_slot1(bindings, led)
    # STATS IS NO LONGER REQUIRED, and the reason is a dispatch rather than a preference. See
    # tools/g17statsprobe.py: on the authored texture kernel, __GPU_STATS_MD set to zeros, to
    # 0xFF, to a BUFFER kernel's own 96 bytes, and removed entirely each leave the texture read
    # exact - in a harness where the identical mutation of __GPU_METADATA, __GPU_LD_MD and
    # __GPU_ARCH_LD_MD kills Apple's driver inside newComputePipelineState. So emitting zeros
    # here is a construction backed by a measurement, not a default standing in for an unknown.
    # If the backend DOES supply it, it is still checked against the corpus width.
    if "stats" in abi and len(abi["stats"]) != 96:
        raise ValueError("__GPU_STATS_MD is 96 bytes in all 21,001 corpus objects; got %d"
                         % len(abi["stats"]))
    if entry % 64:
        raise ValueError("the entry PC is a multiple of 64 in all 21,001 corpus objects")
    from . import metaclass as g17metaclass
    if abi.get('execution') is not None or abi.get("uses_threadgroup") or g17metaclass.select(bindings, abi) is not None:
        if entry != abi["entry"]:
            raise ValueError("entry argument differs from the compiler's measured-class contract")
        prologue = abi["prologue"]
        if isinstance(prologue, str):
            prologue = bytes.fromhex(prologue)
        if text and text[:entry] != prologue:
            raise ValueError("text prologue differs from the compiler's measured-class contract")

    out = {}

    # ---- __GPU_METADATA: every field derived from the contract -------------------------------
    # ONE SERIALIZER, WITH MEASURED CLASS DATA, wherever the class is complete enough to emit its
    # own content. A layout that carries the symbol names is a full class and reproduces Apple's
    # section byte-for-byte; the swept executed class carries none and is emitted the other way.
    # This is what the assignment means by using the same serializer with measured class data
    # rather than a per-kernel patch: the difference between the two paths is DATA.
    from . import mdgen as _MD
    # REPRODUCING AN APPLE OBJECT AND AUTHORING AN IMAGE FOR DISPATCH ARE DIFFERENT JOBS, and
    # collapsing them broke the delivered separate_scan: restoring a class's swept regions is right
    # when the target is Apple's bytes, and wrong when the target is our own dispatch, whose LD must
    # match the measured device-buffer launch contract that has hardware behind it. So it is asked
    # for, not assumed.
    _repro = bool(abi.get("reproduce_measured_class"))
    # THE RANGE IS PART OF THE SELECTION, not an afterthought. Five binding records with a declared
    # promoted range is the SIX-buffer class; the same five with none is the five-buffer one, and
    # the count cannot tell them apart.
    _lay = _MD.layout_for([b[0] for b in bindings], promoted_ranges=abi.get("promoted_ranges"),
                          written=[bool(b[2]) for b in bindings])
    if abi.get("uses_threadgroup") or abi.get('execution') is not None:
        # Binding count cannot select the ordinary FOUR class: it omits the
        # declared scratchpad. The cooperative serializer validates its full
        # resource signature and emits the corresponding measured layout.
        _lay = None
    # FOUR BINDINGS HAVE ONE MEASURED CLASS AND IT IS UNSWEPT, so a four-buffer contract is authored
    # through it for dispatch as well as reproduction: the handoff is explicit that this class must
    # not be swept or assembled from a donor section, and no swept four-binding variant exists to
    # fall back to. Two and three bindings keep their routing - the two-binding executed class IS
    # swept and is what nine hardware runs ran.
    if _lay is _MD.FOUR or _lay is _MD.FIVE:
        # The five-buffer class is unswept for the same reason the four-buffer one is:
        # it was measured from compiled witnesses and no swept variant exists to fall back to.
        _repro = True
    if _repro and not (_lay and _lay.get("name")):
        # Asked to reproduce an Apple object for a class with no COMPLETE layout. The only layout
        # for this binding shape is the swept executed class, whose section is a different class
        # from the one being reproduced - 340 bytes against 452 for the two-binding probe. Emitting
        # it would be a wrong build rather than a near miss, so it is refused.
        raise Missing(
            "no complete measured layout for binding indices %s, so this side cannot reproduce an "
            "object of that class. The layout available is the swept executed class, which is a "
            "different class from the one being reproduced."
            % ([b[0] for b in bindings],))
    if _repro and _lay is not None and _lay.get("name"):
        led["__GPU_METADATA"] = ("MEASURED CLASS: the layout for binding indices %s, emitted whole "
                                 "including its symbol names" % ([b[0] for b in bindings],))
        # THE MEASURED FOUR-BUFFER CLASS HAS FIXED RECORD SHAPES, and by the elision law those
        # shapes ARE the contract: field 1 present means a non-zero index, field 2 present means a
        # non-default offset, field 3 present means written. So a contract whose shapes differ is a
        # different class, and field 3 in particular is the WRITTEN FLAG rather than a property of
        # the "long" shape - emitting it by position would put the write flag on the wrong buffer.
        if _lay is _MD.FOUR:
            got = []
            for bd in bindings:
                f = {0}
                if bd[0]:
                    f.add(1)
                if bd[1]:
                    f.add(2)
                if bd[2]:
                    f.add(3)
                got.append("".join(str(x) for x in sorted(f)))
            _want = _MD.REQUIRED_SHAPES_BY_SIZE.get(_lay["size"], _MD.FOUR_REQUIRED_SHAPES)
            if tuple(got) != _want:
                raise Missing(
                    "the measured class carries record shapes %s in member order and "
                    "this contract needs %s. Those shapes are the contract under the elision law - "
                    "index present when non-zero, offset when non-default, field 3 when written - "
                    "so a different sequence is a different class and needs its own witness, the "
                    "way ln-buf4-indexed was compiled for this one."
                    % (_want, tuple(got)))
        need = sorted(_lay.get("word_vectors") or {})
        given = {int(k): list(v) for k, v in (abi.get("pk_vectors") or {}).items()}
        absent = [q for q in need if q not in given]
        if absent:
            raise Missing(
                "this class carries word vectors at per-kernel slots %s and their entries are the "
                "compiler's: in the measured witness slot 27 holds binding indices and slot 29 six "
                "further values. Supply abi['pk_vectors'] for %s. Filling them from the witness "
                "would copy the very facts that distinguish one program from another." % (need, absent))
        out[SECTIONS[0]] = bytes(_MD.build(
            [b[0] for b in bindings], layout=_lay, restore_swept=True, words=given,
            offsets=[b[1] for b in bindings],
            system_registers=abi.get("system_registers"),
            register_count=_register_count(abi)))
        led["per-kernel slot 0"] = (
            "DELIVERED by the compiler as abi['register_count']=%d, not decoded here and not "
            "inherited from the measured class's witness" % (abi["register_count"],))
        if _lay.get("slot29_vector") is not None:
            led["per-kernel slot 29"] = (
                "DERIVED from abi['system_registers']=%s: the slot-29 entry tracks the declared "
                "special register, measured over 690 objects - SR 160 gives 80 in all 34 that read "
                "it alone. A register outside the measured map is refused, not defaulted."
                % (list(abi.get("system_registers") or []),))
    else:
        out[SECTIONS[0]] = _metadata(bindings, abi, led)

    # ---- __GPU_LD_MD: the format is this side's, the slot set and values are the compiler's ---
    if abi.get('execution') is not None:
        out[SECTIONS[1]]=g17tensormetadata.load_metadata(entry)
        led['__GPU_LD_MD']='MEASURED tensor first-compilation form: T3 slots3/5; main slot1 absent (zero). Reordered fixed-library controls isolate process-order dependence.'
    else:
        out[SECTIONS[1]] = _ld_md(entry, abi, led, bindings)

    # ---- __GPU_ARCH_LD_MD: one boolean, and it is the compiler's ------------------------------
    # The class supplies its ARCH section when it measured one; otherwise the derived form.
    if abi.get('execution') is not None:
        out[SECTIONS[2]]=g17tensormetadata.arch_metadata()
        led['__GPU_ARCH_LD_MD']='MEASURED semantic-false tensor class: present empty subtable, serialized structurally'
    elif _repro and _lay is not None and _lay.get("arch32") and not abi["arch_flag"]:
        led["__GPU_ARCH_LD_MD"] = ("MEASURED CLASS: the 32-byte elided form this class carries, "
                                   "which is not the executed scalar class's zero-tailed one")
        out[SECTIONS[2]] = bytes(_lay["arch32"])
    else:
        out[SECTIONS[2]] = _arch(abi["arch_flag"], led)
    led["__GPU_ARCH_LD_MD flag"] = "ABI INPUT arch_flag"

    # ---- __GPU_STATS_MD and __GPU_REMARKS_MD --------------------------------------------------
    if "stats" in abi:
        out[SECTIONS[3]] = bytes(abi["stats"])
        led["__GPU_STATS_MD"] = "ABI INPUT stats (supplied; the loader was measured not to read it)"
    else:
        if abi.get("reproduce_measured_class"):
            out[SECTIONS[3]] = STATS_TEMPLATE
            led["__GPU_STATS_MD"] = (
                "EMITTED: the 90 constant bytes of the measured template, the two three-byte "
                "compile timings at 24 and 72 zeroed. Those record how long Apple's compiler ran - "
                "387 distinct values in 393 objects, uncorrelated with text length - so no contract "
                "carries them.")
        else:
            out[SECTIONS[3]] = bytes(96)
            # THE ASSIGNMENT USED TO SIT OUTSIDE BOTH BRANCHES and overwrote the one above it, so an
            # image holding ninety-six zeros was recorded as holding a ninety-byte template. Zeros
            # are what the validated dispatches carried; only the wording changed.
            led["__GPU_STATS_MD"] = (
                "EMITTED: 96 zero bytes. The loader was measured not to read this section - four "
                "mutations including total absence leave the kernel exact - and zeros are what the "
                "validated dispatches carried.")

    out[SECTIONS[4]] = b""
    if etypes:
        led["element types"] = (
            "ABI INPUT, CONSUMED AND MEASURED NOT ENCODED: %s - 151 corpus kernels share one "
            "byte-identical metadata blob declaring type sets from ['float'] to 19 types "
            "including half" % ", ".join(sorted(set(etypes))))
    led["__GPU_REMARKS_MD"] = "DERIVED: empty in all 21,001 corpus objects"
    # KEEP THE TWO KINDS OF CLAIM APART, in the object's own ledger and not only in prose. A
    # measured profile carries hardware evidence; this path carries none, and a reader holding a
    # field ledger should not have to know which function produced it to find that out. Silence
    # reads as absence of doubt, which is the opposite of what is true here.
    led["execution evidence"] = (
        "NOT ASSERTED BY AUTHOR. Execution evidence belongs to this delivered archive's hash "
        "and its runtime records; matching a measured metadata class alone is not a dispatch.")
    return out, led


# THE SWEPT TWO-BUFFER CLASS HAS NO APPLE WITNESS. Every object in results/ that carries two user
# buffers and NO slot 0 was authored by this side - 42 of 42 - and every Apple compile of the same
# binding shape declares a register allocation: 380 bytes with no system register, 384 with SR160,
# measured across four distinct programs at register counts 1 and 2. The 48-object agreement that
# once looked like a law was this serializer agreeing with itself.
#
# So the 500-byte class is not "the two-buffer class"; it is a swept blob that the loader happens
# to accept, and it is retained for exactly one reason - packed_scan 5ec51043 and affine e4e01969
# executed as those bytes and their receipts pin them. It therefore stays the DEFAULT, and the
# measured class is reached the way this file has always reached a class: by an explicit request
# from this side, never by inferring one from a contract that does not distinguish them.
#
# It does not distinguish them. A range-store contract and packed_scan's agree on every field the
# author can read except element type and register count, and both compile to an empty constant
# pool; routing on element type would be the correlation this file exists to refuse.
UNSWEPT_TWO_BUFFER_SIGNATURE = ((1, 0, False), (2, 2, True))
# SOURCED FROM THE CLASS, NOT COPIED. This was a fourth place stating which register sets the
# two-buffer class admits, after g17rangemetadata's gate and the two serializers that already share
# g17mdgen.slot29_entries. A copy is a thing that drifts: this one still said [[], [160]] after the
# class itself was witnessed for [160, 161], so a contract the class could serialize was refused
# here by a stale list. The class's own name is imported, so widening the class widens this.
from .rangemetadata import MEASURED_REGISTER_SETS as UNSWEPT_TWO_BUFFER_REGISTERS


def _unswept_two_buffer(bindings, abi, led):
    """Author the measured two-buffer class, only when this side asks for it by name."""
    if not abi.get("unswept_two_buffer"):
        return None
    from . import rangemetadata as g17rangemetadata
    request=abi['unswept_two_buffer']
    indexed_zero=request==g17rangemetadata.INDEXED_ZERO_CLASS
    indexed_nosr=request==g17rangemetadata.INDEXED_ZERO_NOSR_CLASS
    indexed=request==g17rangemetadata.INDEXED_ZERO_SEVEN_CLASS or indexed_zero or indexed_nosr
    if request is not True and not indexed:
        raise ValueError('unknown unswept_two_buffer class request: %r' % (request,))
    output_index=7
    if indexed_nosr:
        # THE SAME ELIDED-FIRST PAIR WITH NO DECLARED REGISTER, and the write may be at either
        # buffer. Both positions are measured from their own CPU control; anything else is refused
        # below rather than served by whichever shape is nearer.
        if (len(bindings)!=2 or type(bindings[1][0]) is not int
                or not 1<=bindings[1][0]<=30):
            raise ValueError('indexed-zero-nosr requires two bindings and output public index1..30')
        output_index=bindings[1][0]
        written=[i for i,b in enumerate(bindings) if b[2]]
        if len(written)!=1:
            raise ValueError('indexed-zero-nosr is measured for exactly one written buffer; this '
                             'contract writes %d' % len(written))
        prologue=abi.get('prologue')
        if isinstance(prologue,str):prologue=bytes.fromhex(prologue)
        if (bytes(prologue or ())!=bytes.fromhex('0e000000')+bytes.fromhex('0600')*30
                or abi.get('entry')!=64 or abi.get('promoted_ranges')
                or abi.get('spill_state') or abi.get('constant_program_sha256')):
            raise ValueError('indexed-zero-nosr requires END-only entry64 and no promotion or '
                             'spill/preload')
    if indexed_zero:
        if (len(bindings)!=2 or type(bindings[1][0]) is not int
                or not 1<=bindings[1][0]<=30):
            raise ValueError('indexed-zero requires two bindings and output public index1..30')
        output_index=bindings[1][0]
        prologue=abi.get('prologue')
        if isinstance(prologue,str):prologue=bytes.fromhex(prologue)
        if (bytes(prologue or ())!=bytes.fromhex('0e000000')+bytes.fromhex('0600')*30
                or abi.get('entry')!=64 or abi.get('promoted_ranges')
                or abi.get('spill_state') or abi.get('constant_program_sha256')):
            raise ValueError('indexed-zero requires END-only entry64 and no promotion or spill/preload')
    if indexed_nosr:
        # The written buffer may be either one, so the expected signature follows the contract's
        # own write position rather than being fixed at the output.
        _w=[i for i,b in enumerate(bindings) if b[2]][0]
        expected=(((0,0,True),(output_index,2,False)) if _w==0
                  else ((0,0,False),(output_index,2,True)))
    else:
        expected=((0,0,False),(output_index,2,True)) if indexed else UNSWEPT_TWO_BUFFER_SIGNATURE
    measured_registers=((),) if indexed_nosr else (((156,),) if indexed else UNSWEPT_TWO_BUFFER_REGISTERS)
    signature = tuple(tuple(b[:3]) for b in bindings)
    if signature != expected:
        raise ValueError(
            "unswept_two_buffer was requested for bindings %s, and this class is measured only "
            "for %s. Widening it to another binding shape would author a section from a witness "
            "that never carried it." % (list(signature), list(expected)))
    registers = tuple(abi.get("system_registers") or ())
    # NAME THE ROUTE THAT RAN, NOT THE FAMILY. `measured_registers` above is already
    # route-specific - ((),) for the no-SR class, ((156,),) for the indexed ones, the shared
    # UNSWEPT_TWO_BUFFER_REGISTERS only for the default short-first class - and this message
    # printed the shared list for all of them. So a contract refused by indexed-zero-sr156-v1,
    # which admits (156,) alone, was told "the measured sets are [[], [160], [160, 161]]":
    # three sets that route accepts none of. The refusal was right and its explanation named a
    # different gate, which is how a reader concludes a class covers sets it refuses.
    if registers not in measured_registers:
        route = repr(request) if isinstance(request, str) else "the default two-buffer route"
        raise ValueError(
            "unswept_two_buffer %s declares system_registers %s; the measured sets FOR THIS ROUTE "
            "are %s. A set with no witness has no measured slot-29 vector and would be dropped."
            % (route, list(registers), [list(r) for r in measured_registers]))
    if abi.get("uses_threadgroup"):
        raise ValueError("unswept_two_buffer carries no threadgroup slots; a contract declaring "
                         "threadgroup memory would author a section omitting slots 18 and 28")
    # Slot 13 is the program's constant pool. The measured class is the EMPTY-pool form: the same
    # kernel compiled with pooled literals moves to 428 bytes with a sixteen-word slot-13 vector.
    # A contract that states a non-empty pool is refused rather than authored into the empty class.
    pool = abi.get("constant_pool")
    if pool:
        raise ValueError(
            "unswept_two_buffer is the empty-constant-pool class (slot 13 vector length 0); this "
            "contract declares a pool of %d bytes, whose measured form is 48 bytes larger"
            % len(pool))
    count = _register_count(abi)
    led["metadata class"] = ("MEASURED unswept two-buffer class, requested explicitly: %d bytes, "
                             "reproduced byte-for-byte on Apple witnesses at register counts 1 "
                             "and 2" % (380 + 4*len(registers)))
    led["per-kernel slot 0"] = ("DELIVERED register_count=%d; the swept default class has no "
                                "slot 0 and would drop it" % count)
    led["constant pool"] = "slot 13 vector length 0 (declared empty)" if pool is not None else \
        "slot 13 vector length 0 (empty class; contract states no pool)"
    if indexed:
        if indexed_nosr:
            _w=[b[0] for b in bindings if b[2]][0]
            led['metadata class']=(
                'MEASURED indexed-zero-nosr-v1: %d bytes; elided input index0, output index%d, '
                'written buffer %d. Reproduces its own CPU control byte-for-byte; the twelve-slot '
                'generic shape is carried by no measured class and is not emitted here.'
                % (376 if _w else 380, output_index, _w))
            return bytes(g17rangemetadata.build_indexed_zero_nosr(count,output_index,_w))
        if indexed_zero:
            led['metadata class']=(
                'MEASURED indexed-zero-sr156-v1: 380 bytes; elided input index0, output index%d '
                'at pointer offset2 from the contract. Apple0/1 and0/7 witnesses differ only '
                'in that public-index field; register allocation comes from delivered code.' % output_index)
            return bytes(g17rangemetadata.build_indexed_zero(count,registers,output_index))
        led["metadata class"] = ("MEASURED indexed-0-7-sr156-v1: 380 bytes; first record "
            "elides public index zero, second names index 7 at pointer offset 2. "
            "Whole section reproduces source witness f83ea2bc4ba5cfe7 at register count 2.")
        return bytes(g17rangemetadata.build_indexed_zero_seven(count,registers))
    return bytes(g17rangemetadata.build(count, registers))


def _metadata(bindings, abi, led):
    from .rangemetadata import INDEXED_ZERO_SEVEN_CLASS,INDEXED_ZERO_CLASS
    indexed_request=abi.get('unswept_two_buffer') in (INDEXED_ZERO_SEVEN_CLASS,INDEXED_ZERO_CLASS)
    if indexed_request and abi.get('execution') is not None:
        raise ValueError('indexed two-buffer class conflicts with an execution-class request')
    marker = abi.get("requantization")
    if marker is not None:
        if marker.get("metadata_class") != "scalar_476":
            raise Missing(
                "requantization %s is compile-admitted for the measured %s-element/%s-group scalar "
                "primitive, but no measured metadata class exists for its two-dispatch image; "
                "authoring refuses until that class and common-runtime boundary are measured"
                % (marker.get("kind"), marker.get("elements"), marker.get("groups")))
        from . import requantmetadata as g17requantmetadata
        try:
            g17requantmetadata.validate_request(
                tuple(tuple(row[:3]) for row in bindings), abi.get("system_registers"))
        except ValueError as error:
            raise Missing(str(error)) from error
        led["metadata class"] = (
            "MEASURED scalar requantization class: 476-byte three-record source/scale/destination "
            "layout, SR160, structurally reserialized from the pinned reqz_round_probe witness; "
            "common two-dispatch admission remains a separate runtime gate")
        return bytes(g17requantmetadata.build(register_count=_register_count(abi)))
    _execution_requirement(abi, led)
    if abi.get('execution') is not None:
        from . import tensormetadata as g17tensormetadata
        return g17tensormetadata.emit_contract(abi,bindings,led)
    # A TEXTURE PROGRAM IS ROUTED BEFORE THE BUFFER CLASSES, so its refusal names ITS question.
    # Without this it reached the generic system-register refusal below and was told which
    # two-buffer sets are witnessed - true, irrelevant, and it points a reader at the wrong
    # question, which is exactly the confusion that cost a day on [160, 161].
    # ROUTED ON EVERY WAY THE CONTRACT CAN SAY "TEXTURE", not on one key. Keying on
    # resources.textures alone let a texture contract that lost that key reach the buffer selector,
    # where it refused with the generic system-register message - the wrong question, which is what
    # this route exists to prevent. g17teximage.declares_a_texture owns the predicate and names
    # which signal fired.
    from . import teximage as g17teximage
    if g17teximage.declares_a_texture(abi):
        if indexed_request:
            raise ValueError('indexed two-buffer class conflicts with a texture declaration')
        return g17teximage.emit(abi, bindings, led)
    # TWO EXPLICIT FACTS IN CONFLICT ARE REFUSED, NOT ORDERED. A contract that declares threadgroup
    # memory AND names the two-buffer class is asking for two different classes, and whichever
    # route is written first would silently win: on ABI v4 the cooperative author would answer a
    # request it was never asked, dropping a named class request without a word. The caller has to
    # say which one it means.
    if abi.get("uses_threadgroup") and abi.get("unswept_two_buffer"):
        raise ValueError(
            "this contract declares threadgroup memory and also names the measured two-buffer "
            "class, which carries no threadgroup slots. These select different classes; drop "
            "unswept_two_buffer to author the cooperative class, or uses_threadgroup to author "
            "the two-buffer one.")
    if abi.get("uses_threadgroup"):
        from . import cooperativemetadata as g17cooperativemetadata
        return g17cooperativemetadata.emit(abi, bindings, led)
    from . import emit as E
    from . import mdgen as M
    from . import schema as S
    from . import metaclass as g17metaclass
    _unswept = _unswept_two_buffer(bindings, abi, led)
    if _unswept is not None:
        return _unswept
    # Exact write-first XY class measured from independent CPU oracle controls.
    if (tuple(tuple(b[:3]) for b in bindings) == ((1,0,True),(2,2,False))
            and tuple(abi.get("system_registers") or ()) == (160,161)):
        from . import rangemetadata as R
        prologue = abi.get("prologue")
        if isinstance(prologue,str):
            prologue = bytes.fromhex(prologue)
        if (tuple(abi.get("system_registers") or ()) != (160,161)
                or abi.get("constant_pool") or abi.get("promoted_ranges")
                or abi.get("spill_state") or abi.get("constant_program_sha256")
                or abi.get("entry") != 64
                or bytes(prologue or ()) != bytes.fromhex("0e000000")+bytes.fromhex("0600")*30):
            raise Missing("write-first two-buffer XY class requires exactly SR160/SR161, "
                          "empty pool, END-only entry64 and no promotion or spill/preload")
        try:
            section = R.build_write_first_xy(_register_count(abi),abi.get("instruction_count"),
                                             bool(abi.get("has_back_edge")))
        except ValueError as error:
            raise Missing(str(error)) from error
        led["metadata class"] = "MEASURED write-first/read-second XY empty-pool class; CPU oracle reproduction"
        led["per-kernel slot 29"] = "MEASURED SR160/SR161 entries [80,81]"
        led["execution evidence"] = "NOT ASSERTED BY AUTHOR; CPU metadata reproduction only"
        return section

    measured = g17metaclass.select(bindings, abi)
    if measured is not None:
        # The class may carry a per-program register count; supply it where the class has the
        # slot, and refuse a missing fact rather than inherit the witness's value.
        return g17metaclass.emit(
            measured, bindings, led,
            register_count=_register_count(abi) if 0 in measured["pk_slots"] else None,
            system_registers=abi.get("system_registers"))
    # THE MEASURED CLASS ROUTE. A contract whose binding indices have a measured layout is
    # authored through it rather than falling to the general serializer, which emits no slot-29
    # vector and would drop a declared register. Every precondition is checked and refused BY NAME:
    # the class has to exist, the compiler has to state the argument block it staged, nothing may
    # be staged beyond the pointers (slot 1 is the pointer block only while that holds), and the
    # bound indices have to be contiguous, because this compiler emits 2 * rank where the class was
    # measured on Apple's 2 * (index - lowest) and the two agree only there.
    # THE INSTRUCTION COUNT IS PART OF THE SELECTION for the two-writable classes: past the
    # slot-32 boundary the short class has nowhere to put the slot, and the count is a fact the
    # contract states rather than one this side predicts from a workload parameter.
    _measured = M.layout_for([b[0] for b in bindings],
                             promoted_ranges=abi.get("promoted_ranges"),
                             written=[bool(b[2]) for b in bindings],
                             instructions=abi.get("instruction_count"),
                             back_edge=bool(abi.get("has_back_edge")),
                             system_registers=abi.get("system_registers"))
    # SELECTED WITHOUT THE WRITE MASK, on purpose. layout_for now refuses a three-binding contract
    # whose written buffer is not the last one, so asking it with the mask returns None here and
    # this branch's own far more specific refusal - which names the dense read/read/write shape the
    # class is measured for - became unreachable, leaving a contract to fail later on its register
    # set instead. The branch re-checks the mask itself two lines below, so nothing is admitted by
    # asking without it.
    _four_separate = (M.layout_for([b[0] for b in bindings],
                                   promoted_ranges=abi.get("promoted_ranges"),
                                   instructions=abi.get("instruction_count"),
                                   back_edge=bool(abi.get("has_back_edge"))) is M.SEPARATE
                      and len(abi.get("system_registers") or ()) == 4)
    if _four_separate:
        # This extension composes the measured separate read/read/write resource
        # records with a four-entry system-value vector. It does not transplant
        # the promoted constant program of the source witness.
        _prologue = abi.get("prologue")
        if isinstance(_prologue, str):
            _prologue = bytes.fromhex(_prologue)
        if (tuple(_prologue or ()) != tuple(bytes.fromhex("0e000000") + bytes.fromhex("0600") * 30)
                or abi.get("entry") != 64 or abi.get("promoted_ranges")
                or abi.get("spill_state") or abi.get("constant_program_sha256")
                or [bool(b[2]) for b in bindings] != [False, False, True]
                or [b[1] for b in bindings] != [0, 2, 4]):
            raise Missing("four-register separate class requires END-only entry64, no promotion or "
                          "spill/preload, and dense read/read/write bindings")
        _measured = M.SEPARATE_FOUR_REGISTERS
    # THE ONE-BINDING CLASS IS WITNESSED AT ONE SYSTEM-REGISTER SET. `slot29_entries` maps other
    # sets happily - (160,) becomes [80] - so without this the class would author a section for an
    # SR set no control ever compiled, at a size that looks entirely plausible. Its three controls
    # are all SR156/SR157, which is also the delivered contract's own declaration.
    if _measured is M.ONE_WRITTEN_ZERO:
        _registers = tuple(abi.get("system_registers") or ())
        if _registers != M.ONE_WRITTEN_ZERO_REGISTERS and _general_can_carry(_registers, abi):
            led["metadata class"] = ("GENERAL SERIALIZER: the one-binding class is witnessed at %s only; "
                                     "this contract declares %s, which g17mdgen.slot29_entries maps"
                                     % (list(M.ONE_WRITTEN_ZERO_REGISTERS), list(_registers)))
            _measured = None
        elif _registers != M.ONE_WRITTEN_ZERO_REGISTERS:
            raise Missing(
                "the measured one-binding written-buffer-0 class is witnessed at system registers "
                "%s, and this contract declares %s. The slot-29 vector moves every structure after "
                "it, so a set admitted by analogy would yield a section of an entirely plausible "
                "size that decodes as if nothing were wrong."
                % (list(M.ONE_WRITTEN_ZERO_REGISTERS), list(_registers) or "none"))
        if _measured is not None and abi.get("constant_pool"):
            raise Missing(
                "the measured one-binding class emits the 8-byte zero vector its controls carry at "
                "slot 13; this contract declares a constant pool, which no control witnesses here")
    # FAIL CLOSED ON AN UNWITNESSED FOUR-BINDING WRITE MASK. layout_for returns None for a
    # buffer-zero four-binding contract whose written buffer is neither the last nor the middle,
    # and returning None here used to mean "fall through to the generic serializer", which authored
    # a section anyway. A refusal that only removes a measured class is not a refusal at all.
    _four_zero = (len(bindings) == 4 and 0 in [b[0] for b in bindings])
    if _four_zero and _measured is None:
        raise Missing(
            "four bindings including buffer 0 with write mask %s: the measured classes place the "
            "written record by POSITION - only the long shape carries slot 3 - and the two "
            "witnessed masks are write-last and write-index-2. Authoring this contract would name "
            "a different buffer as written than the one it writes."
            % ([bool(b[2]) for b in bindings],))
    # AND THE DECLARED REGISTER SET MUST FIT THE CLASS. Routing on the binding indices alone let a
    # contract declaring an unmeasured set - [48], or [48, 160] - reach M.build, which places what
    # it can and leaves the rest out: the exact silent drop the refusal below exists to prevent,
    # reintroduced one branch above it. `slot29_entries` is what decides, and it raises.
    if _measured is not None and abi.get("system_registers"):
        # A MAPPED VALUE IS NOT A PLACE TO PUT IT. `slot29_entries` answers whether each declared
        # register HAS a measured entry; it says nothing about whether THIS class carries a
        # slot-29 slot to hold them. SCALAR carries none - 29 is not in its per-kernel slot map -
        # so g17mdgen.build ignores `system_registers` for it completely: SR [], [156, 164],
        # [160, 161] and the unmeasured [48] all emit the same 500 bytes. The gate below saw
        # [156, 164] map to [0, 48], raised nothing, and kept the class. What actually stopped
        # the two delivered two-binding contracts was the unrelated absence of slot 0, so the
        # protection was accidental and a contract satisfying slot 0 would have been authored
        # with its declaration dropped. Refuse by the declaration, before any build.
        if 29 not in _measured["pk_slots"] and _general_can_carry(abi["system_registers"], abi):
            led["metadata class"] = ("GENERAL SERIALIZER: the measured layout for bindings %s has no "
                                     "slot 29; the general serializer derives it"
                                     % [b[0] for b in bindings])
            _measured = None
        elif 29 not in _measured["pk_slots"]:
            raise Missing(
                "this contract declares system_registers %s, and the measured layout selected for "
                "binding indices %s carries no per-kernel slot 29 to hold them. The section it "
                "would author is byte-identical to one declaring no system registers at all, so "
                "authoring here would silently drop the declaration. A register having a measured "
                "slot-29 entry is not the same fact as this class having somewhere to put it."
                % (list(abi["system_registers"]), [b[0] for b in bindings]))
        # AND THE REFUSAL SAYS WHICH FACT, not "no class matches". Dropping the class here and
        # letting the generic refusal below speak told a write-0 contract declaring two registers
        # that no measured class matched its bindings, when one matches its bindings exactly and
        # is witnessed at one entry. The gate that decided is `slot29_entries`, and its reason -
        # a count no witness carries, or a register outside the map - is the one to print.
        try:
            if _measured is None:
                pass
            elif _measured.get("slot29_sets") is not None:
                M.slot29_entries(tuple(abi["system_registers"]), _measured.get("slot29_counts"),
                                 _measured["slot29_sets"])
            else:
                M.slot29_entries(tuple(abi["system_registers"]), _measured.get("slot29_counts"))
        except (ValueError, KeyError) as reason:
            if _general_can_carry(abi["system_registers"], abi):
                led["metadata class"] = ("GENERAL SERIALIZER: the measured class for bindings %s "
                                         "cannot carry %s (%s); the general serializer derives slot 29"
                                         % ([b[0] for b in bindings], list(abi["system_registers"]),
                                            str(reason).split("\n")[0][:120]))
                _measured = None
            else:
              raise Missing(
                "this contract declares system_registers %s, and the measured class selected for "
                "binding indices %s cannot carry that set: %s The general serializer emits no "
                "slot-29 vector, so authoring here would silently drop the declaration."
                % (list(abi["system_registers"]), [b[0] for b in bindings], reason))
    # THE MEASURED ROUTE IS NOT CONDITIONAL ON DECLARING A SYSTEM REGISTER. This gate read
    # `abi.get("system_registers")`, so the buffer-zero four-binding contracts - which declare
    # SR[] - computed a measured layout and then fell through to the generic serializer, authoring
    # 400 and 408 bytes where their measured classes are 428 and 440. The refusal above and the
    # class selection both existed and neither reached the public author.
    if _measured is not None and (abi.get("system_registers")
                                  or _measured in M.FOUR_WITH_ZERO_CLASSES):
        _state = abi.get("argument_state")
        if _state is None:
            raise Missing(
                "a measured metadata layout exists for the binding indices %s, and this contract "
                "states no argument_state. Per-kernel slot 1 is the argument block the compiler "
                "staged; without it the class would emit the witness's value, which is right for "
                "one Apple arm of seven. Supply ProgramABI.argument_state."
                % ([b[0] for b in bindings],))
        if "per_kernel_slot_1" not in tuple(_state.get("not_stated") or ()):
            raise Missing(
                "this contract's argument_state does not list per_kernel_slot_1 as unstated, so "
                "it is claiming slot 1 directly; the derivation here is no longer what supplies "
                "it and this route must not silently outrank the compiler.")
        # Public indices need not be adjacent in the separate three-buffer
        # class. Apple's retained syn-s7f595f1cd1 names 26/28/29 but stores
        # pointer offsets 0/2/4, exactly the compiler's dense rank block. This
        # is a counterexample to the index-minus-lowest restriction below.
        # Keep other layouts outside this extension, and require the measured
        # read/read/write record shapes and independently stated dense offsets.
        _separate_dense = (_four_separate
                           and [bool(b[2]) for b in bindings] == [False, False, True]
                           and [b[1] for b in bindings] == [0, 2, 4]
                           and all(b[0] > 0 for b in bindings))
        if not _state.get("indices_contiguous") and not _separate_dense:
            raise Missing(
                "this contract's bound indices are not contiguous, and the compiler emits pointer "
                "offsets at 2 * rank while this class was measured on Apple's "
                "2 * (index - lowest bound index). The two agree only on a contiguous set, so "
                "authoring here would place pointers the emitted code does not read.")
        _offsets = [offset for _index, offset in _state.get("pointer_offsets") or ()]
        if _offsets != [b[1] for b in bindings]:
            raise Missing(
                "the contract's argument_state offsets %s and its binding offsets %s disagree, so "
                "which one the emitted code reads is not decided here"
                % (_offsets, [b[1] for b in bindings]))
        led["__GPU_METADATA"] = (
            "MEASURED CLASS: g17mdgen.build on the layout measured for binding indices %s, with "
            "per-kernel slot 1 DERIVED from the compiler's stated argument block (%d words) "
            "rather than inherited from the class's witness"
            % ([b[0] for b in bindings], _state["block_words"]))
        if _four_separate:
            led["__GPU_METADATA"] += (
                "; STRUCTURAL COMPOSITION: separate unpromoted class plus the witnessed "
                "four-entry system-value vector, with all later nodes relocated by 12 bytes; "
                "no execution result is implied by this authoring derivation")
        _layout = dict(_measured)
        _extra = dict(_layout["pk_extra"])
        # SLOT 1 IS THE END OF THE ARGUMENT BLOCK, which is the furthest extent any slot-2 record
        # reaches: max over the emitted records of field 3 + field 2. That relation holds on 8,965
        # of 8,965 corpus sections carrying a kind-6 record and is unreliable without one, so it is
        # applied only where the class emits one. Setting slot 1 to the compiler's `block_words`
        # was right for the six-buffer class - which emits no kind-6, making the max the kind-3
        # record's own extent and the two answers the same number - and wrong here: the authored
        # section said 4 while its own records reached 8, which no witness does.
        # SLOT 32 IS PER-PROGRAM, NOT THE CLASS'S. The measured class carries its witness's value,
        # which is right for that witness and wrong for every other program in the band - the same
        # defect slot 0 had before `register_count` became a parameter. n200 is 997 instructions
        # and was emitting the witness's 1. It is derived from the count the contract states.
        if 32 in _layout["pk_slots"]:
            _band = M.slot32_for(abi.get("instruction_count"), bool(abi.get("has_back_edge")))
            if _band is None:
                raise Missing(
                    "this class carries a per-kernel slot 32 and the contract's instruction count "
                    "%s puts the program below the boundary, so the value would be the witness's "
                    "rather than the program's" % (abi.get("instruction_count"),))
            _extra[32] = ("<B", _band)
        # THE SLOT-2 VALUES MUST FILL THE CLASS'S OWN RECORD LIST. This passed a single kind-3
        # record, which is right for the six-buffer class and leaves a two-record class's second
        # record unwritten - the vector then does not walk and the finished image fails
        # verification with "slot 2 refers to N, where nothing walks as a vector of tables".
        # Defaults come from the measured class; the kind-3 record's extent is the one value the
        # contract decides.
        _second = [dict(row) if isinstance(row, dict) else row for row in _layout["v2_vals"]]
        for _i, _row in enumerate(_second):
            _kind = _row[0] if isinstance(_row, tuple) else _row.get(0)
            if _kind == 3:
                _second[_i] = ((3, _state["block_words"], _row[2]) if isinstance(_row, tuple)
                               else dict(_row, **{2: _state["block_words"]}))
        # NOW slot 1, from the records above rather than from a separate input.
        def _extent(row):
            if isinstance(row, dict):
                return int(row.get(3, 0) or 0) + int(row.get(2, 0) or 0)
            return int(row[2] or 0) + int(row[1] or 0)
        _extra[1] = ("<I", max(_extent(row) for row in _second))
        _layout["pk_extra"] = _extra
        return bytes(M.build([b[0] for b in bindings], layout=_layout, restore_swept=True,
                             offsets=[b[1] for b in bindings],
                             second=_second,
                             system_registers=abi.get("system_registers"),
                             register_count=_register_count(abi)))
    # THE GENERAL SERIALIZER'S THREE WORD VECTORS, DERIVED WHEN THE COMPILER DID NOT STATE THEM.
    # Without slot 13 Apple's driver dies inside newComputePipelineState (7 of 7 real sources,
    # MM 25.149), and scanlink.require_driver_slots now refuses such a section. What goes in them:
    #   13  the program's own constant pool, byte for byte - slot 13 is the kernel's constant data,
    #       present in 1,990 of 1,990 sampled Apple sections
    #   29  g17mdgen.slot29_entries(system_registers): the law that predicts Apple's own slot-29
    #       vector for 37 of the 43 fixed197 sources refused for their register set (the other 6
    #       name a register the law has no entry for, and still refuse)
    #   27  EMPTY, and the ledger says it is assumed: Apple's slot 27 is empty in 28 of the 34
    #       fixed197 sources that reach this path, and the rule for the other six is not known
    # A caller that states pk_vectors keeps them exactly; nothing here overrides a stated value.
    if "pk_vectors" not in abi:
        _vectors = {13: [int(b) for b in (abi.get("constant_pool") or ())], 27: []}
        led["slot 13 words"] = ("DERIVED: the program's constant pool, %d bytes"
                                % len(_vectors[13]))
        led["slot 27 words"] = ("ASSUMED EMPTY: Apple's is empty in 28 of 34 fixed197 sources on "
                                "this path; the rule for the other six is not known")
        _registers = tuple(abi.get("system_registers") or ())
        try:
            _vectors[29] = list(M.slot29_entries(_registers)) if _registers else []
            led["slot 29 words"] = ("DERIVED: g17mdgen.slot29_entries(%s), the law that predicts "
                                    "Apple's vector in 37 of 43 fixed197 sources" % list(_registers))
        except Exception as _e:
            led["slot 29 words"] = "NOT DERIVED: %s" % str(_e).split("\n")[0][:160]
        abi = dict(abi, pk_vectors=_vectors)
    # A DECLARED SYSTEM REGISTER MUST NOT BE DROPPED BY FALLING THROUGH. When no measured class
    # matches, the general serializer below emits a section that carries no slot-29 vector at all,
    # so a contract declaring (160, 162) or three registers would author successfully and produce
    # an image that does not state what the program reads. That is a wrong build, not a lenient
    # one: the refusal exists precisely so an unmeasured register set stops here.
    if abi.get("system_registers") and 29 not in (abi.get("pk_vectors") or {}):
        # TWO DIFFERENT FACTS, AND THIS MESSAGE USED TO CONFLATE THEM. The map says which
        # REGISTERS have a measured slot-29 entry; a class says which SETS it is witnessed for.
        # The old text stated the second from a copy - "(160,) on the two-binding class" - which
        # was still there after the unswept two-buffer class had been witnessed for (160, 161),
        # and pointed a reader at the wrong question. Both halves are now read from their source.
        raise Missing(
            "this contract declares system_registers %s and no measured class matches its "
            "bindings, entry and prologue. The general serializer emits no slot-29 vector, so "
            "authoring here would silently drop the declaration. Registers with a measured entry: "
            "%s. The swept two-binding class is witnessed only for (160,); the unswept two-buffer "
            "class is witnessed for %s and must be named by unswept_two_buffer rather than "
            "selected; the three-binding class takes any mapped set. A set outside these needs "
            "its own witness."
            % (list(abi["system_registers"]), sorted(M.SLOT29_BY_SYSTEM_REGISTER),
               [list(r) for r in UNSWEPT_TWO_BUFFER_REGISTERS if r]))
    order = sorted(bindings, key=lambda b: b[1])
    # THE POINTER BLOCK IS max(offset) + 2, NOT 2 * len(bindings). The two agree only when every
    # rank is occupied, which is 7,108 of 8,415 Apple sections - 84.47%. Where a rank is SKIPPED
    # they diverge: asvAtomicMetalIntersector_IFB_instancing_maxLevels binds 28 resources across
    # THIRTY ranks, leaving 30 and 38 empty, and its kind-3 record says 60 rather than 56.
    #
    #     block == max(binding offset) + 2      8,415 of 8,415, zero exceptions
    #     block == 2 * len(bindings)            7,108 of 8,415
    #
    # The offsets are an ABI input this side already receives, so this is a derivation, and it is
    # the general form of a rule that was right only where nothing was skipped.
    block = (max(b[1] for b in order) + 2) if order else 0
    led["binding offsets"] = "DERIVED: 2*rank, the pointer-block law, 74,325 records"
    led["binding vector order"] = "ABI INPUT bindings (the order is load-bearing: a store ranks into it)"

    recs = []
    for idx, off, wr in bindings:
        vals = {0: 5}
        if idx:
            vals[1] = idx
        if off:
            vals[2] = off
        if wr:
            vals[3] = 1
        omap, tlen = E.record_layout(4, 5, tuple(sorted(vals)))
        W = E.RECW[4]
        recs.append(S.Table(dict(omap),
                            {k: ("<B" if W.get(k, 4) == 1 else "<I", vals[k]) for k in omap},
                            tlen, tlen, name="bind%d" % idx))
    led["record layouts"] = "DERIVED: the emission order, 199,974 of 200,230 record tables, 0 wrong"

    k3map, k3len = E.record_layout(2, 3, (0, 2))
    k3 = S.Table(dict(k3map), {0: ("<B", 3), 2: ("<I", block)}, k3len, k3len, name="k3")

    # THE SLOT-2 VECTOR OFTEN CARRIES A CONSTANT KIND-6 RECORD AHEAD OF THE KIND-3 ONE, and its
    # presence is a program fact this side cannot derive. It is in 4,782 of 6,001 cached sections
    # and its fields are the SAME in every one of them - {0: 11, 2: 12, 3: 4} - so the record costs
    # nothing to write once you know whether to write it. Nothing tested here predicts that:
    # per-kernel slots 13, 27, 29 are present either way, `const` is true in all 12,033 corpus
    # kernels and so cannot discriminate at all, and the best structural correlate found - having
    # two or more bindings - separates the two groups by 35 points. That is a compiler fact, and
    # the twelve bytes it occupies are the largest single residual in the authoring census.
    #
    # IT IS OPTIONAL RATHER THAN REQUIRED, and the ledger says which way it went either way. A
    # default that changes bytes silently is the failure this file exists to prevent; a default
    # that is PRINTED beside the object is a stated assumption, and the caller can see it is
    # standing in for something the backend knows.
    # THE SLOT-2 VECTOR'S KIND-6 RECORD, and it is ONE number now rather than two.
    #
    # Its shape was already known - offsets {0: 11, 2: 12, 3: 4} at declared length 16, 2,040
    # witnesses - and field 0 is the kind. Its two remaining fields resisted every formula scored
    # against them (f3 == block at 59.8%, f2 + f3 == 2*block at 73.9%) until the raw triples were
    # read instead of scored, which gave an invariant:
    #
    #     f2 + f3 is a multiple of 4        15,777 of 15,777, and f2, f3 always share parity
    #     f2 + f3 == PER-KERNEL SLOT 1      15,777 of 15,777, ZERO exceptions
    #
    # Slot 1 is already a required ABI input, so the SUM is determined and only the SPLIT is not.
    # The caller supplies f2 and f3 follows. Nothing here guesses it: the best split rule found is
    # f3 == block at 71.7%, and a rule right seven times in ten is the defect this file exists to
    # refuse.
    k6 = None
    # THE SWEPT CLASS BREAKS THE SUM LAW, and the break is in the delivered bytes rather than in
    # the law. f2 + f3 == per-kernel slot 1 holds in 417 of 417 sampled sections that carry a
    # kind-6 record, and slot 1 is four bytes wide in 592 of 592. The executed scalar class carries
    # f2=20, f3=4 with slot 1 reading 00000000: its slot 1 was ZEROED in delivery, so the sum has
    # nothing to be equal to. Deriving f3 from a swept slot 1 would author 0 where the delivered
    # bytes carry 4. So a class whose slot 1 is swept states the pair, and says which it is.
    pair = abi.get("slot2_kind6")
    if pair is not None:
        f2, f3 = pair
        m6, l6 = E.record_layout(2, 6, (0, 2, 3))
        k6 = S.Table(dict(m6), {0: ("<B", 6), 2: ("<I", f2), 3: ("<I", f3)}, l6, l6, name="k6")
        led["slot-2 kind-6 record"] = (
            "ABI INPUT slot2_kind6=(%d, %d) STATED, not derived: this class's per-kernel slot 1 is "
            "swept to zero, so f2 + f3 == slot 1 has nothing to hold against" % (f2, f3))
    split = abi.get("slot2_kind6_f2")
    if pair is None and split is not None:
        if not 0 <= split <= abi["pk_slot1"]:
            raise ValueError("slot2_kind6_f2 is %d and the two fields sum to per-kernel slot 1 "
                             "(%d), so it cannot lie outside [0, %d]"
                             % (split, abi["pk_slot1"], abi["pk_slot1"]))
        other = abi["pk_slot1"] - split
        if split % 2 != other % 2:
            raise ValueError("the kind-6 record's two fields share parity in 15,777 of 15,777 "
                             "sections; %d and %d do not" % (split, other))
        m6, l6 = E.record_layout(2, 6, (0, 2, 3))
        k6 = S.Table(dict(m6), {0: ("<B", 6), 2: ("<I", split), 3: ("<I", other)}, l6, l6,
                     name="k6")
        led["slot-2 kind-6 record"] = ("ABI INPUT slot2_kind6_f2=%d, and f3=%d DERIVED as "
                                       "pk_slot1 - f2 (15,777 of 15,777)" % (split, other))
    elif pair is None:
        led["slot-2 kind-6 record"] = (
            "ASSUMED ABSENT - 4,782 of 6,001 sections carry it; supply abi['slot2_kind6_f2'] and "
            "f3 follows from pk_slot1")

    # FIELD 0 IS A STRING REFERENCE, not the integer 8 an earlier version wrote here. The 8 was
    # the relative offset Apple's own layout happened to put there, copied out of one section and
    # emitted as a number. The name is g17mdgen.SYMBOLS[1], the same symbol this project's own
    # object declares, so it is written from the program rather than carried as somebody's bytes.
    cpname = S.String(M.SYMBOLS[1], name="constprogname")
    led["constant-program name"] = ("DERIVED: the string %r, this object's own symbol"
                                    % M.SYMBOLS[1])
    # FIELD 3 IS A REFERENCE TO A WORD VECTOR, not the literal 16. The 16 is Apple's relative
    # offset in one layout, copied in as a value - the same offset-map-as-value-map mistake that
    # field 0 had before the String node and that slot 6 had before the pointer guard, for the
    # third time. ad-L_const1 shows the target: the v0 record's body ends at 240 and the vector
    # [0, 1, 2, 3] sits there, length-prefixed, with the constant-program string after it at 260.
    #
    # AND THE OLD SIZE HID IT. This declared the record 24 bytes where its tlen is 20, and 20 + 4
    # is exactly the empty vector's width - so for the 15,962 of 21,110 sections (75.6%) whose
    # vector is EMPTY the bytes came out right by accident. The 118 that differ are the non-empty
    # ones, which carry a consecutive run: [0,1,2,3] most often, then [4,5,6,7] and [8,9,10,11],
    # always starting at a multiple of four.
    v0words = S.Words(abi.get("v0_field3") or (), name="v0words", width=4)
    led["v0 field 3"] = ("DERIVED as a reference; its %d words are ABI INPUT v0_field3 (empty in "
                         "75.6%% of 21,110 objects)" % len(abi.get("v0_field3") or ()))
    # FIELD 2 IS NOT THE CONSTANT 1. It is 1 in most sections and takes values from 2 to 95 in the
    # rest, and the obvious rule - 2 when the field-3 vector is non-empty, else 1 - reaches 85.03%
    # over 21,110 objects. A rule right five times in six is what this file refuses, so it is a
    # named input defaulting to the majority value with the default RECORDED in the ledger.
    _v0f2 = abi.get("v0_field2", 1)
    led["v0 field 2"] = (("ABI INPUT v0_field2=%d" % _v0f2) if "v0_field2" in abi else
                         "ASSUMED 1 - the majority; it runs to 95 and no rule beats 85.03%")
    v0 = S.Table(M.V0_SHAPE[0], {3: ("<I", S.Ref(v0words)), 2: ("<I", _v0f2), 1: ("<B", 3),
                                 0: ("<I", S.Ref(cpname))},
                 M.V0_SHAPE[1], M.V0_SHAPE[1], name="v0rec")
    led["constant-program name"] = ("DERIVED: the string %r, this object's own symbol"
                                    % M.SYMBOLS[1])
    bindvec = S.Vector([S.Ref(r) for r in recs], name="bindings")
    # THE SLOT-2 VECTOR CAN CARRY MORE THAN [6, 3]. This project's own composition law says the
    # order - kind 9, then 6, then 3, then kind-less, then kind 5 - over 30,406 vectors with zero
    # violations, and this side emitted only the middle two. The trailing records' VALUES are the
    # compiler's, so they are a named input: a list of (kind, {slot: value}) appended in that order.
    extra2 = list(abi.get("slot2_extra") or ())
    xrecs = []
    for kind, vals in extra2:
        m, tl = E.record_layout(2, kind, tuple(sorted(vals)))
        W2 = E.RECW.get(2, {})
        xrecs.append(S.Table(dict(m),
                             {k: ("<B" if W2.get(k, 4) == 1 else "<I", v) for k, v in vals.items()},
                             tl, tl, name="x2_%s_%d" % (kind, len(xrecs))))
    if extra2:
        led["slot-2 extra records"] = ("ABI INPUT slot2_extra: %s in the composition order "
                                       "9, 6, 3, none, 5" % [k for k, _v in extra2])
    members = ([] if k6 is None else [S.Ref(k6)]) + [S.Ref(k3)] + [S.Ref(x) for x in xrecs]
    lastvec = S.Vector(members, name="last")
    # THE CONSTANT-PROGRAM RECORD'S PRESENCE IS AN INPUT, not a fact of the format. Slot 26 holds
    # exactly ONE record in 4,001 of 4,001 cached objects - a completely uniform population, which
    # is how an assumption stays invisible - and the executed scalar class holds NONE. Emitting one
    # unconditionally is one of the three differences the v1 image had from the measured scalar.
    #
    # Default is the cached majority and the ledger SAYS SO on every object, so a caller who did
    # not think about it can still see which way it went.
    # THE PROLOGUE DOES NOT ANSWER PRESENCE. Retained three-binding objects with
    # END-only constant programs still contain this record. Its absence in the
    # executed two-binding class is a sweep property. Measured classes decide it
    # above; the general path reports an assumption unless the caller states it.
    _cp_present, _cp_why = _constant_program(abi, led)
    led["constant-program record"] = _cp_why
    if not _cp_present:
        # These nodes are excluded from the emitted document below. Their local
        # defaults cannot be assumptions about delivered fields that do not exist.
        for key in ("v0 field 2", "v0 field 3", "constant-program name"):
            led[key] = "NOT EMITTED: the constant-program record is absent"
    firstvec = S.Vector([S.Ref(v0)] if _cp_present else [], name="first")

    # SLOTS 13, 27 AND 29 ARE VECTORS OF RAW WORDS and their CONTENT is the compiler's - slot 13
    # carries fill-h-100's float constants. The structure is derived here; the words are a named
    # ABI input, and a slot absent from `pk_vectors` is simply not emitted.
    wvecs = {}
    for q, ws in sorted((abi.get("pk_vectors") or {}).items()):
        if q not in (13, 27, 29):
            raise ValueError("slot %d is not one of the word vectors 13, 27, 29" % q)
        wvecs[q] = S.Words(ws, name="w%d" % q, width=(1 if q == 13 else 4))
        led["slot %d words" % q] = "ABI INPUT pk_vectors[%d] (%d words)" % (q, len(ws))

    # SLOTS 6, 8, 10 AND 12 ARE EMPTY VECTORS AND THEY ARE PRESENT ANYWAY. Measured over 2,828
    # cached sections: slot 6 length 0 in 2,828 of 2,828, slot 8 in 2,828 of 2,828, slot 12 in
    # 2,828 of 2,828, slot 10 in 2,823 with 5 sections carrying one entry. They are NOT deduped -
    # a4-cmpx-uni places them at 276, 280, 284 and 288, four distinct four-byte vectors - so this
    # emits four, not one shared one.
    #
    # WHY THEY WERE MISSING. The elision law says a slot is present exactly when its value is not
    # the default, and an empty vector reads like nothing; the author emitted no field and Apple
    # emits a POINTER TO AN EMPTY VECTOR, which is not the default. A pointer to nothing and no
    # pointer at all are different bytes, and every section in the cache takes the first.
    # WHICH EMPTY VECTORS, is a CLASS fact and not a program fact. 6, 8, 10 and 12 is what 2,828
    # cached sections carry; the executed scalar class carries 13 where those carry 12, and no
    # property of the program distinguishes them. So it is named data selected by the contract,
    # not a literal here and not a branch that returns frozen bytes somewhere else.
    empty = tuple(abi.get("pk_empty_vectors") or PK_EMPTY_VECTORS)
    # Typed compiler deliveries carry their own allocation. The generic path must
    # serialize it too; legacy raw authoring inputs without this field retain their
    # existing slot selection. Specialized measured paths return above this point.
    register_count = _register_count(abi) if "register_count" in abi else None
    if register_count is not None:
        if 0 in empty:
            raise ValueError("per-kernel slot 0 empty vector conflicts with register_count")
        override = (abi.get("pk_values") or {}).get(0, register_count)
        if type(override) is not int or override != register_count:
            raise ValueError("per-kernel slot 0 override conflicts with register_count")
    slots = sorted({1, 2, 3, 4, 26} | set(empty) | set(wvecs)
                   | set(abi.get("pk_extra", ())) | ({0} if register_count is not None else set()))
    st = E._store()
    # THE PER-KERNEL TABLE'S DECLARED LENGTH IS NOT 60. It was hardcoded here, and 60 is the value
    # for FIFTEEN slots - which is what the compiled corpus carries and 4.6% of Apple's sections
    # do. Measured over 8,684 Apple sections the length tracks the slot count: 11 slots declare 48,
    # 15 declare 60, 19 declare 72, 24 declare 88, 26 declare 96.
    #
    # The emission-order model already computes it - natural_tlen walks the same precedence the
    # offsets come from - and hardcoding 60 instead is what made the walk run off the front of the
    # table for the acceleration-structure sections, which carry 26 slots. Their symptom was
    # "perkernel slot 26 is at offset -4": a NEGATIVE offset, which the vtable then packed as an
    # unsigned 16-bit field and reported as "'H' format requires 0 <= number <= 65535" - an opaque
    # crash standing in for a layout that was simply too small.
    _pkorder = E.order_for(st["pk_prec"], set(slots))
    pk_tlen = E.natural_tlen(_pkorder, set(slots), st["pk_wid"])
    led["per-kernel declared length"] = ("DERIVED: the emission order's natural extent for %d "
                                         "slots, not a constant 60" % len(slots))
    lay = E.walk(_pkorder, set(slots), st["pk_wid"], pk_tlen)
    led["per-kernel offsets"] = "DERIVED: the emission order at its natural declared length"
    qvecs = {q: S.Vector([], name="q%d" % q) for q in empty}
    led["empty vectors %s" % (empty,)] = ("DERIVED: EMPTY vectors, present not elided - length 0 in "
                              "2,828 of 2,828 cached sections")
    # SLOT 1 IS AN ABI INPUT, AND IT USED TO BE A DERIVATION HERE THAT WAS WRONG.
    #
    # The claim was "slot 1 = align_up(pointer block, 8)", witnessed on tu-use1, ty-2d and
    # mr-tex.uint_read-1 - three kernels, all of them TEXTURE kernels. Over 21,020 cached sections
    # it holds in 5,887, which is 28%. Nothing about the binding structure decides it: sections
    # with ONE binding at offset 0 - the same key, 10,000+ of them - carry slot 1 = 4, 8, 12 and
    # 16. The best structural predictor found, align_up(max(offset)+2, 4), reaches 74.5% and
    # leaves 5,355 exceptions, so it is not a law either.
    #
    # A derivation that is right 28% of the time is worse than a refusal, because it produces a
    # section with no complaint. So this is named and demanded.
    # AND PER-KERNEL SLOT 3 IS FOUR TIMES THE BLOCK, not eight times the binding count - the same
    # correction, measured the same way: slot 3 == 4 * the kind-3 record's block field in 8,415 of
    # 8,415 sections with zero exceptions, where 8 * len(bindings) holds in 7,108. The AS-family
    # instance above carries slot 3 = 240 = 4 * 60, while 8 * 28 would be 224.
    fields = {3: ("<I", 4 * block), 4: ("<I", S.Ref(bindvec)),
              2: ("<I", S.Ref(lastvec)), 26: ("<I", S.Ref(firstvec)),
              1: ("<I", abi["pk_slot1"])}
    if register_count is not None:
        fields[0] = ("<I", register_count)
        led["per-kernel slot 0"] = "DELIVERED compiler register_count=%d" % register_count
    fields.update({q: ("<I", S.Ref(v)) for q, v in qvecs.items()})
    fields.update({q: ("<I", S.Ref(v)) for q, v in wvecs.items()})
    led["slot 1"] = ("ABI INPUT pk_slot1 - NOT derivable: one binding at offset 0 takes "
                     "4, 8, 12 and 16 over 21,020 sections")
    for s, v in (abi.get("pk_values") or {}).items():
        if s == 0 and register_count is not None:
            continue                         # checked above; retain compiler attribution
        if s in POINTER_SLOTS:
            raise ValueError(
                "slot %d is a POINTER in 6,000 of 6,000 sections, not a scalar; a value here "
                "would be emitted as an integer where the loader walks an offset. This side "
                "does not author slot %d's target - that is a gap with a size, and it is "
                "refused rather than filled with a number." % (s, s))
        if s in DERIVED_SLOTS:
            raise ValueError("slot %d is derived here, not an ABI input; supplying it would "
                             "silently override a law" % s)
        if s not in lay:
            raise KeyError("slot %d is not in the per-kernel slot map" % s)
        fields[s] = ("<B" if s in (15, 16, 17, 18, 19, 30, 32, 33, 40, 43) else "<I", v)
        led["per-kernel slot %d" % s] = "ABI INPUT pk_values[%d]" % s
    pk = S.Table(lay, fields, pk_tlen, pk_tlen, name="perkernel")
    # ROOT SLOT 3 IS THE ENTRY POINT'S NAME, not the integer zero this used to write. Apple hangs
    # a two-slot table off it whose slot 1 references the string `agc.main` - a4-cmpx-uni puts the
    # table at 36 and the string at 44. It is this object's OWN symbol, the one g17obj writes into
    # the symbol table as _agc.main, so it is derived from the program rather than copied.
    mainname = S.String(M.SYMBOLS[0], name="mainname")
    # AND IT GAINS A SECOND FIELD WHEN THE KERNEL HAS A THREADGROUP ALLOCATION. The name table
    # carries slots [1, 2] in exactly the 195 sections that have per-kernel slot 28 and slots [1]
    # in exactly the 5,805 that do not - 6,001 sections, zero exceptions - and the extra field is
    # THE SAME ALLOCATION IN FOUR-BYTE WORDS: slot 2 == slot 28 / 4 in 1,993 of 1,993 sections
    # that carry both. at2-tg allocates 4 bytes and stores 1; the atm-*-tg family allocates 256
    # and stores 64. So this is derived from an ABI input already required, not a new handoff.
    if 28 in lay:
        words = (abi.get("pk_values") or {}).get(28)
        if words is None:
            raise Missing("slot 28 is in the slot set but pk_values[28] is absent; the name "
                          "table's second field is the same allocation in words and cannot be "
                          "written without it")
        nametab = S.Table({1: 8, 2: 4},
                          {1: ("<I", S.Ref(mainname)), 2: ("<I", words // 4)}, 12, 12,
                          name="nametab")
        led["name table words"] = ("DERIVED: slot 28 / 4 - 1,993 of 1,993 sections carrying both, "
                                   "0 exceptions")
    else:
        nametab = S.Table({1: 4}, {1: ("<I", S.Ref(mainname))}, 8, 8, name="nametab")
    led["entry point name"] = "DERIVED: the string %r, this object's own symbol" % M.SYMBOLS[0]
    # THE ROOT'S DECLARED LENGTH IS 14 WHEN THE KERNEL HAS A THREADGROUP ALLOCATION, 12 OTHERWISE.
    # Measured over 6,001 cached sections with no exception: every one of the 195 roots declaring
    # 14 carries per-kernel slot 28, and every one of the 5,805 declaring 12 does not. Slot 18 is
    # nearly the same predicate and is NOT the one - it has six sections declaring 12 while slot 18
    # is present, so taking the threadgroup boolean here instead of the allocation would be wrong
    # six times. Slot 28 is already an ABI input, so this is a derivation and not a new handoff.
    root_tlen = 14 if 28 in lay else 12
    led["root declared length"] = ("DERIVED: 14 with a threadgroup allocation (slot 28), else 12 - "
                                   "6,001 of 6,001 sections, 0 exceptions")
    root = S.Table({0: 8, 3: 4}, {0: ("<I", S.Ref(pk)), 3: ("<I", S.Ref(nametab))},
                   root_tlen, root_tlen, name="root")
    # THE PLACEMENT ORDER IS APPLE'S, READ OFF a4-cmpx-uni's OWN ADDRESSES rather than chosen.
    # Ascending: root at 16, the name table at 28, `agc.main` at 44, the per-kernel table at 124,
    # slot 27's vector at 184, slot 29's at 188, slot 26's at 196, the v0 record at 216,
    # `agc.main.constant_program` at 240, slot 13's vector at 272, then slots 12, 10, 8 and 6 at
    # 276, 280, 284 and 288, the binding vector at 292 and the slot-2 vector at 304.
    #
    # THIS ORDER DID NOT CLOSE THE GAP AND IS KEPT ANYWAY, because it is what Apple does and the
    # next person to look should not have to re-derive it. With the content complete, sections come
    # out 4 to 32 bytes LARGER than Apple's and first differ at byte 128 - the first field of the
    # per-kernel table, where the offsets begin - and matching the node order does not change that.
    # What remains is SLACK: g17mdgen.place leaves room this file has not accounted for, and the
    # four q-slots are overwritten there with one shared relative offset (correctly - four
    # consecutive field addresses holding one value ARE four consecutive targets, which is what
    # a4-cmpx-uni does at 144/148/152/156 -> 276/280/284/288). The remaining gap is placement
    # slack, not missing content, and that is a different investigation.
    order = ([root, nametab, mainname, pk]
             + [wvecs[q] for q in (27, 29) if q in wvecs]
             + ([firstvec, v0, v0words, cpname] if _cp_present else [firstvec])
             + [wvecs[q] for q in (13,) if q in wvecs]
             + [qvecs[q] for q in sorted(qvecs, reverse=True)]
             + [bindvec, lastvec]
             # THE SLOT-2 RECORDS ARE PLACED IN REVERSE MEMBER ORDER, which is the same
             # descending-address law the binding records follow - 44,891 vectors with two or more
             # records, zero exceptions. d_cbuf shows it with three: member 0 (kind 6) at 388,
             # member 1 (kind 3) at 364, member 2 (kind 5) at 332. This side placed k3 then k6 then
             # the extras, which is member order, and only broke the law when an extra was present.
             + list(reversed(([] if k6 is None else [k6]) + [k3] + xrecs))
             + list(reversed(recs)))
    # VTABLE SHARING. Apple gives one vtable to every record with the same SLOT MAP and the same
    # DECLARED LENGTH, and the sharer may sit before the vtable it uses: ac2-128x32x64's binding
    # records are at 416, 444 and 472, the middle one owns the vtable at 432, and the first reaches
    # FORWARD to it with a soffset of -16. Members 1 and 2 have identical slot maps and both
    # declare 16; member 0 declares 8 and keeps its own.
    #
    # THE OWNER IS THE ONE PLACED LATER, so in placement order the LAST member of each group owns
    # the vtable and every earlier member points at it.
    #
    # THIS IS THE 784 DEFECT'S SUBJECT FROM THE OTHER SIDE. On the reconstruction path sharing and
    # placement are mutually recursive and 1,064 sections have no fixed point. Here the binding
    # offsets arrive as an ABI INPUT, so the slot maps are fixed before anything is placed, and the
    # decision is made on the NATURAL declared length - the one the emission order gives before any
    # slack is absorbed. If absorbing slack later drives a group's lengths apart, the share is
    # broken and the section is placed again.
    placement = [n for n in order if isinstance(n, S.Table)]

    def decide_sharing():
        owner = {}
        for n in placement:
            n.shares = None
        for n in placement:
            if n is root or n is nametab or n is pk:
                continue
            key = (tuple(sorted(n.slots.items())), n.tlen)
            owner[key] = n                      # the LAST in placement order owns it
        for n in placement:
            if n is root or n is nametab or n is pk:
                continue
            o = owner.get((tuple(sorted(n.slots.items())), n.tlen))
            if o is not None and o is not n:
                n.shares = o

    doc = S.Doc(root, order)
    # NO q_slots, SO g17mdgen.place LEAVES THESE FIELDS ALONE. Its law writes ONE number into all
    # four - the offset computed for SLOT 4 - and that is right only if all five fields sit at the
    # same address. They do not: in a4-cmpx-uni slot 4's field is at 164 holding 128 (-> the
    # binding vector at 292), while slots 6, 8, 10 and 12 sit at 156, 152, 148 and 144 holding
    # 132 (-> 288, 284, 280, 276). One shared VALUE across four consecutive fields is what makes
    # four consecutive targets; borrowing slot 4's value instead lands every one of them four
    # bytes early. The Refs this file already builds resolve each field from its own address,
    # which is the same arithmetic done correctly.
    doc.q_slots = ()
    doc.pk, doc.bindvec = pk, bindvec
    M.place(doc)
    # decide_sharing() IS NOT CALLED, and that is a measured decision rather than an omission.
    # Apple demonstrably shares vtables - ac2-128x32x64's third binding record sits at 416 with a
    # soffset of -16, reaching forward to the vtable at 432 that the second record owns - and
    # g17schema can now express it. But "share whenever the slot map and the declared length match"
    # is too eager: enabling it over 3,000 objects takes byte-exactness from 2,970 to 2,952 and
    # introduces 14 sections whose lengths do not settle. It shares pairs Apple leaves separate.
    #
    # THE REAL RULE IS NARROWER AND IS NOT RECOVERED. Left off rather than shipped at a loss.
    M.place(doc)
    # THE LAST RECORD'S DECLARED LENGTH ABSORBS THE SLACK TO THE SECTION'S END, which is the
    # back-to-front law this project already states for blocks, applied to the final one. A
    # record's fields hang from its END, so a record that absorbs slack does not merely declare a
    # bigger number - its fields MOVE DOWN inside it. a4-cmpx-uni's last binding record has a
    # natural length of 9 and declares 12, and its kind byte sits at offset 11 rather than 8:
    # 372 + 12 = 384, the section's end. Emitting the natural length instead left four bytes wrong
    # in an otherwise byte-identical section, and 680 of 680 corpus records with this slot set
    # declare 12 with the kind at 11.
    # EVERY RECORD ABSORBS THE SLACK TO THE NEXT NODE, not just the last one. A record's fields
    # hang from its END, so absorbing slack does not merely change a declared number - the fields
    # MOVE DOWN inside it. This started as a rule for the final record, whose slack runs to the
    # section's end; atm-f-load-dev-void-1-r0 showed it applies in the middle too, differing from
    # Apple's bytes in exactly ONE place: a kind-6 record declaring 18 where this side emitted its
    # natural 16. That is the 323-of-2,363 case in the kind-6 census.
    # THE SECTION'S LENGTH IS FOUR-ALIGNED, NOT SIXTEEN. Measured over 21,110 objects: every
    # __GPU_METADATA length is a multiple of 4, and only 26.9% are a multiple of 16. g17schema.emit
    # rounds to 16 by default, which is right for a quarter of sections and four to twelve bytes
    # too long for the rest - and because the last record absorbs the slack to the section's end,
    # the wrong end also gives the wrong declared length on that record. One constant, two wrong
    # numbers. The shared default is left alone because the executing g17mdgen path is built on it.
    def _section_end(doc):
        return (max(n.addr + (n.size if isinstance(n, S.Table) else n.total)
                    for n in doc.nodes) + 3) & ~3

    # THE SECTION'S END IS FROZEN BEFORE ABSORBING, and that is what stops the runaway. The last
    # record absorbs the slack to the section's end; if the end is recomputed from the absorbed
    # sizes, growing the tail grows the section, which widens the tail's slack, which grows it
    # again - sixteen bytes per pass, forever. ac2-128x32x64 shows both halves: with sharing
    # applied its records land at 416, 444 and 472 with declared lengths 16, 16 and 8, which is
    # EXACTLY Apple's layout, and the unfrozen loop then walked it away one record-width at a time
    # until the fixed-point guard gave up. The end is a property of the placement, not of the
    # lengths that placement produces.
    frozen = [None]

    def absorb():
        placed = sorted(doc.nodes, key=lambda n: n.addr)
        starts = [n.addr - n.vlen if isinstance(n, S.Table) else n.addr for n in placed]
        if frozen[0] is None:
            frozen[0] = _section_end(doc)
        end = frozen[0]
        moved = False
        for n in doc.nodes:
            if n not in recs and n is not k6 and n is not k3 and n not in xrecs:
                continue
            later = [st for st in starts if st > n.addr]
            want = (min(later) if later else end) - n.addr
            if want > n.tlen:
                if n is k6:
                    vec, kind = 2, 6
                elif n is k3:
                    vec, kind = 2, 3
                elif n in xrecs:
                    vec, kind = 2, extra2[xrecs.index(n)][0]
                else:
                    vec, kind = 4, 5
                # THE OCCUPIED SIZE GROWS WITH THE DECLARED LENGTH, and it has to. Decoupling
                # them breaks the feedback loop - a bigger body pushes the next node later, which
                # widens this node's slack, which grows it again - but it also puts the record's
                # fields OUTSIDE the bytes it occupies, because a record's fields hang from the END
                # of its declared length. Measured: decoupling took the three refusals to zero and
                # produced two sections whose kind-6 record reads as KIND 1. A refusal is honest
                # and a wrong kind byte is not, so the coupling stays and the oscillation is
                # refused by name.
                n.slots = dict(E.record_offsets_at(vec, kind, tuple(sorted(n.fields)), want))
                n.tlen = want
                n.size = max(want, n.tlen)
                moved = True
        return moved

    # A FIXED POINT, bounded. Absorbing slack moves later nodes, which changes the slack of earlier
    # ones. Three passes settle every section measured; the bound is here so a section that does
    # NOT settle refuses instead of looping - the 784 defect is exactly that oscillation and it is
    # not going to be resolved by spinning.
    # A SHARE IS BROKEN AND NEVER RE-MADE, which is what makes this terminate where the
    # reconstruction path does not. Sharing is decided on the natural declared lengths; absorbing
    # slack can then drive a group's lengths apart, and when it does the share is dropped. Dropping
    # is MONOTONIC - a broken share is never restored - so the loop cannot cycle, which is exactly
    # the property the 784's fixed point lacks: there, unsharing changes the layout, which changes
    # the slack, which makes the pair shareable again.
    def unshare_stale():
        dropped = False
        for n in placement:
            o = n.shares
            if o is not None and (n.tlen != o.tlen or n.slots != o.slots):
                n.shares = None
                dropped = True
        return dropped

    # ONE ABSORB PASS, NOT A LOOP. Every `want` is measured against the SAME placement and applied
    # together; re-measuring after applying is self-feedback, because a record that grows pushes
    # the node after it later, which widens that record's own slack by the same amount. Traced on
    # ac2-128x32x64 it advances sixteen bytes a pass and never settles, and the correct layout -
    # records at 416, 444 and 472, Apple's exactly - is the state the SECOND pass destroys.
    for _ in range(2):
        moved = absorb()
        stale = unshare_stale()
        if not moved and not stale:
            break
        M.place(doc)
    else:
        raise ValueError("the declared lengths do not settle: absorbing slack keeps moving nodes, "
                         "which is the mutually-recursive case this side refuses rather than "
                         "iterating")
    led["declared lengths"] = "DERIVED: each record absorbs the slack to the next node"

    # SHARING, DECIDED BY VERIFICATION RATHER THAN DERIVATION. The rule is exactly "same slot map
    # and same DECLARED length" - 35 pairs over 3,001 sections, no counterexample in either
    # direction - but the declared length is the slack the layout produces, so the key cannot be
    # computed before the decision it makes. It CAN be checked afterwards.
    #
    # So: converge without sharing, take the pairs whose FINAL keys match, apply sharing, converge
    # again, and keep the assignment only if every shared pair still has a matching key. If it does
    # not, the assignment was not a fixed point and the section is emitted unshared. Guessing is
    # replaced by trying and checking, which the population size makes affordable.
    def _snapshot():
        return [(n, dict(n.slots), n.tlen, n.size, n.shares) for n in placement]

    def _restore(snap):
        for n, sl2, tl, sz2, sh in snap:
            n.slots, n.tlen, n.size, n.shares = sl2, tl, sz2, sh

    before = _snapshot()
    groups = collections.defaultdict(list)
    for n in placement:
        if n is root or n is nametab or n is pk:
            continue
        groups[(tuple(sorted(n.slots.items())), n.tlen)].append(n)
    cands = [g for g in groups.values() if len(g) > 1]
    if cands:
        for g in cands:
            for n in g[:-1]:
                n.shares = g[-1]
        # NO RE-ABSORPTION HERE. The declared lengths were settled before sharing was applied, and
        # sharing only removes vtable bytes - it does not change which slack each record is
        # claiming. Re-absorbing measures every record against a placement that has already moved
        # and grows it again, which is the self-feedback traced on ac2-128x32x64: the state right
        # after this placement IS Apple's layout, records at 416, 444 and 472, and a second pass
        # walks it away sixteen bytes at a time.
        M.place(doc)
        ok = True
        if ok:
            for g in cands:
                key = (tuple(sorted(g[-1].slots.items())), g[-1].tlen)
                for n in g[:-1]:
                    if (tuple(sorted(n.slots.items())), n.tlen) != key:
                        ok = False
        if ok:
            led["vtable sharing"] = ("VERIFIED: %d group(s) share, and the key still holds after "
                                     "the layout settled" % len(cands))
        else:
            _restore(before)
            frozen[0] = None
            M.place(doc)
            led["vtable sharing"] = ("NOT APPLIED: the shared assignment is not a fixed point - "
                                     "the key contains the length the layout produces")
    led["section length"] = "DERIVED: 4-aligned - 21,110 of 21,110 objects, 16-aligned in 26.9%"
    return S.emit(doc, size=_section_end(doc))


# THE CONTRACTS WHOSE __GPU_LD_MD THIS SIDE CAN DERIVE, and nothing else.
#
# g17ldmd.build(entry) reproduces ONE measured class byte-for-byte. Applying it to every contract is
# how 291 of 293 out-of-sample objects got a confidently wrong LD section: the main table's slot set
# varies by class - a constant core {2, 5, 6, 7, 18, 29, 38, 40} with slot 1 and one of {23, 24}
# appearing or not - and no fact in the contract has been shown to decide it. The compiler owner
# said from the start that they have no measurement determining that slot set, and they were right;
# what was wrong was this side deriving it anyway for classes it never measured.
#
# So the derivation is scoped to the class it was measured on, by exact contract signature, and
# every other contract is REFUSED until its class is measured. A refusal is a missing capability;
# a wrong section is a wrong image.
LD_DERIVABLE = {
    (((1, 0, False), (2, 2, True)), 64): "scalar-buffer-two-bindings-measured-v2",
}


# __GPU_STATS_MD IS A TEMPLATE PLUS TWO TIMINGS, which is a much sharper statement than the one
# this file used to carry. "96 bytes with 20,999 distinct patterns over 20,999 objects" is true and
# it reads as noise; measured across 393 sampled sections, NINETY of the ninety-six bytes are
# constant - down to the trailing string "backend" - and only two three-byte values ever move.
#
# Those two are not derivable and not derivable IN PRINCIPLE: 387 distinct values in 393 objects, no
# correlation with text length (ratios from 441 to 1018), one always larger than the other, sizes
# around 50,000 and 20,000. They record how long Apple's compiler took. No contract can carry them,
# so no authoring system can reproduce this section byte-for-byte - and tools/g17statsprobe.py
# showed by dispatch that the loader never reads it.
#
# So the template is emitted with the timings zeroed: ninety bytes right, six that cannot be known.
STATS_TEMPLATE = bytes.fromhex(
    "140000000000000000000a0018000400100014000a0000000000000000000000000000000c0000000c0000000400040004000000010000000c000000080010000c00040008000000000000000000000004000000070000006261636b656e6400")
STATS_TIMING_SPANS = ((24, 27), (72, 75))


def _constant_program(abi, led):
    """Presence is a class fact: an END-only program can still have its record."""
    if "constant_program" in abi:
        return bool(abi["constant_program"]), "ABI INPUT constant_program=%s" % abi["constant_program"]
    return True, ("ASSUMED PRESENT outside a measured class. END-only programs can retain a "
                  "constant-program record; the executed two-binding class swept it away.")


def _takes_texture_route(abi):
    """Will _metadata dispatch this contract to the texture route? Same condition, one place."""
    resources = abi.get("resources")
    return isinstance(resources, dict) and bool(resources.get("textures"))


def _states_argument_bytes(abi):
    """Does a texture contract state slot 1 itself? The texture route reads it as argument_bytes.

    THE GUARD MUST NOT OUTRANK THE ROUTE'S OWN INPUT. A texture contract that states argument_bytes
    has answered the question; refusing it here would refuse a contract that supplies the very
    fact the refusal asks for - which is the regression case that caught this.
    """
    resources = abi.get("resources")
    return isinstance(resources, dict) and resources.get("argument_bytes") is not None


def _texture_shape(abi):
    """A one-line description of a texture contract's shape, or None if it is not one.

    Named from the contract's own resource lists rather than from the binding count, because the
    count is exactly what must not decide this: the declared buffers are only part of the shape.
    """
    resources = abi.get("resources")
    if not isinstance(resources, dict):
        return None
    textures = resources.get("textures") or []
    samplers = resources.get("samplers") or []
    internal = resources.get("internal") or []
    if not (textures or samplers):
        return None
    return ("texture contract with %d declared buffer(s), %d texture(s), %d sampler(s) and %d "
            "internal resource(s)" % (len(abi.get("bindings") or []), len(textures),
                                      len(samplers), len(internal)))


def _pk_slot1(bindings, led):
    """Per-kernel slot 1, from the measured layout for THIS binding count.

    Not a default. g17mdgen owns one measured layout per binding count and refuses a count it has
    never measured, so this asks it and lets its refusal through: a three-buffer contract is
    rejected here, before anything is authored or loaded, rather than approximated by stretching
    the two-binding layout. The value is read out of a section that generator actually builds, so
    the derivation is a measurement rather than a constant copied into a second place.
    """
    from . import mdgen as M
    try:
        ref = M.describe(bytes(M.build([b[0] for b in bindings], restore_swept=False)))
    except ValueError as e:
        # A count with no measured layout is a MISSING CONTRACT INPUT, not an internal error. The
        # distinction is load-bearing downstream: the acceptance probe reports a Missing as
        # "blocked", which is a capability statement a caller can act on, and an escaping
        # ValueError as "error", which reads as a defect in this side. Refuse in the shape that
        # says whose move it is.
        raise Missing("no measured metadata layout for %d bindings (%s), so pk_slot1 cannot be "
                      "derived here; the backend must state it, or this count needs its own "
                      "measured layout before anything is authored" % (len(bindings), e))
    # The per-kernel table's position is the LAYOUT's, not SCALAR's. Hardcoding SCALAR's 128 here
    # made a three-binding class report "carries no per-kernel slot 1" when its table is at 124.
    # No promoted-range argument here on purpose: this only needs the per-kernel table's POSITION,
    # and the five- and six-buffer classes put it at the same 124, so the count is enough and `abi`
    # is not in scope.
    layout = M.layout_for([b[0] for b in bindings]) or M.SCALAR
    table = ref["tables"].get(layout["pk"])
    field = (table or {}).get("fields", {}).get(1)
    if field is None:
        raise Missing("the measured layout for %d bindings carries no per-kernel slot 1, so the "
                      "backend must state pk_slot1" % len(bindings))
    led["per-kernel slot 1"] = ("DERIVED from the measured layout for %d bindings: a %d-byte field "
                                "at offset %d" % (len(bindings), field[1], field[0]))
    return field[2]


def _ld_md(entry, abi, led, bindings=()):
    """The format is this side's - it round-trips byte-exactly on 20,999 of 21,055 - and the main
    table's slot set is the compiler's WHEN THE COMPILER STATES IT, derived from the entry when it
    does not."""
    from . import emit as E
    from . import mdgen as M
    from . import schema as S
    # The narrow texture class has its own measured LD graph.  Do this before the ordinary buffer
    # layout selector: the latter can produce a valid 224-byte section with the wrong slot set,
    # and the loader accepts that foreign section without complaint.  g17texldmd writes the graph
    # from checked-in structure and constants; it never reads the retained witness while building.
    if _takes_texture_route(abi):
        from . import texldmd as g17texldmd
        reasons = g17texldmd.admit(abi)
        if reasons:
            raise Missing("this texture LD metadata is outside the measured narrow class: %s"
                          % "; ".join(reasons))
        public = ((abi.get("resources") or {}).get("texture_public_indices")
                  if isinstance(abi, dict) else None)
        # SLOT 9 IS THE ONLY FIELD A SECOND TEXTURE MOVES. With no mapping this emits exactly the
        # bytes it always did; with one it authors the mask over the PUBLIC [[texture(n)]] slots,
        # which reproduces the retained tx2f-pair and tx2f-3and7 sections byte for byte.
        mask = g17texldmd.read_mask_for(public) if public else None
        out = g17texldmd.build(entry, read_mask=mask)
        led["__GPU_LD_MD"] = ("DERIVED from the source-owned narrow texture class graph: %d bytes, "
                               "byte-identical to both retained texture witnesses" % len(out)
                              if mask is None else
                              "DERIVED from the source-owned narrow texture class graph: %d bytes, "
                              "slot 9 = %d over the stated public texture indices %s"
                              % (len(out), mask, list(public)))
        return out
    if "ld_md_slots" not in abi:
        from . import ldmd as g17ldmd
        from . import mdgen as _M
        sig = (tuple(tuple(b[:3]) for b in bindings), entry)
        known = LD_DERIVABLE.get(sig)
        if known is None and _M.layout_for(
                [b[0] for b in bindings], promoted_ranges=abi.get("promoted_ranges")) is not None:
            known = "the measured layout for binding indices %s" % ([b[0] for b in bindings],)
        if known is None:
            # ASKED WITH THE CONTRACT'S OWN FACTS TOO. The probe above asks by binding indices
            # alone, which answers where one index list means one class and misses one where the
            # WRITE MASK selects - the one-binding written-buffer-0 class is measured and this
            # probe could not see it, so a contract it covers was refused for want of a class that
            # exists.
            _own = _M.layout_for([b[0] for b in bindings],
                                 promoted_ranges=abi.get("promoted_ranges"),
                                 written=[bool(b[2]) for b in bindings],
                                 instructions=abi.get("instruction_count"),
                                 back_edge=bool(abi.get("has_back_edge")))
            if _own is not None:
                known = ("the measured layout for binding indices %s with the contract's own write "
                         "mask" % ([b[0] for b in bindings],))
        if known is None:
            raise Missing(
                "__GPU_LD_MD is not derivable for this contract. The main table's slot set is a "
                "CLASS fact - a constant core with slot 1 and one of {23, 24} varying - and no "
                "contract fact has been shown to decide it. g17ldmd.build(entry) reproduces the "
                "measured class %s and nothing else, so applying it here would author a confidently "
                "wrong section. Supply ld_md_slots and ld_md_values, or author a measured class."
                % ", ".join(sorted(set(LD_DERIVABLE.values()))))
        # A FULL CLASS GETS ITS SWEPT REGIONS BACK. The executed class zeroes them and keeps its
        # zeros; an unswept class carries eighteen format constants and the stage name "compute",
        # each constant across every sampled section of that size.
        from . import mdgen as _MD
        _l = _MD.layout_for([b[0] for b in bindings], promoted_ranges=abi.get("promoted_ranges"))
        full = bool(abi.get("reproduce_measured_class") and _l and _l.get("name"))
        led["__GPU_LD_MD"] = ("DERIVED from the entry PC for the measured class %s%s" %
                              (known, ", with the swept regions restored" if full else ""))
        return bytes(g17ldmd.build(entry, restore=full))
    slots = {int(k) for k in abi["ld_md_slots"]}
    vals = {int(k): v for k, v in abi["ld_md_values"].items()}
    if 6 not in slots:
        raise ValueError("the LD_MD main table must carry slot 6, the entry PC")
    led["__GPU_LD_MD slot set"] = "ABI INPUT ld_md_slots"
    led["__GPU_LD_MD values"] = "ABI INPUT ld_md_values"
    led["__GPU_LD_MD [main,6]"] = "DERIVED: the entry PC, == _agc.main in 21,001 of 21,001"
    st = E._store()
    # the main table's own emission order is not in the record relation; lay it out by slot id
    # descending from the declared length, which reproduces Apple's maps for this table.
    tlen = 4 + sum(8 if s == 29 else 4 for s in slots)
    lay, pos = {}, tlen
    for s in sorted(slots):
        w = 8 if s == 29 else 4
        pos = (pos - w) - ((pos - w) % w)
        lay[s] = pos
    fields = {s: ("<Q" if s == 29 else "<I", vals.get(s, 0)) for s in slots}
    fields[6] = ("<I", entry)
    main = S.Table(lay, fields, tlen, tlen, name="ldmain")
    t3 = S.Table({3: 4}, {3: ("<B", 1)}, 6, 6, name="ldt3")
    led["__GPU_LD_MD [T3,3]"] = "DERIVED: 1 in 21,001 of 21,001"
    root = S.Table({0: 8, 3: 4}, {0: ("<I", S.Ref(main)), 3: ("<I", S.Ref(t3))}, 12, 12, name="ldroot")
    doc = S.Doc(root, [root, main, t3])
    S.place(doc)
    return S.emit(doc)


def _arch(flag, led):
    """A root table plus a sub-table carrying ONE optional field: 40 bytes set, 32 elided."""
    from . import schema as S
    # THE TWO FORMS, READ OFF THE DELIVERED BYTES rather than assumed.
    #
    # SET (40 bytes, Apple emits it in 263 of 3,001 cached objects). Root at 12 with vtable at 6
    # declaring vlen 6 and tlen 8, slot 0 at offset 4 holding a reference; sub-table at 28 with
    # vtable at 20, slot 1 at offset SEVEN holding a single BYTE. Writing that flag as a four-byte
    # field at offset 4 - which this did - produced 48 bytes, a size Apple never emits. Caught by
    # the ISA peer while checking the codex handoff.
    #
    # ELIDED (32 bytes, 2,738 of 3,001, and what the executed scalar profile ships). The root is
    # the same shape and slot 0 is present holding ZERO: a null reference, and NO sub-table at all.
    # Emitting an empty sub-table and pointing at it is a different section with the same length.
    if flag:
        sub = S.Table({1: 7}, {1: ("<B", 1)}, 8, 8, name="archsub")
        root = S.Table({0: 4}, {0: ("<I", S.Ref(sub))}, 8, 8, name="archroot")
        nodes = [root, sub]
    else:
        sub = None
        root = S.Table({0: 4}, {0: ("<I", 0)}, 8, 8, name="archroot")
        nodes = [root]
    doc = S.Doc(root, nodes)
    S.place(doc)
    # THE SECTION IS AT LEAST 32 BYTES AND 8-ALIGNED: elided ends at 20 and ships 32, set ends at
    # 36 and ships 40. g17schema.emit's default rounds to 16, which gives 32 and 48.
    end = max(n.addr + (n.size if isinstance(n, S.Table) else n.total) for n in doc.nodes)
    led["__GPU_ARCH_LD_MD shape"] = ("DERIVED: 40 bytes with the flag as a BYTE at offset 7 in a "
                                     "sub-table, 32 with slot 0 a null reference and no sub-table")
    return S.emit(doc, size=max(32, (end + 7) & ~7))


# WHAT THIS MUST REFUSE, kept in the tool rather than in a scratch script. A tool that refuses
# only in principle refuses nothing, and one of these cases INVERTED today: a missing `stats` used
# to be refused by name and is now accepted, because a dispatch retired it. A case list in a file
# records that; a case list in someone's memory does not.
#
#   (label, how to damage the inputs, what must happen)
REFUSALS = [
    ("no arch_flag",        lambda a, e: (a.pop("arch_flag"), (a, e))[1],        "Missing"),
    # RETIRED AS A DERIVATION TODAY. It was align_up(block, 8) on three texture witnesses and is
    # right in 28% of 21,020 sections; a wrong number emitted silently is worse than a refusal.
    ("no pk_slot1",         lambda a, e: (a.pop("pk_slot1"), (a, e))[1],          "Missing"),
    ("no ld_md_slots",      lambda a, e: (a.pop("ld_md_slots"), (a, e))[1],      "Missing"),
    ("no ld_md_values",     lambda a, e: (a.pop("ld_md_values"), (a, e))[1],     "Missing"),
    ("stats 64 bytes",      lambda a, e: (a.__setitem__("stats", bytes(64)), (a, e))[1], "ValueError"),
    ("entry PC 100",        lambda a, e: (a, 100),                               "ValueError"),
    ("ld_md_slots has no 6", lambda a, e: (a.__setitem__("ld_md_slots",
                                           [x for x in a["ld_md_slots"] if x != 6]), (a, e))[1],
                                                                                 "ValueError"),
    ("pk value off the map", lambda a, e: (a.__setitem__("pk_values", {99: 1}), (a, e))[1],
                                                                                 "any"),
    # THE GUARD THAT WAS MISSING. Slot 6 carries 132 in a4-cmpx-uni, and 132 is a POINTER to a
    # vector. Before this case existed the author wrote 132 in as an integer and returned a
    # section 64 bytes shorter than the one Apple ships for the identical binding signature.
    ("scalar into a vector slot", lambda a, e: (a.__setitem__("pk_values", {6: 132}),
                                                a.__setitem__("pk_extra", (6,)), (a, e))[2],
                                                                                 "ValueError"),
    ("overriding a derived slot", lambda a, e: (a.__setitem__("pk_values", {3: 16}),
                                                a.__setitem__("pk_extra", (3,)), (a, e))[2],
                                                                                 "ValueError"),
    # THE INVERTED ONE. Not a refusal any more, and the ledger has to SAY so - an accepted input
    # that quietly appeared from nowhere is the failure mode this whole tool exists to avoid.
    ("no stats (now DERIVED)", lambda a, e: (a.pop("stats"), (a, e))[1],         "accepted:DERIVED"),
]


def _abi():
    return {
        "ld_md_slots": [1, 2, 5, 6, 7, 18, 29, 38, 40],
        "ld_md_values": {1: 4, 2: 40, 5: 36, 7: 16, 18: 1, 29: 1, 38: 12, 40: 80},
        "arch_flag": True,
        "pk_slot1": 4,
        "stats": bytes(96),
        "pk_values": {15: 1, 16: 1},
        "pk_extra": (15, 16),
    }


def refuse():
    print("WHAT THE AUTHOR REFUSES, and the one case a dispatch inverted\n")
    bad = 0
    for label, damage, want in REFUSALS:
        abi, entry = damage(_abi(), 64)
        try:
            _secs, led = author(text=b"", entry=entry,
                                bindings=[(1, 0, False), (2, 2, True)], abi=abi)
            if want.startswith("accepted"):
                tag = want.split(":")[1]
                ok = led.get("__GPU_STATS_MD", "").startswith(tag)
                print("   %-24s ACCEPTED, ledger says %-8s %s"
                      % (label, tag, "as required" if ok else "<-- BUT THE LEDGER DOES NOT SAY SO"))
                bad += not ok
            else:
                print("   %-24s ACCEPTED  <-- MUST HAVE BEEN REFUSED (%s)" % (label, want))
                bad += 1
        except Exception as e:
            got = type(e).__name__
            ok = want in ("any", got)
            print("   %-24s %-12s %s" % (label, got, str(e).split("\n")[0][:64]
                                         if ok else "<-- WRONG FAILURE, wanted %s" % want))
            bad += not ok
    # THE ELEMENT-TYPE CASES vary the BINDINGS rather than the abi, so they are checked here
    # rather than through the table above. A half-typed binding must be ACCEPTED and must say in
    # the ledger that the type is not encoded; an imageblock and an unwitnessed type must refuse.
    print("\n   element_type, consumed rather than ignored\n")
    for label, binds, want in (
        ("half buffers accepted", [(1, 0, False, "half"), (2, 2, True, "half")], "accepted"),
        ("float and half mixed", [(1, 0, False, "float"), (2, 2, True, "half")], "accepted"),
        ("an imageblock", [(1, 0, False, "imageblock"), (2, 2, True, "half")], "ValueError"),
        ("a type no kernel declares", [(1, 0, False, "quarterfloat"), (2, 2, True, "half")],
         "ValueError"),
    ):
        try:
            _s, led = author(text=b"", entry=64, bindings=binds, abi=_abi())
            note = led.get("element types", "")
            ok = want == "accepted" and "NOT ENCODED" in note
            print("   %-26s ACCEPTED   %s" % (label, "ledger records it" if ok else
                                              "<-- LEDGER DOES NOT RECORD THE TYPE"))
            bad += not ok
        except Exception as e:
            ok = want == type(e).__name__
            print("   %-26s %-12s %s" % (label, type(e).__name__,
                                         str(e)[:58] if ok else "<-- wanted %s" % want))
            bad += not ok

    print("\n   %s" % ("every case behaves as written" if not bad
                        else "%d CASES DISAGREE - the refusals are not what the file claims" % bad))
    return 1 if bad else 0


def freeze_profile(objpath, name, out=None):
    """Extract all five sections from a DELIVERED object and write a pinned profile.

    This is how a measured class becomes a contract: not by describing it, by taking its bytes.
    The scalar profile was frozen this way and its 844 bytes are what the regression compares
    against. When the FP16 scan compiles, freezing it is one command and no code changes.

    It records the binding list, entry and every section's length and sha256 ALONGSIDE the hex, so
    a later reader can check the freeze against the object it claims to come from.
    """
    import hashlib, json
    from . import facts as g17facts
    from . import gpumd as GM
    from . import obj as g17obj
    raw = open(objpath, "rb").read()
    sects, syms = g17obj.sections_of(raw)
    out_sections = {}
    for nm in ("__GPU_METADATA", "__GPU_LD_MD", "__GPU_ARCH_LD_MD", "__GPU_STATS_MD",
               "__GPU_REMARKS_MD"):
        key = nm + ",__compute"
        if key not in sects:
            raise ValueError("the delivered object has no %s; a profile pins all five sections"
                             % nm)
        off, size = sects[key]
        b = bytes(raw[off:off + size])
        out_sections[nm] = {"size": len(b), "sha256": hashlib.sha256(b).hexdigest(),
                            "hex": b.hex()}
    md = bytes.fromhex(out_sections["__GPU_METADATA"]["hex"])
    recs = g17facts.binding_records(md) or []
    binds = [[r.get(1, 0), r.get(2, 0), bool(r.get(3, 0))] for r in recs]
    entry = syms.get("_agc.main")
    # DOES THIS CLASS ALREADY EXIST? A freeze that reproduces a known profile's bytes is the same
    # architectural class wearing a different name, and saying so is more useful than storing a
    # duplicate. It is also a FALSIFIABLE PREDICTION for the FP16 scan: element type does not reach
    # __GPU_METADATA - g17mdgen.build takes the binding INDICES and has no type parameter at all -
    # so an FP16 image with these two bindings should freeze to the SAME 844 bytes as the scalar
    # class. If it does not, that proof is wrong and this check is where it shows.
    matches = []
    for known, meta in PROFILES.items():
        rp = meta.get("reference")
        if not rp:
            continue
        rp = rp if os.path.isabs(rp) else os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), rp)
        if not os.path.exists(rp):
            continue
        ref = json.load(open(rp))["sections"]
        if all(ref.get(k, {}).get("sha256") == v["sha256"] for k, v in out_sections.items()):
            matches.append(known)
    prof = {"profile": name,
            "matches_known_profile": matches,
            "frozen_from": os.path.abspath(objpath),
            "bindings": binds,
            "entry": entry,
            "scope": "Exact measured metadata for this class. A compatibility contract for one "
                     "signature, not a template to re-point at another.",
            "sections": out_sections}
    if out:
        with open(out, "w") as f:
            json.dump(prof, f, indent=1)
    return prof


def profile_regression():
    """THE 844 DELIVERED BYTES, all of them, and the combinations this refuses.

    Not a size check and not a shape check: every byte of every section is compared against
    isa/g17-scalar-abi-v2.json, which was extracted from the object that executed nine times.
    """
    import json
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ref = os.path.join(root, "isa", "g17-scalar-abi-v2.json")
    want = {k: bytes.fromhex(v["hex"])
            for k, v in json.load(open(ref))["sections"].items()}
    # AND THE FROZEN RECORD IS ITSELF CHECKED AGAINST A DELIVERED OBJECT, because the goal asks for
    # the bytes "extracted from the delivered object" and a JSON is a transcription of them. The
    # 33x384 bundle is one of the three shapes that executed; freezing its scan.o must reproduce
    # the same 844 bytes, or the contract and the artefact have drifted apart.
    delivered = os.path.join(root, "results", "g17-packed-scan-v2-33", "scan.o")
    if os.path.exists(delivered):
        got = freeze_profile(delivered, "check")["sections"]
        drift = [k for k, v in got.items() if bytes.fromhex(v["hex"]) != want[k]]
        print("   frozen record vs the DELIVERED object (%s): %s\n"
              % (os.path.relpath(delivered, root),
                 "all five sections agree" if not drift else "DRIFT in " + ", ".join(drift)))
    else:
        print("   the delivered object is not in this checkout; comparing against the frozen "
              "record only\n")
    name = "scalar-buffer-two-bindings-measured-v2"
    print("THE MEASURED SCALAR CLASS, byte for byte against the delivered object\n")
    secs, led = author(text=b"", entry=64, bindings=list(PROFILES[name]["bindings"]),
                       abi={"measured_class": name, "arch_flag": PROFILES[name]["arch_flag"]})
    bad = tot = ok = 0
    for k, v in secs.items():
        n = k.split(",")[0]
        w = want[n]
        tot += len(w)
        good = bytes(v) == w
        ok += len(w) if good else 0
        bad += not good
        print("   %-18s %4d bytes   %s" % (n, len(v), "BYTE-EXACT" if good else "DIFFERS"))
    print("\n   %d of %d delivered metadata bytes reproduced%s"
          % (ok, tot, "" if not bad else "   <-- %d SECTIONS DIFFER" % bad))
    print("\n   and what it refuses, because a compatibility profile is not a template:\n")
    for label, kw, want_msg in (
        ("an unknown profile", dict(abi={"measured_class": "scalar-v3"}), "no measured profile"),
        ("different bindings", dict(bindings=[(1, 0, False), (3, 2, True)]), "measured for bindings"),
        ("a different entry", dict(entry=128), "pins entry"),
        ("the arch flag SET", dict(abi={"measured_class": name, "arch_flag": True}), "nine hardware runs"),
    ):
        call = dict(text=b"", entry=64, bindings=list(PROFILES[name]["bindings"]),
                    abi={"measured_class": name, "arch_flag": False})
        call.update(kw)
        try:
            author(**call)
            print("   %-22s ACCEPTED  <-- must have refused" % label)
            bad += 1
        except Exception as e:
            hit = want_msg in str(e)
            print("   %-22s %s %s" % (label, type(e).__name__,
                                      str(e)[:64] if hit else "<-- wrong refusal"))
            bad += not hit
    # THE ITEMS THAT ARE NOT IN THE 844 BYTES. Byte-exactness covers the kind-6 and kind-3
    # records, the present-but-empty vectors, the entry and the prologue - they are all IN the
    # metadata, so reproducing it reproduces them. Two things on the list are not, and a
    # regression that only compares bytes would never exercise either.
    print("\n   and the facts that are NOT in the metadata, so bytes cannot check them:\n")
    tb = author(text=b"", entry=64,
                bindings=[(1, 0, False, "half"), (2, 2, True, "half")],
                abi=dict(_abi(), constant_program=True))[1]
    et = tb.get("element types", "")
    print("   %-30s %s" % ("half-typed bindings accepted",
                           "recorded: " + et[:56] if "NOT ENCODED" in et else "<-- NOT RECORDED"))
    bad += "NOT ENCODED" not in et
    for want, lab in ((True, "present"), (False, "absent")):
        cp = author(text=b"", entry=64, bindings=[(1, 0, False), (2, 2, True)],
                    abi=dict(_abi(), constant_program=want))[1].get("constant-program record", "")
        ok = ("constant_program=%s" % want) in cp
        print("   %-30s %s" % ("constant program %s" % lab,
                               "stated in the ledger" if ok else "<-- NOT STATED"))
        bad += not ok
    try:
        author(text=b"", entry=64, bindings=[(1, 0, False, "quarterfloat"), (2, 2, True)],
               abi=_abi())
        print("   %-30s ACCEPTED  <-- must have refused" % "an unrecovered element type")
        bad += 1
    except ValueError as e:
        print("   %-30s refused: %s" % ("an unrecovered element type", str(e)[:44]))

    print("\n   %s" % ("every byte and every refusal behaves as written" if not bad
                        else "%d PROBLEMS" % bad))
    return 1 if bad else 0


def main():
    if "--freeze" in sys.argv:
        i = sys.argv.index("--freeze")
        objpath, name = sys.argv[i + 1], sys.argv[i + 2]
        out = sys.argv[i + 3] if len(sys.argv) > i + 3 else None
        prof = freeze_profile(objpath, name, out)
        print("FROZEN %s from %s" % (name, prof["frozen_from"]))
        print("   bindings %s   entry %s" % (prof["bindings"], prof["entry"]))
        for k, v in prof["sections"].items():
            print("   %-18s %4d bytes  %s" % (k, v["size"], v["sha256"][:16]))
        print("   total %d metadata bytes%s"
              % (sum(v["size"] for v in prof["sections"].values()),
                 "" if not out else "   -> " + out))
        if prof["matches_known_profile"]:
            print("   SAME CLASS as %s - byte-identical in all five sections, so this is that "
                  "class under another name rather than a new one"
                  % ", ".join(prof["matches_known_profile"]))
        else:
            print("   a NEW class: no known profile matches all five sections")
        return 0
    if "--profile" in sys.argv:
        return profile_regression()
    if "--refuse" in sys.argv:
        return refuse()
    if "--demo" not in sys.argv:
        print(__doc__)
        return 0
    abi = {
        "ld_md_slots": [1, 2, 5, 6, 7, 18, 29, 38, 40],
        "ld_md_values": {1: 4, 2: 40, 5: 36, 7: 16, 18: 1, 29: 1, 38: 12, 40: 80},
        "arch_flag": True,
        "pk_slot1": 4,
        "stats": bytes(96),
        "pk_values": {15: 1, 16: 1},
        "pk_extra": (15, 16),
    }
    secs, led = author(text=b"", entry=64,
                       bindings=[(1, 0, False), (2, 2, True)], abi=abi)
    print("AUTHORED an object: %d sections\n" % len(secs))
    for k, v in secs.items():
        print("   %-30s %d bytes" % (k.split(",")[0], len(v)))
    print("\n  THE FIELD LEDGER - every field derived or a named ABI input:")
    for k, v in led.items():
        print("     %-28s %s" % (k, v))
    copied = [k for k, v in led.items() if v.startswith("COPIED")]
    print("\n  COPIED FIELDS: %s" % (copied or "NONE"))
    print("\n  and with `stats` withheld, which is now legitimate:")
    a2 = _abi(); a2.pop("stats")
    _s2, l2 = author(text=b"", entry=64, bindings=[(1, 0, False), (2, 2, True)], abi=a2)
    print("     %-28s %s" % ("__GPU_STATS_MD", l2["__GPU_STATS_MD"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
