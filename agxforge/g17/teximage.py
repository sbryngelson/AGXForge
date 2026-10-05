"""Author the texture section from an ABI v6 contract, and refuse at exactly one point.

    python3 tools/g17teximage.py results/g17-texture-contract-v1/lane31/abi.json

THE ASSIGNMENT SAYS: consume the contract through the common author and produce all native
sections without borrowing ty-2d's; and where a native section lacks a measured rule, deliver the
exact missing fact rather than substituting donor bytes. Two facts lack one - the CONTENTS of
per-kernel slot 27, and whether the slot-2 vector carries a kind-5 resource record - and the
compiler states both on its own `not_stated` list rather than guessing them.

So this places everything the contract and the measurements DO determine, reports each placement
with the rule that decided it, and then refuses by name on the two that are not determined. The
refusal is at one point and everything before it is demonstrated, which is what makes the missing
fact exact rather than a shrug: the section is complete except for two vectors, and this says
which two and what is known about each.

WHAT DECIDES EACH FIELD, and none of it is a donor byte:

    binding records   internals first ascending then users ascending, offset = 2 * rank
                      (g17resource.layout, reproduced over 12,044 corpus kernels with the
                      same-set-different-order bucket EMPTY)
    slot 3            8 * binding count
    slot 38           8 * the published coordinate slots, which the contract states; NOT the
                      texture count - M5 reads one texture twice and carries 16, M6 reads two
                      textures once each and carries 8
    slot 42           equal to slot 38 - exact on all 6,495 Apple texture sections
    slot 0            the contract's register_count
    slot 29           the contract's system_registers through the shared g17mdgen map
    slot 13           the contract's constant_pool
    slot 31           the thread-invariant spill; absent when the backend reports none
    slot 12           present and empty
    kind-6 record     present; a kind-3 record alone is 55 of 6,495 sections
    slot 27           NOT DETERMINED
    slot-2 kind-5     NOT DETERMINED

Nothing here emits bytes. A section cannot be emitted while two of its vectors are unknown, and
emitting one with a plausible value for them is the failure this file exists to refuse.
"""
import json
import os
import sys

from . import mdgen as M
from . import resource as g17resource

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Measured on every Apple section carrying per-kernel slot 38. Kept here as the evidence the
# refusal cites, so the message states the state of the question rather than asserting one.
# The internal whose presence in slot 27 nothing predicts; subtracted in every reading so the
# readings differ on what they include rather than on how they treat this index.
UNPREDICTED_INTERNAL = 44
# Measured on all nine family members: internal 48 appears in slot 27 every time, 44 never. This is
# the linker's fact about its own resources, not something the compiler should be asked to state.
UNIFORM_INTERNALS = frozenset({48})
# The coordinate slot size. 8 is what the family's measured values require: M0 publishes four bytes
# at offset 0 and carries 8, M5 covers offsets 0 and 8 and carries 16. It is the section's rule and
# not the compiler's, which is why the contract states ranges and this states the slot size.
COORDINATE_SLOT_BYTES = 8
# The competing reading: a publication's constant is a plain byte offset and a coordinate pair is
# four bytes, so x and y at 0 and 2 are one pair rather than two slots. Unmeasured either way.
COORDINATE_PAIR_BYTES = 4
# A binding record occupies four bytes of the argument block, internals first. The preload or pool
# publishes immediately after the whole DECLARED record block - not after the records the code
# happens to address, which S2 and S3 show are fewer.
RECORD_BYTES = 4
# The compiler's canonical fields, in their words. Named once so the resolver, the digest and the
# field-by-field comparison cannot describe different layouts.
TEXTURE_ABI_VERSIONS = frozenset({6, 7})
CANONICAL_FIELDS = ("declared", "internal", "order", "ranks", "record_bytes", "block_bytes",
                    "descriptor_offsets")
# An empty constant pool is the eight-byte zero vector, not a zero-length one - the class boundary
# g17rangemetadata measured: length is not emptiness.
#
# THAT JUSTIFICATION COMES FROM ANOTHER CLASS, which is worth saying where the constant is defined.
# Inside the TEXTURE class an all-zero pool takes four lengths - 4 on 235 sections, 8 on 211, 12 on
# 232, 16 on 3 - so eight is one witnessed length of four, not the empty pool's length. And the
# members compiled here from sources with no constants (E3, M2, S4) carry a ZERO-LENGTH vector, not
# a zero-filled one. The constant is kept because the two-option question it serves is the one the
# class turns on, and it is marked here so nobody reads it as "this class's empty pool is 8 bytes".
POOL_EMPTY_BYTES = 8
POOL_ALL_ZERO_LENGTHS_IN_THIS_CLASS = {4: 235, 8: 211, 12: 232, 16: 3}
# The publication operands the extent rule is measured for: op592 publishes a register (M0's
# single, C0's pair), op556 publishes a constant (C1's second, Apple's literal y). Anything else
# is outside what C0 and C1 settled.
# op10298 joins them on ONE witness: L0 publishes x through it - an ALU straight into the
# coordinate block - and its section still reads slot 38 = 8 at const(0) and const(2). One witness
# is said as one witness.
MEASURED_PUBLICATION_OPCODES = frozenset({592, 556, 10298})
# The internals every member of the family carries. A program with a different internal set is a
# different class and slot 27's rule is not measured for it.
SUPPORTED_INTERNALS = frozenset({44, 48})
# THE 6,495 IS A GATE, NOT THE CORPUS, and only one of the two facts below has it as a real bound.
# g17texfamily admits a section only if its per-kernel vtable carries slot 38, which is within one
# slot of "the vtable has 43 slots". Slot 42 cannot be READ on any of the 3,061 it drops, so
# slot42_equals_slot38 is bounded by addressability and stands with no exception. Slot 27 CAN be
# read on all 9,556, and the declared reading agrees on 727 of the dropped 3,061 - 68.2% inside the
# gate, 2.1% one schema slot outside it, 54.0% over every section that carries the field. The
# reading is a property of the vtable cohort and not of slot 27's contents, which makes the refusal
# stronger rather than weaker. Measured by intervention in tools/g17readerconstants.py; the gate
# itself is correct for the question g17texfamily asks and is not changed.
CORPUS = dict(sections=6495, slot27_declared_reading=4429, no_resource_record=2536,
              slot42_equals_slot38=6495, carrying_44_in_slot27=165,
              slot27_addressable_sections=9556, slot27_declared_reading_all=5156,
              slot27_declared_reading_outside_the_gate=727,
              slot42_addressable_sections=6495)
FAMILY = "results/g17-texture-discriminating-family-v1"
# THE FAMILY IS COMPILED AND RE-READ, and this side's refusal said otherwise for long enough to
# cost integration a dispatch: bad0854c asks for a compile of a family whose nine members have been
# committed with their objects for some time. A refusal message that states a stale fact is worse
# than one that states none, because a reader acts on it.
FAMILY_REREAD = "results/g17-texture-family-reread-v1"
SLOT10_WITNESS = "results/g17-slot10-witness-v1"
# The two retained deliveries the kind-6 placements cite. Named once so the refusal, the rules and
# the handoff all point a reader at bytes rather than at a number in a comment.
KIND6_SPLIT = "results/g17-texture-kind6-split-v1"
KIND6_CLOSURE = "results/g17-texture-kind6-closure-v1"
RECORD_ORDER = "results/g17-texture-record-order-v1"
CLASS = "results/g17-texture-class-v1"

# WHAT WOULD SETTLE EACH REFUSED FACT, because integration's dispatch makes a refusal a valid
# deliverable ONLY when it names the unresolved fact AND its settling experiment. Naming a fact
# without naming the measurement that would close it leaves the reader to invent one, and an
# invented experiment is how a class-wide unknown becomes a neighbouring class's value.
#
# Each entry says what would have to be TRUE, not who should do it - an owner is recorded separately
# in g17textureblockers.OWNERS, and two of these currently have no owner at all.
SETTLED_BY = {
    "slot27_contents":
        "integration removing it from the contract's not_stated list. This side already DERIVES it "
        "inside the supported domain - the uniformly read resources with 44 removed - and records "
        "the derivation in `placed`; the contract's own unknown is what keeps it refused, so the "
        "settling act is a decision rather than a measurement",
    "slot2_resource_record":
        "a compiled witness that CARRIES a kind-5 resource record. None of the 131 committed "
        "witnesses does, so this cannot be closed from anything this project has compiled. Eight "
        "candidate readings are scored and killed over Apple's corpus, the best reaching 67.4%, so "
        "the experiment is a source that produces the record - not another reading of the census",
    "slot2_kind9_record":
        "integration removing it from not_stated. It is DERIVED absent here by a measured rule - no "
        "non-zero constant-pool content implies no kind-9 record, 0 of 2,274 across two corpora, "
        "with 254 of 7,391 content-carrying sections doing carry one so the rule is not vacuous",
    "slot10_contents":
        "a reading that separates the 1,113 Apple texture sections carrying content in slot 10 from "
        "the 5,382 that do not. Every committed witness is empty there, so our own compiles cannot "
        "distinguish them and the experiment is on Apple's corpus or on a source that produces one",
    "slot32_value":
        "a decision about which of THREE measured values to write. The field is DECODED - it is "
        "one byte, not four, and it takes exactly 3 values across Apple's compute sections: 3 on "
        "3804, 1 on 3542 and 2 on 765 of the 8111 that carry it. The earlier wording here said "
        "'many distinct values' and 'this side neither places nor decodes it'; both came from a "
        "four-byte read and are retracted. What is missing is not a decode but a rule choosing "
        "among the three, and no readable contract fact predicts which",
    "binding_record_48_declared_length":
        "a placement. The length is the distance to the next structure - 43,411 binding records "
        "across two corpora with no exception - so it follows from laying the section out, which "
        "needs the facts above it. No further measurement of the record itself is required",
    "constant_pool_emptiness":
        "a DECISION, not a measurement. Over the 2,175 Apple sections with no pool content every "
        "readable candidate overlaps between the two states, and the compiler's backend emits no "
        "pool at all so it has chosen neither. Somebody states "
        "resources.constant_pool_emptiness as zero_length_vector or eight_byte_zero_vector",
    "slot2_kind6_split":
        "constant_pool_emptiness being stated. With the pool form known the record's presence "
        "follows from the pool-record law and field2 from the pool length; nothing else is missing",
    "slot27_is_outside_the_population_the_rule_was_measured_on":
        "a scope decision: whether a section may describe a program Apple would never have "
        "produced. Every rule this side has measured is a rule about APPLE'S compiler's output, and "
        "the rule that would bound slot 27 has never had its antecedent true on the population "
        "lane31 belongs to: 157 of 157 executed objects attributed to this project's assembler "
        "carry a slot 27 of length 0 or none; none declares a spill, so the rule is satisfied "
        "VACUOUSLY there and has never been tested. The earlier wording said this backend cannot "
        "spill at all; that absolute was never measured, and the compiler owner has since shown "
        "the spill/reload sequence IS expressible with only the address provenance unresolved, so "
        "has never emitted one is what the evidence supports. That is a measurement of the "
        "rule's REACH and no measurement resolves this fact, which remains a scope decision: "
        "knowing the rule is untested here does not say whether a section may describe such a "
        "program. Measured in g17texclass.our_backend_is_outside_the_population",
}

RECORD_FIELDS = "results/g17-texture-record-fields-v1"
RECORD_LENGTH = "results/g17-texture-record-length-v1"
BINDING_RECORD = "results/g17-texture-binding-record-v1"


# The common author's callers catch g17authorobj.Missing, so a refusal from this route has to be
# one of those or it escapes as an unexpected error two levels up. g17authorobj imports this module
# lazily inside _metadata, so importing it here at module level does not close a cycle.
from . import authorobj as g17authorobj


class Unmeasured(g17authorobj.Missing):
    """Raised for a field this class has no rule for. Never caught to substitute a value."""


def plan(abi):
    """What can be placed, with the rule that decides each, or a refusal naming what cannot."""
    # BOTH VERSION GATES MUST AGREE. The author's allowlist admitted v7 while this one still
    # demanded exactly v6, so a v7 contract was refused here after being accepted there -
    # integration hit it by changing the fixture's version and getting a refusal from the wrong
    # place. v7 is v6 plus the preload block, so the texture class is stated for either.
    if abi.get("abi_version") not in TEXTURE_ABI_VERSIONS:
        raise ValueError("the texture class is stated for ABI %s; this contract is v%s"
                         % (" and ".join("v%d" % v for v in sorted(TEXTURE_ABI_VERSIONS)),
                            abi.get("abi_version")))
    resources = abi.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("a texture contract must state its resources; without them the user "
                         "bindings' ranks are unknown and a store ranks into a list that does "
                         "not exist")
    samplers = resources.get("samplers")
    if samplers is None:
        raise ValueError("the contract must state samplers explicitly, even as an empty list, so "
                         "a sampler is refused by name rather than ignored")
    if samplers:
        raise Unmeasured("this contract names %d sampler(s); no sampler combination is measured "
                         "for this class" % len(samplers))
    textures = resources.get("textures") or []
    if not textures:
        raise ValueError("a texture contract with no texture is not this class")
    for texture in textures:
        if texture.get("rank") is not None:
            raise ValueError("a texture carries no binding rank: the binding vector of every "
                             "measured witness holds buffers and internals only")

    internal = [(r["rank"], r["apple_index"]) for r in resources.get("internal") or []]
    users = [(b["offset"] // 2, b["index"]) for b in abi["bindings"]]
    order, ranks = g17resource.layout([i for _r, i in users], internal=[i for _r, i in internal])
    stated = dict(internal + users)
    if {r: i for r, i in enumerate(order)} != stated:
        raise ValueError("the contract's ranks %s are not the ordering law's %s"
                         % (stated, {r: i for r, i in enumerate(order)}))
    for binding in abi["bindings"]:
        if binding["offset"] != 2 * ranks[binding["index"]]:
            raise ValueError("binding %d states offset %d; the pointer-block law gives %d for "
                             "rank %d" % (binding["index"], binding["offset"],
                                          2 * ranks[binding["index"]], ranks[binding["index"]]))

    # SLOT 38 IS NOT A TEXTURE COUNT, and this side placed it as one until the family said
    # otherwise. M5 reads ONE texture twice at different coordinates and carries 16; M6 reads TWO
    # textures at the same coordinate and carries 8 - the opposite of 8 * textures in both
    # directions. What fits all eight read-path members is 8 * the number of distinct coordinate
    # slots the program publishes. That is an instruction fact, so it is the compiler's to state
    # and this refuses rather than counting textures again.
    # SLOT 1 IS NOT DERIVABLE FROM THE BINDING LIST, and the retained 368-byte section's rule for
    # it - the pointer block rounded up to eight - holds on 319 of 6,495 Apple texture sections,
    # 4.9%. It was read off three sections that all carry 8 with a block of 6, which is a
    # correlation certified on the witnesses that suggested it. Six candidate readings top out at
    # 22.9%. It is the argument-buffer total, a backend fact, and it is stated or the class refuses.
    argument_bytes = resources.get("argument_bytes")

    # THE EXTENT IS DERIVED HERE, FROM ADDRESS UNITS, and the count is not stated. Integration is
    # right that a slot count must not come from the compiler: the slot SIZE is this section's
    # rule, and a count would let two coordinate COMPONENTS become two coordinate PAIRS without
    # anyone noticing. The contract gives each publication's target offset and width; slot 38 is
    # eight times the number of distinct eight-byte slots those ranges touch.
    #
    #   frontier   (coord.x, 0, 4) and (coord.y, 8, 4)  ->  slots {0, 1}  ->  16
    #   M0         one publication at (0, 4)            ->  slots {0}     ->   8   measured 8
    #   M6         one publication, two textures        ->  slots {0}     ->   8   measured 8
    #   M5         two publications, two coordinates    ->  slots {0, 1}  ->  16   measured 16
    #
    # M5 and M6 refute the texture count in both directions but do not settle every layout, so the
    # rule is stated as a derivation over ranges rather than a count of anything.
    publications = resources.get("coordinate_publications")
    if not publications:
        raise Unmeasured(
            "the contract must state coordinate_publications in address units - each publish's "
            "form, target byte offset and width. Slot 38 is the storage EXTENT those cover, not a "
            "texture count and not a publication count: the family measured M5, one texture read "
            "twice at different coordinates, at 16, and M6, two textures read at the same "
            "coordinate, at 8. A count of publications would also have been wrong the moment two "
            "components of one coordinate became two publications, which is the frontier's shape.")
    # THE READING IS DECIDED, AND MINE WAS THE WRONG ONE. slot 38 is eight times the number of
    # distinct coordinate PAIRS the publications cover: a publication's const(k) is byte k, a
    # coordinate component is 16-bit, and (x, y) at 0 and 2 is ONE four-byte pair.
    #
    # I had authored the other factorisation - const(k) is byte 4k into eight-byte slots - which
    # gave the frontier 16. Both fit every witness compiled until then, because the x4 on the
    # constant cancelled the x2 on the unit size: M0 and M6 publish once and give 8 either way,
    # M5 publishes at const(0) and const(4) and gives 16 either way. The compiler compiled the
    # member that parts them, C0 - a 2D read whose y comes from a REGISTER, so Apple emits two
    # register publishes at const(0) and const(2), the frontier's shape exactly:
    #
    #     C0   two register publishes at const(0) and const(2)   slot 38 = 8   (pairs predicted 8,
    #                                                                           slots predicted 16)
    #     C1   the same kernel with a literal y, in the same run  slot 38 = 8   reproduces M0
    #
    # So the slots reading is deleted rather than deprecated, and the frontier's slot 38 is 8.
    # SCOPED TO THE MEASURED PUBLICATION OPERANDS, as integration required, because slot 38 = 8 on
    # C0 supports the pair reading for the operands that were compiled and nothing wider. The
    # measured set is op592 (a register publish, C0's pair and M0's single) and op556 (Apple's
    # CONSTANT publish for a literal y, C1's second). A publication outside it is refused rather
    # than folded into an extent this class has never seen it in.
    #
    # AND ONE DIFFERENCE STAYS UNEXPLAINED beside this rule. Apple's op592 publishes carry a
    # trailing imm:16 on every member including C0; this backend's carry imm:0. Ours were measured
    # to read the correct texel, so it is recorded rather than treated as a defect - but it sits
    # directly beside a result that turned on components being 16-bit, and integration has asked
    # the compiler to discriminate a width role from a lifetime role before anyone names it.
    COMPONENT_BYTES = 2
    covered = set()
    for entry in publications:
        if isinstance(entry, dict):
            opcode = entry.get("opcode")
            form = entry.get("form") or ""
            if opcode is not None and opcode not in MEASURED_PUBLICATION_OPCODES:
                raise Unmeasured(
                    "a coordinate publication uses op%s; the extent rule is measured only for %s "
                    "- op592's register publish, op556's constant publish and op10298's ALU "
                    "publish - and this class has never seen slot 38 with any other publication "
                    "operand"
                    % (opcode, sorted(MEASURED_PUBLICATION_OPCODES)))
            if form and form not in {"publish.coord.x", "publish.coord.y"}:
                raise Unmeasured("a coordinate publication names the form %r, which is not one of "
                                 "the measured coordinate publications" % form)
        if isinstance(entry, (list, tuple)):
            constant = entry[1]
        else:
            constant = entry.get("target_constant",
                                 entry.get("constant",
                                           entry.get("target_offset_bytes", entry.get("offset"))))
        if not isinstance(constant, int) or constant < 0:
            raise ValueError("a coordinate publication needs an integer constant: %r" % (entry,))
        covered.update(range(constant // COORDINATE_PAIR_BYTES,
                             (constant + COMPONENT_BYTES - 1) // COORDINATE_PAIR_BYTES + 1))
    slots = len(covered)

    registers = tuple(abi.get("system_registers") or ())
    entries = M.slot29_entries(registers) if registers else []
    placed = {
        "bindings": [{"index": i, "rank": r, "offset": 2 * r,
                      "written": any(b["index"] == i and b["written"] for b in abi["bindings"]),
                      "internal": i in dict(internal).values()}
                     for r, i in enumerate(order)],
        "slot 3": 8 * len(order),
        "slot 38": 8 * slots,
        "slot 42": 8 * slots,
        "slot 0": abi["register_count"],
        "slot 29": entries,
        "slot 13": list(abi.get("constant_pool") or []),
        "slot 31": None if not resources.get("spill_bytes") else resources["spill_bytes"],
        "slot 12": [],
    }
    rules = {
        "bindings": "internals first ascending then users ascending; offset = 2 * rank",
        "slot 3": "8 * binding count",
        "slot 38": "8 * the distinct %d-byte slots the coordinate publications cover (%d), "
                   "derived here from address units" % (COORDINATE_SLOT_BYTES, slots),
        "slot 42": "equal to slot 38 - exact on all %d Apple texture sections and all nine family "
                   "members" % CORPUS["sections"],
        "slot 0": "the contract's register_count",
        "slot 29": "the contract's system_registers through the shared slot-29 map",
        "slot 13": "the contract's constant_pool",
        "slot 31": "the thread-invariant spill; absent when the backend reports none (%s)"
                   % resources.get("spill_basis", "not stated"),
        "slot 12": "present and empty",
    }
    # THE BINDING RECORDS' VECTOR ORDER IS NOT THEIR RANK ORDER, and integration's dispatch after
    # f1f8b808 is explicit that the two must not be silently equated: the retained M1 witness
    # orders its records [44, 0, 1, 48] while its ranks are [44, 48, 0, 1]. Measured at
    # results/g17-texture-record-order-v1: inside this class - internals exactly {44, 48} - the
    # records are 44, then the user indices ascending, then 48, on 294 of 294 sections, and NONE
    # of those 294 has the two orders coincide. Outside the class the same reading holds on 5,041
    # of 5,367 and the 326 it misses put internals 35, 36, 37 - and sometimes 44 - after the users,
    # so they are a different class and stay refused with the rest of the out-of-domain shapes.
    record_internals = {i for i in order if i in dict(internal).values()}
    if record_internals == SUPPORTED_INTERNALS:
        users_ascending = sorted(i for i in order if i not in record_internals)
        placed["binding record order"] = [44] + users_ascending + [48]
        rules["binding record order"] = (
            "44, then the user indices ascending, then 48 - measured on 294 of 294 sections whose "
            "internals are exactly {44, 48} (%s), and NOT the binding rank order, which differs on "
            "every one of them" % RECORD_ORDER)
    # THE PRELOAD'S CONSUMER CONSTANT IS CHECKED, NEVER TAKEN. The contract restates the constant
    # main's op10282 reads; the block layout says what it must be - 4 x the DECLARED binding count,
    # exact on S1 at 16, S2 at 20, S3 at 24, M1 at 16 and M4 at 12 - and a disagreement means one
    # side has the block wrong, which is worth a refusal rather than a section.
    check_resolved(abi)
    for entry in resources.get("preloads") or ():
        stated = (entry.get("consumer") or {}).get("block_constant")
        expected = RECORD_BYTES * len(order)
        if stated is not None and stated != expected:
            raise Unmeasured(
                "a preload's consumer reads block constant %s; this program declares %d records, "
                "so the block is %d bytes and the preload publishes at %d. The contract does not "
                "carry the offset by agreement - it carries what the code reads, and the two "
                "disagree, which means the binding list or the code is wrong rather than the form"
                % (stated, len(order), expected, expected))
        placed["preload at"] = expected
        rules["preload at"] = ("%d x the DECLARED binding count - exact on S1, S2, S3, M1 and M4, "
                               "including the records the code never addresses" % RECORD_BYTES)


    # THE KIND-6 RECORD IS NOT UNCONDITIONAL, and this side planned it as if it were. M2 and M8
    # carry NO kind-6 record while the other seven members do, and what separates them is whether
    # every user binding is accounted for: M2 writes a second user buffer, M8 reads one at a
    # divergent index, and in both the binding is neither the single output nor in slot 27. The
    # seven that carry it have every user binding either written as the output or uniformly read.
    #
    # AND THAT RULE IS NOW THE NARROWER OF TWO. Resolving lane31's class turned up a wider one that
    # decides the same question from a field every section carries: SLOT 2 HOLDS A POOL RECORD
    # EXACTLY WHEN SLOT 13'S VECTOR IS NOT ZERO-LENGTH - kind-6 ordinarily, kind-9 in the swap
    # regime. Over Apple's 9,547 sections and this checkout's 118 committed witnesses:
    #
    #     zero-length slot-13 vector    582 + 46   no pool record
    #     all-zero pool bytes         1,593 + 53   kind-6
    #     pool with content           7,124 + 13   kind-6   (+248 also kind-9; +6 kind-9 alone)
    #
    # 9,665 sections, no exception. It gives the same answers on the nine family members - M2 and
    # M8 have zero-length pools, which is WHY they carry no record - so it subsumes the binding
    # rule rather than competing with it, and unlike that rule it is checkable outside the nine:
    # no census section states which of its bindings are read uniformly, so the binding rule could
    # never have been scored anywhere but on the witnesses that suggested it.
    #
    # AN ALL-ZERO POOL IS NOT A ZERO-LENGTH ONE. 1,593 Apple sections carry a pool of nothing but
    # zero bytes AND a kind-6 record, which is the distinction POOL_EMPTY_BYTES above already
    # names: length is not emptiness. So a contract whose constant_pool is empty does not yet say
    # which of the two it means, and the two differ on whether this section has a pool record at
    # all. That is refused by name below rather than decided here - and it is the fact that makes
    # this side's own placement inconsistent today, because it placed slot 13 as a zero-length
    # vector and a kind-6 record beside it, which the law says cannot both be true.
    pool = abi.get("constant_pool")
    # A REFUSAL MUST NAME A FACT THE CONTRACT CAN ANSWER, and for one commit this one did not.
    # `constant_pool_emptiness` was named as missing with no field anywhere for the compiler to put
    # the answer in, so a correct answer had nowhere to go and the refusal could not be lifted by
    # supplying it. That is worse than a fact being open: it is a demand with no accepting form.
    # The two values are the two states measured at results/g17-texture-class-v1.
    POOL_FORMS = {"zero_length_vector": [], "eight_byte_zero_vector": [0] * POOL_EMPTY_BYTES}
    stated_form = resources.get("constant_pool_emptiness")
    if stated_form is not None:
        if stated_form not in POOL_FORMS:
            raise ValueError("constant_pool_emptiness is %r; the measured states are %s"
                             % (stated_form, " / ".join(sorted(POOL_FORMS))))
        if pool:
            raise ValueError("constant_pool_emptiness states which EMPTY pool this is, and this "
                             "contract's pool is not empty (%d bytes)" % len(pool))
        pool = POOL_FORMS[stated_form]
        # AND THE RESOLVED FORM MUST REACH THE PLACED VECTOR. `placed["slot 13"]` is built from the
        # contract's raw constant_pool, which is [] for BOTH empty states - so stating
        # eight_byte_zero_vector resolved the derived record facts and left the vector this side
        # would emit at zero length. The plan then read as fully determined while the bytes it
        # planned differed from the witness in the eight bytes the statement was about. Found by
        # checking a placed VALUE rather than trusting an empty not_determined list.
        placed["slot 13"] = list(pool)
        rules["slot 13"] = ("the contract's constant_pool, with an empty one resolved to the "
                            "stated constant_pool_emptiness form")
    pool_ambiguous = stated_form is None and pool is not None and len(pool) == 0
    # AND THE KIND-9 RECORD IS DERIVABLE WHERE THE POOL HAS NO CONTENT. A section with no non-zero
    # constant-pool content carries NO kind-9 record: 0 of 2,274 across Apple's census and this
    # checkout's committed witnesses. One direction only - content does NOT imply a record, 254 of
    # 7,391 - and that direction stays open, which is the half no empty-pool contract needs.
    #
    # It survives constant_pool_emptiness being unresolved: BOTH readings of an empty pool, the
    # eight-byte zero vector and the zero-length one, have no non-zero content, so the rule gives
    # the same answer either way.
    #
    # THE CONTRACT'S not_stated LIST STILL COMES FIRST and this does not lift it. The derivation is
    # recorded in `placed` so that removing slot2_kind9_record from a contract's unknowns is ONE
    # decision for integration rather than one more measurement - the same treatment slot 27 gets,
    # and for the same reason: an author that quietly overrides a stated unknown because it happens
    # to have a rule is doing the thing this route exists to refuse.
    if pool is not None and not any(pool):
        placed["slot-2 kind-9 record"] = "absent"
        rules["slot-2 kind-9 record"] = (
            "no non-zero constant-pool content, and no such section carries a kind-9 record on "
            "2,274 across two corpora; sections WITH content do carry them, 254 of 7,391, so the "
            "rule is a constraint rather than a corpus with none in it (%s). Both readings of an "
            "empty pool have no content, so this holds while constant_pool_emptiness is open"
            % CLASS)
    users = [a for a in (resources.get("access") or ()) if a.get("kind") == "user"]
    # THE POOL IS READ BEFORE ANY FIELD THAT DEPENDS ON IT. It used to be read AFTER the kind-6
    # field2 derivation that needs it, so `nine_absent` short-circuited on an unset placement and
    # field2 was silently not derived - no error, just a field quietly missing from every plan.

    # FIELD2 IS DERIVABLE AFTER ALL, and the distinction integration asked for is a fact the
    # section carries: a kind-9 record in the slot-2 vector.
    #
    #   field2 == constant-pool length / 4, split on the kind-9 record
    #       no kind-9, texture       6,235 of 6,235   100.00%
    #       no kind-9, non-texture   2,487 of 2,487   100.00%
    #       kind-9 present, texture      0 of    95     0.00%
    #       kind-9 present, other        0 of   153     0.00%
    #
    # Perfect separation on 8,970 sections. The 95 texture exceptions integration named ARE the
    # kind-9 ones, and the three small-pool counterexamples it sent me after - two
    # PTSpillCorrection kernels and brnetv3_flow_splat - are the members that gave it up: each
    # carries [9, 6, 3...] where its passing siblings carry [6, 3...].
    #
    # I withdrew this rule when it had 95 exceptions and no checkable distinction. This is the
    # distinction, so it comes back scoped - derived only when the contract states there is no
    # kind-9 record, and refused when it states one or says nothing.
    # STATED FALSE, OR DERIVED ABSENT BY THE CONTENT RULE. The precondition was always "there is no
    # kind-9 record"; requiring the CONTRACT to say so was a proxy for it, adopted when this side
    # had no rule of its own. It now has one - no non-zero pool content implies no kind-9 record,
    # 0 of 2,274 across two corpora - so the derived absence serves the same precondition. A
    # contract that STATES a record present still wins, because a stated fact beats a derived one.
    nine_absent = (resources.get("slot2_kind9_record") is False
                   or (resources.get("slot2_kind9_record") is not True
                       and placed.get("slot-2 kind-9 record") == "absent"))
    # AND ONLY WHEN THE POOL'S FORM IS KNOWN. `len(pool) or POOL_EMPTY_BYTES` defaulted an empty
    # pool to eight bytes and placed field2 = 2 for a contract whose pool form is refused - the
    # exact guess this route exists to refuse, arriving through an `or`. With the form unknown
    # there is not even a record to carry the field.
    if nine_absent and pool and not pool_ambiguous:
        placed["slot-2 kind-6 field2"] = len(pool) // 4
        rules["slot-2 kind-6 field2"] = (
            "the constant-pool length over four; `field2 + field4 == pool/4` is exact on 24,984 of "
            "24,984 sections over two corpora (%s) and field4 is elided in exactly the sections "
            "carrying no kind-9 record, which is why the scoped form holds on all 8,717 of those "
            "and on none of the 248 that do" % KIND6_SPLIT)
    # THE TWO RULES ARE TRIED IN ORDER OF WHAT THEY WERE SCORED ON, and the fields are placed once
    # for whichever decided it. Keeping the field placement inside one rule's branch is how the
    # wider rule silently stopped placing them - caught by the regression case, which asked for a
    # field the new branch never wrote.
    accounted = None
    # THE PER-KERNEL SLOTS THIS SIDE CAN PLACE FROM A MEASUREMENT, and the two it cannot. The
    # coverage check at results/g17-texture-class-v1 walks every occupied slot of the per-kernel
    # table and requires each to be compared, measured constant, or NAMED. It found four constants
    # and two fields nobody here places:
    #
    #   slots 6, 8, 12   empty on all 6,495 Apple texture sections and all 131 witnesses
    #   slot 26          [16] on all of them
    #   slot 10          empty on every witness, and on 5,382 of 6,495 Apple sections - 1,113 carry
    #                    content, so "empty" is a property of our simple programs, not a rule
    #   slot 32          absent on 129 of 131 witnesses, PRESENT on 5,820 of 6,495 Apple sections
    #
    # The last two are named rather than defaulted. An emitted section must decide them, and
    # "absent on our own compiles" is the population argument this route exists to refuse.
    for slot, value in ((6, []), (8, []), (12, []), (26, [16])):
        placed["slot %d" % slot] = value
        rules["slot %d" % slot] = ("the same on all 6,495 Apple texture sections and all 131 "
                                   "committed witnesses (%s)" % CLASS)
    if pool_ambiguous:
        # The form is unknown, so whether there is a record at all is unknown. Neither branch.
        pass
    elif pool is not None:
        # THE LAW DECIDES BOTH WAYS. A non-empty pool carries a record; a ZERO-LENGTH one carries
        # none. The first version only handled the non-empty half and let an empty pool fall
        # through to the binding rule, which answered "present" for a section the law says has no
        # record at all - the wider rule overruled by the narrower one it replaced.
        accounted = bool(pool)
        rules["slot-2 kind-6 record"] = (
            "a pool record accompanies a section whose slot-13 vector is not zero-length, and no "
            "section with a zero-length one carries any - 9,665 across two corpora with no "
            "exception (%s). This pool is %s, so the record is %s"
            % (CLASS, "not zero-length" if pool else "zero-length",
               "present" if pool else "absent"))
    elif users and not pool_ambiguous and all(
            a.get("uniform") is not None or a.get("written") for a in users):
        accounted = all(a.get("written") or a.get("uniform") for a in users)
        rules["slot-2 kind-6 record"] = (
            "present exactly when every user binding is the single output or is uniformly read - "
            "measured on all nine family members, where M2 (a second written user) and M8 (a "
            "divergent read) are the two that carry none. The pool-record law gives the same "
            "answers on those nine and is checkable outside them (%s)" % CLASS)
    if accounted is not None:
        placed["slot-2 kind-6 record"] = "present" if accounted else "absent"
        if accounted:
            # AND THE SPLIT CLOSES HERE, because slot 1 is now an INPUT. `field2 + field3 == slot 1`
            # is exact on 8,965 of 8,965 census sections carrying the record - 4,781 on the
            # preregistered stated half and 4,184 on the held-out half, with zero exceptions - and
            # the competing readings of field3 a reader has in hand reach 219 of 4,781 at best.
            # Slot 1 was never available to this side before: `8 + pool/4` held on 154 of 5,685 and
            # `8 + 4*users_in_27` on 486, so both were recorded refuted rather than adopted. The
            # compiler states it as resources.argument_bytes and validates it against its own
            # access facts, so field3 = slot1 - field2 is read off two stated facts.
            placed["slot-2 kind-6 fields"] = (
                "field2 + field3 = slot 1 = %s%s"
                % (argument_bytes if argument_bytes is not None else "NOT STATED",
                   "" if "slot-2 kind-6 field2" not in placed else
                   "; field2 = %d derived, so field3 = %s"
                   % (placed["slot-2 kind-6 field2"],
                      (argument_bytes - placed["slot-2 kind-6 field2"])
                      if argument_bytes is not None else "follows once slot 1 is stated")))
            rules["slot-2 kind-6 fields"] = (
                "the SUM is exact on 8,965 of 8,965 census sections carrying the record, on a "
                "preregistered even/odd split of the census files that was fixed before the sweep "
                "ran (%s)" % KIND6_CLOSURE)
            if "slot-2 kind-6 field2" in placed and argument_bytes is not None:
                # BOTH FIELDS, FROM THE POOL AND SLOT 1 AND NO KIND-6 VALUE READ. Scored the way
                # the author runs it - derive, then compare - on 8,717 of 8,717 Apple census
                # sections carrying no kind-9 record and 15,970 of 15,970 in this project's own
                # compiled cache, two populations built at different times. The 248 and 61 that
                # carry a kind-9 record are refused, not smoothed: field4 is free there, and two
                # identities in three unknowns determine nothing.
                placed["slot-2 kind-6 field3"] = argument_bytes - placed["slot-2 kind-6 field2"]
                rules["slot-2 kind-6 field3"] = (
                    "slot 1 minus field2, the two identities composed; the pair is re-derived "
                    "from the pool and slot 1 alone on 8,717 of 8,717 Apple sections and 15,970 "
                    "of 15,970 of this project's own, and refused on the 309 carrying a kind-9 "
                    "record (%s)" % KIND6_CLOSURE)
    # SLOT 27 IS DERIVABLE ONCE THE CONTRACT SAYS WHICH READS ARE UNIFORM. The family decided the
    # rule - the resources read at an index constant across the grid, with 44 removed - so what is
    # missing is no longer a rule but one access fact per record. M8 is why it must be `uniform`
    # and not `read`: buffer 1 read at gid.x is read, and is NOT in slot 27.
    access = resources.get("access") or ()
    users_access = [a for a in access if a.get("kind") == "user"]
    # THE INTERNALS' UNIFORMITY IS THIS SIDE'S FACT, NOT THE COMPILER'S, and integration is right
    # that it must not be invented there. Measured over all nine family members: 48 is in slot 27
    # every time and 44 never is. 165 of the 6,495 Apple texture sections DO carry 44, so those are
    # outside this class's domain and are named rather than explained away.
    # THE SUPPORTED DOMAIN, stated before the unknown is lifted, as integration required. H5 holds
    # on the nine family members and is NOT established over the 6,495, so slot 27 is derived only
    # for programs inside the shape those nine share, and refused outside it. The two named
    # exclusions are the reason the domain exists: 165 Apple texture sections carry internal 44
    # inside slot 27 and nothing predicts when, and M7's sampled path publishes through other
    # opcodes and breaks the slot-38 rule as well.
    internals = {r["apple_index"] for r in resources.get("internal") or ()}
    written = [a for a in users_access if a.get("written")]
    in_domain = (internals == SUPPORTED_INTERNALS and not samplers and len(written) == 1
                 and all(a.get("uniform") is not None or a.get("written") for a in users_access))
    placed["supported domain"] = "yes" if in_domain else "no"
    rules["supported domain"] = (
        "internals exactly %s, no sampler, exactly one written user binding, and every other user "
        "binding's uniformity stated - the shape the nine family members share. Outside it slot 27 "
        "is refused, because H5 is measured on those nine and not over the 6,495."
        % sorted(SUPPORTED_INTERNALS))
    if in_domain:
        placed["slot 27"] = sorted(((internals & UNIFORM_INTERNALS)
                                    | {a["record"] for a in users_access
                                       if a.get("uniform") and not a.get("written")})
                                   - {UNPREDICTED_INTERNAL})
        rules["slot 27"] = ("the uniformly read resources with %d removed - the user half decided "
                            "by M8, where the same buffer read at a divergent index leaves slot 27; "
                            "the internal half measured here, 48 present on all nine members and 44 "
                            "on none" % UNPREDICTED_INTERNAL)

    # AND THE RESOURCE RECORD, the same way: the family measured it ABSENT on all nine members,
    # including the two-texture and the sampled one, but 3,959 of 6,495 Apple texture sections
    # carry one, so "absent" is a fact about these programs and not a law. A contract that states
    # it is authored; one that does not is refused, because picking the family's answer for a
    # program outside the family is the donor move in a different coat.
    record = resources.get("slot2_resource_record")
    if record is not None:
        placed["slot-2 kind-5 record"] = "absent" if record is False else record
        rules["slot-2 kind-5 record"] = ("stated by the contract; measured absent on all nine "
                                         "family members and present in %d of %d Apple sections, "
                                         "so it is stated rather than derived"
                                         % (CORPUS["sections"] - CORPUS["no_resource_record"],
                                            CORPUS["sections"]))

    # THE CONTRACT'S not_stated LIST IS AUTHORITATIVE AND COMES FIRST. Deriving slot 27 inside the
    # supported domain does not entitle this side to lift the contract's own unknown: integration
    # decided that slot27_contents stays unknown until the domain is accepted, and an author that
    # quietly overrides a stated unknown because it happens to have a rule is doing the thing this
    # whole route exists to refuse. The derivation is still recorded in `placed`, so lifting it is
    # one decision rather than one more measurement.
    missing_extra = []
    # THE TWO THE COVERAGE CHECK FOUND. An emitted section must decide them and "absent on our own
    # compiles" is the population argument this route exists to refuse - slot 32 is absent on 129
    # of our 131 and PRESENT on 5,820 of Apple's 6,495.
    # SLOT 10 IS DERIVABLE WHEN THE CONTRACT DECLARES NO DYNAMIC THREADGROUP BUFFER, and only
    # then. TWO LINKS, measured separately, because the first is on four compiles and the second on
    # a corpus:
    #
    #   contract declares a dynamic threadgroup  ->  the section carries slot 9
    #       measured at results/g17-slot10-witness-v1 by compiling both arms of both axes - a
    #       texture read and a device-buffer read, each with and without a dynamic threadgroup.
    #       Both threadgroup arms carry slot 9 and a kind-43 slot-10 record; neither other arm
    #       carries either, so the record tracks the THREADGROUP declaration and not the texture.
    #
    #   the section carries slot 9  <->  slot 10 is non-empty
    #       1,416 of 1,416 and 8,131 of 8,131 over Apple's compute sections, no exception in
    #       either direction. This half is a biconditional on a corpus nobody here compiled, which
    #       is what lets the derivation apply to a contract rather than only to our own compiles.
    #
    # Deriving the EMPTY case needs only the absence. Deriving a NON-empty one would need the rule
    # for what the vector holds, which is not measured, so a contract declaring a dynamic
    # threadgroup still refuses.
    dynamic_threadgroup = bool(abi.get("uses_threadgroup")) or bool(
        (abi.get("threadgroup") or {}).get("dynamic_memory") if isinstance(
            abi.get("threadgroup"), dict) else False)
    if not dynamic_threadgroup:
        placed["slot 10"] = {"records": [], "because": (
            "the contract declares no dynamic threadgroup buffer, and slot 10 tracks that "
            "declaration: both arms of the texture axis carry a kind-43 record when one is "
            "declared and neither carries anything when it is not (%s)" % SLOT10_WITNESS)}
        rules["slot 10"] = ("empty exactly when no dynamic threadgroup buffer is declared, "
                            "measured on four compiled arms of two axes")
    else:
        missing_extra.append(
        "slot10_contents [a four-byte pointer to a VECTOR OF KIND-TAGGED RECORDS - decoded here: "
            "6899 records over 1113 sections; every one walks as a table; a one-byte kind at slot "
            "0 taking 93 on 6563; 43 on 336; optional four-byte fields at slots 1 plus 2. This "
            "contract DECLARES a dynamic threadgroup buffer - the one shape measured to produce a "
            "record - so the vector is NOT empty here. The rule for what a non-empty vector HOLDS "
            "is not measured; it is refused rather than filled from the one kind this side can "
            "compile]")
    # SLOT 32'S ABSENCE IS DERIVABLE FROM MAIN'S INSTRUCTION COUNT, and its VALUE is not.
    # g17mdgen.slot32_for is measured at the boundary - 30 instructions carry no slot 32 and 31
    # carries 1 - and four freshly compiled kernels confirm the absence arm on this class: a
    # texture read and a device-buffer read, with and without a dynamic threadgroup, main counts
    # 6 to 9, none carrying slot 32. THE COUNT IS MAIN'S, NOT THE WHOLE TEXT: counting the 64-byte
    # prologue as well gives 34 to 37 on those same four and predicts 1, which is how this side
    # nearly published the law as refuted. Only the ABSENT arm is derived here; a program whose
    # main exceeds the boundary needs the value, which no readable fact predicts.
    main_instructions = abi.get("main_instruction_count")
    if main_instructions is not None and M.slot32_for(main_instructions,
                                                      bool(abi.get("has_back_edge"))) is None:
        placed["slot 32"] = {"value": None, "because": (
            "main is %d instructions and the measured boundary puts no slot 32 below 31; "
            "confirmed on four compiled arms of this class at 6 to 9" % main_instructions)}
        rules["slot 32"] = ("absent when main's instruction count is at or below the measured "
                            "boundary. The VALUE above it is not derived and stays refused.")
    else:
        missing_extra.append(
        "slot32_value [a ONE-BYTE field taking exactly THREE values across the 5820 Apple texture "
            "sections that declare it - 1 on 2745; 2 on 676; 3 on 2399 - absent on 675. Its "
            "ABSENCE is derivable from main's instruction count; this contract does not put main "
            "below the measured boundary so the VALUE is needed. No readable section fact predicts "
            "which: the best single predictor reaches 0.662; a joint one reaches 0.794 over 237 "
            "keys]")
    # WHAT THE RECORDS THEMSELVES HOLD, measured inside this class at results/g17-texture-record-
    # fields-v1 and split three ways rather than refused as one thing:
    #
    #   internal 44's record   one field set at one declared length on 294 of 294 - writable
    #   internal 48's record   one field set on 294 of 294 and TWO declared lengths, 16 or 18 -
    #                          its contents are settled and its length is not
    #   a user record          55 distinct field sets for index 1 alone across the same 294, so
    #                          what it holds is a property of the buffer and not of the class
    #
    # The emitter is refused on the last two by name. Picking 48's commoner length, or the
    # commonest user field set, would be picking one program's description for another program's.
    if "binding record order" in placed:
        placed["binding record 44"] = "kind 5, index 44, no other field, 12 bytes"
        rules["binding record 44"] = ("one field set at one declared length on 294 of 294 in-class "
                                      "sections (%s); the length itself is the distance to the "
                                      "next structure (%s)" % (RECORD_FIELDS, RECORD_LENGTH))
        # NEITHER A COMMA NOR AN " and " IN A FACT NAME OR ITS REASON. g17textureblockers reads
        # the refusal's fact list by splitting on the first comma and then on " and " - so a comma
        # truncates the list and an " and " splits one fact into two. Both have now bitten this
        # message once each.
        # NO LONGER A CHOICE BETWEEN TWO VALUES. The record's contents are one field set at one set
        # of offsets on all 294, ending at byte 16, and the two declared lengths are not two
        # contents: a binding record's declared table length is the DISTANCE TO THE NEXT STRUCTURE
        # in the section - 43,132 Apple binding records and 279 committed ones with no exception,
        # while the per-kernel table satisfies it on none of its 6,495 and other tables on 23,870
        # of 37,700, so the rule is scoped by measurement rather than by assertion.
        #
        # 16 is the record plus two bytes of padding before a 12-byte vtable; 18 is the record
        # absorbing that padding before a 10-byte one. Both reach the same 28. It is the placer's
        # own law read from the other end - alignment belongs to the TABLE.
        #
        # So this is downstream of the placement rather than unmeasured, and it stays named for the
        # same reason the kind-6 split does: a fact that vanishes from a refusal reads exactly like
        # a fact that was settled. It closes when there is a section to lay out, which needs the
        # facts above it.
        # CLOSED, AND BY THE PLACEMENT LAW RATHER THAN AROUND IT.  The declared length is the
        # distance to the next structure - 43,411 binding records over two corpora with no
        # exception - and for internal 48 that distance is decided by ONE contract fact.
        #
        # Measured on the 256 in-class sections whose 48 record is followed by a vtable:
        #
        #     48's length + the following vtable's vlen == 28   on 256 of 256
        #     length 16  <->  the following record is WRITTEN     173
        #     length 18  <->  the following record is READ-ONLY    83
        #
        # perfect separation, both cells populated.  And on the 56 sections carrying lane31's EXACT
        # record set {44, 0, 1, 48}, the record after 48 is always index 1:
        #
        #     index 1 read-only  ->  18   on 39 of 39
        #     index 1 written    ->  16   on 17 of 17     <- the negative control
        #
        # So the length follows from `written` on the binding that sits after 48, which the contract
        # states.  It was refused as "downstream of a placement"; the placement turns out to need
        # one fact rather than the whole section.
        # WHICH RECORD FOLLOWS 48 IS MEASURED, NOT GUESSED. A first version took the lowest user
        # binding and would have published 16 where the corpus says 18. The physical layout is NOT
        # simply the reverse of the vector order - that holds on 113 of 343 - but the successor of
        # 48 is the LAST USER IN THE VECTOR ORDER on 343 of 343, which is the only claim needed.
        # AND THE RULE IS SCOPED TO A SUCCESSOR THAT IS A USER RECORD, which is what the in-class
        # population contains: all 343 have users, and where a vtable follows 48 the successor is a
        # user on 256 of 256. Outside that scope the sum is not even 28 - among the 1,787 sections
        # carrying 48 with NO user binding, 42 have a READ-ONLY successor (internal 44, whose record
        # declares two slots and so has an 8-byte vtable) and still declare 16, because 16 + 8 = 24.
        # The distance law holds there; "written means 16" does not. Returning None with no users is
        # therefore a refusal rather than an oversight, and it is why this places nothing for a
        # contract whose successor would be an internal.
        users_in_order = [i for i in order if i < UNPREDICTED_INTERNAL]
        after48 = users_in_order[-1] if users_in_order else None
        written_after = None
        if after48 is not None:
            written_after = any(b["index"] == after48 and b["written"] for b in abi["bindings"])
        if written_after is None:
            # AND THE REFUSAL IS NAMED RATHER THAN LEFT AS A SILENCE. Placing nothing here was
            # right - the rule is scoped to a USER successor and 42 sections with an internal
            # successor refute it - but the fact then appeared in neither `placed` nor
            # `not_determined`, so a reader asking what this contract leaves open was told
            # nothing at all. A refusal that no one can find is not a refusal.
            missing_extra.append(
                "binding_record_48_declared_length [the record after internal 48 is the LAST USER "
                "in the vector order and this contract declares no user binding, so the successor "
                "is an internal. The length rule is measured only where a USER follows: among the "
                "1787 sections carrying 48 with no user binding, 42 have a read-only successor and "
                "still declare 16, because that successor is internal 44 whose vtable is 8 rather "
                "than 10. Outside that scope this places nothing]")
        if written_after is not None:
            placed["binding record 48"] = {
                "declared length": 16 if written_after else 18,
                "because the record after it": "index %s %s" % (
                    after48, "is WRITTEN" if written_after else "is read-only")}
            rules["binding record 48"] = (
                "the declared length is the distance to the next structure (%s); for internal 48 "
                "that distance is 28 minus the following vtable's length, and the following "
                "record's vtable is 12 when it is written and 10 when it is not - 256 of 256 "
                "in-class sections, and 39 of 39 against 17 of 17 on lane31's exact record set. "
                "The record that follows 48 is the LAST USER in the vector order, on 343 of 343. "
                "Scoped to a USER successor: among the 1787 sections carrying 48 with no user "
                "binding, 42 have a read-only successor and still declare 16, because that "
                "successor is internal 44 whose vtable is 8 rather than 10. PRECONDITION, measured "
                "and previously unstated: this is a rule about the FOLLOWING VTABLE's length, so it "
                "needs a vtable to follow 48. In the in-class population one does on 203 of 241 "
                "and the rule holds on all 203; on the other 38 a TABLE follows and the declared "
                "length is 16 on all 38 - including 25 with a READ-ONLY successor, where this rule "
                "says 18. Which case a composed section falls into is a layout fact this side's "
                "placer decides, not a contract fact, and it is NOT established here"
                % RECORD_LENGTH)
            placed["binding record 48"]["precondition"] = (
                "a vtable must follow 48. Where a TABLE follows instead, Apple declares 16 on 38 "
                "of 38 regardless of the successor - so this length is right only under a "
                "placement this side has not established")
        # AND WHAT EACH RECORD HOLDS, closed on the compiler owner's access family at
        # results/g17-texture-binding-record-v1: six members varying read-only uniform, read-only
        # divergent, written-not-read, read-and-written, written divergently and
        # declared-never-touched, all six predicted exactly from two facts the contract already
        # states. field 2 is the binding's OFFSET, elided at zero; field 3 is 1 when the binding
        # is WRITTEN and elided when it is only read; and a declared-unused binding has NO RECORD,
        # which is an elimination rather than a field change - Apple drops it, measured on V7.
        #
        # Field 4 is not settled by that family and no member carries one. It appears on 95 of the
        # 744 in-class user records, so a contract whose records would need one is outside this.
        written_indices = {b["index"] for b in abi["bindings"] if b["written"]}
        placed["binding records"] = [
            {"index": index, "field2 (offset)": 2 * rank or None,
             "field3 (written)": 1 if index in written_indices else None}
            for rank, index in enumerate(order)]
        rules["binding records"] = (
            "field 2 is the binding's offset elided at zero and field 3 is 1 when it is written; "
            "both are contract facts, predicted exactly on all six members of the access family "
            "(%s). A declared-unused binding has no record at all." % BINDING_RECORD)
    # THE SPLIT IS MISSING ONLY WHEN A FIELD IS. field2 needs the kind-9 record stated absent -
    # with one present, field4 is free and the pool identity gives nothing - and field3 needs
    # slot 1, which is the compiler's argument_bytes. Either absent and the split is unknown, and
    # the refusal below names which one, because "the split" alone sent the last reader to the
    # wrong half of the question.
    if "slot-2 kind-6 fields" in placed and "slot-2 kind-6 field3" not in placed:
        if "slot-2 kind-6 field2" not in placed:
            missing_extra.append("slot2_kind6_split [field2 - the kind-9 record is not stated "
                                 "absent so field4 is free]")
        else:
            missing_extra.append("slot2_kind6_split [field3 - slot 1 is not stated]")
    # THE NEW FACT, and the reason this delivery makes the refusal stronger rather than shorter.
    # With an empty constant_pool the contract has not said which of the two empty states it means,
    # and under the pool-record law they differ on whether slot 2 carries a record AT ALL - so the
    # kind-6 record, its split and the kind-9 record all hang off it. Named here so a reader gets
    # the one fact that unblocks four rather than four that look independent.
    #
    # NO COMMA AND NO " and " IN A FACT NAME: g17textureblockers splits the refusal's fact list on
    # the first comma and then on " and ", so either would truncate the list or split one fact in
    # two. Both have bitten this message once each already.
    if pool_ambiguous:
        missing_extra.append(
            "constant_pool_emptiness [a program with no constants is emitted two ways - a "
            "zero-length slot-13 vector on 582 Apple sections with no pool record; an eight-byte "
            "zero vector on 1593 with a kind-6 one. This is an authoring DECISION rather than a "
            "compiler fact: the backend states an empty constant_pool so it has chosen neither. "
            "State resources.constant_pool_emptiness as zero_length_vector or "
            "eight_byte_zero_vector]")
        # AND THE SPLIT IS STILL REFUSED - said out loud, because it is on integration's list of
        # five and it would otherwise VANISH from this message rather than being closed. It is not
        # closed; it is downstream. There is no split to state until it is known whether there is a
        # record, and a fact that disappears from a refusal reads to a reader exactly like a fact
        # that was settled.
        if "slot-2 kind-6 fields" not in placed:
            missing_extra.append(
                "slot2_kind6_split [downstream of constant_pool_emptiness - not closed; with the "
                "pool state unknown it is unknown whether this section carries a kind-6 record to "
                "split]")
    if argument_bytes is None:
        missing_extra.insert(0, "argument_bytes")
    # OUTSIDE THE DOMAIN NOBODY CAN SUPPLY IT, so it is named here even when the contract did not
    # list it. Inside the domain the contract's list decides; outside it, the rule does not apply
    # and the fact is unknown no matter what any contract says.
    if placed.get("supported domain") == "no" and "slot 27" not in placed:
        missing_extra.append("slot27_contents")
    if argument_bytes is not None:
        placed["slot 1"] = argument_bytes
        rules["slot 1"] = ("the contract's argument_bytes; no candidate derives it from the "
                           "binding list above 22.9% over 6,495 texture sections")
    # AND THE DERIVATION IS CHECKED AGAINST THE CONTRACT'S OWN SPILL, which it never was. A
    # per-kernel slot 27 of two or more entries implies a thread-invariant spill in slot 31, exact
    # on 4,545 Apple texture sections and 40 committed witnesses with ZERO exceptions, and not
    # vacuous - 463 Apple sections and 3 witnesses carry a spill with a shorter slot 27, so the
    # converse fails and the population can express what the rule forbids. Measured at
    # results/g17-texture-class-v1.
    #
    # H5 derives slot 27 = [1, 48] for lane31, length two, while the contract reports no spill. No
    # section in 6,626 has that pair. Rather than emit a derivation the corpus says cannot occur,
    # this names the contradiction and refuses - a derived value disagreeing with a stated one is
    # exactly the case where authoring anything means picking which of them to believe.
    # AND THE REASON IS SCOPE, NOT A WRONG MEASUREMENT. The compiler side settled this: its
    # allocator CANNOT spill - g17cc.Alloc has no spill form and RAISES rather than emitting a
    # store it cannot justify - so for any program it compiled, spill_bytes 0 is exact rather than
    # an unfilled field. Both numbers are right and they describe DIFFERENT PRODUCERS. Apple's
    # compiler spills where two resources are uniformly read; ours refuses, and the program that
    # survives is a different program. lane31 being absent from 6,626 sections is what that
    # predicts, not a gap in the sample.
    #
    # The fact stays named because the AUTHOR must still emit one section and the two rules
    # disagree about it - but the resolution is a decision about whether a section may describe a
    # program Apple would not have produced, which is a question about the format's scope and not
    # a measurement anybody can take.
    derived27 = placed.get("slot 27")
    if derived27 is not None and len(derived27) >= 2 and placed.get("slot 31") is None:
        missing_extra.append(
            "slot27_is_outside_the_population_the_rule_was_measured_on [a slot 27 of two or more "
            "implies a spill on 4545 Apple texture sections plus 40 witnesses with no exception - "
            "but every one of those was produced by APPLE's compiler; on the 157 executed objects "
            "attributed to this project's assembler the rule's antecedent has NEVER been true; it "
            "is satisfied vacuously, untested there; it refuses such programs instead; both figures are right "
            "about different producers; authoring needs a decision on whether a section may "
            "describe a program Apple would never have produced]")
    missing = list(resources.get("not_stated") or ()) + missing_extra
    # EVERY REFUSED FACT CARRIES ITS SETTLING EXPERIMENT, and a fact with none is itself reported -
    # silently omitting it would be the short-identity defect in a different dict.
    settled_by = {}
    for entry in missing:
        name = entry.split(" [")[0]
        settled_by[name] = SETTLED_BY.get(
            name, "NOT STATED: this fact is refused with no settling experiment recorded, which "
                  "makes the refusal one a reader cannot act on")
    return dict(status="planned_not_emitted", gpu_dispatched=False, placed=placed, rules=rules,
                not_determined=missing, settled_by=settled_by, bytes_emitted=0)


def author(abi):
    """Emit the narrow measured class, or refuse by name for every other texture shape."""
    report = plan(abi)
    if report["not_determined"]:
        raise Unmeasured(
            "this class has no measured rule for %s, and the contract states them on its own "
            "not_stated list rather than guessing. What is known: slot 27's declared read-only "
            "reading holds on %d of %d Apple texture sections and no more - and that denominator "
            "is a gate rather than the corpus: slot 27 is readable on all %d Apple sections, the "
            "same reading holds on %d of them, and it drops to 2.1%% on the vtable cohort one slot "
            "short of the gate, so the reading tracks the schema the section was written against "
            "rather than slot 27's contents. %d of the texture sections carry index "
            "44 inside slot 27 with nothing predicting when, and %d carry no kind-5 resource "
            "record at all. Authoring a value for either would be a donor value whoever wrote it. "
            "The discriminating family IS compiled and re-read from its committed objects at %s: "
            "it decided slot 27's rule - the uniformly read resources with 44 removed, refuted for "
            "H2 and H4 by M8 alone - and it decided nothing about the two record-presence facts, "
            "because all nine members carry neither. Every other field of this section is placed "
            "and listed."
            % (" and ".join(report["not_determined"]), CORPUS["slot27_declared_reading"],
               CORPUS["sections"], CORPUS["slot27_addressable_sections"],
               CORPUS["slot27_declared_reading_all"],
               CORPUS["carrying_44_in_slot27"], CORPUS["no_resource_record"],
               FAMILY_REREAD))
    # The first class with an empty refusal list now has a structural, contract-driven emitter.
    # Import lazily to keep the planning module usable by the emitter and its controls.
    from .texnarrowemit import _narrow_shape, emit as emit_narrow
    if _narrow_shape(abi):
        return emit_narrow(abi)
    # The structural class is independent of the compiler's register count.  Keep the original
    # exact witness route above, then admit a source-owned runtime variant whose register count and
    # instruction count come from this program while the shared texture/resource shape remains
    # unchanged.  g17texemit has the corresponding contract guards and dynamic slot-0 emission.
    from . import texemit as g17texemit
    if not g17texemit.admit(abi):
        return g17texemit.emit(abi)
    raise Unmeasured("the not_stated list is empty but no emitter exists for this texture class; "
                     "the measured narrow tex2d-read class is the only authorable route")


# WHAT MAKES A CONTRACT A TEXTURE CONTRACT, and it is not one key.
#
# g17authorobj routed on `abi["resources"]["textures"]` alone. Nothing falls THROUGH that to an
# authored buffer section - every perturbation refuses - but what it refuses WITH depends on an
# accident: with dict-shaped bindings the buffer selector raises a bare KeyError from
# `tuple(b[:3])`, and with tuple-shaped ones it reaches the generic system-register refusal, which
# tells a reader which two-buffer register sets are witnessed. True, irrelevant, and pointing at
# the wrong question - the exact confusion the routing comment in g17authorobj was written about,
# arriving through the back door when the single key it keys on is absent.
#
# A refusal that names the wrong question is worse than a crash, because a reader acts on it. So
# the predicate is every way a contract can say "texture", and the guard is that an ORDINARY
# contract is not captured: a predicate that answered True for everything would route the whole
# author into this refusal and no test could tell.
TEXTURE_SIGNALS = (
    "resources.textures is non-empty",
    "resources.samplers is non-empty",
    "resources.coordinate_publications is non-empty",
    "writes_texture is true",
    "a binding declares a texture element type",
)


def declares_a_texture(abi):
    """Every way this side's contract can say 'texture', and which one said it.

    Returns the list of signals that fired, empty for a contract that declares none. The list
    rather than a boolean, because a refusal that can say WHICH fact routed it is one a reader can
    act on, and because a signal that never fires on any contract is invisible behind an `or`.
    """
    if not isinstance(abi, dict):
        return []
    fired = []
    resources = abi.get("resources")
    if isinstance(resources, dict):
        for key, signal in (("textures", TEXTURE_SIGNALS[0]),
                            ("samplers", TEXTURE_SIGNALS[1]),
                            ("coordinate_publications", TEXTURE_SIGNALS[2])):
            value = resources.get(key)
            if value:
                fired.append(signal)
    if abi.get("writes_texture"):
        fired.append(TEXTURE_SIGNALS[3])
    for binding in abi.get("bindings") or ():
        if isinstance(binding, dict) and "texture" in str(binding.get("element_type", "")).lower():
            fired.append(TEXTURE_SIGNALS[4])
            break
    return fired


def emit(abi, bindings, led):
    """The common author's texture route. Records what it placed in the ledger, then refuses.

    g17authorobj._metadata calls this before the buffer classes so that a texture program's
    refusal names ITS question. Without the route it reached the generic system-register refusal
    and was told which two-buffer register sets are witnessed - true, irrelevant, and pointing at
    the wrong question.
    """
    report = plan(abi)
    for key, value in report["placed"].items():
        if key == "bindings":
            led["binding ranks"] = "; ".join(
                "%s %d at rank %d offset %d%s" % ("INTERNAL" if b["internal"] else "user",
                                                  b["index"], b["rank"], b["offset"],
                                                  " WRITTEN" if b["written"] else "")
                for b in value)
            continue
        led[key] = "%s (%s)" % (value, report["rules"][key])
    return author(abi)


def main(argv):
    if len(argv) == 2 and argv[1] == "--reproduce":
        report = reproduction_report()
        print("REPRODUCTION - %s, dispatch_eligible=%s" % (report["status"], report["dispatch_eligible"]))
        print("   %d retained witnesses in %d class shapes" % (report["witnesses"], report["shapes"]))
        print("   %d of %d reproduced from their OWN structure and stated values"
              % (report["self_reproduced"], report["witnesses"]))
        print("   %d of %d cross-reproductions byte-exact - each witness emitted from a DIFFERENT "
              "witness's structure" % (report["byte_exact"], report["cross_pairs"]))
        for target, donor, why in report["differing"]:
            print("      %s from %s: %s" % (target, donor, why))
        print("   excluded from the cross test (no sibling of their shape): %s"
              % ", ".join(report["excluded"]))
        print("   controls, each field moved on its own:")
        for label, moved, sums, note in report["controls"]:
            print("      %-16s %s byte(s) differ, sum holds: %-5s  (%s)" % (label, moved, sums, note))
        print("   tail provenance, %d tail bytes rebuilt from %d stated values:"
              % (report["tail_bytes"], report["tail_values"]))
        for label, held, note in report["provenance"]:
            print("      %-32s %s  (%s)" % (label, "held" if held else "LEAKED", note))
        print("   post-conditions on the reproduced section: %s" % report["verified"])
        return 0 if (report["byte_exact"] == report["cross_pairs"]
                     and report["self_reproduced"] == report["witnesses"]
                     and all(held for _l, held, _n in report["provenance"])) else 1
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    abi = json.load(open(argv[1]))
    report = plan(abi)
    print("PLACED, each with the rule that decided it:")
    for key, value in report["placed"].items():
        if key == "bindings":
            for b in value:
                print("   rank %d  index %-3d offset %-2d %-8s %s"
                      % (b["rank"], b["index"], b["offset"],
                         "INTERNAL" if b["internal"] else "user",
                         "WRITTEN" if b["written"] else ""))
            print("       (%s)" % report["rules"]["bindings"])
            continue
        print("   %-22s %-12s (%s)" % (key, value, report["rules"][key]))
    try:
        author(abi)
    except Unmeasured as refusal:
        print("\nREFUSED, by name:\n   %s" % str(refusal).replace(". ", ".\n   "))
        return 0
    return 0




# ---------------------------------------------------------------------------
# REPRODUCTION. Separate from authoring on purpose, and never dispatch-eligible.
#
# Integration's assignment: finish the serializer using explicit measured field inputs, reproduce
# the retained witnesses, use the original objects as comparison ORACLES rather than copying their
# sections into the emitter, and keep production authoring refused while the storage facts are
# missing. These two paths therefore share nothing but this file: `plan`/`author` take a contract
# and refuse; `reproduce` takes a class structure plus every measured value and emits bytes.
#
# WHAT MAKES THIS NOT A COPY, and it is checkable rather than asserted: a witness is emitted from a
# DIFFERENT witness's structure. The structure supplies positions, vtables, slot maps and table
# tails - class data, the same split g17mdgen's FOUR_INTERNAL already draws - and every value comes
# from the stated fields. If the emitter were copying, A could not be built from B's shape.
# ---------------------------------------------------------------------------

REPRODUCTION_ONLY = ("reproduced_not_authored", False)   # (status, dispatch_eligible)


# TAIL PROVENANCE. g17mdgen.build_from copies a table's tail bytes, so a reproduction that hands
# it a donor description inherits that donor's tail payload - 94 bytes on M0 and S0 - and the
# cross test cannot see it, because structure_key groups by the tail hash and therefore only ever
# pairs witnesses whose copied tails already agree. Integration found that in review and it is
# right: that is reconstruction and placement evidence, not a donor-independent serializer.
#
# So the tails are enumerated by purpose and rebuilt. A string span is a four-byte length followed
# by its text; a word span is one four-byte value; padding is zeroes to the end of the span the
# layout declares. Strings and words are VALUES and become explicit inputs; padding is structural
# and comes from the declaration. `extra` is inside the same boundary: a description carrying any
# is refused rather than copied.
TAIL_SYMBOL = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."


def tail_spans(tail):
    """[(kind, offset, length, value)] for one table tail: 'string', 'word' or 'padding'.

    A string needs at least two characters of symbol text, because a one-byte span of printable
    bytes is a word that happens to look like text - S6 carries a word whose bytes read as "0",
    and a first version called it a string.
    """
    import struct as _s
    blob = bytes(tail)
    spans, at = [], 0
    while at < len(blob):
        if at + 4 <= len(blob):
            n = _s.unpack_from("<I", blob, at)[0]
            text = blob[at + 4:at + 4 + n]
            if (2 <= n <= len(blob) - at - 4 and len(text) == n
                    and all(chr(c) in TAIL_SYMBOL for c in text)):
                spans.append(("string", at, 4 + n, text.decode()))
                at += 4 + n
                continue
        if all(c == 0 for c in blob[at:]):
            spans.append(("padding", at, len(blob) - at, None))
            break
        if at + 4 <= len(blob):
            spans.append(("word", at, 4, _s.unpack_from("<I", blob, at)[0]))
            at += 4
            continue
        spans.append(("other", at, len(blob) - at, blob[at:].hex()))
        break
    return spans


def tail_inputs(description):
    """{(table position, span index): value} - every value-bearing byte of every tail."""
    out = {}
    for pos, table in description["tables"].items():
        for index, (kind, _off, _len, value) in enumerate(tail_spans(table.get("tail") or b"")):
            if kind in ("string", "word"):
                out[(pos, index)] = value
            elif kind == "other":
                raise Unmeasured("a tail span at table %s is not a string, a word or padding; "
                                 "reproduction will not copy bytes it cannot name" % pos)
    return out


def _rebuild_tail(template, values, pos):
    """A tail built from the declared spans and the stated values - never from the template bytes."""
    import struct as _s
    out = bytearray()
    for index, (kind, off, length, _value) in enumerate(tail_spans(template)):
        if kind == "padding":
            out.extend(b"\0" * length)
            continue
        if (pos, index) not in values:
            raise Unmeasured("tail span %d of table %s is value-bearing and unstated" % (index, pos))
        value = values[(pos, index)]
        if kind == "string":
            text = value.encode()
            out.extend(_s.pack("<I", len(text)) + text)
        else:
            out.extend(_s.pack("<I", value))
    if len(out) != len(template):
        raise Unmeasured("rebuilt tail of table %s is %d bytes against a declared %d"
                         % (pos, len(out), len(template)))
    return bytes(out)


# ---------------------------------------------------------------------------
# THE RESOLVER, and the author-side agreement check integration assigned at 29644d9c.
#
# The split agreed with the compiler was that they state the semantic requirement and this side
# derives the block layout. Integration accepted it with one requirement: omitting an independently
# chosen offset from the semantic REQUEST is right, but omitting the resolved layout from the
# byte-generation CONTRACT is not - the compiler has to encode the publication and main's read
# against a layout it was given, not one it guessed to match.
#
# So: the compiler asks, this returns an immutable resolved layout with an identity, the compiler
# emits against it, and the author RECOMPUTES the identity from the delivered binding list before
# packaging. A program built against one layout and delivered with another binding list is a stale
# pairing, and it refuses - which is the control integration asked for.
# ---------------------------------------------------------------------------


def resolve(internal, user_bindings, preload_terms=0):
    """The resolved layout for a binding list, in the compiler's canonical form.

    NOT IMMUTABLE IN THE PYTHON SENSE, and calling it that was the wrong word: it returns a fresh
    plain dict each call, because the layout has to serialise into a contract. What holds instead
    is that it is a pure function of the declared bindings and that any mutation between resolution
    and packaging is DETECTED at the author boundary, by comparing every field rather than a digest
    the caller supplies. That is the right guarantee HERE, where the value has to cross a contract
    as JSON; it is not a claim that detection beats immutability in general, and integration was
    right to say so. Where a value never has to serialise - the compiler capturing its own encoding
    inputs - immutability is the stronger thing and belongs there.

    THE TWO SIDES MUST COMPUTE THE SAME BYTES OR THE CHECK IS VACUOUS - it would refuse everything
    and look like vigilance. So the fields and their order are the compiler's
    (declared, internal, order, ranks, record_bytes, block_bytes) with ONE addition of mine:
    descriptor_offsets. The compiler encodes 4 * rank in the code and this side encodes 2 * rank in
    the binding records - the same law in two units, witnessed from both artifacts - and an identity
    that pins only their unit does not pin what I emitted.

    `written` is deliberately NOT in it: a binding list that differs only in a written flag resolves
    to the same block, so the compiler's encoding stays valid and it is not a stale pairing.
    Nothing about the program's code reaches this either, which is why S2 and S3 place their
    preloads at 20 and 24 while addressing only bytes 0, 4 and 8.
    """
    import hashlib
    import json as _json
    declared = [i for i, _w in user_bindings]
    internal_indices = sorted({i for _r, i in internal})
    order, ranks = g17resource.layout(declared, internal=internal_indices)
    canonical = {
        "declared": declared,
        "internal": internal_indices,
        "order": list(order),
        "ranks": {str(index): rank for index, rank in ranks.items()},
        "record_bytes": RECORD_BYTES,
        "block_bytes": RECORD_BYTES * len(order),
        "descriptor_offsets": {str(index): 2 * rank for index, rank in ranks.items()},
    }
    resolved = dict(canonical)
    resolved["sha256"] = hashlib.sha256(
        _json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # Derived, and outside the identity: the preload publishes at the end of the whole declared
    # record block, so block_bytes already pins it and restating it inside the hash would be a
    # second copy of the same fact.
    resolved["preload_offset"] = resolved["block_bytes"] if preload_terms else None
    return resolved


# Every field resolve() returns must be inside the identity, be the digest itself, or be compared
# explicitly by check_resolved. preload_offset was outside the identity - correctly, since
# block_bytes already pins it - and for a while was outside the comparison too, which is how a
# delivered 4096 passed. "Not in the identity" is not "not checked", and CHECKED_DERIVED_FIELDS
# exists so the next derived field cannot repeat it silently: the test asserts the partition.
CHECKED_DERIVED_FIELDS = ("preload_offset",)


def resolved_for(abi):
    """The layout this contract's own binding list resolves to - the recomputation side of the check."""
    resources = abi.get("resources") or {}
    internal = [(r["rank"], r["apple_index"]) for r in resources.get("internal") or ()]
    users = [(b["index"], bool(b["written"])) for b in abi["bindings"]]
    return resolve(internal, users, len(resources.get("preloads") or ()))


def check_resolved(abi):
    """Refuse a program whose delivered layout is not the one its binding list resolves to.

    COMPARING DIGESTS IS NOT COMPARING LAYOUTS, and a first version did only that: a caller could
    set block_bytes to 4096, keep the old sha256, and pass - the check validated a number the
    caller supplied against a number I computed, and never looked at the fields the compiler
    actually encoded against. Integration found it with that exact mutation. Every canonical field
    is compared now, and the delivered digest is checked against the delivered FIELDS as well, so a
    mutation with a refreshed sha is caught by the field comparison and one with a stale sha is
    caught twice.
    """
    import hashlib
    import json as _json
    resources = abi.get("resources") or {}
    delivered = resources.get("resolved_layout")
    if delivered is None:
        if resources.get("preloads"):
            raise Unmeasured(
                "this contract carries preloads and no resolved_layout. The publication and main's "
                "block read are encoded against a layout, so the layout it was encoded against has "
                "to arrive with them - otherwise nothing connects the bytes to this binding list.")
        return None
    if not isinstance(delivered, dict):
        raise ValueError("resolved_layout must be a mapping")
    recomputed = resolved_for(abi)
    differing = sorted(field for field in CANONICAL_FIELDS
                       if delivered.get(field) != recomputed[field])
    if differing:
        raise Unmeasured(
            "the delivered resolved layout differs from what this binding list resolves to, on "
            "%s: %s against %s - a stale pairing. The layout is not a value to carry alongside "
            "the bindings - it is a "
            "consequence of them, and the two have come apart, so the publication and main's block "
            "read were encoded against offsets this image does not have."
            % (", ".join(differing),
               {f: delivered.get(f) for f in differing}, {f: recomputed[f] for f in differing}))
    own = hashlib.sha256(_json.dumps({f: delivered.get(f) for f in CANONICAL_FIELDS},
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if delivered.get("sha256") != own:
        raise Unmeasured(
            "the delivered resolved layout's sha256 is not the digest of its own fields: %s "
            "against %s. A layout whose identity does not match its contents cannot pin anything."
            % (str(delivered.get("sha256"))[:16], own[:16]))
    if delivered["sha256"] != recomputed["sha256"]:
        raise Unmeasured("resolved layout %s against %s: a stale pairing"
                         % (delivered["sha256"][:16], recomputed["sha256"][:16]))
    if "preload_offset" in delivered and delivered["preload_offset"] != recomputed["preload_offset"]:
        raise Unmeasured(
            "the delivered preload_offset disagrees with the resolved binding layout: "
            "%r against %r" % (delivered["preload_offset"], recomputed["preload_offset"]))
    return recomputed


def structure_key(description):
    """The class shape of a described section: positions, vtables, slot maps and table tails.

    Tails are IN the key. Without them M7's sampled section groups with the six read-path sections
    of its size and twelve cross-reproductions fail; with them M7 separates and all thirty-six
    succeed. A shape key that is missing a component reports the component as a reproduction error.
    """
    import hashlib
    tables = tuple(sorted(
        (pos, t["vtpos"], t["vlen"], t["tlen"], tuple(sorted(t["slots"].items())),
         len(t.get("tail") or b""))
        for pos, t in description["tables"].items()))
    # The key names the tail LAYOUT, not the tail bytes. Hashing the bytes - which a first
    # version did, and on a field named "residual" that descriptions do not even carry, so the
    # component was a no-op - meant the cross test only ever paired witnesses whose copied tails
    # already agreed. Naming the span kinds and lengths instead lets siblings with different tail
    # CONTENTS pair, which is the pairing that tests anything.
    layout = tuple(sorted((pos, tuple((kind, off, length)
                                      for kind, off, length, _v in tail_spans(t.get("tail") or b"")))
                          for pos, t in description["tables"].items()))
    return (description["size"], description["root"], tables, layout,
            tuple(sorted(description.get("extra") or {})))


def measured_fields(section):
    """{(table position, slot): value} - every value-carrying field of a retained witness.

    This reads the ORACLE. It is the measurement a contract would have to carry, not an input path:
    nothing in `plan` or `author` calls it, and a value that reached production through here would
    be a donor value.
    """
    from . import mdgen as M
    description = M.describe(bytes(section))
    return {(pos, slot): value
            for pos, table in description["tables"].items()
            for slot, (_offset, _width, value) in table["fields"].items()}


def reproduce(structure, fields, tails=None):
    """Emit a section from a class STRUCTURE, explicit FIELDS and explicit TAILS.

    `structure` is a described section used for its DECLARATIONS only - positions, vtables, slot
    maps, and the span layout of each tail. Every value comes from `fields` and `tails`: table
    field values from the first, tail strings and words from the second, and padding from the
    declaration. Anything unstated refuses rather than being inherited, which is the whole
    difference between reproducing and copying.

    The tails are rebuilt rather than passed through, because g17mdgen.build_from copies them and
    a donor's 94 tail bytes would otherwise ride along unnoticed - integration found exactly that
    in review of the first version.
    """
    import copy
    from . import mdgen as M
    if structure.get("extra"):
        raise Unmeasured("this structure carries %d extra byte span(s) outside any table; "
                         "reproduction will not copy bytes it cannot name"
                         % len(structure["extra"]))
    wanted = {(pos, slot)
              for pos, table in structure["tables"].items()
              for slot in table["fields"]}
    missing = sorted(wanted - set(fields))
    if missing:
        raise Unmeasured("reproduction needs every field stated; %d are not: %s"
                         % (len(missing), missing[:6]))
    tails = {} if tails is None else tails
    rebuilt = copy.deepcopy(structure)
    for pos, table in rebuilt["tables"].items():
        template = table.get("tail") or b""
        if template:
            table["tail"] = _rebuild_tail(template, tails, pos)
    return bytes(M.build_from(rebuilt, values={k: fields[k] for k in wanted}))


def verify(section):
    """Post-conditions a texture section must satisfy, checked on the emitted bytes.

    The sum rule is exact on 8,970 of 8,970 Apple sections carrying a kind-6 record, so a section
    whose kind-6 fields do not add to its slot 1 is malformed and saying so here is cheaper than
    discovering it downstream. Checked AFTER emission, on the bytes, rather than trusted before it.
    """
    import struct
    from . import gpumd as GM
    from . import recmine as RM
    pk = GM.kernel_table(section)
    slots, _ = GM.table_at(section, pk)
    out = {}
    out["slot 1"] = struct.unpack_from("<I", section, pk + slots[1])[0] if slots[1] else None
    record = [r for r in (RM.vec_records(section, 2) or []) if r.get(0) == 6]
    out["kind-6"] = (record[0].get(2, 0), record[0].get(3, 0)) if record else None
    if out["kind-6"] is not None and out["slot 1"] is not None:
        f2, f3 = out["kind-6"]
        out["sum holds"] = (f2 + f3 == out["slot 1"])
    else:
        out["sum holds"] = None
    out["slot 38"] = struct.unpack_from("<I", section, pk + slots[38])[0] if len(slots) > 38 and slots[38] else None
    out["slot 42"] = struct.unpack_from("<I", section, pk + slots[42])[0] if len(slots) > 42 and slots[42] else None
    out["42 equals 38"] = out["slot 38"] == out["slot 42"]
    return out


def _witnesses():
    import os
    from . import mdgen as M
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    found = {}
    for family in ("g17-texture-family-compiles-v1", "g17-texture-slot1-compiles-v1"):
        base = os.path.join(root, "results", family)
        if not os.path.isdir(base):
            continue
        for member in sorted(os.listdir(base)):
            path = os.path.join(base, member, "section-GPU_METADATA_compute.bin")
            if os.path.exists(path):
                blob = open(path, "rb").read()
                found[member] = (blob, M.describe(blob))
    return found


def reproduction_report():
    """Reproduce every retained witness from ANOTHER witness's structure, and run the controls."""
    import collections
    import struct
    from . import gpumd as GM
    witnesses = _witnesses()
    shapes = collections.defaultdict(list)
    for member, (_blob, description) in witnesses.items():
        shapes[structure_key(description)].append(member)
    rows, exclusions, pairs, exact = [], [], 0, 0
    for members in shapes.values():
        if len(members) < 2:
            exclusions.append(members[0])
            continue
        for target in members:
            for donor in members:
                if donor == target:
                    continue
                blob, description = witnesses[target]
                built = reproduce(witnesses[donor][1], measured_fields(blob), tail_inputs(description))
                pairs += 1
                exact += built == blob
                if built != blob:
                    rows.append((target, donor, "DIFFERS"))
    # EVERY WITNESS IS ALSO REPRODUCED FROM ITS OWN STRUCTURE, so a shape with no sibling is still
    # emitted from stated values rather than skipped. The CROSS test is the one that needs a
    # sibling, and the five singletons are excluded from that and from nothing else.
    self_exact = 0
    for member, (blob, description) in witnesses.items():
        self_exact += reproduce(description, measured_fields(blob), tail_inputs(description)) == blob

    # THE CONTROLS. Move slot 1 and each kind-6 component ON ITS OWN and check that exactly the
    # intended bytes move, and that the sum post-condition sees the inconsistent triple. The fields
    # are located STRUCTURALLY - the per-kernel table's slot 1, and the table whose kind is 6 - not
    # by searching for a matching value, which a first version did and which moved the wrong field
    # twice: the value 2 and the value 6 each occur in several fields of the same section.
    controls = []
    target = "M0" if "M0" in witnesses else sorted(witnesses)[0]
    blob, description = witnesses[target]
    fields = measured_fields(blob)
    pk = GM.kernel_table(blob)
    tails = tail_inputs(description)
    base = reproduce(description, fields, tails)
    kind6 = [pos for pos, table in description["tables"].items()
             if table["fields"].get(0) and table["fields"][0][2] == 6]
    sites = [("slot 1", (pk, 1), 4)]
    if kind6:
        sites += [("kind-6 field2", (kind6[0], 2), 1), ("kind-6 field3", (kind6[0], 3), 1)]
    for label, key, delta in sites:
        if key not in fields:
            controls.append((label, None, None, "field absent"))
            continue
        moved = dict(fields); moved[key] = fields[key] + delta
        out = reproduce(description, moved, tails)
        differing = sum(1 for i in range(len(out)) if out[i] != base[i])
        controls.append((label, differing, verify(out)["sum holds"], "moved by %+d" % delta))

    # THE PROVENANCE CONTROLS, which are the ones integration's review asked for. Change the
    # DONOR's tail payload while holding the explicit inputs fixed: a copier would carry the
    # change through, a rebuild cannot see it. And change the tail's LAYOUT: the rebuild has to
    # refuse rather than emit a section whose structure it was not given.
    import copy as _copy
    provenance = []
    donor = _copy.deepcopy(description)
    touched = None
    for pos, table in donor["tables"].items():
        spans = tail_spans(table.get("tail") or b"")
        words = [i for i, (kind, _o, _l, _v) in enumerate(spans) if kind == "word"]
        if words:
            index = words[0]
            _kind, off, _len, value = spans[index]
            blob = bytearray(table["tail"])
            blob[off:off + 4] = struct.pack("<I", (value + 0x5A5A) & 0xFFFFFFFF)
            table["tail"] = bytes(blob)
            touched = (pos, index, value)
            break
    if touched is not None:
        out = reproduce(donor, fields, tails)
        provenance.append(("donor tail payload changed", out == base,
                           "table %s span %d, %d -> %d" % (touched[0], touched[1], touched[2],
                                                           (touched[2] + 0x5A5A) & 0xFFFFFFFF)))
    widened = _copy.deepcopy(description)
    for pos, table in widened["tables"].items():
        if table.get("tail"):
            table["tail"] = bytes(table["tail"]) + b"\x01\x00\x00\x00"
            break
    try:
        reproduce(widened, fields, tails)
        provenance.append(("donor tail layout changed", False, "emitted instead of refusing"))
    except Unmeasured as refusal:
        provenance.append(("donor tail layout changed", True, str(refusal)[:60]))
    carrying = _copy.deepcopy(description)
    carrying["extra"] = {0: b"\x00\x00\x00\x00"}
    try:
        reproduce(carrying, fields, tails)
        provenance.append(("structure carrying extra bytes", False, "emitted instead of refusing"))
    except Unmeasured as refusal:
        provenance.append(("structure carrying extra bytes", True, str(refusal)[:60]))

    return dict(status=REPRODUCTION_ONLY[0], dispatch_eligible=REPRODUCTION_ONLY[1],
                gpu_dispatched=False, witnesses=len(witnesses), shapes=len(shapes),
                cross_pairs=pairs, byte_exact=exact, differing=rows,
                self_reproduced=self_exact,
                excluded=sorted(exclusions), controls=controls, provenance=provenance,
                tail_bytes=sum(len(t.get("tail") or b"") for t in description["tables"].values()),
                tail_values=len(tails),
                verified=verify(base))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
