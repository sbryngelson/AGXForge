#!/usr/bin/env python3
"""THE ORDERED BINDING LIST AN IMAGE DECLARES - one authority, so nothing has to guess a rank.

A store's address expression encodes the buffer's RANK: its position in the list the IMAGE
declares. Two sides need that number and neither owns it. The compiler was deriving it from the
source's declaration; the linker emits the list. When those disagree the store ranks into a list
that does not exist, the dispatch returns its fill value at status 0, and it looks exactly like a
broken opcode.

THE TWO HEURISTICS PASS DISJOINT SETS OF KERNELS, which is the proof that neither is the rule:

    declare what the source declares    `identity` passes; six three-buffer kernels fail
    declare only what you touch          those six pass; `identity` fails

A rank derivable from the source list alone could not do that.

WHAT THE CORPUS SAYS, over 12,044 kernels, comparing each source's declared buffer set against the
binding indices its metadata records:

    records EXACTLY what the source declares    11,312   93.9%
    records FEWER than declared (eliminated)       470    3.9%
    records MORE than declared (internals)         256    2.1%
    neither a subset nor a superset                  6

So the source declaration is the right DEFAULT and is wrong for one kernel in sixteen, in BOTH
directions. The 2.1% is the direction the compiler cannot see at all: a4-cmpx-uni declares one
buffer and its section records two, because an internal binding gets added that the source never
names, and a rank computed from the source list is off by one on every one of those.

SO THIS MODULE DOES NOT GUESS. `layout` takes what the caller actually knows and returns the
ordered list plus each index's rank; `from_declaration` applies the 93.9% default and REFUSES when
the caller says something that puts it outside that population. The compiler asks for a rank
instead of deriving one, and when the answer is not derivable it gets an error rather than a
number.

VALIDATED AGAINST APPLE'S OWN ORDER, not just its set. Over the 12,044 corpus kernels with a
declared buffer list, from_declaration reproduces the metadata's recorded binding list EXACTLY -
same members, same order - in 10,964 of them, 91.0%. The remaining 1,080 differ in the SET, and

    the "same set, different order" bucket is EMPTY.

That is the result that matters here: whenever the membership is right the ORDER is right, with no
exceptions, so the ordering law - internals first ascending, then users ascending - is not the open
question. The entire residual is which buffers the image declares, which is the compiler's fact and
is a parameter of this function rather than a guess inside it.

    python3 tools/g17resource.py     the rules, and the refusals demonstrated
"""
import sys


class Undecidable(ValueError):
    """The ordered list is not determined by what the caller supplied. Refusing beats guessing."""


def layout(indices, internal=()):
    """(ordered index list, {index: rank}) for an image declaring `indices`.

    INTERNALS RANK FIRST, ascending, then the user indices ascending - the pointer-block law this
    project measured over 74,325 records. `internal` names indices the backend knows are internal;
    they need not appear in `indices`.
    """
    idx = list(indices)
    if len(set(idx)) != len(idx):
        # 513 of the 854 acceleration-structure sections carry a DUPLICATE index whose two records
        # differ only in their offset, so a duplicate is real in Apple's own bytes - but it makes
        # "the rank of index i" ill-posed, and this function's whole purpose is to answer that.
        raise Undecidable("index %s appears twice; a rank keyed on the index is ill-posed for "
                          "duplicates (513 of 854 AS sections have them)"
                          % [i for i in set(idx) if idx.count(i) > 1])
    ints = sorted(set(internal))
    users = sorted(i for i in idx if i not in set(ints))
    order = ints + users
    return order, {i: r for r, i in enumerate(order)}


def from_declaration(declared, eliminated=(), internal=()):
    """The list for a kernel whose source declares `declared`.

    `eliminated` names buffers the compiler dropped - the 3.9% case - and `internal` names ones it
    added, the 2.1%. Both are the COMPILER'S facts: nothing in the section says a buffer was
    eliminated, and nothing in the source says an internal was added. Supplying neither asks for
    the 93.9% default and gets it.
    """
    keep = [i for i in declared if i not in set(eliminated)]
    if not keep and not internal:
        raise Undecidable("every declared buffer was eliminated and none is internal; an image "
                          "with no bindings is outside this layout's domain")
    return layout(keep, internal=internal)


def rank(index, declared, eliminated=(), internal=()):
    """The rank a store should encode for `index`. Raises rather than returning a wrong number."""
    _order, ranks = from_declaration(declared, eliminated, internal)
    if index not in ranks:
        raise Undecidable("index %d is not in the image's binding list %s; a store to it would "
                          "rank into a list that does not exist" % (index, sorted(ranks)))
    return ranks[index]


# THE RECOVERED ELEMENT TYPES. g17authorobj establishes that a buffer's type does not reach
# __GPU_METADATA - for the measured scalar class that is a proof rather than a correlation, since
# g17mdgen.build takes only the binding INDICES and has no type parameter at all. So a type
# disagreement between the two sides cannot be caught in the delivered bytes, which is exactly why
# it has to be caught here, before the image is built.
RECOVERED_TYPES = frozenset({
    "float", "float2", "float4", "half", "half2", "half4", "bfloat", "bfloat2", "bfloat4",
    "int", "int4", "uint", "uint2", "uint4", "short", "short2", "ushort",
    "long", "ulong", "uchar", "uchar4", "atomic_int", "atomic_uint", "atomic_float"})


def check_agreement(abi, profile=None, profiles=None):
    """Do the compiler's ABI and the linker's contract agree? -> list of disagreements.

    An EMPTY list is agreement. Each entry names one fact both sides must hold and what each said.
    This runs before an image is built, and it is the only place a TYPE disagreement can be caught:
    element type does not reach the metadata, so a delivered-image check cannot see it.

    `abi` is what the compiler hands over - bindings with index/offset/written/element_type, entry,
    and optionally `forms`. `profile` names a measured class whose contract must be matched exactly.
    """
    bad = []
    binds = list(abi.get("bindings") or ())
    if not binds:
        return ["the ABI declares no bindings; an image with none is outside this contract"]

    got, seen = [], set()
    for b in binds:
        idx = b.get("index") if isinstance(b, dict) else b[0]
        off = b.get("offset") if isinstance(b, dict) else b[1]
        wr = b.get("written") if isinstance(b, dict) else b[2]
        et = (b.get("element_type") if isinstance(b, dict)
              else (b[3] if len(b) > 3 else None))
        if idx in seen:
            bad.append("binding index %s appears twice; a rank keyed on the index is ill-posed"
                       % idx)
        seen.add(idx)
        if not isinstance(off, int) or off < 0 or off % 2:
            bad.append("binding %s has pointer-block offset %r; offsets are non-negative and even"
                       % (idx, off))
        if et is not None and et not in RECOVERED_TYPES:
            bad.append("binding %s declares element type %r, which the corpus never declares - "
                       "refusing rather than assuming it behaves like the 24 measured types"
                       % (idx, et))
        got.append((idx, off, bool(wr)))

    # the linker's own ordering authority must reproduce the offsets the compiler states
    try:
        _order, ranks = layout([i for i, _o, _w in got])
        for idx, off, _w in got:
            if ranks[idx] * 2 != off:
                bad.append("binding %s: the compiler states pointer-block offset %d, the linker's "
                           "rank law gives %d (rank %d x 2)"
                           % (idx, off, ranks[idx] * 2, ranks[idx]))
    except Undecidable as e:
        bad.append("the linker cannot rank this list: %s" % str(e)[:80])

    if profile:
        prof = (profiles or {}).get(profile)
        if prof is None:
            bad.append("no measured profile named %r on the linker side" % profile)
        else:
            if tuple(got) != tuple(prof["bindings"]):
                bad.append("profile %s is measured for bindings %s; the ABI states %s"
                           % (profile, list(prof["bindings"]), got))
            if abi.get("entry") is not None and abi["entry"] != prof["entry"]:
                bad.append("profile %s pins entry %d; the ABI states %s"
                           % (profile, prof["entry"], abi["entry"]))
            if abi.get("arch_flag") is not None and bool(abi["arch_flag"]) != prof["arch_flag"]:
                bad.append(
                    "profile %s ships __GPU_ARCH_LD_MD with the flag %s and that section has "
                    "hardware runs behind it; the ABI reports %s. The two paths must be chosen "
                    "deliberately, not silently interchanged."
                    % (profile, "ELIDED" if not prof["arch_flag"] else "SET",
                       "SET" if abi["arch_flag"] else "ELIDED"))
            allowed = prof.get("forms")
            if allowed is not None and abi.get("forms") is not None:
                extra = sorted(set(map(tuple, abi["forms"])) - set(map(tuple, allowed)))
                if extra:
                    bad.append("profile %s accepts a fixed form set; the program emits %d form(s) "
                               "outside it: %s. A widened instruction set is a different class."
                               % (profile, len(extra), extra[:4]))
    return bad


def main():
    print(__doc__.split("\n\n")[0])
    print("\n   the default, and the two corrections:\n")
    for label, kw in (("identity: declares 1 and 2, touches 2", dict(declared=[1, 2])),
                      ("a kernel whose buffer 1 was eliminated",
                       dict(declared=[1, 2], eliminated=[1])),
                      ("a4-cmpx-uni: declares 2, image adds internal 1",
                       dict(declared=[2], internal=[1]))):
        order, ranks = from_declaration(**kw)
        print("   %-42s order %-10s ranks %s" % (label, order, ranks))
    print("\n   compiler/linker agreement, the check that runs before an image is built:\n")
    from . import authorobj as _A
    good = {"bindings": [{"index": 1, "offset": 0, "written": False, "element_type": "half"},
                         {"index": 2, "offset": 2, "written": True, "element_type": "half"}],
            "entry": 64, "arch_flag": False}
    print("   %-42s %s" % ("a half-typed two-binding ABI",
                           check_agreement(good, "scalar-buffer-two-bindings-measured-v2",
                                           _A.PROFILES) or "AGREES"))
    for label, mut in (
        ("an unrecovered element type", {"bindings": [dict(good["bindings"][0],
                                                           element_type="quarterfloat"),
                                                      good["bindings"][1]]}),
        ("an odd pointer-block offset", {"bindings": [dict(good["bindings"][0], offset=1),
                                                      good["bindings"][1]]}),
        ("the arch flag SET", {"arch_flag": True}),
    ):
        errs = check_agreement(dict(good, **mut), "scalar-buffer-two-bindings-measured-v2",
                               _A.PROFILES)
        print("   %-42s %s" % (label, (errs[0][:74] if errs else "AGREES <-- should not")))
    print("\n   and what layout refuses:\n")
    for label, fn in (
        ("a duplicate index", lambda: layout([1, 2, 2])),
        ("everything eliminated", lambda: from_declaration([1], eliminated=[1])),
        ("a store to a buffer not in the list", lambda: rank(3, [1, 2])),
    ):
        try:
            fn()
            print("   %-38s ACCEPTED  <-- must have refused" % label)
        except Undecidable as e:
            print("   %-38s %s" % (label, str(e)[:72]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
