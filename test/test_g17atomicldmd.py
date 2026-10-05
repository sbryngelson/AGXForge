#!/usr/bin/env python3
"""The atomic __GPU_LD_MD declaration, against Apple's own sections rather than against itself.

A device atomic is declared by slot 24 of table 136. THE LAYOUT THAT CARRIES IT DEPENDS ON THE
SECTION'S SIZE, AND THE FIRST VERSION OF THIS FILE GOT THAT WRONG IN A WAY THAT FAULTED THE DRIVER.
What stood here said the declaration requires a different body layout - six fields moving four
bytes, tlen 40 -> 48 - measured on Apple's 461 atomic objects, every one of which is 224 bytes. So
it changed two variables at once and credited both to the atomic. The offsets are section-relative,
so that shape at 216 bytes points every field eight bytes past its data: it described correctly,
round-tripped byte-identically, reported slot 24 present, and SEGFAULTED the host inside
AGX::DynamicLoader at newComputePipelineStateWithDescriptor - no GPU submission, no gpu event.

The separating population was in the cache: 20 objects at 216 bytes with tlen 40 that declare the
atomic, all compiled from Metal source by Apple's toolchain, all byte-identical, and field-for-field
identical to this backend's own 216-byte section. At that size the declaration is FOUR BYTES and
moves nothing. tlen and the field offsets track the SIZE; the declaration is orthogonal to both.

So the cases below compare against APPLE at each size - and the 216 case compares byte for byte
against a vendor-linked section, which is the strongest form available. A round trip proves a
description is complete; it says nothing about whether the candidate matches the target, and every
self-consistent check passed on the candidate that faulted the loader.

ledger/g17-the-atomic-declaration-is-slot-24-of-table-136.toml
ledger/g17-the-atomic-probe-was-never-writing-memory.toml
"""
import hashlib
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

from agxforge.g17 import ldmd, mdgen                                        # noqa: E402

CACHE = os.path.expanduser("~/.cache/agxforge/agx")
DONOR = os.path.join(CACHE, "ms-atomic_add.u-1", "out", "object", "0-0")


def apple_t136():
    from agxforge.g17.bind import section
    return mdgen.describe(section(DONOR, "__GPU_LD_MD", "__compute"))["tables"][136]


# One of the 20 vendor-linked 216-byte sections that DECLARE the atomic. Named rather than
# discovered so the case fails loudly if the witness disappears, instead of silently skipping.
WITNESS_216 = os.path.join(CACHE, "probe-uniatom", "out", "object", "0-0")


def witness_216_section():
    from agxforge.g17.bind import section
    return section(WITNESS_216, "__GPU_LD_MD", "__compute")


class TheExistingPathDoesNotMove(unittest.TestCase):
    """The control, and it comes first. A new layout for atomics must not touch the sections that
    already execute - seventeen kernels run on the default one.
    """

    def test_the_default_output_is_unchanged(self):
        for kwargs, digest, size in (
                (dict(entry=64), "1b14207cdf6df455", 216),
                (dict(entry=64, restore=True), "c494b3c9121d627f", 216),
                (dict(entry=128), "d5bf44c831851793", 216)):
            built = ldmd.build(**kwargs)
            self.assertEqual(len(built), size, kwargs)
            self.assertTrue(hashlib.sha256(built).hexdigest().startswith(digest), kwargs)

    def test_the_default_layout_puts_the_entry_pc_where_slot_24_would_go(self):
        """Why the flag cannot simply be added: in this layout slot 6 owns bytes 20..23, and
        Apple's slot 24 lives at 22. This asserts the overlap rather than describing it, so the
        reason the atomic path exists cannot quietly stop being true.
        """
        table = mdgen.describe(ldmd.build(entry=64))["tables"][136]
        self.assertEqual(table["slots"][6], 20)
        self.assertEqual((table.get("fields") or {})[6][1], 4, "slot 6 is not four bytes wide")
        self.assertLess(table["slots"][6], 22)
        self.assertGreater(table["slots"][6] + 4, 22)


class TheAtomicLayoutMatchesApple(unittest.TestCase):

    def test_the_shape_is_the_measured_atomic_one(self):
        built = ldmd.build(entry=64, atomic=True)
        table = mdgen.describe(built)["tables"][136]
        self.assertEqual(len(built), 224, "the atomic body is eight bytes longer")
        self.assertEqual(table["tlen"], 48)
        self.assertEqual(sorted(table["slots"]), [2, 5, 6, 7, 18, 24, 29, 38, 40])
        self.assertEqual(table["slots"][24], 22, "the declaration is not at its measured offset")
        self.assertEqual(table["slots"][6], 24, "the entry PC did not move out of slot 24's way")

    def test_it_round_trips(self):
        built = ldmd.build(entry=64, atomic=True)
        described = mdgen.describe(built)
        self.assertEqual(bytes(mdgen.build_from(described)), bytes(built))

    def test_every_written_field_equals_apples(self):
        """The case that matters, and the one the near-miss would have failed. `f0_5` is the
        caller's, so it is supplied as Apple's value here to compare layouts rather than defaults.
        """
        if not os.path.exists(DONOR):
            self.skipTest("the build cache donor is absent: %s" % DONOR)
        apple = apple_t136().get("fields") or {}
        mine = mdgen.describe(ldmd.build(entry=64, atomic=True, f0_5=36))["tables"][136]
        fields = mine.get("fields") or {}
        self.assertEqual(sorted(fields), [2, 5, 6, 7, 18, 24, 29, 38, 40])
        for slot, value in sorted(fields.items()):
            self.assertEqual(value, apple.get(slot),
                             "slot %d differs from Apple's section" % slot)
        self.assertEqual(fields[24], (22, 1, 1), "the declaration is not one byte holding 1")

    def test_216_is_a_request_and_not_a_default_to_be_overridden(self):
        """The atomic body occupies 136..184, so it FITS a 216-byte section - which is what makes
        an in-place splice into a host's 216-byte section possible, the host having zero slack
        before __GPU_ARCH_LD_MD.

        This case exists because the first version conflated the default with a request: `if
        atomic and size == 216: size = ATOMIC_SIZE` silently handed 224 bytes to a caller who
        asked for 216. Apple's 224 is what Apple puts after the body, not what the body needs.
        """
        # EACH SIZE KEEPS ITS OWN SHAPE. The wrong version of this case asserted tlen 48 and
        # slot 24 at 22 for every size, which is how the 224-byte offsets reached a 216-byte
        # section and faulted the driver's metadata loader.
        # tlen IS NOT WRITTEN BY THE 216 PATH, AND ASSERTING 40 HERE WOULD HAVE HIDDEN THAT. A
        # bare 216-byte build leaves t136's tlen at ZERO - Apple's is 40, and the 40 in our
        # byte-identical section comes from the RESTORE table (0x34 -> 0x28), not from the layout.
        # Seventeen kernels execute on tlen 0, which is ldmd's own recorded scope limit: the
        # loader does not read it. The 224 path passes ATOMIC_TLEN explicitly, so it reads 48
        # either way.
        for size, expected, tlen, decl in ((None, 224, 48, 22), (216, 216, 0, 18),
                                           (224, 224, 48, 22)):
            built = ldmd.build(entry=64, atomic=True, size=size)
            self.assertEqual(len(built), expected, "size=%r" % (size,))
            table = mdgen.describe(built)["tables"][136]
            self.assertEqual(table["tlen"], tlen, "size=%r" % (size,))
            self.assertEqual(table["slots"][24], decl, "size=%r" % (size,))
        restored = mdgen.describe(
            ldmd.build(entry=64, atomic=True, size=216, restore=True))["tables"][136]
        self.assertEqual(restored["tlen"], 40, "the RESTORE table is what supplies Apple's tlen")
        # An unwitnessed size is refused rather than rescaled: there is no measured layout for it
        # and guessing one is what faulted the loader.
        for size in (180, 232, 240):
            with self.assertRaises(ValueError, msg="size=%d" % size):
                ldmd.build(entry=64, atomic=True, size=size)

    def test_the_216_declaration_equals_a_vendor_section_byte_for_byte(self):
        """The strongest case in the file: not "matches Apple's fields" but "is Apple's bytes".

        Twenty objects hold this section and all twenty are identical, so there is nothing to
        choose. It also pins the negative half - without `atomic` the same call reproduces the
        NON-declaring host section - so the four bytes are demonstrated to be the whole difference
        rather than asserted to be.
        """
        if not os.path.exists(WITNESS_216):
            self.skipTest("the vendor witness is absent: %s" % WITNESS_216)
        witness = witness_216_section()
        self.assertEqual(len(witness), 216)
        built = ldmd.build(entry=64, size=216, f0_5=32, restore=True, atomic=True)
        self.assertEqual(bytes(built), bytes(witness),
                         "%d bytes differ from the vendor section"
                         % sum(1 for a, b in zip(built, witness) if a != b))
        plain = ldmd.build(entry=64, size=216, f0_5=32, restore=True)
        differ = sorted(i for i in range(216) if plain[i] != witness[i])
        self.assertEqual(differ, [40, 48, 102, 154],
                         "the declaration is not the four measured bytes")

    def test_every_written_field_matches_a_vendor_section_of_the_SAME_size(self):
        """A layout belongs to a size, so the reference has to be a section of that size.

        The wrong version of this case compared BOTH sizes against Apple's 224-byte donor, which
        is how "the fields match Apple" stayed true while the 216-byte build pointed every field
        eight bytes past its data.
        """
        if not os.path.exists(DONOR) or not os.path.exists(WITNESS_216):
            self.skipTest("a vendor reference is absent")
        # The 216 arm is compared in its RESTORED form, because tlen is zero without it and
        # mdgen cannot infer a field's width from a table whose length is unset - it reported
        # every four-byte field as one byte, which is an artefact of the description and not a
        # difference in the bytes. The 224 arm needs no restore: it writes tlen itself.
        for size, reference, f0_5, kw in (
                (224, apple_t136().get("fields") or {}, 36, {}),
                (216, mdgen.describe(witness_216_section())["tables"][136].get("fields") or {},
                 32, dict(restore=True))):
            fields = mdgen.describe(ldmd.build(entry=64, atomic=True, f0_5=f0_5,
                                               size=size, **kw))["tables"][136]["fields"]
            self.assertTrue(fields, "size=%d produced no fields" % size)
            for slot, value in sorted(fields.items()):
                self.assertEqual(value, reference.get(slot), "size=%d slot=%d" % (size, slot))

    def test_slot_1_stays_omitted_and_t44_slot_2_is_written_because_every_witness_has_it(self):
        """The two slots this file used to refuse together, now separated by their measurements.

        Slot 1 stays out: a program fact with 218 distinct values in the atomic class, absent from
        every section this backend emits, and copying it from a donor is the donor copy mdgen's own
        comments refuse.

        t44 slot 2 is written, and the earlier refusal was wrong. It is present in 481 of 481
        declaring objects - 20 at 216 bytes and 461 at 224 - and it is one of the four bytes by
        which the vendor's 216-byte declaring section differs from this host's. It is NOT exclusive
        to atomics: 20,282 non-declaring 224-byte sections carry it too. So it is emitted because
        every witness that declares has it, and this case records that reason rather than the
        stronger claim that it declares anything.
        """
        for size in (216, 224):
            described = mdgen.describe(ldmd.build(entry=64, atomic=True, size=size))
            self.assertNotIn(1, described["tables"][136]["slots"], "size=%d" % size)
            self.assertEqual((described["tables"][44].get("fields") or {}).get(2), (4, 1, 1),
                             "size=%d" % size)
        # and the non-atomic path still omits it, so it tracks the declaration in OUR output
        self.assertNotIn(2, mdgen.describe(ldmd.build(entry=64))["tables"][44]["slots"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
