"""Structural handoff for the measured scalar requantization metadata class.

This is a description of Apple's retained ``reqz_round_probe`` section, not a donor byte blob.
``mdgen.build_from`` writes the section from the walked tables and vectors.  The class is kept
separate from the public authoring route until a compiler body using this exact three-buffer shape
and a two-dispatch common-runtime boundary have both been measured.

Measured witness (linker/g17-tensorops-recon, sections 106--110):

* metadata size 476, SHA-256
  ``27bada24513682d9adb60cd3acf0b592c442bfa26cbaf0482f6865d47d37b0dc``;
* records ``(0, 0, read), (1, 2, read), (2, 4, written)``;
* system register ``SR160`` (``thread_position_in_grid``);
* source and scale are 32-bit read-only values and the destination is the third record.

The layout below is intentionally expressed as tables, vectors, field widths and tails.  It is
not a copy of the witness bytes, and changing an unmeasured field is impossible through the
public helper.  Only the program register count is a per-program override; all other values stay
at the measured class values until a new class is established.
"""

from __future__ import annotations

from . import mdgen


PROVENANCE_COMMIT = "da5efd9b"
WITNESS_METADATA_SHA256 = "27bada24513682d9adb60cd3acf0b592c442bfa26cbaf0482f6865d47d37b0dc"
WITNESS_OBJECT_SHA256 = "cb89057190252c1e27b49b360a465b1f8b0c9af7e73ecd63e3e5df55af53bfae"
WITNESS_RECORDS = ((0, 0, False), (1, 2, False), (2, 4, True))
WITNESS_SYSTEM_REGISTERS = (160,)


def _table(vtpos, vlen, tlen, slots, fields, tail=b""):
    return {"vtpos": vtpos, "vlen": vlen, "tlen": tlen, "shared": False,
            "slots": dict(slots), "fields": dict(fields), "tail": bytes(tail)}


# This is the output of mdgen.describe on the retained 476-byte Apple section, transcribed as
# structure.  The serializer test below is the guard against an accidental donor-style edit.
SCALAR_476_LAYOUT = {
    "size": 476,
    "root": 16,
    "tables": {
        16: _table(4, 12, 12, {0: 8, 3: 4}, {0: (8, 4, 104), 3: (4, 4, 16)}),
        128: _table(60, 68, 64,
                    {0: 60, 1: 48, 2: 44, 3: 36, 4: 40, 6: 32, 8: 28, 10: 24,
                     12: 20, 13: 16, 15: 59, 16: 58, 26: 12, 27: 8, 29: 4, 31: 52},
                    {0: (60, 4, 2), 1: (48, 4, 8), 2: (44, 4, 168), 3: (36, 4, 24),
                     4: (40, 4, 156), 6: (32, 4, 160), 8: (28, 4, 160),
                     10: (24, 4, 160), 12: (20, 4, 160), 13: (16, 4, 156),
                     15: (59, 1, 1), 16: (58, 1, 1), 26: (12, 4, 68),
                     27: (8, 4, 56), 29: (4, 4, 68), 31: (52, 4, 32)},
                    bytes.fromhex("01000000010000000100000050000000")),
        36: _table(28, 8, 8, {1: 4}, {1: (4, 4, 4)},
                   b"\x08\x00\x00\x00agc.main\x00\x00\x00\x00"),
        228: _table(216, 12, 20, {0: 16, 1: 15, 2: 8, 3: 4},
                    {0: (16, 4, 24), 1: (15, 1, 3), 2: (8, 4, 2), 3: (4, 4, 16)},
                    bytes.fromhex("040000000400000005000000060000000700000019000000")
                    + b"agc.main.constant_program\x00\x00\x00\x04\x00\x00\x00"
                    + bytes.fromhex("80ffffff00000000000000000000000000000000")),
        468: _table(462, 6, 8, {0: 7}, {0: (7, 1, 5)}),
        444: _table(434, 10, 18, {0: 11, 1: 4, 2: 12},
                    {0: (11, 1, 5), 1: (4, 4, 1), 2: (12, 4, 2)}),
        416: _table(404, 12, 18, {0: 10, 1: 4, 2: 12, 3: 11},
                    {0: (10, 1, 5), 1: (4, 4, 2), 2: (12, 4, 4), 3: (11, 1, 1)}),
        388: _table(376, 12, 16, {0: 11, 2: 12, 3: 4},
                    {0: (11, 1, 6), 2: (12, 4, 1), 3: (4, 4, 7)}),
        364: _table(354, 10, 12, {0: 7, 2: 8},
                    {0: (7, 1, 3), 2: (8, 4, 6)}),
    },
    "vectors": {208: [228], 324: [468, 444, 416], 340: [388, 364]},
    "order": [16, 128, 36, 228, 468, 444, 416, 388, 364],
    "extra": {},
}


def validate_request(bindings, system_registers):
    """Validate the only binding/system-register set this class has measured."""
    got = tuple((int(i), int(o), bool(w)) for i, o, w in bindings)
    if got != WITNESS_RECORDS:
        raise ValueError("requant scalar metadata is measured only for bindings %s; got %s"
                         % (list(WITNESS_RECORDS), list(got)))
    if tuple(system_registers or ()) != WITNESS_SYSTEM_REGISTERS:
        raise ValueError("requant scalar metadata is measured only for system registers %s; got %s"
                         % (list(WITNESS_SYSTEM_REGISTERS), list(system_registers or ())))


def build(register_count=2):
    """Serialize the measured 476-byte class, optionally overriding slot-0 register count."""
    if type(register_count) is not int or register_count < 1:
        raise ValueError("register count must be a positive integer")
    values = {(128, 0): register_count}
    return mdgen.build_from(SCALAR_476_LAYOUT, values=values)


__all__ = ["PROVENANCE_COMMIT", "WITNESS_METADATA_SHA256", "WITNESS_OBJECT_SHA256",
           "WITNESS_RECORDS", "WITNESS_SYSTEM_REGISTERS", "SCALAR_476_LAYOUT",
           "validate_request", "build"]
