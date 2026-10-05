"""The measured constant program for the retained scalar requantization witness.

This is a structural codec, not a runtime admission shortcut.  The Apple probe publishes
``scale_bits[0]`` through a 64-byte constant program before its main code runs.  The bytes below
are assembled from the retained instruction records (two pointer loads, one load of the scale
word, the publish operation, ``end`` and the measured two-byte filler).  No field is inferred
from the ordinary uniform-preload class.

The main ``op3290/6`` scale operand is outside the generic authoring table.  The compiler therefore
selects it only inside the explicit measured requantization stage, where the complete main byte
sequence and this prologue are checked together; this module does not generalize the codec to
arbitrary scale expressions or other scalar programs.
"""

from __future__ import annotations

import hashlib


PROVENANCE_COMMIT = "da5efd9b"
WITNESS_OBJECT_SHA256 = "cb89057190252c1e27b49b360a465b1f8b0c9af7e73ecd63e3e5df55af53bfae"
WITNESS_METADATA_SHA256 = "27bada24513682d9adb60cd3acf0b592c442bfa26cbaf0482f6865d47d37b0dc"

ENTRY = 64
FILLER = bytes.fromhex("0600")

# Offsets 0..37 of reqz_round_probe/code.bin:
#   op14061/8, op14061/4, op12688/14, op592/8, END/4.
MEASURED_HEAD = bytes.fromhex(
    "248021104701a082"
    "1c8a0827"
    "0f000300804400a0410080000000"
    "2304070256a0a41a"
    "0e000000"
)

if len(MEASURED_HEAD) != 38:
    raise AssertionError("the retained requantization prologue head must be 38 bytes")

PROLOGUE = MEASURED_HEAD + FILLER * ((ENTRY - len(MEASURED_HEAD)) // len(FILLER))
PROLOGUE_SHA256 = "830c032dd18a1b0939d59da15a1c00584d0ac51d083452900b5f077b61a1260d"


def build(*, entry=ENTRY):
    """Return the measured prologue, refusing every unmeasured entry size."""
    if type(entry) is not int or entry != ENTRY:
        raise ValueError("requantization constant program is measured only at a 64-byte entry")
    out = bytes(PROLOGUE)
    if hashlib.sha256(out).hexdigest() != PROLOGUE_SHA256:
        raise AssertionError("the checked-in requantization prologue changed")
    return out


def validate_request(bindings, system_registers):
    """Validate the only binding and system-register set carried by this witness.

    The scale argument remains a declared read binding even though the constant program fetches
    its first word.  Keeping that declaration is part of the measured image contract.
    """
    got = tuple((int(index), int(offset), bool(written)) for index, offset, written in bindings)
    expected = ((0, 0, False), (1, 2, False), (2, 4, True))
    if got != expected:
        raise ValueError("requantization preload is measured only for bindings %s; got %s"
                         % (list(expected), list(got)))
    got_sr = tuple(system_registers or ())
    if got_sr != (160,):
        raise ValueError("requantization preload is measured only for system registers [160]; got %s"
                         % (list(got_sr),))


__all__ = [
    "PROVENANCE_COMMIT", "WITNESS_OBJECT_SHA256", "WITNESS_METADATA_SHA256",
    "ENTRY", "FILLER", "MEASURED_HEAD", "PROLOGUE", "PROLOGUE_SHA256",
    "build", "validate_request",
]
