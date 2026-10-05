"""Admission: every store's encoded binding rank must resolve to a declared binding.

THE DEFECT THIS EXISTS FOR HAS HAPPENED TWICE. A store encodes which buffer it
writes as a rank - a position in the program's binding list - and both the
ordinary store and, later, the range store inherited that field from a template
instead of stating it, fixing every emitted store at rank 1. Both times the
delivered kernel wrote rank 1 and was right by accident, so twenty end-to-end
kernels and a hardware campaign passed with the defect present. Nothing caught
it because nothing compared the two sides: the image's binding records were
correct, the code targeted a buffer those records did not name, and no check on
either side read both.

This reads the rank out of the delivered bytes and resolves it against the
bindings the image declares. It needs no device and no decoder agreement.

FIELD, measured: byte 1 bits [6:2], width 5, value = rank (g17asm.STORE_CONST).
One encoder path serves op17229, op17244, op17253 and op17262 at both 8 and 14
bytes; ranks 0..4 through all four opcodes give byte1 = 4*rank in all forty
cells with nothing else in the byte moving.

SCOPE of the field's MEANING. Hardware has read byte1[6:2] as the binding rank
on the ordinary store op17229 at rank 3 (LayerNorm, cooperative) and rank 4
(five-buffer residual LayerNorm), so more than one value is measured there. The
three range opcodes have executed only at rank 1. For those, this check rests on
the shared encoder and byte identity, not on execution at a second rank. A range
store executing at another rank would close that.

LIMIT, stated because it is inherent. The check compares the encoded rank to the
declared bindings; it cannot see a defect whose inherited value happens to be
correct. Both historical instances were invisible on their two-buffer rank-1
deliveries for exactly that reason, and both would have been caught the moment a
program stored to any other rank - which is also the moment they would have
written the wrong buffer. It catches the harmful case, not the dormant one.

The decoder's token stream does not expose this field as an immediate, so the
bytes are read directly. Comparing decoded immediates sees no movement and reads
as "no rank encoded" on a program that encodes one.
"""
import g17packedcheck as D

STORE_FORMS = (17229, 17244, 17253, 17262)
RANK_BYTE, RANK_LOW, RANK_WIDTH = 1, 2, 5
MEASURED_AT_SECOND_RANK = (17229,)


def store_ranks(code):
    """Every store in `code` with the binding rank its bytes encode."""
    found = []
    for offset, size, opcode, _ in D.decode(code):
        if opcode not in STORE_FORMS:
            continue
        byte = code[offset + RANK_BYTE]
        rank = (byte >> RANK_LOW) & ((1 << RANK_WIDTH) - 1)
        found.append(dict(offset=offset, opcode=opcode, length=size, rank=rank, byte1=byte))
    return found


def check(code, bindings):
    """Refuse a store whose encoded rank names no declared writable binding."""
    declared = [tuple(b[:3]) for b in bindings]
    findings = []
    resolved = []
    for store in store_ranks(code):
        rank = store["rank"]
        if rank >= len(declared):
            findings.append(dict(store, reason="encoded rank %d names no declared binding; the "
                                 "image declares %d" % (rank, len(declared))))
            continue
        index, _offset, written = declared[rank]
        if not written:
            findings.append(dict(store, reason="encoded rank %d resolves to binding index %d, "
                                 "which the image declares read-only" % (rank, index)))
            continue
        resolved.append(dict(store, binding_index=index))
    return dict(status="refused" if findings else "passed",
                stores=len(resolved) + len(findings), findings=findings,
                resolved=[(r["offset"], r["opcode"], r["rank"], r["binding_index"]) for r in resolved],
                scope="Encoded store rank against declared bindings, from delivered bytes. The "
                      "field's meaning is executed on %s at more than one rank; the range forms "
                      "share its encoder and byte layout and have executed only at rank 1."
                      % ", ".join("op%d" % o for o in MEASURED_AT_SECOND_RANK))
