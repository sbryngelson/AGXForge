#!/usr/bin/env python3
"""__GPU_METADATA AS AN OBJECT GRAPH, serialised by a placer rather than by offset arithmetic.

The metadata was a table of measured positions, then a table of positions plus equations like
"binding vector = per-kernel table + Q + 36". That equation is right for 7,519 of the 7,557 cached
objects and wrong for the rest, and the reason is that it folds in an accident: where the serialiser
happened to put slot 4. What the format actually says is

    a reference stores TARGET ADDRESS - THE ADDRESS OF THE FIELD THAT HOLDS IT

so every one of those relations is the same relation, and the exceptions stop being exceptions.

This module models the document as typed nodes - tables, vectors, records - with references between
them, and lets a placer decide addresses. Nothing here knows where anything goes; that is the point.
Two consequences the offset formulas could not give:

    a document can be laid out MANY WAYS and mean the same thing, which is a test (g17reloc.py)
    a signature this project has never packaged gets a layout by construction rather than by
    someone measuring an object that has one
"""
import struct


class Ref:
    """A reference to another node, written relative to the address of the field holding it."""
    __slots__ = ("target",)

    def __init__(self, target):
        self.target = target


class Node:
    __slots__ = ("addr", "align", "name")

    def __init__(self, name="", align=4):
        self.addr, self.align, self.name = None, align, name


class Table(Node):
    """A FlatBuffers table: a vtable giving each slot's offset, abutted by the table body.

    `fields` maps slot -> (format, value); a value may be a Ref. `size` is the body's length and
    `tlen` the length the vtable DECLARES, which Apple does not always set to the body's size - the
    per-kernel table declares zero - so it is carried rather than computed.
    """
    __slots__ = ("slots", "fields", "size", "tlen", "shares", "declared_vlen")

    def __init__(self, slots, fields, size, tlen=None, name="", align=4, vtable_length=None):
        Node.__init__(self, name, align)
        # ANOTHER TABLE WHOSE VTABLE THIS ONE USES. Apple shares vtables between records with the
        # same slot map and the same declared length, and the sharer may sit BEFORE the vtable it
        # uses - ac2-128x32x64's third binding record is at 416 with a soffset of -16, reaching
        # forward to the vtable at 432 that the second record also uses. A table that shares emits
        # no vtable of its own and occupies only its body.
        self.shares = None
        self.slots, self.fields = dict(slots), dict(fields)
        minimum_vlen = 4 + 2 * (max(self.slots) + 1) if self.slots else 4
        if vtable_length is not None and (type(vtable_length) is not int or
                vtable_length < minimum_vlen or vtable_length > 65534 or vtable_length % 2):
            raise ValueError("declared vtable length must include every slot and fit an even uint16")
        self.declared_vlen = vtable_length
        self.tlen = size if tlen is None else tlen
        # THE BODY IS AT LEAST THE DECLARED INLINE LENGTH. Apple declares more than it writes -
        # a binding record with a sixteen-byte body declares eighteen - and whatever reads that
        # length reads past the body. Packed tight, that lands on the next node's vtable and
        # segfaults the loader; with sixteen bytes of slack between nodes the same graph runs.
        self.size = max(size, self.tlen)

    @property
    def vlen(self):
        """The SOFFSET written at the table's address: addr - vtable addr, negative when the
        vtable is ahead. For a table with its own vtable this is also the vtable's length."""
        if self.shares is not None:
            return self.addr - (self.shares.addr - self.shares.own_vlen)
        return self.own_vlen

    @property
    def own_vlen(self):
        if self.declared_vlen is not None:
            return self.declared_vlen
        return 4 + 2 * (max(self.slots) + 1) if self.slots else 4

    @property
    def total(self):
        return (self.size if self.shares is not None else self.own_vlen + self.size)


class Vector(Node):
    """A vector of references: a count, then one reference per element."""
    __slots__ = ("items",)

    def __init__(self, items, name="", align=4):
        Node.__init__(self, name, align)
        self.items = list(items)

    @property
    def total(self):
        return 4 + 4 * len(self.items)


class String(Node):
    """A FlatBuffers string: a four-byte length, the bytes, and a terminating NUL.

    Apple's __GPU_METADATA carries two of them - `agc.main` and `agc.main.constant_program`, the
    same symbols this project's own object declares - and until this node existed the authoring
    path had no way to express one. It emitted an INTEGER into the field that references the
    constant-program name, which happened to equal the offset Apple's layout put there, so the
    section came back 48 bytes short with the field pointing at nothing.
    """
    __slots__ = ("text",)

    def __init__(self, text, name="", align=4):
        Node.__init__(self, name, align)
        self.text = text.encode() if isinstance(text, str) else bytes(text)

    @property
    def total(self):
        return 4 + len(self.text) + 1


class Words(Node):
    """A vector of raw BYTES: a count, then that many bytes - no references, and one byte each.

    THE WIDTH IS PER SLOT AND THE ARITHMETIC SETTLES IT, one node at a time, from the distance to
    the next node. Slot 13 is BYTES: a4-load-uni puts its vector at 272 with a length word of 8 and
    the next node - slot 12's vector - at 284, and 284 - 272 - 4 is 8. Slot 29 is WORDS:
    a4-fetchadd-uni puts its vector at 192 with a length word of 2 and the next node at 204, which
    is 4 + 2*4 and not 4 + 2.

    BOTH WRONG WIDTHS PARSE AND BOTH MOVE EVERY LATER NODE. Four bytes per element everywhere made
    the slot-13 sections 24 to 36 bytes long; one byte per element everywhere made the slot-29
    sections four short. A width is a measurement per slot, not a convention.

    Per-kernel slots 13, 27 and 29 hold these. They read as vectors and every earlier reader
    treated their members as offsets to tables, which is why they looked unmodellable: in
    fill-h-100 slot 13's four words are 1065360766 and its siblings, float constants, and as
    offsets they point a gigabyte outside the section. They are the kernel's own data.
    """
    __slots__ = ("values", "width")

    def __init__(self, values, name="", align=4, width=1):
        Node.__init__(self, name, align)
        self.values = [int(v) for v in values]
        self.width = width

    @property
    def total(self):
        return 4 + self.width * len(self.values)


class Doc:
    """The whole section: a root reference and the nodes reachable from it."""

    def __init__(self, root, nodes, size=None, root_at=0):
        self.root, self.nodes, self.size, self.root_at = root, list(nodes), size, root_at


def place(doc, start=4, gap=0, order=None, reserve=None):
    """Assign an address to every node. `order` reorders placement; `gap` pads between nodes.

    Both exist so the same graph can be laid out differently on purpose - if two legal placements
    do not parse to the same meaning, something is reading a fixed offset.
    """
    nodes = list(order) if order else list(doc.nodes)
    reserve = reserve or {}
    cur = start
    for n in nodes:
        extra = reserve.get(n.name, 0)
        if isinstance(n, Table):
            if n.shares is not None:            # no vtable of its own; the body is all it occupies
                cur = (cur + n.align - 1) & ~(n.align - 1)
                n.addr = cur
                cur = n.addr + n.size + gap + extra
            else:
                # the vtable must abut the body, and the BODY carries the aligned scalars
                cur = ((cur + n.own_vlen + n.align - 1) & ~(n.align - 1)) - n.own_vlen
                n.addr = cur + n.own_vlen
                cur = n.addr + n.size + gap + extra
        else:
            cur = (cur + n.align - 1) & ~(n.align - 1)
            n.addr = cur
            cur = n.addr + n.total + gap + extra
    return (cur + 15) & ~15


def emit(doc, size=None):
    """The bytes. Every reference is resolved AFTER placement, from the field's own address."""
    end = max((n.addr + (n.size if isinstance(n, Table) else n.total)) for n in doc.nodes)
    total = size or doc.size or ((end + 15) & ~15)
    b = bytearray(total)
    struct.pack_into("<I", b, doc.root_at, doc.root.addr)
    for n in doc.nodes:
        if isinstance(n, Table):
            struct.pack_into("<i", b, n.addr, n.vlen)
            if n.shares is not None:
                for slot, (fmt, val) in n.fields.items():
                    at = n.addr + n.slots[slot]
                    struct.pack_into(fmt, b, at,
                                     (val.target.addr - at) if isinstance(val, Ref) else val)
                continue
            vt = n.addr - n.own_vlen
            # A VTABLE'S THREE FIELDS ARE ALL SIXTEEN BITS - the vtable length, the declared inline
            # length, and every slot offset - so a value that does not fit is not a big number, it
            # is a layout that cannot exist. Letting struct.pack_into raise reports it as
            # "'H' format requires 0 <= number <= 65535" with no table, no slot and no cause, which
            # is what 141 acceleration-structure sections looked like from the authoring census:
            # an opaque crash where every other refusal in that run names itself.
            for what, val in (("vtable length", n.own_vlen), ("declared length", n.tlen)):
                if not 0 <= val <= 0xFFFF:
                    raise ValueError("%s's %s is %d, and a vtable field is 16 bits"
                                     % (n.name or "a table", what, val))
            struct.pack_into("<HH", b, vt, n.own_vlen, n.tlen)
            for slot, off in n.slots.items():
                if not 0 <= off <= 0xFFFF:
                    raise ValueError("%s slot %d is at offset %d, and a vtable slot offset is "
                                     "16 bits; this layout is not representable"
                                     % (n.name or "a table", slot, off))
                struct.pack_into("<H", b, vt + 4 + 2 * slot, off)
            for slot, (fmt, val) in n.fields.items():
                at = n.addr + n.slots[slot]
                struct.pack_into(fmt, b, at, (val.target.addr - at) if isinstance(val, Ref) else val)
        elif isinstance(n, Words):
            struct.pack_into("<I", b, n.addr, len(n.values))
            for k, w in enumerate(n.values):
                at = n.addr + 4 + n.width * k
                if n.width == 1:
                    b[at] = w & 0xFF
                else:
                    struct.pack_into("<I", b, at, w)
        elif isinstance(n, String):
            struct.pack_into("<I", b, n.addr, len(n.text))
            b[n.addr + 4:n.addr + 4 + len(n.text)] = n.text          # the NUL is already zero
        else:
            struct.pack_into("<I", b, n.addr, len(n.items))
            for k, it in enumerate(n.items):
                at = n.addr + 4 + 4 * k
                struct.pack_into("<I", b, at, it.target.addr - at)
    return bytes(b)
