"""Measured metadata placement expressed as nodes in the common serializer."""
from __future__ import annotations

from . import mdgen as M
from . import schema as S
import struct


def select(bindings, abi):
    """Select a measured device-buffer class from an explicit compiler contract.

    No program name, kernel shape, code hash or profile label participates. The
    ABI v3 states the system registers: the three-binding class is measured for
    the thread-id register, and an opcode name alone cannot state that fact.
    """
    prologue = abi.get("prologue")
    if isinstance(prologue, str):
        try:
            prologue = bytes.fromhex(prologue)
        except ValueError:
            return None
    signature = tuple(tuple(b[:3]) for b in bindings)
    version = abi.get("abi_version")
    if (version not in (2, 3) or abi.get("entry") != 64 or
        prologue != bytes.fromhex("0e000000") + bytes.fromhex("0600")*30 or
        abi.get("uses_threadgroup") is not False or abi.get("writes_texture") is not False or
        abi.get("writes_buffer") is not True or abi.get("has_stores") is not True or
        type(abi.get("arch_flag")) is not bool or
        tuple(abi.get("pk_extra", ())) != (15, 16) or abi.get("pk_values") != {15: 1, 16: 1}):
        return None
    if signature == ((1, 0, False), (2, 2, True)):
        layout = M.SCALAR
    elif signature == ((1, 0, False), (2, 2, False), (3, 4, True)) and version == 3:
        layout = M.SEPARATE
    else:
        return None
    # WHICH REGISTER SETS A CLASS MAY DECLARE DEPENDS ON THE CLASS, not on the ABI version. Only
    # the three-binding class carries a slot-29 vector, so only it can name a second register; the
    # scores and context stages of the attention block need (160, 161) and are measured for it.
    # SCALAR emits no such vector, so a second register there has nowhere measured to go and stays
    # refused rather than being dropped silently.
    declared = tuple(abi.get("system_registers") or ())
    if version == 3:
        if layout is M.SEPARATE:
            try:
                M.slot29_entries(declared)
            except ValueError:
                return None
        elif declared != (160,):
            return None
    # Explicit extra linker inputs must never be silently dropped by selection.
    overrides = {"slot2_kind6", "slot2_kind6_f2", "slot2_extra", "constant_program",
                 "v0_field2", "v0_field3", "pk_vectors", "pk_empty_vectors"}
    if overrides.intersection(abi):
        return None
    if "pk_slot1" in abi and abi["pk_slot1"] != layout["pk_extra"].get(1, ("<I", 0))[1]:
        return None
    return layout


SLOT29_AT = 188          # measured position of the slot-29 vector in this class


def graph(layout, bindings, register_count=None, system_registers=None):
    """Construct measured nodes and resolve references through g17schema.emit.

    The layout states positions, record shapes, inline lengths, trailing absent
    vtable slots and zeroed fields. No bytes are copied from an object or patched
    after serialization. The unswept three-binding class retains its explicit
    vectors, strings and constant-program record, even when that program is END.
    """
    # THE DECLARED REGISTER SET WIDENS THIS CLASS TOO. The slot-29 vector holds one entry per
    # declared system register, so a second one adds four bytes and moves every structure laid out
    # after it. The addresses below are measured for the one-register form; `moved` re-derives them
    # rather than a second class being transcribed. The law and the register-to-entry map live in
    # g17mdgen so this path and the four-buffer path cannot drift apart.
    entries = M.slot29_entries(system_registers) if system_registers is not None else [80]
    extra = 4 * (len(entries) - 1)
    if extra:
        layout = M.with_system_registers(layout, len(entries), vector_at=SLOT29_AT)

    def moved(address):
        return address + extra if address >= SLOT29_AT + 8 else address

    if len(bindings) != len(layout["bind"]):
        raise ValueError("binding count differs from the measured class")
    nodes = []

    def table(name, address, vtable, slots, fields, inline, vlen):
        if address-vtable != vlen:
            raise ValueError("measured vtable must abut its table")
        extent = max([inline, 4] + [off+struct.calcsize(fields.get(slot, ("<I", 0))[0])
                                    for slot, off in slots.items()])
        t = S.Table(slots, fields, extent, inline, name=name, vtable_length=vlen)
        t.addr = address
        nodes.append(t)
        return t

    records = []
    for spec, binding in zip(layout["bind"], bindings):
        index, offset, written = binding[:3]
        address, vtable, shape = spec[:3]
        vlen, inline, slots = M._REC[shape]
        if len(spec) > 3:
            inline = spec[3]
        fields = {0: ("<B", 5)}
        for slot, value, fmt in ((1, index, "<I"), (2, offset, "<I"), (3, int(written), "<B")):
            if slot in slots:
                fields[slot] = (fmt, value)
            elif value:
                raise ValueError(f"measured binding shape {shape} cannot represent field {slot}={value}")
        records.append(table(f"binding{index}", address, vtable, slots, fields, inline, vlen))
    binding_vector = S.Vector([S.Ref(t) for t in records], name="bindings")
    binding_vector.addr = layout["vec_bind"]
    nodes.append(binding_vector)
    resources = []
    pointer_blocks = []
    for spec, values in zip(layout["v2"], layout["v2_vals"], strict=True):
        address, vtable, shape = spec
        vlen, inline, slots = M._V2REC[shape]
        fields = {0: ("<B", values[0]), 2: ("<I", values[1])}
        if 3 in slots:
            fields[3] = ("<I", values[2])
        if values[0] == 3:
            pointer_blocks.append(values[1])
        resources.append(table(f"resource{values[0]}", address, vtable, slots, fields, inline, vlen))
    if pointer_blocks != [max(b[1] for b in bindings)+2]:
        raise ValueError("measured pointer block differs from compiler resource offsets")
    resource_vector = S.Vector([S.Ref(t) for t in resources], name="resources")
    resource_vector.addr = layout["vec2"]
    nodes.append(resource_vector)
    fields = {3: ("<I", 4*pointer_blocks[0]), 4: ("<I", S.Ref(binding_vector)),
              2: ("<I", S.Ref(resource_vector))}
    fields.update(layout["pk_extra"])
    # SLOT 0 IS THE PROGRAM'S REGISTER COUNT, NOT THE CLASS'S. pk_extra carries the value measured
    # from this class's witness, which is right for that one object and wrong for every other
    # program in the class. A caller that knows this program's count overrides it.
    if register_count is not None:
        if 0 not in fields:
            raise ValueError("this measured class has no per-kernel slot 0 to carry a register count")
        fields[0] = ("<I", register_count)
    name_table = None
    if layout is M.SEPARATE or layout.get("_widened_from") is M.SEPARATE:
        # Positions and values are measured in separate-indexed and independently
        # corroborated by renumbered-indexed. Preserve the complete layout rather
        # than erasing references based on the absence of constant-program work.
        main_name = S.String("agc.main", name="main_name")
        main_name.addr = moved(44)
        nodes.append(main_name)
        name_table = table("entry_name", moved(36), moved(28), {1: 4}, {1: ("<I", S.Ref(main_name))}, 8, 8)
        cp_name = S.String("agc.main.constant_program", name="constant_program_name")
        cp_name.addr = moved(240)
        cp_words = S.Words([], name="constant_program_words", width=4)
        cp_words.addr = moved(236)
        nodes.extend([cp_name, cp_words])
        cp = table("constant_program", moved(216), moved(204), M.V0REC[2],
                   {0: ("<I", S.Ref(cp_name)), 1: ("<B", 3), 2: ("<I", 1),
                    3: ("<I", S.Ref(cp_words))}, 20, 12)
        cp_vector = S.Vector([S.Ref(cp)], name="constant_programs")
        cp_vector.addr = moved(196)
        nodes.append(cp_vector)
        fields[26] = ("<I", S.Ref(cp_vector))
        vectors = {29: (SLOT29_AT, 4, entries), 27: (184, 4, []), 13: (272, 1, [0]*8),
                   12: (284, 4, []), 10: (288, 4, []), 8: (292, 4, []), 6: (296, 4, [])}
        vectors = {slot: (moved(address), width, values)
                   for slot, (address, width, values) in vectors.items()}
        for slot, (address, width, values) in vectors.items():
            vector = S.Words(values, name=f"slot{slot}", width=width)
            vector.addr = address
            nodes.append(vector)
            fields[slot] = ("<I", S.Ref(vector))
    pk = table("perkernel", layout["pk"], layout["pkvt"], layout["pk_slots"], fields,
               layout["pk_tlen"], layout["pk_vlen"])
    root_fields = {0: ("<I", S.Ref(pk))}
    if name_table is not None:
        root_fields[3] = ("<I", S.Ref(name_table))
    root = table("root", layout["root"], layout["rvt"], {0: 8, 3: 4},
                 root_fields, layout["root_tlen"], layout["root_vlen"])
    return S.Doc(root, nodes, size=layout["size"])


def emit(layout, bindings, ledger, register_count=None, system_registers=None):
    result = S.emit(graph(layout, bindings, register_count, system_registers))
    if layout is M.SEPARATE:
        ledger["metadata class"] = "MEASURED: unswept three-device-buffer indexed class; ABI v3 system registers (160,)"
        ledger["metadata serializer"] = "g17schema.emit: all 456 bytes match the retained indexed three-buffer object"
        ledger["slot-2 kind-6 record"] = "MEASURED CLASS: kind 6 fields (2, 6), kind 3 pointer block 6"
        ledger["constant-program record"] = "MEASURED PRESENT: an END-only constant program still has a descriptor in unswept objects"
        ledger["v0 fields"] = "MEASURED CLASS: kind 3, field 2 = 1, field 3 empty; reference to this object's constant-program symbol"
        ledger["constant and system vectors"] = "MEASURED CLASS: slot 13 has eight zero bytes; slot 27 empty; slot 29 [80] for the supported thread-id system value"
        ledger["empty vectors"] = "MEASURED CLASS: distinct empty vectors at slots 6, 8, 10 and 12"
        ledger["per-kernel slot 1"] = "MEASURED CLASS: 8, equal to kind-6 fields 2 + 6"
        ledger["per-kernel write flags"] = "ABI INPUT: slots 15 and 16 are 1, matching the measured class"
        return result
    ledger["metadata class"] = "MEASURED: executed two-device-buffer class; selected from explicit ABI resources and prologue"
    ledger["metadata serializer"] = "g17schema.emit: measured placement and trailing absent vtable slots"
    ledger["slot-2 kind-6 record"] = "MEASURED CLASS: kind 6 fields (20, 4), followed by kind 3 pointer block 4"
    ledger["constant-program record"] = "MEASURED EMPTY; compiler prologue is END plus filler"
    ledger["empty vectors"] = "MEASURED SWEPT: slots 6, 8, 10, 13 and 26; no slot 12"
    ledger["per-kernel slot 1"] = "MEASURED SWEPT: present and zero, not the sum of kind-6 fields"
    ledger["per-kernel write flags"] = "MEASURED SWEPT: flags 15 and 16 absent; resource records encode the ABI write binding"
    return result
