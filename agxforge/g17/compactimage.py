"""Measured compact single-indexed-record image, with an explicit projection gate.

The four original-AIR native controls have identical structure and differ only
in register count and public index. This is sparse-use support, not a class for
sixteen active records. No witness bytes are read by the author.
"""
from collections.abc import Mapping
from . import abi as A, authorobj, ldmd, mdgen, model, schema as S

SINGLE_INDEXED_WRITTEN = dict(
    size=396,
    root=16, rvt=4, root_vlen=12, root_tlen=12,             # root table at 16, vtable at 4 (witness: describe()['root'] == 16)
    pk=124, pkvt=60, pk_vlen=64, pk_tlen=60,                # per-kernel table at 124, vtable at 60 (describe: tables[124])
    pk_slots={29: 4, 27: 8, 26: 12, 13: 16, 12: 20, 10: 24, 8: 28, 6: 32,
              3: 36, 4: 40, 2: 44, 1: 48, 16: 54, 15: 55, 0: 56},   # describe: tables[124]['slots']
    # fixed per-kernel scalars: slot 1 = 4 (class-measured, all four), slots 15/16 = 1 (all four),
    # slot 0 = the class witness's register count, OVERRIDDEN by the program's own (build(register_count=))
    pk_extra={1: ('<I', 4), 15: ('<B', 1), 16: ('<B', 1), 0: ('<I', 16)},
    vec_bind=300, bind=[(384, 372, 'written')],             # one record, shape 'written' = {index, written flag}, no offset field
    vec2=308, v2=[(356, 344, 'long'), (332, 322, 'short')], # the kind-6 and kind-3 slot-2 records (describe: vectors[308])
    v2_vals=[(6, 2, 2), (3, 2, None)],                      # kind 6: fields 2 and 3 = 2; kind 3: field 2 = 2 (block words = max offset + 2 = 2)
    vec0=196, v0=(216, 204), q=140,                         # v0 vector at 196 -> table 216 (vtable 204); Q = 140 (slots 6/8/10/12)
    name=(44, 48), cpname=(240, 244),                       # 'agc.main' at 48, 'agc.main.constant_program' at 244
    ptrs={13: 272, 27: 184, 29: 188},                       # slot 13 -> 272, slot 27 -> 184 (empty vector), slot 29 -> 188
    slot29_vector=188, slot29_counts=(1,), slot29_sets=((156,),),   # ONE entry, witnessed only for SR156 -> 0
    nametab=(36, 28, 8, 8, {1: 4}, {1: ('<I', 4)}),         # the kernel-name table at 36 (vtable 28), field 1 = 4
    fills=((20, '<I', 16), (272, '<I', 8)),                 # root slot 3 = 16; the four bytes at 272 (slot 13's target) = 8
)


def author_sections(code, contract, emission):
    """Author only a checked compact contract in the measured image domain."""
    def require(ok, reason):
        if not ok:
            raise authorobj.Missing("compact indexed class: " + reason)

    require(isinstance(contract, A.ProgramABI) and contract.resource_projection is not None,
            "a captured resource projection certificate is required")
    contract.check_code(code)
    require(len(contract.bindings) == 1, "exactly one active binding is required")
    b = contract.bindings[0]
    public = next(d for d in contract.resource_projection.declarations if d.index == b.index)
    require(public.element_type == b.element_type and (b.index, b.element_type, b.element_bytes) in
            ((7, "ushort", 2), (8, "float2", 8), (13, "uint4", 16)),
            "the measured index/type pairs are 7/ushort, 8/float2 and 13/uint4")
    require(b.offset == 0 and b.written, "the indexed record must be written at pointer offset zero")
    require(contract.entry == 64 and contract.prologue == b"\x0e\x00\x00\x00" + b"\x06\x00" * 30,
            "END-only entry 64 is required")
    require(not contract.uses_threadgroup and contract.threadgroup is None and
            contract.resources is None and contract.execution is None and
            contract.spill_state is None and contract.promoted_ranges is None and
            contract.constant_pool is None and contract.constant_program_sha256 is None,
            "no internal resources, scratch, promotion, pool or preloads are measured")
    state = contract.argument_state
    require(state is not None and state.pointer_offsets == ((b.index, 0),) and
            state.pointer_words == 2 and state.block_words == 2 and state.block_bytes == 8 and
            state.indices_contiguous and state.not_stated == ("per_kernel_slot_1",),
            "the compiler must state one pointer at offset zero and leave slot 1 unstated")
    require(contract.arch_flag is False, "only the measured false ARCH form is admitted")
    require(contract.exact_grid_required is True and not contract.writes_texture and
            emission.get("launch", {}).get("bounds_checked") is False,
            "the caller must satisfy an exact grid; no bounds-check or texture claim is admitted")
    allowed = {"abi_version", "arch_flag", "bindings", "entry", "forms", "has_stores", "launch",
               "main_instruction_count", "pk_extra", "pk_values", "profile", "prologue",
               "register_count", "system_registers", "uses_threadgroup", "writes_buffer",
               "writes_texture", "argument_state", "instruction_count", "has_back_edge"}
    require(not (set(emission) - allowed), "unmeasured emission fields/options: " + repr(sorted(set(emission) - allowed)))
    require(type(emission.get("abi_version")) is int and emission["abi_version"] == 3 and
            emission.get("profile") is None, "ordinary ABI v3 without a profile is required")
    require(isinstance(emission.get("system_registers"), (list, tuple)) and
            list(emission["system_registers"]) == [156] and
            all(type(r) is int for r in emission["system_registers"]), "system-register set [156] is required")
    require(isinstance(emission.get("pk_extra"), (list, tuple)) and
            isinstance(emission.get("pk_values"), Mapping) and
            list(emission["pk_extra"]) == [15, 16] and
            {str(k): v for k, v in emission.get("pk_values", {}).items()} == {"15": 1, "16": 1} and
            all(type(v) is int for v in emission["pk_values"].values()),
            "per-kernel slots 15 and 16 must each be stated as 1")
    instructions = list(model.decode(code, 0))
    require([(i.offset, len(i.raw), i.opcode.id) for i in instructions] ==
            [(i.offset, i.length, i.opcode) for i in contract.instructions],
            "typed instruction records must equal the decoded code")
    # This is a conservative bound of this four-target admission, not a claim
    # that instruction count determines this metadata class or slot 32 generally.
    require(1 <= len(instructions) <= 89 and not emission.get("has_back_edge"),
            "the initial straight-line admission is bounded to 89 main instructions")
    require(emission.get("main_instruction_count") == len(instructions),
            "main instruction count must describe the emitted program")
    names = model.registers()
    sr = [i for i in instructions if i.opcode.id == 14059]
    require(bool(sr) and all(len(i.raw) == 4 and len(i.values) == 4 and
            i.values[2][0] == "reg" and names.get(i.values[2][1]) == "SR_TG_X" for i in sr),
            "decoded system-register reads must name SR_TG_X")
    count = authorobj._register_count(emission)
    require(count <= 0xffffffff, "register count must fit its uint32 field")
    metadata = bytes(mdgen.build([b.index], offsets=[0], layout=SINGLE_INDEXED_WRITTEN,
                                system_registers=(156,), register_count=count))
    # Structurally serialize the present-empty-subtable ARCH observed in all four
    # controls. Do not substitute the older null-reference scalar route.
    sub = S.Table({}, {}, 4, 4, name="archsub")
    root = S.Table({0: 4}, {0: ("<I", S.Ref(sub))}, 8, 8, name="archroot")
    doc = S.Doc(root, [root, sub]); S.place(doc)
    sections = dict(zip(authorobj.SECTIONS, (metadata, ldmd.build(64, restore=True),
                                           S.emit(doc, size=32), bytes(96), b"")))
    ledger = {
        "metadata class": "MEASURED compact single-indexed-record class, 396 bytes; one written active record, SR156",
        "__GPU_METADATA": "GENERATED from the measured layout and this program's index and register count; slot 1 is class-measured 4",
        "__GPU_LD_MD": "GENERATED 216-byte restored layout, equal to all four compact native controls",
        "__GPU_ARCH_LD_MD": "GENERATED present empty subtable, equal to all four compact native controls",
        "__GPU_STATS_MD": "96 zero bytes under the existing author policy; no new inertness measurement",
        "__GPU_REMARKS_MD": "empty under the existing author policy",
        "public declarations": "retained in the typed projection certificate; only the active emission map has image records",
        "execution evidence": "NOT ESTABLISHED by authoring; loader and hardware acceptance are separate",
    }
    return sections, ledger
