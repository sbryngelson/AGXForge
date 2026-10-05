"""Explicit compatibility ABI for the previously executed scalar metadata class.

Emission uses recovered tables. Verification uses a separate frozen measurement,
including all five metadata sections, and reads the delivered object's entry.
This is a class contract, not a general formula for arbitrary kernel metadata.
"""
import hashlib
import json
from pathlib import Path
import struct

PROFILE = "scalar-buffer-two-bindings-measured-v2"
REFERENCE = Path(__file__).resolve().parents[1] / "isa/g17-scalar-abi-v2.json"
BINDINGS = [(1, 0, False), (2, 2, True)]
ENTRY = 64


def reference_sections():
    with REFERENCE.open() as f:
        doc = json.load(f)
    if doc["profile"] != PROFILE:
        raise ValueError("scalar ABI reference has the wrong version")
    out = {}
    for name, record in doc["sections"].items():
        b = bytes.fromhex(record["hex"])
        if len(b) != record["size"] or hashlib.sha256(b).hexdigest() != record["sha256"]:
            raise ValueError("scalar ABI reference is inconsistent: " + name)
        out[name] = b
    return out


def compare_sections(actual):
    """All bytes are contractual in this compatibility profile, including zeros."""
    errors = []
    expected = reference_sections()
    for name, want in expected.items():
        got = actual.get(name)
        if got is None:
            errors.append(name + " is absent")
        elif len(got) != len(want):
            errors.append(f"{name} is {len(got)} bytes; scalar ABI requires {len(want)}")
        elif got != want:
            different = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]
            errors.append(f"{name} differs from the measured scalar ABI at byte {different[0]} "
                          f"({len(different)} differing bytes)")
    return errors


def verify_object(obj, code=None, *, arch_flag=False):
    """Verify extracted section bytes and the actual entry symbol, without builders."""
    import g17verify as V
    if len(obj) < 32 or struct.unpack_from("<I", obj)[0] != V.MH_MAGIC_64:
        raise ValueError("scalar ABI requires a complete Mach-O object header")
    actual = {name: V._objsect(obj, name) for name in reference_sections()}
    if type(arch_flag) is not bool:
        raise ValueError("serialized ARCH flag must be a boolean")
    if arch_flag:
        import g17archcheck
        arch = g17archcheck.decode_arch(actual["__GPU_ARCH_LD_MD"])
        if not arch["serialized_flag"]:
            raise ValueError("set ARCH experiment delivered the elided form")
        # All OTHER metadata remains subject to the independent frozen reference.
        comparison = dict(actual)
        comparison["__GPU_ARCH_LD_MD"] = reference_sections()["__GPU_ARCH_LD_MD"]
        errors = compare_sections(comparison)
    else:
        errors = compare_sections(actual)
    entries = []
    off = 32
    for _ in range(struct.unpack_from("<I", obj, 16)[0]):
        command, size = struct.unpack_from("<II", obj, off)
        if size < 8 or off + size > len(obj):
            raise ValueError("invalid object load command")
        if command == 2:  # LC_SYMTAB; read the delivered nlist values.
            if size < 24:
                raise ValueError("truncated object symbol-table command")
            symoff, count, stroff, strsize = struct.unpack_from("<IIII", obj, off + 8)
            if symoff + count * 16 > len(obj) or stroff + strsize > len(obj):
                raise ValueError("object symbol table exceeds its section")
            for i in range(count):
                strx, _, _, _, value = struct.unpack_from("<IBBHQ", obj, symoff + i * 16)
                end = obj.find(b"\0", stroff + strx, stroff + strsize)
                if end < 0:
                    raise ValueError("unterminated object symbol")
                if obj[stroff + strx:end] == b"_agc.main":
                    entries.append(value)
        off += size
    if entries != [ENTRY]:
        errors.append(f"delivered _agc.main entry is {entries}; scalar ABI requires [{ENTRY}]")
    if code is not None:
        text = bytes.fromhex("0e000000") + bytes.fromhex("0600") * 30 + code
        text += bytes.fromhex("0600") * ((-len(text) % 16) // 2)
        if V._objsect(obj, "__text") != text:
            errors.append("delivered text differs from the explicit prologue, program, or padding")
    if errors:
        raise ValueError("scalar ABI mismatch: " + "; ".join(errors))
    return {"profile": PROFILE, "metadata_bytes": sum(len(b) for b in actual.values()),
            "sections": {name: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                         for name, b in actual.items()},
            "entry": ENTRY, "serialized_arch_flag": arch_flag,
            "comparison": ("measured scalar metadata with independently parsed set ARCH experiment"
                           if arch_flag else "byte-exact against measured scalar metadata"),
            "hardware_checked_here": False}


def link(kernel, program):
    """Use a named measured ABI for the scan's bounded scalar instruction subset."""
    import g17ldmd
    import g17mdgen
    import g17scanlink
    resources = [(b.index, i * 2, not b.readonly) for i, b in enumerate(kernel.bindings)]
    if resources != BINDINGS or any(b.kind != "device_buffer" or b.element_type != "float"
                                    for b in kernel.bindings):
        raise ValueError("scalar ABI requires float device bindings (1,0,readonly),(2,2,written)")
    if kernel.entry != ENTRY or kernel.fixups or kernel.constant_program:
        raise ValueError("scalar ABI requires entry 64, no unresolved fixups, and no extra constant program")
    if kernel.code != program.code:
        raise ValueError("kernel code differs from the checked compiler program")
    for _, _, inst in program.layout:
        allowed = inst.form in {"movimm.8", "alu.mul.reg", "load.14", "end"}
        allowed |= inst.form == "read_sr.4" and inst.fields.get("sr") == 160
        allowed |= inst.form == "alu.12" and inst.fields.get("op") == 3
        allowed |= inst.form == "auth" and inst.fields.get("opcode") in {998, 3290, 17229}
        if not allowed:
            raise ValueError("instruction is outside the scan's scalar ABI subset: " + inst.form)
    # Reproduce the measured zeroed profile exactly. Restoring optional tables is
    # a different contract and must not happen implicitly in a compatibility path.
    sections = {
        "__GPU_METADATA": g17mdgen.build([1, 2], restore_swept=False),
        "__GPU_LD_MD": g17ldmd.build(entry=ENTRY),
        "__GPU_ARCH_LD_MD": g17ldmd.build_arch(),
        "__GPU_STATS_MD": bytes(96),
        "__GPU_REMARKS_MD": b"",
    }
    errors = compare_sections(sections)
    if errors:
        raise ValueError("scalar ABI emitter drifted: " + "; ".join(errors))
    ledger = {
        "metadata profile": PROFILE,
        "metadata provenance": "recovered scalar tables, exact measured zeroed layout; fixed class policy",
        "kind-6 record": "required class values kind=6, size=20, extra=4; not inferred from binding count",
        "slot-13 data": "present empty byte vector; retained exactly as measured",
        "LD and ARCH": "measured scalar serializers; ARCH class layout is not the newer semantic flag serializer",
        "semantic flags": "compiler facts are retained in the manifest; this compatibility profile uses its measured layout",
    }
    image = g17scanlink.package_sections(
        kernel, {name + ",__compute": b for name, b in sections.items()}, ledger, BINDINGS)
    verify_object(image.object, program.code)
    return image
