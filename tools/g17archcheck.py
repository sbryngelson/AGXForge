"""CPU-only gate for a one-variable, elided-versus-set ARCH experiment.

This checks delivered images, not compiler declarations. It neither authors an
image nor permits its execution. The ordinary half-image gate must accept the
baseline; the candidate must preserve every section except ARCH and every
symbol. Both archives must independently satisfy the delivered binding check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

ARCH = "__GPU_ARCH_LD_MD,__compute"


def decode_arch(data):
    """Read the two measured table layouts without using their serializer.

    Require the exact supported shape, including zero padding. An empty child
    table is not a null reference, and a four-byte boolean is not this byte field.
    """
    if len(data) not in (32, 40):
        raise ValueError("ARCH must use the measured 32- or 40-byte layout")
    used = bytearray(len(data))

    def field(offset, fmt, expected=None):
        size = struct.calcsize(fmt)
        if offset < 0 or offset + size > len(data):
            raise ValueError("ARCH field is out of bounds")
        value = struct.unpack_from(fmt, data, offset)[0]
        if expected is not None and value != expected:
            raise ValueError(f"ARCH field at {offset} is {value}, expected {expected}")
        used[offset:offset + size] = b"\1" * size
        return value

    root = field(0, "<I", 12)
    vtable = root - field(root, "<i", 6)
    field(vtable, "<H", 6)
    field(vtable + 2, "<H", 8)
    slot = field(vtable + 4, "<H", 4)
    displacement = field(root + slot, "<I")
    flag = len(data) == 40
    if not flag:
        if displacement != 0:
            raise ValueError("elided ARCH must have a null child reference")
    else:
        child = root + slot + displacement
        if child != 28:
            raise ValueError("set ARCH must reference its child at byte 28")
        child_vtable = child - field(child, "<i", 8)
        field(child_vtable, "<H", 8)
        field(child_vtable + 2, "<H", 8)
        field(child_vtable + 4, "<H", 0)
        flag_slot = field(child_vtable + 6, "<H", 7)
        field(child + flag_slot, "<B", 1)
    if any(byte and not owned for byte, owned in zip(data, used)):
        raise ValueError("ARCH has nonzero bytes outside its measured fields")
    return {"serialized_flag": flag, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def object_contents(obj):
    """Bounded Mach-O parsing, retaining duplicate names and symbol attributes.

    Physical offsets can move when ARCH grows. Compare section payloads and
    section-relative symbol values, not the enclosing file's placement bytes.
    """
    if len(obj) < 32 or struct.unpack_from("<I", obj)[0] != 0xFEEDFACF:
        raise ValueError("invalid Mach-O header")
    count, command_bytes = struct.unpack_from("<II", obj, 16)
    end, offset = 32 + command_bytes, 32
    if end > len(obj):
        raise ValueError("object load commands exceed the file")
    sections, symbols = {}, []
    for _ in range(count):
        if offset + 8 > end:
            raise ValueError("truncated object load command")
        command, size = struct.unpack_from("<II", obj, offset)
        if size < 8 or offset + size > end:
            raise ValueError("invalid object load command size")
        if command == 0x19:
            if size < 72:
                raise ValueError("truncated object segment")
            nsects = struct.unpack_from("<I", obj, offset + 64)[0]
            if 72 + nsects * 80 != size:
                raise ValueError("invalid object section count")
            for i in range(nsects):
                pos = offset + 72 + 80 * i
                name = obj[pos:pos + 16].rstrip(b"\0").decode("ascii")
                segment = obj[pos + 16:pos + 32].rstrip(b"\0").decode("ascii")
                name = segment + "," + name
                _, length, start = struct.unpack_from("<QQI", obj, pos + 32)
                if name in sections or start + length > len(obj):
                    raise ValueError("duplicate or out-of-bounds object section")
                sections[name] = obj[start:start + length]
        elif command == 2:
            if size != 24:
                raise ValueError("invalid object symbol-table command")
            symoff, nsym, stroff, strsize = struct.unpack_from("<IIII", obj, offset + 8)
            if symoff + nsym * 16 > len(obj) or stroff + strsize > len(obj):
                raise ValueError("object symbol table exceeds the file")
            for i in range(nsym):
                strx, kind, section, desc, value = struct.unpack_from("<IBBHQ", obj, symoff + i * 16)
                stop = obj.find(b"\0", stroff + strx, stroff + strsize)
                if strx >= strsize or stop < 0:
                    raise ValueError("invalid object symbol name")
                symbols.append((obj[stroff + strx:stop].decode("ascii"), kind, section, desc, value))
        offset += size
    if offset != end:
        raise ValueError("object load-command count disagrees with its size")
    return sections, symbols


def compare_objects(baseline, candidate):
    left, left_symbols = object_contents(baseline)
    right, right_symbols = object_contents(candidate)
    if left.keys() != right.keys() or ARCH not in left:
        raise ValueError("ARCH variants have different or missing section names")
    if left_symbols != right_symbols:
        raise ValueError("ARCH variants have different symbols or entry points")
    changed = [name for name in left if left[name] != right[name]]
    if changed != [ARCH]:
        raise ValueError(f"ARCH must be the only changed section, got {changed}")
    before, after = decode_arch(left[ARCH]), decode_arch(right[ARCH])
    if before["serialized_flag"] or not after["serialized_flag"]:
        raise ValueError("ARCH comparison requires an elided baseline and a set candidate")
    return {"status": "passed", "baseline": before, "candidate": after,
            "unchanged_sections": {k: hashlib.sha256(v).hexdigest()
                                   for k, v in left.items() if k != ARCH},
            "symbols": left_symbols, "gpu_executed": False}


def check_pair(baseline, candidate):
    import g17halfcheck
    import g17scanlink
    baseline, candidate = Path(baseline), Path(candidate)
    # Preserve the full existing CPU arithmetic/addressing gate, including the
    # refusal of the original store-displacement defect.
    cpu = g17halfcheck.check_bundle(baseline)
    manifests, objects = [], []
    for directory in (baseline, candidate):
        manifest = json.loads((directory / "manifest.json").read_text())
        archive = (directory / "scan.arc.metallib").read_bytes()
        library = (directory / "scan.lib.metallib").read_bytes()
        obj = g17scanlink.verify_contract(archive, library, [(1, 0, False), (2, 2, True)])
        for key, data in (("archive", archive), ("object", obj)):
            if hashlib.sha256(data).hexdigest() != manifest["sha256"][key]:
                raise ValueError(f"{directory.name}: {key} hash differs from manifest")
        manifests.append(manifest)
        objects.append(obj)
    for key in ("profile", "shape", "abi"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"ARCH variants have different {key}")
    if manifests[0]["sha256"]["code"] != manifests[1]["sha256"]["code"]:
        raise ValueError("ARCH variants declare different code hashes")
    report = compare_objects(*objects)
    report.update(cpu_baseline=cpu, shape=manifests[0]["shape"],
                  archives=[m["sha256"]["archive"] for m in manifests],
                  scope="one changed ARCH section; CPU checks only, no loader or hardware evidence")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    try:
        result = check_pair(args.baseline, args.candidate)
    except (ValueError, KeyError, OSError, struct.error) as exc:
        print(json.dumps({"status": "refused", "gpu_executed": False, "reason": str(exc)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
