"""Author the delivered GELU activation image.

Consumes the compiler's retained bytes and its typed contract. No compile of a
different program, no device, no numerical claim: the bound is the compiler's
and is checked in g17gelu.

TWO BUFFERS WITH A DECLARED REGISTER COUNT DO NOT DETERMINE A CLASS. The swept
500-byte class carries no slot 0 and the measured 380/384-byte class does, and
the contract cannot choose between them - packed_scan's contract and this one
agree on every field the author reads except element type and register count.
So the measured class is requested by name from here, exactly as the range-store
and read-control deliveries request it, and the two executed swept images stay
pinned by the default. This one declares 22 registers, and only the measured
class has a slot to carry them.

THE KERNEL NAME COMES FROM THE CONTRACT. The ABI dict has never carried one, and
supplying a name of this side's choosing produces a correct object - the object
is name-independent - with a wrong library and archive, because the library is
f(object, name). That shipped once on the projections. Here the name is read
from the compiled contract and its absence is a refusal.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERED = ROOT / "results/g17-minilm-gelu-compiler-v1"
EXPECTED_BINDINGS = [(1, 0, False), (2, 2, True)]
# SLOT 32 IS THE STATED LIMIT HERE, as it is on the projections. This program has no back edge at
# 49 instructions, and the corpus law - derived from Apple's emission - gives slot 32 = 1 for
# anything past 30. The measured two-buffer class has no slot 32. The executed precedent in this
# exact class is the read-control at 32 instructions, which is already past that boundary and ran
# with the slot absent, so absence is measured-tolerable at 32 and unknown at 49. That bounds the
# shape, not the count, and it is recorded rather than resolved.
SLOT32_PRECEDENT = dict(image="g17-rangeread-runtime-v2", object="086fd3e6c151201b",
                        metadata_bytes=384, instructions=32, back_edge=False, slot32=None)


def program():
    """Recompile the delivered activation and require the retained bytes."""
    import g17cc
    import g17minilmffn
    compiled = g17cc.compile_function(g17minilmffn.gelu_ir())
    retained = (DELIVERED / "gelu.bin").read_bytes()
    if compiled.code != retained:
        raise ValueError(
            "the recompiled activation differs from the retained delivery (%s against %s); "
            "author the delivered bytes, not a fresh program that resembles them"
            % (hashlib.sha256(compiled.code).hexdigest()[:16],
               hashlib.sha256(retained).hexdigest()[:16]))
    return compiled


def kernel_name(compiled):
    name = getattr(compiled.contract(), "name", None)
    if not name:
        raise ValueError(
            "the delivered contract states no kernel name. The library is a function of the "
            "object and its name, so a name chosen here would emit a library and archive no "
            "delivered contract asked for while the object still matched.")
    return name


def author():
    import g17link
    import g17scanlink
    compiled = program()
    abi = compiled.abi()
    bindings = [(b["index"], b["offset"], b["written"]) for b in abi["bindings"]]
    if bindings != EXPECTED_BINDINGS:
        raise ValueError("the delivered activation binds %s; the measured two-buffer class is "
                         "measured for %s only" % (bindings, EXPECTED_BINDINGS))
    prologue = abi["prologue"]
    prologue = bytes.fromhex(prologue) if isinstance(prologue, str) else bytes(prologue)
    kernel = g17link.Kernel(
        code=compiled.code, name=kernel_name(compiled), entry=abi["entry"], prologue=prologue,
        bindings=[g17link.Binding(index=b["index"], readonly=not b["written"],
                                  element_type=b["element_type"]) for b in abi["bindings"]])
    image = g17scanlink.link(kernel, dict(abi, unswept_two_buffer=True),
                             binding_offsets=[b["offset"] for b in abi["bindings"]],
                             program_contract=compiled.contract())
    return image, compiled, abi, prologue


def check():
    import g17archcheck
    import g17gpumd
    import g17packedcheck
    import g17scanlink
    import struct
    image, compiled, abi, prologue = author()
    declared = [(b["index"], b["offset"], b["written"]) for b in abi["bindings"]]
    sections, symbols = g17archcheck.object_contents(image.object)
    metadata = sections["__GPU_METADATA,__compute"]
    table = g17gpumd.kernel_table(metadata)
    field13 = g17gpumd._table_slot(metadata, table, 13)
    target = field13 + struct.unpack_from("<I", metadata, field13)[0]
    length = struct.unpack_from("<I", metadata, target)[0]
    pool = bytes(metadata[target + 4:target + 4 + length])
    field29 = g17gpumd._table_slot(metadata, table, 29)
    vector = field29 + struct.unpack_from("<I", metadata, field29)[0]
    count = struct.unpack_from("<I", metadata, vector)[0]
    entry = abi["entry"]
    aligned = ((entry + len(compiled.code)) + 15) & ~15
    decoded = g17packedcheck.decode(compiled.code)
    checks = dict(
        archive_round_trips=g17scanlink.verify_contract(image.archive, image.library, declared)
                            == image.object,
        delivered_bindings_equal_declared=g17scanlink.binding_records(metadata) == declared,
        text_is_prologue_code_alignment=(
            sections["__TEXT,__text"] == prologue + compiled.code
            + bytes.fromhex("0600") * ((aligned - entry - len(compiled.code)) // 2)),
        entry_symbols=sorted((s[0], s[4]) for s in symbols
                             if s[0] in ("_agc.main", "_agc.main.constant_program"))
                      == sorted([("_agc.main.constant_program", 0), ("_agc.main", entry)]),
        arch_flag_matches_abi=(g17archcheck.decode_arch(sections["__GPU_ARCH_LD_MD,__compute"])
                               ["serialized_flag"] == abi["arch_flag"]),
        declared_register_count_delivered=g17gpumd.register_count(metadata) == abi["register_count"],
        forms_equal_declared=(sorted({(o, l) for _off, l, o, _f in decoded})
                              == sorted(map(tuple, abi["forms"]))),
        constant_pool_empty=not any(pool),
        no_back_edge=not any(op == 458 for _o, _l, op, _f in decoded),
    )
    return dict(status="passed" if all(v is True for v in checks.values()) else "failed",
                gpu_dispatched=False, kernel_name=kernel_name(compiled),
                code_bytes=len(compiled.code), instructions=len(decoded),
                metadata_bytes=len(metadata), register_count=g17gpumd.register_count(metadata),
                system_register_entries=list(struct.unpack_from("<" + "I" * count, metadata,
                                                                vector + 4)),
                delivered_bindings=g17scanlink.binding_records(metadata),
                sha256=dict(object=hashlib.sha256(image.object).hexdigest(),
                            library=hashlib.sha256(image.library).hexdigest(),
                            archive=hashlib.sha256(image.archive).hexdigest(),
                            code=hashlib.sha256(compiled.code).hexdigest()),
                slot32=None, slot32_law_would_give=1, slot32_precedent=SLOT32_PRECEDENT,
                checks=checks, failed=[k for k, v in checks.items() if v is not True],
                scope=("Image authored from the delivered activation bytes and checked from its "
                       "own bytes. The numerical bound is the compiler's; nothing here executes "
                       "or claims semantics for the expansion's opcodes. Slot 32 is absent in "
                       "this class; the named precedent executed at 32 instructions with it "
                       "absent and bounds no instruction count beyond its own."))


def retain(destination=None):
    """Write the image and its report where the FFN delivery puts them."""
    destination = Path(destination or ROOT / "results/g17-minilm-gelu-image-v1")
    image, compiled, _abi, _prologue = author()
    programs = destination / "programs/gelu"
    programs.mkdir(parents=True, exist_ok=True)
    for filename, blob in (("program.o", image.object), ("program.bin", compiled.code),
                           ("program.lib.metallib", image.library),
                           ("program.arc.metallib", image.archive)):
        (programs / filename).write_bytes(blob)
    # report.json sits beside programs/, as it does in the FFN delivery, not inside it.
    (destination / "report.json").write_text(json.dumps(check(), indent=2) + "\n")
    return destination


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retain", nargs="?", const=True, default=False,
                        help="write the image and report into the delivery directory")
    arguments = parser.parse_args()
    if arguments.retain:
        print(retain(None if arguments.retain is True else arguments.retain))
    else:
        print(json.dumps(check(), indent=2))
