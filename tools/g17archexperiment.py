"""Author the single-variable ARCH control for the existing half scan.

This is an experiment adapter, not the new general compiler/linker ABI. It uses
the existing section serializer and requires delivered-object equivalence for
everything except ARCH. No instruction or serialized object bytes are patched.
"""


def build(rows, columns):
    import g17archcheck
    import g17authorobj
    import g17halfimage
    import g17link
    import g17scanlink
    baseline, abi, program = g17halfimage.build(rows, columns)
    sections, _ = g17archcheck.object_contents(baseline.object)
    del sections["__TEXT,__text"]
    ledger = dict(baseline.field_ledger)
    sections[g17archcheck.ARCH] = g17authorobj._arch(True, ledger)
    ledger["ARCH experiment"] = "one-variable set-versus-elided comparison; hardware unasserted"
    bindings = [g17link.Binding(b["index"], readonly=not b["written"], element_type="half")
                for b in abi["bindings"]]
    kernel = g17link.Kernel(code=program.code, name="half_scan", entry=abi["entry"],
                           prologue=abi["prologue"], bindings=bindings)
    image = g17scanlink.package_sections(kernel, sections, ledger, g17halfimage.BINDINGS)
    g17archcheck.compare_objects(baseline.object, image.object)
    return image, abi, program
