"""Deliverable 5: our own container. The general lowering's code (tlower.py) is authored into an image through the
repository's public path - scanlink.link with a typed ProgramABI contract and an emission ABI shaped like the
compiler's (modelled on results/g17-tensor-common-delivery-v5/abi.json) - so no byte of the image comes from an
Apple compile: header (END + filler to entry 64), code, metadata (register count, slot 44 through the TENSOR class,
binding records), library and archive are all the repository's.
  python3 ownimage.py M N K [acc]     -> results/g17-tensorops-recon-v1/own/<label>/{program.arc.metallib, program.lib.metallib, program.o, contract.json, abi.json}
"""
import hashlib, json, sys
from pathlib import Path
# NO sys.path MANIPULATION. A module under agxforge/ must not touch sys.path - the library
# is imported as a package, and a module that edits the path changes the import
# behaviour of everything loaded after it. The originals inserted the checkout root so
# they could also be run as scripts; run them with `python -m agxforge.g17.<name>` instead.
from agxforge.g17 import model, abi as g17abi, link as L, scanlink as S
from agxforge.g17 import tlower
from agxforge.g17.registerdomain import registers_in_name_checked
HERE = Path(__file__).resolve().parent
# It wrote its artifacts to `HERE / 'own' / label` - BESIDE THE MODULE. Under the campaign's
# gitignored results/ that was invisible; under agxforge/g17 every lower_gemm(...).image() call
# drops a metallib, an object and four JSONs into the SOURCE TREE, and a second call to the
# same shape silently overwrites them. Generated artifacts belong under results/.
import os as _os
# THE DEFAULT IS RELATIVE ON PURPOSE. As an absolute module-level constant this pointed at a
# directory that does not exist in a clean checkout, and the library's own path check flags a
# module-level path constant that does not resolve. My worktree hid it: results/g17-ownimage
# existed there only because my probe runs had created it. Root caught it on a clean checkout and
# fixed it in efc4823b on codex/g17-tensor-awkward-workloads; this is that fix, on the candidate,
# so the branch is correct on its own rather than only after integration.
OUT_ROOT = Path(_os.environ.get('G17_OWNIMAGE_OUT', 'results/g17-ownimage'))
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _output_root():
    """The default resolved against the checkout; an explicit absolute override used as given."""
    return OUT_ROOT if OUT_ROOT.is_absolute() else _REPO_ROOT / OUT_ROOT
REGS = int(__import__("os").environ.get("OWN_REGS", "126"))
ENTRY = 64; HEADER = bytes.fromhex('0e000000') + bytes.fromhex('0600') * 30

def build(M, N, K, acc=False, name='k', transA=False, transB=False, a_type='half', b_type='half', sg=1):
    lda, ldb = (M if transA else K), (K if transB else N)
    body, plan = tlower.lower(M, N, K, lda, ldb, N, registers=REGS, reserved=(), accumulate=acc, transA=transA, transB=transB, a_type=a_type, b_type=b_type, sg=sg)
    code = body                                                               # the compiler's convention: `code` is the program after the entry; text_of prepends the header
    ins = [i for i in model.decode(code, 0) if i.opcode]
    assert ins[-1].opcode.id == 684 and sum(len(i.raw) for i in ins) == len(code), 'code does not decode end to end'
    forms = sorted({(i.opcode.id, len(i.raw)) for i in ins})
    regs = max(max(registers_in_name_checked(model.registers().get(v, '')), default=0) for i in ins for k, v in i.values if k == 'reg') + 1
    # the tensor metadata class is measured for binding indices 1, 2, 3 at pointer offsets 0, 2, 4 (tensormetadata.py); the code's
    # descriptors are 4 x rank (0, 4, 8), the same bytes as for indices 0, 1, 2, so only the launch-side binding indices move
    ET = {'half': ('half', 2), 'bfloat': ('bfloat', 2), 'float': ('float', 4), 'int8': ('uchar', 1), 'uint8': ('uchar', 1)}
    bindings = [dict(index=1, offset=0, written=False, element_type=ET[a_type][0], element_bytes=ET[a_type][1]), dict(index=2, offset=2, written=False, element_type=ET[b_type][0], element_bytes=ET[b_type][1]), dict(index=3, offset=4, written=True, element_type='uint' if a_type in ('int8', 'uint8') else 'float', element_bytes=4)]
    contract = g17abi.ProgramABI.from_dict(dict(
        version=1, name=name, code_sha256=hashlib.sha256(code).hexdigest(), code_size=len(code), entry=ENTRY, prologue=HEADER.hex(),
        bindings=bindings, instructions=[dict(offset=i.offset, length=len(i.raw), opcode=i.opcode.id) for i in ins],
        arch_flag=False, uses_threadgroup=False, writes_buffer=True, writes_texture=False, exact_grid_required=True,
        constant_pool=[], execution=dict(simd_width=32, tensor=True),
        argument_state=dict(pointer_offsets=[[1, 0], [2, 2], [3, 4]], pointer_words=2, block_words=6, block_bytes=24, basis='emitted_pointer_offsets', offset_rule='2 * rank', indices_contiguous=True, not_stated=['per_kernel_slot_1'])))
    abi = dict(abi_version=5, arch_flag=False, bindings=[dict(b) for b in bindings], constant_pool=[], entry=ENTRY,
               execution=dict(simd_width=32, tensor=True), forms=[list(f) for f in forms], has_stores=True, launch=dict(bounds_checked=False, exact_grid_required=True),
               pk_extra=[15, 16], pk_values={'15': 1, '16': 1}, profile=None, prologue=HEADER.hex(), register_count=regs, system_registers=[130],
               uses_threadgroup=False, writes_buffer=True, writes_texture=False)
    kernel = L.Kernel(code=code, name=name, entry=ENTRY, prologue=HEADER, bindings=[L.Binding(index=b['index'], readonly=not b['written'], element_type=b['element_type']) for b in bindings])
    image = S.link(kernel, abi, binding_offsets=[b['offset'] for b in bindings], program_contract=contract)
    label = '%dx%dx%d%s%s%s%s%s' % (M, N, K, '_acc' if acc else '', '_tA' if transA else '', '_tB' if transB else '', '_%s.%s' % (a_type, b_type) if (a_type, b_type) != ('half', 'half') else '', '_sg%d' % sg if sg > 1 else ''); d = _output_root() / label; d.mkdir(parents=True, exist_ok=True)
    (d / 'program.arc.metallib').write_bytes(image.archive); (d / 'program.lib.metallib').write_bytes(image.library); (d / 'program.o').write_bytes(image.object)
    (d / 'contract.json').write_text(json.dumps(contract.to_dict(), indent=1) + '\n'); (d / 'abi.json').write_text(json.dumps(abi, indent=1) + '\n')
    (d / 'plan.json').write_text(json.dumps(plan, indent=1) + '\n'); (d / 'field-ledger.json').write_text(json.dumps(image.field_ledger, indent=1, default=str) + '\n')
    return d, dict(code_bytes=len(code), instructions=len(ins), registers=regs, object_sha256=hashlib.sha256(image.object).hexdigest(), archive_sha256=hashlib.sha256(image.archive).hexdigest())

def build_from(lowered, name='k'):
    """Author the image for THE BODY THIS Lowered ACTUALLY HOLDS, leading dimensions included.

    tensorgemm.Lowered.image() prefers this function and falls back to build(M, N, K, ...) when it
    is absent. That fallback is lossy and silent: build() takes no lda/ldb/ldc and recomputes them
    as contiguous row-major, so for any GEMM with non-contiguous leading dimensions it re-lowers a
    DIFFERENT program and authors an image of that. Measured before this function existed -
    lower_gemm(32, 32, 64, lda=33, int8) hands back a 1028-byte body hashing 4c425fc7ff3b82e6,
    while the contract written beside the image certified 470ebeda6a7a0b46 at 1002 bytes, which is
    the lda=64 program. The certificate named code that was not in the image and nothing refused.
    The label collided too, because it carries only M, N, K and the types.

    So this is the accurate branch, and it exists so that the fallback is never taken rather than
    being taken quietly.
    """
    body, plan = lowered.body, lowered.plan
    M, N, K = lowered.M, lowered.N, lowered.K
    a_type, b_type, acc, sg = lowered.a_type, lowered.b_type, lowered.accumulate, lowered.simdgroups
    transA, transB = lowered.transA, lowered.transB
    ins = [i for i in model.decode(body, 0) if i.opcode]
    assert ins[-1].opcode.id == 684 and sum(len(i.raw) for i in ins) == len(body), 'code does not decode end to end'
    forms = sorted({(i.opcode.id, len(i.raw)) for i in ins})
    regs = max(max(registers_in_name_checked(model.registers().get(v, '')), default=0) for i in ins for k, v in i.values if k == 'reg') + 1
    ET = {'half': ('half', 2), 'bfloat': ('bfloat', 2), 'float': ('float', 4), 'int8': ('uchar', 1), 'uint8': ('uchar', 1)}
    bindings = [dict(index=1, offset=0, written=False, element_type=ET[a_type][0], element_bytes=ET[a_type][1]),
                dict(index=2, offset=2, written=False, element_type=ET[b_type][0], element_bytes=ET[b_type][1]),
                dict(index=3, offset=4, written=True, element_type='uint' if a_type in ('int8', 'uint8') else 'float', element_bytes=4)]
    contract = g17abi.ProgramABI.from_dict(dict(
        version=1, name=name, code_sha256=hashlib.sha256(body).hexdigest(), code_size=len(body), entry=ENTRY, prologue=HEADER.hex(),
        bindings=bindings, instructions=[dict(offset=i.offset, length=len(i.raw), opcode=i.opcode.id) for i in ins],
        arch_flag=False, uses_threadgroup=False, writes_buffer=True, writes_texture=False, exact_grid_required=True,
        constant_pool=[], execution=dict(simd_width=32, tensor=True),
        argument_state=dict(pointer_offsets=[[1, 0], [2, 2], [3, 4]], pointer_words=2, block_words=6, block_bytes=24,
                            basis='emitted_pointer_offsets', offset_rule='2 * rank', indices_contiguous=True,
                            not_stated=['per_kernel_slot_1'])))
    abi = dict(abi_version=5, arch_flag=False, bindings=[dict(b) for b in bindings], constant_pool=[], entry=ENTRY,
               execution=dict(simd_width=32, tensor=True), forms=[list(f) for f in forms], has_stores=True,
               launch=dict(bounds_checked=False, exact_grid_required=True), pk_extra=[15, 16], pk_values={'15': 1, '16': 1},
               profile=None, prologue=HEADER.hex(), register_count=regs, system_registers=[130],
               uses_threadgroup=False, writes_buffer=True, writes_texture=False)
    kernel = L.Kernel(code=body, name=name, entry=ENTRY, prologue=HEADER,
                      bindings=[L.Binding(index=b['index'], readonly=not b['written'], element_type=b['element_type']) for b in bindings])
    image = S.link(kernel, abi, binding_offsets=[b['offset'] for b in bindings], program_contract=contract)
    # THE LABEL CARRIES THE LEADING DIMENSIONS, because two programs that differ only in lda are
    # two programs and must not share an output directory.
    label = '%dx%dx%d_ld%d.%d.%d%s%s%s%s%s' % (M, N, K, lowered.lda, lowered.ldb, lowered.ldc,
        '_acc' if acc else '', '_tA' if transA else '', '_tB' if transB else '',
        '_%s.%s' % (a_type, b_type) if (a_type, b_type) != ('half', 'half') else '', '_sg%d' % sg if sg > 1 else '')
    d = _output_root() / label; d.mkdir(parents=True, exist_ok=True)
    (d / 'program.arc.metallib').write_bytes(image.archive); (d / 'program.lib.metallib').write_bytes(image.library)
    (d / 'program.o').write_bytes(image.object)
    (d / 'contract.json').write_text(json.dumps(contract.to_dict(), indent=1) + '\n')
    (d / 'abi.json').write_text(json.dumps(abi, indent=1) + '\n')
    (d / 'plan.json').write_text(json.dumps(plan, indent=1) + '\n')
    (d / 'field-ledger.json').write_text(json.dumps(image.field_ledger, indent=1, default=str) + '\n')
    return d, dict(code_bytes=len(body), instructions=len(ins), registers=regs,
                   leading=[lowered.lda, lowered.ldb, lowered.ldc],
                   code_sha256=hashlib.sha256(body).hexdigest(),
                   object_sha256=hashlib.sha256(image.object).hexdigest(),
                   archive_sha256=hashlib.sha256(image.archive).hexdigest())


if __name__ == '__main__':
    d, info = build(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), 'acc' in sys.argv[4:]); print(d, info)
