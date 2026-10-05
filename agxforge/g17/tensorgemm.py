"""The lowering as ONE callable module for the registry (goal deliverable 6).

    import tensorlower
    k = tensorlower.lower_gemm(M, N, K, lda=K, ldb=N, ldc=N, a_type='half', b_type='half',
                               accumulate=False, transA=False, transB=False, simdgroups=1)
    k.body        -> bytes: the complete program after the entry (index prologue, masks, loads, MMAs, C add, stores, END)
    k.plan        -> dict: register map, tile groups, every instruction with its purpose (the receipt)
    k.image()     -> agxforge.g17.scanlink.Image: object, library and archive authored through scanlink.link with a typed
                     ProgramABI (bindings 1, 2, 3; entry 64; TENSOR class metadata; slot 44) - no Apple-compiled byte
    k.launch      -> dict: what the dispatcher must do (buffer indices, threads per threadgroup, exact grid)
    tensorlower.refusals(...) -> the named reasons a request is refused, without lowering it

Everything the registry needs to decide and to author is here; the encoders it depends on (mmaenc, memenc, faddenc,
ledgerenc, indexgen) and the censuses (fieldmap.json, memmap.json) sit beside this file. Integration into
agxforge/g17/tensor.py is proposed in docs/archive/g17-tensor-lowering-handoff.md and proposals/tensor_emitter_arm.patch; it is
the compiler owner's to apply.
"""
import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
# NO sys.path MANIPULATION. A module under agxforge/ must not touch sys.path - the library
# is imported as a package, and a module that edits the path changes the import
# behaviour of everything loaded after it. The originals inserted the checkout root so
# they could also be run as scripts; run them with `python -m agxforge.g17.<name>` instead.
from agxforge.g17 import tlower, ownimage, registerdomain

REGISTER_BUDGET = registerdomain.REGISTER_COUNT  # plan through R125; R126+ operands are squashed by hardware (section 135)
TYPES = ('half', 'bfloat', 'float', 'int8', 'uint8', 'fp8e4m3', 'fp8e5m2')

def refusals(M, N, K, lda, ldb, ldc, a_type='half', b_type='half', accumulate=False, transA=False, transB=False, simdgroups=1, grid_n=1, split_k=1):
    """Named reasons this lowering will not author the request (empty list = it will). Mirrors tlower's own checks."""
    out = []
    if min(M, N, K) < 1: out.append('extents must be >= 1')
    if a_type not in TYPES or b_type not in TYPES: out.append('operand type not in %s' % (TYPES,))
    integer = a_type in ('int8', 'uint8')
    if integer != (b_type in ('int8', 'uint8')): out.append('int8 operands must be paired')
    fp8 = ('fp8e4m3', 'fp8e5m2')
    if (a_type in fp8 or b_type in fp8) and not all(t in fp8 or t == 'bfloat' for t in (a_type, b_type)):
        out.append('an fp8 operand pairs with fp8 or bfloat')
    if (a_type in fp8 or b_type in fp8) and (transA or transB):
        out.append('fp8 with a transpose bit is not implemented')
    # 8 simdgroups admitted by MM P13 (section 25.127: tlower's and16 mask widened to sg - 1); 16 stays refused
    if simdgroups not in (1, 2, 4, 8): out.append('simdgroups must be 1, 2, 4 or 8')
    if simdgroups > 1 and M % (16 * simdgroups): out.append('simdgroups: M must be a multiple of 16 x simdgroups')
    # P3 N-tiled grid: grid_n threadgroups each own N/grid_n columns (whole-GPU parallelism over N)
    if not (isinstance(grid_n, int) and 1 <= grid_n <= 256): out.append('grid_n must be 1..256 threadgroups')
    elif grid_n > 1 and N % (16 * grid_n): out.append('grid_n: N must be a multiple of 16 x grid_n')
    # P3 split-K: split_k threadgroups each own K/split_k of the contraction and write a partial (the fold reduce sums them)
    if not (isinstance(split_k, int) and 1 <= split_k <= 256): out.append('split_k must be 1..256 threadgroups')
    elif split_k > 1 and K % (16 * split_k): out.append('split_k: K must be a multiple of 16 x split_k')
    return out

class Lowered:
    def __init__(self, M, N, K, lda, ldb, ldc, a_type, b_type, accumulate, transA, transB, simdgroups,
                 body, plan, binds=(0, 1, 2), offsets=(0, 0, 0), end=True, reserved=(), grid=1, grid_n=1, split_k=1):
        self.M, self.N, self.K, self.lda, self.ldb, self.ldc = M, N, K, lda, ldb, ldc
        self.a_type, self.b_type, self.accumulate, self.transA, self.transB, self.simdgroups = a_type, b_type, accumulate, transA, transB, simdgroups
        self.binds, self.offsets, self.end, self.reserved = tuple(binds), tuple(offsets), bool(end), tuple(sorted(reserved))
        self.body, self.plan = body, plan
        self.grid, self.grid_n, self.split_k = int(grid), int(grid_n), int(split_k)
        # The exact grid is the row-grid times the column-grid times the K-partition (one axis at a time
        # for now): threadgroups over N (grid_n) or over K (split_k) is where P3's whole-GPU parallelism
        # comes from. A split-K launch has split_k threadgroups, each writing an M x N partial to slot t.
        self.launch = dict(bindings={'A': 1, 'B': 2, 'C': 3}, threads_per_threadgroup=32 * simdgroups,
                           threadgroups=self.grid * self.grid_n * self.split_k, exact_grid_required=True,
                           c_element='int32' if a_type in ('int8', 'uint8') else 'float32')
    def image(self, name='k'):
        """Author through the repository's scanlink path (ownimage.build's construction) and return the scanlink Image."""
        return ownimage.build_from(self, name=name) if hasattr(ownimage, 'build_from') else ownimage.build(self.M, self.N, self.K, self.accumulate, name=name, transA=self.transA, transB=self.transB, a_type=self.a_type, b_type=self.b_type, sg=self.simdgroups)
    def receipt(self):
        return dict(shape=[self.M, self.N, self.K], leading=[self.lda, self.ldb, self.ldc], types=[self.a_type, self.b_type], accumulate=self.accumulate,
                    transA=self.transA, transB=self.transB, simdgroups=self.simdgroups, body_sha256=hashlib.sha256(self.body).hexdigest(), body_bytes=len(self.body),
                    instructions=len(self.plan['ops']), registers=self.plan['registers'], groups=len(self.plan['groups']),
                    binds=list(self.binds), offsets=list(self.offsets), end=self.end, launch=self.launch)

def lower_gemm(M, N, K, lda=None, ldb=None, ldc=None, a_type='half', b_type='half', accumulate=False,
               transA=False, transB=False, simdgroups=1, registers=REGISTER_BUDGET,
               reserved=(), binds=(0, 1, 2), offsets=(0, 0, 0), end=True, keep=False, store=True, a_regs=None, epilogue=(), grid=1, split_fp32=False,
               a_convert=None, saturate=False, kloop=False, b_regs=None, b_convert=None, reduce=None, c_regs=None, grid_n=1, split_k=1,
               b_index=None, index_init=(), kloop_unroll=1, head_index=None, head_slices=None, c_inplace=False,
               fold_offsets=False, hoist=None, a_keep=False, kloop_chunk=None):
    lda = lda if lda is not None else (M if transA else K); ldb = ldb if ldb is not None else (K if transB else N); ldc = ldc if ldc is not None else N
    why = refusals(M, N, K, lda, ldb, ldc, a_type, b_type, accumulate, transA, transB, simdgroups, grid_n=grid_n, split_k=split_k)
    if why: raise ValueError('refused: ' + '; '.join(why))
    if len(tuple(binds)) != 3 or len(tuple(offsets)) != 3:
        raise ValueError('refused: composed tensor lowering requires three binding ranks and offsets')
    if not isinstance(registers, int) or registers < 1 or registers > registerdomain.REGISTER_COUNT:
        raise ValueError('refused: register budget must stay within allocatable R0..R125 (got %r)' % (registers,))
    bad_reserved = registerdomain.invalid_registers(reserved)
    if bad_reserved:
        raise ValueError('refused: reserved register(s) %s are outside allocatable R0..R125' %
                         ', '.join('R%d' % r for r in bad_reserved))
    body, plan = tlower.lower(M, N, K, lda, ldb, ldc, registers=registers, reserved=tuple(reserved),
                              accumulate=accumulate, transA=transA, transB=transB, a_type=a_type,
                              b_type=b_type, sg=simdgroups, binds=tuple(binds), offsets=tuple(offsets),
                              end=bool(end), keep=bool(keep), store=bool(store), a_regs=a_regs,
                              epilogue=tuple(epilogue), grid=int(grid), split_fp32=bool(split_fp32), a_convert=a_convert, saturate=bool(saturate), kloop=bool(kloop),
                              b_regs=b_regs, b_convert=b_convert, reduce=reduce, grid_n=int(grid_n), split_k=int(split_k),
                              **({} if int(kloop_unroll) == 1 else dict(kloop_unroll=int(kloop_unroll))),
                              **({} if c_regs is None else dict(c_regs=c_regs)),
                              **({} if not a_keep else dict(a_keep=True)),
                              **({} if b_index is None and not index_init else dict(b_index=b_index, index_init=tuple(index_init))),
                              **({} if head_index is None else dict(head_index=tuple(head_index))),
                              **({} if head_slices is None else dict(head_slices=head_slices)),
                              **({} if not c_inplace else dict(c_inplace=True)),
                              **({} if not fold_offsets else dict(fold_offsets=True)),
                              **({} if hoist is None else dict(hoist=dict(hoist))),
                              **({} if kloop_chunk is None else dict(kloop_chunk=kloop_chunk)))
    return Lowered(M, N, K, lda, ldb, ldc, a_type, b_type, accumulate, transA, transB, simdgroups,
                   body, plan, binds=binds, offsets=offsets, end=end, reserved=reserved, grid=grid, grid_n=grid_n, split_k=split_k)

if __name__ == '__main__':
    k = lower_gemm(17, 19, 16); print(json.dumps(k.receipt(), indent=1))
    try: lower_gemm(17, 19, 16, a_type='int8', b_type='int8')
    except ValueError as e: print(e)
