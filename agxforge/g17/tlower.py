"""General row-major tensor lowering (goal deliverables 1-3): everything in the body is generated from the account.
  prologue: lane id from SR_SIMD_ELEM; m, col; idxA/idxB/idxC = m*ld + col           (indexgen.py, ledger-encoded ALU)
  masks:    per boundary tile, op612 range masks on the row (x4 trick) and column coordinates, op437 AND       (in the body)
  loads:    op12674 (two words per lane) or op12675 masked; A rows m and m+8 of the tile, B rows k and k+8
  MMAs:     op5106/op5107 chains over K with the scoreboard wait masks; accumulator tile groups from the register budget
  stores:   op17257 or masked op17258, rows m and m+8; optional C add (fadd) for Apple's accumulate order
Inputs are row-major A (M x K), B (K x N), C (M x N) with any M, N, K >= 1 and leading dimensions lda, ldb, ldc.
"""
import sys
from pathlib import Path
# NO sys.path MANIPULATION. A module under agxforge/ must not touch sys.path - the library
# is imported as a package, and a module that edits the path changes the import
# behaviour of everything loaded after it. The originals inserted the checkout root so
# they could also be run as scripts; run them with `python -m agxforge.g17.<name>` instead.
from agxforge.g17 import model, registerdomain
from agxforge.g17 import memenc, mmaenc, faddenc, indexgen, ledgerenc, epienc, fp8enc
from agxforge.g17.lower import END, NOP
names = model.registers(); INV = {v: k for k, v in names.items()}
up = lambda x: -(-x // 16) * 16

def op612(dst16, src, lo, hi, wait=0):
    """bit j of R<dst16>H = [lo <= R<src> + j < hi]   (masks live in H halves: op437's second source field reaches only H)"""
    return ledgerenc.encode(612, {0: 'R%dH' % dst16, 1: wait, 2: 'R%d' % src, 3: 32, 4: lo, 5: hi})   # 32: source stays live (16 would release it)
def and16x16(dst16, a16, b16):
    """R<dst>H = R<a>H & R<b>H (op437)"""
    return ledgerenc.encode(437, {0: 'R%dH' % dst16, 1: 0, 2: 'R%dH' % a16, 3: 32, 4: (b16, 'R%dH' % b16)})   # operand 4's field counts 32-bit registers, H fixed

class Plan:
    def __init__(self, registers, reserved):
        if not isinstance(registers, int) or registers < 1 or registers > registerdomain.REGISTER_COUNT:
            raise ValueError("register budget %r exceeds the allocatable R0..R125 namespace" % (registers,))
        reserved = tuple(int(r) for r in reserved)
        bad = registerdomain.invalid_registers(reserved)
        if bad:
            raise ValueError("reserved register(s) %s are outside the allocatable R0..R125 namespace" %
                             ", ".join("R%d" % r for r in bad))
        self.free = [r for r in range(registers) if r not in reserved]
        self.accumulator_groups = 0
    def take(self, n):
        for r in self.free:
            if r % n == 0 and all((r + i) in self.free for i in range(n)):
                if n == registerdomain.FP32_ACCUMULATOR_GROUP_WIDTH:
                    if self.accumulator_groups >= registerdomain.MAX_FP32_ACCUMULATOR_GROUPS:
                        raise ValueError("FP32 accumulator group limit exceeded: at most %d groups fit in R0..R125" %
                                         registerdomain.MAX_FP32_ACCUMULATOR_GROUPS)
                    try:
                        registerdomain.validate_accumulator_group(r)
                    except ValueError:
                        continue
                for i in range(n): self.free.remove(r + i)
                if n == registerdomain.FP32_ACCUMULATOR_GROUP_WIDTH:
                    self.accumulator_groups += 1
                return r
        raise ValueError('out of registers')

ES = {'half': 2, 'bfloat': 2, 'float': 4, 'int8': 1, 'uint8': 1, 'fp8e4m3': 1, 'fp8e5m2': 1}
WORDS = {'half': 2, 'bfloat': 2, 'float': 4, 'int8': 1, 'uint8': 1, 'fp8e4m3': 1, 'fp8e5m2': 1}      # words per lane per fragment row (four elements)
# FP8 OPERANDS (recon section 137 part 5, Apple's lowering decoded in fp8enc): an fp8 fragment is
# loaded like int8 (one byte per element, one word per row), then four op17642 unpack its four
# 16-bit halves into a four-register bf16 tuple, and the MMA is the ordinary bf16 op5106. fp8
# embeds exactly in bf16, so the arithmetic is section 136's rule on the decoded values.
FP8 = {'fp8e4m3': 'e4m3', 'fp8e5m2': 'e5m2'}


# the memory-stage GELU's constants (tensorreduce.emit_row_gelu), rounded to fp32 as it rounds them
GELU_CONSTANTS = (1.702, -1.4426950408889634, 1.0)

# loop-carried row bases (lower's kloop_bases) when the caller does not say: off, so every existing body is unchanged
KLOOP_BASES = False
# MM 25.144.1 probe: inside the K loop, clear the MMA flags word's 'more' bit (0x20) on every Nth MMA (MLX's steel loop
# clears it on the last of each group of 4); None keeps every body's bytes
MMA_MORE_PERIOD = None

EXEC_RESTORE = bytes.fromhex("3e03400e")     # op577, the pop cc's executed loop emits
EXEC_MASK = bytes.fromhex("1e00000e")        # op582, which gates op458


def _loop_encoders():
    """cc's measured loop encoders, taken at call time (cc imports this module's callers)."""
    from agxforge.g17 import cc, asm
    return asm, cc.CMP_IMM, cc.BRANCH_BACK, cc.BRANCH_TAIL_BACK


def _check_counted_loop(body, cnt, trips):
    """REFUSE A LOOP WHOSE TERMINATION IS NOT EVIDENT IN ITS BYTES. The counter must be written by
    exactly one instruction in the body - its own increment (op10279, cnt = cnt + 1) - and compared
    against a constant below the cmp's 8-bit immediate and the 256-level mask stack. A body in which
    anything else writes the counter could run forever, and a hang wedges the GPU."""
    from agxforge.g17 import tensorlife
    names = model.registers()
    writers = []
    for j, ins in enumerate(model.decode(body, 0)):
        if ins.opcode is None:
            raise RuntimeError('kloop body does not decode at instruction %d' % j)
        regs = [names.get(v, v) for k, v in ins.values if k == 'reg']
        if regs and cnt in tensorlife._regs(regs[0]):
            writers.append((ins.opcode.id, [v for _k, v in ins.values]))
    ok = (len(writers) == 1 and writers[0][0] == 10279 and writers[0][1][2] == 1
          and names.get(writers[0][1][3]) == 'R%d' % cnt)
    if not ok or not 1 <= trips <= 255:
        raise RuntimeError('kloop counter R%d is written by %s in the loop body (trips %d): refusing a loop '
                           'whose termination is not its own increment' % (cnt, writers, trips))

def _check_loop_carried(body):
    """REFUSE A BODY THAT RELEASES A VALUE THE NEXT TRIP READS (MM 25.95, 25.116). A register read in
    the body before any write in it is live across the back edge (kA, kB, stepB, the counter, the lane
    register, the accumulators). A source lifetime of 16 releases it - the executing lanes' copy reads 0
    on the next trip - unless the body writes it again after that read (Apple's own counter increment
    releases its source and rewrites it in the same instruction). Returns the live-in registers."""
    from agxforge.g17 import tensorlife
    names = model.registers()
    ins = list(model.decode(body, 0))
    defs = []
    for i in ins:
        if i.opcode is None:
            raise RuntimeError('kloop body does not decode')
        defs.append(set().union(*[tensorlife._regs(names.get(v, v)) for n, (k, v) in enumerate(i.values)
                                  if k == 'reg' and i.opcode.is_def(n)] or [set()]))
    written, livein = set(), set()
    for j, i in enumerate(ins):
        vals = list(i.values)
        for n, (k, v) in enumerate(vals):
            if k != 'reg' or i.opcode.is_def(n):
                continue
            fresh = tensorlife._regs(names.get(v, v)) - written
            livein |= fresh
            if fresh and n + 1 < len(vals) and vals[n + 1][1] == 16:
                later = set().union(*defs[j:])
                if fresh - later:
                    raise RuntimeError('kloop body instruction %d (op%d) releases the loop-carried %s and '
                                       'nothing in the body writes it again: the next trip would read 0'
                                       % (j, i.opcode.id, names.get(v, v)))
        written |= defs[j]
    return livein


def lower(M, N, K, lda, ldb, ldc, registers=registerdomain.REGISTER_COUNT, reserved=(), accumulate=False, body_room=None, transA=False, transB=False, a_type='half', b_type='half', sg=1,
          binds=(0, 1, 2), offsets=(0, 0, 0), end=True, keep=False, store=True, a_regs=None, epilogue=(), grid=1, split_fp32=False, a_convert=None, saturate=False,
          kloop=False, b_regs=None, b_convert=None, reduce=None, c_regs=None, grid_n=1, split_k=1, b_index=None, index_init=(),
          kloop_unroll=1, head_index=None, head_slices=None, kloop_bases=None, c_inplace=False,
          fold_offsets=False, hoist=None, a_keep=False, kloop_chunk=None):
    """binds: the buffer binding rank of A, B, C (the descriptor byte is 4 x rank); offsets: byte offsets of A, B, C within their buffers
    (folded into every displacement / base register); end=False omits END so bodies can be concatenated (composition).

    REGISTER-DIRECT CHAINING (recon sections 125, 129, 132, 136; Apple's own chain decoded in docs/archive/g17-tensor-register-chain.md):
    keep=True leaves every D tile live after the body (its stores do not release the source) and returns the tile -> register map
    as plan['acc']; it needs every tile in one register group. store=False also drops the C stores (only when the consumer
    overwrites all of C). a_regs={(mi, k): register} feeds the A operand from registers instead of loading it: tile (mi, k) of
    A is the tuple at that register. A fed fp32 D tile is the float A tuple slot for slot (A_eff[r][k] = D[r][k]), so a
    float x half consumer needs no instruction at all; a 16-bit consumer needs one op1016 per element (tchain.py)."""
    # a_convert='half': the fed D tiles are fp32 accumulators and this body's A is half. Each fed tile is
    # narrowed first with eight op1016 (RNE) into a four-register half tuple, slot i to half i - the
    # A fragment and the D fragment share one layout (section 132), and Apple's own register chain is
    # exactly MMA, eight op1016, MMA (docs/archive/g17-tensor-register-chain.md).
    if a_convert is not None and (a_convert != 'half' or a_regs is None or a_type != 'half'):
        raise ValueError('refused: a_convert is the half narrowing of a register-fed fp32 D into a half A')
    # THE FOUR FEED MODES (recon section 132 part 3). A register-fed operand is the producer's D tile
    # read as that operand's fragment: fed as A, A_eff[r][k] = D[r][k] (mode A); fed as B,
    # B_eff[k][c] = D[rotl1(k)][c] (mode B); fed as A under transA, A_eff[r][k] = D[rotl1(k)][rotr1(r)]
    # (mode At); fed as B under transB, B_eff[k][c] = D[rotl1(c)][k] (mode Bt). The instruction is the
    # ordinary MMA with the transpose bit; the rotl1/rotr1 relabelings are data-layout facts the
    # caller's arrangement and reference carry (the host-supplied operand arrangement of part 3).
    # b_regs={(k, ni): register} feeds B the way a_regs feeds A; b_convert='half' narrows it.
    if b_convert is not None and (b_convert != 'half' or b_regs is None or b_type != 'half'):
        raise ValueError('refused: b_convert is the half narrowing of a register-fed fp32 D into a half B')
    if a_regs is not None and b_regs is not None:
        raise ValueError('refused: one operand is register-fed per body (section 132 measures one feed)')
    # THE ACCUMULATOR (C) FEED (production row P2, MM 25.125): c_regs={(mi, ni): register} is the
    # previous body's kept D tile (mi, ni), and this body ACCUMULATES onto it with no memory round
    # trip. It replaces exactly one thing in the memory form: the eight C words that the accumulate
    # path loads into a temporary (slot 6) become the kept registers, and the add after the K chain
    # is the same fadd, D += C, in the same order. So the arithmetic is the memory bridge's, bit for
    # bit. Each kept register is read by one fadd, which releases it (a tile belongs to one group).
    # Refused with any other feed on the same body (the kept D would be read twice, once as an
    # operand released per group - the 25.89 defect), a seeded (saturating) accumulate, int8, a
    # split, a grid or simdgroup split, a K loop, and partial tiles.
    if c_regs is not None:
        # c_inplace takes a simdgroup split (MM 25.144.8): the registers are per thread, so each simdgroup's
        # accumulator is its own M/sg rows at the same register numbers, and no C address is split
        # a register-fed A beside the in-place accumulator (MM 25.163, attention's P into PV onto O): the two register
        # sets are different registers, so no kept D is read twice (the 25.89 defect this refusal guards)
        a_fed_inplace = a_regs is not None and c_inplace and not ({int(v) + i for v in a_regs.values() for i in range(8)} &
                                                                  {int(v) + i for v in c_regs.values() for i in range(8)})
        if (not accumulate or (a_regs is not None and not a_fed_inplace) or b_regs is not None or saturate or split_fp32 or kloop
                or grid != 1 or (sg != 1 and not c_inplace) or M % 16 or N % 16 or a_type in ('int8', 'uint8')
                or reduce):
            raise ValueError('refused: c_regs is the fp32 accumulator feed of a whole-tile accumulate, one '
                             'simdgroup, with no other register feed, saturation, split, grid or K loop')
        c_regs = {tuple(key): int(reg) for key, reg in c_regs.items()}
        want = {(mi, ni) for mi in range(M // 16 // (sg if c_inplace else 1)) for ni in range(N // 16)}
        if set(c_regs) != want:
            raise ValueError('refused: c_regs must name every C tile %s exactly once' % sorted(want))
    # THE LOOP-CARRIED ACCUMULATOR (MM 25.144.8): c_inplace=True writes D + C back INTO the c_regs registers
    # (fadd creg = acc + creg, the chain temporary released, the kept C not), so a body run once per loop trip
    # finds its previous trip's sum where it left it. The arithmetic is the C feed's fadd, operand for operand;
    # only the destination moves. Nothing is stored: the registers are the result (plan['acc'] = c_regs).
    # THE HOISTED PROLOGUE (MM 25.144.8): hoist={'lane', 'idxA', 'idxB0', 'idxA2'[, 'idxC', 'idxC2']: fixed registers}
    # splits the index prologue in two. Everything but the B index add is loop-invariant - the lane read, m/col,
    # the index terms, the simdgroup, head and slice offsets and the folded offsets - and is returned as
    # plan['prologue'] for the caller to place ONCE before its loop, into these registers, which the caller keeps
    # for the whole program. The body starts with the per-trip part: idxB = idxB0 + the B index register (and B's
    # derived index). Whole tiles only (the masks read m and col, which are not kept) and no store-side or
    # K-loop form that reads the prologue's other registers.
    if hoist is not None:
        need = {'lane', 'idxA', 'idxB0', 'idxA2'} | (set() if c_inplace else {'idxC', 'idxC2'})
        if set(hoist) != need:
            raise ValueError('refused: hoist names %s, this body needs exactly %s' % (sorted(hoist), sorted(need)))
        if (M % 16 or N % 16 or K % 16 or kloop or reduce or epilogue or keep or split_fp32 or saturate
                or a_regs is not None or b_regs is not None or grid != 1 or grid_n != 1 or split_k != 1
                or a_type in FP8 or b_type in FP8 or a_type in ('int8', 'uint8')):
            raise ValueError('refused: hoist is a whole-tile memory-operand body without K loop, grid, reduction, '
                             'epilogue, register feed or hand-off')
        if any(int(r) not in set(reserved) for r in hoist.values()):
            raise ValueError('refused: every hoisted register must be reserved from the body')
    if c_inplace and (c_regs is None or store or keep or epilogue or a_type in FP8 or b_type in FP8 or split_fp32):
        raise ValueError('refused: c_inplace is the C feed written back into c_regs, with no store, keep, epilogue, '
                         'fp8 unpack or split (their buffers are indexed by row and column, not by group)')
    if (a_regs is not None or b_regs is not None) and sg != 1:
        raise ValueError('refused: register-fed operands are measured in one simdgroup (section 132 part 4 (c))')
    # REGISTER-RESIDENT COLUMN REDUCTION (goal item 8): reduce=('col', 'sum'|'max') folds the D
    # accumulators in registers, never storing D, and writes the N column results to row 0 of C.
    # A lane's eight D slots are four columns (col .. col+3) on rows r0 and r0+8 (pos_b); the lanes
    # holding one column differ in lane bits 1, 2 and 4. So: fold slot j over the row tiles in
    # ascending order (then the r0+8 half into the r0 half), apply the measured column butterfly
    # (xor masks 2, 4, 16; tensorreduce.COLUMN_BUTTERFLY_MASKS), and store slots 0..3 with the
    # per-lane column index and a row stride of zero - the bias load's addressing, as a store. Every
    # lane of a column group holds the same result, so their writes to row 0 agree.
    if reduce is not None:
        if tuple(reduce) not in (('col', 'sum'), ('col', 'max')):
            raise ValueError('refused: reduce is (col, sum) or (col, max)')
        if (M % 16 or N % 16 or accumulate or epilogue or grid != 1 or sg != 1 or split_fp32 or keep or not store
                or a_type in ('int8', 'uint8') or saturate):
            raise ValueError('refused: a register-resident reduction is implemented for whole tiles, one simdgroup, '
                             'no accumulate, epilogue, grid, split or hand-off')
    if b_regs is not None and (accumulate or grid != 1 or split_fp32):
        raise ValueError('refused: a register-fed B is implemented without accumulate, grid or split')
    if a_regs is not None and transA and (accumulate or grid != 1):
        raise ValueError('refused: mode At is implemented without accumulate or grid (its C relabels, part 4)')
    # SPLIT fp32 (Set A item 10). The accelerator keeps 10 explicit mantissa bits of an fp32 operand
    # (recon section 33). Veltkamp's split with 2^13 + 1 gives hi = c - (c - x), c = 8193 x: eleven
    # significant bits, which the truncation leaves exact, and lo = x - hi, which it keeps to eleven.
    # Three MMAs per K step, hi.hi then hi.lo then lo.hi, recover about 2^-20 relative accuracy
    # (lo.lo, near 2^-22, is dropped). Subtraction is fadd of an fmul by -1: both forms are measured,
    # while the source-negate modifier is not. Seven ALU operations per element, RNE, no contraction.
    # SATURATING int8 (Set A item 4): every MMA issue that takes C carries the saturate bit (mmaenc,
    # byte-exact against Apple's `_saturate` spelling), so D = clip(clip(P0 + P1) + P2 ...) in int32,
    # clipped at each step as recon section 136 part 6 measured. The first issue is the no-C form,
    # whose sixteen int8 products cannot leave int32, so it needs no bit (and the no-C saturating
    # encoding is not reproduced). A saturating ACCUMULATE seeds D with C (tlower's own C-tile loads,
    # into the accumulator, before the K chain) and saturates EVERY issue, the first included - the
    # only way a saturation is observable, since sixteen int8 products stay far inside int32 - and
    # drops the wrapping iadd that a plain accumulate adds after the chain. Whole tiles only.
    # A RUNTIME K LOOP (performance item 2). kloop=True peels K slice 0 exactly as the unrolled body
    # emits it (the no-C first issue included) and runs slices 1..KT-1 in a counted loop whose body is
    # ONE slice: loads from index registers that advance by one slice per iteration (A by 16 elements,
    # B by 16 rows), with the displacements of slice 0, then the MMAs with C. The latch is this
    # repository's EXECUTED counted-loop shape (cc on tools/g17loop.py's loop_ir: a self-referencing
    # increment, op577, cmp.6 keep, op582, op458, and op577 after the loop; counted loops of 1 to 64
    # trips returned the right value on hardware). Apple's own matmul2d loops over K the same way
    # (results/g17-tensor-speed-v1/apple_*: a 32-element K step, a peeled final slice, 5,142 bytes).
    # The trip count is a constant below the cmp's 8-bit immediate and the mask stack, and a static
    # check below refuses the body unless the increment is the ONLY writer of the counter inside it.
    if kloop_chunk is not None:
        # Preserve a fresh 64-K product followed by an FP32 fold, rather than
        # carrying the MMA accumulator through the whole contraction. Four
        # measured 16-K slices form each partial; the ordinary fadd folds it.
        if (kloop_chunk != 64 or not kloop or K < 128 or K % 64 or kloop_unroll != 1
                or a_type not in ('half', 'bfloat', 'float') or b_type not in ('half', 'bfloat', 'float')
                or any(e[0] == "mx" for e in epilogue) or split_fp32 or saturate or reduce or a_regs is not None or b_regs is not None
                or c_regs is not None or transA or transB or split_k != 1):
            raise ValueError('refused: kloop_chunk=64 needs a plain floating memory K loop, K a multiple of 64 >=128, without feeds, transpose, split or reduction')
    if kloop:
        _KT = -(-(K // split_k if isinstance(split_k, int) and split_k >= 1 else K) // 16)  # the loop runs the per-threadgroup K slice K/G
        # b_regs and reduce arrived on a parallel branch (feed modes) and were never composed with the
        # loop: a register-fed B has no per-slice memory step, and the column reduction's per-group
        # plan was not exercised under the one-set loop plan, so both are refused until measured.
        # fp8 operands (MM 25.128): the slice loads bytes like int8 and the four op17642 unpacks sit in the
        # body after them, waiting on the loop's load slot; nothing in the latch touches them.
        # THE BOUND IS THE TRIP COUNT (MM 25.144.1): the latch compares the counter with an 8-bit immediate, and an
        # unrolled loop takes U slices per trip after a peel of (KT - 1) % U + 1, so K 8192 (512 slices) runs 170 trips at
        # U 3; _check_counted_loop still verifies 1..255 on the emitted bytes
        _U = 4 if kloop_chunk else kloop_unroll if isinstance(kloop_unroll, int) and kloop_unroll >= 1 else 1
        _trips = (_KT - ((_KT - 1) % _U + 1)) // _U if _KT >= 1 else 0
        if (K % 16 or _KT < 2 or _trips > 255 or split_fp32 or a_regs is not None or transA or transB
                or (saturate and accumulate) or b_regs is not None or reduce):
            raise ValueError('refused: kloop is implemented for K a multiple of 16 with 2..256 slices, memory '
                             'operands without transpose bits, no split-fp32, saturating seed, '
                             'register-fed B or reduction')
    if saturate and a_type not in ('int8', 'uint8'):
        raise ValueError('refused: saturate is implemented for int8 GEMMs')
    if saturate and accumulate and (M % 16 or N % 16):
        raise ValueError('refused: a saturating accumulate is implemented for whole tiles')
    seeded = bool(saturate and accumulate)
    if split_fp32 and (a_type, b_type) != ('float', 'float'):
        raise ValueError('refused: split_fp32 applies to float x float')
    if split_fp32 and (a_regs is not None or transA or transB):
        raise ValueError('refused: split_fp32 is implemented for memory operands without transpose bits')
    if not store and not keep and not c_inplace:
        raise ValueError('refused: store=False discards the result unless keep=True hands the registers on')
    # REGISTER EPILOGUE (Set A item 6), applied to every D tile in the listed order after the chain
    # (and after an accumulate add) and before the store or the register hand-off:
    #   ('bias', rank, byte_offset)  D[r][c] += bias[c], bias an fp32 vector at that binding rank and offset
    #   ('scale', fp32_bits)         D *= scale
    #   ('relu',)                    D = max(D, 0)
    #   ('exp2',)                    D = 2**D (op1272, within one ulp of exact; 'exp' is ('scale', log2 e) then this)
    #   ('gelu',)                    D = D * sigmoid(1.702 D), the compiler's memory-stage GELU instruction for
    #                                instruction (tensorreduce.emit_row_gelu): t = D*1.702, t = t*(-1/ln 2), t = 2**t,
    #                                t = t + 1, t = 1/t, D = D*t. Three constant registers and one temporary shared by
    #                                every slot, so its register cost is four whatever the tile count
    # A lane's eight D slots are four consecutive columns (col .. col+3) on rows m and m+8 (the pos_b
    # map), so the bias is tlower's own C-tile load with a row stride of zero. The ALU forms are cc's
    # own (epienc). Measured tiles only: partial tiles would read the bias past its end.
    #   ('fp8', 'e4m3fn'|'e5m2')    LAST step only: D is packed to fp8 (op13618, RNE, no saturation, NaN to
    #                                the positive canonical NaN) and stored as BYTES, C row-major with ldc
    #                                bytes per row at C_OFF: a lane's four columns of row m (and of m+8)
    #                                are one 32-bit word, stored by op17202 at word index (idxC + ...)/4
    epilogue = tuple(tuple(e) for e in epilogue)
    # SOFTWARE MICROSCALING (Set A item 9a), a LEADING step ('mx', sa_rank, sa_off, sb_rank, sb_off): OCP MX
    # block scaling of both operands, one power-of-two factor per 32-element K block per A row (table SA)
    # and per B column (table SB). The tables are PREDECODED fp32 (the host turns each E8M0 code e into
    # 2^(e-127)), SA[b][row] at sa_off + 4 (b M + row) of binding sa_rank, SB[b][col] at sb_off + 4 (b N + col).
    # The placement is recon section 138's cheapest exact one, POST-MMA fp32 scaling: each block's two MMAs
    # run into a zeroed temporary (the first is the no-C form), then T = (T * sa) * sb and D += T. The
    # factors are powers of two, so both multiplies are exact unless a value leaves the fp32 normal range
    # (the ALU flushes subnormals), and D += T is the one rounding per block that section 138 measured
    # (an fma gives the same value: its product is exact for the same reason).
    # SCALE-CODE TRANSPORT (production row P5, MM 25.128): ('mx', sa_rank, sa_off, sb_rank, sb_off, 'e8m0')
    # reads the E8M0 CODE BYTES, SA[b][row] at sa_off + b M + row and SB[b][col] at sb_off + b N + col, and
    # decodes each IN THE KERNEL: the fp32 pattern (e << 23) + ([e == 255] << 22). Codes 1..254 give
    # 2^(e-127) exactly; code 255 gives the quiet NaN 0x7fc00000, the OCP MX meaning (the block is NaN).
    # Code 0 (2^-127) would decode to +0, not 2^-127: 2^-127 is an fp32 subnormal and the ALU flushes
    # it (MM 0.11), so no single fp32 factor can carry it. The HOST refuses code 0 by name
    # (g17tensorcommonruntime.check_e8m0_codes); the kernel's flush-to-zero reading of it is measured
    # there as the reason, not admitted. A lane's four SB columns are one aligned word (col is a
    # multiple of 4); its SA row byte sits in the aligned word at row & ~3, at bit 8 (row & 3), which a
    # register shift (op17014, logical on hardware) extracts.
    mx = epilogue[0] if epilogue and epilogue[0][0] == 'mx' else None
    mx_codes = False
    # MM 25.144.1: 2, 3 or 4 slices per trip (MLX's steel fp16 GEMM runs four), each slice's loads in its own slot 1 + kd
    if kloop_unroll != 1 and (not kloop or kloop_unroll not in (2, 3, 4) or K // 16 < kloop_unroll + 1 or mx
                              or a_type in FP8 or b_type in FP8):
        raise ValueError('refused: kloop_unroll is 1, or 2-4 on a plain (non-microscaled, non-fp8) K loop of more slices')
    if mx is not None:
        epilogue = epilogue[1:]
        mx_codes = len(mx) == 6 and mx[5] == 'e8m0'
        if len(mx) != 5 and not mx_codes:
            raise ValueError('refused: the mx step is (mx, sa_rank, sa_off, sb_rank, sb_off[, e8m0])')
        if K % 32 or M % 16 or N % 16:
            raise ValueError('refused: microscaling is implemented for whole 16x16 tiles and whole 32-element K blocks')
        if a_type in ('int8', 'uint8', 'float') or accumulate or split_fp32 or saturate or sg != 1 or transA or transB or a_regs is not None:
            raise ValueError('refused: microscaling is implemented for 16-bit and fp8 memory operands, one simdgroup, '
                             'no transpose, accumulate or split')
        if mx[2] % 4 or mx[4] % 4:
            raise ValueError('refused: the scale tables are read as aligned words: their offsets must be multiples of 4')
        # register-fed B and the column reduction arrived on parallel branches and were never composed
        # with the per-block scale step (its temporaries and k % 2 issue pattern), so refused. The K loop
        # composes with the CODE form only (MM 25.128): its body is one 32-element block, two slices.
        if b_regs is not None or reduce:
            raise ValueError('refused: microscaling is not composed with register-fed B or a reduction')
        if kloop and not mx_codes:
            raise ValueError('refused: microscaling inside the K loop reads E8M0 codes (the e8m0 form); '
                             'the predecoded fp32 tables are the straight-line form only')
        if kloop and (K // 32 < 2 or K // 32 - 1 > 255):
            raise ValueError('refused: a microscaled K loop runs blocks 1..K/32-1, so 2..256 blocks')
    for e in epilogue:
        if e[0] not in ('bias', 'scale', 'relu', 'exp2', 'gelu', 'fp8', 'half', 'half_kv', 'requant') or len(e) != {'bias': 3, 'scale': 2, 'relu': 1, 'exp2': 1, 'gelu': 1, 'fp8': 2, 'half': 1, 'half_kv': 1, 'requant': 5}[e[0]]:
            raise ValueError('refused: unknown epilogue step %r' % (e,))
    #   ('half',)                    LAST step only (MM P13, the named NARROWING EPILOGUE; no MMA accumulates in
    #                                16 bits, section 25.13): D is narrowed to half by op1016 (RNE, the register
    #                                chain's own conversion) and stored as HALVES, C row-major with ldc halves per
    #                                row at C_OFF: a lane's four columns of row m (and of m+8) are two 32-bit words,
    #                                each stored by op17202 (the fp8 path's word store) at word index (idxC + ...)/2
    half_out = bool(epilogue) and epilogue[-1] == ('half',)
    if any(e[0] == 'half' for e in epilogue):
        if not half_out or sum(e[0] == 'half' for e in epilogue) != 1 or any(e[0] == 'fp8' for e in epilogue):
            raise ValueError('refused: the half narrowing is one final epilogue step, not combined with fp8')
        if keep or not store:
            raise ValueError('refused: the half narrowing stores halves; it cannot hand fp32 registers on')
        if ldc % 2 or offsets[2] % 4:
            raise ValueError('refused: the half narrowing stores 32-bit words: ldc must be even and the C offset a multiple of 4 bytes')
        if reduce:
            raise ValueError('refused: the half narrowing stores the GEMM; a reduction stores none')
    # INT32 -> INT8/UINT8 REQUANTIZATION IN THE GEMM'S OWN DISPATCH (production row P6, machine model
    # 25.130). ('requant', signed, zero_point, scale, bias), the ONLY step of an int8 x int8 GEMM:
    #   scale  ('const', fp32_bits) | ('tensor', rank, byte_off) | ('channel', rank, byte_off)
    #   bias   None | ('channel', rank, byte_off)   int32, one per output column
    # Per element, exactly: b = acc + bias[col] (int32, wrapping, op10282); f = float(b) (op11179, RNE);
    # y = f * scale (op3290, RNE); r = rint(y) (op3770, half to even); r = min(max(r, lo - zp), hi - zp)
    # (op9700 fmax then fmin, exact on integer-valued floats); q = int(r) (op9320, exact: r is an integer
    # in [-255, 255]); q += zp (op10282). So q = clamp(rint(float(acc + bias) * scale) + zp, lo, hi), with
    # [lo, hi] = [-128, 127] signed or [0, 255] unsigned. Clamping BEFORE the narrowing keeps op9320's
    # unmeasured mode operand (saturation, rounding) out of the result. A lane's four columns of row m
    # (and of m+8) are packed into one word, byte j = q[col + j] & 255 (op423, op14391, op10282), and
    # stored by op17202 at word index (C_OFF + row * ldc + col) / 4: C is M x N bytes, ldc bytes per row.
    # With keep=True the two packed words (rows m, m+8) are left in acc+0 and acc+1: exactly the int8 A
    # tuple of tile (mi, k = ni) of a next int8 GEMM (four consecutive K bytes of rows m and m+8, the
    # layout tlower's own one-word int8 load produces).
    rq = next((e for e in epilogue if e[0] == 'requant'), None)
    if rq is not None:
        if len(epilogue) != 1:
            raise ValueError('refused: requant is the only epilogue step (its bias is inside it; nothing may follow it)')
        if (a_type, b_type) != ('int8', 'int8'):
            raise ValueError('refused: requant follows an int8 x int8 GEMM (int32 accumulator); uint8 operands are not lowered')
        # the epilogue runs once per tile after the K loop closes, and the C index already carries the row grid,
        # simdgroup and column grid offsets, so kloop, grid, sg and grid_n compose (MM 25.165). split_k does not:
        # its partials are reduced outside this body, and a requantized partial is not a partial of the result.
        if (accumulate or saturate or split_fp32 or split_k != 1 or reduce or M % 16 or N % 16
                or transA or transB or c_regs is not None or b_regs is not None):
            raise ValueError('refused: requant is implemented for a whole-tile int8 GEMM with no accumulate, '
                             'saturation, K split, reduction, transpose or C/B register feed')
        _, rq_signed, rq_zp, rq_scale, rq_bias = rq
        if type(rq_signed) is not bool or type(rq_zp) is not int:
            raise ValueError('refused: requant signed is a bool and zero_point an int')
        if not ((-128 <= rq_zp <= 127) if rq_signed else (0 <= rq_zp <= 255)):
            raise ValueError('refused: requant zero_point %d is outside the output range' % rq_zp)
        if not (isinstance(rq_scale, tuple) and ((rq_scale[0] == 'const' and len(rq_scale) == 2) or
                                                  (rq_scale[0] in ('tensor', 'channel') and len(rq_scale) == 3))):
            raise ValueError('refused: requant scale is (const, bits), (tensor, rank, off) or (channel, rank, off)')
        if rq_bias is not None and not (isinstance(rq_bias, tuple) and len(rq_bias) == 3 and rq_bias[0] == 'channel'):
            raise ValueError('refused: requant bias is None or (channel, rank, off)')
        for spec in (rq_scale, rq_bias):
            if spec is not None and spec[0] != 'const' and (spec[2] % 4 or spec[2] < 0):
                raise ValueError('refused: requant vectors are fp32/int32 words at a 4-byte offset')
        # A VECTOR PAST THE LOAD FIELD: its base is folded into the index register once (a per-channel vector's column
        # index gains base/4 words, the per-tensor scale's zero index becomes base/4), and the displacement is the rest
        rq_chan = [spec[2] for spec in (rq_scale, rq_bias) if spec is not None and spec[0] == 'channel']
        rq_vbase = min(rq_chan) if rq_chan and max(rq_chan) + 4 * N > 32767 else 0
        if rq_chan and max(rq_chan) - rq_vbase + 4 * N > 32767:
            raise ValueError('refused: the per-channel requant vectors must lie within one load field of each other')
        rq_tbase = rq_scale[2] // 4 if rq_scale[0] == 'tensor' and rq_scale[2] + 4 > 32767 else 0
        if ldc % 4 or offsets[2] % 4:
            raise ValueError('refused: requant stores 32-bit words: ldc and the C offset must be multiples of 4 bytes')
    # ('half_kv',) and P13's ('half',) narrow and store identically (RNE op1016, row-major halves, op17202 word
    # stores); they are two lowerings kept apart because each carries its own hardware receipts pinned to its
    # own bytes (P13: results/g17-p13-classes-v1; P7: results/g17-tensor-attnfused-v1). Unifying them means
    # re-dispatching one side's receipts.
    # HALF-KV OUT (production row P7, MM 25.129): ('half_kv',) as the LAST step narrows every D element to
    # half with op1016 (RNE, the register feed's narrowing) and stores C as HALVES, row-major with ldc
    # halves per row at byte offset C_OFF. A lane's four columns of row m become two words (halves
    # 0-1 in the first register's L and H, 2-3 in the second's), in place as the fp8 pack is, and each
    # word goes out through op17202, the fp8 quantize-out's measured word store, at word index
    # (C_OFF + 2 (row ldc + col)) / 4. This is how a projection writes K or V in the operand layout the
    # attention bodies read as their half B, so the fused kernel needs no relayout.
    half_kv = any(e[0] == 'half_kv' for e in epilogue)
    if half_kv:
        if epilogue[-1][0] != 'half_kv' or sum(e[0] in ('half', 'half_kv') for e in epilogue) != 1 or any(e[0] in ('fp8', 'requant') for e in epilogue):
            raise ValueError('refused: half_kv out is one final epilogue step, not combined with fp8 out')
        if keep or not store or reduce:
            raise ValueError('refused: half_kv out stores halves; it cannot hand fp32 registers on or reduce')
        if ldc % 2 or offsets[2] % 4:
            raise ValueError('refused: half_kv out stores 32-bit words: ldc must be even and the C offset a multiple of 4 bytes')
        if accumulate or grid != 1 or sg != 1 or split_fp32 or kloop or a_type in ('int8', 'uint8'):
            raise ValueError('refused: half_kv out is implemented for a plain whole-tile fp32 GEMM in one simdgroup and threadgroup')
    fp8_out = next((e[1] for e in epilogue if e[0] == 'fp8'), None)
    if fp8_out is not None:
        if epilogue[-1][0] != 'fp8' or sum(e[0] == 'fp8' for e in epilogue) != 1 or fp8_out not in ('e4m3fn', 'e5m2'):
            raise ValueError('refused: fp8 quantize-out is one final epilogue step, format e4m3fn or e5m2')
        if keep or not store:
            raise ValueError('refused: fp8 quantize-out stores bytes; it cannot hand fp32 registers on')
        if ldc % 4 or offsets[2] % 4:
            raise ValueError('refused: fp8 quantize-out stores 32-bit words: ldc and the C offset must be multiples of 4 bytes')
        if reduce:
            raise ValueError('refused: fp8 quantize-out stores the GEMM; a reduction stores none')
    if epilogue and rq is None and (M % 16 or N % 16 or a_type in ('int8', 'uint8')):
        raise ValueError('refused: the register epilogue is implemented for fp32 accumulators of whole 16x16 tiles')
    if not isinstance(registers, int) or registers < 1 or registers > registerdomain.REGISTER_COUNT:
        raise ValueError("register budget %r exceeds the allocatable R0..R125 namespace" % (registers,))
    bad_reserved = registerdomain.invalid_registers(reserved)
    if bad_reserved:
        raise ValueError("reserved register(s) %s are outside the allocatable R0..R125 namespace" %
                         ", ".join("R%d" % r for r in bad_reserved))
    A_BIND, B_BIND, C_BIND = binds; A_OFF, B_OFF, C_OFF = offsets
    if kloop_bases is None:
        kloop_bases = KLOOP_BASES
    if kloop_bases and (not kloop or transA or transB or ES[a_type] != 2 or ES[b_type] != 2 or mx or split_fp32):
        kloop_bases = False                                       # only the plain 16-bit K loop carries row bases
    DISP_LIMIT = 32767                                            # the fragment load's byte displacement field (DISP_MAX below)
    # A REGISTER-HELD B OFFSET (production row P7, n key blocks; MM 25.114.3). b_index=(reg, advance):
    # physical register `reg` holds an ELEMENT offset into B that this body adds to its B index
    # (idxB += reg, op10282) right after the index prologue, so every B load reads
    # base + (idxB + reg + row/col terms) x width + displacement; the immediate offset in `offsets`
    # stays as it was. After the body, reg += advance elements (op10279 up to 255, else op11842 into
    # the body's scratch and op10282). index_init: registers this body first sets to 0 (op11842).
    # Every such register must be in `reserved` (no body may allocate it), and the caller keeps it
    # from every other writer between bodies. So one body's bytes serve every key block: block j's
    # K and V addresses differ only in the register, never in an immediate.
    if any(int(r) not in set(reserved) for r in index_init):
        raise ValueError('refused: every index_init register must be reserved from the body')
    if b_index is not None:
        b_index = (int(b_index[0]), int(b_index[1]))
        if b_index[0] not in set(reserved):
            raise ValueError('refused: a B index register must be reserved from the body')
        # transB IS ADMITTED (MM 25.114.4): the stored B^T rows are read at (idxB + terms) x width + disp
        # exactly like untransposed B, and a key block's offset only ever entered as B_OFF in disp, so
        # adding a contiguous block's element offset to idxB moves the transposed read the same way
        # A SIMDGROUP SPLIT KEEPS B SHARED (MM 25.144.8): the split offsets idxA and idxC only, and every lane advances
        # its own copy of the index register by the same step, so all simdgroups read the same key block. Admitted
        # for 2 or 4 simdgroups, accumulating (c_inplace) or storing: M2's QK body (64x16x128, transB, K from the
        # cache, head and slice grid, sg 4) is bit-exact on hardware (25.144.2)
        if (b_regs is not None or kloop or grid != 1 or grid_n != 1 or split_k != 1 or sg not in (1, 2, 4)
                or not 0 <= b_index[1] < (1 << 31)):
            raise ValueError('refused: b_index is a memory B operand, one simdgroup and threadgroup, no K loop, advance >= 0')
    """transA: A is stored as A^T (K x M, leading dimension lda); transB: B stored as B^T (N x K, leading dimension ldb).
    The library's way (section 37): load the stored matrix's rows exactly as the untransposed operand of the OTHER side would
    be loaded (rows k, k+8 of A^T for A; rows n, n+8 of B^T for B) and set the MMA's transpose type bit."""
    # GRID-PARALLEL ROWS: `grid` threadgroups split M, threadgroup t owning rows [t*M/grid, (t+1)*M/grid).
    # The index is threadgroup_position_in_grid.x (SR156, the SR the measured (130,156) tensor metadata
    # class already declares), read like the simdgroup index and scaled the same way. `M` is the whole
    # matrix; each threadgroup runs the same body on its block.
    # THE HEAD GRID (MM 25.135): head_index=(eA, eB, eC) adds t * eA, t * eB and t * eC ELEMENTS to idxA,
    # idxB and idxC, where t is threadgroup_position_in_grid.x (SR_TG_X, read as the row grid reads it).
    # Threadgroup t then runs the same body on its own head's operands, each a fixed stride further into
    # its buffer: one image for every head, with no per-head immediate. Each stride is decomposed into
    # shift-adds, as the row grid's leading dimension is. Absent, nothing is emitted and every body is
    # byte-identical to what it was.
    if head_index is not None:
        head_index = tuple(int(e) for e in head_index)
        if len(head_index) != 3 or any(e < 0 or e >= (1 << 24) for e in head_index):
            raise ValueError('refused: head_index is three element strides, 0 <= stride < 2^24')
        # A SIMDGROUP SPLIT COMPOSES WITH THE HEAD GRID (MM 25.144.8): the split's rows offset idxA/idxC by s rows,
        # and the head (and slice) offset is added once per threadgroup on top - two independent adds, each read
        # from its own system register (SR_SIMD_GRP, SR_TG_X), sequential on wait slot 1
        # a register-fed A (MM 25.163) has no A index for a head stride to scale: its stride must be zero
        a_fed = a_regs is not None and c_regs is not None and not head_index[0] and sg == 1
        if grid != 1 or grid_n != 1 or split_k != 1 or sg not in (1, 2, 4) or kloop or (a_regs is not None and not a_fed) or b_regs is not None or (c_regs is not None and head_index[2]):
            # a register accumulator (c_regs, MM 25.144.8) takes the head grid on A and B only: its C is registers
            raise ValueError('refused: head_index is a memory-operand body in one, two or four simdgroups, without the row, column or K grid, a K loop or a register feed')
    # THE KV SPLIT (MM 25.114.6): head_slices=(S, (sA, sB, sC)) decomposes the ONE threadgroup id as
    # t = head * S + slice: head_index's strides scale head = t >> log2 S and (sA, sB, sC) scale
    # slice = t & (S - 1), as the column and K grids decompose theirs. S = 1 is the head grid itself.
    if head_slices is not None:
        if head_index is None:
            raise ValueError('refused: head_slices extends head_index')
        S_slices, slice_index = int(head_slices[0]), tuple(int(e) for e in head_slices[1])
        # 256 slices (MM 25.185): the slice is op426's AND of the 16-bit id with S - 1, an 8-bit immediate, so 255 is
        # the widest mask; the head is the shift of the 16-bit id, unbounded here
        if S_slices < 1 or S_slices & (S_slices - 1) or S_slices > 256 or len(slice_index) != 3 or \
                any(e < 0 or e >= (1 << 24) for e in slice_index):
            raise ValueError('refused: head_slices is (a power of two 1..256, three element strides 0..2^24)')
        if c_regs is not None and slice_index[2]:
            raise ValueError('refused: a register accumulator (c_regs) has no C slice stride')
        if S_slices == 1:
            head_slices = None
    if not isinstance(grid, int) or grid < 1 or grid > 256:
        raise ValueError('refused: grid must be 1..256 threadgroups (the index mask is eight bits)')
    if grid > 1:
        if M % (16 * grid): raise ValueError('refused: %d threadgroups need M a multiple of %d (whole tiles per threadgroup)' % (grid, 16 * grid))
        if (M // (16 * grid)) & (M // (16 * grid) - 1): raise ValueError('refused: tile rows per threadgroup must be a power of two (the offset is a shift)')
    M_all = M; M = M // grid
    # GRID-PARALLEL COLUMNS (P3, whole-GPU execution): `grid_n` threadgroups split N, threadgroup t
    # owning columns [t*N/grid_n, (t+1)*N/grid_n). It shares SR_TG_X with the row grid (rows in the low bits, MM 25.157).
    # Unlike the row grid (which scales the offset by lda/ldc), a column offset is a DIRECT add to B's and
    # C's inner index (the same fold as a P10 offsetB/offsetC), so it needs no leading-dimension stride.
    if not isinstance(grid_n, int) or grid_n < 1 or grid_n > 256:
        raise ValueError('refused: grid_n must be 1..256 threadgroups (the index mask is eight bits)')
    if grid_n > 1:
        # the row grid DOES combine with the column grid (MM 25.157): tg = col*grid + row, the ROW group in the low
        # log2(grid) bits, so the row groups that read one column block of B run adjacently and share it in cache
        # (rows in the high bits put them grid_n threadgroups apart: B re-streamed per row group, slower in the graph)
        if grid != 1 and (grid & (grid - 1)):
            raise ValueError('refused: a row grid with a column grid needs grid a power of two (the tg-id split is a shift)')
        if N % (16 * grid_n): raise ValueError('refused: %d column threadgroups need N a multiple of %d (whole tiles per threadgroup)' % (grid_n, 16 * grid_n))
        if (N // (16 * grid_n)) & (N // (16 * grid_n) - 1): raise ValueError('refused: tile columns per threadgroup must be a power of two (the offset is a shift)')
    N_all = N; N = N // grid_n
    # SPLIT-K (P3, the K-chain-bound decode path): `split_k` = G threadgroups partition the K contraction,
    # threadgroup t computing the PARTIAL over K-slice [t*K/G, (t+1)*K/G) alone and writing it to output
    # slot t of a (G*M) x N buffer (idxC += t*M*ldc). The G partials are reduced OUTSIDE this body by an
    # ascending-t fp32 left fold (Piece A's gemm_reference(split_k=G): tools/g17decodestep.py); C is added
    # LAST in that reduce, so this body never accumulates. A's K-offset is a DIRECT add (t*ks, A's inner
    # index is stride 1, like grid_n); B's is ld-scaled (t*ks*ldb); C's slot is ld-scaled (t*M*ldc).
    if not isinstance(split_k, int) or split_k < 1 or split_k > 256:
        raise ValueError('refused: split_k must be 1..256 threadgroups (the index mask is eight bits)')
    if split_k > 1:
        # split_k combines with a SIMDGROUP split (MM 25.144.1): simdgroup s offsets its rows (idxA, idxC by
        # s*16*MT*ld) before the K group's offsets, and the partial slot is kg*M*ldc with M the threadgroup's rows
        if grid != 1:
            raise ValueError('refused: split_k does not combine with a row (M) grid yet')
        # split_k DOES combine with the column grid (grid_n): the threadgroup id decomposes as
        # kg*grid_n + col, the low log2(grid_n) bits the column group and the high bits the K group, so a
        # wide projection (N/grid_n <= 256 columns per threadgroup) also partitions its K contraction.
        if grid_n > 1 and (grid_n & (grid_n - 1)):
            raise ValueError('refused: split_k with a column grid needs grid_n a power of two (the tg-id split is a shift)')
        if accumulate:
            raise ValueError('refused: split_k adds C in the fold reduce, not in the partial body (accumulate must be False)')
        if K % (16 * split_k):
            raise ValueError('refused: %d K threadgroups need K a multiple of %d (every slice is whole 16-wide issues)' % (split_k, 16 * split_k))
    K_all = K; K = K // split_k
    MT, NT, KT = up(M) // 16, up(N) // 16, up(K) // 16
    integer = a_type in ('int8', 'uint8')
    if integer != (b_type in ('int8', 'uint8')): raise ValueError('int8 operands must be paired')
    if a_type in FP8 or b_type in FP8:
        if not all(t in FP8 or t == 'bfloat' for t in (a_type, b_type)):
            raise ValueError('refused: an fp8 operand pairs with fp8 or bfloat (the MMA is bf16 x bf16)')
        if transA or transB:
            raise ValueError('refused: fp8 with a transpose bit is measured by the recon but not implemented here yet')
        if a_regs is not None:
            raise ValueError('refused: fp8 operands come from memory; a register feed is fp32')
    if sg > 1:
        if M % (16 * sg): raise ValueError('refused: %d simdgroups need M a multiple of %d (row masks are immediate bounds, so every simdgroup must own whole tiles)' % (sg, 16 * sg))
        if (MT // sg) & (MT // sg - 1): raise ValueError('refused: rows per simdgroup must be a power of two tiles (the offset is a shift of the simdgroup index)')
    MT_all = MT; MT = MT // sg
    wa, wb = WORDS[a_type], WORDS[b_type]; ta, tb = 2 * wa, 2 * wb            # words per fragment row, and per operand tuple (two rows per lane)
    tiles = [(mi, ni) for mi in range(MT) for ni in range(NT)]
    # THE LOOP BODY USES ONE BUFFER SET (it is one slice), so loop mode plans with one: a second set
    # there only costs accumulators, and at M64 N128 it left one tile per group - 32 loops, each
    # re-loading A and B for a single MMA per slice.
    for sets in ((kloop_unroll,) if kloop else (2, 1)):
        for per_group in ((len(tiles),) if (keep or reduce) else range(len(tiles), 0, -1)):
            try:
                P = Plan(registers, set(reserved))
                if hoist is None:
                    lane, m, col, t1, t2 = (P.take(1) for _ in range(5)); idxA, idxB = (P.take(1) for _ in range(2))
                else:           # the kept registers are the caller's; idxB is the per-trip one, idxB0 its kept base
                    lane = hoist['lane']; m, col, t1, t2 = (P.take(1) for _ in range(4))
                    idxA, idxB = hoist['idxA'], P.take(1)
                # idxC2 = 2 idxC (masked fp32 loads use width 2). A c_inplace body addresses no C at all (MM 25.144.8):
                # its C is the kept registers, so it takes neither index nor the C base register - three registers
                # that decide whether a 16 x 128 body fits beside a 64-register accumulator
                if hoist is None or c_inplace:
                    idxC, idxC2 = (None, None) if c_inplace else (P.take(1), P.take(1))
                else:
                    idxC, idxC2 = hoist['idxC'], hoist['idxC2']
                if hoist is None:
                    idxA2, idxB2 = (P.take(1) for _ in range(2))                                                            # 2 idx (fp32 masked loads) or idx/2 (int8 one-word loads)
                else:
                    idxA2, idxB2 = hoist['idxA2'], P.take(1)
                sgreg = P.take(1) if sg > 1 else None
                tgreg = P.take(1) if (grid > 1 or grid_n > 1 or split_k > 1 or head_index is not None) else None
                mk = [P.take(1) for _ in range(3)]                       # mask scratch: rowmask, colmask, mask (H halves)
                bA, bB = (P.take(1) for _ in range(2))                   # per-tile base index registers when a displacement would not fit
                bC = None if c_inplace else P.take(1)
                cnt = P.take(1) if kloop else None                        # the trip counter: cmp.6's source field is six bits, so it is taken low
                acc0 = [P.take(8) for _ in range(per_group)]
                chunk_acc0 = [P.take(8) for _ in range(per_group)] if kloop_chunk else None
                # A c_inplace body (MM 25.144.8) holds operand registers only for ITS GROUP's rows and columns:
                # every group loads its own A rows and B columns anyway, so the other groups' buffers were dead
                # weight - 32 registers for a 16 x 128 body, which is what stops it fitting beside a 64-register
                # accumulator. Indexed by position in the group (_slot below); every other body keeps MT / NT.
                abuf = [[P.take(ta) if a_regs is None else None for _ in range(min(MT, per_group) if c_inplace else MT)]
                        for _ in range(sets)]
                bbuf = [[P.take(tb) if b_regs is None else None for _ in range(min(NT, per_group) if c_inplace else NT)]
                        for _ in range(sets)]
                ctmp = P.take(8) if accumulate and c_regs is None else None
                aunp = [P.take(4) for _ in range(MT)] if a_type in FP8 else None      # bf16 tuples the fp8 loads unpack into
                bunp = [P.take(4) for _ in range(NT)] if b_type in FP8 else None
                ebias = P.take(4) if any(e[0] == 'bias' for e in epilogue) else None
                aconv = {key: P.take(4) for key in a_regs} if a_convert else None
                bconv = {key: P.take(4) for key in b_regs} if b_convert else None
                if split_fp32:
                    ahi, alo = [P.take(8) for _ in range(MT)], [P.take(8) for _ in range(MT)]
                    bhi, blo = [P.take(8) for _ in range(NT)], [P.take(8) for _ in range(NT)]
                    sneg, sconst, stmp, swait = P.take(1), P.take(1), P.take(1), P.take(1)
                econst = {e: P.take(1) for e in epilogue if e[0] in ('scale', 'relu')}
                # GELU: alpha, -1/ln 2, 1.0 and the per-slot temporary (only when asked, so every other body is unchanged)
                egelu = tuple(P.take(1) for _ in range(4)) if any(e[0] == 'gelu' for e in epilogue) else None
                # the loop's index registers, B's per-slice step and the trip counter (loop mode only,
                # so the default allocation - and the default body - is unchanged)
                kA, kB, stepB = (P.take(1) for _ in range(3)) if kloop else (None,) * 3
                if kloop and cnt > 63: raise ValueError('out of registers: the loop counter needs R0..R63')
                # LOOP-CARRIED ROW BASES (kloop_bases, MM 25.144.1): one register per fragment row whose byte
                # displacement from kA / kB overflows the load field, set before the loop and advanced once per
                # trip, so the loop body recomputes no base (it was 32 of 53 instructions at M 256, K 2048)
                kbases = {}
                if kloop_bases:
                    for mi in range(MT):
                        for p in range(2):
                            if 2 * (16 * mi + 8 * p) * lda + 32 * kloop_unroll + A_OFF > DISP_LIMIT:
                                kbases[('A', 16 * mi + 8 * p)] = P.take(1)
                    for kd in range(kloop_unroll):
                        for p in range(2):
                            if 2 * (16 * kd + 8 * p) * ldb + 2 * N + B_OFF > DISP_LIMIT:
                                kbases[('B', 16 * kd + 8 * p)] = P.take(1)
                rtmp = P.take(1) if reduce else None
                idxQ = P.take(1) if (fp8_out or half_out or rq) else None   # word index of the fp8/int8 (idxC / 4) or half (idxC / 2) output
                if rq:                                           # rails, zero point, scale, bias, pack temporaries
                    rq_lo, rq_hi = P.take(1), P.take(1)
                    rq_zpr = P.take(1) if rq_zp else None
                    rq_sc = P.take(1) if rq_scale[0] == 'const' else P.take(4)
                    rq_b = P.take(4) if rq_bias else None
                    rq_ta, rq_tb = P.take(1), P.take(1)
                    rq_zero = P.take(1) if rq_scale[0] == 'tensor' else None
                    rq_wait = P.take(1) if rq_scale[0] != 'const' else None
                    # under a column grid the per-channel vectors are indexed by the GLOBAL column: col + the group's first
                    rq_col = P.take(1) if (grid_n > 1 or rq_vbase) and rq_chan else None
                idxH, hword = (P.take(1), P.take(1)) if half_kv else (None, None)   # idxC / 2, and the second word's index
                if mx:                                            # per-block temporaries, the two tables' fragments, a wait scratch
                    tacc0 = [P.take(8) for _ in range(per_group)]
                    msa, msb, mwait = P.take(8), P.take(4), P.take(1)
                    rowg = P.take(1) if grid > 1 else m          # the lane's row m, plus the threadgroup's first row
                    # the code form: the aligned SA word index (rowg & ~3), the byte's bit offset 8 (rowg & 3),
                    # and the decode's two temporaries (the extracted code, the NaN bit)
                    saq, ssh, mc, mt = (P.take(1) for _ in range(4)) if mx_codes else (None,) * 4
                    # the microscaled loop's scale index registers and their per-block steps
                    kSA, kSB, stepSA, stepSB = (P.take(1) for _ in range(4)) if (kloop and mx) else (None,) * 4
                break
            except ValueError: P = None
        if P: break
    # 'refused: ' so cc's route reports a capacity limit as the lowering's reason, not a compiler defect (M6's fuzzer)
    if not P: raise ValueError('refused: no register plan for %dx%d tiles in %d registers' % (MT, NT, registers))
    if kloop and (M % 16 or N % 16):
        raise ValueError('refused: kloop is implemented for whole 16x16 tiles (the loop body carries no masks)')
    groups = [tiles[i:i + per_group] for i in range(0, len(tiles), per_group)]
    if kloop_unroll > 1 and len(groups) > 1:
        # a second buffer set that splits the tiles over two groups runs the K loop twice and reloads
        # A in each: no fewer load round trips per MMA, so it is refused, not emitted
        raise ValueError('refused: kloop_unroll %d needs every tile in one register group (%d tiles, %d per group)'
                         % (kloop_unroll, len(tiles), per_group))
    out = bytearray(); ops = []
    def emit(op, b, what): out.extend(b); ops.append(dict(op=op, what=what, hex=b.hex()))
    if hoist is not None:
        idxB_trip, idxB = idxB, hoist['idxB0']        # the prologue computes the kept base; the body reads idxB_trip
    pro = bytearray(indexgen.prologue(lane, m, col, t1, t2, [r for r in (idxA, idxB, idxC) if r is not None],
                                      [ld for r, ld in ((idxA, lda), (idxB, ldb), (idxC, ldc)) if r is not None]))
    if sg > 1:                                                   # simdgroup s owns rows [s*16*MT, (s+1)*16*MT): idxA += s*16*MT*lda, idxC += s*16*MT*ldc
        srg = bytearray(indexgen.read_sr_lane(sgreg, slot=1)); srg[1] = 0x85; pro += bytes(srg)          # SR_SIMD_GRP: byte1 0x85 in the 4-byte form
        assert names[list(model.decode(bytes(srg), 0))[0].values[2][1]] == 'SR_SIMD_GRP'
        # the and16 mask is 3 for up to 4 simdgroups (the bytes every 1/2/4-simdgroup receipt ran) and
        # sg - 1 above: a mask of 3 at 8 simdgroups would fold simdgroups 4..7 onto 0..3's rows (MM P13)
        pro += ledgerenc.encode(426, {0: 'R%d' % t1, 1: 1 << 25, 2: 'R%dL' % sgreg, 3: 32, 4: 3 if sg <= 4 else sg - 1})   # t1 = s (waits on slot 1)
        pro += indexgen.shl(t1, t1, (16 * MT).bit_length() - 1)                                              # t1 = s * 16 MT
        for reg, ld in ((r, l) for r, l in ((idxA, 1 if transA else lda), (idxC, ldc)) if r is not None):     # with transA the simdgroup's rows are columns of the stored A^T: offset in elements, not rows
            for kk in [kk for kk in range(31) if ld >> kk & 1]:
                pro += indexgen.shl(t2, t1, kk); pro += faddenc.encode(reg, reg, t2, word=0, f1=0, f2=0, op=10282)
    # GRID AND SIMDGROUP SPLITS COMPOSE (performance item 3): threadgroup t owns 16*MT*sg rows, and
    # simdgroup s owns 16*MT of them, so the threadgroup offset advances by a whole threadgroup's rows,
    # not one simdgroup's. With sg == 1 the shift is unchanged, so grid-only bodies are byte-identical.
    if grid > 1:                                                 # threadgroup t owns rows [t*16*MT*sg, (t+1)*16*MT*sg): idxA += t*16*MT*sg*lda
        srt = bytearray(indexgen.read_sr_lane(tgreg, slot=1)); srt[1] = 0x9c; pro += bytes(srt)          # SR_TG_X: byte1 0x9c in the 4-byte form
        assert names[list(model.decode(bytes(srt), 0))[0].values[2][1]] == 'SR_TG_X'
        pro += ledgerenc.encode(426, {0: 'R%d' % t1, 1: 1 << 25, 2: 'R%dL' % tgreg, 3: 32,
                                      4: grid - 1 if grid_n > 1 else 255})                                   # t1 = t, or the row group tg & (grid-1) (waits on slot 1)
        pro += indexgen.shl(t1, t1, (16 * MT * sg).bit_length() - 1)                                         # t1 = t * 16 MT sg
        if mx: pro += indexgen.add(rowg, m, t1)                                                               # rowg = m + t * 16 MT
        for reg, ld in ((idxA, 1 if transA else lda), (idxC, ldc)):
            for kk in [kk for kk in range(31) if ld >> kk & 1]:
                pro += indexgen.shl(t2, t1, kk); pro += faddenc.encode(reg, reg, t2, word=0, f1=0, f2=0, op=10282)
    if head_index is not None and head_slices is not None:      # t = head * S + slice (the KV split)
        srt = bytearray(indexgen.read_sr_lane(tgreg, slot=1)); srt[1] = 0x9c; pro += bytes(srt)          # SR_TG_X: byte1 0x9c in the 4-byte form
        assert names[list(model.decode(bytes(srt), 0))[0].values[2][1]] == 'SR_TG_X'
        pro += ledgerenc.encode(426, {0: 'R%d' % t1, 1: 1 << 25, 2: 'R%dL' % tgreg, 3: 32, 4: S_slices - 1})   # t1 = slice (waits on slot 1)
        for reg, stride in ((r, st) for r, st in zip((idxA, idxB, idxC), slice_index) if r is not None):
            for kk in [kk for kk in range(31) if stride >> kk & 1]:
                pro += indexgen.shl(t2, t1, kk); pro += faddenc.encode(reg, reg, t2, word=0, f1=0, f2=0, op=10282)
        # tgreg holds ONLY its low half (the 16-bit SR_TG_X read above), so the shift reads that half: a 32-bit shift
        # read tgreg's high half, released and nonzero from scalar code before this prologue (g17emu, MM 25.145)
        pro += indexgen.shr16(t1, tgreg, S_slices.bit_length() - 1)                                        # t1 = head = t >> log2 S
        for reg, stride in ((r, st) for r, st in zip((idxA, idxB, idxC), head_index) if r is not None):
            for kk in [kk for kk in range(31) if stride >> kk & 1]:
                pro += indexgen.shl(t2, t1, kk); pro += faddenc.encode(reg, reg, t2, word=0, f1=0, f2=0, op=10282)
    elif head_index is not None:                                 # threadgroup t owns head t: idx{A,B,C} += t * stride
        srt = bytearray(indexgen.read_sr_lane(tgreg, slot=1)); srt[1] = 0x9c; pro += bytes(srt)          # SR_TG_X: byte1 0x9c in the 4-byte form
        assert names[list(model.decode(bytes(srt), 0))[0].values[2][1]] == 'SR_TG_X'
        pro += ledgerenc.encode(426, {0: 'R%d' % t1, 1: 1 << 25, 2: 'R%dL' % tgreg, 3: 32, 4: 255})        # t1 = t (waits on slot 1)
        for reg, stride in ((r, st) for r, st in zip((idxA, idxB, idxC), head_index) if r is not None):
            for kk in [kk for kk in range(31) if stride >> kk & 1]:
                pro += indexgen.shl(t2, t1, kk); pro += faddenc.encode(reg, reg, t2, word=0, f1=0, f2=0, op=10282)
    # THE COLUMN GRID (grid_n) AND THE K GRID (split_k) SHARE ONE SR_TG_X READ. The threadgroup id
    # decomposes as tg = kg*grid_n + col: the low log2(grid_n) bits are the column group col (each owning
    # 16*NT columns), the high bits the K group kg (each owning K/split_k of the contraction and slot kg of
    # the (split_k*M) x N partial buffer). Either grid alone is the whole id. tgreg holds the raw id.
    if (grid_n > 1 or split_k > 1) and grid == 1:                 # a row grid already read SR_TG_X into tgreg
        srt = bytearray(indexgen.read_sr_lane(tgreg, slot=1)); srt[1] = 0x9c; pro += bytes(srt)          # SR_TG_X: byte1 0x9c in the 4-byte form
        assert names[list(model.decode(bytes(srt), 0))[0].values[2][1]] == 'SR_TG_X'
    if grid_n > 1:                                               # col = tg & (grid_n-1); columns [col*16*NT, (col+1)*16*NT): idxB, idxC += col*16*NT
        # grid_n alone: tg = col < grid_n, so a mask of 255 (the N-tiled grid's measured byte) is identity;
        # combined with split_k the low log2(grid_n) bits are col, so the mask is grid_n-1.
        col_mask = grid_n - 1 if split_k > 1 else 255
        pro += ledgerenc.encode(426, {0: 'R%d' % t1, 1: 1 << 25, 2: 'R%dL' % tgreg, 3: 32, 4: col_mask})    # t1 = col (waits on slot 1)
        if grid > 1:
            pro += indexgen.shr16(t1, t1, grid.bit_length() - 1)                                               # col = tg >> log2(grid): rows are the low bits
        pro += indexgen.shl(t1, t1, (16 * NT).bit_length() - 1)                                            # t1 = col * 16 NT (columns per threadgroup)
        if rq and rq_col is not None: pro += indexgen.add(rq_col, col, t1)                                 # the per-channel vectors' column
        for reg in (idxB, idxC):                                  # a column offset is a DIRECT add (stride 1), unlike the row grid's ld-scaled shift
            pro += faddenc.encode(reg, reg, t1, word=0, f1=0, f2=0, op=10282)
    if split_k > 1:                                              # kg owns K-slice [kg*K, (kg+1)*K) (K already K/G) and slot kg of the (split_k*M)xN partial buffer
        if grid_n > 1:
            pro += indexgen.shr16(t1, tgreg, (grid_n).bit_length() - 1)                                    # t1 = kg = tg >> log2(grid_n) (tgreg already waited by the col op426; its low half only, as above)
        else:
            pro += ledgerenc.encode(426, {0: 'R%d' % t1, 1: 1 << 25, 2: 'R%dL' % tgreg, 3: 32, 4: 255})    # t1 = kg = tg (waits on slot 1)
        # A's K-offset is kg*K (a DIRECT add; A's inner index is stride 1, so mult = K); under transA A is
        # stored K x M and the K-slice is rows kg*K.., so mult = kg*K*lda. B's is kg*K*ldb; C's slot is kg*M*ldc.
        # Each product kg*mult is the set-bit sum of `mult` applied to kg (like the row grid's ld scaling).
        for reg, mult in ((idxA, K * lda if transA else K), (idxB, K * ldb), (idxC, M * ldc)):
            for kk in [kk for kk in range(31) if mult >> kk & 1]:
                pro += indexgen.shl(t2, t1, kk); pro += faddenc.encode(reg, reg, t2, word=0, f1=0, f2=0, op=10282)
    if mx_codes:                                                 # saq = rowg & ~3, ssh = 8 (rowg & 3)
        pro += indexgen.shr32(saq, rowg, 2); pro += indexgen.shl(saq, saq, 2)
        pro += indexgen.and32(ssh, rowg, 3); pro += indexgen.shl(ssh, ssh, 3)
    # FOLD A LARGE BUFFER OFFSET INTO ITS INDEX REGISTER ONCE (fold_offsets, MM 25.144.8). An offset past the
    # displacement field (DISP_MAX bytes) otherwise sends every load and store of that operand through based()'s
    # slow path: a shift-add chain for the tile's rows AND the offset re-added bit by bit, per tile, per body run -
    # 694 of M2's 908 looped QK+PV instructions a trip. Added here, after every other index term, the offset is paid
    # once and each displacement is the tile's own. Opt-in: the default keeps every existing body's bytes.
    if fold_offsets:
        for reg, off_name, t in ((idxA, 'A_OFF', a_type), (idxB, 'B_OFF', b_type), (idxC, 'C_OFF', 'float')):
            off = {'A_OFF': A_OFF, 'B_OFF': B_OFF, 'C_OFF': C_OFF}[off_name]
            if reg is None or off <= 32767:                     # DISP_MAX, the displacement field (assigned below)
                continue
            if off % ES[t]:
                raise ValueError('refused: fold_offsets needs the %s offset %d to be whole %s elements' % (off_name[0], off, t))
            pro += epienc.movimm(t1, off // ES[t]); pro += indexgen.add(reg, reg, t1)
            if off_name == 'A_OFF': A_OFF = 0
            elif off_name == 'B_OFF': B_OFF = 0
            else: C_OFF = 0
    if index_init:
        for r in index_init: pro += epienc.movimm(int(r), 0)                                           # the stream's B index registers start at 0
    if b_index is not None and hoist is None:
        pro += indexgen.add(idxB, idxB, b_index[0])                                                  # idxB += the key block's element offset (register)
    if idxC2 is not None:
        pro += indexgen.shl(idxC2, idxC, 1)
    if fp8_out or rq: pro += indexgen.shr32(idxQ, idxC, 2)      # byte offset m*ldc + col is a multiple of 4
    if half_out: pro += indexgen.shr32(idxQ, idxC, 1)           # half index m*ldc + col is even (col is a multiple of 4)
    if rq and rq_zero is not None:                              # the per-tensor scale's index: 0, or its folded base
        pro += epienc.movimm(rq_zero, rq_tbase) if rq_tbase else indexgen.and32(rq_zero, lane, 0)
    if rq and rq_col is not None and rq_vbase:                  # the per-channel vectors' base, in words
        if grid_n == 1: pro += indexgen.addi(rq_col, col, 0)
        pro += epienc.movimm(t1, rq_vbase // 4); pro += indexgen.add(rq_col, rq_col, t1)
    if half_kv: pro += indexgen.shr32(idxH, idxC, 1)           # halves m*ldc + col is even (ldc even, col a multiple of 4)
    for t, src, dst in ((a_type, idxA, idxA2),) + (((b_type, idxB, idxB2),) if hoist is None else ()):
        if t == 'float': pro += indexgen.shl(dst, src, 1)
        elif ES[t] == 1: pro += indexgen.shr16(dst, src, 1)                                                  # bytes / 2: the one-word load's width is 2
    hoisted_prologue = None
    if hoist is not None:
        # the invariant part goes to the caller; the body opens with the per-trip part
        hoisted_prologue = bytes(pro)
        pro = bytearray()
        if b_index is not None:
            pro += indexgen.add(idxB_trip, idxB, b_index[0])                                         # idxB = idxB0 + the key block's offset
        else:
            pro += indexgen.addi(idxB_trip, idxB, 0)
        if b_type == 'float': pro += indexgen.shl(idxB2, idxB_trip, 1)
        idxB = idxB_trip
    emit(0, bytes(pro), 'index prologue (lane R%d, m R%d, col R%d, idx R%d/%d/%s, 2idxC %s%s)' % (lane, m, col, idxA, idxB, 'R%d' % idxC if idxC is not None else '-', 'R%d' % idxC2 if idxC2 is not None else '-', ', sg R%d' % sgreg if sg > 1 else ''))
    if split_fp32:
        emit(11842, epienc.movimm(sconst, 0x46000400), 'split constant R%d = 8193.0' % sconst)
        emit(11842, epienc.movimm(sneg, 0xBF800000), 'split constant R%d = -1.0' % sneg)
    if a_convert:
        for key in sorted(aconv):
            for i in range(8):
                emit(1016, epienc.cvt_f32_to_f16('R%d%s' % (aconv[key] + i // 2, 'LH'[i % 2]), a_regs[key] + i),
                     'narrow D%s[%d] -> half' % (key, i))
    if b_convert:
        for key in sorted(bconv):
            for i in range(8):
                emit(1016, epienc.cvt_f32_to_f16('R%d%s' % (bconv[key] + i // 2, 'LH'[i % 2]), b_regs[key] + i),
                     'narrow B-fed D%s[%d] -> half' % (key, i))
    for e, reg in econst.items():
        emit(11842, epienc.movimm(reg, e[1] if e[0] == 'scale' else 0), 'epilogue constant R%d = %s' % (reg, e))
    if rq:
        import struct as _struct
        lo, hi = (-128, 127) if rq_signed else (0, 255)
        f32 = lambda v: _struct.unpack('<I', _struct.pack('<f', float(v)))[0]
        emit(11842, epienc.movimm(rq_lo, f32(lo - rq_zp)), 'requant lower rail R%d = %d.0 (lo - zp)' % (rq_lo, lo - rq_zp))
        emit(11842, epienc.movimm(rq_hi, f32(hi - rq_zp)), 'requant upper rail R%d = %d.0 (hi - zp)' % (rq_hi, hi - rq_zp))
        if rq_zpr is not None:
            emit(11842, epienc.movimm(rq_zpr, rq_zp & 0xFFFFFFFF), 'requant zero point R%d = %d' % (rq_zpr, rq_zp))
        if rq_scale[0] == 'const':
            emit(11842, epienc.movimm(rq_sc, rq_scale[1]), 'requant scale R%d = %#010x' % (rq_sc, rq_scale[1]))
    if egelu:
        import struct as _struct
        for reg, value in zip(egelu[:3], GELU_CONSTANTS):
            emit(11842, epienc.movimm(reg, _struct.unpack('<I', _struct.pack('<f', value))[0]),
                 'gelu constant R%d = %r' % (reg, value))
    DISP_MAX = 32767
    def const_times_ld(dst, n, ld):
        """dst = n * ld for a small n (<= 255) through the same shift-add decomposition the prologue uses"""
        b = bytearray(); b += indexgen.and32(t1, lane, 0); b += indexgen.addi(t1, t1, n)      # t1 = n
        bits = [k for k in range(31) if ld >> k & 1]
        b += indexgen.shl(dst, t1, bits[0])
        for k in bits[1:]: b += indexgen.shl(t2, t1, k); b += faddenc.encode(dst, dst, t2, word=0, f1=0, f2=0, op=10282)
        return bytes(b)
    def based(idx, base_reg, rows, ld, elem, rest_disp, what, limit=DISP_MAX, extra_rows=0):
        """(index register, displacement) for (rows + extra_rows) * ld elements + rest_disp bytes: the displacement when it fits the
        form's field (limit), else a base register = idx + rows*ld (and, if still needed, + extra_rows*ld) with the rest as displacement"""
        full = elem * (rows + extra_rows) * ld + rest_disp
        if full <= limit: return idx, full
        if elem * extra_rows * ld + rest_disp <= limit:
            emit(0, const_times_ld(base_reg, rows, ld) + faddenc.encode(base_reg, base_reg, idx, word=0, f1=0, f2=0, op=10282), 'base %s = idx + %d*%d' % (what, rows, ld))
            return base_reg, elem * extra_rows * ld + rest_disp
        b = const_times_ld(base_reg, rows + extra_rows, ld) + faddenc.encode(base_reg, base_reg, idx, word=0, f1=0, f2=0, op=10282)
        if rest_disp > limit:                                                                        # a buffer offset beyond the field: fold it into the base too (index units = elements)
            if rest_disp % elem: raise ValueError('residual displacement %d is not a multiple of the element size %d' % (rest_disp, elem))
            b += indexgen.and32(t1, lane, 0); b += indexgen.addi(t1, t1, 1)
            for k in [k for k in range(31) if (rest_disp // elem) >> k & 1]: b += indexgen.shl(t2, t1, k); b += faddenc.encode(base_reg, base_reg, t2, word=0, f1=0, f2=0, op=10282)
            rest_disp = 0
        emit(0, b, 'base %s = idx + %d*%d%s' % (what, rows + extra_rows, ld, '' if rest_disp else ' + offset'))
        return base_reg, rest_disp
    def mask_for_halves(p, rows_left, cols_left, first):
        """mask for a masked load of two fp32 elements (four halves): bit j = [row valid] & [col + (j >> 1) + 2*first < cols_left]
        via op612 on the doubled column coordinate (2 col + 4 first + j < 2 cols_left)"""
        b = bytearray()
        b += indexgen.shl(t1, m, 2); b += indexgen.addi(t1, t1, 32 * p); b += op612(mk[0], t1, 0, 4 * min(rows_left, 16))
        b += indexgen.shl(t2, col, 1); b += indexgen.addi(t2, t2, 4 * first); b += op612(mk[1], t2, 0, 2 * min(cols_left, 16))
        b += and16x16(mk[2], mk[0], mk[1])
        return bytes(b)
    def mask_for(p, rows_left, cols_left):
        """R<mk[2]>H bit j = [m + 8p < rows_left] & [col + j < cols_left], tile-relative bounds (each in 1..16, so op612's
        eight-bit hi suffices: rows use the x4 trick, 4(m+8p)+j < 4 rows_left  <=>  m+8p < rows_left)"""
        b = bytearray()
        b += indexgen.shl(t1, m, 2); b += indexgen.addi(t1, t1, 32 * p); b += op612(mk[0], t1, 0, 4 * min(rows_left, 16))
        b += op612(mk[1], col, 0, min(cols_left, 16))
        b += and16x16(mk[2], mk[0], mk[1])
        return bytes(b)
    # A REGISTER-FED A IS NEVER RELOADED, so its release belongs to the LAST GROUP that reads the row,
    # not to the last read inside each group: with 3 accumulators a 2x2 tile grid runs as groups
    # [(0,0),(0,1),(1,0)] and [(1,1)], and releasing row 1's A at (1,0) left (1,1) reading released
    # registers (an all-zero tile on hardware: Set C's stages run, chain_M32N32K64_3232half). A
    # loaded A is reloaded by every group, so the per-group rule stays right for it.
    last_group_of_row = {mi: g for g, group in enumerate(groups) for mi, ni in group}
    last_group_of_col = {ni: g for g, group in enumerate(groups) for mi, ni in group}   # a register-fed B, likewise
    for g, group in enumerate(groups):
        acc = {tile: acc0[i] for i, tile in enumerate(group)}
        tacc = ({tile: chunk_acc0[i] for i, tile in enumerate(group)} if kloop_chunk else
                {tile: tacc0[i] for i, tile in enumerate(group)} if mx else acc)
        if kloop_chunk:
            for tile in group:
                for i in range(8):
                    emit(11842, epienc.movimm(acc[tile] + i, 0), 'chunk fold initialize D%s[%d] = 0' % (tile, i))
        gm = sorted({mi for mi, ni in group}); gn = sorted({ni for mi, ni in group})
        # the operand buffer of row mi / column ni: its own under every other body, its place in the group under
        # c_inplace (the buffers are sized to the group there)
        aslot = {mi: (gm.index(mi) if c_inplace else mi) for mi in gm}
        bslot = {ni: (gn.index(ni) if c_inplace else ni) for ni in gn}
        if seeded:                                   # D = C before the chain, into the accumulator itself
            for (mi, ni) in group:
                for p in range(2):
                    ireg, disp = based(idxC, bC, 16 * mi, ldc, 4, 4 * (8 * p * ldc + 16 * ni) + C_OFF, 'C')
                    op, b = memenc.load(acc[mi, ni] + 4 * p, ireg, C_BIND, disp=disp, width=4, slot=6)
                    emit(op, b, 'seed D(%d,%d)p%d = C slot6' % (mi, ni, p))
        # THE SCHEDULE: (label, slice displacement index, in the loop body, opens the loop, closes it).
        # A microscaled loop peels block 0 (slices 0 and 1) and its body is ONE block, two slices, so
        # the per-block scale step closes each iteration (MM 25.128).
        peel = 4 if kloop_chunk else 2 if mx else (KT - 1) % kloop_unroll + 1 if kloop else 0   # slices before the loop
        if not kloop:
            schedule = [(k, k, False, False, False) for k in range(KT)]
        elif kloop_chunk:
            schedule = [(j, j, False, False, False) for j in range(4)] + [
                ('L%d' % j, j, True, j == 0, j == 3) for j in range(4)]
        elif mx:
            schedule = [(0, 0, False, False, False), (1, 1, False, False, False),
                        ('L0', 0, True, True, False), ('L1', 1, True, False, True)]
        elif kloop_unroll == 1:
            schedule = [(0, 0, False, False, False), ('L', 0, True, True, True)]
        else:
            # UNROLLED BY U (MM 25.124.6): the body is U slices, each in its own buffer set and load slot
            # (1..U), so all U slices' loads are in flight before the first MMA waits - the K chain pays
            # one load round trip per U slices, as Apple's matmul2d loop does (U = 2). The first
            # peel = (KT - 1) % U + 1 slices are peeled so the loop runs (KT - peel) / U whole trips.
            schedule = [(k, k, False, False, False) for k in range(peel)] + [
                ('L%d' % j, j, True, j == 0, j == kloop_unroll - 1) for j in range(kloop_unroll)]
        step = 64 if kloop_chunk else 32 if mx else 16 * kloop_unroll        # elements the loop advances per iteration
        held = []                                     # the unrolled body's MMAs, emitted after all of its loads
        loop_mma = [0]
        for k, kd, loop, opens, closes in schedule:
            if opens:
                first = 16 * peel if (kloop_unroll > 1 and not mx) else step   # the loop's first element (== step unless peel != U)
                emit(0, bytes(indexgen.addi(kA, idxA, first)) + const_times_ld(stepB, first, ldb)
                     + faddenc.encode(kB, idxB, stepB, word=0, f1=0, f2=0, op=10282)
                     + (const_times_ld(stepB, step, ldb) if first != step else b'') + bytes(epienc.movimm(cnt, 0)),
                     'loop init: kA = idxA + %d, stepB = %d*ldb, kB = idxB + stepB, cnt = 0' % (first, first)
                     + ('; stepB = %d*ldb' % step if first != step else ''))
                if mx:
                    emit(0, bytes(epienc.movimm(stepSA, M_all)) + bytes(epienc.movimm(stepSB, N))
                         + indexgen.add(kSA, saq, stepSA) + indexgen.add(kSB, col, stepSB),
                         'loop init: kSA = saq + %d, kSB = col + %d (block 1 of the code tables)' % (M_all, N))
                if kbases:
                    hb = bytearray()
                    for (opd, rows), reg in sorted(kbases.items()):
                        hb += const_times_ld(reg, rows, lda if opd == 'A' else ldb)
                        hb += faddenc.encode(reg, reg, kA if opd == 'A' else kB, word=0, f1=0, f2=0, op=10282)
                    emit(0, bytes(hb), 'loop row bases: ' + ', '.join('%s row %d -> R%d' % (o, r, g_) for (o, r), g_ in sorted(kbases.items())))
                loop_top = len(out)
            iA, iB = (kA, kB) if loop else (idxA, idxB)
            s = 0 if kloop_chunk else (kd if kloop_unroll > 1 else 0) if loop else k % sets
            sl = (1 + kd if (mx or kloop_unroll > 1) else 1) if loop else k % 7
            def row_load(dst, idx, idx2, base_reg, binding, es, words, rows_left, cols_left, row_base_rows, col_base, ld, sl, wm, what):
                """one fragment row (four elements) per lane: rows_left/cols_left are the tile-relative bounds (16 = full)"""
                masked = rows_left < 16 or cols_left < 16
                # THE OPERAND, NOT THE BINDING, NAMES THE OFFSET: keyed by binding, A and B read from ONE buffer
                # (P7's PV, P and V both in buffer 3) collapsed to B's offset, and A was read from V's
                # region (results/g17-tensor-attnfused-v1/fused_n2, first run; MM 25.129.2)
                boff = A_OFF if base_reg == bA else B_OFF
                if words == 2:                                                      # 16-bit: one two-word load
                    hkey = ('A' if base_reg == bA else 'B', row_base_rows)
                    if loop and not masked and hkey in kbases and 2 * col_base + boff <= DISP_MAX:
                        ireg, disp = kbases[hkey], 2 * col_base + boff                # the loop-carried row base
                    else:
                        ireg, disp = based(idx, base_reg, row_base_rows, ld, 2, 2 * col_base + boff, what)
                    if masked:
                        emit(0, mask_for(p, rows_left, cols_left), 'mask ' + what)
                        op, b = memenc.mload(dst, ireg, binding, INV['R%dH' % mk[2]], disp=disp, width=2, slot=sl, wait_mask=wm)
                    else: op, b = memenc.tload(dst, ireg, binding, disp=disp, width=2, slot=sl, wait_mask=wm)
                    emit(op, b, what)
                elif words == 4:                                                    # fp32: one four-word load, or two masked two-word loads on the doubled index
                    if masked:
                        ireg, disp = based(idx2, base_reg, row_base_rows, 2 * ld, 2, 4 * col_base + boff, what + '2')
                        for half in range(2):
                            emit(0, mask_for_halves(p, rows_left, cols_left, half), 'mask %s floats %d-%d' % (what, 2 * half, 2 * half + 1))
                            op, b = memenc.mload(dst + 2 * half, ireg, binding, INV['R%dH' % mk[2]], disp=disp + 8 * half, width=2, slot=sl, wait_mask=wm)
                            emit(op, b, what + ' words %d-%d' % (2 * half, 2 * half + 1))
                    else:
                        ireg, disp = based(idx, base_reg, row_base_rows, ld, 4, 4 * col_base + boff, what)
                        op, b = memenc.load(dst, ireg, binding, disp=disp, width=4, slot=sl, wait_mask=wm); emit(op, b, what)
                else:                                                               # int8: the library's one-word tensor load (op12656, width 1 = byte addressing) on the
                    ireg, disp = based(idx, base_reg, row_base_rows, ld, 1, col_base + boff, what)          # plain index (elements = bytes), masked form op12657 at a boundary
                    if masked:
                        emit(0, mask_for(p, rows_left, cols_left), 'mask ' + what)
                        op, b = memenc.mload1w(dst, ireg, binding, INV['R%dH' % mk[2]], disp=disp, width=1, slot=sl, wait_mask=wm)
                    else: op, b = memenc.tload1w(dst, ireg, binding, disp=disp, width=1, slot=sl, wait_mask=wm)
                    emit(op, b, what)
            for mi in (gm if a_regs is None else ()):
                for p in range(2):
                    wm = 0x01 if (g == 0 and k == 0) else 0   # first loads wait for the SR read (slot 0)
                    if transA: row_load(abuf[s][aslot[mi]] + wa * p, iA, idxA2, bA, A_BIND, ES[a_type], wa, K - 16 * kd - 8 * p + 8 * p if False else (K - 16 * kd), M - 16 * mi, 16 * kd + 8 * p, 16 * mi, lda, sl, wm, 'A(%d,k%s)p%d' % (mi, k, p))
                    else: row_load(abuf[s][aslot[mi]] + wa * p, iA, idxA2, bA, A_BIND, ES[a_type], wa, M - 16 * mi, K - 16 * kd, 16 * mi + 8 * p, 16 * kd, lda, sl, wm, 'A(%d,k%s)p%d' % (mi, k, p))
            for ni in (gn if b_regs is None else ()):
                for p in range(2):
                    wm = 0x01 if (g == 0 and k == 0) else 0
                    if transB: row_load(bbuf[s][bslot[ni]] + wb * p, iB, idxB2, bB, B_BIND, ES[b_type], wb, N - 16 * ni, K - 16 * kd, 16 * ni + 8 * p, 16 * kd, ldb, sl, wm, 'B(k%s,%d)p%d' % (k, ni, p))
                    else: row_load(bbuf[s][bslot[ni]] + wb * p, iB, idxB2, bB, B_BIND, ES[b_type], wb, K - 16 * kd, N - 16 * ni, 16 * kd + 8 * p, 16 * ni, ldb, sl, wm, 'B(k%s,%d)p%d' % (k, ni, p))
            if split_fp32:
                # the loads of this step land in slot sl; one waiting add holds every ALU read below
                emit(998, faddenc.encode(swait, abuf[s][gm[0]], abuf[s][gm[0]], word=1 << (24 + sl), f1=0, f2=0, op=998),
                     'wait slot%d before the split reads the loads' % sl)
                for src, hi, lo, idxs in ((abuf, ahi, alo, gm), (bbuf, bhi, blo, gn)):
                    for x in idxs:
                        for i in range(8):
                            xr, h, l = src[s][x] + i, hi[x] + i, lo[x] + i
                            emit(3290, epienc.fmul(h, xr, sconst, keep_a=True), 'c = 8193 x')
                            emit(3290, epienc.fmul(stmp, xr, sneg, keep_a=True), '-x')
                            emit(998, faddenc.encode(l, h, stmp, word=0, f1=0, f2=16, op=998), 'd = c - x')
                            emit(3290, epienc.fmul(stmp, l, sneg), '-d')
                            emit(998, faddenc.encode(h, h, stmp, word=0, f1=16, f2=16, op=998), 'hi = c - d')
                            emit(3290, epienc.fmul(stmp, h, sneg, keep_a=True), '-hi')
                            emit(998, faddenc.encode(l, xr, stmp, word=0, f1=16, f2=16, op=998), 'lo = x - hi')
            first_unpack = True
            for tp, frag, unp, idxs in ((a_type, abuf, aunp, gm), (b_type, bbuf, bunp, gn)):
                if tp not in FP8:
                    continue
                for x in idxs:
                    for j in range(4):                            # halves L, H of the two loaded words -> four bf16 pairs
                        emit(17642, fp8enc.unpack(unp[x] + j, 'R%d%s' % (frag[s][x] + j // 2, 'LH'[j % 2]), FP8[tp],
                                                  wait_slot=sl if first_unpack else None),
                             'unpack %s frag %d half %d -> R%d' % (tp, x, j, unp[x] + j))
                        first_unpack = False
            passes = (('hi', 'hi'), ('hi', 'lo'), ('lo', 'hi')) if split_fp32 else ((None, None),)
            for (mi, ni) in group:
              for pa, pb in passes:
                mask = 1 << sl
                a_src = aunp[mi] if a_type in FP8 else (abuf[s][aslot[mi]] if a_regs is None else
                                                        (aconv[mi, k] if a_convert else a_regs[mi, k]))
                b_src = bunp[ni] if b_type in FP8 else (bbuf[s][bslot[ni]] if b_regs is None else
                                                        (bconv[k, ni] if b_convert else b_regs[k, ni]))
                first_issue = (kd == 0) if kloop_chunk else (kd % 2 == 0) if mx else (k == 0 and (pa, pb) == passes[0] and not seeded)
                last_pass = (pa, pb) == passes[-1]
                if split_fp32:                      # hi/lo tuples; a later pass may reread one, so none is released
                    a_src = (ahi if pa == 'hi' else alo)[mi]; b_src = (bhi if pb == 'hi' else blo)[ni]
                # a_keep (MM 25.178): a register-fed A that a LATER body reads again keeps its registers (no release)
                a_last = ((not split_fp32) and ni == max(n for mm, n in group if mm == mi)
                          and (a_regs is None or (g == last_group_of_row[mi] and not a_keep)))
                b_last = ((not split_fp32) and mi == max(mm for mm, n in group if n == ni)
                          and (b_regs is None or g == last_group_of_col[ni]))
                op, b = mmaenc.mma(tacc[mi, ni], a_src, b_src, None if first_issue else tacc[mi, ni],
                                   'bfloat' if a_type in FP8 else a_type, 'bfloat' if b_type in FP8 else b_type, transA, transB, wait=False, tag=0,
                                   more=(kd % 2 == 0) if mx else ((loop and not (MMA_MORE_PERIOD and loop_mma[0] % MMA_MORE_PERIOD == MMA_MORE_PERIOD - 1)) or (not loop and ((k < KT - 1) or not last_pass))), a_last=a_last, b_last=b_last,
                                   saturate=saturate and not first_issue)
                if seeded and k == 0:
                    mask |= 1 << 6                    # the first issue waits for the seed loads (slot 6)
                vals = {kk: v[1] for kk, v in enumerate(list(model.decode(b, 0))[0].values)}; vals[1] = (vals[1] & ~0x7f000000) | ((mask & 0x7f) << 24); b = mmaenc.encode(op, vals)
                if loop:
                    loop_mma[0] += 1
                if loop and kloop_unroll > 1:           # held back: every slice's loads issue before the first MMA waits
                    held.append((op, b, 'MMA(%d,%d,k%s) wait slot%d' % (mi, ni, k, sl)))
                else:
                    emit(op, b, 'MMA(%d,%d,k%s) wait slot%d' % (mi, ni, k, sl))
            if kloop_chunk and kd == 3:
                for tile in group:
                    for i in range(8):
                        emit(998, faddenc.encode(acc[tile] + i, acc[tile] + i, tacc[tile] + i,
                                                word=0, f1=0, f2=16, op=998),
                             'chunk fold D%s[%d] += fresh 64K partial' % (tile, i))
            if mx and kd % 2:
                blk = 0 if loop else kd // 2          # in the loop, the block is carried by kSA / kSB
                first_block = (not loop) and blk == 0
                for (mi, ni) in group:
                    if mx_codes:
                        # three aligned words, byte addressed (width 1): the lane's four SB column codes,
                        # and the words holding its SA row codes for rows 16 mi + rowg and 16 mi + 8 + rowg
                        loads = [(msb, mx[3], kSB if loop else col, mx[4] + blk * N + 16 * ni, 'SB')] + [
                            (msa + 4 * p, mx[1], kSA if loop else saq, mx[2] + blk * M_all + 16 * mi + 8 * p, 'SA')
                            for p in range(2)]
                    else:
                        loads = [(msb, mx[3], col, mx[4] + 4 * (blk * N + 16 * ni), 'SB')] + [
                            (msa + 4 * p, mx[1], rowg, mx[2] + 4 * (blk * M_all + 16 * mi + 8 * p), 'SA') for p in range(2)]
                    for dst, bind, ireg, disp, what in loads:
                        if disp > DISP_MAX:
                            raise ValueError('refused: scale table displacement %d exceeds the load field' % disp)
                        if mx_codes:
                            op, b = memenc.tload1w(dst, ireg, bind, disp=disp, width=1, slot=6)
                        else:
                            op, b = memenc.load(dst, ireg, bind, disp=disp, width=4, slot=6)
                        emit(op, b, '%s %s block %s tile (%d,%d) -> R%d slot6' % (what, 'codes' if mx_codes else 'fp32',
                                                                                 'L' if loop else blk, mi, ni, dst))
                    emit(998, faddenc.encode(mwait, msb, msb, word=1 << 30, f1=0, f2=0, op=998), 'wait slot6 before the scale reads')
                    if mx_codes:
                        def decode(dst, what):
                            # dst = (e << 23) + (((e + 1) >> 8) << 22), e = the code in mc
                            emit(10279, indexgen.addi(mt, mc, 1), '%s: t = e + 1' % what)
                            emit(17013, indexgen.shr32(mt, mt, 8), '%s: t >>= 8 (1 only for code 255)' % what)
                            emit(14391, indexgen.shl(mt, mt, 22), '%s: t <<= 22 (the quiet bit)' % what)
                            emit(14391, indexgen.shl(dst, mc, 23), '%s: e << 23' % what)
                            emit(10282, indexgen.add(dst, dst, mt), '%s: fp32 factor = (e << 23) + t' % what)
                        for p in range(2):
                            emit(17014, ledgerenc.encode(17014, {0: 'R%d' % mc, 1: 0, 3: 'R%d' % (msa + 4 * p), 4: 32,
                                                                 5: 'R%d' % ssh, 6: 32}),
                                 'SA p%d: word >> 8 (rowg & 3)' % p)
                            emit(423, indexgen.and32(mc, mc, 0xff), 'SA p%d: code byte' % p)
                            decode(msa + 4 * p, 'SA p%d' % p)
                        for j in (3, 2, 1, 0):                # j = 0 last: it overwrites the loaded word
                            if j:
                                emit(17013, indexgen.shr32(mc, msb, 8 * j), 'SB col %d: word >> %d' % (j, 8 * j))
                                emit(423, indexgen.and32(mc, mc, 0xff), 'SB col %d: code byte' % j)
                            else:
                                emit(423, indexgen.and32(mc, msb, 0xff), 'SB col 0: code byte')
                            decode(msb + j, 'SB col %d' % j)
                    for i in range(8):
                        t, d = tacc[mi, ni] + i, acc[mi, ni] + i
                        emit(3290, epienc.fmul(t, t, msa + 4 * (i // 4)), 'T(%d,%d)[%d] *= SA row block %s' % (mi, ni, i, 'L' if loop else blk))
                        if first_block:
                            emit(3290, epienc.fmul(d, t, msb + i % 4), 'D(%d,%d)[%d] = T * SB' % (mi, ni, i))
                        else:
                            emit(3290, epienc.fmul(t, t, msb + i % 4), 'T(%d,%d)[%d] *= SB col' % (mi, ni, i))
                            emit(998, faddenc.encode(d, d, t, word=0, f1=16, f2=16, op=998), 'D(%d,%d)[%d] += T' % (mi, ni, i))
            if closes:
                for held_op in held: emit(*held_op)
                held.clear()
                # advance, then the executed latch: cnt += 1 (source lifetime 0, as cc's executed loop),
                # op577, cmp.6 cnt < trips (keep), op582, op458 back to the first loop load, op577
                trips = K // 64 - 1 if kloop_chunk else K // 32 - 1 if mx else (KT - peel) // kloop_unroll
                g17asm, CMP_IMM, BRANCH_BACK, BRANCH_TAIL_BACK = _loop_encoders()
                emit(0, bytes(indexgen.addi(kA, kA, step)) + faddenc.encode(kB, kB, stepB, word=0, f1=0, f2=0, op=10282),
                     'advance kA += %d, kB += %d*ldb' % (step, step))
                if kbases:
                    hb = bytearray()
                    for (opd, rows), reg in sorted(kbases.items()):
                        hb += (bytes(indexgen.addi(reg, reg, step)) if opd == 'A' else
                               faddenc.encode(reg, reg, stepB, word=0, f1=0, f2=0, op=10282))
                    emit(0, bytes(hb), 'advance the %d loop row bases' % len(kbases))
                if mx:
                    emit(0, indexgen.add(kSA, kSA, stepSA) + indexgen.add(kSB, kSB, stepSB),
                         'advance kSA += %d, kSB += %d (the next block of codes)' % (M_all, N))
                emit(10279, ledgerenc.encode(10279, {0: 'R%d' % cnt, 1: 0, 3: 'R%d' % cnt, 4: 0, 2: 1}), 'cnt += 1')
                emit(577, EXEC_RESTORE, 'exec.restore (rebuild the loop mask)')
                emit(10369, g17asm.encode_cmp_src(cnt) + g17asm.encode_cmp_imm(trips, 'lt', CMP_IMM, keep=True),
                     'cmp cnt < %d' % trips)
                emit(582, EXEC_MASK, 'exec.mask (op582 gates op458)')
                at = len(out)
                emit(458, g17asm.encode_branch10(BRANCH_BACK + BRANCH_TAIL_BACK, loop_top - at),
                     'back edge %+d to the first loop load' % (loop_top - at))
                emit(577, EXEC_RESTORE, 'exec.restore after the loop')
                _check_counted_loop(bytes(out[loop_top:at]), cnt, trips)
                _check_loop_carried(bytes(out[loop_top:at]))
                if g17asm.decode_branch10(bytes(out[at:at + 10])) != loop_top - at:
                    raise RuntimeError('kloop back edge decodes to %+d, not to the first loop instruction %+d'
                                       % (g17asm.decode_branch10(bytes(out[at:at + 10])), loop_top - at))
        for (mi, ni) in group:
            if accumulate and c_regs is not None and c_inplace:
                for i in range(8):
                    b = faddenc.encode(c_regs[mi, ni] + i, acc[mi, ni] + i, c_regs[mi, ni] + i, word=0, f1=16, f2=0, op=998)
                    emit(998, b, 'fadd R%d = C(%d,%d)[%d] + the kept R%d (in place)' % (c_regs[mi, ni] + i, mi, ni, i, c_regs[mi, ni] + i))
            elif accumulate and c_regs is not None:
                for i in range(8):
                    b = faddenc.encode(acc[mi, ni] + i, acc[mi, ni] + i, c_regs[mi, ni] + i, word=0, f1=16, f2=16, op=998)
                    emit(998, b, 'fadd C(%d,%d)[%d] from the kept R%d' % (mi, ni, i, c_regs[mi, ni] + i))
            elif accumulate and not seeded:
                for p in range(2):
                    if 16 * mi + 16 > M or 16 * ni + 16 > N:
                        # the masked load moves two words and its width field offers 2 or 16: two loads on the doubled index (halves),
                        # each with a mask over four halves = two fp32 elements
                        ireg, disp = based(idxC2, bC, 16 * mi, 2 * ldc, 2, 4 * (8 * p * ldc + 16 * ni) + C_OFF, 'C2')
                        for half in range(2):
                            emit(0, mask_for_halves(p, M - 16 * mi, N - 16 * ni, half), 'mask C(%d,%d)p%d floats %d-%d' % (mi, ni, p, 2 * half, 2 * half + 1))
                            op, b = memenc.mload(ctmp + 4 * p + 2 * half, ireg, C_BIND, INV['R%dH' % mk[2]], disp=disp + 8 * half, width=2, slot=6)
                            emit(op, b, 'C(%d,%d)p%d words %d-%d -> tmp slot6' % (mi, ni, p, 2 * half, 2 * half + 1))
                    else:
                        ireg, disp = based(idxC, bC, 16 * mi, ldc, 4, 4 * (8 * p * ldc + 16 * ni) + C_OFF, 'C')
                        op, b = memenc.load(ctmp + 4 * p, ireg, C_BIND, disp=disp, width=4, slot=6); emit(op, b, 'C(%d,%d)p%d -> tmp slot6' % (mi, ni, p))
                for i in range(8):
                    aop = 10282 if integer else 998
                    b = faddenc.encode(acc[mi, ni] + i, acc[mi, ni] + i, ctmp + i, word=(1 << 30) if i == 0 else 0, f1=16, f2=16, op=aop); emit(aop, b, '%s C(%d,%d)[%d]' % ('iadd' if integer else 'fadd', mi, ni, i))
            for e in epilogue:
                if e[0] == 'bias':
                    op, b = memenc.load(ebias, col, e[1], disp=4 * 16 * ni + e[2], width=4, slot=6)
                    emit(op, b, 'bias(%d) -> R%d slot6' % (ni, ebias))
                    for i in range(8):
                        b = faddenc.encode(acc[mi, ni] + i, acc[mi, ni] + i, ebias + i % 4, word=(1 << 30) if i == 0 else 0,
                                           f1=16, f2=16 if i >= 4 else 0, op=998)
                        emit(998, b, 'bias add C(%d,%d)[%d]' % (mi, ni, i))
                elif e[0] == 'scale':
                    for i in range(8):
                        emit(3290, epienc.fmul(acc[mi, ni] + i, acc[mi, ni] + i, econst[e]), 'scale C(%d,%d)[%d]' % (mi, ni, i))
                elif e[0] == 'exp2':
                    for i in range(8):
                        emit(1272, epienc.exp2(acc[mi, ni] + i, acc[mi, ni] + i), 'exp2 C(%d,%d)[%d]' % (mi, ni, i))
                elif e[0] == 'relu':
                    for i in range(8):
                        emit(9700, epienc.fmax(acc[mi, ni] + i, acc[mi, ni] + i, econst[e]), 'relu C(%d,%d)[%d]' % (mi, ni, i))
                elif e[0] == 'gelu':
                    alpha, nil2, one, t = egelu
                    for i in range(8):
                        d = acc[mi, ni] + i
                        emit(3290, epienc.fmul(t, d, alpha, keep_a=True), 'gelu t = C(%d,%d)[%d] * 1.702' % (mi, ni, i))
                        emit(3290, epienc.fmul(t, t, nil2), 'gelu t *= -1/ln2')
                        emit(1272, epienc.exp2(t, t), 'gelu t = 2**t')
                        emit(998, epienc.fadd(t, t, one), 'gelu t += 1')
                        emit(3658, epienc.recip(t, t), 'gelu t = 1/t')
                        emit(3290, epienc.fmul(d, d, t, keep_b=False), 'gelu C(%d,%d)[%d] *= t' % (mi, ni, i))
            if rq:
                _, _, _, sc_spec, b_spec = rq
                if sc_spec[0] == 'tensor':
                    op, b = memenc.load(rq_sc, rq_zero, sc_spec[1], disp=sc_spec[2] - 4 * rq_tbase, width=4, slot=6)
                    emit(op, b, 'requant scale (per tensor) -> R%d slot6' % rq_sc)
                elif sc_spec[0] == 'channel':
                    op, b = memenc.load(rq_sc, rq_col if rq_col is not None else col, sc_spec[1], disp=4 * 16 * ni + sc_spec[2] - rq_vbase, width=4, slot=6)
                    emit(op, b, 'requant scale(%d) (per channel) -> R%d slot6' % (ni, rq_sc))
                if b_spec is not None:
                    op, b = memenc.load(rq_b, rq_col if rq_col is not None else col, b_spec[1], disp=4 * 16 * ni + b_spec[2] - rq_vbase, width=4, slot=6)
                    emit(op, b, 'requant bias(%d) -> R%d slot6' % (ni, rq_b))
                    for i in range(8):
                        b = faddenc.encode(acc[mi, ni] + i, acc[mi, ni] + i, rq_b + i % 4, word=(1 << 30) if i == 0 else 0,
                                           f1=16, f2=16 if i >= 4 else 0, op=10282)
                        emit(10282, b, 'requant bias iadd C(%d,%d)[%d]' % (mi, ni, i))
                for i in range(8):
                    emit(11179, epienc.i2f_inplace(acc[mi, ni] + i), 'requant float C(%d,%d)[%d]' % (mi, ni, i))
                if rq_wait is not None:
                    emit(998, faddenc.encode(rq_wait, rq_sc, rq_sc, word=1 << 30, f1=0, f2=0, op=998),
                         'wait slot6 before the scale reads')
                for i in range(8):
                    sreg = rq_sc + (i % 4 if sc_spec[0] == 'channel' else 0)
                    emit(3290, epienc.fmul(acc[mi, ni] + i, acc[mi, ni] + i, sreg), 'requant scale C(%d,%d)[%d]' % (mi, ni, i))
                for i in range(8):
                    emit(3770, epienc.rint_inplace(acc[mi, ni] + i), 'requant rint C(%d,%d)[%d]' % (mi, ni, i))
                for i in range(8):
                    emit(9700, epienc.fmax(acc[mi, ni] + i, acc[mi, ni] + i, rq_lo), 'requant lower rail C(%d,%d)[%d]' % (mi, ni, i))
                    emit(9700, epienc.fmin(acc[mi, ni] + i, acc[mi, ni] + i, rq_hi), 'requant upper rail C(%d,%d)[%d]' % (mi, ni, i))
                for i in range(8):
                    emit(9320, epienc.f2i_inplace(acc[mi, ni] + i), 'requant int C(%d,%d)[%d]' % (mi, ni, i))
                if rq_zpr is not None:
                    for i in range(8):
                        emit(10282, faddenc.encode(acc[mi, ni] + i, acc[mi, ni] + i, rq_zpr, word=0, f1=16, f2=0, op=10282),
                             'requant zero point C(%d,%d)[%d]' % (mi, ni, i))
                for p in range(2):
                    r = acc[mi, ni] + 4 * p                   # row m (p 0) or m+8 (p 1): columns col..col+3
                    dst = acc[mi, ni] + p                     # the packed word: acc+0 (row m), acc+1 (row m+8)
                    first = r
                    if rq_signed:
                        emit(423, indexgen.and32(rq_ta, r, 255), 'requant byte 0 of C(%d,%d)p%d' % (mi, ni, p)); first = rq_ta
                    for j in (1, 2):
                        if rq_signed:
                            emit(423, indexgen.and32(rq_tb, r + j, 255), 'requant byte %d mask' % j)
                            emit(14391, indexgen.shl(rq_tb, rq_tb, 8 * j), 'requant byte %d shift' % j)
                        else:
                            emit(14391, indexgen.shl(rq_tb, r + j, 8 * j), 'requant byte %d shift' % j)
                        emit(10282, indexgen.add(rq_ta, first, rq_tb), 'requant byte %d insert' % j); first = rq_ta
                    emit(14391, indexgen.shl(rq_tb, r + 3, 24), 'requant byte 3 shift')
                    emit(10282, indexgen.add(dst, rq_ta, rq_tb), 'requant packed word C(%d,%d)p%d -> R%d' % (mi, ni, p, dst))
                    if store:
                        words = (C_OFF + (16 * mi + 8 * p) * ldc + 16 * ni) // 4
                        ireg = idxQ
                        if words:
                            emit(11842, epienc.movimm(bC, words), 'requant base words %d' % words)
                            emit(10282, indexgen.add(bC, bC, idxQ), 'requant base R%d = idxQ + %d' % (bC, words))
                            ireg = bC
                        emit(17202, epienc.store_word(dst, ireg, C_BIND, src_last=not keep),
                             'store int8 C(%d,%d)p%d (4 bytes)%s' % (mi, ni, p, ' (kept live)' if keep else ''))
            if fp8_out:
                # Apple's order (fp8_store_witness): two packs into the halves of the first source
                # register, then the word store. Row m's four bytes land in acc+0, row m+8's in acc+4.
                for p in range(2):
                    r = acc[mi, ni] + 4 * p
                    emit(13618, epienc.fp8pack(r, 'L', r, r + 1, fp8_out), 'fp8 %s C(%d,%d)p%d cols 0-1 -> R%dL' % (fp8_out, mi, ni, p, r))
                    emit(13618, epienc.fp8pack(r, 'H', r + 2, r + 3, fp8_out), 'fp8 %s C(%d,%d)p%d cols 2-3 -> R%dH' % (fp8_out, mi, ni, p, r))
                    words = (C_OFF + (16 * mi + 8 * p) * ldc + 16 * ni) // 4
                    ireg = idxQ
                    if words:
                        emit(11842, epienc.movimm(bC, words), 'fp8 base words %d' % words)
                        emit(10282, indexgen.add(bC, bC, idxQ), 'fp8 base R%d = idxQ + %d' % (bC, words))
                        ireg = bC
                    emit(17202, epienc.store_word(r, ireg, C_BIND), 'store fp8 C(%d,%d)p%d (4 bytes)' % (mi, ni, p))
            if half_out:
                # four op1016 narrow a row's four fp32 slots into the halves of its first two registers
                # (slot i to half i, the A-fragment packing of the register chain), then two word stores
                for p in range(2):
                    r = acc[mi, ni] + 4 * p
                    for i in range(4):
                        emit(1016, epienc.cvt_f32_to_f16('R%d%s' % (r + i // 2, 'LH'[i % 2]), r + i),
                             'half C(%d,%d)p%d col %d -> R%d%s' % (mi, ni, p, i, r + i // 2, 'LH'[i % 2]))
                    for w in range(2):
                        words = (C_OFF // 2 + (16 * mi + 8 * p) * ldc + 16 * ni) // 2 + w
                        ireg = idxQ
                        if words:
                            emit(11842, epienc.movimm(bC, words), 'half base words %d' % words)
                            emit(10282, indexgen.add(bC, bC, idxQ), 'half base R%d = idxQ + %d' % (bC, words))
                            ireg = bC
                        emit(17202, epienc.store_word(r + w, ireg, C_BIND), 'store half C(%d,%d)p%d word %d (cols %d-%d)' % (mi, ni, p, w, 2 * w, 2 * w + 1))
            if half_kv:
                for p in range(2):
                    r = acc[mi, ni] + 4 * p
                    words = (C_OFF + 2 * ((16 * mi + 8 * p) * ldc + 16 * ni)) // 4
                    ireg = idxH
                    if words:
                        emit(11842, epienc.movimm(bC, words), 'half base words %d' % words)
                        emit(10282, indexgen.add(bC, bC, idxH), 'half base R%d = idxH + %d' % (bC, words))
                        ireg = bC
                    emit(11842, epienc.movimm(hword, words + 1), 'half second word %d' % (words + 1))
                    emit(10282, indexgen.add(hword, hword, idxH), 'half second word R%d = idxH + %d' % (hword, words + 1))
                    for i, half in enumerate(('R%dL' % r, 'R%dH' % r, 'R%dL' % (r + 1), 'R%dH' % (r + 1))):
                        emit(1016, epienc.cvt_f32_to_f16(half, r + i), 'half C(%d,%d)p%d col %d -> %s' % (mi, ni, p, i, half))
                    emit(17202, epienc.store_word(r, ireg, C_BIND), 'store half C(%d,%d)p%d cols 0-1' % (mi, ni, p))
                    emit(17202, epienc.store_word(r + 1, hword, C_BIND), 'store half C(%d,%d)p%d cols 2-3' % (mi, ni, p))
            for p in (range(2) if (store and not reduce and not fp8_out and not half_out and not half_kv and not rq) else ()):
                ireg, disp = based(idxC, bC, 16 * mi, ldc, 4, 4 * (8 * p * ldc + 16 * ni) + C_OFF, 'C')
                if 16 * mi + 16 > M or 16 * ni + 16 > N:
                    emit(0, mask_for(p, M - 16 * mi, N - 16 * ni), 'mask C(%d,%d)p%d' % (mi, ni, p))
                    op, b = memenc.mstore(acc[mi, ni] + 4 * p, ireg, C_BIND, INV['R%dH' % mk[2]], disp=disp, width=4, src_last=not keep)
                else:
                    op, b = memenc.tstore(acc[mi, ni] + 4 * p, ireg, C_BIND, disp=disp, width=4, src_last=not keep)
                emit(op, b, 'store C(%d,%d)p%d%s' % (mi, ni, p, ' (kept live)' if keep else ''))
    if reduce:
        comb = (lambda d, a_, b_, last: faddenc.encode(d, a_, b_, word=0, f1=16, f2=16 if last else 0, op=998)) \
            if reduce[1] == 'sum' else (lambda d, a_, b_, last: epienc.fmax(d, a_, b_, keep_a=False, keep_b=not last))
        for ni in range(NT):
            base = acc[0, ni]
            for mi in range(1, MT):                            # fold the row tiles, ascending
                for i in range(8):
                    emit(998 if reduce[1] == 'sum' else 9700, comb(base + i, base + i, acc[mi, ni] + i, True),
                         'reduce %s: D(0,%d)[%d] with D(%d,%d)[%d]' % (reduce[1], ni, i, mi, ni, i))
            for j in range(4):                                 # the r0+8 half into the r0 half
                emit(998 if reduce[1] == 'sum' else 9700, comb(base + j, base + j, base + 4 + j, True),
                     'reduce %s: row halves of column slot %d' % (reduce[1], j))
            for mask in (2, 4, 16):                            # the measured column butterfly
                for j in range(4):
                    emit(14169, epienc.shuffle_xor(rtmp, base + j, mask), 'col butterfly xor %d slot %d' % (mask, j))
                    emit(998 if reduce[1] == 'sum' else 9700, comb(base + j, base + j, rtmp, True),
                         'col butterfly %s slot %d' % (reduce[1], j))
            op, b = memenc.tstore(base, col, C_BIND, disp=4 * 16 * ni + C_OFF, width=4, src_last=True)
            emit(op, b, 'store column %s of tiles (*,%d) to C row 0' % (reduce[1], ni))
    if b_index is not None and b_index[1]:
        if b_index[1] <= 255:
            emit(10279, indexgen.addi(b_index[0], b_index[0], b_index[1]), 'advance B index R%d += %d' % (b_index[0], b_index[1]))
        else:
            emit(11842, epienc.movimm(t1, b_index[1]), 'B index step R%d = %d' % (t1, b_index[1]))
            emit(10282, indexgen.add(b_index[0], b_index[0], t1), 'advance B index R%d += R%d' % (b_index[0], t1))
    if end: emit(684, END, 'END')
    if body_room is not None:
        if len(out) > body_room: raise ValueError('body %d > room %d' % (len(out), body_room))
        while len(out) < body_room: out.extend(NOP)
    # THE BYTES ARE CHECKED, NOT THE REASONING: no MMA may read an operand register an earlier MMA
    # released (tensorlife scans the decoded body; Apple's 720 corpus programs with MMAs pass it).
    # A RuntimeError, not a 'refused:' ValueError, so no caller can absorb it as a shape refusal.
    from agxforge.g17 import tensorlife
    stale = tensorlife.released_reads(bytes(out))
    if stale:
        raise RuntimeError('tlower emitted a read of released registers: %s' % stale[:8])
    # A STORE MUST NOT RELEASE ITS VALUE REGISTER THROUGH ITS INDEX SLOT: the index slot's release
    # clears the register before the value is read, so every lane stores 0 (MM 25.117, op17229).
    aliased = tensorlife.aliased_store_releases(bytes(out))
    if aliased:
        raise RuntimeError('tlower emitted a store that releases its value through its index: %s' % aliased[:8])
    # A LOADED VALUE MUST BE WAITED FOR BY EVERY CONSUMER: the hardware does not interlock on a
    # pending source register (an unwaited imageblock store and an unwaited indirect-load index each
    # read stale data; MM sections 25.96 and 25.104). Piece B's loop-aware scan checks the bytes:
    # every body tlower emits passed it before it was added here (76 configurations across every
    # feature), and it fires when an MMA's wait bits are cleared (test_g17tensorhazards).
    from agxforge.g17 import tensorview
    unwaited = tensorview.hazards(tensorview.view(bytes(out)))
    if unwaited:
        raise RuntimeError('tlower emitted a read of an unwaited load: %s' % unwaited[:4])
    # Keep the historical reported register_count formula so existing authored images stay byte
    # identical.  The actual allocation is now bounded by R0..R125 above; this field is metadata
    # bookkeeping, not permission to address the squashed namespace.
    return bytes(out), dict(ops=ops, groups=[[list(x) for x in g] for g in groups], sets=sets,
                              registers=128 - len(P.free), allocatable_register_count=registerdomain.REGISTER_COUNT,
                              accumulator_groups=P.accumulator_groups,
                              scratch=dict(lane=lane, m=m, col=col, idx=[idxA, idxB, idxC], masks=mk),
                              a_type=a_type, b_type=b_type, simdgroups=sg, transA=transA, transB=transB,
                              prologue=hoisted_prologue,
                              acc=(dict(c_regs) if c_inplace else
                                   {tile: acc0[i] for i, tile in enumerate(tiles)} if keep else None),
                              grid=grid, grid_n=grid_n, split_k=split_k,
                              rows_per_threadgroup=M, k_per_threadgroup=K)
