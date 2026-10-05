"""Compiler-owned cooperative-tensor reduction rules.

This module is deliberately independent of Apple's ``reduce_rows`` and
``reduce_columns`` entry points.  Those routines are useful validation oracles,
but their per-lane chain direction is selected by the closed rtlib kernel that
contains them.  The compiler's contract is therefore the explicit sequence in
this file: a sequential FP32 chain within each lane, followed by the measured
shuffle-XOR butterfly.

The lane input is already in the measured ``pos_b`` order.  Keeping that
conversion at the caller is intentional: the 16x16 tile routing is an
architectural fact, while a new producer's register allocation is a separate
compiler concern.  No reduction operation below accepts a half, bfloat, integer,
non-64-K, or multi-SIMDgroup domain.
"""

from __future__ import annotations

import math
import struct

from agxforge.g17 import ir


FP32_MAX_FINITE = 0x7F7FFFFF
FP32_NEG_MAX = struct.unpack("<f", struct.pack("<I", 0xFF7FFFFF))[0]
FP32_POS_MAX = struct.unpack("<f", struct.pack("<I", FP32_MAX_FINITE))[0]

ROW_BUTTERFLY_MASKS = (1, 8)
COLUMN_BUTTERFLY_MASKS = (2, 4, 16)
SUPPORTED_K = 64
SUPPORTED_SIMDGROUPS = 1


class UnsupportedReduction(ValueError):
    """A reduction outside the measured compiler-owned domain."""


def require_domain(M, N, K=SUPPORTED_K, *, accumulator="f32", simdgroups=1):
    """Validate the only reduction domain this compiler currently promises."""
    if accumulator not in ("f32", "float", "FP32"):
        raise UnsupportedReduction(
            "tensor reduction requires an FP32 accumulator; %s is outside the measured domain"
            % accumulator
        )
    if K != SUPPORTED_K:
        raise UnsupportedReduction(
            "tensor reduction is measured at K=64 only; K=%s is outside the measured domain" % K
        )
    if simdgroups != SUPPORTED_SIMDGROUPS:
        raise UnsupportedReduction(
            "tensor reduction is measured for one SIMDgroup only; simdgroups=%s is unsupported"
            % simdgroups
        )
    if not isinstance(M, int) or not isinstance(N, int) or M <= 0 or N <= 0:
        raise UnsupportedReduction("tensor reduction requires positive integer M and N")
    if M % 16 or N % 16:
        raise UnsupportedReduction(
            "tensor reduction requires 16x16 destination tiles; got %sx%s" % (M, N)
        )
    return dict(M=M, N=N, K=K, accumulator="f32", simdgroups=1)


def tile_slot(row, column, N):
    """Return the first cT slot for a 16x16 ``pos_b`` destination tile."""
    if not isinstance(row, int) or not isinstance(column, int) or row < 0 or column < 0:
        raise ValueError("tile coordinates must be non-negative integers")
    if N <= 0 or N % 16:
        raise ValueError("N must be a positive multiple of 16")
    tiles_per_row = N // 16
    # For the 32x32 witness this is the reported 2*(row>>4)+(col>>4)
    # rule.  The measured wider one-SIMDgroup configurations use the row
    # stride implied by their number of 16-wide tiles.
    return 8 * ((row // 16) * tiles_per_row + (column // 16))


def pos_b(row, column):
    """Return ``(lane, slot)`` for one element of a 16x16 ``pos_b`` tile."""
    if not (0 <= row < 16 and 0 <= column < 16):
        raise ValueError("pos_b coordinates must lie in one 16x16 tile")
    lane = (16 * ((row >> 2) & 1) + 8 * (column >> 3)
            + 2 * (row & 3) + ((column >> 2) & 1))
    slot = 4 * (row >> 3) + (column & 3)
    return lane, slot


def pos_b_global(row, column, N):
    """Return ``(lane, slot)`` for a logical destination element."""
    if N <= 0 or N % 16:
        raise ValueError("N must be a positive multiple of 16")
    lane, slot = pos_b(row & 15, column & 15)
    return lane, slot + tile_slot(row, column, N)


def row_lane_values(matrix, row, *, descending=True):
    """Place one logical row into its 32 lane-local reduction inputs.

    The returned lists use the measured standalone-kernel order: destination
    tiles from right to left, and columns from right to left within each tile.
    ``descending=False`` is available for a compiler-owned ascending policy,
    but callers must choose it explicitly rather than inheriting a library
    kernel's direction.
    """
    M = len(matrix)
    N = len(matrix[0]) if M else 0
    if row < 0 or row >= M or M == 0 or N == 0 or N % 16:
        raise ValueError("row and matrix width must describe a non-empty 16-wide-tiled matrix")
    out = [[] for _ in range(32)]
    tile_columns = range(N - 16, -1, -16) if descending else range(0, N, 16)
    for base in tile_columns:
        columns = range(base + 15, base - 1, -1) if descending else range(base, base + 16)
        for column in columns:
            lane, _slot = pos_b_global(row, column, N)
            out[lane].append(matrix[row][column])
    return out


def tile_slots(M, N, K=SUPPORTED_K, *, accumulator="f32", simdgroups=1):
    """Describe the measured cT slot blocks for a one-SIMDgroup destination."""
    require_domain(M, N, K, accumulator=accumulator, simdgroups=simdgroups)
    return tuple(
        (row, column, tile_slot(row, column, N))
        for row in range(0, M, 16)
        for column in range(0, N, 16)
    )


def _round_fp32(value):
    """Round one Python float through the hardware's binary32 representation."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _fadd(a, b):
    return _round_fp32(_round_fp32(a) + _round_fp32(b))


def _fmax(a, b):
    # The measured library identity is -FLT_MAX and NaNs are ignored.  The
    # compiler-owned path keeps that selected behavior while making the order
    # explicit.  Ties retain the left operand for deterministic signed-zero
    # handling until a separate signed-zero census is available.
    a = _round_fp32(a)
    b = _round_fp32(b)
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return a if a >= b else b


def _local_reduce(values, operation):
    if operation == "max":
        # Starting at the measured identity also makes an all-NaN local
        # sequence produce the selected identity rather than propagating NaN.
        out = FP32_NEG_MAX
        for value in values:
            out = _fmax(out, value)
        return out
    if not values:
        return 0.0
    out = _round_fp32(values[0])
    for value in values[1:]:
        out = _fadd(out, value)
    return out


def butterfly(lane_values, masks, operation="sum"):
    """Apply the measured XOR butterfly to 32 already-local lane values."""
    if operation not in ("sum", "max"):
        raise UnsupportedReduction("reduction operation %r is not measured" % operation)
    if len(lane_values) != 32:
        raise UnsupportedReduction("tensor reduction requires exactly 32 SIMD lanes")
    if tuple(masks) not in (ROW_BUTTERFLY_MASKS, COLUMN_BUTTERFLY_MASKS):
        raise UnsupportedReduction("lane masks %r are outside the measured butterfly" % (masks,))
    values = [_round_fp32(v) for v in lane_values]
    fn = _fadd if operation == "sum" else _fmax
    for mask in masks:
        previous = list(values)
        for lane in range(32):
            values[lane] = fn(previous[lane], previous[lane ^ mask])
    return tuple(values)


def butterfly_array(values, masks, operation="sum"):
    """`butterfly` over the LAST axis (32 lanes) of a float32 array, every row at once: the same pairing, the sum as
    one fp32 add (butterfly rounds the double sum of two fp32 values to fp32, which is the correctly rounded fp32
    sum), the max with _fmax's NaN and tie rules. Bit-identical to `butterfly` wherever every input and result is
    finite; butterfly's struct rounding RAISES on an overflow this would return as inf, so a caller checks
    np.isfinite on the result and falls back to `butterfly` otherwise (test_g17simspeed)."""
    import numpy as np
    if operation not in ("sum", "max"):
        raise UnsupportedReduction("reduction operation %r is not measured" % operation)
    if tuple(masks) not in (ROW_BUTTERFLY_MASKS, COLUMN_BUTTERFLY_MASKS):
        raise UnsupportedReduction("lane masks %r are outside the measured butterfly" % (masks,))
    v = np.asarray(values, np.float32)
    if v.shape[-1] != 32:
        raise UnsupportedReduction("tensor reduction requires exactly 32 SIMD lanes")
    lanes = np.arange(32)
    with np.errstate(over="ignore", invalid="ignore"):
        for mask in masks:
            p = v[..., lanes ^ mask]
            if operation == "sum":
                v = (v + p).astype(np.float32)
            else:
                v = np.where(np.isnan(v), p, np.where(np.isnan(p), v, np.where(v >= p, v, p))).astype(np.float32)
    return v


def reduce_rows(lane_values, *, operation="sum", M=16, N=16, K=SUPPORTED_K,
                accumulator="f32", simdgroups=1):
    """Reduce one row-oriented tile from 32 lane-local sequences.

    ``lane_values[lane]`` is the sequence of values owned by that lane in
    ``pos_b`` order.  Each sequence is folded in ascending order, then the
    ``lane^1`` and ``lane^8`` stages are applied to every lane.
    """
    require_domain(M, N, K, accumulator=accumulator, simdgroups=simdgroups)
    local = [_local_reduce(v, operation) for v in lane_values]
    return butterfly(local, ROW_BUTTERFLY_MASKS, operation)


def reduce_columns(lane_values, *, operation="sum", M=16, N=16, K=SUPPORTED_K,
                   accumulator="f32", simdgroups=1):
    """Reduce one column-oriented tile using the measured 2,4,16 stages."""
    require_domain(M, N, K, accumulator=accumulator, simdgroups=simdgroups)
    local = [_local_reduce(v, operation) for v in lane_values]
    return butterfly(local, COLUMN_BUTTERFLY_MASKS, operation)


def row_sum(lane_values, **domain):
    return reduce_rows(lane_values, operation="sum", **domain)


def row_max(lane_values, **domain):
    return reduce_rows(lane_values, operation="max", **domain)


def emit_butterfly(builder, lane_value, masks, *, operation="sum"):
    """Emit the compiler-owned butterfly for one lane-local FP32 value.

    The caller is responsible for constructing the lane's explicit local
    chain before calling this helper.  Every cross-lane step is represented by
    the ordinary IR `simd_shuffle_xor` plus `fadd`/`fmax`; no Apple reduction
    intrinsic or opaque ordering is introduced.
    """
    if operation not in ("sum", "max"):
        raise UnsupportedReduction("reduction operation %r is not measured" % operation)
    if tuple(masks) not in (ROW_BUTTERFLY_MASKS, COLUMN_BUTTERFLY_MASKS):
        raise UnsupportedReduction("lane masks %r are outside the measured butterfly" % (masks,))
    if getattr(lane_value, "type", None) is not None and getattr(lane_value, "type", None).__str__() not in ("f32", "F32"):
        raise UnsupportedReduction("compiler-owned reduction emits only FP32 values")
    value = lane_value
    for mask in masks:
        peer = builder.simd_shuffle_xor(value, mask)
        value = builder.fadd(value, peer) if operation == "sum" else builder.fmax(value, peer)
    return value


def emit_row_sum(builder, lane_value):
    return emit_butterfly(builder, lane_value, ROW_BUTTERFLY_MASKS, operation="sum")


def emit_row_max(builder, lane_value):
    return emit_butterfly(builder, lane_value, ROW_BUTTERFLY_MASKS, operation="max")


def _f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _f32_const(builder, value, name):
    return builder.const(_f32_bits(value), type=ir.F32, name=name)


def _i32_const(builder, value, name):
    return builder.const(int(value) & 0xFFFFFFFF, type=ir.I32, name=name)


def _float_select(builder, predicate, when_true, when_false, name):
    """Select FP32 values with the measured integer condition-code instruction.

    ``Builder.csel`` historically returns an integer-typed value because its first users were
    integer clamps.  The instruction itself has a width field, so the reduction path carries the
    same operation with an explicitly FP32 destination rather than pretending the values are
    integers.
    """
    zero = _i32_const(builder, 0, name + "_zero")
    return builder._def("csel", [predicate, zero, when_true, when_false], ir.F32, name,
                        rel="gt")


class HeadBase:
    """THE HEAD GRID's scalar half (MM 25.135): threadgroup t's region of the buffer starts t << shift
    words further in. Every row-stage address adds it after the lane arithmetic, so a stage reads and
    writes its own head's tiles. The threadgroup index is read afresh at each use (no scalar is live
    across a tensor body)."""
    def __init__(self, shift_words):
        self.shift = int(shift_words)
        if not 0 <= self.shift < 31:
            raise ValueError("refused: the head base shift is 0..30 words")

    def emit(self, builder, prefix):
        tg = builder.builtin("threadgroup_position_in_grid", name=prefix + "_tg")
        return builder.shl(tg, _i32_const(builder, self.shift, prefix + "_shift"), name=prefix + "_base")


class SlotBase:
    """THE KV SPLIT's scalar half (MM 25.114.6): threadgroup t's partial slot starts base + t * stride
    words into the buffer (t = head * S + slice, so one stride serves both). The emit interface is
    HeadBase's: every row-stage address adds it. `stride_words` is decomposed into shift-adds; the
    threadgroup index is read afresh at each use (no scalar is live across a tensor body)."""
    def __init__(self, stride_words, base_words=0):
        self.stride, self.base = int(stride_words), int(base_words)
        if not 0 < self.stride < (1 << 24) or not 0 <= self.base < (1 << 30):
            raise ValueError("refused: the slot stride is 1..2^24 words and the base 0..2^30 words")

    def emit(self, builder, prefix):
        tg = builder.builtin("threadgroup_position_in_grid", name=prefix + "_tg")
        out = None
        for k in [k for k in range(24) if self.stride >> k & 1]:
            term = tg if k == 0 else builder.shl(tg, _i32_const(builder, k, "%s_shift%d" % (prefix, k)),
                                                 name="%s_term%d" % (prefix, k))
            out = term if out is None else builder.add(out, term, name="%s_sum%d" % (prefix, k))
        if self.base:
            out = builder.add(out, _i32_const(builder, self.base, prefix + "_slot0"), name=prefix + "_base")
        return out


def _hk(head):
    """The head keyword only when there is a head: every call without one is the call it always was
    (tests stand in for these helpers with the old signatures)."""
    return {} if head is None else {"head": head}


def _row0_lane_address(builder, buffer, *, row, stride, M=32, lane=None, base=0, head=None):
    """Build the measured 16-wide tile address for one row's four local values per lane.

    The first 16 rows are the supported reduction tile.  Inactive lanes read and restore a
    disjoint slice of rows 16..19 so the common-runtime output remains unchanged outside the
    reduced row; this is a storage arrangement, not a second reduction domain.
    """
    if not isinstance(row, int) or row < 0 or row >= 16:
        raise UnsupportedReduction("row reduction is measured for rows 0..15 of the first tile")
    if stride != 16 or M not in (16, 32):
        raise UnsupportedReduction(
            "the integrated row reduction is measured for a 16x16 or 32x16 FP32 tile with stride 16"
        )
    if lane is None:
        lane = builder.builtin("thread_index_in_simdgroup", name="reduction_lane")
    lane_hi = builder.shr(lane, _i32_const(builder, 4, "row_lane_shift4"), name="row_lane_hi")
    lane_hi = getattr(builder, "and")(lane_hi, _i32_const(builder, 1, "row_lane_hi_mask"),
                                        name="row_lane_hi_bit")
    row_low = builder.shr(lane, _i32_const(builder, 1, "row_lane_shift1"), name="row_lane_shifted")
    row_low = getattr(builder, "and")(row_low, _i32_const(builder, 3, "row_lane_low_mask"),
                                        name="row_lane_low_bits")
    lane_row = builder.add(builder.shl(lane_hi, _i32_const(builder, 2, "row_lane_hi_scale"),
                                       name="row_lane_hi_scaled"), row_low, name="row_lane_row")
    active = builder.icmp(lane_row, _i32_const(builder, row & 7, "row_target"), rel="eq",
                          name="row_lane_active")

    col_hi = builder.shr(lane, _i32_const(builder, 3, "row_col_shift3"), name="row_col_shifted3")
    col_hi = getattr(builder, "and")(col_hi, _i32_const(builder, 1, "row_col_hi_mask"),
                                      name="row_col_hi_bit")
    # Inverting pos_b is asymmetric: the 4-column group is lane bit 0 itself, while the
    # 8-column half is lane bit 3.  Using lane>>2 here would alias odd/even lanes and silently
    # leave half the row unnormalized (the compiler test exercises those four distinct addresses).
    col_lo = getattr(builder, "and")(lane, _i32_const(builder, 1, "row_col_lo_mask"),
                                      name="row_col_lo_bit")
    col = builder.add(builder.shl(col_hi, _i32_const(builder, 3, "row_col_hi_scale"),
                                  name="row_col_hi_scaled"),
                      builder.shl(col_lo, _i32_const(builder, 2, "row_col_lo_scale"),
                                  name="row_col_lo_scaled"), name="row_col_start")
    # `base` (elements) places the 32x16 tile inside a larger buffer (the streaming-softmax regions);
    # at 0 every constant, and so every emitted byte, is what it was
    active_index = builder.add(_i32_const(builder, base + row * stride, "row_active_base"), col,
                               name="row_active_index")
    # Inactive lanes must not race the four active lanes' source/output addresses.  The fallback
    # occupies four words per lane in rows 16..19, which are outside the measured first tile.
    if M == 16:
        # There are no spare rows in the 16x16 contract. Inactive lanes read the first four words,
        # then select those values away and never store them, keeping all addresses in bounds.
        fallback = _i32_const(builder, base, "row_fallback_index")
    else:
        fallback = builder.add(_i32_const(builder, base + 16 * stride, "row_fallback_base"),
                               builder.shl(lane, _i32_const(builder, 2, "row_fallback_scale"),
                                            name="row_fallback_offset"), name="row_fallback_index")
    index = builder.csel(active, _i32_const(builder, 0, "row_index_cond"), active_index, fallback,
                         rel="gt", name="row_load_index")
    if head is not None:
        # the head grid (MM 25.135): the whole tile, fallback rows included, moves with the head
        index = builder.add(index, head.emit(builder, "row_head"), name="row_head_index")
    return lane, active, index


def _local_values(builder, buffer, *, row, stride, operation, M=32, lane=None, base=0, head=None):
    lane, active, index, values = _load_values(builder, buffer, row=row, stride=stride, M=M,
                                               lane=lane, base=base, **_hk(head))
    local = values[0]
    for offset, value in enumerate(values[1:], 1):
        local = (builder.fadd(local, value, name="row_local_sum%d" % offset)
                 if operation == "sum" else builder.fmax(local, value, name="row_local_max%d" % offset))
    identity = _f32_const(builder, 0.0 if operation == "sum" else FP32_NEG_MAX,
                           "row_identity")
    local = _float_select(builder, active, local, identity, "row_active_local")
    return lane, active, index, values, local


def _load_values(builder, buffer, *, row, stride, M=32, lane=None, base=0, head=None):
    lane, active, index = _row0_lane_address(builder, buffer, row=row, stride=stride, M=M,
                                              lane=lane, base=base, **_hk(head))
    values = []
    for offset in range(4):
        idx = index if offset == 0 else builder.add(index, _i32_const(builder, offset,
                                                                       "row_value_offset%d" % offset),
                                                    name="row_value_index%d" % offset)
        values.append(builder.load(buffer, idx, type=ir.F32, name="row_value%d" % offset))
    return lane, active, index, values


def emit_buffer_row_reduction(builder, buffer, *, row=0, operation="sum", M=32, N=16,
                              K=SUPPORTED_K, lane=None):
    """Emit one measured row reduction from a 32x16 FP32 score tile.

    The operation is intentionally a narrow integration surface.  It reads four ``pos_b``-ordered
    columns per lane, folds them locally, and applies the two measured row butterfly masks.  The
    returned value is replicated on the four lanes belonging to the selected row; callers can use
    ``simd_broadcast_first`` when a scalar epilogue needs one copy.
    """
    require_domain(M, N, K, accumulator="f32", simdgroups=1)
    if N != 16 or M not in (16, 32):
        raise UnsupportedReduction("buffer row reduction is measured only for M=16 or 32, N=16")
    if operation not in ("sum", "max"):
        raise UnsupportedReduction("row reduction operation %r is not measured" % operation)
    _lane, active, _index, _values, local = _local_values(builder, buffer, row=row, stride=N,
                                                          operation=operation, M=M, lane=lane)
    return emit_butterfly(builder, local, ROW_BUTTERFLY_MASKS, operation=operation), active


def emit_row_softmax(builder, buffer, *, row=0, M=32, N=16, K=SUPPORTED_K):
    """Emit the measured-domain row-max/exp2/row-sum/normalize workload.

    This is the first application primitive, deliberately restricted to one 32x16 FP32 tile and
    one SIMDgroup.  It uses the compiler-owned reduction order for both reductions; Apple's
    ``reduce_rows`` is not called and cannot influence the emitted bytes.
    """
    require_domain(M, N, K, accumulator="f32", simdgroups=1)
    if N != 16 or M != 32:
        raise UnsupportedReduction("row softmax is measured only for M=32, N=16")
    _lane, _active, _index, _values, local_max = _local_values(builder, buffer, row=row, stride=N,
                                                                operation="max", M=M)
    row_max_value = emit_butterfly(builder, local_max, ROW_BUTTERFLY_MASKS, operation="max")
    # THE LAYERNORM FIX, WHICH NEVER REACHED THE SOFTMAX. The butterfly (lane bits 0 and 3) already
    # replicates each row's result across the four lanes that own it. Broadcast-first copies LANE 0,
    # which owns rows 0 and 8 only; for any other row lane 0 holds the inactive identity (-FP32_MAX for
    # the max, 0 for the sum), so the row normalised against it: exp2 overflowed, the sum was inf,
    # recip 0, and inf * 0 made every element NaN (Set A fusion, one-dispatch attention row 1 on
    # hardware, 2026-09-23). emit_row_layernorm documents and avoids exactly this; row 0 keeps its
    # measured broadcast bytes, every other row uses the active lanes' own result.
    if row == 0:
        row_max_value = builder.simd_broadcast_first(row_max_value, type=ir.F32, name="row_max_broadcast")
    neg_max = builder.fmul(row_max_value, _f32_const(builder, -1.0, "row_neg_one"),
                           name="row_neg_max")
    # Reload after the max butterfly.  The tensor body already reserves a large physical register
    # set, so retaining the first four score loads through both reductions would turn a correct
    # program into an allocator failure.  The reload is also the ordinary memory bridge the
    # runtime uses between tensor-produced scores and scalar consumers.
    _lane, active, index, values = _load_values(builder, buffer, row=row, stride=N, M=M)
    exponentials = []
    for offset, value in enumerate(values):
        shifted = builder.fadd(value, neg_max, name="row_shift%d" % offset)
        exponentials.append(builder.exp2(shifted, name="row_exp2_%d" % offset))
    local_sum = exponentials[0]
    for offset, value in enumerate(exponentials[1:], 1):
        local_sum = builder.fadd(local_sum, value, name="row_local_exp_sum%d" % offset)
    local_sum = _float_select(builder, active, local_sum, _f32_const(builder, 0.0, "row_sum_zero"),
                              "row_active_exp_sum")
    row_sum_value = emit_butterfly(builder, local_sum, ROW_BUTTERFLY_MASKS, operation="sum")
    if row == 0:
        row_sum_value = builder.simd_broadcast_first(row_sum_value, type=ir.F32, name="row_sum_broadcast")
    inv_sum = builder.recip(row_sum_value, name="row_inv_sum")
    # Recompute the four exponentials after the sum broadcast so no lane keeps both the sum
    # inputs and the output values live across the allocator's tensor register reservation.
    _lane, active_out, index_out, output_values = _load_values(builder, buffer, row=row, stride=N, M=M)
    normalized = []
    for offset, value in enumerate(output_values):
        shifted = builder.fadd(value, neg_max, name="row_output_shift%d" % offset)
        normalized.append(builder.fmul(builder.exp2(shifted, name="row_output_exp2_%d" % offset),
                                       inv_sum, name="row_normalized%d" % offset))
    # Preserve inactive lanes' fallback rows and write the selected row back in place.  Every lane
    # has a unique four-word address, so this is race-free without an extra barrier.
    for offset, (value, original) in enumerate(zip(normalized, output_values)):
        output = _float_select(builder, active_out, value, original, "row_output%d" % offset)
        idx = index_out if offset == 0 else builder.add(index_out, _i32_const(builder, offset,
                                                                       "row_output_offset%d" % offset),
                                                   name="row_output_index%d" % offset)
        builder.store_at(buffer, idx, output)
    return row_max_value, row_sum_value, active


def _store_row(builder, buffer, active, index, values, originals, prefix):
    """Write four values of one row back through the measured row addressing: active lanes store
    `values`, inactive lanes restore the fallback words they read (race-free, as the softmax)."""
    for offset, (value, original) in enumerate(zip(values, originals)):
        output = _float_select(builder, active, value, original, "%s_out%d" % (prefix, offset))
        idx = index if offset == 0 else builder.add(index, _i32_const(builder, offset,
                                                                       "%s_off%d" % (prefix, offset)),
                                                    name="%s_idx%d" % (prefix, offset))
        builder.store_at(buffer, idx, output)


def _store_stat(builder, buffer, *, row, base, value, prefix, head=None):
    """Replicate one per-row scalar across its row of a 32x16 stats tile (all 16 words = value), so a
    later stage reads it back through the same measured addressing: a memory bridge for a value that
    must survive a tensor body (no scalar is live across a body)."""
    _lane, active, index, originals = _load_values(builder, buffer, row=row, stride=16, M=32, base=base, **_hk(head))
    _store_row(builder, buffer, active, index, [value] * 4, originals, prefix)


def _row_stat(builder, buffer, *, row, base, head=None):
    """The per-row scalar a _store_stat wrote (valid on the row's four active lanes). ONE load: the
    first version loaded the row's four words and used one, and the three dead loads' registers,
    reused while the loads were in flight, broke rows 2 and 9 on hardware (cc now refuses an
    unread load whose destination is rewritten)."""
    _lane, _active, index = _row0_lane_address(builder, buffer, row=row, stride=16, M=32, base=base, **_hk(head))
    return builder.load(buffer, index, type=ir.F32, name="row_stat")


def _row_sum_of(builder, active, values, prefix):
    local = values[0]
    for offset, value in enumerate(values[1:], 1):
        local = builder.fadd(local, value, name="%s_local_sum%d" % (prefix, offset))
    local = _float_select(builder, active, local, _f32_const(builder, 0.0, prefix + "_sum_zero"),
                          prefix + "_active_sum")
    return emit_butterfly(builder, local, ROW_BUTTERFLY_MASKS, operation="sum")


# THE CAUSAL MASK (production row P7, machine model 25.129; recon section 155's rule). A masked score
# is replaced by this finite value, the one recon 155 used, not -inf: fmax keeps it out of the row max,
# exp2(MASK - m) is exactly 0 for any finite m, and no infinity reaches the max rule of section 126.
CAUSAL_MASK_VALUE = struct.unpack("<f", struct.pack("<f", -1.0e30))[0]


def causal_threshold(q0, row, key0):
    """The mask threshold of one score row inside one 16-key tile: query position q0 + row sees keys
    at or before it, so tile column c (key key0 + c) is masked when c > q0 + row - key0."""
    return int(q0) + int(row) - int(key0)


def _lane_column(builder, lane):
    """The first of the four columns a lane holds in the row addressing: 8*lane[3] + 4*lane[0]. It is
    the column half of the inverse of pos_b (section 25.44; recon 155's formula), the same expression
    _row0_lane_address builds for the address."""
    hi = getattr(builder, "and")(builder.shr(lane, _i32_const(builder, 3, "mask_col_shift3"),
                                             name="mask_col_shifted3"),
                                 _i32_const(builder, 1, "mask_col_hi_mask"), name="mask_col_hi_bit")
    lo = getattr(builder, "and")(lane, _i32_const(builder, 1, "mask_col_lo_mask"), name="mask_col_lo_bit")
    return builder.add(builder.shl(hi, _i32_const(builder, 3, "mask_col_hi_scale"), name="mask_col_hi_scaled"),
                       builder.shl(lo, _i32_const(builder, 2, "mask_col_lo_scale"), name="mask_col_lo_scaled"),
                       name="mask_col_start")


class RuntimeThreshold:
    """A causal threshold known only at run time (P9's stateful step, MM 25.131): query row `row` at
    position length + row, in the tile of keys key0 .. key0 + 15. `length` is an IR I32 value (the
    loaded uniform), `bias` the constant that keeps both sides of the compare non-negative."""
    value = CAUSAL_MASK_VALUE      # what a masked score becomes
    zero_p = False                 # KeyMask: a masked P is selected to +0 as well
    def __init__(self, length, row, key0, bias):
        self.length, self.row, self.key0, self.bias = length, int(row), int(key0), int(bias)
        if self.bias + self.row - self.key0 - 3 < 0:
            raise ValueError("refused: the runtime mask bias %d does not cover key0 %d" % (self.bias, self.key0))


class KeyMask(RuntimeThreshold):
    """THE KV SPLIT's per-trip key-validity mask (MM 25.114.6): the causal rule with this trip's first
    key `key0` known only at run time (an IR I32 value, loaded each trip: memory-carried in the slot) and
    q0 compile-time, so tile column c is masked when key0 + c > q0 + row. The compare is
    col + key0 + bias > q0 + row - i + bias, both sides non-negative. A masked score becomes -FLT_MAX (the initial m, not -1e30) and its P is
    SELECTED to +0, so a slice whose every key is masked keeps m = -FLT_MAX, l = 0 and O = +0 exactly
    (with -1e30 its first all-masked trip would give m = -1e30 and P = exp2(0) = 1). On a row with any
    unmasked key the values are those of the -1e30 mask: the max ignores both, and exp2 of either
    minus a finite m is +0."""
    value = FP32_NEG_MAX
    zero_p = True

    def __init__(self, key0, q0, row, bias):
        # q0 is an int, or an IR I32 value (the runtime length, MM 25.138.1: loaded each trip, non-negative)
        self.length, self.row, self.bias = key0, int(row), int(bias)
        self.q0 = int(q0) if isinstance(q0, int) or type(q0).__name__.startswith(("int", "uint")) else q0
        if self.bias < 3 or (isinstance(self.q0, int) and self.q0 < 0):
            raise ValueError("refused: the key mask needs q0 >= 0 and a bias >= 3")


def _apply_runtime_mask(builder, lane, values, th, prefix):
    """Column c = col + i is masked when c > length + row - key0, as the compile-time rule, but with
    nothing decidable at compile time: one csel per element, comparing col + bias with
    length + (bias + row - key0 - i). Both sides are non-negative, so signedness cannot matter."""
    neg = _f32_const(builder, th.value, prefix + "_value")
    colb = builder.add(_lane_column(builder, lane), _i32_const(builder, th.bias, prefix + "_bias"),
                       name=prefix + "_col_biased")
    if isinstance(th, KeyMask):
        colb = builder.add(colb, th.length, name=prefix + "_key_biased")
    out = []
    bounds = []
    for i, value in enumerate(values):
        if isinstance(th, KeyMask) and not isinstance(th.q0, int):
            bound = builder.add(th.q0, _i32_const(builder, th.row - i + th.bias, "%s_kq%d" % (prefix, i)),
                                name="%s_kb%d" % (prefix, i))
        elif isinstance(th, KeyMask):
            bound = _i32_const(builder, th.q0 + th.row - i + th.bias, "%s_kb%d" % (prefix, i))
        else:
            bound = builder.add(th.length, _i32_const(builder, th.bias + th.row - th.key0 - i, "%s_rt%d" % (prefix, i)),
                                name="%s_bound%d" % (prefix, i))
        bounds.append(bound)
        out.append(builder._def("csel", [colb, bound, neg, value], ir.F32, "%s_select%d" % (prefix, i), rel="gt"))
    th.preds = (colb, bounds)          # KeyMask selects P by the same compares
    return out


def _zero_masked(builder, th, values, prefix):
    """KeyMask: P = +0 wherever the score was masked (the same compare as the mask's select)."""
    if not getattr(th, "zero_p", False):
        return values
    colb, bounds = th.preds
    zero = _f32_const(builder, 0.0, prefix + "_zero")
    return [builder._def("csel", [colb, b, zero, v], ir.F32, "%s_pzero%d" % (prefix, i), rel="gt")
            for i, (v, b) in enumerate(zip(values, bounds))]


def apply_causal_mask(builder, lane, values, threshold, prefix="mask"):
    """Replace a lane's four row values (columns col .. col+3) by CAUSAL_MASK_VALUE where the column
    exceeds `threshold` (causal_threshold). Everything that can be decided at compile time is: an
    element whose column bound is below 0 is masked on every lane (a constant), one whose bound is
    12 or more is never masked (the value itself, no instruction), and only the rest is one
    csel (col > threshold - i) per element. None or a threshold of 15 or more emits nothing."""
    if isinstance(threshold, RuntimeThreshold):
        return _apply_runtime_mask(builder, lane, values, threshold, prefix)
    if threshold is None or threshold >= 15:
        return list(values)
    neg = _f32_const(builder, CAUSAL_MASK_VALUE, prefix + "_value")
    col = None
    out = []
    for i, value in enumerate(values):
        bound = threshold - i                 # masked when col > bound; col is 0, 4, 8 or 12
        if bound < 0:
            out.append(neg)
        elif bound >= 12:
            out.append(value)
        else:
            if col is None:
                col = _lane_column(builder, lane)
            out.append(builder._def("csel", [col, _i32_const(builder, bound, "%s_bound%d" % (prefix, i)), neg, value],
                                    ir.F32, "%s_select%d" % (prefix, i), rel="gt"))
    return out


def _masked_local_max(builder, lane, active, values, mask, prefix):
    values = apply_causal_mask(builder, lane, values, mask, prefix)
    local = values[0]
    for offset, value in enumerate(values[1:], 1):
        local = builder.fmax(local, value, name="%s_local_max%d" % (prefix, offset))
    local = _float_select(builder, active, local, _f32_const(builder, FP32_NEG_MAX, prefix + "_identity"),
                          prefix + "_active_local")
    return values, local


def emit_stream_first_block(builder, buffer, *, row, s_base, m_base, l_base, mask=None, head=None):
    """ONLINE SOFTMAX, first key block (goal item 6): on one row of the stored score tile S1,
    m1 = rowmax S1, P1 = exp2(S1 - m1) written in place (unnormalized), l1 = rowsum P1; m1 and l1
    are written to their stats tiles so they survive the tensor bodies that follow (a memory bridge).
    Base 2 throughout, as the released row softmax. Rows 0..15 of a 32x16 fp32 tile only.
    `mask` (P7): the causal threshold of this row in this tile (causal_threshold), applied to the
    loaded scores before the row max; None (the default) emits exactly the unmasked bytes.
    `head` (MM 25.135): a HeadBase, every address moved to this threadgroup's head; None emits nothing."""
    if mask is None:
        _lane, active, index, values, local_max = _local_values(builder, buffer, row=row, stride=16,
                                                                operation="max", M=32, base=s_base, **_hk(head))
    else:
        lane, active, index, raw = _load_values(builder, buffer, row=row, stride=16, M=32, base=s_base, **_hk(head))
        values, local_max = _masked_local_max(builder, lane, active, raw, mask, "sf_mask")
    m1 = emit_butterfly(builder, local_max, ROW_BUTTERFLY_MASKS, operation="max")
    neg_m1 = builder.fmul(m1, _f32_const(builder, -1.0, "sf_neg_one"), name="sf_neg_m1")
    p1 = [builder.exp2(builder.fadd(v, neg_m1, name="sf_shift%d" % i), name="sf_p1_%d" % i)
          for i, v in enumerate(values)]
    _store_row(builder, buffer, active, index, p1, values if mask is None else raw, "sf_p1")
    l1 = _row_sum_of(builder, active, p1, "sf_l1")
    _store_stat(builder, buffer, row=row, base=m_base, value=m1, prefix="sf_m1", **_hk(head))
    _store_stat(builder, buffer, row=row, base=l_base, value=l1, prefix="sf_l1s", **_hk(head))


def emit_stream_second_block(builder, buffer, *, row, s_base, o_base, m_base, l_base, rescale=True, mask=None,
                             head=None, o_bases=None):
    """ONLINE SOFTMAX, second key block: with m1, l1 from the stats tiles and S2 the stored second
    score tile, m2 = max(m1, rowmax S2), alpha = exp2(m1 - m2), P2 = exp2(S2 - m2) in place,
    l = alpha * l1 + rowsum P2, and the stored partial output row O = alpha * O. m2 and l go back to
    the stats tiles. `rescale=False` exists only for tests (the program the no-alpha control claims).
    Every later key block of an n-block stream is this stage again (P7). `mask` as in the first block.
    `head` as in the first block. `o_bases` (MM 25.135): the O tiles of a value wider than 16, each a
    32 x 16 tile, all rescaled by the one alpha (None: [o_base], the bytes as before)."""
    m1 = _row_stat(builder, buffer, row=row, base=m_base, **_hk(head))
    l1 = _row_stat(builder, buffer, row=row, base=l_base, **_hk(head))
    if mask is None:
        _lane, active, index, values, local_max = _local_values(builder, buffer, row=row, stride=16,
                                                                operation="max", M=32, base=s_base, **_hk(head))
        raw = values
    else:
        lane, active, index, raw = _load_values(builder, buffer, row=row, stride=16, M=32, base=s_base, **_hk(head))
        values, local_max = _masked_local_max(builder, lane, active, raw, mask, "ss_mask")
    ms2 = emit_butterfly(builder, local_max, ROW_BUTTERFLY_MASKS, operation="max")
    m2 = builder.fmax(m1, ms2, name="ss_m2")
    neg_m2 = builder.fmul(m2, _f32_const(builder, -1.0, "ss_neg_one"), name="ss_neg_m2")
    alpha = builder.exp2(builder.fadd(m1, neg_m2, name="ss_m1_minus_m2"), name="ss_alpha")
    p2 = [builder.exp2(builder.fadd(v, neg_m2, name="ss_shift%d" % i), name="ss_p2_%d" % i)
          for i, v in enumerate(values)]
    if mask is not None:
        p2 = _zero_masked(builder, mask, p2, "ss_mask")
    _store_row(builder, buffer, active, index, p2, raw, "ss_p2")
    s2 = _row_sum_of(builder, active, p2, "ss_l2")
    l = builder.fadd(builder.fmul(alpha, l1, name="ss_alpha_l1"), s2, name="ss_l")
    if rescale:
        for ob in (o_bases if o_bases is not None else [o_base]):
            _lane, o_active, o_index, o_values = _load_values(builder, buffer, row=row, stride=16, M=32,
                                                              base=ob, **_hk(head))
            scaled = [builder.fmul(o, alpha, name="ss_o_scaled%d" % i) for i, o in enumerate(o_values)]
            _store_row(builder, buffer, o_active, o_index, scaled, o_values, "ss_o")
    _store_stat(builder, buffer, row=row, base=m_base, value=m2, prefix="ss_m2s", **_hk(head))
    _store_stat(builder, buffer, row=row, base=l_base, value=l, prefix="ss_ls", **_hk(head))


def emit_oneshot_softmax(builder, buffer, *, row, s1_base, s2_base, l_base):
    """ONE-SHOT SOFTMAX over a 32-key score row held as two 16-column tiles S1 | S2 (goal item 6, the
    comparison arm for online softmax): m = max(rowmax S1, rowmax S2); P1 = exp2(S1 - m) and
    P2 = exp2(S2 - m) written in place (unnormalized); l = rowsum P1 + rowsum P2 to the stats tile,
    for emit_stream_normalize. Every step is the measured row addressing and butterfly of the stream
    stages, straight-line; rows 0..15 of each 32x16 fp32 tile."""
    _lane, active, index, v1, max1 = _local_values(builder, buffer, row=row, stride=16,
                                                   operation="max", M=32, base=s1_base)
    m1 = emit_butterfly(builder, max1, ROW_BUTTERFLY_MASKS, operation="max")
    _lane2, active2, index2, v2, max2 = _local_values(builder, buffer, row=row, stride=16,
                                                      operation="max", M=32, base=s2_base)
    m2 = emit_butterfly(builder, max2, ROW_BUTTERFLY_MASKS, operation="max")
    m = builder.fmax(m1, m2, name="os_m")
    neg_m = builder.fmul(m, _f32_const(builder, -1.0, "os_neg_one"), name="os_neg_m")
    p1 = [builder.exp2(builder.fadd(v, neg_m, name="os_shift1_%d" % i), name="os_p1_%d" % i)
          for i, v in enumerate(v1)]
    _store_row(builder, buffer, active, index, p1, v1, "os_p1")
    p2 = [builder.exp2(builder.fadd(v, neg_m, name="os_shift2_%d" % i), name="os_p2_%d" % i)
          for i, v in enumerate(v2)]
    _store_row(builder, buffer, active2, index2, p2, v2, "os_p2")
    l = builder.fadd(_row_sum_of(builder, active, p1, "os_l1"), _row_sum_of(builder, active2, p2, "os_l2"),
                     name="os_l")
    _store_stat(builder, buffer, row=row, base=l_base, value=l, prefix="os_ls")


def emit_kv_merge(builder, buffer, *, row, o_base, m_base, l_base, o1_base, m1_base, l1_base):
    """THE KV-SPLIT MERGE (P9, MM 25.131; recon section 154's rule) on one row, two partial states:
    M = max(m0, m1), f_s = exp2(m_s - M), l = f0 l0 + f1 l1 and O = f0 O0 + f1 O1, each product and sum
    rounded separately (fmul, fmul, fadd). The merged O, M and l overwrite the first state's rows, so
    emit_stream_normalize finishes it unchanged."""
    m0 = _row_stat(builder, buffer, row=row, base=m_base)
    m1 = _row_stat(builder, buffer, row=row, base=m1_base)
    big = builder.fmax(m0, m1, name="kvm_max")
    neg = builder.fmul(big, _f32_const(builder, -1.0, "kvm_neg_one"), name="kvm_neg")
    f0 = builder.exp2(builder.fadd(m0, neg, name="kvm_d0"), name="kvm_f0")
    f1 = builder.exp2(builder.fadd(m1, neg, name="kvm_d1"), name="kvm_f1")
    l0 = _row_stat(builder, buffer, row=row, base=l_base)
    l1 = _row_stat(builder, buffer, row=row, base=l1_base)
    l = builder.fadd(builder.fmul(f0, l0, name="kvm_f0l0"), builder.fmul(f1, l1, name="kvm_f1l1"), name="kvm_l")
    _lane, active, index, o0 = _load_values(builder, buffer, row=row, stride=16, M=32, base=o_base)
    _lane1, _active1, _index1, o1 = _load_values(builder, buffer, row=row, stride=16, M=32, base=o1_base)
    merged = [builder.fadd(builder.fmul(f0, a, name="kvm_f0o%d" % i), builder.fmul(f1, b, name="kvm_f1o%d" % i),
                           name="kvm_o%d" % i) for i, (a, b) in enumerate(zip(o0, o1))]
    _store_row(builder, buffer, active, index, merged, o0, "kvm_o")
    _store_stat(builder, buffer, row=row, base=m_base, value=big, prefix="kvm_ms")
    _store_stat(builder, buffer, row=row, base=l_base, value=l, prefix="kvm_ls")


class ScaledBase:
    """A threadgroup base that is not a power of two of words (MM 25.135.5): threadgroup t's region starts
    t * words further in, emitted as one shift per set bit of `words` and adds (the shl and add every row
    stage already dispatches; no integer multiply). The S-way merge's slots are t * S * 5,632 words apart
    (22,528 bytes each, S per head). Duck-typed as HeadBase: every row-stage helper calls .emit."""
    def __init__(self, words):
        self.words = int(words)
        if not 0 < self.words < 1 << 24:
            raise ValueError("refused: a scaled threadgroup base is 1 .. 2^24 - 1 words")

    def emit(self, builder, prefix):
        tg = builder.builtin("threadgroup_position_in_grid", name=prefix + "_tg")
        bits = [k for k in range(24) if self.words >> k & 1]
        acc = None
        for k in bits:
            term = tg if k == 0 else builder.shl(tg, _i32_const(builder, k, "%s_sh%d" % (prefix, k)),
                                                 name="%s_t%d" % (prefix, k))
            acc = term if acc is None else builder.add(acc, term, name="%s_acc%d" % (prefix, k))
        return acc


class TileGroupBase:
    """THE TILED MERGE's scalar half (MM 25.132.7): threadgroup t = head * T + tile (T a power of two), and its
    base is base + (t >> log2 T) * head_words + (t & (T - 1)) * tile_words words. Each product is emitted as one
    shift per set bit and adds (no integer multiply). Duck-typed as HeadBase: every row-stage helper calls .emit;
    the threadgroup index is read afresh at each use (no scalar is live across a tensor body)."""
    def __init__(self, groups, head_words, tile_words=0, base_words=0):
        self.groups, self.head, self.tile, self.base = int(groups), int(head_words), int(tile_words), int(base_words)
        if self.groups < 2 or self.groups & (self.groups - 1) or self.groups > 128:
            raise ValueError("refused: a tile-group base takes 2..128 groups per head, a power of two")
        if not (0 < self.head < 1 << 24 and 0 <= self.tile < 1 << 24 and 0 <= self.base < 1 << 30):
            raise ValueError("refused: tile-group strides are below 2^24 words and the base below 2^30")

    def emit(self, builder, prefix):
        tg = builder.builtin("threadgroup_position_in_grid", name=prefix + "_tg")
        head = builder.shr(tg, _i32_const(builder, self.groups.bit_length() - 1, prefix + "_hsh"), name=prefix + "_head")
        acc = None
        for src, words, tag in ((head, self.head, "h"), (None, self.tile, "t")):
            if not words:
                continue
            if src is None:
                src = getattr(builder, "and")(tg, _i32_const(builder, self.groups - 1, prefix + "_tmask"),
                                              name=prefix + "_tile")
            for k in [k for k in range(24) if words >> k & 1]:
                term = src if k == 0 else builder.shl(src, _i32_const(builder, k, "%s_%s%d" % (prefix, tag, k)),
                                                      name="%s_%st%d" % (prefix, tag, k))
                acc = term if acc is None else builder.add(acc, term, name="%s_%sa%d" % (prefix, tag, k))
        if self.base:
            acc = builder.add(acc, _i32_const(builder, self.base, prefix + "_b0"), name=prefix + "_base")
        return acc


def emit_kv_merge_n(builder, buffer, *, row, dst, srcs, rescale=True):
    """THE S-WAY KV-SPLIT MERGE (MM 25.135.5) on one row: S partial states, each (o_bases, m_base, l_base,
    head) with its O tiles un-normalised and scaled to its own running max m_s, into the state `dst` (the
    same tuple form). M = max(m_0, .., m_{S-1}) folded in ascending s; f_s = exp2(m_s - M) as fmul(M, -1),
    fadd, exp2 (the stream's hardware exp2); l = the left fold over ascending s of f_s l_s and every O word
    the same fold of f_s O_s, each fmul and fadd rounded separately. The merged O tiles, M and l are written
    to `dst` (its fallback words restored), so emit_stream_normalize finishes it unchanged. At S = 2 the
    arithmetic is emit_kv_merge's (fmul, fmul, fadd); emit_kv_merge keeps its own bytes for the head-64 step.
    Each O word is folded as its slice is loaded, so a row holds one accumulator per word, not S values.
    `rescale=False` builds f_s = 1 (the merge_without_rescale program; tests only). A fifth tuple element is the
    base of the O tiles alone (the tiled merge, MM 25.132.7: its O tile moves with the threadgroup's tile, its M
    and l do not); without it the O tiles take the state's own base, as before."""
    if len(srcs) < 2:
        raise ValueError("refused: a merge takes at least two partial states")
    d_obases, d_m, d_l, d_head = dst[:4]
    d_ohead = dst[4] if len(dst) > 4 else d_head
    srcs = [tuple(x[:4]) + ((x[4] if len(x) > 4 else x[3]),) for x in srcs]
    ms = [_row_stat(builder, buffer, row=row, base=mb, **_hk(hd)) for (_ob, mb, _lb, hd, _oh) in srcs]
    big = ms[0]
    for s, m in enumerate(ms[1:], 1):
        big = builder.fmax(big, m, name="kvn_max%d" % s)
    if rescale:
        neg = builder.fmul(big, _f32_const(builder, -1.0, "kvn_neg_one"), name="kvn_neg")
        fs = [builder.exp2(builder.fadd(m, neg, name="kvn_d%d" % s), name="kvn_f%d" % s) for s, m in enumerate(ms)]
    else:
        one = _f32_const(builder, 1.0, "kvn_one")
        fs = [one] * len(srcs)
    l = None
    for s, (f, (_ob, _mb, lb, hd, _oh)) in enumerate(zip(fs, srcs)):
        prod = builder.fmul(f, _row_stat(builder, buffer, row=row, base=lb, **_hk(hd)), name="kvn_fl%d" % s)
        l = prod if l is None else builder.fadd(l, prod, name="kvn_l%d" % s)
    for t, dob in enumerate(d_obases):
        _lane, active, index, originals = _load_values(builder, buffer, row=row, stride=16, M=32, base=dob,
                                                       **_hk(d_ohead))
        acc = None
        for s, (f, (obs, _mb, _lb, _hd, ohd)) in enumerate(zip(fs, srcs)):
            _l, _a, _i, vals = _load_values(builder, buffer, row=row, stride=16, M=32, base=obs[t], **_hk(ohd))
            prods = [builder.fmul(f, v, name="kvn_fo%d_%d" % (s, i)) for i, v in enumerate(vals)]
            acc = prods if acc is None else [builder.fadd(a, p, name="kvn_o%d_%d" % (s, i))
                                             for i, (a, p) in enumerate(zip(acc, prods))]
        _store_row(builder, buffer, active, index, acc, originals, "kvn_o%d" % t)
    _store_stat(builder, buffer, row=row, base=d_m, value=big, prefix="kvn_ms", **_hk(d_head))
    _store_stat(builder, buffer, row=row, base=d_l, value=l, prefix="kvn_ls", **_hk(d_head))


def emit_stream_normalize(builder, buffer, *, row, o_base, l_base, head=None, o_bases=None, o_head=None):
    """ONLINE SOFTMAX, the end: O = O * recip(l) on one row. `head` and `o_bases` as in
    emit_stream_second_block (one recip, every O tile scaled by it); `o_head` is the O tiles' own base where
    it differs from l's (the tiled merge, MM 25.132.7)."""
    inv = builder.recip(_row_stat(builder, buffer, row=row, base=l_base, **_hk(head)), name="sn_inv_l")
    for ob in (o_bases if o_bases is not None else [o_base]):
        _lane, active, index, values = _load_values(builder, buffer, row=row, stride=16, M=32, base=ob,
                                                    **_hk(head if o_head is None else o_head))
        out = [builder.fmul(o, inv, name="sn_o%d" % i) for i, o in enumerate(values)]
        _store_row(builder, buffer, active, index, out, values, "sn_o")


def emit_row_gelu(builder, buffer, *, row=0, M=16, N=16, K=SUPPORTED_K, lane=None):
    """Apply the compiler-owned ``x * sigmoid(1.702*x)`` GELU approximation to one row.

    This is ordinary FP32 scalar/vector IR, not an Apple reduction or activation intrinsic. The
    measured application slice is one 16x16 tile; callers choose the row explicitly so a larger
    graph cannot silently imply an unmeasured register budget.
    """
    require_domain(M, N, K, accumulator="f32", simdgroups=1)
    if N != 16 or M != 16:
        raise UnsupportedReduction("compiler-owned GELU is measured only for a 16x16 FP32 tile")
    _lane, active, index, values = _load_values(builder, buffer, row=row, stride=N, M=M, lane=lane)
    alpha = _f32_const(builder, 1.702, "gelu_alpha")
    neg_inv_ln2 = _f32_const(builder, -1.4426950408889634, "gelu_neg_inv_ln2")
    one = _f32_const(builder, 1.0, "gelu_one")
    outputs = []
    for offset, value in enumerate(values):
        scaled = builder.fmul(value, alpha, name="gelu_scale%d" % offset)
        exponent = builder.exp2(builder.fmul(scaled, neg_inv_ln2,
                                             name="gelu_exp_arg%d" % offset),
                                name="gelu_exp2_%d" % offset)
        sigmoid = builder.recip(builder.fadd(exponent, one, name="gelu_den%d" % offset),
                                name="gelu_sigmoid%d" % offset)
        transformed = builder.fmul(value, sigmoid, name="gelu_value%d" % offset)
        outputs.append(_float_select(builder, active, transformed, value, "gelu_output%d" % offset))
    for offset, value in enumerate(outputs):
        idx = index if offset == 0 else builder.add(index, _i32_const(builder, offset,
                                                                       "gelu_store_offset%d" % offset),
                                                   name="gelu_store_index%d" % offset)
        builder.store_at(buffer, idx, value)
    return active


def emit_row_layernorm(builder, buffer, *, row=0, M=16, N=16, K=SUPPORTED_K, lane=None):
    """Normalize one FP32 row with the explicit compiler-owned sum/butterfly order."""
    require_domain(M, N, K, accumulator="f32", simdgroups=1)
    if N != 16 or M != 16:
        raise UnsupportedReduction("compiler-owned LayerNorm is measured only for a 16x16 tile")
    row_sum_value, active = emit_buffer_row_reduction(builder, buffer, row=row, operation="sum",
                                                      M=M, N=N, K=K, lane=lane)
    # The butterfly already replicates the result across the four lanes that own this row.  The
    # old row-zero slice could use broadcast-first because lane 0 was active; for row r>0 lane 0
    # carries the inactive identity, so broadcasting it would normalize against zero.  Keep the
    # measured row-zero bytes and use the active-lane result directly for the spatial path.
    if row == 0:
        row_sum_value = builder.simd_broadcast_first(row_sum_value, type=ir.F32, name="ln_sum_broadcast")

    _lane, active_sq, _index, values = _load_values(builder, buffer, row=row, stride=N, M=M, lane=lane)
    squares = [builder.fmul(value, value, name="ln_square%d" % offset)
               for offset, value in enumerate(values)]
    local_sq = squares[0]
    for offset, value in enumerate(squares[1:], 1):
        local_sq = builder.fadd(local_sq, value, name="ln_local_square_sum%d" % offset)
    local_sq = _float_select(builder, active_sq, local_sq, _f32_const(builder, 0.0, "ln_square_zero"),
                             "ln_active_square_sum")
    square_sum = emit_butterfly(builder, local_sq, ROW_BUTTERFLY_MASKS, operation="sum")
    if row == 0:
        square_sum = builder.simd_broadcast_first(square_sum, type=ir.F32, name="ln_square_sum_broadcast")

    inv_n = _f32_const(builder, 1.0 / 16.0, "ln_inv_n")
    mean = builder.fmul(row_sum_value, inv_n, name="ln_mean")
    mean_sq = builder.fmul(mean, mean, name="ln_mean_square")
    second_moment = builder.fmul(square_sum, inv_n, name="ln_second_moment")
    variance = builder.fadd(second_moment,
                            builder.fmul(mean_sq, _f32_const(builder, -1.0, "ln_neg_one"),
                                         name="ln_sub_mean_square"),
                            name="ln_variance_raw")
    variance = builder.fmax(variance, _f32_const(builder, 0.0, "ln_variance_zero"),
                            name="ln_variance")
    inv_std = builder.rsqrt(builder.fadd(variance, _f32_const(builder, 1.0e-5, "ln_epsilon"),
                                         name="ln_variance_eps"), name="ln_inv_std")

    _lane, active_out, index_out, output_values = _load_values(builder, buffer, row=row, stride=N, M=M,
                                                                 lane=lane)
    neg_mean = builder.fmul(mean, _f32_const(builder, -1.0, "ln_neg_mean"), name="ln_neg_mean_value")
    normalized = []
    for offset, value in enumerate(output_values):
        centered = builder.fadd(value, neg_mean, name="ln_centered%d" % offset)
        scaled = builder.fmul(centered, inv_std, name="ln_normalized%d" % offset)
        normalized.append(_float_select(builder, active_out, scaled, value, "ln_output%d" % offset))
    for offset, value in enumerate(normalized):
        idx = index_out if offset == 0 else builder.add(index_out, _i32_const(builder, offset,
                                                                               "ln_store_offset%d" % offset),
                                                       name="ln_store_index%d" % offset)
        builder.store_at(buffer, idx, value)
    return mean, inv_std, active


def emit_tile_gelu(builder, buffer, *, M=16, N=16, K=SUPPORTED_K):
    """Apply the measured GELU sequence to every row of one 16x16 tile.

    The row loop is explicit in the compiler IR.  A single lane builtin is shared by all rows so
    the allocator does not spend one low register on each repeated address calculation.
    """
    require_domain(M, N, K, accumulator="f32", simdgroups=1)
    if M != 16 or N != 16:
        raise UnsupportedReduction("all-row GELU is measured only for one 16x16 tile")
    lane = builder.builtin("thread_index_in_simdgroup", name="tile_gelu_lane")
    for row in range(M):
        emit_row_gelu(builder, buffer, row=row, M=M, N=N, K=K, lane=lane)
    return lane


def emit_tile_layernorm(builder, buffer, *, M=16, N=16, K=SUPPORTED_K):
    """Normalize every row of one measured 16x16 FP32 tile in explicit compiler order."""
    require_domain(M, N, K, accumulator="f32", simdgroups=1)
    if M != 16 or N != 16:
        raise UnsupportedReduction("all-row LayerNorm is measured only for one 16x16 tile")
    lane = builder.builtin("thread_index_in_simdgroup", name="tile_layernorm_lane")
    for row in range(M):
        emit_row_layernorm(builder, buffer, row=row, M=M, N=N, K=K, lane=lane)
    return lane


def _wide_row_addresses(builder, *, row, stride, lane=None):
    """Return the two 16-wide tile bases for one row of a 16x32 tile."""
    if row < 0 or row >= 16 or stride != 32:
        raise UnsupportedReduction("wide row reduction is measured only for a 16x32 tile")
    if lane is None:
        lane = builder.builtin("thread_index_in_simdgroup", name="wide_reduction_lane")
    lane_hi = builder.shr(lane, _i32_const(builder, 4, "wide_lane_shift4"), name="wide_lane_hi")
    lane_hi = getattr(builder, "and")(lane_hi, _i32_const(builder, 1, "wide_lane_hi_mask"),
                                        name="wide_lane_hi_bit")
    row_low = builder.shr(lane, _i32_const(builder, 1, "wide_lane_shift1"), name="wide_lane_shifted")
    row_low = getattr(builder, "and")(row_low, _i32_const(builder, 3, "wide_lane_low_mask"),
                                       name="wide_lane_low_bits")
    lane_row = builder.add(builder.shl(lane_hi, _i32_const(builder, 2, "wide_lane_hi_scale"),
                                       name="wide_lane_hi_scaled"), row_low, name="wide_lane_row")
    active = builder.icmp(lane_row, _i32_const(builder, row & 7, "wide_row_target"), rel="eq",
                          name="wide_lane_active")
    col_hi = getattr(builder, "and")(
        builder.shr(lane, _i32_const(builder, 3, "wide_col_shift3"), name="wide_col_shifted3"),
        _i32_const(builder, 1, "wide_col_hi_mask"), name="wide_col_hi_bit")
    col_lo = getattr(builder, "and")(lane, _i32_const(builder, 1, "wide_col_lo_mask"),
                                      name="wide_col_lo_bit")
    col = builder.add(builder.shl(col_hi, _i32_const(builder, 3, "wide_col_hi_scale"),
                                  name="wide_col_hi_scaled"),
                      builder.shl(col_lo, _i32_const(builder, 2, "wide_col_lo_scale"),
                                  name="wide_col_lo_scaled"), name="wide_row_col")
    active_index = builder.add(_i32_const(builder, row * stride, "wide_active_base"), col,
                               name="wide_active_index")
    # Inactive lanes use a deterministic first-row fallback.  Their selected value is preserved;
    # the fallback keeps every speculative load inside the declared 16x32 buffer.
    index = builder.csel(active, _i32_const(builder, 0, "wide_index_cond"), active_index,
                         _i32_const(builder, 0, "wide_fallback_index"), rel="gt",
                         name="wide_row_load_index")
    return lane, active, index


def _wide_load_values(builder, buffer, *, row, lane=None):
    lane, active, index = _wide_row_addresses(builder, row=row, stride=32, lane=lane)
    values = []
    indices = []
    for tile in (0, 1):
        tile_index = index if tile == 0 else builder.add(
            index, _i32_const(builder, 16, "wide_tile_offset"), name="wide_tile_index")
        indices.append(tile_index)
        for offset in range(4):
            idx = tile_index if offset == 0 else builder.add(
                tile_index, _i32_const(builder, offset, "wide_value_offset%d" % offset),
                name="wide_value_index%d" % offset)
            values.append(builder.load(buffer, idx, type=ir.F32,
                                       name="wide_row_value%d" % len(values)))
    return lane, active, indices, values


def emit_tile_gelu_wide(builder, buffer, *, M=16, N=32, K=SUPPORTED_K):
    """Apply GELU to every row of one measured two-tile 16x32 FP32 output."""
    require_domain(M, N, K, accumulator="f32", simdgroups=1)
    if (M, N) != (16, 32):
        raise UnsupportedReduction("wide GELU is measured only for one 16x32 tile pair")
    lane = builder.builtin("thread_index_in_simdgroup", name="wide_gelu_lane")
    alpha = _f32_const(builder, 1.702, "wide_gelu_alpha")
    neg_inv_ln2 = _f32_const(builder, -1.4426950408889634, "wide_gelu_neg_inv_ln2")
    one = _f32_const(builder, 1.0, "wide_gelu_one")
    for row in range(16):
        _lane, active, indices, values = _wide_load_values(builder, buffer, row=row, lane=lane)
        outputs = []
        for offset, value in enumerate(values):
            scaled = builder.fmul(value, alpha, name="wide_gelu_scale%d" % offset)
            exponent = builder.exp2(builder.fmul(scaled, neg_inv_ln2,
                                                 name="wide_gelu_exp_arg%d" % offset),
                                    name="wide_gelu_exp2_%d" % offset)
            sigmoid = builder.recip(builder.fadd(exponent, one,
                                                 name="wide_gelu_den%d" % offset),
                                    name="wide_gelu_sigmoid%d" % offset)
            transformed = builder.fmul(value, sigmoid, name="wide_gelu_value%d" % offset)
            outputs.append(_float_select(builder, active, transformed, value,
                                         "wide_gelu_output%d" % offset))
        for tile, tile_index in enumerate(indices):
            for offset in range(4):
                pos = tile * 4 + offset
                idx = tile_index if offset == 0 else builder.add(
                    tile_index, _i32_const(builder, offset, "wide_gelu_store_offset%d" % offset),
                    name="wide_gelu_store_index%d" % offset)
                builder.store_at(buffer, idx, outputs[pos])
    return lane


def emit_tile_layernorm_wide(builder, buffer, *, M=16, N=32, K=SUPPORTED_K):
    """Normalize every row of one 16x32 FP32 output using explicit compiler-owned reductions."""
    require_domain(M, N, K, accumulator="f32", simdgroups=1)
    if (M, N) != (16, 32):
        raise UnsupportedReduction("wide LayerNorm is measured only for one 16x32 tile pair")
    lane = builder.builtin("thread_index_in_simdgroup", name="wide_layernorm_lane")
    inv_n = _f32_const(builder, 1.0 / 32.0, "wide_ln_inv_n")
    for row in range(16):
        _lane, active, indices, values = _wide_load_values(builder, buffer, row=row, lane=lane)
        local = values[0]
        for value in values[1:]:
            local = builder.fadd(local, value, name="wide_ln_local_sum")
        local = _float_select(builder, active, local, _f32_const(builder, 0.0, "wide_ln_sum_zero"),
                              "wide_ln_active_sum")
        total = emit_butterfly(builder, local, ROW_BUTTERFLY_MASKS, operation="sum")
        if row == 0:
            total = builder.simd_broadcast_first(total, type=ir.F32, name="wide_ln_sum_broadcast")
        squares = [builder.fmul(value, value, name="wide_ln_square") for value in values]
        square_local = squares[0]
        for value in squares[1:]:
            square_local = builder.fadd(square_local, value, name="wide_ln_local_square")
        square_local = _float_select(builder, active, square_local,
                                     _f32_const(builder, 0.0, "wide_ln_square_zero"),
                                     "wide_ln_active_square")
        square_total = emit_butterfly(builder, square_local, ROW_BUTTERFLY_MASKS, operation="sum")
        if row == 0:
            square_total = builder.simd_broadcast_first(square_total, type=ir.F32,
                                                         name="wide_ln_square_broadcast")
        mean = builder.fmul(total, inv_n, name="wide_ln_mean")
        variance = builder.fadd(builder.fmul(square_total, inv_n, name="wide_ln_second_moment"),
                                builder.fmul(builder.fmul(mean, mean, name="wide_ln_mean_square"),
                                             _f32_const(builder, -1.0, "wide_ln_neg_one"),
                                             name="wide_ln_sub_mean_square"),
                                name="wide_ln_variance_raw")
        variance = builder.fmax(variance, _f32_const(builder, 0.0, "wide_ln_variance_zero"),
                                name="wide_ln_variance")
        inv_std = builder.rsqrt(builder.fadd(variance, _f32_const(builder, 1.0e-5,
                                                                    "wide_ln_epsilon"),
                                             name="wide_ln_variance_eps"), name="wide_ln_inv_std")
        neg_mean = builder.fmul(mean, _f32_const(builder, -1.0, "wide_ln_neg_mean"),
                                name="wide_ln_neg_mean_value")
        outputs = []
        for value in values:
            outputs.append(_float_select(builder, active,
                                         builder.fmul(builder.fadd(value, neg_mean,
                                                                   name="wide_ln_centered"),
                                                      inv_std, name="wide_ln_normalized"),
                                         value, "wide_ln_output"))
        for tile, tile_index in enumerate(indices):
            for offset in range(4):
                pos = tile * 4 + offset
                idx = tile_index if offset == 0 else builder.add(
                    tile_index, _i32_const(builder, offset, "wide_ln_store_offset%d" % offset),
                    name="wide_ln_store_index%d" % offset)
                builder.store_at(buffer, idx, outputs[pos])
    return lane


def emit_tile_residual_add_wide(builder, target, residual, *, M=16, N=32, K=SUPPORTED_K):
    """Add a half-valued residual tile to a wide FP32 tile in compiler-owned order.

    The measured tensor runtime class has three public buffers, so this first transformer slice
    keeps the residual in the readonly half buffer already used by the graph.  The operation is
    still a real GPU path: each active lane loads its residual half values, widens them with the
    measured f16->f32 instruction, adds them to the FP32 target and stores the result before the
    explicit LayerNorm reduction.  No host-side residual or implicit dtype conversion is allowed.
    """
    require_domain(M, N, K, accumulator="f32", simdgroups=1)
    if (M, N) != (16, 32):
        raise UnsupportedReduction("wide residual add is measured only for one 16x32 tile pair")
    lane = builder.builtin("thread_index_in_simdgroup", name="wide_residual_lane")
    for row in range(16):
        _lane, active, indices, values = _wide_load_values(builder, target, row=row, lane=lane)
        outputs = []
        for tile, tile_index in enumerate(indices):
            for offset in range(4):
                pos = tile * 4 + offset
                idx = tile_index if offset == 0 else builder.add(
                    tile_index, _i32_const(builder, offset, "wide_residual_offset%d" % offset),
                    name="wide_residual_index%d" % pos)
                half_value = builder.load(residual, idx, type=ir.I16, width="half",
                                          name="wide_residual_half%d" % pos)
                residual_value = builder.f16_to_f32(half_value,
                                                    name="wide_residual_float%d" % pos)
                outputs.append(builder.fadd(values[pos], residual_value,
                                            name="wide_residual_sum%d" % pos))
        for tile, tile_index in enumerate(indices):
            for offset in range(4):
                pos = tile * 4 + offset
                idx = tile_index if offset == 0 else builder.add(
                    tile_index, _i32_const(builder, offset, "wide_residual_store_offset%d" % offset),
                    name="wide_residual_store_index%d" % pos)
                builder.store_at(target, idx, _float_select(builder, active, outputs[pos], values[pos],
                                                             "wide_residual_output%d" % pos))
    return lane


def emit_split_k_fold(builder, partials, out, *, G, M, N, M_live=None):
    """Fold G stacked M x N partials into one M x N tile by an ascending-t fp32 LEFT fold.

    ``partials`` is the (G*M) x N row-major fp32 buffer split-K wrote: block t occupies rows
    [t*M, (t+1)*M).  ``out`` is the M x N row-major fp32 result.  This is exactly the reduce half of
    Piece A's ``gemm_reference(split_k=G)`` (tools/g17decodestep.py): a sequential fp32 chain
    ``acc = P[0*M+m, c]; acc = acc + P[t*M+m, c]`` in ascending t.  A 32-wide column grid: threadgroup g
    owns columns [g*32, g*32+32), lane L the single column c = g*32 + L, and folds M_live rows of it.  The
    index is a full register (op17229 computed store), so every lane lands on its own column - no slot race.

    ``M_live`` (default M) folds only rows [0, M_live) - the LIVE rows.  A decode step pads its one token
    to a 16-row tile, so M_live=1 folds only row 0 and skips the 15 padding rows: 16x less work per column
    (about 32 us -> a few us per fold, MM 25.132.3).  The consumer reads only the live rows.  M_live does
    not change the values it does compute - it is the same ascending-t fp32 fold, on fewer rows.
    """
    if not isinstance(G, int) or G < 2:
        raise UnsupportedReduction("split-K fold needs at least two partials (G >= 2)")
    if N % 32 or M < 1:
        raise UnsupportedReduction("split-K fold is a 32-wide column grid; N must be a multiple of 32")
    if M_live is None:
        M_live = M
    if not isinstance(M_live, int) or not 1 <= M_live <= M:
        raise UnsupportedReduction("split-K fold M_live is 1..M (the live rows; the rest are padding)")
    lane = builder.builtin("thread_index_in_simdgroup", name="fold_lane")
    tg = builder.builtin("threadgroup_position_in_grid", name="fold_group")
    col = builder.add(builder.shl(tg, builder.const(5, name="fold_tg_shift"), name="fold_tg_cols"),
                      lane, name="fold_col")                         # c = tg*32 + lane
    for m in range(M_live):
        # index of P[m, c] and of out[m, c]: m*N + c (block-0 row m; later blocks add t*M*N)
        base = col if m == 0 else builder.add(col, builder.const(m * N, name="fold_rowoff_m%d" % m),
                                              name="fold_idx_m%d" % m)
        acc = builder.load(partials, base, type=ir.F32, name="fold_p0_m%d" % m)
        for t in range(1, G):
            tidx = builder.add(base, builder.const(t * M * N, name="fold_blk_t%d_m%d" % (t, m)),
                               name="fold_idx_t%d_m%d" % (t, m))     # P[t*M+m, c] = base + t*M*N
            acc = builder.fadd(acc, builder.load(partials, tidx, type=ir.F32,
                                                 name="fold_p_t%d_m%d" % (t, m)),
                               name="fold_acc_t%d_m%d" % (t, m))
        builder.store_at(out, base, acc)
    return col


def emit_simdgroup_combine(builder, value, *, simdgroups, operation="sum", slot_base=0, sg_index=None):
    """Combine one FP32 value per simdgroup across the threadgroup (Set A item 7).

    Registers never cross a simdgroup, so the handoff is threadgroup memory with a workgroup
    barrier between the stores and the loads: recon sections 118, 119 and 133 measured the
    barrier as necessary (0 of 12 correct without one, or with a simdgroup-only barrier) and any
    workgroup barrier as sufficient. Each simdgroup's lanes all hold the same `value` (a butterfly
    result), so every lane stores it to word `slot_base + sg`; the stores are the same value, and
    storing from every lane avoids selecting one. After the barrier every lane folds words
    `slot_base .. slot_base + simdgroups - 1` in ASCENDING order, so the result is the same on
    every lane and does not depend on which simdgroup arrived first.

    The caller declares the threadgroup words (`declare_threadgroup`) and supplies the simdgroup
    index or lets this read `simdgroup_index_in_threadgroup`.
    """
    if operation not in ("sum", "max"):
        raise UnsupportedReduction("cross-simdgroup operation %r is not measured" % operation)
    if simdgroups not in (2, 4):
        raise UnsupportedReduction("cross-simdgroup combine is written for 2 or 4 simdgroups")
    if sg_index is None:
        sg_index = builder.builtin("simdgroup_index_in_threadgroup", name="combine_sg")
    slot = builder.add(sg_index, _i32_const(builder, slot_base, "combine_slot_base"), name="combine_slot")
    builder.store_tg(value, slot)
    builder.barrier("threadgroup")
    total = builder.load_tg(_i32_const(builder, slot_base, "combine_word0"), type=ir.F32, name="combine_part0")
    for s in range(1, simdgroups):
        part = builder.load_tg(_i32_const(builder, slot_base + s, "combine_word%d" % s), type=ir.F32,
                               name="combine_part%d" % s)
        total = (builder.fadd(total, part, name="combine_sum%d" % s) if operation == "sum"
                 else builder.fmax(total, part, name="combine_max%d" % s))
    return total
