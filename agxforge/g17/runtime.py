"""Native runtime contracts and FP16 references. Importing never loads Metal.

These retained application contracts describe allocation and transport only;
constructing one grants no image admission or execution permission. Existing
FP16 manifests retain version 1.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .abi import Binding, Instruction, allow_requantized_binding

CONFIG = ConfigDict(strict=True, extra="forbid", frozen=True,
                    ser_json_bytes="hex", val_json_bytes="hex")
NAMES = {"packed_scan": "half_scan", "separate_scan": "half_scan_separate", "affine": "half_affine",
         "layernorm": "minilm_layernorm", "minilm_query": "minilm_layer0_query"}

# WHICH MANIFEST VERSION AN APPLICATION KIND CARRIES, IN ONE PLACE. The validator below and the
# image author both need it, and while each stated it separately they disagreed the moment a kind
# was added: minilm_query is a v2 application here and the author was still sending v1 for
# everything that was not literally "layernorm", so a correct query contract was refused as
# "application kind and runtime manifest version disagree". Deriving both from this map is what
# keeps the next added kind from reintroducing it.
FOUR_BUFFER_KINDS = ("layernorm", "minilm_query")
TENSOR_KIND = "tensor_gemm"
REQUANT_KIND = "tensor_requantization"


# MM P12 (docs/g17-tensorops-machine-model.md), CLOSED BY ITS SECOND BRANCH: the production refusal of
# cooperative threadgroup-memory SHARING, by name. What production admits at simdgroups 2 and 4
# (gemm_generic) is a per-simdgroup tile PARTITION, which is also what Apple's own
# execution_simdgroups>1 code emits; no tensor class shares operands through threadgroup memory. Every
# path that could request sharing - a spec key, a TensorSpec key, a tensor ABI that uses or declares
# threadgroup memory, a compiled tensor program that touches it - raises this text, and the native
# worker refuses the same manifests as phase=tensor_cooperative_sharing.
COOPERATIVE_SHARING_REFUSAL = (
    "refused: cooperative threadgroup-memory sharing is not admitted in production (MM P12: the "
    "research form is exact through 16 simdgroups but 32 exceeds the 32 KiB threadgroup memory, "
    "cross-threadgroup sharing cannot synchronize in one dispatch, and Apple's multi-simdgroup MMA "
    "code partitions tiles rather than sharing, mech-4)")
# a key naming sharing, cooperation or threadgroup memory is a sharing REQUEST, whatever its value:
# an unrecognised spelling must not be dropped and built as the tile split
SHARING_KEY_WORDS = ("shar", "cooperat", "threadgroup_mem", "tg_mem")


def sharing_request(mapping):
    """The keys of `mapping` that request cooperative sharing (sorted), or []."""
    if not isinstance(mapping, dict):
        return []
    return sorted(k for k in mapping if isinstance(k, str) and any(w in k.lower() for w in SHARING_KEY_WORDS))


def refuse_sharing(mapping):
    """Raise the named P12 refusal if `mapping` requests cooperative sharing by any key."""
    keys = sharing_request(mapping)
    if keys:
        raise ValueError(COOPERATIVE_SHARING_REFUSAL + "; requested by key(s) %s" % ", ".join(keys))


def refuse_tensor_threadgroup_abi(value):
    """A tensor (ABI v5) contract that uses threadgroup memory or carries a threadgroup block is a
    sharing request: refuse it by name, not as a generic semantic mismatch."""
    if isinstance(value, dict) and value.get("abi_version") == 5 and (
            value.get("uses_threadgroup") is True or value.get("threadgroup") is not None):
        raise ValueError(COOPERATIVE_SHARING_REFUSAL + "; the tensor ABI uses or declares threadgroup memory")


# THE FUSED ATTENTION CLASS (production row P7, docs/g17-tensorops-machine-model.md 25.129). One named
# composition, "attention", carried on gemm_generic's transport (the worker sizes the three buffers from
# M, N and K as it does for every generic program), with its own admission rules and named refusals.
# A program of this class is, in order: the K and V projections of the NEW key blocks, each a
# 16 x 64 x 64 GEMM whose final ("half_kv",) epilogue writes its result as halves straight into a
# keys x head cache region of buffer 3 - the operand layout the attention bodies read, so there is
# no relayout; then, per key block, QK (Q from buffer 1, the K cache block under transB), the
# online-softmax row stage (first block, or the rescaling stage for every later block) with the
# causal mask applied to the scores before the row max, and PV (P from buffer 3, the V cache
# block); then the per-row normalisation. Every boundary is the memory bridge, one simdgroup,
# straight-line code.
ATTENTION_COMPOSITION = "attention"
ATTENTION_D = 64               # model width: X rows and the projection's K
ATTENTION_HEAD = 64            # QK head width (Q, K)
ATTENTION_VALUE = 16           # PV width (V, O)
ATTENTION_BLOCK = 16           # keys per block (one 16-column score tile)
ATTENTION_MAX_ROWS = 16        # the row stages' measured range: rows 0..15 of the first 32-row tile
# THE STRAIGHT-LINE BOUND: the largest key-block count dispatched bit-exact (25.129) with immediate
# offsets. Past it a program needs register-based tensor offsets (25.114.3/4) and, past 16 blocks,
# the counted key-block loop under 25.116's discipline (at most 255 trips, a decoded latch check,
# one dispatch per process) - key_offsets="loop" below (MM 25.114.5).
# ATTENTION_MAX_BLOCKS stays the bound of the IMMEDIATE-offset programs (their pinned bytes and P9's
# step capacity). THE REGISTER ROUTE (MM 25.114.3 and 25.114.4) is key_offsets="register": every QK
# body reads the K cache (buffer 3, transB) and every PV body the V cache (buffer 3) at a fixed
# immediate plus a register the stream advances per block (tlower b_index, R125/R124). It is
# word-identical to the immediate programs at 2 and 8 blocks on hardware, and admitted to
# ATTENTION_MAX_REGISTER_BLOCKS, straight-line. Past that: key_offsets="loop".
ATTENTION_MAX_BLOCKS = 8
ATTENTION_MAX_REGISTER_BLOCKS = 16
# THE COUNTED KEY-BLOCK LOOP (MM 25.114.5): key_offsets="loop" emits ONE QK -> row stage -> PV body
# inside a counted loop of `blocks` trips (a compile-time bound, cc.tensor_loop_route and
# tensorlife.counted_loop_check), the K/V cache offsets carried by the register route's registers,
# set once before the loop (tensor_index_init). Every trip is the general online step from
# m = -FLT_MAX, l = 0, O = 0, so block 0 needs no body of its own. "loop_frozen" is the same program
# with the registers never advanced (the failing control). The bound is the largest count dispatched
# (64 blocks, KV 1,024). 128 blocks (KV 2,048) compiles and passes the latch check, but its C buffer
# needs M = 1,056 transport rows and the worker's generic transport admits at most 1,024 per simdgroup
# (g17commonworker.m genericSpec): refused at load-only, nothing dispatched (MM 25.114.5).
ATTENTION_MAX_LOOP_BLOCKS = 64
ATTENTION_KEY_OFFSETS = ("immediate", "register", "frozen", "loop", "loop_frozen")
ATTENTION_LOOP_OFFSETS = ("loop", "loop_frozen")
# buffer 3 (bytes): the score tile S (32 x 16 fp32, reused per block), the output O, the running max
# and sum tiles, then the K cache (blocks x 16 keys x 64 halves) and the V cache (blocks x 16 x 16
# halves). Buffer 1: Q (32 x 64 halves) then X (16 rows per new block x 64). Buffer 2: Wk (64 x 64)
# then Wv (64 x 16) halves - 64 x 80, the transport's K x N.
ATTENTION_C = {"S": 0, "O": 2048, "M": 4096, "L": 6144, "KC": 8192}
ATTENTION_B = {"WK": 0, "WV": ATTENTION_D * ATTENTION_HEAD * 2}
ATTENTION_A = {"Q": 0, "X": 32 * ATTENTION_D * 2}
ATTENTION_PHASES = ("fused", "project", "attend")
# THE SCHEDULES A SCHEDULER SELECTS BETWEEN, BY NAME (P11 selects; this class does not choose).
# "attention.fused": one dispatch, projections then attention (phase "fused"). It is P11's "fused"
# (agxforge/g17/tensorsched.py choose_attention_cut, MM 25.124.3): S and P never leave the dispatch
# (they cross the row stages through buffer 3 inside it).
# "attention.proj_cut": two dispatches through buffer 3 - phase "project" writes the new K/V cache
# blocks, phase "attend" reads them. It is P11's exposed kind "proj_cut" (choose_attention_cut returns
# schedule kind "attention.proj_cut"). The cut is at the projection, not inside the attention, so it is
# none of P11's p_cut, s_cut or both_cuts (which materialize P or S between dispatches and are not built
# here). Bit-identical results across the two schedules are the class contract (25.129).
ATTENTION_SCHEDULES = {"attention.fused": ("fused",), "attention.proj_cut": ("project", "attend")}
# the subset of tensorsched.ATTN_SCHEDULES this class realizes: choose_attention_cut(workload,
# exposed=ATTENTION_P11_EXPOSED)
ATTENTION_P11_EXPOSED = ("fused", "proj_cut")

# THE HEAD GRID PHASE (MM 25.135; the decode milestone's attention stage, MM 25.132 item 1). Phase "grid"
# attends over a HOST-HELD K/V cache (no projection: the step's qkv_proj and rope_append produce Q, K and
# V) at QK head 128 and value 128, for `heads` heads launched as `heads` threadgroups of one simdgroup.
# Threadgroup t owns head t: every tensor body adds t * stride to its A, B and C indices (tlower's
# head_index, from SR_TG_X) and every row stage adds t * 16384 words to its buffer-3 addresses
# (tensorreduce.HeadBase), so ONE image serves every head, with no per-head immediate.
#   head 128: QK is ONE 32 x 16 x 128 body (eight 16-wide issues in ascending K, one MMA chain), which is
#     g17decodestep's single-chain `gemm(q, K^T)` bit for bit; two 64-wide sub-tiles joined by an
#     accumulate would add the second half once after its own chain, a different rounding.
#   value 128: eight 32 x 16 x 16 PV bodies, one per 16-wide slice of V into its own 32 x 16 O tile, so
#     the row stages keep their measured 32 x 16, stride-16 addressing; alpha rescales all eight tiles and
#     the normalisation scales all eight. The V cache is stored BLOCK-MAJOR in 16-wide slices (block j,
#     slice s at j * 4096 + s * 512 bytes, each 16 keys x 16 halves), so every PV body reads a contiguous
#     16 x 16 B, and consecutive PV bodies read consecutive 512-byte pieces.
#   key blocks: 1..8 with immediate offsets (every B offset inside P10's measured stream domain), or
#     1..ATTENTION_GRID_CAPACITY = 17 with key_offsets="register" (MM 25.114.3/25.114.4: every QK body reads
#     K at the cache base plus register "k", advanced 4096 bytes per block, and every PV body reads V at
#     the base plus register "v", advanced 512 bytes per body), straight-line. 17 blocks hold the
#     milestone's 256 cached keys plus the new one. At one row a block costs about 27 KB of code, so 17
#     blocks stay under the 1,000,000-byte image contract with no loop; many rows at many blocks do not
#     (the contract refuses the image), and that is where the counted key-block loop is needed.
#   rows: 1..16 query rows of the 32-row score tile; rows past `rows` are ZERO-PADDED Q rows, whose
#     scores (0), P (raw 0) and O (0) are computed and discarded. Decode is rows 1: the MMAs do 32 rows of
#     work for one, the row stages one row (the padding cost, MM 25.135).
#   a KV CHAIN ACROSS DISPATCHES (resume / normalize): a program visits at most ATTENTION_MAX_BLOCKS
#     blocks; a longer cache runs as several dispatches in order, each resuming the (O, m, l) state the
#     previous one left in buffer 3 (its first block takes the rescaling stage, its PV accumulates) and
#     only the last normalising. The host writes each dispatch's K and V blocks. This is the sequential
#     online softmax, the same arithmetic as one program over every block (not a KV split: nothing is
#     merged, so no value changes).
#   THE COUNTED KEY-BLOCK LOOP (key_offsets="loop", MM 25.135.4): Piece B's loop (25.114.5) at head 128.
#     One QK -> online step -> eight PV body runs `loop_blocks` trips over the blocks no row masks, from a
#     written state (O = 0, m = -FLT_MAX, l = 0) and the K/V registers set once; the blocks the causal
#     mask reaches are peeled after the loop on the same registers. Word-identical on hardware to the
#     straight-line programs; the milestone image is 73,650 bytes against 519,042. "loop_frozen" never
#     advances the registers (the failing control). Up to ATTENTION_GRID_CAPACITY blocks (the layout),
#     never resumed.
# Buffer layout, per head (bytes): buffer 1 holds Q (32 x 128 halves) at head * 8192; buffer 2 holds the
# V cache (17 blocks x 8 slices x 16 keys x 16 halves) at head * 131072; buffer 3 holds, at head * 131072,
# the K cache (17 blocks x 16 keys x 128 halves, keys x head, read under transB) at 0, then S (32 x 16
# fp32), the eight O tiles, the running max M and sum L (32 x 16 fp32 each). The immediate B offsets are
# inside P10's measured stream domain (at most 28,672 for K and 32,256 for V at 8 blocks; 0 on the register
# route); the head strides and the register offsets are added in registers (up to 69,632 bytes into a
# head's K or V cache at 17 blocks: a new measurement, MM 25.135). Transport: M = 128 rows per head,
# N = 256, K = 4096, so buffer 3 is 128 KB per head and b.f16 holds 16 heads' V caches exactly.
ATTENTION_GRID_PHASE = "grid"
ATTENTION_GRID_SHAPES = ((128, 128),)      # (QK head, value) admitted by phase grid
ATTENTION_MAX_HEADS = 16                   # b.f16's V caches, and the milestone's 16 heads
ATTENTION_GRID_HEADS = (1, 2, 4, 8, 16)    # a power of two of threadgroups (the worker's launch rule)
ATTENTION_GRID_CAPACITY = 17               # key blocks per head in the layout (272 keys)
ATTENTION_GRID_STRIDE = {"A": 8192, "B": 131072, "C": 131072}   # bytes per head in buffers 1, 2, 3
ATTENTION_GRID_C = {"KC": 0, "S": 69632, "O": 71680, "M": 88064, "L": 90112}
# the runtime length word (MM 25.138.1): a uint32 in head 0's region past everything the split, the merge and the
# split's padded key reads touch (the merge's second carrier ends at 94,208; padding reads end at 98,304)
ATTENTION_GRID_LENGTH_BYTE = 131008
ATTENTION_GRID_TRANSPORT = {"rows_per_head": 128, "N": 256, "K": 4096}
# THE KV SPLIT ON THE LOOP (kv_split S, MM 25.114.6; the partial layout is Piece A's contract for P9's merge).
# Threadgroup t = head * S + slice: head = t >> log2 S, slice = t & (S - 1), from one SR_TG_X read. The key
# blocks are PADDED to a multiple of S with masked keys; slice s runs the SAME static trip count
# blocks_per_slice over blocks [s * bps, (s + 1) * bps), its K/V reads moved s * bps blocks by the slice
# stride. A per-trip key mask (tensorreduce.KeyMask, the threshold memory-carried in the slot) masks every
# key past q0 + row, which covers the causal edge and every padded key. Each threadgroup writes its partial
# (the un-normalised stream state: S, eight O tiles, M, L, 32 x 16 fp32 each) to its OWN slot of buffer 3 at
# heads * 131072 + t * 22528 bytes. K and V caches stay per head and are only read.
ATTENTION_GRID_SPLITS = (1, 2, 4, 8)
ATTENTION_SPLIT_SLOT = 22528               # bytes per threadgroup: 11 tiles of 32 x 16 fp32
ATTENTION_SPLIT_C = {"S": 0, "O": 2048, "M": 18432, "L": 20480, "W": 18432 + 4 * 320}   # W: the key threshold
ATTENTION_SPLIT_MAX_PADDED = 32            # padded blocks per head: the K and V head strides hold 32 x 4096

# THE S-WAY KV-SPLIT MERGE (MM 25.135.5): phase "grid_merge", its own dispatch after the split. The SPLIT side
# (Piece B's worker) runs heads * S threadgroups; threadgroup tg = head * S + slice runs the key-block loop over
# its slice and leaves an UN-NORMALISED partial in its own scratch slot, past every head's grid region:
#   slot(tg) = heads * 131072 + tg * 22528 bytes of buffer 3; inside it S at +0, the eight O tiles at
#   +2048 + 2048 t (32 x 16 fp32, row stride 16 words, scaled to the slice's running max m_s), M at +18432 and
#   L at +20480 (32 x 16 fp32, the row scalar replicated across its 16 words: tensorreduce._store_stat).
# Rows 0..rows-1 are meaningful (the split keeps its per-trip key threshold in M row 20; the merge never reads
# it). A slice a row sees no key of holds the stream's initial state: m = -FLT_MAX, l = 0, O = +0. Slice 0
# always holds key 0. The MERGE runs `heads` threadgroups, one per head (head = the threadgroup index, as the
# grid), and per attended row: M = max over s (ascending), f_s = exp2(m_s - M) (fmul by -1, fadd, exp2: the
# hardware exp2 the stream uses), l = the left fold over ascending s of f_s l_s, O likewise of f_s O_s, every
# fmul and fadd rounded separately; the merged O, M and L go to the head's EXISTING grid location (O 71680,
# M 88064, L 90112), then the grid's normalisation (emit_stream_normalize) finishes it unchanged.
ATTENTION_GRID_MERGE_PHASE = "grid_merge"
ATTENTION_KV_SPLITS = (2, 4, 8)
KV_SLOT_BYTES = 22528
KV_SLOT = {"S": 0, "O": 2048, "M": 18432, "L": 20480}
ATTENTION_WORKER_MAX_ROWS = 16384          # the common worker's transport limit on M


def kv_split_c_bytes(heads, S):
    """Buffer 3's extent that both the split and the merge address: every head's grid region, then heads * S
    scratch slots."""
    return ATTENTION_GRID_STRIDE["C"] * heads + KV_SLOT_BYTES * heads * S


def kv_split_transport_rows(c_bytes, threadgroups, N=256):
    """M for a program of `threadgroups` threadgroups over a buffer 3 of c_bytes: the smallest multiple of 16 x
    threadgroups whose M x N fp32 rows cover c_bytes (the rule both sides of the split size by)."""
    unit = 16 * threadgroups
    rows = -(-c_bytes // (4 * N))
    M = -(-rows // unit) * unit
    if M > ATTENTION_WORKER_MAX_ROWS:
        raise ValueError("refused: %d transport rows exceed the worker's %d" % (M, ATTENTION_WORKER_MAX_ROWS))
    return M


def kv_split_slot(heads, S, head, s):
    """Byte offset in buffer 3 of the scratch slot of (head, slice s): heads * 131072 + (head * S + s) * 22528."""
    if not (0 <= head < heads and 0 <= s < S):
        raise ValueError("kv split slot: head %r of %d, slice %r of %d" % (head, heads, s, S))
    return ATTENTION_GRID_STRIDE["C"] * heads + (head * S + s) * KV_SLOT_BYTES


def kv_split_slices(blocks, S):
    """The split's slices: `blocks` key blocks padded up to S equal slices; (padded block count, [slice ->
    list of block indices]). The padded blocks (blocks .. padded - 1) hold no key; a padded block's K and V
    reads land past the 17-block caches (kv_split_padding)."""
    if S not in ATTENTION_KV_SPLITS:
        raise ValueError("refused: kv split %r (the merge is built for %s ranges)" % (S, ATTENTION_KV_SPLITS))
    per = -(-blocks // S)
    return per * S, [list(range(s * per, (s + 1) * per)) for s in range(S)]


def kv_split_padding(blocks, S):
    """Where a padded block's operand reads land, per head (bytes, relative to the head's base in that buffer):
    K in buffer 3 at the cache base + 4096 j (from 69,632, the head's S / O / M / L region and past it: the
    scores are replaced by the causal select, so any finite or non-finite K is harmless), V in buffer 2 at
    4096 j + 512 slice (past the 17-block V cache, inside the 131,072-byte head stride). P is exactly 0 on a
    padded key, so O += 0 x V: V MUST BE FINITE there (0 x inf and 0 x NaN are NaN). The grid harness
    (_grid_buffers) writes buffer 2 from zeros every dispatch, so the padded V bytes are +0."""
    padded, _slices = kv_split_slices(blocks, S)
    kb = ATTENTION_BLOCK * 128 * 2
    return {"padded_blocks": list(range(blocks, padded)),
            "K_buffer3": [(ATTENTION_GRID_C["KC"] + kb * j, ATTENTION_GRID_C["KC"] + kb * (j + 1))
                          for j in range(blocks, padded)],
            "V_buffer2": [(kb * j, kb * (j + 1)) for j in range(blocks, padded)]}

ATTENTION_REFUSALS = {
    "attention_rows": "refused: attention_rows: 1..16 query rows (the row stages' measured range, rows 0..15 of "
                      "the first 32-row tile); a 32-row score tile is not measured",
    "attention_blocks": "refused: attention_blocks: 1..%d key blocks with immediate offsets, 1..%d with "
                        "key_offsets=register (MM 25.114.4), in straight-line code; 1..%d with key_offsets=loop, "
                        "the counted key-block loop (MM 25.114.5)"
                        % (ATTENTION_MAX_BLOCKS, ATTENTION_MAX_REGISTER_BLOCKS, ATTENTION_MAX_LOOP_BLOCKS),
    "attention_loop": "refused: attention_loop: the counted key-block loop runs one body for every block, so it "
                      "takes no compile-time mask: causal only when every key is visible to every row (q0 + 1 >= "
                      "16 * blocks, the decode case)",
    "attention_shape": "refused: attention_shape: model width 64, QK head 64, value width 16 and 16-key blocks "
                       "are the only admitted shape",
    "attention_launch": "refused: attention_launch: one simdgroup in one threadgroup",
    "attention_mask": "refused: attention_mask: the only mask is causal, with a non-negative compile-time first "
                      "query position; other masks (padding, sliding window, arbitrary) are not built",
    "attention_cache": "refused: attention_cache: prefilled cache blocks plus new blocks must be 1..%d, new "
                       "blocks at the end of the cache" % ATTENTION_MAX_BLOCKS,
    "attention_runtime_offset": "refused: attention_runtime_offset: in phases fused, project and attend the cache "
                                "append offset is a compile-time block index; an offset read at run time is phase "
                                "step, the stateful attention runtime (P9, MM 25.131: KVCache owns the cache and "
                                "its length, and the step reads the length from a uniform word)",
    "attention_kv_split": "refused: attention_kv_split: a KV split and its merge change the values (recon 154, and "
                          "MM 25.131 for this class); it is admitted only in phase step, as kv_split 2 with "
                          "allow_value_change=True, which is what P11's choose_kv_split selects with "
                          "allow_value_change, and as phase grid_merge (MM 25.135.5: the S-way merge of a "
                          "head-128 split, S in 2, 4, 8, with allow_value_change=True)",
    "kv_step": "refused: kv_step: phase step takes capacity_blocks, rows, kv_split and allow_value_change only; "
               "its length is read at run time from the uniform word, it is always causal (the queries are the "
               "appended tokens), and it appends exactly one 16-token block",
    "kv_capacity": "refused: kv_capacity: capacity_blocks is 1..%d (the straight-line bound: every capacity "
                   "block is visited and masked past the runtime length; the counted key-block loop of MM 25.114.5 takes a "
                   "compile-time trip count and is not admitted in phase step)"
                   % ATTENTION_MAX_BLOCKS,
    "kv_length": "refused: kv_length: the runtime length is a whole number of tokens with length + 16 <= "
                 "16 * capacity_blocks (the appended block must fit the cache it is written into)",
    "attention_heads": "refused: attention_heads: phase grid launches 1, 2, 4, 8 or 16 heads (a power of two of "
                       "threadgroups, one head each; b.f16 holds 16 heads' V caches); more heads, or heads in "
                       "the other phases, are not built",
    "attention_grid": "refused: attention_grid: phase grid takes heads, rows, blocks, head, value, causal, q0, "
                      "first_block, resume, normalize, key_offsets and frozen_head; it attends over a host-held cache (no "
                      "projection, no append: phase step owns those, at head 64), and its shape is QK head 128 "
                      "with value 128 in 16-key blocks",
    "attention_schedule": "refused: attention_schedule: the phases are fused, project and attend (schedules "
                          "attention.fused and attention.proj_cut); choosing between them is the scheduler's (P11), "
                          "not this class's",
}


class AttentionRefused(ValueError):
    """A request outside the attention class, with its refusal code."""
    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(ATTENTION_REFUSALS[code] + ("; " + detail if detail else ""))


def attention_spec(mapping):
    """Admit and normalise one attention program request, or raise AttentionRefused by name.

    Keys: rows (1..16), cache_blocks (prefilled by the host, default 0), new_blocks (projected in this
    program, default 1), causal (bool), q0 (the first query row's position; causal only; default
    16 * cache_blocks, the new tokens' positions), truncate (causal only: key blocks above the
    diagonal for every row are not emitted, recon 155), phase (fused, project or attend)."""
    if not isinstance(mapping, dict):
        raise AttentionRefused("attention_shape", "the request is not a mapping")
    refuse_sharing(mapping)
    if mapping.get("phase") == KV_STEP_PHASE:
        return _kv_step_spec(mapping)
    if mapping.get("phase") == ATTENTION_GRID_PHASE:
        return _grid_spec(mapping)
    if mapping.get("phase") == ATTENTION_GRID_MERGE_PHASE:
        return _grid_merge_spec(mapping)
    for key in ("append_offset_buffer", "runtime_offset", "cache_offset_register", "offset_from_buffer"):
        if key in mapping:
            raise AttentionRefused("attention_runtime_offset", "requested by key %s" % key)
    for key in ("kv_split", "split"):
        if mapping.get(key) not in (None, 0, 1, False):
            raise AttentionRefused("attention_kv_split", "requested by key %s" % key)
    if "schedule" in mapping or "choose" in mapping:
        raise AttentionRefused("attention_schedule", "select a phase by ATTENTION_SCHEDULES name instead")
    known = {"rows", "cache_blocks", "new_blocks", "causal", "q0", "truncate", "phase", "d", "head", "value",
             "block", "simdgroups", "threadgroups", "mask", "key_offsets"}
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise AttentionRefused("attention_shape", "unknown key(s) %s" % ", ".join(unknown))
    if (mapping.get("d", ATTENTION_D), mapping.get("head", ATTENTION_HEAD), mapping.get("value", ATTENTION_VALUE),
            mapping.get("block", ATTENTION_BLOCK)) != (ATTENTION_D, ATTENTION_HEAD, ATTENTION_VALUE, ATTENTION_BLOCK):
        raise AttentionRefused("attention_shape")
    if (mapping.get("simdgroups", 1), mapping.get("threadgroups", 1)) != (1, 1):
        raise AttentionRefused("attention_launch")
    if mapping.get("mask", "causal" if mapping.get("causal") else None) not in (None, "causal"):
        raise AttentionRefused("attention_mask", "mask %r" % (mapping.get("mask"),))
    rows = mapping.get("rows", ATTENTION_MAX_ROWS)
    if not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= ATTENTION_MAX_ROWS:
        raise AttentionRefused("attention_rows", "rows %r" % (rows,))
    cache_blocks, new_blocks = mapping.get("cache_blocks", 0), mapping.get("new_blocks", 1)
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (cache_blocks, new_blocks)):
        raise AttentionRefused("attention_cache", "block counts %r, %r" % (cache_blocks, new_blocks))
    blocks = cache_blocks + new_blocks
    key_offsets = mapping.get("key_offsets", "immediate")
    if key_offsets not in ATTENTION_KEY_OFFSETS:
        raise AttentionRefused("attention_shape", "key_offsets %r" % (key_offsets,))
    if blocks > (ATTENTION_MAX_BLOCKS if key_offsets == "immediate" else ATTENTION_MAX_LOOP_BLOCKS
                 if key_offsets in ATTENTION_LOOP_OFFSETS else ATTENTION_MAX_REGISTER_BLOCKS):
        raise AttentionRefused("attention_blocks", "%d blocks" % blocks)
    if blocks < 1:
        raise AttentionRefused("attention_cache", "no key block")
    phase = mapping.get("phase", "fused")
    if phase not in ATTENTION_PHASES:
        raise AttentionRefused("attention_schedule", "phase %r" % (phase,))
    if phase == "project" and not new_blocks:
        raise AttentionRefused("attention_cache", "a project phase needs new blocks")
    if phase == "project" and key_offsets != "immediate":
        raise AttentionRefused("attention_schedule", "a project phase reads no key block; key_offsets is immediate")
    causal = bool(mapping.get("causal", mapping.get("mask") == "causal"))
    q0 = mapping.get("q0", ATTENTION_BLOCK * cache_blocks if causal else None)
    if not causal and ("q0" in mapping or mapping.get("truncate")):
        raise AttentionRefused("attention_mask", "q0 and truncate belong to the causal mask")
    if causal and (not isinstance(q0, int) or isinstance(q0, bool) or q0 < 0):
        raise AttentionRefused("attention_mask", "q0 %r" % (q0,))
    truncate = bool(mapping.get("truncate", False))
    if key_offsets in ATTENTION_LOOP_OFFSETS and causal and q0 + 1 < ATTENTION_BLOCK * blocks:
        # one body serves every block, so no block may carry a compile-time mask: causal is admitted
        # only when every key is visible to every row (the decode case, q0 at or past the last key)
        raise AttentionRefused("attention_loop", "q0 %d masks keys below %d" % (q0, ATTENTION_BLOCK * blocks))
    # the key blocks the attention visits: all of them, or (truncate) those with a key at or before
    # the last query row's position. Key 0 is visible to every row, so block 0 always is.
    visible = list(range(blocks))
    if causal and truncate:
        visible = [j for j in visible if ATTENTION_BLOCK * j <= q0 + rows - 1]
    out = dict(rows=rows, cache_blocks=cache_blocks, new_blocks=new_blocks, blocks=blocks, causal=causal,
               q0=q0 if causal else None, truncate=truncate, phase=phase, visible=visible)
    if key_offsets != "immediate":
        # recorded only off the default, so every immediate program's normalised spec is unchanged
        out["key_offsets"] = key_offsets
    return out


def _grid_spec(mapping):
    """Admit phase grid (MM 25.135): {"phase": "grid", "heads": 1|2|4|8|16, "rows": 1..16,
    "blocks": 1..8, "head": 128, "value": 128, "causal": bool, "q0": the first query row's absolute
    position (causal), "first_block": the absolute index of this program's block 0 (default 0),
    "resume": bool (the state comes from buffer 3), "normalize": bool (default True),
    "frozen_head": bool (THE FAILING CONTROL: the same program with every head stride 0)}."""
    known = {"phase", "heads", "rows", "blocks", "head", "value", "block", "d", "causal", "mask", "q0",
             "first_block", "resume", "normalize", "frozen_head", "key_offsets", "simdgroups", "threadgroups",
             "kv_split", "frozen_slice", "key_mask", "runtime_q0"}
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise AttentionRefused("attention_grid", "key(s) %s" % ", ".join(unknown))
    if "kv_split" in mapping:
        return _grid_split_spec(mapping)
    for key in ("frozen_slice", "key_mask", "runtime_q0"):
        if key in mapping:
            raise AttentionRefused("attention_grid", "key %s belongs to kv_split" % key)
    for key in ("append_offset_buffer", "runtime_offset", "kv_split", "split", "cache_blocks", "new_blocks"):
        if key in mapping:
            raise AttentionRefused("attention_grid", "key %s" % key)
    if ((mapping.get("head", 128), mapping.get("value", 128)) not in ATTENTION_GRID_SHAPES or
            mapping.get("block", ATTENTION_BLOCK) != ATTENTION_BLOCK or "d" in mapping):
        raise AttentionRefused("attention_shape", "phase grid is QK head 128 and value 128 in 16-key blocks")
    heads = mapping.get("heads", 1)
    if not isinstance(heads, int) or isinstance(heads, bool) or heads not in ATTENTION_GRID_HEADS:
        raise AttentionRefused("attention_heads", "heads %r" % (heads,))
    if mapping.get("simdgroups", 1) != 1 or mapping.get("threadgroups", heads) != heads:
        raise AttentionRefused("attention_launch", "phase grid is one simdgroup per head and one threadgroup per head")
    rows = mapping.get("rows", ATTENTION_MAX_ROWS)
    if not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= ATTENTION_MAX_ROWS:
        raise AttentionRefused("attention_rows", "rows %r" % (rows,))
    blocks = mapping.get("blocks", 1)
    if not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 1:
        raise AttentionRefused("attention_grid", "blocks %r" % (blocks,))
    key_offsets = mapping.get("key_offsets", "immediate")
    if key_offsets not in ATTENTION_KEY_OFFSETS:
        raise AttentionRefused("attention_grid", "key_offsets %r" % (key_offsets,))
    if blocks > (ATTENTION_MAX_BLOCKS if key_offsets == "immediate" else ATTENTION_GRID_CAPACITY):
        raise AttentionRefused("attention_blocks", "phase grid: %d blocks in one dispatch (1..%d immediate, 1..%d "
                               "with key_offsets=register or loop, the layout's capacity; a longer cache is a chain "
                               "of dispatches: resume and normalize)"
                               % (blocks, ATTENTION_MAX_BLOCKS, ATTENTION_GRID_CAPACITY))
    if mapping.get("mask", "causal" if mapping.get("causal", True) else None) not in (None, "causal"):
        raise AttentionRefused("attention_mask", "mask %r" % (mapping.get("mask"),))
    causal = bool(mapping.get("causal", mapping.get("mask", "causal") == "causal"))
    first = mapping.get("first_block", 0)
    if not isinstance(first, int) or isinstance(first, bool) or first < 0:
        raise AttentionRefused("attention_grid", "first_block %r" % (first,))
    q0 = mapping.get("q0")
    if causal:
        if not isinstance(q0, int) or isinstance(q0, bool) or q0 < 0:
            raise AttentionRefused("attention_mask", "phase grid's causal mask needs q0, a position >= 0")
        # key 0 of the whole cache is visible to every row; in a chain, a dispatch whose every key is past
        # the last query row would add exactly nothing (recon 155's no-op) - admitted, but pointless, so
        # it is refused: its keys belong to no query
        if ATTENTION_BLOCK * first > q0 + rows - 1:
            raise AttentionRefused("attention_mask", "every key of blocks %d.. is past query %d"
                                   % (first, q0 + rows - 1))
    elif q0 is not None:
        raise AttentionRefused("attention_mask", "q0 belongs to the causal mask")
    for key in ("resume", "normalize", "frozen_head"):
        if not isinstance(mapping.get(key, key == "normalize"), bool):
            raise AttentionRefused("attention_grid", "%s is a bool" % key)
    out = dict(rows=rows, cache_blocks=None, new_blocks=0, blocks=blocks, causal=causal,
               q0=q0 if causal else None, truncate=False, phase=ATTENTION_GRID_PHASE,
               visible=list(range(blocks)), heads=heads, head=128, value=128, first_block=first,
               resume=bool(mapping.get("resume", False)), normalize=bool(mapping.get("normalize", True)),
               frozen_head=bool(mapping.get("frozen_head", False)), key_offsets=key_offsets)
    if key_offsets in ATTENTION_LOOP_OFFSETS:
        # THE COUNTED KEY-BLOCK LOOP IN PHASE GRID (MM 25.135.4): the leading blocks that no row masks run
        # as ONE body in a counted loop of loop_blocks trips; the blocks the causal mask reaches (at one
        # decode row, the last, partly filled block) are PEELED after the loop as straight-line bodies that
        # read the same K/V registers, each with its compile-time mask. No chain: the loop's first trip
        # starts from the written state m = -FLT_MAX, l = 0, O = 0, so a resumed state is refused.
        if out["resume"]:
            raise AttentionRefused("attention_loop", "phase grid's loop starts from its own written state; "
                                   "a resumed chain dispatch runs key_offsets=register")
        loop_blocks = blocks
        if causal:
            loop_blocks = 0
            while (loop_blocks < blocks and
                   q0 - ATTENTION_BLOCK * (first + loop_blocks) >= ATTENTION_BLOCK - 1):
                loop_blocks += 1
        if loop_blocks < 1:
            raise AttentionRefused("attention_loop", "phase grid: the causal mask reaches block %d, so no block "
                                   "is unmasked for every row and there is nothing to loop" % first)
        out["loop_blocks"] = loop_blocks
    return out


def _grid_split_spec(mapping):
    """Phase grid with kv_split S (MM 25.114.6): the loop split over S threadgroups per head. Admitted
    only as a decode-shaped causal program on the loop: key_offsets "loop" (or "loop_frozen"), not
    resumed, not normalised (each slice leaves its partial), every padded key past the last query row.
    Controls: frozen_slice (every slice READS slice 0's blocks; its own slot keeps its stride) and
    key_mask=False (the same program with the per-trip mask left out)."""
    S = mapping["kv_split"]
    if not isinstance(S, int) or isinstance(S, bool) or S not in ATTENTION_GRID_SPLITS:
        raise AttentionRefused("attention_kv_split", "kv_split %r (phase grid splits 1, 2, 4 or 8 ways)" % (S,))
    base = {k: v for k, v in mapping.items() if k not in ("kv_split", "frozen_slice", "key_mask", "blocks", "runtime_q0")}
    runtime_q0 = mapping.get("runtime_q0", False)
    if not isinstance(runtime_q0, bool):
        raise AttentionRefused("attention_grid", "runtime_q0 is a bool")
    if runtime_q0:
        # THE RUNTIME LENGTH (MM 25.138.1): the query position is the uint32 word at ATTENTION_GRID_LENGTH_BYTE of
        # buffer 3, loaded each trip, so ONE program serves every length its blocks hold; q0 here is only the
        # largest position admitted (the host checks each length against it)
        base["q0"] = ATTENTION_BLOCK * mapping.get("blocks", 1) - base.get("rows", ATTENTION_MAX_ROWS)
    if base.get("key_offsets", "immediate") not in ATTENTION_LOOP_OFFSETS:
        raise AttentionRefused("attention_kv_split", "kv_split runs on the counted key-block loop (key_offsets=loop)")
    if base.get("normalize", False) or base.get("resume", False):
        raise AttentionRefused("attention_kv_split", "each slice leaves its un-normalised partial for the merge; "
                               "normalize and resume are the merge's")
    if not base.get("causal", True) or base.get("mask", "causal") != "causal":
        raise AttentionRefused("attention_kv_split", "the per-trip key mask is the causal rule (q0 + row)")
    for key in ("frozen_slice", "key_mask"):
        if not isinstance(mapping.get(key, key == "key_mask"), bool):
            raise AttentionRefused("attention_grid", "%s is a bool" % key)
    blocks = mapping.get("blocks", 1)
    if not isinstance(blocks, int) or isinstance(blocks, bool) or not 1 <= blocks <= ATTENTION_GRID_CAPACITY:
        raise AttentionRefused("attention_blocks", "kv_split: %r blocks (1..%d, the layout)" % (blocks, ATTENTION_GRID_CAPACITY))
    # admit the unsplit request (heads, rows, q0, ...) through the ordinary rules, on one block so the
    # loop's own peel rule does not apply: the split masks inside the loop
    out = _grid_spec(dict(base, blocks=1, normalize=False))
    padded = -(-blocks // S) * S
    bps = padded // S
    heads, rows, q0, first = out["heads"], out["rows"], out["q0"], out["first_block"]
    if heads * S > 256:
        raise AttentionRefused("attention_launch", "%d heads x %d slices: at most 256 threadgroups" % (heads, S))
    if first + padded > ATTENTION_SPLIT_MAX_PADDED:
        raise AttentionRefused("attention_kv_split", "%d padded blocks past the %d a head's K/V stride holds"
                               % (first + padded, ATTENTION_SPLIT_MAX_PADDED))
    if runtime_q0 and first:
        raise AttentionRefused("attention_kv_split", "a runtime length is a program from block 0")
    if q0 + rows - 1 >= ATTENTION_BLOCK * (first + blocks):
        raise AttentionRefused("attention_kv_split", "query %d sees a padded key: the padding is masked only "
                               "past the last query row" % (q0 + rows - 1))
    # the mask compares col + bias > w + bias + row - i with w = q0 - 16 (first + block); both sides >= 0
    bias = max(0, ATTENTION_BLOCK * (first + padded) - (0 if runtime_q0 else q0)) + 16
    out.update(blocks=blocks, visible=list(range(blocks)), kv_split=S, padded_blocks=padded,
               blocks_per_slice=bps, loop_blocks=bps, mask_bias=bias, normalize=False,
               frozen_slice=bool(mapping.get("frozen_slice", False)), key_mask=bool(mapping.get("key_mask", True)))
    if runtime_q0:
        out["runtime_q0"] = True
    return out


def attention_split_slots(a):
    """The KV split's slot geometry (bytes): threadgroup t's partial starts at base + t * stride."""
    base = ATTENTION_GRID_STRIDE["C"] * a["heads"]
    return dict(base=base, stride=ATTENTION_SPLIT_SLOT, count=a["heads"] * a["kv_split"],
                O_tiles=[ATTENTION_SPLIT_C["O"] + 2048 * s for s in range(a["value"] // 16)],
                S=ATTENTION_SPLIT_C["S"], Mst=ATTENTION_SPLIT_C["M"], L=ATTENTION_SPLIT_C["L"],
                W=ATTENTION_SPLIT_C["W"])


def attention_grid_layout(a):
    """Phase grid's layout (bytes, per head, ATTENTION_GRID_*) and transport extents."""
    tr = ATTENTION_GRID_TRANSPORT
    if a.get("kv_split"):
        # the slots follow the heads' regions; M is the smallest multiple of 16 x threadgroups rows (the
        # worker's rule: whole tiles per threadgroup) of N fp32 words that holds them
        slots = attention_split_slots(a)
        c_bytes = slots["base"] + slots["count"] * slots["stride"]
        unit = 16 * slots["count"]
        rows = -(-c_bytes // (4 * tr["N"]))
        out = attention_grid_layout(dict(a, kv_split=None))
        out.update(M=-(-rows // unit) * unit, c_bytes=c_bytes, slots=slots)
        return out
    return dict(M=tr["rows_per_head"] * a["heads"], N=tr["N"], K=tr["K"], c_bytes=ATTENTION_GRID_STRIDE["C"] * a["heads"],
                stride=dict(ATTENTION_GRID_STRIDE), capacity=ATTENTION_GRID_CAPACITY,
                O_tiles=[ATTENTION_GRID_C["O"] + 2048 * s for s in range(a["value"] // 16)],
                KC=ATTENTION_GRID_C["KC"], S=ATTENTION_GRID_C["S"], O=ATTENTION_GRID_C["O"],
                Mst=ATTENTION_GRID_C["M"], L=ATTENTION_GRID_C["L"],
                kblock=ATTENTION_BLOCK * a["head"] * 2, vblock=ATTENTION_BLOCK * a["value"] * 2,
                vslice=ATTENTION_BLOCK * 16 * 2)


def _grid_merge_spec(mapping):
    """Admit phase grid_merge (MM 25.135.5): {"phase": "grid_merge", "heads": 1|2|4|8|16, "rows": 1..16,
    "kv_split": 2|4|8, "allow_value_change": True, "normalize": bool (default True)}. The merge changes the
    values against the unsplit stream (a different association), so it needs the opt-in, as P9's split."""
    known = {"phase", "heads", "rows", "kv_split", "allow_value_change", "normalize", "head", "value",
             "simdgroups", "threadgroups", "tile_groups"}
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise AttentionRefused("attention_grid", "phase grid_merge: key(s) %s" % ", ".join(unknown))
    # THE TILED MERGE (MM 25.132.7): tile_groups 8 runs one threadgroup per (head, 16-wide O tile); each recomputes
    # the head's M, f_s and l in the same order, so every word's arithmetic is the per-head merge's
    tiles = mapping.get("tile_groups", 1)
    if tiles not in (1, 8) or isinstance(tiles, bool):
        raise AttentionRefused("attention_grid", "phase grid_merge: tile_groups is 1 or 8 (one per O tile)")
    if (mapping.get("head", 128), mapping.get("value", 128)) not in ATTENTION_GRID_SHAPES:
        raise AttentionRefused("attention_shape", "phase grid_merge is QK head 128 and value 128")
    heads = mapping.get("heads", 1)
    if not isinstance(heads, int) or isinstance(heads, bool) or heads not in ATTENTION_GRID_HEADS:
        raise AttentionRefused("attention_heads", "heads %r" % (heads,))
    if mapping.get("simdgroups", 1) != 1 or mapping.get("threadgroups", heads * tiles) != heads * tiles:
        raise AttentionRefused("attention_launch", "phase grid_merge is one simdgroup per threadgroup, heads x "
                               "tile_groups threadgroups")
    rows = mapping.get("rows", ATTENTION_MAX_ROWS)
    if not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= ATTENTION_MAX_ROWS:
        raise AttentionRefused("attention_rows", "rows %r" % (rows,))
    split = mapping.get("kv_split")
    if split not in ATTENTION_KV_SPLITS or isinstance(split, bool):
        raise AttentionRefused("attention_kv_split", "kv_split %r (the grid merge is built for %s ranges)"
                               % (split, ATTENTION_KV_SPLITS))
    if mapping.get("allow_value_change") is not True:
        raise AttentionRefused("attention_kv_split", "a %d-way split changes the values; opt in with "
                               "allow_value_change=True" % split)
    if not isinstance(mapping.get("normalize", True), bool):
        raise AttentionRefused("attention_grid", "normalize is a bool")
    if tiles > 1 and kv_split_transport_rows(kv_split_c_bytes(heads, split), heads * split) % (16 * heads * tiles):
        raise AttentionRefused("attention_launch", "phase grid_merge tile_groups %d: the split's transport is not "
                               "a multiple of 16 x heads x tile_groups rows" % tiles)
    out = dict(phase=ATTENTION_GRID_MERGE_PHASE, heads=heads, rows=rows, kv_split=split, head=128, value=128,
               allow_value_change=True, normalize=bool(mapping.get("normalize", True)))
    if tiles > 1:
        out["tile_groups"] = tiles
    return out


def attention_grid_merge_layout(a):
    """Phase grid_merge's layout: the grid's per-head regions and offsets (attention_grid_layout), the heads x S
    scratch slots after them, and the transport that covers both (N 256 and K 4096 as the grid). M is the
    SPLIT's M, kv_split_transport_rows at heads x S threadgroups (3,072 / 4,096 / 6,144 rows at 16 heads x
    S = 2 / 4 / 8): a multiple of 16 x heads too, so it is a legal transport for the merge's own launch, and
    the split's buffer 3 is the merge's byte for byte (the worker moves exactly M x N x 4 bytes, so the
    merge's smallest M, 2,816 / 3,584 / 4,864, would not take the split's output file as it is)."""
    lay = attention_grid_layout(a)
    c_bytes = kv_split_c_bytes(a["heads"], a["kv_split"])
    M = kv_split_transport_rows(c_bytes, a["heads"] * a["kv_split"], lay["N"])
    lay.update(M=M, c_bytes=c_bytes, kv_split=a["kv_split"], slot_bytes=KV_SLOT_BYTES, slot=dict(KV_SLOT),
               slots_base=ATTENTION_GRID_STRIDE["C"] * a["heads"])
    return lay


def attention_layout(a):
    """Byte offsets and transport extents (M, N, K) of an admitted attention spec."""
    if a["phase"] == KV_STEP_PHASE:
        return kv_step_layout(a)
    if a["phase"] == ATTENTION_GRID_MERGE_PHASE:
        return attention_grid_merge_layout(a)
    if a["phase"] == ATTENTION_GRID_PHASE:
        return attention_grid_layout(a)
    kc = ATTENTION_C["KC"]
    vc = kc + a["blocks"] * ATTENTION_BLOCK * ATTENTION_HEAD * 2
    c_bytes = vc + a["blocks"] * ATTENTION_BLOCK * ATTENTION_VALUE * 2
    width = ATTENTION_D + ATTENTION_VALUE                       # 80: B is Wk | Wv, 64 x 80 halves
    rows_c = -(-c_bytes // (4 * width))
    rows_a = 32 + ATTENTION_BLOCK * a["new_blocks"]
    M = -(-max(rows_c, rows_a) // 16) * 16
    return dict(M=M, N=width, K=ATTENTION_D, KC=kc, VC=vc, c_bytes=c_bytes)


def attention_schedules(mapping):
    """Both schedules of one attention workload, for a scheduler to select BY NAME (P11 selects):
    {"attention.fused": [spec of its one program], "attention.proj_cut": [project spec, attend spec]}. The
    workload is `mapping` without a phase; each program spec is attention_spec-admitted."""
    base = {k: v for k, v in dict(mapping).items() if k != "phase"}
    out = {}
    for name, phases in ATTENTION_SCHEDULES.items():
        programs = []
        for phase in phases:
            if phase == "project" and not base.get("new_blocks", 1):
                continue
            programs.append(attention_spec(dict(base, phase=phase)))
        out[name] = programs
    return out


# THE STATEFUL ATTENTION RUNTIME (production row P9, docs/g17-tensorops-machine-model.md 25.131).
# Phase "step" of the attention class is ONE program per (capacity, rows, kv_split) that serves every
# cache length: the length is a RUNTIME value, a uint32 uniform word in buffer 1 that the host writes
# per dispatch, not a compile-time block offset. The contract, in three parts:
#   OWNERSHIP. A KVCache owns a region of buffer 3 (kv_step_layout): the K cache (capacity x 16 keys x
#     64 halves, keys x head) and the V cache (keys x 16 halves), P7's operand layout, so the QK
#     (transB) and PV bodies read it with no relayout. It owns the length (tokens) on the host. Buffer 3
#     persists across dispatches; S, O, M and L (and a split's O1, M1, L1) are the dispatch's scratch
#     and output, overwritten by the next step. Buffer 1 (Q, the 16 new tokens X, the length word) and
#     buffer 2 (Wk | Wv) are per-dispatch inputs.
#   OFFSETS. The step projects its 16 new tokens into a fixed staging region (compile-time offsets:
#     tensor bodies take no register offset, 25.131), then copies the staged words into cache rows
#     [length, length + 16) with scalar stores whose index is computed from the loaded length. The
#     attention visits every capacity block at compile-time offsets and masks by the RUNTIME causal
#     threshold length + row - key: keys at or past length + 16 are masked for every row, so blocks past
#     the length contribute exactly nothing (alpha = exp2(0) = 1, P = 0, recon 155's no-op) provided
#     their halves are finite - which is why KVCache zero-fills the cache and re-zeros on truncate.
#   SYNCHRONISATION. Steps that share a KVCache are encoded on ONE MTLCommandQueue in step order; the
#     buffer is an ordinary (hazard-tracked) MTLBuffer. Command-buffer commit order then orders step n's
#     append before step n+1's reads, so no completion wait is needed between steps (measured, 25.131,
#     back to back and as two encoders in one command buffer). The host needs a wait only to READ a
#     step's O before the next step overwrites it, and the length never needs the GPU: the host
#     computes every step's length in advance.
KV_STEP_PHASE = "step"
KV_LENGTH_BYTE = ATTENTION_A["X"] + ATTENTION_BLOCK * ATTENTION_D * 2      # 6144: the uniform in buffer 1
# the runtime mask compares col + BIAS > length + BIAS + row - key0 - i, so both sides are non-negative
# for every admitted capacity (key0 <= 112) and the comparison's signedness cannot matter
KV_MASK_BIAS = 128
KV_SPLITS = (1, 2)
KV_SYNC_CONTRACT = MappingProxyType({
    "queue": "one MTLCommandQueue per KVCache; steps encoded in step order",
    "order": "command-buffer commit order (or encoder order inside one command buffer) orders step n's "
             "append before step n+1's reads of the cache; the buffer is hazard-tracked (the default)",
    "completion_wait_between_steps": False,
    "wait_needed_for": "reading a step's O (scratch, overwritten by the next step) on the host",
    "length": "host-owned; written per dispatch into the uniform word; never read back from the GPU",
    "not_admitted": "concurrent steps on one KVCache from two queues, an untracked buffer, or a step "
                    "whose length is not the previous step's length + 16",
})


def _kv_step_spec(mapping):
    """Admit phase step (P9): {"phase": "step", "capacity_blocks": 1..8, "rows": 1..16, "kv_split": 1|2,
    "allow_value_change": bool}. The normalised spec's q0 is None: the length is read at run time."""
    known = {"phase", "capacity_blocks", "rows", "kv_split", "allow_value_change", "causal", "mask",
             "d", "head", "value", "block", "simdgroups", "threadgroups"}
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise AttentionRefused("kv_step", "key(s) %s" % ", ".join(unknown))
    if (mapping.get("d", ATTENTION_D), mapping.get("head", ATTENTION_HEAD), mapping.get("value", ATTENTION_VALUE),
            mapping.get("block", ATTENTION_BLOCK)) != (ATTENTION_D, ATTENTION_HEAD, ATTENTION_VALUE, ATTENTION_BLOCK):
        raise AttentionRefused("attention_shape")
    if (mapping.get("simdgroups", 1), mapping.get("threadgroups", 1)) != (1, 1):
        raise AttentionRefused("attention_launch")
    if mapping.get("causal", True) is not True or mapping.get("mask", "causal") != "causal":
        raise AttentionRefused("kv_step", "the step is causal: its queries are the appended tokens")
    cap = mapping.get("capacity_blocks")
    if not isinstance(cap, int) or isinstance(cap, bool) or not 1 <= cap <= ATTENTION_MAX_BLOCKS:
        raise AttentionRefused("kv_capacity", "capacity_blocks %r" % (cap,))
    rows = mapping.get("rows", ATTENTION_MAX_ROWS)
    if not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= ATTENTION_MAX_ROWS:
        raise AttentionRefused("attention_rows", "rows %r" % (rows,))
    split = mapping.get("kv_split", 1)
    if split not in KV_SPLITS or isinstance(split, bool):
        raise AttentionRefused("attention_kv_split", "kv_split %r (the merge is built for 2 ranges)" % (split,))
    if split == 2 and mapping.get("allow_value_change") is not True:
        raise AttentionRefused("attention_kv_split", "kv_split 2 changes the values; opt in with "
                               "allow_value_change=True")
    if split == 2 and cap < 2:
        raise AttentionRefused("attention_kv_split", "a split needs two capacity blocks")
    half = -(-cap // 2)
    ranges = [list(range(cap))] if split == 1 else [list(range(half)), list(range(half, cap))]
    return dict(rows=rows, cache_blocks=None, new_blocks=1, blocks=cap, causal=True, q0=None, truncate=False,
                phase=KV_STEP_PHASE, visible=list(range(cap)), capacity_blocks=cap, kv_split=split,
                ranges=ranges, runtime_length=True)


def kv_step_layout(a):
    """Buffer layout of an admitted step (bytes). Buffer 3: S, O, M, L as the attention class; the K
    cache at KC and the V cache at VC (capacity blocks each); the staging slots SK and SV the
    projections write; a split's second partial state O1, M1, L1. Buffer 1: Q, X, the length word."""
    cap = a["capacity_blocks"]
    kc = ATTENTION_C["KC"]
    vc = kc + cap * ATTENTION_BLOCK * ATTENTION_HEAD * 2
    sk = vc + cap * ATTENTION_BLOCK * ATTENTION_VALUE * 2
    sv = sk + ATTENTION_BLOCK * ATTENTION_HEAD * 2
    end = sv + ATTENTION_BLOCK * ATTENTION_VALUE * 2
    split = {}
    if a["kv_split"] == 2:
        for name in ("O1", "M1", "L1"):
            split[name] = end
            end += 32 * 16 * 4
    width = ATTENTION_D + ATTENTION_VALUE
    rows_c = -(-end // (4 * width))
    rows_a = -(-(KV_LENGTH_BYTE + 4) // (2 * ATTENTION_D))
    M = -(-max(rows_c, rows_a) // 16) * 16
    return dict(M=M, N=width, K=ATTENTION_D, KC=kc, VC=vc, SK=sk, SV=sv, c_bytes=end,
                LEN=KV_LENGTH_BYTE, **split)


class KVCache:
    """The host side of one stateful attention cache (P9). It owns: the buffer-3 region (its byte
    ranges, `region`), the capacity, the current length in tokens, and the layout the step writes. It
    produces the per-dispatch uniform (`length_word`), advances the length after encoding a step, and
    names the bytes a truncation must re-zero. It holds no Metal object: the MTLBuffer is the caller's,
    and KV_SYNC_CONTRACT is how the caller must order dispatches on it."""

    def __init__(self, capacity_blocks, rows=ATTENTION_MAX_ROWS, kv_split=1, allow_value_change=False, length=0):
        self.spec = attention_spec({"phase": KV_STEP_PHASE, "capacity_blocks": capacity_blocks, "rows": rows,
                                    "kv_split": kv_split, "allow_value_change": allow_value_change})
        self.layout = kv_step_layout(self.spec)
        self.capacity = capacity_blocks * ATTENTION_BLOCK
        self.length = 0
        self.set_length(length)

    @property
    def region(self):
        """The buffer-3 byte ranges this cache owns, [start, end), and their element layout."""
        lay = self.layout
        return {"buffer": 3,
                "k": (lay["KC"], lay["VC"], "keys x head, float16 row-major (64 halves per key)"),
                "v": (lay["VC"], lay["SK"], "keys x value, float16 row-major (16 halves per key)"),
                "staging": (lay["SK"], lay["SV"] + ATTENTION_BLOCK * ATTENTION_VALUE * 2,
                            "the step's K then V projection, copied to rows [length, length + 16)")}

    def set_length(self, length):
        """The length is 0..capacity tokens; a full cache is a state, the step after it is refused."""
        if not isinstance(length, int) or isinstance(length, bool) or not 0 <= length <= self.capacity:
            raise AttentionRefused("kv_length", "length %r, capacity %d tokens" % (length, self.capacity))
        self.length = length

    def length_word(self):
        """The uniform the next step reads: the current length as a little-endian uint32, written at
        buffer-1 byte KV_LENGTH_BYTE. Refused when the step's block would not fit."""
        if self.length + ATTENTION_BLOCK > self.capacity:
            raise AttentionRefused("kv_length", "a step at length %d needs %d tokens, capacity %d"
                                   % (self.length, self.length + ATTENTION_BLOCK, self.capacity))
        return int(self.length).to_bytes(4, "little")

    def advance(self):
        """After a step is ENCODED (not completed): the next step appends after this one's block."""
        self.length_word()
        self.set_length(self.length + ATTENTION_BLOCK)
        return self.length

    def zero_fill(self, c_bytes):
        """Zero the K and V cache in a buffer-3 image (a bytearray): unwritten rows must hold finite
        halves, because masked keys still pass through the PV MMA with P = 0 (0 x inf is NaN)."""
        lay = self.layout
        c_bytes[lay["KC"]:lay["SK"]] = bytes(lay["SK"] - lay["KC"])
        return c_bytes

    def truncate(self, length):
        """Rewind to `length` tokens and return the byte ranges of buffer 3 the host must re-zero
        before the next step (rows [length, old length) of K and V)."""
        old = self.length
        if not isinstance(length, int) or isinstance(length, bool) or not 0 <= length <= old:
            raise AttentionRefused("kv_length", "truncate to %r from %d" % (length, old))
        self.length = length
        lay = self.layout
        return [(lay["KC"] + 128 * length, lay["KC"] + 128 * old), (lay["VC"] + 32 * length, lay["VC"] + 32 * old)]


def manifest_format(kind):
    """The pipeline manifest version this application kind is contracted at."""
    if kind == TENSOR_KIND:
        return "g17-common-pipeline-v3"
    if kind == REQUANT_KIND:
        return "g17-common-pipeline-v4"
    return "g17-common-pipeline-v2" if kind in FOUR_BUFFER_KINDS else "g17-common-pipeline-v1"


class Launch(BaseModel):
    model_config = CONFIG
    bounds_checked: bool
    exact_grid_required: bool


class CompilerABI(BaseModel):
    """ABI v3 and retained v2 contracts, validated before any loader operation."""
    model_config = CONFIG
    abi_version: Annotated[int, Field(ge=2, le=5)]
    system_registers: tuple[int, ...] | None = None
    arch_flag: bool
    bindings: tuple[Binding, ...]
    entry: Annotated[int, Field(ge=64, le=64)]
    forms: tuple[tuple[int, int], ...]
    has_stores: bool
    launch: Launch
    pk_extra: tuple[int, ...]
    pk_values: dict[int, int]
    profile: str | None
    # The measured requantization class carries a named constant-program identity even though it
    # is not the generic resources/preloads ABI v7 route.
    constant_program_sha256: str | None = None
    # ABI v4/v5 carry these additive facts only for classes that need them. Retained v1-v3
    # manifests omit both keys; tensor_gemm requires the explicit empty pool and execution block.
    constant_pool: tuple[Annotated[int, Field(ge=0, le=255)], ...] | None = None
    execution: dict | None = None
    # ADDITIVE, gemm_generic only (Set A item 12): the compiler's explicit-imageblock statement,
    # {"layout": "explicit", "element_bytes": n}. The linker declares the tile from it; absent means
    # no declaration (the measured control: the pipeline allocates nothing and reads return 0).
    imageblock: dict | None = None
    # The compiler can carry a measured scalar requantization marker, but this released worker
    # has no one-manifest representation for its required two-dispatch boundary yet. Keep the
    # marker visible so the refusal is named rather than falling through the ordinary binding
    # checks.
    requantization: dict | None = None
    # ADDITIVE AT v3, OPTIONAL FOR RETAINED CONTRACTS. The compiler counts only
    # instructions in _agc.main; the runtime carries the fact through to the
    # image/linker boundary and must not derive it by decoding bytes it does
    # not own. Unknown fields remain forbidden, so this accepts only the
    # explicitly agreed field rather than opening the schema generally.
    main_instruction_count: Annotated[int, Field(ge=1)] | None = None
    # ADDITIVE AT v3, AND OPTIONAL ON PURPOSE. The compiler delivers the register count -
    # the highest 32-bit register index named in _agc.main plus one, excluding the constant
    # program - because reading it off the bytes would mean this layer resolving operand kinds
    # and register naming, which is instruction semantics it does not own. Retained v3
    # contracts predate the field and must still load, so absence is not a schema error; it is
    # refused at the point of USE, where a metadata class would otherwise have to invent the
    # value. Defaulting from a decode is what must never happen: the two disagreeing is a
    # finding, and a fallback would convert that finding into silence.
    register_count: Annotated[int, Field(ge=1)] | None = None
    prologue: bytes
    uses_threadgroup: bool
    writes_buffer: bool
    writes_texture: bool

    @model_validator(mode="before")
    @classmethod
    def no_cooperative_sharing(cls, value):
        refuse_tensor_threadgroup_abi(value)      # MM P12: named, before the generic semantic check
        return value

    @model_validator(mode="wrap")
    @classmethod
    def allow_measured_byte_destination(cls, value, handler):
        """Reconstruct the marked physical word bindings before runtime validation."""
        marker = value.get("requantization") if isinstance(value, dict) else None
        destination = marker.get("destination_binding") if isinstance(marker, dict) else None
        if isinstance(destination, int):
            normalized = dict(value)
            raw_bindings = normalized.get("bindings", ())
            with allow_requantized_binding({destination}):
                normalized["bindings"] = tuple(
                    item if isinstance(item, Binding) else Binding(**item)
                    for item in raw_bindings)
                return handler(normalized)
        return handler(value)

    @model_validator(mode="after")
    def supported(self):
        if self.requantization is not None:
            from . import requantpreload
            marker = self.requantization
            if (self.abi_version != 3 or self.system_registers != (160,) or not self.arch_flag or
                    self.uses_threadgroup or not self.writes_buffer or not self.has_stores or
                    self.writes_texture or self.launch.bounds_checked or
                    not self.launch.exact_grid_required or
                    self.prologue != requantpreload.PROLOGUE):
                raise ValueError("unsupported requantization scalar ABI facts")
            if self.constant_program_sha256 != requantpreload.PROLOGUE_SHA256:
                raise ValueError("unsupported requantization constant-program identity")
            if self.pk_extra != (15, 16) or self.pk_values != {15: 1, 16: 1}:
                raise ValueError("unsupported requantization scalar metadata fields")
            if (marker.get("kind") not in ("int32_to_int8", "int32_to_uint8") or
                    marker.get("metadata_class") != "scalar_476" or
                    marker.get("dispatch_boundary") != "required" or
                    marker.get("elements") != 256 or marker.get("groups") != 8 or
                    marker.get("rounding") != "round_half_to_even" or
                    marker.get("scale_binding") != 1 or marker.get("source_binding") != 0 or
                    marker.get("destination_binding") != 2):
                raise ValueError("unsupported requantization scalar boundary")
            if (marker.get("scale_storage") != "uint32_bits" or
                    marker.get("output_storage") != "int32_word"):
                raise ValueError("unsupported requantization physical storage")
            # The retained Apple probe obtains scale_bits[0] through its constant program.  The
            # ordinary IR stage selects that exact measured class; arbitrary indexed scale loads
            # remain outside the runtime class.
            if marker.get("scale_addressing") != "constant_program_preload":
                raise ValueError("requantization scale preload route is not measured")
            if (marker.get("saturation") not in ("signed_int8", "unsigned_uint8") or
                    marker.get("scale") not in ("1/512", "1/16") or
                    (marker.get("saturation") == "unsigned_uint8" and
                     marker.get("scale") != "1/16") or
                    marker.get("kind") != ("int32_to_uint8" if marker.get("saturation") == "unsigned_uint8"
                                            else "int32_to_int8")):
                raise ValueError("unsupported requantization scale or saturation")
            if (tuple((b.index, b.offset, b.element_type, b.written) for b in self.bindings) !=
                    ((0, 0, "uint", False), (1, 2, "uint", False), (2, 4, "uint", True))):
                raise ValueError("unsupported requantization scalar binding contract")
            if not self.forms or self.forms != tuple(sorted(set(self.forms))):
                raise ValueError("invalid requantization compiler form declarations")
            return self
        # ABI v5 is the measured tensor launch contract. It is deliberately a separate branch:
        # the older common programs require ARCH=true and ABI v2/v3 thread-id sets, while tensor
        # images use ARCH=false, SR130 (optionally paired with SR156 for a scalar epilogue), and
        # the three-user-buffer descriptor map. Keeping the branch here makes a tensor manifest
        # fail before the native worker creates a Metal pipeline.
        if self.abi_version == 5:
            # (130, 133, 156): a simdgroup-split body with the SR156 epilogue - the tensormetadata
            # set measured in #145 (slot 29 [0, 52, 53]); ImageContract admits it for gemm_generic only.
            # (130, 164, 165): a tensor body with an explicit imageblock, whose coordinate reads
            # SR_LOCAL_X/Y - measured on Apple's own tensor+imageblock compile (Set A item 12).
            if self.system_registers not in ((130,), (130, 156), (130, 133, 156), (130, 164, 165)):
                raise ValueError("ABI v5 requires measured tensor system-register set SR130, SR130+SR156, "
                                 "SR130+SR_SIMD_GRP+SR156 or SR130+SR_LOCAL_X/Y")
            if self.imageblock is not None and (
                    set(self.imageblock) != {"layout", "element_bytes"} or self.imageblock["layout"] != "explicit"
                    or type(self.imageblock["element_bytes"]) is not int
                    or not 0 < self.imageblock["element_bytes"] < 1 << 16
                    or self.system_registers != (130, 164, 165)):
                raise ValueError("an imageblock statement is {layout: explicit, element_bytes: n} on the "
                                 "SR130+SR_LOCAL_X/Y tensor set")
            if (self.arch_flag or self.uses_threadgroup or not self.writes_buffer or
                    not self.has_stores or self.writes_texture or self.launch.bounds_checked or
                    not self.launch.exact_grid_required):
                raise ValueError("unsupported tensor runtime semantic contract")
            if self.prologue != bytes.fromhex("0e000000") + bytes.fromhex("0600")*30:
                raise ValueError("unsupported tensor constant program")
            if self.pk_extra != (15, 16) or self.pk_values != {15: 1, 16: 1}:
                raise ValueError("tensor metadata fields disagree with device writes")
            if self.constant_pool != () or self.execution != {"simd_width": 32, "tensor": True}:
                raise ValueError("tensor runtime requires an explicit empty pool and execution ABI")
            expected = (1, 2, 3)
            if tuple(b.index for b in self.bindings) != expected:
                raise ValueError("tensor runtime requires public bindings 1, 2 and 3")
            if any(b.offset != 2*i for i, b in enumerate(self.bindings)):
                raise ValueError("tensor runtime requires dense descriptor offsets 0, 2 and 4")
            if any(b.element_type not in ("half", "bfloat", "float", "uchar") or
                   b.written != (i == 2) for i, b in enumerate(self.bindings)):
                raise ValueError("unsupported tensor runtime binding contract")
            if self.bindings[2].element_type != "float" or self.bindings[2].element_bytes != 4:
                raise ValueError("tensor runtime output must be a writable float buffer")
            if not self.forms or self.forms != tuple(sorted(set(self.forms))):
                raise ValueError("invalid tensor compiler form declarations")
            return self
        if self.abi_version == 3 and self.system_registers not in ((160,), (160, 161)):
            raise ValueError("ABI v3 requires the supported thread-id system-register set")
        if self.abi_version == 2 and self.system_registers is not None:
            raise ValueError("system-register declarations require ABI v3")
        if (not self.arch_flag or self.uses_threadgroup or not self.writes_buffer or
                not self.has_stores or self.writes_texture or self.launch.bounds_checked or
                not self.launch.exact_grid_required):
            raise ValueError("unsupported common-runtime semantic contract")
        if self.prologue != bytes.fromhex("0e000000") + bytes.fromhex("0600")*30:
            raise ValueError("unsupported constant program")
        if self.pk_extra != (15, 16) or self.pk_values != {15: 1, 16: 1}:
            raise ValueError("semantic metadata fields disagree with device writes")
        indices = [b.index for b in self.bindings]
        if indices != sorted(set(indices)) or any(
                b.offset != 2*i or b.element_type not in ("half", "float") or
                b.written != (i == len(self.bindings)-1) for i, b in enumerate(self.bindings)):
            raise ValueError("unsupported common-runtime binding contract")
        if not self.forms or self.forms != tuple(sorted(set(self.forms))) or any(
                type(op) is not int or type(n) is not int or not 0 <= op <= 65535 or
                not 2 <= n <= 32 or n % 2 for op, n in self.forms):
            raise ValueError("invalid compiler form declarations")
        return self


def compiler_abi_from_plain(value):
    """Turn the compiler's JSON-shaped ABI into the strict runtime model.

    ``G17Program.abi_plain`` intentionally returns mutable JSON values.  The runtime model is
    strict and frozen, so authoring helpers must cross this boundary explicitly rather than rely
    on Pydantic coercing lists into tuples or dictionaries into its frozen dataclasses.  The
    requant marker is reconstructed with the same strict physical word bindings as every other
    contract.
    """
    if isinstance(value, CompilerABI):
        return value
    if not isinstance(value, dict):
        raise TypeError("compiler ABI must be a dictionary")
    data = dict(value)
    marker = data.get("requantization")
    destination = marker.get("destination_binding") if isinstance(marker, dict) else None
    raw_bindings = data.get("bindings", ())
    if isinstance(destination, int):
        with allow_requantized_binding({destination}):
            data["bindings"] = tuple(
                item if isinstance(item, Binding) else Binding(**item)
                for item in raw_bindings)
    else:
        data["bindings"] = tuple(
            item if isinstance(item, Binding) else Binding(**item)
            for item in raw_bindings)
    data["system_registers"] = (None if data.get("system_registers") is None else
                                 tuple(data["system_registers"]))
    data["forms"] = tuple(tuple(pair) for pair in data.get("forms", ()))
    data["pk_extra"] = tuple(data.get("pk_extra", ()))
    if isinstance(data.get("pk_values"), dict):
        data["pk_values"] = {int(key): item for key, item in data["pk_values"].items()}
    if data.get("constant_pool") is not None:
        data["constant_pool"] = tuple(data["constant_pool"])
    if isinstance(data.get("prologue"), str):
        data["prologue"] = bytes.fromhex(data["prologue"])
    if isinstance(data.get("launch"), dict):
        data["launch"] = Launch(**data["launch"])
    return CompilerABI(**data)


# ---- MM P13: gemm_generic's ADMITTED-CLASS TABLE (docs/g17-tensorops-machine-model.md section 25.127) ----
# P13's closure is "per-class admission ... absence remains a refusal". gemm_generic is a rule-based
# class, and each widening beyond its base rules (sections 25.99 and 25.101) is a named class here with
# the hardware receipt that admitted it, or the reason it is refused. The refusal logic below READS this
# table: a request that belongs to a class whose status is not "admitted", or that falls outside an
# admitted class's rule, or that combines two widening classes no receipt covers, is refused by the
# class's name before anything is built. Both the spec normaliser (tools/g17tensorcommonruntime.py
# generic_spec) and TensorSpec call it; the native worker repeats the numeric bounds.
P13_SECTION = "MM P13, docs/g17-tensorops-machine-model.md section 25.127"
P13_RECEIPTS = "results/g17-p13-classes-v1 (evidence/g17-seta.zip)"
GENERIC_CLASSES = MappingProxyType({
    # admitted, each with its receipt (arm names inside P13_RECEIPTS) and its rule
    "wide_n": dict(status="admitted", receipt=("wide_n256", "wide_n192_split"),
                   rule="N 144..256 in whole tiles on a plain single half x half or bfloat x bfloat GEMM "
                        "(no epilogue, chain, K loop, transpose or other feature), grid and simdgroup splits "
                        "as the base rules allow"),
    "n_tiled_grid": dict(status="admitted", receipt=("ntiled_g8", "ntiled_g16"),
                         rule="N > 256 as an N-tiled grid (MM 25.134): grid_n threadgroups, a power of two, "
                              "each owning N/grid_n <= 256 whole columns read from the threadgroup id, on a "
                              "plain single 16-bit GEMM, N a multiple of 16 x grid_n"),
    "split_k_grid": dict(status="admitted", receipt=("splitk_g8", "splitk_g16"),
                         rule="K partitioned across split_k threadgroups (MM 25.134): threadgroup t computes "
                              "the M x N partial over K-slice [t*K/G, (t+1)*K/G) and writes slot t of a "
                              "(split_k*M) x N buffer; K a multiple of 16 x split_k, one 16-bit GEMM, one axis "
                              "(no grid_n or simdgroup split); the ascending-t fp32 fold reduces the partials"),
    "transposed": dict(status="admitted", receipt=("trans_a", "trans_b", "trans_ab_split"),
                       rule="transA and/or transB (A stored K x M, B stored N x K) on a plain single half x half "
                            "or bfloat x bfloat GEMM, N <= 128, grid and simdgroup splits as the base rules allow"),
    "simdgroups_8": dict(status="admitted", receipt=("sg8", "sg8_grid2"),
                         rule="8 simdgroups (256-thread threadgroups) on a plain single 16-bit GEMM (the K loop "
                              "included, MM 25.144.1), every simdgroup owning a power of two of whole tile rows"),
    "int8_split": dict(status="admitted", receipt=("int8_grid4", "int8_sg2_saturate", "kg_small", "kg_sg4",
                                                   "kg_prefill", "kg_ffn_up", "kg_u2", "kg_u4", "kg_down_u4"),
                       rule="int8 x int8 into int32 under a grid and/or simdgroup split, wrapping or "
                            "(with accumulate) saturating, the K loop included (isa/g17-int8-kloop-grid-results.json: "
                            "K 512..8192 under grid_n 2..128, 1 or 4 simdgroups, kloop_unroll 1..4, bit-exact on hardware); no "
                            "epilogue or chain"),
    "narrow_out_half": dict(status="admitted", receipt=("half_out", "half_out_scale_relu"),
                            rule="a final 'half' epilogue step (fp32 accumulate, op1016 RNE narrowing, C "
                                 "stored as M x N halves) after optional scale/relu steps, on a grid-split "
                                 "one-simdgroup 16-bit GEMM"),
    # refused by name
    "simdgroups_16": dict(status="refused", reason=(
        "16 simdgroups (512-thread threadgroups) are not lowered: tensorgemm admits 1, 2, 4 and 8, and no "
        "16-simdgroup tensor body has a receipt")),
    "sub_byte_int4": dict(status="refused", reason=(
        "int4 is not lowered: there is no 4-bit MMA form, and the int8 widening route (section 25.28) has "
        "no unpack step in tlower and no receipt")),
    "sub_byte_fp4": dict(status="refused", reason=(
        "fp4 (e2m1) is unreachable as an operand on this OS: every air.convert spelling null-dereferences "
        "in the compiler (sections 25.94, 25.94.1); a software decode to fp16 is exact but is not built")),
    "sub_byte_fp6": dict(status="refused", reason=(
        "fp6 (e3m2, e2m3) is unreachable as an operand on this OS (sections 25.94, 25.94.1); a software "
        "decode to fp16 is exact but is not built")),
    "mixed_accumulator": dict(status="refused", reason=(
        "no MMA form accumulates in 16 bits (section 25.13): a half or bfloat C is a conversion of the fp32 "
        "accumulator, which is the narrow_out_half epilogue class, not a native accumulator")),
    "narrow_out_bfloat": dict(status="refused", reason=(
        "a bfloat narrowing epilogue has no measured fp32-to-bfloat conversion in tlower and no receipt; "
        "the half narrowing (narrow_out_half) is the admitted one")),
    "int8_epilogue": dict(status="refused", reason=(
        "an int32 accumulator has no register epilogue: tlower's epilogue steps are fp32 (scale, relu, "
        "exp2, gelu, fp8, half), and requantization is P6's separate class")),
    "int8_chain": dict(status="refused", reason=(
        "an int8 chain would feed an int32 D to the next MMA, which no feed key admits (P2 plan 1, P6)")),
})
# the widening classes: a request in two of them is a combination no single receipt covers
_P13_WIDENING = ("wide_n", "transposed", "simdgroups_8", "int8_split", "narrow_out_half")
# THE K-LOOP HALF EPILOGUE (MM 25.183): the prefill w1/w3 GEMM's delivered launches (M, N, K, simdgroups), each run
# bit-exact on hardware with one final 'half' step, its half_rz and fp32_out controls failing on the same program
_KLOOP_HALF_RECEIPTED = {(128, 8192, 2048, 4), (256, 8192, 2048, 8), (512, 8192, 2048, 4)}
# receipts kloop_half_m128, kloop_half_m256, kloop_half_m512: evidence/g17-prefill-ffn16-v1/half-epilogue-receipts.json


def _kloop_half_fits(view):
    return (view.get("a") == view.get("b") == "half" and list(view.get("epilogue") or ()) == ["half"]
            and view.get("kloop") and not (view.get("stages") or view.get("extras"))
            and (view.get("M"), view.get("N"), view.get("K"), view.get("simdgroups", 1)) in _KLOOP_HALF_RECEIPTED)


# COMBINATIONS WITH THEIR OWN RECEIPT (MM 25.183): a pair of widening classes admitted together, inside the rule below
_P13_COMBINATIONS = {
    frozenset(("simdgroups_8", "narrow_out_half")): dict(
        receipt=("kloop_half_m256",),
        evidence="evidence/g17-prefill-ffn16-v1/half-epilogue-receipts.json",
        rule="the prefill w1/w3 K-loop GEMM in 8 simdgroups (M 128, 256, 512; N 8192, K 2048, grid_n 256) with one "
             "final 'half' epilogue step and nothing else: bit-exact on hardware, the half_rz and fp32_out controls "
             "failing on the same program"),
}
_P13_SUB_BYTE = {"int4": "sub_byte_int4", "uint4": "sub_byte_int4", "s4": "sub_byte_int4", "u4": "sub_byte_int4",
                 "fp4": "sub_byte_fp4", "fp4e2m1": "sub_byte_fp4", "e2m1": "sub_byte_fp4",
                 "fp6": "sub_byte_fp6", "fp6e3m2": "sub_byte_fp6", "fp6e2m3": "sub_byte_fp6",
                 "e3m2": "sub_byte_fp6", "e2m3": "sub_byte_fp6"}
_P13_16BIT = ("half", "bfloat")


def generic_classes(view):
    """The P13 classes a gemm_generic request belongs to, in table order. `view` is a plain dict:
    a, b, c (operand and accumulator type names), M, N, K, simdgroups, groups (threadgroups),
    transA, transB, epilogue (a list of step words), stages, kloop, extras (other features present)."""
    names = []
    for t in (view.get("a"), view.get("b")):
        name = _P13_SUB_BYTE.get(str(t).lower())
        if name and name not in names:
            names.append(name)
    if view.get("c") in _P13_16BIT:
        names.append("mixed_accumulator")
    epilogue = list(view.get("epilogue") or ())
    if "half" in epilogue:
        names.append("narrow_out_half")
    if "bfloat" in epilogue:
        names.append("narrow_out_bfloat")
    sg = view.get("simdgroups", 1)
    if sg == 8:
        names.append("simdgroups_8")
    elif sg == 16:
        names.append("simdgroups_16")
    if isinstance(view.get("N"), int) and view["N"] > 128:
        # A grid split over N (grid_n) is its own class: the per-threadgroup width is a wide_n-shaped
        # tile, but the total N spans the GPU. Without grid_n it is the plain wide_n class (N <= 256).
        names.append("n_tiled_grid" if view.get("grid_n", 1) > 1 else "wide_n")
    if view.get("split_k", 1) > 1:
        # A K-partition grid is its own class: each threadgroup runs a wide_n-shaped tile over K/G, and the
        # partials stack into (split_k*M) x N. Like n_tiled_grid it is not a widening class (it changes the
        # launch and output slot, not the per-threadgroup operand widths).
        names.append("split_k_grid")
    if view.get("transA") or view.get("transB"):
        names.append("transposed")
    if view.get("a") == "int8":
        if sg != 1 or view.get("groups", 1) != 1:
            names.append("int8_split")
        if epilogue:
            names.append("int8_epilogue")
        if view.get("stages"):
            names.append("int8_chain")
    return [n for n in GENERIC_CLASSES if n in names]


def _p13_fits(name, view):
    """Whether a request inside admitted class `name` is inside that class's receipted rule."""
    epilogue = list(view.get("epilogue") or ())
    plain = not (epilogue or view.get("stages") or view.get("kloop") or view.get("extras"))
    sixteen = view.get("a") == view.get("b") and view.get("a") in _P13_16BIT
    if name == "wide_n":
        return plain and sixteen and view["N"] <= 256 and not (view.get("transA") or view.get("transB"))
    if name == "n_tiled_grid":
        gn = view.get("grid_n", 1)
        # the per-threadgroup tile count N/(16*gn) must be a power of two (tlower's column offset is a shift),
        # so N=6144 gn=32 (12 tiles) is outside the class even though gn and N%(16*gn) are fine
        tiles = view["N"] // (16 * gn) if gn and view.get("N") and view["N"] % (16 * gn) == 0 else 0
        return (plain and sixteen and gn >= 2 and not (gn & (gn - 1)) and view["N"] % (16 * gn) == 0
                and tiles and not (tiles & (tiles - 1))
                and view["N"] // gn <= 256 and not (view.get("transA") or view.get("transB")))
    if name == "split_k_grid":
        gk = view.get("split_k", 1)
        # plain-but-for-the-K-loop: a K > 256 GEMM is a K loop, which is exactly the split-K use, so the
        # loop is expected here (n_tiled_grid is treated the same way - both are non-widening launch classes)
        plain_kloop = not (epilogue or view.get("stages") or view.get("extras"))
        gn = view.get("grid_n", 1)
        # when combined with the column grid, the per-tg tile count N/(16*gn) is a shift too (as n_tiled_grid)
        tiles = view["N"] // (16 * gn) if gn and view.get("N") and view["N"] % (16 * gn) == 0 else 0
        return (plain_kloop and sixteen and gk >= 2 and view["K"] % (16 * gk) == 0
                and not (gn & (gn - 1)) and view["N"] % (16 * gn) == 0 and view["N"] // gn <= 256
                and tiles and not (tiles & (tiles - 1))
                and view.get("simdgroups", 1) == 1
                and not (view.get("transA") or view.get("transB")))
    if name == "transposed":
        return plain and sixteen and view.get("N", 0) <= 128
    if name == "simdgroups_8":
        # MM 25.144.1: the K loop in 8 simdgroups (each owning whole tile rows), bit-exact on hardware
        return sixteen and not (epilogue or view.get("stages") or view.get("extras"))
    if name == "int8_split":
        # the K loop is admitted since MM 25.145.4 (kg_small .. kg_ffn_up: kloop with grid_n and 4 simdgroups)
        return not (epilogue or view.get("stages") or view.get("extras"))
    if name == "narrow_out_half" and _kloop_half_fits(view):
        return True                                    # the receipted K-loop launches (MM 25.183)
    if name == "narrow_out_half":
        head = epilogue[:-1]
        return (sixteen and epilogue[-1:] == ["half"] and epilogue.count("half") == 1
                and all(st == "relu" or str(st).startswith("scale:") for st in head)
                and view.get("groups", 1) >= 2 and view.get("simdgroups", 1) == 1
                and not (view.get("stages") or view.get("kloop") or view.get("extras")))
    return False


def generic_class_refusal(view):
    """The named P13 refusal text for a gemm_generic request, or None when every class it belongs to is
    admitted and it sits inside that class's rule (a request in no class is the base rules' business)."""
    names = generic_classes(view)
    for name in names:
        entry = GENERIC_CLASSES[name]
        if entry["status"] != "admitted":
            return "refused: gemm_generic class %s is not admitted (%s): %s" % (name, P13_SECTION, entry["reason"])
    widening = [n for n in names if n in _P13_WIDENING]
    combo = _P13_COMBINATIONS.get(frozenset(widening)) if len(widening) == 2 else None
    if combo is not None:
        if _kloop_half_fits(view):
            return None
        return ("refused: gemm_generic classes %s are admitted together only as %s (%s, receipts %s); this request "
                "is outside that rule" % (" + ".join(widening), combo["rule"], P13_SECTION, ", ".join(combo["receipt"])))
    if len(widening) > 1:
        return ("refused: gemm_generic classes %s are admitted one at a time (%s); no receipt covers the "
                "combination" % (" + ".join(widening), P13_SECTION))
    for name in widening:
        if not _p13_fits(name, view):
            return ("refused: gemm_generic class %s is admitted only as %s (%s, receipts %s); this request "
                    "is outside that rule" % (name, GENERIC_CLASSES[name]["rule"], P13_SECTION,
                                              ", ".join(GENERIC_CLASSES[name]["receipt"])))
    return None


def _generic_view(raw):
    """A TensorSpec payload (raw, before validation) as generic_classes' view. Malformed values give
    a view with neutral defaults; pydantic then refuses them with its own error."""
    def num(key, default):
        v = raw.get(key, default)
        return v if isinstance(v, int) and not isinstance(v, bool) else default
    def first(key):
        v = raw.get(key)
        return v[0] if isinstance(v, (list, tuple)) and v and isinstance(v[0], int) else None
    grid, group = first("grid"), first("threadgroup")
    groups = grid // group if grid and group and grid % group == 0 else 1
    epilogue = raw.get("epilogue")
    return dict(a=raw.get("a_type"), b=raw.get("b_type"), c=raw.get("c_type"), M=num("M", 0), N=num("N", 0),
                K=num("K", 0), simdgroups=num("simdgroups", 1), groups=groups, grid_n=num("grid_n", 1), split_k=num("split_k", 1),
                transA=raw.get("transA") is True, transB=raw.get("transB") is True,
                epilogue=[str(e) for e in epilogue] if isinstance(epilogue, (list, tuple)) else [],
                stages=bool(raw.get("stages")), kloop=num("K", 0) > 256, extras=[])


def refuse_generic_class(view):
    """Raise the named P13 refusal for `view`, if there is one."""
    why = generic_class_refusal(view)
    if why:
        raise ValueError(why)


class TensorSpec(BaseModel):
    """Shape and launch facts for the bounded ordinary tensor runtime class.

    The first runtime class is intentionally one measured composed program: one 17x19x16 GEMM
    followed by the scalar epilogue that reads ``threadgroup_position_in_grid``. The compiler's
    ABI carries the instruction facts; this block carries the dimensions and the host launch
    geometry that cannot be recovered from the image without interpreting tensor instructions.
    """
    model_config = CONFIG
    # M reaches GENERIC_MAX_M only in gemm_generic; every other class keeps 128 (checked below)
    M: Annotated[int, Field(ge=1, le=16384)]
    # N above 128 belongs to gemm_generic's wide_n class alone (MM P13; checked below)
    N: Annotated[int, Field(ge=1, le=16384)]   # total N; a single dispatch bounds N/grid_n <= 256 (per-threadgroup)
    # K above 256 belongs to gemm_generic alone (a runtime K loop re-indexes per slice; checked below).
    # The real bound is PER-THREADGROUP: K/split_k <= 4096 (the kloop's 256 16-wide trips), enforced in the
    # launch validator. split_k lets the total K grow past 4096 while each threadgroup stays within the loop.
    K: Annotated[int, Field(ge=1, le=1_048_576)]
    K2: Annotated[int, Field(ge=1, le=256)] | None = None
    lda: Annotated[int, Field(ge=1, le=1_000_000)]
    ldb: Annotated[int, Field(ge=1, le=1_000_000)]
    ldc: Annotated[int, Field(ge=1, le=1_000_000)]
    # fp8e4m3 / fp8e5m2 only in the gemm_fp8 class (checked below): one byte, unpacked to bf16
    # int8 (and c_type "int", int32) belong to gemm_generic alone (checked below)
    a_type: Literal["half", "bfloat", "float", "uchar", "fp8e4m3", "fp8e5m2", "int8"]
    b_type: Literal["half", "bfloat", "float", "uchar", "fp8e4m3", "fp8e5m2", "int8"]
    c_type: Literal["float", "int"]
    # simdgroups > 1, threadgroups wider than 32 and grids past 256 threads belong to gemm_generic
    # alone; every other class is held to 1 / 32 / its own grid below.
    # 8 is gemm_generic's simdgroups_8 class (MM P13); 16 is refused by name before this Literal
    simdgroups: Literal[1, 2, 4, 8]
    # 32 threads for every class but gemm_grid, whose G threadgroups of 32 split M (checked below).
    grid: tuple[Literal[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536], Literal[1], Literal[1]]   # 65536: 256 threadgroups of 8 simdgroups (MM 25.144.1)
    threadgroup: tuple[Literal[32, 64, 128, 256], Literal[1], Literal[1]]
    # P3 N-tiled grid: grid_n of the grid's threadgroups split N (columns), the rest split M (rows).
    grid_n: Annotated[int, Field(ge=1, le=256)] = 1
    # P3 split-K: split_k of the grid's threadgroups partition K (the contraction); each writes an M x N
    # partial to slot t of a (split_k*M) x N buffer, folded ascending-t in fp32 outside this dispatch.
    split_k: Annotated[int, Field(ge=1, le=256)] = 1
    # gemm_generic only: register epilogue steps, "scale:<fp32 bits hex>" or "relu"
    epilogue: tuple[str, ...] | None = None
    # gemm_generic only: a GEMM CHAIN, one (N_i, K_i, A type) per stage after stage 0; stage 0 is
    # (N, K) above with its own types. Every stage i reads the previous C as its A.
    # a fourth element names a register-feed mode, "A" (the default), "B", "At" or "Bt" (recon
    # section 132 part 3); the non-A modes are one square stage
    stages: tuple[tuple[int, int, str] | tuple[int, int, str, str], ...] | None = None
    # gemm_generic int8 only: C is an input (the worker seeds it from c.f32 every query) and
    # D = C + A.B; `saturate` clips to int32 at every MMA issue instead of wrapping
    accumulate: Literal[True] | None = None
    saturate: Literal[True] | None = None
    # gemm_generic's transposed class only (MM P13): A stored as A^T (K x M) and/or B as B^T (N x K);
    # lda/ldb stay the LOGICAL K and N here, so every buffer keeps its untransposed size
    transA: Literal[True] | None = None
    transB: Literal[True] | None = None
    composition: Literal["gemm_fadd_threadgroup_position", "gemm_fadd_gemm_memory",
                        "gemm_fadd_fmul_gemm_memory", "gemm_fadd_fmul_gemm_fadd_gemm_memory",
                        "row_softmax_fp32", "ffn_gelu_layernorm", "ffn_gelu_layernorm_allrows",
                        "ffn_gelu_layernorm_wide",
                        "transformer_layer", "transformer_layer_weight_offset", "transformer_continuation_weight_offset",
                        "transformer_two_layer",
                        "gemm_relu_vec_gemm_residual_memory", "gemm_weight_offset",
                        # Two adjacent bodies: the second reads the first's C as its fp32 A, either
                        # through memory (the control) or straight from the accumulator registers.
                        "gemm_gemm_memory", "gemm_gemm_register",
                        # the same with a register epilogue (bias, scale, relu) on the first body
                        "gemm_epilogue_gemm_memory", "gemm_epilogue_gemm_register",
                        # one GEMM split over G threadgroups (tlower's grid; SR_TG_X)
                        "gemm_grid",
                        # one fp8 GEMM (e4m3 A, e5m2 B) and the scalar epilogue
                        "gemm_fp8",
                        # ONE RULE-BASED CLASS for new verifications (Set A with Set C): any whole-tile
                        # single GEMM inside the caps below, rather than one class per shape
                        "gemm_generic",
                        # production row P7 (MM 25.129): the named fused attention class, on
                        # gemm_generic's transport; its admission rules are attention_spec's
                        "attention"]
    a2_type: Literal["float"] | None = None
    b2_type: Literal["half"] | None = None
    K3: Annotated[int, Field(ge=1, le=256)] | None = None
    # Byte offset inside public binding 2 for the second GEMM's B matrix.  This is a measured
    # same-binding class, deliberately finite; descriptor offsets remain 0,2,4.
    weight_offset_b: Literal[4096, 4352, 8192, 12288, 16384, 20480] | None = None
    # Three-region transformer extension. The tuple is deliberately exact: these are the
    # measured same-binding positions used by the first offset-bearing layer class.
    weight_offsets_b: tuple[Literal[0], Literal[4096], Literal[8192]] | None = None

    @model_validator(mode="before")
    @classmethod
    def no_cooperative_sharing(cls, value):
        # MM P12 first: a sharing key is refused by name, not as a generic "extra input" (every class)
        refuse_sharing(value)
        # MM P13: a gemm_generic request in an unadmitted class (sub-byte operands, 16 simdgroups, a
        # 16-bit accumulator, ...) is refused by the class's NAME from the admitted-class table, before
        # pydantic's Literals would refuse it as a generic type error
        if isinstance(value, dict) and value.get("composition") == "gemm_generic":
            refuse_generic_class(_generic_view(value))
        return value

    def _generic(self):
        """gemm_generic: the rules, not a table (agreed with Set C, the runtime owner)."""
        fp8 = ("fp8e4m3", "fp8e5m2")
        types = ("half", "bfloat", "float") + fp8
        if any(v % 16 for v in (self.M, self.N, self.K)):
            raise ValueError("gemm_generic: M, N and K must be multiples of 16")
        if (self.lda, self.ldc) != (self.K, self.N) or (
                self.ldb != self.N if self.stages is None else not (16 <= self.ldb <= self.N and self.ldb % 16 == 0)):
            # contiguous row-major; for a chain, ldb is stage 0's width N0 and N the C buffer's width
            raise ValueError("gemm_generic: contiguous row-major operands only")
        if (self.a_type == "int8") != (self.b_type == "int8"):
            raise ValueError("gemm_generic: int8 operands are paired")
        if self.a_type == "int8":
            if self.c_type != "int":
                raise ValueError("gemm_generic: int8 accumulates into int32 (c_type int)")
            # splits are MM P13's int8_split class (admitted by the class table); epilogues and chains
            # were refused there by name already
            if self.epilogue is not None or self.stages is not None:
                raise ValueError("gemm_generic: int8 carries no epilogue and no chain")
            if self.saturate and not self.accumulate:
                raise ValueError("gemm_generic: a saturating int8 GEMM is verified as an accumulate")
            return self._generic_launch()
        if self.accumulate is not None or self.saturate is not None:
            raise ValueError("gemm_generic: accumulate and saturate are int8 only")
        if self.a_type not in types or self.b_type not in types or self.c_type != "float":
            raise ValueError("gemm_generic: operand types are half, bfloat, float, fp8e4m3, fp8e5m2 into float")
        if (self.a_type in fp8 or self.b_type in fp8) and not all(t in fp8 or t == "bfloat" for t in (self.a_type, self.b_type)):
            raise ValueError("gemm_generic: an fp8 operand pairs with fp8 or bfloat")
        return self._generic_launch()

    def _generic_launch(self):
        if self.threadgroup[0] != 32 * self.simdgroups:
            raise ValueError("gemm_generic: a threadgroup is 32 threads per simdgroup")
        if self.grid[0] % self.threadgroup[0]:
            raise ValueError("gemm_generic: the grid is whole threadgroups")
        groups = self.grid[0] // self.threadgroup[0]
        # P3 N-TILED GRID: `grid_n` of the threadgroups split N (columns), each owning N/grid_n whole
        # tiles; the remaining `groups // grid_n` split M (rows) as before. grid_n=1 leaves the M-only
        # check unchanged. Column groups are a power of two of whole tiles (tlower folds the shift).
        col_groups = self.grid_n
        if col_groups > 1:
            if groups % col_groups or (col_groups & (col_groups - 1)) or self.N % (16 * col_groups):
                raise ValueError("gemm_generic: grid_n splits N into a power of two of whole-tile column groups")
            # tlower's column offset is a shift (t << log2(16*NT)), so the per-threadgroup TILE count
            # N/(16*grid_n) must itself be a power of two - not just grid_n. Without this the class admits a
            # spec (e.g. N=6144 grid_n=32 -> 12 tiles) that tlower then refuses at build; carry it here.
            tiles_per_tg = self.N // (16 * col_groups)
            if tiles_per_tg & (tiles_per_tg - 1):
                raise ValueError("gemm_generic: tile columns per threadgroup (N/(16*grid_n)) must be a power of "
                                 "two - the column offset is a shift; N=%d grid_n=%d gives %d, so split it "
                                 "(e.g. 6144 as 3 x 2048)" % (self.N, col_groups, tiles_per_tg))
            row_groups = groups // col_groups
        else:
            row_groups = groups
        # P3 SPLIT-K: `split_k` of the threadgroups partition K (the contraction), each computing an M x N
        # partial over K/split_k and writing it to slot t of a (split_k*M) x N buffer. One parallel axis at
        # a time (as tlower requires); K must split into whole 16-wide issues (no power-of-two constraint:
        # tlower folds t*mult by the set bits of the multiplier, not a single shift).
        k_groups = self.split_k
        if k_groups > 1:
            # split_k with a simdgroup split (MM 25.144.1): each simdgroup owns its rows inside the K group's
            # partial slot; tlower applies the simdgroup's row offset before the K group's
            if self.simdgroups not in (1, 2, 4):
                raise ValueError("gemm_generic: split_k combines with 1, 2 or 4 simdgroups")
            # split_k DOES combine with grid_n: the id is kg*grid_n + col, so col_groups (a power of two) are
            # peeled first, then the K groups. The remaining threadgroups split M (rows) as usual.
            if col_groups > 1 and (col_groups & (col_groups - 1)):
                raise ValueError("gemm_generic: split_k with grid_n needs grid_n a power of two")
            if row_groups % k_groups or self.K % (16 * k_groups):
                raise ValueError("gemm_generic: split_k partitions K into whole-tile (16 x split_k) slices across the threadgroups")
            row_groups = row_groups // k_groups
        # PER-THREADGROUP K is the real bound: each threadgroup runs K/split_k of the contraction as a K loop
        # of at most 256 16-wide trips (4096). split_k lets the TOTAL K grow past 4096 (e.g. the FFN down
        # projection K=8192 as split_k=4 -> 2048 per threadgroup, one dispatch instead of two K-half launches).
        # MM 25.144.1: the loop's bound is TRIPS (<= 255); 2 or more slices per trip reach K 8192 per threadgroup, and
        # tlower refuses a trip count past the compare immediate by name
        if self.K // k_groups > 8192:
            raise ValueError("gemm_generic: per-threadgroup K (K/split_k) must be <= 8192 (255 K-loop trips at 2 or "
                             "more slices per trip); raise split_k so each threadgroup's K slice fits")
        # PER-THREADGROUP N is the real bound: a single threadgroup fits N/grid_n <= 256 whole columns in
        # its register budget. grid_n keeps that bounded while the total N (a wide projection) grows.
        if self.N // col_groups > 256:
            raise ValueError("gemm_generic: per-threadgroup N (N/grid_n) must be <= 256 whole columns (the single-threadgroup register budget)")
        # UP TO 256 THREADGROUPS, any power of two (tlower's index mask is eight bits), and the grid and
        # simdgroup splits TOGETHER (Set A performance item 3: occupancy of 8+ simdgroups per core needs
        # both - 20 cores x 8 is 160 simdgroups, past the old 8 x 4 = 32).
        if groups not in (1, 2, 4, 8, 16, 32, 64, 128, 256) or self.M % (16 * self.simdgroups * row_groups):
            hint = ("" if col_groups == 1 and k_groups == 1 else
                    " (the launch is threadgroups x grid_n x split_k, so %d row group(s) remain after peeling "
                    "grid_n=%d and split_k=%d; `threadgroups` counts row groups only, not the total)"
                    % (row_groups, col_groups, k_groups))
            raise ValueError("gemm_generic: a power of two of threadgroups up to 256, each owning whole tiles "
                             "per simdgroup%s" % hint)
        # AT MOST 64 TILE ROWS PER SIMDGROUP (Set C's review): tlower unrolls every tile row, and the
        # old M <= 1024 cap in one simdgroup was 64 of them. M may grow past 1024 only by adding
        # threadgroups or simdgroups, so no admitted body is larger than a validated one.
        if self.M // (self.simdgroups * row_groups) > 1024:
            raise ValueError("gemm_generic: at most 1024 rows (64 tile rows) per simdgroup")
        if any(v is not None for v in (self.K2, self.K3, self.a2_type, self.b2_type,
                                         self.weight_offset_b, self.weight_offsets_b)):
            raise ValueError("gemm_generic v1 is one GEMM")
        if self.stages is not None:
            # a chain: stage 0 is (N0, K) half x half, and each later stage (N_i, K_i, t) reads the
            # previous stage's C (width N_{i-1}) as A: K_i == N_{i-1}, t "float" (fp32 A) or "half"
            # (narrowed in registers). N above is the C buffer's width, max over every stage.
            if not 1 <= len(self.stages) <= 7 or groups != 1 or self.simdgroups != 1:
                raise ValueError("gemm_generic stages: 2..8 bodies in one simdgroup and one threadgroup")
            if (self.a_type, self.b_type) != ("half", "half") or self.epilogue:
                raise ValueError("gemm_generic stages: stage 0 is half x half, without an epilogue")
            widths = [s[0] for s in self.stages]
            prev = self.ldb                     # stage 0's width N0
            for stage in self.stages:
                n, k, t = stage[:3]
                if len(stage) == 4 and stage[3] not in ("A", "B", "At", "Bt"):
                    raise ValueError("gemm_generic stages: a feed mode is A, B, At or Bt")
                if len(stage) == 4 and stage[3] != "A" and (len(self.stages) != 1 or
                                                             not self.M == self.N == n == k):
                    raise ValueError("gemm_generic stages: a B, At or Bt feed is one square stage")
                if n % 16 or k % 16 or n > 128 or k > 256 or t not in ("float", "half"):
                    raise ValueError("gemm_generic stages: whole tiles, N <= 128, K <= 256, A float or half")
                if k != prev:
                    raise ValueError("gemm_generic stages: each stage's K is the previous stage's N")
                prev = n
            if max(widths) > self.N:
                raise ValueError("gemm_generic stages: N is the C width, at least every stage's N")
        for step in self.epilogue or ():
            # ADDITIVE (Set A item 6): "gelu", the compiler's x*sigmoid(1.702x) as a register epilogue step
            # "half" is MM P13's narrow_out_half class (its rule is in the class table); "mx32e8m0" is MM P5's
            # in-kernel E8M0 scale decode (25.128)
            if not (step in ("relu", "exp2", "gelu", "fp8e4m3", "fp8e5m2", "mx32", "mx32e8m0", "half") or (step.startswith("scale:") and len(step) == 16)):
                raise ValueError("gemm_generic: epilogue steps are relu, exp2, gelu, scale:0x<8 hex digits>, fp8e4m3, fp8e5m2, mx32, mx32e8m0 or half")
        # ADDITIVE (Set A item 9a): mx32 is the first step (block-scaled accumulation). The worker then
        # appends (K/32) x M fp32 factors to A's payload and (K/32) x N to B's
        # ADDITIVE (production row P5, MM 25.128): mx32e8m0 is the same step reading E8M0 CODE BYTES,
        # (K/32) x M bytes after A and (K/32) x N after B, decoded in the kernel
        mx_steps = [st for st in (self.epilogue or ()) if st in ("mx32", "mx32e8m0")]
        if mx_steps and (len(mx_steps) != 1 or self.epilogue[0] != mx_steps[0] or self.K % 32 or self.stages is not None
                         or self.simdgroups != 1 or "float" in (self.a_type, self.b_type)):
            raise ValueError("gemm_generic: mx32 is the first epilogue step of a one-simdgroup 16-bit or fp8 GEMM, K a multiple of 32")
        # ADDITIVE (Set A item 9b): an fp8 quantize-out is the final step and C then carries bytes, so
        # the grid must be split (a single threadgroup's C[0,0] += 1 tail would add to packed bytes)
        fp8_steps = [st for st in (self.epilogue or ()) if st.startswith("fp8")]
        if fp8_steps and (len(fp8_steps) != 1 or self.epilogue[-1] != fp8_steps[0] or self.grid[0] < 64):
            raise ValueError("gemm_generic: fp8 quantize-out is one final epilogue step on a grid-split GEMM")
        return self


    @model_validator(mode="after")
    def dimensions(self):
        if self.composition == "gemm_generic":
            # its own contiguity rule, which for a chain lets ldb (stage 0's width) be narrower than
            # N (the C buffer's width)
            return self._generic()
        if self.composition == ATTENTION_COMPOSITION:
            # the transport facts of the attention class (attention_layout): half Q/X and Wk|Wv into
            # fp32 regions, K 64 and N 80, one simdgroup and threadgroup, nothing else - or phase grid's
            # (MM 25.135): K 1024 and N 256, M = 64 rows per head, one 32-thread threadgroup per head
            tr = ATTENTION_GRID_TRANSPORT
            grid_transport = (self.K, self.N, self.lda, self.ldb, self.ldc) == (tr["K"], tr["N"], tr["K"], tr["N"], tr["N"])
            if not grid_transport and (self.K, self.N, self.lda, self.ldb, self.ldc) != (
                    ATTENTION_D, ATTENTION_D + ATTENTION_VALUE, ATTENTION_D, ATTENTION_D + ATTENTION_VALUE,
                    ATTENTION_D + ATTENTION_VALUE):
                raise ValueError(ATTENTION_REFUSALS["attention_shape"])
            if self.M % 16 or (self.a_type, self.b_type, self.c_type) != ("half", "half", "float"):
                raise ValueError(ATTENTION_REFUSALS["attention_shape"])
            if grid_transport:
                heads = self.grid[0] // 32
                # the KV split (MM 25.114.6): heads x S threadgroups, and M the split layout's rows
                split = any(heads % S == 0 and heads // S in ATTENTION_GRID_HEADS and
                            self.M == attention_grid_layout(dict(heads=heads // S, kv_split=S, head=128, value=128))["M"]
                            for S in ATTENTION_GRID_SPLITS)
                # phase grid_merge (MM 25.135.5) addresses the heads' scratch slots past the grid regions
                merge_rows = {kv_split_transport_rows(kv_split_c_bytes(heads, S), heads * S)
                              for S in ATTENTION_KV_SPLITS} if heads in ATTENTION_GRID_HEADS else set()
                if (self.simdgroups != 1 or self.threadgroup[0] != 32 or self.grid[0] % 32 or
                        not (split or (heads in ATTENTION_GRID_HEADS and
                                       (self.M == tr["rows_per_head"] * heads or self.M in merge_rows)))):
                    raise ValueError(ATTENTION_REFUSALS["attention_heads"])
            elif self.simdgroups != 1 or self.grid[0] != 32 or self.threadgroup[0] != 32:
                raise ValueError(ATTENTION_REFUSALS["attention_launch"])
            if any(v is not None for v in (self.K2, self.K3, self.a2_type, self.b2_type, self.weight_offset_b,
                                             self.weight_offsets_b, self.epilogue, self.stages, self.accumulate,
                                             self.saturate)):
                raise ValueError(ATTENTION_REFUSALS["attention_shape"] + "; the attention transport carries no "
                                 "other generic feature")
            return self
        if self.lda < self.K or self.ldb < self.N or self.ldc < self.N:
            raise ValueError("tensor leading dimensions are smaller than their logical extents")
        # FIRST, before any class returns early: every non-generic class keeps its pre-generic domain
        # (the widened Literals exist for gemm_generic alone)
        if self.K > 256:
            raise ValueError("only gemm_generic exceeds K 256")
        if self.N > 128 or self.transA is not None or self.transB is not None:
            raise ValueError("only gemm_generic exceeds N 128 or carries a transpose (MM P13)")
        if self.M > 128 or self.simdgroups != 1 or self.threadgroup[0] != 32 or self.grid[0] > 256:
            raise ValueError("only gemm_generic exceeds M 128, one simdgroup, 32-thread threadgroups or 256 threads")
        if self.epilogue is not None:
            raise ValueError("only gemm_generic carries an epilogue list")
        if (self.accumulate is not None or self.saturate is not None or self.c_type != "float"
                or "int8" in (self.a_type, self.b_type)):
            raise ValueError("only gemm_generic carries int8, int32 C, accumulate or saturate")
        if self.composition == "row_softmax_fp32":
            if (self.M, self.N, self.K) != (32, 16, 64):
                raise ValueError("compiler-owned row reduction is measured only for 32x16 scores at K=64")
            if self.a_type != "half" or self.b_type != "half" or self.c_type != "float":
                raise ValueError("compiler-owned row reduction is measured only for half inputs and FP32 scores")
            if (self.lda, self.ldb, self.ldc) != (64, 16, 16):
                raise ValueError("compiler-owned row reduction requires the measured contiguous score layout")
            return self
        if self.composition == "transformer_layer":
            if (self.M, self.N, self.K, self.K2, self.K3) != (16, 32, 64, 32, 32):
                raise ValueError("transformer layer slice requires 16x32x64 then two 16x32x32 GEMMs")
            if (self.a_type, self.b_type, self.a2_type, self.b2_type) != ("half", "half", "float", "half"):
                raise ValueError("transformer layer slice requires half/half then float/half operands")
            if (self.lda, self.ldb, self.ldc) != (64, 32, 32):
                raise ValueError("transformer layer slice requires contiguous leading dimensions")
            if self.weight_offsets_b is not None or self.weight_offset_b is not None:
                raise ValueError("ordinary transformer layer carries no B weight offsets")
            return self
        if self.composition == "transformer_layer_weight_offset":
            if (self.M, self.N, self.K, self.K2, self.K3) != (16, 32, 64, 32, 32):
                raise ValueError("offset transformer layer requires 16x32x64 then two 16x32x32 GEMMs")
            if (self.a_type, self.b_type, self.a2_type, self.b2_type) != ("half", "half", "float", "half"):
                raise ValueError("offset transformer layer requires half/half then float/half operands")
            if (self.lda, self.ldb, self.ldc) != (64, 32, 32):
                raise ValueError("offset transformer layer requires contiguous leading dimensions")
            if self.weight_offsets_b != (0, 4096, 8192) or self.weight_offset_b is not None:
                raise ValueError("offset transformer layer has only the measured B positions 0,4096,8192")
            return self
        if self.composition == "transformer_continuation_weight_offset":
            if (self.M, self.N, self.K, self.K2, self.K3) != (16, 32, 32, 32, 32):
                raise ValueError("continuation layer requires three 16x32x32 GEMMs")
            if (self.a_type, self.b_type, self.a2_type, self.b2_type) != ("float", "half", "float", "half"):
                raise ValueError("continuation layer requires float/half operands")
            if (self.lda, self.ldb, self.ldc) != (32, 32, 32):
                raise ValueError("continuation layer requires contiguous leading dimensions")
            if self.weight_offsets_b != (0, 4096, 8192) or self.weight_offset_b is not None:
                raise ValueError("continuation layer has only the measured B positions 0,4096,8192")
            return self
        if self.composition == "transformer_two_layer":
            if (self.M, self.N, self.K, self.K2, self.K3) != (16, 32, 64, 32, 32):
                raise ValueError("two-transformer-layer slice requires 16x32x64 then four 16x32x32 GEMMs")
            if (self.a_type, self.b_type, self.a2_type, self.b2_type) != ("half", "half", "float", "half"):
                raise ValueError("two-transformer-layer slice requires half/half then float/half operands")
            if (self.lda, self.ldb, self.ldc) != (64, 32, 32):
                raise ValueError("two-transformer-layer slice requires contiguous leading dimensions")
            if self.weight_offsets_b is not None or self.weight_offset_b is not None:
                raise ValueError("two-transformer layer carries no B weight offsets")
            return self
        if self.composition in ("ffn_gelu_layernorm", "ffn_gelu_layernorm_allrows",
                                "ffn_gelu_layernorm_wide"):
            expected = ((16, 32, 64, 32) if self.composition == "ffn_gelu_layernorm_wide"
                        else (16, 16, 64, 16))
            if (self.M, self.N, self.K, self.K2) != expected:
                raise ValueError("compiler-owned FFN slice has no measured dimensions for this class")
            if (self.a_type, self.b_type, self.a2_type, self.b2_type) != ("half", "half", "float", "half"):
                raise ValueError("compiler-owned FFN slice requires half/half then float/half operands")
            expected_ld = (64, 32, 32) if self.composition == "ffn_gelu_layernorm_wide" else (64, 16, 16)
            if (self.lda, self.ldb, self.ldc) != expected_ld:
                raise ValueError("compiler-owned FFN slice requires contiguous leading dimensions")
            if self.K3 is not None:
                raise ValueError("compiler-owned FFN slice has no third GEMM")
            return self
        fp8 = ("fp8e4m3", "fp8e5m2")
        if (self.a_type in fp8 or self.b_type in fp8) != (self.composition == "gemm_fp8"):
            raise ValueError("fp8 operands belong to the gemm_fp8 class only")
        if self.composition == "gemm_fp8":
            if (self.M, self.N, self.K, self.lda, self.ldb, self.ldc) != (32, 32, 64, 64, 32, 32):
                raise ValueError("the fp8 class is measured for 32x32x64 contiguous")
            if (self.a_type, self.b_type, self.c_type) != ("fp8e4m3", "fp8e5m2", "float"):
                raise ValueError("the fp8 class is e4m3 x e5m2 into fp32")
            if self.grid[0] != 32 or any(v is not None for v in (self.a2_type, self.b2_type, self.K3)):
                raise ValueError("the fp8 class is one GEMM in one threadgroup")
            return self
        if self.grid[0] != 32 and self.composition != "gemm_grid":
            raise ValueError("only the gemm_grid class launches more than one 32-thread threadgroup")
        if self.composition == "gemm_grid":
            if (self.M, self.N, self.K, self.lda, self.ldb, self.ldc) != (128, 32, 64, 64, 32, 32):
                raise ValueError("the grid class is measured for 128x32x64 contiguous")
            if (self.a_type, self.b_type, self.c_type) != ("half", "half", "float"):
                raise ValueError("the grid class is half/half into fp32")
            if any(v is not None for v in (self.a2_type, self.b2_type, self.K3)):
                raise ValueError("the grid class is one GEMM")
            return self
        if self.composition == "gemm_fadd_threadgroup_position":
            if (self.M, self.N, self.K) != (17, 19, 16):
                raise ValueError("ordinary tensor runtime class is measured only for 17x19x16")
            if self.a_type != "half" or self.b_type != "half":
                raise ValueError("ordinary tensor runtime class is measured only for half operands")
            if (self.lda, self.ldb, self.ldc) != (self.K, self.N, self.N):
                raise ValueError("ordinary tensor runtime class requires contiguous leading dimensions")
            if (self.K2 is not None or self.a2_type is not None or self.b2_type is not None or
                    self.weight_offset_b is not None):
                raise ValueError("single tensor runtime class has no second GEMM")
        elif self.composition == "gemm_weight_offset":
            if (self.M, self.N, self.K, self.K2) != (16, 32, 64, 32):
                raise ValueError("weight-offset class is measured only for 16x32x64 then 16x32x32")
            if (self.a_type, self.b_type, self.a2_type, self.b2_type) != ("half", "half", "float", "half"):
                raise ValueError("weight-offset class requires half/half then float/half operands")
            if (self.lda, self.ldb, self.ldc) != (64, 32, 32):
                raise ValueError("weight-offset class requires contiguous leading dimensions")
            if self.weight_offset_b not in (4096, 4352, 8192, 12288, 16384, 20480):
                raise ValueError("weight-offset class has no measured B byte offset")
            if self.K3 is not None:
                raise ValueError("weight-offset class has exactly two GEMMs")
        else:
            if (self.M, self.N, self.K, self.K2) != (32, 32, 64, 32):
                raise ValueError("multigemm runtime class is measured only for 32x32x64 then 32x32x32")
            if (self.a_type, self.b_type, self.a2_type, self.b2_type) != ("half", "half", "float", "half"):
                raise ValueError("multigemm runtime class requires half/half then float/half operands")
            if (self.lda, self.ldb, self.ldc) != (64, 32, 32):
                raise ValueError("multigemm runtime class requires contiguous leading dimensions")
            if self.composition in ("gemm_fadd_fmul_gemm_fadd_gemm_memory",
                                    "gemm_relu_vec_gemm_residual_memory"):
                if self.K3 != 32:
                    raise ValueError("three-GEMM runtime class requires a 32-wide third reduction")
            elif self.K3 is not None:
                raise ValueError("two-GEMM runtime classes do not carry a third reduction")
            if self.weight_offset_b is not None:
                raise ValueError("ordinary multigemm classes do not carry a B weight offset")
        return self


class RequantizationSpec(BaseModel):
    """Host facts for the measured scalar stage between two tensor dispatches."""
    model_config = CONFIG
    elements: Literal[256]
    groups: Literal[8]
    scale: Literal["1/512", "1/16"]
    saturation: Literal["signed_int8", "unsigned_uint8"]
    grid: tuple[Literal[256], Literal[1], Literal[1]]
    threadgroup: tuple[Literal[32], Literal[1], Literal[1]]
    dispatch_boundary: Literal["required"]

    @model_validator(mode="after")
    def shape(self):
        if self.saturation == "unsigned_uint8" and self.scale != "1/16":
            raise ValueError("unsigned requantization is measured only at scale 1/16")
        if self.saturation == "signed_int8" and self.scale not in ("1/512", "1/16"):
            raise ValueError("signed requantization scale is outside the measured set")
        return self


class Shape(BaseModel):
    model_config = CONFIG
    rows: Annotated[int, Field(ge=1, le=500_000)]
    columns: Annotated[int, Field(ge=1, le=16384)]   # the readback width; a wide N-tiled projection reads back its full N (the MiniLM query still requires exactly 384, checked below)


class RuntimeContract(BaseModel):
    model_config = CONFIG
    format: Literal["g17-common-pipeline-v1", "g17-common-pipeline-v2", "g17-common-pipeline-v3",
                    "g17-common-pipeline-v4"]
    kind: Literal["packed_scan", "separate_scan", "affine", "layernorm", "minilm_query",
                  TENSOR_KIND, REQUANT_KIND]
    name: str
    shape: Shape
    abi: CompilerABI
    tensor: TensorSpec | None = None
    requantization: RequantizationSpec | None = None

    @model_validator(mode="after")
    def agreement(self):
        if self.kind == REQUANT_KIND:
            if self.name != "requant_scalar_stage":
                raise ValueError("requantization runtime function name differs from measured stage")
            if self.format != manifest_format(self.kind):
                raise ValueError("requantization runtime manifest version differs from ABI v3")
            if self.requantization is None or self.tensor is not None:
                raise ValueError("requantization runtime requires its scalar-stage launch block")
            if self.shape.rows != 256 or self.shape.columns != 1:
                raise ValueError("requantization runtime shape is measured only for 256 elements")
            if self.abi.requantization is None:
                raise ValueError("requantization runtime requires the compiler marker")
            if (self.abi.requantization.get("scale") != self.requantization.scale or
                    self.abi.requantization.get("saturation") != self.requantization.saturation):
                raise ValueError("requantization marker and launch facts disagree")
            return self
        if self.kind == TENSOR_KIND:
            if self.name not in ("tensor_runtime_demo", "tensor_multigemm_runtime_demo",
                                 "tensor_multigemm_fadd_fmul_runtime_demo",
                                 "tensor_multigemm_fadd_fmul_gemm_runtime_demo",
                                 "tensor_row_softmax_runtime_demo",
                                 "tensor_ffn_gelu_layernorm_runtime_demo",
                                 "tensor_ffn_gelu_layernorm_allrows_runtime_demo",
                                 "tensor_ffn_gelu_layernorm_wide_runtime_demo",
                                 "tensor_transformer_layer_runtime_demo",
                                 "tensor_transformer_layer_weight_offset_runtime_demo",
                                 "tensor_transformer_continuation_weight_offset_runtime_demo",
                                 "tensor_transformer_two_layer_runtime_demo",
                                 "tensor_multigemm_relu_vec_residual_runtime_demo",
                                 "tensor_multigemm_weight_offset_runtime_demo",
                                 "tensor_multigemm_adjacent_runtime_demo",
                                 "tensor_multigemm_register_runtime_demo",
                                 "tensor_multigemm_epilogue_register_runtime_demo",
                                 "tensor_multigemm_epilogue_adjacent_runtime_demo",
                                 "tensor_gemm_grid_runtime_demo",
                                 "tensor_gemm_fp8_runtime_demo",
                                 "tensor_gemm_generic_runtime_demo"):
                raise ValueError("tensor runtime function name differs from the measured composed program")
            if self.format != manifest_format(self.kind):
                raise ValueError("tensor runtime manifest version differs from ABI v5")
            if self.tensor is None:
                raise ValueError("tensor runtime contract requires its tensor launch block")
            # P3 split-K stacks split_k partials along the row axis, so the readback is (split_k*M) x N
            if (self.shape.rows, self.shape.columns) != (self.tensor.M * self.tensor.split_k, self.tensor.N):
                raise ValueError("tensor shape differs from its launch block")
            if self.abi.abi_version != 5:
                raise ValueError("tensor runtime requires ABI v5")
            if self.abi.system_registers != (130, 156) and not (
                    self.name == "tensor_gemm_generic_runtime_demo" and (
                        (self.tensor.simdgroups > 1 and self.abi.system_registers == (130, 133, 156)) or
                        (self.tensor.simdgroups == 1 and self.abi.system_registers == (130, 164, 165)))):
                raise ValueError("the measured scalar epilogue requires SR130 and SR156")
            expected = {
                "tensor_runtime_demo": "gemm_fadd_threadgroup_position",
                "tensor_multigemm_runtime_demo": "gemm_fadd_gemm_memory",
                "tensor_multigemm_fadd_fmul_runtime_demo": "gemm_fadd_fmul_gemm_memory",
                "tensor_multigemm_fadd_fmul_gemm_runtime_demo": "gemm_fadd_fmul_gemm_fadd_gemm_memory",
                "tensor_row_softmax_runtime_demo": "row_softmax_fp32",
                "tensor_ffn_gelu_layernorm_runtime_demo": "ffn_gelu_layernorm",
                "tensor_ffn_gelu_layernorm_allrows_runtime_demo": "ffn_gelu_layernorm_allrows",
                "tensor_ffn_gelu_layernorm_wide_runtime_demo": "ffn_gelu_layernorm_wide",
                "tensor_transformer_layer_runtime_demo": "transformer_layer",
                "tensor_transformer_layer_weight_offset_runtime_demo": "transformer_layer_weight_offset",
                "tensor_transformer_continuation_weight_offset_runtime_demo": "transformer_continuation_weight_offset",
                "tensor_transformer_two_layer_runtime_demo": "transformer_two_layer",
                "tensor_multigemm_relu_vec_residual_runtime_demo": "gemm_relu_vec_gemm_residual_memory",
                "tensor_multigemm_weight_offset_runtime_demo": "gemm_weight_offset",
                "tensor_multigemm_adjacent_runtime_demo": "gemm_gemm_memory",
                "tensor_multigemm_register_runtime_demo": "gemm_gemm_register",
                "tensor_multigemm_epilogue_register_runtime_demo": "gemm_epilogue_gemm_register",
                "tensor_multigemm_epilogue_adjacent_runtime_demo": "gemm_epilogue_gemm_memory",
                "tensor_gemm_grid_runtime_demo": "gemm_grid",
                "tensor_gemm_fp8_runtime_demo": "gemm_fp8",
                "tensor_gemm_generic_runtime_demo": "gemm_generic",
            }[self.name]
            if self.name == "tensor_gemm_generic_runtime_demo" and self.tensor.composition == ATTENTION_COMPOSITION:
                expected = ATTENTION_COMPOSITION         # P7: the attention class shares the generic program name
            if self.tensor.composition != expected:
                raise ValueError("tensor function name and composition differ")
            return self
        if self.name != NAMES[self.kind]:
            raise ValueError("function name differs from the program contract")
        if self.kind == "affine" and self.shape.columns != 1:
            raise ValueError("affine requires a one-dimensional input")
        layernorm = self.kind in FOUR_BUFFER_KINDS
        query = self.kind == "minilm_query"
        if self.abi.abi_version == 3 and self.abi.system_registers != ((160, 161) if query else (160,)):
            raise ValueError("system-register declarations differ from application grid")
        if query and self.shape.columns != 384:
            raise ValueError("MiniLM query requires width 384")
        if self.format != manifest_format(self.kind):
            raise ValueError("application kind and runtime manifest version disagree")
        if layernorm and (self.abi.abi_version != 3 or self.shape.rows > 128):
            raise ValueError("LayerNorm requires ABI v3 and at most 128 token rows")
        count = 4 if layernorm else 3 if self.kind == "separate_scan" else 2
        if len(self.abi.bindings) != count:
            raise ValueError("binding count differs from the program contract")
        if any(b.element_type != ("float" if layernorm else "half") or
               b.element_bytes != (4 if layernorm else 2) for b in self.abi.bindings):
            raise ValueError("binding storage differs from the application contract")
        return self

    @classmethod
    def read(cls, value):
        # Pydantic's strict Python mode intentionally rejects lists for tuple fields.  A manifest
        # is JSON-shaped at this boundary, so normalize its typed nested records explicitly and
        # retain strict validation for every scalar and cross-field rule.
        data = dict(value)
        data["shape"] = (value["shape"] if isinstance(value.get("shape"), Shape)
                         else Shape(**value["shape"]))
        if isinstance(value.get("abi"), CompilerABI):
            data["abi"] = value["abi"]
        else:
            data["abi"] = compiler_abi_from_plain(value["abi"])
        if isinstance(value.get("tensor"), dict):
            tensor = dict(value["tensor"])
            tensor["grid"] = tuple(tensor["grid"])
            tensor["threadgroup"] = tuple(tensor["threadgroup"])
            if tensor.get("weight_offsets_b") is not None:
                tensor["weight_offsets_b"] = tuple(tensor["weight_offsets_b"])
            if tensor.get("epilogue") is not None:
                tensor["epilogue"] = tuple(tensor["epilogue"])
            if tensor.get("stages") is not None:
                tensor["stages"] = tuple(tuple(stage) for stage in tensor["stages"])
            data["tensor"] = TensorSpec(**tensor)
        if isinstance(value.get("requantization"), dict):
            requant = dict(value["requantization"])
            requant["grid"] = tuple(requant["grid"])
            requant["threadgroup"] = tuple(requant["threadgroup"])
            data["requantization"] = RequantizationSpec(**requant)
        if "instructions" in value:
            data["instructions"] = tuple(
                item if isinstance(item, Instruction) else Instruction(**item)
                for item in value["instructions"])
        return cls(**data)

    def layout(self):
        rows, columns = self.shape.rows, self.shape.columns
        # The tensor class is intentionally kept separate from the FP16 scan layout: its three
        # buffers are matrices with independent leading dimensions and the ordinary worker owns
        # one exact 32-lane launch. The payload sizes are in declared storage bytes, not operand
        # arithmetic width (this first common class admits half operands only).
        if self.kind == TENSOR_KIND:
            t = self.tensor
            b_size = (t.weight_offset_b + t.K2 * t.ldb * 2
                      if t.composition == "gemm_weight_offset" else
                      max(t.weight_offsets_b) + t.K2 * t.ldb * 2
                      if t.composition in ("transformer_layer_weight_offset", "transformer_continuation_weight_offset") else
                      t.K * t.ldb * 2)
            a_width = 4 if t.a_type == "float" else 2
            sizes = (t.M * t.lda * a_width, b_size, t.M * t.ldc * 4)
            return dict(function=self.name, kind=self.kind, storage_dtype="mixed", rows=t.M,
                        columns=t.N, matrix_bytes=sizes[0], request_bytes=sizes[0],
                        reply_bytes=sizes[2], reply_elements=t.M*t.N, completion_markers=False,
                        output_bytes=sizes[2] + 256, buffer_allocations=3,
                        buffer_roles=["a", "b", "output"], buffer_payload_bytes=list(sizes),
                        buffer_allocation_bytes=[n + 256 for n in sizes], buffer_offsets=[128]*3,
                        protocol=3, bindings=self.abi.model_dump(mode="json")["bindings"],
                        gpu_dispatched=False, tensor=t.model_dump(mode="json"))
        if self.kind == REQUANT_KIND:
            sizes = (256 * 4, 256 * 4, 256 * 4)
            return dict(function=self.name, kind=self.kind, storage_dtype="requantized_int8_in_i32_words",
                        rows=256, columns=1, matrix_bytes=sizes[0], request_bytes=sizes[0],
                        reply_bytes=sizes[2], reply_elements=256, completion_markers=False,
                        output_bytes=sizes[2] + 256, buffer_allocations=3,
                        buffer_roles=["accumulator", "scale", "output"],
                        buffer_payload_bytes=list(sizes),
                        buffer_allocation_bytes=[n + 256 for n in sizes],
                        buffer_offsets=[128] * 3, protocol=4,
                        bindings=self.abi.model_dump(mode="json")["bindings"],
                        gpu_dispatched=False,
                        requantization=self.requantization.model_dump(mode="json"))
        if self.kind in ("layernorm", "minilm_query"):
            query = self.kind == "minilm_query"
            payloads = (rows*columns*4, columns*columns*4 if query else columns*4,
                        columns*4, rows*columns*4)
            return dict(function=self.name, kind=self.kind, storage_dtype="float32", rows=rows,
                columns=columns, matrix_bytes=payloads[0], request_bytes=payloads[0],
                reply_bytes=payloads[3], reply_elements=rows*columns, completion_markers=False,
                output_bytes=payloads[3]+256, buffer_allocations=4,
                buffer_roles=["source", "weight", "bias", "output"] if query else
                             ["source", "gamma", "beta", "output"],
                buffer_payload_bytes=list(payloads), buffer_allocation_bytes=[n+256 for n in payloads],
                buffer_offsets=[128]*4, protocol=2,
                bindings=self.abi.model_dump(mode="json")["bindings"], gpu_dispatched=False)
        matrix = rows*columns*2
        separate, affine = self.kind == "separate_scan", self.kind == "affine"
        return dict(function=self.name, kind=self.kind, storage_dtype="float16", rows=rows,
                    columns=columns, matrix_bytes=matrix,
                    request_bytes=matrix if affine else columns*2, reply_bytes=rows*2,
                    first_buffer_bytes=matrix + (columns*2 if self.kind == "packed_scan" else 0),
                    query_buffer_bytes=columns*2 if separate else 0, output_bytes=rows*2+128,
                    buffer_allocations=len(self.abi.bindings),
                    bindings=self.abi.model_dump(mode="json")["bindings"], gpu_dispatched=False)


def contract_from_delivery(report):
    kind = report["kind"]
    if kind == REQUANT_KIND:
        return RuntimeContract.read(dict(format=manifest_format(kind), kind=kind,
            name=report["name"], shape=dict(rows=report["rows"], columns=report["columns"]),
            requantization=report["requantization"], abi=report["abi"]))
    if kind == TENSOR_KIND:
        return RuntimeContract.read(dict(format=manifest_format(kind), kind=kind,
            name=report["name"], shape=dict(rows=report["rows"], columns=report["columns"]),
            tensor=report["tensor"], abi=report["abi"]))
    return RuntimeContract.read(dict(format=("g17-common-pipeline-v2" if kind in ("layernorm", "minilm_query")
                                            else "g17-common-pipeline-v1"), kind=kind,
        name=report["name"], shape=dict(rows=report["rows"], columns=report["columns"]), abi=report["abi"]))


class ImageContract(RuntimeContract):
    """A delivered image plus the compiler facts needed to verify its bytes."""
    code_size: Annotated[int, Field(ge=2, le=1_000_000)]
    sha256: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    instructions: tuple[Instruction, ...]
    field_ledger: dict[str, str]
    source: dict | None = None

    @model_validator(mode="after")
    def complete_code(self):
        if set(self.sha256) != {"archive", "library", "object", "code"}:
            raise ValueError("image requires exactly four content hashes")
        cursor = 0
        for instruction in self.instructions:
            if instruction.offset != cursor:
                raise ValueError("compiler instruction boundaries are not consecutive")
            cursor += instruction.length
        if not self.instructions or cursor != self.code_size:
            raise ValueError("compiler instruction boundaries do not cover the code")
        forms = tuple(sorted({(i.opcode, i.length) for i in self.instructions}))
        if forms != self.abi.forms:
            raise ValueError("instruction declarations disagree with compiler ABI forms")
        if not self.field_ledger:
            raise ValueError("image must state its metadata derivations")
        return self


def write_matrix(matrix, path, contract):
    """Own one half matrix snapshot; bounded scratch and no appended query region."""
    rows, columns = contract.shape.rows, contract.shape.columns
    if not isinstance(matrix, np.ndarray) or matrix.dtype != np.float16 or matrix.shape != (rows, columns):
        raise ValueError("matrix must match the FP16 contract")
    maxima = np.zeros(columns, np.float64)
    with Path(path).open("xb") as stream:
        for start in range(0, rows, 1024):
            chunk = np.array(matrix[start:start+1024], dtype="<f2", order="C", copy=True)
            if not np.isfinite(chunk).all():
                raise ValueError("matrix must be finite")
            maxima = np.maximum(maxima, np.max(np.abs(chunk), axis=0))
            chunk.tofile(stream)
    return maxima


def write_layernorm_inputs(matrix, parameters, directory, contract):
    """Own all three FP32 inputs before validating or writing any of them."""
    if contract.kind != "layernorm" or not isinstance(parameters, (tuple, list)) or len(parameters) != 2:
        raise ValueError("LayerNorm requires separate gamma and beta arrays")
    rows, columns = contract.shape.rows, contract.shape.columns
    snapshots = {}
    for name, value, shape in zip(("source", "gamma", "beta"), (matrix, *parameters),
                                  ((rows, columns), (columns,), (columns,))):
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape:
            raise ValueError(f"{name} must match the FP32 LayerNorm shape {shape}")
        owned = np.array(value, dtype="<f4", order="C", copy=True)
        if not np.isfinite(owned).all():
            raise ValueError(f"{name} must be finite")
        snapshots[name] = owned.tobytes()
    for name, data in snapshots.items():
        with (Path(directory)/(name+".f32")).open("xb") as stream:
            stream.write(data)
    return MappingProxyType(snapshots)


def write_query_inputs(matrix, parameters, directory, contract):
    """Freeze query source and checkpoint parameters using their actual sizes."""
    if contract.kind != "minilm_query" or not isinstance(parameters, (tuple, list)) or len(parameters) != 2:
        raise ValueError("query projection requires separate weight and bias")
    rows, width = contract.shape.rows, contract.shape.columns
    snapshots = {}
    for name, value, shape in zip(("source", "weight", "bias"), (matrix, *parameters),
                                  ((rows, width), (width, width), (width,))):
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape:
            raise ValueError(f"{name} must match the FP32 query shape {shape}")
        owned = np.array(value, dtype="<f4", order="C", copy=True)
        if not np.isfinite(owned).all():
            raise ValueError(f"{name} must be finite")
        snapshots[name] = owned.tobytes()
    for name, data in snapshots.items():
        with (Path(directory) / (name + ".f32")).open("xb") as stream:
            stream.write(data)
    return MappingProxyType(snapshots)


def reference(kind, matrix, request):
    """Independent sequential FP32 operations, followed by one FP16 narrowing."""
    a, x = np.asarray(matrix), np.asarray(request)
    if a.dtype != np.float16 or x.dtype != np.float16 or a.ndim != 2:
        raise ValueError("reference requires FP16 storage")
    if not np.isfinite(a).all() or not np.isfinite(x).all():
        raise ValueError("reference requires finite inputs")
    if kind == "affine":
        if a.shape[1] != 1 or x.shape != (a.shape[0],):
            raise ValueError("affine input shape differs")
        result = x.astype(np.float32)*np.float32(2) + np.float32(1)
    elif kind in ("packed_scan", "separate_scan"):
        if x.shape != (a.shape[1],):
            raise ValueError("scan query shape differs")
        result = np.zeros(a.shape[0], np.float32)
        for k in range(a.shape[1]):
            result = result + a[:, k].astype(np.float32)*np.float32(x[k])
    else:
        raise ValueError("unknown reference program")
    with np.errstate(over="ignore", invalid="ignore"):
        result = result.astype(np.float16)
    if not np.isfinite(result).all():
        raise ValueError("reference exceeds finite FP16 domain")
    return result
