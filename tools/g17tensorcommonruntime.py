"""Build and run the ordinary-runtime TensorOps image classes.

This is a deliberately small bridge between the ordinary compiler/linker path and the common
native worker.  It owns no compiler or metadata facts: the program supplies its ABI, scanlink
authors the image, and the worker rechecks the launch contract before creating a Metal pipeline.
The original one-GEMM-plus-scalar class remains the default; ``--composition multigemm`` selects
the released memory-fed chain, ``multigemm3`` selects the released three-body extension, and
``multigemm_relu_vec`` selects the released nonlinear-vector/residual composition.
"""
from __future__ import annotations

import argparse
import math
import os
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_identity():
    """Record the source tree that produced this bundle and worker."""
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    files = ("agxforge/g17/cc.py", "agxforge/g17/mdgen.py", "agxforge/g17/scanlink.py",
             "agxforge/g17/tensormetadata.py", "agxforge/g17/runtime.py",
             "agxforge/g17/ir.py", "agxforge/g17/tensorreduce.py", "agxforge/g17/tensor.py",
             "agxforge/g17/tensorgemm.py", "agxforge/g17/registerdomain.py",
             "tools/g17commonworker.m", "tools/g17tensorcommonruntime.py")
    return {"commit": commit, "files": {name: sha((ROOT / name).read_bytes()) for name in files}}


def build_program():
    """Compile one composed IR function through the ordinary compiler selector."""
    from agxforge.g17 import cc, ir

    a = ir.Buffer("A", 1, elem=ir.F16)
    b = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_runtime_demo", [a, b, c])
    builder = ir.Builder(fn, fn.block("entry"))
    builder.tensor_matmul(a, b, c, M=17, N=19, K=16)
    position = builder.builtin("threadgroup_position_in_grid", name="group_x")
    loaded = builder.load(c, position, type=ir.F32, name="c_value")
    # Float constants in this IR are IEEE bit patterns. Passing integer 1 would add the tiny
    # denormal 0x00000001 and make the purported scalar epilogue a no-op for these inputs.
    one = builder.const(struct.unpack("<I", struct.pack("<f", 1.0))[0])
    value = builder.fadd(loaded, one, name="c_plus_one")
    builder.store_at(c, position, value)
    builder.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def build_multigemm_program(composition="multigemm"):
    """Compile a measured half/half -> scalar -> float/half memory-fed chain.

    ``multigemm`` and ``multigemm_fadd_fmul`` are released two-body classes. ``multigemm3``
    adds a third float/half GEMM after a second ordinary scalar transform. It keeps the same
    three bindings and measured metadata class, and both tensor boundaries are memory bridges.
    """
    from agxforge.g17 import cc, ir

    a = ir.Buffer("A", 1, elem=ir.F16)
    b = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn_name = ({"multigemm": "tensor_multigemm_runtime_demo",
                "multigemm_fadd_fmul": "tensor_multigemm_fadd_fmul_runtime_demo",
                "multigemm3": "tensor_multigemm_fadd_fmul_gemm_runtime_demo",
                "multigemm_relu_vec": "tensor_multigemm_relu_vec_residual_runtime_demo"}[composition])
    fn = ir.Function(fn_name, [a, b, c])
    builder = ir.Builder(fn, fn.block("entry"))
    builder.tensor_matmul(a, b, c, M=32, N=32, K=64, a_dtype="half", b_dtype="half")
    position = builder.builtin("threadgroup_position_in_grid", name="group_x")
    if composition == "multigemm_relu_vec":
        values = builder.load_vec4_at(c, position, name="relu_input")
        zero = builder.const(struct.unpack("<I", struct.pack("<f", 0.0))[0])
        half = builder.const(struct.unpack("<I", struct.pack("<f", 0.5))[0])
        one = builder.const(struct.unpack("<I", struct.pack("<f", 1.0))[0])
        activated = []
        for index, value in enumerate(values):
            value = builder.fmax(value, zero, name="relu%d" % index)
            value = builder.fmul(value, half, name="relu_scale%d" % index)
            activated.append(builder.fadd(value, one, name="relu_shift%d" % index))
        builder.store_vec4_at(c, position, activated)
    else:
        loaded = builder.load(c, position, type=ir.F32, name="c_value")
        one = builder.const(struct.unpack("<I", struct.pack("<f", 1.0))[0])
        value = builder.fadd(loaded, one, name="c_plus_one")
        if composition in ("multigemm_fadd_fmul", "multigemm3"):
            half = builder.const(struct.unpack("<I", struct.pack("<f", 0.5))[0])
            value = builder.fmul(value, half, name="c_times_half")
        builder.store_at(c, position, value)
    builder.tensor_matmul(c, b, c, M=32, N=32, K=32, a_dtype="float", b_dtype="half")
    if composition in ("multigemm3", "multigemm_relu_vec"):
        position2 = builder.builtin("threadgroup_position_in_grid", name="group_x2")
        loaded2 = builder.load(c, position2, type=ir.F32, name="c_value2")
        # Reuse the already-live scalar constants. The three-body allocator reserves the tensor
        # register span, so introducing a second constant pair would exceed this measured
        # ordinary-runtime register class; the IR still contains two scalar regions and the
        # values are exactly the same constants at both boundaries.
        value2 = builder.fadd(loaded2, one, name="c_plus_one2")
        if composition == "multigemm_relu_vec":
            value2 = builder.fmul(value2, builder.const(struct.unpack("<I", struct.pack("<f", 0.5))[0]),
                                  name="residual_normalized")
        else:
            value2 = builder.fmul(value2, half, name="c_times_half2")
        builder.store_at(c, position2, value2)
        builder.tensor_matmul(c, b, c, M=32, N=32, K=32, a_dtype="float", b_dtype="half")
    builder.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


# ---- gemm_generic: one rule-based single-GEMM class (Set A with Set C) --------------------------
GENERIC_NAME = "tensor_gemm_generic_runtime_demo"
GENERIC_TYPES = {"half": 2, "bfloat": 2, "float": 4, "fp8e4m3": 1, "fp8e5m2": 1, "int8": 1}
FP8_OUT = {"fp8e4m3": "e4m3fn", "fp8e5m2": "e5m2"}      # epilogue word -> tlower's pack format
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


# EVERY KEY generic_spec READS (its normalised output's keys, plus author_generic's verification-only
# unpin_16x32x64). MM P13: this normaliser used to read the keys it knew with .get and DROP the rest, so a
# misspelt or unimplemented request ("transpose_a", "nsg": 8, "c": "half") built silently as some other
# admitted program. Any other key is now refused by name before anything is built.
GENERIC_SPEC_KEYS = frozenset((
    "M", "N", "K", "a", "b", "simdgroups", "threadgroups", "grid_n", "split_k", "epilogue", "split_fp32", "stages", "feed_model",
    "reduce", "reduce_model", "guard", "accumulate", "saturate", "imageblock", "between", "independent",
    "independent_wrong_a", "ib_op1", "kloop", "stage_through", "feed", "gelu_stage", "stream", "stream_model",
    "stream_q_scale", "stream_k2_scale", "stream_program", "transA", "transB", "mx_check_shift", "mx_range",
    "check_as", "check_input", "c", "unpin_16x32x64",
    # production row P2 (25.125): the accumulator feed and its reference control
    "c_feed", "c_feed_model",
    # production row P5 (25.128): in-kernel E8M0 codes, fp8 nonfinite inputs, the K-loop check and their models
    "check_k", "fp8_nonfinite", "mx_code_model", "mx_inject", "nonfinite_model",
    # production row P7 n key blocks (25.114.3): register-held K/V offsets and the frozen-advance control
    "key_blocks", "key_advance",
    # MM 25.124.6: the K loop's body holds kloop_unroll slices (1 or 2), every load before the first MMA
    "kloop_unroll",
    # MM 25.144.1: loop-carried fragment row bases in the K loop (tlower kloop_bases)
    "kloop_bases"))
# the failing controls' knob (reference only; the program is unchanged): score the output against a
# WRONG reading. "roll_a": A's rows rolled by one tile row (a wrong input); "untransposed": the stored
# operands read as if untransposed; "ldb128": B read with a 128-element row stride (the old N cap);
# "sg_fold4": the rows of simdgroups 4..7 left at the zero seed (what the old and16 mask of 3 computes);
# "half_rz": the half narrowing rounded toward zero instead of RNE; "fp32_out": C read as fp32 D
GENERIC_CHECK_INPUTS = (None, "roll_a", "untransposed", "ldb128", "sg_fold4", "half_rz", "fp32_out")


def generic_spec(spec):
    """Normalise and check a generic spec: {M, N, K, a, b, simdgroups, threadgroups, epilogue}."""
    # MM P12: this normaliser reads known keys with .get and drops the rest, so a sharing request
    # would otherwise build silently as the tile split. Refuse it by name first, before any build.
    from agxforge.g17 import runtime as _runtime
    _runtime.refuse_sharing(spec)
    # MM P7: an attention request is its own named class with its own keys (runtime.attention_spec)
    if spec.get("attention") is not None:
        return _attention_generic_spec(spec)
    unknown = sorted(str(k) for k in spec if k not in GENERIC_SPEC_KEYS)
    if unknown:
        raise ValueError("refused: gemm_generic spec key(s) %s are not in the admitted class table (MM P13, "
                         "docs/g17-tensorops-machine-model.md section 25.127); an unknown key is a request "
                         "this compiler would otherwise drop and build as something else" % ", ".join(unknown))
    s = dict(M=int(spec["M"]), N=int(spec["N"]), K=int(spec["K"]), a=spec.get("a", "half"),
             b=spec.get("b", "half"), simdgroups=int(spec.get("simdgroups", 1)),
             threadgroups=int(spec.get("threadgroups", 1)), grid_n=int(spec.get("grid_n", 1)), split_k=int(spec.get("split_k", 1)), epilogue=list(spec.get("epilogue", [])),
             split_fp32=bool(spec.get("split_fp32", False)),
             stages=[list(st) for st in spec.get("stages", [])],
             # reference only, and the program is unaffected. "identity" (the default) reads a fed
             # operand as the logical D or D^T, WITHOUT section 132's relabeling: that is what the
             # hardware computed for this compiler's feeds (results/g17-tensor-feedmodes-v1: every
             # identity arm bit-exact in B, At and Bt, fp32 and half). "measured" is section 132's
             # rotl1/rotr1 model, a property of kernels whose loads put B at pos_b (Apple's
             # simdgroup_matrix API); the name is kept so the v1 receipts reproduce, and it is now the
             # rival (every such arm failed by 290 to 477).
             feed_model=spec.get("feed_model", "identity"),
             # goal item 8: the GEMM's column sums or maxima, reduced in registers into C row 0
             reduce=spec.get("reduce"),
             # reference only: "exact" rounds an exact column sum once instead of the lane tree (a
             # control that the reference sees the emitted order)
             reduce_model=spec.get("reduce_model", "lanes"),
             # the split body's epilogue guard, for the discriminating arms: which simdgroup adds the
             # +1, and whether a threadgroup barrier precedes it
             guard=spec.get("guard", "sg0"),
             # int8 only (Set A item 4): C is seeded from c.f32 (int32 words) and D = C + A.B, either
             # wrapping (tlower's iadd) or saturating at every issue (mmaenc's operand-2 = 41)
             accumulate=bool(spec.get("accumulate", False)), saturate=bool(spec.get("saturate", False)),
             # Set A item 12: the GEMM's first 32 outputs staged through an explicit imageblock
             # ("stage"), or the same program with the declaration withheld ("undeclared", the
             # control: an undeclared imageblock allocates nothing and reads 0)
             imageblock=spec.get("imageblock"),
             # fusion item 6: n independent bodies over shared buffers, a batched GEMM - body k reads
             # A rows [kM/n, (k+1)M/n) through an A offset, all of B, and writes those C rows through
             # a C offset. "independent_wrong_a" is the control: every body reads A from row 0.
             # fusion item 6: scalar work BETWEEN chain bodies - "add1" adds 1.0 to C[0,0] after
             # stage 0, through memory, before the next body reads C as its A
             between=spec.get("between"),
             independent=int(spec.get("independent", 0)), independent_wrong_a=bool(spec.get("independent_wrong_a", False)),
             # item 12 round 8: the imageblock access's operand 1 as Apple's tensor kernel carries it
             ib_op1=spec.get("ib_op1"),
             # performance item 2: a runtime K loop instead of unrolling K (tlower kloop)
             kloop=bool(spec.get("kloop", False)), kloop_unroll=int(spec.get("kloop_unroll", 1)),
             kloop_bases=bool(spec.get("kloop_bases", False)),
             # Set A item 10b: the chain's D fragment crosses the body boundary through the explicit
             # imageblock (agxforge.g17.ibstage) - "imageblock", or "imageblock_noread", the control
             # that stores and clobbers the registers but never reads them back
             stage_through=spec.get("stage_through"),
             # "memory": build with cc.TENSOR_REGISTER_FEED off, so the chain boundary is the memory
             # bridge (the timing comparison for item 10b); None keeps the register feed
             feed=spec.get("feed"),
             # goal item 6: "memory" builds the SAME GELU as the released memory stage
             # (tensor_tile_gelu over the stored 16x16 tile) instead of the register epilogue step, the
             # comparison arm; the reference is the same function, so it is a build choice only
             gelu_stage=spec.get("gelu_stage"),
             # goal item 6: ONLINE-SOFTMAX ATTENTION over two key blocks, rows 0..stream-1 of the first
             # tile (docs/g17-tensorops-machine-model.md 25.114). stream_model is the REFERENCE's claim, the
             # program is the same: "online" (the program), "no_alpha" (a control: the rescale by
             # alpha = exp2(m1 - m2) left out) or "first_block" (a control: O = softmax(S1) V1 only).
             # The two scales shape the drawn inputs so the second block's row max exceeds the first's.
             stream=int(spec.get("stream", 0)), stream_model=spec.get("stream_model", "online"),
             stream_q_scale=float(spec.get("stream_q_scale", 0.125)),
             stream_k2_scale=float(spec.get("stream_k2_scale", 2.0)),
             # "oneshot": the SAME inputs and regions, the program one-shot softmax over all 32 keys
             # (S1 and S2 first, one max over both halves, no rescale) - the hardware comparison for
             # the online program's equivalence claim
             stream_program=spec.get("stream_program", "online"),
             # P7 n key blocks (machine model 25.114.3): online softmax over key_blocks blocks of 16 keys,
             # every K and V address taken from a register the stream advances per block (0 = the
             # released two-block program with immediate offsets). key_advance "frozen" is the failing
             # program control: the same bodies with the advance left out, so every block reads block 0.
             key_blocks=int(spec.get("key_blocks", 0)), key_advance=spec.get("key_advance", "register"),
             # production row P2 (machine model 25.125): THE ACCUMULATOR FEED. Two bodies of M x N x K/2:
             # C = A1 B1, then C += A2 B2 onto the first body's D, A1/A2 the two halves of A's M*K
             # halves and B1/B2 the two halves of B's K*N (byte offsets on the second body). c_feed_model
             # is the REFERENCE's claim, the program is the same: "accumulate" (the program) or "no_c"
             # (a control: the second body without the first's D)
             c_feed=bool(spec.get("c_feed", False)), c_feed_model=spec.get("c_feed_model", "accumulate"),
             # MM P13's transposed class: A stored as A^T (K x M), B as B^T (N x K); tlower's transpose bits
             transA=bool(spec.get("transA", False)), transB=bool(spec.get("transB", False)),
             check_input=spec.get("check_input"))
    # MM P13: the admitted-class table (agxforge.g17.runtime.GENERIC_CLASSES) names every widening class and
    # every refused one; a request in a refused class, outside an admitted class's receipted rule, or in
    # two widening classes at once is refused by that name here, before any other rule or any build
    from agxforge.g17 import runtime as _runtime
    _runtime.refuse_generic_class(dict(
        a=s["a"], b=s["b"], c=spec.get("c"), M=s["M"], N=s["N"], K=s["K"], simdgroups=s["simdgroups"],
        groups=s["threadgroups"], grid_n=s["grid_n"], split_k=s["split_k"], transA=s["transA"], transB=s["transB"], epilogue=s["epilogue"],
        stages=bool(s["stages"]), kloop=s["kloop"],
        extras=[k for k in ("reduce", "imageblock", "stream", "independent", "between", "split_fp32", "gelu_stage",
                            "stage_through", "feed") if s[k]]))
    # the manifest (runtime.TensorSpec) and the worker admit split-K at 1, 2 or 4 simdgroups only; tlower would
    # compile 8, so refuse it here by name before any build (M6's fuzzer: compiled, then the manifest refused it)
    if s["split_k"] > 1 and s["simdgroups"] not in (1, 2, 4):
        raise ValueError("refused: gemm_generic: split_k combines with 1, 2 or 4 simdgroups")
    # "c", when given, must name the accumulator the operands imply (a 16-bit one was refused above by name)
    if spec.get("c") not in (None, "int" if s["a"] == "int8" else "float"):
        raise ValueError("generic: c is float (int for int8 operands); a %r accumulator is not a gemm_generic class"
                         % (spec.get("c"),))
    if s["check_input"] not in GENERIC_CHECK_INPUTS:
        raise ValueError("generic: check_input is one of %s" % (GENERIC_CHECK_INPUTS[1:],))
    # GELU (goal item 6): the register epilogue step is the last one, on a plain GEMM, so the
    # element envelope (generic_abs_bound) is the GELU's alone
    if "gelu" in s["epilogue"] and (s["epilogue"][-1] != "gelu" or s["epilogue"].count("gelu") != 1
                                    or "mx32" in s["epilogue"] or "mx32e8m0" in s["epilogue"] or s["stages"] or s["reduce"] or s["imageblock"]
                                    or s["split_fp32"] or s["accumulate"] or s["a"] not in ("half", "bfloat")):
        raise ValueError("generic: gelu is the last epilogue step of a plain 16-bit GEMM")
    if s["key_blocks"] and (not 2 <= s["key_blocks"] <= 16 or s["stream_program"] != "online" or not s["stream"]
                            or s["K"] != keyblock_spec_k(s["key_blocks"])):
        raise ValueError("generic: key_blocks is 2..16 blocks of the online program, with K = max(64, 16 x "
                         "key_blocks) so that B holds every block's K^T and V")
    if s["key_advance"] not in ("register", "frozen") or (s["key_advance"] != "register" and not s["key_blocks"]):
        raise ValueError("generic: key_advance is register or frozen, and needs key_blocks")
    if s["stream"] and (not 1 <= s["stream"] <= 16 or (s["M"], s["N"], 64 if s["key_blocks"] else s["K"], s["a"], s["b"]) != (32, 80, 64, "half", "half")
                        or (s["simdgroups"], s["threadgroups"]) != (1, 1) or s["epilogue"] or s["stages"]
                        or s["reduce"] or s["imageblock"] or s["split_fp32"] or s["accumulate"] or s["kloop"]
                        or s["independent"] or s["between"]):
        raise ValueError("generic: online-softmax attention is the 32x80x64 half layout (Q 32x64; keys, values "
                         "and regions at fixed offsets), one simdgroup and threadgroup, rows 1..16 (the row "
                         "stages' measured range, rows 0..15 of the first tile)")
    if s["stream_program"] not in ("online", "oneshot"):
        raise ValueError("generic: stream_program is online or oneshot")
    if s["stream_program"] == "oneshot" and not s["stream"]:
        raise ValueError("generic: stream_program oneshot needs stream rows")
    if s["key_blocks"] and s["stream_model"] not in ("online", "frozen"):
        raise ValueError("generic: a key_blocks program's claims are online and frozen")
    if not s["key_blocks"] and s["stream_model"] not in ({"online": ("online", "no_alpha", "first_block"),
                                  "oneshot": ("oneshot", "oneshot_no_softmax", "oneshot_sixteen_keys")}
                                 [s["stream_program"]]):
        raise ValueError("generic: stream_model %r is not a claim about the %s program"
                         % (s["stream_model"], s["stream_program"]))
    if s["gelu_stage"] not in (None, "memory"):
        raise ValueError("generic: gelu_stage is memory or absent")
    if s["gelu_stage"] and (s["epilogue"] != ["gelu"] or (s["M"], s["N"], s["simdgroups"], s["threadgroups"]) != (16, 16, 1, 1)):
        raise ValueError("generic: the memory-stage GELU arm is the measured 16x16 tile, one simdgroup and threadgroup, epilogue [gelu]")
    if s["feed"] not in (None, "memory"):
        raise ValueError("generic: feed is memory or absent")
    if s["c_feed_model"] not in ("accumulate", "no_c") or (s["c_feed_model"] != "accumulate" and not s["c_feed"]):
        raise ValueError("generic: c_feed_model is accumulate or no_c, and needs c_feed")
    if s["c_feed"] and ((s["a"], s["b"], s["simdgroups"], s["threadgroups"]) != ("half", "half", 1, 1) or s["epilogue"]
                        or s["stages"] or s["reduce"] or s["imageblock"] or s["split_fp32"] or s["accumulate"]
                        or s["kloop"] or s["independent"] or s["stream"] or s["between"] or s["M"] % 16
                        or s["N"] % 16 or s["K"] % 32 or not 2 <= (s["K"] // 2) * s["N"] * 2 <= 47104):
        raise ValueError("generic: the accumulator feed is two half x half bodies of M x N x K/2, whole tiles, "
                         "one simdgroup and threadgroup, no other feature, B's second half inside the stream "
                         "offset domain")
    if s["feed"] and (not (s["stages"] or s["c_feed"]) or s["stage_through"]):
        raise ValueError("generic: the memory feed is a chain arm without imageblock staging")
    if s["stage_through"] not in (None, "imageblock", "imageblock_noread"):
        raise ValueError("generic: stage_through is imageblock or imageblock_noread")
    if s["stage_through"] and (len(s["stages"]) != 1 or len(s["stages"][0]) != 3 or s["stages"][0][2] != "float" or
                               (s["simdgroups"], s["threadgroups"]) != (1, 1)):
        raise ValueError("generic: imageblock staging is one fp32-fed stage, one simdgroup, one threadgroup")
    if not (s["ib_op1"] in (None, "apple", "apple_read") or
            (isinstance(s["ib_op1"], str) and s["ib_op1"].startswith("store:0x"))):
        raise ValueError("generic: ib_op1 is apple, apple_read, store:0x<hex> or absent")
    if s["imageblock"] not in (None, "stage", "undeclared", "neighbour", "neighbour_undeclared", "neighbour_open",
                               "x_own", "x_neighbour", "x_neighbour_undeclared", "x_neighbour_early"):
        raise ValueError("generic: unknown imageblock arm %r" % s["imageblock"])
    # FEED MODES (recon section 132 part 3): a stage may name its mode as a fourth element, "A" (the
    # default), "B", "At" or "Bt". The non-A modes are one square stage (M = N = stage N = stage K), so
    # every buffer keeps the mode-A chain's size and stride; their outputs are relabeled or shaped
    # differently, which a third body would have to know.
    for st in s["stages"]:
        if len(st) == 4 and st[3] not in ("A", "B", "At", "Bt"):
            raise ValueError("generic: a stage's feed mode is A, B, At or Bt")
    if any(len(st) == 4 and st[3] != "A" for st in s["stages"]) and (
            len(s["stages"]) != 1 or not (s["M"] == s["N"] == s["stages"][0][0] == s["stages"][0][1])):
        raise ValueError("generic: a B, At or Bt feed is one square stage (M = N = stage N = stage K)")
    if s["reduce"] not in (None, "colsum", "colmax"):
        raise ValueError("generic: reduce is colsum or colmax")
    if s["reduce"] and ((s["simdgroups"], s["threadgroups"], s["a"], s["b"], s["epilogue"], s["stages"],
                         s["split_fp32"], s["imageblock"]) != (1, 1, "half", "half", [], [], False, None)
                        or s["M"] % 16 or s["N"] % 16):
        raise ValueError("generic: a register-resident reduction is one simdgroup, one threadgroup, "
                         "half x half, whole tiles, no epilogue, chain, split or imageblock")
    if s["imageblock"] and ((s["simdgroups"], s["threadgroups"], s["a"], s["b"], s["epilogue"], s["stages"],
                             s["split_fp32"]) != (1, 1, "half", "half", [], [], False) or s["N"] < 32):
        raise ValueError("generic: the imageblock stage is one simdgroup, one threadgroup, half x half, "
                         "no epilogue or chain, N >= 32")
    if s["a"] not in GENERIC_TYPES or s["b"] not in GENERIC_TYPES:
        raise ValueError("generic: operand types are %s" % sorted(GENERIC_TYPES))
    if (s["a"] == "int8") != (s["b"] == "int8"):
        raise ValueError("generic: int8 operands are paired")
    if s["a"] == "int8" and (s["epilogue"], s["split_fp32"], s["stages"]) != ([], False, []):
        raise ValueError("generic: int8 is one simdgroup, one threadgroup, no epilogue, no chain")
    if (s["accumulate"] or s["saturate"]) and s["a"] != "int8":
        raise ValueError("generic: accumulate and saturate are int8 only")
    # G5 (MM 25.144.1): the K loop runs in 1, 2, 4 or 8 simdgroups, each owning whole row tiles (tlower lowers the
    # simdgroup split with the loop; each simdgroup runs the same body on its own rows)
    if s["kloop"] and (s["stages"] or s["imageblock"] or s["split_fp32"] or s["simdgroups"] not in (1, 2, 4, 8)):
        raise ValueError("generic: kloop is one plain GEMM body in 1, 2, 4 or 8 simdgroups")
    if s["kloop_unroll"] != 1 and (not s["kloop"] or s["kloop_unroll"] not in (2, 3, 4)):
        raise ValueError("generic: kloop_unroll is 2, 3 or 4 on a kloop body (MM 25.124.6, 25.144.1), or absent")
    if s["K"] > 256 and not s["kloop"]:
        raise ValueError("generic: K above 256 needs kloop (the unrolled body's displacements overflow)")
    # Set A item 9(b): an fp8 quantize-out is the LAST epilogue step; C then holds M*N fp8 bytes
    # (row-major, N bytes per row) followed by the untouched zero seed. One threadgroup would run
    # the C[0,0] += 1 tail over four packed bytes read as an fp32, so fp8 out needs a grid split.
    fp8 = [st for st in s["epilogue"] if st in FP8_OUT]
    if fp8 and (s["epilogue"][-1] not in FP8_OUT or len(fp8) != 1 or s["threadgroups"] < 2 or s["stages"]
                or s["simdgroups"] != 1 or s["imageblock"] or s["split_fp32"]):
        raise ValueError("generic: fp8 quantize-out is one final epilogue step on a grid-split single GEMM")
    # Set A item 9(a): "mx32" as the FIRST epilogue word is OCP MX block scaling (one power-of-two
    # factor per 32-element K block, per A row and per B column), post-MMA in fp32 (tlower's mx
    # step). The predecoded fp32 tables follow A in a.f16 (SA[b][row]) and B in b.f16 (SB[b][col]).
    # Production row P5 (machine model 25.128): "mx32e8m0" is the same step with the E8M0 CODE BYTES
    # transported ((K/32) x M bytes after A, (K/32) x N after B) and decoded in the kernel. It is the
    # only form admitted inside the K loop.
    mxw = [st for st in s["epilogue"] if st in ("mx32", "mx32e8m0")]
    if mxw and (s["epilogue"][0] != mxw[0] or len(mxw) != 1 or s["K"] % 32 or s["stages"] or s["simdgroups"] != 1
                or s["imageblock"] or s["split_fp32"] or s["a"] in ("int8", "float") or s["b"] in ("int8", "float")):
        raise ValueError("generic: mx32 is the first epilogue step of a single 16-bit or fp8 GEMM with K a multiple of 32")
    if mxw == ["mx32"] and s["kloop"]:
        raise ValueError("generic: inside the K loop microscaling reads E8M0 codes (mx32e8m0)")
    # the special codes a bundle plants in its code tables (machine model 25.128), and the model the
    # REFERENCE decodes them with - a claim about the program, never a program change:
    #   code255: SA block 0 row 5 and SB block 1 column 20 are 255; "ocp" (the block is NaN, the
    #            program's claim) or "inf" (the naive e << 23 decode, a control)
    #   code0:   every block of SA rows 0..3 is 0, which production REFUSES (check_e8m0_codes); this
    #            research arm measures why: "flush" (the kernel's e << 23 is +0) or "exact" (OCP's
    #            2^-127, which no fp32 factor can carry on this ALU: the control)
    s["mx_inject"] = spec.get("mx_inject")
    s["mx_code_model"] = spec.get("mx_code_model", {"code0": "flush"}.get(s["mx_inject"], "ocp"))
    if s["mx_inject"] not in (None, "code255", "code0") or (s["mx_inject"] and mxw != ["mx32e8m0"]):
        raise ValueError("generic: mx_inject is code255 or code0, on an mx32e8m0 GEMM")
    if s["mx_code_model"] not in {None: ("ocp",), "code255": ("ocp", "inf"), "code0": ("flush", "exact")}[s["mx_inject"]]:
        raise ValueError("generic: mx_code_model %r is not a reading of %s" % (s["mx_code_model"], s["mx_inject"]))
    # fp8 INPUT nonfinite codes (machine model 25.128): every NaN and infinity code of both formats is
    # planted at fixed positions of A and B; nonfinite_model is the REFERENCE's reading of op17642's
    # unpack: "ocp" (NaN stays NaN, infinity stays infinity, IEEE through the MMA: the claim), or the
    # rivals "saturate" (to the largest finite magnitude) and "zero"
    s["fp8_nonfinite"] = bool(spec.get("fp8_nonfinite", False))
    s["nonfinite_model"] = spec.get("nonfinite_model", "ocp")
    if s["fp8_nonfinite"] and ((s["a"], s["b"]) not in (("fp8e4m3", "fp8e5m2"), ("fp8e5m2", "fp8e4m3"))
                               or s["epilogue"] or s["M"] < 32 or s["N"] < 32 or s["K"] < 32):
        raise ValueError("generic: fp8_nonfinite plants codes in an fp8 e4m3 x e5m2 GEMM of at least 32x32x32, no epilogue")
    if s["nonfinite_model"] not in ("ocp", "saturate", "zero") or (s["nonfinite_model"] != "ocp" and not s["fp8_nonfinite"]):
        raise ValueError("generic: nonfinite_model is ocp, saturate or zero, and needs fp8_nonfinite")
    # a K loop's failing control: score against the GEMM over the first check_k elements of K only
    # (the loop running short), the program unchanged
    s["check_k"] = spec.get("check_k")
    if s["check_k"] is not None and (not s["kloop"] or s["epilogue"] or not 16 <= int(s["check_k"]) < s["K"]
                                     or int(s["check_k"]) % 16):
        raise ValueError("generic: check_k is a whole-slice K below K, on a plain K-loop GEMM")
    # its failing control's knob: score against tables read one block late (block b uses b+1's factors)
    s["mx_check_shift"] = int(spec.get("mx_check_shift", 0))
    # the E8M0 code range drawn for the tables, as offsets from 127 (the author's knob, not the program's)
    s["mx_range"] = [int(v) for v in spec.get("mx_range", (-12, 12))]
    if s["mx_check_shift"] and not mxw:
        raise ValueError("generic: mx_check_shift needs mx32 or mx32e8m0")
    # the failing control's knob: score the program against the OTHER format's reference
    s["check_as"] = spec.get("check_as")
    if s["check_as"] is not None and (not fp8 or s["check_as"] not in FP8_OUT):
        raise ValueError("generic: check_as names an fp8 format and needs an fp8 quantize-out")
    return s


def generic_ulp_bound(s):
    """The comparison bound, in fp32 ulps, a generic spec's output is checked to: 0 (bit-exact) for
    every spec except those carrying an exp2 epilogue, whose hardware function is measured within
    one ulp of the exact value in both directions (not RNE), so the bound is that measurement."""
    return 1 if "exp2" in (s.get("epilogue") or []) else 0


def _ulp_distance(a, b):
    """|a - b| in fp32 ulps, via the monotone integer mapping of the bit patterns."""
    def key(x):
        i = np.asarray(x, dtype="<f4").view("<i4").astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)
    return np.abs(key(a) - key(b))


def generic_abs_bound(bundle, s):
    """A per-element absolute bound for specs whose output passes a transcendental step into a later
    MMA, else None. The one-dispatch attention's softmaxed rows reach the second GEMM as fp32 A,
    which the MMA truncates to 10 mantissa bits; the hardware exp2 (within one ulp) and the
    compiler's own summation order can move a P entry across one truncation step, so each output
    element on those rows is bounded by 2**-9 * sum_k |V[k, n]|. Every other row is bit-exact."""
    if "gelu" in (s.get("epilogue") or []):
        return generic_gelu_bound(bundle, s)
    if s.get("attention"):
        return attention_bound(bundle, s)
    if s.get("stream"):
        return stream_bound(bundle, s)
    between = str(s.get("between") or "")
    if not between.startswith("softmax:"):
        return None
    rows = int(between.split(":")[1])
    n1, k1, _t = s["stages"][0]
    v = _generic_values((bundle / "b.f16").read_bytes(), "half")[:k1 * n1].reshape(k1, n1)
    width = max([s["N"]] + [n for n, _k, _tt in s["stages"]])
    bound = np.zeros((s["M"], width), dtype=np.float64)
    bound[:rows, :n1] = 2.0 ** -9 * np.abs(v).sum(axis=0)
    return bound


GELU_ALPHA, GELU_NEG_INV_LN2 = np.float32(1.702), np.float32(-1.4426950408889634)


def _one_ulp(x, direction):
    """fp32 x moved one ulp toward +inf (direction 1) or -inf (-1); x itself for 0."""
    x = np.asarray(x, dtype="<f4")
    return np.where(direction == 0, x, np.nextafter(x, np.float32(np.inf) * direction).astype("<f4"))


def gelu_model(x, exp2_ulp=0, recip_ulp=0):
    """The GELU register epilogue (and the memory stage, tensorreduce.emit_row_gelu), instruction for
    instruction: t = x*1.702, t = t*(-1/ln 2), t = 2**t, t = t+1, t = 1/t, y = x*t. Every fmul and
    fadd is RNE fp32. exp2 and recip are the exact value rounded once, then moved `exp2_ulp` and
    `recip_ulp` ulps (-1, 0 or 1): the hardware exp2 is within one ulp both ways (docs/archive/
    g17-settle-20260923.md), and recip is ASSUMED within one ulp (no dense sweep settles it)."""
    # EVERY ALU RESULT FLUSHES A SUBNORMAL TO A SIGNED ZERO (recon section 138 part 5 for fmul and
    # fadd; recip measured here). The first version kept a subnormal recip: at x near -52, 1/(1+2^t)
    # is about 3.6e-39, the GPU returned -0 and the model x * 3.6e-39, for 57 of 65,536 elements
    # (results/g17-tensor-gelu-v1/gelu_M512N128K256_g8).
    def ftz(v):
        v = np.asarray(v, dtype="<f4").copy()
        tiny = (v != 0) & (np.abs(v) < np.float32(1.1754944e-38))
        v[tiny] = np.copysign(np.float32(0.0), v[tiny])
        return v
    x = np.asarray(x, dtype="<f4")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        t = ftz(x * GELU_ALPHA)
        t = ftz(t * GELU_NEG_INV_LN2)
        t = ftz(_one_ulp(np.exp2(t.astype(np.float64)).astype("<f4"), exp2_ulp))
        t = ftz(t + np.float32(1.0))
        t = ftz(_one_ulp((1.0 / t.astype(np.float64)).astype("<f4"), recip_ulp))
        return ftz(x * t)


def generic_gelu_bound(bundle, s):
    """The per-element envelope of a GELU arm, derived from the instruction model rather than chosen:
    over exp2 and recip each at -1, 0 and +1 ulp (nine combinations), the largest distance from the
    reference (both at 0). The one-threadgroup C[0,0] += 1 tail is applied to every combination."""
    M, N, K = s["M"], s["N"], s["K"]
    a = _generic_values((bundle / "a.f16").read_bytes()[:M * K * GENERIC_TYPES[s["a"]]], s["a"]).reshape(M, K)
    b = _generic_values((bundle / "b.f16").read_bytes()[:K * N * GENERIC_TYPES[s["b"]]], s["b"]).reshape(K, N)
    out = _gemm_mma(a, b, None, M, N, K)
    for step in s["epilogue"][:-1]:
        if step == "relu":
            out = np.maximum(out, np.float32(0.0)).astype("<f4")
        elif step.startswith("scale:"):
            out = np.asarray(out * np.frombuffer(struct.pack("<I", int(step.split(":")[1], 16)), dtype="<f4")[0], dtype="<f4")
        else:
            raise ValueError("generic: no envelope for epilogue step %r before gelu" % step)
    def tailed(y):
        y = y.copy()
        if s["threadgroups"] == 1 and s["grid_n"] == 1:
            y[0, 0] = _rne32(y[0, 0] + np.float32(1.0))
        return y.astype(np.float64)
    ref = tailed(gelu_model(out))
    bound = np.zeros_like(ref)
    for de in (-1, 0, 1):
        for dr in (-1, 0, 1):
            bound = np.maximum(bound, np.abs(tailed(gelu_model(out, de, dr)) - ref))
    return bound


def generic_unscored(s):
    """(row, column) outputs a spec leaves without a prediction: only the open-neighbour arm's lane 31."""
    return [(0, 31)] if s.get("imageblock") == "neighbour_open" else []


# ONLINE-SOFTMAX ATTENTION (goal item 6, docs/g17-tensorops-machine-model.md 25.114). Byte layout of the
# stream arm's buffers: B holds the two key blocks transposed (64 x 16 halves each) and the two value
# blocks (16 x 16 halves each); C holds five 32 x 16 fp32 regions - the two score tiles (overwritten
# in place by P1 and P2), the output O, and the per-row stats tiles for the running max m and sum l.
STREAM_B = {"K1": 0, "K2": 2048, "V1": 4096, "V2": 4608}
STREAM_C = {"S1": 0, "S2": 2048, "O": 4096, "M": 6144, "L": 8192}


def _build_oneshot_attention(builder, a, b, c, rows):
    """S1 = Q K1; S2 = Q K2; per row one-shot softmax over S1 | S2 (one max, P1 and P2 in place, l);
    O = P1 V1; O += P2 V2; per row O /= l. Same buffers and regions as the online program."""
    from agxforge.g17 import tensorreduce as TR
    e = {k: v // 4 for k, v in STREAM_C.items()}
    builder.tensor_matmul(a, b, c, M=32, N=16, K=64, offsetB=STREAM_B["K1"], offsetC=STREAM_C["S1"])
    builder.tensor_matmul(a, b, c, M=32, N=16, K=64, offsetB=STREAM_B["K2"], offsetC=STREAM_C["S2"])
    for r in range(rows):
        TR.emit_oneshot_softmax(builder, c, row=r, s1_base=e["S1"], s2_base=e["S2"], l_base=e["L"])
    builder.tensor_matmul(c, b, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half",
                          offsetA=STREAM_C["S1"], offsetB=STREAM_B["V1"], offsetC=STREAM_C["O"])
    builder.tensor_matmul(c, b, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=True,
                          offsetA=STREAM_C["S2"], offsetB=STREAM_B["V2"], offsetC=STREAM_C["O"])
    for r in range(rows):
        TR.emit_stream_normalize(builder, c, row=r, o_base=e["O"], l_base=e["L"])


def _build_stream_attention(builder, a, b, c, rows):
    """S1 = Q K1; per row m1, P1, l1; O = P1 V1; S2 = Q K2; per row m2, alpha, P2, l and O *= alpha;
    O += P2 V2; per row O /= l. Four tensor bodies on the memory-stream route, straight-line code."""
    from agxforge.g17 import tensorreduce as TR
    e = {k: v // 4 for k, v in STREAM_C.items()}
    builder.tensor_matmul(a, b, c, M=32, N=16, K=64, offsetB=STREAM_B["K1"], offsetC=STREAM_C["S1"])
    for r in range(rows):
        TR.emit_stream_first_block(builder, c, row=r, s_base=e["S1"], m_base=e["M"], l_base=e["L"])
    builder.tensor_matmul(c, b, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half",
                          offsetA=STREAM_C["S1"], offsetB=STREAM_B["V1"], offsetC=STREAM_C["O"])
    builder.tensor_matmul(a, b, c, M=32, N=16, K=64, offsetB=STREAM_B["K2"], offsetC=STREAM_C["S2"])
    for r in range(rows):
        TR.emit_stream_second_block(builder, c, row=r, s_base=e["S2"], o_base=e["O"], m_base=e["M"],
                                    l_base=e["L"])
    builder.tensor_matmul(c, b, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=True,
                          offsetA=STREAM_C["S2"], offsetB=STREAM_B["V2"], offsetC=STREAM_C["O"])
    for r in range(rows):
        TR.emit_stream_normalize(builder, c, row=r, o_base=e["O"], l_base=e["L"])


# P7 N KEY BLOCKS (machine model 25.114.3). B holds every block's K^T (64 x 16 halves, 2048 bytes)
# back to back, then every block's V (16 x 16 halves, 512 bytes): at two blocks exactly STREAM_B.
# C keeps STREAM_C's regions; block j's scores go to S1 (j even) or S2 (j odd), so at two blocks the
# program writes exactly the released program's regions.
def keyblock_k(j):
    return 2048 * j


def keyblock_v(j, nblocks):
    return 2048 * nblocks + 512 * j


def keyblock_spec_k(nblocks):
    """The spec's K for n blocks: b.f16 holds K x 80 halves, at least 2560 bytes per block."""
    return max(64, 16 * nblocks)


def _build_keyblock_attention(builder, a, b, c, rows, nblocks, frozen=False):
    """Online softmax over `nblocks` key blocks with every K and V address in a REGISTER: the score
    body reads B at offset 0 plus stream register "k", the value body at 2048 n plus register "v",
    and each advances its register by one block after it runs (2048 and 512 bytes). No key-block
    address is an immediate, so the score bodies of blocks j and j + 2 have the same bytes, as do
    the accumulating value bodies. frozen=True is the failing control: the advances are 0, so every
    block reads block 0's keys and values."""
    from agxforge.g17 import tensorreduce as TR
    e = {k: v // 4 for k, v in STREAM_C.items()}
    kstep, vstep = (0, 0) if frozen else (2048, 512)
    for j in range(nblocks):
        s_name = "S1" if j % 2 == 0 else "S2"
        builder.tensor_matmul(a, b, c, M=32, N=16, K=64, offsetB=0, offsetC=STREAM_C[s_name],
                              offsetB_register="k", offsetB_step=kstep)
        for r in range(rows):
            if j == 0:
                TR.emit_stream_first_block(builder, c, row=r, s_base=e[s_name], m_base=e["M"], l_base=e["L"])
            else:
                TR.emit_stream_second_block(builder, c, row=r, s_base=e[s_name], o_base=e["O"],
                                            m_base=e["M"], l_base=e["L"])
        builder.tensor_matmul(c, b, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=j > 0,
                              offsetA=STREAM_C[s_name], offsetB=keyblock_v(0, nblocks),
                              offsetC=STREAM_C["O"], offsetB_register="v", offsetB_step=vstep)
    for r in range(rows):
        TR.emit_stream_normalize(builder, c, row=r, o_base=e["O"], l_base=e["L"])


def _keyblock_operands(bundle, nblocks):
    q = _generic_values((bundle / "a.f16").read_bytes(), "half")[:32 * 64].reshape(32, 64)
    bh = _generic_values((bundle / "b.f16").read_bytes(), "half")
    blk = lambda off, rows: bh[off // 2: off // 2 + rows * 16].reshape(rows, 16)
    c0 = np.frombuffer((bundle / "c.f32").read_bytes(), dtype="<f4")
    return (q, [blk(keyblock_k(j), 64) for j in range(nblocks)],
            [blk(keyblock_v(j, nblocks), 16) for j in range(nblocks)], c0)


def _keyblock_trace(bundle, s, ar, model):
    """The C buffer the n-block program leaves, in the program's own order (block 0: the first-block
    stage; every later block: the second-block stage, which is the general online step). Claims:
    "online" (the program) and "frozen" (every block reads block 0: what the frozen control computes)."""
    rows, n = s["stream"], s["key_blocks"]
    q, ks, vs, c0 = _keyblock_operands(bundle, n)
    if model == "frozen":
        ks, vs = [ks[0]] * n, [vs[0]] * n
    C = [ar.val(x) for x in c0]
    e = {k: v // 4 for k, v in STREAM_C.items()}
    one_neg = ar.val(-1.0)
    for j in range(n):
        sb = e["S1" if j % 2 == 0 else "S2"]
        C[sb: sb + 512] = [ar.val(x) for x in _gemm_mma(q, ks[j], None, 32, 16, 64).ravel()]
        for r in range(rows):
            base = sb + 16 * r
            vals = C[base: base + 16]
            if j == 0:
                m = _stream_rowmax(ar, vals)
                neg = ar.fmul(m, one_neg)
                p = [ar.exp2(ar.fadd(x, neg)) for x in vals]
                C[base: base + 16] = p
                l = _stream_rowsum(ar, p)
            else:
                m1, l1 = C[e["M"] + 16 * r], C[e["L"] + 16 * r]
                m = ar.fmax(m1, _stream_rowmax(ar, vals))
                neg = ar.fmul(m, one_neg)
                alpha = ar.exp2(ar.fadd(m1, neg))
                p = [ar.exp2(ar.fadd(x, neg)) for x in vals]
                C[base: base + 16] = p
                l = ar.fadd(ar.fmul(alpha, l1), _stream_rowsum(ar, p))
                ob = e["O"] + 16 * r
                C[ob: ob + 16] = [ar.fmul(o, alpha) for o in C[ob: ob + 16]]
            C[e["M"] + 16 * r: e["M"] + 16 * r + 16] = [m] * 16
            C[e["L"] + 16 * r: e["L"] + 16 * r + 16] = [l] * 16
        C[e["O"]: e["O"] + 512] = ar.mma(C[sb: sb + 512], vs[j], None if j == 0 else C[e["O"]: e["O"] + 512])
    for r in range(rows):
        inv = ar.recip(C[e["L"] + 16 * r])
        ob = e["O"] + 16 * r
        C[ob: ob + 16] = [ar.fmul(o, inv) for o in C[ob: ob + 16]]
    C[0] = ar.fadd(C[0], ar.val(1.0))                  # the one-threadgroup C[0,0] += 1 tail
    return C


_FP32_TINY = 2.0 ** -126


def _f32(x):
    return float(np.float32(x))


def _ftz(x):
    return math.copysign(0.0, x) if x != 0 and abs(x) < _FP32_TINY else x


def _next32(x, toward):
    return float(np.nextafter(np.float32(x), np.float32(toward)))


def _rd32(x):
    r = _f32(x)
    return r if r <= x else _next32(r, -math.inf)


def _ru32(x):
    r = _f32(x)
    return r if r >= x else _next32(r, math.inf)


class _StreamPoint:
    """The reference: every fp32 ALU op rounds to nearest even and flushes subnormal inputs and
    results to a signed zero (recon section 138 part 5; recip too, MM section 25.109); exp2 and
    recip are the exact value rounded once; the MMA is the section-136 model (_gemm_mma)."""
    def val(self, x):   return _f32(x)
    def fadd(self, x, y):  return _ftz(_f32(_ftz(x) + _ftz(y)))
    def fmul(self, x, y):  return _ftz(_f32(_ftz(x) * _ftz(y)))
    def fmax(self, x, y):  return x if x >= y else y
    def exp2(self, x):  return _ftz(_f32(2.0 ** _ftz(x)))
    def recip(self, x):  return _ftz(_f32(1.0 / _ftz(x)))
    def mma(self, a_rows, v, c):
        out = _gemm_mma(np.asarray(a_rows, dtype="<f4").reshape(32, 16), v,
                        None if c is None else np.asarray(c, dtype="<f4").reshape(32, 16), 32, 16, 16,
                        truncate_a=True)
        return [float(x) for x in out.ravel()]
    def mid(self, x):   return x


class _StreamInterval:
    """[lo, hi] enclosures of every quantity. mode "hardware": the hardware's exp2 is within one ulp
    of the exact value (measured), recip within one ulp (assumed, as for GELU), each rounding is
    directed outward, and the MMA's fp32-A truncation is applied to both ends (it is monotone), with
    the MMA's own roundings covered by a slack of 6 * 2^-24 * sum|a||b| (at most five roundings on any
    path through one 16-wide issue, section 136). mode "exact": the enclosure also contains exact real
    arithmetic - the truncation widened to the hull of x and trunc(x) - so the exact attention must
    lie inside it (the one-shot equivalence check)."""
    def __init__(self, mode="hardware"):
        self.mode = mode
    def val(self, x):   return (_f32(x), _f32(x))
    def _iv(self, lo, hi):
        return (_ftz(_rd32(lo)), _ftz(_ru32(hi)))
    def fadd(self, x, y):
        return self._iv(_ftz(x[0]) + _ftz(y[0]), _ftz(x[1]) + _ftz(y[1]))
    def fmul(self, x, y):
        c = [_ftz(p) * _ftz(q) for p in x for q in y]
        return self._iv(min(c), max(c))
    def fmax(self, x, y):  return (max(x[0], y[0]), max(x[1], y[1]))
    def exp2(self, x):
        lo, hi = self._iv(2.0 ** _ftz(x[0]), 2.0 ** _ftz(x[1]))
        return (_ftz(_next32(lo, -math.inf)), _ftz(_next32(hi, math.inf)))
    def recip(self, x):
        if x[0] <= 0:
            raise ValueError("stream model: a row sum enclosure reaches zero")
        lo, hi = self._iv(1.0 / x[1], 1.0 / x[0])
        return (_ftz(_next32(lo, -math.inf)), _ftz(_next32(hi, math.inf)))
    def _trunc(self, x):
        t = (_truncate_fp32(x[0]), _truncate_fp32(x[1]))
        if self.mode == "exact":
            return (min(t[0], x[0]), max(t[1], x[1]))
        return t
    def mma(self, a_rows, v, c):
        out = []
        for m in range(32):
            row = [self._trunc(a_rows[16 * m + k]) for k in range(16)]
            degenerate = all(p[0] == p[1] for p in row) and (c is None or c[16 * m][0] == c[16 * m][1])
            for n in range(16):
                cc = None if c is None else c[16 * m + n]
                if degenerate and (cc is None or cc[0] == cc[1]):
                    x = _mma16([p[0] for p in row], [float(v[k, n]) for k in range(16)], None)
                    x = x if cc is None else _f32(np.float32(x) + np.float32(cc[0]))
                    out.append((float(x), float(x)))
                    continue
                lo = sum(min(p[0] * float(v[k, n]), p[1] * float(v[k, n])) for k, p in enumerate(row))
                hi = sum(max(p[0] * float(v[k, n]), p[1] * float(v[k, n])) for k, p in enumerate(row))
                mag = sum(max(abs(p[0]), abs(p[1])) * abs(float(v[k, n])) for k, p in enumerate(row))
                slack = 6 * 2.0 ** -24 * mag
                lo, hi = lo - slack, hi + slack
                if cc is not None:
                    lo, hi = lo + cc[0], hi + cc[1]
                    slack = 2.0 ** -24 * (abs(lo) + abs(hi))
                    lo, hi = lo - slack, hi + slack
                out.append(self._iv(lo, hi))
        return out
    def mid(self, x):   return x


def _stream_rowsum(ar, vals):
    """((v0+v1)+v2)+v3 per lane over its four columns, then the row butterfly: (g0+g4)+(g8+g12)."""
    g = []
    for col in (0, 4, 8, 12):
        acc = vals[col]
        for k in range(1, 4):
            acc = ar.fadd(acc, vals[col + k])
        g.append(acc)
    return ar.fadd(ar.fadd(g[0], g[1]), ar.fadd(g[2], g[3]))


def _stream_rowmax(ar, vals):
    g = []
    for col in (0, 4, 8, 12):
        acc = vals[col]
        for k in range(1, 4):
            acc = ar.fmax(acc, vals[col + k])
        g.append(acc)
    return ar.fmax(ar.fmax(g[0], g[1]), ar.fmax(g[2], g[3]))


def _stream_operands(bundle):
    a = _generic_values((bundle / "a.f16").read_bytes(), "half").reshape(32, 64)
    bh = _generic_values((bundle / "b.f16").read_bytes(), "half")
    blk = lambda off, rows: bh[off // 2: off // 2 + rows * 16].reshape(rows, 16)
    c0 = np.frombuffer((bundle / "c.f32").read_bytes(), dtype="<f4")
    return a, blk(STREAM_B["K1"], 64), blk(STREAM_B["K2"], 64), blk(STREAM_B["V1"], 16), blk(STREAM_B["V2"], 16), c0


def _oneshot_trace(bundle, s, ar, model):
    """The C buffer the ONE-SHOT program leaves, under arithmetic `ar`. Claims: "oneshot" (the
    program); controls "oneshot_no_softmax" (O = S1 V1 + S2 V2, raw scores) and "oneshot_sixteen_keys"
    (O = softmax over S1 alone, times V1) - both change only O's softmaxed rows."""
    rows = s["stream"]
    q, k1, k2, v1, v2, c0 = _stream_operands(bundle)
    C = [ar.val(x) for x in c0]
    e = {k: v // 4 for k, v in STREAM_C.items()}
    def region(name):
        return C[e[name]: e[name] + 512]
    def put(name, flat):
        C[e[name]: e[name] + 512] = flat
    put("S1", [ar.val(x) for x in _gemm_mma(q, k1, None, 32, 16, 64).ravel()])
    put("S2", [ar.val(x) for x in _gemm_mma(q, k2, None, 32, 16, 64).ravel()])
    raw1, raw2 = list(region("S1")), list(region("S2"))
    one_neg = ar.val(-1.0)
    for r in range(rows):
        b1, b2 = e["S1"] + 16 * r, e["S2"] + 16 * r
        x1, x2 = C[b1: b1 + 16], C[b2: b2 + 16]
        m = ar.fmax(_stream_rowmax(ar, x1), _stream_rowmax(ar, x2))
        neg = ar.fmul(m, one_neg)
        p1 = [ar.exp2(ar.fadd(x, neg)) for x in x1]
        p2 = [ar.exp2(ar.fadd(x, neg)) for x in x2]
        C[b1: b1 + 16], C[b2: b2 + 16] = p1, p2
        l = ar.fadd(_stream_rowsum(ar, p1), _stream_rowsum(ar, p2))
        C[e["L"] + 16 * r: e["L"] + 16 * r + 16] = [l] * 16
    put("O", ar.mma(region("S1"), v1, None))
    put("O", ar.mma(region("S2"), v2, region("O")))
    for r in range(rows):
        inv = ar.recip(C[e["L"] + 16 * r])
        ob = e["O"] + 16 * r
        C[ob: ob + 16] = [ar.fmul(o, inv) for o in C[ob: ob + 16]]
    if model == "oneshot_no_softmax":
        o = ar.mma(raw2, v2, ar.mma(raw1, v1, None))
        for r in range(rows):
            C[e["O"] + 16 * r: e["O"] + 16 * r + 16] = o[16 * r: 16 * r + 16]
    elif model == "oneshot_sixteen_keys":
        p = list(raw1)
        for r in range(rows):
            x1 = raw1[16 * r: 16 * r + 16]
            neg = ar.fmul(_stream_rowmax(ar, x1), one_neg)
            p[16 * r: 16 * r + 16] = [ar.exp2(ar.fadd(x, neg)) for x in x1]
        o = ar.mma(p, v1, None)
        for r in range(rows):
            inv = ar.recip(_stream_rowsum(ar, p[16 * r: 16 * r + 16]))
            C[e["O"] + 16 * r: e["O"] + 16 * r + 16] = [ar.fmul(x, inv) for x in o[16 * r: 16 * r + 16]]
    C[0] = ar.fadd(C[0], ar.val(1.0))                  # the one-threadgroup C[0,0] += 1 tail
    return C


def _stream_trace(bundle, s, ar, model=None):
    """The C buffer the stream program leaves, under arithmetic `ar`, as a flat list of 2,560 values
    (points or enclosures). `model` is the reference's claim (s["stream_model"] by default)."""
    model = model or s["stream_model"]
    if s.get("key_blocks"):
        return _keyblock_trace(bundle, s, ar, model)
    if s.get("stream_program", "online") == "oneshot":
        return _oneshot_trace(bundle, s, ar, model)
    rows = s["stream"]
    q, k1, k2, v1, v2, c0 = _stream_operands(bundle)
    C = [ar.val(x) for x in c0]
    e = {k: v // 4 for k, v in STREAM_C.items()}
    def region(name):
        return C[e[name]: e[name] + 512]
    def put(name, flat):
        C[e[name]: e[name] + 512] = flat
    put("S1", [ar.val(x) for x in _gemm_mma(q, k1, None, 32, 16, 64).ravel()])
    one_neg = ar.val(-1.0)
    for r in range(rows):
        base = e["S1"] + 16 * r
        vals = C[base: base + 16]
        m1 = _stream_rowmax(ar, vals)
        neg = ar.fmul(m1, one_neg)
        p1 = [ar.exp2(ar.fadd(x, neg)) for x in vals]
        C[base: base + 16] = p1
        l1 = _stream_rowsum(ar, p1)
        C[e["M"] + 16 * r: e["M"] + 16 * r + 16] = [m1] * 16
        C[e["L"] + 16 * r: e["L"] + 16 * r + 16] = [l1] * 16
    first_l = [C[e["L"] + 16 * r] for r in range(rows)]
    put("O", ar.mma(region("S1"), v1, None))
    o1 = list(region("O"))
    put("S2", [ar.val(x) for x in _gemm_mma(q, k2, None, 32, 16, 64).ravel()])
    for r in range(rows):
        m1, l1 = C[e["M"] + 16 * r], C[e["L"] + 16 * r]
        base = e["S2"] + 16 * r
        vals = C[base: base + 16]
        m2 = ar.fmax(m1, _stream_rowmax(ar, vals))
        neg = ar.fmul(m2, one_neg)
        alpha = ar.val(1.0) if model == "no_alpha" else ar.exp2(ar.fadd(m1, neg))
        p2 = [ar.exp2(ar.fadd(x, neg)) for x in vals]
        C[base: base + 16] = p2
        l = ar.fadd(ar.fmul(alpha, l1), _stream_rowsum(ar, p2))
        ob = e["O"] + 16 * r
        C[ob: ob + 16] = [ar.fmul(o, alpha) for o in C[ob: ob + 16]]
        C[e["M"] + 16 * r: e["M"] + 16 * r + 16] = [m2] * 16
        C[e["L"] + 16 * r: e["L"] + 16 * r + 16] = [l] * 16
    put("O", ar.mma(region("S2"), v2, region("O")))
    for r in range(rows):
        inv = ar.recip(C[e["L"] + 16 * r])
        ob = e["O"] + 16 * r
        C[ob: ob + 16] = [ar.fmul(o, inv) for o in C[ob: ob + 16]]
    if model == "first_block":
        # the claim: O = softmax(S1) V1 on the softmaxed rows, as one block would give it
        for r in range(rows):
            inv = ar.recip(first_l[r])
            C[e["O"] + 16 * r: e["O"] + 16 * r + 16] = [ar.fmul(o, inv) for o in o1[16 * r: 16 * r + 16]]
    C[0] = ar.fadd(C[0], ar.val(1.0))                  # the one-threadgroup C[0,0] += 1 tail
    return C


def stream_reference(bundle, s):
    return np.asarray(_stream_trace(bundle, s, _StreamPoint()), dtype="<f4").reshape(32, 80)


def stream_enclosure(bundle, s, mode="hardware", model=None):
    iv = _stream_trace(bundle, s, _StreamInterval(mode), model=model)
    return (np.array([x[0] for x in iv], dtype=np.float64).reshape(32, 80),
            np.array([x[1] for x in iv], dtype=np.float64).reshape(32, 80))


def stream_bound(bundle, s):
    """Per element, the largest distance from the reference to the ends of the hardware enclosure;
    zero where no transcendental reaches the element (those compare bit for bit)."""
    ref = stream_reference(bundle, s).astype(np.float64)
    lo, hi = stream_enclosure(bundle, s)
    if np.any(ref < lo) or np.any(ref > hi):
        raise ValueError("stream model: the reference lies outside its own enclosure")
    return np.maximum(ref - lo, hi - ref)


def stream_one_shot(bundle, s):
    """The exact one-shot base-2 softmax attention over all 32 keys, in float64, on the softmaxed rows:
    O = softmax2([S1 | S2]) [V1; V2], S computed exactly from the half operands."""
    if s.get("key_blocks"):
        q, ks, vs, _c0 = _keyblock_operands(bundle, s["key_blocks"])
    else:
        q, k1, k2, v1, v2, _c0 = _stream_operands(bundle)
        ks, vs = [k1, k2], [v1, v2]
    S = q.astype(np.float64) @ np.hstack(ks).astype(np.float64)
    P = np.exp2(S - S.max(axis=1, keepdims=True))
    P = P / P.sum(axis=1, keepdims=True)
    return (P @ np.vstack(vs).astype(np.float64))[:s["stream"]]


# THE FUSED ATTENTION CLASS (production row P7, docs/g17-tensorops-machine-model.md 25.129). The
# admission rules and the buffer layout are agxforge.g17.runtime's (attention_spec, attention_layout);
# this is the program, its inputs and its reference.
ATTENTION_MODELS = ("online",
                    # controls - claims about the SAME program's output, each predicted to fail:
                    "no_alpha",          # the later blocks' rescale by alpha = exp2(m_old - m_new) left out
                    "unmasked",          # a causal program scored as if no mask were applied
                    "kv_fp32",           # K and V consumed as the fp32 projection (MMA-truncated), not RNE halves
                    "cache_transposed",  # the K cache written head x keys (a relayout) instead of keys x head
                    "stale_offset",      # the new block appended one block early (over the last prefilled one)
                    # P9's step (MM 25.131), claims about a step program's output:
                    "length_zero",       # the uniform ignored: appended at row 0 and masked as if length were 0
                    "stale_length",      # the previous step's length (length - 16) read instead
                    "unsplit",           # a kv_split 2 program scored as the sequential chain (the value change)
                    "no_merge_scale",    # the merge's f_s = exp2(m_s - M) left out (O = O0 + O1, l = l0 + l1)
                    # MM 25.114.4, a claim about a key_offsets program: every visited block reads block 0's K
                    # and V (what the frozen-register control computes)
                    "block0_only")
ATTENTION_SENTINEL = np.float32(-12345.0)   # every word of C before the dispatch (halves 0xE400, 0xC640)


def _grid_generic_spec(spec):
    """generic_spec for phase grid (MM 25.135): the class's admission, its transport, heads threadgroups."""
    from agxforge.g17 import runtime as R
    raw = dict(spec["attention"])
    keep = ("attention", "attention_model", "c_from", "attention_seed", "grid_inputs")
    if "visible" in raw:
        spec = {k: spec[k] for k in keep if k in spec}
        raw = {k: raw[k] for k in ("phase", "heads", "rows", "blocks", "causal", "q0", "first_block", "resume",
                                   "normalize", "frozen_head", "key_offsets", "kv_split", "frozen_slice", "key_mask",
                                   "runtime_q0")
               if k in raw}
        if not raw.get("causal"):
            raw.pop("q0", None)
    extra = sorted(set(spec) - set(keep) - {"M", "N", "K", "a", "b"})
    if extra:
        raise ValueError("generic: a grid attention spec carries only its attention block (and model); got %s" % extra)
    at = R.attention_spec(raw)
    lay = R.attention_layout(at)
    model = spec.get("attention_model", "online")
    if model not in GRID_MODELS:
        raise ValueError("generic: a grid attention_model is one of %s" % (GRID_MODELS,))
    if model == "no_resume" and not at["resume"]:
        raise ValueError("generic: the no_resume control needs a resumed program")
    if model == "no_alpha" and len(at["visible"]) < 2 and not at["resume"]:
        raise ValueError("generic: the no-alpha control needs two visited blocks")
    if model == "mask_early" and not at["causal"]:
        raise ValueError("generic: the mask_early control needs a causal program")
    if model == "block0_only" and at["key_offsets"] == "immediate":
        raise ValueError("generic: the block0_only claim is about a key_offsets register or frozen program")
    if at["resume"] and not spec.get("c_from"):
        raise ValueError("generic: a resumed grid program reads its state from c_from (the previous dispatch)")
    # the transport's K (1024) sizes buffers 1 and 2 only; no generic body runs at it, so the generic
    # rules are applied at K 256 (they would ask for a K loop) and the extent is set after
    s = generic_spec(dict(M=lay["M"], N=lay["N"], K=256))
    s.update(K=lay["K"], attention=at, attention_model=model, c_from=spec.get("c_from"),
             attention_seed=int(spec.get("attention_seed", 1729)), grid_inputs=spec.get("grid_inputs"),
             threadgroups=at["heads"] * (at.get("kv_split") or 1))
    return s


def _attention_generic_spec(spec):
    """generic_spec for {"attention": {...}}: the class's admission (runtime.attention_spec), its
    transport extents (runtime.attention_layout), every generic feature off."""
    from agxforge.g17 import runtime as R
    raw = dict(spec["attention"])
    if raw.get("phase") == R.ATTENTION_GRID_PHASE:
        return _grid_generic_spec(spec)
    if raw.get("phase") == R.ATTENTION_GRID_MERGE_PHASE:
        return _grid_merge_generic_spec(spec)
    if "visible" in raw:
        # an already-normalised spec (generic.json, or generic_spec applied twice): re-admit its request
        spec = {k: spec[k] for k in ("attention", "attention_model", "c_from", "attention_seed",
                                     "attention_prefill_seed", "kv_length", "kv_token_seed") if k in spec}
        if raw.get("phase") == "step":
            raw = {k: raw[k] for k in ("phase", "capacity_blocks", "rows", "kv_split") if k in raw}
            if raw.get("kv_split") == 2:
                raw["allow_value_change"] = True
        raw = {k: v for k, v in raw.items() if k not in ("blocks", "visible") and not (k == "q0" and v is None)}
    extra = sorted(set(spec) - {"attention", "attention_model", "M", "N", "K", "a", "b", "c_from",
                                "attention_seed", "attention_prefill_seed", "kv_length", "kv_token_seed"})
    if extra:
        raise ValueError("generic: an attention spec carries only its attention block (and model); got %s" % extra)
    at = R.attention_spec(raw)
    lay = R.attention_layout(at)
    model = spec.get("attention_model", "online")
    if model not in ATTENTION_MODELS:
        raise ValueError("generic: attention_model is one of %s" % (ATTENTION_MODELS,))
    if model == "unmasked" and not at["causal"]:
        raise ValueError("generic: the unmasked control needs a causal program")
    if model == "stale_offset" and not (at["cache_blocks"] and at["new_blocks"] and at["phase"] != "attend"):
        raise ValueError("generic: the stale-offset control needs prefilled and new blocks")
    if model == "block0_only" and (at["phase"] == R.KV_STEP_PHASE or at.get("key_offsets", "immediate") == "immediate"):
        raise ValueError("generic: the block0_only claim is about a key_offsets register or frozen program")
    if model == "no_alpha" and len(at["visible"]) < 2:
        raise ValueError("generic: the no-alpha control needs two visited blocks")
    step = at["phase"] == R.KV_STEP_PHASE
    if model in ("length_zero", "stale_length", "unsplit", "no_merge_scale") and not step:
        raise ValueError("generic: the %s control belongs to phase step" % model)
    if model in ("unsplit", "no_merge_scale") and at["kv_split"] != 2:
        raise ValueError("generic: the %s control needs a kv_split 2 program" % model)
    if model == "stale_offset" and step:
        raise ValueError("generic: phase step has no compile-time offset; its controls are length_zero and stale_length")
    kv_length = spec.get("kv_length")
    if step:
        # THE RUNTIME LENGTH IS AN INPUT, NOT A PROGRAM FACT: it is written into buffer 1 by
        # _attention_inputs and never reaches build_generic_program (the same bytes serve every length)
        R.KVCache(at["capacity_blocks"], rows=at["rows"], kv_split=at["kv_split"],
                  allow_value_change=at["kv_split"] == 2, length=int(kv_length if kv_length is not None else 0)).length_word()
    elif kv_length is not None:
        raise ValueError("generic: kv_length is the step's runtime uniform; phase %s has none" % at["phase"])
    s = generic_spec(dict(M=lay["M"], N=lay["N"], K=lay["K"]))
    s.update(attention=at, attention_model=model, c_from=spec.get("c_from"),
             attention_seed=int(spec.get("attention_seed", 1729)),
             attention_prefill_seed=int(spec.get("attention_prefill_seed", 4104)))
    if step:
        s["kv_length"] = int(kv_length if kv_length is not None else 0)
        if spec.get("kv_token_seed") is not None:
            s["kv_token_seed"] = int(spec["kv_token_seed"])
    elif spec.get("kv_token_seed") is not None:
        raise ValueError("generic: kv_token_seed belongs to phase step")
    return s


def _kv_length_value(builder, a):
    """The step's runtime length: the uint32 uniform at buffer-1 byte KV_LENGTH_BYTE, loaded as a word.
    Reloaded after every tensor body, because no scalar value is live across a body."""
    from agxforge.g17 import ir, runtime as R
    return builder.load(a, builder.const(R.KV_LENGTH_BYTE // 4, name="kv_len_word"), type=ir.I32, name="kv_length")


def _build_kv_step(builder, a, b, c, at):
    """PHASE STEP (P9, MM 25.131): project the 16 new tokens into the staging slots, copy the staged
    words to cache rows [length, length + 16) at the RUNTIME length, then the causal online softmax
    over every capacity block with the runtime threshold, then (kv_split 2) the merge, then the
    normalisation. One program for every length."""
    from agxforge.g17 import ir, runtime as R, tensorreduce as TR
    lay = R.attention_layout(at)
    e = {k: v // 4 for k, v in R.ATTENTION_C.items()}
    kblk, vblk = R.ATTENTION_BLOCK * R.ATTENTION_HEAD * 2, R.ATTENTION_BLOCK * R.ATTENTION_VALUE * 2
    x_off = R.ATTENTION_A["X"]
    builder.tensor_matmul(a, b, c, M=16, N=R.ATTENTION_HEAD, K=R.ATTENTION_D, offsetA=x_off,
                          offsetB=R.ATTENTION_B["WK"], offsetC=lay["SK"], epilogue=(("half_kv",),))
    builder.tensor_matmul(a, b, c, M=16, N=R.ATTENTION_VALUE, K=R.ATTENTION_D, offsetA=x_off,
                          offsetB=R.ATTENTION_B["WV"], offsetC=lay["SV"], epilogue=(("half_kv",),))
    # THE APPEND AT THE RUNTIME OFFSET. The staged K block is 16 rows x 32 words and V 16 x 8, both
    # row-major, so cache row `length + r` word w is staged word 32 r + w (K) or 8 r + w (V): lane l
    # moves words l, l + 32, ... and the destination index is length * words-per-row + that word.
    lane = builder.builtin("thread_index_in_simdgroup", name="kv_lane")
    length = _kv_length_value(builder, a)
    i32 = lambda v, n: builder.const(int(v) & 0xFFFFFFFF, type=ir.I32, name=n)
    for tag, stage, cache, rows_words in (("k", lay["SK"], lay["KC"], R.ATTENTION_HEAD // 2),
                                          ("v", lay["SV"], lay["VC"], R.ATTENTION_VALUE // 2)):
        dst = builder.add(builder.shl(length, i32(rows_words.bit_length() - 1, "kv_%s_shift" % tag),
                                      name="kv_%s_rows" % tag), lane, name="kv_%s_dst" % tag)
        for n in range(R.ATTENTION_BLOCK * rows_words // 32):
            src = builder.add(lane, i32(stage // 4 + 32 * n, "kv_%s_src%d" % (tag, n)), name="kv_%s_si%d" % (tag, n))
            word = builder.load(c, src, type=ir.I32, name="kv_%s_w%d" % (tag, n))
            builder.store_at(c, builder.add(dst, i32(cache // 4 + 32 * n, "kv_%s_off%d" % (tag, n)),
                                            name="kv_%s_di%d" % (tag, n)), word)
    regions = [(e["O"], e["M"], e["L"])]
    if at["kv_split"] == 2:
        regions.append((lay["O1"] // 4, lay["M1"] // 4, lay["L1"] // 4))
    for rng, (o_base, m_base, l_base) in zip(at["ranges"], regions):
        first = True
        for j in rng:
            builder.tensor_matmul(a, c, c, M=32, N=16, K=R.ATTENTION_HEAD, transB=True,
                                  offsetB=lay["KC"] + j * kblk, offsetC=R.ATTENTION_C["S"])
            length = _kv_length_value(builder, a)
            for r in range(at["rows"]):
                mask = TR.RuntimeThreshold(length, r, R.ATTENTION_BLOCK * j, R.KV_MASK_BIAS)
                if first:
                    TR.emit_stream_first_block(builder, c, row=r, s_base=e["S"], m_base=m_base, l_base=l_base,
                                               mask=mask)
                else:
                    TR.emit_stream_second_block(builder, c, row=r, s_base=e["S"], o_base=o_base, m_base=m_base,
                                                l_base=l_base, mask=mask)
            builder.tensor_matmul(c, c, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=not first,
                                  offsetA=R.ATTENTION_C["S"], offsetB=lay["VC"] + j * vblk, offsetC=4 * o_base)
            first = False
    if at["kv_split"] == 2:
        (o0, m0, l0), (o1, m1, l1) = regions
        for r in range(at["rows"]):
            TR.emit_kv_merge(builder, c, row=r, o_base=o0, m_base=m0, l_base=l0, o1_base=o1, m1_base=m1, l1_base=l1)
    for r in range(at["rows"]):
        TR.emit_stream_normalize(builder, c, row=r, o_base=e["O"], l_base=e["L"])


def _build_attention(builder, a, b, c, at):
    """The attention class's program (runtime.ATTENTION_COMPOSITION's comment gives the order)."""
    from agxforge.g17 import runtime as R, tensorreduce as TR
    if at["phase"] == R.KV_STEP_PHASE:
        return _build_kv_step(builder, a, b, c, at)
    if at["phase"] == R.ATTENTION_GRID_PHASE:
        return _build_attention_grid(builder, a, b, c, at)
    if at["phase"] == R.ATTENTION_GRID_MERGE_PHASE:
        return _build_attention_grid_merge(builder, a, b, c, at)
    lay = R.attention_layout(at)
    e = {k: v // 4 for k, v in R.ATTENTION_C.items()}
    kblk, vblk = R.ATTENTION_BLOCK * R.ATTENTION_HEAD * 2, R.ATTENTION_BLOCK * R.ATTENTION_VALUE * 2
    if at["phase"] in ("fused", "project"):
        for i in range(at["new_blocks"]):
            j = at["cache_blocks"] + i                   # the cache block this new block appends to
            x_off = R.ATTENTION_A["X"] + i * R.ATTENTION_BLOCK * R.ATTENTION_D * 2
            builder.tensor_matmul(a, b, c, M=16, N=R.ATTENTION_HEAD, K=R.ATTENTION_D, offsetA=x_off,
                                  offsetB=R.ATTENTION_B["WK"], offsetC=lay["KC"] + j * kblk, epilogue=(("half_kv",),))
            builder.tensor_matmul(a, b, c, M=16, N=R.ATTENTION_VALUE, K=R.ATTENTION_D, offsetA=x_off,
                                  offsetB=R.ATTENTION_B["WV"], offsetC=lay["VC"] + j * vblk, epilogue=(("half_kv",),))
    if at["phase"] == "project":
        return
    first = True
    keyed = at.get("key_offsets", "immediate")
    if keyed in R.ATTENTION_LOOP_OFFSETS:
        return _build_attention_loop(builder, a, c, at, lay, e, kblk, vblk)
    if keyed != "immediate" and at["visible"] != list(range(len(at["visible"]))):
        raise ValueError("attention: register key offsets visit blocks 0, 1, 2, ... in order")
    for j in at["visible"]:
        if keyed == "immediate":
            builder.tensor_matmul(a, c, c, M=32, N=16, K=R.ATTENTION_HEAD, transB=True,
                                  offsetB=lay["KC"] + j * kblk, offsetC=R.ATTENTION_C["S"])
        else:
            # MM 25.114.4: block j's K is the cache base plus register "k", advanced one block per body
            builder.tensor_matmul(a, c, c, M=32, N=16, K=R.ATTENTION_HEAD, transB=True,
                                  offsetB=lay["KC"], offsetC=R.ATTENTION_C["S"], offsetB_register="k",
                                  offsetB_step=kblk if keyed == "register" else 0)
        for r in range(at["rows"]):
            mask = TR.causal_threshold(at["q0"], r, R.ATTENTION_BLOCK * j) if at["causal"] else None
            if first:
                TR.emit_stream_first_block(builder, c, row=r, s_base=e["S"], m_base=e["M"], l_base=e["L"],
                                           mask=mask)
            else:
                TR.emit_stream_second_block(builder, c, row=r, s_base=e["S"], o_base=e["O"], m_base=e["M"],
                                            l_base=e["L"], mask=mask)
        if keyed == "immediate":
            builder.tensor_matmul(c, c, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=not first,
                                  offsetA=R.ATTENTION_C["S"], offsetB=lay["VC"] + j * vblk, offsetC=R.ATTENTION_C["O"])
        else:
            builder.tensor_matmul(c, c, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=not first,
                                  offsetA=R.ATTENTION_C["S"], offsetB=lay["VC"], offsetC=R.ATTENTION_C["O"],
                                  offsetB_register="v", offsetB_step=vblk if keyed == "register" else 0)
        first = False
    for r in range(at["rows"]):
        TR.emit_stream_normalize(builder, c, row=r, o_base=e["O"], l_base=e["L"])


# None: the key-block loop's trip count is the compile-time `blocks`. An int K: the trip count is the RUNTIME
# uint32 at buffer-1 byte KV_LENGTH_BYTE, capped at K (cc's capped runtime tensor loop, MM 25.144.8). A caller
# sets it around one build; tools/g17tensorruntimeloop.py is the receipt.
LOOP_RUNTIME_CAP = None


def _build_attention_loop(builder, a, c, at, lay, e, kblk, vblk):
    """THE COUNTED KEY-BLOCK LOOP (MM 25.114.5), after the projections: the state m = -FLT_MAX, l = 0,
    O = 0 is written once, the K/V index registers are set once (tensor_index_init: K at the cache
    base, V at its base - past the immediate's measured 47,104 bytes at 64 blocks, so the register
    carries it), then one block of `blocks` trips runs QK, the general online step on every row and
    PV accumulating into O, and the exit block normalises. "loop_frozen" never advances the
    registers: every trip reads block 0 (the failing control)."""
    from agxforge.g17 import ir, runtime as R, tensorreduce as TR
    if at["visible"] != list(range(at["blocks"])):
        raise ValueError("attention: the key-block loop visits blocks 0, 1, 2, ... in order")
    fn = builder.fn
    lane = builder.builtin("thread_index_in_simdgroup", name="kl_lane")
    # O = 0: all 32 x 16 words (rows past `rows` accumulate raw S V exactly as the straight-line
    # program's first, non-accumulating PV body writes them)
    zero = TR._f32_const(builder, 0.0, "kl_zero")
    obase = builder.add(builder.shl(lane, TR._i32_const(builder, 4, "kl_row_shift"), name="kl_lane16"),
                        TR._i32_const(builder, e["O"], "kl_o_base"), name="kl_o_row")
    for i in range(16):
        idx = obase if i == 0 else builder.add(obase, TR._i32_const(builder, i, "kl_o_col%d" % i), name="kl_o_idx%d" % i)
        builder.store_at(c, idx, zero)
    for r in range(at["rows"]):
        TR._store_stat(builder, c, row=r, base=e["M"], value=TR._f32_const(builder, TR.FP32_NEG_MAX, "kl_m0"),
                       prefix="kl_m0s")
        TR._store_stat(builder, c, row=r, base=e["L"], value=TR._f32_const(builder, 0.0, "kl_l0"), prefix="kl_l0s")
    builder.tensor_index_init("k", lay["KC"])
    builder.tensor_index_init("v", lay["VC"])
    step = at["key_offsets"] == "loop"
    counter0 = builder.const(0, name="kl_counter0")
    hdr, ex = fn.block("keyblock_loop"), fn.block("keyblock_exit")
    builder.br(hdr)
    builder.at(hdr)
    i = builder.phi(counter0, name="kl_block")
    builder.tensor_matmul(a, c, c, M=32, N=16, K=R.ATTENTION_HEAD, transB=True, offsetB=0,
                          offsetC=R.ATTENTION_C["S"], offsetB_register="k", offsetB_step=kblk if step else 0)
    for r in range(at["rows"]):
        TR.emit_stream_second_block(builder, c, row=r, s_base=e["S"], o_base=e["O"], m_base=e["M"],
                                    l_base=e["L"], mask=None)
    builder.tensor_matmul(c, c, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=True,
                          offsetA=R.ATTENTION_C["S"], offsetB=0, offsetC=R.ATTENTION_C["O"],
                          offsetB_register="v", offsetB_step=vblk if step else 0)
    nxt = builder.add(i, ir.Imm(1), name="kl_block_next")
    ir.Builder.phi_latch(i, nxt)
    if LOOP_RUNTIME_CAP is not None:
        # A CAPPED RUNTIME TRIP COUNT (MM 25.144.8): n is the uint32 at buffer-1 byte KV_LENGTH_BYTE, reloaded
        # after the bodies (no scalar is live across a tensor body); the loop runs min(n, cap) trips
        n = _kv_length_value(builder, a)
        builder.br_cond(builder.cmp(nxt, n, "lt", cap=LOOP_RUNTIME_CAP, name="kl_more"), hdr, ex)
    else:
        builder.br_cond(builder.cmp(nxt, at["blocks"], "lt", name="kl_more"), hdr, ex)
    builder.at(ex)
    for r in range(at["rows"]):
        TR.emit_stream_normalize(builder, c, row=r, o_base=e["O"], l_base=e["L"])


def _attention_weights(rng):
    """Wk (64 x 64) and Wv (64 x 16) halves, and the draw's scale: Wk scaled so the scores S = Q K^T
    with Q scaled by 1/8 spread over a few units (the row maxima move between blocks)."""
    from agxforge.g17 import runtime as R
    wk = (rng.uniform(-2.0, 2.0, size=(R.ATTENTION_D, R.ATTENTION_HEAD)) * 0.25).astype(np.float16)
    wv = rng.uniform(-2.0, 2.0, size=(R.ATTENTION_D, R.ATTENTION_VALUE)).astype(np.float16)
    return wk, wv


def _attention_prefill(at, seed):
    """The host-written K and V of the PREFILLED cache blocks (keys x head and keys x value halves),
    drawn as if an earlier projection had written them (the stand-in for P9's earlier dispatch)."""
    from agxforge.g17 import runtime as R
    rng = np.random.default_rng(seed)
    n = at["cache_blocks"] * R.ATTENTION_BLOCK
    k = rng.uniform(-3.0, 3.0, size=(n, R.ATTENTION_HEAD)).astype(np.float16)
    v = rng.uniform(-2.0, 2.0, size=(n, R.ATTENTION_VALUE)).astype(np.float16)
    return k, v


def _attention_inputs(bundle, s, spec):
    """a.f16 = Q (32 x 64, drawn then scaled by 1/8) | X (16 rows per new block x 64); b.f16 = Wk | Wv;
    c.f32 = the sentinel everywhere, the prefilled cache blocks written at their offsets - or, with
    c_from, the named file (the cut schedule's attend phase reads the project phase's GPU output)."""
    from agxforge.g17 import runtime as R
    at = s["attention"]
    if at["phase"] == R.ATTENTION_GRID_PHASE:
        return _grid_inputs(bundle, s, spec)
    if at["phase"] == R.ATTENTION_GRID_MERGE_PHASE:
        return _grid_merge_inputs(bundle, s, spec)
    lay = R.attention_layout(at)
    rng = np.random.default_rng(s["attention_seed"])
    q = (rng.uniform(-2.0, 2.0, size=(32, R.ATTENTION_D)) * 0.125).astype(np.float16)
    x = rng.uniform(-2.0, 2.0, size=(R.ATTENTION_BLOCK * at["new_blocks"], R.ATTENTION_D)).astype(np.float16)
    wk, wv = _attention_weights(rng)
    if s.get("kv_token_seed") is not None:
        # a later step of one KVCache (P9): new queries and tokens, the SAME weights (buffer 2 is shared)
        rng_t = np.random.default_rng(s["kv_token_seed"])
        q = (rng_t.uniform(-2.0, 2.0, size=(32, R.ATTENTION_D)) * 0.125).astype(np.float16)
        x = rng_t.uniform(-2.0, 2.0, size=x.shape).astype(np.float16)
    a = np.zeros(lay["M"] * lay["K"], dtype=np.float16)
    a[:q.size] = q.ravel()
    a[q.size:q.size + x.size] = x.ravel()
    b = np.concatenate([wk.ravel(), wv.ravel()])
    if at["phase"] == R.KV_STEP_PHASE:
        # P9: the runtime length, a uint32 uniform in buffer 1 - an input of this dispatch, not of the program
        a.view(np.uint8)[R.KV_LENGTH_BYTE:R.KV_LENGTH_BYTE + 4] = np.frombuffer(
            int(s["kv_length"]).to_bytes(4, "little"), np.uint8)
    (bundle / "a.f16").write_bytes(a.tobytes())
    (bundle / "b.f16").write_bytes(b.tobytes())
    if s.get("c_from"):
        c = np.frombuffer(Path(s["c_from"]).read_bytes(), dtype="<f4").copy()
        if c.size != lay["M"] * lay["N"]:
            raise ValueError("attention: c_from holds %d words, the layout %d" % (c.size, lay["M"] * lay["N"]))
    elif at["phase"] == R.KV_STEP_PHASE:
        # a KVCache's state before this step: the cache zero-filled (KVCache.zero_fill), its first
        # kv_length rows written as if earlier steps had appended them, every other word the sentinel
        c = np.full(lay["M"] * lay["N"], ATTENTION_SENTINEL, dtype="<f4")
        raw = bytearray(c.tobytes())
        R.KVCache(at["capacity_blocks"], rows=at["rows"], kv_split=at["kv_split"],
                  allow_value_change=at["kv_split"] == 2).zero_fill(raw)
        n = s["kv_length"]
        rng_p = np.random.default_rng(s["attention_prefill_seed"])
        k = rng_p.uniform(-3.0, 3.0, size=(n, R.ATTENTION_HEAD)).astype(np.float16)
        v = rng_p.uniform(-2.0, 2.0, size=(n, R.ATTENTION_VALUE)).astype(np.float16)
        raw[lay["KC"]:lay["KC"] + k.nbytes] = k.tobytes()
        raw[lay["VC"]:lay["VC"] + v.nbytes] = v.tobytes()
        c = np.frombuffer(bytes(raw), dtype="<f4").copy()
    else:
        c = np.full(lay["M"] * lay["N"], ATTENTION_SENTINEL, dtype="<f4")
        k, v = _attention_prefill(at, s["attention_prefill_seed"])
        raw = c.view(np.uint8)
        raw[lay["KC"]:lay["KC"] + k.nbytes] = np.frombuffer(k.tobytes(), np.uint8)
        raw[lay["VC"]:lay["VC"] + v.nbytes] = np.frombuffer(v.tobytes(), np.uint8)
    (bundle / "c.f32").write_bytes(c.tobytes())


def _attention_operands(bundle, s):
    from agxforge.g17 import runtime as R
    at = s["attention"]
    lay = R.attention_layout(at)
    a = np.frombuffer((bundle / "a.f16").read_bytes(), dtype=np.float16)
    q = a[:32 * R.ATTENTION_D].reshape(32, R.ATTENTION_D)
    x = a[32 * R.ATTENTION_D: (32 + R.ATTENTION_BLOCK * at["new_blocks"]) * R.ATTENTION_D].reshape(-1, R.ATTENTION_D)
    b = np.frombuffer((bundle / "b.f16").read_bytes(), dtype=np.float16)
    wk = b[:R.ATTENTION_D * R.ATTENTION_HEAD].reshape(R.ATTENTION_D, R.ATTENTION_HEAD)
    wv = b[R.ATTENTION_D * R.ATTENTION_HEAD:].reshape(R.ATTENTION_D, R.ATTENTION_VALUE)
    c0 = np.frombuffer((bundle / "c.f32").read_bytes(), dtype="<f4").copy()
    return lay, q, x, wk, wv, c0


def attention_cache(bundle, s, model=None):
    """The K and V cache (keys x head and keys x value, float16) the program leaves in buffer 3, as
    the reference models it: prefilled blocks from c.f32 untouched, each new block the projection
    X_i W rounded once to half (RNE, op1016) - under `model`'s claim."""
    from agxforge.g17 import runtime as R
    at = s["attention"]
    model = model or s["attention_model"]
    lay, q, x, wk, wv, c0 = _attention_operands(bundle, s)
    raw = c0.view(np.uint8)
    nk = at["blocks"] * R.ATTENTION_BLOCK
    kc = np.frombuffer(raw[lay["KC"]:lay["KC"] + nk * R.ATTENTION_HEAD * 2].tobytes(), np.float16).reshape(nk, R.ATTENTION_HEAD).copy()
    vc = np.frombuffer(raw[lay["VC"]:lay["VC"] + nk * R.ATTENTION_VALUE * 2].tobytes(), np.float16).reshape(nk, R.ATTENTION_VALUE).copy()
    kf = vf = None
    if at["phase"] in ("fused", "project"):
        f32 = lambda t: t.astype(np.float32)
        kf = {}; vf = {}
        for i in range(at["new_blocks"]):
            xi = f32(x[16 * i:16 * i + 16])
            kd = _gemm_mma(xi, f32(wk), None, 16, R.ATTENTION_HEAD, R.ATTENTION_D)
            vd = _gemm_mma(xi, f32(wv), None, 16, R.ATTENTION_VALUE, R.ATTENTION_D)
            j = at["cache_blocks"] + i - (1 if model == "stale_offset" else 0)
            kf[j], vf[j] = kd, vd
            if model == "cache_transposed":
                flat = kd.astype(np.float16).T.ravel()            # head x keys, written where keys x head goes
                kc[16 * j:16 * j + 16] = flat.reshape(16, R.ATTENTION_HEAD)
            else:
                kc[16 * j:16 * j + 16] = kd.astype(np.float16)
            vc[16 * j:16 * j + 16] = vd.astype(np.float16)
    return kc, vc, kf, vf


def _attention_trace(bundle, s, ar, model=None):
    """The fp32 words of buffer 3 the program leaves, under arithmetic `ar` (a flat list, cache words
    included as their stored bit patterns' fp32 reading, which the caller replaces), and the cache."""
    from agxforge.g17 import runtime as R, tensorreduce as TR
    at = s["attention"]
    if at["phase"] == R.KV_STEP_PHASE:
        return _kv_step_trace(bundle, s, ar, model)
    model = model or s["attention_model"]
    lay, q, _x, _wk, _wv, c0 = _attention_operands(bundle, s)
    kc, vc, kf, vf = attention_cache(bundle, s, model)
    C = [ar.val(float(v)) for v in c0]
    if at["phase"] != "project":
        e = {k: v // 4 for k, v in R.ATTENTION_C.items()}
        neg_one, masked = ar.val(-1.0), ar.val(TR.CAUSAL_MASK_VALUE)
        qf = q.astype(np.float32)
        first = True
        for j in at["visible"]:
            if model == "kv_fp32" and kf is not None and j in kf:
                kt, vj = kf[j].T.copy(), vf[j]
                S = _gemm_mma(qf, kt, None, 32, 16, R.ATTENTION_HEAD, truncate_b=True)
                vmat = np.array([[_truncate_fp32(v) for v in row] for row in vj], dtype=np.float32)
            else:
                jr = 0 if model == "block0_only" else j          # the frozen register reads block 0
                kt = kc[16 * jr:16 * jr + 16].astype(np.float32).T.copy()
                S = _gemm_mma(qf, kt, None, 32, 16, R.ATTENTION_HEAD)
                vmat = vc[16 * jr:16 * jr + 16].astype(np.float32)
            C[e["S"]:e["S"] + 512] = [ar.val(float(v)) for v in S.ravel()]
            for r in range(at["rows"]):
                base = e["S"] + 16 * r
                vals = C[base:base + 16]
                if at["causal"] and model != "unmasked":
                    t = TR.causal_threshold(at["q0"], r, R.ATTENTION_BLOCK * j)
                    vals = [masked if col > t else v for col, v in enumerate(vals)]
                if first:
                    m = _stream_rowmax(ar, vals)
                    neg = ar.fmul(m, neg_one)
                    p = [ar.exp2(ar.fadd(v, neg)) for v in vals]
                    l = _stream_rowsum(ar, p)
                else:
                    m_old, l_old = C[e["M"] + 16 * r], C[e["L"] + 16 * r]
                    m = ar.fmax(m_old, _stream_rowmax(ar, vals))
                    neg = ar.fmul(m, neg_one)
                    alpha = ar.val(1.0) if model == "no_alpha" else ar.exp2(ar.fadd(m_old, neg))
                    p = [ar.exp2(ar.fadd(v, neg)) for v in vals]
                    l = ar.fadd(ar.fmul(alpha, l_old), _stream_rowsum(ar, p))
                    ob = e["O"] + 16 * r
                    C[ob:ob + 16] = [ar.fmul(o, alpha) for o in C[ob:ob + 16]]
                C[base:base + 16] = p
                C[e["M"] + 16 * r:e["M"] + 16 * r + 16] = [m] * 16
                C[e["L"] + 16 * r:e["L"] + 16 * r + 16] = [l] * 16
            C[e["O"]:e["O"] + 512] = ar.mma(C[e["S"]:e["S"] + 512], vmat, None if first else C[e["O"]:e["O"] + 512])
            first = False
        for r in range(at["rows"]):
            inv = ar.recip(C[e["L"] + 16 * r])
            ob = e["O"] + 16 * r
            C[ob:ob + 16] = [ar.fmul(o, inv) for o in C[ob:ob + 16]]
    C[0] = ar.fadd(C[0], ar.val(1.0))                  # the one-threadgroup C[0,0] += 1 tail
    return C, kc, vc


def _attention_cache_words(lay, kc, vc):
    """{word index: uint32} for the cache regions (two halves per word, little-endian)."""
    words = {}
    for base, arr in ((lay["KC"], kc), (lay["VC"], vc)):
        u = np.frombuffer(arr.astype(np.float16).tobytes(), dtype="<u4")
        for i, w in enumerate(u):
            words[base // 4 + i] = int(w)
    return words


def attention_reference(bundle, s, model=None):
    from agxforge.g17 import runtime as R
    if _grid(s):
        return grid_reference(bundle, s, model)
    if _grid_merge(s):
        return grid_merge_reference(bundle, s, model)
    C, kc, vc = _attention_trace(bundle, s, _StreamPoint(), model)
    lay = R.attention_layout(s["attention"])
    if s["attention"]["phase"] == R.KV_STEP_PHASE:
        out = np.asarray(C, dtype="<f4").copy()
        u = out.view("<u4")
        c0 = np.frombuffer((bundle / "c.f32").read_bytes(), dtype="<u4")
        touched = _attention_touched(s)
        keep = np.array([i not in touched for i in range(u.size)])
        u[keep] = c0[keep]
        for i, w in _kv_step_words(bundle, s, model).items():
            u[i] = w
        return out.reshape(lay["M"], lay["N"])
    out = np.asarray(C, dtype="<f4").copy()
    u = out.view("<u4")
    # words no arithmetic touched keep their input BITS (a sentinel, a packed pair of halves)
    c0 = np.frombuffer((bundle / "c.f32").read_bytes(), dtype="<u4")
    touched = _attention_touched(s)
    keep = np.array([i not in touched for i in range(u.size)])
    u[keep] = c0[keep]
    for i, w in _attention_cache_words(lay, kc, vc).items():
        u[i] = w
    return out.reshape(lay["M"], lay["N"])


def _attention_touched(s):
    """The fp32 word indices the program's arithmetic writes (every other word keeps its input)."""
    from agxforge.g17 import runtime as R
    at = s["attention"]
    e = {k: v // 4 for k, v in R.ATTENTION_C.items()}
    out = {0}
    if at["phase"] == R.KV_STEP_PHASE:
        lay = R.attention_layout(at)
        regions = [(e["O"], e["M"], e["L"])]
        if at["kv_split"] == 2:
            regions.append((lay["O1"] // 4, lay["M1"] // 4, lay["L1"] // 4))
        out |= set(range(e["S"], e["S"] + 512))
        for o, m, l in regions:
            out |= set(range(o, o + 512))
            for r in range(at["rows"]):
                out |= set(range(m + 16 * r, m + 16 * r + 16)) | set(range(l + 16 * r, l + 16 * r + 16))
        return out
    if at["phase"] != "project":
        out |= set(range(e["S"], e["S"] + 512)) | set(range(e["O"], e["O"] + 512))
        for r in range(at["rows"]):
            out |= set(range(e["M"] + 16 * r, e["M"] + 16 * r + 16)) | set(range(e["L"] + 16 * r, e["L"] + 16 * r + 16))
    return out


def attention_bound(bundle, s):
    """Per element, the largest distance from the reference to the ends of the hardware enclosure
    (_StreamInterval, as section 25.114's bound); zero for every word no transcendental reaches,
    which therefore compares bit for bit - the cache words among them."""
    from agxforge.g17 import runtime as R
    if _grid(s):
        return grid_bound(bundle, s)
    if _grid_merge(s):
        return grid_merge_bound(bundle, s)
    lay = R.attention_layout(s["attention"])
    ref = attention_reference(bundle, s).astype(np.float64).ravel()
    iv, _kc, _vc = _attention_trace(bundle, s, _StreamInterval("hardware"))
    lo = np.array([x[0] for x in iv], dtype=np.float64)
    hi = np.array([x[1] for x in iv], dtype=np.float64)
    bound = np.zeros_like(ref)
    for i in _attention_touched(s):
        if not (lo[i] <= ref[i] <= hi[i]):
            raise ValueError("attention model: the reference lies outside its own enclosure at word %d" % i)
        bound[i] = max(ref[i] - lo[i], hi[i] - ref[i])
    return bound.reshape(lay["M"], lay["N"])


def attention_exact(bundle, s):
    """The exact base-2 softmax attention (float64) over the keys each row sees, from the cache the
    program should leave - the value the online program approximates (a report, not a pass rule)."""
    from agxforge.g17 import runtime as R
    at = s["attention"]
    if at["phase"] == R.KV_STEP_PHASE:
        _lay, q, _x, _wk, _wv, _c0 = _attention_operands(bundle, s)
        kc, vc, _k16, _v16, q0 = kv_step_cache(bundle, s, "online")
        S = q.astype(np.float64) @ kc.astype(np.float64).T
        S = np.where(np.arange(kc.shape[0])[None, :] > (q0 + np.arange(32))[:, None], -np.inf, S)
        P = np.exp2(S - S.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        return (P @ vc.astype(np.float64))[:at["rows"]]
    _lay, q, _x, _wk, _wv, _c0 = _attention_operands(bundle, s)
    kc, vc, _kf, _vf = attention_cache(bundle, s, "online")
    S = q.astype(np.float64) @ kc.astype(np.float64).T
    if at["causal"]:
        pos = np.arange(kc.shape[0])[None, :] > (at["q0"] + np.arange(32))[:, None]
        S = np.where(pos, -np.inf, S)
    P = np.exp2(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return (P @ vc.astype(np.float64))[:at["rows"]]


def kv_length_of(bundle):
    """The runtime length a step bundle's buffer 1 carries (the uint32 at KV_LENGTH_BYTE)."""
    from agxforge.g17 import runtime as R
    raw = (bundle / "a.f16").read_bytes()
    return int.from_bytes(raw[R.KV_LENGTH_BYTE:R.KV_LENGTH_BYTE + 4], "little")


# the append row and mask position each step claim reads: the correct runtime length, or a control's
KV_CLAIM_LENGTH = {"length_zero": lambda n: 0, "stale_length": lambda n: n - 16}


def kv_step_cache(bundle, s, model=None):
    """A step's cache as the reference models it: (K cache, V cache, staged K, staged V, q0). The cache
    is buffer 3's capacity rows as the step found them, with the new block - X W rounded once to half
    (RNE, op1016) - written at rows [n, n + 16), where n is the length the claim reads (the uniform,
    or a control's); q0 = n is the first query's position."""
    from agxforge.g17 import runtime as R
    at = s["attention"]
    model = model or s["attention_model"]
    lay, q, x, wk, wv, c0 = _attention_operands(bundle, s)
    raw = c0.view(np.uint8)
    nk = at["capacity_blocks"] * R.ATTENTION_BLOCK
    kc = np.frombuffer(raw[lay["KC"]:lay["KC"] + nk * 128].tobytes(), np.float16).reshape(nk, R.ATTENTION_HEAD).copy()
    vc = np.frombuffer(raw[lay["VC"]:lay["VC"] + nk * 32].tobytes(), np.float16).reshape(nk, R.ATTENTION_VALUE).copy()
    f32 = lambda t: t.astype(np.float32)
    k16 = _gemm_mma(f32(x[:16]), f32(wk), None, 16, R.ATTENTION_HEAD, R.ATTENTION_D).astype(np.float16)
    v16 = _gemm_mma(f32(x[:16]), f32(wv), None, 16, R.ATTENTION_VALUE, R.ATTENTION_D).astype(np.float16)
    n = KV_CLAIM_LENGTH.get(model, lambda v: v)(kv_length_of(bundle))
    if not 0 <= n <= nk - 16:
        raise ValueError("kv step: the %s claim's length %d is outside the cache" % (model, n))
    kc[n:n + 16] = k16
    vc[n:n + 16] = v16
    return kc, vc, k16, v16, n


def _kv_step_words(bundle, s, model=None):
    """{word index: uint32} for the cache and staging regions a step leaves."""
    from agxforge.g17 import runtime as R
    lay = R.attention_layout(s["attention"])
    kc, vc, k16, v16, _q0 = kv_step_cache(bundle, s, model)
    words = {}
    for base, arr in ((lay["KC"], kc), (lay["VC"], vc), (lay["SK"], k16), (lay["SV"], v16)):
        for i, w in enumerate(np.frombuffer(arr.astype(np.float16).tobytes(), dtype="<u4")):
            words[base // 4 + i] = int(w)
    return words


def _kv_step_trace(bundle, s, ar, model=None):
    """The step program under arithmetic `ar`: per range (one, or two under kv_split 2) the causal
    online softmax over its capacity blocks with the threshold q0 + row - key0, then the merge, then
    the normalisation - the order _build_kv_step emits. The claim "unsplit" runs the sequential chain
    over every block instead (the program kv_split 1 would be), "no_merge_scale" merges with f_s = 1."""
    from agxforge.g17 import runtime as R, tensorreduce as TR
    at = s["attention"]
    model = model or s["attention_model"]
    lay, q, _x, _wk, _wv, c0 = _attention_operands(bundle, s)
    kc, vc, _k16, _v16, q0 = kv_step_cache(bundle, s, model)
    e = {k: v // 4 for k, v in R.ATTENTION_C.items()}
    C = [ar.val(float(v)) for v in c0]
    neg_one, masked = ar.val(-1.0), ar.val(TR.CAUSAL_MASK_VALUE)
    qf = q.astype(np.float32)
    regions = [(e["O"], e["M"], e["L"])]
    ranges = at["ranges"]
    if model == "unsplit":
        ranges = [list(range(at["capacity_blocks"]))]
    elif at["kv_split"] == 2:
        regions.append((lay["O1"] // 4, lay["M1"] // 4, lay["L1"] // 4))
    for rng, (ob0, mb0, lb0) in zip(ranges, regions):
        first = True
        for j in rng:
            kt = kc[16 * j:16 * j + 16].astype(np.float32).T.copy()
            S = _gemm_mma(qf, kt, None, 32, 16, R.ATTENTION_HEAD)
            vmat = vc[16 * j:16 * j + 16].astype(np.float32)
            C[e["S"]:e["S"] + 512] = [ar.val(float(v)) for v in S.ravel()]
            for r in range(at["rows"]):
                base = e["S"] + 16 * r
                vals = C[base:base + 16]
                if model != "unmasked":
                    t = TR.causal_threshold(q0, r, R.ATTENTION_BLOCK * j)
                    vals = [masked if col > t else v for col, v in enumerate(vals)]
                if first:
                    m = _stream_rowmax(ar, vals)
                    neg = ar.fmul(m, neg_one)
                    p = [ar.exp2(ar.fadd(v, neg)) for v in vals]
                    l = _stream_rowsum(ar, p)
                else:
                    m_old, l_old = C[mb0 + 16 * r], C[lb0 + 16 * r]
                    m = ar.fmax(m_old, _stream_rowmax(ar, vals))
                    neg = ar.fmul(m, neg_one)
                    alpha = ar.val(1.0) if model == "no_alpha" else ar.exp2(ar.fadd(m_old, neg))
                    p = [ar.exp2(ar.fadd(v, neg)) for v in vals]
                    l = ar.fadd(ar.fmul(alpha, l_old), _stream_rowsum(ar, p))
                    ob = ob0 + 16 * r
                    C[ob:ob + 16] = [ar.fmul(o, alpha) for o in C[ob:ob + 16]]
                C[base:base + 16] = p
                C[mb0 + 16 * r:mb0 + 16 * r + 16] = [m] * 16
                C[lb0 + 16 * r:lb0 + 16 * r + 16] = [l] * 16
            C[ob0:ob0 + 512] = ar.mma(C[e["S"]:e["S"] + 512], vmat, None if first else C[ob0:ob0 + 512])
            first = False
    if len(regions) == 2:
        (o0, m0b, l0b), (o1, m1b, l1b) = regions
        for r in range(at["rows"]):
            m0, m1 = C[m0b + 16 * r], C[m1b + 16 * r]
            big = ar.fmax(m0, m1)
            neg = ar.fmul(big, neg_one)
            if model == "no_merge_scale":
                f0 = f1 = ar.val(1.0)
            else:
                f0, f1 = ar.exp2(ar.fadd(m0, neg)), ar.exp2(ar.fadd(m1, neg))
            l = ar.fadd(ar.fmul(f0, C[l0b + 16 * r]), ar.fmul(f1, C[l1b + 16 * r]))
            C[o0 + 16 * r:o0 + 16 * r + 16] = [ar.fadd(ar.fmul(f0, a), ar.fmul(f1, b)) for a, b in
                                              zip(C[o0 + 16 * r:o0 + 16 * r + 16], C[o1 + 16 * r:o1 + 16 * r + 16])]
            C[m0b + 16 * r:m0b + 16 * r + 16] = [big] * 16
            C[l0b + 16 * r:l0b + 16 * r + 16] = [l] * 16
    for r in range(at["rows"]):
        inv = ar.recip(C[e["L"] + 16 * r])
        ob = e["O"] + 16 * r
        C[ob:ob + 16] = [ar.fmul(o, inv) for o in C[ob:ob + 16]]
    C[0] = ar.fadd(C[0], ar.val(1.0))                  # the one-threadgroup C[0,0] += 1 tail
    return C, kc, vc


# THE HEAD GRID PHASE (MM 25.135; runtime.ATTENTION_GRID_*): QK head 128, value 128, `heads` heads as
# `heads` threadgroups over a host-held K/V cache, and the KV chain across dispatches (resume/normalize).
GRID_MODELS = ("online", "frozen_slice", "no_key_mask",   # the last two: KV-split claims (MM 25.114.6)
               # controls - claims about the SAME program's output, each predicted to fail:
               "frozen_head",   # every threadgroup read and wrote head 0 (the head offset absent)
               "k_half",        # QK read only the first 64 of each key's 128 dimensions
               "mask_early",    # the causal threshold one key early (each row's own newest key masked)
               "no_resume",     # a resumed dispatch scored as if it had started a fresh softmax
               "no_alpha",      # the later blocks' rescale left out
               "block0_only")   # a key_offsets=register program scored as if its registers never advanced


def _grid(s):
    at = s.get("attention")
    return bool(at) and at.get("phase") == "grid"


def _grid_draw(seed, h, j, heads_seed_tag):
    """One key block's K and V (16 x 128 halves each) of head h, from its own stream: the same key
    block draws the same values whichever dispatch of a chain carries it."""
    rng = np.random.default_rng([int(seed), int(heads_seed_tag), int(h), int(j)])
    k = rng.uniform(-3.0, 3.0, size=(16, 128)).astype(np.float16)
    v = rng.uniform(-2.0, 2.0, size=(16, 128)).astype(np.float16)
    return k, v


def grid_operands_drawn(at, seed):
    """The seeded inputs of a grid program: Q (heads x rows x 128) and this dispatch's K and V blocks
    (heads x 16 blocks x 128). Q is uniform(-1, 1) / 4, K uniform(-3, 3), V uniform(-2, 2), so the scores
    spread over a few units and the running max moves between blocks. Keys past the last query row's
    position are ZERO, as a cache's unwritten rows are (KVCache.zero_fill; g17decodestep pads so)."""
    heads, rows = at["heads"], at["rows"]
    q = np.stack([(np.random.default_rng([int(seed), 7, h]).uniform(-1.0, 1.0, size=(rows, 128)) * 0.25)
                  .astype(np.float16) for h in range(heads)])
    ks, vs = [], []
    last = at["q0"] + rows - 1 if at["causal"] else None
    for h in range(heads):
        kh, vh = [], []
        for j in range(at["blocks"]):
            gj = at["first_block"] + j
            k, v = _grid_draw(seed, h, gj, 11)
            if last is not None:
                keys = 16 * gj + np.arange(16)
                k[keys > last] = 0
                v[keys > last] = 0
            kh.append(k); vh.append(v)
        ks.append(np.concatenate(kh)); vs.append(np.concatenate(vh))
    return q, np.stack(ks), np.stack(vs)


def _grid_buffers(at, q, k, v, c_from=None):
    """a.f16, b.f16 and c.f32 of a grid program from Q (heads x rows x 128), this dispatch's K and V
    (heads x 16 blocks x 128). Buffer 3 is the sentinel everywhere (or `c_from`, the previous dispatch
    of a chain, whose O, M and L this one resumes), with this dispatch's K blocks written into each
    head's K cache and the rest of the cache zeroed."""
    from agxforge.g17 import runtime as R
    lay = R.attention_layout(at)
    st = lay["stride"]
    heads, nb = at["heads"], at["blocks"]
    a = np.zeros(lay["M"] * lay["K"], np.float16)
    b = np.zeros(lay["K"] * lay["N"], np.float16)
    if c_from is not None:
        c = np.asarray(c_from, dtype="<f4").ravel().copy()
        if c.size != lay["M"] * lay["N"]:
            raise ValueError("grid: c_from holds %d words, the layout %d" % (c.size, lay["M"] * lay["N"]))
    else:
        c = np.full(lay["M"] * lay["N"], ATTENTION_SENTINEL, dtype="<f4")
    craw = c.view(np.uint8)
    for h in range(heads):
        qa = np.zeros((32, 128), np.float16)
        qa[:q.shape[1]] = q[h]
        a[h * st["A"] // 2: h * st["A"] // 2 + qa.size] = qa.ravel()
        kc = np.zeros((lay["capacity"] * 16, 128), np.float16)
        kc[:16 * nb] = k[h]
        base = h * st["C"] + lay["KC"]
        craw[base:base + kc.nbytes] = np.frombuffer(kc.tobytes(), np.uint8)
        vb = h * st["B"] // 2
        for sl in range(8):
            for j in range(nb):
                off = vb + (j * lay["vblock"] + sl * lay["vslice"]) // 2
                b[off:off + 256] = v[h, 16 * j:16 * j + 16, 16 * sl:16 * sl + 16].ravel()
    return a, b, c


def _grid_inputs(bundle, s, spec):
    at = s["attention"]
    c_from = None
    if s.get("c_from"):
        c_from = np.frombuffer(Path(s["c_from"]).read_bytes(), dtype="<f4")
    if s.get("grid_inputs"):
        z = np.load(s["grid_inputs"])
        q, k, v = z["q"], z["k"], z["v"]
    else:
        q, k, v = grid_operands_drawn(at, s["attention_seed"])
    a, b, c = _grid_buffers(at, q, k, v, c_from)
    (bundle / "a.f16").write_bytes(a.tobytes())
    (bundle / "b.f16").write_bytes(b.tobytes())
    (bundle / "c.f32").write_bytes(c.tobytes())


def grid_operands(bundle, s):
    """(Q heads x 32 x 128 fp32, K heads x (16 blocks) x 128, V likewise, c0 fp32) read back from a grid
    bundle's files: the program's actual inputs."""
    from agxforge.g17 import runtime as R
    at = s["attention"]
    lay = R.attention_layout(at)
    st = lay["stride"]
    heads, nb = at["heads"], at["blocks"]
    a = np.frombuffer((bundle / "a.f16").read_bytes(), np.float16)
    b = np.frombuffer((bundle / "b.f16").read_bytes(), np.float16)
    c0 = np.frombuffer((bundle / "c.f32").read_bytes(), dtype="<f4").copy()
    craw = c0.view(np.uint8)
    q = np.stack([a[h * st["A"] // 2: h * st["A"] // 2 + 32 * 128].reshape(32, 128) for h in range(heads)]).astype(np.float32)
    k = np.stack([np.frombuffer(craw[h * st["C"] + lay["KC"]: h * st["C"] + lay["KC"] + nb * lay["kblock"]].tobytes(),
                                np.float16).reshape(16 * nb, 128) for h in range(heads)]).astype(np.float32)
    v = np.zeros((heads, 16 * nb, 128), np.float32)
    for h in range(heads):
        vb = h * st["B"] // 2
        for sl in range(8):
            for j in range(nb):
                off = vb + (j * lay["vblock"] + sl * lay["vslice"]) // 2
                v[h, 16 * j:16 * j + 16, 16 * sl:16 * sl + 16] = b[off:off + 256].reshape(16, 16)
    return q, k, v, c0


def grid_head_trace(ar, q, k, v, *, rows, causal, q0, first_block, state=None, normalize=True, model="online",
                    key_zero=False):
    """ONE HEAD of the grid program, in its order, under arithmetic `ar` (_StreamPoint or _StreamInterval).
    q: 32 x 128 (rows past `rows` zero), k, v: (16 n) x 128. `state` (resume): {"S": 32 x 16, "O": 32 x 128,
    "M": [rows], "L": [rows]} as the previous dispatch left them (points), else None. Returns the state
    this program leaves: S and O as {"att": per attended row, lists of ar values; "pad": numpy points of
    the padded rows}, M and L per attended row. The padded rows carry only MMA arithmetic (their Q is zero
    and no row stage touches them), so they are points in every arithmetic."""
    import g17decodestep as D
    from agxforge.g17 import tensorreduce as TR
    q = np.asarray(q, np.float32)
    if model == "k_half":
        q = q.copy(); q[:, 64:] = 0          # the same as reading only the first 64 dimensions of K
    point = isinstance(ar, _StreamPoint)
    # key_zero (the KV split's KeyMask, MM 25.114.6): a masked score is -FLT_MAX and its P is +0
    neg_one, masked = ar.val(-1.0), ar.val(TR.FP32_NEG_MAX if key_zero else TR.CAUSAL_MASK_VALUE)
    nb = k.shape[0] // 16
    if state is not None and model != "no_resume":
        o_att = [[ar.val(float(x)) for x in state["O"][r]] for r in range(rows)]
        o_pad = np.asarray(state["O"][rows:], np.float32).copy()
        m = [ar.val(float(x)) for x in state["M"]]
        l = [ar.val(float(x)) for x in state["L"]]
        first = False
    else:
        o_att, o_pad, m, l, first = None, None, [None] * rows, [None] * rows, True
    s_att = s_pad = None
    for j in range(nb):
        jr = 0 if model == "block0_only" else j          # a frozen key register reads block 0 every time
        S = D.gemm(q, np.asarray(k[16 * jr:16 * jr + 16], np.float32).T.copy())     # 32 x 16, one chain over 128
        p_att = []
        alphas = []
        for r in range(rows):
            vals = [ar.val(float(x)) for x in S[r]]
            t = 15
            if causal:
                t = TR.causal_threshold(q0 - (1 if model == "mask_early" else 0), r, 16 * (first_block + j))
                vals = [masked if col > t else x for col, x in enumerate(vals)]
            live = (lambda p: [ar.val(0.0) if col > t else x for col, x in enumerate(p)]) if key_zero else (lambda p: p)
            if first:
                m[r] = _stream_rowmax(ar, vals)
                neg = ar.fmul(m[r], neg_one)
                p = live([ar.exp2(ar.fadd(x, neg)) for x in vals])
                l[r] = _stream_rowsum(ar, p)
                alphas.append(None)
            else:
                m_old = m[r]
                m[r] = ar.fmax(m_old, _stream_rowmax(ar, vals))
                neg = ar.fmul(m[r], neg_one)
                alpha = ar.val(1.0) if model == "no_alpha" else ar.exp2(ar.fadd(m_old, neg))
                p = live([ar.exp2(ar.fadd(x, neg)) for x in vals])
                l[r] = ar.fadd(ar.fmul(alpha, l[r]), _stream_rowsum(ar, p))
                o_att[r] = [ar.fmul(o, alpha) for o in o_att[r]]
                alphas.append(alpha)
            p_att.append(p)
        s_att, s_pad = p_att, S[rows:].copy()
        # PV: eight 16-wide slices; the padded rows' A is their raw score row
        vj = np.asarray(v[16 * jr:16 * jr + 16], np.float32)
        pad = D.gemm(S[rows:], vj, None if first else o_pad, truncate_a=True)
        if point:
            pa = np.array([[float(x) for x in p] for p in p_att], np.float32)
            ca = None if first else np.array([[float(x) for x in o] for o in o_att], np.float32)
            att = D.gemm(pa, vj, ca, truncate_a=True)
            o_att = [[float(x) for x in row] for row in att]
        else:
            o_att = [_grid_interval_pv_row(ar, p_att[r], vj, None if first else o_att[r]) for r in range(rows)]
        o_pad = pad
        first = False
    if normalize:
        for r in range(rows):
            inv = ar.recip(l[r])
            o_att[r] = [ar.fmul(o, inv) for o in o_att[r]]
    return {"S": {"att": s_att, "pad": s_pad}, "O": {"att": o_att, "pad": o_pad}, "M": m, "L": l}


def _grid_interval_pv_row(ar, prow, vj, c):
    """One attended row of PV under _StreamInterval: 128 columns, each the section-136 enclosure of
    _StreamInterval.mma (truncated A at both ends, slack for the MMA's roundings, then + C)."""
    row = [ar._trunc(x) for x in prow]
    out = []
    degenerate = all(p[0] == p[1] for p in row)
    for n in range(vj.shape[1]):
        cc = None if c is None else c[n]
        if degenerate and (cc is None or cc[0] == cc[1]):
            x = _mma16([p[0] for p in row], [float(vj[kk, n]) for kk in range(16)], None)
            x = x if cc is None else _f32(np.float32(x) + np.float32(cc[0]))
            out.append((float(x), float(x)))
            continue
        lo = sum(min(p[0] * float(vj[kk, n]), p[1] * float(vj[kk, n])) for kk, p in enumerate(row))
        hi = sum(max(p[0] * float(vj[kk, n]), p[1] * float(vj[kk, n])) for kk, p in enumerate(row))
        mag = sum(max(abs(p[0]), abs(p[1])) * abs(float(vj[kk, n])) for kk, p in enumerate(row))
        slack = 6 * 2.0 ** -24 * mag
        lo, hi = lo - slack, hi + slack
        if cc is not None:
            lo, hi = lo + cc[0], hi + cc[1]
            slack = 2.0 ** -24 * (abs(lo) + abs(hi))
            lo, hi = lo - slack, hi + slack
        out.append(ar._iv(lo, hi))
    return out


def _grid_state_of(c0, at, h):
    """The (S, O, M, L) state head h's region of buffer 3 holds (what a resumed dispatch reads)."""
    from agxforge.g17 import runtime as R
    lay = R.attention_layout(at)
    base = h * lay["stride"]["C"] // 4
    O = np.concatenate([c0[base + t // 4: base + t // 4 + 512].reshape(32, 16) for t in lay["O_tiles"]], axis=1)
    return {"O": O, "M": [c0[base + lay["Mst"] // 4 + 16 * r] for r in range(at["rows"])],
            "L": [c0[base + lay["L"] // 4 + 16 * r] for r in range(at["rows"])]}


SPLIT_MODELS = ("frozen_slice",   # every slice READ slice 0's key blocks (the slice offset absent)
                "no_key_mask")    # the per-trip key mask left out (the padded keys and the causal edge attended)


def split_operands(bundle, s):
    """(Q heads x 32 x 128, K and V heads x (16 (first + padded)) x 128, c0) of a KV-split bundle, read at
    the addresses the program reads: K block j of head h at buffer-3 byte h * 131072 + 4096 j (past the
    17-block capacity that is whatever the head's region holds), V block j slice sl at buffer-2 byte
    h * 131072 + 4096 j + 512 sl (zero past the cache)."""
    from agxforge.g17 import runtime as R
    at = s["attention"]
    lay = R.attention_layout(at)
    st = lay["stride"]
    heads, nb = at["heads"], at["first_block"] + at["padded_blocks"]
    a = np.frombuffer((bundle / "a.f16").read_bytes(), np.float16)
    b = np.frombuffer((bundle / "b.f16").read_bytes(), np.float16)
    c0 = np.frombuffer((bundle / "c.f32").read_bytes(), dtype="<f4").copy()
    craw = c0.view(np.uint8)
    q = np.stack([a[h * st["A"] // 2: h * st["A"] // 2 + 32 * 128].reshape(32, 128) for h in range(heads)]).astype(np.float32)
    k = np.stack([np.frombuffer(craw[h * st["C"] + lay["KC"]: h * st["C"] + lay["KC"] + nb * lay["kblock"]].tobytes(),
                                np.float16).reshape(16 * nb, 128) for h in range(heads)]).astype(np.float32)
    v = np.zeros((heads, 16 * nb, 128), np.float32)
    for h in range(heads):
        vb = h * st["B"] // 2
        for sl in range(8):
            for j in range(nb):
                off = vb + (j * lay["vblock"] + sl * lay["vslice"]) // 2
                v[h, 16 * j:16 * j + 16, 16 * sl:16 * sl + 16] = b[off:off + 256].reshape(16, 16)
    return q, k, v, c0


def split_slice_trace(ar, q, k, v, at, sl, *, model="online", read_slice=None):
    """ONE SLICE's partial, in the program's order: bps trips from the written state (O = 0, m = -FLT_MAX,
    l = 0) over blocks first + sl * bps .., each masked by KeyMask (keys past q0 + row; -FLT_MAX and P = +0),
    not normalised. `read_slice`: the blocks READ (frozen_slice reads slice 0's; the mask stays its own).
    k and v start at absolute block 0."""
    from agxforge.g17 import tensorreduce as TR
    bps, first = at["blocks_per_slice"], at["first_block"]
    rs = sl if read_slice is None else read_slice
    lo = 16 * (first + rs * bps)
    kk, vv = k[lo:lo + 16 * bps], v[lo:lo + 16 * bps]
    if at["key_offsets"] == "loop_frozen" or model == "block0_only":
        kk = np.tile(kk[:16], (bps, 1))
        vv = np.tile(vv[:16, :16], (bps, 8))
    rows = at["rows"]
    state = {"O": np.zeros((32, 128), np.float32), "M": [TR.FP32_NEG_MAX] * rows, "L": [0.0] * rows}
    return grid_head_trace(ar, q, kk, vv, rows=rows, causal=at["key_mask"] and model != "no_key_mask",
                           q0=at["q0"], first_block=first + sl * bps, state=state, normalize=False,
                           model="online" if model in SPLIT_MODELS + ("block0_only",) else model, key_zero=True)


def _split_trace(bundle, s, ar, model=None):
    """{threadgroup: trace}, t = head * S + slice."""
    at = s["attention"]
    model = model or s["attention_model"]
    q, k, v, c0 = split_operands(bundle, s)
    frozen = model == "frozen_head" or at["frozen_head"]
    fslice = model == "frozen_slice" or at["frozen_slice"]
    if not at["key_mask"] and model == "online":
        model = "no_key_mask"
    S = at["kv_split"]
    if at.get("runtime_q0"):
        # the runtime length (MM 25.138.1): the position the program reads, from the dispatch's own buffer 3
        from agxforge.g17 import runtime as R
        at = dict(at, q0=int(c0.view("<u4")[R.ATTENTION_GRID_LENGTH_BYTE // 4]))
    out = {}
    for h in range(at["heads"]):
        src = 0 if frozen else h
        for sl in range(S):
            out[h * S + sl] = split_slice_trace(ar, q[src], k[src], v[src], at, sl, model=model,
                                                read_slice=0 if fslice else None)
    return out, c0


def _split_words(at, t, tr, pick):
    """{word index: value} of threadgroup t's slot: S (32 x 16), the eight O tiles, and M and L rows."""
    from agxforge.g17 import runtime as R
    sl_ = R.attention_layout(at)["slots"]
    base = (sl_["base"] + t * sl_["stride"]) // 4
    rows = at["rows"]
    words = {}
    for r in range(32):
        srow = tr["S"]["att"][r] if r < rows else tr["S"]["pad"][r - rows]
        orow = tr["O"]["att"][r] if r < rows else tr["O"]["pad"][r - rows]
        for col in range(16):
            words[base + sl_["S"] // 4 + 16 * r + col] = pick(srow[col], r < rows)
        for k, ob in enumerate(sl_["O_tiles"]):
            for col in range(16):
                words[base + ob // 4 + 16 * r + col] = pick(orow[16 * k + col], r < rows)
    for r in range(rows):
        for col in range(16):
            words[base + sl_["Mst"] // 4 + 16 * r + col] = pick(tr["M"][r], True)
            words[base + sl_["L"] // 4 + 16 * r + col] = pick(tr["L"][r], True)
    return words


def split_key_words(at):
    """{word index: uint32} the key-mask word each slot ends with: 16 (first + (slice + 1) bps)."""
    from agxforge.g17 import runtime as R
    if not at["key_mask"]:
        return {}
    sl_ = R.attention_layout(at)["slots"]
    S, bps = at["kv_split"], at["blocks_per_slice"]
    return {(sl_["base"] + t * sl_["stride"] + sl_["W"]) // 4: 16 * (at["first_block"] + (t % S + 1) * bps)
            for t in range(at["heads"] * S)}


def split_partials(got, at):
    """{threadgroup: (m rows, l rows, O rows x 128)} read from a buffer-3 image (what Piece A's merge reads)."""
    from agxforge.g17 import runtime as R
    sl_ = R.attention_layout(at)["slots"]
    c = np.asarray(got, dtype="<f4").ravel()
    out = {}
    for t in range(at["heads"] * at["kv_split"]):
        base = (sl_["base"] + t * sl_["stride"]) // 4
        O = np.concatenate([c[base + ob // 4: base + ob // 4 + 512].reshape(32, 16) for ob in sl_["O_tiles"]], axis=1)
        out[t] = (c[base + sl_["Mst"] // 4 + 16 * np.arange(at["rows"])].copy(),
                  c[base + sl_["L"] // 4 + 16 * np.arange(at["rows"])].copy(), O[:at["rows"]].copy())
    return out


def split_fold(ar, parts):
    """recon 154's merge over S partials in ascending slice order, then the normalisation (O x recip(l)):
    M = max m_s, f_s = exp2(m_s - M), l = sum f_s l_s, O = sum f_s O_s, every product and sum rounded
    separately. `parts` = [(m, l, O row)] under `ar` (values or intervals). Returns the normalised row."""
    neg_one = ar.val(-1.0)
    big = parts[0][0]
    for m, _l, _o in parts[1:]:
        big = ar.fmax(big, m)
    neg = ar.fmul(big, neg_one)
    l = o = None
    for m, ls, os_ in parts:
        f = ar.exp2(ar.fadd(m, neg))
        fl = ar.fmul(f, ls)
        fo = [ar.fmul(f, x) for x in os_]
        l = fl if l is None else ar.fadd(l, fl)
        o = fo if o is None else [ar.fadd(a, b) for a, b in zip(o, fo)]
    inv = ar.recip(l)
    return [ar.fmul(x, inv) for x in o]


def _grid_trace(bundle, s, ar, model=None):
    """{head: trace} for every head. Under `frozen_head` (the claim, or the control program) every
    threadgroup READS head 0's Q, K and V and writes its own head's region, so every head holds head 0's
    attention (its resumed state, if any, is its own)."""
    at = s["attention"]
    model = model or s["attention_model"]
    q, k, v, c0 = grid_operands(bundle, s)
    frozen = model == "frozen_head" or at["frozen_head"]
    if at["key_offsets"] in ("frozen", "loop_frozen") and model == "online":
        model = "block0_only"                            # the frozen-register program's own arithmetic
    out = {}
    for h in range(at["heads"]):
        src = 0 if frozen else h
        state = _grid_state_of(c0, at, h) if at["resume"] else None
        out[h] = grid_head_trace(ar, q[src], k[src], v[src], rows=at["rows"], causal=at["causal"], q0=at["q0"],
                                 first_block=at["first_block"], state=state, normalize=at["normalize"],
                                 model=model)
    return out, c0


def _grid_words(at, h, tr, pick):
    """{word index: value} of one head's trace: S (32 x 16), the eight O tiles, and M and L rows."""
    from agxforge.g17 import runtime as R
    lay = R.attention_layout(at)
    base = h * lay["stride"]["C"] // 4
    rows = at["rows"]
    words = {}
    for r in range(32):
        srow = tr["S"]["att"][r] if r < rows else tr["S"]["pad"][r - rows]
        orow = tr["O"]["att"][r] if r < rows else tr["O"]["pad"][r - rows]
        for col in range(16):
            words[base + lay["S"] // 4 + 16 * r + col] = pick(srow[col], r < rows)
        for sl, t in enumerate(lay["O_tiles"]):
            for col in range(16):
                words[base + t // 4 + 16 * r + col] = pick(orow[16 * sl + col], r < rows)
    for r in range(rows):
        for col in range(16):
            words[base + lay["Mst"] // 4 + 16 * r + col] = pick(tr["M"][r], True)
            words[base + lay["L"] // 4 + 16 * r + col] = pick(tr["L"][r], True)
    return words


def grid_reference(bundle, s, model=None):
    """Buffer 3 as the grid program leaves it (M x N fp32): every word no arithmetic touches keeps its
    input bits (the K cache, the sentinel, unattended M and L rows, other heads under frozen_head)."""
    from agxforge.g17 import runtime as R
    at = s["attention"]
    lay = R.attention_layout(at)
    split = bool(at.get("kv_split"))
    traces, c0 = (_split_trace if split else _grid_trace)(bundle, s, _StreamPoint(), model)
    out = c0.copy()
    for h, tr in traces.items():
        for i, x in (_split_words if split else _grid_words)(at, h, tr, lambda x, att: float(x)).items():
            out[i] = x
    if split:
        u = out.view("<u4")
        for i, w in split_key_words(at).items():
            u[i] = w
    return out.reshape(lay["M"], lay["N"])


def grid_bound(bundle, s):
    """Per word, the distance from the reference to the ends of the hardware enclosure; zero wherever no
    transcendental reaches (the padded rows, the untouched words)."""
    from agxforge.g17 import runtime as R
    at = s["attention"]
    lay = R.attention_layout(at)
    ref = grid_reference(bundle, s).astype(np.float64).ravel()
    split = bool(at.get("kv_split"))
    traces, _c0 = (_split_trace if split else _grid_trace)(bundle, s, _StreamInterval("hardware"))
    bound = np.zeros_like(ref)
    for h, tr in traces.items():
        for i, iv in (_split_words if split else _grid_words)(at, h, tr, lambda x, att: x if att else None).items():
            if iv is None:
                continue
            if not (iv[0] <= ref[i] <= iv[1]):
                raise ValueError("grid model: the reference lies outside its own enclosure at word %d" % i)
            bound[i] = max(ref[i] - iv[0], iv[1] - ref[i])
    return bound.reshape(lay["M"], lay["N"])


def grid_o_rows(got, at, h):
    """Head h's attended O rows (rows x 128) from a buffer-3 image."""
    from agxforge.g17 import runtime as R
    lay = R.attention_layout(at)
    c = np.asarray(got, dtype="<f4").ravel()
    base = h * lay["stride"]["C"] // 4
    return np.concatenate([c[base + t // 4: base + t // 4 + 512].reshape(32, 16) for t in lay["O_tiles"]],
                          axis=1)[:at["rows"]]


def _build_attention_grid(builder, a, b, c, at):
    """PHASE GRID (MM 25.135): per key block, QK (one 32 x 16 x 128 body: Q from buffer 1, the K cache
    block from buffer 3 under transB), the online-softmax row stage (the first-block stage, or the
    rescaling stage over all eight O tiles), and eight PV bodies (P from buffer 3, the V slice from
    buffer 2) into the eight O tiles; then (normalize) the per-row normalisation of the eight tiles.
    Every body carries the head strides and every row stage the head base, so threadgroup t runs head t."""
    from agxforge.g17 import runtime as R, tensorreduce as TR
    lay = R.attention_layout(at)
    st = lay["stride"]
    # frozen_head (THE FAILING CONTROL): the strides of the operands that READ a head's inputs (Q, the K
    # cache, the V cache) are 0, so every threadgroup reads head 0; its own S, O, M and L keep their
    # strides, so no two threadgroups share a written word and the output is still deterministic
    frozen = at["frozen_head"]
    qk_stride = (0, 0, st["C"]) if frozen else (st["A"], st["C"], st["C"])
    pv_stride = (st["C"], 0, st["C"]) if frozen else (st["C"], st["B"], st["C"])
    shift = (st["C"] // 4).bit_length() - 1
    if st["C"] // 4 != 1 << shift:
        raise ValueError("grid: the buffer-3 head stride must be a power of two words")
    head = TR.HeadBase(shift)
    o_tiles = [t // 4 for t in lay["O_tiles"]]
    first = not at["resume"]
    # key_offsets (MM 25.114.3/25.114.4): "immediate" folds block j's K and V offsets into each body; "register"
    # reads them at a fixed immediate plus stream registers "k" (advanced one K block per QK body) and "v"
    # (advanced one 512-byte V piece per PV body), so every QK body, and every PV body, has the same bytes;
    # "frozen" is the register program with no advance (every block reads block 0: the failing control)
    keyed = at["key_offsets"]
    if at.get("kv_split"):
        return _build_attention_grid_split(builder, a, b, c, at, lay)
    if keyed in R.ATTENTION_LOOP_OFFSETS:
        return _build_attention_grid_loop(builder, a, b, c, at, lay, qk_stride, pv_stride, head, o_tiles)
    kreg = {} if keyed == "immediate" else dict(offsetB_register="k", offsetB_step=lay["kblock"] if keyed == "register" else 0)
    vreg = {} if keyed == "immediate" else dict(offsetB_register="v", offsetB_step=lay["vslice"] if keyed == "register" else 0)
    for j in at["visible"]:
        builder.tensor_matmul(a, c, c, M=32, N=16, K=at["head"], transB=True, offsetA=0,
                              offsetB=lay["KC"] + (j * lay["kblock"] if keyed == "immediate" else 0), offsetC=lay["S"],
                              head_stride=qk_stride, **kreg)
        for r in range(at["rows"]):
            mask = (TR.causal_threshold(at["q0"], r, R.ATTENTION_BLOCK * (at["first_block"] + j))
                    if at["causal"] else None)
            if first:
                TR.emit_stream_first_block(builder, c, row=r, s_base=lay["S"] // 4, m_base=lay["Mst"] // 4,
                                           l_base=lay["L"] // 4, mask=mask, head=head)
            else:
                TR.emit_stream_second_block(builder, c, row=r, s_base=lay["S"] // 4, o_base=o_tiles[0],
                                            m_base=lay["Mst"] // 4, l_base=lay["L"] // 4, mask=mask,
                                            head=head, o_bases=o_tiles)
        for sl in range(at["value"] // 16):
            vo = j * lay["vblock"] + sl * lay["vslice"] if keyed == "immediate" else 0
            builder.tensor_matmul(c, b, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=not first,
                                  offsetA=lay["S"], offsetB=vo, offsetC=lay["O_tiles"][sl], head_stride=pv_stride,
                                  **vreg)
        first = False
    if at["normalize"]:
        for r in range(at["rows"]):
            TR.emit_stream_normalize(builder, c, row=r, o_base=o_tiles[0], l_base=lay["L"] // 4, head=head,
                                     o_bases=o_tiles)


def _build_attention_grid_loop(builder, a, b, c, at, lay, qk_stride, pv_stride, head, o_tiles):
    """PHASE GRID ON THE COUNTED KEY-BLOCK LOOP (MM 25.135.4; Piece B's loop, 25.114.5, at head 128 and
    value 128 with the head grid). Before the loop: every head's eight O tiles = 0 (all 32 x 16 words of
    each), m = -FLT_MAX and l = 0 on the attended rows, and the K/V index registers set once
    (tensor_index_init: K at the cache base, V at 0). Then ONE block of `loop_blocks` trips: QK from the K
    cache at register "k" (advanced one 4,096-byte block), the general online step on every row (no
    mask: these blocks are unmasked for every row), and eight PV bodies from the V cache at register "v"
    (each advanced one 512-byte slice, so eight per trip = one block). After the loop, the blocks the
    causal mask reaches are PEELED: the same three stages, straight-line, reading the same registers
    (which the loop left at the first peeled block), each row stage with its compile-time mask. Then the
    normalisation. "loop_frozen" never advances the registers (every trip and every peeled block read
    block 0: the failing control). frozen_head composes as in the straight-line program."""
    from agxforge.g17 import ir, runtime as R, tensorreduce as TR
    step = at["key_offsets"] == "loop"
    kreg = dict(offsetB_register="k", offsetB_step=lay["kblock"] if step else 0)
    vreg = dict(offsetB_register="v", offsetB_step=lay["vslice"] if step else 0)
    fn = builder.fn
    # the written state: O = 0 on every word of the eight tiles (the padded rows accumulate raw S V onto
    # +0, as the straight-line program's first, non-accumulating PV bodies write them), m and l per row
    lane = builder.builtin("thread_index_in_simdgroup", name="gl_lane")
    zero = TR._f32_const(builder, 0.0, "gl_zero")
    row = builder.add(builder.shl(lane, TR._i32_const(builder, 4, "gl_row_shift"), name="gl_lane16"),
                      head.emit(builder, "gl_head"), name="gl_row")
    for t, ob in enumerate(o_tiles):
        base = builder.add(row, TR._i32_const(builder, ob, "gl_o%d" % t), name="gl_o%d_row" % t)
        for i in range(16):
            idx = base if i == 0 else builder.add(base, TR._i32_const(builder, i, "gl_col%d" % i),
                                                  name="gl_o%d_idx%d" % (t, i))
            builder.store_at(c, idx, zero)
    for r in range(at["rows"]):
        TR._store_stat(builder, c, row=r, base=lay["Mst"] // 4,
                       value=TR._f32_const(builder, TR.FP32_NEG_MAX, "gl_m0"), prefix="gl_m0s", head=head)
        TR._store_stat(builder, c, row=r, base=lay["L"] // 4, value=TR._f32_const(builder, 0.0, "gl_l0"),
                       prefix="gl_l0s", head=head)
    builder.tensor_index_init("k", lay["KC"])
    builder.tensor_index_init("v", 0)

    def one_block(j):
        """QK, the online step on every row (masked when block j is peeled), eight PV slices."""
        builder.tensor_matmul(a, c, c, M=32, N=16, K=at["head"], transB=True, offsetA=0, offsetB=0,
                              offsetC=lay["S"], head_stride=qk_stride, **kreg)
        for r in range(at["rows"]):
            mask = (TR.causal_threshold(at["q0"], r, R.ATTENTION_BLOCK * (at["first_block"] + j))
                    if j is not None and at["causal"] else None)
            TR.emit_stream_second_block(builder, c, row=r, s_base=lay["S"] // 4, o_base=o_tiles[0],
                                        m_base=lay["Mst"] // 4, l_base=lay["L"] // 4, mask=mask,
                                        head=head, o_bases=o_tiles)
        for sl in range(at["value"] // 16):
            builder.tensor_matmul(c, b, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=True,
                                  offsetA=lay["S"], offsetB=0, offsetC=lay["O_tiles"][sl], head_stride=pv_stride,
                                  **vreg)

    counter0 = builder.const(0, name="gl_counter0")
    hdr, ex = fn.block("grid_keyblock_loop"), fn.block("grid_keyblock_exit")
    builder.br(hdr)
    builder.at(hdr)
    i = builder.phi(counter0, name="gl_block")
    one_block(None)
    nxt = builder.add(i, ir.Imm(1), name="gl_block_next")
    ir.Builder.phi_latch(i, nxt)
    builder.br_cond(builder.cmp(nxt, at["loop_blocks"], "lt", name="gl_more"), hdr, ex)
    builder.at(ex)
    for j in range(at["loop_blocks"], at["blocks"]):
        one_block(j)
    if at["normalize"]:
        for r in range(at["rows"]):
            TR.emit_stream_normalize(builder, c, row=r, o_base=o_tiles[0], l_base=lay["L"] // 4, head=head,
                                     o_bases=o_tiles)


def _build_attention_grid_split(builder, a, b, c, at, lay):
    """THE KV SPLIT ON THE LOOP (MM 25.114.6). Threadgroup t = head * S + slice (head = t >> log2 S, slice =
    t & (S - 1): tlower's head_slices, one SR_TG_X read per body). Every slice runs the SAME
    blocks_per_slice trips; its K and V reads start slice * bps blocks further in (the slice stride on
    B), the K/V index registers are set once to the cache bases as in the unsplit loop, and its S, O, M
    and L are its own slot (heads * 131072 + t * 22528 bytes; the C / PV-A slice and head strides, and
    tensorreduce.SlotBase in the row stages). Before the loop the slot is written: O = 0 on all 32 x 16
    words of the eight tiles, m = -FLT_MAX and l = 0 on the attended rows, and the slot's key word
    W = 16 (first_block + slice * bps), this slice's first key. Each trip: QK; the online step on every
    row under tensorreduce.KeyMask (key0 = W, loaded; masks every key past q0 + row, so the causal edge
    and the padded keys; a masked P is +0); W += 16; eight PV bodies. No peel, no normalisation: the
    slot holds the slice's partial (Piece A's merge reads it). Controls: frozen_slice (the B slice stride
    is 0: every slice READS slice 0's blocks) and key_mask=False (no mask, no W)."""
    from agxforge.g17 import ir, runtime as R, tensorreduce as TR
    st = lay["stride"]
    sl_ = lay["slots"]
    S, bps = at["kv_split"], at["blocks_per_slice"]
    kv_slice = 0 if at["frozen_slice"] else bps * lay["kblock"]
    slot_head = S * sl_["stride"]
    frozen = at["frozen_head"]
    qk_head = (0, 0, slot_head) if frozen else (st["A"], st["C"], slot_head)
    pv_head = (slot_head, 0, slot_head) if frozen else (slot_head, st["B"], slot_head)
    qk_slice = (0, kv_slice, sl_["stride"])
    pv_slice = (sl_["stride"], kv_slice, sl_["stride"])
    slot = TR.SlotBase(sl_["stride"] // 4, sl_["base"] // 4)
    o_tiles = [t // 4 for t in sl_["O_tiles"]]
    step = at["key_offsets"] == "loop"
    kreg = dict(offsetB_register="k", offsetB_step=lay["kblock"] if step else 0)
    vreg = dict(offsetB_register="v", offsetB_step=lay["vslice"] if step else 0)
    fn = builder.fn
    lane = builder.builtin("thread_index_in_simdgroup", name="ks_lane")
    zero = TR._f32_const(builder, 0.0, "ks_zero")
    row = builder.add(builder.shl(lane, TR._i32_const(builder, 4, "ks_row_shift"), name="ks_lane16"),
                      slot.emit(builder, "ks_slot"), name="ks_row")
    for t, ob in enumerate(o_tiles):
        base = builder.add(row, TR._i32_const(builder, ob, "ks_o%d" % t), name="ks_o%d_row" % t)
        for i in range(16):
            idx = base if i == 0 else builder.add(base, TR._i32_const(builder, i, "ks_col%d" % i),
                                                  name="ks_o%d_idx%d" % (t, i))
            builder.store_at(c, idx, zero)
    for r in range(at["rows"]):
        TR._store_stat(builder, c, row=r, base=sl_["Mst"] // 4,
                       value=TR._f32_const(builder, TR.FP32_NEG_MAX, "ks_m0"), prefix="ks_m0s", head=slot)
        TR._store_stat(builder, c, row=r, base=sl_["L"] // 4, value=TR._f32_const(builder, 0.0, "ks_l0"),
                       prefix="ks_l0s", head=slot)
    masked = at["key_mask"]
    if masked:
        # W = 16 (first_block + slice * bps): slice = t & (S - 1), times 16 bps by shift-adds
        tg = builder.builtin("threadgroup_position_in_grid", name="ks_w_tg")
        sl = getattr(builder, "and")(tg, TR._i32_const(builder, S - 1, "ks_w_mask"), name="ks_w_slice")
        key0 = TR._i32_const(builder, R.ATTENTION_BLOCK * at["first_block"], "ks_w_first")
        per = R.ATTENTION_BLOCK * bps
        for k in [k for k in range(16) if per >> k & 1]:
            key0 = builder.add(key0, builder.shl(sl, TR._i32_const(builder, k, "ks_w_s%d" % k), name="ks_w_t%d" % k),
                               name="ks_w_a%d" % k)
        builder.store_at(c, builder.add(slot.emit(builder, "ks_w0"), TR._i32_const(builder, sl_["W"] // 4, "ks_w0_off"),
                                        name="ks_w0_idx"), key0)
    builder.tensor_index_init("k", lay["KC"])
    builder.tensor_index_init("v", 0)
    counter0 = builder.const(0, name="ks_counter0")
    hdr, ex = fn.block("split_keyblock_loop"), fn.block("split_keyblock_exit")
    builder.br(hdr)
    builder.at(hdr)
    i = builder.phi(counter0, name="ks_block")
    builder.tensor_matmul(a, c, c, M=32, N=16, K=at["head"], transB=True, offsetA=0, offsetB=0,
                          offsetC=sl_["base"] + sl_["S"], head_stride=qk_head, head_slices=S, slice_stride=qk_slice,
                          **kreg)
    widx = key = None
    q0 = at["q0"]
    if masked:
        widx = builder.add(slot.emit(builder, "ks_w"), TR._i32_const(builder, sl_["W"] // 4, "ks_w_off"), name="ks_w_idx")
        key = builder.load(c, widx, type=ir.I32, name="ks_key0")
        if at.get("runtime_q0"):
            # THE RUNTIME LENGTH (MM 25.138.1): the query position, reloaded each trip (no scalar lives across a body)
            q0 = builder.load(c, TR._i32_const(builder, R.ATTENTION_GRID_LENGTH_BYTE // 4, "ks_len_idx"),
                              type=ir.I32, name="ks_q0")
    for r in range(at["rows"]):
        mask = TR.KeyMask(key, q0, r, at["mask_bias"]) if masked else None
        TR.emit_stream_second_block(builder, c, row=r, s_base=sl_["S"] // 4, o_base=o_tiles[0],
                                    m_base=sl_["Mst"] // 4, l_base=sl_["L"] // 4, mask=mask,
                                    head=slot, o_bases=o_tiles)
    if masked:
        builder.store_at(c, widx, builder.add(key, TR._i32_const(builder, R.ATTENTION_BLOCK, "ks_w_step"),
                                              name="ks_key_next"))
    for s_ in range(at["value"] // 16):
        builder.tensor_matmul(c, b, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=True,
                              offsetA=sl_["base"] + sl_["S"], offsetB=0, offsetC=sl_["base"] + sl_["O_tiles"][s_],
                              head_stride=pv_head, head_slices=S, slice_stride=pv_slice, **vreg)
    nxt = builder.add(i, ir.Imm(1), name="ks_block_next")
    ir.Builder.phi_latch(i, nxt)
    builder.br_cond(builder.cmp(nxt, bps, "lt", name="ks_more"), hdr, ex)
    builder.at(ex)


# THE S-WAY KV-SPLIT MERGE (MM 25.135.5; runtime.ATTENTION_GRID_MERGE_PHASE): its own dispatch, one threadgroup
# per head, over the split's un-normalised partials in the heads x S scratch slots of buffer 3, into each head's
# grid O / M / L, then the grid's normalisation.
GRID_MERGE_MODELS = ("merge",
                     # controls - claims about the SAME program's output, each predicted to fail:
                     "no_rescale",          # merge_without_rescale: f_s = 1
                     "wrong_slot_stride",   # slots read slice-major, tg = slice * heads + head
                     "drop_last_slice")     # the merge over slices 0 .. S-2
GRID_MERGE_EMPTY_M = np.float32(-3.4028234663852886e38)      # -FLT_MAX: an empty slice's m (the initial state)


def _grid_merge(s):
    at = s.get("attention")
    return bool(at) and at.get("phase") == "grid_merge"


def _grid_merge_generic_spec(spec):
    """generic_spec for phase grid_merge: the class's admission, its transport, heads threadgroups."""
    from agxforge.g17 import runtime as R
    raw = dict(spec["attention"])
    keep = ("attention", "attention_model", "c_from", "attention_seed", "grid_inputs")
    spec = {k: spec[k] for k in keep if k in spec} if "allow_value_change" in raw and "M" in spec else spec
    raw = {k: raw[k] for k in ("phase", "heads", "rows", "kv_split", "allow_value_change", "normalize", "tile_groups")
           if k in raw}
    extra = sorted(set(spec) - set(keep) - {"M", "N", "K", "a", "b"})
    if extra:
        raise ValueError("generic: a grid_merge spec carries only its attention block (and model); got %s" % extra)
    at = R.attention_spec(raw)
    lay = R.attention_layout(at)
    model = spec.get("attention_model", "merge")
    if model not in GRID_MERGE_MODELS:
        raise ValueError("generic: a grid_merge attention_model is one of %s" % (GRID_MERGE_MODELS,))
    if not spec.get("grid_inputs"):
        raise ValueError("generic: a grid_merge program's buffer 3 holds the split's partials; name them in "
                         "grid_inputs (an .npz of q, k0 and the partials, grid_merge_buffers' arguments)")
    s = generic_spec(dict(M=lay["M"], N=lay["N"], K=256))
    s.update(K=lay["K"], attention=at, attention_model=model, c_from=None,
             attention_seed=int(spec.get("attention_seed", 1729)), grid_inputs=spec.get("grid_inputs"),
             threadgroups=at["heads"] * at.get("tile_groups", 1))
    return s


def grid_merge_buffers(at, q, k0, O, Mst, L, *, junk_row20=12345.0):
    """a.f16, b.f16 and c.f32 of a merge dispatch. q: heads x rows x 128 (Q, for the carrier), k0: heads x 16 x
    128 (each head's K cache block 0, for the carrier), and the partials in the peer's contract: O heads x S x
    rows x 128 (un-normalised, at the slice's own max), Mst and L heads x S x rows. Buffer 3 is the sentinel
    everywhere else: the heads' grid O / M / L before the merge, every slot's S tile and its unattended rows.
    Each slot's M row 20 holds `junk_row20` (the split's per-trip key threshold lives there; the merge must
    not read it)."""
    from agxforge.g17 import runtime as R
    lay = R.attention_layout(at)
    st = lay["stride"]
    heads, rows, S = at["heads"], at["rows"], at["kv_split"]
    a = np.zeros(lay["M"] * lay["K"], np.float16)
    b = np.zeros(lay["K"] * lay["N"], np.float16)
    c = np.full(lay["M"] * lay["N"], ATTENTION_SENTINEL, dtype="<f4")
    for h in range(heads):
        qa = np.zeros((32, 128), np.float16)
        qa[:rows] = q[h]
        a[h * st["A"] // 2: h * st["A"] // 2 + qa.size] = qa.ravel()
        base = h * st["C"] + lay["KC"]
        c.view(np.uint8)[base:base + 4096] = np.frombuffer(np.asarray(k0[h], np.float16).tobytes(), np.uint8)
        for sl in range(S):
            w = R.kv_split_slot(heads, S, h, sl) // 4
            for t in range(8):
                ob = w + (R.KV_SLOT["O"] + 2048 * t) // 4
                c[ob:ob + 16 * rows] = np.asarray(O[h][sl], np.float32)[:, 16 * t:16 * t + 16].ravel()
            mb, lb = w + R.KV_SLOT["M"] // 4, w + R.KV_SLOT["L"] // 4
            for r in range(rows):
                c[mb + 16 * r: mb + 16 * r + 16] = Mst[h][sl][r]
                c[lb + 16 * r: lb + 16 * r + 16] = L[h][sl][r]
            c[mb + 320: mb + 336] = junk_row20
    return a, b, c


def _grid_merge_inputs(bundle, s, spec):
    z = np.load(s["grid_inputs"])
    a, b, c = grid_merge_buffers(s["attention"], z["q"], z["k0"], z["O"], z["M"], z["L"])
    (bundle / "a.f16").write_bytes(a.tobytes())
    (bundle / "b.f16").write_bytes(b.tobytes())
    (bundle / "c.f32").write_bytes(c.tobytes())


def grid_merge_partials(c0, at, h, model="merge"):
    """Head h's S partials as buffer 3 holds them: ([m_s per row], [l_s per row], [O_s rows x 128]) per slice.
    The wrong_slot_stride claim reads slot tg = slice * heads + head (slice-major) instead of head * S + slice."""
    from agxforge.g17 import runtime as R
    heads, rows, S = at["heads"], at["rows"], at["kv_split"]
    out = []
    for sl in range(S):
        if model == "wrong_slot_stride":
            tg = sl * heads + h
            w = (R.ATTENTION_GRID_STRIDE["C"] * heads + tg * R.KV_SLOT_BYTES) // 4
        else:
            w = R.kv_split_slot(heads, S, h, sl) // 4
        O = np.concatenate([c0[w + (R.KV_SLOT["O"] + 2048 * t) // 4: w + (R.KV_SLOT["O"] + 2048 * t) // 4 + 512]
                            .reshape(32, 16) for t in range(8)], axis=1)[:rows]
        m = [float(c0[w + R.KV_SLOT["M"] // 4 + 16 * r]) for r in range(rows)]
        l = [float(c0[w + R.KV_SLOT["L"] // 4 + 16 * r]) for r in range(rows)]
        out.append((m, l, O))
    return out


def _grid_merge_trace(bundle, s, ar, model=None):
    """{word: ar value} of every word the merge program writes: each head's carrier S (32 x 16: Q K_0^T, a
    point), and per attended row the merged (then normalised) O words, M and L (16 words each)."""
    import g17decodestep as D
    from agxforge.g17 import runtime as R
    at = s["attention"]
    model = model or s["attention_model"]
    lay = R.attention_layout(at)
    a0 = np.frombuffer((bundle / "a.f16").read_bytes(), np.float16)
    c0 = np.frombuffer((bundle / "c.f32").read_bytes(), dtype="<f4").copy()
    st = lay["stride"]
    words = {}
    for h in range(at["heads"]):
        base = h * st["C"] // 4
        q = a0[h * st["A"] // 2: h * st["A"] // 2 + 32 * 128].reshape(32, 128).astype(np.float32)
        k0 = np.frombuffer(c0.view(np.uint8)[h * st["C"] + lay["KC"]: h * st["C"] + lay["KC"] + 4096].tobytes(),
                           np.float16).reshape(16, 128).astype(np.float32)
        S = D.gemm(q, k0.T.copy())
        for off in grid_merge_carriers(lay):
            for i, x in enumerate(S.ravel()):
                words[base + off // 4 + i] = ar.val(float(x))
        parts = grid_merge_partials(c0, at, h, "merge" if model in ("no_rescale", "drop_last_slice") else model)
        dmodel = model if model in D.KV_MERGE_MODELS else "merge"
        for r in range(at["rows"]):
            big, l, o, _f = D.kv_merge_trace(ar, [ar.val(p[0][r]) for p in parts], [ar.val(p[1][r]) for p in parts],
                                             [[ar.val(float(x)) for x in p[2][r]] for p in parts], dmodel)
            if at["normalize"]:
                # a claim can merge only empty partials (l = 0): recip(0) is +inf and 0 x inf NaN, as IEEE
                inv = math.inf if isinstance(ar, _StreamPoint) and l == 0 else ar.recip(l)
                o = [ar.fmul(x, inv) for x in o]
            for t in range(8):
                ob = base + lay["O_tiles"][t] // 4 + 16 * r
                for col in range(16):
                    words[ob + col] = o[16 * t + col]
            for col in range(16):
                words[base + lay["Mst"] // 4 + 16 * r + col] = big
                words[base + lay["L"] // 4 + 16 * r + col] = l
    return words, c0


def grid_merge_reference(bundle, s, model=None):
    """Buffer 3 as the merge program leaves it: every word it does not write keeps its input bits."""
    from agxforge.g17 import runtime as R
    lay = R.attention_layout(s["attention"])
    words, c0 = _grid_merge_trace(bundle, s, _StreamPoint(), model)
    out = c0.copy()
    for i, x in words.items():
        out[i] = x
    return out.reshape(lay["M"], lay["N"])


def grid_merge_bound(bundle, s):
    """Per word, the distance from the reference to the ends of the hardware enclosure (exp2 within one ulp,
    recip assumed within one ulp, every rounding outward); zero on every word no transcendental reaches (the
    carrier S, the merged M, everything the program does not write)."""
    from agxforge.g17 import runtime as R
    lay = R.attention_layout(s["attention"])
    ref = grid_merge_reference(bundle, s).astype(np.float64).ravel()
    words, _c0 = _grid_merge_trace(bundle, s, _StreamInterval("hardware"))
    bound = np.zeros_like(ref)
    for i, iv in words.items():
        if not (iv[0] <= ref[i] <= iv[1]):
            raise ValueError("grid merge model: the reference lies outside its own enclosure at word %d" % i)
        bound[i] = max(ref[i] - iv[0], iv[1] - ref[i])
    return bound.reshape(lay["M"], lay["N"])


def grid_merge_rows(got, at, h):
    """Head h's merged state from a buffer-3 image: (O rows x 128, M [rows], L [rows])."""
    from agxforge.g17 import runtime as R
    lay = R.attention_layout(at)
    c = np.asarray(got, dtype="<f4").ravel()
    base = h * lay["stride"]["C"] // 4
    O = np.concatenate([c[base + t // 4: base + t // 4 + 512].reshape(32, 16) for t in lay["O_tiles"]], axis=1)
    return (O[:at["rows"]], c[base + lay["Mst"] // 4 + 16 * np.arange(at["rows"])],
            c[base + lay["L"] // 4 + 16 * np.arange(at["rows"])])


def grid_merge_carriers(lay):
    """The carrier bodies' C offsets in a head's region: the S tile, and the 2,048 bytes after L (92,160)."""
    return (lay["S"], lay["L"] + 2048)


def _build_attention_grid_merge(builder, a, b, c, at):
    """PHASE GRID_MERGE (MM 25.135.5). The CARRIER first: the grid's own QK body on K cache block 0, twice, into
    the head's S tile and into the unused 2,048 bytes after L (grid_merge_carriers). The common worker admits
    tensor programs only, and the head stride exists only on the memory-stream route, which takes two or more
    bodies; neither region is read by the merge, and no scalar is live across the bodies. Then per attended row tensorreduce.emit_kv_merge_n from the S slots
    (threadgroup t's slots start t * S * 5,632 words past the slot base: ScaledBase) into the head's grid O / M
    / L (HeadBase, t << 15 words), then (normalize) the grid's emit_stream_normalize on the same tiles."""
    from agxforge.g17 import runtime as R, tensorreduce as TR
    lay = R.attention_layout(at)
    st = lay["stride"]
    S = at["kv_split"]
    T = at.get("tile_groups", 1)
    if T > 1:
        return _build_attention_grid_merge_tiled(builder, a, b, c, at, lay, T)
    for off in grid_merge_carriers(lay):
        builder.tensor_matmul(a, c, c, M=32, N=16, K=at["head"], transB=True, offsetA=0, offsetB=lay["KC"],
                              offsetC=off, head_stride=(st["A"], st["C"], st["C"]))
    shift = (st["C"] // 4).bit_length() - 1
    head = TR.HeadBase(shift)
    slot_head = TR.ScaledBase(S * R.KV_SLOT_BYTES // 4)
    o_tiles = [t // 4 for t in lay["O_tiles"]]
    dst = (o_tiles, lay["Mst"] // 4, lay["L"] // 4, head)
    srcs = []
    for sl in range(S):
        w = (lay["slots_base"] + sl * R.KV_SLOT_BYTES) // 4
        srcs.append(([w + (R.KV_SLOT["O"] + 2048 * t) // 4 for t in range(8)], w + R.KV_SLOT["M"] // 4,
                     w + R.KV_SLOT["L"] // 4, slot_head))
    for r in range(at["rows"]):
        TR.emit_kv_merge_n(builder, c, row=r, dst=dst, srcs=srcs)
    if at["normalize"]:
        for r in range(at["rows"]):
            TR.emit_stream_normalize(builder, c, row=r, o_base=o_tiles[0], l_base=lay["L"] // 4, head=head,
                                     o_bases=o_tiles)


def _build_attention_grid_merge_tiled(builder, a, b, c, at, lay, T):
    """THE TILED MERGE (MM 25.132.7): threadgroup t = head * T + tile, T = 8, one 16-wide O tile each. The carrier
    is the per-head merge's, with the tile groups of a head writing the SAME carrier words (head = t >> 3, a zero
    slice stride), so its values are the per-head program's. Each group then runs emit_kv_merge_n over its one O
    tile: M, every f_s and l are recomputed identically in every group of a head (same loads, same order), the O
    tile is its own, and M and l are written by all eight groups with the same bits. Then the normalisation of
    its tile. Every word the program writes is the per-head merge's word, by construction."""
    from agxforge.g17 import runtime as R, tensorreduce as TR
    st = lay["stride"]
    S = at["kv_split"]
    for off in grid_merge_carriers(lay):
        builder.tensor_matmul(a, c, c, M=32, N=16, K=at["head"], transB=True, offsetA=0, offsetB=lay["KC"],
                              offsetC=off, head_stride=(st["A"], st["C"], st["C"]), head_slices=T,
                              slice_stride=(0, 0, 0))
    hw = st["C"] // 4
    tile = 2048 // 4
    head = TR.TileGroupBase(T, hw)
    ohead = TR.TileGroupBase(T, hw, tile)
    slot_words = S * R.KV_SLOT_BYTES // 4
    shead = TR.TileGroupBase(T, slot_words)
    sohead = TR.TileGroupBase(T, slot_words, tile)
    o0 = lay["O_tiles"][0] // 4
    dst = ([o0], lay["Mst"] // 4, lay["L"] // 4, head, ohead)
    srcs = []
    for sl in range(S):
        w = (lay["slots_base"] + sl * R.KV_SLOT_BYTES) // 4
        srcs.append(([w + R.KV_SLOT["O"] // 4], w + R.KV_SLOT["M"] // 4, w + R.KV_SLOT["L"] // 4, shead, sohead))
    for r in range(at["rows"]):
        TR.emit_kv_merge_n(builder, c, row=r, dst=dst, srcs=srcs)
    if at["normalize"]:
        for r in range(at["rows"]):
            TR.emit_stream_normalize(builder, c, row=r, o_base=o0, l_base=lay["L"] // 4, head=head, o_bases=[o0],
                                     o_head=ohead)


def build_generic_program(spec):
    """tlower's K-loop row bases (spec kloop_bases, MM 25.144.1) apply to this build only; the module switch is restored."""
    from agxforge.g17 import tlower as _tl
    saved = _tl.KLOOP_BASES
    _tl.KLOOP_BASES = bool(spec.get("kloop_bases", False))
    try:
        return _build_generic_program(spec)
    finally:
        _tl.KLOOP_BASES = saved


def _build_generic_program(spec):
    """The GEMM, then (one threadgroup only) the standard C[0,0] += 1 epilogue that supplies SR156;
    with a simdgroup split the epilogue runs in simdgroup 0 alone, because the other simdgroups'
    lanes would race simdgroup 0's tensor stores of row 0. A grid split reads SR156 itself."""
    from agxforge.g17 import cc, ir

    s = generic_spec(spec)
    a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function(GENERIC_NAME, [a, b, c])
    builder = ir.Builder(fn, fn.block("entry"))
    epilogue = []
    for step in s["epilogue"]:
        if step == "gelu" and s["gelu_stage"] == "memory":
            continue                                   # built after the GEMM as the memory stage
        epilogue.append(("relu",) if step == "relu" else ("exp2",) if step == "exp2" else ("gelu",) if step == "gelu" else
                        ("mx", 0, s["M"] * s["K"] * GENERIC_TYPES[s["a"]], 1, s["K"] * s["N"] * GENERIC_TYPES[s["b"]])
                        if step == "mx32" else
                        ("mx", 0, s["M"] * s["K"] * GENERIC_TYPES[s["a"]], 1, s["K"] * s["N"] * GENERIC_TYPES[s["b"]], "e8m0")
                        if step == "mx32e8m0" else
                        ("fp8", FP8_OUT[step]) if step in FP8_OUT else ("half",) if step == "half" else
                        ("scale", int(step.split(":")[1], 16)))
    if s.get("attention"):
        _build_attention(builder, a, b, c, s["attention"])
    elif s["stream"] and s["key_blocks"]:
        _build_keyblock_attention(builder, a, b, c, s["stream"], s["key_blocks"],
                                  frozen=s["key_advance"] == "frozen")
    elif s["stream"]:
        (_build_oneshot_attention if s["stream_program"] == "oneshot" else _build_stream_attention)(
            builder, a, b, c, s["stream"])
    elif s["independent"]:
        rows = s["M"] // s["independent"]
        for k in range(s["independent"]):
            builder.tensor_matmul(a, b, c, M=rows, N=s["N"], K=s["K"],
                                  offsetA=0 if s["independent_wrong_a"] else k * rows * s["K"] * 2,
                                  offsetC=k * rows * s["N"] * 4)
    elif s["c_feed"]:
        # THE ACCUMULATOR FEED (P2): adjacent bodies, the second accumulating onto the first's C with
        # its own A and B halves. With the feed table's key present the compiler hands D1 over in
        # registers; with feed "memory" it is the memory stream (D1 stored, then loaded as C).
        k2 = s["K"] // 2
        builder.tensor_matmul(a, b, c, M=s["M"], N=s["N"], K=k2)
        builder.tensor_matmul(a, b, c, M=s["M"], N=s["N"], K=k2, accumulate=True,
                              offsetA=s["M"] * k2 * 2, offsetB=k2 * s["N"] * 2)
    elif s["stages"]:
        # A CHAIN: stage 0 from A and B into C, each later stage reading C as its A (fp32, or narrowed
        # to half in registers). No strides: the composition route's chain arm is contiguous.
        builder.tensor_matmul(a, b, c, M=s["M"], N=s["N"], K=s["K"])
        if s["between"] and str(s["between"]).startswith("softmax:"):
            # ONE-DISPATCH ATTENTION: the compiler-owned stable row softmax (base 2) on the stored
            # scores between the two GEMMs. Measured rows only (0..15 of the first tile); the
            # composition route's register reservation admits two rows today (machine model 25.102).
            for row in range(int(str(s["between"]).split(":")[1])):
                builder.tensor_row_softmax(c, row=row, M=s["M"], N=s["N"], K=s["K"])
        if s["between"] == "add1":
            at0 = builder.builtin("threadgroup_position_in_grid", name="between_at")
            staged = builder.load(c, at0, type=ir.F32, name="between_value")
            builder.store_at(c, at0, builder.fadd(staged, builder.const(struct.unpack("<I", struct.pack("<f", 1.0))[0],
                                                                        name="between_one"), name="between_plus_one"))
        for st in s["stages"]:
            n, k, t = st[:3]
            mode = st[3] if len(st) == 4 else "A"
            conv = "float" if t == "half" else None
            if mode == "A" and len(st) == 3:          # the released stage spelling, byte for byte
                builder.tensor_matmul(c, b, c, M=s["M"], N=n, K=k, a_dtype=t, b_dtype="half",
                                      a_converted_from=conv, a_staged=s["stage_through"])
            elif mode in ("A", "At"):                 # the fed D is this body's A (At: under transA)
                builder.tensor_matmul(c, b, c, M=s["M"], N=n, K=k, a_dtype=t, b_dtype="half",
                                      a_converted_from=conv, feed=mode)
            else:                                     # B / Bt: A from buffer 2, the fed D is B
                builder.tensor_matmul(b, c, c, M=s["M"], N=n, K=k, a_dtype="half", b_dtype=t,
                                      b_converted_from=conv, feed=mode)
    else:
        # MM P13 transposed class: a stored A^T is K rows of M elements, a stored B^T N rows of K
        builder.tensor_matmul(a, b, c, M=s["M"], N=s["N"], K=s["K"], a_dtype=s["a"], b_dtype=s["b"],
                              strideA=(s["M"] if s["transA"] else s["K"]) * GENERIC_TYPES[s["a"]],
                              strideB=(s["K"] if s["transB"] else s["N"]) * GENERIC_TYPES[s["b"]],
                              transA=s["transA"], transB=s["transB"],
                              threadgroups=s["threadgroups"], grid_n=s["grid_n"], split_k=s["split_k"], epilogue=tuple(epilogue), split_fp32=s["split_fp32"],
                              accumulate=s["accumulate"], saturate=s["saturate"], kloop=s["kloop"],
                              kloop_unroll=s["kloop_unroll"],
                              reduce=(("col", s["reduce"][3:]) if s["reduce"] else None))
        if s["gelu_stage"] == "memory":
            builder.tensor_tile_gelu(c, M=16, N=16, K=s["K"])
    op = next(o for blk in fn.blocks for o in blk.ops if o.kind == "tensor_matmul")   # stage 0
    if s["simdgroups"] > 1:
        op.attrs["simdgroups"] = s["simdgroups"]
    # round-4 arms (docs/archive/g17-split-guard-prereg.md): "none" drops the guard (both simdgroups add);
    # "always" guards with a condition true on every simdgroup; a "_const" arm stores 1234.0
    # instead of loading and adding, so a lost load and a lost store are told apart
    guarded = s["simdgroups"] > 1 and s["guard"] not in ("none", "none_const")
    const_store = s["guard"].endswith("_const")
    if s["imageblock"]:
        # EACH LANE STAGES ONE OUTPUT THROUGH TILE MEMORY: C[lane] -> its imageblock element ->
        # back, plus 1. The lane index is thread_position_in_threadgroup.x, the SR the imageblock
        # coordinate already reads, so the program's set is (130, 164, 165) exactly as Apple's
        # tensor+imageblock witness (results/g17-tensor-imageblock-witness-v1).
        lane = builder.builtin("thread_position_in_threadgroup", name="lane")
        early_x = None
        if s["imageblock"] == "x_neighbour_early":
            # APPLE'S ORDER: the column is computed BEFORE the store, so it cannot land in the store's
            # value register. In `x_neighbour` the allocator reused that register (released by the
            # store) for the column, AFTER the store and barrier, and every lane read 1.0 (Set C,
            # round 5): consistent with the store reading its source late and writing the column.
            early_x = getattr(builder, "and")(
                builder.add(lane, builder.const(1, name="step"), name="lane_plus_one"),
                builder.const(31, name="wrap"), name="neighbour_x")
        builder.imageblock_write(builder.load(c, lane, type=ir.F32, name="staged"), member=0)
        one = builder.const(struct.unpack("<I", struct.pack("<f", 1.0))[0], name="one")
        if s["imageblock"].startswith("x_"):
            # AN EXPLICIT COLUMN, as Apple addresses another lane's element (tensor kernel or not):
            # x = lane (the control) or x = (lane + 1) & 31, read with no offset after an imageblock
            # barrier. The dx operand did nothing in a tensor program on hardware (rounds 2 to 4).
            # Lane 31 wraps to column 0, so all 32 lanes are scored.
            builder.barrier("imageblock")
            x = lane if s["imageblock"] == "x_own" else early_x if early_x is not None else getattr(builder, "and")(
                builder.add(lane, builder.const(1, name="step"), name="lane_plus_one"),
                builder.const(31, name="wrap"), name="neighbour_x")
            back = builder.imageblock_read(member=0, x=x, type=ir.F32, name="column_value")
            builder.store_at(c, lane, builder.fadd(back, one, name="column_plus_one"))
        elif s["imageblock"].startswith("neighbour"):
            # A NEIGHBOUR'S ELEMENT, not this lane's own: an own-element round trip passed with the
            # declaration withheld and 0 tile bytes allocated (Set C, 2026-09-23), so something
            # lane-private can serve it. Lane L reads x = L+1 after an imageblock barrier (Apple's
            # op447, scope 0x47); lane 31 is fenced out by a cmp region so nothing reads outside the
            # 32x1 tile, and C[0,31] keeps the GEMM value.
            builder.barrier("imageblock")
            if s["imageblock"] == "neighbour_open":
                # NO REGION: every lane reads x = lane+1, lane 31 outside the tile and unscored,
                # exactly Set C's validated dx1 program (whose read is byte-identical to this one's).
                # The fenced arm returned its own element; this arm asks whether the region did it.
                back = builder.imageblock_read(member=0, dx=1, type=ir.F32, name="neighbour_value")
                builder.store_at(c, lane, builder.fadd(back, one, name="neighbour_plus_one"))
            else:
                region, join = fn.block("neighbour"), fn.block("join")
                builder.br_cond(builder.cmp(lane, 31, "lt", name="has_neighbour"), region, join)
                builder = ir.Builder(fn, region)
                back = builder.imageblock_read(member=0, dx=1, type=ir.F32, name="neighbour_value")
                builder.store_at(c, lane, builder.fadd(back, one, name="neighbour_plus_one"))
                builder.br(join)
                builder = ir.Builder(fn, join)
        else:
            back = builder.imageblock_read(member=0, type=ir.F32, name="unstaged")
            builder.store_at(c, lane, builder.fadd(back, one, name="unstaged_plus_one"))
    elif s["a"] == "int8" and (s["simdgroups"], s["threadgroups"], s["grid_n"], s["split_k"]) == (1, 1, 1, 1):
        # int32 C: the same C[0,0] += 1 tail as an integer add (it wraps; it supplies SR156). Under a
        # split (MM P13's int8_split class) the tail takes the float path's guard and grid rules below
        position = builder.builtin("threadgroup_position_in_grid", name="group_x")
        loaded = builder.load(c, position, type=ir.I32, name="c_value")
        builder.store_at(c, position, builder.add(loaded, builder.const(1, name="one"), name="c_plus_one"))
    elif (s["threadgroups"] == 1 and s["grid_n"] == 1 and s["split_k"] == 1 and not s["stage_through"]
          and not _grid(s) and not _grid_merge(s)):
        # (phase grid has no tail: its row stages read SR156 themselves, and at one head C[0] is its K cache)
        # (a split-K grid, like grid_n, reads SR156 itself for its slot addressing: no C[0,0]+=1 tail)
        # (an imageblock-staged chain has no SR156 tail: its SR set stays the measured (130, 164, 165))
        # THE GUARD IS A BRANCH REGION, not exec_mask(icmp): an icmp is a 0/1 value and op582
        # needs a compare's FLAG, so that form's mask was empty on every lane (round 4; cc now
        # refuses exec_mask). The measured form is cmp -> op582 -> region -> op577, which
        # br_cond(cmp) lowers to. cmp relations are lt/gt only: "sg == 0" is "sg < 1", "sg == 1"
        # is "sg > 0". The round-1 and round-4 guard bundles hold the refused form's bytes.
        if guarded:
            region, join = fn.block("tail"), fn.block("join")
            sg = builder.builtin("SR_SIMD_GRP", name="epilogue_sg")   # simdgroup_index_in_threadgroup
            # the mask is 3 up to 4 simdgroups (the measured bytes) and sg - 1 above: at 8, a mask
            # of 3 would let simdgroup 4 pass the sg0 guard too (MM P13)
            sg = getattr(builder, "and")(sg, builder.const(3 if s["simdgroups"] <= 4 else s["simdgroups"] - 1,
                                                          name="sg_mask"), name="epilogue_sg_index")
            if s["guard"] == "barrier_sg0":
                builder.barrier("threadgroup")
            rel, bound = {"sg1": ("gt", 0), "always": ("lt", 4)}.get(s["guard"], ("lt", 1))
            builder.br_cond(builder.cmp(sg, bound, rel, name="picked_sg"), region, join)
            builder = ir.Builder(fn, region)
        position = builder.builtin("threadgroup_position_in_grid", name="group_x")
        if const_store:
            builder.store_at(c, position, builder.const(struct.unpack("<I", struct.pack("<f", 1234.0))[0],
                                                        name="marker"))
        elif s["a"] == "int8":                         # int32 C under a simdgroup split (MM P13)
            loaded = builder.load(c, position, type=ir.I32, name="c_value")
            builder.store_at(c, position, builder.add(loaded, builder.const(1, name="one"), name="c_plus_one"))
        else:
            loaded = builder.load(c, position, type=ir.F32, name="c_value")
            one = builder.const(struct.unpack("<I", struct.pack("<f", 1.0))[0])
            builder.store_at(c, position, builder.fadd(loaded, one, name="c_plus_one"))
        if guarded:
            builder.br(join)
            builder = ir.Builder(fn, join)
    builder.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def _generic_values(raw, dtype):
    """Exact float32 values of stored operands (ml_dtypes decodes bfloat and fp8)."""
    import ml_dtypes
    kinds = {"half": np.float16, "bfloat": ml_dtypes.bfloat16, "float": np.float32,
             "fp8e4m3": ml_dtypes.float8_e4m3fn, "fp8e5m2": ml_dtypes.float8_e5m2}
    return np.frombuffer(bytes(raw), dtype=kinds[dtype]).astype(np.float32)


def _generic_draw(rng, count, dtype):
    import ml_dtypes
    if dtype in ("fp8e4m3", "fp8e5m2"):
        return _fp8_codes(rng, count, dtype[3:]).tobytes()
    if dtype == "int8":
        return rng.integers(-128, 128, size=count).astype(np.int8).tobytes()
    values = rng.uniform(-2.0, 2.0, size=count)
    kind = {"half": np.float16, "bfloat": ml_dtypes.bfloat16, "float": np.float32}[dtype]
    return values.astype(kind).tobytes()


def _generic_chain_reference(bundle, s):
    """The C buffer after the chain, modelling the program's memory writes exactly: stage j's D is
    stored (rows at stride N_j) unless the next stage takes it in registers with the same width
    (cc's "elide": the consumer overwrites every element); the last stage always stores. The next
    stage's A is D_j in registers: fp32 (truncated by the MMA) or its RNE half narrowing."""
    M = s["M"]
    if any(len(st) == 4 and st[3] != "A" for st in s["stages"]):
        return _generic_feed_reference(bundle, s)
    widths = [s["N"]] + [st[0] for st in s["stages"]]
    buf = np.zeros(M * max(widths), dtype="<f4")
    bflat = _generic_values((bundle / "b.f16").read_bytes(), "half")
    a = _generic_values((bundle / "a.f16").read_bytes(), "half").reshape(M, s["K"])
    d = _gemm_mma(a, bflat[:s["K"] * s["N"]].reshape(s["K"], s["N"]), None, M, s["N"], s["K"])
    for j, st in enumerate(s["stages"]):
        n, k, t = st[:3]
        if widths[j] != n or (j == 0 and s.get("between")):  # kept, or bridged through memory
            buf[:M * widths[j]] = d.reshape(-1)
        if j == 0 and str(s.get("between") or "").startswith("softmax:"):
            # the released reference's base-2 stable softmax, per row, on the stored scores
            w0 = widths[j]
            for row in range(int(str(s["between"]).split(":")[1])):
                r = np.asarray(buf[row * w0:(row + 1) * w0], dtype="<f4")
                weights = np.exp2(np.asarray(r - np.float32(np.max(r)), dtype="<f4")).astype("<f4")
                buf[row * w0:(row + 1) * w0] = (weights / np.float32(np.sum(weights, dtype="<f4"))).astype("<f4")
            d = buf[:M * w0].reshape(M, w0).copy()
        if j == 0 and s.get("between") == "add1":
            # the scalar step between the bodies, on the stored C, which the next body then reads
            buf[0] = _rne32(buf[0] + np.float32(1.0))
            d = buf[:M * widths[j]].reshape(M, widths[j]).copy()
        bj = bflat[:k * n].reshape(k, n)
        if t == "half":
            d = _gemm_mma(np.asarray(d, np.float16).astype(np.float32), bj, None, M, n, k)
        else:
            d = _gemm_mma(d, bj, None, M, n, k, truncate_a=True)
    buf[:M * widths[-1]] = d.reshape(-1)
    if not s.get("stage_through"):
        buf[0] = _rne32(buf[0] + np.float32(1.0))
    return buf.reshape(M, max(widths))


def _rotl1(k):
    return ((k << 1) | (k >> 3)) & 15


def _rotr1(k):
    return ((k >> 1) | ((k & 1) << 3)) & 15


def feed_operand(H, mode):
    """SECTION 132'S RIVAL, NOT THIS COMPILER'S HARDWARE BEHAVIOUR. On hardware, tlower's feeds read the
    logical D (B) or D^T (At, Bt) bit for bit, and every arm checked against this function failed
    (results/g17-tensor-feedmodes-v1). The rotations below belong to kernels whose loads put A and D
    at pos and B at pos_b; tlower labels every tile's rows alike. Kept as the discriminating control.

    The operand the consumer's MMA reads in such a kernel when the producer's D (H, M x M, whole
    16-tiles) is fed in `mode`, from recon section 132 part 2's fragment facts, tile by tile:
        B   B_eff[16j+k][16n+c]  = H[16j+rotl1(k)][16n+c]
        At  A_eff[16a+r][16t+kk] = H[16t+rotl1(kk)][16a+rotr1(r)]
        Bt  B_eff[16t+k][16i+c]  = H[16i+rotl1(c)][16t+k]"""
    M = H.shape[0]
    out = np.empty_like(H)
    for ti in range(0, M, 16):
        for tj in range(0, M, 16):
            for x in range(16):
                for y in range(16):
                    if mode == "B":
                        out[ti + x, tj + y] = H[ti + _rotl1(x), tj + y]
                    elif mode == "At":
                        out[ti + x, tj + y] = H[tj + _rotl1(y), ti + _rotr1(x)]
                    elif mode == "Bt":
                        out[ti + x, tj + y] = H[tj + _rotl1(y), ti + x]
    return out


def column_reduction(d, operation):
    """The register-resident column reduction as tlower emits it, lane by lane. Lane L holds tile
    rows r0 = 4*bit4(L) + ((L >> 1) & 3) and r0 + 8, columns 8*bit3(L) + 4*bit0(L) + j (pos_b). Its
    value for slot j folds the row tiles in ascending order (rows r0, then r0 + 8 separately), then
    adds the r0 + 8 fold into the r0 fold; then the xor butterfly over lane masks 2, 4, 16
    (tensorreduce.butterfly, the measured model). Returns one value per column."""
    from agxforge.g17 import tensorreduce as TR
    comb = TR._fadd if operation == "sum" else TR._fmax
    M, N = d.shape
    out = np.empty(N, dtype="<f4")
    for tn in range(0, N, 16):
        for j in range(4):
            lane_vals = []
            for L in range(32):
                r0 = 4 * ((L >> 4) & 1) + ((L >> 1) & 3)
                col = tn + 8 * ((L >> 3) & 1) + 4 * (L & 1) + j
                lo = hi = None
                for tm in range(0, M, 16):
                    a_, b_ = float(d[tm + r0, col]), float(d[tm + r0 + 8, col])
                    lo = TR._round_fp32(a_) if lo is None else comb(lo, a_)
                    hi = TR._round_fp32(b_) if hi is None else comb(hi, b_)
                lane_vals.append(comb(lo, hi))
            result = TR.butterfly(lane_vals, TR.COLUMN_BUTTERFLY_MASKS, operation)
            for L in range(32):
                out[tn + 8 * ((L >> 3) & 1) + 4 * (L & 1) + j] = result[L]
    return out


def _generic_feed_reference(bundle, s):
    """C after a one-stage register-fed chain in mode B, At or Bt (square, so D1 is elided and C is
    D2 alone, plus the C[0,0] += 1 tail). The consumer's MMAs read `feed_operand(H)`; the memory
    operand is buffer 2's first M x M halves as stored. This models what the hardware computes, not
    the logical product: the host's operand arrangement and the output relabeling of section 132
    part 3 are the caller's to apply."""
    M, K = s["M"], s["K"]
    n, k, t, mode = s["stages"][0]
    bflat = _generic_values((bundle / "b.f16").read_bytes(), "half")
    a = _generic_values((bundle / "a.f16").read_bytes(), "half").reshape(M, K)
    d1 = _gemm_mma(a, bflat[:K * M].reshape(K, M), None, M, M, K)
    H = np.asarray(d1, np.float16).astype(np.float32) if t == "half" else d1
    if s.get("feed_model") == "identity":
        fed = H if mode == "B" else H.T.copy()        # the logical operand: D, or D^T for At and Bt
    else:
        fed = feed_operand(H, mode)
    mem = bflat[:M * M].reshape(M, M)
    if mode == "At":
        d2 = _gemm_mma(fed, mem, None, M, M, M, truncate_a=(t == "float"))
    else:
        d2 = _gemm_mma(mem, fed, None, M, M, M, truncate_b=(t == "float"))
    out = np.asarray(d2, dtype="<f4").copy()
    out[0, 0] = _rne32(out[0, 0] + np.float32(1.0))
    return out


def generic_int_seed(rng, M, N):
    """int32 C seeds: a quarter within 300,000 of INT32_MAX, a quarter as close to INT32_MIN (a
    16-product issue reaches +-262,144, so these rails are hit and left), half small (no model can
    saturate them: the control that every model agrees where nothing overflows)."""
    c = rng.integers(-1_000_000, 1_000_001, size=(M, N)).astype(np.int64)
    kind = rng.integers(0, 4, size=(M, N))
    c[kind == 0] = INT32_MAX - rng.integers(0, 300_001, size=int((kind == 0).sum()))
    c[kind == 1] = INT32_MIN + rng.integers(0, 300_001, size=int((kind == 1).sum()))
    return c.astype("<i4")


def generic_int_models(a, b, c, M, N, K):
    """The four readings of D = C + A.B in int32, each exact integer arithmetic then a rule:
    wrap (mod 2^32); clip after every product, C first (the rival section 136 refuted for one MMA,
    kept so a chain cannot quietly match it); clip after every 16-product issue (the issue sum
    exact: section 136's one-MMA law, repeated per issue); clip once at the end (what a chain would
    give if the bit saturated only the final write). The preregistration names each arm's model.""" 
    a = a.astype(np.int64).reshape(M, K); b = b.astype(np.int64).reshape(K, N); c = c.astype(np.int64)
    clip = lambda x: np.clip(x, INT32_MIN, INT32_MAX)
    exact = c + a @ b
    wrap = ((exact - INT32_MIN) % (1 << 32)) + INT32_MIN
    per_product = c.copy(); per_issue = c.copy()
    for k0 in range(0, K, 16):
        for j in range(k0, k0 + 16):
            per_product = clip(per_product + np.outer(a[:, j], b[j, :]))
        per_issue = clip(per_issue + a[:, k0:k0 + 16] @ b[k0:k0 + 16, :])
    return {name: v.astype("<i4") for name, v in (("wrap", wrap), ("clip_product", per_product),
                                                   ("clip_issue", per_issue), ("clip_end", clip(exact)))}


def generic_reference(bundle, s):
    if s.get("attention"):
        return attention_reference(bundle, s)
    if s.get("stream"):
        return stream_reference(bundle, s)
    if s.get("reduce"):
        M, N, K = s["M"], s["N"], s["K"]
        a = _generic_values((bundle / "a.f16").read_bytes(), "half").reshape(M, K)
        b = _generic_values((bundle / "b.f16").read_bytes(), "half").reshape(K, N)
        out = np.frombuffer((bundle / "c.f32").read_bytes(), dtype="<f4").reshape(M, N).copy()
        d = _gemm_mma(a, b, None, M, N, K)
        if s.get("reduce_model") == "exact":
            out[0, :] = np.asarray(d.astype(np.float64).sum(axis=0) if s["reduce"] == "colsum"
                                   else d.max(axis=0), dtype="<f4")
        else:
            out[0, :] = column_reduction(d, s["reduce"][3:])
        out[0, 0] = _rne32(out[0, 0] + np.float32(1.0))        # the C[0,0] += 1 tail
        return out
    if s["stages"]:
        return _generic_chain_reference(bundle, s)
    if s.get("c_feed"):
        M, N, k2 = s["M"], s["N"], s["K"] // 2
        a = _generic_values((bundle / "a.f16").read_bytes(), "half")
        b = _generic_values((bundle / "b.f16").read_bytes(), "half")
        a1, a2 = a[:M * k2].reshape(M, k2), a[M * k2:2 * M * k2].reshape(M, k2)
        b1, b2 = b[:k2 * N].reshape(k2, N), b[k2 * N:2 * k2 * N].reshape(k2, N)
        d1 = _gemm_mma(a1, b1, None, M, N, k2)
        out = _gemm_mma(a2, b2, d1 if s["c_feed_model"] == "accumulate" else None, M, N, k2)
        out[0, 0] = _rne32(out[0, 0] + np.float32(1.0))        # the C[0,0] += 1 tail
        return out
    if s["a"] == "int8":
        M, N, K = s["M"], s["N"], s["K"]
        c = np.frombuffer((bundle / "c.f32").read_bytes(), dtype="<i4").reshape(M, N)
        a8 = np.frombuffer((bundle / "a.f16").read_bytes(), np.int8)
        if s.get("check_input") == "roll_a":            # the wrong-input control (MM P13)
            a8 = np.roll(a8.reshape(M, K), 16, axis=0).reshape(-1)
        models = generic_int_models(a8,
                                    np.frombuffer((bundle / "b.f16").read_bytes(), np.int8), c, M, N, K)
        # the preregistered reading (docs/archive/g17-tensor-int8.md): one saturating MMA is
        # clip(C + the exact 16-term sum) (recon section 136 part 6, 2,400 of 2,400 tiles; a
        # per-product clip was refuted there), so a K chain clips once per issue; without the bit
        # int32 wraps (section 136 part 5.1)
        out = models["clip_issue" if s["saturate"] else "wrap"].copy()
        # a grid split has no tail (MM P13 int8_split): the program emits it only when threadgroups, grid_n and
        # split_k are all 1 (the float path's rule, line "elif (s["threadgroups"] == 1 and s["grid_n"] == 1 ...").
        # Testing threadgroups alone added a +1 the GPU never wrote on a grid_n split: word 0 of tail_grid_n2 (MM 25.145.4)
        if s["threadgroups"] == 1 and s["grid_n"] == 1 and s["split_k"] == 1:
            out[0, 0] = np.int32(((int(out[0, 0]) + 1 - INT32_MIN) % (1 << 32)) + INT32_MIN)   # the tail, wrapping
        return out
    M, N, K = s["M"], s["N"], s["K"]
    araw, braw = (bundle / "a.f16").read_bytes(), (bundle / "b.f16").read_bytes()
    abytes, bbytes = M * K * GENERIC_TYPES[s["a"]], K * N * GENERIC_TYPES[s["b"]]
    check = s.get("check_input")
    flat_a, flat_b = _generic_values(araw[:abytes], s["a"]), _generic_values(braw[:bbytes], s["b"])
    # MM P13 transposed class: the stored A^T is K x M and the stored B^T is N x K ("untransposed", the
    # control, reads the same bytes as the untransposed operands)
    a = flat_a.reshape(K, M).T.copy() if s.get("transA") and check != "untransposed" else flat_a.reshape(M, K)
    b = flat_b.reshape(N, K).T.copy() if s.get("transB") and check != "untransposed" else flat_b.reshape(K, N)
    if s["split_k"] > 1:
        # SPLIT-K PARTIALS (P3): the (split_k*M) x N buffer, block t the single-chain partial over
        # K-slice [t*ks, (t+1)*ks) alone (_gemm_mma, the same order the kernel runs per slice). No
        # SR156 tail (a split grid reads SR156 itself). The ascending-t fp32 fold of these blocks is
        # Piece A's gemm_reference(split_k=G) - checked separately, not here (this validates the partials).
        G = s["split_k"]; ks = K // G
        blocks = [_gemm_mma(a[:, t * ks:(t + 1) * ks], b[t * ks:(t + 1) * ks, :], None, M, N, ks,
                            truncate_a=s["a"] == "float", truncate_b=s["b"] == "float") for t in range(G)]
        return np.concatenate(blocks, axis=0)
    if check == "roll_a":
        a = np.roll(a, 16, axis=0)
    if check == "ldb128":                              # B rows 128 elements apart: the old N cap's reading
        b = np.stack([flat_b[k * 128:k * 128 + N] for k in range(K)])
    if s.get("fp8_nonfinite"):
        return _nonfinite_reference(a, b, s)
    if s.get("check_k"):
        kk = int(s["check_k"])
        out = _gemm_mma(a[:, :kk], b[:kk, :], None, M, N, kk)
        if s["threadgroups"] == 1:
            out[0, 0] = _rne32(out[0, 0] + np.float32(1.0))
        return out
    if s["epilogue"][:1] == ["mx32"]:
        sa = np.frombuffer(araw[abytes:], dtype="<f4").reshape(K // 32, M)
        sb = np.frombuffer(braw[bbytes:], dtype="<f4").reshape(K // 32, N)
        out = mx_post_reference(a, b, sa, sb, shift=s.get("mx_check_shift", 0))
    elif s["epilogue"][:1] == ["mx32e8m0"]:
        model = s.get("mx_code_model") or "ocp"
        sa = e8m0_factors(np.frombuffer(araw[abytes:], np.uint8), model).reshape(K // 32, M)
        sb = e8m0_factors(np.frombuffer(braw[bbytes:], np.uint8), model).reshape(K // 32, N)
        out = mx_post_reference(a, b, sa, sb, shift=s.get("mx_check_shift", 0), check=model not in ("exact", "inf"))
    elif s["split_fp32"]:
        out = _split_fp32_gemm(a, b, s["M"], s["N"], s["K"])
    else:
        out = _gemm_mma(a, b, None, s["M"], s["N"], s["K"],
                        truncate_a=s["a"] == "float", truncate_b=s["b"] == "float")
    if check == "sg_fold4":
        # the old and16 mask of 3 at 8 simdgroups: simdgroups 4..7 recompute 0..3's rows, and their own
        # rows keep the zero seed
        rows = M // s["threadgroups"] // s["simdgroups"]
        out = out.copy()
        for t in range(s["threadgroups"]):
            for g in range(4, s["simdgroups"]):
                out[t * rows * s["simdgroups"] + g * rows:t * rows * s["simdgroups"] + (g + 1) * rows, :] = 0
    for step in s["epilogue"]:
        if step in ("mx32", "mx32e8m0"):
            continue
        if step in FP8_OUT:
            return _fp8_out_buffer(out, s.get("check_as") or step)
        if step == "half":                             # MM P13 narrow_out_half: the last step
            return out if check == "fp32_out" else _half_out_buffer(out, rz=check == "half_rz")
        if step == "relu":
            out = np.maximum(out, np.float32(0.0)).astype("<f4")
        elif step == "gelu":
            out = gelu_model(out)
        elif step == "exp2":
            # the EXACT 2**x rounded once to fp32; the hardware is within one ulp of it both ways
            # (docs/archive/g17-settle-20260923.md), so arms carrying exp2 compare within generic_ulp_bound
            out = np.exp2(out.astype(np.float64)).astype("<f4")
        else:
            scale = np.frombuffer(struct.pack("<I", int(step.split(":")[1], 16)), dtype="<f4")[0]
            out = np.asarray(out * scale, dtype="<f4")
    if s.get("imageblock", "") and s["imageblock"].startswith("x_neighbour"):
        # every lane L stores lane (L+1) mod 32's staged value plus 1
        row = out[0, :32].copy()
        out[0, :32] = [_rne32(v + np.float32(1.0)) for v in np.roll(row, -1)]
    elif s.get("imageblock", "") and s["imageblock"].startswith("neighbour"):
        # lane L < 31 stores its right neighbour's staged value plus 1; C[0,31] is the GEMM value.
        # The same reference for the undeclared control, which is predicted to FAIL (reads 0 -> 1.0)
        row = out[0, :32].copy()
        out[0, :31] = [_rne32(v + np.float32(1.0)) for v in row[1:32]]
    elif s.get("imageblock"):
        # row 0, columns 0..31: each staged through the imageblock and back, plus 1
        out[0, :32] = [_rne32(v + np.float32(1.0)) for v in out[0, :32]]
    elif s["threadgroups"] == 1 and s["grid_n"] == 1:      # a grid split (grid_n) reads SR156 itself: no C[0,0]+=1 tail
        out[0, 0] = np.float32(1234.0) if s.get("guard", "").endswith("_const") else _rne32(out[0, 0] + np.float32(1.0))
    return out


def mx_post_reference(a, b, sa, sb, shift=0, check=True):
    """tlower's mx step, instruction for instruction, each RNE fp32: per 32-element K block b the
    two-issue MMA chain into a zero temporary (_gemm_mma), T = (T * SA[b][row]) * SB[b][col], then
    D = T for the first block and D = D + T after. The ALU flushes fp32 subnormals and this model
    does not, so it REFUSES any operand or result outside the normal range rather than guess at a
    flush: the preregistered data stays inside it. `shift` is the failing control (block b scored
    with block b+shift's factors). check=False drops that refusal for a CONTROL model (255 as +inf, or OCP's exact
    2^-127, which the hardware cannot carry): numpy then keeps subnormals the ALU would flush.
    A NaN factor (code 255) passes the check and makes its block NaN, as OCP MX specifies."""
    M, K = a.shape
    N = b.shape[1]
    def normal(value, approx, what):
        # `value` is the fp32 RNE result; `approx` the same operation in float64, which sees a
        # result fp32 would flush, round to zero or overflow. Either leaves the model, so refuse.
        approx = np.asarray(approx, np.float64)
        if check and (np.any((approx != 0) & (np.abs(approx) < 2.0 ** -126)) or np.any(np.abs(approx) >= 2.0 ** 128)):
            raise ValueError("mx reference: %s leaves the fp32 normal range" % what)
        return np.asarray(value, np.float32)
    f32 = np.float32
    blocks = K // 32
    d = None
    with np.errstate(over="ignore", under="ignore"):
        for blk in range(blocks):
            t = _gemm_mma(a[:, 32 * blk:32 * blk + 32], b[32 * blk:32 * blk + 32, :], None, M, N, 32)
            t = normal(t, t, "block sum")
            f = (blk + shift) % blocks
            t = normal(t * sa[f][:, None].astype(f32), t.astype(np.float64) * sa[f][:, None], "T * SA")
            t = normal(t * sb[f][None, :].astype(f32), t.astype(np.float64) * sb[f][None, :], "T * SA * SB")
            d = t if d is None else normal(d + t, d.astype(np.float64) + t, "D + T")
    return d.astype("<f4")


def mx_scale_table(rng, count, lo=-12, hi=12):
    """E8M0 codes 127+lo .. 127+hi (uniform) and their exact fp32 factors 2^(code-127)."""
    codes = rng.integers(127 + lo, 127 + hi + 1, size=count).astype(np.uint8)
    return codes, np.ldexp(np.float32(1.0), codes.astype(np.int32) - 127).astype("<f4")


E8M0_REFUSED = {0: "E8M0 scale code 0 means 2^-127, an fp32 subnormal: the ALU flushes it (machine model "
                   "0.11), so no fp32 factor carries it and the kernel's e << 23 decode would give +0; refused "
                   "by name (machine model 25.128)"}


def check_e8m0_codes(codes):
    """THE HOST ADMISSION OF E8M0 SCALE CODES (production row P5, machine model 25.128). Codes 1..254
    decode exactly in the kernel and code 255 decodes to NaN, OCP MX's meaning. Code 0 is refused by
    name: 2^-127 cannot be carried by one fp32 factor on this ALU."""
    codes = np.asarray(codes, dtype=np.uint8)
    for code, why in E8M0_REFUSED.items():
        n = int(np.count_nonzero(codes == code))
        if n:
            raise ValueError("refused: %d scale code(s) %d: %s" % (n, code, why))
    return codes


def e8m0_decode_bits(codes):
    """The kernel's decode, bit for bit: (e << 23) + (((e + 1) >> 8) << 22) as uint32."""
    e = np.asarray(codes, dtype=np.uint32)
    return (e << 23) + (((e + 1) >> 8) << 22)


def e8m0_factors(codes, model="ocp"):
    """fp32 factors of E8M0 codes under a READING: "ocp" 2^(e-127) with 255 NaN (codes 1..255; the
    kernel's decode, e8m0_decode_bits); "inf" 255 as +inf (the naive e << 23, a control); "flush"
    0 as +0 (what the kernel's decode gives code 0); "exact" 0 as 2^-127 (OCP, which the ALU cannot
    hold: a control). Code 0 under "ocp" or "inf" is refused, as production refuses it."""
    codes = np.asarray(codes, dtype=np.uint8)
    if model in ("ocp", "inf") and np.any(codes == 0):
        check_e8m0_codes(codes)
    bits = e8m0_decode_bits(codes)
    if model == "inf":
        bits = np.where(codes == 255, np.uint32(0x7F800000), bits)
    out = bits.astype("<u4").view("<f4").copy()
    if model == "exact":
        out[codes == 0] = np.float32(2.0 ** -127)
    return out


# fp8 nonfinite codes: e4m3fn has NaN only (0x7f, 0xff), e5m2 infinity (0x7c, 0xfc) and NaN
FP8_NONFINITE = {"e4m3": (0x7F, 0xFF), "e5m2": (0x7C, 0xFC, 0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF)}


def fp8_nonfinite_plant(raw, rows, cols, fmt, operand):
    """Plant every nonfinite code of `fmt` at fixed positions of a row-major rows x cols fp8 operand.
    A: code i at row 2 + 3 i, column 5 + 7 i (mod cols). B: code i at row 3 + 5 i (mod rows), column
    4 + 3 i. Returns the new bytes and the planted positions."""
    buf = np.frombuffer(bytes(raw), np.uint8).copy().reshape(rows, cols)
    placed = []
    for i, code in enumerate(FP8_NONFINITE[fmt]):
        r, c = ((2 + 3 * i) % rows, (5 + 7 * i) % cols) if operand == "a" else ((3 + 5 * i) % rows, (4 + 3 * i) % cols)
        buf[r, c] = code
        placed.append((int(r), int(c), int(code)))
    return buf.tobytes(), placed


def _nonfinite_values(x, fmt, model):
    """Operand values under a reading of the unpack: ocp keeps NaN and infinity; saturate maps each
    to the largest finite magnitude with its sign (NaN positive); zero maps each to 0."""
    x = np.asarray(x, np.float32).copy()
    bad = ~np.isfinite(x)
    if model == "saturate":
        big = np.float32(448.0 if fmt == "e4m3" else 57344.0)
        x[bad] = np.where(np.signbit(x[bad]) & ~np.isnan(x[bad]), -big, big)
    elif model == "zero":
        x[bad] = np.float32(0.0)
    return x


def _nonfinite_reference(a, b, s):
    """The GEMM of operands carrying planted nonfinite codes. Elements whose row of A and column of
    B are finite are the bit-exact MMA chain (_gemm_mma); an element touching a NaN or infinity is
    IEEE's: NaN if any product is NaN (NaN x anything, infinity x 0) or infinities of both signs
    meet, else the infinity. Summation order cannot change that, so the claim is order-free."""
    M, K = a.shape
    N = b.shape[1]
    model = s.get("nonfinite_model", "ocp")
    a = _nonfinite_values(a, s["a"][3:], model)
    b = _nonfinite_values(b, s["b"][3:], model)
    fa = np.where(np.isfinite(a), a, 0).astype(np.float32)
    fb = np.where(np.isfinite(b), b, 0).astype(np.float32)
    out = _gemm_mma(fa, fb, None, M, N, K)
    with np.errstate(invalid="ignore", over="ignore"):
        for r in range(M):
            for c in range(N):
                p = a[r].astype(np.float64) * b[:, c].astype(np.float64)
                if np.all(np.isfinite(p)):
                    continue
                if np.any(np.isnan(p)) or (np.any(p == np.inf) and np.any(p == -np.inf)):
                    out[r, c] = np.float32(np.nan)
                else:
                    out[r, c] = np.float32(np.inf if np.any(p == np.inf) else -np.inf)
    if s["threadgroups"] == 1:
        out[0, 0] = _rne32(out[0, 0] + np.float32(1.0))
    return out


def generic_nan_class(s):
    """True when the spec's claim is only that an element IS NaN (OCP does not fix a NaN payload):
    planted fp8 NaN/infinity codes and planted scale code 255. Everything else compares bit for bit."""
    return bool(s.get("fp8_nonfinite") or s.get("mx_inject") == "code255")


def fp8_quantize(x, step):
    """op13618 as recon section 138 part 1 measured it over all 2^32 fp32 patterns: ml_dtypes'
    round to nearest even without saturation, except that every NaN INPUT gives the positive
    canonical NaN (0x7f e4m3fn, 0x7e e5m2). An OVERFLOW keeps its sign, as ml_dtypes does: e4m3fn
    gives NaN 0x7f or 0xff, e5m2 infinity 0x7c or 0xfc. A first version also canonicalised the NaN
    an overflow produces, reading section 138's "(0x7f)" as unsigned; on hardware every negative
    e4m3fn overflow came back 0xff (174 of 174 and 297 of 297, results/g17-tensor-lowprec-v1)."""
    import ml_dtypes
    x = np.asarray(x, dtype=np.float32)
    kind = {"fp8e4m3": ml_dtypes.float8_e4m3fn, "fp8e5m2": ml_dtypes.float8_e5m2}[step]
    codes = x.astype(kind).view(np.uint8).copy()
    codes[np.isnan(x)] = {"fp8e4m3": 0x7F, "fp8e5m2": 0x7E}[step]
    return codes


def _fp8_out_buffer(d, step):
    """The C buffer after an fp8 quantize-out, viewed as the worker returns it (fp32 words): the
    first M*N bytes are the codes of D row-major, the rest the zero seed."""
    M, N = d.shape
    buf = np.zeros(M * N * 4, dtype=np.uint8)
    buf[:M * N] = fp8_quantize(d, step).reshape(-1)
    return buf.view("<f4").reshape(M, N)


def _half_out_buffer(d, rz=False):
    """The C buffer after the half narrowing epilogue, viewed as the worker returns it (fp32 words):
    the first M*N halves are D row-major rounded to nearest even (numpy's fp32 -> fp16, subnormals
    and overflow to infinity included), the rest the zero seed. rz=True is the failing control: the
    same narrowing rounded toward zero."""
    M, N = d.shape
    x = np.asarray(d, dtype=np.float32)
    with np.errstate(over="ignore"):
        h = x.astype(np.float16)
    if rz:
        away = np.abs(h.astype(np.float64)) > np.abs(x.astype(np.float64))
        h = np.where(away, np.nextafter(h, np.float16(0.0)), h).astype(np.float16)
    buf = np.zeros(M * N * 4, dtype=np.uint8)
    buf[:M * N * 2] = h.reshape(-1).view(np.uint8)
    return buf.view("<f4").reshape(M, N)


def _split_fp32(x):
    """tlower's emitted split, operation for operation, each RNE fp32: c = 8193 x; d = c + (-x);
    hi = c + (-d); lo = x + (-hi)."""
    f = np.float32
    x = np.asarray(x, np.float32)
    c = (x * f(8193.0)).astype(np.float32)
    d = (c + (x * f(-1.0)).astype(np.float32)).astype(np.float32)
    hi = (c + (d * f(-1.0)).astype(np.float32)).astype(np.float32)
    lo = (x + (hi * f(-1.0)).astype(np.float32)).astype(np.float32)
    return hi, lo


def _split_fp32_gemm(a, b, M, N, K):
    """Per K step, three issues in tlower's order - hi.hi, hi.lo, lo.hi - C first, operands
    truncated; the first issue of the chain is the no-C form."""
    out = np.empty((M, N), dtype="<f4")
    for m in range(M):
        for n in range(N):
            acc = None
            for k0 in range(0, K, 16):
                ah, al = _split_fp32(a[m, k0:k0 + 16]); bh, bl = _split_fp32(b[k0:k0 + 16, n])
                for x, y in ((ah, bh), (ah, bl), (al, bh)):
                    acc = _mma16(x, y, acc, truncate_a=True, truncate_b=True)
            out[m, n] = acc
    return out


def author_generic(bundle: Path, spec):
    """Compile, author and write a generic bundle: generic.json holds the spec it was built from."""
    from agxforge.g17 import scanlink
    if bundle.exists():
        raise ValueError(f"refusing to overwrite an existing bundle: {bundle}")
    s = generic_spec(spec)
    from agxforge.g17 import cc as _cc
    saved = _cc.TENSOR_PIN_16X32X64
    _cc.TENSOR_PIN_16X32X64 = not spec.get("unpin_16x32x64", False)     # a verification bundle only
    try:
        saved_feed = _cc.TENSOR_REGISTER_FEED
        if s["feed"] == "memory":
            _cc.TENSOR_REGISTER_FEED = False
        saved_op1 = (_cc.IB_STORE_OP1, _cc.IB_LOAD_OP1)
        if s["ib_op1"] == "apple":
            _cc.IB_STORE_OP1, _cc.IB_LOAD_OP1 = 0x80000010, 0x2000800000
        elif s["ib_op1"] == "apple_read":              # the read alone (Piece B's recipe)
            _cc.IB_LOAD_OP1 = 0x2000800000
        elif s["ib_op1"]:                              # the store alone, at a stated value (round 9)
            _cc.IB_STORE_OP1 = int(s["ib_op1"].split(":")[1], 16)
        try:
            program = build_generic_program(s)
        finally:
            _cc.IB_STORE_OP1, _cc.IB_LOAD_OP1 = saved_op1
            _cc.TENSOR_REGISTER_FEED = saved_feed
        if s["imageblock"] in ("undeclared", "neighbour_undeclared", "x_neighbour_undeclared"):
            program._abi_imageblock = None          # the control: the linker declares nothing
    finally:
        _cc.TENSOR_PIN_16X32X64 = saved
    image = scanlink.author(program)
    manifest = manifest_for(program, generic=s).model_copy(update={
        "sha256": {"archive": sha(image.archive), "library": sha(image.library),
                   "object": sha(image.object), "code": sha(program.code)},
        "field_ledger": image.field_ledger})
    expected = [(b.index, b.offset, b.written) for b in manifest.abi.bindings]
    if scanlink.verify_contract(image.archive, image.library, expected) != image.object:
        raise ValueError("authored archive does not contain its delivered object")
    bundle.mkdir(parents=True)
    rng = np.random.default_rng(1729)
    (bundle / "generic.json").write_text(json.dumps(s, indent=1) + "\n")
    (bundle / "a.f16").write_bytes(_generic_draw(rng, s["M"] * s["K"], s["a"]))
    b_elems = max([s["K"] * s["N"]] + [st[1] * st[0] for st in s["stages"]])
    width = max([s["N"]] + [st[0] for st in s["stages"]])
    (bundle / "b.f16").write_bytes(_generic_draw(rng, b_elems, s["b"]))
    if s.get("attention"):
        _attention_inputs(bundle, s, spec)
    elif s["stream"]:
        # power-of-two rescalings of the SAME draw (exact in half): Q scaled down so rows stay in the
        # fp32 normal range, the second key block scaled up so its row max usually exceeds the first's
        q = np.frombuffer((bundle / "a.f16").read_bytes(), np.float16) * np.float16(s["stream_q_scale"])
        bb = np.frombuffer((bundle / "b.f16").read_bytes(), np.float16).copy()
        if s["key_blocks"]:
            # block j's keys scaled by k2_scale ** (j mod 3): at two blocks exactly the released draw
            for j in range(s["key_blocks"]):
                kj = slice(keyblock_k(j) // 2, keyblock_k(j) // 2 + 64 * 16)
                bb[kj] = bb[kj] * np.float16(s["stream_k2_scale"] ** (j % 3))
        else:
            k2 = slice(STREAM_B["K2"] // 2, STREAM_B["K2"] // 2 + 64 * 16)
            bb[k2] = bb[k2] * np.float16(s["stream_k2_scale"])
        (bundle / "a.f16").write_bytes(q.astype(np.float16).tobytes())
        (bundle / "b.f16").write_bytes(bb.tobytes())
    if s["fp8_nonfinite"]:
        a_new, a_placed = fp8_nonfinite_plant((bundle / "a.f16").read_bytes(), s["M"], s["K"], s["a"][3:], "a")
        b_new, b_placed = fp8_nonfinite_plant((bundle / "b.f16").read_bytes(), s["K"], s["N"], s["b"][3:], "b")
        (bundle / "a.f16").write_bytes(a_new)
        (bundle / "b.f16").write_bytes(b_new)
        (bundle / "nonfinite.json").write_text(json.dumps({"a": a_placed, "b": b_placed}) + "\n")
    if s["epilogue"][:1] == ["mx32e8m0"]:
        # THE CODE BYTES THEMSELVES (P5): the same draw as the fp32-table form, so the operands match;
        # the kernel decodes them. Planted special codes go in before the host admission, which
        # refuses code 0 unless the bundle is the code-0 research arm.
        sa_codes, _ = mx_scale_table(rng, (s["K"] // 32) * s["M"], *s["mx_range"])
        sb_codes, _ = mx_scale_table(rng, (s["K"] // 32) * s["N"], *s["mx_range"])
        sa_codes, sb_codes = sa_codes.reshape(s["K"] // 32, s["M"]), sb_codes.reshape(s["K"] // 32, s["N"])
        if s["mx_inject"] == "code255":
            sa_codes[0, 5] = 255
            sb_codes[1 % (s["K"] // 32), 20 % s["N"]] = 255
        elif s["mx_inject"] == "code0":
            sa_codes[:, 0:4] = 0
        if s["mx_inject"] != "code0":
            check_e8m0_codes(sa_codes)
            check_e8m0_codes(sb_codes)
        with open(bundle / "a.f16", "ab") as f: f.write(sa_codes.tobytes())
        with open(bundle / "b.f16", "ab") as f: f.write(sb_codes.tobytes())
        (bundle / "mx-codes.bin").write_bytes(sa_codes.tobytes() + sb_codes.tobytes())
    if s["epilogue"][:1] == ["mx32"]:
        # the scale tables, drawn AFTER A and B so the operands match the unscaled bundles'. The E8M0
        # codes are kept beside them (the fp32 table is their exact decoding)
        sa_codes, sa = mx_scale_table(rng, (s["K"] // 32) * s["M"], *s["mx_range"])
        sb_codes, sb = mx_scale_table(rng, (s["K"] // 32) * s["N"], *s["mx_range"])
        with open(bundle / "a.f16", "ab") as f: f.write(sa.tobytes())
        with open(bundle / "b.f16", "ab") as f: f.write(sb.tobytes())
        (bundle / "mx-codes.bin").write_bytes(sa_codes.tobytes() + sb_codes.tobytes())
    if s.get("attention"):
        pass                                           # _attention_inputs wrote c.f32 (sentinel and cache)
    elif s["accumulate"]:
        (bundle / "c.f32").write_bytes(generic_int_seed(rng, s["M"], s["N"]).tobytes())
    else:
        # split-K stacks split_k partials along the row axis, so the C buffer is (split_k*M) x width
        (bundle / "c.f32").write_bytes(np.zeros((s["M"] * s["split_k"], width), dtype="<f4").tobytes())
    (bundle / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    for name, data in (("scan.arc.metallib", image.archive), ("scan.lib.metallib", image.library),
                       ("scan.o", image.object), ("program.bin", program.code)):
        (bundle / name).write_bytes(data)
    return dict(status="prepared", spec=s, files={n: sha((bundle / n).read_bytes()) for n in
                ("manifest.json", "program.bin", "a.f16", "b.f16", "c.f32", "generic.json")})


def build_fp8_program():
    """One 32x32x64 GEMM with fp8 operands (e4m3 A, e5m2 B, one byte each) into fp32, then the
    standard C[0,0] += 1 epilogue. tlower loads each fp8 fragment like int8, unpacks it to bf16
    with four op17642 (fp8enc), and issues the ordinary bf16 op5106."""
    from agxforge.g17 import cc, ir

    a = ir.Buffer("A", 1, elem=ir.F16)       # declared 2-byte, as int8 is: the backend has no 1-byte buffer
    b = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_gemm_fp8_runtime_demo", [a, b, c])
    builder = ir.Builder(fn, fn.block("entry"))
    builder.tensor_matmul(a, b, c, M=32, N=32, K=64, a_dtype="fp8e4m3", b_dtype="fp8e5m2",
                          strideA=64, strideB=32)
    position = builder.builtin("threadgroup_position_in_grid", name="group_x")
    loaded = builder.load(c, position, type=ir.F32, name="c_value")
    one = builder.const(struct.unpack("<I", struct.pack("<f", 1.0))[0])
    builder.store_at(c, position, builder.fadd(loaded, one, name="c_plus_one"))
    builder.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def _fp8_codes(rng, count, fmt):
    """Random FINITE fp8 codes of one format, subnormals included (no NaN, no infinity)."""
    import ml_dtypes
    kind = {"e4m3": ml_dtypes.float8_e4m3fn, "e5m2": ml_dtypes.float8_e5m2}[fmt]
    finite = [code for code in range(256)
              if np.isfinite(np.array([code], dtype=np.uint8).view(kind).astype(np.float32))[0]]
    return rng.choice(np.array(finite, dtype=np.uint8), size=count)


def _fp8_values(codes, fmt):
    """The exact value of each fp8 code (ml_dtypes, an independent decoder)."""
    import ml_dtypes
    kind = {"e4m3": ml_dtypes.float8_e4m3fn, "e5m2": ml_dtypes.float8_e5m2}[fmt]
    return np.asarray(codes, dtype=np.uint8).view(kind).astype(np.float32)


def build_grid_program(threadgroups):
    """One 128x32x64 half/half GEMM split over `threadgroups` threadgroups (tlower's grid split).

    Each threadgroup reads SR_TG_X (SR156) and computes its own 128/G rows; G = 1 is the same
    math in one threadgroup, the control. The launch is G threadgroups of 32 threads."""
    from agxforge.g17 import cc, ir

    a = ir.Buffer("A", 1, elem=ir.F16)
    b = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_gemm_grid_runtime_demo", [a, b, c])
    builder = ir.Builder(fn, fn.block("entry"))
    builder.tensor_matmul(a, b, c, M=128, N=32, K=64, a_dtype="half", b_dtype="half",
                          threadgroups=threadgroups)
    builder.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


EPILOGUE_BIAS_OFFSET = 64 * 32 * 2 - 128        # B's last 128 bytes: rows 62 and 63
EPILOGUE_SCALE_BITS = struct.unpack("<I", struct.pack("<f", 0.75))[0]


def build_chain_program(composition="chain_register"):
    """Two ADJACENT tensor bodies: 32x32x64 half/half, then 32x32x32 float/half reading C as A.

    Nothing sits between them, so the compiler may hand GEMM1's accumulators to GEMM2 in
    registers (``chain_register``). ``chain_memory`` compiles the same IR with the register feed
    switched off - the memory bridge - as the control. Both must equal one reference bit for bit."""
    from agxforge.g17 import cc, ir

    a = ir.Buffer("A", 1, elem=ir.F16)
    b = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function({"chain_register": "tensor_multigemm_register_runtime_demo",
                      "chain_memory": "tensor_multigemm_adjacent_runtime_demo",
                      "epilogue_register": "tensor_multigemm_epilogue_register_runtime_demo",
                      "epilogue_memory": "tensor_multigemm_epilogue_adjacent_runtime_demo"}[composition], [a, b, c])
    builder = ir.Builder(fn, fn.block("entry"))
    # The epilogue arms add B's last 128 bytes as an fp32 per-column bias (write_inputs), then
    # scale by 0.75 and take max with zero, in registers, before the result reaches GEMM2.
    epilogue = ((("bias", b, EPILOGUE_BIAS_OFFSET), ("scale", EPILOGUE_SCALE_BITS), ("relu",))
                if composition.startswith("epilogue") else ())
    builder.tensor_matmul(a, b, c, M=32, N=32, K=64, a_dtype="half", b_dtype="half", epilogue=epilogue)
    builder.tensor_matmul(c, b, c, M=32, N=32, K=32, a_dtype="float", b_dtype="half")
    # The measured runtime class carries the scalar epilogue (SR156); it sits AFTER both bodies,
    # so the two GEMMs stay adjacent.
    position = builder.builtin("threadgroup_position_in_grid", name="group_x")
    loaded = builder.load(c, position, type=ir.F32, name="c_value")
    one = builder.const(struct.unpack("<I", struct.pack("<f", 1.0))[0])
    builder.store_at(c, position, builder.fadd(loaded, one, name="c_plus_one"))
    builder.ret()
    ir.verify(fn)
    saved = cc.TENSOR_REGISTER_FEED
    cc.TENSOR_REGISTER_FEED = composition.endswith("_register")
    try:
        return cc.compile_function(fn)
    finally:
        cc.TENSOR_REGISTER_FEED = saved


def build_weight_offset_program(weight_offset=4096):
    """Compile the measured same-binding two-weight transport arm.

    GEMM1 consumes W0 at the start of binding 2.  GEMM2 consumes a distinct W1 at the
    measured byte offset, with the public binding set and all leading dimensions unchanged.
    The offset is an IR addressing attribute, not a metadata binding offset.
    """
    if weight_offset not in (4096, 4352, 8192, 12288, 16384, 20480):
        raise ValueError("weight-offset arm has no measured B byte offset")
    from agxforge.g17 import cc, ir

    a = ir.Buffer("A", 1, elem=ir.F16)
    b = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_multigemm_weight_offset_runtime_demo", [a, b, c])
    builder = ir.Builder(fn, fn.block("entry"))
    builder.tensor_matmul(a, b, c, M=16, N=32, K=64, a_dtype="half", b_dtype="half")
    # Keep the ordinary scalar/system-register region in the same shape as the measured
    # composed runtime class without changing the C value between the two tensor bodies.
    position = builder.builtin("threadgroup_position_in_grid", name="weight_offset_group")
    builder.add(position, builder.const(0), name="weight_offset_group_keep")
    builder.tensor_matmul(c, b, c, M=16, N=32, K=32, a_dtype="float", b_dtype="half",
                          offsetB=weight_offset)
    builder.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def build_transformer_layer_weight_offset_program():
    """Compile one transformer layer with three measured same-binding B regions."""
    from agxforge.g17 import cc, ir

    a = ir.Buffer("A", 1, elem=ir.F16)
    b = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_transformer_layer_weight_offset_runtime_demo", [a, b, c])
    builder = ir.Builder(fn, fn.block("entry"))
    builder.tensor_matmul(a, b, c, M=16, N=32, K=64, a_dtype="half", b_dtype="half",
                          offsetB=0)
    builder.tensor_matmul(c, b, c, M=16, N=32, K=32, a_dtype="float", b_dtype="half",
                          accumulate=True, offsetB=4096)
    builder.tensor_wide_tile_gelu(c, M=16, N=32, K=64)
    builder.tensor_matmul(c, b, c, M=16, N=32, K=32, a_dtype="float", b_dtype="half",
                          accumulate=True, offsetB=8192)
    builder.tensor_wide_residual_add(c, b, M=16, N=32, K=64)
    builder.tensor_wide_tile_layernorm(c, M=16, N=32, K=64)
    group = builder.builtin("threadgroup_position_in_grid", name="offset_layer_group")
    builder.add(group, builder.const(0), name="offset_layer_group_keep")
    builder.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def build_transformer_continuation_weight_offset_program():
    """Compile the measured FP32-activation continuation layer.

    This is deliberately a separate ordinary image from the initial half-input layer.  The
    activation crosses a dispatch boundary as FP32, while every B read remains in the measured
    local binding-2 regions 0, 4096 and 8192.
    """
    from agxforge.g17 import cc, ir

    a = ir.Buffer("A", 1, elem=ir.F32)
    b = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_transformer_continuation_weight_offset_runtime_demo", [a, b, c])
    builder = ir.Builder(fn, fn.block("entry"))
    builder.tensor_matmul(a, b, c, M=16, N=32, K=32, a_dtype="float", b_dtype="half",
                          offsetB=0)
    builder.tensor_matmul(c, b, c, M=16, N=32, K=32, a_dtype="float", b_dtype="half",
                          accumulate=True, offsetB=4096)
    builder.tensor_wide_tile_gelu(c, M=16, N=32, K=64)
    builder.tensor_matmul(c, b, c, M=16, N=32, K=32, a_dtype="float", b_dtype="half",
                          accumulate=True, offsetB=8192)
    builder.tensor_wide_residual_add(c, b, M=16, N=32, K=64)
    builder.tensor_wide_tile_layernorm(c, M=16, N=32, K=64)
    group = builder.builtin("threadgroup_position_in_grid", name="continuation_group")
    builder.add(group, builder.const(0), name="continuation_group_keep")
    builder.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def build_reduction_program():
    """Compile the first compiler-owned row-softmax reduction through the ordinary IR path.

    A and B remain declared readonly tensor-class inputs so the common worker can enforce the
    existing three-buffer contract.  The reduction consumes the FP32 score buffer C; in the full
    application this buffer is the output of the preceding tensor GEMM dispatch.  Keeping this
    stage as a separate ordinary image also reflects the measured one-SIMDgroup/K=64 domain and
    avoids treating Apple's kernel-dependent reduction routine as a compiler primitive.
    """
    from agxforge.g17 import cc, ir

    a = ir.Buffer("A", 1, elem=ir.F16)
    b = ir.Buffer("B", 2, elem=ir.F16)
    scores = ir.Buffer("scores", 3, elem=ir.F32)
    fn = ir.Function("tensor_row_softmax_runtime_demo", [a, b, scores])
    builder = ir.Builder(fn, fn.block("entry"))
    # The score tile is produced by the ordinary tensor GEMM route in this same program.  The
    # following scalar region consumes its memory-mediated FP32 stores; no Apple reduction API is
    # present in the IR.
    builder.tensor_matmul(a, b, scores, M=32, N=16, K=64, a_dtype="half", b_dtype="half")
    group = builder.builtin("threadgroup_position_in_grid", name="reduction_group")
    builder.add(group, builder.const(0), name="reduction_group_keep")
    builder.tensor_row_softmax(scores, row=0, M=32, N=16, K=64)
    builder.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def build_ffn_program():
    """Compile one measured FFN-style row through ordinary IR.

    GEMM1 produces a 16x16 FP32 activation tile.  Row zero receives the compiler-owned GELU
    approximation, GEMM2 accumulates into that transformed C buffer (the residual connection),
    and the same row is normalized by the explicit compiler-owned reduction order.  The other
    rows remain a deliberate named scope boundary for this first application slice.
    """
    from agxforge.g17 import cc, ir

    a = ir.Buffer("A", 1, elem=ir.F16)
    b = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_ffn_gelu_layernorm_runtime_demo", [a, b, c])
    builder = ir.Builder(fn, fn.block("entry"))
    builder.tensor_matmul(a, b, c, M=16, N=16, K=64, a_dtype="half", b_dtype="half")
    group = builder.builtin("threadgroup_position_in_grid", name="ffn_group")
    builder.add(group, builder.const(0), name="ffn_group_keep")
    builder.tensor_row_gelu(c, row=0, M=16, N=16, K=64)
    # Accumulate into the transformed C buffer: this is the residual path, not a host-side
    # shortcut. The compiler composition route reserves both bodies and emits one END.
    builder.tensor_matmul(c, b, c, M=16, N=16, K=16, a_dtype="float", b_dtype="half",
                          accumulate=True)
    builder.tensor_row_layernorm(c, row=0, M=16, N=16, K=64)
    builder.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def build_ffn_allrows_program():
    """Compile the all-row 16x16 FFN slice through the ordinary compiler path."""
    from agxforge.g17 import cc, ir

    a = ir.Buffer("A", 1, elem=ir.F16)
    b = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_ffn_gelu_layernorm_allrows_runtime_demo", [a, b, c])
    builder = ir.Builder(fn, fn.block("entry"))
    builder.tensor_matmul(a, b, c, M=16, N=16, K=64, a_dtype="half", b_dtype="half")
    group = builder.builtin("threadgroup_position_in_grid", name="ffn_allrows_group")
    builder.add(group, builder.const(0), name="ffn_allrows_group_keep")
    builder.tensor_tile_gelu(c, M=16, N=16, K=64)
    builder.tensor_matmul(c, b, c, M=16, N=16, K=16, a_dtype="float", b_dtype="half",
                          accumulate=True)
    builder.tensor_tile_layernorm(c, M=16, N=16, K=64)
    builder.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def build_ffn_wide_program():
    """Compile a two-tile 16x32 FFN graph with explicit per-row reductions."""
    from agxforge.g17 import cc, ir

    a = ir.Buffer("A", 1, elem=ir.F16)
    b = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_ffn_gelu_layernorm_wide_runtime_demo", [a, b, c])
    builder = ir.Builder(fn, fn.block("entry"))
    builder.tensor_matmul(a, b, c, M=16, N=32, K=64, a_dtype="half", b_dtype="half")
    group = builder.builtin("threadgroup_position_in_grid", name="ffn_wide_group")
    builder.add(group, builder.const(0), name="ffn_wide_group_keep")
    builder.tensor_wide_tile_gelu(c, M=16, N=32, K=64)
    builder.tensor_matmul(c, b, c, M=16, N=32, K=32, a_dtype="float", b_dtype="half",
                          accumulate=True)
    builder.tensor_wide_tile_layernorm(c, M=16, N=32, K=64)
    builder.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def build_transformer_layer_program():
    """Compile the smallest ordinary-compiler transformer-layer slice.

    The first tensor body is the attention-output producer, the second is the FFN expansion,
    GELU is compiler-owned scalar/vector code, and the third body is the FFN contraction. The
    residual is a full 16x32 half tile from the readonly B buffer, widened and added on device;
    LayerNorm then consumes the resulting FP32 tile.
    """
    from agxforge.g17 import cc, ir

    a = ir.Buffer("A", 1, elem=ir.F16)
    b = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_transformer_layer_runtime_demo", [a, b, c])
    builder = ir.Builder(fn, fn.block("entry"))
    builder.tensor_matmul(a, b, c, M=16, N=32, K=64, a_dtype="half", b_dtype="half")
    builder.tensor_matmul(c, b, c, M=16, N=32, K=32, a_dtype="float", b_dtype="half",
                          accumulate=True)
    group = builder.builtin("threadgroup_position_in_grid", name="transformer_group")
    builder.add(group, builder.const(0), name="transformer_group_keep")
    builder.tensor_wide_tile_gelu(c, M=16, N=32, K=64)
    builder.tensor_matmul(c, b, c, M=16, N=32, K=32, a_dtype="float", b_dtype="half",
                          accumulate=True)
    builder.tensor_wide_residual_add(c, b, M=16, N=32, K=64)
    builder.tensor_wide_tile_layernorm(c, M=16, N=32, K=64)
    builder.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def build_two_transformer_layers_program():
    """Compile two complete transformer-layer slices with one GPU-resident C buffer.

    Layer 1 consumes the LayerNorm output of layer 0 directly from C.  Both layers use the same
    measured 16x32 tile shape and readonly B residual/weight buffer; the second layer's first GEMM
    is FP32-A, while all subsequent regions retain the measured float/half class.
    """
    from agxforge.g17 import cc, ir

    a = ir.Buffer("A", 1, elem=ir.F16)
    b = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_transformer_two_layer_runtime_demo", [a, b, c])
    builder = ir.Builder(fn, fn.block("entry"))

    # Layer 0: attention-output producer -> FFN expansion -> GELU -> contraction -> residual -> LN.
    builder.tensor_matmul(a, b, c, M=16, N=32, K=64, a_dtype="half", b_dtype="half")
    builder.tensor_matmul(c, b, c, M=16, N=32, K=32, a_dtype="float", b_dtype="half",
                          accumulate=True)
    builder.tensor_wide_tile_gelu(c, M=16, N=32, K=64)
    builder.tensor_matmul(c, b, c, M=16, N=32, K=32, a_dtype="float", b_dtype="half",
                          accumulate=True)
    builder.tensor_wide_residual_add(c, b, M=16, N=32, K=64)
    builder.tensor_wide_tile_layernorm(c, M=16, N=32, K=64)

    # Layer 1 consumes C without a host readback. Its first GEMM starts a fresh accumulator; the
    # following expansion and contraction use the same measured accumulating class.
    builder.tensor_matmul(c, b, c, M=16, N=32, K=32, a_dtype="float", b_dtype="half")
    builder.tensor_matmul(c, b, c, M=16, N=32, K=32, a_dtype="float", b_dtype="half",
                          accumulate=True)
    builder.tensor_wide_tile_gelu(c, M=16, N=32, K=64)
    builder.tensor_matmul(c, b, c, M=16, N=32, K=32, a_dtype="float", b_dtype="half",
                          accumulate=True)
    builder.tensor_wide_residual_add(c, b, M=16, N=32, K=64)
    builder.tensor_wide_tile_layernorm(c, M=16, N=32, K=64)
    group = builder.builtin("threadgroup_position_in_grid", name="two_layer_group")
    builder.add(group, builder.const(0), name="two_layer_group_keep")
    builder.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def manifest_for(program, weight_offset_b=None, weight_offsets_b=None, grid_threadgroups=None, generic=None):
    from agxforge.g17 import runtime

    abi = program.abi_plain(program.abi())
    contract = program.contract()
    if program.name == "tensor_row_softmax_runtime_demo":
        tensor = dict(M=32, N=16, K=64, lda=64, ldb=16, ldc=16,
                      a_type="half", b_type="half", c_type="float", simdgroups=1,
                      grid=[32, 1, 1], threadgroup=[32, 1, 1],
                      composition="row_softmax_fp32")
        shape = {"rows": 32, "columns": 16}
    elif program.name == "tensor_transformer_two_layer_runtime_demo":
        tensor = dict(M=16, N=32, K=64, K2=32, K3=32, lda=64, ldb=32, ldc=32,
                      a_type="half", b_type="half", a2_type="float", b2_type="half",
                      c_type="float", simdgroups=1, grid=[32, 1, 1], threadgroup=[32, 1, 1],
                      composition="transformer_two_layer")
        shape = {"rows": 16, "columns": 32}
    elif program.name in ("tensor_transformer_layer_runtime_demo",
                          "tensor_transformer_layer_weight_offset_runtime_demo"):
        tensor = dict(M=16, N=32, K=64, K2=32, K3=32, lda=64, ldb=32, ldc=32,
                      a_type="half", b_type="half", a2_type="float", b2_type="half",
                      c_type="float", simdgroups=1, grid=[32, 1, 1], threadgroup=[32, 1, 1],
                      composition=("transformer_layer_weight_offset"
                                   if program.name.endswith("_weight_offset_runtime_demo")
                                   else "transformer_layer"))
        if program.name.endswith("_weight_offset_runtime_demo"):
            tensor["weight_offsets_b"] = (0, 4096, 8192)
        shape = {"rows": 16, "columns": 32}
    elif program.name == "tensor_transformer_continuation_weight_offset_runtime_demo":
        tensor = dict(M=16, N=32, K=32, K2=32, K3=32, lda=32, ldb=32, ldc=32,
                      a_type="float", b_type="half", a2_type="float", b2_type="half",
                      c_type="float", simdgroups=1, grid=[32, 1, 1], threadgroup=[32, 1, 1],
                      composition="transformer_continuation_weight_offset",
                      weight_offsets_b=(0, 4096, 8192))
        shape = {"rows": 16, "columns": 32}
    elif program.name in ("tensor_ffn_gelu_layernorm_runtime_demo",
                          "tensor_ffn_gelu_layernorm_allrows_runtime_demo",
                          "tensor_ffn_gelu_layernorm_wide_runtime_demo"):
        wide = program.name.endswith("_wide_runtime_demo")
        tensor = dict(M=16, N=32 if wide else 16, K=64, K2=32 if wide else 16,
                      lda=64, ldb=32 if wide else 16, ldc=32 if wide else 16,
                      a_type="half", b_type="half", a2_type="float", b2_type="half",
                      c_type="float", simdgroups=1, grid=[32, 1, 1], threadgroup=[32, 1, 1],
                      composition=("ffn_gelu_layernorm_wide" if wide else
                                   "ffn_gelu_layernorm_allrows" if
                                   program.name.endswith("_allrows_runtime_demo") else
                                   "ffn_gelu_layernorm"))
        shape = {"rows": 16, "columns": 32 if wide else 16}
    elif program.name == GENERIC_NAME:
        s = generic
        width = max([s["N"]] + [st[0] for st in s["stages"]])         # the C buffer's width
        # P3 split-K: the launch has split_k more threadgroups (each writes an M x N partial), and the C
        # buffer stacks them: (split_k*M) x N. The reduce folds them; the partials are what this dispatch
        # writes and what generic_reference returns.
        tensor = dict(M=s["M"], N=width, K=s["K"], lda=s["K"], ldb=s["N"], ldc=width,
                      a_type=s["a"], b_type=s["b"], c_type="int" if s["a"] == "int8" else "float",
                      simdgroups=s["simdgroups"],
                      grid=[32 * s["simdgroups"] * s["threadgroups"] * s["grid_n"] * s["split_k"], 1, 1],
                      threadgroup=[32 * s["simdgroups"], 1, 1], grid_n=s["grid_n"],
                      split_k=s["split_k"],
                      composition="attention" if s.get("attention") else "gemm_generic",
                      epilogue=tuple(s["epilogue"]) or None,
                      stages=tuple(tuple(st) for st in s["stages"]) or None,
                      accumulate=True if s["accumulate"] else None, saturate=True if s["saturate"] else None,
                      transA=True if s.get("transA") else None, transB=True if s.get("transB") else None)
        shape = {"rows": s["M"] * s["split_k"], "columns": width}
    elif program.name == "tensor_gemm_fp8_runtime_demo":
        tensor = dict(M=32, N=32, K=64, lda=64, ldb=32, ldc=32,
                      a_type="fp8e4m3", b_type="fp8e5m2", c_type="float", simdgroups=1,
                      grid=[32, 1, 1], threadgroup=[32, 1, 1], composition="gemm_fp8")
        shape = {"rows": 32, "columns": 32}
    elif program.name == "tensor_gemm_grid_runtime_demo":
        if grid_threadgroups not in (1, 2, 4, 8):
            raise ValueError("the grid class launches 1, 2, 4 or 8 threadgroups")
        tensor = dict(M=128, N=32, K=64, lda=64, ldb=32, ldc=32,
                      a_type="half", b_type="half", c_type="float", simdgroups=1,
                      grid=[32 * grid_threadgroups, 1, 1], threadgroup=[32, 1, 1],
                      composition="gemm_grid")
        shape = {"rows": 128, "columns": 32}
    elif program.name in ("tensor_multigemm_register_runtime_demo", "tensor_multigemm_adjacent_runtime_demo",
                          "tensor_multigemm_epilogue_register_runtime_demo", "tensor_multigemm_epilogue_adjacent_runtime_demo"):
        tensor = dict(M=32, N=32, K=64, K2=32, lda=64, ldb=32, ldc=32,
                      a_type="half", b_type="half", a2_type="float", b2_type="half",
                      c_type="float", simdgroups=1, grid=[32, 1, 1], threadgroup=[32, 1, 1],
                      composition={"tensor_multigemm_register_runtime_demo": "gemm_gemm_register",
                                   "tensor_multigemm_adjacent_runtime_demo": "gemm_gemm_memory",
                                   "tensor_multigemm_epilogue_register_runtime_demo": "gemm_epilogue_gemm_register",
                                   "tensor_multigemm_epilogue_adjacent_runtime_demo": "gemm_epilogue_gemm_memory"}[program.name])
        shape = {"rows": 32, "columns": 32}
    elif program.name in ("tensor_multigemm_runtime_demo", "tensor_multigemm_fadd_fmul_runtime_demo",
                        "tensor_multigemm_fadd_fmul_gemm_runtime_demo",
                        "tensor_multigemm_relu_vec_residual_runtime_demo"):
        composition = ("gemm_fadd_fmul_gemm_memory" if
                       program.name == "tensor_multigemm_fadd_fmul_runtime_demo" else
                       "gemm_fadd_fmul_gemm_fadd_gemm_memory" if
                       program.name == "tensor_multigemm_fadd_fmul_gemm_runtime_demo" else
                       "gemm_relu_vec_gemm_residual_memory" if
                       program.name == "tensor_multigemm_relu_vec_residual_runtime_demo" else
                       "gemm_fadd_gemm_memory")
        tensor = dict(M=32, N=32, K=64, K2=32, lda=64, ldb=32, ldc=32,
                      a_type="half", b_type="half", a2_type="float", b2_type="half",
                      c_type="float", simdgroups=1, grid=[32, 1, 1], threadgroup=[32, 1, 1],
                      composition=composition)
        if program.name in ("tensor_multigemm_fadd_fmul_gemm_runtime_demo",
                            "tensor_multigemm_relu_vec_residual_runtime_demo"):
            tensor["K3"] = 32
        shape = {"rows": 32, "columns": 32}
    elif program.name == "tensor_multigemm_weight_offset_runtime_demo":
        if weight_offset_b not in (4096, 4352, 8192, 12288, 16384, 20480):
            raise ValueError("weight-offset arm has no measured B byte offset")
        tensor = dict(M=16, N=32, K=64, K2=32, lda=64, ldb=32, ldc=32,
                      a_type="half", b_type="half", a2_type="float", b2_type="half",
                      c_type="float", simdgroups=1, grid=[32, 1, 1], threadgroup=[32, 1, 1],
                      composition="gemm_weight_offset", weight_offset_b=weight_offset_b)
        shape = {"rows": 16, "columns": 32}
    else:
        tensor = dict(M=17, N=19, K=16, lda=16, ldb=19, ldc=19,
                      a_type="half", b_type="half", c_type="float", simdgroups=1,
                      grid=[32, 1, 1], threadgroup=[32, 1, 1],
                      composition="gemm_fadd_threadgroup_position")
        shape = {"rows": 17, "columns": 19}
    tensor["grid"] = tuple(tensor["grid"])
    tensor["threadgroup"] = tuple(tensor["threadgroup"])
    return runtime.ImageContract(
        format=runtime.manifest_format(runtime.TENSOR_KIND), kind=runtime.TENSOR_KIND,
        name=contract.name, shape=runtime.Shape(**shape), tensor=runtime.TensorSpec(**tensor),
        abi=runtime.compiler_abi_from_plain(abi), code_size=len(program.code),
        instructions=tuple(runtime.Instruction(offset=i.offset, length=i.length, opcode=i.opcode)
                           for i in contract.instructions),
        sha256={"archive": "0" * 64, "library": "0" * 64,
                "object": "0" * 64, "code": sha(program.code)},
        field_ledger={"metadata": "compiler and measured tensor class"})


def author(bundle: Path, composition="single", weight_offset=4096):
    """Compile and author a fresh bundle without importing a vendor library."""
    from agxforge.g17 import scanlink

    if bundle.exists():
        raise ValueError(f"refusing to overwrite an existing bundle: {bundle}")
    if composition not in ("single", "multigemm", "multigemm_fadd_fmul", "multigemm3", "chain_register", "chain_memory", "epilogue_register", "epilogue_memory", "grid1", "grid2", "grid4", "grid8", "fp8",
                           "multigemm_relu_vec", "reduction_softmax", "ffn_gelu_layernorm",
                           "ffn_gelu_layernorm_allrows", "ffn_gelu_layernorm_wide",
                           "transformer_layer", "transformer_two_layer",
                           "transformer_layer_weight_offset", "transformer_continuation_weight_offset",
                           "weight_offset"):
        raise ValueError("unknown tensor composition %r" % composition)
    if composition == "single":
        program = build_program()
    elif composition == "reduction_softmax":
        program = build_reduction_program()
    elif composition == "transformer_two_layer":
        program = build_two_transformer_layers_program()
    elif composition == "transformer_layer":
        program = build_transformer_layer_program()
    elif composition == "transformer_layer_weight_offset":
        program = build_transformer_layer_weight_offset_program()
    elif composition == "transformer_continuation_weight_offset":
        program = build_transformer_continuation_weight_offset_program()
    elif composition == "ffn_gelu_layernorm_wide":
        program = build_ffn_wide_program()
    elif composition == "ffn_gelu_layernorm_allrows":
        program = build_ffn_allrows_program()
    elif composition == "ffn_gelu_layernorm":
        program = build_ffn_program()
    elif composition in ("multigemm_fadd_fmul", "multigemm3", "multigemm_relu_vec"):
        program = build_multigemm_program(composition)
    elif composition == "weight_offset":
        program = build_weight_offset_program(weight_offset)
    elif composition == "fp8":
        program = build_fp8_program()
    elif composition in ("grid1", "grid2", "grid4", "grid8"):
        program = build_grid_program(int(composition[4:]))
    elif composition in ("chain_register", "chain_memory", "epilogue_register", "epilogue_memory", "grid1", "grid2", "grid4", "grid8", "fp8"):
        program = build_chain_program(composition)
    else:
        program = build_multigemm_program()
    image = scanlink.author(program)
    manifest = manifest_for(program, grid_threadgroups=(int(composition[4:]) if composition.startswith("grid") else None),
                            weight_offset_b=weight_offset if composition == "weight_offset" else None,
                            weight_offsets_b=((0, 4096, 8192)
                                              if composition in ("transformer_layer_weight_offset",
                                                                  "transformer_continuation_weight_offset")
                                              else None)).model_copy(update={
        "sha256": {"archive": sha(image.archive), "library": sha(image.library),
                   "object": sha(image.object), "code": sha(program.code)},
        "field_ledger": image.field_ledger})
    expected = [(b.index, b.offset, b.written) for b in manifest.abi.bindings]
    delivered = scanlink.verify_contract(image.archive, image.library, expected)
    if delivered != image.object:
        raise ValueError("authored archive does not contain its delivered object")
    bundle.mkdir(parents=True)
    try:
        (bundle / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        for name, data in (("scan.arc.metallib", image.archive),
                           ("scan.lib.metallib", image.library),
                           ("scan.o", image.object), ("program.bin", program.code)):
            (bundle / name).write_bytes(data)
        write_inputs(bundle, composition=composition, weight_offset=weight_offset)
    except BaseException:
        import shutil
        shutil.rmtree(bundle)
        raise
    input_names = ("a.f32", "b.f16", "c.f32") if composition == "transformer_continuation_weight_offset" else ("a.f16", "b.f16", "c.f32")
    return dict(status="prepared", manifest=manifest.model_dump(mode="json"),
                files={name: sha((bundle / name).read_bytes())
                       for name in ("manifest.json", "scan.arc.metallib", "scan.lib.metallib",
                                    "scan.o", "program.bin", *input_names)})


def author_transformer_sequence(bundle: Path, length=2):
    """Author measured initial/continuation images for a GPU-resident layer sequence."""
    if length not in (2, 6):
        raise ValueError("the measured sequence campaign supports exactly two or six layers")
    if bundle.exists():
        raise ValueError(f"refusing to overwrite an existing bundle: {bundle}")
    from agxforge.g17 import runtime
    with tempfile.TemporaryDirectory(prefix="g17-sequence-author-") as td:
        td = Path(td)
        initial, continuation = td / "initial", td / "continuation"
        author(initial, composition="transformer_layer_weight_offset")
        author(continuation, composition="transformer_continuation_weight_offset")
        initial_manifest = json.loads((initial / "manifest.json").read_text())
        continuation_manifest = json.loads((continuation / "manifest.json").read_text())
        runtime.ImageContract.read(initial_manifest)
        runtime.ImageContract.read(continuation_manifest)
        bundle.mkdir(parents=True)
        for name, source in (("initial", initial), ("continuation", continuation)):
            target = bundle / name
            target.mkdir()
            for item in source.iterdir():
                if item.name in ("manifest.json", "scan.arc.metallib", "scan.lib.metallib",
                                 "scan.o", "program.bin"):
                    shutil.copy2(item, target / item.name)
        rng = np.random.default_rng(1731)
        a = rng.uniform(-1.0, 1.0, size=(16, 64)).astype("<f2")
        (bundle / "a.f16").write_bytes(a.tobytes(order="C"))
        for index in range(length):
            weights = rng.uniform(-1.0, 1.0, size=(5120,)).astype("<f2")
            (bundle / f"b{index}.f16").write_bytes(weights.tobytes(order="C"))
        weight_hashes = [sha((bundle / f"b{index}.f16").read_bytes()) for index in range(length)]
        sequence = {
            "format": "g17-common-sequence-v1", "kind": "tensor_gemm_sequence",
            "name": "tensor_transformer_layer_sequence_runtime_demo", "layers": length,
            "activation": {"rows": 16, "columns": 32, "storage": "float32",
                           "transport": "gpu_resident_ping_pong", "host_readback": False},
            "weight_slot": 2, "weight_regions": [0, 4096, 8192],
            "initial": initial_manifest, "continuation": continuation_manifest,
            "initial_manifest_sha256": sha((bundle / "initial/manifest.json").read_bytes()),
            "continuation_manifest_sha256": sha((bundle / "continuation/manifest.json").read_bytes()),
            "distinct_weight_buffers": length, "weight_hashes": weight_hashes,
            "source": source_identity(),
        }
        (bundle / "manifest.json").write_text(json.dumps(sequence, indent=2) + "\n")
    files = ["manifest.json", "a.f16"] + [f"b{i}.f16" for i in range(length)]
    for sub in ("initial", "continuation"):
        files.extend(f"{sub}/{name}" for name in ("manifest.json", "scan.arc.metallib",
                                                   "scan.lib.metallib", "scan.o", "program.bin"))
    return {"status": "prepared", "manifest": sequence,
            "files": {name: sha((bundle / name).read_bytes()) for name in files}}


def _sequence_reference(a, weights):
    state = _transformer_layer_weight_offset_reference(a, weights[0])
    for b in weights[1:]:
        state = _transformer_layer_weight_offset_reference(state, b)
    return state


def run_transformer_sequence(bundle: Path, *, queries=3, worker=None, receipt=None):
    """Run all layer images in one command buffer without host intermediate readback."""
    from agxforge.g17 import runtime
    import g17packeddispatch
    import g17commonstage
    bundle = Path(bundle).resolve()
    sequence = json.loads((bundle / "manifest.json").read_text())
    if sequence.get("kind") != "tensor_gemm_sequence" or sequence.get("layers") not in (2, 6):
        raise ValueError("sequence manifest is outside the measured two/six-layer class")
    length = int(sequence["layers"])
    initial = runtime.ImageContract.read(sequence["initial"])
    continuation = runtime.ImageContract.read(sequence["continuation"])
    if sha((bundle / "initial/manifest.json").read_bytes()) != sequence.get("initial_manifest_sha256") or \
            sha((bundle / "continuation/manifest.json").read_bytes()) != sequence.get("continuation_manifest_sha256"):
        raise ValueError("sequence nested image manifest identity differs from its sequence record")
    if initial.tensor.composition != "transformer_layer_weight_offset" or continuation.tensor.composition != "transformer_continuation_weight_offset":
        raise ValueError("sequence images are outside the measured three-region classes")
    if sequence.get("weight_regions") != [0, 4096, 8192] or sequence.get("weight_slot") != 2:
        raise ValueError("sequence weight transport is outside the measured local class")
    if sequence.get("activation", {}).get("host_readback") is not False:
        raise ValueError("sequence activation transport must remain GPU resident")
    a = np.fromfile(bundle / "a.f16", dtype="<f2").reshape(16, 64)
    weights = [np.fromfile(bundle / f"b{i}.f16", dtype="<f2") for i in range(length)]
    weight_hashes = [sha((bundle / f"b{i}.f16").read_bytes()) for i in range(length)]
    if sequence.get("weight_hashes") != weight_hashes or len(set(weight_hashes)) != length:
        raise ValueError("sequence weight buffers are not distinct or do not match their record")
    expected = _sequence_reference(a, weights)
    reference_bound = 2.0e-5 if length == 2 else 2.0e-3
    if worker is None:
        with tempfile.TemporaryDirectory(prefix="g17-sequence-worker-") as td:
            return run_transformer_sequence(bundle, queries=queries, worker=Path(td) / "common-worker", receipt=receipt)
    worker = Path(worker)
    build = build_worker(worker)
    with g17commonstage.lock_gpu():
        before = g17packeddispatch.gpu_events()
        load = subprocess.run([str(worker), str(bundle), "--tensor-sequence-load-approved"], capture_output=True, timeout=30)
        if load.returncode:
            raise RuntimeError("tensor sequence load-only failed: " + load.stderr.decode(errors="replace")[-4000:])
        if json.loads(load.stdout) != {"status": 0, "load_only": True, "gpu_dispatched": False}:
            raise ValueError("unexpected sequence load-only response")
        if before != g17packeddispatch.gpu_events():
            raise RuntimeError("GPU diagnostics changed during sequence load-only")
        dispatch = subprocess.run([str(worker), str(bundle), "sequence-inputs", str(queries), "--tensor-sequence-dispatch-approved"],
                                  capture_output=True, timeout=90)
        if dispatch.returncode:
            raise RuntimeError("tensor sequence dispatch failed: " + dispatch.stderr.decode(errors="replace")[-4000:])
        frames = _frames(dispatch.stdout)
        if before != g17packeddispatch.gpu_events():
            raise RuntimeError("GPU diagnostics changed during tensor sequence dispatch")
    if len(frames) != queries + 1 or frames[0][0].get("sequence") != 0:
        raise ValueError("tensor sequence handshake or frame count differs from the contract")
    attempts = []
    for sequence_no, (header, payload) in enumerate(frames[1:], 1):
        if (header.get("sequence"), header.get("status"), header.get("gpu_dispatched"),
            header.get("boundary_guard"), header.get("readonly_inputs"), header.get("layer_count")) != (sequence_no, 0, True, True, True, length):
            raise ValueError(f"tensor sequence response {sequence_no} is not fully checked")
        got = np.frombuffer(payload, dtype="<f4").copy().reshape(16, 32)
        max_abs = float(np.max(np.abs(got - expected)))
        if not np.allclose(got, expected, rtol=0.0, atol=reference_bound, equal_nan=True):
            raise RuntimeError(f"tensor sequence query {sequence_no} differs from reference (max_abs={max_abs})")
        attempts.append({"sequence": sequence_no, "layers": length, "status": "passed",
                         "output_sha256": sha(payload), "reference_max_abs": max_abs,
                         "reference_bound": reference_bound, "reference_within_bound": True,
                         "boundary_guard": True, "readonly_inputs": True,
                         "gpu_intermediate_readback": False})
    report = {"status": "passed", "kind": "tensor_gemm_sequence", "name": sequence["name"],
              "layers": length, "queries": attempts, "load_only": True, "worker": build,
              "gpu_dispatched": True, "gpu_intermediate_readback": False,
              "distinct_weight_buffers": length, "weight_slot": 2, "weight_regions": [0, 4096, 8192],
              "weight_hashes": weight_hashes,
              "source": source_identity(),
              "reference": "instruction-level RNE32 MMA with C-first accumulation (section 136), measured FP32-A truncation, three local B regions per layer, GPU-resident ping-pong activations"}
    if receipt is not None:
        Path(receipt).write_text(json.dumps(report, indent=2) + "\n")
    return report


def write_inputs(bundle: Path, composition="single", weight_offset=4096):
    """Write deterministic, finite operands and a zero initial output."""
    rng = np.random.default_rng(1729)
    if composition in ("reduction_softmax", "ffn_gelu_layernorm", "ffn_gelu_layernorm_allrows",
                       "ffn_gelu_layernorm_wide", "transformer_layer", "transformer_two_layer"):
        rows = 32 if composition == "reduction_softmax" else 16
        cols = 32 if composition in ("ffn_gelu_layernorm_wide", "transformer_layer",
                                     "transformer_two_layer") else 16
        a = rng.uniform(-1.0, 1.0, size=(rows, 64)).astype("<f2")
        b = rng.uniform(-1.0, 1.0, size=(64, cols)).astype("<f2")
        c = rng.uniform(-3.0, 3.0, size=(rows, cols)).astype("<f4")
    elif composition == "transformer_layer_weight_offset":
        a = rng.uniform(-1.0, 1.0, size=(16, 64)).astype("<f2")
        b = rng.uniform(-1.0, 1.0, size=(5120,)).astype("<f2")
        c = rng.uniform(-3.0, 3.0, size=(16, 32)).astype("<f4")
    elif composition == "transformer_continuation_weight_offset":
        a = rng.uniform(-3.0, 3.0, size=(16, 32)).astype("<f4")
        b = rng.uniform(-1.0, 1.0, size=(5120,)).astype("<f2")
        c = rng.uniform(-3.0, 3.0, size=(16, 32)).astype("<f4")
    elif composition == "weight_offset":
        if weight_offset not in (4096, 4352, 8192, 12288, 16384, 20480):
            raise ValueError("weight-offset arm has no measured B byte offset")
        a = rng.uniform(-2.0, 2.0, size=(16, 64)).astype("<f2")
        w0 = rng.uniform(-2.0, 2.0, size=(64, 32)).astype("<f2")
        w1 = rng.uniform(-2.0, 2.0, size=(32, 32)).astype("<f2")
        flat = np.zeros(weight_offset // 2 + w1.size, dtype="<f2")
        flat[:w0.size] = w0.reshape(-1)
        flat[weight_offset // 2:weight_offset // 2 + w1.size] = w1.reshape(-1)
        b = flat
        c = np.zeros((16, 32), dtype="<f4")
    elif composition == "fp8":
        # raw fp8 codes, one byte each; the file names stay a.f16/b.f16 because the worker reads those
        a = _fp8_codes(rng, 32 * 64, "e4m3").reshape(32, 64)
        b = _fp8_codes(rng, 64 * 32, "e5m2").reshape(64, 32)
        c = np.zeros((32, 32), dtype="<f4")
    elif composition in ("grid1", "grid2", "grid4", "grid8"):
        a = rng.uniform(-2.0, 2.0, size=(128, 64)).astype("<f2")
        b = rng.uniform(-2.0, 2.0, size=(64, 32)).astype("<f2")
        c = np.zeros((128, 32), dtype="<f4")
    elif composition in ("multigemm", "multigemm_fadd_fmul", "multigemm3", "multigemm_relu_vec", "chain_register", "chain_memory", "epilogue_register", "epilogue_memory", "grid1", "grid2", "grid4", "grid8", "fp8"):
        # B is large enough for both measured regions: GEMM1 consumes rows 0..18 and GEMM2
        # consumes rows 0..31. C is the FP32 intermediate/output buffer.
        # Fractional half values are intentional: small integers can leave the lower fp32-A
        # mantissa bits zero and make the measured truncation rule unobservable.
        a = rng.uniform(-2.0, 2.0, size=(32, 64)).astype("<f2")
        b = rng.uniform(-2.0, 2.0, size=(64, 32)).astype("<f2")
        c = np.zeros((32, 32), dtype="<f4")
        if composition.startswith("epilogue"):
            # 32 fp32 biases k/64, |k| < 128: at most seven significant bits, so each one's low
            # halfword is zero and its high halfword is a finite normal half. Read as B's rows 62
            # and 63 they are ordinary GEMM1 weights, and the reference reads the same bytes both ways.
            bias = (np.round(rng.uniform(-2.0, 2.0, size=32) * 64) / 64).astype("<f4")
            assert not np.any(bias.view("<u4") & 0xFFFF)
            flat = b.reshape(-1).view("<u1").copy()
            flat[EPILOGUE_BIAS_OFFSET:EPILOGUE_BIAS_OFFSET + 128] = bias.view("<u1")
            b = flat.view("<f2").reshape(64, 32)
    else:
        a = rng.integers(-2, 3, size=(17, 16), dtype=np.int16).astype("<f2")
        b = rng.integers(-2, 3, size=(16, 19), dtype=np.int16).astype("<f2")
        c = np.zeros((17, 19), dtype="<f4")
    a_name = "a.f32" if composition == "transformer_continuation_weight_offset" else "a.f16"
    for name, value in ((a_name, a), ("b.f16", b), ("c.f32", c)):
        (bundle / name).write_bytes(value.tobytes(order="C"))


def read_inputs(bundle, composition="single", weight_offset=4096):
    if composition in ("reduction_softmax", "ffn_gelu_layernorm", "ffn_gelu_layernorm_allrows",
                       "ffn_gelu_layernorm_wide", "transformer_layer", "transformer_two_layer"):
        rows = 32 if composition == "reduction_softmax" else 16
        cols = 32 if composition in ("ffn_gelu_layernorm_wide", "transformer_layer",
                                     "transformer_two_layer") else 16
        a = np.fromfile(bundle / "a.f16", dtype="<f2").reshape(rows, 64)
        b = np.fromfile(bundle / "b.f16", dtype="<f2").reshape(64, cols)
        c = np.fromfile(bundle / "c.f32", dtype="<f4").reshape(rows, cols)
        return a, b, c
    if composition == "transformer_layer_weight_offset":
        a = np.fromfile(bundle / "a.f16", dtype="<f2").reshape(16, 64)
        b = np.fromfile(bundle / "b.f16", dtype="<f2")
        c = np.fromfile(bundle / "c.f32", dtype="<f4").reshape(16, 32)
        return a, b, c
    if composition == "transformer_continuation_weight_offset":
        a = np.fromfile(bundle / "a.f32", dtype="<f4").reshape(16, 32)
        b = np.fromfile(bundle / "b.f16", dtype="<f2")
        c = np.fromfile(bundle / "c.f32", dtype="<f4").reshape(16, 32)
        return a, b, c
    if composition == "weight_offset":
        a = np.fromfile(bundle / "a.f16", dtype="<f2").reshape(16, 64)
        b = np.fromfile(bundle / "b.f16", dtype="<f2")
        c = np.fromfile(bundle / "c.f32", dtype="<f4").reshape(16, 32)
        return a, b, c
    if composition == "fp8":
        a = np.fromfile(bundle / "a.f16", dtype=np.uint8).reshape(32, 64)
        b = np.fromfile(bundle / "b.f16", dtype=np.uint8).reshape(64, 32)
        c = np.fromfile(bundle / "c.f32", dtype="<f4").reshape(32, 32)
        return a, b, c
    if composition in ("grid1", "grid2", "grid4", "grid8"):
        a = np.fromfile(bundle / "a.f16", dtype="<f2").reshape(128, 64)
        b = np.fromfile(bundle / "b.f16", dtype="<f2").reshape(64, 32)
        c = np.fromfile(bundle / "c.f32", dtype="<f4").reshape(128, 32)
        return a, b, c
    if composition in ("multigemm", "multigemm_fadd_fmul", "multigemm3", "multigemm_relu_vec", "chain_register", "chain_memory", "epilogue_register", "epilogue_memory", "grid1", "grid2", "grid4", "grid8", "fp8"):
        a = np.fromfile(bundle / "a.f16", dtype="<f2").reshape(32, 64)
        b = np.fromfile(bundle / "b.f16", dtype="<f2").reshape(64, 32)
        c = np.fromfile(bundle / "c.f32", dtype="<f4").reshape(32, 32)
        return a, b, c
    a = np.fromfile(bundle / "a.f16", dtype="<f2").reshape(17, 16)
    b = np.fromfile(bundle / "b.f16", dtype="<f2").reshape(16, 19)
    c = np.fromfile(bundle / "c.f32", dtype="<f4").reshape(17, 19)
    return a, b, c


def _rne32(value):
    return np.float32(value)


def _truncate_fp32(value):
    """Apply the measured fp32-A operand quantisation used by tensor.mac."""
    bits = np.asarray(value, dtype="<f4").view("<u4")
    return (bits & np.uint32(0xFFFFE000)).view("<f4").item()


def _mma16(a, b, c, *, truncate_a=False, truncate_b=False):
    """One measured 16-wide MMA issue (recon section 136 part 2), C FIRST.

    The 16 products are exact; P_i = RNE32(p_2i + p_2i+1), Q_j = RNE32(P_j + P_j+4), then
    acc = C and acc = RNE32(acc + Q_j) for j = 0..3. `c=None` is the no-C form (op5101/5107):
    acc starts at Q_0.

    [Corrected 2026-09-23: this function added C LAST (after the four Q terms), and a unit test
    pinned that order as "the measured tensor.mac rule". Section 136 measured C first, and so does
    tools/g17tensorqkv.py's model. The released multigemm bundle, run on main's code, differs from
    the C-last reference in 417 of 1024 elements and from the C-first model in 0; so do two
    further two-GEMM programs (docs/archive/g17-tensor-register-chain.md). The C-last observation that
    motivated the old order is the ACCUMULATE add - tlower's one explicit fadd of C after the
    whole K chain (section 29) - which `_gemm_mma` now applies separately.]
    """
    aa = [float(a[i]) for i in range(16)]
    if truncate_a:
        aa = [_truncate_fp32(x) for x in aa]
    bb = [float(b[i]) for i in range(16)]
    if truncate_b:
        bb = [_truncate_fp32(x) for x in bb]
    p = [_rne32(_rne32(aa[2*i] * bb[2*i]) + _rne32(aa[2*i+1] * bb[2*i+1]))
         for i in range(8)]
    q = [_rne32(p[j] + p[j + 4]) for j in range(4)]
    if c is None:
        acc = q[0]
        rest = q[1:]
    else:
        acc = _rne32(c)
        rest = q
    for value in rest:
        acc = _rne32(acc + value)
    return acc


def _gemm_mma(a, b, c, M, N, K, *, truncate_a=False, truncate_b=False):
    """Compose 16-wide issues in ascending memory-K order, as tlower emits them.

    The first issue is the no-C form and each later one takes the previous result as C (C
    first, `_mma16`). `c` is the body's ACCUMULATE input or None: tlower adds it once, with an
    explicit fadd after the whole chain, so it is added last here - once, not per issue."""
    if K % 16:
        raise ValueError("reference only covers 16-wide tensor issues")
    if M * N * K >= (1 << 18):
        # the same arithmetic vectorised over (m, n) (MM 25.144.1): the per-element loop below is ~33 million
        # Python issues at a prefill shape. test_g17gemmmmafast pins the two equal, bit for bit.
        return _gemm_mma_fast(a, b, c, M, N, K, truncate_a=truncate_a, truncate_b=truncate_b)
    out = np.empty((M, N), dtype="<f4")
    for m in range(M):
        for n in range(N):
            acc = None
            for start in range(0, K, 16):
                acc = _mma16(a[m, start:start + 16], b[start:start + 16, n], acc,
                             truncate_a=truncate_a, truncate_b=truncate_b)
            if c is not None:
                acc = _rne32(acc + _rne32(c[m, n]))
            out[m, n] = acc
    return out


def _gemm_mma_fast(a, b, c, M, N, K, *, truncate_a=False, truncate_b=False, block=512):
    """_gemm_mma vectorised over (m, n), in numpy float32 (every op rounds to nearest even, as `_rne32`):
    per 16-wide issue the exact products, P_i = RNE32(p_2i + p_2i+1), Q_j = RNE32(P_j + P_j+4), then the
    accumulator first (or Q_0 for the first issue) plus Q_0..Q_3 in order; `c` is added once at the end."""
    def trunc(x):
        return (np.ascontiguousarray(x, dtype="<f4").view("<u4") & np.uint32(0xFFFFE000)).view("<f4")
    A = np.asarray(a, dtype=np.float64).astype("<f4")[:M, :K]
    B = np.asarray(b, dtype=np.float64).astype("<f4")[:K, :N]
    if truncate_a:
        A = trunc(A)
    if truncate_b:
        B = trunc(B)
    out = np.empty((M, N), dtype="<f4")

    # column blocks are independent (each output keeps its own order), so they run on a thread pool; numpy's kernels
    # release the GIL. G17_REF_THREADS=1 runs them in order (test_g17simspeed)
    def col(n0):
        n1 = min(N, n0 + block)
        acc = None
        for s in range(0, K, 16):
            prod = A[:, s:s + 16, None] * B[None, s:s + 16, n0:n1]            # (M, 16, nb), float32 RNE
            P = prod[:, 0::2, :] + prod[:, 1::2, :]                           # P_i, i = 0..7
            Q = P[:, 0:4, :] + P[:, 4:8, :]                                   # Q_j, j = 0..3
            if acc is None:
                acc = Q[:, 0, :].copy()
                rest = (1, 2, 3)
            else:
                rest = (0, 1, 2, 3)
            for j in rest:
                acc = acc + Q[:, j, :]
        if c is not None:
            acc = acc + np.asarray(c, dtype="<f4")[:M, n0:n1]
        out[:, n0:n1] = acc
    starts = range(0, N, block)
    nt = min(int(os.environ.get("G17_REF_THREADS", "8")), len(starts))
    if nt > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(nt) as ex:
            list(ex.map(col, starts))
    else:
        for n0 in starts:
            col(n0)
    return out


def _transformer_layer_reference(inp, b, first_k):
    """Reference one measured 16x32 transformer layer without host-side shortcuts."""
    from agxforge.g17 import tensorreduce

    if first_k == 64:
        first = _gemm_mma(inp, b, None, 16, 32, 64)
    else:
        first = _gemm_mma(inp, b[:32, :], None,
                          16, 32, 32, truncate_a=True)
    expanded = _gemm_mma(first, b[:32, :], first, 16, 32, 32, truncate_a=True)
    scaled = np.asarray(expanded * np.float32(1.702), dtype="<f4")
    exponent = np.asarray(np.exp2(np.asarray(
        scaled * np.float32(-1.4426950408889634), dtype="<f4")), dtype="<f4")
    sigmoid = np.asarray(np.float32(1.0) /
                         np.asarray(exponent + np.float32(1.0), dtype="<f4"), dtype="<f4")
    gelu = np.asarray(expanded * sigmoid, dtype="<f4")
    contracted = _gemm_mma(gelu, b[:32, :], gelu, 16, 32, 32, truncate_a=True)
    contracted = np.asarray(contracted + np.asarray(b[:16, :32], dtype="<f4"), dtype="<f4")
    for row_index in range(16):
        lanes = tensorreduce.row_lane_values([list(map(float, contracted[row_index]))], 0,
                                              descending=False)
        total = np.float32(tensorreduce.row_sum(lanes, M=16, N=32)[0])
        squares = [[float(x * x) for x in contracted[row_index]]]
        square_lanes = tensorreduce.row_lane_values(squares, 0, descending=False)
        square_total = np.float32(tensorreduce.row_sum(square_lanes, M=16, N=32)[0])
        mean = np.float32(total * np.float32(1.0 / 32.0))
        variance = np.float32(square_total * np.float32(1.0 / 32.0) - mean * mean)
        variance = np.float32(max(float(variance), 0.0))
        inv_std = np.float32(1.0 / np.sqrt(np.float32(variance + np.float32(1.0e-5))))
        contracted[row_index] = np.asarray((contracted[row_index] - mean) * inv_std,
                                            dtype="<f4")
    return contracted


def _transformer_layer_weight_offset_reference(inp, b, offsets=(0, 4096, 8192)):
    """Reference one layer with three measured B regions."""
    first_k = int(inp.shape[1])
    w0_start = offsets[0] // 2
    w1_start = offsets[1] // 2
    w2_start = offsets[2] // 2
    w0 = b[w0_start:w0_start + first_k * 32].reshape(first_k, 32)
    w1 = b[w1_start:w1_start + 32 * 32].reshape(32, 32)
    w2 = b[w2_start:w2_start + 32 * 32].reshape(32, 32)
    residual = b[:16 * 32].reshape(16, 32).astype("<f4")
    first = _gemm_mma(inp, w0, None, 16, 32, first_k,
                      truncate_a=(inp.dtype.kind == "f" and inp.dtype.itemsize == 4))
    expanded = _gemm_mma(first, w1, first, 16, 32, 32, truncate_a=True)
    scaled = np.asarray(expanded * np.float32(1.702), dtype="<f4")
    exponent = np.asarray(np.exp2(np.asarray(
        scaled * np.float32(-1.4426950408889634), dtype="<f4")), dtype="<f4")
    sigmoid = np.asarray(np.float32(1.0) /
                         np.asarray(exponent + np.float32(1.0), dtype="<f4"), dtype="<f4")
    gelu = np.asarray(expanded * sigmoid, dtype="<f4")
    contracted = _gemm_mma(gelu, w2, gelu, 16, 32, 32, truncate_a=True)
    contracted = np.asarray(contracted + residual, dtype="<f4")
    from agxforge.g17 import tensorreduce
    for row_index in range(16):
        lanes = tensorreduce.row_lane_values([list(map(float, contracted[row_index]))], 0,
                                              descending=False)
        total = np.float32(tensorreduce.row_sum(lanes, M=16, N=32)[0])
        squares = [[float(x * x) for x in contracted[row_index]]]
        square_lanes = tensorreduce.row_lane_values(squares, 0, descending=False)
        square_total = np.float32(tensorreduce.row_sum(square_lanes, M=16, N=32)[0])
        mean = np.float32(total * np.float32(1.0 / 32.0))
        variance = np.float32(square_total * np.float32(1.0 / 32.0) - mean * mean)
        variance = np.float32(max(float(variance), 0.0))
        inv_std = np.float32(1.0 / np.sqrt(np.float32(variance + np.float32(1.0e-5))))
        contracted[row_index] = np.asarray((contracted[row_index] - mean) * inv_std,
                                            dtype="<f4")
    return contracted


def reference(a, b, c, composition="single", weight_offset=4096):
    """Independent instruction-level reference for the measured TensorOps classes."""
    if composition == "weight_offset":
        if weight_offset not in (4096, 4352, 8192, 12288, 16384, 20480):
            raise ValueError("weight-offset arm has no measured B byte offset")
        w0 = b[:64 * 32].reshape(64, 32)
        start = weight_offset // 2
        w1 = b[start:start + 32 * 32].reshape(32, 32)
        first = _gemm_mma(a, w0, None, 16, 32, 64)
        return _gemm_mma(first, w1, None, 16, 32, 32,
                         truncate_a=True)
    if composition == "reduction_softmax":
        # Independent reference for the compiler-owned application arm.  The implementation uses
        # base-2 exp because that is the existing measured unary operation; the tolerance is
        # preregistered rather than claiming bit identity for a transcendental instruction.  The
        # preceding ordinary tensor body overwrites C with the half/half GEMM, so the input C
        # payload is only a guarded initial value here.
        out = _gemm_mma(a, b, None, 32, 16, 64)
        row = np.asarray(out[0], dtype="<f4")
        maximum = np.float32(np.max(row))
        weights = np.exp2(np.asarray(row - maximum, dtype="<f4")).astype("<f4")
        total = np.float32(np.sum(weights, dtype="<f4"))
        out[0] = (weights / total).astype("<f4")
        return out
    if composition == "transformer_layer_weight_offset":
        return _transformer_layer_weight_offset_reference(a, b)
    if composition == "transformer_continuation_weight_offset":
        return _transformer_layer_weight_offset_reference(a, b)
    if composition == "transformer_two_layer":
        return _transformer_layer_reference(_transformer_layer_reference(a, b, 64), b, 32)
    if composition == "transformer_layer":
        from agxforge.g17 import tensorreduce
        first = _gemm_mma(a, b, None, 16, 32, 64)
        expanded = _gemm_mma(first, b[:32, :], first, 16, 32, 32, truncate_a=True)
        gelu = expanded.copy()
        scaled = np.asarray(gelu * np.float32(1.702), dtype="<f4")
        exponent = np.asarray(np.exp2(np.asarray(
            scaled * np.float32(-1.4426950408889634), dtype="<f4")), dtype="<f4")
        sigmoid = np.asarray(np.float32(1.0) /
                             np.asarray(exponent + np.float32(1.0), dtype="<f4"), dtype="<f4")
        gelu = np.asarray(gelu * sigmoid, dtype="<f4")
        contracted = _gemm_mma(gelu, b[:32, :], gelu, 16, 32, 32, truncate_a=True)
        contracted = np.asarray(contracted + np.asarray(b[:16, :32], dtype="<f4"), dtype="<f4")
        for row_index in range(16):
            lanes = tensorreduce.row_lane_values([list(map(float, contracted[row_index]))], 0,
                                                  descending=False)
            total = np.float32(tensorreduce.row_sum(lanes, M=16, N=32)[0])
            squares = [[float(x * x) for x in contracted[row_index]]]
            square_lanes = tensorreduce.row_lane_values(squares, 0, descending=False)
            square_total = np.float32(tensorreduce.row_sum(square_lanes, M=16, N=32)[0])
            mean = np.float32(total * np.float32(1.0 / 32.0))
            variance = np.float32(square_total * np.float32(1.0 / 32.0) - mean * mean)
            variance = np.float32(max(float(variance), 0.0))
            inv_std = np.float32(1.0 / np.sqrt(np.float32(variance + np.float32(1.0e-5))))
            contracted[row_index] = np.asarray((contracted[row_index] - mean) * inv_std,
                                                dtype="<f4")
        return contracted
    if composition in ("ffn_gelu_layernorm", "ffn_gelu_layernorm_allrows",
                       "ffn_gelu_layernorm_wide"):
        from agxforge.g17 import tensorreduce
        first_n = 32 if composition == "ffn_gelu_layernorm_wide" else 16
        first = _gemm_mma(a, b, None, 16, first_n, 64)
        # Match the emitted scalar instruction sequence: FP32-rounded scale, base-2 exponent,
        # reciprocal, then multiply. The released class applies this to row zero; the spatial
        # class applies the same compiler-owned sequence independently to every row.
        rows = (range(16) if composition in ("ffn_gelu_layernorm_allrows",
                                             "ffn_gelu_layernorm_wide") else (0,))
        for row_index in rows:
            row = first[row_index].copy()
            scaled = np.asarray(row * np.float32(1.702), dtype="<f4")
            exponent = np.asarray(np.exp2(np.asarray(scaled * np.float32(-1.4426950408889634),
                                                     dtype="<f4")), dtype="<f4")
            sigmoid = np.asarray(np.float32(1.0) /
                                 np.asarray(exponent + np.float32(1.0), dtype="<f4"), dtype="<f4")
            first[row_index] = np.asarray(row * sigmoid, dtype="<f4")
        second_n = 32 if composition == "ffn_gelu_layernorm_wide" else 16
        second = _gemm_mma(first, b[:32 if second_n == 32 else 16, :], first,
                           16, second_n, 32 if second_n == 32 else 16, truncate_a=True)
        for row_index in rows:
            lanes = tensorreduce.row_lane_values([list(map(float, second[row_index]))], 0,
                                                  descending=False)
            total = np.float32(tensorreduce.row_sum(lanes, M=16, N=second_n)[0])
            squares = [[float(x * x) for x in second[row_index]]]
            square_lanes = tensorreduce.row_lane_values(squares, 0, descending=False)
            square_total = np.float32(tensorreduce.row_sum(square_lanes, M=16, N=second_n)[0])
            mean = np.float32(total * np.float32(1.0 / second_n))
            variance = np.float32(square_total * np.float32(1.0 / second_n) - mean * mean)
            variance = np.float32(max(float(variance), 0.0))
            inv_std = np.float32(1.0 / np.sqrt(np.float32(variance + np.float32(1.0e-5))))
            second[row_index] = np.asarray((second[row_index] - mean) * inv_std, dtype="<f4")
        return second
    if composition == "fp8":
        # fp8 embeds exactly in bf16, so the MMA rule applies to the exactly decoded values
        out = _gemm_mma(_fp8_values(a, "e4m3"), _fp8_values(b, "e5m2"), None, 32, 32, 64)
        out[0, 0] = _rne32(out[0, 0] + np.float32(1.0))
        return out
    if composition in ("grid1", "grid2", "grid4", "grid8"):
        # every threadgroup's block is the same GEMM's rows; no epilogue
        return _gemm_mma(a, b, None, 128, 32, 64)
    if composition in ("chain_register", "chain_memory", "epilogue_register", "epilogue_memory", "grid1", "grid2", "grid4", "grid8", "fp8"):
        # No scalar boundary: GEMM2 reads GEMM1's result exactly, through memory or registers.
        first = _gemm_mma(a, b, None, 32, 32, 64)
        if composition.startswith("epilogue"):
            bias = np.frombuffer(b.tobytes()[EPILOGUE_BIAS_OFFSET:EPILOGUE_BIAS_OFFSET + 128], dtype="<f4")
            scale = np.frombuffer(struct.pack("<I", EPILOGUE_SCALE_BITS), dtype="<f4")[0]
            first = np.asarray(first + bias[None, :], dtype="<f4")
            first = np.asarray(first * scale, dtype="<f4")
            first = np.maximum(first, np.float32(0.0)).astype("<f4")
        second = _gemm_mma(first, b, None, 32, 32, 32, truncate_a=True)
        second[0, 0] = _rne32(second[0, 0] + np.float32(1.0))
        return second
    if composition in ("multigemm", "multigemm_fadd_fmul", "multigemm3", "multigemm_relu_vec"):
        # The first body writes C, the ordinary scalar epilogue changes C[0,0], and the
        # second body reads that same C buffer as its fp32 A operand.  Both compiler calls use
        # accumulate=False, so GEMM2 starts a fresh zero accumulator even though its A and output
        # storage are the same public buffer.
        first = _gemm_mma(a, b, None, 32, 32, 64)
        first[0, 0] = _rne32(first[0, 0] + np.float32(1.0))
        if composition in ("multigemm_fadd_fmul", "multigemm3"):
            first[0, 0] = _rne32(first[0, 0] * np.float32(0.5))
        if composition == "multigemm_relu_vec":
            first[0, :4] = np.asarray(
                [_rne32(_rne32(max(float(value), 0.0)) * np.float32(0.5) + np.float32(1.0))
                 for value in first[0, :4]], dtype="<f4")
        second = _gemm_mma(first, b, None, 32, 32, 32, truncate_a=True)
        if composition == "multigemm3":
            second[0, 0] = _rne32((second[0, 0] + np.float32(1.0)) * np.float32(0.5))
            return _gemm_mma(second, b, None, 32, 32, 32, truncate_a=True)
        if composition == "multigemm_relu_vec":
            second[0, 0] = _rne32((_rne32(second[0, 0] + np.float32(1.0))) * np.float32(0.5))
            return _gemm_mma(second, b, None, 32, 32, 32, truncate_a=True)
        return second
    out = np.empty((17, 19), dtype="<f4")
    for m in range(17):
        for n in range(19):
            p = []
            for i in range(8):
                p.append(_rne32(_rne32(a[m, 2*i]) * _rne32(b[2*i, n]) +
                                _rne32(a[m, 2*i+1]) * _rne32(b[2*i+1, n])))
            q = [_rne32(p[j] + p[j+4]) for j in range(4)]
            acc = _rne32(c[m, n])
            for value in q:
                acc = _rne32(acc + value)
            out[m, n] = _rne32(acc)
    # The scalar epilogue indexes C by threadgroup_position_in_grid.x.  The exact one-group
    # launch has position zero, so it updates only C[0, 0]; the rest of the GEMM result is kept.
    out[0, 0] = _rne32(out[0, 0] + np.float32(1.0))
    return out


def _frames(stdout: bytes):
    frames, cursor = [], 0
    while cursor < len(stdout):
        if len(stdout) - cursor < 4:
            raise ValueError("tensor worker returned a truncated frame header")
        size = struct.unpack_from("<I", stdout, cursor)[0]; cursor += 4
        if size > 65536 or len(stdout) - cursor < size:
            raise ValueError("tensor worker returned a truncated JSON frame")
        header = json.loads(stdout[cursor:cursor + size]); cursor += size
        payload_size = int(header.get("bytes", 0))
        if payload_size < 0 or len(stdout) - cursor < payload_size:
            raise ValueError("tensor worker returned a truncated payload")
        payload = stdout[cursor:cursor + payload_size]; cursor += payload_size
        frames.append((header, payload))
    return frames


def build_worker(destination: Path):
    destination = Path(destination)
    with tempfile.TemporaryDirectory(prefix="g17-tensor-worker-") as td:
        result = subprocess.run([
            "clang", "-fobjc-arc", "-O2", "-Wall", "-Wextra", "-Werror",
            "-framework", "Foundation", "-framework", "Metal", "-o", str(destination),
            str(ROOT / "tools/g17commonworker.m")], cwd=ROOT, capture_output=True,
            text=True, timeout=30)
        if result.returncode:
            raise RuntimeError("common worker compilation failed: " + result.stderr[-4000:])
    destination.chmod(0o700)
    return {"sha256": sha(destination.read_bytes()), "path": str(destination)}


def _dispatch_never_kill(argv, after, bundle):
    """Run the dispatching worker and NEVER kill it (MM 25.114.5, for looping programs). Killing the
    host does not cancel the kernel (MM 25.116): a killed host leaves an orphan at 100% GPU and frees
    the GPU lock under it. So past `after` seconds this writes DISPATCH-TIMEOUT into the bundle,
    says so on stderr, and goes on waiting with the lock held: the operator stops all GPU work and
    reports, and the lock keeps every other dispatcher off a GPU that may be running away."""
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = proc.communicate(timeout=after)
    except subprocess.TimeoutExpired:
        (Path(bundle) / "DISPATCH-TIMEOUT").write_text(
            "worker pid %d had not returned after %d s; it is NOT killed and the GPU lock stays held\n"
            % (proc.pid, after))
        print("DISPATCH-TIMEOUT: worker pid %d has not returned after %d s. Not killing it (MM 25.116). "
              "STOP ALL GPU WORK and report." % (proc.pid, after), file=sys.stderr, flush=True)
        out, err = proc.communicate()
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def run(bundle: Path, *, queries=3, worker=None, receipt=None, composition=None, mismatch_ok=False, save_output=False,
        never_kill_after=None):
    """...; mismatch_ok=True (timing arms only, tools/g17tensorspeed.py) records a reference mismatch -
    element count, max error, the saved arrays - and keeps timing instead of raising. never_kill_after
    (seconds; the looping programs of MM 25.114.5): the dispatch is waited for without a kill, see
    _dispatch_never_kill."""
    from agxforge.g17 import runtime
    import g17packeddispatch
    import g17commonstage

    bundle = Path(bundle).resolve()
    manifest = runtime.ImageContract.read(json.loads((bundle / "manifest.json").read_text()))
    if manifest.kind != runtime.TENSOR_KIND:
        raise ValueError("bundle is not the measured tensor_gemm class")
    if composition is None:
        composition = ("transformer_two_layer" if manifest.name == "tensor_transformer_two_layer_runtime_demo" else
                       "transformer_continuation_weight_offset" if manifest.name == "tensor_transformer_continuation_weight_offset_runtime_demo" else
                       "transformer_layer_weight_offset" if manifest.name == "tensor_transformer_layer_weight_offset_runtime_demo" else
                       "transformer_layer" if manifest.name == "tensor_transformer_layer_runtime_demo" else
                       "multigemm_relu_vec" if manifest.name == "tensor_multigemm_relu_vec_residual_runtime_demo" else
                       "generic" if manifest.name == GENERIC_NAME else
                       "fp8" if manifest.name == "tensor_gemm_fp8_runtime_demo" else
                       ("grid%d" % (int(manifest.tensor.grid[0]) // 32)) if manifest.name == "tensor_gemm_grid_runtime_demo" else
                       "chain_register" if manifest.name == "tensor_multigemm_register_runtime_demo" else
                       "chain_memory" if manifest.name == "tensor_multigemm_adjacent_runtime_demo" else
                       "epilogue_register" if manifest.name == "tensor_multigemm_epilogue_register_runtime_demo" else
                       "epilogue_memory" if manifest.name == "tensor_multigemm_epilogue_adjacent_runtime_demo" else
                       "weight_offset" if manifest.name == "tensor_multigemm_weight_offset_runtime_demo" else
                       "multigemm3" if manifest.name == "tensor_multigemm_fadd_fmul_gemm_runtime_demo" else
                       "multigemm_fadd_fmul" if manifest.name == "tensor_multigemm_fadd_fmul_runtime_demo"
                       else "multigemm" if manifest.name == "tensor_multigemm_runtime_demo" else
                       "ffn_gelu_layernorm_wide" if manifest.name == "tensor_ffn_gelu_layernorm_wide_runtime_demo" else
                       "ffn_gelu_layernorm_allrows" if manifest.name == "tensor_ffn_gelu_layernorm_allrows_runtime_demo" else
                       "ffn_gelu_layernorm" if manifest.name == "tensor_ffn_gelu_layernorm_runtime_demo" else
                       "reduction_softmax" if manifest.name == "tensor_row_softmax_runtime_demo" else "single")
    if composition not in ("single", "multigemm", "multigemm_fadd_fmul", "multigemm3", "chain_register", "chain_memory", "epilogue_register", "epilogue_memory", "grid1", "grid2", "grid4", "grid8", "fp8", "generic",
                           "multigemm_relu_vec", "reduction_softmax", "ffn_gelu_layernorm",
                           "ffn_gelu_layernorm_allrows", "ffn_gelu_layernorm_wide",
                           "transformer_layer", "transformer_layer_weight_offset", "transformer_two_layer",
                           "transformer_continuation_weight_offset", "weight_offset"):
        raise ValueError("unknown tensor composition %r" % composition)
    weight_offset = int(manifest.tensor.weight_offset_b or 4096)
    if composition == "generic":
        expected = generic_reference(bundle, generic_spec(json.loads((bundle / "generic.json").read_text())))
    else:
        a, b, c = read_inputs(bundle, composition, weight_offset=weight_offset)
        expected = reference(a, b, c, composition, weight_offset=weight_offset)
    reference_stages = None
    if composition == "transformer_two_layer":
        layer0_reference = _transformer_layer_reference(a, b, 64)
        reference_stages = {
            "layer_count": 2,
            "layer0_output_sha256": sha(np.asarray(layer0_reference, dtype="<f4").tobytes()),
            "gpu_intermediate_readback": False,
            "boundary": "GPU-resident C buffer; no host numerical stage or readback",
        }
    if worker is None:
        with tempfile.TemporaryDirectory(prefix="g17-tensor-worker-bin-") as td:
            # mismatch_ok travels with the rebuilt call: dropping it here made every timing arm
            # whose answer is wrong by design raise instead of timing (the rotate() arms, 2026-09-24)
            return run(bundle, queries=queries, worker=Path(td) / "common-worker", receipt=receipt,
                       composition=composition, mismatch_ok=mismatch_ok, save_output=save_output,
                       never_kill_after=never_kill_after)
    worker = Path(worker)
    build = build_worker(worker)
    # The common runtime serializes GPU use under its process-wide advisory lock and checks the
    # vendor event counters around every native stage. Tensor mode uses the same boundary even
    # though its mixed-dtype framing is separate from the scan Worker protocol.
    with g17commonstage.lock_gpu():
        before = g17packeddispatch.gpu_events()
        load = subprocess.run([str(worker), str(bundle), "--tensor-load-approved"],
                              capture_output=True, timeout=30)
        if load.returncode:
            raise RuntimeError("tensor load-only failed: " + load.stderr.decode(errors="replace")[-4000:])
        load_report = json.loads(load.stdout)
        if load_report != {"status": 0, "load_only": True, "gpu_dispatched": False}:
            raise ValueError("unexpected tensor load-only response")
        if before != g17packeddispatch.gpu_events():
            raise RuntimeError("GPU diagnostics changed during tensor load-only")
        # The native worker's established dispatch wire has a positional input-directory slot
        # before the query limit. Tensor mode owns its input files in the bundle, but keeps that
        # slot (a literal marker) so it cannot accidentally select the legacy scalar grammar.
        argv = [str(worker), str(bundle), "tensor-inputs", str(queries), "--tensor-dispatch-approved"]
        if never_kill_after is None:
            dispatch = subprocess.run(argv, capture_output=True, timeout=30)
        else:
            dispatch = _dispatch_never_kill(argv, never_kill_after, bundle)
        if dispatch.returncode:
            raise RuntimeError("tensor dispatch failed: " + dispatch.stderr.decode(errors="replace")[-4000:])
        frames = _frames(dispatch.stdout)
        if before != g17packeddispatch.gpu_events():
            raise RuntimeError("GPU diagnostics changed during tensor dispatch")
    if len(frames) != queries + 1 or frames[0][0].get("sequence") != 0:
        raise ValueError("tensor worker handshake or frame count differs from the contract")
    attempts = []
    for sequence, (header, payload) in enumerate(frames[1:], 1):
        if (header.get("sequence"), header.get("status"), header.get("gpu_dispatched"),
            header.get("boundary_guard"), header.get("readonly_inputs")) != (sequence, 0, True, True, True):
            raise ValueError(f"tensor worker response {sequence} is not fully checked")
        rows = int(manifest.shape.rows)
        columns = int(manifest.shape.columns)
        got = np.frombuffer(payload, dtype="<f4").copy().reshape(rows, columns)
        # UNSCORED ELEMENTS, stated by the spec and recorded rather than dropped: the open-neighbour
        # imageblock arm lets lane 31 read x = 32, outside the 32x1 tile (as Set C's validated dx1
        # program does), so C[0,31] has no prediction. Its value goes in the receipt.
        unscored = {}
        nan_patterns = None
        if composition == "generic":
            for (r, c) in generic_unscored(generic_spec(json.loads((bundle / "generic.json").read_text()))):
                unscored["%d,%d" % (r, c)] = "%08x" % int(got.view("<u4")[r, c])
                got[r, c] = expected[r, c]
        if composition in ("reduction_softmax", "ffn_gelu_layernorm", "ffn_gelu_layernorm_allrows",
                           "ffn_gelu_layernorm_wide", "transformer_layer", "transformer_layer_weight_offset",
                           "transformer_continuation_weight_offset", "transformer_two_layer"):
            max_abs = float(np.max(np.abs(got - expected)))
            same = bool(np.allclose(got, expected, rtol=0.0, atol=2.0e-5, equal_nan=True))
        elif composition == "generic" and generic_abs_bound(bundle, generic_spec(json.loads((bundle / "generic.json").read_text()))) is not None:
            abound = generic_abs_bound(bundle, generic_spec(json.loads((bundle / "generic.json").read_text())))
            max_abs = float(np.max(np.abs(got - expected)))
            with np.errstate(invalid="ignore"):
                same = bool(np.all((got.view("<u4") == np.asarray(expected, dtype="<f4").view("<u4")) |
                                   (np.abs(got.astype(np.float64) - expected.astype(np.float64)) <= abound)))
        elif composition == "generic" and generic_ulp_bound(generic_spec(json.loads((bundle / "generic.json").read_text()))):
            bound = generic_ulp_bound(generic_spec(json.loads((bundle / "generic.json").read_text())))
            max_abs = float(np.max(np.abs(got - expected)))
            same = bool(np.max(_ulp_distance(got, expected)) <= bound)
        else:
            # the bit-exact arm reports the real difference too: a constant 0.0 here made a refusal
            # read "differs (max_abs=0.0)" while all 8,192 elements differed by up to 383
            # (neg_kloop_claims_scale, results/g17-tensor-kloop-v1)
            with np.errstate(invalid="ignore", over="ignore"):
                diff = np.abs(got.astype(np.float64) - np.asarray(expected, dtype=np.float64))
            max_abs = float(np.nanmax(diff)) if np.any(~np.isnan(diff)) else float("nan")
            same = np.array_equal(got.view("<u4"), expected.view("<u4"))
            if composition == "generic" and generic_nan_class(generic_spec(json.loads((bundle / "generic.json").read_text()))):
                # OCP fixes that an element IS NaN, not its payload: NaN positions must agree, every
                # other element (infinities included) bit for bit; the NaN patterns seen are recorded
                en, gn = np.isnan(np.asarray(expected, dtype="<f4")), np.isnan(got)
                same = bool(np.array_equal(en, gn) and
                            np.array_equal(got.view("<u4")[~en], np.asarray(expected, dtype="<f4").view("<u4")[~en]))
                nan_patterns = sorted({"%08x" % v for v in got.view("<u4")[gn].tolist()})
        if save_output:
            # the GPU's output kept even when it passes: a HARDWARE-TO-HARDWARE comparison (the one-shot
            # against the online attention, docs/g17-tensorops-machine-model.md 25.114) needs both sides on disk
            np.savez(bundle / f"output-q{sequence}.npz", got=got, got_u32=got.view("<u4"))
        if not same:
            # Keep the failing evidence: a refusal that discards what the GPU returned cannot be diagnosed.
            np.savez(bundle / f"mismatch-q{sequence}.npz", got=got, expected=np.asarray(expected, dtype="<f4"),
                         got_u32=got.view("<u4"), expected_u32=np.asarray(expected).view("<u4"))
            if not mismatch_ok:
                raise RuntimeError(f"tensor query {sequence} differs from the independent reference (max_abs={max_abs})")
        mismatched = 0 if same else int(np.count_nonzero(got.view("<u4") != np.asarray(expected).view("<u4")))
        attempts.append({"sequence": sequence, "status": "passed" if same else "timed_mismatch",
                         "output_sha256": sha(payload), "mismatched_elements": mismatched,
                         "max_abs_vs_reference": float(np.max(np.abs(got - np.asarray(expected, dtype="<f4")))),
                         **({"unscored": unscored} if unscored else {}),
                         **({"nan_elements": int(np.isnan(got).sum()), "nan_patterns": nan_patterns,
                             "inf_elements": int(np.isinf(got).sum())} if nan_patterns is not None else {}),
                         "gpu_seconds": header.get("gpu_seconds"),
                         "reference_bit_exact": composition not in ("reduction_softmax", "ffn_gelu_layernorm", "ffn_gelu_layernorm_allrows", "ffn_gelu_layernorm_wide", "transformer_layer", "transformer_layer_weight_offset", "transformer_continuation_weight_offset", "transformer_two_layer"), "bytes": len(payload),
                         "reference_max_abs": max_abs, "reference_bound": 2.0e-5 if composition in ("reduction_softmax", "ffn_gelu_layernorm", "ffn_gelu_layernorm_allrows", "ffn_gelu_layernorm_wide", "transformer_layer", "transformer_layer_weight_offset", "transformer_continuation_weight_offset", "transformer_two_layer") else 0.0,
                         "accumulated_error_max_abs": max_abs if composition == "transformer_two_layer" else None,
                         "reference_within_bound": True, "boundary_guard": True, "readonly_inputs": True})
    report_files = ("manifest.json", "scan.arc.metallib", "scan.lib.metallib", "scan.o",
                    "program.bin") + (("a.f32", "b.f16", "c.f32")
                                       if composition == "transformer_continuation_weight_offset"
                                       else ("a.f16", "b.f16", "c.f32"))
    report = {"status": "passed" if all(a["status"] == "passed" for a in attempts) else "timed_mismatch",
              "kind": manifest.kind, "name": manifest.name,
              "shape": manifest.tensor.model_dump(mode="json"), "queries": attempts,
              "load_only": load_report, "worker": build,
              "source": source_identity(),
              "files": {name: sha((bundle / name).read_bytes()) for name in report_files},
              "gpu_dispatched": True,
              "weight_offset_b": weight_offset if composition == "weight_offset" else None,
              "reference_stages": reference_stages,
              "reference": ("instruction-level RNE32 MMA with C-first accumulation (section 136), measured fp32-A truncation, three measured B regions, compiler-owned GELU/residual/LayerNorm, and GPU-resident C"
                            if composition == "transformer_layer_weight_offset" else
                            "instruction-level RNE32 MMA with C-first accumulation (section 136), measured fp32-A truncation, two complete compiler-owned transformer layers, two GELU/residual/LayerNorm regions, and GPU-resident C between layers"
                            if composition == "transformer_two_layer" else
                            "instruction-level RNE32 MMA with C-first accumulation (section 136), measured fp32-A truncation, three tensor regions, compiler-owned GELU, half residual widen/add, and explicit FP32 LayerNorm"
                            if composition == "transformer_layer" else
                            "instruction-level RNE32 MMA with C-first accumulation (section 136), measured fp32-A truncation, vector ReLU-affine block, scalar residual normalization, and fresh GEMM accumulators"
                             if composition == "multigemm_relu_vec" else
                            "instruction-level RNE32 MMA with C-first accumulation (section 136), measured fp32-A truncation, distinct W0/W1 in one binding-2 buffer at a measured byte offset, and fresh GEMM2 accumulator"
                             if composition == "weight_offset" else
                            "compiler-owned GELU approximation, residual GEMM accumulation, and explicit FP32 row LayerNorm"
                             if composition in ("ffn_gelu_layernorm", "ffn_gelu_layernorm_allrows", "ffn_gelu_layernorm_wide") else
                            "FP32 compiler-owned row max/exp2/row sum with explicit pos_b lane routing"
                             if composition == "reduction_softmax" else
                            "instruction-level RNE32 MMA with C-first accumulation (section 136), measured fp32-A truncation, two scalar boundaries, and fresh GEMM accumulators"
                             if composition == "multigemm3" else
                            "instruction-level RNE32 MMA with C-first accumulation (section 136), measured fp32-A truncation, fresh GEMM2 accumulator, and FP32 add/mul"
                             if composition == "multigemm_fadd_fmul" else
                            "instruction-level RNE32 MMA with C-first accumulation (section 136) on ml_dtypes-decoded fp8 (e4m3 A, e5m2 B), then the C[0,0] += 1 epilogue"
                             if composition == "fp8" else
                            "instruction-level RNE32 MMA with C-first accumulation (section 136); one 128x32x64 GEMM whose rows are split over the launched threadgroups"
                             if composition.startswith("grid") else
                            "instruction-level RNE32 MMA with C-first accumulation (section 136), measured fp32-A truncation, adjacent bodies with no scalar boundary, then the C[0,0] += 1 epilogue"
                             if composition in ("chain_register", "chain_memory", "epilogue_register", "epilogue_memory", "grid1", "grid2", "grid4", "grid8", "fp8") else
                            "instruction-level RNE32 MMA with C-first accumulation (section 136), measured fp32-A truncation, and fresh GEMM2 accumulator"
                             if composition == "multigemm" else
                             "instruction-level RNE32 MMA with C-first accumulation (section 136) plus FP32 add"),
              "gpu_diagnostics_unchanged": True, "composition": composition}
    if receipt is not None:
        Path(receipt).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--generic", default=None,
                        help='gemm_generic spec as JSON, e.g. {"M":64,"N":32,"K":64,"a":"half","b":"half","threadgroups":2}')
    parser.add_argument("--composition", choices=("single", "multigemm", "multigemm_fadd_fmul", "multigemm3", "chain_register", "chain_memory", "epilogue_register", "epilogue_memory", "grid1", "grid2", "grid4", "grid8", "fp8", "generic",
                                                   "multigemm_relu_vec", "reduction_softmax", "ffn_gelu_layernorm",
                                                   "ffn_gelu_layernorm_allrows", "ffn_gelu_layernorm_wide",
                                                   "transformer_layer", "transformer_two_layer",
                                                   "transformer_layer_weight_offset", "transformer_continuation_weight_offset",
                                                   "transformer_sequence2", "transformer_sequence6",
                                                   "weight_offset"), default="single")
    parser.add_argument("--weight-offset", type=int, choices=(4096, 4352, 8192, 12288, 16384, 20480), default=4096)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    if args.prepare == args.run:
        parser.error("select exactly one of --prepare or --run")
    if args.composition in ("transformer_sequence2", "transformer_sequence6"):
        length = 2 if args.composition.endswith("2") else 6
        report = (author_transformer_sequence(args.bundle, length=length) if args.prepare else
                  run_transformer_sequence(args.bundle, queries=args.queries, receipt=args.receipt))
    else:
        report = (author_generic(args.bundle, json.loads(args.generic)) if args.prepare and args.composition == "generic" else
                  author(args.bundle, composition=args.composition, weight_offset=args.weight_offset) if args.prepare else
                  run(args.bundle, queries=args.queries, receipt=args.receipt,
                      composition=args.composition))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        # a named refusal (MM P12's, runtime.COOPERATIVE_SHARING_REFUSAL) already starts "refused:"
        text = str(error)
        print(text if text.startswith("refused:") else "refused: " + text, file=sys.stderr)
        raise SystemExit(2)
