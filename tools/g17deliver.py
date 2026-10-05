#!/usr/bin/env python3
"""Every kernel bundle the decode graph runs, built from repository sources and verified on hardware (MM 25.141.15).

Until this tool the graph's bundles were built by one-off scripts into /private/tmp; nothing in the repository could
rebuild them. `build` compiles each kind from the builders in tools/ (g17qmv, g17decodeops, g17attn, g17gen),
authors a bundle, dispatches it ONCE with its output region pre-filled with a 0x7f sentinel, and compares every
output word with the kind's reference. Only if every bundle is bit-exact is the index written.

    python3 tools/g17deliver.py build --bits 4 --cap 272 --out DIR [--kinds qmv,norm,head,attn,gen]
    python3 tools/g17deliver.py check DIR/index.json        rebuild compile-only; every recorded sha must match

The index (DIR/index.json) is ONE flat list; an entry is selected by (kind, bits, role, variant) and carries:
kind (qmv | norm | lm_head | attn | gen_argmax | gen_step | prefill_attn), bits (qmv, lm_head), role (qkv | wo_res1 | ffn | w2_res2 |
head | attn_norm | ffn_norm | final_norm), variant, cap (attn, gen), bundle (a path relative to the index),
threadgroups, threads_per_group, base, slot_map, program_sha256, recipe (builder + the full layout, which `check`
rebuilds from), the layout offsets the graph asserts, and the fields the earlier hand-built index files carried.

Bundles are CONTENT-ADDRESSED: bundles/<name>-<sha256[:16]>. The same sources give the same bytes.

Verification dispatches through tools/g17bundlerun.m, compiled on first use into ~/.cache/agxforge keyed by its
source hash; its command buffers take the machine GPU lock (tools/g17gpulock.h).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import g17attn as A  # noqa: E402
import g17decodeops as O  # noqa: E402
import g17gen as G  # noqa: E402
import g17qmv as Q  # noqa: E402

EPS = 1e-5
NAME = "tensor_gemm_generic_runtime_demo"
SENT = b"\x7f"
# the v2 fp32-x qmv (MM 25.139.7 / 25.141.9): xvec, wpt 2 vector loads, hoisted constants
BASE = dict(interleave=True, lean=True, coalesced=True, a16=True, xvec=True, wpt=2, vload=True, hoist_consts=True)
# THE MODEL'S SHAPES (MM 25.182): d_model, the attention output width (heads x head_dim, wo's K), the fused qkv width, the
# FFN width, the vocabulary (the argmax's padded width and lanes), and the norm eps. InternLM2.5-1.8B is the default and
# every bundle it builds is unchanged; `build --arch qwen3` selects Qwen3-0.6B's (tools/g17qwen3.py). The attention
# (16 q heads, 8 KV heads of 128) is the same in both, so only the single-sequence projections, norms, head and
# generation step read this table.
ARCHS = {"internlm2": dict(d=2048, hd=2048, qkv=4096, ffn=8192, vocab=92544, vocab_pad=92544, per_lane=12, eps=1e-5),
         "qwen3": dict(d=1024, hd=2048, qkv=4096, ffn=3072, ffn_prefill=4096, vocab=151936, vocab_pad=152064, per_lane=24, eps=1e-6,
                       ksplit={4: dict(w2_res2=2)}),     # K 3,072 is 6 trips: split-K 4 does not divide them
         # Qwen3-8B: 32 query heads against 8 KV heads (GQA 4), the attention output 32 x 128 = 4,096 (wo's K)
         "qwen3_8b": dict(d=4096, hd=4096, qkv=6144, ffn=12288, ffn_prefill=16384, vocab=151936, vocab_pad=152064,
                          per_lane=24, eps=1e-6, heads=32, kv_heads=8,
                          qsm_sk_occ={"qkv": 1, "wo": 8, "w1": 1, "w3": 1, "w2": 8, "head": 1},
                          # the verify step's qsm head reads the decode head's weights in place, and the driver takes the
                          # argmax on the host over the true vocabulary: N = 151,936 (the decode head's W / S / B)
                          qsm_head_vocab=True)}
ARCH = dict(ARCHS["internlm2"], name="internlm2")


def set_arch(name):
    global EPS
    ARCH.clear()
    ARCH.update(ARCHS[name], name=name)
    if name.startswith("qwen3"):                            # the QK-norm references read g17qwen3's shapes
        import g17qwen3 as _Q3
        _Q3.configure("qwen3-8b" if name == "qwen3_8b" else "qwen3-0.6b")
    EPS = ARCH["eps"]
    Spec.norm_eps = EPS
    for bits, ks in _KSPLIT0.items():
        KSPLIT[bits] = dict(ks, **ARCH.get("ksplit", {}).get(bits, {}))
    # the prefill (MM 25.188): the quantized GEMMs' shapes and the row kernels' widths
    import g17qmm as _QMM
    # the prefill's FFN width: Qwen3's 3,072 zero-padded to 4,096 (the generic GEMM's launch grid is a power of two)
    d, hd, qkv, f = ARCH["d"], ARCH["hd"], ARCH["qkv"], ARCH.get("ffn_prefill", ARCH["ffn"])
    _QMM.ROLES.clear()
    _QMM.ROLES.update({"qkv": (qkv, d), "wo": (d, hd), "w1": (f, d), "w3": (f, d), "w2": (d, f)})
    ROWS_N.clear()
    ROWS_N.update(swiglu=f, residual=d, fold=d)
    # the batched decode (MM 25.189): the qsm roles at the model's shapes (the head at the padded vocabulary: its pad
    # logits are overwritten with the most negative float before the argmax), the psum widths
    fd = ARCH["ffn"]
    QSM_ROLES.clear()
    QSM_ROLES.update({"qkv": (qkv, d, 2, None), "wo": (d, hd, 4, None), "w1": (fd, d, 1, 0), "w3": (fd, d, 1, 1),
                      "w2": (d, fd, 4, None), "head": (ARCH["vocab"] if ARCH.get("qsm_head_vocab") else ARCH["vocab_pad"], d, 1, None)})
# split-K per projection and width (MM 25.141.15): rows 1, S simdgroups per threadgroup
KSPLIT = {4: dict(qkv=2, wo_res1=2, w2_res2=4, ffn=2), 8: dict(qkv=4, wo_res1=4, w2_res2=4, ffn=2)}
_KSPLIT0 = {k: dict(v) for k, v in KSPLIT.items()}
# the unsplit v2 rows per simdgroup (the q4/q8 v2 packages)
V2_ROWS = dict(qkv=4, wo_res1=4, w2_res2=4, ffn=2)
ROLE_SEED = dict(qkv=13, wo_res1=17, w2_res2=19, ffn=23, head=29)
# lm_head variants per width: {} the delivered form; split-K + lean (MM 25.144.7). q8's "x32" head reads fp32 x from
# the out32 final norm, which the fp32-x lean/ptr path needs.
# ks 2 + lean without ptr (q8 x32) is the batched head's one-sequence twin: a batched graph's single-sequence check
# runs it (g17modelbuild.single_of drops "batch").
HEAD_VARIANTS = {4: ({}, dict(ks=2, lean=True), dict(ks=2, lean=True, ptr=True), dict(ks=2, lean=True, ptr=True, argmax=True)),
                 8: ({}, dict(ks=4, lean=True), dict(ks=2, lean=True, x32=True), dict(ks=4, lean=True, ptr=True, x32=True),
                     dict(ks=2, lean=True, ptr=True, x32=True), dict(ks=2, lean=True, ptr=True, x32=True, argmax=True),
                     dict(ks=2, lean=True, ptr=True, x32=True, w4=True))}


class Spec:
    norm_eps = EPS
    storage = "half"


# THE INDEX CONTRACT with the graph (Piece A's g17q4graph selects by (kind, bits, role, variant, cap)): every entry
# carries COMMON, plus its kind's layout offsets and the fields the earlier hand-built index files carried.
COMMON = ("kind", "bits", "role", "variant", "cap", "bundle", "name", "threadgroups", "threads_per_group", "base",
          "slot_map", "program_sha256", "recipe")
SCHEMA = {
    "qmv": COMMON + ("op", "x_dtype", "x_offset", "X", "W", "S", "B", "OUT", "RES", "out_dtype"),
    "lm_head": COMMON + ("op", "x_dtype", "x_offset", "X", "W", "S", "B", "OUT", "out_dtype"),
    # the multi-vector qmv (MM 25.144.3): the qmv fields plus the batch and the vector-major strides
    "qmv_batch": COMMON + ("op", "x_dtype", "x_offset", "X", "W", "S", "B", "OUT", "RES", "out_dtype", "batch", "x_stride",
                           "out_stride", "h_stride", "res_stride"),
    "norm": COMMON + ("op", "in_dtype", "out_dtype", "X", "G", "OUT"),
    "attn": COMMON + ("ATTN", "attn_bytes", "region3_bytes", "COST", "SINT", "rope_bytes", "KOFF", "VOFF"),
    "gen_argmax": COMMON + ("GEN", "LOG", "R_X16", "region_bytes", "PAIRS"),
    "gen_step": COMMON + ("GEN", "LOG", "R_X16", "region_bytes", "PAIRS"),
    # MM 25.205: the speculative verify step's 16-row argmax pass 1 (the driver reduces each row's G pairs)
    "argmax_rows": COMMON + ("V", "G", "C", "pairs_bytes"),
    "qmvw": COMMON + ("N", "K", "nb", "W", "S", "B", "out_layout"),
    # M2 (MM 25.144.2): the prefill append and attention, sharing decode's cache (KOFF/VOFF) and rope region (COST/SINT)
    "prefill_attn": COMMON + ("mmax", "threadgroups_per_row", "QKVROW", "Q16", "PATTN", "KOFF", "VOFF", "COST", "SINT",
                              "rope_bytes", "prefill_region3_bytes", "region3_bytes"),
    # prefill elementwise over M rows (g17rows, MM 25.144.3)
    "swiglu_rows": COMMON + ("rows", "N", "unroll", "GATE", "UP", "ACT"),
    # MM 25.172: the batched decode on the tensor units. qsm: slot 1 (physical) the q4 weight block (W, S, B byte
    # offsets inside it), slot 2 the batch's x fp16 [16][K] rows, slot 3 y fp32 [16][N] or the sk partials [sk][16][N];
    # psum: slot 1 the partials, slot 2 the residual input at IN, slot 3 the output at OUT
    "qsm": COMMON + ("N", "K", "sk", "W", "S", "B", "out_layout"),
    "psum": COMMON + ("N", "sk", "mode", "rows", "pstride", "IN", "OUT", "out_layout"),
    "residual_rows": COMMON + ("rows", "N", "unroll", "H", "H16"),
    "fold_rows": COMMON + ("rows", "N", "unroll", "P0", "P1", "X"),
    # M1 (MM 25.144.1): the quantized prefill GEMM as two dispatches. qmm_dequant writes W16[K][N] fp16 into its C
    # buffer (slot 3); qmm's B buffer (slot 2) IS that W16, its A (slot 1) is x fp16 [M][K] rows, its C (slot 3)
    # y fp32 [M][N] rows (ldc = N), or for split_k 2 the two K-half partials stacked [2M][N], folded p0 + p1.
    "qmm_dequant": COMMON + ("N", "K", "W", "S", "B", "OUT", "out_layout"),
    "qmm": COMMON + ("M", "N", "K", "split_k", "A_layout", "B_layout", "out_layout"),
    # MM 25.144.12: w3's qmm with the SwiGLU in its tail (g17swigluqmm): slot 1 x fp16 [M][K], slot 2 w3's W16 [K][N],
    # slot 3 one region: U fp32 [M][N] at U_OFF (scratch), w1's gate G fp32 [M][N] at G_OFF, act fp16 [M][N] at ACT_OFF
    "qmm_swiglu": COMMON + ("M", "N", "K", "G_OFF", "U_OFF", "ACT_OFF", "out_layout"),
    # the tensor-unit route (MM 25.144.2): one program per bucket (M, p0 block)
    "prefill_mma": COMMON + ("M", "p0", "NB", "QKVROW", "Q16", "QT", "VB", "SCR", "PATTN", "KOFF", "VOFF", "COST", "SINT",
                             "rope_bytes", "mma_region3_bytes", "region3_bytes", "preconditions", "contract_notes"),
}


def validate(entry):
    """The entry's missing contract fields (empty when it is complete)."""
    return [k for k in SCHEMA[entry["kind"]] if k not in entry]


# ------------------------------------------------------------------------------------------------ layouts
def qmv_layout(bits, role, variant):
    """The layout of one projection: variant {} = the v2 form, {"ks": S} = split-K, {"ks": S, "lean": True} = split-K
    with hi16 scale loads (the form the graph was validated with), {"ks": S, "lean": True, "ptr": True} = that plus
    loop-carried addresses. Returns (layout, flags)."""
    ks = variant.get("ks")
    rows = 1 if ks else V2_ROWS[role]
    extra = dict(sgs=ks, ksplit=True, coop=True) if ks else {}
    flags = []
    if variant.get("lean"):
        extra["hi16_scales"] = True
        flags.append("hi16_scales")
    if role == "ffn":
        lay = dict(Q.qmv_swiglu_layout(ARCH["ffn"], ARCH["d"], bits, rows), **BASE, act32=True, **extra)
    elif role == "qkv":
        lay = dict(Q.case(ARCH["qkv"], ARCH["d"], bits, rows, nocarrier=True)[0], **BASE, **extra)
    elif role == "wo_res1":
        lay = Q.with_residual(dict(Q.case(ARCH["d"], ARCH["hd"], bits, rows, nocarrier=True)[0], **BASE, **extra), "add16")
    elif role == "w2_res2":
        lay = Q.with_residual(dict(Q.case(ARCH["d"], ARCH["ffn"], bits, rows, nocarrier=True)[0],
                                   **dict(BASE, wpt=4 if bits == 8 else 2), **extra), "add32_to16")
    else:
        raise ValueError("unknown projection role %r" % role)
    if variant.get("nodeq"):
        lay = dict(lay, ablate_nodeq=True)         # MM 25.200: the timing ablation (loads kept, no dequant arithmetic)
        flags.append("ablate_nodeq")
    if variant.get("ptr"):
        lay = dict(lay, ptr_addr=True)             # loop-carried addresses (MM 25.141.18); refused outside its shape
        flags.append("ptr_addr")
    if variant.get("chain"):
        # the r1 loop without its per-row extras (MM 25.144.4): position chains against raw x, fused epilogue, op428
        # pool masks, and16 read in place after the load's first waited consumer. Its own fp32 order.
        ch = dict(chains=True, epi_fma=True, and16_direct=True, pool_masks=True)
        lay = dict(lay, interleave=False, hoist_consts=False)      # the timed form (55 -> 51 registers)
        lay = dict(lay, **ch)
        flags += sorted(ch)
    return lay, flags


def head_layout(bits, variant=None):
    """The lm_head (92,544 x 2048). variant {} = the delivered form: q4 the xvec fp32-x form (wpt 2, rows 4), q8 the
    fp16-x (x16) form, wpt 2. {"ks": S, ...} = split-K, rows 1, S simdgroups (the cooperative class), with "lean"
    (hi16 scale loads) and "ptr" (loop-carried addresses, fp32 x only). "x32" gives q8 an fp32-x (xvec) head, which
    reads the out32 final norm (MM 25.144.7). Returns (layout, variant)."""
    variant = dict(variant or {})
    ks = variant.get("ks")
    rows = 1 if ks else 4
    fp32_x = bits == 4 or variant.get("x32")
    if fp32_x:
        lay = dict(Q.case(ARCH["vocab"], ARCH["d"], bits, rows, nocarrier=True)[0], **dict(BASE, wpt=4 if variant.get("w4") else 2))
    else:
        lay = dict(Q.case(92544, 2048, 8, rows, nocarrier=True)[0], interleave=True, lean=True, coalesced=True, a16=True,
                   wpt=2, x16=True)
    if not variant:
        return lay, ({"xvec": True} if bits == 4 else {"x16": True})
    if ks:
        lay.update(sgs=ks, ksplit=True, coop=True)
    if variant.get("lean"):
        lay["hi16_scales"] = True
    if variant.get("ptr"):
        if not fp32_x:
            raise ValueError("lm_head ptr: loop-carried addresses need the fp32-x (xvec) head")
        lay["ptr_addr"] = True
    if variant.get("argmax"):
        lay = Q.with_argmax_chunks(lay)          # the head writes g17gen's pass-1 pairs itself (MM 25.144.7)
    if variant.get("batch"):
        # THE BATCHED HEAD (MM 25.144.7): one dispatch, each weight word read once for nb sequences (M3's multi-vector
        # qmv, build_qmv2_batch). Vector-major: x_b fp32 at X + 4 K b (the batched final norm's rows), logits_b at
        # OUT + 4 V b (where the batched argmax reads them). fp32 x only, and no ptr_addr (outside the batch scope).
        if not fp32_x or variant.get("ptr") or variant.get("argmax"):
            raise ValueError("lm_head batch: the fp32-x split-K lean head without ptr / argmax")
        if variant.get("pass"):
            lay["batch_pass"] = variant["pass"]
        lay = Q.with_batch(lay, variant["batch"])
    return lay, variant


def norm_layout(in_dtype, out32, seed):
    lay = O.rmsnorm_loop_layout(ARCH["d"], in_dtype, groups=32, unroll=16 if in_dtype == "half" else 8, hoist=True)
    if out32:
        lay["out32"] = True
    lay["rs_seed" if seed else "rs_once"] = True
    return lay


def attn_layout(cap, flags=""):
    """The delivered wide butterfly attention; `flags` ("keyblock=2+tgsplit=4", ...) applies the M5 long-context forms
    (g17attn.with_keyblock / with_nsum / with_tgsplit, MM 25.144.5) in order. "" is the base form."""
    lay = A.with_attn32(A.with_bfly_merge(A.with_wide(A.with_fused_merge(A.with_rope_tables(
        A.attn_rope_layout(cap=cap, heads=ARCH.get("heads", 16), kv_heads=ARCH.get("kv_heads", 8)))))))
    for f in (x for x in flags.split("+") if x):
        k, _, v = f.partition("=")
        lay = getattr(A, "with_" + k)(lay, *([int(v)] if v else []))
    return lay


def attn_variant(flags):
    """The index entry's variant dict for an attention flag string."""
    v = dict(widebf=True, attn32=True)
    for f in (x for x in flags.split("+") if x):
        k, _, n = f.partition("=")
        v[k] = int(n) if n else True
    return v


BUILDERS = {
    "g17qmv.build_qmv2": lambda lay: Q.build_qmv2(lay),
    "g17decodeops.build_rmsnorm_wide": lambda lay: O.build_rmsnorm_wide(lay, EPS),
    "g17attn.build_attn_split_rope": lambda lay: A.build_attn_split_rope(lay),
    "g17gen.build_pass1": lambda lay: G.build_pass1(lay),
    "g17gen.build_gen": lambda lay: G.build_gen(lay),
    "g17prefillattn.build_prefill_append": lambda lay: _prefill().build_prefill_append(lay),
    "g17prefillattn.build_prefill_attn": lambda lay: _prefill().build_prefill_attn(lay),
    "g17gen.build_gen_batch": lambda lay: G.build_gen_batch(lay),
    "g17rows.build_rows": lambda lay: __import__("g17rows").build_rows(lay),
    "g17qmm.build_dequant": lambda lay: __import__("g17qmm").build_dequant(lay),
    "g17tensorcommonruntime.gemm_generic": lambda spec: __import__("g17tensorcommonruntime").build_generic_program(
        __import__("g17tensorcommonruntime").generic_spec(spec)),
    "g17prefillmma.build_mma": lambda lay: _mma().build_mma(lay),
    "g17prefillmma.build_mma_rego": lambda lay: _mma().build_mma_rego(lay),
    "g17swigluqmm.build": lambda lay: __import__("g17swigluqmm").build(lay, hold=lay.get("hold")),
}


def _skip_supported():
    """True when this checkout's cc accepts a capped runtime trip count around tensor bodies (the causal skip)."""
    MM = _mma()
    try:
        MM.build_mma(dict(MM.mma_layout(272, 128, 0), runtime_trips=True))
        return True
    except Exception as e:  # noqa: BLE001 - cc refuses by name
        if "runtime trip count" in str(e):
            return False
        raise


def _rego_supported():
    """True when this checkout's cc compiles the register-O route (M8's register accumulator, between-body register
    sharing and the simdgroup split: PR #265)."""
    MM = _mma()
    try:
        MM.build_mma_rego(dict(MM.mma_layout(272, 128, 0, rego=True), runtime_trips=True))
        return True
    except Exception as e:  # noqa: BLE001 - cc refuses by name
        if type(e).__name__ in ("Unsupported", "IRError", "ValueError"):
            return False
        raise


def _rego_options():
    """The register-O code-size options this checkout's IR supports (M8, PR #265): all of fold, scale, hoist, row2 and
    holdk when tensor_acc_scale and hoist_prologue exist, fold alone when only fold_offsets does, else none."""
    import inspect
    from agxforge.g17 import ir
    params = inspect.signature(ir.Builder.tensor_matmul).parameters
    if "hoist_prologue" in params and hasattr(ir.Builder, "tensor_acc_scale"):
        return dict(fold=True, scale=True, hoist=True, row2=True, holdk=True)
    return dict(fold=True) if "fold_offsets" in params else {}


def _mma():
    import g17prefillmma
    return g17prefillmma


def _prefill():
    import g17prefillattn
    return g17prefillattn


def jsonable(lay):
    return json.loads(json.dumps(lay))


# ------------------------------------------------------------------------------------------------ bundles
def _twin():
    from agxforge.g17 import cc, ir
    fn, b, a, bb, c = O._function()
    O._carrier(b, a, bb, c, dict(groups=256))
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def _pad(x, n):
    return bytes(x) + bytes(n - len(x))


def author(d, prog, a, b, c, lay, template=None):
    """A bundle: the manifest of a carrier twin at a transport holding the inputs (only its entry name and buffer
    sizes are read), the three inputs, and THIS program's image. `template` (a qmv layout) authors the manifest
    from the carrier form of that layout instead, for buffers past the twin transport (the lm_head)."""
    from agxforge.g17 import scanlink
    d = Path(d)
    if d.exists():
        shutil.rmtree(d)
    tmp = d.parent / (d.name + "_t")
    if tmp.exists():
        shutil.rmtree(tmp)
    if template is None:
        t = Q.big_transport(256, len(a), len(b), len(c))
        a, b, c = _pad(a, t["a_bytes"]), _pad(b, t["b_bytes"]), _pad(c, t["c_bytes"])
        O.author(tmp, dict(t, groups=256, op="deliver"), _twin(), a, b, c, extra={"arm": d.name})
    else:
        tw = dict(template, res=None, swiglu=False, wpt=1, vload=False, sgs=1, xvec=False, hoist_consts=False, x16=True,
                  act32=False, ksplit=False, coop=False, hi16_scales=False, ptr_addr=False, argmax_chunks=False)
        for k in ("chains", "epi_fma", "pool_masks", "and16_direct"):   # only where set, so other templates are unchanged
            if k in tw:
                tw[k] = False
        tw.pop("nocarrier", None)
        tw["groups"] = min(template["groups"], 256)
        O.author(tmp, tw, Q.build_qmv2(dict(tw)), a, b, bytes(c)[:tw["c_bytes"]], extra={"arm": d.name})
    shutil.copytree(tmp, d)
    shutil.rmtree(tmp)
    for name, data in (("a.f16", a), ("b.f16", b), ("c.f32", c)):
        (d / name).write_bytes(bytes(data))
    img = scanlink.author(prog)
    for name, data in (("scan.arc.metallib", img.archive), ("scan.lib.metallib", img.library), ("scan.o", img.object),
                       ("program.bin", prog.code)):
        (d / name).write_bytes(data)
    (d / "decodeop.json").write_text(json.dumps(dict(layout=jsonable(lay), arm=d.name), indent=1, sort_keys=True) + "\n")
    return d


def runner():
    """tools/g17bundlerun, compiled once per source hash into ~/.cache/agxforge."""
    src = ROOT / "tools" / "g17bundlerun.m"
    h = hashlib.sha256(src.read_bytes() + (ROOT / "tools" / "g17gpulock.h").read_bytes()).hexdigest()[:16]
    cache = Path.home() / ".cache" / "agxforge"
    cache.mkdir(parents=True, exist_ok=True)
    exe = cache / ("g17bundlerun-" + h)
    if not exe.exists():
        subprocess.run(["clang", "-fobjc-arc", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "tools"),
                        "-framework", "Foundation", "-framework", "Metal", "-o", str(exe) + ".tmp", str(src)], check=True)
        os.replace(str(exe) + ".tmp", exe)
    return exe


# `build --emulate` (MM 25.145): every dispatch goes to tools/g17emu.py on the CPU instead of the GPU. A program g17emu
# refuses comes back as None and is reported as refused, never as a pass. Nothing is delivered in this mode.
EMULATE = None


def _emulate(jobs):
    """--emulate: each job on g17emu, EMULATE["workers"] at a time; {tag: output bytes, or None if refused}."""
    import concurrent.futures
    import g17emu as EMU
    outs = {}

    def done(tag, fut):
        try:
            outs[tag], admitted, secs = fut.result()
            how = "ran"
            if admitted:                          # passes only with whole-program-only semantics (tier wp)
                EMULATE["admitted"][tag] = admitted
                how = "ran (wp)"
        except EMU.Refused as e:
            EMULATE["refused"][tag] = str(e)
            outs[tag], secs = None, 0
            how = "refused: %s" % str(e)[:100]
        except Exception as e:                    # an emulator defect: reported as its own cause, never as a pass
            EMULATE["refused"][tag] = "emulator error: %s: %s" % (type(e).__name__, e)
            outs[tag], secs = None, 0
            how = EMULATE["refused"][tag][:120]
        print("emulate %s: %s, %.0f s" % (tag, how, secs), file=sys.stderr, flush=True)

    args = [(j["tag"], (str(j["dir"]), j["threads"], j["group"], j.get("base", 1), "wp")) for j in jobs]
    workers = EMULATE.get("workers", 1)
    if workers <= 1 or len(jobs) == 1:
        for tag, a in args:
            fut = concurrent.futures.Future()
            try:
                fut.set_result(EMU.run_job(*a))
            except EMU.Refused as e:
                fut.set_exception(e)
            done(tag, fut)
        return outs
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        futs = {pool.submit(EMU.run_job, *a): tag for tag, a in args}
        for fut in concurrent.futures.as_completed(futs):
            done(futs[fut], fut)
    return outs


def dispatch(jobs, work):
    """Run every job's bundle once in one runner process; return {tag: output bytes}."""
    if EMULATE is not None:
        return _emulate(jobs)
    plan = work / "plan.json"
    out = work / "out"
    out.mkdir(exist_ok=True)
    json.dump(dict(configs=[dict(tag=j["tag"], bundle=str(j["dir"]), threads=j["threads"], group=j["group"],
                                 base=j["base"], rounds=j.get("rounds", 1)) for j in jobs]), open(plan, "w"))
    r = subprocess.run([str(runner()), str(plan), str(out)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("g17bundlerun failed (%d): %s" % (r.returncode, r.stderr[-2000:]))
    return {j["tag"]: (out / (j["tag"] + ".out")).read_bytes() for j in jobs}


# ------------------------------------------------------------------------------------------------ kinds
def _qmv_io(lay, role, bits, seed):
    """(a, b, c, offset, count, dtype, want words) for one projection, output region sentinel-filled."""
    rng = np.random.default_rng(seed)
    d = ARCH["d"]
    N, K = {"qkv": (ARCH["qkv"], d), "wo_res1": (d, ARCH["hd"]), "w2_res2": (d, ARCH["ffn"]), "ffn": (2 * ARCH["ffn"], d),
            "head": (ARCH["vocab"], d)}[role]
    Wf = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)
    x = np.asarray(rng.standard_normal(K), np.float16).astype(np.float32)
    packed, s16, b16, q = Q.quantize(Wf, bits=bits)
    ref = Q.qmv2_ksplit_reference if lay.get("ksplit") else Q.qmv2_reference_fast
    extra = None
    if role == "ffn":
        want16 = Q.qmv_swiglu_reference(lay, x, q, s16, b16)
        wv = np.asarray(want16, np.float16).astype(np.float32).view(np.uint32)
        off, n, dt, want = lay["OUT"], ARCH["ffn"], "<u4", np.zeros(1, np.float16)
    elif role == "wo_res1":
        xr = rng.standard_normal(d).astype(np.float16)
        want = Q.residual_reference(lay, ref(lay, x, q, s16, b16), x16=xr)
        wv, off, n, dt, extra = np.asarray(want, np.float32).view(np.uint32), lay["OUT"], d, "<u4", (lay["RES"], xr)
    elif role == "w2_res2":
        h = (rng.standard_normal(d) * 3).astype(np.float32)
        want = Q.residual_reference(lay, ref(lay, x, q, s16, b16), h32=h)
        wv, off, n, dt, extra = want.view(np.uint16), lay["RES"], d, "<u2", (0, h.astype("<f4"))
    else:
        want = ref(lay, x, q, s16, b16)
        wv, off, n, dt = np.asarray(want, np.float32).view(np.uint32), lay["OUT"], N if role == "head" else ARCH["qkv"], "<u4"
    a, b, c, _ = Q.qmv_io(lay, x, packed, s16, b16, want)
    c = bytearray(c)
    if extra is not None:
        O._place(c, extra[0], extra[1])
    nb = n * (2 if dt == "<u2" else 4)
    c[off:off + nb] = SENT * nb
    return a, b, bytes(c), off, n, dt, wv


def build_qmv(work, bits, role, variant, head=False):
    if head:
        lay, variant = head_layout(bits, variant)
        flags = sorted(k for k in ("hi16_scales", "ptr_addr", "ksplit", "xvec", "x16") if lay.get(k))
    else:
        lay, flags = qmv_layout(bits, role, variant)
    prog = Q.build_qmv2(lay)
    a, b, c, off, n, dt, wv = _qmv_io(lay, role, bits, ROLE_SEED[role] + bits)
    ks = variant.get("ks")
    if head:
        # the delivered head names are kept; split heads name their variant (MM 25.144.7)
        name = ("qmv_q%d_v2_lmhead" % bits if bits == 4 else "qmv_x16_q8_lmhead") if not ks else \
            "qmv_q%d_lmhead%s_ks%d%s%s%s%s" % (bits, "_x32" if variant.get("x32") else "", ks,
                                                "_w4" if variant.get("w4") else "",
                                                "_lean" if variant.get("lean") else "", "_ptr" if variant.get("ptr") else "",
                                                "_argmax" if variant.get("argmax") else "")
    else:
        name = "qmv_q%d_v2_%s%s%s%s%s" % (bits, role, "_ks%d" % ks if ks else "", "_lean" if variant.get("lean") else "",
                                          "_ptr" if variant.get("ptr") else "", "_chain" if variant.get("chain") else "")
    fused = bool(lay.get("argmax_chunks"))
    if fused:
        # the pairs region sentinel-filled; the counters start at 0 (the graph zeroes them per sequence)
        cb = bytearray(c)
        cb[lay["PAIRS"]:lay["PAIRS"] + 8 * lay["argmax_G"]] = SENT * (8 * lay["argmax_G"])
        c = bytes(cb)
        lv = wv.view(np.float32) + np.float32(0.0)
        want_pairs = np.zeros((lay["argmax_G"], 2), np.float32)
        for t in range(lay["argmax_G"]):
            j = int(np.argmax(lv[384 * t:384 * (t + 1)]))
            want_pairs[t] = (lv[384 * t + j], np.float32(384 * t + j))
    d = author(work / name, prog, a, b, c, lay, template=lay)
    tpg = 32 * (ks or 1)
    entry = dict(kind="lm_head" if head else "qmv", bits=bits, role=role, variant=variant, op=role,
                 threadgroups=lay["groups"], threads_per_group=tpg, base=0 if ks else 1,
                 slot_map=dict(c_written=0, x=1, weights=2) if ks else dict(x=1, weights=2, c_written=3),
                 x_dtype="fp16" if lay.get("x16") else "fp32", x_offset=lay["X"], X=lay["X"], W=lay["W"], S=lay["S"],
                 B=lay["B"], OUT=lay["OUT"], RES=lay.get("RES"),
                 out_dtype=("fp32 act (fp16-rounded)" if role == "ffn" else ("fp16 at RES" if role == "w2_res2" else "fp32")),
                 recipe=dict(builder="g17qmv.build_qmv2", flags=flags, layout=jsonable(lay)))
    if ks:
        entry["reduction"] = "split-K: %d contiguous K slices, each qmv2 order, summed in slice order in fp32" % ks
    if fused:
        entry.update(CNT=lay["CNT"], PAIRS=lay["PAIRS"], argmax_G=lay["argmax_G"], argmax_C=lay["argmax_C"],
                     argmax="g17gen pass 1's (value, index) pairs at PAIRS; per-chunk counters at CNT grow 384 per "
                            "dispatch and must start at 0 per sequence (valid for 32,768 dispatches)")

    def check(out):
        got = np.frombuffer(out, dt, n, off)
        bad = int((got != wv).sum())
        if fused:
            pr = np.frombuffer(out, "<u4", 2 * lay["argmax_G"], lay["PAIRS"])
            bad += int((pr != want_pairs.reshape(-1).view(np.uint32)).sum())
            bad += int((np.frombuffer(out, "<u4", lay["argmax_G"], lay["CNT"]) != 384).sum())
        return bad
    return [dict(name=name, dir=d, prog=prog, entry=entry, threads=tpg * lay["groups"], group=tpg, base=entry["base"],
                 check=check)]


BATCHES = (2, 4, 8)
# the batched wide norm and the rows kernels serving prefill (kinds norm_rows, swiglu/residual/fold_rows). 256 is
# the chunk M1's 2 x 2 GEMM forms exist at (25.144.1); without it a 256-row prefill could not be assembled at all
PREFILL_ROWS = (128, 256, 512, 1024)


def qmv_batch_layout(bits, role, nb, pv=None, dq=False, wide=False, accsplit=1):
    """The multi-vector qmv (MM 25.144.3): the lean split-K form of the role with with_batch(nb). Constants are hoisted
    unless the batched program then runs out of registers (nb = 8 for the FFN and q4 w2), when they stay in the loop.
    pv: vectors per pass (batch_pass), the weight stream read nb / pv times with only pv vectors live at once."""
    lay, flags = qmv_layout(bits, role, dict(ks=KSPLIT[bits][role], lean=True))
    if wide:
        # MLX qmv_wide (MM 25.144.9): 8 K-lanes per row, 4 rows per simdgroup, 2 simdgroups; same X/W/S/B/OUT and
        # vector-major strides as the split-K form, so it drops into the graph. Plain qmv only for now.
        if role != "qkv":
            raise ValueError("qmv wide: qkv only for now (no residual/swiglu wide kernel yet)")
        lay = Q.with_batch(dict(lay), nb) if nb > 1 else dict(lay, batch=1)
        lay = dict(lay, wide=True, klanes=8, rows_per_sg=4, wide_sgs=1, wide_pass=min(nb, pv or 4))
        rows_tg = lay["rows_per_sg"] * lay["wide_sgs"]
        lay["groups"] = (lay["Nout"] // rows_tg) * (nb // lay["wide_pass"])
        return lay, flags + ["wide"], Q.build_qmv2(lay)
    if pv:
        lay = dict(lay, batch_pass=pv)
        flags = flags + ["batch_pass"]
    if dq:
        # dequantize once per weight, reused by every vector of the pass as a plain fp32 dot (its own order)
        lay = dict(lay, dequant_once=True)
        flags = flags + ["dequant_once"]
    if accsplit > 1:
        # split each (vector, row) accumulator into `accsplit` independent partials: 1/accsplit the critical chain
        lay = dict(lay, acc_split=accsplit)
        flags = flags + ["acc_split%d" % accsplit]
    for hoist in (True, False):
        # nb 1 (dq only): the single-vector dequantize-once kernel, the single layout
        cand = Q.with_batch(dict(lay, hoist_consts=hoist), nb) if nb > 1 else dict(lay, hoist_consts=hoist)
        try:
            prog = Q.build_qmv2(cand)
        except Exception:
            if not hoist:
                raise
            continue
        return cand, flags + ["batch"] + ([] if hoist else ["no_hoist"]), prog


def _qmv_batch_io(lay, role, bits, nb, seed):
    """Buffers for nb vectors, every vector's output region sentinel-filled; want words concatenated per vector."""
    rng = np.random.default_rng(seed)
    N, K = {"qkv": (4096, 2048), "wo_res1": (2048, 2048), "w2_res2": (2048, 8192), "ffn": (16384, 2048)}[role]
    Wf = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)
    xs = np.asarray(rng.standard_normal((nb, K)), np.float16).astype(np.float32)
    packed, s16, b16, q = Q.quantize(Wf, bits=bits)
    single = dict(lay)
    single.pop("batch", None)
    ref = (Q.qmv_wide_reference if lay.get("wide") else Q.qmv_dq_reference if lay.get("dequant_once")
           else Q.qmv2_ksplit_reference)
    a, b, c = (bytearray(x) for x in O._buffers(lay))
    O._place(b, lay["W"], packed.astype("<u4"))
    O._place(b, lay["S"], s16.astype("<u2"))
    O._place(b, lay["B"], b16.astype("<u2"))
    for v in range(nb):
        O._place(a, lay["X"] + 4 * K * v, xs[v].astype("<f4"))
    wants = []
    if role == "ffn":
        F = lay["ffn"]
        for v in range(nb):
            wants.append(np.asarray(Q.qmv_swiglu_reference(single, xs[v], q, s16, b16), np.float16).astype(np.float32).view(np.uint32))
        off, n, dt, stride = lay["OUT"], F, "<u4", 4 * F
    elif role == "wo_res1":
        for v in range(nb):
            xr = rng.standard_normal(N).astype(np.float16)
            O._place(c, lay["RES"] + 2 * N * v, xr)
            wants.append(np.asarray(Q.residual_reference(single, ref(single, xs[v], q, s16, b16), x16=xr), np.float32).view(np.uint32))
        off, n, dt, stride = lay["OUT"], N, "<u4", 4 * N
    elif role == "w2_res2":
        for v in range(nb):
            h = (rng.standard_normal(N) * 3).astype(np.float32)
            O._place(c, lay["OUT"] + 4 * N * v, h.astype("<f4"))
            wants.append(Q.residual_reference(single, ref(single, xs[v], q, s16, b16), h32=h).view(np.uint16))
        off, n, dt, stride = lay["RES"], N, "<u2", 2 * N
    else:
        for v in range(nb):
            wants.append(np.asarray(ref(single, xs[v], q, s16, b16), np.float32).view(np.uint32))
        off, n, dt, stride = lay["OUT"], N, "<u4", 4 * N
    esz = 2 if dt == "<u2" else 4
    for v in range(nb):
        c[off + stride * v:off + stride * v + n * esz] = SENT * (n * esz)
    return bytes(a), bytes(b), bytes(c), off, n, dt, stride, wants


def build_qmv_batch(work, bits, role, nb, pv=None, dq=False, wide=False, accsplit=1):
    """One multi-vector projection bundle (kind "qmv_batch"), verified per vector against the single-vector reference.
    pv: vectors per pass (variant "pass")."""
    lay, flags, prog = qmv_batch_layout(bits, role, nb, pv, dq, wide, accsplit)
    S = KSPLIT[bits][role]
    a, b, c, off, n, dt, stride, wants = _qmv_batch_io(lay, role, bits, nb, ROLE_SEED[role] + bits + 100 * nb)
    name = "qmv_q%d_v2_%s_ks%d_lean_b%d%s%s%s%s" % (bits, role, S, nb, "_p%d" % pv if pv else "",
                                                   "_dq" if dq else "", "_wide" if wide else "",
                                                   "_as%d" % accsplit if accsplit > 1 else "")
    # the manifest's carrier template is the single-vector form (its buffer sizes are the batched ones)
    d = author(work / name, prog, a, b, c, lay, template={k: val for k, val in lay.items() if k not in ("batch", "batch_pass", "dequant_once", "wide", "klanes", "rows_per_sg", "wide_sgs", "wide_pass")})
    tpg = 32 * lay.get('wide_sgs', 1) if wide else 32 * S
    N = lay["Nout"]
    variant = dict(ks=S, lean=True, **({"batch": nb} if nb > 1 else {}), **({"pass": pv} if pv else {}),
                   **({"dq": True} if dq else {}), **({"wide": True} if wide else {}),
                   **({"acc_split": accsplit} if accsplit > 1 else {}))
    entry = dict(kind="qmv", bits=bits, role=role, variant=variant, op=role, batch=nb,
                 threadgroups=lay["groups"], threads_per_group=tpg, base=0, slot_map=dict(c_written=0, x=1, weights=2),
                 x_dtype="fp32", x_offset=lay["X"], X=lay["X"], W=lay["W"], S=lay["S"], B=lay["B"], OUT=lay["OUT"],
                 RES=lay.get("RES"), x_stride=4 * lay["Kq"],
                 out_stride=stride, out_dtype=("fp32 act (fp16-rounded)" if role == "ffn" else ("fp16 at RES" if role == "w2_res2" else "fp32")),
                 h_stride=4 * N if lay.get("res") else None, res_stride=2 * N if lay.get("res") else None,
                 layout_note="vector-major: sequence b sees the single-vector layout shifted by its stride",
                 reduction=("split-K: %d contiguous K slices; per lane w = fma32(q, s, b) then acc = fma32(w, x, acc) from 0, "
                            "butterfly 1,8,2,4,16, slices summed in order in fp32; per vector (g17qmv.qmv_dq_reference)" % S if dq else
                            "split-K: %d contiguous K slices, each qmv2 order, summed in slice order in fp32; per vector" % S),
                 recipe=dict(builder="g17qmv.build_qmv2", flags=flags, layout=jsonable(lay)))

    def check(out):
        bad = 0
        for v in range(nb):
            got = np.frombuffer(out, dt, n, off + stride * v)
            bad += int((got != wants[v]).sum())
        return bad
    return [dict(name=name, dir=d, prog=prog, entry=entry, threads=tpg * lay["groups"], group=tpg, base=0, check=check)]


def build_head_batch(work, bits, nb, pv=None, S=2):
    """The batched lm_head (kind "lm_head", variant {ks, lean, [x32,] batch, [pass]}): one dispatch for nb sequences,
    verified per sequence against the single-row split-K head reference over a 0x7f sentinel (MM 25.144.7). Constants
    are hoisted unless that runs out of registers."""
    variant = dict(ks=S, lean=True, batch=nb, **({"x32": True} if bits == 8 else {}), **({"pass": pv} if pv else {}))
    lay, _ = head_layout(bits, variant)
    prog = None
    for hoist in (True, False):
        try:
            cand = dict(lay, hoist_consts=hoist)
            prog = Q.build_qmv2(cand)
            lay = cand
            break
        except Exception:
            if not hoist:
                raise
    N, K = 92544, 2048
    rng = np.random.default_rng(ROLE_SEED["head"] + bits + 100 * nb)
    Wf = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)
    xs = np.asarray(rng.standard_normal((nb, K)), np.float16).astype(np.float32)
    packed, s16, b16, q = Q.quantize(Wf, bits=bits)
    single = {k: v for k, v in lay.items() if k not in ("batch", "batch_pass")}
    a, b, c = (bytearray(x) for x in O._buffers(lay))
    O._place(b, lay["W"], packed.astype("<u4"))
    O._place(b, lay["S"], s16.astype("<u2"))
    O._place(b, lay["B"], b16.astype("<u2"))
    for v in range(nb):
        O._place(a, lay["X"] + 4 * K * v, xs[v].astype("<f4"))
    wants = [np.asarray(Q.qmv2_ksplit_reference(single, xs[v], q, s16, b16), np.float32).view(np.uint32) for v in range(nb)]
    stride = 4 * N
    for v in range(nb):
        c[lay["OUT"] + stride * v:lay["OUT"] + stride * v + 4 * N] = SENT * (4 * N)
    name = "qmv_q%d_lmhead%s_ks%d_lean_b%d%s" % (bits, "_x32" if bits == 8 else "", S, nb, "_p%d" % pv if pv else "")
    d = author(work / name, prog, bytes(a), bytes(b), bytes(c), lay, template=single)
    tpg = 32 * S
    entry = dict(kind="lm_head", bits=bits, role="head", variant=variant, op="head", batch=nb,
                 threadgroups=lay["groups"], threads_per_group=tpg, base=0, slot_map=dict(c_written=0, x=1, weights=2),
                 x_dtype="fp32", x_offset=lay["X"], X=lay["X"], W=lay["W"], S=lay["S"], B=lay["B"], OUT=lay["OUT"],
                 x_stride=4 * K, out_stride=stride, out_dtype="fp32",
                 layout_note="vector-major: sequence b's x at X + 4 K b, its logits at OUT + 4 V b",
                 reduction="split-K: %d contiguous K slices, each qmv2 order, summed in slice order in fp32; per sequence" % S,
                 recipe=dict(builder="g17qmv.build_qmv2", flags=["batch", "hi16_scales", "ksplit", "xvec"] +
                             ([] if lay.get("hoist_consts") else ["no_hoist"]) + (["batch_pass"] if pv else []),
                             layout=jsonable(lay)))

    def check(out):
        return sum(int((np.frombuffer(out, "<u4", N, lay["OUT"] + stride * v) != wants[v]).sum()) for v in range(nb))
    return [dict(name=name, dir=d, prog=prog, entry=entry, threads=tpg * lay["groups"], group=tpg, base=0, check=check)]


def norm_batch_layout(in_dtype, out32, seed, nb):
    """The wide RMSNorm over nb rows in ONE dispatch (MM 25.144.3): nb threadgroups of 1024, row b's x at X + b d
    elements and its out at OUT + b d elements (vector-major); the gain is shared."""
    lay = dict(norm_layout(in_dtype, out32, seed), batch=nb)
    d = lay["d"]
    xs, os_ = (2 if in_dtype == "half" else 4), (4 if out32 else 2)
    lay["a_bytes"] = max(lay["a_bytes"], Q._align(lay["X"] + xs * d * nb))
    lay["c_bytes"] = max(lay["c_bytes"], Q._align(lay["OUT"] + os_ * d * nb))
    return lay


ROWS_N = dict(swiglu=8192, residual=2048, fold=2048)


# THE BATCHED-DECODE PROJECTIONS (MM 25.172): the model's shapes, each role's split-K, and the FFN block's two halves
# (the graph's L<l>_ffn block holds w1 then w3 rows under one W, one S and one B array)
QSM_ROLES = {"qkv": (4096, 2048, 2, None), "wo": (2048, 2048, 4, None), "w1": (8192, 2048, 1, 0), "w3": (8192, 2048, 1, 1),
             "w2": (2048, 8192, 4, None),
             # the batched lm_head (MM 25.174): the vocabulary dequantized once for every sequence, logits [16][V] fp32
             "head": (92544, 2048, 1, None)}


# at mb 32 (MM 25.178) one n-tile a threadgroup: split-K where the head grid's 128 slices allow it
QSM_SK32 = {"qkv": 1, "wo": 4, "w1": 1, "w3": 1, "w2": 4, "head": 1}
# THE OCCUPANCY FORMS (MM 25.185): split-K chosen by the chained sweep (about 512 one-simdgroup threadgroups is the
# knee; 1,024 is past it); w1 / w3 split 2 needs the head grid's 256 slices (tlower head_slices <= 256) and a psum sum
QSM_SK_OCC = {"qkv": 4, "wo": 8, "w1": 2, "w3": 2, "w2": 8, "head": 1}
# at mb 32 only qkv gains (86.8 -> 50.1 us at split 2, 256 slices); wo and w2 at 8 are slower than at 4
QSM_SK32_OCC = dict(QSM_SK32, qkv=2)
# the padded build (MM 25.196, Qwen3 shapes chained): w1 / w3 at 4,096 rows split 4 (22.3 us against 28.9 at 2)
QSM_SK_PAD = {"qkv": 4, "wo": 8, "w1": 4, "w3": 4, "w2": 8, "head": 1}


def _qsm_occ():
    """the occupancy split-K for this arch: QSM_SK_OCC unless ARCHS gives its own. Qwen3-8B's qkv (192 n groups) and
    w1 / w3 (384) are not the power of two <= 256 a split needs (g17qsm), so they run unsplit there."""
    return ARCH.get("qsm_sk_occ", QSM_SK_OCC)


def build_qsm_batch(work, bits, B, with_qsm=True, occ=False, h16=False, pad=None, sks=None, roles=None):
    """g17qsm (xrows) per role and the psum passes a B-sequence graph needs, each verified bit for bit over a sentinel."""
    import g17qsm as QS
    import g17psum as PS
    if bits != 4:
        raise SystemExit("qsm_batch: q4 only")
    jobs = []
    mb = 32 if B == 32 else 16
    for role, (N, K, sk, half) in ((roles or QSM_ROLES).items() if with_qsm else ()):   # 16 rows at every B <= 16
        base_sk = QSM_SK32[role] if mb == 32 else sk
        sk = sks[role] if sks is not None else (QSM_SK32_OCC if mb == 32 else _qsm_occ())[role] if occ else base_sk
        if occ and sk == base_sk and not h16:
            continue                                   # the same bundle as the base build
        if pad is not None:                           # MM 25.196: the FFN zero-padded (w1 / w3 rows, w2's K)
            N, K = (pad, K) if half is not None else (N, pad) if role == "w2" else (N, K)
        offsets = None
        if half is not None:                      # inside the 16,384-row FFN block
            NB = 2 * N
            offsets = (half * N * (K // 2), NB * (K // 2) + half * N * (K // 32),
                       NB * (K // 2) + NB * (K // 32) + half * N * (K // 32))
        lay = QS.layout(N, K, 1 if mb == 32 else 2, 4, sk, xrows=True, offsets=offsets, mb=mb, h16=h16)
        prog = QS.build(lay)
        x, packed, s16, b16, q = QS.case(lay, seed=70 + len(role), batch=B)
        want = QS.reference(lay, x, q, s16, b16)
        a, bb, c = QS.io(lay, x, packed, s16, b16)
        name = "qsm_q4_%s_b%d_sk%d%s%s" % (role, mb, sk, "_h16" if h16 else "", "_pad%d" % pad if pad else "")
        d = QS.author(work / name, prog, a, bb, c, lay)
        wv = np.ascontiguousarray(want, "<f4").view("<u4").reshape(-1)
        entry = dict(kind="qsm", bits=4, role=role, variant=dict(sk=sk, xrows=True, **({"mb": 32} if mb == 32 else {}),
                                                              **({"h16": True} if h16 else {}),
                                                              **({"pad": pad} if pad is not None else {})),
                     N=N, K=K, sk=sk, W=lay["W"],
                     S=lay["S"], B=lay["B"], threadgroups=lay["groups"], threads_per_group=32, base=1,
                     slot_map=dict(weights=1, x_rows=2, y_written=3),
                     out_layout=("y fp32 [mb][N]" if sk == 1 else "partials fp32 [sk][mb][N], summed in ascending order"),
                     recipe=dict(builder="g17qsm.build", layout=jsonable(lay)))
        jobs.append(dict(name=name, dir=d, prog=prog, entry=entry, threads=32 * lay["groups"], group=32, base=1,
                         check=lambda out, wv=wv: int((np.frombuffer(out, "<u4", wv.size) != wv).sum())))
    if h16 and pad is None:
        return jobs                                    # the psum passes are the occupancy build's
    rng = np.random.default_rng(90 + B)
    Aq, Ad, Ah = ARCH["qkv"], ARCH["d"], ARCH["hd"]
    psums = ((("qkv", Aq, QSM_SK32["qkv"] if mb == 32 else 2, "sum"), ("wo", Ad, 4, "add16"),
              ("w2", Ad, 4, "fold16"), ("attn", Ah, 1, "half")) if not occ else
             (("qkv", 4096, QSM_SK32_OCC["qkv"], "sum"),) if mb == 32 else
             (("qkv", Aq, _qsm_occ()["qkv"], "sum"), ("wo", Ad, _qsm_occ()["wo"], "add16"),
              ("w2", Ad, _qsm_occ()["w2"], "fold16"), ("w1", ARCH["ffn"], _qsm_occ()["w1"], "sum"),
              ("w1", ARCH["ffn"], _qsm_occ()["w1"], "swiglu")))
    if pad is not None:                                # the padded build's own passes, at its split-K and widths
        psums = (("qkv", Aq, sks["qkv"], "sum"), ("wo", Ad, sks["wo"], "add16"), ("w2", Ad, sks["w2"], "fold16"),
                 ("w1", pad, sks["w1"], "swiglu"), ("attn", Ah, 1, "half"))
    for role, N, sk, mode in psums:
        lay = PS.layout(N, sk, mode, rows=B, pstride=mb * N)
        prog = PS.build(lay)
        parts = rng.standard_normal((sk, mb, N)).astype(np.float32)
        resid = (rng.standard_normal((B, N)).astype(np.float16) if mode == "add16" else
                 rng.standard_normal((B, N)).astype(np.float32) if mode == "fold16" else
                 rng.standard_normal((sk, mb, N)).astype(np.float32) if mode == "swiglu" else None)
        a = bytearray(lay["a_bytes"]); O._place(a, 0, parts.reshape(-1))
        bb = bytearray(lay["b_bytes"])
        if resid is not None:
            O._place(bb, 0 if mode == "swiglu" else lay["IN"], resid.reshape(-1))
        want = PS.reference(lay, parts, resid)
        wv = want.view("<u2" if want.dtype == np.float16 else "<u4")
        c = SENT * lay["c_bytes"]
        name = "psum_%s_%s_b%d_sk%d%s" % (role, mode, B, sk, "_pad%d" % pad if pad else "")
        d = QS.author(work / name, prog, bytes(a), bytes(bb), c, lay)
        entry = dict(kind="psum", role=role, variant=dict(mode=mode, rows=B, sk=sk, **({"pad": pad} if pad else {})),
                     N=N, sk=sk, mode=mode, rows=B,
                     pstride=lay["pstride"], IN=lay["IN"], OUT=lay["OUT"], threadgroups=lay["groups"], threads_per_group=32,
                     base=1, slot_map=dict(partials=1, residual=2, out_written=3),
                     out_layout={"sum": "fp32 [rows][N]", "add16": "h fp32 [rows][N]", "fold16": "x fp16 [rows][N]",
                                 "half": "fp16 [rows][N]", "swiglu": "act fp16 [rows][N]"}[mode],
                     recipe=dict(builder="g17psum.build", layout=jsonable(lay)))
        dt = "<u2" if want.dtype == np.float16 else "<u4"
        jobs.append(dict(name=name, dir=d, prog=prog, entry=entry, threads=32 * lay["groups"], group=32, base=1,
                         check=lambda out, wv=wv, dt=dt, off=lay["OUT"]: int((np.frombuffer(out, dt, wv.size, off) != wv).sum())))
    if not occ and pad is None:
        jobs += build_rows_kind(work, "swiglu", B, n=ARCH["ffn"])
    return jobs


FAST_SWIGLU_ULPS = 1          # the fast SwiGLU's enclosure (fp16 ulps from the float64 value, MM 25.183)


def build_rows_kind(work, kind, M, in16=False, fast=False, n=None, out32=False):
    """A prefill elementwise kernel over M rows (g17rows), verified per element over a 0x7f sentinel on every output.
    in16: the fp16-input SwiGLU (MM 25.183), reading the half-output w1 / w3 GEMMs."""
    import g17rows as W
    N = n or ROWS_N[kind]                              # n: an explicit width (the batched decode's FFN, MM 25.189)
    lay = W.rows_layout(kind, M, N, in16=in16, fast=fast, out32=out32)
    prog = W.build_rows(lay)
    n = M * N
    rng = np.random.default_rng(500 + M + len(kind))
    a, b, c = bytearray(lay["a_bytes"]), bytearray(lay["b_bytes"]), bytearray(lay["c_bytes"])
    if kind == "swiglu":
        g = (rng.standard_normal(n) * 3).astype(np.float16 if in16 else np.float32)
        u = (rng.standard_normal(n) * 2).astype(np.float16 if in16 else np.float32)
        O._place(a, 0, g)
        O._place(b, 0, u)
        want = W.rows_reference(lay, (g, u))
        checks = [(0, "<u4", want["act"].astype(np.float32).view(np.uint32))] if out32 else \
                 [(0, "<u2", want["act"].view(np.uint16))]
        slots = dict(gate=1, up=2, act_written=3)
    elif kind == "residual":
        y = (rng.standard_normal(n) * 2).astype(np.float32)
        x = rng.standard_normal(n).astype(np.float16)
        O._place(a, 0, y)
        O._place(b, 0, x)
        want = W.rows_reference(lay, (y, x))
        checks = [(0, "<u4", want["h"].view(np.uint32)), (lay["H16"], "<u2", want["h16"].view(np.uint16))]
        slots = dict(y=1, x16=2, h_written=3)
    else:
        p = (rng.standard_normal(2 * n) * 2).astype(np.float32)
        h = (rng.standard_normal(n) * 3).astype(np.float32)
        O._place(a, 0, p)
        O._place(b, 0, h)
        want = W.rows_reference(lay, (p, h))
        checks = [(0, "<u2", want["x"].view(np.uint16))]
        slots = dict(partials=1, h=2, x_written=3)
    for off, dt, wv in checks:
        nb_ = wv.size * (2 if dt == "<u2" else 4)
        c[off:off + nb_] = SENT * nb_
    if len(c) < 4 * max(len(a), len(b)):
        c = c + bytearray(4 * max(len(a), len(b)) - len(c))      # the manifest's transport view (as norm_rows)
    name = "rows_%s_m%d%s%s%s" % (kind, M, "_in16" if in16 else "", "_fast" if fast else "", "_out32" if out32 else "")
    d = author(work / name, prog, bytes(a), bytes(b), bytes(c), lay)
    entry = dict(kind=kind + "_rows", role=kind,
                 variant=dict(rows=M, **({"in16": True} if in16 else {}), **({"fast": True} if fast else {}),
                              **({"out32": True} if out32 else {})), rows=M, N=N,
                 threadgroups=lay["groups"],
                 threads_per_group=32, base=1, slot_map=slots, unroll=W.U,
                 recipe=dict(builder="g17rows.build_rows", layout=jsonable(lay)))
    if fast:
        entry["enclosure"] = ("op1272 and the raw reciprocal: every output within %d fp16 ulp of fp16(silu(g) u) in "
                              "float64, the wrong-base control outside it (not bit-exact)" % FAST_SWIGLU_ULPS)
    if kind == "residual":
        entry.update(H=0, H16=lay["H16"])
    elif kind == "fold":
        entry.update(P0=0, P1=4 * n, X=0)
    else:
        entry.update(GATE=0, UP=0, ACT=0)

    def check(out):
        if fast:
            # THE ENCLOSURE (op1272 is not CPU-reproducible): every output within FAST_SWIGLU_ULPS of the float64
            # value, and the wrong-base control far outside it, or the delivery refuses (a control that cannot fail)
            got = np.frombuffer(out, "<f2", n, 0)
            err, ctrl = W.fast_check(got, g, u), W.fast_check(got, g, u, base="2")
            print("  %s: %d fp16 ulp from float64 (bound %d); the wrong-base control %d" % (name, err, FAST_SWIGLU_ULPS, ctrl))
            return 0 if err <= FAST_SWIGLU_ULPS < ctrl else max(1, err)
        return sum(int((np.frombuffer(out, dt, wv.size, off) != wv).sum()) for off, dt, wv in checks)
    return [dict(name=name, dir=d, prog=prog, entry=entry, threads=32 * lay["groups"], group=32, base=1, check=check)]


def build_norm_batch(work, role, in_dtype, out32, seed, nb):
    lay = norm_batch_layout(in_dtype, out32, seed, nb)
    prog = O.build_rmsnorm_wide(lay, EPS)
    d = lay["d"]
    rng = np.random.default_rng(41 + (in_dtype == "half") + 2 * out32 + 4 * seed + 16 * nb)
    a, b, c = (bytearray(x) for x in O._buffers(lay))
    g = (1 + 0.1 * rng.standard_normal(d)).astype(np.float16)
    O._place(b, lay["G"], g)
    xs, os_ = (2 if in_dtype == "half" else 4), (4 if out32 else 2)
    wants = []
    for v in range(nb):
        row = (rng.standard_normal(d) * rng.uniform(0.5, 4)).astype(np.float16 if in_dtype == "half" else np.float32)
        O._place(a, lay["X"] + xs * d * v, row)
        want = O.rmsnorm_wide_reference(row.astype(np.float32), g, Spec, seed=seed).astype(np.float16)
        wants.append(want.astype(np.float32).view(np.uint32) if out32 else want.view(np.uint16))
    c[lay["OUT"]:lay["OUT"] + os_ * d * nb] = SENT * (os_ * d * nb)
    name = "rmsnorm_wide_%s_d2048_t1024%s%s_%s_b%d" % (in_dtype, "_out32" if out32 else "", "_seed" if seed else "", role, nb)
    if len(c) < 4 * len(a):
        # prefill-size rows: the manifest's transport view needs binding-3 rows in proportion to binding 1
        c = c + bytearray(4 * len(a) - len(c))
    d_ = author(work / name, prog, a, b, c, lay)
    entry = dict(kind="norm", role=role, variant=dict(out32=out32, seed=seed, batch=nb), op="rmsnorm_wide", in_dtype=in_dtype,
                 out_dtype="fp32 of the fp16-rounded row" if out32 else "fp16 row", threadgroups=nb, threads_per_group=1024,
                 base=0, slot_map=dict(c_written=0, x=1, gain=2), X=lay["X"], G=lay["G"], OUT=lay["OUT"], batch=nb,
                 x_stride=xs * d, out_stride=os_ * d,
                 recipe=dict(builder="g17decodeops.build_rmsnorm_wide", eps=EPS, layout=jsonable(lay)))

    def check(out):
        return sum(int((np.frombuffer(out, "<u4" if out32 else "<u2", d, lay["OUT"] + os_ * d * v) != wants[v]).sum())
                   for v in range(nb))
    return [dict(name=name, dir=d_, prog=prog, entry=entry, threads=1024 * nb, group=1024, base=0, check=check)]


def build_norm(work, role, in_dtype, out32, seed):
    lay = norm_layout(in_dtype, out32, seed)
    prog = O.build_rmsnorm_wide(lay, EPS)
    rng = np.random.default_rng(31 + (in_dtype == "half") + 2 * out32 + 4 * seed)
    a, b, c = (bytearray(x) for x in O._buffers(lay))
    d = ARCH["d"]
    v = (rng.standard_normal(d) * rng.uniform(0.5, 4)).astype(np.float16 if in_dtype == "half" else np.float32)
    g = (1 + 0.1 * rng.standard_normal(d)).astype(np.float16)
    O._place(a, lay["X"], v)
    O._place(b, lay["G"], g)
    nb = d * (4 if out32 else 2)
    c[lay["OUT"]:lay["OUT"] + nb] = SENT * nb
    want = O.rmsnorm_wide_reference(v.astype(np.float32), g, Spec, seed=seed).astype(np.float16)
    wv = want.astype(np.float32).view(np.uint32) if out32 else want.view(np.uint16)
    variant = dict(out32=out32, seed=seed)
    name = "rmsnorm_wide_%s_d2048_t1024%s%s_%s" % (in_dtype, "_out32" if out32 else "", "_seed" if seed else "", role)
    d = author(work / name, prog, a, b, c, lay)
    entry = dict(kind="norm", role=role, variant=variant, op="rmsnorm_wide", in_dtype=in_dtype,
                 out_dtype="fp32 of the fp16-rounded row" if out32 else "fp16 row", threadgroups=1, threads_per_group=1024,
                 base=0, slot_map=dict(c_written=0, x=1, gain=2), X=lay["X"], G=lay["G"], OUT=lay["OUT"],
                 recipe=dict(builder="g17decodeops.build_rmsnorm_wide", eps=EPS, layout=jsonable(lay)))

    def check(out):
        got = np.frombuffer(out, "<u4" if out32 else "<u2", ARCH["d"], lay["OUT"])
        return int((got != wv).sum())
    return [dict(name=name, dir=d, prog=prog, entry=entry, threads=1024, group=1024, base=0, check=check)]


def _attn_true_softmax(lay, qkv, cos, sin, Kc, Vc, q0, base=2):
    """The true per-head softmax (float64) over the roped q and the appended K/V row. base=2 is the contract's exp2, the
    enclosure golden for a hw_exp2 (op1272) attention kernel (not bit-exact to exp2_soft, MM 25.144.5). base='e' is the
    WRONG-BASE control: a correct kernel must land OUTSIDE the enclosure bound of it, so the delivered check can fail."""
    import g17decodestep as D_
    H, KVH, Dm = lay["heads"], lay["kv_heads"], lay["head_dim"]
    q = qkv[:H * Dm].reshape(H, Dm)
    k = qkv[H * Dm:(H + KVH) * Dm].reshape(KVH, Dm)
    v = qkv[(H + KVH) * Dm:].reshape(KVH, Dm)
    q16 = D_.narrow(D_.fmul(D_.rope_rotate(q, cos, sin), D_.q_scale(D_.MILESTONE)))
    k16, v16 = D_.narrow(D_.rope_rotate(k, cos, sin)), D_.narrow(v)
    K = np.array(Kc, np.float64); V = np.array(Vc, np.float64)
    K[:, q0] = k16; V[:, q0] = v16
    out = np.zeros((H, Dm))
    for h in range(H):
        kv = h // (H // KVH)
        s = K[kv, :q0 + 1] @ np.asarray(q16[h], np.float64)
        p = (np.exp2(s - s.max()) if base == 2 else np.exp(s - s.max()))
        out[h] = (p @ V[kv, :q0 + 1]) / p.sum()
    return out


ATTN_ENCLOSURE_BOUND = 8.7e-4          # M2's row-scaled softmax enclosure bound, reused for decode attention (MM 25.144.5)


def build_attn(work, cap, flags=""):
    lay = attn_layout(cap, flags)
    prog = A.build_attn_split_rope(lay)
    H, KVH, Dm = lay["heads"], lay["kv_heads"], lay["head_dim"]
    jobs = []
    name = "attn_widebf_s32_attn32%s%s" % ("" if cap == 272 else "_cap%d" % cap,
                                           "".join("_" + f.replace("=", "") for f in flags.split("+") if f))
    for q0 in sorted({0, 5, cap // 2, cap - 1}):
        rng = np.random.default_rng(300 + q0)
        qkv = (rng.standard_normal((H + 2 * KVH) * Dm) * 1.5).astype(np.float32)
        qkv_in = qkv
        if lay.get("qknorm"):
            # MM 25.188: the kernel reads the raw row and the gains; every reference sees the QK-normed row
            import g17qwen3 as Q3
            gq = (1 + 0.3 * rng.standard_normal(Dm)).astype(np.float16)
            gk = (1 + 0.3 * rng.standard_normal(Dm)).astype(np.float16)
            qkv = Q3.headnorm_reference(qkv_in, gq, gk, lay["qknorm_eps"])
        th = np.float64(1e6) ** (-np.arange(Dm // 2) * 2 / Dm)
        pos = np.arange(cap)[:, None] * th[None, :]
        cos, sin = np.cos(pos[q0]).astype(np.float32), np.sin(pos[q0]).astype(np.float32)
        Kc = rng.standard_normal((KVH, cap, Dm)).astype(np.float16)
        Vc = rng.standard_normal((KVH, cap, Dm)).astype(np.float16)
        Kc[:, q0:] = np.float16(np.nan)
        Vc[:, q0:] = np.float16(np.nan)
        want, _, Kw, _ = A.attn_rope_reference(lay, qkv, cos, sin, Kc.astype(np.float32), Vc.astype(np.float32), q0,
                                               partials=True)
        true2 = _attn_true_softmax(lay, qkv, cos, sin, Kc, Vc, q0) if lay.get("hw_exp2") else None
        truee = _attn_true_softmax(lay, qkv, cos, sin, Kc, Vc, q0, base='e') if lay.get("hw_exp2") else None
        a = bytearray(lay.get("q_bytes", lay["LEN"] + 256))
        O._place(a, 0, qkv_in.astype("<f4"))
        if lay.get("qknorm"):
            O._place(a, lay["QG"], np.concatenate([gq, gk]))
        b = bytearray(lay["rope_bytes"])
        O._place(b, 0, np.asarray([q0], "<u4"))
        O._place(b, lay["COST"], np.cos(pos).astype("<f4"))
        O._place(b, lay["SINT"], np.sin(pos).astype("<f4"))
        c = bytearray(lay["region3_bytes"])
        O._place(c, lay["KOFF"], Kc.reshape(-1))
        O._place(c, lay["VOFF"], Vc.reshape(-1))
        O._place(c, lay["ATTN"], np.full(H * Dm, np.float32(np.nan)))
        d = author(work / ("%s_q%d" % (name, q0)), prog, a, b, c, lay)
        wv = np.asarray(want, np.float16).astype(np.float32).view(np.uint32).reshape(-1)
        krow = Kw[:, q0].astype(np.float16).view(np.uint16)

        def check(out, wv=wv, krow=krow, q0=q0, true2=true2, truee=truee):
            kr = np.frombuffer(out, "<u2", KVH * cap * Dm, lay["KOFF"]).reshape(KVH, cap, Dm)[:, q0]
            kbad = int((kr != krow).sum())
            if true2 is not None:
                # hw_exp2 (op1272) is not bit-exact to exp2_soft: ENCLOSE the true base-2 softmax within the row-scaled
                # bound (attn is fp32 of the fp16-rounded row), not the exact wv compare. A CONTROL THAT CANNOT FAIL is
                # no check, so the delivery is REFUSED unless a base-e (wrong-base) golden lands OUTSIDE the same bound
                # - proving the check discriminates. EXCEPTION: at q0 = 0 there is one key, so softmax is exactly 1 for
                # any base and the base-e control coincides with base-2 (it cannot be outside the bound); that bucket
                # does not exercise exp2, so it is enclosure-only. 3 of the 4 delivered q0 buckets exercise the control.
                got = np.frombuffer(out, "<f4", H * Dm, lay["ATTN"]).astype(np.float64).reshape(H, Dm)
                rowmax = np.abs(true2).max(axis=1, keepdims=True)
                encl = float((np.abs(got - true2) / rowmax).max())
                ctrl = float((np.abs(got - truee) / rowmax).max())
                good = encl <= ATTN_ENCLOSURE_BOUND and (q0 == 0 or ctrl > ATTN_ENCLOSURE_BOUND)
                return kbad + (0 if good else 1)
            got = np.frombuffer(out, "<u4", H * Dm, lay["ATTN"])
            return int((got != wv).sum()) + kbad
        jobs.append(dict(name="%s_q%d" % (name, q0), dir=d, prog=prog, threads=lay["split_groups"] * 1024, group=1024,
                         base=0, rounds=3, check=check, entry=None))
    entry = dict(kind="attn", cap=cap, role="attn", variant=attn_variant(flags), threadgroups=lay["split_groups"],
                 threads_per_group=1024, base=0, slot_map=dict(written=0, a=1, b=2), ATTN=lay["ATTN"],
                 attn_bytes=lay["out_bytes"] * 2, region3_bytes=lay["region3_bytes"], COST=lay["COST"], SINT=lay["SINT"],
                 rope_bytes=lay["rope_bytes"], P=lay["P"], KOFF=lay["KOFF"], VOFF=lay["VOFF"], trips_cap=lay["trips_cap"],
                 recipe=dict(builder="g17attn.build_attn_split_rope", layout=jsonable(lay)))
    jobs[-1]["entry"] = entry                       # the delivered bundle carries the last verified q0's inputs
    jobs[-1]["deliver_as"] = name
    return jobs


def build_attn_batch(work, cap, nb, gqa=False):
    """Batched decode attention (MM 25.144.3): ONE dispatch of nb 16 threadgroups, every sequence at its own q0 (0,
    a mid, cap-1 and more) with its own qkv row, caches, q0 word and attn row; each checked against the single-sequence
    reference (its attn32 row and its appended K row), with NaN past each q0 so a wrong-row read shows.
    gqa: M5's GQA-paired form (with_gqapair) batched - its pair counters must read 0 after the run."""
    SS = G.gen_batch_layout(nb, cap=cap)["SS"]
    lay = A.with_batch(A.with_gqapair(attn_layout(cap)) if gqa else attn_layout(cap), nb, SS)
    prog = A.build_attn_split_rope(lay)
    H, KVH, Dm = lay["heads"], lay["kv_heads"], lay["head_dim"]
    rng = np.random.default_rng(700 + nb)
    th = np.float64(1e6) ** (-np.arange(Dm // 2) * 2 / Dm)
    pos = np.arange(cap)[:, None] * th[None, :]
    a = bytearray(lay["LEN"] * nb + 256)
    b = bytearray(lay["rope_bytes"])
    O._place(b, lay["COST"], np.cos(pos).astype("<f4"))
    O._place(b, lay["SINT"], np.sin(pos).astype("<f4"))
    c = bytearray(lay["region3_bytes"])
    O._place(c, lay["ATTN"], np.full(H * Dm * nb, np.float32(np.nan)))
    wants, krows, q0s = [], [], []
    for s in range(nb):
        q0 = [0, cap - 1, cap // 2, 5, 100, cap - 2, 1, 50][s % 8]
        q0s.append(q0)
        qkv = (rng.standard_normal((H + 2 * KVH) * Dm) * 1.5).astype(np.float32)
        cos, sin = np.cos(pos[q0]).astype(np.float32), np.sin(pos[q0]).astype(np.float32)
        Kc = rng.standard_normal((KVH, cap, Dm)).astype(np.float16)
        Vc = rng.standard_normal((KVH, cap, Dm)).astype(np.float16)
        Kc[:, q0:] = np.float16(np.nan)
        Vc[:, q0:] = np.float16(np.nan)
        want, _, Kw, _ = A.attn_rope_reference(lay, qkv, cos, sin, Kc.astype(np.float32), Vc.astype(np.float32), q0,
                                               partials=True)
        O._place(a, lay["LEN"] * s, qkv.astype("<f4"))
        O._place(b, SS * s, np.asarray([q0], "<u4"))
        O._place(c, lay["KOFF"] + lay["KVS"] * s, Kc.reshape(-1))
        O._place(c, lay["VOFF"] + lay["KVS"] * s, Vc.reshape(-1))
        wants.append(np.asarray(want, np.float16).astype(np.float32).view(np.uint32).reshape(-1))
        krows.append(Kw[:, q0].astype(np.float16).view(np.uint16))
    name = "attn_widebf_s32_attn32%s%s_b%d" % ("" if cap == 272 else "_cap%d" % cap, "_gqa" if gqa else "", nb)
    d = author(work / name, prog, a, b, c, lay)

    def check(out):
        bad = 0
        for s in range(nb):
            got = np.frombuffer(out, "<u4", H * Dm, lay["ATTN"] + 4 * H * Dm * s)
            kr = np.frombuffer(out, "<u2", KVH * cap * Dm, lay["KOFF"] + lay["KVS"] * s).reshape(KVH, cap, Dm)[:, q0s[s]]
            bad += int((got != wants[s]).sum()) + int((kr != krows[s]).sum())
        if gqa:
            bad += int(np.frombuffer(out, "<u4", 64, 0).any())          # the pair counters reset to zero
        return bad
    variant = dict(widebf=True, attn32=True, batch=nb, **({"gqapair": True} if gqa else {}))
    entry = dict(kind="attn", cap=cap, role="attn", variant=variant, threadgroups=H * nb,
                 threads_per_group=1024, base=0, slot_map=dict(written=0, a=1, b=2), ATTN=lay["ATTN"],
                 attn_bytes=lay["out_bytes"] * 2, region3_bytes=lay["region3_bytes"], COST=lay["COST"], SINT=lay["SINT"],
                 rope_bytes=lay["rope_bytes"], P=lay["P"], KOFF=lay["KOFF"], VOFF=lay["VOFF"], trips_cap=lay["trips_cap"],
                 batch=nb, qkv_stride=lay["LEN"], q0_stride=SS, kv_stride=lay["KVS"], attn_stride=4 * H * Dm,
                 recipe=dict(builder="g17attn.build_attn_split_rope", layout=jsonable(lay)))
    return [dict(name=name, dir=d, prog=prog, threads=H * nb * 1024, group=1024, base=0, rounds=3, check=check, entry=entry)]


def build_gen(work, cap, batched=False):
    """Pass 1 (argmax partials) and the step (pass 2 + log + embedding row + q0 advance), chained: phase 1 runs pass 1
    for every case; phase 2 authors each step bundle on that case's pairs. Cases: q0 0, cap/2, cap-1, chosen and forced.
    The attention's cos[0] word at GEN + COST is a sentinel the log write must not reach."""
    # x at R + 4 d: where the residual projections put it (g17qmv.with_residual's RES; 8,192 for InternLM2)
    lay = G.gen_layout(V=ARCH["vocab_pad"], per_lane=ARCH["per_lane"], d=ARCH["d"], cap=cap, R_X16=4 * ARCH["d"],
                       GEN=max(12288, 6 * ARCH["d"]))           # the state past h fp32 [d] and x fp16 [d] (Qwen3-8B)
    if batched:
        lay = dict(lay, batched=True)            # loads issued before use, the row copied as words (MM 25.144.7)
    sfx = "_batched" if batched else ""
    COST = A.with_rope_tables(A.attn_rope_layout(cap=cap))["COST"]
    P1, P2 = G.build_pass1(lay), G.build_gen(lay)
    REGION = lay["GEN"] + COST + 256
    V = lay["V"]
    rng = np.random.default_rng(9)
    emb = rng.standard_normal((1000, ARCH["d"])).astype(np.float16)
    COS0 = 0x3F800000
    cases = []
    for q0 in sorted({0, cap // 2, cap - 1}):
        for forced in (None, 777):
            x = rng.standard_normal(V).astype(np.float32)
            x[421] = 40.0
            reg = bytearray(REGION)
            O._place(reg, lay["R_X16"], np.full(ARCH["d"], np.float16(np.nan)))
            O._place(reg, lay["GEN"], np.asarray([q0], "<u4"))
            nxt = min(q0 + 1, cap - 1)
            log = np.full(cap, 0xFFFFFFFF, np.uint32)
            log[:nxt] = np.arange(nxt) + 100
            if forced is not None:
                log[nxt] = forced
            O._place(reg, lay["LOG"], log.astype("<u4"))
            O._place(reg, lay["GEN"] + COST, np.asarray([COS0], "<u4"))
            tag = "gen_cap%d_q%d_%s%s" % (cap, q0, "forced" if forced is not None else "chosen", sfx)
            d = author(work / (tag + "_p1"), P1, x.astype("<f4").tobytes(), bytes(256), bytes(lay["pairs_bytes"]), lay)
            want = forced if forced is not None else int(np.argmax(x))
            cases.append(dict(tag=tag, p1=d, reg=reg, want=want, nxt=nxt))

    def phase2(outs):
        jobs = []
        for cs in cases:
            pairs = outs[cs["tag"] + "_p1"][:lay["pairs_bytes"]]
            d = author(work / (cs["tag"] + "_p2"), P2, pairs, emb.tobytes(), bytes(cs["reg"]), lay)

            def check(out, cs=cs):
                bad = 0
                bad += int(np.frombuffer(out, "<u4", 1, lay["GEN"])[0]) != cs["nxt"]
                lg = np.frombuffer(out, "<u4", cap, lay["LOG"])
                bad += int(lg[cs["nxt"]]) != cs["want"]
                bad += int(not np.array_equal(lg[:cs["nxt"]], np.arange(cs["nxt"]) + 100))
                bad += int(not np.all(lg[cs["nxt"] + 1:] == 0xFFFFFFFF))
                bad += int(not np.array_equal(np.frombuffer(out, "<u2", ARCH["d"], lay["R_X16"]), emb[cs["want"]].view(np.uint16)))
                bad += int(np.frombuffer(out, "<u4", 1, lay["GEN"] + COST)[0]) != COS0
                return bad
            jobs.append(dict(name=cs["tag"] + "_p2", dir=d, prog=P2, threads=32, group=32, base=1, check=check, entry=None))
        jobs[0]["entry"] = dict(kind="gen_step", cap=cap, role="gen_step", variant={"batched": True} if batched else {},
                                threadgroups=1, threads_per_group=32,
                                base=1, slot_map=dict(a=1, b=2, c_written=3), GEN=lay["GEN"], LOG=lay["LOG"],
                                R_X16=lay["R_X16"], region_bytes=lay["region_bytes"], PAIRS=lay["PAIRS"],
                                recipe=dict(builder="g17gen.build_gen", layout=jsonable(lay)))
        jobs[0]["deliver_as"] = "gen_step_cap%d%s" % (cap, sfx)
        return jobs

    def check_pairs(out, cs):
        return 0                                     # pass 1 is judged through the step's token (its only consumer)
    p1 = []
    for i, cs in enumerate(cases):
        p1.append(dict(name=cs["tag"] + "_p1", dir=cs["p1"], prog=P1, threads=lay["G"] * 32, group=32, base=1,
                       check=lambda out, cs=cs: check_pairs(out, cs), entry=None))
    p1[0]["entry"] = dict(kind="gen_argmax", cap=cap, role="gen_argmax", variant={"batched": True} if batched else {},
                          threadgroups=lay["G"],
                          threads_per_group=32, base=1, slot_map=dict(a=1, b=2, c_written=3), GEN=lay["GEN"],
                          LOG=lay["LOG"], R_X16=lay["R_X16"], region_bytes=lay["region_bytes"], PAIRS=lay["PAIRS"],
                          recipe=dict(builder="g17gen.build_pass1", layout=jsonable(lay)))
    p1[0]["deliver_as"] = "gen_argmax_pass1_cap%d%s" % (cap, sfx)
    return p1, phase2


def build_qmvw(work, role, nb):
    """MM 25.207: one g17qmvw projection (nb vectors against one weight stream, each weight dequantized once) at the
    model's shape, verified bit-exact against qmvw.reference over a sentinel. Roles: the decode projections on their own
    blocks (the qmv layout's default W / S / B) and the head."""
    import g17qmvw as QW
    d, hd, qkv, ffn = ARCH["d"], ARCH["hd"], ARCH["qkv"], ARCH["ffn"]
    N, K = dict(qkv=(qkv, d), wo=(d, hd), w1_block=(ffn, d), w2=(d, ffn), head=(ARCH["vocab"], d))[role]
    # the kl8 form: the r4 rows form is 5-10% faster but its order split an 8B draft run from plain at token 18 (MM 25.208)
    lay = QW.layout(N, K, nb)
    prog = QW.build(lay)
    packed, s16, b16, q, x = QW.case(lay, seed=60 + len(role))
    want = QW.reference(lay, q, s16, b16, x)
    a, bb, c = QW.io(lay, packed, s16, b16, x)
    name = "qmvw_q4_%s_nb%d" % (role, nb)
    dd = QW.author(work / name, prog, a, bb, c, lay)
    wv = np.ascontiguousarray(want, "<f4").view("<u4").reshape(-1)
    entry = dict(kind="qmvw", bits=4, cap=None, role=role, variant={"nb": nb}, N=N, K=K, nb=nb, W=lay["W"], S=lay["S"],
                 B=lay["B"], threadgroups=lay["groups"], threads_per_group=64, base=1,
                 slot_map=dict(weights=1, x_rows=2, y_written=3), out_layout="y fp32 [nb][N]",
                 recipe=dict(builder="g17qmvw.build", layout=jsonable(lay)))
    return [dict(name=name, dir=dd, prog=prog, entry=entry, threads=64 * lay["groups"], group=64, base=1,
                 check=lambda out, wv=wv: int((np.frombuffer(out, "<u4", wv.size) != wv).sum()))]


def build_qmvw_check(work, cap, nb=4):
    """MM 25.207: every kernel of the nb-row qmvw verify step beyond the shared prefill attention: the projections and
    head, the batch-nb fp32-out norms, the fp32-out SwiGLU rows, the split-1 residual passes and the nb-row argmax."""
    import g17psum as PS
    jobs = []
    for role in ("qkv", "wo", "w1_block", "w2", "head"):
        jobs += build_qmvw(work, role, nb)
    for role, in_dtype in (("attn_norm", "half"), ("ffn_norm", "float"), ("final_norm", "half")):
        jobs += build_norm_batch(work, role, in_dtype, True, False, nb)
    jobs += build_rows_kind(work, "swiglu", nb, n=ARCH["ffn"], out32=True)
    jobs += build_argmax_rows(work, rows=nb)
    jobs += build_prefill(work, cap, kvvec=True, hw_exp2=True)[1:]          # the fp32-out attention, fast
    return jobs + _psum_rows(work, nb)


def _psum_rows(work, nb):
    """The split-1 residual passes at nb rows: wo's h = y + x16 (add16) and w2's x16 = fp16(y + h) (fold16)."""
    import g17psum as PS
    jobs = []
    rng = np.random.default_rng(97 + nb)
    for role, mode in (("wo", "add16"), ("w2", "fold16")):
        N = ARCH["d"]
        lay = PS.layout(N, 1, mode, rows=nb, pstride=nb * N)
        prog = PS.build(lay)
        parts = rng.standard_normal((1, nb, N)).astype(np.float32)
        resid = (rng.standard_normal((nb, N)).astype(np.float16) if mode == "add16" else
                 rng.standard_normal((nb, N)).astype(np.float32))
        a = bytearray(lay["a_bytes"]); O._place(a, 0, parts.reshape(-1))
        bb = bytearray(lay["b_bytes"]); O._place(bb, lay["IN"], resid.reshape(-1))
        want = PS.reference(lay, parts, resid)
        wv = want.view("<u2" if want.dtype == np.float16 else "<u4")
        dt = "<u2" if want.dtype == np.float16 else "<u4"
        name = "psum_%s_%s_b%d_sk1" % (role, mode, nb)
        import g17qsm as QS
        d = QS.author(work / name, prog, bytes(a), bytes(bb), SENT * lay["c_bytes"], lay)
        entry = dict(kind="psum", role=role, variant=dict(mode=mode, rows=nb, sk=1), N=N, sk=1, mode=mode, rows=nb,
                     pstride=lay["pstride"], IN=lay["IN"], OUT=lay["OUT"], threadgroups=lay["groups"], threads_per_group=32,
                     base=1, slot_map=dict(partials=1, residual=2, out_written=3),
                     out_layout={"add16": "h fp32 [rows][N]", "fold16": "x fp16 [rows][N]"}[mode],
                     recipe=dict(builder="g17psum.build", layout=jsonable(lay)))
        jobs.append(dict(name=name, dir=d, prog=prog, entry=entry, threads=32 * lay["groups"], group=32, base=1,
                         check=lambda out, wv=wv, dt=dt, off=lay["OUT"]: int((np.frombuffer(out, dt, wv.size, off) != wv).sum())))
    return jobs


def argmax_rows_layout(V, rows=16, per_lane=4):
    """MM 25.205: g17gen's argmax pass 1 over `rows` contiguous logit rows (the speculative verify step's 16), its chunk
    C = 32 per_lane dividing the vocabulary (Qwen3's 151,936 = 1,187 x 128). Pass 2's 256-pair cap does not apply: the
    driver reduces each row's G pairs. Indices are global (below 2^23, exact floats); row r's are r V + v."""
    C = 32 * per_lane
    if V % C or rows * V >= (1 << 23):
        raise ValueError("argmax rows: V a multiple of %d, rows x V below 2^23" % C)
    G = V // C
    return dict(op="argmax", V=V, per_lane=per_lane, C=C, G=G, PAIRS=0, TOK=0, rows=rows, batched=True,
                pairs_bytes=8 * G * rows, logits_bytes=4 * V * rows)


def build_argmax_rows(work, rows=16):
    """The rows' argmax pass 1, verified pair for pair against numpy (the larger value, ties to the smaller index)."""
    lay = argmax_rows_layout(ARCH["vocab"], rows)              # the real vocabulary: the qsm head writes N = vocab rows
    prog = G.build_pass1(lay)
    rng = np.random.default_rng(31)
    lg = rng.standard_normal(rows * lay["V"]).astype(np.float32)
    lg[5 * lay["C"] + 7] = lg[5 * lay["C"] + 9] = np.float32(9.0)      # a tie inside one chunk: the smaller index wins
    ch = lg.reshape(-1, lay["C"])
    want = np.stack([ch.max(1), (np.arange(ch.shape[0]) * lay["C"] + ch.argmax(1)).astype(np.float32)], 1).reshape(-1)
    a = lg.astype("<f4").tobytes()
    c = SENT * (-(-lay["pairs_bytes"] // 256) * 256)
    name = "argmax_rows%d_v%d" % (rows, lay["V"])
    import g17qsm as QS
    d = QS.author(work / name, prog, a, bytes(256), c, lay)          # placeholder inputs, the real ones written over them
    wv = want.astype("<f4").view("<u4")
    entry = dict(kind="argmax_rows", bits=None, cap=None, role="argmax_rows", variant={"rows": rows, "V": lay["V"]},
                 V=lay["V"], G=lay["G"],
                 C=lay["C"], threadgroups=lay["G"] * rows, threads_per_group=32, base=1,
                 slot_map=dict(logits=1, pairs_written=3), pairs_bytes=lay["pairs_bytes"],
                 recipe=dict(builder="g17gen.build_pass1", layout=jsonable(lay)))
    return [dict(name=name, dir=d, prog=prog, entry=entry, threads=32 * lay["G"] * rows, group=32, base=1,
                 check=lambda out, wv=wv: int((np.frombuffer(out, "<u4", wv.size) != wv).sum()))]


def build_prefill(work, cap, out16=False, kvvec=False, hw_exp2=False):
    """M2 (MM 25.144.2): the scalar prefill append and attention at mmax = cap. The append is verified on its written
    K/V rows and every q16 value; the attention on a region holding exactly what the verified append writes (the
    reference's), over a 0x7f sentinel. Both run M rows at p0 = cap // 2 - M // 2 (a cache half full)."""
    P = _prefill()
    import g17prefillattn_run as PR
    lay = P.prefill_layout(cap, cap, out16=out16, heads=ARCH.get("heads", 16), kv_heads=ARCH.get("kv_heads", 8))
    opts = {k: True for k, on in (("kvvec", kvvec), ("hw_exp2", hw_exp2)) if on}
    # (MM 25.207: the opts also apply to the fp32-output attention, for the qmvw check's fp32 rows)
    lay = dict(lay, **opts)
    pa, pt = P.build_prefill_append(lay), P.build_prefill_attn(lay)
    H, KVH, D = lay["heads"], lay["kv_heads"], lay["head_dim"]
    M = 64
    p0 = cap // 2 - M // 2
    qkv, cos, sin, Kc, Vc = P.case(lay, M, p0)
    want, q16, K, V = P.prefill_reference(lay, qkv, cos, sin, Kc, Vc, p0, M)
    a, b, c = P.io(lay, M, p0, qkv, cos, sin, Kc, Vc)
    rows = list(range(p0, p0 + M))
    tagx = "".join("_" + k for k in sorted(opts))
    da = PR.author(work / ("prefill_append_cap%d%s%s" % (cap, "_o16" if out16 else "", tagx)), pa, a, b, c, lay)

    def check_append(out):
        kc = np.frombuffer(out, np.float16, KVH * cap * D, lay["KOFF"]).reshape(KVH, cap, D)
        vc = np.frombuffer(out, np.float16, KVH * cap * D, lay["VOFF"]).reshape(KVH, cap, D)
        qg = np.frombuffer(out, np.float16, M * H * D, lay["Q16"]).reshape(M, H, D)
        return (int((kc[:, rows].view(np.uint16) != K[:, rows].astype(np.float16).view(np.uint16)).sum())
                + int((vc[:, rows].view(np.uint16) != V[:, rows].astype(np.float16).view(np.uint16)).sum())
                + int((qg.view(np.uint16) != q16.astype(np.float16).view(np.uint16)).sum()))
    reg = bytearray(c)
    O._place(reg, lay["KOFF"], K.astype(np.float16).reshape(-1))
    O._place(reg, lay["VOFF"], V.astype(np.float16).reshape(-1))
    O._place(reg, lay["Q16"], q16.astype(np.float16).reshape(-1))
    dt = PR.author(work / ("prefill_attn_cap%d%s%s" % (cap, "_out16" if out16 else "", tagx)), pt, a, b, bytes(reg), lay)
    wv = want.view(np.uint16).reshape(-1) if out16 else want.astype(np.float32).view(np.uint32).reshape(-1)

    def check_attn(out):
        if hw_exp2:
            # op1272 is not bit-exact to exp2_soft: each row ENCLOSES the true base-2 softmax over its keys 0 .. p0 + i
            # within decode's bound, and a base-e golden (the wrong base) must land OUTSIDE it (25.144.5's control)
            got = np.frombuffer(out, np.float16 if out16 else "<f4", M * H * D, lay["PATTN"]).astype(np.float64).reshape(M, H, D)
            Kf, Vf, qf = np.asarray(K, np.float64), np.asarray(V, np.float64), np.asarray(q16, np.float64).reshape(M, H, D)
            encl = ctrl = 0.0
            for i in range(M):
                q0 = p0 + i
                for h in range(H):
                    kv = h // (H // KVH)
                    sc = Kf[kv, :q0 + 1] @ qf[i, h]
                    for base in (2, "e"):
                        pr = np.exp2(sc - sc.max()) if base == 2 else np.exp(sc - sc.max())
                        tru = (pr @ Vf[kv, :q0 + 1]) / pr.sum()
                        err = float(np.abs(got[i, h] - tru).max() / max(np.abs(tru).max(), 1e-30))
                        if base == 2:
                            encl = max(encl, err)
                        else:
                            ctrl = max(ctrl, err)
            check_attn.enclosure = (encl, ctrl)
            return 0 if (encl <= ATTN_ENCLOSURE_BOUND and ctrl > ATTN_ENCLOSURE_BOUND) else 1
        return int((np.frombuffer(out, "<u2" if out16 else "<u4", M * H * D, lay["PATTN"]) != wv).sum())
    sfx = ("_out16" if out16 else "") + "".join("_" + k for k in sorted(opts))
    common = dict(kind="prefill_attn", cap=cap, variant=dict(dict(scalar=True, out16=True) if out16 else dict(scalar=True), **opts),
                  out_dtype="fp16" if out16 else "fp32 of the fp16-rounded value", mmax=lay["mmax"], QKVROW=lay["QKVROW"],
                  Q16=lay["Q16"], PATTN=lay["PATTN"], KOFF=lay["KOFF"], VOFF=lay["VOFF"], COST=lay["COST"], SINT=lay["SINT"],
                  rope_bytes=lay["rope_bytes"], prefill_region3_bytes=lay["prefill_region3_bytes"],
                  region3_bytes=lay["region3_bytes"])
    ja = dict(name="prefill_append_cap%d" % cap, dir=da, prog=pa, threads=M * KVH * 32, group=32, base=1, rounds=2,
              check=check_append, deliver_as="prefill_append_cap%d" % cap,
              entry=dict(common, role="append", threadgroups=lay["mmax"] * KVH, threadgroups_per_row=KVH, threads_per_group=32,
                         base=1, slot_map=dict(qkv32=1, rope=2, region3_written=3),
                         recipe=dict(builder="g17prefillattn.build_prefill_append", layout=jsonable(lay))))
    jt = dict(name="prefill_attn_cap%d%s" % (cap, sfx), dir=dt, prog=pt, threads=M * H * 1024, group=1024, base=0, rounds=2,
              check=check_attn, deliver_as="prefill_attn_cap%d%s" % (cap, sfx),
              entry=dict(common, role="attn", threadgroups=lay["mmax"] * H, threadgroups_per_row=H, threads_per_group=1024,
                         base=0, slot_map=dict(region3_written=0, qkv32=1, rope=2),
                         recipe=dict(builder="g17prefillattn.build_prefill_attn", layout=jsonable(lay))))
    return [jt] if out16 else [ja, jt]


def build_gen_batch(work, cap, nb):
    """Batched generation (MM 25.144.3): pass 1 (the unchanged program) over nb contiguous logit rows, then the batched
    step. Per sequence a different q0, chosen or forced, and a TIE (the row's maximum repeated later: the first wins);
    every sequence's q0, log, and x row are checked, and each block's words past its log stay untouched."""
    lay = G.gen_batch_layout(nb, V=ARCH["vocab_pad"], per_lane=ARCH["per_lane"], d=ARCH["d"], cap=cap)
    P1, P2 = G.build_pass1(lay), G.build_gen_batch(lay)
    V, SS, d = lay["V"], lay["SS"], lay["d"]
    rng = np.random.default_rng(90 + nb)
    emb = rng.standard_normal((4096, d)).astype(np.float16)   # the hot tokens 100 + 97 s (+ 311) up to B 32
    x = rng.standard_normal((nb, V)).astype(np.float32)
    reg = bytearray(lay["region_bytes"] + 256)
    O._place(reg, lay["R_X16"], np.full(d * nb, np.float16(np.nan)))
    want, nxts, logs = [], [], []
    for s in range(nb):
        q0 = [0, cap // 2, cap - 1, 7, 100, cap - 2, 1, 50][s % 8]
        forced = 777 if s % 3 == 2 else None
        hot = 100 + 97 * s
        x[s, hot] = 40.0
        x[s, hot + 311] = 40.0                         # a tie: the first index must win
        nxt = min(q0 + 1, cap - 1)
        log = np.full(cap, 0xFFFFFFFF, np.uint32)
        log[:nxt] = np.arange(nxt) + 100 + s
        if forced is not None:
            log[nxt] = forced
        O._place(reg, lay["GEN"] + SS * s, np.asarray([q0], "<u4"))
        O._place(reg, lay["LOG"] + SS * s, log.astype("<u4"))
        want.append(forced if forced is not None else hot)
        nxts.append(nxt)
        logs.append(log)
    tag = "gen_batch_cap%d_b%d" % (cap, nb)
    xb = x.astype("<f4").tobytes()
    # the pairs buffer padded to the logits' size: the manifest's transport view needs rows in proportion to binding 1
    d1 = author(work / (tag + "_p1"), P1, xb, bytes(256), bytes(max(lay["pairs_bytes"], 4 * len(xb))), lay)

    def phase2(outs):
        pairs = outs[tag + "_p1"][:lay["pairs_bytes"]]
        d2 = author(work / (tag + "_p2"), P2, pairs, emb.tobytes(), bytes(reg), lay)

        def check(out):
            bad = 0
            for s in range(nb):
                bad += int(np.frombuffer(out, "<u4", 1, lay["GEN"] + SS * s)[0]) != nxts[s]
                lg = np.frombuffer(out, "<u4", cap, lay["LOG"] + SS * s)
                exp = logs[s].copy()
                exp[nxts[s]] = want[s]
                bad += int((lg != exp).sum())
                bad += int(not np.array_equal(np.frombuffer(out, "<u2", d, lay["R_X16"] + 2 * d * s), emb[want[s]].view(np.uint16)))
            return bad
        j = dict(name=tag + "_p2", dir=d2, prog=P2, threads=32 * nb, group=32, base=1, check=check,
                 entry=dict(kind="gen_step", cap=cap, role="gen_step", variant=dict(batch=nb), threadgroups=nb, threads_per_group=32,
                            base=1, slot_map=dict(a=1, b=2, c_written=3), GEN=lay["GEN"], LOG=lay["LOG"], SS=SS,
                            R_X16=lay["R_X16"], region_bytes=lay["region_bytes"], PAIRS=lay["PAIRS"], batch=nb,
                            state_stride=SS, x16_stride=2 * d,
                            recipe=dict(builder="g17gen.build_gen_batch", layout=jsonable(lay))),
                 deliver_as="gen_step_cap%d_b%d" % (cap, nb))
        return [j]
    p1 = dict(name=tag + "_p1", dir=d1, prog=P1, threads=lay["G"] * 32 * nb, group=32, base=1, check=lambda out: 0,
              entry=dict(kind="gen_argmax", cap=cap, role="gen_argmax", variant=dict(batch=nb), threadgroups=lay["G"] * nb,
                         threads_per_group=32, base=1, slot_map=dict(a=1, b=2, c_written=3), GEN=lay["GEN"], LOG=lay["LOG"],
                         R_X16=lay["R_X16"], region_bytes=lay["region_bytes"], PAIRS=lay["PAIRS"], batch=nb,
                         logits_stride=4 * V, pairs_stride=8 * lay["G"],
                         recipe=dict(builder="g17gen.build_pass1", layout=jsonable(lay))),
              deliver_as="gen_argmax_pass1_cap%d_b%d" % (cap, nb))
    return [p1], phase2


MMA_NOTES = ("block-major V (VB) for cache rows before p0 is written only by this route's own append (decode does not "
             "write it): prefill a prompt in bucket chunks from position 0",
             "region 3 (decode's, extended to mma_region3_bytes) is bound at slots 1, 2 AND 3 of the attention; every "
             "offset is region-absolute")


REGO_NOTES = ("the register-O attention pairs ONLY with the register-O append (variant rego): its Q tiles are "
              "[H][nq][16][128] at QT, and the append writes no block-major V",
              "V (and K) for cache rows before p0 are read straight from the decode cache [KVH][CAP][128] at VOFF (KOFF): "
              "no block-major V precondition; a prompt may start at any bucket p0 over a decode-written cache",
              "region 3 (decode's, extended to mma_region3_bytes) is bound at slots 1, 2 AND 3 of the attention; every "
              "offset is region-absolute",
              "threads_per_group is 32 x sg (sg simdgroups per threadgroup, each owning 16 query rows)")


# THE hw_exp2 ENCLOSURE (MM 25.144.2). Per-element fp16 ulps is the wrong measure: an output near zero is a cancellation
# of +/- V terms, so its ulp spacing is tiny and even the shipped exp2_soft kernel sits up to 7 ulps from the true-exp2
# softmax there. The bound is on the ROW's scale: every element within 2^-10 (one fp16 step at the row's largest output)
# of the true-exp2 softmax, divided by that row's max |output|. Measured 2026-09-26 over the worst cap-2048 buckets
# (M 128 p0 1920, M 256 p0 1792, M 512 p0 1536, M 1024 p0 0): hw_exp2 at most 8.7e-4, exp2_soft at most 7.4e-4, the
# wrong-base (e^x) control at least 1.12. The margin under the bound is about 11 percent.
ENCLOSURE_ROW_BOUND = 2.0 ** -10
ENCLOSURE_CONTROL_MIN = 0.1


def row_scaled_error(got_u16, want_f16, M, H, D):
    """max |got - want| / max_d |want[m, h, :]| per element, as an array [M][H][D] (fp16 words in, errors out)."""
    g = np.asarray(got_u16, np.uint16).view(np.float16).astype(np.float32).reshape(M, H, D)
    w = np.asarray(want_f16, np.float16).astype(np.float32).reshape(M, H, D)
    scale = np.maximum(np.abs(w).max(axis=2, keepdims=True), np.float32(1e-30))
    return np.abs(g - w) / scale


def build_prefill_mma(work, cap, M, p0=0, with_append=True, out16=False, skip=False, rego=False, sg=1, opt=None,
                      hw_exp2=False):
    """M2's tensor-unit route (MM 25.144.2) at bucket (M, p0 = 0), its own stated order (g17prefillmma). The append
    (lay mma: Q tiles and block-major V besides the scalar outputs) is verified on K/V rows, q16, the Q tiles and VB; the
    attention on a region holding exactly what the verified append writes, bound at slots 1, 2 and 3, over a sentinel,
    with the cache past the prompt NaN (the kernel must never read it)."""
    P, MM = _prefill(), _mma()
    import g17prefillattn_run as PR
    # opt["bk"] (MM 25.162): keys per block of the register-O key loop, which sizes its scratch, so the layout takes it
    bk = int((opt or {}).get("bk", 16))
    lay = MM.mma_layout(cap, M, p0, out16=out16, **(dict(rego=True, sg=sg, bk=bk) if rego else {}))
    if skip:
        lay = dict(lay, runtime_trips=True)
    opt = dict(opt or {})
    if opt:
        # register-O code-size options (MM 25.144.2), every one value-preserving: fold (M8's fold_offsets), scale
        # (tensor_acc_scale), hoist (hoist_prologue), row2 (the split, immediate-addressed row stage), holdk (the exp2
        # constants held across the loop)
        lay = dict(lay, **opt)
    if hw_exp2:
        # THE HARDWARE exp2 (op1272) row stage (MM 25.144.2): not bit-exactly reproducible on the CPU, so this bucket
        # is checked against an ENCLOSURE of the true-exp2 softmax, not bitwise (see check_attn); register-O, fp16 out
        if not (rego and out16):
            raise ValueError("hw_exp2 is built on the register-O route with fp16 output only")
        lay = dict(lay, hw_exp2=True)
    pa, pm = P.build_prefill_append(lay), (MM.build_mma_sreg if lay.get("sreg") else
                                           MM.build_mma_rego if rego else MM.build_mma)(lay)
    H, KVH, D, nq = lay["heads"], lay["kv_heads"], lay["head_dim"], lay["nq"]
    # cache rows before p0: finite (an earlier chunk's); from p0 on: NaN (the prompt's rows are written by the append,
    # the rest must never be read)
    qkv, cos, sin, Kc, Vc = P.case(lay, M, p0, seed=11 + p0)
    if hw_exp2:
        # hw_exp2 makes no NaN-tail claim, so its cache past the prompt is ZERO - what the graph guarantees anyway
        # (the prefill_mma precondition kv_zeroed)
        Kc = np.asarray(Kc, np.float16).copy(); Vc = np.asarray(Vc, np.float16).copy()
        Kc[:, p0:] = np.float16(0.0); Vc[:, p0:] = np.float16(0.0)
    q16, K, V = P.rope_append_reference(lay, qkv, cos, sin, Kc, Vc, p0, M)
    if hw_exp2:
        import g17decodestep as DS
        want = MM.mma_prefill_reference(lay, q16, K, V, expf=lambda x: DS.exp2(np.asarray(x, np.float32)))
        want_wrong = MM.mma_prefill_reference(lay, q16, K, V,
                                              expf=lambda x: np.exp(np.asarray(x, np.float32)).astype(np.float32))
    else:
        want = MM.mma_prefill_reference(lay, q16, K, V)
    a, b, c0 = P.io(lay, M, p0, qkv, cos, sin, Kc, Vc)
    c = bytearray(lay["mma_region3_bytes"]); c[:len(c0)] = c0
    qr = q16.astype(np.float16).reshape(nq, 16, H, D)
    qt = (qr.transpose(2, 0, 1, 3) if rego else
          np.concatenate([qr[:, :, 0::2].transpose(2, 0, 1, 3), qr[:, :, 1::2].transpose(2, 0, 1, 3)], axis=2))
    vb = V.astype(np.float16).reshape(KVH, cap // 16, 16, 8, 16).transpose(0, 1, 3, 2, 4)
    if p0 and not rego:
        # the earlier chunk's block-major V (its own append wrote it), blocks before p0 / 16
        tmp = bytearray(len(c)); O._place(tmp, lay["VB"], vb.reshape(-1))
        n0 = p0 // 16 * 8 * 256 * 2
        for kvh in range(KVH):
            lo = lay["VB"] + kvh * cap * D * 2
            c[lo:lo + n0] = tmp[lo:lo + n0]
    rows = list(range(p0, p0 + M))
    rg = "rego_" if rego else ""
    da = PR.author(work / ("prefill_mma_%sappend_cap%d_m%d_p%d" % (rg, cap, M, p0)), pa, a, b, bytes(c), lay)

    def check_append(out):
        kc = np.frombuffer(out, np.float16, KVH * cap * D, lay["KOFF"]).reshape(KVH, cap, D)
        got_qt = np.frombuffer(out, np.float16, qt.size, lay["QT"]).reshape(qt.shape)
        n = (int((kc[:, rows].view(np.uint16) != K[:, rows].astype(np.float16).view(np.uint16)).sum())
             + int((got_qt.view(np.uint16) != qt.view(np.uint16)).sum()))
        if not rego:
            got_vb = np.frombuffer(out, np.float16, KVH * cap * D, lay["VB"]).reshape(KVH, cap // 16, 8, 16, 16)
            blk = slice(p0 // 16, lay["NB"])
            n += int((got_vb[:, blk].view(np.uint16) != vb[:, blk].view(np.uint16)).sum())
        return n
    reg = bytearray(c)
    O._place(reg, lay["KOFF"], K.astype(np.float16).reshape(-1))
    O._place(reg, lay["VOFF"], V.astype(np.float16).reshape(-1))
    O._place(reg, lay["Q16"], q16.astype(np.float16).reshape(-1))
    O._place(reg, lay["QT"], qt.reshape(-1))
    if not rego:
        O._place(reg, lay["VB"], vb.reshape(-1))
    reg = bytes(reg)
    sfx = ("" if p0 == 0 else "_p%d" % p0) + ("_out16" if out16 else "") + ("_skip" if skip else "") + (
        "_sg%d" % sg if rego and sg > 1 else "") + ("_" + "_".join(sorted(opt)) if opt else "") + (
        "_hwexp2" if hw_exp2 else "")
    dm = PR.author(work / ("prefill_mma_%scap%d_m%d%s" % (rg, cap, M, sfx)), pm, reg, reg, reg, lay)
    wv = want.view(np.uint16).reshape(-1) if out16 else want.astype(np.float32).view(np.uint32).reshape(-1)

    def check_attn(out):
        got = np.frombuffer(out, "<u2" if out16 else "<u4", M * H * D, lay["PATTN"])
        if hw_exp2:
            # every element within ENCLOSURE_ROW_BOUND of the true-exp2 softmax at its row's scale, AND the check must be
            # able to fail: against the wrong-base (e^x) golden the same output must land far outside, or it is refused
            inside = int((row_scaled_error(got, want, M, H, D) > ENCLOSURE_ROW_BOUND).sum())
            control = float(row_scaled_error(got, want_wrong.astype(np.float16), M, H, D).max())
            return inside if control > ENCLOSURE_CONTROL_MIN else M * H * D
        return int((got != wv).sum())
    common = dict(kind="prefill_mma", cap=cap, M=M, p0=p0, NB=lay["NB"], QKVROW=lay["QKVROW"], Q16=lay["Q16"], QT=lay["QT"],
                  VB=lay["VB"], SCR=lay["SCR"], PATTN=lay["PATTN"], KOFF=lay["KOFF"], VOFF=lay["VOFF"], COST=lay["COST"],
                  SINT=lay["SINT"], rope_bytes=lay["rope_bytes"], mma_region3_bytes=lay["mma_region3_bytes"],
                  region3_bytes=lay["region3_bytes"], preconditions=["kv_zeroed"],
                  contract_notes=list(REGO_NOTES if rego else MMA_NOTES),
                  out_dtype="fp16" if out16 else "fp32 of the fp16-rounded value")
    ja = dict(name="prefill_mma_%sappend_cap%d_m%d%s" % (rg, cap, M, sfx), dir=da, prog=pa, threads=M * KVH * 32, group=32, base=1,
              rounds=2, check=check_append, deliver_as="prefill_mma_%sappend_cap%d_m%d" % (rg, cap, M),
              entry=dict(common, role="append", variant=dict(dict(mma=True, M=M), **({"rego": True} if rego else {})),
                         threadgroups=M * KVH, threads_per_group=32,
                         base=1, slot_map=dict(qkv32=1, rope=2, region3_written=3),
                         recipe=dict(builder="g17prefillattn.build_prefill_append", layout=jsonable(lay))))
    tpg = 32 * (sg if rego else 1)
    jm = dict(name="prefill_mma_%scap%d_m%d%s" % (rg, cap, M, sfx), dir=dm, prog=pm, threads=lay["grid"] * tpg, group=tpg, base=1,
              rounds=2, check=check_attn, deliver_as="prefill_mma_%scap%d_m%d%s" % (rg, cap, M, sfx),
              entry=dict(common, role="attn", variant=dict(dict(mma=True, M=M, p0_block=p0 // 16), **({"out16": True} if out16 else {}),
                                                           **({"skip": True} if skip else {}),
                                                           **({"rego": True, "sg": sg} if rego else {}),
                                                           **opt, **({"hw_exp2": True} if hw_exp2 else {})),
                         threadgroups=lay["grid"], threads_per_group=tpg,
                         base=1, slot_map=dict(region3_as_a=1, region3_as_b=2, region3_written=3),
                         recipe=dict(builder="g17prefillmma.build_mma_sreg" if lay.get("sreg") else
                                     "g17prefillmma.build_mma_rego" if rego else "g17prefillmma.build_mma",
                                     layout=jsonable(lay))))
    if not with_append:
        ja["entry"] = None                      # the append program does not depend on p0: delivered once per M
    return [ja, jm]


# ------------------------------------------------------------------------------------------------ build / check
class _Code:
    def __init__(self, code):
        self.code = code


QMM_M = (128, 256, 512)
QMM_SWIGLU_M = QMM_M            # every gemm_M the graph slices by (g17swigluqmm.layout: 32 columns a threadgroup)


def build_qmm_swiglu(work, bits, M):
    """MM 25.144.12: w3's quantized-prefill GEMM with the SwiGLU fused into its tail (tools/g17swigluqmm.py), verified
    ONCE on hardware over a 0x7f sentinel: U (slot 3 at U_OFF, the GEMM's own output) against the pinned MMA model of
    x @ W16, and act (slot 3 at ACT_OFF) against g17rows' swiglu of (G, that U), both bitwise. W16 is the dequant
    reference of w3's quantized weights (bit-identical to qmm_dequant's output, M1's receipts); G, w1's gate, is a
    random fp32 input placed in slot 3 at G_OFF (the graph binds w1's C there)."""
    import g17qmm as QMM
    import g17swigluqmm as SW
    lay = SW.layout(M)
    prog = SW.build(lay)
    lay = dict(lay, hold=prog.hold)
    N, K = lay["N"], lay["K"]
    packed, s16, b16, q = QMM.weights(N, K, bits, 41 + bits + len("w3"))
    w16 = QMM.dequant_reference(q, s16, b16, bits)                        # fp16 [K][N]
    rng = np.random.default_rng(900 + M + bits)
    x16 = np.asarray(rng.standard_normal((M, K)), np.float16)
    g = (rng.standard_normal((M, N)) * 3).astype(np.float32)
    u_want, act_want = SW.reference(lay, x16, w16, g)
    c = bytearray(SENT * lay["c_bytes"])
    O._place(c, lay["G_OFF"], g.reshape(-1))
    name = "qmm_swiglu_q%d_w3_m%d" % (bits, M)
    d = author(work / name, prog, x16.astype("<f2").tobytes(), w16.astype("<f2").tobytes(), bytes(c), lay)

    def check(out):
        u = np.frombuffer(out, "<u4", M * N, lay["U_OFF"])
        act = np.frombuffer(out, "<u2", M * N, lay["ACT_OFF"])
        return (int((u != u_want.view(np.uint32).reshape(-1)).sum()) +
                int((act != np.asarray(act_want, np.float16).view(np.uint16).reshape(-1)).sum()))
    entry = dict(kind="qmm_swiglu", bits=bits, role="w3", variant=dict(M=M, swiglu=True), M=M, N=N, K=K,
                 G_OFF=lay["G_OFF"], U_OFF=lay["U_OFF"], ACT_OFF=lay["ACT_OFF"], status="provisional-speed",
                 threadgroups=lay["threadgroups"], threads_per_group=lay["threads_per_group"], base=1,
                 slot_map=dict(x=1, weights=2, region_written=3),
                 out_layout="slot 3: U fp32 [M][N] at U_OFF (scratch), act fp16 [M][N] at ACT_OFF = fp16(silu(G) U)",
                 regions=dict(slot1=dict(A=dict(offset=0, bytes=2 * M * K, layout="fp16 [M][K] rows")),
                              slot2=dict(W16=dict(offset=0, bytes=2 * K * N, layout="fp16 [K][N] rows")),
                              slot3=dict(U=dict(offset=lay["U_OFF"], bytes=4 * M * N, layout="fp32 [M][N] rows"),
                                         G=dict(offset=lay["G_OFF"], bytes=4 * M * N, layout="fp32 [M][N] rows, w1's C"),
                                         ACT=dict(offset=lay["ACT_OFF"], bytes=2 * M * N, layout="fp16 [M][N] rows"))),
                 loop_form=dict(kloop_unroll=lay["kloop_unroll"], kloop_bases=lay["kloop_bases"], hold=lay["hold"]),
                 recipe=dict(builder="g17swigluqmm.build", layout=jsonable(lay)))
    return [dict(name=name, dir=d, prog=prog, entry=entry, threads=lay["threadgroups"] * lay["threads_per_group"],
                 group=lay["threads_per_group"], base=1, check=check)]


def build_ffn16(work, bits):
    """THE FP16 FFN INTERMEDIATE (MM 25.183): w1's (and w3's) prefill GEMM storing fp16 through the 'half' epilogue
    (fp32 accumulate, op1016 RNE; the receipted K-loop launches, agxforge.g17.runtime._KLOOP_HALF_RECEIPTED), and the
    SwiGLU rows reading fp16 gate and up - 6 bytes an element instead of 10, as MLX's fp16 GEMM outputs are. The GEMM is
    run on hardware against the pinned MMA model's narrowing here (the qmm route's worker)."""
    import g17qmm as QMM
    import g17tensorcommonruntime as R
    jobs = []
    N, K = QMM.ROLES["w1"]
    for M in QMM_M:
        spec = dict(QMM.role_spec("w1", M), epilogue=["half"])
        sg, gn, sk, tg = spec["simdgroups"], spec["grid_n"], spec["split_k"], spec["threadgroups"]
        gname = "qmm_q%d_w1_m%d_gemm_half" % (bits, M)
        g = work / gname
        if g.exists():
            shutil.rmtree(g)
        R.author_generic(g, spec)
        rng = np.random.default_rng(M + bits + 1000)
        (g / "a.f16").write_bytes(np.asarray(rng.standard_normal((M, K)), np.float16).astype("<f2").tobytes())
        (g / "b.f16").write_bytes(np.asarray(rng.standard_normal((K, N)) * 0.05, np.float16).astype("<f2").tobytes())
        rep = R.run(g, queries=1, composition="generic")
        gm = 0 if rep["status"] == "passed" else max(1, int(rep["queries"][0].get("mismatched_elements") or 1))
        gext = {s_: (g / f).stat().st_size for s_, f in (("slot1", "a.f16"), ("slot2", "b.f16"), ("slot3", "c.f32"))}
        jobs.append(dict(name=gname, dir=g, prog=_Code((g / "program.bin").read_bytes()), prechecked=True, mismatch=gm,
                         entry=dict(kind="qmm", bits=bits, role="w1", variant=dict(M=M, half=True), M=M, N=N, K=K,
                                    split_k=sk, status="provisional-speed", threadgroups=tg * gn * sk,
                                    threads_per_group=32 * sg, base=1, slot_extents_bytes=gext,
                                    slot_map=dict(x=1, weights=2, c_written=3),
                                    A_layout="x fp16 [M][K] rows at byte 0 of slot 1",
                                    B_layout="W16 fp16 [K][N] rows at byte 0 of slot 2 (qmm_dequant's output)",
                                    out_layout="y fp16 [M][N] rows (fp32 accumulate, RNE) at byte 0 of slot 3",
                                    recipe=dict(builder="g17tensorcommonruntime.gemm_generic", layout=jsonable(spec)))))
        jobs += build_rows_kind(work, "swiglu", M, in16=True)
        jobs += build_rows_kind(work, "swiglu", M, in16=True, fast=True)
    return jobs


def build_qmm(work, bits):
    """M1's quantized GEMM, verified here end to end (these bundles are PRECHECKED, not re-dispatched): per role the
    dequant through the runner against g17qmm.dequant_reference, then per M the gemm_generic body on that dequant's
    OWN output through the common worker, whose reference is the pinned MMA model."""
    import g17qmm as QMM
    import g17tensorcommonruntime as R
    jobs = []
    for role in ("qkv", "wo", "w1", "w2"):
        N, K = QMM.ROLES[role]
        lay = QMM.dequant_layout(N, K, bits)
        packed, s16, b16, q = QMM.weights(N, K, bits, 41 + bits + len(role))
        prog = QMM.build_dequant(lay)
        a, bbuf, c = QMM.dequant_io(lay, packed, s16, b16)
        name = "qmm_q%d_%s_dequant" % (bits, role)
        d = author(work / name, prog, a, bbuf, c, lay)
        run = work / (name + "_run")
        run.mkdir(exist_ok=True)
        got = dispatch([dict(tag="dq", dir=d, threads=lay["threads"], group=32, base=1)], run)["dq"]
        if got is None:                            # emulated and refused: the GEMMs below cannot be fed from it
            EMULATE["refused"][name] = EMULATE["refused"].pop("dq")
            continue
        w16 = np.frombuffer(got, "<u2", K * N, lay["OUT"]).reshape(K, N)
        mism = int((w16 != QMM.dequant_reference(q, s16, b16, bits).view("<u2")).sum())
        ext = {s: (d / f).stat().st_size for s, f in (("slot1", "a.f16"), ("slot2", "b.f16"), ("slot3", "c.f32"))}
        pw = 32 // bits
        jobs.append(dict(name=name, dir=d, prog=prog, prechecked=True, mismatch=mism, entry=dict(
            kind="qmm_dequant", bits=bits, role=role, variant={}, status="provisional-speed",
            threadgroups=lay["threads"] // 32, threads_per_group=32, threads=lay["threads"],
            base=1, slot_map=dict(unused=1, weights=2, c_written=3), N=N, K=K, W=lay["W"], S=lay["S"], B=lay["B"],
            OUT=lay["OUT"], out_layout="W16 fp16 [K][N] row-major at OUT (bytes) of slot 3",
            regions=dict(
                slot2=dict(W=dict(offset=lay["W"], bytes=N * (K // pw) * 4, layout="u32 packed [N][K/%d], row n contiguous" % pw),
                           S=dict(offset=lay["S"], bytes=N * (K // 64) * 2, layout="bf16 scales [N][K/64]"),
                           B=dict(offset=lay["B"], bytes=N * (K // 64) * 2, layout="bf16 biases [N][K/64]")),
                slot3=dict(W16=dict(offset=lay["OUT"], bytes=K * N * 2, layout="fp16 [K][N] row-major"))),
            slot_extents_bytes=ext,
            recipe=dict(builder="g17qmm.build_dequant", layout=jsonable(lay)))))
        for M in QMM_M:
            spec = QMM.role_spec(role, M)
            sg, gn, sk = spec["simdgroups"], spec["grid_n"], spec["split_k"]
            tg = spec["threadgroups"]                  # row groups (MM 25.157): the launch is tg x gn x sk
            gname = "qmm_q%d_%s_m%d_gemm" % (bits, role, M)
            g = work / gname
            if g.exists():
                shutil.rmtree(g)
            R.author_generic(g, spec)
            rng = np.random.default_rng(M + bits)
            (g / "a.f16").write_bytes(np.asarray(rng.standard_normal((M, K)), np.float16).astype("<f2").tobytes())
            (g / "b.f16").write_bytes(np.asarray(w16).tobytes())
            if EMULATE is not None:                # R.run dispatches on its own: emulate against its CPU reference
                import g17emu as EMU
                try:
                    out, _m = EMU.run_bundle(g, tg * gn * sk * 32 * sg, 32 * sg, 1, tier="wp")
                    if _m.admitted:
                        EMULATE["admitted"][gname] = dict(_m.admitted)
                except EMU.Refused as e:
                    EMULATE["refused"][gname] = str(e)
                    continue
                exp = np.asarray(R.generic_reference(g, R.generic_spec(json.loads((g / "generic.json").read_text()))),
                                 "<f4")
                got = np.frombuffer(out, "<f4")[:exp.size].reshape(exp.shape)
                gm = int((got.view("<u4") != exp.view("<u4")).sum())
            else:
                rep = R.run(g, queries=1, composition="generic")
                gm = 0 if rep["status"] == "passed" else max(1, int(rep["queries"][0].get("mismatched_elements") or 1))
            gext = {s: (g / f).stat().st_size for s, f in (("slot1", "a.f16"), ("slot2", "b.f16"), ("slot3", "c.f32"))}
            jobs.append(dict(name=gname, dir=g, prog=_Code((g / "program.bin").read_bytes()), prechecked=True, mismatch=gm,
                             entry=dict(kind="qmm", bits=bits, role=role, variant=dict(M=M), M=M, N=N, K=K, split_k=sk,
                                        status="provisional-speed",
                                        loop_form=dict(kloop_unroll=spec.get("kloop_unroll", 1),
                                                       kloop_bases=bool(spec.get("kloop_bases", False))),
                                        threadgroups=tg * gn * sk, threads_per_group=32 * sg, base=1,
                                        regions=dict(slot1=dict(A=dict(offset=0, bytes=M * K * 2, layout="fp16 [M][K] rows")),
                                                     slot2=dict(W16=dict(offset=0, bytes=K * N * 2, layout="fp16 [K][N] rows")),
                                                     slot3=dict(C=dict(offset=0, bytes=M * N * 4 * sk,
                                                                       layout="fp32 [M][N] rows" if sk == 1 else
                                                                       "fp32 [2M][N]: two K-half partials"))),
                                        slot_extents_bytes=gext,
                                        slot_map=dict(x=1, weights=2, c_written=3),
                                        A_layout="x fp16 [M][K] rows at byte 0 of slot 1",
                                        B_layout="W16 fp16 [K][N] rows at byte 0 of slot 2 (qmm_dequant's output)",
                                        out_layout=("y fp32 [M][N] rows (ldc = N) at byte 0 of slot 3" if sk == 1 else
                                                    "partials fp32 [2M][N]: rows [0,M) K-half 0, [M,2M) K-half 1; "
                                                    "y = p0 + p1 (fp32, ascending)"),
                                        recipe=dict(builder="g17tensorcommonruntime.gemm_generic",
                                                    layout=jsonable(spec)))))
    return jobs


def plan(bits, cap, kinds, work, attn_variants=("",)):
    """(phase-1 jobs, phase-2 builders): every bundle of one build."""
    jobs, later = [], []
    if "qmv" in kinds:
        for role in ("qkv", "wo_res1", "ffn", "w2_res2"):
            S = KSPLIT[bits][role]
            for variant in ({}, dict(ks=S), dict(ks=S, lean=True), dict(ks=S, lean=True, ptr=True)):
                jobs += build_qmv(work, bits, role, variant)
            if bits == 4:
                # q4: the chain form (MM 25.144.4). w2 measured 37.2 us against the delivered 41.0; qkv, wo and ffn were
                # added for the whole-graph A/B of MM 25.153 (85 against 109 loop instructions, 145 against 174 for the
                # fused FFN). q8 is stream-bound and level, so it builds none.
                jobs += build_qmv(work, bits, role, dict(ks=S, lean=True, ptr=True, chain=True))
    if "qmv_ksup" in kinds and bits == 4:
        # MM 25.190: the chain form at the next split-K up (more simdgroups a row), for an in-graph A/B at 1K
        for role, S in (("qkv", 4), ("wo_res1", 4), ("ffn", 4), ("w2_res2", 8)):
            jobs += build_qmv(work, bits, role, dict(ks=S, lean=True, ptr=True, chain=True))
    if "qmv_batch" in kinds:
        for role in ("qkv", "wo_res1", "ffn", "w2_res2"):
            for nb in BATCHES:
                jobs += build_qmv_batch(work, bits, role, nb)
            jobs += build_qmv_batch(work, bits, role, 8, pv=4)          # B 8 as two passes of 4 (MM 25.144.3)
            for nb in BATCHES:
                jobs += build_qmv_batch(work, bits, role, nb, dq=True)   # dequantize once per pass (MM 25.144.3)
            jobs += build_qmv_batch(work, bits, role, 8, pv=4, dq=True)
            jobs += build_qmv_batch(work, bits, role, 1, dq=True)       # the single runs of a dq graph (--check)
    if "norm_batch" in kinds:
        # the final norm too (fp16 out at q8 for the x16 head, fp32 at q4), for the batched decode graph's head
        # (q8 fp32 out as well: the batched x32 head reads it, MM 25.144.7)
        for role, in_dtype, out32 in (("attn_norm", "half", True), ("ffn_norm", "float", True),
                                      ("final_norm", "half", bits == 4)) + ((("final_norm", "half", True),) if bits == 8 else ()):
            for seed in (False, True):
                for nb in BATCHES:
                    jobs += build_norm_batch(work, role, in_dtype, out32, seed, nb)
    if "norm_rows" in kinds:
        # THE SAME batched norm for PREFILL (MM 25.144.3): M prompt rows, one threadgroup each; variant {"batch": M}
        # fp32 out (the fp32-x qmv path) and fp16 out (M1's qmm reads fp16 A)
        for role, in_dtype, out32 in (("attn_norm", "half", True), ("ffn_norm", "float", True),
                                      ("attn_norm", "half", False), ("ffn_norm", "float", False)):
            for seed in (False, True):
                for nb in PREFILL_ROWS:
                    jobs += build_norm_batch(work, role, in_dtype, out32, seed, nb)
    for rk in ("swiglu", "residual", "fold"):
        if rk + "_rows" in kinds:
            for M in PREFILL_ROWS:
                jobs += build_rows_kind(work, rk, M)
    if "attn_batch" in kinds:
        for nb in BATCHES:
            jobs += build_attn_batch(work, cap, nb)
    if "attn_batch_gqa" in kinds:
        for nb in BATCHES:
            jobs += build_attn_batch(work, cap, nb, gqa=True)
    if "gen_batch" in kinds:
        for nb in BATCHES:
            p1, phase2 = build_gen_batch(work, cap, nb)
            jobs += p1
            later.append(phase2)
    if "head" in kinds:
        for variant in HEAD_VARIANTS[bits]:
            if variant.get("argmax") and ARCH["vocab"] % 384:
                continue                     # the fused argmax chunks by 384; Qwen3's 151,936 is not a multiple
            jobs += build_qmv(work, bits, "head", variant, head=True)
    if "head_batch" in kinds:
        for nb in BATCHES:
            jobs += build_head_batch(work, bits, nb)
        jobs += build_head_batch(work, bits, 8, pv=4)                   # B 8 as two passes of 4 (the M3 form)
        for nb in (4, 8):
            jobs += build_head_batch(work, bits, nb, S=4)               # four K slices: half the rows per simdgroup
    if "norm" in kinds:
        # q8's final norm both ways: fp16 out for the fp16-x head, fp32 out for the x32 head (MM 25.144.7)
        finals = (True,) if bits == 4 else (False, True)
        for role, in_dtype, out32 in (("attn_norm", "half", True), ("ffn_norm", "float", True)) + \
                tuple(("final_norm", "half", o) for o in finals):
            for seed in (False, True):
                jobs += build_norm(work, role, in_dtype, out32, seed)
    if "attn" in kinds:
        for flags in attn_variants:
            jobs += build_attn(work, cap, flags)
    if "prefill_attn" in kinds:
        jobs += build_prefill(work, cap) + build_prefill(work, cap, out16=True)
    if "prefill_attn_fast" in kinds:
        # MM 25.205: the verify step's attention with decode's kvvec loads and hardware exp2 (out16; hw_exp2 enclosure-checked)
        jobs += build_prefill(work, cap, out16=True, kvvec=True) + build_prefill(work, cap, out16=True, kvvec=True, hw_exp2=True)
    if "prefill_mma" in kinds:
        # chunk boundaries (Piece A chunks prompts by the GEMM's M cap): p0 = 0 and every multiple of 512 the bucket fits
        # buckets: M 128 at every p0 multiple of 128 (Piece A's cheapest chunk per row), M 256 at 0 / 512, M 512 at every
        # multiple of 512, M 1024 at 0
        buckets = sorted({(128, p0) for p0 in range(0, cap - 127, 128)} |
                         {(256, p0) for p0 in (0, 512) if p0 + 256 <= cap} |
                         {(512, p0) for p0 in range(0, cap - 511, 512)} |
                         ({(1024, 0)} if cap >= 1024 else set()))
        skip_ok = _skip_supported()
        for M, p0 in buckets:
            if M <= cap:
                    for o16 in (False, True):
                        jobs += build_prefill_mma(work, cap, M, p0, with_append=(p0 == 0 and not o16), out16=o16)
                        if skip_ok:
                            # THE CAUSAL SKIP (a capped runtime trip count; needs cc from M8's tensor-loop work)
                            jobs += build_prefill_mma(work, cap, M, p0, with_append=False, out16=o16, skip=True)
    if "prefill_mma_rego" in kinds:
        # THE REGISTER-O ROUTE with the causal skip (MM 25.144.2): the same buckets as prefill_mma, plus M 2048 at p0 0
        # on 4 simdgroups (one simdgroup would need 256 slices); needs PR #265's cc
        if not _rego_supported():
            raise SystemExit("prefill_mma_rego: this checkout's cc does not compile the register-O route (needs PR #265)")
        opt = _rego_options()
        # M 256 at EVERY multiple of 256: M1's 2 x 2 GEMM forms exist only at M 256 (25.144.1), so Piece A chunks a
        # prompt by 256, and a 1,024-token prompt starts chunks at 256 and 768 as well as 0 and 512
        buckets = sorted({(128, p0) for p0 in range(0, cap - 127, 128)} |
                         {(256, p0) for p0 in range(0, cap - 255, 256)} |
                         {(512, p0) for p0 in range(0, cap - 511, 512)} |
                         ({(1024, 0)} if cap >= 1024 else set()) |
                         ({(2048, 0)} if cap >= 2048 else set()))
        for M, p0 in buckets:
            if M <= cap:
                for o16 in (False, True):
                    jobs += build_prefill_mma(work, cap, M, p0, with_append=(p0 == 0 and not o16), out16=o16, skip=True,
                                              rego=True, sg=4 if M == 2048 else 1)
                    if opt:
                        # the same bucket with the value-preserving code-size options: the same values, fewer
                        # instructions a trip
                        jobs += build_prefill_mma(work, cap, M, p0, with_append=False, out16=o16, skip=True,
                                                  rego=True, sg=4 if M == 2048 else 1, opt=opt)
    if "qsm_pad" in kinds:
        # MM 25.196: the FFN zero-padded to the prefill's power-of-two width so w1 / w3 split (Qwen3's 3,072 -> 4,096),
        # every projection at its chained-best split-K, the fp16 dequant, and the passes at those splits, B 16
        pad = ARCH.get("ffn_prefill", ARCH["ffn"])
        jobs += build_qsm_batch(work, bits, 16, occ=True, h16=True, pad=pad, sks=QSM_SK_PAD)
    if "qmvw_check" in kinds:
        # MM 25.207: the nb = 4 verify step on the wide multi-vector projection
        jobs += build_qmvw_check(work, cap)
    if "argmax_rows" in kinds:
        # MM 25.205: the speculative verify step's 16-row argmax pass 1 on the GPU
        jobs += build_argmax_rows(work)
    if "qsm_prefill" in kinds:
        # MM 25.202: the short-prompt prefill's w1 / w3 qsm on a STANDALONE block (the prefill's own qmm_dequant block:
        # W / S / B at g17qsm's default offsets), the fp16 dequant, at the occupancy split; qkv / wo / w2 are qsm_h16's
        jobs += build_qsm_batch(work, bits, 16, occ=True, h16=True, sks={"w1_block": _qsm_occ()["w1"]},
                                roles={"w1_block": (ARCH["ffn"], ARCH["d"], _qsm_occ()["w1"], None)})
    if "norm_rows16" in kinds:
        # the verify step's 16-row norms alone (MM 25.203: attention, FFN and final norms, fp16 out, no seed), for an arch
        # whose qsm_batch kind does not build (Qwen3-8B)
        for role, in_dtype in (("attn_norm", "half"), ("ffn_norm", "float"), ("final_norm", "half")):
            jobs += build_norm_batch(work, role, in_dtype, False, False, 16)
    if "psum_occ16" in kinds:
        # the occupancy psum passes alone at 16 rows (the verify step's, MM 25.203), at the arch's split-K and widths:
        # an arch whose other occupancy qsm forms do not build (Qwen3-8B) still gets the passes its h16 route needs
        jobs += build_qsm_batch(work, bits, 16, with_qsm=False, occ=True)
    if "qsm_h16" in kinds:
        # MM 25.196: the occupancy forms with the dequant on the fp16 pipe (op798 into a half A accumulator), 16 rows
        jobs += build_qsm_batch(work, bits, 16, occ=True, h16=True)
    if "qsm_occ" in kinds:
        # MM 25.185: the occupancy split-K forms (16 rows, built once) and their psum passes at B 16 and 8
        for B in (16, 8, 32):
            jobs += build_qsm_batch(work, bits, B, with_qsm=(B != 8), occ=True)
    if "qsm_batch" in kinds:
        # the batched decode on the tensor units (MM 25.172), at B 8 and 16
        for B in ((16, 8, 32) if ARCH["name"] == "internlm2" else (16, 8)):
            jobs += build_qsm_batch(work, bits, B, with_qsm=(B != 8))
            # the batched norms with fp16 out: g17qsm reads the x16 rows
            for role, in_dtype in (("attn_norm", "half"), ("ffn_norm", "float"), ("final_norm", "half")):
                jobs += build_norm_batch(work, role, in_dtype, False, False, B)
        # B 16's and 32's other batched ops (the batched decode's BATCHES stop at 8): the attention, the generation step;
        # at 16 also the qmv batched head and its fp32-out final norm
        for B in ((16, 32) if ARCH["name"] == "internlm2" else (16,)):
            jobs += build_attn_batch(work, cap, B)
            p1, phase2 = build_gen_batch(work, cap, B)
            jobs += p1
            later.append(phase2)
        if ARCH["name"] == "internlm2":            # the qmv batched head; another arch's batched graph runs the qsm head
            jobs += build_head_batch(work, bits, 16, pv=4)
            jobs += build_norm_batch(work, "final_norm", "half", True, False, 16)
    if "prefill_mma_sreg" in kinds:
        # THE REGISTER-S route (MM 25.163): S and P in registers (the ninth accumulator group), rows reduced by simd
        # shuffles, P fed to PV as A from registers. Its own stated order (mma_prefill_reference with sreg), bit-exact
        # with exp2_soft; with hw_exp2 enclosure-checked. The 512-row chunks the prefill graph runs, fp16 output.
        opt = dict(fold=True, holdk=True, sreg=True)
        for p0 in range(0, cap - 511, 512):
            for hw in (False, True):
                jobs += build_prefill_mma(work, cap, 512, p0, with_append=False, out16=True, skip=True, rego=True,
                                          opt=opt, hw_exp2=hw)
    if "prefill_mma_bk32" in kinds:
        # THE 32-KEY BLOCK register-O route (MM 25.162): the code-size options plus bk 32 - half the key-loop trips, one
        # O rescale and one max/sum per 32 keys. Its own stated order (mma_prefill_reference at bk 32), bit-exact with
        # exp2_soft; with hw_exp2 it is enclosure-checked like prefill_mma_hwexp2. The 512-row chunks the prefill graph
        # runs, fp16 output.
        if not _rego_supported():
            raise SystemExit("prefill_mma_bk32: this checkout's cc does not compile the register-O route")
        opt = dict(_rego_options(), bk=32)
        if not opt.get("row2"):
            raise SystemExit("prefill_mma_bk32: needs the row2 stage (this checkout's IR lacks the code-size options)")
        for p0 in range(0, cap - 511, 512):
            for hw in (False, True):
                jobs += build_prefill_mma(work, cap, 512, p0, with_append=False, out16=True, skip=True, rego=True,
                                          opt=opt, hw_exp2=hw)
    if "prefill_mma_hwexp2" in kinds:
        # THE HARDWARE-exp2 REGISTER-O ROUTE (MM 25.144.2, PR #289): opt-in, NOT bit-exact - each bucket is checked
        # within ENCLOSURE_ROW_BOUND of the true-exp2 softmax (with a wrong-base control that must fail), and it ships on
        # model tokens (token-identical to the exact kernel at 128 and 1,024 tokens, q4 and q8, 2026-09-26). The
        # rego buckets, fp16 output, with the code-size options.
        if not _rego_supported():
            raise SystemExit("prefill_mma_hwexp2: this checkout's cc does not compile the register-O route")
        opt = _rego_options()
        buckets = sorted({(128, p0) for p0 in range(0, cap - 127, 128)} |
                         {(256, p0) for p0 in range(0, cap - 255, 256)} |
                         {(512, p0) for p0 in range(0, cap - 511, 512)} |
                         ({(1024, 0)} if cap >= 1024 else set()))
        for M, p0 in buckets:
            if M <= cap:
                jobs += build_prefill_mma(work, cap, M, p0, with_append=False, out16=True, skip=True, rego=True,
                                          opt=opt, hw_exp2=True)
    if "gen" in kinds:
        for batched in (False, True):
            p1, phase2 = build_gen(work, cap, batched)
            jobs += p1
            later.append(phase2)
    if "ffn16" in kinds:
        jobs += build_ffn16(work, bits)
    if "qmm" in kinds:
        jobs += build_qmm(work, bits)
    if "qmm_swiglu" in kinds:
        for M in QMM_SWIGLU_M:
            jobs += build_qmm_swiglu(work, bits, M)
    return jobs, later


def deliver(job, out, root):
    """Copy a verified bundle to its content-addressed home and return its index entry."""
    sha = hashlib.sha256(job["prog"].code).hexdigest()
    name = job.get("deliver_as", job["name"])
    dest = root / "bundles" / ("%s-%s" % (name, sha[:16]))
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(job["dir"], dest)
    e = dict(job["entry"], bundle=str(dest.relative_to(out)), name=name, program_sha256=sha, sha256=sha,
             code_bytes=len(job["prog"].code))
    e.setdefault("bits", None)
    e.setdefault("cap", None)
    e["verified"] = "hardware, bit-exact over a 0x7f sentinel (tools/g17deliver.py)"
    return e


def build(args):
    set_arch(getattr(args, "arch", "internlm2"))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    kinds = set(args.kinds.split(","))
    work = Path(tempfile.mkdtemp(prefix="g17deliver-", dir=str(out)))
    try:
        jobs, later = plan(args.bits, args.cap, kinds, work, attn_variants=tuple(args.attn_variants.split(",")))
        results = []
        run_jobs = [j for j in jobs if not j.get("prechecked")]
        outs = dispatch([dict(tag=j["name"], dir=j["dir"], threads=j["threads"], group=j["group"], base=j["base"],
                              rounds=j.get("rounds", 1)) for j in run_jobs], work) if run_jobs else {}
        for j in jobs:
            if j.get("prechecked"):
                results.append((j, j["mismatch"]))
            elif outs[j["name"]] is None:
                results.append((j, "refused"))
            else:
                results.append((j, j["check"](outs[j["name"]])))
        for phase2 in later:
            try:
                jobs2 = phase2(_Phase1(outs))
            except _Phase1Refused as e:           # emulate mode: the phase-1 program it compiles from was refused
                EMULATE["refused"]["%s (phase 2)" % e.args[0]] = "its phase-1 program was refused"
                continue
            outs2 = dispatch([dict(tag=j["name"], dir=j["dir"], threads=j["threads"], group=j["group"], base=j["base"])
                              for j in jobs2], work)
            for j in jobs2:
                results.append((j, "refused" if outs2[j["name"]] is None else j["check"](outs2[j["name"]])))
        if EMULATE is not None:
            return _emulate_report(results, jobs, later, getattr(args, "against", None),
                                   getattr(args, "emulate_json", None))
        bad = [(j["name"], n) for j, n in results if n]
        for j, n in results:
            print("verify %-48s %s" % (j["name"], "bit-exact" if not n else "DIFFERS (%d)" % n), flush=True)
        if bad:
            raise SystemExit("NOT bit-exact, nothing delivered: %s" % bad)
        index_path = out / "index.json"
        index = json.load(open(index_path)) if index_path.exists() else []
        new = [deliver(j, out, out) for j, _ in results if j.get("entry")]
        missing = {e["name"]: validate(e) for e in new if validate(e)}
        if missing:
            raise SystemExit("index entries miss contract fields: %s" % missing)
        sel = [(e["kind"], e["bits"], e["role"], json.dumps(e["variant"], sort_keys=True), e["cap"]) for e in new]
        if len(set(sel)) != len(sel):
            raise SystemExit("two entries share a selection key (kind, bits, role, variant, cap)")
        keys = {(e["kind"], e["bits"], e["role"], json.dumps(e["variant"], sort_keys=True), e["cap"]) for e in new}
        index = [e for e in index if (e["kind"], e["bits"], e["role"], json.dumps(e["variant"], sort_keys=True), e["cap"])
                 not in keys] + new
        json.dump(index, open(index_path, "w"), indent=1)
        print("delivered %d bundles; index %s" % (len(new), index_path))
    finally:
        shutil.rmtree(work, ignore_errors=True)


class _Phase1Refused(Exception):
    pass


class _Phase1(dict):
    """Phase-1 outputs as a phase-2 builder reads them; in emulate mode a refused program's output is None."""
    def __getitem__(self, tag):
        out = dict.__getitem__(self, tag)
        if out is None:
            raise _Phase1Refused(tag)
        return out


def _emulate_report(results, jobs, later, against=None, json_out=None):
    """--emulate: per job bit-exact / DIFFERS / refused, then per kind, then the refusals by cause. Delivers nothing."""
    import collections
    by = collections.defaultdict(collections.Counter)
    for j, n in results:
        kind = (j.get("entry") or {}).get("kind") or j["name"].split("_")[0]
        adm = EMULATE["admitted"].get(j["name"])
        v = "refused" if n == "refused" else "DIFFERS" if n else "bit-exact (wp)" if adm else "bit-exact"
        by[kind][v] += 1
        print("emulate %-48s %s%s" % (j["name"], v if v != "DIFFERS" else "DIFFERS (%d)" % n,
                                     "  admitted %s" % adm if adm else ""), flush=True)
    for name, why in sorted(EMULATE["refused"].items()):
        if name not in {j["name"] for j, _ in results}:
            print("emulate %-48s refused" % name)
            by[name.split("_")[0]]["refused"] += 1
    print("\nby kind (bit-exact strict / + whole-program tier / differs / refused):")
    for kind, c in sorted(by.items()):
        print("  %-16s %4d / %4d / %d / %d" % (kind, c["bit-exact"], c["bit-exact"] + c["bit-exact (wp)"],
                                            c["DIFFERS"], c["refused"]))
    adm = collections.Counter(k for d in EMULATE["admitted"].values() for k in d)
    print("whole-program semantics admitted (jobs using each):")
    for what, n in adm.most_common():
        print("  %4d  %s" % (n, what))
    causes = collections.Counter(why.split(":")[0] if "op" not in why else why.split(" at ")[0]
                                 for why in EMULATE["refused"].values())
    print("refusals by cause:")
    for why, n in causes.most_common():
        print("  %4d  %s" % (n, why))
    if against:
        _emulate_against(results, against, json_out)
    return 0 if not any(c["DIFFERS"] for c in by.values()) else 1


def _emulate_against(results, index_path, json_out=None):
    """--against INDEX: judge a DELIVERED root. Each index entry maps to the emulated jobs whose program.bin has the
    same sha256 (a delivered program this build no longer produces is "not rebuilt", never a pass); its verdict is
    theirs. Prints per delivered kind and the refusals by opcode; --emulate-json writes every row."""
    import collections, re
    verdict = {}
    for j, n in results:
        v = "refused" if n == "refused" else "DIFFERS" if n else \
            "bit-exact (wp)" if EMULATE["admitted"].get(j["name"]) else "bit-exact"
        h = hashlib.sha256((Path(j["dir"]) / "program.bin").read_bytes()).hexdigest()
        verdict.setdefault(h, []).append((j["name"], v))
    rank = ("DIFFERS", "refused", "bit-exact (wp)", "bit-exact")
    table, refusals, rows = collections.defaultdict(collections.Counter), collections.Counter(), []
    for e in json.loads(Path(index_path).read_text()):
        got = verdict.get(e["program_sha256"], [])
        v = min((x for _n, x in got), key=rank.index) if got else "not rebuilt"
        why = sorted({EMULATE["refused"].get(n, "") for n, x in got if x == "refused"} - {""})
        for w in why:
            m = re.match(r"(op\d+)", w)
            refusals[(e["kind"], m.group(1) if m else w.split(":")[0][:60])] += 1
        table[e["kind"]][v] += 1
        rows.append(dict(kind=e["kind"], name=e.get("name"), program_sha256=e["program_sha256"], verdict=v,
                         jobs=[n for n, _x in got], refused=why,
                         admitted=sorted({k for n, _x in got for k in EMULATE["admitted"].get(n, {})})))
    print("\nagainst %s: %d delivered bundles (strict / + whole-program tier / differs / refused / not rebuilt):"
          % (index_path, len(rows)))
    tot = collections.Counter()
    for kind, c in sorted(table.items()):
        tot.update(c)
        print("  %-16s %4d / %4d / %d / %d / %d  of %d" % (kind, c["bit-exact"], c["bit-exact"] + c["bit-exact (wp)"],
                                                         c["DIFFERS"], c["refused"], c["not rebuilt"], sum(c.values())))
    print("  %-16s %4d / %4d / %d / %d / %d  of %d" % ("TOTAL", tot["bit-exact"], tot["bit-exact"] + tot["bit-exact (wp)"],
                                                     tot["DIFFERS"], tot["refused"], tot["not rebuilt"], len(rows)))
    print("refused delivered bundles by (kind, opcode):")
    for (kind, op), n in sorted(refusals.items()):
        print("  %-16s %-10s %d" % (kind, op, n))
    if json_out:
        Path(json_out).write_text(json.dumps(dict(index=str(index_path), table={k: dict(c) for k, c in table.items()},
                                                  refusals=[[k, op, n] for (k, op), n in sorted(refusals.items())],
                                                  rows=rows), indent=1) + "\n")


def rebuild(entry):
    """The entry's program, rebuilt compile-only from its recipe."""
    return BUILDERS[entry["recipe"]["builder"]](entry["recipe"]["layout"])


def check(args):
    index = json.load(open(args.index))
    bad = 0
    for e in index:
        sha = hashlib.sha256(rebuild(e).code).hexdigest()
        ok = sha == e["program_sha256"]
        bad += not ok
        print("%-9s %-48s %s" % ("ok" if ok else "DIFFERS", e.get("name", e["bundle"]), sha[:16]))
    if bad:
        raise SystemExit("%d of %d recorded programs no longer rebuild to their sha" % (bad, len(index)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--bits", type=int, choices=(4, 8), required=True)
    b.add_argument("--arch", choices=sorted(ARCHS), default="internlm2",
                   help="the model's shapes (MM 25.182); internlm2 is every bundle as before")
    b.add_argument("--cap", type=int, default=272)
    b.add_argument("--out", required=True)
    b.add_argument("--kinds", default="qmv,norm,head,attn,gen")
    b.add_argument("--attn-variants", default="",
                   help='comma list of attention flag strings, e.g. ",keyblock=2,tgsplit=4+keyblock=2" ("" = the base form)')
    b.add_argument("--emulate", action="store_true",
                   help="run every check on the CPU with tools/g17emu.py (no GPU dispatch); report, deliver nothing")
    b.add_argument("--emulate-workers", type=int, default=1,
                   help="with --emulate: emulate this many bundles at once, one process each")
    b.add_argument("--against", metavar="INDEX",
                   help="with --emulate: judge a delivered root's index.json, matching bundles by program sha256")
    b.add_argument("--emulate-json", metavar="PATH", help="with --against: write every delivered bundle's verdict")
    c = sub.add_parser("check")
    c.add_argument("index")
    args = ap.parse_args(argv)
    if getattr(args, "emulate", False):
        global EMULATE
        EMULATE = dict(refused={}, admitted={}, workers=args.emulate_workers)
    (build if args.cmd == "build" else check)(args)


if __name__ == "__main__":
    main()
