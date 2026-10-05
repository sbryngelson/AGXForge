#!/usr/bin/env python3
"""THE QUANTIZED CHAINED TOKEN (docs/g17-tensorops-machine-model.md 25.138.5): InternLM2.5-1.8B at 4 (or 8) bits,
mlx-lm's own affine group-64 weights, as one ordered dispatch list for Set C's executor (tools/g17decodegen.m).

    python3 tools/g17q4graph.py prepare --bits 4    (system python3 with MLX, its own process) the mlx checkpoint's
                                                    tensors as npy in our row order, and the dequantized embedding
    python3 tools/g17q4graph.py build --bits 4 --deliver DIR --config tools/models/internlm2_q4_best.json
                                                    graph.json + weight arenas
    python3 tools/g17q4graph.py simulate --bits 4 --deliver DIR --config FILE
                                                    the graph on the CPU, every dispatch by its exact-order reference
    (tools/g17modelbuild.py runs build, simulate and the GPU check in one command.)

A LAYER is 7 dispatches (the fused, device-resident graph; the 9-dispatch form below was the first, 25.138.5):
  attn_norm -> qkv qmv -> attention (RoPE, append, split and merge in one dispatch, 25.140.5) ->
  wo qmv + residual1 -> ffn_norm -> w1+w3+SwiGLU qmv -> w2 qmv + residual2
[Corrected 2026-09-26: this said 9 dispatches: attn_norm -> qkv -> RoPE append -> split -> merge -> wo -> ffn_norm ->
ffn -> w2. That route and its fixed bundle directories are gone; see the note above DELIVER_ROOT.]
ONE region R of 12,288 bytes carries the residual stream through every layer: h fp32 [2048] at R + 0 (written by
wo + residual1, read by ffn_norm and w2 + residual2), x fp16 [2048] at R + 8,192 (written by w2 + residual2 as the
next layer's input, read by attn_norm and wo + residual1; the embedding row is written there for layer 0). The head
is the final norm and ONE lm_head qmv (92,544 rows).

WEIGHTS are mlx_lm.convert's -q output as it is (W uint32 low field first, scales and biases bf16, group 64). Only
rows move: wqkv from the checkpoint's per-KV-group packing [q 2g, q 2g+1, k g, v g] to q heads 0..15 | k 0..7 |
v 0..7, and w1 and w3 concatenated for the fused SwiGLU. A row permutation moves each row's scales and biases
with it, so no value changes. The embedding is mx.dequantize of the quantized table (MLX's own embedding values),
stored fp16.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import g17decodeops as O  # noqa: E402
import g17decodestep as D  # noqa: E402
import g17modelgraph as MG  # noqa: E402
import g17realmodel as M  # noqa: E402

F32 = np.float32
# THE GRAPH HAS ONE ROUTE: a deliver root and a model config (the source-to-tokens build, tools/g17modelbuild.py). Every
# bundle comes from the deliver root's index.json (tools/g17deliver.py: flat entries keyed by kind / bits / role /
# variant / cap, paths relative to the index), and the config (a JSON file) names the variant per role: {"name", "bits",
# "cap", "prompt_len", "qmv": {role: variant}, "norm": variant, "final_norm": variant, "attn": variant, "gen": variant}.
# It is the fused, device-resident, wide-norm, v2 graph: 7 dispatches a layer (the RoPE-fused split with its merge in the
# last threadgroup behind the device fence (op14156), 25.140.5), device-resident generation (25.138.6, 25.140.2), the
# wide RMSNorm in tree order (25.141.4), v2 fp32 activations (25.141.9), the butterfly-merge wide attention (25.141.14).
# [2026-09-26: the older per-feature switches (G17_Q4_UNFUSED / FUSED / GEN / CAP / CAP_DIR / PROMPT_LEN / WIDENORM /
# WIDENORM_DIR / V2 / WIDEBF / KSPLIT / LEAN / SEEDNORM / V2_ROOT) and their fixed bundle directories under /private/tmp
# are gone: split-K, lean and seed norms are config variants, and the cap and prompt length are config fields. Those
# directories were other worktrees' results/, and even this route read two attention index files from them.]
ROPE_FUSED = FUSED = GEN = WIDENORM = Q4V2 = WIDEBF = True     # what the one route is; kept as names for readers
DELIVER_ROOT = None
CONFIG = {}
CAP = 272
PROMPT_LEN = 0
# BATCHED DECODE (MM 25.144.3): the config's "batch" B runs B independent sequences through one graph, every dispatch
# covering all B (the batched qmv / norm / attention / generation bundles), each sequence with its own KV cache, q0 and
# token log. Vector-major throughout: sequence b sees the single-sequence layout shifted by a fixed stride. B = 1 is the
# single-sequence graph, byte for byte.
BATCH = 1


def configure(deliver, config):
    """Point the graph at a deliver root and a model config (paths). Sets the cap and prompt length from the config."""
    global DELIVER_ROOT, CAP, PROMPT_LEN, BATCH, M
    DELIVER_ROOT = Path(deliver)
    CONFIG.clear()
    CONFIG.update(json.loads(Path(config).read_text()))
    # A SECOND ARCHITECTURE (MM 25.182): "arch": "qwen3" takes the model's constants, weights and base record from
    # tools/g17qwen3.py (d_model 1,024, 28 layers, QK-norm, a tied 151,936-token head); absent, it is InternLM2
    if CONFIG.get("arch", "internlm2") == "qwen3":
        import g17qwen3
        g17qwen3.configure(CONFIG.get("model", "qwen3-0.6b"))      # the Qwen3 model's shapes (g17qwen3.MODELS)
        M = g17qwen3
    elif CONFIG.get("arch", "internlm2") != "internlm2":
        raise SystemExit("g17q4graph: unknown arch %r" % CONFIG["arch"])
    global R_GEN
    # the generation state follows h fp32 [d] and x fp16 [d] in R: 12,288 for d <= 2,048 (every graph before
    # Qwen3-8B, byte-unchanged), 6 d past it
    R_GEN = max(12288, 6 * M.D_MODEL)
    CAP = int(CONFIG.get("cap", 272))
    PROMPT_LEN = int(CONFIG.get("prompt_len", 0))
    BATCH = int(CONFIG.get("batch", 1))
    if BATCH > 1 and CONFIG.get("prefill"):
        raise SystemExit("g17q4graph: a batched decode config has no prefill section (the prefill arena is one sequence's)")
    _INDEX.clear()


def _r_res():
    """byte offset of the fp16 x rows in the region R: after the B fp32 h rows (the batched w2 residual's RES)"""
    # single-sequence: after h fp32 [d], where g17qmv.with_residual's RES puts x (8,192 for InternLM2, 4,096 for Qwen3)
    return MG._al(4 * M.D_MODEL * BATCH) if BATCH > 1 else 4 * M.D_MODEL


def _r_gen():
    """byte offset of the generation state in R: after the B fp16 x rows (g17gen.gen_batch_layout's GEN)"""
    return MG._al(_r_res() + 2 * M.D_MODEL * BATCH) if BATCH > 1 else R_GEN


def _require():
    if DELIVER_ROOT is None:
        raise SystemExit("g17q4graph: no deliver root; pass --deliver DIR --config FILE (tools/g17modelbuild.py does)")


def _prompt(g0, seq=0):
    """the prompt of sequence `seq`: the base prompt rotated by the config's prompt_rotate + seq (0 is the base prompt, so
    a single-sequence graph is unchanged); every sequence has the same length, as the batched graph steps them together"""
    p = list(CONFIG.get("prompt_ids") or g0["prompt_ids"])       # an explicit prompt (MM 25.203's workloads) overrides
    p = p if not PROMPT_LEN else (p * (PROMPT_LEN // len(p) + 1))[:PROMPT_LEN]
    r = (int(CONFIG.get("prompt_rotate", 0)) + seq) % len(p)
    return p[r:] + p[:r]


_INDEX = []
if os.environ.get("G17_Q4_DELIVER"):
    # DEPRECATED (2026-09-26): the environment interface, still honoured for callers that import this module after
    # setting it (tools/g17prefillhead.py). Pass --deliver / --config instead.
    print("g17q4graph: G17_Q4_DELIVER / G17_Q4_CONFIG are deprecated; pass --deliver DIR --config FILE",
          file=sys.stderr)
    if not os.environ.get("G17_Q4_CONFIG"):
        raise SystemExit("g17q4graph: G17_Q4_DELIVER is set without G17_Q4_CONFIG")
    configure(os.environ["G17_Q4_DELIVER"], os.environ["G17_Q4_CONFIG"])


def _pick(kind, **match):
    """The one index entry of this kind whose fields equal `match` (variant compared as a whole dict)."""
    if not _INDEX:
        _INDEX.extend(json.loads((DELIVER_ROOT / "index.json").read_text()))
    hits = [e for e in _INDEX if e["kind"] == kind and all(e.get(k) == v for k, v in match.items())]
    if len(hits) != 1:
        raise KeyError("deliver index %s: %d entries for %s %s" % (DELIVER_ROOT, len(hits), kind, match))
    return hits[0]


def prefill_cfg():
    """The model config's prefill section resolved for this prompt: {} when there is none or the prompt is shorter than
    its min_len (then the prompt is fed token by token, MM 25.142.10); "M": "auto" becomes the smallest chunk bucket
    every prefill kind shares (128 or 512) that covers the prompt."""
    pcfg = dict(CONFIG.get("prefill") or {})
    if not pcfg:
        return {}
    L = len(_prompt(json.loads((M.OUT / "graph" / "graph.json").read_text())))
    if L < int(pcfg.get("min_len", 0)):
        return {}
    if pcfg.get("M") == "auto":
        pcfg["M"] = 128 if L <= 128 else 512
    return pcfg


def _pick_rooted(kind, **match):
    return dict(_pick(kind, **match), _root=str(DELIVER_ROOT))


def _qmv_op_delivered(name, key, kind, wname):
    import re
    bits = int(re.search(r"_q(\d)_", key).group(1))
    role = "head" if name.startswith("head.") else name.split(".", 1)[1]
    e = _pick("lm_head" if role == "head" else "qmv", bits=bits, role=role, variant=CONFIG["qmv"][role])
    d = DELIVER_ROOT / e["bundle"]
    lay = json.loads((d / "decodeop.json").read_text())["layout"]
    N, K = lay["Nout"], lay["Kq"]
    x32 = not e["variant"].get("x16")
    nb = e["variant"].get("batch", 1)                  # the batched qmv covers nb vectors (vector-major)
    b3 = {"qmv": N * 4 * nb, "res1": R_GEN, "res2": R_GEN, "swiglu": lay.get("ffn", 0) * (4 if x32 else 2) * nb}[kind]
    if nb > 1 and kind in ("res1", "res2"):
        b3 = _r_gen()                                  # the batched residual reads and writes all of R's h and x rows
    op = MG.Op(name, e["bundle"], kind, dict(lay, S_off=e["S"], B_off=e["B"]),
               {1: K * (4 if x32 else 2) * nb, 2: e["B"] + N * (K // 64) * 2, 3: b3}, weights=wname)
    tpg = e.get("threads_per_group", 32)
    op.bundle, op.key, op.threads, op.group = d, e["bundle"], e["threadgroups"] * tpg, tpg
    if e.get("base", 1) == 0:
        op.meta["phys"] = {"0": 3, "1": 1, "2": 2}
    if x32:
        op.meta["x32"] = True
        if kind == "swiglu":
            op.meta["act32"] = True
    return op


R_GEN = 12288
ACT = MG.ACT


def out_dir(bits):
    _require()
    return M.OUT / ("graph_q%d_%s" % (bits, CONFIG.get("name", "model")))


def weights_dir(bits):
    """where `prepare` writes the checkpoint's weights; g17modelbuild links each config's graph directory to it"""
    return M.OUT / ("graph_q%d" % bits) / "weights"


def mlx_dir(bits):
    return Path.home() / "models" / ("internlm2_5-1_8b-chat-mlx-q%d" % bits)


# ---------------------------------------------------------------------------------------------------
# weights

def _qkv_rows():
    """new row -> checkpoint row, for the packed wqkv (4,096 rows)."""
    groups = M.HEADS // M.KV_HEADS
    old = np.arange(4096).reshape(M.KV_HEADS, groups + 2, M.HEAD_DIM)
    q = old[:, :groups].reshape(-1)
    return np.concatenate([q, old[:, groups].reshape(-1), old[:, groups + 1].reshape(-1)])


def prepare(bits):
    import mlx.core as mx
    src = mlx_dir(bits)
    t = {}
    for f in sorted(src.glob("*.safetensors")):
        t.update(mx.load(str(f)))
    W = weights_dir(bits)
    W.mkdir(parents=True, exist_ok=True)

    def u16(a):
        return np.array(a.view(mx.uint16))

    def trio(prefix):
        return np.array(t[prefix + ".weight"]), u16(t[prefix + ".scales"]), u16(t[prefix + ".biases"])

    perm = _qkv_rows()
    for l in range(M.LAYERS):
        p = "model.layers.%d." % l
        w, s, b = trio(p + "attention.wqkv")
        np.savez(W / ("L%d_qkv.npz" % l), W=w[perm], S=s[perm], B=b[perm])
        np.savez(W / ("L%d_wo.npz" % l), **dict(zip("WSB", trio(p + "attention.wo"))))
        w1, s1, b1 = trio(p + "feed_forward.w1")
        w3, s3, b3 = trio(p + "feed_forward.w3")
        np.savez(W / ("L%d_ffn.npz" % l), W=np.concatenate([w1, w3]), S=np.concatenate([s1, s3]),
                 B=np.concatenate([b1, b3]))
        np.savez(W / ("L%d_w2.npz" % l), **dict(zip("WSB", trio(p + "feed_forward.w2"))))
        for name, key in (("g1", "attention_norm"), ("g2", "ffn_norm")):
            np.save(W / ("L%d_%s.npy" % (l, name)), np.array(t[p + key + ".weight"].astype(mx.float16)))
    np.save(W / "norm.npy", np.array(t["model.norm.weight"].astype(mx.float16)))
    np.savez(W / "lm.npz", **dict(zip("WSB", trio("output"))))
    e = t["model.tok_embeddings.weight"]
    emb = mx.dequantize(e, t["model.tok_embeddings.scales"], t["model.tok_embeddings.biases"], group_size=64,
                        bits=bits)
    np.save(W / "embed.npy", np.array(emb.astype(mx.float16)))
    print("prepared", W)


# ---------------------------------------------------------------------------------------------------
# the ops

def _qmv_op(name, bundle, kind, wname):
    return _qmv_op_delivered(name, bundle, kind, wname)


def _deco(name, key, kind, lay, **meta):
    op = MG._deco_op(name, key, kind, lay, **meta)
    op.bundle, op.threads, op.group = None, None, None
    if kind == "norm" and DELIVER_ROOT:
        role = "final_norm" if name.startswith("head.") else name.split(".", 1)[1]
        var = CONFIG["final_norm" if role == "final_norm" else "norm"]
        e = _pick("norm", role=role, variant=var)
        assert (e["X"], e["G"], e["OUT"]) == (lay["X"], lay["G"], lay["OUT"]), "delivered norm offsets differ"
        nb = var.get("batch", 1)
        op.kind, op.bundle, op.threads, op.group = "wnorm", DELIVER_ROOT / e["bundle"], 1024 * nb, 1024
        if nb > 1:
            # the batched norm: nb threadgroups, row b's x and out offset b d elements
            xs, os_ = (2 if lay["in_dtype"] == "half" else 4), (4 if var.get("out32") else 2)
            op.ext[1] = lay["X"] + xs * lay["d"] * nb
            op.ext[3] = lay["OUT"] + os_ * lay["d"] * nb
            op.meta["batch"] = nb
        op.meta["phys"] = {"0": 3, "1": 1, "2": 2}
        op.meta.update({k: True for k in ("out32", "seed") if var.get(k)})
        return op
    return op


def _attn_rope(A):
    """the attention's region layout at the model's head counts (16 / 8 unless the model says otherwise: Qwen3-8B 32 / 8)"""
    return A.attn_rope_layout(cap=CAP, heads=getattr(M, "HEADS", 16), kv_heads=getattr(M, "KV_HEADS", 8))


def _attn_batch_lay():
    """the batched wide butterfly attention layout (g17attn.with_batch): per-sequence caches and attn rows, the q0 words in
    the generation state blocks (SS apart), the rope tables shared after them"""
    import g17attn as A
    import g17gen as G
    base = A.with_attn32(A.with_bfly_merge(A.with_wide(A.with_fused_merge(A.with_rope_tables(_attn_rope(A))))))
    return A.with_batch(base, BATCH, G.gen_batch_layout(BATCH, cap=CAP)["SS"])


def _gen_lay():
    import g17attn as A
    if BATCH > 1:
        return _attn_batch_lay()                       # its COST / SINT / rope_bytes / SS place the batched tables
    return A.with_rope_tables(_attn_rope(A))


def _spec_cfg():
    """The verify step's section config (MM 25.203): M 16 rows, the scalar attention (runtime p0), the qsm projections;
    or (check "qmvw", MM 25.207) M 4 rows through the qmvw projections (fp32 rows)."""
    opts = {"attn_opts": list(CONFIG["spec"]["attn_opts"])} if CONFIG["spec"].get("attn_opts") else {}
    if CONFIG["spec"].get("check") == "qmvw":
        return dict(M=4, attn="scalar", gemm_M=4, qmvw=4, **opts)
    return dict(M=16, attn="scalar", gemm_M=16, qsm=dict(CONFIG["spec"].get("qsm", dict(qkv=4, wo=8, w1=2, w2=8))),
                **({"attn_opts": list(CONFIG["spec"]["attn_opts"])} if CONFIG["spec"].get("attn_opts") else {}))


def _fused_attn_op(l):
    import g17attn as A
    lay = A.with_fused_merge(A.with_rope_tables(_attn_rope(A)))
    if DELIVER_ROOT:
        e = _pick("attn", cap=CAP, variant=CONFIG["attn"])
        # kvvec (one 8-byte K/V load per row per lane) and hwexp2 (op1272's softmax 2^x, enclosure-checked, MM 25.144.5)
        # change the CODE only: the same buffers, offsets and threadgroups as the butterfly form they build on
        code_only = {"kvvec", "hwexp2", "qknorm"}
        # the long-context forms (MM 25.144.5: keyblock, tgsplit, gqapair, nsum; wired MM 25.199) change the code and,
        # for tgsplit / gqapair, the grid and the partial scratch: they run on their DELIVERED layout, held to the base
        # form's cache, rope and output offsets
        m5 = {"keyblock", "tgsplit", "gqapair", "nsum"}
        single = {k: v for k, v in e["variant"].items() if k != "batch" and k not in code_only and k not in m5}
        if single not in ({"widebf": True, "attn32": True}, {"wide": True, "bfly": True, "attn32": True}):
            raise ValueError("only the butterfly-merge wide attention is wired: %s" % e["variant"])
        if e["variant"].get("batch", 1) != BATCH:
            raise ValueError("the attention variant's batch %s is not the config's %d" % (e["variant"].get("batch"), BATCH))
        if code_only & set(e["variant"]) and BATCH > 1:
            raise ValueError("kvvec / hwexp2 are single-sequence forms: %s" % e["variant"])
        lay = _attn_batch_lay() if BATCH > 1 else A.with_attn32(A.with_bfly_merge(A.with_wide(lay)))
        if e["variant"].get("qknorm"):
            # MM 25.188: QK-norm in the attention; its gains sit in binding 1 at QG (a per-layer persistent init)
            lay = A.with_qknorm(lay, e["recipe"]["layout"]["qknorm_eps"])
        assert m5 & set(e["variant"]) or (lay["ATTN"], lay["region3_bytes"]) == (e["ATTN"], e["region3_bytes"]), \
            "delivered attention layout differs"
        if m5 & set(e["variant"]):
            if BATCH > 1:
                raise ValueError("the long-context attention forms are single-sequence: %s" % e["variant"])
            got = dict(e["recipe"]["layout"])
            for k in ("KOFF", "VOFF", "ATTN", "COST", "SINT", "LEN", "cap", "heads", "kv_heads", "head_dim"):
                if got.get(k) != lay.get(k):
                    raise ValueError("the %s attention's %s differs from the base form's" % (e["variant"], k))
            lay = got
        elif code_only & set(e["variant"]):
            flags = {"kvvec", "hw_exp2", "qknorm", "qknorm_eps", "QG", "q_bytes"}
            got = {k: v for k, v in e["recipe"]["layout"].items() if k not in flags}
            if got != {k: v for k, v in lay.items() if k not in flags}:
                raise ValueError("the %s attention's layout differs beyond its code flags" % e["variant"])
        r3 = lay["region3_bytes"]
        if prefill_cfg():                                # the prefill kernels write q16 / attention past region3
            import g17prefillgraph as PG
            r3 = max(r3, PG.region3_bytes(_pick_rooted, CAP, prefill_cfg()))
        if CONFIG.get("spec"):                           # ... and so does the speculative verify step (MM 25.203)
            import g17prefillgraph as PG
            r3 = max(r3, PG.region3_bytes(_pick_rooted, CAP, _spec_cfg()))
        op = MG.Op("L%d.attn_fused" % l, e["bundle"], "ffused", lay,
                   {0: r3, 1: (lay["q_bytes"] if lay.get("qknorm") else 16384 * BATCH), 2: lay["rope_bytes"]})
        if lay.get("qknorm"):
            op.meta["qk_gain"] = "L%d_qk" % l
        tpg = e.get("threads_per_group", 32)
        op.bundle, op.threads, op.group = DELIVER_ROOT / e["bundle"], e["threadgroups"] * tpg, tpg
        op.meta["slots"] = (0, 1, 2)
        op.meta["attn32"] = True
        return op
    raise SystemExit("g17q4graph: no deliver root")


def _qsm_op(name, role, wname):
    """a g17qsm projection (MM 25.172): physical binding 1 the weight block (logical slot 2, the weights arena), 2 the x16
    rows (logical 1, an activation), 3 y or the sk partials"""
    if BATCH == 32:                                     # one n-tile a threadgroup, two batch-half bodies (MM 25.178)
        var = {"sk": dict({"qkv": 1, "wo": 4, "w1": 1, "w3": 1, "w2": 4, "head": 1},
                          **({"qkv": 2} if CONFIG.get("qsm_occ") else {}))[role], "xrows": True, "mb": 32}
    else:
        var = {"sk": _QSM_SK[role], "xrows": True}
        if CONFIG.get("qsm_h16"):                       # MM 25.196: the dequant on the fp16 pipe
            var["h16"] = True
        if CONFIG.get("qsm_pad"):                       # MM 25.196: the padded build (every qsm of it carries the key)
            var["pad"] = CONFIG["qsm_pad"]
    e = _pick("qsm", bits=4, role=role, variant=var)
    lay = e["recipe"]["layout"]
    NB = 2 * lay["N"] if role in ("w1", "w3") else lay["N"]           # the FFN block holds w1 then w3
    wb = MG._al(NB * (lay["K"] // 2) + 2 * NB * (lay["K"] // 32))
    op = MG.Op(name, e["bundle"], "qsm", dict(lay, block_rows=NB), {1: lay.get("mb", 16) * lay["K"] * 2, 2: wb, 3: lay["c_bytes"]},
               weights=wname)
    op.bundle, op.threads, op.group = DELIVER_ROOT / e["bundle"], e["threadgroups"] * 32, 32
    op.meta["phys"] = {"1": 2, "2": 1, "3": 3}
    return op


_QSM_SK_BASE = {"qkv": 2, "wo": 4, "w1": 1, "w3": 1, "w2": 4, "head": 1}
# MM 25.185, config "qsm_occ": the occupancy split-K (g17deliver.QSM_SK_OCC), w1 / w3 summed by a psum pass
_QSM_SK_OCC = {"qkv": 4, "wo": 8, "w1": 2, "w3": 2, "w2": 8, "head": 1}


class _QskView(dict):
    def __getitem__(self, role):
        if CONFIG.get("qsm_sk"):                        # MM 25.196: an explicit split per role (the padded build's)
            return CONFIG["qsm_sk"][role]
        return (_QSM_SK_OCC if CONFIG.get("qsm_occ") else _QSM_SK_BASE)[role]


_QSM_SK = _QskView()


def _psum_op(name, role, mode, sk):
    e = _pick("psum", role=role, variant=dict({"mode": mode, "rows": BATCH, "sk": sk},
                                              **({"pad": CONFIG["qsm_pad"]} if CONFIG.get("qsm_pad") else {})))
    lay = e["recipe"]["layout"]
    op = MG.Op(name, e["bundle"], "psum", lay, {1: lay["a_bytes"], 2: lay["b_bytes"], 3: lay["c_bytes"]})
    op.bundle, op.threads, op.group = DELIVER_ROOT / e["bundle"], e["threadgroups"] * 32, 32
    return op


def _swiglu_rows_op(name):
    e = _pick("swiglu_rows", variant={"rows": BATCH})
    lay = e["recipe"]["layout"]
    op = MG.Op(name, e["bundle"], "swiglu_rows", lay, {1: lay["a_bytes"], 2: lay["b_bytes"], 3: lay["c_bytes"]})
    op.bundle, op.threads, op.group = DELIVER_ROOT / e["bundle"], e["threadgroups"] * 32, 32
    return op


def layer_ops_qsm(l, bits, spec):
    """THE BATCHED DECODE ON THE TENSOR UNITS (MM 25.172, config "qsm"): every projection is a g17qsm (the weights
    dequantized once, op5106 against the batch's x16 rows), its split-K partials summed by g17psum with the residual
    folded in, the FFN's SwiGLU a rows kernel. The norms write fp16 rows (config norm out32 false); the attention is
    the batched butterfly form (fp32 out, narrowed to fp16 rows by a psum "half" pass)."""
    d = spec.d_model
    n1 = _deco("L%d.attn_norm" % l, "rmsnorm_half_d%d_g32_u16_h" % d, "norm",
               O.rmsnorm_loop_layout(d, "half", groups=32, unroll=16, hoist=True), g="L%d_g1" % l)
    qkv = _qsm_op("L%d.qsm_qkv" % l, "qkv", "L%d_qkv" % l)
    psq = _psum_op("L%d.psum_qkv" % l, "qkv", "sum", (2 if CONFIG.get("qsm_occ") else 1) if BATCH == 32 else _QSM_SK["qkv"])
    fs = _fused_attn_op(l)
    cvt = _psum_op("L%d.attn_half" % l, "attn", "half", 1)
    wo = _qsm_op("L%d.qsm_wo" % l, "wo", "L%d_wo" % l)
    pso = _psum_op("L%d.psum_wo" % l, "wo", "add16", _QSM_SK["wo"] if BATCH != 32 else 4)
    n2 = _deco("L%d.ffn_norm" % l, "rmsnorm_float_d%d_g32_u8_h" % d, "norm",
               O.rmsnorm_loop_layout(d, "float", groups=32, unroll=8, hoist=True), g="L%d_g2" % l)
    w1 = _qsm_op("L%d.qsm_w1" % l, "w1", "L%d_ffn" % l)
    w3 = _qsm_op("L%d.qsm_w3" % l, "w3", "L%d_ffn" % l)
    sw = _swiglu_rows_op("L%d.swiglu" % l)
    w2 = _qsm_op("L%d.qsm_w2" % l, "w2", "L%d_w2" % l)
    psw = _psum_op("L%d.psum_w2" % l, "w2", "fold16", _QSM_SK["w2"] if BATCH != 32 else 4)
    ops = [n1, qkv, psq, fs, cvt, wo, pso, n2, w1, w3, sw, w2, psw]
    qin = psq
    if getattr(M, "QK_NORM", False):
        # Qwen3 batched (MM 25.189): the B-row QK-norm on the summed qkv rows, ahead of the batched attention
        e = _pick("headnorm", role="qknorm", variant={"rows": BATCH})
        hn = MG.Op("L%d.qk_norm" % l, e["bundle"], "headnorm", dict(e["recipe"]["layout"]),
                   {1: 4 * M.QKV * BATCH, 2: 4 * M.HEAD_DIM, 3: 4 * M.QKV * BATCH}, weights="L%d_qk" % l)
        hn.bundle, hn.threads, hn.group = DELIVER_ROOT / e["bundle"], 1024 * BATCH, 1024
        hn.meta["phys"] = {"0": 3, "1": 1, "2": 2}
        ops.insert(ops.index(fs), hn)
        qin = hn
    edges = [((n1, 3, n1.lay["OUT"]), (qkv, 1, 0)), ((qkv, 3, 0), (psq, 1, 0)), ((qin, 3, 0), (fs, 1, 0)),
             ((fs, 0, fs.lay["ATTN"]), (cvt, 1, 0)), ((cvt, 3, 0), (wo, 1, 0)), ((wo, 3, 0), (pso, 1, 0)),
             ((n2, 3, n2.lay["OUT"]), (w1, 1, 0)), ((n2, 3, n2.lay["OUT"]), (w3, 1, 0)),
             ((sw, 3, 0), (w2, 1, 0)), ((w2, 3, 0), (psw, 1, 0))]
    if qin is not psq:
        edges.append(((psq, 3, 0), (qin, 1, 0)))
    if _QSM_SK["w1"] > 1 and BATCH != 32 and CONFIG.get("psum_swiglu"):
        # MM 25.185: ONE pass sums w1's and w3's partials and applies the SwiGLU (g17psum mode swiglu)
        pss = _psum_op("L%d.psum_swiglu" % l, "w1", "swiglu", _QSM_SK["w1"])
        ops = [n1, qkv, psq, fs, cvt, wo, pso, n2, w1, w3, pss, w2, psw]
        edges = [e_ for e_ in edges if e_[0][0] is not sw and e_[1][0] is not sw]
        edges += [((w1, 3, 0), (pss, 1, 0)), ((w3, 3, 0), (pss, 2, 0)), ((pss, 3, 0), (w2, 1, 0))]
    elif _QSM_SK["w1"] > 1 and BATCH != 32:
        # w1 / w3 split-K (MM 25.185): each one's partials summed into fp32 rows before the SwiGLU
        ps1 = _psum_op("L%d.psum_w1" % l, "w1", "sum", _QSM_SK["w1"])
        ps3 = _psum_op("L%d.psum_w3" % l, "w1", "sum", _QSM_SK["w3"])
        ops = [n1, qkv, psq, fs, cvt, wo, pso, n2, w1, ps1, w3, ps3, sw, w2, psw]
        edges += [((w1, 3, 0), (ps1, 1, 0)), ((w3, 3, 0), (ps3, 1, 0)), ((ps1, 3, 0), (sw, 1, 0)), ((ps3, 3, 0), (sw, 2, 0))]
    else:
        edges += [((w1, 3, 0), (sw, 1, 0)), ((w3, 3, 0), (sw, 2, 0))]
    if qin is not psq and qin not in ops:
        # the split-K branches above rebuild the list; Qwen3's QK-norm goes back ahead of the attention
        ops.insert(ops.index(fs), qin)
    # R: h fp32 rows at 0, the x16 rows at _r_res(). wo's psum reads x16 and writes h; w2's reads h and writes x16
    r_edges = [((n1, 1, n1.lay["X"]), _r_res()), ((pso, 2, 0), _r_res()), ((pso, 3, 0), 0), ((n2, 1, n2.lay["X"]), 0),
               ((psw, 2, 0), 0), ((psw, 3, 0), _r_res()), ((fs, 2, 0), _r_gen())]
    return ops, edges, r_edges


def layer_ops(l, bits, spec):
    if CONFIG.get("qsm"):
        return layer_ops_qsm(l, bits, spec)
    d = spec.d_model
    q = "q%d" % bits
    n1 = _deco("L%d.attn_norm" % l, "rmsnorm_half_d%d_g32_u16_h" % d, "norm",
               O.rmsnorm_loop_layout(d, "half", groups=32, unroll=16, hoist=True), g="L%d_g1" % l)
    qkv = _qmv_op("L%d.qkv" % l, "qmv_x16_%s_4096x2048" % q, "qmv", "L%d_qkv" % l)
    wo = _qmv_op("L%d.wo_res1" % l, "qmv_x16_res1_%s_2048x2048" % q, "res1", "L%d_wo" % l)
    n2 = _deco("L%d.ffn_norm" % l, "rmsnorm_float_d%d_g32_u8_h" % d, "norm",
               O.rmsnorm_loop_layout(d, "float", groups=32, unroll=8, hoist=True), g="L%d_g2" % l)
    ffn = _qmv_op("L%d.ffn" % l, "qmv_swiglu_x16_%s_8192x2048" % q, "swiglu", "L%d_ffn" % l)
    w2 = _qmv_op("L%d.w2_res2" % l, "qmv_x16_res2_%s_2048x8192" % q, "res2", "L%d_w2" % l)
    fs = _fused_attn_op(l)
    if getattr(M, "QK_NORM", False) and not fs.lay.get("qknorm"):
        # QK-NORM (Qwen3, MM 25.182): the per-head RMSNorm of q and k, between the qkv projection and the attention
        e = _pick("headnorm", role="qknorm")
        hn = MG.Op("L%d.qk_norm" % l, e["bundle"], "headnorm", dict(e["recipe"]["layout"]),
                   {1: 4 * M.QKV, 2: 4 * M.HEAD_DIM, 3: 4 * M.QKV}, weights="L%d_qk" % l)
        hn.bundle, hn.threads, hn.group = DELIVER_ROOT / e["bundle"], 1024, 1024
        hn.meta["phys"] = {"0": 3, "1": 1, "2": 2}
        ops = [n1, qkv, hn, fs, wo, n2, ffn, w2]
        edges = [((n1, 3, n1.lay["OUT"]), (qkv, 1, 0)), ((qkv, 3, 0), (hn, 1, 0)), ((hn, 3, 0), (fs, 1, 0)),
                 ((fs, 0, fs.lay["ATTN"]), (wo, 1, 0)), ((n2, 3, n2.lay["OUT"]), (ffn, 1, 0)), ((ffn, 3, 0), (w2, 1, 0))]
        r_edges = [((n1, 1, n1.lay["X"]), _r_res()), ((wo, 3, 0), 0), ((n2, 1, n2.lay["X"]), 0), ((w2, 3, 0), 0),
                   ((fs, 2, 0), _r_gen())]
        return ops, edges, r_edges
    ops = [n1, qkv, fs, wo, n2, ffn, w2]
    edges = [((n1, 3, n1.lay["OUT"]), (qkv, 1, 0)), ((qkv, 3, 0), (fs, 1, 0)), ((fs, 0, fs.lay["ATTN"]), (wo, 1, 0)),
             ((n2, 3, n2.lay["OUT"]), (ffn, 1, 0)), ((ffn, 3, 0), (w2, 1, 0))]
    r_edges = [((n1, 1, n1.lay["X"]), _r_res()), ((wo, 3, 0), 0), ((n2, 1, n2.lay["X"]), 0), ((w2, 3, 0), 0),
               ((fs, 2, 0), _r_gen())]
    return ops, edges, r_edges


def all_ops(bits):
    spec = M.layer_spec(0)
    R = MG.Op("R", "R", "region", {}, {1: 0, 2: 0, 3: (_r_gen() + _gen_lay()["rope_bytes"]) if GEN else R_GEN})
    ops, edges = [], []
    for l in range(M.LAYERS):
        lo, le, re = layer_ops(l, bits, spec)
        ops += lo
        edges += le
        edges += [(slot, (R, 3, off)) for slot, off in re]
    fn = _deco("head.final_norm", "rmsnorm_half_d%d_g32_u16_h" % M.D_MODEL, "norm",
               O.rmsnorm_loop_layout(M.D_MODEL, "half", groups=32, unroll=16, hoist=True), g="norm", head_out32=bits == 4)
    # the lm_head: ONE dispatch per sequence of the single-vector bundle (sequence b reads the final norm's row b and
    # writes its logits at b V), the weights shared - so the batched argmax reads the nb logit rows contiguously
    # (or, with the head config's "batch", ONE batched dispatch reading each weight word once for all nb: its x rows at
    # 4 K b are the out32 final norm's rows and its logit rows at 4 V b are the argmax's, MM 25.144.7)
    hb = (CONFIG["qmv"]["head"].get("batch") or 1) if DELIVER_ROOT else 1
    if hb > 1 and not CONFIG.get("qsm_head") and (hb != BATCH or not CONFIG["final_norm"].get("out32")):
        raise SystemExit("g17q4graph: the batched head needs head batch == batch and the out32 final norm")
    if CONFIG.get("qsm_head"):
        # THE BATCHED HEAD ON THE TENSOR UNITS (MM 25.174): one g17qsm over the vocabulary for every sequence, reading the
        # fp16 final-norm rows, its logits [16][V] fp32 where the argmax reads rows b at 4 V b
        if CONFIG["final_norm"].get("out32"):
            raise SystemExit("g17q4graph: the qsm head reads the fp16 final norm (final_norm out32 false)")
        lms = [_qsm_op("head.lm", "head", "lm")]
    else:
        lms = [_qmv_op("head.lm" if b == 0 else "head.lm.b%d" % b, "qmv_x16_q%d_92544x2048" % bits, "qmv", "lm")
               for b in range(BATCH if hb == 1 else 1)]
    lm = lms[0]
    ops += [fn] + lms
    pad_op = None
    if CONFIG.get("qsm_head") and getattr(M, "VOCAB_PAD", M.VOCAB) > M.VOCAB:
        # MM 25.189: the batched head's pad logits (rows VOCAB..VOCAB_PAD, zero weights) set to the most negative float
        e = _pick("pad_fill", role="head_pad", variant={"rows": BATCH})
        pad_op = MG.Op("head.pad_fill", e["bundle"], "pad_fill", dict(e["recipe"]["layout"]),
                       {1: 256, 2: 256, 3: 4 * M.VOCAB_PAD * BATCH})
        pad_op.bundle, pad_op.threads, pad_op.group = DELIVER_ROOT / e["bundle"], e["threadgroups"] * 32, 32
        ops.append(pad_op)
    fstride = (4 if CONFIG["final_norm"].get("out32") else 2) * M.D_MODEL
    edges += [((fn, 1, fn.lay["X"]), (R, 3, _r_res()))]
    edges += [((fn, 3, fn.lay["OUT"] + fstride * b), (lms[b], 1, 0)) for b in range(len(lms))]
    edges += [((lms[b], 3, 0), (lms[0], 3, 4 * M.VOCAB * b)) for b in range(1, len(lms))]
    if GEN:
        VP = getattr(M, "VOCAB_PAD", M.VOCAB)           # the argmax's padded width (Qwen3: 152,064)
        am = MG.Op("head.argmax_pass1", "argmax_pass1", "argmax", {}, {1: 4 * VP * BATCH, 2: 256, 3: 8 * 241 * BATCH})
        gs = MG.Op("head.gen_step", "gen_step", "gen", {}, {1: 8 * 241 * BATCH, 2: M.VOCAB * M.D_MODEL * 2, 3: R.ext[3]})
        for op, gk in ((am, "gen_argmax"), (gs, "gen_step")):
            if DELIVER_ROOT:
                e = _pick(gk, cap=CAP, variant=CONFIG.get("gen", {}))
                tpg = e.get("threads_per_group", 32)
                op.bundle, op.threads, op.group = DELIVER_ROOT / e["bundle"], e["threadgroups"] * tpg, tpg
                continue
            raise SystemExit("g17q4graph: no deliver root")
        ops += [am, gs]
        edges += [((lm, 3, 0), (am, 1, 0)), ((am, 3, 0), (gs, 1, 0)), ((gs, 3, 0), (R, 3, 0))]
        if pad_op is not None:
            edges += [((lm, 3, 0), (pad_op, 3, 0))]
    return spec, R, ops, edges


def solve(bits):
    spec, R, ops, edges = all_ops(bits)
    S = MG.Solver()
    act = set()
    for (p, ps, po), (c, cs, co) in edges:
        S.same((p.name, ps), po, (c.name, cs), co)
        act |= {(p.name, ps), (c.name, cs)}
    for o in ops:
        for slot in o.meta.get("slots", (1, 3) + ((2,) if o.kind == "frsplit" else ())):
            S.find((o.name, slot))
            act.add((o.name, slot))
    act.add(("R", 3))
    byname = {o.name: o for o in ops + [R]}
    trees = {}
    for v in act:
        r, off = S.find(v)
        trees.setdefault(r, []).append((v, off))
    base, cursor = {}, MG.GAP
    for r, members in sorted(trees.items(), key=lambda kv: str(kv[0])):
        lo = min(off for v, off in members)
        hi = max(off + byname[v[0]].ext[v[1]] for v, off in members)
        start = MG._al(cursor - lo)
        for v, off in members:
            base[v] = start + off
        cursor = start + hi + MG.GAP
    return spec, R, ops, base, MG._al(cursor)


def _wbytes(op, W):
    """buffer 2 of a qmv: W at 0, S and B at the bundle's offsets; of a norm: the carrier tile and the gain."""
    if op.kind in ("qmv", "res1", "res2", "swiglu"):
        z = np.load(W / (op.meta["weights"] + ".npz"))
        b = bytearray(op.ext[2])
        O._place(b, 0, z["W"].astype("<u4"))
        O._place(b, op.lay["S_off"], z["S"].astype("<u2"))
        O._place(b, op.lay["B_off"], z["B"].astype("<u2"))
        return bytes(b)
    if op.kind == "qsm":
        # the whole block the op's weights key names, in the qmv layout (W, then S, then B): an FFN block holds w1 then
        # w3 rows, and both of its qsm ops read the one copy (the batched graph's weights dedup)
        z = np.load(W / (op.meta["weights"] + ".npz"))
        NB, K = op.lay["block_rows"], op.lay["K"]
        Wz, Sz, Bz = (np.asarray(z[k]) for k in "WSB")
        if Wz.shape[0] < NB:
            # MM 25.196, the padded FFN: each half of the w1 / w3 block zero-padded to the qsm's rows; zero words, scales
            # and biases dequantize to exact zeros, so the pad rows' outputs are 0 and SwiGLU(0, 0) = 0
            h = Wz.shape[0] // 2
            Wz, Sz, Bz = (np.concatenate([x[:h], np.zeros((NB // 2 - h, x.shape[1]), x.dtype), x[h:],
                                          np.zeros((NB // 2 - h, x.shape[1]), x.dtype)]) for x in (Wz, Sz, Bz))
        if Wz.shape[1] * 8 < K:
            # ... and w2's K padded with zero columns (the pad activations are 0 as well)
            Wz = np.concatenate([Wz, np.zeros((Wz.shape[0], K // 8 - Wz.shape[1]), Wz.dtype)], 1)
            Sz, Bz = (np.concatenate([x, np.zeros((x.shape[0], K // 64 - x.shape[1]), x.dtype)], 1) for x in (Sz, Bz))
        if Wz.shape != (NB, K // 8):
            raise SystemExit("g17q4graph: %s holds W %s, the qsm reads %s" % (op.meta["weights"], Wz.shape, (NB, K // 8)))
        b = bytearray(op.ext[2])
        O._place(b, 0, Wz.astype("<u4"))
        O._place(b, NB * (K // 2), Sz.astype("<u2"))
        O._place(b, NB * (K // 2) + NB * (K // 32), Bz.astype("<u2"))
        return bytes(b)
    if op.kind == "headnorm":
        b = bytearray(op.ext[2])
        O._place(b, op.lay["G"], np.load(W / (op.meta["weights"] + ".npy")).astype(np.float16))
        return bytes(b)
    if op.kind in ("norm", "wnorm"):
        b = bytearray(op.lay["b_bytes"])
        if op.kind == "norm":
            O._place(b, 0, O.carrier_tiles(op.lay)[1])
        O._place(b, op.lay["G"], np.load(W / (op.meta["g"] + ".npy")).astype(np.float16))
        return bytes(b)
    return None


def build(bits):
    spec, R, ops, base, act_bytes = solve(bits)
    OUTD = out_dir(bits)
    W = OUTD / "weights"
    bdir = MG.BUNDLES
    for o in ops:
        if o.bundle is None:
            MG._author(o, spec) if o.kind != "rope" else _author_rope(o, spec)
    arenas, files, binds, warena = {ACT: act_bytes, "zero": 4 << 20}, {}, [], {}
    wseen = {}                                         # (arena, weights) -> offset: the batched head's ops share one copy
    for o in ops:
        an = "W" + o.name.split(".")[0]
        wkey = (an, o.meta.get("weights"))
        wb = None if (wkey[1] and wkey in wseen) else _wbytes(o, W)
        b2 = (an, wseen[wkey]) if (wkey[1] and wkey in wseen) else None
        if wb is not None:
            wa = warena.setdefault(an, bytearray())
            b2 = (an, len(wa))
            if wkey[1] and BATCH > 1:
                wseen[wkey] = len(wa)
            wa += wb
            wa += bytes(MG._al(len(wa)) - len(wa))
        bundle = o.bundle or (bdir / o.key)
        if o.threads is None:
            m = json.loads((bundle / "manifest.json").read_text())["tensor"]
            threads, group = m["grid"][0], m["threadgroup"][0]
        else:
            threads, group = o.threads, o.group
        if o.kind == "gen":
            b2 = ("EMB", 0)
        bind2 = (dict(arena=b2[0], offset=b2[1], bytes=o.ext[2]) if b2 else
                 dict(arena=ACT, offset=base[(o.name, 2)], bytes=o.ext[2]) if (o.name, 2) in base else
                 dict(arena="zero", offset=0, bytes=o.ext[2]))
        if o.meta.get("phys"):
            binds.append(dict(name=o.name, bundle=str(bundle), threads=threads, group=group,
                              binds={k: bind2 if v == 2 else dict(arena=ACT, offset=base[(o.name, v)], bytes=o.ext[v])
                                     for k, v in o.meta["phys"].items()}))
            continue
        if o.meta.get("slots"):
            binds.append(dict(name=o.name, bundle=str(bundle), threads=threads, group=group,
                              binds={str(k): dict(arena=ACT, offset=base[(o.name, k)], bytes=o.ext[k])
                                     for k in o.meta["slots"]}))
            continue
        binds.append(dict(name=o.name, bundle=str(bundle), threads=threads, group=group,
                          binds={"1": dict(arena=ACT, offset=base[(o.name, 1)], bytes=o.ext[1]), "2": bind2,
                                 "3": dict(arena=ACT, offset=base[(o.name, 3)], bytes=o.ext[3])}))
    for name, wa in warena.items():
        f = OUTD / ("%s.bin" % name)
        f.write_bytes(bytes(wa))
        arenas[name], files[name] = len(wa), str(f)
    byname = {o.name: o for o in ops}
    r0 = base[("R", 3)]
    g0 = json.loads((M.OUT / "graph" / "graph.json").read_text())
    if GEN:
        emb = np.load(W / "embed.npy", mmap_mode="r")
        (OUTD / "embed.f16").write_bytes(np.ascontiguousarray(emb).tobytes())
        arenas["EMB"], files["EMB"] = emb.nbytes, str(OUTD / "embed.f16")
        gl = _gen_lay()
        RG, SS = _r_gen(), gl.get("SS", 0)
        rinit = bytearray(RG + gl["rope_bytes"])
        # sequence b: its first prompt token's embedding at x row b, q0 = 0 and its prompt in the log of state block b
        for s in range(BATCH):
            prompt = _prompt(g0, s)
            O._place(rinit, _r_res() + 2 * M.D_MODEL * s, np.asarray(emb[prompt[0]], np.float16))
            log = np.full(CAP, 0xFFFFFFFF, "<u4"); log[:len(prompt)] = prompt
            O._place(rinit, RG + SS * s, np.array([0], "<u4"))
            O._place(rinit, RG + SS * s + 4, log)
        prompt = _prompt(g0)
        pos = np.arange(CAP)[:, None] * (M.ROPE_THETA ** (-np.arange(0, 128, 2, dtype=np.float64) / 128))[None, :]
        assert (SS * (BATCH - 1) + 4 + 4 * CAP) <= gl["COST"], "the token logs overlap the rope tables at this cap"
        O._place(rinit, RG + gl["COST"], np.cos(pos).astype("<f4"))      # g17modelgraph's formula, CAP rows
        O._place(rinit, RG + gl["SINT"], np.sin(pos).astype("<f4"))
        (OUTD / "R_init.bin").write_bytes(bytes(rinit))
    # THE KV CACHE STARTS AT EXACTLY ZERO (a stated precondition, MM 25.144.2): a matrix-unit prefill multiplies P = 0
    # by every row of a key block, and 0 x NaN = NaN, so never-written cache rows must be finite. The whole activation
    # arena is zero-initialised before the first dispatch; check that every attention region lies inside that range.
    zero_init = [dict(arena=ACT, offset=0, bytes=act_bytes)]
    pad_init, keep = [], []                            # persistent inits: (offset, bytes) outside every zero_init range
    for o in ops:
        if o.meta.get("qk_gain"):
            # MM 25.188: the fused attention's QK-norm gains in its binding 1 at QG, written once
            f = OUTD / ("%s_qkgain.bin" % o.name)
            f.write_bytes(np.load(W / (o.meta["qk_gain"] + ".npy")).astype("<f2").tobytes())
            p0 = base[(o.name, 1)] + o.lay["QG"]
            pad_init.append(dict(arena=ACT, offset=p0, file=str(f)))
            keep.append((p0, 4 * M.HEAD_DIM))
    if getattr(M, "VOCAB_PAD", M.VOCAB) > M.VOCAB:
        # THE ARGMAX PAD (MM 25.182): the logits past the vocabulary hold the most negative finite float, written once by
        # arena_init and outside every zero_init range (decodegen's pipelined reinit re-zeroes zero_init only)
        lm0 = [o for o in ops if o.name == "head.lm"][0]
        p0, pn = base[(lm0.name, 3)] + 4 * M.VOCAB, 4 * (M.VOCAB_PAD - M.VOCAB)
        (OUTD / "logits_pad.bin").write_bytes(np.full(pn // 4, np.finfo(np.float32).min, "<f4").tobytes())
        pad_init.append(dict(arena=ACT, offset=p0, file=str(OUTD / "logits_pad.bin")))
        keep.append((p0, pn))
    if keep:
        zero_init, cur = [], 0
        for p0, pn in sorted(keep):
            if p0 > cur:
                zero_init.append(dict(arena=ACT, offset=cur, bytes=p0 - cur))
            cur = max(cur, p0 + pn)
        zero_init.append(dict(arena=ACT, offset=cur, bytes=act_bytes - cur))
    for d in binds:
        if "attn" not in d["name"]:
            continue
        for bd in d["binds"].values():
            if bd["arena"] == ACT and any(bd["offset"] <= p0 and p0 + pn <= bd["offset"] + bd["bytes"] for p0, pn in keep):
                continue                               # holds a persistent init (the fused QK-norm's gains, MM 25.188)
            if bd["arena"] == ACT and not any(z["arena"] == ACT and z["offset"] <= bd["offset"] and
                                              bd["offset"] + bd["bytes"] <= z["offset"] + z["bytes"] for z in zero_init):
                raise ValueError("%s binds %s outside every zero_init range" % (d["name"], bd))
    per_token = [dict(arena=ACT, offset=r0 + 8192, bytes=4096, source="embedding_row")]
    for l in range(M.LAYERS):
        if GEN:
            per_token = []
            break
        if ROPE_FUSED:
            sp = byname["L%d.attn_rope_split" % l]
            per_token += [dict(arena=ACT, offset=base[(sp.name, 2)], bytes=256, source="rope_cos_row"),
                          dict(arena=ACT, offset=base[(sp.name, 2)] + sp.lay["SIN"], bytes=256, source="rope_sin_row"),
                          dict(arena=ACT, offset=base[(sp.name, 1)] + sp.lay["LEN"], bytes=4, source="split_len_word")]
            continue
        rp = byname["L%d.rope" % l]
        ra, rc = base[(rp.name, 1)], base[(rp.name, 3)]
        per_token += [dict(arena=ACT, offset=ra + rp.lay["LEN"], bytes=4, source="rope_len_word"),
                      dict(arena=ACT, offset=ra + rp.lay["COS"], bytes=256, source="rope_cos_row"),
                      dict(arena=ACT, offset=ra + rp.lay["SIN"], bytes=256, source="rope_sin_row"),
                      dict(arena=ACT, offset=rc + rp.lay["QF"] + 4096, bytes=4, source="split_len_word")]
    lm = byname["head.lm"]
    prefill = None
    pcfg = prefill_cfg()
    # the model's widths for a prefill section (and the verify step's): Qwen3's qkv, attention-output and FFN widths and
    # its QK-norm with the fused attention's gains (MM 25.188); None is InternLM2's defaults
    pf_dims = (dict(qkv=M.QKV, hd=M.HEADS * M.HEAD_DIM, ffn=getattr(M, "FFN_PREFILL", M.FFN), qknorm=True,
                    # the fused attention's gains (MM 25.188): the prefill QK-norm reads the same words
                    qk_gain={int(o.name.split(".")[0][1:]): dict(arena=ACT, offset=base[(o.name, 1)] + o.lay["QG"],
                                                                 bytes=4 * M.HEAD_DIM)
                             for o in ops if o.meta.get("qk_gain")})
               if getattr(M, "QK_NORM", False) else None)
    if pcfg:
        import g17prefillgraph as PG
        emb_pf = np.load(W / "embed.npy", mmap_mode="r")
        pf = PG.section(_pick_rooted, bits, CAP, pcfg, _prompt(g0), {d["name"]: d for d in binds}, W, OUTD,
                        emb_pf, byname["head.final_norm"].lay["X"],
                        logits=dict(arena=ACT, offset=base[(lm.name, 3)], bytes=4 * M.VOCAB),
                        # the qsm route (MM 25.202) runs the unpadded FFN; FFN_PREFILL's padding is the W16 GEMM grid's
                        dims=dict(pf_dims, ffn=M.FFN) if (pf_dims and pcfg.get("qsm")) else pf_dims)
        arenas.update(pf["arenas"]); files.update(pf["files"])
        zero_init = zero_init + pf["zero_init"]
        prefill = pf["prefill"]
    spec = None
    if CONFIG.get("spec"):
        # THE SPECULATIVE VERIFY STEP (MM 25.203): 16 rows at runtime positions q0 .. q0 + 15 against the decode cache -
        # the prefill section at M 16 (the scalar append and attention, which read p0 from decode's q0 word and write
        # what decode writes; the qsm projections), then a 16-row final norm and the qsm head. The driver
        # (tools/g17specgen.m) writes the rows' embeddings and reads the 16 logit rows.
        import g17prefillgraph as PG
        emb_v = np.load(W / "embed.npy", mmap_mode="r")
        saved = PG.PF, PG.PE
        PG.PF, PG.PE = "PFV", "PEV"
        try:
            # the verify step's widths: the model's (its qsm route runs the unpadded FFN, not FFN_PREFILL's)
            vcfg = _spec_cfg()
            R, QV = vcfg["M"], bool(vcfg.get("qmvw"))
            vs = PG.section(_pick_rooted, bits, CAP, vcfg, [0] * R, {d["name"]: d for d in binds}, W, OUTD,
                            emb_v, byname["head.final_norm"].lay["X"], dims=dict(pf_dims, ffn=M.FFN) if pf_dims else None)
        finally:
            PG.PF, PG.PE = saved
        arenas.update(vs["arenas"]); files.update(vs["files"])
        zero_init = zero_init + vs["zero_init"]
        vlayers = [st for st in vs["prefill"]["steps"] if not st.get("setup") and st["dispatches"]][0]["dispatches"]
        x16 = [r for r in vs["prefill"]["dump"] if r["name"] == "x_last_chunk"][0]
        fn_e = _pick_rooted("norm", role="final_norm", variant={"out32": QV, "seed": False, "batch": R})
        hd_e = (_pick_rooted("qmvw", bits=4, role="head", variant={"nb": R}) if QV else
                _pick_rooted("qsm", bits=4, role="head", variant={"sk": 1, "xrows": True, "h16": True}))
        hl = hd_e["recipe"]["layout"]
        fin = {d["name"]: d for d in binds}
        nrm_at = MG._al(MG.GAP + fn_e["OUT"])
        lg_at = MG._al(nrm_at + R * M.D_MODEL * (4 if QV else 2) + MG.GAP)
        # the rows' argmax pass 1 on the GPU (MM 25.205) when one is delivered at the head's row width
        try:
            am = _pick_rooted("argmax_rows", variant={"rows": R, "V": hl["N"]})
        except KeyError:
            am = None
        pr_at = MG._al(lg_at + hl["c_bytes"] + MG.GAP)
        arenas["SPEC"] = MG._al(pr_at + (am["pairs_bytes"] if am else 0) + MG.GAP)
        wh = fin["head.lm"]["binds"]["2"]
        tail = [dict(name="V.final_norm", bundle=str(Path(fn_e["_root"]) / fn_e["bundle"]), threads=fn_e["threadgroups"] * fn_e.get("threads_per_group", 32),
                     group=fn_e.get("threads_per_group", 32),
                     binds={"0": dict(arena="SPEC", offset=nrm_at - fn_e["OUT"], bytes=fn_e["OUT"] + R * M.D_MODEL * (4 if QV else 2)),
                            "1": dict(arena=x16["arena"], offset=x16["offset"] - fn_e["X"], bytes=fn_e["X"] + R * M.D_MODEL * 2),
                            "2": fin["head.final_norm"]["binds"]["2"]}),
                dict(name="V.head", bundle=str(Path(hd_e["_root"]) / hd_e["bundle"]),
                     threads=(hd_e["threadgroups"] * 64 if QV else hd_e["threadgroups"] * 32), group=64 if QV else 32,
                     binds={"1": dict(wh, bytes=hl["a_bytes"]), "2": dict(arena="SPEC", offset=nrm_at, bytes=hl["b_bytes"]),
                            "3": dict(arena="SPEC", offset=lg_at, bytes=hl["c_bytes"])})]
        if am:
            tail.append(dict(name="V.argmax", bundle=str(Path(am["_root"]) / am["bundle"]), threads=am["threadgroups"] * 32,
                             group=32, binds={"1": dict(arena="SPEC", offset=lg_at, bytes=R * 4 * hl["N"]),
                                              "3": dict(arena="SPEC", offset=pr_at, bytes=am["pairs_bytes"])}))
        spec = dict(rows=R, dispatches=vlayers + tail, x16=dict(arena=x16["arena"], offset=x16["offset"]),
                    **({"pairs": dict(arena="SPEC", offset=pr_at, G=am["G"], V=am["V"])} if am else {}),
                    logits=dict(arena="SPEC", offset=lg_at, row_bytes=4 * hl["N"]), vocab=M.VOCAB,
                    # decode's layer-0 input row (the gen step writes the next token's embedding there)
                    r_x16=dict(arena=fin["L0.attn_norm"]["binds"]["1"]["arena"],
                               offset=fin["L0.attn_norm"]["binds"]["1"]["offset"] + byname["L0.attn_norm"].lay["X"]),
                    ngram=int(CONFIG["spec"].get("ngram", 3)), min_ngram=int(CONFIG["spec"].get("min_ngram", 2)),
                    max_draft=int(CONFIG["spec"].get("max_draft", 15)))
    init = [dict(arena=k, offset=0, file=v) for k, v in files.items()] + pad_init
    if GEN:
        init.append(dict(arena=ACT, offset=r0, file=str(OUTD / "R_init.bin")))
    g = dict(model=M.MODEL_ID, bits=bits, arenas=arenas, arena_init=init,
             # a device-resident graph advances its own q0, log and KV cache every step, so the executor must not
             # run warm-up steps on it (they would consume the first tokens)
             **({"gen_region": dict(arena=ACT, offset=r0 + _r_gen(), log_offset=4, q0_offset=0, log_entries=CAP,
                                    # batched: sequence b's q0 and log in the state block at offset + state_stride b
                                    **(dict(batch=BATCH, state_stride=_gen_lay()["SS"]) if BATCH > 1 else {})),
                 "warmup": 0} if GEN else {}),
             **({"batch": BATCH, "prompts": [_prompt(g0, s) for s in range(BATCH)]} if BATCH > 1 else {}),
             zero_init=zero_init, dispatches=binds, per_token_writes=per_token, **({"prefill": prefill} if prefill else {}),
             **({"spec": spec} if spec else {}),
             # decode's layer-0 input row (the gen step writes the next token's embedding there): a driver that sets the
             # position itself (tools/g17specgen, a draft model's catch-up) writes it
             **({"decode_x_row": dict(arena=_b0["arena"], offset=_b0["offset"] + byname["L0.attn_norm"].lay["X"])}
                if (_b0 := next((d["binds"]["1"] for d in binds if d["name"] == "L0.attn_norm"), None)) and BATCH == 1 else {}),
             tables=dict(embedding=str(W / "embed.npy"), rope=g0["tables"]["rope"]),
             rope_cos_file=g0["rope_cos_file"], rope_sin_file=g0["rope_sin_file"], prompt_ids=_prompt(g0),
             logits_readback=dict(arena=ACT, offset=base[(lm.name, 3)], bytes=4 * M.VOCAB), logits_dtype="f32",
             vocab=M.VOCAB, d_model=M.D_MODEL, capacity=CAP, notes=g0["notes"])
    (OUTD / "graph.json").write_text(json.dumps(g, indent=1) + "\n")
    print("graph.json: %d dispatches, act %.1f MB, weights %.0f MB" % (
        len(binds), act_bytes / 1e6, sum(v for k, v in arenas.items() if k.startswith("W")) / 1e6))
    return g, ops


def _author_rope(op, spec):
    path = MG.BUNDLES / op.key
    if (path / "manifest.json").exists():
        return path
    prog = O.build_rope_append(op.lay, D.q_scale(spec))
    a, b, c = O._buffers(op.lay)
    O._with_carrier(op.lay, (a, b, c))
    O.author(path, op.lay, prog, bytes(a), bytes(b), bytes(c), extra={"program": op.key})
    return path


# ---------------------------------------------------------------------------------------------------
# the simulator

def _unpack(Wd, bits, N, K):
    """The packed fields, low field first, as uint8 (the same values the int64 form held; every reader converts them,
    and uint32 shifts with a uint8 result take a fifth of the time and an eighth of the memory)."""
    per = 32 // bits
    w = np.asarray(Wd).reshape(N, K // per).astype(np.uint32, copy=False)
    sh = (bits * np.arange(per)).astype(np.uint32)
    return ((w[:, :, None] >> sh) & np.uint32((1 << bits) - 1)).astype(np.uint8).reshape(N, K)


class QSim(MG.Sim):
    def __init__(self, g):
        acts = [i for i in g["arena_init"] if i["arena"] == ACT]
        g2 = dict(g, arena_init=[i for i in g["arena_init"] if i["arena"] != ACT])
        MG.Sim.__init__(self, g2)
        for i in acts:
            raw = np.fromfile(i["file"], np.uint8)
            self.arena[ACT][i["offset"]:i["offset"] + raw.size] = raw

    def run(self, d, op, pos):
        import g17qmv as Q
        import g17attn as A
        if op.kind == "wnorm":
            c, a = (d["binds"][k]["offset"] for k in ("0", "1"))
            b2, lay, sp = d["binds"]["2"], op.lay, M.layer_spec(pos)
            n = sp.d_model
            x = self.v(ACT, a + lay["X"], n, np.float16 if lay["in_dtype"] == "half" else "<f4").astype(F32)
            gg = self.v(b2["arena"], b2["offset"] + lay["G"], n, np.float16).astype(F32)
            y = np.asarray(O.rmsnorm_wide_reference(x, gg, sp, seed=bool(op.meta.get("seed"))), np.float16)
            self.w(c + lay["OUT"], y.astype("<f4") if op.meta.get("out32") else y)
            return
        if op.kind == "ffused":
            w0, a1, g2 = (d["binds"][k]["offset"] for k in ("0", "1", "2"))
            lay = op.lay
            qkv = self.v(ACT, a1, 32 * 128, "<f4")
            q0 = int(self.v(ACT, g2, 1, "<u4")[0])
            cs = self.v(ACT, g2 + lay["COST"] + q0 * 256, 64, "<f4")
            sn = self.v(ACT, g2 + lay["SINT"] + q0 * 256, 64, "<f4")
            Kc = self.v(ACT, w0 + lay["KOFF"], 8 * lay["cap"] * 128, np.float16).reshape(8, lay["cap"], 128)
            Vc = self.v(ACT, w0 + lay["VOFF"], 8 * lay["cap"] * 128, np.float16).reshape(8, lay["cap"], 128)
            out, parts, K, V = A.attn_rope_reference(lay, qkv, cs, sn, Kc, Vc, q0, partials=True)
            if not lay.get("wide"):                      # the wide form merges in threadgroup memory: no partials
                self.w(w0 + lay["P"], np.asarray(parts, "<f4"))
            q0c = min(q0, lay["cap"] - 1)
            for g in range(8):
                self.w(w0 + lay["KOFF"] + (g * lay["cap"] + q0c) * 256, np.asarray(K[g, q0c], np.float16))
                self.w(w0 + lay["VOFF"] + (g * lay["cap"] + q0c) * 256, np.asarray(V[g, q0c], np.float16))
            out16 = np.asarray(out, np.float16)
            self.w(w0 + lay["ATTN"], out16.astype("<f4") if op.meta.get("attn32") else out16)
            return
        if op.meta.get("phys") and op.kind != "wnorm":
            d = dict(d, binds={str(v): d["binds"][k] for k, v in op.meta["phys"].items()})
        b1, b2, b3 = (d["binds"][k] for k in ("1", "2", "3"))
        a, c = b1["offset"], b3["offset"]
        if op.kind in ("qmv", "res1", "res2", "swiglu"):
            lay = op.lay
            N, K, bits = lay["Nout"], lay["Kq"], lay["bits"]
            wa, wo = b2["arena"], b2["offset"]
            s16 = self.v(wa, wo + lay["S_off"], N * K // 64, "<u2").reshape(N, K // 64)
            b16 = self.v(wa, wo + lay["B_off"], N * K // 64, "<u2").reshape(N, K // 64)
            x = self.v(ACT, a, K, "<f4" if op.meta.get("x32") else np.float16).astype(F32)
            # the weight arenas are never written by a step (only ACT is), so each matrix's fields are unpacked once
            # per simulator (uint8, N K bytes each)
            key = (wa, wo, N, K, bits)
            cache = self.__dict__.setdefault("_fields", {})
            q = cache.get(key) if wa != ACT else None
            if q is None:
                q = _unpack(self.v(wa, wo, N * K * bits // 32, "<u4"), bits, N, K)
                if wa != ACT:
                    cache[key] = q
            if op.kind == "swiglu":
                act = Q.qmv_swiglu_reference(lay, x, q, s16, b16)
                self.w(c, np.asarray(act, np.float16).astype("<f4") if op.meta.get("act32") else act)
                return
            # each op in its bundle's own fp32 order: dequantize-once (the dq batch variant and its single form),
            # split-K or the single qmv2 order
            y = (Q.qmv_dq_reference if lay.get("dequant_once") else Q.qmv2_ksplit_reference if lay.get("ksplit")
                 else Q.qmv2_reference_fast)(lay, x, q, s16, b16)
            if op.kind == "qmv":
                self.w(c, np.asarray(y, "<f4"))
            elif op.kind == "res1":
                self.w(c, Q.residual_reference(lay, y, x16=self.v(ACT, c + 8192, 2048, np.float16)))
            else:
                self.w(c + 8192, Q.residual_reference(lay, y, h32=self.v(ACT, c, 2048, "<f4")))
            return
        if op.kind == "fsplit":
            lay = op.lay
            q16 = self.v(ACT, a, 16 * 128, np.float16).reshape(16, 128)
            q0 = int(self.v(ACT, a + lay["LEN"], 1, "<u4")[0])
            Kc = self.v(ACT, b2["offset"], 8 * lay["cap"] * 128, np.float16).reshape(8, lay["cap"], 128)
            Vc = self.v(ACT, b2["offset"] + lay["VOFF"], 8 * lay["cap"] * 128, np.float16).reshape(8, lay["cap"], 128)
            out, parts = A.attn_reference(lay, q16, Kc, Vc, q0, partials=True)
            self.w(c, np.asarray(parts, "<f4"))
            self._attn_out = out
            return
        if op.kind == "frtsplit":
            lay = op.lay
            qkv = self.v(ACT, a, 32 * 128, "<f4")
            g0 = b2["offset"]
            q0 = int(self.v(ACT, g0, 1, "<u4")[0])
            cs = self.v(ACT, g0 + lay["COST"] + q0 * 256, 64, "<f4")
            sn = self.v(ACT, g0 + lay["SINT"] + q0 * 256, 64, "<f4")
            Kc = self.v(ACT, c + lay["KOFF"], 8 * lay["cap"] * 128, np.float16).reshape(8, lay["cap"], 128)
            Vc = self.v(ACT, c + lay["VOFF"], 8 * lay["cap"] * 128, np.float16).reshape(8, lay["cap"], 128)
            out, parts, K, V = A.attn_rope_reference(lay, qkv, cs, sn, Kc, Vc, q0, partials=True)
            self.w(c, np.asarray(parts, "<f4"))
            q0c = min(q0, 271)
            for g in range(8):
                self.w(c + lay["KOFF"] + (g * lay["cap"] + q0c) * 256, np.asarray(K[g, q0c], np.float16))
                self.w(c + lay["VOFF"] + (g * lay["cap"] + q0c) * 256, np.asarray(V[g, q0c], np.float16))
            self._attn_out = out
            return
        if op.kind == "argmax":
            self._argmax = int(np.argmax(self.v(ACT, a, M.VOCAB, "<f4")))
            return
        if op.kind == "gen":
            r = c
            q0 = int(self.v(ACT, r + R_GEN, 1, "<u4")[0])
            forced = int(self.v(ACT, r + R_GEN + 4 + 4 * (q0 + 1), 1, "<u4")[0]) if q0 + 1 < CAP else 0xFFFFFFFF
            tok = forced if forced != 0xFFFFFFFF else self._argmax
            if q0 + 1 < CAP:
                self.w(r + R_GEN + 4 + 4 * (q0 + 1), np.array([tok], "<u4"))
            self.w(r + 8192, np.asarray(self.embed[tok], np.float16))
            self.w(r + R_GEN, np.array([min(q0 + 1, CAP - 1)], "<u4"))
            return
        if op.kind == "frsplit":
            lay = op.lay
            qkv = self.v(ACT, a, 32 * 128, "<f4")
            q0 = int(self.v(ACT, a + lay["LEN"], 1, "<u4")[0])
            cs, sn = self.v(ACT, b2["offset"], 64, "<f4"), self.v(ACT, b2["offset"] + lay["SIN"], 64, "<f4")
            Kc = self.v(ACT, c + lay["KOFF"], 8 * lay["cap"] * 128, np.float16).reshape(8, lay["cap"], 128)
            Vc = self.v(ACT, c + lay["VOFF"], 8 * lay["cap"] * 128, np.float16).reshape(8, lay["cap"], 128)
            out, parts, K, V = A.attn_rope_reference(lay, qkv, cs, sn, Kc, Vc, q0, partials=True)
            self.w(c, np.asarray(parts, "<f4"))
            q0c = min(q0, 271)
            for g in range(8):
                self.w(c + lay["KOFF"] + (g * lay["cap"] + q0c) * 256, np.asarray(K[g, q0c], np.float16))
                self.w(c + lay["VOFF"] + (g * lay["cap"] + q0c) * 256, np.asarray(V[g, q0c], np.float16))
            self._attn_out = out
            return
        if op.kind == "fmerge":
            self.w(c, np.asarray(self._attn_out, np.float16))
            return
        if op.kind == "rope":
            lay = op.lay
            L = int(self.v(ACT, a + lay["LEN"], 1, "<u4")[0])
            sp = M.layer_spec(L)
            qkv = self.v(ACT, a + lay["QKV"], sp.qkv_width, "<f4")
            cs, sn = self.v(ACT, a + lay["COS"], 64, "<f4"), self.v(ACT, a + lay["SIN"], 64, "<f4")
            k8 = self.v(ACT, c + lay["KC"], 8 * 272 * 128, np.float16).reshape(8, 272, 128).astype(F32)
            v8 = self.v(ACT, c + lay["VC"], 8 * 272 * 128, np.float16).reshape(8, 272, 128).astype(F32)
            ref = D.stage_rope_append(sp, qkv, cs, sn, np.repeat(k8, 2, 0)[:, :L], np.repeat(v8, 2, 0)[:, :L])
            self.poison(c, O._carrier_bytes(lay["groups"]))
            self.w(c + lay["QF"], ref["q16"].astype(np.float16))
            for g in range(8):
                self.w(c + lay["KC"] + (g * 272 + L) * 256, ref["k_new"][2 * g].astype(np.float16))
                self.w(c + lay["VC"] + (g * 272 + L) * 256, ref["v_new"][2 * g].astype(np.float16))
            return
        return MG.Sim.run(self, d, op, pos)


def simulate(bits, tokens):
    g = json.loads((out_dir(bits) / "graph.json").read_text())
    spec, R, ops, base, act_bytes = solve(bits)
    sim = QSim(g)
    prompt = g["prompt_ids"]
    toks, logits, nxt = [], [], None
    for pos in range(len(prompt) + tokens):
        tok = prompt[pos] if pos < len(prompt) else nxt
        t0 = time.time()
        lg = sim.token(tok, pos, ops)
        if GEN:
            gr = g["gen_region"]
            log = sim.v(ACT, gr["offset"] + 4, CAP, "<u4")
            if pos + 1 < CAP and pos >= len(prompt) - 1 and int(log[pos + 1]) != int(np.argmax(lg)):
                print("GEN MISMATCH at pos %d: log %d, argmax %d" % (pos, int(log[pos + 1]), int(np.argmax(lg))))
        nxt = int(np.argmax(lg))
        logits.append(lg)
        if pos >= len(prompt) - 1:
            toks.append(nxt)
        print("pos %d tok %d -> %d (%.0f s)" % (pos, tok, nxt, time.time() - t0), flush=True)
    np.savez(out_dir(bits) / "simulate.npz", tokens=np.array(toks), logits=np.stack(logits))
    print(json.dumps(dict(out_ids=toks)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prepare", "build", "simulate"))
    ap.add_argument("--bits", type=int, choices=(4, 8), default=4)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--deliver", type=Path, help="the deliver root (tools/g17deliver.py build --out); build and simulate")
    ap.add_argument("--config", type=Path, help="the model config (tools/models/*.json); build and simulate")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        return prepare(a.bits)
    if (a.deliver is None) != (a.config is None):
        ap.error("--deliver and --config go together")
    if a.deliver is not None:
        configure(a.deliver, a.config)
    _require()
    if a.cmd == "build":
        build(a.bits)
        return 0
    simulate(a.bits, a.tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
