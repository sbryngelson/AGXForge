#!/usr/bin/env python3
"""A SECOND ARCHITECTURE THROUGH THE COMPILER (MM 25.182): Qwen3-0.6B at 4 bits, greedy decode on the GPU, every
dispatch a program this compiler built, tokens against mlx-lm's on the same checkpoint.

Qwen3-0.6B is not InternLM2: d_model 1,024 (not 2,048), 28 layers, an FFN of 3,072, RMSNorm eps 1e-6, a per-head
RMSNorm on q and k before RoPE (QK-norm, which InternLM2 does not have), a tied embedding over a 151,936-token
vocabulary, and a query projection wider than d_model (16 x 128 = 2,048). Its attention shape (16 query heads, 8 KV
heads of 128, RoPE theta 1e6) is InternLM2's, so the attention kernel is the delivered one. Everything else is built at
Qwen3's shapes by `g17deliver build --arch qwen3`, plus the one kernel InternLM2 has no use for, QK-norm (here).

    python3 tools/g17qwen3.py prepare --mlx DIR      the mlx_lm.convert -q checkpoint's tensors in our row order
    python3 tools/g17qwen3.py deliver --out ROOT      QK-norm, verified bit-exact on hardware, added to ROOT's index
    python3 tools/g17qwen3.py build --deliver ROOT --prompt-ids 1,2,3 --out DIR    graph.json + weight arenas

A LAYER is 8 dispatches: attn_norm -> qkv qmv -> QK-norm -> attention (RoPE, append, split and merge) ->
wo qmv + residual1 -> ffn_norm -> w1+w3+SwiGLU qmv -> w2 qmv + residual2. The residual region R is InternLM2's with
d 1,024 (h fp32 at R + 0, x fp16 at R + 8,192, the generation state at R + 12,288).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import g17decodeops as O  # noqa: E402

F32 = np.float32
D_MODEL, LAYERS, HEADS, KV_HEADS, HEAD_DIM, FFN, VOCAB = 1024, 28, 16, 8, 128, 3072, 151936
VOCAB_PAD = 152064                    # the argmax's 198 chunks of 768; logits past VOCAB hold the most negative float
EPS = 1e-6
ROPE_THETA = 1e6
QKV = (HEADS + 2 * KV_HEADS) * HEAD_DIM
FFN_DIM = FFN
FFN_PREFILL = 4096                    # the prefill's FFN, zero-padded (exact) to the GEMM grid's power of two (MM 25.188)
QK_NORM = True                        # g17q4graph inserts the per-head RMSNorm of q and k after the qkv projection
MODEL_ID = "Qwen/Qwen3-0.6B"
OUT = ROOT / "results" / "g17-model-qwen3"
TIED = True                           # the head is the embedding's own q4 trio (0.6B); an untied head has lm_head

# THE QWEN3 FAMILY'S SHAPES. configure() rebinds the constants above, so every reader (g17q4graph's M, the deliverer's
# references) sees one model. qwen3-0.6b is the default and is byte-for-byte what 25.182-25.189 built. qwen3-8b has 32
# query heads against 8 KV heads (GQA 4), d_model 4,096, 36 layers and an untied head.
MODELS = {
    "qwen3-0.6b": dict(D_MODEL=1024, LAYERS=28, HEADS=16, KV_HEADS=8, HEAD_DIM=128, FFN=3072, VOCAB=151936,
                       VOCAB_PAD=152064, FFN_PREFILL=4096, TIED=True, MODEL_ID="Qwen/Qwen3-0.6B", OUT="g17-model-qwen3"),
    "qwen3-8b": dict(D_MODEL=4096, LAYERS=36, HEADS=32, KV_HEADS=8, HEAD_DIM=128, FFN=12288, VOCAB=151936,
                     VOCAB_PAD=152064, FFN_PREFILL=16384, TIED=False, MODEL_ID="Qwen/Qwen3-8B", OUT="g17-model-qwen3-8b"),
}
MODEL = "qwen3-0.6b"


def configure(model="qwen3-0.6b"):
    """Bind the module's shape constants to one Qwen3 model (MODELS)."""
    global MODEL, D_MODEL, LAYERS, HEADS, KV_HEADS, HEAD_DIM, FFN, VOCAB, VOCAB_PAD, FFN_PREFILL, TIED, MODEL_ID, OUT
    global QKV, FFN_DIM, PAD
    m = MODELS[model]
    MODEL = model
    D_MODEL, LAYERS, HEADS, KV_HEADS, HEAD_DIM = m["D_MODEL"], m["LAYERS"], m["HEADS"], m["KV_HEADS"], m["HEAD_DIM"]
    FFN, VOCAB, VOCAB_PAD, FFN_PREFILL, TIED = m["FFN"], m["VOCAB"], m["VOCAB_PAD"], m["FFN_PREFILL"], m["TIED"]
    MODEL_ID, OUT = m["MODEL_ID"], ROOT / "results" / m["OUT"]
    QKV = (HEADS + 2 * KV_HEADS) * HEAD_DIM
    FFN_DIM = FFN
    PAD = VOCAB_PAD - VOCAB


def layer_spec(length):
    """g17realmodel.layer_spec's fields at Qwen3's shapes, as a plain record: g17decodestep.LayerSpec requires
    n_heads x head_dim == d_model, which is InternLM2's shape and not Qwen3's (16 x 128 = 2,048 against d_model 1,024).
    The delivered graph reads only d_model from it."""
    from types import SimpleNamespace
    return SimpleNamespace(d_model=D_MODEL, n_heads=HEADS, head_dim=HEAD_DIM, ffn_dim=FFN, kv_len=length,
                           storage="half", norm_eps=EPS, rope_base=ROPE_THETA, k_route=True, n_kv_heads=KV_HEADS,
                           kv_heads=KV_HEADS)


# ---------------------------------------------------------------------------------------------------------- weights
def prepare(mlx_dir, prompt_ids):
    """The mlx_lm.convert -q checkpoint as g17q4graph's weights (OUT/graph_q4/weights) and the base graph record
    (OUT/graph/graph.json: the prompt and the notes g17q4graph copies). Only rows move: q | k | v concatenated into
    the fused qkv (4,096 rows), gate | up into the fused SwiGLU (6,144). The head is the tied embedding's own q4 trio
    (what mlx-lm's as_linear multiplies by); the embedding row gather reads mx.dequantize of the same trio, fp16."""
    import mlx.core as mx
    t = {}
    for f in sorted(Path(mlx_dir).glob("*.safetensors")):
        t.update(mx.load(str(f)))
    W = OUT / "graph_q4" / "weights"
    W.mkdir(parents=True, exist_ok=True)

    def u16(a):
        return np.array(a.view(mx.uint16))

    def trio(prefix):
        return np.array(t[prefix + ".weight"]), u16(t[prefix + ".scales"]), u16(t[prefix + ".biases"])

    def cat(*prefixes):
        parts = [trio(p) for p in prefixes]
        return dict(W=np.concatenate([x[0] for x in parts]), S=np.concatenate([x[1] for x in parts]),
                    B=np.concatenate([x[2] for x in parts]))

    def f16(k):
        return np.array(t[k].astype(mx.float16))
    for l in range(LAYERS):
        p = "model.layers.%d." % l
        np.savez(W / ("L%d_qkv.npz" % l), **cat(p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"))
        np.savez(W / ("L%d_wo.npz" % l), **dict(zip("WSB", trio(p + "self_attn.o_proj"))))
        np.savez(W / ("L%d_ffn.npz" % l), **cat(p + "mlp.gate_proj", p + "mlp.up_proj"))
        np.savez(W / ("L%d_w2.npz" % l), **dict(zip("WSB", trio(p + "mlp.down_proj"))))
        np.save(W / ("L%d_g1.npy" % l), f16(p + "input_layernorm.weight"))
        np.save(W / ("L%d_g2.npy" % l), f16(p + "post_attention_layernorm.weight"))
        np.save(W / ("L%d_qk.npy" % l), np.concatenate([f16(p + "self_attn.q_norm.weight"),
                                                        f16(p + "self_attn.k_norm.weight")]))
    np.save(W / "norm.npy", f16("model.norm.weight"))
    # the head: the tied embedding's trio (0.6B), or the untied lm_head's (8B)
    np.savez(W / "lm.npz", **dict(zip("WSB", trio("model.embed_tokens" if TIED else "lm_head"))))
    e = t["model.embed_tokens.weight"]
    emb = mx.dequantize(e, t["model.embed_tokens.scales"], t["model.embed_tokens.biases"], group_size=64, bits=4)
    np.save(W / "embed.npy", np.array(emb.astype(mx.float16)))
    G = OUT / "graph"
    G.mkdir(parents=True, exist_ok=True)
    (G / "graph.json").write_text(json.dumps(dict(
        model=MODEL_ID, prompt_ids=list(prompt_ids), tables=dict(rope="device-resident (the attention's tables)"),
        rope_cos_file="", rope_sin_file="",
        notes="Qwen3-0.6B q4 (MM 25.182): the device-resident graph; embedding_row: fp16 row of the dequantized tied "
              "embedding at the token; one command buffer per token."), indent=1) + "\n")
    print("prepared", W)


# ---------------------------------------------------------------------------------------------------------- QK-norm
def headnorm_layout(rows=1):
    """qkv fp32 [rows][4096] in at binding 1 (X), the gains fp16 at binding 2 (q_norm [128] then k_norm [128]), the
    normalized qkv fp32 [rows][4096] out at binding 0 (OUT): q heads and k heads normalized, v heads copied. rows > 1
    (the prefill, MM 25.188): threadgroup r normalizes row r."""
    lay = dict(op="headnorm", X=0, G=0, OUT=0, rows=rows, a_bytes=4 * QKV * rows, b_bytes=4 * HEAD_DIM,
               c_bytes=4 * QKV * rows)
    nh = HEADS + 2 * KV_HEADS
    if nh > 32:
        # MORE HEADS THAN A THREADGROUP HAS SIMDGROUPS (Qwen3-8B: 32 q + 8 k + 8 v = 48): two threadgroups a row, each of
        # nh / 2 simdgroups (24, 768 threads); threadgroup 2 r + p takes heads 24 p .. 24 p + 23 of row r
        if nh % 2 or nh // 2 > 32:
            raise ValueError("headnorm: %d heads do not split into two threadgroups" % nh)
        lay.update(parts=2, per=nh // 2)
    lay.update(groups=rows * lay.get("parts", 1), tpg=32 * lay.get("per", nh))
    return lay


def build_headnorm(lay, eps=EPS):
    """ONE threadgroup of 1,024 threads, simdgroup s = head s (0..15 q, 16..23 k, 24..31 v): lane l owns dims 4l..4l+3.
    Its squares are summed in dim order (the first a product, then fadd), the lanes combine with the row then column
    butterflies (build_rmsnorm_wide's), mean = sum x (1/128), r = the corrected rsqrt of mean + eps, and each dim is
    (x r) g - g the q gain for a q head, the k gain for a k head. A v head is copied. The store is a counted loop over
    the lane's four dims (the cooperative class is witnessed above 31 instructions only with a back edge)."""
    from agxforge.g17 import cc, ir, tensorreduce as TR
    _c, _cf = O._c, O._cf
    c = ir.Buffer("C", 0, elem=ir.F32); a = ir.Buffer("A", 1, elem=ir.F16); bb = ir.Buffer("B", 2, elem=ir.F16)
    fn = ir.Function("tensor_gemm_generic_runtime_demo", [c, a, bb])
    # no threadgroup memory: each head reduces within its own simdgroup
    b = ir.Builder(fn, fn.block("entry"))
    t0 = b.builtin("thread_position_in_threadgroup", name="t0")
    tg = b.builtin("threadgroup_position_in_grid", name="tg")
    split = lay.get("parts", 1) == 2
    if split:
        # two threadgroups a row: part = tg & 1 takes heads per p .. per p + per - 1, so the head (and its element
        # offset) is the simdgroup's plus per p; the row is tg >> 1
        part = getattr(b, "and")(tg, ir.Imm(1), name="part")
        t = b.add(t0, b.mul(part, _c(b, 32 * lay["per"], "partw"), name="part_t"), name="t")
    else:
        t = b.add(t0, b.mul(tg, _c(b, 0, "tgz"), name="tg0"), name="t")
    lane = getattr(b, "and")(t, ir.Imm(31), name="lane")
    sg = b.shr(t, _c(b, 5, "five"), name="sg")                      # the head (global under the split)
    e0 = b.shl(t, _c(b, 2, "two"), name="e0")                       # element 4 t = 128 s + 4 l
    if lay.get("rows", 1) > 1 or split:                             # row tg (tg >> 1 under the split): its QKV elements
        row = b.shr(tg, _c(b, 1, "one_r"), name="row") if split else tg
        e0 = b.add(e0, b.mul(row, _c(b, QKV, "qkvw"), name="rowe"), name="e0r")
    xs = [b.load(a, b.add(e0, _c(b, i + lay["X"] // 4, "xo%d" % i), name="xi%d" % i), type=ir.I32, name="x%d" % i)
          for i in range(4)]
    acc = None
    for i, x in enumerate(xs):
        sq = b.fmul(x, x, type=ir.F32, name="sq%d" % i)
        acc = sq if acc is None else b.fadd(acc, sq, type=ir.F32, name="acc%d" % i)
    s = TR.emit_butterfly(b, acc, TR.ROW_BUTTERFLY_MASKS, operation="sum")
    s = TR.emit_butterfly(b, s, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
    mean = b.fmul(s, _cf(b, F32(1.0 / HEAD_DIM), "inv_hd"), name="mean")
    K = O.emit_constants(b)
    r = O.emit_rn(b, "rsqrt", b.fadd(mean, _cf(b, F32(eps), "eps"), type=ir.I32, name="var"), K, "rs")
    # the gain row: q heads (s < 16) read g[0:128], k heads g[128:256]; a v head's value is its input
    gofs = b.csel(sg, _c(b, HEADS - 1, "hq"), _c(b, HEAD_DIM, "kg"), _c(b, 0, "qg"), rel="gt", name="gofs")
    gb = b.add(b.add(gofs, b.shl(lane, _c(b, 2, "two_l"), name="l4"), name="gl"), _c(b, lay["G"] // 2, "g_o"), name="gb")
    hdr, post = fn.block("store_loop"), fn.block("store_done")
    j0 = _c(b, 0, "j0")
    b.br(hdr)
    b.at(hdr)
    j = b.phi(j0, name="j")
    xv = b.fadd(b.load(a, b.add(b.add(e0, j, name="ej"), _c(b, lay["X"] // 4, "xs_o"), name="xs_i"), type=ir.I32,
                       name="xs_l"), _cf(b, F32(0.0), "xs_z"), type=ir.I32, name="xs")
    g = b.f16_to_f32(b.load(bb, b.add(gb, j, name="gj"), width="half", name="g_h"), name="g")
    y = b.fmul(b.fmul(xv, r, name="xr"), g, name="y")
    out = b.csel(sg, _c(b, HEADS + KV_HEADS - 1, "hk"), xv, y, rel="gt", name="yo")
    b.store_at(c, b.add(b.add(e0, j, name="oj"), _c(b, lay["OUT"] // 4, "o_o"), name="o_i"), out)
    jn = b.add(j, ir.Imm(1), name="j_next")
    ir.Builder.phi_latch(j, jn)
    b.br_cond(b.cmp(jn, 4, "lt", name="more"), hdr, post)
    b.at(post)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def headnorm_rows_reference(X, gq, gk, eps=EPS):
    X = np.asarray(X, F32).reshape(-1, QKV)
    return np.stack([headnorm_reference(r, gq, gk, eps) for r in X]).reshape(-1)


def headnorm_reference(x, gq, gk, eps=EPS):
    """build_headnorm's value: per lane the ordered squares of its four dims, the lane butterflies, mean, eps, the
    corrected rsqrt, (x r) g; v heads unchanged."""
    import g17decodestep as D
    from agxforge.g17 import tensorreduce as TR
    x = np.asarray(x, F32).reshape(HEADS + 2 * KV_HEADS, 32, 4)
    out = x.copy()
    for h in range(HEADS + KV_HEADS):
        sq = D.fmul(x[h], x[h])
        acc = sq[:, 0].copy()
        for i in range(1, 4):
            acc = D.fadd(acc, sq[:, i])
        lanes = TR.butterfly([float(v) for v in acc], TR.ROW_BUTTERFLY_MASKS, "sum")
        lanes = TR.butterfly(list(lanes), TR.COLUMN_BUTTERFLY_MASKS, "sum")
        r = D.rsqrt(D.fadd(D.fmul(F32(lanes[0]), F32(1.0 / HEAD_DIM)), F32(eps)))
        g = np.asarray(gq if h < HEADS else gk, np.float16).astype(F32).reshape(32, 4)
        out[h] = D.fmul(D.fmul(x[h], r), g)
    return out.reshape(-1)


# ------------------------------------------------------------------------------------------------ the batched head pad
PAD = VOCAB_PAD - VOCAB                                  # 128 pad logits a row


def pad_fill_layout(rows):
    """Binding 3 the batched head's logits fp32 [rows][VOCAB_PAD]; the pass writes the most negative finite float into
    each row's PAD words from VOCAB, so the argmax (over VOCAB_PAD) can never take a pad index (MM 25.189)."""
    return dict(op="pad_fill", rows=rows, groups=rows * PAD // 32, a_bytes=256, b_bytes=256,
                c_bytes=4 * VOCAB_PAD * rows)


def build_pad_fill(lay):
    """Thread tid = 32 t + lane: row tid >> 7, column VOCAB + (tid & 127): one store each."""
    from agxforge.g17 import cc, ir
    fn, b, a, bb, c = O._function()
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    t = b.builtin("threadgroup_position_in_grid", name="t")
    tid = b.add(b.shl(t, O._c(b, 5, "k5"), name="t32"), lane, name="tid")
    row = b.shr(tid, O._c(b, 7, "k7"), name="row")
    col = getattr(b, "and")(tid, O._c(b, PAD - 1, "k127"), name="col")
    idx = b.add(b.add(b.mul(row, O._c(b, VOCAB_PAD, "vp"), name="rv"), col, name="rvc"), O._c(b, VOCAB, "v0"), name="idx")
    b.store_at(c, idx, O._cf(b, np.finfo(F32).min, "negmax"))
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def pad_fill_reference(logits, rows):
    out = np.array(logits, F32).reshape(rows, VOCAB_PAD)
    out[:, VOCAB:] = np.finfo(F32).min
    return out.reshape(-1)


def deliver_pad_fill(root, rows_list=(8, 16)):
    """The pad fill on hardware over a 0x7f sentinel (every logit word checked: the pads written, the rest untouched)."""
    import g17deliver as DL
    import shutil
    root = Path(root)
    work = root / "work_padfill"
    index = json.loads((root / "index.json").read_text())
    for R in rows_list:
        lay = pad_fill_layout(R)
        prog = build_pad_fill(lay)
        c = bytearray(DL.SENT * lay["c_bytes"])
        name = "pad_fill_b%d" % R
        d = DL.author(work / name, prog, bytes(256), bytes(256), bytes(c), lay)
        got = DL.dispatch([dict(tag=name, dir=d, threads=32 * lay["groups"], group=32, base=1, rounds=1)], work)[name]
        want = pad_fill_reference(np.frombuffer(bytes(c), "<f4"), R).view(np.uint32)
        bad = int((np.frombuffer(got, "<u4", R * VOCAB_PAD, 0) != want).sum())
        print("pad fill rows %d on hardware: %d of %d words differ" % (R, bad, R * VOCAB_PAD))
        if bad:
            raise SystemExit("pad fill: not exact")
        dst = root / "bundles" / ("pad_fill_b%d-%s" % (R, prog_sha(prog)[:16]))
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(d, dst)
        index = [e for e in index if not (e["kind"] == "pad_fill" and e.get("variant") == {"rows": R})]
        index.append(dict(kind="pad_fill", bits=None, role="head_pad", variant={"rows": R}, cap=None,
                          bundle=str(dst.relative_to(root)), name=name, threadgroups=lay["groups"], threads_per_group=32,
                          base=1, slot_map=dict(logits_written=3), program_sha256=prog_sha(prog), sha256=prog_sha(prog),
                          code_bytes=len(prog.code), verified="hardware, every logit word over a 0x7f sentinel",
                          recipe=dict(builder="g17qwen3.build_pad_fill", layout=lay)))
    shutil.rmtree(work)
    (root / "index.json").write_text(json.dumps(index, indent=1) + "\n")


def deliver_rows(root, rows_list=(128, 512), verify_rows=16):
    """The prefill's QK-norm over R rows (one threadgroup a row) into ROOT/index.json as kind headnorm, variant
    {"rows": R}. The program is the same for every R > 1 (the row is the threadgroup id; R is only the launch), so it is
    run on hardware over a 0x7f sentinel at verify_rows rows - what the deliverer's transport holds - and every R
    entry names that one verified program with R threadgroups."""
    import g17deliver as DL
    import shutil
    root = Path(root)
    work = root / "work_headnorm_rows"
    index = json.loads((root / "index.json").read_text())
    lay = headnorm_layout(verify_rows)
    prog = build_headnorm(lay)
    for R in rows_list:
        assert build_headnorm(headnorm_layout(R)).code == prog.code, "the rows program depends on R"
    for R in (verify_rows,):
        rng = np.random.default_rng(800 + R)
        x = (rng.standard_normal(QKV * R) * rng.uniform(0.2, 6)).astype(F32)
        gq = (1 + 0.3 * rng.standard_normal(HEAD_DIM)).astype(np.float16)
        gk = (1 + 0.3 * rng.standard_normal(HEAD_DIM)).astype(np.float16)
        a, bb = bytearray(lay["a_bytes"]), bytearray(lay["b_bytes"])
        O._place(a, 0, x.astype("<f4")); O._place(bb, 0, np.concatenate([gq, gk]))
        name = "headnorm_rows%d" % R
        d = DL.author(work / name, prog, bytes(a), bytes(bb), DL.SENT * lay["c_bytes"], lay)
        got = DL.dispatch([dict(tag=name, dir=d, threads=lay["tpg"] * lay["groups"], group=lay["tpg"], base=0, rounds=1)], work)[name]
        bad = int((np.frombuffer(got, "<u4", QKV * R, 0) != headnorm_rows_reference(x, gq, gk).view(np.uint32)).sum())
        print("headnorm rows %d on hardware: %d of %d words differ" % (R, bad, QKV * R))
        if bad:
            raise SystemExit("headnorm rows: not bit-exact")
        dst = root / "bundles" / ("headnorm_rows-%s" % prog_sha(prog)[:16])
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(d, dst)
    for R in rows_list:
        index = [e for e in index if not (e["kind"] == "headnorm" and e.get("variant") == {"rows": R})]
        index.append(dict(kind="headnorm", bits=None, role="qknorm", variant={"rows": R}, cap=None,
                          bundle=str(dst.relative_to(root)), name="headnorm_rows%d" % R, threadgroups=R * lay.get("parts", 1),
                          threads_per_group=lay["tpg"], base=0, slot_map=dict(written=0, x=1, gain=2), X=0, G=0, OUT=0,
                          program_sha256=prog_sha(prog), sha256=prog_sha(prog), code_bytes=len(prog.code),
                          verified="hardware, bit-exact over a 0x7f sentinel at %d rows, the same program at every R "
                                   "(tools/g17qwen3.py deliver --rows)" % verify_rows,
                          recipe=dict(builder="g17qwen3.build_headnorm", eps=EPS, layout=headnorm_layout(R))))
    shutil.rmtree(work)
    (root / "index.json").write_text(json.dumps(index, indent=1) + "\n")


def deliver(root):
    """Build QK-norm, run it on hardware at three seeds over a 0x7f sentinel, and add it to ROOT/index.json."""
    import g17deliver as DL
    root = Path(root)
    lay = headnorm_layout()
    prog = build_headnorm(lay)
    work = root / "work_headnorm"
    jobs, wants = [], {}
    for seed in (1, 2, 3):
        rng = np.random.default_rng(700 + seed)
        x = (rng.standard_normal(QKV) * rng.uniform(0.2, 6)).astype(F32)
        gq = (1 + 0.3 * rng.standard_normal(HEAD_DIM)).astype(np.float16)
        gk = (1 + 0.3 * rng.standard_normal(HEAD_DIM)).astype(np.float16)
        a, bb = bytearray(lay["a_bytes"]), bytearray(lay["b_bytes"])
        O._place(a, lay["X"], x.astype("<f4")); O._place(bb, lay["G"], np.concatenate([gq, gk]))
        c = DL.SENT * lay["c_bytes"]
        name = "headnorm_q%d" % seed
        wants[name] = headnorm_reference(x, gq, gk).view(np.uint32)
        d = DL.author(work / name, prog, bytes(a), bytes(bb), c, lay)
        jobs.append(dict(tag=name, dir=d, threads=lay["tpg"] * lay["groups"], group=lay["tpg"], base=0, rounds=1))
    outs = DL.dispatch(jobs, work)
    bad = {n: int((np.frombuffer(outs[n], "<u4", QKV, lay["OUT"]) != w).sum()) for n, w in wants.items()}
    print("headnorm on hardware, words differing of %d: %s" % (QKV, bad))
    if any(bad.values()):
        raise SystemExit("headnorm: not bit-exact")
    import shutil
    dst = root / "bundles" / ("headnorm_qwen3-%s" % prog_sha(prog)[:16])
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(jobs[-1]["dir"], dst)
    shutil.rmtree(work)
    index = json.loads((root / "index.json").read_text())
    index = [e for e in index if e["kind"] != "headnorm"]
    index.append(dict(kind="headnorm", bits=None, role="qknorm", variant={}, cap=None, bundle=str(dst.relative_to(root)),
                      name="headnorm_qwen3", threadgroups=lay["groups"], threads_per_group=lay["tpg"], base=0,
                      slot_map=dict(written=0, x=1, gain=2), X=lay["X"], G=lay["G"], OUT=lay["OUT"],
                      program_sha256=prog_sha(prog), sha256=prog_sha(prog), code_bytes=len(prog.code),
                      verified="hardware, bit-exact over a 0x7f sentinel at 3 seeds (tools/g17qwen3.py deliver)",
                      recipe=dict(builder="g17qwen3.build_headnorm", eps=EPS, layout=lay)))
    (root / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    return dst


def prog_sha(prog):
    import hashlib
    return hashlib.sha256(prog.code).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(MODELS), default="qwen3-0.6b", help="the Qwen3 model's shapes (MODELS)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("deliver")
    d.add_argument("--out", required=True)
    d.add_argument("--rows", action="store_true", help="the prefill's many-row QK-norm (MM 25.188)")
    d.add_argument("--batch", action="store_true", help="the batched decode's QK-norm rows and head pad fill (MM 25.189)")
    p = sub.add_parser("prepare")
    p.add_argument("--mlx", required=True)
    p.add_argument("--prompt-ids", required=True)
    args = ap.parse_args(argv)
    configure(args.model)
    if args.cmd == "deliver":
        if args.batch:
            deliver_rows(args.out, rows_list=(8, 16, 128, 512))
            print(deliver_pad_fill(args.out))
        else:
            print(deliver_rows(args.out) if args.rows else deliver(args.out))
    elif args.cmd == "prepare":
        prepare(args.mlx, [int(x) for x in args.prompt_ids.split(",")])


if __name__ == "__main__":
    main()
