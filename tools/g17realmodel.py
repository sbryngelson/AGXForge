#!/usr/bin/env python3
"""THE REAL MODEL (docs/g17-tensorops-machine-model.md 25.138): InternLM2.5-1.8B-chat, greedy decoding through the
decode step of 25.132, one layer at a time, on the host reference or on the GPU pipeline.

The checkpoint has the milestone's shapes exactly: d_model 2048, 16 query heads of 128, 8 KV heads, a SwiGLU FFN of
8192, 24 layers, a 92,544-token vocabulary with an untied output projection, RoPE theta 1e6, RMSNorm eps 1e-5.

    python3 tools/g17realmodel.py prepare              (system python3 with MLX, its OWN process) the checkpoint's
                                                   bf16 tensors as fp16 .npy files in our layout, and a manifest
    python3 tools/g17realmodel.py generate --backend reference|gpu --prompt-ids 1,2,3 --tokens N
                                                   greedy tokens and per-token logits (npz), every GPU dispatch
                                                   checked by the pipeline's own rules

THE LAYOUT (what `prepare` writes; every weight K x N, fp16, the reference's orientation):
  - wqkv 2048 x 4096: q heads 0..15 | k heads 0..7 | v heads 0..7. The checkpoint packs each KV group g as
    [q head 2g, q head 2g+1, k g, v g] (mlx-lm's reshape to (.., 2 + groups, 128)). GQA is in the RoPE append: query
    head h reads KV head h // 2 and the cache is written per query head, so the attention is the milestone's.
    (The first version duplicated K and V into N 6144, repack_wqkv; bit-identical, and 8 MB more per layer.)
  - wo 2048 x 2048, wgate (w1) and wup (w3) 2048 x 8192, wdown (w2) 8192 x 2048, g1 (attention_norm), g2 (ffn_norm).
  - embed 92,544 x 2,048 (a row gather), norm 2,048, lm_head 2,048 x 98,304: the output projection padded with zero
    columns to 12 blocks of 8,192 (the N-tiled grid needs power-of-two tile columns); argmax over the first 92,544.

STORAGE is fp16 (the GPU pipeline's; tlower has no bfloat narrowing). bf16 weights convert to fp16 exactly except
below fp16's normal range. The correctness pair is therefore mlx-lm on an fp16 copy of the same checkpoint.
ROPE: mlx-lm 0.31.3's InternLM2 passes scale 2.0 (positions doubled) for any rope_scaling type but "linear"; the
checkpoint says "dynamic", which Hugging Face applies only past 32k positions. The local MLX copies therefore carry
rope_scaling {"type": "linear", "factor": 1.0}: plain RoPE at theta 1e6 on both sides.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

MODEL_ID = "internlm/internlm2_5-1_8b-chat"
OUT = ROOT / "results" / "g17-model-internlm2"
WEIGHTS = OUT / "weights"
D_MODEL, HEADS, KV_HEADS, HEAD_DIM, FFN, LAYERS, VOCAB = 2048, 16, 8, 128, 8192, 24, 92544
LM_BLOCK = 8192
VOCAB_PADDED = -(-VOCAB // LM_BLOCK) * LM_BLOCK                 # 98,304
ROPE_THETA, EPS = 1.0e6, 1.0e-5
F32 = np.float32


LOCAL = Path.home() / "models" / "internlm2_5-1_8b-chat"       # the checkpoint as downloaded (curl, resumable)


def snapshot():
    return LOCAL


def repack_wqkv_gqa(w):
    """The checkpoint's wqkv as our K x N 2048 x 4096: q heads 0..15 | k heads 0..7 | v heads 0..7 (MM 25.138.2). The
    RoPE append reads KV head h // 2 for query head h, so nothing is duplicated in the weights."""
    groups = HEADS // KV_HEADS
    w = np.asarray(w).reshape(KV_HEADS, groups + 2, HEAD_DIM, D_MODEL)
    q = w[:, :groups].reshape(HEADS * HEAD_DIM, D_MODEL)
    return np.concatenate([q, w[:, groups].reshape(-1, D_MODEL), w[:, groups + 1].reshape(-1, D_MODEL)], axis=0).T


def repack_wqkv(w):
    """The checkpoint's wqkv (4096 out x 2048 in: (16 + 2 x 8) x 128, per KV group [q 2g, q 2g+1, k g, v g]) as our K x N 2048 x 6144:
    q heads 0..15, then k and v with query head h taking KV head h // 2 (GQA folded by duplication)."""
    groups = HEADS // KV_HEADS
    w = np.asarray(w).reshape(KV_HEADS, groups + 2, HEAD_DIM, D_MODEL)
    q = w[:, :groups].reshape(HEADS, HEAD_DIM, D_MODEL)
    k = np.repeat(w[:, groups], groups, axis=0)
    v = np.repeat(w[:, groups + 1], groups, axis=0)
    return np.concatenate([q, k, v], axis=0).reshape(3 * D_MODEL, D_MODEL).T


def prepare():
    """Run with the system python3 (MLX reads bf16 safetensors), in its own process: no agxforge GPU code."""
    import mlx.core as mx
    snap = snapshot()
    tensors = {}
    for f in sorted(snap.glob("model-*.safetensors")):
        tensors.update(mx.load(str(f)))
    def get(name):
        return np.asarray(tensors[name].astype(mx.float32))
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    manifest = {}
    def save(name, arr):
        a = np.ascontiguousarray(np.asarray(arr, F32).astype(np.float16))
        if not np.all(np.isfinite(a)):
            raise SystemExit("%s does not fit fp16" % name)
        np.save(WEIGHTS / (name + ".npy"), a)
        manifest[name] = dict(shape=list(a.shape), dtype="float16", bytes=int(a.nbytes))
    for l in range(LAYERS):
        p = "model.layers.%d." % l
        save("L%d_wqkv" % l, repack_wqkv_gqa(get(p + "attention.wqkv.weight")))
        save("L%d_wo" % l, get(p + "attention.wo.weight").T)
        save("L%d_wgate" % l, get(p + "feed_forward.w1.weight").T)
        save("L%d_wup" % l, get(p + "feed_forward.w3.weight").T)
        save("L%d_wdown" % l, get(p + "feed_forward.w2.weight").T)
        save("L%d_g1" % l, get(p + "attention_norm.weight"))
        save("L%d_g2" % l, get(p + "ffn_norm.weight"))
    save("embed", get("model.tok_embeddings.weight"))
    save("norm", get("model.norm.weight"))
    lm = np.zeros((D_MODEL, VOCAB_PADDED), F32)
    lm[:, :VOCAB] = get("output.weight").T
    save("lm_head", lm)
    (OUT / "manifest.json").write_text(json.dumps(dict(model=MODEL_ID, snapshot=str(snap), layout=__doc__.split(
        "THE LAYOUT")[1].split("STORAGE")[0].strip(), tensors=manifest), indent=1) + "\n")
    print(len(manifest), "tensors,", sum(t["bytes"] for t in manifest.values()) / 1e9, "GB")


def load(name):
    """A tensor as fp32 holding its fp16 values (the reference's convention)."""
    return np.load(WEIGHTS / (name + ".npy"), mmap_mode="r").astype(F32)


def layer_spec(length):
    import g17decodestep as D
    return D.LayerSpec(d_model=D_MODEL, n_heads=HEADS, head_dim=HEAD_DIM, ffn_dim=FFN, kv_len=length,
                       storage="half", norm_eps=EPS, rope_base=ROPE_THETA, k_route=True, n_kv_heads=KV_HEADS)


def rope_tables(pos):
    """cos and sin at position `pos`, as g17decodestep.make_inputs forms them (the angle in float64, rounded once)."""
    inv_freq = ROPE_THETA ** (-np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM)
    angle = pos * inv_freq
    return np.cos(angle).astype(F32), np.sin(angle).astype(F32)


class Model:
    """Greedy decoding, one token per decode step, the prompt fed through the same steps (no prefill kernel).
    backend "reference": g17decodestep.reference per layer. backend "gpu": g17decodestep_gpu.Pipeline per layer
    (every dispatch checked by the pipeline), the final norm and the output projection included."""

    def __init__(self, backend="reference", workdir=None, kv_split=None, log=print, layers=LAYERS):
        self.backend, self.log, self.layers = backend, log, layers
        self.w = [{k: load("L%d_%s" % (l, k)) for k in ("wqkv", "wo", "wgate", "wup", "wdown", "g1", "g2")}
                  for l in range(layers)]
        self.embed = np.load(WEIGHTS / "embed.npy", mmap_mode="r")
        self.norm = load("norm")
        self.lm_head = None
        self.k = [np.zeros((HEADS, 0, HEAD_DIM), F32) for _ in range(layers)]
        self.v = [np.zeros((HEADS, 0, HEAD_DIM), F32) for _ in range(layers)]
        self.pos = 0
        self.checks = []
        if backend in ("gpu", "dry"):
            # "dry": the same pipeline with every dispatch replaced by its repository reference (no GPU): the
            # chain the chained graph (tools/g17modelgraph.py) must reproduce bit for bit
            import g17decodestep_gpu as G
            self.G = G
            self.dx = (G.DryRunDispatcher if backend == "dry" else G.Dispatcher)(Path(workdir))
            # a HOST STUB is a fallback the pipeline records as passing: count every one, and fail the run on any
            self.stubs = []
            self._stublog = lambda m: self.stubs.append(m) if "stub" in m.lower() else None
            self.ops = G.DecodeOps(Path(workdir), dry_run=backend == "dry", log=self._stublog)
            self.kv_split = kv_split

    def _layer(self, l, x):
        import g17decodestep as D
        spec = layer_spec(self.pos)
        c, s = rope_tables(self.pos)
        inputs = dict(x=x, rope_cos=c, rope_sin=s, k_cache=self.k[l], v_cache=self.v[l], **self.w[l])
        if self.backend == "reference":
            env = D.reference(spec, inputs)["env"]
        else:
            pipe = self.G.Pipeline(spec, self.dx, log=self._stublog, ops=self.ops, kv_split=self.kv_split,
                                   runtime_length=bool(self.kv_split), value_change_report=False)
            env = pipe.run(inputs)
            if self.stubs:
                raise RuntimeError("host stub at position %d layer %d: %s" % (self.pos, l, self.stubs[0]))
            self.checks.append((self.pos, l, all(e for *_x, e in pipe.checks), len(pipe.checks)))
        self.k[l], self.v[l] = env["k_all"], env["v_all"]
        return env["out"]

    def step(self, token):
        """One token in, fp32 logits (92,544) out; the caches grow by one row."""
        import g17decodestep as D
        x = self.embed[int(token)].astype(F32)
        for l in range(self.layers):
            x = self._layer(l, x)
        if self.lm_head is None:
            self.lm_head = load("lm_head")
        spec = layer_spec(self.pos)
        if self.backend != "reference":
            # the final norm and the output projection on the GPU (MM 25.138.3): the decode ops' RMSNorm, then 12
            # launches of the N-tiled grid at N 8192 (gate/up's program), each checked bit-exact by `project`
            pipe = self.G.Pipeline(spec, self.dx, log=self._stublog, ops=self.ops)
            h = self.ops.rmsnorm("final_norm", spec, x, self.norm, "half")
            if h is None:
                h = D.rmsnorm(x, self.norm, spec)
            logits = self.G.project(pipe, "lm_head", h, self.lm_head, blocks=VOCAB_PADDED // LM_BLOCK)
            self.checks.append((self.pos, "head", all(e for *_x, e in pipe.checks), len(pipe.checks)))
        else:
            h = D.rmsnorm(x, self.norm, spec)
            logits = np.zeros(VOCAB_PADDED, F32)
            for b in range(0, VOCAB_PADDED, LM_BLOCK):
                G = D.projection_route(LM_BLOCK, D_MODEL)[1]
                logits[b:b + LM_BLOCK] = D.gemm_reference(h[None, :], self.lm_head[:, b:b + LM_BLOCK], split_k=G)[0]
        self.pos += 1
        return np.asarray(logits, F32).ravel()[:VOCAB]

    def generate(self, prompt, n):
        tokens, logits = [], []
        nxt = None
        for i, t in enumerate(list(prompt)):
            lg = self.step(t)
            nxt = int(np.argmax(lg))
        for _ in range(n):
            tokens.append(nxt)
            t0 = time.time()
            lg = self.step(nxt)
            logits.append(lg)
            self.log("token %d: %d (%.1f s)" % (len(tokens), nxt, time.time() - t0))
            nxt = int(np.argmax(lg))
        return tokens, np.stack(logits) if logits else np.zeros((0, VOCAB), F32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("prepare", "generate"))
    ap.add_argument("--backend", choices=("reference", "gpu", "dry"), default="reference")
    ap.add_argument("--prompt-ids", default="1")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--layers", type=int, default=LAYERS, help="first N layers only (a smoke test)")
    ap.add_argument("--kv-split", type=int, choices=(2, 4, 8), default=None)
    ap.add_argument("--workdir", type=Path, default=OUT / "work")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        return prepare()
    m = Model(args.backend, args.workdir, args.kv_split, log=lambda s: print(s, file=sys.stderr, flush=True),
              layers=args.layers)
    prompt = [int(t) for t in args.prompt_ids.split(",")]
    tokens, logits = m.generate(prompt, args.tokens)
    ops = m.ops.checks if args.backend != "reference" else []
    rep = dict(model=MODEL_ID, backend=args.backend, prompt=prompt, tokens=tokens, layers=args.layers,
               kv_split=args.kv_split, dispatches=m.dx.count if args.backend != "reference" else 0,
               checks_pass=all(c[2] for c in m.checks) and all(c[-1] for c in ops),
               pipeline_checks=sum(c[3] for c in m.checks), decodeop_checks=len(ops),
               failing=[c for c in m.checks if not c[2]] + [c for c in ops if not c[-1]])
    print(json.dumps(rep))
    if args.out:
        np.savez(args.out, tokens=np.array(tokens), logits=logits, prompt=np.array(prompt))
        args.out.with_suffix(".json").write_text(json.dumps(rep, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
