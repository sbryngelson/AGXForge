#!/usr/bin/env python3
"""mlx-lm on the SAME checkpoint as tools/g17realmodel.py (MM 25.138): greedy tokens and logits, and the tokenizer.
Run with the system python3 (MLX), in its OWN process; it imports no agxforge GPU code.

    python3 tools/g17model_mlx.py tokens --prompt "..." [--dtype float16|bfloat16] --tokens N --out F.npz
    python3 tools/g17model_mlx.py encode --prompt "..."          the prompt's token ids (comma-separated)
    python3 tools/g17model_mlx.py decode --ids 1,2,3              text
    python3 tools/g17model_mlx.py bench --bits 8 --ids 1,2,3 --tokens N   mlx-lm's decode tok/s, fixed length

`tokens` feeds the prompt one token at a time through the KV cache, as tools/g17realmodel.py does (no prefill pass),
then decodes greedily, saving each step's fp32 logits.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

LOCAL = Path.home() / "models" / "internlm2_5-1_8b-chat"

REQUIRED_ROPE_SCALING = {"type": "linear", "factor": 1.0}


def _check_rope_scaling(mlx_path):
    """mlx-lm 0.31.3's InternLM2 passes scale 2.0 (positions doubled) for any rope_scaling type but "linear"
    (tools/g17realmodel.py's ROPE note). The checkpoint's raw HF config says {"type": "dynamic", "factor": 2.0},
    which only Hugging Face's own implementation treats as a no-op below 32k positions; mlx-lm does not. A fresh
    `mlx_lm.convert` carries the raw value through, so every checkpoint under ~/models needs this patched by hand
    after conversion - on 2026-09-26 a re-convert silently skipped it and mlx-lm's own top-1 token changed while
    nothing about our graph did, which took real time to isolate. Fail loud here instead of a wrong-but-plausible
    token."""
    cfg_path = Path(mlx_path) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    got = cfg.get("rope_scaling")
    if got != REQUIRED_ROPE_SCALING:
        raise SystemExit(
            "g17model_mlx: %s has rope_scaling=%r, required %r for mlx-lm 0.31.3's InternLM2 to match our graph's "
            "plain-RoPE assumption. Patch it: python3 -c \"import json,pathlib; p=pathlib.Path(%r); "
            "c=json.loads(p.read_text()); c['rope_scaling']=%r; p.write_text(json.dumps(c, indent=2))\""
            % (cfg_path, got, REQUIRED_ROPE_SCALING, str(cfg_path), REQUIRED_ROPE_SCALING))


def load(dtype):
    import mlx.core as mx
    from mlx_lm import load as mload
    _check_rope_scaling(LOCAL)
    model, tok = mload(str(LOCAL), tokenizer_config={"trust_remote_code": True})
    model.set_dtype(getattr(mx, dtype))
    return model, tok


def tokens(prompt_ids, n, dtype):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    model, _tok = load(dtype)
    cache = make_prompt_cache(model)
    logits = None
    for t in prompt_ids:
        logits = model(mx.array([[t]]), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(logits)
    out, lg = [], []
    for _ in range(n):
        nxt = int(mx.argmax(logits).item())
        out.append(nxt)
        logits = model(mx.array([[nxt]]), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(logits)
        lg.append(np.array(logits))
    return out, np.stack(lg)


def bench(bits, prompt_ids, n, batch=1):
    """mlx-lm's own quantized checkpoint decoding exactly n greedy tokens after the prompt (prefilled in one call), with
    no end-of-sequence stop, async eval as mlx-lm's generate does; after a warm-up. Returns (tok/s, first 16 tokens).
    batch B > 1 (MM 25.144.3): B sequences, sequence b's prompt the given one rotated by b (the batched graph's prompts),
    prefilled together as a (B, L) call and decoded together; the rate is the AGGREGATE, B n tokens over the wall, and
    the first tokens returned are sequence 0's."""
    import time
    import mlx.core as mx
    from mlx_lm.utils import load_model
    from mlx_lm.models.cache import make_prompt_cache
    q_path = Path.home() / "models" / ("internlm2_5-1_8b-chat-mlx-q%d" % bits)
    _check_rope_scaling(q_path)
    model, _ = load_model(q_path)
    prompt = mx.array([prompt_ids[b % len(prompt_ids):] + prompt_ids[:b % len(prompt_ids)] for b in range(batch)])

    def run(k, keep=0):
        cache = make_prompt_cache(model)
        t_p = time.perf_counter()
        y = mx.argmax(model(prompt, cache=cache)[:, -1], axis=-1)
        mx.eval(y)
        run.ttft = time.perf_counter() - t_p                 # the whole prompt in one call, to the first token
        toks = [int(y[0].item())] if keep else []
        t0 = time.perf_counter()
        for _ in range(k):
            y = mx.argmax(model(y[:, None], cache=cache)[:, -1], axis=-1)
            if len(toks) < keep:
                toks.append(int(y[0].item()))
            else:
                mx.async_eval(y)
        mx.eval(y)
        return batch * k / (time.perf_counter() - t0), toks

    _, first = run(16, keep=16)
    tps, _ = run(n)
    return tps, first, run.ttft


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tokens", "encode", "decode", "bench"))
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--ids", default="")
    ap.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--batch", type=int, default=1, help="bench: B sequences, prompts rotated by b; aggregate tok/s")
    a = ap.parse_args(argv)
    if a.cmd == "bench":
        tps, first, ttft = bench(a.bits, [int(i) for i in a.ids.split(",")], a.tokens, a.batch)
        print(json.dumps(dict(bits=a.bits, prompt_len=len(a.ids.split(",")), tokens=a.tokens, tokens_per_s=tps,
                              ttft_s=ttft, first16=first, batch=a.batch)))
        return 0
    if a.cmd in ("encode", "decode"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(LOCAL), trust_remote_code=True)
        if a.cmd == "encode":
            print(",".join(str(i) for i in tok.encode(a.prompt)))
        else:
            print(tok.decode([int(i) for i in a.ids.split(",") if i]))
        return 0
    ids = [int(i) for i in a.ids.split(",")] if a.ids else None
    if ids is None:
        from transformers import AutoTokenizer
        ids = AutoTokenizer.from_pretrained(str(LOCAL), trust_remote_code=True).encode(a.prompt)
    out, lg = tokens(ids, a.tokens, a.dtype)
    rep = dict(prompt_ids=ids, tokens=out, dtype=a.dtype)
    print(json.dumps(rep))
    if a.out:
        np.savez(a.out, tokens=np.array(out), logits=lg, prompt=np.array(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
