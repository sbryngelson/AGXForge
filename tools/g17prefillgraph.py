#!/usr/bin/env python3
"""THE PREFILL SECTION OF THE GRAPH (MM 25.142.10): the whole prompt as M-row passes, run once before decode.

tools/g17q4graph.py calls `section(...)` when the model config has {"prefill": {"M": 128|512|1024}}. It returns the arenas,
initial files and the `prefill` steps decodegen runs (host writes, then one serial command buffer per step), so that
after the last step the KV cache holds positions 0..L-1, the token log holds the first generated token at L, region R
holds that token's embedding, and the q0 word is L: decode continues exactly where token-by-token feeding would.

Per chunk of M rows (positions p0..p0+M-1), per layer, kernels from the deliver index (Piece B's milestones, 25.144):
    norm_rows(attn_norm, fp16 out) -> qmm_dequant(qkv) + qmm(qkv) -> prefill_append -> prefill_attn(out16)
    -> qmm_dequant(wo) + qmm(wo) -> residual_rows(add16) -> norm_rows(ffn_norm, fp16 out)
    -> qmm(w1), qmm(w3) -> swiglu_rows -> qmm(w2, split-K) -> fold_rows = fp16((p0 + p1) + h)
Then the tail, once, on the last row: a host copy of x[L-1] into R's input row, q0 = L - 1, and decode's own
final_norm, head, argmax and gen_step dispatches (which write log[L] and advance q0 to L).

Every kernel kind this needs must be in the index, or `section` refuses naming the missing (kind, role, variant). Region
placement uses tools/g17modelgraph.Solver in a separate arena "PF"; attention binds decode's own region3 (grown to the
prefill kernels' prefill_region3_bytes) and GEN region.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import g17modelgraph as MG  # noqa: E402

PF = "PF"
PE = "PE"              # the prompt-embedding arena; a second section in one graph (MM 25.203's verify step) renames both


class Missing(KeyError):
    pass


def need(pick, kind, **match):
    try:
        return pick(kind, **match)
    except KeyError as e:
        raise Missing("prefill needs %s %s in the deliver index (%s)" % (kind, match, e)) from None


REGO_OPT = dict(fold=True, scale=True, hoist=True, row2=True, holdk=True)   # g17deliver._rego_options(), all of them


def _rego_attn(pick, cap, M, p0_block, hw_exp2=False, bk=16, sreg=False):
    """M2's register-O attention with the causal skip (MM 25.144.2): the code-size variant when the index has it (the
    same values in fewer instructions a trip), else the plain one. hw_exp2: the hardware-exp2 row stage (enclosure-
    checked, token-gated; built with the code-size options)."""
    base = dict({"mma": True, "M": M, "p0_block": p0_block, "out16": True, "skip": True, "rego": True, "sg": 1},
                **({"hw_exp2": True} if hw_exp2 else {}))
    if sreg:                                             # the register-S route (MM 25.163)
        return need(pick, "prefill_mma", role="attn", cap=cap, variant=dict(base, fold=True, holdk=True, sreg=True))
    if bk != 16:                                         # 32-key blocks (MM 25.162): built only with the code-size options
        return need(pick, "prefill_mma", role="attn", cap=cap, variant=dict(base, **REGO_OPT, bk=bk))
    try:
        return pick("prefill_mma", role="attn", cap=cap, variant=dict(base, **REGO_OPT))
    except KeyError:
        return need(pick, "prefill_mma", role="attn", cap=cap, variant=base)


def kernels(pick, bits, cap, M, seed=False, route="scalar", chunks=1, skip=False, rego=False, hw_exp2=False, gemm_M=None,
            swiglu_fused=False, bk=16, sreg=False, ffn16=False, swiglu_fast=False, qsm=None, attn_opts=None, qmvw=None):
    """Resolve every prefill kernel for (bits, cap, M) from the index, or raise Missing naming the first absent one.
    gemm_M: the row count of the norms, projections and row kernels, run as M / gemm_M slices of each M-row chunk; the
    append and the attention keep M (default: gemm_M = M)."""
    k = {}
    G = gemm_M or M
    for role, in_dtype in (() if qmvw else (("attn_norm", "half"), ("ffn_norm", "float"))):
        k[role] = need(pick, "norm", role=role, variant={"out32": False, "seed": seed, "batch": G})
    if route == "mma" and rego:                          # register-O: pairs ONLY with the register-O append (REGO_NOTES)
        k["append"] = need(pick, "prefill_mma", role="append", cap=cap, variant={"mma": True, "M": M, "rego": True})
        k["attn_chunks"] = [_rego_attn(pick, cap, M, c * M // 16, hw_exp2, bk, sreg) for c in range(chunks)]
        k["attn"] = k["attn_chunks"][0]
    elif route == "mma":                                 # M2's tensor-unit route: one attention bundle per chunk start
        k["append"] = need(pick, "prefill_mma", role="append", cap=cap, variant={"mma": True, "M": M})
        k["attn_chunks"] = [need(pick, "prefill_mma", role="attn", cap=cap,
                                 variant=dict({"mma": True, "M": M, "p0_block": c * M // 16, "out16": True},
                                              **({"skip": True} if skip else {}))) for c in range(chunks)]
        k["attn"] = k["attn_chunks"][0]
    else:
        k["append"] = need(pick, "prefill_attn", role="append", cap=cap, variant={"scalar": True})
        # attn_opts (MM 25.205): decode's kvvec loads and hardware exp2 in the scalar attention, {"kvvec": True, ...}
        k["attn"] = need(pick, "prefill_attn", role="attn", cap=cap,
                         variant=dict({"scalar": True} if qmvw else {"scalar": True, "out16": True},
                                      **{o: True for o in (attn_opts or ())}))
    if qmvw:
        # THE SMALL-k VERIFY ROUTE (MM 25.207): G = qmvw rows through g17qmvw (each weight dequantized once, every row a
        # plain fp32 dot), reading fp32 rows: the norms out32, the attention fp32 (attn32), the SwiGLU rows' act32; the
        # residuals are the split-1 psum passes (h = y + x16; x16 = fp16(y + h)). The blocks are the prefill's own, at
        # g17qmvw's default W / S / B (the qmv layout); w1 and w3 share one standalone-block bundle.
        if G != qmvw or bits != 4 or route != "scalar":
            raise ValueError("prefill qmvw route: %d-row slices of q4 weights on the scalar attention" % qmvw)
        for role, in_dtype in (("attn_norm", "half"), ("ffn_norm", "float")):
            k[role] = need(pick, "norm", role=role, variant={"out32": True, "seed": False, "batch": G})
        for proj in ("qkv", "wo", "w1", "w2"):
            q = k["qmvw_" + proj] = need(pick, "qmvw", bits=4, role=proj if proj != "w1" else "w1_block", variant={"nb": G})
            k["dq_" + proj] = dict(S=q["S"], B=q["B"], slot_extents_bytes=dict(slot1=256, slot2=q["recipe"]["layout"]["a_bytes"],
                                                                                 slot3=0))
        k["swiglu"] = need(pick, "swiglu_rows", variant={"rows": G, "out32": True})
        k["ps_wo"] = need(pick, "psum", role="wo", variant={"mode": "add16", "rows": G, "sk": 1})
        k["ps_w2"] = need(pick, "psum", role="w2", variant={"mode": "fold16", "rows": G, "sk": 1})
        return k
    for proj in ("qkv", "wo", "w1", "w2"):
        if qsm is None:
            k["dq_" + proj] = need(pick, "qmm_dequant", bits=bits, role=proj, variant={})
            k["mm_" + proj] = need(pick, "qmm", bits=bits, role=proj, variant={"M": G})
    if qsm is not None:
        # THE q4 ROUTE FOR SHORT PROMPTS (MM 25.202): 16-row slices through g17qsm, the batched decode's projection (q4
        # weights read once, dequantized on the fp16 pipe, 25.196) and its psum passes, in place of the W16 GEMMs and the
        # row kernels. qsm = {role: split-K}. The weights are the prefill's own blocks (the qmm_dequant layout is
        # g17qsm's default W / S / B), so w1 and w3 run one standalone-block bundle.
        if G != 16 or bits != 4:
            raise ValueError("prefill qsm route: 16-row slices of q4 weights")
        for proj in ("qkv", "wo", "w1", "w2"):
            k["qsm_" + proj] = need(pick, "qsm", bits=4, role=proj if proj != "w1" else "w1_block",
                                    variant={"sk": qsm[proj], "xrows": True, "h16": True})
        for key, role, mode in (("ps_qkv", "qkv", "sum"), ("ps_wo", "wo", "add16"), ("ps_w1", "w1", "swiglu"),
                                ("ps_w2", "w2", "fold16")):
            k[key] = need(pick, "psum", role=role, variant={"mode": mode, "rows": 16, "sk": qsm[role]})
        # THE BLOCK GEOMETRY. The qsm route never dispatches qmm_dequant; it packs each projection's block in that layout,
        # which is g17qsm's default W / S / B. A root with the qmm_dequant entries (InternLM2, Qwen3-0.6B) keeps them, and
        # they must agree with the qsm entry; one without them (Qwen3-8B: qkv's 6,144 rows are not the power of two the
        # dequant program needs) takes the geometry from the qsm entry: no W16 (the route never binds it), and DUMMY
        # (the psum sum pass's unbound slot 2) at least every pass's b_bytes.
        for proj in ("qkv", "wo", "w1", "w2"):
            q = k["qsm_" + proj]
            try:
                dq = pick("qmm_dequant", bits=bits, role=proj, variant={})
            except KeyError:
                dq = None
            if dq is not None and "S" in dq and "S" in q:          # real index entries carry the geometry; check it
                qa = q.get("recipe", {}).get("layout", {}).get("a_bytes")
                want = (dq["S"], dq["B"]) + ((dq["slot_extents_bytes"]["slot2"],) if qa is not None else ())
                got = (q["S"], q["B"]) + ((qa,) if qa is not None else ())
                if want != got:
                    raise ValueError("prefill qsm route: %s's qmm_dequant block %s is not its qsm block's %s" % (proj, want, got))
                k["dq_" + proj] = dq
            elif dq is not None:
                k["dq_" + proj] = dq
            else:
                ps_b = max(k[key]["recipe"]["layout"]["b_bytes"] for key in ("ps_qkv", "ps_wo", "ps_w1", "ps_w2"))
                k["dq_" + proj] = dict(S=q["S"], B=q["B"], slot_extents_bytes=dict(
                    slot1=ps_b, slot2=q["recipe"]["layout"]["a_bytes"], slot3=0))
        return k
    k["residual"] = need(pick, "residual_rows", variant={"rows": G})    # h fp32 at 0 and fp16(h) at H16
    if ffn16:
        # MM 25.183: w1 and w3 store fp16 (the receipted K-loop 'half' epilogue) and the SwiGLU reads fp16
        if swiglu_fused:
            raise ValueError("prefill: ffn16 and swiglu_fused are two routes for the same SwiGLU; pick one")
        k["mm_w1"] = need(pick, "qmm", bits=bits, role="w1", variant={"M": G, "half": True})
        # swiglu_fast: op1272 and the raw reciprocal, within 1 fp16 ulp of float64 (an enclosure, not bit-exact)
        k["swiglu"] = need(pick, "swiglu_rows", variant=dict({"rows": G, "in16": True}, **({"fast": True} if swiglu_fast else {})))
    elif swiglu_fused:
        # MM 25.144.12: w3's qmm with the SwiGLU in its tail, in place of qmm(w3) + swiglu_rows (the same values)
        k["mm_w3sw"] = need(pick, "qmm_swiglu", bits=bits, role="w3", variant={"M": G, "swiglu": True})
    else:
        k["swiglu"] = need(pick, "swiglu_rows", variant={"rows": G})
    k["fold"] = need(pick, "fold_rows", variant={"rows": G})            # fp16((p0 + p1) + h), exactly that order
    return k


def tail_step(fin, r_x16, final_norm_x, x16_at, gen0, L, M, chunks, d, logits=None):
    """The prefill's last step: the LAST prompt row only (x[L - 1], row (L - 1) - (chunks - 1) M of the last chunk)
    copied into decode's final-norm input row, q0 = L - 1, and decode's own final norm, lm_head, argmax pass 1 and gen
    step (`fin`) run once. With `logits`, the head's output is first filled with the 0x7f sentinel from arena "PS", so an
    unwritten logit cannot pass the tail check (MM 25.144.7)."""
    last = (L - 1) - (chunks - 1) * M
    writes = [dict(arena=r_x16["arena"], offset=r_x16["offset"] + final_norm_x, bytes=d * 2,
                   copy_from=dict(arena=PF, offset=x16_at + last * d * 2)),
              dict(arena=gen0["arena"], offset=gen0["offset"], u32=L - 1)]
    if logits:
        writes.append(dict(arena=logits["arena"], offset=logits["offset"], bytes=int(logits["bytes"]),
                           copy_from=dict(arena="PS", offset=0)))
    return dict(writes=writes, dispatches=fin)


def region3_bytes(pick, cap, cfg=None):
    """The attention region decode must reserve when prefill is on: the prefill kernels write q16 and their attention
    output past decode's region3 (Q16, PATTN), up to prefill_region3_bytes (the MMA route: mma_region3_bytes, which
    adds its Q tiles, block-major V and scratch)."""
    cfg = cfg or {}
    if cfg.get("attn") == "mma":
        M = int(cfg["M"])
        variant = dict({"mma": True, "M": M}, **({"rego": True} if cfg.get("rego") else {}))
        return need(pick, "prefill_mma", role="append", cap=cap, variant=variant)["mma_region3_bytes"]
    variant = dict({"scalar": True} if cfg.get("qmvw") else {"scalar": True, "out16": True},
                   **{o: True for o in (cfg.get("attn_opts") or ())})
    return need(pick, "prefill_attn", role="attn", cap=cap, variant=variant)["prefill_region3_bytes"]


def _al(v, a=256):
    return (v + a - 1) // a * a


def _block(npz, rows=None, pad_n=None, pad_k=None):
    """One projection's packed block in the qmm_dequant layout: W u32 [N][K/per] at 0, then S, then B (bf16 [N][K/64]).
    pad_n / pad_k (MM 25.188): zero rows / zero K columns up to that size - zero words, scales and biases dequantize to
    exact zeros, so a padded FFN (w1 / w3 rows, w2's K) gives the same outputs bit for bit."""
    import numpy as np
    W, S, B = (np.asarray(npz[k]) for k in "WSB")
    if rows is not None:
        W, S, B = W[rows], S[rows], B[rows]
    if pad_n is not None and pad_n > W.shape[0]:
        extra = pad_n - W.shape[0]
        W, S, B = (np.concatenate([x, np.zeros((extra, x.shape[1]), x.dtype)]) for x in (W, S, B))
    if pad_k is not None:
        per = pad_k // 64                              # groups a row; words a row = pad_k / (32 / bits)
        wpr = W.shape[1] * pad_k // (S.shape[1] * 64)
        if wpr > W.shape[1]:
            W = np.concatenate([W, np.zeros((W.shape[0], wpr - W.shape[1]), W.dtype)], 1)
            S, B = (np.concatenate([x, np.zeros((x.shape[0], per - x.shape[1]), x.dtype)], 1) for x in (S, B))
    return (np.ascontiguousarray(W, dtype="<u4").tobytes(), np.ascontiguousarray(S, dtype="<u2").tobytes(),
            np.ascontiguousarray(B, dtype="<u2").tobytes())


def section(pick, bits, cap, cfg, prompt, decode, weights_dir, out_dir, embed, final_norm_x, logits=None, dims=None):
    """The prefill steps for `prompt`. decode: the decode graph's dispatch dicts by name (their binds give each layer's
    attention region3 and GEN region, the norm gains, and the tail); final_norm_x: the byte offset of the input row inside
    decode's final-norm slot-1 binding. pick(kind, **match) returns an index entry with "_root" (its deliver root).
    logits: the head's output {arena, offset, bytes} (the graph's logits_readback). When given, the tail step first
    fills it with a 0x7f sentinel (from arena "PS") and the dump carries it, so the tail's logits are checked, not
    only its token (MM 25.144.7).
    dims (MM 25.188): the model's widths beside d_model - qkv (the fused projection), hd (the attention output, wo's
    K), ffn - and qknorm (Qwen3's per-head RMSNorm of q and k, one pass over the chunk's rows before the append).
    Absent, InternLM2's (4,096, 2,048, 8,192, no QK-norm), so its graphs are unchanged.
    Returns dict(arenas, files, zero_init, prefill)."""
    import numpy as np
    dims = dict(dict(qkv=4096, hd=2048, ffn=8192, qknorm=False), **(dims or {}))
    QW, HD, F = dims["qkv"], dims["hd"], dims["ffn"]
    M = int(cfg["M"])
    L = len(prompt)
    if L > cap:
        raise ValueError("prefill: prompt %d exceeds the cache %d" % (L, cap))
    route = cfg.get("attn", "scalar")
    chunks = -(-L // M)
    # "gemm_M": the norms, projections and row kernels run as S = M / gemm_M slices of each M-row chunk, and only the
    # append and the attention see all M rows at once. Every row gets the same kernels, so the values are unchanged;
    # what changes is that the attention runs fewer, larger dispatches (MM 25.142.11)
    gm = int(cfg.get("gemm_M", M))
    gw = 2 if cfg.get("ffn16") else 4                   # the w1 / w3 outputs: fp16 under ffn16 (MM 25.183), else fp32
    if M % gm or gm > M:
        raise ValueError("prefill: gemm_M %d must divide M %d" % (gm, M))
    S = M // gm
    k = kernels(pick, bits, cap, M, seed=bool(cfg.get("seed")), route=route, chunks=chunks, skip=bool(cfg.get("skip")),
                rego=bool(cfg.get("rego")), hw_exp2=bool(cfg.get("hw_exp2")), gemm_M=gm,
                swiglu_fused=bool(cfg.get("swiglu_fused")), bk=int(cfg.get("bk", 16)), sreg=bool(cfg.get("sreg")),
                ffn16=bool(cfg.get("ffn16")), swiglu_fast=bool(cfg.get("swiglu_fast")), qsm=cfg.get("qsm"),
                attn_opts=cfg.get("attn_opts"), qmvw=cfg.get("qmvw"))
    qsmr = cfg.get("qsm") is not None
    qmvwr = bool(cfg.get("qmvw"))
    if dims["qknorm"]:
        k["qknorm"] = need(pick, "headnorm", role="qknorm", variant={"rows": M})
    W = Path(weights_dir)
    out_dir = Path(out_dir)
    nl = sum(1 for n in decode if n.endswith(".attn_fused"))
    # the repacked projection weights, one arena per layer: qkv, wo, w1, w3, w2 blocks in the qmm_dequant layout
    arenas, files, blocks = {}, {}, {}
    for l in range(nl):
        buf, offs = bytearray(), {}
        ffn = np.load(W / ("L%d_ffn.npz" % l))
        Fr = ffn["W"].shape[0] // 2                        # the checkpoint's FFN; F may be it padded (MM 25.188)
        pn, pk = (F, F) if F > Fr else (None, None)
        parts = dict(qkv=_block(np.load(W / ("L%d_qkv.npz" % l))), wo=_block(np.load(W / ("L%d_wo.npz" % l))),
                     w1=_block(ffn, slice(0, Fr), pad_n=pn), w3=_block(ffn, slice(Fr, 2 * Fr), pad_n=pn),
                     w2=_block(np.load(W / ("L%d_w2.npz" % l)), pad_k=pk))
        for proj, (wb, sb, bb) in parts.items():
            e = k["dq_" + ("w1" if proj == "w3" else proj)]
            if (len(wb), len(sb)) != (e["S"], e["B"] - e["S"]):
                raise ValueError("L%d %s: packed sizes %d/%d do not match the dequant layout S %d B %d" %
                                 (l, proj, len(wb), len(sb), e["S"], e["B"]))
            offs[proj] = len(buf)
            blk = bytearray(e["slot_extents_bytes"]["slot2"])
            blk[0:len(wb)] = wb
            blk[e["S"]:e["S"] + len(sb)] = sb
            blk[e["B"]:e["B"] + len(bb)] = bb
            buf += blk
            buf += bytes(_al(len(buf)) - len(buf))
        name = "PW%d" % l
        f = out_dir / (name + ".bin")
        f.write_bytes(bytes(buf))
        arenas[name], files[name], blocks[l] = len(buf), str(f), offs
    # the prompt's embedding rows, padded with zeros to whole chunks
    pe = np.zeros((chunks * M, embed.shape[1]), np.float16)
    pe[:L] = np.asarray(embed[list(prompt)], np.float16)
    f = out_dir / (PE + ".bin")
    f.write_bytes(pe.tobytes())
    arenas[PE], files[PE] = pe.nbytes, str(f)
    if logits:
        f = out_dir / "PS.bin"                               # the logits sentinel the tail step copies in first
        f.write_bytes(b"\x7f" * int(logits["bytes"]))
        arenas["PS"], files["PS"] = int(logits["bytes"]), str(f)
    # the prefill working set in arena PF (one chunk of M rows; reused by every layer and chunk). Every buffer is
    # preceded by a gap, so a binding's base (buffer start minus the kernel's field offset) is never negative. X16 and
    # QKV hold the whole chunk (the append reads all M rows of QKV); the rest serve one gm-row slice at a time.
    d = embed.shape[1]
    sizes = dict(X16=M * d * 2, N1=gm * d * 2, QKV=M * QW * 4, **({"QKN": M * QW * 4} if dims["qknorm"] else {}), W16=max(e["slot_extents_bytes"]["slot3"] for key, e in k.items() if key.startswith("dq_")),
                 Y=gm * d * 4, H=(k["residual"]["H16"] if "residual" in k else 0) + gm * d * 2, N2=gm * d * 2, G=gm * F * 4, U=gm * F * 4, ACT=gm * F * 2,
                 P=2 * gm * d * 4, DUMMY=max(e["slot_extents_bytes"]["slot1"] for key, e in k.items() if key.startswith("dq_")))
    if qsmr:
        # the qsm partials (the widest split-K projection's) and w3's own, beside w1's for the fused SwiGLU pass
        sizes["PQ"] = max(e["recipe"]["layout"]["c_bytes"] for key, e in k.items() if key.startswith("qsm_"))
        sizes["PQ3"] = k["qsm_w1"]["recipe"]["layout"]["c_bytes"]
        sizes["H"] = gm * d * 4
    if qmvwr:
        # the qmvw route's rows are fp32 (its x is fp32 [rows][K])
        sizes.update(N1=gm * d * 4, N2=gm * d * 4, H=gm * d * 4, ACT=gm * F * 4, P=gm * d * 4)
    fused = k.get("mm_w3sw")
    if fused:
        # the fused w3 binds ONE region at slot 3: U (its scratch) at U_OFF, w1's C (G) at G_OFF, act at ACT_OFF
        for n in ("G", "ACT"):
            sizes.pop(n)
        sizes["U"] = fused["regions"]["slot3"]["ACT"]["offset"] + gm * F * 2
    at, cur = {}, MG.GAP
    for n, sz in sizes.items():
        at[n] = _al(cur)
        cur = at[n] + sz + MG.GAP
    if fused:
        at["G"] = at["U"] + fused["G_OFF"]
        at["ACT"] = at["U"] + fused["ACT_OFF"]
    arenas[PF] = _al(cur)

    def pf(off, nbytes):
        assert off >= 0, off
        return dict(arena=PF, offset=off, bytes=nbytes)

    def disp(name, e, binds, threads=None):
        tpg = e.get("threads_per_group", 32)
        return dict(name=name, bundle=str(Path(e["_root"]) / e["bundle"]), threads=threads or e["threadgroups"] * tpg,
                    group=tpg, binds={str(s): b for s, b in binds.items()})

    nm, ffm = k["attn_norm"], k["ffn_norm"]
    # PERSISTENT W16 (cfg "persistent_w16"): every projection dequantised ONCE, in a setup step outside the timed
    # prefill, into per-layer fp16 arenas W16L<l>; the GEMMs then read those, and no dequant runs per prefill.
    persist = bool(cfg.get("persistent_w16"))
    w16at, setup = {}, []
    if persist:
        for l in range(nl):
            off = 0
            for proj in ("qkv", "wo", "w1", "w3", "w2"):
                e = k["dq_" + ("w1" if proj == "w3" else proj)]
                w16at[(l, proj)] = off
                off = _al(off + e["slot_extents_bytes"]["slot3"])
            arenas["W16L%d" % l] = off
    def make_layers(aw, collect, rows=M):
      # THE REMAINDER CHUNK (MM 25.169): a prompt that is not whole chunks leaves the last chunk `rows` real rows. Only
      # the gm-row slices that hold one run their norms, projections and row kernels; the attention and append keep
      # the whole chunk (the rows past the prompt are never read: causal rows see keys up to their own, and decode
      # rewrites cache rows from L on). The last x row the tail reads is inside the last real slice.
      live = -(-rows // gm)
      layers = []
      for l in range(nl):
          att = decode["L%d.attn_fused" % l]["binds"]
          r3, gen = att["0"], att["2"]
          g1 = decode["L%d.attn_norm" % l]["binds"]["2"]
          g2 = decode["L%d.ffn_norm" % l]["binds"]["2"]
          pw = "PW%d" % l

          def w16(proj, nbytes):
              if persist:
                  return dict(arena="W16L%d" % l, offset=w16at[(l, proj)], bytes=nbytes)
              return pf(at["W16"], nbytes)

          def dq(proj, e, first=True):
              dd = disp("P%d.dq_%s" % (l, proj), e, {1: pf(at["DUMMY"], e["slot_extents_bytes"]["slot1"]),
                                                   2: dict(arena=pw, offset=blocks[l][proj], bytes=e["slot_extents_bytes"]["slot2"]),
                                                   3: w16(proj, e["slot_extents_bytes"]["slot3"])})
              if persist:
                  if collect and first:              # the one-time dequant, collected once (not per chunk or slice)
                      setup.append(dd)
                  return None
              return dd                              # without persist_w16 every slice re-dequantises (W16 is shared)

          def mm(proj, e, a, c):
              return disp("P%d.mm_%s" % (l, proj), e, {1: a, 2: w16(proj, e["slot_extents_bytes"]["slot2"]), 3: c})

          def xs(i):                                 # slice i of the chunk's x rows
              return at["X16"] + i * gm * d * 2
          ap = k["append"]
          def qsm_op(proj, e, x, xbytes, parts):
              lay_ = e["recipe"]["layout"]
              blk = "w1" if proj == "w3" else proj
              return disp("P%d.qsm_%s" % (l, proj), e, {1: dict(arena=pw, offset=blocks[l][proj], bytes=lay_["a_bytes"]),
                                                       2: dict(x, bytes=max(xbytes, lay_["b_bytes"])),
                                                       3: pf(at[parts], lay_["c_bytes"])})

          def psum_op(name, e, parts, second, out):
              lay_ = e["recipe"]["layout"]
              b2 = (pf(at[second], lay_["b_bytes"]) if second == "PQ3" else
                    None if second is None else dict(second, offset=second["offset"] - lay_["IN"], bytes=lay_["b_bytes"]))
              binds = {1: pf(at[parts], lay_["a_bytes"]), 3: dict(out, offset=out["offset"] - lay_["OUT"], bytes=lay_["c_bytes"])}
              binds[2] = b2 if b2 is not None else pf(at["DUMMY"], lay_["b_bytes"])     # sum: no residual, slot 2 bound
              return disp("P%d.%s" % (l, name), e, binds)
          def qmvw_op(proj, e, x, parts_at, nbytes=None):
              lay_ = e["recipe"]["layout"]
              return disp("P%d.qmvw_%s" % (l, proj), e, {1: dict(arena=pw, offset=blocks[l][proj], bytes=lay_["a_bytes"]),
                                                        2: dict(x, bytes=lay_["b_bytes"]),
                                                        3: pf(parts_at, lay_["c_bytes"])})
          for i in range(live if qmvwr else 0):      # the small-k verify route (MM 25.207): norm (fp32 out), qmvw qkv
              layers += [
                  disp("P%d.attn_norm" % l, nm, {0: pf(at["N1"] - nm["OUT"], nm["OUT"] + gm * d * 4),
                                                 1: pf(xs(i) - nm["X"], nm["X"] + gm * d * 2), 2: g1}),
                  qmvw_op("qkv", k["qmvw_qkv"], pf(at["N1"], 0), at["QKV"] + i * gm * QW * 4)]
          for i in range(live if qsmr else 0):       # the q4 route (MM 25.202): norm, qsm qkv, its partials summed
              layers += [
                  disp("P%d.attn_norm" % l, nm, {0: pf(at["N1"] - nm["OUT"], nm["OUT"] + gm * d * 2),
                                                 1: pf(xs(i) - nm["X"], nm["X"] + gm * d * 2), 2: g1}),
                  qsm_op("qkv", k["qsm_qkv"], pf(at["N1"], 0), gm * d * 2, "PQ"),
                  psum_op("ps_qkv", k["ps_qkv"], "PQ", None, pf(at["QKV"] + i * gm * QW * 4, gm * QW * 4))]
          for i in range(0 if (qsmr or qmvwr) else live):   # attention input, one gm-row slice at a time, into the whole QKV
              layers += [
                  disp("P%d.attn_norm" % l, nm, {0: pf(at["N1"] - nm["OUT"], nm["OUT"] + gm * d * 2),
                                                 1: pf(xs(i) - nm["X"], nm["X"] + gm * d * 2), 2: g1}),
                  dq("qkv", k["dq_qkv"], i == 0),
                  mm("qkv", k["mm_qkv"], pf(at["N1"], gm * d * 2), pf(at["QKV"] + i * gm * QW * 4, gm * QW * 4))]
          qin = "QKV"
          if dims["qknorm"]:
              # QK-NORM over the chunk's M rows (Qwen3, MM 25.188): QKV -> QKN, the gain the decode layer's own
              hn = k["qknorm"]
              layers += [disp("P%d.qk_norm" % l, hn, {0: pf(at["QKN"], M * QW * 4), 1: pf(at["QKV"], M * QW * 4),
                                                       2: (dims["qk_gain"][l] if dims.get("qk_gain")
                                                           else decode["L%d.qk_norm" % l]["binds"]["2"])})]
              qin = "QKN"
          layers += [
              (disp("P%d.append" % l, ap, {1: pf(at[qin], M * QW * 4), 2: gen, 3: dict(r3, bytes=ap["mma_region3_bytes"])})
               if route == "mma" else
               disp("P%d.append" % l, ap, {1: pf(at[qin], M * QW * 4), 2: gen, 3: dict(r3, bytes=ap["prefill_region3_bytes"])},
                    threads=M * ap["threadgroups_per_row"] * ap["threads_per_group"])),
              # the MMA attention binds decode's region3 at slots 1, 2 and 3 (every offset region-absolute)
              (disp("P%d.attn" % l, aw, {s_: dict(r3, bytes=aw["mma_region3_bytes"]) for s_ in (1, 2, 3)})
               if route == "mma" else
               disp("P%d.attn" % l, aw, {0: dict(r3, bytes=aw["prefill_region3_bytes"]), 1: pf(at[qin], M * QW * 4), 2: gen},
                    threads=M * aw["threadgroups_per_row"] * aw["threads_per_group"]))]
          for i in range(live if qmvwr else 0):      # the small-k route's rest of the layer: every row fp32
              pa = dict(arena=r3["arena"], offset=r3["offset"] + aw["PATTN"] + i * gm * HD * 4, bytes=gm * HD * 4)
              ps1, ps2 = k["ps_wo"]["recipe"]["layout"], k["ps_w2"]["recipe"]["layout"]
              sw = k["swiglu"]
              layers += [
                  qmvw_op("wo", k["qmvw_wo"], pa, at["Y"]),
                  disp("P%d.ps_wo" % l, k["ps_wo"], {1: pf(at["Y"], ps1["a_bytes"]),
                                                    2: pf(xs(i) - ps1["IN"], ps1["b_bytes"]),
                                                    3: pf(at["H"] - ps1["OUT"], ps1["c_bytes"])}),     # h = y + x
                  disp("P%d.ffn_norm" % l, ffm, {0: pf(at["N2"] - ffm["OUT"], ffm["OUT"] + gm * d * 4),
                                                 1: pf(at["H"] - ffm["X"], ffm["X"] + gm * d * 4), 2: g2}),
                  qmvw_op("w1", k["qmvw_w1"], pf(at["N2"], 0), at["G"]),
                  qmvw_op("w3", k["qmvw_w1"], pf(at["N2"], 0), at["U"]),
                  disp("P%d.swiglu" % l, sw, {1: pf(at["G"], gm * F * 4), 2: pf(at["U"], gm * F * 4),
                                              3: pf(at["ACT"], gm * F * 4)}),
                  qmvw_op("w2", k["qmvw_w2"], pf(at["ACT"], 0), at["P"]),
                  disp("P%d.ps_w2" % l, k["ps_w2"], {1: pf(at["P"], ps2["a_bytes"]),
                                                    2: pf(at["H"] - ps2["IN"], ps2["b_bytes"]),
                                                    3: pf(xs(i) - ps2["OUT"], ps2["c_bytes"])})]       # x = fp16(y + h)
          for i in range(live if qsmr else 0):       # the q4 route's rest of the layer
              pa = dict(arena=r3["arena"], offset=r3["offset"] + aw["PATTN"] + i * gm * HD * 2, bytes=gm * HD * 2)
              layers += [
                  qsm_op("wo", k["qsm_wo"], pa, gm * HD * 2, "PQ"),
                  psum_op("ps_wo", k["ps_wo"], "PQ", pf(xs(i), gm * d * 2), pf(at["H"], gm * d * 4)),   # h = y + x
                  disp("P%d.ffn_norm" % l, ffm, {0: pf(at["N2"] - ffm["OUT"], ffm["OUT"] + gm * d * 2),
                                                 1: pf(at["H"] - ffm["X"], ffm["X"] + gm * d * 4), 2: g2}),
                  qsm_op("w1", k["qsm_w1"], pf(at["N2"], 0), gm * d * 2, "PQ"),
                  qsm_op("w3", k["qsm_w1"], pf(at["N2"], 0), gm * d * 2, "PQ3"),
                  psum_op("ps_swiglu", k["ps_w1"], "PQ", "PQ3", pf(at["ACT"], gm * F * 2)),
                  qsm_op("w2", k["qsm_w2"], pf(at["ACT"], 0), gm * F * 2, "PQ"),
                  psum_op("ps_w2", k["ps_w2"], "PQ", pf(at["H"], gm * d * 4), pf(xs(i), gm * d * 2))]   # x = fp16(y + h)
          for i in range(0 if (qsmr or qmvwr) else live):   # the rest of the layer, one gm-row slice at a time
              layers += [
                  dq("wo", k["dq_wo"], i == 0),
                  mm("wo", k["mm_wo"], dict(arena=r3["arena"], offset=r3["offset"] + aw["PATTN"] + i * gm * HD * 2, bytes=gm * HD * 2),
                     pf(at["Y"], gm * d * 4)),
                  disp("P%d.residual" % l, k["residual"], {1: pf(at["Y"], gm * d * 4), 2: pf(xs(i), gm * d * 2), 3: pf(at["H"], sizes["H"])}),
                  disp("P%d.ffn_norm" % l, ffm, {0: pf(at["N2"] - ffm["OUT"], ffm["OUT"] + gm * d * 2),
                                                 1: pf(at["H"] - ffm["X"], ffm["X"] + gm * d * 4), 2: g2}),
                  dq("w1", k["dq_w1"], i == 0),
                  mm("w1", k["mm_w1"], pf(at["N2"], gm * d * 2), pf(at["G"], gm * F * gw)),
                  dq("w3", k["dq_w1"], i == 0)] + ([
                  disp("P%d.mm_w3sw" % l, fused, {1: pf(at["N2"], gm * d * 2),
                                                  2: w16("w3", fused["regions"]["slot2"]["W16"]["bytes"]),
                                                  3: pf(at["U"], sizes["U"])})] if fused else [
                  mm("w3", k["mm_w1"], pf(at["N2"], gm * d * 2), pf(at["U"], gm * F * gw)),
                  disp("P%d.swiglu" % l, k["swiglu"], {1: pf(at["G"], gm * F * gw), 2: pf(at["U"], gm * F * gw),
                                                       3: pf(at["ACT"], gm * F * 2)})]) + [
                  dq("w2", k["dq_w2"], i == 0),
                  mm("w2", k["mm_w2"], pf(at["ACT"], gm * F * 2), pf(at["P"], 2 * gm * d * 4)),
                  disp("P%d.fold" % l, k["fold"], {1: pf(at["P"], 2 * gm * d * 4), 2: pf(at["H"], gm * d * 4), 3: pf(xs(i), gm * d * 2)})]
          layers = [x for x in layers if x is not None]
      return layers
    per_chunk = [make_layers(k["attn_chunks"][c] if route == "mma" else k["attn"], c == 0, min(M, L - c * M))
                 for c in range(chunks)]
    layers = per_chunk[0]
    gen0 = decode["L0.attn_fused"]["binds"]["2"]
    steps = [dict(setup=True, writes=[], dispatches=setup)] if setup else []
    for c in range(chunks):
        p0 = c * M
        steps.append(dict(writes=[dict(arena=gen0["arena"], offset=gen0["offset"], u32=p0),
                                  dict(arena=PF, offset=at["X16"], bytes=M * d * 2, copy_from=dict(arena=PE, offset=p0 * d * 2))],
                          dispatches=per_chunk[c]))
    # the tail on the last row: decode's own final norm, head, argmax and generation step (gen writes log[L] and x[tok])
    r_x16 = decode["head.final_norm"]["binds"]["1"]
    fin = [decode[n] for n in ("head.final_norm", "head.lm", "head.argmax_pass1", "head.gen_step")]
    steps.append(tail_step(fin, r_x16, final_norm_x, at["X16"], gen0, L, M, chunks, d, logits))
    # what the bit-exact check reads after the prefill: every layer's K and V cache rows, the last chunk's x rows (after
    # the last layer), and the token log / q0 (the tail's first generated token)
    kv = k["attn"]
    dump = []
    for l in range(nl):
        r3 = decode["L%d.attn_fused" % l]["binds"]["0"]
        for nmk, off in (("K", kv["KOFF"]), ("V", kv["VOFF"])):
            dump.append(dict(name="L%d_%s" % (l, nmk), arena=r3["arena"], offset=r3["offset"] + off, bytes=8 * cap * 128 * 2))
    dump.append(dict(name="x_last_chunk", arena=PF, offset=at["X16"], bytes=M * d * 2))
    dump.append(dict(name="gen", arena=gen0["arena"], offset=gen0["offset"], bytes=4 + 4 * cap))
    if logits:
        dump.append(dict(name="logits", arena=logits["arena"], offset=logits["offset"], bytes=int(logits["bytes"])))
    return dict(arenas=arenas, files=files, zero_init=[dict(arena=PF, offset=0, bytes=arenas[PF])],
                prefill=dict(steps=steps, q0_after=L, M=M, chunks=chunks, dispatches_per_chunk=len(layers), dump=dump,
                             persistent_w16=persist))


# ---------------------------------------------------------------------------------------------------------------------
# THE CPU REFERENCE of the whole prefill: every op in its kernel's own arithmetic (M1 dequant and MMA order, M2 append and
# attention, M3 row norms and elementwise), composed per layer. It is what the GPU prefill must reproduce bit for bit
# (the KV cache rows and the last row's x), independent of placement.

def dequant_reference(npz, bits, N, K):
    """W16 [N][K] = fp16_rne(fp32(fp32(q) * s) + b), bf16 s/b widened (M1's contract, g17qmm.dequant_reference)."""
    import numpy as np
    from g17q4graph import _unpack
    q = _unpack(npz["W"], bits, N, K).astype(np.float32)
    s = (np.asarray(npz["S"], np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(N, K // 64)
    b = (np.asarray(npz["B"], np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(N, K // 64)
    s = np.repeat(s, 64, axis=1); b = np.repeat(b, 64, axis=1)
    return ((q * s).astype(np.float32) + b).astype(np.float32).astype(np.float16)


def gemm_reference(a16, w16, split_k=1):
    """y fp32 [M][N] = A fp16 [M][K] x W16^T, in the tensor unit's issue order (g17tensorcommonruntime._gemm_mma_fast);
    split_k halves K and sums the partials p0 + p1 in fp32 (M1's w2)."""
    import numpy as np
    import g17tensorcommonruntime as TCR
    A = np.asarray(a16, np.float16).astype(np.float32)
    B = np.asarray(w16, np.float16).astype(np.float32).T          # [K][N]
    M, K = A.shape
    N = B.shape[1]
    if split_k == 1:
        return np.asarray(TCR._gemm_mma_fast(A, B, None, M, N, K), np.float32)
    h = K // split_k
    parts = [np.asarray(TCR._gemm_mma_fast(A[:, i * h:(i + 1) * h], B[i * h:(i + 1) * h], None, M, N, h), np.float32)
             for i in range(split_k)]
    return parts


def reference(bits, prompt_ids, cap, weights_dir, layers=None, ffn16=False):
    """The prefill of `prompt_ids` through `layers` (default all): per layer's K/V rows, and the last row's x fp16 after
    the last layer. Returns dict(x=[L][2048] fp16 after the last layer run, K=[l] [8][cap][128], V=...). ffn16: the w1 and
    w3 outputs rounded to fp16 (RNE) before the SwiGLU, as the half-epilogue GEMMs store them (MM 25.183)."""
    import numpy as np
    import g17decodeops as O
    import g17prefillattn as P
    import g17rows as RW
    import g17realmodel as M
    W = Path(weights_dir)
    L = len(prompt_ids)
    spec = M.layer_spec(0)
    emb = np.load(W / "embed.npy", mmap_mode="r")
    x16 = np.asarray(emb[list(prompt_ids)], np.float16)
    lay = P.prefill_layout(cap, cap, out16=True)
    pos = np.arange(cap)[:, None] * (M.ROPE_THETA ** (-np.arange(0, 128, 2, dtype=np.float64) / 128))[None, :]
    cos, sin = np.cos(pos).astype(np.float32), np.sin(pos).astype(np.float32)
    out = dict(K=[], V=[])
    for l in (range(M.LAYERS) if layers is None else layers):
        g1 = np.load(W / ("L%d_g1.npy" % l)).astype(np.float32)
        g2 = np.load(W / ("L%d_g2.npy" % l)).astype(np.float32)
        n1 = np.asarray(O.rmsnorm_wide_rows_reference(x16.astype(np.float32), g1, spec), np.float16)
        qkv = gemm_reference(n1, dequant_reference(np.load(W / ("L%d_qkv.npz" % l)), bits, 4096, 2048))
        Kc = np.zeros((8, cap, 128), np.float16); Vc = np.zeros((8, cap, 128), np.float16)
        attn16, _q16, K, V = P.prefill_reference(lay, qkv, cos, sin, Kc, Vc, 0, L)
        out["K"].append(np.asarray(K, np.float16)); out["V"].append(np.asarray(V, np.float16))
        y = gemm_reference(np.asarray(attn16, np.float16).reshape(L, 2048),
                           dequant_reference(np.load(W / ("L%d_wo.npz" % l)), bits, 2048, 2048))
        h = RW.rows_reference(dict(kind="residual"), (y, x16))["h"]
        n2 = np.asarray(O.rmsnorm_wide_rows_reference(h, g2, spec), np.float16)
        ffn = np.load(W / ("L%d_ffn.npz" % l))
        w13 = dequant_reference(ffn, bits, 16384, 2048)
        g = gemm_reference(n2, w13[:8192]); u = gemm_reference(n2, w13[8192:])
        if ffn16:
            g, u = np.asarray(g, np.float16), np.asarray(u, np.float16)
        act = RW.rows_reference(dict(kind="swiglu", in16=ffn16), (g, u))["act"]
        p0, p1 = gemm_reference(act, dequant_reference(np.load(W / ("L%d_w2.npz" % l)), bits, 2048, 8192), split_k=2)
        x16 = RW.rows_reference(dict(kind="fold"), (np.stack([p0, p1]).reshape(2, -1), h.reshape(-1)))["x"].reshape(L, 2048)
    out["x"] = x16
    return out
