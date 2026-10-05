#!/usr/bin/env python3
"""THE MODEL FROM SOURCE, IN ONE COMMAND (MM 25.142.8): a model config and a deliver root in; a checked graph, its
tokens and its tokens per second against mlx-lm out.

    python3 tools/g17modelbuild.py --config tools/models/internlm2_q8_best.json --deliver DIR [--check N] [--bench R]

1. build  - tools/g17q4graph.py builds the graph from the deliver root's index.json (flat entries keyed by kind / bits
            / role / variant / cap; tools/g17deliver.py writes it from main's kernel builders) and the config's variant
            per role. No other bundle source and no feature switches.
2. check  - the CPU simulator (every kernel's reference, in the kernel's own reduction order) runs the prompt and N
            generated tokens; decodegen runs the graph on the GPU; the N generated tokens must be identical.
3. bench  - decodegen's pipelined decode of 249 tokens and mlx-lm's own quantized checkpoint decoding the same prompt
            for 249 tokens (tools/g17model_mlx.py bench), R pairs with the order alternated (MM rule 43).

Writes the report to the graph directory (modelbuild.json) and prints it. Exit 0 only if every requested step passed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _env(config, deliver):
    """the environment for the graph's processes: the deliver root and the config go as ARGUMENTS (_graph_args), and no
    G17_Q4_* variable may leak in from the caller's shell (the graph has no feature switches; the two it still honours
    are deprecated)"""
    return {k: v for k, v in os.environ.items() if not k.startswith("G17_Q4_")}


def _graph_args(config, deliver):
    return ["--deliver", str(Path(deliver).resolve()), "--config", str(Path(config).resolve())]


def _run(cmd, env=None, timeout=3600):
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    if p.returncode != 0:
        raise RuntimeError("%s failed (%d): %s" % (" ".join(map(str, cmd[:3])), p.returncode, (p.stderr or p.stdout)[-2000:]))
    return p.stdout


def _graph_dir(cfg):
    sys.path[:0] = [str(TOOLS), str(ROOT)]
    import g17realmodel as M
    return M.OUT / ("graph_q%d_%s" % (cfg["bits"], cfg.get("name", "model")))


ROLES = ("qkv", "wo_res1", "ffn", "w2_res2", "head")


def outside_root(bundles, root):
    """The bundles (paths) that do not resolve, symlinks followed, to somewhere inside the deliver root."""
    r = os.path.realpath(root)
    return sorted(b for b in bundles if os.path.commonpath([os.path.realpath(b), r]) != r)


def _prefill_check(a, cfg, graph, gdir, bitexact_max=128):
    """A prefill graph's checks: its generated tokens against the same graph fed token by token (the decode path's own
    check covers that path), and, for the scalar attention route with a prompt of at most `bitexact_max` tokens, every
    layer's K/V rows and the last x rows against g17prefillgraph.reference, bit for bit."""
    import numpy as np
    import g17prefillgraph as PG
    import g17realmodel as M
    L, cap = len(graph["prompt_ids"]), graph["capacity"]
    res = dict(prompt_len=L, route=cfg["prefill"].get("attn", "scalar"))
    if res["route"] == "scalar" and L <= bitexact_max:
        d = gdir / "prefill_dump"
        d.mkdir(exist_ok=True)
        env = dict(os.environ, DECODEGEN_PIPELINED="1", DECODEGEN_PREFILL_DUMP=str(d))
        _run([str(a.decodegen), str(gdir / "graph.json"), str(gdir / "prefill_dump.json"), "8"], env=env, timeout=900)
        ref = PG.reference(cfg["bits"], graph["prompt_ids"], cap, M.OUT / ("graph_q%d" % cfg["bits"]) / "weights")
        bad = 0
        for l in range(len(ref["K"])):
            for nm in ("K", "V"):
                got = np.fromfile(d / ("L%d_%s.bin" % (l, nm)), np.uint16).reshape(8, cap, 128)[:, :L]
                bad += int((got != np.asarray(ref[nm][l], np.float16).view(np.uint16)[:, :L]).sum())
        x = np.fromfile(d / "x_last_chunk.bin", np.uint16).reshape(-1, 2048)
        M_ = graph["prefill"]["M"]
        last = np.asarray(ref["x"], np.float16).view(np.uint16)[(graph["prefill"]["chunks"] - 1) * M_:]
        xbad = int((x[:len(last)] != last).sum())
        res.update(kv_halves_differing=bad, x_halves_differing=xbad, bitexact=bad == 0 and xbad == 0)
    res["ok"] = res.get("bitexact", True)
    return res


def _single_key(scfg, graph, order):
    """What a single run's tokens depend on: the config without its name and rotation (the rotation is in the prompt
    ids), the prompt ids and the reference order. In-process only."""
    import hashlib
    c = {k: v for k, v in scfg.items() if k not in ("name", "prompt_rotate")}
    h = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]
    return (scfg["bits"], h, tuple(graph["prompt_ids"]), order)


def batch_check_many(items, deliver_for, decodegen, tokens, workers):
    """One invocation's batch checks (MM 25.144.3): items = [(config path, single order)], every batched graph built
    and run, then each DISTINCT single run (_single_key) simulated and GPU-run ONCE and shared by every item that needs
    it. B 2 and 4 are prefixes of B 8 and a rotation by the prompt length is sequence 0, so the keys are far fewer than
    the sequences. The simulations run in `workers` parallel processes (G17_REF_THREADS=1 each); the GPU runs are
    serial (the GPU lock). Each item's verdict is _batch_check's: every sequence's batched GPU tokens, single GPU
    tokens and single simulator tokens identical."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    env = _env(None, None)
    plans, singles = [], {}
    for cpath, order in items:
        cfg = json.loads(Path(cpath).read_text())
        bits, deliver = cfg["bits"], deliver_for(cfg["bits"])
        gdir = _graph_dir(cfg)
        gdir.mkdir(parents=True, exist_ok=True)
        if not (gdir / "weights").exists():
            (gdir / "weights").symlink_to(Path("..") / ("graph_q%d" % bits) / "weights")
        _run([sys.executable, str(TOOLS / "g17q4graph.py"), "build", "--bits", str(bits)] + _graph_args(cpath, deliver), env=env)
        graph = json.loads((gdir / "graph.json").read_text())
        outside = outside_root(sorted({d["bundle"] for d in graph["dispatches"]}), deliver)
        seqs = []
        for sq in range(int(cfg["batch"])):
            scfg = single_of(cfg, sq, order)
            sfile = gdir / ("single_seq%d%s.json" % (sq, "_splitk" if order == "splitk" else ""))
            sfile.write_text(json.dumps(scfg) + "\n")
            sdir = _graph_dir(scfg)
            sdir.mkdir(parents=True, exist_ok=True)
            if not (sdir / "weights").exists():
                (sdir / "weights").symlink_to(Path("..") / ("graph_q%d" % bits) / "weights")
            _run([sys.executable, str(TOOLS / "g17q4graph.py"), "build", "--bits", str(bits)] + _graph_args(sfile, deliver), env=env)
            key = _single_key(scfg, json.loads((sdir / "graph.json").read_text()), order)
            singles.setdefault(key, dict(sfile=sfile, sdir=sdir, bits=bits, deliver=deliver))
            seqs.append(key)
        plans.append(dict(config=str(cpath), order=order, cfg=cfg, gdir=gdir, seqs=seqs, outside=outside))
    t_build = time.time() - t0

    def simulate(key):
        u = singles[key]
        out = _run([sys.executable, str(TOOLS / "g17q4graph.py"), "simulate", "--bits", str(u["bits"]), "--tokens",
                    str(tokens)] + _graph_args(u["sfile"], u["deliver"]), env=dict(env, G17_REF_THREADS="1"))
        return json.loads(out.strip().splitlines()[-1])["out_ids"][:tokens]

    with ThreadPoolExecutor(workers) as ex:                   # each thread waits on one simulator PROCESS
        futs = {k: ex.submit(simulate, k) for k in singles}
        # the GPU runs, serial, while the simulations run
        batched = {i: [ids[:tokens] for ids in _decodegen(decodegen, pl["gdir"] / "graph.json", max(tokens, 8),
                                                            pl["gdir"] / "modelbuild_check.json")["out_ids_per_seq"]]
                   for i, pl in enumerate(plans)}
        gpu = {k: _decodegen(decodegen, u["sdir"] / "graph.json", max(tokens, 8), u["sdir"] / "modelbuild_check.json")
               ["out_ids"][:tokens] for k, u in singles.items()}
        sim = {k: f.result() for k, f in futs.items()}
    reps = []
    for i, pl in enumerate(plans):
        rows, ok = [], not pl["outside"]
        for sq, key in enumerate(pl["seqs"]):
            same = batched[i][sq] == gpu[key] == sim[key]
            ok &= same
            rows.append(dict(seq=sq, batched_gpu=batched[i][sq], single_gpu=gpu[key], single_simulator=sim[key],
                             identical=same))
        reps.append(dict(config=pl["config"], single_order=pl["order"], ok=bool(ok), bundles_outside_deliver_root=pl["outside"],
                         check=dict(batch=int(pl["cfg"]["batch"]), tokens=tokens, sequences=rows,
                                    identical=all(r["identical"] for r in rows),
                                    distinct_sequences=len({tuple(x) for x in batched[i]}))))
    return dict(items=reps, sequences=sum(len(pl["seqs"]) for pl in plans), distinct_single_runs=len(singles),
                workers=workers, seconds=dict(build=round(t_build, 1), total=round(time.time() - t0, 1)))


def _decodegen(binary, graph, tokens, out):
    env = dict(os.environ, DECODEGEN_PIPELINED="1")
    _run([binary, str(graph), str(out), str(tokens)], env=env, timeout=900)
    return json.loads(Path(out).read_text())


def single_of(cfg, seq, order="config"):
    """The B = 1 config of a batched config's sequence `seq`: the same model, the split-K lean variants without the batch
    (the batched kernels compute each sequence's value exactly as these do), and the prompt rotated as that sequence's.
    A dequantize-once ("dq") projection keeps "dq": its single run is the single-vector dq kernel and the simulator
    follows that bundle's order (g17qmv.qmv_dq_reference), so the B single runs share the batched GPU's fp32 order.
    order="splitk" is the CONTROL: "dq" dropped, the single runs in the split-K order, which a near-tie can separate
    from the batched dq tokens (MM 25.144.3, q4 B 4 sequence 0)."""
    s = json.loads(json.dumps(cfg))
    s.pop("batch", None)
    s["name"] = "%s_seq%d%s" % (cfg.get("name", "model"), seq, "_splitk" if order == "splitk" else "")
    s["prompt_rotate"] = int(cfg.get("prompt_rotate", 0)) + seq
    for k in ("norm", "final_norm", "attn"):
        s[k] = {a: b for a, b in s[k].items() if a != "batch"}
    s["qmv"] = {r: {a: b for a, b in v.items() if a not in ("batch", "pass") + (("dq",) if order == "splitk" else ())}
                for r, v in s["qmv"].items()}
    s.pop("gen", None)
    return s


def _batch_check(a, cfg, gdir, gargs_for, env, tokens):
    """Each sequence of the batched graph against B separate single-sequence runs of its prompt: the GPU batched tokens,
    the single GPU tokens and the single simulator tokens must be identical, sequence by sequence."""
    B = int(cfg["batch"])
    got = _decodegen(a.decodegen, gdir / "graph.json", max(tokens, 8), gdir / "modelbuild_check.json")
    per = [ids[:tokens] for ids in got["out_ids_per_seq"]]
    seqs, ok = [], True
    for s in range(B):
        scfg = single_of(cfg, s, a.single_order)
        sfile = gdir / ("single_seq%d.json" % s)
        sfile.write_text(json.dumps(scfg) + "\n")
        sdir = _graph_dir(scfg)
        sdir.mkdir(parents=True, exist_ok=True)
        wl = sdir / "weights"
        if not wl.exists():
            wl.symlink_to(Path("..") / ("graph_q%d" % cfg["bits"]) / "weights")
        sargs = gargs_for(sfile)
        _run([sys.executable, str(TOOLS / "g17q4graph.py"), "build", "--bits", str(cfg["bits"])] + sargs, env=env)
        sim = json.loads(_run([sys.executable, str(TOOLS / "g17q4graph.py"), "simulate", "--bits", str(cfg["bits"]),
                               "--tokens", str(tokens)] + sargs, env=env).strip().splitlines()[-1])["out_ids"][:tokens]
        one = _decodegen(a.decodegen, sdir / "graph.json", max(tokens, 8), sdir / "modelbuild_check.json")["out_ids"][:tokens]
        same = per[s] == one == sim
        ok &= same
        seqs.append(dict(seq=s, batched_gpu=per[s], single_gpu=one, single_simulator=sim, identical=same))
    return dict(batch=B, tokens=tokens, sequences=seqs, identical=ok,
                distinct_sequences=len({tuple(x) for x in per}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path)
    ap.add_argument("--deliver", required=True, type=Path)
    ap.add_argument("--check", type=int, default=2, help="generated tokens the simulator checks (0 skips)")
    ap.add_argument("--bench", type=int, default=0, help="alternated ours/mlx-lm pairs (0 skips)")
    ap.add_argument("--decodegen", type=Path, default=TOOLS / "g17decodegen")
    ap.add_argument("--single-order", choices=("config", "splitk"), default="config",
                    help="a batch check's single runs: the config's own order, or the split-K order (the control)")
    ap.add_argument("--configs", nargs="+", type=Path, help="batch checks in ONE invocation, single runs shared "
                    "(batch_check_many); --deliver may hold {bits}")
    ap.add_argument("--controls", nargs="*", type=Path, default=[],
                    help="with --configs: also checked with the single runs in the split-K order (expected to diverge "
                         "where a near-tie falls; the control that the check sees the order)")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    a = ap.parse_args(argv)
    if (a.config is None) == (not a.configs):
        ap.error("one of --config or --configs")
    if a.configs:
        items = [(c, "config") for c in a.configs] + [(c, "splitk") for c in a.controls]
        rep = batch_check_many(items, lambda b: Path(str(a.deliver).format(bits=b)).resolve(), a.decodegen, a.check,
                               a.workers)
        print(json.dumps(rep, indent=1))
        return 0 if all(r["ok"] for r in rep["items"] if r["single_order"] == "config") else 1
    cfg = json.loads(a.config.read_text())
    bits, env, gargs = cfg["bits"], _env(a.config, a.deliver), _graph_args(a.config, a.deliver)
    gdir = _graph_dir(cfg)
    rep = dict(config=str(a.config), deliver=str(a.deliver.resolve()), bits=bits, graph=str(gdir / "graph.json"))
    (gdir).mkdir(parents=True, exist_ok=True)
    wlink = gdir / "weights"
    if not wlink.exists():
        wlink.symlink_to(Path("..") / ("graph_q%d" % bits) / "weights")
    rep["build"] = _run([sys.executable, str(TOOLS / "g17q4graph.py"), "build", "--bits", str(bits)] + gargs, env=env).strip().splitlines()[-1]
    graph = json.loads((gdir / "graph.json").read_text())
    paths = sorted({d["bundle"] for d in graph["dispatches"]})
    rep["bundles"] = [os.path.relpath(os.path.realpath(b), os.path.realpath(a.deliver)) for b in paths]
    outside = outside_root(paths, a.deliver)
    rep["bundles_outside_deliver_root"] = outside
    ok = not outside
    if a.check and cfg.get("batch", 1) > 1:
        rep["check"] = _batch_check(a, cfg, gdir, lambda f: _graph_args(f, a.deliver), env, a.check)
        ok &= rep["check"]["identical"]
    elif a.check:
        sim = _run([sys.executable, str(TOOLS / "g17q4graph.py"), "simulate", "--bits", str(bits), "--tokens", str(a.check)] + gargs, env=env)
        sim_ids = json.loads(sim.strip().splitlines()[-1])["out_ids"][:a.check]
        gpu_ids = _decodegen(a.decodegen, gdir / "graph.json", max(a.check, 8), gdir / "modelbuild_check.json")["out_ids"][:a.check]
        rep["check"] = dict(tokens=a.check, simulator=sim_ids, gpu=gpu_ids, identical=sim_ids == gpu_ids)
        ok &= sim_ids == gpu_ids
    pf = graph.get("prefill")
    if a.check and pf:
        rep["prefill_check"] = _prefill_check(a, cfg, graph, gdir)
        ok &= rep["prefill_check"]["ok"]
    if a.bench:
        pairs = []
        ids = ",".join(str(i) for i in graph["prompt_ids"])
        for r in range(a.bench):
            def ours():
                d = _decodegen(a.decodegen, gdir / "graph.json", 249, gdir / "modelbuild_bench.json")
                return d["tokens_per_s"], d.get("ttft_s")
            def mlx():
                out = _run([sys.executable, str(TOOLS / "g17model_mlx.py"), "bench", "--bits", str(bits), "--ids", ids,
                            "--tokens", "249", "--batch", str(cfg.get("batch", 1))], timeout=900)
                d = json.loads(out.strip().splitlines()[-1])
                return d["tokens_per_s"], d.get("ttft_s")
            (o, ot), (m, mt) = (ours(), mlx()) if r % 2 == 0 else tuple(reversed((mlx(), ours())))
            pairs.append(dict(ours=round(o, 1), mlx_lm=round(m, 1), ratio=round(o / m, 3),
                              ttft_ours_s=round(ot, 4) if ot is not None else None,
                              ttft_mlx_s=round(mt, 4) if mt is not None else None))
        rep["bench"] = dict(pairs=pairs, mean_ratio=round(sum(p["ratio"] for p in pairs) / len(pairs), 3),
                            protocol="pipelined decodegen 249 tokens vs mlx-lm fixed-length 249 tokens, order alternated")
    rep["ok"] = bool(ok)
    (gdir / "modelbuild.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
