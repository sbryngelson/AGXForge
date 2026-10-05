#!/usr/bin/env python3
"""Acceptance entry point for AGXForge's half-precision embedding scan.

This command never loads Metal or dispatches a kernel. It writes a reproducible
blocker report, the exact operation contract, reference fixtures, and compiler
diagnostics. A diagnostic passing never marks the actual scan as built.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


@dataclass(frozen=True)
class Scan:
    rows: int = 500_000
    columns: int = 384
    input_dtype: str = "float16"
    accumulator_dtype: str = "float32"
    output_dtype: str = "float16"

    def __post_init__(self):
        if not (1 <= self.rows and 1 <= self.columns):
            raise ValueError("scan dimensions must be positive")
        if self.rows * self.columns > 2**32:
            raise ValueError("this acceptance target requires 32-bit element indices")
        if (self.input_dtype, self.accumulator_dtype, self.output_dtype) != (
                "float16", "float32", "float16"):
            raise ValueError("changing the scan's precision changes the acceptance target")

    def contract(self):
        return {
            **asdict(self),
            "operation": "scores[row] = sum(index[row, k] * embedding[k], k=0..columns-1)",
            "source": "spike/serve/mix_mlx.py:scan_fn",
            "layout": "contiguous row-major; independent index and embedding allocations",
            "bindings": [
                {"index": 0, "name": "index", "readonly": True,
                 "element_type": "half", "bytes": self.rows * self.columns * 2},
                {"index": 1, "name": "embedding", "readonly": True,
                 "element_type": "half", "bytes": self.columns * 2},
                {"index": 2, "name": "scores", "readonly": False,
                 "element_type": "half", "bytes": self.rows * 2},
            ],
            "precision_policy": (
                "FP32 accumulation followed by one round-to-nearest-even half conversion. "
                "This is the proposed native numerical contract; equivalence to the current "
                "MLX implementation must still be measured, including output dtype."),
            "domain": "finite half inputs whose absolute product sum is <= 60000",
            "launch": {"logical_threads": self.rows, "one_output_per_thread": True,
                       "exact_grid_required": True,
                       "note": "the SSA candidate requires exactly rows threads; a rounded-up launch needs an explicit row guard"},
        }

    def metal_reference(self):
        return f"""#include <metal_stdlib>
using namespace metal;
kernel void scan(device const half *index [[buffer(0)]],
                 device const half *embedding [[buffer(1)]],
                 device half *scores [[buffer(2)]],
                 uint row [[thread_position_in_grid]]) {{
    if (row >= {self.rows}u) return;
    float acc = 0.0f;
    for (uint k = 0; k < {self.columns}u; ++k)
        acc = fma(float(index[row * {self.columns}u + k]), float(embedding[k]), acc);
    scores[row] = half(acc);
}}
"""


def reference(matrix, vector):
    """Exact half products summed independently with math.fsum, plus an error bound.

    A half product is exactly representable in FP32. The bound covers FP32
    summation/reassociation and the final half rounding, including cancellation.
    No relative-error division by a near-zero reference is used.
    """
    import numpy as np
    a, x = np.asarray(matrix), np.asarray(vector)
    if a.dtype != np.float16 or x.dtype != np.float16:
        raise ValueError("reference inputs must be float16, without implicit conversion")
    if a.ndim != 2 or x.ndim != 1 or a.shape[1] != x.size or not x.size:
        raise ValueError("expected a nonempty MxK matrix and a K-vector")
    if not (np.isfinite(a).all() and np.isfinite(x).all()):
        raise ValueError("nonfinite values are outside the initial numerical contract")
    n = x.size
    eps = 2.0**-24
    if n * eps >= 1:
        raise ValueError("reduction too long for this error bound")
    gamma = n * eps / (1 - n * eps)
    exact, bounds = [], []
    for row in a:
        products = [float(v) * float(w) for v, w in zip(row, x)]
        total = math.fsum(products)
        magnitude = math.fsum(abs(v) for v in products)
        if magnitude > 60000:
            raise ValueError("absolute product sum exceeds the finite-output contract")
        fp32_error = gamma * magnitude
        upper = abs(total) + fp32_error
        # Largest half spacing over the possible accumulated value, conservatively
        # using a full ULP. 2^-24 also covers half subnormal output rounding.
        half_ulp = max(2.0**-24, 2.0**(math.frexp(upper)[1] - 11)) if upper else 2.0**-24
        exact.append(total)
        bounds.append(fp32_error + half_ulp)
    return np.array(exact, dtype=np.float64), np.array(bounds, dtype=np.float64)


def check_output(matrix, vector, output):
    import numpy as np
    want, tolerance = reference(matrix, vector)
    got = np.asarray(output)
    if got.dtype != np.float16 or got.shape != want.shape:
        return {"ok": False, "reason": "output must be a float16 vector with one value per row"}
    good = np.isfinite(got) & (np.abs(got.astype(np.float64) - want) <= tolerance)
    return {"ok": bool(good.all()), "rows": len(got),
            "failed_rows": np.flatnonzero(~good).tolist(),
            "max_absolute_error": float(np.max(np.abs(got.astype(np.float64) - want))),
            "max_allowed_error": float(tolerance.max())}


def write_fixtures(directory, columns):
    import numpy as np
    rng = np.random.default_rng(1729)
    cases = []
    shapes = sorted({(1, 1), (3, 7), (33, columns)})
    for m, k in shapes:
        a = (rng.standard_normal((m, k)) * 0.05).astype(np.float16)
        x = (rng.standard_normal(k) * 0.05).astype(np.float16)
        # A zero row, a basis-vector row and alternating signs distinguish common
        # wrong-buffer, missing-store and cancellation failures.
        if m > 1:
            a[0] = 0
            a[1] = 0
            a[1, -1] = 1
        if m > 3:
            a[2] = np.where(np.arange(k) % 2, -0.125, 0.125)
        exact, tolerance = reference(a, x)
        name = f"scan-{m}x{k}.npz"
        np.savez(directory / name, index=a, embedding=x, exact=exact, tolerance=tolerance)
        cases.append({"path": name, "rows": m, "columns": k,
                      "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()})
    return cases


def float_scan_ir(columns, *, loop=False, packed=False):
    """FP32 diagnostic only. It deliberately does not stand in for the half target."""
    import g17ir as ir
    buffers = ([ir.Buffer("packed", 1), ir.Buffer("scores", 2)] if packed else
               [ir.Buffer("index", 0), ir.Buffer("embedding", 1), ir.Buffer("scores", 2)])
    fn = ir.Function("scan_f32_diagnostic", buffers)
    pre = fn.block("pre")
    b = ir.Builder(fn, pre)
    row = b.builtin("thread_position_in_grid", name="row")
    factor = ir.Imm(columns) if columns <= 255 else b.const(columns, name="stride")
    base = b.mul(row, factor, name="base")
    acc = b.const(0, name="zero")
    matrix, vector = (buffers[0], buffers[0]) if packed else buffers[:2]

    def term(k, acc, suffix):
        ai = b.add(base, k, name="ai" + suffix)
        # The packed diagnostic assumes exactly 32 matrix rows, a CONTROL layout.
        xi = b.add(k, ir.Imm(32 * columns), name="xi" + suffix) if packed else k
        a = b.load(matrix, ai, name="a" + suffix)
        x = b.load(vector, xi, name="x" + suffix)
        return b.fadd(acc, b.fmul(a, x, name="product" + suffix), name="sum" + suffix)

    if loop:
        hdr, end = fn.block("loop"), fn.block("exit")
        zero = b.const(0, name="k0")
        b.br(hdr)
        b.at(hdr)
        carried = b.phi(acc, name="acc")
        k = b.phi(zero, name="k")
        acc = term(k, carried, "")
        nxt = b.add(k, ir.Imm(1), name="next")
        ir.Builder.phi_latch(k, nxt)
        ir.Builder.phi_latch(carried, acc)
        b.br_cond(b.cmp(nxt, columns, "lt", name="continue"), hdr, end)
        b.at(end)
    else:
        for k in range(columns):
            acc = term(b.const(k, name=f"k{k}"), acc, str(k))
    b.store_at(buffers[-1], row, acc)
    b.ret()
    return fn


def target_ir(spec):
    """Actual scan SSA with explicit missing precision operations.

    These names request semantic extensions, never silent word accesses. The
    existing verifier must reject them until compiler lowerings exist.
    """
    import g17ir as ir
    f = ir.Function("scan", [ir.Buffer("index", 0, elem="f16"),
                             ir.Buffer("embedding", 1, elem="f16"),
                             ir.Buffer("scores", 2, elem="f16")])
    pre, hdr, end = f.block("pre"), f.block("loop"), f.block("exit")
    b = ir.Builder(f, pre)
    row = b.builtin("thread_position_in_grid", name="row")
    stride = b.const(spec.columns, name="stride")
    base = b.mul(row, stride, name="base")
    zero = b.const(0, name="zero")
    k0 = b.const(0, name="k0")
    b.br(hdr)
    b.at(hdr)
    acc, k = b.phi(zero, name="acc"), b.phi(k0, name="k")
    index = b.add(base, k, name="index_address")
    a = b._def("load_f16", [f.buffers[0], index], type=ir.I16, name="a_half")
    x = b._def("load_f16", [f.buffers[1], k], type=ir.I16, name="x_half")
    af = b._def("f16_to_f32", [a], name="a_float")
    xf = b._def("f16_to_f32", [x], name="x_float")
    nxtacc = b.fma(af, xf, acc, name="sum")
    nxtk = b.add(k, ir.Imm(1), name="next_k")
    ir.Builder.phi_latch(acc, nxtacc)
    ir.Builder.phi_latch(k, nxtk)
    b.br_cond(b.cmp(nxtk, spec.columns, "lt", name="more"), hdr, end)
    b.at(end)
    half = b._def("f32_to_f16_rte", [nxtacc], type=ir.I16, name="result_half")
    b.b.add(ir.Op("store_f16", None, [f.buffers[2], row, half]))
    b.ret()
    return f


class DependencyBlocked(RuntimeError):
    pass


def install_dependency_guard(events):
    """Reject compiler subprocesses and external project caches before importing it.

    Recovered ISA files in the checkout are allowed. Python/runtime imports are
    allowed. This is an input-dependency test, not a security sandbox.
    """
    def hook(event, args):
        if event in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.system"):
            events.append({"event": event, "dependency": str(args[0])})
            raise DependencyBlocked("compiler attempted subprocess: " + str(args[0]))
        if event == "ctypes.dlopen" and isinstance(args[0], str):
            name = args[0]
            if any(s in name for s in ("Metal", "AGX", "libaccel", "CoreML")):
                events.append({"event": event, "dependency": name})
                raise DependencyBlocked("compiler attempted hardware library: " + name)
        if event == "open" and isinstance(args[0], (str, bytes)):
            name = os.fsdecode(args[0])
            path = Path(name).expanduser().resolve()
            if "/.cache/agxforge/" in str(path):
                events.append({"event": event, "dependency": str(path)})
                raise DependencyBlocked("compiler attempted external agxforge cache: " + str(path))
    sys.addaudithook(hook)


def compile_stages(fn):
    import g17ir as ir
    import g17cc as cc
    result = {"stages": [], "kind": "diagnostic", "scope": "FP32 control; not the half scan"}
    def stage(name, function):
        result["active_stage"] = name
        value = function()
        result["stages"].append({"name": name, "status": "passed"})
        return value
    try:
        stage("ir", lambda: ir.verify(fn))
        selected = stage("selection", lambda: cc.select(fn))
        allocated = stage("allocation", lambda: cc.Alloc(range(4, 16)).run(selected))
        code, layout = stage("encoding", lambda: cc.emit(allocated))
        def verify():
            errors = cc.selfcheck(layout) + cc._check_flag_discipline(layout)
            if errors:
                raise AssertionError("; ".join(errors))
        stage("compiler_selfcheck", verify)
        result.update(status="passed", bytes=len(code), instructions=len(layout),
                      sha256=hashlib.sha256(code).hexdigest())
    except Exception as exc:
        result.update(status="blocked" if isinstance(exc, (cc.Unsupported, DependencyBlocked)) else "error",
                      error_type=type(exc).__name__, reason=str(exc))
    return result


def worker(name, spec=None):
    """One probe in a fresh process, with its own empty solver cache."""
    events = []
    install_dependency_guard(events)
    import g17ir as ir
    import g17cc as cc
    if name == "target":
        fn = target_ir(spec or Scan())
        result = compile_stages(fn)
        result.update(kind="target", scope="actual half scan with explicit precision operations",
                      requested_ir=repr(fn))
        needed = {"load_f16", "f16_to_f32", "f32_to_f16_rte"} - ir.DEFS
        result["unregistered_value_ops"] = sorted(needed)
        if needed:
            result.update(status="blocked", compiler_diagnostic=result.get("reason"),
                          reason="scan adapter requires explicit value operations: " + ", ".join(sorted(needed)))
        elif result["status"] == "passed":
            result.update(status="needs_review", reason="target encoded; precision semantics and complete ABI handoff still require acceptance")
    elif name.startswith("f32_"):
        _, form, count = name.split("_")
        result = compile_stages(float_scan_ir(int(count), loop=form == "loop"))
    elif name == "half_load":
        def selected(width):
            f = ir.Function("half_load", [ir.Buffer("x", 1, elem="f16"), ir.Buffer("y", 2)])
            b = ir.Builder(f, f.block("entry"))
            i = b.builtin("thread_position_in_grid")
            v = b.load(f.buffers[0], i, type=ir.I16, width=width)
            b.store_at(f.buffers[1], i, b.add(v, ir.Imm(0)))
            b.ret()
            ir.verify(f)
            return [{"form": m.form, "fields": m.fields} for m in cc.select(f) if m.form.startswith("load")]
        word, half = selected("word"), selected("half")
        same = half == word
        result = {"status": "blocked" if same else "needs_review", "kind": "target_requirement",
                  "reason": ("width='half' and width='word' select identical memory fields; "
                             "the adapter cannot treat this as a 16-bit load with FP32 conversion"
                             if same else "half and word loads select distinct memory forms; "
                             "decoded addressing, conversion, and hardware behavior still require validation"),
                  "word": word, "half": half, "identical": same}
    elif name == "half_store":
        def selected(elem):
            f = ir.Function("half_store", [ir.Buffer("scores", 2, elem=elem)])
            b = ir.Builder(f, f.block("entry"))
            i = b.builtin("thread_position_in_grid")
            v = b.const(0x3c00, type=ir.I16)
            b.store_at(f.buffers[0], i, v)
            b.ret()
            ir.verify(f)
            return [{"form": m.form, "fields": m.fields} for m in cc.select(f) if m.form.startswith("store")]
        word, half = selected(ir.I32), selected("f16")
        same = half == word
        result = {"status": "blocked" if same else "needs_review", "kind": "target_requirement",
                  "reason": "half and word output buffers select identical stores; half conversion and two-byte output stride need an explicit lowering",
                  "word": word, "half": half, "identical": same}
    elif name == "input_bindings":
        result = {"status": "blocked", "kind": "target_requirement",
                  "reason": "two independent inputs need confirmed descriptor-to-load-base assignments",
                  "known_base_registers": cc.BUFFER_BASE_REG,
                  "missing_input_slots": [s for s in (0, 1) if s not in cc.BUFFER_BASE_REG]}
        if not result["missing_input_slots"]:
            result["status"] = "needs_review"
    elif name == "abi_handoff":
        import g17authorobj as author
        f = ir.Function("abi_control", [ir.Buffer("scores", 2)])
        b = ir.Builder(f, f.block("entry"))
        row = b.builtin("thread_position_in_grid")
        value = b.const(0)
        b.store_at(f.buffers[0], row, value)
        b.ret()
        # Semantic facts are produced from selected machine IR. This separate
        # control can expose ABI omissions even when scan encoding is blocked.
        layout = [(0, b"", m) for m in cc.select(f)]
        prog = cc.G17Program("abi_control", b"", layout, inherited={})
        abi = prog.abi_inputs()
        result = {"kind": "diagnostic", "scope": "single-store ABI control; not a scan image",
                  "provided_keys": sorted(abi), "abi": abi,
                  "missing_required_keys": sorted(set(("ld_md_slots", "ld_md_values", "arch_flag", "pk_slot1")) - set(abi))}
        try:
            author.author(text=b"", entry=64, bindings=[(2, 0, True)], abi=abi)
        except author.Missing as exc:
            result.update(status="blocked", reason=str(exc), error_type="Missing")
        else:
            result.update(status="needs_review", reason="author accepted the control; scan-specific inputs still need verification")
    elif name == "object_adapter_control":
        import g17link as L
        import g17authorobj as A
        import g17scanlink as adapter
        k = L.Kernel(bytes.fromhex("0e000000"),
                     [L.Binding(1, readonly=True), L.Binding(2, readonly=False)],
                     prologue=L.PROLOGUE_WORD + L.FILLER * 30)
        image = adapter.link(k, A._abi(), binding_offsets=[0, 2])
        result = {"status": "passed", "kind": "synthetic_control",
                  "scope": "end-only program with the author's explicit demo ABI; not a scan",
                  "abi_source": "g17authorobj._abi(), fixture only",
                  "object_bytes": len(image.object), "archive_bytes": len(image.archive),
                  "archive_sha256": image.sha256,
                  "verification": "finished archive and ordered binding records checked"}
    elif name == "guard_control":
        caught = []
        for label, action in (
            ("subprocess", lambda: subprocess.run([sys.executable, "-c", "pass"])),
            ("shader_cache", lambda: open(Path.home() / ".cache/agxforge/agx/scan-control", "rb")),
        ):
            try:
                action()
            except DependencyBlocked:
                caught.append(label)
        result = {"status": "passed" if len(caught) == 2 else "error", "caught": caught}
    else:
        raise ValueError("unknown probe " + name)
    result["dependency_events"] = events
    return result


PROBES = ("guard_control", "target", "half_load", "half_store", "input_bindings", "abi_handoff",
          "object_adapter_control", "f32_unrolled_1", "f32_unrolled_8", "f32_loop_8", "f32_loop_384")


def run_probe(name, timeout, spec=None):
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="g17scan-") as tmp:
        env = dict(os.environ, G17_CACHE_DIR=tmp, PYTHONDONTWRITEBYTECODE="1")
        # An ambient loop override must not change the reported readiness.
        env.pop("G17_ALLOW_LOOP", None)
        try:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", name]
            if spec is not None:
                command += ["--rows", str(spec.rows), "--columns", str(spec.columns)]
            p = subprocess.run(command,
                               env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "reason": f"probe exceeded {timeout:g}s; no capability verdict",
                    "seconds": round(time.monotonic() - start, 3)}
    try:
        result = json.loads(p.stdout)
    except json.JSONDecodeError:
        result = {"status": "error", "reason": "worker returned no structured result",
                  "stderr": p.stderr[-2000:], "stdout": p.stdout[-1000:]}
    result.update(seconds=round(time.monotonic() - start, 3), returncode=p.returncode)
    return result


def provenance():
    def git(*args):
        p = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    # Content hashes, not mtimes, tie this report to the actual compiler and ISA.
    files = (sorted((ROOT / "tools").glob("*.py"))
             + sorted((ROOT / "isa").glob("*"))
             + sorted((ROOT / "spike/accel/re").glob("*.py"))
             + [ROOT / "spike/serve/mix_mlx.py"])
    hashes = {}
    for path in files:
        if path.is_file():
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    aggregate = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"commit": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain"),
            "root": str(ROOT), "python": sys.version, "platform": platform.platform(),
            "architecture": os.environ.get("AGXFORGE_ARCH", "applegpu_g17s"),
            "input_digest": aggregate, "source_hashes": hashes,
            "cache_policy": "fresh private G17_CACHE_DIR per probe; external agxforge caches refused",
            "loop_override": "removed from each probe environment"}


def blockers(probes):
    rows = []
    for id_, owner, close in (
        ("target", "compiler + integration", "Lower the explicit half scan IR without replacing precision operations with word accesses."),
        ("half_load", "compiler", "Author actual half input access and FP32 conversion; verify element stride and both halfwords."),
        ("half_store", "compiler", "Convert the accumulated float to half and store at a two-byte row stride, preserving adjacent output elements."),
        ("input_bindings", "compiler + linker", "Agree on both input buffers' descriptor/base assignments and output store rank."),
        ("abi_handoff", "compiler + linker", "Supply measured per-program launch slots/values and pk_slot1 through the existing ABI."),
    ):
        r = probes[id_]
        if r["status"] != "passed":
            rows.append({"id": id_, "owner": owner, "status": r["status"],
                         "evidence": r.get("reason", "probe failed"), "close_when": close})
    grouped = {}
    for name in PROBES:
        if not name.startswith("f32_"):
            continue
        r = probes[name]
        if r["status"] != "passed":
            key = (r["status"], r.get("error_type"), r.get("reason"))
            if key in grouped:
                grouped[key]["affected_probes"].append(name)
                continue
            row = {"id": name, "owner": "compiler", "status": r["status"],
                         "affected_probes": [name],
                         "scope": "diagnostic implementation, not a universal scan requirement",
                         "evidence": r.get("reason", "probe failed"),
                         "close_when": "This exact diagnostic reaches encoding and selfcheck under the dependency guard."}
            grouped[key] = row
            rows.append(row)
    return rows


def render(report):
    p = report["provenance"]
    lines = ["# Native scan acceptance", "", f"Commit: `{p['commit']}`",
             f"Input digest: `{p['input_digest']}`", "",
             f"**Scan status: {report['status'].upper()}. No native scan image has been built or dispatched.**", "",
             "Target: contiguous half inputs, FP32 accumulation, half output; "
             f"{report['target']['rows']:,} rows × {report['target']['columns']} columns.", "",
             "The precision policy is proposed; comparison with MLX is pending. "
             "FP32 controls and synthetic ABI controls cannot satisfy the target.", "",
             "## Ordered handoffs", ""]
    for i, b in enumerate(report["blockers"], 1):
        lines += [f"{i}. **{b['id']}** — {b['owner']} ({b['status']})",
                  f"   Evidence: {b['evidence']}", f"   Closes when: {b['close_when']}", ""]
        if len(b.get("affected_probes", [])) > 1:
            lines += ["   Also blocks: " + ", ".join(b["affected_probes"][1:]) + ".", ""]
    lines += ["## Probe results", "", "| Probe | Status | Stage | Seconds |",
              "|---|---|---|---|"]
    for name, r in report["probes"].items():
        lines.append(f"| {name} | {r['status']} | {r.get('active_stage', 'semantic/ABI')} | {r['seconds']} |")
    lines += ["", "## Acceptance stages", ""]
    for stage, result in report["stages"].items():
        lines.append(f"- {stage}: {result['status']} — {result['reason']}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/g17-scan-acceptance")
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--columns", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=20,
                        help="seconds per compiler probe; timeouts are not unsupported verdicts")
    parser.add_argument("--worker", choices=PROBES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                result = worker(args.worker, Scan(args.rows, args.columns))
            result["environment"] = {"loop_override": os.environ.get("G17_ALLOW_LOOP"),
                                     "solver_cache": os.environ.get("G17_CACHE_DIR")}
        except Exception as exc:
            result = {"status": "error", "error_type": type(exc).__name__, "reason": str(exc)}
        if captured.getvalue():
            result["log"] = captured.getvalue()[-2000:]
        print(json.dumps(result))
        return 0
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    spec = Scan(args.rows, args.columns)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scan.json").write_text(json.dumps(spec.contract(), indent=2) + "\n")
    (args.out / "scan-reference.metal").write_text(spec.metal_reference())
    inputs = provenance()
    fixtures = write_fixtures(args.out, spec.columns)
    probes = {}
    for name in PROBES:
        print(f"{name} ...", flush=True)
        probes[name] = run_probe(name, args.timeout, spec)
        print(f"  {probes[name]['status']}: {probes[name].get('reason', 'checks completed')}", flush=True)
    if "requested_ir" in probes["target"]:
        (args.out / "scan-ir.txt").write_text(probes["target"]["requested_ir"] + "\n")
    report = {
        "schema_version": 1, "status": "blocked", "target": spec.contract(),
        "provenance": inputs, "fixtures": fixtures, "probes": probes,
        "blockers": blockers(probes),
        "stages": {
            "reference": {"status": "prepared", "reason": "deterministic finite half fixtures and independent exact-product reference"},
            "target_lowering": {"status": probes["target"]["status"], "reason": probes["target"].get("reason", "target worker failed")},
            "target_image": {"status": "blocked", "reason": "requires target lowering and complete compiler-owned ABI inputs"},
            "finished_image_verification": {"status": "not_run", "reason": "no target image exists"},
            "hardware_correctness": {"status": "not_run", "reason": "no dispatch performed"},
            "mlx_equivalence": {"status": "not_run", "reason": "output dtype, error and layout comparison remain unmeasured"},
            "agxforge_integration": {"status": "not_run", "reason": "no accepted native implementation to install"},
            "performance": {"status": "not_run", "reason": "requires correctness and an agreed performance criterion"},
        },
    }
    if any(p["status"] in ("error", "timeout") for p in probes.values()):
        report["status"] = "error"
    after = provenance()
    report["inputs_unchanged_during_run"] = inputs["input_digest"] == after["input_digest"]
    if not report["inputs_unchanged_during_run"]:
        report["status"] = "error"
        report["input_change"] = "compiler or ISA inputs changed during the run; rerun on a stable snapshot"
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render(report))
    print(f"\n{report['status'].upper()}: {len(report['blockers'])} handoffs; {args.out / 'report.md'}")
    return 1 if report["status"] == "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
