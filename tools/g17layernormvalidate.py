"""Validate every native LayerNorm output using owned inputs and fixed FP64 error bounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

import g17commonimage as image
import g17commonruntime as runtime
import g17commonstage as stage
import g17layernorm

ERROR_BUDGET = 2e-5


def cases(source):
    """The full acceptance campaign retains the compiled shape for every request."""
    rows, columns = source.shape
    rng = np.random.default_rng(734)
    yield "real", source.copy()
    yield "negated", -source
    yield "repeated_real", source.copy()
    yield "zeros", np.zeros_like(source)
    yield "constant_rows", np.resize(np.array([-7,0,7],np.float32),rows)[:,None]*np.ones((1,columns),np.float32)
    yield "near_zero", (rng.standard_normal(source.shape)*1e-7).astype(np.float32)
    yield "large_row_offset", (1e6+rng.standard_normal(source.shape)).astype(np.float32)


def compare(result, source, gamma, beta):
    """All elements, fixed mixed absolute/relative budget, no fitted tolerance."""
    if not isinstance(result,np.ndarray) or result.dtype != np.float32 or result.shape != source.shape:
        raise ValueError("LayerNorm result must contain the complete FP32 matrix")
    expected = g17layernorm.reference(source,gamma,beta)
    if not np.isfinite(expected).all():
        raise ValueError("LayerNorm reference is outside the finite domain")
    delta = np.abs(result.astype(np.float64)-expected)
    budget = ERROR_BUDGET*(1+np.abs(expected))
    failed = ~np.isfinite(result) | (delta > budget)
    report = dict(ok=not bool(failed.any()), outputs_checked=int(result.size),
        failures=int(failed.sum()), first_bad_coordinates=np.argwhere(failed)[:8].tolist(),
        max_abs_error=float(delta.max()) if np.isfinite(delta).all() else None,
        error_budget="2e-5*(1+abs(reference))")
    return report, expected


def validate(bundle, source, gamma, beta, *, approved=False, single=False):
    if not approved:
        raise ValueError("dispatch stage requires explicit selection")
    bundle = Path(bundle)
    contract = runtime.ImageContract.read(json.loads((bundle/"manifest.json").read_text()))
    if contract.kind != "layernorm":
        raise ValueError("this campaign requires a LayerNorm image")
    previous = {}
    for receipt in bundle.glob("layernorm-validation-*.json"):
        record = json.loads(receipt.read_text())
        if record.get("status") != "passed":
            raise ValueError(f"prior LayerNorm validation failed or is incomplete: {receipt.name}")
        previous[receipt.name] = record
    if not single:
        first = previous.get("layernorm-validation-1.json", {})
        if first.get("sha256") != contract.sha256 or first.get("hardware_success") is not True:
            raise ValueError("a successful single-query stage is required for this exact image")
    count = 1 if single else 7
    # Preserve failed/pending evidence. Calling the command again cannot retry it.
    path = bundle/f"layernorm-validation-{count}.json"
    report = dict(status="pending", kind="layernorm", shape=contract.shape.model_dump(),
        sha256=contract.sha256, queries_requested=count, attempts=[], gpu_dispatched=False,
        hardware_success=False, epsilon=g17layernorm.EPSILON,
        error_budget="2e-5*(1+abs(reference))")
    start = time.monotonic()
    with path.open("x") as evidence:
        try:
            with stage.NativeProgram(source,bundle,parameters=(gamma,beta),approved=True,queries=count) as native:
                report.update(identity=native.identity,rebuild=native.rebuild,
                    loader_receipt_sha256=stage.digest(Path(native.directory.name)/stage.RECEIPT),
                    input_sha256={k:image.sha(v) for k,v in native.input_snapshots.items()})
                shape = (contract.shape.rows,contract.shape.columns)
                owned = np.frombuffer(native.input_snapshots["source"],"<f4").reshape(shape)
                g,b = (np.frombuffer(native.input_snapshots[k],"<f4") for k in ("gamma","beta"))
                with (bundle/f"layernorm-parameters-{count}.npz").open("xb") as out:
                    np.savez_compressed(out,gamma=g,beta=b)
                first = None
                for sequence,(name,request) in enumerate(cases(owned),1):
                    if sequence > count:
                        break
                    attempt = dict(sequence=sequence,case=name,status="pending",submitted=False,
                                   submission_may_have_occurred=True)
                    report["attempts"].append(attempt)
                    if report["gpu_dispatched"] is not True:
                        report["gpu_dispatched"] = None
                    result = native(request)
                    report["gpu_dispatched"] = True
                    attempt.update(submitted=True,worker=native.last_report)
                    sent = np.frombuffer(native.last_request_bytes,"<f4").reshape(shape)
                    checked, expected = compare(result,sent,g,b)
                    attempt.update(reference=checked,request_sha256=image.sha(native.last_request_bytes),
                                   result_sha256=image.sha(result.tobytes()))
                    output = bundle/f"layernorm-{count}-{sequence}-{name}.npz"
                    with output.open("xb") as out:
                        np.savez_compressed(out,source=sent,result=result,reference=expected)
                    attempt.update(output=output.name,output_sha256=stage.digest(output))
                    if not checked["ok"]:
                        raise RuntimeError(f"LayerNorm query {sequence} differs from the independent reference")
                    if sequence == 1:
                        first = result.tobytes()
                    elif sequence == 3:
                        same = first == result.tobytes()
                        attempt["repeated_output_bit_identical"] = same
                        if not same:
                            raise RuntimeError("repeated LayerNorm input returned different FP32 bits")
                    attempt["status"] = "passed"
                if native.calls != count:
                    raise RuntimeError("LayerNorm campaign did not execute every requested query")
                report.update(status="passed",hardware_success=True,queries_completed=native.calls,
                              guards_passed=True,readonly_inputs_passed=True)
        except BaseException as error:
            report.update(status="failed",error=f"{type(error).__name__}: {error}")
            if report["attempts"] and report["attempts"][-1]["status"] == "pending":
                report["attempts"][-1]["status"] = "failed"
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic()-start
            evidence.write(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle",type=Path)
    parser.add_argument("--fixture",type=Path,required=True,help="NPZ with FP32 source, gamma and beta")
    parser.add_argument("--load-approved",action="store_true")
    parser.add_argument("--dispatch-approved",action="store_true")
    parser.add_argument("--single",action="store_true",help="one query before the seven-query campaign")
    args = parser.parse_args()
    reports = {}
    with np.load(args.fixture,allow_pickle=False) as data:
        source,gamma,beta = (data[k].copy() for k in ("source","gamma","beta"))
    if args.load_approved:
        reports["loader"] = stage.load_only(args.bundle,approved=True)
    if args.dispatch_approved:
        reports["validation"] = validate(args.bundle,source,gamma,beta,approved=True,single=args.single)
    if not reports:
        raise ValueError("select a loader or dispatch stage")
    print(json.dumps(reports,indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError,ValueError,RuntimeError) as error:
        import sys
        print(f"refused: {error}",file=sys.stderr)
        raise SystemExit(2)
