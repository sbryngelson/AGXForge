"""Workflow 3: inspect the matched decode study (MM 25.211) and re-derive its numbers from the retained receipts.

    python3 examples/decode_study.py

READS RETAINED EVIDENCE ONLY. Nothing is compiled or dispatched. The study itself ran THROUGH METAL on one M5 Pro;
reproducing it needs the GPU, mlx-lm and the InternLM2.5-1.8B-chat weights (examples/README.md, workflow 3).

The question the study answers: would this project's decode design - its algorithms, fusions and execution
structure - run as fast if Apple's compiler built the kernels? Each kernel has a Metal twin (tools/twins/*.metal) with
the same threads, work partition and fp32 operation order, compiled by `xcrun metal -O3 -fno-fast-math
-ffp-contract=off`. Three arms decode mlx-lm's own 128-token greedy sequence after a 1,792-token prompt.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidence" / "g17-matched-study-v1"
ARMS = (("mlx", "mlx-lm generate_step"), ("graph_forced", "this project's kernels, this project's compiler"),
        ("graph_twin_all_forced", "the same kernels, Apple's compiler"))
SOURCES = {
    "graph configuration": ["tools/models/internlm2_q4_spec_code.json"],
    "kernel builders": ["tools/g17qmv.py", "tools/g17decodeops.py", "tools/g17attn.py", "tools/g17gen.py",
                        "tools/g17deliver.py"],
    "graph assembly and check": ["tools/g17q4graph.py", "tools/g17modelbuild.py"],
    "Metal twins": ["tools/twins/qmv_twin.metal", "tools/twins/attn_twin.metal", "tools/twins/misc_twin.metal"],
    "harness": ["tools/g17twin.py", "tools/g17twinrun.m", "tools/g17decodegen.m", "tools/g17model_mlx.py"],
    "reproduction helpers": ["tools/g17forcehistory.py", "tools/g17promptids.py"],
}


def main():
    shared = json.loads((EV / "decode-shared-history.json").read_text())
    kernels = json.loads((EV / "kernels.json").read_text())
    print("receipt:", (EV / "decode-shared-history.json").relative_to(ROOT))
    print("instrument:", shared["instrument"])
    print("shared history:", shared["shared_history"][:160], "...\n")

    by_arm = {}
    for row in shared["rows"]:
        by_arm.setdefault(row["arm"], {})[row["rep"]] = row["tok_s"]
    medians = {arm: statistics.median(reps.values()) for arm, reps in by_arm.items()}
    base = medians["mlx"]
    print("%-50s %10s %22s" % ("arm (all through Metal)", "median", "per-repetition / mlx-lm"))
    failures = []
    for arm, label in ARMS:
        reps = by_arm[arm]
        ratios = [reps[r] / by_arm["mlx"][r] for r in sorted(reps)]
        print("%-50s %10.1f %22s" % (label, medians[arm], "%.2f-%.2f" % (min(ratios), max(ratios))))
        if abs(round(medians[arm], 1) - round(shared["tok_s"][arm]["median"], 1)) > 1e-9:
            failures.append("median of %s does not match the receipt's summary" % arm)
    for arm, want in (("mlx", 162.2), ("graph_forced", 184.5), ("graph_twin_all_forced", 213.4)):
        if round(medians[arm], 1) != want:
            failures.append("%s median %.1f, published %.1f" % (arm, medians[arm], want))
    print("\ntoken agreement under the shared history:", shared["forcing_check"][:220], "...")
    print("disagreements with mlx-lm:", [d["index"] for d in shared["disagreements_vs_mlx_full_forward"]])

    print("\nper kernel (kernels.json): Apple-compiled twin time / ours, and whether the outputs are identical")
    for name, row in list(kernels["qmv"].items()) + [("attention q0=" + k, v) for k, v in kernels["attn"].items()]:
        identical = row.get("bytes_differ", row.get("out_bytes_differ")) == 0
        if not identical:
            failures.append("%s outputs differ" % name)
        print("  %-22s ours %7.1f us  twin %7.1f us  ratio %.2f  outputs %s"
              % (name, row["ours_us"], row["apple_us"], row["apple_over_ours"], "identical" if identical else "DIFFER"))

    print("\nwhat this establishes: the speed comes from the decode implementation, which survives Apple compilation;"
          "\nit does not establish any benefit from this project's native instruction control (Apple's compiler is"
          "\n10-15 percent faster on the same kernels). One model, one context, four repetitions; at a 196-token"
          "\ncontext this project is around parity with mlx-lm (MM 25.211).")
    print("\nsources:")
    for what, paths in SOURCES.items():
        missing = [p for p in paths if not (ROOT / p).is_file()]
        failures += ["missing source %s" % p for p in missing]
        print("  %-26s %s" % (what, ", ".join(paths)))
    rec = json.loads((EV / "inputs-recovery.json").read_text())
    print("\nreproduction:", rec["conclusion"])
    if failures:
        print("\nFAILED:", *failures, sep="\n  ")
        return 1
    print("\nall published numbers re-derived from the per-repetition rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
