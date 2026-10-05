"""Workflow 4: inspect bounded MiniLM and Qwen execution below Metal (MM 25.210) from its retained receipts.

    python3 examples/native_inference.py

READS RETAINED EVIDENCE ONLY. Nothing is compiled or dispatched. The runs themselves took place on one M5 Pro under
macOS 26.6.2 (build 25G83), with no Metal device or command buffer in the process: this project's runtime wrote the
resources and launch state and submitted through Apple's private IOGPU interface. Reproducing them dispatches GPU
work and needs the pinned checkpoints (examples/README.md, workflow 4).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidence"


def load(name):
    return json.loads((EV / name).read_text())


def main():
    failures = []
    print("PINNED INPUTS (evidence/g17-inference-models-v1)")
    for model in ("qwen", "minilm"):
        m = json.loads((EV / "g17-inference-models-v1" / model / "model.json").read_text())
        print("  %-38s revision %s  checkpoint %s bytes, sha256 %s..."
              % (m["repository"], m["revision"][:12], format(m.get("checkpoint_bytes", 0), ","),
                 m.get("expected_checkpoint_sha256", "")[:16]))

    print("\nQWEN2.5-0.5B-INSTRUCT, RECORDED BELOW METAL (g17-native-qwen-generation.json)")
    gen = load("g17-native-qwen-generation.json")
    for r in gen["requests"]:
        print("  %-32r -> %r" % (r["prompt"], r["text"]))
    if not any(r["text"] == "Hello! How can I assist you today?" for r in gen["requests"]):
        failures.append("the greeting is not the recorded one")
    chk = load("g17-native-qwen-generation-independent-check.json")
    checks = [c for r in chk["requests"] for c in r["checks"]]
    within = sum(1 for c in checks if c["passed"] and c["max_budget_fraction"] <= 1.0)
    top1 = sum(1 for c in checks if c["native_top1"] == c["framework_top1"])
    print("  independent check (no GPU): %d of %d logit vectors within 0.05 + 0.003*|reference| of the original"
          " checkpoint in FP64; top-1 agrees on %d; worst budget fraction %.2f"
          % (within, len(checks), top1, max(c["max_budget_fraction"] for c in checks)))
    print("  scope:", chk["scope"])
    if (within, len(checks), top1) != (37, 37, 37):
        failures.append("Qwen independent check is not 37 of 37")

    print("\nMINILM-L6, RECORDED BELOW METAL (g17-native-encoder-guarded-retrieval.json)")
    ret = load("g17-native-encoder-guarded-retrieval.json")
    print("  query %r" % ret["requests"][0]["text"])
    for r in ret["ranking"]:
        print("    %.4f  %r" % (r["cosine_similarity"], r["text"]))
    ind = load("g17-native-encoder-guarded-retrieval-independent.json")
    print("  independent check (no GPU): passed=%s, ranking exact=%s, repeat exact=%s"
          % (ind["passed"], ind["ranking_exact"], ind["repeat_exact"]))
    if not (ind["passed"] and ret["ranking"][0]["text"].startswith("A puppy")):
        failures.append("MiniLM receipt does not show the recorded ranking")

    print("\nCONTROLS THAT FAIL, KEPT AS FAILURES")
    for name in ("g17-native-qwen-framework-fp32-control.json", "g17-native-qwen-incomplete-prefix-quality-failure.json",
                 "g17-native-qwen-decode-pack-failure.json"):
        x = load(name)
        print("  %-52s passed=%s  %s" % (name, x.get("passed"), str(x.get("scope", ""))[:70]))

    plat = load("g17-native-callback-ownership.json")
    print("\nMEASURED PLATFORM: macOS %s (build %s), %s; one M5 Pro (H17s)" % (plat["os_version"], plat["os_build"],
                                                                        plat["machine"]))
    print("APPLE COMPONENTS THAT REMAIN: the IOGPU framework (private), the AGX kernel driver and the GPU firmware;"
          "\n  compilation and validation use Apple's G17 decoder (GPUCompiler.framework). No Metal in the process.")
    print("NOT ESTABLISHED: a speed benefit (native is slower than matched Metal and ~7-15x slower than Transformers"
          "\n  on MPS), arbitrary models, or any other OS build: the runtime's launch layouts were measured on 25G83.")
    if failures:
        print("\nFAILED:", *failures, sep="\n  ")
        return 1
    print("\nthe published outputs and checks are re-derived from the receipts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
