"""Workflow 2: one tensor coordinate rule, a CPU calculation, and the rival reading the hardware rejected.

    python3 examples/tensor_feed_rule.py

READS RETAINED EVIDENCE AND CALCULATES ON THE CPU. Nothing is compiled or dispatched. Part 2 is a SIMULATION: the
package's arithmetic model of the MMA, run on the CPU, compared with an output the GPU recorded on 2026-09-24.

THE RULE (technical reference 10.5, MM 25.103). A tensor MMA leaves its 16 x 16 fp32 result in 32 lanes x 8 registers
("slots"). The next MMA, fed those registers as its B operand, reads lane l, slot j as canonical B row
k = 4*(l>>4) + ((l>>1)&3) + 8*(j>>2). That register holds canonical D row rotl1(k), a one-bit rotation of the 4-bit
row index. So whether the consumer sees the logical operand depends on how the PRODUCER labelled its rows:

  - canonical packing (Apple-compiled simdgroup_matrix code): B row k receives D row rotl1(k); the operand arrives
    row-rotated (the relabelling the recon's section 132 measured on Apple's kernels);
  - this compiler's B-row packing: the producer puts application row k where B row k is read, so the consumer reads
    D itself, with no shuffle.

THE EVIDENCE. results/g17-tensor-feedmodes-v1/feed_B_half is a retained run of this compiler's program (code SHA-256
320ac87a...) that computes D = A.B and feeds D, narrowed to half, as the next MMA's B operand (M = N = 32, K = 64, one
SIMD group). Its preregistered reference applied section 132's rotation and FAILED; the logical reading was the rival.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

RUN = ROOT / "results" / "g17-tensor-feedmodes-v1" / "feed_B_half"
RECEIPT = ROOT / "results" / "g17-tensor-feedmodes-v1" / "dispatch-receipt.json"


def rotl1(k):
    return ((k << 1) | (k >> 3)) & 15


def part1():
    """Pure arithmetic: lane 0's eight slots under the two packings."""
    from agxforge.g17 import ir
    print("PART 1 - calculation (no evidence): lane 0, the accumulator fed as the next MMA's B operand")
    print("slot  canonical D  read as B   D row under canonical packing   D row under this compiler's packing")
    lane = 0
    for j in range(8):
        k = 4 * (lane >> 4) + ((lane >> 1) & 3) + 8 * (j >> 2)
        col = (lane & 8) + 4 * (lane & 1) + (j & 3)
        canonical_d = (rotl1(k), col)
        ours = ir.tensor_acc_position(lane, j)          # the compiler's own (row, col) for this lane and slot
        print("  %d     %-11s  %-9s  %-31d  %d" % (j, canonical_d, (k, col), canonical_d[0], ours[0]))
    print("slot 4: B row 8 reads a register that holds canonical D row rotl1(8) = %d; this compiler stored"
          " application row 8 there\n" % rotl1(8))


def part2():
    print("PART 2 - SIMULATION on the CPU against a retained GPU output (MM 25.103)")
    if not (RUN / "generic.json").is_file():
        raise SystemExit("missing %s; in the research checkout run: python3 tools/g17evidence.py extract "
                         "results/g17-tensor-feedmodes-v1" % RUN.relative_to(ROOT))
    import g17tensorcommonruntime as R          # the arithmetic model; importing it dispatches nothing
    spec = json.loads((RUN / "generic.json").read_text())
    recorded = np.load(RUN / "mismatch-q1.npz")["got"]          # the GPU's output, recorded at dispatch
    print("retained run: M=%d N=%d K=%d, stage %s, %d output elements recorded on the GPU"
          % (spec["M"], spec["N"], spec["K"], spec["stages"], recorded.size))
    verdicts = {}
    for reading, label in (("identity", "logical operand (this compiler's packing)"),
                           ("measured", "row-rotated operand (section 132's relabelling)")):
        s = dict(spec, feed_model=reading)
        predicted = R._generic_chain_reference(RUN, s)
        differ = int((predicted.view("<u4") != recorded.view("<u4")).sum())
        worst = float(np.abs(predicted.astype(np.float64) - recorded.astype(np.float64)).max())
        verdicts[reading] = differ
        print("  %-48s differs from the GPU in %4d of %d elements, max |error| %.4g"
              % (label, differ, recorded.size, worst))
    receipt = json.loads(RECEIPT.read_text())
    print("dispatch receipt: feed_B_half (rotated reference) %s; neg_B_identity (same program, logical reference) %s"
          % (receipt["feed_B_half"]["status"], receipt["neg_B_identity"]["status"]))
    ok = verdicts["identity"] == 0 and verdicts["measured"] > 0
    print("verdict: %s" % ("the hardware output is the logical product bit for bit; the rotated reading is rejected"
                           if ok else "UNEXPECTED - the retained evidence does not separate the readings"))
    print("scope: one square 32 x 32 stage, one SIMD group, half-narrowed feed; no timing claim")
    return 0 if ok else 1


if __name__ == "__main__":
    part1()
    sys.exit(part2())
