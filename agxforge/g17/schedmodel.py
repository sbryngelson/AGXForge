"""The G17 scalar timing model: latency and issue cost per op, by measurement and by scheduling class.

Units are ISSUE INTERVALS - the time one simdgroup takes to issue one simple ALU instruction when
nothing stalls it, measured at 3.1 ns on this machine (isa/g17-latency.json loop.interpretation).
Nanoseconds are not used because the clock is not measured; intervals are what a scheduler compares.

WHERE THE NUMBERS COME FROM. tools/g17latency.py --loop: a body of B ops inside a 128-trip counted
loop, so the body stays in the instruction cache, timed at B = 8, 16, 32, 64 in one simdgroup, as one
dependent chain and as eight independent chains. Every fit is linear to within 0.7 us.

    simple   latency 2, issue 1    iadd, iadd-imm, xor-imm, icmp, fadd, fmul, ffma, fmax
    slow     latency 5, issue 2    imul, shl-imm            (measured 4.8 and 1.96)

THE WAIT-BIT MODEL, from the same day's evidence. An ALU result is INTERLOCKED: all 52 adjacent
ordered pairs of those ten ops read the previous result correctly with nothing between them
(ledger/g17-alu-pairs-read-the-previous-result.toml), so an ALU dependence costs time and never
correctness. A LATE value is not interlocked: a device or threadgroup load, an imageblock read, a
vector load's lanes and a special-register read are read early unless the consumer waits - byte0[3]
on the forms that carry it, an intervening waiting copy on the forms that do not (cc.LATE_KINDS,
_wait_for_load, _materialise_sr). So the scheduler below never needs to insert a wait for an ALU
edge, and never moves an instruction across a late value's producer and its waiting consumer.

BY SCHEDULING CLASS. The decoder's schedclass separates the two measured classes exactly: 5, 10,
56, 87 and 279 are all simple; 7 and 312 are both slow. PREDICTED_BY_CLASS extends that to
opcodes of the same classes that were not timed, and PREDICTIONS names the ones committed before
they were measured (the preregistration: msb, reverse and sar are class 7, addsat/subsat class 5).
An unmeasured class gets UNKNOWN, which is deliberately pessimistic.
"""

# (latency intervals, issue intervals), rounded from isa/g17-latency.json loop.interpretation
MEASURED = {
    10282: (2, 1), 10279: (2, 1), 17770: (2, 1), 11372: (2, 1),     # iadd, iadd-imm, xor-imm, icmp
    998: (2, 1), 3290: (2, 1), 2190: (2, 1), 9700: (2, 1),           # fadd, fmul, ffma, fmax
    10825: (5, 2), 14391: (5, 2),                                    # imul, shl-imm
    # timed 2026-09-23 after their predictions were committed: msb, reverse, sar (class 7) slow and
    # addsat, subsat (class 5) simple, as predicted; the class-73 transcendentals EQUAL to one another
    # (14.88 / 6.03 ns each), as predicted, and in the slow class; rint (144) slow; fsat (op904, class
    # 51) a THIRD class - latency 3.3 intervals, full-rate issue: an add with a saturation stage
    9986: (5, 2), 14047: (5, 2), 16805: (5, 2), 10239: (2, 1), 11624: (2, 1),
    3658: (5, 2), 3850: (5, 2), 1272: (5, 2), 2570: (5, 2), 3770: (5, 2), 904: (3, 1),
}
SIMPLE, SLOW = (2, 1), (5, 2)
PREDICTED_BY_CLASS = {5: SIMPLE, 10: SIMPLE, 56: SIMPLE, 87: SIMPLE, 279: SIMPLE, 7: SLOW, 312: SLOW,
                      73: SLOW, 144: SLOW, 51: (3, 1)}
UNKNOWN = (6, 2)
# committed 2026-09-23 BEFORE these were timed; tools/g17latency.py scores them
PREDICTIONS = {9986: ("msb", 7, SLOW), 14047: ("reverse", 7, SLOW), 16805: ("sar", 7, SLOW),
               10239: ("addsat", 5, SIMPLE), 11624: ("subsat", 5, SIMPLE)}


# A STRUCTURAL PREDICTION, committed 2026-09-23 before these were timed: ops sharing an untimed
# scheduling class cost the same. recip, rsqrt, exp2 and log2 are all class 73; the prediction holds
# when their (latency, issue), rounded to intervals, are equal - whatever the value is.
PREDICTED_SAME_CLASS = {73: [3658, 3850, 1272, 2570]}


def cost(opcode, schedclass=None):
    """(latency, issue) in intervals: measured, else by class, else UNKNOWN."""
    if opcode in MEASURED:
        return MEASURED[opcode]
    if schedclass is None:
        try:
            from agxforge.g17 import opclass
            schedclass = opclass.instrs().get(opcode, {}).get("schedclass")
        except Exception:
            schedclass = None
    return PREDICTED_BY_CLASS.get(schedclass, UNKNOWN)
