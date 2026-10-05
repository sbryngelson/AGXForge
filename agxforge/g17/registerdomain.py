"""Measured G17 register-domain invariants shared by compiler lowerings.

The hardware executes an instruction only when every encoded register operand is in
the implemented R0--R125 namespace.  R126 and above are silently squashed, so they
must be rejected before authoring.  FP32 tensor accumulators consume eight
consecutive registers and therefore have fifteen legal, eight-aligned groups.
"""

import re

# An MCRegister name is a single register (`R12`), a half (`R12H`), or an underscore-joined tuple
# (`R12L_R13H`, `R124_R125_R126_R127`).  Two things about this pattern are load-bearing:
#
#   The trailing `(?=_|$)` is a LOOKAHEAD and must not become a consuming `(?:_|$)`.  A consuming
#   form eats the separator that should anchor the next element, so it reads ALTERNATE elements of
#   a tuple -- `R125_R126` reads as ['125'] and `R124_R125_R126_R127` as ['124', '126'].  That is
#   how the earlier form failed to flag every operand tuple ending at R126 while correctly flagging
#   those ending at R127.
#
#   The leading `(?:^|_)` excludes the `IR*` register file, which is a distinct 16-register file
#   whose names contain `R<digits>`.  An unanchored `R(\d+)` reads `IR7` as GPR 7.
#
# `registers_in_name` is the single definition; callers must not re-spell it, and
# test_g17registerdomain pins its behaviour on synthesized names rather than on an emitted body.
NAME_RE = re.compile(r"(?:^|_)R(\d+)(?:[AHL]+)?(?=_|$)")


def registers_in_name(name):
    """Every GPR index named by an MCRegister name, including tuple elements and halves.

    Returns [] for a name that denotes no GPR: the `IR*` file, the system registers (`SR_CLUSTER`,
    `FLAG0`, `SP`, `LR`), and MCRegister 0, whose name is the empty string.
    """
    return [int(n) for n in NAME_RE.findall(name or "")]


# Anything SHAPED like a GPR token, whatever suffix follows it.  `NAME_RE` is the subset this
# module can actually read, so comparing the two counts asks "did the parse cover the whole name".
_TOKEN_RE = re.compile(r"(?:^|_)R\d+")


def unmodelled_name(name):
    r"""True if *name* carries a GPR-shaped token this module cannot read.

    Coverage, not mere non-emptiness: `R12_R13X` parses to [12] and would satisfy a "matched at
    least one" check while silently dropping R13X, which is the failure this exists to catch.

    No allowlist is needed for names that legitimately denote no GPR.  `(?:^|_)R\d+` does not match
    inside `IR7` either, so the `IR*` file yields zero tokens and zero parses; the same holds for
    `SR_*`, `FLAG*`, `SP`, `LR`, `SMP_BATON`, `CTLFLOWST` and MCRegister 0's empty name.  An
    allowlist derived with this same regex would also be circular -- every current name would pass
    by construction.
    """
    if not name:
        return False
    return len(NAME_RE.findall(name)) != len(_TOKEN_RE.findall(name))


def registers_in_name_checked(name):
    """`registers_in_name`, but refusing a name shape this module does not model.

    Use this wherever an empty result is indistinguishable from "no registers" and would fail
    OPEN -- an occupancy set that silently shrinks lets the allocator reuse a live register, and an
    empty violation list silently withholds a refusal.  Never let an unrecognized name default to
    zero registers: the fallback is uncounted and grows silently.
    """
    if unmodelled_name(name):
        raise ValueError("register name %r contains a register token this build does not model; "
                         "registerdomain.NAME_RE needs widening before it can be resolved" % (name,))
    return registers_in_name(name)


ALLOCATABLE_MIN = 0
ALLOCATABLE_MAX = 125
REGISTER_COUNT = ALLOCATABLE_MAX + 1
FP32_ACCUMULATOR_GROUP_WIDTH = 8
MAX_FP32_ACCUMULATOR_GROUPS = REGISTER_COUNT // FP32_ACCUMULATOR_GROUP_WIDTH


def invalid_registers(values):
    """Return sorted distinct register numbers outside the implemented namespace."""
    return sorted({int(value) for value in values if int(value) < ALLOCATABLE_MIN or int(value) > ALLOCATABLE_MAX})


def validate_accumulator_group(start):
    """Raise if *start* is not the first register of a legal FP32 accumulator group."""
    start = int(start)
    if start % FP32_ACCUMULATOR_GROUP_WIDTH:
        raise ValueError("FP32 accumulator group must start on an 8-register boundary (R%d)" % start)
    if start < ALLOCATABLE_MIN or start + FP32_ACCUMULATOR_GROUP_WIDTH - 1 > ALLOCATABLE_MAX:
        raise ValueError("FP32 accumulator group R%d..R%d exceeds the allocatable R0..R125 namespace" %
                         (start, start + FP32_ACCUMULATOR_GROUP_WIDTH - 1))

