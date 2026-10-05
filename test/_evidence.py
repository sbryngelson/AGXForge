"""Skip, rather than error, when a test's gitignored evidence is absent - and say what is lost.

T8: "Cold-checkout tests need committed compact fixtures, extracted evidence, or explicit skips that
state the lost invariant."

Measured 2026-09-21: **283 of 476** `test_g17*.py` files reference `results/`, and only 21 guard it.
With `results/` moved aside, **35 of 35** sampled unguarded suites errored - none passed, and none
skipped. `make test` depends on the `evidence` target so the gate is safe; the exposure is a bare
`unittest discover`, or running one file directly, on a fresh clone. That produces hundreds of
tracebacks whose single cause is one unextracted archive, and the prior-art lane lost most of a day
to exactly that: ten suites reported as broken, all green once the evidence was extracted.

The repair T8 asks for is not "make them pass". It is to make the absence SAY what went unverified,
because a skip that reads like a pass is worse than the error it replaces. So `require` names the
invariant in its own skip message:

    from _evidence import require

    def setUp(self):
        require("results/g17-tensor-common-witness-v1/tensor-common.o",
                invariant="the witnessed shape is untouched by the route")

`test_g17evidencedependency.py` ratchets the unguarded count so it can shrink and not grow.
"""
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT = "python3 tools/g17evidence.py extract"


def paths(*rel):
    return [p if os.path.isabs(p) else os.path.join(ROOT, p) for p in rel]


def missing(*rel):
    return [r for r, p in zip(rel, paths(*rel)) if not os.path.exists(p)]


def require(*rel, invariant):
    """Skip with the lost invariant named, or return the absolute paths.

    `invariant` is required and positional-only by keyword on purpose: a skip that says "file not
    found" tells a reader nothing about what stopped being checked, which is the whole complaint
    T8 records.
    """
    if not invariant or not str(invariant).strip():
        raise ValueError("require() needs an invariant: say what goes unverified without the file")
    gone = missing(*rel)
    if gone:
        raise unittest.SkipTest(
            "evidence absent (%s): %r is UNVERIFIED here. These paths are gitignored, so a fresh "
            "clone has none of them; run `%s` to restore, or read this skip as 'not checked' "
            "rather than 'checked and fine'." % (", ".join(gone), invariant, EXTRACT))
    return paths(*rel)


def evidence_present():
    """True when the extracted archive is here at all. For module-level guards."""
    return os.path.isdir(os.path.join(ROOT, "results"))
