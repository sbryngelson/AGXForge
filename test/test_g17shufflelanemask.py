"""Operand 4 of the shuffle family is bounded by the LANE GROUP WIDTH, recomputed from the corpus.

A peer lane reported a correlation: in the native code of 9 objects, op14169's final immediate was
1 or 8 in every row reduction and 2, 4 or 16 in every column reduction, matching the lane XOR masks
of the measured summation trees. They declined to promote it, correctly - it is two kernel families
agreeing, and the immediate was never varied.

This asks the same question of a population where a false positive is possible, and on an axis the
kernel families cannot supply: the WIDTH OF THE LANE GROUP. A quad is four lanes, so an XOR mask
inside one cannot exceed 2; a simdgroup is thirty-two, so it cannot exceed 16. The field is eight
bits wide, reaches 255, and shares no bit with the opcode, so nothing in the ENCODING stops either
opcode from carrying any value at all. If the values Apple emits respect each opcode's own group
width, that is the field's meaning showing through, and it is 122 instances rather than two
families.

WHAT THIS IS NOT. It is corpus evidence about an encoding, not hardware semantics. It cannot
separate `lane XOR mask` from `lane + delta`, because every value observed is a power of two and
the compiler had no reason to emit anything else. Settling that needs a dispatch whose observable
is a NAMED LANE's output, and the oracle in this repository stores its result at a slot indexed by
the CASE, not the thread - so with 32 lanes racing to one slot, and every permutation hypothesis
leaving the multiset of lane values unchanged, the observation carries no information about which
lane was read. That is recorded as a named unknown, not as a weak claim.
"""
import collections
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agxforge.g17 import auth

# THE EXPRESSIBLE bound for a group of N lanes is N-1, not N/2: in a 4-lane quad, mask 3 pairs
# lane 0 with lane 3, which is inside the group. An earlier version of this file asserted N/2 and
# justified it as "anything larger addresses a lane the group does not contain", which is FALSE
# for mask 3 in a quad - a true observation resting on a wrong reason. The recon lane's
# thread-indexed probe measured mask 3 working in-quad, which is what refuted it.
#
# So the two facts are separated below. `ADDRESSABLE` is what the lane group can express and is a
# hard bound: an instance above it would be addressing a lane that does not exist. `N/2` is where
# the CORPUS stops, which is a fact about the compiler - a reduction tree exchanges over one BIT
# of the lane index at a time, so it emits 1, 2, 4, ... up to N/2 and never needs N-1.
GROUP = {'quad': 4, 'simd': 32}
ADDRESSABLE = {name: lanes - 1 for name, lanes in GROUP.items()}


def _corpus_instances():
    out = collections.defaultdict(list)
    path = os.path.join(ROOT, 'isa', 'g17-corpus-programs.jsonl')
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            text = bytes.fromhex(row['text'])
            for offset, length, opcode in row['spans']:
                if offset + length <= len(text):
                    out[opcode].append(text[offset:offset + length])
    return out


def _named():
    out = {}
    path = os.path.join(ROOT, 'isa', 'g17-contract.jsonl')
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            out[row['opcode']] = row.get('name') or ''
    return out


class TheLaneParameterIsBoundedByItsGroupWidth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = _corpus_instances()
        cls.names = _named()
        cls.xor = {op: n for op, n in cls.names.items()
                   if n.endswith('.shuffle_xor') and n.split('.')[0] in GROUP}

    def _operand4(self, opcode):
        return collections.Counter(auth.decode(opcode, b).get(4) for b in self.instances[opcode])

    def test_both_group_widths_are_present_so_the_comparison_can_fail(self):
        """A one-width population could not distinguish a bound from a coincidence."""
        widths = {self.names[op].split('.')[0] for op in self.xor}
        self.assertEqual(widths, {'quad', 'simd'},
                         'only %s is represented, so no width comparison is possible' % widths)

    def test_the_field_could_hold_a_value_that_would_refute_the_bound(self):
        """If the encoding pinned the field, respecting the bound would prove nothing."""
        for op in sorted(self.xor):
            with self.subTest(op=op, name=self.xor[op]):
                self.assertGreaterEqual(auth.reach(op, 4), 255)
                bits = {(by, bi) for _, by, bi, _ in auth.fields(op)[4][1]}
                path = os.path.join(ROOT, 'isa', 'g17-contract.jsonl')
                with open(path) as fh:
                    row = [json.loads(l) for l in fh]
                row = [r for r in row if r['opcode'] == op][0]
                opcode_bits = {(by, bi) for by, bi, _ in row['encoding']['opcode_bits']}
                self.assertEqual(bits & opcode_bits, set(),
                                 'operand 4 of op%d shares bits with the opcode, so its range is '
                                 'not a free choice' % op)

    def test_no_instance_exceeds_the_mask_its_own_group_can_ADDRESS(self):
        """The hard bound: a mask above N-1 would name a lane the group does not contain."""
        for op in sorted(self.xor):
            name = self.xor[op]
            lanes = GROUP[name.split('.')[0]]
            with self.subTest(op=op, name=name, lanes=lanes):
                seen = self._operand4(op)
                self.assertTrue(seen, 'op%d has no corpus instance' % op)
                over = {v: n for v, n in seen.items()
                        if v is None or v > ADDRESSABLE[name.split('.')[0]]}
                self.assertEqual(over, {},
                                 'op%d (%s, %d lanes) carries operand 4 = %s, above the largest '
                                 'lane a %d-lane group contains'
                                 % (op, name, lanes, sorted(over), lanes))

    def test_the_corpus_stops_at_half_the_group_which_is_a_COMPILER_fact(self):
        """Separate from the bound above, and held to a weaker claim on purpose.

        Every corpus value is a power of two no greater than N/2, which is what a reduction tree
        emits: it exchanges over one bit of the lane index per stage. It is NOT a limit of the
        instruction - the recon lane measured mask 3 working inside a quad - so this asserts a
        fact about what Apple's compiler emits and says so, rather than dressing it up as an
        encoding constraint.
        """
        for op in sorted(self.xor):
            name = self.xor[op]
            lanes = GROUP[name.split('.')[0]]
            with self.subTest(op=op, name=name):
                seen = self._operand4(op)
                over = {v for v in seen if v is None or v > lanes // 2}
                self.assertEqual(over, set(),
                                 'op%d emits operand 4 = %s, above N/2. That is not a violation '
                                 'of anything the hardware enforces - it means the compiler emits '
                                 'a mask no reduction tree needs, and this claim must be narrowed '
                                 'rather than asserted' % (op, sorted(over)))

    def test_every_value_is_a_power_of_two(self):
        """A reduction tree exchanges over one bit of the lane index at a time."""
        for op in sorted(self.xor):
            with self.subTest(op=op, name=self.xor[op]):
                bad = [v for v in self._operand4(op) if v == 0 or (v & (v - 1))]
                self.assertEqual(bad, [], 'op%d carries non-power-of-two mask(s) %s' % (op, bad))

    def test_the_opcodes_whose_name_ends_in_one_carry_exactly_one_IN_THE_CORPUS(self):
        """A fact about what Apple EMITS, and NOT evidence that the name pins the field.

        This case used to be titled "Apple's own naming pins the field's value, which is evidence
        independent of the range", and that inference is RETRACTED. A compiled
        `quad_shuffle_down(v, 2u)` emits op14022 - whose Apple name is `quad.shuffle_down1` - with
        operand 4 = 2, and the lane probe measures it shifting by two. `simd.rotate_up1_16` takes
        a 3 the same way. The corpus never varied the immediate on these opcodes, which is a fact
        about the compiler's choices; reading it as a property of the encoding was
        [[the-vendors-choice-is-not-the-requirement]] one more time.

        The corpus observation is kept because it is true and useful - it is what made operand 4
        identifiable as the lane parameter in the first place - and the test name now says which
        population it is about.
        """
        suffixed = {op: n for op, n in self.names.items()
                    if n.endswith('1') and '.shuffle' in n and self.instances.get(op)}
        self.assertGreaterEqual(len(suffixed), 3,
                                'too few `*1` shuffle opcodes in the corpus to make the point')
        for op in sorted(suffixed):
            with self.subTest(op=op, name=suffixed[op]):
                seen = self._operand4(op)
                self.assertEqual(set(seen), {1},
                                 'op%d is named %s but carries operand 4 = %s'
                                 % (op, suffixed[op], dict(seen)))

    def test_the_same_operand_is_unbounded_where_the_name_is_not_an_xor(self):
        """simd.shuffle_up uses the same position over the full lane range, so the bound above is
        a fact about the XOR forms and not about the field being small."""
        up = [op for op, n in self.names.items() if n == 'simd.shuffle_up' and self.instances.get(op)]
        self.assertTrue(up, 'simd.shuffle_up is absent from the corpus')
        seen = set()
        for op in up:
            seen |= set(self._operand4(op))
        self.assertGreater(max(seen), 16,
                           'simd.shuffle_up never exceeds 16 either, so the observed ceiling may '
                           'be a property of the field rather than of the XOR forms')
        self.assertTrue({v for v in seen if v and (v & (v - 1))},
                        'simd.shuffle_up carries only powers of two, so "powers of two" is not '
                        'specific to the XOR forms')
class ALaneBoundFormMustNotBeToldToVaryLanes(unittest.TestCase):
    """A cross-lane form's next step must name a THREAD-INDEXED destination, not lane variation.

    The audit's generic advice for a degenerate probe is "re-measure with a second base, lane
    variation, or a case set the absorbing function cannot pass". For a shuffle or a prefix that is
    wrong, and wrong in a way that cost a request to another lane: this oracle stores its result at
    a slot indexed by the CASE, not the thread, so only lane zero's store lands. Lane zero is
    exactly where a shuffle offset and an XOR mask agree (0^k == 0+k) and where a prefix reduction
    returns its own input, so adding lane variation to such a probe changes nothing about what can
    be read. The observable that is missing is a per-lane destination, and plain Metal source with
    `C[lane] = r` has one.
    """

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, 'isa', 'g17-coverage.json')) as fh:
            cls.audit = json.load(fh)['untouched_form_audit']['forms']

    def test_lane_bound_forms_are_present_so_this_is_not_vacuous(self):
        lane = [f for f, e in self.audit.items()
                if any(w in (e.get('apple_name') or '') for w in ('shuffle', 'prefix', 'rotate'))]
        self.assertGreaterEqual(len(lane), 10,
                                'too few cross-lane forms in the untouched audit for this guard '
                                'to be checking anything')

    def test_no_cross_lane_form_is_advised_to_vary_lanes_through_this_oracle(self):
        for form, entry in self.audit.items():
            apple = entry.get('apple_name') or ''
            if not any(w in apple for w in ('shuffle', 'prefix', 'rotate', 'broadcast')):
                continue
            step = entry.get('next_step') or ''
            if 'lane variation' in step:
                self.assertIn('THREAD-INDEXED', step,
                              '%s (%s) is told to vary lanes, but only lane zero is readable '
                              'through this oracle and lane zero cannot see a permutation'
                              % (form, apple))

    def test_the_forms_whose_degeneracy_is_lane_bound_say_what_reads_them(self):
        lane_bound = [(f, e) for f, e in self.audit.items()
                      if 'THREAD-INDEXED' in (e.get('next_step') or '')]
        self.assertTrue(lane_bound,
                        'no form names a thread-indexed destination as its next step, so either '
                        'the cross-lane population left the audit or the advice regressed')
        for form, entry in lane_bound:
            self.assertTrue(any(w in (entry.get('apple_name') or '')
                                for w in ('shuffle', 'prefix', 'rotate', 'broadcast')),
                            '%s is advised to use a thread-indexed destination but is not a '
                            'cross-lane form' % form)

    def test_the_name_does_NOT_pin_the_immediate(self):
        """The counterexample, asserted so the retracted inference cannot quietly return."""
        path = os.path.join(ROOT, 'isa', 'g17-lane-semantics.json')
        if not os.path.exists(path):
            self.skipTest('the lane probe has not run')
        with open(path) as fh:
            probes = json.load(fh)['probes']
        # `rstrip('_16')` was the first spelling and it strips CHARACTERS, not a suffix - so it
        # removed the very `1` the filter was looking for and the counterexample came back empty.
        suffixed = ('down1', 'up1', 'xor1')
        counter = [p for p in probes
                   if p.get('status') == 'ok'
                   and any(tag in str(p.get('apple_name') or '') for tag in suffixed)
                   and int((p.get('operands') or {}).get('4', 1)) != 1]
        self.assertTrue(counter,
                        'no compiled probe emits a `*1`-named opcode with an immediate other '
                        'than 1, so the retraction above has lost its evidence - either the '
                        'probe set changed or the claim needs re-examining')
        for probe in counter:
            self.assertNotEqual(int(probe['operands']['4']), 1, probe['id'])


if __name__ == '__main__':
    unittest.main()
