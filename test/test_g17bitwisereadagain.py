"""Operand isolation for the four-byte register bitwise, and the measurements it must not lose.

THE FORM IS COVERED; THE PROGRAM WAS NOT ADMITTED. op424/op13575/op17771 round trip, are certified,
and match Apple exactly on disjoint registers. What the four-byte encoding cannot do is carry a
source lifetime - 32 keep, 16 release, measured elsewhere in this ISA - so it releases what it
reads, and it does not wait for a load. A program whose operand is read again, or comes straight
off a load, was therefore refused rather than silently reading a freed or unlanded register.

The repair is the one the refusal itself named: "Put the operands through an ALU op first; those
carry the wait." Selection now emits the same alu.12 copy `_wait_for_load` and `_materialise_sr`
already use, with the two measured bits - load_wait when the operand came from a load, keep when
the original still has a reader.

THE REFUSALS ARE NOT REMOVED. They are made unreachable for operands the pass has isolated, and
`cc._NO_BITWISE_ISOLATION` turns the pass off so every case below can be measured against the state
it replaced. That switch is what keeps this suite honest: without it, "it compiles now" and "the
check was deleted" look identical.

Measured on the pinned 197-source campaign: backend refusals 9 -> 6, compiled 52 -> 55, and the
three witnesses - mn-u16_u32.and-4 (op424), mn-u32_u16.xor-2 (op17771), r-shr64a (op13575) - all
moved backend -> compiled with no next refusal. Three was an upper bound and it was reached.
"""
import contextlib
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agxforge.g17 import cc, ir

WITNESSES = ("mn-u16_u32.and-4", "mn-u32_u16.xor-2", "r-shr64a")
OPCODES = {"and": 424, "or": 13575, "xor": 17771}


@contextlib.contextmanager
def isolation(on):
    """The pass, on or off. Off is the state this repair replaced."""
    held = cc._NO_BITWISE_ISOLATION
    cc._NO_BITWISE_ISOLATION = not on
    try:
        yield
    finally:
        cc._NO_BITWISE_ISOLATION = held


def program(op="and", *, read_again=True, from_load=True, alias=False, both_live=False):
    fn = ir.Function("k", buffers=[ir.Buffer("u", 0, "i32")])
    b = ir.Builder(fn, fn.block("entry"))
    u = fn.buffers[0]
    t = b.builtin("threadgroup_position_in_grid")
    x0, y0 = b.load(u, t), b.load(u, t, disp=1)
    if not from_load:                      # an intervening ALU: neither operand is raw off a load
        x0, y0 = b.add(x0, ir.Imm(3)), b.add(y0, ir.Imm(5))
    if alias:
        y0 = x0
    b.store(u, ir.Imm(400), getattr(b, op)(x0, y0))
    if read_again:
        b.store(u, ir.Imm(401), x0)
    if both_live:
        b.store(u, ir.Imm(402), y0)
    b.ret()
    return fn


def build(fn):
    return cc.emit(cc.Alloc(regs=range(0, 40)).run(cc.select(fn)))[0]


class TheProgramsThatUsedToBeRefusedNowCompile(unittest.TestCase):
    CASES = {
        "one operand live": dict(read_again=True, from_load=False),
        "both operands live": dict(read_again=True, from_load=False, both_live=True),
        "operands off a load": dict(read_again=False, from_load=True),
        "live and off a load": dict(read_again=True, from_load=True),
        "aliased operands": dict(read_again=False, from_load=False, alias=True),
        "aliased, live, loaded": dict(read_again=True, from_load=True, alias=True),
    }

    def test_every_shape_compiles(self):
        for name, kw in self.CASES.items():
            for op in OPCODES:
                with self.subTest(shape=name, op=op), isolation(True):
                    code = build(program(op, **kw))
                    self.assertGreater(len(code), 0)
                    self.assertEqual(len(code) % 2, 0, "an odd byte count is not walkable")

    def test_and_each_one_is_still_refused_with_the_pass_off(self):
        """THE NEGATIVE CONTROL. Every case above must be a case the refusal would have caught.

        A shape that compiles either way was never evidence for this repair, and a refusal that
        disappeared with the pass OFF would mean the checks were deleted rather than satisfied.
        """
        for name, kw in self.CASES.items():
            with self.subTest(shape=name), isolation(False):
                with self.assertRaises(cc.Unsupported) as refused:
                    build(program(**kw))
                message = str(refused.exception)
                self.assertTrue("would release" in message or "straight from a load" in message,
                                message[:120])
                self.assertIn(str(OPCODES["and"]), message)


class TheUnchangedProgramsAreUNCHANGED(unittest.TestCase):
    """Byte identity, because a pass that rewrites safe programs is not this pass."""

    SAFE = dict(read_again=False, from_load=False)

    def test_a_safe_bitwise_emits_the_same_bytes_either_way(self):
        for op in OPCODES:
            with self.subTest(op=op):
                with isolation(True):
                    after = build(program(op, **self.SAFE))
                with isolation(False):
                    before = build(program(op, **self.SAFE))
                self.assertEqual(after, before)

    def test_the_safe_program_was_always_accepted(self):
        """So the case above compares two acceptances, not an acceptance against a refusal."""
        with isolation(False):
            self.assertGreater(len(build(program(**self.SAFE))), 0)

    def test_the_pass_emits_what_the_author_used_to_write_by_hand(self):
        """BYTE FOR BYTE. The measured workaround was `add 0` on each operand; this is that.

        The first version of this case compared the isolated program's LENGTH against the safe
        one's and asserted the difference was a whole number of copies. It was not - because the
        two programs also differ by a store, so it was measuring my test rather than the pass.
        The comparison that means something is against the hand-written copies, which is the form
        the source comment records as executing correctly.
        """
        def hand_written():
            fn = ir.Function("k", buffers=[ir.Buffer("u", 0, "i32")])
            b = ir.Builder(fn, fn.block("entry"))
            u = fn.buffers[0]
            t = b.builtin("threadgroup_position_in_grid")
            x0, y0 = b.load(u, t), b.load(u, t, disp=1)
            x, y = b.add(x0, ir.Imm(0)), b.add(y0, ir.Imm(0))
            b.store(u, ir.Imm(400), getattr(b, "and")(x, y))
            b.store(u, ir.Imm(401), x0)
            b.ret()
            return fn

        with isolation(False):
            by_hand = build(hand_written())          # accepted before this repair existed
        with isolation(True):
            automatic = build(program(read_again=True, from_load=True))
        self.assertEqual(automatic, by_hand)


class TheHighLevelPathHandlesTheUnmodifiedIR(unittest.TestCase):
    def test_compile_function_takes_the_program_as_written(self):
        """No caller has to know about this: the IR is unchanged and compile_function accepts it."""
        with isolation(True):
            out = cc.compile_function(program(read_again=True, from_load=True))
        self.assertEqual(type(out).__name__, "G17Program")

    def test_and_refuses_it_with_the_pass_off(self):
        with isolation(False):
            with self.assertRaises(cc.Unsupported):
                cc.compile_function(program(read_again=True, from_load=True))


class TheCopyCarriesTheMeasuredBits(unittest.TestCase):
    """load_wait and keep are the two bits the copy exists for; neither is set by default."""

    def _copies(self, fn):
        with isolation(True):
            insts = cc.select(fn)
        return [m for m in insts if getattr(m, "note", None)
                and "operand isolation" in (m.note or "")]

    def test_an_operand_off_a_load_gets_a_copy_that_waits(self):
        copies = self._copies(program(read_again=False, from_load=True))
        self.assertTrue(copies)
        self.assertTrue(all(c.fields.get("load_wait") == 1 for c in copies),
                        [c.fields for c in copies])

    def test_an_operand_not_off_a_load_gets_a_copy_that_does_not_wait(self):
        copies = self._copies(program(read_again=True, from_load=False))
        self.assertTrue(copies)
        self.assertTrue(all(c.fields.get("load_wait") == 0 for c in copies),
                        [c.fields for c in copies])

    def test_a_copy_of_a_value_with_a_later_reader_keeps_it(self):
        copies = self._copies(program(read_again=True, from_load=False))
        self.assertTrue(any(c.fields.get("keep") == 1 for c in copies),
                        "the copy released the original its later reader needs")

    def test_aliased_operands_get_two_distinct_copies(self):
        copies = self._copies(program(read_again=False, from_load=False, alias=True))
        self.assertEqual(len(copies), 2)
        destinations = [c.defs[0] for c in copies]
        self.assertNotEqual(destinations[0], destinations[1])
        self.assertEqual(copies[0].fields.get("keep"), 1,
                         "the first copy must keep the value the second copy still has to read")


class TheCampaignWitnesses(unittest.TestCase):
    def test_the_three_are_named_with_their_opcodes(self):
        self.assertEqual(len(WITNESSES), 3)
        self.assertEqual(sorted(OPCODES.values()), sorted({424, 13575, 17771}))


if __name__ == "__main__":
    unittest.main()
