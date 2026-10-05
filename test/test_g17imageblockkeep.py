"""The imageblock store's coordinate lifetime (slot 6) comes from liveness: keep when read again."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from agxforge.g17 import ir, cc
import g17ref, g17packedcheck as D


def prog(read_after=True, reuse_value=False):
    f = ir.Function("ib", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("e"))
    tid = b.builtin("thread_position_in_threadgroup", axis="x", name="tid")
    v = b.add(b.const(0xA5000000, name="tag"), tid, name="v")
    b.imageblock_write(v, member=0)
    at = b.add(b.const(256, name="base"), tid, name="at")
    if read_after:
        b.barrier("imageblock")
        b.store_at(f.buffers[2], at, b.imageblock_read(member=0, dx=0, name="r"), width="word")
    elif reuse_value:
        b.store_at(f.buffers[2], at, b.add(v, ir.Imm(1), name="w"), width="word")
    else:
        b.store_at(f.buffers[2], at, tid, width="word")
    b.ret(); ir.verify(f)
    return f


def slot6(f):
    code = bytes(cc.emit(cc.Alloc(regs=range(0, 40)).run(cc.select(f)))[0])
    for a, l, o in g17ref.walk(code, 0):
        if o == 13075:
            return D.decode(code[a:a + l] + bytes.fromhex("0e000000"))[0][3][6]


class TheCoordinateLifetime(unittest.TestCase):

    def test_a_coordinate_read_after_the_store_is_kept(self):
        self.assertEqual(slot6(prog(read_after=True)), "imm:0")

    def test_a_dead_coordinate_is_released_as_before(self):
        self.assertEqual(slot6(prog(read_after=False)), "imm:16")

    def test_a_stored_value_read_again_is_refused(self):
        with self.assertRaisesRegex(cc.Unsupported, "VALUE is read again"):
            slot6(prog(read_after=False, reuse_value=True))



class ACrossLaneOperandFromALoadWaits(unittest.TestCase):
    """simd_shuffle_xor and simd_broadcast read a loaded value only through the waiting copy."""

    def test_the_shuffle_of_a_load_goes_through_a_waiting_copy(self):
        f = ir.Function("sh", [ir.Buffer("S", 1), ir.Buffer("O", 2)])
        b = ir.Builder(f, f.block("e"))
        t = b.builtin("thread_position_in_grid", name="t")
        v = b.load(f.buffers[0], t, type=ir.F32, name="v")
        b.store_at(f.buffers[1], t, b.fadd(v, b.simd_shuffle_xor(v, 1)))
        b.ret()
        forms = [(m.form, m.fields.get("load_wait"), m.fields.get("opcode")) for m in cc.select(f)]
        at = [i for i, x in enumerate(forms) if x[2] == 14169][0]
        self.assertEqual(forms[at - 1][:2], ("alu.12", 1), forms)


if __name__ == "__main__":
    unittest.main()
