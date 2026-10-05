"""CPU instruction-model controls; these token fixtures are not native images."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import g17queryimagecheck as machine

BINDINGS = [(1, 0, False), (2, 2, False), (3, 4, False), (4, 6, True)]


def stream():
    result = []
    def emit(size, opcode, *tokens):
        off = sum(i[1] for i in result)
        result.append((off, size, opcode, list(tokens)))
        return off
    emit(4, 14059, "reg:109", "imm:1048576", "reg:61", "imm:0")
    emit(4, 14059, "reg:110", "imm:1048576", "reg:62", "imm:0")
    emit(8, 11842, "reg:111", "imm:68736253952", "imm:384")
    emit(14, 10825, "reg:112", "imm:0", "reg:110", "imm:16", "reg:111", "imm:16", "imm:0")
    emit(12, 10282, "reg:113", "imm:0", "reg:112", "imm:16", "reg:109", "imm:0")
    emit(14, 12682, "reg:114", "imm:137464119296", "imm:2066", "expr:bin(op0,const(0),8)",
         "imm:0", "reg:113", "imm:0", "imm:0", "imm:4")
    emit(14, 12682, "reg:115", "imm:137464119296", "imm:2066", "expr:bin(op0,const(8),8)",
         "imm:0", "reg:109", "imm:0", "imm:0", "imm:4")
    emit(8, 11842, "reg:116", "imm:68736253952", "imm:0")
    emit(8, 11842, "reg:117", "imm:68736253952", "imm:0")
    loop = emit(16, 2190, "reg:116", "imm:10737418240", "reg:114", "imm:32", "reg:115", "imm:32", "reg:116", "imm:32")
    emit(12, 10279, "reg:117", "imm:0", "imm:1", "reg:117", "imm:0")
    emit(4, 577, "imm:2345052143616", "imm:1")
    emit(6, 10369, "reg:74", "imm:0", "imm:9", "reg:117", "imm:0", "imm:2")
    emit(4, 582, "imm:0", "reg:74", "imm:1")
    off = sum(i[1] for i in result)
    emit(10, 458, "imm:0", f"imm:{loop-off}")
    emit(4, 577, "imm:2345052143616", "imm:1")
    emit(8, 17229, "reg:116", "imm:16", "imm:2066", "expr:bin(op0,const(12),8)",
         "imm:0", "reg:113", "imm:16", "imm:0", "imm:4")
    emit(4, 684, "imm:0")
    return result


class QueryInstructionModel(unittest.TestCase):
    def inputs(self):
        return (np.arange(768, dtype=np.float32).reshape(2, 384),
                np.zeros((384, 384), np.float32), np.full(384, 0.5, np.float32))

    def test_two_coordinates_fma_and_decoded_loop_target(self):
        x, w, b = self.inputs()
        result, report = machine.simulate(stream(), x, w, b, BINDINGS)
        np.testing.assert_array_equal(result, x)
        self.assertEqual(report["branches"], 2)
        self.assertEqual(report["stores"], 768)

    def test_noninstruction_branch_target_refuses(self):
        program = stream()
        next(i for i in program if i[2] == 458)[3][-1] = "imm:-1"
        with self.assertRaisesRegex(ValueError, "instruction boundary"):
            machine.simulate(program, *self.inputs(), BINDINGS)

    def test_wrong_buffer_and_unknown_opcode_refuse(self):
        program = stream()
        next(i for i in program if i[2] == 17229)[3][3] = "expr:bin(op0,const(0),8)"
        with self.assertRaisesRegex(ValueError, "read-only"):
            machine.simulate(program, *self.inputs(), BINDINGS)
        program = stream()
        off, size, _, tokens = program[0]
        program[0] = (off, size, 99999, tokens)
        with self.assertRaisesRegex(ValueError, "unsupported delivered instruction"):
            machine.simulate(program, *self.inputs(), BINDINGS)
