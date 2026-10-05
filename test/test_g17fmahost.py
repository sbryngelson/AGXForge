"""Caching the foreign-function binding must preserve fused FP32 arithmetic."""
from pathlib import Path
import ctypes
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import g17normcheck as N


class HostFma(unittest.TestCase):
    def test_resolves_once_and_retains_single_rounding(self):
        N._host_fmaf.cache_clear()
        self.addCleanup(N._host_fmaf.cache_clear)
        a=np.float32(1+2**-23);b=np.float32(1-2**-23);c=np.float32(-1)
        self.assertEqual(np.float32(np.float32(a*b)+c),0)
        machine=N.Machine([],{},0,frozenset(),frozenset())
        for reg,value in zip(('reg:106','reg:107','reg:108'),(a,b,c)):
            machine.regs[reg]=N._to_bits(value)
        with patch('ctypes.CDLL',wraps=ctypes.CDLL) as library:
            for _ in range(64):
                machine.step(0,16,N.FMA,['reg:105','imm:0','reg:106','reg:107','reg:108'])
                self.assertEqual(machine.regs['reg:105'],N._to_bits(-2**-46))
            library.assert_called_once_with(None)
