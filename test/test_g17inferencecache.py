"""Structural reuse must distinguish semantic IR changes, not SSA numbering."""
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from g17inferencelower import function_key,element_ir
from agxforge.g17 import ir


class Cache(unittest.TestCase):
    def test_fresh_ssa_names_share_key(self):
        a=element_ir('bias',896);b=element_ir('bias',896)
        self.assertNotEqual(repr(a),repr(b))
        self.assertEqual(function_key(a),function_key(b))

    def test_types_slots_constants_and_attributes_distinguish(self):
        original=function_key(element_ir('bias',896))
        self.assertNotEqual(original,function_key(element_ir('bias',128)))
        for change in ('type','slot','declared','attrs','threadgroup'):
            fn=element_ir('bias',896)
            if change=='type':fn.blocks[0].ops[0].dest.type=ir.F32
            elif change=='slot':fn.buffers[0].slot=4
            elif change=='declared':fn.buffers[0].declared_element='different'
            elif change=='attrs':fn.blocks[0].ops[0].attrs['axis']='z'
            else:fn.declare_threadgroup(32)
            with self.subTest(change=change):self.assertNotEqual(original,function_key(fn))

    def test_operand_identity_is_preserved(self):
        a=element_ir('bias',896);b=element_ir('bias',896)
        b.blocks[0].ops[-2].args[1]=b.blocks[0].ops[-2].args[0]
        self.assertNotEqual(function_key(a),function_key(b))

    def test_unknown_attributes_fail_closed(self):
        fn=element_ir('bias',896);fn.blocks[0].ops[0].attrs['unknown']=object()
        with self.assertRaisesRegex(TypeError,'cache field'):function_key(fn)


if __name__=='__main__':unittest.main()
