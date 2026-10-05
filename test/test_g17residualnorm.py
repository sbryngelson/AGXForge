from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import g17ir as ir
import g17layernorm
import g17residualnorm as R


class ResidualLayerNorm(unittest.TestCase):
    def test_fusion_keeps_rounding_before_normalization(self):
        # FP64 addition would retain the small term and change the norm.
        x=np.array([[2**24,0,0,0]],np.float32)
        y=np.array([[1,2,3,4]],np.float32)
        g=np.array([1,2,3,4],np.float32);b=np.array([.5,0,-.5,1],np.float32)
        got=R.reference(x,y,g,b)
        expected=g17layernorm.reference(np.add(x,y,dtype=np.float32),g,b)
        np.testing.assert_array_equal(got,expected)
        wide=x.astype(np.float64)+y.astype(np.float64)
        centered=wide-wide.mean(axis=1,keepdims=True)
        wrong=centered/np.sqrt(np.mean(centered*centered,axis=1,keepdims=True)+1e-12)*g+b
        self.assertFalse(np.array_equal(got,wrong))

    def test_opposite_inputs_return_beta(self):
        x=np.arange(12,dtype=np.float32).reshape(3,4)
        b=np.array([1,-1,.25,.5],np.float32)
        np.testing.assert_array_equal(R.reference(x,-x,np.ones(4,np.float32),b),np.broadcast_to(b,x.shape))

    def test_every_layernorm_source_is_replaced_by_two_loads_and_a_sum(self):
        fn=R.residualnorm_ir(1,4)
        self.assertEqual([(b.name,b.slot) for b in fn.buffers],
                         [("source",1),("residual",2),("gamma",3),("beta",4),("output",5)])
        ops=fn.blocks[0].ops
        source_loads=[op for op in ops if op.kind=="load" and op.args[0].name=="source"]
        self.assertEqual(len(source_loads),4)
        for op in source_loads:
            i=ops.index(op)
            self.assertEqual(ops[i+1].args[0].name,"residual")
            self.assertIs(ops[i+1].args[1],op.args[1])
            self.assertEqual(ops[i+2].kind,"fadd")
            self.assertEqual(ops[i+2].args,[op.dest,ops[i+1].dest])
        self.assertEqual(sum(op.kind=="store_at" for op in ops),4)

    def test_original_layernorm_remains_four_bindings(self):
        R.residualnorm_ir(1,4)
        self.assertEqual([(b.name,b.slot) for b in g17layernorm.layernorm_ir(1,4).buffers],
                         [("source",1),("gamma",2),("beta",3),("output",4)])

    def test_mismatched_or_nonfinite_inputs_refuse(self):
        x=np.ones((1,4),np.float32);g=np.ones(4,np.float32);b=np.zeros(4,np.float32)
        for y in (x[:,:3],x.astype(np.float16),x*np.nan):
            with self.assertRaises(ValueError):R.reference(x,y,g,b)


if __name__=="__main__":unittest.main()
