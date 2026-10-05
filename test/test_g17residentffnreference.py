import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import g17residentffnreference as R

class ReferenceTests(unittest.TestCase):
    def test_mma_order_is_not_dot_product(self):
        a = np.ones((1,16), np.float16)
        w = np.zeros((1,16), np.float16)
        w[0,0] = 4096; w[0,1] = 1; w[0,8] = -4096
        self.assertEqual(float(R.half_mma(a,w)[0,0]), 1)
    def test_mma_multiple_issues(self):
        a = np.ones((2,32), np.float16)
        w = np.full((3,32),2,np.float16)
        np.testing.assert_array_equal(R.half_mma(a,w), np.full((2,3),64,np.float32))
    def test_old_block_fold_is_a_distinct_algorithm(self):
        rng=np.random.default_rng(421)
        a=rng.normal(size=(2,128)).astype(np.float16)
        w=rng.normal(size=(5,128)).astype(np.float16)
        old=R.legacy_block_mma(a,w)
        expected=R._add(R.half_mma(a[:,:64],w[:,:64]),R.half_mma(a[:,64:],w[:,64:]))
        np.testing.assert_array_equal(old,expected)
        self.assertTrue(np.any(old.view('u4')!=R.half_mma(a,w).view('u4')))
    def test_default_reference_uses_64k_partial_fold(self):
        from unittest.mock import patch
        arrays={k:np.zeros(shape,np.float32) for k,shape in R.SHAPES.items()}
        arrays['gamma'].fill(1)
        calls=[]
        def folded(a,w):
            calls.append((a.shape,w.shape))
            return np.zeros((a.shape[0],w.shape[0]),np.float32)
        with patch.object(R,'legacy_block_mma',side_effect=folded), patch.object(R,'half_mma',side_effect=AssertionError('continuous chain selected')):
            got=R.references(arrays)
        self.assertEqual(calls,[((32,384),(1536,384)),((32,1536),(384,1536))])
        self.assertTrue(got['application_error']['passed'])
    def test_hidden_pack_reference_uses_actual_gpu_activation(self):
        arrays={k:np.zeros(shape,np.float32) for k,shape in R.SHAPES.items()}
        stages={k:np.zeros(shape,np.float16 if k in ('pack','pack_hidden') else np.float32) for k,shape in R.STAGE_SHAPES.items()}
        stages['activation'][0,0]=np.float32(1.00049)
        wanted=R.stagewise_references(arrays,stages)
        self.assertEqual(wanted['pack_hidden'][0,0],np.float16(stages['activation'][0,0]))
        self.assertNotEqual(wanted['pack_hidden'][0,0],np.float16(wanted['activation'][0,0]))
        self.assertEqual(R.validation_contract()['submissions'],7)
    def test_nonfinite_never_passes(self):
        self.assertFalse(R.compare(np.array([np.nan]),np.array([np.nan]))['passed'])
    def test_bound_not_relaxed(self):
        self.assertTrue(R.compare(np.array([1.00003]),np.array([1.]))['passed'])
        self.assertFalse(R.compare(np.array([1.00005]),np.array([1.]))['passed'])
    def test_layernorm_constant(self):
        x=np.full((2,384),123,np.float32)
        beta=np.arange(384,dtype=np.float32)/384
        np.testing.assert_array_equal(R.layernorm_model(x,np.ones(384,np.float32),beta),np.broadcast_to(beta,x.shape))
    def test_gelu_against_true_erf(self):
        import g17minilmffn
        x=np.linspace(-8,8,1001,dtype=np.float32)
        self.assertTrue(R.compare(R.gelu_model(x),g17minilmffn.gelu_reference(x))['passed'])
    def test_domains_refuse(self):
        arrays={k:np.zeros(s,np.float32) for k,s in R.SHAPES.items()}
        arrays['source'][0,0]=70000
        with self.assertRaisesRegex(ValueError,'half transport overflow'):
            R.require_arrays(arrays)

if __name__=='__main__': unittest.main()
