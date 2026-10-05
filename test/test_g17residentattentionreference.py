import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import g17residentattentionreference as A

class AttentionReferenceTests(unittest.TestCase):
    def arrays(self):
        return dict(source=np.zeros((32,384),np.float32),**{k:np.zeros(s,np.float32) for k,s in A.PARAM_SHAPES.items()})
    def test_scope_and_stages(self):
        c=A.validation_contract();self.assertEqual(c['submissions'],14)
        self.assertIn('Normalized',c['input']);self.assertEqual(len(A.STAGE_SHAPES),14)
    def test_zero_scores_softmax(self):
        p=A.scalar_softmax(np.zeros((12,32,32),np.float32))
        np.testing.assert_array_equal(p,np.full((12,32,32),1/32,np.float32))
    def test_scores_head_routing(self):
        q=np.zeros((32,384),np.float32);k=q.copy();q[:,64:96]=1;k[:,64:96]=2
        s=A.scalar_scores(q,k)
        self.assertTrue(np.all(s[2]>0));np.testing.assert_array_equal(s[:2],0);np.testing.assert_array_equal(s[3:],0)
    def test_context_head_routing(self):
        p=np.full((12,32,32),1/32,np.float32);v=np.repeat(np.arange(12,dtype=np.float32),32)[None,:].repeat(32,axis=0)
        np.testing.assert_array_equal(A.scalar_context(p,v),v)
    def test_original_and_quantized_are_distinct(self):
        a=self.arrays();a['source'].fill(.1234567)
        a['query_weight'].fill(.01234567)
        r=A.application_reference(a);q=A.application_reference(a,quantized=True)
        self.assertTrue(np.any(r['query']!=q['query']))
    def test_original_final_bound_is_existing_factor(self):
        self.assertTrue(A.original_final_compare(np.array([1.0003]),np.array([1.]))['passed'])
        self.assertFalse(A.original_final_compare(np.array([1.0005]),np.array([1.]))['passed'])
    def test_refuses_half_overflow(self):
        a=self.arrays();a['key_weight'][0,0]=70000
        with self.assertRaisesRegex(ValueError,'half transport overflow'):A.require_arrays(a)
    def test_context_pack_uses_captured_predecessor(self):
        a=self.arrays();p={k:np.zeros(s,np.float16 if k in A.HALF_STAGES else np.float32) for k,s in A.STAGE_SHAPES.items()}
        p['context'][0,0]=1.25
        self.assertEqual(A.stagewise_references(a,p)['context_pack'][0,0],np.float16(1.25))

if __name__=='__main__':unittest.main()
