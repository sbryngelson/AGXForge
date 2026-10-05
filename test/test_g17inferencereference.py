import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from g17inferencereference import graph_reference

class Reference(unittest.TestCase):
    def setUp(self):
        self.graph=dict(profile=dict(family='bert',width=384,heads=12,vocabulary=1000),tokens=32,nodes=[
            dict(name='mean',op='masked_mean',inputs=['x','attention_mask'],outputs=['mean'],attrs=dict(denominator_min=1e-9)),
            dict(name='norm',op='l2_normalize',inputs=['mean'],outputs=['norm'],attrs=dict(epsilon=1e-12))])
        self.inputs=dict(token_ids=np.zeros(32,np.int32),position_ids=np.arange(32,dtype=np.int32),
                         token_type_ids=np.zeros(32,np.int32),attention_mask=np.array([1,1]+[0]*30,np.int32))
        self.parameters=dict(x=np.ones((32,384),np.float32))
        self.parameters['x'][2:]=10000

    def test_padding_is_excluded_from_mean_and_normalization(self):
        for policy in ('original_fp64','half_transport_fp64','native_schedule_estimate'):
            r=graph_reference(self.graph,self.parameters,self.inputs,policy=policy)
            np.testing.assert_array_equal(r['mean'],np.ones(384))
            np.testing.assert_allclose(r['norm'],np.full(384,1/np.sqrt(384)),rtol=2e-7,atol=0)

    def test_zero_normalization_has_defined_zero_output(self):
        self.parameters['x'][:]=0
        for policy in ('original_fp64','native_schedule_estimate'):
            r=graph_reference(self.graph,self.parameters,self.inputs,policy=policy)
            np.testing.assert_array_equal(r['norm'],np.zeros(384))

    def test_all_masked_nonbinary_and_out_of_bounds_refuse(self):
        for field,value,message in [('attention_mask',0,'valid token'),('attention_mask',2,'0/1'),
                                     ('token_ids',1000,'index'),('position_ids',512,'index'),('token_type_ids',2,'index')]:
            bad=copy.deepcopy(self.inputs);bad[field][:]=value
            with self.assertRaisesRegex(ValueError,message):
                graph_reference(self.graph,self.parameters,bad,policy='original_fp64')

    def test_unknown_policy_and_architecture_refuse(self):
        with self.assertRaisesRegex(ValueError,'policy'):
            graph_reference(self.graph,self.parameters,self.inputs,policy='whatever')
        self.graph['profile']['width']=896
        with self.assertRaisesRegex(ValueError,'encoder'):
            graph_reference(self.graph,self.parameters,self.inputs,policy='original_fp64')

if __name__=='__main__':unittest.main()
