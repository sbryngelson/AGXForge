import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from g17modelimport import import_model
import g17inferencelower as L
import g17packedcheck as D

ROOT=Path(__file__).resolve().parents[1]


class FullEncoderLowering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph=import_model(ROOT/'evidence/g17-inference-models-v1/minilm',sentence_pooling=True)
        cls.programs,cls.report=L.compile_graph(cls.graph,policy=L.POLICY)

    def test_every_model_node_has_native_stages_in_order(self):
        self.assertEqual([n['name'] for n in self.graph['nodes']],[n['node'] for n in self.report['coverage']])
        self.assertEqual(len(self.report['coverage']),79)
        self.assertEqual(len(self.report['stages']),170)
        self.assertEqual(len(self.programs),16)
        self.assertLess(self.report['unique_code_bytes']+64*len(self.programs)+0x6c0,65536)
        self.assertFalse(self.report['gpu_admitted'])
        self.assertTrue(self.report['pending'])
        self.assertEqual(self.report['stages'][-1]['node'],'sentence.embedding')

    def test_embedding_binding_order_is_ids_then_table(self):
        nodes={n['name']:n for n in self.graph['nodes']}
        for stage in self.report['stages']:
            if stage['name'].endswith('.gather'):
                node=nodes[stage['node']]
                self.assertEqual([b['name'] for b in stage['bindings']],[node['inputs'][1],node['inputs'][0],node['outputs'][0]])
                self.assertEqual([b['written'] for b in stage['bindings']],[False,False,True])

    def test_projection_bridges_and_mmas_survive_lowering(self):
        stages=self.report['stages']
        mm=[s for s in stages if s['name'].endswith('.gemm')]
        self.assertEqual(len(mm),36)
        for s in mm:
            ops=[op for _,_,op,_ in D.decode(self.programs[s['program_sha256']])]
            self.assertIn(5106,ops)
            self.assertIn(458,ops)
            self.assertIn(17257,ops)
            self.assertEqual(ops.count(684),1)
            self.assertTrue(s['bindings'][1]['name'].startswith('prepared:'))
        for stage in stages:
            self.assertLessEqual(len(stage['bindings']),3)
            self.assertTrue(all(b['bytes']>0 for b in stage['bindings']))
            self.assertLessEqual(stage['abi']['register_count'],126)

    def test_attention_padding_is_not_dropped(self):
        for i in range(6):
            rows=[s for s in self.report['stages'] if s['node']==f'layer.{i}.attention']
            self.assertEqual([s['name'].rsplit('.',1)[-1] for s in rows],['scores','mask','softmax','context'])
            self.assertEqual(rows[1]['bindings'][1]['name'],'attention_mask')
            self.assertTrue(rows[1]['bindings'][0]['written'])

    def test_existing_projection_bodies_are_byte_identical(self):
        import g17residentffnprograms as F
        import g17residentattentionprograms as A
        from agxforge.g17 import cc
        expected={(1536,384):cc.compile_function(F.tensor_ir('expand')).code,
                  (384,1536):cc.compile_function(F.tensor_ir('contract')).code,
                  (384,384):cc.compile_function(A.tensor_ir('q_projection')).code}
        for stage in self.report['stages']:
            if stage['name'].endswith('.gemm'):
                weight=self.report['prepared_parameters'][stage['bindings'][1]['name']]
                k,n=weight['shape']
                self.assertEqual(self.programs[stage['program_sha256']],expected[n,k])

    def test_mask_pool_and_normalize_decoded_bytes(self):
        import numpy as np
        import g17emu as E
        from agxforge.g17 import cc
        mask=(np.arange(32)%3!=0).astype('<u4')
        scores=np.linspace(-3,3,12288,dtype='<f4')
        buffers={1:scores.copy().view(np.uint8),2:mask.view(np.uint8)}
        E.Machine(cc.compile_function(L.score_mask_ir()).code,buffers,12288,32,views=True).run()
        expected=scores+np.where(np.tile(mask,384)==1,np.float32(0),np.float32(-3.4028234663852886e38))
        self.assertTrue(np.array_equal(buffers[1].view('<u4'),expected.view('<u4')))
        x=np.random.default_rng(12).normal(size=(32,384)).astype('<f4')
        buffers={1:x.reshape(-1).view(np.uint8),2:mask.view(np.uint8),3:np.full(384*4,0xa5,np.uint8)}
        E.Machine(cc.compile_function(L.mean_ir()).code,buffers,384,32,views=True).run()
        reference=x[mask==1].astype(np.float64).mean(axis=0)
        self.assertLess(float(np.max(np.abs(buffers[3].view('<f4')-reference))),1e-5)
        for x in (np.linspace(-1,1,384,dtype='<f4'),np.zeros(384,dtype='<f4')):
            buffers={1:x.view(np.uint8),2:np.full(384*4,0xa5,np.uint8)}
            E.Machine(cc.compile_function(L.normalize_ir()).code,buffers,32,32,views=True).run()
            reference=x.astype(np.float64)/max(float(np.linalg.norm(x.astype(np.float64))),1e-12)
            self.assertLess(float(np.max(np.abs(buffers[2].view('<f4')-reference))),1e-5)

    def test_unknown_operation_and_changed_attention_refuse(self):
        graph=copy.deepcopy(self.graph);graph['nodes'][0]['op']='host_embedding'
        with self.assertRaisesRegex(L.Unsupported,'native graph operation'):L.compile_graph(graph,policy=L.POLICY)
        graph=copy.deepcopy(self.graph)
        next(n for n in graph['nodes'] if n['op']=='attention')['attrs']['scale']=1
        with self.assertRaisesRegex(L.Unsupported,'attention'):L.compile_graph(graph,policy=L.POLICY)

    def test_no_implicit_numerical_policy_or_decoder_admission(self):
        with self.assertRaisesRegex(L.Unsupported,'numerical policy'):L.compile_graph(self.graph,policy='original_fp32')
        decoder=import_model(ROOT/'evidence/g17-inference-models-v1/qwen')
        with self.assertRaisesRegex(L.Unsupported,'explicit Qwen'):L.compile_graph(decoder,policy=L.POLICY)


if __name__=='__main__':unittest.main()
