"""Decoder graph, tied-weight storage and persistent-state admission checks."""
import copy
import json
import struct
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
from g17modelimport import import_model
from g17inferencelower import compile_graph
from g17decoderlower import POLICY
from agxforge.g17.inferenceresources import plan
from agxforge.g17.inferencegraph import Unsupported
from g17inferencebundle import decoder_contract,decoder_allocations,DECODER_HEADER,STAGE,REGION


class Decoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=[]
        for tokens in (32,1):
            graph=import_model(ROOT/'evidence/g17-inference-models-v1/qwen',tokens=tokens,cache_capacity=256)
            programs,lowering=compile_graph(graph,policy=POLICY)
            cls.cases.append((graph,programs,lowering,plan(graph,lowering)))

    def test_every_graph_node_and_cache_effect_survives(self):
        for graph,programs,lowering,layout in self.cases:
            self.assertEqual([n['name'] for n in graph['nodes']],[n['node'] for n in lowering['coverage']])
            self.assertEqual(len(lowering['state_effects']),24)
            writes=[b['name'] for stage in lowering['stages'] for b in stage['bindings'] if b['written'] and '.cache_' in b['name']]
            self.assertEqual(len(writes),48); self.assertEqual(len(set(writes)),48)
            self.assertFalse(lowering['gpu_admitted'])
            self.assertEqual(lowering['host_control_inputs'],['cache_valid_length'])
            self.assertEqual(layout['regions']['cache_valid_length']['arena'],'inputs')

    def test_live_cache_is_separate_and_never_reused_as_scratch(self):
        for graph,programs,lowering,layout in self.cases:
            regions=[r for r in layout['regions'].values() if r['arena']=='state']
            self.assertEqual(len(regions),48)
            self.assertTrue(all(r['birth']==-1 and r['last_use']==len(lowering['stages']) and not r['readonly'] for r in regions))
            intervals=sorted((r['guard_before'],r['guard_after']+256) for r in regions)
            self.assertTrue(all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])))

    def test_missing_state_effects_refuse(self):
        graph,programs,lowering,layout=self.cases[0]
        changed=dict(lowering,state_effects=[])
        with self.assertRaisesRegex(Unsupported,'exact decoder write effects'):plan(graph,changed)

    def test_prefill_and_decode_share_parameters_and_cache_offsets(self):
        a,b=[case[3] for case in self.cases]
        for name,r in a['regions'].items():
            if r['arena'] in ('parameters','state'):
                self.assertEqual(r['offset'],b['regions'][name]['offset'],name)
                self.assertEqual(r['bytes'],b['regions'][name]['bytes'],name)

    def test_tied_embedding_has_no_second_logits_matrix(self):
        for graph,programs,lowering,layout in self.cases:
            logits=lowering['stages'][-1]
            self.assertEqual(logits['bindings'][1]['name'],'model.embed_tokens.weight')
            self.assertNotIn('prepared:model.embed_tokens.weight',layout['regions'])
            self.assertLess(sum(layout['arenas'].values()),1<<30)
            self.assertLess(lowering['unique_code_bytes'],0x10000-0x6c0)

    def test_unmeasured_domain_refuses(self):
        graph=self.cases[0][0]
        with self.assertRaisesRegex(Unsupported,'explicit Qwen'):compile_graph(graph,policy='guess')
        changed=copy.deepcopy(graph);changed['profile']['heads']=7
        with self.assertRaisesRegex(Unsupported,'profile'):compile_graph(changed,policy=POLICY)
        changed=copy.deepcopy(graph)
        for tensor in changed['tensors'].values():
            if tensor['role']=='state':tensor['shape'][1]=128
        with self.assertRaisesRegex(Unsupported,'capacity'):compile_graph(changed,policy=POLICY)

    def test_decode_padding_and_crop_are_gpu_stages(self):
        graph,programs,lowering,layout=self.cases[1]
        for node in lowering['coverage']:
            if node['op']=='linear' and node['node']!='logits':
                suffixes=[name.rsplit('.',1)[-1] for name in node['stages']]
                self.assertEqual(suffixes[:3],['pack','gemm','crop'])

    def test_one_gib_bundle_uses_measured_low_mirror_and_shared_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,programs,_,layout=self.cases[0]
            manifest=dict(policy=POLICY,layout=layout,parameter_file=dict(bytes=layout['arenas']['parameters'],sha256='0'*64),
                          programs={sha:len(code) for sha,code in programs.items()})
            (root/'manifest.json').write_text(json.dumps(manifest))
            for sha,code in programs.items():(root/(sha+'.bin')).write_bytes(code)
            with patch('g17inferencebundle.compile_graph',side_effect=[(c[1],c[2]) for c in self.cases]):
                report,native,wires=decoder_contract(root)
            small=decoder_allocations(report,native)
            self.assertNotIn(20,small)  # the GiB payload is streamed
            self.assertEqual(small[17][0x10000:0x18000],small[23])
            self.assertEqual(small[17][:0xc000],small[28])
            self.assertEqual(struct.unpack_from('<H',small[25],9)[0],0x001a)
            self.assertLessEqual(report['state_base']+report['state_bytes'],1<<30)
            self.assertLess(report['code_end'],1<<16)
            for mode in ('prefill','decode'):
                schedule=report['schedules'][mode];wire=wires[mode+'-schedule']
                h=DECODER_HEADER.unpack_from(wire)
                self.assertEqual(h[:4],(0x47494e32,len(schedule['stages']),len(schedule['regions']),1284 if mode=='prefill' else 1036))
                self.assertEqual(len(wire),DECODER_HEADER.size+h[1]*STAGE.size+h[2]*REGION.size)
                state=[r for r in schedule['regions'].values() if r['arena']=='state']
                self.assertEqual(len(state),48)
                self.assertTrue(all(r['guard_before']>=report['state_base'] for r in state))
            before=report['schedules']['prefill']['regions'];after=report['schedules']['decode']['regions']
            for name,r in before.items():
                if r['arena'] in ('parameters','state'):
                    self.assertEqual(r['gpu_address'],after[name]['gpu_address'])
                    self.assertEqual(r['bytes'],after[name]['bytes'])
            # A changed code file cannot hide behind the manifest's digest key.
            sha=next(iter(programs));(root/(sha+'.bin')).write_bytes(b'broken')
            with patch('g17inferencebundle.compile_graph',side_effect=[(c[1],c[2]) for c in self.cases]):
                with self.assertRaisesRegex(ValueError,'native identity'):decoder_contract(root)


if __name__=='__main__':unittest.main()
