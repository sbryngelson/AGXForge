"""Pack reuse preserves programs and extends shared inputs' planned lifetimes."""
import copy
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from g17modelimport import import_model
from g17decoderlower import compile_graph,POLICY,reuse_projection_packs,fuse_crop_bias
from agxforge.g17.inferenceresources import plan
from agxforge.g17.inferencegraph import Unsupported


class Reuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=[]
        for rows in (32,1):
            graph=import_model(ROOT/'evidence/g17-inference-models-v1/qwen',tokens=rows,cache_capacity=256)
            programs,baseline=compile_graph(graph,policy=POLICY)
            optimized=reuse_projection_packs(baseline)
            cls.cases.append((graph,programs,baseline,optimized,plan(graph,optimized)))

    def test_only_duplicate_pack_producers_are_removed(self):
        for graph,programs,baseline,optimized,layout in self.cases:
            self.assertEqual(len(baseline['stages'])-len(optimized['stages']),72)
            removed={r['stage'] for r in optimized['optimization']['removed_stages']}
            self.assertEqual([s['name'] for s in optimized['stages']],
                             [s['name'] for s in baseline['stages'] if s['name'] not in removed])
            self.assertTrue(all(s.endswith('.pack') for s in removed))
            self.assertEqual(set(programs),{s['program_sha256'] for s in optimized['stages']})
            self.assertEqual(optimized['state_effects'],baseline['state_effects'])
            self.assertEqual(optimized['prepared_parameters'],baseline['prepared_parameters'])
            self.assertEqual([r['node'] for r in optimized['coverage']],[n['name'] for n in graph['nodes']])

    def test_remaining_native_code_and_launches_are_unchanged(self):
        for _,_,baseline,optimized,_ in self.cases:
            old={s['name']:s for s in baseline['stages']}
            for s in optimized['stages']:
                for key in ('program_sha256','code_bytes','grid','threadgroup','abi'):
                    self.assertEqual(s[key],old[s['name']][key])

    def test_shared_pack_survives_through_every_consumer(self):
        for _,_,_,optimized,layout in self.cases:
            for removed in optimized['optimization']['removed_stages']:
                name=removed['reused_output'];r=layout['regions'][name]
                consumers=[i for i,s in enumerate(optimized['stages'])
                           if any(b['name']==name and not b['written'] for b in s['bindings'])]
                self.assertGreater(len(consumers),1)
                self.assertEqual(r['last_use'],max(consumers))
                self.assertNotIn(removed['removed_output'],layout['regions'])
                for other_name,other in layout['regions'].items():
                    if other_name==name or other['arena']!='activations':continue
                    if other['offset']==r['offset']:
                        self.assertTrue(other['last_use']<r['birth'] or r['last_use']<other['birth'])

    def test_changed_source_invalidates_reuse(self):
        baseline=copy.deepcopy(self.cases[0][2])
        first_removed=self.cases[0][3]['optimization']['removed_stages'][0]
        position=next(i for i,s in enumerate(baseline['stages']) if s['name']==first_removed['stage'])
        mutation=copy.deepcopy(baseline['stages'][position-1])
        mutation['name']='source_mutation';mutation['bindings'][-1]['name']=first_removed['source']
        baseline['stages'].insert(position,mutation)
        optimized=reuse_projection_packs(baseline)
        self.assertIn(first_removed['stage'],{s['name'] for s in optimized['stages']})

    def test_mutable_pack_and_unknown_policy_refuse(self):
        changed=copy.deepcopy(self.cases[0][2]);changed['numerical_policy']='unknown'
        with self.assertRaisesRegex(Unsupported,'optimization domain'):reuse_projection_packs(changed)
        changed=copy.deepcopy(self.cases[0][2]);pack=next(s for s in changed['stages'] if s['name'].endswith('.pack'))
        changed['stages'].append(copy.deepcopy(pack))
        with self.assertRaisesRegex(Unsupported,'mutable packed'):reuse_projection_packs(changed)

    def test_crop_bias_fusion_preserves_source_bias_output_and_grid(self):
        for graph,programs,_,reuse,_ in self.cases:
            native=dict(programs);fused=fuse_crop_bias(reuse,native);layout=plan(graph,fused)
            self.assertEqual(len(fused['stages']),676 if graph['tokens']==32 else 772)
            self.assertEqual(len(fused['optimization']['fused_stages']),0 if graph['tokens']==32 else 72)
            originals={s['name']:s for s in reuse['stages']};new={s['name']:s for s in fused['stages']}
            for pair in fused['optimization']['fused_stages']:
                crop=originals[pair['crop']];bias=originals[pair['bias']];replacement=new[pair['replacement']]
                self.assertEqual([b['name'] for b in replacement['bindings']],
                                 [crop['bindings'][0]['name'],bias['bindings'][1]['name'],bias['bindings'][2]['name']])
                self.assertEqual(replacement['grid'],crop['grid'])
                self.assertEqual([b['written'] for b in replacement['bindings']],[False,False,True])
                self.assertNotIn(pair['crop'],new);self.assertNotIn(pair['bias'],new)
            self.assertTrue(all(native[sha]==raw for sha,raw in programs.items()))
            self.assertEqual(layout['arenas']['parameters'],plan(graph,reuse)['arenas']['parameters'])

    def test_crop_bias_unknown_width_and_mutable_bias_refuse(self):
        for change in ('width','bias_write'):
            changed=copy.deepcopy(self.cases[1][3])
            crop=next(s for s in changed['stages'] if s['name'].endswith('.crop'))
            bias=next(s for s in changed['stages'] if s['name']==crop['node']+'.bias')
            if change=='width':crop['grid']=[32,1,1];bias['grid']=[32,1,1]
            else:bias['bindings'][1]['written']=True
            with self.assertRaisesRegex(Unsupported,'binding/shape domain'):fuse_crop_bias(changed,{})


if __name__=='__main__':unittest.main()
