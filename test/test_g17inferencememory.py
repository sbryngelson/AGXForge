import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from g17modelimport import import_model
from agxforge.g17.inferencememory import plan
from agxforge.g17.inferencegraph import Unsupported

ROOT = Path(__file__).resolve().parents[1]


class MemoryPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = import_model(ROOT/'evidence/g17-inference-models-v1/minilm', sentence_pooling=True)
        cls.decoder = import_model(ROOT/'evidence/g17-inference-models-v1/qwen')

    def test_live_ranges_never_overlap(self):
        for graph in (self.encoder, self.decoder):
            result = plan(graph)
            regions = list(result['regions'].values())
            for i, a in enumerate(regions):
                for b in regions[i+1:]:
                    if a['arena'] != b['arena'] or a['last_use'] < b['birth'] or b['last_use'] < a['birth']:
                        continue
                    self.assertTrue(a['guard_after']+a['guard_bytes'] <= b['guard_before'] or
                                    b['guard_after']+b['guard_bytes'] <= a['guard_before'])
            self.assertTrue(all(r['offset'] % result['alignment'] == 0 for r in regions))

    def test_residual_and_graph_output_lifetimes(self):
        result=plan(self.encoder)
        residual=self.encoder['nodes'][next(i for i,n in enumerate(self.encoder['nodes']) if n['name']=='layer.0.attention_residual')]
        name=residual['inputs'][1]
        self.assertGreaterEqual(result['regions'][name]['last_use'], self.encoder['nodes'].index(residual))
        self.assertEqual(result['regions'][self.encoder['outputs'][0]]['last_use'],len(self.encoder['nodes']))
        self.assertLess(result['activation_slots'],len(self.encoder['nodes']))

    def test_prefill_decode_share_parameter_and_state_offsets(self):
        prefill=plan(self.decoder)
        decode=plan(import_model(ROOT/'evidence/g17-inference-models-v1/qwen',tokens=1))
        for name,r in prefill['regions'].items():
            if r['persistent']:
                self.assertEqual(r,decode['regions'][name])
        self.assertEqual(len(prefill['reset_state']),48)
        self.assertEqual(sum(r['bytes'] for r in prefill['reset_state']),6291456)
        self.assertLess(decode['arenas']['activations'],prefill['arenas']['activations'])

    def test_read_before_producer_refused(self):
        graph=copy.deepcopy(self.encoder)
        graph['nodes'][0]['inputs'].append(graph['outputs'][0])
        with self.assertRaisesRegex(Unsupported,'read before producer'):plan(graph)

    def test_wrong_control_order_refused(self):
        graph=copy.deepcopy(self.decoder)
        graph['nodes'][0]['control_dependencies']=['layer.0.cache_update']
        with self.assertRaisesRegex(Unsupported,'control dependency'):plan(graph)

    def test_parameter_write_refused(self):
        graph=copy.deepcopy(self.encoder)
        graph['nodes'][0]['attrs']['writes']=[graph['nodes'][0]['inputs'][0]]
        with self.assertRaisesRegex(Unsupported,'state inputs'):plan(graph)

    def test_bool_extent_and_invalid_alignment_refused(self):
        graph=copy.deepcopy(self.encoder)
        graph['tensors']['token_ids']['shape']=[True]
        with self.assertRaises(Unsupported):plan(graph)
        with self.assertRaises(Unsupported):plan(self.encoder,alignment=48)


if __name__=='__main__':unittest.main()
