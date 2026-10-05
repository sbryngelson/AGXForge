"""CPU importer boundaries and whole-model structure, with no checkpoint payload."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import g17modelimport as M
from agxforge.g17 import inferencegraph as I


class Header(unittest.TestCase):
    def valid(self, records=None):
        records = records or {'x': dict(dtype='F32', shape=[2], data_offsets=[0, 8])}
        raw = json.dumps(records).encode()
        return raw, 8 + len(raw) + 8

    def test_valid_extent(self):
        raw, size = self.valid()
        p = I.checkpoint_header(raw, file_bytes=size)['x']
        self.assertEqual((p.dtype, p.shape, p.bytes), ('F32', (2,), 8))

    def test_duplicate_names_refused(self):
        raw = b'{"x":{},"x":{}}'
        with self.assertRaisesRegex(I.InvalidCheckpoint, 'duplicate'):
            I.checkpoint_header(raw, file_bytes=8 + len(raw))

    def test_overlapping_payload_refused(self):
        raw, size = self.valid({n: dict(dtype='F32', shape=[2], data_offsets=[0, 8]) for n in ['x', 'y']})
        with self.assertRaisesRegex(I.InvalidCheckpoint, 'overlap'):
            I.checkpoint_header(raw, file_bytes=size)

    def test_hole_refused(self):
        raw, size = self.valid({'x': dict(dtype='F32', shape=[2], data_offsets=[4, 12])})
        with self.assertRaisesRegex(I.InvalidCheckpoint, 'hole'):
            I.checkpoint_header(raw, file_bytes=size + 4)

    def test_dtype_extent_disagreement_refused(self):
        raw, size = self.valid({'x': dict(dtype='BF16', shape=[2], data_offsets=[0, 8])})
        with self.assertRaisesRegex(I.InvalidCheckpoint, 'extent'):
            I.checkpoint_header(raw, file_bytes=size)

    def test_truncation_or_suffix_refused(self):
        raw, size = self.valid()
        for delta in (-1, 1):
            with self.assertRaisesRegex(I.InvalidCheckpoint, 'whole file'):
                I.checkpoint_header(raw, file_bytes=size + delta)

    def test_boolean_shape_is_not_integer_extent(self):
        raw, size = self.valid({'x': dict(dtype='F32', shape=[True, 2], data_offsets=[0, 8])})
        with self.assertRaisesRegex(I.InvalidCheckpoint, 'shape'):
            I.checkpoint_header(raw, file_bytes=size)

    def test_unsupported_dtype_is_named(self):
        raw, size = self.valid({'x': dict(dtype='F8_E4M3', shape=[8], data_offsets=[0, 8])})
        with self.assertRaisesRegex(I.Unsupported, 'checkpoint dtype'):
            I.checkpoint_header(raw, file_bytes=size)


class Model(unittest.TestCase):
    def load(self, name):
        return M.load(ROOT / 'evidence/g17-inference-models-v1' / name)

    def test_complete_encoder_and_verified_sentence_wrapper(self):
        g = M.import_model(ROOT / 'evidence/g17-inference-models-v1/minilm', sentence_pooling=True)
        self.assertEqual(g['outputs'], ['sentence.embedding'])
        self.assertEqual(g['tensors']['sentence.embedding']['shape'], [384])
        self.assertEqual(sum(n['op'] == 'attention' for n in g['nodes']), 6)
        self.assertEqual(sum(n['op'] == 'gelu_erf' for n in g['nodes']), 6)
        self.assertEqual(g['required_parameter_bytes'], 90261504)
        self.assertFalse(g['gpu_admitted'])
        self.assertEqual(g['status'], 'logical_graph_not_lowered')

    def test_full_decoder_prefill_and_decode_share_parameters(self):
        directory = ROOT / 'evidence/g17-inference-models-v1/qwen'
        prefill, decode = M.import_model(directory), M.import_model(directory, tokens=1)
        self.assertEqual(prefill['used_parameters'], decode['used_parameters'])
        self.assertEqual(len(prefill['nodes']), len(decode['nodes']))
        self.assertEqual(prefill['tensors']['logits']['shape'], [32, 151936])
        self.assertEqual(decode['tensors']['logits']['shape'], [1, 151936])
        self.assertEqual(prefill['required_parameter_bytes'], 988065536)
        self.assertEqual(prefill['kv_state_bytes'], 6291456)
        self.assertEqual(sum(n['op'] == 'attention' for n in prefill['nodes']), 24)
        self.assertFalse(prefill['gpu_admitted'])

    def test_tied_embedding_is_not_duplicate_storage(self):
        g = M.import_model(ROOT / 'evidence/g17-inference-models-v1/qwen')
        logits = next(n for n in g['nodes'] if n['name'] == 'logits')
        self.assertEqual(logits['inputs'][1], 'model.embed_tokens.weight')
        self.assertNotIn('lm_head.weight', g['tensors'])

    def test_cache_update_precedes_attention_with_explicit_dependency(self):
        g = M.import_model(ROOT / 'evidence/g17-inference-models-v1/qwen', tokens=1)
        for i in range(24):
            update = next(n for n in g['nodes'] if n['name'] == f'layer.{i}.cache_update')
            attn = next(n for n in g['nodes'] if n['name'] == f'layer.{i}.attention')
            self.assertEqual(attn['control_dependencies'], [update['name']])
            self.assertEqual(attn['attrs']['heads'], 14)
            self.assertEqual(attn['attrs']['kv_heads'], 2)
            self.assertTrue(attn['attrs']['causal'])
            self.assertEqual(update['attrs']['writes'], [f'layer.{i}.cache_k', f'layer.{i}.cache_v'])

    def test_config_cannot_silently_truncate_layers(self):
        _, cfg, params, _ = self.load('qwen')
        cfg['num_hidden_layers'] = 1
        with self.assertRaisesRegex(I.Unsupported, 'unconsumed'):
            I.import_graph(cfg, params)

    def test_missing_layer_parameter_refused(self):
        _, cfg, params, _ = self.load('minilm')
        del params['encoder.layer.5.output.dense.weight']
        with self.assertRaisesRegex(I.InvalidCheckpoint, 'missing model parameter'):
            I.import_graph(cfg, params)

    def test_transposed_weight_shape_not_silently_accepted(self):
        _, cfg, params, _ = self.load('qwen')
        name = 'model.layers.0.mlp.down_proj.weight'
        p = params[name]
        params[name] = I.Parameter(p.name, p.dtype, p.shape[::-1], p.begin, p.end)
        with self.assertRaisesRegex(I.InvalidCheckpoint, 'shape/type'):
            I.import_graph(cfg, params)

    def test_new_architecture_and_activation_refused(self):
        _, cfg, _, _ = self.load('qwen')
        for field, value in [('model_type', 'llama'), ('hidden_act', 'relu'),
                             ('architectures', ['Qwen2ForSequenceClassification'])]:
            bad = copy.deepcopy(cfg);bad[field] = value
            with self.assertRaises(I.Unsupported):I.profile(bad)

    def test_unmeasured_rope_and_window_not_imported(self):
        _, cfg, _, _ = self.load('qwen')
        for field, value in [('rope_scaling', {'type': 'dynamic'}), ('use_sliding_window', True)]:
            bad = copy.deepcopy(cfg);bad[field] = value
            with self.assertRaisesRegex(I.Unsupported, 'sliding-window or scaled RoPE'):I.profile(bad)

    def test_head_grouping_and_cache_capacity_refused(self):
        _, cfg, params, _ = self.load('qwen')
        bad = copy.deepcopy(cfg);bad['num_key_value_heads'] = 3
        with self.assertRaisesRegex(I.Unsupported, 'grouping'):I.profile(bad)
        with self.assertRaisesRegex(I.Unsupported, 'KV capacity'):I.import_graph(cfg, params, cache_capacity=16)

    def test_sentence_wrapper_limit_is_not_bert_position_table_limit(self):
        with self.assertRaisesRegex(I.Unsupported, 'sentence wrapper sequence limit'):
            M.import_model(ROOT / 'evidence/g17-inference-models-v1/minilm', tokens=257, sentence_pooling=True)

    def test_unrecognized_extra_tensor_not_silently_discarded(self):
        _, cfg, params, _ = self.load('minilm')
        params['unrecognized.weight'] = I.Parameter('unrecognized.weight', 'F32', (1,), 0, 4)
        with self.assertRaisesRegex(I.Unsupported, 'unconsumed'):
            I.import_graph(cfg, params)


if __name__ == '__main__':
    unittest.main()
