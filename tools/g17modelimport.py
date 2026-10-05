"""Inspect/import pinned model metadata into a logical graph, without GPU work.

The directory must contain model.json with per-file hashes, config.json and
safetensors-header.json. Import validates whole-file extents but does not read,
download, hash or execute the weight payload. Output is explicitly not lowered.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agxforge.g17 import inferencegraph as I


def load(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'model.json').read_text())
    if manifest.get('format') != 'g17-checkpoint-metadata-v1':
        raise ValueError('refused: checkpoint metadata manifest format')
    if not isinstance(manifest.get('revision'), str) or len(manifest['revision']) != 40 or any(c not in '0123456789abcdef' for c in manifest['revision']):
        raise ValueError('refused: full pinned repository revision required')
    blobs = {}
    for name, identity in manifest['files'].items():
        if name not in ('config.json', 'safetensors-header.json', 'modules.json',
                        'pooling.json', 'sentence-config.json'):
            raise ValueError('refused: unexpected metadata filename')
        raw = (directory / name).read_bytes()
        if len(raw) != identity['bytes'] or hashlib.sha256(raw).hexdigest() != identity['sha256']:
            raise ValueError('refused: pinned metadata changed: ' + name)
        blobs[name] = raw
    config = json.loads(blobs['config.json'])
    params = I.checkpoint_header(blobs['safetensors-header.json'], file_bytes=manifest['checkpoint_bytes'])
    return manifest, config, params, blobs


def import_model(directory, *, tokens=32, cache_capacity=256, sentence_pooling=False):
    manifest, config, params, blobs = load(directory)
    if sentence_pooling:
        modules = json.loads(blobs['modules.json'])
        if [m['type'] for m in modules] != ['sentence_transformers.models.Transformer',
                                           'sentence_transformers.models.Pooling',
                                           'sentence_transformers.models.Normalize']:
            raise I.Unsupported('refused: sentence module sequence')
        pool = json.loads(blobs['pooling.json'])
        if pool.get('word_embedding_dimension') != config['hidden_size'] or pool.get('pooling_mode_mean_tokens') is not True or any(pool.get(k, False) for k in pool if k.startswith('pooling_mode_') and k != 'pooling_mode_mean_tokens'):
            raise I.Unsupported('refused: pooling must be masked mean only')
        if tokens > json.loads(blobs['sentence-config.json'])['max_seq_length']:
            raise I.Unsupported('refused: sentence wrapper sequence limit')
    graph = I.import_graph(config, params, tokens=tokens, cache_capacity=cache_capacity,
                           sentence_pooling=sentence_pooling)
    graph['checkpoint'] = manifest
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--cache-capacity', type=int, default=256)
    parser.add_argument('--sentence-pooling', action='store_true')
    args = parser.parse_args()
    graph = import_model(args.directory, tokens=args.tokens, cache_capacity=args.cache_capacity,
                         sentence_pooling=args.sentence_pooling)
    with args.output.open('x') as stream:
        json.dump(graph, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=graph['status'], gpu_admitted=False, family=graph['profile']['family'],
                          nodes=len(graph['nodes']), outputs=graph['outputs'],
                          parameter_bytes=graph['required_parameter_bytes'],
                          kv_bytes=graph['kv_state_bytes'], native_operations=graph['required_native_operations']), indent=2))


if __name__ == '__main__':
    main()
