"""Pinned CPU tokenization for native inference; no model arithmetic or GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import importlib.metadata
from g17modelimport import load

FILES = {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
         'vocab.txt', 'vocab.json', 'merges.txt'}


def validate_tokenizer(directory, model):
    directory = Path(directory)
    identity = json.loads((directory / 'identity.json').read_text())
    if identity.get('format') != 'g17-pinned-tokenizer-v1' or any(
            identity.get(k) != model[k] for k in ('repository', 'revision')):
        raise ValueError('refused: tokenizer/model identity mismatch')
    if not {'tokenizer.json', 'tokenizer_config.json'} <= identity['files'].keys():
        raise ValueError('refused: required tokenizer files absent')
    # No unpinned local configuration may influence AutoTokenizer.
    actual = {p.name for p in directory.iterdir()}
    if actual != set(identity['files']) | {'identity.json'}:
        raise ValueError('refused: unpinned tokenizer directory entries')
    for name, expected in identity['files'].items():
        if name not in FILES or (directory / name).is_symlink():
            raise ValueError('refused: unexpected tokenizer file')
        raw = (directory / name).read_bytes()
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError('refused: tokenizer file changed: ' + name)
    return identity


def prepare(metadata, tokenizer_directory, text, *, chat=False, limit=256):
    model, config, _, _ = load(metadata)
    identity = validate_tokenizer(tokenizer_directory, model)
    if not isinstance(text, str) or not text.strip():
        raise ValueError('refused: nonempty text required')
    if type(limit) is not int or not 1 <= limit <= config['max_position_embeddings']:
        raise ValueError('refused: token limit outside model context')
    if config['model_type'] == 'bert':
        wrapper = json.loads((Path(metadata) / 'sentence-config.json').read_text())
        if limit > wrapper['max_seq_length'] or chat:
            raise ValueError('refused: sentence wrapper limit or chat mode')
    elif config['model_type'] != 'qwen2':
        raise ValueError('refused: tokenizer model family')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_directory),
                                               local_files_only=True, trust_remote_code=False)
    if chat:
        encoded = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                                 tokenize=True, add_generation_prompt=True, return_dict=False)
    else:
        encoded = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    if not encoded or len(encoded) > limit:
        raise ValueError('refused: input exceeds token limit; truncation is never implicit')
    if any(type(i) is not int or not 0 <= i < config['vocab_size'] for i in encoded):
        raise ValueError('refused: token ID outside embedding vocabulary')
    return dict(format='g17-native-text-input-v1', checkpoint_revision=model['revision'],
                tokenizer=identity, tokenizer_engine={p: importlib.metadata.version(p)
                    for p in ('transformers', 'tokenizers')}, token_ids=encoded,
                position_ids=list(range(len(encoded))), attention_mask=[1]*len(encoded),
                segment_ids=[0]*len(encoded) if config['model_type'] == 'bert' else None,
                chat=chat, gpu_dispatched=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('tokenizer', type=Path)
    parser.add_argument('text')
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--limit', type=int, default=256)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.metadata, args.tokenizer, args.text, chat=args.chat, limit=args.limit)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(tokens=len(result['token_ids']), chat=args.chat, gpu_dispatched=False)))
