"""Capture tokenizer files from the same immutable revision as a model manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
from g17modelimport import load

FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
         'vocab.txt', 'vocab.json', 'merges.txt')
LIMIT = 16 * 1024 * 1024


def read(url):
    with urllib.request.urlopen(url, timeout=15) as response:
        raw = response.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError('refused: tokenizer file exceeds 16 MiB')
    return raw


def capture(metadata, destination):
    manifest, _, _, _ = load(metadata)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('tokenizer capture requires a new directory')
    api = json.loads(read(manifest['source_api']))
    if api['sha'] != manifest['revision']:
        raise ValueError('repository API revision mismatch')
    siblings = {s['rfilename']: s for s in api['siblings']}
    captured = {}
    blobs = {}
    for name in FILES:
        if name not in siblings:
            continue
        raw = read('https://huggingface.co/{}/resolve/{}/{}'.format(
            manifest['repository'], manifest['revision'], urllib.parse.quote(name)))
        sibling = siblings[name]
        if len(raw) != sibling['size']:
            raise ValueError('tokenizer size mismatch: ' + name)
        digest = hashlib.sha256(raw).hexdigest()
        if sibling.get('lfs'):
            if digest != sibling['lfs']['sha256']:
                raise ValueError('tokenizer LFS digest mismatch: ' + name)
        else:
            git_blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            if git_blob != sibling['blobId']:
                raise ValueError('tokenizer git blob mismatch: ' + name)
        blobs[name] = raw
        captured[name] = dict(bytes=len(raw), sha256=digest, git_blob=sibling['blobId'])
    if not {'tokenizer.json', 'tokenizer_config.json'} <= blobs.keys():
        raise ValueError('required tokenizer files absent')
    result = dict(format='g17-pinned-tokenizer-v1', repository=manifest['repository'],
                  revision=manifest['revision'], files=captured)
    destination.mkdir(parents=True)
    for name, raw in blobs.items():
        with (destination / name).open('xb') as stream:
            stream.write(raw)
    with (destination / 'identity.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(capture(args.metadata, args.destination), indent=2))
