"""Fetch and verify pinned safetensors weights; never compile or dispatch them."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time
import urllib.request

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g17modelimport import load

CHUNK = 1024 * 1024


def _verify_stream(stream, manifest, header):
    before = os.fstat(stream.fileno())
    if not stat.S_ISREG(before.st_mode) or before.st_size != manifest['checkpoint_bytes']:
        raise ValueError('checkpoint size or file type mismatch')
    stream.seek(0)
    if int.from_bytes(stream.read(8), 'little') != len(header) or stream.read(len(header)) != header:
        raise ValueError('checkpoint header differs from pinned metadata')
    stream.seek(0)
    digest = hashlib.sha256()
    while block := stream.read(CHUNK):
        digest.update(block)
    after = os.fstat(stream.fileno())
    if _identity(before) != _identity(after):
        raise ValueError('checkpoint changed while verifying')
    if digest.hexdigest() != manifest['expected_checkpoint_sha256']:
        raise ValueError('checkpoint SHA256 mismatch')
    return dict(format='g17-checkpoint-payload-verification-v1', repository=manifest['repository'],
                revision=manifest['revision'], bytes=before.st_size, sha256=digest.hexdigest(),
                weight_payload_verified=True, gpu_admitted=False)


def _identity(value):
    return value.st_size, value.st_mtime_ns, value.st_ctime_ns


def verify(path, manifest, header):
    """Hash an opened regular file and compare the actual header and payload."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        return _verify_stream(stream, manifest, header)


class CheckpointReader:
    """Verify one opened checkpoint once, then read named tensors from that inode.

    A tensor never reopens the path or hashes the whole model again. Size/mtime/
    ctime checks surround each traversal and closing the reader. Concurrent
    mutation refuses; this is not a filesystem snapshot against hostile writers.
    """
    def __init__(self, path, manifest, header):
        from agxforge.g17.inferencegraph import checkpoint_header
        self.path, self.manifest, self.header = path, manifest, header
        self.parameters = checkpoint_header(header, file_bytes=manifest['checkpoint_bytes'])
        self.stream = None

    def __enter__(self):
        fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        self.stream = os.fdopen(fd, 'rb')
        try:
            self.receipt = _verify_stream(self.stream, self.manifest, self.header)
            self.identity = _identity(os.fstat(fd))
        except BaseException:
            self.stream.close()
            self.stream = None
            raise
        return self

    def check(self):
        if self.stream is None:
            raise ValueError('checkpoint reader is closed')
        if _identity(os.fstat(self.stream.fileno())) != self.identity:
            raise ValueError('checkpoint changed during tensor read')

    def chunks(self, name, *, chunk_bytes=CHUNK):
        if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= CHUNK:
            raise ValueError('tensor chunk size must be 1..1 MiB')
        self.check()
        parameter = self.parameters[name]
        offset = 8 + len(self.header) + parameter.begin
        remaining = parameter.bytes
        while remaining:
            block = os.pread(self.stream.fileno(), min(remaining, chunk_bytes), offset)
            if not block:
                raise ValueError('checkpoint truncated during tensor read')
            offset += len(block)
            remaining -= len(block)
            yield block
        self.check()

    def small_tensor(self, name, *, limit=4 * CHUNK):
        if self.parameters[name].bytes > limit:
            raise ValueError('refused: tensor exceeds bounded conversion buffer')
        return b''.join(self.chunks(name))

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self.check()
        finally:
            self.stream.close()
            self.stream = None


def fetch(path, manifest, header, *, opener=urllib.request.urlopen, deadline_seconds=600):
    path = Path(path)
    # Refuse symlink ancestors as well as a symlink destination. Never replace a file.
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('refused: symlink checkpoint destination')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return verify(path, manifest, header)
    url = 'https://huggingface.co/{}/resolve/{}/model.safetensors'.format(manifest['repository'], manifest['revision'])
    start = time.monotonic()
    fd, temporary = tempfile.mkstemp(prefix='.checkpoint-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as output, opener(url, timeout=15) as response:
            total = 0
            while block := response.read(CHUNK):
                if time.monotonic() - start > deadline_seconds:
                    raise TimeoutError('checkpoint download deadline exceeded')
                total += len(block)
                if total > manifest['checkpoint_bytes']:
                    raise ValueError('download exceeds pinned checkpoint size')
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        receipt = verify(temporary, manifest, header)
        # Exclusive publication: a concurrent creator is never overwritten.
        os.link(temporary, path)
        return receipt
    finally:
        os.unlink(temporary)


def tensor_chunks(path, manifest, header, parameter, *, chunk_bytes=CHUNK):
    """Read a verified tensor in bounded raw chunks, without dtype conversion.

    Reverification precedes each traversal. Consumers may stream into their
    eventual measured upload/conversion path; this never creates activations.
    """
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= CHUNK:
        raise ValueError('tensor chunk size must be 1..1 MiB')
    verify(path, manifest, header)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        # Verify this opened inode too, rather than trusting the pathname after
        # a concurrent replacement. The digest covers the full payload.
        digest = hashlib.sha256()
        while block := stream.read(CHUNK):
            digest.update(block)
        if digest.hexdigest() != manifest['expected_checkpoint_sha256']:
            raise ValueError('checkpoint replaced before tensor read')
        payload_base = 8 + len(header)
        if not 0 <= parameter.begin <= parameter.end <= before.st_size - payload_base:
            raise ValueError('tensor extent outside checkpoint')
        stream.seek(payload_base + parameter.begin)
        remaining = parameter.end - parameter.begin
        while remaining:
            block = stream.read(min(remaining, chunk_bytes))
            if not block:
                raise ValueError('checkpoint truncated during tensor read')
            remaining -= len(block)
            yield block
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('checkpoint changed during tensor read')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    manifest, _, _, blobs = load(args.metadata)
    action = fetch if args.fetch else verify
    receipt = action(args.checkpoint, manifest, blobs['safetensors-header.json'])
    with args.receipt.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
