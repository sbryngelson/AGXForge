"""Cold-build input audit with one batched Git read, never a decoder fallback."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from . import evidence


def sha(data):
    return hashlib.sha256(data).hexdigest()


def committed_hashes(root, revision, paths):
    """Read all pinned Git blobs in one process, parsing binary sizes exactly."""
    paths = list(paths)
    if any("\n" in p or "\r" in p for p in paths):
        raise ValueError("newline in build input path")
    # Only results/ can contain archived inputs (evidence.valid_path enforces
    # that namespace). Source-only builds need not read the entire archive.
    # When needed, it travels in the same batch at the same pinned revision.
    archived_inputs = any(p.startswith("results/") for p in paths)
    requested = list(dict.fromkeys(paths + ([evidence.ARCHIVE] if archived_inputs else [])))
    result = subprocess.run(["git", "-C", str(root), "cat-file", "--batch"],
                            input="".join(f"{revision}:{p}\n" for p in requested).encode(),
                            capture_output=True, check=True, timeout=15)
    cursor, hashes, missing, archive = 0, {}, [], None
    for path in requested:
        end = result.stdout.find(b"\n", cursor)
        if end < 0:
            raise ValueError("truncated Git input header")
        header = result.stdout[cursor:end].split()
        if result.stdout[cursor:end].endswith(b" missing"):
            cursor = end + 1
            missing.append(path)
            continue
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError(f"build input is not a committed blob: {path}")
        size = int(header[2])
        cursor = end+1
        data = result.stdout[cursor:cursor+size]
        if len(data) != size or result.stdout[cursor+size:cursor+size+1] != b"\n":
            raise ValueError("truncated Git input body")
        if path == evidence.ARCHIVE:
            archive = data
        if path in paths:
            hashes[path] = sha(data)
        cursor += size+1
    if cursor != len(result.stdout):
        raise ValueError("unexpected trailing Git input data")
    missing = [p for p in missing if p in paths]
    if missing:
        if archive is None:
            raise ValueError("build input is not a committed blob: " + ", ".join(missing))
        for blob in _evidence_stores(root, revision, archive):
            held = sorted(set(missing) & evidence.destinations(blob))
            hashes.update({p: sha(b) for p, b in evidence.members(blob, held).items()})
            missing = [p for p in missing if p not in hashes]
            if not missing:
                break
        if missing:
            raise ValueError("input is not a committed blob or evidence member: "
                             + ", ".join(missing))
    return hashes


def _evidence_stores(root, revision, default):
    """The default store first, then every OTHER committed store - read only if still needed.

    THERE IS MORE THAN ONE STORE. evidence/g17-generality.zip reached 92 MiB and GitHub refuses a
    file over 100 MB, so the vendor shader corpus went into a second archive in the same format.
    Resolving archived inputs through `evidence.ARCHIVE` alone therefore reported every member of
    that second store as "not a committed blob or evidence member" - an instrument pointed at a
    population narrower than the one holding the evidence, which is the same defect that made the
    release guard in tools/g17frontierrelease.py diff results/ against git history.

    The extra stores are listed and read LAZILY, after the default archive has been tried, so a
    build whose inputs all live in the default store still spends exactly one Git blob process -
    which is what `verified_build` reports.
    """
    yield default
    listed = subprocess.run(["git", "-C", str(root), "ls-tree", "-r", "--name-only",
                             revision, "--", "evidence"],
                            capture_output=True, text=True, check=True, timeout=15)
    for name in listed.stdout.split("\n"):
        if name.endswith(".zip") and name != evidence.ARCHIVE:
            yield subprocess.run(["git", "-C", str(root), "cat-file", "blob", revision + ":" + name],
                                 capture_output=True, check=True, timeout=60).stdout


def verified_build(root, builder):
    return _verified(root,builder)


def verified_reference(root, builder, decoder):
    """Audit interpreted reference artifacts with one explicitly named decoder.

    This is separate from native generation: calls and instrument identity are
    recorded, and executable artifact outputs are prohibited.
    """
    return _verified(root,builder,verification_decoder=Path(decoder).resolve())


def _verified(root, builder, verification_decoder=None):
    """Run a pure builder and verify every observed repository input against HEAD.

    Production callers must enter in a fresh process before importing compiler
    or linker modules. Existing repository modules are included too, so a warm
    caller cannot hide source modules; cold tests cover data files behind caches.
    Git is consulted before and after generation, never while choosing bytes.
    """
    root = Path(root).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    observed, external, platform_inputs = {}, set(), {}
    decoder_calls=[]
    decoder_sha=sha(verification_decoder.read_bytes()) if verification_decoder else None
    active, reading = True, False
    python_roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}

    def record(path):
        nonlocal reading
        path = Path(path).resolve()
        if path.suffix == ".pyc":
            path = Path(importlib.util.source_from_cache(str(path))).resolve()
        if not path.is_file():
            return
        if path.is_relative_to(root):
            reading = True
            try:
                digest = sha(path.read_bytes())
            finally:
                reading = False
            rel = str(path.relative_to(root))
            if rel in observed and observed[rel] != digest:
                raise ValueError(f"build input changed between reads: {rel}")
            observed[rel] = digest
        elif str(path) == "/System/Library/CoreServices/SystemVersion.plist":
            # Python's platform detection reads this on a cold import. Record
            # the OS input with the interpreter, rather than calling it a
            # repository source or silently excluding every /System file.
            reading = True
            try:
                platform_inputs[str(path)] = sha(path.read_bytes())
            finally:
                reading = False
        elif not any(path.is_relative_to(base) for base in python_roots) and "/site-packages/" not in str(path):
            external.add(str(path))

    def hook(event, values):
        nonlocal reading
        if not active or reading:
            return
        if event in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.system"):
            if event=='subprocess.Popen' and verification_decoder is not None:
                executable,args=values[:2]
                if (isinstance(args,(list,tuple)) and len(args)==5 and
                    Path(os.fsdecode(executable)).resolve()==verification_decoder and
                    Path(os.fsdecode(args[0])).resolve()==verification_decoder and
                    args[2]=='0' and str(args[3]).isdigit() and args[4]=='--expr'):
                    reading=True
                    try:raw=Path(args[1]).read_bytes()
                    finally:reading=False
                    if Path(args[1]).resolve().is_relative_to(root):record(args[1])
                    if len(raw)!=int(args[3]):raise ValueError('reference decoder input length differs')
                    decoder_calls.append(dict(input_sha256=sha(raw),bytes=len(raw)))
                    return
            raise RuntimeError("build attempted an external process")
        if event == "ctypes.dlopen" and any(x in str(values[0]) for x in ("Metal", "AGX", "libaccel")):
            raise RuntimeError("build attempted a hardware library")
        if event != "open" or not isinstance(values[0], (str, bytes)):
            return
        if isinstance(values[1], str) and any(x in values[1] for x in "wa+"):
            return
        if values[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if "/.cache/agxforge/" in str(path):
            raise RuntimeError("build attempted an external agxforge cache")
        record(path)

    sys.addaudithook(hook)
    try:
        if verification_decoder is not None:
            source=verification_decoder.with_name(verification_decoder.name+'.c')
            if source.exists():record(source)
            record('/System/Library/CoreServices/SystemVersion.plist')
        built = builder()
        if verification_decoder is not None:
            if not isinstance(built,dict) or any(str(k).endswith(('.o','.bin','.metallib')) for k in built):
                raise ValueError('reference analysis cannot author executable artifacts')
            reading=True
            try:current=sha(verification_decoder.read_bytes())
            finally:reading=False
            if current!=decoder_sha:raise ValueError('reference decoder changed during analysis')
        for module in tuple(sys.modules.values()):
            path = getattr(module, "__file__", None)
            if path and Path(path).resolve().is_relative_to(root):
                record(path)
    finally:
        active = False
    if external:
        raise ValueError("unverified external build inputs: " + ", ".join(sorted(external)))
    committed = committed_hashes(root, revision, sorted(observed))
    for rel, digest in observed.items():
        if committed[rel] != digest or sha((root / rel).read_bytes()) != digest:
            raise ValueError(f"build input is dirty or changed: {rel}")
    if subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() != revision:
        raise ValueError("repository HEAD moved during the build")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pydantic", "pydantic_core")}
    report=dict(status="passed", commit=revision, inputs=observed,
                external_processes=len(decoder_calls), git_blob_processes=1,
                python=sys.version, dependencies=versions, platform_inputs=platform_inputs)
    if verification_decoder is not None:
        report.update(kind='reference_analysis',verification_instrument=dict(
            path=str(verification_decoder),sha256=decoder_sha,calls=decoder_calls,
            limitation='Decoder binary and OS identified; linked Apple framework behavior is an external validation instrument.'))
    return built,report
