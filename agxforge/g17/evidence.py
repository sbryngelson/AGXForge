"""Pack retained experiments once; materialize verified fixtures only when needed.

    python3 tools/g17evidence.py extract [results/experiment-prefix]
    python3 tools/g17evidence.py check
    python3 tools/g17evidence.py pack results/experiment ...

The archive is a committed input, not an external download or a trusted cache.
Every member keeps its original path, length and SHA256. Extraction never
overwrites a differing file. Production source and ISA specifications stay loose.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile
import fnmatch
import functools
import subprocess

ROOT = Path(__file__).resolve().parents[2]

# THE PROVENANCE PATHS OF THIS PROJECT'S OWN ASSEMBLER, in both spellings.
#
# An execution receipt attributes an object to this backend by naming one of these among its
# `source.inputs`. Every retained receipt - 110 of them - names the tools/ spelling, because that
# is where the assembler lived when they were written, and those receipts can never be rewritten:
# the object they describe was built then. A receipt produced now records the package path, and a
# matcher that knows only the legacy spelling SKIPS it - `continue`, no error - so the attributed
# set silently shrinks as new evidence arrives.
#
# Both spellings are therefore accepted, and the legacy ones are kept rather than translated.
# This lives here, in one place, because the same three-path tuple was inline in two tools and a
# matcher repaired in one of them would not have travelled to the other.
OUR_ASSEMBLER = ("tools/g17asm.py", "tools/g17cc.py", "tools/g17as.py")


def assembler_provenance_paths(root=None):
    """Every path a receipt may name to attribute an object to this project's assembler."""
    from . import compat
    paths = []
    for legacy in OUR_ASSEMBLER:
        paths.append(legacy)
        implementation = compat.source_of(legacy, root)
        if implementation != legacy:
            paths.append(implementation)
            paths.extend(compat.legacy_spellings(implementation))   # receipts written before the rename
    return tuple(paths)


def with_implementations(paths, root=None):
    """The given paths, plus the file that actually implements any forwarding entry among them.

    The legacy path is KEPT: a retained report's entry stays comparable, and the shim is a real
    input to a build that imports it. What it cannot do any more is notice a change to the
    compiler, which is what the added path is for.
    """
    from . import compat
    out = []
    for path in paths:
        text = str(path)
        out.append(text)
        implementation = compat.source_of(text, root)
        if implementation != text and implementation not in out:
            out.append(implementation)
    return tuple(out)


def source_identities(paths, root=None):
    """{path: sha256} for each input AND for the implementation behind a forwarding entry.

    A provenance field exists so a later reader can tell whether a result's inputs changed under
    it. Once a module moves into the package, hashing only the legacy path fingerprints a
    forwarding entry - a file that CANNOT change when the compiler does - so the audit keeps
    passing while the thing it audits is rewritten. Hashing only the implementation would be the
    opposite error: a retained report's own entry would no longer be comparable.

    So both are recorded. The legacy key keeps its meaning for every report already written, and
    the package key is what makes a changed implementation invalidate the audit.
    """
    return {path: _digest_of(path, root) for path in with_implementations(paths, root)}


def _digest_of(path, root=None):
    base = Path(root) if root is not None else ROOT
    return hashlib.sha256((base / path).read_bytes()).hexdigest()


def attributes_to_our_assembler(inputs, root=None):
    """Whether a receipt's `source.inputs` names this project's assembler.

    Path recognition only. It says which receipts are OURS to look at; it is not evidence that
    anything executed - the caller still reads the recorded object digests for that.
    """
    return any(path in inputs for path in assembler_provenance_paths(root))
ARCHIVE = "evidence/g17-generality.zip"
INDEX = "MANIFEST.json"


@functools.lru_cache(maxsize=4)
def _paths_at(root, revision):
    paths = subprocess.check_output(
        ["git", "-C", root, "ls-tree", "-r", "--name-only", revision], text=True).splitlines()
    if ARCHIVE in paths:
        blob = subprocess.check_output(["git", "-C", root, "show", revision + ":" + ARCHIVE])
        paths.extend(members(blob))
    return tuple(sorted(set(paths)))


def tracked_paths(root=ROOT, pattern="*"):
    """Logical committed paths: packing must not shrink an evidence population."""
    root = str(Path(root).resolve())
    revision = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    return [p for p in _paths_at(root, revision) if fnmatch.fnmatchcase(p, pattern)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def valid_path(name):
    p = PurePosixPath(name)
    if (not name.startswith("results/") or p.as_posix() != name or
            any(x in ("", ".", "..") for x in name.split("/")) or
            "\\" in name or "\n" in name or "\r" in name):
        raise ValueError("invalid evidence member path: " + repr(name))
    return name


def members(blob, paths=None):
    """Read and verify archive members, including its complete indexed namespace."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate evidence member")
        manifest = json.loads(archive.read(INDEX))
        if manifest.get("format") not in ("g17-evidence-v1", "g17-evidence-v2"):
            raise ValueError("unknown evidence archive format")
        index = manifest["members"]
        compact = manifest["format"] == "g17-evidence-v2"
        payload = lambda name: "blobs/" + index[name]["sha256"] if compact else name
        if set(names) != {payload(name) for name in index} | {INDEX}:
            raise ValueError("evidence index differs from archive members")
        for name in index:
            valid_path(name)
            if 'executable' in index[name] and type(index[name]['executable']) is not bool:
                raise ValueError('invalid executable flag: ' + name)
        selected = index if paths is None else paths
        out = {}
        for name in selected:
            if name not in index:
                raise ValueError("input is not a committed blob or evidence member: " + name)
            data = archive.read(payload(name))
            entry = dict(index[name])
            entry.pop("executable", None)
            if entry != {"bytes": len(data), "sha256": sha(data)}:
                raise ValueError("evidence member hash or size differs: " + name)
            out[name] = data
        return out


def destinations(blob):
    """The original paths an archive records, from its index alone.

    `members()` reads and verifies payloads; this answers only WHICH destinations a store holds,
    which is what a caller needs to route a path to the store that owns it before verifying it.
    """
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        return set(json.loads(archive.read(INDEX))["members"])


def packed(files, executables=(), *, compact=True):
    """Stable member order, compression, timestamps and permissions.

    THE EXECUTABLE BIT IS RECORDED IN THE INDEX, NOT IN THE ZIP ENTRY. Every member is written with
    the same `external_attr` so the archive bytes stay deterministic - that part was deliberate and
    is kept - but normalising the mode away entirely meant a retained WORKER BINARY came back from
    extraction at 0644 and could not be run: `PermissionError: ... /common-worker` out of
    subprocess, in every test that executes retained evidence. 123 of the packed members were
    committed at 100755. A flag in the index costs nothing, stays deterministic and reviewable in
    the manifest, and restores fidelity where it matters.
    """
    executables = set(executables)
    if executables - files.keys():
        raise ValueError('executable member is absent from payload')
    index = {}
    for name, data in files.items():
        valid_path(name)
        index[name] = {"bytes": len(data), "sha256": sha(data)}
        if name in executables:
            index[name]["executable"] = True
    entries = {"blobs/" + sha(data): data for data in files.values()} if compact else dict(files)
    entries[INDEX] = (json.dumps({"format": "g17-evidence-v2" if compact else "g17-evidence-v1", "members": index},
                                sort_keys=True, separators=(",", ":")) + "\n").encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=6)
    return output.getvalue()


def executable_members(blob):
    """The members the index records as executable."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        manifest = json.loads(archive.read(INDEX))
    return {name for name, entry in manifest["members"].items() if entry.get("executable")}


def merge_archives(base, left, right):
    """Three-way merge of verified member contents and executable flags.

    Independent edits combine; conflicting bytes, modes, or delete/edit pairs
    refuse with the member's name. No side is selected by branch ownership.
    """
    def records(blob):
        data = members(blob)
        executable = executable_members(blob)
        return {name: (raw, name in executable) for name, raw in data.items()}
    def choose(base, left, right, name):
        if left == right or right == base:
            return left
        if left == base:
            return right
        raise ValueError('conflicting evidence changes: ' + name)
    b, l, r = map(records, (base, left, right))
    result = {}
    for name in sorted(b.keys() | l.keys() | r.keys()):
        before, ours, theirs = b.get(name), l.get(name), r.get(name)
        if before is not None and ours is not None and theirs is not None:
            merged = tuple(choose(x, y, z, name) for x, y, z in zip(before, ours, theirs))
        else:
            merged = choose(before, ours, theirs, name)
        if merged is not None:
            result[name] = merged
    return packed({name: value[0] for name, value in result.items()},
                  {name for name, value in result.items() if value[1]}, compact=True)


# PER-LANE STORES, beside the shared one. evidence/g17-generality.zip reached 99.7 MB on 2026-09-23 and
# GitHub refuses a file over 100 MB, so the next pack by any lane would have been unpushable; and one
# shared binary archive forced a three-way archive merge at nearly every merge. A lane packs its new
# result sets into its own store; extraction, status and mmlint read every store listed here.
LANE_ARCHIVES = ("evidence/g17-seta.zip", "evidence/g17-setb.zip", "evidence/g17-cited-v1.zip",
                 "evidence/g17-first-dispatch-trials.zip", "evidence/g17-compiler-probes-v1.zip",
                 "evidence/g17-tensor-acc-registers-v1.zip",
                 "evidence/g17-authorship-v1.zip", "evidence/g17-belowmetal-demo-fixtures.zip")
ARCHIVES = (ARCHIVE, "evidence/g17-vendor-shader-corpus.zip") + LANE_ARCHIVES


def unextracted(root=ROOT, archives=None):
    """-> {archive: (missing, total)} for every store whose members are not all on disk.

    THE MOST EXPENSIVE FALSE ALARM IN THIS REPOSITORY, and it has no cheap tell. An unextracted
    tree does not look empty - it looks BROKEN. `g17pinnedfiles.stale()` returns 547 stale pins,
    `g17capcompiler.py check` refuses on a missing evidence file, and a bare `unittest discover`
    is ~1,365 errors of missing fixtures. Every one of those reads as a catastrophic regression
    and every one of them is a checkout nobody ran `make evidence` in.

    It cost four false reds in one session, each chased for minutes, by someone who had written
    the note saying this happens. A note is not a check. This is the check: it answers "is this
    tree prepared" in one call, cheaply, so a tool can say `run make evidence` instead of
    reporting 547 of something.
    """
    # AN ABSENT STORE IS NOT AN EXTRACTED ONE. This skipped a store whose archive file was missing,
    # so a checkout without evidence/*.zip - sparse, or a deleted file - answered {"extracted":
    # true} and exited 0, the one tree in which nothing it guards could be on disk. It is now
    # reported as (None, None): members unknown, because the list of members lives in the archive.
    out = {}
    root = Path(root).resolve()
    for name in (ARCHIVES if archives is None else archives):
        store = root / name
        if not store.exists():
            out[name] = (None, None)
            continue
        declared = members(store.read_bytes())
        missing = [p for p in declared if not (root / p).exists()]
        if missing:
            out[name] = (len(missing), len(declared))
    return out


def refuse_if_unextracted(root=ROOT, what="this tool"):
    """Raise SystemExit naming the fix, or return quietly. For callers that read `results/`."""
    gaps = unextracted(root)
    if not gaps:
        return
    lines = ["%s needs the archived evidence on disk and it is not extracted:" % what]
    for name, (missing, total) in sorted(gaps.items()):
        lines.append("    %-44s %s" % (name, "the archive itself is absent" if total is None
                                       else "%d of %d members missing" % (missing, total)))
    lines.append("")
    lines.append("    make evidence            # the default store")
    lines.append("    python3 tools/g17evidence.py extract --archive %s"
                 % "evidence/g17-vendor-shader-corpus.zip")
    lines.append("")
    lines.append("Without it this reports failures that are about the CHECKOUT, not the code.")
    raise SystemExit("\n".join(lines))


def extract(root=ROOT, prefixes=(), archive=None):
    """Materialize archived evidence.

    `archive` selects which store to read; it defaults to ARCHIVE. A second store exists because
    the archive is a COMMITTED file and GitHub refuses one over 100 MB: g17-generality.zip was
    already at 92 MiB, so the vendor shader corpus could not go in it and lives in
    evidence/g17-vendor-shader-corpus.zip in the identical format.
    """
    root = Path(root).resolve()
    blob = (root / (archive or ARCHIVE)).read_bytes()
    data = members(blob)
    runnable = executable_members(blob)
    if prefixes:
        prefixes = [valid_path(x.rstrip("/")) for x in prefixes]
        data = {p: b for p, b in data.items()
                if any(p == x or p.startswith(x + "/") for x in prefixes)}
        if not data:
            raise ValueError("no archived evidence matches the requested prefix")
    # Validate every destination before writing any, including symlink escapes.
    for name, content in data.items():
        target = root / name
        if not target.resolve().is_relative_to(root):
            raise ValueError("evidence destination escapes repository: " + name)
        if target.exists() and (not target.is_file() or target.read_bytes() != content):
            raise ValueError("refusing to overwrite differing evidence: " + name)
    written = restored = 0
    for name, content in data.items():
        target = root / name
        if target.exists():
            if name in runnable and target.stat().st_mode & 0o111 != 0o111:
                target.chmod(target.stat().st_mode | 0o111)
                restored += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
        if name in runnable:
            target.chmod(target.stat().st_mode | 0o111)
            restored += 1
        written += 1
    return {"verified_members": len(data), "extracted": written,
            "restored_executable": restored}


def search(blob, query, *, pattern="*", limit=100):
    """Literal UTF-8 text search over verified evidence, without materialization.

    Results identify original paths and line numbers. Historical prose remains
    historical evidence: a match does not promote its claims to current support.
    """
    if not isinstance(query, str) or not query:
        raise ValueError("search needs a nonempty literal query")
    if type(limit) is not int or limit < 1:
        raise ValueError("search limit must be a positive integer")
    hits, total, scanned = [], 0, 0
    for name, data in sorted(members(blob).items()):
        if not fnmatch.fnmatchcase(name, pattern) or b"\0" in data:
            continue
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        for number, line in enumerate(content.splitlines(), 1):
            if query not in line:
                continue
            total += 1
            if len(hits) < limit:
                start = max(0, line.index(query) - 100)
                hits.append(dict(path=name, line=number, sha256=sha(data),
                    text=line[start:start + 500], text_truncated=len(line) > 500))
    return dict(query=query, pattern=pattern, scanned_utf8_files=scanned,
                matching_lines=total, truncated=total > len(hits), matches=hits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",
                        choices=("extract", "check", "pack", "merge", "search", "status"))
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--pattern", default="*", help="search original paths with this glob")
    parser.add_argument("--limit", type=int, default=100, help="maximum returned search lines")
    parser.add_argument("--archive", default=None,
                        help="evidence store to read or write (default: %s)" % ARCHIVE)
    args = parser.parse_args()
    destination = ROOT / (args.archive or ARCHIVE)
    if args.action == "status":
        gaps = unextracted(ROOT)
        if not gaps:
            print(json.dumps({"extracted": True}))
            return 0
        print(json.dumps({"extracted": False,
                          "missing": {k: ("archive absent" if v[1] is None else v[0]) for k, v in gaps.items()},
                          "run": "make evidence"}))
        return 1
    if args.action == "search":
        result = search(destination.read_bytes(), " ".join(args.paths),
                        pattern=args.pattern, limit=args.limit)
    elif args.action == "merge":
        if len(args.paths) != 3:
            parser.error('merge requires BASE LEFT RIGHT archive paths; writes LEFT')
        base, destination, right = map(Path, args.paths)
        blob = merge_archives(base.read_bytes(), destination.read_bytes(), right.read_bytes())
        # Construct and validate the whole result before changing the merge target.
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(blob)
        os.replace(temporary, destination)
        result = {'members': len(members(blob)), 'sha256': sha(blob)}
    elif args.action == "extract":
        result = extract(prefixes=args.paths, archive=args.archive)
    elif args.action == "check":
        result = {"verified_members": len(members(destination.read_bytes()))}
    else:
        if not args.paths:
            parser.error("pack requires explicit experiment paths")
        existing = destination.read_bytes() if destination.exists() else None
        files = members(existing) if existing else {}
        # Flags already recorded stay recorded; a member re-packed from disk takes the mode it has
        # on disk now, so a bit that was never set cannot be invented here.
        runnable = executable_members(existing) if existing else set()
        for arg in args.paths:
            path = ROOT / valid_path(arg.rstrip("/"))
            for entry in sorted(path.rglob("*")) if path.is_dir() else [path]:
                if entry.is_file():
                    if entry.is_symlink() or not entry.resolve().is_relative_to(ROOT):
                        raise ValueError("refusing linked evidence: " + str(entry))
                    member = entry.relative_to(ROOT).as_posix()
                    files[member] = entry.read_bytes()
                    if entry.stat().st_mode & 0o111:
                        runnable.add(member)
                    else:
                        runnable.discard(member)
        blob = packed(files, runnable, compact=True)
        members(blob)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(blob)
        os.replace(temporary, destination)
        result = {"members": len(files), "archive_bytes": len(blob), "sha256": sha(blob)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
