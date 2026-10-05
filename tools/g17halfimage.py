#!/usr/bin/env python3
"""ACCEPTANCE FOR THE FP16 SCAN IMAGE: four gates that cannot substitute for one another.

An image was recorded here as "built and structurally checked" and its delivered CODE is wrong: the
final half store at +0x991e carries the 800-byte displacement of the h[400 + row] probe template
where the scan needs zero, and a 33-half score buffer is 66 bytes, so that displacement cannot
address the outputs at all. Every structural check passed. The archive hash matched, the metadata
was byte-exact against the executed scalar class, the delivered binding contract was right, and a
reversed contract was refused.

    NONE OF THAT LOOKS AT AN INSTRUCTION.

So the gates are separated and ordered, and a later one passing can never excuse an earlier one
failing:

    1 PROVENANCE   do the inputs exist in HEAD and match the working tree? A clean `git status` is
                   NOT this check - it says nothing about whether HEAD's content is what is on
                   disk, and a file can be absent from HEAD while the tree is clean.
    2 STRUCTURE    archive parses, object round-trips, metadata valid, binding contract delivered
    3 CODE         the delivered instruction memory: forms, lengths, operands, displacements
    4 ABI          compiler and linker agree - and CONDITIONAL while the ARCH flag is unresolved
    -  HARDWARE    never asserted here. This module dispatches nothing and creates no pipeline.

Gate 3 is the one that was missing. It is not a refinement of gate 2; it asks a different question
of a different part of the image, and gate 2 cannot see the answer.

    python3 tools/g17halfimage.py [bundle]
"""
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

# The checkout root, then the shared provenance helpers and the source resolver.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from agxforge.g17 import evidence as _prov
from agxforge.g17 import compat as _compat_src

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "spike", "accel", "re"))

BUNDLE = os.path.join(ROOT, "results", "g17-half-scan-33x384-unvalidated")
PROFILE = "scalar-buffer-two-bindings-measured-v2"
BINDINGS = [(1, 0, False), (2, 2, True)]
# The checkout whose compiler is pinned. It was a second checkout at a fixed path on one machine; the compiler lives in
# this repository now, so the default is this checkout. Pass repo= to pin another tree.
COMPILER_REPO = ROOT


class Uncommitted(Exception):
    """A NEEDED input in COMPILER_REPO differs from that repository's HEAD. Raised by
    pin_compiler_modules so a build never silently takes another checkout's work in progress."""

# The emitted store depends on the ASSEMBLER and on the corrective operand map as much as on
# the compiler: the displacement fix is a re-fit of op17193 operand 7 that excludes the two
# instruction-length bits, and it lives in a data overlay. An artifact built from an
# uncommitted overlay is no more reproducible than one built from an uncommitted compiler.
# THE IMPLEMENTATIONS ARE INPUTS TOO, for the same reason the data files below are. Four of these
# are forwarding entries now, and a forwarding entry cannot differ from HEAD when the compiler
# does - so this audit would have reported a clean tree while the file that emits the image was
# uncommitted. The legacy paths are kept: they are real inputs to anything that imports them.
NEEDED = _prov.with_implementations((
          "tools/g17halfscan.py", "tools/g17cc.py", "tools/g17ir.py", "tools/g17as.py",
          "isa/g17-operand-maps-store.jsonl",
          # Data files are inputs too. This is the gap that started the thread: a module
          # audit cannot see a table, and the displacement fix lived in one.
          "isa/g17-form-bases.json", "isa/g17-form-opcodes.json"))
MODULES = tuple(("tools/%s.py" % n, n) for n in
                ("g17halfscan", "g17cc", "g17ir", "g17as", "g17asm", "g17auth", "g17forms",
                 "g17tensor", "agxdis", "g17cf", "g17bases", "g17encode", "g17formops"))


REPOS = ((ROOT, "integration"), (COMPILER_REPO, "compiler"))


def audit_loaded(repos=REPOS):
    """Every module the build actually loaded, checked against the HEAD of whichever repo it is in.

    The generalisation of the g17asm defect, and the compiler owner named the rule it comes from:
    enumerate what you did NOT pin and ask what fills it. An unnamed dependency is supplied by
    whatever happens to be nearby, and it presents as a disagreement about behaviour rather than as
    a provenance gap - which is exactly how operand 7 came to carry 800, the form not naming it and
    the template supplying it.

    Pinning the compiler's closure fixed one half. This is the other: THIS side's author, linker and
    packagers are equally inputs to the artifact, and a dirty one of those makes the image just as
    unreproducible as a dirty compiler. Nothing verified them, and being clean today is luck rather
    than a check.
    """
    out, unverified = [], []
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None) or ""
        if not path or not os.path.isfile(path):
            continue
        path = os.path.abspath(path)
        for root, label in repos:
            if not path.startswith(os.path.abspath(root) + os.sep):
                continue
            rel = os.path.relpath(path, root)
            blob = subprocess.run(["git", "-C", root, "cat-file", "-p", "HEAD:" + rel],
                                  capture_output=True)
            disk = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if blob.returncode != 0:
                unverified.append("%s (%s) is NOT IN HEAD of %s" % (rel, name, label))
            elif hashlib.sha256(blob.stdout).hexdigest() != disk:
                unverified.append("%s (%s) DIFFERS from HEAD of %s" % (rel, name, label))
            out.append(name)
            break
    return unverified, len(out)


VALIDATION = os.path.join(ROOT, "isa", "g17-half-runtime-validation.json")


def execution_evidence(archive_sha):
    """Whether the INTEGRATION OWNER has recorded a hardware run of THIS EXACT archive.

    This gate still asserts nothing about hardware and still dispatches nothing. It reports someone
    else's record, keyed by archive hash so it cannot be read across to a different image - which is
    the mistake the negative control exists to prevent, in the opposite direction. "Hardware not
    asserted" should not be read as "no execution evidence exists" when the owner has published
    some, and it must never be read as "this image ran" when they have not.
    """
    try:
        doc = json.load(open(VALIDATION))
    except Exception:
        return None
    for stage in doc.get("stages") or []:
        if stage.get("archive_sha256") == archive_sha:
            return {"queries": stage.get("gpu_queries"),
                    "bit_exact": stage.get("returned_half_bit_exact"),
                    "shape": "%sx%s" % (stage.get("rows"), stage.get("columns")),
                    "status": doc.get("status"),
                    "arch_true_variant_executed": doc.get("arch_true_variant_executed"),
                    "record": os.path.relpath(VALIDATION, ROOT)}
    return None


def anchor_check():
    """Is the tool that is running the tool that was invoked?

    pin_local_modules() pins from HERE, and HERE is this file's own directory. Both trees now carry
    g17halfimage.py, so in a warm process `import g17halfimage` can resolve to the OTHER checkout -
    and then the pinning faithfully pins that checkout's modules. The instrument that enforces
    provenance had no provenance of its own, and it failed in the one direction that cannot be seen
    from inside: everything it reports is true of a tree nobody asked about.

    So the anchor is checked against the repository the caller is actually in, taken from git rather
    than from any module's __file__. Identical file contents today is not a defence; it is the
    reason this went unnoticed.
    """
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                             text=True, cwd=os.getcwd())
        if top.returncode != 0:
            return None
        root = os.path.realpath(top.stdout.strip())
    except OSError:
        return None
    mine = os.path.realpath(ROOT)
    if root != mine:
        return ("this module was loaded from %s but the working directory belongs to %s, so every "
                "provenance answer below would be about the wrong checkout" % (mine, root))
    return None


def conflicted(path):
    """A file with merge-conflict markers is not a verified input, and it is not a syntax error.

    The acceptance path crashed on one: another party's merge left tools/g17halfcheck.py with both
    sides in it, and gate 3 died with SyntaxError halfway through a run whose first two gates had
    already printed "pass". A gate that cannot run has not passed, and it must say which file and
    why rather than raising from an import.
    """
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if line.startswith("<<<<<<< ") or line.startswith(">>>>>>> "):
                    return True
    except OSError:
        return False
    return False


# g17cc's whole import closure, in dependency order. Pinning only the three files whose names
# appear in the defect was not enough: g17asm differed between the trees, resolved from this
# worktree, and selection failed with "mul with an immediate in slot A" on a program the same
# sources compile cleanly in the compiler repo. A build is reproducible against the compiler's
# HEAD or it is not; there is no partial version of that.
# THIS SIDE'S OWN CLOSURE, and it needs pinning for the same reason the compiler's did. 201 module
# names exist in BOTH tools/ directories and at least eleven differ, including g17authorobj and
# g17mdgen - the two that author the metadata. Which copy a build gets is decided by whatever
# inserted a path first, so importing an unrelated tool ahead of the acceptance path silently
# swapped in the compiler repo's g17mdgen, whose build() has no restore_swept parameter. That one
# failed loudly. A copy whose signature happened to match would have authored different bytes in
# silence, which is the same defect wearing a quieter coat.
LOCAL = ("g17arc", "g17authorobj", "g17canon", "g17const", "g17container", "g17emit",
         "g17halfcheck", "g17imgconst_scalar", "g17ldmd", "g17link", "g17mdgen", "g17mtlb",
         "g17obj", "g17opclass", "g17packedcheck", "g17ref", "g17resource", "g17scalarabi",
         "g17scan", "g17scanlink", "g17schema", "g17verify")


def caller_tools():
    """The tools/ directory of the repository the CALLER is in, from git, not from __file__.

    HERE is this file's own directory, and both checkouts now carry g17halfimage.py, so a warm
    process can import the other one - after which HERE, ROOT, and every module this pins are that
    other tree's. The failure is invisible from inside: the answers are all true, about a checkout
    nobody asked about. Anchoring on the caller's repository is what makes the pin mean "this one".
    """
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                           cwd=os.getcwd())
        if r.returncode == 0:
            cand = os.path.join(os.path.realpath(r.stdout.strip()), "tools")
            if os.path.isdir(cand):
                return cand
    except OSError:
        pass
    return None


def pin_local_modules(where=None):
    """Freeze this side's authoring, linking and checking modules to THE CALLER'S checkout."""
    return _pin(LOCAL, where or caller_tools() or HERE)


def _pin(names, directory):
    done = []
    for name in names:
        path = os.path.join(directory, name + ".py")
        if not os.path.exists(path):
            continue
        saved = list(sys.path)
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            done.append(name)
        finally:
            sys.path[:] = saved
    return done


# DEPENDENCY ORDER MATTERS, and getting it wrong duplicates a module rather than replacing it.
# g17as imports g17encode, so pinning g17as first created instance one, and pinning g17encode after
# it created instance two - same file, two copies of module state. The build was correct and the
# MEASUREMENT was not: g17encode.bases_misses() is the compiler owner's designed instrument for
# "was the decoder consulted", and read from the idle instance it answers empty no matter what the
# other one did. So the leaves come first.
PINNED = ("g17ir", "g17bases", "g17formops", "g17encode", "g17asm", "g17as", "g17auth",
          "g17forms", "g17tensor", "agxdis", "g17cf", "g17cc", "g17halfscan")


def pin_compiler_modules(repo=COMPILER_REPO):
    """Load the compiler's modules from the repo whose HEAD gate 1 verifies, by explicit path.

    Otherwise the two questions gate 1 asks can have different answers for a reason no one intends:
    tools/g17halfscan.py inserts its own directory at sys.path[0] when it is imported, so whichever
    copy of it loads first decides where g17cc comes from afterwards. That is how this worktree's
    g17cc - staged, uncommitted, carrying an unrelated edit - would supply the compiler for an
    image whose provenance line says the verified HEAD blob.

    Loading by path rather than by search makes "the file verified is the file built with" true by
    construction. sys.path is restored around each load so that this side's own linker and author
    keep resolving from this worktree, which is where they belong.
    """
    # This side first, then the compiler's: a compiler module that imports a shared name must get
    # the copy this checkout intends, not whichever the search path reaches.
    pin_local_modules()
    # AND REFUSE AN UNCOMMITTED ONE, which is this function's own doctrine turned on itself. COMPILER_REPO
    # is an absolute path to another checkout, and that checkout has another owner working in it: when
    # their tools/g17cc.py differs from their HEAD, this loader was silently building with their
    # work in progress. It cost a confusing failure in test_g17halfimage, where the compiler raised
    # a refusal whose message does not exist anywhere in THIS tree - integration had dispatched the
    # half-store family and broadened the guard, mid-edit, and my suite compiled with it.
    #
    # The comment fifty lines up already says "an artifact built from an uncommitted overlay is no
    # more reproducible than one built from an uncommitted compiler". in_head() is right here. So
    # the check that was written for the inputs now also covers the loader that reads them.
    dirty = []
    for rel in NEEDED:
        try:
            present, matches = in_head(rel, repo)
        except Exception as ex:
            dirty.append("%s (%s: %s)" % (rel, type(ex).__name__, ex)); continue
        if not present or not matches:
            dirty.append("%s (%s)" % (rel, "absent from HEAD" if not present else "differs from HEAD"))
    if dirty:
        raise Uncommitted(
            "refusing to pin the compiler from %s: %s. That repository is a different checkout with "
            "its own owner; building from its working tree makes this image depend on an edit no "
            "commit here records. Wait for their commit, or point COMPILER_REPO at a clean tree."
            % (repo, "; ".join(dirty)))
    return _pin(PINNED, os.path.join(repo, "tools"))


def in_head(rel, repo):
    """(present_in_HEAD, matches_working_tree). A clean status is not this question.

    `git status` reports whether the tree differs from the index; it says nothing about a file that
    is absent from HEAD entirely, and an untracked file leaves a clean status for every path that
    IS tracked. So HEAD's blob is read and compared to the bytes on disk.
    """
    blob = subprocess.run(["git", "-C", repo, "cat-file", "-p", "HEAD:" + rel],
                          capture_output=True)
    if blob.returncode != 0:
        return False, False
    path = os.path.join(repo, rel)
    if not os.path.exists(path):
        return True, False
    return True, hashlib.sha256(blob.stdout).digest() == \
        hashlib.sha256(open(path, "rb").read()).digest()


def gate_provenance(repo=COMPILER_REPO):
    """Two separate questions, because passing the first one does not answer the second.

    (a) Do the required compiler sources exist in the compiler repo's HEAD, and do its working
    files match those blobs? A clean `git status` answers neither: it says nothing about a file
    absent from HEAD, and an untracked file leaves every tracked path clean.

    (b) Is the file this process would actually IMPORT that same blob? It usually is not. The
    files verified in (a) live in the compiler repo; `import g17ir` resolves against this
    checkout's sys.path. Verifying one file and running a different one is the exact substitution
    this gate exists to prevent, so the identity is checked rather than assumed.
    """
    pin_compiler_modules(repo)
    out = []
    ok = True
    out.append("this gate is %s" % os.path.relpath(os.path.abspath(__file__), ROOT))
    drift = anchor_check()
    if drift:
        out.append("ANCHOR: %s" % drift)
        ok = False
    for name in ("g17halfcheck", "g17scanlink", "g17authorobj", "g17verify"):
        local = os.path.join(HERE, name + ".py")
        if conflicted(local):
            out.append("%s.py carries MERGE CONFLICT MARKERS: an unresolved merge is in progress "
                       "and this file has both sides in it" % name)
            ok = False
    for rel in NEEDED:
        present, same = in_head(rel, repo)
        if not present:
            out.append("%s is NOT IN HEAD of %s" % (rel, os.path.basename(repo)))
            ok = False
        elif not same:
            out.append("%s is in HEAD but the working file DIFFERS" % rel)
            ok = False
        else:
            out.append("%s in HEAD and identical on disk" % rel)
    for rel, name in MODULES:
        blob = subprocess.run(["git", "-C", repo, "cat-file", "-p", "HEAD:" + rel],
                              capture_output=True)
        head = hashlib.sha256(blob.stdout).hexdigest() if blob.returncode == 0 else None
        try:
            module = importlib.import_module(name)
        except Exception as e:
            out.append("%s will not import here: %s" % (name, str(e)[:60]))
            ok = False
            continue
        where = os.path.abspath(getattr(module, "__file__", "") or "")
        got = hashlib.sha256(open(where, "rb").read()).hexdigest() if where else None
        if head is None or got != head:
            out.append("%s imports %s, which is NOT the %s blob verified above"
                       % (name, where, rel))
            ok = False
        else:
            out.append("%s imports the verified HEAD blob" % name)
            # A FORWARDING ENTRY IS NOT THE MODULE'S BEHAVIOUR. `import g17cc` loads a twelve-line
            # shim whose bytes are stable no matter what the compiler does, so verifying THAT
            # against HEAD says nothing about the file that emits the image. The implementation is
            # verified by name, and it is also in NEEDED above, so an uncommitted one refuses.
            implementation = _compat_src.source_of(rel)
            if implementation != rel:
                blob = subprocess.run(["git", "-C", repo, "cat-file", "-p", "HEAD:" + implementation],
                                      capture_output=True)
                want = hashlib.sha256(blob.stdout).hexdigest() if blob.returncode == 0 else None
                have = hashlib.sha256(open(os.path.join(ROOT, implementation), "rb").read()).hexdigest()
                if want is None:
                    out.append("%s forwards to %s, which is NOT IN HEAD of %s"
                               % (name, implementation, os.path.basename(repo)))
                    ok = False
                elif want != have:
                    out.append("%s forwards to %s, which is in HEAD but DIFFERS on disk"
                               % (name, implementation))
                    ok = False
                else:
                    out.append("%s forwards to %s, verified against HEAD" % (name, implementation))
    try:
        import g17ir
        if not hasattr(g17ir, "F16"):
            out.append("g17ir importable here has no F16: this checkout predates the half types")
            ok = False
    except Exception as e:
        out.append("g17ir will not import: %s" % str(e)[:60])
        ok = False
    return ok, out


def gate_structure(bundle):
    import g17scanlink as SL
    import g17verify as V
    man = json.load(open(os.path.join(bundle, "manifest.json")))
    archive = open(os.path.join(bundle, "scan.arc.metallib"), "rb").read()
    library = open(os.path.join(bundle, "scan.lib.metallib"), "rb").read()
    out, ok = [], True
    same = hashlib.sha256(archive).hexdigest() == man["sha256"]["archive"]
    out.append("archive hash %s the manifest" % ("matches" if same else "DIFFERS from"))
    ok &= same
    r = V.verify(archive, library)
    out.append("g17verify: %s" % ("no findings" if not r else r[:2]))
    ok &= not r
    try:
        obj = SL.verify_contract(archive, library, BINDINGS)
        out.append("delivered binding contract: %s" % (BINDINGS,))
    except Exception as e:
        out.append("delivered binding contract REFUSED: %s" % str(e)[:60])
        ok = False
        obj = None
    return ok, out, man, obj


def gate_code(bundle):
    """The delivered instruction memory. Independent of every hash above."""
    import g17halfcheck
    try:
        r = g17halfcheck.check_bundle(bundle)
    except Exception as e:
        return False, ["check_bundle raised: %s" % str(e)[:100]]
    if r.get("status") == "refused":
        return False, ["REFUSED: %s" % r.get("reason", "")]
    return True, ["delivered instruction memory accepted: %s" % r.get("scope", "")]


def gate_abi(man):
    """Conditional by construction while the ARCH flag is unresolved.

    The condition is written out in docs/archive/g17-arch-flag-contract.md rather than left as a
    standing hedge: what each side asserts, why the recovered rule agrees with the compiler,
    why the contradicting form is shipped regardless, and the two dispatches that settle it.
    """
    import g17authorobj as A
    import g17resource as R
    abi = dict(man.get("abi") or {})
    abi["bindings"] = abi.get("bindings") or [
        {"index": 1, "offset": 0, "written": False, "element_type": "half"},
        {"index": 2, "offset": 2, "written": True, "element_type": "half"}]
    withheld = {k: v for k, v in abi.items() if k != "arch_flag"}
    errs = R.check_agreement(withheld, PROFILE, A.PROFILES)
    out = ["with the ARCH flag WITHHELD: %s" % ("agrees" if not errs else errs[:1])]
    stated = R.check_agreement(abi, PROFILE, A.PROFILES) if "arch_flag" in abi else ["not stated"]
    out.append("with the compiler's arch_flag stated: %s"
               % ("agrees" if not stated else "refused - unresolved"))
    out.append("CONDITIONAL: agreement holds only because the ARCH flag is excluded from it.")
    out.append("  the recovered rule - 47 one-variable probes, 98.05% of 21,001 objects - sets the "
               "flag when every thread stands alone, and this scan's threads do: one read_sr at "
               "sr=160, no barrier, no cross-lane op, no threadgroup atomic or coordinate. So the "
               "compiler's True is CONSISTENT with this side's own measurement.")
    out.append("  the ELIDED form is shipped anyway because it is the form with nine hardware runs "
               "behind it. What the driver DOES with the flag is unmeasured, and that - not the "
               "rule - is the open question.")
    arch = os.path.join(ROOT, "isa", "g17-arch-runtime-validation.json")
    if os.path.exists(arch):
        try:
            doc = json.load(open(arch))
            shapes = ", ".join(str(st.get("shape")) for st in (doc.get("stages") or []))
            out.append("  the PAIRED experiment has run, by the integration owner: %s GPU queries "
                       "across elided and set variants at %s, status %s, every returned score "
                       "matching the same CPU reference bit for bit (%s)"
                       % (doc.get("gpu_queries"), shapes or "the tested shapes", doc.get("status"),
                          os.path.relpath(arch, ROOT)))
            out.append("  and it does NOT lift this condition: their record scopes it to those "
                       "images, shapes, device and inputs, and states it does not establish the "
                       "flag is non-semantic, measure scheduling or performance, or authorize "
                       "dropping it from compiler/linker agreement.")
        except Exception:
            pass
    out.append("  the compiler's semantic flag (True) and the serialized form (32-byte, elided) "
               "stay SEPARATE facts. Matching dispatches would establish compatibility for the "
               "tested variants, shapes, device and queries only; they cannot show the flag is "
               "non-semantic, and withholding it from agreement is not a general ABI pass. "
               "docs/archive/g17-arch-flag-contract.md")
    return (not errs), out


def build(rows=33, columns=384, arch_flag=False):
    """The image, from the compiler's program and the measured class's sections."""
    import g17authorobj as A
    import g17halfscan as H
    import g17link as L
    import g17scanlink as SL
    prog, abi = H.compile_scan(rows=rows, columns=columns)
    secs, led = A.author(text=b"", entry=abi["entry"], bindings=BINDINGS,
                         # The measured class is requested BY THIS SIDE, explicitly. The
                         # compiler's `profile` is a label and no longer selects a path.
                         abi={"measured_class": PROFILE, "arch_flag": arch_flag})
    binds = [L.Binding(index=b["index"], kind="device_buffer", readonly=not b["written"])
             for b in abi["bindings"]]
    k = L.Kernel(code=bytes(prog.code), bindings=binds, name="half_scan", entry=abi["entry"],
                 prologue=abi["prologue"])
    return SL.package_sections(k, secs, led, BINDINGS), abi, prog


def check_boundaries(code, layout):
    """Every instruction the compiler EMITTED must be the one the decoder CONSUMES.

    This is the naive fix's failure made general. Clearing the store's displacement operand also
    cleared a form bit: the assembler still wrote fourteen bytes, the decoder read ten, and the four
    bytes left over decoded as an entire extra instruction - 3079 where the compiler laid out 3078.
    Nothing about the store's opcode or its operands shows that. The disagreement is between two
    lengths for the same bytes, so both lengths are compared, at every instruction rather than at
    the one that is known to be wrong.
    """
    import g17packedcheck as PC
    notes, ok = [], True
    try:
        ins = PC.decode(code)
    except Exception as e:
        return False, ["the delivered text does not fully decode: %s" % str(e)[:70]], None
    emitted = [(off, len(b)) for off, b, _ in layout]
    decoded = [(off, size) for off, size, _, _ in ins]
    if len(emitted) != len(decoded):
        notes.append("compiler emitted %d instructions; the decoder consumes %d"
                     % (len(emitted), len(decoded)))
        ok = False
    for i, (e, d) in enumerate(zip(emitted, decoded)):
        if e != d:
            notes.append("first disagreement at instruction %d: emitted %d bytes at +%#x, "
                         "decoded %d bytes at +%#x" % (i, e[1], e[0], d[1], d[0]))
            ok = False
            break
    if sum(sz for _, sz in decoded) != len(code):
        notes.append("the decoded instructions do not tile the %d text bytes" % len(code))
        ok = False
    if ok:
        notes.append("all %d instructions: emitted offset and length equal decoded" % len(emitted))
    return ok, notes, ins


def attribute(fresh, recorded):
    """Say WHICH stage moved a hash, because "the hashes changed" is not an explanation.

    The three digests nest: code is inside the object, the object is inside the archive. So the
    innermost one that moved is the one that caused the others, and a change that appears only in
    an outer digest did not come from the compiler at all.
    """
    out = []
    same = {k: fresh[k] == recorded.get(k) for k in ("code", "object", "archive")}
    if all(same.values()):
        return ["all three digests unchanged: code, object, archive"]
    if not same["code"]:
        out.append("CODE moved: the compiler's emitted text differs. %s -> %s"
                   % (recorded.get("code", "?")[:16], fresh["code"][:16]))
        out.append("  the object and archive digests follow from it and are not separate findings")
    elif not same["object"]:
        out.append("code identical, OBJECT moved: the change is in authored metadata, not the "
                   "compiler. %s -> %s" % (recorded.get("object", "?")[:16], fresh["object"][:16]))
    elif not same["archive"]:
        out.append("code and object identical, ARCHIVE moved: the change is in packaging alone. "
                   "%s -> %s" % (recorded.get("archive", "?")[:16], fresh["archive"][:16]))
    return out


def destination(out_dir, control=BUNDLE):
    """Refuse to write anywhere that already holds something, and never at the negative control.

    exist_ok=True was the wrong call twice over: it would let a rebuild overwrite the refused image
    whose whole purpose is to stay byte-for-byte what it was, and it would let a second rebuild
    silently merge into the first, leaving a directory whose manifest describes one image and whose
    files are partly another.
    """
    dest = os.path.abspath(out_dir)
    if dest == os.path.abspath(control):
        raise ValueError("refusing to write over the negative control at %s" % out_dir)
    if os.path.exists(dest):
        raise ValueError("refusing to write into an existing destination: %s" % out_dir)
    return dest


def serialisable(abi):
    """The compiler's ABI as JSON, losing nothing the delivered-bundle checker needs."""
    out = dict(abi)
    if isinstance(out.get("prologue"), (bytes, bytearray)):
        out["prologue"] = bytes(out["prologue"]).hex()
    out["forms"] = [list(x) for x in (out.get("forms") or [])]
    # STRINGIFYING A KEY IS LOSSY, so the count is checked rather than assumed. int 15 and the
    # string "15" are different keys that json cannot tell apart, and a silent overwrite here would
    # put a manifest with a missing pk_value in front of a checker that reads it as complete. The
    # compiler owner hit the same shape in a bases table, where a lost key read back as a base of
    # zero rather than as an error; the lesson is that the check must compare the UNPACKED value
    # against the source, never one packing against another packing.
    pk = out.get("pk_values") or {}
    packed = {str(k): v for k, v in pk.items()}
    if len(packed) != len(pk):
        seen = [str(k) for k in pk]
        collided = sorted({k for k in seen if seen.count(k) > 1})
        raise ValueError("pk_values keys collide when stringified for JSON: %r - a manifest "
                         "written from this would silently drop an ABI value" % collided)
    out["pk_values"] = packed
    out["pk_extra"] = list(out.get("pk_extra") or [])
    # ABI v2 freezes nested records too. Convert only at this JSON boundary;
    # keep the collision check above before JSON can stringify mapping keys.
    from collections.abc import Mapping
    def plain(value):
        if isinstance(value, Mapping):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        if isinstance(value, bytes):
            return value.hex()
        return value
    return plain(out)


def rebuild(bundle=BUNDLE, out_dir=None, write=False):
    """Item 3: rebuild once the compiler fix lands, explain the hashes, re-verify the image.

    It refuses to run at all while gate 1 fails. Rebuilding from inputs that are not the verified
    ones would produce an artifact with the same problem as the one being replaced, and recording
    its hashes would make that problem harder to see rather than easier.

    It never writes over the negative control. The refused image keeps its bytes and its manifest.
    """
    print("REBUILD")
    ok1, notes = gate_provenance()
    for n in notes:
        print("   %s" % n)
    if not ok1:
        print("\n   REFUSED to rebuild: the inputs are not verified. Fix gate 1 first.")
        return 2
    recorded = json.load(open(os.path.join(bundle, "manifest.json")))["sha256"]
    # What the build TOUCHES, not only what it imports. A module audit cannot see data or
    # subprocesses, and the defect that started this workstream lived in a data overlay.
    import g17buildinputs as BI
    with BI.watch() as (reads, runs):
        img, abi, prog = build()
    forks = sorted({os.path.basename(str(r)) for r in runs
                    if os.path.sep in str(r) and BI.classify(str(r))[0]
                    not in ("TRACKED AND CLEAN", "BUILT FROM TRACKED SOURCE")})
    dirty = sorted({os.path.relpath(p2, ROOT) for p2 in reads
                    if os.path.isfile(p2) and BI.classify(p2)[0] not in ("TRACKED AND CLEAN",)})
    print("\n   what the build touched: %d files read, %d unpinned; subprocesses run from "
          "untracked tools: %s" % (len(reads), len(dirty), ", ".join(forks) or "none"))
    for d in dirty[:5]:
        print("      unpinned read: %s" % d)
    fresh = {"archive": img.sha256,
             "object": hashlib.sha256(img.object).hexdigest(),
             "code": hashlib.sha256(bytes(prog.code)).hexdigest()}
    print("\n   hashes:")
    for n in attribute(fresh, recorded):
        print("      %s" % n)
    unverified, checked = audit_loaded()
    print("\n   every module the build loaded: %d from a tracked repo, %d unverified"
          % (checked, len(unverified)))
    for u in unverified[:6]:
        print("      %s" % u)
    if unverified:
        print("   REFUSED: the artifact would not be reproducible from committed sources.")
        return 2
    okb, notes, ins = check_boundaries(bytes(prog.code), prog.layout)
    print("\n   instruction boundaries: %s" % ("pass" if okb else "FAIL"))
    for n in notes:
        print("      %s" % n)
    oka = True
    try:
        import g17halfcheck as HC
        print("\n   addressing: %s" % HC.check_memory(ins) if ins else "   addressing: skipped")
    except Exception as e:
        oka = False
        print("\n   addressing: FAIL - %s" % str(e)[:100])
    okc = True
    try:
        import g17scanlink as SL
        SL.verify_contract(img.archive, img.library, BINDINGS)
        print("\n   delivered binding contract: %s" % (BINDINGS,))
    except Exception as e:
        okc = False
        print("\n   delivered binding contract: FAIL - %s" % str(e)[:80])
    if not (okb and oka and okc):
        print("\n   NOT RECORDED. A rebuilt image that fails its own checks is not an artifact.")
        return 2
    # Compare against the REFUSED image specifically, not against whichever bundle supplied the
    # hash baseline. Comparing against the baseline meant that rebuilding to verify the corrected
    # bundle - where identical digests are the point - reported it as the defective image and
    # refused. The guard is "this is not the artifact we rejected", and that is one fixed set of
    # bytes, not a relative statement.
    refused = json.load(open(os.path.join(BUNDLE, "manifest.json")))["sha256"].get("code")
    if fresh["code"] == refused:
        print("\n   the rebuilt code is byte-identical to the REFUSED image at %s. The defect is"
              % os.path.relpath(BUNDLE, ROOT))
        print("   not fixed; nothing is recorded.")
        return 2
    if fresh == {k: recorded.get(k) for k in fresh}:
        print("\n   byte-identical to %s: that bundle is reproducible from committed sources."
              % os.path.relpath(bundle, ROOT))
        return 0
    print("\n   the rebuild passes boundaries, addressing and the binding contract, and its code")
    print("   differs from the refused image.")
    if not write:
        print("   not written: pass --write to record it as a new bundle. The negative control")
        print("   at %s is never overwritten." % os.path.relpath(bundle, ROOT))
        return 0
    if forks or dirty:
        print("\n   NOT RECORDED: the emitted bytes were chosen with help from something no HEAD")
        print("   pins (%s). The build is byte-deterministic on this machine and it is not"
              % (", ".join(forks + dirty[:2]) or "an unpinned input"))
        print("   reproducible from tracked sources, which is the claim a recorded bundle makes.")
        print("   Verification above still stands; only recording a NEW bundle is refused.")
        return 2
    out_dir = out_dir or os.path.join(ROOT, "results", "g17-half-scan-33x384-corrected")
    try:
        dest = destination(out_dir, bundle)
    except ValueError as e:
        print("\n   %s" % e)
        return 2
    os.makedirs(dest)
    open(os.path.join(dest, "scan.arc.metallib"), "wb").write(img.archive)
    open(os.path.join(dest, "scan.lib.metallib"), "wb").write(img.library)
    open(os.path.join(dest, "scan.o"), "wb").write(img.object)
    man = {"profile": "half-buffer-two-bindings-33x384", "shape": {"rows": 33, "columns": 384},
           "status": "code gate passed; HARDWARE VALIDATION NOT RUN",
           "gpu_executed": False, "sha256": fresh,
           # The compiler's ABI is part of the bundle, not a build-time detail: the delivered-bundle
           # checker reads the entry from it to find __text, and without it the artifact this path
           # writes cannot be checked by the checker that refused its predecessor.
           "abi": serialisable(abi),
           "supersedes": {"bundle": os.path.relpath(bundle, ROOT), "code": recorded.get("code"),
                          "reason": "800-byte store displacement at 0x991e"},
           "arch_flag": "ELIDED. The compiler's semantic flag is True and the serialized form is "
                        "the 32-byte elided one; those are separate facts and both are recorded. "
                        "docs/archive/g17-arch-flag-contract.md"}
    json.dump(man, open(os.path.join(dest, "manifest.json"), "w"), indent=1)
    # The checks above ran on objects in memory. This one runs on the bytes that were written,
    # through the same checker that refuses the negative control, arithmetic included.
    try:
        import g17halfcheck as HC
        report = HC.check_bundle(dest)
    except Exception as e:
        shutil.rmtree(dest)
        print("\n   the WRITTEN bundle fails the delivered checker: %s" % str(e)[:100])
        print("   removed %s. A bundle that cannot pass the checker is not an artifact."
              % os.path.relpath(dest, ROOT))
        return 2
    print("   wrote %s" % os.path.relpath(dest, ROOT))
    print("   delivered-bundle checker: %s" % {k: report[k] for k in list(report)[:4]})
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--rebuild" in argv:
        return rebuild(write="--write" in argv)
    bundle = next((a for a in argv if not a.startswith("-")), BUNDLE)
    print(__doc__.split("\n\n")[0])
    print("\n   bundle: %s\n" % os.path.relpath(bundle, ROOT))
    results = []

    ok1, notes = gate_provenance()
    print("   GATE 1  PROVENANCE  %s" % ("pass" if ok1 else "FAIL"))
    for n in notes:
        print("             %s" % n)
    results.append(("provenance", ok1))

    ok2, notes, man, _obj = gate_structure(bundle)
    print("\n   GATE 2  STRUCTURE   %s" % ("pass" if ok2 else "FAIL"))
    for n in notes:
        print("             %s" % n)
    results.append(("structure", ok2))

    ok3, notes = gate_code(bundle)
    print("\n   GATE 3  CODE        %s" % ("pass" if ok3 else "FAIL"))
    for n in notes:
        print("             %s" % n)
    results.append(("code", ok3))

    ok4, notes = gate_abi(man)
    print("\n   GATE 4  ABI         %s (conditional)" % ("pass" if ok4 else "FAIL"))
    for n in notes:
        print("             %s" % n)
    results.append(("abi", ok4))

    print("\n   HARDWARE            NOT ASSERTED - nothing here dispatches or creates a pipeline")
    ev = execution_evidence(man.get("sha256", {}).get("archive"))
    if ev:
        print("             the integration owner records a run of THIS archive: %s, %s GPU "
              "queries, returned half bit-exact %s" % (ev["shape"], ev["queries"], ev["bit_exact"]))
        print("             their record, not this gate's finding: %s" % ev["record"])
        if ev.get("arch_true_variant_executed") is False:
            print("             the arch_flag=True variant has NOT been executed, so gate 4 stays "
                  "conditional on exactly the evidence its contract names")
    failed = [n for n, v in results if not v]
    if failed:
        print("\n   NOT ELIGIBLE for hardware validation. Failing gates: %s" % ", ".join(failed))
        print("   A passing gate does not excuse a failing one: this image's archive hash matches")
        print("   and its metadata is byte-exact, and its delivered CODE is still wrong.")
        return 2
    print("\n   all four gates pass; hardware validation remains the integration owner's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
