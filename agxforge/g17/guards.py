"""Compile-time guards: every program cc compiles is checked for two defects no value check can see (MM 25.144.6).

  * DEAD OPERATIONS. cc has no dead-code pass: a value built and never read is still emitted and executed. It
    cost the hottest loop twice - 15 unread x-index adds per trip (25.141.3), then 12-14 unread pre-scale
    fmuls per trip in every shipped q4/q8 qmv (25.141.17) - and neither changed a value, so no bit-exact
    check could see it. A dead op is a side-effect-free op whose result nothing reads, iterated to a fixed
    point so chains that feed only dead ops count; an unread constant counts only inside a loop body (a block
    holding a phi), since outside one it is a single instruction per thread, once.
  * A REGISTER READ AFTER ITS RELEASE, on the emitted bytes, on any path one lane can take, loops to a fixed
    point (agxforge/g17/releasecheck.py; 0 findings over Apple's 6,594 programs; it reproduces both past cc
    bugs, f453622af and a64e5b34e, at the exact register and instruction).

MODES (keyword `guards=` on cc.compile_function, else the environment variable G17_CC_GUARDS):
  default  a READ AFTER RELEASE refuses; an unread op is recorded (warn). The default, because the suite-wide
           survey (MM 25.144.6) found the two populations different in kind: every read-after-release outside a
           deliberate bug reproduction was a real cc defect (five with the fuzzer, all fixed), while the unread ops were 85
           probe and witness programs that leave a builtin, an index or a load unread on purpose
  refuse   either finding raises GuardRefused (a cc.Unsupported), unless it is a named KNOWN exception; the
           shipped builders are held to this (test_g17ccguards)
  warn     findings are recorded on the program (`program.guards`) and appended to G17_CC_GUARD_LOG if set;
           a deliberate reproduction of a fixed bug compiles under this
  off      no check (for a caller that must not pay the check's ~28 ms)
Whatever the mode, the guards never change emitted bytes: they read the IR and the finished program.
"""
import hashlib, json, os, time

MODES = ("default", "refuse", "warn", "off")
DEFAULT_MODE = "default"

# Side-effecting value ops: an unread result is still not dead.
EFFECT_KINDS = ("machine",)
EFFECT_PREFIXES = ("store", "atomic", "barrier", "br", "ret", "exec_", "imageblock_write", "tensor_")

# KNOWN EXCEPTIONS, each with its reason. A dead op is matched by (kind, value name); a release finding by the
# program's sha256. Adding one is a decision to keep bytes something else pins.
KNOWN_DEAD = {
    ("sub", "q0m1"): "g17attn wide form: q0 - 1 is read only by the register-select form; one instruction "
                     "outside the key loop, and removing it moves the delivered widebf bytes test_g17cap2048 "
                     "pins (MM 25.141.17)",
}
# The ladder's three SEQUENCE EXPERIMENTS (tools/g17ladder.py tensor_matmul / mixed_scalar_tensor /
# tensor_unseen_shape; tensor_matmul(sequence_experiment=True)): Apple's cached tensor MAC block (op5106) emitted
# WITHOUT its operand loads - cc's own diagnosis calls it "an EXPERIMENT, not an executable kernel". Nothing in
# them writes the MACs' A and B registers, each MAC releases them (lifetime operands 4 and 7 = 0x10), and the next
# MAC pair reads the same registers: a true read-after-release IN THESE BYTES, and the reason they are not a
# kernel. The ladder never executes a rung; g17regress pins the bytes (hash prefixes 012542f0 / 7d328534 /
# ae8405dc). Excused by exact sha only: any other program with the same pattern still refuses, and so do these the
# day their bytes change.
KNOWN_RELEASE = {
    "012542f0633e4c73dbd14c15244c9c1a82ad3e29a53d91c9dedbe5b217c92e4d":
        "ladder tensor_matmul: sequence experiment, the MAC block without its operand loads (never dispatched)",
    "7d3285342fb4fb5441af7a066bd1212eee659111c1df59c0dd0238f0ee03a2f1":
        "ladder mixed_scalar_tensor: sequence experiment, the MAC block without its operand loads (never dispatched)",
    "ae8405dcda134c354c137486d184032c662824275c3acc8beb99e5d326a0b022":
        "ladder tensor_unseen_shape: sequence experiment, the MAC block without its operand loads (never dispatched)",
}


class GuardRefused(Exception):
    """A guard finding in refuse mode; cc re-raises it as Unsupported."""


def dead_ops(fn):
    """[(kind, value name)] of unread side-effect-free ops, iterated to a fixed point; unread constants only
    inside a loop body."""
    ops = [o for b in fn.blocks for o in b.ops]
    in_loop = {id(o) for b in fn.blocks if any(o.kind == "phi" for o in b.ops) for o in b.ops}
    dead = set()
    while True:
        used = {id(a) for o in ops if id(o) not in dead for a in o.args}
        new = {id(o) for o in ops if id(o) not in dead and o.dest is not None and id(o.dest) not in used
               and o.kind not in EFFECT_KINDS and not o.kind.startswith(EFFECT_PREFIXES)}
        if not new:
            return [(o.kind, getattr(o.dest, "name", "") or "") for o in ops
                    if id(o) in dead and (o.kind != "const" or id(o) in in_loop)]
        dead |= new


def release_findings(code):
    """The release analysis over Apple's decode, read IN THIS PROCESS (model.decode_nofork, tools/libagx3dis.dylib):
    a compile must not start an external process (the delivery paths audit it). -> findings, or None when the
    in-process decoder is not built (make native-tools) - recorded, never a refusal."""
    from agxforge.g17 import releasecheck, model
    code = bytes(code)
    try:
        decoded = list(model.decode_nofork(code, 0))
    except model.DecoderUnavailable:
        return None
    try:
        return releasecheck.check(code, decoded)["findings"]
    except Exception:
        # THE GUARD NEVER BREAKS A COMPILE ON BYTES IT CANNOT WALK (a caller that corrupts emitted bytes on
        # purpose, as test_g17tensorreloadprobe does, must see its own refusal): unchecked, and recorded so
        return None


def mode_of(guards=None):
    m = guards if guards is not None else os.environ.get("G17_CC_GUARDS", DEFAULT_MODE)
    if m not in MODES:
        raise ValueError("guards mode %r is not one of %s" % (m, MODES))
    return m


STATS = {"programs": 0, "seconds": 0.0, "dead": 0, "release": 0}


def check(fn, program, guards=None):
    """Run both guards on the IR `fn` (as compiled) and the finished `program`. Records program.guards =
    {mode, dead, release, known, sha256, seconds}; raises GuardRefused in refuse mode on an unexcused finding."""
    mode = mode_of(guards)
    if mode == "off":
        return None
    t0 = time.perf_counter()
    dead = dead_ops(fn)
    rel = release_findings(program.code)
    unavailable = rel is None
    rel = rel or []
    sha = hashlib.sha256(bytes(program.code)).hexdigest()
    known = [list(d) for d in dead if d in KNOWN_DEAD] + (["release:" + sha] if rel and sha in KNOWN_RELEASE else [])
    bad_dead = [d for d in dead if d not in KNOWN_DEAD]
    bad_rel = [] if sha in KNOWN_RELEASE else rel
    dt = time.perf_counter() - t0
    STATS["programs"] += 1
    STATS["seconds"] += dt
    STATS["dead"] += bool(bad_dead)
    STATS["release"] += bool(bad_rel)
    rec = dict(mode=mode, dead=bad_dead, release=bad_rel, known=known, sha256=sha, seconds=round(dt, 4),
               release_checked=not unavailable)
    try:
        program.guards = rec
    except Exception:
        pass
    log = os.environ.get("G17_CC_GUARD_LOG")
    if log and (bad_dead or bad_rel):
        import traceback
        stack = [f for f in traceback.extract_stack()[:-1] if "/agxforge/g17/" not in f.filename]
        caller = ["%s:%d:%s" % (f.filename.split("/")[-1], f.lineno, f.name) for f in stack[-4:]]
        with open(log, "a") as f:
            f.write(json.dumps(dict(rec, fn=getattr(fn, "name", "?"), caller=caller)) + "\n")
    refuse_dead = mode == "refuse"
    refuse_rel = mode in ("refuse", "default")
    if (refuse_dead and bad_dead) or (refuse_rel and bad_rel):
        why = []
        if bad_dead and refuse_dead:
            why.append("%d unread op(s): %s" % (len(bad_dead), ", ".join("%s %s" % d for d in bad_dead[:6])))
        if bad_rel and refuse_rel:
            why.append("register read after release: %s" % (bad_rel[:4],))
        raise GuardRefused("; ".join(why))
    return rec


class Mode:
    """`with guards.Mode("warn"):` - every compile inside runs in that mode (the environment variable, restored
    on exit). For code that rebuilds a FIXED bug on purpose, so the release guard records it instead of
    refusing it."""

    def __init__(self, mode):
        self.mode = mode_of(mode)

    def __enter__(self):
        self.saved = os.environ.get("G17_CC_GUARDS")
        os.environ["G17_CC_GUARDS"] = self.mode
        return self

    def __exit__(self, *exc):
        if self.saved is None:
            os.environ.pop("G17_CC_GUARDS", None)
        else:
            os.environ["G17_CC_GUARDS"] = self.saved
        return False
