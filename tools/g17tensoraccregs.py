#!/usr/bin/env python3
"""THE REGISTER ACCUMULATOR, on hardware (docs/g17-tensorops-machine-model.md 25.144.8).

cc's `tensor_matmul(..., accumulate=True, acc="O")` keeps C in fixed registers and writes D = A @ B + C back
into them (tlower's c_inplace); `tensor_acc_read` / `tensor_acc_write` move one lane's register to and from a
scalar value. Two programs, one dispatch per process each:

    probe       O = 0 (one 16 x 16 tile), then ONE body with A = I and B[k][c] = 16 k + c + 1, so D[r][c] = 16 r + c + 1
                names its own element. Each lane then stores its eight registers by plain word stores at word
                8 lane + slot. Decoding every word gives the (lane, register) -> (row, col) map, measured by
                scalar reads. It must equal ir.tensor_acc_position on all 256 (lane, slot) pairs.
    loop        a 16 x 128 O - EIGHT tiles, 64 registers, M2's prefill shape - zeroed; four trips of
                { O = 0.5 O (scalar: read, fmul, write, every register) ; O += A @ B_j } with B_j the j-th 16 x 128
                block (the stream's B index register, set once before the loop); then each lane stores tile t's
                registers at word 256 t + 8 lane + slot. The accumulator is carried in registers across trips with
                scalar code between the bodies, written by both write forms (the move below R64, the OR above).
                Scored bit for bit against O_{j+1} = _gemm_mma(A, B_j, fl(0.5 O_j)).

Controls, each of which must FAIL: the probe decoded under transpose.pos (the rival row labeling), and the loop
scored against the reference without the rescale and against the reference of three trips.

SAFETY. The loop is the counted key-block loop's shape (a compile-time trip count of 4), and cc proves its
latch from the emitted bytes (tensorlife.counted_loop_check); `run` re-checks the exact bytes, holds while a
gate runs, and dispatches ONE query in ONE process.

    python3 tools/g17tensoraccregs.py prepare
    python3 tools/g17tensoraccregs.py run probe|loop
    python3 tools/g17tensoraccregs.py compare
"""
import hashlib, json, os, struct, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402
import g17tensorcommonruntime as T  # noqa: E402

OUT = Path(ROOT) / "results" / "g17-tensor-acc-registers-v1"
TRIPS = 4
SCALE = 0.5
LOOP_N = 128
TIMEOUT = 60
ARMS = ("probe", "loop", "sg4", "sg4h", "sg4h4", "sg4h8", "far", "farfold", "loopscale", "fast")
SG = 4
# far / farfold: the loop with A and B at byte offsets past the 32,767-byte displacement field, without and with
# fold_offsets (the offset added to the index register once, instead of re-derived for every tile)
FAR_A, FAR_B = 40960, 40960
HEADS = 2          # sg4h: two threadgroups, head h's A 16 SG rows further on (head_stride), B shared by all


def _h(b):
    return hashlib.sha256(b).hexdigest()


def _f32_const(bd, value, name):
    from agxforge.g17 import ir
    return bd.const(struct.unpack("<I", struct.pack("<f", float(value)))[0], type=ir.F32, name=name)


def _store_lane_registers(bd, c, lane, acc, tiles=1, sg=1, witness=None):
    """Each lane stores tile t's eight registers of `acc` at word 256 (tiles threadgroup + t) + 8 lane + slot (the
    raw layout; one threadgroup - the threadgroup read is the measured class's SR156)."""
    from agxforge.g17 import ir
    tg = bd.shl(bd.builtin("threadgroup_position_in_grid", name="tg"), ir.Imm(8), name="tg256")
    if tiles * sg > 1:
        tg = bd.mul(tg, ir.Imm(tiles * sg), name="tg_tiles")
    base = bd.add(bd.shl(lane, ir.Imm(3), name="lane8"), tg, name="st_base")
    if sg > 1:
        # simdgroup s stores its own accumulator after the previous simdgroups' (256 tiles words each); the index
        # is masked as tlower masks it (& 3 up to four simdgroups)
        sgi = bd.builtin("simdgroup_index_in_threadgroup", name="sg")
        sgi = getattr(bd, "and")(sgi, ir.Imm(3), name="sg3")
        assert tiles & (tiles - 1) == 0
        base = bd.add(base, bd.shl(sgi, ir.Imm((256 * tiles).bit_length() - 1), name="sg_words"), name="st_base_sg")
    for t in range(tiles):
        for s in range(8):
            w = 256 * t + s
            # A loop-exit witness replaces one distinct output word per lane
            # and SIMDgroup. The other words still expose the tensor result.
            v = (witness if w == 0 and witness is not None else
                 bd.tensor_acc_read(acc, s, tile=(0, t), name="st_d%d_%d" % (t, s)))
            bd.store_at(c, base if w == 0 else bd.add(base, ir.Imm(w), name="st_i%d_%d" % (t, s)), v)


def probe_fn():
    from agxforge.g17 import ir
    a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function(T.GENERIC_NAME, [a, b, c]); bd = ir.Builder(fn, fn.block("entry"))
    zero = _f32_const(bd, 0.0, "zero")
    for s in range(8):
        bd.tensor_acc_write("O", s, zero)
    bd.tensor_matmul(a, b, c, M=16, N=16, K=16, accumulate=True, acc="O")
    _store_lane_registers(bd, c, bd.builtin("thread_index_in_simdgroup", name="lane"), "O")
    bd.ret(); ir.verify(fn)
    return fn


def loop_fn(trips=TRIPS, scale=SCALE, N=LOOP_N, sg=1, heads=1, far=False, fold=False, fused=False, hoist=False):
    from agxforge.g17 import ir
    a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function(T.GENERIC_NAME, [a, b, c]); bd = ir.Builder(fn, fn.block("entry"))
    zero = _f32_const(bd, 0.0, "zero")
    tiles = N // 16
    for t in range(tiles):
        for s in range(8):
            bd.tensor_acc_write("O", s, zero, tile=(0, t))
    bd.tensor_index_init("k", 0)
    i0 = bd.const(0, name="i0")
    hdr, ex = fn.block("trips"), fn.block("done")
    bd.br(hdr); bd.at(hdr)
    i = bd.phi(i0, name="i")
    half = _f32_const(bd, scale, "scale")
    for t in range(tiles):
        for s in range(8):
            if fused:        # tensor_acc_scale: one multiply on the register (loopscale)
                bd.tensor_acc_scale("O", s, half, tile=(0, t))
                continue
            bd.tensor_acc_write("O", s, bd.fmul(bd.tensor_acc_read("O", s, tile=(0, t), name="o%d_%d" % (t, s)), half,
                                                name="h%d_%d" % (t, s)), tile=(0, t))
    bd.tensor_matmul(a, b, c, M=16 * sg, N=N, K=16, accumulate=True, acc="O",
                     offsetB_register="k", offsetB_step=16 * N * 2, simdgroups=sg,
                     **({} if not far else dict(offsetA=FAR_A, offsetB=FAR_B)),
                     **({} if not fold else dict(fold_offsets=True)),
                     **({} if not hoist else dict(hoist_prologue=True)),
                     **({} if heads == 1 else dict(head_stride=(16 * sg * 16 * 2, 0, 0))))
    nxt = bd.add(i, ir.Imm(1), name="n")
    ir.Builder.phi_latch(i, nxt)
    bd.br_cond(bd.cmp(nxt, trips, "lt", name="more"), hdr, ex)
    bd.at(ex)
    _store_lane_registers(bd, c, bd.builtin("thread_index_in_simdgroup", name="lane"), "O", tiles=tiles, sg=sg)
    bd.ret(); ir.verify(fn)
    return fn


BUILDERS = {"probe": probe_fn, "loop": loop_fn, "sg4": lambda: loop_fn(sg=SG),
            "sg4h": lambda: loop_fn(sg=SG, heads=HEADS),
            "sg4h3": lambda: loop_fn(sg=SG, heads=3),
            "sg4h4": lambda: loop_fn(sg=SG, heads=4),
            "sg4h8": lambda: loop_fn(sg=SG, heads=8),
            "far": lambda: loop_fn(far=True), "farfold": lambda: loop_fn(far=True, fold=True),
            "loopscale": lambda: loop_fn(fused=True),
            # everything at once: far offsets folded, the fused rescale, the prologue hoisted out of the loop
            "fast": lambda: loop_fn(far=True, fold=True, fused=True, hoist=True)}
SPECS = {"probe": {"M": 16, "N": 16, "K": 16}, "loop": {"M": 16, "N": LOOP_N, "K": 16 * TRIPS},
         "sg4": {"M": 16 * SG, "N": LOOP_N, "K": 16 * TRIPS, "simdgroups": SG},
         "sg4h": {"M": 16 * SG * HEADS, "N": LOOP_N, "K": 16 * TRIPS, "simdgroups": SG, "threadgroups": HEADS},
         "sg4h3": {"M": 16 * SG * 3, "N": LOOP_N, "K": 16 * TRIPS, "simdgroups": SG, "threadgroups": 3},
         "sg4h4": {"M": 16 * SG * 4, "N": LOOP_N, "K": 16 * TRIPS, "simdgroups": SG, "threadgroups": 4},
         "sg4h8": {"M": 16 * SG * 8, "N": LOOP_N, "K": 16 * TRIPS, "simdgroups": SG, "threadgroups": 8},
         # sized so A (M x K halves) and B (K x N halves) reach past the far offsets; the class caps K at 256, and the
         # program itself reads 16 rows of A and writes the first 2,048 words of C
         "far": {"M": 96, "N": LOOP_N, "K": 256}, "farfold": {"M": 96, "N": LOOP_N, "K": 256},
         "loopscale": {"M": 16, "N": LOOP_N, "K": 16 * TRIPS},
         "fast": {"M": 96, "N": LOOP_N, "K": 256}}


def compile_arm(arm):
    from agxforge.g17 import cc
    return cc.compile_function(BUILDERS[arm]())


def _author(arm, bundle):
    """T.author_generic with this arm's program in place of the generic GEMM (author_generic resolves
    build_generic_program from T's globals, so the patch is seen there)."""
    saved = T.build_generic_program
    T.build_generic_program = lambda s: compile_arm(arm)
    try:
        T.author_generic(bundle, SPECS[arm])
    finally:
        T.build_generic_program = saved


def _inputs(arm, bundle):
    if arm in ("sg4h3", "sg4h4", "sg4h8"):
        source = OUT / "sg4h"
        a = (source / "a.f16").read_bytes()
        want_a = SPECS[arm]["M"] * SPECS[arm]["K"] * 2
        if len(a) > want_a:
            raise ValueError("source A exceeds the multi-head input")
        (bundle / "a.f16").write_bytes(a + bytes(want_a-len(a)))
        (bundle / "b.f16").write_bytes((source / "b.f16").read_bytes())
        (bundle / "c.f32").write_bytes(b"\xff" * (98304 if arm == "sg4h3" else 131072 if arm == "sg4h4" else 262144))
        return
    if arm == "probe":
        a = np.zeros((16, 16), np.float16); np.fill_diagonal(a, 1)
        k, col = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
        b = (16 * k + col + 1).astype(np.float16)
    else:
        rng = np.random.default_rng(2144)
        # drawn at the class's M x K (the runtime reads that many); the body reads the first M x 16 halves, row-major
        a = rng.standard_normal((SPECS[arm]["M"], SPECS[arm]["K"])).astype(np.float16)
        b = rng.standard_normal((SPECS[arm]["K"], LOOP_N)).astype(np.float16)  # block j = rows 16 j .. 16 j + 15
    (bundle / "a.f16").write_bytes(a.astype("<f2").tobytes())
    (bundle / "b.f16").write_bytes(b.astype("<f2").tobytes())
    words = SPECS[arm]["M"] * SPECS[arm]["N"]
    (bundle / "c.f32").write_bytes(np.full(words, 0xFFFFFFFF, "<u4").tobytes())     # the sentinel


def prepare():
    from agxforge.g17 import tensorlife, cc
    OUT.mkdir(parents=True, exist_ok=True)
    pre = {}
    for arm in ARMS:
        bundle = OUT / arm
        if not bundle.exists():
            _author(arm, bundle)
            _inputs(arm, bundle)
        code = (bundle / "program.bin").read_bytes()
        pre[arm] = dict(program=_h(code), bytes=len(code))
        if arm != "probe":
            # THE CARRIED SET IS THE ONE cc RECORDED: the index registers this program's bodies name
            # (cc._TENSOR_INDEX_USED), not both stream registers - an unnamed one is an ordinary scalar register,
            # and checking it as carried refuses programs cc admits (the M6 agent's fuzz, MM 25.144.8)
            if compile_arm(arm).code != code:
                raise SystemExit("REFUSED: %s's bundle is not what cc compiles now" % arm)
            carried = tuple(cc._TENSOR_INDEX_USED)
            pre[arm]["carried"] = list(carried)
            pre[arm]["latch_check"] = tensorlife.counted_loop_check(code, TRIPS, carried=carried)
    (OUT / "prereg.json").write_text(json.dumps(pre, indent=1, sort_keys=True) + "\n")
    print(json.dumps(pre, indent=1, sort_keys=True))


def _gate_running():
    ps = subprocess.run(["ps", "-Ao", "command"], capture_output=True, text=True).stdout
    return [l for l in ps.splitlines() if ("make check" in l or "g17test.py" in l)
            and "grep" not in l and "--no-gpu" not in l]


def run(arm):
    from agxforge.g17 import tensorlife, cc
    pre = json.loads((OUT / "prereg.json").read_text())
    b = OUT / arm
    code = (b / "program.bin").read_bytes()
    if _h(code) != pre[arm]["program"]:
        raise SystemExit("REFUSED: %s's program is not the preregistered one" % arm)
    if arm != "probe":
        tensorlife.counted_loop_check(code, TRIPS, carried=tuple(pre[arm]["carried"]))
    if _gate_running():
        raise SystemExit("HOLD: a gate is running")
    path = OUT / "dispatch-receipt.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    rec = dict(program=_h(code)[:16], queries_per_process=1)
    try:
        rep = T.run(b, queries=1, composition="generic", save_output=True, mismatch_ok=True, never_kill_after=TIMEOUT)
        rec.update(status=rep["status"], gpu_seconds=rep["queries"][0]["gpu_seconds"])
    except Exception as e:
        rec.update(status="FAILED", error=str(e)[:400])
    out[arm] = rec
    print(arm, json.dumps(rec), flush=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


def _got(arm):
    return np.load(OUT / arm / "output-q1.npz")["got"].view("<u4").ravel()[:SPECS[arm]["M"] * SPECS[arm]["N"]]


def loop_reference(bundle, trips=TRIPS, scale=SCALE, M=16, head0=False, far=False):
    # the body's A: the first 256 halves, row-major 16 x 16 (lda = K = 16). A first version took columns 0..15 of the
    # 16 x 64 drawn matrix and scored the bit-exact loop as 1,920 wrong words
    # sg4h: head h's A is the h-th block of 16 SG rows x 16 (head_stride), all heads stacked row-wise
    a0 = FAR_A // 2 if far else 0
    a = np.frombuffer((bundle / "a.f16").read_bytes(), "<f2")[a0:a0 + 16 * M].reshape(M, 16).astype(np.float32)
    if head0:          # the control: every head reads head 0's A (a head stride that was dropped)
        a = np.tile(a[:16 * SG], (M // (16 * SG), 1))
    b0 = FAR_B // 2 if far else 0
    b = np.frombuffer((bundle / "b.f16").read_bytes(), "<f2")[b0:b0 + 16 * trips * LOOP_N]
    b = b.reshape(16 * trips, LOOP_N).astype(np.float32)
    o = np.zeros((M, LOOP_N), "<f4")
    for j in range(trips):
        o = (o * np.float32(scale)).astype("<f4") if scale != 1 else o
        o = T._gemm_mma(a, b[16 * j:16 * j + 16], o, M, LOOP_N, 16)
    return o


def transpose_pos(r, c):
    """tools/tensorops-model/transpose.py pos() (A and D, "rows {2m, 2m+1}"), quoted: that module imports its
    campaign runner and cannot be loaded alone. The rival labeling of the accumulator's rows."""
    return 16 * (r >> 3) + 8 * (c >> 3) + 2 * ((r >> 1) & 3) + ((c >> 2) & 1), 4 * (r & 1) + (c & 3)


def lane_words(matrix):
    """The words the program stores: word 256 (tiles sg + t) + 8 lane + slot = simdgroup sg's tile t (rows 16 sg ..)
    of matrix at tensor_acc_position."""
    from agxforge.g17 import ir
    tiles, groups = matrix.shape[1] // 16, matrix.shape[0] // 16
    out = np.empty(256 * tiles * groups, "<u4")
    m = np.ascontiguousarray(matrix, "<f4").view("<u4")
    for g in range(groups):
        for t in range(tiles):
            for lane in range(32):
                for s in range(8):
                    r, c = ir.tensor_acc_position(lane, s)
                    out[256 * (tiles * g + t) + 8 * lane + s] = m[16 * g + r, 16 * t + c]
    return out


def compare():
    from agxforge.g17 import ir
    res = {}
    got = _got("probe").view("<f4")
    inv_pos = {transpose_pos(r, c): (r, c) for r in range(16) for c in range(16)}
    ok = rival = unwritten = 0
    for lane in range(32):
        for s in range(8):
            w = got[8 * lane + s]
            if _got("probe")[8 * lane + s] == 0xFFFFFFFF:
                unwritten += 1
                continue
            rc = divmod(int(w) - 1, 16)
            ok += rc == ir.tensor_acc_position(lane, s)
            rival += rc == inv_pos[(lane, s)]
    res["probe"] = dict(pairs=256, match_tensor_acc_position=ok, unwritten=unwritten,
                        values_distinct=len(set(got.tolist())))
    res["ctl_probe_under_transpose_pos"] = dict(match=rival, must_be_below=256)
    cases = [("loop", "loop", {}), ("ctl_loop_without_rescale", "loop", dict(scale=1)),
             ("ctl_loop_three_trips", "loop", dict(trips=TRIPS - 1))]
    if (OUT / "loopscale" / "output-q1.npz").exists():
        cases += [("loopscale", "loopscale", {}), ("ctl_loopscale_without_rescale", "loopscale", dict(scale=1))]
    for arm in ("far", "farfold", "fast"):
        if (OUT / arm / "output-q1.npz").exists():
            cases += [(arm, arm, dict(far=True)), ("ctl_%s_offset_ignored" % arm, arm, {})]
    if (OUT / "sg4h" / "output-q1.npz").exists():
        cases += [("sg4h", "sg4h", dict(M=16 * SG * HEADS)),
                  ("ctl_sg4h_head0_everywhere", "sg4h", dict(M=16 * SG * HEADS, head0=True))]
    if (OUT / "sg4" / "output-q1.npz").exists():
        cases += [("sg4", "sg4", dict(M=16 * SG)), ("ctl_sg4_without_rescale", "sg4", dict(M=16 * SG, scale=1)),
                  ("ctl_sg4_three_trips", "sg4", dict(M=16 * SG, trips=TRIPS - 1))]
    for name, arm, kw in cases:
        want = lane_words(loop_reference(OUT / arm, **kw))
        g = _got(arm)[:len(want)]
        res[name] = dict(differing_words=int((g != want).sum()), unwritten=int((g == 0xFFFFFFFF).sum()),
                         max_abs_err=float(np.max(np.abs(g.view("<f4").astype(np.float64) -
                                                         want.view("<f4").astype(np.float64)))))
    (OUT / "compare.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "prepare":
        prepare()
    elif cmd == "run" and len(sys.argv) == 3 and sys.argv[2] in ARMS:
        run(sys.argv[2])
    elif cmd == "compare":
        compare()
    else:
        raise SystemExit(__doc__)
