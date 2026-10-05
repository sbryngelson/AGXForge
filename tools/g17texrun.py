#!/usr/bin/env python3
"""ONE TEXTURE KERNEL, END TO END: authored here, dispatched against a real MTLTexture.

    A[400] = TX.read(x, y).x        with x and y in registers this compiler computed

IT READS. Four dispatches, status 0, my code and my constant program throughout:

    coordinate (5, 1) in registers          A[400] = 1012 = texel(5, 1)   CORRECT
    coordinate (2, 3) in registers          A[400] = 3009 = texel(2, 3)   CORRECT
    x = t & 7, y = 3, all 32 lanes race     A[400] = 3007                 ambiguous
    x = t & 7, y = 3, ONLY LANE 31 STORES   A[400] = 3014 = texel(7, 3)   CORRECT

WHAT HAD TO BE TRUE FOR THIS TO EXIST:

    the texture is slot 7, a DENSE index over the textures a function uses, not the Metal binding
      index - five constructed kernels, tools/g17texture.py --selector
    the coordinate is NOT in the instruction - two reads at different coordinates are byte
      identical - so the backend publishes it into [op4+0*4] and [op4+2*4] and the read takes no
      coordinate operand
    op592 takes a REGISTER source with an op4 destination, which Apple never emits, and operand 2
      must carry 1048576 rather than the 16777216 an op0 destination pins it to. Getting that
      wrong is silent: status 0, the fetch runs, every lane reads texel (0, 0)
    only the fourteen-byte SLOT store reads a fetch - not op17229, not an alu.12 - so the
      observable is one slot and a per-lane answer needs a branch to leave one writer
    a host path that binds an MTLTexture at all - spike/accel/textest.mm, its own dylib, because
      libaccel.dylib is shared with a live session

THE EXPECTATION IS DISCRIMINATING. The texture holds texel(x, y) = y*1000 + x + 7, so a wrong
texel, a wrong ROW, a broadcast coordinate and an unbound texture are four different wrong answers
rather than one.

    python3 tools/g17texrun.py            the per-lane read, one lane storing
    python3 tools/g17texrun.py 5 1        a uniform coordinate in registers, expect texel(5, 1)
"""
import ctypes, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(T, "spike", "accel", "re"))

W, H = 8, 4
THREADS = W * H
# THE ROW the per-lane kernel reads. Row 0 would make texel(0,0) - the answer a coordinate that
# never arrived gives - one of the answers a working kernel could give.
ROW = 3
SCRATCH = os.path.expanduser("~/.cache/agxforge/agx/texrun")
DYLIB = os.path.join(T, "spike", "accel", "libtextest.dylib")


def texture():
    """The pattern, row-major. Distinct per texel, and the row is visible in the value."""
    return [y * 1000 + x + 7 for y in range(H) for x in range(W)]


def kernel_ir_op17235(x_const=None, y_const=None):
    """THE FETCH-CONSUMER CONTROL (handoff 10t; integration's 1a88f45e): kernel_ir with ONE change -
    the fetch is stored through the one-component slot store, Apple's op17235, the only consumer
    measured to read a fetch, with a stated zero companion at slot 403 - instead of the two-component
    op17244 whose ability to read a fetch is untested. Same coordinates, same guard, same buffers;
    the pair differs in the store form and in which neighbour slot carries the companion zero
    (401 for op17244, 403 for op17235). Word 400 is the discriminator: under 'op17244 reads the
    fetch' both programs write the texel there; under 'op17244 returns the register's prior
    contents' only this one does."""
    return _kernel_ir(x_const, y_const, consumer="op17235")


def kernel_ir(x_const=None, y_const=None):
    return _kernel_ir(x_const, y_const, consumer="op17244")


def executable_ir(x_const=None, y_const=None, consumer="op17244"):
    """THE SHAPE THAT RUNS (handoff 10t, agreed with the linker): kernel_ir without the F(1) buffer that
    nothing references. It was declared to fit ty-2d's borrowed section; Apple's own compile of that
    shape (family member M0) ELIMINATES an unreferenced buffer, so a section with it would have no
    witness anywhere. Dropping it leaves user 0 at rank 2 / offset 4 - the store moves nothing - and
    gives exactly M0's bindings [44, 48, 0]. The two-buffer programs stay retained as the frontier
    record; they are not what runs."""
    return _kernel_ir(x_const, y_const, consumer=consumer, declare_unused=False)


def _kernel_ir(x_const=None, y_const=None, consumer="op17244", declare_unused=True):
    """The texture kernel. With no constants, ONE LANE STORES and the coordinate is per-lane.

    THE STORE IS THE CONSTRAINT. op17229 - the indexed store - is measured not to read a fetch, and
    neither is an alu.12; both return the destination register's prior contents and six
    instructions of distance do not help. The only consumer that reads a fetch is the fourteen-byte
    SLOT store, which writes ONE location, so 32 lanes with 32 coordinates race for it and the
    winner is unknown. A branch fixes that: `t > 30` leaves exactly one writer, lane 31, whose own
    x is 7. A publish that broadcast lane 0's register could only ever return texel(0, y), so the
    two answers are distinguishable with one dispatch.

    The relation has to be `gt` - cmp.pair.imm's relation encoding is unresolved and every other
    comparison is a blocked rung - so the guarded lane is the last one rather than a chosen one.

    With x_const and y_const the coordinate is UNIFORM but still computed: (t & 0) + c, so it
    travels the same register path while every lane fetches the same texel and no branch is needed.
    Two different constants giving two different correct texels is what separates a coordinate that
    arrives from a fixed artifact.
    """
    import g17ir as ir
    # BUFFERS u(0) AND f(1), the store going to buffer 0, so this runs on ty-2d's OWN metadata
    # section - known good, because Apple's code reads a texture through it. g17mdgen composes
    # sections from recorded classes keyed on binding count and has no notion of a texture, so it
    # cannot declare one; the CODE is authored here and inherits nothing.
    f = ir.Function("texread", [ir.Buffer("U", 0)] + ([ir.Buffer("F", 1)] if declare_unused else []))
    if x_const is None:
        e = f.block("entry"); th = f.block("then"); jn = f.block("join")
        b = ir.Builder(f, e)
        t = b.builtin("thread_position_in_grid", name="t")
        x = getattr(b, "and")(t, ir.Imm(W - 1), name="x")
        y = b.add(getattr(b, "and")(t, ir.Imm(0), name="z"), ir.Imm(ROW), name="y")
        v = b.texture_read(x, y, tex=0, name="v")
        b.br_cond(b.cmp(t, THREADS - 2, "gt", name="p"), th, jn)
        b.at(th)
        _emit_store(b, f, v, consumer)
        b.br(jn)
        b.at(jn); b.ret()
        return f
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    z = getattr(b, "and")(t, ir.Imm(0), name="z")
    x = b.add(z, ir.Imm(x_const), name="x")
    y = b.add(z, ir.Imm(y_const), name="y")
    _emit_store(b, f, b.texture_read(x, y, tex=0, name="v"), consumer)
    b.ret()
    return f


def _emit_store(b, f, v, consumer):
    import g17ir as ir
    if consumer == "op17244":
        b.store(f.buffers[0], ir.Imm(400), v)
    elif consumer == "op17235":        # Apple's exact texel store: sub-form 01 with the one-component encoding (two bytes from the delivered store)
        b.store_fetch(f.buffers[0], ir.Imm(400), v, b.const(0, name="companion"), components=1)
    elif consumer == "op17235_n2":     # sub-form 01 with the delivered store's two-component field: ONE byte from the delivered store, companion at 401
        b.store_fetch(f.buffers[0], ir.Imm(400), v, b.const(0, name="companion"), components=2)
    else:
        raise ValueError("consumer %r: op17244 (the delivered store), op17235 (Apple's texel store) or op17235_n2 (sub-form 01, one byte from the delivered store)" % consumer)


def _lib():
    L = ctypes.CDLL(DYLIB)
    L.tx_error.restype = ctypes.c_char_p
    L.tx_lib_from_url.argtypes = [ctypes.c_char_p]
    L.tx_pipeline_from_archive.restype = ctypes.c_void_p
    L.tx_pipeline_from_archive.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    L.tx_run.restype = ctypes.c_int
    L.tx_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                         ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
    return L


TY2D = os.path.expanduser("~/.cache/agxforge/agx/ty-2d/out/object/0-0")


def ty2d_sections():
    """ty-2d's OWN metadata, taken from the corpus object rather than a file beside this one.

    g17mdgen composes __GPU_METADATA from recorded classes keyed on binding count and has no notion
    of a texture, so it cannot declare one; without a section that declares a texture there is no
    pipeline to dispatch. Borrowing Apple's is the honest way to keep the CODE the thing under
    test, and taking it from the object means it is reproducible from `python3 tools/g17corpus.py`
    rather than from a scratch directory. The first supplied section this project dispatched HUNG
    THE GPU (ledger/g17-a-guard-passed-and-the-gpu-stopped.toml); this one is Apple's own, and
    Apple's code reads a texture through it.

    ty-2d's entry is 64, which is g17oracle.ENTRY, and its __GPU_LD_MD is the 336-byte TEXTURE
    shape rather than the 216-byte buffer shape g17ldmd.build produces.
    """
    import g17obj
    if not os.path.exists(TY2D):
        raise SystemExit("ty-2d is not built: run `python3 tools/g17corpus.py` first (%s)" % TY2D)
    raw = open(TY2D, "rb").read()
    sects, _syms = g17obj.sections_of(raw)
    out = {}
    for name in ("__GPU_METADATA,__compute", "__GPU_LD_MD,__compute",
                 "__GPU_ARCH_LD_MD,__compute", "__GPU_STATS_MD,__compute"):
        o, sz = sects[name]
        out[name.split(",")[0]] = bytes(raw[o:o + sz])
    return out


def main():
    import numpy as np
    import g17cc, g17oracle, g17program, g17ldmd

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    xc, yc = (int(args[0]), int(args[1])) if len(args) >= 2 else (None, None)
    os.makedirs(SCRATCH, exist_ok=True)
    p = g17cc.compile_function(kernel_ir(xc, yc))
    # A TRIVIAL CONSTANT PROGRAM - `end` plus filler, the only shape this compiler authors. It is
    # not a placeholder: with it the fetch still runs and the coordinate still arrives, which is
    # what retired "a texture read needs a constant program".
    text = bytes.fromhex("0e000000") + g17oracle.FILLER * ((g17oracle.ENTRY - 4) // 2) + p.code
    if len(text) % 16:
        text += g17oracle.FILLER * ((16 - len(text) % 16) // 2)
    print("AUTHORED TEXTURE KERNEL\n")
    for _at, b, m in p.layout:
        print("   %-18s %s" % (m.form, b.hex()))
    print("\n   %d instructions, %d bytes, constant program trivial" % (len(p.layout), len(p.code)))

    sec = ty2d_sections()
    assert g17oracle.ENTRY == 64, "ty-2d's load metadata is measured for entry 64, not %d" % g17oracle.ENTRY
    g17ldmd.build = lambda entry=None, _b=sec["__GPU_LD_MD"]: _b
    g17ldmd.build_arch = lambda _b=sec["__GPU_ARCH_LD_MD"]: _b
    P = g17program.G17Program(text=text, entry=g17oracle.ENTRY, buffers=[0, 1],
                              stats_md=sec["__GPU_STATS_MD"])
    P.metadata = lambda _b=sec["__GPU_METADATA"]: _b
    P._obj = None
    open(SCRATCH + "/x.arc", "wb").write(P.image())
    open(SCRATCH + "/x.lib", "wb").write(P.library())

    L = _lib()
    if L.tx_init() != 0:
        print("\n   no Metal device"); return 1
    if L.tx_lib_from_url((SCRATCH + "/x.lib").encode()) != 0:
        print("\n   library: %s" % L.tx_error().decode()); return 1
    ps = L.tx_pipeline_from_archive((SCRATCH + "/x.arc").encode(), b"k")
    if not ps:
        print("\n   NO PIPELINE: %s" % L.tx_error().decode())
        print("   This is the metadata gate: the image declares no texture.")
        return 1

    tex = texture()
    texa = np.array(tex, dtype=np.uint32)
    N = 4096
    # THE STORE GOES TO BUFFER 0, so the observable is A. Reading the wrong buffer is how this
    # project once scored a kernel against an array nothing ever wrote.
    A = np.zeros(N // 4, dtype=np.uint32); A[0], A[1] = 3, 2
    B = np.zeros(N // 4, dtype=np.uint32)
    C = np.full(N // 4, 0xDEADBEEF, dtype=np.uint32)
    st = L.tx_run(ctypes.c_void_p(ps), texa.ctypes.data, W, H,
                  A.ctypes.data, B.ctypes.data, C.ctypes.data, N, THREADS, 1, 1)
    val = int(A[400])
    print("\n   dispatch status %d %s" % (st, L.tx_error().decode()))
    print("   A[400] = %d" % val)
    if xc is not None:
        print("   texel(%d,%d) = %d" % (xc, yc, tex[yc * W + xc]))
        ok = val == tex[yc * W + xc]
        print("\n   THE COORDINATE ARRIVES." if ok else
              "\n   not the coordinate: %d (texel(0,0) is %d - a coordinate that never arrived)"
              % (val, tex[0]))
    else:
        row = tex[ROW * W:(ROW + 1) * W]
        print("   row %d is %s" % (ROW, row))
        if val == row[W - 1]:
            print("\n   EACH LANE PUBLISHES ITS OWN REGISTER: lane %d fetched texel(%d,%d)."
                  % (THREADS - 1, W - 1, ROW))
            ok = True
        elif val == row[0]:
            print("\n   texel(0,%d): the publish is broadcasting lane 0's register" % ROW)
            ok = False
        else:
            print("\n   neither: %d" % val)
            ok = False
    json.dump({"status": int(st), "A400": val, "coord": [xc, yc], "texture": tex, "ok": bool(ok)},
              open(os.path.join(T, "isa", "g17-execution-texture-results.json"), "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
