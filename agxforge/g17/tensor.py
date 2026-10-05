#!/usr/bin/env python3
"""The TENSOR SEQUENCE REGISTRY: what the compiler can author on the tensor side, and what it
still has to inherit, per matmul shape.

WHY A REGISTRY AND NOT AN EMITTER. isa/tensor-isa.toml is explicit about the boundary:

    "Only the horizontal (N) extent is mapped. M, K, dtype, accumulate-vs-multiply, transpose,
     and simdgroup count have not been located in these 12-byte units."
    "No instruction has been synthesised from nothing: every mutation so far rewrites fields
     inside compiler-emitted instructions. Composing a novel instruction SEQUENCE remains [open]."

SUPERSEDED 2026-09-17, so the quotation above is history rather than the boundary. The
linker/resource session's lowering authors a GEMM from its shape - `agxforge/g17/tensorgemm.py`,
`docs/archive/g17-tensor-lowering-handoff.md` - with receipts: 35 repository-authored images across
operand types, simdgroup counts and transposes, every instruction synthesised from the field maps
and the certified ledger with decode-back guards. M, N, the operand types,
accumulate-vs-multiply, both transposes and the simdgroup count are authorable fields or sequence
choices, and K is a CHAIN LENGTH (one op5106 per 16-wide K step per tile, the first op5107) rather
than a field. "Cannot yet be built from its shape" and "no instruction has been synthesised from
nothing" are both false as of that work.

This registry is NOT superseded and is not deleted: it authors every recovered field onto a
reference sequence, which remains the only path for the shapes the general lowering REFUSES.
MEASURED on the snapshot integrated here rather than copied from the handoff's list, because two
of the four refusals that list names have since been LIFTED upstream - int8 partial tiles and int8
odd leading dimensions both author now, closed by the masked one-word load - and writing them here
as refusals would have shipped a false statement and invited someone to reinstate them. What
actually refuses:

    programs shorter than the TENSOR metadata class's measured tail (16x16x16: "a tensor program
      of 25 instructions carries no slot 32; no witness has that shape")
    simdgroup counts other than 1, 2 or 4
    M not a multiple of 16 x simdgroups, when simdgroups > 1 (at simdgroups = 1 M is unconstrained;
      the boundary masks handle it)

What CAN be done on the registry path is to author every recovered field onto a reference sequence
of the right shape:

    AUTHORED   a_dtype, b_dtype (tensor.mac byte6[2], byte7[6])
               k_slice, enable  (tensor.mac byte3[4], byte3[5])
               N extent W and offset O (tensor.bound byte8/byte9, causal on 256/256 values)
    INHERITED  the number of mac units and their order, byte1's accumulator one-hot, byte2,
               byte3's remaining bits, byte5, byte9 - and therefore M, K and the mode

Keeping that split in a registry rather than inside an emitter is what stops the compiler from
appearing to generate a tensor kernel it is in fact copying. A shape with no reference sequence
BLOCKS, and says so.
"""
import os, sys, glob, re, collections
_T = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_T))
TOOLS = os.path.join(ROOT, "tools")
from agxforge.g17 import agxdis, machobj

CACHE = os.path.expanduser("~/.cache/agxforge/agx")
# sc-* are shapes compiled by spike/accel/re/shapecount.py, a COMPILE-ONLY differential. They
# widen the set of matmul shapes the compiler can lower, which is the only way that set grows
# until a tensor sequence can be synthesised rather than referenced.
SHAPE = re.compile(r"^(?:ac2?|sc)-(\d+)x(\d+)x(\d+)$")

class TensorSeq:
    """One reference matmul: its mac units and bound units, with the shape they came from."""
    def __init__(self, shape, source, macs, bounds):
        self.shape = shape; self.source = source; self.macs = macs; self.bounds = bounds
    @property
    def authored_bits(self):
        # per mac unit: a_dtype, b_dtype, k_slice, enable = 4 bits; per bound unit: W (4 bits
        # representable) + O (4 bits) = 8 bits of byte8/byte9.
        return 4 * len(self.macs) + 8 * len(self.bounds)
    @property
    def total_bits(self):
        return 8 * (agxdis.MAC_UNIT * len(self.macs) + agxdis.UNIT * len(self.bounds))
    def __repr__(self):
        return ("%-16s %3d macs %2d bounds  %4d/%4d bits authored (%.1f%%)  %s"
                % ("%dx%dx%d" % self.shape, len(self.macs), len(self.bounds),
                   self.authored_bits, self.total_bits,
                   100.0 * self.authored_bits / max(self.total_bits, 1), self.source))

# --- K: A DECODE CORRELATION, NOT AN ENCODING (retarget_k is NOT used by the compiler) -------------------------------------------------------------------------
# K is NOT in the tensor.mac units. For K = 64..224 at M=N=32 all sixteen macs are byte-identical
# and __text is the same size; K lives entirely in the single tensor.bound.b unit, as three bits:
#
#     K / 32  =  byte8[5]  +  2 * byte10[7]  +  4 * byte11[0]
#
# Fitted on four cached shapes and then PREREGISTERED for two that were not in the cache - the
# whole twelve-byte unit predicted before either was compiled, both exact.
#
# THAT IS AS FAR AS IT GOES. Authoring these bits does NOT change what the kernel computes: the
# patched kernel's output is bit-identical to K=64 while Apple's native 32x32x160 differs
# completely. A full __text diff - the check I skipped, having compared only the mac units - shows
# K=64 and K=96 differing in FIVE bytes, only one of them inside this unit. So these three bits
# are a correlate that a decoder may read, not the encoding a compiler can write.
# ledger/g17-tensor-k-field-retracted.toml
#
# THIS SENTENCE IS STILL TRUE; THE INFERENCE DRAWN FROM IT WAS NOT. Reviewed 2026-09-17 against
# the general lowering, which DOES author K - as a CHAIN LENGTH, one op5106 per 16-wide K step per
# tile with the first being op5107, and not as a field anywhere. So these three bits remain a
# correlate and remain unusable for authoring, exactly as retracted; what is superseded is the
# conclusion that a compiler therefore cannot express K. Annotated rather than deleted, because
# removing a true retraction to make room for a later capability is how the retracted reading
# comes back.
#
# retarget_k is kept for DECODE experiments and is deliberately not wired into the compiler.
#
# RANGE. Only K/32 in 2..7. K=256 needs a fourth bit the fit cannot supply and the compiler
# changes strategy there (__text grows at 384), so retargeting refuses outside the fitted range
# rather than extrapolating - the same rule that keeps a shape without a reference from being
# faked.
K_QUOTIENT_RANGE = range(2, 8)

def retarget_k(seq, K):
    """A reference sequence at (M,N,K0) serving (M,N,K), by authoring the K field."""
    q, r = divmod(K, 32)
    if r or q not in K_QUOTIENT_RANGE:
        raise ValueError("K=%d is outside the recovered field: K must be a multiple of 32 with "
                         "K/32 in %d..%d (K=%d needs a bit the fit does not cover)"
                         % (K, K_QUOTIENT_RANGE[0]*32, K_QUOTIENT_RANGE[-1]*32, K))
    bounds = [(o, bytearray(by), nm) for o, by, nm in seq.bounds]
    n = 0
    for o, u, nm in bounds:
        if nm != "tensor.bound.b": continue
        u[8]  = (u[8]  & ~0x20) | ((q & 1) << 5)
        u[10] = (u[10] & ~0x80) | (((q >> 1) & 1) << 7)
        u[11] = (u[11] & ~0x01) | ((q >> 2) & 1)
        n += 1
    if n != 1:
        raise ValueError("expected exactly one tensor.bound.b to carry K, found %d" % n)
    out = TensorSeq((seq.shape[0], seq.shape[1], K), seq.source + " (K authored)",
                    list(seq.macs), [(o, bytes(u), nm) for o, u, nm in bounds])
    out.k_authored = True
    return out

def decode_k(seq):
    """The K a sequence's bound unit encodes, or None."""
    for _, by, nm in seq.bounds:
        if nm == "tensor.bound.b":
            return 32 * (((by[8] >> 5) & 1) | (((by[10] >> 7) & 1) << 1) | ((by[11] & 1) << 2))
    return None

# --- COMPOSING A MAC SEQUENCE FOR A SHAPE WITH NO REFERENCE --------------------------------------
# This is the piece the tensor_unseen_shape rung was blocked on: "composing a novel tensor
# instruction SEQUENCE is unsolved - M, K, mode and transpose are not located in the 12-byte
# units". The units were the wrong frame. Apple's decoder frames a MAC as a TEN-byte instruction
# with three register operands, and once the corrected reference kernels are read that way
# (spike/accel/re/tgen.py - the old ones have their output slice extents reversed and compute only
# min(M,N)^2), the sequence is a plain nested loop.
#
# THE LAW, validated below against every corrected reference:
#
#     rows = M/16, cols = N/16                        each 16x16 sub-tile has one accumulator
#     accumulator index for (row, col) = 1 + row*cols + col            row-major over the grid
#     operand A is indexed by (row, k_slice)          two per row block
#     operand B is indexed by (col, k_slice)          two per column block
#     issue order: for col descending, for row descending, emit k_slice 1 then 0
#     and the whole list is issued TWICE
#
# so macs = rows * cols * 2 * 2 = M*N/64, which is the count law from the other direction.
#
# THE REGISTER NUMBERS ARE NOT PART OF THE LAW. Apple's are a descending allocation over a shared
# pool that skips whatever the A operands took - in 16x64 the B registers are 40,44,...,76 with 44
# and 52 missing because A holds them. So a composer picks its own, and what has to be consistent
# is the STRUCTURE: which accumulator each MAC accumulates into, and which row and column its
# operands come from.
#
# SCOPE: exact for M*N <= 1536. At 2048 and above the issue order changes and the counts stop
# following M*N/64 - 64x64 emits 32 MACs where the law says 64 - which is the saturation regime
# ledger/g17-tensor-mac-count-law.toml identified from the counts alone. Refused rather than
# extrapolated.
# THE COUNT LAWS AND THE STRUCTURE LAW HAVE DIFFERENT RANGES, and this is the number for the one
# implemented here. The ISA agent derived macs = 4mn (the tile grid's AREA) and
# loads = 8(m+n) (its PERIMETER) independently, and those hold up to and including M*N = 2048 -
# 32x64 and 64x32 fit all four counts exactly. The ISSUE ORDER does not: raising this to 2048 and
# re-running --check made those two shapes fail structurally, which is how the difference was
# found. So counts extend to 2048 and the sequence law is validated to 1536, and this refuses
# above what it can reproduce rather than above what some law covers.
# MEASURED ACROSS THE WHOLE CORPUS GRID, 2026-09-05, rather than assumed from where the sample
# stopped. compose_macs' issue order was checked against Apple's own MAC sequence for all sixteen
# shapes M,N in {16,32,48,64} at K=64, comparing the pattern of accumulator repeats and the k_slice
# alternation:
#
#     matches   16x32 16x48 16x64  32x16 32x32 32x48 32x64  48x16 48x32  64x16 64x32
#     differs   16x16                     the smallest shape, a different regime in its loads too
#     Apple has MORE macs   48x48 (48 v 36)  48x64 (64 v 48)
#     Apple has FEWER       64x48 (32 v 48)  64x64 (32 v 64)   - it loops where the composer unrolls
#
# So the law holds to M*N = 2048 (32x64 and 64x32 both match) and breaks above it in BOTH
# directions, which is why the ceiling is a refusal and not a clamp: above it the issue order is a
# different program, not a longer one.
COMPOSE_MAX_TILE = 2048

def compose_macs(M, N, K=64):
    """The MAC sequence for (M,N,K) as (accumulator index, row, col, k_slice), in issue order.

    Indices, not registers: the caller assigns registers, because the numbering is an allocation
    and only the structure is architectural."""
    if K != 64:
        raise ValueError("compose_macs is validated at K=64 only; K=%d is not covered" % K)
    if M % 16 or N % 16:
        raise ValueError("M and N must be multiples of the 16x16 sub-tile (got %dx%d)" % (M, N))
    if M * N > COMPOSE_MAX_TILE:
        raise ValueError("M*N = %d is in the saturation regime above %d, where the issue order "
                         "changes and the count stops following M*N/64. Refused rather than "
                         "extrapolated." % (M * N, COMPOSE_MAX_TILE))
    rows, cols = M // 16, N // 16
    seq = []
    for _repeat in range(2):
        for col in range(cols - 1, -1, -1):
            for row in range(rows - 1, -1, -1):
                for k_slice in (1, 0):
                    seq.append((1 + row * cols + col, row, col, k_slice))
    return seq


# THE ACCUMULATOR INITIALISER, AND WHEN IT MAY BE USED.
#
# op5107 declares op5106's operands MINUS the accumulator use, so it computes A*B where op5106
# computes A*B + acc: it is the accumulator initialiser, and it replaces the 8mn+1 movimm run that
# zeroes the accumulators. The peer established the reading from the corpus - its count equals the
# number of distinct accumulators in 32 of 32 objects that use it, and it is the first MAC written
# to each accumulator in 125 of 125.
#
# IT MAY ONLY BE USED WHERE THE MAC DOES NOT RE-EXECUTE, and that is a rule about the program, not
# about the instruction. Substituting it for the first MAC of each accumulator in tg-16x48x64 -
# whose K loop contains half its MACs - gave 7 of 768 cells, because it reset the accumulator on
# every iteration. The corpus agrees without being asked: every object that uses op5107 has NO
# back edge at all (ac2-32x32x32, c16-base, cm-M64), and tg-16x48x64, which has one, uses the
# movimm run instead.
#
# That failure is also the positive evidence for the reading: if op5107 accumulated, replacing the
# first MAC of an already-zeroed accumulator would have changed nothing.
# ledger/g17-tensor-accumulator-initialiser.toml
MAC_INIT_OPCODE = 5107
MAC_ACC_OPCODE = 5106


def mac_is_initialiser(seq, i):
    """True if MAC i is the first in `seq` to write its accumulator, so op5107 may carry it -
    PROVIDED the sequence is straight-line. Callers with a loop must use the movimm run."""
    k = seq[i][0]
    return all(seq[j][0] != k for j in range(i))


def compose_mac_bytes(M, N, acc_regs, a_regs, b_regs, template, K=64):
    """The MAC sequence as BYTES, from the law plus a register assignment.

    acc_regs is indexed by the law's accumulator index (1-based); a_regs by (row, k_slice) and
    b_regs by (col, k_slice), each ordered so that index 2*row + k_slice selects the register.
    The caller owns the assignment - the numbering is an allocation, not architecture - and this
    turns the structure into instructions through g17asm's canonical encoder.
    """
    from agxforge.g17 import asm as g17asm
    seq = compose_macs(M, N, K)
    n = len(seq) // 2                       # distinct MACs; the list is issued twice
    out = []
    for i, (k, row, col, ks) in enumerate(seq):
        if k > len(acc_regs):
            raise ValueError("accumulator index %d but only %d registers supplied" % (k, len(acc_regs)))
        pos = i % n
        if pos == 0:
            seen, tag = set(), 0             # the tag counter restarts with each repeat
        # A MAC takes the NEXT wait tag exactly when it is the first user, within this repeat, of
        # at least one of its two tile operands; otherwise it carries none. That is the ISA agent's
        # rule - "consumes a load newer than anything consumed earlier in the repeat" - written in
        # the composer's own coordinates, where a tile operand IS a load.
        fresh = [o for o in (("A", row, ks), ("B", col, ks)) if o not in seen]
        seen.update(fresh)
        this_tag = tag if fresh else None
        if fresh:
            tag += 1
        out.append(g17asm.encode_tensor_mac(
            acc_regs[k - 1], a_regs[2 * row + ks], b_regs[2 * col + ks], template,
            first=int(pos == 0), col0=int(col == 0), row0=int(row == 0), kslice=ks,
            tag=this_tag))
    return out


def mac_pools(macs):
    """Recover (acc_regs, a_regs, b_regs) from a decoded reference, in the law's index order.

    Accumulators ascend with the law's index; A and B are ordered so that 2*row+k_slice indexes
    them, which is the order they appear in ascending register number - checked by
    compose_mac_bytes reproducing the reference it came from."""
    accs = sorted({a for a, _, _ in macs})
    a_regs = sorted({x for _, x, _ in macs})
    b_regs = sorted({y for _, _, y in macs})
    return accs, a_regs, b_regs


def check_compose(verbose=True):
    """Validate compose_macs against every corrected reference kernel.

    Apple's registers are mapped to indices BY RANK - the accumulators sorted ascending become
    1..n, the A operands paired two-per-row, the B operands two-per-column - because the numbers
    are an allocation and only the structure is being checked."""
    import glob, os, re, subprocess, tempfile, sys as _sys
    from agxforge.g17 import machobj, asm as g17asm
    dis = os.path.join(TOOLS, "agx3dis")
    ok = bad = refused = 0
    for d in sorted(glob.glob(os.path.expanduser("~/.cache/agxforge/agx/tg-*"))):
        m0 = re.search(r"(\d+)x(\d+)x(\d+)$", os.path.basename(d))
        if not m0 or not os.path.exists(d + "/out/object/0-0"):
            continue
        M, N, K = (int(x) for x in m0.groups())
        try:
            pred = compose_macs(M, N, K)
        except ValueError:
            refused += 1
            if verbose: print("  %-12s refused (outside the validated regime)" % ("%dx%dx%d" % (M, N, K)))
            continue
        loc = machobj.locate(d + "/s.arc.metallib", d + "/out/object/0-0")
        f, sz = agxdis.sections(loc["obj"])
        t = bytes(loc["obj"][f:f+sz]); e = loc["syms"]["_agc.main"]
        with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
            fh.write(t); fh.flush()
            out = subprocess.run([dis, fh.name, str(e), str(len(t)-e), "--pc", str(e)],
                                 capture_output=True, text=True).stdout
        got = []
        for line in out.splitlines():
            p = line.split()
            if len(p) >= 3 and p[1] != "bad" and p[2] == "5106":
                o, l = int(p[0], 16), int(p[1])
                dd = g17asm.decode_tensor_mac(t[o:o+l])
                got.append((dd["acc"], dd["a"], dd["b"]))
        rank = {v: i + 1 for i, v in enumerate(sorted({a for a, _, _ in got}))}
        ar = {v: i for i, v in enumerate(sorted({x for _, x, _ in got}))}
        br = {v: i for i, v in enumerate(sorted({y for _, _, y in got}))}
        good = len(got) == len(pred) and all(
            rank[a] == k and ar[x] // 2 == row and br[y] // 2 == col
            for (a, x, y), (k, row, col, _ks) in zip(got, pred))
        ok += good; bad += not good
        if verbose:
            print("  %-12s %2d macs  %s" % ("%dx%dx%d" % (M, N, K), len(got),
                                            "MATCH" if good else "DIFFERS"))
    if verbose:
        print("compose_macs: %d references reproduced exactly, %d differ, %d outside the regime"
              % (ok, bad, refused))
    return ok, bad, refused


def build():
    out = {}
    for d in sorted(glob.glob(CACHE + "/*")):
        m = SHAPE.match(os.path.basename(d))
        if not m: continue
        arc, obj = d + "/s.arc.metallib", d + "/out/object/0-0"
        if not (os.path.exists(arc) and os.path.exists(obj)): continue
        try:
            loc = machobj.locate(arc, obj)
            f, sz = agxdis.sections(loc["obj"]); text = loc["obj"][f:f+sz]
        except Exception:
            continue
        macs, bounds, p = [], [], 0
        while p < len(text):
            if agxdis.is_mac(text[p:p+agxdis.MAC_UNIT]):
                macs.append((p, bytes(text[p:p+agxdis.MAC_UNIT]))); p += agxdis.MAC_UNIT; continue
            d12 = agxdis.decode(text[p:p+agxdis.UNIT])
            if d12:
                bounds.append((p, bytes(text[p:p+agxdis.UNIT]), d12["name"])); p += agxdis.UNIT; continue
            p += 2
        shape = tuple(int(x) for x in m.groups())
        if macs:
            # Prefer the ac2 build when both exist: it is the newer probe generation.
            key = shape
            if key not in out or os.path.basename(d).startswith("ac2-"):
                out[key] = TensorSeq(shape, os.path.basename(d), macs, bounds)
    return out

# THE cm- LEVER FAMILY: nine probes at identical tensor extents and slices, each varying ONE thing
# from cm-base (32,32,64, no transpose, half, one simdgroup). It was in the cache the whole time and
# nothing had diffed it. LEVERS[tag] = what that tag changes.
LEVERS = {
    "cm-transposeL": "transpose A",
    "cm-transposeR": "transpose B",
    "cm-M64":        "M 32->64",
    "cm-N64":        "N 32->64",
    "cm-K128":       "K 64->128",
    "cm-dtypeBf":    "dtype half->bf16",
    "cm-simd2":      "simdgroups 1->2",
    "cm-bufferA3":   "a third buffer",
}


def mac_units(tag):
    """The mac units of one built probe, in program order."""
    from agxforge.g17 import agxdis, machobj
    d = os.path.join(CACHE, tag)
    loc = machobj.locate(d + "/s.arc.metallib", d + "/out/object/0-0")
    f, sz = agxdis.sections(loc["obj"])
    text = bytes(loc["obj"][f:f + sz])
    out, p = [], 0
    while p < len(text):
        u = text[p:p + agxdis.MAC_UNIT]
        if agxdis.is_mac(u):
            out.append(bytes(u)); p += agxdis.MAC_UNIT; continue
        if agxdis.decode(text[p:p + agxdis.UNIT]):
            p += agxdis.UNIT; continue
        p += 2
    return text, out


def lever_bits(base, tag):
    """(base text len, tag text len, base macs, tag macs, {(byte,bit): how many macs differ}).

    The bit dict is meaningful only when the two mac COUNTS match; a different count means the
    lever changed the shape of the sequence rather than a field inside it, and comparing unit i to
    unit i is then comparing different things.
    """
    import collections
    tb, mb = mac_units(base)
    tt, mt = mac_units(tag)
    if len(mb) != len(mt):
        return len(tb), len(tt), len(mb), len(mt), None
    bits = collections.Counter()
    for x, y in zip(mb, mt):
        for i, (p_, q) in enumerate(zip(x, y)):
            for k in range(8):
                if ((p_ >> k) & 1) != ((q >> k) & 1):
                    bits[(i, k)] += 1
    return len(tb), len(tt), len(mb), len(mt), dict(bits)


def mode_lever(a="ac2-32x32x64", b="lever_mode_mac"):
    """The one-variable pair for accumulate-vs-multiply, and the bits it moves.

    isa/tensor-isa.toml lists "accumulate-vs-multiply" among the fields not located in these
    12-byte units. It is located, and the pair that does it was already in the cache: 527 probes
    carry mode::multiply and 4 carry mode::multiply_accumulate, all four at shape 32x32x64, which
    ac2-32x32x64 also has. Same M, N, K; same dtypes; one keyword different in the source.

        byte0 bit 7    differs in 16 of 16 macs
        byte2 bit 7    differs in  8 of 16 macs
        nothing else differs at all

    and everything decode_mac reports - the accumulator one-hot, k_slice, enable, a_dtype,
    b_dtype - is unchanged position for position. The three sib-lever_mode_mac builds are
    byte-identical to lever_mode_mac, so the bytes are a stable function of the source.

    IT IS NOT A MODE FLAG, and saying so is the difference between this being usable and being
    wrong. byte0[7] varies WITHIN a single mode - the pattern is 1111000011110000 for multiply and
    exactly its complement for accumulate - so the bit carries something positional that the mode
    inverts, rather than announcing the mode. Both halves of the sixteen are identical, so it does
    not distinguish a first K pass from a second either, which is what an initialise bit would do.

    SO: located, not authorable. Emitting multiply_accumulate needs the positional part understood,
    and this changes the inherited bit count by nothing.

    MEASURED ON: one shape, half/half dtypes, execution_simdgroups<1>. Nothing executed.
    """
    import os
    from agxforge.g17 import agxdis, machobj

    def units(tag):
        d = os.path.join(CACHE, tag)
        loc = machobj.locate(d + "/s.arc.metallib", d + "/out/object/0-0")
        f, sz = agxdis.sections(loc["obj"])
        text = bytes(loc["obj"][f:f + sz])
        out, p = [], 0
        while p < len(text):
            u = text[p:p + agxdis.MAC_UNIT]
            if agxdis.is_mac(u):
                out.append(bytes(u)); p += agxdis.MAC_UNIT; continue
            if agxdis.decode(text[p:p + agxdis.UNIT]):
                p += agxdis.UNIT; continue
            p += 2
        return out

    import collections
    x, y = units(a), units(b)
    bits = collections.Counter()
    for u, v in zip(x, y):
        for i, (p_, q) in enumerate(zip(u, v)):
            for k in range(8):
                if ((p_ >> k) & 1) != ((q >> k) & 1):
                    bits[(i, k)] += 1
    return len(x), len(y), dict(bits)


def shape_collisions():
    """Shapes whose TENSOR code is byte-identical, and whether their sources differ.

    The registry's standing claim is that M, K, dtype, accumulate-vs-multiply, transpose and
    simdgroup count "have not been located in these 12-byte units". For M there is a reason, and it
    is not that the field is hidden: past a point it is NOT THERE.

        M = 80, 96 and 128 at N=32, K=64   byte-identical macs AND bounds
        M = 48 and 64 at N=48, K=64        byte-identical macs AND bounds

    The sources are genuinely different - ac2-96x32x64 and ac2-128x32x64 differ in exactly
    matmul2d_descriptor(96,32,64,...) vs (128,32,64,...) and the slice extents, nothing else - and
    the WHOLE PROGRAMS differ (3,078 bytes each, different hashes). So the difference between a
    96-row and a 128-row matmul is entirely in the scalar code around the tensor block, and the
    tensor units are a fixed tile that says nothing about how many times it runs.

    WHAT THIS DOES AND DOES NOT DO TO THE DEBT. It does not reduce it by one bit: no field became
    authored. What it does is remove one of the four named blocked fields from the search, because
    an M field cannot be recovered from units that do not vary with M. The route to a tensor
    emitter is generating the scalar code that surrounds the tile, not finding M inside it.

    NOT A LAW ABOUT MAC COUNT. M does move the number of mac units elsewhere in the range - 16, 24
    and 32 for M = 32, 48, 64 at N=32,K=64 - but `macs = min(M*N/64, 32)` has five exceptions in
    twenty-eight shapes, so no closed form is claimed here. The identity above is exact and needs
    none.

    MEASURED ON THIS POPULATION: 28 shapes, execution_simdgroups<1>, mode::multiply. A different
    simdgroup count is not covered.
    """
    import collections
    seqs = build()
    groups = collections.defaultdict(list)
    for shape, s in sorted(seqs.items()):
        blob = b"".join(b for _p, b in s.macs) + b"|" + b"".join(b for _p, b, _n in s.bounds)
        groups[blob].append(shape)
    return {k: v for k, v in ((tuple(v[0]), v) for v in groups.values()) if len(v) > 1}


def main():
    if "--levers" in sys.argv:
        print("THE cm- LEVER FAMILY, each probe varying ONE thing from cm-base\n")
        tb, mb = mac_units("cm-base")
        print("   cm-base  32,32,64 half, one simdgroup: %d bytes of text, %d mac units\n"
              % (len(tb), len(mb)))
        print("   %-15s %-18s %-7s %s" % ("probe", "changes", "macs", "what moves in the macs"))
        for tag in sorted(LEVERS):
            try:
                lb, lt, nb, nt, bits = lever_bits("cm-base", tag)
            except Exception as e:
                print("   %-15s %-18s ERROR %s" % (tag, LEVERS[tag], str(e)[:40])); continue
            if bits is None:
                print("   %-15s %-18s %d->%-4d the SEQUENCE changes, not a field in it"
                      % (tag, LEVERS[tag], nb, nt))
            elif not bits:
                print("   %-15s %-18s %-7d BYTE-IDENTICAL - not in the mac units at all"
                      % (tag, LEVERS[tag], nt))
            else:
                print("   %-15s %-18s %-7d %s" % (tag, LEVERS[tag], nt,
                      ", ".join("byte%d[%d] in %d" % (b, k, c) for (b, k), c in sorted(bits.items()))))
        print("\n   dtype is the CONTROL: byte6[2] and byte7[6] are the two fields the registry")
        print("   already authors, and the lever recovers exactly those - so the method is")
        print("   reading real fields and not noise.")
        print("   Nothing here executed. One shape family, one probe generation.")
        return 0
    if "--mode" in sys.argv:
        na, nb, bits = mode_lever()
        print("ACCUMULATE-vs-MULTIPLY, from the one-variable pair\n")
        print("   ac2-32x32x64 (multiply) %d macs   lever_mode_mac (accumulate) %d macs" % (na, nb))
        for (by, bi), c in sorted(bits.items()):
            print("   byte%d bit%d differs in %2d of %d macs" % (by, bi, c, na))
        print("\n   Everything decode_mac reports is unchanged position for position.")
        print("   NOT a mode flag: byte0[7] is 1111000011110000 under multiply and exactly its")
        print("   complement under accumulate, so it carries something positional that the mode")
        print("   inverts. Located, not authorable, and the inherited bit count is unchanged.")
        return 0
    if "--shapes" in sys.argv:
        coll = shape_collisions()
        print("SHAPES WHOSE TENSOR CODE IS BYTE-IDENTICAL\n")
        if not coll:
            print("   none")
        for _k, shapes in sorted(coll.items()):
            varies = [i for i in range(3) if len({s[i] for s in shapes}) > 1]
            print("   %d shapes, differing only in %s: %s"
                  % (len(shapes), "MNK"[varies[0]] if len(varies) == 1 else "MNK", shapes))
        print("\n   The sources differ and the whole programs differ; the tensor units do not.")
        print("   So that dimension is carried by the scalar code around the tile, not by the")
        print("   tile - which is why no field for it was ever found in these units.")
        return 0

    if "--check" in sys.argv:
        ok, bad, refused = check_compose()
        sys.exit(0 if bad == 0 and ok else 1)
    reg = build()
    print("=== TENSOR SEQUENCE REGISTRY ===")
    tot_a = tot_t = 0
    for k in sorted(reg):
        s = reg[k]; tot_a += s.authored_bits; tot_t += s.total_bits
        print(" ", s)
    print("\n%d shapes; %d of %d tensor bits authored (%.1f%%)"
          % (len(reg), tot_a, tot_t, 100.0 * tot_a / max(tot_t, 1)))
    print("\nINHERITED per shape: mac count and order, byte1 accumulator one-hot, byte2, byte3's\n"
          "remaining bits, byte5, byte9 - and with them M, K, transpose and mode. A shape absent\n"
          "from this table cannot be emitted at all (isa/tensor-isa.toml: composing a novel\n"
          "instruction sequence is unsolved).")

if __name__ == "__main__":
    main()


# --- PROPOSED EMITTER ARM (linker/resource session, docs/archive/g17-tensor-lowering-handoff.md) --------------------------
# The registry above keeps reference sequences for shapes the general lowering cannot author (programs shorter
# than the TENSOR metadata class's measured tail, int8 partial K). For everything else a GEMM is EMITTED from its
# shape by the lowering under results/g17-tensorops-recon-v1 (to be copied under agxforge/g17 by the compiler owner,
# with the manifest hashes as provenance). This arm is the call; it changes nothing above it.
def emit_gemm(M, N, K, lda=None, ldb=None, ldc=None, a_type="half", b_type="half", accumulate=False,
              transA=False, transB=False, simdgroups=1, registers=None, reserved=(), binds=(0, 1, 2),
              offsets=(0, 0, 0), end=True, keep=False, store=True, a_regs=None, epilogue=(), grid=1, split_fp32=False,
              a_convert=None, saturate=False, kloop=False, b_regs=None, b_convert=None, reduce=None, c_regs=None, grid_n=1, split_k=1,
              b_index=None, index_init=(), kloop_unroll=1, head_index=None, head_slices=None, c_inplace=False,
              fold_offsets=False, hoist=None, a_keep=False, kloop_chunk=None):
    """Body bytes, plan and authored image for C = A . B (+ C) on row-major device buffers, or a ValueError naming
    why the request is refused. See tensorlower.lower_gemm for the contract; the image binds A, B, C at 1, 2, 3."""
    # CORRECTED FROM THE PROPOSED PATCH, which its own comment marks provisional ("until the
    # module is copied under agxforge/g17"). As written it inserted ROOT/results/g17-tensorops-recon-v1
    # on sys.path and imported `tensorlower` by bare name: that directory is gitignored and absent
    # from any checkout but the authoring one, so the import raised here, and the insert mutated
    # sys.path on every call. The modules are under agxforge/g17 now - the entry point as
    # `tensorgemm`, because `tensorlower` is a DIFFERENT live module that cc.py imports - so this
    # is an ordinary package import.
    from agxforge.g17 import tensorgemm
    kwargs = dict(lda=lda, ldb=ldb, ldc=ldc, a_type=a_type, b_type=b_type,
                  accumulate=accumulate, transA=transA, transB=transB,
                  simdgroups=simdgroups, reserved=reserved, binds=binds,
                  offsets=offsets, end=end, keep=keep, store=store, a_regs=a_regs, epilogue=epilogue, grid=grid, split_fp32=split_fp32,
                  a_convert=a_convert, saturate=saturate, kloop=kloop, b_regs=b_regs, b_convert=b_convert, reduce=reduce, grid_n=grid_n, split_k=split_k,
                  kloop_unroll=kloop_unroll)
    if kloop_chunk is not None:
        kwargs['kloop_chunk'] = kloop_chunk
    if registers is not None:
        kwargs['registers'] = registers
    if c_regs is not None:
        kwargs['c_regs'] = c_regs          # the accumulator feed (P2); absent, the call is unchanged
    if b_index is not None or index_init:
        kwargs['b_index'] = b_index        # a register-held B offset (P7 key blocks); absent, unchanged
        kwargs['index_init'] = tuple(index_init)
    if head_index is not None:
        kwargs['head_index'] = tuple(head_index)   # the head grid (MM 25.135); absent, unchanged
    if head_slices is not None:
        kwargs['head_slices'] = head_slices        # the KV split (MM 25.114.6); absent, unchanged
    if c_inplace:
        kwargs['c_inplace'] = True                 # the register accumulator (MM 25.144.8); absent, unchanged
    if fold_offsets:
        kwargs['fold_offsets'] = True              # large offsets folded into the index once (MM 25.144.8)
    if hoist is not None:
        kwargs['hoist'] = dict(hoist)              # the invariant prologue kept in these registers (MM 25.144.8)
    if a_keep:
        kwargs['a_keep'] = True                    # a register-fed A a later body reads again (MM 25.178)
    return tensorgemm.lower_gemm(M, N, K, **kwargs)
