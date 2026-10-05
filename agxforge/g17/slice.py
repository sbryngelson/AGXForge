#!/usr/bin/env python3
"""Forward slices through the unified SSA graph, and immediates classified by what consumes them.

Once register, memory and exec state are in one graph, an immediate stops being an opaque field
and becomes a field whose CONSUMER is typed. Asking what an immediate means globally has not
worked; asking whether the value it helps produce ends up in an address, a flag, or a branch is
a question the graph can answer.

Roles are assigned structurally, from Apple's metadata alone, and none of them is a semantic
claim about the operation:

    sr_read    uses a SIR32 operand              compare  defines a FLAGR
    exec       scheduling class 25               branch   scheduling class 6
    load/store the memory family, split by NumDefs
    tensor     scheduling classes 172 and 173
    alu        scheduling classes 5, 7 and 312 - confirmed against source-level ground truth,
               where add/sub/and/or/xor land in 5, shl/shr in 7 and mul in 312

BACKWARD slicing is the more useful direction. Given an endpoint - a store, a tensor op, an EXEC
update - it walks uses back to producers and prints the expression that computes it, with every
operation this project cannot name left as an explicit typed hole:

    store(M:0x8a3..., UNKNOWN_10830(UNKNOWN_10295(SR_TP_IN_GRID_X, 16777248), 32))

That is a far better reverse-engineering target than "what does op10295 mean". The slice fixes
the input domain, the output use, the register width, the resource and the neighbours all at
once, so a differential probe can vary one of them.

    python3 tools/g17slice.py <object> --from SR_TP_IN_GRID_X   forward slice from a value
    python3 tools/g17slice.py <object> --to store               backward slices from every store
    python3 tools/g17slice.py <object> --to 0x1c4               backward slice from one offset
    python3 tools/g17slice.py <object> --immediates             immediates by consumer role
    python3 tools/g17slice.py <object> --memchain               store to load value chains
    python3 tools/g17slice.py <dir> --immfields alu store       one bucket, clustered by field

IMMEDIATES are clustered per (opcode, operand position), NEVER pooled across opcodes because
two opcodes happen to put an immediate in the same place. Two immediates at operand 3 of two
different opcodes are two different fields, and merging their value distributions would
manufacture a pattern out of nothing.
"""
import collections, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
from agxforge.g17 import cfg as g17cfg, model as g17model, ssa as g17ssa

ALU_SCHED = frozenset((5, 7, 312))
TENSOR_SCHED = frozenset((172, 173))

# Opcodes whose operation is known from SOURCE-LEVEL ground truth: one arithmetic op per kernel,
# runtime-loaded operands, result stored, so the opcode Apple selected is the operation the
# source asked for. This names Apple's opcodes; it says nothing about whether this project can
# encode them (an authored `sub` built on add's template negates instead of subtracting).
KNOWN_OPS = {
    # EXTRAPOLATED from the nearest named neighbour at distance 3 to 8 in the same scheduling
    # class with a DIFFERENT operand signature, iterated with interpolation, width and tsflags
    # for twelve rounds.
    #
    # CROSS-VALIDATED TWICE, AND THE FIRST NUMBER WAS WRONG. Measured against the table as it
    # stood, distance 3-4 came out at 98.8%. But that table already contained the 495
    # INTERPOLATED names, so derived names were serving as ground truth for a derived rule -
    # circular, and it inflated the figure. Re-measured against the PRIMARY table only, using
    # no derived name as either source or truth:
    #
    #     extrapolation 3-8   90.0%   (1188 correct, 132 wrong, over 1320 held-out cases)
    #     interpolation       95.2%   (1043 correct, 53 wrong)
    #
    # So roughly SEVENTY-SEVEN of the 770 names in this block are wrong, not fifteen. The
    # interpolated block's stated rate barely moved and stands.
    # Distance 1-2 is WORSE at 89%, which reads as an operation BOUNDARY - TableGen keeps one
    # operation's shape block together, so an immediate neighbour is as likely to be the next
    # operation as the same one, while a few slots away is inside the block. That is why the
    # rule excludes the closest neighbours rather than preferring them.
    #
    # SAME STATUS AS THE INTERPOLATED BLOCK but a worse rate: a measured ten percent,
    # shipped because a name with a stated error rate beats no name and because one dispatch
    # overturns any of them. An executed answer always wins.
    393   : "andn",
    420   : "andn",
    438   : "and.a",
    470   : "popcount.a",
    471   : "popcount.a",
    472   : "popcount.a",
    585   : "mov",
    591   : "mov",
    631   : "addr16.a",
    632   : "addr16.a",
    633   : "addr16.a",
    634   : "addr16.a",
    635   : "addr16.a",
    636   : "addr16.a",
    637   : "addr16.a",
    638   : "addr16.a",
    639   : "addr16.a",
    640   : "addr16.a",
    641   : "addr16.a",
    642   : "addr16.a",
    643   : "addr16.a",
    644   : "addr16.a",
    645   : "addr16.a",
    646   : "addr16.a",
    647   : "addr16.a",
    648   : "addr16.a",
    649   : "addr16.a",
    650   : "addr16.a",
    651   : "addr16.a",
    652   : "addr16.a",
    653   : "addr16.a",
    654   : "addr16.a",
    655   : "addr16.a",
    656   : "addr16.a",
    690   : "funnel.shr",
    692   : "funnel.shr",
    693   : "funnel.shr",
    696   : "funnel.shr",
    739   : "funnel.shr.a",
    740   : "funnel.shr.a",
    741   : "funnel.shr.a",
    742   : "funnel.shr.a",
    744   : "funnel.shr.a",
    745   : "funnel.shr.a",
    747   : "funnel.shr.a",
    748   : "funnel.shr.a",
    750   : "funnel.shr.a",
    752   : "funnel.shr.a",
    753   : "funnel.shr.a",
    754   : "funnel.shr.a",
    755   : "funnel.shr.a",
    756   : "funnel.shr.a",
    757   : "funnel.shr.a",
    759   : "funnel.shr.a",
    761   : "funnel.shr.a",
    762   : "funnel.shr.a",
    763   : "funnel.shr.a",
    764   : "funnel.shr.a",
    765   : "funnel.shr.a",
    766   : "fadd.sat.f16",
    770   : "fadd.sat.f16.a",
    778   : "fadd.f16.a",
    854   : "fmul.sat.f16",
    858   : "fmul.sat.f16.a",
    862   : "fmul.f16",
    866   : "fmul.f16.a",
    1267  : "exp2.a",
    2062  : "ffma.a",
    2065  : "ffma.a",
    2074  : "ffma.a",
    2077  : "ffma.a",
    2110  : "ffma.a",
    2113  : "ffma.a",
    2122  : "ffma.a",
    2125  : "ffma.a",
    2318  : "ffma.a",
    2320  : "ffma.imm2.a",
    2321  : "ffma.a",
    2326  : "ffma.imm0.a",
    2330  : "ffma.a",
    2333  : "ffma.a",
    2366  : "ffma.a",
    2369  : "ffma.a",
    2378  : "ffma.a",
    2381  : "ffma.a",
    2565  : "log2.a",
    3621  : "recip.a",
    3637  : "recip.a",
    3653  : "recip.a",
    3845  : "rsqrt.a",
    3941  : "trig.a",
    3973  : "rsqrt.a",
    9328  : "cvt.f2i.a",
    9741  : "csel.reg.imm",
    9744  : "csel",
    9985  : "msb",
    9991  : "msb.a",
    9992  : "msb.a",
    9993  : "msb.a",
    10235 : "addsat",
    10253 : "addsat.a",
    10257 : "addsat.a",
    10260 : "addsat.a",
    10278 : "add",
    10296 : "add.a",
    10297 : "add.a",
    10298 : "add.a",
    10299 : "add.a",
    10300 : "add.a",
    10301 : "add.a",
    10302 : "add.a",
    10303 : "add.a",
    10305 : "add.a",
    10306 : "add.a",
    10307 : "add.a",
    10309 : "add.a",
    10310 : "add.a",
    10311 : "add.a",
    10777 : "mulhi",
    10778 : "mulhi",
    10779 : "mulhi",
    10780 : "mulhi",
    10781 : "mulhi",
    10782 : "mulhi",
    10783 : "mulhi",
    10784 : "mulhi",
    10785 : "mulhi",
    10786 : "mulhi",
    10787 : "mulhi",
    10788 : "mulhi",
    10789 : "mulhi",
    10790 : "mulhi",
    10791 : "madd.wide",
    10792 : "madd.wide",
    10794 : "madhi",
    10798 : "madd.wide",
    10801 : "mulhi",
    10802 : "madd.wide",
    10803 : "madd.wide",
    10804 : "madd.wide",
    10806 : "madhi",
    10810 : "madd.wide",
    10813 : "mul",
    10814 : "mul",
    10815 : "mul",
    10816 : "mul",
    10817 : "mul",
    10818 : "mul",
    10819 : "mul",
    10820 : "mul",
    10821 : "mul",
    10867 : "mul.a",
    10868 : "madd.a",
    10869 : "madd.a",
    10870 : "madd.a",
    10871 : "madd.a",
    10872 : "madd.a",
    10873 : "madd.a",
    10874 : "madd.a",
    10875 : "madd.a",
    10876 : "madd.a",
    10877 : "madd.a",
    10878 : "madd.a",
    10879 : "madd.a",
    10880 : "madd.a",
    10881 : "madd.a",
    10882 : "madd.a",
    10883 : "madd.a",
    10884 : "madd.a",
    10885 : "madd.a",
    10886 : "madd.a",
    10887 : "madd.a",
    10888 : "madd.a",
    10889 : "madd.a",
    10890 : "madd.a",
    10891 : "madd.a",
    10892 : "madd.a",
    10893 : "madd.a",
    10894 : "madd.a",
    10895 : "madd.a",
    10896 : "madd.a",
    10897 : "madd.a",
    10898 : "madd.a",
    10899 : "madd.a",
    10900 : "madd.a",
    10901 : "madd.a",
    10902 : "madd.a",
    11050 : "mul",
    11051 : "mul",
    11052 : "mul",
    11053 : "mul",
    11054 : "mul",
    11055 : "mul",
    11056 : "mul",
    11057 : "mul",
    11058 : "mul",
    11059 : "madd",
    11060 : "madd",
    11062 : "madd",
    11063 : "madd",
    11104 : "madd.a",
    11105 : "madd.a",
    11106 : "madd.a",
    11107 : "madd.a",
    11108 : "madd.a",
    11109 : "madd.a",
    11110 : "madd.a",
    11111 : "madd.a",
    11112 : "madd.a",
    11113 : "madd.a",
    11114 : "madd.a",
    11115 : "madd.a",
    11116 : "madd.a",
    11117 : "madd.a",
    11118 : "madd.a",
    11119 : "madd.a",
    11120 : "madd.a",
    11121 : "madd.a",
    11122 : "madd.a",
    11123 : "madd.a",
    11124 : "madd.a",
    11125 : "madd.a",
    11126 : "madd.a",
    11127 : "madd.a",
    11128 : "madd.a",
    11129 : "madd.a",
    11130 : "madd.a",
    11131 : "madd.a",
    11132 : "madd.a",
    11133 : "madd.a",
    11134 : "madd.a",
    11135 : "madd.a",
    11136 : "madd.a",
    11137 : "madd.a",
    11138 : "madd.a",
    11139 : "madd.a",
    11185 : "cvt.i2f",
    11186 : "cvt.i2f",
    11187 : "cvt.i2f",
    11189 : "not",
    11195 : "not.a",
    11196 : "not.a",
    11379 : "csel.reg.imm",
    11386 : "csel.reg.imm",
    11389 : "csel",
    11406 : "csel.reg.imm",
    11409 : "csel.reg.imm",
    11620 : "subsat",
    11638 : "subsat.a",
    11642 : "subsat.a",
    11645 : "subsat.a",
    11647 : "sub.64",
    11648 : "sub.64",
    11649 : "sub.64",
    11650 : "sub.wide",
    11651 : "sub.wide",
    11653 : "sub.wide",
    11654 : "sub.wide",
    11663 : "sub",
    11681 : "sub.a",
    11682 : "sub.a",
    11685 : "sub.a",
    11686 : "sub.a",
    11687 : "sub.a",
    11688 : "sub.a",
    11689 : "sub.a",
    11690 : "sub.a",
    11691 : "sub.a",
    11694 : "sub.a",
    11695 : "sub.a",
    13429 : "orna",
    13438 : "orna",
    13447 : "orna.a",
    13454 : "nand.a",
    13456 : "nand",
    13484 : "nand",
    13514 : "unpack",
    13517 : "nor",
    13526 : "nor",
    13535 : "nor.a",
    13542 : "orn.a",
    13544 : "orn",
    13571 : "orn",
    13589 : "or.a",
    13849 : "quad.and.a",
    13851 : "quad.and.a",
    13877 : "quad.fmax.f32.a",
    13879 : "quad.fmax.f16.a",
    13885 : "quad.fmin.f32.a",
    13887 : "quad.fmin.f16.a",
    13893 : "quad.sum.a",
    13895 : "quad.sum.a",
    13901 : "quad.or.a",
    13903 : "quad.or.a",
    13909 : "quad.smax.a",
    13911 : "quad.smax.a",
    13917 : "quad.smin.a",
    13919 : "quad.smin.a",
    13925 : "quad.umax.a",
    13927 : "quad.umax.a",
    13933 : "quad.umin.a",
    13935 : "quad.umin.a",
    13941 : "quad.xor.a",
    13943 : "quad.xor.a",
    13952 : "simd.shuffle_xor1.a",
    13953 : "simd.shuffle_xor1.a",
    13954 : "simd.shuffle_xor1.a",
    13955 : "simd.shuffle_xor1.a",
    14018 : "quad.shuffle.a",
    14019 : "quad.shuffle.a",
    14020 : "quad.shuffle.a",
    14021 : "quad.shuffle_down1.a",
    14030 : "quad.shuffle_down1.a",
    14031 : "quad.shuffle_down1.a",
    14032 : "quad.shuffle_down1.a",
    14033 : "quad.shuffle_up1.a",
    14042 : "simd.identity.a",
    14043 : "quad.shuffle_up1.a",
    14044 : "simd.identity.a",
    14045 : "simd.identity.a",
    14046 : "reverse",
    14052 : "reverse.a",
    14053 : "reverse.a",
    14054 : "reverse.a",
    14079 : "flag.mov",
    14086 : "flag.mov",
    14095 : "flag.mov",
    14098 : "flag.mov",
    14106 : "flag.mov",
    14115 : "flag.mov",
    14118 : "flag.mov",
    14119 : "flag.mov.a",
    14165 : "simd.shuffle.a",
    14166 : "simd.shuffle.a",
    14167 : "simd.shuffle.a",
    14168 : "simd.shuffle_xor4.a",
    14174 : "simd.shuffle_xor4",
    14177 : "simd.shuffle_xor4.a",
    14178 : "simd.shuffle_xor1.a",
    14179 : "simd.shuffle_xor4.a",
    14180 : "simd.shuffle_xor4.a",
    14183 : "simd.reduce_bool",
    14186 : "simd.reduce_bool",
    14243 : "simd.rotate_down1_16",
    14245 : "simd.rotate_down1_16",
    14267 : "simd.rotate_up1_16",
    14269 : "simd.rotate_up1_16",
    14291 : "simd.shuffle_down1.a",
    14292 : "simd.shuffle_down1.a",
    14293 : "simd.shuffle_down1.a",
    14294 : "simd.shuffle_up1.a",
    14303 : "simd.identity.a",
    14304 : "simd.shuffle_up1.a",
    14305 : "simd.identity.a",
    14306 : "simd.identity.a",
    14361 : "shl.hi.a",
    14362 : "shl.hi.a",
    14363 : "shl.hi.a",
    14364 : "shl.hi.a",
    14366 : "shl.hi.a",
    14367 : "shl.hi.a",
    14369 : "shl.hi.a",
    14370 : "shl.hi.a",
    14372 : "shl.hi.a",
    14374 : "funnel.shl.hi.a",
    14375 : "funnel.shl.hi.a",
    14376 : "shl.a",
    14377 : "funnel.shl.a",
    14378 : "funnel.shl.a",
    14379 : "shl.a",
    14381 : "shl.a",
    14383 : "funnel.shl.a",
    14384 : "funnel.shl.a",
    14385 : "shl.a",
    14386 : "funnel.shl.a",
    14387 : "funnel.shl.a",
    14389 : "shl",
    14399 : "shl",
    14415 : "funnel.shl",
    14416 : "funnel.shl",
    14424 : "shl",
    14425 : "shl",
    14426 : "shl",
    14442 : "funnel.shl.a",
    14443 : "funnel.shl.a",
    14444 : "funnel.shl.a",
    14445 : "funnel.shl.a",
    14446 : "funnel.shl.a",
    14447 : "funnel.shl.a",
    14448 : "funnel.shl.a",
    14449 : "funnel.shl.a",
    14450 : "funnel.shl.a",
    14451 : "funnel.shl.a",
    14453 : "funnel.shl.a",
    14455 : "funnel.shl.a",
    14456 : "funnel.shl.a",
    14457 : "funnel.shl.a",
    14458 : "funnel.shl.a",
    14459 : "funnel.shl.a",
    14460 : "funnel.shl.a",
    14462 : "funnel.shl.a",
    14464 : "funnel.shl.a",
    14465 : "funnel.shl.a",
    14466 : "funnel.shl.a",
    14467 : "funnel.shl.a",
    14468 : "funnel.shl.a",
    16775 : "shr.lo",
    16793 : "shr.lo.a",
    16794 : "shr.lo.a",
    16795 : "shr.lo.a",
    16796 : "shr.lo.a",
    16798 : "shr.lo.a",
    16799 : "asr.a",
    16801 : "asr.a",
    16802 : "asr",
    16820 : "asr.a",
    16821 : "sar.a",
    16822 : "asr.a",
    16823 : "asr.a",
    16825 : "asr.a",
    16826 : "asr.a",
    16828 : "asr.a",
    16829 : "simd.prefix_and",
    16831 : "simd.prefix_and",
    16833 : "simd.prefix_and.a",
    16834 : "simd.and.a",
    16835 : "simd.prefix_and.a",
    16836 : "simd.and.a",
    16845 : "simd.prefix_sum.f32.a",
    16846 : "simd.sum.f32.a",
    16853 : "simd.prefix_product.f32.a",
    16854 : "simd.product.f32.a",
    16857 : "simd.prefix_fmax.f32",
    16859 : "simd.prefix_fmax.f16",
    16861 : "simd.prefix_fmax.f32.a",
    16862 : "simd.fmax.f32.a",
    16863 : "simd.prefix_fmax.f16.a",
    16864 : "simd.fmax.f16.a",
    16865 : "simd.prefix_fmin.f32",
    16867 : "simd.prefix_fmin.f16",
    16869 : "simd.prefix_fmin.f32.a",
    16870 : "simd.fmin.f32.a",
    16871 : "simd.prefix_fmin.f16.a",
    16872 : "simd.fmin.f16.a",
    16877 : "simd.prefix_sum.a",
    16878 : "simd.sum.a",
    16879 : "simd.prefix_sum.a",
    16880 : "simd.sum.a",
    16881 : "simd.prefix_or",
    16883 : "simd.prefix_or",
    16885 : "simd.prefix_or.a",
    16886 : "simd.or.a",
    16887 : "simd.prefix_or.a",
    16888 : "simd.or.a",
    16889 : "simd.prefix_smax",
    16891 : "simd.prefix_max",
    16893 : "simd.prefix_smax.a",
    16894 : "simd.smax.a",
    16895 : "simd.prefix_max.a",
    16896 : "simd.max.a",
    16897 : "simd.prefix_smin",
    16899 : "simd.prefix_min",
    16901 : "simd.prefix_smin.a",
    16902 : "simd.smin.a",
    16903 : "simd.prefix_min.a",
    16904 : "simd.min.a",
    16905 : "simd.prefix_umax",
    16907 : "simd.prefix_max",
    16909 : "simd.prefix_umax.a",
    16910 : "simd.umax.a",
    16911 : "simd.prefix_max.a",
    16912 : "simd.max.a",
    16913 : "simd.prefix_umin",
    16915 : "simd.prefix_min",
    16917 : "simd.prefix_umin.a",
    16918 : "simd.umin.a",
    16919 : "simd.prefix_min.a",
    16920 : "simd.min.a",
    16921 : "simd.prefix_xor",
    16923 : "simd.prefix_xor",
    16925 : "simd.prefix_xor.a",
    16926 : "simd.xor.a",
    16927 : "simd.prefix_xor.a",
    16928 : "simd.xor.a",
    16934 : "shr",
    16936 : "shr",
    16937 : "shr",
    16940 : "shr",
    16943 : "funnel.shr",
    16944 : "shr",
    16945 : "funnel.shr",
    16946 : "funnel.shr",
    16948 : "shr",
    16949 : "shr",
    16950 : "shr",
    16951 : "funnel.shr",
    16952 : "funnel.shr",
    16953 : "shr",
    16954 : "funnel.shr",
    16955 : "funnel.shr",
    16956 : "shr",
    16957 : "shr",
    16959 : "shr",
    16960 : "shr",
    16961 : "shr",
    16963 : "shr",
    16965 : "shr",
    16966 : "shr",
    16967 : "shr",
    16968 : "shr",
    16969 : "funnel.shr",
    16970 : "funnel.shr",
    16971 : "shr",
    16972 : "funnel.shr",
    16973 : "funnel.shr",
    16975 : "shr",
    16977 : "shr",
    16978 : "funnel.shr",
    16979 : "funnel.shr",
    16981 : "funnel.shr",
    16982 : "funnel.shr",
    16983 : "shr.a",
    16984 : "shr.a",
    16985 : "shr.a",
    16986 : "shr.a",
    16988 : "shr.a",
    16989 : "shr.a",
    16991 : "shr.a",
    16992 : "shr.a",
    16994 : "shr.a",
    16996 : "funnel.shr.a",
    16997 : "funnel.shr.a",
    16998 : "shr.a",
    16999 : "funnel.shr.a",
    17000 : "funnel.shr.a",
    17001 : "shr.a",
    17003 : "shr.a",
    17005 : "funnel.shr.a",
    17006 : "funnel.shr.a",
    17007 : "shr.a",
    17008 : "funnel.shr.a",
    17009 : "funnel.shr.a",
    17064 : "shr.a",
    17065 : "shr.a",
    17066 : "shr.a",
    17067 : "shr.a",
    17068 : "shr.a",
    17069 : "shr.a",
    17070 : "shr.a",
    17071 : "shr.a",
    17072 : "shr.a",
    17073 : "shr.a",
    17075 : "shr.a",
    17077 : "funnel.shr.a",
    17078 : "funnel.shr.a",
    17079 : "shr.a",
    17080 : "funnel.shr.a",
    17081 : "funnel.shr.a",
    17082 : "shr.a",
    17084 : "shr.a",
    17086 : "funnel.shr.a",
    17087 : "funnel.shr.a",
    17088 : "shr.a",
    17089 : "funnel.shr.a",
    17090 : "funnel.shr.a",
    17740 : "xnor",
    17767 : "xnor",
    17776 : "xor",
    17785 : "xor.a",
    # INTERPOLATED between two EXACTLY-agreeing named neighbours in the same scheduling class,
    # iterated with the width and tsflags rules until nothing moved. Six rounds, 495 names.
    #
    # CROSS-VALIDATED BEFORE USE, by holding out each already-named opcode and asking whether
    # its two nearest named neighbours predict it: 94.7% correct at span <= 8, 96.5% at <= 32,
    # 96.2% at <= 128 and 85.7% beyond, over 1,088 held-out cases. The span cap is 128 and
    # THIS BLOCK CARRIES A MEASURED ERROR RATE OF ROUGHLY FOUR PERCENT - about twenty of these
    # names are wrong and which twenty is not known.
    #
    # It is applied anyway because a name with a stated error rate is more useful to a
    # composer than no name, and because the encode session can overturn any of them with one
    # dispatch. An executed answer always wins over an interpolated one.
    402   : "andn",
    411   : "andn.a",
    429   : "and",
    441   : "and.a",
    448   : "branch",
    450   : "branch",
    456   : "branch",
    460   : "branch",
    467   : "popcount",
    569   : "branch",
    571   : "branch",
    580   : "branch",
    581   : "branch",
    588   : "mov",
    604   : "addr16",
    605   : "addr16",
    606   : "addr16",
    607   : "addr16",
    608   : "addr16",
    609   : "addr16",
    610   : "addr16",
    611   : "addr16",
    613   : "addr16",
    614   : "addr16",
    615   : "addr16",
    616   : "addr16",
    617   : "addr16",
    618   : "addr16",
    619   : "addr16",
    620   : "addr16",
    622   : "addr16",
    623   : "addr16",
    624   : "addr16",
    625   : "addr16",
    626   : "addr16",
    627   : "addr16",
    628   : "addr16",
    629   : "addr16",
    698   : "funnel.shr",
    699   : "funnel.shr",
    701   : "funnel.shr",
    702   : "funnel.shr",
    704   : "funnel.shr",
    705   : "funnel.shr",
    707   : "funnel.shr",
    708   : "funnel.shr",
    710   : "funnel.shr",
    711   : "funnel.shr",
    712   : "funnel.shr",
    713   : "funnel.shr",
    715   : "funnel.shr",
    716   : "funnel.shr",
    717   : "funnel.shr",
    719   : "funnel.shr",
    721   : "funnel.shr",
    722   : "funnel.shr",
    723   : "funnel.shr",
    725   : "funnel.shr",
    726   : "funnel.shr",
    728   : "funnel.shr",
    729   : "funnel.shr",
    731   : "funnel.shr",
    734   : "funnel.shr",
    735   : "funnel.shr",
    737   : "funnel.shr",
    738   : "funnel.shr",
    926   : "fadd.imm.sat.f32.to.f16",
    942   : "fadd.imm.sat.f32.a",
    1022  : "fadd.imm.f32.to.f16",
    1030  : "fadd.a",
    1033  : "fadd.a",
    1038  : "fadd.imm.a",
    1041  : "fadd.imm.f16.a",
    1042  : "fadd.a",
    1044  : "fadd.imm.f16.a",
    1276  : "exp2",
    1280  : "exp2.a",
    1283  : "exp2.a",
    2578  : "log2.a",
    2581  : "log2.a",
    3269  : "fadd.imm.f16.a",
    3272  : "fadd.imm.f16.a",
    3322  : "fmul.a",
    3325  : "fmul.a",
    3330  : "fmul.imm.a",
    3333  : "fadd.imm.f16.a",
    3334  : "fmul.a",
    3336  : "fadd.imm.f16.a",
    3662  : "recip",
    3669  : "recip.a",
    3778  : "rint.a",
    3781  : "rint.a",
    3794  : "floor.a",
    3797  : "floor.a",
    3810  : "ceil.a",
    3813  : "ceil.a",
    3826  : "trunc.a",
    3829  : "trunc.a",
    3858  : "rsqrt.a",
    3861  : "rsqrt.a",
    3957  : "trig.a",
    3989  : "rsqrt.a",
    9988  : "msb",
    10244 : "addsat",
    10245 : "addsat",
    10247 : "addsat",
    10284 : "add",
    10287 : "add",
    10290 : "add",
    10823 : "mul",
    10824 : "mul",
    10832 : "mul",
    10833 : "mul",
    10840 : "madd",
    10841 : "madd",
    10842 : "madd",
    10843 : "madd",
    10844 : "madd",
    10845 : "madd",
    10846 : "madd",
    10847 : "madd",
    10848 : "madd",
    10850 : "mul",
    10851 : "mul",
    10859 : "mul",
    10860 : "mul",
    11061 : "mul",
    11065 : "madd",
    11066 : "madd",
    11068 : "madd",
    11069 : "madd",
    11070 : "madd",
    11071 : "madd",
    11072 : "madd",
    11074 : "madd",
    11075 : "madd",
    11077 : "madd",
    11078 : "madd",
    11079 : "madd",
    11080 : "madd",
    11081 : "madd",
    11082 : "madd",
    11083 : "madd",
    11084 : "madd",
    11085 : "madd",
    11086 : "madd",
    11087 : "madd",
    11088 : "madd",
    11089 : "madd",
    11090 : "madd",
    11092 : "madd",
    11093 : "madd",
    11095 : "madd",
    11096 : "madd",
    11097 : "madd",
    11098 : "madd",
    11099 : "madd",
    11101 : "madd",
    11102 : "madd",
    11181 : "cvt.i2f",
    11184 : "cvt.i2f",
    11192 : "not",
    11371 : "csel.imm",
    11381 : "csel.reg",
    11391 : "csel.reg",
    11401 : "csel.imm",
    11411 : "csel.reg",
    11416 : "csel",
    11419 : "csel",
    11468 : "csel",
    11478 : "csel",
    11498 : "csel",
    11508 : "csel",
    11567 : "publish.coord.a",
    11579 : "publish.coord.a",
    11603 : "publish.coord.a",
    11615 : "publish.coord.a",
    11629 : "subsat",
    11630 : "subsat",
    11632 : "subsat",
    11656 : "sub.wide",
    11660 : "sub.wide",
    11672 : "sub",
    12069 : "branch",
    13465 : "nand",
    13474 : "nand.a",
    13493 : "nandn",
    13502 : "nandn.a",
    13553 : "orn",
    13562 : "orn.a",
    13569 : "orn.a",
    13580 : "or",
    13592 : "or.a",
    13595 : "or.a",
    13947 : "simd.shuffle_xor1",
    13949 : "simd.shuffle_xor1",
    13951 : "simd.shuffle_xor1",
    14011 : "quad.shuffle",
    14013 : "quad.shuffle",
    14015 : "quad.shuffle",
    14017 : "quad.shuffle",
    14025 : "quad.shuffle_down1",
    14027 : "quad.shuffle_down1",
    14029 : "quad.shuffle_down1",
    14049 : "reverse",
    14143 : "branch",
    14158 : "simd.shuffle",
    14160 : "simd.shuffle",
    14162 : "simd.shuffle",
    14164 : "simd.shuffle",
    14172 : "simd.shuffle_xor4",
    14176 : "simd.shuffle_xor4",
    14188 : "simd.reduce_bool",
    14201 : "simd.reduce_bool.a",
    14204 : "simd.reduce_bool.a",
    14205 : "simd.reduce_bool.a",
    14206 : "simd.reduce_bool.a",
    14209 : "simd.reduce_bool",
    14210 : "simd.reduce_bool",
    14227 : "simd.reduce_bool.a",
    14228 : "simd.reduce_bool.a",
    14229 : "simd.reduce_bool.a",
    14232 : "simd.reduce_bool.a",
    14286 : "simd.shuffle_down1",
    14288 : "simd.shuffle_down1",
    14290 : "simd.shuffle_down1",
    14312 : "shl.hi",
    14314 : "shl.hi",
    14315 : "shl.hi",
    14318 : "shl.hi",
    14321 : "funnel.shl.hi",
    14322 : "shl.hi",
    14323 : "funnel.shl.hi",
    14324 : "funnel.shl.hi",
    14326 : "shl.hi",
    14327 : "shl.hi",
    14328 : "shl.hi",
    14329 : "funnel.shl.hi",
    14330 : "funnel.shl.hi",
    14331 : "shl.hi",
    14332 : "funnel.shl.hi",
    14333 : "funnel.shl.hi",
    14334 : "shl.hi",
    14335 : "shl.hi",
    14338 : "shl.hi",
    14339 : "shl.hi",
    14341 : "shl.hi",
    14343 : "shl.hi",
    14344 : "shl.hi",
    14345 : "shl.hi",
    14346 : "shl.hi",
    14347 : "funnel.shl.hi",
    14348 : "funnel.shl.hi",
    14349 : "shl.hi",
    14350 : "funnel.shl.hi",
    14351 : "funnel.shl.hi",
    14353 : "shl.hi",
    14355 : "shl.hi",
    14356 : "funnel.shl.hi",
    14357 : "funnel.shl.hi",
    14359 : "funnel.shl.hi",
    14360 : "funnel.shl.hi",
    14401 : "funnel.shl",
    14402 : "funnel.shl",
    14404 : "funnel.shl",
    14405 : "funnel.shl",
    14407 : "funnel.shl",
    14408 : "funnel.shl",
    14410 : "funnel.shl",
    14411 : "funnel.shl",
    14413 : "funnel.shl",
    14414 : "funnel.shl",
    14428 : "funnel.shl",
    14429 : "funnel.shl",
    14431 : "funnel.shl",
    14432 : "funnel.shl",
    14434 : "funnel.shl",
    14437 : "funnel.shl",
    14438 : "funnel.shl",
    14440 : "funnel.shl",
    14441 : "funnel.shl",
    16773 : "branch",
    16774 : "branch",
    16780 : "shr.lo",
    16782 : "shr.lo",
    16783 : "shr.lo",
    16784 : "shr.lo",
    16785 : "shr.lo",
    16788 : "shr.lo",
    16789 : "shr.lo",
    16791 : "shr.lo",
    16811 : "asr",
    16812 : "asr",
    17021 : "shr",
    17024 : "funnel.shr",
    17025 : "shr",
    17026 : "funnel.shr",
    17027 : "funnel.shr",
    17029 : "shr",
    17030 : "shr",
    17031 : "shr",
    17032 : "funnel.shr",
    17033 : "funnel.shr",
    17034 : "shr",
    17035 : "funnel.shr",
    17036 : "funnel.shr",
    17037 : "shr",
    17038 : "shr",
    17046 : "shr",
    17047 : "shr",
    17048 : "shr",
    17049 : "shr",
    17050 : "funnel.shr",
    17051 : "funnel.shr",
    17052 : "shr",
    17053 : "funnel.shr",
    17054 : "funnel.shr",
    17056 : "shr",
    17058 : "shr",
    17059 : "funnel.shr",
    17060 : "funnel.shr",
    17062 : "funnel.shr",
    17063 : "funnel.shr",
    17749 : "xnor",
    17758 : "xnor.a",
    17765 : "xnor.a",
    17788 : "xor.a",
    # SIX CONTESTED NAMES. The width rule and the truth-table reading disagree on these, and
    # neither is discarded because the disagreement is informative.
    #
    # The truth-table code positions - b2[0], b4[0], b4[1], b5[3] - explain ten of ten anchors
    # in the four widest operand shapes and only six or seven of seven in the narrow ones,
    # which is exactly where these six live. So the code reading is weaker HERE than it is in
    # general, and the width names they contradict are themselves inherited by adjacency.
    #
    # Two weak methods disagreeing is a question for execution, not a tie to break by
    # preference. The kept name is the one already in the table; the contested reading is
    # recorded beside it.
    # THE ATOMIC FAMILY, READ RATHER THAN INFERRED. Apple's decoder PRINTS the operation as
    # 0x40200 + code on one operand, and the code table is established:
    #   0 add  1 sub  2 exchange  3 cmpxchg  4 umin  5 smin  6 umax  7 smax  8 and  9 or  10 xor
    # so every admitted member of the family names itself. ndefs gives the .noret suffix -
    # a returning atomic defines a register and a discarding one does not - and scheduling
    # class 329 is the threadgroup scope.
    9996  : "atomic.exchange.noret",
    10070 : "atomic.add.noret",
    10071 : "atomic.cmpxchg.noret",
    11702 : "atomic.tg.cmpxchg.noret",
    11705 : "atomic.tg.exchange.noret",
    11706 : "atomic.tg.cmpxchg.noret",
    11766 : "atomic.tg.cmpxchg",
    11770 : "atomic.tg.cmpxchg",
    # THE STRUCTURAL RULES ITERATED TO SATURATION - width and unanimous tsflags groups, run
    # alternately until neither adds anything. Three rounds: +78, +7, then nothing.
    #
    # PROVISIONAL, on the same grounds as the earlier iterated block: the width rule is
    # validated for ONE step from an anchor established by compilation or execution, and its
    # closure is not - it merges all ten boolean operations. Rounds 2 and 3 chain. See
    # ledger/g17-the-width-rule-does-not-chain.toml.
    403   : "andn",
    446   : "and.a",
    593   : "publish",
    985   : "fadd",
    1001  : "fadd",
    1013  : "fadd",
    1049  : "fadd",
    1061  : "fadd",
    9738  : "csel.reg.imm",
    9740  : "select",
    10796 : "madd.wide",
    10800 : "madd.wide",
    10808 : "madd.wide",
    10812 : "madd.wide",
    10827 : "madd",
    10830 : "madd",
    10836 : "madd",
    10839 : "madd",
    10854 : "madd",
    10857 : "madd",
    10863 : "madd",
    10866 : "madd",
    11064 : "madd",
    11067 : "madd",
    11073 : "madd",
    11076 : "madd",
    11091 : "madd",
    11094 : "madd",
    11100 : "madd",
    11103 : "madd",
    11382 : "csel.reg",
    11383 : "csel.reg.imm",
    11384 : "csel",
    11385 : "select",
    11405 : "select",
    11413 : "csel.reg.imm",
    11414 : "csel",
    11472 : "cmp",
    11473 : "csel",
    11476 : "csel",
    11477 : "select",
    11480 : "csel",
    11481 : "csel",
    11487 : "csel",
    11493 : "csel",
    11497 : "select",
    11500 : "csel",
    11503 : "csel",
    11506 : "csel",
    11507 : "select",
    11510 : "csel",
    11576 : "publish.coord.a",
    11612 : "publish.coord.a",
    11658 : "sub.wide",
    11662 : "sub.wide",
    11670 : "sub",
    11671 : "sub",
    11679 : "sub",
    13481 : "nand.a",
    13509 : "nandn.a",
    13561 : "orn",
    13594 : "or.a",
    14050 : "reverse",
    14403 : "funnel.shl",
    14430 : "funnel.shl",
    14439 : "funnel.shl",
    16816 : "asr",
    16819 : "asr",
    17757 : "xnor",
    17784 : "xor",   # [CONTESTED: truth table says xor]
    17790 : "xor.a",
    17792 : "xor.a",
    17793 : "xor.a",
    # UNANIMOUS (SCHEDULING CLASS, TSFLAGS) GROUPS. Where THREE OR MORE named members of one
    # group all carry the same name, its unnamed members carry it too, with ndefs required
    # to agree with a representative.
    #
    # Three is the threshold and it is not arbitrary: this key is COARSE - the whole atomic
    # family is one group of 208 - so one or two agreeing names prove nothing. Requiring
    # three, with unanimity, keeps it to groups that are genuinely one operation.
    573   : "exec",
    574   : "exec",
    576   : "exec",
    594   : "mov.a",
    595   : "mov.a",
    597   : "mov.a",
    984   : "fadd.imm.f16",
    996   : "fadd.imm.f16",
    1000  : "fadd.imm",
    1012  : "fadd.imm",
    1048  : "fadd.imm.f16",
    1060  : "fadd.imm.f16",
    1934  : "ffma",
    1937  : "ffma",
    1944  : "fadd.imm",
    1946  : "ffma",
    1949  : "ffma",
    1968  : "fadd.imm",
    1974  : "fadd.imm",
    1977  : "fadd.imm",
    1982  : "ffma",
    1985  : "ffma",
    1992  : "fadd.imm",
    1994  : "ffma",
    1997  : "ffma",
    2126  : "ffma",
    2129  : "ffma",
    2136  : "fadd.imm.f16",
    2138  : "ffma",
    2141  : "ffma",
    2160  : "fadd.imm.f16",
    2166  : "fadd.imm.f16",
    2169  : "fadd.imm.f16",
    2172  : "fadd.imm.f16",
    2174  : "ffma",
    2177  : "ffma",
    2184  : "fadd.imm.f16",
    2186  : "ffma",
    2189  : "ffma",
    2200  : "fadd.imm",
    2224  : "fadd.imm",
    2230  : "fadd.imm",
    2233  : "fadd.imm",
    2236  : "fadd.imm",
    2248  : "fadd.imm",
    2392  : "fadd.imm.f16",
    2416  : "fadd.imm.f16",
    2422  : "fadd.imm.f16",
    2425  : "fadd.imm.f16",
    2428  : "fadd.imm.f16",
    2440  : "fadd.imm.f16",
    3228  : "fadd.imm",
    3234  : "fadd.imm",
    3237  : "fadd.imm",
    3240  : "fadd.imm",
    3276  : "fadd.imm.f16",
    3282  : "fadd.imm.f16",
    3285  : "fadd.imm.f16",
    3288  : "fadd.imm.f16",
    3292  : "fadd.imm",
    3298  : "fadd.imm",
    3301  : "fadd.imm",
    3304  : "fadd.imm",
    3340  : "fadd.imm.f16",
    3346  : "fadd.imm.f16",
    3349  : "fadd.imm.f16",
    3352  : "fadd.imm.f16",
    # THE MEMORY CLASSES. A (scheduling class, tsflags) group whose every named member is one
    # memory role - all load, all store, all publish - identifies its unnamed members by
    # that role, with ndefs used as the check rather than the evidence: a load defines a
    # register and a store does not, so an opcode whose ndefs disagrees with the role is
    # left alone. This is Apple's own grouping, not an inference from behaviour.
    9994  : "atomic.idx",
    9995  : "atomic.idx",
    10018 : "atomic.idx",
    10019 : "atomic.idx",
    10020 : "atomic.idx",
    10021 : "atomic.idx",
    10066 : "atomic.idx",
    10067 : "atomic.idx",
    10068 : "atomic.idx",
    10069 : "atomic.idx",
    10091 : "atomic.idx",
    10093 : "atomic.idx",
    12310 : "load",
    12313 : "load",
    12314 : "load",
    12315 : "load",
    12316 : "load",
    12319 : "load",
    12320 : "load",
    12321 : "load",
    12322 : "load",
    12325 : "load",
    12326 : "load",
    12327 : "load",
    12328 : "load",
    12331 : "load",
    12332 : "load",
    12333 : "load",
    12334 : "load",
    12337 : "load",
    12338 : "load",
    12339 : "load",
    12340 : "load",
    12343 : "load",
    12344 : "load",
    12345 : "load",
    12346 : "load",
    12349 : "load",
    12350 : "load",
    12351 : "load",
    12352 : "load",
    12355 : "load",
    12356 : "load",
    12357 : "load",
    12358 : "load",
    12362 : "load",
    12363 : "load",
    12368 : "load",
    12369 : "load",
    12370 : "load",
    12373 : "load",
    12374 : "load",
    12375 : "load",
    12379 : "load",
    12380 : "load",
    12381 : "load",
    12382 : "load",
    12385 : "load",
    12386 : "load",
    12387 : "load",
    12388 : "load",
    12391 : "load",
    12392 : "load",
    12393 : "load",
    12394 : "load",
    12397 : "load",
    12398 : "load",
    12399 : "load",
    12400 : "load",
    12403 : "load",
    12404 : "load",
    12405 : "load",
    12647 : "load",
    12648 : "load",
    12650 : "load",
    12651 : "load",
    12653 : "load",
    12654 : "load",
    12655 : "load",
    12656 : "load",
    12657 : "load",
    12658 : "load",
    12659 : "load",
    12660 : "load",
    12661 : "load",
    12662 : "load",
    12663 : "load",
    12664 : "load",
    12665 : "load",
    12666 : "load",
    12667 : "load",
    12668 : "load",
    12669 : "load",
    12670 : "load",
    12671 : "load",
    12672 : "load",
    12679 : "load",
    12683 : "load",
    12684 : "load",
    12686 : "load",
    12687 : "load",
    12689 : "load",
    12690 : "load",
    12692 : "load",
    12693 : "load",
    12695 : "load",
    12696 : "load",
    12698 : "load",
    12699 : "load",
    12700 : "load",
    12701 : "load",
    12702 : "load",
    12703 : "load",
    12704 : "load",
    12705 : "load",
    12707 : "load",
    12708 : "load",
    12711 : "load",
    12714 : "load",
    12717 : "load",
    13234 : "store",
    13237 : "store",
    13238 : "store",
    13239 : "store",
    13240 : "store",
    13243 : "store",
    13244 : "store",
    13245 : "store",
    13246 : "store",
    13249 : "store",
    13250 : "store",
    13251 : "store",
    13252 : "store",
    13255 : "store",
    13256 : "store",
    13257 : "store",
    13258 : "store",
    13261 : "store",
    13262 : "store",
    13263 : "store",
    13264 : "store",
    13267 : "store",
    13268 : "store",
    13269 : "store",
    13270 : "store",
    13273 : "store",
    13274 : "store",
    13275 : "store",
    13276 : "store",
    13279 : "store",
    13280 : "store",
    13281 : "store",
    13282 : "store",
    13286 : "store",
    13287 : "store",
    13291 : "store",
    13292 : "store",
    13293 : "store",
    13294 : "store",
    13297 : "store",
    13298 : "store",
    13299 : "store",
    13300 : "store",
    13303 : "store",
    13304 : "store",
    13305 : "store",
    13306 : "store",
    13309 : "store",
    13310 : "store",
    13311 : "store",
    13312 : "store",
    13315 : "store",
    13316 : "store",
    13317 : "store",
    13318 : "store",
    13321 : "store",
    13322 : "store",
    13323 : "store",
    13324 : "store",
    13327 : "store",
    13328 : "store",
    13329 : "store",
    14062 : "base.wide",
    17193 : "store",
    17194 : "store",
    17195 : "store",
    17196 : "store",
    17197 : "store",
    17198 : "store",
    17200 : "store",
    17201 : "store",
    17202 : "store",
    17203 : "store",
    17204 : "store",
    17205 : "store",
    17206 : "store",
    17207 : "store",
    17208 : "store",
    17209 : "store",
    17210 : "store",
    17211 : "store",
    17212 : "store",
    17213 : "store",
    17214 : "store",
    17215 : "store",
    17216 : "store",
    17217 : "store",
    17218 : "store",
    17219 : "store",
    17221 : "store",
    17222 : "store",
    17224 : "store",
    17225 : "store",
    17227 : "store",
    17228 : "store",
    17230 : "store",
    17231 : "store",
    17233 : "store",
    17234 : "store",
    17236 : "store",
    17237 : "store",
    17239 : "store",
    17240 : "store",
    17242 : "store",
    17243 : "store",
    17245 : "store",
    17246 : "store",
    17247 : "store",
    17248 : "store",
    17249 : "store",
    17250 : "store",
    17251 : "store",
    17252 : "store",
    17254 : "store",
    17255 : "store",
    17264 : "store",
    # THE FLAG WRITERS. 379 opcodes declare a FLAGR destination and were dark because a flag
    # cannot be read back without a consumer. They do not all need one: sweeping the bits
    # that carry operand 2 shows which take values ONLY from a condition-code space - the
    # integer 8..15 or the float 0..7 - and an opcode whose second operand is a condition
    # code, writing a flag, is a comparison. The relation is the code, so the name is not.
    5073  : "fcmp",
    5075  : "fcmp",
    5079  : "fcmp",
    5081  : "fcmp",
    5086  : "fcmp",
    5089  : "fcmp",
    9625  : "fcmp",
    9626  : "fselect",
    9627  : "fselect",
    9628  : "fselect",
    9629  : "fcmp",
    9630  : "fselect",
    9631  : "fselect",
    9632  : "fselect",
    9633  : "fcmp",
    9634  : "fcmp",
    9635  : "fcmp",
    9636  : "fselect",
    9637  : "fcmp",
    9638  : "fselect",
    9639  : "fselect",
    9640  : "fselect",
    9641  : "fcmp",
    9642  : "fselect",
    9643  : "fselect",
    9644  : "fselect",
    9645  : "fcmp",
    9646  : "fcmp",
    9647  : "fcmp",
    9648  : "fselect",
    9649  : "fcmp",
    9650  : "fcmp",
    9651  : "fcmp",
    9652  : "fselect",
    9653  : "fcmp",
    9654  : "fcmp",
    9655  : "fcmp",
    9656  : "fselect",
    9657  : "fcmp",
    9658  : "fcmp",
    9659  : "fcmp",
    9660  : "fcmp",
    9661  : "fcmp",
    9662  : "fselect",
    9663  : "fselect",
    9664  : "fselect",
    9665  : "fcmp",
    9666  : "fselect",
    9667  : "fselect",
    9668  : "fselect",
    9669  : "fcmp",
    9670  : "fcmp",
    9671  : "fcmp",
    9672  : "fselect",
    9673  : "fcmp",
    9674  : "fselect",
    9675  : "fselect",
    9676  : "fselect",
    9677  : "fcmp",
    9678  : "fselect",
    9679  : "fselect",
    9680  : "fselect",
    9681  : "fcmp",
    9682  : "fcmp",
    9683  : "fcmp",
    9684  : "fselect",
    9685  : "fcmp",
    9686  : "fcmp",
    9687  : "fcmp",
    9688  : "fselect",
    9689  : "fcmp",
    9690  : "fcmp",
    9691  : "fcmp",
    9692  : "fselect",
    9693  : "fcmp",
    9694  : "fcmp",
    9695  : "fcmp",
    9696  : "fcmp",
    10366 : "cmp",
    10375 : "cmp",
    11260 : "cmp",
    11261 : "cmp",
    11262 : "cmp",
    11263 : "cmp",
    11264 : "cmp",
    11265 : "cmp",
    11266 : "cmp",
    11267 : "csel",
    11268 : "cmp",
    11269 : "cmp",
    11270 : "cmp",
    11271 : "csel",
    11272 : "cmp",
    11273 : "cmp",
    11274 : "cmp",
    11275 : "csel",
    11276 : "cmp",
    11277 : "csel",
    11278 : "csel",
    11279 : "csel",
    11280 : "cmp",
    11281 : "csel",
    11282 : "csel",
    11283 : "csel",
    11284 : "cmp",
    11285 : "cmp",
    11286 : "cmp",
    11287 : "csel",
    11288 : "cmp",
    11289 : "csel",
    11290 : "csel",
    11291 : "csel",
    11292 : "cmp",
    11293 : "csel",
    11294 : "csel",
    11295 : "csel",
    11296 : "cmp",
    11297 : "cmp",
    11298 : "cmp",
    11299 : "cmp",
    11300 : "cmp",
    11301 : "cmp",
    11302 : "cmp",
    11303 : "csel",
    11304 : "cmp",
    11305 : "cmp",
    11306 : "cmp",
    11307 : "csel",
    11308 : "cmp",
    11309 : "cmp",
    11310 : "cmp",
    11311 : "csel",
    11312 : "cmp",
    11313 : "csel",
    11314 : "csel",
    11315 : "csel",
    11316 : "cmp",
    11317 : "csel",
    11318 : "csel",
    11319 : "csel",
    11320 : "cmp",
    11321 : "cmp",
    11322 : "cmp",
    11323 : "csel",
    11324 : "cmp",
    11325 : "csel",
    11326 : "csel",
    11327 : "csel",
    11328 : "cmp",
    11329 : "csel",
    11330 : "csel",
    11331 : "csel",
    # MORE COMPARES AND FUSED COMPARE-SELECTS FROM THE CONDITION CODE. Each carries a code
    # at operand 2 drawn only from the integer space (8..15) or the float space (0..7), in a
    # scheduling class whose already-named members carry one in the same position. The
    # cmp/csel split is the register count past the destination; the relation is the code,
    # so the names carry none.
    5072  : "csel",
    5076  : "csel",
    5082  : "csel",
    9721  : "csel",
    9724  : "csel",
    9749  : "csel",
    9751  : "csel",
    9754  : "csel",
    9811  : "csel",
    9813  : "csel",
    9815  : "csel",
    9841  : "csel",
    9843  : "csel",
    9845  : "csel",
    9858  : "csel",
    9861  : "csel",
    9863  : "csel",
    9865  : "csel",
    9902  : "csel.a",
    9903  : "csel.a",
    9904  : "csel.a",
    9906  : "csel.a",
    9907  : "csel.a",
    9909  : "csel.a",
    9910  : "csel.a",
    9911  : "csel.a",
    9938  : "csel.a",
    9939  : "csel.a",
    9940  : "csel.a",
    9942  : "csel.a",
    9943  : "csel.a",
    9945  : "csel.a",
    9946  : "csel.a",
    9947  : "csel.a",
    9950  : "csel.a",
    9951  : "csel.a",
    9952  : "csel.a",
    9954  : "csel.a",
    9955  : "csel.a",
    9957  : "csel.a",
    9958  : "csel.a",
    9959  : "csel.a",
    9962  : "csel.a",
    9963  : "csel.a",
    9964  : "csel.a",
    9966  : "csel.a",
    9967  : "csel.a",
    9969  : "csel.a",
    9970  : "csel.a",
    9971  : "csel.a",
    10371 : "csel",
    10374 : "csel",
    10380 : "csel",
    10382 : "csel",
    10383 : "csel",
    11443 : "csel",
    11446 : "csel",
    11448 : "csel",
    11450 : "csel",
    11458 : "csel",
    11488 : "csel",
    11525 : "csel.a",
    11526 : "csel.a",
    11527 : "csel.a",
    11529 : "csel.a",
    11530 : "csel.a",
    11532 : "csel.a",
    11533 : "csel.a",
    11534 : "csel.a",
    11537 : "csel.a",
    11538 : "csel.a",
    11539 : "csel.a",
    11541 : "csel.a",
    11542 : "csel.a",
    11544 : "csel.a",
    11545 : "csel.a",
    11546 : "csel.a",
    11549 : "csel.a",
    11550 : "csel.a",
    11551 : "csel.a",
    11553 : "csel.a",
    11554 : "csel.a",
    11556 : "csel.a",
    11557 : "csel.a",
    11558 : "csel.a",
    11585 : "csel.a",
    11586 : "csel.a",
    11587 : "csel.a",
    11589 : "csel.a",
    11590 : "csel.a",
    11592 : "csel.a",
    11593 : "csel.a",
    11594 : "csel.a",
    # NAMED BY CODE CORRESPONDENCE. Two operand shapes in the same scheduling class share
    # their operation code bits, so a well-named shape is a key for its neighbours. This is
    # the same argument that named the quad reductions from the SIMD ones, applied within a
    # class rather than across two.
    14012 : "quad.shuffle",
    14014 : "quad.shuffle",
    14016 : "quad.shuffle",
    # THE QUAD REDUCTIONS, NAMED BY CORRESPONDENCE WITH THE SIMD ONES. Classes 446 and 448
    # use the SAME four code bits at b6[0..3], and 448 is fully named. Mapping code to code
    # gives the quad family.
    #
    # IT ALSO CORRECTS FIVE OF THE EXECUTED NAMES, and the reason is a degenerate input. The
    # 32-lane sweep seeded lane i with 3i+1, and a quad MIN over 1, 4, 7, 10 returns 1 - which
    # is lane 0's seed, so it is indistinguishable from a broadcast of lane 0. Likewise max
    # against a broadcast of lane 3. Predicting the sweep's observation from the SIMD code map
    # reproduces what it saw on 7 of 7 codes, including the float ones where the integer seeds
    # are denormal bit patterns.
    #
    # PROVISIONAL until re-run with NON-MONOTONIC seeds, which separates min from broadcast in
    # one dispatch.
    # NAMED BY CONSTRUCTION, from Apple's own compiler rather than from structure.
    # tools/g17metal.py compiles each Metal construct with the expression evaluated once, twice
    # and four times over independent operands and keeps only the opcodes whose count goes k, 2k,
    # 4k - the operation, as against the plumbing, which stays flat. An opcode is attributed only
    # if it scales in ITS probe and in no other probe of the same operand type, which subtracts
    # the shared lowering without having to model it. Each of these scales in exactly ONE of 486
    # probes.
    #
    # The half-precision block is the clean harvest: for eight transcendental and rounding
    # functions the float form was already named and the half form was not, and the half probe
    # attributes exactly one unnamed opcode.
    # THE OPCODE LATTICE. tools/g17lattice.py. The opcode space is built of BLOCKS:
    # within a maximal run whose (scheduling class, operand shape, ndefs) sequence is
    # genuinely periodic, the BLOCK is the function and the OFFSET is the operand form.
    # So one named member names the rest of its block, and the form is read off the
    # offset rather than guessed. This does NOT chain - it reads a coordinate - which is
    # the difference from the width rule that had to be withdrawn.
    #
    # The period is LOCAL: globally only 6.8%% of opcodes have the same signature sixteen
    # away, and the regions use 8, 12, 16 and 32. So blocks are detected by requiring two
    # consecutive blocks to agree at every offset, which makes the period a measurement.
    #
    # LEAVE-ONE-OUT, scored the way the rule is actually applied - hide one named opcode,
    # predict it from the others in its block, and skip blocks whose named members
    # disagree, because those are exactly the blocks the rule refuses: 130 correct, 0
    # wrong. Every earlier miss was a family whose block holds several related
    # operations - quad.or against quad.sum, quad.shuffle_up1 against shuffle_down1 -
    # and the unanimity requirement declines all of them.
    #
    # `exact` marks the strong case: another opcode in the SAME block with the SAME
    # operand shape already carries this name, so the entry copies rather than infers.
    # block 774, period 8, from fadd.f16
    776   : "fadd.imm.f16",            # offset 2 
    777   : "fadd.f16",            # offset 3 
    779   : "fadd.imm.f16.a",            # offset 5 
    780   : "fadd.imm.f16.a",            # offset 6 
    781   : "fadd.a",                # offset 7 
    # block 798, period 16, from ffma.f16
    799   : "ffma.imm2.f16",            # offset 1 
    800   : "ffma.imm0.f16",            # offset 2 
    801   : "ffma.f16",            # offset 3 
    803   : "ffma.f16",            # offset 5 
    804   : "ffma.f16",            # offset 6 
    805   : "ffma.f16",            # offset 7 
    806   : "ffma.f16.a",            # offset 8 
    807   : "ffma.f16.a",            # offset 9 
    808   : "ffma.f16.a",            # offset 10
    809   : "ffma.f16.a",            # offset 11
    810   : "ffma.f16.a",            # offset 12
    811   : "ffma.f16.a",            # offset 13
    812   : "ffma.f16.a",            # offset 14
    813   : "ffma.a",                # offset 15
    # block 854, period 8, from fadd.f16
    856   : "fmul.imm.sat.f16",            # offset 2 
    857   : "fadd.f16",            # offset 3 
    859   : "fmul.imm.sat.f16.a",            # offset 5 
    860   : "fmul.imm.sat.f16.a",            # offset 6 
    861   : "fadd.a",                # offset 7 
    # block 862, period 8, from fadd.f16
    864   : "fmul.imm.f16",            # offset 2 
    865   : "fadd.f16",            # offset 3 
    867   : "fmul.imm.f16.a",            # offset 5 
    868   : "fmul.imm.f16.a",            # offset 6 
    869   : "fadd.a",                # offset 7 
    # block 1256, period 16, from exp2
    1258  : "exp2",                # offset 2 
    1262  : "exp2.f16",            # offset 6 
    1265  : "exp2.a",                # offset 9   exact
    1266  : "exp2.a",                # offset 10
    1270  : "exp2.f16",            # offset 14
    # block 1272, period 16, from exp2, exp2.f16
    1274  : "exp2",                # offset 2 
    1278  : "exp2.f16",            # offset 6 
    1281  : "exp2.a",                # offset 9   exact
    1282  : "exp2.a",                # offset 10
    1286  : "exp2.f16",            # offset 14
    # block 2554, period 16, from log2
    2556  : "log2",                # offset 2 
    2560  : "log2.f16",            # offset 6 
    2563  : "log2.a",                # offset 9   exact
    2564  : "log2.a",                # offset 10
    2568  : "log2.f16",            # offset 14
    # block 2570, period 16, from log2, log2.f16
    2571  : "log2",                # offset 1   exact
    2572  : "log2",                # offset 2 
    2576  : "log2.f16",            # offset 6 
    2579  : "log2.a",                # offset 9   exact
    2580  : "log2.a",                # offset 10
    2584  : "log2.f16",            # offset 14
    # block 3626, period 16, from recip
    3628  : "recip",               # offset 2 
    3632  : "recip.f16",           # offset 6 
    3635  : "recip.a",               # offset 9   exact
    3636  : "recip.a",               # offset 10
    3640  : "recip.f16",           # offset 14
    # block 3642, period 16, from recip
    3644  : "recip",               # offset 2 
    3648  : "recip.f16",           # offset 6 
    3651  : "recip.a",               # offset 9   exact
    3652  : "recip.a",               # offset 10
    3656  : "recip.f16",           # offset 14
    # block 3770, period 16, from rint, rint.f16
    3771  : "rint",                # offset 1   exact
    3772  : "rint",                # offset 2 
    3774  : "rint",                # offset 4   exact
    3776  : "rint.f16",            # offset 6 
    3777  : "rint.f16",            # offset 7   exact
    3779  : "rint.a",                # offset 9   exact
    3780  : "rint.a",                # offset 10
    3783  : "rint.f16",            # offset 13  exact
    3784  : "rint.f16",            # offset 14
    # block 3786, period 16, from floor, floor.f16
    3787  : "floor",               # offset 1   exact
    3788  : "floor",               # offset 2 
    3790  : "floor",               # offset 4   exact
    3792  : "floor.f16",           # offset 6 
    3793  : "floor.f16",           # offset 7   exact
    3795  : "floor.a",               # offset 9   exact
    3796  : "floor.a",               # offset 10
    3799  : "floor.f16",           # offset 13  exact
    3800  : "floor.f16",           # offset 14
    # block 3802, period 16, from ceil, ceil.f16
    3803  : "ceil",                # offset 1   exact
    3804  : "ceil",                # offset 2 
    3806  : "ceil",                # offset 4   exact
    3808  : "ceil.f16",            # offset 6 
    3809  : "ceil.f16",            # offset 7   exact
    3811  : "ceil.a",                # offset 9   exact
    3812  : "ceil.a",                # offset 10
    3815  : "ceil.f16",            # offset 13  exact
    3816  : "ceil.f16",            # offset 14
    # block 3818, period 16, from trunc, trunc.f16
    3819  : "trunc",               # offset 1   exact
    3820  : "trunc",               # offset 2 
    3822  : "trunc",               # offset 4   exact
    3824  : "trunc.f16",           # offset 6 
    3825  : "trunc.f16",           # offset 7   exact
    3827  : "trunc.a",               # offset 9   exact
    3828  : "trunc.a",               # offset 10
    3831  : "trunc.f16",           # offset 13  exact
    3832  : "trunc.f16",           # offset 14
    # block 3834, period 16, from rsqrt
    3836  : "rsqrt",               # offset 2 
    3840  : "rsqrt.f16",           # offset 6 
    3843  : "rsqrt.a",               # offset 9   exact
    3844  : "rsqrt.a",               # offset 10
    3848  : "rsqrt.f16",           # offset 14
    # block 3850, period 16, from rsqrt, rsqrt.f16
    3852  : "rsqrt",               # offset 2 
    3856  : "rsqrt.f16",           # offset 6 
    3859  : "rsqrt.a",               # offset 9   exact
    3860  : "rsqrt.a",               # offset 10
    3864  : "rsqrt.f16",           # offset 14
    # block 3930, period 16, from trig
    3932  : "trig",                # offset 2 
    3936  : "trig.f16",            # offset 6 
    3939  : "trig.a",                # offset 9   exact
    3940  : "trig.a",                # offset 10
    3944  : "trig.f16",            # offset 14
    # block 3946, period 16, from trig
    3948  : "trig",                # offset 2 
    3952  : "trig.f16",            # offset 6 
    3955  : "trig.a",                # offset 9   exact
    3956  : "trig.a",                # offset 10
    3960  : "trig.f16",            # offset 14
    # block 3978, period 16, from rsqrt
    3979  : "rsqrt",               # offset 1   exact
    3980  : "rsqrt",               # offset 2 
    3984  : "rsqrt.f16",           # offset 6 
    3987  : "rsqrt.a",               # offset 9   exact
    3988  : "rsqrt.a",               # offset 10
    3992  : "rsqrt.f16",           # offset 14
    # block 13856, period 8, from quad.sum.f32
    13856 : "quad.prefix_sum.f32",        # offset 0   exact
    13858 : "quad.prefix_sum.f16",        # offset 2 
    13859 : "quad.sum.f16",        # offset 3 
    13860 : "quad.prefix_sum.a",            # offset 4 
    13861 : "quad.sum.a",            # offset 5 
    13862 : "quad.prefix_sum.f16.a",        # offset 6 
    13863 : "quad.sum.f16.a",        # offset 7 
    # block 13864, period 8, from quad.product.f32
    13864 : "quad.prefix_product.f32",    # offset 0   exact
    13866 : "quad.prefix_product.f16",    # offset 2 
    13867 : "quad.product.f16",    # offset 3 
    13868 : "quad.prefix_product.a",        # offset 4 
    13869 : "quad.product.a",        # offset 5 
    13870 : "quad.prefix_product.f16.a",    # offset 6 
    13871 : "quad.product.f16.a",    # offset 7 
    # block 13936, period 8, from quad.xor
    13936 : "quad.prefix_xor",            # offset 0   exact
    13938 : "quad.prefix_xor",            # offset 2   exact
    13940 : "quad.prefix_xor.a",            # offset 4   exact
    13942 : "quad.prefix_xor.a",            # offset 6   exact
    # A SECOND LATTICE PASS after widening the detector: a non-admitted opcode is a
    # WILDCARD rather than a mismatch (the decoder refuses 11,061 of 17,779 declared, so
    # requiring every offset present broke almost every run at its first hole), and a
    # trailing partial block is kept. Coverage 820 -> 956 opcodes, leave-one-out 266/266.
    # block 1288, period 16, from exp2
    1292  : "exp2.f16.a",            # offset 4 
    1293  : "exp2.f16.a",            # offset 5 
    # block 3662, period 8, from recip, recip.f16
    3664  : "recip.f16",           # offset 2 
    3667  : "recip.a",               # offset 5   exact
    3668  : "recip.a",               # offset 6 
    # block 3670, period 8, from recip
    3672  : "recip.f16",           # offset 2 
    # block 9328, period 8, from cvt.f2i
    9329  : "cvt.f2i.f16.a",         # offset 1 
    9330  : "cvt.f2i.a",             # offset 2 
    9331  : "cvt.f2i.f16.a",         # offset 3 
    # block 10909, period 16, from texture.write
    10910 : "texture.write",       # offset 1 
    10911 : "texture.write",       # offset 2 
    10912 : "texture.write",       # offset 3 
    10913 : "texture.write",       # offset 4 
    10914 : "texture.write",       # offset 5 
    10915 : "texture.write",       # offset 6 
    10916 : "texture.write",       # offset 7 
    # block 13838, period 8, from quad.and
    13842 : "quad.and",            # offset 4 
    13843 : "quad.and",            # offset 5 
    13844 : "quad.prefix_and",            # offset 6   exact
    # block 14241, period 12, from simd.rotate_down1_16
    14242 : "simd.rotate_down1_16.f16", # offset 1 
    14244 : "simd.rotate_down1_16", # offset 3 
    14246 : "simd.rotate_down1_16.f16", # offset 5 
    # block 14253, period 12, from simd.rotate_up1_16
    14260 : "simd.rotate_up1_16",  # offset 7 
    14262 : "simd.rotate_up1_16",  # offset 9 
    14264 : "simd.rotate_up1_16",  # offset 11
    # block 14265, period 12, from simd.rotate_up1_16
    14266 : "simd.rotate_up1_16.f16", # offset 1 
    14268 : "simd.rotate_up1_16",  # offset 3 
    14270 : "simd.rotate_up1_16.f16", # offset 5 
    # CORRECTED 2026-09-05 by the lattice. Thirteen opcodes were named quad.broadcast0 or
    # quad.broadcast3 and one quad.smax, and every one of them sits at offset 3 or 7 of a block
    # whose offsets 1 and 5 are a REDUCTION. Offsets 1,3,5,7 of these blocks are one function in
    # four forms - {destination present, absent} x {32-bit, 16-bit} - so a "broadcast" sharing a
    # block with quad.umin is the 16-bit form of quad.umin.
    #
    # The peer's execution had already overturned five of these on real seeds (quad.broadcast0
    # reading as QUAD min). The lattice overturns the remaining eight the same way, including
    # forms no dispatch has touched, and it also caught op13899 which was named quad.smax inside
    # the quad.or block. A structural rule and an execution agreeing on the five that overlap is
    # what licenses the eight that do not.
    #
    # `saturate` and `fsat` were likewise one operation under two spellings - they occur at
    # DIFFERENT OFFSETS OF THE SAME BLOCK, which is the definition of one function in two forms -
    # and the disagreement was refusing four blocks outright. Normalised to fsat.
    # LATTICE, round 1. Each correction unlocks more blocks: normalising a synonym or
    # fixing a wrong name makes its block unanimous, and a unanimous block names its rest.
    # block 870, period 16, from fsat
    # block 886, period 16, from fsat
    # block 950, period 16, from fsat
    # block 966, period 16, from fsat
    # block 13872, period 8, from quad.fmax.f16, quad.fmax.f32
    13872 : "quad.prefix_fmax.f32",       # offset 0   exact
    13874 : "quad.prefix_fmax.f16",       # offset 2   exact
    13876 : "quad.prefix_fmax.f32.a",       # offset 4   exact
    13878 : "quad.prefix_fmax.f16.a",       # offset 6   exact
    # block 13880, period 8, from quad.fmin.f16, quad.fmin.f32
    13880 : "quad.prefix_fmin.f32",       # offset 0   exact
    13882 : "quad.prefix_fmin.f16",       # offset 2   exact
    13884 : "quad.prefix_fmin.f32.a",       # offset 4   exact
    13886 : "quad.prefix_fmin.f16.a",       # offset 6   exact
    # block 13896, period 8, from quad.or
    13896 : "quad.prefix_or",             # offset 0   exact
    13898 : "quad.prefix_or",             # offset 2   exact
    13900 : "quad.prefix_or.a",             # offset 4   exact
    13902 : "quad.prefix_or.a",             # offset 6   exact
    # block 13904, period 8, from quad.smax
    13904 : "quad.prefix_smax",           # offset 0   exact
    13906 : "quad.prefix_smax",           # offset 2   exact
    13908 : "quad.prefix_smax.a",           # offset 4   exact
    13910 : "quad.prefix_smax.a",           # offset 6   exact
    # block 13912, period 8, from quad.smin
    13912 : "quad.prefix_smin",           # offset 0   exact
    13914 : "quad.prefix_smin",           # offset 2   exact
    13916 : "quad.prefix_smin.a",           # offset 4   exact
    13918 : "quad.prefix_smin.a",           # offset 6   exact
    # block 13920, period 8, from quad.umax
    13920 : "quad.prefix_umax",           # offset 0   exact
    13922 : "quad.prefix_umax",           # offset 2   exact
    13924 : "quad.prefix_umax.a",           # offset 4   exact
    13926 : "quad.prefix_umax.a",           # offset 6   exact
    # block 13928, period 8, from quad.umin
    13928 : "quad.prefix_umin",           # offset 0   exact
    13930 : "quad.prefix_umin",           # offset 2   exact
    13932 : "quad.prefix_umin.a",           # offset 4   exact
    13934 : "quad.prefix_umin.a",           # offset 6   exact
    # THE TWIN RULE. An unnamed opcode inside a maximal run of consecutive opcodes whose
    # named members all agree, which ALSO has two or more named opcodes of identical
    # (scheduling class, operand shape, ndefs) in that run. It copies their name.
    # Leave-one-out: 30 correct, 0 wrong - 100%% on 30 calls, which is a small sample and is
    # recorded as such. The weaker variants were measured and REJECTED rather than used:
    # run membership alone is 78.0%%, and run plus a SINGLE identical twin is 95.1%%, which
    # across 2,402 candidates would have injected well over a hundred wrong names. A rule
    # that names more than it gets right is worse than no rule for a peer building programs.
    3616  : "recip.f16",           # twins [3664, 3672]
    3624  : "recip.f16",           # twins [3664, 3672]
    3964  : "trig",                # twins [3932, 3948]
    3968  : "trig.f16",            # twins [3936, 3944]
    3976  : "rsqrt.f16",           # twins [3984, 3992]
    # COMBINED ROUND 1: the lattice (block=function, offset=form) and the form-bit rule
    # (a bit whose flips never leave a block keeps the function) applied together and
    # iterated - each new name is a new voter, so the two rules feed each other.
    3236  : "fadd.imm",               # formbit
    3268  : "fadd.imm.a",               # formbit
    3284  : "fadd.imm.f16",               # formbit
    3300  : "fadd.imm",               # formbit
    3332  : "fadd.imm.a",               # formbit
    3348  : "fadd.imm.f16",               # formbit
    # COMBINED ROUND 2: the lattice (block=function, offset=form) and the form-bit rule
    # (a bit whose flips never leave a block keeps the function) applied together and
    # iterated - each new name is a new voter, so the two rules feed each other.
    3235  : "fadd.imm",               # formbit
    3267  : "fadd.imm.f16.a",               # formbit
    # COMBINED ROUND 3: the lattice (block=function, offset=form) and the form-bit rule
    # (a bit whose flips never leave a block keeps the function) applied together and
    # iterated - each new name is a new voter, so the two rules feed each other.
    3283  : "fadd.imm.f16",               # formbit
    # COMBINED ROUND 4: the lattice (block=function, offset=form) and the form-bit rule
    # (a bit whose flips never leave a block keeps the function) applied together and
    # iterated - each new name is a new voter, so the two rules feed each other.
    # COMBINED ROUND 5: the lattice (block=function, offset=form) and the form-bit rule
    # (a bit whose flips never leave a block keeps the function) applied together and
    # iterated - each new name is a new voter, so the two rules feed each other.
    # COMBINED ROUND 6: the lattice (block=function, offset=form) and the form-bit rule
    # (a bit whose flips never leave a block keeps the function) applied together and
    # iterated - each new name is a new voter, so the two rules feed each other.
    # COMBINED ROUND 7: the lattice (block=function, offset=form) and the form-bit rule
    # (a bit whose flips never leave a block keeps the function) applied together and
    # iterated - each new name is a new voter, so the two rules feed each other.
    # WITHDRAWN 2026-09-05: 45 names the form-bit rule produced inside the CODE FAMILIES, where
    # a code is a function and a bit can change the operation while leaving the scheduling class
    # and the operand shape untouched. 44 csel and one fcmp. They were landed when the rule scored
    # 100% against a name set it had helped produce; scored against the 405 names grounded in
    # Apple's own compiler output it scores 96.2%, and every miss was one of these families. With
    # them excluded it is 60 of 60 on that independent set.
    # THE 32-BIT-BASE ADDRESSING MODE, reached only by RAY TRACING. The fourth probe family -
    # ray tracing, atomic floats, and the texture types beyond texture2d - moved what Apple's
    # compiler has ever been made to emit from 434 opcodes to 502, and an intersector call is the
    # only construct in 890 probes that emits these eight.
    #
    # They are loads and stores, and four independent facts say so rather than one:
    #
    #   their scheduling classes hold 112 named members and every one is a load or a store
    #   ndefs discriminates exactly as it does for all 112 - load defines its destination and
    #     stores define nothing, with no exception in either direction
    #   the destination and value operand classes are the same set the named loads and stores use
    #   and the ONE difference is the base register: all 112 named take a GPR32tup2 - a 64-bit
    #     address - and these take a bare GPR32
    #
    # A 32-bit base is a distinct addressing mode, and a BVH traversal is exactly where one would
    # be used: op17520 carries a GPR32tup3, three registers, which is a float3 bounding-box corner.
    # WHAT IS NOT ESTABLISHED is which address space the 32-bit base names; the suffix records the
    # base width, which is what was measured, and claims nothing further.
    11970: "load.b32",    # GPR16 destination
    11994: "load.b32",    # GPR32
    12000: "load.b32",    # GPR32tup2
    11972: "load.b32",    # GPR16, no offset operand
    17484: "store.b32",   # GPR16 value
    17520: "store.b32",   # GPR32tup3 value - a float3
    17486: "store.b32",   # GPR16 value, no offset operand
    # LEFT UNNAMED DELIBERATELY, and recorded in isa/g17-raytracing-block.txt: op11188 (its own
    # scheduling class, no destination, one per intersect), op584 (its own class, one immediate
    # operand, one per intersect), op11993 (a load class but no address operand at all), and
    # op473/op500/op527/op13808/op13820, which only a cube, cube-array, multisample or LOD-query
    # texture access emits. Each is a single attribution to a construct that plainly lowers to a
    # sequence, which is the case this project has been wrong about before.
    # THE FIFTH PROBE FAMILY: threadgroup atomics, memory orderings, barrier flag variants and
    # argument buffers. Threadgroup atomics are why it exists - the device and threadgroup forms
    # of an ordinary load are different opcodes here, so the atomics split the same way, and every
    # atomic probe before this one was on device memory.
    #
    # Scheduling class 328 turns out to be class 329 - the named atomic.tg family - with ONE extra
    # GPR16 operand, which is the same index operand that separates atomic.idx from atomic in the
    # device classes. Three of its members are probe-confirmed:
    #
    #   op11703 appears in tgatomic add, sub, and, or, xor, min, max AND int_min - all eight, one
    #           per copy. That is not eight operations sharing an opcode by accident: the atomic
    #           operation is a FOUR-BIT CODE in a field, already established, and this is the one
    #           opcode that carries it.
    #   op11767 from the exchange probe, ndefs 1 - a result is returned
    #   op11768 from the compare-exchange probe, whose last operand is a GPR32tup2, exactly as
    #           op11766 atomic.tg.cmpxchg differs from op11765 atomic.tg by that operand alone
    11703: "atomic.tg.idx.noret",
    11767: "atomic.tg.idx",
    11768: "atomic.tg.idx.cmpxchg",
    # BINDLESS TEXTURE. op14665 is op14661 texture.sample with ONE extra GPR32 operand and nothing
    # else changed - same GPR32 destination, same two GPR32tup2 operands, ndefs 1 - and it scales
    # in exactly one probe of 914: a texture sampled through an ARGUMENT BUFFER. The extra operand
    # is the descriptor, and `.idx` is the suffix this table already uses for an added index
    # operand (atomic.idx against atomic). It anchors scheduling class 417, 74 opcodes with
    # nothing named.
    #
    # The first version of that probe did a buffer load AND a texture sample and attributed
    # op14665 to the pair, which says nothing about which half emits it. Splitting the probe is
    # what made it a result.
    14665: "texture.sample.idx",
    # RETRACTED 2026-09-05: 72 opcodes were named cvt.h and NONE of them is a convert. The peer
    # executed op1000 and it returns x + imm at f32 - source and destination both 32-bit, so there
    # is nothing to convert - and the second operand it adds is an 8-BIT IMMEDIATE IN THE
    # INSTRUCTION, which is why these looked like one-source forms at all.
    #
    # Every one of the 72 sits in that same position: the one-source member of a float-add family,
    # in scheduling classes 58, 61 and 62, adjacent to two-source fadd opcodes. And my own probe
    # data had been saying so: op1048 is selected by fabs.bf AND fneg.bf AND copysign.bf, which is
    # exactly the signature that retracted op775 - a unary construct reaching a binary opcode by
    # setting a source modifier and adding zero. op1006 is selected by sin and cos, not by any
    # conversion.
    #
    # Where the destination is narrower than the source the instruction does convert, but that is
    # a consequence of the destination class, not the operation, and naming the family after it
    # made a 32-to-32 add read as a half conversion. Renamed by RESULT WIDTH, which is what a
    # compiler selects on. isa/g17-float-immediate.toml carries the immediate's format.
    # THERE IS NO BROADCAST OPCODE. This closes the thirteen names this table carried as
    # quad.broadcast0 and quad.broadcast3, which the peer's execution and my lattice had already
    # shown to be the 16-bit forms of the min, max and or reductions. The remaining question was
    # what the REAL broadcast compiles to, and it had never been probed:
    #
    #     quad_broadcast(x, 1)  ->  op14010  quad.shuffle
    #     simd_broadcast(x, 3)  ->  op14157  simd.shuffle
    #     simd_broadcast_first  ->  op14158  simd.shuffle
    #
    # A broadcast is a SHUFFLE with a constant lane index. So no opcode should ever carry the name,
    # and the thirteen were not merely mis-assigned - the name had nothing to refer to.
    # THE SIGNATURE-RUN RULE. A block enumerates the operand FORMS of one function and each
    # form appears once, so a block is a maximal run of consecutive opcode numbers with no
    # repeated signature - a definition needing no periodicity, which is what lets it reach
    # blocks the period detector cannot see (it needs two adjacent blocks of the same shape).
    # Alone it is 80.7%% against the grounded names, because a run keeps going through a
    # function boundary when the next function's forms happen not to collide. Requiring the
    # run's WIDTH to be a power of two - a block enumerates a product of per-operand choices,
    # so its size is a product of small factors - takes it to 28 correct, 0 wrong.
    3659  : "recip",               # sigrun
    3660  : "recip",               # sigrun
    3690  : "recip",               # sigrun
    3691  : "recip",               # sigrun
    3692  : "recip",               # sigrun
    3693  : "recip",               # sigrun
    3694  : "recip",               # sigrun
    3696  : "recip.f16",           # sigrun
    3697  : "recip.f16",           # sigrun
    3698  : "recip.a",               # sigrun
    3699  : "recip.f16.a",           # sigrun
    3700  : "recip.a",               # sigrun
    3701  : "recip.f16.a",           # sigrun
    9327  : "cvt.f2i.f16",         # sigrun
    # THE MATRIX UNIT SPLITS ON TWO AXES, and probing both named four more opcodes and anchored a
    # second scheduling class. Each of these scales in exactly ONE probe of 940.
    #
    #   sched 118  multiply-ACCUMULATE     sched 119  multiply, no accumulator
    #
    # and the register widths confirm the element types without reference to the probe, the same
    # way they did for op2842: an 8x8 matrix is 64 elements over 32 lanes, so two per lane, and
    # two floats are a GPR32tup2 while two halves or two bfloats are one GPR32.
    #
    #   op2842  tup2 <- tup2, tup2, tup2   f32 in, f32 accumulator
    #   op2902  tup2 <- GPR32, GPR32, tup2 bfloat in, 32-bit accumulator
    #   op2862  tup2 <- GPR32, GPR32, tup2 half in, float accumulator - the mixed form
    #   op2844  tup2 <- tup2, tup2         f32, no accumulator operand at all
    #   op2864  tup2 <- GPR32, GPR32       half in, float out, no accumulator
    2902 : "simdgroup.mma.bf16",
    2862 : "simdgroup.mma.f16.f32",
    2844 : "simdgroup.mul.f32",
    2864 : "simdgroup.mul.f16.f32",
    # CORRECTED 2026-09-05: op13891 was quad.or and it is quad.sum. Its block is op13888..13895,
    # eight consecutive opcodes whose scheduling classes alternate 445/446 across four operand
    # forms - 32-bit with a destination, 16-bit with one, and the two no-destination forms - and
    # its other three named members are op13889, op13893 and op13895, all quad.sum. The quad.or
    # block is the NEXT one, op13896..13899, and all four of its members read quad.or.
    #
    # The pairing is not an assumption: within op13840..13900 every (445,446), (449,450), (453,454)
    # and (457,458) pair that has both members named agrees - about twenty pairs, no exception -
    # and op13891 was the only opcode in the region contradicting its own block. It was also what
    # made the signature-run rule refuse the block, so one wrong name was costing four others.
    # RE-ITERATED after correcting op13891, which had been making its block disagree.
    13888 : "quad.prefix_sum",            # lattice
    13890 : "quad.prefix_sum",            # lattice
    13892 : "quad.prefix_sum.a",            # lattice
    13894 : "quad.prefix_sum.a",            # lattice
    # SCHEDULING CLASS 413 IS SELECT ON A FLAG, sixteen opcodes named by the peer's execution, one
    # dispatch each. The result is always ONE of the instruction's two value operands and never a
    # function of both; the offset in the block chooses which operand is a register and which an
    # immediate, and the opcode chooses the immediate's encoding and the flag polarity.
    # isa/g17-flag-select-block.txt has the per-opcode table.
    #
    # TWO DEGENERACIES THE PROBE HAD TO DODGE, both the shape this project keeps meeting. One flag
    # state cannot name a select - "returns its register source" and "returns its register source
    # when the flag is set" are the same observation until the flag is false - so both states run
    # against both register values in one dispatch. And an immediate of ZERO cannot say how it is
    # read, integer 0 and float 0.0 being the same bits; four of the sixteen carry zero in the
    # witness and writing 48 separates them, since 48 is 48 as an integer and 1.0 as the eight-bit
    # float immediate. All four are float. That is the immediate format from
    # isa/g17-float-immediate.toml doing work in a place neither of us was looking for it.
    #
    # `.rev` is the reversed polarity - the flag state that selects the FIRST value operand in
    # `flagsel` selects the second here - and it is a different operation, not decoration, so the
    # base names differ deliberately and the block rules will decline to merge them.
    # `.fimm` reads its immediate as the eight-bit float; `.16` has a 16-bit destination.
    14080 : "flagsel",
    14081 : "flagsel",
    14083 : "flagsel.fimm",
    14087 : "flagsel.rev",
    14088 : "flagsel.rev",
    14090 : "flagsel.fimm.rev",
    14093 : "flagsel.fimm.rev",
    14096 : "flagsel.fimm",
    14100 : "flagsel.16",
    14103 : "flagsel.16",
    14105 : "flagsel.16.fimm",
    14107 : "flagsel.16.rev",
    14110 : "flagsel.16.rev",
    14112 : "flagsel.16.fimm.rev",
    14114 : "flagsel.16.fimm.rev",
    14117 : "flagsel.16.fimm",
    # AND NOW THEY ARE NAMED, because the peer's execution read the destination back.
    #
    # These sixteen have no destination operand, take the same flag and value operands as the
    # sixteen above, and write somewhere the decoder cannot see. Two independent facts settle
    # what that somewhere is.
    #
    # THE PEER MEASURED THE WRITE. Store a marker in the special file, run op14120 holding a
    # different value, read the file back: op14120's value comes out, not the marker. The reader
    # is op14061 (`base.wide`), found by scanning Apple's own convolution kernel for mov.a and
    # printing its neighbours - eight writes then eight reads, index for index, 232 instances in
    # conv-c64k1. So op14120 writes THE FILE mov.a WRITES. Apple's own MCInstrDesc agrees from
    # the other side: op14120 and mov.a both carry MayStore at offset +8 and neither declares a
    # register destination.
    #
    # THE FORM SEQUENCE PAIRS THEM ONE TO ONE. Within class 413 the eight 32-bit named forms run
    # A B B A B B A A by operand shape (A = `? GPR32 ?`, B = `GPR32 ? ?`), and the eight 32-bit
    # destination-less forms run A B B A B B A A. The sixteen-bit halves do the same. That is the
    # lattice's offset axis: same block, same order, one axis removed. The correspondence is
    # exact and order-preserving, so each name below is its paired form plus `.a`.
    #
    # WHAT IS STILL OPEN, and it is not the name: HOW THE FILE IS INDEXED. Writing slots 68 and
    # 70 and reading them back returned the second value then nothing, which fits a latch or a
    # queue and does not fit an addressed file. The single-slot round trip is consistent with
    # both. So `.a` here means "writes mov.a's file", which is measured - not "writes element k
    # of an addressable file", which is not.
    # =========================================================================================
    # THE MIXED-WIDTH FLOAT ALU. Classes 49, 50, 52, 79, 81, 83 and 84 held 464 opcodes with
    # nothing named - the largest dark region left in the ISA - and the ten transcendental
    # classes that vacated when 794 fsat names were retracted were part of it.
    #
    # THEY ARE NOT TRANSCENDENTAL FUNCTIONS. They are ordinary float arithmetic where the
    # operand widths differ from the destination width. cos.h and log.h and sqrt.h all reach
    # them for one reason: a half-precision transcendental evaluates in f32 and rounds once at
    # the end, and THAT rounding multiply is the instruction. Naming class 49 after `cos` would
    # have repeated the fsat error exactly.
    #
    # HOW IT WAS ESTABLISHED, and the order matters. Five predictions were written to
    # isa/g17-half-alu-predictions.txt BEFORE the probes that test them were built, because the
    # 794 wrong names happened when every rule was scored against a table that already held the
    # error, and a prediction recorded first cannot be fitted to its own result. All five held:
    #
    #   P1  saturate(h * h)          -> op854   predicted from the +8 block offset
    #   P2  saturate(h + h)          -> op766
    #   P3  (half)(f32 * f32)        -> op3306  so class 49 is arithmetic, not transcendence
    #   P4  (half)(f32 * f32 + f32)  -> op2254  so class 79 is the three-source form
    #   P5  (float)(f32 * f32)       -> fmul    the control, unchanged
    #
    # Then 210 probes swept operation against width signature. The abs and neg modifiers turned
    # out NOT to change the opcode - they are operand flags - so every opcode below is selected
    # by one operation across all six of its modifier variants, which is the test that the
    # constructs share a MEANING and not merely a lowering.
    #
    # THE CLASS IS THE WIDTH SIGNATURE, the opcode is the operation:
    #   c438/c439  f16, f16   -> f16      c49  f32, f32 -> f16      c50  f32, f16 -> f16
    #   c52   f16, f32 -> f16             c57  f32, f16 -> f32
    #   c79/c81/c83/c84  the three-source forms of the same widths
    #
    # NAMING. `.to.f16` states a destination narrower than the sources; a name with no `.to.`
    # keeps the older convention that op999 set, where the suffixes are the SOURCE widths and an
    # f32 destination is the default. `.a` is the destination-less form: every one of them
    # carries MayStore in Apple's own flags word at +8, the same bit that mov.a and flagsel.a
    # carry, and the peer measured that such an instruction writes mov.a's file.
    #
    # FOUR NAMES THIS EVIDENCE DOES NOT SUPPORT, recorded so nobody adds them later. fdim, step,
    # min, max and clamp each select an opcode cleanly - and every one of those opcodes lives in
    # a class whose named members are csel, cmp, fcmp, select or fselect. They are the SELECT
    # that the lowering ends with, not the function. op9704, op9795, op9805, op9835 (fdim),
    # op9796, op9806, op9836 (step), op9832, op9881 (min/max/clamp) stay unnamed.
    768 : "fadd.imm.sat.f16",
    771 : "fadd.imm.sat.f16.a",
    772 : "fadd.imm.sat.f16.a",
    918 : "fadd.sat.f32.f32.to.f16",
    934 : "fadd.sat.f32.f32.a",
    1014 : "fadd.f32.f32.to.f16",
    3114 : "fmul.sat.f32.f32.to.f16",
    3130 : "fmul.sat.f32.f32.a",
    3306 : "fmul.f32.f32.to.f16",
    919 : "fadd.sat.f32.f16.to.f16",
    1015 : "fadd.f32.f16.to.f16",
    3115 : "fmul.sat.f32.f16.to.f16",
    3307 : "fmul.f32.f16.to.f16",
    922 : "fadd.sat.f16.f32.to.f16",
    1018 : "fadd.f16.f32.to.f16",
    3310 : "fmul.f16.f32.to.f16",
    3291 : "fmul.f32.f16",
    1486 : "ffma.sat.f32.f32.f32.to.f16",
    1550 : "ffma.sat.f32.f32.f32.a",
    2254 : "ffma.f32.f32.f32.to.f16",
    1490 : "ffma.sat.f32.f16.f32.to.f16",
    2258 : "ffma.f32.f16.f32.to.f16",
    2270 : "ffma.f16.f32.f32.to.f16",
    1503 : "ffma.sat.f16.f32.f16.to.f16",
    2271 : "ffma.f16.f32.f16.to.f16",


    # =========================================================================================
    # THE IMMEDIATE FORM OF THE MIXED-WIDTH ALU - and the end of the fsat error.
    #
    # Class 51 looked unary: one register source, one register destination. It is not. The second
    # source is an 8-BIT FLOAT IMMEDIATE, so the class is the immediate form of the same ALU, and
    # the probes say which operation each opcode is:
    #
    #     (half)saturate(f + 0.5f) -> op926      (half)saturate(f * 0.5f) -> op3122
    #     (half)(f + 0.5f)         -> op1022     (half)(f * 0.5f)         -> op3314
    #      saturate(f)             -> op904       saturate(bf)            -> op964
    #
    # `saturate(x)` reaches op904 because saturate IS this instruction with the immediate set to
    # zero. That is why 794 names went wrong: the construct that selects an opcode most obviously
    # can be a DEGENERATE CASE of what the opcode does, and a degenerate case makes a bad name.
    #
    # op1022 was named cvt.f32.f16. It is selected by (half)(f + 0.5f) and by (half)(0.5f - f):
    # an add-immediate that happens to narrow. The convert was the side effect, not the operation.
    #
    # RETRACTED HERE: the fsat names on class 51 members no probe reaches, and the `mix` names on
    # class 56. Class 51 provably holds both an add block and a multiply block, so a class-wide
    # name is falsified the same way `fadd.f16` was falsified for class 438. `mix` was worse than
    # unmeasured - it is a Metal construct, not an ISA operation, and the two members a probe does
    # reach are fused multiply-add with an immediate operand (op2326 with the immediate as the
    # multiplicand, op2320 as the addend). The other members of that block appear in the acos,
    # log and sqrt lowerings, which are chains of exactly that instruction; that is suggestive and
    # it is not measurement, so they are holes until a probe reaches them.
    #
    # WHAT WOULD FILL THEM: the width combinations the immediate sweep has not covered yet -
    # f16 source into an f32 destination, the bfloat forms, and the non-saturating variants at
    # each width. The harness is tools/g17metal.py --build-fi and it takes about four minutes.
    3314 : "fmul.imm.f32.to.f16",
    2262 : "ffma.imm0.f32.f32.to.f16",
    2256 : "ffma.imm2.f32.f32.to.f16",


    # bfloat IS A THIRD ELEMENT TYPE IN THE SAME ALU, and the immediate sweep across every width
    # combination found 38 more opcodes, each selected by ONE operation across every construct
    # that reaches it.
    #
    # THE RULE THAT SEPARATES AN IMMEDIATE FORM FROM A REGISTER FORM, and it is checkable rather
    # than assumed: the probes use three constants - 0.5, which the 8-bit float immediate
    # represents exactly; 0.1, which it does not; and 1e30, which is outside its range. A constant
    # that fits reaches an `.imm` opcode and one that does not is materialised into a register and
    # reaches the register form. It holds where it can be checked against names that were already
    # grounded: saturate(h + 0.5h) reaches op768, which was independently named fadd.imm.sat.f16,
    # and saturate(h + 0.1h) reaches op766, fadd.sat.f16.
    #
    # bfloat and half are BOTH GPR16, so the register class cannot tell them apart - the opcode
    # encodes the element type. That is why op964, reached only by saturate(bfloat), is bf16 and
    # not the f16 it was named: nothing in the operand signature could have said so.
    946 : "fadd.sat.bf16.f32.a",
    950 : "fadd.sat.f32.f32.to.bf16",
    962 : "fadd.sat.bf16.f32.to.bf16",
    3142 : "fmul.sat.bf16.f32.a",
    3146 : "fmul.sat.f32.f32.to.bf16",
    3158 : "fmul.sat.bf16.f32.to.bf16",
    963 : "fadd.sat.bf16",
    3159 : "fmul.sat.bf16",
    945 : "fadd.imm.sat.bf16.a",
    958 : "fadd.imm.sat.f32.to.bf16",
    961 : "fadd.imm.sat.bf16",
    3141 : "fmul.imm.sat.bf16.a",
    3154 : "fmul.imm.sat.f32.to.bf16",
    3157 : "fmul.imm.sat.bf16",
    938 : "fadd.sat.f16.f32.a",
    954 : "fadd.sat.f16.f32.to.bf16",
    3134 : "fmul.sat.f16.f32.a",
    3150 : "fmul.sat.f16.f32.to.bf16",
    943 : "fadd.imm.sat.f16.a",
    959 : "fadd.imm.sat.f16.to.bf16",
    3139 : "fmul.imm.sat.f16.a",
    3155 : "fmul.imm.sat.f16.to.bf16",
    2377 : "ffma.imm0.bf16.a",
    2380 : "ffma.imm2.bf16.a",
    2384 : "ffma.imm2.f32.f32.to.bf16",
    2390 : "ffma.imm0.f32.f32.to.bf16",
    2441 : "ffma.imm0.bf16",
    2444 : "ffma.imm2.bf16",
    1031 : "fadd.f32.f16.a",
    1047 : "fadd.f32.f16.to.bf16",
    1059 : "fadd.bf16",
    3351 : "fmul.bf16",
    2443 : "ffma.bf16",
    2437 : "ffma.bf16",
    2335 : "ffma.f16.f32.f16.a",
    2399 : "ffma.f16.f32.f16.to.bf16",
    2338 : "ffma.f16.f16.f32.a",
    2402 : "ffma.f16.f16.f32.to.bf16",


    # THE IMAGEBLOCK. 132 memory opcodes declare an implicit use of SR_LOCAL_X and SR_LOCAL_Y at
    # MCInstrDesc +24 - they read the thread's position in the tile without any operand saying so,
    # which means their effective address is NOT the one their printed operands describe. 128 were
    # named plain `load` or `store`, none appears in 1,795 corpus objects, and no probe had ever
    # reached one, so a compiler selecting one for an ordinary load would have addressed something
    # nobody intended.
    #
    # They are imageblock accesses. `threadgroup_imageblock` is its own address space and an
    # imageblock is indexed by the thread's position in the tile by definition. Six are measured,
    # attributed by ELEMENT WIDTH because that is what separates them:
    #
    #     load.ib.16  op12119 <- uchar, ushort      store.ib.16  op13043
    #     load.ib.32  op12151 <- uint, float        store.ib.32  op13075
    #     load.ib.64  op12143 <- half4              store.ib.64  op13067
    #
    # The other 126 take the bare name. That is not propagation from an anchor: EVERY ONE of the
    # 132 carries the implicit thread-position read in ITS OWN MCInstrDesc entry, which is a
    # per-opcode observation from Apple's table, and the six measured ones establish what that
    # property means. What is not yet measured is which of them differ by data rate, coverage or
    # sample index rather than by width.
    #
    # THE FIRST GUESS WAS WRONG AND IS WORTH KEEPING: thread-private arrays. An array too large
    # for registers is per-lane and needs a per-lane address, so it looked like the same shape.
    # It lowers to ORDINARY DEVICE MEMORY - op12073 and op17580, no implicit operand anywhere -
    # and neither of the two SP-touching opcodes appears either. This ISA has no stack-relative
    # addressing that Metal can reach.
    13830 : "load.ib",
    13831 : "load.ib",
    13832 : "load.ib",
    13833 : "load.ib",


    # The matrix sweep - 28 probes moving the accumulator, the transpose, the stride, the memory
    # space and the element type one axis at a time - reached 23 opcodes and only these two were
    # new, which is what a saturated surface looks like. c44 is the f16 counterpart of c119, the
    # multiply-without-accumulate class, and op2905 is a second bfloat multiply-accumulate form:
    # both bf16.mac probes select it while the ones that leave the accumulator uninitialised
    # select op2902, so the two differ by operand form and not by element type.
    839 : "simdgroup.mul.f16",
    2905 : "simdgroup.mma.bf16",


    # =========================================================================================
    # 492 NAMES RETRACTED HERE, and this is the measurement that forced it.
    #
    # 504 opcodes are both reached by a Metal construct and carry a name, so their names are
    # ground truth that no naming rule has ever been fitted to. Leave-one-out over exactly those:
    # predict an opcode's name from the NEAREST opcode identical in scheduling class, definition
    # count and operand register classes -
    #
    #     full name from the nearest structural twin          22.6%
    #     full name from the group majority                   19.9%
    #     only the FAMILY, the first token of the name        53.2%
    #     groups that are even family-pure                    40.7%
    #
    # Structure predicts a name about as well as a coin. And the errors are not random - they are
    # near misses. fadd.sat.f16 predicted from fadd.f16 eight opcodes away; ffma.imm0.f16 from
    # ffma.imm2.f16 one away; `and` from `andn`. The rule finds the right neighbourhood and the
    # wrong instruction, which is precisely the failure that put 794 fsat names in this table.
    #
    # WHAT WAS RETRACTED: every name on an opcode that no construct reaches, no corpus object
    # contains, and that sits in a group where two or more MEASURED members have DIFFERENT names.
    # In such a group nothing structural distinguishes opcodes whose meanings differ, so a name
    # there is a coin flip - and a coin flip is worse than a hole, because a hole stops a compiler
    # selecting the instruction while a wrong name invites it.
    #
    # NOTHING IS LOST. isa/g17-structurally-unsafe-groups.txt lists every one of these groups with
    # its measured members and their names, so each retracted opcode keeps its CANDIDATE SET. A
    # probe that reaches one member, or an execution that distinguishes two, names it for real.


    # THE FUSED FLOAT COMPARE-SELECT, and `step` was a degenerate case of it.
    #
    # These five were verified by EXECUTION before they were named: a probe containing no other
    # arithmetic instruction produced exactly the value the operation predicts, on hardware. But
    # execution alone would have named op9836 `step`, because step was the only construct that
    # reached it - and the fsat retraction says a construct that selects an opcode can be a
    # degenerate case of what it does. So the degenerate part was varied:
    #
    #   step(a, b)                      -> op9836     the construct that found it
    #   (b >= a) ? 1.0 : 0.0            -> op9836     the same thing written out
    #   (b >= a) ? 3.0 : 7.0            -> op9836     ARBITRARY constants, same opcode
    #   (b >= a) ? -1.0 : 1.0           -> op9836
    #   a == b, a != b, a < b, a <= b,
    #   a > b, a >= b, all ? 3.0 : 7.0  -> op9836     ALL SIX comparisons, same opcode
    #
    # So the comparison direction is an operand and the selected values are immediates: the
    # instruction is a fused float compare-and-select-immediate, and `step` is the case where the
    # immediates happen to be 1 and 0. Naming it `step` would have been op904 all over again.
    #
    # c279 is the same instruction selecting between two REGISTERS rather than two immediates -
    # min(a,b), max(a,b) and (a < b) ? a : b all reach op9832, so min and max are not opcodes here
    # either, they are this instruction with a condition code and two register sources.
    9796 : "fcsel.imm.f32.f32.to.f16",
    9806 : "fcsel.imm.f32.f16.to.f16",
    9836 : "fcsel.imm.f16",
    9832 : "fcsel.f16",
    9881 : "fcsel.a",


    # VERIFIED BY VALUE AND THEN NAMED, which is the order this project should have used all along.
    # Both are destination-less members of the integer ALU class, both carry MayStore at +8 like
    # mov.a and flagsel.a, and execution says what they compute: op444 ANDs its source with a small
    # immediate across every width probed, op13597 ORs two registers.
    444 : "and.imm.a",
    13597 : "or.a",


    # THE .a SUFFIX IS NOW SYSTEMATIC, and 696 names were missing it.
    #
    # An instruction with NO register destination that nevertheless declares MayStore writes
    # somewhere the operand list does not name. For arithmetic that somewhere is mov.a's file -
    # the peer measured it for op14120, execution confirmed it for op441, op444 and op13597, and
    # the whole flagsel block was named from it.
    #
    # The test is three per-opcode facts from Apple's own tables and nothing structural:
    #     ndefs == 0                       no register destination
    #     MayStore at +8                   it stores
    #     memory, texture and atomic bits clear in the word at +16, and no control-flow flag
    #                                      so it is arithmetic, not a memory or branch instruction
    #
    # It was validated before it was applied: all 55 names that already ended in .a - established
    # independently by hardware readback and by execution - satisfy it, and no memory or control
    # instruction does. 383 real stores and loads are excluded, as are barrier, branch and publish.
    #
    # WHY IT MATTERS TO A COMPILER: `madd` and `madd.a` differ in where the result goes. Selecting
    # the second where the first was meant produces no value in any register and writes the
    # address file instead, which is not a wrong number but a wrong program.


    # FORMAT CONVERSION, verified by value and reachable by nothing else. These six are the only
    # opcodes in the ISA that thirteen probe families reach through pack_float_to_* and
    # unpack_*_to_float and through no other construct, and each was checked by computing the
    # expected bit pattern in Python and comparing:
    #
    #     pack_float_to_unorm4x8(float4(1.0))  -> 0xffffffff   op9341
    #     pack_float_to_snorm4x8(float4(1.0))  -> 0x7f7f7f7f   op9341, THE SAME OPCODE
    #     unpack_unorm4x8_to_float(3).x        -> 3/255        op13512
    #     unpack_snorm4x8_to_float(3).x        -> 3/127        op13512, THE SAME OPCODE
    #     unpack_unorm10a2_to_float(3).x       -> 3/1023       op17649
    #
    # THE FORMAT IS AN OPERAND, not part of the opcode - that is what the same opcode returning
    # 0xffffffff for unorm and 0x7f7f7f7f for snorm means, and it is why these are named for the
    # FIELD LAYOUT they pack into rather than for any one of the three formats that reach them.
    # Naming op9341 `pack.unorm4x8` would have been the fsat mistake in its purest form: three
    # constructs, one instruction, and the obvious name belongs to only one of them.
    9341 : "pack.4x8",
    9352 : "pack.2x16",
    13512 : "unpack.4x8",
    13513 : "unpack.2x16",
    13515 : "unpack.4x8.to.f16",
    17649 : "unpack.10a2",


    # op10794 and op10806 were named `mulhi` and take THREE register sources. A multiply-high has
    # two. An immediate operand can only reduce the register count, never raise it, so this is the
    # one arity contradiction no encoding detail explains - they are the multiply-high-accumulate,
    # madhi, exactly as 72 three-source `shr` turned out to be funnel shifts.
    # An arity sweep over the whole table found only these two, which is the first audit this
    # session that came back nearly clean.


    # op590 was the only opcode in the table named `base`, and it is not one. It is GPR16 <- GPR16
    # in scheduling class 27, which is exactly op586 `mov` (GPR32 <- GPR32) one width down, and
    # the 24 constructs that select it are conversions and narrowing stores - cvt.float2half,
    # cvt.half2short, cvt.float2bfloat, the pack probes - where the conversion happens elsewhere
    # and this instruction moves the sixteen-bit result.
    #
    # The peer decoded it out of the imageblock kernel I sent them and read it correctly as a move
    # before I did: SR 25 goes into the low half through op590 and 0x5a5a into the high half
    # through a movimm, and that pair is the 0x5a5a0000 + lane the run reported. A `base` that
    # sets up an address does not have a destination register and this one does.


    # THREE NAMES THE PEER'S EXECUTION OVERTURNED, on two input pairs chosen so ten candidates
    # give ten different answers:
    #
    #     inputs 0x0f0f, 0x3333   inputs 0xaaaa, 0x0f0f   is        was
    #     op410    -> 0x0c0c      -> 0xa0a0               andn      andn   confirmed
    #     op437    -> 0x0303      -> 0x0a0a               and       andn
    #     op13588  -> 0x3f3f      -> 0xafaf               or        andn
    #     op17784  -> 0x3c3c      -> 0xa5a5               xor       and
    #
    # Four opcodes returning four different answers on identical inputs is itself the proof that
    # the inputs reach them, which is the check the class-51 group failed - forty walked witnesses
    # that return the same constant whatever they are given.
    #
    # ALL FOUR ARE CORPUS-ATTESTED, so these were not obscure forms: they are instructions Apple
    # ships, and three of the four carried a name from the wrong bitwise operation.


    # NAMED BY SUBSTITUTION AND EXECUTION - the first names in this table that came from running
    # the instruction rather than from watching which construct emits it.
    #
    # THE METHOD. Take an instruction Apple compiled, find the bits that select the OPCODE by
    # flipping one at a time and asking the decoder, and change only those. Every operand bit
    # stays as Apple wrote it, so the substituted opcode inherits a real encoding - which matters
    # because forty class-51 opcodes authored from repair-walk witnesses ignore their inputs
    # entirely and return a constant of the encoding.
    #
    # TWO CALIBRATIONS, because one cannot separate saturation. With a=2.25, b=3.5 a saturating
    # form returns 1 and a plain one returns the arithmetic; with a=0.125, b=0.25 both return the
    # arithmetic. An opcode is named only when both agree with the same operation:
    #
    #     op902   1        0.375     saturating add        plain add would give 5.75 on the first
    #     op3098  1        0.03125   saturating multiply   plain multiply would give 7.875
    #     op3226  7.875    0.03125   plain multiply        no saturation on either
    #     op3242  7.875    0.03125   plain multiply, f16 destination
    #
    # The runs also re-confirmed ten names already in the table - fadd, fmul, fadd.f16, fmul.f16,
    # their saturating forms, or, andn - and showed op950 and op1046 returning what looked like
    # wrong values until the destination width was accounted for: a bf16 0.375 read as a half is
    # 1.6875, which is exactly what came back.
    902 : "fadd.sat",
    3098 : "fmul.sat",
    3226 : "fmul",
    3242 : "fmul.f32.f32.to.f16",


    # THE FUSED COMPARE-SELECT WITH AN IMMEDIATE, five members measured by the peer's execution -
    # and the vindication of a refusal.
    #
    # I had left op9704, op9795, op9835 and op9701 unnamed after they were cleanly attributed to
    # fdim and step and cos and sin, on the rule that a clean single-construct attribution is not
    # enough if the CLASS disagrees: class 278's named members are csel and csel.reg.imm, so these
    # are the select a lowering ends with rather than the function. Execution says exactly that,
    # and it says more - the relation is in the opcode:
    #
    #     inputs             a=3,b=1  a=1,b=3  a=2,b=2   with the immediate at 0.5 and again at 1.0
    #     op9704 9795 9835      a       imm       a      dest = (a >= b) ? a : imm
    #     op9701               imm       a        a      dest = (a <= b) ? a : imm
    #     op11376              imm       a       imm     dest = (a <  b) ? a : imm
    #
    # Three orderings fix WHICH relation; moving the immediate from 0.5 to 1.0 and watching only
    # the imm column follow fixes WHAT the other operand is. Two calibrations doing two different
    # jobs, which is the same shape as large-and-small inputs separating saturation from operation.
    #
    # op9795 and op9835 are the mixed-width and half forms of op9704: same relation, same class,
    # sources and destination narrowed.
    #
    # STILL UNNAMED AND WORTH SAYING WHY: op9788 and op9831 have the same shapes as op9795 and
    # op9835 and each returned 1.9e-06 on one ordering. That is the signature of a value read at
    # the wrong width, not a wrong answer - the same trap as a bf16 0.375 reading as a half 1.6875.
    # bfloat and half are both GPR16, so the operand class cannot say which, and they stay holes.
    9704 : "fcsel.ge.imm",
    9795 : "fcsel.ge.imm.f32.f32.to.f16",
    9835 : "fcsel.ge.imm.f16",
    9701 : "fcsel.le.imm",
    11376 : "fcsel.lt.imm",


    # EXECUTION-VALIDATED BY THE ORACLE, authored from the specification into an image with no
    # Apple bytes at all - the route substitution cannot reach, because it needs no host slot.
    #
    # Four exact at 32 bits on the tuples (3,5) (7,7) (1,0) (255,16):
    #     op10295 add   ->  8, 14, 1, 271        op10864 mul  -> 15, 49, 0, 4080
    #     op13588 or    ->  7,  7, 1, 255        op17784 xor  ->  6,  0, 1, 239
    # Three exact once read as SIXTEEN-bit destinations, which is what their values say they are:
    #     op11680 sub   -> 65534 = -2            op13473 nand -> ~(a&b) in 16 bits
    #     op17757 xnor  -> ~(a^b) in 16 bits
    #
    # AND ONE OPERAND-SENSE CORRECTION. op410 is `andn` and it computes a & ~b, not ~a & b: on
    # (3,5) it returns 2, and 3 & ~5 is 2 where ~3 & 5 is 4. The name stands, the operand order
    # does not - anything generating an andn has to know which source is complemented.
    #
    # TWO LEFT OPEN RATHER THAN NAMED. op10860 is named `mul` and returned 26, 56, 7, 1801 where a
    # multiply gives 15, 49, 0, 4080 - not a product, and one tuple set is not enough to say what
    # it is instead. op13561 `orn` returned all-ones on every input, which is what a | ~b gives
    # when b is zero-extended from sixteen bits, so the inputs could not distinguish it.

    14120 : "flagsel.a",
    14122 : "flagsel.a",
    14124 : "flagsel.fimm.a",
    14129 : "flagsel.rev.a",
    14131 : "flagsel.rev.a",
    14133 : "flagsel.fimm.rev.a",
    14137 : "flagsel.fimm.rev.a",
    14140 : "flagsel.fimm.a",
    14121 : "flagsel.16.a",
    14125 : "flagsel.16.a",
    14127 : "flagsel.16.fimm.a",
    14130 : "flagsel.16.rev.a",
    14134 : "flagsel.16.rev.a",
    14136 : "flagsel.16.fimm.rev.a",
    14138 : "flagsel.16.fimm.rev.a",
    14141 : "flagsel.16.fimm.a",
    # THE PAIR RULE WAS WRONG AND THE PEER'S EXECUTION SAYS SO. I had read every (445,446),
    # (449,450), (453,454) and (457,458) pair in op13840..13900 as one operation in two forms,
    # from about twenty pairs whose named members agreed. They agree on the OPERATOR and not on
    # the operation: the lower scheduling class of each pair is the EXCLUSIVE PREFIX SCAN and the
    # upper is the reduction. Fourteen scans and fifteen reductions at 32 lanes, no exception.
    #
    # THE IDENTITY IN LANE 0 IS THE EVIDENCE and it names the type where arithmetic cannot:
    #
    #     and 0xffffffff        or, xor, integer sum  0
    #     f32 sum 0x80000000    f16 sum 0x8000        both -0.0
    #     f32 max/min 0x7fc00000  f16 max/min 0x7e00  both NaN
    #
    # A scan of f16 denormals and a scan of the same bits as integers agree everywhere EXCEPT lane
    # 0, where a float scan opens with -0.0; and a max scan opening with NaN rather than -inf is
    # Apple spelling "nothing seen yet", which is why the max and min scans stayed unclassified
    # until the candidate carried it.
    #
    # My correction of op13891 to quad.sum survives - it is in the UPPER class, so it is the
    # reduction - but it survived for the wrong reason, and the even members the rules then named
    # from it are scans. op13854 and op13864 the peer measured and did not resolve; they stay as
    # they were rather than being given a scan name on a rule that has just been shown to overreach.

    13846 : "quad.prefix_and",
    13848 : "quad.prefix_and.a",
    13850 : "quad.prefix_and.a",
    13852 : "quad.prefix_sum.f16",
    # THE SIMD BLOCK HAD THE SAME ERROR AS THE QUAD BLOCK, and worse: every member of scheduling
    # class 447 was named simd.prefix_sum whatever its pair computed, so op16829 read prefix_sum
    # beside op16830 simd.and and op16857 read prefix_sum beside op16858 simd.fmax.f32. The peer's
    # execution on the quad block establishes the structure - lower class is the exclusive prefix
    # scan of the upper's operator - and classes 451/452 and 455/456 in this same block were
    # already named that way, which is the internal corroboration.
    #
    # APPLIED ONLY WHERE THE UPPER MEMBER IS AN ASSOCIATIVE REDUCTION. The same sched-pair shape
    # appears in the shuffle block, where op14241's pair is simd.rotate_down1_16 - and a prefix
    # scan of a rotate is not a thing. Those are left alone: the pairing means something else
    # there, and this rule has already been shown once to overreach by exactly one step.

    # op13864 RESOLVED by the peer, and the reason it had not been is worth the line: their
    # comparison was bit-exact, and the hardware's quad product lands one ULP from a left-to-right
    # float32 multiply. Bit-exactness asks the silicon to associate its multiplies in the same
    # order the classifier does. Two ULP resolves it - while the IDENTITY lane stays compared
    # exactly, since -0.0 against 0.0 and NaN against an infinity is the one place the type is
    # visible and a tolerance there would discard the argument that named the block.
    # op13854 stays unnamed: it returns the f16 product identity in EVERY lane and no scan does.
    # SCHEDULING CLASS 63 IS SATURATE, thirty opcodes, named by the peer's execution of op1062
    # (REFUTED 2026-09-23 for op1062 itself - see the correction at its entry below):
    #
    #     modifier 0   0.25 -> 0     -0.75 -> 0.75      clamp(-x, 0, 1)
    #     modifier 2   2.5 -> 1  -1 -> 0  0.7 -> 0.7  3.9 -> 1   clamp(x, 0, 1)
    #     modifier 4   0.25 -> 0     -0.75 -> 0         clamp(-|x|, 0, 1)
    #     modifier 6   0.25 -> 0.25  -0.75 -> 0.75      clamp(|x|, 0, 1)
    #
    # What settles it against a plain max(x, 0) is 2.5 and 3.9 coming back as 1. Everything below
    # one agrees with both readings, and the probe nearly stopped at 0.25 and -0.75, which do.
    #
    # AND THE OPERAND I HAD LISTED AS A WRITABLE IMMEDIATE IS THE SOURCE MODIFIER. op1062's
    # operand 3 carries value bits 1, 2 and 5 and nothing else - negate, absolute value, keep -
    # which is the modifier word's exact shape, and the sweep reproduces all three meanings. My
    # filter passed it because the table records no lifetime operand for that source, so nothing
    # marked it. The peer's rule, adopted: AN OPERAND WHOSE ONLY MAPPED VALUE BITS LIE WITHIN
    # {1,2,4,5} IS A MODIFIER whatever else the table says, because those four are the only
    # meanings that word has ever carried. 2,395 operands on 1,760 opcodes match, and 367 of those
    # opcodes were being refused as carrying a witness-fixed immediate.
    # CORRECTED 2026-09-23: op1062 IS NOT SATURATE, it is CROSS-LANE, and every measurement above ran
    # ONE lane, where the missing neighbour reads as 0. On 2 lanes (0.7, 0.3) both lanes get 0.4; on
    # 4 lanes each pair (2k, 2k+1) gets clamp(x[2k] - x[2k+1], 0, 1) under modifier 2, which is
    # the modifier negating the source of saturate(x[odd] - x[even]) - a fine horizontal
    # derivative, saturated. The four modifier rows above are that law with the neighbour at 0.
    # Apple emits op904 for saturate(x) (ledger/g17-fsat-is-op904-and-op1062-is-cross-lane.toml).
    # The siblings below were named by analogy with op1062; that premise is refuted and their
    # names are UNVERIFIED until each is run on more than one lane. The MNEMONIC stays "fsat" (the
    # assembler keys forms by it); the executed name is EXECUTED_NAMES[1062] at the end of this file.
    1062 : "fsat",
    # RE-ITERATED after op1062 anchored the saturate class - see the correction above.
    1063  : "fsat",
    1064  : "fsat",
    1065  : "fsat",
    1066  : "fsat",
    1067  : "fsat.f16",
    1068  : "fsat.f16",
    1069  : "fsat.f16",
    1070  : "fsat.a",
    1071  : "fsat.f16.a",
    1072  : "fsat.a",
    1073  : "fsat.f16.a",
    1074  : "fsat",
    1075  : "fsat.f16",
    1076  : "fsat.f16",
    1077  : "fsat.f16",
    1078  : "fsat",
    1079  : "fsat",
    1080  : "fsat",
    1081  : "fsat",
    1082  : "fsat",
    1083  : "fsat.f16",
    1084  : "fsat.f16",
    1085  : "fsat.f16",
    1086  : "fsat.a",
    1087  : "fsat.f16.a",
    1088  : "fsat.a",
    1089  : "fsat.f16.a",
    1090  : "fsat",
    1091  : "fsat.f16",
    1092  : "fsat.f16",
    1093  : "fsat.f16",
    # MCInstrDesc CARRIES THE OPERATION AND NOBODY HAD READ IT. agx3meta dumps the word at
    # offset +16 of each MCInstrDesc and calls it `tsflags`; in LLVM's layout that offset is the
    # GENERIC Flags word, and its bits say what the instruction DOES. Differencing the word across
    # groups this project had already named identifies them without reference to any header:
    #
    #     bit 22   touches memory     293 named opcodes, 0 false positives, 0 false negatives
    #     bit 25   atomic              32 named opcodes, 0 and 0
    #     bit 8    loads              140 named, the 6 misses all texture reads, the 2 extras
    #                                 base.wide, which materialises an address
    #     bit 9    stores             139 named, the misses texture writes, the extras flag.mov
    #                                 and mov.a - which write a special file, arguably a store
    #
    # Scored as a CLASSIFIER over every named opcode it fires on - memory and load, memory and
    # store, or atomic - it is 311 correct and 0 wrong. So these names are read out of Apple's own
    # instruction table, not inferred: this is the one source of per-opcode meaning that survived
    # the name table being stripped.
    #
    # THE ADDRESS SPACE IS DELIBERATELY NOT STATED. Only 42 of 693 witnesses carry the address
    # expression whose scale separates device from threadgroup, and the scheduling class is mixed
    # on that axis for 149 of them, so a .tg suffix here would be a guess where the base name is
    # a measurement. `load` is what is established; whether it is `load.tg` is not.
    11699 : "atomic",
    11700 : "atomic",
    11704 : "atomic",
    11763 : "atomic",
    11764 : "atomic",
    11871 : "load",
    11872 : "load",
    11873 : "load",
    11874 : "load",
    11875 : "load",
    11876 : "load",
    11877 : "load",
    11878 : "load",
    11879 : "load",
    11880 : "load",
    11881 : "load",
    11882 : "load",
    11883 : "load",
    11884 : "load",
    11885 : "load",
    11886 : "load",
    11887 : "load",
    11888 : "load",
    11889 : "load",
    11890 : "load",
    11891 : "load",
    11892 : "load",
    11893 : "load",
    11894 : "load",
    11895 : "load",
    11896 : "load",
    11897 : "load",
    11898 : "load",
    11899 : "load",
    11900 : "load",
    11901 : "load",
    11902 : "load",
    11903 : "load",
    11904 : "load",
    11905 : "load",
    11906 : "load",
    11907 : "load",
    11908 : "load",
    11909 : "load",
    11910 : "load",
    11911 : "load",
    11912 : "load",
    11913 : "load",
    11914 : "load",
    11915 : "load",
    11916 : "load",
    11917 : "load",
    11918 : "load",
    11967 : "load",
    11968 : "load",
    11969 : "load",
    11971 : "load",
    11973 : "load",
    11974 : "load",
    11975 : "load",
    11976 : "load",
    11977 : "load",
    11978 : "load",
    11979 : "load",
    11980 : "load",
    11981 : "load",
    11982 : "load",
    11983 : "load",
    11984 : "load",
    11985 : "load",
    11986 : "load",
    11987 : "load",
    11988 : "load",
    11989 : "load",
    11990 : "load",
    11991 : "load",
    11992 : "load",
    11993 : "load",
    11995 : "load",
    11996 : "load",
    11997 : "load",
    11998 : "load",
    11999 : "load",
    12001 : "load",
    12002 : "load",
    12003 : "load",
    12004 : "load",
    12005 : "load",
    12006 : "load",
    12007 : "load",
    12008 : "load",
    12009 : "load",
    12010 : "load",
    12011 : "load",
    12012 : "load",
    12013 : "load",
    12014 : "load",
    12070 : "load",
    12071 : "load",
    12072 : "load",
    12073 : "load",
    12074 : "load",
    12075 : "load",
    12076 : "load",
    12077 : "load",
    12078 : "load",
    12079 : "load",
    12080 : "load",
    12081 : "load",
    12082 : "load",
    12083 : "load",
    12084 : "load",
    12085 : "load",
    12086 : "load",
    12087 : "load",
    12088 : "load",
    12089 : "load",
    12090 : "load",
    12091 : "load",
    12092 : "load",
    12093 : "load",
    12118 : "load.ib",
    12119 : "load.ib.16",
    12120 : "load.ib",
    12121 : "load.ib",
    12122 : "load.ib",
    12123 : "load.ib",
    12124 : "load.ib",
    12125 : "load.ib",
    12126 : "load.ib",
    12127 : "load.ib",
    12128 : "load.ib",
    12129 : "load.ib",
    12130 : "load.ib",
    12131 : "load.ib",
    12132 : "load.ib",
    12133 : "load.ib",
    12134 : "load.ib",
    12135 : "load.ib",
    12136 : "load.ib",
    12137 : "load.ib",
    12138 : "load.ib",
    12139 : "load.ib",
    12140 : "load.ib",
    12141 : "load.ib",
    12142 : "load.ib",
    12143 : "load.ib.64",
    12144 : "load.ib",
    12145 : "load.ib",
    12146 : "load.ib",
    12147 : "load.ib",
    12148 : "load.ib",
    12149 : "load.ib",
    12150 : "load.ib",
    12151 : "load.ib.32",
    12152 : "load.ib",
    12153 : "load.ib",
    12154 : "load.ib",
    12155 : "load.ib",
    12156 : "load.ib",
    12157 : "load.ib",
    12158 : "load.ib",
    12159 : "load.ib",
    12160 : "load.ib",
    12161 : "load.ib",
    12162 : "load.ib",
    12163 : "load.ib",
    12164 : "load.ib",
    12165 : "load.ib",
    12166 : "load.ib",
    12167 : "load.ib",
    12168 : "load.ib",
    12169 : "load.ib",
    12170 : "load.ib",
    12171 : "load.ib",
    12172 : "load.ib",
    12173 : "load.ib",
    12174 : "load.ib",
    12175 : "load.ib",
    12176 : "load.ib",
    12177 : "load.ib",
    12178 : "load.ib",
    12179 : "load.ib",
    12180 : "load.ib",
    12181 : "load.ib",
    12311 : "load",
    12312 : "load",
    12317 : "load",
    12318 : "load",
    12323 : "load",
    12324 : "load",
    12329 : "load",
    12330 : "load",
    12335 : "load",
    12336 : "load",
    12341 : "load",
    12342 : "load",
    12347 : "load",
    12348 : "load",
    12353 : "load",
    12354 : "load",
    12359 : "load",
    12360 : "load",
    12365 : "load",
    12366 : "load",
    12371 : "load",
    12372 : "load",
    12377 : "load",
    12378 : "load",
    12383 : "load",
    12384 : "load",
    12389 : "load",
    12390 : "load",
    12395 : "load",
    12396 : "load",
    12401 : "load",
    12402 : "load",
    13042 : "store.ib",
    13043 : "store.ib.16",
    13044 : "store.ib",
    13045 : "store.ib",
    13046 : "store.ib",
    13047 : "store.ib",
    13048 : "store.ib",
    13049 : "store.ib",
    13050 : "store.ib",
    13051 : "store.ib",
    13052 : "store.ib",
    13053 : "store.ib",
    13054 : "store.ib",
    13055 : "store.ib",
    13056 : "store.ib",
    13057 : "store.ib",
    13058 : "store.ib",
    13059 : "store.ib",
    13060 : "store.ib",
    13061 : "store.ib",
    13062 : "store.ib",
    13063 : "store.ib",
    13064 : "store.ib",
    13065 : "store.ib",
    13066 : "store.ib",
    13067 : "store.ib.64",
    13068 : "store.ib",
    13069 : "store.ib",
    13070 : "store.ib",
    13071 : "store.ib",
    13072 : "store.ib",
    13073 : "store.ib",
    13074 : "store.ib",
    13075 : "store.ib.32",
    13076 : "store.ib",
    13077 : "store.ib",
    13078 : "store.ib",
    13079 : "store.ib",
    13080 : "store.ib",
    13081 : "store.ib",
    13082 : "store.ib",
    13083 : "store.ib",
    13084 : "store.ib",
    13085 : "store.ib",
    13086 : "store.ib",
    13087 : "store.ib",
    13088 : "store.ib",
    13089 : "store.ib",
    13090 : "store.ib",
    13091 : "store.ib",
    13092 : "store.ib",
    13093 : "store.ib",
    13094 : "store.ib",
    13095 : "store.ib",
    13096 : "store.ib",
    13097 : "store.ib",
    13098 : "store.ib",
    13099 : "store.ib",
    13100 : "store.ib",
    13101 : "store.ib",
    13102 : "store.ib",
    13103 : "store.ib",
    13104 : "store.ib",
    13105 : "store.ib",
    13235 : "store",
    13236 : "store",
    13241 : "store",
    13242 : "store",
    13247 : "store",
    13248 : "store",
    13253 : "store",
    13254 : "store",
    13259 : "store",
    13260 : "store",
    13265 : "store",
    13266 : "store",
    13271 : "store",
    13272 : "store",
    13277 : "store",
    13278 : "store",
    13283 : "store",
    13284 : "store",
    13289 : "store",
    13290 : "store",
    13295 : "store",
    13296 : "store",
    13301 : "store",
    13302 : "store",
    13307 : "store",
    13308 : "store",
    13313 : "store",
    13314 : "store",
    13319 : "store",
    13320 : "store",
    13325 : "store",
    13326 : "store",
    17481 : "store",
    17482 : "store",
    17483 : "store",
    17485 : "store",
    17487 : "store",
    17488 : "store",
    17489 : "store",
    17490 : "store",
    17491 : "store",
    17492 : "store",
    17493 : "store",
    17494 : "store",
    17495 : "store",
    17496 : "store",
    17497 : "store",
    17498 : "store",
    17499 : "store",
    17500 : "store",
    17501 : "store",
    17502 : "store",
    17503 : "store",
    17504 : "store",
    17505 : "store",
    17506 : "store",
    17507 : "store",
    17508 : "store",
    17509 : "store",
    17510 : "store",
    17511 : "store",
    17512 : "store",
    17513 : "store",
    17514 : "store",
    17515 : "store",
    17516 : "store",
    17517 : "store",
    17518 : "store",
    17519 : "store",
    17521 : "store",
    17522 : "store",
    17523 : "store",
    17524 : "store",
    17525 : "store",
    17526 : "store",
    17527 : "store",
    17528 : "store",
    17577 : "store",
    17578 : "store",
    17579 : "store",
    17580 : "store",
    17581 : "store",
    17582 : "store",
    17583 : "store",
    17584 : "store",
    17585 : "store",
    17586 : "store",
    17587 : "store",
    17588 : "store",
    17589 : "store",
    17590 : "store",
    17591 : "store",
    17592 : "store",
    17593 : "store",
    17594 : "store",
    17595 : "store",
    17596 : "store",
    17597 : "store",
    17598 : "store",
    17599 : "store",
    17600 : "store",
    # THE FLAGS WORD ALSO CARRIES THE TEXTURE AND SATURATE FAMILIES, and three of its bits
    # partition them exactly:
    #
    #     bit 27   texture WRITE          8 named, all texture.write, and bit 1 clear on all
    #     bit 1    texture sample/gather/read   6 named, all of those and nothing else
    #     bit 40   subset of bit 1, READ  2 named, both texture.read; every named sample and
    #              gather has it clear, and bit 40 never occurs outside bit 1
    #     bit 7    the saturate family   32 named, 16 fsat and 16 fsat.f16, split by operand width
    #
    # bit 1 covers scheduling classes 417, 418, 419 and 420 - which were the four largest classes
    # in this project's dark list, 110, 74, 74 and 16 opcodes with nothing named - so Apple's own
    # instruction table says the biggest unnamed region of the ISA is the texture unit. It also
    # agrees with the two names reached independently: op14665 texture.sample.idx, named by a
    # bindless-texture probe, is in class 417; and op1062, named fsat by the peer's execution, is
    # in class 63, which is where bit 7 lives.
    #
    # WHERE THE BIT DOES NOT SEPARATE, THE NAME DOES NOT EITHER. bit 1 without bit 40 holds both
    # samples and gathers - four named, three sample-flavoured and one gather - so those are named
    # `texture` at the family level rather than guessed at 75%.
    1094  : "fsat",
    1095  : "fsat",
    1096  : "fsat",
    1097  : "fsat",
    1098  : "fsat",
    1099  : "fsat.f16",
    1100  : "fsat.f16",
    1101  : "fsat.f16",
    1102  : "fsat.a",
    1103  : "fsat.f16.a",
    1104  : "fsat.a",
    1105  : "fsat.f16.a",
    1106  : "fsat",
    1107  : "fsat.f16",
    1108  : "fsat.f16",
    1109  : "fsat.f16",
    1110  : "fsat",
    1111  : "fsat",
    1112  : "fsat",
    1113  : "fsat",
    1114  : "fsat",
    1115  : "fsat.f16",
    1116  : "fsat.f16",
    1117  : "fsat.f16",
    1118  : "fsat.a",
    1119  : "fsat.f16.a",
    1120  : "fsat.a",
    1121  : "fsat.f16.a",
    1122  : "fsat",
    1123  : "fsat.f16",
    1124  : "fsat.f16",
    1125  : "fsat.f16",
    14471 : "texture",
    14473 : "texture",
    14475 : "texture",
    14477 : "texture",
    14479 : "texture",
    14481 : "texture",
    14483 : "texture",
    14484 : "texture",
    14485 : "texture",
    14487 : "texture",
    14489 : "texture",
    14491 : "texture",
    14493 : "texture",
    14495 : "texture",
    14496 : "texture",
    14497 : "texture",
    14499 : "texture",
    14502 : "texture",
    14503 : "texture",
    14504 : "texture",
    14505 : "texture",
    14507 : "texture",
    14509 : "texture",
    14511 : "texture",
    14512 : "texture",
    14513 : "texture",
    14515 : "texture",
    14517 : "texture",
    14519 : "texture",
    14520 : "texture",
    14521 : "texture",
    14523 : "texture",
    14524 : "texture",
    14525 : "texture",
    14527 : "texture",
    14529 : "texture",
    14531 : "texture",
    14533 : "texture",
    14534 : "texture",
    14535 : "texture",
    14537 : "texture",
    14539 : "texture",
    14540 : "texture",
    14541 : "texture",
    14543 : "texture",
    14545 : "texture",
    14547 : "texture",
    14549 : "texture",
    14550 : "texture",
    14551 : "texture",
    14553 : "texture",
    14555 : "texture",
    14557 : "texture",
    14559 : "texture",
    14561 : "texture",
    14563 : "texture",
    14565 : "texture",
    14566 : "texture",
    14567 : "texture",
    14568 : "texture",
    14569 : "texture",
    14570 : "texture",
    14571 : "texture",
    14572 : "texture",
    14573 : "texture",
    14574 : "texture",
    14575 : "texture",
    14577 : "texture",
    14579 : "texture",
    14580 : "texture",
    14581 : "texture",
    14583 : "texture",
    14584 : "texture",
    14585 : "texture",
    14586 : "texture",
    14587 : "texture",
    14589 : "texture",
    14591 : "texture",
    14593 : "texture",
    14595 : "texture",
    14663 : "texture",
    14667 : "texture",
    14669 : "texture",
    14670 : "texture",
    14671 : "texture",
    14673 : "texture",
    14674 : "texture",
    14675 : "texture",
    14677 : "texture",
    14679 : "texture",
    14681 : "texture",
    14683 : "texture",
    14685 : "texture",
    14687 : "texture",
    14689 : "texture",
    14691 : "texture",
    14693 : "texture",
    14695 : "texture",
    14697 : "texture",
    14699 : "texture",
    14701 : "texture",
    14703 : "texture",
    14704 : "texture",
    14705 : "texture",
    14707 : "texture",
    14709 : "texture",
    14711 : "texture",
    14713 : "texture",
    14714 : "texture",
    14715 : "texture",
    14717 : "texture",
    14719 : "texture",
    14721 : "texture",
    14723 : "texture",
    14725 : "texture",
    14727 : "texture",
    14729 : "texture",
    14730 : "texture",
    14731 : "texture",
    14733 : "texture",
    14735 : "texture",
    14737 : "texture",
    14739 : "texture",
    14741 : "texture",
    14743 : "texture",
    14745 : "texture",
    14747 : "texture",
    14748 : "texture",
    14749 : "texture",
    14751 : "texture",
    14753 : "texture",
    14755 : "texture",
    14759 : "texture",
    14760 : "texture",
    14761 : "texture",
    14762 : "texture",
    14763 : "texture",
    14765 : "texture",
    14767 : "texture",
    14769 : "texture",
    14770 : "texture",
    14771 : "texture",
    14772 : "texture",
    14773 : "texture",
    14775 : "texture",
    14776 : "texture",
    14777 : "texture",
    14778 : "texture",
    14779 : "texture",
    14780 : "texture",
    14781 : "texture",
    14783 : "texture",
    14785 : "texture",
    14786 : "texture",
    14787 : "texture",
    15621 : "texture.read",
    15623 : "texture.read",
    15625 : "texture.read",
    15627 : "texture.read",
    15629 : "texture.read",
    15631 : "texture.read",
    15633 : "texture.read",
    15635 : "texture.read",
    15637 : "texture.read",
    15639 : "texture.read",
    15641 : "texture.read",
    15643 : "texture.read",
    15645 : "texture.read",
    15647 : "texture.read",
    15649 : "texture.read",
    15651 : "texture.read",
    15655 : "texture.read",
    15657 : "texture.read",
    15659 : "texture.read",
    15661 : "texture.read",
    15663 : "texture.read",
    15665 : "texture.read",
    15666 : "texture.read",
    15667 : "texture.read",
    15669 : "texture.read",
    15671 : "texture.read",
    15673 : "texture.read",
    15675 : "texture.read",
    15677 : "texture.read",
    15679 : "texture.read",
    15681 : "texture.read",
    15683 : "texture.read",
    15685 : "texture.read",
    15687 : "texture.read",
    15689 : "texture.read",
    15691 : "texture.read",
    15693 : "texture.read",
    15695 : "texture.read",
    15697 : "texture.read",
    15699 : "texture.read",
    15701 : "texture.read",
    15703 : "texture.read",
    15705 : "texture.read",
    15707 : "texture.read",
    15708 : "texture.read",
    15709 : "texture.read",
    15711 : "texture.read",
    15713 : "texture.read",
    15715 : "texture.read",
    15717 : "texture.read",
    15719 : "texture.read",
    15720 : "texture.read",
    15721 : "texture.read",
    15723 : "texture.read",
    15725 : "texture.read",
    15727 : "texture.read",
    15729 : "texture.read",
    15731 : "texture.read",
    15733 : "texture.read",
    15735 : "texture.read",
    15737 : "texture.read",
    15739 : "texture.read",
    15741 : "texture.read",
    15743 : "texture.read",
    15744 : "texture.read",
    15745 : "texture.read",
    15747 : "texture.read",
    15815 : "texture.read",
    15817 : "texture.read",
    15819 : "texture.read",
    15821 : "texture.read",
    15823 : "texture.read",
    15825 : "texture.read",
    15827 : "texture.read",
    15829 : "texture.read",
    15831 : "texture.read",
    15833 : "texture.read",
    15835 : "texture.read",
    15837 : "texture.read",
    15839 : "texture.read",
    15841 : "texture.read",
    15843 : "texture.read",
    15845 : "texture.read",
    15847 : "texture.read",
    15849 : "texture.read",
    15851 : "texture.read",
    15853 : "texture.read",
    15855 : "texture.read",
    15857 : "texture.read",
    15859 : "texture.read",
    15861 : "texture.read",
    15863 : "texture.read",
    15865 : "texture.read",
    15867 : "texture.read",
    15869 : "texture.read",
    15871 : "texture.read",
    15873 : "texture.read",
    15875 : "texture.read",
    15877 : "texture.read",
    15879 : "texture.read",
    15881 : "texture.read",
    15883 : "texture.read",
    15885 : "texture.read",
    15887 : "texture.read",
    15889 : "texture.read",
    15891 : "texture.read",
    15893 : "texture.read",
    15895 : "texture.read",
    15897 : "texture.read",
    15899 : "texture.read",
    15901 : "texture.read",
    15903 : "texture.read",
    15905 : "texture.read",
    15907 : "texture.read",
    15911 : "texture.read",
    15913 : "texture.read",
    15915 : "texture.read",
    15916 : "texture.read",
    15917 : "texture.read",
    15919 : "texture.read",
    15921 : "texture.read",
    15923 : "texture.read",
    15925 : "texture.read",
    15927 : "texture.read",
    15929 : "texture.read",
    15931 : "texture.read",
    15933 : "texture.read",
    15935 : "texture.read",
    15937 : "texture.read",
    15939 : "texture.read",
    # THE FLAGS WORD AND THE SCHEDULING CLASS TOGETHER. Neither alone finishes the job - there
    # are only 34 distinct Flags words over the whole ISA, and a scheduling class is a pipe rather
    # than an operation - but the PAIR is fine-grained: what the instruction does, crossed with
    # which unit runs it.
    #
    # Scored against the 469 names grounded in Apple's own compiler output, requiring two named
    # members to agree: 78 correct, 1 wrong. The one miss is op556, which the table calls `zero`
    # and the rule calls `movimm` - and zero IS movimm with a zero immediate, so it is a
    # refinement rather than a contradiction, in the same family as store.tg against store. At the
    # function level the rule is 79 of 79.
    769   : "fadd.f16",
    773   : "fadd.a",
    782   : "ffma.f16",
    785   : "ffma.f16",
    787   : "ffma.f16",
    788   : "ffma.f16",
    789   : "ffma.f16",
    790   : "ffma.f16.a",
    791   : "ffma.f16.a",
    792   : "ffma.f16.a",
    793   : "ffma.f16.a",
    794   : "ffma.f16.a",
    795   : "ffma.f16.a",
    796   : "ffma.f16.a",
    797   : "ffma.a",
    988   : "fadd.imm",
    991   : "fadd.imm",
    992   : "fadd.imm.f16",
    1004  : "fadd.imm",
    1007  : "fadd.imm",
    1008  : "fadd.imm",
    1036  : "fadd.imm.a",
    1039  : "fadd.imm.a",
    1040  : "fadd.imm.a",
    1052  : "fadd.imm",
    1055  : "fadd.imm",
    1056  : "fadd.imm.f16",
    1960  : "fadd.imm",
    1972  : "fadd.imm",
    1975  : "fadd.imm",
    1976  : "fadd.imm",
    2088  : "fadd.imm.a",
    2100  : "fadd.imm.a",
    2103  : "fadd.imm.a",
    2104  : "fadd.imm.a",
    2105  : "fadd.imm.a",
    2108  : "fadd.imm.a",
    2120  : "fadd.imm.a",
    2152  : "fadd.imm",
    2164  : "fadd.imm",
    2167  : "fadd.imm",
    2168  : "fadd.imm",
    2216  : "fadd.imm",
    2228  : "fadd.imm",
    2231  : "fadd.imm",
    2232  : "fadd.imm",
    2344  : "fadd.imm.a",
    2356  : "fadd.imm.a",
    2359  : "fadd.imm.a",
    2360  : "fadd.imm.a",
    2361  : "fadd.imm.a",
    2364  : "fadd.imm.a",
    2376  : "fadd.imm.a",
    2408  : "fadd.imm",
    2420  : "fadd.imm",
    2423  : "fadd.imm",
    2424  : "fadd.imm",
    2848  : "simdgroup.mul",
    2856  : "simdgroup.mul",
    2860  : "simdgroup.mul",
    2872  : "simdgroup.mul.f16.f32",
    2892  : "simdgroup.mul",
    2896  : "simdgroup.mul.f16.f32",
    2904  : "simdgroup.mul.f16.f32",
    2972  : "simdgroup.mul.a",
    2976  : "simdgroup.mul.a",
    2984  : "simdgroup.mul.a",
    2988  : "simdgroup.mul.a",
    2992  : "simdgroup.mul.a",
    3000  : "simdgroup.mul.a",
    3020  : "simdgroup.mul.a",
    3024  : "simdgroup.mul.a",
    3032  : "simdgroup.mul.a",
    3036  : "simdgroup.mul",
    3040  : "simdgroup.mul",
    3048  : "simdgroup.mul",
    3052  : "simdgroup.mul",
    3056  : "simdgroup.mul",
    3064  : "simdgroup.mul",
    3084  : "simdgroup.mul",
    3088  : "simdgroup.mul",
    3096  : "simdgroup.mul",
    3232  : "fadd.imm",
    3264  : "fadd.imm.a",
    3280  : "fadd.imm",
    3296  : "fadd.imm",
    3299  : "fadd.imm",
    3328  : "fadd.imm.a",
    3331  : "fadd.imm.a",
    3344  : "fadd.imm",
    3347  : "fadd.imm",
    9321  : "cvt.f2i",
    9323  : "cvt.f2i",
    9325  : "cvt.f2i.f16",
    # RE-ITERATED after the flags-and-schedclass rule.
    1956  : "fadd.imm",
    2404  : "fadd.imm",
    2412  : "fadd.imm",
    3122  : "fmul.imm.sat.f32.to.f16",
    3138  : "fmul.imm.sat.f32.a",
    3233  : "fadd.imm",
    3279  : "fadd.imm",
    9326  : "cvt.f2i",
    # RE-ITERATED after the flags-and-schedclass rule.
    # RE-ITERATED after the flags-and-schedclass rule.
    # COMBINED ROUND 1: flags-and-schedclass with the three structural rules, iterated.
    986   : "fadd.imm",
    987   : "fadd.imm",
    989   : "fadd.imm",
    1002  : "fadd.imm",
    1003  : "fadd.imm",
    1005  : "fadd.imm",
    1034  : "fadd.imm.a",
    1035  : "fadd.imm.a",
    1037  : "fadd.imm.a",
    1050  : "fadd.imm",
    1051  : "fadd.imm",
    1053  : "fadd.imm",
    1952  : "fadd.imm",
    1958  : "fadd.imm",
    1959  : "fadd.imm",
    1961  : "fadd.imm",
    1964  : "fadd.imm",
    1970  : "fadd.imm",
    1971  : "fadd.imm",
    1973  : "fadd.imm",
    2080  : "fadd.imm.a",
    2084  : "fadd.imm.a",
    2086  : "fadd.imm.a",
    2087  : "fadd.imm.a",
    2089  : "fadd.imm.a",
    2092  : "fadd.imm.a",
    2098  : "fadd.imm.a",
    2099  : "fadd.imm.a",
    2101  : "fadd.imm.a",
    2144  : "fadd.imm",
    2148  : "fadd.imm",
    2150  : "fadd.imm",
    2151  : "fadd.imm",
    2153  : "fadd.imm",
    2156  : "fadd.imm",
    2162  : "fadd.imm",
    2163  : "fadd.imm",
    2165  : "fadd.imm",
    2208  : "fadd.imm",
    2212  : "fadd.imm",
    2214  : "fadd.imm",
    2215  : "fadd.imm",
    2217  : "fadd.imm",
    2220  : "fadd.imm",
    2226  : "fadd.imm",
    2227  : "fadd.imm",
    2229  : "fadd.imm",
    2336  : "fadd.imm.a",
    2340  : "fadd.imm.a",
    2342  : "fadd.imm.a",
    2343  : "fadd.imm.a",
    2345  : "fadd.imm.a",
    2348  : "fadd.imm.a",
    2354  : "fadd.imm.a",
    2355  : "fadd.imm.a",
    2357  : "fadd.imm.a",
    2400  : "fadd.imm",
    2406  : "fadd.imm",
    2407  : "fadd.imm",
    2409  : "fadd.imm",
    2418  : "fadd.imm",
    2419  : "fadd.imm",
    2421  : "fadd.imm",
    3230  : "fadd.imm",
    3231  : "fadd.imm",
    3262  : "fadd.imm.a",
    3263  : "fadd.imm.a",
    3265  : "fadd.imm.a",
    3278  : "fadd.imm",
    3281  : "fadd.imm",
    3294  : "fadd.imm",
    3295  : "fadd.imm",
    3297  : "fadd.imm",
    3326  : "fadd.imm.a",
    3327  : "fadd.imm.a",
    3329  : "fadd.imm.a",
    3342  : "fadd.imm",
    3343  : "fadd.imm",
    3345  : "fadd.imm",
    9322  : "cvt.f2i",
    # COMBINED ROUND 2: flags-and-schedclass with the three structural rules, iterated.
    2415  : "fadd",
    # COMBINED ROUND 3: flags-and-schedclass with the three structural rules, iterated.
    # RETRACTED: 702 opcodes were named fsat and are not saturate. The peer's compiler audit
    # against the Flags word found it - bit 7 says op903 is not in the saturate family and op1062
    # is - and two structural facts confirm it without any execution: SATURATE IS UNARY, and 702
    # of the 930 opcodes carrying the name have TWO OR THREE register source operands.
    #
    # What they actually are, from the probe sweep, is not one thing:
    #
    #     op1015  fmod        op1552  texture cube / LOD query    op2254  acos, asin, atan
    #     op3306  cos, log, acosh and more                        op3307  sqrt, distance, length
    #     op3310  fdiv, ldexp
    #
    # Only op904 and op964 - one source, class 51 - are selected by Metal's saturate(), and only
    # op1062 - one source, class 63, bit 7 - has been executed as clamp(x, 0, 1). So the name is
    # kept where the operand count is right AND the class is one of those two families, and
    # withdrawn everywhere else. The retracted opcodes go back to unnamed rather than to a guess.
    #
    # HOW IT HAPPENED is the more useful half. `fsat` was propagated across scheduling classes by
    # rules that were each individually measured - and they were measured on a table that already
    # contained the error, so every one of them scored well while spreading it. An anchor that is
    # wrong makes every rule downstream of it wrong in a way no holdout against that same table
    # can see. The only thing that caught it was an independent instrument: Apple's own Flags word,
    # read by a session that was not the one that made the mistake.
    # THE ARITY AUDIT, run after the fsat retraction because the same failure could be anywhere.
    # For each name, compare the number of REGISTER SOURCE operands its members carry against the
    # arity the operation actually has. Saturate is unary; a three-source fsat is not a saturate.
    # Beyond fsat it found only 101 more, which is the reassuring half:
    #
    #     48 shr and 24 shl with THREE sources, all in class 7 - which also holds funnel.shr and
    #        funnel.shl, and a funnel shift is exactly two data operands and a count. Corrected to
    #        funnel.* rather than retracted, since the direction was already established and only
    #        the funnel was missing.
    #     19 fadd with three sources, in classes 92/93/94 - retracted; a three-source add is not
    #        an add and nothing here says which of several things it is.
    #     10 `not` with two sources, in class 5 alongside add, sub, nand and andn - retracted for
    #        the same reason.
    # THE SIMDGROUP MATRIX UNIT, reached by the second probe family - atomics, textures,
    # threadgroup memory, barriers, control flow and simdgroup_matrix, none of which is an
    # expression over two buffer loads and so none of which the first family could express.
    #
    # Each scales in exactly ONE probe of 741, and the register widths confirm them independently
    # of the probe: an 8x8 matrix is 64 elements over a 32-lane simdgroup, so every lane holds
    # TWO elements. Two floats are 64 bits - a GPR32tup2 - and two halves are 32 bits, one GPR32.
    # That is exactly the declared shape of each, and it is not something the attribution knew.
    2842 : "simdgroup.mma.f32",  # dest, a, b, c all GPR32tup2: D = A*B + C over the simdgroup.
                                 # Anchors scheduling class 118, 81 opcodes with nothing named.
    838  : "simdgroup.mma.f16",  # the same, four GPR32 - two halves per lane
    # Texture forms that differ from the named ones only in what they return.
    14469: "texture.sample.f16", # GPR16 destination; op14661 is the same sample returning a float
    15909: "texture.read.v4",    # GPR32tup4 destination; op15813 reads one component
    # op999: THE MIXED-PRECISION ADD, f32 + f16 -> f32. 566 corpus instances, the highest-
    # frequency unnamed opcode in the ISA until now, and it took all three routes to settle.
    #
    #   reading   scales in idiv and imod and in NO other probe of 741, and sits in place between
    #             the Newton refinement and the conversion back to integer:
    #                 ffma   r4 = r0*r4 + r8
    #                 op999  r4 = r4 (op) E        dest == source 0 in 463 of 566 instances
    #                 fadd   r4 = r4 + r8
    #                 cvt.f2i r4
    #             Apple's form is SIX bytes with the second source an expression taking only four
    #             distinct values in the whole corpus - 2 in every division kernel.
    #   encoding  byte5[1] is the kind bit: set, the second source is an expression; cleared, it
    #             is a register. 092dac0c1013 -> 092dac0c1011 is Apple's own instruction with the
    #             operand exposed.
    #   execution the peer, on that encoding: 2.7 + 1.5 -> 4.2, -3.3 + 0.5 -> -2.8, 7.1 + 2.0 ->
    #             9.1, 0.3 + 4.0 -> 4.3, 1e-8 + 1.5 -> 1.5, 65600 + 0.5 -> 65600.5. Six of six,
    #             and a+b the only candidate consistent with all of them. 0x3E00 in the low half
    #             is 1.5 as f16 and a denormal as f32, and the answer is 4.2 - so the second
    #             source is read at HALF precision.
    #
    # THE SAME OPCODE COMPUTES SOMETHING ELSE IN THE TWELVE-BYTE FORM, where it returns its first
    # source whatever the second holds. See ledger/g17-an-opcode-is-not-one-instruction.toml: the
    # encoding is part of the identity, and this name is the name of the six-byte form.
    999  : "fadd.f32.f16",
    # THE SHARED-PRIMITIVE RULE. An opcode that scales in several probes is usually unattributable,
    # but not when the constructs it scales in have exactly one primitive in common and that
    # primitive is itself one of them. Then the intersection names it, and the extra probes are
    # corroboration rather than confounding.
    3791 : "floor.f16",  # scales in floor.h/h2/h4 AND fract.h/h2/h4, and fract(x) = x - floor(x).
                         # Class 142 is the f16 rounding class - rint.f16, ceil.f16, trunc.f16 -
                         # and op3791 has the identical GPR16 -> GPR16 shape. floor was the one
                         # member of that family with no name.
    798  : "ffma.f16",   # scales in distance, dot, fma, length, mix and normalize at half
                         # precision. Every one of those lowers through a multiply-add - dot is an
                         # fma chain, length is sqrt(dot), distance is length(a-b), normalize is
                         # v*rsqrt(dot), mix is an fma - and `fma` is itself in the set, so the
                         # intersection is the fma. Four GPR16 operands, one definition: a
                         # three-source half-precision FMA, the f32 form being op2190.
    # RETRACTED 2026-09-05, and the lattice caught it. I named op775 fabs.f16 because it scales in
    # fabs.h and in no integer bitwise probe. The block structure disagreed - op775 sits at offset
    # 1 of a block whose offset-0 and offset-4 members are both fadd.f16 - and the block was right.
    # Two facts settle it:
    #
    #   op775 scales in fabs.h AND fneg.h. It cannot be both, so it is neither.
    #   what the compiler emits for each differs only in the SOURCE MODIFIER:
    #       fabs      op775  reg:425 imm:2147483648 reg:426 imm:20  imm:128
    #       saturate  op767  reg:425 imm:2147483648 reg:426 imm:16  imm:128
    #       fadd      op774  reg:425 imm:2147483648 reg:427 imm:16  reg:428 imm:16
    #   and 20 is 16 (release) + 4 (absolute value, bit 2), which is the documented modifier.
    #
    # So op775 is the ONE-SOURCE form of fadd.f16 - a second operand encoded as an immediate
    # rather than a register - and fabs and fneg both select it by setting a modifier bit. This is
    # the same error the peer caught on op397 abs -> andn: naming an opcode after the construct
    # when a modifier does the work. The failure mode of naming by construction is exactly this,
    # and the guard against it is that a unary construct emitting a one-source form of a known
    # binary opcode is a modifier until proved otherwise.
    775  : "fadd.imm.f16",       # one-source form; fabs and fneg select it via the source modifier
    # op767 is the same one-source form in the ADJACENT block, and the two blocks are otherwise
    # identical in shape at every offset. saturate() selects this block and plain fadd selects
    # op774's, so the saturation is carried by the opcode rather than by a modifier. That is an
    # inference from selection, not an execution, and it is the one claim here worth testing.
    767  : "fadd.imm.sat.f16",
    1277 : "exp2.f16",       # float form op1272
    2575 : "log2.f16",
    3663 : "recip.f16",      # float form op3658
    3775 : "rint.f16",       # float form op3770
    3807 : "ceil.f16",
    3823 : "trunc.f16",
    3855 : "rsqrt.f16",      # float form op3850
    # The quad reductions complete a row that was two thirds named: f16 by code correspondence and
    # confirmed by the peer's execution, int by execution, and f32 now by construction.
    13857: "quad.sum.f32",     # f16 op13853, int op13889
    13865: "quad.product.f32", # f16 op13855
    13853 : "quad.sum.f16",
    13855 : "quad.product.f16",
    13873 : "quad.fmax.f32",
    13881 : "quad.fmin.f32",
    13889 : "quad.sum",
    13905 : "quad.smax",
    13913 : "quad.smin",
    13921 : "quad.umax",
    13929 : "quad.umin",
    # AND EVERY UNNAMED OPCODE TESTED FOR BOOLEANNESS. The four code bits are fixed, so any
    # opcode can be asked: sweep them, and if three or more of the opcodes reached are named
    # boolean AND each agrees with its own code, this one is boolean too and its code names
    # it. 6,092 unnamed opcodes tested, 56 pass. The test is falsifiable - an arithmetic
    # opcode in the same class reaches nothing that agrees.
    418   : "andn.a",
    430   : "and",
    442   : "and.a",
    443   : "and.a",
    445   : "and.a",
    596   : "mov.a",
    13439 : "orna",
    13466 : "nand",
    13494 : "nandn",
    13527 : "nor",
    13554 : "orn",
    13581 : "or",
    13593 : "or.a",
    13596 : "or.a",
    17750 : "xnor",
    17777 : "xor",
    17789 : "xor.a",
    # THE TRUTH-TABLE READING APPLIED TO EVERY BOOLEAN OPERAND SHAPE. The four code bits sit
    # at b2[0], b4[0], b4[1] and b5[3] in all of them - established on ten anchors in four
    # shapes where they explain 10 of 10, and 6 or 7 of 7 in the narrower ones where a
    # disagreeing anchor is most likely a provisional width-rule name rather than a misfit.
    396   : "andn",
    405   : "andn",
    599   : "mov.a",
    13441 : "orna",
    13445 : "orna",
    13468 : "nand",
    13496 : "nandn",
    13529 : "nor",
    13556 : "orn",
    17748 : "xnor",
    17752 : "xnor",
    17783 : "xor",
    # NAMED BY TRUTH TABLE. A two-input boolean opcode's operation is not an opaque code -
    # the code IS the truth table. Four bits carry f(1,1), f(0,0), f(1,0) and f(0,1), so the
    # operation is readable from the encoding without executing anything.
    #
    # Established on the nine named by execution and compilation: normalise all nine to
    # identical operand values and exactly FOUR bits differ - b2[0], b4[0], b4[1], b5[3] -
    # and one fixed permutation maps each truth table onto them. Nine of nine, then eight
    # and seven again in three other operand shapes, with ZERO clashes.
    #
    # THREE ARE CORRECTIONS TO EXECUTION, and the direction is worth noting: op13522,
    # op17745 and op13434 were named `not` by a sweep with a fixed second operand. But
    # nor(a,0), xnor(a,0) and orna(a,0) are ALL ~a, so that sweep could not separate them.
    # The truth table does not depend on what the inputs happened to be.
    556   : "zero.a",              # read from the truth-table field
    598   : "mov.a",             # read from the truth-table field
    11197 : "not.a",             # read from the truth-table field
    13442 : "orna",              # read from the truth-table field
    17780 : "xor",               # read from the truth-table field
    # TWENTY-FIVE OF THE WIDTH-RULE NAMES REQUIRED CHAINING and are marked [CHAINED].
    # The rule is validated for ONE step from an anchor established by compilation or
    # execution. Its TRANSITIVE CLOSURE IS NOT VALID, and the demonstration is decisive:
    # taken as an equivalence over all 6,668 admitted opcodes it merges and, or, xor,
    # nand, nor, xnor, andn, orn, nandn and not into ONE component of 80.
    #
    # The cause is that the test compares SIGNATURES, and two different operations can
    # have signatures differing by exactly one narrowing - so a chain can step sideways
    # into another operation while every individual step looks like a width change.
    # 18 of 213 named components are merged that way.
    #
    # These 25 are the ones at risk. They should be verified by execution before being
    # relied on, and they are flagged rather than removed because a single step from them
    # is still more likely right than absent.
    # THE WIDTH AXIS, ITERATED. Each pass creates anchors for the next - naming op9720
    # fselect from its condition code let the rule reach op9750, which has 374 corpus
    # instances. Three rounds: +75, +11, +1.
    #
    # THE WHOLE BLOCK IS PROVISIONAL AND SHOULD BE VERIFIED BY EXECUTION. The rule is
    # validated for ONE step from an anchor established by compilation or execution; its
    # TRANSITIVE CLOSURE IS NOT VALID, and the demonstration is decisive - taken as an
    # equivalence over all 6,668 admitted opcodes it merges and, or, xor, nand, nor, xnor,
    # andn, orn, nandn and not into ONE component of 80. 18 of 213 named components hold
    # two names that way.
    #
    # The cause is that the test compares SIGNATURES, and two different operations can have
    # signatures differing by exactly one narrowing - so a chain steps sideways into another
    # operation while every individual step looks like a width change. Rounds 2 and 3 are
    # chained and about 25 names are at risk; which 25 could not be pinned down exactly, so
    # the whole block carries the caveat rather than a subset of it.
    # See ledger/g17-the-width-rule-does-not-chain.toml.
    410   : "andn",                # andn (op401) at b5[4], operand 0 narrowed; n=276  [CHAINED - see below]
    11183 : "cvt.i2f",             # cvt.i2f (op11180) at b3[1], operand 0 narrowed; n=12
    401   : "andn",                # andn (op398) at b3[7], operand 2 narrowed; n=0
    407   : "andn",                # andn (op398) at b5[4], operand 0 narrowed; n=0
    409   : "andn",                # andn (op400) at b5[4], operand 0 narrowed; n=0
    436   : "and",                 # and (op427) at b5[4], operand 0 narrowed; n=0
    469   : "popcount",            # popcount (op466) at b5[4], operand 0 narrowed; n=0
    709   : "funnel.shr",          # funnel.shr (op700) at b10[6], operand 2 narrowed; n=0
    964   : "fadd.imm.sat.bf16",            # saturate (op916) at b3[1], operand 0 narrowed; n=0
    993   : "fadd.imm.f16",               # cvt.h (op990) at b10[4], operand 3 narrowed; n=0
    1057  : "fadd.imm.f16",               # cvt.h (op1009) at b3[1], operand 0 narrowed; n=0
    1058  : "fadd",                # fadd (op1010) at b3[1], operand 0 narrowed; n=0
    1287  : "exp2",                # exp2 (op1275) at b3[1], operand 0 narrowed; n=0
    2385  : "ffma",                # ffma (op2193) at b3[1], operand 0 narrowed; n=0
    2205  : "ffma",                # ffma (op2193) at b8[3], operand 4 narrowed; n=0
    2241  : "ffma",                # ffma (op2193) at b8[4], operand 2 narrowed; n=0
    2394  : "ffma",                # ffma (op2202) at b3[1], operand 0 narrowed; n=0
    2250  : "ffma",                # ffma (op2202) at b8[4], operand 2 narrowed; n=0
    2430  : "ffma",                # ffma (op2238) at b3[1], operand 0 narrowed; n=0
    2585  : "log2",                # log2 (op2573) at b3[1], operand 0 narrowed; n=0
    3341  : "fmul",                # fmul (op3293) at b3[1], operand 0 narrowed; n=0
    3305  : "fmul",                # fmul (op3293) at b8[4], operand 2 narrowed; n=0
    3350  : "fmul",                # fmul (op3302) at b3[1], operand 0 narrowed; n=0
    3673  : "recip",               # recip (op3661) at b3[3], operand 0 narrowed; n=0
    3785  : "rint",                # rint (op3773) at b3[3], operand 0 narrowed; n=0
    3801  : "floor",               # floor (op3789) at b3[3], operand 0 narrowed; n=0
    3817  : "ceil",                # ceil (op3805) at b3[3], operand 0 narrowed; n=0
    3833  : "trunc",               # trunc (op3821) at b3[3], operand 0 narrowed; n=0
    3865  : "rsqrt",               # rsqrt (op3853) at b3[3], operand 0 narrowed; n=0
    9838  : "fselect",             # fselect (op9748) at b2[1], operands 0,7 narrowed; n=0
    10249 : "addsat",              # addsat (op10240) at b3[1], operand 0 narrowed; n=0
    10243 : "addsat",              # addsat (op10240) at b8[3], operand 2 narrowed; n=0
    10251 : "addsat",              # addsat (op10242) at b3[1], operand 0 narrowed; n=0
    10809 : "mulhi",               # mulhi (op10797) at b8[3], operand 2 narrowed; n=0
    10811 : "madd.wide",           # madd.wide (op10799) at b8[3], operand 2 narrowed; n=0
    10865 : "madd",                # madd (op10838) at b3[1], operand 0 narrowed; n=0
    11492 : "csel.reg",            # csel.reg (op11402) at b2[1], operand 0 narrowed; n=0
    11511 : "fcmp",                # fcmp (op11421) at b2[1], operand 0 narrowed; n=0
    11634 : "subsat",              # subsat (op11625) at b3[1], operand 0 narrowed; n=0
    11628 : "subsat",              # subsat (op11625) at b8[3], operand 2 narrowed; n=0
    11636 : "subsat",              # subsat (op11627) at b3[1], operand 0 narrowed; n=0
    13464 : "nand",                # nand (op13461) at b3[7], operand 2 narrowed; n=0
    13470 : "nand",                # nand (op13461) at b5[4], operand 0 narrowed; n=0
    13472 : "nand",                # nand (op13463) at b5[4], operand 0 narrowed; n=0
    13492 : "nandn",               # nandn (op13489) at b3[7], operand 2 narrowed; n=0
    13498 : "nandn",               # nandn (op13489) at b5[4], operand 0 narrowed; n=0
    13500 : "nandn",               # nandn (op13491) at b5[4], operand 0 narrowed; n=0
    13533 : "nor",                 # nor (op13524) at b5[4], operand 0 narrowed; n=0
    13552 : "orn",                 # orn (op13549) at b3[7], operand 2 narrowed; n=0
    13558 : "orn",                 # orn (op13549) at b5[4], operand 0 narrowed; n=0
    13560 : "orn",                 # orn (op13551) at b5[4], operand 0 narrowed; n=0
    13587 : "or",                  # or (op13578) at b5[4], operand 0 narrowed; n=0  [CONTESTED: truth table says nor]
    14163 : "simd.shuffle",        # simd.shuffle (op14159) at b3[1], operand 0 narrowed; n=0
    14422 : "shl",                 # shl (op14395) at b3[1], operand 0 narrowed; n=0
    14423 : "shl",                 # shl (op14396) at b3[1], operand 0 narrowed; n=0
    14436 : "funnel.shl",          # funnel.shl (op14409) at b3[1], operand 0 narrowed; n=0
    14412 : "funnel.shl",          # funnel.shl (op14409) at b8[3], operand 4 narrowed; n=0
    16818 : "sar",                 # sar (op16809) at b3[1], operand 0 narrowed; n=0
    17044 : "shr",                 # shr (op17017) at b3[1], operand 0 narrowed; n=0
    17756 : "xnor",                # xnor (op17747) at b5[4], operand 0 narrowed; n=0
    2397  : "ffma",                # ffma (op2205) at b3[1], operand 0 narrowed; n=0  [CHAINED - see below]
    2253  : "ffma",                # ffma (op2205) at b8[4], operand 2 narrowed; n=0  [CHAINED - see below]
    2433  : "ffma",                # ffma (op2241) at b3[1], operand 0 narrowed; n=0  [CHAINED - see below]
    2442  : "ffma",                # ffma (op2250) at b3[1], operand 0 narrowed; n=0  [CHAINED - see below]
    3353  : "fmul",                # fmul (op3305) at b3[1], operand 0 narrowed; n=0  [CHAINED - see below]
    10252 : "addsat",              # addsat (op10243) at b3[1], operand 0 narrowed; n=0  [CHAINED - see below]
    11637 : "subsat",              # subsat (op11628) at b3[1], operand 0 narrowed; n=0  [CHAINED - see below]
    13473 : "nand",                # nand (op13464) at b5[4], operand 0 narrowed; n=0  [CHAINED - see below]
    2445  : "ffma",                # ffma (op2253) at b3[1], operand 0 narrowed; n=0  [CHAINED - see below]
    # THE WIDTH AXIS, SECOND PASS. With the cross-lane and bitwise blocks named, the same rule
    # reaches many more: one bit from a named opcode, SAME scheduling class, and the only
    # difference in Apple's declared operand list is a GPR32 narrowed to GPR16. Validated by
    # compilation on madd, add and sub - ledger/g17-the-width-axis-names-by-adjacency.toml.
    # Same-signature neighbours are excluded: an identical signature one bit away is a
    # different OPERATION, which op5074 against op10369 demonstrates.
    9750  : "fselect",             # fselect (op9720) at b8[4], operand 3 narrowed; n=374
    5077  : "fcmp",                # fcmp (op5074) at b1[0], operand 3 narrowed; n=22
    10858 : "mul",                 # mul (op10831) at b3[1], operand 0 narrowed; n=18
    10838 : "madd",                # madd (op10829) at b8[3], operand 2 narrowed; n=12
    621   : "addr16",              # addr16 (op612) at b10[2], operand 2 narrowed; n=7
    14421 : "shl",                 # shl (op14394) at b3[1], operand 0 narrowed; n=3
    9748  : "fselect",             # fselect (op9718) at b1[0], operand 3 narrowed; n=1
    9808  : "fselect",             # fselect (op9718) at b2[1], operands 0,7 narrowed; n=1
    406   : "andn",                # andn (op397) at b5[4], operand 0 narrowed; n=0
    432   : "and",                 # and (op423) at b5[4], operand 0 narrowed; n=0
    433   : "and",                 # and (op424) at b5[4], operand 0 narrowed; n=0
    434   : "and",                 # and (op425) at b5[4], operand 0 narrowed; n=0
    468   : "popcount",            # popcount (op465) at b5[4], operand 0 narrowed; n=0
    589   : "mov",                 # mov (op586) at b5[7], operand 0 narrowed; n=0
    587   : "mov",                 # mov (op586) at b6[2], operand 2 narrowed; n=0
    700   : "funnel.shr",          # funnel.shr (op697) at b10[2], operand 4 narrowed; n=0
    706   : "funnel.shr",          # funnel.shr (op697) at b10[6], operand 2 narrowed; n=0
    733   : "funnel.shr",          # funnel.shr (op724) at b10[6], operand 2 narrowed; n=0
    1046  : "fadd",                # fadd (op998) at b3[1], operand 0 narrowed; n=0
    1010  : "fadd",                # fadd (op998) at b10[1], operand 2 narrowed; n=0
    1054  : "fadd.imm.f16",               # cvt.h (op1006) at b3[1], operand 0 narrowed; n=0
    990   : "fadd.imm.f16",               # cvt.h (op1006) at b9[5], operand 0 narrowed; n=0
    1009  : "fadd.imm",               # cvt.h (op1006) at b10[4], operand 3 narrowed; n=0
    1284  : "exp2",                # exp2 (op1272) at b3[1], operand 0 narrowed; n=0
    2382  : "ffma",                # ffma (op2190) at b3[1], operand 0 narrowed; n=0
    2202  : "ffma",                # ffma (op2190) at b8[3], operand 4 narrowed; n=0
    2238  : "ffma",                # ffma (op2190) at b8[4], operand 2 narrowed; n=0
    2193  : "ffma",                # ffma (op2190) at b10[7], operand 6 narrowed; n=0
    2573  : "log2",                # log2 (op2570) at b8[4], operand 2 narrowed; n=0
    3338  : "fmul",                # fmul (op3290) at b3[1], operand 0 narrowed; n=0
    3293  : "fmul",                # fmul (op3290) at b8[3], operand 4 narrowed; n=0
    3302  : "fmul",                # fmul (op3290) at b8[4], operand 2 narrowed; n=0
    3661  : "recip",               # recip (op3658) at b8[4], operand 2 narrowed; n=0
    3782  : "rint",                # rint (op3770) at b3[3], operand 0 narrowed; n=0
    3773  : "rint",                # rint (op3770) at b8[4], operand 2 narrowed; n=0
    3798  : "floor",               # floor (op3786) at b3[3], operand 0 narrowed; n=0
    3789  : "floor",               # floor (op3786) at b8[4], operand 2 narrowed; n=0
    3814  : "ceil",                # ceil (op3802) at b3[3], operand 0 narrowed; n=0
    3805  : "ceil",                # ceil (op3802) at b8[4], operand 2 narrowed; n=0
    3830  : "trunc",               # trunc (op3818) at b3[3], operand 0 narrowed; n=0
    3821  : "trunc",               # trunc (op3818) at b8[4], operand 2 narrowed; n=0
    3981  : "rsqrt",               # rsqrt (op3978) at b8[4], operand 2 narrowed; n=0
    9324  : "cvt.f2i",             # cvt.f2i (op9320) at b3[1], operand 0 narrowed; n=0
    9792  : "fselect",             # fselect (op9700) at b2[1], operands 0,7,9 narrowed; n=0
    11415 : "fselect",             # fselect (op9700) at b6[2], operands 3,5 narrowed; n=0
    11421 : "fcmp",                # fcmp (op9706) at b6[2], operands 3,5 narrowed; n=0
    9846  : "fcmp",                # fcmp (op9816) at b8[4], operand 3 narrowed; n=0
    9990  : "msb",                 # msb (op9989) at b8[3], operand 2 narrowed; n=0
    10092 : "atomic.idx",          # atomic.idx (op10090) at b3[0], operand 5 narrowed; n=0
    10248 : "addsat",              # addsat (op10239) at b3[1], operand 0 narrowed; n=0
    10288 : "add",                 # add (op10279) at b3[1], operand 0 narrowed; n=0
    10291 : "add",                 # add (op10282) at b3[1], operand 0 narrowed; n=0
    10292 : "add",                 # add (op10283) at b3[1], operand 0 narrowed; n=0
    10294 : "add",                 # add (op10285) at b3[1], operand 0 narrowed; n=0
    10373 : "cmp",                 # cmp (op10370) at b7[3], operand 3 narrowed; n=0
    10805 : "mulhi",               # mulhi (op10793) at b8[3], operand 2 narrowed; n=0
    10797 : "mulhi",               # mulhi (op10793) at b9[7], operand 4 narrowed; n=0
    10807 : "madd.wide",           # madd.wide (op10795) at b8[3], operand 2 narrowed; n=0
    10799 : "madd.wide",           # madd.wide (op10795) at b9[7], operand 4 narrowed; n=0
    10849 : "mul",                 # mul (op10822) at b3[1], operand 0 narrowed; n=0
    10852 : "mul",                 # mul (op10825) at b3[1], operand 0 narrowed; n=0
    10853 : "madd",                # madd (op10826) at b3[1], operand 0 narrowed; n=0
    10855 : "mul",                 # mul (op10828) at b3[1], operand 0 narrowed; n=0
    10856 : "madd",                # madd (op10829) at b3[1], operand 0 narrowed; n=0
    10861 : "mul",                 # mul (op10834) at b3[1], operand 0 narrowed; n=0
    10862 : "madd",                # madd (op10835) at b3[1], operand 0 narrowed; n=0
    11182 : "cvt.i2f",             # cvt.i2f (op11179) at b3[1], operand 0 narrowed; n=0
    11180 : "cvt.i2f",             # cvt.i2f (op11179) at b8[3], operand 4 narrowed; n=0
    11191 : "not",                 # not (op11190) at b6[3], operand 2 narrowed; n=0
    11194 : "not",                 # not (op11193) at b6[3], operand 2 narrowed; n=0
    11402 : "csel.reg",            # csel.reg (op11372) at b1[0], operand 3 narrowed; n=0
    11403 : "csel.reg.imm",        # csel.reg.imm (op11373) at b6[3], operand 3 narrowed; n=0
    9739  : "csel",                # csel (op11374) at b6[2], operands 3,5 narrowed; n=0
    11404 : "csel",                # csel (op11374) at b6[3], operand 3 narrowed; n=0
    11447 : "csel",                # csel (op11437) at b9[7], operand 4 narrowed; n=0
    11490 : "csel",                # csel (op11460) at b1[0], operand 3 narrowed; n=0
    11496 : "csel",                # csel (op11466) at b1[0], operand 3 narrowed; n=0
    11501 : "csel",                # csel (op11471) at b1[0], operand 3 narrowed; n=0
    11600 : "publish.coord.a",       # publish.coord (op11564) at b6[3], operand 4 narrowed; n=0
    11633 : "subsat",              # subsat (op11624) at b3[1], operand 0 narrowed; n=0
    11659 : "sub.wide",            # sub.wide (op11655) at b8[3], operand 2 narrowed; n=0
    11661 : "sub.wide",            # sub.wide (op11657) at b8[3], operand 2 narrowed; n=0
    11673 : "sub",                 # sub (op11664) at b3[1], operand 0 narrowed; n=0
    11675 : "sub",                 # sub (op11666) at b3[1], operand 0 narrowed; n=0
    11676 : "sub",                 # sub (op11667) at b3[1], operand 0 narrowed; n=0
    11677 : "sub",                 # sub (op11668) at b3[1], operand 0 narrowed; n=0
    11678 : "sub",                 # sub (op11669) at b3[1], operand 0 narrowed; n=0
    12649 : "load",                # load (op12646) at b3[0], operand 5 narrowed; n=0
    12676 : "load",                # load (op12673) at b3[0], operand 5 narrowed; n=0
    12677 : "load",                # load (op12674) at b3[0], operand 5 narrowed; n=0
    12678 : "load",                # load (op12675) at b3[0], operand 5 narrowed; n=0
    12685 : "load",                # load (op12682) at b3[0], operand 5 narrowed; n=0
    12694 : "load",                # load (op12691) at b3[0], operand 5 narrowed; n=0
    12712 : "load",                # load (op12709) at b3[0], operand 5 narrowed; n=0
    12713 : "load",                # load (op12710) at b3[0], operand 5 narrowed; n=0
    13469 : "nand",                # nand (op13460) at b5[4], operand 0 narrowed; n=0
    13497 : "nandn",               # nandn (op13488) at b5[4], operand 0 narrowed; n=0
    13530 : "nor",                 # nor (op13521) at b5[4], operand 0 narrowed; n=0
    13557 : "orn",                 # orn (op13548) at b5[4], operand 0 narrowed; n=0
    13583 : "or",                  # or (op13574) at b5[4], operand 0 narrowed; n=0
    13584 : "or",                  # or (op13575) at b5[4], operand 0 narrowed; n=0
    13585 : "or",                  # or (op13576) at b5[4], operand 0 narrowed; n=0  [CONTESTED: truth table says nor]
    14048 : "reverse",             # reverse (op14047) at b8[3], operand 2 narrowed; n=0
    14161 : "simd.shuffle",        # simd.shuffle (op14157) at b3[1], operand 0 narrowed; n=0
    14159 : "simd.shuffle",        # simd.shuffle (op14157) at b8[3], operand 2 narrowed; n=0
    14214 : "simd.reduce_bool",    # simd.reduce_bool (op14211) at b8[3], operand 3 narrowed; n=0
    14337 : "shl.hi",              # shl.hi (op14310) at b3[1], operand 0 narrowed; n=0
    14418 : "shl",                 # shl (op14391) at b3[1], operand 0 narrowed; n=0
    14419 : "shl",                 # shl (op14392) at b3[1], operand 0 narrowed; n=0
    14395 : "shl",                 # shl (op14392) at b8[3], operand 3 narrowed; n=0
    14420 : "shl",                 # shl (op14393) at b3[1], operand 0 narrowed; n=0
    14396 : "shl",                 # shl (op14393) at b8[3], operand 3 narrowed; n=0
    14427 : "funnel.shl",          # funnel.shl (op14400) at b3[1], operand 0 narrowed; n=0
    14409 : "funnel.shl",          # funnel.shl (op14400) at b11[2], operand 2 narrowed; n=0
    16787 : "shr.lo",              # shr.lo (op16778) at b3[1], operand 0 narrowed; n=0
    16814 : "asr",                 # asr (op16805) at b3[1], operand 0 narrowed; n=0
    16808 : "asr",                 # asr (op16805) at b8[3], operand 2 narrowed; n=0
    16815 : "sar",                 # sar (op16806) at b3[1], operand 0 narrowed; n=0
    16809 : "sar",                 # sar (op16806) at b8[3], operand 2 narrowed; n=0
    16810 : "asr",                 # asr (op16807) at b8[3], operand 2 narrowed; n=0
    17040 : "shr",                 # shr (op17013) at b3[1], operand 0 narrowed; n=0
    17041 : "shr",                 # shr (op17014) at b3[1], operand 0 narrowed; n=0
    17017 : "shr",                 # shr (op17014) at b8[3], operand 3 narrowed; n=0
    17042 : "shr",                 # shr (op17015) at b3[1], operand 0 narrowed; n=0
    17018 : "shr",                 # shr (op17015) at b8[3], operand 3 narrowed; n=0
    17223 : "store",               # store (op17220) at b3[0], operand 5 narrowed; n=0
    17232 : "store",               # store (op17229) at b3[0], operand 5 narrowed; n=0
    17241 : "store",               # store (op17238) at b3[0], operand 5 narrowed; n=0
    17259 : "store",               # store (op17256) at b3[0], operand 5 narrowed; n=0
    17260 : "store",               # store (op17257) at b3[0], operand 5 narrowed; n=0
    17261 : "store",               # store (op17258) at b3[0], operand 5 narrowed; n=0
    17753 : "xnor",                # xnor (op17744) at b5[4], operand 0 narrowed; n=0
    17779 : "xor",                 # xor (op17770) at b5[4], operand 0 narrowed; n=0
    17781 : "xor",                 # xor (op17772) at b5[4], operand 0 narrowed; n=0
    # THE CROSS-LANE BLOCK, named by the encode session with a 32-LANE grid - 210 targets,
    # 27 dispatches. Each lane is seeded with 3i+1 so a returned value names the lane it came
    # from, and the permutation reads straight off the output. The reductions were checked
    # against the seeds arithmetically: max 94, min 1, sum 1520, or 127, xor 0.
    #
    # A ONE-THREAD GRID CANNOT DO THIS and it is not a matter of effort: a shuffle has no other
    # lane to read and returns a constant. 81 opcodes had looked like they computed nothing.
    #
    # What unblocked it was a STORE, not the grid. Every store in the backend carried a slot,
    # so 32 threads all wrote one element; op17229 takes its index in a REGISTER, which Apple
    # emits for f[tg.x] = ... and which the corpus had all along.
    14037 : "simd.identity",
    14039 : "simd.identity",
    14041 : "simd.identity",
    14298 : "simd.identity",
    14300 : "simd.identity",
    14302 : "simd.identity",
    # 13873 : "quad.broadcast0",   SUPERSEDED by the quad/simd code correspondence
    # 13881 : "quad.broadcast0",   SUPERSEDED by the quad/simd code correspondence
    13883 : "quad.fmin.f16",
    # 13913 : "quad.broadcast0",   SUPERSEDED by the quad/simd code correspondence
    # 13929 : "quad.broadcast0",   SUPERSEDED by the quad/simd code correspondence
    13875 : "quad.fmax.f16",
    # 13905 : "quad.broadcast3",   SUPERSEDED by the quad/simd code correspondence
    13946 : "simd.shuffle_xor1",
    13948 : "simd.shuffle_xor1",
    13950 : "simd.shuffle_xor1",
    14175 : "simd.shuffle_xor1",
    14171 : "simd.shuffle_xor4",
    14173 : "simd.shuffle_xor4",
    14022 : "quad.shuffle_down1",
    14024 : "quad.shuffle_down1",
    14026 : "quad.shuffle_down1",
    14028 : "quad.shuffle_down1",
    14036 : "quad.shuffle_up1",
    14038 : "quad.shuffle_up1",
    14040 : "quad.shuffle_up1",
    14297 : "simd.shuffle_up1",
    14299 : "simd.shuffle_up1",
    14301 : "simd.shuffle_up1",
    14285 : "simd.shuffle_down1",
    14287 : "simd.shuffle_down1",
    14289 : "simd.shuffle_down1",
    14259 : "simd.rotate_up1_16",
    14261 : "simd.rotate_up1_16",
    14263 : "simd.rotate_up1_16",
    14265 : "simd.rotate_up1_16",
    14239 : "simd.rotate_down1_16",
    14241 : "simd.rotate_down1_16",
    14237 : "simd.rotate_down4_16",
    16922 : "simd.xor",
    16875 : "simd.prefix_sum",
    # SIX MORE FROM THE CONDITION CODE, found by censusing what values each unnamed opcode's
    # immediates take in Apple's own code rather than in a mutation-walk witness.
    #
    # Operand 2 of each is drawn ONLY from a condition-code space - the integer one (8, 9, 10, 12,
    # 13, 14) or the float one (0, 1, 2, 5, 6) - in the same operand position as the compares whose
    # codes were established by compiling the six relational operators. The cmp/csel split is the
    # register count past the destination, as before, and the names carry no relation because the
    # relation is the code.
    #
    # NOT NAMED: op14062, whose operand 2 takes 0, 1 and 2 but which has NO register operand past
    # its destination and sits in base.wide's class. Five instances and no shape that reads as a
    # comparison, so the code space alone is not enough.
    10381: "cmp",       # integer cc 8/10/12, FLAGR destination, one GPR16 source
    10378: "cmp",       # integer cc 10/13, FLAGR destination, one GPR32 source
    9807:  "fcmp",      # float cc 0/1/2/5/6, one source
    9816:  "fcmp",      # float cc 2/5, one source
    9720:  "fselect",   # float cc 0/1/5/6, three registers - a fused compare-and-select
    9718:  "fselect",   # float cc 0/1, two registers
    # THE FLAG-WRITING COMPARES AND THE FLAG READ-BACK, isolated by predicated control flow.
    #
    #     if (f[tg.x] > 5.0f) { ... }   ->  op5074 writes FLAGR with cc 2, which is
    #                                       greater-than in the FLOAT code space
    #     a > 5u || b < 9u              ->  cmp, op14099, cmp, op14099, then a compare of the
    #                                       two materialised values with cc 8
    #
    # op14099 reads a FLAGR and writes a GPR16 - it turns a flag into a value, which is how a
    # short-circuit `or` combines two conditions. Its cc 8 is outside the integer namespace
    # (9, 10, 12, 13, 14) and looks like a non-zero test.
    #
    # op10381 and op10378 stay UNNAMED and that is the rule working, not a gap: they have exactly
    # op10372's and op10369's signatures in the same scheduling class, and an identical signature
    # one bit away means a different OPERATION, not a variant. op5074 is the proof - same
    # signature as op10369, and it is the float compare where op10369 is the integer one.
    5074:  "fcmp",       # float compare to a flag
    14099: "flag.mov",   # FLAGR to GPR16
    # NAMED BY EXECUTION, by the encode session. CORRECTED 2026-09-22: this said Apple's compiler
    # never emits these and "there is no Metal source that selects a nand". ~(x&y) selects one; so do
    # ~(x|y), x|~y and ~(x^y) for nor, orn and xnor. The four construct sweeps never wrote the source.
    # Each was authored from isa/g17-authoring.jsonl, fed six input tuples and matched against a
    # candidate library, with the 84 already-named opcodes as the control.
    13460: "nand",      # ~(a & b)
    13521: "nor",       # ~(a | b)
    13548: "orn",       # a | ~b
    17744: "xnor",      # ~(a ^ b)
    # 13434: "not",   SUPERSEDED by the truth-table reading above
    # 13522: "not",   SUPERSEDED by the truth-table reading above
    # 17745: "not",   SUPERSEDED by the truth-table reading above
                        # immediate. A compiler that only has the immediate form misses this one.
    # THE 64-BIT ARITHMETIC FAMILY, scheduling class 289, whose operation is a two-bit code at
    # byte6[1:0] and whose codes 0 and 1 are the same two operations with a GPR16 CARRY-IN:
    #
    #     0  op11033  two GPR32 sources and a GPR16   1  op10796  the same shape
    #     2  op11657  two GPR32 sources               3  op10272  the same shape
    #     4..7 do not decode
    #
    # Code 2 is established by compilation: (ulong)u[x] - (ulong)u[y] is ONE op11657, a GPR32tup2
    # destination from two GPR32 sources, and (ulong)u[x] - 1UL is one op11655, its immediate
    # form. Code 3 is its sibling and is NOT identified - Apple's compiler emits add + cmp + add
    # for the widening ADD rather than selecting it, so nothing here differences it.
    #
    # There IS a native 64-bit subtract on register PAIRS: L[x] - L[y] is one op11652 with three
    # GPR32tup2 operands, where L[x] + L[y] takes four instructions and L[x] & L[y] takes two.
    # THE WIDENING PAIR, and it is NOT symmetric. Executed by the peer on controlled seeds:
    #
    #   op11657   3 - 5   ->  hi 0xffffffff lo 0xfffffffe     SIGN EXTENDED, so signed
    #   op10272   0xffffffff + 1  ->  hi 0x00000001           CARRY in the high half, so unsigned
    #
    # An unsigned widening subtract would borrow and give hi 0; a signed widening add would read
    # -1 + 1 = 0. So the two are not a matched pair and the bare names lose exactly the fact a
    # compiler needs to choose between them. op10272 could not have been named from the corpus
    # despite 371 instances: Apple emits add + cmp + add for a 64-bit addition rather than
    # selecting it, so nothing differences it. Frequency is not reachability.
    11657: "sub.wide.s",   # (long)a - (long)b, one instruction
    10272: "add.wide.u",   # a + b -> {carry, sum}, 371 corpus instances
    11655: "sub.wide",    # the immediate form
    11652: "sub.64",      # pair minus pair, one instruction
    # THE 64-BIT SHIFT COMPONENTS. op697 funnel.shr and op14310 shl.hi were named earlier; these
    # complete the set, each isolated in a 64-bit shift kernel where it is the only unnamed
    # instruction:
    #     L[x] << 5   ->  shl.hi, shl, op14400          op14400 combines them
    #     S[x] >> 5   ->  op16778, shr, asr, or         op16778 supplies the bits moving down
    # op14400 has op697's exact operand signature and scheduling class, in the opposite direction.
    14400: "funnel.shl",
    16778: "shr.lo",
    # op724 and op727 are funnel.shr with a narrowed destination and source - op697's signature
    # with GPR16 in place of GPR32, same scheduling class, and one op724 is the whole of
    # ((x & 0xff) << 8) | ((x >> 8) & 0xff).
    724:   "funnel.shr",
    727:   "funnel.shr",
    # SELECTED BY mulhi(ulong, ulong), which is 21 instructions. op10795 is one of them, in the
    # mulhi scheduling class with a GPR32tup2 destination and three GPR32 sources, which is the
    # shape of a widening multiply-add. Not isolated, so the name records the shape.
    10795: "madd.wide",   # SELECTED BY the 64-bit mulhi sequence
    # op612 PRODUCES THE 16-BIT OPERAND A LOAD TAKES. One GPR32 source, one GPR16 destination, no
    # other operand at all - the rest of its operand list is widths. In 873 of its 891 corpus
    # instances the destination is read by the IMMEDIATELY FOLLOWING load, at operand 9; 16 more
    # go to a csel and 2 to op599, and none is ever unread.
    #
    # The name says what is established: it materialises a load's operand 9. WHAT ARITHMETIC IT
    # PERFORMS IS NOT ESTABLISHED - it may be a truncation of its source or an address transform,
    # and no Metal construct this project can write selects it outside the tensor and convolution
    # kernels, so there is nothing to difference it against. It is the largest single opcode in
    # the corpus whose operation is unknown.
    612:   "addr16",
    # TEXTURES, an area no earlier sweep reached at all. Each is ONE instruction in a kernel that
    # contains it once, preceded only by the publishes that place its coordinates.
    #
    # THE SAMPLE MODE IS AN OPERAND, NOT AN OPCODE: tex.sample, sample(level(2.0)),
    # sample(bias(1.0)) and sample(gradient2d(..)) ALL compile to op14661. A backend picks the
    # opcode once and writes the mode.
    14661: "texture.sample",
    14757: "texture.gather",
    15813: "texture.read",
    # The write is a SEQUENCE and is labelled accordingly: two op11564 publish the coordinates,
    # four op592 publish the value, and op10909 is the instruction that writes. Calling op10909
    # "texture.write" is a statement about which instruction Apple selects for the write, and the
    # coordinate publishes are not part of it.
    10909: "texture.write",   # SELECTED BY wtex.write(); preceded by six publishes
    11564: "publish.coord.a",   # publishes a texture coordinate, two per write
    # THE QUAD GROUP, likewise untouched before. One instruction each, one per kernel.
    # (duplicate key 13889 removed - a later identical entry silently won)
    # 13921: "quad.max",           # quad_max   SUPERSEDED by the quad/simd code correspondence
    13944: "quad.shuffle_xor",   # quad_shuffle_xor
    14010: "quad.shuffle",       # quad_shuffle
    14034: "quad.shuffle_up",    # quad_shuffle_up
    # And the four SIMD shuffle directions, which the first sweep had collapsed into op14157.
    14169: "simd.shuffle_xor",           # simd_shuffle_xor
    14235: "simd.shuffle_rotate_down",   # simd_shuffle_rotate_down
    14283: "simd.shuffle_down",          # simd_shuffle_down
    14295: "simd.shuffle_up",            # simd_shuffle_up, and 186 instances in the corpus
    # half4 addition emits FOUR of op774, one per lane, on GPR16 operands - so it is the scalar
    # half add, and op17226 is the four-wide store that follows.
    774: "fadd.f16",
    17226: "store",
    # SEVENTEEN MORE COMPARES AND FUSED COMPARE-SELECTS, named from the condition code.
    #
    # Each carries an INTEGER CONDITION CODE at operand 2 - values drawn only from 9, 10, 12, 13,
    # 14 and 15 - in the same operand position as op11462 cmp, op11452 cmp.imm and the csel family
    # whose codes were established by compiling the six relational operators. A code from that
    # namespace in that position is not something an unrelated instruction carries.
    #
    # The split between cmp and csel is the operand count: one register past the destination is a
    # comparison, two or more are a comparison whose two results are selected. Which relation each
    # one implements is the code, not the opcode, so the names carry no relation.
    #
    # NOT NAMED HERE: op9750, whose operand 2 is 0 in all 374 instances. Zero is not in the
    # integer namespace and its operand 3 is an expr, so it stays a typed hole rather than being
    # swept in with its neighbours.
    11437: "csel",          # cc 12 at operand 2; 3 register operands past the destination; n=431
    11432: "cmp",           # cc 9 at operand 2; 1 register operand past the destination; n=431
    11374: "csel",          # cc 9/15 at operand 2; 3 register operands past the destination; n=222
    11466: "csel",          # cc 9 at operand 2; 3 register operands past the destination; n=140
    11456: "csel",          # cc 9/10/12 at operand 2; 2 register operands past the destination; n=88
    11491: "cmp",           # cc 12 at operand 2; 1 register operand past the destination; n=12
    11463: "csel",          # cc 10 at operand 2; 3 register operands past the destination; n=12
    11460: "csel",          # cc 9/10 at operand 2; 2 register operands past the destination; n=10
    11502: "csel",          # cc 9/10/12 at operand 2; 2 register operands past the destination; n=6
    11461: "cmp",           # cc 9/10 at operand 2; 1 register operand past the destination; n=6
    11486: "csel",          # cc 12 at operand 2; 2 register operands past the destination; n=4
    11457: "csel",          # cc 12 at operand 2; 3 register operands past the destination; n=4
    11471: "csel",          # cc 9 at operand 2; 2 register operands past the destination; n=4
    11470: "csel",          # cc 9 at operand 2; 3 register operands past the destination; n=3
    11483: "csel",          # cc 12 at operand 2; 2 register operands past the destination; n=1
    # ONE BIT FROM A NAMED OPCODE, SAME SCHEDULING CLASS, and the only difference in
    # Apple's declared operand list is that one or more GPR32 operands became GPR16. That
    # is the WIDTH axis, and it is the same operation at a narrower operand.
    #
    # VALIDATED BY COMPILATION, not by the adjacency alone:
    #     u[x] * u[y] + u[z]              -> op10826 madd
    #     u[x] * uint(w[y]) + u[z]        -> op10829, exactly what the rule predicts
    #     u[x] + uint(w[y]) -> op10283    u[x] - uint(w[y]) -> op11668  (both already named)
    #
    # THE FILTER MATTERS. A one-bit neighbour with the SAME operand signature is a
    # different OPERATION, not a variant - sqrt and op3946 are one bit apart and op3946 is
    # what sin, cos and tan select. Those are excluded here and stay unnamed.
    11395: "select.cc",   # select.cc (op11365) at b6[3], operand 3 narrowed to GPR16; n=553
    11394: "clamp",       # clamp (op11364) at b6[3], operand 3 narrowed to GPR16; n=370
    10280: "add",         # add (op10279) at b9[7], operand 3 narrowed to GPR16; n=212
    11669: "sub",         # sub (op11666) at b8[3], operand 2 narrowed to GPR16; n=182
    10829: "madd",        # madd (op10826) at b9[7], operand 4 narrowed to GPR16; n=82
    437  : "and",        # andn (op397) at b2[2], operands 0,2,4 narrowed to GPR16; n=57  [CONTESTED: truth table says and]
    10831: "mul",         # mul (op10822) at b8[3], operand 2 narrowed to GPR16; n=54
    11674: "sub.rev",     # sub.rev (op11665) at b3[1], operand 0 narrowed to GPR16; n=34
    9989 : "msb",         # msb (op9986) at b3[1], operand 0 narrowed to GPR16; n=32
    11393: "csel.mixed",  # csel.mixed (op11363) at b1[0], operand 3 narrowed to GPR16; n=28
    10834: "mul",         # mul (op10825) at b8[3], operand 2 narrowed to GPR16; n=25
    14394: "shl",         # shl (op14391) at b8[3], operand 3 narrowed to GPR16; n=23
    11467: "select",      # select (op11375) at b2[1], operands 0,7,9 narrowed to GPR16; n=22
    17773: "xor",         # xor (op17770) at b3[7], operand 2 narrowed to GPR16; n=20
    11193: "not",         # not (op11190) at b3[1], operand 0 narrowed to GPR16; n=17
    17772: "xor",         # xor (op17771) at b9[7], operand 4 narrowed to GPR16; n=17
    10835: "madd",        # madd (op10826) at b8[3], operand 2 narrowed to GPR16; n=13
    13576: "or",          # or (op13575) at b9[7], operand 4 narrowed to GPR16; n=13
    11453: "csel.mixed",  # csel.mixed (op11363) at b2[1], operands 0,7 narrowed to GPR16; n=11
    11392: "csel.imm",    # csel.imm (op11362) at b6[3], operand 3 narrowed to GPR16; n=10
    13588: "or",         # abs (op397) at b2[1], operands 0,2,4 narrowed to GPR16; n=3  [CONTESTED: truth table says or]
    17775: "and",         # and (op424) at b2[2], operands 2,4 narrowed to GPR16; n=1  [CONTESTED: truth table says xor]
    17774: "xor",         # xor (op17771) at b3[7], operand 2 narrowed to GPR16; n=1
    # FLOAT SOURCE MODIFIERS. A float source operand's immediate is width PLUS a modifier, and the
    # decoder surfaces the modifier as an offset: +2 negates the operand, +4 takes its absolute
    # value. Measured by holding a kernel fixed and changing only the source expression:
    #
    #     f[x] * f[y]        op3290  byte2 05   operand imm 16
    #     f[x] * -f[y]       op3290  byte2 25   operand imm 18     b2[5] is the negate
    #     f[x] * fabs(f[y])  op3290  byte10 24  operand imm 20     b10[2] is the abs
    #     f[x] + f[y]        op998   byte2 04   operand imm 16
    #     f[x] - f[y]        op998   byte2 24   operand imm 18     the SAME negate bit
    #     fma(a,b,c)         op2190  byte15 80  addend imm 16
    #     fma(a,b,-c)        op2190  byte15 c0                     b15[6] negates the addend
    #
    # So there is no fsub opcode and no fnma opcode: subtraction is addition with the negate
    # modifier, and fma(-a,b,c) is the same instruction with its multiply operands commuted.
    # op586 IS A MOVE. Its dominant form is 4 bytes, GPR32 <- GPR32, carrying the same operand-
    # width immediates the ALU forms use (32 for a 32-bit operand, 0 otherwise). 143 of its 720
    # instances take an EXPR source instead of a register, which is a uniform the prologue
    # published.
    #
    # Isolated: `u[200] = u[7]` compiles to exactly THREE instructions - op586, a store, and end.
    # The load itself happens in the constant program, where op12688 fetches the constant address
    # (the index sits in bytes 6..7 at 128 per element, measured across u[0] through u[1024]), so
    # what op586 does in main is move the published value into place for the store.
    #
    # Its first consumer is a compare in 390 of 720, an add in 66 and a store in 59.
    586: "mov",
    # THE PROLOGUE. Some opcodes occur ONLY in constant programs - 23 of them - and the commonest
    # constant program in the whole corpus is two instructions long.
    #
    # op590 IS that program. In 1310 objects the entire constant program is `op590; end`, and:
    #   it writes R0L in all 1310, with byte-identical immediates in all 1310
    #   its definition is read in _agc.main in 1310 of 1310 - never dead
    #   its first reader is a load in 1036 of them
    #   r0 is the base register of 79% of those kernels' memory traffic
    # So it materialises the buffer base that the kernel addresses through. Named for what it
    # establishes, not for an operation - it is prologue, and the prologue's job is to publish.
    590: "mov.16",
    # op14061 does the same job in the LONGER prologues: it reads no register defined earlier -
    # a genuine source - lives in the constant program in 364 of 396 instances, and feeds loads
    # 314 times, adds 172 and carries 86, which is a 64-bit address being built in halves before
    # op592 publishes it.
    14061: "base.wide",
    # ROUND THREE: COUNT SCALING. Differencing says which opcodes an operation brings in; only
    # scaling says which of them IS the operation, because a lowering emits plumbing too. Each
    # kernel emits the same expression 1, 2 and 4 times over independent operands, and an opcode
    # whose count goes 1, 2, 4 is the operation while one that stays flat is setup.
    998:  "fadd",   # scales 1:1 with float ADD and with float SUBTRACT alike, so subtract is a
                    # negate modifier on the same opcode - the same shape as the integer ALU,
                    # where sub is its own opcode but the float side reuses one
    # op2192 IS NOT ffma. RETRACTED 2026-09-05. Metal's fma(p,q,r) compiles to op2190 - every
    # time, for float, and for the a*b+c spelling too - and to zero op2192. op2192 has 8 instances
    # in 1795 objects and nothing here attributes them; it is not the half-precision fma either,
    # since half fma produces neither. It goes back to being unnamed.
    #
    # op2190 is the fused multiply-add. fma(a,b,c), fma(-a,b,c), fma(a,-b,c) and fma(a,b,-c) each
    # compile to exactly one of it, which is also what makes the SIGN MODIFIERS visible: they are
    # not separate opcodes.
    2190: "ffma",   # fma(p,q,r) and p*q+r, float; see the source-modifier note below
    1006: "fadd.imm",  # scales with half(x) conversions in both directions
    # ROUND TWO OF THE ISOLATION SWEEP: atomics, SIMD-group reductions, threadgroup memory and
    # 64-bit arithmetic. Same rule - a kernel containing exactly one occurrence of the operation,
    # differenced against a baseline.
    # THE SIMD REDUCTION BLOCK, opcodes 16830..16920. One opcode per (operation, element type,
    # scan-or-reduce). Named by compiling ONE Metal builtin per kernel and reading back the single
    # opcode from this block that appears - not by isolation against a baseline, so the encode
    # session's "selected by, not computed by" caveat does not bite here: there is nothing else in
    # the kernel for the difference to be credited to.
    #
    # The operand model is uniform across the whole block and is Apple's own declaration:
    #     0 def  IRGPR32 (GPR16 for the 16-bit forms)     the reduced result
    #     1 use  imm
    #     2 use  GPR32   (GPR16 for the 16-bit forms)     the per-lane input
    #     3 use  imm                                       operand width
    # and the slot bits are in the same places for every member - destination at b0[4], b0[7],
    # b2[3], b2[4], b2[7], b7[3], b7[4], b7[5]; source at b1[1], b3[4], b3[5], b3[6], b3[7],
    # b5[0], b8[0], b8[1]. Measured by mutation on op16873, op16874 and op16906, identical in all
    # three. simd_xor does not exist in Metal, which is why no opcode carries that name.
    16830: "simd.and",           # simd_and, uint and int
    16882: "simd.or",            # simd_or, uint and int
    16874: "simd.sum",           # simd_sum, uint and int
    16838: "simd.sum.f16",       # simd_sum, half
    16842: "simd.sum.f32",       # simd_sum, float
    16840: "simd.product.f16",   # simd_product, half
    16850: "simd.product.f32",   # simd_product, float
    16906: "simd.umax",          # simd_max, uint
    16914: "simd.umin",          # simd_min, uint
    16890: "simd.smax",          # simd_max, int
    16898: "simd.smin",          # simd_min, int
    16858: "simd.fmax.f32",      # simd_max, float
    16860: "simd.fmax.f16",      # simd_max, half
    16866: "simd.fmin.f32",      # simd_min, float
    16868: "simd.fmin.f16",      # simd_min, half
    16873: "simd.prefix_sum",         # simd_prefix_{ex,in}clusive_sum, uint and int
    16837: "simd.prefix_sum.f16",     # simd_prefix_{ex,in}clusive_sum, half
    16841: "simd.prefix_sum.f32",     # simd_prefix_{ex,in}clusive_sum, float
    16839: "simd.prefix_product.f16", # simd_prefix_{ex,in}clusive_product, half
    16849: "simd.prefix_product.f32", # simd_prefix_{ex,in}clusive_product, float
    14157: "simd.shuffle",    # simd_shuffle AND simd_broadcast, each alone - broadcast is a
                              # shuffle from a constant lane, so one opcode serving both is what
                              # the ISA should look like
    14211: "simd.reduce_bool",# simd_any and simd_all both; op14208 accompanies all only
    # THE ATOMIC FAMILY. 208 opcodes share (schedclass, tsflags) in Apple's MCInstrDesc, and the
    # ones below are the members this corpus contains. The OPERATION is not the opcode: it is a
    # four-bit code at b4[5], b5[3], b6[3] and b7[7] that is identical in every member, and the
    # opcode carries the operand SHAPE instead. isa/g17-atomic-family.toml has both tables.
    10022: "atomic.noret",     # atomic_store_explicit - the exchange code with the result dropped
    10023: "atomic.noret.64",  # same, 64-bit value operand
    10090: "atomic.idx",       # atomic RMW with a register index: atomic_fetch_add(&a[tg.x], ..)
    10095: "atomic.cmpxchg",   # atomic_compare_exchange_weak_explicit; code 3 selects it
    11701: "atomic.tg.noret",  # threadgroup-scope, no result
    11765: "atomic.tg",        # threadgroup-scope, returns the old value
    11769: "atomic.tg.idx",    # threadgroup-scope with a GPR16 index
    10094: "atomic",          # atomic_exchange alone, and present in fetch_add, fetch_and and
                              # fetch_max - one RMW opcode with the operation as a field
    # THREADGROUP MEMORY, which CONFIRMS a reading this project declined to claim. The memory
    # family splits on TSFlags into a ...500/...600 group that takes a GPR32tup2 base and a
    # ...900/...a00 group that addresses from GPR16 or immediates. isa/g17-handoff.toml recorded
    # "a 16-bit address register cannot reach device memory, so these are PLAUSIBLY threadgroup
    # local - plausibly, and this file does not claim it".
    #
    # Isolation settles it. A kernel whose only memory traffic is `threadgroup uint *tgm` emits
    # op12364 and op12367 to read it and op13288 to write it, and those are exactly the
    # ...900/...a00 opcodes.
    12364: "load.tg", 12367: "load.tg", 12361: "load.tg", 12376: "load.tg",
    13288: "store.tg", 13285: "store.tg",
    # -----------------------------------------------------------------------------------------
    # NAMED BY ISOLATION. tools/g17opsweep.py compiles 102 kernels, each loading two runtime
    # values, applying exactly ONE Metal operation, and storing the result. Differenced against a
    # baseline that only loads and stores, what remains is the instructions that operation needs.
    # Where the difference is a SINGLE new opcode occurring once, the attribution is direct: the
    # opcode Apple selected is the operation the source asked for.
    #
    # Operands are loaded at runtime so nothing folds, and results are stored so nothing is dead.
    # Both matter: a folded operand emits no instruction and a dead result can emit none either.
    #
    # Opcodes appearing in several operations' differences are NOT named here - op2190, op998,
    # op1006, op11363 and op11365 recur across the transcendentals and are sequence members, not
    # operations. They stay typed holes.
    465:   "popcount",   # u_popcount alone
    11190: "not",        # u_not alone
    14047: "reverse",    # reverse_bits, unique to it
    # op397 IS andn, NOT abs. CORRECTED 2026-09-05 by execution: it computes a & ~b. Apple
    # selects it for abs(p) with the sign mask in the second operand, so `abs` named the USE and
    # `andn` names the instruction - the same distinction as op9986, where clz named the use and
    # msb the instruction.
    #
    # op437 and op13588 were named abs by the width-adjacency rule and inherit the correction,
    # which is exactly what ledger/g17-the-width-axis-names-by-adjacency.toml said would happen:
    # naming by adjacency propagates confidence, it does not create it.
    397:   "andn",
    3802:  "ceil",       # f_ceil alone
    3786:  "floor",      # f_floor; also f_fract, which is x - floor(x)
    3770:  "rint",       # f_rint alone
    3818:  "trunc",      # f_trunc; also f_fmod
    904: "fadd.imm.sat.f32",   # f_saturate and clamp(p,0,1) give the same opcode
    # op3978 IS NOT sqrt. RETRACTED 2026-09-05: the encode session EXECUTED it and it returns the
    # RECIPROCAL square root - 2.75 gives 0.60302269, 16 gives 0.25, 0.25 gives 2. The multiply
    # that follows it in Apple's sqrt lowering is not a rounding step, it is the algorithm:
    # sqrt(x) = x * rsqrt(x). So sqrt is NOT one instruction on this machine.
    #
    # Third time this shape has appeared - op9986 clz, op10822 multiply, now this. Isolation names
    # what Apple SELECTS for a source construct, which is the instruction's meaning only when the
    # lowering is one instruction long.
    #
    # What separates op3978 from op3850, which also returns 1/sqrt on 2.75, is NOT established.
    3978: "rsqrt",
    3850:  "rsqrt",      # f_rsqrt
    3658:  "recip",      # f_recip; also every division and tangent lowering
    # THE TRANSCENDENTAL UNIT'S OPERATION CODE is byte6[2:0], three bits, swept through the
    # decoder from one sqrt encoding:
    #
    #     0 rint    1 recip    2 rsqrt   3 rsqrt   4 log2    5 exp2    6 trig    7 bad
    # Codes 2 and 3 BOTH return the reciprocal square root when executed; what separates them
    # is not established, and neither is sqrt, which is x * rsqrt(x) and not an instruction.
    #
    # Cross-checked against Apple's own encodings: exp2 carries byte6 = a5, log2 a4, sqrt a2,
    # rsqrt a3, op3946 a6. Five of five.
    #
    # CODE 6 IS NOT NAMED "sin". It is the one instruction common to sin, cos, sinpi, cospi and
    # the fast:: and precise:: forms of each - twelve instructions in the sin lowering and this is
    # one of them - so what distinguishes sine from cosine is elsewhere (op9710 against op9711)
    # and this opcode alone does not compute either.
    3946: "trig",
    1272:  "exp2",       # f_exp2; f_exp and f_pow reach it through a scale
    2570:  "log2",       # f_log2; f_log and f_pow likewise
    3290:  "fmul",       # f_mul
    9320:  "cvt.f2i",    # c_f2u and c_f2s both
    11179: "cvt.i2f",    # c_u2f and c_s2f both
    10239: "addsat",     # u_addsat and s_addsat
    11624: "subsat",     # u_subsat
    # ONE INSTRUCTION EACH, verified by counting: the kernel contains exactly one unnamed opcode
    # and the construct is a single hardware operation.
    10826: "madd",        # a*b+c on three GPR32 sources, one instruction; also once per division
    11665: "sub.rev",     # imm - reg. 32u-x gives imm 32, 7u-x gives imm 7, nothing else moves
    1016:  "cvt.f32.f16", # half(float) - the whole kernel is load, this, store
    13511: "unpack",      # unpack_unorm4x8_to_float, one instruction with a format immediate
    # WITH A GPR32tup2 DESTINATION THIS IS THE FULL WIDENING MULTIPLY, not just the high half:
    # (ulong)u[x] * (ulong)u[y] compiles to ONE of these writing a register pair, and u[x]*u[x]
    # with only the high half used compiles to the same instruction. The name is kept because the
    # high-half-only spelling is how Apple's own code overwhelmingly uses it.
    10793: "mulhi",      # u_mulhi, s_mulhi, u_madhi
    11364: "clamp",      # u_clamp and s_clamp, twice each - a two-instruction lowering
    16807: "asr",        # s_shr, the arithmetic shift
    9706:  "fcmp",       # f_step isolated it, and COUNT SCALING on float comparisons confirms
                              # it scales 1:1 with them - step(p,q) IS a comparison, so the
                              # comparison is the operation and step the source that reaches it
    # NOT clz. THE PEER SESSION EXECUTED IT: input 109 returns 6, and clz(109) is 25 - 6 is the
    # index of the highest set bit. Metal's clz() lowers to this opcode PLUS a subtract from 31,
    # and the subtract was already in the isolation baseline, so the difference credited the pair
    # to the one new opcode. The opcode itself is "position of the most significant set bit".
    9986:  "msb",        # SELECTED BY u_clz; computes msb, confirmed by execution.
                     # msb(0) = 0xffff, which is -1, measured by the encode session.
    # SHARED BY A FAMILY, named for the family rather than one member.
    # THE FUSED COMPARE-AND-SELECT FAMILY. Metal's ternary does not lower to a compare and a
    # select; it lowers to ONE instruction carrying a condition code, a comparison operand and
    # BOTH results. The operand order is
    #
    #     dest, modifier, CONDITION CODE, source, width, COMPARED VALUE, IF TRUE, IF FALSE
    #
    # established by varying each of the three independently and watching exactly one operand
    # move:
    #     u[tg.x] > 5u ? 1u : 2u   ->  op11362  cc 10  src  5  1  2
    #     u[tg.x] > 5u ? 7u : 9u   ->  op11362  cc 10  src  5  7  9
    #     u[tg.x] > 40u ? 7u : 9u  ->  op11362  cc 10  src 40  7  9
    #
    # The members differ in which of the three values are registers and in the register width.
    # Condition codes are the shared namespace in isa/g17-condition-codes.toml.
    11362: "csel.imm",     # 32-bit, compared value and both results immediate
    11482: "csel.imm.16",  # the GPR16 form: simd_is_first() is lane == 0 ? 1 : 0
    11363: "csel.mixed",   # if-true immediate, if-false a register; ctz's x == 0 ? 32 : ... case
    # MORE OF THE FUSED COMPARE-AND-SELECT FAMILY, each isolated in a one-instruction kernel:
    #     u[x] < u[y] ? u[1] : u[2]   -> op11372   compare two registers, select two registers
    #     u[x] < u[y] ? 0u   : u[2]   -> op11373   select an immediate against a register
    #     the same on ushort          -> op11412   the GPR16 form of op11372
    # All three carry cc 9, unsigned less-than, in the same operand position as the compares.
    11372: "csel.reg",
    11373: "csel.reg.imm",
    11412: "csel.reg.16",
    # THE 64-BIT SHIFT PAIR, both one instruction and neither expressible as a plain shift.
    #     ((hi<<32)|lo) >> 3, low word   -> op697    takes hi, lo AND the amount: a funnel shift
    #     ((hi<<32)|lo) << n, high word  -> op14310  takes the source and n, and produces the bits
    #                                                a left shift by n moves OUT of the word
    # op14310's amount operand carries n, not 32-n: shifting by 3 and by 7 give 3 and 7. Writing
    # the same thing by hand as `(x<<3)|(y>>29)` gets a plain shr instead, so this form is only
    # reachable through a real 64-bit shift.
    697:   "funnel.shr",
    14310: "shl.hi",
    11375: "select",     # min, max, select and absdiff on both signednesses all emit it
    # THE IMMEDIATE-FORM INTEGER COMPARE, the sibling of op11462. `x < 5` selects it with the same
    # condition codes; `x <= 5` does not get a new code, it gets the immediate 6. That is why only
    # five codes exist. Recovered from ci_0, a six-instruction kernel where it is the only
    # arithmetic.
    11452: "cmp.imm",
    # THE FLOAT COMPARE, and its condition codes are a DIFFERENT namespace from the integer ones:
    #     eq 0   lt 1   gt 2   ge 5   le 6
    # Float keeps its own codes for le and ge instead of negating lt and gt, which is what NaN
    # requires - `not (a < b)` is not `a >= b` when either is NaN.
    9787:  "fcmp.cc",
    # A CONDITIONAL SELECT whose condition is a code immediate. abs(int) compiles to exactly one
    # of it - the sequence being `sub t = 0 - x` then op11365 choosing between t and x - and so
    # does `x < 0 ? -y : y`. Two independent kernels, one instruction each. It is also what
    # applies the quotient's sign at the end of a signed division.
    11365: "select.cc",
    9700:  "fselect",    # f_min and f_max both; the relation must be a field within it
    # ---------------------------------------------------------------------------------------
    # THE MEMORY FAMILY, named from APPLE'S OWN DECLARATIONS rather than from behaviour. Two
    # fields decide it and they agree with each other and with the opcode numbering:
    #
    #     TSFlags 0x400402500 or 0x400402900 with NumDefs 1  ->  load   (all in 126xx/127xx/123xx)
    #     TSFlags 0x400402600 or 0x400402a00 with NumDefs 0  ->  store  (all in 172xx/132xx)
    #
    # The split is exactly whether the instruction defines a register. 34 opcodes, 22876
    # instructions.
    #
    # THE TWO TSFlags VALUES ARE TWO ADDRESSING MODES, not one. The ...500/...600 family takes a
    # GPR32tup2 BASE - the address pair whose register byte1[5:1] carries - in 16 of 20 loads and
    # 12 of 14 stores. The ...900/...a00 family takes no register pair at all: op12361 and op13285
    # address from immediates alone, and op12364, op12376, op12367 and op13288 from GPR16 operands.
    # A 16-bit address register cannot reach device memory, so these are plausibly threadgroup
    # local - PLAUSIBLY, and this file does not claim it.
    #
    # This is structural: it says which of Apple's classes an opcode belongs to, not what its
    # addressing mode computes.
    12674: "load", 12675: "load", 12682: "load", 12688: "load", 12646: "load",
    12697: "load", 12680: "load", 12715: "load", 12652: "load", 12691: "load",
    12706: "load", 12681: "load", 12709: "load", 12710: "load", 12716: "load",
 12673: "load",
    17257: "store", 17235: "store", 17262: "store", 17263: "store", 17244: "store",
    17258: "store", 17229: "store", 17253: "store", 17199: "store", 17256: "store",
    17238: "store", 17220: "store",
    # read_sr: Apple declares a SIR32 operand on exactly these two opcodes and on no others, and
    # byte1[5:0] names the register through the 6-bit code table in isa/g17-special-registers.toml.
    14059: "read_sr",   # GPR32 destination
    14060: "read_sr",   # GPR16 destination
    # compare: defines a FLAGR. The relation the compiler emits was measured as `gt` by the encode
    # side (100 and 20 pass against an immediate of 8, 7 and 2 do not), and the flag it writes is
    # selected by byte0[7:5] in the SIX-byte forms.
    10369: "cmp", 10370: "cmp", 10372: "cmp",
    # EXEC-mask ops: scheduling class 25, NumDefs 0, and a FLAGR USE. They consume a flag and
    # change architectural control state - the encode side ran cmp/exec pairs where a mismatched
    # flag suppresses a guarded store and a matched one lets it run. op575 reads FLAGTRUE and
    # nothing else, 178 times, which is how an unconditional region is written.
    582: "exec", 579: "exec", 583: "exec", 578: "exec", 575: "exec", 577: "exec",
    # movimm: DEFINES a register and takes NO register input, so its result can only be a function
    # of its own immediates - there is no other source. All four share TSFlags 0x200400002001 and
    # differ only in destination class. op11842 is the one the loop work already called movimm,
    # from its role initialising an induction register before a header; op554 is the same form and
    # is the single most common unnamed opcode in the corpus at 13306 instructions.
    # CONTROL AND SYNCHRONISATION. Scheduling class 6 holds exactly two opcodes and both are
    # NumDefs 0 with an imm.t4 target operand; this project already established that a branch's
    # destination is offset + displacement, 1867 of 1867. Class 3 holds two: op684 terminates every
    # program in the corpus, 1323 of 1323, and op447 is the barrier - named causally by the encode
    # side, whose byte1 carries the SCOPE and whose bar_both probe emits exactly one of each value.
    # THE TENSOR MAC HAS TWO FORMS AND THEY DIFFER BY ONE OPERAND. op5106 declares FOUR register
    # operands - destination, tile A, tile B, and the accumulator as a USE - and op5107 declares
    # the first three and no accumulator input. So op5107 cannot add to a previous value: it
    # writes A*B, and op5106 writes A*B + acc.
    #
    # The corpus agrees without being asked. In all 32 objects that use op5107, its count equals
    # the number of distinct accumulators exactly, and over 125 accumulators the FIRST mac written
    # to each one is op5107 in 125 cases and op5106 in none. The sequence alternates 5107, 5106,
    # 5107, 5106 with matching accumulator registers.
    5107: "tensor.mac.init",
    462: "branch", 458: "branch",
    684: "end",
    447: "barrier",
    554:   "movimm",   # IRGPR32
    11842: "movimm",   # IRGPR32, two immediates
    555:   "movimm",   # GPR16
    11843: "movimm",   # GPR16, two immediates
    # add / sub, register-immediate and register-register
    10279: "add", 10282: "add", 11664: "sub", 11666: "sub", 11667: "sub",
    # bitwise, immediate forms
    423: "and", 13574: "or", 17770: "xor",
    # bitwise, register-register forms
    424: "and", 13575: "or", 17771: "xor",
    # shifts and multiply
    14392: "shl", 17014: "shr", 10822: "mul", 10825: "mul",
    # Narrow-operand forms, identified by COMPILATION rather than execution: one operation per
    # kernel with runtime-loaded operands, so the opcode Apple selects is the operation the
    # source asked for. Validated by two controls in the same harness - uint & imm returns
    # op423 and uint & uint returns op424, both of which the encode side confirmed on hardware.
    10283: "add",     # uint + ushort      (also confirmed by the encode side)
    10295: "add",     # ushort + ushort
    435:   "and",     # ushort & imm, ushort DESTINATION - a different cell from op426 below,
                      # which the encode side attributes to and(GPR16, imm) with a wider dest
    17043: "shr",     # ushort >> imm
    # The rest of the narrow-operand grid, from the encode side's isa/g17-opmap.toml. Each row
    # is attributable to a stated operand class in a kernel; the ones marked executed also have
    # a dispatch against preregistered values behind them.
    426:   "and",     # GPR16, imm
    425:   "and",     # GPR32, GPR16
    428:   "and",     # GPR16, GPR16
    13579: "or",      # GPR16, GPR16
    11668: "sub",     # u32, u16
    11680: "sub",     # u16, u16
    10828: "mul",     # u32, u16
    10864: "mul",     # u16, u16
    10289: "add",     # signed 16-bit, its own cell
    14391: "shl",     # executed
    14393: "shl",     # u32 by u16
    17013: "shr",     # uint, logical, executed
    17015: "shr",     # GPR32 by GPR16
    17045: "shr",     # GPR16, GPR16
    # Arithmetic shift right is a SEPARATE FAMILY, not a modifier bit: different opcode and
    # different length from the logical shifts.
    16805: "asr",     # int
    16817: "asr",     # short
    # Confirmed here by compilation, isolated to a single operation in the kernel. The source
    # was `(ushort tp >> 2) & 15` with a 32-bit result: the decoder reports shift amount 2 in
    # the operand, matching the source, and the instruction is fed by op14060, the 16-bit
    # special-register read, exactly as every corpus instance is.
    17016: "shr",     # GPR16 source by immediate, 32-bit destination
    # add with a 32-bit destination and two 16-bit sources. The source is
    # `(uint)a[i] + (uint)b[i]` over FULL-RANGE ushorts: the sum needs 17 bits, so the
    # destination must widen while the operands stay 16-bit. Masking the operands - which is
    # what an earlier attempt did - lets the sum fit in 16 bits and yields op10295 instead,
    # which is why this cell took two tries. One add in the kernel, one op10286 in the output.
    10286: "add",
    10837: "mul",     # same widening construction: (uint)ushort * (uint)ushort
    # The 64-bit add sequence, isolated against a baseline with no arithmetic at all. A kernel
    # that only loads and stores a ulong emits just the load and the store; adding ONE 64-bit
    # add introduces exactly op10279 twice, plus these two. op11462 takes two GPR32 values and
    # produces a GPR16, sitting between the low add and the high add, which is the carry-out;
    # op10285 then consumes that GPR16 together with the high word.
    # NAME REVISED 2026-09-05. op11462 is the register-form INTEGER COMPARE, and carry-out is one
    # of the things a compare computes: the carry of a 32-bit add is exactly "did the sum wrap",
    # which is an unsigned less-than. Compiling the six relational operators shows it directly -
    # every one of them selects op11462, and what changes is an operand:
    #
    #     u <  v   cc 9    u >  v   cc 10   u == v  cc 12
    #     s <  v   cc 13   s >  v   cc 14   s == v  cc 12
    #
    # with a trailing POLARITY operand that negates the result, so `le` is `not gt` and `ge` is
    # `not lt` and `ne` is `not eq`. Five codes and a negate cover all twelve signed and unsigned
    # relations. See isa/g17-condition-codes.toml.
    11462: "cmp",     # register-form integer compare; carry-out is cc 9 with both operands
    10285: "add",     # add with carry: GPR16 carry plus GPR32
    # The Neural Accelerator MAC. Not a new identification: this project recovered tensor.mac
    # by a separate byte-level route (isa/tensor-isa.toml), and the two agree independently -
    # every op5106 instance is 10 bytes, its byte0 values are the recorded byte0_seen set, and
    # the project's own agxdis.is_mac signature validator returns True on all of them.
    5106: "tensor.mac",
    # Program structure, from the constant program. Both are exceptionless over 1323 objects.
    # (duplicate key 684 removed - a later identical entry silently won)
    592: "publish",   # ndefs 0, and 389 of 389 instances carry a relocation: it writes a
                      # computed value to a relocation-addressed constant slot, which is what
                      # main later reads through the same relocations
    13483: "pad",     # the 2-byte filler, 89% of all constant-program instructions and the
                      # form with zero varying bits across the whole corpus
}


def role(inst):
    e = inst.opcode
    if e is None:
        return "unknown"
    sig = e.signature()
    if e.sched == g17ssa.BRANCH_SCHED:
        return "branch"
    if e.sched in g17ssa.EXEC_SCHED:
        return "exec"
    access = g17ssa.memory_access(inst)
    if access:
        return access[0]
    if any(sig[i] == "FLAGR" for i in range(min(e.ndefs, len(sig)))):
        return "compare"
    if "SIR32" in sig:
        return "sr_read"
    if e.sched in TENSOR_SCHED:
        return "tensor"
    if e.sched in ALU_SCHED:
        return "alu"
    return "other"


def graph(insts):
    """(producers, consumers) mapping SSA values to instruction offsets."""
    blocks, phis, defs, uses = g17ssa.build(insts)
    producers, consumers = {}, collections.defaultdict(list)
    for off, ds in defs.items():
        for lv in ds:
            producers[lv] = off
    for off, us in uses.items():
        for lv in us:
            consumers[lv].append(off)
    phi_args = collections.defaultdict(list)
    for block, table in phis.items():
        for leaf, phi in table.items():
            for _, v in phi.args:
                phi_args[(leaf, v)].append((leaf, phi.version))
    return blocks, phis, defs, uses, producers, consumers, phi_args


def forward(insts, start_name, depth=12):
    """Breadth-first forward slice from every value whose location matches start_name."""
    by_offset = {i.offset: i for i in insts}
    _, _, defs, uses, producers, consumers, phi_args = graph(insts)
    seeds = [lv for lv in list(consumers) + list(producers) if lv[0] == start_name]
    if not seeds:
        return []
    seen, frontier, out = set(seeds), list(seeds), []
    for level in range(depth):
        nxt = []
        for lv in frontier:
            for off in consumers.get(lv, ()):
                inst = by_offset[off]
                out.append((level, off, inst, role(inst)))
                for d in defs.get(off, ()):
                    if d not in seen:
                        seen.add(d)
                        nxt.append(d)
            for p in phi_args.get(lv, ()):
                if p not in seen:
                    seen.add(p)
                    nxt.append(p)
        if not nxt:
            break
        frontier = nxt
    return out


def _operand_sources(inst, use_list):
    """Pair each USE operand with the SSA values it reads.

    _locations appends register leaves in operand order, definitions to one list and uses to the
    other, so walking the operands and consuming the use list in step recovers the association
    that flattening lost.
    """
    from agxforge.g17 import regs as g17regs
    names = g17model.registers()
    out, idx = [], 0
    for k, (kind, val) in enumerate(inst.values):
        if inst.opcode.is_def(k):
            continue
        if kind == "reg":
            reg = names.get(val)
            n = len(g17regs.leaves(reg)) if reg else 0
            out.append((k, "reg", reg, use_list[idx:idx + n]))
            idx += n
        else:
            out.append((k, kind, val, None))
    return out


def backward(insts, target, depth=8):
    """Backward slice as a bound DAG rather than a nested string.

    A nested expression is unreadable for real code: these slices reuse subexpressions heavily,
    and printing them inline re-expands the same computation dozens of times. So each
    contributing instruction gets one binding, emitted in dependency order, and everything else
    refers to it. That is also the form a differential probe wants, because each binding is one
    instruction with named inputs.
    """
    blocks, phis, defs, uses = g17ssa.build(insts)
    by_offset = {i.offset: i for i in insts}
    producer = {}
    for off, ds in defs.items():
        for lv in ds:
            producer[lv] = off
    phi_at = {}
    for block, table in phis.items():
        for loc, phi in table.items():
            phi_at[(loc, phi.version)] = phi

    nodes, order = {}, []

    def visit(off, level):
        if off in nodes:
            return ("ref", off)
        if level >= depth:
            return ("deep", off)
        nodes[off] = None                      # placeholder breaks cycles
        inst = by_offset[off]
        kids = []
        for k, kind, val, lvs in _operand_sources(inst, uses.get(off, [])):
            if kind == "imm":
                kids.append(("imm", val))
            elif kind == "expr":
                kids.append(("reloc", val))
            elif kind == "reg":
                if not lvs:
                    kids.append(("live", val or "?"))
                    continue
                srcs, live, merged = [], False, []
                for lv in lvs:
                    p = producer.get(lv)
                    if p is None:
                        (merged if lv in phi_at else [None]).append(lv) if lv in phi_at else None
                        if lv not in phi_at:
                            live = True
                    elif p not in srcs:
                        srcs.append(p)
                parts = [visit(p, level + 1) for p in srcs]
                parts += [("phi", lv[0]) for lv in merged]
                if not parts:
                    kids.append(("live", val))
                elif len(parts) == 1 and not live:
                    kids.append(parts[0])
                else:
                    kids.append(("join", val, parts))
        # Only the resource this instruction itself names is shown. The conservative fan-out a
        # register-addressed access creates would otherwise flood the slice with every resource
        # in the function while saying nothing about this one.
        access = g17ssa.memory_access(inst)
        if access:
            kids.append(("state", access[1] or g17ssa.UNKNOWN_RESOURCE))
        if inst.opcode and inst.opcode.sched in g17ssa.EXEC_SCHED:
            kids.append(("state", g17ssa.EXEC))
        nodes[off] = (inst.opcode.id if inst.opcode else -1, kids)
        order.append(off)
        return ("ref", off)

    visit(target, 0)
    return order, nodes


def structural_name(opid):
    """A name an opcode earns from Apple's own metadata, without any semantic inference.

    Two families qualify. The memory family is identified by scheduling class plus
    MCInstrDesc.NumDefs, which is what already separates 14,499 loads from 5,497 stores. The
    special-register readers are identified by carrying a SIR32 operand, and their operands
    resolve to Apple's own register names - op14059 reads SR_TP_IN_GRID_X.

    These are not guesses about what the instruction computes. They are classifications the
    tables make, and printing UNKNOWN over them understates what is established.
    """
    e = g17model.opcodes().get(opid)
    if e is None:
        return None
    if e.sched in g17ssa.MEMORY_SCHED:
        return "load" if e.ndefs else "store"
    if "SIR32" in e.signature():
        return "read_sr"
    return None


def render(slice_result, known=None):
    """Emit the DAG as bindings, deepest first, one line per contributing instruction."""
    known = KNOWN_OPS if known is None else known
    order, nodes = slice_result

    def arg(a):
        if a[0] == "ref":
            return "t%08x" % a[1]
        if a[0] == "deep":
            return "t%08x?" % a[1]
        if a[0] == "imm":
            return str(a[1])
        if a[0] == "live":
            return str(a[1])
        if a[0] == "reloc":
            return "reloc:0x%x" % a[1]
        if a[0] == "phi":
            return "phi(%s)" % a[1]
        if a[0] == "state":
            return str(a[1])
        if a[0] == "join":
            return "%s{%s}" % (a[1], " | ".join(arg(x) for x in a[2]))
        return "?"

    out = []
    for off in order:
        opid, kids = nodes[off]
        name = known.get(opid) or structural_name(opid) or "UNKNOWN_%d" % opid
        out.append("    t%08x = %s(%s)" % (off, name, ", ".join(arg(k) for k in kids)))
    return "\n".join(out)


def memory_chains(insts):
    """Store-to-load pairs where the load reads exactly one reaching store.

    A value that goes register -> store -> load -> ALU can be followed end to end, which is how
    address fields, binding fields, element-width fields and data registers get separated
    without depending on what a shader finally writes out.
    """
    blocks, phis, defs, uses = g17ssa.build(insts)
    store_of, phi_at = {}, {}
    for off, ds in defs.items():
        for loc, ver in ds:
            if loc.startswith("M:"):
                store_of[(loc, ver)] = off
    for block, table in phis.items():
        for loc, phi in table.items():
            if loc.startswith("M:"):
                phi_at[(loc, phi.version)] = phi

    def sources(lv, seen=None):
        seen = seen or set()
        if lv in seen:
            return set()
        seen.add(lv)
        if lv in store_of:
            return {store_of[lv]}
        if lv in phi_at:
            out = set()
            for _, v in phi_at[lv].args:
                out |= sources((lv[0], v), seen)
            return out
        return {None}

    consumers = collections.defaultdict(list)
    for off, us in uses.items():
        for lv in us:
            consumers[lv].append(off)
    by_offset = {i.offset: i for i in insts}
    out = []
    for inst in insts:
        a = g17ssa.memory_access(inst)
        if not a or a[0] != "load" or not a[1]:
            continue
        for lv in uses.get(inst.offset, []):
            if lv[0] != a[1]:
                continue
            srcs = [x for x in sources(lv) if x is not None]
            if len(srcs) != 1:
                continue
            after = sorted({role(by_offset[o]) for d in defs.get(inst.offset, ())
                            for o in consumers.get(d, ())})
            out.append((srcs[0], inst.offset, a[1], after))
    return out


def immediate_fields(insts, in_role, feeds_role):
    """Value distributions for immediates in `in_role` instructions feeding `feeds_role`.

    Keyed by (opcode, operand index). The register class of the instruction's definition and its
    scheduling class come along, because a field's meaning is a property of the opcode and the
    position, not of the byte offset it happens to land on.
    """
    _, _, defs, uses, producers, consumers, _ = graph(insts)
    by_offset = {i.offset: i for i in insts}
    fields = collections.defaultdict(collections.Counter)
    for inst in insts:
        if role(inst) != in_role:
            continue
        downstream = {role(by_offset[o]) for d in defs.get(inst.offset, ())
                      for o in consumers.get(d, ())}
        if feeds_role not in downstream:
            continue
        for k, (kind, val) in enumerate(inst.values):
            if kind == "imm":
                fields[(inst.opcode.id, k, inst.opcode.sched)][val] += 1
    return fields


def main():
    from agxforge.g17 import machobj, agxdis
    path = sys.argv[1]
    if os.path.isdir(path):
        loc = machobj.locate(path + "/s.arc.metallib", path + "/out/object/0-0")
        f, sz = agxdis.sections(loc["obj"])
        insts = list(g17model.decode(loc["obj"][f:f + sz], loc["syms"]["_agc.main"]))
    else:
        blob = open(path, "rb").read()
        f, sz = agxdis.sections(blob)
        insts = list(g17model.decode(blob[f:f + sz], 0))

    if "--immfields" in sys.argv:
        i = sys.argv.index("--immfields")
        in_role, feeds = sys.argv[i + 1], sys.argv[i + 2]
        fields = immediate_fields(insts, in_role, feeds)
        print("immediates in %s instructions whose result reaches a %s" % (in_role, feeds))
        print("%-8s %-4s %-6s %-7s %-8s %s" % ("opcode", "op#", "sched", "values", "distinct", "most common"))
        for (opid, k, sched), dist in sorted(fields.items(), key=lambda kv: -sum(kv[1].values())):
            top = ", ".join("%s x%d" % (v, n) for v, n in dist.most_common(4))
            print("%-8d %-4d %-6d %-7d %-8d %s" % (opid, k, sched, sum(dist.values()), len(dist), top))
        return
    if "--memchain" in sys.argv:
        chains = memory_chains(insts)
        by_offset = {i.offset: i for i in insts}
        print("%d store-to-load chains with exactly one reaching store" % len(chains))
        for st, ld, res, after in chains[:10]:
            print("\n  %s" % res)
            print("    store op%-6d @%08x" % (by_offset[st].opcode.id, st))
            print(render(backward(insts, st, depth=3)))
            print("    load  op%-6d @%08x   result feeds: %s"
                  % (by_offset[ld].opcode.id, ld, ", ".join(after) or "nothing"))
        return
    if "--to" in sys.argv:
        what = sys.argv[sys.argv.index("--to") + 1]
        if what.startswith("0x"):
            off = int(what, 16)
            targets = [i for i in insts if i.offset == off]
        else:
            targets = [i for i in insts if role(i) == what]
        blocks = g17cfg.build(insts)
        cd = g17cfg.control_dependence(blocks)
        block_of = {i.offset: b.start for b in blocks.values() for i in b.insts}
        print("%d target instructions with role %s" % (len(targets), what))
        for inst in targets[:8]:
            # The guard chain: every branch this instruction's block is control dependent on,
            # with the expression that produced the flag that branch was standing on.
            guards = []
            for a in sorted(cd.get(block_of.get(inst.offset), ())):
                g = blocks[a].guard
                if g is None:
                    guards.append("<branch@%08x, no flag guard>" % a)
                else:
                    guards.append(render(backward(insts, g.offset, depth=4)))
            print("\n  %08x  %s" % (inst.offset, role(inst)))
            for g in guards:
                print("  guarded by:")
                print(g)
            print("  computes:")
            print(render(backward(insts, inst.offset)))
        return
    if "--from" in sys.argv:
        start = sys.argv[sys.argv.index("--from") + 1]
        hits = forward(insts, start)
        if not hits:
            print("no value named %s in this object" % start)
            return
        print("forward slice from %s: %d consumer steps" % (start, len(hits)))
        chain = collections.Counter()
        seen_at = {}
        for level, off, inst, r in hits:
            chain[(level, r)] += 1
            seen_at.setdefault((level, r), (off, inst))
        for (level, r), n in sorted(chain.items()):
            off, inst = seen_at[(level, r)]
            print("  depth %-2d %-8s x%-4d  e.g. op%-6d @%08x" % (level, r, n, inst.opcode.id, off))
        print("\n  path: %s" % " -> ".join(
            r for _, r in sorted({(l, r) for l, r in chain})))
        return

    # --immediates: what role consumes the values an immediate-bearing instruction produces
    _, _, defs, uses, producers, consumers, _ = graph(insts)
    by_offset = {i.offset: i for i in insts}
    table = collections.Counter()
    for inst in insts:
        n_imm = sum(1 for k, _ in inst.values if k == "imm")
        if not n_imm:
            continue
        downstream = set()
        for d in defs.get(inst.offset, ()):
            for off in consumers.get(d, ()):
                downstream.add(role(by_offset[off]))
        for r in (downstream or {"nothing"}):
            table[(role(inst), r)] += n_imm
    print("%-10s %-10s %s" % ("in role", "feeds", "immediate operands"))
    for (a, b), n in table.most_common(20):
        print("%-10s %-10s %d" % (a, b, n))


if __name__ == "__main__":
    main()


# NAMES DETERMINED BY EXECUTION WHERE KNOWN_OPS DIFFERS OR IS SILENT (tools/g17nameaudit.py,
# ledger/g17-opcode-map-names-audited-against-execution.toml).
#
# THEY LIVE HERE AND NOT IN KNOWN_OPS BECAUSE A KNOWN_OPS NAME IS MACHINERY, NOT A LABEL: the
# assembler derives each form's mnemonic from it (`fadd.imm@1006` when a name is shared), and operand
# maps and authored lines are keyed by the mnemonic. Renaming op1004 there took op1006's register map
# with it and failed nineteen modules of the gate (2026-09-23). So the correction is recorded beside
# the mnemonic name, not over it.
EXECUTED_NAMES = {
    1062: "dfdx.fine.sat", # KNOWN_OPS "fsat": cross-lane, a fine horizontal derivative saturated
    1004: "cvt.f16.f32",   # KNOWN_OPS "fadd.imm": executed as the f32 -> f16 conversion cc lowers
    11063: "msub",         # KNOWN_OPS "madd" (a sibling's name): execution fits a*b - c uniquely
    16808: "shr.b16",      # KNOWN_OPS "asr" (narrowed from op16805): bit-15 inputs fit a LOGICAL shift
    13488: "andn",         # unnamed: ~a & b, cc's measured andn
    16806: "sarv",         # unnamed: arithmetic shift by a register amount
    408: "and.imm",        # unnamed: an AND with an immediate mask
}
