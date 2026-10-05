"""Value models for exact logic configurations measured in scalar probes.

Evidence: tensor-logic-runtime-v1/measurement-b (423) and
store-lifetime-runtime-v1/measurement-a (426 with preserved sources).
tensor-mask-runtime-v1/measurement-a adds op423 mask4 with source modifier16.
These rules do not establish source availability after preceding instructions.
"""
DOMAIN = frozenset(range(32)) | {255,256,4096,4097,4352,8192,32768,65535}
CONFIGURATIONS = {423: {(32,32,2), (32,16,4)}, 426: {(32,16,1)}}
# The projection's three op423 configurations, measured by integration's op423-runtime-v1 and -v2
# (the compiler's g17op423probe, handoff 10l/10m): the complete index domain 0..12287 as the source
# of the (0, 0, 63) form and of the (0, 0, 31) form (whose source in the probe is the thread id), the
# shifted index 0..191 as the source of the (0, 16, 31) form, and - from the v2 boundary query -
# the 64 mask-boundary controls as sources of the (0, 0, 63) form and their >> 6 as sources of the
# (0, 16, 31) form: 73,728 words equal to value & mask over three queries. Each configuration
# carries ITS OWN measured source set; a value outside it refuses by name. The controls are a copy
# of g17op423probe.CONTROLS (the regression asserts equality); measured points, not a licence for
# every 32-bit input.
PROJECTION_CONTROLS = [31, 32, 63, 64, 65, 127, 128, 255, 256, 12287, 12288, 65535, 65536, 2147483647, 2147483648, 4294967295, 4294967232, 4294967264, 63, 1984, 3735928559, 3405691582, 305419896, 252645135, 4042322160, 2863311530, 1431655765, 65472, 65504, 4032, 2147483584, 2147483680, 2147483711, 4294901760, 31, 32, 127, 64, 511, 512, 16383, 16384, 32767, 32768, 131071, 131072, 16777215, 16777216, 1073741823, 1073741824, 3221225471, 3221225472, 4294967231, 4294967263, 4294967168, 4294967040, 192, 224, 4095, 4096, 65535, 65537, 2147483584, 2147483679]
PROJECTION_CONFIGURATIONS = {423: {(0, 16, 31): frozenset(range(192)) | frozenset(c >> 6 for c in PROJECTION_CONTROLS),
                                   (0, 0, 63): frozenset(range(12288)) | frozenset(PROJECTION_CONTROLS),
                                   (0, 0, 31): frozenset(range(12288))}}
PROJECTION_EVIDENCE = 'results/g17-op423-runtime-v2/measurement-a (full_index, boundary, repeat_full_index) and -v1/measurement-a'



def interpret(opcode, fields, value):
    if opcode not in CONFIGURATIONS or len(fields) != 5:
        raise ValueError('logic form has no measured value model')
    if any(not fields[i].startswith('imm:') for i in (1,3,4)):
        raise ValueError('unexpected logic modifier operand kind')
    modifiers = tuple(int(fields[i][4:]) for i in (1,3,4))
    if modifiers in PROJECTION_CONFIGURATIONS.get(opcode, {}):
        if type(value) is not int or value not in PROJECTION_CONFIGURATIONS[opcode][modifiers]:
            raise ValueError('logic input is outside the measured domain of configuration %s (%s)' % (modifiers, PROJECTION_EVIDENCE))
        return value & modifiers[2]
    if type(value) is not int or value not in DOMAIN:
        raise ValueError('logic input is outside the measured 40-input domain')
    if modifiers not in CONFIGURATIONS[opcode]:
        raise ValueError('logic configuration has no execution evidence')
    return value & modifiers[2]
