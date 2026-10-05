"""Measured value models for the exact tensor op17016 configurations.

All 32 lane values and eight additional inputs were observed on hardware. This
does not generalize to other modifier tuples or all 16-bit inputs/registers.
"""
DOMAIN = frozenset(range(32)) | {255,256,4096,4097,4352,8192,32768,65535}
MODIFIER_POSITIONS = (1,2,4,5,6)
CONFIGURATIONS = ((16777248,0,32,2,32), (32,0,32,1,2))


def interpret(fields, value):
    if len(fields) != 7 or type(value) is not int or value not in DOMAIN:
        raise ValueError('op17016 input is outside the measured setup domain')
    if any(not fields[i].startswith('imm:') for i in MODIFIER_POSITIONS):
        raise ValueError('unexpected setup modifier operand kind')
    modifiers = tuple(int(fields[i].split(':')[1]) for i in MODIFIER_POSITIONS)
    if modifiers == CONFIGURATIONS[0]:
        return value >> 2
    if modifiers == CONFIGURATIONS[1]:
        return (value >> 1) & 3
    raise ValueError('op17016 configuration has no setup execution evidence')
