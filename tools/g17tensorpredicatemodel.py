"""Measured scalar value models for the tensor predicate experiment.

Kept-source and released-source op612 variants have separate retained campaigns.
Both agree on the same forty inputs. Other modifiers and inputs still refuse.
"""
DOMAIN=frozenset(range(32,66))|{0,1,16,31,127,65535}


def interpret(opcode,size,fields,value):
    if type(value) is not int or value not in DOMAIN:raise ValueError('predicate input outside measured domain')
    if opcode==612 and size==12 and len(fields)==6:
        if tuple(fields[i] for i in (1,4,5))!=('imm:0','imm:0','imm:64') or fields[3] not in ('imm:32','imm:16'):
            raise ValueError('op612 modifiers outside measured configuration')
        return (1<<max(0,min(4,64-value)))-1
    if opcode==11452 and size==10 and len(fields)==8:
        if tuple(fields[i] for i in (1,2,4,5,6,7))!=('imm:0','imm:9','imm:16','imm:64','imm:15','imm:0'):
            raise ValueError('op11452 modifiers outside measured configuration')
        return 15 if value<64 else 0
    raise ValueError('predicate form has no measurement')
