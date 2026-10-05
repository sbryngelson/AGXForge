"""Exact-domain compare predicates measured through straight-line if/pop readout.

This models neither op579 nor branch gating. Source modifier0 is not covered by
this campaign; the retained modifier0 bundle was never dispatched.
"""
DOMAIN=frozenset(range(32,66))|{0,1,16,31,0x80000000,0xffffffff}

def interpret(size,fields,value):
    if type(value) is not int or value not in DOMAIN:raise ValueError('compare input outside measured domain')
    if size!=6 or len(fields)!=6 or fields[0]!='reg:74' or fields[1]!='imm:0' or fields[4]!='imm:32':
        raise ValueError('compare configuration outside source32 measurement')
    pair=(fields[2],fields[5])
    if pair==('imm:10','imm:32'):return value>32
    if pair==('imm:9','imm:64'):return value<64
    raise ValueError('compare relation/bound pair outside measurement')
