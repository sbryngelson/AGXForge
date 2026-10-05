#!/usr/bin/env python3
"""Construct measured base allocations 1/2, including the exact 1024-row table.

The table formula reproduces observations; its runtime purpose is unproven.
Other retained constants remain only partly decoded. Allocation 0 is excluded.
"""
import struct


ONE_WORDS={
    0xa0:0x60000000,0xa4:0x75b,0x120:0xfffeffff,0x124:0xaaab,
    0x160:0x50001c00,0x164:0x4b000000,0x168:0xa030401,0x16c:0x14051400,
    0x170:0x4f800000,0x174:0xffff7fff,0x178:0x3ff0000,0x17c:0xfc00fc00,
    0x180:0xc0,0x1a0:0xc00fc03,0x1a4:0x14001c00,0x1a8:0x34002400,
    0x1ac:0x54004400,0x1b0:0x74006400,0x1b4:0x2c00f000,
    0x1b8:0x4c003c00,0x1bc:0x6c005c00,0x1c0:0x1007c00,
    0x1c4:0x200,0x1c8:0x400,0x240:0xfdc0,
}
TWO_WORDS={
    0xc00:0xb6d0019,0xc08:0x5800,0xc0c:0x10,
    0xe00:0x19,0xe04:0x10000000,0xe08:0x73c7774,0xe0c:0x80000000,
    0xe14:0x22000,0xe18:0x100,0x1120:0x60000000,0x1124:0x75b,
    0x2500:0x1ab0300,0x2600:0x1ab0400,
}


def table_row(divisor):
    if not 1<=divisor<=1024:
        raise ValueError('only measured divisors 1..1024 are supported')
    shift=max(0,(divisor-1).bit_length()-1)
    return ((1<<(32+shift))-1)//divisor,shift


def allocation(index):
    if index not in (1,2):
        raise ValueError('only measured base allocations 1 and 2 are supported')
    raw=bytearray(0x10000 if index==1 else 0x20000)
    for offset,value in (ONE_WORDS if index==1 else TWO_WORDS).items():
        struct.pack_into('<I',raw,offset,value)
    if index==1:
        for offset in range(0,0x180,0x40):
            struct.pack_into('<I',raw,offset,0x40)
        struct.pack_into('<H',raw,0x1cc,0xf800)
        for j,value in enumerate(range(0x104,0x1a5,4)):
            struct.pack_into('<H',raw,0x1ce+2*j,value)
    else:
        for i in range(256):
            struct.pack_into('<I',raw,4*i,11*i+5)
            struct.pack_into('<I',raw,0x400+4*i,7*i+3)
            struct.pack_into('<I',raw,0x800+4*i,0xdeadbeef)
        for divisor in range(1,1025):
            struct.pack_into('<II',raw,0x3900+8*(divisor-1),*table_row(divisor))
    return bytes(raw)
