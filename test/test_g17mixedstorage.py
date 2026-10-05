"""Actual native storage with odd half counts and mixed types; CPU only."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
C = r'''
#include <assert.h>
#include <stdlib.h>
#include "g17attentionstorage.h"
int main(void) {
  size_t sizes[]={6,10,6,12};
  unsigned roles[]={G17_ATTN_INPUT,G17_ATTN_PARAMETER,G17_ATTN_INTERMEDIATE,G17_ATTN_OUTPUT};
  unsigned types[]={G17_ATTN_FLOAT16,G17_ATTN_FLOAT16,G17_ATTN_FLOAT16,G17_ATTN_FLOAT32};
  G17AttentionStorage s;assert(g17AttentionStorageInitTyped(&s,4,sizes,roles,types));
  // Zero, negative zero, smallest subnormal, maximum finite and a normal half.
  uint16_t input[]={0,0x8000,1},parameter[]={0,0x8000,1,0x7bff,0x3c00};
  void *buffers[4];const void *views[4];
  const void *snapshots[]={input,parameter,NULL,NULL};
  bool produced[]={false,false,true,true};size_t failed=99;
  for(size_t i=0;i<4;++i)views[i]=buffers[i]=malloc(s.allocation[i]);
  assert(!g17AttentionPrepare(&s,buffers,snapshots));
  assert(!memcmp((char*)buffers[0]+128,input,6));
  assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"nonfinite_or_unwritten_output"));
  assert(failed==2);
  // Leaving the last half unwritten must fail, including an odd element count.
  memset((char*)buffers[2]+128,0,4);
  assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"nonfinite_or_unwritten_output"));
  assert(failed==2);memset((char*)buffers[2]+132,0,2);
  assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"nonfinite_or_unwritten_output"));
  assert(failed==3);memset((char*)buffers[3]+128,0,12);
  assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  for(size_t i=0;i<4;++i) {
    size_t edges[]={0,127,128+sizes[i],s.allocation[i]-1};
    for(size_t j=0;j<4;++j) {
      ((char*)buffers[i])[edges[j]]^=1;
      assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"boundary_guard"));
      assert(failed==i);((char*)buffers[i])[edges[j]]^=1;
    }
  }
  for(size_t i=0;i<2;++i) {
    ((char*)buffers[i])[128+sizes[i]-1]^=1;
    assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"readonly_input_changed"));
    assert(failed==i);((char*)buffers[i])[128+sizes[i]-1]^=1;
  }
  uint16_t bad[]={0x7c00,0xfc00,0x7e01};
  for(size_t i=0;i<3;++i) {
    uint16_t x[]={0,0,bad[i]};
    assert(!strcmp(g17AttentionBegin(&s,buffers,x),"nonfinite_input"));
    assert(!memcmp((char*)buffers[0]+128,input,6));
  }
  uint16_t next[]={0x3c00,0x4000,0x4200};
  assert(!g17AttentionBegin(&s,buffers,next));snapshots[0]=next;
  assert(!memcmp((char*)buffers[1]+128,parameter,10));
  assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"nonfinite_or_unwritten_output"));
  assert(failed==2);
  // Invalid extents/types refuse without replacing an initialized layout.
  G17AttentionStorage saved=s;sizes[0]=5;
  assert(!g17AttentionStorageInitTyped(&s,4,sizes,roles,types));assert(!memcmp(&s,&saved,sizeof s));
  sizes[0]=6;types[0]=99;
  assert(!g17AttentionStorageInitTyped(&s,4,sizes,roles,types));assert(!memcmp(&s,&saved,sizeof s));
  for(size_t i=0;i<4;++i)free(buffers[i]);
  return 0;
}
'''


class MixedStorage(unittest.TestCase):
    def test_native_half_guards_completion_and_mixed_allocations(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'storage.c'
            binary = Path(tmp) / 'storage'
            source.write_text(C)
            subprocess.run(['clang', '-std=c11', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-I', str(ROOT/'tools'), str(source),
                '-o', str(binary)], check=True, capture_output=True, timeout=30)
            subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)
