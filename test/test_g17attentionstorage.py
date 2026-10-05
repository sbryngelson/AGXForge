"""Real C storage checks against the application graph; no Metal or GPU calls."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import g17attentiongraph

HARNESS=r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "g17attentionstorage.h"
int main(void) {
  const size_t sizes[]={SIZES};const unsigned roles[]={ROLES};
  const size_t count=sizeof(sizes)/sizeof(sizes[0]);
  G17AttentionStorage s;assert(g17AttentionStorageInit(&s,count,sizes,roles));
  void *buffers[G17_ATTN_MAX]={0};const void *views[G17_ATTN_MAX]={0};
  void *owned[G17_ATTN_MAX]={0};const void *snapshots[G17_ATTN_MAX]={0};
  bool produced[G17_ATTN_MAX]={0};size_t failed=0,guards=0,readonly=0,outputs=0;
  for (size_t i=0;i<count;++i) {
    views[i]=buffers[i]=malloc(s.allocation[i]);assert(buffers[i]);
    if (roles[i]<=G17_ATTN_PARAMETER) snapshots[i]=owned[i]=calloc(1,sizes[i]);
  }
  assert(!g17AttentionPrepare(&s,buffers,snapshots));
  assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  for (size_t i=0;i<count;++i) if (roles[i]>=G17_ATTN_INTERMEDIATE) {
    produced[i]=true;
    assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"nonfinite_or_unwritten_output"));
    assert(failed==i);
    memset((char *)buffers[i]+G17_ATTN_GUARD,0,sizes[i]);outputs+=sizes[i]/4;
  }
  assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  for (size_t i=0;i<count;++i) {
    size_t corners[]={0,127,G17_ATTN_GUARD+sizes[i],s.allocation[i]-1};
    for (size_t j=0;j<4;++j) {
      ((unsigned char *)buffers[i])[corners[j]]^=1;
      assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"boundary_guard"));
      assert(failed==i);((unsigned char *)buffers[i])[corners[j]]^=1;++guards;
    }
    if (roles[i]<=G17_ATTN_PARAMETER) for (size_t j=0;j<2;++j) {
      size_t off=G17_ATTN_GUARD+(j?sizes[i]-1:0);
      ((unsigned char *)buffers[i])[off]^=1;
      assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"readonly_input_changed"));
      assert(failed==i);((unsigned char *)buffers[i])[off]^=1;++readonly;
    }
  }
  float *next=malloc(sizes[s.source]);assert(next);
  for (size_t j=0;j<sizes[s.source]/4;++j) next[j]=1.0f;
  assert(!g17AttentionBegin(&s,buffers,next));snapshots[s.source]=next;
  assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"nonfinite_or_unwritten_output"));
  memset(produced,0,sizeof(produced));assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  void *saved=buffers[1];buffers[1]=buffers[0];
  assert(!strcmp(g17AttentionPrepare(&s,buffers,snapshots),"aliased_allocation"));buffers[1]=saved;
  size_t bad[]={SIZE_MAX};unsigned badrole[]={G17_ATTN_INPUT};
  assert(!g17AttentionStorageInit(&s,1,bad,badrole));
  for (size_t i=0;i<count;++i) {free(buffers[i]);free(owned[i]);}free(next);
  printf("{\"allocations\":%zu,\"guard_controls\":%zu,\"readonly_controls\":%zu,\"output_words\":%zu}\n",count,guards,readonly,outputs);
}
'''


class AttentionStorage(unittest.TestCase):
    def test_raw_bits_admits_nan_payloads_without_disabling_guards_or_readonly(self):
        source=r'''#include <assert.h>
#include <stdlib.h>
#include "g17attentionstorage.h"
int main(void) {
 size_t sizes[]={16,4,4};unsigned roles[]={0,2,3},types[]={1,2,0};
 G17AttentionStorage s;assert(g17AttentionStorageInitTyped(&s,3,sizes,roles,types));
 assert(g17AttentionOutputFromInputBytes(&s,4));
 assert(g17AttentionIntermediateFromInputBytes(&s,1,0));
 uint32_t data[]={0xffffffff,0x7f800001,0,0};
 void *buffers[3];const void *views[3],*snapshots[]={data,NULL,NULL};
 for(size_t i=0;i<3;++i)views[i]=buffers[i]=malloc(s.allocation[i]);
 bool produced[]={false,true,true};size_t failed=0;
 assert(!strcmp(g17AttentionPrepare(&s,buffers,snapshots),"nonfinite_initial_output"));
 assert(g17AttentionSetValuePolicy(&s,2,G17_ATTN_RAW_BITS));
 assert(!strcmp(g17AttentionPrepare(&s,buffers,snapshots),"nonfinite_initial_intermediate"));
 assert(g17AttentionSetValuePolicy(&s,1,G17_ATTN_RAW_BITS));
 assert(!g17AttentionPrepare(&s,buffers,snapshots));
 assert(!g17AttentionBegin(&s,buffers,data));
 assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
 assert(s.type[1]==G17_ATTN_FLOAT16 && s.binding_width[1]==2);
 assert(s.type[2]==G17_ATTN_FLOAT32 && s.binding_width[2]==4);
 assert(!g17AttentionSetValuePolicy(&s,3,G17_ATTN_RAW_BITS));
 assert(!g17AttentionSetValuePolicy(&s,2,2));
 ((char *)buffers[2])[127]^=1;
 assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"boundary_guard"));
 ((char *)buffers[2])[127]^=1;((char *)buffers[0])[128]^=1;
 assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"readonly_input_changed"));
 ((char *)buffers[0])[128]^=1;
 assert(g17AttentionSetValuePolicy(&s,2,G17_ATTN_FINITE_VALUES));
 assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"nonfinite_or_unwritten_output"));
 for(size_t i=0;i<3;++i)free(buffers[i]);
}'''
        with tempfile.TemporaryDirectory() as tmp:
            c=Path(tmp)/'raw.c';binary=Path(tmp)/'raw';c.write_text(source)
            subprocess.run(['clang','-std=c11','-O2','-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-I',str(ROOT/'tools'),str(c),'-o',str(binary)],
                capture_output=True,check=True,timeout=30)
            subprocess.run([str(binary)],capture_output=True,check=True,timeout=10)

    def test_vector_binding_width_does_not_change_scalar_guard_or_reply_width(self):
        source=r'''#include <assert.h>
#include "g17attentionstorage.h"
int main(void) {
 size_t sizes[]={32,32};unsigned roles[]={G17_ATTN_INPUT,G17_ATTN_OUTPUT};
 unsigned types[]={G17_ATTN_UINT32,G17_ATTN_FLOAT32};G17AttentionStorage s;
 assert(g17AttentionStorageInitTyped(&s,2,sizes,roles,types));
 assert(g17AttentionSetBindingLanes(&s,1,4));assert(s.binding_width[1]==16);
 assert(g17AttentionElementBytes(s.type[1])==4);
 assert(!g17AttentionSetBindingLanes(&s,1,3));assert(s.binding_width[1]==16);
 assert(!g17AttentionOutputFromInputBytes(&s,4));
 assert(g17AttentionOutputFromInputBytes(&s,0));
 float data[8]={1,2,3,4,5,6,7,8};assert(g17AttentionValuesValid(&s,1,data));
 uint32_t *bits=(uint32_t *)data;bits[7]=0x7fc01234;
 assert(!g17AttentionValuesValid(&s,1,data)); // last lane cannot escape checking
 sizes[1]=12;assert(g17AttentionStorageInitTyped(&s,2,sizes,roles,types));
 assert(!g17AttentionSetBindingLanes(&s,1,4));assert(s.binding_width[1]==4);
 types[1]=G17_ATTN_FLOAT16;assert(g17AttentionStorageInitTyped(&s,2,sizes,roles,types));
 assert(!g17AttentionSetBindingLanes(&s,1,2));
}'''
        with tempfile.TemporaryDirectory() as tmp:
            c=Path(tmp)/'vector.c';binary=Path(tmp)/'vector';c.write_text(source)
            subprocess.run(['clang','-std=c11','-O2','-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-I',str(ROOT/'tools'),str(c),'-o',str(binary)],capture_output=True,check=True,timeout=30)
            subprocess.run([str(binary)],capture_output=True,check=True,timeout=10)

    def test_mutable_output_is_reinitialized_without_weakening_readonly_or_guards(self):
        source=r'''#include <assert.h>
#include <stdlib.h>
#include "g17attentionstorage.h"
int main(void) {
  size_t sizes[]={16,16,16};unsigned roles[]={0,1,3},types[]={1,1,1};
  G17AttentionStorage s;assert(g17AttentionStorageInitTyped(&s,3,sizes,roles,types));
  assert(!s.output_from_input);assert(g17AttentionOutputFromInput(&s));
  uint32_t first[]={1,2,3,4},next[]={0xffffffff,9,8,7},parameter[]={5,6,7,8};
  void *buffers[3];const void *views[3],*snapshots[]={first,parameter,NULL};
  for(size_t i=0;i<3;++i) views[i]=buffers[i]=malloc(s.allocation[i]);
  bool produced[]={false,false,false};size_t failed=0;
  assert(!g17AttentionPrepare(&s,buffers,snapshots));
  uint32_t *out=(uint32_t *)((char *)buffers[2]+128);
  assert(!memcmp(out,first,16));out[0]=123;
  assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  assert(!g17AttentionBegin(&s,buffers,next));snapshots[0]=next;
  assert(!memcmp(out,next,16));assert(!produced[2]);
  out[0]=456;produced[2]=true;
  assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  ((char *)buffers[0])[128]^=1;
  assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"readonly_input_changed"));
  ((char *)buffers[0])[128]^=1;((char *)buffers[2])[127]^=1;
  assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"boundary_guard"));
  for(size_t i=0;i<3;++i)free(buffers[i]);
  sizes[2]=32;assert(g17AttentionStorageInitTyped(&s,3,sizes,roles,types));
  assert(!g17AttentionOutputFromInput(&s));assert(!s.output_from_input);
  sizes[2]=16;types[2]=0;assert(g17AttentionStorageInitTyped(&s,3,sizes,roles,types));
  assert(!g17AttentionOutputFromInput(&s));
}'''
        with tempfile.TemporaryDirectory() as tmp:
            c=Path(tmp)/'storage.c';binary=Path(tmp)/'storage';c.write_text(source)
            subprocess.run(['clang','-std=c11','-O2','-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-I',str(ROOT/'tools'),str(c),'-o',str(binary)],capture_output=True,check=True,timeout=30)
            subprocess.run([str(binary)],capture_output=True,check=True,timeout=10)

    def test_slice_copy_uses_independent_input_and_checks_bounds_and_float_bits(self):
        source=r'''#include <assert.h>
#include <stdlib.h>
#include "g17attentionstorage.h"
int main(void) {
  size_t sizes[]={32,16};unsigned roles[]={0,3},types[]={1,0};
  G17AttentionStorage s;assert(g17AttentionStorageInitTyped(&s,2,sizes,roles,types));
  assert(!g17AttentionOutputFromInput(&s));
  size_t bad[]={1,2,17,32,SIZE_MAX};
  for(size_t i=0;i<5;++i) {
    assert(!g17AttentionOutputFromInputSlice(&s,bad[i]));assert(!s.output_from_input);
  }
  assert(g17AttentionOutputFromInputSlice(&s,16));
  uint32_t input[]={0xffffffff,2,3,4,0x3f800000,0x40000000,0x40400000,0x40800000};
  void *buffers[2];const void *views[2],*snapshots[]={input,NULL};
  for(size_t i=0;i<2;++i)views[i]=buffers[i]=malloc(s.allocation[i]);
  bool produced[]={false,false};size_t failed=0;
  assert(!g17AttentionPrepare(&s,buffers,snapshots));
  uint32_t *out=(uint32_t *)((char *)buffers[1]+128);
  assert(!memcmp(out,input+4,16));assert(memcmp(out,input,16));
  assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  out[0]=0;input[4]=0x41000000;
  assert(!g17AttentionBegin(&s,buffers,input));assert(!memcmp(out,input+4,16));
  ((char *)buffers[1])[144]^=1;
  assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"boundary_guard"));
  ((char *)buffers[1])[144]^=1;((char *)buffers[0])[128]^=1;
  assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"readonly_input_changed"));
  input[4]=0x7fc00000;
  assert(!strcmp(g17AttentionBegin(&s,buffers,input),"nonfinite_initial_output"));
  assert(!strcmp(g17AttentionPrepare(&s,buffers,snapshots),"nonfinite_initial_output"));
  for(size_t i=0;i<2;++i)free(buffers[i]);
}'''
        with tempfile.TemporaryDirectory() as tmp:
            c=Path(tmp)/'storage.c';binary=Path(tmp)/'storage';c.write_text(source)
            subprocess.run(['clang','-std=c11','-O2','-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-I',str(ROOT/'tools'),str(c),'-o',str(binary)],capture_output=True,check=True,timeout=30)
            subprocess.run([str(binary)],capture_output=True,check=True,timeout=10)

    def test_initialized_readonly_half_input_tracks_each_query_snapshot(self):
        source=r'''#include <assert.h>
#include <stdlib.h>
#include "g17attentionstorage.h"
int main(void) {
  size_t sizes[]={16,8,8};unsigned roles[]={0,2,3},types[]={1,2,1};
  G17AttentionStorage s;assert(g17AttentionStorageInitTyped(&s,3,sizes,roles,types));
  assert(g17AttentionIntermediateFromInputBytes(&s,1,4));
  uint16_t first[]={0x3800,0x3a00,0x3c00,0x4000,0x4200,0x4400,0x4500,0x4600};
  uint16_t second[]={0x4800,0x4900,0x4a00,0x4b00,0x4c00,0x4d00,0x4e00,0x4f00};
  void *buffers[3];const void *views[3],*snapshots[]={first,NULL,NULL};
  for(size_t i=0;i<3;++i)views[i]=buffers[i]=malloc(s.allocation[i]);
  bool produced[]={false,false,false};size_t failed=0;
  assert(!g17AttentionPrepare(&s,buffers,snapshots));
  uint16_t *half=(uint16_t *)((char *)buffers[1]+128);
  assert(!memcmp(half,first+2,8));assert(memcmp(half,first,8));
  assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  half[2]^=1;
  assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),
                 "readonly_initialized_intermediate_changed"));assert(failed==1);
  // A completed shader write may change initialized scratch, still checked for finiteness.
  produced[1]=true;assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  half[2]=0x7e00;
  assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"nonfinite_or_unwritten_output"));
  produced[1]=false;assert(!g17AttentionBegin(&s,buffers,second));snapshots[0]=second;
  assert(!memcmp(half,second+2,8));assert(memcmp(half,first+2,8));
  assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  // Old query bytes must fail even though they were valid at the first query.
  memcpy(half,first+2,8);
  assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),
                 "readonly_initialized_intermediate_changed"));assert(failed==1);
  assert(!g17AttentionBegin(&s,buffers,second));
  assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  for(size_t i=0;i<3;++i)free(buffers[i]);
}'''
        with tempfile.TemporaryDirectory() as tmp:
            c=Path(tmp)/'storage.c';binary=Path(tmp)/'storage';c.write_text(source)
            subprocess.run(['clang','-std=c11','-O2','-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-I',str(ROOT/'tools'),str(c),'-o',str(binary)],capture_output=True,check=True,timeout=30)
            subprocess.run([str(binary)],capture_output=True,check=True,timeout=10)

    def test_extended_storage_checks_last_allocation_and_capacity(self):
        source=HARNESS.replace('SIZES',','.join(['16']*64)).replace('ROLES',','.join(['0']+['1']*62+['3']))
        source=source.replace('  float *next=', '  assert(!g17AttentionStorageInit(&s,G17_ATTN_MAX+1,sizes,roles));\n  float *next=')
        with tempfile.TemporaryDirectory() as tmp:
            c=Path(tmp)/'storage.c';binary=Path(tmp)/'storage';c.write_text(source)
            subprocess.run(['clang','-std=c11','-O2','-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-I',str(ROOT/'tools'),str(c),'-o',str(binary)],capture_output=True,check=True,timeout=30)
            result=subprocess.run([str(binary)],capture_output=True,text=True,check=True,timeout=30)
        report=json.loads(result.stdout)
        self.assertEqual(report['allocations'],64);self.assertEqual(report['guard_controls'],256)
        self.assertEqual(report['readonly_controls'],126)

    def test_full_graph_guards_parameters_and_unwritten_intermediates(self):
        allocations=list(g17attentiongraph.graph()["allocations"].values())
        roles={"input":0,"parameter":1,"intermediate":2,"output":3}
        source=HARNESS.replace("SIZES",",".join(str(x["payload_bytes"]) for x in allocations))
        source=source.replace("ROLES",",".join(str(roles[x["role"]]) for x in allocations))
        with tempfile.TemporaryDirectory() as tmp:
            c=Path(tmp)/"storage.c";binary=Path(tmp)/"storage";c.write_text(source)
            subprocess.run(["clang","-std=c11","-O2","-Wall","-Wextra","-Werror",
                "-fsanitize=address,undefined","-I",str(ROOT/"tools"),str(c),"-o",str(binary)],
                capture_output=True,check=True,timeout=30)
            result=subprocess.run([str(binary)],capture_output=True,text=True,check=True,timeout=30)
        report=json.loads(result.stdout)
        self.assertEqual(report["allocations"],23)
        self.assertEqual(report["guard_controls"],92)
        self.assertEqual(report["readonly_controls"],26)
        self.assertEqual(report["output_words"],122880)


if __name__=="__main__":unittest.main()
