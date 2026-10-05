"""Integer storage and transport controls: real host code, no GPU."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agxforge.g17.transport import Worker

C = r'''
#include <assert.h>
#include <stdlib.h>
#include "g17attentionstorage.h"
int main(void) {
  size_t sizes[]={368,368};unsigned roles[]={0,3},types[]={1,1};
  G17AttentionStorage s;assert(g17AttentionStorageInitTyped(&s,2,sizes,roles,types));
  uint32_t input[92];for(int i=0;i<92;++i)input[i]=0xffffffffu-i;
  void *buffers[]={malloc(624),malloc(624)};
  const void *views[]={buffers[0],buffers[1]},*snapshots[]={input,NULL};
  bool produced[]={false,true};size_t failed;
  assert(!g17AttentionPrepare(&s,buffers,snapshots));
  assert(!g17AttentionBegin(&s,buffers,input));
  assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  // Identical NaN-like bits are valid uint32, but refused as FP32 input.
  G17AttentionStorage f;assert(g17AttentionStorageInit(&f,2,sizes,roles));
  assert(!strcmp(g17AttentionPrepare(&f,buffers,snapshots),"nonfinite_input"));
  for(size_t k=0;k<2;++k) {
    ((char*)buffers[k])[623]^=1;
    assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"boundary_guard"));
    assert(failed==k);((char*)buffers[k])[623]^=1;
  }
  ((char*)buffers[0])[128]^=1;
  assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"readonly_input_changed"));
  types[0]=99;assert(!g17AttentionStorageInitTyped(&s,2,sizes,roles,types));
  free(buffers[0]);free(buffers[1]);return 0;
}
'''
ECHO = r'''
import json, struct, sys
identity=dict(storage_dtype='uint32',matrix_bytes=368,request_bytes=368,
 reply_bytes=368,reply_elements=92,completion_markers=False)
def emit(seq,payload=b'',**extra):
 h=json.dumps(dict(protocol=2,sequence=seq,bytes=len(payload),identity=identity,**extra)).encode()
 sys.stdout.buffer.write(struct.pack('<I',len(h))+h+payload);sys.stdout.buffer.flush()
emit(0,rows=1,columns=92)
seq=0
while True:
 n=sys.stdin.buffer.read(4)
 if not n: break
 payload=sys.stdin.buffer.read(struct.unpack('<I',n)[0]);seq+=1
 emit(seq,payload,status=0,boundary_guard=True,readonly_inputs=True)
'''


C16 = r'''
#include <assert.h>
#include <stdlib.h>
#include "g17attentionstorage.h"
int main(void) {
  enum { N=65537 }; // Odd count includes every uint16 pattern and an extra final word.
  size_t sizes[]={2*N,6,6,2*N};
  unsigned roles[]={G17_ATTN_INPUT,G17_ATTN_PARAMETER,G17_ATTN_INTERMEDIATE,G17_ATTN_OUTPUT};
  unsigned types[]={G17_ATTN_UINT16,G17_ATTN_UINT16,G17_ATTN_UINT16,G17_ATTN_UINT16};
  uint16_t *input=malloc(2*N),parameter[]={0x7c00,0x7e01,0xffff};
  for(size_t j=0;j<N;++j)input[j]=(uint16_t)j;
  G17AttentionStorage s;assert(g17AttentionStorageInitTyped(&s,4,sizes,roles,types));
  assert(g17AttentionOutputFromInputBytes(&s,0));
  assert(g17AttentionIntermediateFromInputBytes(&s,2,2*(N-3)));
  void *buffers[4];const void *views[4],*snapshots[]={input,parameter,NULL,NULL};
  bool produced[]={false,false,false,true};size_t failed;
  for(size_t i=0;i<4;++i)views[i]=buffers[i]=malloc(s.allocation[i]);
  assert(!g17AttentionPrepare(&s,buffers,snapshots));
  assert(!memcmp((char*)buffers[3]+128,input,2*N));
  assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  for(size_t i=0;i<4;++i) {
    size_t edges[]={0,127,128+sizes[i],s.allocation[i]-1};
    for(size_t j=0;j<4;++j) {
      ((char*)buffers[i])[edges[j]]^=1;
      assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),"boundary_guard"));
      assert(failed==i);((char*)buffers[i])[edges[j]]^=1;
    }
  }
  for(size_t i=0;i<3;++i) {
    ((char*)buffers[i])[127+sizes[i]]^=1;
    assert(!strcmp(g17AttentionCheck(&s,views,snapshots,produced,&failed),
      i==2?"readonly_initialized_intermediate_changed":"readonly_input_changed"));
    assert(failed==i);((char*)buffers[i])[127+sizes[i]]^=1;
  }
  // A writable integer output is not mistaken for FP16, including NaNs and infinity bits.
  memcpy((char*)buffers[3]+128,parameter,6);
  assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  for(size_t j=0;j<N;++j)input[j]^=0xffff;
  assert(!g17AttentionBegin(&s,buffers,input));
  assert(!memcmp((char*)buffers[3]+128,input,2*N));
  assert(!g17AttentionCheck(&s,views,snapshots,produced,&failed));
  G17AttentionStorage saved=s;sizes[0]--;
  assert(!g17AttentionStorageInitTyped(&s,4,sizes,roles,types));assert(!memcmp(&s,&saved,sizeof s));
  sizes[0]++;types[0]=99;
  assert(!g17AttentionStorageInitTyped(&s,4,sizes,roles,types));assert(!memcmp(&s,&saved,sizeof s));
  // Same raw parameter is refused as float16; ushort must never select that check.
  s.type[1]=G17_ATTN_FLOAT16;
  assert(!strcmp(g17AttentionPrepare(&s,buffers,snapshots),"nonfinite_input"));
  for(size_t i=0;i<4;++i)free(buffers[i]);free(input);return 0;
}
'''


class UintStorage(unittest.TestCase):
    def test_uint16_all_bits_odd_extents_guards_and_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'test.c';binary=Path(tmp)/'test';source.write_text(C16)
            subprocess.run(['clang','-std=c11','-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-I',str(ROOT/'tools'),str(source),'-o',str(binary)],
                check=True,capture_output=True,timeout=30)
            subprocess.run([str(binary)],check=True,capture_output=True,timeout=10)

    def test_uint16_protocol_preserves_all_patterns_and_rejects_mismatches(self):
        # Odd element count and exhaustive patterns include half NaNs without float conversion.
        echo=ECHO.replace("'uint32'","'uint16'").replace('368','131074').replace('92','65537')
        original=np.arange(65537,dtype=np.uint32).astype('<u2')
        def worker(script=echo):
            return Worker([sys.executable,'-u','-c',script],rows=1,columns=65537,
                storage_dtype='uint16',request_elements=65537,reply_elements=65537,completion_markers=False)
        with closing(worker()) as w:
            a=w.query(original);b=w.query(original ^ np.uint16(0xffff));c=w.query(original)
            self.assertEqual(a.dtype,np.dtype('<u2'))
            self.assertEqual(a.tobytes(),original.tobytes());self.assertEqual(c.tobytes(),a.tobytes())
            self.assertEqual(b.tobytes(),(original ^ np.uint16(0xffff)).tobytes())
            c[0]=1;self.assertEqual(a[0],0)
        for bad in (original.view('<f2'),original.astype('<u4'),original[:-1]):
            with closing(worker()) as w:
                with self.assertRaises(ValueError):w.query(bad)
                self.assertTrue(w.closed)
        for wrong in (echo.replace("'uint16'","'float16'"),echo.replace('reply_bytes=131074','reply_bytes=131072')):
            with self.assertRaisesRegex(RuntimeError,'layout'):
                worker(wrong)
        with self.assertRaises(ValueError):
            Worker(['never-run'],rows=1,columns=3,storage_dtype='uint16')

    def test_mixed_uint32_request_uint16_reply_identity(self):
        script=ECHO.replace("matrix_bytes=368,request_bytes=368", "matrix_bytes=12,request_bytes=12")
        script=script.replace('reply_bytes=368,reply_elements=92','reply_bytes=6,reply_elements=3,reply_dtype="uint16"')
        script=script.replace('columns=92','columns=3').replace('emit(seq,payload,status=0',
            "emit(seq,b''.join(payload[j:j+2] for j in range(0,len(payload),4)),status=0")
        with closing(Worker([sys.executable,'-u','-c',script],rows=1,columns=3,
                storage_dtype='uint32',reply_dtype='uint16',request_elements=3,
                reply_elements=3,completion_markers=False)) as w:
            result=w.query(np.array([0x12347e01,0xabcdffff,0x43218000],dtype='<u4'))
            self.assertEqual(result.dtype,np.dtype('<u2'))
            self.assertEqual(result.tolist(),[0x7e01,0xffff,0x8000])

    def test_native_guards_and_readonly_work_for_all_integer_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'test.c'; binary=Path(tmp)/'test';source.write_text(C)
            subprocess.run(['clang','-std=c11','-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-I',str(ROOT/'tools'),str(source),'-o',str(binary)],
                check=True,capture_output=True,timeout=30)
            subprocess.run([str(binary)],check=True,capture_output=True,timeout=10)

    def test_real_protocol_preserves_nan_like_integer_words_and_repeats(self):
        w=Worker([sys.executable,'-u','-c',ECHO],rows=1,columns=92,
                 storage_dtype='uint32',request_elements=92,reply_elements=92,
                 completion_markers=False)
        try:
            original=np.full(92,0xffffffff,dtype=np.uint32);original[0]=0x7fc01234
            first=w.query(original); changed=w.query(original ^ np.uint32(0x80000000))
            repeated=w.query(original)
            np.testing.assert_array_equal(first,original)
            np.testing.assert_array_equal(changed,original ^ np.uint32(0x80000000))
            np.testing.assert_array_equal(first,repeated)
            repeated[0]=0;self.assertEqual(first[0],0x7fc01234)
            with self.assertRaisesRegex(ValueError,'implicit conversion'):
                w.query(original.astype(np.float32))
            self.assertTrue(w.closed)
        finally:w.close()


if __name__ == '__main__':
    unittest.main()
