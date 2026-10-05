"""The actual native Submit/wait helper against CPU callbacks, not GPU evidence."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

HOST=r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <Block.h>
#include <dlfcn.h>
struct shmem_result {uint64_t cpu,size;uint32_t id;};
static int no_agx_metal(void){return 1;}
static uint64_t ids=100;
static unsigned mode;
static uint64_t next_id(void* dev){(void)dev;return ++ids;}
static int submit(void* queue,void* ignored,unsigned count,void* raw,unsigned bytes,void* out){
  (void)queue;(void)ignored;assert(count==1 && bytes==64);memset(out,0,64);
  uintptr_t sp,cp;memcpy(&sp,(uint8_t*)raw+0x10,sizeof sp);memcpy(&cp,(uint8_t*)raw+0x18,sizeof cp);
  void (^scheduled)(void)=(void (^)(void))sp;
  void (^completed)(void)=(void (^)(void))cp;
  if(mode==0){scheduled();completed();usleep(2000);}
  else if(mode==1){dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT,0),^{scheduled();usleep(1000);completed();});}
  else {scheduled();completed();completed();}
  return 0;
}
static void* host_dlsym(void* handle,const char* name){
  (void)handle;
  if(!strcmp(name,"IOGPUCommandQueueSubmitCommandBuffers"))return (void*)submit;
  if(!strcmp(name,"IOGPUDeviceGetNextGlobalTraceID"))return (void*)next_id;
  return NULL;
}
#define dlsym host_dlsym
#define G17_BLOCK_PREFLIGHT 1
#include "g17workload_tile.h"
static void check(const struct workload_completion *r){
  assert(r->scheduled==1 && r->completed==1 && r->status==0 && r->outword==0);
  assert(r->first_trace_id && r->second_trace_id==r->first_trace_id+1);
  assert(r->submit_start_ns<=r->scheduled_callback_entry_ns && r->scheduled_callback_entry_ns<=r->wait_return_ns);
  assert(r->submit_start_ns<=r->completed_callback_entry_ns && r->completed_callback_entry_ns<=r->wait_return_ns);
  assert(r->submit_start_ns<=r->submit_return_ns && r->submit_return_ns<=r->wait_return_ns);
  assert(r->submit_ns==r->submit_return_ns-r->submit_start_ns);
  assert(r->completion_ns==(r->completed_callback_entry_ns-r->submit_start_ns)+(r->wait_return_ns-r->completed_callback_entry_ns));
}
int main(void){
  uint8_t kernel[0x4000]={0},segment[0x4000]={0};
  struct shmem_result memory[2]={{(uint64_t)(uintptr_t)segment,sizeof segment,8},{(uint64_t)(uintptr_t)kernel,sizeof kernel,9}};
  struct workload_queue q;assert(!workload_queue_init(&q,NULL,NULL,NULL,memory));q.callback_timing=1;
  struct workload_completion before_return,asynchronous,second,disabled;
  mode=0;assert(!workload_queue_fire(&q,&before_return));check(&before_return);
  assert(before_return.completed_callback_entry_ns<before_return.submit_return_ns);
  assert(before_return.wait_return_ns-before_return.completed_callback_entry_ns>=1000000);
  uint64_t trace;memcpy(&trace,segment,8);assert(trace==before_return.first_trace_id);
  mode=1;assert(!workload_queue_fire(&q,&asynchronous));check(&asynchronous);
  assert(asynchronous.first_trace_id==before_return.second_trace_id+1);
  assert(asynchronous.completed_callback_entry_ns>before_return.wait_return_ns);
  assert(!workload_queue_fire(&q,&second));check(&second);
  assert(second.scheduled_callback_entry_ns>asynchronous.wait_return_ns);
  assert(second.completed_callback_entry_ns>asynchronous.wait_return_ns);
  q.callback_timing=0;assert(!workload_queue_fire(&q,&disabled));
  assert(disabled.scheduled==1 && disabled.completed==1 && disabled.submit_ns && disabled.completion_ns);
  assert(!disabled.submit_start_ns && !disabled.submit_return_ns && !disabled.scheduled_callback_entry_ns);
  assert(!disabled.completed_callback_entry_ns && !disabled.wait_return_ns && !disabled.first_trace_id);
  dispatch_release(q.done);
  // A stale/duplicate delivery must not become a valid diagnostic sample.
  assert(!workload_queue_init(&q,NULL,NULL,NULL,memory));q.callback_timing=1;mode=2;
  assert(workload_queue_fire(&q,&second)==-1);assert(second.completed==2);
  dispatch_release(q.done);
  puts("CPU callback timing: synchronous, asynchronous, fresh timestamps, disabled path and duplicate refusal passed");
  return 0;
}
'''


class CallbackTiming(unittest.TestCase):
    def test_real_queue_helper_preserves_completion_and_timestamp_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);src=p/'callbacks.c';src.write_text(HOST);binary=p/'callbacks'
            built=subprocess.run(['clang','-std=c11','-O1','-fblocks','-Wall','-Wextra','-Werror',
                                  '-Wno-unused-function','-I'+str(ROOT/'spike/agxsub'),str(src),'-o',str(binary)],
                                 capture_output=True,text=True,timeout=30)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertIn('duplicate refusal passed',run.stdout)


if __name__=='__main__':unittest.main()
