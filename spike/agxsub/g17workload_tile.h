// Persistent execution control for the measured 32x32x64 TensorOps tile.
// Included by the existing graph initializer; this is not a general IOGPU ABI.
#include <time.h>
#include <stdatomic.h>
static uint64_t workload_ns(void){
  struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
  return (uint64_t)t.tv_sec*1000000000ull+(uint64_t)t.tv_nsec;
}
static int workload_read(void* dst,size_t length){
  return fread(dst,1,length,stdin)==length?0:-1;
}
static int workload_write(const void* src,size_t length){
  return fwrite(src,1,length,stdout)==length && fflush(stdout)==0?0:-1;
}
#ifdef G17_BLOCK_PREFLIGHT
// The submission/completion mechanism is shared by application stages. It does
// not choose a resource graph, program, bindings, scratch or launch dimensions:
// those remain the responsibility of a measured workload schedule.
struct workload_queue {
  void *dev,*queue;
  const struct shmem_result *shmem;
  int (*submit)(void*,void*,unsigned,void*,unsigned,void*);
  uint64_t (*next_id)(void*);
  dispatch_semaphore_t done;
  volatile uint32_t scheduled,completed;
  unsigned calls;
  // Opt-in diagnostic only. FFN owns enablement; other workloads retain their
  // stdout protocols and execute without the callback clock calls.
  int callback_timing;
  _Atomic uint64_t scheduled_entry_ns,completed_entry_ns;
};
struct workload_completion {
  int status;
  uint32_t outword,scheduled,completed;
  uint64_t submit_ns,completion_ns;
  uint64_t submit_start_ns,submit_return_ns,scheduled_callback_entry_ns;
  uint64_t completed_callback_entry_ns,wait_return_ns,first_trace_id,second_trace_id;
};
static int workload_queue_init(struct workload_queue* q,void* io,void* dev,
                               void* queue,const struct shmem_result shmem[2]){
  memset(q,0,sizeof *q);
  atomic_init(&q->scheduled_entry_ns,0);atomic_init(&q->completed_entry_ns,0);
  q->dev=dev;q->queue=queue;q->shmem=shmem;
  q->submit=dlsym(io,"IOGPUCommandQueueSubmitCommandBuffers");
  q->next_id=dlsym(io,"IOGPUDeviceGetNextGlobalTraceID");
  if(!q->submit || !q->next_id || shmem[0].size!=0x4000 ||
     shmem[1].size!=0x4000)return -1;
  q->done=dispatch_semaphore_create(0);
  return q->done?0:-1;
}
static int workload_queue_fire(struct workload_queue* q,struct workload_completion* result){
  memset(result,0,sizeof *result);
  uint64_t first=q->next_id(q->dev),second=q->next_id(q->dev);
  if(!first || second!=first+1)return -1;
  const struct shmem_result* shmem=q->shmem;
  memcpy((void*)(uintptr_t)(shmem[1].cpu+0x234),&second,4);
  memcpy((void*)(uintptr_t)shmem[0].cpu,&first,8);
  memcpy((void*)(uintptr_t)(shmem[0].cpu+0x18),&first,8);
  memcpy((void*)(uintptr_t)(shmem[0].cpu+0x28),&second,8);
  *(volatile uint32_t*)(uintptr_t)(shmem[0].cpu+0x24)=0x800000f0u;
  // Fresh completion blocks follow the measured multi-Submit ownership path.
  // Do not release or reuse them based on an inferred private-framework ABI.
  if(q->callback_timing){
    atomic_store_explicit(&q->scheduled_entry_ns,0,memory_order_release);
    atomic_store_explicit(&q->completed_entry_ns,0,memory_order_release);
  }
  void (^scheduled)(void)=Block_copy(^{
    if(q->callback_timing) atomic_store_explicit(&q->scheduled_entry_ns,workload_ns(),memory_order_release);
    q->scheduled++;
  });
  void (^completed)(void)=Block_copy(^{
    if(q->callback_timing) atomic_store_explicit(&q->completed_entry_ns,workload_ns(),memory_order_release);
    q->completed++;dispatch_semaphore_signal(q->done);
  });
  if(!scheduled || !completed)return -1;
  uint8_t record[64]={0},out[64]={0};
  uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
  memcpy(record,&shmem[1].id,4);memcpy(record+4,&shmem[0].id,4);
  memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
  uint32_t before_s=q->scheduled,before_c=q->completed;
  uint64_t started=workload_ns();q->calls++;
  result->status=q->submit(q->queue,NULL,1,record,64,out);
  uint64_t submitted=workload_ns();
  if(result->status || dispatch_semaphore_wait(q->done,dispatch_time(DISPATCH_TIME_NOW,5ll*NSEC_PER_SEC))){
    fprintf(stderr,"WORKLOAD REFUSED: status=%d scheduled=%u completed=%u after Submit %u.\n",
            result->status,q->scheduled-before_s,q->completed-before_c,q->calls);
    return -1;
  }
  uint64_t wait_returned=workload_ns();
  result->submit_ns=submitted-started;result->completion_ns=wait_returned-started;
  if(q->callback_timing){
    uint64_t scheduled_entry=atomic_load_explicit(&q->scheduled_entry_ns,memory_order_acquire);
    uint64_t completed_entry=atomic_load_explicit(&q->completed_entry_ns,memory_order_acquire);
    // Completion can occur before Submit returns. There is deliberately no
    // completed_entry >= submitted or scheduled_entry <= completed_entry rule.
    if(scheduled_entry<started || scheduled_entry>wait_returned ||
       completed_entry<started || completed_entry>wait_returned ||
       submitted<started || submitted>wait_returned){
      fprintf(stderr,"WORKLOAD REFUSED: invalid callback timing after Submit %u.\n",q->calls);
      return -1;
    }
    result->submit_start_ns=started;result->submit_return_ns=submitted;
    result->scheduled_callback_entry_ns=scheduled_entry;
    result->completed_callback_entry_ns=completed_entry;result->wait_return_ns=wait_returned;
    result->first_trace_id=first;result->second_trace_id=second;
  }
  memcpy(&result->outword,out,4);
  result->scheduled=q->scheduled-before_s;result->completed=q->completed-before_c;
  return result->outword || result->scheduled!=1 || result->completed!=1?-1:0;
}
#endif
// Zero-Submit preparation of the measured 3 MiB application graph. Its FFN
// stage launches are not admitted here until the new schedule is validated.
static int g17_workload_ffn_prepare(uint8_t* const mapped[30],const uint64_t sizes[30]){
  if(!mapped[0] || !mapped[19] || !mapped[20] || !mapped[21] ||
     !mapped[23] || !mapped[25] || !mapped[28] || sizes[0]!=0x10000 ||
     sizes[19]!=0x1c000 || sizes[20]!=0x300000 || sizes[21]!=0x64000 ||
     !no_agx_metal())return 2;
  uint16_t shader_resource,launch_resource;
  memcpy(&shader_resource,mapped[23]+6,2);memcpy(&launch_resource,mapped[25]+9,2);
  if(shader_resource!=0x238 || launch_resource!=0x108)return 2;
  const uint32_t ready[4]={0x47575231u,2,49152,0};
  if(workload_write(ready,sizeof ready))return 2;
  uint32_t request[4];
  if(workload_read(request,sizeof request) || request[0]!=0x47575231u ||
     request[1] || request[2] || request[3])return 2;
  fprintf(stderr,"WORKLOAD FFN PREPARED: measured 3 MiB graph, resident programs and weights, zero Submits; execution not admitted.\n");
  return 0;
}
static int g17_workload_tile(void* io,void* dev,void* queue,
                             uint8_t* const mapped[30],const uint64_t sizes[30],
                             const struct shmem_result shmem[2]){
#ifndef G17_BLOCK_PREFLIGHT
  (void)io;(void)dev;(void)queue;(void)mapped;(void)sizes;(void)shmem;
  return 2;
#else
  if(!mapped[0] || !mapped[2] || !mapped[23] ||
     !mapped[28] || sizes[0]!=0x10000 || sizes[2]!=0x20000 ||
     shmem[0].size!=0x4000 || shmem[1].size!=0x4000 || !no_agx_metal())return 2;
  uint32_t packet;memcpy(&packet,mapped[23]+0x40,4);
  uint64_t bindings[3];memcpy(bindings,mapped[28]+0x1ba0,sizeof bindings);
  if(packet!=0x0e5806c7u || bindings[0]!=0x10000034d80ull ||
     bindings[1]!=0x10000035e80ull || bindings[2]!=0x10000036f80ull)return 2;
  uint8_t *a=mapped[2]+0x4d80,*b=mapped[2]+0x5e80,*c=mapped[2]+0x6f80;
  uint8_t readonly_b[4096],left[128],right[128],code[1276];
  memcpy(readonly_b,b,sizeof readonly_b);memcpy(left,c-128,sizeof left);
  memcpy(right,c+4096,sizeof right);memcpy(code,mapped[0]+0x6c0,sizeof code);
  // Bounded discriminatory entry experiment: both bodies remain resident.
  // The address halfwords follow LoadShader::emit's measured encoding; +0x42
  // is separate TensorOps instruction state and is never changed here.
  const char* alternate_path=getenv("ORDERED_WORKLOAD_TILE_ALTERNATE_CODE");
  uint8_t alternate[4096];size_t alternate_bytes=0;
  if(alternate_path){
    FILE* input=fopen(alternate_path,"rb");if(!input)return 2;
    alternate_bytes=fread(alternate,1,sizeof alternate,input);
    int good=alternate_bytes && alternate_bytes<sizeof alternate && fgetc(input)==EOF && !ferror(input);
    fclose(input);if(!good)return 2;
    memcpy(mapped[0]+0xe500,alternate,alternate_bytes);
  }
  // Reply precedes any Submit. A client can prepare and close without firing.
  const uint32_t ready[4]={0x47575231u,1,4096,0};
  if(workload_write(ready,sizeof ready))return 2;
  struct workload_queue submissions;
  if(workload_queue_init(&submissions,io,dev,queue,shmem))return 2;
  unsigned calls=0;
  while(1){
    uint32_t request[4];
    if(workload_read(request,sizeof request))return 2;
    if(request[0]!=0x47575231u || request[3]>1 || (request[3] && !alternate_bytes))return 2;
    if(request[1]==0 && request[2]==0)break;
    // 1 = validate/read final output, 2 = benchmark without output readback.
    if((request[1]!=1 && request[1]!=2) || request[2]!=4096 || ++calls>64)return 2;
    if(workload_read(a,4096))return 2;
    uint8_t readonly_a[4096];memcpy(readonly_a,a,sizeof readonly_a);
    for(unsigned i=0;i<1024;i++)((uint32_t*)c)[i]=0x7fc01234u;
    uint64_t pc=0x10000000000ull+(request[3]?0xe500u:0x6c0u);
    uint16_t low=(uint16_t)pc|7u,mid=(uint16_t)(pc>>16),high=(uint16_t)(pc>>32);
    memcpy(mapped[23]+0x40,&low,2);memcpy(mapped[23]+0x46,&mid,2);memcpy(mapped[23]+0x48,&high,2);
    struct workload_completion completion;
    if(workload_queue_fire(&submissions,&completion))return 2;
    uint32_t flags=1u;
    if(!memcmp(b,readonly_b,sizeof readonly_b))flags|=2u;
    if(!memcmp(c-128,left,sizeof left) && !memcmp(c+4096,right,sizeof right))flags|=4u;
    if(!memcmp(code,mapped[0]+0x6c0,sizeof code) &&
       (!alternate_bytes || !memcmp(alternate,mapped[0]+0xe500,alternate_bytes)))flags|=8u;
    if(no_agx_metal())flags|=16u;
    if(!memcmp(a,readonly_a,sizeof readonly_a))flags|=32u;
    uint32_t reply[8]={0x47575231u,(uint32_t)completion.status,completion.outword,flags,
                       completion.scheduled,completion.completed,
                       request[1]==1?4096u:0u,calls};
    uint64_t timing[2]={completion.submit_ns,completion.completion_ns};
    if(workload_write(reply,sizeof reply) || workload_write(timing,sizeof timing) ||
       (request[1]==1 && workload_write(c,4096)))return 2;
    if(flags!=63u)return 2;
  }
  dispatch_release(submissions.done);
  fprintf(stderr,"WORKLOAD CLOSED: %u Submits, one graph, immutable program and B.\n",calls);
  return 0;
#endif
}
