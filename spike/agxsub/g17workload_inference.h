// Shared model-stage schedule in the exact recovered 128 MiB publication graph.
// New launches are validation candidates; graph recovery alone admits no model.
struct inference_region {uint32_t allocation,offset,bytes,capacity,flags,birth,last,slot;};
_Static_assert(sizeof(struct inference_region)==32,"inference region wire size");
static int inference_guard_a5(const uint8_t* p,unsigned bytes){
  // memcpy permits unaligned padding starts and avoids reading beyond the
  // region. Check the identical bytes as the scalar loop, including tails.
  const uint64_t expected=0xa5a5a5a5a5a5a5a5ull;
  while(bytes>=32){
    uint64_t words[4];memcpy(words,p,sizeof words);
    if((words[0]^expected)|(words[1]^expected)|(words[2]^expected)|(words[3]^expected))return 0;
    p+=32;bytes-=32;
  }
  while(bytes--)if(*p++!=0xa5)return 0;
  return 1;
}
static int inference_guards(const struct inference_region* r,unsigned count,unsigned step,
                            uint8_t* const mapped[30]){
  for(unsigned i=0;i<count;i++)if(r[i].birth<=step && step<=r[i].last){
    const uint8_t* p=mapped[r[i].allocation]+r[i].offset;
    if(!inference_guard_a5(p-256,256) || !inference_guard_a5(p+r[i].capacity,256) ||
       !inference_guard_a5(p+r[i].bytes,r[i].capacity-r[i].bytes))return 0;
  }
  return 1;
}
static int g17_workload_inference(void* io,void* dev,void* queue,
                                  uint8_t* const mapped[30],const uint64_t sizes[30],
                                  const struct shmem_result shmem[2]){
#ifndef G17_BLOCK_PREFLIGHT
  (void)io;(void)dev;(void)queue;(void)mapped;(void)sizes;(void)shmem;return 2;
#else
  if(sizes[0]!=0x10000 || sizes[20]!=0x8000000 || sizes[17]!=0x20000 ||
     !workload_ffn_bounds(17,0,0xc000,mapped,sizes) ||
     !workload_ffn_bounds(22,0,0xb4,mapped,sizes) ||
     !workload_ffn_bounds(23,0,0x4a,mapped,sizes) ||
     !workload_ffn_bounds(25,0,0x20,mapped,sizes) || !no_agx_metal())return 2;
  uint16_t metadata,record;memcpy(&metadata,mapped[23]+6,2);memcpy(&record,mapped[25]+9,2);
  if(metadata!=0x2c || record!=0x2048)return 2;
  const char* path=getenv("ORDERED_INFERENCE_SCHEDULE");if(!path)return 2;
  FILE* f=fopen(path,"rb");if(!f)return 2;
  uint32_t h[4];struct workload_ffn_stage stage[256];struct inference_region region[512];
  int good=fread(h,1,sizeof h,f)==sizeof h && h[0]==0x47494e31 && h[1]>0 && h[1]<=256 &&
    h[2]>0 && h[2]<=512 && h[3]==512;
  if(good)good=fread(stage,sizeof stage[0],h[1],f)==h[1] &&
    fread(region,sizeof region[0],h[2],f)==h[2] && fgetc(f)==EOF && !ferror(f);
  fclose(f);if(!good)return 2;
  // Exact compiler-prepared encoder storage extent; decoder state is separate.
  const unsigned parameter_bytes=69072896,activity_bytes=642560;
  if(!workload_ffn_bounds(20,parameter_bytes,activity_bytes,mapped,sizes))return 2;
  unsigned inputs[4]={512,512,512,512};
  for(unsigned i=0;i<h[2];i++){
    const struct inference_region* r=&region[i];
    if((r->allocation!=19 && r->allocation!=20) || !r->bytes || r->bytes>r->capacity ||
       r->offset<256 || r->offset%256 || r->capacity%256 || r->flags>3 ||
       r->birth>r->last || r->last>h[1] ||
       !workload_ffn_bounds(r->allocation,r->offset-256,(uint64_t)r->capacity+512,mapped,sizes))return 2;
    if(r->flags&2){
      if(r->flags!=3 || r->allocation!=19 || r->bytes!=128 || r->slot<1 || r->slot>4 || inputs[r->slot-1]!=512)return 2;
      inputs[r->slot-1]=i;
    }else{
      if(r->slot || r->allocation!=20)return 2;
      if(r->flags&1){if((uint64_t)r->offset+r->capacity+256>parameter_bytes)return 2;}
      else if(r->offset-256<parameter_bytes || (uint64_t)r->offset+r->capacity+256>parameter_bytes+activity_bytes)return 2;
    }
    for(unsigned j=0;j<i;j++){
      const struct inference_region* p=&region[j];
      if(r->allocation==p->allocation && r->birth<=p->last && p->birth<=r->last &&
         (uint64_t)r->offset-256<(uint64_t)p->offset+p->capacity+256 &&
         (uint64_t)p->offset-256<(uint64_t)r->offset+r->capacity+256)return 2;
    }
  }
  for(unsigned i=0;i<4;i++)if(inputs[i]==512)return 2;
  for(unsigned s=0;s<h[1];s++){
    const struct workload_ffn_stage* p=&stage[s];
    if(p->bindcount<1 || p->bindcount>3 || p->threads!=32 || !p->gx || !p->gy || p->gz!=1 ||
       p->gx%32 || p->codeoffset<0x6c0 || p->codeoffset%64 || !p->codebytes ||
       !workload_ffn_bounds(0,p->codeoffset,p->codebytes,mapped,sizes) ||
       (p->state!=0x0e40 && p->state!=0x0e58) ||
       p->scratch!=(p->state==0x0e40?0x0c000007u:0x0c00100fu) || p->reserved0 || p->reserved1)return 2;
    int found=0;
    for(unsigned i=0;i<h[2];i++)if(region[i].allocation==p->outalloc && region[i].offset==p->outoff &&
      region[i].bytes==p->outbytes && !region[i].flags && region[i].birth<=s && s<=region[i].last)found=1;
    if(!found)return 2;
    for(unsigned b=0;b<4;b++){
      if(b>=p->bindcount){if(p->bindings[b])return 2;continue;}
      int resident=0;
      for(unsigned i=0;i<h[2];i++){
        uint64_t base=region[i].allocation==19?0x10000080000ull:0x100000a0000ull;
        if(p->bindings[b]==base+region[i].offset && region[i].birth<=s && s<=region[i].last)resident=1;
      }
      if(!resident)return 2;
    }
  }
  uint8_t* readonly=malloc(parameter_bytes);uint8_t* code=malloc(0x10000);
  if(!readonly || !code){free(readonly);free(code);return 2;}
  memcpy(readonly,mapped[20],parameter_bytes);memcpy(code,mapped[0],0x10000);
  struct workload_queue q;if(workload_queue_init(&q,io,dev,queue,shmem)){free(readonly);free(code);return 2;}
  unsigned requests=0;int result=2;uint32_t ready[4]={0x47575231,4,512,h[1]};
  if(workload_write(ready,sizeof ready))goto done;
  while(1){
    uint32_t request[4];if(workload_read(request,sizeof request))goto done;
    if(request[0]!=0x47575231)goto done;
    if(!request[1] && !request[2] && !request[3]){result=0;break;}
    if((request[1]!=1 && request[1]!=2) || request[2]!=512 || !request[3] || request[3]>h[1] || ++requests>16)goto done;
    uint32_t input_words[128];uint8_t* input=(uint8_t*)input_words;
    if(workload_read(input,512))goto done;
    const uint32_t* words=input_words;unsigned valid_tokens=0;
    for(unsigned i=0;i<32;i++){
      if(words[i]>=30522 || words[32+i]>=512 || words[64+i]>=2 || words[96+i]>1)goto done;
      valid_tokens+=words[96+i];
    }
    if(!valid_tokens)goto done;
    for(unsigned i=0;i<4;i++)memcpy(mapped[19]+region[inputs[i]].offset,input+i*128,128);
    memset(mapped[20]+parameter_bytes,0xa5,activity_bytes);
    for(unsigned s=0;s<request[3];s++){
      for(unsigned i=0;i<h[2];i++)if(!region[i].flags && region[i].birth==s){
        uint8_t* dst=mapped[region[i].allocation]+region[i].offset;
        memset(dst,0xff,region[i].bytes);memset(dst+region[i].bytes,0xa5,region[i].capacity-region[i].bytes);
      }
      const struct workload_ffn_stage* p=&stage[s];uint64_t pc=0x10000000000ull+p->codeoffset;
      uint16_t low=(uint16_t)pc|7,mid=(uint16_t)(pc>>16),high=(uint16_t)(pc>>32),state=(uint16_t)p->state;
      memcpy(mapped[23]+0x40,&low,2);memcpy(mapped[23]+0x42,&state,2);
      memcpy(mapped[23]+0x46,&mid,2);memcpy(mapped[23]+0x48,&high,2);
      uint32_t scratch[2]={p->scratch,0};memcpy(mapped[23]+0x38,scratch,8);
      uint32_t geometry[3]={p->gx,p->gy,p->gz};memcpy(mapped[22]+0xa8,geometry,12);
      memcpy(mapped[25]+0x10,geometry,12);memcpy(mapped[25]+0x1c,&p->threads,4);
      // This graph selects the complete low binding copy, not high allocation 28.
      memcpy(mapped[17]+0x1ba0,p->bindings,32);memcpy(mapped[28]+0x1ba0,p->bindings,32);
      struct workload_completion completion;if(workload_queue_fire(&q,&completion))goto done;
      unsigned flags=1;
      if(inference_guards(region,h[2],s,mapped))flags|=4;
      if(no_agx_metal())flags|=16;
      int thorough=s+1==request[3];
      if(thorough){
        if(!memcmp(readonly,mapped[20],parameter_bytes))flags|=2;
        if(!memcmp(code,mapped[0],0x10000))flags|=8;
        int same=1;for(unsigned i=0;i<4;i++)if(memcmp(mapped[19]+region[inputs[i]].offset,input+i*128,128))same=0;
        if(same)flags|=32;
      }
      uint32_t bytes=request[1]==1 || thorough?p->outbytes:0;
      uint32_t reply[8]={0x47575231,(uint32_t)completion.status,completion.outword,flags,
        completion.scheduled,completion.completed,bytes,requests*1024+s};
      uint64_t timing[2]={completion.submit_ns,completion.completion_ns};
      if(workload_write(reply,sizeof reply) || workload_write(timing,sizeof timing) ||
         (bytes && workload_write(mapped[p->outalloc]+p->outoff,bytes)))goto done;
      if(flags!=(thorough?63u:21u))goto done;
    }
  }
done:
  fprintf(stderr,"INFERENCE CLOSED: requests=%u Submits=%u result=%d.\n",requests,q.calls,result);
  dispatch_release(q.done);free(readonly);free(code);return result;
#endif
}
