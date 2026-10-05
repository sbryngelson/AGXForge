// Fixed seven-stage resident MiniLM FFN schedule. Included after workload_tile.h.
// This admits only the measured 3 MiB application graph, not arbitrary launches.
struct workload_ffn_stage {
  uint32_t codeoffset,codebytes,state,gx,gy,gz,threads,scratch;
  uint32_t bindcount,outalloc,outoff,outbytes,reserved0,reserved1;
  uint64_t bindings[4];
};
struct workload_ffn_region { uint32_t allocation,offset,bytes,readonly; };
_Static_assert(sizeof(struct workload_ffn_stage)==88,"FFN schedule stage wire size");
_Static_assert(sizeof(struct workload_ffn_region)==16,"FFN schedule region wire size");
static int workload_ffn_bounds(unsigned allocation,uint64_t offset,uint64_t bytes,
                               uint8_t* const mapped[30],const uint64_t sizes[30]){
  return allocation<30 && mapped[allocation] && offset<=sizes[allocation] &&
         bytes<=sizes[allocation]-offset;
}
static int workload_ffn_guards(const struct workload_ffn_region regions[12],
                               uint8_t* const mapped[30]){
  for(unsigned r=0;r<12;r++){
    const struct workload_ffn_region* p=&regions[r];
    const uint8_t* before=mapped[p->allocation]+p->offset-128;
    const uint8_t* after=mapped[p->allocation]+p->offset+p->bytes;
    for(unsigned j=0;j<128;j++)if(before[j]!=0xa5 || after[j]!=0xa5)return 0;
  }
  return 1;
}
static int g17_workload_ffn(void* io,void* dev,void* queue,
                            uint8_t* const mapped[30],const uint64_t sizes[30],
                            const struct shmem_result shmem[2]){
#ifndef G17_BLOCK_PREFLIGHT
  (void)io;(void)dev;(void)queue;(void)mapped;(void)sizes;(void)shmem;
  return 2;
#else
  if(!workload_ffn_bounds(0,0,0x10000,mapped,sizes) || sizes[0]!=0x10000 ||
     !workload_ffn_bounds(19,0,0x1c000,mapped,sizes) || sizes[19]!=0x1c000 ||
     !workload_ffn_bounds(20,0,0x300000,mapped,sizes) || sizes[20]!=0x300000 ||
     !workload_ffn_bounds(21,0,0x64000,mapped,sizes) || sizes[21]!=0x64000 ||
     !workload_ffn_bounds(22,0,0xb8,mapped,sizes) ||
     !workload_ffn_bounds(23,0,0x4a,mapped,sizes) ||
     !workload_ffn_bounds(25,0,0x20,mapped,sizes) ||
     !workload_ffn_bounds(28,0x1ba0,32,mapped,sizes) || !no_agx_metal())return 2;
  uint16_t shader_resource,launch_resource;
  memcpy(&shader_resource,mapped[23]+6,2);memcpy(&launch_resource,mapped[25]+9,2);
  if(shader_resource!=0x238 || launch_resource!=0x108)return 2;
  const char* path=getenv("ORDERED_WORKLOAD_FFN_SCHEDULE");
  if(!path)return 2;
  FILE* schedule=fopen(path,"rb");if(!schedule)return 2;
  uint32_t header[3];struct workload_ffn_stage stages[7];
  struct workload_ffn_region regions[12];
  int valid=fread(header,1,sizeof header,schedule)==sizeof header &&
    fread(stages,1,sizeof stages,schedule)==sizeof stages &&
    fread(regions,1,sizeof regions,schedule)==sizeof regions &&
    fgetc(schedule)==EOF && !ferror(schedule);
  fclose(schedule);
  if(!valid || header[0]!=0x47574631u || header[1]!=7 || header[2]!=12)return 2;
  static const uint32_t gx[7]={12288,1536,1536,49152,384,384,32};
  static const uint32_t gy[7]={1,1,32,1,1,32,32};
  static const uint32_t counts[7]={2,3,2,2,3,3,3};
  static const uint32_t lengths[7]={24576,196608,196608,98304,49152,49152,49152};
  unsigned source_region=12;
  for(unsigned r=0;r<12;r++){
    const struct workload_ffn_region* p=&regions[r];
    if((p->allocation!=19 && p->allocation!=20 && p->allocation!=21) ||
       p->readonly>1 || !p->bytes || p->offset<128 ||
       !workload_ffn_bounds(p->allocation,p->offset-128,(uint64_t)p->bytes+256,mapped,sizes))return 2;
    if(p->allocation==20 && !p->readonly)return 2;
    if(p->allocation==19 && p->readonly && p->bytes==49152){
      if(source_region!=12)return 2;source_region=r;
    }
    for(unsigned previous=0;previous<r;previous++){
      const struct workload_ffn_region* q=&regions[previous];
      // Residual updates contraction output in place; norm_input names that
      // exact same writable region. Partial overlap remains unsupported.
      if(p->allocation==21 && q->allocation==21 && p->offset==q->offset &&
         p->bytes==49152 && q->bytes==49152 && !p->readonly && !q->readonly)continue;
      if(p->allocation==q->allocation &&
         (uint64_t)p->offset-128<(uint64_t)q->offset+q->bytes+128 &&
         (uint64_t)q->offset-128<(uint64_t)p->offset+p->bytes+128)return 2;
    }
  }
  if(source_region==12 || !workload_ffn_guards(regions,mapped))return 2;
  for(unsigned s=0;s<7;s++){
    const struct workload_ffn_stage* p=&stages[s];
    uint32_t state=(s==0 || s==2 || s==3 || s==5)?0x0e40u:0x0e58u;
    uint32_t scratch=state==0x0e40u?0x0c000007u:0x0c00100fu;
    if(p->gx!=gx[s] || p->gy!=gy[s] || p->gz!=1 || p->threads!=32 ||
       p->state!=state || p->scratch!=scratch || p->bindcount!=counts[s] ||
       p->outbytes!=lengths[s] || p->reserved0 || p->reserved1 ||
       p->codeoffset<0x6c0 || p->codeoffset%64 || !p->codebytes ||
       !workload_ffn_bounds(0,p->codeoffset,p->codebytes,mapped,sizes) ||
       p->outalloc==20 || !workload_ffn_bounds(p->outalloc,p->outoff,p->outbytes,mapped,sizes))return 2;
    int output_found=0;
    for(unsigned r=0;r<12;r++)if(regions[r].allocation==p->outalloc &&
       regions[r].offset==p->outoff && regions[r].bytes==p->outbytes &&
       !regions[r].readonly)output_found=1;
    if(!output_found)return 2;
    for(unsigned b=0;b<4;b++)if(b<p->bindcount){
      uint64_t address=p->bindings[b];
      int resident=(address>=0x10000080000ull && address<0x10000080000ull+sizes[19]) ||
        (address>=0x100000a0000ull && address<0x100000a0000ull+sizes[20]) ||
        (address>=0x100003a8000ull && address<0x100003a8000ull+sizes[21]);
      if(!resident || address%2)return 2;
    }else if(p->bindings[b])return 2;
    for(unsigned prior=0;prior<s;prior++)if(
      (uint64_t)p->codeoffset<(uint64_t)stages[prior].codeoffset+stages[prior].codebytes &&
      (uint64_t)stages[prior].codeoffset<(uint64_t)p->codeoffset+p->codebytes)return 2;
  }
  uint8_t* code=malloc(0x10000);uint8_t* weights=malloc(0x300000);
  uint8_t* source=malloc(49152);
  if(!code || !weights || !source){free(code);free(weights);free(source);return 2;}
  memcpy(code,mapped[0],0x10000);memcpy(weights,mapped[20],0x300000);
  struct workload_queue submissions;
  if(workload_queue_init(&submissions,io,dev,queue,shmem)){
    free(code);free(weights);free(source);return 2;
  }
  const char* timing_option=getenv("ORDERED_WORKLOAD_FFN_CALLBACK_TIMING");
  if(timing_option && strcmp(timing_option,"1")){
    dispatch_release(submissions.done);free(code);free(weights);free(source);return 2;
  }
  submissions.callback_timing=timing_option!=NULL;
  struct {unsigned request,stage;struct workload_completion timing;} callback_records[112];
  unsigned callback_record_count=0;
  int result=2;unsigned requests=0;
  const uint32_t ready[4]={0x47575231u,3,49152,7};
  if(workload_write(ready,sizeof ready))goto finished;
  while(1){
    uint32_t request[4];if(workload_read(request,sizeof request))goto finished;
    if(request[0]!=0x47575231u || request[3])goto finished;
    if(!request[1] && !request[2]){result=0;break;}
    if((request[1]!=1 && request[1]!=2) || request[2]!=49152 || ++requests>16)goto finished;
    for(unsigned r=0;r<12;r++)if(!regions[r].readonly)
      memset(mapped[regions[r].allocation]+regions[r].offset,0xff,regions[r].bytes);
    uint8_t* input=mapped[19]+regions[source_region].offset;
    if(workload_read(input,49152))goto finished;
    memcpy(source,input,49152);
    for(unsigned s=0;s<7;s++){
      const struct workload_ffn_stage* p=&stages[s];
      uint64_t pc=0x10000000000ull+p->codeoffset;
      uint16_t low=(uint16_t)pc|7u,mid=(uint16_t)(pc>>16),high=(uint16_t)(pc>>32);
      uint16_t state=(uint16_t)p->state;
      memcpy(mapped[23]+0x40,&low,2);memcpy(mapped[23]+0x42,&state,2);
      memcpy(mapped[23]+0x46,&mid,2);memcpy(mapped[23]+0x48,&high,2);
      uint32_t scratch[2]={p->scratch,0};memcpy(mapped[23]+0x38,scratch,sizeof scratch);
      uint32_t geometry[3]={p->gx,p->gy,p->gz};
      memcpy(mapped[22]+0xa8,geometry,sizeof geometry);
      memcpy(mapped[25]+0x10,geometry,sizeof geometry);
      memcpy(mapped[25]+0x1c,&p->threads,4);
      memcpy(mapped[28]+0x1ba0,p->bindings,sizeof p->bindings);
      struct workload_completion completion;
      if(workload_queue_fire(&submissions,&completion))goto finished;
      if(submissions.callback_timing){
        if(callback_record_count>=112)goto finished;
        callback_records[callback_record_count].request=requests;
        callback_records[callback_record_count].stage=s;
        callback_records[callback_record_count++].timing=completion;
      }
      uint32_t flags=1;
      if(workload_ffn_guards(regions,mapped))flags|=4;
      if(no_agx_metal())flags|=16;
      int thorough=request[1]==1 || s==6;
      if(thorough){
        if(!memcmp(weights,mapped[20],0x300000))flags|=2;
        if(!memcmp(code,mapped[0],0x10000))flags|=8;
        if(!memcmp(source,input,49152))flags|=32;
      }
      uint32_t bytes=request[1]==1 || s==6?p->outbytes:0;
      uint32_t reply[8]={0x47575231u,(uint32_t)completion.status,completion.outword,
        flags,completion.scheduled,completion.completed,bytes,requests*8+s};
      uint64_t timing[2]={completion.submit_ns,completion.completion_ns};
      if(workload_write(reply,sizeof reply) || workload_write(timing,sizeof timing) ||
         (bytes && workload_write(mapped[p->outalloc]+p->outoff,bytes)))goto finished;
      if(flags!=(thorough?63u:21u)){
        fprintf(stderr,"WORKLOAD FFN REFUSED: request=%u stage=%u preservation flags=%u.\n",requests,s,flags);
        goto finished;
      }
    }
  }
finished:
  // No logging inside the request's timed path. The unchanged stdout reply is
  // already delivered; diagnostics are emitted at close with explicit units.
  for(unsigned r=0;r<callback_record_count;r++){
    const struct workload_completion* t=&callback_records[r].timing;
    fprintf(stderr,"WORKLOAD FFN CALLBACK_TIMING {\"schema\":1,\"unit\":\"ns\",\"clock\":\"CLOCK_MONOTONIC\","
      "\"request\":%u,\"stage\":%u,\"first_trace_id\":%llu,\"second_trace_id\":%llu,"
      "\"submit_start_ns\":%llu,\"submit_return_ns\":%llu,\"scheduled_callback_entry_ns\":%llu,"
      "\"completed_callback_entry_ns\":%llu,\"wait_return_ns\":%llu}\n",
      callback_records[r].request,callback_records[r].stage,
      (unsigned long long)t->first_trace_id,(unsigned long long)t->second_trace_id,
      (unsigned long long)t->submit_start_ns,(unsigned long long)t->submit_return_ns,
      (unsigned long long)t->scheduled_callback_entry_ns,(unsigned long long)t->completed_callback_entry_ns,
      (unsigned long long)t->wait_return_ns);
  }
  fprintf(stderr,"WORKLOAD FFN CLOSED: requests=%u Submits=%u result=%d; benchmark stages 0..5 omit code/weight/source scans.\n",
          requests,submissions.calls,result);
  dispatch_release(submissions.done);free(code);free(weights);free(source);
  return result;
#endif
}
