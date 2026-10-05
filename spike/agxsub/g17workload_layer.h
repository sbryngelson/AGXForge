// Fixed twenty-one-stage resident MiniLM attention followed by FFN schedule. Included after workload_tile.h.
// This admits only the measured 4 MiB application graph, not arbitrary launches.
struct workload_layer_stage {
  uint32_t codeoffset,codebytes,state,gx,gy,gz,threads,scratch;
  uint32_t bindcount,outalloc,outoff,outbytes,reserved0,reserved1;
  uint64_t bindings[4];
};
struct workload_layer_region { uint32_t allocation,offset,bytes,readonly; };
_Static_assert(sizeof(struct workload_layer_stage)==88,"LAYER schedule stage wire size");
_Static_assert(sizeof(struct workload_layer_region)==16,"LAYER schedule region wire size");
static int workload_layer_bounds(unsigned allocation,uint64_t offset,uint64_t bytes,
                               uint8_t* const mapped[30],const uint64_t sizes[30]){
  return allocation<30 && mapped[allocation] && offset<=sizes[allocation] &&
         bytes<=sizes[allocation]-offset;
}
static int workload_layer_guards(const struct workload_layer_region regions[],
                               uint8_t* const mapped[30],unsigned count){
  for(unsigned r=0;r<count;r++){
    const struct workload_layer_region* p=&regions[r];
    const uint8_t* before=mapped[p->allocation]+p->offset-128;
    const uint8_t* after=mapped[p->allocation]+p->offset+p->bytes;
    for(unsigned j=0;j<128;j++)if(before[j]!=0xa5 || after[j]!=0xa5)return 0;
  }
  return 1;
}
// Called only after both phase inventories and the fixed offsets are checked.
static void workload_layer_enter_ffn(uint8_t* const mapped[30],
                                    const struct workload_layer_region ffn[21]){
  memset(mapped[21],0xff,344960);
  for(unsigned r=18;r<21;r++){
    memset(mapped[21]+ffn[r].offset-128,0xa5,128);
    memset(mapped[21]+ffn[r].offset+ffn[r].bytes,0xa5,128);
  }
}
static int g17_workload_layer(void* io,void* dev,void* queue,
                            uint8_t* const mapped[30],const uint64_t sizes[30],
                            const struct shmem_result shmem[2]){
#ifndef G17_BLOCK_PREFLIGHT
  (void)io;(void)dev;(void)queue;(void)mapped;(void)sizes;(void)shmem;
  return 2;
#else
  if(!workload_layer_bounds(0,0,0x10000,mapped,sizes) || sizes[0]!=0x10000 ||
     !workload_layer_bounds(19,0,0x1c000,mapped,sizes) || sizes[19]!=0x1c000 ||
     !workload_layer_bounds(20,0,0x400000,mapped,sizes) || sizes[20]!=0x400000 ||
     !workload_layer_bounds(21,0,0x64000,mapped,sizes) || sizes[21]!=0x64000 ||
     !workload_layer_bounds(22,0,0xb8,mapped,sizes) ||
     !workload_layer_bounds(23,0,0x4a,mapped,sizes) ||
     !workload_layer_bounds(25,0,0x20,mapped,sizes) ||
     !workload_layer_bounds(28,0x1ba0,32,mapped,sizes) || !no_agx_metal())return 2;
  uint16_t shader_resource,launch_resource;
  memcpy(&shader_resource,mapped[23]+6,2);memcpy(&launch_resource,mapped[25]+9,2);
  if(shader_resource!=0x2b8 || launch_resource!=0x148)return 2;
  const char* path=getenv("ORDERED_WORKLOAD_LAYER_SCHEDULE");
  if(!path)return 2;
  FILE* schedule=fopen(path,"rb");if(!schedule)return 2;
  uint32_t header[4];struct workload_layer_stage stages[21];
  struct workload_layer_region attention[25],ffn[21];
  int valid=fread(header,1,sizeof header,schedule)==sizeof header &&
    fread(stages,1,sizeof stages,schedule)==sizeof stages &&
    fread(attention,1,sizeof attention,schedule)==sizeof attention &&
    fread(ffn,1,sizeof ffn,schedule)==sizeof ffn &&
    fgetc(schedule)==EOF && !ferror(schedule);
  fclose(schedule);
  if(!valid || header[0]!=0x47574c31u || header[1]!=21 || header[2]!=25 || header[3]!=21)return 2;
  static const uint32_t gx[21]={12288,384,384,384,384,384,384,32,384,384,12288,384,384,32,
                              12288,1536,1536,49152,384,384,32};
  static const uint32_t gy[21]={1,1,32,1,32,1,32,384,1,32,1,1,32,32,1,1,32,1,1,32,32};
  static const uint32_t counts[21]={2,3,2,3,2,3,2,3,2,3,2,3,3,3,2,3,2,2,3,3,3};
  static const uint32_t lengths[21]={24576,49152,49152,49152,49152,49152,49152,49152,
    49152,49152,24576,49152,49152,49152,24576,196608,196608,98304,49152,49152,49152};
  static const uint32_t parameter_bytes[14]={294912,294912,294912,294912,1536,1536,1536,1536,
                                          3072,1179648,1179648,6144,1536,3072};
  for(unsigned phase=0;phase<2;phase++){
    const struct workload_layer_region* regions=phase?ffn:attention;
    unsigned count=phase?21:25;
    for(unsigned r=0;r<count;r++){
      const struct workload_layer_region* p=&regions[r];
      unsigned alloc=r<3?19:r<17?20:21;
      unsigned length=r==0?49152:r<3?24576:r<17?parameter_bytes[r-3]:
        !phase?49152:r==17?49152:r==18?196608:r==19?98304:49152;
      unsigned readonly=r==0 || (r>=3 && r<17) || (phase && r==17);
      if(p->allocation!=alloc || p->bytes!=length || p->readonly!=readonly || p->offset<128 ||
         !workload_layer_bounds(p->allocation,p->offset-128,(uint64_t)p->bytes+256,mapped,sizes))return 2;
      for(unsigned prior=0;prior<r;prior++){
        const struct workload_layer_region* q=&regions[prior];
        if(p->allocation==q->allocation &&
           (uint64_t)p->offset-128<(uint64_t)q->offset+q->bytes+128 &&
           (uint64_t)q->offset-128<(uint64_t)p->offset+p->bytes+128)return 2;
      }
      if(phase && r<17 && memcmp(p,&attention[r],sizeof *p))return 2;
    }
  }
  if(attention[24].offset!=346112 || ffn[17].offset!=346112 ||
     ffn[18].offset!=256 || ffn[19].offset!=197120 || ffn[20].offset!=295680)return 2;
  if(!workload_layer_guards(attention,mapped,25))return 2;
  static const uint32_t bind_regions[21][3]={
    {0,1,0},{1,3,17},{17,7,0},{1,4,18},{18,8,0},{1,5,19},{19,9,0},
    {17,18,20},{20,21,0},{21,19,22},{22,2,0},{2,6,23},{23,10,0},{23,11,24},
    {17,1,0},{1,12,18},{18,14,0},{18,19,0},{19,13,20},{20,15,17},{20,16,20}};
  static const uint32_t output_regions[21]={1,17,17,18,18,19,19,20,21,22,2,23,23,24,
                                          1,18,18,19,20,20,20};
  for(unsigned s=0;s<21;s++){
    const struct workload_layer_stage* p=&stages[s];
    const struct workload_layer_region* regions=s<14?attention:ffn;
    uint32_t state=(s==1 || s==3 || s==5 || s==11 || s==13 || s==15 || s==18 || s==20)?0x0e58u:0x0e40u;
    uint32_t scratch=state==0x0e40u?0x0c000007u:0x0c00100fu;
    const struct workload_layer_region* out=&regions[output_regions[s]];
    if(p->gx!=gx[s] || p->gy!=gy[s] || p->gz!=1 || p->threads!=32 ||
       p->state!=state || p->scratch!=scratch || p->bindcount!=counts[s] ||
       p->outbytes!=lengths[s] || p->reserved0 || p->reserved1 ||
       p->codeoffset<0x6c0 || p->codeoffset%64 || !p->codebytes ||
       !workload_layer_bounds(0,p->codeoffset,p->codebytes,mapped,sizes) ||
       p->outalloc!=out->allocation || p->outoff!=out->offset || p->outbytes!=out->bytes || out->readonly)return 2;
    for(unsigned b=0;b<4;b++)if(b<p->bindcount){
      const struct workload_layer_region* binding=&regions[bind_regions[s][b]];
      uint64_t base=binding->allocation==19?0x10000080000ull:
                    binding->allocation==20?0x100000a0000ull:0x100004a8000ull;
      if(p->bindings[b]!=base+binding->offset)return 2;
    }else if(p->bindings[b])return 2;
    for(unsigned prior=0;prior<s;prior++)if(
      (uint64_t)p->codeoffset<(uint64_t)stages[prior].codeoffset+stages[prior].codebytes &&
      (uint64_t)stages[prior].codeoffset<(uint64_t)p->codeoffset+p->codebytes)return 2;
  }
  uint8_t* code=malloc(0x10000);uint8_t* weights=malloc(0x400000);
  uint8_t* source=malloc(49152);uint8_t* saved=malloc(49152);
  if(!code || !weights || !source || !saved){free(code);free(weights);free(source);free(saved);return 2;}
  memcpy(code,mapped[0],0x10000);memcpy(weights,mapped[20],0x400000);
  struct workload_queue submissions;
  if(workload_queue_init(&submissions,io,dev,queue,shmem)){
    free(code);free(weights);free(source);free(saved);return 2;
  }
  int result=2;unsigned requests=0;
  const uint32_t ready[4]={0x47575231u,5,49152,21};
  if(workload_write(ready,sizeof ready))goto finished;
  while(1){
    uint32_t request[4];if(workload_read(request,sizeof request))goto finished;
    if(request[0]!=0x47575231u || request[3])goto finished;
    if(!request[1] && !request[2]){result=0;break;}
    if((request[1]!=1 && request[1]!=2) || request[2]!=49152 || ++requests>16)goto finished;
    // Restore the attention lifetime/guards for every fresh request. The old
    // saved output is no longer live; its new value comes only from GPU stage13.
    memset(mapped[21],0xff,sizes[21]);
    for(unsigned r=0;r<25;r++){
      const struct workload_layer_region* p=&attention[r];
      if(!p->readonly)memset(mapped[p->allocation]+p->offset,0xff,p->bytes);
      memset(mapped[p->allocation]+p->offset-128,0xa5,128);
      memset(mapped[p->allocation]+p->offset+p->bytes,0xa5,128);
    }
    uint8_t* input=mapped[19]+attention[0].offset;
    if(workload_read(input,49152))goto finished;
    memcpy(source,input,49152);
    for(unsigned s=0;s<21;s++){
      const struct workload_layer_stage* p=&stages[s];
      if(s==14){
        // Never reset the saved attention payload or its two guards. Only the
        // dead attention prefix becomes FFN storage; no host intermediate upload.
        workload_layer_enter_ffn(mapped,ffn);
      }
      const struct workload_layer_region* regions=s<14?attention:ffn;
      unsigned region_count=s<14?25:21;
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
      if(s==13)memcpy(saved,mapped[21]+346112,49152);
      uint32_t flags=1;
      if(workload_layer_guards(regions,mapped,region_count) &&
         (s<14 || !memcmp(saved,mapped[21]+346112,49152)))flags|=4;
      if(no_agx_metal())flags|=16;
      int thorough=request[1]==1 || s==20;
      if(thorough){
        if(!memcmp(weights,mapped[20],0x400000))flags|=2;
        if(!memcmp(code,mapped[0],0x10000))flags|=8;
        if(!memcmp(source,input,49152) && (s<14 || !memcmp(saved,mapped[21]+346112,49152)))flags|=32;
      }
      uint32_t bytes=request[1]==1 || s==20?p->outbytes:0;
      uint32_t reply[8]={0x47575231u,(uint32_t)completion.status,completion.outword,
        flags,completion.scheduled,completion.completed,bytes,requests*32+s};
      uint64_t timing[2]={completion.submit_ns,completion.completion_ns};
      if(workload_write(reply,sizeof reply) || workload_write(timing,sizeof timing) ||
         (bytes && workload_write(mapped[p->outalloc]+p->outoff,bytes)))goto finished;
      if(flags!=(thorough?63u:21u)){
        fprintf(stderr,"WORKLOAD LAYER REFUSED: request=%u stage=%u preservation flags=%u.\n",requests,s,flags);
        goto finished;
      }
    }
  }
finished:
  fprintf(stderr,"WORKLOAD LAYER CLOSED: requests=%u Submits=%u result=%d; benchmark stages 0..19 omit code/weight/source scans.\n",
          requests,submissions.calls,result);
  dispatch_release(submissions.done);free(code);free(weights);free(source);free(saved);
  return result;
#endif
}
