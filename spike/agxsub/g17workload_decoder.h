// GIN2: two compiler-derived schedules in the previously measured 1 GiB graph.
// Host protocol fields are not GPU metadata. The selected shader record is the
// proven low mirror at allocation 17 + 0x10000, not allocation 23.
#include <sys/resource.h>
struct decoder_schedule {
  uint32_t h[10],inputs[4];
  struct workload_ffn_stage stages[1024];
  struct inference_region regions[1280];
};
static int decoder_schedule_load(const char* path,struct decoder_schedule* s,unsigned tokens,
                                 uint8_t* const mapped[30],const uint64_t sizes[30]){
  if(!path)return 0;
  FILE* f=fopen(path,"rb");if(!f)return 0;
  int good=fread(s->h,1,sizeof s->h,f)==sizeof s->h;
  const uint32_t* h=s->h;
  // These are compiler-derived host schedules over the same measured GPU
  // resource class. Pack reuse removes 72 producers and their temporary names;
  // Crop+bias fusion removes another 72 decode stages. GPU launch metadata
  // and resource extent remain the same, with one additional scalar body.
  int reused=(h[1]==(tokens==32?676u:844u) || (tokens==1 && h[1]==772u)) &&
    h[2]==(tokens==32?875u:1043u);
  int baseline=h[1]==(tokens==32?748u:916u) && h[2]==(tokens==32?947u:1115u);
  good=good && h[0]==0x47494e32 && (baseline || reused) && h[3]==tokens*8+1028 &&
    h[4]==988423168 && h[5]==h[4] && h[6]==(reused?22682112u:22780416u) &&
    h[7]==h[5]+h[6] && h[8]==6316032 && h[9]==tokens;
  if(good)good=fread(s->stages,sizeof s->stages[0],h[1],f)==h[1] &&
    fread(s->regions,sizeof s->regions[0],h[2],f)==h[2] && fgetc(f)==EOF && !ferror(f);
  fclose(f);if(!good || !workload_ffn_bounds(20,h[5],(uint64_t)h[6]+h[8],mapped,sizes))return 0;
  for(unsigned i=0;i<4;i++)s->inputs[i]=1280;
  unsigned states=0;
  for(unsigned i=0;i<h[2];i++){
    const struct inference_region* r=&s->regions[i];
    if((r->allocation!=19 && r->allocation!=20) || !r->bytes || r->bytes>r->capacity ||
       r->offset<256 || r->offset%256 || r->capacity%256 ||
       (r->flags!=0 && r->flags!=1 && r->flags!=3 && r->flags!=4) ||
       r->birth>r->last || r->last>h[1] ||
       !workload_ffn_bounds(r->allocation,r->offset-256,(uint64_t)r->capacity+512,mapped,sizes))return 0;
    uint64_t end=(uint64_t)r->offset+r->capacity+256;
    if(r->flags==3){
      if(r->allocation!=19 || r->slot<1 || r->slot>4 || s->inputs[r->slot-1]!=1280 || r->birth || r->last!=h[1])return 0;
      unsigned expected=r->slot<3?tokens*4:(r->slot==3?1024:4);
      if(r->bytes!=expected)return 0;
      s->inputs[r->slot-1]=i;
    }else{
      if(r->slot || r->allocation!=20)return 0;
      if(r->flags==1){if(end>h[4] || r->birth || r->last!=h[1])return 0;}
      else if(r->flags==4){
        if(r->offset-256<h[7] || end>(uint64_t)h[7]+h[8] || r->birth || r->last!=h[1] || r->bytes!=131072)return 0;
        states++;
      }else if(r->offset-256<h[5] || end>(uint64_t)h[5]+h[6])return 0;
    }
    for(unsigned j=0;j<i;j++){
      const struct inference_region* p=&s->regions[j];
      if(r->allocation==p->allocation && r->birth<=p->last && p->birth<=r->last &&
         (uint64_t)r->offset-256<(uint64_t)p->offset+p->capacity+256 &&
         (uint64_t)p->offset-256<(uint64_t)r->offset+r->capacity+256)return 0;
    }
  }
  if(states!=48)return 0;
  for(unsigned i=0;i<4;i++)if(s->inputs[i]==1280)return 0;
  for(unsigned step=0;step<h[1];step++){
    const struct workload_ffn_stage* p=&s->stages[step];
    if(p->bindcount<1 || p->bindcount>3 || p->threads!=32 || !p->gx || !p->gy || p->gz!=1 || p->gx%32 ||
       p->codeoffset<0x6c0 || p->codeoffset%64 || !p->codebytes ||
       !workload_ffn_bounds(0,p->codeoffset,p->codebytes,mapped,sizes) ||
       (p->state!=0x0e40 && p->state!=0x0e58) ||
       p->scratch!=(p->state==0x0e40?0x0c000007u:0x0c00100fu) || p->reserved0 || p->reserved1)return 0;
    int output=0;
    for(unsigned i=0;i<h[2];i++){
      const struct inference_region* r=&s->regions[i];
      if(r->allocation==p->outalloc && r->offset==p->outoff && r->bytes==p->outbytes &&
         (r->flags==0 || r->flags==4) && r->birth<=step && step<=r->last)output=1;
    }
    if(!output)return 0;
    for(unsigned b=0;b<4;b++){
      if(b>=p->bindcount){if(p->bindings[b])return 0;continue;}
      int resident=0;
      for(unsigned i=0;i<h[2];i++){
        const struct inference_region* r=&s->regions[i];
        uint64_t base=r->allocation==19?0x10000080000ull:0x100000a0000ull;
        if(p->bindings[b]==base+r->offset && r->birth<=step && step<=r->last)resident=1;
      }
      if(!resident)return 0;
    }
  }
  return 1;
}
static int g17_workload_decoder(void* io,void* dev,void* queue,uint8_t* const mapped[30],
                                const uint64_t sizes[30],const struct shmem_result shmem[2]){
#ifndef G17_BLOCK_PREFLIGHT
  (void)io;(void)dev;(void)queue;(void)mapped;(void)sizes;(void)shmem;return 2;
#else
  if(sizes[0]!=0x10000 || sizes[20]!=0x40000000 || sizes[17]!=0x20000 ||
     !workload_ffn_bounds(17,0x10000,0x8000,mapped,sizes) || !no_agx_metal())return 2;
  uint16_t metadata,record;memcpy(&metadata,mapped[23]+6,2);memcpy(&record,mapped[25]+9,2);
  if(metadata!=0x2c || record!=0x001a || memcmp(mapped[17]+0x10000,mapped[23],0x8000) ||
     memcmp(mapped[17],mapped[28],0xc000))return 2;
  struct decoder_schedule* plans=calloc(2,sizeof *plans);
  if(!plans)return 2;
  if(!decoder_schedule_load(getenv("ORDERED_DECODER_PREFILL"),plans,32,mapped,sizes) ||
     !decoder_schedule_load(getenv("ORDERED_DECODER_DECODE"),plans+1,1,mapped,sizes)){free(plans);return 2;}
  // The two schedules must share the same immutable and persistent regions.
  for(unsigned i=0;i<plans[0].h[2];i++){
    const struct inference_region* a=&plans[0].regions[i];
    if(a->flags!=1 && a->flags!=4)continue;
    unsigned matches=0;
    for(unsigned j=0;j<plans[1].h[2];j++){
      const struct inference_region* b=&plans[1].regions[j];
      if(a->allocation==b->allocation && a->offset==b->offset && a->bytes==b->bytes && a->capacity==b->capacity && a->flags==b->flags)matches++;
    }
    if(matches!=1){free(plans);return 2;}
  }
  uint8_t* readonly=malloc(plans[0].h[4]);uint8_t* code=malloc(0x10000);
  if(!readonly || !code){free(readonly);free(code);free(plans);return 2;}
  memcpy(readonly,mapped[20],plans[0].h[4]);memcpy(code,mapped[0],0x10000);
  struct workload_queue q;
  if(workload_queue_init(&q,io,dev,queue,shmem)){free(readonly);free(code);free(plans);return 2;}
  unsigned requests=0,valid=0,dirty=0;int result=2;
  uint32_t ready[4]={0x47575231,5,2,256};if(workload_write(ready,sizeof ready))goto done;
  while(1){
    uint32_t request[4];if(workload_read(request,sizeof request) || request[0]!=0x47575231)goto done;
    if(!request[1] && !request[2] && !request[3]){result=0;break;}
    unsigned mode=request[1]&3,decode=(request[1]>>4)&1,reset=(request[1]>>5)&1;
    if(request[1]&~51u || (mode!=1 && mode!=2) || (dirty && !reset) || ++requests>256)goto done;
    const struct decoder_schedule* s=&plans[decode];unsigned tokens=s->h[9];
    if(request[2]!=s->h[3] || !request[3] || request[3]>s->h[1])goto done;
    uint32_t input[321];if(workload_read(input,request[2]))goto done;
    unsigned previous=reset?0:valid;
    if(previous+tokens>256 || (decode && !previous))goto done;
    const uint32_t* mask=input+tokens*2;unsigned next=mask[256];
    if(next<=previous || next>previous+tokens)goto done;
    for(unsigned i=0;i<tokens;i++)if(input[i]>=151936 || input[tokens+i]!=previous+i)goto done;
    for(unsigned i=0;i<256;i++)if(mask[i]!=(i<next?1u:0u))goto done;
    if(reset){
      memset(mapped[20]+s->h[7],0xa5,s->h[8]);
      for(unsigned i=0;i<s->h[2];i++)if(s->regions[i].flags==4)
        memset(mapped[20]+s->regions[i].offset,0,s->regions[i].bytes);
    }
    unsigned input_offset=0;
    for(unsigned i=0;i<4;i++){
      const struct inference_region* r=&s->regions[s->inputs[i]];
      uint8_t* destination=mapped[19]+r->offset;
      memcpy(destination,(uint8_t*)input+input_offset,r->bytes);
      // Decode narrows token/position inputs from 128 to 4 bytes in the same
      // 256-byte slots. Retained prefill bytes are padding, not live inputs.
      memset(destination+r->bytes,0xa5,r->capacity-r->bytes);input_offset+=r->bytes;
    }
    memset(mapped[20]+s->h[5],0xa5,s->h[6]);dirty=1;
    for(unsigned step=0;step<request[3];step++){
      for(unsigned i=0;i<s->h[2];i++)if(!s->regions[i].flags && s->regions[i].birth==step){
        const struct inference_region* r=&s->regions[i];uint8_t* dst=mapped[r->allocation]+r->offset;
        memset(dst,0xff,r->bytes);memset(dst+r->bytes,0xa5,r->capacity-r->bytes);
      }
      const struct workload_ffn_stage* p=&s->stages[step];uint64_t pc=0x10000000000ull+p->codeoffset;
      uint8_t* shader=mapped[17]+0x10000;
      uint16_t low=(uint16_t)pc|7,mid=(uint16_t)(pc>>16),high=(uint16_t)(pc>>32),state=(uint16_t)p->state;
      memcpy(shader+0x40,&low,2);memcpy(shader+0x42,&state,2);
      memcpy(shader+0x46,&mid,2);memcpy(shader+0x48,&high,2);
      uint32_t scratch[2]={p->scratch,0};memcpy(shader+0x38,scratch,8);
      uint32_t geometry[3]={p->gx,p->gy,p->gz};memcpy(mapped[22]+0xa8,geometry,12);
      memcpy(mapped[25]+0x10,geometry,12);memcpy(mapped[25]+0x1c,&p->threads,4);
      memcpy(mapped[17]+0x1ba0,p->bindings,32);memcpy(mapped[28]+0x1ba0,p->bindings,32);
      struct workload_completion completion;if(workload_queue_fire(&q,&completion))goto done;
      unsigned flags=1;
      if(inference_guards(s->regions,s->h[2],step,mapped))flags|=4;
      if(no_agx_metal())flags|=16;
      int thorough=step+1==request[3];
      if(thorough){
        if(!memcmp(readonly,mapped[20],s->h[4]))flags|=2;
        if(!memcmp(code,mapped[0],0x10000))flags|=8;
        int same=1;input_offset=0;
        for(unsigned i=0;i<4;i++){
          const struct inference_region* r=&s->regions[s->inputs[i]];
          if(memcmp(mapped[19]+r->offset,(uint8_t*)input+input_offset,r->bytes))same=0;
          input_offset+=r->bytes;
        }
        if(same)flags|=32;
      }
      uint32_t bytes=mode==1 || thorough?p->outbytes:0;
      uint32_t reply[8]={0x47575231,(uint32_t)completion.status,completion.outword,flags,
        completion.scheduled,completion.completed,bytes,requests*1024+step};
      uint64_t timing[2]={completion.submit_ns,completion.completion_ns};
      if(workload_write(reply,sizeof reply) || workload_write(timing,sizeof timing) ||
         (bytes && workload_write(mapped[p->outalloc]+p->outoff,bytes)))goto done;
      if(flags!=(thorough?63u:21u))goto done;
    }
    if(request[3]==s->h[1]){valid=next;dirty=0;}
  }
done:
  fprintf(stderr,"DECODER CLOSED: requests=%u Submits=%u valid=%u result=%d.\n",requests,q.calls,valid,result);
  struct rusage usage;
  if(!getrusage(RUSAGE_SELF,&usage))
    fprintf(stderr,"DECODER PEAK RSS BYTES: %ld.\n",usage.ru_maxrss);
  dispatch_release(q.done);free(readonly);free(code);free(plans);return result;
#endif
}
