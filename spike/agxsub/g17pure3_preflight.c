// Below-Metal setup and bounded authored dispatch. By default this binary
// makes NO Submit call. ORDERED_VALID_ONE nominates one captured command with
// authored straight-line code; ORDERED_INVALID_ONE is its invalid-ID control.
// It checks every setup return and reports selector-14 shared-memory IDs and
// CPU addresses. IOGPUCommandQueueCreate performs selector 7/16/28 internally;
// repeating those calls makes selector 28 fail. The staged command pages are
// historical templates, not a validated submission. The firmware code-window
// mapping and submission-specific completion are verified for authored A/B
// at the captured layout, not for arbitrary programs. The guarded count-one
// path waits for a Frida interposer to replace every known submission syscall.
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <dlfcn.h>
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <libproc.h>
#include <mach-o/dyld.h>
#ifdef G17_BLOCK_PREFLIGHT
#include <Block.h>
#endif
#include "templates.h"
#include "g17layout_requests.h"

// captured metal_min offsets within the shared data window (base 0x10000000000)
#define WIN_BASE   0x10000000000ULL
#define OFF_OUTPUT 0x30000
#define OFF_CODE   0x90000     // shader machine code storage (data-window)
#define OFF_CDM    0xb0000     // CDM/USC command stream
#define OFF_ARGS   0xe0000     // argument buffer
#define ARG_OUTDESC 0x1ba0     // arg+0x1ba0 = output buffer aperture (buffer-0 binding)
#define BIGSZ      0x100000    // 1MB single buffer spanning all offsets

static unsigned CONN;
__attribute__((visibility("default"))) volatile uint32_t g17_guard_waiting=0;
__attribute__((visibility("default"))) volatile uint32_t g17_guard_ready=0;
struct shmem_result { uint64_t cpu; uint32_t size, id; };
static const uint8_t G17_INPUT_SUM_CODE[56]={
  12,160,16,6,15,0,67,0,24,34,16,192,65,0,128,0,0,0,
  31,4,67,0,24,34,16,192,65,0,128,0,0,0,
  47,2,4,26,33,128,163,42,40,136,33,0,
  47,8,67,0,1,6,16,64,14,0,0,0};
static const uint8_t G17_INPUT_ADD7_CODE[42]={
  12,160,16,6,15,0,67,0,24,34,16,192,65,0,128,0,0,0,
  63,3,4,58,37,0,163,34,40,136,33,0,
  31,8,67,0,1,6,16,64,14,0,0,0};
static const uint8_t G17_INPUT_MULADD_CODE[70]={
  12,160,16,6,15,0,67,0,24,34,16,192,65,0,128,0,0,0,
  31,4,67,0,24,34,16,192,65,0,128,0,0,0,
  47,0,4,26,33,128,161,42,168,136,32,0,1,2,
  39,3,4,58,37,0,163,34,40,137,33,0,
  15,8,67,0,1,6,16,64,14,0,0,0};
static int snapshot_command_pages(const char* path,const struct shmem_result shmem[2]){
  if(!path || shmem[0].size!=0x4000 || shmem[1].size!=0x4000)return -1;
  FILE* f=fopen(path,"wb");if(!f)return -1;
  size_t kernel=fwrite((void*)(uintptr_t)shmem[1].cpu,1,0x4000,f);
  size_t segment=fwrite((void*)(uintptr_t)shmem[0].cpu,1,0x4000,f);
  int status=fclose(f);
  return kernel==0x4000 && segment==0x4000 && status==0?0:-1;
}
static int no_agx_metal(void){
  for(uint32_t i=0;i<_dyld_image_count();i++){
    const char* path=_dyld_get_image_name(i);
    if(path && strstr(path,"AGXMetal")){ fprintf(stderr,"PREFLIGHT REFUSED: AGX Metal image %s\n",path); return 0; }
  }
  return 1;
}
#include "g17workload_tile.h"
#include "g17workload_ffn.h"
#include "g17workload_inference.h"
#include "g17workload_decoder.h"
#include "g17workload_attention.h"
#include "g17workload_attention_fused.h"
#include "g17workload_layer.h"
static int ordered_tg_param(unsigned* grid,unsigned* xorbit){
  const char* gs=getenv("ORDERED_TG_PARAM_GRID");
  const char* xs=getenv("ORDERED_TG_PARAM_XOR");
  if(!gs && !xs)return 0;
  if(!gs || !xs || !*gs || !*xs)return -1;
  char *ge=NULL,*xe=NULL;
  unsigned long g=strtoul(gs,&ge,10),x=strtoul(xs,&xe,10);
  if(*ge || *xe || g<64 || g>=256 || !(x==1 || x==2 || x==4 ||
                                       x==8 || x==16 || x==32))return -1;
  for(unsigned i=0;i<g;i++)if(((i&~63u)+((i&63u)^x))>=g)return -1;
  *grid=(unsigned)g;*xorbit=(unsigned)x;return 1;
}
static uint32_t tg_b_word(unsigned i,int scrambled){
  return scrambled?(0x10000u+((29u*i)^0x5a5u)):(7u*i+3u);
}
static int check(kern_return_t k, const char* what){
  if(k==0)return 0;
  fprintf(stderr,"PREFLIGHT REFUSED: %s returned kr=0x%x\n",what,k);
  return -1;
}

// sel-9 alloc. outputStruct (88B): out[0]=aperture, out[1]=host VA,
// out[5] behaves as rounded size in retained and live captures, not a handle.
static int gpu_alloc(uint64_t size, uint64_t* aperture, uint64_t* hostva, uint64_t* handle){
  unsigned char in[104]; memcpy(in, ALLOC_PARENT, 104);
  *(uint64_t*)(in+0x48) = size;                 // size field (decoded from capture)
  unsigned long long o[11]; size_t os=88; memset(o,0,sizeof o);
  kern_return_t k=IOConnectCallMethod(CONN,9,0,0,in,104,0,0,o,&os);
  if(k || os!=88 || !o[0] || !o[1] || o[5]<size || o[5]-size>=0x4000){
    fprintf(stderr,"  sel-9(size=0x%llx) REFUSED kr=0x%x bytes=%zu aperture=0x%llx host=0x%llx rounded_size=0x%llx\n",
            size,k,os,o[0],o[1],o[5]); return -1;
  }
  if(aperture)*aperture=o[0]; if(hostva)*hostva=o[1]; if(handle)*handle=o[5];
  fprintf(stderr,"  sel-9 size=0x%llx: aperture=0x%llx host=0x%llx returned_word5_size=0x%llx\n",size,o[0],o[1],o[5]);
  return 0;
}
// The two 64 KiB shader allocations in the exact authored Metal A/B capture
// use the same 104-byte selector-9 request (sha256 f23c87348edbabd6...).
// Probe this class independently of the old 1 MiB ALLOC_PARENT template.
static int code_alloc_control(void){
  static const unsigned char request[104]={
    0,0,0,0,0,0,0,0, 1,0,1,0,1,0,0,0,
    1,1,0,1,48,4,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,8,24,0,0,0,
    0,0,0,0,0,0,0,0};
  for(int i=0;i<2;i++){
    unsigned long long o[11]={0};size_t n=sizeof o;
    kern_return_t k=IOConnectCallMethod(CONN,9,0,0,request,sizeof request,0,0,o,&n);
    uint64_t expected=WIN_BASE+(i?0x18000:0);
    fprintf(stderr,"CODE ALLOC #%d kr=0x%x out=%zu aperture=0x%llx host=0x%llx size=0x%llx expected=0x%llx\n",
            i,k,n,o[0],o[1],o[5],(unsigned long long)expected);
    if(k || n!=88 || o[0]!=expected || !o[1] || o[5]!=0x10000)return 2;
    volatile unsigned char* cpu=(volatile unsigned char*)(uintptr_t)o[1];
    unsigned char before=cpu[0];cpu[0]=0xA5;
    if(cpu[0]!=0xA5)return 2;
    cpu[0]=before;
  }
  if(!no_agx_metal())return 2;
  fprintf(stderr,"CODE ALLOC CONTROL PASS; no command buffer or Submit call.\n");
  return 0;
}
static int layout_alloc_control(void* io,void* dev){
  const char* payload_path=getenv("STAGE_CAPTURED_PAYLOAD");
  void* base_class=NULL;
  const char* class_name=getenv("VIEW5_BASE_CLASS");
  if(class_name){
    if(strcmp(class_name,"IOGPUMetalBuffer")){fprintf(stderr,"CLASS REFUSED: unsupported class\n");return 2;}
    void*(*GetClass)(const char*)=dlsym(RTLD_DEFAULT,"objc_getClass");
    base_class=GetClass?GetClass(class_name):NULL;
    fprintf(stderr,"CLASS CONTROL: %s=%p with AGXMetal absent\n",class_name,base_class);
    if(!base_class || !no_agx_metal())return 2;
  }
  FILE* payload=NULL;
  unsigned char* mapped[11]={0};
  if(payload_path){
    payload=fopen(payload_path,"rb");
    if(!payload){perror("STAGE_CAPTURED_PAYLOAD fopen");return 2;}
  }
  for(size_t i=0;i<sizeof G17_LAYOUT_REQUESTS/sizeof G17_LAYOUT_REQUESTS[0];i++){
    const struct g17_layout_request* row=&G17_LAYOUT_REQUESTS[i];
    uint8_t request[sizeof row->input];memcpy(request,row->input,sizeof request);
    if(base_class && (row->original_index==2 || row->original_index==20))
      memcpy(request+96,&base_class,sizeof base_class);
    unsigned long long o[11]={0};size_t n=sizeof o;
    kern_return_t k=IOConnectCallMethod(CONN,9,0,0,request,sizeof request,0,0,o,&n);
    fprintf(stderr,"LAYOUT ALLOC original#%d kr=0x%x bytes=%zu aperture=0x%llx expected=0x%llx host=0x%llx size=0x%llx expected_size=0x%llx\n",
            row->original_index,k,n,o[0],(unsigned long long)row->aperture,
            o[1],o[5],(unsigned long long)row->size);
    if(k || n!=88 || o[0]!=row->aperture || !o[1] || o[5]!=row->size){
      fprintf(stderr,"LAYOUT PREFLIGHT REFUSED at original allocation #%d\n",row->original_index);
      return 2;
    }
    mapped[i]=(unsigned char*)(uintptr_t)o[1];
    if(payload){
      unsigned char* expected=malloc(row->size);
      if(!expected){fprintf(stderr,"STAGE REFUSED: allocation failed\n");return 2;}
      if(fread(expected,1,row->size,payload)!=row->size){
        fprintf(stderr,"STAGE REFUSED: short payload at original allocation #%d\n",row->original_index);
        free(expected);
        return 2;
      }
      memcpy(mapped[i],expected,row->size);
      int readback_ok=memcmp(mapped[i],expected,row->size)==0;
      free(expected);
      fprintf(stderr,"  STAGED original#%d bytes=0x%llx cpu_readback=%s\n",
              row->original_index,(unsigned long long)row->size,readback_ok?"ok":"failed");
      if(!readback_ok)return 2;
    }
    if(!no_agx_metal())return 2;
  }
  if(payload){
    if(fgetc(payload)!=EOF || ferror(payload)){
      fprintf(stderr,"STAGE REFUSED: trailing payload bytes or read error\n");return 2;
    }
    fclose(payload);
    uint32_t packet=0;
    uint16_t packet_code_low=0,packet_code_high=0;
    uint64_t args[3]={0};
    memcpy(&packet,mapped[5]+0x40,sizeof packet);
    memcpy(&packet_code_low,mapped[5]+0x40,sizeof packet_code_low);
    memcpy(&packet_code_high,mapped[5]+0x48,sizeof packet_code_high);
    const uint64_t code_aperture=WIN_BASE+0x6c0;
    for(int i=0;i<3;i++)memcpy(&args[i],mapped[10]+0x1ba0+8*i,sizeof args[i]);
    if(memcmp(mapped[0]+0x6c0,KERNEL_3I1,sizeof KERNEL_3I1)!=0 ||
       mapped[0][0x6d3]!=1 || packet!=0x0e4006c7 ||
       packet_code_low!=(((uint16_t)code_aperture)|7u) ||
       packet_code_high!=(uint16_t)(code_aperture>>32) ||
       args[0]!=WIN_BASE+0x30000 || args[1]!=WIN_BASE+0x30400 || args[2]!=WIN_BASE+0x30800){
      fprintf(stderr,"STAGE REFUSED: authored code, packet, or argument descriptor mismatch\n");
      return 2;
    }
    fprintf(stderr,"STAGE PASS: authored A code=42B at 0x%llx, immediate=1, packet=0x%08x, encoded code low/high=0x%04x/0x%04x, args=0x%llx/0x%llx/0x%llx; CPU readback only.\n",
            (unsigned long long)code_aperture,packet,packet_code_low,packet_code_high,
            (unsigned long long)args[0],(unsigned long long)args[1],(unsigned long long)args[2]);
  }
  const char* view_path=getenv("STAGE_VIEW5_TEMPLATE");
  if(view_path){
    if(!mapped[2] || !payload){fprintf(stderr,"VIEW5 REFUSED: backing or staged payload absent\n");return 2;}
    FILE* view=fopen(view_path,"rb");
    if(!view){perror("VIEW5 fopen");return 2;}
    uint8_t request[104];
    if(fread(request,1,sizeof request,view)!=sizeof request || fgetc(view)!=EOF || ferror(view)){
      fprintf(stderr,"VIEW5 REFUSED: template length\n");fclose(view);return 2;
    }
    fclose(view);
    uint64_t view_offset=0,owner=0;
    memcpy(&view_offset,request,8);memcpy(&owner,request+96,8);
    if(view_offset!=0x80 || owner!=0 || memcmp(request+56,"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",16)){
      fprintf(stderr,"VIEW5 REFUSED: template not normalized\n");return 2;
    }
    uint64_t base_cpu=(uint64_t)(uintptr_t)mapped[2],view_cpu=base_cpu+0x800;
    memcpy(request+56,&view_cpu,8);memcpy(request+64,&base_cpu,8);
    uint64_t o[11]={0};size_t n=sizeof o;
    kern_return_t k=IOConnectCallMethod(CONN,9,0,0,request,sizeof request,0,0,o,&n);
    fprintf(stderr,"VIEW5 sel-9 kr=0x%x bytes=%zu aperture=0x%llx base_cpu=0x%llx input_view_cpu=0x%llx host=0x%llx returned_view=0x%llx size=0x%llx\n",
            k,n,(unsigned long long)o[0],(unsigned long long)base_cpu,
            (unsigned long long)view_cpu,(unsigned long long)o[1],
            (unsigned long long)o[2],(unsigned long long)o[5]);
    if(k || n!=88 || o[0]!=WIN_BASE+0x30800 || o[1]!=0 || !o[2] || o[5]!=0x20000){
      fprintf(stderr,"VIEW5 REFUSED: output mismatch\n");return 2;
    }
    unsigned char view_bytes[0x100];mach_vm_size_t copied=0;
    k=mach_vm_read_overwrite(mach_task_self(),o[2],sizeof view_bytes,
                             (mach_vm_address_t)(uintptr_t)view_bytes,&copied);
    if(k || copied!=sizeof view_bytes){fprintf(stderr,"VIEW5 REFUSED: returned CPU view unreadable kr=0x%x bytes=%llu\n",k,copied);return 2;}
    fprintf(stderr,"VIEW5 returned descriptor prefix256=");
    for(unsigned i=0;i<256;i++)fprintf(stderr,"%02x",view_bytes[i]);
    fprintf(stderr,"\n");
    if(!no_agx_metal())return 2;
    fprintf(stderr,"VIEW5 CONTROL PASS: selector-9 accepted view aperture, returned descriptor readable; no Submit.\n");
  }
  const char* view9_path=getenv("STAGE_VIEW9_TEMPLATE");
  if(view9_path){
    if(!mapped[2] || !payload){fprintf(stderr,"VIEW9 REFUSED: backing or payload absent\n");return 2;}
    FILE* view=fopen(view9_path,"rb");if(!view){perror("VIEW9 fopen");return 2;}
    uint8_t request[104];
    if(fread(request,1,sizeof request,view)!=sizeof request || fgetc(view)!=EOF || ferror(view)){
      fprintf(stderr,"VIEW9 REFUSED: template length\n");fclose(view);return 2;
    }
    fclose(view);
    uint64_t owner=1;memcpy(&owner,request+96,8);
    if(owner || memcmp(request+56,"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",16)){
      fprintf(stderr,"VIEW9 REFUSED: template not normalized\n");return 2;
    }
    uint64_t base_cpu=(uint64_t)(uintptr_t)mapped[2],view_cpu=base_cpu+0x1100;
    memcpy(request+56,&view_cpu,8);memcpy(request+64,&base_cpu,8);
    uint64_t o[11]={0};size_t n=sizeof o;
    kern_return_t k=IOConnectCallMethod(CONN,9,0,0,request,sizeof request,0,0,o,&n);
    fprintf(stderr,"VIEW9 sel-9 kr=0x%x bytes=%zu aperture=0x%llx descriptor=0x%llx size=0x%llx\n",
            k,n,(unsigned long long)o[0],(unsigned long long)o[2],(unsigned long long)o[5]);
    if(k || n!=88 || o[0]!=WIN_BASE+0x31100 || o[1]!=0 || !o[2] || o[5]!=0x20000){
      fprintf(stderr,"VIEW9 REFUSED: output mismatch\n");return 2;
    }
    unsigned char descriptor[256];mach_vm_size_t copied=0;
    k=mach_vm_read_overwrite(mach_task_self(),o[2],sizeof descriptor,
                             (mach_vm_address_t)(uintptr_t)descriptor,&copied);
    if(k || copied!=sizeof descriptor){fprintf(stderr,"VIEW9 REFUSED: descriptor unreadable kr=0x%x bytes=%llu\n",k,copied);return 2;}
    fprintf(stderr,"VIEW9 descriptor prefix256=");
    for(unsigned i=0;i<sizeof descriptor;i++)fprintf(stderr,"%02x",descriptor[i]);
    fprintf(stderr,"\n");
    if(!no_agx_metal())return 2;
    fprintf(stderr,"VIEW9 CONTROL PASS: selector-9 accepted zeroed-owner view and returned readable descriptor; no Submit.\n");
  }
  const char* all_views_path=getenv("STAGE_ALL_VIEWS");
  if(all_views_path){
    if(!payload || !mapped[2] || !mapped[3]){fprintf(stderr,"ALL VIEWS REFUSED: backing absent\n");return 2;}
    FILE* views=fopen(all_views_path,"rb");if(!views){perror("ALL VIEWS fopen");return 2;}
    static const uint32_t expected_indices[17]={3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21};
    for(unsigned row=0;row<17;row++){
      uint8_t blob[136];
      if(fread(blob,1,sizeof blob,views)!=sizeof blob){fprintf(stderr,"ALL VIEWS REFUSED: short row %u\n",row);fclose(views);return 2;}
      uint32_t index=0,slot=0;uint64_t offset=0,aperture=0,size=0;
      memcpy(&index,blob,4);memcpy(&slot,blob+4,4);
      memcpy(&offset,blob+8,8);memcpy(&aperture,blob+16,8);memcpy(&size,blob+24,8);
      uint8_t request[104];memcpy(request,blob+32,sizeof request);
      static const uint8_t zero16[16]={0};
      if(index!=expected_indices[row] || slot!=(index==21?3u:2u) ||
         offset>=0x20000 || size!=0x20000 ||
         memcmp(request+56,zero16,16) || memcmp(request+96,zero16,8)){
        fprintf(stderr,"ALL VIEWS REFUSED: malformed row %u\n",row);fclose(views);return 2;
      }
      uint64_t base_cpu=(uint64_t)(uintptr_t)mapped[slot],view_cpu=base_cpu+offset;
      memcpy(request+56,&view_cpu,8);memcpy(request+64,&base_cpu,8);
      if(index==21 && getenv("ALL_VIEWS_REBASE_PARENT")){
        uint64_t parent=4;memcpy(request+80,&parent,8);
        fprintf(stderr,"ALL VIEWS index=21 parent reference rebased 0x15->0x4 for pure allocation order\n");
      }
      uint64_t o[11]={0};size_t n=sizeof o;
      kern_return_t k=IOConnectCallMethod(CONN,9,0,0,request,sizeof request,0,0,o,&n);
      fprintf(stderr,"ALL VIEWS index=%u kr=0x%x bytes=%zu aperture=0x%llx expected=0x%llx descriptor=0x%llx\n",
              index,k,n,(unsigned long long)o[0],(unsigned long long)aperture,
              (unsigned long long)o[2]);
      if(k || n!=88 || o[0]!=aperture || o[1]!=0 || !o[2] || o[5]!=size){
        fprintf(stderr,"ALL VIEWS REFUSED: mismatch at index %u\n",index);fclose(views);return 2;
      }
      if(!no_agx_metal()){fclose(views);return 2;}
    }
    if(fgetc(views)!=EOF || ferror(views)){fprintf(stderr,"ALL VIEWS REFUSED: trailing data\n");fclose(views);return 2;}
    fclose(views);
    fprintf(stderr,"ALL VIEWS CONTROL PASS: 17 selector-9 views registered at captured apertures, class/object pointers zero; no Submit.\n");
  }
  const char* special18_path=getenv("STAGE_SPECIAL18");
  if(special18_path){
    FILE* f=fopen(special18_path,"rb");if(!f){perror("SPECIAL18 fopen");return 2;}
    uint8_t request[104];
    if(fread(request,1,sizeof request,f)!=sizeof request || fgetc(f)!=EOF || ferror(f)){
      fprintf(stderr,"SPECIAL18 REFUSED: template length\n");fclose(f);return 2;
    }
    fclose(f);
    uint64_t o[11]={0};size_t n=sizeof o;
    kern_return_t k=IOConnectCallMethod(CONN,9,0,0,request,sizeof request,0,0,o,&n);
    fprintf(stderr,"SPECIAL18 sel-9 kr=0x%x bytes=%zu aperture=0x%llx host=0x%llx size=0x%llx\n",
            k,n,(unsigned long long)o[0],(unsigned long long)o[1],(unsigned long long)o[5]);
    if(k || n!=88 || o[0]!=0 || !o[1] || o[5]!=0x10000 || !no_agx_metal()){
      fprintf(stderr,"SPECIAL18 REFUSED: output mismatch\n");return 2;
    }
    fprintf(stderr,"SPECIAL18 CONTROL PASS: no-aperture selector-9 allocation accepted; no Submit.\n");
  }
  if(getenv("LAYOUT_QUEUE_PREFLIGHT")){
    void*(*QCreate)(void*,void*,unsigned long)=dlsym(io,"IOGPUCommandQueueCreate");
    if(!QCreate){fprintf(stderr,"LAYOUT QUEUE REFUSED: QueueCreate unavailable\n");return 2;}
    void* notification=NULL;
    void*(*NCreate)(void*,unsigned,unsigned)=NULL;
    void*(*NSet)(void*,void*,unsigned)=NULL;
    void* notification_dispatch=NULL;
    if(getenv("NOTIFICATION_PREFLIGHT")){
      NCreate=dlsym(io,"IOGPUNotificationQueueCreate");
      NSet=dlsym(io,"IOGPUNotificationQueueSetDispatchQueue");
      if(!NCreate || !NSet){fprintf(stderr,"NOTIFICATION REFUSED: exports unavailable\n");return 2;}
      notification=NCreate(dev,0x100,0x28);
      fprintf(stderr,"NOTIFICATION create=%p params=0x100/0x28\n",notification);
      if(!notification)return 2;
    }
    unsigned char qdesc[0x410]={0};
    char process_path[PROC_PIDPATHINFO_MAXSIZE]={0};
    if(proc_pidpath(getpid(),process_path,sizeof process_path)<=0 || strlen(process_path)>=29){
      fprintf(stderr,"LAYOUT QUEUE REFUSED: process path cannot fit descriptor\n");return 2;
    }
    memcpy(qdesc,process_path,strlen(process_path)+1);
    memcpy(qdesc+0x3e3,process_path,strlen(process_path)+1);
    *(uint32_t*)(qdesc+0x400)=2;
    *(uint32_t*)(qdesc+0x408)=0xffffffffu;
    *(uint32_t*)(qdesc+0x40c)=1;
    unsigned old=*(unsigned*)((char*)dev+0x14);
    *(unsigned*)((char*)dev+0x14)=CONN;
    void* queue=QCreate(dev,qdesc,sizeof qdesc);
    fprintf(stderr,"LAYOUT QUEUE create=%p connection=0x%x prior=0x%x\n",queue,CONN,old);
    if(!queue)return 2;
    if(notification){
      dispatch_queue_t dq=dispatch_queue_create("agxforge.g17.pure-notification-preflight",DISPATCH_QUEUE_SERIAL);
      if(!dq){fprintf(stderr,"NOTIFICATION REFUSED: dispatch queue unavailable\n");return 2;}
      notification_dispatch=NSet(notification,(void*)dq,1);
      fprintf(stderr,"NOTIFICATION bind queue=%p result=%p\n",(void*)dq,notification_dispatch);
      if(!notification_dispatch)return 2;
      fprintf(stderr,"NOTIFICATION PREFLIGHT PASS: created IOGPU notification queue and bound a serial dispatch queue; no Submit or callback delivery.\n");
    }
    unsigned char ob[16];size_t obc=sizeof ob;
    kern_return_t k=IOConnectCallStructMethod(CONN,6,0,0,ob,&obc);
    fprintf(stderr,"LAYOUT QUEUE sel-6 kr=0x%x bytes=%zu\n",k,obc);
    if(k || obc!=sizeof ob)return 2;
    struct shmem_result shmem[2]={0};
    for(int i=0;i<2;i++){
      uint64_t args[2]={0x4000,(uint64_t)i};obc=sizeof shmem[i];
      k=IOConnectCallMethod(CONN,14,args,2,0,0,0,0,&shmem[i],&obc);
      fprintf(stderr,"LAYOUT QUEUE sel-14#%d kr=0x%x bytes=%zu cpu=0x%llx size=0x%x id=%u\n",
              i,k,obc,(unsigned long long)shmem[i].cpu,shmem[i].size,shmem[i].id);
      if(k || obc!=sizeof shmem[i] || !shmem[i].cpu || shmem[i].size!=0x4000 ||
         shmem[i].id!=(i==0?1:2))return 2;
    }
    uint64_t first=0,second=0;
    if(getenv("TRACE_ID_PREFLIGHT")){
      uint64_t(*NextTraceID)(void*)=dlsym(io,"IOGPUDeviceGetNextGlobalTraceID");
      if(!NextTraceID){fprintf(stderr,"TRACE ID REFUSED: export absent\n");return 2;}
      first=NextTraceID(dev);second=NextTraceID(dev);
      fprintf(stderr,"TRACE ID control first=0x%llx second=0x%llx\n",
              (unsigned long long)first,(unsigned long long)second);
      if(!first || second!=first+1)return 2;
      fprintf(stderr,"TRACE ID PASS: two current-process consecutive IDs; no storage or Submit.\n");
    }
    const char* command_path=getenv("STAGE_COMMAND_PAGES");
    if(command_path){
      if(!first || !second){fprintf(stderr,"COMMAND PAGES REFUSED: trace IDs absent\n");return 2;}
      FILE* f=fopen(command_path,"rb");
      if(!f){perror("STAGE_COMMAND_PAGES fopen");return 2;}
      unsigned char* pages=malloc(0x8000);
      if(!pages){fclose(f);return 2;}
      if(fread(pages,1,0x8000,f)!=0x8000 || fgetc(f)!=EOF || ferror(f)){
        fprintf(stderr,"COMMAND PAGES REFUSED: template length or read error\n");
        free(pages);fclose(f);return 2;
      }
      fclose(f);
      memcpy(pages+0x234,&second,sizeof(uint32_t));
      memcpy(pages+0x4000,&first,sizeof first);
      memcpy(pages+0x4018,&first,sizeof first);
      memcpy(pages+0x4028,&second,sizeof second);
      uint32_t ready=0;
      memcpy(&ready,pages+0x4024,sizeof ready);
      if(ready!=0xf0){fprintf(stderr,"COMMAND PAGES REFUSED: ready bit set\n");free(pages);return 2;}
      memcpy((void*)(uintptr_t)shmem[1].cpu,pages,0x4000);
      memcpy((void*)(uintptr_t)shmem[0].cpu,pages+0x4000,0x4000);
      int match=memcmp((void*)(uintptr_t)shmem[1].cpu,pages,0x4000)==0 &&
                memcmp((void*)(uintptr_t)shmem[0].cpu,pages+0x4000,0x4000)==0;
      free(pages);
      if(!match){fprintf(stderr,"COMMAND PAGES REFUSED: CPU readback mismatch\n");return 2;}
      fprintf(stderr,"COMMAND PAGES PASS: 32 KiB staged in selector-14 IDs 2/1 with current trace IDs and ready bit clear; no Submit.\n");
    }
#ifdef G17_BLOCK_PREFLIGHT
    if(getenv("RECORD_PREFLIGHT")){
      volatile uint64_t marks[2]={0,0};
      volatile uint64_t* mark_ptr=marks;
      void (^scheduled)(void)=Block_copy(^{mark_ptr[0]=0x17a;});
      void (^completed)(void)=Block_copy(^{mark_ptr[1]=0x17b;});
      if(!scheduled || !completed || scheduled==completed)return 2;
      unsigned char record[64]={0};
      uint32_t kernel_id=shmem[1].id,segment_id=shmem[0].id;
      uintptr_t scheduled_ptr=(uintptr_t)scheduled,completed_ptr=(uintptr_t)completed;
      memcpy(record,&kernel_id,sizeof kernel_id);
      memcpy(record+4,&segment_id,sizeof segment_id);
      memcpy(record+0x10,&scheduled_ptr,sizeof scheduled_ptr);
      memcpy(record+0x18,&completed_ptr,sizeof completed_ptr);
      if(*(uint32_t*)record!=2 || *(uint32_t*)(record+4)!=1 ||
         !scheduled_ptr || !completed_ptr)return 2;
      scheduled();completed();
      if(marks[0]!=0x17a || marks[1]!=0x17b)return 2;
      fprintf(stderr,"RECORD PREFLIGHT PASS: 64B record IDs 2/1 and two distinct live CPU blocks at +0x10/+0x18; callbacks invoked locally only; no Submit.\n");
      Block_release(scheduled);Block_release(completed);
    }
#endif
    if(!no_agx_metal())return 2;
    fprintf(stderr,"LAYOUT QUEUE PASS: two selector-14 pages registered, IDs 1/2.\n");
    if(getenv("ZERO_SUBMIT_PREFLIGHT")){
      if(!notification || !payload || !command_path){
        fprintf(stderr,"ZERO SUBMIT REFUSED: notification, payload, or command pages absent\n");return 2;
      }
      typedef int (*submit_t)(void*,void*,unsigned,void*,unsigned,void*);
      submit_t Submit=dlsym(io,"IOGPUCommandQueueSubmitCommandBuffers");
      if(!Submit){fprintf(stderr,"ZERO SUBMIT REFUSED: export absent\n");return 2;}
      unsigned intact_before=0,intact_after=0;
      // The third authored binding is at aperture WIN_BASE+0x30800,
      // i.e. +0x800 within the original#2 allocation at WIN_BASE+0x30000.
      uint32_t* output_words=(uint32_t*)(mapped[2]+0x800);
      for(unsigned i=0;i<64;i++)intact_before+=(output_words[i]==0xDEADBEEFu);
      if(intact_before!=64){fprintf(stderr,"ZERO SUBMIT REFUSED: output sentinels not intact before call\n");return 2;}
      unsigned char out[64]={0};
      int result=-1;
      unsigned long long mark0=0,mark1=0;
      if(getenv("ZERO_SUBMIT_NULL_RECORD")){
        fprintf(stderr,"ZERO SUBMIT entering: count=0 records=NULL stride=64 ready=0xf0; no GPU command nominated.\n");
        result=Submit(queue,NULL,0,NULL,64,out);
      } else {
#ifdef G17_BLOCK_PREFLIGHT
      static volatile uint64_t zero_marks[2]={0,0};
      volatile uint64_t* mark_ptr=zero_marks;
      void (^scheduled)(void)=Block_copy(^{mark_ptr[0]=0x17c;});
      void (^completed)(void)=Block_copy(^{mark_ptr[1]=0x17d;});
      if(!scheduled || !completed || scheduled==completed)return 2;
      unsigned char record[64]={0};
      uint32_t kernel_id=shmem[1].id,segment_id=shmem[0].id;
      uintptr_t scheduled_ptr=(uintptr_t)scheduled,completed_ptr=(uintptr_t)completed;
      memcpy(record,&kernel_id,4);memcpy(record+4,&segment_id,4);
      memcpy(record+0x10,&scheduled_ptr,sizeof scheduled_ptr);
      memcpy(record+0x18,&completed_ptr,sizeof completed_ptr);
      fprintf(stderr,"ZERO SUBMIT entering: count=0 records=live stride=64 ready=0xf0; no GPU command nominated.\n");
      result=Submit(queue,NULL,0,record,64,out);
      mark0=zero_marks[0];mark1=zero_marks[1];
#else
      fprintf(stderr,"ZERO SUBMIT REFUSED: build lacks live block support\n");return 2;
#endif
      }
      for(unsigned i=0;i<64;i++)intact_after+=(output_words[i]==0xDEADBEEFu);
      fprintf(stderr,"ZERO SUBMIT returned status=%d out[0:8]=%02x%02x%02x%02x%02x%02x%02x%02x sentinels=%u/64 marks=%llu/%llu; no GPU command nominated.\n",
              result,out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],intact_after,
              mark0,mark1);
      if(intact_after!=64)return 2;
    }
  }
  fprintf(stderr,"LAYOUT CONTROL PASS; %s.\n",
          getenv("ZERO_SUBMIT_PREFLIGHT")?"zero-count Submit only, no GPU command":"no Submit call");
  return 0;
}
static unsigned ffn_cross_guards(const uint8_t* allocation2,const uint8_t* allocation17,unsigned candidate){
  const uint32_t* rival=(const uint32_t*)(candidate?allocation2+0x8000:allocation17+0x8000);
  unsigned intact=1;
  for(unsigned i=0;i<1024;i++)intact&=rival[i]==0u;
  for(unsigned i=0;i<0x80;i++){
    intact&=allocation17[0x7f80+i]==0xc7;
    intact&=allocation17[0x9000+i]==0xc7;
  }
  return intact;
}
static int ordered_alloc_control(void* io,void* dev){
  const char* path=getenv("ORDERED_ALLOC_PREFLIGHT");
  int tensor_graph=getenv("ORDERED_TENSOR_GRAPH")!=NULL;
  int tensor_common=getenv("ORDERED_TENSOR_COMMON")!=NULL;
  int tensor_loop=getenv("ORDERED_TENSOR_LOOP")!=NULL;
  int tensor_loop5=getenv("ORDERED_TENSOR_LOOP5")!=NULL;
  int tensor_loop7=getenv("ORDERED_TENSOR_LOOP7")!=NULL;
  int tensor_loop8=getenv("ORDERED_TENSOR_LOOP8")!=NULL;
  int tensor_loop15=getenv("ORDERED_TENSOR_LOOP15")!=NULL;
  int tensor_loop16=getenv("ORDERED_TENSOR_LOOP16")!=NULL;
  int tensor_loopscale=getenv("ORDERED_TENSOR_LOOPSCALE")!=NULL;
  int tensor_sg4=getenv("ORDERED_TENSOR_SG4")!=NULL;
  int tensor_sg4loop16=getenv("ORDERED_TENSOR_SG4LOOP16")!=NULL;
  int tensor_sg4h=getenv("ORDERED_TENSOR_SG4H")!=NULL;
  int tensor_sg4hloop16=getenv("ORDERED_TENSOR_SG4HLOOP16")!=NULL;
  int tensor_sg4h3loop16=getenv("ORDERED_TENSOR_SG4H3LOOP16")!=NULL;
  int tensor_sg4h3=getenv("ORDERED_TENSOR_SG4H3")!=NULL;
  int tensor_sg4h4=getenv("ORDERED_TENSOR_SG4H4")!=NULL;
  int tensor_sg4h5=getenv("ORDERED_TENSOR_SG4H5")!=NULL;
  int tensor_sg4h6=getenv("ORDERED_TENSOR_SG4H6")!=NULL;
  int tensor_sg4h7=getenv("ORDERED_TENSOR_SG4H7")!=NULL;
  int tensor_sg4h8=getenv("ORDERED_TENSOR_SG4H8")!=NULL;
  int tensor_sg4h9=getenv("ORDERED_TENSOR_SG4H9")!=NULL;
  int tensor_sg4h10=getenv("ORDERED_TENSOR_SG4H10")!=NULL;
  int tensor_sg4h12=getenv("ORDERED_TENSOR_SG4H12")!=NULL;
  int tensor_sg4h12loop16=getenv("ORDERED_TENSOR_SG4H12LOOP16")!=NULL;
  int tensor_sg4h12loop17=getenv("ORDERED_TENSOR_SG4H12LOOP17")!=NULL;
  int tensor_sg4h12loop23=getenv("ORDERED_TENSOR_SG4H12LOOP23")!=NULL;
  int tensor_sg4h12loop24=getenv("ORDERED_TENSOR_SG4H12LOOP24")!=NULL;
  int tensor_sg4h12loop31=getenv("ORDERED_TENSOR_SG4H12LOOP31")!=NULL;
  int tensor_sg4h12loop32=getenv("ORDERED_TENSOR_SG4H12LOOP32")!=NULL;
  int tensor_sg4h12loop40=getenv("ORDERED_TENSOR_SG4H12LOOP40")!=NULL;
  int tensor_sg4h12loop64=getenv("ORDERED_TENSOR_SG4H12LOOP64")!=NULL;
  int tensor_sg4h12loop128=getenv("ORDERED_TENSOR_SG4H12LOOP128")!=NULL;
  int tensor_sg4h12loop136=getenv("ORDERED_TENSOR_SG4H12LOOP136")!=NULL;
  int tensor_sg4h12loop254=getenv("ORDERED_TENSOR_SG4H12LOOP254")!=NULL;
  int tensor_sg4h12loop255=getenv("ORDERED_TENSOR_SG4H12LOOP255")!=NULL;
  int tensor_sg4h12loop256tail=getenv("ORDERED_TENSOR_SG4H12LOOP256TAIL")!=NULL;
  int tensor_sg4h12loop510dual=getenv("ORDERED_TENSOR_SG4H12LOOP510DUAL")!=NULL;
  int tensor_sg4h12loop765triple=getenv("ORDERED_TENSOR_SG4H12LOOP765TRIPLE")!=NULL;
  int tensor_sg4h12loop1020quad=getenv("ORDERED_TENSOR_SG4H12LOOP1020QUAD")!=NULL;
  int tensor_sg4h12loop1275quint=getenv("ORDERED_TENSOR_SG4H12LOOP1275QUINT")!=NULL;
  int tensor_sg4h12loop2040oct=getenv("ORDERED_TENSOR_SG4H12LOOP2040OCT")!=NULL;
  int tensor_sg4h12loop8160codecontrol=getenv("ORDERED_TENSOR_SG4H12LOOP8160CODECONTROL")!=NULL;
  int tensor_sg4h12loop9180reuse=getenv("ORDERED_TENSOR_SG4H12LOOP9180REUSE")!=NULL;
  int tensor_sg4h12loop18360reuse=getenv("ORDERED_TENSOR_SG4H12LOOP18360REUSE")!=NULL;
  int tensor_sg4h12loop36720reuse=getenv("ORDERED_TENSOR_SG4H12LOOP36720REUSE")!=NULL;
  int tensor_sg4h12loop73728reuse=getenv("ORDERED_TENSOR_SG4H12LOOP73728REUSE")!=NULL;
  int tensor_sg4h12loop137700reuse=getenv("ORDERED_TENSOR_SG4H12LOOP137700REUSE")!=NULL;
  int tensor_sg4h12loop8160reuse=getenv("ORDERED_TENSOR_SG4H12LOOP8160REUSE")!=NULL;
  int tensor_sg4h12loop7650runtime=getenv("ORDERED_TENSOR_SG4H12LOOP7650RUNTIME")!=NULL;
  int tensor_sg4h12loop6120runtime=getenv("ORDERED_TENSOR_SG4H12LOOP6120RUNTIME")!=NULL;
  int tensor_sg4h12loop8160runtime=getenv("ORDERED_TENSOR_SG4H12LOOP8160RUNTIME")!=NULL;
  int tensor_sg4h12loop4080sixteen=getenv("ORDERED_TENSOR_SG4H12LOOP4080SIXTEEN")!=NULL;
  const char* b_physical_mib_text=getenv("ORDERED_TENSOR_B_PHYSICAL_MIB");
  unsigned b_physical_mib=b_physical_mib_text?(unsigned)strtoul(b_physical_mib_text,NULL,10):0u;
  const char* post_b_gap_mib_text=getenv("ORDERED_TENSOR_POST_B_GAP_MIB");
  unsigned post_b_gap_mib=post_b_gap_mib_text?(unsigned)strtoul(post_b_gap_mib_text,NULL,10):0u;
  const char* b_physical_extra_kib_text=getenv("ORDERED_TENSOR_B_PHYSICAL_EXTRA_KIB");
  unsigned b_physical_extra_kib=b_physical_extra_kib_text?(unsigned)strtoul(b_physical_extra_kib_text,NULL,10):0u;
  const char* pre_b_shrink_kib_text=getenv("ORDERED_TENSOR_PRE_B_SHRINK_KIB");
  unsigned pre_b_shrink_kib=pre_b_shrink_kib_text?(unsigned)strtoul(pre_b_shrink_kib_text,NULL,10):0u;
  const char* resource27_grow_kib_text=getenv("ORDERED_TENSOR_RESOURCE27_GROW_KIB");
  unsigned resource27_grow_kib=resource27_grow_kib_text?(unsigned)strtoul(resource27_grow_kib_text,NULL,10):0u;
  const char* resource27_alias_mode=getenv("ORDERED_TENSOR_RESOURCE27_ALIAS_MODE");
  if(resource27_alias_mode && strcmp(resource27_alias_mode,"zero") &&
     strcmp(resource27_alias_mode,"mirror") &&
     strcmp(resource27_alias_mode,"redirect") &&
     strcmp(resource27_alias_mode,"front"))return 2;
  int b_tail_metadata_copy=getenv("ORDERED_TENSOR_B_TAIL_METADATA_COPY")!=NULL;
  int alloc17_metadata_copy=getenv("ORDERED_TENSOR_ALLOC17_METADATA_COPY")!=NULL;
  int tensor_double_logical_b=getenv("ORDERED_TENSOR_DOUBLE_LOGICAL_B")!=NULL;
  int tensor_triple_logical_b=getenv("ORDERED_TENSOR_TRIPLE_LOGICAL_B")!=NULL;
  int tensor_quadruple_logical_b=getenv("ORDERED_TENSOR_QUADRUPLE_LOGICAL_B")!=NULL;
  int tensor_octuple_logical_b=getenv("ORDERED_TENSOR_OCTUPLE_LOGICAL_B")!=NULL;
  int tensor_sixteenfold_logical_b=getenv("ORDERED_TENSOR_SIXTEENFOLD_LOGICAL_B")!=NULL;
  const char* addressed_b_trips_text=getenv("ORDERED_TENSOR_ADDRESSED_B_TRIPS");
  int tensor_addressed_16320=addressed_b_trips_text && !strcmp(addressed_b_trips_text,"16320");
  if(addressed_b_trips_text && !tensor_addressed_16320)return 2;
  if(tensor_double_logical_b && (!tensor_sg4h12loop8160runtime ||
                                  !alloc17_metadata_copy || b_physical_mib))return 2;
  if(tensor_triple_logical_b && (tensor_double_logical_b ||
                                  !tensor_sg4h12loop8160runtime ||
                                  !alloc17_metadata_copy || b_physical_mib))return 2;
  if(tensor_quadruple_logical_b && (tensor_double_logical_b || tensor_triple_logical_b ||
                                     !tensor_sg4h12loop8160runtime ||
                                     !alloc17_metadata_copy || b_physical_mib))return 2;
  if(tensor_octuple_logical_b && (tensor_double_logical_b || tensor_triple_logical_b ||
                                   tensor_quadruple_logical_b ||
                                   !tensor_sg4h12loop8160runtime ||
                                   !alloc17_metadata_copy || b_physical_mib))return 2;
  if(tensor_sixteenfold_logical_b && (tensor_double_logical_b || tensor_triple_logical_b ||
                                      tensor_quadruple_logical_b || tensor_octuple_logical_b ||
                                      !tensor_sg4h12loop8160runtime ||
                                      !alloc17_metadata_copy || b_physical_mib))return 2;
  if(tensor_addressed_16320 && (tensor_double_logical_b || tensor_triple_logical_b ||
                                 tensor_quadruple_logical_b || tensor_octuple_logical_b ||
                                 tensor_sixteenfold_logical_b ||
                                 !tensor_sg4h12loop8160runtime ||
                                 !alloc17_metadata_copy || b_physical_mib))return 2;
  const char* alloc25_lookup_arm=getenv("ORDERED_TENSOR_ALLOC25_LOOKUP_ARM");
  if(alloc25_lookup_arm && strcmp(alloc25_lookup_arm,"mirror") &&
     strcmp(alloc25_lookup_arm,"empty") &&
     strcmp(alloc25_lookup_arm,"b_tail") &&
     strcmp(alloc25_lookup_arm,"alias0048") &&
     strcmp(alloc25_lookup_arm,"alias0048_zero_shader") &&
     strcmp(alloc25_lookup_arm,"alias0048_witness") &&
     strcmp(alloc25_lookup_arm,"tailffff_witness") &&
     strcmp(alloc25_lookup_arm,"tailfffe_witness") &&
     strcmp(alloc25_lookup_arm,"taildual_fffd_tensor") &&
     strcmp(alloc25_lookup_arm,"taildual_ffff_witness"))return 2;
  if(alloc25_lookup_arm && !alloc17_metadata_copy)return 2;
  if(alloc17_metadata_copy && (!tensor_sg4h12loop8160runtime || b_physical_mib ||
                               b_tail_metadata_copy || resource27_alias_mode ||
                               resource27_grow_kib))return 2;
  int tensor_sg4h12fused16=getenv("ORDERED_TENSOR_SG4H12FUSED16")!=NULL;
  int tensor_far=getenv("ORDERED_TENSOR_FAR")!=NULL;
  int tensor_farfast=getenv("ORDERED_TENSOR_FARFAST")!=NULL;
  int tensor_farunfold=getenv("ORDERED_TENSOR_FARUNFOLD")!=NULL;
  int tensor_far_noextra=getenv("ORDERED_TENSOR_FAR_NOEXTRA")!=NULL;
  const char* alloc24_request_arm=getenv("VALID_ALLOC24_REQUEST_ARM");
  if(!path)return 2;
  FILE* f=fopen(path,"rb");if(!f){perror("ORDERED fopen");return 2;}
  uint8_t* mapped[30]={0};
  uint8_t kinds[30]={0};uint64_t sizes[30]={0};
  int residual_row29=getenv("ORDERED_FFN_RESIDUAL_ROW29")!=NULL;
  if(residual_row29 && (tensor_farunfold || !getenv("ORDERED_FFN_RESIDUAL_KEEP")))return 2;
  for(unsigned expected=0;expected<(tensor_farunfold || residual_row29?30u:29u);expected++){
    uint8_t blob[144];
    if(fread(blob,1,sizeof blob,f)!=sizeof blob){fprintf(stderr,"ORDERED REFUSED: short row %u\n",expected);fclose(f);return 2;}
    uint32_t index=0,kind=0,parent=0,reserved=0;
    uint64_t offset=0,aperture=0,size=0;
    memcpy(&index,blob,4);memcpy(&kind,blob+4,4);memcpy(&parent,blob+8,4);
    memcpy(&reserved,blob+12,4);memcpy(&offset,blob+16,8);
    memcpy(&aperture,blob+24,8);memcpy(&size,blob+32,8);
    uint8_t request[104];memcpy(request,blob+40,sizeof request);
    if(index==24 && alloc24_request_arm){
      uint32_t type=0;memcpy(&type,request+0x58,4);
      if((strcmp(alloc24_request_arm,"code") && strcmp(alloc24_request_arm,"data")) ||
         type!=(strcmp(alloc24_request_arm,"code")==0?0x08000000u:0u)){
        fprintf(stderr,"ORDERED REFUSED: allocation-24 request arm\n");fclose(f);return 2;
      }
      fprintf(stderr,"ORDERED allocation-24 request arm=%s type=0x%08x checked before selector-9.\n",
              alloc24_request_arm,type);
    }
    if(index!=expected || kind>2 || reserved || size==0){fprintf(stderr,"ORDERED REFUSED: metadata row %u\n",expected);fclose(f);return 2;}
    kinds[index]=(uint8_t)kind;sizes[index]=size;
    if(kind==1){
      static const uint8_t zero16[16]={0};
      if(parent>=index || !mapped[parent] || offset>=size ||
         memcmp(request+56,zero16,16) || memcmp(request+96,zero16,8)){
        fprintf(stderr,"ORDERED REFUSED: view row %u\n",expected);fclose(f);return 2;
      }
      uint64_t base=(uint64_t)(uintptr_t)mapped[parent],view=base+offset;
      memcpy(request+56,&view,8);memcpy(request+64,&base,8);
    } else if(parent || offset){fprintf(stderr,"ORDERED REFUSED: non-view row %u\n",expected);fclose(f);return 2;}
    uint64_t o[11]={0};size_t n=sizeof o;
    kern_return_t k=IOConnectCallMethod(CONN,9,0,0,request,sizeof request,0,0,o,&n);
    fprintf(stderr,"ORDERED index=%u kind=%u kr=0x%x bytes=%zu aperture=0x%llx expected=0x%llx host=0x%llx size=0x%llx\n",
            index,kind,k,n,(unsigned long long)o[0],(unsigned long long)aperture,
            (unsigned long long)o[1],(unsigned long long)o[5]);
    if(k || n!=88 || o[0]!=aperture || o[5]!=size ||
       (kind==1?o[1]!=0 || !o[2]:!o[1])){
      fprintf(stderr,"ORDERED REFUSED: call %u output mismatch\n",expected);fclose(f);return 2;
    }
    if(kind!=1)mapped[index]=(uint8_t*)(uintptr_t)o[1];
    if(!no_agx_metal()){fclose(f);return 2;}
  }
  if(fgetc(f)!=EOF || ferror(f)){fprintf(stderr,"ORDERED REFUSED: trailing template\n");fclose(f);return 2;}
  fclose(f);
  if(getenv("ORDERED_FFN_APPEND_OUTPUT_BASE")){
    uint8_t request[104]={0};
    const uint32_t word08=0x10001u,one=1u,flags=0x1000101u,
                   kind=0x470u;
    const uint64_t length64=0x10000ull;
    memcpy(request+8,&word08,4);memcpy(request+12,&one,4);
    memcpy(request+16,&flags,4);memcpy(request+20,&kind,4);
    memcpy(request+0x30,&one,4);memcpy(request+0x48,&length64,8);
    uint64_t out[11]={0};size_t out_count=sizeof out;
    kern_return_t kr=IOConnectCallMethod(CONN,9,0,0,request,sizeof request,
                                        0,0,out,&out_count);
    fprintf(stderr,"PURE FFN APPENDED OUTPUT: index=29 kr=0x%x bytes=%zu aperture=0x%llx size=0x%llx host=0x%llx.\n",
            kr,out_count,(unsigned long long)out[0],
            (unsigned long long)out[5],(unsigned long long)out[1]);
    if(kr || out_count!=88 || out[0]!=0x100000f0000ull ||
       out[5]!=0x10000ull || !out[1] || !no_agx_metal())return 2;
    mapped[29]=(uint8_t*)(uintptr_t)out[1];sizes[29]=out[5];kinds[29]=0;
  }
  if(getenv("ORDERED_FFN_STANDALONE_N1_RESIDENT")){
    if(residual_row29){
      if(!mapped[29] || sizes[29]!=0x40000u || kinds[29]!=0)return 2;
      fprintf(stderr,"PURE STANDALONE N1 RESIDENT BASE: index=29 row=template aperture=0x100000f0000 size=0x40000.\n");
    }else{
      if(mapped[29])return 2;
      uint8_t request[104]={0};
      const uint32_t word08=0x10001u,one=1u,flags=0x1000101u,kind=0x470u;
      const uint64_t length64=getenv("ORDERED_FFN_N1_ALLOC_64K")?0x10000ull:0x40000ull;
      memcpy(request+8,&word08,4);memcpy(request+12,&one,4);
      memcpy(request+16,&flags,4);memcpy(request+20,&kind,4);
      memcpy(request+0x30,&one,4);memcpy(request+0x48,&length64,8);
      uint64_t out[11]={0};size_t out_count=sizeof out;
      kern_return_t kr=IOConnectCallMethod(CONN,9,0,0,request,sizeof request,
                                          0,0,out,&out_count);
      fprintf(stderr,"PURE STANDALONE N1 RESIDENT BASE: index=29 kr=0x%x bytes=%zu aperture=0x%llx size=0x%llx host=0x%llx.\n",
              kr,out_count,(unsigned long long)out[0],
              (unsigned long long)out[5],(unsigned long long)out[1]);
      if(kr || out_count!=88 || out[0]!=0x100000f0000ull ||
         out[5]!=length64 || !out[1] || !no_agx_metal())return 2;
      mapped[29]=(uint8_t*)(uintptr_t)out[1];sizes[29]=out[5];kinds[29]=0;
    }
  }
  const char* payload_path=getenv("STAGE_CAPTURED_PAYLOAD");
  if(payload_path){
    FILE* payload=fopen(payload_path,"rb");if(!payload){perror("ORDERED payload");return 2;}
    size_t total=0;
    for(size_t i=0;i<(tensor_graph?(tensor_farunfold || residual_row29?30u:29u):sizeof G17_LAYOUT_REQUESTS/sizeof G17_LAYOUT_REQUESTS[0]);i++){
      unsigned index=tensor_graph?(unsigned)i:G17_LAYOUT_REQUESTS[i].original_index;
      if(tensor_graph && kinds[index]==1)continue;
      size_t allocation_size=tensor_graph?(size_t)sizes[index]:G17_LAYOUT_REQUESTS[i].size;
      uint8_t* cpu=mapped[index];
      if(!cpu){fprintf(stderr,"ORDERED REFUSED: physical mapping %u absent\n",index);fclose(payload);return 2;}
      uint8_t* expected=malloc(allocation_size);if(!expected){fclose(payload);return 2;}
      if(fread(expected,1,allocation_size,payload)!=allocation_size){fprintf(stderr,"ORDERED REFUSED: short payload\n");free(expected);fclose(payload);return 2;}
      memcpy(cpu,expected,allocation_size);
      int match=memcmp(cpu,expected,allocation_size)==0;
      free(expected);
      if(!match){fprintf(stderr,"ORDERED REFUSED: readback %u\n",index);fclose(payload);return 2;}
      total+=allocation_size;
    }
    if(fgetc(payload)!=EOF || ferror(payload)){fprintf(stderr,"ORDERED REFUSED: trailing payload\n");fclose(payload);return 2;}
    fclose(payload);
    uint32_t packet=0;memcpy(&packet,mapped[tensor_farunfold?24:23]+0x40,4);
    if(getenv("ORDERED_TG_ONE")){
      int tg_ab=getenv("ORDERED_TG_AB")!=NULL;
      int tg_upper=getenv("ORDERED_TG_UPPER")!=NULL;
      int tg_dual_bank=getenv("ORDERED_TG_DUAL_BANK")!=NULL;
      int tg_high_1024=getenv("ORDERED_TG_HIGH_1024")!=NULL;
      int tg_four_bank=getenv("ORDERED_TG_FOUR_BANK")!=NULL;
      int tg_readout=getenv("ORDERED_TG_READOUT")!=NULL;
      int tg_high_2048=getenv("ORDERED_TG_HIGH_2048")!=NULL;
      const char* tg_half_2048=getenv("ORDERED_TG_HALF_2048");
      int tg_resource_2048=getenv("ORDERED_TG_RESOURCE_2048")!=NULL;
      int tg_scrambled=getenv("ORDERED_TG_B_SCRAMBLED")!=NULL;
      int tg_two_groups=getenv("ORDERED_TG_TWO_GROUPS")!=NULL;
      int tg_three_groups=getenv("ORDERED_TG_THREE_GROUPS")!=NULL;
      int tg_y_two=getenv("ORDERED_TG_Y_TWO")!=NULL;
      int tg_z_two=getenv("ORDERED_TG_Z_TWO")!=NULL;
      int tg_xy_two=getenv("ORDERED_TG_XY_TWO")!=NULL;
      int tg_partial96=getenv("ORDERED_TG_PARTIAL96")!=NULL;
      int tg_full64=getenv("ORDERED_TG_FULL64")!=NULL;
      int tg_full32=getenv("ORDERED_TG_FULL32")!=NULL;
      int tensor_stage=getenv("ORDERED_TENSOR_STAGE")!=NULL;
      int tensor_code_alloc0=getenv("ORDERED_TENSOR_CODE_ALLOC0")!=NULL;
      int tensor_witness=getenv("ORDERED_TENSOR_WITNESS")!=NULL;
      int tg_partial_tg96=getenv("ORDERED_TG_PARTIAL_TG96")!=NULL;
      int tg_full_tg64=getenv("ORDERED_TG_FULL_TG64")!=NULL;
      int tg_partial_tg80=getenv("ORDERED_TG_PARTIAL_TG80")!=NULL;
      int tg_full_tg80_control=getenv("ORDERED_TG_FULL_TG80_CONTROL")!=NULL;
      unsigned tg_param_grid=0,tg_param_xor=0;
      int tg_param=ordered_tg_param(&tg_param_grid,&tg_param_xor);
      if(tg_param<0)return 2;
      const char* code_path=getenv("ORDERED_TG_CODE");
      static const uint8_t resource_256[8]={0x0f,0x20,0x00,0x0c,0,0,0,0};
      static const uint8_t resource_zero[8]={0x07,0,0,0x0c,0,0,0,0};
      static const uint8_t resource_512[8]={0x0f,0x40,0x00,0x0c,0,0,0,0};
      static const uint8_t resource_1024[8]={0x0f,0x80,0x00,0x0c,0,0,0,0};
      static const uint8_t resource_2048[8]={0x0f,0x00,0x08,0x0c,0,0,0,0};
      const uint8_t* resource_word=resource_256;
      size_t code_bytes=tg_ab?124u:98u;
      unsigned scratch_bytes=256u;
      if(tg_upper || tg_dual_bank){resource_word=resource_512;scratch_bytes=512u;}
      if(tg_upper)code_bytes=148u;
      if(tg_dual_bank)code_bytes=172u;
      if(tg_high_1024 || tg_four_bank || tg_readout){resource_word=resource_1024;scratch_bytes=1024u;}
      if(tg_high_1024)code_bytes=208u;
      if(tg_four_bank)code_bytes=368u;
      if(tg_readout)code_bytes=440u;
      if(tg_high_2048){resource_word=resource_2048;scratch_bytes=2048u;code_bytes=260u;}
      if(tg_half_2048){resource_word=resource_2048;scratch_bytes=2048u;
                       code_bytes=strcmp(tg_half_2048,"lower")==0?664u:708u;}
      if(tg_two_groups || tg_three_groups)code_bytes=128u;
      if(tg_y_two || tg_z_two || tg_xy_two)code_bytes=158u;
      if(tg_partial96 || tg_full64 || tg_full32){resource_word=resource_zero;scratch_bytes=0u;code_bytes=56u;}
      if(tensor_stage){resource_word=resource_zero;scratch_bytes=0u;code_bytes=(tensor_farunfold||tensor_far_noextra)?10404u:(tensor_far?(tensor_farfast?5132u:5980u):(tensor_sg4h12loop137700reuse?62580u:tensor_sg4h12loop73728reuse?62580u:tensor_sg4h12loop36720reuse?62620u:tensor_sg4h12loop18360reuse?62700u:tensor_sg4h12loop9180reuse?62860u:tensor_sg4h12loop8160reuse?56220u:tensor_sg4h12loop7650runtime?52900u:tensor_sg4h12loop6120runtime?42880u:tensor_sg4h12loop8160runtime?(tensor_sg4h12loop8160codecontrol?56174u:56240u):tensor_sg4h12loop4080sixteen?56174u:tensor_sg4h12loop2040oct?29454u:tensor_sg4h12loop1275quint?19434u:tensor_sg4h12loop1020quad?16094u:tensor_sg4h12loop765triple?12754u:(tensor_sg4h12loop256tail || tensor_sg4h12loop510dual)?9414u:tensor_sg4h?6074u:(tensor_sg4?6034u:(tensor_loopscale?5092u:(tensor_loop?5940u:(tensor_common?1232u:(tensor_witness?80u:648u)))))));}
      if(tensor_sg4h12fused16)code_bytes=5226u;
      if(tensor_sg4h12loop256tail || tensor_sg4h12loop510dual)code_bytes=9414u;
      if(tensor_sg4h12loop765triple)code_bytes=12754u;
      if(tensor_sg4h12loop1020quad)code_bytes=16094u;
      if(tensor_sg4h12loop1275quint)code_bytes=19434u;
      if(tensor_sg4h12loop2040oct)code_bytes=29454u;
      if(tensor_sg4h12loop9180reuse)code_bytes=62860u;
      if(tensor_sg4h12loop18360reuse)code_bytes=62700u;
      if(tensor_sg4h12loop36720reuse)code_bytes=62620u;
      if(tensor_sg4h12loop73728reuse)code_bytes=62580u;
      if(tensor_sg4h12loop137700reuse)code_bytes=62580u;
      if(tensor_sg4h12loop137700reuse && getenv("ORDERED_TENSOR_FUSED24"))code_bytes=62148u;
      if(tensor_sg4h12loop137700reuse && getenv("ORDERED_TENSOR_STATIONARY8192"))code_bytes=62128u;
      if(tensor_sg4h12loop137700reuse && getenv("ORDERED_TENSOR_STATIONARY_COUNT_WITNESS"))code_bytes=62124u;
      if(tensor_sg4h12loop8160reuse)code_bytes=56220u;
      if(tensor_sg4h12loop7650runtime)code_bytes=52900u;
      if(tensor_sg4h12loop6120runtime)code_bytes=42880u;
      if(tensor_sg4h12loop8160runtime)code_bytes=tensor_sg4h12loop8160codecontrol?56174u:56240u;
      if(tensor_sg4h12loop4080sixteen)code_bytes=56174u;
      if(tg_partial_tg96 || tg_full_tg64 || tg_partial_tg80 || tg_full_tg80_control || tg_param)code_bytes=128u;
      if(tg_resource_2048){resource_word=resource_2048;scratch_bytes=2048u;}
      uint8_t authored[65536]={0};
      if(code_bytes>sizeof authored){
        fprintf(stderr,"ORDERED TG REFUSED: code exceeds verifier buffer\n");return 2;
      }
      if(!code_path || ((tg_upper||tg_dual_bank||tg_high_1024||tg_four_bank||tg_readout||tg_high_2048||tg_half_2048) && !tg_ab) ||
         (tg_half_2048 && strcmp(tg_half_2048,"lower") && strcmp(tg_half_2048,"upper")) ||
         (tg_resource_2048 && !tg_readout) ||
         (tg_upper+tg_dual_bank+tg_high_1024+tg_four_bank+tg_readout+tg_high_2048+(tg_half_2048!=NULL)+tg_two_groups+tg_three_groups+tg_y_two+tg_z_two+tg_xy_two+tg_partial96+tg_full64+tg_full32+tg_partial_tg96+tg_full_tg64+tg_partial_tg80+tg_full_tg80_control+tg_param+tensor_stage>1) ||
         ((tg_two_groups || tg_three_groups || tg_y_two || tg_z_two || tg_xy_two || tg_partial96 || tg_full64 || tg_full32 || tg_partial_tg96 || tg_full_tg64 || tg_partial_tg80 || tg_full_tg80_control || tg_param || tensor_stage) && (!tg_ab || tg_resource_2048)) ||
         (tg_scrambled && !tg_ab) ||
         getenv("ORDERED_ATOMIC_ONE") ||
         getenv("ORDERED_UNIFORM_ATOMIC_ONE") || getenv("ORDERED_VALID_ONE") ||
         getenv("ORDERED_VALID_TWO") || getenv("ORDERED_VALID_BATCH_TWO"))return 2;
      FILE* cf=fopen(code_path,"rb");if(!cf){perror("ORDERED TG code");return 2;}
      int code_ok=fread(authored,1,code_bytes,cf)==code_bytes &&
                  fgetc(cf)==EOF && !ferror(cf);
      fclose(cf);
      uint32_t packet_word=0;
      uint16_t packet_mid=0,packet_high=0;
      uint64_t binding_a=0,binding_b=0,binding_c=0;
      unsigned resource_index=tensor_farunfold?24u:23u;
      unsigned binding_index=tensor_farunfold?29u:28u;
      unsigned geometry_index=tensor_farunfold?23u:22u;
      unsigned extent_index=tensor_farunfold?26u:25u;
      memcpy(&packet_word,mapped[resource_index]+0x40,4);
      memcpy(&packet_mid,mapped[resource_index]+0x46,2);
      memcpy(&packet_high,mapped[resource_index]+0x48,2);
      memcpy(&binding_a,mapped[binding_index]+0x1ba0,8);
      memcpy(&binding_b,mapped[binding_index]+0x1ba8,8);
      memcpy(&binding_c,mapped[binding_index]+0x1bb0,8);
      unsigned expected_total=tensor_farunfold?1179648u:tensor_far?917504u:(tensor_sg4h12loop9180reuse||tensor_sg4h12loop18360reuse||tensor_sg4h12loop36720reuse||tensor_sg4h12loop73728reuse||tensor_sg4h12loop137700reuse)?32686080u:tensor_sg4h12loop8160reuse?32686080u:tensor_sg4h12loop7650runtime?32686080u:tensor_sg4h12loop6120runtime?28491776u:tensor_sg4h12loop8160runtime?36880384u:tensor_sg4h12loop4080sixteen?19054592u:tensor_sg4h12loop2040oct?10665984u:tensor_sg4h12loop1275quint?6471680u:tensor_sg4h12loop1020quad?5423104u:tensor_sg4h12loop765triple?4374528u:tensor_sg4h12loop510dual?3325952u:tensor_sg4h12loop256tail?2310144u:(tensor_sg4h12loop254 || tensor_sg4h12loop255)?2277376u:tensor_sg4h12loop136?1818624u:tensor_sg4h12loop128?1785856u:tensor_sg4h12loop64?1523712u:tensor_sg4h12loop40?1425408u:tensor_sg4h12loop32?1392640u:(tensor_sg4h12loop31 || tensor_sg4h12loop24)?1359872u:tensor_sg4h12loop16?1327104u:tensor_sg4h12?1261568u:tensor_sg4h10?1179648u:tensor_sg4h9?1130496u:tensor_sg4h8?1097728u:tensor_sg4h7?1048576u:tensor_sg4h6?1015808u:tensor_sg4h5?966656u:tensor_sg4h4?933888u:tensor_sg4h3loop16?950272u:tensor_sg4hloop16?917504u:tensor_sg4h3?884736u:tensor_sg4h?851968u:tensor_sg4loop16?851968u:tensor_sg4?786432u:tensor_loop16?802816u:(tensor_loop8 || tensor_loop15)?770048u:tensor_loop?737280u:tensor_graph?704512u:638976u;
      if(tensor_double_logical_b)expected_total=68337664u;
      if(tensor_triple_logical_b)expected_total=101892096u;
      if(tensor_quadruple_logical_b)expected_total=135446528u;
      if(tensor_octuple_logical_b)expected_total=269664256u;
      if(tensor_sixteenfold_logical_b)expected_total=538099712u;
      if(tensor_addressed_16320)expected_total=1074970624u;
      if(b_physical_mib){
        if(!tensor_sg4h12loop4080sixteen || b_physical_mib<17u ||
           (b_physical_mib>34u && b_physical_mib!=48u && b_physical_mib!=64u))return 2;
        if(post_b_gap_mib && (post_b_gap_mib!=1u || b_physical_mib!=30u))return 2;
        if(b_physical_extra_kib && (b_physical_mib!=30u || post_b_gap_mib ||
                                    b_physical_extra_kib>=1024u || b_physical_extra_kib%64u))return 2;
        if(pre_b_shrink_kib && (pre_b_shrink_kib!=64u || b_physical_mib!=30u ||
                                (b_physical_extra_kib!=512u && b_physical_extra_kib!=576u &&
                                 b_physical_extra_kib!=640u) ||
                                post_b_gap_mib))return 2;
        if(resource27_grow_kib && ((resource27_grow_kib!=64u && resource27_grow_kib!=96u) ||
                                   b_physical_mib!=30u ||
                                   (b_physical_extra_kib!=512u && b_physical_extra_kib!=576u &&
                                    b_physical_extra_kib!=640u) ||
                                   pre_b_shrink_kib!=64u ||
                                   post_b_gap_mib))return 2;
        if(resource27_alias_mode &&
           (!strcmp(resource27_alias_mode,"front")?
            (resource27_grow_kib!=64u || b_physical_extra_kib!=640u):
            (resource27_grow_kib!=96u || b_physical_extra_kib!=576u)))return 2;
        if(resource27_grow_kib==96u && !resource27_alias_mode)return 2;
        if(resource27_grow_kib && b_physical_extra_kib==640u &&
           (!resource27_alias_mode || strcmp(resource27_alias_mode,"front")))return 2;
        if(b_tail_metadata_copy &&
           (resource27_alias_mode || resource27_grow_kib || post_b_gap_mib ||
            !((b_physical_mib==30u && b_physical_extra_kib==640u &&
               pre_b_shrink_kib==64u) ||
              ((b_physical_mib==34u || b_physical_mib==48u || b_physical_mib==64u) &&
               !b_physical_extra_kib && !pre_b_shrink_kib))))return 2;
        if((b_physical_mib==48u || b_physical_mib==64u) && !b_tail_metadata_copy)return 2;
        expected_total=19054592u+((b_physical_mib-17u)<<20)+
                       (b_physical_extra_kib<<10)-(pre_b_shrink_kib<<10)+
                       (resource27_grow_kib<<10);
      }
      if(residual_row29)expected_total+=0x40000u;
      if(!code_ok || total!=expected_total ||
         (tensor_common && (!tensor_graph || !tensor_stage || !tensor_code_alloc0)) ||
         (tensor_loop && (!tensor_graph || !tensor_stage || !tensor_code_alloc0 || tensor_common)) ||
         (tensor_loopscale && !tensor_loop) ||
         (tensor_sg4 && (!tensor_loop || tensor_loopscale)) ||
         (tensor_sg4h && (!tensor_loop || tensor_loopscale || tensor_sg4)) ||
         (tensor_sg4hloop16 && !tensor_sg4h) ||
         (tensor_sg4h3loop16 && (!tensor_sg4hloop16 || !tensor_sg4h3)) ||
         (tensor_sg4h12loop16 && (!tensor_sg4hloop16 || !tensor_sg4h12)) ||
         (tensor_sg4h12loop17 && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop23 && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop24 && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop31 && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop32 && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop40 && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop64 && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop128 && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop136 && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop254 && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop255 && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop256tail && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop510dual && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop765triple && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop1020quad && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop1275quint && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop2040oct && !tensor_sg4h12loop16) ||
         ((tensor_sg4h12loop9180reuse||tensor_sg4h12loop18360reuse||tensor_sg4h12loop36720reuse||tensor_sg4h12loop73728reuse||tensor_sg4h12loop137700reuse) && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop8160reuse && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop7650runtime && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop6120runtime && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop8160runtime && !tensor_sg4h12loop16) ||
         (tensor_sg4h12loop4080sixteen && !tensor_sg4h12loop16) ||
         (tensor_sg4h12fused16 && (!tensor_sg4h12loop16 || !tensor_stage)) ||
         (tensor_sg4h3 && !tensor_sg4h) ||
         (tensor_sg4h4 && (!tensor_sg4h || tensor_sg4h3)) ||
         (tensor_sg4h5 && (!tensor_sg4h || tensor_sg4h3 || tensor_sg4h4)) ||
         (tensor_sg4h6 && (!tensor_sg4h || tensor_sg4h3 || tensor_sg4h4 || tensor_sg4h5)) ||
         (tensor_sg4h7 && (!tensor_sg4h || tensor_sg4h3 || tensor_sg4h4 || tensor_sg4h5 || tensor_sg4h6)) ||
         (tensor_sg4h8 && (!tensor_sg4h || tensor_sg4h3 || tensor_sg4h4 || tensor_sg4h5 || tensor_sg4h6 || tensor_sg4h7)) ||
         (tensor_sg4h9 && (!tensor_sg4h || tensor_sg4h3 || tensor_sg4h4 || tensor_sg4h5 || tensor_sg4h6 || tensor_sg4h7 || tensor_sg4h8)) ||
         (tensor_sg4h10 && (!tensor_sg4h || tensor_sg4h3 || tensor_sg4h4 || tensor_sg4h5 || tensor_sg4h6 || tensor_sg4h7 || tensor_sg4h8 || tensor_sg4h9)) ||
         (tensor_sg4h12 && (!tensor_sg4h || tensor_sg4h3 || tensor_sg4h4 || tensor_sg4h5 || tensor_sg4h6 || tensor_sg4h7 || tensor_sg4h8 || tensor_sg4h9 || tensor_sg4h10)) ||
         (tensor_far && (!tensor_loop || tensor_loopscale || tensor_sg4 || tensor_sg4h)) ||
         (tensor_farfast && !tensor_far) ||
         (tensor_farunfold && (!tensor_far || tensor_farfast)) ||
         (tensor_far_noextra && (!tensor_far || tensor_farfast || tensor_farunfold)) ||
         (tensor_graph && (!tensor_stage || !tensor_code_alloc0)) ||
         (tensor_code_alloc0 && (!tensor_stage || tensor_witness)) ||
         memcmp(tensor_code_alloc0?mapped[0]+0x6c0:mapped[20]+0x600,authored,code_bytes) ||
         (tensor_code_alloc0 && !tensor_graph && memcmp(mapped[20]+0x600,(uint8_t[648]){0},648)) ||
         memcmp(mapped[resource_index]+0x38,resource_word,8) ||
         packet_word!=(tensor_stage?(tensor_code_alloc0?0x0e5806c7u:0x0e588607u):0x0e408607u) ||
         packet_mid!=(tensor_code_alloc0?0x0000u:0x0005u) ||
         packet_high!=0x0100u ||
         binding_a!=(((tensor_far || tensor_sg4h)?0x10000080080ull:(tensor_graph?0x10000034d80ull:0x10000030000ull))-((uint64_t)pre_b_shrink_kib<<10)) ||
         binding_b!=(((tensor_sg4h10 || tensor_sg4h12)?0x100000a0080ull:tensor_far?0x10000098080ull:((tensor_sg4h6 || tensor_sg4h7 || tensor_sg4h8 || tensor_sg4h9)?0x10000098080ull:(tensor_sg4h?0x10000090080ull:(tensor_loop?0x10000080080ull:(tensor_common?0x10000035180ull:(tensor_graph?0x10000035080ull:0x10000030400ull))))))-((uint64_t)pre_b_shrink_kib<<10)) ||
         binding_c!=(tensor_addressed_16320?0x100400a8080ull:tensor_sixteenfold_logical_b?0x100200a8080ull:tensor_octuple_logical_b?0x100100a8080ull:tensor_quadruple_logical_b?0x100080a8080ull:tensor_triple_logical_b?0x100060a8080ull:tensor_double_logical_b?0x100040a8080ull:b_physical_mib?(0x100011a8080ull+((uint64_t)(b_physical_mib-17u+post_b_gap_mib)<<20)+((uint64_t)b_physical_extra_kib<<10)-((uint64_t)pre_b_shrink_kib<<10)):(tensor_sg4h12loop9180reuse||tensor_sg4h12loop18360reuse||tensor_sg4h12loop36720reuse||tensor_sg4h12loop73728reuse||tensor_sg4h12loop137700reuse)?0x10001ea8080ull:tensor_sg4h12loop8160reuse?0x10001ea8080ull:tensor_sg4h12loop7650runtime?0x10001ea8080ull:tensor_sg4h12loop6120runtime?0x10001aa8080ull:tensor_sg4h12loop8160runtime?0x100022a8080ull:tensor_sg4h12loop4080sixteen?0x100011a8080ull:tensor_sg4h12loop2040oct?0x100009a8080ull:tensor_sg4h12loop1275quint?0x100005a8080ull:tensor_sg4h12loop1020quad?0x100004a8080ull:tensor_sg4h12loop765triple?0x100003a8080ull:tensor_sg4h12loop510dual?0x100002a8080ull:tensor_sg4h12loop256tail?0x100001b0080ull:(tensor_sg4h12loop254 || tensor_sg4h12loop255)?0x100001a8080ull:tensor_sg4h12loop136?0x10000138080ull:tensor_sg4h12loop128?0x10000130080ull:tensor_sg4h12loop64?0x100000f0080ull:tensor_sg4h12loop40?0x100000d8080ull:tensor_sg4h12loop32?0x100000d0080ull:(tensor_sg4h12loop31 || tensor_sg4h12loop24)?0x100000c8080ull:tensor_sg4h12loop16?0x100000c0080ull:(tensor_sg4hloop16 || tensor_sg4h10 || tensor_sg4h12)?0x100000b0080ull:tensor_far?0x100000b0080ull:((tensor_sg4h6 || tensor_sg4h7 || tensor_sg4h8 || tensor_sg4h9)?0x100000a8080ull:((tensor_sg4h || tensor_sg4loop16)?0x100000a0080ull:(tensor_sg4?0x10000090080ull:(tensor_loop?0x10000035680ull:(tensor_common?0x10000035580ull:(tensor_graph?0x10000035380ull:0x10000030800ull))))))) || !no_agx_metal()){
        fprintf(stderr,"PURE TG RESOURCE REFUSED: code, resource word, bindings, or mapping\n");return 2;
      }
      if(resource27_alias_mode && strcmp(resource27_alias_mode,"zero") &&
         strcmp(resource27_alias_mode,"front") &&
         (!mapped[27] || !mapped[28] || sizes[27]!=0x20000u || sizes[28]!=0xc000u ||
          memcmp(mapped[27]+0x10000,mapped[28],0xc000u))){
        fprintf(stderr,"PURE TG RESOURCE REFUSED: allocation-27 metadata mirror\n");return 2;
      }
      if(resource27_alias_mode && !strcmp(resource27_alias_mode,"zero")){
        if(!mapped[27] || sizes[27]!=0x20000u)return 2;
        for(size_t i=0;i<0x20000u;i++)if(mapped[27][i])return 2;
      }
      if(resource27_alias_mode && !strcmp(resource27_alias_mode,"front") &&
         (!mapped[27] || !mapped[28] || sizes[27]!=0x18000u || sizes[28]!=0xc000u ||
          memcmp(mapped[27],mapped[28],0xc000u))){
        fprintf(stderr,"PURE TG RESOURCE REFUSED: allocation-27 front metadata copy\n");return 2;
      }
      if(b_tail_metadata_copy &&
         (!mapped[20] || !mapped[28] || sizes[20]<0x100c000u || sizes[28]!=0xc000u ||
          memcmp(mapped[20]+0x1000000,mapped[28],0xc000u))){
        fprintf(stderr,"PURE TG RESOURCE REFUSED: B-tail metadata copy\n");return 2;
      }
      if(alloc17_metadata_copy &&
         (!mapped[17] || !mapped[28] || sizes[17]!=0x20000u || sizes[28]!=0xc000u ||
          memcmp(mapped[17],mapped[28],0xc000u))){
        fprintf(stderr,"PURE TG RESOURCE REFUSED: allocation-17 metadata copy\n");return 2;
      }
      if(alloc25_lookup_arm){
        uint16_t coordinate=0;
        if(!mapped[17] || !mapped[23] || !mapped[25] ||
           sizes[17]!=0x20000u || sizes[23]!=0x8000u || sizes[25]!=0x8000u)return 2;
        memcpy(&coordinate,mapped[25]+9,2);
        uint16_t expected_coordinate=(!strcmp(alloc25_lookup_arm,"taildual_fffd_tensor")?0xfffdu:
                                      !strcmp(alloc25_lookup_arm,"taildual_ffff_witness")?0xffffu:
                                      !strcmp(alloc25_lookup_arm,"tailfffe_witness")?0xfffeu:
                                      !strcmp(alloc25_lookup_arm,"tailffff_witness")?0xffffu:
                                      !strcmp(alloc25_lookup_arm,"b_tail")?0xfff0u:
                                      !strcmp(alloc25_lookup_arm,"alias0048") ||
                                      !strcmp(alloc25_lookup_arm,"alias0048_zero_shader") ||
                                      !strcmp(alloc25_lookup_arm,"alias0048_witness")?0x0048u:0x001au);
        if(coordinate!=expected_coordinate){
          fprintf(stderr,"PURE TG RESOURCE REFUSED: allocation-25 coordinate\n");return 2;
        }
        if(!strcmp(alloc25_lookup_arm,"b_tail")){
          if(!mapped[20] || sizes[20]!=0x40000000u ||
             memcmp(mapped[20]+0x3ff20000u,mapped[23],0x8000u)!=0){
            fprintf(stderr,"PURE TG RESOURCE REFUSED: allocation-23 B-tail target\n");return 2;
          }
        }else if(!strcmp(alloc25_lookup_arm,"taildual_fffd_tensor") ||
                 !strcmp(alloc25_lookup_arm,"taildual_ffff_witness")){
          const uint8_t alternate_packet[10]={0x07,0xf0,0x58,0x0e,0,0,0,0,0,1};
          if(!mapped[0] || sizes[0]!=0x10000u || !mapped[20] ||
             sizes[20]!=0x40000000u ||
             memcmp(mapped[20]+0x3ff54000u,mapped[23],0x8000u)!=0 ||
             memcmp(mapped[20]+0x3ff5c000u,mapped[23],0x40u)!=0 ||
             memcmp(mapped[20]+0x3ff5c040u,alternate_packet,10u)!=0 ||
             memcmp(mapped[20]+0x3ff5c04au,mapped[23]+0x4au,0x8000u-0x4au)!=0 ||
             !memcmp(mapped[0]+0xf000u,(uint8_t[80]){0},80u)){
            fprintf(stderr,"PURE TG RESOURCE REFUSED: dual B-tail records\n");return 2;
          }
        }else if(!strcmp(alloc25_lookup_arm,"tailffff_witness") ||
                 !strcmp(alloc25_lookup_arm,"tailfffe_witness")){
          const uint8_t alternate_packet[10]={0x07,0xf0,0x58,0x0e,0,0,0,0,0,1};
          size_t tail_offset=!strcmp(alloc25_lookup_arm,"tailfffe_witness")?0x3ff58000u:0x3ff5c000u;
          if(!mapped[0] || sizes[0]!=0x10000u || !mapped[20] ||
             sizes[20]!=0x40000000u ||
             memcmp(mapped[20]+tail_offset,mapped[23],0x40u)!=0 ||
             memcmp(mapped[20]+tail_offset+0x40u,alternate_packet,10u)!=0 ||
             memcmp(mapped[20]+tail_offset+0x4au,mapped[23]+0x4au,0x8000u-0x4au)!=0 ||
             !memcmp(mapped[0]+0xf000u,(uint8_t[80]){0},80u)){
            fprintf(stderr,"PURE TG RESOURCE REFUSED: competing B-tail witness\n");return 2;
          }
        }else if(!strcmp(alloc25_lookup_arm,"alias0048")){
          if(!mapped[20] || sizes[20]!=0x40000000u ||
             memcmp(mapped[20]+0x80000u,mapped[23],0x8000u)!=0){
            fprintf(stderr,"PURE TG RESOURCE REFUSED: allocation-23 low alias target\n");return 2;
          }
        }else if(!strcmp(alloc25_lookup_arm,"alias0048_zero_shader")){
          if(!mapped[20] || sizes[20]!=0x40000000u ||
             memcmp(mapped[20]+0x80000u,mapped[23],0x40u)!=0 ||
             memcmp(mapped[20]+0x80040u,(uint8_t[10]){0},10u)!=0 ||
             memcmp(mapped[20]+0x8004au,mapped[23]+0x4au,0x8000u-0x4au)!=0){
            fprintf(stderr,"PURE TG RESOURCE REFUSED: zeroed low alias LoadShader\n");return 2;
          }
        }else if(!strcmp(alloc25_lookup_arm,"alias0048_witness")){
          const uint8_t alternate_packet[10]={0x07,0xf0,0x58,0x0e,0,0,0,0,0,1};
          if(!mapped[0] || sizes[0]!=0x10000u || !mapped[20] ||
             sizes[20]!=0x40000000u ||
             memcmp(mapped[20]+0x80000u,mapped[23],0x40u)!=0 ||
             memcmp(mapped[20]+0x80040u,alternate_packet,10u)!=0 ||
             memcmp(mapped[20]+0x8004au,mapped[23]+0x4au,0x8000u-0x4au)!=0 ||
             !memcmp(mapped[0]+0xf000u,(uint8_t[80]){0},80u)){
            fprintf(stderr,"PURE TG RESOURCE REFUSED: competing low alias witness\n");return 2;
          }
        }else{
          const uint8_t* low=mapped[17]+0x10000u;
          if(!strcmp(alloc25_lookup_arm,"mirror") ? memcmp(low,mapped[23],0x8000u)!=0 :
             memcmp(low,(uint8_t[0x8000]){0},0x8000u)!=0){
            fprintf(stderr,"PURE TG RESOURCE REFUSED: allocation-23 low target\n");return 2;
          }
        }
      }
      if(tg_two_groups || tg_three_groups){
        uint32_t count=0,threads=0;
        memcpy(&count,mapped[22]+0xa8,4);memcpy(&threads,mapped[25]+0x10,4);
        if(count!=(tg_three_groups?3u:2u) || threads!=(tg_three_groups?192u:128u)){
          fprintf(stderr,"PURE TG GEOMETRY REFUSED: groups=%u threads=%u\n",count,threads);return 2;
        }
      }
      if(tg_y_two){
        uint32_t xgroups=0,ygroups=0,xthreads=0,ythreads=0;
        memcpy(&xgroups,mapped[22]+0xa8,4);memcpy(&ygroups,mapped[22]+0xac,4);
        memcpy(&xthreads,mapped[25]+0x10,4);memcpy(&ythreads,mapped[25]+0x14,4);
        if(xgroups!=1 || ygroups!=2 || xthreads!=64 || ythreads!=2){
          fprintf(stderr,"PURE TG Y GEOMETRY REFUSED: groups=%u/%u threads=%u/%u\n",
                  xgroups,ygroups,xthreads,ythreads);return 2;
        }
      }
      if(tg_z_two){
        uint32_t xgroups=0,zgroups=0,xthreads=0,zthreads=0;
        memcpy(&xgroups,mapped[22]+0xa8,4);memcpy(&zgroups,mapped[22]+0xb0,4);
        memcpy(&xthreads,mapped[25]+0x10,4);memcpy(&zthreads,mapped[25]+0x18,4);
        if(xgroups!=1 || zgroups!=2 || xthreads!=64 || zthreads!=2){
          fprintf(stderr,"PURE TG Z GEOMETRY REFUSED: groups=%u/%u threads=%u/%u\n",
                  xgroups,zgroups,xthreads,zthreads);return 2;
        }
      }
      if(tg_xy_two){
        uint32_t xgroups=0,ygroups=0,xthreads=0,ythreads=0;
        memcpy(&xgroups,mapped[22]+0xa8,4);memcpy(&ygroups,mapped[22]+0xac,4);
        memcpy(&xthreads,mapped[25]+0x10,4);memcpy(&ythreads,mapped[25]+0x14,4);
        if(xgroups!=2 || ygroups!=2 || xthreads!=128 || ythreads!=2){
          fprintf(stderr,"PURE TG XY GEOMETRY REFUSED: groups=%u/%u threads=%u/%u\n",
                  xgroups,ygroups,xthreads,ythreads);return 2;
        }
      }
      if(tg_partial96 || tg_full64 || tg_full32 || tensor_stage || tg_partial_tg96 || tg_full_tg64 || tg_partial_tg80 || tg_full_tg80_control || tg_param){
        uint32_t grid=0,mode=0,threads=0,total_threads=0;
        memcpy(&grid,mapped[geometry_index]+0xa8,4);memcpy(&mode,mapped[geometry_index]+0xb4,4);
        memcpy(&threads,mapped[extent_index]+0x10,4);
        memcpy(&total_threads,mapped[extent_index]+0x1c,4);
        unsigned expected_grid=tg_param?tg_param_grid:tensor_sg4h12?1536u:tensor_sg4h10?1280u:tensor_sg4h9?1152u:tensor_sg4h8?1024u:tensor_sg4h7?896u:tensor_sg4h6?768u:tensor_sg4h5?640u:tensor_sg4h4?512u:tensor_sg4h3?384u:tensor_sg4h?256u:tensor_sg4?128u:(tg_full32 || tensor_stage)?32u:tg_partial_tg80?80u:(tg_partial96 || tg_partial_tg96)?96u:64u;
        if(grid!=expected_grid || mode!=1 || threads!=expected_grid ||
           ((tg_full32 || tensor_stage) && total_threads!=(tensor_sg4h?128u:expected_grid))){
          fprintf(stderr,"PURE PARTIAL GEOMETRY REFUSED: grid=%u mode=%u threads=%u total=%u\n",
                  grid,mode,threads,total_threads);return 2;
        }
      }
      if(tensor_stage){
        uint32_t state=0;
        memcpy(&state,mapped[extent_index],4);
        if(state!=0x00880000u || getenv("ORDERED_TG_FIRE")){
          fprintf(stderr,"PURE TENSOR STAGE REFUSED: state word or fire request\n");return 2;
        }
      }
      fprintf(stderr,"PURE TG RESOURCE STAGE PASS: %zu authored bytes at %s, %u-byte static threadgroup resource word, A/B/C descriptors and physical readback checked; no Submit.\n",
              code_bytes,tensor_code_alloc0?"0x100000006c0":"0x10000058600",scratch_bytes);
    } else if(!getenv("ORDERED_FFN_TILE")) {
    int uniform_one=getenv("ORDERED_UNIFORM_ATOMIC_ONE")!=NULL;
    int report_width=getenv("ORDERED_UNIFORM_REPORT_WIDTH")!=NULL;
    int atomic_one=getenv("ORDERED_ATOMIC_ONE")!=NULL || uniform_one;
    if(atomic_one){
      const char* code_path=getenv(uniform_one?"ORDERED_UNIFORM_ATOMIC_CODE":"ORDERED_ATOMIC_CODE");
      const char* atomic_op=getenv("ORDERED_ATOMIC_OP")?getenv("ORDERED_ATOMIC_OP"):"add";
      int atomic_and=strcmp(atomic_op,"and")==0,atomic_xor=strcmp(atomic_op,"xor")==0;
      int atomic_cmpxchg5=strcmp(atomic_op,"cmpxchg5")==0;
      int atomic_cmpxchg6=strcmp(atomic_op,"cmpxchg6")==0;
      int atomic_cmpxchg5ret=strcmp(atomic_op,"cmpxchg5ret")==0;
      int atomic_cmpxchg5retwide0=strcmp(atomic_op,"cmpxchg5retwide0")==0;
      int atomic_cmpxchg5retwide7=strcmp(atomic_op,"cmpxchg5retwide7")==0;
      int atomic_addret=strcmp(atomic_op,"addret")==0;
      int atomic_addretwitness=strcmp(atomic_op,"addretwitness")==0;
      int atomic_addretwitnessb=strcmp(atomic_op,"addretwitnessb")==0;
      int atomic_or=strcmp(atomic_op,"or")==0;
      int atomic_sub=strcmp(atomic_op,"sub")==0;
      int atomic_exchange=strcmp(atomic_op,"exchange")==0;
      int atomic_cmpxchg=strcmp(atomic_op,"cmpxchg")==0;
      int atomic_smin=strcmp(atomic_op,"smin")==0;
      int atomic_smax=strcmp(atomic_op,"smax")==0;
      int atomic_umin=strcmp(atomic_op,"umin")==0;
      int atomic_umax=strcmp(atomic_op,"umax")==0;
      int atomic_fadd32=strcmp(atomic_op,"fadd32")==0;
      int atomic_fadd32ret7=strcmp(atomic_op,"fadd32ret7")==0;
      int atomic_contendfadd32ret7=strcmp(atomic_op,"contendfadd32ret7")==0;
      int atomic_contendfadd32ret7nan=strcmp(atomic_op,"contendfadd32ret7nan")==0;
      int atomic_loop1=strcmp(atomic_op,"loop1")==0;
      int atomic_loop3=strcmp(atomic_op,"loop3")==0;
      int atomic_loopdiv13=strcmp(atomic_op,"loopdiv13")==0;
      int atomic_loopdiv31=strcmp(atomic_op,"loopdiv31")==0;
      int atomic_loopcap53=strcmp(atomic_op,"loopcap53")==0;
      int atomic_loopzero03=strcmp(atomic_op,"loopzero03")==0;
      int atomic_loopregion2=strcmp(atomic_op,"loopregion2")==0;
      int atomic_shuffle1norm=strcmp(atomic_op,"shuffle1norm")==0;
      int atomic_shuffle1sub=strcmp(atomic_op,"shuffle1sub")==0;
      int atomic_contendfadd32halfret7even=strcmp(atomic_op,"contendfadd32halfret7even")==0;
      int atomic_contendfadd32halfret7odd=strcmp(atomic_op,"contendfadd32halfret7odd")==0;
      int atomic_contendadd=strcmp(atomic_op,"contendadd")==0;
      int atomic_compiledaddret=strcmp(atomic_op,"compiledaddret")==0;
      int atomic_compiledaddretplus1=strcmp(atomic_op,"compiledaddretplus1")==0;
      int atomic_compiledxorret=strcmp(atomic_op,"compiledxorret")==0;
      int atomic_compiledxorretplus1=strcmp(atomic_op,"compiledxorretplus1")==0;
      int atomic_compiledorret=strcmp(atomic_op,"compiledorret")==0;
      int atomic_compiledsubret=strcmp(atomic_op,"compiledsubret")==0;
      int atomic_compiledandret=strcmp(atomic_op,"compiledandret")==0;
      int atomic_compiledcmpxchg5ret=strcmp(atomic_op,"compiledcmpxchg5ret")==0;
      int atomic_compiledcmpxchg5shift1ret=strcmp(atomic_op,"compiledcmpxchg5shift1ret")==0;
      int atomic_compiledcmpxchg5shift5ret=strcmp(atomic_op,"compiledcmpxchg5shift5ret")==0;
      int atomic_compiledcmpxchg5shift7ret=strcmp(atomic_op,"compiledcmpxchg5shift7ret")==0;
      int atomic_compiledcmpxchg5shift12ret=strcmp(atomic_op,"compiledcmpxchg5shift12ret")==0;
      int atomic_authoredxorret7=strcmp(atomic_op,"authoredxorret7")==0;
      int atomic_authoredorret7=strcmp(atomic_op,"authoredorret7")==0;
      int atomic_authoredsubret7=strcmp(atomic_op,"authoredsubret7")==0;
      int atomic_authoredandret7=strcmp(atomic_op,"authoredandret7")==0;
      int atomic_contendfadd32=strcmp(atomic_op,"contendfadd32")==0;
      int atomic_contendfadd32halfeven=strcmp(atomic_op,"contendfadd32halfeven")==0;
      int atomic_contendfadd32halfodd=strcmp(atomic_op,"contendfadd32halfodd")==0;
      int atomic_addretwide=strcmp(atomic_op,"addretwide")==0;
      int atomic_addretwidedelay=strcmp(atomic_op,"addretwidedelay")==0;
      int atomic_addretwide7=strcmp(atomic_op,"addretwide7")==0;
      int atomic_contendaddret7=strcmp(atomic_op,"contendaddret7")==0;
      int atomic_contendaddretvary7=strcmp(atomic_op,"contendaddretvary7")==0;
      int atomic_min=strcmp(atomic_op,"min")==0;
      int atomic_max=strcmp(atomic_op,"max")==0;
      int atomic_hole=strcmp(atomic_op,"hole12")==0;
      int atomic_hole_plus1=strcmp(atomic_op,"hole12plus1")==0;
      int atomic_hole_minus32=strcmp(atomic_op,"hole12minus32")==0;
      int atomic_hole_float1=strcmp(atomic_op,"hole12float1")==0;
      int atomic_hole_ulp=strcmp(atomic_op,"hole12ulp")==0;
      int atomic_hole_halfulp=strcmp(atomic_op,"hole12halfulp")==0;
      int atomic_hole_minnormal=strcmp(atomic_op,"hole12minnormal")==0;
      int atomic_hole_maxsub=strcmp(atomic_op,"hole12maxsub")==0;
      int atomic_hole_negminnormal=strcmp(atomic_op,"hole12negminnormal")==0;
      size_t code_bytes=uniform_one?((atomic_hole_float1 || atomic_hole_ulp || atomic_hole_halfulp || atomic_hole_minnormal || atomic_hole_maxsub || atomic_hole_negminnormal)?140u:((atomic_hole_plus1 || atomic_hole_minus32)?144u:((atomic_xor || atomic_sub || atomic_exchange || atomic_cmpxchg || atomic_umin || atomic_umax)?138u:(report_width?136u:124u)))):(atomic_contendadd || atomic_compiledaddret || atomic_compiledaddretplus1 || atomic_compiledxorret || atomic_compiledxorretplus1 || atomic_compiledorret || atomic_compiledsubret || atomic_compiledandret || atomic_authoredxorret7 || atomic_authoredorret7 || atomic_authoredsubret7 || atomic_authoredandret7?52u:((atomic_compiledcmpxchg5ret || atomic_compiledcmpxchg5shift1ret || atomic_compiledcmpxchg5shift5ret || atomic_compiledcmpxchg5shift7ret || atomic_compiledcmpxchg5shift12ret)?60u:(atomic_addretwidedelay?192u:((atomic_addretwide || atomic_addretwide7)?64u:((atomic_addretwitness || atomic_addretwitnessb || atomic_fadd32 || atomic_contendfadd32 || atomic_contendfadd32halfeven || atomic_contendfadd32halfodd)?62u:(atomic_addret || atomic_contendaddret7 || atomic_contendaddretvary7 || atomic_cmpxchg5retwide0 || atomic_cmpxchg5retwide7?66u:(atomic_cmpxchg5ret?68u:((atomic_and || atomic_xor || atomic_sub || atomic_umin || atomic_umax || atomic_exchange || atomic_cmpxchg5 || atomic_cmpxchg6)?78u:76u))))))));
      if(atomic_fadd32ret7 && !uniform_one)code_bytes=48u;
      if((atomic_contendfadd32ret7 || atomic_contendfadd32ret7nan || atomic_contendfadd32halfret7even || atomic_contendfadd32halfret7odd) && !uniform_one)code_bytes=74u;
      if((atomic_loop1 || atomic_loop3) && !uniform_one)code_bytes=88u;
      if((atomic_loopdiv13 || atomic_loopdiv31 || atomic_loopcap53 || atomic_loopzero03) && !uniform_one)code_bytes=146u;
      if(atomic_loopregion2 && !uniform_one)code_bytes=140u;
      if((atomic_shuffle1norm || atomic_shuffle1sub) && !uniform_one)code_bytes=52u;
      if(!code_path || getenv("ORDERED_VALID_ONE") || getenv("ORDERED_VALID_TWO") ||
         getenv("ORDERED_VALID_BATCH_TWO") ||
         (uniform_one && getenv("ORDERED_ATOMIC_ONE")) ||
         (report_width && !uniform_one) ||
         (strcmp(atomic_op,"add") && !atomic_and && !atomic_xor && !atomic_addret && !atomic_addretwitness && !atomic_addretwitnessb && !atomic_addretwide && !atomic_addretwide7 && !atomic_addretwidedelay && !atomic_contendaddret7 && !atomic_contendaddretvary7 && !atomic_cmpxchg5 && !atomic_cmpxchg6 && !atomic_cmpxchg5ret && !atomic_cmpxchg5retwide0 && !atomic_cmpxchg5retwide7 && !atomic_or && !atomic_sub && !atomic_min && !atomic_max && !atomic_exchange && !atomic_cmpxchg && !atomic_smin && !atomic_smax && !atomic_umin && !atomic_umax && !atomic_fadd32 && !atomic_fadd32ret7 && !atomic_contendfadd32ret7 && !atomic_contendfadd32ret7nan && !atomic_loop1 && !atomic_loop3 && !atomic_loopdiv13 && !atomic_loopdiv31 && !atomic_loopcap53 && !atomic_loopzero03 && !atomic_loopregion2 && !atomic_shuffle1norm && !atomic_shuffle1sub && !atomic_contendfadd32halfret7even && !atomic_contendfadd32halfret7odd && !atomic_contendadd && !atomic_compiledaddret && !atomic_compiledaddretplus1 && !atomic_compiledxorret && !atomic_compiledxorretplus1 && !atomic_compiledorret && !atomic_compiledsubret && !atomic_compiledandret && !atomic_compiledcmpxchg5ret && !atomic_compiledcmpxchg5shift1ret && !atomic_compiledcmpxchg5shift5ret && !atomic_compiledcmpxchg5shift7ret && !atomic_compiledcmpxchg5shift12ret && !atomic_authoredxorret7 && !atomic_authoredorret7 && !atomic_authoredsubret7 && !atomic_authoredandret7 && !atomic_contendfadd32 && !atomic_contendfadd32halfeven && !atomic_contendfadd32halfodd && !atomic_hole && !atomic_hole_plus1 && !atomic_hole_minus32 && !atomic_hole_float1 && !atomic_hole_ulp && !atomic_hole_halfulp && !atomic_hole_minnormal && !atomic_hole_maxsub && !atomic_hole_negminnormal) ||
         (!uniform_one && (atomic_cmpxchg || atomic_smin || atomic_smax || atomic_hole || atomic_hole_plus1 || atomic_hole_minus32 || atomic_hole_float1 || atomic_hole_ulp || atomic_hole_halfulp || atomic_hole_minnormal || atomic_hole_maxsub || atomic_hole_negminnormal)) ||
         (!uniform_one && (atomic_and || atomic_xor || atomic_sub || atomic_or || atomic_min || atomic_max || atomic_umin || atomic_umax || atomic_exchange || atomic_fadd32 || atomic_fadd32ret7 || atomic_contendfadd32ret7 || atomic_contendfadd32ret7nan || atomic_loop1 || atomic_loop3 || atomic_loopdiv13 || atomic_loopdiv31 || atomic_loopcap53 || atomic_loopzero03 || atomic_loopregion2 || atomic_shuffle1norm || atomic_shuffle1sub || atomic_contendfadd32halfret7even || atomic_contendfadd32halfret7odd || atomic_contendadd || atomic_compiledaddret || atomic_compiledaddretplus1 || atomic_compiledxorret || atomic_compiledxorretplus1 || atomic_compiledorret || atomic_compiledsubret || atomic_compiledandret || atomic_compiledcmpxchg5ret || atomic_compiledcmpxchg5shift1ret || atomic_compiledcmpxchg5shift5ret || atomic_compiledcmpxchg5shift7ret || atomic_compiledcmpxchg5shift12ret || atomic_authoredxorret7 || atomic_authoredorret7 || atomic_authoredsubret7 || atomic_authoredandret7 || atomic_contendfadd32 || atomic_contendfadd32halfeven || atomic_contendfadd32halfodd || atomic_addret || atomic_addretwitness || atomic_addretwitnessb || atomic_addretwide || atomic_addretwidedelay || atomic_addretwide7 || atomic_contendaddret7 || atomic_contendaddretvary7 || atomic_cmpxchg5 || atomic_cmpxchg6 || atomic_cmpxchg5ret || atomic_cmpxchg5retwide0 || atomic_cmpxchg5retwide7) && !getenv("ORDERED_ATOMIC_EXPLICIT_B")) ||
         (uniform_one && ((atomic_xor || atomic_sub || atomic_exchange || atomic_cmpxchg || atomic_smin || atomic_smax || atomic_umin || atomic_umax || atomic_and || atomic_or || atomic_hole || atomic_hole_plus1 || atomic_hole_minus32 || atomic_hole_float1 || atomic_hole_ulp || atomic_hole_halfulp || atomic_hole_minnormal || atomic_hole_maxsub || atomic_hole_negminnormal) && !report_width)))return 2;
      FILE* cf=fopen(code_path,"rb");if(!cf)return 2;
      uint8_t authored[256];
      int code_ok=fread(authored,1,code_bytes,cf)==code_bytes &&
                  fgetc(cf)==EOF && !ferror(cf);
      fclose(cf);
      uint16_t packet_mid=0,packet_high=0;
      memcpy(&packet_mid,mapped[23]+0x46,2);
      memcpy(&packet_high,mapped[23]+0x48,2);
      uint64_t binding_a=0,binding_b=0,binding_c=0;
      memcpy(&binding_a,mapped[28]+0x1ba0,8);
      memcpy(&binding_b,mapped[28]+0x1ba8,8);
      memcpy(&binding_c,mapped[28]+0x1bb0,8);
      if(!code_ok || total!=638976 ||
         memcmp(mapped[0]+0x6c0,KERNEL_3I1,sizeof KERNEL_3I1) ||
         memcmp(mapped[20]+0x600,authored,code_bytes) ||
         packet!=0x0e408607u || packet_mid!=0x0005u || packet_high!=0x0100u ||
         binding_a!=0x10000030000ull || binding_b!=0x10000030400ull ||
         binding_c!=0x10000030800ull)return 2;
      char stage_label[64];
      snprintf(stage_label,sizeof stage_label,"uniform-%s",
               report_width && !strcmp(atomic_op,"add")?"width":atomic_op);
      fprintf(stderr,"PURE ATOMIC RESOURCE STAGE PASS: op=%s %zu authored bytes at 0x10000058600, packet=0x%08x, A/B/C descriptors checked, physical bytes read back; no Submit.\n",
              uniform_one?stage_label:atomic_op,code_bytes,packet);
    } else {
    int valid_bias=getenv("VALID_BIAS")?atoi(getenv("VALID_BIAS")):1;
    int valid_scale=getenv("VALID_SCALE")?atoi(getenv("VALID_SCALE")):3;
    const char* valid_op=getenv("VALID_OP")?getenv("VALID_OP"):"add";
    const char* placement=getenv("VALID_PLACEMENT")?getenv("VALID_PLACEMENT"):"captured";
    int tail=strcmp(placement,"tail")==0;
    int tail_control=strcmp(placement,"tail-control")==0;
    int tail_any=tail || tail_control;
    int second_allocation=strcmp(placement,"second-allocation")==0;
    int second_candidate=strcmp(placement,"second-allocation-candidate")==0;
    int second_control=strcmp(placement,"second-allocation-control")==0;
    int second_any=second_candidate || second_control;
    int third_candidate=strcmp(placement,"third-allocation-candidate")==0;
    int third_control=strcmp(placement,"third-allocation-control")==0;
    int third_any=third_candidate || third_control;
    int alloc24_candidate=strcmp(placement,"alloc24-candidate")==0;
    int alloc24_control=strcmp(placement,"alloc24-control")==0;
    int alloc24_any=alloc24_candidate || alloc24_control;
    int input_sum_placement=strcmp(placement,"input-sum-alloc1")==0;
    int input_add7_placement=strcmp(placement,"input-add7-alloc1")==0;
    int two_program_placement=strcmp(placement,"input-muladd-then-add7")==0;
    int input_muladd_placement=strcmp(placement,"input-muladd-alloc1")==0 || two_program_placement;
    int input_any=input_sum_placement || input_add7_placement || input_muladd_placement;
    if(second_allocation){
      fprintf(stderr,"ORDERED REFUSED: faulted second-allocation packet construction is archived for offline audit only\n");
      return 2;
    }
    int is_sub=strcmp(valid_op,"sub")==0;
    int is_xor=strcmp(valid_op,"xor")==0;
    int is_and=strcmp(valid_op,"and")==0;
    int is_or=strcmp(valid_op,"or")==0;
    int is_input_sum=strcmp(valid_op,"input-sum")==0;
    int is_input_add7=strcmp(valid_op,"input-add7")==0;
    int is_input_muladd=strcmp(valid_op,"input-muladd")==0;
    if((strcmp(placement,"captured") && !tail_any && !second_any && !third_any && !alloc24_any && !input_any) ||
       ((tail_any || second_any || third_any || alloc24_any) && (strcmp(valid_op,"add") || valid_scale!=3 ||
                     valid_bias!=((tail || second_candidate || third_candidate || alloc24_candidate)?2:1))) ||
       (input_sum_placement!=(is_input_sum && valid_scale==0 && valid_bias==0)) ||
       (input_add7_placement!=(is_input_add7 && valid_scale==0 && valid_bias==7)) ||
       (input_muladd_placement!=(is_input_muladd && valid_scale==0 && valid_bias==7)) ||
       !((!strcmp(valid_op,"add") &&
          ((valid_scale==3 && (valid_bias==1 || valid_bias==2 || valid_bias==3 || valid_bias==7)) ||
           (valid_scale==5 && valid_bias==7))) ||
         ((is_sub || is_xor || is_and || is_or) && valid_scale==3 && valid_bias==7) ||
         (is_input_sum && valid_scale==0 && valid_bias==0) ||
         (is_input_add7 && valid_scale==0 && valid_bias==7) ||
         (is_input_muladd && valid_scale==0 && valid_bias==7))){
      fprintf(stderr,"ORDERED REFUSED: invalid placement/op/scale/bias\n");return 2;
    }
    uint8_t expected_code[sizeof KERNEL_3I1];memcpy(expected_code,KERNEL_3I1,sizeof expected_code);
    if(tail_control || second_control || third_control || alloc24_control){
      expected_code[19]=2;
    } else if(is_xor || is_and || is_or){
      static const uint8_t xor7[42]={12,160,16,6,39,128,4,26,33,0,161,34,200,0,10,0,
        1,2,51,0,6,136,50,56,163,34,198,129,31,8,67,0,1,6,16,64,14,0,0,0,0,0};
      memcpy(expected_code,xor7,sizeof xor7);
      if(is_and){expected_code[20]=7;expected_code[22]=48;expected_code[23]=48;expected_code[26]=196;}
      if(is_or)expected_code[20]=7;
    } else if(is_sub){
      static const uint8_t sub7[42]={12,160,16,6,39,128,4,26,33,0,161,34,200,0,10,0,
        1,2,55,0,4,26,37,128,162,34,200,1,11,0,31,8,67,0,1,6,16,64,14,0,0,0};
      memcpy(expected_code,sub7,sizeof sub7);
    } else {
      expected_code[19]=(uint8_t)(valid_bias<=3?valid_bias:3);
      if(valid_bias==7)expected_code[21]=58;
      if(valid_scale==5){expected_code[12]=72;expected_code[13]=1;}
    }
    uint16_t packet_high=0;memcpy(&packet_high,mapped[23]+0x48,2);
    uint16_t packet_mid=0;memcpy(&packet_mid,mapped[23]+0x46,2);
    int code_match=input_muladd_placement?
      !memcmp(mapped[0]+0x6c0,two_program_placement?G17_INPUT_ADD7_CODE:KERNEL_3I1,
              two_program_placement?sizeof G17_INPUT_ADD7_CODE:sizeof KERNEL_3I1) &&
      !memcmp(mapped[1]+0x500,G17_INPUT_MULADD_CODE,sizeof G17_INPUT_MULADD_CODE):input_add7_placement?
      !memcmp(mapped[0]+0x6c0,KERNEL_3I1,sizeof KERNEL_3I1) &&
      !memcmp(mapped[1]+0x500,G17_INPUT_ADD7_CODE,sizeof G17_INPUT_ADD7_CODE):input_sum_placement?
      !memcmp(mapped[0]+0x6c0,KERNEL_3I1,sizeof KERNEL_3I1) &&
      !memcmp(mapped[1]+0x500,G17_INPUT_SUM_CODE,sizeof G17_INPUT_SUM_CODE):alloc24_any?
      !memcmp(mapped[0]+0x6c0,KERNEL_3I1,sizeof KERNEL_3I1) &&
      !memcmp(mapped[24]+0x500,expected_code,sizeof expected_code):third_any?
      !memcmp(mapped[0]+0x6c0,KERNEL_3I1,sizeof KERNEL_3I1) &&
      !memcmp(mapped[20]+0x500,expected_code,sizeof expected_code):second_any?
      !memcmp(mapped[0]+0x6c0,KERNEL_3I1,sizeof KERNEL_3I1) &&
      !memcmp(mapped[1]+0x500,expected_code,sizeof expected_code):tail_any?
      !memcmp(mapped[0]+0x6c0,KERNEL_3I1,sizeof KERNEL_3I1) &&
      !memcmp(mapped[0]+0xe500,expected_code,sizeof expected_code):
      !memcmp(mapped[0]+0x6c0,expected_code,sizeof expected_code);
    if(total!=638976 || !code_match ||
       packet!=(tail?0x0e40e507u:(second_candidate || third_candidate || input_any)?0x0e408507u:
                alloc24_candidate?0x0e400507u:0x0e4006c7u) ||
       packet_mid!=(alloc24_candidate?0x000au:third_candidate?0x0005u:
                    (second_candidate || input_any)?0x0001u:0x0000u) ||
       packet_high!=0x0100u){
      fprintf(stderr,"ORDERED REFUSED: authored code or packet\n");return 2;
    }
    if(input_any){
      uint32_t* input_a=(uint32_t*)mapped[2];
      uint32_t* input_b=(uint32_t*)(mapped[2]+0x400);
      for(unsigned i=0;i<64;i++){
        if(input_a[i]!=11*i+5 || ((input_sum_placement || input_muladd_placement) && input_b[i]!=7*i+3)){
          fprintf(stderr,"ORDERED REFUSED: captured input data changed at %u\n",i);return 2;
        }
      }
    }
    fprintf(stderr,"ORDERED STAGE PASS: %zu physical bytes CPU readback, authored code and packet checked; no Submit.\n",total);
    }
    }
  }
  if(getenv("ORDERED_QUEUE_PREFLIGHT")){
    if(!payload_path){fprintf(stderr,"ORDERED QUEUE REFUSED: physical payload absent\n");return 2;}
    void*(*NSet)(void*,void*,unsigned)=dlsym(io,"IOGPUNotificationQueueSetDispatchQueue");
    void*(*QCreate)(void*,void*,unsigned long)=dlsym(io,"IOGPUCommandQueueCreate");
    uint64_t(*NextTraceID)(void*)=dlsym(io,"IOGPUDeviceGetNextGlobalTraceID");
    if(!NSet || !QCreate || !NextTraceID){fprintf(stderr,"ORDERED QUEUE REFUSED: IOGPU export absent\n");return 2;}
    unsigned old=*(unsigned*)((char*)dev+0x14);
    *(unsigned*)((char*)dev+0x14)=CONN;
    unsigned char qdesc[0x410]={0};
    char process_path[PROC_PIDPATHINFO_MAXSIZE]={0};
    if(proc_pidpath(getpid(),process_path,sizeof process_path)<=0 || strlen(process_path)>=29){
      fprintf(stderr,"ORDERED QUEUE REFUSED: descriptor process path\n");return 2;
    }
    memcpy(qdesc,process_path,strlen(process_path)+1);
    memcpy(qdesc+0x3e3,process_path,strlen(process_path)+1);
    *(uint32_t*)(qdesc+0x400)=2;
    *(uint32_t*)(qdesc+0x408)=0xffffffffu;
    *(uint32_t*)(qdesc+0x40c)=1;
    void* queue=QCreate(dev,qdesc,sizeof qdesc);
    void* notification=NULL;
    if(queue)memcpy(&notification,(char*)queue+0x28,sizeof notification);
    fprintf(stderr,"ORDERED QUEUE create=%p connection=0x%x prior=0x%x notification=%p\n",
            queue,CONN,old,notification);
    if(!queue || !notification)return 2;
    dispatch_queue_t dq=dispatch_queue_create("agxforge.g17.ordered-preflight",DISPATCH_QUEUE_SERIAL);
    if(!dq || !NSet(notification,(void*)dq,1)){
      fprintf(stderr,"ORDERED QUEUE REFUSED: notification bind\n");return 2;
    }
    unsigned char ob[16];size_t obc=sizeof ob;
    kern_return_t k=IOConnectCallStructMethod(CONN,6,0,0,ob,&obc);
    fprintf(stderr,"ORDERED QUEUE sel-6 kr=0x%x bytes=%zu\n",k,obc);
    if(k || obc!=sizeof ob)return 2;
    struct shmem_result shmem[2]={0};
    for(int i=0;i<2;i++){
      uint64_t args[2]={0x4000,(uint64_t)i};obc=sizeof shmem[i];
      k=IOConnectCallMethod(CONN,14,args,2,0,0,0,0,&shmem[i],&obc);
      fprintf(stderr,"ORDERED QUEUE sel-14#%d kr=0x%x bytes=%zu cpu=0x%llx size=0x%x id=%u\n",
              i,k,obc,(unsigned long long)shmem[i].cpu,shmem[i].size,shmem[i].id);
      if(k || obc!=sizeof shmem[i] || !shmem[i].cpu || shmem[i].size!=0x4000 ||
         shmem[i].id!=(i==0?1:2))return 2;
    }
    uint64_t first=NextTraceID(dev),second=NextTraceID(dev);
    fprintf(stderr,"ORDERED QUEUE trace IDs first=0x%llx second=0x%llx\n",
            (unsigned long long)first,(unsigned long long)second);
    if(!first || second!=first+1)return 2;
    const char* command_path=getenv("STAGE_COMMAND_PAGES");
    if(!command_path){fprintf(stderr,"ORDERED QUEUE REFUSED: command template absent\n");return 2;}
    FILE* command=fopen(command_path,"rb");if(!command){perror("ORDERED command");return 2;}
    uint8_t* pages=malloc(0x8000);if(!pages){fclose(command);return 2;}
    if(fread(pages,1,0x8000,command)!=0x8000 || fgetc(command)!=EOF || ferror(command)){
      fprintf(stderr,"ORDERED QUEUE REFUSED: command template length\n");free(pages);fclose(command);return 2;
    }
    fclose(command);
    memcpy(pages+0x234,&second,4);
    memcpy(pages+0x4000,&first,8);
    memcpy(pages+0x4018,&first,8);
    memcpy(pages+0x4028,&second,8);
    uint32_t ready=0;memcpy(&ready,pages+0x4024,4);
    if(ready!=0xf0){fprintf(stderr,"ORDERED QUEUE REFUSED: ready word changed\n");free(pages);return 2;}
    memcpy((void*)(uintptr_t)shmem[1].cpu,pages,0x4000);
    memcpy((void*)(uintptr_t)shmem[0].cpu,pages+0x4000,0x4000);
    int match=memcmp((void*)(uintptr_t)shmem[1].cpu,pages,0x4000)==0 &&
              memcmp((void*)(uintptr_t)shmem[0].cpu,pages+0x4000,0x4000)==0;
    free(pages);
    if(!match || !no_agx_metal()){fprintf(stderr,"ORDERED QUEUE REFUSED: page readback or image check\n");return 2;}
    if(getenv("ORDERED_WORKLOAD_DECODER")){
      if(!getenv("ORDERED_FFN_TILE") || !getenv("ORDERED_TENSOR_GRAPH"))return 2;
      return g17_workload_decoder(io,dev,queue,mapped,sizes,shmem);
    }
    if(getenv("ORDERED_WORKLOAD_INFERENCE")){
      if(!getenv("ORDERED_FFN_TILE") || !getenv("ORDERED_TENSOR_GRAPH"))return 2;
      return g17_workload_inference(io,dev,queue,mapped,sizes,shmem);
    }
    if(getenv("ORDERED_WORKLOAD_LAYER")){
      if(!getenv("ORDERED_FFN_TILE") || !getenv("ORDERED_TENSOR_GRAPH"))return 2;
      return g17_workload_layer(io,dev,queue,mapped,sizes,shmem);
    }
    if(getenv("ORDERED_WORKLOAD_ATTENTION_FUSED")){
      if(!getenv("ORDERED_FFN_TILE") || !getenv("ORDERED_TENSOR_GRAPH"))return 2;
      return g17_workload_attention_fused(io,dev,queue,mapped,sizes,shmem);
    }
    if(getenv("ORDERED_WORKLOAD_ATTENTION")){
      if(!getenv("ORDERED_FFN_TILE") || !getenv("ORDERED_TENSOR_GRAPH"))return 2;
      return g17_workload_attention(io,dev,queue,mapped,sizes,shmem);
    }
    if(getenv("ORDERED_WORKLOAD_FFN")){
      if(!getenv("ORDERED_FFN_TILE") || !getenv("ORDERED_TENSOR_GRAPH"))return 2;
      return g17_workload_ffn(io,dev,queue,mapped,sizes,shmem);
    }
    if(getenv("ORDERED_WORKLOAD_FFN_PREPARE")){
      if(!getenv("ORDERED_FFN_TILE") || !getenv("ORDERED_TENSOR_GRAPH"))return 2;
      return g17_workload_ffn_prepare(mapped,sizes);
    }
    if(getenv("ORDERED_WORKLOAD_TILE")){
      if(!getenv("ORDERED_FFN_TILE") || !getenv("ORDERED_TENSOR_GRAPH"))return 2;
      return g17_workload_tile(io,dev,queue,mapped,sizes,shmem);
    }
    if(getenv("ORDERED_FFN_TILE")){
      const char* shift_setting=getenv("ORDERED_FFN_SHIFT");
      if(shift_setting && strcmp(shift_setting,"0x4000"))return 2;
      unsigned shift=shift_setting?0x4000u:0u;
      unsigned stale_c_page=getenv("ORDERED_FFN_STALE_C_PAGE")!=NULL;
      if(stale_c_page && shift!=0x4000u)return 2;
      unsigned c_competition=getenv("ORDERED_FFN_C_COMPETITION")!=NULL;
      unsigned c_binding_base=getenv("ORDERED_FFN_C_BINDING_BASE")!=NULL;
      if((c_competition && shift!=0x4000u) || (c_binding_base && !c_competition))return 2;
      unsigned b_competition=getenv("ORDERED_FFN_B_COMPETITION")!=NULL;
      unsigned b_binding_base=getenv("ORDERED_FFN_B_BINDING_BASE")!=NULL;
      if((b_competition && shift!=0x4000u) || (b_binding_base && !b_competition) ||
         (b_competition && c_competition))return 2;
      unsigned a_competition=getenv("ORDERED_FFN_A_COMPETITION")!=NULL;
      unsigned a_binding_base=getenv("ORDERED_FFN_A_BINDING_BASE")!=NULL;
      if((a_competition && shift!=0x4000u) || (a_binding_base && !a_competition) ||
         (a_competition && (b_competition || c_competition)))return 2;
      unsigned ffn_pair=getenv("ORDERED_FFN_PAIR")!=NULL;
      unsigned ffn_acc=getenv("ORDERED_FFN_ACCUMULATE")!=NULL;
      unsigned ffn_six=getenv("ORDERED_FFN_SIX")!=NULL;
      unsigned ffn_c_upper=getenv("ORDERED_FFN_C_UPPER")!=NULL;
      unsigned ffn_c_tail=getenv("ORDERED_FFN_C_TAIL")!=NULL;
      unsigned ffn_c_cross=getenv("ORDERED_FFN_C_CROSS")!=NULL;
      unsigned ffn_c_xcontrol=getenv("ORDERED_FFN_C_XCONTROL")!=NULL;
      unsigned ffn_resident_pair=getenv("ORDERED_FFN_RESIDENT_PAIR")!=NULL;
      unsigned ffn_resident_full=getenv("ORDERED_FFN_RESIDENT_FULL")!=NULL;
      unsigned ffn_standalone_query=getenv("ORDERED_FFN_STANDALONE_QUERY")!=NULL;
      unsigned ffn_standalone_embedding_query=getenv("ORDERED_FFN_STANDALONE_EMBEDDING_QUERY")!=NULL;
      unsigned ffn_standalone_query_all=getenv("ORDERED_FFN_STANDALONE_QUERY_ALL")!=NULL;
      unsigned ffn_standalone_key_all=getenv("ORDERED_FFN_STANDALONE_KEY_ALL")!=NULL;
      unsigned ffn_standalone_scores=getenv("ORDERED_FFN_STANDALONE_SCORES")!=NULL;
      unsigned ffn_standalone_softmax=getenv("ORDERED_FFN_STANDALONE_SOFTMAX")!=NULL;
      unsigned ffn_standalone_value=getenv("ORDERED_FFN_STANDALONE_VALUE")!=NULL;
      unsigned ffn_standalone_context=getenv("ORDERED_FFN_STANDALONE_CONTEXT")!=NULL;
      unsigned ffn_standalone_output=getenv("ORDERED_FFN_STANDALONE_OUTPUT")!=NULL;
      unsigned ffn_standalone_residual=getenv("ORDERED_FFN_STANDALONE_RESIDUAL")!=NULL;
      unsigned ffn_standalone_attention_ln=getenv("ORDERED_FFN_STANDALONE_ATTENTION_LN")!=NULL;
      unsigned ffn_standalone_ffn_pack=getenv("ORDERED_FFN_STANDALONE_FFN_PACK")!=NULL;
      unsigned ffn_standalone_ffn_tile=getenv("ORDERED_FFN_STANDALONE_FFN_TILE")!=NULL;
      unsigned ffn_standalone_ffn_k6=getenv("ORDERED_FFN_STANDALONE_FFN_K6")!=NULL;
      unsigned ffn_standalone_n1_resident=getenv("ORDERED_FFN_STANDALONE_N1_RESIDENT")!=NULL;
      unsigned ffn_standalone_expand_all=getenv("ORDERED_FFN_EXPAND_ALL")!=NULL;
      unsigned ffn_standalone_wide_probe=getenv("ORDERED_FFN_WIDE_PROBE")!=NULL;
      unsigned ffn_standalone_act_all=getenv("ORDERED_FFN_ACT_ALL")!=NULL;
      unsigned ffn_standalone_contract_k0=getenv("ORDERED_FFN_CONTRACT_K0")!=NULL;
      unsigned ffn_standalone_contract_k24=getenv("ORDERED_FFN_CONTRACT_K24")!=NULL;
      unsigned ffn_standalone_contract_n01=getenv("ORDERED_FFN_CONTRACT_N01")!=NULL;
      unsigned ffn_standalone_contract_n8=getenv("ORDERED_FFN_CONTRACT_N8")!=NULL;
      unsigned ffn_standalone_contract_n12=getenv("ORDERED_FFN_CONTRACT_N12")!=NULL;
      unsigned ffn_standalone_contract_n12_bias=getenv("ORDERED_FFN_CONTRACT_N12_BIAS")!=NULL;
      unsigned ffn_standalone_residual_keep=getenv("ORDERED_FFN_RESIDUAL_KEEP")!=NULL;
      const char* ffn_contract_slot_text=getenv("ORDERED_FFN_CONTRACT_SLOT_PROBE");
      unsigned ffn_contract_slot=ffn_contract_slot_text?(unsigned)(ffn_contract_slot_text[0]-'0'):0u;
      unsigned ffn_full_append_add7=getenv("ORDERED_FFN_FULL_APPEND_ADD7")!=NULL;
      unsigned ffn_full_append_gelu=getenv("ORDERED_FFN_FULL_APPEND_GELU")!=NULL;
      unsigned ffn_full_append_bias=getenv("ORDERED_FFN_FULL_APPEND_BIAS")!=NULL;
      unsigned ffn_full_bias_gelu=getenv("ORDERED_FFN_FULL_BIAS_GELU")!=NULL;
      unsigned ffn_full_inplace=getenv("ORDERED_FFN_FULL_INPLACE32")!=NULL;
      unsigned ffn_full_activation=getenv("ORDERED_FFN_FULL_ACTIVATION")!=NULL;
      unsigned ffn_pack32=getenv("ORDERED_FFN_PACK32")!=NULL;
      unsigned ffn_pack_full=getenv("ORDERED_FFN_PACK_FULL")!=NULL;
      unsigned ffn_contract_tile=getenv("ORDERED_FFN_CONTRACT_TILE")!=NULL;
      unsigned ffn_contract_reduce=getenv("ORDERED_FFN_CONTRACT_REDUCE")!=NULL;
      unsigned ffn_contract_alloc0=getenv("ORDERED_FFN_CONTRACT_C_ALLOC0")!=NULL;
      unsigned ffn_contract_alloc1=getenv("ORDERED_FFN_CONTRACT_C_ALLOC1")!=NULL;
      unsigned ffn_copy_packed=getenv("ORDERED_FFN_COPY_PACKED")!=NULL;
      unsigned ffn_copy_dead_tile=getenv("ORDERED_FFN_COPY_DEAD_TILE")!=NULL;
      unsigned ffn_consume_reused=getenv("ORDERED_FFN_CONSUME_REUSED")!=NULL;
      unsigned ffn_pack_persistent=getenv("ORDERED_FFN_PACK_PERSISTENT")!=NULL;
      unsigned ffn_consume_persistent=getenv("ORDERED_FFN_CONSUME_PERSISTENT")!=NULL;
      unsigned ffn_contract_all=getenv("ORDERED_FFN_CONTRACT_ALL")!=NULL;
      unsigned ffn_contract_bias=getenv("ORDERED_FFN_CONTRACT_BIAS")!=NULL;
      unsigned ffn_residual_add=getenv("ORDERED_FFN_RESIDUAL_ADD")!=NULL;
      unsigned ffn_gather_residual=getenv("ORDERED_FFN_GATHER_RESIDUAL")!=NULL;
      unsigned ffn_coop_layernorm=getenv("ORDERED_FFN_COOP_LAYERNORM")!=NULL;
      unsigned ffn_four_binding=getenv("ORDERED_FFN_FOUR_BINDING_CONTROL")!=NULL;
      unsigned ffn_append_output=getenv("ORDERED_FFN_APPEND_OUTPUT_BASE")!=NULL;
      unsigned ffn_probe_appended_third=getenv("ORDERED_FFN_PROBE_APPENDED_THIRD")!=NULL;
      unsigned ffn_packed_ln=getenv("ORDERED_FFN_PACKED_LAYERNORM")!=NULL;
      unsigned ffn_query_tile=getenv("ORDERED_FFN_QUERY_TILE")!=NULL;
      unsigned ffn_query_all=getenv("ORDERED_FFN_QUERY_ALL")!=NULL;
      unsigned ffn_key_all=getenv("ORDERED_FFN_KEY_ALL")!=NULL;
      unsigned ffn_scores=getenv("ORDERED_FFN_SCORES")!=NULL;
      unsigned ffn_scores_relocated=getenv("ORDERED_FFN_SCORES_RELOCATED")!=NULL;
      unsigned ffn_softmax=getenv("ORDERED_FFN_SOFTMAX")!=NULL;
      unsigned ffn_value_all=getenv("ORDERED_FFN_VALUE_ALL")!=NULL;
      unsigned ffn_context=getenv("ORDERED_FFN_CONTEXT")!=NULL;
      unsigned ffn_context_relocated=getenv("ORDERED_FFN_CONTEXT_RELOCATED")!=NULL;
      unsigned ffn_attention_output_all=getenv("ORDERED_FFN_ATTENTION_OUTPUT_ALL")!=NULL;
      unsigned ffn_attention_residual=getenv("ORDERED_FFN_ATTENTION_RESIDUAL")!=NULL;
      unsigned ffn_attention_layernorm=getenv("ORDERED_FFN_ATTENTION_LAYERNORM")!=NULL;
      unsigned ffn_c_slot=ffn_c_tail?0x1f000u:((ffn_c_upper || ffn_c_xcontrol)?0x8000u:0u);
      if(ffn_acc && !ffn_pair)return 2;
      if(ffn_six && !ffn_acc)return 2;
      if(ffn_resident_pair && (!ffn_six || !ffn_c_xcontrol))return 2;
      if(ffn_resident_full && (!ffn_six || !ffn_c_xcontrol || ffn_resident_pair))return 2;
      if(ffn_standalone_query && (ffn_resident_full || !ffn_c_xcontrol))return 2;
      if(ffn_standalone_embedding_query && !ffn_standalone_query)return 2;
      if(ffn_standalone_query_all && !ffn_standalone_embedding_query)return 2;
      if(ffn_standalone_key_all && !ffn_standalone_query_all)return 2;
      if(ffn_standalone_scores && !ffn_standalone_key_all)return 2;
      if(ffn_standalone_softmax && !ffn_standalone_scores)return 2;
      if(ffn_standalone_value && !ffn_standalone_softmax)return 2;
      if(ffn_standalone_context && !ffn_standalone_value)return 2;
      if(ffn_standalone_output && !ffn_standalone_context)return 2;
      if(ffn_standalone_residual && !ffn_standalone_output)return 2;
      if(ffn_standalone_attention_ln && !ffn_standalone_residual)return 2;
      if(ffn_standalone_ffn_pack && !ffn_standalone_attention_ln)return 2;
      if(ffn_standalone_ffn_tile && !ffn_standalone_ffn_pack)return 2;
      if(ffn_standalone_ffn_k6 && !ffn_standalone_ffn_tile)return 2;
      if(ffn_standalone_n1_resident && (!ffn_standalone_ffn_k6 || !mapped[29] ||
                                        sizes[29]!=(getenv("ORDERED_FFN_N1_ALLOC_64K")?
                                                     0x10000u:0x40000u)))return 2;
      if(ffn_standalone_expand_all && (!ffn_standalone_n1_resident ||
                                       !getenv("ORDERED_FFN_N1_KNOWN_C")))return 2;
      if(ffn_standalone_wide_probe && !ffn_standalone_expand_all)return 2;
      if(ffn_standalone_act_all && (!ffn_standalone_expand_all ||
                                    ffn_standalone_wide_probe))return 2;
      if(ffn_standalone_contract_k0 && !ffn_standalone_act_all)return 2;
      if(ffn_standalone_contract_k24 && !ffn_standalone_contract_k0)return 2;
      if(ffn_standalone_contract_n01 && (!ffn_standalone_act_all ||
                                          ffn_standalone_contract_k0))return 2;
      if(ffn_standalone_contract_n8 && (!ffn_standalone_act_all ||
                                         ffn_standalone_contract_k0 ||
                                         ffn_standalone_contract_n01))return 2;
      if(ffn_standalone_contract_n12 && (!ffn_standalone_act_all ||
                                          ffn_standalone_contract_k0 ||
                                          ffn_standalone_contract_n01 ||
                                          ffn_standalone_contract_n8))return 2;
      if(ffn_standalone_contract_n12_bias && !ffn_standalone_contract_n12)return 2;
      if(ffn_standalone_residual_keep && !ffn_standalone_contract_n12_bias)return 2;
      if(ffn_contract_slot_text && (!ffn_standalone_contract_n8 ||
                                    ffn_contract_slot_text[0]<'0' ||
                                    ffn_contract_slot_text[0]>'3' ||
                                    ffn_contract_slot_text[1]))return 2;
      if(ffn_full_append_add7 && !ffn_resident_full)return 2;
      if(ffn_full_append_gelu && (!ffn_resident_full || ffn_full_append_add7))return 2;
      if(ffn_full_append_bias && (!ffn_resident_full || ffn_full_append_add7 || ffn_full_append_gelu))return 2;
      if(ffn_full_bias_gelu && (!ffn_resident_full || ffn_full_append_add7 ||
                                 ffn_full_append_gelu || ffn_full_append_bias))return 2;
      if(ffn_full_inplace && (!ffn_resident_full || ffn_full_append_add7 ||
                              ffn_full_append_gelu || ffn_full_append_bias ||
                              ffn_full_bias_gelu))return 2;
      if(ffn_full_activation && (!ffn_resident_full || ffn_full_append_add7 ||
                                  ffn_full_append_gelu || ffn_full_append_bias ||
                                  ffn_full_bias_gelu || ffn_full_inplace))return 2;
      if(ffn_pack32 && !ffn_full_activation)return 2;
      if(ffn_pack_full && (!ffn_full_activation || ffn_pack32))return 2;
      if(ffn_contract_tile && !ffn_pack_full)return 2;
      if(ffn_contract_reduce && (!ffn_pack_full || ffn_contract_tile))return 2;
      if(ffn_contract_alloc0 && !ffn_contract_tile)return 2;
      if(ffn_contract_alloc1 && (!ffn_contract_tile || ffn_contract_alloc0))return 2;
      if(ffn_contract_alloc1 && !mapped[1])return 2;
      if(ffn_copy_packed && (!ffn_pack_full || ffn_contract_tile || ffn_contract_reduce))return 2;
      if(ffn_copy_dead_tile && !ffn_copy_packed)return 2;
      if(ffn_consume_reused && !ffn_copy_dead_tile)return 2;
      if(ffn_pack_persistent && (!ffn_pack_full || ffn_copy_packed ||
                                 ffn_contract_reduce || ffn_contract_tile))return 2;
      if(ffn_consume_persistent && !ffn_pack_persistent)return 2;
      if(ffn_contract_all && (!ffn_pack_persistent || ffn_consume_persistent))return 2;
      if(ffn_contract_bias && !ffn_contract_all)return 2;
      if(ffn_residual_add && !ffn_contract_bias)return 2;
      if(ffn_gather_residual && !ffn_residual_add)return 2;
      if(ffn_coop_layernorm && !ffn_gather_residual)return 2;
      if(ffn_four_binding && (!ffn_gather_residual || ffn_coop_layernorm))return 2;
      if(ffn_append_output && (!ffn_four_binding || !mapped[29] || sizes[29]!=0x10000u))return 2;
      if(ffn_probe_appended_third && !ffn_append_output)return 2;
      if(ffn_packed_ln && !ffn_coop_layernorm)return 2;
      if(ffn_query_tile && !ffn_packed_ln)return 2;
      if(ffn_query_all && !ffn_query_tile)return 2;
      if(ffn_key_all && !ffn_query_all)return 2;
      if(ffn_scores && !ffn_key_all)return 2;
      if(ffn_scores_relocated && !ffn_scores)return 2;
      if(ffn_softmax && !ffn_scores_relocated)return 2;
      if(ffn_value_all && !ffn_softmax)return 2;
      if(ffn_context && !ffn_value_all)return 2;
      if(ffn_context_relocated && !ffn_context)return 2;
      if(ffn_attention_output_all && !ffn_context_relocated)return 2;
      if(ffn_attention_residual && !ffn_attention_output_all)return 2;
      if(ffn_attention_layernorm && !ffn_attention_residual)return 2;
      if(ffn_c_upper+ffn_c_tail+ffn_c_cross+ffn_c_xcontrol>1 ||
         ((ffn_c_slot || ffn_c_cross) && (!ffn_acc || shift || c_competition || c_binding_base)))return 2;
      if(ffn_pair && (shift || stale_c_page || a_competition || b_competition || c_competition))return 2;
      if(!tensor_graph || !mapped[0] || !mapped[2] || !mapped[23] || !mapped[28] ||
         sizes[0]!=0x10000u || sizes[2]!=0x20000u || sizes[28]!=0xc000u)return 2;
      if((ffn_c_cross || ffn_c_xcontrol) && (!mapped[17] || sizes[17]!=0x20000u))return 2;
      const char* input_paths[3]={getenv("ORDERED_FFN_CODE"),getenv("ORDERED_FFN_A"),getenv("ORDERED_FFN_B")};
      const unsigned input_offsets[3]={0x6c0u,0x4d80u+shift,0x5e80u+shift};
      const unsigned input_sizes[3]={ffn_acc?1458u:1276u,4096u,4096u};
      const uint8_t* input_bases[3]={mapped[0],mapped[2],mapped[2]};
      uint8_t expected_inputs[3][4096]={0};
      for(unsigned j=0;j<3;j++){
        if(!input_paths[j])return 2;
        FILE* source=fopen(input_paths[j],"rb");if(!source)return 2;
        int good=fread(expected_inputs[j],1,input_sizes[j],source)==input_sizes[j] &&
                 fgetc(source)==EOF && !ferror(source);
        fclose(source);
        if(!good || memcmp(input_bases[j]+input_offsets[j],expected_inputs[j],input_sizes[j]))return 2;
      }
      uint8_t next_inputs[2][4096]={0};
      uint8_t six_inputs[4][2][4096]={0};
      uint8_t resident_b[6][4096]={0};
      uint8_t* full_b=NULL;
      uint8_t gelu_code[536]={0};
      uint8_t bias_code[56]={0},bias_values[128]={0};
      uint8_t* full_bias=NULL;
      uint8_t pack_code[178]={0},pack_expected[64]={0};
      uint8_t* pack_full_expected=NULL;
      uint8_t contract_b[4096]={0};
      uint8_t* contract_b_full=NULL,*contract_b_all=NULL;
      uint8_t* contract_bias_values=NULL,*contract_bias_expected=NULL;
      uint8_t* residual_source=NULL,*residual_expected=NULL;
      uint8_t* gather_old_target=NULL;
      uint8_t coop_code[5316]={0},coop_gamma[1536]={0},coop_beta[1536]={0};
      double* coop_reference=NULL;
      uint8_t four_code[82]={0},four_gamma[1536]={0},four_beta[1536]={0};
      uint32_t four_expected[32]={0};
      uint8_t query_code[350]={0};
      uint8_t* query_packed=NULL;
      uint8_t* query_source_expected=NULL;
      double* query_reference=NULL;
      uint8_t embedding_layernorm_code[5316]={0};
      uint8_t embedding_raw_source[49152]={0},embedding_parameters[3072]={0};
      double* embedding_layernorm_reference=NULL;
      uint8_t* key_packed=NULL;
      double* key_reference=NULL;
      uint8_t scores_code[2374]={0};
      uint8_t* scores_query_expected=NULL,*scores_key_expected=NULL;
      double* scores_reference=NULL;
      uint8_t softmax_code[3960]={0};
      uint8_t* softmax_scores_expected=NULL;
      double* softmax_reference=NULL;
      uint8_t* value_packed=NULL,*value_probabilities_expected=NULL;
      double* value_reference=NULL;
      uint8_t context_code[2548]={0};
      uint8_t* context_probabilities_expected=NULL,*context_value_expected=NULL;
      double* context_reference=NULL;
      uint8_t* attention_output_packed=NULL,*attention_context_expected=NULL;
      double* attention_output_reference=NULL;
      uint8_t attention_residual_code[94]={0};
      uint8_t* attention_projected_expected=NULL,*attention_residual_expected=NULL;
      uint8_t attention_layernorm_code[5316]={0},attention_layernorm_parameters[3072]={0};
      double* attention_layernorm_reference=NULL;
      uint8_t standalone_ffn_pack_code[164]={0};
      uint8_t* standalone_ffn_pack_expected=NULL,*standalone_ffn_pack_source=NULL;
      uint8_t standalone_ffn_tile_code[1276]={0},standalone_ffn_tile_b[4096]={0};
      double* standalone_ffn_tile_reference=NULL;
      uint8_t* standalone_ffn_k6_b=NULL,*standalone_ffn_k6_prior=NULL;
      uint8_t standalone_ffn_k6_accum_code[1458]={0};
      double* standalone_ffn_k6_reference=NULL;
      uint8_t* standalone_n1_b=NULL;
      double* standalone_n1_reference=NULL;
      uint8_t* standalone_expand_b=NULL;
      double* standalone_expand_reference=NULL;
      uint8_t* standalone_wide_bias=NULL,*standalone_wide_prior=NULL;
      uint8_t* standalone_wide_biased=NULL;
      double* standalone_wide_gelu_reference=NULL;
      uint8_t* standalone_act_bias=NULL,*standalone_act_prior=NULL;
      uint8_t* standalone_act_biased=NULL;
      double* standalone_act_gelu_reference=NULL;
      uint8_t standalone_pair_pack_code[120]={0};
      uint8_t* standalone_contract_k24_packed=NULL,*standalone_contract_k24_b=NULL;
      double* standalone_contract_k24_reference=NULL;
      uint8_t* standalone_contract_n01_activated=NULL;
      uint8_t* standalone_contract_n01_packed=NULL,*standalone_contract_n01_b=NULL;
      double* standalone_contract_n01_reference=NULL;
      uint8_t* standalone_contract_n12_bias_prior=NULL;
      uint8_t* standalone_contract_n12_bias_operands=NULL;
      uint8_t* standalone_contract_n12_bias_expected=NULL;
      uint8_t standalone_contract_slot_b[4096]={0};
      double* standalone_contract_slot_reference=NULL;
      uint8_t* standalone_contract_activated=NULL;
      uint8_t standalone_pair_expected[4096]={0};
      uint8_t standalone_contract_b[4096]={0};
      double* standalone_contract_reference=NULL;
      uint8_t copy_code[56]={0};
      if(ffn_pair){
        const char* next_paths[2]={getenv("ORDERED_FFN_A_NEXT"),getenv("ORDERED_FFN_B_NEXT")};
        for(unsigned j=0;j<2;j++){
          if(!next_paths[j])return 2;
          FILE* source=fopen(next_paths[j],"rb");if(!source)return 2;
          int good=fread(next_inputs[j],1,4096,source)==4096 &&
                   fgetc(source)==EOF && !ferror(source);
          fclose(source);
          if(!good || !memcmp(next_inputs[j],expected_inputs[j+1],4096))return 2;
        }
        fprintf(stderr,"PURE FFN PAIR NEXT PASS: distinct 4096-byte A/B windows staged; no Submit.\n");
      }
      if(ffn_six){
        for(unsigned k=2;k<6;k++)for(unsigned j=0;j<2;j++){
          char key[32];snprintf(key,sizeof key,"ORDERED_FFN_%c_%u",j?'B':'A',k);
          const char* path=getenv(key);if(!path)return 2;
          FILE* source=fopen(path,"rb");if(!source)return 2;
          int good=fread(six_inputs[k-2][j],1,4096,source)==4096 &&
                   fgetc(source)==EOF && !ferror(source);
          fclose(source);if(!good)return 2;
        }
        fprintf(stderr,"PURE FFN SIX NEXT PASS: four further 4096-byte A/B pairs staged; no Submit.\n");
      }
      if(ffn_resident_pair){
        for(unsigned k=0;k<6;k++){
          char key[40];snprintf(key,sizeof key,"ORDERED_FFN_RESIDENT_B_%u",k);
          const char* path=getenv(key);if(!path)return 2;
          FILE* source=fopen(path,"rb");if(!source)return 2;
          int good=fread(resident_b[k],1,4096,source)==4096 &&
                   fgetc(source)==EOF && !ferror(source);
          fclose(source);if(!good)return 2;
        }
        fprintf(stderr,"PURE FFN RESIDENT PAIR NEXT PASS: six n=1 B windows staged; no Submit.\n");
      }
      if(ffn_standalone_query){
        size_t standalone_query_tiles=ffn_standalone_query_all?12u:1u;
        query_packed=malloc(standalone_query_tiles*49280u);
        query_source_expected=malloc(49152u);
        query_reference=malloc(standalone_query_tiles*1024u*sizeof(double));
        if(!query_packed || !query_source_expected || !query_reference)return 2;
        const char* paths[4]={getenv("ORDERED_FFN_STANDALONE_QUERY_CODE"),
                              getenv("ORDERED_FFN_STANDALONE_QUERY_PACKED"),
                              getenv("ORDERED_FFN_STANDALONE_QUERY_SOURCE"),
                              getenv("ORDERED_FFN_STANDALONE_QUERY_REFERENCE")};
        uint8_t* targets[4]={query_code,query_packed,query_source_expected,
                             (uint8_t*)query_reference};
        const size_t lengths[4]={sizeof query_code,standalone_query_tiles*49280u,49152u,
                                 standalone_query_tiles*1024u*sizeof(double)};
        for(unsigned j=0;j<4;j++){
          if(!paths[j])return 2;
          FILE* source=fopen(paths[j],"rb");if(!source)return 2;
          int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                   fgetc(source)==EOF && !ferror(source);
          fclose(source);if(!good)return 2;
        }
        if(!ffn_standalone_embedding_query)
          fprintf(stderr,"PURE STANDALONE QUERY NEXT PASS: exact 49152-byte prior GPU source, model tile, and 350-byte image staged in 29-object graph; no Submit.\n");
        if(ffn_standalone_embedding_query){
          embedding_layernorm_reference=malloc(12288u*sizeof(double));
          if(!embedding_layernorm_reference)return 2;
          const char* paths[4]={getenv("ORDERED_FFN_EMBEDDING_LN_CODE"),
                                getenv("ORDERED_FFN_EMBEDDING_SOURCE"),
                                getenv("ORDERED_FFN_EMBEDDING_PARAMETERS"),
                                getenv("ORDERED_FFN_EMBEDDING_LN_REFERENCE")};
          uint8_t* targets[4]={embedding_layernorm_code,embedding_raw_source,
                               embedding_parameters,(uint8_t*)embedding_layernorm_reference};
          const size_t lengths[4]={sizeof embedding_layernorm_code,
                                   sizeof embedding_raw_source,
                                   sizeof embedding_parameters,
                                   12288u*sizeof(double)};
          for(unsigned j=0;j<4;j++){
            if(!paths[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          fprintf(stderr,ffn_standalone_query_all?
                  "PURE EMBEDDING QUERY ALL NEXT PASS: raw MiniLM input, model embedding gamma/beta, 5316-byte LayerNorm, twelve query tiles, and FP64 references staged; no Submit.\n":
                  "PURE EMBEDDING QUERY NEXT PASS: raw MiniLM input, model embedding gamma/beta, 5316-byte LayerNorm and query image, and two FP64 references staged; no Submit.\n");
        }
        if(ffn_standalone_key_all){
          key_packed=malloc(12u*49280u);
          key_reference=malloc(12u*1024u*sizeof(double));
          if(!key_packed || !key_reference)return 2;
          const char* paths[2]={getenv("ORDERED_FFN_STANDALONE_KEY_PACKED"),
                                getenv("ORDERED_FFN_STANDALONE_KEY_REFERENCE")};
          uint8_t* targets[2]={key_packed,(uint8_t*)key_reference};
          const size_t lengths[2]={12u*49280u,12u*1024u*sizeof(double)};
          for(unsigned j=0;j<2;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING KEY ALL NEXT PASS: twelve packed model key tiles and FP64 references staged; no Submit.\n");
        }
        if(ffn_standalone_scores){
          scores_query_expected=malloc(49152u);
          scores_key_expected=malloc(49152u);
          scores_reference=malloc(12288u*sizeof(double));
          if(!scores_query_expected || !scores_key_expected || !scores_reference)return 2;
          const char* paths[4]={getenv("ORDERED_FFN_STANDALONE_SCORES_CODE"),
                                getenv("ORDERED_FFN_STANDALONE_SCORES_QUERY"),
                                getenv("ORDERED_FFN_STANDALONE_SCORES_KEY"),
                                getenv("ORDERED_FFN_STANDALONE_SCORES_REFERENCE")};
          uint8_t* targets[4]={scores_code,scores_query_expected,scores_key_expected,
                               (uint8_t*)scores_reference};
          const size_t lengths[4]={sizeof scores_code,49152u,49152u,
                                   12288u*sizeof(double)};
          for(unsigned j=0;j<4;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING SCORES NEXT PASS: 2374-byte authored image, exact prior GPU Q/K references, and FP64 score reference staged; no Submit.\n");
        }
        if(ffn_standalone_softmax){
          softmax_scores_expected=malloc(49152u);
          softmax_reference=malloc(12288u*sizeof(double));
          if(!softmax_scores_expected || !softmax_reference)return 2;
          const char* paths[3]={getenv("ORDERED_FFN_STANDALONE_SOFTMAX_CODE"),
                                getenv("ORDERED_FFN_STANDALONE_SOFTMAX_SCORES"),
                                getenv("ORDERED_FFN_STANDALONE_SOFTMAX_REFERENCE")};
          uint8_t* targets[3]={softmax_code,softmax_scores_expected,
                               (uint8_t*)softmax_reference};
          const size_t lengths[3]={sizeof softmax_code,49152u,
                                   12288u*sizeof(double)};
          for(unsigned j=0;j<3;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING SOFTMAX NEXT PASS: 3960-byte authored row-softmax image, exact prior GPU scores, and FP64 probability reference staged; no Submit.\n");
        }
        if(ffn_standalone_value){
          value_packed=malloc(12u*49280u);
          value_probabilities_expected=malloc(49152u);
          value_reference=malloc(12u*1024u*sizeof(double));
          if(!value_packed || !value_probabilities_expected || !value_reference)return 2;
          const char* paths[3]={getenv("ORDERED_FFN_STANDALONE_VALUE_PACKED"),
                                getenv("ORDERED_FFN_STANDALONE_VALUE_PROBABILITIES"),
                                getenv("ORDERED_FFN_STANDALONE_VALUE_REFERENCE")};
          uint8_t* targets[3]={value_packed,value_probabilities_expected,
                               (uint8_t*)value_reference};
          const size_t lengths[3]={12u*49280u,49152u,12u*1024u*sizeof(double)};
          for(unsigned j=0;j<3;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING VALUE NEXT PASS: twelve packed model value tiles, exact prior GPU probabilities, and FP64 reference staged; no Submit.\n");
        }
        if(ffn_standalone_context){
          context_probabilities_expected=malloc(49152u);
          context_value_expected=malloc(49152u);
          context_reference=malloc(12288u*sizeof(double));
          if(!context_probabilities_expected || !context_value_expected ||
             !context_reference)return 2;
          const char* paths[4]={getenv("ORDERED_FFN_STANDALONE_CONTEXT_CODE"),
                                getenv("ORDERED_FFN_STANDALONE_CONTEXT_PROBABILITIES"),
                                getenv("ORDERED_FFN_STANDALONE_CONTEXT_VALUE"),
                                getenv("ORDERED_FFN_STANDALONE_CONTEXT_REFERENCE")};
          uint8_t* targets[4]={context_code,context_probabilities_expected,
                               context_value_expected,(uint8_t*)context_reference};
          const size_t lengths[4]={sizeof context_code,49152u,49152u,
                                   12288u*sizeof(double)};
          for(unsigned j=0;j<4;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING CONTEXT NEXT PASS: 2548-byte authored image, exact prior GPU probabilities/value, and FP64 context reference staged; no Submit.\n");
        }
        if(ffn_standalone_output){
          attention_output_packed=malloc(12u*49280u);
          attention_context_expected=malloc(49152u);
          attention_output_reference=malloc(12u*1024u*sizeof(double));
          if(!attention_output_packed || !attention_context_expected ||
             !attention_output_reference)return 2;
          const char* paths[3]={getenv("ORDERED_FFN_STANDALONE_OUTPUT_PACKED"),
                                getenv("ORDERED_FFN_STANDALONE_OUTPUT_CONTEXT"),
                                getenv("ORDERED_FFN_STANDALONE_OUTPUT_REFERENCE")};
          uint8_t* targets[3]={attention_output_packed,attention_context_expected,
                               (uint8_t*)attention_output_reference};
          const size_t lengths[3]={12u*49280u,49152u,12u*1024u*sizeof(double)};
          for(unsigned j=0;j<3;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING OUTPUT NEXT PASS: twelve packed model attention-output tiles, exact prior GPU context, and FP64 reference staged; no Submit.\n");
        }
        if(ffn_standalone_residual){
          attention_projected_expected=malloc(49152u);
          attention_residual_expected=malloc(49152u);
          if(!attention_projected_expected || !attention_residual_expected)return 2;
          const char* paths[3]={getenv("ORDERED_FFN_STANDALONE_RESIDUAL_CODE"),
                                getenv("ORDERED_FFN_STANDALONE_RESIDUAL_PROJECTED"),
                                getenv("ORDERED_FFN_STANDALONE_RESIDUAL_EXPECTED")};
          uint8_t* targets[3]={attention_residual_code,attention_projected_expected,
                               attention_residual_expected};
          const size_t lengths[3]={sizeof attention_residual_code,49152u,49152u};
          for(unsigned j=0;j<3;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING RESIDUAL NEXT PASS: 94-byte authored in-place add, exact prior GPU projection, and predicted FP32 residual staged; no Submit.\n");
        }
        if(ffn_standalone_attention_ln){
          attention_layernorm_reference=malloc(12288u*sizeof(double));
          if(!attention_layernorm_reference)return 2;
          const char* paths[4]={getenv("ORDERED_FFN_STANDALONE_ATTENTION_LN_CODE"),
                                getenv("ORDERED_FFN_STANDALONE_ATTENTION_LN_PARAMETERS"),
                                getenv("ORDERED_FFN_STANDALONE_ATTENTION_LN_REFERENCE"),
                                getenv("ORDERED_FFN_STANDALONE_ATTENTION_LN_RESIDUAL")};
          uint8_t* targets[4]={attention_layernorm_code,attention_layernorm_parameters,
                               (uint8_t*)attention_layernorm_reference,attention_residual_expected};
          const size_t lengths[4]={sizeof attention_layernorm_code,
                                   sizeof attention_layernorm_parameters,
                                   12288u*sizeof(double),49152u};
          for(unsigned j=0;j<4;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING ATTENTION LN NEXT PASS: 5316-byte authored image, exact prior GPU residual, model gamma/beta, and FP64 reference staged; no Submit.\n");
        }
        if(ffn_standalone_ffn_pack){
          standalone_ffn_pack_expected=malloc(24576u);
          standalone_ffn_pack_source=malloc(49152u);
          if(!standalone_ffn_pack_expected || !standalone_ffn_pack_source)return 2;
          const char* paths[3]={getenv("ORDERED_FFN_STANDALONE_FFN_PACK_CODE"),
                                getenv("ORDERED_FFN_STANDALONE_FFN_PACK_EXPECTED"),
                                getenv("ORDERED_FFN_STANDALONE_FFN_PACK_SOURCE")};
          uint8_t* targets[3]={standalone_ffn_pack_code,standalone_ffn_pack_expected,
                               standalone_ffn_pack_source};
          const size_t lengths[3]={sizeof standalone_ffn_pack_code,24576u,49152u};
          for(unsigned j=0;j<3;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING FFN PACK NEXT PASS: 164-byte authored pack image and 24576 exact FP16 output bytes staged from prior GPU attention LayerNorm; no Submit.\n");
          if(ffn_standalone_residual_keep)
            fprintf(stderr,"PURE EMBEDDING FFN RESIDUAL KEEP NEXT PASS: twelve 4-KiB GPU copies of exact attention LayerNorm output to allocation 29, then full FFN contraction and bias; no Submit.\n");
        }
        if(ffn_standalone_ffn_tile){
          standalone_ffn_tile_reference=malloc(1024u*sizeof(double));
          if(!standalone_ffn_tile_reference)return 2;
          const char* paths[3]={getenv("ORDERED_FFN_STANDALONE_FFN_TILE_CODE"),
                                getenv("ORDERED_FFN_STANDALONE_FFN_TILE_B"),
                                getenv("ORDERED_FFN_STANDALONE_FFN_TILE_REFERENCE")};
          uint8_t* targets[3]={standalone_ffn_tile_code,standalone_ffn_tile_b,
                               (uint8_t*)standalone_ffn_tile_reference};
          const size_t lengths[3]={sizeof standalone_ffn_tile_code,4096u,
                                   1024u*sizeof(double)};
          for(unsigned j=0;j<3;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING FFN TILE NEXT PASS: 1276-byte authored TensorOps image, real model B[0,0], and FP64 reference staged; no Submit.\n");
        }
        if(ffn_standalone_ffn_k6){
          standalone_ffn_k6_b=malloc(5u*4096u);
          standalone_ffn_k6_prior=malloc(4096u);
          standalone_ffn_k6_reference=malloc(5u*1024u*sizeof(double));
          if(!standalone_ffn_k6_b || !standalone_ffn_k6_prior ||
             !standalone_ffn_k6_reference)return 2;
          const char* paths[4]={getenv("ORDERED_FFN_STANDALONE_FFN_K6_ACCUM_CODE"),
                                getenv("ORDERED_FFN_STANDALONE_FFN_K6_B"),
                                getenv("ORDERED_FFN_STANDALONE_FFN_K6_PRIOR"),
                                getenv("ORDERED_FFN_STANDALONE_FFN_K6_REFERENCE")};
          uint8_t* targets[4]={standalone_ffn_k6_accum_code,standalone_ffn_k6_b,standalone_ffn_k6_prior,
                               (uint8_t*)standalone_ffn_k6_reference};
          const size_t lengths[4]={sizeof standalone_ffn_k6_accum_code,
                                   5u*4096u,4096u,5u*1024u*sizeof(double)};
          for(unsigned j=0;j<4;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING FFN K6 NEXT PASS: 1458-byte accumulating image, five model B blocks, exact GPU k0 prior, and five cumulative FP64 references staged; no Submit.\n");
        }
        if(ffn_standalone_n1_resident){
          const char* copy_path=getenv("ORDERED_FFN_COPY_CODE");
          if(!copy_path)return 2;
          FILE* copy_file=fopen(copy_path,"rb");if(!copy_file)return 2;
          int copy_good=fread(copy_code,1,sizeof copy_code,copy_file)==sizeof copy_code &&
                        fgetc(copy_file)==EOF && !ferror(copy_file);
          fclose(copy_file);if(!copy_good)return 2;
          standalone_n1_b=malloc(6u*4096u);
          standalone_n1_reference=malloc(6u*1024u*sizeof(double));
          if(!standalone_n1_b || !standalone_n1_reference)return 2;
          const char* paths[2]={getenv("ORDERED_FFN_STANDALONE_N1_B"),
                                getenv("ORDERED_FFN_STANDALONE_N1_REFERENCE")};
          uint8_t* targets[2]={standalone_n1_b,(uint8_t*)standalone_n1_reference};
          const size_t lengths[2]={6u*4096u,6u*1024u*sizeof(double)};
          for(unsigned j=0;j<2;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          memset(mapped[29],0x5a,(size_t)sizes[29]);
          memset(mapped[29]+0x1000u,0,4096u);
          unsigned good=1;
          for(unsigned i=0;i<0x1000u;i++)good&=mapped[29][i]==0x5a;
          for(unsigned i=0;i<4096u;i++)good&=mapped[29][0x1000u+i]==0;
          for(unsigned i=0x2000u;i<(unsigned)sizes[29];i++)good&=mapped[29][i]==0x5a;
          if(!good || !no_agx_metal())return 2;
          fprintf(stderr,getenv("ORDERED_FFN_N1_KNOWN_C")?
                  "PURE EMBEDDING FFN N1 KNOWN C NEXT PASS: six model B blocks, existing allocation-2 C slot, appended-allocation guards, and cumulative FP64 references staged; no Submit.\n":
                  "PURE EMBEDDING FFN N1 RESIDENT NEXT PASS: 256-KiB appended allocation, six model B blocks, zero 4-KiB C slot, guards, and cumulative FP64 references staged; no Submit.\n");
        }
        if(ffn_standalone_expand_all){
          standalone_expand_b=malloc(46u*6u*4096u);
          standalone_expand_reference=malloc(46u*6u*1024u*sizeof(double));
          if(!standalone_expand_b || !standalone_expand_reference)return 2;
          const char* paths[2]={getenv("ORDERED_FFN_EXPAND_B"),
                                getenv("ORDERED_FFN_EXPAND_REFERENCE")};
          uint8_t* targets[2]={standalone_expand_b,(uint8_t*)standalone_expand_reference};
          const size_t lengths[2]={46u*6u*4096u,46u*6u*1024u*sizeof(double)};
          for(unsigned j=0;j<2;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING FFN EXPAND ALL NEXT PASS: n2-n47 model B blocks and cumulative FP64 references staged; existing allocation-2 and allocation-17 C slots selected after n1. No Submit.\n");
        }
        if(ffn_standalone_wide_probe){
          standalone_wide_bias=malloc(4096u);
          standalone_wide_prior=malloc(4096u);
          standalone_wide_biased=malloc(4096u);
          standalone_wide_gelu_reference=malloc(1024u*sizeof(double));
          if(!standalone_wide_bias || !standalone_wide_prior ||
             !standalone_wide_biased || !standalone_wide_gelu_reference)return 2;
          const char* paths[6]={getenv("ORDERED_FFN_WIDE_BIAS_CODE"),
                                getenv("ORDERED_FFN_WIDE_GELU_CODE"),
                                getenv("ORDERED_FFN_WIDE_BIAS"),
                                getenv("ORDERED_FFN_WIDE_PRIOR"),
                                getenv("ORDERED_FFN_WIDE_BIASED"),
                                getenv("ORDERED_FFN_WIDE_GELU_REFERENCE")};
          uint8_t* targets[6]={bias_code,gelu_code,standalone_wide_bias,
                               standalone_wide_prior,standalone_wide_biased,
                               (uint8_t*)standalone_wide_gelu_reference};
          const size_t lengths[6]={sizeof bias_code,sizeof gelu_code,4096u,4096u,
                                   4096u,1024u*sizeof(double)};
          for(unsigned j=0;j<6;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING FFN WIDE PROBE NEXT PASS: 1024-value n47 in-place bias and GELU images, repeated model bias, exact prior GPU tile, and FP64 GELU reference staged; no Submit.\n");
        }
        if(ffn_standalone_act_all){
          standalone_act_bias=malloc(48u*4096u);
          standalone_act_prior=malloc(48u*4096u);
          standalone_act_biased=malloc(48u*4096u);
          standalone_act_gelu_reference=malloc(48u*1024u*sizeof(double));
          if(!standalone_act_bias || !standalone_act_prior ||
             !standalone_act_biased || !standalone_act_gelu_reference)return 2;
          const char* paths[6]={getenv("ORDERED_FFN_ACT_BIAS_CODE"),
                                getenv("ORDERED_FFN_ACT_GELU_CODE"),
                                getenv("ORDERED_FFN_ACT_BIAS"),
                                getenv("ORDERED_FFN_ACT_PRIOR"),
                                getenv("ORDERED_FFN_ACT_BIASED"),
                                getenv("ORDERED_FFN_ACT_GELU_REFERENCE")};
          uint8_t* targets[6]={bias_code,gelu_code,standalone_act_bias,
                               standalone_act_prior,standalone_act_biased,
                               (uint8_t*)standalone_act_gelu_reference};
          const size_t lengths[6]={sizeof bias_code,sizeof gelu_code,
                                   48u*4096u,48u*4096u,48u*4096u,
                                   48u*1024u*sizeof(double)};
          for(unsigned j=0;j<6;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING FFN ACT ALL NEXT PASS: 48 GPU tiles, repeated model bias operands, exact biased FP32 outputs, and FP64 GELU references staged for two wide phases; no Submit.\n");
        }
        if(ffn_standalone_contract_k0){
          standalone_contract_activated=malloc(48u*4096u);
          standalone_contract_reference=malloc(1024u*sizeof(double));
          if(!standalone_contract_activated || !standalone_contract_reference)return 2;
          const char* paths[5]={getenv("ORDERED_FFN_CONTRACT_PAIR_CODE"),
                                getenv("ORDERED_FFN_CONTRACT_ACTIVATED"),
                                getenv("ORDERED_FFN_CONTRACT_PACKED"),
                                getenv("ORDERED_FFN_CONTRACT_B"),
                                getenv("ORDERED_FFN_CONTRACT_REFERENCE")};
          uint8_t* targets[5]={standalone_pair_pack_code,standalone_contract_activated,
                               standalone_pair_expected,standalone_contract_b,
                               (uint8_t*)standalone_contract_reference};
          const size_t lengths[5]={sizeof standalone_pair_pack_code,
                                   48u*4096u,4096u,4096u,1024u*sizeof(double)};
          for(unsigned j=0;j<5;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING FFN CONTRACT K0 NEXT PASS: two 32x32 activated source tiles, authored pair pack, model contraction B, and FP64 TensorOps reference staged; no Submit.\n");
        }
        if(ffn_standalone_contract_k24){
          standalone_contract_k24_packed=malloc(23u*4096u);
          standalone_contract_k24_b=malloc(23u*4096u);
          standalone_contract_k24_reference=malloc(23u*1024u*sizeof(double));
          if(!standalone_contract_k24_packed || !standalone_contract_k24_b ||
             !standalone_contract_k24_reference)return 2;
          const char* paths[4]={getenv("ORDERED_FFN_CONTRACT_K24_PACKED"),
                                getenv("ORDERED_FFN_CONTRACT_K24_B"),
                                getenv("ORDERED_FFN_CONTRACT_K24_REFERENCE"),
                                getenv("ORDERED_FFN_CONTRACT_K24_ACCUM_CODE")};
          uint8_t* targets[4]={standalone_contract_k24_packed,
                               standalone_contract_k24_b,
                               (uint8_t*)standalone_contract_k24_reference,
                               standalone_ffn_k6_accum_code};
          const size_t lengths[4]={23u*4096u,23u*4096u,
                                   23u*1024u*sizeof(double),
                                   sizeof standalone_ffn_k6_accum_code};
          for(unsigned j=0;j<4;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING FFN CONTRACT K24 NEXT PASS: remaining 23 GPU pair packs, model B blocks, accumulating TensorOps image, and cumulative FP64 references staged; no Submit.\n");
        }
        if(ffn_standalone_contract_n01 || ffn_standalone_contract_n8 ||
           ffn_standalone_contract_n12){
          unsigned count=ffn_standalone_contract_n12?12u:
                         ffn_standalone_contract_n8?8u:2u;
          const char* prefix=ffn_standalone_contract_n12?
            "ORDERED_FFN_CONTRACT_N12_":ffn_standalone_contract_n8?
            "ORDERED_FFN_CONTRACT_N8_":"ORDERED_FFN_CONTRACT_N01_";
          char names[6][80];
          const char* suffixes[6]={"ACTIVATED","PACKED","B","REFERENCE",
                                   "PAIR_CODE","ACCUM_CODE"};
          for(unsigned j=0;j<6;j++)snprintf(names[j],sizeof names[j],"%s%s",prefix,suffixes[j]);
          standalone_contract_n01_activated=malloc(48u*4096u);
          standalone_contract_n01_packed=malloc(24u*4096u);
          standalone_contract_n01_b=malloc(count*24u*4096u);
          standalone_contract_n01_reference=malloc(count*24u*1024u*sizeof(double));
          if(!standalone_contract_n01_activated || !standalone_contract_n01_packed ||
             !standalone_contract_n01_b || !standalone_contract_n01_reference)return 2;
          const char* paths[6]={getenv(names[0]),getenv(names[1]),getenv(names[2]),
                                getenv(names[3]),getenv(names[4]),getenv(names[5])};
          uint8_t* targets[6]={standalone_contract_n01_activated,
                               standalone_contract_n01_packed,
                               standalone_contract_n01_b,
                               (uint8_t*)standalone_contract_n01_reference,
                               standalone_pair_pack_code,
                               standalone_ffn_k6_accum_code};
          const size_t lengths[6]={48u*4096u,24u*4096u,
                                   count*24u*4096u,count*24u*1024u*sizeof(double),
                                   sizeof standalone_pair_pack_code,
                                   sizeof standalone_ffn_k6_accum_code};
          for(unsigned j=0;j<6;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          if(ffn_standalone_contract_n12)
            fprintf(stderr,"PURE EMBEDDING FFN CONTRACT N12 NEXT PASS: 24 exact packed K blocks, twelve model output-column tiles, and 288 cumulative FP64 references staged; no Submit.\n");
          else if(ffn_standalone_contract_n8)
            fprintf(stderr,"PURE EMBEDDING FFN CONTRACT N8 NEXT PASS: 24 exact packed K blocks, eight model output-column tiles, and 192 cumulative FP64 references staged; no Submit.\n");
          else
            fprintf(stderr,"PURE EMBEDDING FFN CONTRACT N01 NEXT PASS: 24 exact packed K blocks, two model output-column tiles, and 48 cumulative FP64 references staged; no Submit.\n");
        }
        if(ffn_standalone_contract_n12_bias){
          const char* names[3]={"ORDERED_FFN_CONTRACT_N12_BIAS_PRIOR",
                                "ORDERED_FFN_CONTRACT_N12_BIAS_OPERANDS",
                                "ORDERED_FFN_CONTRACT_N12_BIAS_EXPECTED"};
          uint8_t** targets[3]={&standalone_contract_n12_bias_prior,
                                &standalone_contract_n12_bias_operands,
                                &standalone_contract_n12_bias_expected};
          for(unsigned j=0;j<3;j++){
            const char* path=getenv(names[j]);if(!path)return 2;
            *targets[j]=malloc(12u*4096u);if(!*targets[j])return 2;
            FILE* file=fopen(path,"rb");if(!file)return 2;
            int good=fread(*targets[j],1,12u*4096u,file)==12u*4096u &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING FFN CONTRACT N12 BIAS NEXT PASS: twelve retained GPU C tiles, repeated model bias, 56-byte add image, and exact FP32 targets staged; no Submit.\n");
        }
        if(ffn_contract_slot_text){
          standalone_contract_slot_reference=malloc(1024u*sizeof(double));
          if(!standalone_contract_slot_reference)return 2;
          const char* paths[2]={getenv("ORDERED_FFN_CONTRACT_SLOT_B"),
                                getenv("ORDERED_FFN_CONTRACT_SLOT_REFERENCE")};
          uint8_t* targets[2]={standalone_contract_slot_b,
                               (uint8_t*)standalone_contract_slot_reference};
          const size_t lengths[2]={4096u,1024u*sizeof(double)};
          for(unsigned j=0;j<2;j++){
            if(!paths[j])return 2;
            FILE* file=fopen(paths[j],"rb");if(!file)return 2;
            int good=fread(targets[j],1,lengths[j],file)==lengths[j] &&
                     fgetc(file)==EOF && !ferror(file);
            fclose(file);if(!good)return 2;
          }
          fprintf(stderr,"PURE EMBEDDING FFN CONTRACT SLOT PROBE NEXT PASS: slot=%u, GPU-packed K23 A, model B[8,23], and FP64 reference staged; no Submit.\n",
                  ffn_contract_slot);
        }
      }
      if(ffn_resident_full){
        const char* path=getenv("ORDERED_FFN_FULL_B_BLOB");if(!path)return 2;
        full_b=malloc(48u*6u*4096u);if(!full_b)return 2;
        FILE* source=fopen(path,"rb");if(!source)return 2;
        int good=fread(full_b,1,48u*6u*4096u,source)==48u*6u*4096u &&
                 fgetc(source)==EOF && !ferror(source);
        fclose(source);
        if(!good || memcmp(full_b,expected_inputs[2],4096))return 2;
        for(unsigned n=0;n<48;n++){
          uint8_t* slot=n<24?mapped[2]+0x8000u+n*4096u:
                              mapped[17]+0x8000u+(n-24u)*4096u;
          if(!slot)return 2;
          for(unsigned i=0;i<4096;i++){
            unsigned expected=(n==25 && i<0x80u)?0xc7u:0u;
            if(slot[i]!=expected)return 2;
          }
        }
        fprintf(stderr,"PURE FFN RESIDENT FULL NEXT PASS: 288 B windows and 48 distinct C slots staged; no Submit.\n");
        if(ffn_full_append_add7){
          uint32_t planned_packet=0;
          memcpy(&planned_packet,mapped[23]+0x40,4);
          if(memcmp(mapped[0]+0x6c0,expected_inputs[0],1458u) ||
             planned_packet!=0x0e5806c7u)return 2;
          fprintf(stderr,"PURE FFN APPEND NEXT PASS: scalar +7 code and packet switch nominated after 288 tensor Submits; no Submit.\n");
        }
        if(ffn_full_append_gelu || ffn_full_bias_gelu || ffn_full_inplace || ffn_full_activation){
          const char* path=getenv("ORDERED_FFN_APPEND_GELU_CODE");if(!path)return 2;
          FILE* source=fopen(path,"rb");if(!source)return 2;
          int good=fread(gelu_code,1,sizeof gelu_code,source)==sizeof gelu_code &&
                   fgetc(source)==EOF && !ferror(source);
          fclose(source);if(!good)return 2;
          fprintf(stderr,"PURE FFN GELU NEXT PASS: 536-byte delivered code and two bindings staged after 288 tensor Submits; no Submit.\n");
        }
        if(ffn_full_append_bias || ffn_full_bias_gelu || ffn_full_inplace || ffn_full_activation){
          const char* paths[2]={getenv("ORDERED_FFN_APPEND_BIAS_CODE"),
                                getenv("ORDERED_FFN_APPEND_BIAS_VALUES")};
          uint8_t* targets[2]={bias_code,bias_values};
          const size_t lengths[2]={sizeof bias_code,sizeof bias_values};
          for(unsigned j=0;j<2;j++){
            if(!paths[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          fprintf(stderr,"PURE FFN BIAS NEXT PASS: 56-byte compiled FP32 add and 32 bias values staged after 288 tensor Submits; no Submit.\n");
        }
        if(ffn_full_bias_gelu)
          fprintf(stderr,"PURE FFN CHAIN NEXT PASS: bias output feeds delivered GELU on same queue/pages; no Submit.\n");
        if(ffn_full_inplace)
          fprintf(stderr,"PURE FFN INPLACE NEXT PASS: bias and GELU read/write the same n=0 tile segment; no Submit.\n");
        if(ffn_full_activation){
          const char* path=getenv("ORDERED_FFN_FULL_BIAS_BLOB");if(!path)return 2;
          full_bias=malloc(48u*128u);if(!full_bias)return 2;
          FILE* source=fopen(path,"rb");if(!source)return 2;
          int good=fread(full_bias,1,48u*128u,source)==48u*128u &&
                   fgetc(source)==EOF && !ferror(source);
          fclose(source);if(!good || memcmp(full_bias,bias_values,128))return 2;
          fprintf(stderr,"PURE FFN FULL ACTIVATION NEXT PASS: 48 bias windows and two scalar images staged; no Submit.\n");
        }
        if(ffn_pack32){
          const char* paths[2]={getenv("ORDERED_FFN_PACK_CODE"),getenv("ORDERED_FFN_PACK_EXPECTED")};
          uint8_t* targets[2]={pack_code,pack_expected};
          const size_t lengths[2]={sizeof pack_code,sizeof pack_expected};
          for(unsigned j=0;j<2;j++){
            if(!paths[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          for(unsigned i=0;i<192;i++)if(mapped[17][0xfc0u+i])return 2;
          fprintf(stderr,"PURE FFN PACK32 NEXT PASS: delivered 178-byte image and 64-byte expected half output staged in allocation 17; no Submit.\n");
        }
        if(ffn_pack_full){
          const char* code_path=getenv("ORDERED_FFN_PACK_CODE");
          const char* expected_path=getenv("ORDERED_FFN_PACK_FULL_EXPECTED");
          if(!code_path || !expected_path)return 2;
          FILE* source=fopen(code_path,"rb");if(!source)return 2;
          int good=fread(pack_code,1,sizeof pack_code,source)==sizeof pack_code &&
                   fgetc(source)==EOF && !ferror(source);
          fclose(source);if(!good)return 2;
          pack_full_expected=malloc(24u*4096u);if(!pack_full_expected)return 2;
          source=fopen(expected_path,"rb");if(!source)return 2;
          good=fread(pack_full_expected,1,24u*4096u,source)==24u*4096u &&
               fgetc(source)==EOF && !ferror(source);
          fclose(source);if(!good)return 2;
          for(unsigned i=0;i<4224;i++)if(mapped[17][0xfc0u+i])return 2;
          fprintf(stderr,"PURE FFN PACKFULL NEXT PASS: delivered 178-byte image and 24 exact 4 KiB half blocks staged; no Submit.\n");
        }
        if(ffn_contract_tile){
          const char* path=getenv("ORDERED_FFN_CONTRACT_B");if(!path)return 2;
          FILE* source=fopen(path,"rb");if(!source)return 2;
          int good=fread(contract_b,1,sizeof contract_b,source)==sizeof contract_b &&
                   fgetc(source)==EOF && !ferror(source);
          fclose(source);if(!good)return 2;
          uint8_t* c_stage=ffn_contract_alloc0?mapped[0]+0x1000u:
                           ffn_contract_alloc1?mapped[1]+0x1000u:mapped[17]+0x3000u;
          for(unsigned i=0;i<4096;i++)if(c_stage[i])return 2;
          fprintf(stderr,"PURE FFN CONTRACT TILE NEXT PASS: K23 packed A from GPU scratch, committed B, and zero C staged; no Submit.\n");
          if(ffn_contract_alloc0)
            fprintf(stderr,"PURE FFN CONTRACT ALLOC0 NEXT PASS: zero 4 KiB C slot at allocation-0 +0x1000 staged; no Submit.\n");
          if(ffn_contract_alloc1)
            fprintf(stderr,"PURE FFN CONTRACT ALLOC1 NEXT PASS: zero 4 KiB C slot at allocation-1 +0x1000 staged; no Submit.\n");
        }
        if(ffn_contract_reduce || ffn_consume_persistent){
          const char* path=getenv("ORDERED_FFN_CONTRACT_B_FULL");if(!path)return 2;
          contract_b_full=malloc(24u*4096u);if(!contract_b_full)return 2;
          FILE* source=fopen(path,"rb");if(!source)return 2;
          int good=fread(contract_b_full,1,24u*4096u,source)==24u*4096u &&
                   fgetc(source)==EOF && !ferror(source);
          fclose(source);if(!good)return 2;
          for(unsigned i=0;i<4096;i++)if(mapped[17][0x3000u+i])return 2;
          fprintf(stderr,"PURE FFN CONTRACT REDUCE NEXT PASS: 24 committed B blocks and zero resident C staged; no Submit.\n");
          if(ffn_consume_persistent)
            fprintf(stderr,"PURE FFN CONTRACT PERSISTENT NEXT PASS: tensor A bound to 24 retained even activation slots; no Submit.\n");
        }
        if(ffn_contract_all){
          const char* path=getenv("ORDERED_FFN_CONTRACT_B_ALL");if(!path)return 2;
          contract_b_all=malloc(12u*24u*4096u);if(!contract_b_all)return 2;
          FILE* source=fopen(path,"rb");if(!source)return 2;
          int good=fread(contract_b_all,1,12u*24u*4096u,source)==12u*24u*4096u &&
                   fgetc(source)==EOF && !ferror(source);
          fclose(source);if(!good)return 2;
          fprintf(stderr,"PURE FFN CONTRACT ALL NEXT PASS: 12 B groups, GPU zero of 12 odd C slots, 288 tensor reductions staged; no Submit.\n");
        }
        if(ffn_contract_bias){
          const char* paths[2]={getenv("ORDERED_FFN_CONTRACT_BIAS_VALUES"),
                                getenv("ORDERED_FFN_CONTRACT_BIAS_EXPECTED")};
          uint8_t** targets[2]={&contract_bias_values,&contract_bias_expected};
          const size_t lengths[2]={1536u,49152u};
          for(unsigned j=0;j<2;j++){
            if(!paths[j])return 2;
            *targets[j]=malloc(lengths[j]);if(!*targets[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(*targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          fprintf(stderr,"PURE FFN CONTRACT BIAS NEXT PASS: compiled 56-byte FP32 add, 384 bias values, and exact GPU-C-derived reference staged; no Submit.\n");
        }
        if(ffn_residual_add){
          const char* paths[2]={getenv("ORDERED_FFN_RESIDUAL_SOURCE"),
                                getenv("ORDERED_FFN_RESIDUAL_EXPECTED")};
          uint8_t** targets[2]={&residual_source,&residual_expected};
          for(unsigned j=0;j<2;j++){
            if(!paths[j])return 2;
            *targets[j]=malloc(49152u);if(!*targets[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(*targets[j],1,49152u,source)==49152u &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          fprintf(stderr,"PURE FFN RESIDUAL ADD NEXT PASS: 32x384 source and GPU-biased-derived exact target staged for in-place FP32 add; no Submit.\n");
        }
        if(ffn_gather_residual){
          const char* path=getenv("ORDERED_FFN_GATHER_OLD_TARGET");if(!path)return 2;
          gather_old_target=malloc(49152u);if(!gather_old_target)return 2;
          FILE* source=fopen(path,"rb");if(!source)return 2;
          int good=fread(gather_old_target,1,49152u,source)==49152u &&
                   fgetc(source)==EOF && !ferror(source);
          fclose(source);if(!good)return 2;
          fprintf(stderr,"PURE FFN GATHER NEXT PASS: 384 GPU row copies from residual C tiles into contiguous allocation-17 target staged; no Submit.\n");
        }
        if(ffn_coop_layernorm){
          const char* paths[4]={getenv("ORDERED_FFN_COOP_CODE"),
                                getenv("ORDERED_FFN_COOP_GAMMA"),
                                getenv("ORDERED_FFN_COOP_BETA"),
                                getenv("ORDERED_FFN_COOP_REFERENCE")};
          uint8_t* targets[4]={coop_code,coop_gamma,coop_beta,NULL};
          const size_t lengths[4]={ffn_packed_ln?5316u:5076u,sizeof coop_gamma,
                                   sizeof coop_beta,12288u*sizeof(double)};
          coop_reference=malloc(lengths[3]);if(!coop_reference)return 2;
          targets[3]=(uint8_t*)coop_reference;
          for(unsigned j=0;j<4;j++){
            if(!paths[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          if(!mapped[22] || !mapped[25] || sizes[17]!=0x20000u ||
             *(uint32_t*)(mapped[22]+0xac)!=1u ||
             *(uint32_t*)(mapped[25]+0x14)!=1u ||
             *(uint32_t*)(mapped[25]+0x1c)!=32u ||
             memcmp(mapped[23]+0x38,"\x07\x00\x00\x0c\x00\x00\x00\x00",8))return 2;
          if(ffn_packed_ln)
            fprintf(stderr,"PURE FFN PACKED LN NEXT PASS: 5316-byte three-buffer image, 128-byte scratch, 32x32 launch, packed gamma/beta, and FP64 reference staged; no Submit.\n");
          else
            fprintf(stderr,"PURE FFN COOP LN NEXT PASS: 5076-byte four-buffer image, 128-byte scratch, 32x32 launch, gamma/beta, and FP64 reference staged; no Submit.\n");
        }
        if(ffn_four_binding){
          const char* paths[4]={getenv("ORDERED_FFN_FOUR_CODE"),
                                getenv("ORDERED_FFN_FOUR_GAMMA"),
                                getenv("ORDERED_FFN_FOUR_BETA"),
                                getenv("ORDERED_FFN_FOUR_EXPECTED")};
          uint8_t* targets[4]={four_code,four_gamma,four_beta,(uint8_t*)four_expected};
          const size_t lengths[4]={sizeof four_code,sizeof four_gamma,
                                   sizeof four_beta,sizeof four_expected};
          for(unsigned j=0;j<4;j++){
            if(!paths[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          if(memcmp(mapped[28]+0x1bb8,"\0\0\0\0\0\0\0\0",8))return 2;
          if(ffn_append_output)for(unsigned i=0;i<0x10000u;i++)
            if(mapped[29][i])return 2;
          fprintf(stderr,"PURE FFN FOUR BINDING NEXT PASS: 82-byte scalar four-buffer control and exact first32 output staged; no Submit.\n");
        }
        if(ffn_query_tile){
          const char* paths[4]={getenv("ORDERED_FFN_QUERY_CODE"),
                                getenv("ORDERED_FFN_QUERY_PACKED"),
                                getenv("ORDERED_FFN_QUERY_SOURCE"),
                                getenv("ORDERED_FFN_QUERY_REFERENCE")};
          query_source_expected=malloc(49152u);if(!query_source_expected)return 2;
          size_t query_tiles=ffn_query_all?12u:1u;
          query_packed=malloc(query_tiles*49280u);
          query_reference=malloc(query_tiles*1024u*sizeof(double));
          if(!query_packed || !query_reference)return 2;
          uint8_t* targets[4]={query_code,query_packed,query_source_expected,
                               (uint8_t*)query_reference};
          const size_t lengths[4]={sizeof query_code,query_tiles*49280u,
                                   49152u,query_tiles*1024u*sizeof(double)};
          for(unsigned j=0;j<4;j++){
            if(!paths[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          fprintf(stderr,"PURE FFN QUERY TILE NEXT PASS: 350-byte three-binding projection, one packed 32-column model-weight tile, and exact LN source staged; no Submit.\n");
        }
        if(ffn_key_all){
          key_packed=malloc(12u*49280u);
          key_reference=malloc(12u*1024u*sizeof(double));
          if(!key_packed || !key_reference)return 2;
          const char* paths[2]={getenv("ORDERED_FFN_KEY_PACKED"),
                                getenv("ORDERED_FFN_KEY_REFERENCE")};
          uint8_t* targets[2]={key_packed,(uint8_t*)key_reference};
          const size_t lengths[2]={12u*49280u,12u*1024u*sizeof(double)};
          for(unsigned j=0;j<2;j++){
            if(!paths[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          if(sizes[2]!=0x20000u || sizes[17]!=0x20000u)return 2;
          fprintf(stderr,"PURE FFN KEY ALL NEXT PASS: twelve packed model-weight tiles, resident query preservation, and upper allocation-2 output staged; no Submit.\n");
        }
        if(ffn_scores){
          scores_query_expected=malloc(49152u);
          scores_key_expected=malloc(49152u);
          scores_reference=malloc(12288u*sizeof(double));
          if(!scores_query_expected || !scores_key_expected || !scores_reference)return 2;
          const char* paths[4]={getenv("ORDERED_FFN_SCORES_CODE"),
                                getenv("ORDERED_FFN_SCORES_QUERY"),
                                getenv("ORDERED_FFN_SCORES_KEY"),
                                getenv("ORDERED_FFN_SCORES_REFERENCE")};
          uint8_t* targets[4]={scores_code,scores_query_expected,
                               scores_key_expected,(uint8_t*)scores_reference};
          const size_t lengths[4]={sizeof scores_code,49152u,49152u,
                                   12288u*sizeof(double)};
          for(unsigned j=0;j<4;j++){
            if(!paths[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          fprintf(stderr,"PURE FFN SCORES NEXT PASS: 2374-byte three-binding image, exact resident Q/K, 32x384 launch, and FP64 reference staged; no Submit.\n");
          if(ffn_scores_relocated)
            fprintf(stderr,"PURE FFN SCORES RELOCATED NEXT PASS: output selects dead allocation-17 parameter window, retaining LayerNorm source; no Submit.\n");
        }
        if(ffn_softmax){
          softmax_scores_expected=malloc(49152u);
          softmax_reference=malloc(12288u*sizeof(double));
          if(!softmax_scores_expected || !softmax_reference)return 2;
          const char* paths[3]={getenv("ORDERED_FFN_SOFTMAX_CODE"),
                                getenv("ORDERED_FFN_SOFTMAX_SCORES"),
                                getenv("ORDERED_FFN_SOFTMAX_REFERENCE")};
          uint8_t* targets[3]={softmax_code,softmax_scores_expected,
                               (uint8_t*)softmax_reference};
          const size_t lengths[3]={sizeof softmax_code,49152u,
                                   12288u*sizeof(double)};
          for(unsigned j=0;j<3;j++){
            if(!paths[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          fprintf(stderr,"PURE FFN SOFTMAX NEXT PASS: 3960-byte two-binding image, exact resident scores, 384x1 launch, and FP64 reference staged; no Submit.\n");
        }
        if(ffn_value_all){
          value_packed=malloc(12u*49280u);
          value_probabilities_expected=malloc(49152u);
          value_reference=malloc(12u*1024u*sizeof(double));
          if(!value_packed || !value_probabilities_expected || !value_reference)return 2;
          const char* paths[3]={getenv("ORDERED_FFN_VALUE_PACKED"),
                                getenv("ORDERED_FFN_VALUE_PROBABILITIES"),
                                getenv("ORDERED_FFN_VALUE_REFERENCE")};
          uint8_t* targets[3]={value_packed,value_probabilities_expected,
                               (uint8_t*)value_reference};
          const size_t lengths[3]={12u*49280u,49152u,12u*1024u*sizeof(double)};
          for(unsigned j=0;j<3;j++){
            if(!paths[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          fprintf(stderr,"PURE FFN VALUE ALL NEXT PASS: twelve packed model-weight tiles, exact resident probabilities, and dead-key output staged; no Submit.\n");
        }
        if(ffn_context){
          context_probabilities_expected=malloc(49152u);
          context_value_expected=malloc(49152u);
          context_reference=malloc(12288u*sizeof(double));
          if(!context_probabilities_expected || !context_value_expected ||
             !context_reference)return 2;
          const char* paths[4]={getenv("ORDERED_FFN_CONTEXT_CODE"),
                                getenv("ORDERED_FFN_CONTEXT_PROBABILITIES"),
                                getenv("ORDERED_FFN_CONTEXT_VALUE"),
                                getenv("ORDERED_FFN_CONTEXT_REFERENCE")};
          uint8_t* targets[4]={context_code,context_probabilities_expected,
                               context_value_expected,(uint8_t*)context_reference};
          const size_t lengths[4]={sizeof context_code,49152u,49152u,
                                   12288u*sizeof(double)};
          for(unsigned j=0;j<4;j++){
            if(!paths[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          fprintf(stderr,"PURE FFN CONTEXT NEXT PASS: 2548-byte three-binding image, exact GPU probabilities and V, 384x32 launch, and FP64 reference staged; no Submit.\n");
          if(ffn_context_relocated)
            fprintf(stderr,"PURE FFN CONTEXT RELOCATED NEXT PASS: consumed parameter window selected, LayerNorm source retained for residual; no Submit.\n");
        }
        if(ffn_attention_output_all){
          attention_output_packed=malloc(12u*49280u);
          attention_context_expected=malloc(49152u);
          attention_output_reference=malloc(12u*1024u*sizeof(double));
          if(!attention_output_packed || !attention_context_expected ||
             !attention_output_reference)return 2;
          const char* paths[3]={getenv("ORDERED_FFN_ATTENTION_OUTPUT_PACKED"),
                                getenv("ORDERED_FFN_ATTENTION_CONTEXT"),
                                getenv("ORDERED_FFN_ATTENTION_OUTPUT_REFERENCE")};
          uint8_t* targets[3]={attention_output_packed,attention_context_expected,
                               (uint8_t*)attention_output_reference};
          const size_t lengths[3]={12u*49280u,49152u,12u*1024u*sizeof(double)};
          for(unsigned j=0;j<3;j++){
            if(!paths[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          fprintf(stderr,"PURE FFN ATTENTION OUTPUT ALL NEXT PASS: twelve model output-dense tiles, exact resident context, saved LayerNorm source, and dead-V output staged; no Submit.\n");
        }
        if(ffn_attention_residual){
          attention_projected_expected=malloc(49152u);
          attention_residual_expected=malloc(49152u);
          if(!attention_projected_expected || !attention_residual_expected)return 2;
          const char* paths[3]={getenv("ORDERED_FFN_ATTENTION_RESIDUAL_CODE"),
                                getenv("ORDERED_FFN_ATTENTION_PROJECTED"),
                                getenv("ORDERED_FFN_ATTENTION_RESIDUAL_EXPECTED")};
          uint8_t* targets[3]={attention_residual_code,attention_projected_expected,
                               attention_residual_expected};
          const size_t lengths[3]={sizeof attention_residual_code,49152u,49152u};
          for(unsigned j=0;j<3;j++){
            if(!paths[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          fprintf(stderr,"PURE FFN ATTENTION RESIDUAL NEXT PASS: 94-byte three-binding image, exact resident projection/source, 384x32 grid, and FP32 output staged; no Submit.\n");
        }
        if(ffn_attention_layernorm){
          attention_layernorm_reference=malloc(12288u*sizeof(double));
          if(!attention_layernorm_reference)return 2;
          const char* paths[3]={getenv("ORDERED_FFN_ATTENTION_LAYERNORM_CODE"),
                                getenv("ORDERED_FFN_ATTENTION_LAYERNORM_PARAMETERS"),
                                getenv("ORDERED_FFN_ATTENTION_LAYERNORM_REFERENCE")};
          uint8_t* targets[3]={attention_layernorm_code,attention_layernorm_parameters,
                               (uint8_t*)attention_layernorm_reference};
          const size_t lengths[3]={sizeof attention_layernorm_code,
                                   sizeof attention_layernorm_parameters,
                                   12288u*sizeof(double)};
          for(unsigned j=0;j<3;j++){
            if(!paths[j])return 2;
            FILE* source=fopen(paths[j],"rb");if(!source)return 2;
            int good=fread(targets[j],1,lengths[j],source)==lengths[j] &&
                     fgetc(source)==EOF && !ferror(source);
            fclose(source);if(!good)return 2;
          }
          fprintf(stderr,"PURE FFN ATTENTION LAYERNORM NEXT PASS: 5316-byte cooperative three-binding image, model gamma/beta, exact GPU residual, and FP64 reference staged; no Submit.\n");
        }
        if(ffn_copy_packed || ffn_pack_persistent){
          const char* path=getenv("ORDERED_FFN_COPY_CODE");if(!path)return 2;
          FILE* source=fopen(path,"rb");if(!source)return 2;
          int good=fread(copy_code,1,sizeof copy_code,source)==sizeof copy_code &&
                   fgetc(source)==EOF && !ferror(source);
          fclose(source);if(!good)return 2;
          for(unsigned i=0;i<8192;i++)if(mapped[17][0x2000u+i])return 2;
          fprintf(stderr,"PURE FFN GPU COPY NEXT PASS: 56-byte authored integer-sum image, zero B, and zero 4 KiB target staged; no Submit.\n");
          if(ffn_copy_dead_tile)
            fprintf(stderr,"PURE FFN DEAD TILE NEXT PASS: activation slot n=46 selected for packed K23 after consumption; no Submit.\n");
          if(ffn_pack_persistent)
            fprintf(stderr,"PURE FFN PACK PERSISTENT NEXT PASS: 24 consumed even activation slots nominated for exact GPU-packed A; no Submit.\n");
        }
        if(ffn_consume_reused){
          const char* path=getenv("ORDERED_FFN_CONTRACT_B");if(!path)return 2;
          FILE* source=fopen(path,"rb");if(!source)return 2;
          int good=fread(contract_b,1,sizeof contract_b,source)==sizeof contract_b &&
                   fgetc(source)==EOF && !ferror(source);
          fclose(source);if(!good)return 2;
          for(unsigned i=0;i<4096;i++)if(mapped[17][0x3000u+i])return 2;
          fprintf(stderr,"PURE FFN CONSUME REUSED NEXT PASS: K23 B and zero C staged for tensor read from former tile n=46; no Submit.\n");
        }
      }
      uint64_t binding[3]={0};
      for(unsigned j=0;j<3;j++)memcpy(&binding[j],mapped[28]+0x1ba0u+8u*j,8);
      if(binding[0]!=0x10000034d80ull+shift-(a_binding_base?0x80u:0u) ||
         binding[1]!=0x10000035e80ull+shift-(b_binding_base?0x80u:0u) ||
         binding[2]!=(ffn_c_cross?0x10000060000ull:
                       ffn_c_slot?0x10000030000ull+ffn_c_slot:
                       0x10000036f80ull+shift-(c_binding_base?0x80u:0u)))return 2;
      uint32_t packet=0,word3c0=0,word3f8=0;
      uint64_t c_view=0;
      uint16_t code_mid=0,code_hi=0;
      memcpy(&packet,mapped[23]+0x40,4);
      memcpy(&code_mid,mapped[23]+0x46,2);memcpy(&code_hi,mapped[23]+0x48,2);
      memcpy(&word3c0,(void*)(uintptr_t)(shmem[1].cpu+0x3c0),4);
      memcpy(&c_view,(void*)(uintptr_t)(shmem[1].cpu+0x3ec),8);
      memcpy(&word3f8,(void*)(uintptr_t)(shmem[1].cpu+0x3f8),4);
      if(packet!=0x0e5806c7u || code_mid!=0 || code_hi!=0x100u ||
         word3c0!=0x01000000u || c_view!=0x10000036f00ull+(stale_c_page?0u:shift) ||
         word3f8!=0x22u)return 2;
      uint32_t* c_view_words=(uint32_t*)(mapped[2]+0x6f00+shift);
      uint32_t* ffn_output=ffn_c_cross?(uint32_t*)(mapped[17]+0x8000):
                           ffn_c_slot?(uint32_t*)(mapped[2]+ffn_c_slot):c_view_words+(c_binding_base?0:32);
      unsigned sentinels=0,guards=1;
      for(unsigned i=0;i<1024;i++)sentinels+=ffn_output[i]==(ffn_acc?0u:0x7fc01234u);
      if(ffn_c_slot || ffn_c_cross)for(unsigned i=0;i<1024;i++)guards&=c_view_words[32+i]==0x7fc01234u;
      if(ffn_c_cross || ffn_c_xcontrol)guards&=ffn_cross_guards(mapped[2],mapped[17],ffn_c_cross);
      if(c_competition){
        for(unsigned i=0;i<1056;i++)guards&=c_view_words[i]==0x7fc01234u;
      }
      for(unsigned i=0;i<0x80;i++){
        guards&=mapped[2][0x5d80+shift+i]==0xa5;
        guards&=a_competition?mapped[2][0x4d00+shift+i]==((i&1u)?0x3cu:0x00u):
                               mapped[2][0x4d00+shift+i]==0xa5;
        guards&=mapped[2][0x6e80+shift+i]==0xa4;
        guards&=b_competition?mapped[2][0x5e00+shift+i]==((i&1u)?0x3cu:0x00u):
                               mapped[2][0x5e00+shift+i]==0xa4;
        if(!c_competition)guards&=mapped[2][0x6f00+shift+i]==0xa7 && mapped[2][0x7f80+shift+i]==0xa7;
      }
      fprintf(stderr,"PURE FFN REGISTERED: code=%u grid=32 bindings=%llx/%llx/%llx initial=%u/1024 guards=%u AGXMetal=%u.\n",
              input_sizes[0],
              (unsigned long long)binding[0],(unsigned long long)binding[1],
              (unsigned long long)binding[2],sentinels,guards,!no_agx_metal());
      if(sentinels!=1024 || !guards || !no_agx_metal())return 2;
      if(ffn_c_slot)fprintf(stderr,"PURE FFN C SLOT PASS: offset=0x%x binding=0x%llx old C=1024/1024 sentinel; no Submit.\n",
                             ffn_c_slot,(unsigned long long)binding[2]);
      if(ffn_c_cross || ffn_c_xcontrol)fprintf(stderr,"PURE FFN C CROSS PASS: arm=%s binding=0x%llx rival=1024/1024 zero canaries=1; no Submit.\n",
                                               ffn_c_cross?"candidate":"control",(unsigned long long)binding[2]);
      if(!getenv("ORDERED_FFN_FIRE")){
        fprintf(stderr,"PURE FFN STAGE PASS: 29 registrations, exact authored image and A/B, command pages, three bindings and C initial words; no Submit.\n");
        return 0;
      }
      typedef int (*submit_t)(void*,void*,unsigned,void*,unsigned,void*);
      submit_t Submit=dlsym(io,"IOGPUCommandQueueSubmitCommandBuffers");if(!Submit)return 2;
      if(ffn_standalone_query){
        volatile uint32_t* full_ready=(volatile uint32_t*)(uintptr_t)(shmem[0].cpu+0x24);
        uint8_t* source=mapped[2]+0x8000u;
        uint8_t* weight=mapped[17]+0x1000u;
        uint8_t* output=mapped[17]+0x13000u;
        uint8_t left[64],right[64];
        memcpy(left,output-64,64);memcpy(right,output+49152u,64);
        if(*full_ready!=0xf0u || shmem[1].id!=2u || shmem[0].id!=1u ||
           !no_agx_metal())return 2;
        if(ffn_standalone_embedding_query){
          uint8_t* parameters=mapped[2]+0x4d80u;
          uint8_t ln_left[64],ln_right[64];
          memcpy(ln_left,source-64,64);memcpy(ln_right,source+49152u,64);
          memcpy(weight,embedding_raw_source,sizeof embedding_raw_source);
          memcpy(parameters,embedding_parameters,sizeof embedding_parameters);
          for(unsigned i=0;i<12288;i++)((uint32_t*)source)[i]=0x7fc01234u;
          uint32_t ln_scratch[2]={0x0c00100fu,0};
          uint32_t ln_packet=0x0e5806c7u;
          uint32_t ln_threads=32u;
          const uint64_t ln_bindings[4]={0x10000059000ull,0x10000034d80ull,
                                         0x10000038000ull,0};
          memcpy(mapped[0]+0x6c0,embedding_layernorm_code,sizeof embedding_layernorm_code);
          memcpy(mapped[23]+0x38,ln_scratch,8);
          memcpy(mapped[23]+0x40,&ln_packet,4);
          memcpy(mapped[22]+0xa8,&ln_threads,4);
          memcpy(mapped[22]+0xac,&ln_threads,4);
          memcpy(mapped[25]+0x10,&ln_threads,4);
          memcpy(mapped[25]+0x14,&ln_threads,4);
          memcpy(mapped[28]+0x1ba0,ln_bindings,sizeof ln_bindings);
          *full_ready=0x800000f0u;
          if(memcmp(weight,embedding_raw_source,sizeof embedding_raw_source) ||
             memcmp(parameters,embedding_parameters,sizeof embedding_parameters) ||
             memcmp(mapped[0]+0x6c0,embedding_layernorm_code,sizeof embedding_layernorm_code) ||
             memcmp(mapped[28]+0x1ba0,ln_bindings,sizeof ln_bindings) ||
             !no_agx_metal())return 2;
          static volatile uint64_t embedding_marks[2]={0};
          volatile uint64_t* em=embedding_marks;
          void (^ln_scheduled)(void)=Block_copy(^{em[0]=0x1a0000u;});
          void (^ln_completed)(void)=Block_copy(^{em[1]=0x1a0001u;});
          if(!ln_scheduled || !ln_completed || ln_scheduled==ln_completed)return 2;
          uint8_t ln_record[64]={0},ln_out[64]={0};
          uint32_t kid=shmem[1].id,sid=shmem[0].id;
          uintptr_t sp=(uintptr_t)ln_scheduled,cp=(uintptr_t)ln_completed;
          memcpy(ln_record,&kid,4);memcpy(ln_record+4,&sid,4);
          memcpy(ln_record+0x10,&sp,sizeof sp);
          memcpy(ln_record+0x18,&cp,sizeof cp);
          int ln_status=Submit(queue,NULL,1,ln_record,64,ln_out);
          unsigned ln_wait=0;for(;ln_wait<3000 && !em[1];ln_wait++)usleep(10000);
          unsigned ln_changed=0,ln_finite=1,ln_within=1;
          double ln_max_abs=0,ln_max_fraction=0;
          for(unsigned i=0;i<12288;i++){
            uint32_t word=((uint32_t*)source)[i];
            ln_changed+=word!=0x7fc01234u;
            float result=0;memcpy(&result,&word,4);
            ln_finite&=isfinite(result);
            double error=fabs((double)result-embedding_layernorm_reference[i]);
            double limit=2e-5*(1.0+fabs(embedding_layernorm_reference[i]));
            ln_within&=isfinite(result) && error<=limit;
            if(error>ln_max_abs)ln_max_abs=error;
            if(error/limit>ln_max_fraction)ln_max_fraction=error/limit;
          }
          unsigned ln_guards=memcmp(source-64,ln_left,64)==0 &&
                             memcmp(source+49152u,ln_right,64)==0;
          unsigned ln_inputs=memcmp(weight,embedding_raw_source,sizeof embedding_raw_source)==0 &&
                             memcmp(parameters,embedding_parameters,sizeof embedding_parameters)==0;
          uint32_t ln_outword=0;memcpy(&ln_outword,ln_out,4);
          fprintf(stderr,"PURE EMBEDDING LAYERNORM returned status=%d outword=0x%08x changed=%u/12288 finite=%u within=%u guards=%u inputs=%u max_abs=%.9g max_fraction=%.9g marks=%llu/%llu wait=%u/3000.\n",
                  ln_status,ln_outword,ln_changed,ln_finite,ln_within,ln_guards,ln_inputs,
                  ln_max_abs,ln_max_fraction,(unsigned long long)em[0],
                  (unsigned long long)em[1],ln_wait);
          fprintf(stderr,"PURE EMBEDDING LAYERNORM words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)source)[i]);
          fputc('\n',stderr);
          if(ln_status || ln_outword || ln_changed!=12288u || !ln_finite ||
             !ln_within || !ln_guards || !ln_inputs ||
             em[0]!=0x1a0000u || em[1]!=0x1a0001u || !no_agx_metal())return 2;
          memcpy(query_source_expected,source,49152u);
          fprintf(stderr,"PURE EMBEDDING LAYERNORM COMPLETE: model input normalized on GPU for query.\n");
        }else memcpy(source,query_source_expected,49152u);
        memcpy(weight,query_packed,49280u);
        for(unsigned i=0;i<12288;i++)((uint32_t*)output)[i]=0x7fc01234u;
        uint32_t scratch_word[2]={0x0c000007u,0};
        uint32_t scalar_packet=0x0e4006c7u;
        uint32_t xy_threads=32u;
        const uint64_t bindings[3]={0x10000038000ull,0x10000059000ull,
                                    0x1000006b000ull};
        memcpy(mapped[0]+0x6c0,query_code,sizeof query_code);
        memcpy(mapped[23]+0x38,scratch_word,8);
        memcpy(mapped[23]+0x40,&scalar_packet,4);
        memcpy(mapped[22]+0xa8,&xy_threads,4);
        memcpy(mapped[22]+0xac,&xy_threads,4);
        memcpy(mapped[25]+0x10,&xy_threads,4);
        memcpy(mapped[25]+0x14,&xy_threads,4);
        memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
        *full_ready=0x800000f0u;
        if(memcmp(source,query_source_expected,49152u) ||
           memcmp(weight,query_packed,49280u) ||
           memcmp(mapped[0]+0x6c0,query_code,sizeof query_code) ||
           memcmp(mapped[23]+0x38,scratch_word,8) ||
           memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
           !no_agx_metal())return 2;
        static volatile uint64_t standalone_marks[2]={0};
        volatile uint64_t* sm=standalone_marks;
        void (^scheduled)(void)=Block_copy(^{sm[0]=0x190000u;});
        void (^completed)(void)=Block_copy(^{sm[1]=0x190001u;});
        if(!scheduled || !completed || scheduled==completed)return 2;
        uint8_t record[64]={0},out[64]={0};
        uint32_t kid=shmem[1].id,sid=shmem[0].id;
        uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
        memcpy(record,&kid,4);memcpy(record+4,&sid,4);
        memcpy(record+0x10,&sp,sizeof sp);
        memcpy(record+0x18,&cp,sizeof cp);
        int status=Submit(queue,NULL,1,record,64,out);
        unsigned waited=0;for(;waited<3000 && !sm[1];waited++)usleep(10000);
        unsigned changed=0,finite=1,within=1,rest=1;
        double max_abs=0,max_fraction=0;
        for(unsigned row=0;row<32;row++)for(unsigned col=0;col<384;col++){
          uint32_t word=((uint32_t*)output)[row*384u+col];
          if(col>=32){rest&=word==0x7fc01234u;continue;}
          changed+=word!=0x7fc01234u;
          float result=0;memcpy(&result,&word,4);
          finite&=isfinite(result);
          double error=fabs((double)result-query_reference[row*32u+col]);
          double limit=2e-5*(1.0+fabs(query_reference[row*32u+col]));
          within&=isfinite(result) && error<=limit;
          if(error>max_abs)max_abs=error;
          if(error/limit>max_fraction)max_fraction=error/limit;
        }
        unsigned guards=memcmp(output-64,left,64)==0 &&
                        memcmp(output+49152u,right,64)==0;
        unsigned inputs=memcmp(source,query_source_expected,49152u)==0 &&
                        memcmp(weight,query_packed,49280u)==0;
        uint32_t outword=0;memcpy(&outword,out,4);
        fprintf(stderr,"PURE STANDALONE QUERY returned status=%d outword=0x%08x changed=%u/1024 finite=%u within=%u rest=%u guards=%u inputs=%u max_abs=%.9g max_fraction=%.9g marks=%llu/%llu wait=%u/3000.\n",
                status,outword,changed,finite,within,rest,guards,inputs,
                max_abs,max_fraction,(unsigned long long)sm[0],
                (unsigned long long)sm[1],waited);
        fprintf(stderr,"PURE STANDALONE QUERY words:");
        for(unsigned row=0;row<32;row++)for(unsigned col=0;col<32;col++)
          fprintf(stderr," %08x",((uint32_t*)output)[row*384u+col]);
        fputc('\n',stderr);
        if(status || outword || changed!=1024u || !finite || !within ||
           !rest || !guards || !inputs || sm[0]!=0x190000u ||
           sm[1]!=0x190001u || !no_agx_metal())return 2;
        if(ffn_standalone_query_all){
          static volatile uint64_t all_marks[24]={0};
          volatile uint64_t* am=all_marks;
          for(unsigned n=1;n<12;n++){
            memcpy(weight,query_packed+n*49280u,49280u);
            uint64_t c_va=0x1000006b000ull+(uint64_t)n*128u;
            memcpy(mapped[28]+0x1bb0,&c_va,8);
            if(memcmp(source,query_source_expected,49152u) ||
               memcmp(weight,query_packed+n*49280u,49280u) ||
               memcmp(mapped[28]+0x1bb0,&c_va,8) ||
               !no_agx_metal())return 2;
            void (^scheduled)(void)=Block_copy(^{am[2*n]=0x1b0000u+2u*n;});
            void (^completed)(void)=Block_copy(^{am[2*n+1]=0x1b0001u+2u*n;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t tile_record[64]={0},tile_out[64]={0};
            uintptr_t tile_sp=(uintptr_t)scheduled,tile_cp=(uintptr_t)completed;
            memcpy(tile_record,&kid,4);memcpy(tile_record+4,&sid,4);
            memcpy(tile_record+0x10,&tile_sp,sizeof tile_sp);
            memcpy(tile_record+0x18,&tile_cp,sizeof tile_cp);
            int tile_status=Submit(queue,NULL,1,tile_record,64,tile_out);
            unsigned tile_wait=0;
            for(;tile_wait<3000 && !am[2*n+1];tile_wait++)usleep(10000);
            unsigned tile_changed=0,tile_finite=1,tile_within=1,tile_rest=1;
            double tile_max_abs=0;
            for(unsigned row=0;row<32;row++)for(unsigned col=0;col<384;col++){
              uint32_t word=((uint32_t*)output)[row*384u+col];
              if(col>=(n+1u)*32u){tile_rest&=word==0x7fc01234u;continue;}
              unsigned tile=col/32u,within_tile=col%32u;
              tile_changed+=word!=0x7fc01234u;
              float value=0;memcpy(&value,&word,4);
              tile_finite&=isfinite(value);
              double error=fabs((double)value-query_reference[tile*1024u+row*32u+within_tile]);
              double limit=2e-5*(1.0+fabs(query_reference[tile*1024u+row*32u+within_tile]));
              tile_within&=isfinite(value) && error<=limit;
              if(error>tile_max_abs)tile_max_abs=error;
            }
            unsigned tile_guards=memcmp(output-64,left,64)==0 &&
                                 memcmp(output+49152u,right,64)==0;
            unsigned tile_inputs=memcmp(source,query_source_expected,49152u)==0 &&
                                 memcmp(weight,query_packed+n*49280u,49280u)==0;
            uint32_t tile_outword=0;memcpy(&tile_outword,tile_out,4);
            fprintf(stderr,"PURE STANDALONE QUERY ALL n=%u status=%d outword=0x%08x changed=%u/%u finite=%u within=%u rest=%u guards=%u inputs=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                    n,tile_status,tile_outword,tile_changed,(n+1u)*1024u,
                    tile_finite,tile_within,tile_rest,tile_guards,tile_inputs,
                    tile_max_abs,(unsigned long long)am[2*n],
                    (unsigned long long)am[2*n+1],tile_wait);
            if(tile_status || tile_outword || tile_changed!=(n+1u)*1024u ||
               !tile_finite || !tile_within || !tile_rest || !tile_guards ||
               !tile_inputs || am[2*n]!=0x1b0000u+2u*n ||
               am[2*n+1]!=0x1b0001u+2u*n || !no_agx_metal())return 2;
          }
          fprintf(stderr,"PURE STANDALONE QUERY ALL words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)output)[i]);
          fputc('\n',stderr);
          fprintf(stderr,"PURE STANDALONE QUERY ALL COMPLETE: embedding LayerNorm fed all twelve model query tiles.\n");
        }
        if(ffn_standalone_key_all){
          uint8_t* key_output=mapped[2]+0x14000u;
          uint8_t key_lower[64];memcpy(key_lower,key_output-64,64);
          uint8_t* query_snapshot=malloc(49152u);if(!query_snapshot)return 2;
          memcpy(query_snapshot,output,49152u);
          if(sizes[2]!=0x20000u || key_output+49152u!=mapped[2]+sizes[2] ||
             memcmp(source,query_source_expected,49152u) || !no_agx_metal())return 2;
          for(unsigned i=0;i<12288;i++)((uint32_t*)key_output)[i]=0x7fc01234u;
          static volatile uint64_t key_marks[24]={0};
          volatile uint64_t* km=key_marks;
          for(unsigned n=0;n<12;n++){
            memcpy(weight,key_packed+n*49280u,49280u);
            uint64_t key_bindings[3]={0x10000038000ull,0x10000059000ull,
                                      0x10000044000ull+(uint64_t)n*128u};
            memcpy(mapped[28]+0x1ba0,key_bindings,sizeof key_bindings);
            if(memcmp(mapped[0]+0x6c0,query_code,sizeof query_code) ||
               memcmp(mapped[28]+0x1ba0,key_bindings,sizeof key_bindings) ||
               memcmp(weight,key_packed+n*49280u,49280u) ||
               memcmp(source,query_source_expected,49152u) ||
               memcmp(output,query_snapshot,49152u) || !no_agx_metal())return 2;
            void (^scheduled)(void)=Block_copy(^{km[2*n]=0x1c0000u+2u*n;});
            void (^completed)(void)=Block_copy(^{km[2*n+1]=0x1c0001u+2u*n;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t key_record[64]={0},key_out[64]={0};
            uintptr_t key_sp=(uintptr_t)scheduled,key_cp=(uintptr_t)completed;
            memcpy(key_record,&kid,4);memcpy(key_record+4,&sid,4);
            memcpy(key_record+0x10,&key_sp,sizeof key_sp);
            memcpy(key_record+0x18,&key_cp,sizeof key_cp);
            int key_status=Submit(queue,NULL,1,key_record,64,key_out);
            unsigned key_wait=0;
            for(;key_wait<3000 && !km[2*n+1];key_wait++)usleep(10000);
            unsigned key_changed=0,key_finite=1,key_within=1,key_rest=1;
            double key_max_abs=0;
            for(unsigned row=0;row<32;row++)for(unsigned col=0;col<384;col++){
              uint32_t word=((uint32_t*)key_output)[row*384u+col];
              if(col>=(n+1u)*32u){key_rest&=word==0x7fc01234u;continue;}
              unsigned tile=col/32u,within_tile=col%32u;
              key_changed+=word!=0x7fc01234u;
              float value=0;memcpy(&value,&word,4);
              key_finite&=isfinite(value);
              double expected=key_reference[tile*1024u+row*32u+within_tile];
              double error=fabs((double)value-expected);
              double limit=2e-5*(1.0+fabs(expected));
              key_within&=isfinite(value) && error<=limit;
              if(error>key_max_abs)key_max_abs=error;
            }
            unsigned key_lower_ok=memcmp(key_output-64,key_lower,64)==0;
            unsigned key_inputs=memcmp(source,query_source_expected,49152u)==0 &&
                                memcmp(weight,key_packed+n*49280u,49280u)==0;
            unsigned query_ok=memcmp(output,query_snapshot,49152u)==0;
            uint32_t key_outword=0;memcpy(&key_outword,key_out,4);
            fprintf(stderr,"PURE STANDALONE KEY ALL n=%u status=%d outword=0x%08x changed=%u/%u finite=%u within=%u rest=%u lower=%u inputs=%u query=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                    n,key_status,key_outword,key_changed,(n+1u)*1024u,
                    key_finite,key_within,key_rest,key_lower_ok,key_inputs,query_ok,
                    key_max_abs,(unsigned long long)km[2*n],
                    (unsigned long long)km[2*n+1],key_wait);
            if(key_status || key_outword || key_changed!=(n+1u)*1024u ||
               !key_finite || !key_within || !key_rest || !key_lower_ok ||
               !key_inputs || !query_ok || km[2*n]!=0x1c0000u+2u*n ||
               km[2*n+1]!=0x1c0001u+2u*n || !no_agx_metal())return 2;
          }
          fprintf(stderr,"PURE STANDALONE KEY ALL words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)key_output)[i]);
          fputc('\n',stderr);
          fprintf(stderr,"PURE STANDALONE KEY ALL COMPLETE: GPU-produced embedding LayerNorm fed model query and key.\n");
          free(query_snapshot);
        }
        if(ffn_standalone_scores){
          uint8_t* key_output=mapped[2]+0x14000u;
          uint8_t* score_output=mapped[17]+0x1000u;
          uint8_t score_left[64],score_right[64];
          memcpy(score_left,score_output-64,64);
          memcpy(score_right,score_output+49152u,64);
          if(memcmp(output,scores_query_expected,49152u) ||
             memcmp(key_output,scores_key_expected,49152u) ||
             memcmp(source,query_source_expected,49152u) || !no_agx_metal())return 2;
          for(unsigned i=0;i<12288;i++)((uint32_t*)score_output)[i]=0x7fc01234u;
          uint32_t score_scratch[2]={0x0c000007u,0};
          uint32_t score_packet=0x0e4006c7u;
          uint32_t score_x=32u,score_y=384u;
          const uint64_t score_bindings[3]={0x1000006b000ull,0x10000044000ull,
                                            0x10000059000ull};
          memcpy(mapped[0]+0x6c0,scores_code,sizeof scores_code);
          memcpy(mapped[23]+0x38,score_scratch,8);
          memcpy(mapped[23]+0x40,&score_packet,4);
          memcpy(mapped[22]+0xa8,&score_x,4);
          memcpy(mapped[22]+0xac,&score_y,4);
          memcpy(mapped[25]+0x10,&score_x,4);
          memcpy(mapped[25]+0x14,&score_y,4);
          memcpy(mapped[28]+0x1ba0,score_bindings,sizeof score_bindings);
          if(memcmp(mapped[0]+0x6c0,scores_code,sizeof scores_code) ||
             memcmp(mapped[23]+0x38,score_scratch,8) ||
             memcmp(mapped[28]+0x1ba0,score_bindings,sizeof score_bindings) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t score_marks[2]={0};
          volatile uint64_t* cm=score_marks;
          void (^scheduled)(void)=Block_copy(^{cm[0]=0x1d0000u;});
          void (^completed)(void)=Block_copy(^{cm[1]=0x1d0001u;});
          if(!scheduled || !completed || scheduled==completed)return 2;
          uint8_t score_record[64]={0},score_out[64]={0};
          uintptr_t score_sp=(uintptr_t)scheduled,score_cp=(uintptr_t)completed;
          memcpy(score_record,&kid,4);memcpy(score_record+4,&sid,4);
          memcpy(score_record+0x10,&score_sp,sizeof score_sp);
          memcpy(score_record+0x18,&score_cp,sizeof score_cp);
          int score_status=Submit(queue,NULL,1,score_record,64,score_out);
          unsigned score_wait=0;
          for(;score_wait<3000 && !cm[1];score_wait++)usleep(10000);
          unsigned score_changed=0,score_finite=1,score_within=1;
          double score_max_abs=0;
          for(unsigned i=0;i<12288;i++){
            uint32_t word=((uint32_t*)score_output)[i];
            score_changed+=word!=0x7fc01234u;
            float value=0;memcpy(&value,&word,4);
            score_finite&=isfinite(value);
            double error=fabs((double)value-scores_reference[i]);
            double limit=2e-5*(1.0+fabs(scores_reference[i]));
            score_within&=isfinite(value) && error<=limit;
            if(error>score_max_abs)score_max_abs=error;
          }
          unsigned score_guards=memcmp(score_output-64,score_left,64)==0 &&
                                memcmp(score_output+49152u,score_right,64)==0;
          unsigned score_inputs=memcmp(output,scores_query_expected,49152u)==0 &&
                                memcmp(key_output,scores_key_expected,49152u)==0 &&
                                memcmp(source,query_source_expected,49152u)==0;
          uint32_t score_outword=0;memcpy(&score_outword,score_out,4);
          fprintf(stderr,"PURE STANDALONE SCORES status=%d outword=0x%08x changed=%u/12288 finite=%u within=%u guards=%u inputs=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                  score_status,score_outword,score_changed,score_finite,
                  score_within,score_guards,score_inputs,score_max_abs,
                  (unsigned long long)cm[0],(unsigned long long)cm[1],score_wait);
          fprintf(stderr,"PURE STANDALONE SCORES words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)score_output)[i]);
          fputc('\n',stderr);
          if(score_status || score_outword || score_changed!=12288u ||
             !score_finite || !score_within || !score_guards || !score_inputs ||
             cm[0]!=0x1d0000u || cm[1]!=0x1d0001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE STANDALONE SCORES COMPLETE: resident ordered Q/K produced attention logits.\n");
        }
        if(ffn_standalone_softmax){
          uint8_t* score_input=mapped[17]+0x1000u;
          uint8_t* probability_output=mapped[17]+0x13000u;
          uint8_t probability_left[64],probability_right[64];
          memcpy(probability_left,probability_output-64,64);
          memcpy(probability_right,probability_output+49152u,64);
          if(memcmp(score_input,softmax_scores_expected,49152u) ||
             memcmp(source,query_source_expected,49152u) ||
             memcmp(mapped[2]+0x14000u,scores_key_expected,49152u) ||
             !no_agx_metal())return 2;
          for(unsigned i=0;i<12288;i++)((uint32_t*)probability_output)[i]=0x7fc01234u;
          uint32_t softmax_scratch[2]={0x0c000007u,0};
          uint32_t softmax_packet=0x0e4006c7u;
          uint32_t softmax_x=384u,softmax_y=1u;
          const uint64_t softmax_bindings[3]={0x10000059000ull,0x1000006b000ull,0};
          memcpy(mapped[0]+0x6c0,softmax_code,sizeof softmax_code);
          memcpy(mapped[23]+0x38,softmax_scratch,8);
          memcpy(mapped[23]+0x40,&softmax_packet,4);
          memcpy(mapped[22]+0xa8,&softmax_x,4);
          memcpy(mapped[22]+0xac,&softmax_y,4);
          memcpy(mapped[25]+0x10,&softmax_x,4);
          memcpy(mapped[25]+0x14,&softmax_y,4);
          memcpy(mapped[28]+0x1ba0,softmax_bindings,sizeof softmax_bindings);
          if(memcmp(mapped[0]+0x6c0,softmax_code,sizeof softmax_code) ||
             memcmp(mapped[28]+0x1ba0,softmax_bindings,sizeof softmax_bindings) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t softmax_marks[2]={0};
          volatile uint64_t* pm=softmax_marks;
          void (^scheduled)(void)=Block_copy(^{pm[0]=0x1e0000u;});
          void (^completed)(void)=Block_copy(^{pm[1]=0x1e0001u;});
          if(!scheduled || !completed || scheduled==completed)return 2;
          uint8_t probability_record[64]={0},probability_out[64]={0};
          uintptr_t probability_sp=(uintptr_t)scheduled,probability_cp=(uintptr_t)completed;
          memcpy(probability_record,&kid,4);memcpy(probability_record+4,&sid,4);
          memcpy(probability_record+0x10,&probability_sp,sizeof probability_sp);
          memcpy(probability_record+0x18,&probability_cp,sizeof probability_cp);
          int probability_status=Submit(queue,NULL,1,probability_record,64,probability_out);
          unsigned probability_wait=0;
          for(;probability_wait<3000 && !pm[1];probability_wait++)usleep(10000);
          unsigned probability_changed=0,probability_finite=1,probability_within=1;
          unsigned probability_range=1,probability_rows=1;
          double probability_max_abs=0,probability_max_row_error=0;
          for(unsigned row=0;row<384;row++){
            double sum=0;
            for(unsigned col=0;col<32;col++){
              unsigned i=row*32u+col;
              uint32_t word=((uint32_t*)probability_output)[i];
              probability_changed+=word!=0x7fc01234u;
              float value=0;memcpy(&value,&word,4);
              probability_finite&=isfinite(value);
              probability_range&=value>=0.0f && value<=1.0f;
              sum+=(double)value;
              double error=fabs((double)value-softmax_reference[i]);
              double limit=2e-5*(1.0+fabs(softmax_reference[i]));
              probability_within&=isfinite(value) && error<=limit;
              if(error>probability_max_abs)probability_max_abs=error;
            }
            double row_error=fabs(sum-1.0);
            probability_rows&=row_error<=1e-4;
            if(row_error>probability_max_row_error)probability_max_row_error=row_error;
          }
          unsigned probability_guards=memcmp(probability_output-64,probability_left,64)==0 &&
                                      memcmp(probability_output+49152u,probability_right,64)==0;
          unsigned probability_inputs=memcmp(score_input,softmax_scores_expected,49152u)==0 &&
                                      memcmp(source,query_source_expected,49152u)==0 &&
                                      memcmp(mapped[2]+0x14000u,scores_key_expected,49152u)==0;
          uint32_t probability_outword=0;memcpy(&probability_outword,probability_out,4);
          fprintf(stderr,"PURE STANDALONE SOFTMAX status=%d outword=0x%08x changed=%u/12288 finite=%u within=%u range=%u rows=%u guards=%u inputs=%u max_abs=%.9g max_row_error=%.9g marks=%llu/%llu wait=%u/3000.\n",
                  probability_status,probability_outword,probability_changed,
                  probability_finite,probability_within,probability_range,
                  probability_rows,probability_guards,probability_inputs,
                  probability_max_abs,probability_max_row_error,
                  (unsigned long long)pm[0],(unsigned long long)pm[1],probability_wait);
          fprintf(stderr,"PURE STANDALONE SOFTMAX words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)probability_output)[i]);
          fputc('\n',stderr);
          if(probability_status || probability_outword || probability_changed!=12288u ||
             !probability_finite || !probability_within || !probability_range ||
             !probability_rows || !probability_guards || !probability_inputs ||
             pm[0]!=0x1e0000u || pm[1]!=0x1e0001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE STANDALONE SOFTMAX COMPLETE: resident ordered scores normalized into 384 probability rows.\n");
        }
        if(ffn_standalone_value){
          uint8_t* value_output=mapped[2]+0x14000u;
          uint8_t* value_weight=mapped[17]+0x1000u;
          uint8_t* probabilities=mapped[17]+0x13000u;
          uint8_t value_lower[64];memcpy(value_lower,value_output-64,64);
          if(memcmp(source,query_source_expected,49152u) ||
             memcmp(probabilities,value_probabilities_expected,49152u) ||
             sizes[2]!=0x20000u || value_output+49152u!=mapped[2]+sizes[2] ||
             value_weight+49280u>probabilities-64 || !no_agx_metal())return 2;
          for(unsigned i=0;i<12288;i++)((uint32_t*)value_output)[i]=0x7fc01234u;
          static volatile uint64_t value_marks[24]={0};
          volatile uint64_t* vm=value_marks;
          for(unsigned n=0;n<12;n++){
            memcpy(value_weight,value_packed+n*49280u,49280u);
            uint64_t value_bindings[3]={0x10000038000ull,0x10000059000ull,
                                        0x10000044000ull+(uint64_t)n*128u};
            uint32_t value_scratch[2]={0x0c000007u,0};
            uint32_t value_packet=0x0e4006c7u;
            uint32_t value_threads=32u;
            memcpy(mapped[0]+0x6c0,query_code,sizeof query_code);
            memcpy(mapped[23]+0x38,value_scratch,8);
            memcpy(mapped[23]+0x40,&value_packet,4);
            memcpy(mapped[22]+0xa8,&value_threads,4);
            memcpy(mapped[22]+0xac,&value_threads,4);
            memcpy(mapped[25]+0x10,&value_threads,4);
            memcpy(mapped[25]+0x14,&value_threads,4);
            memcpy(mapped[28]+0x1ba0,value_bindings,sizeof value_bindings);
            if(memcmp(mapped[0]+0x6c0,query_code,sizeof query_code) ||
               memcmp(mapped[28]+0x1ba0,value_bindings,sizeof value_bindings) ||
               memcmp(source,query_source_expected,49152u) ||
               memcmp(probabilities,value_probabilities_expected,49152u) ||
               memcmp(value_weight,value_packed+n*49280u,49280u) ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            void (^scheduled)(void)=Block_copy(^{vm[2*n]=0x1f0000u+2u*n;});
            void (^completed)(void)=Block_copy(^{vm[2*n+1]=0x1f0001u+2u*n;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t value_record[64]={0},value_out[64]={0};
            uintptr_t value_sp=(uintptr_t)scheduled,value_cp=(uintptr_t)completed;
            memcpy(value_record,&kid,4);memcpy(value_record+4,&sid,4);
            memcpy(value_record+0x10,&value_sp,sizeof value_sp);
            memcpy(value_record+0x18,&value_cp,sizeof value_cp);
            int value_status=Submit(queue,NULL,1,value_record,64,value_out);
            unsigned value_wait=0;
            for(;value_wait<3000 && !vm[2*n+1];value_wait++)usleep(10000);
            unsigned value_changed=0,value_finite=1,value_within=1,value_rest=1;
            double value_max_abs=0;
            for(unsigned row=0;row<32;row++)for(unsigned col=0;col<384;col++){
              uint32_t word=((uint32_t*)value_output)[row*384u+col];
              if(col>=(n+1u)*32u){value_rest&=word==0x7fc01234u;continue;}
              unsigned tile=col/32u,within_tile=col%32u;
              value_changed+=word!=0x7fc01234u;
              float result=0;memcpy(&result,&word,4);
              value_finite&=isfinite(result);
              double expected=value_reference[tile*1024u+row*32u+within_tile];
              double error=fabs((double)result-expected);
              double limit=2e-5*(1.0+fabs(expected));
              value_within&=isfinite(result) && error<=limit;
              if(error>value_max_abs)value_max_abs=error;
            }
            unsigned value_lower_ok=memcmp(value_output-64,value_lower,64)==0;
            unsigned value_inputs=memcmp(source,query_source_expected,49152u)==0 &&
                                  memcmp(probabilities,value_probabilities_expected,49152u)==0 &&
                                  memcmp(value_weight,value_packed+n*49280u,49280u)==0;
            uint32_t value_outword=0;memcpy(&value_outword,value_out,4);
            fprintf(stderr,"PURE STANDALONE VALUE n=%u status=%d outword=0x%08x changed=%u/%u finite=%u within=%u rest=%u lower=%u inputs=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                    n,value_status,value_outword,value_changed,(n+1u)*1024u,
                    value_finite,value_within,value_rest,value_lower_ok,value_inputs,
                    value_max_abs,(unsigned long long)vm[2*n],
                    (unsigned long long)vm[2*n+1],value_wait);
            if(value_status || value_outword || value_changed!=(n+1u)*1024u ||
               !value_finite || !value_within || !value_rest || !value_lower_ok ||
               !value_inputs || vm[2*n]!=0x1f0000u+2u*n ||
               vm[2*n+1]!=0x1f0001u+2u*n || !no_agx_metal())return 2;
          }
          fprintf(stderr,"PURE STANDALONE VALUE words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)value_output)[i]);
          fputc('\n',stderr);
          fprintf(stderr,"PURE STANDALONE VALUE COMPLETE: model value projection retained resident probabilities and LayerNorm source.\n");
        }
        if(ffn_standalone_context){
          uint8_t* probabilities=mapped[17]+0x13000u;
          uint8_t* value=mapped[2]+0x14000u;
          uint8_t* context_output=mapped[17]+0x1000u;
          uint8_t context_left[64],context_right[64];
          memcpy(context_left,context_output-64,64);
          memcpy(context_right,context_output+49152u,64);
          if(memcmp(probabilities,context_probabilities_expected,49152u) ||
             memcmp(value,context_value_expected,49152u) ||
             memcmp(source,query_source_expected,49152u) || !no_agx_metal())return 2;
          for(unsigned i=0;i<12288;i++)((uint32_t*)context_output)[i]=0x7fc01234u;
          uint32_t context_scratch[2]={0x0c000007u,0};
          uint32_t context_packet=0x0e4006c7u;
          uint32_t context_x=384u,context_y=32u;
          const uint64_t context_bindings[3]={0x1000006b000ull,0x10000044000ull,
                                              0x10000059000ull};
          memcpy(mapped[0]+0x6c0,context_code,sizeof context_code);
          memcpy(mapped[23]+0x38,context_scratch,8);
          memcpy(mapped[23]+0x40,&context_packet,4);
          memcpy(mapped[22]+0xa8,&context_x,4);
          memcpy(mapped[22]+0xac,&context_y,4);
          memcpy(mapped[25]+0x10,&context_x,4);
          memcpy(mapped[25]+0x14,&context_y,4);
          memcpy(mapped[28]+0x1ba0,context_bindings,sizeof context_bindings);
          if(memcmp(mapped[0]+0x6c0,context_code,sizeof context_code) ||
             memcmp(mapped[28]+0x1ba0,context_bindings,sizeof context_bindings) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t context_marks[2]={0};
          volatile uint64_t* tm=context_marks;
          void (^scheduled)(void)=Block_copy(^{tm[0]=0x200000u;});
          void (^completed)(void)=Block_copy(^{tm[1]=0x200001u;});
          if(!scheduled || !completed || scheduled==completed)return 2;
          uint8_t context_record[64]={0},context_out[64]={0};
          uintptr_t context_sp=(uintptr_t)scheduled,context_cp=(uintptr_t)completed;
          memcpy(context_record,&kid,4);memcpy(context_record+4,&sid,4);
          memcpy(context_record+0x10,&context_sp,sizeof context_sp);
          memcpy(context_record+0x18,&context_cp,sizeof context_cp);
          int context_status=Submit(queue,NULL,1,context_record,64,context_out);
          unsigned context_wait=0;
          for(;context_wait<3000 && !tm[1];context_wait++)usleep(10000);
          unsigned context_changed=0,context_finite=1,context_within=1;
          double context_max_abs=0;
          for(unsigned i=0;i<12288;i++){
            uint32_t word=((uint32_t*)context_output)[i];
            context_changed+=word!=0x7fc01234u;
            float result=0;memcpy(&result,&word,4);
            context_finite&=isfinite(result);
            double error=fabs((double)result-context_reference[i]);
            double limit=2e-5*(1.0+fabs(context_reference[i]));
            context_within&=isfinite(result) && error<=limit;
            if(error>context_max_abs)context_max_abs=error;
          }
          unsigned context_guards=memcmp(context_output-64,context_left,64)==0 &&
                                  memcmp(context_output+49152u,context_right,64)==0;
          unsigned context_inputs=memcmp(probabilities,context_probabilities_expected,49152u)==0 &&
                                  memcmp(value,context_value_expected,49152u)==0 &&
                                  memcmp(source,query_source_expected,49152u)==0;
          uint32_t context_outword=0;memcpy(&context_outword,context_out,4);
          fprintf(stderr,"PURE STANDALONE CONTEXT status=%d outword=0x%08x changed=%u/12288 finite=%u within=%u guards=%u inputs=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                  context_status,context_outword,context_changed,context_finite,
                  context_within,context_guards,context_inputs,context_max_abs,
                  (unsigned long long)tm[0],(unsigned long long)tm[1],context_wait);
          fprintf(stderr,"PURE STANDALONE CONTEXT words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)context_output)[i]);
          fputc('\n',stderr);
          if(context_status || context_outword || context_changed!=12288u ||
             !context_finite || !context_within || !context_guards || !context_inputs ||
             tm[0]!=0x200000u || tm[1]!=0x200001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE STANDALONE CONTEXT COMPLETE: resident ordered probabilities and V produced 32x384 context.\n");
        }
        if(ffn_standalone_output){
          uint8_t* projection_source=mapped[17]+0x1000u;
          uint8_t* projection_weight=mapped[17]+0x13000u;
          uint8_t* projection_output=mapped[2]+0x14000u;
          uint8_t projection_lower[64];memcpy(projection_lower,projection_output-64,64);
          if(memcmp(projection_source,attention_context_expected,49152u) ||
             memcmp(source,query_source_expected,49152u) ||
             projection_weight+49280u>mapped[17]+sizes[17] ||
             projection_output+49152u!=mapped[2]+sizes[2] ||
             !no_agx_metal())return 2;
          for(unsigned i=0;i<12288;i++)((uint32_t*)projection_output)[i]=0x7fc01234u;
          static volatile uint64_t projection_marks[24]={0};
          volatile uint64_t* om=projection_marks;
          for(unsigned n=0;n<12;n++){
            memcpy(projection_weight,attention_output_packed+n*49280u,49280u);
            uint64_t projection_bindings[3]={0x10000059000ull,0x1000006b000ull,
                                             0x10000044000ull+(uint64_t)n*128u};
            uint32_t projection_scratch[2]={0x0c000007u,0};
            uint32_t projection_packet=0x0e4006c7u;
            uint32_t projection_threads=32u;
            memcpy(mapped[0]+0x6c0,query_code,sizeof query_code);
            memcpy(mapped[23]+0x38,projection_scratch,8);
            memcpy(mapped[23]+0x40,&projection_packet,4);
            memcpy(mapped[22]+0xa8,&projection_threads,4);
            memcpy(mapped[22]+0xac,&projection_threads,4);
            memcpy(mapped[25]+0x10,&projection_threads,4);
            memcpy(mapped[25]+0x14,&projection_threads,4);
            memcpy(mapped[28]+0x1ba0,projection_bindings,sizeof projection_bindings);
            if(memcmp(mapped[0]+0x6c0,query_code,sizeof query_code) ||
               memcmp(mapped[28]+0x1ba0,projection_bindings,sizeof projection_bindings) ||
               memcmp(projection_source,attention_context_expected,49152u) ||
               memcmp(source,query_source_expected,49152u) ||
               memcmp(projection_weight,attention_output_packed+n*49280u,49280u) ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            void (^scheduled)(void)=Block_copy(^{om[2*n]=0x210000u+2u*n;});
            void (^completed)(void)=Block_copy(^{om[2*n+1]=0x210001u+2u*n;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t projection_record[64]={0},projection_out[64]={0};
            uintptr_t projection_sp=(uintptr_t)scheduled,projection_cp=(uintptr_t)completed;
            memcpy(projection_record,&kid,4);memcpy(projection_record+4,&sid,4);
            memcpy(projection_record+0x10,&projection_sp,sizeof projection_sp);
            memcpy(projection_record+0x18,&projection_cp,sizeof projection_cp);
            int projection_status=Submit(queue,NULL,1,projection_record,64,projection_out);
            unsigned projection_wait=0;
            for(;projection_wait<3000 && !om[2*n+1];projection_wait++)usleep(10000);
            unsigned projection_changed=0,projection_finite=1,projection_within=1,projection_rest=1;
            double projection_max_abs=0;
            for(unsigned row=0;row<32;row++)for(unsigned col=0;col<384;col++){
              uint32_t word=((uint32_t*)projection_output)[row*384u+col];
              if(col>=(n+1u)*32u){projection_rest&=word==0x7fc01234u;continue;}
              unsigned tile=col/32u,within_tile=col%32u;
              projection_changed+=word!=0x7fc01234u;
              float result=0;memcpy(&result,&word,4);
              projection_finite&=isfinite(result);
              double expected=attention_output_reference[tile*1024u+row*32u+within_tile];
              double error=fabs((double)result-expected);
              double limit=2e-5*(1.0+fabs(expected));
              projection_within&=isfinite(result) && error<=limit;
              if(error>projection_max_abs)projection_max_abs=error;
            }
            unsigned projection_lower_ok=memcmp(projection_output-64,projection_lower,64)==0;
            unsigned projection_inputs=memcmp(projection_source,attention_context_expected,49152u)==0 &&
                                       memcmp(source,query_source_expected,49152u)==0 &&
                                       memcmp(projection_weight,attention_output_packed+n*49280u,49280u)==0;
            uint32_t projection_outword=0;memcpy(&projection_outword,projection_out,4);
            fprintf(stderr,"PURE STANDALONE OUTPUT n=%u status=%d outword=0x%08x changed=%u/%u finite=%u within=%u rest=%u lower=%u inputs=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                    n,projection_status,projection_outword,projection_changed,(n+1u)*1024u,
                    projection_finite,projection_within,projection_rest,projection_lower_ok,
                    projection_inputs,projection_max_abs,(unsigned long long)om[2*n],
                    (unsigned long long)om[2*n+1],projection_wait);
            if(projection_status || projection_outword || projection_changed!=(n+1u)*1024u ||
               !projection_finite || !projection_within || !projection_rest ||
               !projection_lower_ok || !projection_inputs ||
               om[2*n]!=0x210000u+2u*n || om[2*n+1]!=0x210001u+2u*n ||
               !no_agx_metal())return 2;
          }
          fprintf(stderr,"PURE STANDALONE OUTPUT words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)projection_output)[i]);
          fputc('\n',stderr);
          fprintf(stderr,"PURE STANDALONE OUTPUT COMPLETE: model attention output projection retained context and LayerNorm source.\n");
        }
        if(ffn_standalone_residual){
          uint8_t* projection_output=mapped[2]+0x14000u;
          uint8_t* context=mapped[17]+0x1000u;
          uint8_t residual_lower[64];memcpy(residual_lower,projection_output-64,64);
          if(memcmp(source,query_source_expected,49152u) ||
             memcmp(projection_output,attention_projected_expected,49152u) ||
             memcmp(context,attention_context_expected,49152u) || !no_agx_metal())return 2;
          uint32_t residual_scratch[2]={0x0c000007u,0};
          uint32_t residual_packet=0x0e4006c7u;
          uint32_t residual_x=384u,residual_y=32u;
          const uint64_t residual_bindings[3]={0x10000038000ull,0x10000044000ull,
                                               0x10000044000ull};
          memcpy(mapped[0]+0x6c0,attention_residual_code,sizeof attention_residual_code);
          memcpy(mapped[23]+0x38,residual_scratch,8);
          memcpy(mapped[23]+0x40,&residual_packet,4);
          memcpy(mapped[22]+0xa8,&residual_x,4);
          memcpy(mapped[22]+0xac,&residual_y,4);
          memcpy(mapped[25]+0x10,&residual_x,4);
          memcpy(mapped[25]+0x14,&residual_y,4);
          memcpy(mapped[28]+0x1ba0,residual_bindings,sizeof residual_bindings);
          if(memcmp(mapped[0]+0x6c0,attention_residual_code,sizeof attention_residual_code) ||
             memcmp(mapped[28]+0x1ba0,residual_bindings,sizeof residual_bindings) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t residual_marks[2]={0};
          volatile uint64_t* rm=residual_marks;
          void (^scheduled)(void)=Block_copy(^{rm[0]=0x220000u;});
          void (^completed)(void)=Block_copy(^{rm[1]=0x220001u;});
          if(!scheduled || !completed || scheduled==completed)return 2;
          uint8_t residual_record[64]={0},residual_out[64]={0};
          uintptr_t residual_sp=(uintptr_t)scheduled,residual_cp=(uintptr_t)completed;
          memcpy(residual_record,&kid,4);memcpy(residual_record+4,&sid,4);
          memcpy(residual_record+0x10,&residual_sp,sizeof residual_sp);
          memcpy(residual_record+0x18,&residual_cp,sizeof residual_cp);
          int residual_status=Submit(queue,NULL,1,residual_record,64,residual_out);
          unsigned residual_wait=0;
          for(;residual_wait<3000 && !rm[1];residual_wait++)usleep(10000);
          unsigned residual_changed=0,residual_finite=1;
          for(unsigned i=0;i<12288;i++){
            uint32_t actual=((uint32_t*)projection_output)[i];
            uint32_t prior=((uint32_t*)attention_projected_expected)[i];
            residual_changed+=actual!=prior;
            float value=0;memcpy(&value,&actual,4);
            residual_finite&=isfinite(value);
          }
          unsigned residual_exact=memcmp(projection_output,attention_residual_expected,49152u)==0;
          unsigned residual_lower_ok=memcmp(projection_output-64,residual_lower,64)==0;
          unsigned residual_inputs=memcmp(source,query_source_expected,49152u)==0 &&
                                   memcmp(context,attention_context_expected,49152u)==0;
          uint32_t residual_outword=0;memcpy(&residual_outword,residual_out,4);
          fprintf(stderr,"PURE STANDALONE RESIDUAL status=%d outword=0x%08x changed=%u/12288 finite=%u exact=%u lower=%u inputs=%u marks=%llu/%llu wait=%u/3000.\n",
                  residual_status,residual_outword,residual_changed,residual_finite,
                  residual_exact,residual_lower_ok,residual_inputs,
                  (unsigned long long)rm[0],(unsigned long long)rm[1],residual_wait);
          fprintf(stderr,"PURE STANDALONE RESIDUAL words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)projection_output)[i]);
          fputc('\n',stderr);
          if(residual_status || residual_outword || residual_changed!=12288u ||
             !residual_finite || !residual_exact || !residual_lower_ok ||
             !residual_inputs || rm[0]!=0x220000u || rm[1]!=0x220001u ||
             !no_agx_metal())return 2;
          fprintf(stderr,"PURE STANDALONE RESIDUAL COMPLETE: GPU embedding LayerNorm added to resident attention output.\n");
        }
        if(ffn_standalone_attention_ln){
          uint8_t* ln_source=mapped[2]+0x14000u;
          uint8_t* ln_parameters=mapped[2]+0x4d80u;
          uint8_t* ln_output=mapped[17]+0x1000u;
          uint8_t ln_left[64],ln_right[64];
          memcpy(ln_left,ln_output-64,64);
          memcpy(ln_right,ln_output+49152u,64);
          if(memcmp(ln_source,attention_residual_expected,49152u) ||
             !no_agx_metal())return 2;
          memcpy(ln_parameters,attention_layernorm_parameters,
                 sizeof attention_layernorm_parameters);
          for(unsigned i=0;i<12288;i++)((uint32_t*)ln_output)[i]=0x7fc01234u;
          uint32_t ln_scratch[2]={0x0c00100fu,0};
          uint32_t ln_packet=0x0e5806c7u,ln_xy=32u;
          const uint64_t ln_bindings[4]={0x10000044000ull,0x10000034d80ull,
                                         0x10000059000ull,0};
          memcpy(mapped[0]+0x6c0,attention_layernorm_code,
                 sizeof attention_layernorm_code);
          memcpy(mapped[23]+0x38,ln_scratch,8);
          memcpy(mapped[23]+0x40,&ln_packet,4);
          memcpy(mapped[22]+0xa8,&ln_xy,4);
          memcpy(mapped[22]+0xac,&ln_xy,4);
          memcpy(mapped[25]+0x10,&ln_xy,4);
          memcpy(mapped[25]+0x14,&ln_xy,4);
          memcpy(mapped[28]+0x1ba0,ln_bindings,sizeof ln_bindings);
          if(memcmp(mapped[0]+0x6c0,attention_layernorm_code,
                    sizeof attention_layernorm_code) ||
             memcmp(mapped[23]+0x38,ln_scratch,8) ||
             memcmp(mapped[23]+0x40,&ln_packet,4) ||
             memcmp(mapped[28]+0x1ba0,ln_bindings,sizeof ln_bindings) ||
             memcmp(ln_parameters,attention_layernorm_parameters,
                    sizeof attention_layernorm_parameters) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t ln_marks[2]={0};
          volatile uint64_t* lm=ln_marks;
          void (^scheduled)(void)=Block_copy(^{lm[0]=0x230000u;});
          void (^completed)(void)=Block_copy(^{lm[1]=0x230001u;});
          if(!scheduled || !completed || scheduled==completed)return 2;
          uint8_t ln_record[64]={0},ln_out[64]={0};
          uintptr_t ln_sp=(uintptr_t)scheduled,ln_cp=(uintptr_t)completed;
          memcpy(ln_record,&kid,4);memcpy(ln_record+4,&sid,4);
          memcpy(ln_record+0x10,&ln_sp,sizeof ln_sp);
          memcpy(ln_record+0x18,&ln_cp,sizeof ln_cp);
          int ln_status=Submit(queue,NULL,1,ln_record,64,ln_out);
          unsigned ln_wait=0;
          for(;ln_wait<3000 && !lm[1];ln_wait++)usleep(10000);
          unsigned ln_changed=0,ln_finite=1,ln_within=1;
          double ln_max_abs=0;
          for(unsigned i=0;i<12288;i++){
            uint32_t word=((uint32_t*)ln_output)[i];
            ln_changed+=word!=0x7fc01234u;
            float value=0;memcpy(&value,&word,4);
            ln_finite&=isfinite(value);
            double error=fabs((double)value-attention_layernorm_reference[i]);
            double limit=2e-5*(1.0+fabs(attention_layernorm_reference[i]));
            ln_within&=isfinite(value) && error<=limit;
            if(error>ln_max_abs)ln_max_abs=error;
          }
          unsigned ln_guards=memcmp(ln_output-64,ln_left,64)==0 &&
                             memcmp(ln_output+49152u,ln_right,64)==0;
          unsigned ln_inputs=memcmp(ln_source,attention_residual_expected,49152u)==0 &&
                             memcmp(ln_parameters,attention_layernorm_parameters,
                                    sizeof attention_layernorm_parameters)==0;
          uint32_t ln_outword=0;memcpy(&ln_outword,ln_out,4);
          fprintf(stderr,"PURE STANDALONE ATTENTION LN status=%d outword=0x%08x changed=%u/12288 finite=%u within=%u guards=%u inputs=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                  ln_status,ln_outword,ln_changed,ln_finite,ln_within,ln_guards,
                  ln_inputs,ln_max_abs,(unsigned long long)lm[0],
                  (unsigned long long)lm[1],ln_wait);
          fprintf(stderr,"PURE STANDALONE ATTENTION LN words:");
          for(unsigned i=0;i<12288;i++)
            fprintf(stderr," %08x",((uint32_t*)ln_output)[i]);
          fputc('\n',stderr);
          if(ln_status || ln_outword || ln_changed!=12288u || !ln_finite ||
             !ln_within || !ln_guards || !ln_inputs ||
             lm[0]!=0x230000u || lm[1]!=0x230001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE STANDALONE ATTENTION LN COMPLETE: GPU residual normalized with model gamma/beta.\n");
        }
        if(ffn_standalone_ffn_pack){
          uint8_t* pack_source=mapped[17]+0x1000u;
          uint8_t* pack_target=mapped[2]+0x8000u;
          uint8_t pack_left[64],pack_right[64];
          memcpy(pack_left,pack_target-64,64);
          memcpy(pack_right,pack_target+24576u,64);
          if(memcmp(pack_source,standalone_ffn_pack_source,49152u) ||
             !no_agx_metal())return 2;
          memset(pack_target,0x7e,24576u);
          uint32_t pack_scratch[2]={0x0c000007u,0};
          uint32_t pack_packet=0x0e4006c7u;
          uint32_t pack_x=12288u,pack_y=1u;
          const uint64_t pack_bindings[3]={0x10000059000ull,0x10000038000ull,0};
          memcpy(mapped[0]+0x6c0,standalone_ffn_pack_code,sizeof standalone_ffn_pack_code);
          memcpy(mapped[23]+0x38,pack_scratch,8);
          memcpy(mapped[23]+0x40,&pack_packet,4);
          memcpy(mapped[22]+0xa8,&pack_x,4);
          memcpy(mapped[22]+0xac,&pack_y,4);
          memcpy(mapped[25]+0x10,&pack_x,4);
          memcpy(mapped[25]+0x14,&pack_y,4);
          memcpy(mapped[28]+0x1ba0,pack_bindings,sizeof pack_bindings);
          if(memcmp(mapped[0]+0x6c0,standalone_ffn_pack_code,sizeof standalone_ffn_pack_code) ||
             memcmp(mapped[23]+0x38,pack_scratch,8) ||
             memcmp(mapped[23]+0x40,&pack_packet,4) ||
             memcmp(mapped[28]+0x1ba0,pack_bindings,sizeof pack_bindings) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t pack_marks[2]={0};
          volatile uint64_t* pm=pack_marks;
          void (^scheduled)(void)=Block_copy(^{pm[0]=0x240000u;});
          void (^completed)(void)=Block_copy(^{pm[1]=0x240001u;});
          if(!scheduled || !completed || scheduled==completed)return 2;
          uint8_t pack_record[64]={0},pack_out[64]={0};
          uintptr_t pack_sp=(uintptr_t)scheduled,pack_cp=(uintptr_t)completed;
          memcpy(pack_record,&kid,4);memcpy(pack_record+4,&sid,4);
          memcpy(pack_record+0x10,&pack_sp,sizeof pack_sp);
          memcpy(pack_record+0x18,&pack_cp,sizeof pack_cp);
          int pack_status=Submit(queue,NULL,1,pack_record,64,pack_out);
          unsigned pack_wait=0;
          for(;pack_wait<3000 && !pm[1];pack_wait++)usleep(10000);
          unsigned pack_exact=memcmp(pack_target,standalone_ffn_pack_expected,24576u)==0;
          unsigned pack_source_ok=memcmp(pack_source,standalone_ffn_pack_source,49152u)==0;
          unsigned pack_guards=memcmp(pack_target-64,pack_left,64)==0 &&
                               memcmp(pack_target+24576u,pack_right,64)==0;
          unsigned pack_changed=0;
          for(unsigned i=0;i<12288;i++)pack_changed+=((uint16_t*)pack_target)[i]!=0x7e7eu;
          uint32_t pack_outword=0;memcpy(&pack_outword,pack_out,4);
          fprintf(stderr,"PURE STANDALONE FFN PACK status=%d outword=0x%08x changed=%u/12288 exact=%u source=%u guards=%u marks=%llu/%llu wait=%u/3000.\n",
                  pack_status,pack_outword,pack_changed,pack_exact,pack_source_ok,
                  pack_guards,(unsigned long long)pm[0],(unsigned long long)pm[1],pack_wait);
          fprintf(stderr,"PURE STANDALONE FFN PACK words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %04x",((uint16_t*)pack_target)[i]);
          fputc('\n',stderr);
          if(pack_status || pack_outword || pack_changed!=12288u || !pack_exact ||
             !pack_source_ok || !pack_guards || pm[0]!=0x240000u ||
             pm[1]!=0x240001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE STANDALONE FFN PACK COMPLETE: GPU attention LayerNorm output converted to six resident FP16 K64 blocks.\n");
        }
        if(ffn_standalone_ffn_tile){
          uint8_t* tile_a=mapped[2]+0x8000u;
          uint8_t* tile_b=mapped[2]+0x5e80u;
          uint8_t* tile_c=mapped[2]+0x6f80u;
          uint8_t tile_b_left[64],tile_b_right[64],tile_c_left[64],tile_c_right[64];
          memcpy(tile_b_left,tile_b-64,64);memcpy(tile_b_right,tile_b+4096u,64);
          memcpy(tile_c_left,tile_c-64,64);memcpy(tile_c_right,tile_c+4096u,64);
          if(memcmp(tile_a,standalone_ffn_pack_expected,24576u) ||
             memcmp(mapped[17]+0x1000u,standalone_ffn_pack_source,49152u) ||
             !no_agx_metal())return 2;
          memcpy(tile_b,standalone_ffn_tile_b,4096u);
          for(unsigned i=0;i<1024;i++)((uint32_t*)tile_c)[i]=0x7fc01234u;
          uint32_t tile_scratch[2]={0x0c00100fu,0};
          uint32_t tile_packet=0x0e5806c7u;
          uint32_t tile_x=32u,tile_y=1u;
          const uint64_t tile_bindings[4]={0x10000038000ull,0x10000035e80ull,
                                           0x10000036f80ull,0};
          memcpy(mapped[0]+0x6c0,standalone_ffn_tile_code,sizeof standalone_ffn_tile_code);
          memcpy(mapped[23]+0x38,tile_scratch,8);
          memcpy(mapped[23]+0x40,&tile_packet,4);
          memcpy(mapped[22]+0xa8,&tile_x,4);
          memcpy(mapped[22]+0xac,&tile_y,4);
          memcpy(mapped[25]+0x10,&tile_x,4);
          memcpy(mapped[25]+0x14,&tile_y,4);
          memcpy(mapped[28]+0x1ba0,tile_bindings,sizeof tile_bindings);
          if(memcmp(mapped[0]+0x6c0,standalone_ffn_tile_code,
                    sizeof standalone_ffn_tile_code) ||
             memcmp(mapped[23]+0x38,tile_scratch,8) ||
             memcmp(mapped[23]+0x40,&tile_packet,4) ||
             memcmp(mapped[28]+0x1ba0,tile_bindings,sizeof tile_bindings) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t tile_marks[2]={0};
          volatile uint64_t* tm=tile_marks;
          void (^scheduled)(void)=Block_copy(^{tm[0]=0x250000u;});
          void (^completed)(void)=Block_copy(^{tm[1]=0x250001u;});
          if(!scheduled || !completed || scheduled==completed)return 2;
          uint8_t tile_record[64]={0},tile_out[64]={0};
          uintptr_t tile_sp=(uintptr_t)scheduled,tile_cp=(uintptr_t)completed;
          memcpy(tile_record,&kid,4);memcpy(tile_record+4,&sid,4);
          memcpy(tile_record+0x10,&tile_sp,sizeof tile_sp);
          memcpy(tile_record+0x18,&tile_cp,sizeof tile_cp);
          int tile_status=Submit(queue,NULL,1,tile_record,64,tile_out);
          unsigned tile_wait=0;
          for(;tile_wait<3000 && !tm[1];tile_wait++)usleep(10000);
          unsigned tile_changed=0,tile_finite=1,tile_within=1;
          double tile_max_abs=0;
          for(unsigned i=0;i<1024;i++){
            uint32_t word=((uint32_t*)tile_c)[i];
            tile_changed+=word!=0x7fc01234u;
            float value=0;memcpy(&value,&word,4);
            tile_finite&=isfinite(value);
            double error=fabs((double)value-standalone_ffn_tile_reference[i]);
            double limit=2e-5*(1.0+fabs(standalone_ffn_tile_reference[i]));
            tile_within&=isfinite(value) && error<=limit;
            if(error>tile_max_abs)tile_max_abs=error;
          }
          unsigned tile_inputs=memcmp(tile_a,standalone_ffn_pack_expected,24576u)==0 &&
                               memcmp(tile_b,standalone_ffn_tile_b,4096u)==0 &&
                               memcmp(mapped[17]+0x1000u,standalone_ffn_pack_source,49152u)==0;
          unsigned tile_guards=memcmp(tile_b-64,tile_b_left,64)==0 &&
                               memcmp(tile_b+4096u,tile_b_right,64)==0 &&
                               memcmp(tile_c-64,tile_c_left,64)==0 &&
                               memcmp(tile_c+4096u,tile_c_right,64)==0;
          uint32_t tile_outword=0;memcpy(&tile_outword,tile_out,4);
          fprintf(stderr,"PURE STANDALONE FFN TILE status=%d outword=0x%08x changed=%u/1024 finite=%u within=%u inputs=%u guards=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                  tile_status,tile_outword,tile_changed,tile_finite,tile_within,
                  tile_inputs,tile_guards,tile_max_abs,
                  (unsigned long long)tm[0],(unsigned long long)tm[1],tile_wait);
          fprintf(stderr,"PURE STANDALONE FFN TILE words:");
          for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",((uint32_t*)tile_c)[i]);
          fputc('\n',stderr);
          if(tile_status || tile_outword || tile_changed!=1024u || !tile_finite ||
             !tile_within || !tile_inputs || !tile_guards ||
             tm[0]!=0x250000u || tm[1]!=0x250001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE STANDALONE FFN TILE COMPLETE: tensor program consumed GPU-packed attention LayerNorm and model FFN weights.\n");
        }
        if(ffn_standalone_ffn_k6){
          uint8_t* output=mapped[2]+0x6f80u;
          uint8_t* weights=mapped[2]+0x5e80u;
          uint8_t* packed=mapped[2]+0x8000u;
          uint8_t left[64],right[64];
          memcpy(left,output-64,64);memcpy(right,output+4096u,64);
          if(memcmp(output,standalone_ffn_k6_prior,4096u) ||
             memcmp(packed,standalone_ffn_pack_expected,24576u) ||
             !no_agx_metal())return 2;
          memcpy(mapped[0]+0x6c0,standalone_ffn_k6_accum_code,
                 sizeof standalone_ffn_k6_accum_code);
          if(memcmp(mapped[0]+0x6c0,standalone_ffn_k6_accum_code,
                    sizeof standalone_ffn_k6_accum_code) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t marks[10]={0};
          volatile uint64_t* km=marks;
          for(unsigned k=1;k<6;k++){
            uint8_t prior[4096];memcpy(prior,output,4096u);
            memcpy(weights,standalone_ffn_k6_b+(k-1u)*4096u,4096u);
            uint64_t a_va=0x10000038000ull+(uint64_t)k*4096u;
            memcpy(mapped[28]+0x1ba0,&a_va,8);
            if(memcmp(mapped[28]+0x1ba0,&a_va,8) ||
               memcmp(weights,standalone_ffn_k6_b+(k-1u)*4096u,4096u) ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            unsigned index=k-1u;
            void (^scheduled)(void)=Block_copy(^{km[2u*index]=0x260000u+2u*index;});
            void (^completed)(void)=Block_copy(^{km[2u*index+1u]=0x260001u+2u*index;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t record[64]={0},out[64]={0};
            uint32_t kid=shmem[1].id,sid=shmem[0].id;
            uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
            memcpy(record,&kid,4);memcpy(record+4,&sid,4);
            memcpy(record+0x10,&sp,sizeof sp);
            memcpy(record+0x18,&cp,sizeof cp);
            int status=Submit(queue,NULL,1,record,64,out);
            unsigned waited=0;
            for(;waited<3000 && !km[2u*index+1u];waited++)usleep(10000);
            unsigned changed=0,finite=1,within=1;
            double max_abs=0;
            for(unsigned i=0;i<1024;i++){
              uint32_t word=((uint32_t*)output)[i];
              changed+=word!=((uint32_t*)prior)[i];
              float value=0;memcpy(&value,&word,4);
              finite&=isfinite(value);
              double reference=standalone_ffn_k6_reference[index*1024u+i];
              double error=fabs((double)value-reference);
              double limit=2e-5*(1.0+fabs(reference));
              within&=isfinite(value) && error<=limit;
              if(error>max_abs)max_abs=error;
            }
            unsigned inputs=memcmp(packed,standalone_ffn_pack_expected,24576u)==0 &&
                            memcmp(weights,standalone_ffn_k6_b+index*4096u,4096u)==0 &&
                            memcmp(mapped[17]+0x1000u,standalone_ffn_pack_source,49152u)==0;
            unsigned guards=memcmp(output-64,left,64)==0 &&
                            memcmp(output+4096u,right,64)==0;
            uint32_t outword=0;memcpy(&outword,out,4);
            fprintf(stderr,"PURE STANDALONE FFN K6 k=%u status=%d outword=0x%08x changed=%u/1024 finite=%u within=%u inputs=%u guards=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                    k,status,outword,changed,finite,within,inputs,guards,max_abs,
                    (unsigned long long)km[2u*index],
                    (unsigned long long)km[2u*index+1u],waited);
            fprintf(stderr,"PURE STANDALONE FFN K6 k=%u words:",k);
            for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",((uint32_t*)output)[i]);
            fputc('\n',stderr);
            if(status || outword || changed!=1024u || !finite || !within ||
               !inputs || !guards || km[2u*index]!=0x260000u+2u*index ||
               km[2u*index+1u]!=0x260001u+2u*index || !no_agx_metal())return 2;
          }
          fprintf(stderr,"PURE STANDALONE FFN K6 COMPLETE: one model FFN output tile accumulated across six GPU-resident K64 blocks.\n");
        }
        if(ffn_standalone_n1_resident){
          unsigned known_c=getenv("ORDERED_FFN_N1_KNOWN_C")!=NULL;
          uint8_t* output=known_c?mapped[2]+0xe000u:mapped[29]+0x1000u;
          uint8_t known_left[64],known_right[64];
          if(known_c){memcpy(known_left,output-64,64);memcpy(known_right,output+4096u,64);}
          uint8_t* weights=mapped[2]+0x5e80u;
          uint8_t* packed=mapped[2]+0x8000u;
          uint8_t n0_prior[4096];memcpy(n0_prior,mapped[2]+0x6f80u,4096u);
          if(!no_agx_metal())return 2;
          for(unsigned i=0;i<0x1000u;i++)if(mapped[29][i]!=0x5a)return 2;
          if(known_c)memset(output,0,4096u);
          for(unsigned i=0;i<4096u;i++)if(output[i]!=0)return 2;
          for(unsigned i=0x2000u;i<(unsigned)sizes[29];i++)
            if(mapped[29][i]!=0x5a)return 2;
          // Use the already proven scalar copy image as a positive control for
          // GPU writes to this new allocation before asking TensorOps to use it.
          if(!known_c){
          memset(weights,0,128u);
          uint32_t scalar_packet=0x0e4006c7u;
          const uint64_t probe_bindings[3]={0x10000059000ull,0x10000035e80ull,
                                            0x100000f0080ull};
          uint8_t* probe_target=mapped[29]+0x80u;
          uint8_t probe_prior[128];memcpy(probe_prior,probe_target,128u);
          memset(probe_target,0,128u);
          memcpy(mapped[0]+0x6c0,copy_code,sizeof copy_code);
          memcpy(mapped[23]+0x40,&scalar_packet,4);
          memcpy(mapped[28]+0x1ba0,probe_bindings,sizeof probe_bindings);
          uint32_t probe_x=32u,probe_y=1u;
          memcpy(mapped[22]+0xa8,&probe_x,4);
          memcpy(mapped[22]+0xac,&probe_y,4);
          memcpy(mapped[25]+0x10,&probe_x,4);
          memcpy(mapped[25]+0x14,&probe_y,4);
          if(memcmp(mapped[0]+0x6c0,copy_code,sizeof copy_code) ||
             memcmp(mapped[28]+0x1ba0,probe_bindings,sizeof probe_bindings) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t probe_marks[2]={0};
          volatile uint64_t* pm=probe_marks;
          void (^scheduled_probe)(void)=Block_copy(^{pm[0]=0x268000u;});
          void (^completed_probe)(void)=Block_copy(^{pm[1]=0x268001u;});
          if(!scheduled_probe || !completed_probe ||
             scheduled_probe==completed_probe)return 2;
          uint8_t probe_record[64]={0},probe_out[64]={0};
          uint32_t probe_kid=shmem[1].id,probe_sid=shmem[0].id;
          uintptr_t probe_sp=(uintptr_t)scheduled_probe,
                    probe_cp=(uintptr_t)completed_probe;
          memcpy(probe_record,&probe_kid,4);memcpy(probe_record+4,&probe_sid,4);
          memcpy(probe_record+0x10,&probe_sp,sizeof probe_sp);
          memcpy(probe_record+0x18,&probe_cp,sizeof probe_cp);
          int probe_status=Submit(queue,NULL,1,probe_record,64,probe_out);
          unsigned probe_waited=0;
          for(;probe_waited<3000 && !pm[1];probe_waited++)usleep(10000);
          uint32_t probe_outword=0;memcpy(&probe_outword,probe_out,4);
          unsigned probe_exact=memcmp(probe_target,mapped[17]+0x1000u,128u)==0;
          unsigned probe_matching=0,probe_first_bad=32u;
          for(unsigned i=0;i<32;i++){
            unsigned same=memcmp(probe_target+4u*i,mapped[17]+0x1000u+4u*i,4u)==0;
            probe_matching+=same;if(!same && probe_first_bad==32u)probe_first_bad=i;
          }
          unsigned probe_tail=1;
          for(unsigned i=128;i<4096u;i++)probe_tail&=output[i]==0;
          uint32_t observed=0,expected=0;
          memcpy(&observed,probe_target,4);memcpy(&expected,mapped[17]+0x1000u,4);
          fprintf(stderr,"PURE STANDALONE N1 SCALAR PROBE status=%d outword=0x%08x exact=%u matching=%u/32 first_bad=%u tail=%u observed=0x%08x expected=0x%08x marks=%llu/%llu wait=%u/3000.\n",
                  probe_status,probe_outword,probe_exact,probe_matching,probe_first_bad,probe_tail,observed,expected,
                  (unsigned long long)pm[0],(unsigned long long)pm[1],probe_waited);
          if(probe_status || probe_outword || !probe_exact || !probe_tail ||
             pm[0]!=0x268000u || pm[1]!=0x268001u || !no_agx_metal())return 2;
          memcpy(probe_target,probe_prior,128u);
          memset(output,0,4096u);
          uint32_t tensor_packet=0x0e5806c7u;
          memcpy(mapped[0]+0x6c0,standalone_ffn_k6_accum_code,
                 sizeof standalone_ffn_k6_accum_code);
          memcpy(mapped[23]+0x40,&tensor_packet,4);
          uint32_t tensor_x=32u;
          memcpy(mapped[22]+0xa8,&tensor_x,4);
          memcpy(mapped[25]+0x10,&tensor_x,4);
          }
          memcpy(mapped[0]+0x6c0,standalone_ffn_k6_accum_code,
                 sizeof standalone_ffn_k6_accum_code);
          uint32_t n1_tensor_packet=0x0e5806c7u;
          memcpy(mapped[23]+0x40,&n1_tensor_packet,4);
          static volatile uint64_t marks[12]={0};
          volatile uint64_t* nm=marks;
          for(unsigned k=0;k<6;k++){
            uint8_t prior[4096];memcpy(prior,output,4096u);
            memcpy(weights,standalone_n1_b+k*4096u,4096u);
            const uint64_t bindings[4]={0x10000038000ull+(uint64_t)k*4096u,
                                        0x10000035e80ull,
                                        known_c?0x1000003e000ull:0x100000f1000ull,0};
            memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
            if(memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
               memcmp(weights,standalone_n1_b+k*4096u,4096u) ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            void (^scheduled)(void)=Block_copy(^{nm[2u*k]=0x270000u+2u*k;});
            void (^completed)(void)=Block_copy(^{nm[2u*k+1u]=0x270001u+2u*k;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t record[64]={0},out[64]={0};
            uint32_t kid=shmem[1].id,sid=shmem[0].id;
            uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
            memcpy(record,&kid,4);memcpy(record+4,&sid,4);
            memcpy(record+0x10,&sp,sizeof sp);
            memcpy(record+0x18,&cp,sizeof cp);
            int status=Submit(queue,NULL,1,record,64,out);
            unsigned waited=0;
            for(;waited<3000 && !nm[2u*k+1u];waited++)usleep(10000);
            unsigned changed=0,finite=1,within=1;
            double max_abs=0;
            for(unsigned i=0;i<1024;i++){
              uint32_t word=((uint32_t*)output)[i];
              changed+=word!=((uint32_t*)prior)[i];
              float value=0;memcpy(&value,&word,4);
              finite&=isfinite(value);
              double reference=standalone_n1_reference[k*1024u+i];
              double error=fabs((double)value-reference);
              double limit=2e-5*(1.0+fabs(reference));
              within&=isfinite(value) && error<=limit;
              if(error>max_abs)max_abs=error;
            }
            unsigned inputs=memcmp(packed,standalone_ffn_pack_expected,24576u)==0 &&
                            memcmp(weights,standalone_n1_b+k*4096u,4096u)==0 &&
                            memcmp(mapped[17]+0x1000u,standalone_ffn_pack_source,49152u)==0 &&
                            memcmp(mapped[2]+0x6f80u,n0_prior,4096u)==0;
            unsigned guards=1;
            for(unsigned i=0;i<0x1000u;i++)guards&=mapped[29][i]==0x5a;
            for(unsigned i=0x2000u;i<(unsigned)sizes[29];i++)
              guards&=mapped[29][i]==0x5a;
            if(known_c)guards&=memcmp(output-64,known_left,64)==0 &&
                               memcmp(output+4096u,known_right,64)==0;
            uint32_t outword=0;memcpy(&outword,out,4);
            fprintf(stderr,"PURE STANDALONE FFN N1 k=%u status=%d outword=0x%08x changed=%u/1024 finite=%u within=%u inputs=%u guards=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                    k,status,outword,changed,finite,within,inputs,guards,max_abs,
                    (unsigned long long)nm[2u*k],
                    (unsigned long long)nm[2u*k+1u],waited);
            fprintf(stderr,"PURE STANDALONE FFN N1 k=%u words:",k);
            for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",((uint32_t*)output)[i]);
            fputc('\n',stderr);
            if(status || outword || changed!=1024u || !finite || !within ||
               !inputs || !guards || nm[2u*k]!=0x270000u+2u*k ||
               nm[2u*k+1u]!=0x270001u+2u*k || !no_agx_metal())return 2;
          }
          fprintf(stderr,known_c?
                  "PURE STANDALONE FFN N1 KNOWN C COMPLETE: second complete FFN output tile retained in existing GPU allocation 2.\n":
                  "PURE STANDALONE FFN N1 COMPLETE: second complete FFN output tile retained in newly registered 256-KiB GPU allocation.\n");
        }
        if(ffn_standalone_residual_keep){
          uint8_t* source=mapped[17]+0x1000u;
          uint8_t* target=mapped[29]+0x1000u;
          uint8_t* zero=mapped[2]+0x5e80u;
          static const uint8_t zeros[4096]={0};
          if(memcmp(source,standalone_ffn_pack_source,49152u))return 2;
          for(unsigned i=0;i<49152u;i++)
            if(target[i]!=(i<4096u?0u:0x5au))return 2;
          memset(zero,0,4096u);
          memcpy(mapped[0]+0x6c0,copy_code,sizeof copy_code);
          if(getenv("ORDERED_FFN_RESIDUAL_SLOT29")){
            uint8_t* segment=(uint8_t*)(uintptr_t)shmem[0].cpu;
            uint32_t before_slot=0,after_slot=30u;
            memcpy(&before_slot,segment+0x5c,4);
            if(before_slot!=28u || !residual_row29)return 2;
            memcpy(segment+0x5c,&after_slot,4);
            fprintf(stderr,"PURE STANDALONE RESIDUAL SLOT29: segment +0x5c ordinal %u -> %u before bounded probe.\n",
                    before_slot,after_slot);
          }
          const uint32_t packet=0x0e4006c7u,grid_x=1024u,grid_y=1u;
          memcpy(mapped[23]+0x40,&packet,4);
          {
            uint8_t* probe=mapped[29]+0x80u;
            uint8_t prior[128];memcpy(prior,probe,128u);
            const uint32_t probe_x=32u;
            const uint64_t probe_bindings[3]={0x10000059000ull,0x10000035e80ull,
                                              0x100000f0080ull};
            memcpy(mapped[22]+0xa8,&probe_x,4);memcpy(mapped[22]+0xac,&grid_y,4);
            memcpy(mapped[25]+0x10,&probe_x,4);memcpy(mapped[25]+0x14,&grid_y,4);
            memcpy(mapped[28]+0x1ba0,probe_bindings,sizeof probe_bindings);
            static volatile uint64_t probe_marks[2]={0};
            volatile uint64_t* pm=probe_marks;
            void (^scheduled)(void)=Block_copy(^{pm[0]=0x30f000u;});
            void (^completed)(void)=Block_copy(^{pm[1]=0x30f001u;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t record[64]={0},out[64]={0};
            uint32_t kid=shmem[1].id,sid=shmem[0].id;
            uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
            memcpy(record,&kid,4);memcpy(record+4,&sid,4);
            memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
            int status=Submit(queue,NULL,1,record,64,out);
            unsigned waited=0;for(;waited<3000 && !pm[1];waited++)usleep(10000);
            unsigned matching=0,changed=0;
            for(unsigned i=0;i<32;i++){
              matching+=((uint32_t*)probe)[i]==((uint32_t*)source)[i];
              changed+=((uint32_t*)probe)[i]!=((uint32_t*)prior)[i];
            }
            uint32_t outword=0;memcpy(&outword,out,4);
            fprintf(stderr,"PURE STANDALONE RESIDUAL KEEP PROBE status=%d outword=0x%08x matching=%u/32 changed=%u/32 marks=%llu/%llu wait=%u/3000.\n",
                    status,outword,matching,changed,
                    (unsigned long long)pm[0],(unsigned long long)pm[1],waited);
            if(status || outword || matching!=32u || changed!=32u ||
               pm[0]!=0x30f000u || pm[1]!=0x30f001u || !no_agx_metal())return 2;
            memcpy(probe,prior,128u);
          }
          memcpy(mapped[22]+0xa8,&grid_x,4);memcpy(mapped[22]+0xac,&grid_y,4);
          memcpy(mapped[25]+0x10,&grid_x,4);memcpy(mapped[25]+0x14,&grid_y,4);
          static volatile uint64_t marks[24]={0};
          volatile uint64_t* pm=marks;
          for(unsigned chunk=0;chunk<12;chunk++){
            const uint64_t bindings[3]={0x10000059000ull+(uint64_t)chunk*4096u,
                                        0x10000035e80ull,
                                        0x100000f1000ull+(uint64_t)chunk*4096u};
            memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
            if(memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
               memcmp(source,standalone_ffn_pack_source,49152u) ||
               memcmp(zero,zeros,4096u) ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            void (^scheduled)(void)=Block_copy(^{pm[2u*chunk]=0x300000u+2u*chunk;});
            void (^completed)(void)=Block_copy(^{pm[2u*chunk+1u]=0x300001u+2u*chunk;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t record[64]={0},out[64]={0};
            uint32_t kid=shmem[1].id,sid=shmem[0].id;
            uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
            memcpy(record,&kid,4);memcpy(record+4,&sid,4);
            memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
            int status=Submit(queue,NULL,1,record,64,out);
            unsigned waited=0;for(;waited<3000 && !pm[2u*chunk+1u];waited++)usleep(10000);
            uint32_t outword=0;memcpy(&outword,out,4);
            unsigned exact=memcmp(target,source,(chunk+1u)*4096u)==0;
            unsigned matching=0,changed=0,first_bad=1024u;
            for(unsigned i=0;i<1024u;i++){
              uint32_t got=((uint32_t*)(target+chunk*4096u))[i];
              uint32_t want=((uint32_t*)(source+chunk*4096u))[i];
              matching+=got==want;changed+=got!=0x5a5a5a5au;
              if(got!=want && first_bad==1024u)first_bad=i;
            }
            unsigned tail=1;for(unsigned i=(chunk+1u)*4096u;i<49152u;i++)tail&=target[i]==0x5a;
            unsigned input=memcmp(source,standalone_ffn_pack_source,49152u)==0;
            fprintf(stderr,"PURE STANDALONE RESIDUAL KEEP chunk=%u status=%d outword=0x%08x exact=%u tail=%u input=%u marks=%llu/%llu wait=%u/3000.\n",
                    chunk,status,outword,exact,tail,input,
                    (unsigned long long)pm[2u*chunk],(unsigned long long)pm[2u*chunk+1u],waited);
            uint32_t got=first_bad<1024u?((uint32_t*)(target+chunk*4096u))[first_bad]:0;
            uint32_t want=first_bad<1024u?((uint32_t*)(source+chunk*4096u))[first_bad]:0;
            fprintf(stderr,"PURE STANDALONE RESIDUAL KEEP DIAG chunk=%u matching=%u/1024 changed=%u/1024 first_bad=%u got=0x%08x want=0x%08x.\n",
                    chunk,matching,changed,first_bad,got,want);
            if(status || outword || !exact || !tail || !input ||
               pm[2u*chunk]!=0x300000u+2u*chunk ||
               pm[2u*chunk+1u]!=0x300001u+2u*chunk || !no_agx_metal())return 2;
          }
          if(getenv("ORDERED_FFN_RESIDUAL_SLOT29")){
            uint8_t* segment=(uint8_t*)(uintptr_t)shmem[0].cpu;
            uint32_t before_slot=0,after_slot=28u;
            memcpy(&before_slot,segment+0x5c,4);
            if(before_slot!=30u)return 2;
            memcpy(segment+0x5c,&after_slot,4);
            fprintf(stderr,"PURE STANDALONE RESIDUAL SLOT29 RESTORE: segment +0x5c ordinal %u -> %u after twelve copies.\n",
                    before_slot,after_slot);
          }
          memcpy(mapped[0]+0x6c0,standalone_ffn_k6_accum_code,
                 sizeof standalone_ffn_k6_accum_code);
          {
            const uint32_t tensor_packet=0x0e5806c7u,tensor_x=32u;
            memcpy(mapped[23]+0x40,&tensor_packet,4);
            memcpy(mapped[22]+0xa8,&tensor_x,4);
            memcpy(mapped[25]+0x10,&tensor_x,4);
          }
          fprintf(stderr,"PURE STANDALONE RESIDUAL KEEP COMPLETE: 49152 attention-output bytes copied into allocation 29 by twelve GPU launches.\n");
        }
        if(ffn_standalone_expand_all){
          uint8_t n0_saved[4096],n1_saved[4096];
          memcpy(n0_saved,mapped[2]+0x6f80u,4096u);
          memcpy(n1_saved,mapped[2]+0xe000u,4096u);
          uint8_t* completed=malloc(46u*4096u);
          uint8_t* completed_slots[46]={0};
          if(!completed)return 2;
          static volatile uint64_t marks[46u*6u*2u]={0};
          volatile uint64_t* em=marks;
          for(unsigned n=2;n<48;n++){
            unsigned index=n-2u;
            unsigned offset=n<=18u?0xf000u+(n-2u)*4096u:
                                    0x1000u+(n-19u)*4096u;
            unsigned allocation=n<=18u?2u:17u;
            uint8_t* output=mapped[allocation]+offset;
            uint64_t c_va=(n<=18u?0x10000030000ull:0x10000058000ull)+offset;
            uint8_t left[64],right[64];
            memcpy(left,output-64,64);
            unsigned right_valid=offset+4096u+64u<=sizes[allocation];
            if(right_valid)memcpy(right,output+4096u,64);
            memset(output,0,4096u);
            completed_slots[index]=output;
            for(unsigned k=0;k<6;k++){
              unsigned trial=index*6u+k;
              uint8_t prior[4096];memcpy(prior,output,4096u);
              uint8_t* weights=mapped[2]+0x5e80u;
              const uint8_t* b=standalone_expand_b+trial*4096u;
              memcpy(weights,b,4096u);
              const uint64_t bindings[4]={0x10000038000ull+(uint64_t)k*4096u,
                                          0x10000035e80ull,c_va,0};
              memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
              if(memcmp(mapped[0]+0x6c0,standalone_ffn_k6_accum_code,
                        sizeof standalone_ffn_k6_accum_code) ||
                 memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
                 memcmp(weights,b,4096u) || *full_ready!=0x800000f0u ||
                 !no_agx_metal())return 2;
              void (^scheduled)(void)=Block_copy(^{em[2u*trial]=0x280000u+2u*trial;});
              void (^completed_callback)(void)=Block_copy(^{em[2u*trial+1u]=0x280001u+2u*trial;});
              if(!scheduled || !completed_callback ||
                 scheduled==completed_callback)return 2;
              uint8_t record[64]={0},out[64]={0};
              uint32_t kid=shmem[1].id,sid=shmem[0].id;
              uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed_callback;
              memcpy(record,&kid,4);memcpy(record+4,&sid,4);
              memcpy(record+0x10,&sp,sizeof sp);
              memcpy(record+0x18,&cp,sizeof cp);
              int status=Submit(queue,NULL,1,record,64,out);
              unsigned waited=0;
              for(;waited<3000 && !em[2u*trial+1u];waited++)usleep(10000);
              uint32_t outword=0;memcpy(&outword,out,4);
              unsigned changed=0,finite=1,within=1;
              double max_abs=0;
              for(unsigned i=0;i<1024;i++){
                uint32_t word=((uint32_t*)output)[i];
                changed+=word!=((uint32_t*)prior)[i];
                float value=0;memcpy(&value,&word,4);
                finite&=isfinite(value);
                double reference=standalone_expand_reference[trial*1024u+i];
                double error=fabs((double)value-reference);
                double limit=2e-5*(1.0+fabs(reference));
                within&=isfinite(value) && error<=limit;
                if(error>max_abs)max_abs=error;
              }
              unsigned guards=memcmp(output-64,left,64)==0 &&
                              (!right_valid || memcmp(output+4096u,right,64)==0);
              unsigned prior_outputs=memcmp(mapped[2]+0x6f80u,n0_saved,4096u)==0 &&
                                     memcmp(mapped[2]+0xe000u,n1_saved,4096u)==0;
              for(unsigned p=0;p<index;p++)prior_outputs&=
                memcmp(completed_slots[p],completed+p*4096u,4096u)==0;
              unsigned inputs=memcmp(mapped[2]+0x8000u,
                                     standalone_ffn_pack_expected,24576u)==0 &&
                              memcmp(weights,b,4096u)==0;
              fprintf(stderr,"PURE STANDALONE EXPAND n=%u k=%u status=%d outword=0x%08x changed=%u/1024 finite=%u within=%u inputs=%u guards=%u prior=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                      n,k,status,outword,changed,finite,within,inputs,guards,
                      prior_outputs,max_abs,(unsigned long long)em[2u*trial],
                      (unsigned long long)em[2u*trial+1u],waited);
              fprintf(stderr,"PURE STANDALONE EXPAND n=%u k=%u words:",n,k);
              for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",((uint32_t*)output)[i]);
              fputc('\n',stderr);
              if(status || outword || changed!=1024u || !finite || !within ||
                 !inputs || !guards || !prior_outputs ||
                 em[2u*trial]!=0x280000u+2u*trial ||
                 em[2u*trial+1u]!=0x280001u+2u*trial ||
                 !no_agx_metal())return 2;
            }
            memcpy(completed+index*4096u,output,4096u);
          }
          free(completed);
          fprintf(stderr,"PURE STANDALONE EXPAND ALL COMPLETE: 48 six-block FFN expansion tiles retained across allocations 2 and 17.\n");
        }
        if(ffn_standalone_wide_probe){
          uint8_t* tile=mapped[17]+0x1d000u;
          uint64_t tile_va=0x10000075000ull;
          uint8_t* bias=mapped[2]+0x5e80u;
          if(memcmp(tile,standalone_wide_prior,4096u) || !no_agx_metal())return 2;
          uint8_t left[64],right[64];
          memcpy(left,tile-64,64);memcpy(right,tile+4096u,64);
          uint8_t* others=malloc(47u*4096u);
          uint8_t* other_slots[47]={0};
          if(!others)return 2;
          other_slots[0]=mapped[2]+0x6f80u;
          other_slots[1]=mapped[2]+0xe000u;
          for(unsigned n=2;n<47;n++)
            other_slots[n]=n<=18u?mapped[2]+0xf000u+(n-2u)*4096u:
                                   mapped[17]+0x1000u+(n-19u)*4096u;
          for(unsigned n=0;n<47;n++)memcpy(others+n*4096u,other_slots[n],4096u);
          memcpy(bias,standalone_wide_bias,4096u);
          const uint32_t packet=0x0e4006c7u,wide_x=1024u,wide_y=1u;
          memcpy(mapped[23]+0x40,&packet,4);
          memcpy(mapped[22]+0xa8,&wide_x,4);memcpy(mapped[22]+0xac,&wide_y,4);
          memcpy(mapped[25]+0x10,&wide_x,4);memcpy(mapped[25]+0x14,&wide_y,4);
          static volatile uint64_t marks[4]={0};
          volatile uint64_t* wm=marks;
          for(unsigned phase=0;phase<2;phase++){
            const uint64_t bindings[4]={tile_va,phase?tile_va:0x10000035e80ull,
                                        tile_va,0};
            memcpy(mapped[0]+0x6c0,phase?gelu_code:bias_code,
                   phase?sizeof gelu_code:sizeof bias_code);
            memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
            if(memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
               memcmp(bias,standalone_wide_bias,4096u) ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            void (^scheduled)(void)=Block_copy(^{wm[2u*phase]=0x290000u+2u*phase;});
            void (^completed)(void)=Block_copy(^{wm[2u*phase+1u]=0x290001u+2u*phase;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t record[64]={0},out[64]={0};
            uint32_t kid=shmem[1].id,sid=shmem[0].id;
            uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
            memcpy(record,&kid,4);memcpy(record+4,&sid,4);
            memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
            int status=Submit(queue,NULL,1,record,64,out);
            unsigned waited=0;for(;waited<3000 && !wm[2u*phase+1u];waited++)usleep(10000);
            uint32_t outword=0;memcpy(&outword,out,4);
            unsigned finite=1,within=1,changed=0;
            double max_abs=0;
            for(unsigned i=0;i<1024;i++){
              uint32_t word=((uint32_t*)tile)[i];
              uint32_t prior=0;memcpy(&prior,standalone_wide_prior+4u*i,4u);
              changed+=word!=prior;
              float value=0;memcpy(&value,&word,4);finite&=isfinite(value);
              if(phase){
                double reference=standalone_wide_gelu_reference[i];
                double error=fabs((double)value-reference);
                double limit=2e-5*(1.0+fabs(reference));
                within&=isfinite(value) && error<=limit;
                if(error>max_abs)max_abs=error;
              }
            }
            if(!phase)within=memcmp(tile,standalone_wide_biased,4096u)==0;
            unsigned guards=memcmp(tile-64,left,64)==0 &&
                            memcmp(tile+4096u,right,64)==0;
            unsigned prior_outputs=1;
            for(unsigned n=0;n<47;n++)prior_outputs&=
              memcmp(other_slots[n],others+n*4096u,4096u)==0;
            unsigned inputs=memcmp(bias,standalone_wide_bias,4096u)==0;
            fprintf(stderr,"PURE STANDALONE WIDE phase=%u status=%d outword=0x%08x changed=%u/1024 finite=%u within=%u inputs=%u guards=%u prior=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                    phase,status,outword,changed,finite,within,inputs,guards,
                    prior_outputs,max_abs,(unsigned long long)wm[2u*phase],
                    (unsigned long long)wm[2u*phase+1u],waited);
            fprintf(stderr,"PURE STANDALONE WIDE phase=%u words:",phase);
            for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",((uint32_t*)tile)[i]);
            fputc('\n',stderr);
            if(status || outword || !changed || !finite || !within ||
               !inputs || !guards || !prior_outputs ||
               wm[2u*phase]!=0x290000u+2u*phase ||
               wm[2u*phase+1u]!=0x290001u+2u*phase ||
               !no_agx_metal())return 2;
          }
          free(others);
          fprintf(stderr,"PURE STANDALONE WIDE COMPLETE: n47 bias and GELU each processed 1024 GPU-resident values in one Submit.\n");
        }
        if(ffn_standalone_act_all){
          uint8_t* slots[48]={0};
          uint64_t vas[48]={0};
          slots[0]=mapped[2]+0x6f80u;vas[0]=0x10000036f80ull;
          slots[1]=mapped[2]+0xe000u;vas[1]=0x1000003e000ull;
          for(unsigned n=2;n<48;n++){
            unsigned offset=n<=18u?0xf000u+(n-2u)*4096u:
                                    0x1000u+(n-19u)*4096u;
            unsigned allocation=n<=18u?2u:17u;
            slots[n]=mapped[allocation]+offset;
            vas[n]=(n<=18u?0x10000030000ull:0x10000058000ull)+offset;
          }
          uint8_t* state=malloc(48u*4096u);
          if(!state)return 2;
          memcpy(state,standalone_act_prior,48u*4096u);
          for(unsigned n=0;n<48;n++)
            if(memcmp(slots[n],state+n*4096u,4096u))return 2;
          const uint32_t packet=0x0e4006c7u,wide_x=1024u,wide_y=1u;
          memcpy(mapped[23]+0x40,&packet,4);
          memcpy(mapped[22]+0xa8,&wide_x,4);memcpy(mapped[22]+0xac,&wide_y,4);
          memcpy(mapped[25]+0x10,&wide_x,4);memcpy(mapped[25]+0x14,&wide_y,4);
          static volatile uint64_t marks[2u*48u*2u]={0};
          volatile uint64_t* am=marks;
          for(unsigned phase=0;phase<2;phase++){
            memcpy(mapped[0]+0x6c0,phase?gelu_code:bias_code,
                   phase?sizeof gelu_code:sizeof bias_code);
            for(unsigned n=0;n<48;n++){
              unsigned t=phase*48u+n;
              uint8_t* tile=slots[n];
              uint8_t left[64],right[64];
              memcpy(left,tile-64,64);
              unsigned right_valid=n!=18u;
              if(right_valid)memcpy(right,tile+4096u,64);
              if(memcmp(tile,state+n*4096u,4096u))return 2;
              uint8_t* bias=mapped[2]+0x5e80u;
              memcpy(bias,standalone_act_bias+n*4096u,4096u);
              const uint64_t bindings[4]={vas[n],phase?vas[n]:0x10000035e80ull,
                                          vas[n],0};
              memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
              if(memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
                 memcmp(bias,standalone_act_bias+n*4096u,4096u) ||
                 *full_ready!=0x800000f0u || !no_agx_metal())return 2;
              void (^scheduled)(void)=Block_copy(^{am[2u*t]=0x2a0000u+2u*t;});
              void (^completed)(void)=Block_copy(^{am[2u*t+1u]=0x2a0001u+2u*t;});
              if(!scheduled || !completed || scheduled==completed)return 2;
              uint8_t record[64]={0},out[64]={0};
              uint32_t kid=shmem[1].id,sid=shmem[0].id;
              uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
              memcpy(record,&kid,4);memcpy(record+4,&sid,4);
              memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
              int status=Submit(queue,NULL,1,record,64,out);
              unsigned waited=0;for(;waited<3000 && !am[2u*t+1u];waited++)usleep(10000);
              uint32_t outword=0;memcpy(&outword,out,4);
              unsigned changed=0,finite=1,within=1;
              double max_abs=0;
              for(unsigned i=0;i<1024;i++){
                uint32_t word=((uint32_t*)tile)[i];
                uint32_t old=0;memcpy(&old,state+n*4096u+4u*i,4u);
                changed+=word!=old;
                float value=0;memcpy(&value,&word,4);finite&=isfinite(value);
                if(phase){
                  double ref=standalone_act_gelu_reference[n*1024u+i];
                  double error=fabs((double)value-ref);
                  double limit=2e-5*(1.0+fabs(ref));
                  within&=isfinite(value) && error<=limit;
                  if(error>max_abs)max_abs=error;
                }
              }
              if(!phase)within=memcmp(tile,standalone_act_biased+n*4096u,4096u)==0;
              unsigned guards=memcmp(tile-64,left,64)==0 &&
                              (!right_valid || memcmp(tile+4096u,right,64)==0);
              unsigned other_tiles=1;
              for(unsigned p=0;p<48;p++)if(p!=n)other_tiles&=
                memcmp(slots[p],state+p*4096u,4096u)==0;
              unsigned input=memcmp(bias,standalone_act_bias+n*4096u,4096u)==0;
              fprintf(stderr,"PURE STANDALONE ACT phase=%u n=%u status=%d outword=0x%08x changed=%u/1024 finite=%u within=%u input=%u guards=%u others=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                      phase,n,status,outword,changed,finite,within,input,guards,
                      other_tiles,max_abs,(unsigned long long)am[2u*t],
                      (unsigned long long)am[2u*t+1u],waited);
              fprintf(stderr,"PURE STANDALONE ACT phase=%u n=%u words:",phase,n);
              for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",((uint32_t*)tile)[i]);
              fputc('\n',stderr);
              if(status || outword || !changed || !finite || !within ||
                 !input || !guards || !other_tiles ||
                 am[2u*t]!=0x2a0000u+2u*t ||
                 am[2u*t+1u]!=0x2a0001u+2u*t || !no_agx_metal())return 2;
              memcpy(state+n*4096u,tile,4096u);
            }
            fprintf(stderr,"PURE STANDALONE ACT PHASE %u COMPLETE: 48 resident tiles.\n",phase);
          }
          free(state);
          fprintf(stderr,"PURE STANDALONE ACT ALL COMPLETE: full 32x1536 bias and GELU output resident after 96 wide Submits.\n");
        }
        if(ffn_standalone_contract_k0){
          uint8_t* scratch=mapped[17]+0x1e000u;
          uint8_t* sources[2]={mapped[2]+0x6f80u,mapped[2]+0xe000u};
          const uint64_t source_vas[2]={0x10000036f80ull,0x1000003e000ull};
          const uint64_t scratch_va=0x10000076000ull;
          for(unsigned n=0;n<2;n++)
            if(memcmp(sources[n],standalone_contract_activated+n*4096u,4096u))return 2;
          uint8_t scratch_left[64],scratch_right[64];
          memcpy(scratch_left,scratch-64,64);memcpy(scratch_right,scratch+4096u,64);
          memset(scratch,0,4096u);
          const uint32_t scalar_packet=0x0e4006c7u,pack_x=32u,pack_y=32u;
          memcpy(mapped[0]+0x6c0,standalone_pair_pack_code,sizeof standalone_pair_pack_code);
          memcpy(mapped[23]+0x40,&scalar_packet,4);
          memcpy(mapped[22]+0xa8,&pack_x,4);memcpy(mapped[22]+0xac,&pack_y,4);
          memcpy(mapped[25]+0x10,&pack_x,4);memcpy(mapped[25]+0x14,&pack_y,4);
          static volatile uint64_t marks[6]={0};
          volatile uint64_t* cm=marks;
          for(unsigned half=0;half<2;half++){
            const uint64_t bindings[3]={source_vas[half],scratch_va+64u*half,0};
            memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
            if(memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
               memcmp(mapped[0]+0x6c0,standalone_pair_pack_code,
                      sizeof standalone_pair_pack_code) ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            void (^scheduled)(void)=Block_copy(^{cm[2u*half]=0x2b0000u+2u*half;});
            void (^completed)(void)=Block_copy(^{cm[2u*half+1u]=0x2b0001u+2u*half;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t record[64]={0},out[64]={0};
            uint32_t kid=shmem[1].id,sid=shmem[0].id;
            uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
            memcpy(record,&kid,4);memcpy(record+4,&sid,4);
            memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
            int status=Submit(queue,NULL,1,record,64,out);
            unsigned waited=0;for(;waited<3000 && !cm[2u*half+1u];waited++)usleep(10000);
            uint32_t outword=0;memcpy(&outword,out,4);
            unsigned exact=1;
            for(unsigned row=0;row<32;row++){
              exact&=memcmp(scratch+row*128u+half*64u,
                            standalone_pair_expected+row*128u+half*64u,64u)==0;
              if(!half)for(unsigned i=0;i<64u;i++)
                exact&=scratch[row*128u+64u+i]==0;
            }
            unsigned guards=memcmp(scratch-64,scratch_left,64)==0 &&
                            memcmp(scratch+4096u,scratch_right,64)==0;
            unsigned sources_ok=memcmp(sources[0],standalone_contract_activated,4096u)==0 &&
                                memcmp(sources[1],standalone_contract_activated+4096u,4096u)==0;
            fprintf(stderr,"PURE STANDALONE PAIR PACK half=%u status=%d outword=0x%08x exact=%u sources=%u guards=%u marks=%llu/%llu wait=%u/3000.\n",
                    half,status,outword,exact,sources_ok,guards,
                    (unsigned long long)cm[2u*half],
                    (unsigned long long)cm[2u*half+1u],waited);
            if(status || outword || !exact || !sources_ok || !guards ||
               cm[2u*half]!=0x2b0000u+2u*half ||
               cm[2u*half+1u]!=0x2b0001u+2u*half || !no_agx_metal())return 2;
          }
          fprintf(stderr,"PURE STANDALONE PAIR PACK half words:");
          for(unsigned i=0;i<2048;i++)fprintf(stderr," %04x",((uint16_t*)scratch)[i]);
          fputc('\n',stderr);
          if(memcmp(scratch,standalone_pair_expected,4096u))return 2;
          uint8_t* output=sources[0];
          uint8_t out_left[64],out_right[64];
          memcpy(out_left,output-64,64);memcpy(out_right,output+4096u,64);
          memcpy(mapped[2]+0x5e80u,standalone_contract_b,4096u);
          for(unsigned i=0;i<1024;i++)((uint32_t*)output)[i]=0x7fc01234u;
          const uint32_t tensor_packet=0x0e5806c7u;
          const uint32_t tensor_x=32u,tensor_y=1u;
          const uint32_t scratch_word[2]={0x0c00100fu,0};
          const uint64_t bindings[4]={scratch_va,0x10000035e80ull,0x10000036f80ull,0};
          memcpy(mapped[0]+0x6c0,standalone_ffn_tile_code,sizeof standalone_ffn_tile_code);
          memcpy(mapped[23]+0x38,scratch_word,sizeof scratch_word);
          memcpy(mapped[23]+0x40,&tensor_packet,4);
          memcpy(mapped[22]+0xa8,&tensor_x,4);memcpy(mapped[22]+0xac,&tensor_y,4);
          memcpy(mapped[25]+0x10,&tensor_x,4);memcpy(mapped[25]+0x14,&tensor_y,4);
          memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
          if(memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
             memcmp(mapped[2]+0x5e80u,standalone_contract_b,4096u) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          void (^scheduled_tensor)(void)=Block_copy(^{cm[4]=0x2b0004u;});
          void (^completed_tensor)(void)=Block_copy(^{cm[5]=0x2b0005u;});
          if(!scheduled_tensor || !completed_tensor ||
             scheduled_tensor==completed_tensor)return 2;
          uint8_t record[64]={0},out[64]={0};
          uint32_t kid=shmem[1].id,sid=shmem[0].id;
          uintptr_t sp=(uintptr_t)scheduled_tensor,cp=(uintptr_t)completed_tensor;
          memcpy(record,&kid,4);memcpy(record+4,&sid,4);
          memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
          int status=Submit(queue,NULL,1,record,64,out);
          unsigned waited=0;for(;waited<3000 && !cm[5];waited++)usleep(10000);
          uint32_t outword=0;memcpy(&outword,out,4);
          unsigned changed=0,finite=1,within=1;
          double max_abs=0;
          for(unsigned i=0;i<1024;i++){
            uint32_t word=((uint32_t*)output)[i];
            changed+=word!=0x7fc01234u;
            float value=0;memcpy(&value,&word,4);finite&=isfinite(value);
            double ref=standalone_contract_reference[i];
            double error=fabs((double)value-ref);
            double limit=2e-5*(1.0+fabs(ref));
            within&=isfinite(value) && error<=limit;
            if(error>max_abs)max_abs=error;
          }
          unsigned inputs=memcmp(scratch,standalone_pair_expected,4096u)==0 &&
                          memcmp(mapped[2]+0x5e80u,standalone_contract_b,4096u)==0;
          unsigned guards=memcmp(output-64,out_left,64)==0 &&
                          memcmp(output+4096u,out_right,64)==0;
          unsigned others=1;
          for(unsigned n=1;n<48;n++){
            uint8_t* tile=n==1u?mapped[2]+0xe000u:
                          n<=18u?mapped[2]+0xf000u+(n-2u)*4096u:
                                  mapped[17]+0x1000u+(n-19u)*4096u;
            others&=memcmp(tile,standalone_contract_activated+n*4096u,4096u)==0;
          }
          fprintf(stderr,"PURE STANDALONE CONTRACT K0 status=%d outword=0x%08x changed=%u/1024 finite=%u within=%u inputs=%u guards=%u others=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                  status,outword,changed,finite,within,inputs,guards,others,max_abs,
                  (unsigned long long)cm[4],(unsigned long long)cm[5],waited);
          fprintf(stderr,"PURE STANDALONE CONTRACT K0 words:");
          for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",((uint32_t*)output)[i]);
          fputc('\n',stderr);
          if(status || outword || changed!=1024u || !finite || !within ||
             !inputs || !guards || !others || cm[4]!=0x2b0004u ||
             cm[5]!=0x2b0005u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE STANDALONE CONTRACT K0 COMPLETE: GPU-packed activation pair consumed by model-weight TensorOps.\n");
        }
        if(ffn_standalone_contract_k24){
          uint8_t* scratch=mapped[17]+0x1e000u;
          uint8_t* output=mapped[2]+0x6f80u;
          uint8_t* weights=mapped[2]+0x5e80u;
          const uint64_t scratch_va=0x10000076000ull;
          const uint64_t output_va=0x10000036f80ull;
          const uint64_t weight_va=0x10000035e80ull;
          const uint32_t scalar_packet=0x0e4006c7u,tensor_packet=0x0e5806c7u;
          const uint32_t pack_x=32u,pack_y=32u,tensor_x=32u,tensor_y=1u;
          const uint32_t scratch_word[2]={0x0c00100fu,0};
          uint8_t scratch_left[64],scratch_right[64],out_left[64],out_right[64];
          memcpy(scratch_left,scratch-64,64);memcpy(scratch_right,scratch+4096u,64);
          memcpy(out_left,output-64,64);memcpy(out_right,output+4096u,64);
          static volatile uint64_t marks[138]={0};
          volatile uint64_t* km=marks;
          for(unsigned k=1;k<24;k++){
            const uint8_t* expected_pack=standalone_contract_k24_packed+(k-1u)*4096u;
            const uint8_t* expected_b=standalone_contract_k24_b+(k-1u)*4096u;
            const double* reference=standalone_contract_k24_reference+(k-1u)*1024u;
            memset(scratch,0,4096u);
            memcpy(mapped[0]+0x6c0,standalone_pair_pack_code,sizeof standalone_pair_pack_code);
            memcpy(mapped[23]+0x40,&scalar_packet,4);
            memcpy(mapped[22]+0xa8,&pack_x,4);memcpy(mapped[22]+0xac,&pack_y,4);
            memcpy(mapped[25]+0x10,&pack_x,4);memcpy(mapped[25]+0x14,&pack_y,4);
            for(unsigned half=0;half<2;half++){
              unsigned n=2u*k+half;
              uint32_t offset=n<=18u?0xf000u+(n-2u)*4096u:
                                      0x1000u+(n-19u)*4096u;
              unsigned allocation=n<=18u?2u:17u;
              uint8_t* source=mapped[allocation]+offset;
              uint64_t source_va=(allocation==2u?0x10000030000ull:
                                                  0x10000058000ull)+offset;
              const uint64_t bindings[3]={source_va,scratch_va+64u*half,0};
              unsigned t=3u*(k-1u)+half;
              memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
              if(memcmp(source,standalone_contract_activated+n*4096u,4096u) ||
                 memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
                 *full_ready!=0x800000f0u || !no_agx_metal())return 2;
              void (^scheduled)(void)=Block_copy(^{km[2u*t]=0x2c0000u+2u*t;});
              void (^completed)(void)=Block_copy(^{km[2u*t+1u]=0x2c0001u+2u*t;});
              if(!scheduled || !completed || scheduled==completed)return 2;
              uint8_t record[64]={0},out[64]={0};
              uint32_t kid=shmem[1].id,sid=shmem[0].id;
              uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
              memcpy(record,&kid,4);memcpy(record+4,&sid,4);
              memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
              int status=Submit(queue,NULL,1,record,64,out);
              unsigned waited=0;for(;waited<3000 && !km[2u*t+1u];waited++)usleep(10000);
              uint32_t outword=0;memcpy(&outword,out,4);
              unsigned exact=1;
              for(unsigned row=0;row<32;row++){
                exact&=memcmp(scratch+row*128u+half*64u,
                              expected_pack+row*128u+half*64u,64u)==0;
                if(!half)for(unsigned i=0;i<64;i++)
                  exact&=scratch[row*128u+64u+i]==0;
              }
              unsigned guards=memcmp(scratch-64,scratch_left,64)==0 &&
                              memcmp(scratch+4096u,scratch_right,64)==0;
              fprintf(stderr,"PURE STANDALONE CONTRACT K24 PACK k=%u half=%u status=%d outword=0x%08x exact=%u source=%u guards=%u marks=%llu/%llu wait=%u/3000.\n",
                      k,half,status,outword,exact,
                      memcmp(source,standalone_contract_activated+n*4096u,4096u)==0,
                      guards,(unsigned long long)km[2u*t],
                      (unsigned long long)km[2u*t+1u],waited);
              if(status || outword || !exact || !guards ||
                 km[2u*t]!=0x2c0000u+2u*t ||
                 km[2u*t+1u]!=0x2c0001u+2u*t || !no_agx_metal())return 2;
            }
            fprintf(stderr,"PURE STANDALONE CONTRACT K24 PACK k=%u words:",k);
            for(unsigned i=0;i<2048;i++)fprintf(stderr," %04x",((uint16_t*)scratch)[i]);
            fputc('\n',stderr);
            if(memcmp(scratch,expected_pack,4096u))return 2;
            uint8_t prior[4096];memcpy(prior,output,4096u);
            memcpy(weights,expected_b,4096u);
            memcpy(mapped[0]+0x6c0,standalone_ffn_k6_accum_code,
                   sizeof standalone_ffn_k6_accum_code);
            memcpy(mapped[23]+0x38,scratch_word,sizeof scratch_word);
            memcpy(mapped[23]+0x40,&tensor_packet,4);
            memcpy(mapped[22]+0xa8,&tensor_x,4);memcpy(mapped[22]+0xac,&tensor_y,4);
            memcpy(mapped[25]+0x10,&tensor_x,4);memcpy(mapped[25]+0x14,&tensor_y,4);
            const uint64_t bindings[4]={scratch_va,weight_va,output_va,0};
            memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
            if(memcmp(weights,expected_b,4096u) ||
               memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            unsigned t=3u*(k-1u)+2u;
            void (^scheduled)(void)=Block_copy(^{km[2u*t]=0x2c0000u+2u*t;});
            void (^completed)(void)=Block_copy(^{km[2u*t+1u]=0x2c0001u+2u*t;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t record[64]={0},out[64]={0};
            uint32_t kid=shmem[1].id,sid=shmem[0].id;
            uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
            memcpy(record,&kid,4);memcpy(record+4,&sid,4);
            memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
            int status=Submit(queue,NULL,1,record,64,out);
            unsigned waited=0;for(;waited<3000 && !km[2u*t+1u];waited++)usleep(10000);
            uint32_t outword=0;memcpy(&outword,out,4);
            unsigned changed=0,finite=1,within=1;double max_abs=0;
            for(unsigned i=0;i<1024;i++){
              changed+=((uint32_t*)output)[i]!=((uint32_t*)prior)[i];
              float value=((float*)output)[i];finite&=isfinite(value);
              double error=fabs((double)value-reference[i]);
              within&=isfinite(value) && error<=2e-5*(1.0+fabs(reference[i]));
              if(error>max_abs)max_abs=error;
            }
            unsigned inputs=memcmp(scratch,expected_pack,4096u)==0 &&
                            memcmp(weights,expected_b,4096u)==0;
            unsigned guards=memcmp(output-64,out_left,64)==0 &&
                            memcmp(output+4096u,out_right,64)==0;
            unsigned others=1;
            for(unsigned n=1;n<48;n++){
              uint8_t* tile=n==1u?mapped[2]+0xe000u:
                            n<=18u?mapped[2]+0xf000u+(n-2u)*4096u:
                                    mapped[17]+0x1000u+(n-19u)*4096u;
              others&=memcmp(tile,standalone_contract_activated+n*4096u,4096u)==0;
            }
            fprintf(stderr,"PURE STANDALONE CONTRACT K24 k=%u status=%d outword=0x%08x changed=%u/1024 finite=%u within=%u inputs=%u guards=%u others=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                    k,status,outword,changed,finite,within,inputs,guards,others,
                    max_abs,(unsigned long long)km[2u*t],
                    (unsigned long long)km[2u*t+1u],waited);
            fprintf(stderr,"PURE STANDALONE CONTRACT K24 k=%u words:",k);
            for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",((uint32_t*)output)[i]);
            fputc('\n',stderr);
            if(status || outword || !changed || !finite || !within || !inputs ||
               !guards || !others || km[2u*t]!=0x2c0000u+2u*t ||
               km[2u*t+1u]!=0x2c0001u+2u*t || !no_agx_metal())return 2;
          }
          fprintf(stderr,"PURE STANDALONE CONTRACT K24 COMPLETE: one output tile accumulated across 24 GPU-packed K64 blocks.\n");
        }
        if(ffn_standalone_contract_n01 || ffn_standalone_contract_n8 ||
           ffn_standalone_contract_n12){
          unsigned count=ffn_standalone_contract_n12?12u:
                         ffn_standalone_contract_n8?8u:2u;
          unsigned span=2u+count;
          const char* label=ffn_standalone_contract_n12?"N12":
                            ffn_standalone_contract_n8?"N8":"N01";
          uint8_t* scratch=mapped[17]+0x1e000u;
          uint8_t* outputs[12]={mapped[2]+0x6f80u,mapped[2]+0xe000u};
          uint64_t output_vas[12]={0x10000036f80ull,0x1000003e000ull};
          unsigned old_count=count<8u?count:8u;
          for(unsigned n=2;n<old_count;n++){
            outputs[n]=mapped[2]+0x8000u+(n-2u)*4096u;
            output_vas[n]=0x10000038000ull+(uint64_t)(n-2u)*4096u;
          }
          if(ffn_standalone_contract_n12){
            outputs[8]=mapped[2]+0x800u; output_vas[8]=0x10000030800ull;
            outputs[9]=mapped[2]+0x1c00u;output_vas[9]=0x10000031c00ull;
            outputs[10]=mapped[2]+0x4d80u;output_vas[10]=0x10000034d80ull;
            outputs[11]=mapped[17]+0x1f000u;output_vas[11]=0x10000077000ull;
          }
          uint8_t* weights=mapped[2]+0x5e80u;
          const uint64_t scratch_va=0x10000076000ull,weight_va=0x10000035e80ull;
          const uint32_t scalar_packet=0x0e4006c7u,tensor_packet=0x0e5806c7u;
          const uint32_t pack_x=32u,pack_y=32u,tensor_x=32u,tensor_y=1u;
          const uint32_t scratch_word[2]={0x0c00100fu,0};
          uint8_t scratch_left[64],scratch_right[64],out_left[2][64],out_right[2][64];
          uint8_t extra_left[3][64],extra_right[3][64];
          memcpy(scratch_left,scratch-64,64);memcpy(scratch_right,scratch+4096u,64);
          for(unsigned n=0;n<2;n++){
            if(memcmp(outputs[n],standalone_contract_n01_activated+n*4096u,4096u))return 2;
            memcpy(out_left[n],outputs[n]-64,64);
            memcpy(out_right[n],outputs[n]+4096u,64);
          }
          if(ffn_standalone_contract_n12)for(unsigned n=8;n<=10;n++){
            memcpy(extra_left[n-8u],outputs[n]-64,64);
            memcpy(extra_right[n-8u],outputs[n]+4096u,64);
          }
          static volatile uint64_t marks[672]={0};
          volatile uint64_t* pm=marks;
          for(unsigned k=0;k<24;k++){
            const uint8_t* expected_pack=standalone_contract_n01_packed+k*4096u;
            // In the twelve-output schedule the next 4 KiB is n11 C.
            // Its contents change after each K, so guard this pack against
            // the current C bytes rather than the pre-K0 contents.
            if(ffn_standalone_contract_n12)
              memcpy(scratch_right,scratch+4096u,64);
            memset(scratch,0,4096u);
            memcpy(mapped[0]+0x6c0,standalone_pair_pack_code,sizeof standalone_pair_pack_code);
            memcpy(mapped[23]+0x40,&scalar_packet,4);
            memcpy(mapped[22]+0xa8,&pack_x,4);memcpy(mapped[22]+0xac,&pack_y,4);
            memcpy(mapped[25]+0x10,&pack_x,4);memcpy(mapped[25]+0x14,&pack_y,4);
            for(unsigned half=0;half<2;half++){
              unsigned source_n=2u*k+half;
              uint8_t* source=NULL;uint64_t source_va=0;
              if(source_n==0u){source=outputs[0];source_va=output_vas[0];}
              else if(source_n==1u){source=outputs[1];source_va=output_vas[1];}
              else{
                unsigned offset=source_n<=18u?
                  0xf000u+(source_n-2u)*4096u:
                  0x1000u+(source_n-19u)*4096u;
                unsigned allocation=source_n<=18u?2u:17u;
                source=mapped[allocation]+offset;
                source_va=(allocation==2u?0x10000030000ull:
                                           0x10000058000ull)+offset;
              }
              const uint64_t bindings[3]={source_va,scratch_va+64u*half,0};
              unsigned t=span*k+half;
              memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
              if(memcmp(source,standalone_contract_n01_activated+source_n*4096u,4096u) ||
                 memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
                 *full_ready!=0x800000f0u || !no_agx_metal())return 2;
              void (^scheduled)(void)=Block_copy(^{pm[2u*t]=0x2d0000u+2u*t;});
              void (^completed)(void)=Block_copy(^{pm[2u*t+1u]=0x2d0001u+2u*t;});
              if(!scheduled || !completed || scheduled==completed)return 2;
              uint8_t record[64]={0},out[64]={0};
              uint32_t kid=shmem[1].id,sid=shmem[0].id;
              uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
              memcpy(record,&kid,4);memcpy(record+4,&sid,4);
              memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
              int status=Submit(queue,NULL,1,record,64,out);
              unsigned waited=0;for(;waited<3000 && !pm[2u*t+1u];waited++)usleep(10000);
              uint32_t outword=0;memcpy(&outword,out,4);
              unsigned exact=1;
              for(unsigned row=0;row<32;row++){
                exact&=memcmp(scratch+row*128u+half*64u,
                              expected_pack+row*128u+half*64u,64u)==0;
                if(!half)for(unsigned i=0;i<64;i++)
                  exact&=scratch[row*128u+64u+i]==0;
              }
              unsigned guards=memcmp(scratch-64,scratch_left,64)==0 &&
                              memcmp(scratch+4096u,scratch_right,64)==0;
              fprintf(stderr,"PURE STANDALONE CONTRACT %s PACK k=%u half=%u status=%d outword=0x%08x exact=%u source=%u guards=%u marks=%llu/%llu wait=%u/3000.\n",
                      label,k,half,status,outword,exact,
                      memcmp(source,standalone_contract_n01_activated+source_n*4096u,4096u)==0,
                      guards,(unsigned long long)pm[2u*t],
                      (unsigned long long)pm[2u*t+1u],waited);
              if(status || outword || !exact || !guards ||
                 pm[2u*t]!=0x2d0000u+2u*t ||
                 pm[2u*t+1u]!=0x2d0001u+2u*t || !no_agx_metal())return 2;
            }
            fprintf(stderr,"PURE STANDALONE CONTRACT %s PACK k=%u words:",label,k);
            for(unsigned i=0;i<2048;i++)fprintf(stderr," %04x",((uint16_t*)scratch)[i]);
            fputc('\n',stderr);
            if(memcmp(scratch,expected_pack,4096u))return 2;
            if(k==0u)for(unsigned n=0;n<count;n++)
              for(unsigned i=0;i<1024;i++)((uint32_t*)outputs[n])[i]=0x7fc01234u;
            for(unsigned n=0;n<count;n++){
              uint8_t prior[4096],other_copy[12][4096];
              memcpy(prior,outputs[n],4096u);
              for(unsigned j=0;j<count;j++)if(j!=n)
                memcpy(other_copy[j],outputs[j],4096u);
              const uint8_t* expected_b=standalone_contract_n01_b+(n*24u+k)*4096u;
              const double* reference=standalone_contract_n01_reference+
                                      (n*24u+k)*1024u;
              memcpy(weights,expected_b,4096u);
              memcpy(mapped[0]+0x6c0,k?standalone_ffn_k6_accum_code:
                                          standalone_ffn_tile_code,
                     k?sizeof standalone_ffn_k6_accum_code:
                       sizeof standalone_ffn_tile_code);
              memcpy(mapped[23]+0x38,scratch_word,sizeof scratch_word);
              memcpy(mapped[23]+0x40,&tensor_packet,4);
              memcpy(mapped[22]+0xa8,&tensor_x,4);memcpy(mapped[22]+0xac,&tensor_y,4);
              memcpy(mapped[25]+0x10,&tensor_x,4);memcpy(mapped[25]+0x14,&tensor_y,4);
              const uint64_t bindings[4]={scratch_va,weight_va,output_vas[n],0};
              memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
              unsigned t=span*k+2u+n;
              if(memcmp(weights,expected_b,4096u) ||
                 memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
                 *full_ready!=0x800000f0u || !no_agx_metal())return 2;
              void (^scheduled)(void)=Block_copy(^{pm[2u*t]=0x2d0000u+2u*t;});
              void (^completed)(void)=Block_copy(^{pm[2u*t+1u]=0x2d0001u+2u*t;});
              if(!scheduled || !completed || scheduled==completed)return 2;
              uint8_t record[64]={0},out[64]={0};
              uint32_t kid=shmem[1].id,sid=shmem[0].id;
              uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
              memcpy(record,&kid,4);memcpy(record+4,&sid,4);
              memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
              int status=Submit(queue,NULL,1,record,64,out);
              unsigned waited=0;for(;waited<3000 && !pm[2u*t+1u];waited++)usleep(10000);
              uint32_t outword=0;memcpy(&outword,out,4);
              unsigned changed=0,finite=1,within=1;double max_abs=0;
              for(unsigned i=0;i<1024;i++){
                changed+=((uint32_t*)outputs[n])[i]!=((uint32_t*)prior)[i];
                float value=((float*)outputs[n])[i];finite&=isfinite(value);
                double error=fabs((double)value-reference[i]);
                within&=isfinite(value) && error<=2e-5*(1.0+fabs(reference[i]));
                if(error>max_abs)max_abs=error;
              }
              unsigned inputs=memcmp(scratch,expected_pack,4096u)==0 &&
                              memcmp(weights,expected_b,4096u)==0;
              unsigned guards=(ffn_standalone_contract_n8 || ffn_standalone_contract_n12)?
                (memcmp(outputs[0]-64,out_left[0],64)==0 &&
                 memcmp(outputs[0]+4096u,out_right[0],64)==0 &&
                 memcmp(outputs[1]+4096u,out_right[1],64)==0):
                (memcmp(outputs[n]-64,out_left[n],64)==0 &&
                 memcmp(outputs[n]+4096u,out_right[n],64)==0);
              if(ffn_standalone_contract_n12)for(unsigned j=8;j<=10;j++)
                guards&=memcmp(outputs[j]-64,extra_left[j-8u],64)==0 &&
                        memcmp(outputs[j]+4096u,extra_right[j-8u],64)==0;
              unsigned other_ok=1;
              for(unsigned j=0;j<count;j++)if(j!=n)
                other_ok&=memcmp(outputs[j],other_copy[j],4096u)==0;
              unsigned sources=1;
              for(unsigned tile=2;tile<48;tile++){
                uint8_t* source=tile<=18u?
                  mapped[2]+0xf000u+(tile-2u)*4096u:
                  mapped[17]+0x1000u+(tile-19u)*4096u;
                sources&=memcmp(source,standalone_contract_n01_activated+tile*4096u,4096u)==0;
              }
              fprintf(stderr,"PURE STANDALONE CONTRACT %s k=%u n=%u status=%d outword=0x%08x changed=%u/1024 finite=%u within=%u inputs=%u guards=%u other=%u sources=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                      label,k,n,status,outword,changed,finite,within,inputs,guards,
                      other_ok,sources,max_abs,(unsigned long long)pm[2u*t],
                      (unsigned long long)pm[2u*t+1u],waited);
              fprintf(stderr,"PURE STANDALONE CONTRACT %s k=%u n=%u words:",label,k,n);
              for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",((uint32_t*)outputs[n])[i]);
              fputc('\n',stderr);
              if(status || outword || !changed || !finite || !within || !inputs ||
                 !guards || !other_ok || !sources ||
                 pm[2u*t]!=0x2d0000u+2u*t ||
                 pm[2u*t+1u]!=0x2d0001u+2u*t || !no_agx_metal())return 2;
            }
          }
          if(ffn_standalone_contract_n12)
            fprintf(stderr,"PURE STANDALONE CONTRACT N12 COMPLETE: twelve 32-column output tiles accumulated across shared GPU-packed K blocks.\n");
          else if(ffn_standalone_contract_n8)
            fprintf(stderr,"PURE STANDALONE CONTRACT N8 COMPLETE: eight 32-column output tiles accumulated across shared GPU-packed K blocks.\n");
          else
            fprintf(stderr,"PURE STANDALONE CONTRACT N01 COMPLETE: two 32-column output tiles accumulated across shared GPU-packed K blocks.\n");
        }
        if(ffn_standalone_contract_n12_bias){
          uint8_t* outputs[12]={mapped[2]+0x6f80u,mapped[2]+0xe000u};
          uint64_t vas[12]={0x10000036f80ull,0x1000003e000ull};
          for(unsigned n=2;n<8;n++){
            outputs[n]=mapped[2]+0x8000u+(n-2u)*4096u;
            vas[n]=0x10000038000ull+(uint64_t)(n-2u)*4096u;
          }
          outputs[8]=mapped[2]+0x800u;vas[8]=0x10000030800ull;
          outputs[9]=mapped[2]+0x1c00u;vas[9]=0x10000031c00ull;
          outputs[10]=mapped[2]+0x4d80u;vas[10]=0x10000034d80ull;
          outputs[11]=mapped[17]+0x1f000u;vas[11]=0x10000077000ull;
          for(unsigned n=0;n<12;n++)
            if(memcmp(outputs[n],standalone_contract_n12_bias_prior+n*4096u,4096u))return 2;
          const uint32_t packet=0x0e4006c7u,grid_x=1024u,grid_y=1u;
          memcpy(mapped[0]+0x6c0,bias_code,sizeof bias_code);
          memcpy(mapped[23]+0x40,&packet,4);
          memcpy(mapped[22]+0xa8,&grid_x,4);memcpy(mapped[22]+0xac,&grid_y,4);
          memcpy(mapped[25]+0x10,&grid_x,4);memcpy(mapped[25]+0x14,&grid_y,4);
          static volatile uint64_t marks[24]={0};
          volatile uint64_t* bm=marks;
          for(unsigned n=0;n<12;n++){
            uint8_t* tile=outputs[n];
            uint8_t left[64],right[64];
            memcpy(left,tile-64,64);
            unsigned right_valid=n!=11u;
            if(right_valid)memcpy(right,tile+4096u,64);
            uint8_t others[12][4096];
            for(unsigned j=0;j<12;j++)if(j!=n)memcpy(others[j],outputs[j],4096u);
            memcpy(mapped[2]+0x5e80u,standalone_contract_n12_bias_operands+n*4096u,4096u);
            const uint64_t bindings[4]={vas[n],0x10000035e80ull,vas[n],0};
            memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
            if(memcmp(tile,standalone_contract_n12_bias_prior+n*4096u,4096u) ||
               memcmp(mapped[2]+0x5e80u,standalone_contract_n12_bias_operands+n*4096u,4096u) ||
               memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            void (^scheduled)(void)=Block_copy(^{bm[2u*n]=0x2f0000u+2u*n;});
            void (^completed)(void)=Block_copy(^{bm[2u*n+1u]=0x2f0001u+2u*n;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t record[64]={0},out[64]={0};
            uint32_t kid=shmem[1].id,sid=shmem[0].id;
            uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
            memcpy(record,&kid,4);memcpy(record+4,&sid,4);
            memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
            int status=Submit(queue,NULL,1,record,64,out);
            unsigned waited=0;for(;waited<3000 && !bm[2u*n+1u];waited++)usleep(10000);
            uint32_t outword=0;memcpy(&outword,out,4);
            unsigned exact=memcmp(tile,standalone_contract_n12_bias_expected+n*4096u,4096u)==0;
            unsigned input=memcmp(mapped[2]+0x5e80u,standalone_contract_n12_bias_operands+n*4096u,4096u)==0;
            unsigned guards=memcmp(tile-64,left,64)==0 &&
                            (!right_valid || memcmp(tile+4096u,right,64)==0);
            unsigned other=1;
            for(unsigned j=0;j<12;j++)if(j!=n)
              other&=memcmp(outputs[j],others[j],4096u)==0;
            fprintf(stderr,"PURE STANDALONE CONTRACT N12 BIAS n=%u status=%d outword=0x%08x exact=%u input=%u guards=%u other=%u marks=%llu/%llu wait=%u/3000.\n",
                    n,status,outword,exact,input,guards,other,
                    (unsigned long long)bm[2u*n],(unsigned long long)bm[2u*n+1u],waited);
            fprintf(stderr,"PURE STANDALONE CONTRACT N12 BIAS n=%u words:",n);
            for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",((uint32_t*)tile)[i]);
            fputc('\n',stderr);
            if(status || outword || !exact || !input || !guards || !other ||
               bm[2u*n]!=0x2f0000u+2u*n || bm[2u*n+1u]!=0x2f0001u+2u*n ||
               !no_agx_metal())return 2;
          }
          fprintf(stderr,"PURE STANDALONE CONTRACT N12 BIAS COMPLETE: twelve wide adds biased retained GPU C tiles in place.\n");
        }
        if(ffn_standalone_residual_keep){
          unsigned exact=memcmp(mapped[29]+0x1000u,standalone_ffn_pack_source,49152u)==0;
          fprintf(stderr,"PURE STANDALONE RESIDUAL KEEP FINAL exact=%u bytes=49152.\n",exact);
          if(!exact || !no_agx_metal())return 2;
        }
        if(ffn_contract_slot_text){
          uint8_t* candidates[4]={mapped[2]+0x800u,mapped[2]+0x1c00u,
                                  mapped[2]+0x4d80u,mapped[17]+0x1f000u};
          const uint64_t candidate_vas[4]={0x10000030800ull,0x10000031c00ull,
                                            0x10000034d80ull,0x10000077000ull};
          uint8_t* target=candidates[ffn_contract_slot];
          uint8_t* scratch=mapped[17]+0x1e000u;
          uint8_t* weights=mapped[2]+0x5e80u;
          uint8_t* outputs[8]={mapped[2]+0x6f80u,mapped[2]+0xe000u};
          for(unsigned n=2;n<8;n++)outputs[n]=mapped[2]+0x8000u+(n-2u)*4096u;
          uint8_t saved[8][4096],left[64],right[64];
          for(unsigned n=0;n<8;n++)memcpy(saved[n],outputs[n],4096u);
          memcpy(left,target-64,64);
          if(ffn_contract_slot!=3u)memcpy(right,target+4096u,64);
          if(memcmp(scratch,standalone_contract_n01_packed+23u*4096u,4096u))return 2;
          for(unsigned i=0;i<1024;i++)((uint32_t*)target)[i]=0x7fc01234u;
          memcpy(weights,standalone_contract_slot_b,4096u);
          memcpy(mapped[0]+0x6c0,standalone_ffn_tile_code,sizeof standalone_ffn_tile_code);
          const uint32_t scratch_word[2]={0x0c00100fu,0};
          const uint32_t tensor_packet=0x0e5806c7u,tensor_x=32u,tensor_y=1u;
          const uint64_t bindings[4]={0x10000076000ull,0x10000035e80ull,
                                      candidate_vas[ffn_contract_slot],0};
          memcpy(mapped[23]+0x38,scratch_word,sizeof scratch_word);
          memcpy(mapped[23]+0x40,&tensor_packet,4);
          memcpy(mapped[22]+0xa8,&tensor_x,4);memcpy(mapped[22]+0xac,&tensor_y,4);
          memcpy(mapped[25]+0x10,&tensor_x,4);memcpy(mapped[25]+0x14,&tensor_y,4);
          memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
          if(memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
             memcmp(weights,standalone_contract_slot_b,4096u) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t marks[2]={0};
          volatile uint64_t* pm=marks;
          void (^scheduled)(void)=Block_copy(^{pm[0]=0x2e0000u;});
          void (^completed)(void)=Block_copy(^{pm[1]=0x2e0001u;});
          if(!scheduled || !completed || scheduled==completed)return 2;
          uint8_t record[64]={0},out[64]={0};
          uint32_t kid=shmem[1].id,sid=shmem[0].id;
          uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
          memcpy(record,&kid,4);memcpy(record+4,&sid,4);
          memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
          int status=Submit(queue,NULL,1,record,64,out);
          unsigned waited=0;for(;waited<3000 && !pm[1];waited++)usleep(10000);
          uint32_t outword=0;memcpy(&outword,out,4);
          unsigned changed=0,finite=1,within=1;double max_abs=0;
          for(unsigned i=0;i<1024;i++){
            uint32_t word=((uint32_t*)target)[i];
            changed+=word!=0x7fc01234u;
            float value=((float*)target)[i];finite&=isfinite(value);
            double error=fabs((double)value-standalone_contract_slot_reference[i]);
            within&=isfinite(value) &&
              error<=2e-5*(1.0+fabs(standalone_contract_slot_reference[i]));
            if(error>max_abs)max_abs=error;
          }
          unsigned guards=memcmp(target-64,left,64)==0 &&
            (ffn_contract_slot==3u || memcmp(target+4096u,right,64)==0);
          unsigned inputs=memcmp(scratch,standalone_contract_n01_packed+23u*4096u,4096u)==0 &&
                          memcmp(weights,standalone_contract_slot_b,4096u)==0;
          unsigned other=1;
          for(unsigned n=0;n<8;n++)other&=memcmp(outputs[n],saved[n],4096u)==0;
          unsigned sources=1;
          for(unsigned tile=2;tile<48;tile++){
            uint8_t* source=tile<=18u?
              mapped[2]+0xf000u+(tile-2u)*4096u:
              mapped[17]+0x1000u+(tile-19u)*4096u;
            sources&=memcmp(source,standalone_contract_n01_activated+tile*4096u,4096u)==0;
          }
          fprintf(stderr,"PURE STANDALONE CONTRACT SLOT slot=%u va=0x%llx status=%d outword=0x%08x changed=%u/1024 finite=%u within=%u inputs=%u guards=%u other=%u sources=%u max_abs=%.9g marks=%llu/%llu wait=%u/3000.\n",
                  ffn_contract_slot,(unsigned long long)candidate_vas[ffn_contract_slot],
                  status,outword,changed,finite,within,inputs,guards,other,sources,
                  max_abs,(unsigned long long)pm[0],(unsigned long long)pm[1],waited);
          fprintf(stderr,"PURE STANDALONE CONTRACT SLOT slot=%u words:",ffn_contract_slot);
          for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",((uint32_t*)target)[i]);
          fputc('\n',stderr);
          if(status || outword || changed!=1024u || !finite || !within || !inputs ||
             !guards || !other || !sources || pm[0]!=0x2e0000u ||
             pm[1]!=0x2e0001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE STANDALONE CONTRACT SLOT COMPLETE: slot=%u received GPU-produced K23 operand through TensorOps.\n",
                  ffn_contract_slot);
        }
        fprintf(stderr,ffn_standalone_query_all?
                "PURE STANDALONE QUERY COMPLETE: embedding LayerNorm then all query tiles.\n":
                ffn_standalone_embedding_query?
                "PURE STANDALONE QUERY COMPLETE: embedding LayerNorm then query in two pure Submits.\n":
                "PURE STANDALONE QUERY COMPLETE: one pure Submit, no FFN prefix.\n");
        return 0;
      }
      if(ffn_resident_full){
        volatile uint32_t* full_ready=(volatile uint32_t*)(uintptr_t)(shmem[0].cpu+0x24);
        if(*full_ready!=0xf0u || shmem[1].id!=2u || shmem[0].id!=1u)return 2;
        // The initial cross-control canary occupies the first 128 bytes of
        // n=25's slot. It was checked during staging and must be cleared
        // before this slot becomes a destination.
        memset(mapped[17]+0x9000,0,0x80);
        uint32_t* slots[48]={0};
        uint32_t* completed_tiles=calloc(48u*1024u,sizeof(uint32_t));
        if(!completed_tiles)return 2;
        for(unsigned n=0;n<48;n++){
          slots[n]=(uint32_t*)(n<24?mapped[2]+0x8000u+n*4096u:
                                   mapped[17]+0x8000u+(n-24u)*4096u);
          for(unsigned i=0;i<1024;i++)if(slots[n][i]!=0u)return 2;
        }
        static volatile uint64_t full_marks[576]={0};
        volatile uint64_t* mark_ptr=full_marks;
        *full_ready=0x800000f0u;
        fprintf(stderr,"PURE FFN FULL FIRE: 48 slots zero; same queue/pages; ready=0x%08x.\n",*full_ready);
        for(unsigned n=0;n<48;n++){
          uint64_t c_address=(n<24?0x10000038000ull:0x10000060000ull)+
                             (uint64_t)(n%24u)*4096u;
          memcpy(mapped[28]+0x1bb0,&c_address,8);
          if(memcmp(mapped[28]+0x1bb0,&c_address,8))return 2;
          for(unsigned k=0;k<6;k++){
            const uint8_t* a_window=k==0?expected_inputs[1]:
                                    k==1?next_inputs[0]:six_inputs[k-2][0];
            const uint8_t* b_window=full_b+(n*6u+k)*4096u;
            memcpy(mapped[2]+0x4d80,a_window,4096);
            memcpy(mapped[2]+0x5e80,b_window,4096);
            uint32_t prior[1024];memcpy(prior,slots[n],sizeof prior);
            unsigned t=n*6u+k;
            void (^scheduled_full)(void)=Block_copy(^{mark_ptr[2*t]=0x800u+2u*t;});
            void (^completed_full)(void)=Block_copy(^{mark_ptr[2*t+1]=0x801u+2u*t;});
            if(!scheduled_full || !completed_full || scheduled_full==completed_full ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            uint8_t record_full[64]={0},out_full[64]={0};
            uint32_t kernel_id_full=shmem[1].id,segment_id_full=shmem[0].id;
            uintptr_t sp_full=(uintptr_t)scheduled_full,cp_full=(uintptr_t)completed_full;
            memcpy(record_full,&kernel_id_full,4);memcpy(record_full+4,&segment_id_full,4);
            memcpy(record_full+0x10,&sp_full,sizeof sp_full);
            memcpy(record_full+0x18,&cp_full,sizeof cp_full);
            fprintf(stderr,"PURE FFN FULL n=%u k=%u entering: C=0x%llx ready=0x%08x.\n",
                    n,k,(unsigned long long)c_address,*full_ready);
            int status=Submit(queue,NULL,1,record_full,64,out_full);
            unsigned waited_full=0;
            for(;waited_full<3000 && (!mark_ptr[2*t+1] || slots[n][1023]==prior[1023]);waited_full++)usleep(10000);
            unsigned changed_full=0,prior_intact=1,guards_full=1;
            for(unsigned i=0;i<1024;i++)changed_full+=slots[n][i]!=prior[i];
            for(unsigned p=0;p<n;p++)prior_intact&=
              memcmp(slots[p],completed_tiles+p*1024u,4096)==0;
            for(unsigned i=0;i<1024;i++)guards_full&=c_view_words[32+i]==0x7fc01234u;
            for(unsigned i=0;i<0x80;i++){
              guards_full&=mapped[17][0x7f80+i]==0xc7;
              guards_full&=mapped[2][0x4d00+i]==0xa5 && mapped[2][0x5d80+i]==0xa5;
              guards_full&=mapped[2][0x5e00+i]==0xa4 && mapped[2][0x6e80+i]==0xa4;
              guards_full&=mapped[2][0x6f00+i]==0xa7 && mapped[2][0x7f80+i]==0xa7;
            }
            unsigned input_full=memcmp(mapped[2]+0x4d80,a_window,4096)==0 &&
                                memcmp(mapped[2]+0x5e80,b_window,4096)==0;
            uint32_t outword_full=0;memcpy(&outword_full,out_full,4);
            fprintf(stderr,"PURE FFN FULL n=%u k=%u returned status=%d outword=0x%08x changed=%u/1024 inputs=%u guards=%u prior=%u marks=%llu/%llu wait=%u/3000.\n",
                    n,k,status,outword_full,changed_full,input_full,guards_full,prior_intact,
                    (unsigned long long)mark_ptr[2*t],(unsigned long long)mark_ptr[2*t+1],waited_full);
            fprintf(stderr,"PURE FFN FULL n=%u k=%u C words:",n,k);
            for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",slots[n][i]);
            fputc('\n',stderr);
            if(status || outword_full || changed_full!=1024u || !input_full || !guards_full ||
               !prior_intact || mark_ptr[2*t]!=0x800u+2u*t ||
               mark_ptr[2*t+1]!=0x801u+2u*t || !no_agx_metal())return 2;
          }
          memcpy(completed_tiles+n*1024u,slots[n],4096);
        }
        for(unsigned n=0;n<48;n++)if(memcmp(slots[n],completed_tiles+n*1024u,4096))return 2;
        fprintf(stderr,"PURE FFN FULL COMPLETE: 48 GPU slots retained; 288 Submits.\n");
        if(ffn_full_append_add7){
          uint32_t* target=c_view_words+32;
          for(unsigned i=0;i<1024;i++)if(target[i]!=0x7fc01234u)return 2;
          uint64_t append_a=0x10000038000ull,append_c=0x10000036f80ull;
          memcpy(mapped[28]+0x1ba0,&append_a,8);
          memcpy(mapped[28]+0x1bb0,&append_c,8);
          memcpy(mapped[0]+0x6c0,G17_INPUT_ADD7_CODE,sizeof G17_INPUT_ADD7_CODE);
          uint32_t append_packet=0x0e4006c7u;
          memcpy(mapped[23]+0x40,&append_packet,4);
          if(memcmp(mapped[28]+0x1ba0,&append_a,8) ||
             memcmp(mapped[28]+0x1bb0,&append_c,8) ||
             memcmp(mapped[0]+0x6c0,G17_INPUT_ADD7_CODE,sizeof G17_INPUT_ADD7_CODE) ||
             memcmp(mapped[23]+0x40,&append_packet,4) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t append_marks[2]={0};
          volatile uint64_t* append_mark_ptr=append_marks;
          void (^scheduled_append)(void)=Block_copy(^{append_mark_ptr[0]=0x999u;});
          void (^completed_append)(void)=Block_copy(^{append_mark_ptr[1]=0x99au;});
          if(!scheduled_append || !completed_append || scheduled_append==completed_append)return 2;
          uint8_t record_append[64]={0},out_append[64]={0};
          uint32_t kernel_id_append=shmem[1].id,segment_id_append=shmem[0].id;
          uintptr_t sp_append=(uintptr_t)scheduled_append,cp_append=(uintptr_t)completed_append;
          memcpy(record_append,&kernel_id_append,4);
          memcpy(record_append+4,&segment_id_append,4);
          memcpy(record_append+0x10,&sp_append,sizeof sp_append);
          memcpy(record_append+0x18,&cp_append,sizeof cp_append);
          fprintf(stderr,"PURE FFN APPEND entering: code=0x100000006c0 packet=0x%08x A=0x%llx C=0x%llx ready=0x%08x.\n",
                  append_packet,(unsigned long long)append_a,(unsigned long long)append_c,*full_ready);
          int append_status=Submit(queue,NULL,1,record_append,64,out_append);
          unsigned append_wait=0;
          for(;append_wait<3000 && (!append_mark_ptr[1] || target[31]==0x7fc01234u);append_wait++)usleep(10000);
          unsigned exact_append=0,sentinel_append=0,resident_intact=1;
          for(unsigned i=0;i<32;i++)exact_append+=target[i]==slots[0][i]+7u;
          for(unsigned i=32;i<1024;i++)sentinel_append+=target[i]==0x7fc01234u;
          for(unsigned n=0;n<48;n++)resident_intact&=
            memcmp(slots[n],completed_tiles+n*1024u,4096)==0;
          uint32_t outword_append=0;memcpy(&outword_append,out_append,4);
          fprintf(stderr,"PURE FFN APPEND returned status=%d outword=0x%08x exact=%u/32 tail=%u/992 resident=%u marks=%llu/%llu wait=%u/3000.\n",
                  append_status,outword_append,exact_append,sentinel_append,resident_intact,
                  (unsigned long long)append_mark_ptr[0],(unsigned long long)append_mark_ptr[1],append_wait);
          fprintf(stderr,"PURE FFN APPEND C words:");
          for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",target[i]);
          fputc('\n',stderr);
          if(append_status || outword_append || exact_append!=32u || sentinel_append!=992u ||
             !resident_intact || append_mark_ptr[0]!=0x999u ||
             append_mark_ptr[1]!=0x99au || !no_agx_metal())return 2;
        }
        if(ffn_full_append_gelu){
          uint32_t* target=c_view_words+32;
          for(unsigned i=0;i<1024;i++)if(target[i]!=0x7fc01234u)return 2;
          uint64_t gelu_a=0x10000038000ull,gelu_b=0x10000036f80ull;
          memcpy(mapped[28]+0x1ba0,&gelu_a,8);
          memcpy(mapped[28]+0x1ba8,&gelu_b,8);
          memcpy(mapped[0]+0x6c0,gelu_code,sizeof gelu_code);
          uint32_t gelu_packet=0x0e4006c7u;
          memcpy(mapped[23]+0x40,&gelu_packet,4);
          if(memcmp(mapped[28]+0x1ba0,&gelu_a,8) ||
             memcmp(mapped[28]+0x1ba8,&gelu_b,8) ||
             memcmp(mapped[0]+0x6c0,gelu_code,sizeof gelu_code) ||
             memcmp(mapped[23]+0x40,&gelu_packet,4) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t gelu_marks[2]={0};
          volatile uint64_t* gelu_mark_ptr=gelu_marks;
          void (^scheduled_gelu)(void)=Block_copy(^{gelu_mark_ptr[0]=0x99bu;});
          void (^completed_gelu)(void)=Block_copy(^{gelu_mark_ptr[1]=0x99cu;});
          if(!scheduled_gelu || !completed_gelu || scheduled_gelu==completed_gelu)return 2;
          uint8_t record_gelu[64]={0},out_gelu[64]={0};
          uint32_t kernel_id_gelu=shmem[1].id,segment_id_gelu=shmem[0].id;
          uintptr_t sp_gelu=(uintptr_t)scheduled_gelu,cp_gelu=(uintptr_t)completed_gelu;
          memcpy(record_gelu,&kernel_id_gelu,4);
          memcpy(record_gelu+4,&segment_id_gelu,4);
          memcpy(record_gelu+0x10,&sp_gelu,sizeof sp_gelu);
          memcpy(record_gelu+0x18,&cp_gelu,sizeof cp_gelu);
          fprintf(stderr,"PURE FFN GELU entering: code=0x100000006c0 packet=0x%08x A=0x%llx B=0x%llx ready=0x%08x.\n",
                  gelu_packet,(unsigned long long)gelu_a,(unsigned long long)gelu_b,*full_ready);
          int gelu_status=Submit(queue,NULL,1,record_gelu,64,out_gelu);
          unsigned gelu_wait=0;
          for(;gelu_wait<3000 && (!gelu_mark_ptr[1] || target[31]==0x7fc01234u);gelu_wait++)usleep(10000);
          unsigned changed_gelu=0,sentinel_gelu=0,resident_intact=1,finite_gelu=1;
          for(unsigned i=0;i<32;i++){
            changed_gelu+=target[i]!=0x7fc01234u;
            float value=0;memcpy(&value,target+i,4);finite_gelu&=isfinite(value);
          }
          for(unsigned i=32;i<1024;i++)sentinel_gelu+=target[i]==0x7fc01234u;
          for(unsigned n=0;n<48;n++)resident_intact&=
            memcmp(slots[n],completed_tiles+n*1024u,4096)==0;
          uint32_t outword_gelu=0;memcpy(&outword_gelu,out_gelu,4);
          fprintf(stderr,"PURE FFN GELU returned status=%d outword=0x%08x changed=%u/32 tail=%u/992 finite=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                  gelu_status,outword_gelu,changed_gelu,sentinel_gelu,finite_gelu,resident_intact,
                  (unsigned long long)gelu_mark_ptr[0],(unsigned long long)gelu_mark_ptr[1],gelu_wait);
          fprintf(stderr,"PURE FFN GELU C words:");
          for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",target[i]);
          fputc('\n',stderr);
          if(gelu_status || outword_gelu || changed_gelu!=32u || sentinel_gelu!=992u ||
             !finite_gelu || !resident_intact || gelu_mark_ptr[0]!=0x99bu ||
             gelu_mark_ptr[1]!=0x99cu || !no_agx_metal())return 2;
        }
        if(ffn_full_append_bias || ffn_full_bias_gelu){
          uint32_t* target=c_view_words+32;
          for(unsigned i=0;i<1024;i++)if(target[i]!=0x7fc01234u)return 2;
          uint64_t bias_a=0x10000038000ull,bias_b=0x10000035e80ull,
                   bias_c=0x10000036f80ull;
          memcpy(mapped[28]+0x1ba0,&bias_a,8);
          memcpy(mapped[28]+0x1ba8,&bias_b,8);
          memcpy(mapped[28]+0x1bb0,&bias_c,8);
          memcpy(mapped[2]+0x5e80,bias_values,sizeof bias_values);
          memcpy(mapped[0]+0x6c0,bias_code,sizeof bias_code);
          uint32_t bias_packet=0x0e4006c7u;
          memcpy(mapped[23]+0x40,&bias_packet,4);
          if(memcmp(mapped[28]+0x1ba0,&bias_a,8) ||
             memcmp(mapped[28]+0x1ba8,&bias_b,8) ||
             memcmp(mapped[28]+0x1bb0,&bias_c,8) ||
             memcmp(mapped[2]+0x5e80,bias_values,sizeof bias_values) ||
             memcmp(mapped[0]+0x6c0,bias_code,sizeof bias_code) ||
             memcmp(mapped[23]+0x40,&bias_packet,4) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t bias_marks[2]={0};
          volatile uint64_t* bias_mark_ptr=bias_marks;
          void (^scheduled_bias)(void)=Block_copy(^{bias_mark_ptr[0]=0x99du;});
          void (^completed_bias)(void)=Block_copy(^{bias_mark_ptr[1]=0x99eu;});
          if(!scheduled_bias || !completed_bias || scheduled_bias==completed_bias)return 2;
          uint8_t record_bias[64]={0},out_bias[64]={0};
          uint32_t kernel_id_bias=shmem[1].id,segment_id_bias=shmem[0].id;
          uintptr_t sp_bias=(uintptr_t)scheduled_bias,cp_bias=(uintptr_t)completed_bias;
          memcpy(record_bias,&kernel_id_bias,4);
          memcpy(record_bias+4,&segment_id_bias,4);
          memcpy(record_bias+0x10,&sp_bias,sizeof sp_bias);
          memcpy(record_bias+0x18,&cp_bias,sizeof cp_bias);
          fprintf(stderr,"PURE FFN BIAS entering: code=0x100000006c0 packet=0x%08x A=0x%llx B=0x%llx C=0x%llx ready=0x%08x.\n",
                  bias_packet,(unsigned long long)bias_a,(unsigned long long)bias_b,
                  (unsigned long long)bias_c,*full_ready);
          int bias_status=Submit(queue,NULL,1,record_bias,64,out_bias);
          unsigned bias_wait=0;
          for(;bias_wait<3000 && (!bias_mark_ptr[1] || target[31]==0x7fc01234u);bias_wait++)usleep(10000);
          unsigned changed_bias=0,sentinel_bias=0,resident_intact=1;
          for(unsigned i=0;i<32;i++)changed_bias+=target[i]!=0x7fc01234u;
          for(unsigned i=32;i<1024;i++)sentinel_bias+=target[i]==0x7fc01234u;
          for(unsigned n=0;n<48;n++)resident_intact&=
            memcmp(slots[n],completed_tiles+n*1024u,4096)==0;
          unsigned bias_intact=memcmp(mapped[2]+0x5e80,bias_values,sizeof bias_values)==0;
          uint32_t outword_bias=0;memcpy(&outword_bias,out_bias,4);
          fprintf(stderr,"PURE FFN BIAS returned status=%d outword=0x%08x changed=%u/32 tail=%u/992 input=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                  bias_status,outword_bias,changed_bias,sentinel_bias,bias_intact,resident_intact,
                  (unsigned long long)bias_mark_ptr[0],(unsigned long long)bias_mark_ptr[1],bias_wait);
          fprintf(stderr,"PURE FFN BIAS C words:");
          for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",target[i]);
          fputc('\n',stderr);
          if(bias_status || outword_bias || changed_bias!=32u || sentinel_bias!=992u ||
             !bias_intact || !resident_intact || bias_mark_ptr[0]!=0x99du ||
             bias_mark_ptr[1]!=0x99eu || !no_agx_metal())return 2;
        }
        if(ffn_full_bias_gelu){
          uint32_t* source=(uint32_t*)(mapped[2]+0x6f80);
          uint32_t* target=(uint32_t*)(mapped[2]+0x4d80);
          uint32_t biased_words[32];memcpy(biased_words,source,sizeof biased_words);
          for(unsigned i=0;i<1024;i++)target[i]=0x7fc01234u;
          uint64_t gelu_a=0x10000036f80ull,gelu_b=0x10000034d80ull;
          memcpy(mapped[28]+0x1ba0,&gelu_a,8);
          memcpy(mapped[28]+0x1ba8,&gelu_b,8);
          memcpy(mapped[0]+0x6c0,gelu_code,sizeof gelu_code);
          uint32_t gelu_packet=0x0e4006c7u;
          memcpy(mapped[23]+0x40,&gelu_packet,4);
          if(memcmp(mapped[28]+0x1ba0,&gelu_a,8) ||
             memcmp(mapped[28]+0x1ba8,&gelu_b,8) ||
             memcmp(mapped[0]+0x6c0,gelu_code,sizeof gelu_code) ||
             memcmp(mapped[23]+0x40,&gelu_packet,4) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t chain_marks[2]={0};
          volatile uint64_t* chain_mark_ptr=chain_marks;
          void (^scheduled_chain)(void)=Block_copy(^{chain_mark_ptr[0]=0x99fu;});
          void (^completed_chain)(void)=Block_copy(^{chain_mark_ptr[1]=0x9a0u;});
          if(!scheduled_chain || !completed_chain || scheduled_chain==completed_chain)return 2;
          uint8_t record_chain[64]={0},out_chain[64]={0};
          uint32_t kernel_id_chain=shmem[1].id,segment_id_chain=shmem[0].id;
          uintptr_t sp_chain=(uintptr_t)scheduled_chain,cp_chain=(uintptr_t)completed_chain;
          memcpy(record_chain,&kernel_id_chain,4);
          memcpy(record_chain+4,&segment_id_chain,4);
          memcpy(record_chain+0x10,&sp_chain,sizeof sp_chain);
          memcpy(record_chain+0x18,&cp_chain,sizeof cp_chain);
          fprintf(stderr,"PURE FFN CHAIN entering: code=0x100000006c0 packet=0x%08x A=0x%llx B=0x%llx ready=0x%08x.\n",
                  gelu_packet,(unsigned long long)gelu_a,(unsigned long long)gelu_b,*full_ready);
          int chain_status=Submit(queue,NULL,1,record_chain,64,out_chain);
          unsigned chain_wait=0;
          for(;chain_wait<3000 && (!chain_mark_ptr[1] || target[31]==0x7fc01234u);chain_wait++)usleep(10000);
          unsigned changed_chain=0,sentinel_chain=0,resident_intact=1,finite_chain=1;
          for(unsigned i=0;i<32;i++){
            changed_chain+=target[i]!=0x7fc01234u;
            float value=0;memcpy(&value,target+i,4);finite_chain&=isfinite(value);
          }
          for(unsigned i=32;i<1024;i++)sentinel_chain+=target[i]==0x7fc01234u;
          for(unsigned n=0;n<48;n++)resident_intact&=
            memcmp(slots[n],completed_tiles+n*1024u,4096)==0;
          unsigned biased_intact=memcmp(source,biased_words,sizeof biased_words)==0;
          uint32_t outword_chain=0;memcpy(&outword_chain,out_chain,4);
          fprintf(stderr,"PURE FFN CHAIN returned status=%d outword=0x%08x changed=%u/32 tail=%u/992 finite=%u biased=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                  chain_status,outword_chain,changed_chain,sentinel_chain,finite_chain,
                  biased_intact,resident_intact,(unsigned long long)chain_mark_ptr[0],
                  (unsigned long long)chain_mark_ptr[1],chain_wait);
          fprintf(stderr,"PURE FFN CHAIN C words:");
          for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",target[i]);
          fputc('\n',stderr);
          if(chain_status || outword_chain || changed_chain!=32u || sentinel_chain!=992u ||
             !finite_chain || !biased_intact || !resident_intact ||
             chain_mark_ptr[0]!=0x99fu || chain_mark_ptr[1]!=0x9a0u || !no_agx_metal())return 2;
        }
        if(ffn_full_inplace){
          uint32_t* tile=slots[0];
          uint32_t original_first[32],biased_first[32];
          memcpy(original_first,tile,sizeof original_first);
          uint64_t tile_va=0x10000038000ull,bias_va=0x10000035e80ull;
          memcpy(mapped[28]+0x1ba0,&tile_va,8);
          memcpy(mapped[28]+0x1ba8,&bias_va,8);
          memcpy(mapped[28]+0x1bb0,&tile_va,8);
          memcpy(mapped[2]+0x5e80,bias_values,sizeof bias_values);
          memcpy(mapped[0]+0x6c0,bias_code,sizeof bias_code);
          uint32_t scalar_packet=0x0e4006c7u;
          memcpy(mapped[23]+0x40,&scalar_packet,4);
          if(memcmp(mapped[28]+0x1ba0,&tile_va,8) ||
             memcmp(mapped[28]+0x1ba8,&bias_va,8) ||
             memcmp(mapped[28]+0x1bb0,&tile_va,8) ||
             memcmp(mapped[2]+0x5e80,bias_values,sizeof bias_values) ||
             memcmp(mapped[0]+0x6c0,bias_code,sizeof bias_code) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t inplace_marks[4]={0};
          volatile uint64_t* inmark=inplace_marks;
          void (^scheduled_bias)(void)=Block_copy(^{inmark[0]=0x9a1u;});
          void (^completed_bias)(void)=Block_copy(^{inmark[1]=0x9a2u;});
          if(!scheduled_bias || !completed_bias || scheduled_bias==completed_bias)return 2;
          uint8_t bias_record[64]={0},bias_out[64]={0};
          uint32_t kernel_id=shmem[1].id,segment_id=shmem[0].id;
          uintptr_t bp=(uintptr_t)scheduled_bias,bq=(uintptr_t)completed_bias;
          memcpy(bias_record,&kernel_id,4);memcpy(bias_record+4,&segment_id,4);
          memcpy(bias_record+0x10,&bp,sizeof bp);memcpy(bias_record+0x18,&bq,sizeof bq);
          fprintf(stderr,"PURE FFN INPLACE BIAS entering: A=C=0x%llx B=0x%llx packet=0x%08x ready=0x%08x.\n",
                  (unsigned long long)tile_va,(unsigned long long)bias_va,scalar_packet,*full_ready);
          int bias_status=Submit(queue,NULL,1,bias_record,64,bias_out);
          unsigned wait_bias=0;
          for(;wait_bias<3000 && (!inmark[1] || tile[31]==original_first[31]);wait_bias++)usleep(10000);
          unsigned changed_bias=0,tail_bias=1,others_bias=1;
          for(unsigned i=0;i<32;i++)changed_bias+=tile[i]!=original_first[i];
          tail_bias=memcmp(tile+32,completed_tiles+32,992u*4u)==0;
          for(unsigned n=1;n<48;n++)others_bias&=
            memcmp(slots[n],completed_tiles+n*1024u,4096)==0;
          uint32_t outword_bias=0;memcpy(&outword_bias,bias_out,4);
          fprintf(stderr,"PURE FFN INPLACE BIAS returned status=%d outword=0x%08x changed=%u/32 tail=%u others=%u marks=%llu/%llu wait=%u/3000.\n",
                  bias_status,outword_bias,changed_bias,tail_bias,others_bias,
                  (unsigned long long)inmark[0],(unsigned long long)inmark[1],wait_bias);
          fprintf(stderr,"PURE FFN INPLACE BIAS C words:");
          for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",tile[i]);
          fputc('\n',stderr);
          if(bias_status || outword_bias || changed_bias!=32u || !tail_bias || !others_bias ||
             inmark[0]!=0x9a1u || inmark[1]!=0x9a2u || !no_agx_metal())return 2;
          memcpy(biased_first,tile,sizeof biased_first);
          memcpy(mapped[28]+0x1ba0,&tile_va,8);
          memcpy(mapped[28]+0x1ba8,&tile_va,8);
          memcpy(mapped[0]+0x6c0,gelu_code,sizeof gelu_code);
          if(memcmp(mapped[28]+0x1ba8,&tile_va,8) ||
             memcmp(mapped[0]+0x6c0,gelu_code,sizeof gelu_code) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          void (^scheduled_gelu)(void)=Block_copy(^{inmark[2]=0x9a3u;});
          void (^completed_gelu)(void)=Block_copy(^{inmark[3]=0x9a4u;});
          if(!scheduled_gelu || !completed_gelu || scheduled_gelu==completed_gelu)return 2;
          uint8_t gelu_record[64]={0},gelu_out[64]={0};
          uintptr_t gp=(uintptr_t)scheduled_gelu,gq=(uintptr_t)completed_gelu;
          memcpy(gelu_record,&kernel_id,4);memcpy(gelu_record+4,&segment_id,4);
          memcpy(gelu_record+0x10,&gp,sizeof gp);memcpy(gelu_record+0x18,&gq,sizeof gq);
          fprintf(stderr,"PURE FFN INPLACE GELU entering: A=B=0x%llx packet=0x%08x ready=0x%08x.\n",
                  (unsigned long long)tile_va,scalar_packet,*full_ready);
          int gelu_status=Submit(queue,NULL,1,gelu_record,64,gelu_out);
          unsigned wait_gelu=0;
          for(;wait_gelu<3000 && (!inmark[3] || tile[31]==biased_first[31]);wait_gelu++)usleep(10000);
          unsigned changed_gelu=0,finite_gelu=1,others_gelu=1;
          for(unsigned i=0;i<32;i++){
            changed_gelu+=tile[i]!=biased_first[i];
            float value=0;memcpy(&value,tile+i,4);finite_gelu&=isfinite(value);
          }
          unsigned tail_gelu=memcmp(tile+32,completed_tiles+32,992u*4u)==0;
          for(unsigned n=1;n<48;n++)others_gelu&=
            memcmp(slots[n],completed_tiles+n*1024u,4096)==0;
          uint32_t outword_gelu=0;memcpy(&outword_gelu,gelu_out,4);
          fprintf(stderr,"PURE FFN INPLACE GELU returned status=%d outword=0x%08x changed=%u/32 finite=%u tail=%u others=%u marks=%llu/%llu wait=%u/3000.\n",
                  gelu_status,outword_gelu,changed_gelu,finite_gelu,tail_gelu,others_gelu,
                  (unsigned long long)inmark[2],(unsigned long long)inmark[3],wait_gelu);
          fprintf(stderr,"PURE FFN INPLACE GELU C words:");
          for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",tile[i]);
          fputc('\n',stderr);
          if(gelu_status || outword_gelu || changed_gelu!=32u || !finite_gelu ||
             !tail_gelu || !others_gelu || inmark[2]!=0x9a3u ||
             inmark[3]!=0x9a4u || !no_agx_metal())return 2;
        }
        if(ffn_full_activation){
          static volatile uint64_t activation_marks[6144]={0};
          volatile uint64_t* amark=activation_marks;
          const uint32_t scalar_packet=0x0e4006c7u;
          memcpy(mapped[23]+0x40,&scalar_packet,4);
          memcpy(mapped[0]+0x6c0,bias_code,sizeof bias_code);
          if(memcmp(mapped[23]+0x40,&scalar_packet,4) ||
             memcmp(mapped[0]+0x6c0,bias_code,sizeof bias_code))return 2;
          for(unsigned phase=0;phase<2;phase++){
            if(phase){
              memcpy(mapped[0]+0x6c0,gelu_code,sizeof gelu_code);
              if(memcmp(mapped[0]+0x6c0,gelu_code,sizeof gelu_code))return 2;
            }
            for(unsigned n=0;n<48;n++){
              if(!phase){
                memcpy(mapped[2]+0x5e80,full_bias+n*128u,128);
                if(memcmp(mapped[2]+0x5e80,full_bias+n*128u,128))return 2;
              }
              for(unsigned row=0;row<32;row++){
                uint32_t* lane=slots[n]+row*32u;
                uint32_t prior_lane[32];memcpy(prior_lane,lane,sizeof prior_lane);
                uint64_t target_va=(n<24?0x10000038000ull:0x10000060000ull)+
                                   (uint64_t)(n%24u)*4096u+(uint64_t)row*128u;
                uint64_t second_va=phase?target_va:0x10000035e80ull;
                memcpy(mapped[28]+0x1ba0,&target_va,8);
                memcpy(mapped[28]+0x1ba8,&second_va,8);
                if(!phase)memcpy(mapped[28]+0x1bb0,&target_va,8);
                if(memcmp(mapped[28]+0x1ba0,&target_va,8) ||
                   memcmp(mapped[28]+0x1ba8,&second_va,8) ||
                   (!phase && memcmp(mapped[28]+0x1bb0,&target_va,8)) ||
                   *full_ready!=0x800000f0u || !no_agx_metal())return 2;
                unsigned t=phase*1536u+n*32u+row;
                void (^scheduled_act)(void)=Block_copy(^{amark[2*t]=0x10000u+2u*t;});
                void (^completed_act)(void)=Block_copy(^{amark[2*t+1]=0x10001u+2u*t;});
                if(!scheduled_act || !completed_act || scheduled_act==completed_act)return 2;
                uint8_t record_act[64]={0},out_act[64]={0};
                uint32_t kid=shmem[1].id,sid=shmem[0].id;
                uintptr_t sp=(uintptr_t)scheduled_act,cp=(uintptr_t)completed_act;
                memcpy(record_act,&kid,4);memcpy(record_act+4,&sid,4);
                memcpy(record_act+0x10,&sp,sizeof sp);memcpy(record_act+0x18,&cp,sizeof cp);
                fprintf(stderr,"PURE FFN ACT phase=%u n=%u row=%u entering: A=0x%llx B=0x%llx ready=0x%08x.\n",
                        phase,n,row,(unsigned long long)target_va,
                        (unsigned long long)second_va,*full_ready);
                int status=Submit(queue,NULL,1,record_act,64,out_act);
                unsigned waited=0;
                for(;waited<3000 && !amark[2*t+1];waited++)usleep(10000);
                unsigned changed=0,finite=1;
                for(unsigned i=0;i<32;i++){
                  changed+=lane[i]!=prior_lane[i];
                  float value=0;memcpy(&value,lane+i,4);finite&=isfinite(value);
                }
                memcpy(completed_tiles+n*1024u+row*32u,lane,128);
                unsigned resident_intact=1;
                for(unsigned p=0;p<48;p++)resident_intact&=
                  memcmp(slots[p],completed_tiles+p*1024u,4096)==0;
                unsigned bias_intact=phase?1u:
                  memcmp(mapped[2]+0x5e80,full_bias+n*128u,128)==0;
                uint32_t outword=0;memcpy(&outword,out_act,4);
                fprintf(stderr,"PURE FFN ACT phase=%u n=%u row=%u returned status=%d outword=0x%08x changed=%u/32 finite=%u bias=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                        phase,n,row,status,outword,changed,finite,bias_intact,resident_intact,
                        (unsigned long long)amark[2*t],(unsigned long long)amark[2*t+1],waited);
                fprintf(stderr,"PURE FFN ACT phase=%u n=%u row=%u words:",phase,n,row);
                for(unsigned i=0;i<32;i++)fprintf(stderr," %08x",lane[i]);
                fputc('\n',stderr);
                if(status || outword || !changed || !finite || !bias_intact ||
                   !resident_intact || amark[2*t]!=0x10000u+2u*t ||
                   amark[2*t+1]!=0x10001u+2u*t || !no_agx_metal())return 2;
              }
            }
            fprintf(stderr,"PURE FFN ACT PHASE %u COMPLETE: 1536 rows retained.\n",phase);
          }
          for(unsigned p=0;p<48;p++)if(memcmp(slots[p],completed_tiles+p*1024u,4096))return 2;
          fprintf(stderr,"PURE FFN FULL ACTIVATION COMPLETE: 49152 GPU words resident; 3360 Submits.\n");
          free(full_bias);
        }
        if(ffn_pack32){
          uint8_t* target=mapped[17]+0x1000u;
          uint8_t left[64],right[64];
          memcpy(left,target-64,64);memcpy(right,target+64,64);
          uint32_t pack_packet=0x0e4006c7u;
          uint64_t source_va=0x10000038000ull,target_va=0x10000059000ull;
          memcpy(mapped[0]+0x6c0,pack_code,sizeof pack_code);
          memcpy(mapped[23]+0x40,&pack_packet,4);
          memcpy(mapped[28]+0x1ba0,&source_va,8);
          memcpy(mapped[28]+0x1ba8,&target_va,8);
          if(memcmp(mapped[0]+0x6c0,pack_code,sizeof pack_code) ||
             memcmp(mapped[28]+0x1ba0,&source_va,8) ||
             memcmp(mapped[28]+0x1ba8,&target_va,8) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t pack_marks[2]={0};
          volatile uint64_t* pm=pack_marks;
          void (^pack_scheduled)(void)=Block_copy(^{pm[0]=0x18000u;});
          void (^pack_completed)(void)=Block_copy(^{pm[1]=0x18001u;});
          if(!pack_scheduled || !pack_completed || pack_scheduled==pack_completed)return 2;
          uint8_t record_pack[64]={0},out_pack[64]={0};
          uint32_t kid=shmem[1].id,sid=shmem[0].id;
          uintptr_t sp=(uintptr_t)pack_scheduled,cp=(uintptr_t)pack_completed;
          memcpy(record_pack,&kid,4);memcpy(record_pack+4,&sid,4);
          memcpy(record_pack+0x10,&sp,sizeof sp);memcpy(record_pack+0x18,&cp,sizeof cp);
          fprintf(stderr,"PURE FFN PACK32 entering: code=0x100000006c0 A=0x%llx B=0x%llx packet=0x%08x ready=0x%08x.\n",
                  (unsigned long long)source_va,(unsigned long long)target_va,pack_packet,*full_ready);
          int status=Submit(queue,NULL,1,record_pack,64,out_pack);
          unsigned waited=0;for(;waited<3000 && !pm[1];waited++)usleep(10000);
          unsigned exact=memcmp(target,pack_expected,64)==0;
          unsigned guards=memcmp(target-64,left,64)==0 && memcmp(target+64,right,64)==0;
          unsigned resident=1;for(unsigned p=0;p<48;p++)resident&=
            memcmp(slots[p],completed_tiles+p*1024u,4096)==0;
          uint32_t outword=0;memcpy(&outword,out_pack,4);
          fprintf(stderr,"PURE FFN PACK32 returned status=%d outword=0x%08x exact=%u guards=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                  status,outword,exact,guards,resident,
                  (unsigned long long)pm[0],(unsigned long long)pm[1],waited);
          fprintf(stderr,"PURE FFN PACK32 half words:");
          for(unsigned i=0;i<32;i++){
            uint16_t word=0;memcpy(&word,target+2u*i,2);
            fprintf(stderr," %04x",word);
          }
          fputc('\n',stderr);
          if(status || outword || !exact || !guards || !resident ||
             pm[0]!=0x18000u || pm[1]!=0x18001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE FFN PACK32 COMPLETE: 32 GPU-produced activation values converted to FP16.\n");
        }
        if(ffn_pack_full){
          uint8_t* scratch=mapped[17]+0x1000u;
          uint8_t zero_block[4096]={0};
          uint8_t left[64],right[64];
          memcpy(left,scratch-64,64);memcpy(right,scratch+4096,64);
          uint32_t pack_packet=0x0e4006c7u;
          uint64_t scratch_va=0x10000059000ull;
          memcpy(mapped[0]+0x6c0,pack_code,sizeof pack_code);
          memcpy(mapped[23]+0x40,&pack_packet,4);
          if(memcmp(mapped[0]+0x6c0,pack_code,sizeof pack_code) || !no_agx_metal())return 2;
          static volatile uint64_t pack_marks[3072]={0};
          volatile uint64_t* pm=pack_marks;
          static volatile uint64_t reduce_marks[48]={0};
          volatile uint64_t* rm=reduce_marks;
          static volatile uint64_t persist_marks[1536]={0};
          volatile uint64_t* xm=persist_marks;
          uint32_t* reduce_output=(uint32_t*)(uintptr_t)(mapped[17]+0x3000u);
          uint8_t reduce_left[64],reduce_right[64];
          if(ffn_contract_reduce || ffn_consume_persistent){
            memcpy(reduce_left,(uint8_t*)reduce_output-64,64);
            memcpy(reduce_right,(uint8_t*)reduce_output+4096,64);
          }
          for(unsigned k=0;k<24;k++){
            memcpy(scratch,zero_block,4096);
            for(unsigned row=0;row<32;row++)for(unsigned segment=0;segment<2;segment++){
              unsigned n=2u*k+segment,t=k*64u+row*2u+segment;
              uint64_t source_va=(n<24?0x10000038000ull:0x10000060000ull)+
                                 (uint64_t)(n%24u)*4096u+(uint64_t)row*128u;
              uint64_t target_va=scratch_va+(uint64_t)row*128u+(uint64_t)segment*64u;
              memcpy(mapped[28]+0x1ba0,&source_va,8);
              memcpy(mapped[28]+0x1ba8,&target_va,8);
              if(memcmp(mapped[28]+0x1ba0,&source_va,8) ||
                 memcmp(mapped[28]+0x1ba8,&target_va,8) ||
                 *full_ready!=0x800000f0u || !no_agx_metal())return 2;
              void (^scheduled_pack)(void)=Block_copy(^{pm[2*t]=0x20000u+2u*t;});
              void (^completed_pack)(void)=Block_copy(^{pm[2*t+1]=0x20001u+2u*t;});
              if(!scheduled_pack || !completed_pack || scheduled_pack==completed_pack)return 2;
              uint8_t record_pack[64]={0},out_pack[64]={0};
              uint32_t kid=shmem[1].id,sid=shmem[0].id;
              uintptr_t sp=(uintptr_t)scheduled_pack,cp=(uintptr_t)completed_pack;
              memcpy(record_pack,&kid,4);memcpy(record_pack+4,&sid,4);
              memcpy(record_pack+0x10,&sp,sizeof sp);memcpy(record_pack+0x18,&cp,sizeof cp);
              int status=Submit(queue,NULL,1,record_pack,64,out_pack);
              unsigned waited=0;for(;waited<3000 && !pm[2*t+1];waited++)usleep(10000);
              unsigned prefix=row*128u+(segment+1u)*64u;
              unsigned exact=memcmp(scratch,pack_full_expected+k*4096u,prefix)==0 &&
                             memcmp(scratch+prefix,zero_block,4096u-prefix)==0;
              unsigned guards=memcmp(scratch-64,left,64)==0 &&
                              memcmp(scratch+4096,right,64)==0;
              unsigned resident=1;for(unsigned p=0;p<48;p++)resident&=
                memcmp(slots[p],completed_tiles+p*1024u,4096)==0;
              uint32_t outword=0;memcpy(&outword,out_pack,4);
              fprintf(stderr,"PURE FFN PACKFULL k=%u row=%u segment=%u returned status=%d outword=0x%08x exact=%u guards=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                      k,row,segment,status,outword,exact,guards,resident,
                      (unsigned long long)pm[2*t],(unsigned long long)pm[2*t+1],waited);
              if(status || outword || !exact || !guards || !resident ||
                 pm[2*t]!=0x20000u+2u*t || pm[2*t+1]!=0x20001u+2u*t ||
                 !no_agx_metal())return 2;
            }
            fprintf(stderr,"PURE FFN PACKFULL k=%u half words:",k);
            for(unsigned i=0;i<2048;i++){
              uint16_t word=0;memcpy(&word,scratch+2u*i,2);
              fprintf(stderr," %04x",word);
            }
            fputc('\n',stderr);
            if(ffn_pack_persistent){
              uint8_t* target=(uint8_t*)slots[2u*k];
              uint8_t old_target[4096];memcpy(old_target,target,4096);
              uint8_t* zero_b=mapped[17]+0x2000u;
              uint8_t left[64],right[64];
              memcpy(left,target-64,64);memcpy(right,target+4096,64);
              uint32_t scalar_packet=0x0e4006c7u;
              uint64_t base_va=(k<12u?0x10000038000ull:0x10000060000ull)+
                               (uint64_t)(k%12u)*8192u;
              memcpy(mapped[0]+0x6c0,copy_code,sizeof copy_code);
              memcpy(mapped[23]+0x40,&scalar_packet,4);
              if(memcmp(mapped[0]+0x6c0,copy_code,sizeof copy_code))return 2;
              for(unsigned row=0;row<32;row++){
                unsigned t=k*32u+row;
                uint64_t a_va=0x10000059000ull+(uint64_t)row*128u;
                uint64_t b_va=0x1000005a000ull+(uint64_t)row*128u;
                uint64_t c_va=base_va+(uint64_t)row*128u;
                memcpy(mapped[28]+0x1ba0,&a_va,8);
                memcpy(mapped[28]+0x1ba8,&b_va,8);
                memcpy(mapped[28]+0x1bb0,&c_va,8);
                if(memcmp(mapped[28]+0x1ba0,&a_va,8) ||
                   memcmp(mapped[28]+0x1ba8,&b_va,8) ||
                   memcmp(mapped[28]+0x1bb0,&c_va,8) ||
                   *full_ready!=0x800000f0u || !no_agx_metal())return 2;
                void (^scheduled_persist)(void)=Block_copy(^{xm[2*t]=0x60000u+2u*t;});
                void (^completed_persist)(void)=Block_copy(^{xm[2*t+1]=0x60001u+2u*t;});
                if(!scheduled_persist || !completed_persist ||
                   scheduled_persist==completed_persist)return 2;
                uint8_t record_persist[64]={0},out_persist[64]={0};
                uint32_t kid=shmem[1].id,sid=shmem[0].id;
                uintptr_t sp=(uintptr_t)scheduled_persist,cp=(uintptr_t)completed_persist;
                memcpy(record_persist,&kid,4);memcpy(record_persist+4,&sid,4);
                memcpy(record_persist+0x10,&sp,sizeof sp);memcpy(record_persist+0x18,&cp,sizeof cp);
                int status=Submit(queue,NULL,1,record_persist,64,out_persist);
                unsigned waited=0;for(;waited<3000 && !xm[2*t+1];waited++)usleep(10000);
                unsigned prefix=(row+1u)*128u;
                unsigned exact=memcmp(target,pack_full_expected+k*4096u,prefix)==0 &&
                               memcmp(target+prefix,old_target+prefix,4096u-prefix)==0;
                unsigned inputs=memcmp(scratch,pack_full_expected+k*4096u,4096)==0 &&
                                memcmp(zero_b,zero_block,4096)==0;
                unsigned guards=memcmp(target-64,left,64)==0 &&
                                memcmp(target+4096,right,64)==0;
                unsigned resident=1;for(unsigned p=0;p<48;p++)if(p!=2u*k)
                  resident&=memcmp(slots[p],completed_tiles+p*1024u,4096)==0;
                uint32_t outword=0;memcpy(&outword,out_persist,4);
                fprintf(stderr,"PURE FFN PACK PERSISTENT k=%u row=%u returned status=%d outword=0x%08x exact=%u inputs=%u guards=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                        k,row,status,outword,exact,inputs,guards,resident,
                        (unsigned long long)xm[2*t],(unsigned long long)xm[2*t+1],waited);
                if(status || outword || !exact || !inputs || !guards || !resident ||
                   xm[2*t]!=0x60000u+2u*t || xm[2*t+1]!=0x60001u+2u*t ||
                   !no_agx_metal())return 2;
              }
              fprintf(stderr,"PURE FFN PACK PERSISTENT k=%u half words:",k);
              for(unsigned i=0;i<2048;i++){
                uint16_t word=0;memcpy(&word,target+2u*i,2);
                fprintf(stderr," %04x",word);
              }
              fputc('\n',stderr);
              memcpy(completed_tiles+2u*k*1024u,target,4096);
              if(k<23u){
                memcpy(mapped[0]+0x6c0,pack_code,sizeof pack_code);
                memcpy(mapped[23]+0x40,&pack_packet,4);
                if(memcmp(mapped[0]+0x6c0,pack_code,sizeof pack_code))return 2;
              }
            }
            if(ffn_contract_reduce){
              uint32_t previous_c[1024];memcpy(previous_c,reduce_output,sizeof previous_c);
              uint32_t tensor_packet=0x0e5806c7u;
              uint64_t a_va=0x10000059000ull,b_va=0x10000035e80ull,c_va=0x1000005b000ull;
              memcpy(mapped[2]+0x5e80,contract_b_full+k*4096u,4096);
              memcpy(mapped[0]+0x6c0,expected_inputs[0],1458);
              memcpy(mapped[23]+0x40,&tensor_packet,4);
              memcpy(mapped[28]+0x1ba0,&a_va,8);
              memcpy(mapped[28]+0x1ba8,&b_va,8);
              memcpy(mapped[28]+0x1bb0,&c_va,8);
              if(memcmp(mapped[0]+0x6c0,expected_inputs[0],1458) ||
                 memcmp(mapped[2]+0x5e80,contract_b_full+k*4096u,4096) ||
                 memcmp(mapped[28]+0x1ba0,&a_va,8) ||
                 memcmp(mapped[28]+0x1ba8,&b_va,8) ||
                 memcmp(mapped[28]+0x1bb0,&c_va,8) ||
                 *full_ready!=0x800000f0u || !no_agx_metal())return 2;
              void (^scheduled_reduce)(void)=Block_copy(^{rm[2*k]=0x30000u+2u*k;});
              void (^completed_reduce)(void)=Block_copy(^{rm[2*k+1]=0x30001u+2u*k;});
              if(!scheduled_reduce || !completed_reduce || scheduled_reduce==completed_reduce)return 2;
              uint8_t record_reduce[64]={0},out_reduce[64]={0};
              uint32_t kid=shmem[1].id,sid=shmem[0].id;
              uintptr_t sp=(uintptr_t)scheduled_reduce,cp=(uintptr_t)completed_reduce;
              memcpy(record_reduce,&kid,4);memcpy(record_reduce+4,&sid,4);
              memcpy(record_reduce+0x10,&sp,sizeof sp);memcpy(record_reduce+0x18,&cp,sizeof cp);
              int status=Submit(queue,NULL,1,record_reduce,64,out_reduce);
              unsigned waited=0;for(;waited<3000 && !rm[2*k+1];waited++)usleep(10000);
              unsigned changed=0,finite=1;
              for(unsigned i=0;i<1024;i++){
                changed+=reduce_output[i]!=previous_c[i];
                float value=0;memcpy(&value,reduce_output+i,4);finite&=isfinite(value);
              }
              unsigned guards=memcmp((uint8_t*)reduce_output-64,reduce_left,64)==0 &&
                              memcmp((uint8_t*)reduce_output+4096,reduce_right,64)==0;
              unsigned inputs=memcmp(scratch,pack_full_expected+k*4096u,4096)==0 &&
                              memcmp(mapped[2]+0x5e80,contract_b_full+k*4096u,4096)==0;
              unsigned resident=1;for(unsigned p=0;p<48;p++)resident&=
                memcmp(slots[p],completed_tiles+p*1024u,4096)==0;
              uint32_t outword=0;memcpy(&outword,out_reduce,4);
              fprintf(stderr,"PURE FFN CONTRACT REDUCE k=%u returned status=%d outword=0x%08x changed=%u/1024 finite=%u guards=%u inputs=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                      k,status,outword,changed,finite,guards,inputs,resident,
                      (unsigned long long)rm[2*k],(unsigned long long)rm[2*k+1],waited);
              fprintf(stderr,"PURE FFN CONTRACT REDUCE k=%u C words:",k);
              for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",reduce_output[i]);
              fputc('\n',stderr);
              if(status || outword || !changed || !finite || !guards || !inputs ||
                 !resident || rm[2*k]!=0x30000u+2u*k ||
                 rm[2*k+1]!=0x30001u+2u*k || !no_agx_metal())return 2;
              if(k<23u){
                memcpy(mapped[0]+0x6c0,pack_code,sizeof pack_code);
                memcpy(mapped[23]+0x40,&pack_packet,4);
                if(memcmp(mapped[0]+0x6c0,pack_code,sizeof pack_code))return 2;
              }
            }
          }
          fprintf(stderr,"PURE FFN PACKFULL COMPLETE: 49152 GPU-produced activation values converted into 24 exact FP16 blocks.\n");
          if(ffn_contract_reduce)
            fprintf(stderr,"PURE FFN CONTRACT REDUCE COMPLETE: 24 GPU-packed blocks accumulated into one resident tensor tile.\n");
          if(ffn_pack_persistent){
            for(unsigned k=0;k<24;k++)if(memcmp(slots[2u*k],pack_full_expected+k*4096u,4096))return 2;
            fprintf(stderr,"PURE FFN PACK PERSISTENT COMPLETE: 24 exact packed A blocks retained in consumed activation slots.\n");
          }
          if(ffn_consume_persistent){
            uint32_t tensor_packet=0x0e5806c7u;
            uint64_t b_va=0x10000035e80ull,c_va=0x1000005b000ull;
            memcpy(mapped[0]+0x6c0,expected_inputs[0],1458);
            memcpy(mapped[23]+0x40,&tensor_packet,4);
            if(memcmp(mapped[0]+0x6c0,expected_inputs[0],1458))return 2;
            for(unsigned k=0;k<24;k++){
              uint32_t previous_c[1024];memcpy(previous_c,reduce_output,sizeof previous_c);
              uint64_t a_va=(k<12u?0x10000038000ull:0x10000060000ull)+
                            (uint64_t)(k%12u)*8192u;
              memcpy(mapped[2]+0x5e80,contract_b_full+k*4096u,4096);
              memcpy(mapped[28]+0x1ba0,&a_va,8);
              memcpy(mapped[28]+0x1ba8,&b_va,8);
              memcpy(mapped[28]+0x1bb0,&c_va,8);
              if(memcmp(mapped[2]+0x5e80,contract_b_full+k*4096u,4096) ||
                 memcmp(mapped[28]+0x1ba0,&a_va,8) ||
                 memcmp(mapped[28]+0x1ba8,&b_va,8) ||
                 memcmp(mapped[28]+0x1bb0,&c_va,8) ||
                 *full_ready!=0x800000f0u || !no_agx_metal())return 2;
              void (^scheduled_reduce)(void)=Block_copy(^{rm[2*k]=0x30000u+2u*k;});
              void (^completed_reduce)(void)=Block_copy(^{rm[2*k+1]=0x30001u+2u*k;});
              if(!scheduled_reduce || !completed_reduce || scheduled_reduce==completed_reduce)return 2;
              uint8_t record_reduce[64]={0},out_reduce[64]={0};
              uint32_t kid=shmem[1].id,sid=shmem[0].id;
              uintptr_t sp=(uintptr_t)scheduled_reduce,cp=(uintptr_t)completed_reduce;
              memcpy(record_reduce,&kid,4);memcpy(record_reduce+4,&sid,4);
              memcpy(record_reduce+0x10,&sp,sizeof sp);memcpy(record_reduce+0x18,&cp,sizeof cp);
              int status=Submit(queue,NULL,1,record_reduce,64,out_reduce);
              unsigned waited=0;for(;waited<3000 && !rm[2*k+1];waited++)usleep(10000);
              unsigned changed=0,finite=1;
              for(unsigned i=0;i<1024;i++){
                changed+=reduce_output[i]!=previous_c[i];
                float value=0;memcpy(&value,reduce_output+i,4);finite&=isfinite(value);
              }
              unsigned guards=memcmp((uint8_t*)reduce_output-64,reduce_left,64)==0 &&
                              memcmp((uint8_t*)reduce_output+4096,reduce_right,64)==0;
              unsigned inputs=memcmp(slots[2u*k],pack_full_expected+k*4096u,4096)==0 &&
                              memcmp(mapped[2]+0x5e80,contract_b_full+k*4096u,4096)==0;
              unsigned resident=1;for(unsigned p=0;p<48;p++)resident&=
                memcmp(slots[p],completed_tiles+p*1024u,4096)==0;
              uint32_t outword=0;memcpy(&outword,out_reduce,4);
              fprintf(stderr,"PURE FFN CONTRACT REDUCE k=%u returned status=%d outword=0x%08x changed=%u/1024 finite=%u guards=%u inputs=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                      k,status,outword,changed,finite,guards,inputs,resident,
                      (unsigned long long)rm[2*k],(unsigned long long)rm[2*k+1],waited);
              fprintf(stderr,"PURE FFN CONTRACT REDUCE k=%u C words:",k);
              for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",reduce_output[i]);
              fputc('\n',stderr);
              if(status || outword || !changed || !finite || !guards || !inputs ||
                 !resident || rm[2*k]!=0x30000u+2u*k ||
                 rm[2*k+1]!=0x30001u+2u*k || !no_agx_metal())return 2;
            }
            fprintf(stderr,"PURE FFN CONTRACT REDUCE COMPLETE: 24 GPU-packed blocks accumulated into one resident tensor tile.\n");
            fprintf(stderr,"PURE FFN CONTRACT PERSISTENT COMPLETE: 24 relocated A blocks consumed by one tensor output tile.\n");
          }
          if(ffn_contract_all){
            static volatile uint64_t clear_marks[768]={0},tensor_marks[576]={0};
            volatile uint64_t* zm=clear_marks;
            volatile uint64_t* tm=tensor_marks;
            uint8_t* zero_b=mapped[17]+0x2000u;
            uint32_t scalar_packet=0x0e4006c7u,tensor_packet=0x0e5806c7u;
            for(unsigned n=0;n<12;n++){
              unsigned slot=2u*n+1u;
              uint8_t* target=(uint8_t*)slots[slot];
              uint8_t original[4096];memcpy(original,target,4096);
              uint64_t c_base=0x10000038000ull+(uint64_t)slot*4096u;
              uint8_t left[64],right[64];
              memcpy(left,target-64,64);
              if(n<11u)memcpy(right,target+4096,64);
              memcpy(mapped[0]+0x6c0,copy_code,sizeof copy_code);
              memcpy(mapped[23]+0x40,&scalar_packet,4);
              if(memcmp(mapped[0]+0x6c0,copy_code,sizeof copy_code))return 2;
              for(unsigned row=0;row<32;row++){
                unsigned t=n*32u+row;
                uint64_t source_va=0x1000005a000ull+(uint64_t)row*128u;
                uint64_t c_va=c_base+(uint64_t)row*128u;
                memcpy(mapped[28]+0x1ba0,&source_va,8);
                memcpy(mapped[28]+0x1ba8,&source_va,8);
                memcpy(mapped[28]+0x1bb0,&c_va,8);
                if(memcmp(mapped[28]+0x1ba0,&source_va,8) ||
                   memcmp(mapped[28]+0x1ba8,&source_va,8) ||
                   memcmp(mapped[28]+0x1bb0,&c_va,8) ||
                   *full_ready!=0x800000f0u || !no_agx_metal())return 2;
                void (^scheduled_clear)(void)=Block_copy(^{zm[2*t]=0x70000u+2u*t;});
                void (^completed_clear)(void)=Block_copy(^{zm[2*t+1]=0x70001u+2u*t;});
                if(!scheduled_clear || !completed_clear || scheduled_clear==completed_clear)return 2;
                uint8_t record_clear[64]={0},out_clear[64]={0};
                uint32_t kid=shmem[1].id,sid=shmem[0].id;
                uintptr_t sp=(uintptr_t)scheduled_clear,cp=(uintptr_t)completed_clear;
                memcpy(record_clear,&kid,4);memcpy(record_clear+4,&sid,4);
                memcpy(record_clear+0x10,&sp,sizeof sp);memcpy(record_clear+0x18,&cp,sizeof cp);
                int status=Submit(queue,NULL,1,record_clear,64,out_clear);
                unsigned waited=0;for(;waited<3000 && !zm[2*t+1];waited++)usleep(10000);
                unsigned prefix=(row+1u)*128u;
                unsigned exact=memcmp(target,zero_block,prefix)==0 &&
                               memcmp(target+prefix,original+prefix,4096u-prefix)==0;
                unsigned guards=memcmp(target-64,left,64)==0 &&
                                (n==11u || memcmp(target+4096,right,64)==0);
                unsigned resident=1;for(unsigned p=0;p<48;p++)if(p!=slot)
                  resident&=memcmp(slots[p],completed_tiles+p*1024u,4096)==0;
                uint32_t outword=0;memcpy(&outword,out_clear,4);
                fprintf(stderr,"PURE FFN CONTRACT ALL CLEAR n=%u row=%u returned status=%d outword=0x%08x exact=%u guards=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                        n,row,status,outword,exact,guards,resident,
                        (unsigned long long)zm[2*t],(unsigned long long)zm[2*t+1],waited);
                if(status || outword || !exact || !guards || !resident ||
                   zm[2*t]!=0x70000u+2u*t || zm[2*t+1]!=0x70001u+2u*t ||
                   !no_agx_metal())return 2;
              }
              memcpy(completed_tiles+slot*1024u,target,4096);
              for(unsigned k=0;k<24;k++){
                unsigned t=n*24u+k;
                uint32_t previous_c[1024];memcpy(previous_c,target,4096);
                uint64_t a_va=(k<12u?0x10000038000ull:0x10000060000ull)+
                              (uint64_t)(k%12u)*8192u;
                uint64_t b_va=0x10000035e80ull;
                memcpy(mapped[2]+0x5e80,contract_b_all+t*4096u,4096);
                memcpy(mapped[0]+0x6c0,expected_inputs[0],1458);
                memcpy(mapped[23]+0x40,&tensor_packet,4);
                memcpy(mapped[28]+0x1ba0,&a_va,8);
                memcpy(mapped[28]+0x1ba8,&b_va,8);
                memcpy(mapped[28]+0x1bb0,&c_base,8);
                if(memcmp(mapped[0]+0x6c0,expected_inputs[0],1458) ||
                   memcmp(mapped[2]+0x5e80,contract_b_all+t*4096u,4096) ||
                   memcmp(mapped[28]+0x1ba0,&a_va,8) ||
                   memcmp(mapped[28]+0x1ba8,&b_va,8) ||
                   memcmp(mapped[28]+0x1bb0,&c_base,8) ||
                   *full_ready!=0x800000f0u || !no_agx_metal())return 2;
                void (^scheduled_tensor)(void)=Block_copy(^{tm[2*t]=0x80000u+2u*t;});
                void (^completed_tensor)(void)=Block_copy(^{tm[2*t+1]=0x80001u+2u*t;});
                if(!scheduled_tensor || !completed_tensor || scheduled_tensor==completed_tensor)return 2;
                uint8_t record_tensor[64]={0},out_tensor[64]={0};
                uint32_t kid=shmem[1].id,sid=shmem[0].id;
                uintptr_t sp=(uintptr_t)scheduled_tensor,cp=(uintptr_t)completed_tensor;
                memcpy(record_tensor,&kid,4);memcpy(record_tensor+4,&sid,4);
                memcpy(record_tensor+0x10,&sp,sizeof sp);memcpy(record_tensor+0x18,&cp,sizeof cp);
                int status=Submit(queue,NULL,1,record_tensor,64,out_tensor);
                unsigned waited=0;for(;waited<3000 && !tm[2*t+1];waited++)usleep(10000);
                unsigned changed=0,finite=1;
                for(unsigned i=0;i<1024;i++){
                  uint32_t word=0;memcpy(&word,target+4u*i,4);
                  changed+=word!=previous_c[i];
                  float value=0;memcpy(&value,&word,4);finite&=isfinite(value);
                }
                unsigned guards=memcmp(target-64,left,64)==0 &&
                                (n==11u || memcmp(target+4096,right,64)==0);
                unsigned inputs=memcmp(slots[2u*k],pack_full_expected+k*4096u,4096)==0 &&
                                memcmp(mapped[2]+0x5e80,contract_b_all+t*4096u,4096)==0 &&
                                memcmp(zero_b,zero_block,4096)==0;
                unsigned resident=1;for(unsigned p=0;p<48;p++)if(p!=slot)
                  resident&=memcmp(slots[p],completed_tiles+p*1024u,4096)==0;
                uint32_t outword=0;memcpy(&outword,out_tensor,4);
                fprintf(stderr,"PURE FFN CONTRACT ALL n=%u k=%u returned status=%d outword=0x%08x changed=%u/1024 finite=%u guards=%u inputs=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                        n,k,status,outword,changed,finite,guards,inputs,resident,
                        (unsigned long long)tm[2*t],(unsigned long long)tm[2*t+1],waited);
                fprintf(stderr,"PURE FFN CONTRACT ALL n=%u k=%u C words:",n,k);
                for(unsigned i=0;i<1024;i++){
                  uint32_t word=0;memcpy(&word,target+4u*i,4);fprintf(stderr," %08x",word);
                }
                fputc('\n',stderr);
                if(status || outword || !changed || !finite || !guards || !inputs ||
                   !resident || tm[2*t]!=0x80000u+2u*t ||
                   tm[2*t+1]!=0x80001u+2u*t || !no_agx_metal())return 2;
              }
              memcpy(completed_tiles+slot*1024u,target,4096);
            }
            fprintf(stderr,"PURE FFN CONTRACT ALL COMPLETE: 12 tensor output tiles retained beside 24 packed A blocks.\n");
          }
        }
        if(ffn_contract_bias){
          static volatile uint64_t bias_marks[768]={0};
          volatile uint64_t* bm=bias_marks;
          uint32_t scalar_packet=0x0e4006c7u;
          memcpy(mapped[0]+0x6c0,bias_code,sizeof bias_code);
          memcpy(mapped[23]+0x40,&scalar_packet,4);
          if(memcmp(mapped[0]+0x6c0,bias_code,sizeof bias_code))return 2;
          for(unsigned n=0;n<12;n++){
            unsigned slot=2u*n+1u;
            uint8_t* target=(uint8_t*)slots[slot];
            uint64_t base_va=0x10000038000ull+(uint64_t)slot*4096u;
            memcpy(mapped[2]+0x5e80,contract_bias_values+n*128u,128);
            for(unsigned row=0;row<32;row++){
              unsigned t=n*32u+row;
              uint64_t a_va=base_va+(uint64_t)row*128u;
              uint64_t b_va=0x10000035e80ull;
              memcpy(mapped[28]+0x1ba0,&a_va,8);
              memcpy(mapped[28]+0x1ba8,&b_va,8);
              memcpy(mapped[28]+0x1bb0,&a_va,8);
              if(memcmp(mapped[28]+0x1ba0,&a_va,8) ||
                 memcmp(mapped[28]+0x1ba8,&b_va,8) ||
                 memcmp(mapped[28]+0x1bb0,&a_va,8) ||
                 memcmp(mapped[2]+0x5e80,contract_bias_values+n*128u,128) ||
                 *full_ready!=0x800000f0u || !no_agx_metal())return 2;
              uint32_t prior[32];memcpy(prior,target+row*128u,128);
              void (^scheduled_bias)(void)=Block_copy(^{bm[2*t]=0x90000u+2u*t;});
              void (^completed_bias)(void)=Block_copy(^{bm[2*t+1]=0x90001u+2u*t;});
              if(!scheduled_bias || !completed_bias || scheduled_bias==completed_bias)return 2;
              uint8_t record_bias[64]={0},out_bias[64]={0};
              uint32_t kid=shmem[1].id,sid=shmem[0].id;
              uintptr_t sp=(uintptr_t)scheduled_bias,cp=(uintptr_t)completed_bias;
              memcpy(record_bias,&kid,4);memcpy(record_bias+4,&sid,4);
              memcpy(record_bias+0x10,&sp,sizeof sp);memcpy(record_bias+0x18,&cp,sizeof cp);
              int status=Submit(queue,NULL,1,record_bias,64,out_bias);
              unsigned waited=0;for(;waited<3000 && !bm[2*t+1];waited++)usleep(10000);
              unsigned changed=0,finite=1;
              for(unsigned i=0;i<32;i++){
                uint32_t word=0;memcpy(&word,target+row*128u+4u*i,4);
                changed+=word!=prior[i];
                float value=0;memcpy(&value,&word,4);finite&=isfinite(value);
              }
              unsigned exact=memcmp(target+row*128u,
                contract_bias_expected+((size_t)row*384u+n*32u)*4u,128)==0;
              memcpy(completed_tiles+slot*1024u+row*32u,target+row*128u,128);
              unsigned resident=1;for(unsigned p=0;p<48;p++)resident&=
                memcmp(slots[p],completed_tiles+p*1024u,4096)==0;
              uint32_t outword=0;memcpy(&outword,out_bias,4);
              fprintf(stderr,"PURE FFN CONTRACT BIAS n=%u row=%u returned status=%d outword=0x%08x changed=%u/32 finite=%u exact=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                      n,row,status,outword,changed,finite,exact,resident,
                      (unsigned long long)bm[2*t],(unsigned long long)bm[2*t+1],waited);
              fprintf(stderr,"PURE FFN CONTRACT BIAS n=%u row=%u words:",n,row);
              for(unsigned i=0;i<32;i++){
                uint32_t word=0;memcpy(&word,target+row*128u+4u*i,4);
                fprintf(stderr," %08x",word);
              }
              fputc('\n',stderr);
              if(status || outword || !changed || !finite || !exact || !resident ||
                 bm[2*t]!=0x90000u+2u*t || bm[2*t+1]!=0x90001u+2u*t ||
                 !no_agx_metal())return 2;
            }
          }
          fprintf(stderr,"PURE FFN CONTRACT BIAS COMPLETE: 12 resident tensor C tiles biased in place.\n");
        }
        if(ffn_residual_add){
          static volatile uint64_t residual_marks[768]={0};
          volatile uint64_t* rm=residual_marks;
          uint32_t scalar_packet=0x0e4006c7u;
          memcpy(mapped[0]+0x6c0,bias_code,sizeof bias_code);
          memcpy(mapped[23]+0x40,&scalar_packet,4);
          if(memcmp(mapped[0]+0x6c0,bias_code,sizeof bias_code))return 2;
          for(unsigned n=0;n<12;n++){
            unsigned slot=2u*n+1u;
            uint8_t* target=(uint8_t*)slots[slot];
            uint64_t base_va=0x10000038000ull+(uint64_t)slot*4096u;
            for(unsigned row=0;row<32;row++){
              unsigned t=n*32u+row;
              const uint8_t* source_row=residual_source+((size_t)row*384u+n*32u)*4u;
              memcpy(mapped[2]+0x5e80,source_row,128);
              uint64_t a_va=base_va+(uint64_t)row*128u;
              uint64_t b_va=0x10000035e80ull;
              memcpy(mapped[28]+0x1ba0,&a_va,8);
              memcpy(mapped[28]+0x1ba8,&b_va,8);
              memcpy(mapped[28]+0x1bb0,&a_va,8);
              if(memcmp(mapped[28]+0x1ba0,&a_va,8) ||
                 memcmp(mapped[28]+0x1ba8,&b_va,8) ||
                 memcmp(mapped[28]+0x1bb0,&a_va,8) ||
                 memcmp(mapped[2]+0x5e80,source_row,128) ||
                 *full_ready!=0x800000f0u || !no_agx_metal())return 2;
              uint32_t prior[32];memcpy(prior,target+row*128u,128);
              void (^scheduled_residual)(void)=Block_copy(^{rm[2*t]=0xa0000u+2u*t;});
              void (^completed_residual)(void)=Block_copy(^{rm[2*t+1]=0xa0001u+2u*t;});
              if(!scheduled_residual || !completed_residual ||
                 scheduled_residual==completed_residual)return 2;
              uint8_t record_residual[64]={0},out_residual[64]={0};
              uint32_t kid=shmem[1].id,sid=shmem[0].id;
              uintptr_t sp=(uintptr_t)scheduled_residual,cp=(uintptr_t)completed_residual;
              memcpy(record_residual,&kid,4);memcpy(record_residual+4,&sid,4);
              memcpy(record_residual+0x10,&sp,sizeof sp);memcpy(record_residual+0x18,&cp,sizeof cp);
              int status=Submit(queue,NULL,1,record_residual,64,out_residual);
              unsigned waited=0;for(;waited<3000 && !rm[2*t+1];waited++)usleep(10000);
              unsigned changed=0,finite=1;
              for(unsigned i=0;i<32;i++){
                uint32_t word=0;memcpy(&word,target+row*128u+4u*i,4);
                changed+=word!=prior[i];
                float value=0;memcpy(&value,&word,4);finite&=isfinite(value);
              }
              unsigned exact=memcmp(target+row*128u,
                residual_expected+((size_t)row*384u+n*32u)*4u,128)==0;
              memcpy(completed_tiles+slot*1024u+row*32u,target+row*128u,128);
              unsigned resident=1;for(unsigned p=0;p<48;p++)resident&=
                memcmp(slots[p],completed_tiles+p*1024u,4096)==0;
              unsigned source_ok=memcmp(mapped[2]+0x5e80,source_row,128)==0;
              uint32_t outword=0;memcpy(&outword,out_residual,4);
              fprintf(stderr,"PURE FFN RESIDUAL ADD n=%u row=%u returned status=%d outword=0x%08x changed=%u/32 finite=%u exact=%u source=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                      n,row,status,outword,changed,finite,exact,source_ok,resident,
                      (unsigned long long)rm[2*t],(unsigned long long)rm[2*t+1],waited);
              fprintf(stderr,"PURE FFN RESIDUAL ADD n=%u row=%u words:",n,row);
              for(unsigned i=0;i<32;i++){
                uint32_t word=0;memcpy(&word,target+row*128u+4u*i,4);
                fprintf(stderr," %08x",word);
              }
              fputc('\n',stderr);
              if(status || outword || changed!=32u || !finite || !exact ||
                 !source_ok || !resident || rm[2*t]!=0xa0000u+2u*t ||
                 rm[2*t+1]!=0xa0001u+2u*t || !no_agx_metal())return 2;
            }
          }
          fprintf(stderr,"PURE FFN RESIDUAL ADD COMPLETE: 12 resident biased C tiles received the full model source.\n");
        }
        if(ffn_gather_residual){
          uint8_t* target=mapped[17]+0x8000u;
          uint8_t* zero_b=mapped[17]+0x2000u;
          uint8_t zero_block[4096]={0};
          uint8_t left[64],right[64];
          uint8_t* gather_progress=malloc(49152u);if(!gather_progress)return 2;
          memcpy(gather_progress,gather_old_target,49152u);
          memcpy(left,target-64,64);
          memcpy(right,target+49152u,64);
          if(memcmp(target,gather_old_target,49152u) ||
             memcmp(zero_b,zero_block,4096u))return 2;
          uint32_t scalar_packet=0x0e4006c7u;
          memcpy(mapped[0]+0x6c0,copy_code,sizeof copy_code);
          memcpy(mapped[23]+0x40,&scalar_packet,4);
          if(memcmp(mapped[0]+0x6c0,copy_code,sizeof copy_code))return 2;
          static volatile uint64_t gather_marks[768]={0};
          volatile uint64_t* gm=gather_marks;
          for(unsigned n=0;n<12;n++){
            uint8_t* source=(uint8_t*)slots[2u*n+1u];
            for(unsigned row=0;row<32;row++){
              unsigned t=n*32u+row;
              size_t offset=((size_t)row*384u+n*32u)*4u;
              uint64_t a_va=0x10000038000ull+(uint64_t)(2u*n+1u)*4096u+(uint64_t)row*128u;
              uint64_t b_va=0x1000005a000ull+(uint64_t)row*128u;
              uint64_t c_va=0x10000060000ull+offset;
              memcpy(mapped[28]+0x1ba0,&a_va,8);
              memcpy(mapped[28]+0x1ba8,&b_va,8);
              memcpy(mapped[28]+0x1bb0,&c_va,8);
              if(memcmp(mapped[28]+0x1ba0,&a_va,8) ||
                 memcmp(mapped[28]+0x1ba8,&b_va,8) ||
                 memcmp(mapped[28]+0x1bb0,&c_va,8) ||
                 memcmp(source+row*128u,residual_expected+offset,128u) ||
                 memcmp(zero_b,zero_block,4096u) ||
                 *full_ready!=0x800000f0u || !no_agx_metal())return 2;
              void (^scheduled_gather)(void)=Block_copy(^{gm[2*t]=0xb0000u+2u*t;});
              void (^completed_gather)(void)=Block_copy(^{gm[2*t+1]=0xb0001u+2u*t;});
              if(!scheduled_gather || !completed_gather ||
                 scheduled_gather==completed_gather)return 2;
              uint8_t record_gather[64]={0},out_gather[64]={0};
              uint32_t kid=shmem[1].id,sid=shmem[0].id;
              uintptr_t sp=(uintptr_t)scheduled_gather,cp=(uintptr_t)completed_gather;
              memcpy(record_gather,&kid,4);memcpy(record_gather+4,&sid,4);
              memcpy(record_gather+0x10,&sp,sizeof sp);
              memcpy(record_gather+0x18,&cp,sizeof cp);
              int status=Submit(queue,NULL,1,record_gather,64,out_gather);
              unsigned waited=0;for(;waited<3000 && !gm[2*t+1];waited++)usleep(10000);
              uint32_t outword=0;memcpy(&outword,out_gather,4);
              unsigned exact=memcmp(target+offset,residual_expected+offset,128u)==0;
              memcpy(gather_progress+offset,residual_expected+offset,128u);
              unsigned rest=memcmp(target,gather_progress,49152u)==0;
              unsigned guards=memcmp(target-64,left,64)==0 &&
                              memcmp(target+49152u,right,64)==0;
              unsigned sources=memcmp(source+row*128u,residual_expected+offset,128u)==0 &&
                               memcmp(zero_b,zero_block,4096u)==0;
              fprintf(stderr,"PURE FFN GATHER n=%u row=%u returned status=%d outword=0x%08x exact=%u rest=%u guards=%u sources=%u marks=%llu/%llu wait=%u/3000.\n",
                      n,row,status,outword,exact,rest,guards,sources,
                      (unsigned long long)gm[2*t],(unsigned long long)gm[2*t+1],waited);
              if(status || outword || !exact || !rest || !guards || !sources ||
                 gm[2*t]!=0xb0000u+2u*t || gm[2*t+1]!=0xb0001u+2u*t ||
                 !no_agx_metal())return 2;
            }
          }
          if(memcmp(target,residual_expected,49152u))return 2;
          for(unsigned n=0;n<12;n++)
            memcpy(completed_tiles+(24u+n)*1024u,target+n*4096u,4096u);
          unsigned resident=1;for(unsigned p=0;p<48;p++)resident&=
            memcmp(slots[p],completed_tiles+p*1024u,4096u)==0;
          fprintf(stderr,"PURE FFN GATHER COMPLETE: 32x384 residual matrix contiguous, exact=%u resident=%u.\n",
                  memcmp(target,residual_expected,49152u)==0,resident);
          if(!resident)return 2;
          free(gather_progress);
        }
        if(ffn_four_binding){
          uint8_t* source=mapped[17]+0x8000u;
          uint8_t* output=ffn_append_output?mapped[29]+0x80u:mapped[2]+0x8000u;
          if(ffn_probe_appended_third){
            uint8_t* zero_b=mapped[17]+0x2000u;
            uint8_t zero_block[4096]={0};
            uint8_t probe_left[64],probe_right[64];
            memcpy(probe_left,output-64,64);memcpy(probe_right,output+4096u,64);
            if(memcmp(source,residual_expected,49152u) ||
               memcmp(zero_b,zero_block,4096u))return 2;
            for(unsigned i=0;i<32;i++)((uint32_t*)output)[i]=0x7fc01234u;
            uint32_t scalar_packet=0x0e4006c7u;
            const uint64_t probe_bindings[3]={0x10000060000ull,0x1000005a000ull,
                                              0x100000f0080ull};
            memcpy(mapped[0]+0x6c0,copy_code,sizeof copy_code);
            memcpy(mapped[23]+0x40,&scalar_packet,4);
            memcpy(mapped[28]+0x1ba0,probe_bindings,sizeof probe_bindings);
            if(memcmp(mapped[0]+0x6c0,copy_code,sizeof copy_code) ||
               memcmp(mapped[28]+0x1ba0,probe_bindings,sizeof probe_bindings) ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            static volatile uint64_t probe_marks[2]={0};
            volatile uint64_t* pm=probe_marks;
            void (^scheduled_probe)(void)=Block_copy(^{pm[0]=0xe0000u;});
            void (^completed_probe)(void)=Block_copy(^{pm[1]=0xe0001u;});
            if(!scheduled_probe || !completed_probe ||
               scheduled_probe==completed_probe)return 2;
            uint8_t record_probe[64]={0},out_probe[64]={0};
            uint32_t kid=shmem[1].id,sid=shmem[0].id;
            uintptr_t sp=(uintptr_t)scheduled_probe,cp=(uintptr_t)completed_probe;
            memcpy(record_probe,&kid,4);memcpy(record_probe+4,&sid,4);
            memcpy(record_probe+0x10,&sp,sizeof sp);
            memcpy(record_probe+0x18,&cp,sizeof cp);
            int status=Submit(queue,NULL,1,record_probe,64,out_probe);
            unsigned waited=0;for(;waited<3000 && !pm[1];waited++)usleep(10000);
            uint32_t outword=0;memcpy(&outword,out_probe,4);
            unsigned exact=memcmp(output,residual_expected,128u)==0;
            unsigned guards=memcmp(output-64,probe_left,64)==0 &&
                            memcmp(output+4096u,probe_right,64)==0;
            fprintf(stderr,"PURE FFN APPENDED THIRD returned status=%d outword=0x%08x exact=%u guards=%u marks=%llu/%llu wait=%u/3000.\n",
                    status,outword,exact,guards,
                    (unsigned long long)pm[0],(unsigned long long)pm[1],waited);
            if(status || outword || !exact || !guards ||
               pm[0]!=0xe0000u || pm[1]!=0xe0001u || !no_agx_metal())return 2;
          }
          uint8_t original[4096],left[64],right[64];
          memcpy(original,output,4096u);
          memcpy(left,output-64,64);memcpy(right,output+4096u,64);
          if(memcmp(source,residual_expected,49152u))return 2;
          memcpy(mapped[2]+0x4d80,four_gamma,sizeof four_gamma);
          memcpy(mapped[2]+0x5e80,four_beta,sizeof four_beta);
          for(unsigned i=0;i<32;i++)((uint32_t*)output)[i]=0x7fc01234u;
          uint32_t scalar_packet=0x0e4006c7u;
          const uint64_t bindings[4]={0x10000060000ull,0x10000034d80ull,
                                      0x10000035e80ull,
                                      ffn_append_output?0x100000f0080ull:0x10000038000ull};
          memcpy(mapped[0]+0x6c0,four_code,sizeof four_code);
          memcpy(mapped[23]+0x40,&scalar_packet,4);
          memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
          if(memcmp(mapped[0]+0x6c0,four_code,sizeof four_code) ||
             memcmp(mapped[23]+0x40,&scalar_packet,4) ||
             memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
             memcmp(mapped[2]+0x4d80,four_gamma,sizeof four_gamma) ||
             memcmp(mapped[2]+0x5e80,four_beta,sizeof four_beta) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t four_marks[2]={0};
          volatile uint64_t* fm=four_marks;
          void (^scheduled_four)(void)=Block_copy(^{fm[0]=0xd0000u;});
          void (^completed_four)(void)=Block_copy(^{fm[1]=0xd0001u;});
          if(!scheduled_four || !completed_four ||
             scheduled_four==completed_four)return 2;
          uint8_t record_four[64]={0},out_four[64]={0};
          uint32_t kid=shmem[1].id,sid=shmem[0].id;
          uintptr_t sp=(uintptr_t)scheduled_four,cp=(uintptr_t)completed_four;
          memcpy(record_four,&kid,4);memcpy(record_four+4,&sid,4);
          memcpy(record_four+0x10,&sp,sizeof sp);
          memcpy(record_four+0x18,&cp,sizeof cp);
          int status=Submit(queue,NULL,1,record_four,64,out_four);
          unsigned waited=0;for(;waited<3000 && !fm[1];waited++)usleep(10000);
          unsigned exact=memcmp(output,four_expected,128u)==0;
          unsigned tail=memcmp(output+128u,original+128u,4096u-128u)==0;
          unsigned guards=memcmp(output-64,left,64)==0 &&
                          memcmp(output+4096u,right,64)==0;
          unsigned inputs=memcmp(source,residual_expected,49152u)==0 &&
                          memcmp(mapped[2]+0x4d80,four_gamma,sizeof four_gamma)==0 &&
                          memcmp(mapped[2]+0x5e80,four_beta,sizeof four_beta)==0;
          uint32_t outword=0;memcpy(&outword,out_four,4);
          fprintf(stderr,"PURE FFN FOUR BINDING returned status=%d outword=0x%08x exact=%u tail=%u guards=%u inputs=%u marks=%llu/%llu wait=%u/3000.\n",
                  status,outword,exact,tail,guards,inputs,
                  (unsigned long long)fm[0],(unsigned long long)fm[1],waited);
          fprintf(stderr,"PURE FFN FOUR BINDING output words:");
          for(unsigned i=0;i<32;i++)fprintf(stderr," %08x",((uint32_t*)output)[i]);
          fputc('\n',stderr);
          if(status || outword || !exact || !tail || !guards || !inputs ||
             fm[0]!=0xd0000u || fm[1]!=0xd0001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE FFN FOUR BINDING COMPLETE: fourth output pointer wrote exact 32 words.\n");
        }
        if(ffn_coop_layernorm){
          uint8_t* source=mapped[17]+0x8000u;
          uint8_t* output=mapped[2]+0x8000u;
          uint8_t left[64],right[64];
          memcpy(left,output-64,64);memcpy(right,output+49152u,64);
          if(memcmp(source,residual_expected,49152u))return 2;
          memcpy(mapped[2]+0x4d80,coop_gamma,sizeof coop_gamma);
          memcpy(mapped[2]+(ffn_packed_ln?0x5380u:0x5e80u),coop_beta,sizeof coop_beta);
          for(unsigned i=0;i<12288;i++)((uint32_t*)output)[i]=0x7fc01234u;
          if(memcmp(mapped[2]+0x4d80,coop_gamma,sizeof coop_gamma) ||
             memcmp(mapped[2]+(ffn_packed_ln?0x5380u:0x5e80u),coop_beta,sizeof coop_beta))return 2;
          uint32_t scratch_word[2]={0x0c00100fu,0};
          uint32_t tensor_packet=0x0e5806c7u;
          uint32_t y_threads=32u;
          const uint64_t bindings[4]={0x10000060000ull,0x10000034d80ull,
                                      ffn_packed_ln?0x10000038000ull:0x10000035e80ull,
                                      ffn_packed_ln?0:0x10000038000ull};
          size_t coop_code_bytes=ffn_packed_ln?5316u:5076u;
          memcpy(mapped[0]+0x6c0,coop_code,coop_code_bytes);
          memcpy(mapped[23]+0x38,scratch_word,8);
          memcpy(mapped[23]+0x40,&tensor_packet,4);
          memcpy(mapped[22]+0xac,&y_threads,4);
          memcpy(mapped[25]+0x14,&y_threads,4);
          memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
          if(memcmp(mapped[0]+0x6c0,coop_code,coop_code_bytes) ||
             memcmp(mapped[23]+0x38,scratch_word,8) ||
             memcmp(mapped[23]+0x40,&tensor_packet,4) ||
             memcmp(mapped[22]+0xac,&y_threads,4) ||
             memcmp(mapped[25]+0x14,&y_threads,4) ||
             memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t ln_marks[2]={0};
          volatile uint64_t* lm=ln_marks;
          void (^scheduled_ln)(void)=Block_copy(^{lm[0]=0xc0000u;});
          void (^completed_ln)(void)=Block_copy(^{lm[1]=0xc0001u;});
          if(!scheduled_ln || !completed_ln || scheduled_ln==completed_ln)return 2;
          uint8_t record_ln[64]={0},out_ln[64]={0};
          uint32_t kid=shmem[1].id,sid=shmem[0].id;
          uintptr_t sp=(uintptr_t)scheduled_ln,cp=(uintptr_t)completed_ln;
          memcpy(record_ln,&kid,4);memcpy(record_ln+4,&sid,4);
          memcpy(record_ln+0x10,&sp,sizeof sp);
          memcpy(record_ln+0x18,&cp,sizeof cp);
          fprintf(stderr,"PURE FFN COOP LN entering: source=0x%llx gamma=0x%llx beta_or_output=0x%llx output_or_unused=0x%llx grid=32x32 scratch=128.\n",
                  (unsigned long long)bindings[0],(unsigned long long)bindings[1],
                  (unsigned long long)bindings[2],(unsigned long long)bindings[3]);
          int status=Submit(queue,NULL,1,record_ln,64,out_ln);
          unsigned waited=0;for(;waited<3000 && !lm[1];waited++)usleep(10000);
          unsigned changed=0,finite=1,within=1;
          double max_abs=0,max_fraction=0;
          for(unsigned i=0;i<12288;i++){
            uint32_t word=((uint32_t*)output)[i];
            float value=0;memcpy(&value,&word,4);
            changed+=word!=0x7fc01234u;
            finite&=isfinite(value);
            double error=fabs((double)value-coop_reference[i]);
            double limit=2e-5*(1.0+fabs(coop_reference[i]));
            within&=isfinite(value) && error<=limit;
            if(error>max_abs)max_abs=error;
            if(error/limit>max_fraction)max_fraction=error/limit;
          }
          unsigned guards=memcmp(output-64,left,64)==0 &&
                          memcmp(output+49152u,right,64)==0;
          unsigned inputs=memcmp(source,residual_expected,49152u)==0 &&
                          memcmp(mapped[2]+0x4d80,coop_gamma,sizeof coop_gamma)==0 &&
                          memcmp(mapped[2]+(ffn_packed_ln?0x5380u:0x5e80u),coop_beta,sizeof coop_beta)==0;
          unsigned untouched=1;for(unsigned p=12;p<48;p++)untouched&=
            memcmp(slots[p],completed_tiles+p*1024u,4096u)==0;
          uint32_t outword=0;memcpy(&outword,out_ln,4);
          fprintf(stderr,"PURE FFN COOP LN returned status=%d outword=0x%08x changed=%u/12288 finite=%u within=%u max_abs=%.9g max_fraction=%.9g guards=%u inputs=%u untouched=%u marks=%llu/%llu wait=%u/3000.\n",
                  status,outword,changed,finite,within,max_abs,max_fraction,
                  guards,inputs,untouched,(unsigned long long)lm[0],
                  (unsigned long long)lm[1],waited);
          fprintf(stderr,"PURE FFN COOP LN output words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)output)[i]);
          fputc('\n',stderr);
          if(status || outword || changed!=12288u || !finite || !within ||
             !guards || !inputs || !untouched || lm[0]!=0xc0000u ||
             lm[1]!=0xc0001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE FFN COOP LN COMPLETE: all 12288 normalized values meet FP64 budget.\n");
        }
        if(ffn_query_tile){
          uint8_t* source=mapped[2]+0x8000u;
          uint8_t* weight=mapped[17]+0x1000u;
          uint8_t* output=mapped[17]+0x13000u;
          uint8_t left[64],right[64];
          memcpy(left,output-64,64);memcpy(right,output+49152u,64);
          if(memcmp(source,query_source_expected,49152u) ||
             weight+49280u>output-64)return 2;
          memcpy(weight,query_packed,49280u);
          for(unsigned i=0;i<12288;i++)((uint32_t*)output)[i]=0x7fc01234u;
          uint32_t scratch_word[2]={0x0c000007u,0};
          uint32_t scalar_packet=0x0e4006c7u;
          uint32_t xy_threads=32u;
          const uint64_t bindings[3]={0x10000038000ull,0x10000059000ull,
                                      0x1000006b000ull};
          memcpy(mapped[0]+0x6c0,query_code,sizeof query_code);
          memcpy(mapped[23]+0x38,scratch_word,8);
          memcpy(mapped[23]+0x40,&scalar_packet,4);
          memcpy(mapped[22]+0xa8,&xy_threads,4);
          memcpy(mapped[22]+0xac,&xy_threads,4);
          memcpy(mapped[25]+0x10,&xy_threads,4);
          memcpy(mapped[25]+0x14,&xy_threads,4);
          memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
          if(memcmp(mapped[0]+0x6c0,query_code,sizeof query_code) ||
             memcmp(mapped[23]+0x38,scratch_word,8) ||
             memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
             memcmp(weight,query_packed,49280u) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t query_marks[2]={0};
          volatile uint64_t* qm=query_marks;
          void (^scheduled_query)(void)=Block_copy(^{qm[0]=0xf0000u;});
          void (^completed_query)(void)=Block_copy(^{qm[1]=0xf0001u;});
          if(!scheduled_query || !completed_query ||
             scheduled_query==completed_query)return 2;
          uint8_t record_query[64]={0},out_query[64]={0};
          uint32_t kid=shmem[1].id,sid=shmem[0].id;
          uintptr_t sp=(uintptr_t)scheduled_query,cp=(uintptr_t)completed_query;
          memcpy(record_query,&kid,4);memcpy(record_query+4,&sid,4);
          memcpy(record_query+0x10,&sp,sizeof sp);
          memcpy(record_query+0x18,&cp,sizeof cp);
          int status=Submit(queue,NULL,1,record_query,64,out_query);
          unsigned waited=0;for(;waited<3000 && !qm[1];waited++)usleep(10000);
          unsigned changed=0,finite=1,within=1,rest=1;
          double max_abs=0,max_fraction=0;
          for(unsigned row=0;row<32;row++)for(unsigned col=0;col<384;col++){
            unsigned i=row*384u+col;
            uint32_t word=((uint32_t*)output)[i];
            if(col>=32){rest&=word==0x7fc01234u;continue;}
            changed+=word!=0x7fc01234u;
            float value=0;memcpy(&value,&word,4);
            finite&=isfinite(value);
            double error=fabs((double)value-query_reference[row*32u+col]);
            double limit=2e-5*(1.0+fabs(query_reference[row*32u+col]));
            within&=isfinite(value) && error<=limit;
            if(error>max_abs)max_abs=error;
            if(error/limit>max_fraction)max_fraction=error/limit;
          }
          unsigned guards=memcmp(output-64,left,64)==0 &&
                          memcmp(output+49152u,right,64)==0;
          unsigned inputs=memcmp(source,query_source_expected,49152u)==0 &&
                          memcmp(weight,query_packed,49280u)==0;
          uint32_t outword=0;memcpy(&outword,out_query,4);
          fprintf(stderr,"PURE FFN QUERY TILE returned status=%d outword=0x%08x changed=%u/1024 finite=%u within=%u rest=%u guards=%u inputs=%u max_abs=%.9g max_fraction=%.9g marks=%llu/%llu wait=%u/3000.\n",
                  status,outword,changed,finite,within,rest,guards,inputs,
                  max_abs,max_fraction,(unsigned long long)qm[0],
                  (unsigned long long)qm[1],waited);
          fprintf(stderr,"PURE FFN QUERY TILE words:");
          for(unsigned row=0;row<32;row++)for(unsigned col=0;col<32;col++)
            fprintf(stderr," %08x",((uint32_t*)output)[row*384u+col]);
          fputc('\n',stderr);
          if(status || outword || changed!=1024u || !finite || !within ||
             !rest || !guards || !inputs || qm[0]!=0xf0000u ||
             qm[1]!=0xf0001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE FFN QUERY TILE COMPLETE: 32x32 model-weight projection consumed GPU LayerNorm output.\n");
          if(ffn_query_all){
            static volatile uint64_t all_marks[24]={0};
            volatile uint64_t* am=all_marks;
            for(unsigned n=1;n<12;n++){
              memcpy(weight,query_packed+n*49280u,49280u);
              uint64_t c_va=0x1000006b000ull+(uint64_t)n*128u;
              memcpy(mapped[28]+0x1bb0,&c_va,8);
              if(memcmp(weight,query_packed+n*49280u,49280u) ||
                 memcmp(mapped[28]+0x1bb0,&c_va,8) ||
                 memcmp(source,query_source_expected,49152u) ||
                 *full_ready!=0x800000f0u || !no_agx_metal())return 2;
              void (^scheduled)(void)=Block_copy(^{am[2*n]=0xf0000u+2u*n;});
              void (^completed)(void)=Block_copy(^{am[2*n+1]=0xf0001u+2u*n;});
              if(!scheduled || !completed || scheduled==completed)return 2;
              uint8_t record[64]={0},out[64]={0};
              uint32_t kid=shmem[1].id,sid=shmem[0].id;
              uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
              memcpy(record,&kid,4);memcpy(record+4,&sid,4);
              memcpy(record+0x10,&sp,sizeof sp);
              memcpy(record+0x18,&cp,sizeof cp);
              int tile_status=Submit(queue,NULL,1,record,64,out);
              unsigned tile_wait=0;for(;tile_wait<3000 && !am[2*n+1];tile_wait++)usleep(10000);
              unsigned tile_changed=0,tile_finite=1,tile_within=1,tile_rest=1;
              double tile_max_abs=0,tile_max_fraction=0;
              for(unsigned row=0;row<32;row++)for(unsigned col=0;col<384;col++){
                uint32_t word=((uint32_t*)output)[row*384u+col];
                if(col>=(n+1u)*32u){tile_rest&=word==0x7fc01234u;continue;}
                unsigned tile=col/32u,within_tile=col%32u;
                tile_changed+=word!=0x7fc01234u;
                float value=0;memcpy(&value,&word,4);
                tile_finite&=isfinite(value);
                double expected=query_reference[tile*1024u+row*32u+within_tile];
                double error=fabs((double)value-expected);
                double limit=2e-5*(1.0+fabs(expected));
                tile_within&=isfinite(value) && error<=limit;
                if(error>tile_max_abs)tile_max_abs=error;
                if(error/limit>tile_max_fraction)tile_max_fraction=error/limit;
              }
              unsigned tile_guards=memcmp(output-64,left,64)==0 &&
                                   memcmp(output+49152u,right,64)==0;
              unsigned tile_inputs=memcmp(source,query_source_expected,49152u)==0 &&
                                   memcmp(weight,query_packed+n*49280u,49280u)==0;
              uint32_t tile_outword=0;memcpy(&tile_outword,out,4);
              fprintf(stderr,"PURE FFN QUERY ALL n=%u returned status=%d outword=0x%08x changed=%u/%u finite=%u within=%u rest=%u guards=%u inputs=%u max_abs=%.9g max_fraction=%.9g marks=%llu/%llu wait=%u/3000.\n",
                      n,tile_status,tile_outword,tile_changed,(n+1u)*1024u,
                      tile_finite,tile_within,tile_rest,tile_guards,tile_inputs,
                      tile_max_abs,tile_max_fraction,
                      (unsigned long long)am[2*n],(unsigned long long)am[2*n+1],tile_wait);
              fprintf(stderr,"PURE FFN QUERY ALL n=%u words:",n);
              for(unsigned row=0;row<32;row++)for(unsigned col=0;col<32;col++)
                fprintf(stderr," %08x",((uint32_t*)output)[row*384u+n*32u+col]);
              fputc('\n',stderr);
              if(tile_status || tile_outword || tile_changed!=(n+1u)*1024u ||
                 !tile_finite || !tile_within || !tile_rest || !tile_guards ||
                 !tile_inputs || am[2*n]!=0xf0000u+2u*n ||
                 am[2*n+1]!=0xf0001u+2u*n || !no_agx_metal())return 2;
            }
            fprintf(stderr,"PURE FFN QUERY ALL COMPLETE: 32x384 model-weight projection consumed GPU LayerNorm output.\n");
          }
        }
        if(ffn_key_all){
          uint8_t* source=mapped[2]+0x8000u;
          uint8_t* output=mapped[2]+0x14000u;
          uint8_t* weight=mapped[17]+0x1000u;
          uint8_t* query=mapped[17]+0x13000u;
          uint8_t* query_snapshot=malloc(49152u);if(!query_snapshot)return 2;
          memcpy(query_snapshot,query,49152u);
          uint8_t lower_guard[64];memcpy(lower_guard,output-64,64);
          if(memcmp(source,query_source_expected,49152u) ||
             sizes[2]!=0x20000u || output+49152u!=mapped[2]+sizes[2] ||
             weight+49280u>query-64 || !no_agx_metal())return 2;
          for(unsigned i=0;i<12288;i++)((uint32_t*)output)[i]=0x7fc01234u;
          static volatile uint64_t key_marks[24]={0};
          volatile uint64_t* km=key_marks;
          for(unsigned n=0;n<12;n++){
            memcpy(weight,key_packed+n*49280u,49280u);
            uint64_t bindings[3]={0x10000038000ull,0x10000059000ull,
                                  0x10000044000ull+(uint64_t)n*128u};
            memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
            if(memcmp(mapped[0]+0x6c0,query_code,sizeof query_code) ||
               memcmp(mapped[23]+0x38,"\x07\x00\x00\x0c\x00\x00\x00\x00",8) ||
               memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
               memcmp(weight,key_packed+n*49280u,49280u) ||
               memcmp(source,query_source_expected,49152u) ||
               memcmp(query,query_snapshot,49152u) ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            void (^scheduled)(void)=Block_copy(^{km[2*n]=0x110000u+2u*n;});
            void (^completed)(void)=Block_copy(^{km[2*n+1]=0x110001u+2u*n;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t record[64]={0},out[64]={0};
            uint32_t kid=shmem[1].id,sid=shmem[0].id;
            uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
            memcpy(record,&kid,4);memcpy(record+4,&sid,4);
            memcpy(record+0x10,&sp,sizeof sp);
            memcpy(record+0x18,&cp,sizeof cp);
            int status=Submit(queue,NULL,1,record,64,out);
            unsigned waited=0;for(;waited<3000 && !km[2*n+1];waited++)usleep(10000);
            unsigned changed=0,finite=1,within=1,rest=1;
            double max_abs=0,max_fraction=0;
            for(unsigned row=0;row<32;row++)for(unsigned col=0;col<384;col++){
              uint32_t word=((uint32_t*)output)[row*384u+col];
              if(col>=(n+1u)*32u){rest&=word==0x7fc01234u;continue;}
              unsigned tile=col/32u,within_tile=col%32u;
              changed+=word!=0x7fc01234u;
              float value=0;memcpy(&value,&word,4);
              finite&=isfinite(value);
              double expected=key_reference[tile*1024u+row*32u+within_tile];
              double error=fabs((double)value-expected);
              double limit=2e-5*(1.0+fabs(expected));
              within&=isfinite(value) && error<=limit;
              if(error>max_abs)max_abs=error;
              if(error/limit>max_fraction)max_fraction=error/limit;
            }
            unsigned lower=memcmp(output-64,lower_guard,64)==0;
            unsigned inputs=memcmp(source,query_source_expected,49152u)==0 &&
                            memcmp(weight,key_packed+n*49280u,49280u)==0;
            unsigned query_ok=memcmp(query,query_snapshot,49152u)==0;
            uint32_t outword=0;memcpy(&outword,out,4);
            fprintf(stderr,"PURE FFN KEY ALL n=%u returned status=%d outword=0x%08x changed=%u/%u finite=%u within=%u rest=%u lower=%u inputs=%u query=%u max_abs=%.9g max_fraction=%.9g marks=%llu/%llu wait=%u/3000.\n",
                    n,status,outword,changed,(n+1u)*1024u,finite,within,rest,lower,inputs,query_ok,
                    max_abs,max_fraction,(unsigned long long)km[2*n],
                    (unsigned long long)km[2*n+1],waited);
            fprintf(stderr,"PURE FFN KEY ALL n=%u words:",n);
            for(unsigned row=0;row<32;row++)for(unsigned col=0;col<32;col++)
              fprintf(stderr," %08x",((uint32_t*)output)[row*384u+n*32u+col]);
            fputc('\n',stderr);
            if(status || outword || changed!=(n+1u)*1024u || !finite ||
               !within || !rest || !lower || !inputs || !query_ok ||
               km[2*n]!=0x110000u+2u*n ||
               km[2*n+1]!=0x110001u+2u*n || !no_agx_metal())return 2;
          }
          fprintf(stderr,"PURE FFN KEY ALL COMPLETE: 32x384 key projection and resident query are intact.\n");
          free(query_snapshot);
        }
        if(ffn_scores){
          uint8_t* query=mapped[17]+0x13000u;
          uint8_t* key=mapped[2]+0x14000u;
          uint8_t* output=ffn_scores_relocated?mapped[17]+0x1000u:mapped[2]+0x8000u;
          uint8_t left[64],right[64];
          memcpy(left,output-64,64);memcpy(right,output+49152u,64);
          if(memcmp(query,scores_query_expected,49152u) ||
             memcmp(key,scores_key_expected,49152u) ||
             (ffn_scores_relocated && memcmp(mapped[2]+0x8000u,query_source_expected,49152u)) ||
             sizes[2]!=0x20000u || !no_agx_metal())return 2;
          for(unsigned i=0;i<12288;i++)((uint32_t*)output)[i]=0x7fc01234u;
          uint32_t scratch_word[2]={0x0c000007u,0};
          uint32_t scalar_packet=0x0e4006c7u;
          uint32_t x_threads=32u,y_threads=384u;
          const uint64_t bindings[3]={0x1000006b000ull,0x10000044000ull,
                                      ffn_scores_relocated?0x10000059000ull:0x10000038000ull};
          memcpy(mapped[0]+0x6c0,scores_code,sizeof scores_code);
          memcpy(mapped[23]+0x38,scratch_word,8);
          memcpy(mapped[23]+0x40,&scalar_packet,4);
          memcpy(mapped[22]+0xa8,&x_threads,4);
          memcpy(mapped[22]+0xac,&y_threads,4);
          memcpy(mapped[25]+0x10,&x_threads,4);
          memcpy(mapped[25]+0x14,&y_threads,4);
          memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
          if(memcmp(mapped[0]+0x6c0,scores_code,sizeof scores_code) ||
             memcmp(mapped[23]+0x38,scratch_word,8) ||
             memcmp(mapped[22]+0xac,&y_threads,4) ||
             memcmp(mapped[25]+0x14,&y_threads,4) ||
             memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t score_marks[2]={0};
          volatile uint64_t* sm=score_marks;
          void (^scheduled)(void)=Block_copy(^{sm[0]=0x120000u;});
          void (^completed)(void)=Block_copy(^{sm[1]=0x120001u;});
          if(!scheduled || !completed || scheduled==completed)return 2;
          uint8_t record[64]={0},out[64]={0};
          uint32_t kid=shmem[1].id,sid=shmem[0].id;
          uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
          memcpy(record,&kid,4);memcpy(record+4,&sid,4);
          memcpy(record+0x10,&sp,sizeof sp);
          memcpy(record+0x18,&cp,sizeof cp);
          int status=Submit(queue,NULL,1,record,64,out);
          unsigned waited=0;for(;waited<3000 && !sm[1];waited++)usleep(10000);
          unsigned changed=0,finite=1,within=1;
          double max_abs=0,max_fraction=0;
          for(unsigned i=0;i<12288;i++){
            uint32_t word=((uint32_t*)output)[i];
            changed+=word!=0x7fc01234u;
            float value=0;memcpy(&value,&word,4);
            finite&=isfinite(value);
            double error=fabs((double)value-scores_reference[i]);
            double limit=2e-5*(1.0+fabs(scores_reference[i]));
            within&=isfinite(value) && error<=limit;
            if(error>max_abs)max_abs=error;
            if(error/limit>max_fraction)max_fraction=error/limit;
          }
          unsigned guards=memcmp(output-64,left,64)==0 &&
                          memcmp(output+49152u,right,64)==0;
          unsigned inputs=memcmp(query,scores_query_expected,49152u)==0 &&
                          memcmp(key,scores_key_expected,49152u)==0 &&
                          (!ffn_scores_relocated ||
                           memcmp(mapped[2]+0x8000u,query_source_expected,49152u)==0);
          uint32_t outword=0;memcpy(&outword,out,4);
          fprintf(stderr,"PURE FFN SCORES returned status=%d outword=0x%08x changed=%u/12288 finite=%u within=%u guards=%u inputs=%u max_abs=%.9g max_fraction=%.9g marks=%llu/%llu wait=%u/3000.\n",
                  status,outword,changed,finite,within,guards,inputs,
                  max_abs,max_fraction,(unsigned long long)sm[0],
                  (unsigned long long)sm[1],waited);
          fprintf(stderr,"PURE FFN SCORES words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)output)[i]);
          fputc('\n',stderr);
          if(status || outword || changed!=12288u || !finite || !within ||
             !guards || !inputs || sm[0]!=0x120000u ||
             sm[1]!=0x120001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE FFN SCORES COMPLETE: 12x32x32 attention logits from resident Q/K.\n");
        }
        if(ffn_softmax){
          uint8_t* scores=mapped[17]+0x1000u;
          uint8_t* output=mapped[17]+0x13000u;
          uint8_t left[64],right[64];
          memcpy(left,output-64,64);memcpy(right,output+49152u,64);
          if(memcmp(scores,softmax_scores_expected,49152u) ||
             memcmp(mapped[2]+0x8000u,query_source_expected,49152u) ||
             memcmp(mapped[2]+0x14000u,scores_key_expected,49152u) ||
             !no_agx_metal())return 2;
          for(unsigned i=0;i<12288;i++)((uint32_t*)output)[i]=0x7fc01234u;
          uint32_t scratch_word[2]={0x0c000007u,0};
          uint32_t scalar_packet=0x0e4006c7u;
          uint32_t x_threads=384u,y_threads=1u;
          const uint64_t bindings[3]={0x10000059000ull,0x1000006b000ull,0};
          memcpy(mapped[0]+0x6c0,softmax_code,sizeof softmax_code);
          memcpy(mapped[23]+0x38,scratch_word,8);
          memcpy(mapped[23]+0x40,&scalar_packet,4);
          memcpy(mapped[22]+0xa8,&x_threads,4);
          memcpy(mapped[22]+0xac,&y_threads,4);
          memcpy(mapped[25]+0x10,&x_threads,4);
          memcpy(mapped[25]+0x14,&y_threads,4);
          memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
          if(memcmp(mapped[0]+0x6c0,softmax_code,sizeof softmax_code) ||
             memcmp(mapped[22]+0xa8,&x_threads,4) ||
             memcmp(mapped[25]+0x10,&x_threads,4) ||
             memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t softmax_marks[2]={0};
          volatile uint64_t* mm=softmax_marks;
          void (^scheduled)(void)=Block_copy(^{mm[0]=0x130000u;});
          void (^completed)(void)=Block_copy(^{mm[1]=0x130001u;});
          if(!scheduled || !completed || scheduled==completed)return 2;
          uint8_t record[64]={0},out[64]={0};
          uint32_t kid=shmem[1].id,sid=shmem[0].id;
          uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
          memcpy(record,&kid,4);memcpy(record+4,&sid,4);
          memcpy(record+0x10,&sp,sizeof sp);
          memcpy(record+0x18,&cp,sizeof cp);
          int status=Submit(queue,NULL,1,record,64,out);
          unsigned waited=0;for(;waited<3000 && !mm[1];waited++)usleep(10000);
          unsigned changed=0,finite=1,within=1,range=1,rows=1;
          double max_abs=0,max_fraction=0,max_row_error=0;
          for(unsigned row=0;row<384;row++){
            double sum=0;
            for(unsigned col=0;col<32;col++){
              unsigned i=row*32u+col;
              uint32_t word=((uint32_t*)output)[i];
              changed+=word!=0x7fc01234u;
              float value=0;memcpy(&value,&word,4);
              finite&=isfinite(value);range&=value>=0.0f && value<=1.0f;
              sum+=(double)value;
              double error=fabs((double)value-softmax_reference[i]);
              double limit=2e-5*(1.0+fabs(softmax_reference[i]));
              within&=isfinite(value) && error<=limit;
              if(error>max_abs)max_abs=error;
              if(error/limit>max_fraction)max_fraction=error/limit;
            }
            double row_error=fabs(sum-1.0);
            rows&=row_error<=1e-4;
            if(row_error>max_row_error)max_row_error=row_error;
          }
          unsigned guards=memcmp(output-64,left,64)==0 &&
                          memcmp(output+49152u,right,64)==0;
          unsigned inputs=memcmp(scores,softmax_scores_expected,49152u)==0 &&
                          memcmp(mapped[2]+0x8000u,query_source_expected,49152u)==0 &&
                          memcmp(mapped[2]+0x14000u,scores_key_expected,49152u)==0;
          uint32_t outword=0;memcpy(&outword,out,4);
          fprintf(stderr,"PURE FFN SOFTMAX returned status=%d outword=0x%08x changed=%u/12288 finite=%u within=%u range=%u rows=%u guards=%u inputs=%u max_abs=%.9g max_fraction=%.9g max_row_error=%.9g marks=%llu/%llu wait=%u/3000.\n",
                  status,outword,changed,finite,within,range,rows,guards,inputs,
                  max_abs,max_fraction,max_row_error,(unsigned long long)mm[0],
                  (unsigned long long)mm[1],waited);
          fprintf(stderr,"PURE FFN SOFTMAX words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)output)[i]);
          fputc('\n',stderr);
          if(status || outword || changed!=12288u || !finite || !within ||
             !range || !rows || !guards || !inputs ||
             mm[0]!=0x130000u || mm[1]!=0x130001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE FFN SOFTMAX COMPLETE: 384 attention rows normalized from resident logits.\n");
        }
        if(ffn_value_all){
          uint8_t* source=mapped[2]+0x8000u;
          uint8_t* output=mapped[2]+0x14000u;
          uint8_t* weight=mapped[17]+0x1000u;
          uint8_t* probabilities=mapped[17]+0x13000u;
          uint8_t lower_guard[64];memcpy(lower_guard,output-64,64);
          if(memcmp(source,query_source_expected,49152u) ||
             memcmp(probabilities,value_probabilities_expected,49152u) ||
             sizes[2]!=0x20000u || output+49152u!=mapped[2]+sizes[2] ||
             weight+49280u>probabilities-64 || !no_agx_metal())return 2;
          for(unsigned i=0;i<12288;i++)((uint32_t*)output)[i]=0x7fc01234u;
          static volatile uint64_t value_marks[24]={0};
          volatile uint64_t* vm=value_marks;
          for(unsigned n=0;n<12;n++){
            memcpy(weight,value_packed+n*49280u,49280u);
            uint64_t bindings[3]={0x10000038000ull,0x10000059000ull,
                                  0x10000044000ull+(uint64_t)n*128u};
            uint32_t scratch_word[2]={0x0c000007u,0};
            uint32_t scalar_packet=0x0e4006c7u;
            uint32_t xy_threads=32u;
            memcpy(mapped[0]+0x6c0,query_code,sizeof query_code);
            memcpy(mapped[23]+0x38,scratch_word,8);
            memcpy(mapped[23]+0x40,&scalar_packet,4);
            memcpy(mapped[22]+0xa8,&xy_threads,4);
            memcpy(mapped[22]+0xac,&xy_threads,4);
            memcpy(mapped[25]+0x10,&xy_threads,4);
            memcpy(mapped[25]+0x14,&xy_threads,4);
            memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
            if(memcmp(mapped[0]+0x6c0,query_code,sizeof query_code) ||
               memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
               memcmp(source,query_source_expected,49152u) ||
               memcmp(probabilities,value_probabilities_expected,49152u) ||
               memcmp(weight,value_packed+n*49280u,49280u) ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            void (^scheduled)(void)=Block_copy(^{vm[2*n]=0x140000u+2u*n;});
            void (^completed)(void)=Block_copy(^{vm[2*n+1]=0x140001u+2u*n;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t record[64]={0},out[64]={0};
            uint32_t kid=shmem[1].id,sid=shmem[0].id;
            uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
            memcpy(record,&kid,4);memcpy(record+4,&sid,4);
            memcpy(record+0x10,&sp,sizeof sp);
            memcpy(record+0x18,&cp,sizeof cp);
            int status=Submit(queue,NULL,1,record,64,out);
            unsigned waited=0;for(;waited<3000 && !vm[2*n+1];waited++)usleep(10000);
            unsigned changed=0,finite=1,within=1,rest=1;
            double max_abs=0,max_fraction=0;
            for(unsigned row=0;row<32;row++)for(unsigned col=0;col<384;col++){
              uint32_t word=((uint32_t*)output)[row*384u+col];
              if(col>=(n+1u)*32u){rest&=word==0x7fc01234u;continue;}
              unsigned tile=col/32u,within_tile=col%32u;
              changed+=word!=0x7fc01234u;
              float value=0;memcpy(&value,&word,4);
              finite&=isfinite(value);
              double expected=value_reference[tile*1024u+row*32u+within_tile];
              double error=fabs((double)value-expected);
              double limit=2e-5*(1.0+fabs(expected));
              within&=isfinite(value) && error<=limit;
              if(error>max_abs)max_abs=error;
              if(error/limit>max_fraction)max_fraction=error/limit;
            }
            unsigned lower=memcmp(output-64,lower_guard,64)==0;
            unsigned inputs=memcmp(source,query_source_expected,49152u)==0 &&
                            memcmp(weight,value_packed+n*49280u,49280u)==0 &&
                            memcmp(probabilities,value_probabilities_expected,49152u)==0;
            uint32_t outword=0;memcpy(&outword,out,4);
            fprintf(stderr,"PURE FFN VALUE ALL n=%u returned status=%d outword=0x%08x changed=%u/%u finite=%u within=%u rest=%u lower=%u inputs=%u max_abs=%.9g max_fraction=%.9g marks=%llu/%llu wait=%u/3000.\n",
                    n,status,outword,changed,(n+1u)*1024u,finite,within,rest,lower,inputs,
                    max_abs,max_fraction,(unsigned long long)vm[2*n],
                    (unsigned long long)vm[2*n+1],waited);
            fprintf(stderr,"PURE FFN VALUE ALL n=%u words:",n);
            for(unsigned row=0;row<32;row++)for(unsigned col=0;col<32;col++)
              fprintf(stderr," %08x",((uint32_t*)output)[row*384u+n*32u+col]);
            fputc('\n',stderr);
            if(status || outword || changed!=(n+1u)*1024u || !finite ||
               !within || !rest || !lower || !inputs ||
               vm[2*n]!=0x140000u+2u*n ||
               vm[2*n+1]!=0x140001u+2u*n || !no_agx_metal())return 2;
          }
          fprintf(stderr,"PURE FFN VALUE ALL COMPLETE: 32x384 value projection retained probabilities and LayerNorm source.\n");
        }
        if(ffn_context){
          uint8_t* probabilities=mapped[17]+0x13000u;
          uint8_t* value=mapped[2]+0x14000u;
          uint8_t* output=ffn_context_relocated?mapped[17]+0x1000u:mapped[2]+0x8000u;
          uint8_t left[64],right[64];
          memcpy(left,output-64,64);memcpy(right,output+49152u,64);
          if(memcmp(probabilities,context_probabilities_expected,49152u) ||
             memcmp(value,context_value_expected,49152u) ||
             (ffn_context_relocated &&
              memcmp(mapped[2]+0x8000u,query_source_expected,49152u)) ||
             !no_agx_metal())return 2;
          for(unsigned i=0;i<12288;i++)((uint32_t*)output)[i]=0x7fc01234u;
          uint32_t scratch_word[2]={0x0c000007u,0};
          uint32_t scalar_packet=0x0e4006c7u;
          uint32_t x_threads=384u,y_threads=32u;
          const uint64_t bindings[3]={0x1000006b000ull,0x10000044000ull,
                                      ffn_context_relocated?0x10000059000ull:0x10000038000ull};
          memcpy(mapped[0]+0x6c0,context_code,sizeof context_code);
          memcpy(mapped[23]+0x38,scratch_word,8);
          memcpy(mapped[23]+0x40,&scalar_packet,4);
          memcpy(mapped[22]+0xa8,&x_threads,4);
          memcpy(mapped[22]+0xac,&y_threads,4);
          memcpy(mapped[25]+0x10,&x_threads,4);
          memcpy(mapped[25]+0x14,&y_threads,4);
          memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
          if(memcmp(mapped[0]+0x6c0,context_code,sizeof context_code) ||
             memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t context_marks[2]={0};
          volatile uint64_t* cm=context_marks;
          void (^scheduled)(void)=Block_copy(^{cm[0]=0x150000u;});
          void (^completed)(void)=Block_copy(^{cm[1]=0x150001u;});
          if(!scheduled || !completed || scheduled==completed)return 2;
          uint8_t record[64]={0},out[64]={0};
          uint32_t kid=shmem[1].id,sid=shmem[0].id;
          uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
          memcpy(record,&kid,4);memcpy(record+4,&sid,4);
          memcpy(record+0x10,&sp,sizeof sp);
          memcpy(record+0x18,&cp,sizeof cp);
          int status=Submit(queue,NULL,1,record,64,out);
          unsigned waited=0;for(;waited<3000 && !cm[1];waited++)usleep(10000);
          unsigned changed=0,finite=1,within=1;
          double max_abs=0,max_fraction=0;
          for(unsigned i=0;i<12288;i++){
            uint32_t word=((uint32_t*)output)[i];
            changed+=word!=0x7fc01234u;
            float result=0;memcpy(&result,&word,4);
            finite&=isfinite(result);
            double error=fabs((double)result-context_reference[i]);
            double limit=2e-5*(1.0+fabs(context_reference[i]));
            within&=isfinite(result) && error<=limit;
            if(error>max_abs)max_abs=error;
            if(error/limit>max_fraction)max_fraction=error/limit;
          }
          unsigned guards=memcmp(output-64,left,64)==0 &&
                          memcmp(output+49152u,right,64)==0;
          unsigned inputs=memcmp(probabilities,context_probabilities_expected,49152u)==0 &&
                          memcmp(value,context_value_expected,49152u)==0 &&
                          (!ffn_context_relocated ||
                           memcmp(mapped[2]+0x8000u,query_source_expected,49152u)==0);
          uint32_t outword=0;memcpy(&outword,out,4);
          fprintf(stderr,"PURE FFN CONTEXT returned status=%d outword=0x%08x changed=%u/12288 finite=%u within=%u guards=%u inputs=%u max_abs=%.9g max_fraction=%.9g marks=%llu/%llu wait=%u/3000.\n",
                  status,outword,changed,finite,within,guards,inputs,max_abs,
                  max_fraction,(unsigned long long)cm[0],
                  (unsigned long long)cm[1],waited);
          fprintf(stderr,"PURE FFN CONTEXT words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)output)[i]);
          fputc('\n',stderr);
          if(status || outword || changed!=12288u || !finite || !within ||
             !guards || !inputs || cm[0]!=0x150000u ||
             cm[1]!=0x150001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE FFN CONTEXT COMPLETE: resident probabilities and V produced 32x384 context.\n");
        }
        if(ffn_attention_output_all){
          uint8_t* source=mapped[17]+0x1000u;
          uint8_t* weight=mapped[17]+0x13000u;
          uint8_t* output=mapped[2]+0x14000u;
          uint8_t lower_guard[64];memcpy(lower_guard,output-64,64);
          if(memcmp(source,attention_context_expected,49152u) ||
             memcmp(mapped[2]+0x8000u,query_source_expected,49152u) ||
             weight+49280u>mapped[17]+sizes[17] ||
             output+49152u!=mapped[2]+sizes[2] || !no_agx_metal())return 2;
          for(unsigned i=0;i<12288;i++)((uint32_t*)output)[i]=0x7fc01234u;
          static volatile uint64_t output_marks[24]={0};
          volatile uint64_t* om=output_marks;
          for(unsigned n=0;n<12;n++){
            memcpy(weight,attention_output_packed+n*49280u,49280u);
            uint64_t bindings[3]={0x10000059000ull,0x1000006b000ull,
                                  0x10000044000ull+(uint64_t)n*128u};
            uint32_t scratch_word[2]={0x0c000007u,0};
            uint32_t scalar_packet=0x0e4006c7u;
            uint32_t xy_threads=32u;
            memcpy(mapped[0]+0x6c0,query_code,sizeof query_code);
            memcpy(mapped[23]+0x38,scratch_word,8);
            memcpy(mapped[23]+0x40,&scalar_packet,4);
            memcpy(mapped[22]+0xa8,&xy_threads,4);
            memcpy(mapped[22]+0xac,&xy_threads,4);
            memcpy(mapped[25]+0x10,&xy_threads,4);
            memcpy(mapped[25]+0x14,&xy_threads,4);
            memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
            if(memcmp(mapped[0]+0x6c0,query_code,sizeof query_code) ||
               memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
               memcmp(source,attention_context_expected,49152u) ||
               memcmp(mapped[2]+0x8000u,query_source_expected,49152u) ||
               memcmp(weight,attention_output_packed+n*49280u,49280u) ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            void (^scheduled)(void)=Block_copy(^{om[2*n]=0x160000u+2u*n;});
            void (^completed)(void)=Block_copy(^{om[2*n+1]=0x160001u+2u*n;});
            if(!scheduled || !completed || scheduled==completed)return 2;
            uint8_t record[64]={0},out[64]={0};
            uint32_t kid=shmem[1].id,sid=shmem[0].id;
            uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
            memcpy(record,&kid,4);memcpy(record+4,&sid,4);
            memcpy(record+0x10,&sp,sizeof sp);
            memcpy(record+0x18,&cp,sizeof cp);
            int status=Submit(queue,NULL,1,record,64,out);
            unsigned waited=0;for(;waited<3000 && !om[2*n+1];waited++)usleep(10000);
            unsigned changed=0,finite=1,within=1,rest=1;
            double max_abs=0,max_fraction=0;
            for(unsigned row=0;row<32;row++)for(unsigned col=0;col<384;col++){
              uint32_t word=((uint32_t*)output)[row*384u+col];
              if(col>=(n+1u)*32u){rest&=word==0x7fc01234u;continue;}
              unsigned tile=col/32u,within_tile=col%32u;
              changed+=word!=0x7fc01234u;
              float result=0;memcpy(&result,&word,4);
              finite&=isfinite(result);
              double expected=attention_output_reference[tile*1024u+row*32u+within_tile];
              double error=fabs((double)result-expected);
              double limit=2e-5*(1.0+fabs(expected));
              within&=isfinite(result) && error<=limit;
              if(error>max_abs)max_abs=error;
              if(error/limit>max_fraction)max_fraction=error/limit;
            }
            unsigned lower=memcmp(output-64,lower_guard,64)==0;
            unsigned inputs=memcmp(source,attention_context_expected,49152u)==0 &&
                            memcmp(mapped[2]+0x8000u,query_source_expected,49152u)==0 &&
                            memcmp(weight,attention_output_packed+n*49280u,49280u)==0;
            uint32_t outword=0;memcpy(&outword,out,4);
            fprintf(stderr,"PURE FFN ATTENTION OUTPUT ALL n=%u returned status=%d outword=0x%08x changed=%u/%u finite=%u within=%u rest=%u lower=%u inputs=%u max_abs=%.9g max_fraction=%.9g marks=%llu/%llu wait=%u/3000.\n",
                    n,status,outword,changed,(n+1u)*1024u,finite,within,rest,lower,inputs,
                    max_abs,max_fraction,(unsigned long long)om[2*n],
                    (unsigned long long)om[2*n+1],waited);
            fprintf(stderr,"PURE FFN ATTENTION OUTPUT ALL n=%u words:",n);
            for(unsigned row=0;row<32;row++)for(unsigned col=0;col<32;col++)
              fprintf(stderr," %08x",((uint32_t*)output)[row*384u+n*32u+col]);
            fputc('\n',stderr);
            if(status || outword || changed!=(n+1u)*1024u || !finite ||
               !within || !rest || !lower || !inputs ||
               om[2*n]!=0x160000u+2u*n ||
               om[2*n+1]!=0x160001u+2u*n || !no_agx_metal())return 2;
          }
          fprintf(stderr,"PURE FFN ATTENTION OUTPUT ALL COMPLETE: 32x384 output projection retained resident context and LayerNorm source.\n");
        }
        if(ffn_attention_residual){
          uint8_t* source=mapped[2]+0x8000u;
          uint8_t* output=mapped[2]+0x14000u;
          uint8_t* context=mapped[17]+0x1000u;
          uint8_t lower_guard[64];memcpy(lower_guard,output-64,64);
          if(memcmp(source,query_source_expected,49152u) ||
             memcmp(output,attention_projected_expected,49152u) ||
             memcmp(context,attention_context_expected,49152u) ||
             !no_agx_metal())return 2;
          uint32_t scratch_word[2]={0x0c000007u,0};
          uint32_t scalar_packet=0x0e4006c7u;
          uint32_t x_threads=384u,y_threads=32u;
          const uint64_t bindings[3]={0x10000038000ull,0x10000044000ull,
                                      0x10000044000ull};
          memcpy(mapped[0]+0x6c0,attention_residual_code,sizeof attention_residual_code);
          memcpy(mapped[23]+0x38,scratch_word,8);
          memcpy(mapped[23]+0x40,&scalar_packet,4);
          memcpy(mapped[22]+0xa8,&x_threads,4);
          memcpy(mapped[22]+0xac,&y_threads,4);
          memcpy(mapped[25]+0x10,&x_threads,4);
          memcpy(mapped[25]+0x14,&y_threads,4);
          memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
          if(memcmp(mapped[0]+0x6c0,attention_residual_code,sizeof attention_residual_code) ||
             memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t attention_residual_marks[2]={0};
          volatile uint64_t* arm=attention_residual_marks;
          void (^scheduled)(void)=Block_copy(^{arm[0]=0x170000u;});
          void (^completed)(void)=Block_copy(^{arm[1]=0x170001u;});
          if(!scheduled || !completed || scheduled==completed)return 2;
          uint8_t record[64]={0},out[64]={0};
          uint32_t kid=shmem[1].id,sid=shmem[0].id;
          uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
          memcpy(record,&kid,4);memcpy(record+4,&sid,4);
          memcpy(record+0x10,&sp,sizeof sp);
          memcpy(record+0x18,&cp,sizeof cp);
          int status=Submit(queue,NULL,1,record,64,out);
          unsigned waited=0;for(;waited<3000 && !arm[1];waited++)usleep(10000);
          unsigned changed=0,finite=1;
          for(unsigned i=0;i<12288;i++){
            uint32_t actual=((uint32_t*)output)[i];
            uint32_t prior=((uint32_t*)attention_projected_expected)[i];
            changed+=actual!=prior;
            float value=0;memcpy(&value,&actual,4);finite&=isfinite(value);
          }
          unsigned exact=memcmp(output,attention_residual_expected,49152u)==0;
          unsigned lower=memcmp(output-64,lower_guard,64)==0;
          unsigned inputs=memcmp(source,query_source_expected,49152u)==0 &&
                          memcmp(context,attention_context_expected,49152u)==0;
          uint32_t outword=0;memcpy(&outword,out,4);
          fprintf(stderr,"PURE FFN ATTENTION RESIDUAL returned status=%d outword=0x%08x changed=%u/12288 finite=%u exact=%u lower=%u inputs=%u marks=%llu/%llu wait=%u/3000.\n",
                  status,outword,changed,finite,exact,lower,inputs,
                  (unsigned long long)arm[0],(unsigned long long)arm[1],waited);
          fprintf(stderr,"PURE FFN ATTENTION RESIDUAL words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)output)[i]);
          fputc('\n',stderr);
          if(status || outword || changed!=12288u || !finite || !exact ||
             !lower || !inputs || arm[0]!=0x170000u ||
             arm[1]!=0x170001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE FFN ATTENTION RESIDUAL COMPLETE: saved LayerNorm input added to resident output projection.\n");
        }
        if(ffn_attention_layernorm){
          uint8_t* source=mapped[2]+0x14000u;
          uint8_t* parameters=mapped[2]+0x4d80u;
          uint8_t* output=mapped[17]+0x1000u;
          uint8_t left[64],right[64];
          memcpy(left,output-64,64);memcpy(right,output+49152u,64);
          if(memcmp(source,attention_residual_expected,49152u) ||
             !no_agx_metal())return 2;
          memcpy(parameters,attention_layernorm_parameters,sizeof attention_layernorm_parameters);
          for(unsigned i=0;i<12288;i++)((uint32_t*)output)[i]=0x7fc01234u;
          uint32_t scratch_word[2]={0x0c00100fu,0};
          uint32_t tensor_packet=0x0e5806c7u;
          uint32_t xy_threads=32u;
          const uint64_t bindings[4]={0x10000044000ull,0x10000034d80ull,
                                      0x10000059000ull,0};
          memcpy(mapped[0]+0x6c0,attention_layernorm_code,sizeof attention_layernorm_code);
          memcpy(mapped[23]+0x38,scratch_word,8);
          memcpy(mapped[23]+0x40,&tensor_packet,4);
          memcpy(mapped[22]+0xa8,&xy_threads,4);
          memcpy(mapped[22]+0xac,&xy_threads,4);
          memcpy(mapped[25]+0x10,&xy_threads,4);
          memcpy(mapped[25]+0x14,&xy_threads,4);
          memcpy(mapped[28]+0x1ba0,bindings,sizeof bindings);
          if(memcmp(mapped[0]+0x6c0,attention_layernorm_code,sizeof attention_layernorm_code) ||
             memcmp(mapped[23]+0x38,scratch_word,8) ||
             memcmp(mapped[23]+0x40,&tensor_packet,4) ||
             memcmp(mapped[28]+0x1ba0,bindings,sizeof bindings) ||
             memcmp(parameters,attention_layernorm_parameters,sizeof attention_layernorm_parameters) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t attention_ln_marks[2]={0};
          volatile uint64_t* alm=attention_ln_marks;
          void (^scheduled)(void)=Block_copy(^{alm[0]=0x180000u;});
          void (^completed)(void)=Block_copy(^{alm[1]=0x180001u;});
          if(!scheduled || !completed || scheduled==completed)return 2;
          uint8_t record[64]={0},out[64]={0};
          uint32_t kid=shmem[1].id,sid=shmem[0].id;
          uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
          memcpy(record,&kid,4);memcpy(record+4,&sid,4);
          memcpy(record+0x10,&sp,sizeof sp);
          memcpy(record+0x18,&cp,sizeof cp);
          int status=Submit(queue,NULL,1,record,64,out);
          unsigned waited=0;for(;waited<3000 && !alm[1];waited++)usleep(10000);
          unsigned changed=0,finite=1,within=1;
          double max_abs=0,max_fraction=0;
          for(unsigned i=0;i<12288;i++){
            uint32_t word=((uint32_t*)output)[i];
            changed+=word!=0x7fc01234u;
            float result=0;memcpy(&result,&word,4);
            finite&=isfinite(result);
            double error=fabs((double)result-attention_layernorm_reference[i]);
            double limit=2e-5*(1.0+fabs(attention_layernorm_reference[i]));
            within&=isfinite(result) && error<=limit;
            if(error>max_abs)max_abs=error;
            if(error/limit>max_fraction)max_fraction=error/limit;
          }
          unsigned guards=memcmp(output-64,left,64)==0 &&
                          memcmp(output+49152u,right,64)==0;
          unsigned inputs=memcmp(source,attention_residual_expected,49152u)==0 &&
                          memcmp(parameters,attention_layernorm_parameters,
                                 sizeof attention_layernorm_parameters)==0;
          uint32_t outword=0;memcpy(&outword,out,4);
          fprintf(stderr,"PURE FFN ATTENTION LAYERNORM returned status=%d outword=0x%08x changed=%u/12288 finite=%u within=%u guards=%u inputs=%u max_abs=%.9g max_fraction=%.9g marks=%llu/%llu wait=%u/3000.\n",
                  status,outword,changed,finite,within,guards,inputs,max_abs,
                  max_fraction,(unsigned long long)alm[0],
                  (unsigned long long)alm[1],waited);
          fprintf(stderr,"PURE FFN ATTENTION LAYERNORM words:");
          for(unsigned i=0;i<12288;i++)fprintf(stderr," %08x",((uint32_t*)output)[i]);
          fputc('\n',stderr);
          if(status || outword || changed!=12288u || !finite || !within ||
             !guards || !inputs || alm[0]!=0x180000u ||
             alm[1]!=0x180001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE FFN ATTENTION LAYERNORM COMPLETE: resident residual normalized with model gamma/beta.\n");
        }
        if(ffn_contract_tile){
          uint8_t* scratch=mapped[17]+0x1000u;
          uint32_t* output=(uint32_t*)(uintptr_t)
            (ffn_contract_alloc0?mapped[0]+0x1000u:
             ffn_contract_alloc1?mapped[1]+0x1000u:mapped[17]+0x3000u);
          if(memcmp(scratch,pack_full_expected+23u*4096u,4096))return 2;
          uint8_t left[64],right[64];
          memcpy(left,(uint8_t*)output-64,64);
          memcpy(right,(uint8_t*)output+4096,64);
          memcpy(mapped[2]+0x5e80,contract_b,4096);
          uint32_t tensor_packet=0x0e5806c7u;
          uint64_t a_va=0x10000059000ull,b_va=0x10000035e80ull,
                   c_va=ffn_contract_alloc0?0x10000001000ull:
                        ffn_contract_alloc1?0x10000019000ull:0x1000005b000ull;
          memcpy(mapped[0]+0x6c0,expected_inputs[0],1458);
          memcpy(mapped[23]+0x40,&tensor_packet,4);
          memcpy(mapped[28]+0x1ba0,&a_va,8);
          memcpy(mapped[28]+0x1ba8,&b_va,8);
          memcpy(mapped[28]+0x1bb0,&c_va,8);
          if(memcmp(mapped[0]+0x6c0,expected_inputs[0],1458) ||
             memcmp(mapped[2]+0x5e80,contract_b,4096) ||
             memcmp(mapped[28]+0x1ba0,&a_va,8) ||
             memcmp(mapped[28]+0x1ba8,&b_va,8) ||
             memcmp(mapped[28]+0x1bb0,&c_va,8) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t contract_marks[2]={0};
          volatile uint64_t* cm=contract_marks;
          void (^scheduled_contract)(void)=Block_copy(^{cm[0]=0x30000u;});
          void (^completed_contract)(void)=Block_copy(^{cm[1]=0x30001u;});
          if(!scheduled_contract || !completed_contract || scheduled_contract==completed_contract)return 2;
          uint8_t record_contract[64]={0},out_contract[64]={0};
          uint32_t kid=shmem[1].id,sid=shmem[0].id;
          uintptr_t sp=(uintptr_t)scheduled_contract,cp=(uintptr_t)completed_contract;
          memcpy(record_contract,&kid,4);memcpy(record_contract+4,&sid,4);
          memcpy(record_contract+0x10,&sp,sizeof sp);memcpy(record_contract+0x18,&cp,sizeof cp);
          fprintf(stderr,"PURE FFN CONTRACT TILE entering: A=0x%llx B=0x%llx C=0x%llx packet=0x%08x ready=0x%08x.\n",
                  (unsigned long long)a_va,(unsigned long long)b_va,(unsigned long long)c_va,
                  tensor_packet,*full_ready);
          int status=Submit(queue,NULL,1,record_contract,64,out_contract);
          unsigned waited=0;for(;waited<3000 && !cm[1];waited++)usleep(10000);
          unsigned changed=0,finite=1;
          for(unsigned i=0;i<1024;i++){
            changed+=output[i]!=0u;
            float value=0;memcpy(&value,output+i,4);finite&=isfinite(value);
          }
          unsigned guards=memcmp((uint8_t*)output-64,left,64)==0 &&
                          memcmp((uint8_t*)output+4096,right,64)==0;
          unsigned inputs=memcmp(scratch,pack_full_expected+23u*4096u,4096)==0 &&
                          memcmp(mapped[2]+0x5e80,contract_b,4096)==0;
          unsigned resident=1;for(unsigned p=0;p<48;p++)resident&=
            memcmp(slots[p],completed_tiles+p*1024u,4096)==0;
          uint32_t outword=0;memcpy(&outword,out_contract,4);
          fprintf(stderr,"PURE FFN CONTRACT TILE returned status=%d outword=0x%08x changed=%u/1024 finite=%u guards=%u inputs=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                  status,outword,changed,finite,guards,inputs,resident,
                  (unsigned long long)cm[0],(unsigned long long)cm[1],waited);
          fprintf(stderr,"PURE FFN CONTRACT TILE C words:");
          for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",output[i]);
          fputc('\n',stderr);
          if(status || outword || changed!=1024u || !finite || !guards ||
             !inputs || !resident || cm[0]!=0x30000u ||
             cm[1]!=0x30001u || !no_agx_metal())return 2;
          fprintf(stderr,"PURE FFN CONTRACT TILE COMPLETE: one GPU-produced packed K23 block consumed by tensor matmul.\n");
        }
        if(ffn_copy_packed){
          uint8_t* scratch=mapped[17]+0x1000u;
          uint8_t* zero_b=mapped[17]+0x2000u;
          uint8_t* target=ffn_copy_dead_tile?(uint8_t*)slots[46]:mapped[17]+0x3000u;
          uint8_t zero_block[4096]={0};
          uint8_t prior_target[4096];memcpy(prior_target,target,4096);
          uint8_t left[64],right[64];
          memcpy(left,target-64,64);memcpy(right,target+4096,64);
          if(memcmp(scratch,pack_full_expected+23u*4096u,4096) ||
             memcmp(zero_b,zero_block,4096) ||
             (!ffn_copy_dead_tile && memcmp(target,zero_block,4096)) ||
             (ffn_copy_dead_tile &&
              memcmp(target,completed_tiles+46u*1024u,4096)))return 2;
          uint32_t scalar_packet=0x0e4006c7u;
          memcpy(mapped[0]+0x6c0,copy_code,sizeof copy_code);
          memcpy(mapped[23]+0x40,&scalar_packet,4);
          if(memcmp(mapped[0]+0x6c0,copy_code,sizeof copy_code))return 2;
          fprintf(stderr,"PURE FFN GPU COPY target=0x%llx source=0x10000059000.\n",
                  (unsigned long long)(ffn_copy_dead_tile?0x10000076000ull:0x1000005b000ull));
          static volatile uint64_t copy_marks[64]={0};
          volatile uint64_t* cm=copy_marks;
          for(unsigned row=0;row<32;row++){
            uint64_t a_va=0x10000059000ull+(uint64_t)row*128u;
            uint64_t b_va=0x1000005a000ull+(uint64_t)row*128u;
            uint64_t c_va=(ffn_copy_dead_tile?0x10000076000ull:0x1000005b000ull)+
                          (uint64_t)row*128u;
            memcpy(mapped[28]+0x1ba0,&a_va,8);
            memcpy(mapped[28]+0x1ba8,&b_va,8);
            memcpy(mapped[28]+0x1bb0,&c_va,8);
            if(memcmp(mapped[28]+0x1ba0,&a_va,8) ||
               memcmp(mapped[28]+0x1ba8,&b_va,8) ||
               memcmp(mapped[28]+0x1bb0,&c_va,8) ||
               *full_ready!=0x800000f0u || !no_agx_metal())return 2;
            void (^scheduled_copy)(void)=Block_copy(^{cm[2*row]=0x40000u+2u*row;});
            void (^completed_copy)(void)=Block_copy(^{cm[2*row+1]=0x40001u+2u*row;});
            if(!scheduled_copy || !completed_copy || scheduled_copy==completed_copy)return 2;
            uint8_t record_copy[64]={0},out_copy[64]={0};
            uint32_t kid=shmem[1].id,sid=shmem[0].id;
            uintptr_t sp=(uintptr_t)scheduled_copy,cp=(uintptr_t)completed_copy;
            memcpy(record_copy,&kid,4);memcpy(record_copy+4,&sid,4);
            memcpy(record_copy+0x10,&sp,sizeof sp);memcpy(record_copy+0x18,&cp,sizeof cp);
            int status=Submit(queue,NULL,1,record_copy,64,out_copy);
            unsigned waited=0;for(;waited<3000 && !cm[2*row+1];waited++)usleep(10000);
            unsigned prefix=(row+1u)*128u;
            unsigned exact=memcmp(target,pack_full_expected+23u*4096u,prefix)==0 &&
                           memcmp(target+prefix,
                                  ffn_copy_dead_tile?prior_target+prefix:zero_block,
                                  4096u-prefix)==0;
            unsigned inputs=memcmp(scratch,pack_full_expected+23u*4096u,4096)==0 &&
                            memcmp(zero_b,zero_block,4096)==0;
            unsigned guards=memcmp(target-64,left,64)==0 &&
                            memcmp(target+4096,right,64)==0;
            unsigned resident=1;for(unsigned p=0;p<48;p++)if(!ffn_copy_dead_tile || p!=46u)
              resident&=memcmp(slots[p],completed_tiles+p*1024u,4096)==0;
            uint32_t outword=0;memcpy(&outword,out_copy,4);
            fprintf(stderr,"PURE FFN GPU COPY row=%u returned status=%d outword=0x%08x exact=%u inputs=%u guards=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                    row,status,outword,exact,inputs,guards,resident,
                    (unsigned long long)cm[2*row],(unsigned long long)cm[2*row+1],waited);
            if(status || outword || !exact || !inputs || !guards || !resident ||
               cm[2*row]!=0x40000u+2u*row || cm[2*row+1]!=0x40001u+2u*row ||
               !no_agx_metal())return 2;
          }
          fprintf(stderr,"PURE FFN GPU COPY C words:");
          for(unsigned i=0;i<1024;i++){
            uint32_t word=0;memcpy(&word,target+4u*i,4);
            fprintf(stderr," %08x",word);
          }
          fputc('\n',stderr);
          if(ffn_copy_dead_tile)
            fprintf(stderr,"PURE FFN DEAD TILE COMPLETE: n=46 repurposed from activation to packed A with other 47 tiles retained.\n");
          fprintf(stderr,"PURE FFN GPU COPY COMPLETE: GPU-packed K23 mirrored byte-exact in another registered 4 KiB slot.\n");
        }
        if(ffn_consume_reused){
          uint8_t* reused=(uint8_t*)slots[46];
          uint8_t* output=mapped[17]+0x3000u;
          uint8_t zero_block[4096]={0};
          uint8_t left[64],right[64];
          memcpy(left,output-64,64);memcpy(right,output+4096,64);
          if(memcmp(reused,pack_full_expected+23u*4096u,4096) ||
             memcmp(output,zero_block,4096))return 2;
          memcpy(mapped[2]+0x5e80,contract_b,4096);
          uint32_t tensor_packet=0x0e5806c7u;
          uint64_t a_va=0x10000076000ull,b_va=0x10000035e80ull,c_va=0x1000005b000ull;
          memcpy(mapped[0]+0x6c0,expected_inputs[0],1458);
          memcpy(mapped[23]+0x40,&tensor_packet,4);
          memcpy(mapped[28]+0x1ba0,&a_va,8);
          memcpy(mapped[28]+0x1ba8,&b_va,8);
          memcpy(mapped[28]+0x1bb0,&c_va,8);
          if(memcmp(mapped[0]+0x6c0,expected_inputs[0],1458) ||
             memcmp(mapped[2]+0x5e80,contract_b,4096) ||
             memcmp(mapped[28]+0x1ba0,&a_va,8) ||
             memcmp(mapped[28]+0x1ba8,&b_va,8) ||
             memcmp(mapped[28]+0x1bb0,&c_va,8) ||
             *full_ready!=0x800000f0u || !no_agx_metal())return 2;
          static volatile uint64_t consume_marks[2]={0};
          volatile uint64_t* cm=consume_marks;
          void (^scheduled_consume)(void)=Block_copy(^{cm[0]=0x50000u;});
          void (^completed_consume)(void)=Block_copy(^{cm[1]=0x50001u;});
          if(!scheduled_consume || !completed_consume || scheduled_consume==completed_consume)return 2;
          uint8_t record_consume[64]={0},out_consume[64]={0};
          uint32_t kid=shmem[1].id,sid=shmem[0].id;
          uintptr_t sp=(uintptr_t)scheduled_consume,cp=(uintptr_t)completed_consume;
          memcpy(record_consume,&kid,4);memcpy(record_consume+4,&sid,4);
          memcpy(record_consume+0x10,&sp,sizeof sp);memcpy(record_consume+0x18,&cp,sizeof cp);
          fprintf(stderr,"PURE FFN CONSUME REUSED entering: A=0x%llx B=0x%llx C=0x%llx.\n",
                  (unsigned long long)a_va,(unsigned long long)b_va,(unsigned long long)c_va);
          int status=Submit(queue,NULL,1,record_consume,64,out_consume);
          unsigned waited=0;for(;waited<3000 && !cm[1];waited++)usleep(10000);
          unsigned changed=0,finite=1;
          for(unsigned i=0;i<1024;i++){
            uint32_t word=0;memcpy(&word,output+4u*i,4);changed+=word!=0u;
            float value=0;memcpy(&value,output+4u*i,4);finite&=isfinite(value);
          }
          unsigned inputs=memcmp(reused,pack_full_expected+23u*4096u,4096)==0 &&
                          memcmp(mapped[2]+0x5e80,contract_b,4096)==0;
          unsigned guards=memcmp(output-64,left,64)==0 &&
                          memcmp(output+4096,right,64)==0;
          unsigned resident=1;for(unsigned p=0;p<48;p++)if(p!=46u)resident&=
            memcmp(slots[p],completed_tiles+p*1024u,4096)==0;
          uint32_t outword=0;memcpy(&outword,out_consume,4);
          fprintf(stderr,"PURE FFN CONSUME REUSED returned status=%d outword=0x%08x changed=%u/1024 finite=%u inputs=%u guards=%u resident=%u marks=%llu/%llu wait=%u/3000.\n",
                  status,outword,changed,finite,inputs,guards,resident,
                  (unsigned long long)cm[0],(unsigned long long)cm[1],waited);
          fprintf(stderr,"PURE FFN CONSUME REUSED C words:");
          for(unsigned i=0;i<1024;i++){
            uint32_t word=0;memcpy(&word,output+4u*i,4);fprintf(stderr," %08x",word);
          }
          fputc('\n',stderr);
          if(status || outword || changed!=1024u || !finite || !inputs ||
             !guards || !resident || cm[0]!=0x50000u || cm[1]!=0x50001u ||
             !no_agx_metal())return 2;
          fprintf(stderr,"PURE FFN CONSUME REUSED COMPLETE: relocated GPU-packed K23 consumed by tensor matmul.\n");
        }
        free(pack_full_expected);
        free(contract_b_all);
        free(contract_bias_values);free(contract_bias_expected);
        free(residual_source);free(residual_expected);
        free(contract_b_full);
        free(completed_tiles);free(full_b);
        return 0;
      }
      static volatile uint64_t ffn_marks[24]={0};
      volatile uint64_t* ffn_mark_ptr=ffn_marks;
      void (^scheduled)(void)=Block_copy(^{ffn_mark_ptr[0]=0x41e;});
      void (^completed)(void)=Block_copy(^{ffn_mark_ptr[1]=0x41f;});
      if(!scheduled || !completed || scheduled==completed)return 2;
      uint8_t record[64]={0},out[64]={0};
      uint32_t kernel_id=shmem[1].id,segment_id=shmem[0].id;
      uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
      memcpy(record,&kernel_id,4);memcpy(record+4,&segment_id,4);
      memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
      volatile uint32_t* ready_word=(volatile uint32_t*)(uintptr_t)(shmem[0].cpu+0x24);
      if(*ready_word!=0xf0u)return 2;
      *ready_word=0x800000f0u;
      fprintf(stderr,"PURE FFN entering: count=1 IDs=2/1 stride=64 ready=0x%08x.\n",*ready_word);
      int result=Submit(queue,NULL,1,record,64,out);
      unsigned waited=0;
      for(;waited<3000 && (!ffn_marks[1] || ffn_output[1023]==(ffn_acc?0u:0x7fc01234u));waited++)usleep(10000);
      unsigned changed=0;for(unsigned i=0;i<1024;i++)changed+=ffn_output[i]!=(ffn_acc?0u:0x7fc01234u);
      unsigned inputs_intact=memcmp(mapped[2]+0x4d80+shift,expected_inputs[1],4096)==0 &&
                            memcmp(mapped[2]+0x5e80+shift,expected_inputs[2],4096)==0;
      if(a_competition){
        for(unsigned i=0;i<0x80;i++)inputs_intact&=mapped[2][0x4d00+shift+i]==((i&1u)?0x3cu:0x00u);
      }
      if(b_competition){
        for(unsigned i=0;i<0x80;i++)inputs_intact&=mapped[2][0x5e00+shift+i]==((i&1u)?0x3cu:0x00u);
      }
      unsigned guards_intact=1;
      if(ffn_c_slot || ffn_c_cross)for(unsigned i=0;i<1024;i++)guards_intact&=c_view_words[32+i]==0x7fc01234u;
      if(ffn_c_cross || ffn_c_xcontrol)guards_intact&=ffn_cross_guards(mapped[2],mapped[17],ffn_c_cross);
      for(unsigned i=0;i<0x80;i++){
        guards_intact&=mapped[2][0x5d80+shift+i]==0xa5;
        guards_intact&=a_competition?mapped[2][0x4d00+shift+i]==((i&1u)?0x3cu:0x00u):
                                      mapped[2][0x4d00+shift+i]==0xa5;
        guards_intact&=mapped[2][0x6e80+shift+i]==0xa4;
        guards_intact&=b_competition?mapped[2][0x5e00+shift+i]==((i&1u)?0x3cu:0x00u):
                                      mapped[2][0x5e00+shift+i]==0xa4;
        if(!c_competition)guards_intact&=mapped[2][0x6f00+shift+i]==0xa7 && mapped[2][0x7f80+shift+i]==0xa7;
      }
      unsigned edge_intact=0;
      if(c_competition){
        uint32_t* edge=c_view_words+(c_binding_base?1024:0);
        for(unsigned i=0;i<32;i++)edge_intact+=edge[i]==0x7fc01234u;
        guards_intact&=edge_intact==32;
      }
      uint32_t outword=0;memcpy(&outword,out,4);
      fprintf(stderr,"PURE FFN returned status=%d outword=0x%08x changed=%u/1024 inputs=%u guards=%u marks=%llu/%llu wait=%u/3000.\n",
              result,outword,changed,inputs_intact,guards_intact,
              (unsigned long long)ffn_marks[0],(unsigned long long)ffn_marks[1],waited);
      if(c_competition)fprintf(stderr,"PURE FFN C edge=%u/32.\n",edge_intact);
      fprintf(stderr,"PURE FFN C words:");for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",ffn_output[i]);fputc('\n',stderr);
      if(c_competition){
        fprintf(stderr,"PURE FFN C view words:");
        for(unsigned i=0;i<1056;i++)fprintf(stderr," %08x",c_view_words[i]);
        fputc('\n',stderr);
      }
      if(result || outword || changed!=1024u || !inputs_intact || !guards_intact ||
         ffn_marks[0]!=0x41e || ffn_marks[1]!=0x41f || !no_agx_metal())return 2;
      if(ffn_pair){
        if(*ready_word!=0x800000f0u)return 2;
        uint32_t first_output[1024];
        if(ffn_acc)memcpy(first_output,ffn_output,sizeof first_output);
        memcpy(mapped[2]+0x4d80,next_inputs[0],4096);
        memcpy(mapped[2]+0x5e80,next_inputs[1],4096);
        if(!ffn_acc)for(unsigned i=0;i<1024;i++)ffn_output[i]=0x7fc01234u;
        unsigned reset=0;
        for(unsigned i=0;i<1024;i++)reset+=ffn_output[i]==(ffn_acc?first_output[i]:0x7fc01234u);
        if(reset!=1024u || memcmp(mapped[2]+0x4d80,next_inputs[0],4096) ||
           memcmp(mapped[2]+0x5e80,next_inputs[1],4096) || !no_agx_metal())return 2;
        void (^scheduled2)(void)=Block_copy(^{ffn_mark_ptr[2]=0x51e;});
        void (^completed2)(void)=Block_copy(^{ffn_mark_ptr[3]=0x51f;});
        if(!scheduled2 || !completed2 || scheduled2==completed2 ||
           scheduled2==scheduled || completed2==completed)return 2;
        uint8_t record2[64]={0},out2[64]={0};
        uintptr_t sp2=(uintptr_t)scheduled2,cp2=(uintptr_t)completed2;
        memcpy(record2,&kernel_id,4);memcpy(record2+4,&segment_id,4);
        memcpy(record2+0x10,&sp2,sizeof sp2);memcpy(record2+0x18,&cp2,sizeof cp2);
        fprintf(stderr,"PURE FFN SECOND entering: same queue/pages/allocations ready=0x%08x %s=%u/1024.\n",
                *ready_word,ffn_acc?"preserved":"reset",reset);
        int result2=Submit(queue,NULL,1,record2,64,out2);
        unsigned waited2=0;
        for(;waited2<3000 && (!ffn_marks[3] || ffn_output[1023]==(ffn_acc?first_output[1023]:0x7fc01234u));waited2++)usleep(10000);
        unsigned changed2=0;
        for(unsigned i=0;i<1024;i++)changed2+=ffn_output[i]!=(ffn_acc?first_output[i]:0x7fc01234u);
        unsigned inputs2_intact=memcmp(mapped[2]+0x4d80,next_inputs[0],4096)==0 &&
                                memcmp(mapped[2]+0x5e80,next_inputs[1],4096)==0;
        unsigned guards2_intact=1;
        if(ffn_c_slot || ffn_c_cross)for(unsigned i=0;i<1024;i++)guards2_intact&=c_view_words[32+i]==0x7fc01234u;
        if(ffn_c_cross || ffn_c_xcontrol)guards2_intact&=ffn_cross_guards(mapped[2],mapped[17],ffn_c_cross);
        for(unsigned i=0;i<0x80;i++){
          guards2_intact&=mapped[2][0x4d00+i]==0xa5 && mapped[2][0x5d80+i]==0xa5;
          guards2_intact&=mapped[2][0x5e00+i]==0xa4 && mapped[2][0x6e80+i]==0xa4;
          guards2_intact&=mapped[2][0x6f00+i]==0xa7 && mapped[2][0x7f80+i]==0xa7;
        }
        uint32_t outword2=0;memcpy(&outword2,out2,4);
        fprintf(stderr,"PURE FFN SECOND returned status=%d outword=0x%08x changed=%u/1024 inputs=%u guards=%u marks=%llu/%llu wait=%u/3000.\n",
                result2,outword2,changed2,inputs2_intact,guards2_intact,
                (unsigned long long)ffn_marks[2],(unsigned long long)ffn_marks[3],waited2);
        fprintf(stderr,"PURE FFN SECOND C words:");
        for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",ffn_output[i]);
        fputc('\n',stderr);
        if(result2 || outword2 || changed2!=1024u || !inputs2_intact || !guards2_intact ||
           ffn_marks[2]!=0x51e || ffn_marks[3]!=0x51f || !no_agx_metal())return 2;
        if(ffn_six){
          uint32_t prior_output[1024];
          for(unsigned k=2;k<6;k++){
            memcpy(prior_output,ffn_output,sizeof prior_output);
            memcpy(mapped[2]+0x4d80,six_inputs[k-2][0],4096);
            memcpy(mapped[2]+0x5e80,six_inputs[k-2][1],4096);
            if(*ready_word!=0x800000f0u || !no_agx_metal())return 2;
            void (^scheduledn)(void)=Block_copy(^{ffn_mark_ptr[2*k]=0x600u+2u*k;});
            void (^completedn)(void)=Block_copy(^{ffn_mark_ptr[2*k+1]=0x601u+2u*k;});
            if(!scheduledn || !completedn || scheduledn==completedn)return 2;
            uint8_t recordn[64]={0},outn[64]={0};
            uintptr_t spn=(uintptr_t)scheduledn,cpn=(uintptr_t)completedn;
            memcpy(recordn,&kernel_id,4);memcpy(recordn+4,&segment_id,4);
            memcpy(recordn+0x10,&spn,sizeof spn);memcpy(recordn+0x18,&cpn,sizeof cpn);
            fprintf(stderr,"PURE FFN EXTRA k=%u entering: same queue/pages/allocations ready=0x%08x preserved=1024/1024.\n",k,*ready_word);
            int resultn=Submit(queue,NULL,1,recordn,64,outn);
            unsigned waitedn=0;
            for(;waitedn<3000 && (!ffn_marks[2*k+1] || ffn_output[1023]==prior_output[1023]);waitedn++)usleep(10000);
            unsigned changedn=0;
            for(unsigned i=0;i<1024;i++)changedn+=ffn_output[i]!=prior_output[i];
            unsigned inputsn=memcmp(mapped[2]+0x4d80,six_inputs[k-2][0],4096)==0 &&
                             memcmp(mapped[2]+0x5e80,six_inputs[k-2][1],4096)==0;
            unsigned guardsn=1;
            if(ffn_c_slot || ffn_c_cross)for(unsigned i=0;i<1024;i++)guardsn&=c_view_words[32+i]==0x7fc01234u;
            if(ffn_c_cross || ffn_c_xcontrol)guardsn&=ffn_cross_guards(mapped[2],mapped[17],ffn_c_cross);
            for(unsigned i=0;i<0x80;i++){
              guardsn&=mapped[2][0x4d00+i]==0xa5 && mapped[2][0x5d80+i]==0xa5;
              guardsn&=mapped[2][0x5e00+i]==0xa4 && mapped[2][0x6e80+i]==0xa4;
              guardsn&=mapped[2][0x6f00+i]==0xa7 && mapped[2][0x7f80+i]==0xa7;
            }
            uint32_t outwordn=0;memcpy(&outwordn,outn,4);
            fprintf(stderr,"PURE FFN EXTRA k=%u returned status=%d outword=0x%08x changed=%u/1024 inputs=%u guards=%u marks=%llu/%llu wait=%u/3000.\n",
                    k,resultn,outwordn,changedn,inputsn,guardsn,
                    (unsigned long long)ffn_marks[2*k],(unsigned long long)ffn_marks[2*k+1],waitedn);
            fprintf(stderr,"PURE FFN EXTRA k=%u C words:",k);
            for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",ffn_output[i]);
            fputc('\n',stderr);
            if(resultn || outwordn || changedn!=1024u || !inputsn || !guardsn ||
               ffn_marks[2*k]!=0x600u+2u*k || ffn_marks[2*k+1]!=0x601u+2u*k ||
               !no_agx_metal())return 2;
          }
          if(ffn_resident_pair){
            uint32_t resident0[1024],prior1[1024];
            uint32_t* resident1=(uint32_t*)(mapped[17]+0x8000);
            memcpy(resident0,ffn_output,sizeof resident0);
            for(unsigned i=0;i<1024;i++)if(resident1[i]!=0u)return 2;
            uint64_t next_c=0x10000060000ull;
            memcpy(mapped[28]+0x1bb0,&next_c,8);
            if(memcmp(mapped[28]+0x1bb0,&next_c,8) || *ready_word!=0x800000f0u || !no_agx_metal())return 2;
            fprintf(stderr,"PURE FFN RESIDENT SWITCH: C=0x%llx n0=1024/1024 retained n1=1024/1024 zero ready=0x%08x.\n",
                    (unsigned long long)next_c,*ready_word);
            for(unsigned k=0;k<6;k++){
              const uint8_t* a_window=k==0?expected_inputs[1]:k==1?next_inputs[0]:six_inputs[k-2][0];
              memcpy(mapped[2]+0x4d80,a_window,4096);
              memcpy(mapped[2]+0x5e80,resident_b[k],4096);
              memcpy(prior1,resident1,sizeof prior1);
              unsigned t=6+k;
              void (^scheduledr)(void)=Block_copy(^{ffn_mark_ptr[2*t]=0x700u+2u*k;});
              void (^completedr)(void)=Block_copy(^{ffn_mark_ptr[2*t+1]=0x701u+2u*k;});
              if(!scheduledr || !completedr || scheduledr==completedr)return 2;
              uint8_t recordr[64]={0},outr[64]={0};
              uintptr_t spr=(uintptr_t)scheduledr,cpr=(uintptr_t)completedr;
              memcpy(recordr,&kernel_id,4);memcpy(recordr+4,&segment_id,4);
              memcpy(recordr+0x10,&spr,sizeof spr);memcpy(recordr+0x18,&cpr,sizeof cpr);
              fprintf(stderr,"PURE FFN RESIDENT n=1 k=%u entering: same queue/pages ready=0x%08x.\n",k,*ready_word);
              int resultr=Submit(queue,NULL,1,recordr,64,outr);
              unsigned waitedr=0;
              for(;waitedr<3000 && (!ffn_marks[2*t+1] || resident1[1023]==prior1[1023]);waitedr++)usleep(10000);
              unsigned changedr=0;
              for(unsigned i=0;i<1024;i++)changedr+=resident1[i]!=prior1[i];
              unsigned inputr=memcmp(mapped[2]+0x4d80,a_window,4096)==0 &&
                              memcmp(mapped[2]+0x5e80,resident_b[k],4096)==0;
              unsigned guardsr=memcmp(ffn_output,resident0,sizeof resident0)==0;
              for(unsigned i=0;i<1024;i++)guardsr&=c_view_words[32+i]==0x7fc01234u;
              for(unsigned i=0;i<0x80;i++){
                guardsr&=mapped[17][0x7f80+i]==0xc7 && mapped[17][0x9000+i]==0xc7;
                guardsr&=mapped[2][0x4d00+i]==0xa5 && mapped[2][0x5d80+i]==0xa5;
                guardsr&=mapped[2][0x5e00+i]==0xa4 && mapped[2][0x6e80+i]==0xa4;
                guardsr&=mapped[2][0x6f00+i]==0xa7 && mapped[2][0x7f80+i]==0xa7;
              }
              uint32_t outwordr=0;memcpy(&outwordr,outr,4);
              fprintf(stderr,"PURE FFN RESIDENT n=1 k=%u returned status=%d outword=0x%08x changed=%u/1024 inputs=%u guards=%u marks=%llu/%llu wait=%u/3000.\n",
                      k,resultr,outwordr,changedr,inputr,guardsr,
                      (unsigned long long)ffn_marks[2*t],(unsigned long long)ffn_marks[2*t+1],waitedr);
              fprintf(stderr,"PURE FFN RESIDENT n=1 k=%u C words:",k);
              for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",resident1[i]);
              fputc('\n',stderr);
              if(resultr || outwordr || changedr!=1024u || !inputr || !guardsr ||
                 ffn_marks[2*t]!=0x700u+2u*k || ffn_marks[2*t+1]!=0x701u+2u*k ||
                 !no_agx_metal())return 2;
            }
            fprintf(stderr,"PURE FFN RESIDENT FINAL n0:");
            for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",ffn_output[i]);
            fputc('\n',stderr);
            fprintf(stderr,"PURE FFN RESIDENT FINAL n1:");
            for(unsigned i=0;i<1024;i++)fprintf(stderr," %08x",resident1[i]);
            fputc('\n',stderr);
          }
        }
      }
      return 0;
    }
    uint32_t* output=(uint32_t*)((tensor_far || tensor_sg4 || tensor_sg4h)?mapped[21]+0x80:mapped[2]+(tensor_loop?0x5680:(tensor_common?0x5580:(tensor_graph?0x5380:0x800))));
    int tensor_stage=getenv("ORDERED_TENSOR_STAGE")!=NULL;
    unsigned sentinel_count=tensor_sg4h12?98304u:tensor_sg4h10?81920u:tensor_sg4h9?73728u:tensor_sg4h8?65536u:tensor_sg4h7?57344u:tensor_sg4h6?49152u:tensor_sg4h5?40960u:tensor_sg4h4?32768u:tensor_sg4h3?24576u:tensor_sg4h?16384u:(tensor_sg4?8192u:(tensor_loop?2048u:(tensor_common?323u:(tensor_stage?256u:64u))));
    uint32_t sentinel_value=tensor_common?0u:(tensor_stage?0xffffffffu:0xDEADBEEFu);
    unsigned sentinels=0;for(unsigned i=0;i<sentinel_count;i++)sentinels+=(output[i]==sentinel_value);
    if(sentinels!=sentinel_count){fprintf(stderr,"ORDERED QUEUE REFUSED: output sentinels %u/%u\n",sentinels,sentinel_count);return 2;}
    fprintf(stderr,"ORDERED QUEUE PASS: notification bound, selector-14 IDs 1/2 staged 32 KiB with fresh trace IDs, ready=0xf0, output sentinels=%u/%u; no Submit.\n",sentinels,sentinel_count);
    if(getenv("ORDERED_TG_ONE")){
#ifdef G17_BLOCK_PREFLIGHT
      if(getenv("ORDERED_TENSOR_STAGE")){
        int tensor_witness=getenv("ORDERED_TENSOR_WITNESS")!=NULL;
        const char* paths[3]={getenv("ORDERED_TENSOR_A"),getenv("ORDERED_TENSOR_B"),getenv("ORDERED_TENSOR_C")};
        const unsigned offsets[3]={(tensor_far || tensor_sg4h)?0x80u:(tensor_graph?0x4d80u:0u),
                                   tensor_loop?0x80u:(tensor_common?0x5180u:(tensor_graph?0x5080u:0x400u)),
                                   (tensor_far || tensor_sg4 || tensor_sg4h)?0x80u:(tensor_loop?0x5680u:(tensor_common?0x5580u:(tensor_graph?0x5380u:0x800u)))};
        const unsigned sizes[3]={tensor_far?49152u:tensor_sg4h12?98304u:tensor_sg4h10?81920u:tensor_sg4h9?73728u:tensor_sg4h8?65536u:tensor_sg4h7?57344u:tensor_sg4h6?49152u:tensor_sg4h5?40960u:tensor_sg4h4?32768u:tensor_sg4h3?24576u:tensor_sg4h?16384u:tensor_sg4?8192u:tensor_loop?2048u:tensor_common?544u:512u,
                                 tensor_sg4h12loop137700reuse?31338496u:tensor_sg4h12loop73728reuse?16781312u:tensor_sg4h12loop36720reuse?25071616u:tensor_sg4h12loop18360reuse?29249536u:tensor_sg4h12loop9180reuse?31338496u:tensor_sg4h12loop8160reuse?31338496u:tensor_sg4h12loop7650runtime?31334400u:tensor_sg4h12loop6120runtime?25067520u:tensor_sg4h12loop8160runtime?(tensor_addressed_16320?1069547520u:tensor_sixteenfold_logical_b?534773760u:tensor_octuple_logical_b?267386880u:tensor_quadruple_logical_b?133693440u:tensor_triple_logical_b?100270080u:tensor_double_logical_b?66846720u:33423360u):tensor_sg4h12loop4080sixteen?16711680u:tensor_sg4h12loop2040oct?8355840u:tensor_sg4h12loop1275quint?5222400u:tensor_sg4h12loop1020quad?4177920u:tensor_sg4h12loop765triple?3133440u:tensor_sg4h12loop510dual?2088960u:tensor_sg4h12loop256tail?1048576u:tensor_sg4h12loop255?1044480u:tensor_sg4h12loop254?1040384u:tensor_sg4h12loop136?557056u:tensor_sg4h12loop128?524288u:tensor_sg4h12loop64?262144u:tensor_sg4h12loop40?163840u:tensor_sg4h12loop32?131072u:tensor_sg4h12loop31?126976u:tensor_sg4h12loop24?98304u:tensor_sg4h12loop23?94208u:tensor_sg4h12loop17?69632u:tensor_far?65536u:(tensor_loop16 || tensor_sg4loop16 || tensor_sg4hloop16 || tensor_sg4h3loop16)?65536u:tensor_loop15?61440u:tensor_loop8?32768u:tensor_loop7?28672u:tensor_loop5?20480u:tensor_loop?16384u:tensor_common?608u:512u,
                                 tensor_far?49152u:tensor_sg4h12?393216u:tensor_sg4h10?327680u:tensor_sg4h9?294912u:tensor_sg4h8?262144u:tensor_sg4h7?229376u:tensor_sg4h6?196608u:tensor_sg4h5?163840u:tensor_sg4h4?131072u:tensor_sg4h3?98304u:tensor_sg4h?65536u:tensor_sg4?32768u:tensor_loop?8192u:tensor_common?1292u:1024u};
        uint8_t* input_base[3]={(tensor_far || tensor_sg4h)?mapped[19]:mapped[2],tensor_loop?mapped[20]:mapped[2],(tensor_far || tensor_sg4 || tensor_sg4h)?mapped[21]:mapped[2]};
        /* Bound and retain each authored input for post-Submit integrity checks. */
        unsigned char* expected[3]={0};
        for(unsigned i=0;i<3;i++){
          if(sizes[i]>1069547520u){
            fprintf(stderr,"PURE TENSOR STAGE REFUSED: input %u exceeds verifier buffer\n",i);return 2;
          }
          if(!paths[i]){fprintf(stderr,"PURE TENSOR STAGE REFUSED: input path absent\n");return 2;}
          expected[i]=malloc(sizes[i]);
          if(!expected[i]){fprintf(stderr,"PURE TENSOR STAGE REFUSED: input %u verifier allocation\n",i);return 2;}
          FILE* f=fopen(paths[i],"rb");
          if(!f){perror("PURE TENSOR input");return 2;}
          int ok=fread(expected[i],1,sizes[i],f)==sizes[i] && fgetc(f)==EOF && !ferror(f);
          fclose(f);
          if(!ok || memcmp(input_base[i]+offsets[i],expected[i],sizes[i])){
            fprintf(stderr,"PURE TENSOR STAGE REFUSED: input %u differs from authored bytes\n",i);return 2;
          }
        }
        const uint32_t control_offsets[]={tensor_graph?0u:0xc00u,
                                          tensor_graph?8u:0xc08u,
                                          tensor_graph?0xcu:0xc0cu,
                                          tensor_graph?0x200u:0xe00u,
                                          tensor_graph?0x204u:0xe04u,
                                          tensor_graph?0x208u:0xe08u,
                                          tensor_graph?0x20cu:0xe0cu};
        const uint32_t control_values[]={0x0b6d0019,0x5800,0x10,0x19,0x10000000,0x073c7774,0x80000000};
        for(unsigned i=0;i<sizeof control_offsets/sizeof control_offsets[0];i++){
          uint32_t got=0;memcpy(&got,mapped[2]+control_offsets[i],4);
          if(got!=control_values[i]){
            fprintf(stderr,"PURE TENSOR STAGE REFUSED: allocation-2 control +0x%x changed\n",control_offsets[i]);return 2;
          }
        }
        if(getenv("ORDERED_TG_FIRE") || !no_agx_metal())return 2;
        fprintf(stderr,"PURE TENSOR REGISTERED STAGE PASS: exact %u-byte code, tensor-correlated packet bits, %u-thread grid, FP16 A/B, FP32 C sentinels and adjacent controls, %u allocations, selector-14 pages, no AGXMetal; no Submit yet.\n",(tensor_sg4h12loop137700reuse && getenv("ORDERED_TENSOR_STATIONARY_COUNT_WITNESS"))?62124u:(tensor_sg4h12loop137700reuse && getenv("ORDERED_TENSOR_STATIONARY8192"))?62128u:(tensor_sg4h12loop137700reuse && getenv("ORDERED_TENSOR_FUSED24"))?62148u:(tensor_farunfold?10404u:(tensor_far?(tensor_farfast?5132u:5980u):(tensor_sg4h12loop137700reuse?62580u:tensor_sg4h12loop73728reuse?62580u:tensor_sg4h12loop36720reuse?62620u:tensor_sg4h12loop18360reuse?62700u:tensor_sg4h12loop9180reuse?62860u:tensor_sg4h12loop8160reuse?56220u:tensor_sg4h12loop7650runtime?52900u:tensor_sg4h12loop6120runtime?42880u:tensor_sg4h12loop8160runtime?(tensor_sg4h12loop8160codecontrol?56174u:56240u):tensor_sg4h12loop4080sixteen?56174u:tensor_sg4h12loop2040oct?29454u:tensor_sg4h12loop1275quint?19434u:tensor_sg4h12loop1020quad?16094u:tensor_sg4h12loop765triple?12754u:(tensor_sg4h12loop256tail || tensor_sg4h12loop510dual)?9414u:tensor_sg4h?6074u:(tensor_sg4?6034u:(tensor_loopscale?5092u:(tensor_loop?5940u:(tensor_common?1232u:(tensor_witness?80u:648u)))))))),tensor_sg4h12?1536u:(tensor_sg4h10?1280u:(tensor_sg4h9?1152u:(tensor_sg4h8?1024u:(tensor_sg4h7?896u:(tensor_sg4h6?768u:(tensor_sg4h5?640u:(tensor_sg4h4?512u:(tensor_sg4h3?384u:(tensor_sg4h?256u:(tensor_sg4?128u:32u)))))))))),tensor_farunfold?30u:29u);
        if(!getenv("ORDERED_TENSOR_FIRE"))return 0;
        typedef int (*submit_t)(void*,void*,unsigned,void*,unsigned,void*);
        submit_t Submit=dlsym(io,"IOGPUCommandQueueSubmitCommandBuffers");
        if(!Submit)return 2;
        static volatile uint64_t tensor_marks[6]={0,0,0,0,0,0};
        volatile uint64_t* tensor_mark_ptr=tensor_marks;
        void (^scheduled)(void)=Block_copy(^{tensor_mark_ptr[0]=0x17e;});
        void (^completed)(void)=Block_copy(^{tensor_mark_ptr[1]=0x17f;});
        if(!scheduled || !completed || scheduled==completed)return 2;
        uint8_t record[64]={0},out[64]={0};
        uint32_t kernel_id=shmem[1].id,segment_id=shmem[0].id;
        uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
        memcpy(record,&kernel_id,4);memcpy(record+4,&segment_id,4);
        memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
        volatile uint32_t* ready_word=(volatile uint32_t*)(uintptr_t)(shmem[0].cpu+0x24);
        if(*ready_word!=0xf0u || !no_agx_metal())return 2;
        *ready_word=0x800000f0u;
        fprintf(stderr,"PURE TENSOR entering: count=1 IDs=2/1 stride=64 ready=0x%08x.\n",*ready_word);
        int result=Submit(queue,NULL,1,record,64,out);
        unsigned wait_ticks=1000;
        const char* wait_text=getenv("ORDERED_TENSOR_WAIT_TICKS");
        if(wait_text){
          char* end=NULL;unsigned long requested=strtoul(wait_text,&end,10);
          if(!*wait_text || *end || requested<1000 || requested>30000)return 2;
          wait_ticks=(unsigned)requested;
        }
        unsigned waited=0;
        for(;waited<wait_ticks && (!tensor_marks[1] || output[sentinel_count-1]==sentinel_value);waited++)usleep(10000);
        fprintf(stderr,"PURE TENSOR completion wait=%u/%u ticks marks=%llu/%llu last=0x%08x.\n",
                waited,wait_ticks,(unsigned long long)tensor_marks[0],
                (unsigned long long)tensor_marks[1],output[sentinel_count-1]);
        unsigned a_intact=memcmp(input_base[0]+offsets[0],expected[0],sizes[0])==0;
        unsigned b_intact=memcmp(input_base[1]+offsets[1],expected[1],sizes[1])==0;
        unsigned changed=0;for(unsigned i=0;i<sentinel_count;i++)changed+=(output[i]!=sentinel_value);
        unsigned controls_intact=1;
        for(unsigned i=0;i<sizeof control_offsets/sizeof control_offsets[0];i++){
          uint32_t got=0;memcpy(&got,mapped[2]+control_offsets[i],4);
          controls_intact&=got==control_values[i];
        }
        uint32_t outword=0;memcpy(&outword,out,4);
        fprintf(stderr,"PURE TENSOR returned status=%d outword=0x%08x changed=%u/%u A=%u B=%u controls=%u marks=%llu/%llu.\n",
                result,outword,changed,sentinel_count,a_intact,b_intact,controls_intact,
                (unsigned long long)tensor_marks[0],(unsigned long long)tensor_marks[1]);
        unsigned guards_intact=1;
        if(tensor_loop5){
          for(unsigned byte=0;byte<0x80;byte++)
            guards_intact&=mapped[20][0x5080+byte]==0xa5;
          fprintf(stderr,"PURE TENSOR LOOP5 B guard=%u.\n",guards_intact);
        }
        if(tensor_loop7){
          for(unsigned byte=0;byte<0x80;byte++)
            guards_intact&=mapped[20][0x7080+byte]==0xa5;
          fprintf(stderr,"PURE TENSOR LOOP7 B guard=%u.\n",guards_intact);
        }
        if(tensor_loop8){
          for(unsigned byte=0;byte<0x80;byte++)
            guards_intact&=mapped[20][0x8080+byte]==0xa5;
          fprintf(stderr,"PURE TENSOR LOOP8 B guard=%u.\n",guards_intact);
        }
        if(tensor_loop15){
          for(unsigned byte=0;byte<0x80;byte++)
            guards_intact&=mapped[20][0xf080+byte]==0xa5;
          fprintf(stderr,"PURE TENSOR LOOP15 B guard=%u.\n",guards_intact);
        }
        if(tensor_loop16){
          for(unsigned byte=0;byte<0x80;byte++)
            guards_intact&=mapped[20][0x10080+byte]==0xa5;
          fprintf(stderr,"PURE TENSOR LOOP16 B guard=%u.\n",guards_intact);
        }
        if(tensor_sg4loop16){
          for(unsigned byte=0;byte<0x80;byte++)
            guards_intact&=mapped[20][0x10080+byte]==0xa5;
          fprintf(stderr,"PURE TENSOR SG4LOOP16 B guard=%u.\n",guards_intact);
        }
        if(tensor_far || tensor_sg4 || tensor_sg4h){
          const uint8_t* guard_base[6]={(tensor_far || tensor_sg4h)?mapped[19]:mapped[2]+0x4d00,
                                        tensor_far?mapped[19]+0xc080:tensor_sg4h12?mapped[19]+0x18080:tensor_sg4h10?mapped[19]+0x14080:tensor_sg4h9?mapped[19]+0x12080:tensor_sg4h8?mapped[19]+0x10080:tensor_sg4h7?mapped[19]+0xe080:tensor_sg4h6?mapped[19]+0xc080:tensor_sg4h5?mapped[19]+0xa080:tensor_sg4h4?mapped[19]+0x8080:tensor_sg4h3?mapped[19]+0x6080:tensor_sg4h?mapped[19]+0x4080:mapped[2]+0x6d80,
                                        mapped[20],mapped[20]+(tensor_sg4h12loop137700reuse?0x1de3080:tensor_sg4h12loop73728reuse?0x1001080:tensor_sg4h12loop36720reuse?0x17e9080:tensor_sg4h12loop18360reuse?0x1be5080:tensor_sg4h12loop9180reuse?0x1de3080:(tensor_sg4h12loop8160reuse?0x1de3080:(tensor_sg4h12loop7650runtime?0x1de2080:(tensor_sg4h12loop6120runtime?0x17e8080:(tensor_sg4h12loop8160runtime?(tensor_addressed_16320?0x3fc00080:tensor_sixteenfold_logical_b?0x1fe00080:tensor_octuple_logical_b?0xff00080:tensor_quadruple_logical_b?0x7f80080:tensor_triple_logical_b?0x5fa0080:tensor_double_logical_b?0x3fc0080:0x1fe0080):(tensor_sg4h12loop4080sixteen?0xff0080:(tensor_sg4h12loop2040oct?0x7f8080:(tensor_sg4h12loop1275quint?0x4fb080:(tensor_sg4h12loop1020quad?0x3fc080:(tensor_sg4h12loop765triple?0x2fd080:(tensor_sg4h12loop510dual?0x1fe080:(tensor_sg4h12loop256tail?0x100080:(tensor_sg4h12loop255?0xff080:(tensor_sg4h12loop254?0xfe080:(tensor_sg4h12loop136?0x88080:(tensor_sg4h12loop128?0x80080:(tensor_sg4h12loop64?0x40080:(tensor_sg4h12loop40?0x28080:(tensor_sg4h12loop32?0x20080:(tensor_sg4h12loop31?0x1f080:(tensor_sg4h12loop24?0x18080:(tensor_sg4h12loop23?0x17080:(tensor_sg4h12loop17?0x11080:((tensor_far || tensor_sg4loop16 || tensor_sg4hloop16 || tensor_sg4h3loop16)?0x10080:0x4080)))))))))))))))))))))))),
                                        mapped[21],mapped[21]+(tensor_far?0xc080:tensor_sg4h12?0x60080:tensor_sg4h10?0x50080:tensor_sg4h9?0x48080:tensor_sg4h8?0x40080:tensor_sg4h7?0x38080:tensor_sg4h6?0x30080:tensor_sg4h5?0x28080:tensor_sg4h4?0x20080:tensor_sg4h3?0x18080:tensor_sg4h?0x10080:0x8080)};
          for(unsigned region=0;region<6;region++)
            for(unsigned byte=0;byte<0x80;byte++)
              guards_intact&=guard_base[region][byte]==0xa5;
          fprintf(stderr,"PURE TENSOR SG4 guards=%u.\n",guards_intact);
        }
        unsigned tail_intact=1;
        if(tensor_far){
          for(unsigned i=2048;i<12288;i++)tail_intact&=output[i]==0xffffffffu;
          fprintf(stderr,"PURE TENSOR FAR tail=%u.\n",tail_intact);
        }
        if(tensor_farunfold){
          unsigned extra_changed=0;
          for(unsigned i=0;i<0x40000;i++)extra_changed+=mapped[22][i]!=0;
          fprintf(stderr,"PURE TENSOR FAR extra22_changed=%u.\n",extra_changed);
        }
        fprintf(stderr,"PURE TENSOR C words:");for(unsigned i=0;i<sentinel_count;i++)fprintf(stderr," %08x",output[i]);fputc('\n',stderr);
        if(result || outword || !a_intact || !b_intact || !controls_intact || !guards_intact || !tail_intact ||
           tensor_marks[0]!=0x17e || tensor_marks[1]!=0x17f || !no_agx_metal())return 2;
        if(getenv("ORDERED_TENSOR_SECOND_SELECTOR_FFFF")){
          fprintf(stderr,"PURE TENSOR SECOND precondition addressed=%u arm=%s count=%u alloc25=%u selector=%02x%02x ready=0x%08x.\n",
                  tensor_addressed_16320,alloc25_lookup_arm?alloc25_lookup_arm:"(null)",
                  sentinel_count,mapped[25]!=NULL,mapped[25]?mapped[25][9]:0,
                  mapped[25]?mapped[25][10]:0,*ready_word);
          if(!tensor_addressed_16320 || !alloc25_lookup_arm ||
             strcmp(alloc25_lookup_arm,"taildual_fffd_tensor") ||
             sentinel_count!=98304u || !mapped[25] ||
             mapped[25][9]!=0xfdu || mapped[25][10]!=0xffu ||
             *ready_word!=0x800000f0u)return 2;
          int keep_ready=getenv("ORDERED_TENSOR_SECOND_READY_KEEP")!=NULL;
          if(!keep_ready){
            *ready_word=0xf0u;
            if(*ready_word!=0xf0u)return 2;
            fprintf(stderr,"PURE TENSOR SECOND reset page ready=0x%08x after completed first Submit.\n",*ready_word);
          }else{
            fprintf(stderr,"PURE TENSOR SECOND kept page ready=0x%08x after completed first Submit.\n",*ready_word);
          }
          mapped[25][9]=0xffu;
          for(unsigned i=0;i<sentinel_count;i++)output[i]=sentinel_value;
          unsigned reset_count=0;
          for(unsigned i=0;i<sentinel_count;i++)reset_count+=(output[i]==sentinel_value);
          if(reset_count!=sentinel_count || !no_agx_metal())return 2;
          void (^scheduled2)(void)=Block_copy(^{tensor_mark_ptr[2]=0x27e;});
          void (^completed2)(void)=Block_copy(^{tensor_mark_ptr[3]=0x27f;});
          if(!scheduled2 || !completed2 || scheduled2==completed2 ||
             scheduled2==scheduled || completed2==completed)return 2;
          uint8_t record2[64]={0},out2[64]={0};
          uintptr_t sp2=(uintptr_t)scheduled2,cp2=(uintptr_t)completed2;
          memcpy(record2,&kernel_id,4);memcpy(record2+4,&segment_id,4);
          memcpy(record2+0x10,&sp2,sizeof sp2);memcpy(record2+0x18,&cp2,sizeof cp2);
          if(!keep_ready)*ready_word=0x800000f0u;
          fprintf(stderr,"PURE TENSOR SECOND entering: same queue/pages/allocations, selector +9=ff +10=ff, ready=0x%08x mode=%s sentinels=%u/%u.\n",
                  *ready_word,keep_ready?"keep":"reset",reset_count,sentinel_count);
          int result2=Submit(queue,NULL,1,record2,64,out2);
          unsigned waited2=0;
          for(;waited2<wait_ticks && (!tensor_marks[3] || output[0]==sentinel_value);waited2++)usleep(10000);
          unsigned changed2=0;
          for(unsigned i=0;i<sentinel_count;i++)changed2+=(output[i]!=sentinel_value);
          unsigned a2_intact=memcmp(input_base[0]+offsets[0],expected[0],sizes[0])==0;
          unsigned b2_intact=memcmp(input_base[1]+offsets[1],expected[1],sizes[1])==0;
          unsigned controls2_intact=1;
          for(unsigned i=0;i<sizeof control_offsets/sizeof control_offsets[0];i++){
            uint32_t got=0;memcpy(&got,mapped[2]+control_offsets[i],4);
            controls2_intact&=got==control_values[i];
          }
          unsigned guards2_intact=1;
          for(unsigned byte=0;byte<0x80;byte++){
            guards2_intact&=mapped[20][byte]==0xa5;
            guards2_intact&=mapped[20][0x3fc00080u+byte]==0xa5;
            guards2_intact&=mapped[21][byte]==0xa5;
            guards2_intact&=mapped[21][0x60080u+byte]==0xa5;
          }
          uint32_t outword2=0;memcpy(&outword2,out2,4);
          fprintf(stderr,"PURE TENSOR SECOND returned status=%d outword=0x%08x changed=%u/%u A=%u B=%u controls=%u guards=%u marks=%llu/%llu wait=%u/%u.\n",
                  result2,outword2,changed2,sentinel_count,a2_intact,b2_intact,controls2_intact,
                  guards2_intact,(unsigned long long)tensor_marks[2],
                  (unsigned long long)tensor_marks[3],waited2,wait_ticks);
          fprintf(stderr,"PURE TENSOR SECOND C words:");for(unsigned i=0;i<sentinel_count;i++)fprintf(stderr," %08x",output[i]);fputc('\n',stderr);
          if(result2 || outword2 || changed2!=1536u || !a2_intact || !b2_intact ||
             !controls2_intact || !guards2_intact ||
             tensor_marks[2]!=0x27e || tensor_marks[3]!=0x27f ||
             mapped[25][9]!=0xffu || mapped[25][10]!=0xffu || !no_agx_metal())return 2;
          if(getenv("ORDERED_TENSOR_THIRD_SELECTOR_FFFD")){
            if(!keep_ready || *ready_word!=0x800000f0u)return 2;
            mapped[25][9]=0xfdu;
            for(unsigned i=0;i<sentinel_count;i++)output[i]=sentinel_value;
            unsigned reset3=0;
            for(unsigned i=0;i<sentinel_count;i++)reset3+=(output[i]==sentinel_value);
            if(reset3!=sentinel_count || !no_agx_metal())return 2;
            void (^scheduled3)(void)=Block_copy(^{tensor_mark_ptr[4]=0x37e;});
            void (^completed3)(void)=Block_copy(^{tensor_mark_ptr[5]=0x37f;});
            if(!scheduled3 || !completed3 || scheduled3==completed3 ||
               scheduled3==scheduled || scheduled3==scheduled2 ||
               completed3==completed || completed3==completed2)return 2;
            uint8_t record3[64]={0},out3[64]={0};
            uintptr_t sp3=(uintptr_t)scheduled3,cp3=(uintptr_t)completed3;
            memcpy(record3,&kernel_id,4);memcpy(record3+4,&segment_id,4);
            memcpy(record3+0x10,&sp3,sizeof sp3);memcpy(record3+0x18,&cp3,sizeof cp3);
            fprintf(stderr,"PURE TENSOR THIRD entering: same queue/pages/allocations, selector +9=fd +10=ff, ready=0x%08x sentinels=%u/%u.\n",
                    *ready_word,reset3,sentinel_count);
            int result3=Submit(queue,NULL,1,record3,64,out3);
            unsigned waited3=0;
            for(;waited3<wait_ticks && (!tensor_marks[5] || output[sentinel_count-1]==sentinel_value);waited3++)usleep(10000);
            unsigned changed3=0;
            for(unsigned i=0;i<sentinel_count;i++)changed3+=(output[i]!=sentinel_value);
            unsigned a3_intact=memcmp(input_base[0]+offsets[0],expected[0],sizes[0])==0;
            unsigned b3_intact=memcmp(input_base[1]+offsets[1],expected[1],sizes[1])==0;
            unsigned controls3_intact=1;
            for(unsigned i=0;i<sizeof control_offsets/sizeof control_offsets[0];i++){
              uint32_t got=0;memcpy(&got,mapped[2]+control_offsets[i],4);
              controls3_intact&=got==control_values[i];
            }
            unsigned guards3_intact=1;
            for(unsigned byte=0;byte<0x80;byte++){
              guards3_intact&=mapped[20][byte]==0xa5;
              guards3_intact&=mapped[20][0x3fc00080u+byte]==0xa5;
              guards3_intact&=mapped[21][byte]==0xa5;
              guards3_intact&=mapped[21][0x60080u+byte]==0xa5;
            }
            uint32_t outword3=0;memcpy(&outword3,out3,4);
            fprintf(stderr,"PURE TENSOR THIRD returned status=%d outword=0x%08x changed=%u/%u A=%u B=%u controls=%u guards=%u marks=%llu/%llu wait=%u/%u.\n",
                    result3,outword3,changed3,sentinel_count,a3_intact,b3_intact,controls3_intact,
                    guards3_intact,(unsigned long long)tensor_marks[4],
                    (unsigned long long)tensor_marks[5],waited3,wait_ticks);
            fprintf(stderr,"PURE TENSOR THIRD C words:");for(unsigned i=0;i<sentinel_count;i++)fprintf(stderr," %08x",output[i]);fputc('\n',stderr);
            if(result3 || outword3 || changed3!=sentinel_count || !a3_intact || !b3_intact ||
               !controls3_intact || !guards3_intact || tensor_marks[4]!=0x37e ||
               tensor_marks[5]!=0x37f || mapped[25][9]!=0xfdu || mapped[25][10]!=0xffu ||
               !no_agx_metal())return 2;
          }
        }
        fprintf(stderr,"PURE TENSOR SUBMIT RETURNED: output requires independent register-map scoring.\n");
        return 0;
      }
      int tg_ab=getenv("ORDERED_TG_AB")!=NULL;
      int tg_two_groups=getenv("ORDERED_TG_TWO_GROUPS")!=NULL;
      int tg_three_groups=getenv("ORDERED_TG_THREE_GROUPS")!=NULL;
      int tg_y_two=getenv("ORDERED_TG_Y_TWO")!=NULL;
      int tg_z_two=getenv("ORDERED_TG_Z_TWO")!=NULL;
      int tg_xy_two=getenv("ORDERED_TG_XY_TWO")!=NULL;
      int tg_partial96=getenv("ORDERED_TG_PARTIAL96")!=NULL;
      int tg_full64=getenv("ORDERED_TG_FULL64")!=NULL;
      int tg_full32=getenv("ORDERED_TG_FULL32")!=NULL;
      int tg_partial_tg96=getenv("ORDERED_TG_PARTIAL_TG96")!=NULL;
      int tg_full_tg64=getenv("ORDERED_TG_FULL_TG64")!=NULL;
      int tg_partial_tg80=getenv("ORDERED_TG_PARTIAL_TG80")!=NULL;
      int tg_full_tg80_control=getenv("ORDERED_TG_FULL_TG80_CONTROL")!=NULL;
      unsigned tg_param_grid=0,tg_param_xor=0;
      int tg_param=ordered_tg_param(&tg_param_grid,&tg_param_xor);
      if(tg_param<0)return 2;
      int tg_four_bank=getenv("ORDERED_TG_FOUR_BANK")!=NULL;
      const char* tg_half_2048=getenv("ORDERED_TG_HALF_2048");
      int tg_readout=getenv("ORDERED_TG_READOUT")!=NULL || tg_half_2048!=NULL;
      unsigned tg_bank_offset=tg_half_2048 && strcmp(tg_half_2048,"upper")==0?4u:0u;
      int tg_scrambled=getenv("ORDERED_TG_B_SCRAMBLED")!=NULL;
      uint32_t* a=(uint32_t*)mapped[2];
      uint32_t* b=(uint32_t*)(mapped[2]+0x400);
      unsigned a_before=0,b_before=0,guard_before=0;
      unsigned input_words=tg_param?tg_param_grid:tg_full32?32u:tg_xy_two?256u:(tg_three_groups?192u:((tg_two_groups || tg_y_two || tg_z_two)?128u:(tg_partial_tg80?80u:((tg_partial96 || tg_partial_tg96)?96u:64u))));
      for(unsigned i=0;i<input_words;i++){
        a_before+=(a[i]==11u*i+5u);
        b_before+=(b[i]==tg_b_word(i,tg_scrambled));
      }
      for(unsigned i=0;i<256;i++)guard_before+=(output[i]==0xDEADBEEFu);
      uint32_t* readout_guards=(uint32_t*)(mapped[2]+0x2000);
      unsigned distant_guards_before=0;
      if(tg_readout || tg_xy_two)for(unsigned i=0;i<192;i++)distant_guards_before+=(readout_guards[i]==0u);
      if(a_before!=input_words || b_before!=input_words || guard_before!=256 ||
         ((tg_readout || tg_xy_two) && distant_guards_before!=192) || !no_agx_metal()){
        fprintf(stderr,"PURE TG REFUSED: input or output preconditions\n");return 2;
      }
      fprintf(stderr,"PURE TG STAGE PASS: A=%u/%u B=%u/%u C+guards=%u/256, %s, legacy pure command pages; no Submit.\n",
              a_before,input_words,b_before,input_words,guard_before,
              tg_param?"parameterized partial scratch grid":
              tg_partial_tg80?"80-thread grid with 16 active lanes in the second 64-thread group":
              (tg_partial96 || tg_partial_tg96)?"96-thread grid with partial second 64-thread group":
              tg_full32?"full 32-thread grid through dispatchThreads mode":
              (tg_full64 || tg_full_tg64 || tg_full_tg80_control)?"full 64-thread grid through dispatchThreads mode":
              tg_xy_two?"2x2 X/Y 64-thread groups":
              tg_z_two?"two Z-axis 64-thread groups":
              tg_y_two?"two Y-axis 64-thread groups":
              tg_three_groups?"three 64-thread groups":
              (tg_two_groups?"two 64-thread groups":"one 64-thread group"));
      if(!getenv("ORDERED_TG_FIRE"))return 0;
      if(getenv("ORDERED_ZERO_SUBMIT") || getenv("ORDERED_GUARDED_ONE") ||
         getenv("ORDERED_GUARDED_TWO") || getenv("ORDERED_BATCH_PREFLIGHT") ||
         getenv("ORDERED_INVALID_ONE"))return 2;
      typedef int (*submit_t)(void*,void*,unsigned,void*,unsigned,void*);
      submit_t Submit=dlsym(io,"IOGPUCommandQueueSubmitCommandBuffers");
      if(!Submit)return 2;
      static volatile uint64_t marks[2]={0,0};
      volatile uint64_t* mark_ptr=marks;
      void (^scheduled)(void)=Block_copy(^{mark_ptr[0]=0x17e;});
      void (^completed)(void)=Block_copy(^{mark_ptr[1]=0x17f;});
      if(!scheduled || !completed || scheduled==completed)return 2;
      uint8_t record[64]={0},out[64]={0};
      uint32_t kernel_id=shmem[1].id,segment_id=shmem[0].id;
      uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
      memcpy(record,&kernel_id,4);memcpy(record+4,&segment_id,4);
      memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
      volatile uint32_t* ready_word=(volatile uint32_t*)(uintptr_t)(shmem[0].cpu+0x24);
      if(*ready_word!=0xf0u || !no_agx_metal())return 2;
      *ready_word=0x800000f0u;
      fprintf(stderr,"PURE TG entering: count=1 IDs=2/1 stride=64 ready=0x%08x, authored static threadgroup cross-SIMD exchange.\n",*ready_word);
      int result=Submit(queue,NULL,1,record,64,out);
      unsigned c_words=tg_param?tg_param_grid:tg_full32?32u:tg_xy_two?256u:(tg_three_groups?192u:((tg_two_groups || tg_y_two || tg_z_two)?128u:(tg_partial_tg80?80u:((tg_partial96 || tg_partial_tg96)?96u:(tg_readout?256u:64u)))));
      unsigned guard_words=tg_param?256u-tg_param_grid:tg_full32?224u:tg_three_groups?64u:((tg_two_groups || tg_y_two || tg_z_two)?128u:(tg_partial_tg80?176u:((tg_partial96 || tg_partial_tg96)?160u:192u)));
      for(unsigned i=0;i<300 && (!marks[1] || output[c_words-1]==0xDEADBEEFu);i++)usleep(10000);
      unsigned a_exact=0,b_exact=0,c_exact=0,guards=0;
      for(unsigned i=0;i<input_words;i++){
        a_exact+=(a[i]==11u*i+5u);
        b_exact+=(b[i]==tg_b_word(i,tg_scrambled));
        unsigned peer=(tg_partial96 || tg_full64 || tg_full32)?i:
                      tg_param?((i&~63u)+((i&63u)^tg_param_xor)):
                      (tg_partial_tg80 || tg_full_tg80_control)?
                      ((i&~63u)+((i&63u)^8u)):
                      (tg_partial_tg96 || tg_full_tg64)?
                      ((i&~63u)+((i&63u)^16u)):
                      (tg_two_groups || tg_three_groups || tg_y_two || tg_z_two || tg_xy_two)?
                      ((i&~63u)+((i&63u)^32u)):(i^32u);
        uint32_t expected_c=tg_four_bank?(4u*(11u*peer+5u)+6u):
                            (tg_ab?(11u*peer+5u+tg_b_word(peer,tg_scrambled)):
                                   (11u*peer+5u));
        if(!tg_readout)c_exact+=(output[i]==expected_c);
      }
      if(tg_readout){
        for(unsigned bank=0;bank<4;bank++)for(unsigned i=0;i<64;i++){
          unsigned peer=i^32u;
          c_exact+=(output[64*bank+i]==11u*peer+5u+bank+tg_bank_offset);
        }
        for(unsigned i=0;i<192;i++)guards+=(readout_guards[i]==0u);
      }else if(tg_xy_two){
        for(unsigned i=0;i<192;i++)guards+=(readout_guards[i]==0u);
      }else for(unsigned i=c_words;i<256;i++)guards+=(output[i]==0xDEADBEEFu);
      uint32_t outword=0;memcpy(&outword,out,4);
      fprintf(stderr,"PURE TG returned status=%d outword=0x%08x A=%u/%u B=%u/%u C=%u/%u guards=%u/%u marks=%llu/%llu.\n",
              result,outword,a_exact,input_words,b_exact,input_words,c_exact,c_words,guards,guard_words,
              (unsigned long long)marks[0],(unsigned long long)marks[1]);
      fprintf(stderr,"PURE TG A words:");for(unsigned i=0;i<input_words;i++)fprintf(stderr," %u",a[i]);fputc('\n',stderr);
      fprintf(stderr,"PURE TG B words:");for(unsigned i=0;i<input_words;i++)fprintf(stderr," %u",b[i]);fputc('\n',stderr);
      fprintf(stderr,"PURE TG C words:");for(unsigned i=0;i<c_words;i++)fprintf(stderr," %u",output[i]);fputc('\n',stderr);
      fprintf(stderr,"PURE TG guard words:");for(unsigned i=0;i<guard_words;i++)fprintf(stderr," %u",(tg_readout || tg_xy_two)?readout_guards[i]:output[c_words+i]);fputc('\n',stderr);
      if(result || outword || a_exact!=input_words || b_exact!=input_words || c_exact!=c_words ||
         guards!=guard_words || marks[0]!=0x17e || marks[1]!=0x17f || !no_agx_metal())return 2;
      fprintf(stderr,(tg_partial96 || tg_full64 || tg_full32)?
              "PURE TG PASS: authored direct-grid A+B executed below Metal.\n":
              (tg_partial_tg96 || tg_full_tg64)?
              "PURE TG PASS: authored partial-grid scratch exchange executed below Metal.\n":
              (tg_partial_tg80 || tg_full_tg80_control)?
              "PURE TG PASS: authored 16-lane partial-grid scratch exchange executed below Metal.\n":
              tg_param?
              "PURE TG PASS: authored parameterized partial-grid scratch exchange executed below Metal.\n":
              "PURE TG PASS: authored cross-SIMDgroup threadgroup exchange executed below Metal.\n");
      return 0;
#else
      fprintf(stderr,"PURE TG REFUSED: build lacks block support\n");return 2;
#endif
    }
    const char* code_output=getenv("VALID_OUTPUT_CROSS");
    if(code_output && strcmp(code_output,"alloc24-code-output")==0){
      const char* arm=getenv("VALID_ALLOC24_REQUEST_ARM");
      const char* placement=getenv("VALID_PLACEMENT");
      if(!arm || (strcmp(arm,"code") && strcmp(arm,"data")) ||
         !placement || strcmp(placement,"alloc24-candidate") ||
         !getenv("VALID_OP") || strcmp(getenv("VALID_OP"),"add") ||
         !getenv("VALID_SCALE") || strcmp(getenv("VALID_SCALE"),"3") ||
         !getenv("VALID_BIAS") || strcmp(getenv("VALID_BIAS"),"2"))return 2;
      uint64_t addresses[3]={0};memcpy(addresses,mapped[28]+0x1ba0,sizeof addresses);
      uint32_t* target=(uint32_t*)(mapped[24]+0x2000);
      unsigned intact=0;for(unsigned i=0;i<256;i++)intact+=(target[i]==0xDEADBEEFu);
      if(addresses[0]!=0x10000030000ull || addresses[1]!=0x10000030400ull ||
         addresses[2]!=0x100000a2000ull || intact!=256)return 2;
      fprintf(stderr,"VALID ALLOC24 CODE OUTPUT STAGE PASS: arm=%s code=0x100000a0500 C=0x100000a2000 sentinels=%u/256; no Submit.\n",
              arm,intact);
    }
    if(getenv("ORDERED_ATOMIC_ONE") || getenv("ORDERED_UNIFORM_ATOMIC_ONE")){
#ifdef G17_BLOCK_PREFLIGHT
      int uniform_one=getenv("ORDERED_UNIFORM_ATOMIC_ONE")!=NULL;
      int report_width=getenv("ORDERED_UNIFORM_REPORT_WIDTH")!=NULL;
      int uniform_and=uniform_one && getenv("ORDERED_ATOMIC_OP") &&
                      strcmp(getenv("ORDERED_ATOMIC_OP"),"and")==0;
      int uniform_or=uniform_one && getenv("ORDERED_ATOMIC_OP") &&
                     strcmp(getenv("ORDERED_ATOMIC_OP"),"or")==0;
      int uniform_xor=uniform_one && getenv("ORDERED_ATOMIC_OP") &&
                      strcmp(getenv("ORDERED_ATOMIC_OP"),"xor")==0;
      int uniform_sub=uniform_one && getenv("ORDERED_ATOMIC_OP") &&
                      strcmp(getenv("ORDERED_ATOMIC_OP"),"sub")==0;
      int uniform_exchange=uniform_one && getenv("ORDERED_ATOMIC_OP") &&
                           strcmp(getenv("ORDERED_ATOMIC_OP"),"exchange")==0;
      int uniform_hole=uniform_one && getenv("ORDERED_ATOMIC_OP") &&
                       (strcmp(getenv("ORDERED_ATOMIC_OP"),"hole12")==0 ||
                        strcmp(getenv("ORDERED_ATOMIC_OP"),"hole12plus1")==0 ||
                        strcmp(getenv("ORDERED_ATOMIC_OP"),"hole12minus32")==0 ||
                        strcmp(getenv("ORDERED_ATOMIC_OP"),"hole12float1")==0 ||
                        strcmp(getenv("ORDERED_ATOMIC_OP"),"hole12ulp")==0 ||
                        strcmp(getenv("ORDERED_ATOMIC_OP"),"hole12halfulp")==0 ||
                        strcmp(getenv("ORDERED_ATOMIC_OP"),"hole12minnormal")==0 ||
                        strcmp(getenv("ORDERED_ATOMIC_OP"),"hole12maxsub")==0 ||
                        strcmp(getenv("ORDERED_ATOMIC_OP"),"hole12negminnormal")==0 ||
                        strcmp(getenv("ORDERED_ATOMIC_OP"),"smin")==0 ||
                        strcmp(getenv("ORDERED_ATOMIC_OP"),"smax")==0 ||
                        strcmp(getenv("ORDERED_ATOMIC_OP"),"umin")==0 ||
                        strcmp(getenv("ORDERED_ATOMIC_OP"),"umax")==0 ||
                        strcmp(getenv("ORDERED_ATOMIC_OP"),"cmpxchg")==0);
      uint32_t uniform_seed=getenv("ORDERED_UNIFORM_SEED")?(uint32_t)strtoul(getenv("ORDERED_UNIFORM_SEED"),NULL,10):5u;
      if(uniform_seed!=5u && !uniform_hole)return 2;
      if(getenv("ORDERED_ZERO_SUBMIT") || getenv("ORDERED_GUARDED_ONE") ||
         getenv("ORDERED_GUARDED_TWO") || getenv("ORDERED_BATCH_PREFLIGHT") ||
         getenv("ORDERED_INVALID_ONE"))return 2;
      uint32_t* input_a=(uint32_t*)mapped[2];
      uint32_t* input_b=(uint32_t*)(mapped[2]+0x400);
      const char* profile_op=getenv("ORDERED_ATOMIC_OP");
      int atomic_signed_profile=!uniform_one && profile_op &&
        (!strcmp(profile_op,"min") || !strcmp(profile_op,"max") ||
         !strcmp(profile_op,"umin") || !strcmp(profile_op,"umax"));
      int atomic_float_profile=!uniform_one && profile_op &&
        (!strcmp(profile_op,"fadd32") || !strcmp(profile_op,"fadd32ret7"));
      int atomic_contend_float_profile=!uniform_one && profile_op &&
        (!strcmp(profile_op,"contendfadd32") ||
         !strcmp(profile_op,"contendfadd32ret7"));
      int atomic_contend_float_nan=!uniform_one && profile_op &&
        !strcmp(profile_op,"contendfadd32ret7nan");
      int atomic_loopdiv13_profile=!uniform_one && profile_op &&
        !strcmp(profile_op,"loopdiv13");
      int atomic_loopdiv31_profile=!uniform_one && profile_op &&
        !strcmp(profile_op,"loopdiv31");
      int atomic_loopcap53_profile=!uniform_one && profile_op &&
        !strcmp(profile_op,"loopcap53");
      int atomic_loopzero03_profile=!uniform_one && profile_op &&
        !strcmp(profile_op,"loopzero03");
      int atomic_shuffle1norm_profile=!uniform_one && profile_op &&
        !strcmp(profile_op,"shuffle1norm");
      int atomic_contend_float_half_even=!uniform_one && profile_op &&
        (!strcmp(profile_op,"contendfadd32halfeven") ||
         !strcmp(profile_op,"contendfadd32halfret7even"));
      int atomic_contend_float_half_odd=!uniform_one && profile_op &&
        (!strcmp(profile_op,"contendfadd32halfodd") ||
         !strcmp(profile_op,"contendfadd32halfret7odd"));
      for(unsigned i=0;i<64;i++){
        uint32_t expected_a=atomic_loopdiv13_profile?(i%2?3u:1u):
          atomic_loopdiv31_profile?(i%2?1u:3u):
          atomic_loopcap53_profile?(i%2?3u:5u):
          atomic_loopzero03_profile?(i%2?3u:0u):
          atomic_shuffle1norm_profile?(0x3f800000u+256u*i):
          atomic_contend_float_nan && i==0?0x7fc12345u:
          atomic_contend_float_half_odd && i==0?0x3f800001u:
          (atomic_contend_float_profile || atomic_contend_float_half_even) && i==0?0x3f800000u:
          atomic_float_profile?(i%2==0?0x3f800000u:0xbf800000u):
          atomic_signed_profile?
          (i%4==0?11*i+5:(i%4==1?0xffffffffu:(i%4==2?0x80000000u:0u))):
          (i==0?uniform_seed:11*i+5);
        if(input_a[i]!=expected_a || input_b[i]!=7*i+3 ||
           output[i]!=0xDEADBEEFu)return 2;
      }
      for(unsigned i=64;i<256;i++)if(output[i]!=0xDEADBEEFu)return 2;
      if(!no_agx_metal())return 2;
      fprintf(stderr,"PURE ATOMIC STAGE PASS: 64 A, 64 B, and 256 output words checked; IDs 2/1, ready=0xf0; no Submit.\n");
      if(!getenv(uniform_one?"ORDERED_UNIFORM_ATOMIC_FIRE":"ORDERED_ATOMIC_FIRE"))return 0;
      typedef int (*submit_t)(void*,void*,unsigned,void*,unsigned,void*);
      submit_t Submit=dlsym(io,"IOGPUCommandQueueSubmitCommandBuffers");
      if(!Submit)return 2;
      static volatile uint64_t marks[2]={0,0};
      volatile uint64_t* mark_ptr=marks;
      void (^scheduled)(void)=Block_copy(^{mark_ptr[0]=0x17e;});
      void (^completed)(void)=Block_copy(^{mark_ptr[1]=0x17f;});
      if(!scheduled || !completed || scheduled==completed)return 2;
      uint8_t record[64]={0},out[64]={0};
      uint32_t kernel_id=shmem[1].id,segment_id=shmem[0].id;
      uintptr_t sp=(uintptr_t)scheduled,cp=(uintptr_t)completed;
      memcpy(record,&kernel_id,4);memcpy(record+4,&segment_id,4);
      memcpy(record+0x10,&sp,sizeof sp);memcpy(record+0x18,&cp,sizeof cp);
      volatile uint32_t* ready_word=(volatile uint32_t*)(uintptr_t)(shmem[0].cpu+0x24);
      if(*ready_word!=0xf0u)return 2;
      *ready_word=0x800000f0u;
      const char* run_op=getenv("ORDERED_ATOMIC_OP")?getenv("ORDERED_ATOMIC_OP"):"add";
      char run_label[64];
      snprintf(run_label,sizeof run_label,"uniform-%s",
               report_width && !strcmp(run_op,"add")?"width":run_op);
      fprintf(stderr,"PURE ATOMIC entering: op=%s count=1 IDs=2/1 stride=64; atomic A[i], dependent load, descriptor store.\n",
              uniform_one?run_label:run_op);
      int result=Submit(queue,NULL,1,record,64,out);
      for(unsigned i=0;i<300 && (!marks[1] || output[63]==0xDEADBEEFu);i++)usleep(10000);
      if(uniform_hole){
        uint32_t outword=0;memcpy(&outword,out,4);
        unsigned guards=0;for(unsigned i=64;i<256;i++)guards+=(output[i]==0xDEADBEEFu);
        fprintf(stderr,"PURE UNIFORM HOLE observation: status=%d outword=0x%08x A0=%u B0=%u C0=%u C63=%u guards=%u/192 marks=%llu/%llu.\n",
                result,outword,input_a[0],input_b[0],output[0],output[63],guards,
                (unsigned long long)marks[0],(unsigned long long)marks[1]);
        fprintf(stderr,"PURE UNIFORM HOLE A words:");
        for(unsigned i=0;i<64;i++)fprintf(stderr," %u",input_a[i]);
        fputc('\n',stderr);
        fprintf(stderr,"PURE UNIFORM HOLE B words:");
        for(unsigned i=0;i<64;i++)fprintf(stderr," %u",input_b[i]);
        fputc('\n',stderr);
        fprintf(stderr,"PURE UNIFORM HOLE C words:");
        for(unsigned i=0;i<64;i++)fprintf(stderr," %u",output[i]);
        fputc('\n',stderr);
        if(!no_agx_metal())return 2;
        return result || outword || marks[0]!=0x17e || marks[1]!=0x17f || guards!=192 ? 2:0;
      }
      if(uniform_one){
        unsigned a_other=0,b_exact=0,guards=0,c_permutation=0;
        unsigned seen[64]={0};
        uint32_t width=report_width?input_b[0]:0;
        for(unsigned i=1;i<64;i++)a_other+=(input_a[i]==11*i+5);
        for(unsigned i=0;i<64;i++){
          b_exact+=(input_b[i]==(report_width?width:7*i+3));
          if(uniform_and){
            if(output[i]<64)seen[output[i]]++;
          } else if(output[i]>=5 && output[i]<69)seen[output[i]-5]++;
        }
        for(unsigned i=0;i<64;i++){
          unsigned expected=uniform_and?(i<5?1:(i<32?2:(i<37?1:0))):1;
          c_permutation+=(seen[i]==expected);
        }
        if(uniform_sub){
          unsigned low_first=0,high_first=0;
          for(unsigned i=0;i<32;i++){
            uint32_t wrapped=5u-32u+i;
            low_first+=(output[i]==5u+i)+(output[i+32]==wrapped);
            high_first+=(output[i]==wrapped)+(output[i+32]==5u+i);
          }
          c_permutation=low_first>high_first?low_first:high_first;
        }
        if(uniform_exchange){
          unsigned low_first=0,high_first=0;
          for(unsigned i=0;i<32;i++){
            low_first+=(output[i]==5u+i)+(output[i+32]==32u+i);
            high_first+=(output[i]==32u+i)+(output[i+32]==5u+i);
          }
          c_permutation=low_first>high_first?low_first:high_first;
        }
        for(unsigned i=64;i<256;i++)guards+=(output[i]==0xDEADBEEFu);
        uint32_t outword=0;memcpy(&outword,out,4);
        fprintf(stderr,"PURE UNIFORM returned status=%d outword=0x%08x A0=%u A_other=%u/63 C_%s=%u/64 B=%u/64 width=%u guards=%u/192 marks=%llu/%llu.\n",
                result,outword,input_a[0],a_other,uniform_exchange?"exchangeseq":(uniform_sub?"subseq":(uniform_and?"multiset":"permutation")),c_permutation,b_exact,width,guards,
                (unsigned long long)marks[0],(unsigned long long)marks[1]);
        fprintf(stderr,"PURE UNIFORM A words:");
        for(unsigned i=0;i<64;i++)fprintf(stderr," %u",input_a[i]);
        fputc('\n',stderr);
        fprintf(stderr,"PURE UNIFORM C words:");
        for(unsigned i=0;i<64;i++)fprintf(stderr," %u",output[i]);
        fputc('\n',stderr);
        fprintf(stderr,"PURE UNIFORM B words:");
        for(unsigned i=0;i<64;i++)fprintf(stderr," %u",input_b[i]);
        fputc('\n',stderr);
        if(result || outword || input_a[0]!=(uniform_exchange?32u:(uniform_sub?(5u-64u):(uniform_xor?5u:(uniform_and?0u:(uniform_or?37u:69u))))) || a_other!=63 ||
           c_permutation!=64 || b_exact!=64 || guards!=192 ||
           (report_width && (!width || width>64 || 64%width)) ||
           marks[0]!=0x17e || marks[1]!=0x17f || !no_agx_metal())return 2;
        fprintf(stderr,"PURE UNIFORM PASS: elected-lane uniform atomic and broadcast executed below Metal.\n");
        return 0;
      }
      int explicit_b=getenv("ORDERED_ATOMIC_EXPLICIT_B")!=NULL;
      const char* atomic_op=getenv("ORDERED_ATOMIC_OP")?getenv("ORDERED_ATOMIC_OP"):"add";
      int atomic_and=strcmp(atomic_op,"and")==0,atomic_xor=strcmp(atomic_op,"xor")==0;
      int atomic_sub=strcmp(atomic_op,"sub")==0,atomic_or=strcmp(atomic_op,"or")==0;
      int atomic_min=strcmp(atomic_op,"min")==0,atomic_max=strcmp(atomic_op,"max")==0;
      int atomic_umin=strcmp(atomic_op,"umin")==0,atomic_umax=strcmp(atomic_op,"umax")==0;
      int atomic_exchange=strcmp(atomic_op,"exchange")==0;
      int atomic_fadd32=strcmp(atomic_op,"fadd32")==0;
      int atomic_fadd32ret7=strcmp(atomic_op,"fadd32ret7")==0;
      int atomic_contendfadd32ret7=strcmp(atomic_op,"contendfadd32ret7")==0;
      int atomic_contendfadd32ret7nan=strcmp(atomic_op,"contendfadd32ret7nan")==0;
      int atomic_loop1=strcmp(atomic_op,"loop1")==0;
      int atomic_loop3=strcmp(atomic_op,"loop3")==0;
      int atomic_loopdiv13=strcmp(atomic_op,"loopdiv13")==0;
      int atomic_loopdiv31=strcmp(atomic_op,"loopdiv31")==0;
      int atomic_loopcap53=strcmp(atomic_op,"loopcap53")==0;
      int atomic_loopzero03=strcmp(atomic_op,"loopzero03")==0;
      int atomic_loopregion2=strcmp(atomic_op,"loopregion2")==0;
      int atomic_shuffle1norm=strcmp(atomic_op,"shuffle1norm")==0;
      int atomic_shuffle1sub=strcmp(atomic_op,"shuffle1sub")==0;
      int atomic_contendfadd32halfret7even=strcmp(atomic_op,"contendfadd32halfret7even")==0;
      int atomic_contendfadd32halfret7odd=strcmp(atomic_op,"contendfadd32halfret7odd")==0;
      int atomic_contendadd=strcmp(atomic_op,"contendadd")==0;
      int atomic_compiledaddret=strcmp(atomic_op,"compiledaddret")==0;
      int atomic_compiledaddretplus1=strcmp(atomic_op,"compiledaddretplus1")==0;
      int atomic_compiledxorret=strcmp(atomic_op,"compiledxorret")==0;
      int atomic_compiledxorretplus1=strcmp(atomic_op,"compiledxorretplus1")==0;
      int atomic_compiledorret=strcmp(atomic_op,"compiledorret")==0;
      int atomic_compiledsubret=strcmp(atomic_op,"compiledsubret")==0;
      int atomic_compiledandret=strcmp(atomic_op,"compiledandret")==0;
      int atomic_compiledcmpxchg5ret=strcmp(atomic_op,"compiledcmpxchg5ret")==0;
      int atomic_compiledcmpxchg5shift1ret=strcmp(atomic_op,"compiledcmpxchg5shift1ret")==0;
      int atomic_compiledcmpxchg5shift5ret=strcmp(atomic_op,"compiledcmpxchg5shift5ret")==0;
      int atomic_compiledcmpxchg5shift7ret=strcmp(atomic_op,"compiledcmpxchg5shift7ret")==0;
      int atomic_compiledcmpxchg5shift12ret=strcmp(atomic_op,"compiledcmpxchg5shift12ret")==0;
      int atomic_authoredxorret7=strcmp(atomic_op,"authoredxorret7")==0;
      int atomic_authoredorret7=strcmp(atomic_op,"authoredorret7")==0;
      int atomic_authoredsubret7=strcmp(atomic_op,"authoredsubret7")==0;
      int atomic_authoredandret7=strcmp(atomic_op,"authoredandret7")==0;
      int atomic_contendfadd32=strcmp(atomic_op,"contendfadd32")==0;
      int atomic_contendfadd32halfeven=strcmp(atomic_op,"contendfadd32halfeven")==0;
      int atomic_contendfadd32halfodd=strcmp(atomic_op,"contendfadd32halfodd")==0;
      int atomic_cmpxchg5=strcmp(atomic_op,"cmpxchg5")==0;
      int atomic_cmpxchg6=strcmp(atomic_op,"cmpxchg6")==0;
      int atomic_cmpxchg5ret=strcmp(atomic_op,"cmpxchg5ret")==0;
      int atomic_cmpxchg5retwide0=strcmp(atomic_op,"cmpxchg5retwide0")==0;
      int atomic_cmpxchg5retwide7=strcmp(atomic_op,"cmpxchg5retwide7")==0;
      int atomic_addret=strcmp(atomic_op,"addret")==0;
      int atomic_addretwitness=strcmp(atomic_op,"addretwitness")==0;
      int atomic_addretwitnessb=strcmp(atomic_op,"addretwitnessb")==0;
      int atomic_addretwide=strcmp(atomic_op,"addretwide")==0;
      int atomic_addretwidedelay=strcmp(atomic_op,"addretwidedelay")==0;
      int atomic_addretwide7=strcmp(atomic_op,"addretwide7")==0;
      int atomic_contendaddret7=strcmp(atomic_op,"contendaddret7")==0;
      int atomic_contendaddretvary7=strcmp(atomic_op,"contendaddretvary7")==0;
      unsigned a_exact=0,c_exact=0,b_exact=0,guards=0;
      for(unsigned i=0;i<64;i++){
        uint32_t signed_input=i%4==0?11*i+5:(i%4==1?0xffffffffu:
                               (i%4==2?0x80000000u:0u));
        int32_t signed_value=(int32_t)signed_input;
        uint32_t signed_min=(uint32_t)(signed_value<(int32_t)(i+1)?signed_value:(int32_t)(i+1));
        uint32_t signed_max=(uint32_t)(signed_value>(int32_t)(i+1)?signed_value:(int32_t)(i+1));
        uint32_t predicted=atomic_loopdiv13?(i%2?3u:1u):
                           atomic_loopdiv31?(i%2?1u:3u):
                           atomic_loopcap53?(i%2?3u:5u):
                           atomic_loopzero03?(i%2?3u:0u):
                           atomic_loopregion2?(11*i+5):
                           atomic_shuffle1norm?(0x3f800000u+256u*i):
                           atomic_shuffle1sub?(11*i+5):
                           (atomic_loop1 || atomic_loop3)?(11*i+5):
                           atomic_and?((11*i+5)&0x55u):
                           (atomic_authoredandret7 || atomic_compiledandret)?((11*i+5)&(i+1u)):
                           (atomic_xor || atomic_authoredxorret7 || atomic_compiledxorret || atomic_compiledxorretplus1)?((11*i+5)^(i+1u)):
                           (atomic_sub || atomic_authoredsubret7 || atomic_compiledsubret)?(10*i+4):
                           (atomic_or || atomic_authoredorret7 || atomic_compiledorret)?((11*i+5)|(i+1u)):
                           atomic_min?signed_min:
                           atomic_max?signed_max:
                           atomic_umin?(signed_input<(i+1)?signed_input:(i+1)):
                           atomic_umax?(signed_input>(i+1)?signed_input:(i+1)):
                           atomic_exchange?(i+1):
                           (atomic_fadd32 || atomic_fadd32ret7)?(i%2==0?0x40000000u:0u):
                           atomic_contendadd?(i==0?2085u:11*i+5):
                           atomic_contendaddret7?(i==0?69u:11*i+5):
                           atomic_contendaddretvary7?(i==0?2085u:11*i+5):
                           (atomic_contendfadd32 || atomic_contendfadd32ret7)?(i==0?0x42820000u:11*i+5):
                           atomic_contendfadd32ret7nan?(i==0?0x7fc00000u:11*i+5):
                           (atomic_contendfadd32halfeven || atomic_contendfadd32halfret7even)?(i==0?0x3f800000u:11*i+5):
                           (atomic_contendfadd32halfodd || atomic_contendfadd32halfret7odd)?(i==0?0x3f800002u:11*i+5):
                           (atomic_cmpxchg5 || atomic_cmpxchg5ret || atomic_cmpxchg5retwide0 || atomic_cmpxchg5retwide7 || atomic_compiledcmpxchg5ret || atomic_compiledcmpxchg5shift1ret || atomic_compiledcmpxchg5shift5ret || atomic_compiledcmpxchg5shift7ret || atomic_compiledcmpxchg5shift12ret)?(i==0?1u:11*i+5):
                           atomic_cmpxchg6?(11*i+5):(12*i+6);
        uint32_t predicted_c=atomic_loopdiv13?(i+7u*(i%2?3u:1u)):
                             atomic_loopdiv31?(i+7u*(i%2?1u:3u)):
                             atomic_loopcap53?(i+21u):
                             atomic_loopzero03?(i+(i%2?21u:7u)):
                             atomic_loopregion2?(i+(i<32?14u:0u)):
                             atomic_shuffle1norm?(0x3f800000u+256u*(i^1u)):
                             atomic_shuffle1sub?(11u*(i^1u)+5u):
                             (atomic_loop1 || atomic_loop3)?(i+(atomic_loop1?7u:21u)):
                             (atomic_contendadd || atomic_contendfadd32 ||
                              atomic_contendfadd32halfeven || atomic_contendfadd32halfodd)?(i+1):
                             (atomic_cmpxchg5ret || atomic_cmpxchg5retwide0 || atomic_cmpxchg5retwide7 || atomic_compiledcmpxchg5ret || atomic_compiledcmpxchg5shift1ret || atomic_compiledcmpxchg5shift5ret || atomic_compiledcmpxchg5shift7ret || atomic_compiledcmpxchg5shift12ret || atomic_addret || atomic_addretwitness || atomic_addretwide || atomic_addretwidedelay || atomic_addretwide7 || atomic_compiledaddret || atomic_authoredxorret7 || atomic_authoredorret7 || atomic_authoredsubret7 || atomic_authoredandret7 || atomic_compiledorret || atomic_compiledsubret || atomic_compiledandret || atomic_compiledxorret)?11*i+5:
                             atomic_fadd32ret7?(i%2==0?0x3f800000u:0xbf800000u):
                             (atomic_compiledaddretplus1 || atomic_compiledxorretplus1)?11*i+6:
                             atomic_addretwitnessb?0xDEADBEEFu:predicted;
        a_exact+=(input_a[i]==predicted);
        c_exact+=(output[i]==(explicit_b?(atomic_contendaddret7?(5u+i):predicted_c):0xDEADBEEFu));
        b_exact+=(input_b[i]==(explicit_b?(atomic_addretwitnessb?11*i+5:7*i+3):predicted));
      }
      if(atomic_contendaddret7){
        unsigned seen[64]={0};
        c_exact=0;
        for(unsigned i=0;i<64;i++){
          uint32_t old=output[i];
          if(old>=5u && old<69u)seen[old-5u]++;
        }
        for(unsigned i=0;i<64;i++)c_exact+=(seen[i]==1u);
      }
      if(atomic_contendfadd32ret7){
        c_exact=0;
        for(unsigned k=1;k<=64;k++){
          float value=(float)k;
          uint32_t bits=0;
          memcpy(&bits,&value,sizeof(bits));
          unsigned count=0;
          for(unsigned i=0;i<64;i++)count+=(output[i]==bits);
          c_exact+=(count==1u);
        }
      }
      if(atomic_contendfadd32ret7nan){
        unsigned first=0,canonical=0;
        for(unsigned i=0;i<64;i++){
          first+=(output[i]==0x7fc12345u);
          canonical+=(output[i]==0x7fc00000u);
        }
        c_exact=(first==1u && canonical==63u)?64u:0u;
      }
      if(atomic_contendfadd32halfret7even){
        c_exact=0;
        for(unsigned i=0;i<64;i++)c_exact+=(output[i]==0x3f800000u);
      }
      if(atomic_contendfadd32halfret7odd){
        unsigned before=0,after=0;
        for(unsigned i=0;i<64;i++){
          before+=(output[i]==0x3f800001u);
          after+=(output[i]==0x3f800002u);
        }
        c_exact=(before==1u && after==63u)?64u:0u;
      }
      if(atomic_contendaddretvary7){
        uint8_t used[64]={0};
        uint32_t current=5u;
        c_exact=0;
        for(unsigned step=0;step<64;step++){
          unsigned match=64;
          for(unsigned i=0;i<64;i++)if(!used[i] && output[i]==current){
            if(match!=64){match=64;break;}
            match=i;
          }
          if(match==64)break;
          used[match]=1;
          current+=match+1u;
          c_exact++;
        }
        if(current!=2085u)c_exact=0;
      }
      for(unsigned i=64;i<256;i++)guards+=(output[i]==0xDEADBEEFu);
      uint32_t outword=0;memcpy(&outword,out,4);
      fprintf(stderr,"PURE ATOMIC returned op=%s explicit_B=%d status=%d outword=0x%08x A=%u/64 C=%u/64 B=%u/64 guards=%u/192 marks=%llu/%llu.\n",
              atomic_op,explicit_b,result,outword,a_exact,c_exact,b_exact,guards,
              (unsigned long long)marks[0],(unsigned long long)marks[1]);
      fprintf(stderr,"PURE ATOMIC A words:");
      for(unsigned i=0;i<64;i++)fprintf(stderr," %u",input_a[i]);
      fputc('\n',stderr);
      fprintf(stderr,"PURE ATOMIC C words:");
      for(unsigned i=0;i<64;i++)fprintf(stderr," %u",output[i]);
      fputc('\n',stderr);
      fprintf(stderr,"PURE ATOMIC B words:");
      for(unsigned i=0;i<64;i++)fprintf(stderr," %u",input_b[i]);
      fputc('\n',stderr);
      if(result || outword || a_exact!=64 || c_exact!=64 || b_exact!=64 ||
         guards!=192 || marks[0]!=0x17e || marks[1]!=0x17f || !no_agx_metal())return 2;
        fprintf(stderr,"PURE ATOMIC PASS: authored per-lane %s executed below Metal.\n",
                (atomic_shuffle1norm || atomic_shuffle1sub)?"SIMD shuffle XOR1 and indexed store":
                (atomic_loop1 || atomic_loop3 || atomic_loopdiv13 || atomic_loopdiv31 || atomic_loopcap53 || atomic_loopzero03 || atomic_loopregion2)?
                  "bounded loop and indexed store":"device atomic, dependent load and store");
      return 0;
#else
      return 2;
#endif
    }
    if(getenv("ORDERED_ZERO_SUBMIT")){
#ifdef G17_BLOCK_PREFLIGHT
      typedef int (*submit_t)(void*,void*,unsigned,void*,unsigned,void*);
      submit_t Submit=dlsym(io,"IOGPUCommandQueueSubmitCommandBuffers");
      if(!Submit){fprintf(stderr,"ORDERED ZERO REFUSED: Submit export absent\n");return 2;}
      static volatile uint64_t marks[2]={0,0};
      volatile uint64_t* mark_ptr=marks;
      void (^scheduled)(void)=Block_copy(^{mark_ptr[0]=0x17e;});
      void (^completed)(void)=Block_copy(^{mark_ptr[1]=0x17f;});
      if(!scheduled || !completed || scheduled==completed)return 2;
      uint8_t record[64]={0};
      uint32_t kernel_id=shmem[1].id,segment_id=shmem[0].id;
      uintptr_t scheduled_ptr=(uintptr_t)scheduled,completed_ptr=(uintptr_t)completed;
      memcpy(record,&kernel_id,4);memcpy(record+4,&segment_id,4);
      memcpy(record+0x10,&scheduled_ptr,sizeof scheduled_ptr);
      memcpy(record+0x18,&completed_ptr,sizeof completed_ptr);
      uint8_t out[64]={0};
      fprintf(stderr,"ORDERED ZERO entering: count=0 record IDs=2/1 stride=64 ready=0xf0; no GPU command nominated.\n");
      int result=Submit(queue,NULL,0,record,64,out);
      unsigned intact=0;for(unsigned i=0;i<64;i++)intact+=(output[i]==0xDEADBEEFu);
      fprintf(stderr,"ORDERED ZERO returned status=%d sentinels=%u/64 marks=%llu/%llu; no GPU command nominated.\n",
              result,intact,(unsigned long long)marks[0],(unsigned long long)marks[1]);
      if(result || intact!=64 || marks[0] || marks[1] || !no_agx_metal())return 2;
      fprintf(stderr,"ORDERED ZERO PASS: zero-count Submit accepted on original-order allocation state; no GPU work.\n");
#else
      fprintf(stderr,"ORDERED ZERO REFUSED: build lacks block support\n");return 2;
#endif
    }
    if(getenv("ORDERED_GUARDED_ONE")){
#ifdef G17_BLOCK_PREFLIGHT
      typedef int (*submit_t)(void*,void*,unsigned,void*,unsigned,void*);
      submit_t Submit=dlsym(io,"IOGPUCommandQueueSubmitCommandBuffers");
      if(!Submit || getenv("ORDERED_ZERO_SUBMIT")){fprintf(stderr,"GUARDED ONE REFUSED: Submit or mode\n");return 2;}
      static volatile uint64_t marks[2]={0,0};
      volatile uint64_t* mark_ptr=marks;
      void (^scheduled)(void)=Block_copy(^{mark_ptr[0]=0x17e;});
      void (^completed)(void)=Block_copy(^{mark_ptr[1]=0x17f;});
      if(!scheduled || !completed || scheduled==completed)return 2;
      uint8_t record[64]={0};
      uint32_t kernel_id=shmem[1].id,segment_id=shmem[0].id;
      uintptr_t scheduled_ptr=(uintptr_t)scheduled,completed_ptr=(uintptr_t)completed;
      memcpy(record,&kernel_id,4);memcpy(record+4,&segment_id,4);
      memcpy(record+0x10,&scheduled_ptr,sizeof scheduled_ptr);
      memcpy(record+0x18,&completed_ptr,sizeof completed_ptr);
      uint8_t out[64]={0};
      g17_guard_waiting=1;
      for(unsigned i=0;i<10000 && g17_guard_ready!=0x17c0de03u;i++)usleep(1000);
      if(g17_guard_ready!=0x17c0de03u){fprintf(stderr,"GUARDED ONE REFUSED: interposer gate timed out\n");return 2;}
      volatile uint32_t* ready_word=(volatile uint32_t*)(uintptr_t)(shmem[0].cpu+0x24);
      if(*ready_word!=0xf0u){fprintf(stderr,"GUARDED ONE REFUSED: staged ready word\n");return 2;}
      *ready_word=0x800000f0u;
      fprintf(stderr,"GUARDED ONE entering: count=1 record IDs=2/1 stride=64 ready=0x%08x; kernel trap interposed.\n",*ready_word);
      int result=Submit(queue,NULL,1,record,64,out);
      unsigned intact=0;for(unsigned i=0;i<64;i++)intact+=(output[i]==0xDEADBEEFu);
      fprintf(stderr,"GUARDED ONE returned status=%d sentinels=%u/64 marks=%llu/%llu ready=0x%08x.\n",
              result,intact,(unsigned long long)marks[0],(unsigned long long)marks[1],*ready_word);
      if(result!=-536870206 || intact!=64 || marks[0] || marks[1] || !no_agx_metal())return 2;
      fprintf(stderr,"GUARDED ONE PASS: count-one userspace path intercepted before kernel; no GPU work.\n");
#else
      fprintf(stderr,"GUARDED ONE REFUSED: build lacks block support\n");return 2;
#endif
    }
    if(getenv("ORDERED_GUARDED_TWO")){
#ifdef G17_BLOCK_PREFLIGHT
      typedef int (*submit_t)(void*,void*,unsigned,void*,unsigned,void*);
      submit_t Submit=dlsym(io,"IOGPUCommandQueueSubmitCommandBuffers");
      if(!Submit || getenv("ORDERED_ZERO_SUBMIT") || getenv("ORDERED_GUARDED_ONE") ||
         getenv("ORDERED_VALID_ONE") || getenv("ORDERED_VALID_TWO"))return 2;
      static volatile uint64_t marks[4]={0,0,0,0};
      volatile uint64_t* mark_ptr=marks;
      void (^s0)(void)=Block_copy(^{mark_ptr[0]=0x17e;});
      void (^c0)(void)=Block_copy(^{mark_ptr[1]=0x17f;});
      void (^s1)(void)=Block_copy(^{mark_ptr[2]=0x27e;});
      void (^c1)(void)=Block_copy(^{mark_ptr[3]=0x27f;});
      if(!s0 || !c0 || !s1 || !c1 || s0==c0 || s0==s1 || s0==c1 ||
         c0==s1 || c0==c1 || s1==c1)return 2;
      uint8_t records[128]={0},out[64]={0};
      uint32_t kernel_id=shmem[1].id,segment_id=shmem[0].id;
      uintptr_t pointers[4]={(uintptr_t)s0,(uintptr_t)c0,(uintptr_t)s1,(uintptr_t)c1};
      for(unsigned i=0;i<2;i++){
        uint8_t* record=records+i*64;
        memcpy(record,&kernel_id,4);memcpy(record+4,&segment_id,4);
        memcpy(record+0x10,&pointers[2*i],sizeof(uintptr_t));
        memcpy(record+0x18,&pointers[2*i+1],sizeof(uintptr_t));
      }
      volatile uint32_t* ready_word=(volatile uint32_t*)(uintptr_t)(shmem[0].cpu+0x24);
      if(*ready_word!=0xf0u || !no_agx_metal())return 2;
      g17_guard_waiting=1;
      for(unsigned i=0;i<10000 && g17_guard_ready!=0x17c0de03u;i++)usleep(1000);
      if(g17_guard_ready!=0x17c0de03u)return 2;
      *ready_word=0x800000f0u;
      fprintf(stderr,"GUARDED TWO entering: count=2 stride=64 IDs=2/1 twice, four live blocks, ready=0x800000f0; all kernel calls interposed.\n");
      int result=Submit(queue,NULL,2,records,64,out);
      unsigned intact=0;for(unsigned i=0;i<64;i++)intact+=(output[i]==0xDEADBEEFu);
      fprintf(stderr,"GUARDED TWO returned status=%d sentinels=%u/64 marks=%llu/%llu/%llu/%llu.\n",
              result,intact,(unsigned long long)marks[0],(unsigned long long)marks[1],
              (unsigned long long)marks[2],(unsigned long long)marks[3]);
      if(result!=-536870206 || intact!=64 || marks[0] || marks[1] || marks[2] || marks[3] ||
         !no_agx_metal())return 2;
      fprintf(stderr,"GUARDED TWO PASS: two-record userspace path intercepted before kernel; no GPU work.\n");
#else
      fprintf(stderr,"GUARDED TWO REFUSED: build lacks block support\n");return 2;
#endif
    }
    if(getenv("ORDERED_BATCH_PREFLIGHT") || getenv("ORDERED_VALID_BATCH_TWO")){
#ifdef G17_BLOCK_PREFLIGHT
      int split_output=getenv("ORDERED_BATCH_SPLIT_OUTPUT")!=NULL;
      int split_first=getenv("ORDERED_BATCH_SPLIT_FIRST")!=NULL;
      int self_accum=getenv("ORDERED_BATCH_SELF_ACCUM")!=NULL;
      int self_single=getenv("ORDERED_BATCH_SELF_SINGLE")!=NULL;
      int self_three=getenv("ORDERED_BATCH_SELF_THREE")!=NULL;
      int self_pattern=getenv("ORDERED_BATCH_SELF_PATTERN")!=NULL;
      int distinct_graph=getenv("ORDERED_BATCH_DISTINCT_GRAPH")!=NULL;
      int graph_program_b=getenv("ORDERED_BATCH_GRAPH_PROGRAM_B")!=NULL;
      const char* appended_code32_arm=getenv("ORDERED_BATCH_APPENDED_CODE32_ARM");
      int appended_code32_control=appended_code32_arm && !strcmp(appended_code32_arm,"control");
      int appended_code32_trial=appended_code32_arm && !strcmp(appended_code32_arm,"trial");
      int graph_three=getenv("ORDERED_BATCH_THIRD_GRAPH")!=NULL;
      int third_program_c=getenv("ORDERED_BATCH_THIRD_PROGRAM_C")!=NULL;
      int graph_four=getenv("ORDERED_BATCH_FOURTH_GRAPH")!=NULL;
      int graph_five=getenv("ORDERED_BATCH_FIFTH_GRAPH")!=NULL;
      int graph_six=getenv("ORDERED_BATCH_SIXTH_GRAPH")!=NULL;
      int graph_seven=getenv("ORDERED_BATCH_SEVENTH_GRAPH")!=NULL;
      int graph_eight=getenv("ORDERED_BATCH_EIGHTH_GRAPH")!=NULL;
      int graph_nine=getenv("ORDERED_BATCH_NINTH_GRAPH")!=NULL;
      int graph_ten=getenv("ORDERED_BATCH_TENTH_GRAPH")!=NULL;
      int seventh_alloc24=getenv("ORDERED_BATCH_SEVENTH_OUTPUT_ALLOC24")!=NULL;
      if(split_first && !split_output)return 2;
      if(self_single && !self_accum)return 2;
      if(self_three && (!self_accum || self_single))return 2;
      if(self_pattern && !self_accum)return 2;
      if(distinct_graph && (self_accum || split_output))return 2;
      if(graph_program_b && !distinct_graph)return 2;
      if(appended_code32_arm && (!graph_program_b ||
         (!appended_code32_control && !appended_code32_trial)))return 2;
      if(graph_three && (!distinct_graph || self_three || self_accum))return 2;
      if(third_program_c && (!graph_three || !graph_program_b))return 2;
      if(graph_four && (!graph_three || !third_program_c))return 2;
      if(graph_five && !graph_four)return 2;
      if(graph_six && !graph_five)return 2;
      if(graph_seven && !graph_six)return 2;
      if(seventh_alloc24 && !graph_seven)return 2;
      if(graph_eight && (!graph_seven || !seventh_alloc24 ||
                         !getenv("ORDERED_BATCH_ALLOC24_TYPE_DATA")))return 2;
      if(graph_nine && !graph_eight)return 2;
      if(graph_ten && (!graph_nine || !third_program_c))return 2;
      if(self_accum && split_output)return 2;
      if(getenv("ORDERED_ZERO_SUBMIT") || getenv("ORDERED_GUARDED_ONE") ||
         getenv("ORDERED_GUARDED_TWO") || getenv("ORDERED_VALID_ONE") ||
         getenv("ORDERED_VALID_TWO") || getenv("ORDERED_INVALID_ONE"))return 2;
      struct shmem_result graph_alloc[7]={0};
      if(distinct_graph){
        const char* request_path=getenv("ORDERED_SECOND_GRAPH_REQUESTS");
        const char* payload_path=getenv("ORDERED_SECOND_GRAPH_PAYLOAD");
        if(!request_path || !payload_path)return 2;
        FILE* request_file=fopen(request_path,"rb");
        FILE* payload_file=fopen(payload_path,"rb");
        if(!request_file || !payload_file)return 2;
        for(unsigned i=0;i<7;i++){
          uint8_t row[120];uint64_t expected_aperture=0,expected_size=0;
          if(fread(row,1,sizeof row,request_file)!=sizeof row)return 2;
          memcpy(&expected_aperture,row,8);memcpy(&expected_size,row+8,8);
          if(appended_code32_arm && i==3){
            uint32_t request_type=0;memcpy(&request_type,row+16+0x58,4);
            if(request_type!=0x08000000u ||
               expected_aperture!=0x10000120000ull || expected_size!=0x8000u)return 2;
          }
          uint64_t result_words[11]={0};size_t result_size=sizeof result_words;
          kern_return_t kr=IOConnectCallMethod(CONN,9,0,0,row+16,104,0,0,
                                                result_words,&result_size);
          fprintf(stderr,"BATCH GRAPH alloc#%u kr=0x%x bytes=%zu aperture=0x%llx size=0x%llx\n",
                  i,kr,result_size,(unsigned long long)result_words[0],
                  (unsigned long long)result_words[5]);
          if(kr || result_size!=88 || result_words[0]!=expected_aperture ||
             result_words[5]!=expected_size || !result_words[1] ||
             expected_size!=(i==6?0xc000u:0x8000u))return 2;
          uint8_t* expected=malloc((size_t)expected_size);
          if(!expected)return 2;
          if(fread(expected,1,(size_t)expected_size,payload_file)!=(size_t)expected_size)return 2;
          memcpy((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size);
          int same=memcmp((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size)==0;
          free(expected);
          if(!same || !no_agx_metal())return 2;
          graph_alloc[i].cpu=result_words[1];graph_alloc[i].size=(uint32_t)expected_size;
          graph_alloc[i].id=(uint32_t)(29+i);
        }
        if(fgetc(request_file)!=EOF || ferror(request_file) ||
           fgetc(payload_file)!=EOF || ferror(payload_file))return 2;
        fclose(request_file);fclose(payload_file);
        uint64_t arg_addresses[3]={0};
        memcpy(arg_addresses,(void*)(uintptr_t)(graph_alloc[6].cpu+0x1ba0),sizeof arg_addresses);
        uint32_t packet_word=0;
        memcpy(&packet_word,(void*)(uintptr_t)(graph_alloc[1].cpu+0x40),4);
        uint16_t packet_mid=0;
        memcpy(&packet_mid,(void*)(uintptr_t)(graph_alloc[1].cpu+0x46),2);
        uint16_t packet_high=0;
        memcpy(&packet_high,(void*)(uintptr_t)(graph_alloc[1].cpu+0x48),2);
        uint8_t b_code[sizeof KERNEL_3I1];memcpy(b_code,KERNEL_3I1,sizeof b_code);
        b_code[19]=2;
        uint8_t c_code[sizeof KERNEL_3I1];memcpy(c_code,KERNEL_3I1,sizeof c_code);
        c_code[19]=3;
        if(arg_addresses[0]!=0x10000030000ull ||
           arg_addresses[1]!=0x10000030400ull ||
           arg_addresses[2]!=0x10000030900ull ||
           packet_word!=(appended_code32_trial?0x0e400507u:
                         graph_program_b?0x0e408507u:0x0e4006c7u) ||
           packet_mid!=(appended_code32_trial?0x0012u:
                        graph_program_b?0x0001u:0u) ||
           packet_high!=0x0100u ||
           (graph_program_b && memcmp(mapped[1]+0x500,b_code,sizeof b_code)) ||
           (appended_code32_arm &&
            memcmp((void*)(uintptr_t)(graph_alloc[3].cpu+0x500),c_code,sizeof c_code)))return 2;
        if(appended_code32_arm)
          fprintf(stderr,"BATCH APPENDED CODE32 STAGE PASS: arm=%s code=0x10000120500 bytes=42 packet=0x%08x mid=0x%04x high=0x%04x output=0x%llx; CPU readback, no Submit.\n",
                  appended_code32_arm,packet_word,packet_mid,packet_high,
                  (unsigned long long)arg_addresses[2]);
        fprintf(stderr,"BATCH GRAPH RESOURCE PASS: seven appended physical allocations, packet=0x%08x, program=%c, second output=0x%llx; byte-exact CPU readback, no Submit.\n",
                packet_word,appended_code32_trial?'C':graph_program_b?'B':'A',
                (unsigned long long)arg_addresses[2]);
      }
      struct shmem_result more[2]={0};
      for(unsigned i=0;i<2;i++){
        uint64_t args[2]={0x4000,(uint64_t)i};size_t count=sizeof more[i];
        kern_return_t more_kr=IOConnectCallMethod(CONN,14,args,2,0,0,0,0,&more[i],&count);
        fprintf(stderr,"BATCH TWO sel-14#%u kr=0x%x bytes=%zu cpu=0x%llx size=0x%x id=%u\n",
                i,more_kr,count,(unsigned long long)more[i].cpu,more[i].size,more[i].id);
        if(more_kr || count!=sizeof more[i] || !more[i].cpu ||
           more[i].size!=0x4000 || more[i].id!=i+3)return 2;
      }
      uint64_t third=NextTraceID(dev),fourth=NextTraceID(dev);
      if(!third || third<=second || fourth!=third+1)return 2;
      fprintf(stderr,"BATCH TWO trace IDs first=0x%llx second=0x%llx third=0x%llx fourth=0x%llx\n",
              (unsigned long long)first,(unsigned long long)second,
              (unsigned long long)third,(unsigned long long)fourth);
      uint8_t kernel_page[0x4000],segment_page[0x4000];
      if(distinct_graph){
        const char* page_path=getenv("ORDERED_SECOND_GRAPH_PAGES");
        if(!page_path)return 2;
        FILE* pages=fopen(page_path,"rb");
        if(!pages)return 2;
        if(fread(kernel_page,1,0x4000,pages)!=0x4000 ||
           fread(segment_page,1,0x4000,pages)!=0x4000 ||
           fgetc(pages)!=EOF || ferror(pages))return 2;
        fclose(pages);
        if(kernel_page[0x176]!=0x12 || kernel_page[0x1ce]!=0x15 ||
           kernel_page[0x1d6]!=0x12 ||
           segment_page[0x48]!=0x1e || segment_page[0x8c]!=0x24 ||
           *(uint32_t*)(segment_page+0x24)!=0xf0u)return 2;
        for(unsigned i=0;i<4;i++){
          uint64_t p=0;memcpy(&p,kernel_page+0x1e4+8*i,8);
          if(p!=0x10000151b80ull+8*i)return 2;
        }
      } else {
        memcpy(kernel_page,(void*)(uintptr_t)shmem[1].cpu,0x4000);
        memcpy(segment_page,(void*)(uintptr_t)shmem[0].cpu,0x4000);
      }
      uint32_t fourth_low=(uint32_t)fourth;
      memcpy(kernel_page+0x234,&fourth_low,4);
      memcpy(segment_page+0x00,&third,8);
      memcpy(segment_page+0x18,&third,8);
      memcpy(segment_page+0x28,&fourth,8);
      uint32_t* split_target=NULL;
      if(split_output){
        uint8_t* split_page=split_first?(uint8_t*)(uintptr_t)shmem[1].cpu:kernel_page;
        uint8_t* args=mapped[28]+0x1b80;
        uint8_t* clone=mapped[28]+0x1c80;
        uint64_t old_addresses[3]={0},pointer=0;
        memcpy(old_addresses,args+0x20,sizeof old_addresses);
        if(old_addresses[0]!=0x10000030000ull ||
           old_addresses[1]!=0x10000030400ull ||
           old_addresses[2]!=0x10000030800ull)return 2;
        for(unsigned i=0;i<0x100;i++)if(clone[i])return 2;
        split_target=(uint32_t*)(mapped[20]+0x9000);
        for(unsigned i=0;i<256;i++)if(split_target[i])return 2;
        for(unsigned i=0;i<256;i++)split_target[i]=0xDEADBEEFu;
        memcpy(clone,args,0x100);
        uint64_t target_address=0x10000061000ull;
        memcpy(clone+0x30,&target_address,8);
        for(unsigned i=0;i<4;i++){
          uint64_t expected=0x100000e1b80ull+8*i;
          memcpy(&pointer,split_page+0x1e4+8*i,8);
          if(pointer!=expected)return 2;
          pointer+=0x100;
          memcpy(split_page+0x1e4+8*i,&pointer,8);
        }
        memcpy(&pointer,split_page+0x3ec,8);
        if(pointer!=old_addresses[2])return 2;
        memcpy(split_page+0x3ec,&target_address,8);
        if(memcmp(clone,args,0x30) ||
           memcmp(clone+0x38,args+0x38,0xc8) ||
           memcmp(clone+0x30,&target_address,8))return 2;
        fprintf(stderr,"BATCH SPLIT STAGE: %s kernel page arg pointers +0x100, output=0x10000061000; clone and 256 target sentinels read back.\n",
                split_first?"first":"second");
      }
      memcpy((void*)(uintptr_t)more[1].cpu,kernel_page,0x4000);
      memcpy((void*)(uintptr_t)more[0].cpu,segment_page,0x4000);
      if(memcmp((void*)(uintptr_t)more[1].cpu,kernel_page,0x4000) ||
         memcmp((void*)(uintptr_t)more[0].cpu,segment_page,0x4000) ||
         *(uint32_t*)(uintptr_t)(shmem[0].cpu+0x24)!=0xf0u ||
         *(uint32_t*)(uintptr_t)(more[0].cpu+0x24)!=0xf0u ||
         !no_agx_metal())return 2;
      struct shmem_result third_more[2]={0};
      if(graph_three){
        const char* request_path=getenv("ORDERED_THIRD_GRAPH_REQUESTS");
        const char* payload_path=getenv("ORDERED_THIRD_GRAPH_PAYLOAD");
        if(!request_path || !payload_path)return 2;
        FILE* request_file=fopen(request_path,"rb");
        FILE* payload_file=fopen(payload_path,"rb");
        if(!request_file || !payload_file)return 2;
        struct shmem_result third_alloc[7]={0};
        for(unsigned i=0;i<7;i++){
          uint8_t row[120];uint64_t expected_aperture=0,expected_size=0;
          if(fread(row,1,sizeof row,request_file)!=sizeof row)return 2;
          memcpy(&expected_aperture,row,8);memcpy(&expected_size,row+8,8);
          uint64_t result_words[11]={0};size_t result_size=sizeof result_words;
          kern_return_t kr=IOConnectCallMethod(CONN,9,0,0,row+16,104,0,0,
                                                result_words,&result_size);
          fprintf(stderr,"BATCH THIRD GRAPH alloc#%u kr=0x%x bytes=%zu aperture=0x%llx size=0x%llx\n",
                  i,kr,result_size,(unsigned long long)result_words[0],
                  (unsigned long long)result_words[5]);
          if(kr || result_size!=88 || result_words[0]!=expected_aperture ||
             result_words[5]!=expected_size || !result_words[1] ||
             expected_size!=(i==6?0xc000u:0x8000u))return 2;
          uint8_t* expected=malloc((size_t)expected_size);
          if(!expected)return 2;
          if(fread(expected,1,(size_t)expected_size,payload_file)!=(size_t)expected_size)return 2;
          memcpy((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size);
          int same=memcmp((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size)==0;
          free(expected);
          if(!same || !no_agx_metal())return 2;
          third_alloc[i].cpu=result_words[1];third_alloc[i].size=(uint32_t)expected_size;
        }
        if(fgetc(request_file)!=EOF || ferror(request_file) ||
           fgetc(payload_file)!=EOF || ferror(payload_file))return 2;
        fclose(request_file);fclose(payload_file);
        uint64_t output_address=0;
        memcpy(&output_address,(void*)(uintptr_t)(third_alloc[6].cpu+0x1bb0),8);
        uint32_t packet_word=0;
        memcpy(&packet_word,(void*)(uintptr_t)(third_alloc[1].cpu+0x40),4);
        uint16_t packet_mid=0;
        memcpy(&packet_mid,(void*)(uintptr_t)(third_alloc[1].cpu+0x46),2);
        uint8_t c_code[sizeof KERNEL_3I1];memcpy(c_code,KERNEL_3I1,sizeof c_code);
        c_code[19]=3;
        if(output_address!=0x10000030a00ull ||
           packet_word!=(third_program_c?0x0e408507u:0x0e4006c7u) ||
           packet_mid!=(third_program_c?0x0005u:0u) ||
           (third_program_c && memcmp(mapped[20]+0x500,c_code,sizeof c_code)))return 2;
        fprintf(stderr,"BATCH THIRD GRAPH RESOURCE PASS: seven appended allocations, packet=0x%08x, program=%c, output=0x%llx; byte-exact CPU readback, no Submit.\n",
                packet_word,third_program_c?'C':'A',(unsigned long long)output_address);
      }
      if(self_three || graph_three){
        for(unsigned i=0;i<2;i++){
          uint64_t args[2]={0x4000,(uint64_t)i};size_t count=sizeof third_more[i];
          kern_return_t kr=IOConnectCallMethod(CONN,14,args,2,0,0,0,0,&third_more[i],&count);
          fprintf(stderr,"BATCH THREE sel-14#%u kr=0x%x bytes=%zu size=0x%x id=%u\n",
                  i,kr,count,third_more[i].size,third_more[i].id);
          if(kr || count!=sizeof third_more[i] || !third_more[i].cpu ||
             third_more[i].size!=0x4000 || third_more[i].id!=i+5)return 2;
        }
        uint64_t fifth=NextTraceID(dev),sixth=NextTraceID(dev);
        if(fifth!=fourth+1 || sixth!=fifth+1)return 2;
        if(graph_three){
          const char* page_path=getenv("ORDERED_THIRD_GRAPH_PAGES");
          if(!page_path)return 2;
          FILE* pages=fopen(page_path,"rb");
          if(!pages)return 2;
          if(fread(kernel_page,1,0x4000,pages)!=0x4000 ||
             fread(segment_page,1,0x4000,pages)!=0x4000 ||
             fgetc(pages)!=EOF || ferror(pages))return 2;
          fclose(pages);
          if(kernel_page[0x176]!=0x19 || kernel_page[0x1ce]!=0x1c ||
             kernel_page[0x1d6]!=0x19 || segment_page[0x48]!=0x25 ||
             segment_page[0x8c]!=0x2b ||
             *(uint32_t*)(segment_page+0x24)!=0xf0u)return 2;
          for(unsigned i=0;i<4;i++){
            uint64_t p=0;memcpy(&p,kernel_page+0x1e4+8*i,8);
            if(p!=0x100001c1b80ull+8*i)return 2;
          }
        } else {
          memcpy(kernel_page,(void*)(uintptr_t)shmem[1].cpu,0x4000);
          memcpy(segment_page,(void*)(uintptr_t)shmem[0].cpu,0x4000);
        }
        uint32_t sixth_low=(uint32_t)sixth;
        memcpy(kernel_page+0x234,&sixth_low,4);
        memcpy(segment_page+0x00,&fifth,8);
        memcpy(segment_page+0x18,&fifth,8);
        memcpy(segment_page+0x28,&sixth,8);
        memcpy((void*)(uintptr_t)third_more[1].cpu,kernel_page,0x4000);
        memcpy((void*)(uintptr_t)third_more[0].cpu,segment_page,0x4000);
        if(memcmp((void*)(uintptr_t)third_more[1].cpu,kernel_page,0x4000) ||
           memcmp((void*)(uintptr_t)third_more[0].cpu,segment_page,0x4000) ||
           *(uint32_t*)(uintptr_t)(third_more[0].cpu+0x24)!=0xf0u ||
           !no_agx_metal())return 2;
        fprintf(stderr,"BATCH THREE STAGE PASS: page IDs 6/5, trace IDs fifth=0x%llx sixth=0x%llx, ready=0xf0; no Submit.\n",
                (unsigned long long)fifth,(unsigned long long)sixth);
      }
      struct shmem_result fourth_more[2]={0};
      if(graph_four){
        const char* request_path=getenv("ORDERED_FOURTH_GRAPH_REQUESTS");
        const char* payload_path=getenv("ORDERED_FOURTH_GRAPH_PAYLOAD");
        const char* page_path=getenv("ORDERED_FOURTH_GRAPH_PAGES");
        if(!request_path || !payload_path || !page_path)return 2;
        FILE* request_file=fopen(request_path,"rb");
        FILE* payload_file=fopen(payload_path,"rb");
        if(!request_file || !payload_file)return 2;
        struct shmem_result fourth_alloc[7]={0};
        for(unsigned i=0;i<7;i++){
          uint8_t row[120];uint64_t expected_aperture=0,expected_size=0;
          if(fread(row,1,sizeof row,request_file)!=sizeof row)return 2;
          memcpy(&expected_aperture,row,8);memcpy(&expected_size,row+8,8);
          uint64_t result_words[11]={0};size_t result_size=sizeof result_words;
          kern_return_t kr=IOConnectCallMethod(CONN,9,0,0,row+16,104,0,0,
                                                result_words,&result_size);
          fprintf(stderr,"BATCH FOURTH GRAPH alloc#%u kr=0x%x bytes=%zu aperture=0x%llx size=0x%llx\n",
                  i,kr,result_size,(unsigned long long)result_words[0],
                  (unsigned long long)result_words[5]);
          if(kr || result_size!=88 || result_words[0]!=expected_aperture ||
             result_words[5]!=expected_size || !result_words[1] ||
             expected_size!=(i==6?0xc000u:0x8000u))return 2;
          uint8_t* expected=malloc((size_t)expected_size);
          if(!expected)return 2;
          if(fread(expected,1,(size_t)expected_size,payload_file)!=(size_t)expected_size)return 2;
          memcpy((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size);
          int same=memcmp((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size)==0;
          free(expected);
          if(!same || !no_agx_metal())return 2;
          fourth_alloc[i].cpu=result_words[1];fourth_alloc[i].size=(uint32_t)expected_size;
        }
        if(fgetc(request_file)!=EOF || ferror(request_file) ||
           fgetc(payload_file)!=EOF || ferror(payload_file))return 2;
        fclose(request_file);fclose(payload_file);
        uint64_t fourth_output=0;
        memcpy(&fourth_output,(void*)(uintptr_t)(fourth_alloc[6].cpu+0x1bb0),8);
        uint32_t fourth_packet=0;
        memcpy(&fourth_packet,(void*)(uintptr_t)(fourth_alloc[1].cpu+0x40),4);
        if(fourth_output!=0x10000030b00ull || fourth_packet!=0x0e4006c7u)return 2;
        fprintf(stderr,"BATCH FOURTH GRAPH RESOURCE PASS: seven appended allocations, packet=0x%08x, program=A, output=0x%llx; byte-exact CPU readback, no Submit.\n",
                fourth_packet,(unsigned long long)fourth_output);
        for(unsigned i=0;i<2;i++){
          uint64_t args[2]={0x4000,(uint64_t)i};size_t count=sizeof fourth_more[i];
          kern_return_t kr=IOConnectCallMethod(CONN,14,args,2,0,0,0,0,&fourth_more[i],&count);
          fprintf(stderr,"BATCH FOUR sel-14#%u kr=0x%x bytes=%zu size=0x%x id=%u\n",
                  i,kr,count,fourth_more[i].size,fourth_more[i].id);
          if(kr || count!=sizeof fourth_more[i] || !fourth_more[i].cpu ||
             fourth_more[i].size!=0x4000 || fourth_more[i].id!=i+7)return 2;
        }
        uint64_t seventh=NextTraceID(dev),eighth=NextTraceID(dev);
        uint64_t sixth_staged=0;
        memcpy(&sixth_staged,(void*)(uintptr_t)(third_more[0].cpu+0x28),8);
        if(seventh!=sixth_staged+1 || eighth!=seventh+1)return 2;
        FILE* pages=fopen(page_path,"rb");
        if(!pages)return 2;
        if(fread(kernel_page,1,0x4000,pages)!=0x4000 ||
           fread(segment_page,1,0x4000,pages)!=0x4000 ||
           fgetc(pages)!=EOF || ferror(pages))return 2;
        fclose(pages);
        if(kernel_page[0x176]!=0x20 || kernel_page[0x1ce]!=0x23 ||
           kernel_page[0x1d6]!=0x20 || segment_page[0x48]!=0x2c ||
           segment_page[0x8c]!=0x32 ||
           *(uint32_t*)(segment_page+0x24)!=0xf0u)return 2;
        for(unsigned i=0;i<4;i++){
          uint64_t p=0;memcpy(&p,kernel_page+0x1e4+8*i,8);
          if(p!=0x10000231b80ull+8*i)return 2;
        }
        uint32_t eighth_low=(uint32_t)eighth;
        memcpy(kernel_page+0x234,&eighth_low,4);
        memcpy(segment_page+0x00,&seventh,8);
        memcpy(segment_page+0x18,&seventh,8);
        memcpy(segment_page+0x28,&eighth,8);
        memcpy((void*)(uintptr_t)fourth_more[1].cpu,kernel_page,0x4000);
        memcpy((void*)(uintptr_t)fourth_more[0].cpu,segment_page,0x4000);
        if(memcmp((void*)(uintptr_t)fourth_more[1].cpu,kernel_page,0x4000) ||
           memcmp((void*)(uintptr_t)fourth_more[0].cpu,segment_page,0x4000) ||
           *(uint32_t*)(uintptr_t)(fourth_more[0].cpu+0x24)!=0xf0u ||
           !no_agx_metal())return 2;
        for(unsigned i=0;i<64;i++){
          uint32_t a=0,b=0;
          memcpy(&a,mapped[2]+4*i,4);memcpy(&b,mapped[2]+0x400+4*i,4);
          if(a!=11*i+5 || b!=7*i+3)return 2;
        }
        fprintf(stderr,"BATCH FOUR STAGE PASS: page IDs 8/7, trace IDs seventh=0x%llx eighth=0x%llx, ready=0xf0, A/B inputs=64/64; no Submit.\n",
                (unsigned long long)seventh,(unsigned long long)eighth);
      }
      struct shmem_result fifth_more[2]={0};
      uint32_t* fifth_target=(uint32_t*)(mapped[20]+0x9000);
      if(graph_five){
        const char* request_path=getenv("ORDERED_FIFTH_GRAPH_REQUESTS");
        const char* payload_path=getenv("ORDERED_FIFTH_GRAPH_PAYLOAD");
        const char* page_path=getenv("ORDERED_FIFTH_GRAPH_PAGES");
        if(!request_path || !payload_path || !page_path)return 2;
        FILE* request_file=fopen(request_path,"rb");
        FILE* payload_file=fopen(payload_path,"rb");
        if(!request_file || !payload_file)return 2;
        struct shmem_result fifth_alloc[7]={0};
        for(unsigned i=0;i<7;i++){
          uint8_t row[120];uint64_t expected_aperture=0,expected_size=0;
          if(fread(row,1,sizeof row,request_file)!=sizeof row)return 2;
          memcpy(&expected_aperture,row,8);memcpy(&expected_size,row+8,8);
          uint64_t result_words[11]={0};size_t result_size=sizeof result_words;
          kern_return_t kr=IOConnectCallMethod(CONN,9,0,0,row+16,104,0,0,
                                                result_words,&result_size);
          fprintf(stderr,"BATCH FIFTH GRAPH alloc#%u kr=0x%x bytes=%zu aperture=0x%llx size=0x%llx\n",
                  i,kr,result_size,(unsigned long long)result_words[0],
                  (unsigned long long)result_words[5]);
          if(kr || result_size!=88 || result_words[0]!=expected_aperture ||
             result_words[5]!=expected_size || !result_words[1] ||
             expected_size!=(i==6?0xc000u:0x8000u))return 2;
          uint8_t* expected=malloc((size_t)expected_size);
          if(!expected)return 2;
          if(fread(expected,1,(size_t)expected_size,payload_file)!=(size_t)expected_size)return 2;
          memcpy((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size);
          int same=memcmp((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size)==0;
          free(expected);
          if(!same || !no_agx_metal())return 2;
          fifth_alloc[i].cpu=result_words[1];fifth_alloc[i].size=(uint32_t)expected_size;
        }
        if(fgetc(request_file)!=EOF || ferror(request_file) ||
           fgetc(payload_file)!=EOF || ferror(payload_file))return 2;
        fclose(request_file);fclose(payload_file);
        uint64_t fifth_output=0;
        memcpy(&fifth_output,(void*)(uintptr_t)(fifth_alloc[6].cpu+0x1bb0),8);
        uint32_t fifth_packet=0;
        memcpy(&fifth_packet,(void*)(uintptr_t)(fifth_alloc[1].cpu+0x40),4);
        if(fifth_output!=0x10000061000ull || fifth_packet!=0x0e4006c7u)return 2;
        fprintf(stderr,"BATCH FIFTH GRAPH RESOURCE PASS: seven appended allocations, packet=0x%08x, program=A, output=0x%llx; byte-exact CPU readback, no Submit.\n",
                fifth_packet,(unsigned long long)fifth_output);
        for(unsigned i=0;i<2;i++){
          uint64_t args[2]={0x4000,(uint64_t)i};size_t count=sizeof fifth_more[i];
          kern_return_t kr=IOConnectCallMethod(CONN,14,args,2,0,0,0,0,&fifth_more[i],&count);
          fprintf(stderr,"BATCH FIVE sel-14#%u kr=0x%x bytes=%zu size=0x%x id=%u\n",
                  i,kr,count,fifth_more[i].size,fifth_more[i].id);
          if(kr || count!=sizeof fifth_more[i] || !fifth_more[i].cpu ||
             fifth_more[i].size!=0x4000 || fifth_more[i].id!=i+9)return 2;
        }
        uint64_t ninth=NextTraceID(dev),tenth=NextTraceID(dev);
        uint64_t eighth_staged=0;
        memcpy(&eighth_staged,(void*)(uintptr_t)(fourth_more[0].cpu+0x28),8);
        if(ninth!=eighth_staged+1 || tenth!=ninth+1)return 2;
        FILE* pages=fopen(page_path,"rb");
        if(!pages)return 2;
        if(fread(kernel_page,1,0x4000,pages)!=0x4000 ||
           fread(segment_page,1,0x4000,pages)!=0x4000 ||
           fgetc(pages)!=EOF || ferror(pages))return 2;
        fclose(pages);
        if(kernel_page[0x176]!=0x27 || kernel_page[0x1ce]!=0x2a ||
           kernel_page[0x1d6]!=0x27 || segment_page[0x48]!=0x33 ||
           segment_page[0x8c]!=0x39 ||
           *(uint32_t*)(segment_page+0x24)!=0xf0u)return 2;
        for(unsigned i=0;i<4;i++){
          uint64_t p=0;memcpy(&p,kernel_page+0x1e4+8*i,8);
          if(p!=0x100002a1b80ull+8*i)return 2;
        }
        uint32_t tenth_low=(uint32_t)tenth;
        memcpy(kernel_page+0x234,&tenth_low,4);
        memcpy(segment_page+0x00,&ninth,8);
        memcpy(segment_page+0x18,&ninth,8);
        memcpy(segment_page+0x28,&tenth,8);
        memcpy((void*)(uintptr_t)fifth_more[1].cpu,kernel_page,0x4000);
        memcpy((void*)(uintptr_t)fifth_more[0].cpu,segment_page,0x4000);
        if(memcmp((void*)(uintptr_t)fifth_more[1].cpu,kernel_page,0x4000) ||
           memcmp((void*)(uintptr_t)fifth_more[0].cpu,segment_page,0x4000) ||
           *(uint32_t*)(uintptr_t)(fifth_more[0].cpu+0x24)!=0xf0u ||
           !no_agx_metal())return 2;
        for(unsigned i=0;i<256;i++){
          if(fifth_target[i]!=0u)return 2;
          fifth_target[i]=0xDEADBEEFu;
        }
        unsigned sentinels=0;
        for(unsigned i=0;i<256;i++)sentinels+=(fifth_target[i]==0xDEADBEEFu);
        if(sentinels!=256)return 2;
        fprintf(stderr,"BATCH FIVE STAGE PASS: page IDs 10/9, trace IDs ninth=0x%llx tenth=0x%llx, ready=0xf0, separate target sentinels=256/256; no Submit.\n",
                (unsigned long long)ninth,(unsigned long long)tenth);
      }
      struct shmem_result sixth_more[2]={0};
      uint32_t* sixth_target=(uint32_t*)(mapped[20]+0xa000);
      if(graph_six){
        const char* request_path=getenv("ORDERED_SIXTH_GRAPH_REQUESTS");
        const char* payload_path=getenv("ORDERED_SIXTH_GRAPH_PAYLOAD");
        const char* page_path=getenv("ORDERED_SIXTH_GRAPH_PAGES");
        if(!request_path || !payload_path || !page_path)return 2;
        FILE* request_file=fopen(request_path,"rb");
        FILE* payload_file=fopen(payload_path,"rb");
        if(!request_file || !payload_file)return 2;
        struct shmem_result sixth_alloc[7]={0};
        for(unsigned i=0;i<7;i++){
          uint8_t row[120];uint64_t expected_aperture=0,expected_size=0;
          if(fread(row,1,sizeof row,request_file)!=sizeof row)return 2;
          memcpy(&expected_aperture,row,8);memcpy(&expected_size,row+8,8);
          uint64_t result_words[11]={0};size_t result_size=sizeof result_words;
          kern_return_t kr=IOConnectCallMethod(CONN,9,0,0,row+16,104,0,0,
                                                result_words,&result_size);
          fprintf(stderr,"BATCH SIXTH GRAPH alloc#%u kr=0x%x bytes=%zu aperture=0x%llx size=0x%llx\n",
                  i,kr,result_size,(unsigned long long)result_words[0],
                  (unsigned long long)result_words[5]);
          if(kr || result_size!=88 || result_words[0]!=expected_aperture ||
             result_words[5]!=expected_size || !result_words[1] ||
             expected_size!=(i==6?0xc000u:0x8000u))return 2;
          uint8_t* expected=malloc((size_t)expected_size);
          if(!expected)return 2;
          if(fread(expected,1,(size_t)expected_size,payload_file)!=(size_t)expected_size)return 2;
          memcpy((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size);
          int same=memcmp((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size)==0;
          free(expected);
          if(!same || !no_agx_metal())return 2;
          sixth_alloc[i].cpu=result_words[1];sixth_alloc[i].size=(uint32_t)expected_size;
        }
        if(fgetc(request_file)!=EOF || ferror(request_file) ||
           fgetc(payload_file)!=EOF || ferror(payload_file))return 2;
        fclose(request_file);fclose(payload_file);
        uint64_t sixth_output=0;
        memcpy(&sixth_output,(void*)(uintptr_t)(sixth_alloc[6].cpu+0x1bb0),8);
        uint32_t sixth_packet=0;
        memcpy(&sixth_packet,(void*)(uintptr_t)(sixth_alloc[1].cpu+0x40),4);
        if(sixth_output!=0x10000062000ull || sixth_packet!=0x0e4006c7u)return 2;
        fprintf(stderr,"BATCH SIXTH GRAPH RESOURCE PASS: seven appended allocations, packet=0x%08x, program=A, output=0x%llx; byte-exact CPU readback, no Submit.\n",
                sixth_packet,(unsigned long long)sixth_output);
        for(unsigned i=0;i<2;i++){
          uint64_t args[2]={0x4000,(uint64_t)i};size_t count=sizeof sixth_more[i];
          kern_return_t kr=IOConnectCallMethod(CONN,14,args,2,0,0,0,0,&sixth_more[i],&count);
          fprintf(stderr,"BATCH SIX sel-14#%u kr=0x%x bytes=%zu size=0x%x id=%u\n",
                  i,kr,count,sixth_more[i].size,sixth_more[i].id);
          if(kr || count!=sizeof sixth_more[i] || !sixth_more[i].cpu ||
             sixth_more[i].size!=0x4000 || sixth_more[i].id!=i+11)return 2;
        }
        uint64_t eleventh=NextTraceID(dev),twelfth=NextTraceID(dev);
        uint64_t tenth_staged=0;
        memcpy(&tenth_staged,(void*)(uintptr_t)(fifth_more[0].cpu+0x28),8);
        if(eleventh!=tenth_staged+1 || twelfth!=eleventh+1)return 2;
        FILE* pages=fopen(page_path,"rb");
        if(!pages)return 2;
        if(fread(kernel_page,1,0x4000,pages)!=0x4000 ||
           fread(segment_page,1,0x4000,pages)!=0x4000 ||
           fgetc(pages)!=EOF || ferror(pages))return 2;
        fclose(pages);
        if(kernel_page[0x176]!=0x2e || kernel_page[0x1ce]!=0x31 ||
           kernel_page[0x1d6]!=0x2e || segment_page[0x48]!=0x3a ||
           segment_page[0x8c]!=0x40 ||
           *(uint32_t*)(segment_page+0x24)!=0xf0u)return 2;
        for(unsigned i=0;i<4;i++){
          uint64_t p=0;memcpy(&p,kernel_page+0x1e4+8*i,8);
          if(p!=0x10000311b80ull+8*i)return 2;
        }
        uint32_t twelfth_low=(uint32_t)twelfth;
        memcpy(kernel_page+0x234,&twelfth_low,4);
        memcpy(segment_page+0x00,&eleventh,8);
        memcpy(segment_page+0x18,&eleventh,8);
        memcpy(segment_page+0x28,&twelfth,8);
        memcpy((void*)(uintptr_t)sixth_more[1].cpu,kernel_page,0x4000);
        memcpy((void*)(uintptr_t)sixth_more[0].cpu,segment_page,0x4000);
        if(memcmp((void*)(uintptr_t)sixth_more[1].cpu,kernel_page,0x4000) ||
           memcmp((void*)(uintptr_t)sixth_more[0].cpu,segment_page,0x4000) ||
           *(uint32_t*)(uintptr_t)(sixth_more[0].cpu+0x24)!=0xf0u ||
           !no_agx_metal())return 2;
        for(unsigned i=0;i<256;i++){
          if(sixth_target[i]!=0u)return 2;
          sixth_target[i]=0xDEADBEEFu;
        }
        unsigned sentinels=0;
        for(unsigned i=0;i<256;i++)sentinels+=(sixth_target[i]==0xDEADBEEFu);
        if(sentinels!=256)return 2;
        fprintf(stderr,"BATCH SIX STAGE PASS: page IDs 12/11, trace IDs eleventh=0x%llx twelfth=0x%llx, ready=0xf0, separate target sentinels=256/256; no Submit.\n",
                (unsigned long long)eleventh,(unsigned long long)twelfth);
      }
      struct shmem_result seventh_more[2]={0};
      uint32_t* seventh_target=(uint32_t*)(seventh_alloc24?
        mapped[24]+0x2000:mapped[20]+0xb000);
      if(graph_seven){
        const char* request_path=getenv("ORDERED_SEVENTH_GRAPH_REQUESTS");
        const char* payload_path=getenv("ORDERED_SEVENTH_GRAPH_PAYLOAD");
        const char* page_path=getenv("ORDERED_SEVENTH_GRAPH_PAGES");
        if(!request_path || !payload_path || !page_path)return 2;
        FILE* request_file=fopen(request_path,"rb");
        FILE* payload_file=fopen(payload_path,"rb");
        if(!request_file || !payload_file)return 2;
        struct shmem_result seventh_alloc[7]={0};
        for(unsigned i=0;i<7;i++){
          uint8_t row[120];uint64_t expected_aperture=0,expected_size=0;
          if(fread(row,1,sizeof row,request_file)!=sizeof row)return 2;
          memcpy(&expected_aperture,row,8);memcpy(&expected_size,row+8,8);
          uint64_t result_words[11]={0};size_t result_size=sizeof result_words;
          kern_return_t kr=IOConnectCallMethod(CONN,9,0,0,row+16,104,0,0,
                                                result_words,&result_size);
          fprintf(stderr,"BATCH SEVENTH GRAPH alloc#%u kr=0x%x bytes=%zu aperture=0x%llx size=0x%llx\n",
                  i,kr,result_size,(unsigned long long)result_words[0],
                  (unsigned long long)result_words[5]);
          if(kr || result_size!=88 || result_words[0]!=expected_aperture ||
             result_words[5]!=expected_size || !result_words[1] ||
             expected_size!=(i==6?0xc000u:0x8000u))return 2;
          uint8_t* expected=malloc((size_t)expected_size);
          if(!expected)return 2;
          if(fread(expected,1,(size_t)expected_size,payload_file)!=(size_t)expected_size)return 2;
          memcpy((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size);
          int same=memcmp((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size)==0;
          free(expected);
          if(!same || !no_agx_metal())return 2;
          seventh_alloc[i].cpu=result_words[1];seventh_alloc[i].size=(uint32_t)expected_size;
        }
        if(fgetc(request_file)!=EOF || ferror(request_file) ||
           fgetc(payload_file)!=EOF || ferror(payload_file))return 2;
        fclose(request_file);fclose(payload_file);
        uint64_t seventh_output=0;
        memcpy(&seventh_output,(void*)(uintptr_t)(seventh_alloc[6].cpu+0x1bb0),8);
        uint32_t seventh_packet=0;
        memcpy(&seventh_packet,(void*)(uintptr_t)(seventh_alloc[1].cpu+0x40),4);
        if(seventh_output!=(seventh_alloc24?0x100000a2000ull:0x10000063000ull) ||
           seventh_packet!=0x0e4006c7u)return 2;
        fprintf(stderr,"BATCH SEVENTH GRAPH PREDICTED RESOURCE PASS: seven appended allocations, packet=0x%08x, program=A, output=0x%llx; byte-exact CPU readback, no Submit.\n",
                seventh_packet,(unsigned long long)seventh_output);
        for(unsigned i=0;i<2;i++){
          uint64_t args[2]={0x4000,(uint64_t)i};size_t count=sizeof seventh_more[i];
          kern_return_t kr=IOConnectCallMethod(CONN,14,args,2,0,0,0,0,&seventh_more[i],&count);
          fprintf(stderr,"BATCH SEVEN sel-14#%u kr=0x%x bytes=%zu size=0x%x id=%u\n",
                  i,kr,count,seventh_more[i].size,seventh_more[i].id);
          if(kr || count!=sizeof seventh_more[i] || !seventh_more[i].cpu ||
             seventh_more[i].size!=0x4000 || seventh_more[i].id!=i+13)return 2;
        }
        uint64_t thirteenth=NextTraceID(dev),fourteenth=NextTraceID(dev);
        uint64_t twelfth_staged=0;
        memcpy(&twelfth_staged,(void*)(uintptr_t)(sixth_more[0].cpu+0x28),8);
        if(thirteenth!=twelfth_staged+1 || fourteenth!=thirteenth+1)return 2;
        FILE* pages=fopen(page_path,"rb");
        if(!pages)return 2;
        if(fread(kernel_page,1,0x4000,pages)!=0x4000 ||
           fread(segment_page,1,0x4000,pages)!=0x4000 ||
           fgetc(pages)!=EOF || ferror(pages))return 2;
        fclose(pages);
        if(kernel_page[0x176]!=0x35 || kernel_page[0x1ce]!=0x38 ||
           kernel_page[0x1d6]!=0x35 || segment_page[0x48]!=0x41 ||
           segment_page[0x8c]!=0x47 ||
           *(uint32_t*)(segment_page+0x24)!=0xf0u)return 2;
        for(unsigned i=0;i<4;i++){
          uint64_t p=0;memcpy(&p,kernel_page+0x1e4+8*i,8);
          if(p!=0x10000381b80ull+8*i)return 2;
        }
        uint32_t fourteenth_low=(uint32_t)fourteenth;
        memcpy(kernel_page+0x234,&fourteenth_low,4);
        memcpy(segment_page+0x00,&thirteenth,8);
        memcpy(segment_page+0x18,&thirteenth,8);
        memcpy(segment_page+0x28,&fourteenth,8);
        memcpy((void*)(uintptr_t)seventh_more[1].cpu,kernel_page,0x4000);
        memcpy((void*)(uintptr_t)seventh_more[0].cpu,segment_page,0x4000);
        if(memcmp((void*)(uintptr_t)seventh_more[1].cpu,kernel_page,0x4000) ||
           memcmp((void*)(uintptr_t)seventh_more[0].cpu,segment_page,0x4000) ||
           *(uint32_t*)(uintptr_t)(seventh_more[0].cpu+0x24)!=0xf0u ||
           !no_agx_metal())return 2;
        for(unsigned i=0;i<256;i++){
          if(seventh_target[i]!=0u)return 2;
          seventh_target[i]=0xDEADBEEFu;
        }
        unsigned sentinels=0;
        for(unsigned i=0;i<256;i++)sentinels+=(seventh_target[i]==0xDEADBEEFu);
        if(sentinels!=256)return 2;
        fprintf(stderr,"BATCH SEVEN PREDICTED STAGE PASS: page IDs 14/13, trace IDs thirteenth=0x%llx fourteenth=0x%llx, ready=0xf0, separate target sentinels=256/256; no Submit.\n",
                (unsigned long long)thirteenth,(unsigned long long)fourteenth);
      }
      struct shmem_result eighth_more[2]={0};
      uint32_t* eighth_target=(uint32_t*)(mapped[24]+0x3000);
      if(graph_eight){
        const char* request_path=getenv("ORDERED_EIGHTH_GRAPH_REQUESTS");
        const char* payload_path=getenv("ORDERED_EIGHTH_GRAPH_PAYLOAD");
        const char* page_path=getenv("ORDERED_EIGHTH_GRAPH_PAGES");
        if(!request_path || !payload_path || !page_path)return 2;
        FILE* request_file=fopen(request_path,"rb");
        FILE* payload_file=fopen(payload_path,"rb");
        if(!request_file || !payload_file)return 2;
        struct shmem_result eighth_alloc[7]={0};
        for(unsigned i=0;i<7;i++){
          uint8_t row[120];uint64_t expected_aperture=0,expected_size=0;
          if(fread(row,1,sizeof row,request_file)!=sizeof row)return 2;
          memcpy(&expected_aperture,row,8);memcpy(&expected_size,row+8,8);
          uint64_t result_words[11]={0};size_t result_size=sizeof result_words;
          kern_return_t kr=IOConnectCallMethod(CONN,9,0,0,row+16,104,0,0,
                                                result_words,&result_size);
          fprintf(stderr,"BATCH EIGHTH GRAPH alloc#%u kr=0x%x bytes=%zu aperture=0x%llx size=0x%llx\n",
                  i,kr,result_size,(unsigned long long)result_words[0],
                  (unsigned long long)result_words[5]);
          if(kr || result_size!=88 || result_words[0]!=expected_aperture ||
             result_words[5]!=expected_size || !result_words[1] ||
             expected_size!=(i==6?0xc000u:0x8000u))return 2;
          uint8_t* expected=malloc((size_t)expected_size);
          if(!expected)return 2;
          if(fread(expected,1,(size_t)expected_size,payload_file)!=(size_t)expected_size)return 2;
          memcpy((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size);
          int same=memcmp((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size)==0;
          free(expected);
          if(!same || !no_agx_metal())return 2;
          eighth_alloc[i].cpu=result_words[1];eighth_alloc[i].size=(uint32_t)expected_size;
        }
        if(fgetc(request_file)!=EOF || ferror(request_file) ||
           fgetc(payload_file)!=EOF || ferror(payload_file))return 2;
        fclose(request_file);fclose(payload_file);
        uint64_t eighth_output=0;
        memcpy(&eighth_output,(void*)(uintptr_t)(eighth_alloc[6].cpu+0x1bb0),8);
        uint32_t eighth_packet=0;
        memcpy(&eighth_packet,(void*)(uintptr_t)(eighth_alloc[1].cpu+0x40),4);
        if(eighth_output!=0x100000a3000ull ||
           eighth_packet!=0x0e4006c7u)return 2;
        fprintf(stderr,"BATCH EIGHTH GRAPH PREDICTED RESOURCE PASS: seven appended allocations, packet=0x%08x, program=A, output=0x%llx; byte-exact CPU readback, no Submit.\n",
                eighth_packet,(unsigned long long)eighth_output);
        for(unsigned i=0;i<2;i++){
          uint64_t args[2]={0x4000,(uint64_t)i};size_t count=sizeof eighth_more[i];
          kern_return_t kr=IOConnectCallMethod(CONN,14,args,2,0,0,0,0,&eighth_more[i],&count);
          fprintf(stderr,"BATCH EIGHT sel-14#%u kr=0x%x bytes=%zu size=0x%x id=%u\n",
                  i,kr,count,eighth_more[i].size,eighth_more[i].id);
          if(kr || count!=sizeof eighth_more[i] || !eighth_more[i].cpu ||
             eighth_more[i].size!=0x4000 || eighth_more[i].id!=i+15)return 2;
        }
        uint64_t fifteenth=NextTraceID(dev),sixteenth=NextTraceID(dev);
        uint64_t fourteenth_staged=0;
        memcpy(&fourteenth_staged,(void*)(uintptr_t)(seventh_more[0].cpu+0x28),8);
        if(fifteenth!=fourteenth_staged+1 || sixteenth!=fifteenth+1)return 2;
        FILE* pages=fopen(page_path,"rb");
        if(!pages)return 2;
        if(fread(kernel_page,1,0x4000,pages)!=0x4000 ||
           fread(segment_page,1,0x4000,pages)!=0x4000 ||
           fgetc(pages)!=EOF || ferror(pages))return 2;
        fclose(pages);
        if(kernel_page[0x176]!=0x3c || kernel_page[0x1ce]!=0x3f ||
           kernel_page[0x1d6]!=0x3c || segment_page[0x48]!=0x48 ||
           segment_page[0x8c]!=0x4e ||
           *(uint32_t*)(segment_page+0x24)!=0xf0u)return 2;
        for(unsigned i=0;i<4;i++){
          uint64_t p=0;memcpy(&p,kernel_page+0x1e4+8*i,8);
          if(p!=0x100003f1b80ull+8*i)return 2;
        }
        uint32_t sixteenth_low=(uint32_t)sixteenth;
        memcpy(kernel_page+0x234,&sixteenth_low,4);
        memcpy(segment_page+0x00,&fifteenth,8);
        memcpy(segment_page+0x18,&fifteenth,8);
        memcpy(segment_page+0x28,&sixteenth,8);
        memcpy((void*)(uintptr_t)eighth_more[1].cpu,kernel_page,0x4000);
        memcpy((void*)(uintptr_t)eighth_more[0].cpu,segment_page,0x4000);
        if(memcmp((void*)(uintptr_t)eighth_more[1].cpu,kernel_page,0x4000) ||
           memcmp((void*)(uintptr_t)eighth_more[0].cpu,segment_page,0x4000) ||
           *(uint32_t*)(uintptr_t)(eighth_more[0].cpu+0x24)!=0xf0u ||
           !no_agx_metal())return 2;
        for(unsigned i=0;i<256;i++){
          if(eighth_target[i]!=0u)return 2;
          eighth_target[i]=0xDEADBEEFu;
        }
        unsigned sentinels=0;
        for(unsigned i=0;i<256;i++)sentinels+=(eighth_target[i]==0xDEADBEEFu);
        if(sentinels!=256)return 2;
        fprintf(stderr,"BATCH EIGHT PREDICTED STAGE PASS: page IDs 16/15, trace IDs fifteenth=0x%llx sixteenth=0x%llx, ready=0xf0, separate target sentinels=256/256; no Submit.\n",
                (unsigned long long)fifteenth,(unsigned long long)sixteenth);
      }
      struct shmem_result ninth_more[2]={0};
      uint32_t* ninth_target=(uint32_t*)(mapped[24]+0x4000);
      if(graph_nine){
        const char* request_path=getenv("ORDERED_NINTH_GRAPH_REQUESTS");
        const char* payload_path=getenv("ORDERED_NINTH_GRAPH_PAYLOAD");
        const char* page_path=getenv("ORDERED_NINTH_GRAPH_PAGES");
        if(!request_path || !payload_path || !page_path)return 2;
        FILE* request_file=fopen(request_path,"rb");
        FILE* payload_file=fopen(payload_path,"rb");
        if(!request_file || !payload_file)return 2;
        struct shmem_result ninth_alloc[7]={0};
        for(unsigned i=0;i<7;i++){
          uint8_t row[120];uint64_t expected_aperture=0,expected_size=0;
          if(fread(row,1,sizeof row,request_file)!=sizeof row)return 2;
          memcpy(&expected_aperture,row,8);memcpy(&expected_size,row+8,8);
          uint64_t result_words[11]={0};size_t result_size=sizeof result_words;
          kern_return_t kr=IOConnectCallMethod(CONN,9,0,0,row+16,104,0,0,
                                                result_words,&result_size);
          fprintf(stderr,"BATCH NINTH GRAPH alloc#%u kr=0x%x bytes=%zu aperture=0x%llx size=0x%llx\n",
                  i,kr,result_size,(unsigned long long)result_words[0],
                  (unsigned long long)result_words[5]);
          if(kr || result_size!=88 || result_words[0]!=expected_aperture ||
             result_words[5]!=expected_size || !result_words[1] ||
             expected_size!=(i==6?0xc000u:0x8000u))return 2;
          uint8_t* expected=malloc((size_t)expected_size);
          if(!expected)return 2;
          if(fread(expected,1,(size_t)expected_size,payload_file)!=(size_t)expected_size)return 2;
          memcpy((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size);
          int same=memcmp((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size)==0;
          free(expected);
          if(!same || !no_agx_metal())return 2;
          ninth_alloc[i].cpu=result_words[1];ninth_alloc[i].size=(uint32_t)expected_size;
        }
        if(fgetc(request_file)!=EOF || ferror(request_file) ||
           fgetc(payload_file)!=EOF || ferror(payload_file))return 2;
        fclose(request_file);fclose(payload_file);
        uint64_t ninth_output=0;
        memcpy(&ninth_output,(void*)(uintptr_t)(ninth_alloc[6].cpu+0x1bb0),8);
        uint32_t ninth_packet=0;
        memcpy(&ninth_packet,(void*)(uintptr_t)(ninth_alloc[1].cpu+0x40),4);
        if(ninth_output!=0x100000a4000ull ||
           ninth_packet!=0x0e4006c7u)return 2;
        fprintf(stderr,"BATCH NINTH GRAPH PREDICTED RESOURCE PASS: seven appended allocations, packet=0x%08x, program=A, output=0x%llx; byte-exact CPU readback, no Submit.\n",
                ninth_packet,(unsigned long long)ninth_output);
        for(unsigned i=0;i<2;i++){
          uint64_t args[2]={0x4000,(uint64_t)i};size_t count=sizeof ninth_more[i];
          kern_return_t kr=IOConnectCallMethod(CONN,14,args,2,0,0,0,0,&ninth_more[i],&count);
          fprintf(stderr,"BATCH NINE sel-14#%u kr=0x%x bytes=%zu size=0x%x id=%u\n",
                  i,kr,count,ninth_more[i].size,ninth_more[i].id);
          if(kr || count!=sizeof ninth_more[i] || !ninth_more[i].cpu ||
             ninth_more[i].size!=0x4000 || ninth_more[i].id!=i+17)return 2;
        }
        uint64_t seventeenth=NextTraceID(dev),eighteenth=NextTraceID(dev);
        uint64_t sixteenth_staged=0;
        memcpy(&sixteenth_staged,(void*)(uintptr_t)(eighth_more[0].cpu+0x28),8);
        if(seventeenth!=sixteenth_staged+1 || eighteenth!=seventeenth+1)return 2;
        FILE* pages=fopen(page_path,"rb");
        if(!pages)return 2;
        if(fread(kernel_page,1,0x4000,pages)!=0x4000 ||
           fread(segment_page,1,0x4000,pages)!=0x4000 ||
           fgetc(pages)!=EOF || ferror(pages))return 2;
        fclose(pages);
        if(kernel_page[0x176]!=0x43 || kernel_page[0x1ce]!=0x46 ||
           kernel_page[0x1d6]!=0x43 || segment_page[0x48]!=0x4f ||
           segment_page[0x8c]!=0x55 ||
           *(uint32_t*)(segment_page+0x24)!=0xf0u)return 2;
        for(unsigned i=0;i<4;i++){
          uint64_t p=0;memcpy(&p,kernel_page+0x1e4+8*i,8);
          if(p!=0x10000461b80ull+8*i)return 2;
        }
        uint32_t eighteenth_low=(uint32_t)eighteenth;
        memcpy(kernel_page+0x234,&eighteenth_low,4);
        memcpy(segment_page+0x00,&seventeenth,8);
        memcpy(segment_page+0x18,&seventeenth,8);
        memcpy(segment_page+0x28,&eighteenth,8);
        memcpy((void*)(uintptr_t)ninth_more[1].cpu,kernel_page,0x4000);
        memcpy((void*)(uintptr_t)ninth_more[0].cpu,segment_page,0x4000);
        if(memcmp((void*)(uintptr_t)ninth_more[1].cpu,kernel_page,0x4000) ||
           memcmp((void*)(uintptr_t)ninth_more[0].cpu,segment_page,0x4000) ||
           *(uint32_t*)(uintptr_t)(ninth_more[0].cpu+0x24)!=0xf0u ||
           !no_agx_metal())return 2;
        for(unsigned i=0;i<256;i++){
          if(ninth_target[i]!=0u)return 2;
          ninth_target[i]=0xDEADBEEFu;
        }
        unsigned sentinels=0;
        for(unsigned i=0;i<256;i++)sentinels+=(ninth_target[i]==0xDEADBEEFu);
        if(sentinels!=256)return 2;
        fprintf(stderr,"BATCH NINE PREDICTED STAGE PASS: page IDs 18/17, trace IDs seventeenth=0x%llx eighteenth=0x%llx, ready=0xf0, separate target sentinels=256/256; no Submit.\n",
                (unsigned long long)seventeenth,(unsigned long long)eighteenth);
      }
      struct shmem_result tenth_more[2]={0};
      uint32_t* tenth_target=(uint32_t*)(mapped[24]+0x5000);
      if(graph_ten){
        const char* request_path=getenv("ORDERED_TENTH_GRAPH_REQUESTS");
        const char* payload_path=getenv("ORDERED_TENTH_GRAPH_PAYLOAD");
        const char* page_path=getenv("ORDERED_TENTH_GRAPH_PAGES");
        if(!request_path || !payload_path || !page_path)return 2;
        FILE* request_file=fopen(request_path,"rb");
        FILE* payload_file=fopen(payload_path,"rb");
        if(!request_file || !payload_file)return 2;
        struct shmem_result tenth_alloc[7]={0};
        for(unsigned i=0;i<7;i++){
          uint8_t row[120];uint64_t expected_aperture=0,expected_size=0;
          if(fread(row,1,sizeof row,request_file)!=sizeof row)return 2;
          memcpy(&expected_aperture,row,8);memcpy(&expected_size,row+8,8);
          uint64_t result_words[11]={0};size_t result_size=sizeof result_words;
          kern_return_t kr=IOConnectCallMethod(CONN,9,0,0,row+16,104,0,0,
                                                result_words,&result_size);
          fprintf(stderr,"BATCH TENTH GRAPH alloc#%u kr=0x%x bytes=%zu aperture=0x%llx size=0x%llx\n",
                  i,kr,result_size,(unsigned long long)result_words[0],
                  (unsigned long long)result_words[5]);
          if(kr || result_size!=88 || result_words[0]!=expected_aperture ||
             result_words[5]!=expected_size || !result_words[1] ||
             expected_size!=(i==6?0xc000u:0x8000u))return 2;
          uint8_t* expected=malloc((size_t)expected_size);
          if(!expected)return 2;
          if(fread(expected,1,(size_t)expected_size,payload_file)!=(size_t)expected_size)return 2;
          memcpy((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size);
          int same=memcmp((void*)(uintptr_t)result_words[1],expected,(size_t)expected_size)==0;
          free(expected);
          if(!same || !no_agx_metal())return 2;
          tenth_alloc[i].cpu=result_words[1];tenth_alloc[i].size=(uint32_t)expected_size;
        }
        if(fgetc(request_file)!=EOF || ferror(request_file) ||
           fgetc(payload_file)!=EOF || ferror(payload_file))return 2;
        fclose(request_file);fclose(payload_file);
        uint64_t tenth_output=0;
        memcpy(&tenth_output,(void*)(uintptr_t)(tenth_alloc[6].cpu+0x1bb0),8);
        uint32_t tenth_packet=0;uint16_t tenth_packet_mid=0;
        memcpy(&tenth_packet,(void*)(uintptr_t)(tenth_alloc[1].cpu+0x40),4);
        memcpy(&tenth_packet_mid,(void*)(uintptr_t)(tenth_alloc[1].cpu+0x46),2);
        if(tenth_output!=0x100000a5000ull ||
           tenth_packet!=0x0e408507u || tenth_packet_mid!=0x0005u)return 2;
        fprintf(stderr,"BATCH TENTH GRAPH PREDICTED RESOURCE PASS: seven appended allocations, packet=0x%08x, program=C, output=0x%llx; byte-exact CPU readback, no Submit.\n",
                tenth_packet,(unsigned long long)tenth_output);
        for(unsigned i=0;i<2;i++){
          uint64_t args[2]={0x4000,(uint64_t)i};size_t count=sizeof tenth_more[i];
          kern_return_t kr=IOConnectCallMethod(CONN,14,args,2,0,0,0,0,&tenth_more[i],&count);
          fprintf(stderr,"BATCH TEN sel-14#%u kr=0x%x bytes=%zu size=0x%x id=%u\n",
                  i,kr,count,tenth_more[i].size,tenth_more[i].id);
          if(kr || count!=sizeof tenth_more[i] || !tenth_more[i].cpu ||
             tenth_more[i].size!=0x4000 || tenth_more[i].id!=i+19)return 2;
        }
        uint64_t nineteenth=NextTraceID(dev),twentieth=NextTraceID(dev);
        uint64_t eighteenth_staged=0;
        memcpy(&eighteenth_staged,(void*)(uintptr_t)(ninth_more[0].cpu+0x28),8);
        if(nineteenth!=eighteenth_staged+1 || twentieth!=nineteenth+1)return 2;
        FILE* pages=fopen(page_path,"rb");
        if(!pages)return 2;
        if(fread(kernel_page,1,0x4000,pages)!=0x4000 ||
           fread(segment_page,1,0x4000,pages)!=0x4000 ||
           fgetc(pages)!=EOF || ferror(pages))return 2;
        fclose(pages);
        if(kernel_page[0x176]!=0x4a || kernel_page[0x1ce]!=0x4d ||
           kernel_page[0x1d6]!=0x4a || segment_page[0x48]!=0x56 ||
           segment_page[0x8c]!=0x5c ||
           *(uint32_t*)(segment_page+0x24)!=0xf0u)return 2;
        for(unsigned i=0;i<4;i++){
          uint64_t p=0;memcpy(&p,kernel_page+0x1e4+8*i,8);
          if(p!=0x100004d1b80ull+8*i)return 2;
        }
        uint32_t twentieth_low=(uint32_t)twentieth;
        memcpy(kernel_page+0x234,&twentieth_low,4);
        memcpy(segment_page+0x00,&nineteenth,8);
        memcpy(segment_page+0x18,&nineteenth,8);
        memcpy(segment_page+0x28,&twentieth,8);
        memcpy((void*)(uintptr_t)tenth_more[1].cpu,kernel_page,0x4000);
        memcpy((void*)(uintptr_t)tenth_more[0].cpu,segment_page,0x4000);
        if(memcmp((void*)(uintptr_t)tenth_more[1].cpu,kernel_page,0x4000) ||
           memcmp((void*)(uintptr_t)tenth_more[0].cpu,segment_page,0x4000) ||
           *(uint32_t*)(uintptr_t)(tenth_more[0].cpu+0x24)!=0xf0u ||
           !no_agx_metal())return 2;
        for(unsigned i=0;i<256;i++){
          if(tenth_target[i]!=0u)return 2;
          tenth_target[i]=0xDEADBEEFu;
        }
        unsigned sentinels=0;
        for(unsigned i=0;i<256;i++)sentinels+=(tenth_target[i]==0xDEADBEEFu);
        if(sentinels!=256)return 2;
        fprintf(stderr,"BATCH TEN PREDICTED STAGE PASS: page IDs 20/19, trace IDs nineteenth=0x%llx twentieth=0x%llx, ready=0xf0, separate target sentinels=256/256; no Submit.\n",
                (unsigned long long)nineteenth,(unsigned long long)twentieth);
      }
      fprintf(stderr,"BATCH TWO STAGE PASS: distinct page IDs 2/1 and 4/3, fresh trace IDs, both segment ready words clear; no Submit.\n");
      if(distinct_graph){
        unsigned sentinels=0;
        for(unsigned i=0;i<256;i++)sentinels+=(output[i]==0xDEADBEEFu);
        if(sentinels!=256 || !no_agx_metal())return 2;
        fprintf(stderr,"BATCH GRAPH STAGE PASS: first output=0x10000030800, second=0x10000030900%s%s, 256/256 sentinels, distinct page resources; no Submit.\n",
                graph_three?", third=0x10000030a00":"",
                graph_four?", fourth=0x10000030b00":"");
      }
      if(self_accum){
        uint8_t* descriptors=mapped[28]+0x1ba0;
        uint64_t input_address=0,output_address=0;
        memcpy(&input_address,descriptors,8);
        memcpy(&output_address,descriptors+16,8);
        if(input_address!=0x10000030000ull || output_address!=0x10000030800ull ||
           memcmp(mapped[1]+0x500,G17_INPUT_ADD7_CODE,sizeof G17_INPUT_ADD7_CODE))return 2;
        memcpy(descriptors,&output_address,8);
        for(unsigned i=0;i<64;i++)output[i]=self_pattern?
          ((i&1u)?(0xfffffff0u-2u*i):(0x9e3779b9u^(i*0x045d9f3bu))):0u;
        memcpy(&input_address,descriptors,8);
        unsigned initial_exact=0,canaries=0;
        for(unsigned i=0;i<64;i++)initial_exact+=(output[i]==(self_pattern?
          ((i&1u)?(0xfffffff0u-2u*i):(0x9e3779b9u^(i*0x045d9f3bu))):0u));
        for(unsigned i=64;i<256;i++)canaries+=(output[i]==0xDEADBEEFu);
        if(input_address!=output_address || initial_exact!=64 || canaries!=192 || !no_agx_metal())return 2;
        if(self_pattern){
          fprintf(stderr,"BATCH SELF PATTERN STAGE: authored input-add7 at allocation 1, input and output=0x10000030800, initial=64/64 canaries=192/192; no Submit.\n");
          fprintf(stderr,"BATCH SELF seed words:");
          for(unsigned i=0;i<64;i++)fprintf(stderr," %u",output[i]);
          fputc('\n',stderr);
        } else {
          fprintf(stderr,"BATCH SELF STAGE: authored input-add7 at allocation 1, input and output=0x10000030800, zeroes=64/64 canaries=192/192; no Submit.\n");
        }
      }
      if(getenv("ORDERED_VALID_BATCH_TWO")){
        typedef int (*submit_t)(void*,void*,unsigned,void*,unsigned,void*);
        submit_t Submit=dlsym(io,"IOGPUCommandQueueSubmitCommandBuffers");
        if(!Submit)return 2;
        static volatile uint64_t marks[20]={0};
        volatile uint64_t* mark_ptr=marks;
        void (^s0)(void)=Block_copy(^{mark_ptr[0]=0x17e;});
        void (^c0)(void)=Block_copy(^{mark_ptr[1]=0x17f;});
        void (^s1)(void)=Block_copy(^{mark_ptr[2]=0x27e;});
        void (^c1)(void)=Block_copy(^{mark_ptr[3]=0x27f;});
        void (^s2)(void)=Block_copy(^{mark_ptr[4]=0x37e;});
        void (^c2)(void)=Block_copy(^{mark_ptr[5]=0x37f;});
        void (^s3)(void)=Block_copy(^{mark_ptr[6]=0x47e;});
        void (^c3)(void)=Block_copy(^{mark_ptr[7]=0x47f;});
        void (^s4)(void)=Block_copy(^{mark_ptr[8]=0x57e;});
        void (^c4)(void)=Block_copy(^{mark_ptr[9]=0x57f;});
        void (^s5)(void)=Block_copy(^{mark_ptr[10]=0x67e;});
        void (^c5)(void)=Block_copy(^{mark_ptr[11]=0x67f;});
        void (^s6)(void)=Block_copy(^{mark_ptr[12]=0x77e;});
        void (^c6)(void)=Block_copy(^{mark_ptr[13]=0x77f;});
        void (^s7)(void)=Block_copy(^{mark_ptr[14]=0x87e;});
        void (^c7)(void)=Block_copy(^{mark_ptr[15]=0x87f;});
        void (^s8)(void)=Block_copy(^{mark_ptr[16]=0x97e;});
        void (^c8)(void)=Block_copy(^{mark_ptr[17]=0x97f;});
        void (^s9)(void)=Block_copy(^{mark_ptr[18]=0xa7e;});
        void (^c9)(void)=Block_copy(^{mark_ptr[19]=0xa7f;});
        if(!s0 || !c0 || !s1 || !c1 || s0==c0 || s0==s1 || s0==c1 ||
           c0==s1 || c0==c1 || s1==c1 || !s2 || !c2 || s2==c2 ||
           s2==s0 || s2==c0 || s2==s1 || s2==c1 ||
           c2==s0 || c2==c0 || c2==s1 || c2==c1 || !s3 || !c3 || !s4 || !c4 ||
           !s5 || !c5 || !s6 || !c6 || !s7 || !c7 || !s8 || !c8 || !s9 || !c9)return 2;
        uint8_t records[640]={0},out[64]={0};
        uint32_t kernel_ids[10]={shmem[1].id,more[1].id,third_more[1].id,fourth_more[1].id,fifth_more[1].id,sixth_more[1].id,seventh_more[1].id,eighth_more[1].id,ninth_more[1].id,tenth_more[1].id};
        uint32_t segment_ids[10]={shmem[0].id,more[0].id,third_more[0].id,fourth_more[0].id,fifth_more[0].id,sixth_more[0].id,seventh_more[0].id,eighth_more[0].id,ninth_more[0].id,tenth_more[0].id};
        uintptr_t pointers[20]={(uintptr_t)s0,(uintptr_t)c0,(uintptr_t)s1,(uintptr_t)c1,
                                (uintptr_t)s2,(uintptr_t)c2,(uintptr_t)s3,(uintptr_t)c3,
                                (uintptr_t)s4,(uintptr_t)c4,(uintptr_t)s5,(uintptr_t)c5,
                                (uintptr_t)s6,(uintptr_t)c6,(uintptr_t)s7,(uintptr_t)c7,(uintptr_t)s8,(uintptr_t)c8,(uintptr_t)s9,(uintptr_t)c9};
        for(unsigned i=0;i<20;i++)for(unsigned j=0;j<i;j++)
          if(pointers[i]==pointers[j])return 2;
        for(unsigned i=0;i<(graph_ten?10u:graph_nine?9u:graph_eight?8u:graph_seven?7u:graph_six?6u:graph_five?5u:graph_four?4u:(self_three || graph_three)?3u:2u);i++){
          uint8_t* record=records+i*64;
          memcpy(record,&kernel_ids[i],4);memcpy(record+4,&segment_ids[i],4);
          memcpy(record+0x10,&pointers[2*i],sizeof(uintptr_t));
          memcpy(record+0x18,&pointers[2*i+1],sizeof(uintptr_t));
        }
        *(uint32_t*)(uintptr_t)(shmem[0].cpu+0x24)=0x800000f0u;
        *(uint32_t*)(uintptr_t)(more[0].cpu+0x24)=0x800000f0u;
        if(self_three || graph_three)*(uint32_t*)(uintptr_t)(third_more[0].cpu+0x24)=0x800000f0u;
        if(graph_four)*(uint32_t*)(uintptr_t)(fourth_more[0].cpu+0x24)=0x800000f0u;
        if(graph_five)*(uint32_t*)(uintptr_t)(fifth_more[0].cpu+0x24)=0x800000f0u;
        if(graph_six)*(uint32_t*)(uintptr_t)(sixth_more[0].cpu+0x24)=0x800000f0u;
        if(graph_seven)*(uint32_t*)(uintptr_t)(seventh_more[0].cpu+0x24)=0x800000f0u;
        if(graph_eight)*(uint32_t*)(uintptr_t)(eighth_more[0].cpu+0x24)=0x800000f0u;
        if(graph_nine)*(uint32_t*)(uintptr_t)(ninth_more[0].cpu+0x24)=0x800000f0u;
        if(graph_ten)*(uint32_t*)(uintptr_t)(tenth_more[0].cpu+0x24)=0x800000f0u;
        unsigned submit_count=self_single?1:graph_ten?10:graph_nine?9:graph_eight?8:graph_seven?7:graph_six?6:graph_five?5:graph_four?4:(self_three || graph_three)?3:2;
        fprintf(stderr,"BATCH TWO entering: count=%u stride=64 record IDs 2/1 and 4/3%s%s%s%s%s%s%s%s, distinct page pairs and callbacks.\n",
                submit_count,(self_three || graph_three)?" and 6/5":"",
                graph_four?" and 8/7":"",graph_five?" and 10/9":"",
                graph_six?" and 12/11":"",graph_seven?" and 14/13":"",
                graph_eight?" and 16/15":"",graph_nine?" and 18/17":"",graph_ten?" and 20/19":"");
        int result=Submit(queue,NULL,submit_count,records,64,out);
        for(unsigned i=0;i<300 && (!marks[1] || (!self_single && !marks[3]) ||
               ((self_three || graph_three) && !marks[5]) ||
               (graph_four && !marks[7]) ||
               (graph_five && (!marks[9] || fifth_target[63]==0xDEADBEEFu)) ||
               (graph_six && (!marks[11] || sixth_target[63]==0xDEADBEEFu)) ||
               (graph_seven && (!marks[13] || seventh_target[63]==0xDEADBEEFu)) ||
               (graph_eight && (!marks[15] || eighth_target[63]==0xDEADBEEFu)) ||
               (graph_nine && (!marks[17] || ninth_target[63]==0xDEADBEEFu)) ||
               (graph_ten && (!marks[19] || tenth_target[63]==0xDEADBEEFu)) ||
               (self_accum?output[63]==0:output[63]==0xDEADBEEFu) ||
               (distinct_graph && output[127]==0xDEADBEEFu) ||
               (graph_three && output[191]==0xDEADBEEFu) ||
               (graph_four && output[255]==0xDEADBEEFu));i++)usleep(10000);
        unsigned exact=0,canaries=0;
        for(unsigned i=0;i<64;i++){
          uint32_t seed=self_pattern?((i&1u)?(0xfffffff0u-2u*i):
                        (0x9e3779b9u^(i*0x045d9f3bu))):0u;
          exact+=(output[i]==(self_accum?(seed+7u*submit_count):3*i+1));
        }
        for(unsigned i=graph_four?256u:graph_three?192u:distinct_graph?128u:64u;i<256;i++)
          canaries+=(output[i]==0xDEADBEEFu);
        unsigned graph_exact=0;
        if(distinct_graph){
          for(unsigned i=0;i<64;i++)graph_exact+=(output[64+i]==3*i+
              (appended_code32_trial?3u:graph_program_b?2u:1u));
          fprintf(stderr,"BATCH GRAPH returned first=%u/64 second=%u/64 canaries=%u/%u.\n",
                  exact,graph_exact,canaries,graph_four?0u:graph_three?64u:128u);
          fprintf(stderr,"BATCH GRAPH second words:");
          for(unsigned i=64;i<128;i++)fprintf(stderr," %u",output[i]);
          fputc('\n',stderr);
        }
        unsigned third_exact=0;
        if(graph_three){
          for(unsigned i=0;i<64;i++)third_exact+=(output[128+i]==3*i+(third_program_c?3u:1u));
          fprintf(stderr,"BATCH THIRD GRAPH returned third=%u/64 canaries=%u/%u.\n",
                  third_exact,canaries,graph_four?0u:64u);
          fprintf(stderr,"BATCH THIRD GRAPH words:");
          for(unsigned i=128;i<192;i++)fprintf(stderr," %u",output[i]);
          fputc('\n',stderr);
        }
        unsigned fourth_exact=0,input_exact=0;
        if(graph_four){
          for(unsigned i=0;i<64;i++){
            fourth_exact+=(output[192+i]==3*i+1);
            uint32_t a=0,b=0;
            memcpy(&a,mapped[2]+4*i,4);memcpy(&b,mapped[2]+0x400+4*i,4);
            input_exact+=(a==11*i+5 && b==7*i+3);
          }
          fprintf(stderr,"BATCH FOURTH GRAPH returned fourth=%u/64 A/B inputs=%u/64.\n",
                  fourth_exact,input_exact);
          fprintf(stderr,"BATCH FOURTH GRAPH words:");
          for(unsigned i=192;i<256;i++)fprintf(stderr," %u",output[i]);
          fputc('\n',stderr);
        }
        unsigned fifth_exact=0,fifth_canaries=0;
        if(graph_five){
          for(unsigned i=0;i<64;i++)fifth_exact+=(fifth_target[i]==3*i+1);
          for(unsigned i=64;i<256;i++)fifth_canaries+=(fifth_target[i]==0xDEADBEEFu);
          fprintf(stderr,"BATCH FIFTH GRAPH returned fifth=%u/64 canaries=%u/192.\n",
                  fifth_exact,fifth_canaries);
          fprintf(stderr,"BATCH FIFTH GRAPH words:");
          for(unsigned i=0;i<64;i++)fprintf(stderr," %u",fifth_target[i]);
          fputc('\n',stderr);
        }
        unsigned sixth_exact=0,sixth_canaries=0;
        if(graph_six){
          for(unsigned i=0;i<64;i++)sixth_exact+=(sixth_target[i]==3*i+1);
          for(unsigned i=64;i<256;i++)sixth_canaries+=(sixth_target[i]==0xDEADBEEFu);
          fprintf(stderr,"BATCH SIXTH GRAPH returned sixth=%u/64 canaries=%u/192.\n",
                  sixth_exact,sixth_canaries);
          fprintf(stderr,"BATCH SIXTH GRAPH words:");
          for(unsigned i=0;i<64;i++)fprintf(stderr," %u",sixth_target[i]);
          fputc('\n',stderr);
        }
        unsigned seventh_exact=0,seventh_canaries=0;
        if(graph_seven){
          for(unsigned i=0;i<64;i++)seventh_exact+=(seventh_target[i]==3*i+1);
          for(unsigned i=64;i<256;i++)seventh_canaries+=(seventh_target[i]==0xDEADBEEFu);
          fprintf(stderr,"BATCH SEVENTH GRAPH returned seventh=%u/64 canaries=%u/192.\n",
                  seventh_exact,seventh_canaries);
          fprintf(stderr,"BATCH SEVENTH GRAPH words:");
          for(unsigned i=0;i<64;i++)fprintf(stderr," %u",seventh_target[i]);
          fputc('\n',stderr);
        }
        unsigned eighth_exact=0,eighth_canaries=0;
        if(graph_eight){
          for(unsigned i=0;i<64;i++)eighth_exact+=(eighth_target[i]==3*i+1);
          for(unsigned i=64;i<256;i++)eighth_canaries+=(eighth_target[i]==0xDEADBEEFu);
          fprintf(stderr,"BATCH EIGHTH GRAPH returned eighth=%u/64 canaries=%u/192.\n",
                  eighth_exact,eighth_canaries);
          fprintf(stderr,"BATCH EIGHTH GRAPH words:");
          for(unsigned i=0;i<64;i++)fprintf(stderr," %u",eighth_target[i]);
          fputc('\n',stderr);
        }
        unsigned ninth_exact=0,ninth_canaries=0;
        if(graph_nine){
          for(unsigned i=0;i<64;i++)ninth_exact+=(ninth_target[i]==3*i+1);
          for(unsigned i=64;i<256;i++)ninth_canaries+=(ninth_target[i]==0xDEADBEEFu);
          fprintf(stderr,"BATCH NINTH GRAPH returned ninth=%u/64 canaries=%u/192.\n",
                  ninth_exact,ninth_canaries);
          fprintf(stderr,"BATCH NINTH GRAPH words:");
          for(unsigned i=0;i<64;i++)fprintf(stderr," %u",ninth_target[i]);
          fputc('\n',stderr);
        }
        unsigned tenth_exact=0,tenth_canaries=0;
        if(graph_ten){
          for(unsigned i=0;i<64;i++)tenth_exact+=(tenth_target[i]==3*i+3);
          for(unsigned i=64;i<256;i++)tenth_canaries+=(tenth_target[i]==0xDEADBEEFu);
          fprintf(stderr,"BATCH TENTH GRAPH returned tenth=%u/64 canaries=%u/192.\n",
                  tenth_exact,tenth_canaries);
          fprintf(stderr,"BATCH TENTH GRAPH words:");
          for(unsigned i=0;i<64;i++)fprintf(stderr," %u",tenth_target[i]);
          fputc('\n',stderr);
        }
        unsigned split_exact=0,split_canaries=0,split_sentinels=0;
        if(split_output){
          for(unsigned i=0;i<64;i++){
            split_exact+=(split_target[i]==3*i+1);
            split_sentinels+=(split_target[i]==0xDEADBEEFu);
          }
          for(unsigned i=64;i<256;i++)split_canaries+=(split_target[i]==0xDEADBEEFu);
          fprintf(stderr,"BATCH SPLIT returned exact=%u/64 sentinels=%u/64 canaries=%u/192.\n",
                  split_exact,split_sentinels,split_canaries);
        }
        uint32_t outword=0;memcpy(&outword,out,4);
        fprintf(stderr,"BATCH TWO returned status=%d outword=0x%08x exact=%u/64 canaries=%u/%u marks=%llu/%llu/%llu/%llu.\n",
                result,outword,exact,canaries,graph_four?0u:graph_three?64u:distinct_graph?128u:192u,(unsigned long long)marks[0],
                (unsigned long long)marks[1],(unsigned long long)marks[2],
                (unsigned long long)marks[3]);
        fprintf(stderr,"BATCH TWO words:");
        for(unsigned i=0;i<64;i++)fprintf(stderr," %u",output[i]);
        fputc('\n',stderr);
        if(self_three || graph_three)fprintf(stderr,"BATCH THREE callbacks=%llu/%llu.\n",
                              (unsigned long long)marks[4],(unsigned long long)marks[5]);
        if(graph_four)fprintf(stderr,"BATCH FOUR callbacks=%llu/%llu.\n",
                              (unsigned long long)marks[6],(unsigned long long)marks[7]);
        if(graph_five)fprintf(stderr,"BATCH FIVE callbacks=%llu/%llu.\n",
                              (unsigned long long)marks[8],(unsigned long long)marks[9]);
        if(graph_six)fprintf(stderr,"BATCH SIX callbacks=%llu/%llu.\n",
                              (unsigned long long)marks[10],(unsigned long long)marks[11]);
        if(graph_seven)fprintf(stderr,"BATCH SEVEN callbacks=%llu/%llu.\n",
                              (unsigned long long)marks[12],(unsigned long long)marks[13]);
        if(graph_eight)fprintf(stderr,"BATCH EIGHT callbacks=%llu/%llu.\n",
                              (unsigned long long)marks[14],(unsigned long long)marks[15]);
        if(graph_nine)fprintf(stderr,"BATCH NINE callbacks=%llu/%llu.\n",
                              (unsigned long long)marks[16],(unsigned long long)marks[17]);
        if(graph_ten)fprintf(stderr,"BATCH TEN callbacks=%llu/%llu.\n",
                              (unsigned long long)marks[18],(unsigned long long)marks[19]);
        if(self_pattern)fprintf(stderr,"BATCH SELF PATTERN returned count=%u exact=%u/64.\n",
                                submit_count,exact);
        else if(self_accum)fprintf(stderr,"BATCH SELF returned count=%u expected=%u exact=%u/64.\n",
                                    submit_count,7u*submit_count,exact);
        if(result || outword || exact!=64 ||
           canaries!=(graph_four?0u:graph_three?64u:distinct_graph?128u:192u) ||
           (distinct_graph && graph_exact!=64) ||
           (graph_three && third_exact!=64) ||
           (graph_four && (fourth_exact!=64 || input_exact!=64)) ||
           (graph_five && (fifth_exact!=64 || fifth_canaries!=192)) ||
           (graph_six && (sixth_exact!=64 || sixth_canaries!=192)) ||
           (graph_seven && (seventh_exact!=64 || seventh_canaries!=192)) ||
           (graph_eight && (eighth_exact!=64 || eighth_canaries!=192)) ||
           (graph_nine && (ninth_exact!=64 || ninth_canaries!=192)) ||
           (graph_ten && (tenth_exact!=64 || tenth_canaries!=192)) ||
           (split_output && (split_exact!=64 || split_canaries!=192)) ||
           marks[0]!=0x17e || marks[1]!=0x17f ||
           (!self_single && (marks[2]!=0x27e || marks[3]!=0x27f)) ||
           (self_single && (marks[2] || marks[3])) ||
           ((self_three || graph_three) && (marks[4]!=0x37e || marks[5]!=0x37f)) ||
           (!(self_three || graph_three) && (marks[4] || marks[5])) ||
           (graph_four && (marks[6]!=0x47e || marks[7]!=0x47f)) ||
           (!graph_four && (marks[6] || marks[7])) ||
           (graph_five && (marks[8]!=0x57e || marks[9]!=0x57f)) ||
           (!graph_five && (marks[8] || marks[9])) ||
           (graph_six && (marks[10]!=0x67e || marks[11]!=0x67f)) ||
           (!graph_six && (marks[10] || marks[11])) ||
           (graph_seven && (marks[12]!=0x77e || marks[13]!=0x77f)) ||
           (!graph_seven && (marks[12] || marks[13])) ||
           (graph_eight && (marks[14]!=0x87e || marks[15]!=0x87f)) ||
           (!graph_eight && (marks[14] || marks[15])) ||
           (graph_nine && (marks[16]!=0x97e || marks[17]!=0x97f)) ||
           (!graph_nine && (marks[16] || marks[17])) ||
           (graph_ten && (marks[18]!=0xa7e || marks[19]!=0xa7f)) ||
           (!graph_ten && (marks[18] || marks[19])) || !no_agx_metal())return 2;
        fprintf(stderr,"BATCH TWO PASS: %u record%s completed in one below-Metal Submit.\n",
                submit_count,submit_count==1?"":"s");
      }
#else
      fprintf(stderr,"BATCH TWO REFUSED: build lacks block support\n");return 2;
#endif
    }
    if(getenv("ORDERED_INVALID_ONE")){
#ifdef G17_BLOCK_PREFLIGHT
      typedef int (*submit_t)(void*,void*,unsigned,void*,unsigned,void*);
      submit_t Submit=dlsym(io,"IOGPUCommandQueueSubmitCommandBuffers");
      if(!Submit || getenv("ORDERED_ZERO_SUBMIT") || getenv("ORDERED_GUARDED_ONE"))return 2;
      static volatile uint64_t marks[2]={0,0};
      volatile uint64_t* mark_ptr=marks;
      void (^scheduled)(void)=Block_copy(^{mark_ptr[0]=0x17e;});
      void (^completed)(void)=Block_copy(^{mark_ptr[1]=0x17f;});
      if(!scheduled || !completed || scheduled==completed)return 2;
      uint8_t record[64]={0},out[64]={0};
      uint32_t invalid=UINT32_MAX;
      uintptr_t scheduled_ptr=(uintptr_t)scheduled,completed_ptr=(uintptr_t)completed;
      memcpy(record,&invalid,4);memcpy(record+4,&invalid,4);
      memcpy(record+0x10,&scheduled_ptr,sizeof scheduled_ptr);
      memcpy(record+0x18,&completed_ptr,sizeof completed_ptr);
      volatile uint32_t* ready_word=(volatile uint32_t*)(uintptr_t)(shmem[0].cpu+0x24);
      if(*ready_word!=0xf0u || !no_agx_metal())return 2;
      fprintf(stderr,"INVALID ONE entering: count=1 record IDs=ffffffff/ffffffff stride=64 ready=0xf0; one kernel-visible call.\n");
      int result=Submit(queue,NULL,1,record,64,out);
      if(result==0){
        for(unsigned i=0;i<200 && !marks[0] && !marks[1];i++)usleep(10000);
      }
      unsigned intact=0;for(unsigned i=0;i<64;i++)intact+=(output[i]==0xDEADBEEFu);
      uint32_t outword=0;memcpy(&outword,out,4);
      fprintf(stderr,"INVALID ONE returned status=%d outword=0x%08x sentinels=%u/64 marks=%llu/%llu ready=0x%08x.\n",
              result,outword,intact,(unsigned long long)marks[0],(unsigned long long)marks[1],*ready_word);
      if(intact!=64 || marks[0] || marks[1] || *ready_word!=0xf0u || !no_agx_metal())return 2;
      fprintf(stderr,"INVALID ONE OBSERVED: kernel trap returned; no output or callback after bounded wait.\n");
#else
      fprintf(stderr,"INVALID ONE REFUSED: build lacks block support\n");return 2;
#endif
    }
    if(getenv("ORDERED_VALID_ONE") || getenv("ORDERED_VALID_TWO")){
#ifdef G17_BLOCK_PREFLIGHT
      typedef int (*submit_t)(void*,void*,unsigned,void*,unsigned,void*);
      submit_t Submit=dlsym(io,"IOGPUCommandQueueSubmitCommandBuffers");
      if(!Submit || getenv("ORDERED_ZERO_SUBMIT") || getenv("ORDERED_GUARDED_ONE") ||
         getenv("ORDERED_INVALID_ONE"))return 2;
      static volatile uint64_t marks[4]={0,0,0,0};
      volatile uint64_t* mark_ptr=marks;
      void (^scheduled)(void)=Block_copy(^{mark_ptr[0]=0x17e;});
      void (^completed)(void)=Block_copy(^{mark_ptr[1]=0x17f;});
      if(!scheduled || !completed || scheduled==completed)return 2;
      uint8_t record[64]={0},out[64]={0};
      uint32_t kernel_id=shmem[1].id,segment_id=shmem[0].id;
      uintptr_t scheduled_ptr=(uintptr_t)scheduled,completed_ptr=(uintptr_t)completed;
      memcpy(record,&kernel_id,4);memcpy(record+4,&segment_id,4);
      memcpy(record+0x10,&scheduled_ptr,sizeof scheduled_ptr);
      memcpy(record+0x18,&completed_ptr,sizeof completed_ptr);
      volatile uint32_t* ready_word=(volatile uint32_t*)(uintptr_t)(shmem[0].cpu+0x24);
      if(*ready_word!=0xf0u || !no_agx_metal())return 2;
      const char* two_placement=getenv("VALID_PLACEMENT");
      const char* two_op=getenv("VALID_OP");
      int two_switch_b=two_placement && strcmp(two_placement,"second-allocation-control")==0;
      int pipeline_two=getenv("ORDERED_PIPELINE_TWO")!=NULL;
      int program_switch_two=getenv("ORDERED_PROGRAM_SWITCH_TWO")!=NULL;
      if(program_switch_two && (!pipeline_two || !two_placement ||
                                strcmp(two_placement,"input-muladd-then-add7")))return 2;
      if(pipeline_two && (!getenv("ORDERED_VALID_TWO") || getenv("ORDERED_VALID_ONE") ||
                          !two_placement ||
                          strcmp(two_placement,program_switch_two?
                                 "input-muladd-then-add7":"input-muladd-alloc1") ||
                          !two_op || strcmp(two_op,"input-muladd") ||
                          !getenv("VALID_SCALE") || atoi(getenv("VALID_SCALE"))!=0 ||
                          !getenv("VALID_BIAS") || atoi(getenv("VALID_BIAS"))!=7))return 2;
      if(getenv("ORDERED_VALID_TWO") && !pipeline_two &&
         (getenv("ORDERED_VALID_ONE") || !two_placement ||
          (strcmp(two_placement,"captured") && !two_switch_b) ||
          !two_op || strcmp(two_op,"add") || !getenv("VALID_SCALE") ||
          atoi(getenv("VALID_SCALE"))!=3 || !getenv("VALID_BIAS") ||
          atoi(getenv("VALID_BIAS"))!=1)){
        fprintf(stderr,"VALID TWO REFUSED: only captured A or second-allocation A/B control\n");return 2;
      }
      const char* pages_before=getenv("VALID_PAGES_BEFORE");
      const char* pages_after=getenv("VALID_PAGES_AFTER");
      if(!!pages_before!=!!pages_after)return 2;
      const char* input_pattern=getenv("VALID_INPUT_PATTERN");
      if(input_pattern){
        const char* input_placement=getenv("VALID_PLACEMENT");
        if(strcmp(input_pattern,"scrambled") || !input_placement ||
           (strcmp(input_placement,"input-sum-alloc1") &&
            strcmp(input_placement,"input-add7-alloc1") &&
            strcmp(input_placement,"input-muladd-alloc1")) ||
           getenv("ORDERED_VALID_TWO"))return 2;
        uint32_t* input_a=(uint32_t*)mapped[2];
        uint32_t* input_b=(uint32_t*)(mapped[2]+0x400);
        for(uint32_t i=0;i<64;i++){
          input_a[i]=0x9e3779b9u ^ (i*0x045d9f3bu);
          input_b[i]=0x243f6a88u + (i*i*0x1021u);
        }
        for(uint32_t i=0;i<64;i++){
          if(input_a[i]!=(0x9e3779b9u ^ (i*0x045d9f3bu)) ||
             input_b[i]!=(0x243f6a88u + (i*i*0x1021u)))return 2;
        }
        fprintf(stderr,"VALID INPUT pattern=scrambled A0=%u B0=%u A63=%u B63=%u; 128 words read back before Submit.\n",
                input_a[0],input_b[0],input_a[63],input_b[63]);
      }
      const char* binding_swap=getenv("VALID_BINDING_SWAP");
      const char* binding_relocate=getenv("VALID_BINDING_RELOCATE");
      const char* binding_cross=getenv("VALID_BINDING_CROSS");
      const char* binding_alloc24=getenv("VALID_BINDING_ALLOC24");
      const char* output_cross=getenv("VALID_OUTPUT_CROSS");
      if((!!binding_swap+!!binding_relocate+!!binding_cross+!!binding_alloc24)>1)return 2;
      if(binding_swap){
        const char* input_placement=getenv("VALID_PLACEMENT");
        if(strcmp(binding_swap,"1") || !input_placement ||
           strcmp(input_placement,"input-add7-alloc1") || getenv("ORDERED_VALID_TWO"))return 2;
        uint8_t* descriptors=mapped[28]+0x1ba0;
        uint64_t binding_a=0,binding_b=0,binding_c=0;
        memcpy(&binding_a,descriptors,8);
        memcpy(&binding_b,descriptors+8,8);
        memcpy(&binding_c,descriptors+16,8);
        if(binding_a!=0x10000030000ull || binding_b!=0x10000030400ull ||
           binding_c!=0x10000030800ull)return 2;
        memcpy(descriptors,&binding_b,8);
        memcpy(descriptors+8,&binding_a,8);
        uint64_t check_a=0,check_b=0,check_c=0;
        memcpy(&check_a,descriptors,8);
        memcpy(&check_b,descriptors+8,8);
        memcpy(&check_c,descriptors+16,8);
        if(check_a!=binding_b || check_b!=binding_a || check_c!=binding_c)return 2;
        fprintf(stderr,"VALID BINDING swap=1 A=0x%llx B=0x%llx C=0x%llx; descriptors read back before Submit.\n",
                (unsigned long long)check_a,(unsigned long long)check_b,(unsigned long long)check_c);
      }
      if(binding_relocate){
        const char* input_placement=getenv("VALID_PLACEMENT");
        if(strcmp(binding_relocate,"0x8000") || !input_placement ||
           strcmp(input_placement,"input-add7-alloc1") || getenv("ORDERED_VALID_TWO") ||
           input_pattern)return 2;
        uint8_t* descriptors=mapped[28]+0x1ba0;
        uint64_t binding_a=0,binding_b=0,binding_c=0;
        memcpy(&binding_a,descriptors,8);
        memcpy(&binding_b,descriptors+8,8);
        memcpy(&binding_c,descriptors+16,8);
        if(binding_a!=0x10000030000ull || binding_b!=0x10000030400ull ||
           binding_c!=0x10000030800ull)return 2;
        uint32_t* alternate=(uint32_t*)(mapped[2]+0x8000);
        for(uint32_t i=0;i<64;i++)if(alternate[i])return 2;
        for(uint32_t i=0;i<64;i++)alternate[i]=0x6a09e667u ^ (i*0x9e3779b9u);
        for(uint32_t i=0;i<64;i++)
          if(alternate[i]!=(0x6a09e667u ^ (i*0x9e3779b9u)))return 2;
        uint64_t relocated=0x10000038000ull;
        memcpy(descriptors,&relocated,8);
        uint64_t check_a=0,check_b=0,check_c=0;
        memcpy(&check_a,descriptors,8);
        memcpy(&check_b,descriptors+8,8);
        memcpy(&check_c,descriptors+16,8);
        if(check_a!=relocated || check_b!=binding_b || check_c!=binding_c)return 2;
        fprintf(stderr,"VALID BINDING relocate=0x8000 A=0x%llx B=0x%llx C=0x%llx D0=%u D63=%u; 64 words and descriptors read back before Submit.\n",
                (unsigned long long)check_a,(unsigned long long)check_b,
                (unsigned long long)check_c,alternate[0],alternate[63]);
      }
      if(binding_cross){
        const char* input_placement=getenv("VALID_PLACEMENT");
        if(strcmp(binding_cross,"alloc20+0x8000") || !input_placement ||
           strcmp(input_placement,"input-add7-alloc1") || getenv("ORDERED_VALID_TWO") ||
           input_pattern)return 2;
        uint8_t* descriptors=mapped[28]+0x1ba0;
        uint64_t binding_a=0,binding_b=0,binding_c=0;
        memcpy(&binding_a,descriptors,8);
        memcpy(&binding_b,descriptors+8,8);
        memcpy(&binding_c,descriptors+16,8);
        if(binding_a!=0x10000030000ull || binding_b!=0x10000030400ull ||
           binding_c!=0x10000030800ull)return 2;
        uint32_t* alternate=(uint32_t*)(mapped[20]+0x8000);
        for(uint32_t i=0;i<64;i++)if(alternate[i])return 2;
        for(uint32_t i=0;i<64;i++)alternate[i]=0xbb67ae85u ^ (i*0x3c6ef372u);
        for(uint32_t i=0;i<64;i++)
          if(alternate[i]!=(0xbb67ae85u ^ (i*0x3c6ef372u)))return 2;
        uint64_t cross=0x10000060000ull;
        memcpy(descriptors,&cross,8);
        uint64_t check_a=0,check_b=0,check_c=0;
        memcpy(&check_a,descriptors,8);
        memcpy(&check_b,descriptors+8,8);
        memcpy(&check_c,descriptors+16,8);
        if(check_a!=cross || check_b!=binding_b || check_c!=binding_c)return 2;
        fprintf(stderr,"VALID BINDING cross=alloc20+0x8000 A=0x%llx B=0x%llx C=0x%llx D0=%u D63=%u; 64 words and descriptors read back before Submit.\n",
                (unsigned long long)check_a,(unsigned long long)check_b,
                (unsigned long long)check_c,alternate[0],alternate[63]);
      }
      if(binding_alloc24){
        const char* input_placement=getenv("VALID_PLACEMENT");
        if(strcmp(binding_alloc24,"alloc24+0x2000") || !input_placement ||
           strcmp(input_placement,"input-add7-alloc1") || getenv("ORDERED_VALID_TWO") ||
           input_pattern)return 2;
        uint8_t* descriptors=mapped[28]+0x1ba0;
        uint64_t binding_a=0,binding_b=0,binding_c=0;
        memcpy(&binding_a,descriptors,8);
        memcpy(&binding_b,descriptors+8,8);
        memcpy(&binding_c,descriptors+16,8);
        if(binding_a!=0x10000030000ull || binding_b!=0x10000030400ull ||
           binding_c!=0x10000030800ull)return 2;
        uint32_t* alternate=(uint32_t*)(mapped[24]+0x2000);
        for(uint32_t i=0;i<64;i++)if(alternate[i])return 2;
        for(uint32_t i=0;i<64;i++)alternate[i]=0x510e527fu ^ (i*0x1f83d9abu);
        for(uint32_t i=0;i<64;i++)
          if(alternate[i]!=(0x510e527fu ^ (i*0x1f83d9abu)))return 2;
        uint64_t target=0x100000a2000ull;
        memcpy(descriptors,&target,8);
        uint64_t check_a=0,check_b=0,check_c=0;
        memcpy(&check_a,descriptors,8);
        memcpy(&check_b,descriptors+8,8);
        memcpy(&check_c,descriptors+16,8);
        if(check_a!=target || check_b!=binding_b || check_c!=binding_c)return 2;
        fprintf(stderr,"VALID BINDING cross=alloc24+0x2000 A=0x%llx B=0x%llx C=0x%llx D0=%u D63=%u; 64 words and descriptors read back before Submit.\n",
                (unsigned long long)check_a,(unsigned long long)check_b,
                (unsigned long long)check_c,alternate[0],alternate[63]);
      }
      uint32_t* captured_output=output;
      if(output_cross){
        int output_to_alloc24=strcmp(output_cross,"alloc24+0x2000")==0;
        int alloc24_code_output=strcmp(output_cross,"alloc24-code-output")==0;
        const char* input_placement=getenv("VALID_PLACEMENT");
        const char* request_arm=getenv("VALID_ALLOC24_REQUEST_ARM");
        if((!output_to_alloc24 && !alloc24_code_output && strcmp(output_cross,"alloc20+0x9000")) ||
           !input_placement || (alloc24_code_output?
             strcmp(input_placement,"alloc24-candidate") || !request_arm ||
             (strcmp(request_arm,"code") && strcmp(request_arm,"data")) ||
             !getenv("VALID_OP") || strcmp(getenv("VALID_OP"),"add") ||
             !getenv("VALID_SCALE") || strcmp(getenv("VALID_SCALE"),"3") ||
             !getenv("VALID_BIAS") || strcmp(getenv("VALID_BIAS"),"2"):
             strcmp(input_placement,"input-add7-alloc1") || request_arm) ||
           getenv("ORDERED_VALID_TWO") ||
           input_pattern || binding_swap || binding_relocate || binding_cross ||
           binding_alloc24)return 2;
        uint8_t* descriptors=mapped[28]+0x1ba0;
        uint64_t binding_a=0,binding_b=0,binding_c=0;
        memcpy(&binding_a,descriptors,8);
        memcpy(&binding_b,descriptors+8,8);
        memcpy(&binding_c,descriptors+16,8);
        if(binding_a!=0x10000030000ull || binding_b!=0x10000030400ull ||
           binding_c!=(alloc24_code_output?0x100000a2000ull:0x10000030800ull))return 2;
        uint32_t* alternate=(uint32_t*)(output_to_alloc24?
          mapped[24]+0x2000:alloc24_code_output?
          mapped[24]+0x2000:mapped[20]+0x9000);
        if(!alloc24_code_output){
          for(uint32_t i=0;i<256;i++)if(alternate[i])return 2;
          for(uint32_t i=0;i<256;i++)alternate[i]=0xDEADBEEFu;
        }
        for(uint32_t i=0;i<256;i++)if(alternate[i]!=0xDEADBEEFu)return 2;
        uint64_t target=(output_to_alloc24 || alloc24_code_output)?
          0x100000a2000ull:0x10000061000ull;
        if(!alloc24_code_output)memcpy(descriptors+16,&target,8);
        uint64_t check_a=0,check_b=0,check_c=0;
        memcpy(&check_a,descriptors,8);
        memcpy(&check_b,descriptors+8,8);
        memcpy(&check_c,descriptors+16,8);
        if(check_a!=binding_a || check_b!=binding_b || check_c!=target)return 2;
        output=alternate;
        fprintf(stderr,"VALID OUTPUT cross=%s A=0x%llx B=0x%llx C=0x%llx; 256 sentinels and descriptors read back before Submit.\n",
                alloc24_code_output?"alloc24-code-output":
                output_to_alloc24?"alloc24+0x2000":"alloc20+0x9000",
                (unsigned long long)check_a,(unsigned long long)check_b,
                (unsigned long long)check_c);
      }
      *ready_word=0x800000f0u;
      if(pages_before && snapshot_command_pages(pages_before,shmem)){
        fprintf(stderr,"VALID ONE REFUSED: before-page snapshot failed\n");return 2;
      }
      fprintf(stderr,"VALID ONE entering: count=1 record IDs=2/1 stride=64 ready=0x800000f0; one kernel-visible call.\n");
      int result=Submit(queue,NULL,1,record,64,out);
      for(unsigned i=0;i<300 && (!marks[1] || output[63]==0xDEADBEEFu);i++)usleep(10000);
      if(pages_after && snapshot_command_pages(pages_after,shmem)){
        fprintf(stderr,"VALID ONE REFUSED: after-page snapshot failed\n");return 2;
      }
      unsigned exact=0,canaries=0;
      int bias=getenv("VALID_BIAS")?atoi(getenv("VALID_BIAS")):1;
      int scale=getenv("VALID_SCALE")?atoi(getenv("VALID_SCALE")):3;
      const char* op=getenv("VALID_OP")?getenv("VALID_OP"):"add";
      const char* placement=getenv("VALID_PLACEMENT")?getenv("VALID_PLACEMENT"):"captured";
      int subtract=strcmp(op,"sub")==0;
      int do_xor=strcmp(op,"xor")==0;
      int do_and=strcmp(op,"and")==0;
      int do_or=strcmp(op,"or")==0;
      int do_input_sum=strcmp(op,"input-sum")==0;
      int do_input_add7=strcmp(op,"input-add7")==0;
      int do_input_muladd=strcmp(op,"input-muladd")==0;
      if(!((!strcmp(op,"add") &&
            ((scale==3 && (bias==1 || bias==2 || bias==3 || bias==7)) ||
             (scale==5 && bias==7))) ||
           ((subtract || do_xor || do_and || do_or) && scale==3 && bias==7) ||
           (do_input_sum && scale==0 && bias==0) ||
           (do_input_add7 && scale==0 && bias==7) ||
           (do_input_muladd && scale==0 && bias==7)))return 2;
      for(unsigned i=0;i<64;i++){
        uint32_t predicted=(uint32_t)((unsigned)scale*i);
        predicted=do_input_muladd?((uint32_t*)mapped[2])[i]*((uint32_t*)(mapped[2]+0x400))[i]+7u:
                  do_input_sum?((uint32_t*)mapped[2])[i]+((uint32_t*)(mapped[2]+0x400))[i]:
                  do_input_add7?(binding_alloc24?((uint32_t*)(mapped[24]+0x2000))[i]:
                                 binding_cross?((uint32_t*)(mapped[20]+0x8000))[i]:
                                 ((uint32_t*)(mapped[2]+(binding_relocate?0x8000:binding_swap?0x400:0)))[i])+7u:
                  do_and?(predicted&(uint32_t)bias):
                  do_or?(predicted|(uint32_t)bias):
                  do_xor?(predicted^(uint32_t)bias):
                  subtract?predicted-(uint32_t)bias:predicted+(uint32_t)bias;
        exact+=(output[i]==predicted);
      }
      for(unsigned i=64;i<256;i++)canaries+=(output[i]==0xDEADBEEFu);
      unsigned silent=0;
      if(output_cross && strcmp(output_cross,"alloc24-code-output")==0)
        for(unsigned i=0;i<64;i++)silent+=(output[i]==0xDEADBEEFu);
      if(output_cross){
        unsigned original_sentinels=0;
        for(unsigned i=0;i<256;i++)original_sentinels+=(captured_output[i]==0xDEADBEEFu);
        fprintf(stderr,"VALID OUTPUT original_sentinels=%u/256 after Submit.\n",original_sentinels);
        if(original_sentinels!=256)return 2;
      }
      uint32_t outword=0;memcpy(&outword,out,4);
      fprintf(stderr,"VALID ONE returned placement=%s op=%s scale=%d bias=%d status=%d outword=0x%08x exact=%u/64 canaries=%u/192 marks=%llu/%llu ready=0x%08x.\n",
              placement,op,scale,bias,result,outword,exact,canaries,(unsigned long long)marks[0],(unsigned long long)marks[1],*ready_word);
      fprintf(stderr,"VALID ONE words:");
      for(unsigned i=0;i<64;i++)fprintf(stderr," %u",output[i]);
      fputc('\n',stderr);
      int expect_silent=output_cross && strcmp(output_cross,"alloc24-code-output")==0 &&
                        getenv("VALID_ALLOC24_REQUEST_ARM") &&
                        strcmp(getenv("VALID_ALLOC24_REQUEST_ARM"),"code")==0;
      if(result || outword || (expect_silent?silent!=64:exact!=64) || canaries!=192 ||
         marks[0]!=0x17e || marks[1]!=0x17f || !no_agx_metal())return 2;
      if(output_cross && strcmp(output_cross,"alloc24-code-output")==0)
        fprintf(stderr,"VALID ALLOC24 CODE OUTPUT PASS: arm=%s exact=%u/64 silent=%u/64 canaries=%u/192; callbacks complete.\n",
                getenv("VALID_ALLOC24_REQUEST_ARM"),exact,silent,canaries);
      else fprintf(stderr,"VALID ONE PASS: below-Metal authored output and notification completion verified.\n");
      if(getenv("ORDERED_VALID_TWO")){
        uint64_t third=NextTraceID(dev),fourth=NextTraceID(dev);
        fprintf(stderr,"VALID TWO trace IDs third=0x%llx fourth=0x%llx\n",
                (unsigned long long)third,(unsigned long long)fourth);
        if(third<=second || fourth!=third+1)return 2;
        uint32_t fourth_low=(uint32_t)fourth;
        *ready_word=0xf0u;
        memcpy((void*)(uintptr_t)(shmem[1].cpu+0x234),&fourth_low,4);
        memcpy((void*)(uintptr_t)(shmem[0].cpu+0x00),&third,8);
        memcpy((void*)(uintptr_t)(shmem[0].cpu+0x18),&third,8);
        memcpy((void*)(uintptr_t)(shmem[0].cpu+0x28),&fourth,8);
        if(two_switch_b){
          uint16_t code_low=0x8507u,code_mid=0x0001u;
          memcpy(mapped[23]+0x40,&code_low,2);
          memcpy(mapped[23]+0x46,&code_mid,2);
          if(memcmp(mapped[1]+0x500,KERNEL_3I2,sizeof KERNEL_3I2)!=0 ||
             memcmp(mapped[23]+0x40,&code_low,2)!=0 ||
             memcmp(mapped[23]+0x46,&code_mid,2)!=0)return 2;
        }
        if(program_switch_two){
          uint16_t code_low=0x06c7u,code_mid=0x0000u;
          memcpy(mapped[23]+0x40,&code_low,2);
          memcpy(mapped[23]+0x46,&code_mid,2);
          if(memcmp(mapped[0]+0x6c0,G17_INPUT_ADD7_CODE,sizeof G17_INPUT_ADD7_CODE)!=0 ||
             memcmp(mapped[1]+0x500,G17_INPUT_MULADD_CODE,sizeof G17_INPUT_MULADD_CODE)!=0 ||
             memcmp(mapped[23]+0x40,&code_low,2)!=0 ||
             memcmp(mapped[23]+0x46,&code_mid,2)!=0)return 2;
          fprintf(stderr,"VALID TWO program switch packet=0x0e4006c7 code=0x100000006c0; second authored input-add7 selected.\n");
        }
        uint32_t* second_output=output;
        if(pipeline_two){
          uint32_t* target=(uint32_t*)(mapped[20]+0x9000);
          for(unsigned i=0;i<256;i++)if(target[i])return 2;
          for(unsigned i=0;i<256;i++)target[i]=0xDEADBEEFu;
          for(unsigned i=0;i<256;i++)if(target[i]!=0xDEADBEEFu)return 2;
          uint8_t* descriptors=mapped[28]+0x1ba0;
          uint64_t old_a=0,old_b=0,old_c=0;
          memcpy(&old_a,descriptors,8);
          memcpy(&old_b,descriptors+8,8);
          memcpy(&old_c,descriptors+16,8);
          if(old_a!=0x10000030000ull || old_b!=0x10000030400ull ||
             old_c!=0x10000030800ull)return 2;
          uint64_t next_a=old_c,next_c=0x10000061000ull;
          memcpy(descriptors,&next_a,8);
          memcpy(descriptors+16,&next_c,8);
          uint64_t check_a=0,check_b=0,check_c=0;
          memcpy(&check_a,descriptors,8);
          memcpy(&check_b,descriptors+8,8);
          memcpy(&check_c,descriptors+16,8);
          if(check_a!=next_a || check_b!=old_b || check_c!=next_c)return 2;
          second_output=target;
          fprintf(stderr,"VALID TWO pipeline A=0x%llx B=0x%llx D=0x%llx; intermediate retained, 256 target sentinels and descriptors read back.\n",
                  (unsigned long long)check_a,(unsigned long long)check_b,
                  (unsigned long long)check_c);
        } else {
          for(unsigned i=0;i<64;i++)output[i]=0xDEADBEEFu;
        }
        unsigned sentinels2=0;
        for(unsigned i=0;i<64;i++)sentinels2+=(second_output[i]==0xDEADBEEFu);
        if(sentinels2!=64 || *ready_word!=0xf0u)return 2;
        void (^scheduled2)(void)=Block_copy(^{mark_ptr[2]=0x27e;});
        void (^completed2)(void)=Block_copy(^{mark_ptr[3]=0x27f;});
        if(!scheduled2 || !completed2 || scheduled2==completed2 ||
           scheduled2==scheduled || completed2==completed)return 2;
        uint8_t record2[64]={0},out2[64]={0};
        uintptr_t scheduled2_ptr=(uintptr_t)scheduled2,completed2_ptr=(uintptr_t)completed2;
        memcpy(record2,&kernel_id,4);memcpy(record2+4,&segment_id,4);
        memcpy(record2+0x10,&scheduled2_ptr,sizeof scheduled2_ptr);
        memcpy(record2+0x18,&completed2_ptr,sizeof completed2_ptr);
        if(!no_agx_metal())return 2;
        *ready_word=0x800000f0u;
        if(!pipeline_two)fprintf(stderr,"VALID TWO target bias=%d packet=%s\n",two_switch_b?2:1,
                                 two_switch_b?"second-allocation":"captured");
        fprintf(stderr,"VALID TWO entering: count=1 record IDs=2/1 stride=64 ready=0x800000f0; second Submit on same queue/pages.\n");
        int result2=Submit(queue,NULL,1,record2,64,out2);
        for(unsigned i=0;i<300 && (!marks[3] || second_output[63]==0xDEADBEEFu);i++)usleep(10000);
        unsigned exact2=0,canaries2=0;
        for(unsigned i=0;i<64;i++){
          uint32_t first=((uint32_t*)mapped[2])[i]*((uint32_t*)(mapped[2]+0x400))[i]+7u;
          uint32_t predicted=pipeline_two?
            (program_switch_two?first+7u:first*((uint32_t*)(mapped[2]+0x400))[i]+7u):
            3*i+(two_switch_b?2u:1u);
          exact2+=(second_output[i]==predicted);
        }
        for(unsigned i=64;i<256;i++)canaries2+=(second_output[i]==0xDEADBEEFu);
        if(pipeline_two){
          unsigned intermediate_exact=0,intermediate_canaries=0;
          for(unsigned i=0;i<64;i++)intermediate_exact+=(output[i]==
            ((uint32_t*)mapped[2])[i]*((uint32_t*)(mapped[2]+0x400))[i]+7u);
          for(unsigned i=64;i<256;i++)intermediate_canaries+=(output[i]==0xDEADBEEFu);
          fprintf(stderr,"VALID TWO intermediate exact=%u/64 canaries=%u/192 after second Submit.\n",
                  intermediate_exact,intermediate_canaries);
          if(intermediate_exact!=64 || intermediate_canaries!=192)return 2;
        }
        uint32_t outword2=0;memcpy(&outword2,out2,4);
        fprintf(stderr,"VALID TWO returned status=%d outword=0x%08x exact=%u/64 canaries=%u/192 marks=%llu/%llu ready=0x%08x.\n",
                result2,outword2,exact2,canaries2,(unsigned long long)marks[2],
                (unsigned long long)marks[3],*ready_word);
        fprintf(stderr,"VALID TWO words:");
        for(unsigned i=0;i<64;i++)fprintf(stderr," %u",second_output[i]);
        fputc('\n',stderr);
        if(result2 || outword2 || exact2!=64 || canaries2!=192 ||
           marks[2]!=0x27e || marks[3]!=0x27f || !no_agx_metal())return 2;
        fprintf(stderr,"VALID TWO PASS: second below-Metal authored output and callbacks verified on reused queue/pages.\n");
      }
#else
      fprintf(stderr,"VALID ONE REFUSED: build lacks block support\n");return 2;
#endif
    }
  }
  if(!no_agx_metal())return 2;
  fprintf(stderr,"ORDERED CONTROL PASS: %u selector-9 calls in original order, no parent rebase, %s.\n",
          getenv("ORDERED_BATCH_TENTH_GRAPH")?92u:
          getenv("ORDERED_BATCH_NINTH_GRAPH")?85u:
          getenv("ORDERED_BATCH_EIGHTH_GRAPH")?78u:
          getenv("ORDERED_BATCH_SEVENTH_GRAPH")?71u:
          getenv("ORDERED_BATCH_SIXTH_GRAPH")?64u:
          getenv("ORDERED_BATCH_FIFTH_GRAPH")?57u:
          getenv("ORDERED_BATCH_FOURTH_GRAPH")?50u:
          getenv("ORDERED_BATCH_THIRD_GRAPH")?43u:
          getenv("ORDERED_BATCH_DISTINCT_GRAPH")?36u:29u,
          getenv("ORDERED_VALID_TWO")?"two sequential valid count-one kernel submissions":
          getenv("ORDERED_VALID_ONE")?"valid count-one kernel submission":
          getenv("ORDERED_INVALID_ONE")?"invalid-ID count-one kernel boundary only":
          getenv("ORDERED_GUARDED_TWO")?"guarded two-record userspace path only":
          getenv("ORDERED_VALID_BATCH_TWO")?(getenv("ORDERED_BATCH_TENTH_GRAPH")?
            "ten records in one valid Submit":getenv("ORDERED_BATCH_NINTH_GRAPH")?
            "nine records in one valid Submit":getenv("ORDERED_BATCH_EIGHTH_GRAPH")?
            "eight records in one valid Submit":getenv("ORDERED_BATCH_SEVENTH_GRAPH")?
            "seven records in one valid Submit":getenv("ORDERED_BATCH_SIXTH_GRAPH")?
            "six records in one valid Submit":getenv("ORDERED_BATCH_FIFTH_GRAPH")?
            "five records in one valid Submit":getenv("ORDERED_BATCH_FOURTH_GRAPH")?
            "four records in one valid Submit":getenv("ORDERED_BATCH_THIRD_GRAPH")?
            "three records in one valid Submit":"two records in one valid Submit"):
          getenv("ORDERED_BATCH_PREFLIGHT")?(getenv("ORDERED_BATCH_TENTH_GRAPH")?
            "ten-record graph and page staging only":getenv("ORDERED_BATCH_NINTH_GRAPH")?
            "nine-record graph and page staging only":getenv("ORDERED_BATCH_EIGHTH_GRAPH")?
            "eight-record graph and page staging only":getenv("ORDERED_BATCH_SEVENTH_GRAPH")?
            "seven-record graph and page staging only":getenv("ORDERED_BATCH_SIXTH_GRAPH")?
            "six-record graph and page staging only":getenv("ORDERED_BATCH_FIFTH_GRAPH")?
            "five-record graph and page staging only":getenv("ORDERED_BATCH_FOURTH_GRAPH")?
            "four-record graph and page staging only":getenv("ORDERED_BATCH_THIRD_GRAPH")?
            "three-record graph and page staging only":"two-record page-pair staging only"):
          getenv("ORDERED_GUARDED_ONE")?"guarded count-one userspace path only":
          getenv("ORDERED_ZERO_SUBMIT")?"zero-count Submit only":"no Submit");
  return 0;
}
static kern_return_t cm(unsigned sel,const uint64_t*in,uint32_t incnt,const void*ins,size_t inscnt){
  return IOConnectCallMethod(CONN,sel,in,incnt,ins,inscnt,0,0,0,0);
}

int main(void){
  if(!no_agx_metal())return 2;
  void* io=dlopen("/System/Library/PrivateFrameworks/IOGPU.framework/IOGPU",RTLD_NOW);
  if(!io){ fprintf(stderr,"IOGPU dlopen failed\n"); return 1; }
  void*(*DevCreate)(io_service_t)=dlsym(io,"IOGPUDeviceCreate");
  if(!DevCreate){ fprintf(stderr,"IOGPUDeviceCreate unavailable\n"); return 1; }
  io_service_t svc=IOServiceGetMatchingService(kIOMainPortDefault,IOServiceMatching("AGXAcceleratorG17X"));
  if(!svc){ fprintf(stderr,"no AGXAcceleratorG17X\n"); return 1; }
  void* dev=DevCreate(svc);
  if(!dev){ fprintf(stderr,"IOGPUDeviceCreate failed\n"); return 1; }
  // metal_min routes ALL IOConnectCallMethod (sel-9/261 + channels) through the user client opened with
  // connection type 0x100005 (traced). *(dev+0x14) is a different-type client that lacks the channel methods.
  unsigned c5=0; kern_return_t ok=IOServiceOpen(svc, mach_task_self(), 0x100005, &c5);
  if(check(ok,"IOServiceOpen(0x100005)"))return 2;
  CONN = c5;
  fprintf(stderr,"STAGE 1: device=%p  type-0x100005 conn=0x%x (open kr=0x%x)  dev+0x14=0x%x\n",
          dev,c5,ok,*(unsigned*)((char*)dev+0x14));

  // STAGE 1.4: the 4x 4MB GPU-address-space reservations metal_min creates via mach_vm_map BEFORE sel-13.
  // Hypothesis: sel-13/32/28 fail because they depend on these existing. Test with anonymous ANYWHERE maps.
  fprintf(stderr,"STAGE 1.4: mach_vm_map 4MB pipeline reservations\n");
  { for(int i=0;i<4;i++){ mach_vm_address_t a=0; mach_vm_size_t sz=0x400000;
      kern_return_t k=mach_vm_map(mach_task_self(),&a,sz,0,VM_FLAGS_ANYWHERE,MEMORY_OBJECT_NULL,0,0,
                                  VM_PROT_READ|VM_PROT_WRITE,VM_PROT_READ|VM_PROT_WRITE,VM_INHERIT_NONE);
      fprintf(stderr,"  resv[%d] kr=0x%x @0x%llx\n",i,k,(unsigned long long)a);
      if(check(k,"mach_vm_map reservation"))return 2; } }

  // STAGE 1.5: the preamble metal_min runs on the 0x100005 connection (no inputs; capability/init queries that
  // establish the connection state the channels require). Generous output buffers.
  fprintf(stderr,"STAGE 1.5: preamble on the 0x100005 connection\n");
  { // EXACT signatures captured from metal_min. NOTE: sel-13 and sel-32 are already run by IOGPUDeviceCreate on
    // its type-0 connection (capability queries) and are NON-ESSENTIAL here; sel-28 likewise. We run only the
    // preamble calls that the type-0x100005 connection actually needs for the channels (2/0/5/258/256, all kr=0).
    unsigned char ob[600]; size_t obc;
    const unsigned sel[5]={2,0,5,258,256}; const size_t osz[5]={536,64,32,456,152};
    for(int i=0;i<5;i++){ obc=osz[i]; memset(ob,0,sizeof ob);
      kern_return_t k=IOConnectCallStructMethod(CONN,sel[i],0,0,ob,&obc);
      fprintf(stderr,"  sel-%u kr=0x%x\n",sel[i],k);
      if(check(k,"preamble selector"))return 2; } }

  static const unsigned char CTX[72]={0x0b,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0};
  unsigned char co[72]; size_t cs=72;
  kern_return_t kr=IOConnectCallMethod(CONN,261,0,0,CTX,72,0,0,co,&cs);
  fprintf(stderr,"STAGE 2: sel-261 context kr=0x%x\n",kr);
  if(check(kr,"sel-261"))return 2;

  if(getenv("ORDERED_ALLOC_PREFLIGHT"))return ordered_alloc_control(io,dev);
  if(getenv("LAYOUT_ALLOC_PREFLIGHT"))return layout_alloc_control(io,dev);
  if(getenv("CODE_ALLOC_PREFLIGHT"))return code_alloc_control();

  // STAGE 3: the big buffer FIRST (=> aperture base 0x10000000000), then the 0xc000 submission buffer.
  fprintf(stderr,"STAGE 3: allocate buffers\n");
  uint64_t big_ap,big_hv,big_h;
  if(gpu_alloc(BIGSZ,&big_ap,&big_hv,&big_h)) return 2;
  if(big_ap!=WIN_BASE) fprintf(stderr,"  NOTE: big aperture 0x%llx != expected 0x%llx (templated apertures would need patching)\n",big_ap,WIN_BASE);
  uint64_t sub_ap,sub_hv,sub_h;
  if(gpu_alloc(0xc000,&sub_ap,&sub_hv,&sub_h)) return 2;

  // STAGE 4: author into the big buffer at metal_min's offsets
  fprintf(stderr,"STAGE 4: author kernel/CDM/output/args into the big buffer\n");
  unsigned char* B=(unsigned char*)(uintptr_t)big_hv;
  const unsigned char* KERN = getenv("CONTROL")?KERNEL_3I2:KERNEL_3I1;
  memset(B+OFF_CODE,0,0x1000);   memcpy(B+OFF_CODE, KERN, KLEN);             // shader machine code
  memcpy(B+OFF_CDM,  TPL_CDM, sizeof TPL_CDM);                              // CDM/USC stream (references code window)
  { uint32_t* o=(uint32_t*)(B+OFF_OUTPUT); for(int i=0;i<0x1000/4;i++) o[i]=0xDEADBEEFu; } // sentinel
  memset(B+OFF_ARGS,0,0x2000);
  *(uint64_t*)(B+OFF_ARGS+ARG_OUTDESC) = WIN_BASE+OFF_OUTPUT;               // output binding (buffer 0)
  fprintf(stderr,"  code@+0x%x(%dB) CDM@+0x%x output@+0x%x sentinel argdesc@+0x%x=0x%llx\n",
          OFF_CODE,KLEN,OFF_CDM,OFF_OUTPUT,OFF_ARGS+ARG_OUTDESC,WIN_BASE+OFF_OUTPUT);

  // STAGE 5: the submission buffer's 3 pages (ring/header, command stream, shmem) - templated verbatim.
  fprintf(stderr,"STAGE 5: submission buffer pages (templated)\n");
  unsigned char* S=(unsigned char*)(uintptr_t)sub_hv;
  memcpy(S+0x0000, TPL_RING,  4096);  // page0 ring/header (0 apertures)
  memcpy(S+0x4000, TPL_CMD,   4096);  // page1 command stream (apertures point into big buffer @ WIN_BASE)
  memcpy(S+0x8000, TPL_SHMEM, 4096);  // page2 shmem
  // If the big buffer did NOT land at WIN_BASE, the page1 apertures are wrong; refuse to proceed to fire.
  int aperture_ok = (big_ap==WIN_BASE);

  // STAGE 6: queue + channel/ring setup (the exact reversed recipe)
  fprintf(stderr,"STAGE 6: queue + channels\n");
  void*(*QCreate)(void*,void*,unsigned long)=dlsym(io,"IOGPUCommandQueueCreate");
  if(!QCreate){ fprintf(stderr,"QueueCreate unavailable\n"); return 2; }
  unsigned char qdesc[0x410]; memset(qdesc,0,sizeof qdesc);
  char process_path[PROC_PIDPATHINFO_MAXSIZE]={0};
  if(proc_pidpath(getpid(),process_path,sizeof process_path)<=0 || strlen(process_path)>=29){
    fprintf(stderr,"PREFLIGHT REFUSED: process path cannot fit observed queue descriptor\n"); return 2;
  }
  // Queue-only Metal control places the executable path at 0 and 0x3e3 and
  // carries 2, 0xffffffff, 1 in the final three u32 fields.
  memcpy(qdesc,process_path,strlen(process_path)+1);
  memcpy(qdesc+0x3e3,process_path,strlen(process_path)+1);
  *(uint32_t*)(qdesc+0x400)=0x2;
  *(uint32_t*)(qdesc+0x408)=0xffffffffu;
  *(uint32_t*)(qdesc+0x40c)=1;
  // Unify: make the device object use OUR type-0x100005 connection so the queue registers on the same
  // connection the channels use (metal_min funnels sel-9/QueueCreate/channels through one 0x100005 client).
  unsigned save14=*(unsigned*)((char*)dev+0x14); *(unsigned*)((char*)dev+0x14)=CONN;
  void* q=QCreate(dev,qdesc,0x410);
  fprintf(stderr,"  QueueCreate -> %p (dev+0x14 %x->%x)\n",q,save14,CONN);
  if(!q){ fprintf(stderr,"PREFLIGHT REFUSED: queue absent\n"); return 2; }
  // IOGPUCommandQueueCreate made selector 7, 16, and 28 internally, in order.
  // Reissuing them was the old harness's selector-28 error (0xe00002c9).
  unsigned char ob16[16]; size_t obc;
  fprintf(stderr,"  queue internal sel-7/16/28: not duplicated\n");
  obc=16; kr=IOConnectCallStructMethod(CONN,6,0,0,ob16,&obc); fprintf(stderr,"  sel-6 kr=0x%x\n",kr);
  if(check(kr,"sel-6") || obc!=16)return 2;
  struct shmem_result shmem[2]={0};
  for(int i=0;i<2;i++){
    uint64_t a[2]={0x4000,(uint64_t)i}; obc=sizeof shmem[i];
    kr=IOConnectCallMethod(CONN,14,a,2,0,0,0,0,&shmem[i],&obc);
    fprintf(stderr,"  sel-14#%d kr=0x%x size=%zu cpu=0x%llx bytes=%u id=%u\n",
            i,kr,obc,(unsigned long long)shmem[i].cpu,shmem[i].size,shmem[i].id);
    if(check(kr,"sel-14") || obc!=16 || shmem[i].size!=0x4000 || !shmem[i].cpu || !shmem[i].id)return 2;
  }

  fprintf(stderr,"\nSTATE BUILT below Metal (no MTLDevice). big_ap=0x%llx (expected 0x%llx, match=%d) sub_ap=0x%llx\n",
          big_ap,WIN_BASE,aperture_ok,sub_ap);

  if(!aperture_ok){ fprintf(stderr,"PREFLIGHT REFUSED: aperture mismatch\n"); return 2; }
  if(!no_agx_metal())return 2;
  fprintf(stderr,"PREFLIGHT SETUP PASS; SUBMISSION NOT AVAILABLE: selector-14 shmem contents, 64-byte record, and code-window mapping remain unverified.\n");
  return 0; // This binary deliberately contains no Submit symbol lookup or call.
}
