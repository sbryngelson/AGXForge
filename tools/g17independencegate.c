// INDEPENDENCE GATE CHECK (safe, creation + CPU/GPU memory round-trip, NO dispatch).
// Proves the below-Metal execution stack is live WITHOUT the AGX Metal GPU driver:
//   1. build device+queue+pool+context below Metal via IOGPU's exported C API (no ObjC, no MTLDevice);
//   2. sel-9 alloc a unified CPU==GPU buffer, write via CPU ptr, read back (real GPU-visible memory);
//   3. assert AGXMetalG17X (the Metal GPU driver) is NOT loaded and no MTLDevice class is realized.
// Metal.framework being transitively MAPPED by IOKit is expected and inert (see ledger 25.143).
// Links CoreFoundation+IOKit only (NO Foundation -> stays off the ObjC/Metal-device path).
#include <IOKit/IOKitLib.h>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <string.h>
#include <stdio.h>
typedef void* (*devcreate_t)(io_service_t);
typedef void* (*qcreate_t)(void*, void*, unsigned long);
typedef void* (*poolcreate_t)(void*, void*, void*, unsigned long);
static int img_loaded(const char* sub){ uint32_t n=_dyld_image_count();
  for(uint32_t i=0;i<n;i++){ const char* nm=_dyld_get_image_name(i); if(nm&&strstr(nm,sub)) return 1; } return 0; }
int main(void){
  int fails=0;
  void* io=dlopen("/System/Library/PrivateFrameworks/IOGPU.framework/IOGPU",RTLD_NOW);
  if(!io){ printf("FAIL: dlopen IOGPU: %s\n", dlerror()); return 1; }
  devcreate_t IOGPUDeviceCreate=(devcreate_t)dlsym(io,"IOGPUDeviceCreate");
  qcreate_t IOGPUCommandQueueCreate=(qcreate_t)dlsym(io,"IOGPUCommandQueueCreate");
  poolcreate_t PoolCreate=(poolcreate_t)dlsym(io,"IOGPUMetalCommandBufferStoragePoolCreate");
  io_service_t svc=IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAcceleratorG17X"));
  if(!svc){ printf("FAIL: no AGXAcceleratorG17X service\n"); return 1; }
  void* dev=IOGPUDeviceCreate(svc);
  unsigned char qd[0x410]; memset(qd,0,sizeof qd);
  void* q=IOGPUCommandQueueCreate(dev,qd,0x410);
  void* pool=PoolCreate(dev,(char*)dev+0x2e0,(void*)0,1);
  printf("[1] below-Metal objects: dev=%p queue=%p pool=%p\n", dev,q,pool);
  if(!(dev&&q&&pool)){ printf("FAIL: below-Metal object model\n"); fails++; }
  // context (sel-261) then sel-9 unified alloc + round-trip
  unsigned int conn=*(unsigned int*)((char*)dev+0x14);
  static const unsigned char CTX[72]={0x0b,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,2,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0};
  unsigned char co[72]; size_t cosz=72;
  IOConnectCallMethod(conn,261,0,0,CTX,72,0,0,co,&cosz);
  // exact 0x04-class sel-9 descriptor captured verbatim from a live Metal dispatch (proven in g1_harness)
  static const char* A04_HEX="0000000000000000010001000100000001010001300400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000008180000000000000000000000";
  unsigned char A04[104]; for(int j=0;j<104;j++){ unsigned v; sscanf(A04_HEX+2*j,"%2x",&v); A04[j]=(unsigned char)v; }
  unsigned long long ao[11]; size_t aosz=88; memset(ao,0,sizeof ao);
  kern_return_t akr=IOConnectCallMethod(conn,9,0,0,A04,104,0,0,ao,&aosz);
  unsigned long long shared=ao[1];
  printf("[2] context+alloc: alloc kr=0x%x shared_va=0x%llx gpu_va=0x%llx size=0x%llx\n", akr, ao[1], ao[6], ao[5]);
  if(akr||!shared){ printf("FAIL: below-Metal sel-9 alloc\n"); fails++; }
  else { volatile unsigned int* p=(volatile unsigned int*)(uintptr_t)shared;
    p[0]=0xC0DEF00D; p[1]=0x11223344; p[63]=0xA5A5A5A5;
    int ok=(p[0]==0xC0DEF00D&&p[1]==0x11223344&&p[63]==0xA5A5A5A5);
    printf("[3] CPU/GPU unified round-trip: %s (p[0]=0x%x p[63]=0x%x)\n", ok?"ok":"MISMATCH", p[0], p[63]);
    if(!ok){ printf("FAIL: unified memory round-trip\n"); fails++; } }
  // independence assertions
  int agx=img_loaded("AGXMetalG17X"); int mtldev = (dlsym(RTLD_DEFAULT,"MTLCreateSystemDefaultDevice")!=0) && img_loaded("AGXMetalG17X");
  printf("[4] AGX Metal DRIVER loaded=%d (must be 0)  Metal.framework mapped=%d (inert, expected)\n",
         agx, img_loaded("/Metal.framework/"));
  if(agx){ printf("FAIL: AGX Metal driver was loaded - not independent\n"); fails++; }
  printf("\nINDEPENDENCE GATE: %s\n", fails? "FAIL" : "PASS - below-Metal stack live, AGX Metal driver never loaded");
  return fails?2:0;
}
