// A COMPUTE HARNESS THAT BINDS A TEXTURE. Separate from accel.mm on purpose.
//
// accel.mm binds three buffers and no MTLTexture, and adding one there would mean rebuilding
// libaccel.dylib - which every tool in this repo loads, and which the ISA peer's session is
// running against live. A broken rebuild would take both sides down for a capability only this
// side needs yet. So this is its own device, queue, library and pipeline, in its own dylib, and
// nothing existing changes.
//
// The buffer convention is the end-to-end harness's: A at 0, B at 1, C at 2, C copied back. The
// texture is bound at index 0 and filled from the host with a known pattern.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <string.h>
#include "../../tools/g17gpulock.h"

static id<MTLDevice> t_dev;
static id<MTLCommandQueue> t_q;
static id<MTLLibrary> t_lib;
static char t_err[512];

extern "C" int tx_init(void) {
  t_dev = MTLCreateSystemDefaultDevice();
  t_q = [t_dev newCommandQueue];
  return t_dev ? 0 : -1;
}
extern "C" const char *tx_error(void) { return t_err; }

// FROM A URL, not from data - same reason accel.mm gives: newLibraryWithData leaves a placeholder
// path in the archive's function script that the translator refuses.
extern "C" int tx_lib_from_url(const char *path) {
  NSError *err = nil;
  NSURL *u = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  t_lib = [t_dev newLibraryWithURL:u error:&err];
  if (!t_lib) { snprintf(t_err, sizeof t_err, "newLibraryWithURL: %s", err.description.UTF8String); return -2; }
  return 0;
}
extern "C" int tx_compile(const char *src) {
  NSError *err = nil;
  MTLCompileOptions *o = [MTLCompileOptions new];
  t_lib = [t_dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&err];
  if (!t_lib) { snprintf(t_err, sizeof t_err, "MSL: %s", err.description.UTF8String); return -1; }
  return 0;
}

// ONE PIPELINE PER PROCESS, kept from accel.mm and for its reason: Metal caches pipelines by AIR
// function hash, so a second build in the same process returns the FIRST one and every byte
// patched into the archive silently does nothing. That has produced wrong readings twice here.
extern "C" void *tx_pipeline_from_archive(const char *path, const char *kernel) {
  static int built = 0;
  if (built++) {
    snprintf(t_err, sizeof t_err, "refusing a SECOND pipeline in this process");
    fprintf(stderr, "tx_pipeline_from_archive: %s\n", t_err);
    return NULL;
  }
  NSError *err = nil;
  MTLBinaryArchiveDescriptor *d = [MTLBinaryArchiveDescriptor new];
  d.url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  id<MTLBinaryArchive> ar = [t_dev newBinaryArchiveWithDescriptor:d error:&err];
  if (!ar) { snprintf(t_err, sizeof t_err, "load archive: %s", err.description.UTF8String); return NULL; }
  MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
  pd.computeFunction = [t_lib newFunctionWithName:[NSString stringWithUTF8String:kernel]];
  pd.binaryArchives = @[ar];
  id<MTLComputePipelineState> ps = [t_dev newComputePipelineStateWithDescriptor:pd
      options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&err];
  if (!ps) { snprintf(t_err, sizeof t_err, "pipeline: %s", err.description.UTF8String); return NULL; }
  return (void *)CFBridgingRetain(ps);
}

// A PIPELINE STRAIGHT FROM THE COMPILED LIBRARY, no archive. This is the REFERENCE side: it is
// how Apple's own compilation of a kernel gets run, and it is also the control that says the
// harness itself works before any authored bytes are trusted to it. The archive path above is the
// one an authored program takes.
extern "C" void *tx_pipeline_from_source(const char *kernel) {
  NSError *err = nil;
  id<MTLFunction> fn = [t_lib newFunctionWithName:[NSString stringWithUTF8String:kernel]];
  if (!fn) { snprintf(t_err, sizeof t_err, "no function %s", kernel); return NULL; }
  id<MTLComputePipelineState> ps = [t_dev newComputePipelineStateWithFunction:fn error:&err];
  if (!ps) { snprintf(t_err, sizeof t_err, "pipeline: %s", err.description.UTF8String); return NULL; }
  return (void *)CFBridgingRetain(ps);
}

static int tx_finish(id<MTLCommandBuffer> cb) {
  if (cb.status == MTLCommandBufferStatusCompleted) { t_err[0] = 0; return 0; }
  NSError *e = cb.error;
  snprintf(t_err, sizeof t_err, "status=%ld domain=%s code=%ld %s",
           (long)cb.status, e ? e.domain.UTF8String : "-", e ? (long)e.code : -1L,
           e ? e.localizedDescription.UTF8String : "no NSError");
  return -1;
}

// ONE 2D r32Uint TEXTURE, w x h, filled from `tex` (w*h uint32 in row-major order), bound at
// texture index 0. Buffers A, B, C are `nbytes` each; C is copied back EVEN ON FAILURE, because a
// faulted dispatch still ran part of the program and what it wrote is the evidence.
extern "C" int tx_run(void *psh, const void *tex, unsigned w, unsigned h,
                      void *a, const void *b, void *c, unsigned nbytes,
                      unsigned tgw, unsigned gw, unsigned gh) {
  id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)psh;
  MTLTextureDescriptor *td =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                         width:w height:h mipmapped:NO];
  td.usage = MTLTextureUsageShaderRead;
  td.storageMode = MTLStorageModeShared;
  id<MTLTexture> T = [t_dev newTextureWithDescriptor:td];
  if (!T) { snprintf(t_err, sizeof t_err, "newTextureWithDescriptor returned nil"); return -2; }
  [T replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0
         withBytes:tex bytesPerRow:(NSUInteger)w * 4];

  id<MTLBuffer> A = [t_dev newBufferWithBytes:a length:nbytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> B = [t_dev newBufferWithBytes:b length:nbytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> C = [t_dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
  memcpy(C.contents, c, nbytes);

  id<MTLCommandBuffer> cb = g17_gpu_cb(t_q);
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:ps];
  [e setBuffer:A offset:0 atIndex:0];
  [e setBuffer:B offset:0 atIndex:1];
  [e setBuffer:C offset:0 atIndex:2];
  [e setTexture:T atIndex:0];
  [e dispatchThreadgroups:MTLSizeMake(gw, gh, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
  [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
  int st = tx_finish(cb);
  // BOTH C AND A COME BACK. C is where this project's own kernels write; A is buffer 0, where
  // Apple's texture probes write - ty-2d does `u[400] = X.read(...)`, and without copying A back
  // the control that decides whether a texture read works AT ALL is invisible. Copying an extra
  // buffer costs nothing and a missing observable has cost this project a session before
  // (ledger/g17-the-observable-was-never-copied-back.toml).
  memcpy(c, C.contents, nbytes);
  memcpy((void *)a, A.contents, nbytes);
  return st;
}
