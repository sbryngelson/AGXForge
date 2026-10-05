// Runtime-compiled Metal kernels for one question: which code paths run on the
// M5's per-core Neural Accelerators? the only honest proof is throughput - the
// SIMD ALUs top out near 7-8 TFLOPS here, so a kernel sustaining ~30 TFLOPS fp16
// is on the matrix units. exposes: compile(source) -> library, and bench(kernel,
// N, iters, tg_w, tg_h, grid_w, grid_h) -> seconds per GEMM of N^3.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <string.h>
#include <string>
extern "C" int ac_alloc2(unsigned, unsigned, unsigned, const void *, const void *);
static id<MTLDevice> g_dev; static id<MTLCommandQueue> g_q; static id<MTLLibrary> g_lib;
static id<MTLBuffer> g_a, g_b, g_c; static unsigned g_n;

#include "../../tools/g17gpulock.h"
extern "C" int ac_gpu_lock(void) { return g17_gpu_lock(); }
extern "C" int ac_init(void) { g_dev = MTLCreateSystemDefaultDevice(); g_q = [g_dev newCommandQueue]; return g_dev ? 0 : -1; }
extern "C" const char *ac_name(void) { return g_dev.name.UTF8String; }
// LIBRARY FROM A FILE URL, not from data. The archive's Metal function script records how the
// library was obtained: newLibraryWithData leaves a placeholder
// "_Path_not_available_for_lib_from_data_with_UUID_..." that applegpu-nt cannot resolve, so the
// translator refuses the script and never reaches code generation. From a URL it records the real
// path. ledger/g17-airnt-emit-assembly-is-compiled-out.toml
extern "C" int ac_lib_from_url(const char *path) {
  NSError *err = nil;
  NSURL *u = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  g_lib = [g_dev newLibraryWithURL:u error:&err];
  if (!g_lib) { fprintf(stderr, "newLibraryWithURL: %s\n", err.description.UTF8String); return -2; }
  return 0;
}
extern "C" int ac_lib_from_data(const char *path) {
  NSError *err = nil;
  NSData *d = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
  if (!d) { fprintf(stderr, "no file %s\n", path); return -1; }
  dispatch_data_t dd = dispatch_data_create(d.bytes, d.length, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  g_lib = [g_dev newLibraryWithData:dd error:&err];
  if (!g_lib) { fprintf(stderr, "newLibraryWithData: %s\n", err.description.UTF8String); return -2; }
  return 0;
}
extern "C" int ac_compile(const char *src) {
  NSError *err = nil;
  MTLCompileOptions *o = [MTLCompileOptions new];
  g_lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&err];
  if (!g_lib) { fprintf(stderr, "MSL: %s\n", err.description.UTF8String); return -1; }
  return 0;
}
extern "C" int ac_alloc(unsigned n, unsigned elem, const void *a, const void *b) { return ac_alloc2(n, elem, elem, a, b); }
extern "C" int ac_alloc2(unsigned n, unsigned elem, unsigned celem, const void *a, const void *b) {
  g_n = n; size_t bytes = (size_t)n * n * elem;
  g_a = [g_dev newBufferWithBytes:a length:bytes options:MTLResourceStorageModeShared];
  g_b = [g_dev newBufferWithBytes:b length:bytes options:MTLResourceStorageModeShared];
  g_c = [g_dev newBufferWithLength:(size_t)n * n * celem options:MTLResourceStorageModeShared];
  return 0;
}
extern "C" void *ac_c(void) { return g_c.contents; }
extern "C" double ac_bench(const char *kernel, int iters, unsigned tgw, unsigned tgh, unsigned gw, unsigned gh) {
  NSError *err = nil;
  id<MTLFunction> f = [g_lib newFunctionWithName:[NSString stringWithUTF8String:kernel]];
  if (!f) { fprintf(stderr, "no kernel %s\n", kernel); return -1; }
  id<MTLComputePipelineState> ps = [g_dev newComputePipelineStateWithFunction:f error:&err];
  if (!ps) { fprintf(stderr, "pipeline: %s\n", err.description.UTF8String); return -2; }
  double best = 1e9;
  for (int rep = 0; rep < iters + 2; rep++) {
    id<MTLCommandBuffer> cb = g17_gpu_cb(g_q);
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:ps];
    [e setBuffer:g_a offset:0 atIndex:0]; [e setBuffer:g_b offset:0 atIndex:1]; [e setBuffer:g_c offset:0 atIndex:2];
    [e setBytes:&g_n length:4 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(gw, gh, 1) threadsPerThreadgroup:MTLSizeMake(tgw, tgh, 1)];
    [e endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    double dt = cb.GPUEndTime - cb.GPUStartTime;
    if (rep >= 2 && dt < best) best = dt;
  }
  return best;
}
// serialize the compiled pipeline for a kernel into a Metal binary archive - the
// container that holds the GPU machine code for this device family.
extern "C" int ac_archive(const char *kernel, const char *path) {
  NSError *err = nil;
  MTLBinaryArchiveDescriptor *d = [MTLBinaryArchiveDescriptor new];
  id<MTLBinaryArchive> ar = [g_dev newBinaryArchiveWithDescriptor:d error:&err];
  if (!ar) { fprintf(stderr, "archive: %s\n", err.description.UTF8String); return -1; }
  MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
  pd.computeFunction = [g_lib newFunctionWithName:[NSString stringWithUTF8String:kernel]];
  if (![ar addComputePipelineFunctionsWithDescriptor:pd error:&err]) { fprintf(stderr, "add: %s\n", err.description.UTF8String); return -2; }
  if (![ar serializeToURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&err]) { fprintf(stderr, "serialize: %s\n", err.description.UTF8String); return -3; }
  return 0;
}
// The same for a VERTEX function. Apple ships seven blit_vertex_* driver shaders whose AIR is on
// disk but which the compute path cannot archive - a vertex function is not a compute function -
// so they were silently absent from the corpus. Rasterization is disabled and no fragment
// function is attached, which is enough to compile the vertex stage and get its machine code.
extern "C" int ac_archive_vertex(const char *fn, const char *path) {
  NSError *err = nil;
  MTLBinaryArchiveDescriptor *d = [MTLBinaryArchiveDescriptor new];
  id<MTLBinaryArchive> ar = [g_dev newBinaryArchiveWithDescriptor:d error:&err];
  if (!ar) { fprintf(stderr, "varchive: %s\n", err.description.UTF8String); return -1; }
  MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
  pd.vertexFunction = [g_lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
  if (!pd.vertexFunction) { fprintf(stderr, "varchive: no function %s\n", fn); return -4; }
  pd.rasterizationEnabled = NO;
  if (![ar addRenderPipelineFunctionsWithDescriptor:pd error:&err]) {
    fprintf(stderr, "vadd: %s\n", err.description.UTF8String); return -2; }
  if (![ar serializeToURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&err]) {
    fprintf(stderr, "vserialize: %s\n", err.description.UTF8String); return -3; }
  return 0;
}
// A RASTERIZING RENDER PIPELINE: vertex + fragment, one BGRA8 colour attachment. ac_archive_vertex
// disables rasterization, which the driver accepts only for a VOID vertex function, so a vertex
// function with outputs and every fragment stage were unreachable from this harness. Compile only:
// the archive is serialised, nothing is encoded or committed.
extern "C" int ac_archive_render(const char *vs, const char *fs, const char *path) {
  NSError *err = nil;
  MTLBinaryArchiveDescriptor *d = [MTLBinaryArchiveDescriptor new];
  id<MTLBinaryArchive> ar = [g_dev newBinaryArchiveWithDescriptor:d error:&err];
  if (!ar) { fprintf(stderr, "rarchive: %s\n", err.description.UTF8String); return -1; }
  MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
  pd.vertexFunction = [g_lib newFunctionWithName:[NSString stringWithUTF8String:vs]];
  pd.fragmentFunction = [g_lib newFunctionWithName:[NSString stringWithUTF8String:fs]];
  if (!pd.vertexFunction || !pd.fragmentFunction) { fprintf(stderr, "rarchive: missing %s or %s\n", vs, fs); return -4; }
  pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
  pd.rasterizationEnabled = YES;
  if (![ar addRenderPipelineFunctionsWithDescriptor:pd error:&err]) {
    fprintf(stderr, "radd: %s\n", err.description.UTF8String); return -2; }
  if (![ar serializeToURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&err]) {
    fprintf(stderr, "rserialize: %s\n", err.description.UTF8String); return -3; }
  return 0;
}
// Load a (possibly patched) binary archive and build a pipeline from it, so the
// GPU executes bytes we chose rather than bytes the compiler just produced.
extern "C" void *ac_pipeline_from_archive(const char *path, const char *kernel) {
  // ONE PIPELINE PER PROCESS, enforced. Metal caches pipelines by AIR function hash, so a second
  // build in the same process returns the FIRST one - and every byte patched into the archive
  // silently does nothing. That defect has now produced wrong readings twice (see
  // ledger/g17-branch-conditional-back.toml), each time looking like "the mutation is inert".
  // Refusing the second call turns a silent wrong answer into a loud failure.
  static int built = 0;
  if (built++) { fprintf(stderr, "ac_pipeline_from_archive: refusing a SECOND pipeline in this "
                                 "process - Metal would return the cached, unpatched one\n");
                 return NULL; }
  NSError *err = nil;
  MTLBinaryArchiveDescriptor *d = [MTLBinaryArchiveDescriptor new];
  d.url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  id<MTLBinaryArchive> ar = [g_dev newBinaryArchiveWithDescriptor:d error:&err];
  if (!ar) { fprintf(stderr, "load archive: %s\n", err.description.UTF8String); return NULL; }
  MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
  pd.computeFunction = [g_lib newFunctionWithName:[NSString stringWithUTF8String:kernel]];
  pd.binaryArchives = @[ar];
  id<MTLComputePipelineState> ps = [g_dev newComputePipelineStateWithDescriptor:pd
      options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&err];
  if (!ps) { fprintf(stderr, "pipeline from archive: %s\n", err.description.UTF8String); return NULL; }
  return (void *)CFBridgingRetain(ps);
}
// esz is the OPERAND element size in bytes: 2 for half and bfloat, 4 for the fp32 and
// relaxed-fp32 paths. C is always fp32.
// THE COMMAND BUFFER'S ERROR, KEPT. Every dispatch path used to collapse a failure into -1, so
// "the kernel faulted", "the GPU timed out" and "the encoder was rejected" were the same result.
// They are not the same finding: a fault is a bug in the emitted code, a timeout is a hang and a
// hazard to the machine. ledger/g17-hang-poisons-the-run.toml
static char g_err[1024];
extern "C" const char *ac_last_error(void) { return g_err; }
static int ac_finish(id<MTLCommandBuffer> cb) {
  if (cb.status == MTLCommandBufferStatusCompleted) { g_err[0] = 0; return 0; }
  NSError *e = cb.error;
  snprintf(g_err, sizeof g_err, "status=%ld domain=%s code=%ld %s",
           (long)cb.status, e ? e.domain.UTF8String : "-", e ? (long)e.code : -1L,
           e ? e.localizedDescription.UTF8String : "no NSError");
  return -1;
}

extern "C" int ac_run_ps_es(void *psh, void *a, void *b, void *c, unsigned n, unsigned esz,
                            unsigned tgw, unsigned gw, unsigned gh) {
  id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)psh;
  id<MTLBuffer> A = [g_dev newBufferWithBytes:a length:(size_t)n*n*esz options:MTLResourceStorageModeShared];
  id<MTLBuffer> B = [g_dev newBufferWithBytes:b length:(size_t)n*n*esz options:MTLResourceStorageModeShared];
  id<MTLBuffer> C = [g_dev newBufferWithLength:(size_t)n*n*4 options:MTLResourceStorageModeShared];
  memcpy(C.contents, c, (size_t)n*n*4);   // caller supplies the initial C: multiply vs
                                          // multiply_accumulate is only visible against a prefill
  id<MTLCommandBuffer> cb = g17_gpu_cb(g_q);
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:ps];
  [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1]; [e setBuffer:C offset:0 atIndex:2];
  [e dispatchThreadgroups:MTLSizeMake(gw,gh,1) threadsPerThreadgroup:MTLSizeMake(tgw,1,1)];
  [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
  // The output is copied back EVEN ON FAILURE. A faulted dispatch still ran some of the
  // program, and what it managed to write is the evidence for where it died - discarding it
  // turned every fault into "no information at all".
  int st = ac_finish(cb);
  memcpy(c, C.contents, (size_t)n*n*4);
  return st;
}
// ac_run_ps_es, TIMED: the same buffers and dispatch, run warm+reps times, the best GPU time
// (GPUEndTime - GPUStartTime, seconds) written to *best. ac_bench times Apple-compiled Metal only;
// an authored pipeline had no timed path, and Apple's compiler reassociates the dependent chains a
// latency measurement needs. Stops at the first failed command buffer and returns its status.
extern "C" int ac_time_ps_es(void *psh, void *a, void *b, void *c, unsigned n, unsigned esz,
                             unsigned tgw, unsigned gw, unsigned gh, int reps, double *best) {
  id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)psh;
  id<MTLBuffer> A = [g_dev newBufferWithBytes:a length:(size_t)n*n*esz options:MTLResourceStorageModeShared];
  id<MTLBuffer> B = [g_dev newBufferWithBytes:b length:(size_t)n*n*esz options:MTLResourceStorageModeShared];
  id<MTLBuffer> C = [g_dev newBufferWithLength:(size_t)n*n*4 options:MTLResourceStorageModeShared];
  memcpy(C.contents, c, (size_t)n*n*4);
  *best = 1e9;
  for (int rep = 0; rep < reps + 2; rep++) {
    id<MTLCommandBuffer> cb = g17_gpu_cb(g_q);
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:ps];
    [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1]; [e setBuffer:C offset:0 atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(gw,gh,1) threadsPerThreadgroup:MTLSizeMake(tgw,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    int st = ac_finish(cb);
    if (st) { memcpy(c, C.contents, (size_t)n*n*4); return st; }
    double dt = cb.GPUEndTime - cb.GPUStartTime;
    if (rep >= 2 && dt < *best) *best = dt;
  }
  memcpy(c, C.contents, (size_t)n*n*4);
  return 0;
}
extern "C" int ac_run_ps(void *psh, void *a, void *b, void *c, unsigned n,
                         unsigned tgw, unsigned gw, unsigned gh) {
  return ac_run_ps_es(psh, a, b, c, n, 2, tgw, gw, gh);
}

// Same dispatch, but ALSO copies buffer 1 back to the host. ac_run_ps returns only C, which is
// right for our own kernels - they take A and B as inputs and write C - and wrong for Apple's
// driver shaders, which write buffer 1. blit_fast_clear_gen2_1 stores to `data` at index 1, so
// its output was invisible and read as "the shader wrote nothing".
// ledger/g17-apple-shader-dispatch-blocker.toml
extern "C" int ac_run_ps_rb(void *psh, void *a, void *b, void *c, unsigned n, unsigned esz,
                            unsigned tgw, unsigned gw, unsigned gh) {
  id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)psh;
  size_t alen = (size_t)n*n*esz, clen = (size_t)n*n*4;
  id<MTLBuffer> A = [g_dev newBufferWithBytes:a length:alen options:MTLResourceStorageModeShared];
  id<MTLBuffer> B = [g_dev newBufferWithBytes:b length:alen options:MTLResourceStorageModeShared];
  id<MTLBuffer> C = [g_dev newBufferWithLength:clen options:MTLResourceStorageModeShared];
  memcpy(C.contents, c, clen);
  id<MTLCommandBuffer> cb = g17_gpu_cb(g_q);
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:ps];
  [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1]; [e setBuffer:C offset:0 atIndex:2];
  [e dispatchThreadgroups:MTLSizeMake(gw,gh,1) threadsPerThreadgroup:MTLSizeMake(tgw,1,1)];
  [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
  if (ac_finish(cb)) return -1;
  memcpy(b, B.contents, alen);          // the readback ac_run_ps lacks
  memcpy(c, C.contents, clen);
  return 0;
}

// EVERY BUFFER COPIED BACK. ac_run_ps returns C and ac_run_ps_rb returns B and C, and neither
// returns A - which is invisible until a program has to be built on a host whose kernel binds
// buffer 0 and nothing else. a6-tgcalc is such a host (its only device argument is `u`), and its
// own device store therefore writes A, so a threadgroup round trip authored on it cannot be read
// at all through the older entry points.
// `tglen` is the DYNAMIC threadgroup allocation, in bytes. A kernel that declares
// `threadgroup uint *tgm [[threadgroup(0)]]` gets its threadgroup memory from the encoder, not
// from the function - so without this call the allocation is zero, a store into it is dropped and
// a load returns zero, which is exactly what a threadgroup round trip looked like before.
extern "C" int ac_run_ps_abc(void *psh, void *a, void *b, void *c, unsigned n, unsigned esz,
                             unsigned tgw, unsigned gw, unsigned gh, unsigned tglen) {
  id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)psh;
  size_t alen = (size_t)n*n*esz, clen = (size_t)n*n*4;
  id<MTLBuffer> A = [g_dev newBufferWithBytes:a length:alen options:MTLResourceStorageModeShared];
  id<MTLBuffer> B = [g_dev newBufferWithBytes:b length:alen options:MTLResourceStorageModeShared];
  id<MTLBuffer> C = [g_dev newBufferWithLength:clen options:MTLResourceStorageModeShared];
  memcpy(C.contents, c, clen);
  id<MTLCommandBuffer> cb = g17_gpu_cb(g_q);
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:ps];
  [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1]; [e setBuffer:C offset:0 atIndex:2];
  if (tglen) [e setThreadgroupMemoryLength:((tglen + 15) & ~15u) atIndex:0];
  [e dispatchThreadgroups:MTLSizeMake(gw,gh,1) threadsPerThreadgroup:MTLSizeMake(tgw,1,1)];
  [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
  int st = ac_finish(cb);            // the contents come back even on failure, as elsewhere here
  memcpy(a, A.contents, alen); memcpy(b, B.contents, alen); memcpy(c, C.contents, clen);
  return st;
}
// HOW MUCH IMAGEBLOCK MEMORY DOES THIS PIPELINE ASK FOR? The number the driver reports for a given
// tile size, straight from the pipeline state. Zero from a pipeline that should have a tile is the
// difference between "the instruction is wrong" and "the tile was never allocated" - the ambiguity
// that read as zero three times today.
extern "C" unsigned long ac_ib_len(void *psh, unsigned w, unsigned h) {
  id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)psh;
  return (unsigned long)[ps imageblockMemoryLengthForDimensions:MTLSizeMake(w, h, 1)];
}
// The same question for a pipeline built the ORDINARY way, from the library function with no
// binary archive - which is how the peer's runner builds it, and the only difference left between
// their run that works and this harness's that does not.
extern "C" unsigned long ac_ib_len_fn(const char *kernel, unsigned w, unsigned h) {
  NSError *err = nil;
  MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
  pd.computeFunction = [g_lib newFunctionWithName:[NSString stringWithUTF8String:kernel]];
  id<MTLComputePipelineState> ps = [g_dev newComputePipelineStateWithDescriptor:pd
      options:MTLPipelineOptionNone reflection:nil error:&err];
  if (!ps) { fprintf(stderr, "pipeline from function: %s\n", err.description.UTF8String); return 0; }
  return (unsigned long)[ps imageblockMemoryLengthForDimensions:MTLSizeMake(w, h, 1)];
}

// H4, route 1: the driver's own limit for a pipeline, read at creation with NO dispatch.
// maxTotalThreadsPerThreadgroup reflects the register pressure of the compiled function, so it
// bounds the occupancy question cheaply. It is threads per THREADGROUP, not residency per core, so
// it cannot answer H4 by itself - if it does not move as a kernel's live registers rise, that is
// itself the informative result, and it is the thing to establish before building a counter path.
// Returns 0 on failure, which is distinguishable from any real limit (those are >= 1).
extern "C" unsigned long ac_max_threads_fn(const char *kernel) {
  NSError *err = nil;
  id<MTLFunction> f = [g_lib newFunctionWithName:[NSString stringWithUTF8String:kernel]];
  if (!f) { fprintf(stderr, "no function %s\n", kernel); return 0; }
  id<MTLComputePipelineState> ps = [g_dev newComputePipelineStateWithFunction:f error:&err];
  if (!ps) { fprintf(stderr, "pipeline: %s\n", err.description.UTF8String); return 0; }
  return (unsigned long)ps.maxTotalThreadsPerThreadgroup;
}

// The same pipeline's static threadgroup memory, so a caller can tell a register-driven limit from
// a threadgroup-memory-driven one without a second build.
extern "C" unsigned long ac_tg_mem_fn(const char *kernel) {
  NSError *err = nil;
  id<MTLFunction> f = [g_lib newFunctionWithName:[NSString stringWithUTF8String:kernel]];
  if (!f) return 0;
  id<MTLComputePipelineState> ps = [g_dev newComputePipelineStateWithFunction:f error:&err];
  if (!ps) { fprintf(stderr, "pipeline: %s\n", err.description.UTF8String); return 0; }
  return (unsigned long)ps.staticThreadgroupMemoryLength;
}

// AN IMAGEBLOCK IS SIZED BY THE ENCODER TOO, and by a different call from threadgroup memory:
// setImageblockWidthAndHeight. A kernel that declares imageblock<T, layout_explicit> gets a tile of
// zero by zero without it, and then every imageblock read returns zero and every write goes nowhere
// - which is indistinguishable from an address that is not indexed the way it is declared. This is
// a NEW entry point rather than two more parameters on the old one: adding a column to agx3meta's
// output broke every parser that read it positionally this afternoon, and a signature is the same
// kind of contract.
extern "C" int ac_run_ps_ib(void *psh, void *a, void *b, void *c, unsigned n, unsigned esz,
                            unsigned tgw, unsigned gw, unsigned gh, unsigned tglen,
                            unsigned ibw, unsigned ibh) {
  id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)psh;
  size_t alen = (size_t)n*n*esz, clen = (size_t)n*n*4;
  id<MTLBuffer> A = [g_dev newBufferWithBytes:a length:alen options:MTLResourceStorageModeShared];
  id<MTLBuffer> B = [g_dev newBufferWithBytes:b length:alen options:MTLResourceStorageModeShared];
  id<MTLBuffer> C = [g_dev newBufferWithLength:clen options:MTLResourceStorageModeShared];
  memcpy(C.contents, c, clen);
  id<MTLCommandBuffer> cb = g17_gpu_cb(g_q);
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:ps];
  [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1]; [e setBuffer:C offset:0 atIndex:2];
  if (tglen) [e setThreadgroupMemoryLength:((tglen + 15) & ~15u) atIndex:0];
  if (ibw && ibh) [e setImageblockWidth:ibw height:ibh];
  [e dispatchThreadgroups:MTLSizeMake(gw,gh,1) threadsPerThreadgroup:MTLSizeMake(tgw,1,1)];
  [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
  int st = ac_finish(cb);
  memcpy(a, A.contents, alen); memcpy(b, B.contents, alen); memcpy(c, C.contents, clen);
  return st;
}

// Persistent bindings. ac_run_ps allocates three buffers per call, which at this tile
// size costs far more than the kernel and would make any latency comparison a
// measurement of the harness. Bind once, then dispatch repeatedly.
static id<MTLBuffer> p_a, p_b, p_c; static unsigned p_n;
extern "C" int ac_bind(unsigned n, unsigned esz, void *a, void *b, void *c) {
  p_n = n;
  p_a = [g_dev newBufferWithBytes:a length:(size_t)n*n*esz options:MTLResourceStorageModeShared];
  p_b = [g_dev newBufferWithBytes:b length:(size_t)n*n*esz options:MTLResourceStorageModeShared];
  p_c = [g_dev newBufferWithBytes:c length:(size_t)n*n*4 options:MTLResourceStorageModeShared];
  return (p_a && p_b && p_c) ? 0 : -1;
}
// reps dispatches inside ONE command buffer, so the measurement is per-dispatch cost and
// not per-command-buffer submission. ps2 non-NULL adds a second pipeline per rep: that is
// the unfused schedule, two dispatches for the two products.
extern "C" int ac_run_bound(void *psh, void *psh2, unsigned reps, unsigned tgw,
                            unsigned gw, unsigned gh) {
  id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)psh;
  id<MTLComputePipelineState> ps2 = psh2 ? (__bridge id<MTLComputePipelineState>)psh2 : nil;
  id<MTLCommandBuffer> cb = g17_gpu_cb(g_q);
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setBuffer:p_a offset:0 atIndex:0]; [e setBuffer:p_b offset:0 atIndex:1];
  [e setBuffer:p_c offset:0 atIndex:2];
  for (unsigned i = 0; i < reps; i++) {
    [e setComputePipelineState:ps];
    [e dispatchThreadgroups:MTLSizeMake(gw,gh,1) threadsPerThreadgroup:MTLSizeMake(tgw,1,1)];
    if (ps2) {
      [e setComputePipelineState:ps2];
      [e dispatchThreadgroups:MTLSizeMake(gw,gh,1) threadsPerThreadgroup:MTLSizeMake(tgw,1,1)];
    }
  }
  [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
  return cb.status == MTLCommandBufferStatusCompleted ? 0 : -1;
}
extern "C" void ac_readback(void *c) { memcpy(c, p_c.contents, (size_t)p_n*p_n*4); }

// Enumerate the compute functions in the loaded library. Needed to drive Apple's own driver
// shaders (AGXCompilerCore.framework/ds/*.ds) through the same archive path as our own kernels:
// those modules are not ours, so their entry-point names have to be read rather than assumed.
extern "C" const char *ac_functions(void) {
  static std::string out;
  out.clear();
  if (!g_lib) return "";
  for (NSString *n in g_lib.functionNames) { out += n.UTF8String; out += "\n"; }
  return out.c_str();
}
