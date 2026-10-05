// Chained, warm-clock timing of Apple's own int8 matmul2d (Metal Performance Primitives), the like-for-like of
// tools/g17chainwarm.m for a Metal-source kernel: per config and per chain, three clock-ramp command buffers, then
// ONE command buffer of `n` serial dispatches (A and B fixed, so B is cache-warm), recording GPU span / n. Chains of
// configs are interleaved in a seeded shuffle. After each chain C is compared byte for byte with `expected`.
//
//   g17mppint8 plan.json records.json
//   plan: {"chains", "n", "seed", "configs": [{"tag", "lib", "fn", "M", "N", "K", "tm", "tn", "sg", "a", "b", "expected"}]}
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "g17gpulock.h"

static NSString *KSRC =
  @"#include <metal_stdlib>\nusing namespace metal;\n"
   "kernel void kwarm(device half *A [[buffer(0)]], device half *B [[buffer(1)]], device float *C [[buffer(2)]]) {\n"
   "  simdgroup_half8x8 a, b; simdgroup_float8x8 acc[8];\n"
   "  simdgroup_load(a, A, 256); simdgroup_load(b, B, 256);\n"
   "  for (int i=0;i<8;++i) acc[i] = make_filled_simdgroup_matrix<float,8,8>(0.0f);\n"
   "  for (uint r=0;r<32768;++r) { for (int i=0;i<8;++i) simdgroup_multiply_accumulate(acc[i],a,b,acc[i]); }\n"
   "  simdgroup_store(acc[0], C, 256);\n}\n";

static void finish(id<MTLCommandBuffer> cb) {
  [cb commit]; [cb waitUntilCompleted];
  if (cb.status != MTLCommandBufferStatusCompleted || cb.error) { fprintf(stderr, "command failed: %s\n", cb.error.description.UTF8String); exit(3); }
}

static id<MTLBuffer> filled(id<MTLDevice> dev, NSData *d) {
  id<MTLBuffer> b = [dev newBufferWithLength:d.length options:MTLResourceStorageModeShared];
  memcpy(b.contents, d.bytes, d.length);
  return b;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 3) { fprintf(stderr, "usage: g17mppint8 plan.json records.json\n"); return 2; }
    NSDictionary *plan = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[1])] options:0 error:nil];
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError *e = nil;
    id<MTLLibrary> wl = [dev newLibraryWithSource:KSRC options:nil error:&e];
    if (!wl) { fprintf(stderr, "warm compile: %s\n", e.description.UTF8String); return 2; }
    id<MTLComputePipelineState> warm = [dev newComputePipelineStateWithFunction:[wl newFunctionWithName:@"kwarm"] error:&e];
    NSUInteger WN = 256 * 256;
    id<MTLBuffer> wa = [dev newBufferWithLength:WN * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> wb = [dev newBufferWithLength:WN * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> wc = [dev newBufferWithLength:WN * 4 options:MTLResourceStorageModeShared];
    memset(wa.contents, 0, WN * 2); memset(wb.contents, 0, WN * 2);
    NSUInteger n = [plan[@"n"] unsignedIntegerValue], chains = [plan[@"chains"] unsignedIntegerValue];
    NSMutableDictionary *data = [NSMutableDictionary dictionary];   // one buffer set per shape, shared by its configs
    NSMutableArray *cfg = [NSMutableArray array];
    for (NSDictionary *c in plan[@"configs"]) {
      NSUInteger M = [c[@"M"] unsignedIntegerValue], N = [c[@"N"] unsignedIntegerValue], K = [c[@"K"] unsignedIntegerValue];
      NSUInteger tm = [c[@"tm"] unsignedIntegerValue], tn = [c[@"tn"] unsignedIntegerValue], sg = [c[@"sg"] unsignedIntegerValue];
      if (M % tm || N % tn) { fprintf(stderr, "%s: tile does not divide the shape\n", [c[@"tag"] UTF8String]); return 2; }
      NSDictionary *d = data[c[@"expected"]];
      if (!d) {
        NSData *a = [NSData dataWithContentsOfFile:c[@"a"]], *b = [NSData dataWithContentsOfFile:c[@"b"]];
        NSData *ex = [NSData dataWithContentsOfFile:c[@"expected"]];
        if (a.length != M * K || b.length != K * N || ex.length != M * N * 4) { fprintf(stderr, "inputs for %s\n", [c[@"tag"] UTF8String]); return 2; }
        uint32_t dims[4] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, 0};
        d = @{@"a": filled(dev, a), @"b": filled(dev, b), @"ex": ex,
              @"c": [dev newBufferWithLength:ex.length options:MTLResourceStorageModeShared],
              @"dims": [NSData dataWithBytes:dims length:sizeof dims]};
        data[c[@"expected"]] = d;
      }
      id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:c[@"lib"]] error:&e];
      id<MTLFunction> fn = [lib newFunctionWithName:c[@"fn"]];
      id<MTLComputePipelineState> p = fn ? [dev newComputePipelineStateWithFunction:fn error:&e] : nil;
      if (!p) { fprintf(stderr, "pipeline %s: %s\n", [c[@"tag"] UTF8String], e.description.UTF8String); return 2; }
      if (p.maxTotalThreadsPerThreadgroup < 32 * sg) { fprintf(stderr, "%s: needs %lu threads, pipeline allows %lu\n",
          [c[@"tag"] UTF8String], (unsigned long)(32 * sg), (unsigned long)p.maxTotalThreadsPerThreadgroup); return 2; }
      [cfg addObject:@{@"tag": c[@"tag"], @"pipe": p, @"d": d, @"grid": @[@(N / tn), @(M / tm)], @"threads": @(32 * sg)}];
    }
    NSMutableArray *units = [NSMutableArray array];
    for (NSUInteger r = 0; r < chains; ++r) for (NSUInteger i = 0; i < cfg.count; ++i) [units addObject:@[@(i), @(r)]];
    srand48([plan[@"seed"] longValue]);
    for (NSUInteger i = units.count; i > 1; --i) [units exchangeObjectAtIndex:i - 1 withObjectAtIndex:(NSUInteger)(drand48() * i)];
    NSMutableArray *recs = [NSMutableArray array];
    for (NSArray *u in units) {
      NSDictionary *c = cfg[[u[0] unsignedIntegerValue]], *d = c[@"d"];
      id<MTLBuffer> bc = d[@"c"]; NSData *ex = d[@"ex"], *dims = d[@"dims"];
      memset(bc.contents, 0xA5, ex.length);                           // a stale result cannot pass the comparison
      for (int w = 0; w < 3; ++w) {                                   // clock ramp, its own command buffer
        id<MTLCommandBuffer> cb = g17_gpu_cb(q); id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:warm]; [enc setBuffer:wa offset:0 atIndex:0]; [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:wc offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(960, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding]; finish(cb);
      }
      id<MTLCommandBuffer> cb = g17_gpu_cb(q);
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      [enc setComputePipelineState:c[@"pipe"]];
      [enc setBuffer:d[@"a"] offset:0 atIndex:0]; [enc setBuffer:d[@"b"] offset:0 atIndex:1]; [enc setBuffer:bc offset:0 atIndex:2];
      [enc setBytes:dims.bytes length:dims.length atIndex:3];
      NSArray *g = c[@"grid"];
      for (NSUInteger i = 0; i < n; ++i)
        [enc dispatchThreadgroups:MTLSizeMake([g[0] unsignedIntegerValue], [g[1] unsignedIntegerValue], 1)
            threadsPerThreadgroup:MTLSizeMake([c[@"threads"] unsignedIntegerValue], 1, 1)];
      [enc endEncoding]; finish(cb);
      double span = cb.GPUEndTime - cb.GPUStartTime;
      BOOL exact = memcmp(bc.contents, ex.bytes, ex.length) == 0;
      [recs addObject:@{@"tag": c[@"tag"], @"chain": u[1], @"n": @(n), @"span_us": @(span * 1e6),
                        @"per_dispatch_us": @(span * 1e6 / n), @"bit_exact": @(exact)}];
      if (!exact) fprintf(stderr, "NOT BIT-EXACT: %s chain %s\n", [c[@"tag"] UTF8String], [u[1] description].UTF8String);
    }
    [[NSJSONSerialization dataWithJSONObject:@{@"records": recs, @"n": @(n), @"chains": @(chains), @"seed": plan[@"seed"],
                                               @"device": dev.name}
                                      options:NSJSONWritingPrettyPrinted error:nil] writeToFile:@(argv[2]) atomically:YES];
    printf("records %lu\n", (unsigned long)recs.count);
  }
  return 0;
}
