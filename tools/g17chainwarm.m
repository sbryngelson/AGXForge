// Chained, warm-clock timing of single-dispatch tensor bundles: the like-for-like of an MLX op chain.
//
// g17projwarm times one dispatch per command buffer. MLX's microbenchmarks chain many ops inside one eval,
// so the GPU stays busy and the per-command-buffer cost is amortised. This tool does the same for our
// bundles: per config and per chain, one clock-ramp command buffer, then ONE command buffer holding `n`
// serial dispatches of the bundle (B rotated over `copies` buffers), and records that buffer's GPU span / n.
// Chains of configs are interleaved in a seeded shuffle. After the last chain the output is compared bit for
// bit with `expected` (a raw file of C's bytes) when one is given.
//
//   g17chainwarm plan.json records.json
//   plan: {"chains": R, "n": N, "seed": S, "configs": [{"tag", "bundle", "threads", "group", "copies", "expected", "rotate"}]}
//   "rotate": "a" rotates buffer 1 (A) over the copies instead of buffer 2 (B): for a program whose streamed operand is
//   there (g17qsm's q4 weights, MM 25.166); absent or "b", B is rotated as before.
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

static id<MTLComputePipelineState> bundlePipeline(id<MTLDevice> dev, NSString *dir) {
  NSError *e = nil;
  NSDictionary *m = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:
      [dir stringByAppendingPathComponent:@"manifest.json"]] options:0 error:&e];
  id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"scan.lib.metallib"]] error:&e];
  id<MTLFunction> fn = [lib newFunctionWithName:m[@"name"]];
  MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
  ad.url = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"scan.arc.metallib"]];
  id<MTLBinaryArchive> ar = [dev newBinaryArchiveWithDescriptor:ad error:&e];
  if (!fn || !ar) { fprintf(stderr, "bundle %s: %s\n", dir.UTF8String, e.description.UTF8String); exit(2); }
  MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new]; pd.computeFunction = fn; pd.binaryArchives = @[ar];
  id<MTLComputePipelineState> p = [dev newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                                 reflection:nil error:&e];
  if (!p) { fprintf(stderr, "pipeline %s: %s\n", dir.UTF8String, e.description.UTF8String); exit(2); }
  return p;
}

static id<MTLBuffer> filled(id<MTLDevice> dev, NSData *d) {
  id<MTLBuffer> b = [dev newBufferWithLength:d.length + 256 options:MTLResourceStorageModeShared];
  memset(b.contents, 0xA5, d.length + 256);
  memcpy((uint8_t *)b.contents + 128, d.bytes, d.length);
  return b;
}

static void finish(id<MTLCommandBuffer> cb) {
  [cb commit]; [cb waitUntilCompleted];
  if (cb.status != MTLCommandBufferStatusCompleted || cb.error) { fprintf(stderr, "command failed: %s\n", cb.error.description.UTF8String); exit(3); }
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 3) { fprintf(stderr, "usage: g17chainwarm plan.json records.json\n"); return 2; }
    NSDictionary *plan = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[1])] options:0 error:nil];
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError *e = nil;
    id<MTLLibrary> wl = [dev newLibraryWithSource:KSRC options:nil error:&e];
    if (!wl) { fprintf(stderr, "warm compile: %s\n", e.description.UTF8String); return 2; }
    id<MTLComputePipelineState> warm = [dev newComputePipelineStateWithFunction:[wl newFunctionWithName:@"kwarm"] error:&e];
    NSUInteger WN = 256 * 256;
    id<MTLBuffer> wa = [dev newBufferWithLength:WN * 2 + 256 options:MTLResourceStorageModeShared];
    id<MTLBuffer> wb = [dev newBufferWithLength:WN * 2 + 256 options:MTLResourceStorageModeShared];
    id<MTLBuffer> wc = [dev newBufferWithLength:WN * 4 + 256 options:MTLResourceStorageModeShared];
    memset(wa.contents, 0, WN * 2 + 256); memset(wb.contents, 0, WN * 2 + 256);
    NSUInteger n = [plan[@"n"] unsignedIntegerValue], chains = [plan[@"chains"] unsignedIntegerValue];
    NSMutableArray *cfg = [NSMutableArray array];
    for (NSDictionary *c in plan[@"configs"]) {
      NSString *dir = c[@"bundle"];
      NSData *a = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"a.f16"]];
      NSData *b = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"b.f16"]];
      NSData *c0 = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"c.f32"]];
      NSData *ex = [c[@"expected"] isKindOfClass:[NSString class]] ? [NSData dataWithContentsOfFile:c[@"expected"]] : nil;
      NSUInteger copies = [c[@"copies"] unsignedIntegerValue]; if (!copies) copies = 1;
      if (!a || !b || !c0 || (ex && ex.length != c0.length)) { fprintf(stderr, "inputs for %s\n", [c[@"tag"] UTF8String]); return 2; }
      BOOL rotA = [c[@"rotate"] isEqual:@"a"];
      NSMutableArray *bs = [NSMutableArray array];
      for (NSUInteger k = 0; k < copies; ++k) [bs addObject:filled(dev, rotA ? a : b)];
      [cfg addObject:@{@"tag": c[@"tag"], @"pipe": bundlePipeline(dev, dir), @"a": filled(dev, rotA ? b : a), @"bs": bs,
                       @"rotA": @(rotA),
                       @"c": filled(dev, c0), @"c0": c0, @"ex": ex ?: [NSNull null],
                       @"threads": c[@"threads"], @"group": c[@"group"]}];
    }
    NSMutableArray *units = [NSMutableArray array];
    for (NSUInteger r = 0; r < chains; ++r) for (NSUInteger i = 0; i < cfg.count; ++i) [units addObject:@[@(i), @(r)]];
    srand48([plan[@"seed"] longValue]);
    for (NSUInteger i = units.count; i > 1; --i) [units exchangeObjectAtIndex:i - 1 withObjectAtIndex:(NSUInteger)(drand48() * i)];
    NSMutableArray *recs = [NSMutableArray array];
    for (NSArray *u in units) {
      NSDictionary *c = cfg[[u[0] unsignedIntegerValue]];
      id<MTLBuffer> bc = c[@"c"]; NSData *c0 = c[@"c0"]; NSArray *bs = c[@"bs"];
      memcpy((uint8_t *)bc.contents + 128, c0.bytes, c0.length);
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
      NSUInteger thr = [c[@"threads"] unsignedIntegerValue], grp = [c[@"group"] unsignedIntegerValue];
      for (NSUInteger i = 0; i < n; ++i) {
        if ([c[@"rotA"] boolValue]) { [enc setBuffer:bs[i % bs.count] offset:128 atIndex:1]; [enc setBuffer:c[@"a"] offset:128 atIndex:2]; }
        else { [enc setBuffer:c[@"a"] offset:128 atIndex:1]; [enc setBuffer:bs[i % bs.count] offset:128 atIndex:2]; }
        [enc setBuffer:bc offset:128 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(thr / grp, 1, 1) threadsPerThreadgroup:MTLSizeMake(grp, 1, 1)];
      }
      [enc endEncoding]; finish(cb);
      double span = cb.GPUEndTime - cb.GPUStartTime;
      id ex = c[@"ex"];
      id exact = [ex isKindOfClass:[NSData class]] ? @(memcmp((uint8_t *)bc.contents + 128, [ex bytes], c0.length) == 0) : [NSNull null];
      [recs addObject:@{@"tag": c[@"tag"], @"chain": u[1], @"n": @(n), @"span_us": @(span * 1e6), @"per_dispatch_us": @(span * 1e6 / n),
                        @"bit_exact": exact}];
      if ([exact isEqual:@NO]) fprintf(stderr, "NOT BIT-EXACT: %s chain %s\n", [c[@"tag"] UTF8String], [u[1] description].UTF8String);
    }
    [[NSJSONSerialization dataWithJSONObject:@{@"records": recs, @"n": @(n), @"chains": @(chains), @"seed": plan[@"seed"]}
                                      options:NSJSONWritingPrettyPrinted error:nil] writeToFile:@(argv[2]) atomically:YES];
    printf("records %lu\n", (unsigned long)recs.count);
  }
  return 0;
}
