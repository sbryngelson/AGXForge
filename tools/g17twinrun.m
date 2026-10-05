// g17twinrun: the matched study's kernel harness (MM 25.211). Runs several ARMS of one kernel - our bundle (program
// bytes through its binary archive) and its Apple-compiled twin (a metallib xcrun metal built from Metal source) - on
// the same buffers, in one process.
//   g17twinrun spec.json out.json
// spec: {arms: [{kind: "bundle", dir | kind: "apple", metallib, fn; dump: {"0": path}}], threads, group,
//        buffers: {"0": {file}, ...}, rotate: "2" (timing rotates `copies` copies of that buffer), copies, chains, n, warm_s}
// Correctness: each arm once on copy 0 from freshly loaded buffers, then its `dump` buffers are written.
// Timing: warm_s seconds of the arms' own dispatches first (the clock), then `chains` command buffers of `n`
// dispatches per arm, the arm order reversed every chain; out.json {per_dispatch_us: [[arm 0 ...], [arm 1 ...]]}.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "g17gpulock.h"

static id<MTLComputePipelineState> pipe_for(id<MTLDevice> dev, NSDictionary *s) {
  NSError *e = nil;
  if ([s[@"kind"] isEqual:@"apple"]) {
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:s[@"metallib"]] error:&e];
    id<MTLFunction> fn = [lib newFunctionWithName:s[@"fn"]];
    if (!fn) { fprintf(stderr, "apple fn: %s\n", e.description.UTF8String); exit(2); }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&e];
    if (!p) { fprintf(stderr, "apple pipe: %s\n", e.description.UTF8String); exit(2); }
    return p;
  }
  NSString *dir = s[@"dir"];
  NSDictionary *m = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:
      [dir stringByAppendingPathComponent:@"manifest.json"]] options:0 error:&e];
  id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"scan.lib.metallib"]] error:&e];
  id<MTLFunction> fn = [lib newFunctionWithName:m[@"name"]];
  MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
  ad.url = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"scan.arc.metallib"]];
  id<MTLBinaryArchive> ar = [dev newBinaryArchiveWithDescriptor:ad error:&e];
  MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new]; pd.computeFunction = fn; pd.binaryArchives = @[ar];
  id<MTLComputePipelineState> p = [dev newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                                 reflection:nil error:&e];
  if (!p) { fprintf(stderr, "bundle pipe: %s\n", e.description.UTF8String); exit(2); }
  return p;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 3) { fprintf(stderr, "usage: twinrun spec.json out.json\n"); return 2; }
    NSDictionary *s = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[1])] options:0 error:nil];
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSArray *arms = s[@"arms"];
    NSMutableArray *pipes = [NSMutableArray array];
    for (NSDictionary *a in arms) [pipes addObject:pipe_for(dev, a)];
    __block id<MTLComputePipelineState> p = pipes[0];
    NSUInteger threads = [s[@"threads"] unsignedIntegerValue], group = [s[@"group"] unsignedIntegerValue];
    NSString *rot = s[@"rotate"];
    NSUInteger copies = rot ? MAX(1, [s[@"copies"] unsignedIntegerValue]) : 1;
    NSMutableDictionary<NSString *, NSArray<id<MTLBuffer>> *> *bufs = [NSMutableDictionary dictionary];
    for (NSString *k in s[@"buffers"]) {
      NSDictionary *b = s[@"buffers"][k];
      NSData *d = b[@"file"] ? [NSData dataWithContentsOfFile:b[@"file"]] : nil;
      NSUInteger n = MAX([b[@"bytes"] unsignedIntegerValue], d.length);
      NSMutableArray *arr = [NSMutableArray array];
      for (NSUInteger c = 0; c < ([k isEqual:rot] ? copies : 1); ++c) {
        id<MTLBuffer> mb = [dev newBufferWithLength:n + 256 options:MTLResourceStorageModeShared];
        memset(mb.contents, 0, n + 256);
        if (d) memcpy(mb.contents, d.bytes, d.length);
        [arr addObject:mb];
      }
      bufs[k] = arr;
    }
    void (^encode)(id<MTLComputeCommandEncoder>, NSUInteger) = ^(id<MTLComputeCommandEncoder> en, NSUInteger c) {
      [en setComputePipelineState:p];
      for (NSString *k in bufs) {
        NSArray *arr = bufs[k];
        [en setBuffer:arr[c % arr.count] offset:0 atIndex:(NSUInteger)k.integerValue];
      }
      [en dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    };
    // correctness: each arm once on copy 0, buffers restored from the files between arms
    for (NSUInteger ai = 0; ai < pipes.count; ++ai) {
      for (NSString *k in s[@"buffers"]) {
        NSDictionary *b = s[@"buffers"][k];
        id<MTLBuffer> mb = bufs[k][0];
        memset(mb.contents, 0, mb.length);
        if (b[@"file"]) { NSData *d = [NSData dataWithContentsOfFile:b[@"file"]]; memcpy(mb.contents, d.bytes, d.length); }
      }
      p = pipes[ai];
      id<MTLCommandBuffer> cb = g17_gpu_cb(q);
      id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
      encode(en, 0);
      [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
      if (cb.error) { fprintf(stderr, "cb error: %s\n", cb.error.description.UTF8String); return 3; }
      for (NSString *k in arms[ai][@"dump"]) {
        id<MTLBuffer> mb = bufs[k][0];
        [[NSData dataWithBytes:mb.contents length:mb.length - 256] writeToFile:arms[ai][@"dump"][k] atomically:YES];
      }
    }
    NSUInteger chains = [s[@"chains"] unsignedIntegerValue], n = [s[@"n"] unsignedIntegerValue];
    double warm = [s[@"warm_s"] doubleValue] ?: 0.5;
    double t0 = CFAbsoluteTimeGetCurrent();
    NSUInteger wi = 0;
    while (CFAbsoluteTimeGetCurrent() - t0 < warm) {          // warm the clock on the arms themselves
      p = pipes[wi++ % pipes.count];
      id<MTLCommandBuffer> tc = g17_gpu_cb(q);
      id<MTLComputeCommandEncoder> te = [tc computeCommandEncoder];
      for (NSUInteger i = 0; i < n; ++i) encode(te, i + 1);
      [te endEncoding]; [tc commit]; [tc waitUntilCompleted];
    }
    NSMutableArray *per = [NSMutableArray array];
    for (NSUInteger ai = 0; ai < pipes.count; ++ai) [per addObject:[NSMutableArray array]];
    for (NSUInteger ch = 0; ch < chains; ++ch) {
      for (NSUInteger j = 0; j < pipes.count; ++j) {          // ABBA: the order flips every chain
        NSUInteger ai = (ch % 2) ? pipes.count - 1 - j : j;
        p = pipes[ai];
        id<MTLCommandBuffer> tc = g17_gpu_cb(q);
        id<MTLComputeCommandEncoder> te = [tc computeCommandEncoder];
        for (NSUInteger i = 0; i < n; ++i) encode(te, i + 1);
        [te endEncoding]; [tc commit]; [tc waitUntilCompleted];
        [per[ai] addObject:@((tc.GPUEndTime - tc.GPUStartTime) * 1e6 / n)];
      }
    }
    NSDictionary *out = @{@"per_dispatch_us": per};
    [[NSJSONSerialization dataWithJSONObject:out options:0 error:nil] writeToFile:@(argv[2]) atomically:YES];
  }
  return 0;
}
