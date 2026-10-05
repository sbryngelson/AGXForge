// g17bundlerun: run each bundle of a plan ONCE (or `rounds` times) and write its output buffer to <outdir>/<tag>.out.
// The verification runner of tools/g17deliver.py: no timing, no warm-up.
//
// A bundle is a directory holding scan.lib.metallib + scan.arc.metallib (the program), manifest.json (its entry
// name), and the three inputs a.f16, b.f16, c.f32. Each input is placed at byte 128 of a buffer pre-filled with
// 0xA5 (so a read before or past it is visible), buffer 3 is re-loaded from c.f32 before every round, and the
// output is buffer 3's c.f32-sized window after the dispatch.
//
// Binding (`base` in the plan, default 1): base 1 binds a, b, c at 1, 2, 3 (the ordinary class); base 0 binds
// c, a, b at 0, 1, 2 (the cooperative three-binding class: slot 0 written). Every round after the first must give
// the same bytes; a difference is reported and the exit status is 4.
//
// Command buffers come from g17_gpu_cb (tools/g17gpulock.h), so a run takes the machine GPU lock.
//
//   g17bundlerun plan.json outdir        plan = {"configs": [{tag, bundle, threads, group, base?, rounds?}]}
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "g17gpulock.h"

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

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 3) { fprintf(stderr, "usage: g17bundlerun plan.json outdir\n"); return 2; }
    NSString *outdir = @(argv[2]);
    NSDictionary *plan = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[1])] options:0 error:nil];
    if (!plan) { fprintf(stderr, "plan unreadable\n"); return 2; }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue];
    int status = 0;
    for (NSDictionary *c in plan[@"configs"]) {
      NSString *dir = c[@"bundle"], *tag = c[@"tag"];
      NSData *a = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"a.f16"]];
      NSData *b = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"b.f16"]];
      NSData *c0 = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"c.f32"]];
      if (!a || !b || !c0) { fprintf(stderr, "inputs for %s\n", tag.UTF8String); return 2; }
      id<MTLComputePipelineState> p = bundlePipeline(dev, dir);
      id<MTLBuffer> ba = filled(dev, a), bb = filled(dev, b), bc = filled(dev, c0);
      NSUInteger base = [(c[@"base"] ?: @1) unsignedIntegerValue], threads = [c[@"threads"] unsignedIntegerValue];
      NSUInteger group = [c[@"group"] unsignedIntegerValue], rounds = [(c[@"rounds"] ?: @1) unsignedIntegerValue];
      NSArray *bufs = base == 0 ? @[bc, ba, bb] : @[ba, bb, bc];
      NSData *first = nil;
      for (NSUInteger r = 0; r < rounds; ++r) {
        memcpy((uint8_t *)bc.contents + 128, c0.bytes, c0.length);
        id<MTLCommandBuffer> cb = g17_gpu_cb(q);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p];
        for (NSUInteger i = 0; i < 3; ++i) [enc setBuffer:bufs[i] offset:128 atIndex:base + i];
        [enc dispatchThreadgroups:MTLSizeMake(threads / group, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted || cb.error) {
          fprintf(stderr, "command failed for %s: %s\n", tag.UTF8String, cb.error.description.UTF8String); return 3;
        }
        // one line per round on stdout: the command buffer's GPU time (MM 25.144.1's timing); readers of the
        // outputs ignore stdout
        printf("time %s %lu %.3f\n", tag.UTF8String, (unsigned long)r, (cb.GPUEndTime - cb.GPUStartTime) * 1e6);
        NSData *got = [NSData dataWithBytes:(uint8_t *)bc.contents + 128 length:c0.length];
        if (!first) {
          first = got;
          [got writeToFile:[outdir stringByAppendingPathComponent:[tag stringByAppendingString:@".out"]] atomically:YES];
        } else if (![got isEqualToData:first]) {
          fprintf(stderr, "OUTPUT DIFFERS FROM THE FIRST: %s round %lu\n", tag.UTF8String, (unsigned long)r); status = 4;
        }
      }
    }
    return status;
  }
}
