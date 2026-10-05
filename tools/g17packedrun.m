// One bounded dispatch of a prepared native scan. Build does not touch the GPU:
// clang -fobjc-arc -O2 -framework Foundation -framework Metal -o tools/g17packedrun tools/g17packedrun.m
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "g17gpulock.h"

static int fail(NSString *where, NSError *error) {
  fprintf(stderr, "%s: %s\n", where.UTF8String, error ? error.description.UTF8String : "failed");
  return 1;
}

static void stage(const char *phase) {
  fprintf(stderr, "phase=%s\n", phase);
  fflush(stderr);
}

int main(int argc, const char **argv) {
  @autoreleasepool {
    BOOL loadOnly = argc == 3 && !strcmp(argv[2], "--load-only");
    if (argc != 3 || (!loadOnly && strcmp(argv[2], "--dispatch-approved"))) {
      fprintf(stderr, "usage: g17packedrun BUNDLE --load-only|--dispatch-approved (dispatch requires Spencer's explicit GPU OK)\n");
      return 2;
    }
    NSString *dir = [NSString stringWithUTF8String:argv[1]];
    NSError *error = nil;
    NSData *manifest = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"manifest.json"]];
    if (!manifest) return fail(@"manifest", nil);
    NSDictionary *m = [NSJSONSerialization JSONObjectWithData:manifest options:0 error:&error];
    if (![m isKindOfClass:[NSDictionary class]] || ![m[@"format"] isEqual:@"g17-packed-scan-v1"])
      return fail(@"manifest format", error);
    NSUInteger rows = [m[@"rows"] unsignedIntegerValue], columns = [m[@"columns"] unsignedIntegerValue];
    // First validation only: one dispatch, no loops, and at most 128 rows.
    if (!rows || rows > 128 || !columns || columns > 384) return fail(@"initial validation dimensions", nil);
    NSData *input = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"packed.f32"]];
    if (input.length != (rows + 1) * columns * sizeof(float)) return fail(@"packed input size", nil);

    stage("create_device");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fail(@"device", nil);
    NSURL *libURL = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"scan.lib.metallib"]];
    stage("load_library");
    id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
    if (!library) return fail(@"library", error);
    id<MTLFunction> function = [library newFunctionWithName:@"packed_scan"];
    if (!function) return fail(@"function", nil);
    MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
    ad.url = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"scan.arc.metallib"]];
    stage("open_archive");
    id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:ad error:&error];
    if (!archive) return fail(@"archive", error);
    MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = function;
    pd.binaryArchives = @[archive];
    stage("create_pipeline");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pd
      options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&error];
    if (!pipeline) return fail(@"pipeline from native archive", error);
    if (loadOnly) {
      printf("{\"status\":0,\"load_only\":true,\"gpu_dispatched\":false}\n");
      return 0; // No buffers, command queue, encoder, or command buffer exist yet.
    }

    stage("allocate_buffers");
    id<MTLBuffer> packed = [device newBufferWithBytes:input.bytes length:input.length options:MTLResourceStorageModeShared];
    NSUInteger count = 2 * rows, guards = 32;
    id<MTLBuffer> output = [device newBufferWithLength:(count + guards) * 4 options:MTLResourceStorageModeShared];
    if (!packed || !output) return fail(@"buffer allocation", nil);
    uint32_t *words = output.contents;
    for (NSUInteger i = 0; i < count + guards; ++i) words[i] = 0xDEADBEEF;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cb = g17_gpu_cb(queue);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!queue || !cb || !enc) return fail(@"command allocation", nil);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:packed offset:0 atIndex:1];
    [enc setBuffer:output offset:0 atIndex:2];
    NSUInteger group = MIN(rows, MIN((NSUInteger)32, pipeline.maxTotalThreadsPerThreadgroup));
    if (!group) return fail(@"threadgroup limit", nil);
    [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    [enc endEncoding];
    stage("submitting");
    [cb commit];
    stage("submitted");
    [cb waitUntilCompleted];
    stage("command_finished");
    if (cb.status != MTLCommandBufferStatusCompleted || cb.error) return fail(@"command buffer", cb.error);
    for (NSUInteger i = rows; i < count; ++i)
      if (words[i] != 0x5A17C0DE) return fail(@"row completion marker", nil);
    for (NSUInteger i = count; i < count + guards; ++i)
      if (words[i] != 0xDEADBEEF) return fail(@"output boundary guard", nil);
    NSData *result = [NSData dataWithBytes:words length:count * 4];
    if (![result writeToFile:[dir stringByAppendingPathComponent:@"output.f32"] options:NSDataWritingAtomic error:&error])
      return fail(@"result file", error);
    printf("{\"status\":0,\"rows\":%lu,\"gpu_seconds\":%.9f,\"completion_markers\":true,\"boundary_guard\":true}\n",
           (unsigned long)rows, cb.GPUEndTime - cb.GPUStartTime);
    return 0;
  }
}
