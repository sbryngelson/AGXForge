// Persistent, bounded scalar/half worker. No runtime compilation.
// Build: clang -fobjc-arc -O2 -Wall -Wextra -Werror -framework Foundation -framework Metal
//        -o tools/g17scanworker tools/g17scanworker.m
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "g17scanstorage.h"
#include "g17gpulock.h"

static int fail(const char *where, NSError *error) {
  fprintf(stderr, "phase=%s error=%s\n", where,
          error ? error.description.UTF8String : "invalid state");
  fflush(stderr);
  return 1;
}

static BOOL readAll(void *buffer, size_t count) {
  return fread(buffer, 1, count, stdin) == count;
}

static BOOL reply(NSDictionary *header, const void *data, size_t count) {
  NSData *json = [NSJSONSerialization dataWithJSONObject:header options:0 error:nil];
  uint32_t length = (uint32_t)json.length;
  return fwrite(&length, 4, 1, stdout) == 1 &&
         fwrite(json.bytes, 1, length, stdout) == length &&
         (!count || fwrite(data, 1, count, stdout) == count) && fflush(stdout) == 0;
}

int main(int argc, const char **argv) {
  @autoreleasepool {
    BOOL describe = argc == 3 && !strcmp(argv[2], "--describe-layout");
    BOOL fullLoad = argc == 3 && !strcmp(argv[2], "--half-full-load-approved");
    BOOL fullDispatch = argc == 5 && !strcmp(argv[4], "--half-full-validation-approved");
    BOOL wordControl = argc == 5 && !strcmp(argv[4], "--half-word-control-approved");
    BOOL loadOnly = fullLoad || (argc == 3 && !strcmp(argv[2], "--half-load-approved"));
    BOOL halfApproved = wordControl || fullDispatch || (argc == 5 && !strcmp(argv[4], "--half-validation-approved"));
    BOOL scalarApproved = argc == 5 && !strcmp(argv[4], "--dispatch-approved");
    if (!describe && !loadOnly && !halfApproved && !scalarApproved) {
      fprintf(stderr, "usage: g17scanworker BUNDLE PACKED_INPUT MAX_QUERIES "
              "--dispatch-approved|--half-validation-approved\n"
              "       g17scanworker BUNDLE --describe-layout|--half-load-approved\n");
      return 2;
    }
    char *end = NULL;
    unsigned long limit = 0;
    if (!describe && !loadOnly) {
      limit = strtoul(argv[3], &end, 10);
      if (!end || *end || limit < 1 || limit > 100) return fail("query_limit", nil);
    }
    NSString *dir = [NSString stringWithUTF8String:argv[1]];
    NSData *manifest = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"manifest.json"]];
    NSError *error = nil;
    if (!manifest) return fail("manifest", nil);
    NSDictionary *m = [NSJSONSerialization JSONObjectWithData:manifest options:0 error:&error];
    if (![m isKindOfClass:[NSDictionary class]]) return fail("manifest", error);
    BOOL half = [m[@"profile"] isKindOfClass:[NSString class]];
    NSDictionary *shape = half ? m[@"shape"] : m;
    if (![shape isKindOfClass:[NSDictionary class]] ||
        ![shape[@"rows"] isKindOfClass:[NSNumber class]] ||
        ![shape[@"columns"] isKindOfClass:[NSNumber class]]) return fail("shape", nil);
    NSUInteger rows = [shape[@"rows"] unsignedIntegerValue];
    NSUInteger columns = [shape[@"columns"] unsignedIntegerValue];
    BOOL controlImage = [m[@"experiment"] isEqual:@"halfword-preservation-word-control-v1"];
    if (wordControl && (!controlImage || rows != 1 || columns != 1 || limit != 1))
      return fail("word_control_contract", nil);
    if (controlImage && !wordControl && !loadOnly && !describe)
      return fail("word_control_requires_separate_mode", nil);
    if (half) {
      NSString *profile = [NSString stringWithFormat:@"half-buffer-two-bindings-%lux%lu",
                           (unsigned long)rows, (unsigned long)columns];
      if (![m[@"profile"] isEqual:profile] || (!describe && !halfApproved && !loadOnly))
        return fail("half_validation_approval", nil);
    } else if (![m[@"profile"] isKindOfClass:[NSDictionary class]] ||
               ![m[@"format"] isEqual:@"g17-packed-scan-v1"] ||
               ![m[@"profile"][@"name"] isEqual:@"scalar-buffer-two-bindings-measured-v2"] ||
               halfApproved || loadOnly) return fail("profile", nil);
    // Full execution is a separate stage, limited to the specified half target.
    BOOL full = half && rows == 500000 && columns == 384 && (fullLoad || fullDispatch || describe);
    if (!rows || (!full && rows > 128) || !columns || columns > 384) return fail("small_fixture_limit", nil);
    G17Storage storage;
    if (!g17StorageInit(&storage, rows, columns, half)) return fail("storage_layout", nil);
    NSString *functionName = half ? @"half_scan" : @"packed_scan";
    if (describe) {
      NSDictionary *layout = @{@"function": functionName, @"storage_dtype": half ? @"float16" : @"float32",
        @"matrix_bytes": @(storage.matrixBytes), @"query_bytes": @(storage.queryBytes),
        @"input_bytes": @(storage.inputBytes), @"reply_bytes": @(storage.replyBytes),
        @"output_bytes": @(storage.outputBytes), @"gpu_executed": @NO};
      NSData *json = [NSJSONSerialization dataWithJSONObject:layout options:0 error:nil];
      fwrite(json.bytes, 1, json.length, stdout);
      return 0;
    }
    FILE *input = loadOnly ? NULL : fopen(argv[2], "rb");
    NSUInteger inputBytes = storage.inputBytes;
    if (!loadOnly && (!input || fseek(input, 0, SEEK_END) || ftell(input) != (long)inputBytes ||
        fseek(input, 0, SEEK_SET))) return fail("input_size", nil);

    fprintf(stderr, "phase=create_pipeline\n"); fflush(stderr);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fail("device", nil);
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:
      [dir stringByAppendingPathComponent:@"scan.lib.metallib"]] error:&error];
    if (!library) return fail("library", error);
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    if (!function) return fail("function", nil);
    MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
    ad.url = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"scan.arc.metallib"]];
    id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:ad error:&error];
    if (!archive) return fail("archive", error);
    MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = function;
    pd.binaryArchives = @[archive];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pd
      options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&error];
    if (!pipeline) return fail("pipeline", error);
    if (loadOnly) {
      puts("{\"status\":0,\"load_only\":true,\"gpu_dispatched\":false}");
      return 0;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> packed = [device newBufferWithLength:inputBytes
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:storage.outputBytes
      options:MTLResourceStorageModeShared];
    if (!queue || !packed || !output) return fail("allocation", nil);
    if (fread(packed.contents, 1, inputBytes, input) != inputBytes) return fail("input_read", nil);
    fclose(input);
    if (!g17StorageFinite(&storage, packed.contents, inputBytes / storage.elementBytes))
      return fail("nonfinite_input", nil);
    void *words = output.contents;
    void *query = (uint8_t *)packed.contents + storage.matrixBytes;
    NSUInteger group = MIN(rows, MIN((NSUInteger)32, pipeline.maxTotalThreadsPerThreadgroup));
    if (!group) return fail("threadgroup_limit", nil);
    NSDictionary *identity = @{
      @"pipeline": @((uintptr_t)(__bridge void *)pipeline),
      @"packed": @((uintptr_t)packed.contents), @"output": @((uintptr_t)output.contents),
      @"matrix_bytes": @(storage.matrixBytes), @"storage_dtype": half ? @"float16" : @"float32",
      @"input_bytes": @(storage.inputBytes), @"output_bytes": @(storage.outputBytes),
      @"pipeline_builds": @1, @"matrix_uploads": @1, @"buffer_allocations": @2
    };
    if (!reply(@{@"protocol": @1, @"sequence": @0, @"bytes": @0,
                 @"rows": @(rows), @"columns": @(columns), @"identity": identity}, NULL, 0))
      return fail("handshake", nil);

    for (NSUInteger sequence = 1; sequence <= limit; ++sequence) {
      @autoreleasepool {
        uint32_t size;
        size_t got = fread(&size, 1, 4, stdin);
        if (!got && feof(stdin)) return 0;
        if (got != 4 || size != storage.queryBytes || !readAll(query, size))
          return fail("query_frame", nil);
        if (!g17StorageFinite(&storage, query, columns)) return fail("nonfinite_query", nil);
        // Reset every marker/guard so an earlier query cannot establish completion.
        g17StorageReset(&storage, words);
        id<MTLCommandBuffer> cb = g17_gpu_cb(queue);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) return fail("command", nil);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:packed offset:0 atIndex:1];
        [enc setBuffer:output offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
        [enc endEncoding];
        fprintf(stderr, "phase=submitting sequence=%lu\n", (unsigned long)sequence); fflush(stderr);
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted || cb.error) return fail("command_status", cb.error);
        const char *invalid = g17StorageCheck(&storage, words);
        if (wordControl) {
          // This experiment deliberately overwrites bytes 2..3 inside the 130-byte
          // allocation. Only that guard halfword may change. A failed command
          // buffer was already refused above, before any observation is returned.
          uint32_t value; memcpy(&value, words, sizeof(value));
          BOOL tailGuard = YES;
          for (NSUInteger i = 4; i < storage.outputBytes; i += 2) {
            uint16_t guard; memcpy(&guard, (uint8_t *)words + i, 2);
            if (guard != 0xa55a) tailGuard = NO;
          }
          if (!tailGuard || !invalid || strcmp(invalid, "boundary_guard"))
            return fail("word_control_expected_guard_change", nil);
          if (!reply(@{@"protocol": @1, @"sequence": @(sequence), @"bytes": @4,
                       @"status": @0, @"boundary_guard": @NO, @"tail_guard": @YES,
                       @"expected_control": @YES, @"word_bits": @(value),
                       @"adjacent_halfword": @(value >> 16), @"identity": identity}, words, 4))
            return fail("reply", nil);
          continue;
        }
        if (invalid) return fail(invalid, nil);
        if (!reply(@{@"protocol": @1, @"sequence": @(sequence), @"bytes": @(storage.replyBytes),
                     @"status": @0, @"boundary_guard": @YES, @"identity": identity,
                     @"gpu_seconds": @(cb.GPUEndTime - cb.GPUStartTime)}, words, storage.replyBytes))
          return fail("reply", nil);
      }
    }
  }
  return 0;
}
