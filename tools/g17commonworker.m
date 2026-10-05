// One native runtime for ABI-v3 FP16 programs and retained v2 images. No runtime compilation.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>
#include "g17scanstorage.h"
#include "g17gpulock.h"

static int fail(const char *phase, NSError *error) {
  fprintf(stderr, "phase=%s error=%s\n", phase,
          error ? error.description.UTF8String : "invalid state");
  fflush(stderr);
  return 1;
}

static BOOL integer(id value, NSUInteger maximum, NSUInteger *out) {
  if (![value isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID()) return NO;
  const char *type = [value objCType];
  if (!strchr("cCsSiIlLqQ", type[0]) || [value longLongValue] < 0 ||
      [value unsignedLongLongValue] > maximum) return NO;
  *out = [value unsignedIntegerValue];
  return YES;
}

static BOOL boolean(id value, BOOL expected) {
  return value && CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID() &&
         [value boolValue] == expected;
}

static BOOL reply(NSDictionary *header, const void *data, size_t count) {
  NSData *json = [NSJSONSerialization dataWithJSONObject:header options:0 error:nil];
  uint32_t length = (uint32_t)json.length;
  return json && length <= 4096 && fwrite(&length, 4, 1, stdout) == 1 &&
         fwrite(json.bytes, 1, length, stdout) == length &&
         (!count || fwrite(data, 1, count, stdout) == count) && !fflush(stdout);
}

static NSData *boundedInput(NSString *path, NSUInteger length) {
  NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:path error:nil];
  if (!attributes || [attributes fileSize] != length) return nil;
  NSData *data = [NSData dataWithContentsOfFile:path];
  return data.length == length ? data : nil;
}

// Called only after common image admission and pipeline loading.
static int runFP32(id<MTLDevice> device, id<MTLComputePipelineState> pipeline,
    NSString *inputsDir, const G17LayerNormStorage *s, NSArray *bindings,
    const NSUInteger indices[4], NSUInteger limit, BOOL queryProjection) {
  NSMutableData *source = [boundedInput([inputsDir stringByAppendingPathComponent:@"source.f32"],
                                         s->payloadBytes[0]) mutableCopy];
  NSData *gamma = boundedInput([inputsDir stringByAppendingPathComponent:
      queryProjection ? @"weight.f32" : @"gamma.f32"], s->payloadBytes[1]);
  NSData *beta = boundedInput([inputsDir stringByAppendingPathComponent:
      queryProjection ? @"bias.f32" : @"beta.f32"], s->payloadBytes[2]);
  if (!source || !gamma || !beta) return fail("input_size", nil);
  const void *snapshots[3] = {source.bytes, gamma.bytes, beta.bytes};
  const G17Storage f32 = {.half=false};
  for (NSUInteger i=0; i<3; ++i)
    if (!g17StorageFinite(&f32, snapshots[i], s->payloadBytes[i]/4)) return fail("nonfinite_input", nil);
  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (!queue) return fail("queue", nil);
  NSMutableArray<id<MTLBuffer>> *buffers = [NSMutableArray array];
  NSMutableArray *addresses = [NSMutableArray array];
  void *allocations[4];
  const void *views[4];
  for (NSUInteger i=0; i<4; ++i) {
    id<MTLBuffer> buffer = [device newBufferWithLength:s->allocationBytes[i]
                                                options:MTLResourceStorageModeShared];
    if (!buffer) return fail("allocation", nil);
    [buffers addObject:buffer];
    allocations[i] = buffer.contents; views[i] = buffer.contents;
    [addresses addObject:@((uintptr_t)buffer.contents)];
  }
  const char *invalid = g17LayerNormPrepare(s, allocations, snapshots);
  if (invalid) return fail(invalid, nil);
  NSDictionary *identity = @{@"pipeline": @((uintptr_t)(__bridge void *)pipeline),
    @"buffers": addresses, @"matrix_bytes": @(s->payloadBytes[0]), @"storage_dtype": @"float32",
    @"request_bytes": @(s->payloadBytes[0]), @"reply_bytes": @(s->payloadBytes[3]),
    @"reply_elements": @(s->rows*s->columns), @"completion_markers": @NO,
    @"output_bytes": @(s->allocationBytes[3]), @"pipeline_builds": @1, @"matrix_uploads": @1,
    @"parameter_uploads": @2, @"buffer_allocations": @4, @"bindings": bindings,
    @"buffer_offsets": @[@(G17_LN_GUARD), @(G17_LN_GUARD), @(G17_LN_GUARD), @(G17_LN_GUARD)],
    @"buffer_payload_bytes": @[@(s->payloadBytes[0]), @(s->payloadBytes[1]),
                               @(s->payloadBytes[2]), @(s->payloadBytes[3])],
    @"buffer_allocation_bytes": @[@(s->allocationBytes[0]), @(s->allocationBytes[1]),
                                  @(s->allocationBytes[2]), @(s->allocationBytes[3])]};
  NSUInteger launchX = queryProjection ? s->columns : s->rows;
  NSUInteger launchY = queryProjection ? s->rows : 1;
  NSUInteger group = MIN(launchX, MIN((NSUInteger)32, pipeline.maxTotalThreadsPerThreadgroup));
  if (!group || limit < 1 || limit > 16) return fail("launch_limits", nil);
  if (!reply(@{@"protocol": @2, @"sequence": @0, @"bytes": @0, @"rows": @(s->rows),
               @"columns": @(s->columns), @"identity": identity}, NULL, 0)) return fail("handshake", nil);
  for (NSUInteger sequence=1; sequence<=limit; ++sequence) {
    @autoreleasepool {
      uint32_t size;
      size_t got = fread(&size, 1, 4, stdin);
      if (!got && feof(stdin)) return 0;
      if (got != 4 || size != s->payloadBytes[0] || fread(source.mutableBytes, 1, size, stdin) != size)
        return fail("request_frame", nil);
      snapshots[0] = source.bytes;
      invalid = g17LayerNormBeginQuery(s, allocations, snapshots[0]);
      if (invalid) return fail(invalid, nil);
      id<MTLCommandBuffer> cb = g17_gpu_cb(queue);
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      if (!cb || !enc) return fail("command", nil);
      [enc setComputePipelineState:pipeline];
      for (NSUInteger i=0; i<4; ++i)
        [enc setBuffer:buffers[i] offset:G17_LN_GUARD atIndex:indices[i]];
      [enc dispatchThreads:MTLSizeMake(launchX, launchY, 1)
          threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
      [enc endEncoding];
      fprintf(stderr, "phase=submitting sequence=%lu\n", (unsigned long)sequence); fflush(stderr);
      [cb commit]; [cb waitUntilCompleted];
      if (cb.status != MTLCommandBufferStatusCompleted || cb.error) return fail("command_status", cb.error);
      size_t failedBuffer;
      invalid = g17LayerNormCheck(s, views, snapshots, &failedBuffer);
      if (invalid) {
        fprintf(stderr, "buffer=%zu\n", failedBuffer);
        return fail(invalid, nil);
      }
      if (!reply(@{@"protocol": @2, @"sequence": @(sequence), @"bytes": @(s->payloadBytes[3]),
        @"status": @0, @"boundary_guard": @YES, @"readonly_inputs": @YES, @"identity": identity,
        @"source_uploads": @(sequence+1), @"gpu_seconds": @(cb.GPUEndTime-cb.GPUStartTime)},
        (uint8_t *)allocations[3]+G17_LN_GUARD, s->payloadBytes[3])) return fail("reply", nil);
    }
  }
  return 0;
}

// THE ADMITTED TEXTURE PROFILE, and nothing wider. results/g17-texture-resource-contract-v1
// states it and this is the worker's half of the same contract: one read-only two-dimensional
// texture in a MEASURED pixel format, one mip level, rankless, at dense index 0, with no sampler
// and no second texture. Every field is checked and a contract differing in any of them is refused BY NAME -
// the worker must not fall back to tools/g17texrun.py's standalone probe, which is where a
// borrowed section would come from.
//
// THE STRIDE IS COMPUTED, NOT ACCEPTED. bytesPerRow is width * 4 and the payload must be exactly
// that times the height. A host that proposes its own stride can disagree with the image about
// what was uploaded and has no way to notice, so a proposal that differs is a refusal rather than
// an override.
#define G17_TEXTURE_BYTES_PER_TEXEL 4u

// THE ADMITTED PIXEL FORMATS ARE A TABLE, NOT A LITERAL, and it currently has ONE ENTRY. This file
// wrote MTLPixelFormatR32Uint at every descriptor and reported "R32Uint" in every identity, so the
// format was a constant three places deep while the manifest's `element` was checked against one
// string. The format is not in the image at all - a uint and a float texture read compile to the
// SAME 492-byte metadata section and, for a bitcast store, the same program bytes
// (results/g17-texture-format-v1) - so which format a bundle gets is entirely this side's
// declaration, and a declaration checked against a literal is not a measurement. Making it a table
// does not widen it: an entry needs a dispatched witness, and the second format cannot be declared
// at all yet.
typedef struct {
  NSUInteger width, height, bytesPerRow, payloadBytes, denseIndex;
  MTLPixelFormat pixelFormat;
  NSString *formatName;
  BOOL formatWitnessed;
} G17TexturePlan;

// ADMITTED AND WITNESSED ARE TWO FACTS, AND THIS TABLE NOW CARRIES BOTH. It held one entry because
// one format had a dispatched receipt, which was the right rule while nothing was assigned to run
// a second: a format admitted on the strength of a descriptor call compiling is not a measured
// format. But a format that cannot be admitted can never BE run, so "no entry without a receipt"
// would have made the first receipt unreachable. The entry is what lets a run happen; `witnessed`
// is what says a run has happened. R32Float is admitted and NOT witnessed, and every identity this
// worker reports carries that bit, so the first float receipt labels itself as the first.
static BOOL g17TexturePixelFormat(id element, MTLPixelFormat *format, NSString **name,
                                  BOOL *witnessed) {
  if ([element isEqual:@"uint32"]) {
    // results/g17-texture-query-runtime-v1 and -v2, g17-tex-op17229-runtime-v1,
    // g17-texture-8x8-runtime-v1, g17-texture-op17244-v1: sixteen one-thread dispatches.
    *format = MTLPixelFormatR32Uint; *name = @"R32Uint"; *witnessed = YES; return YES;
  }
  if ([element isEqual:@"float32"]) {
    // ADMITTED, NOT WITNESSED. The declaration exists as of the compiler owner's e455f331, merged
    // at 2c9b9f17, and g17abi.TextureResource.element now states it; the linker boundary at
    // results/g17-texture-format-linker-v1 preregisters the probe that would witness it. NOTHING
    // HAS DISPATCHED A FLOAT TEXEL. Until one does, this reports witnessed=NO and the host
    // checker refuses any claim that it is measured.
    *format = MTLPixelFormatR32Float; *name = @"R32Float"; *witnessed = NO; return YES;
  }
  *name = nil;
  *witnessed = NO;
  return NO;
}

static BOOL textureAdmitCount(NSDictionary *resources, NSUInteger requiredCount,
                         NSUInteger proposedWidth,
                         NSUInteger proposedHeight, NSUInteger proposedPayload,
                         NSUInteger proposedIndex, G17TexturePlan *plan, const char **why) {
  *why = "texture_contract";
  if (![resources isKindOfClass:[NSDictionary class]]) { *why = "texture_metadata_missing"; return NO; }
  NSArray *textures = resources[@"textures"];
  if (![textures isKindOfClass:[NSArray class]] || textures.count == 0) {
    *why = "texture_metadata_missing"; return NO;
  }
  if (textures.count != requiredCount) { *why = "texture_second_texture"; return NO; }
  if (proposedIndex >= requiredCount) { *why = "texture_wrong_dense_index"; return NO; }
  NSArray *samplers = resources[@"samplers"];
  if (![samplers isKindOfClass:[NSArray class]]) { *why = "texture_samplers_unstated"; return NO; }
  if (samplers.count != 0) { *why = "texture_sampler_declared"; return NO; }
  for (NSUInteger ordinal = 0; ordinal < requiredCount; ++ordinal) {
    NSDictionary *texture = textures[ordinal];
    if (![texture isKindOfClass:[NSDictionary class]]) { *why = "texture_descriptor_malformed"; return NO; }
    if (![texture[@"dimension"] isEqual:@"2d"]) { *why = "texture_not_2d"; return NO; }
    // THE REFUSAL NAME CHANGED WITH THE CHECK. It was `texture_element_not_uint32`, which stops being
    // true the moment a second element is admitted; a receipt naming a rule the worker no longer
    // applies is worse than no name at all. results/g17-texture-worker-v1 keeps the old name as the
    // historical record of when that WAS the rule.
    MTLPixelFormat pixelFormat = MTLPixelFormatInvalid;
    NSString *formatName = nil;
    BOOL formatWitnessed = NO;
    if (!g17TexturePixelFormat(texture[@"element"], &pixelFormat, &formatName, &formatWitnessed)) {
      *why = "texture_element_unmeasured"; return NO;
    }
    if (![texture[@"access"] isEqual:@"read"]) { *why = "texture_not_read_only"; return NO; }
    if (texture[@"rank"] && texture[@"rank"] != [NSNull null]) { *why = "texture_ranked"; return NO; }
    NSUInteger dense;
    if (!integer(texture[@"dense_index"], requiredCount - 1, &dense)) { *why = "texture_dense_index"; return NO; }
    if (dense != ordinal) { *why = "texture_wrong_dense_index"; return NO; }
    if (!proposedWidth || !proposedHeight) { *why = "texture_extent"; return NO; }
    NSUInteger bytesPerRow = proposedWidth * G17_TEXTURE_BYTES_PER_TEXEL;
    if (bytesPerRow / G17_TEXTURE_BYTES_PER_TEXEL != proposedWidth) { *why = "texture_extent"; return NO; }
    NSUInteger wanted = bytesPerRow * proposedHeight;
    if (proposedHeight && wanted / proposedHeight != bytesPerRow) { *why = "texture_extent"; return NO; }
    if (proposedPayload != wanted) { *why = "texture_wrong_payload_length"; return NO; }
    if (dense == proposedIndex) {
      plan->width = proposedWidth;
      plan->height = proposedHeight;
      plan->bytesPerRow = bytesPerRow;
      plan->payloadBytes = wanted;
      plan->denseIndex = dense;
      plan->pixelFormat = pixelFormat;
      plan->formatName = formatName;
      plan->formatWitnessed = formatWitnessed;
    }
  }
  *why = NULL;
  return YES;
}

// Legacy callers remain single-texture. The pair admission entry is host-only until
// the measured image and explicit public-index mapping reach the runtime path.
static BOOL textureAdmit(NSDictionary *resources, NSUInteger proposedWidth,
                         NSUInteger proposedHeight, NSUInteger proposedPayload,
                         NSUInteger proposedIndex, G17TexturePlan *plan, const char **why) {
  return textureAdmitCount(resources, 1, proposedWidth, proposedHeight,
                           proposedPayload, proposedIndex, plan, why);
}

// Host payload declarations are separate from the compiler's resource facts.
// The pair campaign explicitly binds dense 0/1 at public Metal indices 0/1;
// other mappings remain refused until their image class is established.
static NSArray *textureInputs(NSString *dir, NSDictionary *manifest,
                               G17TexturePlan plans[2], NSMutableArray *payloads,
                               const char **why) {
  *why = "texture_inputs";
  NSDictionary *abi = manifest[@"abi"];
  if (![abi isKindOfClass:[NSDictionary class]]) return nil;
  NSDictionary *resources = abi[@"resources"];
  if (![resources isKindOfClass:[NSDictionary class]]) return nil;
  NSArray *textures = resources[@"textures"];
  if (![textures isKindOfClass:[NSArray class]] || textures.count < 1 || textures.count > 2)
    return nil;
  if (textures.count == 2) {
    NSDictionary *options = manifest[@"authoring_options"];
    if (![options isKindOfClass:[NSDictionary class]] ||
        ![options[@"texture_public_indices"] isEqual:@[@0, @1]]) return nil;
    NSArray *bindings = abi[@"bindings"];
    if (![bindings isKindOfClass:[NSArray class]] || bindings.count != 1 ||
        ![bindings[0] isKindOfClass:[NSDictionary class]]) return nil;
    NSDictionary *binding = bindings[0];
    NSUInteger index, offset, bytes;
    if (!integer(binding[@"index"], 0, &index) ||
        !integer(binding[@"offset"], 4, &offset) || offset != 4 ||
        !integer(binding[@"element_bytes"], 4, &bytes) || bytes != 4 ||
        ![binding[@"element_type"] isEqual:@"uint"] || !boolean(binding[@"written"], YES))
      return nil;
    for (id descriptor in textures)
      if (![descriptor isKindOfClass:[NSDictionary class]] ||
          ![descriptor[@"element"] isEqual:@"float32"]) return nil;
  }
  NSArray *inputs = manifest[@"texture_inputs"];
  if (!inputs && textures.count == 1) {
    NSDictionary *extent = manifest[@"texture_extent"];
    if (![extent isKindOfClass:[NSDictionary class]]) return nil;
    inputs = @[@{@"dense_index": @0, @"public_index": @0, @"payload": @"texture.bin",
                  @"width": extent[@"width"] ?: [NSNull null],
                  @"height": extent[@"height"] ?: [NSNull null]}];
  }
  if (![inputs isKindOfClass:[NSArray class]] || inputs.count != textures.count) return nil;
  NSMutableArray *identities = [NSMutableArray array];
  NSMutableSet *paths = [NSMutableSet set];
  for (NSUInteger i = 0; i < inputs.count; ++i) {
    NSDictionary *input = inputs[i];
    NSUInteger dense, publicIndex, width, height;
    if (![input isKindOfClass:[NSDictionary class]] ||
        !integer(input[@"dense_index"], 1, &dense) || dense != i ||
        !integer(input[@"public_index"], 1, &publicIndex) || publicIndex != dense ||
        !integer(input[@"width"], 16384, &width) ||
        !integer(input[@"height"], 16384, &height)) return nil;
    NSString *name = input[@"payload"];
    if (![name isKindOfClass:[NSString class]] || !name.length ||
        ![name.lastPathComponent isEqual:name] || [name isEqual:@"."] ||
        [name isEqual:@".."] || [paths containsObject:name]) return nil;
    [paths addObject:name];
    NSString *path = [dir stringByAppendingPathComponent:name];
    NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:path error:nil];
    if (![attributes[NSFileType] isEqual:NSFileTypeRegular] ||
        ![attributes fileSize] || [attributes fileSize] > 16*1024*1024) {
      *why = "texture_payload_missing"; return nil;
    }
    NSData *payload = [NSData dataWithContentsOfFile:path];
    if (!payload) { *why = "texture_payload_missing"; return nil; }
    if (!textureAdmitCount(resources, textures.count, width, height, payload.length,
                            dense, &plans[i], why)) return nil;
    [payloads addObject:payload];
    [identities addObject:@{@"dense_index": @(dense), @"public_index": @(publicIndex),
                           @"payload": name, @"payload_bytes": @(payload.length),
                           @"width": @(width), @"height": @(height),
                           @"bytes_per_row": @(plans[i].bytesPerRow),
                           @"pixel_format": plans[i].formatName}];
  }
  *why = NULL;
  return identities;
}

// Shared upload path for load-only and dispatch. Each texture is created from its
// own admitted plan; callers retain the object and original bytes for readback.
static id<MTLTexture> textureCreate(id<MTLDevice> device, const G17TexturePlan *plan,
                                    NSData *payload, const char **why) {
  *why = "texture_descriptor";
  MTLTextureDescriptor *descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:plan->pixelFormat
                                   width:plan->width height:plan->height mipmapped:NO];
  if (!descriptor) return nil;
  descriptor.usage = MTLTextureUsageShaderRead;
  descriptor.mipmapLevelCount = 1;
  descriptor.storageMode = MTLStorageModeShared;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (!texture) return nil;
  if (payload.length != plan->payloadBytes) { *why = "texture_wrong_payload_length"; return nil; }
  [texture replaceRegion:MTLRegionMake2D(0, 0, plan->width, plan->height)
             mipmapLevel:0 withBytes:payload.bytes bytesPerRow:plan->bytesPerRow];
  *why = NULL;
  return texture;
}

static BOOL textureUnchanged(id<MTLTexture> texture, const G17TexturePlan *plan,
                              NSData *payload) {
  NSMutableData *observed = [NSMutableData dataWithLength:plan->payloadBytes];
  if (!observed) return NO;
  [texture getBytes:observed.mutableBytes bytesPerRow:plan->bytesPerRow
        fromRegion:MTLRegionMake2D(0, 0, plan->width, plan->height) mipmapLevel:0];
  return [observed isEqualToData:payload];
}

// CREATE, UPLOAD, BIND, REPORT - and commit nothing. Binding needs an encoder,
// which is ended without submitting its command buffer.
static NSDictionary *textureBind(id<MTLDevice> device, id<MTLComputePipelineState> pipeline,
                                 const G17TexturePlan *plan, NSData *payload, const char **why) {
  id<MTLTexture> texture = textureCreate(device, plan, payload, why);
  if (!texture) return nil;
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> buffer = g17_gpu_cb(queue);
  id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
  if (!queue || !buffer || !encoder) { *why = "texture_encoder"; return nil; }
  [encoder setComputePipelineState:pipeline];
  [encoder setTexture:texture atIndex:plan->denseIndex];
  [encoder endEncoding];
  *why = NULL;
  return @{@"pixel_format": plan->formatName, @"format_witnessed": @(plan->formatWitnessed),
           @"usage": @"shader_read", @"mip_levels": @1,
           @"dimension": @"2d", @"width": @(plan->width), @"height": @(plan->height),
           @"bytes_per_row": @(plan->bytesPerRow), @"payload_bytes": @(plan->payloadBytes),
           @"dense_index": @(plan->denseIndex), @"texels": @(plan->width * plan->height),
           @"bytes_per_texel": @(G17_TEXTURE_BYTES_PER_TEXEL),
           @"bound": @YES, @"committed": @NO, @"gpu_dispatched": @NO};
}

// The texture path is deliberately separate from the scalar storage paths below.  Its ABI is
// resource-shaped (version 6/7 and a texture descriptor), so making it pass through the buffer
// binding, shape and prologue checks would either reject a valid texture contract or silently
// supply buffer assumptions.  This helper is load-only: the descriptor is created and bound on
// an encoder that is never committed, and dispatch is refused by main until a texture image has
// a result-specific execution protocol.
// Coordinate input storage is explicit and distinct from descriptive query metadata.
// This validator allocates no Metal object. An absent contract preserves legacy queries.
static BOOL textureQueryContract(NSDictionary *query, BOOL inputCoordinates,
                                 NSUInteger *expected, NSUInteger *slot) {
  if (![query isKindOfClass:[NSDictionary class]] ||
      !integer(query[@"expected"], UINT32_MAX, expected) ||
      !integer(query[@"read back word"], UINT32_MAX, slot) ||
      *expected == 0 || *expected == *slot) return NO;
  if (inputCoordinates) return query[@"coordinate at that thread"] == nil;
  NSArray *xy = query[@"coordinate at that thread"];
  NSUInteger x, y;
  return [xy isKindOfClass:[NSArray class]] && xy.count == 2 &&
         integer(xy[0], UINT32_MAX, &x) && integer(xy[1], UINT32_MAX, &y);
}

static NSArray *textureCoordinateInputs(NSDictionary *manifest, G17TexturePlan plans[2],
                                         NSUInteger textureCount, const char **why) {
  *why = "texture_coordinate_inputs";
  id query = manifest[@"query"];
  if (!query) { *why = NULL; return @[]; }
  if (![query isKindOfClass:[NSDictionary class]]) return nil;
  id positions = query[@"coordinate input words"], inputs = query[@"input words"];
  if (!positions && !inputs) { *why = NULL; return @[]; }
  if (textureCount != 2 || ![positions isKindOfClass:[NSArray class]] ||
      ![positions isEqual:@[@[@4, @5], @[@6, @7]]] ||
      ![inputs isKindOfClass:[NSArray class]] || [inputs count] != 4) return nil;
  for (NSUInteger i = 0; i < 4; ++i) {
    id item = inputs[i];
    NSUInteger word, value;
    if (![item isKindOfClass:[NSDictionary class]] ||
        !integer(item[@"word"], 7, &word) || word != 4 + i ||
        !integer(item[@"value"], UINT32_MAX, &value)) return nil;
    NSUInteger bound = i % 2 ? plans[i / 2].height : plans[i / 2].width;
    if (value >= bound) { *why = "texture_coordinate_bounds"; return nil; }
    for (NSString *key in @[@"guards", @"observe", @"result words"]) {
      id list = query[key];
      if (!list) continue;
      if (![list isKindOfClass:[NSArray class]]) return nil;
      for (id entry in list) {
        id index = entry;
        if ([key isEqualToString:@"result words"]) {
          if (![entry isKindOfClass:[NSDictionary class]]) return nil;
          index = entry[@"word"];
        }
        NSUInteger other;
        if (!integer(index, UINT32_MAX, &other) || other == word) return nil;
      }
    }
  }
  *why = NULL;
  return inputs;
}

static int runTextureLoadOnly(NSString *dir, NSDictionary *manifest, NSDictionary *abi) {
  for (NSString *file in @[@"scan.arc.metallib", @"scan.lib.metallib", @"scan.o", @"program.bin"]) {
    NSDictionary *attributes = [[NSFileManager defaultManager]
      attributesOfItemAtPath:[dir stringByAppendingPathComponent:file] error:nil];
    unsigned long long size = [attributes fileSize];
    if (![attributes[NSFileType] isEqual:NSFileTypeRegular] || !size || size > 16*1024*1024)
      return fail("image_files", nil);
  }
  G17TexturePlan plans[2];
  NSMutableArray *payloads = [NSMutableArray array];
  const char *why = NULL;
  NSArray *inputs = textureInputs(dir, manifest, plans, payloads, &why);
  if (!inputs) return fail(why ? why : "texture_inputs", nil);
  NSArray *coordinateInputs = textureCoordinateInputs(manifest, plans, inputs.count, &why);
  if (!coordinateInputs) return fail(why, nil);
  NSError *error = nil;
  fprintf(stderr, "phase=create_pipeline\n"); fflush(stderr);
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) return fail("device", nil);
  id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:
    [dir stringByAppendingPathComponent:@"scan.lib.metallib"]] error:&error];
  if (!library) return fail("library", error);
  id<MTLFunction> function = [library newFunctionWithName:@"texture_read"];
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

  (void)abi;
  NSMutableArray *identities = [NSMutableArray array];
  for (NSUInteger i = 0; i < inputs.count; ++i) {
    NSDictionary *identity = textureBind(device, pipeline, &plans[i], payloads[i], &why);
    if (!identity) return fail(why ? why : "texture_bind", nil);
    [identities addObject:identity];
  }
  NSData *out = [NSJSONSerialization dataWithJSONObject:
      @{@"status": @0, @"load_only": @YES, @"gpu_dispatched": @NO, @"texture": identities[0],
        @"textures": identities, @"texture_inputs": inputs}
      options:0 error:nil];
  if (!out || fwrite(out.bytes, 1, out.length, stdout) != out.length) return fail("reply", nil);
  putchar('\n');
  return fflush(stdout) ? 1 : 0;
}

// The first executable texture protocol is intentionally narrow.  The authored query bundles
// write one fixed uint32 slot, so one thread is the largest launch whose result is determinate.
// This path checks the same resource contract as load-only, binds the delivered user buffer at
// its declared offset, and records the two guard words around the destination.  It is not a
// general texture runtime: every wider shape remains refused by the metadata and query checks.
static int runTextureDispatch(NSString *dir, NSDictionary *manifest, NSDictionary *abi,
                              NSUInteger limit) {
  for (NSString *file in @[@"scan.arc.metallib", @"scan.lib.metallib", @"scan.o",
                           @"program.bin"]) {
    NSDictionary *attributes = [[NSFileManager defaultManager]
      attributesOfItemAtPath:[dir stringByAppendingPathComponent:file] error:nil];
    unsigned long long size = [attributes fileSize];
    if (![attributes[NSFileType] isEqual:NSFileTypeRegular] || !size || size > 16*1024*1024)
      return fail("image_files", nil);
  }
  G17TexturePlan plans[2];
  NSMutableArray *payloads = [NSMutableArray array];
  const char *why = NULL;
  NSArray *inputs = textureInputs(dir, manifest, plans, payloads, &why);
  if (!inputs) return fail(why ? why : "texture_inputs", nil);
  NSArray *coordinateInputs = textureCoordinateInputs(manifest, plans, inputs.count, &why);
  if (!coordinateInputs) return fail(why, nil);
  NSError *error = nil;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) return fail("device", nil);
  id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:
    [dir stringByAppendingPathComponent:@"scan.lib.metallib"]] error:&error];
  if (!library) return fail("library", error);
  id<MTLFunction> function = [library newFunctionWithName:@"texture_read"];
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
  (void)abi;
  NSMutableArray *textures = [NSMutableArray array];
  for (NSUInteger i = 0; i < inputs.count; ++i) {
    id<MTLTexture> texture = textureCreate(device, &plans[i], payloads[i], &why);
    if (!texture) return fail(why ? why : "texture_allocation", nil);
    [textures addObject:texture];
  }
  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (!queue) return fail("command_queue", nil);
  // The image's user record has offset 4.  THE STORE'S FOOTPRINT IS DECLARED BY THE BUNDLE, not
  // assumed here: the two-component form writes the slot AND its companion, so a guard at slot+1
  // sits inside what the store legitimately writes and reports a correct dispatch as a boundary
  // violation.  `result words` names every word the store is expected to write with the value
  // preregistered for it; `guards` names words that must stay at the sentinel.  Both default to
  // the one-word geometry so a bundle written before this still runs unchanged.
  const uint32_t sentinel = 0xA5A5A5A5u;
  NSDictionary *query = manifest[@"query"];
  NSUInteger expected, slot;
  if (!textureQueryContract(query, coordinateInputs.count != 0, &expected, &slot))
    return fail("texture_query_contract", nil);
  NSArray *declaredResults = query[@"result words"], *declaredGuards = query[@"guards"];
  NSMutableArray *resultWords = [NSMutableArray array], *resultValues = [NSMutableArray array];
  NSMutableArray *guardWords = [NSMutableArray array];
  if (declaredResults) {
    if (![declaredResults isKindOfClass:[NSArray class]] || declaredResults.count < 1 ||
        declaredResults.count > 8) return fail("texture_query_result_words", nil);
    for (NSDictionary *row in declaredResults) {
      NSUInteger word, value;
      if (![row isKindOfClass:[NSDictionary class]] ||
          !integer(row[@"word"], UINT32_MAX, &word) ||
          !integer(row[@"expected"], UINT32_MAX, &value))
        return fail("texture_query_result_words", nil);
      [resultWords addObject:@(word)]; [resultValues addObject:@(value)];
    }
    if (![resultWords[0] isEqual:@(slot)] || ![resultValues[0] isEqual:@(expected)])
      return fail("texture_query_result_words", nil);
  } else {
    [resultWords addObject:@(slot)]; [resultValues addObject:@(expected)];
  }
  if (declaredGuards) {
    if (![declaredGuards isKindOfClass:[NSArray class]] || declaredGuards.count < 2 ||
        declaredGuards.count > 8) return fail("texture_query_guards", nil);
    for (NSNumber *row in declaredGuards) {
      NSUInteger word;
      if (!integer(row, UINT32_MAX, &word) || [resultWords containsObject:@(word)])
        return fail("texture_query_guards", nil);
      [guardWords addObject:@(word)];
    }
  } else {
    [guardWords addObject:@(slot - 1)]; [guardWords addObject:@(slot + 1)];
  }
  // OBSERVED, NOT ASSERTED. A word whose behaviour is unmeasured must not be a guard: guarding it
  // turns "we do not know" into a failed dispatch, and leaving it out turns it into nothing. These
  // are read back and reported, and the run does not depend on what they hold.
  NSArray *declaredObserve = query[@"observe"];
  NSMutableArray *observeWords = [NSMutableArray array];
  if (declaredObserve) {
    if (![declaredObserve isKindOfClass:[NSArray class]] || declaredObserve.count > 8)
      return fail("texture_query_observe", nil);
    for (NSNumber *row in declaredObserve) {
      NSUInteger word;
      if (!integer(row, UINT32_MAX, &word) || [resultWords containsObject:@(word)] ||
          [guardWords containsObject:@(word)]) return fail("texture_query_observe", nil);
      [observeWords addObject:@(word)];
    }
  }
  NSUInteger highest = 0;
  for (NSNumber *row in resultWords) highest = MAX(highest, row.unsignedIntegerValue);
  for (NSNumber *row in guardWords) highest = MAX(highest, row.unsignedIntegerValue);
  for (NSNumber *row in observeWords) highest = MAX(highest, row.unsignedIntegerValue);
  if (coordinateInputs.count && highest < 7) return fail("texture_coordinate_storage", nil);
  NSUInteger bindingOffset = 4, words = highest + 1;
  id<MTLBuffer> output = [device newBufferWithLength:bindingOffset + words * sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
  if (!output) return fail("allocation", nil);
  uint32_t *base = (uint32_t *)((uint8_t *)output.contents + bindingOffset);
  for (NSUInteger i = 0; i < words; ++i) base[i] = sentinel;
  NSMutableArray *rows = [NSMutableArray array];
  for (NSUInteger sequence = 1; sequence <= limit; ++sequence) {
    for (NSUInteger i = 0; i < words; ++i) base[i] = sentinel;
    for (NSDictionary *input in coordinateInputs)
      base[[input[@"word"] unsignedIntegerValue]] = [input[@"value"] unsignedIntValue];
    id<MTLCommandBuffer> cb = g17_gpu_cb(queue);
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    if (!cb || !encoder) return fail("command", nil);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:output offset:bindingOffset atIndex:0];
    for (NSUInteger i = 0; i < textures.count; ++i)
      [encoder setTexture:textures[i] atIndex:plans[i].denseIndex];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    fprintf(stderr, "phase=submitting sequence=%lu\n", (unsigned long)sequence); fflush(stderr);
    [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted || cb.error) return fail("command_status", cb.error);
    for (NSUInteger i = 0; i < textures.count; ++i)
      if (!textureUnchanged(textures[i], &plans[i], payloads[i])) return fail("texture_read_only", nil);
    NSMutableDictionary *inputsSeen = [NSMutableDictionary dictionary];
    for (NSDictionary *input in coordinateInputs) {
      NSUInteger word = [input[@"word"] unsignedIntegerValue];
      if (base[word] != [input[@"value"] unsignedIntValue])
        return fail("texture_coordinate_input_changed", nil);
      inputsSeen[[input[@"word"] stringValue]] = @(base[word]);
    }
    NSMutableDictionary *guardsSeen = [NSMutableDictionary dictionary];
    for (NSNumber *row in guardWords) {
      uint32_t seen = base[row.unsignedIntegerValue];
      guardsSeen[row.stringValue] = @(seen);
      if (seen != sentinel) return fail("boundary_guard", nil);
    }
    NSMutableDictionary *resultsSeen = [NSMutableDictionary dictionary];
    for (NSUInteger i = 0; i < resultWords.count; ++i) {
      NSUInteger word = [resultWords[i] unsignedIntegerValue];
      uint32_t seen = base[word];
      resultsSeen[[resultWords[i] stringValue]] = @(seen);
      if (seen != [resultValues[i] unsignedIntValue]) {
        fprintf(stderr, "texture_value word=%lu observed=%08x expected=%08x buffer=",
                (unsigned long)word, seen, [resultValues[i] unsignedIntValue]);
        for (NSUInteger w = 0; w < words; ++w) fprintf(stderr, "%08x%s", base[w], w+1 == words ? "\n" : ",");
        fflush(stderr);
        return fail("texture_value", nil);
      }
    }
    NSMutableDictionary *observed = [NSMutableDictionary dictionary];
    for (NSNumber *row in observeWords)
      observed[row.stringValue] = @(base[row.unsignedIntegerValue]);
    [rows addObject:@{ @"sequence": @(sequence), @"word": @(base[slot]),
                       @"result_words": resultsSeen, @"guards": guardsSeen,
                       @"texture_read_only": @YES,
                       @"observed": observed,
                       @"guard_before": @(base[[guardWords[0] unsignedIntegerValue]]),
                       @"guard_after": @(base[[guardWords.lastObject unsignedIntegerValue]]),
                       @"gpu_seconds": @(cb.GPUEndTime - cb.GPUStartTime) }];
    if (coordinateInputs.count) {
      NSMutableDictionary *record = [rows.lastObject mutableCopy];
      record[@"input_words"] = inputsSeen;
      record[@"coordinate_input_words"] = query[@"coordinate input words"];
      rows[rows.count - 1] = record;
    }
  }
  NSMutableArray *textureIdentities = [NSMutableArray array];
  for (NSUInteger i = 0; i < textures.count; ++i) {
    G17TexturePlan plan = plans[i];
    id<MTLTexture> texture = textures[i];
    NSMutableDictionary *entry = [inputs[i] mutableCopy];
    [entry addEntriesFromDictionary:@{@"format_witnessed": @(plan.formatWitnessed),
       @"texels": @(plan.width * plan.height), @"resource_id": @(texture.gpuResourceID._impl)}];
    [textureIdentities addObject:entry];
  }
  NSDictionary *identity = @{ @"pipeline_builds": @1,
      @"worker_pid": @([[NSProcessInfo processInfo] processIdentifier]), @"texture": textureIdentities[0],
      @"textures": textureIdentities, @"output_buffer_bytes": @(output.length),
      @"binding_offset": @(bindingOffset), @"read_back_word": @(slot) };
  NSData *json = [NSJSONSerialization dataWithJSONObject:
      @{ @"status": @0, @"gpu_dispatched": @YES, @"queries": rows, @"identity": identity }
      options:0 error:nil];
  fwrite(json.bytes, 1, json.length, stdout); putchar('\n');
  return 0;
}

// The first ordinary-runtime tensor class is deliberately narrow.  It is the measured composed
// compiler program: one 17x19x16 half GEMM, a threadgroup-position read and an FP32 add/store.
// This worker owns transport and launch only; the Python contract model and scanlink own image
// admission.  Every field is repeated here so malformed JSON cannot reach Metal by bypassing the
// Python caller.
static BOOL tensorImageFiles(NSString *dir) {
  for (NSString *file in @[@"scan.arc.metallib", @"scan.lib.metallib", @"scan.o", @"program.bin"]) {
    NSDictionary *attributes = [[NSFileManager defaultManager]
      attributesOfItemAtPath:[dir stringByAppendingPathComponent:file] error:nil];
    unsigned long long size = [attributes fileSize];
    if (![attributes[NSFileType] isEqual:NSFileTypeRegular] || !size || size > 16*1024*1024)
      return NO;
  }
  return YES;
}

// gemm_generic (Set A with Set C): one rule-based single-GEMM class for new verifications, so a new
// shape, type or launch is not a new name. Every other tensor class keeps its own exact checks.
static NSUInteger elementBytes(id type) {
  if ([type isEqual:@"half"] || [type isEqual:@"bfloat"]) return 2;
  if ([type isEqual:@"float"]) return 4;
  if ([type isEqual:@"fp8e4m3"] || [type isEqual:@"fp8e5m2"] || [type isEqual:@"int8"]) return 1;
  return 0;
}
static BOOL genericSpec(NSDictionary *tensor, NSUInteger *M, NSUInteger *N, NSUInteger *K,
                        NSUInteger *lda, NSUInteger *ldb, NSUInteger *ldc, NSUInteger *threads,
                        NSUInteger *groupThreads) {
  NSUInteger sg;
  // K reaches 4096 in gemm_generic alone: a runtime K loop re-indexes per slice (Set A item 5).
  // MM P13 (agxforge.g17.runtime.GENERIC_CLASSES, section 25.127): N reaches 256 (wide_n) and simdgroups 8
  // (simdgroups_8); the class rules themselves are Python's, and these are the numeric bounds.
  if (!integer(tensor[@"M"], 16384, M) || !integer(tensor[@"N"], 16384, N) || !integer(tensor[@"K"], 1048576, K) ||
      !integer(tensor[@"lda"], 1000000, lda) || !integer(tensor[@"ldb"], 1000000, ldb) ||
      !integer(tensor[@"ldc"], 1000000, ldc) || !integer(tensor[@"simdgroups"], 8, &sg)) return NO;
  // transA / transB (P13's transposed class): absent, null or true; lda and ldb stay the logical K and N,
  // so every payload keeps its untransposed size
  for (NSString *key in @[@"transA", @"transB"]) {
    id t = tensor[key];
    if (t && t != [NSNull null] && ![t isEqual:@YES]) return NO;
  }
  if (*M == 0 || *N == 0 || *K == 0 || *M % 16 || *N % 16 || *K % 16) return NO;
  BOOL chain = tensor[@"stages"] && ![tensor[@"stages"] isKindOfClass:[NSNull class]];
  // contiguous; for a chain, ldb is stage 0's width N0 and N the C buffer's width (the widest stage)
  if (*lda != *K || *ldc != *N || (chain ? (*ldb > *N || *ldb % 16 || !*ldb) : *ldb != *N)) return NO;
  NSUInteger ea = elementBytes(tensor[@"a_type"]), eb = elementBytes(tensor[@"b_type"]);
  if (!ea || !eb) return NO;
  // int8 pairs with int8 into int32 C, one simdgroup, one threadgroup, no epilogue; C is an input
  // either way (c.f32 is copied in before every query), so an accumulate needs no new buffer
  BOOL i8a = [tensor[@"a_type"] isEqual:@"int8"], i8b = [tensor[@"b_type"] isEqual:@"int8"];
  if (i8a != i8b) return NO;
  if (i8a ? ![tensor[@"c_type"] isEqual:@"int"] : ![tensor[@"c_type"] isEqual:@"float"]) return NO;
  // MM P13 int8_split: the grid and simdgroup splits are admitted; an epilogue or chain is not
  if (i8a && (chain || (tensor[@"epilogue"] && tensor[@"epilogue"] != [NSNull null]))) return NO;
  if (!i8a && ((tensor[@"accumulate"] && tensor[@"accumulate"] != [NSNull null]) ||
               (tensor[@"saturate"] && tensor[@"saturate"] != [NSNull null]))) return NO;
  // an fp8 operand pairs with fp8 or bfloat, as the Python schema says (the MMA is bf16 x bf16)
  if ((ea == 1 || eb == 1) && !((ea == 1 || [tensor[@"a_type"] isEqual:@"bfloat"]) &&
                                (eb == 1 || [tensor[@"b_type"] isEqual:@"bfloat"]))) return NO;
  if (!(sg == 1 || sg == 2 || sg == 4 || sg == 8)) return NO;
  NSArray *tg = tensor[@"threadgroup"], *grid = tensor[@"grid"];
  if (![tg isKindOfClass:[NSArray class]] || ![grid isKindOfClass:[NSArray class]] || tg.count != 3 || grid.count != 3) return NO;
  if (![tg[1] isEqual:@1] || ![tg[2] isEqual:@1] || ![grid[1] isEqual:@1] || ![grid[2] isEqual:@1]) return NO;
  if (!integer(tg[0], 256, groupThreads) || !integer(grid[0], 65536, threads)) return NO;   // 256 tgs x 8 simdgroups (MM 25.144.1)
  if (*groupThreads != 32 * sg || !*groupThreads || *threads % *groupThreads) return NO;
  NSUInteger groups = *threads / *groupThreads;
  // P3 N-TILED GRID: grid_n of the threadgroups split N (columns), each owning N/grid_n whole tiles;
  // the rest split M (rows) as before. Absent grid_n leaves the M-only check unchanged (Set A item 3).
  NSUInteger colGroups = 1;
  if (tensor[@"grid_n"] != nil && ![tensor[@"grid_n"] isEqual:[NSNull null]]) {
    if (!integer(tensor[@"grid_n"], 256, &colGroups)) return NO;
  }
  // P3 SPLIT-K: kGroups of the threadgroups partition K (the contraction), each computing a partial over
  // K/kGroups and writing it to slot t of a (kGroups*M) x N buffer. K must split into whole 16-wide issues.
  NSUInteger kGroups = 1;
  if (tensor[@"split_k"] != nil && ![tensor[@"split_k"] isEqual:[NSNull null]]) {
    if (!integer(tensor[@"split_k"], 256, &kGroups)) return NO;
  }
  if (!groups || groups > 256 || (groups & (groups - 1))) return NO;
  if (colGroups > 1) {
    if (groups % colGroups || (colGroups & (colGroups - 1)) || *N % (16 * colGroups)) return NO;
    // the per-threadgroup TILE count N/(16*grid_n) is a shift in tlower, so it must be a power of two too
    // (not just grid_n): N=6144 grid_n=32 gives 12 tiles and is refused here rather than at build
    NSUInteger tilesPerTg = *N / (16 * colGroups);
    if (tilesPerTg & (tilesPerTg - 1)) return NO;
  }
  if (*N / colGroups > 256) return NO;                     // per-threadgroup N <= 256 (register budget); grid_n keeps it bounded
  if (kGroups > 1) {
    if (!(sg == 1 || sg == 2 || sg == 4)) return NO;        // split_k with 1, 2 or 4 simdgroups (MM 25.144.1; matches tlower)
    if (colGroups > 1 && (colGroups & (colGroups - 1))) return NO;   // grid_n a power of two (the id split is a shift)
    if ((groups / colGroups) % kGroups || *K % (16 * kGroups)) return NO;
  }
  // PER-THREADGROUP K is the real bound (the kloop's 256 16-wide trips): K/split_k <= 4096. split_k lets the
  // total K grow past 4096 (the FFN down K=8192 as split_k=4 = 2048 per threadgroup, one dispatch).
  if (*K / kGroups > 8192) return NO;                      // 255 trips at 2+ slices per trip (MM 25.144.1); tlower checks the trips
  NSUInteger rowGroups = groups / colGroups / kGroups;
  if (*M % (16 * sg * rowGroups)) return NO;               // each row group owns whole tiles per simdgroup
  if (*M / (sg * rowGroups) > 1024) return NO;             // at most 64 tile rows per simdgroup
  return YES;
}

// MM P12, closed by its second branch: cooperative threadgroup-memory SHARING is refused by name
// (phase=tensor_cooperative_sharing), the same manifests agxforge.g17.runtime refuses with
// COOPERATIVE_SHARING_REFUSAL. gemm_generic's simdgroups 2 and 4 are a per-simdgroup tile partition
// with no threadgroup memory, as Apple's own multi-simdgroup MMA code is. A key naming sharing,
// cooperation or threadgroup memory is a request, whatever its value, so it is not ignored.
static BOOL sharingKeys(id object) {
  if (![object isKindOfClass:[NSDictionary class]]) return NO;
  for (id key in (NSDictionary *)object) {
    if (![key isKindOfClass:[NSString class]]) continue;
    NSString *k = [(NSString *)key lowercaseString];
    for (NSString *w in @[@"shar", @"cooperat", @"threadgroup_mem", @"tg_mem"])
      if ([k containsString:w]) return YES;
  }
  return NO;
}
static BOOL sharingRequest(NSDictionary *manifest) {
  NSDictionary *abi = manifest[@"abi"];
  if (sharingKeys(manifest) || sharingKeys(manifest[@"tensor"]) || sharingKeys(abi)) return YES;
  return [abi isKindOfClass:[NSDictionary class]] &&
         (boolean(abi[@"uses_threadgroup"], YES) || (abi[@"threadgroup"] && abi[@"threadgroup"] != [NSNull null]));
}

static BOOL tensorContract(NSDictionary *manifest, NSUInteger *M, NSUInteger *N, NSUInteger *K,
                           NSUInteger *lda, NSUInteger *ldb, NSUInteger *ldc,
                           NSUInteger *aBytes, NSUInteger *bBytes, NSUInteger *cBytes,
                           NSString **nameOut, BOOL *multigemmOut, const char **why) {
  *why = "tensor_contract";
  if (![manifest isKindOfClass:[NSDictionary class]] ||
      ![manifest[@"format"] isEqual:@"g17-common-pipeline-v3"] ||
      ![manifest[@"kind"] isEqual:@"tensor_gemm"])
    { *why = "tensor_manifest_format"; return NO; }
  NSString *name = manifest[@"name"];
  BOOL multigemm = [name isEqual:@"tensor_multigemm_runtime_demo"] ||
                   [name isEqual:@"tensor_multigemm_fadd_fmul_runtime_demo"] ||
                   [name isEqual:@"tensor_multigemm_fadd_fmul_gemm_runtime_demo"] ||
                   [name isEqual:@"tensor_multigemm_relu_vec_residual_runtime_demo"] ||
                   [name isEqual:@"tensor_multigemm_weight_offset_runtime_demo"] ||
                   [name isEqual:@"tensor_multigemm_register_runtime_demo"] ||
                   [name isEqual:@"tensor_multigemm_adjacent_runtime_demo"] ||
                   [name isEqual:@"tensor_multigemm_epilogue_register_runtime_demo"] ||
                   [name isEqual:@"tensor_multigemm_epilogue_adjacent_runtime_demo"] ||
                   [name isEqual:@"tensor_transformer_layer_runtime_demo"] ||
                   [name isEqual:@"tensor_transformer_layer_weight_offset_runtime_demo"] ||
                   [name isEqual:@"tensor_transformer_continuation_weight_offset_runtime_demo"] ||
                   [name isEqual:@"tensor_transformer_two_layer_runtime_demo"];
  BOOL weightOffset = [name isEqual:@"tensor_multigemm_weight_offset_runtime_demo"];
  // THE ONE CLASS THAT LAUNCHES MORE THAN ONE THREADGROUP: G threadgroups of 32 threads split M
  // (tlower's grid split, reading SR_TG_X). Every other tensor class keeps its literal 32 threads.
  BOOL gridGemm = [name isEqual:@"tensor_gemm_grid_runtime_demo"];
  // ONE fp8 CLASS: e4m3 A and e5m2 B, one byte per element, one GEMM plus the scalar epilogue.
  BOOL fp8Gemm = [name isEqual:@"tensor_gemm_fp8_runtime_demo"];
  BOOL genericGemm = [name isEqual:@"tensor_gemm_generic_runtime_demo"];
  BOOL transformer = [name isEqual:@"tensor_transformer_layer_runtime_demo"];
  BOOL transformerOffset = [name isEqual:@"tensor_transformer_layer_weight_offset_runtime_demo"];
  BOOL transformerContinuation = [name isEqual:@"tensor_transformer_continuation_weight_offset_runtime_demo"];
  BOOL transformerTwo = [name isEqual:@"tensor_transformer_two_layer_runtime_demo"];
  BOOL reduction = [name isEqual:@"tensor_row_softmax_runtime_demo"];
  BOOL ffn = [name isEqual:@"tensor_ffn_gelu_layernorm_runtime_demo"] ||
             [name isEqual:@"tensor_ffn_gelu_layernorm_allrows_runtime_demo"];
  BOOL wideFfn = [name isEqual:@"tensor_ffn_gelu_layernorm_wide_runtime_demo"];
  BOOL threeGemm = [name isEqual:@"tensor_multigemm_fadd_fmul_gemm_runtime_demo"] ||
                   [name isEqual:@"tensor_multigemm_relu_vec_residual_runtime_demo"] ||
                   transformer || transformerOffset || transformerContinuation || transformerTwo;
  if (![name isKindOfClass:[NSString class]] ||
      (!multigemm && !reduction && !ffn && !wideFfn && !gridGemm && !fp8Gemm && !genericGemm && ![name isEqual:@"tensor_runtime_demo"])) {
    *why = "tensor_function_name"; return NO;
  }
  NSDictionary *tensor = manifest[@"tensor"];
  if (![tensor isKindOfClass:[NSDictionary class]]) { *why = "tensor_spec"; return NO; }
  if (sharingRequest(manifest)) { *why = "tensor_cooperative_sharing"; return NO; }
  NSUInteger simdgroups, genericThreads = 0, genericGroup = 0;
  if (genericGemm) {
    if (!genericSpec(tensor, M, N, K, lda, ldb, ldc, &genericThreads, &genericGroup)) { *why = "tensor_spec"; return NO; }
  } else if (!integer(tensor[@"M"], 128, M) || !integer(tensor[@"N"], 128, N) ||
      !integer(tensor[@"K"], 256, K) || !integer(tensor[@"lda"], 1000000, lda) ||
      !integer(tensor[@"ldb"], 1000000, ldb) || !integer(tensor[@"ldc"], 1000000, ldc) ||
      (reduction ? (*M != 32 || *N != 16 || *K != 64 || *lda != 64 || *ldb != 16 || *ldc != 16) :
       ((ffn || wideFfn) ? (*M != 16 || *N != (wideFfn ? 32 : 16) || *K != 64 ||
                            *lda != 64 || *ldb != (wideFfn ? 32 : 16) ||
                            *ldc != (wideFfn ? 32 : 16)) :
       (transformerContinuation ? (*M != 16 || *N != 32 || *K != 32 || *lda != 32 || *ldb != 32 || *ldc != 32) :
       ((transformer || transformerOffset || transformerTwo) ? (*M != 16 || *N != 32 || *K != 64 || *lda != 64 || *ldb != 32 || *ldc != 32) :
       (weightOffset ? (*M != 16 || *N != 32 || *K != 64 || *lda != 64 || *ldb != 32 || *ldc != 32) :
       gridGemm ? (*M != 128 || *N != 32 || *K != 64 || *lda != 64 || *ldb != 32 || *ldc != 32) :
       fp8Gemm ? (*M != 32 || *N != 32 || *K != 64 || *lda != 64 || *ldb != 32 || *ldc != 32) :
       (multigemm ? (*M != 32 || *N != 32 || *K != 64 || *lda != 64 || *ldb != 32 || *ldc != 32)
                  : (*M != 17 || *N != 19 || *K != 16 || *lda != 16 || *ldb != 19 || *ldc != 19))))))) ||
      (fp8Gemm ? (![tensor[@"a_type"] isEqual:@"fp8e4m3"] || ![tensor[@"b_type"] isEqual:@"fp8e5m2"]) :
       ((![tensor[@"a_type"] isEqual:@"half"] && !(transformerContinuation && [tensor[@"a_type"] isEqual:@"float"])) || ![tensor[@"b_type"] isEqual:@"half"])) ||
      ![tensor[@"c_type"] isEqual:@"float"] || !integer(tensor[@"simdgroups"], 1, &simdgroups) ||
      simdgroups != 1 ||
      !(gridGemm ? ([tensor[@"grid"] isEqual:@[@32, @1, @1]] || [tensor[@"grid"] isEqual:@[@64, @1, @1]] ||
                    [tensor[@"grid"] isEqual:@[@128, @1, @1]] || [tensor[@"grid"] isEqual:@[@256, @1, @1]])
                 : [tensor[@"grid"] isEqual:@[@32, @1, @1]]) ||
      ![tensor[@"threadgroup"] isEqual:@[@32, @1, @1]])
    { *why = "tensor_spec"; return NO; }
  // production row P7 (MM 25.129): the named fused attention class shares gemm_generic's transport
  // (buffers sized from M, N and K by genericSpec); agxforge.g17.runtime holds its admission rules
  BOOL compositionOK = genericGemm ? ([tensor[@"composition"] isEqual:@"gemm_generic"] ||
                                      [tensor[@"composition"] isEqual:@"attention"]) :
      gridGemm ? [tensor[@"composition"] isEqual:@"gemm_grid"] :
      fp8Gemm ? [tensor[@"composition"] isEqual:@"gemm_fp8"] :
      reduction ? [tensor[@"composition"] isEqual:@"row_softmax_fp32"] :
      (ffn || wideFfn) ? ([tensor[@"composition"] isEqual:@"ffn_gelu_layernorm"] ||
                          [tensor[@"composition"] isEqual:@"ffn_gelu_layernorm_allrows"] ||
                          [tensor[@"composition"] isEqual:@"ffn_gelu_layernorm_wide"]) :
      transformer ? [tensor[@"composition"] isEqual:@"transformer_layer"] :
      transformerOffset ? [tensor[@"composition"] isEqual:@"transformer_layer_weight_offset"] :
      transformerContinuation ? [tensor[@"composition"] isEqual:@"transformer_continuation_weight_offset"] :
      transformerTwo ? [tensor[@"composition"] isEqual:@"transformer_two_layer"] :
      threeGemm ?
      ([tensor[@"composition"] isEqual:@"gemm_fadd_fmul_gemm_fadd_gemm_memory"] ||
       [tensor[@"composition"] isEqual:@"gemm_relu_vec_gemm_residual_memory"]) :
      weightOffset ? [tensor[@"composition"] isEqual:@"gemm_weight_offset"] :
      ([tensor[@"composition"] isEqual:@"gemm_fadd_gemm_memory"] ||
       [tensor[@"composition"] isEqual:@"gemm_fadd_fmul_gemm_memory"] ||
       ([name isEqual:@"tensor_multigemm_adjacent_runtime_demo"] &&
        [tensor[@"composition"] isEqual:@"gemm_gemm_memory"]) ||
       ([name isEqual:@"tensor_multigemm_register_runtime_demo"] &&
        [tensor[@"composition"] isEqual:@"gemm_gemm_register"]) ||
       ([name isEqual:@"tensor_multigemm_epilogue_register_runtime_demo"] &&
        [tensor[@"composition"] isEqual:@"gemm_epilogue_gemm_register"]) ||
       ([name isEqual:@"tensor_multigemm_epilogue_adjacent_runtime_demo"] &&
        [tensor[@"composition"] isEqual:@"gemm_epilogue_gemm_memory"]));
  BOOL hasK3 = tensor[@"K3"] != nil && ![tensor[@"K3"] isKindOfClass:[NSNull class]];
  if ((ffn || wideFfn) && (!compositionOK || ![tensor[@"K2"] isEqual:@(wideFfn ? 32 : 16)] ||
              ![tensor[@"a2_type"] isEqual:@"float"] ||
              ![tensor[@"b2_type"] isEqual:@"half"])) {
    *why = "tensor_spec"; return NO;
  }
  if (multigemm && (!compositionOK || ![tensor[@"K2"] isEqual:@32] ||
                    ![tensor[@"a2_type"] isEqual:@"float"] ||
                    ![tensor[@"b2_type"] isEqual:@"half"] ||
                    (threeGemm ? ![tensor[@"K3"] isEqual:@32] : hasK3))) {
    *why = "tensor_spec"; return NO;
  }
  if (weightOffset) {
    NSUInteger weightOffset;
    if (![tensor[@"M"] isEqual:@16] || ![tensor[@"N"] isEqual:@32] ||
        ![tensor[@"K"] isEqual:@64] || ![tensor[@"K2"] isEqual:@32] ||
        !integer(tensor[@"weight_offset_b"], 1000000, &weightOffset) ||
        !(weightOffset == 4096 || weightOffset == 4352 || weightOffset == 8192 || weightOffset == 12288 || weightOffset == 16384 || weightOffset == 20480)) {
      *why = "tensor_weight_offset"; return NO;
    }
  }
  if (transformerOffset) {
    NSArray *offsets = tensor[@"weight_offsets_b"];
    if (![offsets isEqual:@[@0, @4096, @8192]]) {
      *why = "tensor_weight_offsets"; return NO;
    }
  }
  if (transformerContinuation) {
    NSArray *offsets = tensor[@"weight_offsets_b"];
    if (![offsets isEqual:@[@0, @4096, @8192]] || ![tensor[@"K2"] isEqual:@32] ||
        ![tensor[@"K3"] isEqual:@32] || ![tensor[@"a2_type"] isEqual:@"float"] ||
        ![tensor[@"b2_type"] isEqual:@"half"]) {
      *why = "tensor_continuation_spec"; return NO;
    }
  }
  if ((gridGemm || fp8Gemm || genericGemm) && !compositionOK) { *why = "tensor_spec"; return NO; }
  if (!multigemm && !reduction && !ffn && !wideFfn && !gridGemm && !fp8Gemm && !genericGemm &&
      ![tensor[@"composition"] isEqual:@"gemm_fadd_threadgroup_position"]) {
    *why = "tensor_spec"; return NO;
  }
  // Payload sizes are declared storage bytes, independent of tensor arithmetic operand types.
  *aBytes = *M * *lda * (transformerContinuation ? 4u : 2u); *bBytes = *K * *ldb * 2; *cBytes = *M * *ldc * 4;
  if (fp8Gemm) { *aBytes = *M * *lda; *bBytes = *K * *ldb; }      // one byte per fp8 element
  if (genericGemm) {                                                // each operand at its own width
    *aBytes = *M * *lda * elementBytes(tensor[@"a_type"]); *bBytes = *K * *ldb * elementBytes(tensor[@"b_type"]);
    // ADDITIVE (Set A item 9a): an "mx32" epilogue (OCP MX block scaling, checked in runtime.py) reads
    // predecoded fp32 scale tables appended to the operands: (K/32) x M after A, (K/32) x N after B
    NSArray *steps = tensor[@"epilogue"];
    if ([steps isKindOfClass:[NSArray class]] && [steps containsObject:@"mx32"]) {
      if (*K % 32) { *why = "tensor_spec"; return NO; }
      *aBytes += (*K / 32) * *M * 4; *bBytes += (*K / 32) * *N * 4;
    }
    // ADDITIVE (production row P5, MM 25.128): "mx32e8m0" reads the E8M0 CODE BYTES instead, one byte
    // per factor, decoded in the kernel: (K/32) x M bytes after A, (K/32) x N after B
    if ([steps isKindOfClass:[NSArray class]] && [steps containsObject:@"mx32e8m0"]) {
      if (*K % 32 || [steps containsObject:@"mx32"]) { *why = "tensor_spec"; return NO; }
      *aBytes += (*K / 32) * *M; *bBytes += (*K / 32) * *N;
    }
    // a CHAIN (stages): every stage reads B from offset 0 as K_i x N_i halves, so B is the largest of
    // those; C (the reply) is M x N, N being the widest stage (checked in runtime.py and here).
    NSArray *stages = tensor[@"stages"];
    if (stages && ![stages isKindOfClass:[NSNull class]]) {
      if (![stages isKindOfClass:[NSArray class]] || stages.count < 1 || stages.count > 7) { *why = "tensor_spec"; return NO; }
      NSUInteger prevN = *ldb, biggest = *K * *ldb;
      for (NSArray *s in stages) {
        NSUInteger n, k;
        if (![s isKindOfClass:[NSArray class]] || (s.count != 3 && s.count != 4) || !integer(s[0], 128, &n) || !integer(s[1], 256, &k) ||
            !n || !k || n % 16 || k % 16 || n > *N || !([s[2] isEqual:@"float"] || [s[2] isEqual:@"half"])) { *why = "tensor_spec"; return NO; }
        // a feed mode (recon section 132 part 3): A, B, At or Bt; the non-A modes are one square stage
        if (s.count == 4) {
          if (!([s[3] isEqual:@"A"] || [s[3] isEqual:@"B"] || [s[3] isEqual:@"At"] || [s[3] isEqual:@"Bt"])) { *why = "tensor_spec"; return NO; }
          if (![s[3] isEqual:@"A"] && (stages.count != 1 || n != k || n != *N || *M != *N)) { *why = "tensor_spec"; return NO; }
        }
        if (k != prevN) { *why = "tensor_spec"; return NO; }   // each stage's K is the previous stage's N
        prevN = n;
        if (k * n > biggest) biggest = k * n;
      }
      *bBytes = biggest * 2;
    }
    // P3 SPLIT-K: the C buffer holds kGroups partials stacked along the row axis ((kGroups*M) x N); each
    // threadgroup t writes its M x N partial to slot t. The reduce folds them (outside this dispatch).
    if (tensor[@"split_k"] != nil && ![tensor[@"split_k"] isEqual:[NSNull null]]) {
      NSUInteger kGroups = 1;
      if (!integer(tensor[@"split_k"], 256, &kGroups)) { *why = "tensor_spec"; return NO; }
      if (kGroups > 1) *cBytes *= kGroups;
    }
  }
  if (weightOffset) {
    NSUInteger weightOffset;
    if (!integer(tensor[@"weight_offset_b"], 1000000, &weightOffset) ||
        !(weightOffset == 4096 || weightOffset == 4352 || weightOffset == 8192 || weightOffset == 12288 || weightOffset == 16384 || weightOffset == 20480)) {
      *why = "tensor_weight_offset"; return NO;
    }
    *bBytes = weightOffset + 32u * 32u * 2u;
  }
  if (transformerOffset || transformerContinuation) *bBytes = 8192u + 32u * 32u * 2u;
  NSDictionary *abi = manifest[@"abi"];
  if (![abi isKindOfClass:[NSDictionary class]]) { *why = "tensor_abi"; return NO; }
  NSUInteger version, entry;
  if (!integer(abi[@"abi_version"], 5, &version) || version != 5 ||
      !integer(abi[@"entry"], 64, &entry) || entry != 64 ||
      !boolean(abi[@"arch_flag"], NO) || !boolean(abi[@"uses_threadgroup"], NO) ||
      !boolean(abi[@"writes_buffer"], YES) || !boolean(abi[@"has_stores"], YES) ||
      !boolean(abi[@"writes_texture"], NO)) { *why = "tensor_abi_semantics"; return NO; }
  NSDictionary *launch = abi[@"launch"];
  if (![launch isKindOfClass:[NSDictionary class]] || !boolean(launch[@"bounds_checked"], NO) ||
      !boolean(launch[@"exact_grid_required"], YES)) { *why = "tensor_launch"; return NO; }
  NSString *prologue = abi[@"prologue"];
  NSMutableString *expected = [NSMutableString stringWithString:@"0e000000"];
  for (NSUInteger i = 4; i < 64; i += 2) [expected appendString:@"0600"];
  if (![prologue isKindOfClass:[NSString class]] || ![prologue isEqual:expected]) {
    *why = "tensor_prologue"; return NO;
  }
  NSArray *srs = abi[@"system_registers"];
  // gemm_generic's simdgroup-split body adds SR133 to the SR130+SR156 epilogue set (ABI v5, #145);
  // every other tensor class keeps exactly SR130+SR156.
  // Set A item 12: a gemm_generic body staging through an explicit imageblock reads SR_LOCAL_X/Y
  // for the tile coordinate, the (130, 164, 165) set of Apple's own tensor+imageblock compile.
  if (![srs isEqual:@[@130, @156]] && !(genericGemm && ([srs isEqual:@[@130, @133, @156]] ||
                                                         [srs isEqual:@[@130, @164, @165]])))
    { *why = "tensor_system_registers"; return NO; }
  NSArray *extra = abi[@"pk_extra"]; NSDictionary *pk = abi[@"pk_values"];
  if (![extra isEqual:@[@15, @16]] || ![pk isKindOfClass:[NSDictionary class]] ||
      ![pk[@"15"] isEqual:@1] || ![pk[@"16"] isEqual:@1]) {
    *why = "tensor_metadata"; return NO;
  }
  NSArray *bindings = abi[@"bindings"];
  if (![bindings isKindOfClass:[NSArray class]] || bindings.count != 3) {
    *why = "tensor_bindings"; return NO;
  }
  NSArray *indices = @[@1, @2, @3], *offsets = @[@0, @2, @4];
  NSArray *types = transformerContinuation ? @[@"float", @"half", @"float"] : @[@"half", @"half", @"float"];
  NSArray *widths = transformerContinuation ? @[@4, @2, @4] : @[@2, @2, @4];
  NSArray *writes = @[@NO, @NO, @YES];
  for (NSUInteger i = 0; i < 3; ++i) {
    NSDictionary *binding = bindings[i]; NSUInteger v;
    if (![binding isKindOfClass:[NSDictionary class]] || ![binding[@"index"] isEqual:indices[i]] ||
        ![binding[@"offset"] isEqual:offsets[i]] || ![binding[@"element_type"] isEqual:types[i]] ||
        ![binding[@"element_bytes"] isEqual:widths[i]] || ![binding[@"written"] isEqual:writes[i]] ||
        !integer(binding[@"index"], 30, &v)) { *why = "tensor_bindings"; return NO; }
  }
  NSDictionary *execution = abi[@"execution"];
  if (![execution isKindOfClass:[NSDictionary class]] ||
      ![execution[@"simd_width"] isEqual:@32] || ![execution[@"tensor"] isEqual:@YES]) {
    *why = "tensor_execution"; return NO;
  }
  NSArray *forms = abi[@"forms"];
  if (![forms isKindOfClass:[NSArray class]] || !forms.count) { *why = "tensor_forms"; return NO; }
  *nameOut = name; if (multigemmOut) *multigemmOut = multigemm; *why = NULL;
  return YES;
}


static NSDictionary *tensorLayout(NSDictionary *manifest, const char **why) {
  NSUInteger M, N, K, lda, ldb, ldc, aBytes, bBytes, cBytes; NSString *name = nil;
  BOOL multigemm = NO;
  if (!tensorContract(manifest, &M, &N, &K, &lda, &ldb, &ldc, &aBytes, &bBytes, &cBytes, &name, &multigemm, why)) return nil;
  NSString *composition = [name isEqual:@"tensor_row_softmax_runtime_demo"] ? @"row_softmax_fp32" :
      [name isEqual:@"tensor_ffn_gelu_layernorm_wide_runtime_demo"] ? @"ffn_gelu_layernorm_wide" :
      [name isEqual:@"tensor_ffn_gelu_layernorm_allrows_runtime_demo"] ? @"ffn_gelu_layernorm_allrows" :
      [name isEqual:@"tensor_ffn_gelu_layernorm_runtime_demo"] ? @"ffn_gelu_layernorm" :
      [name isEqual:@"tensor_transformer_layer_runtime_demo"] ? @"transformer_layer" :
      [name isEqual:@"tensor_transformer_layer_weight_offset_runtime_demo"] ? @"transformer_layer_weight_offset" :
      [name isEqual:@"tensor_transformer_continuation_weight_offset_runtime_demo"] ? @"transformer_continuation_weight_offset" :
      [name isEqual:@"tensor_transformer_two_layer_runtime_demo"] ? @"transformer_two_layer" :
      [name isEqual:@"tensor_multigemm_weight_offset_runtime_demo"] ? @"weight_offset" :
      (multigemm ? @"multigemm" : @"single");
  return @{@"function": name, @"kind": @"tensor_gemm", @"storage_dtype": @"mixed",
           @"rows": @(M), @"columns": @(N), @"matrix_bytes": @(aBytes),
           @"request_bytes": @(aBytes), @"reply_bytes": @(cBytes), @"reply_elements": @(M*N),
           @"completion_markers": @NO, @"output_bytes": @(cBytes + 256), @"buffer_allocations": @3,
           @"buffer_roles": @[@"a", @"b", @"output"],
           @"buffer_payload_bytes": @[@(aBytes), @(bBytes), @(cBytes)],
           @"buffer_allocation_bytes": @[@(aBytes + 256), @(bBytes + 256), @(cBytes + 256)],
           @"buffer_offsets": @[@128, @128, @128], @"protocol": @3,
           @"bindings": manifest[@"abi"][@"bindings"], @"gpu_dispatched": @NO,
           @"tensor": manifest[@"tensor"], @"composition": composition};
}

static int runTensorLoadOnly(NSString *dir, NSDictionary *manifest) {
  if (!tensorImageFiles(dir)) return fail("image_files", nil);
  const char *why = NULL; NSUInteger M, N, K, lda, ldb, ldc, aBytes, bBytes, cBytes; NSString *name = nil;
  BOOL multigemm = NO;
  if (!tensorContract(manifest, &M, &N, &K, &lda, &ldb, &ldc, &aBytes, &bBytes, &cBytes, &name, &multigemm, &why))
    return fail(why, nil);
  NSError *error = nil; fprintf(stderr, "phase=create_pipeline\n"); fflush(stderr);
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) return fail("device", nil);
  id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:
      [dir stringByAppendingPathComponent:@"scan.lib.metallib"]] error:&error];
  if (!library) return fail("library", error);
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (!function) return fail("function", nil);
  MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
  ad.url = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"scan.arc.metallib"]];
  id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:ad error:&error];
  if (!archive) return fail("archive", error);
  MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new]; pd.computeFunction = function;
  pd.binaryArchives = @[archive];
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pd
      options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&error];
  if (!pipeline) return fail("pipeline", error);
  (void)pipeline; (void)M; (void)N; (void)K; (void)lda; (void)ldb; (void)cBytes;
  puts("{\"status\":0,\"load_only\":true,\"gpu_dispatched\":false}");
  return fflush(stdout) ? 1 : 0;
}

static BOOL tensorGuards(void *base, NSUInteger payload) {
  uint8_t *bytes = base;
  for (NSUInteger i = 0; i < 128; ++i)
    if (bytes[i] != 0xA5 || bytes[128 + payload + i] != 0xA5) return NO;
  return YES;
}

static int runTensorDispatch(NSString *dir, NSDictionary *manifest, NSUInteger limit) {
  if (!tensorImageFiles(dir)) return fail("image_files", nil);
  const char *why = NULL; NSUInteger M, N, K, lda, ldb, ldc, aBytes, bBytes, cBytes; NSString *name = nil;
  BOOL multigemm = NO;
  if (!tensorContract(manifest, &M, &N, &K, &lda, &ldb, &ldc, &aBytes, &bBytes, &cBytes, &name, &multigemm, &why))
    return fail(why, nil);
  BOOL continuation = [name isEqual:@"tensor_transformer_continuation_weight_offset_runtime_demo"];
  // tensorContract admitted this width: 32 for every class, 32*G only for the grid class.
  BOOL generic = [name isEqual:@"tensor_gemm_generic_runtime_demo"];
  NSUInteger launchThreads = ([name isEqual:@"tensor_gemm_grid_runtime_demo"] || generic)
      ? [manifest[@"tensor"][@"grid"][0] unsignedIntegerValue] : 32;
  // tensorContract admitted this width: 32 for every class but the generic one's 32 * simdgroups
  NSUInteger groupThreads = generic ? [manifest[@"tensor"][@"threadgroup"][0] unsignedIntegerValue] : 32;
  NSData *a = boundedInput([dir stringByAppendingPathComponent:(continuation ? @"a.f32" : @"a.f16")], aBytes);
  NSData *b = boundedInput([dir stringByAppendingPathComponent:@"b.f16"], bBytes);
  NSData *c = boundedInput([dir stringByAppendingPathComponent:@"c.f32"], cBytes);
  if (!a || !b || !c) return fail("input_size", nil);
  NSError *error = nil; id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) return fail("device", nil);
  id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:
      [dir stringByAppendingPathComponent:@"scan.lib.metallib"]] error:&error];
  if (!library) return fail("library", error);
  id<MTLFunction> function = [library newFunctionWithName:name]; if (!function) return fail("function", nil);
  MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
  ad.url = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"scan.arc.metallib"]];
  id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:ad error:&error];
  if (!archive) return fail("archive", error);
  MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new]; pd.computeFunction = function;
  pd.binaryArchives = @[archive];
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pd
      options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&error];
  if (!pipeline) return fail("pipeline", error);
  id<MTLCommandQueue> queue = [device newCommandQueue]; if (!queue) return fail("command_queue", nil);
  id<MTLBuffer> ba = [device newBufferWithLength:aBytes + 256 options:MTLResourceStorageModeShared];
  id<MTLBuffer> bb = [device newBufferWithLength:bBytes + 256 options:MTLResourceStorageModeShared];
  id<MTLBuffer> bc = [device newBufferWithLength:cBytes + 256 options:MTLResourceStorageModeShared];
  if (!ba || !bb || !bc) return fail("allocation", nil);
  memset(ba.contents, 0xA5, aBytes + 256); memset(bb.contents, 0xA5, bBytes + 256); memset(bc.contents, 0xA5, cBytes + 256);
  memcpy((uint8_t *)ba.contents + 128, a.bytes, aBytes);
  memcpy((uint8_t *)bb.contents + 128, b.bytes, bBytes);
  NSDictionary *identity = @{ @"pipeline_builds": @1, @"bindings": manifest[@"abi"][@"bindings"],
      @"buffer_offsets": @[@128, @128, @128], @"buffer_payload_bytes": @[@(aBytes), @(bBytes), @(cBytes)],
      @"grid": @[@(launchThreads), @1, @1], @"threadgroup": @[@(groupThreads), @1, @1] };
  if (!reply(@{@"protocol": @3, @"sequence": @0, @"bytes": @0, @"rows": @(M), @"columns": @(N), @"identity": identity}, NULL, 0))
    return fail("handshake", nil);
  // Set A item 12: the (130, 164, 165) tensor set stages through an explicit imageblock. A pipeline
  // built by name does not size the tile (tools/g17ibrun.m), so the threadgroup's is set here, and
  // for the undeclared control too, as g17ibrun's control arm does: the declaration is the only
  // variable. The bytes the pipeline asks for are logged - 0 means nothing was declared.
  BOOL stagesImageblock = [manifest[@"abi"][@"system_registers"] isEqual:@[@130, @164, @165]];
  if (stagesImageblock) {
    fprintf(stderr, "imageblock_bytes=%lu\n", (unsigned long)[pipeline imageblockMemoryLengthForDimensions:
                                                               MTLSizeMake(groupThreads, 1, 1)]);
    fflush(stderr);
  }
  for (NSUInteger sequence = 1; sequence <= limit; ++sequence) {
    memcpy((uint8_t *)bc.contents + 128, c.bytes, cBytes);
    id<MTLCommandBuffer> cb = g17_gpu_cb(queue); id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!cb || !enc) return fail("command", nil);
    [enc setComputePipelineState:pipeline];
    // ONLY WHEN THE ABI DECLARES ONE: sizing the tile for an undeclared arm gave it a working
    // tile regardless (or something lane-private served it); either way the control stopped being
    // a control (Set C, 2026-09-23). An undeclared arm is now a plain tensor dispatch.
    if (stagesImageblock && [manifest[@"abi"][@"imageblock"] isKindOfClass:[NSDictionary class]])
      [enc setImageblockWidth:groupThreads height:1];
    [enc setBuffer:ba offset:128 atIndex:1]; [enc setBuffer:bb offset:128 atIndex:2]; [enc setBuffer:bc offset:128 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(launchThreads, 1, 1) threadsPerThreadgroup:MTLSizeMake(groupThreads, 1, 1)]; [enc endEncoding];
    fprintf(stderr, "phase=submitting sequence=%lu\n", (unsigned long)sequence); fflush(stderr);
    [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted || cb.error) return fail("command_status", cb.error);
    if (!tensorGuards(ba.contents, aBytes) || !tensorGuards(bb.contents, bBytes) || !tensorGuards(bc.contents, cBytes))
      return fail("boundary_guard", nil);
    if (memcmp((uint8_t *)ba.contents + 128, a.bytes, aBytes) || memcmp((uint8_t *)bb.contents + 128, b.bytes, bBytes))
      return fail("readonly_inputs", nil);
    if (!reply(@{@"protocol": @3, @"sequence": @(sequence), @"bytes": @(cBytes), @"status": @0,
                  @"boundary_guard": @YES, @"readonly_inputs": @YES, @"gpu_dispatched": @YES,
                  @"identity": identity, @"gpu_seconds": @(cb.GPUEndTime-cb.GPUStartTime)},
               (uint8_t *)bc.contents + 128, cBytes)) return fail("reply", nil);
  }
  return 0;
}

static BOOL sequenceImageFiles(NSString *dir, NSString *subdir) {
  NSString *where = [dir stringByAppendingPathComponent:subdir];
  for (NSString *file in @[@"scan.arc.metallib", @"scan.lib.metallib", @"scan.o", @"program.bin"]) {
    NSDictionary *attributes = [[NSFileManager defaultManager]
      attributesOfItemAtPath:[where stringByAppendingPathComponent:file] error:nil];
    unsigned long long size = [attributes fileSize];
    if (![attributes[NSFileType] isEqual:NSFileTypeRegular] || !size || size > 16*1024*1024) return NO;
  }
  return YES;
}

static id<MTLComputePipelineState> sequencePipeline(id<MTLDevice> device, NSString *dir,
                                                     NSString *subdir, NSDictionary *manifest,
                                                     NSError **error) {
  NSString *where = [dir stringByAppendingPathComponent:subdir];
  id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:
      [where stringByAppendingPathComponent:@"scan.lib.metallib"]] error:error];
  if (!library) return nil;
  id<MTLFunction> function = [library newFunctionWithName:manifest[@"name"]];
  if (!function) return nil;
  MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
  ad.url = [NSURL fileURLWithPath:[where stringByAppendingPathComponent:@"scan.arc.metallib"]];
  id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:ad error:error];
  if (!archive) return nil;
  MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
  pd.computeFunction = function; pd.binaryArchives = @[archive];
  return [device newComputePipelineStateWithDescriptor:pd
      options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:error];
}

static BOOL tensorSequenceContract(NSDictionary *manifest, NSUInteger *layers,
                                   NSDictionary **initial, NSDictionary **continuation,
                                   const char **why) {
  *why = "tensor_sequence";
  if (![manifest isKindOfClass:[NSDictionary class]] ||
      ![manifest[@"format"] isEqual:@"g17-common-sequence-v1"] ||
      ![manifest[@"kind"] isEqual:@"tensor_gemm_sequence"] ||
      ![manifest[@"name"] isEqual:@"tensor_transformer_layer_sequence_runtime_demo"] ||
      ![manifest[@"weight_slot"] isEqual:@2] ||
      ![manifest[@"weight_regions"] isEqual:@[@0, @4096, @8192]] ||
      ![manifest[@"distinct_weight_buffers"] isEqual:manifest[@"layers"]]) {
    *why = "tensor_sequence_manifest"; return NO;
  }
  NSUInteger n = 0;
  if (!integer(manifest[@"layers"], 6, &n) || (n != 2 && n != 6)) { *why = "tensor_sequence_layers"; return NO; }
  NSDictionary *activation = manifest[@"activation"];
  if (![activation isKindOfClass:[NSDictionary class]] || ![activation[@"rows"] isEqual:@16] ||
      ![activation[@"columns"] isEqual:@32] || ![activation[@"storage"] isEqual:@"float32"] ||
      ![activation[@"transport"] isEqual:@"gpu_resident_ping_pong"] ||
      ![activation[@"host_readback"] isEqual:@NO]) { *why = "tensor_sequence_activation"; return NO; }
  NSDictionary *i = manifest[@"initial"], *c = manifest[@"continuation"];
  if (![i isKindOfClass:[NSDictionary class]] || ![c isKindOfClass:[NSDictionary class]]) {
    *why = "tensor_sequence_images"; return NO;
  }
  NSUInteger M, N, K, lda, ldb, ldc, aBytes, bBytes, cBytes; NSString *name = nil; BOOL multi = NO;
  if (!tensorContract(i, &M, &N, &K, &lda, &ldb, &ldc, &aBytes, &bBytes, &cBytes, &name, &multi, why) ||
      ![i[ @"tensor"][@"composition"] isEqual:@"transformer_layer_weight_offset"] ||
      !tensorContract(c, &M, &N, &K, &lda, &ldb, &ldc, &aBytes, &bBytes, &cBytes, &name, &multi, why) ||
      ![c[@"tensor"][@"composition"] isEqual:@"transformer_continuation_weight_offset"]) return NO;
  *layers = n; *initial = i; *continuation = c; *why = NULL; return YES;
}

static int runTensorSequenceLoadOnly(NSString *dir, NSDictionary *manifest) {
  NSUInteger layers; NSDictionary *initial, *continuation; const char *why = NULL;
  if (!tensorSequenceContract(manifest, &layers, &initial, &continuation, &why)) return fail(why, nil);
  if (!sequenceImageFiles(dir, @"initial") || !sequenceImageFiles(dir, @"continuation")) return fail("image_files", nil);
  NSError *error = nil; id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) return fail("device", nil);
  if (!sequencePipeline(device, dir, @"initial", initial, &error) ||
      !sequencePipeline(device, dir, @"continuation", continuation, &error)) return fail("pipeline", error);
  puts("{\"status\":0,\"load_only\":true,\"gpu_dispatched\":false}");
  return fflush(stdout) ? 1 : 0;
}

static int runTensorSequenceDispatch(NSString *dir, NSDictionary *manifest, NSUInteger limit) {
  NSUInteger layers; NSDictionary *initial, *continuation; const char *why = NULL;
  if (!tensorSequenceContract(manifest, &layers, &initial, &continuation, &why)) return fail(why, nil);
  if (!sequenceImageFiles(dir, @"initial") || !sequenceImageFiles(dir, @"continuation")) return fail("image_files", nil);
  NSData *a = boundedInput([dir stringByAppendingPathComponent:@"a.f16"], 16u * 64u * 2u);
  if (!a) return fail("input_size", nil);
  NSMutableArray *weights = [NSMutableArray array];
  for (NSUInteger index = 0; index < layers; ++index) {
    NSData *weight = boundedInput([dir stringByAppendingPathComponent:[NSString stringWithFormat:@"b%lu.f16", (unsigned long)index]], 10240);
    if (!weight) return fail("weight_input_size", nil);
    [weights addObject:weight];
  }
  NSError *error = nil; id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) return fail("device", nil);
  id<MTLComputePipelineState> first = sequencePipeline(device, dir, @"initial", initial, &error);
  id<MTLComputePipelineState> next = sequencePipeline(device, dir, @"continuation", continuation, &error);
  if (!first || !next) return fail("pipeline", error);
  id<MTLCommandQueue> queue = [device newCommandQueue]; if (!queue) return fail("command_queue", nil);
  id<MTLBuffer> ba = [device newBufferWithLength:2048 + 256 options:MTLResourceStorageModeShared];
  id<MTLBuffer> c0 = [device newBufferWithLength:2048 + 256 options:MTLResourceStorageModeShared];
  id<MTLBuffer> c1 = [device newBufferWithLength:2048 + 256 options:MTLResourceStorageModeShared];
  NSMutableArray *bbs = [NSMutableArray array];
  for (NSUInteger weightCount = 0; weightCount < weights.count; ++weightCount) {
    id<MTLBuffer> bb = [device newBufferWithLength:10240 + 256 options:MTLResourceStorageModeShared];
    if (!bb) return fail("allocation", nil); [bbs addObject:bb];
  }
  if (!ba || !c0 || !c1) return fail("allocation", nil);
  memset(ba.contents, 0xA5, 2304); memcpy((uint8_t *)ba.contents + 128, a.bytes, 2048);
  for (id<MTLBuffer> out in @[c0, c1]) memset(out.contents, 0xA5, 2304);
  for (NSUInteger index = 0; index < layers; ++index) {
    id<MTLBuffer> bb = bbs[index]; memset(bb.contents, 0xA5, 10496);
    memcpy((uint8_t *)bb.contents + 128, [weights[index] bytes], 10240);
  }
  NSDictionary *identity = @{ @"layer_count": @(layers), @"weight_slot": @2,
      @"weight_regions": @[@0, @4096, @8192], @"activation_transport": @"gpu_resident_ping_pong",
      @"buffer_offsets": @[@128, @128, @128], @"gpu_intermediate_readback": @NO };
  if (!reply(@{@"protocol": @3, @"sequence": @0, @"bytes": @0, @"rows": @16, @"columns": @32,
                @"layer_count": @(layers), @"identity": identity}, NULL, 0)) return fail("handshake", nil);
  for (NSUInteger query = 1; query <= limit; ++query) {
    memset(c0.contents, 0xA5, 2304); memset(c1.contents, 0xA5, 2304);
    id<MTLCommandBuffer> cb = g17_gpu_cb(queue); if (!cb) return fail("command", nil);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:first]; [enc setBuffer:ba offset:128 atIndex:1]; [enc setBuffer:bbs[0] offset:128 atIndex:2]; [enc setBuffer:c0 offset:128 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding];
    id<MTLBuffer> current = c0;
    for (NSUInteger layer = 1; layer < layers; ++layer) {
      id<MTLBuffer> output = (layer & 1) ? c1 : c0;
      enc = [cb computeCommandEncoder]; [enc setComputePipelineState:next];
      [enc setBuffer:current offset:128 atIndex:1]; [enc setBuffer:bbs[layer] offset:128 atIndex:2]; [enc setBuffer:output offset:128 atIndex:3];
      [enc dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [enc endEncoding]; current = output;
    }
    fprintf(stderr, "phase=submitting sequence=%lu layers=%lu\n", (unsigned long)query, (unsigned long)layers); fflush(stderr);
    [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted || cb.error) return fail("command_status", cb.error);
    if (!tensorGuards(ba.contents, 2048) || !tensorGuards(c0.contents, 2048) || !tensorGuards(c1.contents, 2048)) return fail("boundary_guard", nil);
    for (id<MTLBuffer> bb in bbs) if (!tensorGuards(bb.contents, 10240)) return fail("boundary_guard", nil);
    if (memcmp((uint8_t *)ba.contents + 128, a.bytes, 2048)) return fail("readonly_activation", nil);
    for (NSUInteger index = 0; index < layers; ++index)
      if (memcmp((uint8_t *)[bbs[index] contents] + 128, [weights[index] bytes], 10240)) return fail("readonly_weights", nil);
    if (!reply(@{@"protocol": @3, @"sequence": @(query), @"bytes": @2048, @"status": @0,
                  @"layer_count": @(layers), @"boundary_guard": @YES, @"readonly_inputs": @YES,
                  @"gpu_dispatched": @YES, @"gpu_intermediate_readback": @NO, @"identity": identity,
                  @"gpu_seconds": @(cb.GPUEndTime-cb.GPUStartTime)},
               (uint8_t *)current.contents + 128, 2048)) return fail("reply", nil);
  }
  return 0;
}

// The requantization stage is a separate measured runtime kind.  It is intentionally not folded
// into tensorContract: the high-level TensorOps API requires a real dispatch boundary between an
// int32-producing GEMM and the next int8/uint8 GEMM, and the scalar metadata class is SR160 with
// three bindings rather than the tensor SR130/SR156 class.  This worker checks that distinction
// before creating any Metal object.
static BOOL requantContract(NSDictionary *manifest, NSUInteger *payloadSource,
                            NSUInteger *payloadScale, NSUInteger *payloadDestination,
                            NSString **nameOut, const char **why) {
  *why = "requant_contract";
  if (![manifest isKindOfClass:[NSDictionary class]] ||
      ![manifest[@"format"] isEqual:@"g17-common-pipeline-v4"] ||
      ![manifest[@"kind"] isEqual:@"tensor_requantization"] ||
      ![manifest[@"name"] isEqual:@"requant_scalar_stage"]) {
    *why = "requant_manifest_format"; return NO;
  }
  NSDictionary *shape = manifest[@"shape"], *spec = manifest[@"requantization"];
  NSDictionary *abi = manifest[@"abi"], *marker = [abi isKindOfClass:[NSDictionary class]] ? abi[@"requantization"] : nil;
  if (![shape isKindOfClass:[NSDictionary class]] || ![shape[@"rows"] isEqual:@256] ||
      ![shape[@"columns"] isEqual:@1] || ![spec isKindOfClass:[NSDictionary class]] ||
      ![spec[@"elements"] isEqual:@256] || ![spec[@"groups"] isEqual:@8] ||
      ![spec[@"grid"] isEqual:@[@256, @1, @1]] ||
      ![spec[@"threadgroup"] isEqual:@[@32, @1, @1]] ||
      ![spec[@"dispatch_boundary"] isEqual:@"required"] ||
      (![spec[@"scale"] isEqual:@"1/512"] && ![spec[@"scale"] isEqual:@"1/16"]) ||
      (![spec[@"saturation"] isEqual:@"signed_int8"] && ![spec[@"saturation"] isEqual:@"unsigned_uint8"]) ||
      ([spec[@"saturation"] isEqual:@"unsigned_uint8"] && ![spec[@"scale"] isEqual:@"1/16"]) ||
      ![abi isKindOfClass:[NSDictionary class]] || ![marker isKindOfClass:[NSDictionary class]]) {
    *why = "requant_shape"; return NO;
  }
  if ((![marker[@"kind"] isEqual:@"int32_to_int8"] && ![marker[@"kind"] isEqual:@"int32_to_uint8"]) ||
      ![marker[@"metadata_class"] isEqual:@"scalar_476"] ||
      ![marker[@"dispatch_boundary"] isEqual:@"required"] ||
      ![marker[@"elements"] isEqual:@256] || ![marker[@"groups"] isEqual:@8] ||
      ![marker[@"rounding"] isEqual:@"round_half_to_even"] ||
      ![marker[@"scale_binding"] isEqual:@1] || ![marker[@"source_binding"] isEqual:@0] ||
      ![marker[@"destination_binding"] isEqual:@2] ||
      ![marker[@"scale_storage"] isEqual:@"uint32_bits"] ||
      ![marker[@"output_storage"] isEqual:@"int32_word"] ||
      ![marker[@"scale_addressing"] isEqual:@"constant_program_preload"] ||
      ![marker[@"scale"] isEqual:spec[@"scale"]] ||
      ![marker[@"saturation"] isEqual:spec[@"saturation"]] ||
      (![spec[@"saturation"] isEqual:@"unsigned_uint8"] && ![marker[@"kind"] isEqual:@"int32_to_int8"]) ||
      ([spec[@"saturation"] isEqual:@"unsigned_uint8"] && ![marker[@"kind"] isEqual:@"int32_to_uint8"])) {
    *why = "requant_marker"; return NO;
  }
  NSUInteger version, entry;
  if (!integer(abi[@"abi_version"], 5, &version) || version != 3 ||
      !integer(abi[@"entry"], 64, &entry) || entry != 64 ||
      !boolean(abi[@"arch_flag"], YES) || !boolean(abi[@"uses_threadgroup"], NO) ||
      !boolean(abi[@"writes_buffer"], YES) || !boolean(abi[@"has_stores"], YES) ||
      !boolean(abi[@"writes_texture"], NO)) { *why = "requant_abi_semantics"; return NO; }
  if (![abi[@"system_registers"] isEqual:@[@160]]) { *why = "requant_system_registers"; return NO; }
  NSDictionary *launch = abi[@"launch"];
  if (![launch isKindOfClass:[NSDictionary class]] || !boolean(launch[@"bounds_checked"], NO) ||
      !boolean(launch[@"exact_grid_required"], YES)) { *why = "requant_launch"; return NO; }
  // The scalar requantization class carries the retained 64-byte constant program: two pointer
  // loads, the scale-word load, the publish op, END and measured filler.  This is the worker's
  // wire check for the same bytes validated by agxforge.g17.requantpreload; it is not a generic
  // non-trivial-prologue admission.
  NSString *expected = @"248021104701a0821c8a08270f000300804400a04100800000002304070256a0a41a0e0000000600060006000600060006000600060006000600060006000600";
  if (![abi[@"prologue"] isKindOfClass:[NSString class]] || ![abi[@"prologue"] isEqual:expected]) {
    *why = "requant_prologue"; return NO;
  }
  NSArray *bindings = abi[@"bindings"];
  if (![bindings isKindOfClass:[NSArray class]] || bindings.count != 3) {
    *why = "requant_bindings"; return NO;
  }
  NSArray *indices = @[@0, @1, @2], *offsets = @[@0, @2, @4];
  NSArray *types = @[@"uint", @"uint", @"uint"], *widths = @[@4, @4, @4];
  NSArray *writes = @[@NO, @NO, @YES];
  for (NSUInteger i = 0; i < 3; ++i) {
    NSDictionary *binding = bindings[i]; NSUInteger value;
    if (![binding isKindOfClass:[NSDictionary class]] || ![binding[@"index"] isEqual:indices[i]] ||
        ![binding[@"offset"] isEqual:offsets[i]] || ![binding[@"element_type"] isEqual:types[i]] ||
        ![binding[@"element_bytes"] isEqual:widths[i]] || ![binding[@"written"] isEqual:writes[i]] ||
        !integer(binding[@"index"], 30, &value)) { *why = "requant_bindings"; return NO; }
  }
  NSArray *forms = abi[@"forms"];
  if (![forms isKindOfClass:[NSArray class]] || !forms.count) { *why = "requant_forms"; return NO; }
  *payloadSource = 1024; *payloadScale = 1024; *payloadDestination = 1024;
  *nameOut = manifest[@"name"]; *why = NULL; return YES;
}

static NSDictionary *requantLayout(NSDictionary *manifest, const char **why) {
  NSUInteger source, scale, destination; NSString *name = nil;
  if (!requantContract(manifest, &source, &scale, &destination, &name, why)) return nil;
  return @{@"function": name, @"kind": @"tensor_requantization",
           @"storage_dtype": @"requantized_int8_in_i32_words", @"rows": @256, @"columns": @1,
           @"matrix_bytes": @(source), @"request_bytes": @(source),
           @"reply_bytes": @(destination), @"reply_elements": @256,
           @"completion_markers": @NO, @"output_bytes": @(destination + 256),
           @"buffer_allocations": @3,
           @"buffer_roles": @[@"accumulator", @"scale", @"output"],
           @"buffer_payload_bytes": @[@(source), @(scale), @(destination)],
           @"buffer_allocation_bytes": @[@(source + 256), @(scale + 256), @(destination + 256)],
           @"buffer_offsets": @[@128, @128, @128], @"protocol": @4,
           @"bindings": manifest[@"abi"][@"bindings"], @"gpu_dispatched": @NO,
           @"requantization": manifest[@"requantization"]};
}

static int runRequantLoadOnly(NSString *dir, NSDictionary *manifest) {
  if (!tensorImageFiles(dir)) return fail("image_files", nil);
  const char *why = NULL; NSUInteger source, scale, destination; NSString *name = nil;
  if (!requantContract(manifest, &source, &scale, &destination, &name, &why))
    return fail(why, nil);
  NSError *error = nil; id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) return fail("device", nil);
  id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:
      [dir stringByAppendingPathComponent:@"scan.lib.metallib"]] error:&error];
  if (!library) return fail("library", error);
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (!function) return fail("function", nil);
  MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
  ad.url = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"scan.arc.metallib"]];
  id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:ad error:&error];
  if (!archive) return fail("archive", error);
  MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new]; pd.computeFunction = function;
  pd.binaryArchives = @[archive];
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pd
      options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&error];
  if (!pipeline) return fail("pipeline", error);
  (void)pipeline; puts("{\"status\":0,\"load_only\":true,\"gpu_dispatched\":false}");
  return fflush(stdout) ? 1 : 0;
}

static int runRequantDispatch(NSString *dir, NSDictionary *manifest, NSUInteger limit) {
  if (!tensorImageFiles(dir)) return fail("image_files", nil);
  const char *why = NULL; NSUInteger source, scale, destination; NSString *name = nil;
  if (!requantContract(manifest, &source, &scale, &destination, &name, &why))
    return fail(why, nil);
  NSData *accumulator = boundedInput([dir stringByAppendingPathComponent:@"acc.i32"], source);
  NSData *scaleData = boundedInput([dir stringByAppendingPathComponent:@"scale.bits"], scale);
  NSData *initial = boundedInput([dir stringByAppendingPathComponent:@"output.i32"], destination);
  if (!accumulator || !scaleData || !initial) return fail("input_size", nil);
  NSError *error = nil; id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) return fail("device", nil);
  id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:
      [dir stringByAppendingPathComponent:@"scan.lib.metallib"]] error:&error];
  if (!library) return fail("library", error);
  id<MTLFunction> function = [library newFunctionWithName:name]; if (!function) return fail("function", nil);
  MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
  ad.url = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"scan.arc.metallib"]];
  id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:ad error:&error];
  if (!archive) return fail("archive", error);
  MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new]; pd.computeFunction = function;
  pd.binaryArchives = @[archive];
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pd
      options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&error];
  if (!pipeline) return fail("pipeline", error);
  id<MTLCommandQueue> queue = [device newCommandQueue]; if (!queue) return fail("command_queue", nil);
  id<MTLBuffer> ba = [device newBufferWithLength:source + 256 options:MTLResourceStorageModeShared];
  id<MTLBuffer> bs = [device newBufferWithLength:scale + 256 options:MTLResourceStorageModeShared];
  id<MTLBuffer> bd = [device newBufferWithLength:destination + 256 options:MTLResourceStorageModeShared];
  if (!ba || !bs || !bd) return fail("allocation", nil);
  memset(ba.contents, 0xA5, source + 256); memset(bs.contents, 0xA5, scale + 256);
  memset(bd.contents, 0xA5, destination + 256);
  memcpy((uint8_t *)ba.contents + 128, accumulator.bytes, source);
  memcpy((uint8_t *)bs.contents + 128, scaleData.bytes, scale);
  NSDictionary *identity = @{ @"pipeline_builds": @1, @"bindings": manifest[@"abi"][@"bindings"],
      @"buffer_offsets": @[@128, @128, @128],
      @"buffer_payload_bytes": @[@(source), @(scale), @(destination)],
      @"grid": @[@256, @1, @1], @"threadgroup": @[@32, @1, @1],
      @"dispatch_boundary": @"required" };
  if (!reply(@{@"protocol": @4, @"sequence": @0, @"bytes": @0,
               @"rows": @256, @"columns": @1, @"identity": identity}, NULL, 0))
    return fail("handshake", nil);
  for (NSUInteger sequence = 1; sequence <= limit; ++sequence) {
    memcpy((uint8_t *)bd.contents + 128, initial.bytes, destination);
    id<MTLCommandBuffer> cb = g17_gpu_cb(queue); id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!cb || !enc) return fail("command", nil);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:ba offset:128 atIndex:0]; [enc setBuffer:bs offset:128 atIndex:1];
    [enc setBuffer:bd offset:128 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    fprintf(stderr, "phase=submitting sequence=%lu\\n", (unsigned long)sequence); fflush(stderr);
    [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted || cb.error) return fail("command_status", cb.error);
    if (!tensorGuards(ba.contents, source) || !tensorGuards(bs.contents, scale) ||
        !tensorGuards(bd.contents, destination)) return fail("boundary_guard", nil);
    if (memcmp((uint8_t *)ba.contents + 128, accumulator.bytes, source) ||
        memcmp((uint8_t *)bs.contents + 128, scaleData.bytes, scale)) return fail("readonly_inputs", nil);
    if (!reply(@{@"protocol": @4, @"sequence": @(sequence), @"bytes": @(destination),
                  @"status": @0, @"boundary_guard": @YES, @"readonly_inputs": @YES,
                  @"gpu_dispatched": @YES, @"identity": identity,
                  @"gpu_seconds": @(cb.GPUEndTime-cb.GPUStartTime)},
               (uint8_t *)bd.contents + 128, destination)) return fail("reply", nil);
  }
  return 0;
}

int main(int argc, const char **argv) {
  @autoreleasepool {
    if (argc == 3 && (!strcmp(argv[2], "--texture-inputs-admit") || !strcmp(argv[2], "--texture-query-admit"))) {
      NSString *dir = [NSString stringWithUTF8String:argv[1]];
      NSData *raw = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"manifest.json"]];
      if (!raw) return fail("manifest", nil);
      NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:raw options:0 error:nil];
      if (![manifest isKindOfClass:[NSDictionary class]]) return fail("manifest_format", nil);
      G17TexturePlan plans[2];
      NSMutableArray *payloads = [NSMutableArray array];
      const char *why = NULL;
      NSArray *identities = textureInputs(dir, manifest, plans, payloads, &why);
      if (!identities) return fail(why ? why : "texture_inputs", nil);
      NSArray *coordinateInputs = textureCoordinateInputs(manifest, plans, identities.count, &why);
      if (!coordinateInputs) return fail(why, nil);
      if (!strcmp(argv[2], "--texture-query-admit")) {
        NSUInteger expected, slot;
        if (!textureQueryContract(manifest[@"query"], coordinateInputs.count != 0, &expected, &slot))
          return fail("texture_query_contract", nil);
      }
      NSData *json = [NSJSONSerialization dataWithJSONObject:
          @{@"status": @0, @"admitted": @YES, @"gpu_dispatched": @NO, @"textures": identities}
          options:0 error:nil];
      if (!json || fwrite(json.bytes, 1, json.length, stdout) != json.length) return fail("reply", nil);
      putchar('\n');
      return 0;
    }
    // HOST-ONLY ADMISSION, no device and no Metal object of any kind: the four controls
    // integration named are contract questions and are answered as contract questions.
    //   worker <dir> --texture-admit <width> <height> <payload-bytes> <dense-index>
    if (argc == 7 && (!strcmp(argv[2], "--texture-admit") ||
                      !strcmp(argv[2], "--texture-pair-admit"))) {
      NSString *where = [NSString stringWithUTF8String:argv[1]];
      NSData *blob = [NSData dataWithContentsOfFile:
          [where stringByAppendingPathComponent:@"manifest.json"]];
      if (!blob) return fail("manifest", nil);
      NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:blob options:0 error:nil];
      if (![manifest isKindOfClass:[NSDictionary class]]) return fail("manifest_format", nil);
      NSDictionary *abiHere = manifest[@"abi"];
      NSDictionary *resources = [abiHere isKindOfClass:[NSDictionary class]]
          ? abiHere[@"resources"] : nil;
      G17TexturePlan plan;
      const char *why = NULL;
      char *end = NULL;
      unsigned long values[4];
      for (int i = 0; i < 4; ++i) {
        values[i] = strtoul(argv[3 + i], &end, 10);
        if (!end || *end) return fail("texture_argument", nil);
      }
      NSUInteger requiredCount = !strcmp(argv[2], "--texture-pair-admit") ? 2 : 1;
      if (!textureAdmitCount(resources, requiredCount, values[0], values[1], values[2], values[3], &plan, &why))
        return fail(why ? why : "texture_contract", nil);
      printf("{\"status\":0,\"admitted\":true,\"gpu_dispatched\":false,\"width\":%lu,"
             "\"height\":%lu,\"bytes_per_row\":%lu,\"payload_bytes\":%lu,"
             "\"dense_index\":%lu}\n",
             (unsigned long)plan.width, (unsigned long)plan.height,
             (unsigned long)plan.bytesPerRow, (unsigned long)plan.payloadBytes,
             (unsigned long)plan.denseIndex);
      return 0;
    }
    BOOL describe = argc == 3 && !strcmp(argv[2], "--describe-layout");
    BOOL fullLoad = argc == 3 && !strcmp(argv[2], "--full-load-approved");
    BOOL tensorLoad = argc == 3 && !strcmp(argv[2], "--tensor-load-approved");
    BOOL tensorSequenceLoad = argc == 3 && !strcmp(argv[2], "--tensor-sequence-load-approved");
    BOOL requantLoad = argc == 3 && !strcmp(argv[2], "--requant-load-approved");
    BOOL loadOnly = fullLoad || tensorLoad || tensorSequenceLoad || requantLoad ||
                    (argc == 3 && !strcmp(argv[2], "--load-approved"));
    BOOL fullDispatch = argc == 5 && !strcmp(argv[4], "--full-dispatch-approved");
    BOOL textureDispatch = argc == 5 && !strcmp(argv[4], "--texture-dispatch-approved");
    BOOL tensorDispatch = argc == 5 && !strcmp(argv[4], "--tensor-dispatch-approved");
    BOOL tensorSequenceDispatch = argc == 5 && !strcmp(argv[4], "--tensor-sequence-dispatch-approved");
    BOOL requantDispatch = argc == 5 && !strcmp(argv[4], "--requant-dispatch-approved");
    BOOL dispatch = fullDispatch || textureDispatch || tensorDispatch || tensorSequenceDispatch || requantDispatch ||
                    (argc == 5 && !strcmp(argv[4], "--dispatch-approved"));
    if (!describe && !loadOnly && !dispatch) return fail("usage", nil);
    unsigned long limit = 0;
    if (dispatch) {
      char *end = NULL;
      limit = strtoul(argv[3], &end, 10);
      if (!end || *end || limit < 1 || limit > 100) return fail("query_limit", nil);
    }
    NSString *dir = [NSString stringWithUTF8String:argv[1]];
    NSData *raw = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"manifest.json"]];
    NSError *error = nil;
    if (!raw) return fail("manifest", nil);
    NSDictionary *m = [NSJSONSerialization JSONObjectWithData:raw options:0 error:&error];
    if (![m isKindOfClass:[NSDictionary class]]) return fail("manifest_format", error);
    NSString *kind = m[@"kind"];
    if ([kind isEqual:@"tensor_gemm_sequence"]) {
      if (describe || tensorLoad || tensorDispatch || requantLoad || requantDispatch) return fail("tensor_sequence_mode", nil);
      if (tensorSequenceLoad) return runTensorSequenceLoadOnly(dir, m);
      if (tensorSequenceDispatch) return runTensorSequenceDispatch(dir, m, limit);
      return fail("tensor_sequence_mode", nil);
    }
    if ([kind isEqual:@"tensor_gemm"]) {
      const char *why = NULL;
      if (describe) {
        NSDictionary *layout = tensorLayout(m, &why);
        if (!layout) return fail(why ? why : "tensor_contract", nil);
        NSData *json = [NSJSONSerialization dataWithJSONObject:layout options:0 error:nil];
        if (!json || fwrite(json.bytes, 1, json.length, stdout) != json.length) return fail("reply", nil);
        putchar('\n'); return 0;
      }
      if (tensorLoad) return runTensorLoadOnly(dir, m);
      if (tensorDispatch) return runTensorDispatch(dir, m, limit);
      return fail("tensor_mode", nil);
    }
    if ([kind isEqual:@"tensor_requantization"]) {
      const char *why = NULL;
      if (describe) {
        NSDictionary *layout = requantLayout(m, &why);
        if (!layout) return fail(why ? why : "requant_contract", nil);
        NSData *json = [NSJSONSerialization dataWithJSONObject:layout options:0 error:nil];
        if (!json || fwrite(json.bytes, 1, json.length, stdout) != json.length) return fail("reply", nil);
        putchar('\n'); return 0;
      }
      if (requantLoad) return runRequantLoadOnly(dir, m);
      if (requantDispatch) return runRequantDispatch(dir, m, limit);
      return fail("requant_mode", nil);
    }
    BOOL textureRead = [kind isEqual:@"texture_read"];
    if (textureRead) {
      NSDictionary *textureABI = m[@"abi"];
      NSUInteger textureVersion;
      if (![m[@"format"] isEqual:@"g17-common-pipeline-v1"] ||
          ![m[@"name"] isEqual:@"texture_read"])
        return fail("manifest_format", nil);
      if (![textureABI isKindOfClass:[NSDictionary class]] ||
          !integer(textureABI[@"abi_version"], 7, &textureVersion) ||
          (textureVersion != 6 && textureVersion != 7) ||
          ![textureABI[@"resources"] isKindOfClass:[NSDictionary class]])
        return fail("texture_abi", nil);
      if (textureDispatch) return runTextureDispatch(dir, m, textureABI, limit);
      if (!loadOnly) return fail("texture_dispatch_unimplemented", nil);
      return runTextureLoadOnly(dir, m, textureABI);
    }
    BOOL packed = [kind isEqual:@"packed_scan"], separate = [kind isEqual:@"separate_scan"];
    BOOL affine = [kind isEqual:@"affine"], layernorm = [kind isEqual:@"layernorm"];
    BOOL queryProjection = [kind isEqual:@"minilm_query"];
    BOOL fp32 = layernorm || queryProjection;
    if (!packed && !separate && !affine && !fp32) return fail("program_kind", nil);
    if (![m[@"format"] isEqual:fp32 ? @"g17-common-pipeline-v2" : @"g17-common-pipeline-v1"])
      return fail("manifest_format", nil);
    NSString *name = queryProjection ? @"minilm_layer0_query" : layernorm ? @"minilm_layernorm" : affine ? @"half_affine" :
      separate ? @"half_scan_separate" : @"half_scan";
    if (![m[@"name"] isEqual:name]) return fail("function_name", nil);
    NSDictionary *shape = m[@"shape"], *abi = m[@"abi"];
    if (![shape isKindOfClass:[NSDictionary class]] || ![abi isKindOfClass:[NSDictionary class]])
      return fail("contract", nil);
    NSUInteger rows, columns, version, entry;
    if (!integer(shape[@"rows"], 500000, &rows) || !rows ||
        !integer(shape[@"columns"], 384, &columns) || !columns || (affine && columns != 1) ||
        (fp32 && rows > 128) || (queryProjection && columns != 384))
      return fail("shape", nil);
    BOOL full = packed && rows == 500000 && columns == 384;
    if (!describe && rows > 128 && !(full && (fullLoad || fullDispatch)))
      return fail("staged_shape_limit", nil);
    if ((fullLoad || fullDispatch) && !full) return fail("full_stage_shape", nil);
    NSDictionary *launch = abi[@"launch"];
    if (!integer(abi[@"abi_version"], 3, &version) || (version != 2 && version != 3) ||
        (fp32 && version != 3) ||
        !integer(abi[@"entry"], 64, &entry) || entry != 64 ||
        !boolean(abi[@"arch_flag"], YES) || !boolean(abi[@"uses_threadgroup"], NO) ||
        !boolean(abi[@"writes_buffer"], YES) || !boolean(abi[@"has_stores"], YES) ||
        !boolean(abi[@"writes_texture"], NO) || ![launch isKindOfClass:[NSDictionary class]] ||
        !boolean(launch[@"bounds_checked"], NO) || !boolean(launch[@"exact_grid_required"], YES))
      return fail("abi_semantics", nil);
    if (version == 3) {
      NSArray *registers = abi[@"system_registers"];
      NSUInteger reg;
      if (![registers isKindOfClass:[NSArray class]] || registers.count != (queryProjection ? 2 : 1) ||
          !integer(registers[0], 255, &reg) || reg != 160)
        return fail("system_registers", nil);
      if (queryProjection && (!integer(registers[1], 255, &reg) || reg != 161))
        return fail("system_registers", nil);
    }
    NSMutableString *prologue = [NSMutableString stringWithString:@"0e000000"];
    for (NSUInteger i = 4; i < 64; i += 2) [prologue appendString:@"0600"];
    if (![abi[@"prologue"] isEqual:prologue]) return fail("prologue", nil);
    NSArray *forms = abi[@"forms"], *extra = abi[@"pk_extra"];
    NSDictionary *pk = abi[@"pk_values"];
    NSUInteger firstFlag, secondFlag, firstValue, secondValue;
    if (![extra isKindOfClass:[NSArray class]] || extra.count != 2 ||
        !integer(extra[0], 16, &firstFlag) || firstFlag != 15 ||
        !integer(extra[1], 16, &secondFlag) || secondFlag != 16 ||
        ![pk isKindOfClass:[NSDictionary class]] || pk.count != 2 ||
        !integer(pk[@"15"], 1, &firstValue) || firstValue != 1 ||
        !integer(pk[@"16"], 1, &secondValue) || secondValue != 1)
      return fail("metadata_semantics", nil);
    if (![forms isKindOfClass:[NSArray class]] || !forms.count) return fail("forms", nil);
    NSUInteger previousOp = 0, previousLength = 0;
    for (NSUInteger i = 0; i < forms.count; ++i) {
      NSArray *form = forms[i];
      NSUInteger op, length;
      if (![form isKindOfClass:[NSArray class]] || form.count != 2 ||
          !integer(form[0], 65535, &op) || !integer(form[1], 32, &length) ||
          length < 2 || length % 2 ||
          (i && (op < previousOp || (op == previousOp && length <= previousLength))))
        return fail("forms", nil);
      previousOp = op; previousLength = length;
    }
    NSArray *bindings = abi[@"bindings"];
    NSUInteger count = fp32 ? 4 : separate ? 3 : 2, indices[4] = {0};
    if (![bindings isKindOfClass:[NSArray class]] || bindings.count != count)
      return fail("binding_count", nil);
    for (NSUInteger i = 0; i < count; ++i) {
      NSDictionary *b = bindings[i];
      NSUInteger offset, width;
      if (![b isKindOfClass:[NSDictionary class]] || !integer(b[@"index"], 30, &indices[i]) ||
          (i && indices[i] <= indices[i-1]) || !integer(b[@"offset"], 60, &offset) || offset != 2*i ||
          !integer(b[@"element_bytes"], 4, &width) || width != (fp32 ? 4 : 2) ||
          ![b[@"element_type"] isEqual:fp32 ? @"float" : @"half"] ||
          !boolean(b[@"written"], i == count-1))
        return fail("binding_contract", nil);
    }
    G17LayerNormStorage lnStorage = {0};
    if (fp32) {
      if (!(queryProjection ? g17QueryProjectionStorageInit(&lnStorage, rows, columns) :
                             g17LayerNormStorageInit(&lnStorage, rows, columns))) return fail("storage", nil);
      G17LayerNormStorage storage = lnStorage;
      NSUInteger matrixBytes = storage.payloadBytes[0], parameterBytes = storage.payloadBytes[1];
      NSDictionary *layout = @{@"function": name, @"kind": kind, @"storage_dtype": @"float32",
        @"rows": @(rows), @"columns": @(columns), @"matrix_bytes": @(matrixBytes),
        @"request_bytes": @(matrixBytes), @"reply_bytes": @(matrixBytes),
        @"reply_elements": @(rows*columns), @"completion_markers": @NO,
        @"output_bytes": @(storage.allocationBytes[3]), @"buffer_allocations": @4,
        @"buffer_roles": queryProjection ? @[@"source", @"weight", @"bias", @"output"] :
                                           @[@"source", @"gamma", @"beta", @"output"],
        @"buffer_payload_bytes": @[@(matrixBytes), @(parameterBytes), @(storage.payloadBytes[2]), @(matrixBytes)],
        @"buffer_allocation_bytes": @[@(storage.allocationBytes[0]), @(storage.allocationBytes[1]),
                                       @(storage.allocationBytes[2]), @(storage.allocationBytes[3])],
        @"buffer_offsets": @[@(G17_LN_GUARD), @(G17_LN_GUARD), @(G17_LN_GUARD), @(G17_LN_GUARD)],
        @"protocol": @2,
        @"bindings": bindings, @"gpu_dispatched": @NO};
      if (describe) {
        NSData *json = [NSJSONSerialization dataWithJSONObject:layout options:0 error:nil];
        fwrite(json.bytes, 1, json.length, stdout);
        return 0;
      }
      // The Python staging owner verifies the delivered instructions, image
      // sections and a fresh committed-source rebuild before invoking this
      // private worker. A shape-only layout request is still insufficient to
      // enter Metal: require every delivered image file before device creation.
      for (NSString *file in @[@"scan.arc.metallib", @"scan.lib.metallib", @"scan.o", @"program.bin"]) {
        NSDictionary *attributes = [[NSFileManager defaultManager]
          attributesOfItemAtPath:[dir stringByAppendingPathComponent:file] error:nil];
        unsigned long long size = [attributes fileSize];
        if (![attributes[NSFileType] isEqual:NSFileTypeRegular] || !size || size > 16*1024*1024)
          return fail("image_files", nil);
      }
    }
    G17Storage storage = {0};
    NSUInteger requestBytes = 0, firstBytes = 0;
    FILE *input = NULL;
    if (!fp32) {
      if (!g17StorageInit(&storage, rows, columns, true)) return fail("storage", nil);
      requestBytes = affine ? storage.matrixBytes : storage.queryBytes;
      firstBytes = storage.matrixBytes + (packed ? storage.queryBytes : 0);
      NSDictionary *layout = @{@"function": name, @"kind": kind, @"storage_dtype": @"float16",
        @"rows": @(rows), @"columns": @(columns), @"matrix_bytes": @(storage.matrixBytes),
        @"request_bytes": @(requestBytes), @"reply_bytes": @(storage.replyBytes),
        @"first_buffer_bytes": @(firstBytes), @"query_buffer_bytes": @(separate ? storage.queryBytes : 0),
        @"output_bytes": @(storage.outputBytes), @"buffer_allocations": @(count),
        @"bindings": bindings, @"gpu_dispatched": @NO};
      if (describe) {
        NSData *json = [NSJSONSerialization dataWithJSONObject:layout options:0 error:nil];
        fwrite(json.bytes, 1, json.length, stdout);
        return 0;
      }
      input = dispatch ? fopen(argv[2], "rb") : NULL;
      if (dispatch && (!input || fseek(input, 0, SEEK_END) ||
          ftell(input) != (long)storage.matrixBytes || fseek(input, 0, SEEK_SET)))
        return fail("input_size", nil);
    }

    fprintf(stderr, "phase=create_pipeline\n"); fflush(stderr);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fail("device", nil);
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:
      [dir stringByAppendingPathComponent:@"scan.lib.metallib"]] error:&error];
    if (!library) return fail("library", error);
    id<MTLFunction> function = [library newFunctionWithName:name];
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
      // A TEXTURE IMAGE IS ADMITTED AND BOUND HERE, not in the dispatch path, and nothing is
      // committed. An image that declares a texture whose contract does not admit is REFUSED -
      // the worker does not fall back to the standalone probe, which is the only other thing in
      // this tree that binds one.
      NSDictionary *resources = abi[@"resources"];
      if ([resources isKindOfClass:[NSDictionary class]] && resources[@"textures"]) {
        NSDictionary *extent = m[@"texture_extent"];
        NSUInteger width, height;
        if (![extent isKindOfClass:[NSDictionary class]] ||
            !integer(extent[@"width"], 16384, &width) ||
            !integer(extent[@"height"], 16384, &height))
          return fail("texture_extent", nil);
        NSData *payload = [NSData dataWithContentsOfFile:
            [dir stringByAppendingPathComponent:@"texture.bin"]];
        if (!payload) return fail("texture_payload_missing", nil);
        G17TexturePlan plan;
        const char *why = NULL;
        if (!textureAdmit(resources, width, height, payload.length, 0, &plan, &why))
          return fail(why ? why : "texture_contract", nil);
        NSDictionary *identity = textureBind(device, pipeline, &plan, payload, &why);
        if (!identity) return fail(why ? why : "texture_bind", nil);
        NSData *out = [NSJSONSerialization dataWithJSONObject:
            @{@"status": @0, @"load_only": @YES, @"gpu_dispatched": @NO, @"texture": identity}
            options:0 error:nil];
        fwrite(out.bytes, 1, out.length, stdout);
        putchar('\n');
        return 0;
      }
      puts("{\"status\":0,\"load_only\":true,\"gpu_dispatched\":false}");
      return 0;
    }
    if (fp32)
      return runFP32(device, pipeline, [NSString stringWithUTF8String:argv[2]],
                     &lnStorage, bindings, indices, limit, queryProjection);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> first = [device newBufferWithLength:firstBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> queryBuffer = separate ? [device newBufferWithLength:storage.queryBytes
      options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> output = [device newBufferWithLength:storage.outputBytes options:MTLResourceStorageModeShared];
    if (!queue || !first || !output || (separate && !queryBuffer)) return fail("allocation", nil);
    if (fread(first.contents, 1, storage.matrixBytes, input) != storage.matrixBytes)
      return fail("input_read", nil);
    fclose(input);
    if (!g17StorageFinite(&storage, first.contents, rows*columns)) return fail("nonfinite_input", nil);
    void *request = affine ? first.contents : separate ? queryBuffer.contents :
      (uint8_t *)first.contents + storage.matrixBytes;
    NSArray<id<MTLBuffer>> *buffers = separate ? @[first, queryBuffer, output] : @[first, output];
    NSMutableArray *addresses = [NSMutableArray array];
    for (id<MTLBuffer> b in buffers) [addresses addObject:@((uintptr_t)b.contents)];
    NSDictionary *identity = @{@"pipeline": @((uintptr_t)(__bridge void *)pipeline),
      @"buffers": addresses, @"matrix_bytes": @(storage.matrixBytes), @"storage_dtype": @"float16",
      @"output_bytes": @(storage.outputBytes), @"pipeline_builds": @1,
      @"matrix_uploads": @1, @"buffer_allocations": @(count), @"bindings": bindings};
    NSUInteger group = MIN(rows, MIN((NSUInteger)32, pipeline.maxTotalThreadsPerThreadgroup));
    if (!group) return fail("threadgroup_limit", nil);
    if (!reply(@{@"protocol": @1, @"sequence": @0, @"bytes": @0, @"rows": @(rows),
                 @"columns": @(columns), @"identity": identity}, NULL, 0)) return fail("handshake", nil);
    for (NSUInteger sequence = 1; sequence <= limit; ++sequence) {
      @autoreleasepool {
        uint32_t size;
        size_t got = fread(&size, 1, 4, stdin);
        if (!got && feof(stdin)) return 0;
        if (got != 4 || size != requestBytes || fread(request, 1, size, stdin) != size)
          return fail("request_frame", nil);
        if (!g17StorageFinite(&storage, request, size/2)) return fail("nonfinite_request", nil);
        g17StorageReset(&storage, output.contents);
        id<MTLCommandBuffer> cb = g17_gpu_cb(queue);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) return fail("command", nil);
        [enc setComputePipelineState:pipeline];
        for (NSUInteger i = 0; i < count; ++i) [enc setBuffer:buffers[i] offset:0 atIndex:indices[i]];
        [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
        [enc endEncoding];
        fprintf(stderr, "phase=submitting sequence=%lu\n", (unsigned long)sequence); fflush(stderr);
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted || cb.error) return fail("command_status", cb.error);
        const char *invalid = g17StorageCheck(&storage, output.contents);
        if (invalid) return fail(invalid, nil);
        if (!reply(@{@"protocol": @1, @"sequence": @(sequence), @"bytes": @(storage.replyBytes),
          @"status": @0, @"boundary_guard": @YES, @"identity": identity,
          @"gpu_seconds": @(cb.GPUEndTime-cb.GPUStartTime)}, output.contents, storage.replyBytes))
          return fail("reply", nil);
      }
    }
  }
  return 0;
}
