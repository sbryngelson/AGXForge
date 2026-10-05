#import "g17attentionexecutor.h"
#include "g17launch.h"
#include <CommonCrypto/CommonDigest.h>

static BOOL integer(id value,NSUInteger maximum,NSUInteger *out) {
  if (![value isKindOfClass:NSNumber.class] || CFGetTypeID((__bridge CFTypeRef)value)==CFBooleanGetTypeID()) return NO;
  const char *type=[value objCType];
  if (!strchr("cCsSiIlLqQ",type[0]) || [value longLongValue]<0 || [value unsignedLongLongValue]>maximum) return NO;
  *out=[value unsignedIntegerValue];return YES;
}
#include "g17textureresources.h"
#include "g17gpulock.h"

static BOOL attentionMultipleWrites(id policy) {
  return [policy isEqual:@"multiple-v1"] || [policy isEqual:@"inplace-v1"];
}

static BOOL attentionFinalOutputProfile(NSDictionary *stage) {
  return [stage[@"binding_windows"] isEqual:@"disjoint-v1"] ||
    ([stage[@"write_policy"] isEqual:@"inplace-v1"] &&
     [stage[@"reply_selection"] isEqual:@"final-output-only"]);
}

static void attentionError(NSError **error,NSString *reason) {
  if (error) *error=[NSError errorWithDomain:@"G17Attention" code:2
                                  userInfo:@{NSLocalizedDescriptionKey:reason}];
}

@implementation G17AttentionExecutor {
  G17AttentionStorage _storage;
  NSArray *_schedule;
  NSDictionary<NSString *,id<MTLComputePipelineState>> *_pipelines;
  NSMutableArray *_snapshots;
  NSArray<id<MTLBuffer>> *_buffers;
  id<MTLCommandQueue> _queue;
  BOOL _failed;
  NSUInteger _queries;
  BOOL _finalOutputOnly;
  NSArray *_bindingTrace;
  NSArray *_submissionTrace;
  NSArray *_launches;
  NSArray *_resolvedLaunches;
  NSArray *_texturePlan;
  NSDictionary *_textureSnapshots;
  NSDictionary<NSString *,id<MTLTexture>> *_textures;
  NSArray *_textureBindingTrace;

}

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       storage:(const G17AttentionStorage *)storage
                      schedule:(NSArray *)schedule
                     pipelines:(NSDictionary<NSString *,id<MTLComputePipelineState>> *)pipelines
                     snapshots:(NSArray *)snapshots error:(NSError **)error {
  return [self initWithDevice:device storage:storage schedule:schedule pipelines:pipelines
                   snapshots:snapshots textures:nil textureSnapshots:nil error:error];
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       storage:(const G17AttentionStorage *)storage
                      schedule:(NSArray *)schedule
                     pipelines:(NSDictionary<NSString *,id<MTLComputePipelineState>> *)pipelines
                     snapshots:(NSArray *)snapshots
                      textures:(NSDictionary *)textures
              textureSnapshots:(NSDictionary<NSString *,NSData *> *)textureSnapshots
                         error:(NSError **)error {
  self=[super init];if (!self) return nil;
  if (!device || !storage || !storage->count || storage->count>G17_ATTN_MAX ||
      snapshots.count!=storage->count || !schedule.count) {
    attentionError(error,@"executor_inputs");return nil;
  }
  if (!g17AttentionStorageInitTyped(&_storage,storage->count,storage->payload,storage->role,storage->type)) {
    attentionError(error,@"storage_contract");return nil;
  }
  // Reconstruct all declared element facts, not only scalar transport types.
  // InitTyped intentionally starts scalar; losing this width made uint4 traces
  // report four bytes even though the parser and scheduler retained sixteen.
  for (size_t i=0;i<storage->count;++i) {
    if (!g17AttentionSetBindingWidth(&_storage,i,storage->binding_width[i])) {
      attentionError(error,@"binding_lane_contract");return nil;
    }
    if (!g17AttentionSetValuePolicy(&_storage,i,storage->value_policy[i])) {
      attentionError(error,@"allocation_value_policy");return nil;
    }
  }
  if (storage->output_from_input && !(storage->output_bytes ?
      g17AttentionOutputFromInputBytes(&_storage,storage->initialization_offset) : storage->output_slice ?
      g17AttentionOutputFromInputSlice(&_storage,storage->initialization_offset) :
      g17AttentionOutputFromInput(&_storage))) {
    attentionError(error,@"output_initialization");return nil;
  }
  for(size_t i=0;i<storage->count;++i) if(storage->intermediate_from_input[i])
    if(!g17AttentionIntermediateFromInputBytes(&_storage,i,storage->intermediate_offset[i])) {
      attentionError(error,@"intermediate_initialization");return nil;
    }
  for (NSDictionary *stage in schedule) {
    if (!g17AttentionIntermediatesMatch(&_storage,stage,nil)) {
      attentionError(error,@"intermediate_initialization");return nil;
    }
    if (!g17AttentionOutputMatches(&_storage,stage)) {
      attentionError(error,@"output_initialization");return nil;
    }
  }
  if(textures) {
    _texturePlan=attentionTexturePlan(textures);
    if(!_texturePlan || ![textureSnapshots isKindOfClass:NSDictionary.class] ||
       ![[NSSet setWithArray:textureSnapshots.allKeys] isEqual:[NSSet setWithArray:textures.allKeys]]) {
      attentionError(error,@"texture_descriptor_or_snapshot_contract");return nil;
    }
    NSMutableDictionary *owned=[NSMutableDictionary dictionary];
    for(NSDictionary *p in _texturePlan) {
      id data=textureSnapshots[p[@"name"]];
      if(![data isKindOfClass:NSData.class] || [data length]!=[p[@"payload_bytes"] unsignedIntegerValue]) {
        attentionError(error,@"texture_snapshot_extent");return nil;
      }
      owned[p[@"name"]]=[data copy];
    }
    _textureSnapshots=[owned copy];
  } else if(textureSnapshots) {attentionError(error,@"texture_snapshot_without_descriptor");return nil;}
  NSUInteger resourceBytes=0;
  for(NSUInteger i=0;i<storage->count;++i) {
    if(storage->allocation[i]>128*1024*1024-resourceBytes) {attentionError(error,@"joint_resource_budget");return nil;}
    resourceBytes+=storage->allocation[i];
  }
  for(NSDictionary *p in _texturePlan) {
    NSUInteger bytes=[p[@"payload_bytes"] unsignedIntegerValue];
    if(bytes>128*1024*1024-resourceBytes) {attentionError(error,@"joint_resource_budget");return nil;}
    resourceBytes+=bytes;
  }
  BOOL referenced=NO;
  for(NSDictionary *stage in schedule) {
    id bindings=stage[@"texture_bindings"];
    if(bindings && (!_texturePlan || !attentionTextureBindings(_texturePlan,bindings))) {
      attentionError(error,@"texture_stage_binding_contract");return nil;
    }
    if(bindings) referenced=YES;
  }
  if(_texturePlan && !referenced) {attentionError(error,@"unused_texture_allocation");return nil;}
  NSData *scheduleBytes=[NSJSONSerialization dataWithJSONObject:schedule options:0 error:nil];
  _schedule=scheduleBytes?[NSJSONSerialization JSONObjectWithData:scheduleBytes options:0 error:nil]:nil;
  if(!_schedule) {attentionError(error,@"schedule_snapshot");return nil;}
  _pipelines=[pipelines copy];
  _finalOutputOnly=attentionFinalOutputProfile(schedule[0]);
  for(NSDictionary *stage in schedule) {
    if(attentionFinalOutputProfile(stage)!=_finalOutputOnly ||
       ([stage[@"write_policy"] isEqual:@"inplace-v1"] &&
        ![stage[@"reply_selection"] isEqual:@"final-output-only"])) {
      attentionError(error,@"mixed_schedule_profiles");return nil;
    }
    id policy=stage[@"write_policy"];
    BOOL multiple=attentionMultipleWrites(policy);
    if(multiple!=attentionMultipleWrites(schedule[0][@"write_policy"]) ||
       (multiple && ![policy isEqual:schedule[0][@"write_policy"]])) {
      attentionError(error,@"mixed_write_policies");return nil;
    }
    NSMutableSet *written=[NSMutableSet set];
    for(NSDictionary *binding in stage[@"bindings"]) if([binding[@"written"] boolValue]) {
      NSNumber *position=binding[@"buffer_position"];
      if(![position isKindOfClass:NSNumber.class] || position.unsignedIntegerValue>=_storage.count ||
          _storage.role[position.unsignedIntegerValue]<G17_ATTN_INTERMEDIATE || [written containsObject:position]) {
        attentionError(error,@"written_allocation_contract");return nil;
      }
      [written addObject:position];
    }
    if((policy && (!multiple || (_finalOutputOnly && ![policy isEqual:@"inplace-v1"]))) || ![written containsObject:stage[@"output_position"]] ||
        (!multiple && written.count!=1)) {
      attentionError(error,@"written_allocation_contract");return nil;
    }
  }
  if(attentionMultipleWrites(schedule[0][@"write_policy"]) &&
      _storage.role[[schedule.lastObject[@"output_position"] unsignedIntegerValue]]!=G17_ATTN_OUTPUT) {
    attentionError(error,@"final_result_allocation");return nil;
  }
  if(_finalOutputOnly) {
    NSUInteger last=[schedule.lastObject[@"output_position"] unsignedIntegerValue];
    if(last>=storage->count || storage->role[last]!=G17_ATTN_OUTPUT ||
        ![schedule.lastObject[@"output_complete"] boolValue]) {
      attentionError(error,@"incomplete_final_output");return nil;
    }
  }
  _snapshots=[NSMutableArray array];
  for (NSUInteger i=0;i<storage->count;++i) {
    if (storage->role[i]<=G17_ATTN_PARAMETER) {
      id snapshot=snapshots[i];
      if (![snapshot isKindOfClass:NSData.class] || [snapshot length]!=storage->payload[i]) {
        attentionError(error,@"snapshot_extent");return nil;
      }
      [_snapshots addObject:[snapshot copy]];
    } else [_snapshots addObject:NSNull.null];
  }
  NSMutableArray *launches=[NSMutableArray array];
  NSMutableArray *resolved=[NSMutableArray array];
  for (NSDictionary *stage in _schedule) {
    id<MTLComputePipelineState> pipeline=_pipelines[stage[@"program"]];
    if (!pipeline || pipeline.device!=device || !pipeline.maxTotalThreadsPerThreadgroup || !pipeline.threadExecutionWidth) {
      attentionError(error,@"missing_or_incompatible_pipeline");return nil;
    }
    NSDictionary *tg=stage[@"threadgroup"];
    NSArray *grid=stage[@"grid"];
    NSArray *group=tg[@"required_size"];
    if(!group) group=@[@(MIN([grid[0] unsignedIntegerValue],
      MIN((NSUInteger)32,pipeline.maxTotalThreadsPerThreadgroup))),@1,@1];
    NSDictionary *execution=stage[@"execution"];
    if(execution) {
      size_t shape[3];for(NSUInteger i=0;i<3;++i)shape[i]=[group[i] unsignedIntegerValue];
      const char *failure=g17ValidateSIMDLaunch([execution[@"simd_width"] unsignedIntegerValue],
        pipeline.threadExecutionWidth,shape,pipeline.maxTotalThreadsPerThreadgroup);
      if(failure){attentionError(error,@(failure));return nil;}
    }
    NSMutableDictionary *resolvedLaunch=[@{@"stage":stage[@"name"],@"program":stage[@"program"],
      @"grid":grid,@"threadgroup":group,@"explicit_group":@(tg!=nil),
      @"pipeline_simd_width":@(pipeline.threadExecutionWidth),
      @"pipeline_thread_limit":@(pipeline.maxTotalThreadsPerThreadgroup),
      @"pipeline_static_memory_bytes":@(pipeline.staticThreadgroupMemoryLength)} mutableCopy];
    if(execution) resolvedLaunch[@"execution"]=[execution copy];
    [resolved addObject:resolvedLaunch];
    if(tg) {
      NSArray *g=stage[@"grid"],*t=tg[@"required_size"];
      if(g.count!=3 || t.count!=3) {attentionError(error,@"launch_shape");return nil;}
      size_t grid[3],group[3];for(NSUInteger i=0;i<3;++i){grid[i]=[g[i] unsignedIntegerValue];group[i]=[t[i] unsignedIntegerValue];}
      MTLSize limit=device.maxThreadsPerThreadgroup;
      size_t axes[3]={limit.width,limit.height,limit.depth};
      const char *failure=g17ValidateLaunch(grid,group,axes,pipeline.maxTotalThreadsPerThreadgroup,
        [tg[@"static_memory_bytes"] unsignedIntegerValue],
        [tg[@"static_memory_alignment"] unsignedIntegerValue],
        pipeline.staticThreadgroupMemoryLength,device.maxThreadgroupMemoryLength);
      if(failure){attentionError(error,@(failure));return nil;}
      [launches addObject:@{@"stage":stage[@"name"],@"grid":g,@"threadgroup":t,
        @"pipeline_static_memory_bytes":@(pipeline.staticThreadgroupMemoryLength),
        @"pipeline_thread_limit":@(pipeline.maxTotalThreadsPerThreadgroup),
        @"device_thread_limits":@[@(axes[0]),@(axes[1]),@(axes[2])],
        @"device_memory_limit":@(device.maxThreadgroupMemoryLength)}];
    }
  }
  _launches=[launches copy];
  _resolvedLaunches=[resolved copy];
  NSMutableDictionary *textureObjects=[NSMutableDictionary dictionary];
  for(NSDictionary *p in _texturePlan) {
    NSUInteger width=[p[@"width"] unsignedIntegerValue],height=[p[@"height"] unsignedIntegerValue];
    MTLTextureDescriptor *descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
      width:width height:height mipmapped:NO];
    descriptor.storageMode=MTLStorageModeShared;descriptor.usage=MTLTextureUsageShaderRead;
    descriptor.hazardTrackingMode=MTLHazardTrackingModeTracked;
    id<MTLTexture> texture=[device newTextureWithDescriptor:descriptor];
    if(!texture || texture.device!=device || texture.width!=width || texture.height!=height ||
       texture.pixelFormat!=MTLPixelFormatR32Uint || texture.textureType!=MTLTextureType2D ||
       texture.mipmapLevelCount!=1 || texture.arrayLength!=1 || texture.sampleCount!=1 ||
       texture.storageMode!=MTLStorageModeShared || texture.usage!=MTLTextureUsageShaderRead ||
       texture.hazardTrackingMode!=MTLHazardTrackingModeTracked || !texture.gpuResourceID._impl) {
      attentionError(error,@"texture_allocation_contract");return nil;
    }
    [texture replaceRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:0
      withBytes:[_textureSnapshots[p[@"name"]] bytes] bytesPerRow:[p[@"bytes_per_row"] unsignedIntegerValue]];
    textureObjects[p[@"name"]]=texture;
  }
  _textures=[textureObjects copy];
  if(![self checkTextures]) {attentionError(error,@"texture_upload_mismatch");return nil;}
  _queue=[device newCommandQueue];
  if (!_queue) {attentionError(error,@"command_queue");return nil;}
  NSMutableArray *buffers=[NSMutableArray array];
  void *pointers[G17_ATTN_MAX]={0};const void *original[G17_ATTN_MAX]={0};
  for (NSUInteger i=0;i<storage->count;++i) {
    id<MTLBuffer> buffer=[device newBufferWithLength:storage->allocation[i]
      options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeTracked];
    if (!buffer || !buffer.contents) {attentionError(error,@"allocation");return nil;}
    [buffers addObject:buffer];pointers[i]=buffer.contents;
    if (storage->role[i]<=G17_ATTN_PARAMETER) original[i]=[_snapshots[i] bytes];
  }
  _buffers=[buffers copy];
  const char *failure=g17AttentionPrepare(&_storage,pointers,original);
  if (failure) {attentionError(error,@(failure));return nil;}
  return self;
}

- (NSArray<NSData *> *)runSource:(NSData *)source prefix:(NSUInteger)prefix error:(NSError **)error {
  return [self runSource:source prefix:prefix batchStages:128 error:error];
}

- (NSArray<NSData *> *)runSource:(NSData *)source prefix:(NSUInteger)prefix
                   batchStages:(NSUInteger)batchStages error:(NSError **)error {
  if (!batchStages || batchStages>128) {
    attentionError(error,@"batch_stage_limit");return nil;
  }
  if (_failed || !prefix || prefix>_schedule.count || (_finalOutputOnly && prefix!=_schedule.count) || source.length!=_storage.payload[_storage.source]) {
    attentionError(error,@"query_contract_or_failed_session");return nil;
  }
  if(![self checkTextures]) {attentionError(error,@"readonly_texture_changed");_failed=YES;return nil;}
  // This API is synchronous. No allocation can be reset while its preceding
  // command buffer is running, and a failed query permanently closes admission.
  void *pointers[G17_ATTN_MAX]={0};const void *original[G17_ATTN_MAX]={0};
  for (NSUInteger i=0;i<_storage.count;++i) pointers[i]=_buffers[i].contents;
  NSData *ownedSource=[source copy];
  const char *failure=g17AttentionBegin(&_storage,pointers,ownedSource.bytes);
  if (failure) {attentionError(error,@(failure));return nil;}
  _snapshots[_storage.source]=ownedSource;
  for (NSUInteger i=0;i<_storage.count;++i)
    if (_storage.role[i]<=G17_ATTN_PARAMETER) original[i]=[_snapshots[i] bytes];
  id<MTLCommandBuffer> command=nil;
  NSUInteger commandStages=0;
  NSUInteger commandStart=0;
  NSMutableArray *submissions=[NSMutableArray array];
  _submissionTrace=@[];_bindingTrace=@[];_textureBindingTrace=@[];
  NSMutableArray *textureTrace=[NSMutableArray array];
  bool produced[G17_ATTN_MAX]={0};
  NSMutableArray *trace=[NSMutableArray array];
  for (NSUInteger s=0;s<prefix;++s) {
    if (!command) {
      command=g17_gpu_cb(_queue);
      commandStages=0;
      commandStart=s;
      if (!command) {attentionError(error,@"command_buffer");_failed=YES;return nil;}
    }
    NSDictionary *stage=_schedule[s];
    id<MTLComputePipelineState> pipeline=_pipelines[stage[@"program"]];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if (!encoder) {attentionError(error,@"compute_encoder");_failed=YES;return nil;}
    encoder.label=stage[@"name"];
    [encoder setComputePipelineState:pipeline];
    for (NSDictionary *binding in stage[@"bindings"]) {
      NSUInteger position=[binding[@"buffer_position"] unsignedIntegerValue];
      NSUInteger offset=[binding[@"offset"] unsignedIntegerValue];
      NSUInteger index=[binding[@"index"] unsignedIntegerValue];
      id<MTLBuffer> buffer=_buffers[position];
      [encoder setBuffer:buffer offset:offset atIndex:index];
      [trace addObject:@[@(s),@(index),@(position),@(offset),binding[@"length"],
                        @(_storage.binding_width[position]),@([binding[@"written"] boolValue]),@(buffer.gpuAddress)]];
    }
    for(NSDictionary *binding in stage[@"texture_bindings"]) {
      id<MTLTexture> texture=_textures[binding[@"texture"]];
      NSUInteger index=[binding[@"index"] unsignedIntegerValue];
      [encoder setTexture:texture atIndex:index];
      [textureTrace addObject:@{@"stage":@(s),@"index":@(index),@"texture":binding[@"texture"],
        @"resource_id":@(texture.gpuResourceID._impl)}];
    }
    NSArray *grid=stage[@"grid"];
    MTLSize size=MTLSizeMake([grid[0] unsignedIntegerValue],
                            [grid[1] unsignedIntegerValue],[grid[2] unsignedIntegerValue]);
    // Dispatch exactly the group recorded from pipeline queries during admission.
    NSArray *required=_resolvedLaunches[s][@"threadgroup"];
    MTLSize group=MTLSizeMake([required[0] unsignedIntegerValue],
      [required[1] unsignedIntegerValue],[required[2] unsignedIntegerValue]);
    [encoder dispatchThreads:size threadsPerThreadgroup:group];
    [encoder endEncoding];
    for(NSDictionary *binding in stage[@"bindings"]) if([binding[@"written"] boolValue])
      produced[[binding[@"buffer_position"] unsignedIntegerValue]]=_finalOutputOnly?[stage[@"output_complete"] boolValue]:true;
    ++commandStages;
    if (commandStages==batchStages || [stage[@"batch_end"] boolValue] || s+1==prefix) {
      // A completed earlier command buffer publishes its GPU writes before
      // the next is submitted. No activation readback or reinitialization.
      [command commit];[command waitUntilCompleted];
      [submissions addObject:@{@"batch_index":@(submissions.count),
        @"first_stage":@(commandStart),@"last_stage":@(s),
        @"stage_count":@(commandStages),@"status":@(command.status)}];
      _submissionTrace=[submissions copy];
      _bindingTrace=[trace copy];
      _textureBindingTrace=[textureTrace copy];
      if (command.status!=MTLCommandBufferStatusCompleted) {
        attentionError(error,command.error.localizedDescription?:@"command_failed");_failed=YES;return nil;
      }
      if(![self checkTextures]) {attentionError(error,@"readonly_texture_changed");_failed=YES;return nil;}
      command=nil;
    }
  }
  if(![self checkTextures]) {attentionError(error,@"readonly_texture_changed");_failed=YES;return nil;}
  size_t failedIndex=0;
  failure=g17AttentionCheck(&_storage,(const void *const *)pointers,original,produced,&failedIndex);
  if (failure) {
    attentionError(error,[NSString stringWithFormat:@"%s allocation=%zu",failure,failedIndex]);
    _failed=YES;return nil;
  }
  NSMutableArray *outputs=[NSMutableArray array];
  for (NSUInteger s=_finalOutputOnly?prefix-1:0;s<prefix;++s) {
    NSUInteger i=[_schedule[s][@"output_position"] unsignedIntegerValue];
    [outputs addObject:[NSData dataWithBytes:(uint8_t *)pointers[i]+G17_ATTN_GUARD
                                      length:_storage.payload[i]]];
  }
  ++_queries;return outputs;
}

// Observe completed scratch payloads directly; output correctness alone is not
// evidence that a declared writable allocation was written. No extra GPU work.
- (NSArray *)scratchObservations {
  NSMutableIndexSet *positions=[NSMutableIndexSet indexSet];
  for(NSDictionary *stage in _schedule) if(attentionMultipleWrites(stage[@"write_policy"]))
    for(NSDictionary *binding in stage[@"bindings"]) {
      NSUInteger i=[binding[@"buffer_position"] unsignedIntegerValue];
      if([binding[@"written"] boolValue] && _storage.role[i]==G17_ATTN_INTERMEDIATE)
        [positions addIndex:i];
    }
  NSMutableArray *observations=[NSMutableArray array];
  [positions enumerateIndexesUsingBlock:^(NSUInteger i,BOOL *stop) {
    (void)stop;
    const uint8_t *p=(const uint8_t *)self->_buffers[i].contents+G17_ATTN_GUARD;
    NSUInteger bytes=self->_storage.payload[i],changed=0;
    const uint32_t sentinel=0x7fc01234;const uint16_t halfSentinel=0x7e12;
    NSUInteger width=g17AttentionElementBytes(self->_storage.type[i]);
    const uint8_t *fill=width==2?(const uint8_t *)&halfSentinel:(const uint8_t *)&sentinel;
    for(NSUInteger j=0;j<bytes;++j) if(p[j]!=fill[j%width]) ++changed;
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(p,(CC_LONG)bytes,digest);
    NSMutableString *hex=[NSMutableString string];
    for(NSUInteger j=0;j<sizeof(digest);++j) [hex appendFormat:@"%02x",digest[j]];
    [observations addObject:@{@"allocation_position":@(i),@"payload_bytes":@(bytes),
      @"sha256":hex,@"changed_bytes_from_reset":@(changed),@"completed_queries":@(self->_queries)}];
  }];
  return observations;
}

- (NSDictionary *)identity {
  NSMutableArray *addresses=[NSMutableArray array],*policies=[NSMutableArray array];
  for(size_t i=0;i<_storage.count;++i)
    [policies addObject:_storage.value_policy[i]==G17_ATTN_RAW_BITS?@"raw-bits-v1":@"finite-v1"];
  for (id<MTLBuffer> buffer in _buffers) [addresses addObject:@(buffer.gpuAddress)];
  NSMutableDictionary *result=[@{@"buffer_addresses":addresses,@"completed_queries":@(_queries),
           @"allocation_count":@(_storage.count),@"pipeline_count":@(_pipelines.count),
           @"failed":@(_failed),@"explicit_launches":_launches,
           @"resolved_launches":_resolvedLaunches} mutableCopy];
  if([policies containsObject:@"raw-bits-v1"]) result[@"allocation_value_policies"]=policies;
  if(_texturePlan) {
    NSMutableArray *textures=[NSMutableArray array];
    for(NSDictionary *p in _texturePlan) {
      id<MTLTexture> texture=_textures[p[@"name"]];
      [textures addObject:@{@"name":p[@"name"],@"resource_id":@(texture.gpuResourceID._impl),
        @"width":@(texture.width),@"height":@(texture.height),@"pixel_format":@"R32Uint",@"native_pixel_format":@(texture.pixelFormat)}];
    }
    result[@"texture_allocations"]=textures;
  }
  return result;
}
- (BOOL)checkTextures {
  for(NSDictionary *p in _texturePlan) {
    NSMutableData *readback=[NSMutableData dataWithLength:[p[@"payload_bytes"] unsignedIntegerValue]];
    [_textures[p[@"name"]] getBytes:readback.mutableBytes bytesPerRow:[p[@"bytes_per_row"] unsignedIntegerValue]
      fromRegion:MTLRegionMake2D(0,0,[p[@"width"] unsignedIntegerValue],[p[@"height"] unsignedIntegerValue]) mipmapLevel:0];
    if(![readback isEqualToData:_textureSnapshots[p[@"name"]]]) return NO;
  }
  return YES;
}
- (NSArray *)textureBindingTrace {return _textureBindingTrace?:@[];}
- (NSArray *)bindingTrace {return _bindingTrace?:@[];}
- (NSArray *)submissionTrace {return _submissionTrace?:@[];}
@end
