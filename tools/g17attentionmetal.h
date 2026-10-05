#import "g17attentionexecutor.h"

static BOOL attentionTransportDimensions(id shape,NSUInteger payload,NSUInteger width,NSUInteger *rows,NSUInteger *columns) {
  if (![shape isKindOfClass:NSArray.class] || [shape count]!=2 || !width || !payload || payload%width ||
      !integer(shape[0],500000,rows) || !*rows ||
      !integer(shape[1],payload/width,columns) || !*columns) return NO;
  // Bound by the admitted allocation, not the first application's width.
  // Division avoids overflow even for malformed dimensions.
  return payload/width % *rows==0 && payload/width / *rows==*columns;
}

static NSDictionary *attentionReplyPlan(NSArray *schedule,const G17AttentionStorage *storage,NSUInteger prefix) {
  if(!prefix || prefix>schedule.count) return nil;
  BOOL inplace=[schedule[0][@"write_policy"] isEqual:@"inplace-v1"];
  if(inplace && ![schedule[0][@"reply_selection"] isEqual:@"final-output-only"]) return nil;
  BOOL finalOnly=inplace || [schedule[0][@"binding_windows"] isEqual:@"disjoint-v1"];
  if(inplace) for(NSDictionary *stage in schedule)
    if(![stage[@"write_policy"] isEqual:@"inplace-v1"] ||
       ![stage[@"reply_selection"] isEqual:@"final-output-only"] ||
       ![stage[@"output_complete"] boolValue]) return nil;
  NSUInteger first=finalOnly?prefix-1:0;
  NSUInteger position=[schedule[first][@"output_position"] unsignedIntegerValue];
  if(position>=storage->count || (finalOnly &&
      (prefix!=schedule.count || storage->role[position]!=G17_ATTN_OUTPUT ||
       ![schedule[first][@"output_complete"] boolValue]))) return nil;
  unsigned type=storage->type[position];NSUInteger bytes=0;
  for(NSUInteger i=first;i<prefix;++i) {
    NSUInteger at=[schedule[i][@"output_position"] unsignedIntegerValue];
    if(at>=storage->count || storage->type[at]!=type ||
        storage->payload[at]>128*1024*1024-bytes) return nil;
    bytes+=storage->payload[at];
  }
  return @{@"selection":finalOnly?@"final-output-only":@"all-stage-outputs",
    @"first_stage":@(first),@"output_count":@(prefix-first),@"reply_bytes":@(bytes),
    @"reply_type":@(type),@"header_limit":@(schedule[0][@"batch_index"]?1048576:(finalOnly?65536:4096))};
}

static BOOL attentionFrame(NSDictionary *header,NSArray<NSData *> *payloads,NSUInteger headerLimit) {
  NSData *json=[NSJSONSerialization dataWithJSONObject:header options:0 error:nil];
  if ((headerLimit!=4096 && headerLimit!=65536 && headerLimit!=1048576) || !json || !json.length || json.length>headerLimit) return NO;
  uint32_t length=(uint32_t)json.length;
  if (fwrite(&length,4,1,stdout)!=1 || fwrite(json.bytes,1,length,stdout)!=length) return NO;
  for (NSData *data in payloads)
    if (fwrite(data.bytes,1,data.length,stdout)!=data.length) return NO;
  return fflush(stdout)==0;
}

static int attentionMetal(NSDictionary *manifest,NSString *bundle,NSArray *schedule,
                          NSArray *names,const G17AttentionStorage *storage,
                          NSString *inputs,NSUInteger prefix,NSUInteger limit) {
  id programs=manifest[@"programs"];
  if (![programs isKindOfClass:NSDictionary.class] || ![programs count] || [programs count]>attentionProgramLimit(manifest[@"graph"]))
    return refuse("program_contract");
  // The independently admitted graph selects programs. This runtime checks
  // exact registry closure; Python admission still verifies code/images and
  // each program's launch and allocation requirements before invoking it.
  NSMutableSet *expected=[NSMutableSet set];
  for (NSDictionary *stage in schedule) [expected addObject:stage[@"program"]];
  if (![[NSSet setWithArray:[programs allKeys]] isEqual:expected]) return refuse("program_set");
  NSCharacterSet *safe=[NSCharacterSet characterSetWithCharactersInString:
                       @"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"].invertedSet;
  for (id key in programs) {
    if (![key isKindOfClass:NSString.class] || ![key length] ||
        [key rangeOfCharacterFromSet:safe].location!=NSNotFound ||
        ![programs[key] isKindOfClass:NSDictionary.class] ||
        ![programs[key][@"name"] isKindOfClass:NSString.class]) return refuse("program_name");
  }
  for (NSDictionary *stage in schedule)
    if (!programs[stage[@"program"]]) return refuse("missing_program");
  NSDictionary *textureDescriptors=manifest[@"graph"][@"textures"];
  NSMutableDictionary *textureSnapshots=nil;
  if(textureDescriptors) {
    NSArray *plan=attentionTexturePlan(textureDescriptors);
    if(!plan) return refuse("texture_contract");
    NSUInteger resourceBytes=0;
    for(NSUInteger i=0;i<storage->count;++i) {
      if(storage->allocation[i]>128*1024*1024-resourceBytes) return refuse("joint_resource_budget");
      resourceBytes+=storage->allocation[i];
    }
    textureSnapshots=[NSMutableDictionary dictionary];
    for(NSDictionary *p in plan) {
      NSUInteger bytes=[p[@"payload_bytes"] unsignedIntegerValue];
      if(bytes>128*1024*1024-resourceBytes) return refuse("joint_resource_budget");
      resourceBytes+=bytes;
      NSString *file=[[bundle stringByAppendingPathComponent:@"textures"] stringByAppendingPathComponent:
        [p[@"name"] stringByAppendingString:@".u32"]];
      NSDictionary *attributes=[[NSFileManager defaultManager] attributesOfItemAtPath:file error:nil];
      if(!attributes || [attributes fileSize]!=bytes) return refuse("texture_snapshot_extent");
      NSData *snapshot=[NSData dataWithContentsOfFile:file];
      if(snapshot.length!=bytes) return refuse("texture_snapshot_read");
      textureSnapshots[p[@"name"]]=snapshot;
    }
  }
  fprintf(stderr,"phase=create_pipelines\n");fflush(stderr);
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();
  if (!device) return refuse("device");
  NSMutableDictionary *pipelines=[NSMutableDictionary dictionary];
  for (NSString *key in [[programs allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
    NSError *error=nil;
    NSString *directory=[[bundle stringByAppendingPathComponent:@"programs"] stringByAppendingPathComponent:key];
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:
      [directory stringByAppendingPathComponent:@"program.lib.metallib"]] error:&error];
    if (!library) {fprintf(stderr,"%s\n",error.description.UTF8String);return refuse("library");}
    id<MTLFunction> function=[library newFunctionWithName:programs[key][@"name"]];
    if (!function) return refuse("function");
    MTLBinaryArchiveDescriptor *archiveDescriptor=[MTLBinaryArchiveDescriptor new];
    archiveDescriptor.url=[NSURL fileURLWithPath:[directory stringByAppendingPathComponent:@"program.arc.metallib"]];
    id<MTLBinaryArchive> archive=[device newBinaryArchiveWithDescriptor:archiveDescriptor error:&error];
    if (!archive) {fprintf(stderr,"%s\n",error.description.UTF8String);return refuse("archive");}
    MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
    descriptor.computeFunction=function;descriptor.binaryArchives=@[archive];
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithDescriptor:descriptor
      options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&error];
    if (!pipeline) {fprintf(stderr,"%s\n",error.description.UTF8String);return refuse("pipeline");}
    pipelines[key]=pipeline;
  }
  if (!inputs) {
    printf("{\"status\":0,\"load_only\":true,\"gpu_dispatched\":false,\"pipelines\":%lu}\n",
           (unsigned long)pipelines.count);return 0;
  }
  NSMutableArray *snapshots=[NSMutableArray array];
  for (NSUInteger i=0;i<names.count;++i) {
    if (storage->role[i]<=G17_ATTN_PARAMETER) {
      NSString *extension=storage->type[i]==G17_ATTN_UINT16?@".u16":storage->type[i]==G17_ATTN_FLOAT16?@".f16":(storage->type[i]==G17_ATTN_UINT32?@".u32":@".f32");
      NSString *file=[inputs stringByAppendingPathComponent:[names[i] stringByAppendingString:extension]];
      NSDictionary *attributes=[[NSFileManager defaultManager] attributesOfItemAtPath:file error:nil];
      if (!attributes || [attributes fileSize]!=storage->payload[i]) return refuse("snapshot_extent");
      NSData *data=[NSData dataWithContentsOfFile:file];
      if (!data || data.length!=storage->payload[i]) return refuse("snapshot_read");
      [snapshots addObject:data];
    } else [snapshots addObject:NSNull.null];
  }
  NSDictionary *replyPlan=attentionReplyPlan(schedule,storage,prefix);
  if(!replyPlan) return refuse("reply_plan");
  unsigned transportType=storage->type[storage->source];
  unsigned replyType=[replyPlan[@"reply_type"] unsignedIntValue];
  NSUInteger headerLimit=[replyPlan[@"header_limit"] unsignedIntegerValue];
  NSArray *sourceShape=manifest[@"graph"][@"allocations"][names[storage->source]][@"shape"];
  NSUInteger rows=0,columns=0;
  if (!attentionTransportDimensions(sourceShape,storage->payload[storage->source],g17AttentionElementBytes(transportType),&rows,&columns))
    return refuse("transport_shape");
  NSError *error=nil;
  G17AttentionExecutor *executor=[[G17AttentionExecutor alloc] initWithDevice:device storage:storage
    schedule:schedule pipelines:pipelines snapshots:snapshots textures:textureDescriptors
    textureSnapshots:textureSnapshots error:&error];
  if (!executor) {fprintf(stderr,"%s\n",error.description.UTF8String);return refuse("executor");}
  NSUInteger replyBytes=[replyPlan[@"reply_bytes"] unsignedIntegerValue];
  NSMutableDictionary *identity=[[executor identity] mutableCopy];
  [identity removeObjectForKey:@"completed_queries"];[identity removeObjectForKey:@"failed"];
  [identity addEntriesFromDictionary:@{@"storage_dtype":transportType==G17_ATTN_UINT16?@"uint16":transportType==G17_ATTN_FLOAT16?@"float16":(transportType==G17_ATTN_UINT32?@"uint32":@"float32"),
    @"reply_dtype":replyType==G17_ATTN_UINT16?@"uint16":replyType==G17_ATTN_FLOAT16?@"float16":(replyType==G17_ATTN_UINT32?@"uint32":@"float32"),
    @"matrix_bytes":@(storage->payload[storage->source]),
    @"request_bytes":@(storage->payload[storage->source]),@"reply_bytes":@(replyBytes),
    @"reply_elements":@(replyBytes/g17AttentionElementBytes(replyType)),@"completion_markers":@NO,@"prefix":@(prefix),
    @"binding_trace_schema":@[@"stage",@"index",@"allocation_position",@"byte_offset",
                             @"payload_bytes",@"element_bytes",@"written",@"gpu_address"]}];
  if(headerLimit!=4096) {
    identity[@"header_limit"]=@(headerLimit);identity[@"reply_selection"]=replyPlan[@"selection"];
  }
  // Preflight the largest planned reply header before accepting any request.
  // This is a sizing template, not an execution trace; actual setBuffer calls
  // produce the trace returned with a completed command.
  NSMutableArray *plannedTrace=[NSMutableArray array];
  NSMutableArray *plannedSubmissions=[NSMutableArray array];NSUInteger batchStart=0;
  BOOL batched=schedule[0][@"batch_index"]!=nil;
  if(batched) for(NSUInteger i=0;i<prefix;++i) if([schedule[i][@"batch_end"] boolValue]) {
    [plannedSubmissions addObject:@{@"batch_index":@(plannedSubmissions.count),
      @"first_stage":@(batchStart),@"last_stage":@(i),@"stage_count":@(i-batchStart+1),
      @"status":@(MTLCommandBufferStatusCompleted)}];batchStart=i+1;
  }
  for(NSUInteger i=0;i<prefix;++i) for(NSDictionary *binding in schedule[i][@"bindings"]) {
    NSUInteger at=[binding[@"buffer_position"] unsignedIntegerValue];
    [plannedTrace addObject:@[@(i),binding[@"index"],@(at),binding[@"offset"],binding[@"length"],
      @(storage->binding_width[at]),binding[@"written"],identity[@"buffer_addresses"][at]]];
  }
  NSMutableDictionary *plannedHeader=[@{@"protocol":@2,@"sequence":@(limit),
    @"bytes":@(replyBytes),@"status":@0,@"boundary_guard":@YES,@"readonly_inputs":@YES,
    @"identity":identity,@"binding_trace":plannedTrace} mutableCopy];
  if(batched)plannedHeader[@"submission_trace"]=plannedSubmissions;
  if(textureDescriptors) {
    NSMutableArray *plannedTextures=[NSMutableArray array];
    NSDictionary *resource=identity[@"texture_allocations"][0];
    for(NSUInteger i=0;i<prefix;++i) for(NSDictionary *binding in schedule[i][@"texture_bindings"])
      [plannedTextures addObject:@{@"stage":@(i),@"index":binding[@"index"],@"texture":binding[@"texture"],@"resource_id":resource[@"resource_id"]}];
    plannedHeader[@"texture_binding_trace"]=plannedTextures;plannedHeader[@"readonly_textures"]=@YES;
  }

  NSArray *scratchPlan=[executor scratchObservations];
  if(scratchPlan.count) {
    NSMutableArray *planned=[NSMutableArray array];
    for(NSDictionary *row in scratchPlan) {
      NSMutableDictionary *r=[row mutableCopy];r[@"changed_bytes_from_reset"]=r[@"payload_bytes"];
      r[@"completed_queries"]=@(limit);[planned addObject:r];
    }
    plannedHeader[@"scratch_observations"]=planned;
  }
  NSData *sized=[NSJSONSerialization dataWithJSONObject:plannedHeader options:0 error:nil];
  if(!sized || sized.length>headerLimit) return refuse("reply_header_budget");
  if (!attentionFrame(@{@"protocol":@2,@"sequence":@0,@"bytes":@0,@"rows":@(rows),
                        @"columns":@(columns),@"identity":identity},@[],headerLimit)) return refuse("handshake");
  for (NSUInteger sequence=1;sequence<=limit;++sequence) {
    @autoreleasepool {
      uint32_t length=0;size_t got=fread(&length,1,4,stdin);
      if (!got && feof(stdin)) return 0;
      if (got!=4 || length!=storage->payload[storage->source]) return refuse("request_length");
      NSMutableData *source=[NSMutableData dataWithLength:length];
      if (fread(source.mutableBytes,1,length,stdin)!=length) return refuse("request_truncated");
      NSArray *outputs=[executor runSource:source prefix:prefix error:&error];
      if (!outputs) {fprintf(stderr,"%s\n",error.description.UTF8String);return refuse("query");}
      if (![[executor identity][@"buffer_addresses"] isEqual:identity[@"buffer_addresses"]])
        return refuse("allocation_identity_changed");
      NSMutableDictionary *reply=[@{@"protocol":@2,@"sequence":@(sequence),@"bytes":@(replyBytes),
        @"status":@0,@"boundary_guard":@YES,@"readonly_inputs":@YES,@"identity":identity,
        @"binding_trace":[executor bindingTrace]} mutableCopy];
      if(scratchPlan.count)reply[@"scratch_observations"]=[executor scratchObservations];
      if(batched)reply[@"submission_trace"]=[executor submissionTrace];
      if(textureDescriptors) {
        if(![[executor identity][@"texture_allocations"] isEqual:identity[@"texture_allocations"]])
          return refuse("texture_identity_changed");
        reply[@"texture_binding_trace"]=[executor textureBindingTrace];
        reply[@"readonly_textures"]=@YES;
      }

      if (!attentionFrame(reply,outputs,headerLimit))
        return refuse("reply");
    }
  }
  return 0;
}
