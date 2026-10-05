// Resolve graph bindings to persistent allocation positions before Metal loading.
// This checks host scheduling only. Image/ABI and arithmetic admission are separate.
// Sorted, non-overlapping byte intervals. At most one output interval per stage.
// Counts are bounded independently of payload size; no bitmap scales with memory.
enum { G17_ATTN_MAX_WINDOW_STAGES=128, G17_ATTN_MAX_BATCHES=8,
       G17_ATTN_MAX_RANGES=G17_ATTN_MAX_WINDOW_STAGES*G17_ATTN_MAX_BATCHES };
typedef struct {
  size_t count, start[G17_ATTN_MAX_RANGES], end[G17_ATTN_MAX_RANGES];
} G17AttentionRanges;

static bool attentionRangeCovered(const G17AttentionRanges *ranges,size_t start,size_t end) {
  if (start>=end) return false;
  size_t cursor=start;
  for(size_t i=0;i<ranges->count && cursor<end;++i) {
    if(ranges->end[i]<=cursor) continue;
    if(ranges->start[i]>cursor) return false;
    cursor=ranges->end[i];
  }
  return cursor>=end;
}

static bool attentionRangeInsert(G17AttentionRanges *ranges,size_t start,size_t end) {
  if(start>=end || ranges->count>=G17_ATTN_MAX_RANGES) return false;
  size_t at=0;
  while(at<ranges->count && ranges->start[at]<start) ++at;
  if((at && ranges->end[at-1]>start) || (at<ranges->count && ranges->start[at]<end)) return false;
  for(size_t i=ranges->count;i>at;--i) {
    ranges->start[i]=ranges->start[i-1];ranges->end[i]=ranges->end[i-1];
  }
  ranges->start[at]=start;ranges->end[at]=end;++ranges->count;return true;
}

// Extended allocation capacity is explicit and belongs only to resident batches.
static NSUInteger attentionProgramLimit(NSDictionary *graph) {
  return [graph[@"format"] isEqual:@"g17-resident-batches-v1"]?16:10;
}

static NSUInteger attentionAllocationLimit(NSDictionary *graph) {
  id profile=graph[@"allocation_profile"];
  if(!profile) return G17_ATTN_LEGACY_MAX;
  if([profile isKindOfClass:NSString.class] && [profile isEqual:@"resident-64-v1"] &&
     [graph[@"format"] isEqual:@"g17-resident-batches-v1"] &&
     [graph[@"binding_windows"] isEqual:@"disjoint-v1"]) return G17_ATTN_MAX;
  return 0;
}

static NSArray *attentionSchedule(NSDictionary *graph, NSArray *names,
                                  const G17AttentionStorage *storage) {
  if(!storage || !storage->count || storage->count>attentionAllocationLimit(graph)) return nil;
  if(graph[@"textures"] || graph[@"texture_bindings"]) return nil;
  id profile=graph[@"binding_windows"];
  BOOL views=[profile isKindOfClass:NSString.class] && [profile isEqual:@"disjoint-v1"];
  if(profile && !views) return nil;
  id writePolicy=graph[@"write_policy"];
  BOOL inplace=[writePolicy isEqual:@"inplace-v1"];
  BOOL multiple=inplace || [writePolicy isEqual:@"multiple-v1"];
  if(inplace && ![graph[@"reply_selection"] isEqual:@"final-output-only"]) return nil;
  if(writePolicy && (!multiple || views)) return nil;
  id initialization=graph[@"output_initialization"];
  if (!g17AttentionOutputMatches(storage,graph) || !g17AttentionIntermediatesMatch(storage,graph,names)) return nil;
  id stages=graph[@"stages"];
  BOOL batched=[graph[@"format"] isEqual:@"g17-resident-batches-v1"];
  NSMutableArray *batchIds=[NSMutableArray array],*batchEnds=[NSMutableArray array];
  if(batched) {
    id batches=graph[@"batches"];
    if(!views || stages || ![batches isKindOfClass:NSArray.class] ||
       ![batches count] || [batches count]>G17_ATTN_MAX_BATCHES) return nil;
    NSMutableArray *flat=[NSMutableArray array];NSUInteger batchIndex=0;
    for(id batch in batches) {
      if(![batch isKindOfClass:NSArray.class] || ![batch count] ||
         [batch count]>G17_ATTN_MAX_WINDOW_STAGES) return nil;
      NSUInteger at=0;
      for(id stage in batch) {
        [flat addObject:stage];[batchIds addObject:@(batchIndex)];
        [batchEnds addObject:@(++at==[batch count])];
      }
      ++batchIndex;
    }
    stages=flat;
  } else if(graph[@"batches"]) return nil;
  if (!batched && (![stages isKindOfClass:NSArray.class] || ![stages count] ||
      [stages count]>(views?G17_ATTN_MAX_WINDOW_STAGES:32))) return nil;
  BOOL available[G17_ATTN_MAX]={0};
  // Up to one MiB at the extended capacity: keep range state off the stack.
  NSMutableData *rangeStorage=[NSMutableData dataWithLength:storage->count*sizeof(G17AttentionRanges)];
  if(!rangeStorage) return nil;
  G17AttentionRanges *ranges=(G17AttentionRanges *)rangeStorage.mutableBytes;
  for (NSUInteger i=0;i<names.count;++i) {
    available[i]=storage->role[i]<=G17_ATTN_PARAMETER;
    if(views && available[i] && !attentionRangeInsert(&ranges[i],G17_ATTN_GUARD,
        G17_ATTN_GUARD+storage->payload[i])) return nil;
  }
  NSMutableSet *stageNames=[NSMutableSet set];
  NSMutableArray *schedule=[NSMutableArray array];
  for (id stage in stages) {
    if (![stage isKindOfClass:NSDictionary.class] || stage[@"textures"] || stage[@"texture_bindings"]) return nil;
    id name=stage[@"name"],program=stage[@"program"],grid=stage[@"grid"];
    if (![name isKindOfClass:NSString.class] || ![name length] ||
        [stageNames containsObject:name] || ![program isKindOfClass:NSString.class] ||
        ![program length] || ![grid isKindOfClass:NSArray.class] || [grid count]!=3)
      return nil;
    [stageNames addObject:name];
    NSUInteger threads=1;
    for (id value in grid) {
      NSUInteger n;
      if (!integer(value,16*1024*1024,&n) || !n || threads>16*1024*1024/n) return nil;
      threads*=n;
    }
    id bindings=stage[@"bindings"];
    if (![bindings isKindOfClass:NSArray.class] || ![bindings count] || [bindings count]>32)
      return nil;
    // Whole-payload in-place writes are explicit and ordered. Availability here
    // means an earlier shader publication, not a CPU initialization copy.
    NSMutableSet *inplaceClaims=[NSMutableSet set],*rewrites=[NSMutableSet set];
    if(inplace) {
      id claims=stage[@"inplace_allocations"];
      if(![claims isKindOfClass:NSArray.class]) return nil;
      for(id allocation in claims) {
        if(![allocation isKindOfClass:NSString.class] ||
           [inplaceClaims containsObject:allocation]) return nil;
        [inplaceClaims addObject:allocation];
      }
    }
    NSMutableSet *indices=[NSMutableSet set],*used=[NSMutableSet set];
    NSMutableArray *resolved=[NSMutableArray array];
    NSUInteger output=NSNotFound,outputStart=0,outputEnd=0;
    NSMutableArray *writtenPositions=[NSMutableArray array];
    for (id binding in bindings) {
      if (![binding isKindOfClass:NSDictionary.class]) return nil;
      id allocation=binding[@"allocation"],written=binding[@"written"];
      NSUInteger index,offset,length;
      if (![allocation isKindOfClass:NSString.class] ||
          ![written isKindOfClass:NSNumber.class] ||
          CFGetTypeID((__bridge CFTypeRef)written)!=CFBooleanGetTypeID() ||
          !integer(binding[@"index"],30,&index) ||
          !integer(binding[@"offset"],64*1024*1024+G17_ATTN_GUARD,&offset) ||
          offset<G17_ATTN_GUARD || (!views && offset!=G17_ATTN_GUARD) ||
          !integer(binding[@"length"],64*1024*1024,&length) ||
          [indices containsObject:@(index)] || [used containsObject:allocation]) return nil;
      NSUInteger position=[names indexOfObject:allocation];
      if (position==NSNotFound || !length) return nil;
      NSUInteger width=storage->binding_width[position];
      NSUInteger payloadEnd=G17_ATTN_GUARD+storage->payload[position];
      if(!width || offset%width || length%width || offset>payloadEnd || length>payloadEnd-offset ||
          (!views && length!=storage->payload[position])) return nil;
      [indices addObject:@(index)];[used addObject:allocation];
      if ([written boolValue]) {
        if ((!multiple && output!=NSNotFound) || storage->role[position]<G17_ATTN_INTERMEDIATE ||
            (!views && available[position] && !inplace)) return nil;
        if(inplace && available[position]) [rewrites addObject:allocation];
        [writtenPositions addObject:@(position)];
        if(!multiple || [allocation isEqual:stage[@"result_allocation"]]) {
          output=position;outputStart=offset;outputEnd=offset+length;
        }
      // ConfigureIntermediates validated this complete source-byte copy before
      // scheduling; Prepare/Begin perform it before every query. It permits a
      // read, but does not consume the allocation's first shader write.
      } else if (views ? !attentionRangeCovered(&ranges[position],offset,offset+length) :
                 !(available[position] || storage->intermediate_from_input[position])) return nil;
      [resolved addObject:@{@"index":@(index),@"allocation":allocation,
        @"buffer_position":@(position),@"offset":@(offset),@"length":@(length),
        @"written":written}];
    }
    if(inplace && ![inplaceClaims isEqualToSet:rewrites]) return nil;
    if (output==NSNotFound) return nil;
    if(!multiple && stage[@"result_allocation"]) return nil;
    // Publish only after checking every read, so a same-stage read cannot
    // borrow availability from an earlier write in the binding list.
    if(views && !attentionRangeInsert(&ranges[output],outputStart,outputEnd)) return nil;
    available[output]=views ? attentionRangeCovered(&ranges[output],G17_ATTN_GUARD,
        G17_ATTN_GUARD+storage->payload[output]) : YES;
    if(multiple) for(NSNumber *position in writtenPositions) available[position.unsignedIntegerValue]=YES;
    NSMutableDictionary *record=[@{@"name":name,@"program":program,@"grid":grid,
      @"bindings":resolved,@"output_position":@(output)} mutableCopy];
    if(initialization) record[@"output_initialization"]=initialization;
    if(graph[@"output_initialization_offset"])
      record[@"output_initialization_offset"]=graph[@"output_initialization_offset"];
    if(multiple) record[@"write_policy"]=writePolicy;
    if(inplace) {
      record[@"inplace_allocations"]=[stage[@"inplace_allocations"] copy];
      record[@"reply_selection"]=@"final-output-only";
      // Every binding above is the complete payload; final-only transport must
      // not confuse this with the partial publication of a windowed graph.
      record[@"output_complete"]=@YES;
    }
    if(graph[@"intermediate_initialization"])
      record[@"intermediate_initialization"]=g17AttentionIntermediateRecords(storage);
    if(batched) {
      record[@"batch_index"]=batchIds[schedule.count];
      record[@"batch_end"]=batchEnds[schedule.count];
    }
    if(views) {
      record[@"binding_windows"]=@"disjoint-v1";
      record[@"output_offset"]=@(outputStart);record[@"output_length"]=@(outputEnd-outputStart);
      record[@"output_complete"]=@(available[output]);
    }
    id tg=stage[@"threadgroup"];
    if(tg) {
      if(![tg isKindOfClass:NSDictionary.class] || [tg count]!=4) return nil;
      id group=tg[@"required_size"],dynamic=tg[@"dynamic_memory"];
      NSUInteger bytes,alignment;
      if(![group isKindOfClass:NSArray.class] || [group count]!=3 ||
          ![dynamic isKindOfClass:NSArray.class] || [dynamic count] ||
          !integer(tg[@"static_memory_bytes"],64*1024*1024,&bytes) ||
          !integer(tg[@"static_memory_alignment"],64*1024*1024,&alignment) ||
          !alignment || (alignment&(alignment-1)) || bytes%alignment) return nil;
      for(NSUInteger axis=0;axis<3;++axis) {
        NSUInteger n;
        if(!integer(group[axis],1024,&n) || !n || [grid[axis] unsignedIntegerValue]%n) return nil;
      }
      record[@"threadgroup"]=[tg copy];
    }
    id execution=stage[@"execution"];
    if(execution) {
      if(![execution isKindOfClass:NSDictionary.class]) return nil;
      NSUInteger width;id tensor=execution[@"tensor"];
      if([execution count]!=2 ||
          !integer(execution[@"simd_width"],32,&width) || width!=32 ||
          ![tensor isKindOfClass:NSNumber.class] ||
          CFGetTypeID((__bridge CFTypeRef)tensor)!=CFBooleanGetTypeID() || ![tensor boolValue] ||
          [grid[0] unsignedIntegerValue]%width) return nil;
      record[@"execution"]=[execution copy];
    }
    [schedule addObject:record];
  }
  for (NSUInteger i=0;i<names.count;++i)
    if (!available[i] && !storage->intermediate_from_input[i]) return nil;
  if(multiple && storage->role[[schedule.lastObject[@"output_position"] unsignedIntegerValue]]!=G17_ATTN_OUTPUT)
    return nil;
  if([[NSSet setWithArray:[schedule valueForKey:@"program"]] count]>attentionProgramLimit(graph)) return nil;
  return schedule;
}
