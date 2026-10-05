// Foundation-only host planning. No device, pipeline, allocation or dispatch.
// The including host supplies the common strict integer() predicate.
static NSArray *attentionTexturePlan(id textures) {
  if(![textures isKindOfClass:NSDictionary.class] || [textures count]!=1) return nil;
  NSMutableArray *rows=[NSMutableArray array];
  NSSet *keys=[NSSet setWithArray:@[@"type",@"pixel_format",@"width",@"height",@"access",@"payload_bytes"]];
  for(id name in textures) {
    if(![name isKindOfClass:NSString.class] || ![name length]) return nil;
    NSRegularExpression *pattern=[NSRegularExpression regularExpressionWithPattern:@"^[A-Za-z_][A-Za-z_0-9]*\\z" options:0 error:nil];
    if([pattern numberOfMatchesInString:name options:0 range:NSMakeRange(0,[name length])]!=1) return nil;
    id t=textures[name];
    if(![t isKindOfClass:NSDictionary.class] || ![[NSSet setWithArray:[t allKeys]] isEqual:keys] ||
       ![t[@"type"] isEqual:@"2d"] || ![t[@"pixel_format"] isEqual:@"R32Uint"] || ![t[@"access"] isEqual:@"read"]) return nil;
    NSUInteger width,height,length;
    if(!integer(t[@"width"],4096,&width) || !width || !integer(t[@"height"],4096,&height) || !height ||
       !integer(t[@"payload_bytes"],64*1024*1024,&length) || length!=width*height*4) return nil;
    NSMutableDictionary *row=[t mutableCopy];
    [row addEntriesFromDictionary:@{@"name":name,@"bytes_per_row":@(width*4),@"mip_levels":@1,@"array_length":@1,@"sample_count":@1}];
    [rows addObject:row];
  }
  return rows;
}

static NSArray *attentionTextureBindings(NSArray *descriptors,id requested) {
  if(![descriptors isKindOfClass:NSArray.class] || descriptors.count!=1 || ![requested isKindOfClass:NSArray.class] || [requested count]!=1) return nil;
  id r=requested[0];NSUInteger index;
  if(![r isKindOfClass:NSDictionary.class] ||
     ![[NSSet setWithArray:[r allKeys]] isEqual:[NSSet setWithArray:@[@"index",@"texture",@"access"]]] ||
     !integer(r[@"index"],0,&index) || ![r[@"access"] isEqual:@"read"] ||
     ![r[@"texture"] isEqual:descriptors[0][@"name"]]) return nil;
  return @[[r copy]];
}
