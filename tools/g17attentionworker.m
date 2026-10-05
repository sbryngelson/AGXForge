// Host-side parser for the resident attention worker. No shader compilation.
#import <Foundation/Foundation.h>
#include "g17attentionstorage.h"

static BOOL integer(id value,NSUInteger maximum,NSUInteger *out) {
  if (![value isKindOfClass:NSNumber.class] ||
      CFGetTypeID((__bridge CFTypeRef)value)==CFBooleanGetTypeID()) return NO;
  const char *type=[value objCType];
  if (!strchr("cCsSiIlLqQ",type[0]) || [value longLongValue]<0 ||
      [value unsignedLongLongValue]>maximum) return NO;
  *out=[value unsignedIntegerValue];return YES;
}

static int refuse(const char *reason) {
  fprintf(stderr,"phase=%s\n",reason);return 2;
}

#include "g17attentionschedule.h"
#include "g17textureresources.h"
#include "g17attentionmetal.h"

int main(int argc,const char **argv) {
  @autoreleasepool {
    BOOL load=argc==3 && !strcmp(argv[2],"--load-approved");
    BOOL run=argc==6 && !strcmp(argv[5],"--run-approved");
    BOOL describe=argc==3 && (!strcmp(argv[2],"--describe-storage") || !strcmp(argv[2],"--describe-schedule") || !strcmp(argv[2],"--describe-textures"));
    if (!load && !run && !describe)
      return refuse("attention_admission_pending");
    NSString *bundle=[NSString stringWithUTF8String:argv[1]];
    NSString *path=(load||run)?[bundle stringByAppendingPathComponent:@"manifest.json"]:bundle;
    NSDictionary *attributes=[[NSFileManager defaultManager] attributesOfItemAtPath:path error:nil];
    if (!attributes || [attributes fileSize]>16*1024*1024) return refuse("graph_file");
    NSData *data=[NSData dataWithContentsOfFile:path];
    id manifest=data?[NSJSONSerialization JSONObjectWithData:data options:0 error:nil]:nil;
    if ((load||run) && (![manifest isKindOfClass:NSDictionary.class] ||
        ![manifest[@"format"] isEqual:@"g17-attention-images-v1"])) return refuse("manifest_format");
    id graph=(load||run)?manifest[@"graph"]:manifest;
    if(describe && !strcmp(argv[2],"--describe-textures")) {
      if(![graph isKindOfClass:NSDictionary.class]) return refuse("texture_contract");
      NSArray *textures=attentionTexturePlan(graph[@"textures"]);
      NSArray *bindings=textures?attentionTextureBindings(textures,graph[@"texture_bindings"]):nil;
      if(!textures || !bindings) return refuse("texture_contract");
      NSData *report=[NSJSONSerialization dataWithJSONObject:@{@"textures":textures,@"bindings":bindings,
        @"gpu_dispatched":@NO,@"loader_eligible":@NO,@"scope":@"host texture planning only"} options:0 error:nil];
      if(!report) return refuse("texture_report");
      fwrite(report.bytes,1,report.length,stdout);return 0;
    }

    if (![graph isKindOfClass:NSDictionary.class] ||
        (![graph[@"format"] isEqual:@"g17-attention-graph-v1"] &&
         ![graph[@"format"] isEqual:@"g17-resident-batches-v1"]))
      return refuse("graph_format");
    // Both profiles reach the same bounded scheduler below. The projection
    // wrapper verifies source, images, scalar evidence and numerical admission
    // before invoking the explicit --load-approved / --run-approved modes.
    NSDictionary *allocations=graph[@"allocations"];
    if (![allocations isKindOfClass:NSDictionary.class] || !allocations.count ||
        allocations.count>attentionAllocationLimit(graph)) return refuse("allocation_count");
    NSArray *names=[[allocations allKeys] sortedArrayUsingSelector:@selector(compare:)];
    NSArray *roles=@[@"input",@"parameter",@"intermediate",@"output"];
    size_t bindingWidth[G17_ATTN_MAX]={0};
    unsigned valuePolicy[G17_ATTN_MAX]={0};
    size_t payload[G17_ATTN_MAX]={0};unsigned role[G17_ATTN_MAX]={0},type[G17_ATTN_MAX]={0};
    NSUInteger i=0;
    for (NSString *name in names) {
      NSCharacterSet *unsafe=[NSCharacterSet characterSetWithCharactersInString:
        @"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."].invertedSet;
      if (!name.length || [name rangeOfCharacterFromSet:unsafe].location!=NSNotFound ||
          [name isEqual:@"."] || [name isEqual:@".."]) return refuse("allocation_name");
      NSDictionary *a=allocations[name];NSUInteger length,offset,allocation;
      if (![a isKindOfClass:NSDictionary.class] ||
          (![a[@"element_type"] isEqual:@"atomic_float"] && ![a[@"element_type"] isEqual:@"atomic_uint"] && ![a[@"element_type"] isEqual:@"long"] && ![a[@"element_type"] isEqual:@"ulong"] && ! [a[@"element_type"] isEqual:@"float2"] && ![a[@"element_type"] isEqual:@"float4"] && ![a[@"element_type"] isEqual:@"uint2"] && ![a[@"element_type"] isEqual:@"uint4"] && ! [a[@"element_type"] isEqual:@"float"] && ![a[@"element_type"] isEqual:@"uint"] && ![a[@"element_type"] isEqual:@"half"] && ![a[@"element_type"] isEqual:@"ushort"]) ||
          !integer(a[@"payload_bytes"],64*1024*1024,&length) || !length ||
          !integer(a[@"allocation_bytes"],64*1024*1024+256,&allocation) || allocation!=length+256 ||
          !integer(a[@"offset"],128,&offset) || offset!=128) return refuse("allocation_extent");
      id policy=a[@"value_policy"];
      if (policy && ![policy isEqual:@"finite-v1"] && ![policy isEqual:@"raw-bits-v1"])
        return refuse("allocation_value_policy");
      valuePolicy[i]=[policy isEqual:@"raw-bits-v1"]?G17_ATTN_RAW_BITS:G17_ATTN_FINITE_VALUES;
      NSArray *shape=a[@"shape"];
      if (![shape isKindOfClass:NSArray.class] || !shape.count || shape.count>4)
        return refuse("allocation_shape");
      NSUInteger words=1;
      for (id dimension in shape) {
        NSUInteger n;
        if (!integer(dimension,16*1024*1024,&n) || !n || words>16*1024*1024/n)
          return refuse("allocation_shape");
        words*=n;
      }
      NSUInteger width=([a[@"element_type"] isEqual:@"half"] || [a[@"element_type"] isEqual:@"ushort"])?2:4;
      NSString *kind=a[@"element_type"];
      NSUInteger lanes=([kind isEqual:@"float2"] || [kind isEqual:@"uint2"])?2:
               ([kind isEqual:@"float4"] || [kind isEqual:@"uint4"])?4:1;
      width*=lanes;
      if ([kind isEqual:@"long"] || [kind isEqual:@"ulong"]) width=8;
      bindingWidth[i]=width;
      if (words*width!=length) return refuse("shape_extent_disagreement");
      NSUInteger r=[roles indexOfObject:a[@"role"]];
      if (r==NSNotFound) return refuse("allocation_role");
      payload[i]=length;role[i]=(unsigned)r;
      type[i]=[a[@"element_type"] isEqual:@"ushort"]?G17_ATTN_UINT16:[a[@"element_type"] isEqual:@"half"]?G17_ATTN_FLOAT16:(([kind isEqual:@"atomic_uint"] || [kind isEqual:@"long"] || [kind isEqual:@"ulong"] || [kind isEqual:@"uint"] || [kind isEqual:@"uint2"] || [kind isEqual:@"uint4"])?G17_ATTN_UINT32:G17_ATTN_FLOAT32);++i;
    }
    G17AttentionStorage storage;
    if (!g17AttentionStorageInitTyped(&storage,names.count,payload,role,type))
      return refuse("storage_contract");
    for (NSUInteger j=0;j<names.count;++j) {
      if (!g17AttentionSetBindingWidth(&storage,j,bindingWidth[j])) return refuse("binding_lane_extent");
      if (!g17AttentionSetValuePolicy(&storage,j,valuePolicy[j])) return refuse("allocation_value_policy");
    }
    if (!g17AttentionConfigureOutput(&storage,graph))
      return refuse("output_initialization");
    if (!g17AttentionConfigureIntermediates(&storage,graph,names))
      return refuse("intermediate_initialization");
    NSUInteger transportRows=0,transportColumns=0;
    if (!attentionTransportDimensions(allocations[names[storage.source]][@"shape"],
          storage.payload[storage.source],g17AttentionElementBytes(storage.type[storage.source]),&transportRows,&transportColumns))
      return refuse("transport_shape");
    NSArray *schedule=nil;
    if (load || run || !strcmp(argv[2],"--describe-schedule")) {
      schedule=attentionSchedule(graph,names,&storage);
      if (!schedule) return refuse("schedule_contract");
    }
    if (load || run) {
      NSUInteger prefix=0,limit=0;
      if (run) {
        char *end=NULL;prefix=strtoul(argv[3],&end,10);
        if (!end || *end || !prefix || prefix>schedule.count) return refuse("prefix");
        limit=strtoul(argv[4],&end,10);
        if (!end || *end || !limit || limit>10) return refuse("query_limit");
      }
      return attentionMetal(manifest,bundle,schedule,names,&storage,
        run?[NSString stringWithUTF8String:argv[2]]:nil,prefix,limit);
    }
    NSMutableArray *layout=[NSMutableArray array];
    for (i=0;i<names.count;++i)
      [layout addObject:@{@"name":names[i],@"payload_bytes":@(storage.payload[i]),
        @"allocation_bytes":@(storage.allocation[i]),@"offset":@(G17_ATTN_GUARD),
        @"role":roles[storage.role[i]],
        @"value_policy":storage.value_policy[i]==G17_ATTN_RAW_BITS?@"raw-bits-v1":@"finite-v1"}];
    NSMutableDictionary *report=[@{@"allocations":layout,@"source":names[storage.source],
      @"gpu_dispatched":@NO,@"scope":@"host planning only; images and arithmetic not admitted"} mutableCopy];
    if (schedule) {
      report[@"schedule"]=schedule;
      NSDictionary *reply=attentionReplyPlan(schedule,&storage,schedule.count);
      if(reply) report[@"reply_plan"]=reply;
    }
    NSData *json=[NSJSONSerialization dataWithJSONObject:report options:0 error:nil];
    if (!json) return refuse("report");
    fwrite(json.bytes,1,json.length,stdout);return 0;
  }
}
