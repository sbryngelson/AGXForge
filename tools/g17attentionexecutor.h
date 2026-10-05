#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "g17attentionstorage.h"

// Construct only after independent graph, image and delivered-code admission.
// A session owns all intermediate storage; returned bytes are diagnostics.
@interface G17AttentionExecutor : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)device
                       storage:(const G17AttentionStorage *)storage
                      schedule:(NSArray *)schedule
                     pipelines:(NSDictionary<NSString *,id<MTLComputePipelineState>> *)pipelines
                     snapshots:(NSArray *)snapshots
                         error:(NSError **)error;
// Optional read-only textures. The ordinary initializer supplies none.
- (instancetype)initWithDevice:(id<MTLDevice>)device
                       storage:(const G17AttentionStorage *)storage
                      schedule:(NSArray *)schedule
                     pipelines:(NSDictionary<NSString *,id<MTLComputePipelineState>> *)pipelines
                     snapshots:(NSArray *)snapshots
                      textures:(NSDictionary *)textures
              textureSnapshots:(NSDictionary<NSString *,NSData *> *)textureSnapshots
                         error:(NSError **)error;
- (NSArray *)textureBindingTrace;
- (NSArray<NSData *> *)runSource:(NSData *)source prefix:(NSUInteger)prefix error:(NSError **)error;
// One query and allocation lifetime, multiple ordered command buffers. The
// caller must admit the entire schedule and its cross-batch dependencies first.
- (NSArray<NSData *> *)runSource:(NSData *)source prefix:(NSUInteger)prefix
                   batchStages:(NSUInteger)batchStages error:(NSError **)error;
- (NSArray *)scratchObservations;
- (NSDictionary *)identity;
- (NSArray *)bindingTrace;
- (NSArray *)submissionTrace;
@end
