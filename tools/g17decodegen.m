// decodegen: the FUNCTIONAL real-generation executor for rung 2. Unlike decoderun (which times the dispatch
// chain with recorded per-stage inputs), this runs a genuine greedy decode: each token is ONE command buffer
// of the whole model (24 layers x 16 dispatches + final norm + 12 lm_head), with every stage's output flowing
// on-device into the next stage's input through shared ARENAS (no host copies between stages), the KV cache
// grown in place. After each token: wait, read the logits, host argmax, look up the next token's embedding and
// write it as the next step's input, bump the runtime length words, repeat. Correctness first (synchronous
// loop; the commit->complete round trip is ~0.16 ms against ~14 ms of GPU time, ~1%); double-buffering the
// token command buffers is a later optimization.
//
// The plan is Piece A's graph.json (solved + CPU-verified against the model reference BEFORE any GPU run):
//   arenas:            {name: total_bytes}                       one MTLBuffer per arena (+ guard)
//   arena_init:        [{arena, offset, file}]                   initial bytes: weights, rope tables,
//                                                                prefilled KV at the prompt length, token-P embedding
//   dispatches:        [{bundle, binds:{"1":{arena,offset,bytes},"2":..,"3":..}, threads, group}]  in order
//   per_token_writes:  [{arena, offset, bytes, source}]          source in {embedding_row, rope_cos_row,
//                                                                rope_sin_row, rope_len_word, split_len_word}
//   logits_readback:   {arena, offset, bytes}
//   embed_table:       file (vocab x d_model fp16)               argmax id -> embedding row
//   d_model, vocab, guard (default 128)
//
//   decodegen graph.json out.json [tokens]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include "g17gpulock.h"

static double now_s(void) {
  static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

static id<MTLComputePipelineState> bundlePipeline(id<MTLDevice> dev, NSString *dir) {
  NSError *e = nil;
  // THE MATCHED STUDY (MM 25.211): a twin directory holds twin.json {metallib, fn}, a library Apple's compiler built
  // from Metal source (tools/g17twin.py graph); its pipeline is Apple's code, everything else in the graph unchanged
  NSString *tj = [dir stringByAppendingPathComponent:@"twin.json"];
  if ([[NSFileManager defaultManager] fileExistsAtPath:tj]) {
    NSDictionary *t = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:tj] options:0 error:&e];
    id<MTLLibrary> tl = [dev newLibraryWithURL:[NSURL fileURLWithPath:t[@"metallib"]] error:&e];
    id<MTLFunction> tf = [tl newFunctionWithName:t[@"fn"]];
    id<MTLComputePipelineState> tp = tf ? [dev newComputePipelineStateWithFunction:tf error:&e] : nil;
    if (!tp) { fprintf(stderr, "twin %s: %s\n", dir.UTF8String, e.description.UTF8String); exit(2); }
    return tp;
  }
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

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 3) { fprintf(stderr, "usage: decodegen graph.json out.json [tokens]\n"); return 2; }
    NSUInteger tokens = argc >= 4 ? (NSUInteger)atoi(argv[3]) : 224;
    NSDictionary *g = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[1])] options:0 error:nil];
    if (!g) { fprintf(stderr, "graph unreadable\n"); return 2; }
    NSUInteger guard = g[@"guard"] ? [g[@"guard"] unsignedIntegerValue] : 128;
    NSUInteger vocab = [g[@"vocab"] unsignedIntegerValue];
    NSUInteger dmodel = [g[@"d_model"] unsignedIntegerValue];
    if (!dmodel) {                                         // derive from the embedding_row write width (fp16)
      for (NSDictionary *w in g[@"per_token_writes"] ?: @[])
        if ([w[@"source"] isEqualToString:@"embedding_row"]) { dmodel = [w[@"bytes"] unsignedIntegerValue] / 2; break; }
    }
    if (!dmodel) { fprintf(stderr, "d_model unknown (no field, no embedding_row write)\n"); return 2; }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue];

    // one MTLBuffer per arena: total_bytes + 2*guard, guard-filled so an overrun is visible
    NSMutableDictionary<NSString *, id<MTLBuffer>> *arena = [NSMutableDictionary dictionary];
    [g[@"arenas"] enumerateKeysAndObjectsUsingBlock:^(NSString *name, NSNumber *bytes, BOOL *stop) {
      NSUInteger n = bytes.unsignedIntegerValue + 2 * guard;
      id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
      memset(b.contents, 0xA5, n);
      arena[name] = b;
    }];
    // helper: a raw pointer into an arena at a graph offset (past the guard base)
    void *(^at)(NSString *, NSUInteger) = ^void *(NSString *name, NSUInteger off) {
      id<MTLBuffer> b = arena[name];
      if (!b) { fprintf(stderr, "unknown arena %s\n", name.UTF8String); exit(2); }
      return (uint8_t *)b.contents + guard + off;
    };

    // zero_init: regions that must be zero once at allocation (KV cache past the length, GEMM A padding rows
    // 1-15, attention Q padding rows). Done AFTER the 0xA5 fill and BEFORE weights load in.
    for (NSDictionary *z in g[@"zero_init"] ?: @[]) {
      memset(at(z[@"arena"], [z[@"offset"] unsignedIntegerValue]), 0, [z[@"bytes"] unsignedIntegerValue]);
    }

    // arena_init: memcpy initial bytes (weights, rope tables, prefilled KV, the prompt-last-token embedding)
    for (NSDictionary *ini in g[@"arena_init"] ?: @[]) {
      NSData *d = [NSData dataWithContentsOfFile:ini[@"file"]];
      if (!d) { fprintf(stderr, "arena_init file %s\n", [ini[@"file"] UTF8String]); return 2; }
      memcpy(at(ini[@"arena"], [ini[@"offset"] unsignedIntegerValue]), d.bytes, d.length);
    }

    // the embedding table (vocab x d_model fp16), for argmax id -> next input embedding. A carries it as
    // tables.embedding (a .npy) or embed_table (raw); accept either, and skip a .npy header if present.
    NSString *embed_path = g[@"embed_table"] ?: (g[@"tables"] ? g[@"tables"][@"embedding"] : nil);
    NSData *embed = embed_path ? [NSData dataWithContentsOfFile:embed_path] : nil;
    if (!embed) { fprintf(stderr, "embedding table missing (embed_table / tables.embedding)\n"); return 2; }
    NSUInteger edata = 0;                                   // .npy: 6B magic \x93NUMPY, 2B ver, 2B hlen, header
    const uint8_t *eb = (const uint8_t *)embed.bytes;
    if (embed.length > 10 && eb[0] == 0x93 && !memcmp(eb + 1, "NUMPY", 5)) {
      uint16_t hlen = (uint16_t)(eb[8] | (eb[9] << 8)); edata = 10 + hlen;   // v1.0 little-endian header length
    }
    const uint16_t *embed_rows = (const uint16_t *)(eb + edata);  // fp16 rows, dmodel wide

    // build the ordered dispatch list: pipeline + bound buffers/offsets + grid
    NSArray *disp = g[@"dispatches"];
    NSMutableArray *pipes = [NSMutableArray array];
    for (NSDictionary *c in disp) [pipes addObject:bundlePipeline(dev, c[@"bundle"])];
    // PREFILL (MM 25.142.10): optional steps the graph runs ONCE before decode - the whole prompt as M-row passes. Each
    // step applies its host writes (u32 words: a chunk's start position in the q0 word, say), then runs its dispatches
    // as one serial command buffer and waits. A write is {arena, offset, u32} or {arena, offset, bytes, copy_from: {arena,
    // offset}} (a host copy of GPU results between steps). After the last step the q0 word holds prompt_len (q0_after) and
    // decode continues from there. {"prefill": {"steps": [{"writes": [{arena, offset, u32}], "dispatches": [...]}],
    // "q0_after": L}}; the older {"dispatches": [...]} form is one step with no writes.
    NSArray *psteps = g[@"prefill"] ? (g[@"prefill"][@"steps"] ?: @[@{@"dispatches": g[@"prefill"][@"dispatches"] ?: @[]}]) : @[];
    NSMutableArray *ppipes = [NSMutableArray array];            // one pipeline array per step
    NSUInteger pdcount = 0;
    for (NSDictionary *st in psteps) {
      NSMutableArray *pl = [NSMutableArray array];
      for (NSDictionary *c in st[@"dispatches"] ?: @[]) [pl addObject:bundlePipeline(dev, c[@"bundle"])];
      [ppipes addObject:pl]; pdcount += pl.count;
    }
    NSArray *pdisp = psteps.count ? psteps : nil;

    NSDictionary *lr = g[@"logits_readback"];
    NSArray *ptw = g[@"per_token_writes"];

    // DECODEGEN_CONCURRENT: encode with MTLDispatchTypeConcurrent + a buffer-scope barrier only before a
    // dispatch that READS a region a prior (since-last-barrier) dispatch WROTE - like MLX - instead of the
    // serial encoder's implicit drain between EVERY dispatch. Slot 3 (C) is the write, all other bound slots
    // are reads (the tensor/qmv ABI: output carrier at binding+0). Outputs must equal the serial run (the check).
    BOOL concurrent = getenv("DECODEGEN_CONCURRENT") != NULL;
    if (concurrent) fprintf(stderr, "WARNING: DECODEGEN_CONCURRENT is EXPERIMENTAL and INCORRECT - the barrier "
        "analysis infers writes=slot3 only, but rope_append/attention write the cache via other bindings, so "
        "readers race (tokens differ from serial). It also showed no GPU speedup (serial chain). Do not use for "
        "correctness; needs per-binding read/write metadata in graph.json.\n");

    // DECODEGEN_TRUNC=K encodes only the first K dispatches per token, so timing K and K-1 and differencing gives
    // dispatch K-1's TRUE in-chain cost (real weight streaming + cache state + preceding-dispatch effect) -
    // Piece B's truncation-differencing cross-check, since AGX G17 counters hard-assert. K defaults to all.
    NSUInteger trunc = getenv("DECODEGEN_TRUNC") ? (NSUInteger)atoi(getenv("DECODEGEN_TRUNC")) : (NSUInteger)disp.count;
    if (trunc > (NSUInteger)disp.count) trunc = disp.count;

    void (^encodeList)(id<MTLComputeCommandEncoder>, NSArray *, NSArray *) = ^(id<MTLComputeCommandEncoder> enc, NSArray *list, NSArray *pl) {
      for (NSUInteger i = 0; i < list.count; ++i) {
        NSDictionary *c = list[i]; NSDictionary *binds = c[@"binds"];
        [enc setComputePipelineState:pl[i]];
        for (NSString *slot in binds) { NSDictionary *bd = binds[slot];
          [enc setBuffer:arena[bd[@"arena"]] offset:guard + [bd[@"offset"] unsignedIntegerValue] atIndex:(NSUInteger)[slot integerValue]]; }
        NSUInteger threads = [c[@"threads"] unsignedIntegerValue], group = [c[@"group"] unsignedIntegerValue];
        [enc dispatchThreadgroups:MTLSizeMake(threads / group, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
      }
    };
    // encode one token's whole-model command buffer: every dispatch binds its arenas at graph offsets (+guard)
    void (^encodeToken)(id<MTLComputeCommandEncoder>) = ^(id<MTLComputeCommandEncoder> enc) {
      NSMutableArray *written = [NSMutableArray array];       // {arena, lo, hi} written since the last barrier
      for (NSUInteger i = 0; i < trunc; ++i) {
        NSDictionary *c = disp[i];
        NSDictionary *binds = c[@"binds"];
        if (concurrent) {                                    // barrier before this dispatch iff it reads a pending write
          BOOL dep = NO;
          for (NSString *slot in binds) {
            if ([slot isEqualToString:@"3"]) continue;       // slot 3 is the write, not a read
            NSDictionary *bd = binds[slot];
            NSString *ar = bd[@"arena"]; NSUInteger lo = [bd[@"offset"] unsignedIntegerValue];
            NSUInteger hi = lo + [bd[@"bytes"] unsignedIntegerValue];
            for (NSDictionary *w in written)
              if ([w[@"arena"] isEqualToString:ar] && lo < [w[@"hi"] unsignedIntegerValue] && [w[@"lo"] unsignedIntegerValue] < hi) { dep = YES; break; }
            if (dep) break;
          }
          if (dep) { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; [written removeAllObjects]; }
        }
        [enc setComputePipelineState:pipes[i]];
        for (NSString *slot in binds) {                       // slot "1"/"2"/"3" -> index 1/2/3
          NSDictionary *bd = binds[slot];
          id<MTLBuffer> b = arena[bd[@"arena"]];
          [enc setBuffer:b offset:guard + [bd[@"offset"] unsignedIntegerValue] atIndex:(NSUInteger)[slot integerValue]];
        }
        NSUInteger threads = [c[@"threads"] unsignedIntegerValue], group = [c[@"group"] unsignedIntegerValue];
        [enc dispatchThreadgroups:MTLSizeMake(threads / group, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
        if (concurrent) {                                    // record this dispatch's write (slot 3)
          NSDictionary *w3 = binds[@"3"];
          if (w3) [written addObject:@{@"arena": w3[@"arena"], @"lo": w3[@"offset"],
                                       @"hi": @([w3[@"offset"] unsignedIntegerValue] + [w3[@"bytes"] unsignedIntegerValue])}];
        }
      }
    };

    // decode protocol (A's): the prompt runs through the SAME decode steps - no prefilled KV, the cache fills as
    // the prompt runs. For position p < prompt_len the input is embed[prompt_ids[p]]; after that it is the
    // previous step's argmax. The runtime length word = p (kv_len). cos/sin are PER-TOKEN rows: row p of the
    // rope tables (rope_cos_file / rope_sin_file, [C x n] fp32). All of these are optional (a bare single-op
    // smoke graph carries none), defaulting to prompt_len 0 and last_argmax 0.
    NSArray *prompt_ids = g[@"prompt_ids"] ?: @[];
    uint32_t prompt_len = g[@"prompt_len"] ? [g[@"prompt_len"] unsignedIntValue] : (uint32_t)prompt_ids.count;
    NSData *rcos = g[@"rope_cos_file"] ? [NSData dataWithContentsOfFile:g[@"rope_cos_file"]] : nil;
    NSData *rsin = g[@"rope_sin_file"] ? [NSData dataWithContentsOfFile:g[@"rope_sin_file"]] : nil;
    const float *cos_tab = rcos ? (const float *)rcos.bytes : NULL;
    const float *sin_tab = rsin ? (const float *)rsin.bytes : NULL;

    NSMutableArray<NSNumber *> *out_ids = [NSMutableArray array];
    double t_decode = 0.0, t_gpu = 0.0;                     // cb-roundtrip wall and GPU-execution time (timed tokens)
    double t_full = 0.0, t_argmax = 0.0;                    // FULL per-token wall (incl writes+argmax) and argmax time
    uint32_t last_argmax = g[@"first_token"] ? [g[@"first_token"] unsignedIntValue] : 0;
    NSInteger total = (NSInteger)prompt_len + (NSInteger)tokens;    // prefill positions + decode positions
    // warm-up runs of position 0 (clock ramp + resident pipelines), not timed/collected. It writes the KV
    // cache at position 0, which the real prefill overwrites identically - but it is configurable (warmup 0)
    // so a correctness run can start from a pristine zero_init cache. Default 3 for timing.
    NSInteger warmup = g[@"warmup"] ? [g[@"warmup"] integerValue] : 3;
    // localization mode (env): dump each dispatch's bound regions at a chosen step, running dispatches in
    // SEPARATE command buffers so each completes (its outputs visible) before the next - and before any later
    // dispatch overwrites a shared region. DECODEGEN_DUMP_DIR set = on; STEP (default 0); MAX dispatches (default all).
    const char *dump_dir = getenv("DECODEGEN_DUMP_DIR");
    NSInteger dump_step = getenv("DECODEGEN_DUMP_STEP") ? atoi(getenv("DECODEGEN_DUMP_STEP")) : 0;
    NSInteger dump_max = getenv("DECODEGEN_DUMP_MAX") ? atoi(getenv("DECODEGEN_DUMP_MAX")) : (NSInteger)disp.count;
    // logits dump: save the logits_readback region (vocab fp32) every N decode positions, for the logit-bound
    // comparison against mlx-lm fp16. DECODEGEN_LOGITS_DIR set = on; EVERY (default 8).
    const char *logits_dir = getenv("DECODEGEN_LOGITS_DIR");
    NSInteger logits_every = getenv("DECODEGEN_LOGITS_EVERY") ? atoi(getenv("DECODEGEN_LOGITS_EVERY")) : 8;
    // profile mode: at dump_step, run each dispatch in its own command buffer and record its GPU time
    // (GPUEndTime-GPUStartTime, clean of the per-cb fixed cost) to DECODEGEN_PROFILE as [{i,name,gpu_us}].
    const char *profile_path = getenv("DECODEGEN_PROFILE");
    // stage profile: per-dispatch GPU timestamps INSIDE the one chained token command buffer (sampleCounters at
    // each dispatch boundary), so in-chain effects (real weight streaming, arena warmup, preceding dispatch)
    // show - unlike the own-cb profile. Normalized to the cb's true GPU span, so tick units don't matter.
    const char *stage_profile = getenv("DECODEGEN_STAGE_PROFILE");
    id<MTLCounterSampleBuffer> sampleBuf = nil;
    if (stage_profile) {
      // AGX G17 reports supportsCounterSampling:AtStageBoundary == YES but sampleCountersInBuffer:atSampleIndex:
      // withBarrier: hard-asserts as unsupported. So the flag is NOT trustworthy here and the per-dispatch
      // counter path is dead on this GPU. Gate it behind DECODEGEN_FORCE_COUNTERS (off) to avoid the crash;
      // for in-chain per-dispatch timing use the truncation-differencing fallback instead.
      if (getenv("DECODEGEN_FORCE_COUNTERS")) {
        id<MTLCounterSet> tsSet = nil;
        for (id<MTLCounterSet> cs in dev.counterSets) if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) tsSet = cs;
        if (tsSet) {
          MTLCounterSampleBufferDescriptor *sd = [[MTLCounterSampleBufferDescriptor alloc] init];
          sd.counterSet = tsSet; sd.storageMode = MTLStorageModeShared; sd.sampleCount = disp.count + 2;
          NSError *se = nil; sampleBuf = [dev newCounterSampleBufferWithDescriptor:sd error:&se];
        }
      } else {
        fprintf(stderr, "stage profile: per-dispatch counter sampling is unsupported on AGX G17 (the "
            "stage-boundary flag lies); skipping. Use the own-cb profile or truncation differencing.\n");
      }
    }
    // PIPELINED mode (device-resident generation graph: per_token_writes=[], gen_step advances q0 + writes the
    // next embedding on the GPU). Encode T identical token command buffers and commit them ALL back to back with
    // NO wait between - the GPU runs them saturated (no host gap), so the commit->complete round trip is hidden
    // AND the clock stays high. Read the generated tokens from gen_region's log at the end. This is the fair
    // match to mlx-lm, which pipelines its own sampling. R (q0 + log + tables) is initialised by arena_init's
    // R_init.bin; between the warm and timed passes only that + zero_init are re-applied (weights are read-only).
    if (getenv("DECODEGEN_PIPELINED")) {
      NSDictionary *gr = g[@"gen_region"];
      if (!gr) { fprintf(stderr, "pipelined mode needs gen_region {arena,offset,log_offset,q0_offset,log_entries}\n"); return 2; }
      // Each command buffer is ONE position: the first prompt_len are prefill (they build the KV and re-emit the
      // prompt), the rest generate. To GENERATE `tokens` new tokens we submit prompt_len + tokens command
      // buffers, and report throughput over the generated steps only (as mlx-lm's generation_tps does).
      NSUInteger PL = g[@"prompt_ids"] ? [g[@"prompt_ids"] count] : 0;
      NSUInteger N = tokens;                                  // generated tokens
      NSUInteger M = PL + N;                                  // total command buffers (prefill + generate)
      NSString *garena = gr[@"arena"];
      NSUInteger logoff = [gr[@"offset"] unsignedIntegerValue] + [gr[@"log_offset"] unsignedIntegerValue];
      NSUInteger entries = gr[@"log_entries"] ? [gr[@"log_entries"] unsignedIntegerValue] : 272;
      if (M > entries) { fprintf(stderr, "pipelined: prompt_len %lu + tokens %lu > log capacity %lu\n",
                                 (unsigned long)PL, (unsigned long)N, (unsigned long)entries); return 2; }
      void (^reinit)(void) = ^{                              // reset the mutable gen state (q0/log), keep weights
        for (NSDictionary *z in g[@"zero_init"] ?: @[]) memset(at(z[@"arena"], [z[@"offset"] unsignedIntegerValue]), 0, [z[@"bytes"] unsignedIntegerValue]);
        for (NSDictionary *ini in g[@"arena_init"] ?: @[])
          if ([[ini[@"file"] lastPathComponent] isEqualToString:@"R_init.bin"]) {
            NSData *d = [NSData dataWithContentsOfFile:ini[@"file"]];
            if (d) memcpy(at(ini[@"arena"], [ini[@"offset"] unsignedIntegerValue]), d.bytes, d.length);
          }
      };
      // DECODEGEN_PIPE_GROUP=G packs G token-steps into each command buffer (device-gen chains them within the
      // cb via serial dispatch), cutting the number of cbs and any inter-cb scheduling gap. G=1 is the default
      // (one cb per token). Keep G small enough that a cb stays well under the GPU watchdog (G*~8ms).
      NSUInteger G = getenv("DECODEGEN_PIPE_GROUP") ? (NSUInteger)atoi(getenv("DECODEGEN_PIPE_GROUP")) : 1;
      if (G < 1) G = 1;
      // one pass submits ceil(count/G) command buffers (G token-steps each) back-to-back; returns the wall
      double (^pass)(NSUInteger) = ^double(NSUInteger count){
        NSMutableArray *cbs = [NSMutableArray array];
        double t0 = now_s();
        for (NSUInteger n = 0; n < count; n += G) {
          id<MTLCommandBuffer> cb = g17_gpu_cb(q);
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
          for (NSUInteger g = 0; g < G && n + g < count; ++g) encodeToken(enc);   // G token-steps in one cb
          [enc endEncoding]; [cb commit]; [cbs addObject:cb];
        }
        [cbs.lastObject waitUntilCompleted];
        return now_s() - t0;
      };
      // TIME TO FIRST TOKEN: from submission to the first generated token being in the log. Token by token that is the
      // prompt_len prompt steps plus the first generating step; with a prefill list it is the one prefill command
      // buffer (which leaves q0 = prompt_len) plus the first decode step.
      double ttft = -1, prefill_s = -1;
      NSUInteger dsteps = M;                                 // decode command buffers after the (optional) prefill
      if (pdisp.count) {
        NSUInteger q0a = [g[@"prefill"][@"q0_after"] unsignedIntegerValue];
        if (q0a != PL) { fprintf(stderr, "prefill: q0_after %lu != prompt_len %lu\n", (unsigned long)q0a, (unsigned long)PL); return 2; }
        dsteps = N;                                          // the prompt steps are replaced by the prefill pass
      }
      void (^applyWrites)(NSDictionary *) = ^(NSDictionary *st) {   // a step's host writes, before its command buffer
        for (NSDictionary *w in st[@"writes"] ?: @[]) {
          if (w[@"copy_from"]) {                               // a host copy between steps (the last prompt row, say)
            NSDictionary *f = w[@"copy_from"];
            memcpy(at(w[@"arena"], [w[@"offset"] unsignedIntegerValue]), at(f[@"arena"], [f[@"offset"] unsignedIntegerValue]),
                   [w[@"bytes"] unsignedIntegerValue]);
          } else {
            *(uint32_t *)at(w[@"arena"], [w[@"offset"] unsignedIntegerValue]) = (uint32_t)[w[@"u32"] unsignedIntegerValue];
          }
        }
      };
      // DECODEGEN_PREFILL_ONECB=1 (MM 25.183): the whole prefill in ONE command buffer. Each step's host writes become a
      // blit encoder ahead of its compute encoder - a copy_from is a buffer copy, a u32 comes from a staging buffer
      // filled here, outside every timing - so the GPU never idles through a commit, a wait and a host memcpy between
      // steps. The arenas are tracked resources, so the encoders run in order and a copy_from reads the step before it.
      BOOL onecb = getenv("DECODEGEN_PREFILL_ONECB") && atoi(getenv("DECODEGEN_PREFILL_ONECB")) > 0;
      NSUInteger nstage = 0;
      for (NSDictionary *st in psteps) for (NSDictionary *w in st[@"writes"] ?: @[]) if (!w[@"copy_from"]) ++nstage;
      id<MTLBuffer> stage = [dev newBufferWithLength:4 * (nstage ? nstage : 1) options:MTLResourceStorageModeShared];
      {
        NSUInteger k = 0;
        for (NSDictionary *st in psteps) for (NSDictionary *w in st[@"writes"] ?: @[])
          if (!w[@"copy_from"]) ((uint32_t *)stage.contents)[k++] = (uint32_t)[w[@"u32"] unsignedIntegerValue];
      }
      double (^first)(void) = ^double(void) {                // prefill (or the prompt steps) + one generating step
        double t0 = now_s();
        if (pdisp.count && onecb) {
          id<MTLCommandBuffer> cb = g17_gpu_cb(q);
          NSUInteger k = 0;
          for (NSUInteger si = 0; si < psteps.count; ++si) {
            NSDictionary *st = psteps[si];
            NSArray *ws = st[@"writes"] ?: @[];
            if ([st[@"setup"] boolValue]) {                 // setup steps ran once; skip their staged words too
              for (NSDictionary *w in ws) if (!w[@"copy_from"]) ++k;
              continue;
            }
            if (ws.count) {
              id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
              for (NSDictionary *w in ws) {
                NSUInteger dst = guard + [w[@"offset"] unsignedIntegerValue];
                if (w[@"copy_from"]) {
                  NSDictionary *f = w[@"copy_from"];
                  [bl copyFromBuffer:arena[f[@"arena"]] sourceOffset:guard + [f[@"offset"] unsignedIntegerValue]
                            toBuffer:arena[w[@"arena"]] destinationOffset:dst size:[w[@"bytes"] unsignedIntegerValue]];
                } else {
                  [bl copyFromBuffer:stage sourceOffset:4 * k++ toBuffer:arena[w[@"arena"]] destinationOffset:dst size:4];
                }
              }
              [bl endEncoding];
            }
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
            encodeList(enc, st[@"dispatches"] ?: @[], ppipes[si]); [enc endEncoding];
          }
          [cb commit]; [cb waitUntilCompleted];
          return now_s() - t0;
        }
        if (pdisp.count) {
          for (NSUInteger si = 0; si < psteps.count; ++si) {
            NSDictionary *st = psteps[si];
            if ([st[@"setup"] boolValue]) continue;         // setup steps ran once, before any timing
            applyWrites(st);
            id<MTLCommandBuffer> cb = g17_gpu_cb(q);
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
            encodeList(enc, st[@"dispatches"] ?: @[], ppipes[si]); [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
          }
          return now_s() - t0;
        }
        return 0;
      };
      // SETUP steps (persistent W16: every projection dequantised once) run once here, outside every timing; their
      // outputs live in arenas that reinit() does not clear
      for (NSUInteger si = 0; si < psteps.count; ++si) {
        if (![psteps[si][@"setup"] boolValue]) continue;
        id<MTLCommandBuffer> cb = g17_gpu_cb(q);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        encodeList(enc, psteps[si][@"dispatches"] ?: @[], ppipes[si]); [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
      }
      // PREFILL DUMP (DECODEGEN_PREFILL_DUMP=dir): run the prefill once from a fresh state and write each of the graph's
      // prefill.dump ranges ({name, arena, offset, bytes}) to dir/<name>.bin, for a bit-exact check against the CPU
      // reference (g17prefillgraph.reference); then exit
      const char *pdump = getenv("DECODEGEN_PREFILL_DUMP");
      if (pdump && pdisp.count) {
        reinit(); first();
        for (NSDictionary *r in g[@"prefill"][@"dump"] ?: @[]) {
          NSData *d = [NSData dataWithBytes:at(r[@"arena"], [r[@"offset"] unsignedIntegerValue]) length:[r[@"bytes"] unsignedIntegerValue]];
          [d writeToFile:[NSString stringWithFormat:@"%s/%@.bin", pdump, r[@"name"]] atomically:YES];
        }
        printf("prefill dump: %lu ranges to %s\n", (unsigned long)[g[@"prefill"][@"dump"] count], pdump);
        return 0;
      }
      reinit(); if (pdisp.count) first(); pass(dsteps);      // warm: saturate + ramp the clock, discard
      reinit();
      if (pdisp.count) prefill_s = first();
      // DECODEGEN_PREFILL_REPEAT=R times R more prefills in this process, each after its own reinit(), and reports
      // them all: one timed prefill per process swung 20-40 ms between identical runs (MM 25.142.10), and a
      // distribution from one process says whether that swing is inside a process or between processes
      NSMutableArray *prefill_repeat = [NSMutableArray array];
      NSUInteger reps = getenv("DECODEGEN_PREFILL_REPEAT") ? (NSUInteger)atoi(getenv("DECODEGEN_PREFILL_REPEAT")) : 0;
      for (NSUInteger r = 0; pdisp.count && r < reps; ++r) { reinit(); [prefill_repeat addObject:@(first())]; }
      // DECODEGEN_PREFILL_PROFILE=path: where the prefill's time goes, two views, as JSON {steps, dispatches}.
      //  steps: each step is already its own command buffer, so its GPU span (GPUEnd - GPUStart) and the host wall
      //    around it - wall minus GPU is the host time between steps, which the GPU idles through;
      //  dispatches: each dispatch alone in its own command buffer, its GPU time alone. No in-chain overlap, so the
      //    sum exceeds the chained span; the per-name totals say which kernel kind to work on.
      // Run after the timed prefill, so it cannot disturb any number reported above.
      const char *pprof = getenv("DECODEGEN_PREFILL_PROFILE");
      if (pprof && pdisp.count) {
        NSMutableArray *stepRows = [NSMutableArray array], *dispRows = [NSMutableArray array];
        reinit();
        for (NSUInteger si = 0; si < psteps.count; ++si) {
          NSDictionary *st = psteps[si];
          if ([st[@"setup"] boolValue]) continue;
          double h0 = now_s();
          applyWrites(st);
          id<MTLCommandBuffer> cb = g17_gpu_cb(q);
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
          encodeList(enc, st[@"dispatches"] ?: @[], ppipes[si]); [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
          [stepRows addObject:@{@"step": @(si), @"dispatches": @([st[@"dispatches"] count]),
                                @"gpu_us": @((cb.GPUEndTime - cb.GPUStartTime) * 1e6), @"wall_us": @((now_s() - h0) * 1e6)}];
        }
        reinit();
        for (NSUInteger si = 0; si < psteps.count; ++si) {
          NSDictionary *st = psteps[si];
          if ([st[@"setup"] boolValue]) continue;
          applyWrites(st);
          NSArray *list = st[@"dispatches"] ?: @[];
          for (NSUInteger i = 0; i < list.count; ++i) {
            id<MTLCommandBuffer> cb = g17_gpu_cb(q);
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
            encodeList(enc, @[list[i]], @[ppipes[si][i]]); [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            NSDictionary *c = list[i];
            [dispRows addObject:@{@"step": @(si), @"i": @(i), @"name": c[@"name"] ?: @"", @"bundle": c[@"bundle"] ?: @"",
                                  @"threads": c[@"threads"] ?: @0, @"gpu_us": @((cb.GPUEndTime - cb.GPUStartTime) * 1e6)}];
          }
        }
        [[NSJSONSerialization dataWithJSONObject:@{@"steps": stepRows, @"dispatches": dispRows} options:0 error:nil]
            writeToFile:@(pprof) atomically:YES];
        printf("prefill profile: %lu steps, %lu dispatches to %s\n", (unsigned long)stepRows.count, (unsigned long)dispRows.count, pprof);
      }
      // with a prefill section its tail's gen_step already writes the first generated token: TTFT is the prefill alone
      if (pdisp.count) ttft = prefill_s;
      else { double t_first0 = now_s(); pass(PL + 1); ttft = now_s() - t_first0; }
      reinit(); if (pdisp.count) first();
      double wall_full = pass(dsteps);                       // timed: the decode steps (prompt steps too, token by token)
      // per-step cost is ~uniform, so gen-only wall = wall_full * N/M and gen tok/s = N/gen_wall = M/wall_full.
      double gen_wall = wall_full * (double)N / (double)dsteps;
      double per = gen_wall / N * 1e3;                        // per-generated-token ms (protocol-exact vs mlx-lm)
      double tps = (double)N / gen_wall;                      // == dsteps/wall_full (steady per-step rate)
      int32_t *lg = (int32_t *)at(garena, logoff);
      NSMutableArray *gen = [NSMutableArray array];           // GENERATED tokens = log[prompt_len .. prompt_len+N)
      for (NSUInteger i = PL; i < PL + N && i < entries; ++i) [gen addObject:@(lg[i])];
      NSMutableArray *prompt = [NSMutableArray array];
      for (NSUInteger i = 0; i < PL && i < entries; ++i) [prompt addObject:@(lg[i])];
      // BATCHED DECODE (gen_region.batch, MM 25.144.3): B sequences step together; sequence b's log sits state_stride b
      // bytes after sequence 0's. Report every sequence's generated tokens and the AGGREGATE rate (B N tokens per wall)
      NSUInteger B = gr[@"batch"] ? [gr[@"batch"] unsignedIntegerValue] : 1;
      NSUInteger sstride = gr[@"state_stride"] ? [gr[@"state_stride"] unsignedIntegerValue] : 0;
      NSMutableArray *perseq = [NSMutableArray array];
      for (NSUInteger s = 0; s < B; ++s) {
        int32_t *ls = (int32_t *)at(garena, logoff + sstride * s);
        NSMutableArray *gs = [NSMutableArray array];
        for (NSUInteger i = PL; i < PL + N && i < entries; ++i) [gs addObject:@(ls[i])];
        [perseq addObject:gs];
      }
      NSMutableDictionary *rep = [@{@"mode": @"decodegen_pipelined", @"generated_tokens": @(N), @"prompt_len": @(PL),
                            @"total_steps": @(M), @"dispatches_per_token": @(disp.count),
                            @"full_wall_s": @(wall_full), @"gen_wall_s": @(gen_wall),
                            @"per_token_ms": @(per), @"tokens_per_s": @(tps * (double)B), @"batch": @(B),
                            @"steps_per_s": @(tps), @"out_ids_per_seq": perseq,
                            @"out_ids": gen, @"prompt": prompt, @"ttft_s": @(ttft), @"prefill_s": @(prefill_s),
                            @"prefill_dispatches": @(pdcount), @"prefill_steps": @(psteps.count)} mutableCopy];
      if (reps) rep[@"prefill_repeat_s"] = prefill_repeat;
      [[NSJSONSerialization dataWithJSONObject:rep options:NSJSONWritingPrettyPrinted error:nil] writeToFile:@(argv[2]) atomically:YES];
      printf("decodegen PIPELINED: %lu generated (+%lu prompt), %.3f ms/token gen (%.1f tok/s), %lu disp\n",
             (unsigned long)N, (unsigned long)PL, per, tps, (unsigned long)disp.count);
      return 0;
    }

    for (NSInteger step = -warmup; step < total; ++step) {
      double iter0 = now_s();                                // FULL per-token wall: writes + dispatch + wait + argmax
      NSInteger p = step < 0 ? 0 : step;
      uint32_t input_token = (p < (NSInteger)prompt_len) ? (uint32_t)[prompt_ids[p] unsignedIntValue] : last_argmax;
      uint32_t kv_len = (uint32_t)p;
      // apply per-token host writes: the input embedding row, the runtime length words, and the rope rows
      for (NSDictionary *w in ptw) {
        NSString *src = w[@"source"]; NSUInteger off = [w[@"offset"] unsignedIntegerValue];
        NSUInteger bytes = [w[@"bytes"] unsignedIntegerValue];
        if ([src isEqualToString:@"embedding_row"]) {
          memcpy(at(w[@"arena"], off), embed_rows + (size_t)input_token * dmodel, dmodel * sizeof(uint16_t));
        } else if ([src hasSuffix:@"len_word"]) {           // any *_len_word slot (rope/split/attn/...) = kv_len uint32
          *(uint32_t *)at(w[@"arena"], off) = kv_len;
        } else if ([src isEqualToString:@"rope_cos_row"] && cos_tab) {
          memcpy(at(w[@"arena"], off), cos_tab + (size_t)p * (bytes / 4), bytes);
        } else if ([src isEqualToString:@"rope_sin_row"] && sin_tab) {
          memcpy(at(w[@"arena"], off), sin_tab + (size_t)p * (bytes / 4), bytes);
        }
      }
      double t0 = now_s(); double gpu_dt = 0.0;
      if (stage_profile && sampleBuf && step == dump_step) {
        // ONE command buffer, sampling the GPU timestamp at each dispatch boundary -> in-chain per-dispatch time
        id<MTLCommandBuffer> cb = g17_gpu_cb(q);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        NSUInteger si = 0;
        [enc sampleCountersInBuffer:sampleBuf atSampleIndex:si++ withBarrier:YES];
        for (NSUInteger i = 0; i < disp.count; ++i) {
          NSDictionary *c = disp[i]; NSDictionary *binds = c[@"binds"];
          [enc setComputePipelineState:pipes[i]];
          for (NSString *slot in binds) { NSDictionary *bd = binds[slot];
            [enc setBuffer:arena[bd[@"arena"]] offset:guard + [bd[@"offset"] unsignedIntegerValue] atIndex:(NSUInteger)[slot integerValue]]; }
          NSUInteger threads = [c[@"threads"] unsignedIntegerValue], group = [c[@"group"] unsignedIntegerValue];
          [enc dispatchThreadgroups:MTLSizeMake(threads / group, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
          [enc sampleCountersInBuffer:sampleBuf atSampleIndex:si++ withBarrier:YES];
        }
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        double span = cb.GPUEndTime - cb.GPUStartTime;
        NSData *resd = [sampleBuf resolveCounterRange:NSMakeRange(0, si)];
        const MTLCounterResultTimestamp *ts = (const MTLCounterResultTimestamp *)resd.bytes;
        double total = (double)(ts[si - 1].timestamp - ts[0].timestamp);
        NSMutableArray *prof = [NSMutableArray array];
        for (NSUInteger i = 0; i < disp.count; ++i) {
          double frac = total > 0 ? (double)(ts[i + 1].timestamp - ts[i].timestamp) / total : 0;
          [prof addObject:@{@"i": @(i), @"name": disp[i][@"name"] ?: @(i).stringValue, @"gpu_us": @(frac * span * 1e6)}];
        }
        [[NSJSONSerialization dataWithJSONObject:@{@"step": @(step), @"kv_len": @(kv_len), @"span_ms": @(span * 1e3),
            @"in_chain": @YES, @"dispatches": prof} options:NSJSONWritingPrettyPrinted error:nil] writeToFile:@(stage_profile) atomically:YES];
        fprintf(stderr, "stage-profiled step %ld (kv_len %u, in-chain span %.3f ms) to %s\n", (long)step, kv_len, span * 1e3, stage_profile);
      } else if ((dump_dir || profile_path) && step == dump_step) {
        // run each dispatch in its own command buffer (ordered, each visible before the next); record its GPU
        // time for the profile and/or dump its bound regions for a GPU-vs-reference byte diff.
        NSMutableArray *prof = [NSMutableArray array];
        for (NSUInteger i = 0; i < disp.count; ++i) {
          NSDictionary *c = disp[i]; NSDictionary *binds = c[@"binds"];
          id<MTLCommandBuffer> cb = g17_gpu_cb(q);
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
          [enc setComputePipelineState:pipes[i]];
          for (NSString *slot in binds) { NSDictionary *bd = binds[slot];
            [enc setBuffer:arena[bd[@"arena"]] offset:guard + [bd[@"offset"] unsignedIntegerValue] atIndex:(NSUInteger)[slot integerValue]]; }
          NSUInteger threads = [c[@"threads"] unsignedIntegerValue], group = [c[@"group"] unsignedIntegerValue];
          [enc dispatchThreadgroups:MTLSizeMake(threads / group, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
          [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
          if (profile_path)
            [prof addObject:@{@"i": @(i), @"name": c[@"name"] ?: @(i).stringValue,
                              @"gpu_us": @((cb.GPUEndTime - cb.GPUStartTime) * 1e6)}];
          if (dump_dir && (NSInteger)i < dump_max) for (NSString *slot in binds) {   // dump bound regions
            NSDictionary *bd = binds[slot];
            NSData *d = [NSData dataWithBytes:at(bd[@"arena"], [bd[@"offset"] unsignedIntegerValue]) length:[bd[@"bytes"] unsignedIntegerValue]];
            [d writeToFile:[NSString stringWithFormat:@"%s/%03lu_%@_s%@.bin", dump_dir, (unsigned long)i, c[@"name"] ?: @(i).stringValue, slot] atomically:YES];
          }
        }
        if (profile_path) {
          [[NSJSONSerialization dataWithJSONObject:@{@"step": @(step), @"kv_len": @(kv_len), @"dispatches": prof}
                                           options:NSJSONWritingPrettyPrinted error:nil] writeToFile:@(profile_path) atomically:YES];
          fprintf(stderr, "profiled step %ld (kv_len %u): %ld dispatches to %s\n", (long)step, kv_len, (long)disp.count, profile_path);
        }
      } else {
        id<MTLCommandBuffer> cb = g17_gpu_cb(q);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:
            (concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial)];
        encodeToken(enc); [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        gpu_dt = cb.GPUEndTime - cb.GPUStartTime;           // GPU execution time of this token (vs the wall dt)
      }
      double dt = now_s() - t0;

      // read logits, host argmax over the real vocab, that is the next token. The lm_head runs through a
      // split-K fp32 fold, so logits are fp32 by default; a graph may say "logits_dtype":"f16".
      BOOL logits_f16 = [g[@"logits_dtype"] isEqualToString:@"f16"];
      void *lp = at(lr[@"arena"], [lr[@"offset"] unsignedIntegerValue]);
      double arg0 = now_s();                                 // logits readback + argmax = part of the host time (c)
      float best = -1e30f; uint32_t arg = 0;
      for (uint32_t v = 0; v < vocab; ++v) {
        float fv;
        if (logits_f16) {
          uint16_t h = ((const uint16_t *)lp)[v];
          uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
          if (exp == 0) { if (man == 0) f = sign; else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3FF; f = sign | (exp << 23) | (man << 13); } }
          else if (exp == 0x1F) f = sign | 0x7F800000u | (man << 13);
          else f = sign | ((exp - 15 + 127) << 23) | (man << 13);
          memcpy(&fv, &f, 4);
        } else {
          fv = ((const float *)lp)[v];
        }
        if (fv > best) { best = fv; arg = v; }
      }
      double arg_dt = now_s() - arg0;                        // argmax + readback time
      last_argmax = arg;
      double full_dt = now_s() - iter0;                      // FULL per-token wall (writes + dispatch + wait + argmax)
      if (logits_dir && step >= 0 && (step % logits_every == 0)) {   // save logits (vocab wide) for the logit bound
        NSData *ld = [NSData dataWithBytes:lp length:(logits_f16 ? vocab * 2 : vocab * 4)];
        [ld writeToFile:[NSString stringWithFormat:@"%s/pos%04ld.f32", logits_dir, (long)p] atomically:YES];
      }
      // generated tokens = argmax(p) for p >= prompt_len-1 (the first prediction after the prompt), capped at
      // `tokens`; time the decode forwards p >= prompt_len (prefill excluded, matching the MLX bar protocol).
      if (step >= 0) {
        if (p >= (NSInteger)prompt_len - 1 && (NSInteger)out_ids.count < (NSInteger)tokens) [out_ids addObject:@(arg)];
        if (p >= (NSInteger)prompt_len) { t_decode += dt; t_gpu += gpu_dt; t_full += full_dt; t_argmax += arg_dt; }
      }
    }

    // FULL wall (t_full: writes+dispatch+wait+argmax) is the honest throughput; per_token_ms (cb roundtrip,
    // t_decode) matches the earlier dt-based number. Breakdown for the kernels-vs-gaps-vs-host question:
    //  gpu_ms = GPU span of the token's cb (a); host = full - gpu (c: writes+argmax+commit+wait);
    //  cb_roundtrip_ms = dt - gpu (commit+wait only); argmax_ms (part of c). Kernel sum (b) = the profile.
    double tps = tokens / t_full;                            // honest: includes the between-token host work
    NSMutableArray *ids = [NSMutableArray array]; for (NSNumber *n in out_ids) [ids addObject:n];
    NSDictionary *rep = @{@"mode": @"decodegen", @"tokens": @(tokens),
                          @"dispatches_per_token": @(disp.count),
                          @"full_wall_s": @(t_full), @"full_per_token_ms": @(t_full / tokens * 1e3),
                          @"cb_roundtrip_ms_per_token": @(t_decode / tokens * 1e3),
                          @"per_token_ms": @(t_full / tokens * 1e3),
                          @"gpu_ms_per_token": @(t_gpu / tokens * 1e3), @"gpu_s": @(t_gpu),
                          @"host_ms_per_token": @((t_full - t_gpu) / tokens * 1e3),
                          @"commit_wait_ms_per_token": @((t_decode - t_gpu) / tokens * 1e3),
                          @"argmax_ms_per_token": @(t_argmax / tokens * 1e3),
                          @"tokens_per_s": @(tps), @"out_ids": ids};
    NSData *out = [NSJSONSerialization dataWithJSONObject:rep options:NSJSONWritingPrettyPrinted error:nil];
    [out writeToFile:@(argv[2]) atomically:YES];
    printf("decodegen: %lu tokens | FULL %.3f ms/tok (%.1f tok/s) | GPU %.3f | host %.3f (argmax %.3f, commit+wait %.3f) | %lu disp\n",
           (unsigned long)tokens, t_full / tokens * 1e3, tps, t_gpu / tokens * 1e3,
           (t_full - t_gpu) / tokens * 1e3, t_argmax / tokens * 1e3, (t_decode - t_gpu) / tokens * 1e3,
           (unsigned long)disp.count);
  }
  return 0;
}
