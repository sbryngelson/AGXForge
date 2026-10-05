// specgen: greedy SPECULATIVE decoding, verified 16 rows at a time (MM 25.203), with prompt-lookup drafts or a DRAFT
// MODEL on the same GPU (MM 25.205).
//
// The target graph is g17q4graph's with a "spec" section (config "spec"): decode's own dispatches (one position, the
// device-resident generation step), its prefill steps, and spec.dispatches - one VERIFY step: 16 rows at positions
// q0 .. q0+15 through every layer against the decode cache (the scalar prefill append / attention read p0 from decode's q0
// word and write what decode writes), then a 16-row final norm and head writing 16 logit rows.
//
// Protocol. State: the committed tokens t_0 .. t_{n-1}; the cache holds positions 0 .. n-2; q0 = n - 1; decode's input
// row holds embed(t_{n-1}); log[i] = t_i for i < n, and log[n..] = -1 (model-chosen).
//   plain step  (no draft):  decode's dispatches once; its gen step writes log[n] and the next input row, q0 = n.
//   verify step (a draft d_1 .. d_m, m <= 15): rows = t_{n-1}, d_1 .. d_m (padded with t_{n-1}) at positions n-1 ..;
//               row i's argmax a_i predicts position n + i. Accept d_1 .. d_j while a_{i-1} = d_i; commit a_0 .. a_j
//               (j + 1 tokens). The cache rows written past the committed ones are rewritten before any read (causal
//               rows never read them; a later append overwrites them). Then q0 = n + j, log, input row = embed(t_last).
// Every committed token is an argmax of the target at its position, so the output is the target's greedy decode.
//
// Drafts. Prompt lookup: the longest suffix of the history (ngram down to min_ngram tokens) that occurred earlier, and the
// tokens that followed its latest earlier occurrence, up to max_draft. With --draft G2 (a plain decode graph of a model
// with the same tokenizer, built for the same prompt), the draft model is used whenever prompt lookup finds nothing:
//   its state mirrors the target's (its own q0, log and input row, its own cache); `fed[i]` is the token its cache
//   holds at position i. Each round it first catches up - positions whose fed token is not the committed one are
//   re-fed with the committed token FORCED through its log (the gen step takes log[q0 + 1] when it is not -1) - then
//   runs k plain steps in ONE command buffer (its gen step keeps the next input on the device) and reads the k drafts
//   from its log.
//
// The policy (MM 25.205): a verify step costs c plain steps (c measured in the run: its wall over the plain step's), and
// it commits 1 + j tokens. Each draft source (lookup, model) keeps an exponential mean of 1 + j over its verifies; a
// draft is verified only when that mean exceeds c, so a source that stops paying falls back to plain steps. A declined
// source is still probed every 8 rounds, so the mean follows the text. --always restores "verify every draft".
//
//   specgen graph.json out.json [tokens] [--plain] [--draft draft_graph.json] [--k K] [--no-lookup] [--always]
//     --plain      plain steps only (the same driver, the baseline)
//     --k K        draft-model tokens per round (default 4, at most 15)
//     --no-lookup  the draft model only (prompt lookup off)
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

// ONE MODEL: its graph, its arenas (a model's arena names are its own), pipelines and generation state
@interface Model : NSObject
@property NSDictionary *g;
@property NSMutableDictionary<NSString *, id<MTLBuffer>> *arena;
@property NSUInteger guard, dmodel, cap, L;
@property NSMutableArray *pipes, *vpipes, *ppipes;
@property uint32_t *q0p; @property int32_t *logp; @property uint16_t *rx16;
@property const uint16_t *embed_rows; @property NSData *embed;
@end
@implementation Model
- (void *)at:(NSString *)name off:(NSUInteger)off {
  id<MTLBuffer> b = self.arena[name];
  if (!b) { fprintf(stderr, "unknown arena %s\n", name.UTF8String); exit(2); }
  return (uint8_t *)b.contents + self.guard + off;
}
@end

static Model *loadModel(id<MTLDevice> dev, const char *path, BOOL need_spec) {
  Model *m = [Model new];
  NSDictionary *g = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(path)] options:0 error:nil];
  if (!g || !g[@"gen_region"] || (need_spec && !g[@"spec"])) { fprintf(stderr, "%s: unreadable, or no gen_region / spec\n", path); exit(2); }
  m.g = g; m.guard = g[@"guard"] ? [g[@"guard"] unsignedIntegerValue] : 128; m.dmodel = [g[@"d_model"] unsignedIntegerValue];
  m.arena = [NSMutableDictionary dictionary];
  [g[@"arenas"] enumerateKeysAndObjectsUsingBlock:^(NSString *name, NSNumber *bytes, BOOL *stop) {
    (void)stop;
    NSUInteger n = bytes.unsignedIntegerValue + 2 * m.guard;
    id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
    memset(b.contents, 0xA5, n); m.arena[name] = b;
  }];
  for (NSDictionary *z in g[@"zero_init"] ?: @[]) memset([m at:z[@"arena"] off:[z[@"offset"] unsignedIntegerValue]], 0, [z[@"bytes"] unsignedIntegerValue]);
  for (NSDictionary *ini in g[@"arena_init"] ?: @[]) {
    NSData *d = [NSData dataWithContentsOfFile:ini[@"file"]];
    if (!d) { fprintf(stderr, "arena_init file %s\n", [ini[@"file"] UTF8String]); exit(2); }
    memcpy([m at:ini[@"arena"] off:[ini[@"offset"] unsignedIntegerValue]], d.bytes, d.length);
  }
  m.embed = [NSData dataWithContentsOfFile:g[@"tables"][@"embedding"]];
  const uint8_t *eb = (const uint8_t *)m.embed.bytes; NSUInteger edata = 0;
  if (m.embed.length > 10 && eb[0] == 0x93 && !memcmp(eb + 1, "NUMPY", 5)) edata = 10 + (uint16_t)(eb[8] | (eb[9] << 8));
  m.embed_rows = (const uint16_t *)(eb + edata);
  m.pipes = [NSMutableArray array]; m.vpipes = [NSMutableArray array]; m.ppipes = [NSMutableArray array];
  for (NSDictionary *c in g[@"dispatches"]) [m.pipes addObject:bundlePipeline(dev, c[@"bundle"])];
  for (NSDictionary *c in g[@"spec"][@"dispatches"] ?: @[]) [m.vpipes addObject:bundlePipeline(dev, c[@"bundle"])];
  for (NSDictionary *st in (g[@"prefill"] ? g[@"prefill"][@"steps"] : @[])) {
    NSMutableArray *pl = [NSMutableArray array];
    for (NSDictionary *c in st[@"dispatches"] ?: @[]) [pl addObject:bundlePipeline(dev, c[@"bundle"])];
    [m.ppipes addObject:pl];
  }
  NSDictionary *gr = g[@"gen_region"];
  NSUInteger goff = [gr[@"offset"] unsignedIntegerValue];
  m.q0p = (uint32_t *)[m at:gr[@"arena"] off:goff + [gr[@"q0_offset"] unsignedIntegerValue]];
  m.logp = (int32_t *)[m at:gr[@"arena"] off:goff + [gr[@"log_offset"] unsignedIntegerValue]];
  m.cap = [gr[@"log_entries"] unsignedIntegerValue];
  // decode's layer-0 input row: the graph names it (g17q4graph writes decode_x_row on every generation graph; a spec
  // graph also carries it as spec.r_x16)
  NSDictionary *rx = g[@"decode_x_row"] ?: (g[@"spec"] ? g[@"spec"][@"r_x16"] : nil);
  if (!rx) { fprintf(stderr, "%s: no decode_x_row (rebuild the graph)\n", path); exit(2); }
  m.rx16 = (uint16_t *)[m at:rx[@"arena"] off:[rx[@"offset"] unsignedIntegerValue]];
  m.L = [g[@"prompt_ids"] count];
  return m;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 3) { fprintf(stderr, "usage: specgen graph.json out.json [tokens] [--plain] [--draft G] [--k K] [--no-lookup]\n"); return 2; }
    NSUInteger tokens = argc >= 4 && argv[3][0] != '-' ? (NSUInteger)atoi(argv[3]) : 128;
    BOOL plain_only = NO, lookup = YES, always = NO; const char *draft_path = NULL, *vprof = NULL; NSUInteger K = 4;
    for (int i = 3; i < argc; ++i) {
      if (!strcmp(argv[i], "--plain")) plain_only = YES;
      else if (!strcmp(argv[i], "--no-lookup")) lookup = NO;
      else if (!strcmp(argv[i], "--always")) always = YES;
      else if (!strcmp(argv[i], "--draft") && i + 1 < argc) draft_path = argv[++i];
      else if (!strcmp(argv[i], "--k") && i + 1 < argc) K = (NSUInteger)atoi(argv[++i]);
      else if (!strcmp(argv[i], "--profile-verify") && i + 1 < argc) vprof = argv[++i];
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue];
    Model *T = loadModel(dev, argv[1], YES);
    Model *D = draft_path ? loadModel(dev, draft_path, NO) : nil;
    NSDictionary *sp = T.g[@"spec"];
    NSUInteger vocab = [sp[@"vocab"] unsignedIntegerValue], dmodel = T.dmodel;
    NSUInteger rows = [sp[@"rows"] unsignedIntegerValue], row_bytes = [sp[@"logits"][@"row_bytes"] unsignedIntegerValue];
    NSUInteger ngram = [sp[@"ngram"] unsignedIntegerValue], max_draft = [sp[@"max_draft"] unsignedIntegerValue];
    NSUInteger min_ngram = sp[@"min_ngram"] ? [sp[@"min_ngram"] unsignedIntegerValue] : 1;
    if (max_draft > rows - 1) max_draft = rows - 1;
    if (K > rows - 1) K = rows - 1;
    if (D && ![D.g[@"prompt_ids"] isEqual:T.g[@"prompt_ids"]]) { fprintf(stderr, "draft and target prompts differ\n"); return 2; }

    void (^encodeList)(Model *, id<MTLComputeCommandEncoder>, NSArray *, NSArray *) =
        ^(Model *m, id<MTLComputeCommandEncoder> enc, NSArray *list, NSArray *pl) {
      for (NSUInteger i = 0; i < list.count; ++i) {
        NSDictionary *c = list[i]; NSDictionary *binds = c[@"binds"];
        [enc setComputePipelineState:pl[i]];
        for (NSString *slot in binds) { NSDictionary *bd = binds[slot];
          [enc setBuffer:m.arena[bd[@"arena"]] offset:m.guard + [bd[@"offset"] unsignedIntegerValue] atIndex:(NSUInteger)[slot integerValue]]; }
        NSUInteger threads = [c[@"threads"] unsignedIntegerValue], group = [c[@"group"] unsignedIntegerValue];
        [enc dispatchThreadgroups:MTLSizeMake(threads / group, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
      }
    };
    // `reps` back-to-back encodings of one list in ONE command buffer (the draft's k plain steps)
    void (^runList)(Model *, NSArray *, NSArray *, NSUInteger) = ^(Model *m, NSArray *list, NSArray *pl, NSUInteger reps) {
      id<MTLCommandBuffer> cb = g17_gpu_cb(q);
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      for (NSUInteger r = 0; r < reps; ++r) encodeList(m, enc, list, pl);
      [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    };
    void (^applyWrites)(Model *, NSDictionary *) = ^(Model *m, NSDictionary *st) {
      for (NSDictionary *w in st[@"writes"] ?: @[]) {
        if (w[@"copy_from"]) { NSDictionary *f = w[@"copy_from"];
          memcpy([m at:w[@"arena"] off:[w[@"offset"] unsignedIntegerValue]], [m at:f[@"arena"] off:[f[@"offset"] unsignedIntegerValue]], [w[@"bytes"] unsignedIntegerValue]);
        } else *(uint32_t *)[m at:w[@"arena"] off:[w[@"offset"] unsignedIntegerValue]] = (uint32_t)[w[@"u32"] unsignedIntegerValue];
      }
    };
    void (^reinit)(Model *) = ^(Model *m) {
      for (NSDictionary *z in m.g[@"zero_init"] ?: @[]) memset([m at:z[@"arena"] off:[z[@"offset"] unsignedIntegerValue]], 0, [z[@"bytes"] unsignedIntegerValue]);
      for (NSDictionary *ini in m.g[@"arena_init"] ?: @[])
        if ([[ini[@"file"] lastPathComponent] isEqualToString:@"R_init.bin"]) {
          NSData *d = [NSData dataWithContentsOfFile:ini[@"file"]];
          memcpy([m at:ini[@"arena"] off:[ini[@"offset"] unsignedIntegerValue]], d.bytes, d.length);
        }
    };
    NSArray *(^psteps)(Model *) = ^NSArray *(Model *m) { return m.g[@"prefill"] ? m.g[@"prefill"][@"steps"] : @[]; };
    void (^prefill)(Model *) = ^(Model *m) {
      NSArray *ps = psteps(m);
      for (NSUInteger si = 0; si < ps.count; ++si) {
        if ([ps[si][@"setup"] boolValue]) continue;
        applyWrites(m, ps[si]); runList(m, ps[si][@"dispatches"], m.ppipes[si], 1);
      }
    };
    for (Model *m in (D ? @[T, D] : @[T])) {                          // setup steps (persistent W16) once
      NSArray *ps = psteps(m);
      for (NSUInteger si = 0; si < ps.count; ++si) if ([ps[si][@"setup"] boolValue]) runList(m, ps[si][@"dispatches"], m.ppipes[si], 1);
    }
    uint16_t *x16v = (uint16_t *)[T at:sp[@"x16"][@"arena"] off:[sp[@"x16"][@"offset"] unsignedIntegerValue]];
    const float *lg = (const float *)[T at:sp[@"logits"][@"arena"] off:[sp[@"logits"][@"offset"] unsignedIntegerValue]];
    // the GPU argmax's pairs (MM 25.205): row r's G (value, global index) pairs; the larger value, ties to the smaller index
    const float *pairs = sp[@"pairs"] ? (const float *)[T at:sp[@"pairs"][@"arena"] off:[sp[@"pairs"][@"offset"] unsignedIntegerValue]] : NULL;
    NSUInteger PG = sp[@"pairs"] ? [sp[@"pairs"][@"G"] unsignedIntegerValue] : 0, PV = sp[@"pairs"] ? [sp[@"pairs"][@"V"] unsignedIntegerValue] : 0;
    NSArray *prompt = T.g[@"prompt_ids"];
    NSUInteger L = prompt.count;

    NSMutableDictionary *(^generate)(void) = ^NSMutableDictionary *(void) {
      reinit(T); if (D) reinit(D);
      double t0 = now_s();
      prefill(T);
      double ttft = now_s() - t0;
      NSMutableArray<NSNumber *> *hist = [NSMutableArray array];
      for (NSNumber *p in prompt) [hist addObject:p];
      [hist addObject:@(T.logp[L])];                                 // the prefill tail's first token
      if (*T.q0p != L) { fprintf(stderr, "q0 after prefill %u != %lu\n", *T.q0p, (unsigned long)L); exit(3); }
      // the draft model's cache: fed[i] the token at position i (the prompt, after its own prefill)
      NSMutableArray<NSNumber *> *fed = [NSMutableArray array];
      if (D) {
        prefill(D);
        for (NSNumber *p in prompt) [fed addObject:p];                // positions 0 .. L-1 hold the prompt
      }
      NSUInteger nplain = 0, nverify = 0, accepted = 0, drafted = 0, nlookup = 0, nmodel = 0, dsteps = 0;
      NSMutableArray *acc_hist = [NSMutableArray array];
      double t_draft = 0, t_verify = 0, t_plain = 0;
      // the policy's state: the exponential mean of committed tokens per verify, per source (0 lookup, 1 model), seeded
      // optimistic so each source is tried; the cost ratio from the measured step walls (seeded 2 until both are seen)
      double gain[2] = {4.0, 4.0}, tv = 0, tp = 0; NSUInteger nv = 0, np = 0, declined[2] = {0, 0}, skipped = 0;
      double tg0 = now_s();
      while (hist.count - L < tokens) {
        NSUInteger n = hist.count;
        NSUInteger room = tokens - (hist.count - L);                  // never commit past the requested count
        NSMutableArray<NSNumber *> *draft = [NSMutableArray array];
        // prompt lookup: the longest earlier occurrence of the history's suffix, and what followed it
        for (NSUInteger ng = (plain_only || !lookup) ? 0 : ngram; ng >= min_ngram && ng >= 1 && !draft.count; --ng) {
          if (n < ng + 1) continue;
          for (NSInteger j = (NSInteger)(n - ng) - 1; j >= 0 && !draft.count; --j) {
            BOOL mt = YES;
            for (NSUInteger k = 0; k < ng && mt; ++k) mt = [hist[j + k] isEqual:hist[n - ng + k]];
            if (!mt) continue;
            for (NSUInteger k = j + ng; k < n && draft.count < max_draft; ++k) [draft addObject:hist[k]];
          }
          if (ng == 1) break;
        }
        if (draft.count) ++nlookup;
        double cost = (nv && np) ? (tv / nv) / (tp / np) : 2.0;       // a verify, in plain steps
        if (draft.count && !always && gain[0] < cost && (++declined[0] % 8)) { [draft removeAllObjects]; ++skipped; }
        int src = draft.count ? 0 : 1;
        NSUInteger kk = room > 1 ? MIN(K, room - 1) : 0;
        BOOL use_model = !draft.count && D && !plain_only && kk && n + rows < T.cap;
        if (use_model && !always && gain[1] < cost && (++declined[1] % 8)) { use_model = NO; ++skipped; }
        if (use_model) {
          // THE DRAFT MODEL: catch up to the committed tokens (forced), then kk free steps, all in one command buffer
          double d0 = now_s();
          NSUInteger v = 0;                                          // the first position whose cache token is stale
          while (v < fed.count && v < n - 1 && [fed[v] isEqual:hist[v]]) ++v;
          [fed removeObjectsInRange:NSMakeRange(v, fed.count - v)];
          // positions v .. n-2 are re-fed with the committed tokens, forced through the log; then positions n-1 ..
          // n+kk-2 generate: the gen step at q0 = p reads log[p + 1] (forced when p + 1 < n, free after)
          for (NSUInteger i = v + 1; i < n; ++i) D.logp[i] = [hist[i] intValue];
          for (NSUInteger i = n; i < n + kk + 1 && i < D.cap; ++i) D.logp[i] = -1;
          *D.q0p = (uint32_t)v;
          memcpy(D.rx16, D.embed_rows + (size_t)[hist[v] unsignedIntValue] * D.dmodel, D.dmodel * 2);   // the draft's width
          NSUInteger steps = (n - 1 - v) + kk;                         // catch-up steps, then kk drafting steps
          runList(D, D.g[@"dispatches"], D.pipes, steps); dsteps += steps;
          for (NSUInteger i = v; i < n; ++i) [fed addObject:hist[i]];            // fed v .. n-1
          for (NSUInteger i = 0; i < kk; ++i) [draft addObject:@(D.logp[n + i])];   // d_1 .. d_kk
          for (NSUInteger i = 0; i + 1 < kk; ++i) [fed addObject:draft[i]];       // fed n .. n+kk-2 (the last is not fed)
          ++nmodel;
          t_draft += now_s() - d0;
        }
        while (draft.count && draft.count + 1 > room) [draft removeLastObject];
        if (!draft.count || n + rows >= T.cap) {
          double p0 = now_s();
          runList(T, T.g[@"dispatches"], T.pipes, 1);               // plain: decode's own step
          [hist addObject:@(T.logp[n])]; ++nplain;
          t_plain += now_s() - p0; tp += now_s() - p0; ++np;
          continue;
        }
        // verify: rows = t_{n-1}, d_1 .. d_m, padded with t_{n-1}
        double v0 = now_s();
        for (NSUInteger r = 0; r < rows; ++r) {
          uint32_t tok = r == 0 || r > draft.count ? (uint32_t)[hist[n - 1] unsignedIntValue] : (uint32_t)[draft[r - 1] unsignedIntValue];
          memcpy(x16v + r * dmodel, T.embed_rows + (size_t)tok * dmodel, dmodel * 2);
        }
        runList(T, sp[@"dispatches"], T.vpipes, 1); ++nverify; drafted += draft.count;
        NSUInteger j = 0;                                           // accepted drafts
        for (NSUInteger r = 0; r <= draft.count; ++r) {
          float best = -INFINITY; uint32_t arg = 0;
          if (pairs) {
            const float *pr = pairs + 2 * PG * r; float bi = 0;
            for (NSUInteger gq = 0; gq < PG; ++gq) if (pr[2 * gq] > best || (pr[2 * gq] == best && pr[2 * gq + 1] < bi)) { best = pr[2 * gq]; bi = pr[2 * gq + 1]; }
            arg = (uint32_t)bi - (uint32_t)(r * PV);
          } else {
            const float *row = (const float *)((const uint8_t *)lg + r * row_bytes);
            for (uint32_t w = 0; w < vocab; ++w) if (row[w] > best) { best = row[w]; arg = w; }
          }
          [hist addObject:@(arg)];
          if (r < draft.count && arg == [draft[r] unsignedIntValue]) ++j; else break;
        }
        accepted += j; [acc_hist addObject:@(j)];
        gain[src] = 0.7 * gain[src] + 0.3 * (double)(j + 1);
        NSUInteger last = hist.count - 1;                           // the new state: q0 = last index, log, input row
        for (NSUInteger i = n; i <= last && i < T.cap; ++i) T.logp[i] = [hist[i] intValue];
        *T.q0p = (uint32_t)last;
        memcpy(T.rx16, T.embed_rows + (size_t)[hist[last] unsignedIntValue] * dmodel, dmodel * 2);
        t_verify += now_s() - v0; tv += now_s() - v0; ++nv;
      }
      double tgen = now_s() - tg0;
      NSMutableArray *gen = [NSMutableArray array];
      for (NSUInteger i = L; i < L + tokens; ++i) [gen addObject:hist[i]];
      return [@{@"out_ids": gen, @"generated_tokens": @(tokens), @"prompt_len": @(L), @"ttft_s": @(ttft),
                @"gen_wall_s": @(tgen), @"tokens_per_s": @((double)tokens / tgen), @"plain_steps": @(nplain),
                @"verify_steps": @(nverify), @"drafted": @(drafted), @"accepted": @(accepted),
                @"lookup_rounds": @(nlookup), @"model_rounds": @(nmodel), @"draft_model_steps": @(dsteps), @"k": @(K),
                @"draft_s": @(t_draft), @"verify_s": @(t_verify), @"plain_s": @(t_plain),
                @"accepted_per_verify": acc_hist, @"declined": @(skipped), @"cost_ratio": @((nv && np) ? (tv / nv) / (tp / np) : 0),
                @"mode": plain_only ? @"plain" : (D ? (lookup ? @"lookup+draft" : @"draft") : @"lookup")} mutableCopy];
    };
    generate();                                                     // warm: pipelines, clock
    NSMutableDictionary *rep = generate();
    if (vprof) {
      // --profile-verify F: after the run (its state: a mid-generation cache), one verify step with each dispatch in its
      // own command buffer, its GPU time alone; then the whole step chained, its GPU span. Per-dispatch times do not
      // overlap, so their sum exceeds the span; the split by kind says where the step goes.
      NSArray *vl = sp[@"dispatches"];
      NSMutableArray *rowsj = [NSMutableArray array];
      for (NSUInteger i = 0; i < vl.count; ++i) {
        id<MTLCommandBuffer> cb = g17_gpu_cb(q);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        encodeList(T, enc, @[vl[i]], @[T.vpipes[i]]); [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        [rowsj addObject:@{@"i": @(i), @"name": vl[i][@"name"] ?: @"", @"gpu_us": @((cb.GPUEndTime - cb.GPUStartTime) * 1e6)}];
      }
      id<MTLCommandBuffer> cb = g17_gpu_cb(q);
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      encodeList(T, enc, vl, T.vpipes); [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
      double vspan = cb.GPUEndTime - cb.GPUStartTime;
      cb = g17_gpu_cb(q); enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      encodeList(T, enc, T.g[@"dispatches"], T.pipes); [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
      double pspan = cb.GPUEndTime - cb.GPUStartTime;
      [[NSJSONSerialization dataWithJSONObject:@{@"verify_span_ms": @(vspan * 1e3), @"plain_span_ms": @(pspan * 1e3),
          @"q0": @(*T.q0p), @"dispatches": rowsj} options:NSJSONWritingPrettyPrinted error:nil] writeToFile:@(vprof) atomically:YES];
      printf("verify step %.2f ms GPU (plain step %.2f ms) at q0 %u; per-dispatch to %s\n", vspan * 1e3, pspan * 1e3, *T.q0p, vprof);
    }
    [[NSJSONSerialization dataWithJSONObject:rep options:NSJSONWritingPrettyPrinted error:nil] writeToFile:@(argv[2]) atomically:YES];
    printf("specgen %s: %lu tokens, %.1f tok/s, plain %lu verify %lu (drafted %lu, accepted %lu; draft-model rounds %lu)\n",
           [rep[@"mode"] UTF8String], (unsigned long)tokens, [rep[@"tokens_per_s"] doubleValue],
           [rep[@"plain_steps"] unsignedLongValue], [rep[@"verify_steps"] unsignedLongValue],
           [rep[@"drafted"] unsignedLongValue], [rep[@"accepted"] unsignedLongValue], [rep[@"model_rounds"] unsignedLongValue]);
  }
  return 0;
}
