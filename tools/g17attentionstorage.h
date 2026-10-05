#ifndef G17_ATTENTION_STORAGE_H
#define G17_ATTENTION_STORAGE_H
#include "g17scanstorage.h"

enum { G17_ATTN_LEGACY_MAX=32, G17_ATTN_MAX=64, G17_ATTN_GUARD=128 };
enum { G17_ATTN_INPUT, G17_ATTN_PARAMETER, G17_ATTN_INTERMEDIATE, G17_ATTN_OUTPUT };
enum { G17_ATTN_FLOAT32, G17_ATTN_UINT32, G17_ATTN_FLOAT16, G17_ATTN_UINT16 };
enum { G17_ATTN_FINITE_VALUES, G17_ATTN_RAW_BITS };
static inline size_t g17AttentionElementBytes(unsigned type) {
  return (type==G17_ATTN_FLOAT16 || type==G17_ATTN_UINT16)?2:(type<=G17_ATTN_UINT32?4:0);
}
typedef struct {
  size_t count, source, payload[G17_ATTN_MAX], allocation[G17_ATTN_MAX];
  unsigned role[G17_ATTN_MAX], type[G17_ATTN_MAX], value_policy[G17_ATTN_MAX];
  size_t binding_width[G17_ATTN_MAX]; // declared element; type remains scalar transport/guard interpretation
  bool output_from_input, output_slice, output_bytes;
  size_t initialized_output, initialization_offset;
  bool intermediate_from_input[G17_ATTN_MAX];
  size_t intermediate_offset[G17_ATTN_MAX];
} G17AttentionStorage;

static inline bool g17AttentionStorageInitTyped(G17AttentionStorage *s,size_t count,
    const size_t *payload,const unsigned *role,const unsigned *type) {
  if (!s || !payload || !role || !count || count>G17_ATTN_MAX) return false;
  G17AttentionStorage next={.count=count};
  size_t total=0,inputs=0,outputs=0;
  for (size_t i=0;i<count;++i) {
    next.type[i]=type?type[i]:G17_ATTN_FLOAT32;
    size_t width=g17AttentionElementBytes(next.type[i]);
    if (!width || !payload[i] || payload[i]%width || payload[i]>64*1024*1024 || role[i]>G17_ATTN_OUTPUT)
      return false;
    next.binding_width[i]=width;
    next.payload[i]=payload[i];next.allocation[i]=payload[i]+2*G17_ATTN_GUARD;
    next.role[i]=role[i];total+=next.allocation[i];
    if (role[i]==G17_ATTN_INPUT) {++inputs;next.source=i;}
    if (role[i]==G17_ATTN_OUTPUT) ++outputs;
  }
  if (inputs!=1 || outputs!=1 || total>128*1024*1024) return false;
  *s=next;return true;
}

// Declared elements and reply units are separate: an eight-byte integer is one
// scalar binding element transported as two raw words, not a vector declaration.
static inline bool g17AttentionSetBindingWidth(G17AttentionStorage *s,size_t i,size_t width) {
  if (!s || i>=s->count || (width!=2 && width!=4 && width!=8 && width!=16)) return false;
  size_t unit=g17AttentionElementBytes(s->type[i]);
  if (!unit || width<unit || width%unit || s->payload[i]%width) return false;
  if (width!=unit && s->type[i]!=G17_ATTN_FLOAT32 && s->type[i]!=G17_ATTN_UINT32) return false;
  s->binding_width[i]=width;return true;
}

// Raw transport permits every bit pattern, without changing the declared type
// or width. It cannot use a NaN sentinel to detect unwritten output; callers
// still need a complete reference and explicit preservation checks.
static inline bool g17AttentionSetValuePolicy(G17AttentionStorage *s,size_t i,unsigned policy) {
  if (!s || i>=s->count || policy>G17_ATTN_RAW_BITS) return false;
  s->value_policy[i]=policy;return true;
}

static inline bool g17AttentionSetBindingLanes(G17AttentionStorage *s,size_t i,size_t lanes) {
  if (!s || i>=s->count || (lanes!=1 && lanes!=2 && lanes!=4)) return false;
  if (lanes!=1 && s->type[i]!=G17_ATTN_FLOAT32 && s->type[i]!=G17_ATTN_UINT32) return false;
  size_t width=g17AttentionElementBytes(s->type[i])*lanes;
  return g17AttentionSetBindingWidth(s,i,width);
}

static inline bool g17AttentionStorageInit(G17AttentionStorage *s,size_t count,
    const size_t *payload,const unsigned *role) {
  return g17AttentionStorageInitTyped(s,count,payload,role,NULL);
}

// Explicit host initialization for an in-place kernel. The input allocation
// remains separate and read-only. Initialization is not evidence of GPU stores.
static inline bool g17AttentionOutputFromInputSlice(G17AttentionStorage *s,size_t offset) {
  if (!s || !s->count) return false;
  size_t source=s->source, width=s->binding_width[source];
  if (!width || offset%width || offset>s->payload[source]) return false;
  for (size_t i=0;i<s->count;++i) if (s->role[i]==G17_ATTN_OUTPUT) {
    // A byte copy permits uint input transport and float output interpretation.
    if (s->binding_width[i]!=width ||
        s->payload[i]>s->payload[source]-offset) return false;
    s->output_from_input=true;s->output_slice=true;s->output_bytes=false;s->initialized_output=i;
    s->initialization_offset=offset;return true;
  }
  return false;
}

// Explicit mixed-width transport: offset is in bytes, aligned to the output
// element. This copies bits, never converts input elements into output elements.
static inline bool g17AttentionOutputFromInputBytes(G17AttentionStorage *s,size_t offset) {
  if (!s || !s->count) return false;
  size_t source=s->source;
  if (offset>s->payload[source]) return false;
  for (size_t i=0;i<s->count;++i) if (s->role[i]==G17_ATTN_OUTPUT) {
    size_t width=s->binding_width[i];
    if (!width || offset%width || s->payload[i]>s->payload[source]-offset) return false;
    s->output_from_input=true;s->output_slice=false;s->output_bytes=true;
    s->initialized_output=i;s->initialization_offset=offset;return true;
  }
  return false;
}

static inline bool g17AttentionOutputFromInput(G17AttentionStorage *s) {
  if (!s || !s->count) return false;
  for (size_t i=0;i<s->count;++i) if (s->role[i]==G17_ATTN_OUTPUT) {
    if (s->payload[i]!=s->payload[s->source] || s->type[i]!=s->type[s->source]) return false;
    if (!g17AttentionOutputFromInputSlice(s,0)) return false;
    s->output_slice=false;return true;
  }
  return false;
}

static inline bool g17AttentionIntermediateFromInputBytes(G17AttentionStorage *s,size_t target,size_t offset) {
  if (!s || !s->count || target>=s->count || s->role[target]!=G17_ATTN_INTERMEDIATE ||
      s->intermediate_from_input[target]) return false;
  size_t width=s->binding_width[target];
  if (!width || offset%width || offset>s->payload[s->source] ||
      s->payload[target]>s->payload[s->source]-offset) return false;
  s->intermediate_from_input[target]=true;s->intermediate_offset[target]=offset;return true;
}

#ifdef __OBJC__
// Shared by the parser, schedule validator and executor. An offset never has
// an implicit default and cannot silently modify the legacy whole-input mode.
static inline bool g17AttentionConfigureOutput(G17AttentionStorage *s,NSDictionary *record) {
  id mode=record[@"output_initialization"], value=record[@"output_initialization_offset"];
  if (!mode) return !value && !s->output_from_input;
  if (record[@"binding_windows"]) return false;
  if ([mode isEqual:@"copy-input-v1"])
    return !value && g17AttentionOutputFromInput(s);
  if ((![mode isEqual:@"copy-input-slice-v1"] && ![mode isEqual:@"copy-input-bytes-v1"]) ||
      ![value isKindOfClass:NSNumber.class] ||
      CFGetTypeID((__bridge CFTypeRef)value)==CFBooleanGetTypeID() ||
      !strchr("cCsSiIlLqQ",[value objCType][0]) || [value longLongValue]<0 ||
      [value unsignedLongLongValue]>SIZE_MAX) return false;
  return [mode isEqual:@"copy-input-bytes-v1"] ?
    g17AttentionOutputFromInputBytes(s,[value unsignedIntegerValue]) :
    g17AttentionOutputFromInputSlice(s,[value unsignedIntegerValue]);
}

// Graph records name allocations; resolved schedule records name positions.
// Exact schemas and unique targets keep spelling errors from becoming defaults.
static inline bool g17AttentionConfigureIntermediates(G17AttentionStorage *s,NSDictionary *record,NSArray *names) {
  id rows=record[@"intermediate_initialization"];
  if (!rows) return true;
  if (![rows isKindOfClass:NSArray.class] || ![rows count] ||
      record[@"binding_windows"] ||
      (![record[@"write_policy"] isEqual:@"multiple-v1"] &&
       ![record[@"write_policy"] isEqual:@"inplace-v1"])) return false;
  G17AttentionStorage candidate=*s;
  for (id row in rows) {
    if (![row isKindOfClass:NSDictionary.class] || [row count]!=3 ||
        ![row[@"mode"] isEqual:@"copy-input-bytes-v1"]) return false;
    id value=row[@"source_offset"];
    if (![value isKindOfClass:NSNumber.class] ||
        CFGetTypeID((__bridge CFTypeRef)value)==CFBooleanGetTypeID() ||
        !strchr("cCsSiIlLqQ",[value objCType][0]) || [value longLongValue]<0 ||
        [value unsignedLongLongValue]>SIZE_MAX) return false;
    NSUInteger target;
    if (names) {
      id name=row[@"allocation"];
      if (![name isKindOfClass:NSString.class]) return false;
      target=[names indexOfObject:name];
    } else {
      id position=row[@"buffer_position"];
      if (![position isKindOfClass:NSNumber.class] ||
          CFGetTypeID((__bridge CFTypeRef)position)==CFBooleanGetTypeID() ||
          !strchr("cCsSiIlLqQ",[position objCType][0]) || [position longLongValue]<0 ||
          [position unsignedLongLongValue]>=s->count) return false;
      target=[position unsignedIntegerValue];
    }
    if (!g17AttentionIntermediateFromInputBytes(&candidate,target,[value unsignedIntegerValue])) return false;
  }
  *s=candidate;return true;
}

static inline NSArray *g17AttentionIntermediateRecords(const G17AttentionStorage *s) {
  NSMutableArray *rows=[NSMutableArray array];
  for (size_t i=0;i<s->count;++i) if (s->intermediate_from_input[i])
    [rows addObject:@{@"buffer_position":@(i),@"mode":@"copy-input-bytes-v1",@"source_offset":@(s->intermediate_offset[i])}];
  return rows;
}

static inline bool g17AttentionIntermediatesMatch(const G17AttentionStorage *s,NSDictionary *record,NSArray *names) {
  G17AttentionStorage candidate=*s;
  memset(candidate.intermediate_from_input,0,sizeof(candidate.intermediate_from_input));
  memset(candidate.intermediate_offset,0,sizeof(candidate.intermediate_offset));
  if (!g17AttentionConfigureIntermediates(&candidate,record,names)) return false;
  for (size_t i=0;i<s->count;++i)
    if (candidate.intermediate_from_input[i]!=s->intermediate_from_input[i] ||
        candidate.intermediate_offset[i]!=s->intermediate_offset[i]) return false;
  return true;
}

static inline bool g17AttentionOutputMatches(const G17AttentionStorage *s,NSDictionary *record) {
  G17AttentionStorage candidate=*s;
  if (!g17AttentionConfigureOutput(&candidate,record)) return false;
  return candidate.output_from_input==s->output_from_input && candidate.output_slice==s->output_slice &&
         candidate.output_bytes==s->output_bytes && candidate.initialization_offset==s->initialization_offset;
}
#endif

static inline void g17AttentionInitializeOutput(const G17AttentionStorage *s,
    void *const *buffers,const void *source) {
  if (s->output_from_input)
    memcpy((uint8_t *)buffers[s->initialized_output]+G17_ATTN_GUARD,
           (const uint8_t *)source+s->initialization_offset,s->payload[s->initialized_output]);
}

static inline bool g17AttentionValuesValid(const G17AttentionStorage *s,size_t i,
    const void *data) {
  if (!s || i>=s->count || !data) return false;
  if (s->value_policy[i]==G17_ATTN_RAW_BITS) return true;
  if ((s->type[i]==G17_ATTN_UINT32 || s->type[i]==G17_ATTN_UINT16)) return true;
  size_t width=g17AttentionElementBytes(s->type[i]);
  const G17Storage format={.half=s->type[i]==G17_ATTN_FLOAT16};
  return width && g17StorageFinite(&format,data,s->payload[i]/width);
}

static inline const char *g17AttentionInitializeIntermediates(const G17AttentionStorage *s,
    void *const *buffers,const void *source) {
  for (size_t i=0;i<s->count;++i) if (s->intermediate_from_input[i]) {
    void *target=(uint8_t *)buffers[i]+G17_ATTN_GUARD;
    memcpy(target,(const uint8_t *)source+s->intermediate_offset[i],s->payload[i]);
    if (!g17AttentionValuesValid(s,i,target)) return "nonfinite_initial_intermediate";
  }
  return NULL;
}

static inline void g17AttentionReset(const G17AttentionStorage *s,size_t i,void *buffer) {
  uint8_t *p=buffer;const uint32_t sentinel=0x7fc01234;
  const uint16_t halfSentinel=0x7e12;
  size_t width=g17AttentionElementBytes(s->type[i]);
  memset(p,(int)(0xa5^i),s->allocation[i]);
  for (size_t j=0;j<s->payload[i];j+=width)
    memcpy(p+G17_ATTN_GUARD+j,width==2?(const void *)&halfSentinel:(const void *)&sentinel,width);
}

static inline const char *g17AttentionPrepare(const G17AttentionStorage *s,
    void *const *buffers,const void *const *snapshots) {
  for (size_t i=0;i<s->count;++i) {
    if (!buffers[i]) return "missing_buffer";
    for (size_t j=0;j<i;++j) if (buffers[i]==buffers[j]) return "aliased_allocation";
    if (s->role[i]<=G17_ATTN_PARAMETER && (!snapshots[i] ||
        !g17AttentionValuesValid(s,i,snapshots[i]))) return "nonfinite_input";
  }
  for (size_t i=0;i<s->count;++i) {
    g17AttentionReset(s,i,buffers[i]);
    if (s->role[i]<=G17_ATTN_PARAMETER)
      memcpy((uint8_t *)buffers[i]+G17_ATTN_GUARD,snapshots[i],s->payload[i]);
  }
  g17AttentionInitializeOutput(s,buffers,snapshots[s->source]);
  if (s->output_from_input && !g17AttentionValuesValid(s,s->initialized_output,
      (uint8_t *)buffers[s->initialized_output]+G17_ATTN_GUARD)) return "nonfinite_initial_output";
  return g17AttentionInitializeIntermediates(s,buffers,snapshots[s->source]);
}

// Call only after the previous query's completion and checks. Parameters are
// neither uploaded nor changed here. Under the default finite policy, NaNs
// make unwritten intermediate/output elements fail completion checks. Raw-bit
// allocations require the caller to establish writes with its source reference.
static inline const char *g17AttentionBegin(const G17AttentionStorage *s,
    void *const *buffers,const void *source) {
  if (!g17AttentionValuesValid(s,s->source,source)) return "nonfinite_input";
  for (size_t i=0;i<s->count;++i) if (!buffers[i]) return "missing_buffer";
  for (size_t i=0;i<s->count;++i) {
    if (s->role[i]==G17_ATTN_INPUT)
      memcpy((uint8_t *)buffers[i]+G17_ATTN_GUARD,source,s->payload[i]);
    else if (s->role[i]>=G17_ATTN_INTERMEDIATE) g17AttentionReset(s,i,buffers[i]);
  }
  g17AttentionInitializeOutput(s,buffers,source);
  if (s->output_from_input && !g17AttentionValuesValid(s,s->initialized_output,
      (uint8_t *)buffers[s->initialized_output]+G17_ATTN_GUARD)) return "nonfinite_initial_output";
  return g17AttentionInitializeIntermediates(s,buffers,source);
}

// 'produced' is set from completed stage bindings, never inferred from finite
// bytes. It permits bounded prefix validation without accepting missing stores
// in any stage that has actually run. Application arithmetic is checked apart.
static inline const char *g17AttentionCheck(const G17AttentionStorage *s,
    const void *const *buffers,const void *const *snapshots,const bool *produced,
    size_t *failed) {
  for (size_t i=0;i<s->count;++i) {
    if (failed) *failed=i;
    const uint8_t *p=buffers[i];if (!p) return "missing_buffer";
    for (size_t j=0;j<G17_ATTN_GUARD;++j)
      if (p[j]!=(uint8_t)(0xa5^i) || p[G17_ATTN_GUARD+s->payload[i]+j]!=(uint8_t)(0xa5^i))
        return "boundary_guard";
    if (s->role[i]<=G17_ATTN_PARAMETER && (!snapshots[i] ||
        memcmp(p+G17_ATTN_GUARD,snapshots[i],s->payload[i]))) return "readonly_input_changed";
    // A source-byte initialized intermediate remains an exact copy until a
    // completed stage writes it. The executor supplies this query's owned
    // source snapshot, including after Begin replaces the previous input.
    if (s->intermediate_from_input[i] && !produced[i] &&
        (!snapshots[s->source] || memcmp(p+G17_ATTN_GUARD,
          (const uint8_t *)snapshots[s->source]+s->intermediate_offset[i],s->payload[i])))
      return "readonly_initialized_intermediate_changed";
    // Every uint16/uint32 bit pattern is legal, including the reset sentinel. Integer
    // completeness and untouched slots require the independent output checker;
    // command completion plus these memory checks do not establish arithmetic.
    if (produced[i] && !g17AttentionValuesValid(s,i,p+G17_ATTN_GUARD))
      return "nonfinite_or_unwritten_output";
  }
  return NULL;
}
#endif
