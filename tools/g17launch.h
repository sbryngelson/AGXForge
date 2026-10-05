#ifndef G17_LAUNCH_H
#define G17_LAUNCH_H
#include <stddef.h>
#include <stdint.h>

// Host-only validation of an explicit cooperative launch. No scheduling or
// instruction semantics are inferred from a successful limit check.
static inline const char *g17ValidateLaunch(const size_t grid[3],const size_t group[3],
    const size_t deviceAxes[3],size_t pipelineThreads,size_t staticBytes,
    size_t staticAlignment,size_t pipelineStaticBytes,size_t deviceMemoryBytes) {
  if (!grid || !group || !deviceAxes || !pipelineThreads) return "launch_limits";
  size_t count=1;
  for(size_t i=0;i<3;++i) {
    if (!grid[i] || !group[i] || !deviceAxes[i] || group[i]>deviceAxes[i] ||
        grid[i]%group[i]) return "launch_group_shape";
    if (count>SIZE_MAX/group[i]) return "launch_group_overflow";
    count*=group[i];
  }
  if(count>pipelineThreads) return "launch_pipeline_limit";
  if(!staticAlignment || (staticAlignment&(staticAlignment-1)) ||
      staticBytes%staticAlignment) return "launch_memory_alignment";
  if(staticBytes!=pipelineStaticBytes) return "launch_static_memory_disagreement";
  if(staticBytes>deviceMemoryBytes) return "launch_device_memory_limit";
  return NULL;
}
// ABI execution requirements apply even when no threadgroup memory is declared.
static inline const char *g17ValidateSIMDLaunch(size_t requiredWidth,size_t actualWidth,
    const size_t group[3],size_t pipelineThreads) {
  if(!requiredWidth || requiredWidth!=actualWidth) return "launch_simd_width";
  if(!group || !pipelineThreads) return "launch_limits";
  size_t count=1;
  for(size_t i=0;i<3;++i) {
    if(!group[i] || count>SIZE_MAX/group[i]) return "launch_group_overflow";
    count*=group[i];
  }
  if(count>pipelineThreads) return "launch_pipeline_limit";
  if(count<requiredWidth || count%requiredWidth) return "launch_partial_simdgroup";
  return NULL;
}
#endif
