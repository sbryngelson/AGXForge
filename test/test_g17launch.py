"""Native launch-limit admission without creating a Metal device or queue."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=r'''
#include <assert.h>
#include <string.h>
#include "g17launch.h"
int main(void) {
  size_t grid[]={32,32,1},group[]={32,1,1},axes[]={1024,1024,64};
  assert(!g17ValidateLaunch(grid,group,axes,1024,128,4,128,32768));
  assert(!g17ValidateSIMDLaunch(32,32,group,1024));
  assert(!strcmp(g17ValidateSIMDLaunch(32,16,group,1024),"launch_simd_width"));
  assert(!strcmp(g17ValidateSIMDLaunch(32,32,group,16),"launch_pipeline_limit"));
  group[0]=16;assert(!strcmp(g17ValidateSIMDLaunch(32,32,group,1024),"launch_partial_simdgroup"));
  group[0]=48;assert(g17ValidateSIMDLaunch(32,32,group,1024));
  group[0]=64;assert(!g17ValidateSIMDLaunch(32,32,group,1024));group[0]=32;
  grid[1]=1;assert(!g17ValidateLaunch(grid,group,axes,1024,128,4,128,32768));
  grid[0]=31;assert(!strcmp(g17ValidateLaunch(grid,group,axes,1024,128,4,128,32768),"launch_group_shape"));
  grid[0]=32;
  assert(!strcmp(g17ValidateLaunch(grid,group,axes,16,128,4,128,32768),"launch_pipeline_limit"));
  assert(!strcmp(g17ValidateLaunch(grid,group,axes,1024,128,4,0,32768),"launch_static_memory_disagreement"));
  assert(!strcmp(g17ValidateLaunch(grid,group,axes,1024,128,4,128,64),"launch_device_memory_limit"));
  assert(!strcmp(g17ValidateLaunch(grid,group,axes,1024,128,3,128,32768),"launch_memory_alignment"));
  group[0]=0;assert(g17ValidateLaunch(grid,group,axes,1024,128,4,128,32768));
  group[0]=grid[0]=SIZE_MAX;group[1]=grid[1]=2;axes[0]=axes[1]=SIZE_MAX;
  assert(!strcmp(g17ValidateLaunch(grid,group,axes,SIZE_MAX,128,4,128,32768),"launch_group_overflow"));
  return 0;
}
'''


class Launch(unittest.TestCase):
    def test_full_groups_limits_memory_and_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'launch.c';binary=Path(tmp)/'launch';source.write_text(SOURCE)
            subprocess.run(['clang','-std=c11','-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-I',str(ROOT/'tools'),str(source),'-o',str(binary)],
                capture_output=True,check=True,timeout=30)
            subprocess.run([str(binary)],capture_output=True,check=True,timeout=10)
