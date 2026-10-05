"""Run the native lifetime reset on ordinary CPU memory, never GPU."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeLayerLifetime(unittest.TestCase):
    def test_actual_native_phase_reset_preserves_saved_attention_and_its_guards(self):
        source = r'''
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
struct shmem_result { unsigned unused; };
#include "g17workload_layer.h"
int main(void){
  uint8_t* mapped[30]={0};
  mapped[21]=malloc(0x64000);if(!mapped[21])return 1;
  for(unsigned j=0;j<0x64000;j++)mapped[21][j]=(uint8_t)(j*17+3);
  uint8_t saved[49408];memcpy(saved,mapped[21]+345984,sizeof saved);
  struct workload_layer_region ffn[21]={0};
  ffn[18]=(struct workload_layer_region){21,256,196608,0};
  ffn[19]=(struct workload_layer_region){21,197120,98304,0};
  ffn[20]=(struct workload_layer_region){21,295680,49152,0};
  workload_layer_enter_ffn(mapped,ffn);
  if(memcmp(saved,mapped[21]+345984,sizeof saved))return 2;
  for(unsigned r=18;r<21;r++){
    const struct workload_layer_region* p=&ffn[r];
    for(unsigned j=0;j<p->bytes;j++)if(mapped[21][p->offset+j]!=0xff)return 3;
    for(unsigned j=0;j<128;j++)if(mapped[21][p->offset-128+j]!=0xa5 ||
      mapped[21][p->offset+p->bytes+j]!=0xa5)return 4;
  }
  // The intervening gap and the suffix after the saved-output guard also survive.
  for(unsigned j=344960;j<345984;j++)if(mapped[21][j]!=(uint8_t)(j*17+3))return 5;
  for(unsigned j=395392;j<0x64000;j++)if(mapped[21][j]!=(uint8_t)(j*17+3))return 6;
  free(mapped[21]);return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / 'test.c').write_text(source)
            subprocess.run(['clang', '-I', str(ROOT / 'spike/agxsub'),
                            str(directory / 'test.c'), '-o', str(directory / 'test')], check=True)
            subprocess.run([str(directory / 'test')], check=True)


if __name__ == '__main__':
    unittest.main()
