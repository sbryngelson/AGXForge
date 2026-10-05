from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import g17inferencekernels as K
import g17packedcheck as D
from agxforge.g17 import cc


class GatherKernel(unittest.TestCase):
    def test_real_embedding_body_has_indexed_reads_and_store(self):
        fn=K.gather_rows_ir(rows=32,width=384,vocabulary=30522)
        p=cc.compile_function(fn)
        ops=[op for _,_,op,_ in D.decode(p.code)]
        self.assertEqual(ops.count(12682),2)
        self.assertEqual(ops.count(17229),1)
        self.assertEqual(ops.count(684),1)
        abi=p.abi_plain(p.abi())
        self.assertEqual([b['index'] for b in abi['bindings']],[1,2,3])
        self.assertEqual([b['written'] for b in abi['bindings']],[False,False,True])
        self.assertEqual(abi['system_registers'],[160,161])
        self.assertTrue(abi['launch']['exact_grid_required'])
        self.assertLessEqual(abi['register_count'],126)
        kinds=[op.kind for block in fn.blocks for op in block.ops]
        self.assertEqual(kinds.count('load'),2)
        self.assertEqual(kinds.count('store_at'),1)
        self.assertEqual(kinds.count('mul'),2)

    def test_fp32_body_preserved(self):
        import hashlib
        p=cc.compile_function(K.gather_rows_ir(rows=32,width=384,vocabulary=30522))
        self.assertEqual(hashlib.sha256(p.code).hexdigest(),'78d3a66dc69df19217ebb899d7c5c47d0c6f4ddb8a2e3795ed229799b45e7107')

    def test_bf16_all_bit_patterns_through_decoded_bytes(self):
        import numpy as np
        import g17emu as E
        # The emulator's existing grid is 1-D. Each invocation checks a whole
        # row and an independently varying ID; no 2-D mapping is invented.
        width, rows=896,74
        words=(np.arange(rows*width,dtype=np.uint32)&65535).astype('<u2')
        table=words.view(np.uint8)
        p=cc.compile_function(K.gather_rows_ir(rows=1,width=width,vocabulary=rows,dtype='BF16'))
        for row in range(rows):
            buffers={1:np.array([row],dtype='<u4').view(np.uint8),2:table,
                     3:np.full(width*4,0xa5,np.uint8)}
            machine=E.Machine(p.code,buffers,width,32,views=True).run()
            expected=words[row*width:(row+1)*width].astype(np.uint32)<<16
            self.assertTrue(np.array_equal(buffers[3].view('<u4'),expected))
            self.assertEqual(machine.admitted,{})
        ops=[op for _,_,op,_ in D.decode(p.code)]
        self.assertIn(424,ops)
        self.assertIn(14392,ops)
        self.assertIn(17014,ops)
        self.assertNotIn(3290,ops)
        self.assertEqual(ops.count(17229),1)

    def test_unmeasured_dtype_and_bad_address_extent_refuse(self):
        with self.assertRaisesRegex(ValueError,'FP32/BF16'):K.gather_rows_ir(rows=32,width=896,vocabulary=151936,dtype='F16')
        with self.assertRaisesRegex(ValueError,'32-bit'):K.gather_rows_ir(rows=32,width=384,vocabulary=2**32)
        with self.assertRaises(ValueError):K.gather_rows_ir(rows=32,width=383,vocabulary=30522)


if __name__=='__main__':unittest.main()
