"""Decoder cache-footprint admission before any native request is issued."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from g17inferencesession import decoder_inputs


def inputs(rows,start,valid):
    return dict(token_ids=np.arange(rows,dtype=np.int32),position_ids=np.arange(start,start+rows,dtype=np.int32),
                attention_mask=(np.arange(256)<valid).astype(np.int32),cache_valid_length=np.array(valid,dtype=np.int32))


class Requests(unittest.TestCase):
    def test_full_and_padded_prefill_then_decode_preserve_actual_length(self):
        for rows,start,valid,size in ((32,0,32,1284),(32,32,37,1284),(1,37,38,1036),(1,255,256,1036)):
            raw,next_valid=decoder_inputs(inputs(rows,start,valid),rows,start)
            self.assertEqual(len(raw),size);self.assertEqual(next_valid,valid)
            self.assertEqual(np.frombuffer(raw,dtype='<i4')[-1],valid)

    def test_padding_is_not_cache_growth(self):
        a=inputs(1,64,65)
        with self.assertRaisesRegex(ValueError,'position'):decoder_inputs(a,1,37)

    def test_decode_before_prefill_and_cache_overflow_refuse(self):
        for rows,start,valid in ((1,0,1),(32,240,250),(1,256,257)):
            with self.assertRaises(ValueError):decoder_inputs(inputs(rows,start,valid),rows,start)

    def test_mask_holes_and_valid_length_mismatch_refuse(self):
        for update in ('hole','extra','length'):
            a=inputs(32,0,17)
            if update=='hole':a['attention_mask'][3]=0
            elif update=='extra':a['attention_mask'][17]=1
            else:a['cache_valid_length']=np.array(33,dtype=np.int32)
            with self.assertRaises(ValueError):decoder_inputs(a,32,0)

    def test_negative_oov_wrong_dtype_shape_and_inventory_refuse(self):
        for update in ('negative','oov','dtype','shape','inventory'):
            a=inputs(32,0,32)
            if update=='negative':a['token_ids'][0]=-1
            elif update=='oov':a['token_ids'][0]=151936
            elif update=='dtype':a['token_ids']=a['token_ids'].astype(np.int64)
            elif update=='shape':a['cache_valid_length']=np.array([32],dtype=np.int32)
            else:a['mystery']=np.array(0)
            with self.assertRaises(ValueError):decoder_inputs(a,32,0)


if __name__=='__main__':unittest.main()
