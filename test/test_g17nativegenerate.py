"""CPU generation control must chunk complete prompts and advance resident KV."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from g17nativegenerate import generate


class Tokenizer:
    def decode(self,ids,skip_special_tokens=True):return ' '.join(map(str,ids))


class Generation(unittest.TestCase):
    def test_full_prompt_chunks_then_incremental_tokens_and_eos(self):
        calls=[];selected=iter((5,1,3,2))
        def request(session,ids,position,valid,reset=False):
            calls.append((ids,position,valid,reset))
            logits=np.zeros((len(ids),6),np.float32);logits[:,next(selected)]=1
            return logits,dict(submissions=748 if len(ids)==32 else 916,client_roundtrip_ns=1000)
        encoded=dict(token_ids=list(range(37)),tokenizer={'revision':'test'},tokenizer_engine={'test':'1'})
        with patch('g17nativegenerate.prepare',return_value=encoded),patch('g17nativegenerate.model_request',side_effect=request):
            report=generate(None,None,Tokenizer(),'prompt',8,{2})
        self.assertEqual(calls[0],(list(range(32)),0,32,True))
        self.assertEqual(calls[1],(list(range(32,37))+[0]*27,32,37,False))
        self.assertEqual(calls[2],([1],37,38,False));self.assertEqual(calls[3],([3],38,39,False))
        self.assertEqual(report['generated_token_ids'],[1,3,2]);self.assertEqual(report['stop_reason'],'eos')
        self.assertEqual(report['submissions_per_decode_token'],[916,916])

    def test_one_generated_token_does_not_request_unneeded_decode(self):
        encoded=dict(token_ids=list(range(32)),tokenizer={},tokenizer_engine={})
        logits=np.zeros((32,6),np.float32);logits[:,4]=1
        with patch('g17nativegenerate.prepare',return_value=encoded),patch('g17nativegenerate.model_request',return_value=(logits,dict(submissions=748,client_roundtrip_ns=1))) as request:
            report=generate(None,None,Tokenizer(),'prompt',1,{2})
        self.assertEqual(request.call_count,1);self.assertEqual(report['generated_token_ids'],[4]);self.assertEqual(report['stop_reason'],'max_new_tokens')

    def test_invalid_bound_refuses_before_model_request(self):
        for bound in (0,129,True):
            with self.assertRaisesRegex(ValueError,'bound'):generate(None,None,Tokenizer(),'prompt',bound,{2})


if __name__=='__main__':unittest.main()
