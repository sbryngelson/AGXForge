"""Mixed FP16 request/FP32 reply framing over a real CPU subprocess, no Metal."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from g17native import Worker

FAKE = r'''
import json,struct,sys
mode=sys.argv[1]
identity=dict(storage_dtype='float16',reply_dtype='float32',matrix_bytes=4096,
 request_bytes=4096,reply_bytes=4096,reply_elements=1024,completion_markers=False)
if mode=='missing':del identity['reply_dtype']
if mode=='wrong':identity['reply_dtype']='uint32'
def send(seq,payload=b'',**kw):
 h=json.dumps(dict(protocol=2,sequence=seq,bytes=len(payload),identity=identity,**kw)).encode()
 sys.stdout.buffer.write(struct.pack('<I',len(h))+h+payload);sys.stdout.buffer.flush()
send(0,rows=32,columns=64)
seq=0
while True:
 raw=sys.stdin.buffer.read(4)
 if not raw:break
 size=struct.unpack('<I',raw)[0]
 if size!=4096:raise ValueError('request is not 2048 FP16 values')
 values=struct.unpack('<2048e',sys.stdin.buffer.read(size));seq+=1
 result=struct.pack('<1024f',*[65536+x for x in values[:1024]])
 if mode=='nonfinite':result=struct.pack('<1024I',*([0x7f800001,0xff800000,0x80000000,0x7fc12345]*256))
 send(seq,result,status=0,boundary_guard=True)
'''


def worker(mode='normal',reply_value_policy='finite-v1'):
    return Worker([sys.executable, '-u', '-c', FAKE, mode], rows=32, columns=64,
                  storage_dtype='float16', reply_dtype='float32', request_elements=2048,
                  reply_elements=1024, completion_markers=False,reply_value_policy=reply_value_policy)


class MixedTransport(unittest.TestCase):
    def test_raw_reply_policy_preserves_bits_while_default_refuses_same_frame(self):
        x=np.zeros(2048,dtype=np.float16)
        w=worker('nonfinite')
        try:
            with self.assertRaisesRegex(RuntimeError,'nonfinite'):
                w.query(x)
            self.assertIsNotNone(w.last_raw_reply)
        finally:w.close()
        w=worker('nonfinite','raw-bits-v1')
        try:
            got=w.query(x)
            self.assertEqual(got.dtype,np.dtype('float32'))
            self.assertEqual(got.view('<u4').tolist(),[0x7f800001,0xff800000,0x80000000,0x7fc12345]*256)
        finally:w.close()

    def test_invalid_raw_policy_configuration_refuses_before_spawn(self):
        with patch('g17native.subprocess.Popen') as spawn:
            for policy in (None,False,{},'raw'):
                with self.assertRaises(ValueError):
                    Worker(['unused'],rows=1,columns=1,reply_value_policy=policy)
            with self.assertRaisesRegex(ValueError,'protocol 2'):
                Worker(['unused'],rows=1,columns=1,reply_value_policy='raw-bits-v1')
            spawn.assert_not_called()

    def test_tensor_sized_half_request_has_owned_float_reply(self):
        w = worker()
        try:
            x = np.linspace(-100, 100, 2048, dtype=np.float16)
            first = w.query(x)
            self.assertEqual(first.dtype, np.float32)
            self.assertEqual(first.shape, (1024,))
            np.testing.assert_array_equal(first, x[:1024].astype(np.float32) + np.float32(65536))
            saved = first.copy()
            np.testing.assert_array_equal(w.query(-x), -x[:1024].astype(np.float32) + np.float32(65536))
            np.testing.assert_array_equal(first, saved)
            np.testing.assert_array_equal(w.query(x), saved)
        finally:
            w.close()

    def test_equal_byte_count_cannot_replace_a_dtype_declaration(self):
        for mode in ('missing', 'wrong'):
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, 'explicit result layout'):
                worker(mode)

    def test_invalid_reply_type_and_mixed_markers_refuse_before_spawn(self):
        with patch('g17native.subprocess.Popen') as spawn:
            with self.assertRaises(ValueError):
                Worker(['unused'], rows=32, columns=64, reply_dtype='float64')
            with self.assertRaises(ValueError):
                Worker(['unused'], rows=32, columns=64, storage_dtype='float32',
                       reply_dtype='float16', completion_markers=True)
            spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
