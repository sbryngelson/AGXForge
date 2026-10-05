"""CPU subprocess controls for opt-in bounded graph trace headers."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from g17native import Worker

FAKE=r'''
import sys,os,json,struct
mode=sys.argv[1]
identity=dict(storage_dtype='float32',reply_dtype='float32',matrix_bytes=8,
 request_bytes=8,reply_bytes=8,reply_elements=2,completion_markers=False,
 header_limit=4096 if mode=='wrong_limit' else (1048576 if mode.startswith('batch') else 65536),
 reply_selection='all-stage-outputs' if mode=='wrong_selection' else 'final-output-only',
 trace_description='x'*(70000 if mode.startswith('batch') else 5000))
def emit(sequence,payload=b''):
 h=dict(protocol=2,sequence=sequence,bytes=len(payload),rows=1,columns=2,
 identity=identity,status=0,boundary_guard=not(mode=='badguard' and sequence>0),readonly_inputs=True)
 raw=json.dumps(h).encode();frame=struct.pack('<I',len(raw))+raw+payload
 for start in range(0,len(frame),127):os.write(1,frame[start:start+127])
emit(0)
sequence=0
while True:
 n=sys.stdin.buffer.read(4)
 if not n:break
 raw=sys.stdin.buffer.read(struct.unpack('<I',n)[0]);sequence+=1
 if mode=='oversized':os.write(1,struct.pack('<I',65537));break
 if mode=='batch_oversized':os.write(1,struct.pack('<I',1048577));break
 if mode=='nonfinite':emit(sequence,struct.pack('<II',0x7fc12345,0x3f800000));continue
 values=struct.unpack('<ff',raw);emit(sequence,struct.pack('<ff',*(v*2 for v in values)))
'''


class NativeHeaders(unittest.TestCase):
    def create(self,mode='normal',**kw):
        return Worker([sys.executable,'-u','-c',FAKE,mode],rows=1,columns=2,
                      request_elements=2,reply_elements=2,completion_markers=False,timeout=3,**kw)

    def test_large_fragmented_headers_require_explicit_matching_contract(self):
        with self.assertRaisesRegex(RuntimeError,'header size'):self.create()
        worker=self.create(header_limit=65536)
        try:
            for values in (np.array([1,2],np.float32),np.array([-3,4],np.float32)):
                np.testing.assert_array_equal(worker.query(values),values*2)
            self.assertEqual(worker.sequence,2)
        finally:worker.close()

    def test_peer_cannot_select_its_own_limit_or_reply_policy(self):
        for mode in ('wrong_limit','wrong_selection'):
            with self.subTest(mode=mode),self.assertRaisesRegex(RuntimeError,'layout differs'):
                self.create(mode,header_limit=65536)

    def test_batch_trace_contract_is_explicit_and_remains_bounded(self):
        with self.assertRaisesRegex(RuntimeError,'header size'):self.create('batch',header_limit=65536)
        worker=self.create('batch',header_limit=1048576)
        try:np.testing.assert_array_equal(worker.query(np.ones(2,np.float32)),np.full(2,2,np.float32))
        finally:worker.close()
        worker=self.create('batch_oversized',header_limit=1048576)
        with self.assertRaisesRegex(RuntimeError,'header size'):worker.query(np.ones(2,np.float32))
        self.assertTrue(worker.closed)

    def test_oversize_reply_fails_closed_without_retry(self):
        worker=self.create('oversized',header_limit=65536)
        with self.assertRaisesRegex(RuntimeError,'header size'):worker.query(np.ones(2,np.float32))
        self.assertTrue(worker.closed)
        with self.assertRaisesRegex(RuntimeError,'closed or failed'):worker.query(np.ones(2,np.float32))

    def test_rejected_complete_replies_preserve_raw_evidence(self):
        import struct
        for mode in ('nonfinite','badguard'):
            worker=self.create(mode,header_limit=65536)
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):worker.query(np.ones(2,np.float32))
            self.assertTrue(worker.closed)
            header,payload=worker.last_raw_reply
            self.assertEqual(header['sequence'],1);self.assertEqual(len(payload),8)
            if mode=='nonfinite':self.assertEqual(struct.unpack('<II',payload),(0x7fc12345,0x3f800000))
            else:self.assertIs(header['boundary_guard'],False)

    def test_invalid_limits_are_refused_before_process_creation(self):
        with patch('g17native.subprocess.Popen',side_effect=AssertionError('must not start')):
            for limit in (True,0,4096.0,65535,65537):
                with self.subTest(limit=limit),self.assertRaises(ValueError):self.create(header_limit=limit)
            with self.assertRaisesRegex(ValueError,'protocol 2'):
                Worker(['unused'],rows=1,columns=2,header_limit=65536)


if __name__=='__main__':unittest.main()
