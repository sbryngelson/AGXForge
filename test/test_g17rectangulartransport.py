"""Actual process framing for rectangular buffers; no Metal or GPU calls."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from g17native import Worker, StorageLayout, BufferTransportLayout

ECHO = r'''
import json,struct,sys
identity=dict(storage_dtype='float32',matrix_bytes=6144,request_bytes=6144,
 reply_bytes=1536,reply_elements=384,completion_markers=False)
def send(seq,payload=b'',**kw):
 h=json.dumps(dict(protocol=2,sequence=seq,bytes=len(payload),identity=identity,**kw)).encode()
 sys.stdout.buffer.write(struct.pack('<I',len(h))+h+payload);sys.stdout.buffer.flush()
send(0,rows=1,columns=1536)
seq=0
while True:
 n=sys.stdin.buffer.read(4)
 if not n:break
 data=sys.stdin.buffer.read(struct.unpack('<I',n)[0]);seq+=1
 send(seq,data[-1536:],status=0,boundary_guard=True,readonly_inputs=True)
'''


class RectangularTransport(unittest.TestCase):
    def test_full_input_and_shorter_output_use_distinct_extents(self):
        worker=Worker([sys.executable,'-u','-c',ECHO],rows=1,columns=1536,
            request_elements=1536,reply_elements=384,completion_markers=False)
        try:
            x=np.arange(1536,dtype=np.float32)
            np.testing.assert_array_equal(worker.query(x),x[-384:])
            np.testing.assert_array_equal(worker.query(-x),-x[-384:])
            np.testing.assert_array_equal(worker.query(x),x[-384:])
        finally:worker.close()

    def test_scan_limit_and_total_transport_bound_remain(self):
        with self.assertRaises(ValueError):StorageLayout(1,1536)
        self.assertEqual(BufferTransportLayout(32,1536).matrix_bytes,32*1536*4)
        self.assertEqual(BufferTransportLayout(500000,384,'float16').matrix_bytes,384000000)
        for rows,columns in [(0,1536),(True,1536),(32,0),(500000,385),(1,500000*384+1)]:
            with self.assertRaises(ValueError):BufferTransportLayout(rows,columns)

    def test_invalid_transport_is_rejected_before_process_start(self):
        with patch('g17native.subprocess.Popen') as spawn:
            with self.assertRaises(ValueError):
                Worker(['unused'],rows=500000,columns=385,reply_elements=1)
        spawn.assert_not_called()


if __name__=='__main__':unittest.main()
