"""Captured JSON embeddings cannot bypass their FP32 identity check."""
import hashlib
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from g17retrievalcheck import captured_embedding


class Capture(unittest.TestCase):
    def setUp(self):
        self.values=np.arange(384,dtype='<f4')/1024
        self.values[0]=-0.
        self.request=dict(output_sha256=hashlib.sha256(self.values.tobytes()).hexdigest())

    def test_json_roundtrip_retains_signed_zero_and_all_fp32_values(self):
        import json
        values=json.loads(json.dumps(self.values.tolist()))
        self.assertEqual(captured_embedding(self.request,values).tobytes(),self.values.tobytes())

    def test_one_bit_difference_or_zero_sign_is_not_a_tolerance_pass(self):
        for index in (0,1,383):
            changed=self.values.copy();changed.view(np.uint32)[index]^=1 if index else 0x80000000
            with self.assertRaisesRegex(ValueError,'hash'):captured_embedding(self.request,changed.tolist())

    def test_shape_or_nonfinite_refuses_before_numerics(self):
        for changed in (self.values[:-1],self.values.reshape(2,192),np.full(384,np.nan),np.full(384,np.inf)):
            with self.assertRaisesRegex(ValueError,'shape/finiteness'):captured_embedding(self.request,changed.tolist())


if __name__=='__main__':unittest.main()
