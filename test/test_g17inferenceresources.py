import copy
import struct
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from g17modelimport import import_model
from g17inferencelower import compile_graph, POLICY
from g17inferenceprepare import convert_weight, widen_bfloat
from agxforge.g17.inferenceresources import plan
from agxforge.g17.inferencegraph import Unsupported
ROOT=Path(__file__).resolve().parents[1]

class Resources(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph=import_model(ROOT/'evidence/g17-inference-models-v1/minilm',sentence_pooling=True)
        _,cls.lowering=compile_graph(cls.graph,policy=POLICY)
        cls.layout=plan(cls.graph,cls.lowering)

    def test_complete_physical_binding_coverage_and_live_nonoverlap(self):
        r=self.layout['regions']
        self.assertEqual(set(r),{b['name'] for s in self.lowering['stages'] for b in s['bindings']})
        self.assertEqual(len(self.layout['stages']),170)
        for n,a in r.items():
            self.assertEqual(a['offset']%256,0)
            self.assertLessEqual(a['offset']+a['capacity']+256,self.layout['arenas'][a['arena']])
            for m,b in r.items():
                if n>=m or a['arena']!=b['arena']:continue
                live=max(a['birth'],b['birth'])<=min(a['last_use'],b['last_use'])
                spatial=max(a['guard_before'],b['guard_before'])<min(a['guard_after']+256,b['guard_after']+256)
                self.assertFalse(live and spatial,(n,m))
        for n in self.graph['outputs']:
            self.assertEqual(r[n]['last_use'],170)
        self.assertEqual(self.layout['activation_slots'],7)
        self.assertFalse(self.layout['gpu_admitted'])

    def test_immutable_and_extent_corruption_refuse(self):
        for corruption,reason in [('write','immutable'),('size','extent')]:
            bad=copy.deepcopy(self.lowering)
            binding=bad['stages'][0]['bindings'][0]
            if corruption=='write':binding['written']=True
            else:binding['bytes']+=4
            with self.assertRaisesRegex(Unsupported,reason):plan(self.graph,bad)

    def test_read_before_producer_refuses(self):
        bad=copy.deepcopy(self.lowering)
        output=next(b for b in bad['stages'][0]['bindings'] if b['written'])
        output['written']=False
        with self.assertRaisesRegex(Unsupported,'read before producer'):plan(self.graph,bad)

    def test_half_conversion_orientation_and_independent_rounding(self):
        x=np.array([[0.,-0.,1.00048828125,1.00146484375],[-2.,65504.,2**-24,2**-25]],dtype='<f4')
        expected=b''.join(struct.pack('<e',float(x[r,c])) for c in range(4) for r in range(2))
        self.assertEqual(convert_weight(x.tobytes(),x.shape),expected)
        for value in [np.nan,np.inf,70000.]:
            with self.assertRaises((ValueError,FloatingPointError)):
                convert_weight(np.array([[value]],dtype='<f4').tobytes(),(1,1))

    def test_bfloat_widening_and_projection_preserve_bits_and_orientation(self):
        words=[0x0000,0x8000,0x3f80,0x3f81,0xbf00,0x3800]
        raw=struct.pack('<6H',*words)
        self.assertEqual(widen_bfloat(raw).tobytes(),struct.pack('<6I',*[v<<16 for v in words]))
        values=[struct.unpack('<f',struct.pack('<I',v<<16))[0] for v in words]
        expected=b''.join(struct.pack('<e',values[r*3+c]) for c in range(3) for r in range(2))
        self.assertEqual(convert_weight(raw,(2,3),'BF16'),expected)
        for word in (0x7f80,0xff80,0x7fc0,0x4789):
            with self.assertRaises((ValueError,FloatingPointError)):
                convert_weight(struct.pack('<H',word),(1,1),'BF16')
        with self.assertRaisesRegex(ValueError,'dtype'):
            convert_weight(raw,(2,3),'unknown')

if __name__=='__main__':unittest.main()
