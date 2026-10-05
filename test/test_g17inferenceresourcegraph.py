import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from g17inferenceresourcegraph import recover

class Graph(unittest.TestCase):
    def test_only_measured_classes_are_served(self):
        for size in (8,96,256,512):
            with self.assertRaisesRegex(ValueError,'measured'):recover(size)

    def test_exact_128m_graph_and_low_binding_carrier(self):
        request,pages,r=recover(128)
        self.assertEqual(r['requests_sha256'],'ee69f6606bb4779aaa29800c63eb0948c44075059ec0adc4de3af2edff29eec4')
        self.assertEqual(r['pages_sha256'],'8e52f1ff70271dd4137e87f5222ac411f671c33488348c5b5bcfceebc263f334')
        self.assertEqual(r['rows'][20]['bytes'],128<<20)
        self.assertEqual(r['binding_metadata']['coordinate'],0x2c)
        self.assertEqual(r['binding_metadata']['destination_allocation'],17)
        self.assertEqual(r['shader_record']['coordinate'],0x2048)
        self.assertFalse(r['gpu_admitted'])

    def test_1g_uses_measured_low_shader_record_not_wrapped_coordinate(self):
        _,_,r=recover(1024)
        self.assertEqual(r['rows'][20]['bytes'],1<<30)
        self.assertEqual(r['shader_record']['destination_allocation'],17)
        self.assertEqual(r['shader_record']['destination_offset'],0x10000)
        self.assertEqual(r['shader_record']['coordinate'],0x1a)
        self.assertNotEqual(r['shader_record']['coordinate'],0x48)
        b=r['binding_metadata'];s=r['shader_record']
        self.assertLessEqual(b['destination_offset']+b['bytes'],s['destination_offset'])
        self.assertLessEqual(s['destination_offset']+s['bytes'],r['rows'][17]['bytes'])

if __name__=='__main__':unittest.main()
