import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import g17bufferbatches as B
import g17buffergraph as G
import g17tensorprojection as P


class ResidentBatches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.req=P.requirements(P.programs(1536),1536)
        cls.graph=P.graph(cls.req,1536)
        cls.plan=B.partition(cls.graph)

    def test_actual_expansion_is_three_batches_and_old_graph_still_refuses(self):
        self.assertEqual([len(b) for b in self.plan['batches']],[128,128,34])
        self.assertEqual(B.validate(self.plan,self.req),[])
        self.assertEqual(B.flatten(self.plan),self.graph)
        self.assertEqual(G.validate(self.graph,self.req),['runtime allocation/stage limit exceeded'])

    def test_cross_batch_dependency_mutations_refuse(self):
        def duplicate(p):p['batches'][1][0]['bindings'][-1]['offset']=p['batches'][0][-1]['bindings'][-1]['offset']
        def missing(p):p['batches'][0].pop()
        def early(p):p['batches'][0][0],p['batches'][-1][-1]=p['batches'][-1][-1],p['batches'][0][0]
        def guard(p):p['batches'][1][0]['bindings'][-1]['offset']=124
        def extent(p):p['batches'][1][0]['bindings'][-1]['length']=2048
        for mutate,message in [(duplicate,'overlaps'),(missing,'no earlier stage produced'),
                               (early,'no earlier stage produced'),(guard,'leaves the payload'),(extent,'extent differs')]:
            p=copy.deepcopy(self.plan);mutate(p)
            with self.subTest(mutation=mutate.__name__):
                self.assertTrue(any(message in f for f in B.validate(p,self.req)))

    def test_batch_limits_and_conflicting_representations_refuse(self):
        for batches in ([],[[]],self.plan['batches']*3,[self.graph['stages']]):
            p=copy.deepcopy(self.plan);p['batches']=batches
            self.assertTrue(B.validate(p,self.req))
        p=copy.deepcopy(self.plan);p['stages']=self.graph['stages']
        self.assertIn('unambiguous',B.validate(p,self.req)[0])
        for size in (0,129,True):
            with self.assertRaises(ValueError):B.partition(self.graph,size)


if __name__=='__main__':unittest.main()
