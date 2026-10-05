"""CPU-only author controls consuming the compiler's declared constant pool."""
import os
import sys
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import g17authorobj as A
import g17cc as C
import g17cooplayernorm as P
import g17gpumd as M
import g17scanlink as S
import g17verify as V


class CooperativeAuthor(unittest.TestCase):
    def fixture(self, rows=32):
        p=C.compile_function(P.cooperative_layernorm_ir(rows))
        abi=dict(p.abi(),instruction_count=len(p.contract().instructions),has_back_edge=False)
        bindings=[(b['index'],b['offset'],b['written'],'float') for b in abi['bindings']]
        return abi['prologue']+p.code,bindings,abi

    def test_both_shapes_deliver_declared_scratchpad(self):
        for rows in (1,32):
            with self.subTest(rows=rows):
                text,bindings,abi=self.fixture(rows)
                sections,_=A.author(text,64,bindings,abi)
                md=sections[A.SECTIONS[0]]
                self.assertEqual(len(md),516)
                self.assertEqual(M.register_count(md),abi['register_count'])
                self.assertEqual(V.verify_metadata(md),[])
                self.assertEqual(M.threadgroup_declaration(md),dict(memory_use=1,static_memory_bytes=128))
                self.assertEqual(S.binding_records(md),[b[:3] for b in bindings])

    def test_unsupported_or_conflicting_declarations_refuse(self):
        text,bindings,abi=self.fixture()
        # WIDENED WHERE A WITNESS REPRODUCES, NOT WHERE IT IS CONVENIENT. instruction_count 300
        # gives slot 32 = 1, which coop4-flat-b carries at 516 bytes and this author now rebuilds
        # byte-exact, so it is no longer a refusal; the one-register set stays refused at this pool
        # length because the witness for it carries the 128-byte pool, and a register set admitted
        # by analogy would move every structure after the slot-29 vector.
        # A BACK EDGE ALONE IS NOT A REFUSAL, and the replacement keeps the control's teeth.
        # coop4-loop-nolt and coop4-loop-tg witness (16, (160,161)) and (16, (160,161,164)) WITH a
        # loop and the author rebuilds both byte-exact. What stays refused is a back edge on a form
        # no witness covers at all: the one-register set (160,), which has a witness only at the
        # 128-byte pool. (160,164) was here until results/g17-coop-x2-v1 witnessed both its forms.
        for patch in ({'constant_pool':None},{'constant_pool':[1]}, {'abi_version':3},
                      {'has_back_edge':True,'system_registers':(160,)},
                      {'pk_values':{15:1,16:1,18:0}}, {'pk_extra':(15,16)},
                      {'system_registers':(160,)},{'register_count':0},
                      {'reproduce_measured_class':True},
                      {'threadgroup':dict(abi['threadgroup'],static_memory_bytes=64)}):
            with self.subTest(patch=patch),self.assertRaises(A.Missing):
                A.author(text,64,bindings,dict(abi,**patch))

    def test_a_witnessed_loop_form_authors(self):
        # The other half of this widening, asserted rather than left implied: the fixture's own
        # (16, (160,161)) with a back edge is the loop witness's form and authors.
        text,bindings,abi=self.fixture()
        sections,_=A.author(text,64,bindings,dict(abi,has_back_edge=True))
        self.assertEqual(len(sections[A.SECTIONS[0]]),524)

    def test_a_witnessed_slot32_value_authors(self):
        # The other half of the widening: a straight-line program short enough to carry slot 32 = 1
        # is the flat-b witness's own shape, so it authors rather than refusing.
        text,bindings,abi=self.fixture()
        sections,_=A.author(text,64,bindings,dict(abi,instruction_count=300))
        self.assertEqual(len(sections[A.SECTIONS[0]]),516)

    def test_actual_entry_and_prologue_must_match(self):
        text,bindings,abi=self.fixture()
        for changed,entry in ((bytes(64)+text[64:],64),(text,128)):
            with self.subTest(entry=entry),self.assertRaisesRegex(ValueError,'differs'):
                A.author(changed,entry,bindings,abi)
