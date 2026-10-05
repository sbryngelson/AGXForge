import copy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import g17abi
import g17cc
import g17halfscan
import g17ir as ir
import g17packedcheck


class CompilerABI(unittest.TestCase):
    def test_public_ir_imports_without_tools_and_shares_compatibility_classes(self):
        script = '''
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
assert str(root / 'tools') not in sys.path
from agxforge.g17 import ir
assert Path(ir.__file__).resolve() == root / 'agxforge/g17/ir.py'
sys.path.insert(0, str(root / 'tools'))
import g17ir
for name in ('Function', 'Builder', 'Buffer', 'Imm', '_fbits', '_expand_erf'):
    assert getattr(ir, name) is getattr(g17ir, name), name
'''
        result = subprocess.run([sys.executable, '-I', '-c', script, str(ROOT)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_system_registers_are_captured_before_the_first_abi_query(self):
        program = self.compile()
        reads = [m for _, _, m in program.layout if m.form == "read_sr.4"]
        self.assertEqual({m.fields["sr"] for m in reads}, {160})
        for m in reads:
            m.fields["sr"] = 164
        self.assertEqual(program.abi()["system_registers"], (160,))
        self.assertEqual(program.abi(profile="fresh_cache_key")["system_registers"], (160,))

    def test_constructor_accepts_the_compiler_owners_ranks_keyword(self):
        p = self.compile()
        q = g17cc.G17Program(p.name, p.code, p.layout, p.inherited,
                            buffers=p.buffers, written=p.written, ranks={1: 0, 2: 1})
        self.assertEqual(q.abi(), p.abi())
        with self.assertRaisesRegex(g17cc.Unsupported, "disagree"):
            g17cc.G17Program(p.name, p.code, p.layout, p.inherited,
                            buffers=p.buffers, written=p.written,
                            binding_ranks={1: 0, 2: 1}, ranks={1: 1, 2: 0})

    def test_unknown_opcode_raises_in_both_abi_views(self):
        with patch("g17formops.load", return_value={}):
            program = self.compile()
        for make in (program.abi, program.contract):
            with self.assertRaisesRegex(KeyError, "no opcode recorded"):
                make()

    def compile(self, rows=1, columns=1):
        return g17cc.compile_function(g17halfscan.scan_ir(rows, columns))

    def test_compiling_another_program_cannot_change_the_first_abi(self):
        program = self.compile()
        before, contract = program.abi(), program.contract()
        output = ir.Buffer("output", 2, elem=ir.F32)
        f = ir.Function("other", [output])
        b = ir.Builder(f, f.block("entry"))
        b.store_at(output, b.builtin("thread_position_in_grid"), b.const(7))
        b.ret()
        g17cc.compile_function(f)
        self.assertEqual(program.abi(), before)
        self.assertEqual(program.contract(), contract)
        self.assertEqual([b.offset for b in contract.bindings], [0, 2])

    def test_source_and_returned_container_mutations_cannot_change_abi(self):
        program = self.compile()
        original, contract = program.abi(), program.contract()
        program.buffers[0].elem = ir.I32
        program.written.clear()
        program.layout[0][2].fields["sr"] = 0
        returned = program.abi()
        self.assertIs(returned, program.abi())
        self.assertEqual(returned["abi_version"], 3)
        self.assertEqual(returned["system_registers"], (160,))
        with self.assertRaises(TypeError):
            returned["bindings"][0]["offset"] = 99
        with self.assertRaises(AttributeError):
            returned["pk_values"].clear()
        plain = program.abi_plain(returned)
        plain["bindings"][0]["offset"] = 99
        program.abi_inputs()["pk_values"].clear()
        self.assertEqual(program.abi(), original)
        # A new cache key must also use captured facts, not the edited source.
        self.assertEqual(program.abi(profile="after_mutation")["bindings"], original["bindings"])
        self.assertEqual(program.abi(profile="after_mutation")["forms"], original["forms"])
        self.assertEqual(program.contract(), contract)
        with self.assertRaises(FrozenInstanceError):
            contract.bindings[0].offset = 99
        program.code += b"\0\0"
        with self.assertRaisesRegex(g17cc.Unsupported, "stale"):
            program.contract()
        with self.assertRaisesRegex(g17cc.Unsupported, "code changed"):
            program.abi()

    def test_instruction_declarations_match_independent_vendor_decoder(self):
        for rows, columns in ((1, 1), (3, 7), (33, 384)):
            p = self.compile(rows, columns)
            abi = p.contract()
            decoded = g17packedcheck.decode(p.code)
            self.assertEqual([(i.offset, i.length, i.opcode) for i in abi.instructions],
                             [(o, n, op) for o, n, op, _ in decoded])
            self.assertEqual(abi.forms, p.abi()["forms"])
            abi.check_code(p.code)
            if rows == 33:
                self.assertEqual(abi.code_sha256, "6c2f696341e9f0b25316408417ec8450efcf58a9af7734404d046224fee7e31f")

    def test_json_round_trip_and_malformed_contracts(self):
        abi = self.compile().contract()
        self.assertEqual(g17abi.ProgramABI.from_dict(json.loads(json.dumps(abi.to_dict()))), abi)
        for mutation in ("version", "missing", "extra", "boolean", "duplicate", "offset",
                         "element", "instruction", "truncated", "prologue", "writes"):
            bad = copy.deepcopy(abi.to_dict())
            if mutation == "version": bad["version"] = 2
            elif mutation == "missing": del bad["arch_flag"]
            elif mutation == "extra": bad["assume_defaults"] = True
            elif mutation == "boolean": bad["arch_flag"] = 1
            elif mutation == "duplicate": bad["bindings"][1]["index"] = 1
            elif mutation == "offset": bad["bindings"][1]["offset"] = 0
            elif mutation == "element": bad["bindings"][1]["element_bytes"] = 4
            elif mutation == "instruction": bad["instructions"][1]["offset"] += 2
            elif mutation == "truncated": bad["instructions"].pop()
            elif mutation == "prologue": bad["prologue"] = "00"
            elif mutation == "writes": bad["writes_buffer"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                g17abi.ProgramABI.from_dict(bad)

    def test_cold_compile_and_abi_permit_no_external_process(self):
        script = '''
import sys,subprocess
sys.path.insert(0,'tools')
def denied(*a,**k): raise AssertionError('build attempted an external process: '+str(a[:1]))
subprocess.Popen=denied
import g17cc,g17halfscan
p=g17cc.compile_function(g17halfscan.scan_ir(33,384))
p.abi()
a=p.contract()
assert a.code_sha256=='6c2f696341e9f0b25316408417ec8450efcf58a9af7734404d046224fee7e31f'
print('zero external processes, 3078 compiler-declared instructions')
'''
        p = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("zero external processes", p.stdout)


class ThreadgroupBlock(unittest.TestCase):
    """ABI v4: a program that uses threadgroup memory declares its requirement; the typed contract
    carries the block OPTIONAL at the schema and REQUIRED at the point of use, so retained v3
    contracts load and a threadgroup use without a declaration is refused."""
    def _v3(self):
        import g17cc, g17layernorm
        return g17cc.compile_function(g17layernorm.layernorm_ir(1, 4))

    def test_retained_v3_contracts_load_and_carry_no_block(self):
        p = self._v3()
        self.assertEqual(p.abi()["abi_version"], 3)
        self.assertNotIn("threadgroup", p.abi())
        c = p.contract()
        self.assertIsNone(c.threadgroup)
        from g17abi import ProgramABI
        self.assertEqual(ProgramABI.from_dict(c.to_dict()), c)

    def test_a_threadgroup_program_is_v4_with_the_agreed_block(self):
        import g17cc, g17cooplayernorm as C
        p = g17cc.compile_function(C.cooperative_layernorm_ir(1))
        a = p.abi()
        self.assertEqual(a["abi_version"], 4)
        self.assertEqual(dict(a["threadgroup"]), dict(required_size=(32, 1, 1), static_memory_bytes=128,
                                                      static_memory_alignment=4, dynamic_memory=()))
        c = p.contract()
        self.assertEqual(c.threadgroup.static_memory_bytes, 128)
        from g17abi import ProgramABI
        self.assertEqual(ProgramABI.from_dict(c.to_dict()), c)

    def test_use_without_declaration_is_refused_at_the_point_of_use(self):
        import g17cc, g17cooplayernorm as C
        from g17abi import ProgramABI
        d = g17cc.compile_function(C.cooperative_layernorm_ir(1)).contract().to_dict()
        d["threadgroup"] = None
        with self.assertRaisesRegex(Exception, "uses_threadgroup without a threadgroup declaration"):
            ProgramABI.from_dict(d)
        e = self._v3().contract().to_dict()
        e["threadgroup"] = dict(required_size=[32, 1, 1], static_memory_bytes=128, static_memory_alignment=4, dynamic_memory=[])
        with self.assertRaisesRegex(Exception, "does not use threadgroup memory"):
            ProgramABI.from_dict(e)

    def test_undeclared_threadgroup_use_is_refused_at_compile(self):
        import g17cc, g17ir as ir
        f = ir.Function("u", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
        b = ir.Builder(f, f.block("entry"))
        t = b.builtin("thread_position_in_grid", name="t")
        b.store_tg(b.add(t, ir.Imm(1)), t); b.barrier()
        b.store_at(f.buffers[1], t, b.add(b.load_tg(t), ir.Imm(0)))      # through an ALU: the store's own load-wait refusal must not fire first
        b.ret()
        with self.assertRaisesRegex(g17cc.Unsupported, "declares no scratchpad"):
            g17cc.compile_function(f)


class ConstantPool(unittest.TestCase):
    """ABI v4's second key: the program's slot-13 constant bytes, stated from emitted code. Empty
    for the cooperative program (every literal is movimm); absent on v3; refused in both
    mismatched directions."""
    def test_v4_states_an_empty_pool_and_round_trips(self):
        import g17cc, g17cooplayernorm as C
        from g17abi import ProgramABI
        p = g17cc.compile_function(C.cooperative_layernorm_ir(1))
        self.assertEqual(p.abi()["abi_version"], 4)
        self.assertEqual(p.abi()["constant_pool"], ())
        c = p.contract()
        self.assertEqual(c.constant_pool, ())
        self.assertEqual(ProgramABI.from_dict(c.to_dict()), c)
        self.assertEqual(c.to_dict()["constant_pool"], [])

    def test_v3_carries_no_pool_key(self):
        import g17cc, g17layernorm
        p = g17cc.compile_function(g17layernorm.layernorm_ir(1, 4))
        self.assertEqual(p.abi()["abi_version"], 3)
        self.assertNotIn("constant_pool", p.abi())
        self.assertIsNone(p.contract().constant_pool)

    def test_mismatched_directions_are_refused(self):
        import g17cc, g17cooplayernorm as C, g17layernorm
        from g17abi import ProgramABI
        d = g17cc.compile_function(C.cooperative_layernorm_ir(1)).contract().to_dict()
        d["constant_pool"] = None
        with self.assertRaisesRegex(Exception, "v4 contract without its constant_pool"):
            ProgramABI.from_dict(d)
        e = g17cc.compile_function(g17layernorm.layernorm_ir(1, 4)).contract().to_dict()
        e["constant_pool"] = []
        with self.assertRaisesRegex(Exception, "constant_pool on a v3 contract"):
            ProgramABI.from_dict(e)
        f = dict(d); f["constant_pool"] = [256]
        with self.assertRaises(Exception):
            ProgramABI.from_dict(f)

    def test_the_pool_is_captured_at_construction_not_read_off_the_mutable_layout(self):
        """A classification-changing mutation: register the literal form as constant-reading AFTER
        the program exists. Its ABI and contract for unchanged bytes must not move; constructing a
        program from the same layout under that classification must refuse."""
        import g17cc, g17cooplayernorm as C
        p = g17cc.compile_function(C.cooperative_layernorm_ir(1))
        before_abi, before_contract = p.abi(), p.contract()
        saved = g17cc.CONSTANT_READING_FORMS
        g17cc.CONSTANT_READING_FORMS = frozenset(["movimm.8"])
        try:
            self.assertEqual(p.abi()["constant_pool"], ())
            self.assertEqual(p.contract(), before_contract)
            self.assertEqual(dict(p.abi()), dict(before_abi))
            with self.assertRaisesRegex(g17cc.Unsupported, "reads constant memory through \\['movimm.8'\\]"):
                g17cc.G17Program(p.name, p.code, p.layout, inherited=p.inherited, threadgroup=p._abi_threadgroup)
        finally:
            g17cc.CONSTANT_READING_FORMS = saved

    def test_a_constant_reading_form_would_be_refused_by_name(self):
        import g17cc
        class M:  # a layout entry whose form is registered as constant-reading
            form = "load.constant.14"
        g17cc.CONSTANT_READING_FORMS = frozenset(["load.constant.14"])
        try:
            with self.assertRaisesRegex(g17cc.Unsupported, "reads constant memory"):
                g17cc._constant_pool([(0, None, M())])
            self.assertEqual(g17cc._constant_pool([]), ())
        finally:
            g17cc.CONSTANT_READING_FORMS = frozenset()


class ExecutionRequirement(unittest.TestCase):
    """ABI v5's key: the execution requirement, present exactly when the program executes tensor
    forms (handoff 9e; the linker's amendments b846283a: simd_width and tensor only, both facts of
    the bytes; symmetric refusal so the field cannot become decorative)."""
    def test_the_tensor_program_is_v5_and_round_trips(self):
        import g17cc, g17tensordelivery as T
        from g17abi import ProgramABI
        p = g17cc.compile_function(T.program())
        a = p.abi()
        self.assertEqual(a["abi_version"], 5)
        self.assertEqual(dict(a["execution"]), {"simd_width": 32, "tensor": True})
        self.assertNotIn("threadgroup", a)
        c = p.contract()
        self.assertEqual((c.execution.simd_width, c.execution.tensor), (32, True))
        self.assertEqual(ProgramABI.from_dict(c.to_dict()), c)
        self.assertEqual(c.to_dict()["execution"], {"simd_width": 32, "tensor": True})

    def test_the_compiler_and_the_schema_name_the_same_tensor_opcodes(self):
        import g17cc
        from g17abi import TENSOR_OPCODES
        self.assertEqual(g17cc.TENSOR_OPCODES, TENSOR_OPCODES)

    def test_v5_states_its_constant_pool(self):
        """Integration 614b7635: the tensor metadata class distinguishes empty from non-empty pools,
        so a v5 contract must state its pool explicitly - an empty tuple for every accepted tensor
        program - and a v5 contract without it is refused."""
        import g17cc, g17tensordelivery as T
        from g17abi import ProgramABI
        p = g17cc.compile_function(T.program())
        self.assertEqual(p.abi()["constant_pool"], ())
        c = p.contract()
        self.assertEqual(c.constant_pool, ())
        self.assertEqual(c.to_dict()["constant_pool"], [])
        d = c.to_dict(); d["constant_pool"] = None
        with self.assertRaisesRegex(Exception, "v5 contract without its constant_pool"):
            ProgramABI.from_dict(d)

    def test_a_pooled_tensor_stream_is_refused_not_declared_empty(self):
        """The retained a-rows-18 witness reads its multiplier from slot 13 through op10828 at +74
        and op10829 at +704 (results/g17-tensor-stride-prediction-v1, the linker's pool-word
        measurement). Its rows, as the compiler's own tensor.inherited instructions, must refuse
        the pool predicate by name rather than yield an empty declaration."""
        import json, os, g17cc
        from g17cc import MInst
        rows = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "g17-tensor-stride-prediction-v1", "a-rows-18.instructions.json")))
        layout = [(r["offset"], r["length"], MInst("tensor.inherited", r["length"], dict(opcode=r["opcode"], bytes=b""))) for r in rows]
        self.assertTrue(any(r["opcode"] in (10828, 10829) for r in rows))
        with self.assertRaisesRegex(g17cc.Unsupported, "constant pool .slot 13. through op10828/op10829"):
            g17cc._constant_pool(layout)
        # and the baseline's rows state the empty pool from the same classification
        base = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "g17-tensor-common-witness-v1", "tensor-common.instructions.json")))
        layout = [(o, l, MInst("tensor.inherited", l, dict(opcode=op, bytes=b""))) for o, l, op, f in base]
        self.assertEqual(g17cc._constant_pool(layout), ())

    def test_v3_and_v4_carry_no_execution_key(self):
        import g17cc, g17layernorm, g17cooplayernorm as C
        for fn, version in ((g17layernorm.layernorm_ir(1, 4), 3), (C.cooperative_layernorm_ir(1), 4)):
            p = g17cc.compile_function(fn)
            self.assertEqual(p.abi()["abi_version"], version)
            self.assertNotIn("execution", p.abi())
            self.assertIsNone(p.contract().execution)

    def test_both_mismatched_directions_are_refused(self):
        import g17cc, g17layernorm, g17tensordelivery as T
        from g17abi import ProgramABI
        d = g17cc.compile_function(T.program()).contract().to_dict()
        d["execution"] = None
        with self.assertRaisesRegex(Exception, "tensor forms without a stated execution requirement"):
            ProgramABI.from_dict(d)
        e = g17cc.compile_function(g17layernorm.layernorm_ir(1, 4)).contract().to_dict()
        e["execution"] = {"simd_width": 32, "tensor": True}
        with self.assertRaisesRegex(Exception, "execution requirement on a program with no tensor forms"):
            ProgramABI.from_dict(e)
        for bad in ({"simd_width": 64, "tensor": True}, {"simd_width": 32, "tensor": False},
                    {"simd_width": 32, "tensor": True, "simdgroups": 1}):
            f = dict(d); f["execution"] = bad
            with self.assertRaises(Exception):
                ProgramABI.from_dict(f)



class TextureResources(unittest.TestCase):
    """ABI v6 (handoff 10s): a program that reads a texture states its resource layout; every
    contract without one is unchanged; the block cannot be omitted, invented, or filled beyond
    what the compiler knows."""

    def _tex(self, *args):
        import g17cc, g17texrun
        return g17cc.compile_function(g17texrun.kernel_ir(*args))

    def test_the_three_frontier_programs_construct_a_v6_contract(self):
        import g17abi
        for args in ((5, 1), (2, 3), ()):
            p = self._tex(*args); c = p.contract(); a = p.abi()
            self.assertEqual(a["abi_version"], 6); self.assertEqual(a["spill_bytes"], 0); self.assertEqual(tuple(a["constant_pool"]), ())
            self.assertEqual([(r.rank, r.apple_index) for r in c.resources.internal], [(0, 44), (1, 48)])
            self.assertEqual([(t.dense_index, t.rank, t.element) for t in c.resources.textures], [(0, None, "uint32")])
            self.assertEqual(c.resources.samplers, ()); self.assertEqual(c.resources.spill_bytes, 0); self.assertEqual(c.resources.spill_basis, "no_spill_form")
            self.assertEqual([(b.index, b.offset, b.written) for b in c.bindings], [(0, 4, True), (1, 6, False)])
            self.assertEqual({(x.record, x.kind, x.written, x.read, x.uniform) for x in c.resources.access}, {(44, "internal", False, None, None), (48, "internal", False, None, None), (0, "user", True, False, True), (1, "user", False, False, True)})
            self.assertEqual(c.resources.not_stated, ("slot27_contents", "slot2_resource_record", "slot2_kind9_record")); self.assertEqual([(x.form, x.target_constant, x.operand_code) for x in c.resources.coordinate_publications], [("publish.coord.x", 0, 4), ("publish.coord.y", 2, 4)])
            self.assertTrue({592, 15813} <= {i.opcode for i in c.instructions}); self.assertEqual(tuple(a["system_registers"]), (160,))
            # the JSON round trip is exact and the sampler list is carried, empty
            d = c.to_dict(); self.assertEqual(d["resources"]["samplers"], []); self.assertEqual(g17abi.ProgramABI.from_dict(d).to_dict(), d)

    def test_executed_contracts_are_unchanged(self):
        import g17tensorprojection as TP, g17cc
        for fn in (TP.pack_ir(384), TP.finish_ir(384, 384)):
            c = g17cc.compile_function(fn).contract()
            self.assertIsNone(c.resources); self.assertIsNone(c.constant_pool); self.assertNotIn("resources", c.to_dict()), "serialised as before the block existed"
            self.assertEqual([b.offset for b in c.bindings], [2 * i for i in range(len(c.bindings))])

    def _plain(self):
        return self._tex().contract().to_dict()

    def test_texture_forms_without_the_block_refuse(self):
        import g17abi
        d = self._plain(); d["resources"] = None; d["constant_pool"] = None
        with self.assertRaisesRegex(Exception, "texture forms without a stated resource layout"): g17abi.ProgramABI.from_dict(d)

    def test_the_block_on_a_buffer_program_refuses(self):
        import g17abi, g17tensorprojection as TP, g17cc
        d = g17cc.compile_function(TP.pack_ir(384)).contract().to_dict(); d["resources"] = self._plain()["resources"]; d["constant_pool"] = []
        with self.assertRaisesRegex(Exception, "no texture forms"): g17abi.ProgramABI.from_dict(d)

    def test_a_named_sampler_refuses_by_type(self):
        import g17abi
        d = self._plain(); d["resources"]["samplers"] = [{"index": 0}]
        with self.assertRaises(Exception): g17abi.ProgramABI.from_dict(d)

    def test_a_texture_with_a_rank_refuses(self):
        import g17abi
        d = self._plain(); d["resources"]["textures"][0]["rank"] = 2
        with self.assertRaises(Exception): g17abi.ProgramABI.from_dict(d)

    def test_user_offsets_must_follow_the_internal_records(self):
        import g17abi
        d = self._plain(); d["bindings"][0]["offset"] = 0; d["bindings"][1]["offset"] = 2
        with self.assertRaisesRegex(Exception, "after the 2 internal records"): g17abi.ProgramABI.from_dict(d)
        d = self._plain(); d["resources"]["internal"] = d["resources"]["internal"][:1]; d["resources"]["access"] = [a for a in d["resources"]["access"] if a["record"] != 48]
        with self.assertRaisesRegex(Exception, "after the 1 internal records"): g17abi.ProgramABI.from_dict(d)

    def test_an_internal_read_fact_cannot_be_stated(self):
        import g17abi
        d = self._plain()
        for a in d["resources"]["access"]:
            if a["record"] == 48: a["read"] = True
        with self.assertRaisesRegex(Exception, "not a compiler fact"): g17abi.ProgramABI.from_dict(d)

    def test_the_unstated_facts_are_named_and_closed(self):
        import g17abi
        d = self._plain(); d["resources"]["not_stated"] = ["slot27_contents", "slot13_pool"]
        with self.assertRaises(Exception): g17abi.ProgramABI.from_dict(d)

    def test_the_executable_shape_and_the_fetch_consumer_control(self):
        """The shape that runs drops the never-referenced buffer (Apple's M0 eliminates it): one user
        binding at rank 2 / offset 4 after the two internals. The control differs from the delivered
        program in exactly its store: sub-form 01, op17235, source naming the fetch, a stated companion."""
        import g17cc, g17texrun, g17asm
        for args in ((5, 1), (2, 3), ()):
            e = g17cc.compile_function(g17texrun.executable_ir(*args)); c = e.contract()
            self.assertEqual([(b.index, b.offset, b.written) for b in c.bindings], [(0, 4, True)])
            self.assertEqual({(x.record, x.kind) for x in c.resources.access}, {(44, "internal"), (48, "internal"), (0, "user")})
            self.assertEqual([(r.rank, r.apple_index) for r in c.resources.internal], [(0, 44), (1, 48)]); self.assertEqual(len(c.resources.coordinate_publications), 2)
            for consumer, n, nbytes in (("op17235", 1, 2), ("op17235_n2", 2, 1)):
                k = g17cc.compile_function(g17texrun.executable_ir(*args, consumer=consumer)); kc = k.contract()
                self.assertIn((17235, 14), kc.forms); self.assertNotIn((17244, 14), kc.forms); self.assertIn((17244, 14), c.forms); self.assertNotIn((17235, 14), c.forms)
                st = [(o, r, m) for o, r, m in k.layout if m.form == "store.14"][0]; d = g17asm.decode_store(st[1])
                self.assertEqual((d["subform"], d["n"], d["slot"], d["const"]), (1, n, 400, 8)); self.assertEqual(st[2].fields["opcode"], 17235)
                self.assertEqual([(o, m.form) for o, r, m in e.layout if m.form != "store.14"], [(o, m.form) for o, r, m in k.layout if m.form != "store.14"], "the pair differs only in its store")
                self.assertEqual(sum(1 for a, b in zip(e.code, k.code) if a != b), nbytes, consumer)
        # a general value cannot go through the fetch consumer
        import g17ir as ir
        f = ir.Function("bad", [ir.Buffer("U", 0)]); b = ir.Builder(f, f.block("entry")); t = b.builtin("thread_position_in_grid", name="t")
        b.store_fetch(f.buffers[0], ir.Imm(400), b.add(t, ir.Imm(1)), b.const(0)); b.ret()
        with self.assertRaisesRegex(g17cc.Unsupported, "not a texture fetch"): g17cc.compile_function(f)


    def test_a_texture_kernel_with_a_device_load_refuses_in_select_and_the_read_predicate_uses_the_load_base_unit(self):
        """Integration's 6b6c9dda: the first cut tested `2 * rank in loaded`, the descriptor OFFSET unit,
        where a load's base is 4 * rank. Both facts stay pinned; ONE of them was lifted.

        The combination no longer refuses outright: root measured the load's base inside a texture
        kernel with four Apple-compiled sources, so select admits ONE user buffer with indexed word
        loads and refuses the rest by name (docs/archive/g17-coordinate-load-admission.md). The program below
        binds TWO, so it still refuses - and the refusal's text is checked on the phrase that
        survives. What was deleted is the guard that said "a read fact here is a defect, not a fact":
        it existed only to agree with the blanket refusal, and a `read` is now a fact. The unit
        distinction this case is named for is unchanged and still asserted both ways."""
        import g17cc, g17ir as ir, g17texrun
        f = ir.Function("texload", [ir.Buffer("U", 0), ir.Buffer("F", 1)]); b = ir.Builder(f, f.block("entry"))
        t = b.builtin("thread_position_in_grid", name="t"); x = getattr(b, "and")(t, ir.Imm(7), name="x")
        v = b.texture_read(x, ir.Imm(3) if False else b.add(getattr(b, "and")(t, ir.Imm(0)), ir.Imm(3), name="y"), tex=0, name="v")
        w = b.load(f.buffers[1], ir.Imm(0), name="w"); b.store(f.buffers[0], ir.Imm(400), b.add(v, w)); b.ret()
        with self.assertRaisesRegex(g17cc.Unsupported, "device load in a texture kernel"): g17cc.compile_function(f)
        # the predicate itself, on a synthetic layout: a load at base 4 * rank marks the binding read; at 2 * rank it does not
        class M:
            def __init__(self, form, **fields): self.form, self.fields = form, fields
        ranks = {44: 0, 48: 1, 0: 2, 1: 3}; binds = ((0, 4, True, "uint", 4), (1, 6, False, "uint", 4))
        base = [(0, b"", M("publish.coord.x")), (1, b"", M("publish.coord.y")), (2, b"", M("texture.read.32", tex=0))]
        lay = g17cc._resource_layout(base + [(1, b"", M("load.14", base=2 * ranks[1]))], ranks, binds)
        self.assertEqual([a["read"] for a in lay["access"] if a["kind"] == "user"], [False, False], "the offset unit is not the load unit")
        # AND AT THE LOAD UNIT IT MARKS THE BINDING READ, which is the half that was lifted: this
        # used to raise "a texture program with a device load reached the resource layout".
        lay = g17cc._resource_layout(base + [(1, b"", M("load.14", base=4 * ranks[1]))], ranks, binds)
        self.assertEqual([a["read"] for a in lay["access"] if a["kind"] == "user"], [False, True],
                         "a load at 4 * rank marks its binding read")
        # and the load makes that record DIVERGENT - the predicate used to read a field the load
        # forms do not carry, and reported a lane-addressed coordinate read as uniform
        self.assertEqual([a["uniform"] for a in lay["access"] if a["kind"] == "user"], [True, False])


class RetainedABIEvolution(unittest.TestCase):
    def test_bundle_evolution_preserves_native_bytes_and_every_other_fact(self):
        import hashlib
        # Import through the compatibility entry first so direct test execution
        # and discovery use the same repository package.
        g17abi.with_instruction_count({}, instructions=[{'offset': 0, 'length': 4}], code=bytes(4))
        from agxforge.g17.abi import verify_rebuild
        enc = lambda value: json.dumps(value).encode()
        abi = {'bindings': [{'index': 1}], 'semantic_flag': True}
        manifest = {'format': 'g17-attention-images-v1', 'programs': {'p': {
            'abi': abi, 'instructions': [{'offset': 0, 'length': 4}],
            'sha256': {'program.bin': hashlib.sha256(bytes(4)).hexdigest()}}}}
        old = {'manifest.json': enc(manifest), 'abi.json': enc(abi),
               'programs/p/program.bin': bytes(4), 'reference.bin': b'reference'}
        updated = copy.deepcopy(manifest)
        updated['programs']['p']['abi']['main_instruction_count'] = 1
        new = dict(old, **{'manifest.json': enc(updated),
                          'abi.json': enc(dict(abi, main_instruction_count=1))})
        verify_rebuild(old, new)
        for path, data in [('programs/p/program.bin', b'xxxx'),
                           ('reference.bin', b'wrong'),
                           ('abi.json', enc(dict(abi, main_instruction_count=2))),
                           ('abi.json', enc(dict(abi, main_instruction_count=1, invented=0)))]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                verify_rebuild(old, dict(new, **{path: data}))
        broken = copy.deepcopy(manifest)
        broken['programs']['p']['instructions'][0]['length'] = 2
        with self.assertRaisesRegex(ValueError, 'cover the exact code'):
            verify_rebuild(dict(old, **{'manifest.json': enc(broken)}), new)

    def test_complete_boundaries_supply_only_the_missing_count(self):
        original = {"bindings": [{"index": 1}], "unknown_future_field": 7}
        before = copy.deepcopy(original)
        result = g17abi.with_instruction_count(original, instructions=[
            {"offset": 0, "length": 4}, {"offset": 4, "length": 2}], code=bytes(6))
        self.assertEqual(result, dict(before, main_instruction_count=2))
        result["bindings"][0]["index"] = 9
        self.assertEqual(original, before)

    def test_incomplete_or_conflicting_evidence_refuses(self):
        for instructions in ([], [{"offset": 2, "length": 4}],
                             [{"offset": 0, "length": 2}],
                             [{"offset": 0, "length": 3}]):
            with self.subTest(instructions=instructions), self.assertRaises(ValueError):
                g17abi.with_instruction_count({}, instructions=instructions, code=bytes(4))
        for count in (0, 2, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                g17abi.with_instruction_count({"main_instruction_count": count},
                    instructions=[{"offset": 0, "length": 4}], code=bytes(4))


class RecordedAuthoringOptionsAreAnAdditionNotALicence(unittest.TestCase):
    """A retained bundle predates root's authoring_options field; a rebuild records it.

    Demanding equality would refuse every historical bundle forever, and accepting whatever the
    rebuild says would accept anything. The addition is allowed only when the rebuild's manifest
    and its requirements say the SAME thing - the invariant that makes "which record carries the
    complete inputs" answerable - and every other difference is refused exactly as before.

    This is the failure the frozen 7987d58c gate found, reproduced in miniature:
    `rebuilt document differs from explicit ABI evolution: manifest.json`, where the only differing
    fields were `/programs/<name>/authoring_options` present in the rebuild and absent from the
    retained document.
    """

    OPTIONS = {'unswept_two_buffer': True}

    def bundle(self, *, retained_options=None, rebuilt_options=None,
               requirements_options="same", extra_manifest=None):
        import hashlib
        enc = lambda value: json.dumps(value).encode()
        code = bytes(4)
        abi = {'bindings': [{'index': 1}]}
        instructions = [{'offset': 0, 'length': 4}]

        def manifest_for(options, extra):
            record = {'abi': copy.deepcopy(abi), 'instructions': instructions,
                      'sha256': {'program.bin': hashlib.sha256(code).hexdigest()}}
            if options is not None:
                record['authoring_options'] = copy.deepcopy(options)
            if extra is not None:
                record.update(extra)
            return {'format': 'g17-attention-images-v1', 'programs': {'p': record}}

        def requirements_for(options):
            entry = {'abi': copy.deepcopy(abi)}
            if options is not None:
                entry['authoring_options'] = copy.deepcopy(options)
            return {'p': entry}

        retained = {'manifest.json': enc(manifest_for(retained_options, None)),
                    'requirements.json': enc(requirements_for(retained_options)),
                    'programs/p/program.bin': code, 'reference.bin': b'reference'}
        evolved = copy.deepcopy(abi)
        evolved['main_instruction_count'] = 1
        rebuilt_manifest = manifest_for(rebuilt_options, extra_manifest)
        rebuilt_manifest['programs']['p']['abi'] = evolved
        mirror = (rebuilt_options if requirements_options == "same" else requirements_options)
        rebuilt_requirements = requirements_for(mirror)
        rebuilt_requirements['p']['abi'] = copy.deepcopy(evolved)
        rebuilt = dict(retained, **{'manifest.json': enc(rebuilt_manifest),
                                    'requirements.json': enc(rebuilt_requirements)})
        return retained, rebuilt

    def test_a_rebuild_may_add_options_a_retained_bundle_predates(self):
        from agxforge.g17.abi import verify_rebuild
        retained, rebuilt = self.bundle(retained_options=None, rebuilt_options=self.OPTIONS)
        verify_rebuild(retained, rebuilt)

    def test_the_two_documents_must_record_the_same_options(self):
        from agxforge.g17.abi import verify_rebuild
        for requirements_options in ({'unswept_two_buffer': False}, None, {}):
            with self.subTest(requirements=requirements_options):
                retained, rebuilt = self.bundle(rebuilt_options=self.OPTIONS,
                                                requirements_options=requirements_options)
                with self.assertRaisesRegex(ValueError, 'different authoring options'):
                    verify_rebuild(retained, rebuilt)

    def test_options_that_record_nothing_are_refused(self):
        from agxforge.g17.abi import verify_rebuild
        # PAIRED, so the case cannot pass for the old reason: before the allowance existed every
        # addition was refused, and each of these would have "passed" while proving nothing.
        verify_rebuild(*self.bundle(rebuilt_options=self.OPTIONS))
        for empty in ({}, [], 'unswept_two_buffer', 0):
            with self.subTest(options=empty):
                retained, rebuilt = self.bundle(rebuilt_options=empty)
                with self.assertRaises(ValueError):
                    verify_rebuild(retained, rebuilt)

    def test_a_bundle_that_already_names_its_options_is_compared_exactly(self):
        """Evolution is for what a document could not have said, not for changing what it did."""
        from agxforge.g17.abi import verify_rebuild
        retained, rebuilt = self.bundle(retained_options=self.OPTIONS,
                                        rebuilt_options=self.OPTIONS)
        verify_rebuild(retained, rebuilt)
        retained, rebuilt = self.bundle(retained_options=self.OPTIONS,
                                        rebuilt_options={'unswept_two_buffer': False})
        with self.assertRaisesRegex(ValueError, 'explicit ABI evolution'):
            verify_rebuild(retained, rebuilt)

    def test_an_unrelated_manifest_change_is_still_refused(self):
        """THE DISCRIMINATING CONTROL: the allowance is one key, not a relaxed comparison.

        Paired with the accepted bundle, so the refusal is attributable to the unrelated field
        rather than to additions being refused wholesale, which is what the old contract did.
        """
        from agxforge.g17.abi import verify_rebuild
        verify_rebuild(*self.bundle(rebuilt_options=self.OPTIONS))
        for extra in ({'invented': 1}, {'name': 'renamed'},
                      {'sha256': {'program.bin': '0' * 64}}):
            with self.subTest(extra=sorted(extra)):
                retained, rebuilt = self.bundle(rebuilt_options=self.OPTIONS, extra_manifest=extra)
                with self.assertRaises(ValueError):
                    verify_rebuild(retained, rebuilt)

    def test_the_native_identities_are_still_byte_compared(self):
        """The addition never widens what the BYTES may be - paired with the accepted bundle."""
        from agxforge.g17.abi import verify_rebuild
        verify_rebuild(*self.bundle(rebuilt_options=self.OPTIONS))
        for path, data in (('programs/p/program.bin', b'xxxx'), ('reference.bin', b'wrong')):
            with self.subTest(path=path):
                retained, rebuilt = self.bundle(rebuilt_options=self.OPTIONS)
                with self.assertRaises(ValueError):
                    verify_rebuild(retained, dict(rebuilt, **{path: data}))


class TheExplicitContractFieldsAreDerivedNotTolerated(unittest.TestCase):
    """agxforge.g17.abi's evolution for the four fields retained evidence predates.

    The full-FFN release gate carried this derivation privately and the composed-graph test had the
    same problem with no rule at all. The six controls below were the gate's; they live here now, so
    both callers inherit them instead of each keeping a copy - and a SECOND permissive allowlist
    cannot be written without failing one of them.
    """

    RETAINED = {"name": "pack", "entry": 64, "version": 3,
                "bindings": [{"index": 1, "offset": 0, "written": False, "element_type": "float"},
                             {"index": 2, "offset": 2, "written": True, "element_type": "half"}]}

    def evolved(self, **kwargs):
        from agxforge.g17.abi import with_explicit_fields
        return with_explicit_fields(self.RETAINED, **kwargs)

    def test_the_addition_is_derived_from_the_retained_bindings(self):
        state = self.evolved()["argument_state"]
        self.assertEqual(state["pointer_offsets"], [[1, 0], [2, 2]])
        self.assertEqual(state["block_words"], 4)
        self.assertEqual(state["block_bytes"], 16)
        self.assertTrue(state["indices_contiguous"])
        self.assertEqual(self.evolved()["promoted_ranges"], None)
        self.assertEqual(self.evolved()["spill_state"], None)

    def test_the_instruction_count_is_the_retained_boundary_count(self):
        from agxforge.g17.abi import evolution_differences
        fresh = self.evolved(instructions=[{"offset": 0, "length": 2}] * 17)
        self.assertEqual(fresh["main_instruction_count"], 17)
        self.assertEqual(evolution_differences(self.RETAINED, fresh, instructions=17), [])

    def test_a_fabricated_spill_state_is_not_tolerated(self):
        from agxforge.g17.abi import evolution_differences
        fresh = dict(self.evolved(), spill_state={"binding_index": 2, "words_per_thread": 4})
        self.assertTrue(evolution_differences(self.RETAINED, fresh))

    def test_a_fabricated_promoted_range_is_not_tolerated(self):
        from agxforge.g17.abi import evolution_differences
        fresh = dict(self.evolved(), promoted_ranges=[{"index": 1, "length": 4}])
        self.assertTrue(evolution_differences(self.RETAINED, fresh))

    def test_an_extra_field_inside_argument_state_is_not_tolerated(self):
        from agxforge.g17.abi import evolution_differences
        fresh = self.evolved()
        fresh["argument_state"] = dict(fresh["argument_state"], block_bytes_hint=16)
        self.assertTrue(evolution_differences(self.RETAINED, fresh))

    def test_a_wrong_value_under_an_added_key_is_not_tolerated(self):
        from agxforge.g17.abi import evolution_differences
        fresh = self.evolved()
        fresh["argument_state"] = dict(fresh["argument_state"], pointer_offsets=[[1, 0], [2, 4]])
        self.assertTrue(evolution_differences(self.RETAINED, fresh))

    def test_a_removed_or_changed_retained_key_is_not_tolerated(self):
        from agxforge.g17.abi import evolution_differences
        removed = self.evolved()
        removed.pop("entry")
        self.assertTrue(evolution_differences(self.RETAINED, removed))
        changed = dict(self.evolved(), entry=128)
        self.assertTrue(evolution_differences(self.RETAINED, changed))

    def test_a_contract_that_already_states_the_field_must_agree(self):
        from agxforge.g17.abi import with_explicit_fields
        contradicting = dict(self.RETAINED, argument_state={"pointer_offsets": [[9, 9]]})
        with self.assertRaises(ValueError):
            with_explicit_fields(contradicting)
        with self.assertRaises(ValueError):
            with_explicit_fields(dict(self.RETAINED, spill_state={"binding_index": 2}))
        stated = dict(self.RETAINED, main_instruction_count=17)
        with self.assertRaises(ValueError):
            with_explicit_fields(stated, instructions=3)
        self.assertEqual(with_explicit_fields(stated, instructions=17)
                         ["main_instruction_count"], 17)

    def test_offsets_that_are_not_the_measured_block_raise(self):
        from agxforge.g17.abi import with_explicit_fields
        odd = dict(self.RETAINED,
                   bindings=[{"index": 1, "offset": 0}, {"index": 2, "offset": 6}])
        with self.assertRaises(ValueError):
            with_explicit_fields(odd)

    def test_numeric_equality_does_not_hide_a_changed_json_type(self):
        from agxforge.g17.abi import evolution_differences
        for key, value in (("written", 0), ("index", 1.0), ("offset", False)):
            fresh = self.evolved()
            fresh["bindings"][0][key] = value
            with self.subTest(key=key):
                self.assertTrue(evolution_differences(self.RETAINED, fresh))


class AWrittenVectorBinding(unittest.TestCase):
    """A vector binding may be WRITTEN now, because its lanes are what the compiler writes.

    The refusal this replaces was right when it was written: a declaration-only element is carried
    and never touched, so marking one written meant the contract disagreed with the program. Lane
    accesses changed the second half of that - the element is still never touched WHOLE, and the
    lanes are - so the rule is now about the lane, and the declaration is untouched: uint4 stays
    uint4 at 16 bytes with 16-byte alignment.

    root's concrete defect: p.contract() on the unchanged source syn-s209163dca8 raised "binding
    declares uint4 ... and is marked written" while the raw wire ABI succeeded.
    """

    def _binding(self, spelling, written=True):
        from agxforge.g17.abi import Binding, ELEMENT_PHYSICAL
        return Binding(index=0, offset=0, written=written, element_type=spelling,
                       element_bytes=ELEMENT_PHYSICAL[spelling][0])

    def test_a_written_vector_binding_is_admitted_and_keeps_its_declaration(self):
        for spelling, lane, lanes, size in (("uint4", "uint", 4, 16), ("int4", "uint", 4, 16),
                                            ("float4", "float", 4, 16), ("uint2", "uint", 2, 8),
                                            ("float2", "float", 2, 8), ("half2", "half", 2, 4),
                                            ("half4", "half", 4, 8), ("short2", "ushort", 2, 4)):
            with self.subTest(spelling=spelling):
                binding = self._binding(spelling)
                self.assertEqual(binding.element_type, spelling)      # never relabelled
                self.assertEqual(binding.element_bytes, size)
                self.assertEqual(binding.element_alignment, size)
                self.assertEqual(binding.element_lane, lane)
                self.assertEqual(binding.element_lanes, lanes)
                self.assertFalse(binding.element_accessible,
                                 "the WHOLE element is still not accessible")
                self.assertTrue(binding.element_lane_accessible)

    def test_a_written_wide_scalar_is_ADMITTED_by_its_WORD_components(self):
        """`long` and `ulong` moved out of the negative controls, and this records why.

        A written binding is legal when SOME access reaches it. Word components are the fourth such
        access after whole-element, lane and atomic, so a written wide scalar is now valid - while
        an ordinary whole-element load or store of it still refuses, which is what keeps the
        declaration meaningful. The declaration itself is unchanged: one 8-byte element, 8-byte
        alignment, and NOT a vector.
        """
        from agxforge.g17.abi import ELEMENT_WORD_COMPONENT
        for spelling in ("long", "ulong"):
            with self.subTest(spelling=spelling):
                written = self._binding(spelling)
                self.assertEqual(written.element_type, spelling)
                self.assertEqual(written.element_bytes, 8)
                self.assertTrue(written.element_word_component_accessible)
                self.assertFalse(written.element_accessible,
                                 "the WHOLE element is still not accessible")
                self.assertFalse(written.element_lane_accessible,
                                 "and it is not a vector with lanes")
                self.assertEqual(ELEMENT_WORD_COMPONENT[spelling], "uint")

    def test_a_written_binding_with_no_reachable_access_still_refuses(self):
        """The negative controls that remain: bfloat vectors and the scalars with no access."""
        for spelling in ("bfloat2", "bfloat4", "bfloat", "uchar"):
            with self.subTest(spelling=spelling):
                with self.assertRaises(Exception) as caught:
                    self._binding(spelling)
                self.assertIn("not as a whole element, not one lane at a time, not one word at a time and not atomically",
                              str(caught.exception))
                unwritten = self._binding(spelling, written=False)   # still carried
                self.assertEqual(unwritten.element_type, spelling)
                self.assertFalse(unwritten.element_lane_accessible)
                self.assertFalse(unwritten.element_word_component_accessible,
                                 "and none of these has a word decomposition either")

    def test_bfloat2_and_half2_are_the_pair_that_makes_the_lane_column_necessary(self):
        from agxforge.g17.abi import ELEMENT_PHYSICAL
        self.assertEqual(ELEMENT_PHYSICAL["bfloat2"][:3], ELEMENT_PHYSICAL["half2"][:3],
                         "size, alignment and whole-element accessibility are identical")
        self.assertIsNone(ELEMENT_PHYSICAL["bfloat2"][3])
        self.assertEqual(ELEMENT_PHYSICAL["half2"][3], "half")

    def test_the_two_admitted_vector_sources_produce_a_contract_that_round_trips(self):
        """The whole point: root's two sources, their real contracts, serialized and rebuilt."""
        import json
        import re
        from pathlib import Path
        snapshot = Path(ROOT) / "results/g17-source-admission-v3/air-snapshot.json"
        if not snapshot.is_file():
            self.skipTest("the frozen v3 AIR is not extracted in this checkout")
        sys.path.insert(0, str(ROOT / "tools"))
        import g17cc
        import g17front
        from agxforge.g17 import abi
        records = {r["tag"]: r["air_text"] for r in json.loads(snapshot.read_text())["records"]
                   if "air_text" in r}
        want = {"syn-s209163dca8": ("uint4", 16, True), "hold-b-vec0": ("float2", 8, False)}
        for tag, (spelling, size, written) in want.items():
            with self.subTest(tag=tag):
                name = re.sub(r"[^A-Za-z0-9_]", "_", tag)
                program = g17cc.compile_function(g17front.to_ir(records[tag], name=name))
                contract = program.contract()
                vectors = [b for b in contract.bindings if b.element_lane is not None]
                self.assertEqual([(b.element_type, b.element_bytes, b.written) for b in vectors],
                                 [(spelling, size, written)])
                rebuilt = abi.ADAPTER.validate_json(abi.ADAPTER.dump_json(contract))
                self.assertEqual(rebuilt, contract)


if __name__ == "__main__":
    unittest.main()
