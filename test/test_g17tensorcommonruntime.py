"""Focused contract and native-worker checks for the measured ordinary tensor class."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import g17tensorcommonruntime as tensor_runtime
import g17commonruntime as runtime


class TensorCommonRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = Path(tempfile.mkdtemp(prefix="g17-tensor-worker-test-")) / "common-worker"
        subprocess.run(["clang", "-fobjc-arc", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-framework", "Foundation", "-framework", "Metal", "-o", str(cls.worker),
                        str(ROOT / "tools/g17commonworker.m")], cwd=ROOT, check=True,
                       capture_output=True, timeout=30)

    def bundle(self, composition="single", weight_offset=4096):
        directory = Path(tempfile.mkdtemp(prefix="g17-tensor-bundle-test-")) / "bundle"
        tensor_runtime.author(directory, composition=composition, weight_offset=weight_offset)
        return directory

    def test_reference_uses_measured_c_first_mma_order(self):
        """The C placement is observable and must stay independent of the implementation.

        [Corrected 2026-09-23: this test pinned C LAST. Recon section 136 measured C first, and the
        released multigemm bundle on main's code matches the C-first model in 1024 of 1024
        elements and the C-last one in 607 (docs/archive/g17-tensor-register-chain.md).]"""
        rng = np.random.default_rng(1)
        a = rng.uniform(-10000.0, 10000.0, size=16).astype("<f4")
        b = rng.uniform(-10000.0, 10000.0, size=16).astype("<f4")
        c = np.float32(2469.7951107500085)

        p = [np.float32(np.float32(float(a[2 * i]) * float(b[2 * i])) +
                        np.float32(float(a[2 * i + 1]) * float(b[2 * i + 1])))
             for i in range(8)]
        q = [np.float32(p[j] + p[j + 4]) for j in range(4)]
        c_first = np.float32(c)
        for value in q:
            c_first = np.float32(c_first + value)
        c_last = np.float32(0.0)
        for value in q:
            c_last = np.float32(c_last + value)
        c_last = np.float32(c_last + c)

        got = tensor_runtime._mma16(a, b, c)
        self.assertEqual(got.view("<u4"), c_first.view("<u4"))
        self.assertNotEqual(c_first.view("<u4"), c_last.view("<u4"),
                            "the regression stimulus must distinguish C-first from C-last")

    def test_gemm_reference_adds_the_accumulate_input_once_after_the_chain(self):
        """tlower adds an accumulate C with one fadd after the whole K chain, not per issue."""
        rng = np.random.default_rng(7)
        a = rng.uniform(-2.0, 2.0, size=(16, 32)).astype("<f2")
        b = rng.uniform(-2.0, 2.0, size=(32, 16)).astype("<f2")
        c = rng.uniform(-3000.0, 3000.0, size=(16, 16)).astype("<f4")
        chain = tensor_runtime._gemm_mma(a, b, None, 16, 16, 32)
        got = tensor_runtime._gemm_mma(a, b, c, 16, 16, 32)
        self.assertTrue(np.array_equal(got.view("<u4"), np.asarray(chain + c, dtype="<f4").view("<u4")))

    def test_continuation_reference_truncates_its_fp32_first_operand(self):
        """A continuation body's first float-A GEMM uses the measured operand truncation."""
        rng = np.random.default_rng(1729)
        inp = rng.uniform(-2.0, 2.0, size=(16, 32)).astype("<f4")
        weights = rng.uniform(-2.0, 2.0, size=5120).astype("<f2")
        calls = []
        real = tensor_runtime._gemm_mma

        def observe(*args, **kwargs):
            calls.append(bool(kwargs.get("truncate_a", False)))
            return real(*args, **kwargs)

        tensor_runtime._gemm_mma = observe
        try:
            tensor_runtime._transformer_layer_weight_offset_reference(inp, weights)
        finally:
            tensor_runtime._gemm_mma = real
        self.assertGreaterEqual(len(calls), 3)
        self.assertTrue(calls[0], "the continuation's first FP32-A body lost truncation")

    def test_compiled_composed_contract_is_the_tensor_class(self):
        bundle = self.bundle()
        manifest = json.loads((bundle / "manifest.json").read_text())
        contract = runtime.ImageContract.read(manifest)
        self.assertEqual(contract.kind, runtime.TENSOR_KIND)
        self.assertEqual(contract.format, "g17-common-pipeline-v3")
        self.assertEqual(contract.tensor.composition, "gemm_fadd_threadgroup_position")
        self.assertEqual(contract.abi.system_registers, (130, 156))
        self.assertEqual(contract.abi.bindings[-1].index, 3)
        self.assertEqual(contract.layout()["buffer_payload_bytes"], [544, 608, 1292])

    def test_describe_layout_is_worker_side_and_does_not_dispatch(self):
        bundle = self.bundle()
        result = subprocess.run([str(self.worker), str(bundle), "--describe-layout"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        layout = json.loads(result.stdout)
        self.assertEqual(layout["kind"], "tensor_gemm")
        self.assertEqual(layout["gpu_dispatched"], False)
        self.assertEqual(layout["buffer_offsets"], [128, 128, 128])

    def test_tensor_contract_mutations_refuse_by_their_own_boundary(self):
        bundle = self.bundle()
        original = json.loads((bundle / "manifest.json").read_text())
        mutations = {
            "format": lambda m: m.update(format="g17-common-pipeline-v2"),
            "kind": lambda m: m.update(kind="packed_scan"),
            "name": lambda m: m.update(name="other"),
            "shape": lambda m: m["shape"].update(rows=16),
            "tensor_shape": lambda m: m["tensor"].update(K=32),
            "composition": lambda m: m["tensor"].update(composition="gemm_only"),
            "sr": lambda m: m["abi"].update(system_registers=[130]),
            "binding": lambda m: m["abi"]["bindings"][2].update(index=2),
            "execution": lambda m: m["abi"].update(execution={"simd_width": 16, "tensor": True}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                value = copy.deepcopy(original)
                mutate(value)
                with self.assertRaises(ValueError):
                    runtime.ImageContract.read(value)

    def test_worker_requires_the_tensor_approval_token(self):
        bundle = self.bundle()
        result = subprocess.run([str(self.worker), str(bundle), "--load-approved"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("phase=tensor_mode", result.stderr)

    def test_compiler_owned_row_softmax_contract_is_admitted_without_dispatch(self):
        bundle = self.bundle("reduction_softmax")
        manifest = json.loads((bundle / "manifest.json").read_text())
        contract = runtime.ImageContract.read(manifest)
        self.assertEqual(contract.name, "tensor_row_softmax_runtime_demo")
        self.assertEqual(contract.tensor.composition, "row_softmax_fp32")
        self.assertEqual((contract.tensor.M, contract.tensor.N, contract.tensor.K), (32, 16, 64))
        self.assertEqual(contract.abi.system_registers, (130, 156))
        self.assertIn([14169, 10], manifest["abi"]["forms"])
        result = subprocess.run([str(self.worker), str(bundle), "--describe-layout"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["composition"], "row_softmax_fp32")

    def test_row_softmax_contract_refuses_unmeasured_domains(self):
        bundle = self.bundle("reduction_softmax")
        original = json.loads((bundle / "manifest.json").read_text())
        for field, value in (("M", 16), ("K", 32), ("simdgroups", 2), ("a_type", "bfloat")):
            with self.subTest(field=field):
                value_copy = copy.deepcopy(original)
                value_copy["tensor"][field] = value
                with self.assertRaises(ValueError):
                    runtime.ImageContract.read(value_copy)

    def test_ffn_gelu_layernorm_contract_and_worker_layout(self):
        bundle = self.bundle("ffn_gelu_layernorm")
        manifest = json.loads((bundle / "manifest.json").read_text())
        contract = runtime.ImageContract.read(manifest)
        self.assertEqual(contract.name, "tensor_ffn_gelu_layernorm_runtime_demo")
        self.assertEqual(contract.tensor.composition, "ffn_gelu_layernorm")
        self.assertEqual((contract.tensor.M, contract.tensor.N, contract.tensor.K,
                          contract.tensor.K2), (16, 16, 64, 16))
        self.assertEqual((contract.tensor.a_type, contract.tensor.b_type,
                          contract.tensor.a2_type, contract.tensor.b2_type),
                         ("half", "half", "float", "half"))
        self.assertEqual(contract.abi.system_registers, (130, 156))
        result = subprocess.run([str(self.worker), str(bundle), "--describe-layout"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        layout = json.loads(result.stdout)
        self.assertEqual(layout["composition"], "ffn_gelu_layernorm")
        self.assertEqual(layout["gpu_dispatched"], False)

    def test_ffn_contract_refuses_unmeasured_domains(self):
        bundle = self.bundle("ffn_gelu_layernorm")
        original = json.loads((bundle / "manifest.json").read_text())
        mutations = {
            "M": lambda m: m["tensor"].update(M=32),
            "K2": lambda m: m["tensor"].update(K2=32),
            "a2_type": lambda m: m["tensor"].update(a2_type="half"),
            "composition": lambda m: m["tensor"].update(composition="gemm_fadd_gemm_memory"),
            "sr": lambda m: m["abi"].update(system_registers=[130]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                value = copy.deepcopy(original)
                mutate(value)
                with self.assertRaises(ValueError):
                    runtime.ImageContract.read(value)

    def test_ffn_all_rows_contract_is_admitted_without_dispatch(self):
        bundle = self.bundle("ffn_gelu_layernorm_allrows")
        manifest = json.loads((bundle / "manifest.json").read_text())
        contract = runtime.ImageContract.read(manifest)
        self.assertEqual(contract.name, "tensor_ffn_gelu_layernorm_allrows_runtime_demo")
        self.assertEqual(contract.tensor.composition, "ffn_gelu_layernorm_allrows")
        self.assertEqual((contract.tensor.M, contract.tensor.N, contract.tensor.K,
                          contract.tensor.K2), (16, 16, 64, 16))
        result = subprocess.run([str(self.worker), str(bundle), "--describe-layout"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["composition"], "ffn_gelu_layernorm_allrows")

    def test_ffn_wide_two_tile_contract_is_admitted_without_dispatch(self):
        bundle = self.bundle("ffn_gelu_layernorm_wide")
        manifest = json.loads((bundle / "manifest.json").read_text())
        contract = runtime.ImageContract.read(manifest)
        self.assertEqual(contract.name, "tensor_ffn_gelu_layernorm_wide_runtime_demo")
        self.assertEqual(contract.tensor.composition, "ffn_gelu_layernorm_wide")
        self.assertEqual((contract.tensor.M, contract.tensor.N, contract.tensor.K,
                          contract.tensor.K2), (16, 32, 64, 32))
        result = subprocess.run([str(self.worker), str(bundle), "--describe-layout"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["composition"], "ffn_gelu_layernorm_wide")

    def test_transformer_layer_contract_is_admitted_without_dispatch(self):
        bundle = self.bundle("transformer_layer")
        manifest = json.loads((bundle / "manifest.json").read_text())
        contract = runtime.ImageContract.read(manifest)
        self.assertEqual(contract.name, "tensor_transformer_layer_runtime_demo")
        self.assertEqual(contract.tensor.composition, "transformer_layer")
        self.assertEqual((contract.tensor.M, contract.tensor.N, contract.tensor.K,
                          contract.tensor.K2, contract.tensor.K3), (16, 32, 64, 32, 32))
        self.assertEqual(contract.abi.system_registers, (130, 156))
        forms = {(item.opcode, item.length) for item in contract.instructions}
        self.assertGreaterEqual(sum(opcode in (5100, 5101, 5106, 5107) for opcode, _ in forms), 4)
        for opcode in (1004, 9700, 10279, 11375, 14169, 3290, 17257):
            self.assertIn(opcode, {opcode for opcode, _ in forms})
        result = subprocess.run([str(self.worker), str(bundle), "--describe-layout"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["composition"], "transformer_layer")

    def test_two_transformer_layer_contract_is_admitted_without_dispatch(self):
        bundle = self.bundle("transformer_two_layer")
        manifest = json.loads((bundle / "manifest.json").read_text())
        contract = runtime.ImageContract.read(manifest)
        self.assertEqual(contract.name, "tensor_transformer_two_layer_runtime_demo")
        self.assertEqual(contract.tensor.composition, "transformer_two_layer")
        self.assertEqual((contract.tensor.M, contract.tensor.N, contract.tensor.K,
                          contract.tensor.K2, contract.tensor.K3), (16, 32, 64, 32, 32))
        self.assertEqual(contract.abi.system_registers, (130, 156))
        forms = {(item.opcode, item.length) for item in contract.instructions}
        self.assertGreaterEqual(sum(opcode in (5100, 5101, 5106, 5107)
                                    for opcode, _ in forms), 4)
        for opcode in (1004, 9700, 10279, 11375, 14169, 3290, 17257, 12646):
            self.assertIn(opcode, {opcode for opcode, _ in forms})
        result = subprocess.run([str(self.worker), str(bundle), "--describe-layout"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["composition"], "transformer_two_layer")

    def test_two_transformer_layer_contract_refuses_unmeasured_domains(self):
        bundle = self.bundle("transformer_two_layer")
        original = json.loads((bundle / "manifest.json").read_text())
        mutations = {
            "K3": lambda m: m["tensor"].update(K3=64),
            "a2_type": lambda m: m["tensor"].update(a2_type="half"),
            "simdgroups": lambda m: m["tensor"].update(simdgroups=2),
            "sr": lambda m: m["abi"].update(system_registers=[130]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                value = copy.deepcopy(original)
                mutate(value)
                with self.assertRaises(ValueError):
                    runtime.ImageContract.read(value)

    def test_multigemm_memory_chain_contract_is_admitted_without_dispatch(self):
        bundle = self.bundle("multigemm")
        manifest = json.loads((bundle / "manifest.json").read_text())
        contract = runtime.ImageContract.read(manifest)
        self.assertEqual(contract.name, "tensor_multigemm_runtime_demo")
        self.assertEqual(contract.tensor.composition, "gemm_fadd_gemm_memory")
        self.assertEqual((contract.tensor.M, contract.tensor.N, contract.tensor.K, contract.tensor.K2),
                         (32, 32, 64, 32))
        self.assertEqual((contract.tensor.a2_type, contract.tensor.b2_type), ("float", "half"))
        result = subprocess.run([str(self.worker), str(bundle), "--describe-layout"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["composition"], "multigemm")

    def test_multigemm_fadd_fmul_contract_is_admitted_without_dispatch(self):
        bundle = self.bundle("multigemm_fadd_fmul")
        manifest = json.loads((bundle / "manifest.json").read_text())
        contract = runtime.ImageContract.read(manifest)
        self.assertEqual(contract.name, "tensor_multigemm_fadd_fmul_runtime_demo")
        self.assertEqual(contract.tensor.composition, "gemm_fadd_fmul_gemm_memory")
        self.assertEqual((contract.tensor.M, contract.tensor.N, contract.tensor.K, contract.tensor.K2),
                         (32, 32, 64, 32))
        result = subprocess.run([str(self.worker), str(bundle), "--describe-layout"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["composition"], "multigemm")

    def test_multigemm_fadd_fmul_does_not_accept_the_one_add_contract_label(self):
        bundle = self.bundle("multigemm_fadd_fmul")
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["tensor"]["composition"] = "gemm_fadd_gemm_memory"
        with self.assertRaises(ValueError):
            runtime.ImageContract.read(manifest)

    def test_three_gemm_memory_chain_contract_is_admitted_without_dispatch(self):
        bundle = self.bundle("multigemm3")
        manifest = json.loads((bundle / "manifest.json").read_text())
        contract = runtime.ImageContract.read(manifest)
        self.assertEqual(contract.name, "tensor_multigemm_fadd_fmul_gemm_runtime_demo")
        self.assertEqual(contract.tensor.composition, "gemm_fadd_fmul_gemm_fadd_gemm_memory")
        self.assertEqual((contract.tensor.K, contract.tensor.K2, contract.tensor.K3), (64, 32, 32))
        result = subprocess.run([str(self.worker), str(bundle), "--describe-layout"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["composition"], "multigemm")

    def test_relu_vector_residual_composition_is_structurally_admitted(self):
        bundle = self.bundle("multigemm_relu_vec")
        manifest = json.loads((bundle / "manifest.json").read_text())
        contract = runtime.ImageContract.read(manifest)
        self.assertEqual(contract.name, "tensor_multigemm_relu_vec_residual_runtime_demo")
        self.assertEqual(contract.tensor.composition, "gemm_relu_vec_gemm_residual_memory")
        self.assertEqual((contract.tensor.K, contract.tensor.K2, contract.tensor.K3), (64, 32, 32))
        forms = {(item.opcode, item.length) for item in contract.instructions}
        self.assertIn((9700, 14), forms)       # compiler-owned fmax nonlinear stage
        self.assertIn((17256, 8), forms)       # measured four-word vector store
        self.assertGreaterEqual(sum(opcode in (5100, 5101, 5106, 5107) for opcode, _ in forms), 4)
        result = subprocess.run([str(self.worker), str(bundle), "--describe-layout"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["composition"], "multigemm")

    def test_weight_offset_class_admits_measured_same_binding_offsets(self):
        code_hashes = set()
        for offset in (4096, 4352, 8192, 12288, 16384, 20480):
            with self.subTest(offset=offset):
                bundle = self.bundle("weight_offset", weight_offset=offset)
                manifest = json.loads((bundle / "manifest.json").read_text())
                contract = runtime.ImageContract.read(manifest)
                self.assertEqual(contract.name, "tensor_multigemm_weight_offset_runtime_demo")
                self.assertEqual(contract.tensor.composition, "gemm_weight_offset")
                self.assertEqual(contract.tensor.weight_offset_b, offset)
                self.assertEqual(contract.abi.system_registers, (130, 156))
                self.assertEqual(contract.layout()["buffer_payload_bytes"], [2048, offset + 2048, 2048])
                code_hashes.add(manifest["sha256"]["code"])
                result = subprocess.run([str(self.worker), str(bundle), "--describe-layout"],
                                        cwd=ROOT, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                layout = json.loads(result.stdout)
                self.assertEqual(layout["composition"], "weight_offset")
                self.assertEqual(layout["buffer_payload_bytes"], [2048, offset + 2048, 2048])
        self.assertEqual(len(code_hashes), 6)

    def test_weight_offset_class_refuses_unmeasured_offset_and_extra_binding(self):
        bundle = self.bundle("weight_offset", weight_offset=4096)
        original = json.loads((bundle / "manifest.json").read_text())
        for name, mutate in (
                ("offset", lambda m: m["tensor"].update(weight_offset_b=1024)),
                ("shape", lambda m: m["tensor"].update(M=32)),
                ("binding", lambda m: m["abi"]["bindings"].append({
                    "index": 4, "offset": 6, "element_type": "half", "element_bytes": 2,
                    "written": False}))):
            with self.subTest(name=name):
                value = copy.deepcopy(original)
                mutate(value)
                with self.assertRaises(ValueError):
                    runtime.ImageContract.read(value)

    def test_transformer_layer_offset_class_admits_exact_three_regions(self):
        bundle = self.bundle("transformer_layer_weight_offset")
        manifest = json.loads((bundle / "manifest.json").read_text())
        contract = runtime.ImageContract.read(manifest)
        self.assertEqual(contract.name, "tensor_transformer_layer_weight_offset_runtime_demo")
        self.assertEqual(contract.tensor.composition, "transformer_layer_weight_offset")
        self.assertEqual(contract.tensor.weight_offsets_b, (0, 4096, 8192))
        self.assertEqual(contract.layout()["buffer_payload_bytes"], [2048, 10240, 2048])
        self.assertEqual(contract.abi.system_registers, (130, 156))
        result = subprocess.run([str(self.worker), str(bundle), "--describe-layout"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        layout = json.loads(result.stdout)
        self.assertEqual(layout["composition"], "transformer_layer_weight_offset")
        self.assertEqual(layout["buffer_payload_bytes"], [2048, 10240, 2048])

    def test_transformer_layer_offset_class_refuses_unmeasured_region(self):
        bundle = self.bundle("transformer_layer_weight_offset")
        original = json.loads((bundle / "manifest.json").read_text())
        value = copy.deepcopy(original)
        value["tensor"]["weight_offsets_b"] = [0, 4096, 12288]
        with self.assertRaises(ValueError):
            runtime.ImageContract.read(value)

    def test_transformer_continuation_accepts_fp32_activation_and_local_regions(self):
        bundle = self.bundle("transformer_continuation_weight_offset")
        manifest = json.loads((bundle / "manifest.json").read_text())
        contract = runtime.ImageContract.read(manifest)
        self.assertEqual(contract.tensor.composition, "transformer_continuation_weight_offset")
        self.assertEqual((contract.tensor.a_type, contract.tensor.b_type), ("float", "half"))
        self.assertEqual(contract.tensor.weight_offsets_b, (0, 4096, 8192))
        self.assertEqual(contract.layout()["buffer_payload_bytes"], [2048, 10240, 2048])
        self.assertEqual([b.element_type for b in contract.abi.bindings], ["float", "half", "float"])
        result = subprocess.run([str(self.worker), str(bundle), "--describe-layout"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        layout = json.loads(result.stdout)
        self.assertEqual(layout["composition"], "transformer_continuation_weight_offset")
        self.assertEqual(layout["buffer_payload_bytes"], [2048, 10240, 2048])

    def test_gpu_resident_sequence_manifest_contains_two_measured_layer_contracts(self):
        directory = Path(tempfile.mkdtemp(prefix="g17-tensor-sequence-test-")) / "bundle"
        tensor_runtime.author_transformer_sequence(directory, length=2)
        sequence = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(sequence["kind"], "tensor_gemm_sequence")
        self.assertEqual(sequence["layers"], 2)
        self.assertEqual(sequence["activation"]["host_readback"], False)
        self.assertEqual(sequence["weight_regions"], [0, 4096, 8192])
        initial = runtime.ImageContract.read(sequence["initial"])
        continuation = runtime.ImageContract.read(sequence["continuation"])
        self.assertEqual(initial.tensor.composition, "transformer_layer_weight_offset")
        self.assertEqual(continuation.tensor.composition, "transformer_continuation_weight_offset")
        self.assertEqual(len(list(directory.glob("b*.f16"))), 2)


if __name__ == "__main__":
    unittest.main()
