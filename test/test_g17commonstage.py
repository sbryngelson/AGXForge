"""Exercise staging refusals and persistent transport without using a GPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import g17commonimage as image
import g17commonruntime as runtime
import g17commonstage as stage
from g17native import Worker


FAKE = r'''
import json,struct,sys
bundle=sys.argv[1]
m=json.load(open(bundle+'/manifest.json'))
rows=m['shape']['rows']
identity=dict(pipeline=1,buffers=[2,3],matrix_bytes=2*rows,output_bytes=2*rows+128,
 storage_dtype='float16',pipeline_builds=1,matrix_uploads=1,buffer_allocations=2,
 bindings=m['abi']['bindings'])
def emit(sequence,payload=b'',**extra):
 data=json.dumps(dict(protocol=1,sequence=sequence,bytes=len(payload),identity=identity,**extra)).encode()
 sys.stdout.buffer.write(struct.pack('<I',len(data))+data+payload);sys.stdout.buffer.flush()
emit(0,rows=rows,columns=1)
for i in range(1,4):
 raw=sys.stdin.buffer.read(4)
 if not raw:break
 size,=struct.unpack('<I',raw)
 assert size==2*rows
 values=struct.unpack('<%de'%rows,sys.stdin.buffer.read(size))
 result=struct.pack('<%de'%rows,*(2*x+1 for x in values))
 emit(i,result,status=0,boundary_guard=True)
'''


class CommonStages(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bundle = Path(self.temporary.name)/"bundle"
        image.prepare(self.bundle, "affine", 33, 1, verify_source=False)

    def test_unresolved_class_never_reaches_worker_build_or_lock_and_cannot_retry(self):
        # Inject an explicit unresolved fact so this refusal remains a control
        # after the linker owner finishes today's missing measurements.
        path = self.bundle/"manifest.json"
        m = json.loads(path.read_text())
        m["field_ledger"]["unresolved control"] = "ASSUMED control, no measurement"
        path.write_text(json.dumps(m))
        with patch.object(stage, "build_runner", side_effect=AssertionError("worker build reached")), \
             patch.object(stage, "lock_gpu", side_effect=AssertionError("GPU lock reached")):
            with self.assertRaisesRegex(ValueError, "metadata class still assumes"):
                stage.load_only(self.bundle, approved=True)
            receipt = (self.bundle/stage.RECEIPT).read_bytes()
            self.assertEqual(json.loads(receipt)["status"], "failed")
            self.assertFalse(json.loads(receipt)["gpu_dispatched"])
            with self.assertRaises(FileExistsError):
                stage.load_only(self.bundle, approved=True)
            self.assertEqual((self.bundle/stage.RECEIPT).read_bytes(), receipt)

    def test_erasing_assumptions_cannot_pass_the_fresh_source_rebuild(self):
        path = self.bundle/"manifest.json"
        m = json.loads(path.read_text())
        # A forged but syntactically valid ledger: no assumption keywords remain.
        m["field_ledger"] = {"pretend": "measured"}
        path.write_text(json.dumps(m))
        with patch.object(stage, "build_runner", side_effect=AssertionError("worker build reached")), \
             patch.object(stage, "lock_gpu", side_effect=AssertionError("GPU lock reached")):
            with self.assertRaisesRegex(ValueError, "rebuild refused.*contract differs"):
                stage.load_only(self.bundle, approved=True)

    def test_receipt_requires_every_delivered_file_and_the_exact_worker(self):
        (self.bundle/stage.RUNNER).write_bytes(b"fake worker bytes, never executable")
        receipt = dict(status="passed", returncode=0,
            native=dict(status=0, load_only=True, gpu_dispatched=False),
            rebuild=dict(status="passed"),
            files={name: stage.digest(self.bundle/name) for name in image.FILES},
            worker=dict(sha256=stage.digest(self.bundle/stage.RUNNER)))
        (self.bundle/stage.RECEIPT).write_text(json.dumps(receipt))
        self.assertTrue(stage.receipt_matches(self.bundle))
        for name in (*image.FILES, stage.RUNNER):
            with self.subTest(name=name):
                path = self.bundle/name
                original = path.read_bytes()
                path.write_bytes(original+b"changed")
                self.assertFalse(stage.receipt_matches(self.bundle))
                path.write_bytes(original)

    def test_snapshot_owns_files_and_full_stage_is_explicitly_bounded(self):
        frozen = Path(self.temporary.name)/"frozen"
        frozen.mkdir()
        c = stage.snapshot(self.bundle, frozen)
        self.assertFalse(stage.stage_shape(c, False))
        with self.assertRaisesRegex(ValueError, "500000 x 384"):
            stage.stage_shape(c, True)
        original = (frozen/"program.bin").read_bytes()
        (self.bundle/"program.bin").write_bytes(b"changed")
        self.assertEqual((frozen/"program.bin").read_bytes(), original)
        value = c.model_dump(mode="json")
        value["shape"]["rows"] = 129
        large = runtime.ImageContract.read(value)
        with self.assertRaisesRegex(ValueError, "128 rows"):
            stage.stage_shape(large, False)

    def test_affine_session_checks_shape_domain_identity_budget_and_owned_replies(self):
        (self.bundle/stage.RECEIPT).write_text("{}")
        (self.bundle/stage.RUNNER).write_text("not executed")
        # Stub only the hardware prerequisites and executable choice. Use the
        # real snapshot, matrix ownership, framing, validation and session code.
        def fake_worker(command, **kwargs):
            return Worker([sys.executable, "-u", "-c", FAKE, command[1]], **kwargs)
        with patch.object(image, "verify", return_value={}), \
             patch.object(stage, "receipt_matches", return_value=True), \
             patch.object(stage, "source_check", return_value={}), \
             patch.object(stage, "Worker", side_effect=fake_worker), \
             patch("g17packeddispatch.gpu_events", return_value={}):
            # A private file models the lock lifetime, without acquiring a GPU lock.
            lock = tempfile.TemporaryFile()
            with patch.object(stage, "lock_gpu", return_value=lock):
                matrix = np.zeros((33,1), np.float16)
                with stage.NativeProgram(matrix, self.bundle, approved=True) as native:
                    matrix[:] = 100
                    self.assertEqual((Path(native.directory.name)/"matrix.f16").read_bytes(), bytes(66))
                    for invalid in (np.ones(1,np.float16), np.full(33,np.inf,np.float16),
                                    np.full(33,40000,np.float16)):
                        with self.assertRaises(ValueError):
                            native(invalid)
                    self.assertEqual(native.calls, 0)
                    x = np.arange(33,dtype=np.float16)
                    first = native(x)
                    first[:] = -1
                    second = native(-x)
                    third = native(x)
                    np.testing.assert_array_equal(second, 1-2*x)
                    np.testing.assert_array_equal(third, 1+2*x)
                    with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                        native(x)
                self.assertTrue(native.closed)
                self.assertTrue(lock.closed)

    def test_failed_guard_closes_worker_and_prevents_another_query(self):
        (self.bundle/stage.RECEIPT).write_text("{}")
        (self.bundle/stage.RUNNER).write_text("not executed")
        failing = FAKE.replace("boundary_guard=True", "boundary_guard=(i == 1)")
        def fake_worker(command, **kwargs):
            return Worker([sys.executable, "-u", "-c", failing, command[1]], **kwargs)
        lock = tempfile.TemporaryFile()
        with patch.object(image, "verify", return_value={}), \
             patch.object(stage, "receipt_matches", return_value=True), \
             patch.object(stage, "source_check", return_value={}), \
             patch.object(stage, "lock_gpu", return_value=lock), \
             patch.object(stage, "Worker", side_effect=fake_worker), \
             patch("g17packeddispatch.gpu_events", return_value={}):
            native = stage.NativeProgram(np.zeros((33,1),np.float16), self.bundle, approved=True)
            x = np.zeros(33,np.float16)
            np.testing.assert_array_equal(native(x), np.ones(33,np.float16))
            with self.assertRaisesRegex(RuntimeError, "boundary guard"):
                native(x)
            self.assertTrue(native.closed)
            self.assertTrue(native.worker.closed)
            self.assertTrue(lock.closed)
            self.assertIsNotNone(native.worker.proc.poll())
            with self.assertRaisesRegex(RuntimeError, "closed or failed"):
                native(x)


if __name__ == "__main__":
    unittest.main()
