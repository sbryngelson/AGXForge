"""Exercise private runtime snapshots and the committed native source closure."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import g17attentiongraph
import g17attentionstage as stage


class AttentionPreparation(unittest.TestCase):
    def test_snapshot_copies_exact_members_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"source";target=Path(tmp)/"target"
            for name in stage.MEMBERS:
                p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
            report=stage.snapshot(source,target)
            self.assertEqual(report,{n:stage.sha(n.encode()) for n in stage.MEMBERS})
            with self.assertRaises(FileExistsError):stage.snapshot(source,target)

    def test_snapshot_detects_source_replacement_during_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"source";target=Path(tmp)/"target"
            for name in stage.MEMBERS:
                p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
            original=Path.read_bytes;reads=0
            def moving(path):
                nonlocal reads
                value=original(path);reads+=1
                if reads==len(stage.MEMBERS): (source/"manifest.json").write_bytes(b"moved")
                return value
            with mock.patch.object(Path,"read_bytes",moving),self.assertRaisesRegex(ValueError,"changed during snapshot"):
                stage.snapshot(source,target)

    def test_worker_build_uses_declared_committed_inputs_and_parses_real_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);report=stage.build_worker(tmp/"worker")
            self.assertEqual(set(report["inputs"]),set(stage.WORKER_SOURCES))
            self.assertIn("tools/g17textureresources.h",report["inputs"])
            self.assertIn('tools/g17launch.h',report['inputs'])
            self.assertEqual(report["sha256"],stage.sha((tmp/"worker").read_bytes()))
            (tmp/"graph.json").write_text(json.dumps(g17attentiongraph.graph()))
            result=subprocess.run([str(tmp/"worker"),str(tmp/"graph.json"),"--describe-schedule"],
                                  capture_output=True,text=True,check=True,timeout=10)
            native=json.loads(result.stdout)
            self.assertEqual(len(native["schedule"]),10)
            self.assertFalse(native["gpu_dispatched"])


if __name__=="__main__":unittest.main()
