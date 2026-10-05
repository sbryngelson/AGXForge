"""Freeze and rebuild attention deliveries and the native worker before Metal use."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
PROGRAMS=("projection","layernorm","scores","softmax","context","residual")
MEMBERS=("manifest.json",)+tuple(f"programs/{p}/{f}" for p in PROGRAMS for f in
    ("program.bin","program.o","program.lib.metallib","program.arc.metallib"))
WORKER_SOURCES=tuple("tools/"+f for f in ("g17attentionworker.m","g17attentionexecutor.m",
    "g17attentionexecutor.h","g17attentionmetal.h","g17attentionschedule.h",
    "g17attentionstorage.h","g17scanstorage.h","g17launch.h","g17textureresources.h",
    "g17gpulock.h"))


def sha(data):return hashlib.sha256(data).hexdigest()


def snapshot(source,destination):
    """Copy a fixed member set and verify both sides after copying."""
    source,destination=Path(source),Path(destination)
    data={name:(source/name).read_bytes() for name in MEMBERS}
    identities={name:sha(value) for name,value in data.items()}
    for name,value in data.items():
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("xb") as out:out.write(value)
    if identities!={name:sha((source/name).read_bytes()) for name in MEMBERS} or \
       identities!={name:sha((destination/name).read_bytes()) for name in MEMBERS}:
        raise ValueError("attention delivery changed during snapshot")
    return identities


def build_worker(destination):
    import g17buildaudit
    revision=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    expected=g17buildaudit.committed_hashes(ROOT,revision,WORKER_SOURCES)
    with tempfile.TemporaryDirectory(prefix="g17-attention-worker-") as tmp:
        tmp=Path(tmp)
        for name in WORKER_SOURCES:
            data=(ROOT/name).read_bytes()
            if sha(data)!=expected[name]:raise ValueError(f"worker source differs from commit: {name}")
            (tmp/Path(name).name).write_bytes(data)
        result=subprocess.run(["clang","-fobjc-arc","-O2","-Wall","-Wextra","-Werror",
            "-framework","Foundation","-framework","Metal",str(tmp/"g17attentionworker.m"),
            str(tmp/"g17attentionexecutor.m"),"-o",str(tmp/"attention-worker")],
            capture_output=True,text=True,timeout=30)
        if result.returncode:raise ValueError("attention worker build failed: "+result.stderr)
        data=(tmp/"attention-worker").read_bytes()
        with Path(destination).open("xb") as out:out.write(data)
        Path(destination).chmod(0o700)
    return dict(commit=revision,inputs=expected,sha256=sha(data))


def prepare(bundle,destination):
    """Prepare immutable artifacts and record the exact remaining admission facts.

    No Metal call is made. Failed preparations are retained, with a report, and
    cannot be overwritten by another attempt at the same destination.
    """
    import g17attentionadmit
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    report=dict(status="pending",gpu_dispatched=False,loader_eligible=False)
    try:
        report["files"]=snapshot(bundle,destination)
        result=subprocess.run([sys.executable,str(ROOT/"tools/g17attentionimage.py"),
            str(destination),"--rebuild"],capture_output=True,text=True,timeout=60)
        if result.returncode:raise ValueError("committed-source rebuild refused: "+(result.stderr or result.stdout)[-3000:])
        report["rebuild"]=json.loads(result.stdout)
        if report["rebuild"].get("status")!="passed":raise ValueError("rebuild did not pass")
        report["images"]=g17attentionadmit.inspect(destination)
        report["worker"]=build_worker(destination/"attention-worker")
        graph=json.loads((destination/"manifest.json").read_text())["graph"]
        (destination/"graph.json").write_text(json.dumps(graph,indent=2)+"\n")
        result=subprocess.run([str(destination/"attention-worker"),str(destination/"graph.json"),
            "--describe-schedule"],capture_output=True,text=True,timeout=10)
        if result.returncode:raise ValueError("native scheduling refused: "+result.stderr)
        report["native_schedule"]=json.loads(result.stdout)
        if report["native_schedule"].get("gpu_dispatched") is not False:
            raise ValueError("unexpected native preparation result")
        report["status"]="prepared_unvalidated"
        report["blockers"]=report["images"]["blockers"]
        return report
    except BaseException as error:
        report.update(status="refused",error=str(error));raise
    finally:
        (destination/"preparation.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle",type=Path);parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    try:
        report=prepare(args.bundle,args.destination)
        print(json.dumps(dict(status=report["status"],blockers=report["blockers"],
                              gpu_dispatched=False,loader_eligible=False),indent=2))
    except (ValueError,KeyError,TypeError,OSError) as error:parser.exit(2,f"refused: {error}\n")
