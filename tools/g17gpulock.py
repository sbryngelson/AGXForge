#!/usr/bin/env python3
"""The machine-wide GPU lock: one dispatcher at a time, and the second WAITS instead of overlapping.

Three sessions share this machine's GPU. A message protocol ("notice", then "done") kept them
apart until it did not: on 2026-09-23 a session dispatched while another's gate held the GPU,
because the check it ran printed the other gate and the same command dispatched anyway. A rule
both sides must read in time is not an interlock; this is.

ONE FILE, ~/.cache/agxforge/g17-execution.lock, the one g17commonstage.lock_gpu already used. Two
lock files would let the two families overlap each other. Taken by:

    g17commonstage.lock_gpu()        the common-worker paths (tensor runtime, encoder, ffn, ...)
    tools that opened the file       g17native, g17halfruntime, g17wordcontrol, g17pipelineprobe,
      themselves                     g17tensorstridepredict
    spike/accel/libaccel.dylib       in C, at the FIRST command buffer a process creates - so every
                                     tool that loads libaccel is covered without importing this,
                                     and a compile-only process never takes it
    g17test                          SHARED, for the whole suite, so its GPU modules run together
                                     as before while a standalone dispatcher waits for all of them

MODES. exclusive for a dispatcher; shared for the suite. flock gives the rest: shared holders
coexist, an exclusive request waits for all of them, and a shared request waits for an exclusive.

INHERITANCE, which is what makes the shared mode usable. A holder exports
AGXFORGE_GPU_LOCK_HELD=<mode>:<pid>, and any process that sees it with that pid alive skips the
acquire - the suite's GPU modules, the tensor runtime a module launches, and libaccel inside a
Python process that already holds the lock. Without it a child's exclusive request would wait on
its own parent's shared hold forever, and flock on a second descriptor of the same file conflicts
even inside one process. A dead pid is not honoured, so a stale variable in a shell cannot switch
the lock off.

WAITING, not raising. The old lock_gpu was LOCK_NB and raised; a caller now waits, printing who
holds the lock every 30 s (the holder writes its pid, mode, program and directory into the file).
AGXFORGE_GPU_LOCK_TIMEOUT=<seconds> restores a bounded wait: TimeoutError after that long.
"""
import fcntl
import os
import sys
import time
from pathlib import Path

ENV = "AGXFORGE_GPU_LOCK_HELD"
TIMEOUT_ENV = "AGXFORGE_GPU_LOCK_TIMEOUT"
REPORT_EVERY = 30.0


def path():
    return Path.home() / ".cache/agxforge/g17-execution.lock"


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def inherited():
    """The live holder this process inherits the lock from, as (mode, pid), or None."""
    v = os.environ.get(ENV, "")
    mode, _, pid = v.partition(":")
    if mode in ("exclusive", "shared") and pid.isdigit() and _alive(int(pid)):
        return mode, int(pid)
    return None


def holder_text():
    try:
        return path().read_text().strip() or "an unnamed process"
    except OSError:
        return "an unknown process"


class Held:
    """The held lock. A context manager whose exit - or close() - releases it and withdraws the
    inheritance variable; `.name` is the lock file's path, as the old file object's was."""

    def __init__(self, fh, mode, inherited_from=None):
        self._fh, self.mode, self.inherited_from = fh, mode, inherited_from
        self.name = str(path())
        self._prev = os.environ.get(ENV)
        if fh is not None:
            os.environ[ENV] = "%s:%d" % (mode, os.getpid())

    def close(self):
        if self._fh is None:
            return
        if self._prev is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = self._prev
        try:
            self._fh.close()
        finally:
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def acquire(mode="exclusive", timeout=None, report=sys.stderr):
    """Hold the GPU lock in `mode`, waiting as long as `timeout` (seconds; None = the
    AGXFORGE_GPU_LOCK_TIMEOUT variable, or forever). Returns a Held."""
    if mode not in ("exclusive", "shared"):
        raise ValueError("mode is exclusive or shared, not %r" % mode)
    up = inherited()
    if up is not None:
        return Held(None, up[0], inherited_from=up[1])
    if timeout is None and os.environ.get(TIMEOUT_ENV):
        timeout = float(os.environ[TIMEOUT_ENV])
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = p.open("a+")
    flag = fcntl.LOCK_EX if mode == "exclusive" else fcntl.LOCK_SH
    start = last = time.monotonic()
    first = True
    while True:
        try:
            fcntl.flock(fh, flag | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            now = time.monotonic()
            if timeout is not None and now - start >= timeout:
                who = holder_text()
                fh.close()
                raise TimeoutError("GPU lock not acquired in %.1f s: held by %s" % (timeout, who))
            if report is not None and (first or now - last >= REPORT_EVERY):
                print("[gpu-lock] waiting for the GPU (%s): held by %s" % (mode, holder_text()),
                      file=report, flush=True)
                first, last = False, now
            time.sleep(0.2)
        except BaseException:
            fh.close()
            raise
    # THE HOLDER NAMES ITSELF, so a waiter can say who it is waiting for. Shared holders overwrite
    # one another's line; the suite is the only shared holder, so the last one is the suite.
    fh.seek(0)
    fh.truncate()
    fh.write("%s pid %d %s in %s\n" % (mode, os.getpid(), " ".join(sys.argv)[:200], os.getcwd()))
    fh.flush()
    return Held(fh, mode)


if __name__ == "__main__":
    # `python3 tools/g17gpulock.py` says who holds the GPU, without taking it
    fh = path().open("a+") if path().parent.exists() else None
    if fh is None:
        print("free (no lock file)")
        sys.exit(0)
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("free")
    except BlockingIOError:
        print("held by %s" % holder_text())
        sys.exit(1)
