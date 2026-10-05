# energy, measured the way this repo's three failed attempts taught: integrate the MEAN
# (a median lands on the idle level of a bimodal window), keep the window FILLED with
# work (a sampler outliving the work dilutes the mean), and never trust an unflushed
# sampler. powermetrics streams per interval once its ~0.9 s init is paid, so one
# long-running invocation read live gives full coverage; the init means blocks under
# about two seconds cannot be metered honestly, and the meter refuses them by name.
from __future__ import annotations
import os, re, subprocess, threading, time

RAILS = ("ANE Power", "GPU Power", "CPU Power")
_LINE = re.compile(r"^(ANE|GPU|CPU) Power: (\d+) mW")

class Meter:
  """with pool.meter() as m: ...; then m.mj (package millijoules), m.rails (the split),
  or None with m.why saying what was wrong. pass sample= for a fake in tests."""
  def __init__(self, sample=None):
    self._fake, self._stop, self._vals = sample, threading.Event(), {k: [] for k in RAILS}

  def _reader(self, proc):
    for line in proc.stdout:
      if (m := _LINE.match(line)): self._vals[f"{m.group(1)} Power"].append(float(m.group(2)))
      if self._stop.is_set(): break

  def _fake_loop(self):
    while not self._stop.is_set():
      if (s := self._fake()) is not None:
        for k, v in s.items(): self._vals[k].append(v)
      time.sleep(0.01)

  def __enter__(self):
    self.t0, self._proc = time.perf_counter(), None
    if self._fake or os.environ.get("AGXFORGE_NO_POWER"):
      target = self._fake_loop if self._fake else self._stop.wait
    else:
      try:
        self._proc = subprocess.Popen(
          ["sudo", "-n", "powermetrics", "--samplers", "ane_power,gpu_power,cpu_power",
           "-i", "200", "-n", "100000"], stdout=subprocess.PIPE,
          stderr=subprocess.DEVNULL, text=True)
        target = lambda: self._reader(self._proc)
      except Exception:
        target = self._stop.wait
    self._t = threading.Thread(target=target, daemon=True); self._t.start()
    return self

  def __exit__(self, *a):
    self.secs = time.perf_counter() - self.t0
    self._stop.set()
    if self._proc: self._proc.terminate()
    self._t.join(timeout=6)
    n = len(self._vals["ANE Power"])
    if n < 3:
      self.mj = self.rails = None
      self.why = (f"window too short or rails unreadable: {n} samples in {self.secs:.2f}s; "
                  "the sampler needs ~0.9s to start, so meter blocks of 2s or more")
      return
    self.why = None
    mean = {k: sum(v) / len(v) for k, v in self._vals.items() if v}
    self.rails = {k: round(mean.get(k, 0.0) * self.secs, 1) for k in RAILS}
    self.mj = round(sum(self.rails.values()), 1)
