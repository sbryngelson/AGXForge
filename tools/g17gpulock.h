// The machine-wide GPU lock for native code: one dispatcher at a time, and the second WAITS.
// Same file, mode and inheritance rule as tools/g17gpulock.py (read that docstring for why):
// ~/.cache/agxforge/g17-execution.lock, EXCLUSIVE, skipped when AGXFORGE_GPU_LOCK_HELD names a live
// holder (a Python parent that already holds it, or this process's own Python lock), waiting with
// a report every 30 s, bounded by AGXFORGE_GPU_LOCK_TIMEOUT. Taken at the FIRST command buffer, so a
// process that only compiles never takes it, and held until the process exits.
//
// Every native program that dispatches creates its command buffers through g17_gpu_cb(queue), so
// the lock cannot be skipped by a new entry point: tools/test_g17gpulock checks that no
// `[queue commandBuffer]` survives outside this header.
#pragma once
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
static int g17_lock_fd = -1;
static bool g17_gpu_lock_inherited(void) {
  const char *h = getenv("AGXFORGE_GPU_LOCK_HELD");
  if (!h) return false;
  const char *c = strchr(h, ':');
  if (!c || (strncmp(h, "exclusive:", 10) && strncmp(h, "shared:", 7))) return false;
  long pid = strtol(c + 1, NULL, 10);
  return pid > 0 && (kill((pid_t)pid, 0) == 0 || errno == EPERM);
}
static void g17_gpu_lock_holder(int fd, char *out, size_t n) {
  ssize_t k = pread(fd, out, n - 1, 0);
  out[k > 0 ? k : 0] = 0;
  for (char *e = out + strlen(out); e > out && (e[-1] == '\n' || e[-1] == ' '); ) *--e = 0;
  if (!*out) snprintf(out, n, "an unnamed process");
}
static int g17_gpu_lock(void) {
  if (g17_lock_fd >= 0 || g17_gpu_lock_inherited()) return 0;
  const char *home = getenv("HOME");
  char path[1024];
  snprintf(path, sizeof path, "%s/.cache", home ? home : "/tmp"); mkdir(path, 0755);
  snprintf(path, sizeof path, "%s/.cache/agxforge", home ? home : "/tmp"); mkdir(path, 0755);
  snprintf(path, sizeof path, "%s/.cache/agxforge/g17-execution.lock", home ? home : "/tmp");
  int fd = open(path, O_RDWR | O_CREAT, 0644);
  if (fd < 0) { fprintf(stderr, "[gpu-lock] cannot open %s: %s\n", path, strerror(errno)); return -1; }
  const char *t = getenv("AGXFORGE_GPU_LOCK_TIMEOUT");
  double timeout = t ? atof(t) : -1.0;
  time_t start = time(NULL), last = 0;
  char who[512];
  while (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    if (errno != EWOULDBLOCK) { fprintf(stderr, "[gpu-lock] flock: %s\n", strerror(errno)); close(fd); return -1; }
    time_t now = time(NULL);
    g17_gpu_lock_holder(fd, who, sizeof who);
    if (timeout >= 0 && difftime(now, start) >= timeout) {
      fprintf(stderr, "[gpu-lock] not acquired in %.1f s: held by %s\n", timeout, who);
      close(fd); return -2;
    }
    if (!last || now - last >= 30) { fprintf(stderr, "[gpu-lock] waiting for the GPU (exclusive): held by %s\n", who); last = now; }
    usleep(200000);
  }
  char line[600], cwd[400] = "?";
  if (!getcwd(cwd, sizeof cwd)) snprintf(cwd, sizeof cwd, "?");
  int m = snprintf(line, sizeof line, "exclusive pid %d %s (native) in %s\n", getpid(), getprogname(), cwd);
  if (ftruncate(fd, 0) == 0 && m > 0) { ssize_t w = pwrite(fd, line, (size_t)m, 0); (void)w; }
  char env[48]; snprintf(env, sizeof env, "exclusive:%d", getpid());
  setenv("AGXFORGE_GPU_LOCK_HELD", env, 1);
  g17_lock_fd = fd;
  return 0;
}
// A process that could not take the lock in time exits rather than dispatch unlocked.
static inline id<MTLCommandBuffer> g17_gpu_cb(id<MTLCommandQueue> q) {
  if (g17_gpu_lock() != 0) exit(75);
  return [q commandBuffer];
}
