"""F1: does the OS compiler take air.convert with fp4/fp6 tokens, the way it took fp8?

Section 137 recorded fp4/fp6 as blocked because the PACKED (pn) tokens crash AGCLLVMAirBuiltins::buildConvert.
But fp8 was unblocked by abandoning air.pack for air.convert (section 138), so the same move is worth trying
before leaving F1 as blocked. This is that test, byte-for-byte the fp8 probe with the format tokens swapped.
Compile only, no GPU.

Original fp8 docstring: does the OS compiler take air.convert with fp8 tokens (both directions), exposing the hardware unpack (op17642) and any pack instruction outside the MMA?  Vector forms air.convert.f.<dst>.f.<src>(vec) over
{v8f32, v8f16, v8bf16} x {v8f8e4m3fn, v8f8e5m2} and back, plus scalar forms.  Compile only; each failure is a compile-service crash whose site is read from the report (scaled_crash_sites.reports_since).
    python3 fp8_convert_probe.py > fp8_convert_probe.log"""
import sys, collections, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import regfam as F, build
from agxforge.g17 import model
import scaled_crash_sites as SCS
LL = {'v8f32': '<8 x float>', 'v8f16': '<8 x half>', 'v8bf16': '<8 x bfloat>', 'v8f8e4m3fn': '<8 x i8>', 'v8f8e5m2': '<8 x i8>', 'v8f4e2m1': '<8 x i8>', 'v8f6e2m3': '<8 x i8>', 'v8f6e3m2': '<8 x i8>', 'f32': 'float', 'f16': 'half', 'bf16': 'bfloat', 'f8e4m3fn': 'i8', 'f8e5m2': 'i8', 'f4e2m1': 'i8', 'f6e2m3': 'i8', 'f6e3m2': 'i8'}
def kern(tag, dst, src):
    nm = f'air.convert.f.{dst}.f.{src}'; dt, st = LL[dst], LL[src]; ldt = 'i32' if False else st
    L = ['entry:', '  %lane = zext i32 %4 to i64', f'  %Av = bitcast half addrspace(1)* %0 to {st} addrspace(1)*', f'  %Ov = bitcast float addrspace(1)* %2 to {dt} addrspace(1)*',
         f'  %ap = getelementptr inbounds {st}, {st} addrspace(1)* %Av, i64 %lane', f'  %a = load {st}, {st} addrspace(1)* %ap, align 8', f'  %d = call {dt} @{nm}({st} %a)',
         '  %oi = add i64 %lane, 1024', f'  %op = getelementptr inbounds {dt}, {dt} addrspace(1)* %Ov, i64 %oi', f'  store {dt} %d, {dt} addrspace(1)* %op, align 8', '  ret void', '}']
    src_ = F.assemble_module(L).replace('declare void @air.wg.barrier', f'declare {dt} @{nm}({st}) local_unnamed_addr #1\ndeclare void @air.wg.barrier', 1); (HERE / (tag + '.ll')).write_text(src_)
    t0 = time.time() - 1; r = build.build(tag); st_ = r.get('status'); time.sleep(3 if st_ != 'ok' else 0)
    if st_ == 'ok':
        ins = list(model.decode(open(HERE / tag / 'code.bin', 'rb').read())); c = collections.Counter(i.opcode.id for i in ins); print(f'{nm:52s} ok | {len(ins)} instr | {dict(sorted((k, v) for k, v in c.items() if k not in (13483, 684)))}', flush=True)
    else:
        sites = SCS.reports_since(t0); print(f'{nm:52s} {st_} | {"; ".join(k + " @ " + top[0] for k, top in sites)[:170]}', flush=True)
if __name__ == '__main__':
    for fmt in ('f4e2m1', 'f6e2m3', 'f6e3m2'):
        kern(f'fcv_f32_{fmt}_s', 'f32', fmt); kern(f'fcv_{fmt}_f32_s', fmt, 'f32')
    for fmt in ('f8e4m3fn',):   # CONTROL: the scalar fp8 path, known to work
        kern(f'fcv_f32_{fmt}_s', 'f32', fmt); kern(f'fcv_{fmt}_f32_s', fmt, 'f32')
