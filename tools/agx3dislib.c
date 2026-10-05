// agx3dislib - tools/agx3dis as a LIBRARY, so a caller decodes G17 bytes with Apple's own MCDisassembler without
// starting a process (MM 25.144.6).
//
// cc's compile-time release guard needs the reference decode of every program it compiles, and a compile must
// not start an external process (the delivery paths audit it: g17programs, g17scan, g17layernorm, the build
// audit). tools/agx3dis is a subprocess; this is the same decoder - the same dlopen of GPUCompiler's libLLVM, the
// same printer factory installed at llvm::Target +0x88, the same MCInst capture - behind a function, with the
// CLI's output format written into a caller buffer instead of stdout. See tools/agx3dis.c for how the decoder
// is reached and what its output means; nothing about the decode differs.
//
// Build:  clang -O2 -dynamiclib -o tools/libagx3dis.dylib tools/agx3dislib.c     (make native-tools)
// API:    int  agx3dis_lib_init(void)                       0 ok, else a failure code (dlopen/target/ctor)
//         long agx3dis_lib_decode(buf, n, off, len, pc, out, cap)
//              decodes buf[off, off+len) like `agx3dis <file> off len --pc pc`; writes the same lines into out;
//              returns bytes written, or -(bytes needed) if cap is too small, or -1 on a failure before output
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#define LIBLLVM "/System/Library/PrivateFrameworks/GPUCompiler.framework" \
                "/Versions/32023/Libraries/libLLVM.dylib"
#define TARGET_MCDISASSEMBLER_CTOR 16
#define TARGET_MCINSTPRINTER_CTOR  17
#define MCINST_OPS_PTR   0x10
#define MCINST_OPS_SIZE  0x18
#define MCOPERAND_STRIDE 16
#define MCOPERAND_VALUE  8

typedef void   (*VoidFn)(void);
typedef int    (*GetTargetFromTriple)(const char *, void **, char **);
typedef void  *(*CreateDisasmCPU)(const char *, const char *, void *, int, void *, void *);
typedef size_t (*DisasmInstruction)(void *, uint8_t *, uint64_t, uint64_t, char *, size_t);

#define MAX_OPERANDS 64
static unsigned g_opcode, g_count;
static uint8_t  g_kind[MAX_OPERANDS];
static int64_t  g_value[MAX_OPERANDS];
static int      g_captured;
static void    *g_dc;
static DisasmInstruction g_disasm;
#include "agx3remap.h"

static long stub(void *a, void *b, void *c, void *d, void *e, void *f) { return 0; }

static long capture_printInst(void *self, const void *mi, uint64_t addr, void *annot, void *sti, void *os) {
    const uint8_t *inst = (const uint8_t *)mi;
    g_opcode = *(const unsigned *)inst;
    g_count  = *(const unsigned *)(inst + MCINST_OPS_SIZE);
    const uint8_t *ops = *(const uint8_t **)(inst + MCINST_OPS_PTR);
    if (!ops || g_count > MAX_OPERANDS) { g_count = 0; g_captured = ops ? -1 : 0; return 0; }
    for (unsigned k = 0; k < g_count; k++) {
        const uint8_t *op = ops + MCOPERAND_STRIDE * k;
        g_kind[k]  = op[0];
        g_value[k] = *(const int64_t *)(op + MCOPERAND_VALUE);
    }
    g_captured = 1;
    return 0;
}

static void *vtable[32];

static void *inst_printer_factory(void *a, void *b, void *c, void *d, void *e) {
    void **obj = calloc(1, 1024);
    obj[0] = vtable;
    return obj;
}

static const char *operand_kind(unsigned k) {
    switch (k) {
    case 1: return "reg";
    case 2: return "imm";
    case 3: return "sfp";
    case 4: return "dfp";
    case 5: return "expr";
    case 6: return "inst";
    default: return NULL;
    }
}

int agx3dis_lib_init(void) {
    if (g_dc) return 0;
    for (int i = 0; i < 32; i++) vtable[i] = (void *)stub;
    vtable[4] = (void *)capture_printInst;
    void *llvm = dlopen(LIBLLVM, RTLD_NOW | RTLD_LOCAL);
    if (!llvm) return 1;
    const char *init[] = { "LLVMInitializeAGX3TargetInfo", "LLVMInitializeAGX3TargetMC",
                           "LLVMInitializeAGX3Disassembler", NULL };
    for (int i = 0; init[i]; i++) {
        VoidFn fn = (VoidFn)dlsym(llvm, init[i]);
        if (!fn) return 2;
        fn();
    }
    void *target = NULL; char *err = NULL;
    GetTargetFromTriple gt = (GetTargetFromTriple)dlsym(llvm, "LLVMGetTargetFromTriple");
    if (!gt) return 3;
    gt("agx3---macho", &target, &err);
    if (!target) return 3;
    uint64_t *fields = (uint64_t *)target;
    if (!fields[TARGET_MCDISASSEMBLER_CTOR]) return 4;
    fields[TARGET_MCINSTPRINTER_CTOR] = (uint64_t)(uintptr_t)inst_printer_factory;
    CreateDisasmCPU cd = (CreateDisasmCPU)dlsym(llvm, "LLVMCreateDisasmCPU");
    if (!cd) return 5;
    g_dc = cd("agx3---macho", "g17s", NULL, 0, NULL, NULL);
    if (!g_dc) return 5;
    g_disasm = (DisasmInstruction)dlsym(llvm, "LLVMDisasmInstruction");
    if (!g_disasm) return 6;
    // the same renumbering guard as tools/agx3dis (tools/agx3remap.h); 7 = an unknown decoder build
    if (agx3_remap_setup(g_dc, g_disasm, getenv("AGX3_RAW") != NULL) < 0) { g_dc = NULL; return 7; }
    return 0;
}

// why init returned 7: the canary that did not decode under either known numbering
const char *agx3dis_lib_why(void) { return g_remap_why; }

// append formatted text to out; tracks the total needed even past cap
static void emit(char *out, long cap, long *used, const char *fmt, ...) {
    char tmp[128];
    va_list ap; va_start(ap, fmt);
    int k = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (k < 0) return;
    if (*used + k < cap) memcpy(out + *used, tmp, (size_t)k);
    *used += k;
}

long agx3dis_lib_decode(const uint8_t *buf, long n, long off, long len, long pc, char *out, long cap) {
    if (!g_dc && agx3dis_lib_init() != 0) return -1;
    if (off < 0 || len < 0 || off + len > n) return -1;
    char text[512];
    long used = 0;
    for (long p = off; p < off + len; ) {
        g_captured = 0; g_count = 0; g_opcode = 0;
        size_t size = g_disasm(g_dc, (uint8_t *)buf + p, off + len - p, pc + (p - off), text, sizeof text);
        if (size == 0) { emit(out, cap, &used, "%08lx  bad\n", pc + (p - off)); break; }
        if (g_captured > 0) agx3_remap_capture();
        emit(out, cap, &used, "%08lx %2zu %6u", pc + (p - off), size, g_opcode);
        if (g_captured < 0) emit(out, cap, &used, " operands-unreadable");
        for (unsigned k = 0; k < g_count; k++) {
            const char *name = operand_kind(g_kind[k]);
            if (name) emit(out, cap, &used, " %s:%lld", name, (long long)g_value[k]);
            else      emit(out, cap, &used, " unknown:0x%02x", g_kind[k]);
        }
        emit(out, cap, &used, "\n");
        p += (long)size;
    }
    if (used < cap) { out[used] = 0; return used; }
    return -(used + 1);
}
