// agx3dis - decode G17 (AGX3) machine code with Apple's own MCDisassembler.
//
// Apple's toolchain refuses to disassemble AGX, but the refusal is narrower than it looks.
// air-objdump registers agx1, agx2 and agx3 as targets, and the errors it returns separate
// two different failures:
//
//     --triple=agx1  ->  no disassembler for target agx1
//     --triple=agx2  ->  no instruction printer for target agx2
//     --triple=agx3  ->  no instruction printer for target agx3
//
// llvm-objdump builds the MCDisassembler before the MCInstPrinter, so reaching the printer
// error proves an AGX3 decoder exists and ran. It is not confined to that binary either: the
// shared-cache libLLVM under GPUCompiler.framework exports LLVMInitializeAGX3TargetInfo,
// LLVMInitializeAGX3TargetMC and LLVMInitializeAGX3Disassembler, all reachable by dlsym.
//
// Calling those three still leaves LLVMCreateDisasm returning null. Dumping llvm::Target for
// agx3 and resolving its fields against the cache symbol table shows why: field +0x80,
// MCDisassemblerCtorFn, is filled in by LLVMInitializeAGX3Disassembler, and field +0x88,
// MCInstPrinterCtorFn, is the single null. Apple withheld the printer and nothing else.
//
// So this installs its own printer factory at +0x88. The object it returns needs no real
// behaviour: LLVMDisasmInstruction calls printInst after the decode has already happened, and
// printInst is handed the decoded MCInst. Reading opcode and operands off that MCInst is the
// whole tool. What comes back is Apple's decode, not a model of it: exact instruction length,
// a stable opcode id, and the operand list the decoder built.
//
// Two limits, both from Apple's build rather than from this approach. Instruction names are
// compiled out (MCInstrInfo::InstrNameData holds "0", "1", "2", ...), so opcodes are numeric
// ids out of 17796, not mnemonics. And the meaning of an operand value is not established
// here: kinds come from LLVM's MCOperand enum, but the register numbering they index has not
// been checked against anything, so registers print as their raw MCRegister number.
//
// Build:  clang -O2 -o tools/agx3dis tools/agx3dis.c        (includes tools/agx3remap.h)
// Usage:  tools/agx3dis <file> <byte-offset> <length> [--pc N]
// Output: one line per instruction, "offset size opcode operand..."
//
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LIBLLVM "/System/Library/PrivateFrameworks/GPUCompiler.framework" \
                "/Versions/32023/Libraries/libLLVM.dylib"

// llvm::Target field indices, resolved by dumping the struct and matching each pointer
// against the shared-cache symbol table. Verified by the fact that index 16 holds the
// function LLVMInitializeAGX3Disassembler registers.
#define TARGET_MCDISASSEMBLER_CTOR 16
#define TARGET_MCINSTPRINTER_CTOR  17

// MCInst layout, read out of a live decode rather than assumed from headers:
//   +0x00 u32   Opcode
//   +0x08 ptr   SMLoc
//   +0x10 ptr   SmallVector BeginX (points at the inline storage at +0x20)
//   +0x18 u32   Size          +0x1c u32 Capacity (8, matching SmallVector<MCOperand, 8>)
// and each MCOperand is 16 bytes, kind in the first byte, value at +8.
#define MCINST_OPS_PTR   0x10
#define MCINST_OPS_SIZE  0x18
#define MCOPERAND_STRIDE 16
#define MCOPERAND_VALUE  8

typedef void   (*VoidFn)(void);
typedef int    (*GetTargetFromTriple)(const char *, void **, char **);
typedef void  *(*CreateDisasmCPU)(const char *, const char *, void *, int, void *, void *);
typedef size_t (*DisasmInstruction)(void *, uint8_t *, uint64_t, uint64_t, char *, size_t);

// The MCInst lives in LLVMDisasmInstruction's frame and its operand storage is freed when
// that returns, so everything wanted is copied out here, inside the call, rather than read
// back afterwards. Reading it afterwards works only while the operands fit SmallVector's
// inline capacity of 8; past that the vector is on the heap and the buffer is gone.
#define MAX_OPERANDS 64
static unsigned g_opcode, g_count;
static uint8_t  g_kind[MAX_OPERANDS];
static int64_t  g_value[MAX_OPERANDS];
static int      g_captured;
#include "agx3remap.h"

static long stub(void *a, void *b, void *c, void *d, void *e, void *f) { return 0; }

static long capture_printInst(void *self, const void *mi, uint64_t addr,
                              void *annot, void *sti, void *os) {
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

// Slot 4 is printInst in this build's MCInstPrinter vtable, found by pointing every slot at a
// recording stub and seeing which one the decode path calls with an MCInst in argument 1.
static void *vtable[32];

static void *inst_printer_factory(void *a, void *b, void *c, void *d, void *e) {
    void **obj = calloc(1, 1024);   // oversized: setPrintImmHex and friends write members
    obj[0] = vtable;
    return obj;
}

// AN EXPRESSION OPERAND PRINTS AS THE ADDRESS OF THE MCExpr THE DECODER JUST ALLOCATED, which is
// noise: it changes on every decode. 682 opcodes declare a register operand that never prints as a
// register in any encoding this project can construct, and every probe against them has been
// reading allocator addresses.
//
// If the MCExpr is a constant, its VALUE is there to be read. LLVM's MCExpr has no vtable - it
// discriminates on an ExprKind field - so the layout is a small header followed by the subclass's
// data, and --expr dumps the first bytes of the object so the offsets can be found by
// self-validation rather than assumed from a header this build may not match.
static int g_dump_expr = 0;

// The object is an MCExpr with no vtable, discriminating on an ExprKind word at offset 0:
// 1 Constant, 2 SymbolRef, 3 Unary, 4 Binary in this build (validated below by what the fields
// do). A Binary holds its operator at +16, its LHS pointer at +24 and its RHS at +32, and the
// LHS is followed one level so the leaf can be named rather than guessed at.
static void dump_expr(const void *p, int depth) {
    if (!p) { printf("(null)"); return; }
    const uint8_t *b = (const uint8_t *)p;
    unsigned kind = *(const unsigned *)b;
    if (kind == 4 && depth < 3) {
        unsigned long long op = *(const unsigned long long *)(b + 16);
        const void *lhs = *(const void **)(b + 24);
        unsigned long long rhs = *(const unsigned long long *)(b + 32);
        printf("bin(op%llu,", op);
        dump_expr(lhs, depth + 1);
        printf(",%llu)", rhs);
        return;
    }
    if (kind == 1) { printf("const(%lld)", *(const long long *)(b + 16)); return; }
    printf("kind%u[", kind);
    for (int i = 8; i < 40; i++) printf("%02x", b[i]);
    printf("]");
}

static const char *operand_kind(unsigned k) {
    switch (k) {
    case 1: return "reg";
    case 2: return "imm";
    case 3: return "sfp";
    case 4: return "dfp";
    case 5: return "expr";
    case 6: return "inst";
    default: return NULL;      // including 0, kInvalid
    }
}

// ONE PROGRAM, DECODED WITH AN ALREADY-BUILT CONTEXT. Lifted verbatim out of main so that a
// batch can reuse the dlopen and the disassembler: the corpus walk called this binary 23,028
// times, once per program, and each call paid a process spawn plus a dlopen of libLLVM before
// decoding anything. That setup, not the decoding, was ~90% of the 87.8s the atomic regression
// case spent. Nothing about the decode changed; only how often the setup runs.
static int decode_one(void *dc, DisasmInstruction disasm,
                      const char *path, long off, long len, long pc, long stride) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(n);
    if (fread(buf, 1, n, f) != (size_t)n) { fprintf(stderr, "agx3dis: short read\n"); free(buf); fclose(f); return 1; }
    fclose(f);
    if (off < 0 || len < 0 || off + len > n) { fprintf(stderr, "agx3dis: range outside file\n"); free(buf); return 1; }

    char text[512];
    int rc = 0;
    for (long p = off; p < off + len; ) {
        g_captured = 0; g_count = 0; g_opcode = 0;
        size_t size = disasm(dc, buf + p, off + len - p, pc + (p - off), text, sizeof text);
        if (size == 0) {
            printf("%08lx  bad\n", pc + (p - off));
            if (!stride) { rc = 3; break; }
            p += stride;
            continue;
        }
        if (g_captured > 0) agx3_remap_capture();
        printf("%08lx %2zu %6u", pc + (p - off), size, g_opcode);
        if (g_captured < 0) printf(" operands-unreadable");
        for (unsigned k = 0; k < g_count; k++) {
            const char *name = operand_kind(g_kind[k]);
            if (name && g_kind[k] == 5 && g_dump_expr) {
                printf(" expr:");
                dump_expr((const void *)(uintptr_t)g_value[k], 0);
            }
            else if (name) printf(" %s:%lld", name, (long long)g_value[k]);
            else      printf(" unknown:0x%02x", g_kind[k]);
        }
        printf("\n");
        p += stride ? stride : (long)size;
    }
    free(buf);
    return rc;
}

int main(int argc, char **argv) {
    // --batch MANIFEST: one "path off len pc" per line, decoded in order with one setup. Each
    // entry's output is preceded by "=== <line-number> <rc>" so a caller can split the stream
    // and see a per-entry return code; the single-file CLI below is byte-for-byte unchanged.
    const char *batch = NULL;
    for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], "--batch")) batch = argv[i + 1];
    if (!batch && argc < 4) {
        fprintf(stderr, "usage: agx3dis <file> <byte-offset> <length> [--pc N]\n");
        fprintf(stderr, "       agx3dis --batch <manifest>\n");
        return 2;
    }
    for (int i = 0; i < 32; i++) vtable[i] = (void *)stub;
    vtable[4] = (void *)capture_printInst;

    void *llvm = dlopen(LIBLLVM, RTLD_NOW | RTLD_GLOBAL);
    if (!llvm) { fprintf(stderr, "agx3dis: dlopen: %s\n", dlerror()); return 1; }

    const char *init[] = { "LLVMInitializeAGX3TargetInfo",
                           "LLVMInitializeAGX3TargetMC",
                           "LLVMInitializeAGX3Disassembler", NULL };
    for (int i = 0; init[i]; i++) {
        VoidFn fn = (VoidFn)dlsym(llvm, init[i]);
        if (!fn) { fprintf(stderr, "agx3dis: %s not exported\n", init[i]); return 1; }
        fn();
    }

    void *target = NULL; char *err = NULL;
    ((GetTargetFromTriple)dlsym(llvm, "LLVMGetTargetFromTriple"))("agx3---macho", &target, &err);
    if (!target) { fprintf(stderr, "agx3dis: no agx3 target: %s\n", err ? err : ""); return 1; }
    uint64_t *fields = (uint64_t *)target;
    if (!fields[TARGET_MCDISASSEMBLER_CTOR]) {
        fprintf(stderr, "agx3dis: MCDisassemblerCtorFn is null, field layout has moved\n");
        return 1;
    }
    fields[TARGET_MCINSTPRINTER_CTOR] = (uint64_t)(uintptr_t)inst_printer_factory;

    void *dc = ((CreateDisasmCPU)dlsym(llvm, "LLVMCreateDisasmCPU"))
                   ("agx3---macho", "g17s", NULL, 0, NULL, NULL);
    if (!dc) { fprintf(stderr, "agx3dis: LLVMCreateDisasm failed\n"); return 1; }
    DisasmInstruction disasm = (DisasmInstruction)dlsym(llvm, "LLVMDisasmInstruction");

    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--expr")) g_dump_expr = 1;

    // APPLE'S NUMBERING IS NOT STABLE ACROSS OS BUILDS (tools/agx3remap.h). --raw prints the running
    // build's own ids, which only tools/g17renumber.py wants.
    int raw = getenv("AGX3_RAW") != NULL;
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--raw")) raw = 1;
    if (agx3_remap_setup(dc, disasm, raw) < 0) { fprintf(stderr, "agx3dis: %s\n", g_remap_why); return 4; }

    if (batch) {
        FILE *m = fopen(batch, "r");
        if (!m) { perror(batch); return 1; }
        char line[4096];
        long lineno = 0;
        while (fgets(line, sizeof line, m)) {
            char path[3072]; long off, len, pc;
            lineno++;
            if (sscanf(line, "%3071s %ld %ld %ld", path, &off, &len, &pc) != 4) continue;
            printf("=== %ld\n", lineno);
            int rc = decode_one(dc, disasm, path, off, len, pc, 0);
            printf("=== end %ld %d\n", lineno, rc);
        }
        fclose(m);
        return 0;
    }

    long off = strtol(argv[2], NULL, 0);
    long len = strtol(argv[3], NULL, 0);
    long pc  = 0;
    long stride = 0;
    for (int i = 4; i < argc - 1; i++) if (!strcmp(argv[i], "--pc")) pc = strtol(argv[i + 1], NULL, 0);
    // SLOTTED MODE. A mutation sweep lays each candidate in a fixed-width slot, so the next
    // start is known a priori and a rejected candidate cannot desynchronise anything. Only in
    // that mode is it safe to report `bad` and keep going; the default walk still stops, because
    // there the next start is exactly what a bad decode has destroyed.
    for (int i = 4; i < argc - 1; i++) if (!strcmp(argv[i], "--stride")) stride = strtol(argv[i + 1], NULL, 0);
    return decode_one(dc, disasm, argv[1], off, len, pc, stride);
}
