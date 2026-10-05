// agx3meta - dump AGX3's own instruction and register metadata out of Apple's libLLVM.
//
// Companion to tools/agx3dis.c, which explains how the AGX3 target is reached. Once
// llvm::Target is in hand, its constructor slots hand over the generated TableGen tables that
// describe the ISA, so an opcode id stops being opaque: it has an operand count, a scheduling
// class, a target flag word, and a register class per operand.
//
// Every layout below was recovered by self-validation rather than assumed from headers, because
// this is Apple's LLVM and its struct layouts are not guaranteed to match any public release:
//
//   MCInstrDesc     +24 implicit uses, +32 implicit defs, both null-terminated MCPhysReg
//                   arrays; stride 48, found by requiring the first u16 to equal the array index for
//                   all 17796 entries. Only stride 48 satisfies that.
//   MCOperandInfo   stride 6, confirmed against the decoder: over 555785 operand slots in the
//                   corpus, RegClass >= 0 predicts a register operand and RegClass == -1
//                   predicts an immediate, agreeing 96.1% of the time, and every disagreement
//                   is a register-class slot that decoded as MCExpr rather than MCReg, which is
//                   what a relocation-bearing operand looks like.
//   MCRegisterClass stride 32, found by requiring the u16 at +24 to equal the array index for
//                   all 500 entries.
//
// Instruction NAMES are not here and are not anywhere. Apple built this LLVM with instruction
// names disabled, so MCInstrInfo's name table holds "0", "1", "2" and so on. Register and
// register-class names are NOT affected by that switch and did survive, which is why an operand
// can be typed as GPR32tup2 while the instruction using it stays a number.
//
// Build:  clang -O2 -o tools/agx3meta tools/agx3meta.c     (includes tools/agx3renumber.h)
// Usage:  tools/agx3meta instrs      one line per opcode: id nops ndefs sched tsflags operands
//         tools/agx3meta classes     one line per register class: id name bits nregs
//         tools/agx3meta regs        one line per register: id name
//         tools/agx3meta regmodel    id name | direct sub-registers - the aliasing model
//
// The atomic locations are the LEAF registers, of which there are exactly 410, matching
// NumRegUnits. A 32-bit R3 covers the leaves R3L and R3H, so a def of R3H kills one of R3's two
// leaves and leaves the other live. Any dataflow over this ISA has to work in leaves rather than
// in register names, because the corpus really does mix widths: R3H, R3L, R2L and R4H all appear
// in a single driver shader. tools/g17regs.py takes the transitive closure.
//
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define LIBLLVM "/System/Library/PrivateFrameworks/GPUCompiler.framework" \
                "/Versions/32023/Libraries/libLLVM.dylib"
#define TARGET_MCINSTRINFO_CTOR 8
#define TARGET_MCREGINFO_CTOR   10
#define INSTRDESC_STRIDE  48
// Ceiling for an implicit-operand register id. The register file this target
// reports is well under this; anything above it means the pointer is not an
// MCPhysReg array and the entry is printed as a hole instead of a guess.
#define IMPLICIT_REG_LIMIT 4096
#define OPERANDINFO_STRIDE 6
#define REGCLASS_STRIDE   32
#define REGDESC_STRIDE    24
#define REGDESC_SUBREGS    4
#define REGDESC_REGUNITS  16

typedef void  (*VoidFn)(void);
typedef int   (*GetTargetFromTriple)(const char *, void **, char **);
typedef void *(*Ctor0)(void);
typedef void *(*Ctor1)(void *);
typedef void *(*TripleCtor)(void *, const void *);

// APPLE'S NUMBERING IS NOT STABLE ACROSS OS BUILDS (tools/agx3remap.h, tools/g17renumber.py). On the
// 26A434 build every table here is printed in this repository's numbering: opcodes through the generated
// opcode table, registers and register classes by name, scheduling classes through the mapped opcodes.
// Rows are emitted in old-id order; a row with no old id is printed as AGX3_UNMAPPED_BASE + its new id,
// after the rest. The build is recognised by its table sizes (the original 17,796 opcodes / 3,567
// registers, or the 26A434 build's), anything else is refused. AGX3_RAW=1 prints the running build as is.
#include "agx3renumber.h"
static int g_translate = 0;
static int meta_mode(unsigned count, unsigned old_count, unsigned new_count, const char *what) {
    if (getenv("AGX3_RAW") || count == old_count) return 0;
    if (count == new_count) return 1;
    fprintf(stderr, "agx3meta: %u %s matches neither the original build (%u) nor 26A434 (%u); "
                    "tools/g17renumber.py has no table for this decoder\n", count, what, old_count, new_count);
    return -1;
}
static long tr(const int *table, unsigned n, long v) {
    if (!g_translate || v < 0) return v;
    return (unsigned long)v < n ? table[v] : AGX3_UNMAPPED_BASE + v;
}
#define TR_OP(v)    tr(AGX3_OP_NEW2OLD, AGX3_NEW_OPCODES, (v))
#define TR_REG(v)   ((v) == 0 ? 0 : tr(AGX3_REG_NEW2OLD, AGX3_NEW_REGISTERS, (v)))
#define TR_CLASS(v) tr(AGX3_CLASS_NEW2OLD, AGX3_NEW_CLASSES, (v))
#define TR_SCHED(v) tr(AGX3_SCHED_NEW2OLD, AGX3_NEW_SCHED, (v))

// Rows are formatted into per-row buffers, keyed by their translated id, then printed in id order.
struct row { long id; char *text; };
static struct row *g_rows; static unsigned g_nrows, g_caprows;
static FILE *row_begin(long id, char **buf, size_t *len) {
    if (g_nrows == g_caprows) { g_caprows = g_caprows ? 2 * g_caprows : 1024; g_rows = realloc(g_rows, g_caprows * sizeof *g_rows); }
    g_rows[g_nrows].id = id;
    return open_memstream(buf, len);
}
static void row_end(FILE *f, char **buf) { fclose(f); g_rows[g_nrows++].text = *buf; }
static int row_cmp(const void *a, const void *b) {
    long x = ((const struct row *)a)->id, y = ((const struct row *)b)->id;
    return x < y ? -1 : x > y;
}
static void rows_flush(void) {
    qsort(g_rows, g_nrows, sizeof *g_rows, row_cmp);
    for (unsigned i = 0; i < g_nrows; i++) { fputs(g_rows[i].text, stdout); free(g_rows[i].text); }
    g_nrows = 0;
}

static void *open_target(void **llvm_out) {
    void *llvm = dlopen(LIBLLVM, RTLD_NOW | RTLD_GLOBAL);
    if (!llvm) { fprintf(stderr, "agx3meta: dlopen: %s\n", dlerror()); return NULL; }
    const char *init[] = { "LLVMInitializeAGX3TargetInfo", "LLVMInitializeAGX3TargetMC",
                           "LLVMInitializeAGX3Disassembler", NULL };
    for (int i = 0; init[i]; i++) {
        VoidFn fn = (VoidFn)dlsym(llvm, init[i]);
        if (!fn) { fprintf(stderr, "agx3meta: %s not exported\n", init[i]); return NULL; }
        fn();
    }
    void *target = NULL; char *err = NULL;
    ((GetTargetFromTriple)dlsym(llvm, "LLVMGetTargetFromTriple"))("agx3---macho", &target, &err);
    if (!target) fprintf(stderr, "agx3meta: no agx3 target: %s\n", err ? err : "");
    *llvm_out = llvm;
    return target;
}

// MCRegInfoCtorFn takes a const Triple &, so one has to be built. Twine's layout is two
// 8-byte children followed by two 1-byte kind tags; CStringKind is 3 and EmptyKind is 1.
static void *make_triple(void *llvm, void *storage) {
    TripleCtor ctor = (TripleCtor)dlsym(llvm, "_ZN4llvm6TripleC1ERKNS_5TwineE");
    if (!ctor) { fprintf(stderr, "agx3meta: Triple ctor not exported\n"); return NULL; }
    static const char *tt = "agx3---macho";
    uint8_t twine[24];
    memset(twine, 0, sizeof twine);
    *(const char **)twine = tt;
    twine[16] = 3;
    twine[17] = 1;
    memset(storage, 0, 512);
    ctor(storage, twine);
    return storage;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: agx3meta instrs|classes|regs|members|regmodel\n"); return 2; }
    void *llvm = NULL;
    void *target = open_target(&llvm);
    if (!target) return 1;
    uint64_t *fields = (uint64_t *)target;

    // NAMES. LLVM's MCInstrInfo carries the TableGen instruction name table: a uint32 index array
    // and one character blob, with getName(op) = InstrNameData + InstrNameIndices[op]. If those
    // two fields can be located, every opcode carries Apple's own name and no differential
    // compilation is needed to learn what an instruction is called.
    //
    // "namesprobe" dumps the object's first words with a heuristic for which look like a string
    // blob and which like an index array, so the layout can be identified rather than assumed.
    // THE SCHEDULING MODEL. Every opcode carries a scheduling class, and this project has been
    // using those classes as an opaque grouping key that works. What they MEAN - which functional
    // unit, what latency, how many can issue at once - lives in MCSchedModel, reachable through
    // MCSubtargetInfo. That is per-opcode information about execution rather than encoding, which
    // is the one kind this project cannot get by decoding.
    //
    // "targetprobe" dumps the target registry entry so the subtarget constructor can be located
    // the same way the instruction and register constructors were.
    if (!strcmp(argv[1], "targetprobe")) {
        for (int i = 0; i < 24; i++) {
            uint64_t v = fields[i];
            printf("[%2d] %016llx", i, (unsigned long long)v);
            if (v > 0x100000000ULL && v < 0x7fffffffffffULL) {
                const char *s = (const char *)(uintptr_t)v;
                int printable = 1, k;
                for (k = 0; k < 20; k++) {
                    unsigned char c = (unsigned char)s[k];
                    if (c == 0) break;
                    if (c < 32 || c > 126) { printable = 0; break; }
                }
                if (printable && k > 2 && s[0]) printf("   \"%.20s\"", s);
                else {
                    Dl_info info;
                    if (dladdr((void *)(uintptr_t)v, &info) && info.dli_sname)
                        printf("   fn %s", info.dli_sname);
                }
            }
            printf("\n");
        }
        return 0;
    }

    if (!strcmp(argv[1], "namesprobe")) {
        uint64_t *mii = (uint64_t *)((Ctor0)(uintptr_t)fields[TARGET_MCINSTRINFO_CTOR])();
        for (int i = 0; i < 10; i++) {
            uint64_t v = mii[i];
            printf("[%d] %016llx", i, (unsigned long long)v);
            if (v > 0x100000000ULL && v < 0x7fffffffffffULL) {
                const char *s = (const char *)(uintptr_t)v;
                int printable = 1, k;
                for (k = 0; k < 24; k++) {
                    unsigned char c = (unsigned char)s[k];
                    if (c == 0) break;
                    if (c < 32 || c > 126) { printable = 0; break; }
                }
                if (printable && k > 2 && s[0]) printf("  string \"%.28s\"", s);
                else {
                    const uint32_t *u = (const uint32_t *)(uintptr_t)v;
                    printf("  u32 %u %u %u %u %u %u", u[0], u[1], u[2], u[3], u[4], u[5]);
                }
            }
            printf("\n");
        }
        {
            const uint32_t *idx = (const uint32_t *)(uintptr_t)mii[1];
            const char *data = (const char *)(uintptr_t)mii[2];
            printf("\nfirst 20 indices:");
            for (int i = 0; i < 20; i++) printf(" %u", idx[i]);
            printf("\nname blob, first 160 bytes (dots for non-printable):\n  ");
            for (int i = 0; i < 160; i++) {
                unsigned char c = (unsigned char)data[i];
                putchar((c >= 32 && c < 127) ? c : '.');
            }
            printf("\nblob at index[1]=%u:\n  ", idx[1]);
            for (int i = 0; i < 80; i++) {
                unsigned char c = (unsigned char)data[idx[1] + i];
                putchar((c >= 32 && c < 127) ? c : '.');
            }
            printf("\n");
        }
        return 0;
    }

    // "names" emits one line per opcode once the two field offsets are known.
    if (!strcmp(argv[1], "names")) {
        if (argc < 4) { fprintf(stderr, "usage: names <idx_word> <data_word>\n"); return 1; }
        uint64_t *mii = (uint64_t *)((Ctor0)(uintptr_t)fields[TARGET_MCINSTRINFO_CTOR])();
        const uint32_t *idx = (const uint32_t *)(uintptr_t)mii[atoi(argv[2])];
        const char *data = (const char *)(uintptr_t)mii[atoi(argv[3])];
        unsigned n = (unsigned)mii[5];
        if (!idx || !data) { fprintf(stderr, "names: null field\n"); return 1; }
        for (unsigned i = 0; i < n; i++)
            printf("%u %s\n", i, data + idx[i]);
        return 0;
    }

    if (!strcmp(argv[1], "instrs")) {
        uint64_t *mii = (uint64_t *)((Ctor0)(uintptr_t)fields[TARGET_MCINSTRINFO_CTOR])();
        const uint8_t *desc = (const uint8_t *)(uintptr_t)mii[0];
        unsigned n = (unsigned)mii[5];
        for (unsigned i = 0; i < n; i++)
            if (*(const uint16_t *)(desc + (size_t)INSTRDESC_STRIDE * i) != (uint16_t)i) {
                fprintf(stderr, "agx3meta: MCInstrDesc stride 48 no longer validates at %u\n", i);
                return 1;
            }
        /* THE WORD AT +8 IS LLVM'S GENERIC Flags - MayLoad, MayStore, Branch, Call, Return,
           Terminator, side effects - and the one at +16, printed here as tsflags since this tool
           was written, is the target-specific one. Both are dumped now: the generic word audits an
           instruction selector's control flow, which nothing else in this project can. */
        /* IMPLICIT OPERANDS, at +24 (uses) and +32 (defs). An instruction that reads or writes a
           register WITHOUT naming it in its operand list is invisible to a scheduler that reads
           only the operand list, and it will reorder across a dependency that is really there.
           These are the registers the ISA touches behind the encoding, so they are exactly what a
           compiler needs and exactly what nothing else in this project exports.

           They are null-terminated MCPhysReg (uint16) arrays. VALIDATED, not assumed: every entry
           must be a register id below the count the register info reports, and the array must
           terminate within a small bound. A pointer that fails either test is reported as "?"
           rather than printed, because a plausible-looking wrong register is worse than a hole. */
        if ((g_translate = meta_mode(n, AGX3_OLD_OPCODES, AGX3_NEW_OPCODES, "opcodes")) < 0) return 4;
        printf("# opcode nops ndefs schedclass tsflags flags8 uses=<r,..> defs=<r,..> "
               "[regclass:type:flags per operand]\n");
        for (unsigned i = 0; i < n; i++) {
            const uint8_t *d = desc + (size_t)INSTRDESC_STRIDE * i;
            unsigned nops = *(const uint16_t *)(d + 2);
            const uint8_t *oi = *(const uint8_t **)(d + 40);
            char *buf; size_t len;
            long id = TR_OP(i);
            if (g_translate && i == AGX3_MERGED_NEW) id = AGX3_MERGED_HIGH;   // and 14156 below
            FILE *o = row_begin(id, &buf, &len);
            // the decoder drops the merged form's extra operand (always 0), so its descriptor does too
            if (g_translate && i == AGX3_MERGED_NEW && nops == 3) nops = 2;
            fprintf(o, "%ld %u %u %ld %llx %x", id, nops, d[4], TR_SCHED(*(const uint16_t *)(d + 6)),
                   (unsigned long long)*(const uint64_t *)(d + 16),
                   *(const uint32_t *)(d + 8));
            for (int which = 0; which < 2; which++) {
                const uint16_t *p = *(const uint16_t **)(d + (which ? 32 : 24));
                fprintf(o, " %s=", which ? "defs" : "uses");
                if (!p) { fprintf(o, "-"); continue; }
                unsigned k = 0;
                for (; k < 64 && p[k]; k++)
                    if (p[k] >= IMPLICIT_REG_LIMIT) break;
                if (k == 64 || (p[k] && p[k] >= IMPLICIT_REG_LIMIT)) { fprintf(o, "?"); continue; }
                if (!k) { fprintf(o, "-"); continue; }
                for (unsigned j = 0; j < k; j++) fprintf(o, j ? ",%ld" : "%ld", TR_REG(p[j]));
            }
            if (oi)
                for (unsigned k = 0; k < nops && k < 32; k++)
                    fprintf(o, " %ld:%u:%u", TR_CLASS(*(const int16_t *)(oi + OPERANDINFO_STRIDE * k)),
                           oi[OPERANDINFO_STRIDE * k + 3], oi[OPERANDINFO_STRIDE * k + 2]);
            fprintf(o, "\n");
            row_end(o, &buf);
            if (g_translate && i == AGX3_MERGED_NEW) {       // the merged form describes both old ids
                char *dup = strdup(g_rows[g_nrows - 1].text), *sp = strchr(dup, ' ');
                FILE *o2 = row_begin(AGX3_MERGED_LOW, &buf, &len);
                fprintf(o2, "%d%s", AGX3_MERGED_LOW, sp ? sp : "\n");
                row_end(o2, &buf);
                free(dup);
            }
        }
        rows_flush();
        return 0;
    }

    uint8_t triple[512];
    if (!make_triple(llvm, triple)) return 1;
    uint64_t *mri = (uint64_t *)((Ctor1)(uintptr_t)fields[TARGET_MCREGINFO_CTOR])(triple);
    const uint8_t *regdesc = (const uint8_t *)(uintptr_t)mri[0];
    unsigned num_regs = (unsigned)(mri[1] & 0xffffffff);
    const uint8_t *classes = (const uint8_t *)(uintptr_t)mri[3];
    unsigned num_classes = (unsigned)(mri[4] & 0xffffffff);
    const char *reg_strings = (const char *)(uintptr_t)mri[8];
    const char *class_strings = (const char *)(uintptr_t)mri[9];
    if ((g_translate = meta_mode(num_regs, AGX3_OLD_REGISTERS, AGX3_NEW_REGISTERS, "registers")) < 0) return 4;

    if (!strcmp(argv[1], "classes")) {
        for (unsigned i = 0; i < num_classes; i++)
            if (*(const uint16_t *)(classes + (size_t)REGCLASS_STRIDE * i + 24) != (uint16_t)i) {
                fprintf(stderr, "agx3meta: MCRegisterClass stride 32 no longer validates at %u\n", i);
                return 1;
            }
        printf("# id name bits nregs\n");
        for (unsigned i = 0; i < num_classes; i++) {
            const uint8_t *c = classes + (size_t)REGCLASS_STRIDE * i;
            char *buf; size_t len;
            FILE *o = row_begin(TR_CLASS(i), &buf, &len);
            fprintf(o, "%ld %s %u %u\n", TR_CLASS(i), class_strings + *(const uint32_t *)(c + 16),
                   *(const uint16_t *)(c + 28), *(const uint16_t *)(c + 20));
            row_end(o, &buf);
        }
        rows_flush();
        return 0;
    }

    // MEMBERS. MCRegisterClass holds RegsBegin at +0 and RegsSize at +20, so a class is an
    // ORDERED list of MCRegisters. That order is what an instruction encodes: the flag selector in
    // op582 carries 0..6 and resolves through FLAGR's member list, not through the MCRegister id -
    // FLAGR member 6 is FLAGTRUE, whose id is 3. Correlating a field against register ids, or
    // against the R_n slot, cannot see a class-relative index at all.
    if (!strcmp(argv[1], "members")) {
        printf("# class name nregs | members in class order\n");
        for (unsigned i = 0; i < num_classes; i++) {
            const uint8_t *c = classes + (size_t)REGCLASS_STRIDE * i;
            const uint16_t *members = *(const uint16_t **)(c + 0);
            unsigned nregs = *(const uint16_t *)(c + 20);
            if (!members) continue;
            char *buf; size_t len;
            FILE *o = row_begin(TR_CLASS(i), &buf, &len);
            fprintf(o, "%ld %s %u |", TR_CLASS(i), class_strings + *(const uint32_t *)(c + 16), nregs);
            for (unsigned k = 0; k < nregs; k++) {
                unsigned r = members[k];
                fprintf(o, " %s", r < num_regs
                       ? reg_strings + *(const uint32_t *)(regdesc + (size_t)REGDESC_STRIDE * r)
                       : "?");
            }
            fprintf(o, "\n");
            row_end(o, &buf);
        }
        rows_flush();
        return 0;
    }

    if (!strcmp(argv[1], "regmodel")) {
        const int16_t *difflists = (const int16_t *)(uintptr_t)mri[6];
        unsigned num_units = (unsigned)(mri[4] >> 32);
        unsigned leaves = 0;
        // A DiffList is seeded with the register itself and difference-encoded until a 0.
        #define WALK(field, body) do {                                                        \
            uint32_t _off = *(const uint32_t *)(regdesc + (size_t)REGDESC_STRIDE * i + (field)); \
            const int16_t *_p = difflists + _off;                                             \
            long _v = i;                                                                      \
            while (*_p) { _v += *_p++; body }                                                 \
        } while (0)
        for (unsigned i = 0; i < num_regs; i++) {
            uint32_t off = *(const uint32_t *)(regdesc + (size_t)REGDESC_STRIDE * i + REGDESC_SUBREGS);
            if (!difflists[off]) leaves++;
        }
        if (leaves != num_units) {
            fprintf(stderr, "agx3meta: %u leaf registers but NumRegUnits is %u; the leaf model "
                            "no longer holds\n", leaves, num_units);
            return 1;
        }
        printf("# id name | direct sub-registers (%u leaves == NumRegUnits %u)\n", leaves, num_units);
        for (unsigned i = 0; i < num_regs; i++) {
            char *buf; size_t len;
            FILE *o = row_begin(TR_REG(i), &buf, &len);
            fprintf(o, "%ld %s |", TR_REG(i), reg_strings + *(const uint32_t *)(regdesc + (size_t)REGDESC_STRIDE * i));
            WALK(REGDESC_SUBREGS, {
                if (_v > 0 && (unsigned long)_v < num_regs)
                    fprintf(o, " %s", reg_strings + *(const uint32_t *)(regdesc + (size_t)REGDESC_STRIDE * _v));
            });
            fprintf(o, "\n");
            row_end(o, &buf);
        }
        rows_flush();
        return 0;
    }

    if (!strcmp(argv[1], "regs")) {
        // MCRegisterDesc stride is taken as the smallest 4-aligned width whose Name field
        // stays inside the string blob for every register.
        for (int stride = 16; stride <= 40; stride += 4) {
            int ok = 1;
            for (unsigned i = 1; i < num_regs && ok; i++)
                if (*(const uint32_t *)(regdesc + (size_t)stride * i) > 200000) ok = 0;
            if (!ok) continue;
            printf("# id name (stride %d)\n", stride);
            for (unsigned i = 0; i < num_regs; i++) {
                char *buf; size_t len;
                FILE *o = row_begin(TR_REG(i), &buf, &len);
                fprintf(o, "%ld %s\n", TR_REG(i), reg_strings + *(const uint32_t *)(regdesc + (size_t)stride * i));
                row_end(o, &buf);
            }
            rows_flush();
            return 0;
        }
        fprintf(stderr, "agx3meta: no MCRegisterDesc stride validates\n");
        return 1;
    }
    fprintf(stderr, "agx3meta: unknown mode %s\n", argv[1]);
    return 2;
}
