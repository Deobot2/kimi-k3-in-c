/* test_trunk.c - the streaming trunk, without a checkpoint.
 *
 * WHY THIS EXISTS. Everything the trunk does that can go wrong goes wrong SILENTLY. A
 * layer read into the slot another layer is being computed on does not crash, does not
 * short-read and does not change a single pointer: the run finishes and emits fluent,
 * wrong tokens. That failure has actually happened here (see the note in k3_trunk.c on
 * the single-slot ring), it was found by comparing token ids against a previous run, and
 * nothing in `make test` could have caught it, because the whole trunk path needed a
 * 108.81 GB checkpoint to reach.
 *
 * It does not. A trunk is a directory with a trunk.json and a trunk.bin, and this file
 * writes a small one whose every layer is filled with a byte pattern derived from its
 * own index. Then it walks the layers the way the engine does -- fetch L, hint L+1,
 * compute -- and checks after every fetch that the bytes really are layer L's. A slot
 * reused underneath the caller changes the pattern, so the check that cannot be made
 * against real weights (they all look like noise) is trivial against these.
 *
 * SIX THINGS ARE GATED
 *   1. the pinned set is chosen LARGEST FIRST, and the ring slot is therefore sized to
 *      the largest UNPINNED layer rather than the largest layer;
 *   2. largest-first pins at least as many bytes as prefix pinning at the same budget,
 *      which is the entire claim the change rests on;
 *   3. a fetched layer's bytes survive an arbitrary number of prefetch hints landing in
 *      other slots, at ring depths 1, 2 and 3;
 *   4. NO LAYER IS READ MORE THAN THE WALK OWES -- once for a pinned layer, once per pass
 *      for a streamed one. This is bytes rather than correctness, so nothing else can see
 *      it, and it is checked per layer as well as in aggregate because the aggregate says
 *      a run went over without saying where. t_sweep then asserts it across the whole
 *      space of pinned shapes at the real layer count, which is what finally caught the
 *      escape two hand-built fixtures and two arguments from mechanism all missed;
 *   5. the io_uring and pread paths return identical bytes, so the fast path cannot
 *      quietly differ from the portable one;
 *   6. a packed run's WEIGHT FORMAT is worked out correctly -- all bf16, all MXFP4, and
 *      a mixture refused. See the note above t_formats: that logic shipped wrong and
 *      rejected every quantised trunk ever built, and nothing here could see it.
 *
 * usage: test_trunk <scratch_dir>
 */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "k3.h"
#include "k3_bind.h"
#include "k3_trunk.h"

static int g_fail = 0;

static void ok(const char *what, int cond, const char *fmt, ...)
{
    va_list ap;
    printf("  %s  %-26s ", cond ? "PASS" : "FAIL", what);
    va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    printf("\n");
    if (!cond) g_fail++;
}

/* Layer sizes in 64 KB units, deliberately unsorted and with one outlier that stands in
 * for the released model's dense layer 0: 2.34 GB against a 1.17 GB mean. Prefix pinning
 * takes that one FIRST, which is exactly why it wasted a ring slot sized for it.
 *
 * The two largest are ADJACENT on purpose. Largest-first therefore pins a RUN of layers,
 * and a run is what makes the prefetcher scan past them -- it skips pinned layers rather
 * than stopping at them -- and queue a layer further ahead than the ring's own depth.
 * With the sizes spread out, every queued layer happened to sit within nslot of the walk
 * and the re-read bug below could not appear. A fixture whose shape hides the bug is not
 * a fixture, so the shape is chosen to expose it. */
#define NLAY 8
static const int SIZE_UNITS[NLAY] = { 12, 3, 5, 2, 7, 4, 6, 1 };

/* A second shape, whose two biggest layers are ADJACENT. Largest-first then pins a RUN,
 * and a run is what makes the prefetcher queue further ahead than the ring is deep: it
 * SKIPS pinned layers while scanning rather than stopping at them, so from layer 1 it
 * skips 2 and 3 and queues 4 -- three ahead, with a two-slot ring. A slot holding a
 * layer that far out was not protected, so the next claim evicted it and the walk read
 * it again a moment later.
 *
 * The first shape cannot produce that: at its budget largest-first picks layers 0, 2 and
 * 7, never two in a row. A fixture that cannot reach the state is not evidence the state
 * is handled, which is exactly how this survived the first bound check. */
static const int SIZE_RUN[NLAY] = { 5, 5, 4, 4, 4, 4, 4, 4 };
#define UNIT (64 * 1024)

static unsigned char pattern_of(int L, int64_t off)
{
    /* Distinct per layer AND varying within a layer, so a partially-overwritten run is
     * caught as surely as a wholly wrong one. */
    return (unsigned char)((L * 37 + (int)(off / 4096) * 11 + 1) & 0xFF);
}

static int write_trunk(const char *dir, const int *units, int nlay,
                       int64_t *off_out, int64_t *len_out)
{
    char p[1024];
    snprintf(p, sizeof p, "%s/trunk.bin", dir);
    FILE *f = fopen(p, "wb");
    if (!f) { perror(p); return -1; }

    /* Size the scratch from the TABLE, not from a constant. It was UNIT * 16, which
     * covered the first shape's 12-unit maximum and silently overran on the second
     * shape's 20-unit layers -- a heap smash inside the test harness, which glibc
     * reported as a double free three cases later and nowhere near the cause. */
    int maxu = 0;
    for (int L = 0; L < nlay; L++) if (units[L] > maxu) maxu = units[L];
    unsigned char *buf = (unsigned char *)malloc((size_t)maxu * UNIT);
    if (!buf) { fclose(f); return -1; }
    int64_t at = 0;
    for (int L = 0; L < nlay; L++) {
        const int64_t n = (int64_t)units[L] * UNIT;        /* already 4096-aligned */
        for (int64_t i = 0; i < n; i++) buf[i] = pattern_of(L, i);
        if (fwrite(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return -1; }
        off_out[L] = at; len_out[L] = n;
        at += n;
    }
    free(buf);
    fclose(f);

    snprintf(p, sizeof p, "%s/trunk.json", dir);
    f = fopen(p, "w");
    if (!f) { perror(p); return -1; }
    fprintf(f, "{\"align\":%d,\"layers\":[", K3_TRUNK_ALIGN);
    for (int L = 0; L < nlay; L++)
        fprintf(f, "%s{\"file_off\":%lld,\"nbytes\":%lld,\"tensors\":"
                   "{\"dummy\":{\"off\":0,\"nbytes\":4,\"dtype\":\"F32\"}}}",
                L ? "," : "", (long long)off_out[L], (long long)len_out[L]);
    fprintf(f, "]}\n");
    fclose(f);
    return 0;
}

/* A config whose only role here is k3_bind_widen_bytes: the trunk asks it how much fp32
 * expansion room each slot needs. Small values keep the widen area from dwarfing the
 * synthetic layers and turning every budget comparison into a comparison of slack. */
static void tiny_cfg(K3Cfg *c)
{
    memset(c, 0, sizeof *c);
    c->hidden = 64; c->n_layers = NLAY; c->vocab = 32; c->rms_eps = 1e-5f;
    c->q_lora = 16; c->kv_lora = 8; c->latent = 16; c->n_experts = 4;
}

/* Walk the layers the way forward() does and verify every byte handed back.
 *
 * Returns the number of layers read more times than the walk owed, so the sweep below can
 * call it a few hundred times and report only what fails. quiet suppresses the per-case
 * PASS lines that a named case wants and a sweep would drown in. */
static int walk(K3Trunk *tr, const int64_t *len, int nlay, int passes,
                const char *label, int quiet)
{
    int bad = 0, checked = 0;
    for (int pass = 0; pass < passes; pass++) {
        for (int L = 0; L < nlay; L++) {
            unsigned char *base = NULL;
            if (k3_trunk_fetch(tr, L, &base) != 0) { bad++; continue; }
            /* Hint the next layers BEFORE inspecting this one. That is the order the
             * engine uses, and it is the order in which a slot-reuse bug can bite: the
             * reader is now free to be writing into other slots while these bytes are
             * read. */
            k3_trunk_prefetch(tr, L + 1);
            for (int64_t i = 0; i < len[L]; i += 4093) {   /* prime stride, not 4096 */
                if (base[i] != pattern_of(L, i)) { bad++; break; }
            }
            checked++;
        }
    }
    if (!quiet) ok(label, bad == 0, "%d fetches over %d passes, %d wrong",
                   checked, passes, bad);
    else if (bad) ok(label, 0, "%d fetches over %d passes, %d wrong", checked, passes, bad);

    /* HOW MANY BYTES DID THAT COST? A layer needed in a pass is read at most ONCE:
     * pinned layers are read on first touch and never again, and a streamed layer is
     * read when the walk reaches it -- by the reader thread if the prefetch got there
     * first, by the caller otherwise, but not both. So
     *
     *     bytes_read <= pinned_bytes + passes * unpinned_bytes
     *
     * and anything above that means a layer was fetched twice, or counted twice, which
     * are the same symptom from outside: a device rate that flatters and an I/O share
     * that is wrong. Neither shows up in the token stream, so nothing else here would
     * catch it. A real run reported 491 GB against a 29.81 GB trunk over ten passes,
     * which is what this bound exists to reproduce or refute. */
    int64_t pinned_b = 0, unpinned_b = 0;
    for (int L = 0; L < nlay; L++)
        if (tr->pin_of[L] >= 0) pinned_b += len[L]; else unpinned_b += len[L];
    const double bound = (double)pinned_b + (double)passes * (double)unpinned_b;
    if (!quiet)
        ok("  bytes read <= one per pass", (double)tr->bytes_read <= bound * 1.02,
           "%.2f MB read, bound %.2f MB (%d pinned + %d passes x %d streamed)",
           (double)tr->bytes_read / 1e6, bound / 1e6,
           (int)(pinned_b / 1024), passes, (int)(unpinned_b / 1024));

    /* The same claim per LAYER, which is the form that can be acted on. The aggregate says
     * a run went over and says nothing about where, and on the released checkpoint that
     * gap produced two confident explanations from the pinned set's shape, neither
     * supported by evidence. Naming the layers is what distinguishes "the prefetcher
     * reaches past a pinned run" from "the wrap at the end of a pass loses the last two",
     * which have the same aggregate and different fixes. */
    char detail[512]; int at = 0, over = 0;
    detail[0] = 0;
    for (int L = 0; L < nlay; L++) {
        const int want = (tr->pin_of[L] >= 0) ? 1 : passes;
        if ((int)tr->reads_of[L] > want) {
            over++;
            if (at < (int)sizeof detail - 32)
                at += snprintf(detail + at, sizeof detail - (size_t)at, " %d(%ux/%d%s)",
                               L, tr->reads_of[L], want,
                               tr->pin_of[L] >= 0 ? ",pin" : "");
        }
    }
    if (!quiet)
        ok("  no layer read twice", over == 0,
           over ? "re-read:%s" : "every layer read exactly what the walk owed", detail);
    return over ? over : (bad ? -1 : 0);
}

/* ------------------------------------------------------------------ the sweep ----
 * Two hand-built shapes prove that a particular pinned arrangement is handled. They
 * cannot prove that EVERY arrangement is, and that distinction stopped being academic:
 * a measured run on the released checkpoint read 13.7% more than the walk owed at a
 * budget that pins 13 of 93 layers, after a fix that both hand-built shapes accept. Two
 * explanations were then argued from the pinned set's shape and neither had evidence
 * behind it, because the fixtures could not reach the shape in question.
 *
 * So rather than guess which arrangement escapes, enumerate them. A trunk at the real
 * layer count, swept across budgets from "nothing pinned" to "almost everything pinned"
 * and across every ring depth, walks the whole space of pinned patterns the fixed-point
 * loop can produce -- runs, singletons, alternations, and the boundary cases at each end.
 * Each case asserts the same per-layer bound. A silent pass means the arrangement that
 * escapes is not one of these, which is itself worth knowing; a failure names it.
 *
 * Layer sizes: one fat layer 0, as in the released model, and five sizes cycling on a
 * stride coprime with the count, so consecutive budgets pin genuinely different sets
 * rather than lengthening one prefix. */
#define SWEEP_LAY 93

static void t_sweep(const char *dir, const K3Cfg *c)
{
    char sub[1100];
    snprintf(sub, sizeof sub, "%s/sweep", dir);
    if (mkdir(sub, 0777) != 0 && errno != EEXIST) { ok("sweep", 0, "cannot create %s", sub); return; }

    int *units = (int *)malloc(SWEEP_LAY * sizeof(int));
    int64_t *off = (int64_t *)malloc(SWEEP_LAY * sizeof(int64_t));
    int64_t *len = (int64_t *)malloc(SWEEP_LAY * sizeof(int64_t));
    if (!units || !off || !len) { ok("sweep", 0, "out of memory"); goto out; }

    for (int L = 0; L < SWEEP_LAY; L++) units[L] = 2 + (L * 7) % 5;
    units[0] = 9;                                  /* the dense layer */

    if (write_trunk(sub, units, SWEEP_LAY, off, len) != 0) {
        ok("sweep", 0, "cannot write the %d-layer trunk", SWEEP_LAY);
        goto out;
    }

    int64_t total = 0, biggest = 0;
    for (int L = 0; L < SWEEP_LAY; L++) { total += len[L]; if (len[L] > biggest) biggest = len[L]; }

    /* stdout would carry one open banner per case. The cases that matter print through
     * ok(); the rest is noise, and a few hundred banners would bury it. */
    fflush(stdout);
    const int saved = dup(1);
    const int devnull = open("/dev/null", O_WRONLY);

    int cases = 0, failed = 0, worst_pins = -1, worst_ring = 0, worst_over = 0;
    int64_t worst_budget = 0;
    const int STEPS = 24;
    for (int ring = 1; ring <= 4; ring++) {
        for (int step = 0; step <= STEPS; step++) {
            /* From a bare ring to the whole trunk resident. */
            const int64_t budget = (int64_t)ring * biggest * 2
                                 + (int64_t)((double)total * (double)step / (double)STEPS);
            K3Trunk tr;
            if (devnull >= 0) { fflush(stdout); dup2(devnull, 1); }
            const int rc = k3_trunk_open(&tr, sub, c, budget, ring);
            int over = 0;
            if (rc == 0) {
                over = walk(&tr, len, SWEEP_LAY, 3, "sweep", 1);
                if (over > worst_over) {
                    worst_over = over; worst_pins = tr.npin;
                    worst_ring = tr.nslot; worst_budget = budget;
                }
                k3_trunk_close(&tr);
            }
            if (devnull >= 0) { fflush(stdout); dup2(saved, 1); }
            if (rc != 0) { failed++; continue; }
            cases++;
            if (over) failed++;
        }
    }
    if (devnull >= 0) close(devnull);
    close(saved);

    ok("sweep: every pin shape", failed == 0,
       failed ? "%d of %d cases over the bound; worst %d layer(s) at budget %lld, "
                "%d pinned, ring %d"
              : "%d cases, %d layers, rings 1-4, budgets from bare ring to fully resident",
       failed ? failed : cases, failed ? cases : SWEEP_LAY,
       worst_over, (long long)worst_budget, worst_pins, worst_ring);

out:
    free(units); free(off); free(len);
}

static void run_case(const char *dir, const K3Cfg *c, const int64_t *len,
                     int64_t budget, int ring, const char *label)
{
    K3Trunk tr;
    if (k3_trunk_open(&tr, dir, c, budget, ring) != 0) {
        ok(label, 0, "k3_trunk_open failed");
        return;
    }
    (void)walk(&tr, len, NLAY, 3, label, 0);
    k3_trunk_close(&tr);
}

/* ---------------------------------------------------------- weight formats ----
 * A packed trunk can hold its matmul weights as bf16, as per-row int8 (the draft
 * container) or as MXFP4 (tools/mxfp4_trunk.py). The format tag lives on the WEIGHT
 * STRUCT, not on each tensor, so a layer must be entirely one of them -- and the binder
 * has to work out which, from the dtypes alone, with no manifest field to consult.
 *
 * That logic had no test, and it was wrong: the flag it read as "a bf16 tensor was seen"
 * actually meant "no narrow tensor was an unknown format", which is true of a correctly
 * packed MXFP4 trunk. Every such trunk was refused as mixed, at layer 0, before a single
 * token. Nothing in the suite could see it, because binding from a packed run needs
 * ~28 named tensors at consistent shapes and no fixture provided them.
 *
 * This does. The table below is exactly what plan_layer asks for on a KDA + MoE layer,
 * at dimensions small enough to build in memory, and the run is assembled three ways.
 */
#define BPRE "language_model.model."

typedef struct {
    const char *name;
    int rows, cols;      /* cols == 0 marks a 1D tensor of `rows` elements */
    int narrow;          /* 1 = a matmul weight, so quantisable            */
    int64_t off, nbytes;
    int dtype;
} BTensor;

/* Tiny but structurally faithful: hidden 128, 4 KDA heads of 32, latent 64, 8 experts.
 * Every narrow tensor's COLUMN count is a multiple of 32, which is the packer's
 * requirement and is true of every narrow tensor in the released model. */
static BTensor g_bt[] = {
    { BPRE "layers.1.input_layernorm.weight",              128, 0, 0, 0, 0, 0 },
    { BPRE "layers.1.post_attention_layernorm.weight",     128, 0, 0, 0, 0, 0 },
    { BPRE "layers.1.self_attention_res_norm.weight",      128, 0, 0, 0, 0, 0 },
    { BPRE "layers.1.self_attention_res_proj.weight",      128, 0, 0, 0, 0, 0 },
    { BPRE "layers.1.mlp_res_norm.weight",                 128, 0, 0, 0, 0, 0 },
    { BPRE "layers.1.mlp_res_proj.weight",                 128, 0, 0, 0, 0, 0 },
    { BPRE "layers.1.self_attn.q_proj.weight",             128, 128, 1, 0, 0, 0 },
    { BPRE "layers.1.self_attn.k_proj.weight",             128, 128, 1, 0, 0, 0 },
    { BPRE "layers.1.self_attn.v_proj.weight",             128, 128, 1, 0, 0, 0 },
    { BPRE "layers.1.self_attn.g_proj.weight",             128, 128, 1, 0, 0, 0 },
    { BPRE "layers.1.self_attn.o_proj.weight",             128, 128, 1, 0, 0, 0 },
    { BPRE "layers.1.self_attn.q_conv1d.weight",           512, 0, 0, 0, 0, 0 },
    { BPRE "layers.1.self_attn.k_conv1d.weight",           512, 0, 0, 0, 0, 0 },
    { BPRE "layers.1.self_attn.v_conv1d.weight",           512, 0, 0, 0, 0, 0 },
    { BPRE "layers.1.self_attn.f_a_proj.weight",            32, 128, 1, 0, 0, 0 },
    { BPRE "layers.1.self_attn.f_b_proj.weight",           128,  32, 1, 0, 0, 0 },
    { BPRE "layers.1.self_attn.b_proj.weight",               4, 128, 1, 0, 0, 0 },
    { BPRE "layers.1.self_attn.A_log",                      32, 0, 0, 0, 0, 0 },
    { BPRE "layers.1.self_attn.dt_bias",                   128, 0, 0, 0, 0, 0 },
    { BPRE "layers.1.self_attn.o_norm.weight",              32, 0, 0, 0, 0, 0 },
    /* The router gate is 2D, bf16 and large, and the engine STILL wants it as fp32:
     * k3_router walks it with its own inline matmul. It is the one tensor a size-based
     * packer rule would wrongly take, so it is here with narrow = 0. */
    { BPRE "layers.1.block_sparse_moe.gate.weight",       1024, 0, 0, 0, 0, 0 },
    { BPRE "layers.1.block_sparse_moe.gate.e_score_correction_bias", 8, 0, 0, 0, 0, 0 },
    { BPRE "layers.1.block_sparse_moe.routed_expert_down_proj.weight",  64, 128, 1, 0, 0, 0 },
    { BPRE "layers.1.block_sparse_moe.routed_expert_up_proj.weight",   128,  64, 1, 0, 0, 0 },
    { BPRE "layers.1.block_sparse_moe.routed_expert_norm.weight",       64, 0, 0, 0, 0, 0 },
    { BPRE "layers.1.block_sparse_moe.shared_experts.gate_proj.weight", 64, 128, 1, 0, 0, 0 },
    { BPRE "layers.1.block_sparse_moe.shared_experts.up_proj.weight",   64, 128, 1, 0, 0, 0 },
    { BPRE "layers.1.block_sparse_moe.shared_experts.down_proj.weight",128,  64, 1, 0, 0, 0 },
};
enum { BT_N = (int)(sizeof g_bt / sizeof g_bt[0]) };

static int bfind(void *ctx, const char *name, int64_t *off, int64_t *nb, int *dt)
{
    (void)ctx;
    for (int i = 0; i < BT_N; i++)
        if (!strcmp(g_bt[i].name, name)) {
            *off = g_bt[i].off; *nb = g_bt[i].nbytes; *dt = g_bt[i].dtype;
            return 0;
        }
    return -1;
}

/* Lay the run out. `fmt` is the dtype every NARROW tensor gets; `bf16_at` names one
 * narrow tensor forced back to bf16, for the mixed case, or -1. */
static int64_t blayout(int fmt, int bf16_at)
{
    int64_t at = 0;
    int narrow_i = 0;
    for (int i = 0; i < BT_N; i++) {
        BTensor *t = &g_bt[i];
        int64_t nb;
        if (!t->narrow) {
            t->dtype = K3_DT_F32;
            nb = (int64_t)t->rows * 4;             /* wide tensors are fp32 on disk */
        } else {
            const int use = (narrow_i++ == bf16_at) ? K3_DT_BF16 : fmt;
            t->dtype = use;
            nb = (use == K3_DT_MX4)
               ? (int64_t)t->rows * (t->cols / 2 + t->cols / 32)
               : (int64_t)t->rows * t->cols * 2;
        }
        at = (at + 7) & ~(int64_t)7;
        t->off = at; t->nbytes = nb;
        at += nb;
    }
    return at;
}

static void bcfg(K3Cfg *c, int *fa)
{
    memset(c, 0, sizeof *c);
    c->hidden = 128; c->n_layers = 13; c->vocab = 256; c->rms_eps = 1e-5f;
    c->kda_heads = 4; c->kda_head_dim = 32; c->conv_k = 4; c->gate_lb = -5.0f;
    c->n_experts = 8; c->topk = 2; c->n_shared = 2;
    c->latent = 64; c->moe_inter = 32; c->routed_scale = 1.0f;
    c->moe_renorm = 1; c->latent_norm = 1;
    c->first_dense = 1; c->dense_inter = 64;
    c->attn_res_block = 3;
    c->situ_b1 = 4.0f; c->situ_b2 = 25.0f;
    fa[0] = 4; fa[1] = 8; fa[2] = 12;      /* ONE-based: layer 1 is KDA, not MLA */
    c->n_full_attn = 3; c->full_attn = fa;
}

static void t_formats(void)
{
    K3Cfg c; int fa[4]; bcfg(&c, fa);
    const size_t widen_cap = k3_bind_widen_bytes(&c);

    struct { int fmt; int bf16_at; int want_ok; int want_wdt; const char *label; } cases[] = {
        { K3_DT_BF16, -1, 1, K3_WBF16, "bind: all bf16"          },
        { K3_DT_MX4,  -1, 1, K3_WMX4,  "bind: all MXFP4"         },
        { K3_DT_MX4,   0, 0, 0,        "bind: refuses mixed"     },
    };

    for (int k = 0; k < 3; k++) {
        const int64_t n = blayout(cases[k].fmt, cases[k].bf16_at);
        unsigned char *run = (unsigned char *)calloc((size_t)n, 1);
        unsigned char *wid = (unsigned char *)calloc(widen_cap, 1);
        K3LayerBind b;
        K3MemSrc src; src.find = bfind; src.ctx = NULL;
        if (!run || !wid) { ok(cases[k].label, 0, "alloc"); free(run); free(wid); continue; }
        const int rc = k3_bind_layer_mem(&c, 1, &b, run, &src, wid, widen_cap, NULL);
        if (cases[k].want_ok) {
            ok(cases[k].label, rc == 0 && b.kda.wdt == cases[k].want_wdt,
               "rc %d, layer tagged %d (want %d)", rc, b.kda.wdt, cases[k].want_wdt);
        } else {
            /* The refusal is the point: one bf16 matmul weight among MXFP4 ones cannot
             * be described by a per-struct tag, and binding it anyway would read the
             * other format's bytes and run a different model without saying so. */
            ok(cases[k].label, rc != 0, "rc %d (must be non-zero)", rc);
        }
        free(run); free(wid);
    }
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    printf("Kimi K3 streaming trunk, on a synthetic trunk in %s\n\n", dir);

    int64_t off[NLAY], len[NLAY];
    if (write_trunk(dir, SIZE_UNITS, NLAY, off, len) != 0) {
        fprintf(stderr, "cannot write the synthetic trunk into %s\n", dir);
        return 1;
    }
    K3Cfg c; tiny_cfg(&c);

    int64_t total = 0, biggest = 0;
    for (int L = 0; L < NLAY; L++) { total += len[L]; if (len[L] > biggest) biggest = len[L]; }

    /* A budget that pins some but not all: two ring slots plus roughly a third of the
     * layer bytes. The interesting regime is exactly here, where the ring slot is a
     * meaningful fraction of the budget. */
    const int64_t budget = 2 * biggest + total / 3;

    /* ---- 1 & 2: the pinned set ---- */
    {
        K3Trunk a, b;
        unsetenv("K3_PIN_PREFIX");
        int rc_a = k3_trunk_open(&a, dir, &c, budget, 2);
        setenv("K3_PIN_PREFIX", "1", 1);
        int rc_b = k3_trunk_open(&b, dir, &c, budget, 2);
        unsetenv("K3_PIN_PREFIX");

        if (rc_a != 0 || rc_b != 0) {
            ok("trunk_open", 0, "largest-first %d, prefix %d", rc_a, rc_b);
        } else {
            int64_t big_unpinned_a = 0, pinned_a = 0, pinned_b = 0;
            for (int L = 0; L < NLAY; L++) {
                if (a.pin_of[L] >= 0) pinned_a += len[L];
                else if (len[L] > big_unpinned_a) big_unpinned_a = len[L];
                if (b.pin_of[L] >= 0) pinned_b += len[L];
            }
            /* The largest layer must be pinned first, so the ring never has to hold it. */
            int argmax = 0;
            for (int L = 1; L < NLAY; L++) if (len[L] > len[argmax]) argmax = L;
            ok("largest layer pinned", a.pin_of[argmax] >= 0,
               "layer %d is the biggest at %lld KB", argmax, (long long)(len[argmax] / 1024));
            ok("ring sized to unpinned", a.slot_bytes < biggest + (int64_t)k3_bind_widen_bytes(&c) + 4096
                                         && a.slot_bytes >= big_unpinned_a,
               "slot %lld KB, largest unpinned %lld KB, largest overall %lld KB",
               (long long)(a.slot_bytes / 1024), (long long)(big_unpinned_a / 1024),
               (long long)(biggest / 1024));
            ok("pins >= prefix pins", pinned_a >= pinned_b,
               "largest-first %lld KB vs prefix %lld KB at the same budget",
               (long long)(pinned_a / 1024), (long long)(pinned_b / 1024));
            k3_trunk_close(&a);
            k3_trunk_close(&b);
        }
    }

    /* ---- 3: the ring, at every depth the engine can be asked for ---- */
    run_case(dir, &c, len, budget, 1, "ring depth 1");
    run_case(dir, &c, len, budget, 2, "ring depth 2");
    run_case(dir, &c, len, budget, 3, "ring depth 3");
    /* And at the floor, where nothing is pinned and every layer streams. */
    run_case(dir, &c, len, 3 * biggest, 3, "floor, nothing pinned");

    /* ---- 4b: a pinned RUN, which is what makes the prefetcher reach past the ring ----
     * Its own directory, because it needs a different layer-size shape. See SIZE_RUN. */
    {
        char sub[1100];
        snprintf(sub, sizeof sub, "%s/run", dir);
        if (mkdir(sub, 0777) != 0 && errno != EEXIST) {
            ok("pinned run", 0, "cannot create %s", sub);
        } else {
            int64_t off2[NLAY], len2[NLAY];
            if (write_trunk(sub, SIZE_RUN, NLAY, off2, len2) != 0) {
                ok("pinned run", 0, "cannot write the second trunk");
            } else {
                int64_t big2 = 0;
                for (int L = 0; L < NLAY; L++) if (len2[L] > big2) big2 = len2[L];
                /* Enough for a two-slot ring plus BOTH giants. They are only slightly
                 * larger than the rest, which is what lets the fixed-point loop pin them
                 * both -- a big gap would make a single ring slot cost more than a pin
                 * and the loop would settle on small layers instead. */
                run_case(sub, &c, len2, 4 * big2, 2, "pinned run, ring 2");
            }
        }
    }

    /* ---- 4c: the whole shape space, at the real layer count ---- */
    t_sweep(dir, &c);

    /* ---- 5: weight formats in a packed run ---- */
    t_formats();

    /* ---- 4: io_uring and pread must agree ---- */
    {
        setenv("K3_NOURING", "1", 1);
        run_case(dir, &c, len, budget, 3, "pread only (K3_NOURING)");
        unsetenv("K3_NOURING");
    }

    printf("\n%s\n", g_fail ? "TRUNK TEST FAILED" : "TRUNK TEST PASSED");
    return g_fail ? 1 : 0;
}
