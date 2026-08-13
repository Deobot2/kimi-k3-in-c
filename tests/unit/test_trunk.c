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
 * FIVE THINGS ARE GATED
 *   1. the pinned set is chosen LARGEST FIRST, and the ring slot is therefore sized to
 *      the largest UNPINNED layer rather than the largest layer;
 *   2. largest-first pins at least as many bytes as prefix pinning at the same budget,
 *      which is the entire claim the change rests on;
 *   3. a fetched layer's bytes survive an arbitrary number of prefetch hints landing in
 *      other slots, at ring depths 1, 2 and 3;
 *   4. the io_uring and pread paths return identical bytes, so the fast path cannot
 *      quietly differ from the portable one;
 *   5. a packed run's WEIGHT FORMAT is worked out correctly -- all bf16, all MXFP4, and
 *      a mixture refused. See the note above t_formats: that logic shipped wrong and
 *      rejected every quantised trunk ever built, and nothing here could see it.
 *
 * usage: test_trunk <scratch_dir>
 */
#define _POSIX_C_SOURCE 200809L

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
 * takes that one FIRST, which is exactly why it wasted a ring slot sized for it. */
#define NLAY 8
static const int SIZE_UNITS[NLAY] = { 12, 3, 5, 2, 7, 4, 6, 1 };
#define UNIT (64 * 1024)

static unsigned char pattern_of(int L, int64_t off)
{
    /* Distinct per layer AND varying within a layer, so a partially-overwritten run is
     * caught as surely as a wholly wrong one. */
    return (unsigned char)((L * 37 + (int)(off / 4096) * 11 + 1) & 0xFF);
}

static int write_trunk(const char *dir, int64_t *off_out, int64_t *len_out)
{
    char p[1024];
    snprintf(p, sizeof p, "%s/trunk.bin", dir);
    FILE *f = fopen(p, "wb");
    if (!f) { perror(p); return -1; }

    unsigned char *buf = (unsigned char *)malloc(UNIT * 16);
    if (!buf) { fclose(f); return -1; }
    int64_t at = 0;
    for (int L = 0; L < NLAY; L++) {
        const int64_t n = (int64_t)SIZE_UNITS[L] * UNIT;   /* already 4096-aligned */
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
    for (int L = 0; L < NLAY; L++)
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

/* Walk the layers the way forward() does and verify every byte handed back. */
static void walk(K3Trunk *tr, const int64_t *len, int passes, const char *label)
{
    int bad = 0, checked = 0;
    for (int pass = 0; pass < passes; pass++) {
        for (int L = 0; L < NLAY; L++) {
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
    ok(label, bad == 0, "%d fetches over %d passes, %d wrong", checked, passes, bad);

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
    for (int L = 0; L < NLAY; L++)
        if (tr->pin_of[L] >= 0) pinned_b += len[L]; else unpinned_b += len[L];
    const double bound = (double)pinned_b + (double)passes * (double)unpinned_b;
    ok("  bytes read <= one per pass", (double)tr->bytes_read <= bound * 1.02,
       "%.2f MB read, bound %.2f MB (%d pinned + %d passes x %d streamed)",
       (double)tr->bytes_read / 1e6, bound / 1e6,
       (int)(pinned_b / 1024), passes, (int)(unpinned_b / 1024));
}

static void run_case(const char *dir, const K3Cfg *c, const int64_t *len,
                     int64_t budget, int ring, const char *label)
{
    K3Trunk tr;
    if (k3_trunk_open(&tr, dir, c, budget, ring) != 0) {
        ok(label, 0, "k3_trunk_open failed");
        return;
    }
    walk(&tr, len, 3, label);
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
    if (write_trunk(dir, off, len) != 0) {
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
