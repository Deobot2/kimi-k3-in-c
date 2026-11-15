/* test_prefill.c - the multi-token paths must be the per-token paths, to the bit.
 *
 * Prefill, full recompute and --ppl run every layer with T > 1, and the kernels batch
 * those tokens so each weight matrix is streamed once per block of tokens instead of
 * once per token. Batching is only allowed to change the ORDER in which dot products
 * are visited, never any floating-point result, so every check here is memcmp, not a
 * tolerance:
 *
 *   mmw_n      k3_mmw_n(T) against T calls of k3_mmw, for all four weight formats and
 *              for T on both sides of the 16-token block
 *   moe_pref   k3_moe_prefill(T) with a streamed MXFP4 expert source against T calls
 *              of k3_moe(T=1), across a 64-token chunk boundary. This is the only test
 *              that reaches moe_prefill_chunk at all: the oracle runs resident experts,
 *              and the streamed batch path needs a source.
 *
 * Synthetic weights only; no fixtures, no checkpoint.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3.h"

static int g_bad = 0;
static unsigned g_s = 2463534242u;

static unsigned rnd(void) { g_s ^= g_s << 13; g_s ^= g_s >> 17; g_s ^= g_s << 5; return g_s; }
static float frnd(void) { return (float)(rnd() >> 8) / 8388608.0f - 1.0f; }

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "allocation of %zu bytes failed\n", n); exit(2); }
    return p;
}

static void fill_f(float *p, size_t n, float scale)
{
    for (size_t i = 0; i < n; i++) p[i] = frnd() * scale;
}

static uint16_t *make_bf16(size_t n, float scale)
{
    uint16_t *p = (uint16_t *)xmalloc(n * sizeof *p);
    for (size_t i = 0; i < n; i++) {
        float f = frnd() * scale;
        uint32_t u; memcpy(&u, &f, 4);
        p[i] = (uint16_t)(u >> 16);
    }
    return p;
}

/* Random packed nibbles and E8M0 scales near 2^0, with one NaN (255) group so the
 * skip rule is exercised on both sides. */
static void fill_mx(unsigned char *packed, unsigned char *scales, size_t npk, size_t nsc)
{
    for (size_t i = 0; i < npk; i++) packed[i] = (unsigned char)(rnd() >> 11);
    for (size_t i = 0; i < nsc; i++) scales[i] = (unsigned char)(120 + (rnd() >> 7) % 12);
    if (nsc > 3) scales[3] = 255;
}

/* A matrix of `out` rows of width `in` in the given format, as k3_mmw reads it. */
static void *make_matrix(int wdt, int in, int out)
{
    const size_t rb = k3_row_bytes(wdt, in);
    unsigned char *W = (unsigned char *)xmalloc(rb * (size_t)out);
    for (int r = 0; r < out; r++) {
        unsigned char *row = W + rb * (size_t)r;
        if (wdt == K3_WF32) {
            fill_f((float *)row, (size_t)in, 0.1f);
        } else if (wdt == K3_WBF16) {
            uint16_t *b = make_bf16((size_t)in, 0.1f);
            memcpy(row, b, (size_t)in * 2);
            free(b);
        } else if (wdt == K3_WI8) {
            const float s = 0.001f * (float)(1 + r % 7);
            memcpy(row, &s, 4);
            for (int i = 0; i < in; i++) row[4 + i] = (unsigned char)(rnd() >> 13);
        } else {
            fill_mx(row, row + in / 2, (size_t)in / 2, (size_t)in / K3_MXFP4_GROUP);
        }
    }
    return W;
}

static void check(const char *name, const float *a, const float *b, size_t n)
{
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < n; i++)
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) { if (!bad) first = i; bad++; }
    if (bad) {
        printf("  FAIL  %-30s %zu of %zu differ, first at %zu (%.9g vs %.9g)\n",
               name, bad, n, first, (double)a[first], (double)b[first]);
        g_bad = 1;
    } else {
        printf("  PASS  %-30s %zu values bit-identical\n", name, n);
    }
}

/* ----------------------------------------------------------------- mmw_n ---- */
static void t_mmw_n(void)
{
    static const int fmts[4] = { K3_WF32, K3_WBF16, K3_WI8, K3_WMX4 };
    static const char *fname[4] = { "f32", "bf16", "int8", "mxfp4" };
    static const int Ts[6] = { 1, 2, 15, 16, 17, 40 };
    const int in = 96, out = 70;   /* in a multiple of 32 for MXFP4; out above the OpenMP gate */
    for (int f = 0; f < 4; f++) {
        void *W = make_matrix(fmts[f], in, out);
        for (int k = 0; k < 6; k++) {
            const int T = Ts[k];
            float *x  = (float *)xmalloc((size_t)T * in * sizeof(float));
            float *ya = (float *)xmalloc((size_t)T * out * sizeof(float));
            float *yb = (float *)xmalloc((size_t)T * out * sizeof(float));
            fill_f(x, (size_t)T * in, 1.0f);
            for (int t = 0; t < T; t++)
                k3_mmw(ya + (size_t)t * out, x + (size_t)t * in, W, fmts[f], in, out);
            k3_mmw_n(yb, x, W, fmts[f], in, out, T);
            char name[64];
            snprintf(name, sizeof name, "mmw_n %s T=%d", fname[f], T);
            check(name, ya, yb, (size_t)T * out);
            free(x); free(ya); free(yb);
        }
        free(W);
    }
}

/* ------------------------------------------------- streamed MoE prefill ---- */
typedef struct {
    int L, I, n_experts;
    unsigned char *p1, *s1, *p3, *s3, *p2, *s2;   /* [n_experts] banks, contiguous */
    int gets;
} MemSrc;

static int mem_get(K3ExpertSrc *self, int layer, int e, K3ExpertQ *q)
{
    MemSrc *m = (MemSrc *)self->ctx;
    (void)layer;
    if (e < 0 || e >= m->n_experts) return -1;
    const size_t up_pk = (size_t)m->I * m->L / 2, up_sc = (size_t)m->I * m->L / K3_MXFP4_GROUP;
    const size_t dn_pk = (size_t)m->L * m->I / 2, dn_sc = (size_t)m->L * m->I / K3_MXFP4_GROUP;
    q->p1 = m->p1 + up_pk * e;  q->s1 = m->s1 + up_sc * e;
    q->p3 = m->p3 + up_pk * e;  q->s3 = m->s3 + up_sc * e;
    q->p2 = m->p2 + dn_pk * e;  q->s2 = m->s2 + dn_sc * e;
    m->gets++;
    return 0;
}

static void t_moe_prefill(void)
{
    K3Cfg c;
    memset(&c, 0, sizeof c);
    c.hidden = 256; c.latent = 64; c.moe_inter = 96; c.n_shared = 2;
    c.n_experts = 12; c.topk = 4; c.moe_renorm = 1; c.routed_scale = 2.5f;
    c.latent_norm = 1; c.rms_eps = 1e-5f; c.situ_b1 = 4.0f; c.situ_b2 = 25.0f;
    const int E = c.hidden, L = c.latent, I = c.moe_inter, SI = I * c.n_shared;
    const int NE = c.n_experts;

    MemSrc m = { L, I, NE, NULL, NULL, NULL, NULL, NULL, NULL, 0 };
    const size_t up_pk = (size_t)I * L / 2, up_sc = (size_t)I * L / K3_MXFP4_GROUP;
    m.p1 = xmalloc(up_pk * NE); m.s1 = xmalloc(up_sc * NE);
    m.p3 = xmalloc(up_pk * NE); m.s3 = xmalloc(up_sc * NE);
    m.p2 = xmalloc(up_pk * NE); m.s2 = xmalloc(up_sc * NE);
    fill_mx(m.p1, m.s1, up_pk * NE, up_sc * NE);
    fill_mx(m.p3, m.s3, up_pk * NE, up_sc * NE);
    fill_mx(m.p2, m.s2, up_pk * NE, up_sc * NE);
    K3ExpertSrc src;
    memset(&src, 0, sizeof src);
    src.get = mem_get; src.ctx = &m;

    float *gate = (float *)xmalloc((size_t)NE * E * sizeof(float));
    float *bias = (float *)xmalloc((size_t)NE * sizeof(float));
    float *lnorm = (float *)xmalloc((size_t)L * sizeof(float));
    fill_f(gate, (size_t)NE * E, 0.2f);
    fill_f(bias, (size_t)NE, 0.05f);
    for (int i = 0; i < L; i++) lnorm[i] = 1.0f + 0.1f * frnd();

    K3MoeW w;
    memset(&w, 0, sizeof w);
    w.gate = gate; w.bias = bias; w.latent_norm = lnorm; w.wdt = K3_WBF16;
    w.down = make_bf16((size_t)L * E, 0.1f);
    w.up   = make_bf16((size_t)E * L, 0.1f);
    w.sh1  = make_bf16((size_t)SI * E, 0.1f);
    w.sh3  = make_bf16((size_t)SI * E, 0.1f);
    w.sh2  = make_bf16((size_t)E * SI, 0.1f);
    w.src = &src; w.layer = 0;

    const int T = 70;                     /* crosses the 64-token prefill chunk */
    float *x   = (float *)xmalloc((size_t)T * E * sizeof(float));
    float *ref = (float *)xmalloc((size_t)T * E * sizeof(float));
    float *got = (float *)xmalloc((size_t)T * E * sizeof(float));
    float *scr = (float *)xmalloc(k3_moe_scratch(&c) * sizeof(float));
    fill_f(x, (size_t)T * E, 1.0f);
    int idx[K3_MAX_TOPK]; float wt[K3_MAX_TOPK];

    for (int t = 0; t < T; t++)
        k3_moe(ref + (size_t)t * E, x + (size_t)t * E, &w, &c, 1, idx, wt, scr);
    const int per_token_gets = m.gets;
    m.gets = 0;
    k3_moe_prefill(got, x, &w, &c, T, idx, wt, scr);
    check("moe_prefill T=70 vs k3_moe", ref, got, (size_t)T * E);
    printf("        expert fetches: %d per-token, %d batched\n", per_token_gets, m.gets);
    if (m.gets >= per_token_gets) {
        printf("  FAIL  batched prefill fetched no fewer experts than per-token\n");
        g_bad = 1;
    }

    free(x); free(ref); free(got); free(scr);
    free(gate); free(bias); free(lnorm);
    free((void *)w.down); free((void *)w.up);
    free((void *)w.sh1); free((void *)w.sh3); free((void *)w.sh2);
    free(m.p1); free(m.s1); free(m.p3); free(m.s3); free(m.p2); free(m.s2);
}

int main(void)
{
    printf("multi-token paths against their per-token definitions (memcmp)\n");
    t_mmw_n();
    t_moe_prefill();
    printf(g_bad ? "PREFILL TESTS FAILED\n" : "PREFILL TESTS PASSED\n");
    return g_bad;
}
