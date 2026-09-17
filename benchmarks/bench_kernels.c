/* bench_kernels.c - where does the arithmetic actually go?
 *
 * The engine's decode time at the memory floor splits roughly 36 s trunk read, 11 s
 * expert read, 10 s compute. The read paths are now within 10-20% of what the device can
 * deliver, so the only lossless win left is the compute. Before optimising it, measure
 * which kernel owns it -- guessing which loop is hot is how people spend a week making
 * something 2% faster.
 *
 * Dimensions here are the REAL ones, per token:
 *   bf16 trunk matmuls   the attention projections, latent down/up, shared experts and
 *                        the dense MLP. Sized from the actual layer shapes.
 *   MXFP4 expert matmuls 16 experts x 3 matrices x 92 layers, in latent space.
 *
 * Reports GFLOP/s and the projected per-token seconds for each, so the two can be
 * compared directly against the measured 10 s compute budget.
 */
#define _POSIX_C_SOURCE 199309L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3.h"

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Exact bits of a whole output vector.
 *
 * Building this file with and without AVX2 and comparing these hashes is the ONLY real
 * proof that a vector path is bit-identical rather than merely close. A tolerance check
 * would happily pass a kernel that quietly reassociated the reduction, which is exactly
 * the mistake worth catching: this engine's claim is that its output equals the
 * reference, and "equals" has to mean equals. */
static void fnv(const char *label, const float *v, int n)
{
    unsigned long long h = 1469598103934665603ull;
    for (int k = 0; k < n; k++) {
        union { float f; unsigned u; } b; b.f = v[k];
        for (int t = 0; t < 4; t++) { h ^= (b.u >> (8 * t)) & 0xFFu; h *= 1099511628211ull; }
    }
    printf("             %s OUTPUT FNV1a = %016llx\n", label, h);
}

static void fillf(float *p, size_t n, unsigned s)
{
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = ((float)(s >> 8) / 8388608.0f - 1.0f) * 0.05f;
    }
}
static void fillb(unsigned char *p, size_t n, unsigned s)
{
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = (unsigned char)(s >> 13);
    }
}

int main(void)
{
    printf("kernel benchmark at REAL Kimi K3 dimensions\n");
#ifdef __AVX2__
    printf("built WITH AVX2\n\n");
#else
    printf("built WITHOUT AVX2 (scalar)\n\n");
#endif

    /* ---------- bf16 trunk matmul: KDA q_proj, 7168 -> 12288 ---------- */
    {
        const int in = 7168, out = 12288;
        uint16_t *W = (uint16_t *)malloc((size_t)in * out * sizeof(uint16_t));
        float *x = (float *)malloc((size_t)in * sizeof(float));
        float *y = (float *)malloc((size_t)out * sizeof(float));
        if (!W || !x || !y) { printf("alloc failed\n"); return 1; }
        fillb((unsigned char *)W, (size_t)in * out * 2, 12345u);
        fillf(x, in, 999u);

        k3_matmul_bf16(y, x, W, in, out);              /* warm */
        const int reps = 5;
        const double t0 = now_s();
        for (int r = 0; r < reps; r++) k3_matmul_bf16(y, x, W, in, out);
        const double dt = (now_s() - t0) / reps;
        const double gflop = 2.0 * in * out / 1e9;
        printf("bf16 matmul  %5d x %-5d  %7.2f ms  %8.1f GFLOP/s\n",
               out, in, dt * 1e3, gflop / dt);

        /* Per token the trunk is 56.74 G always-active params = 113.49 GFLOP of
         * multiply-add. Project from the rate just measured. */
        fnv("bf16 ", y, out);
        printf("             trunk is 56.74 G params/token -> %.2f s/token at this rate\n",
               2.0 * 56.74e9 / 1e9 / (gflop / dt));
        free(W); free(x); free(y);
    }

    /* ---------- MXFP4 expert matmul: w1, latent 3584 -> inter 3072 ---------- */
    {
        const int in = 3584, rows = 3072, group = K3_MXFP4_GROUP;
        const int pcols = in / 2, ngrp = in / group;
        unsigned char *pk = (unsigned char *)malloc((size_t)rows * pcols);
        unsigned char *sc = (unsigned char *)malloc((size_t)rows * ngrp);
        float *x = (float *)malloc((size_t)in * sizeof(float));
        float *y = (float *)malloc((size_t)rows * sizeof(float));
        if (!pk || !sc || !x || !y) { printf("alloc failed\n"); return 1; }
        fillb(pk, (size_t)rows * pcols, 777u);
        memset(sc, 127, (size_t)rows * ngrp);          /* scale 2^0, none skipped */
        fillf(x, in, 4242u);

        k3_matmul_mxfp4(y, x, pk, sc, in, rows, group);
        const int reps = 5;
        const double t0 = now_s();
        for (int r = 0; r < reps; r++) k3_matmul_mxfp4(y, x, pk, sc, in, rows, group);
        const double dt = (now_s() - t0) / reps;
        const double gflop = 2.0 * in * rows / 1e9;
        printf("\nMXFP4 matmul %5d x %-5d  %7.2f ms  %8.1f GFLOP/s\n",
               rows, in, dt * 1e3, gflop / dt);

        /* One expert is w1 + w3 (both 3072x3584) + w2 (3584x3072) = 3 of these.
         * 16 experts x 92 MoE layers per token. */
        fnv("mxfp4", y, rows);
        const double per_tok = dt * 3.0 * 16 * 92;
        printf("             16 experts x 3 mats x 92 layers -> %.2f s/token\n", per_tok);
        free(pk); free(sc); free(x); free(y);
    }

    /* ---------- KDA recurrence: one step, real head_dim ----------
     * ROADMAP.md item 4 called this "the largest remaining un-vectorised kernel on the
     * non-I/O path" and AVX2 has since landed for it, argued from mechanism (see the
     * comment on k3_kda_step) but never timed. This times it: 69 KDA layers x 96 heads
     * x this call, once per generated token. */
    {
        const int dk = 128, dv = 128, heads = 96, kda_layers = 69;
        float *S = (float *)malloc((size_t)dk * dv * sizeof(float));
        float *o = (float *)malloc((size_t)dv * sizeof(float));
        float *q = (float *)malloc((size_t)dk * sizeof(float));
        float *k = (float *)malloc((size_t)dk * sizeof(float));
        float *v = (float *)malloc((size_t)dv * sizeof(float));
        float *al = (float *)malloc((size_t)dk * sizeof(float));
        float *u = (float *)malloc((size_t)dv * sizeof(float));
        if (!S || !o || !q || !k || !v || !al || !u) { printf("alloc failed\n"); return 1; }
        fillf(S, (size_t)dk * dv, 2024u);
        fillf(q, dk, 55u); fillf(k, dk, 66u); fillf(v, dv, 77u);
        for (int i = 0; i < dk; i++) al[i] = 0.98f;    /* a plausible decay, never zero */

        k3_kda_step(S, o, q, k, v, al, 0.4f, dk, dv, u);       /* warm */
        const int reps = 2000;
        const double t0 = now_s();
        for (int r = 0; r < reps; r++) k3_kda_step(S, o, q, k, v, al, 0.4f, dk, dv, u);
        const double dt = (now_s() - t0) / reps;
        printf("\nKDA step     dk=%d dv=%d      %7.4f ms/call\n", dk, dv, dt * 1e3);
        fnv("kda  ", o, dv);
        printf("             %d heads x %d layers -> %.4f s/token\n",
               heads, kda_layers, dt * heads * kda_layers);
        free(S); free(o); free(q); free(k); free(v); free(al); free(u);
    }

    /* ---------- k3_matmul_tr: MLA latent query absorption, real dims ----------
     * kv_lora=512, qk_nope=128, once per head per MLA layer per token (24 layers). The
     * comment at its call site calls it "small next to the recurrence"; this checks
     * that against the number above rather than leaving it asserted. */
    {
        const int in = 512, rows = 128, heads = 96, mla_layers = 24;
        uint16_t *W = (uint16_t *)malloc((size_t)in * rows * sizeof(uint16_t));
        float *x = (float *)malloc((size_t)rows * sizeof(float));
        float *y = (float *)malloc((size_t)in * sizeof(float));
        if (!W || !x || !y) { printf("alloc failed\n"); return 1; }
        fillb((unsigned char *)W, (size_t)in * rows * 2, 88u);
        fillf(x, rows, 33u);

        k3_matmul_tr(y, x, W, K3_WBF16, in, rows);             /* warm */
        const int reps = 2000;
        const double t0 = now_s();
        for (int r = 0; r < reps; r++) k3_matmul_tr(y, x, W, K3_WBF16, in, rows);
        const double dt = (now_s() - t0) / reps;
        printf("\nmatmul_tr    in=%d rows=%d    %7.4f ms/call\n", in, rows, dt * 1e3);
        fnv("mtr  ", y, in);
        printf("             %d heads x %d layers -> %.4f s/token\n",
               heads, mla_layers, dt * heads * mla_layers);
        free(W); free(x); free(y);
    }

    printf("\nmeasured compute budget at the floor is about 10 s/token; whichever line\n"
           "above dominates it is the one worth vectorising.\n");
    return 0;
}
