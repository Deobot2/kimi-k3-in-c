/* test_st.c - exercise the safetensors reader and emit its index for external checking.
 *
 * The point of this program is NOT to print a pass/fail of its own opinion. It dumps
 * exactly what the C parser believes about every tensor, so tools/verify_st.py can
 * reparse the same files with Python's json and safetensors and compare field by
 * field. A parser that agrees with itself proves nothing.
 *
 * usage: test_st <dir> [index.json] [tensor_name ...]
 *          <dir>          directory of .safetensors shards
 *          index.json     where to write the full index (default st_index.json)
 *          tensor_name    zero or more tensors to read and dump values for
 *        test_st reject <scratch-dir>
 *          writes a handful of hand-crafted malformed headers under scratch-dir and
 *          asserts k3_st_open refuses every one of them (see run_reject_suite)
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "k3_st.h"

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static const char *dt_name(K3Dtype d)
{
    switch (d) {
    case K3_DT_U8:   return "U8";
    case K3_DT_BF16: return "BF16";
    case K3_DT_F16:  return "F16";
    case K3_DT_F32:  return "F32";
    default:         return "?";
    }
}

static const char *base_name(const char *p)
{
    const char *s = strrchr(p, '/');
    return s ? s + 1 : p;
}

/* Names may legally contain a quote or a backslash. Emitting them raw produces a file
 * that Python then fails to parse, which would look like a reader bug when it is only
 * a dumper bug. Escape on the way out. */
static void put_json_str(FILE *f, const char *s)
{
    fputc('"', f);
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); }
        else if (c < 0x20) fprintf(f, "\\u%04x", c);
        else fputc(c, f);
    }
    fputc('"', f);
}

/* ------------------------------------------------------------------ reject suite ----
 * SECURITY.md puts crafted headers in scope: this engine treats safetensors files as
 * untrusted input, downloaded from third-party mirrors, and must bound or refuse
 * implausible values rather than trust them. Each case below hand-writes the smallest
 * possible header that isolates one specific refusal path in k3_st.c -- no numpy, no
 * tensor data, because header validation runs (and must fail) before any data is ever
 * read. */
static int write_shard_raw(const char *dir, const char *json_body)
{
    char path[560];
    snprintf(path, sizeof path, "%s/bad.safetensors", dir);
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const uint64_t hlen = (uint64_t)strlen(json_body);
    unsigned char lenbuf[8];
    for (int i = 0; i < 8; i++) lenbuf[i] = (unsigned char)(hlen >> (8 * i));
    int ok = fwrite(lenbuf, 1, 8, f) == 8 && fwrite(json_body, 1, hlen, f) == hlen;
    fclose(f);
    return ok ? 0 : -1;
}

static int reject_case(const char *scratch, const char *label, const char *json_body)
{
    char dir[512];
    snprintf(dir, sizeof dir, "%s/%s", scratch, label);
    mkdir(dir, 0755);                       /* ignoring EEXIST: a rerun reuses it */
    if (write_shard_raw(dir, json_body) != 0) {
        printf("  FAIL  %-16s could not write the fixture\n", label);
        return 1;
    }
    K3St s;
    if (k3_st_open(&s, dir) != 0) {
        printf("  ok    %-16s correctly rejected\n", label);
        return 0;
    }
    printf("  FAIL  %-16s opened with %d tensor(s); should have been rejected\n",
           label, s.nt);
    k3_st_close(&s);
    return 1;
}

static int run_reject_suite(const char *scratch)
{
    int bad = 0;
    /* i64_'s digit accumulator must refuse rather than silently overflow. */
    bad += reject_case(scratch, "overflow_digits",
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[0,999999999999999999999999999999]}}");
    /* TWO negative dims, so their product (and so k3_st_numel(t)*elemsize) is
     * POSITIVE and equal to data_offsets' own span -- the pre-existing byte-span
     * consistency check alone does not catch this, only an explicit "no dimension is
     * negative" check does. (-1)*(-4) = 4 elements, 16 bytes as F32. */
    bad += reject_case(scratch, "negative_dim",
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[-1,-4],\"data_offsets\":[0,16]}}");
    /* data_offsets[0] negative, but [1]-[0] still equals the correct 4-byte span for
     * a 1-element F32 tensor, and base+data_offsets[1] still lands inside the file --
     * so this also passes the byte-span consistency check. Unfixed, this resolves to
     * a tensor whose absolute offset is 8 bytes before the header even starts. */
    bad += reject_case(scratch, "negative_offset",
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[-8,-4]}}");
    /* data_offsets[1] < data_offsets[0]: a negative span. Unlike the two cases above
     * this one IS still caught by the pre-existing consistency check on its own
     * (a negative nbytes cannot equal a non-negative want without also using a
     * negative dimension), but the explicit o1 >= o0 check makes that a stated
     * invariant rather than an accident of what else happens to be checked. */
    bad += reject_case(scratch, "backwards_offset",
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[10,2]}}");
    /* 3037000500 is just past floor(sqrt(INT64_MAX)); squaring it overflows a signed
     * 64-bit product. Like overflow_digits, the pre-fix behaviour here is undefined
     * rather than reliably wrong, which is the argument for the explicit check, not
     * a counter-argument: relying on whatever one compiler's UB happens to do is
     * never a substitute for refusing the input outright. */
    bad += reject_case(scratch, "numel_overflow",
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[3037000500,3037000500],"
        "\"data_offsets\":[0,4]}}");
    return bad;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && !strcmp(argv[1], "reject")) {
        if (argc < 3) { fprintf(stderr, "usage: test_st reject <scratch-dir>\n"); return 2; }
        int bad = run_reject_suite(argv[2]);
        if (bad) { fprintf(stderr, "%d case(s) were NOT rejected\n", bad); return 1; }
        return 0;
    }
    if (argc < 2) { fprintf(stderr, "usage: test_st <dir> [index.json] [tensor ...]\n"); return 2; }
    const char *dir = argv[1];
    const char *out = argc > 2 ? argv[2] : "st_index.json";

    printf("opening %s\n", dir);
    K3St s;
    double t0 = now_s();
    if (k3_st_open(&s, dir) != 0) { fprintf(stderr, "OPEN FAILED\n"); return 1; }
    double t_open = now_s() - t0;

    int64_t total = 0;
    int64_t by_dt[8] = {0};
    for (int i = 0; i < s.nt; i++) { total += s.t[i].nbytes; by_dt[s.t[i].dtype]++; }

    printf("  shards        : %d\n", s.nshard);
    printf("  tensors       : %d\n", s.nt);
    printf("  indexed bytes : %.2f GB\n", (double)total / 1e9);
    printf("  index built in: %.3f s  (%.1f us/tensor)\n",
           t_open, 1e6 * t_open / (s.nt ? s.nt : 1));
    printf("  dtypes        : U8 %lld, BF16 %lld, F16 %lld, F32 %lld\n",
           (long long)by_dt[K3_DT_U8], (long long)by_dt[K3_DT_BF16],
           (long long)by_dt[K3_DT_F16], (long long)by_dt[K3_DT_F32]);
    printf("  hash buckets  : %d for %d tensors (load %.2f)\n",
           s.nbucket, s.nt, (double)s.nt / s.nbucket);

    /* ---- round trip: every name must find its own entry, not a neighbour ---- */
    t0 = now_s();
    int bad = 0;
    for (int i = 0; i < s.nt; i++) {
        const K3Tensor *f = k3_st_find(&s, s.t[i].name);
        if (f != &s.t[i]) bad++;
    }
    double t_look = now_s() - t0;
    printf("  round trip    : %d/%d resolve to themselves%s\n",
           s.nt - bad, s.nt, bad ? "   <-- COLLISION BUG" : "");
    printf("  lookup cost   : %.0f ns each (%d lookups in %.3f s)\n",
           1e9 * t_look / (s.nt ? s.nt : 1), s.nt, t_look);

    /* ---- a name that is not present must return NULL, not a near miss ---- */
    const char *ghosts[] = { "", "no.such.tensor",
                             "language_model.model.layers.999.self_attn.A_log" };
    int ghost_bad = 0;
    for (unsigned i = 0; i < sizeof ghosts / sizeof *ghosts; i++)
        if (k3_st_find(&s, ghosts[i])) ghost_bad++;
    printf("  absent names  : %d/%zu correctly return NULL\n",
           (int)(sizeof ghosts / sizeof *ghosts) - ghost_bad, sizeof ghosts / sizeof *ghosts);

    /* ---- dump the index ---- */
    FILE *f = fopen(out, "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); k3_st_close(&s); return 1; }
    fprintf(f, "{\"nshard\":%d,\"nt\":%d,\"tensors\":{", s.nshard, s.nt);
    for (int i = 0; i < s.nt; i++) {
        const K3Tensor *t = &s.t[i];
        if (i) fputc(',', f);
        put_json_str(f, t->name);
        fprintf(f, ":{\"shard\":\"%s\",\"dtype\":\"%s\",\"shape\":[",
                base_name(s.path[t->shard]), dt_name(t->dtype));
        for (int d = 0; d < t->ndim; d++)
            fprintf(f, "%s%lld", d ? "," : "", (long long)t->shape[d]);
        fprintf(f, "],\"off\":%lld,\"nbytes\":%lld}",
                (long long)t->off, (long long)t->nbytes);
    }
    fprintf(f, "}}\n");
    fclose(f);
    printf("  wrote %s\n", out);

    /* ---- read the named tensors and dump values for comparison ---- */
    if (argc > 3) {
        FILE *v = fopen("st_values.json", "w");
        fprintf(v, "{");
        for (int a = 3; a < argc; a++) {
            const K3Tensor *t = k3_st_find(&s, argv[a]);
            if (!t) { printf("  MISSING: %s\n", argv[a]); continue; }
            int64_t n = k3_st_numel(t);
            int64_t nd = n < 4096 ? n : 4096;      /* a prefix is enough to catch a shift */
            float *buf = (float *)malloc((size_t)n * sizeof(float));
            if (!buf) { printf("  alloc failed for %s\n", argv[a]); continue; }
            double t1 = now_s();
            int64_t got = k3_st_read_f32(&s, t, buf);
            double dt = now_s() - t1;

            double mn = 1e300, mx = -1e300, sum = 0.0;
            int nonfinite = 0;
            for (int64_t i = 0; i < got; i++) {
                if (!isfinite(buf[i])) { nonfinite++; continue; }
                if (buf[i] < mn) mn = buf[i];
                if (buf[i] > mx) mx = buf[i];
                sum += buf[i];
            }
            /* The min/max/mean above are computed over the FINITE values only (the loop
             * `continue`s past the rest), so a bare "NON-FINITE VALUES" banner sitting
             * beside perfectly finite statistics reads like a contradiction and has
             * already been reported as one. It is not: the tensor genuinely contains
             * non-finite entries AND finite ones. Print the count and say which set the
             * statistics describe, so the line is unambiguous on its own.
             *
             * `tricky.f16.1d` in the fixture set carries inf/nan deliberately, so this
             * firing is the reader working, not failing. */
            printf("  %s\n", argv[a]);
            if (nonfinite) {
                printf("    %s %lld elems, %.2f MB in %.3f s (%.0f MB/s), "
                       "min %.6g max %.6g mean %.6g  "
                       "<-- %d of %lld values are non-finite (stats cover the other %lld)\n",
                       dt_name(t->dtype), (long long)got, (double)t->nbytes / 1e6, dt,
                       (double)t->nbytes / 1e6 / (dt > 0 ? dt : 1e-9), mn, mx,
                       (got - nonfinite) ? sum / (got - nonfinite) : 0.0,
                       nonfinite, (long long)got, (long long)(got - nonfinite));
            } else {
                printf("    %s %lld elems, %.2f MB in %.3f s (%.0f MB/s), "
                       "min %.6g max %.6g mean %.6g\n",
                       dt_name(t->dtype), (long long)got, (double)t->nbytes / 1e6, dt,
                       (double)t->nbytes / 1e6 / (dt > 0 ? dt : 1e-9), mn, mx,
                       got ? sum / got : 0.0);
            }

            /* Values go out as raw float32 BIT PATTERNS, not decimals. Three reasons:
             * printf writes "nan" and "-inf", which Python's json rejects; %.9g cannot
             * distinguish -0.0 from 0.0; and a NaN payload vanishes entirely. Bits make
             * the comparison exact and turn every one of those into a visible mismatch. */
            if (a > 3) fputc(',', v);
            put_json_str(v, argv[a]);
            fprintf(v, ":{\"n\":%lld,\"first_bits\":[", (long long)got);
            for (int64_t i = 0; i < nd; i++) {
                union { float f; uint32_t u; } b; b.f = buf[i];
                fprintf(v, "%s%u", i ? "," : "", b.u);
            }
            /* The tail matters as much as the head: an off-by-one in the offset shifts
             * everything, but a wrong nbytes only shows up at the end. */
            fprintf(v, "],\"last_bits\":[");
            const int64_t tail = got > 64 ? got - 64 : 0;
            for (int64_t i = tail; i < got; i++) {
                union { float f; uint32_t u; } b; b.f = buf[i];
                fprintf(v, "%s%u", i == tail ? "" : ",", b.u);
            }
            fprintf(v, "],\"tail_start\":%lld}", (long long)tail);
            free(buf);
        }
        fprintf(v, "}\n");
        fclose(v);
        printf("  wrote st_values.json\n");
    }

    k3_st_close(&s);
    printf("\n%s\n", (bad == 0 && ghost_bad == 0)
           ? "READER SELF-CHECKS PASSED (now run tools/verify_st.py for the external check)"
           : "READER SELF-CHECKS FAILED");
    return (bad == 0 && ghost_bad == 0) ? 0 : 1;
}
