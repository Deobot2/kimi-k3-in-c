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
 * FOUR THINGS ARE GATED
 *   1. the pinned set is chosen LARGEST FIRST, and the ring slot is therefore sized to
 *      the largest UNPINNED layer rather than the largest layer;
 *   2. largest-first pins at least as many bytes as prefix pinning at the same budget,
 *      which is the entire claim the change rests on;
 *   3. a fetched layer's bytes survive an arbitrary number of prefetch hints landing in
 *      other slots, at ring depths 1, 2 and 3;
 *   4. the io_uring and pread paths return identical bytes, so the fast path cannot
 *      quietly differ from the portable one.
 *
 * usage: test_trunk <scratch_dir>
 */
#define _POSIX_C_SOURCE 200809L

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3.h"
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

    /* ---- 4: io_uring and pread must agree ---- */
    {
        setenv("K3_NOURING", "1", 1);
        run_case(dir, &c, len, budget, 3, "pread only (K3_NOURING)");
        unsetenv("K3_NOURING");
    }

    printf("\n%s\n", g_fail ? "TRUNK TEST FAILED" : "TRUNK TEST PASSED");
    return g_fail ? 1 : 0;
}
