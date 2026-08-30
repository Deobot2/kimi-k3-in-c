/* k3_run.c - run the REAL Kimi K3, all 93 layers, from the released checkpoint.
 *
 * WHAT THIS IS
 *   The full engine: safetensors index over 96 shards, resident trunk bound by name,
 *   routed experts streamed from disk through an LRU cache and multiplied straight out
 *   of MXFP4. Greedy decode. Token ids in, token ids out.
 *
 * MEMORY. The banner this program prints before allocating is a PLAN, not a measurement.
 *   It reports requested budgets rather than actual reservations, and in practice it
 *   OVERSTATES: across the 12-rung ladder in docs/data/ the planned total exceeded
 *   measured peak RSS by 0.13-1.84 GB, because both budgets round down to whole slots and
 *   that rounding outweighs the safetensors index it omits. Quote the "PEAK RSS" line
 *   instead, which comes from
 *   getrusage after the run. Fully resident, the weights are 108.81 GB of bf16 trunk plus
 *   4.70 GB of embed and lm_head, so 113.49 GB; streamed, the resident set is whatever
 *   budget is given, down to about 8.2 GB. The 1.45 TB of routed experts is never
 *   resident at any budget.
 *
 * THIS ENGINE IS I/O BOUND at small budgets and roughly balanced at large ones. The
 *   measured I/O share runs 40.9%-60.6% across the 12-rung ladder (docs/data/), dropping
 *   below 50% at 96 GB and above. The "I/O share" line printed at the end of every run
 *   reports it for that run. Going faster still means moving fewer bytes before it means
 *   computing less, which is why docs/TUNING.md is mostly about allocation.
 *
 * DECODE STRATEGY
 *   By default each step re-runs the whole prefix rather than carrying state forward.
 *   That is O(T^2), but it is the path the full-model oracle validates in
 *   tests/unit/k3_model.c. --incremental switches to prefill-then-one-token-at-a-time,
 *   carrying the KDA recurrent state and an MLA KV cache. GATE 3 of the tiny-model
 *   oracle requires it to produce the SAME tokens as full recompute, so the equivalence
 *   is tested rather than assumed. Context is limited by the MLA KV cache
 *   (~2.37 MB/position), not by array sizes; the engine computes the requirement up
 *   front and refuses the run if it will not fit.
 *
 * COMMAND LINE
 *   usage() below is the single source of truth for options and defaults; `k3 --help`
 *   prints it. It is not duplicated here, because a second copy is a second thing to
 *   keep correct and the copy is the one that goes stale.
 */
#define _POSIX_C_SOURCE 200809L
/* _POSIX_C_SOURCE alone hides the BSD rusage fields, ru_maxrss among them, from
 * <sys/resource.h> on Darwin. peak_rss_bytes() below needs it. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

#include "k3.h"
#include "k3_bind.h"
#include "k3_cache.h"
#include "k3_trunk.h"
#include "k3_tok.h"   /* text in/out; the --ids path never touches it */
#include "k3_cfg.h"   /* read the checkpoint's own config rather than assuming it */

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void human(double b, char *o, size_t n)
{
    const char *u[] = {"B", "KB", "MB", "GB", "TB"};
    int i = 0; while (b >= 1000.0 && i < 4) { b /= 1000.0; i++; }
    snprintf(o, n, "%.2f %s", b, u[i]);
}

/* The released constants, kept ONLY as a fallback for runs against a shard directory
 * that has no config.json (partial fixtures, hand-assembled trunks). Every value here
 * matches the released config.json, but a hardcoded table cannot notice a checkpoint
 * revision -- so k3_cfg_load_file() is preferred whenever a config is present, and this
 * path announces itself loudly rather than passing for the real thing. */
static void real_cfg_hardcoded(K3Cfg *c, int *fa)
{
    memset(c, 0, sizeof *c);
    c->hidden = 7168;  c->n_layers = 93;   c->vocab = 163840; c->rms_eps = 1e-5f;
    c->kda_heads = 96; c->kda_head_dim = 128; c->conv_k = 4;  c->gate_lb = -5.0f;
    c->n_heads = 96;   c->q_lora = 1536;   c->kv_lora = 512;
    c->qk_nope = 128;  c->qk_rope = 64;    c->v_head = 128;   c->mla_out_gate = 1;
    c->n_experts = 896; c->topk = 16;      c->n_shared = 2;
    c->latent = 3584;  c->moe_inter = 3072; c->routed_scale = 1.0f;
    c->moe_renorm = 1; c->latent_norm = 1;
    c->first_dense = 1; c->dense_inter = 33792;
    c->attn_res_block = 12;
    c->situ_b1 = 4.0f; c->situ_b2 = 25.0f;
    int n = 0;
    for (int i = 4; i <= 93; i += 4) fa[n++] = i;     /* config lists are ONE-based */
    fa[n++] = 93;
    c->n_full_attn = n; c->full_attn = fa;
}

/* Prefer the checkpoint's own config; fall back only when there is none.
 * cfg_path may be NULL, in which case <shard_dir>/config.json is tried.
 * Returns 1 on success, 0 if a config was found but could not be trusted -- and in
 * that case the caller MUST abort rather than fall back: a config that was found but
 * could not be parsed is evidence that the checkpoint is not what the fallback table
 * describes, which is exactly when the fallback is most dangerous. */
static int real_cfg(K3Cfg *c, int *fa, int fa_max,
                    const char *shard_dir, const char *cfg_path)
{
    char guess[4096];
    if (!cfg_path) {
        snprintf(guess, sizeof guess, "%s/config.json", shard_dir);
        FILE *probe = fopen(guess, "rb");
        if (probe) { fclose(probe); cfg_path = guess; }
    }
    if (cfg_path) return k3_cfg_load_file(c, fa, fa_max, cfg_path);

    real_cfg_hardcoded(c, fa);
    printf("config: NO config.json found under %s\n"
           "        falling back to the built-in Kimi K3 constants (93 layers, 24 MLA).\n"
           "        These match the released checkpoint but are NOT read from it; pass\n"
           "        --config PATH to validate against the real file.\n", shard_dir);
    return 1;
}

static int argmax_(const float *v, int n)
{ int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }

/* ---------------------------------------------------------------- perplexity ----
 * ONE RECORD PER PREDICTED POSITION, because a mean cannot answer the question a
 * quantisation change actually raises.
 *
 * The question is not "is the quantised model worse" -- it always is -- but "worse in
 * what way, and by how much against the model it replaces". A perplexity ratio answers
 * neither. It is an average over a heavy-tailed per-token quantity, so 1.13x is equally
 * consistent with a uniform small loss and with an intact model that falls apart on a
 * few percent of positions, and the second is what makes generation collapse while the
 * first does not.
 *
 * These five numbers, aligned position-for-position between a baseline run and a
 * candidate run, give: perplexity (target_lp), the TOP-1 AGREEMENT RATE between the two
 * models (top1), how confidently each was wrong (top1_lp), and whether the distribution
 * is flattening or collapsing (entropy). Agreement in particular is the sharp
 * instrument here -- it compares the candidate against the model it must stand in for,
 * rather than against the text, and it does not average away.
 *
 * 20 bytes a position: a 4,096-position suite is 80 KB. */
typedef struct {
    int32_t target;      /* the id the text actually continues with */
    int32_t top1;        /* the id this model would have predicted   */
    float   target_lp;   /* log p(target), nats. perplexity's numerator */
    float   top1_lp;     /* log p(top1), nats. the model's confidence   */
    float   entropy;     /* H(p) in nats over the full vocabulary       */
} K3PplRec;

typedef struct {
    double    nll;       /* summed negative log-likelihood over counted positions */
    long      n;         /* positions counted                                     */
    K3PplRec *rec;       /* optional, capacity >= total predicted positions        */
    long      nrec;      /* records written so far; runs across documents          */
} K3Ppl;

/* One document of the evaluation suite. Several short documents beat one long one here:
 * attention in the MLA layers is quadratic in sequence length, so doubling a document
 * more than doubles its cost, while the perplexity estimate only needs POSITIONS and
 * does not care how they are grouped. Splitting also gives a per-document breakdown,
 * which is where a register-specific failure (code fine, prose broken) becomes visible
 * instead of being averaged into one number. */
typedef struct {
    char *name;
    int  *ids;
    int   n;
} K3PplDoc;

/* Read an evaluation suite: one document per line, "name<TAB>id,id,id,...".
 * Blank lines and lines beginning with '#' are ignored, so the file can carry a header
 * saying which tokenizer produced it -- ids from a different vocabulary would score as
 * noise and there would be nothing in the numbers to say so.
 *
 * Returns the document array and sets *ndoc / *maxlen; NULL on any failure, having said
 * why. Refuses an empty suite rather than reporting a perplexity over zero positions. */
static K3PplDoc *ppl_load_suite(const char *path, int *ndoc, int *maxlen, int vocab)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "k3: cannot read --ppl-file %s\n", path); return NULL; }

    size_t cap = 16, nd = 0;
    K3PplDoc *d = (K3PplDoc *)calloc(cap, sizeof *d);
    if (!d) { fclose(f); return NULL; }

    char *line = NULL; size_t lcap = 0; ssize_t ln;
    int  longest = 0, lineno = 0, bad = 0;
    while ((ln = getline(&line, &lcap, f)) > 0) {
        lineno++;
        while (ln > 0 && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = 0;
        if (ln == 0 || line[0] == '#') continue;

        char *tab = strchr(line, '\t');
        if (!tab) {
            fprintf(stderr, "k3: %s:%d has no TAB; expected \"name<TAB>ids\"\n", path, lineno);
            bad = 1; break;
        }
        *tab = 0;
        if (nd == cap) {
            cap *= 2;
            K3PplDoc *nx = (K3PplDoc *)realloc(d, cap * sizeof *d);
            if (!nx) { bad = 1; break; }
            d = nx;
            memset(d + nd, 0, (cap - nd) * sizeof *d);
        }
        /* An upper bound on the id count: every id needs at least one separator byte
         * after the first, so the remaining text length always covers it. */
        const size_t room = (size_t)(ln - (tab - line)) + 1;
        d[nd].name = strdup(line);
        d[nd].ids  = (int *)malloc(room * sizeof(int));
        if (!d[nd].name || !d[nd].ids) { bad = 1; break; }
        int n = 0;
        for (const char *p = tab + 1; *p; ) {
            char *end = NULL;
            const long v = strtol(p, &end, 10);
            if (end == p) break;
            if (v < 0 || v >= vocab) {
                fprintf(stderr, "k3: %s:%d (%s) has id %ld outside the vocabulary of %d\n",
                        path, lineno, d[nd].name, v, vocab);
                bad = 1; break;
            }
            d[nd].ids[n++] = (int)v;
            p = end;
            while (*p == ',' || *p == ' ') p++;
        }
        if (bad) break;
        /* One id predicts nothing: perplexity is scored on transitions, so a
         * single-token document contributes zero positions and would silently dilute
         * the document count it is averaged over. */
        if (n < 2) {
            fprintf(stderr, "k3: %s:%d (%s) has %d id(s); a document needs at least 2\n",
                    path, lineno, d[nd].name, n);
            bad = 1; break;
        }
        d[nd].n = n;
        if (n > longest) longest = n;
        nd++;
    }
    free(line);
    fclose(f);

    if (!bad && nd == 0) {
        fprintf(stderr, "k3: --ppl-file %s contains no documents\n", path);
        bad = 1;
    }
    if (bad) {
        for (size_t i = 0; i < nd; i++) { free(d[i].name); free(d[i].ids); }
        free(d);
        return NULL;
    }
    *ndoc = (int)nd;
    *maxlen = longest;
    return d;
}

typedef struct {
    K3LayerBind *lay;
    K3ModelBind  mb;
    int          n_bound;
    K3Trunk     *trunk;      /* non-NULL when the trunk is streamed rather than resident */
    /* Incremental decode state. Only MLA layers need a KV cache, so the 24 of them are
     * numbered densely rather than indexing all 93 and wasting 74% of the allocation. */
    float       *kvc, *ropec;   /* K3_KV_EXPANDED: per-head k/v, and the shared rope row */
    float       *latc;          /* K3_KV_LATENT: the 576-float latent, 42.9x smaller     */
    int          kv_mode;       /* K3_KV_EXPANDED or K3_KV_LATENT                        */
    int          kv_on;         /* 1 once a cache is allocated; selects incremental decode */
    int          kv_slots;      /* latent slots per layer: kv_cap, or the window          */
    int          kv_window, kv_sinks;
    int         *mla_slot;   /* [n_layers] -> dense MLA index, or -1 */
    int          n_mla, kv_cap, cached;
    int          draft_mode;   /* 1 for the hybrid draft: cache-only expert routing */
} Weights;

/* Floats of latent per position per MLA layer: the compressed latent plus the shared
 * rope slot, which is exactly one kv_a output row. */
static size_t kv_latent_width(const K3Cfg *c)
{
    return (size_t)c->kv_lora + (size_t)c->qk_rope;
}

/* Describe this layer's slice of whichever cache is in use. One place builds the
 * descriptor so the two layouts cannot drift apart in the offsets they imply. */
static void kv_for_layer(const Weights *w, const K3Cfg *c, int mi, K3KvCache *kv)
{
    memset(kv, 0, sizeof *kv);
    kv->mode   = w->kv_mode;
    kv->cached = w->cached;
    kv->window = w->kv_window;
    kv->sinks  = w->kv_sinks;
    if (w->kv_mode == K3_KV_LATENT) {
        kv->lat = w->latc + (size_t)mi * (size_t)w->kv_slots * kv_latent_width(c);
        kv->cap = w->kv_slots;
    } else {
        const size_t kvper = (size_t)w->kv_cap * c->n_heads * (c->qk_nope + c->v_head);
        const size_t rpper = (size_t)w->kv_cap * c->qk_rope;
        kv->kv   = w->kvc   + kvper * (size_t)mi;
        kv->rope = w->ropec + rpper * (size_t)mi;
        kv->cap  = w->kv_cap;
    }
}

/* ------------------------------------------------------- conversation state ----
 * Everything the engine carries between tokens, on disk. The point is turn two of a
 * conversation: without this, resuming re-reads the whole prompt through all 93 layers,
 * which on a streamed trunk costs minutes; with it, a resumed session pays only for the
 * tokens actually new.
 *
 * Three things are carried, and only three: the KDA recurrent matrices plus ShortConv
 * history (fixed size, independent of context), the MLA KV cache, and the shared rope
 * rows. The AttnRes block buffer is NOT carried because forward() clears it on entry and
 * rebuilds it from the layer outputs every pass; saving it would be saving scratch.
 *
 * The KV cache is stored position-major inside each MLA layer's slice, so only the
 * OCCUPIED positions are written and a resumed run may size its cache differently. The
 * header carries a config fingerprint: restoring state built by a different architecture
 * would produce fluent, wrong output with nothing to indicate it, which is the one
 * failure mode this engine refuses to have.
 *
 * THE KV LAYOUT IS PART OF THE FINGERPRINT, for the same reason the architecture is.
 * The expanded and latent caches hold different tensors of different widths; a file
 * written by one and read by the other would restore numbers of the right count and the
 * wrong meaning. Version 2 records the layout and the window geometry and refuses a
 * mismatch, rather than trusting the caller to pass the same flags twice. A window also
 * makes the cache a RING, so the slot array is saved whole -- slots are a pure function
 * of absolute position, so restoring them verbatim is exact as long as the geometry
 * agrees, which the header checks. */
#define K3_STATE_MAGIC "K3ST"
#define K3_STATE_VER   2

typedef struct {
    char    magic[4];
    int32_t version;
    int32_t fp[12];        /* config fingerprint */
    int32_t n_bound, n_mla, cached, nseq;
    int64_t kper;          /* KDA+conv floats per layer */
    int64_t kvpp, ropepp;  /* KV / rope floats per position, per MLA layer */
    int32_t kv_mode;       /* K3_KV_EXPANDED or K3_KV_LATENT                 */
    int32_t kv_window, kv_sinks;
    int32_t kv_rows;       /* rows written per MLA layer: positions, or slots */
    int64_t kv_row_floats; /* floats per row: kv_lora + qk_rope in latent mode */
} K3StateHdr;

static void k3_state_fp(const K3Cfg *c, int32_t *fp)
{
    fp[0] = c->hidden;      fp[1] = c->n_layers;  fp[2]  = c->vocab;
    fp[3] = c->kda_heads;   fp[4] = c->kda_head_dim; fp[5] = c->conv_k;
    fp[6] = c->n_heads;     fp[7] = c->qk_nope;   fp[8]  = c->qk_rope;
    fp[9] = c->v_head;      fp[10] = c->n_experts; fp[11] = c->topk;
}

#define K3_SPEC_MAX 8
/* Longest-suffix n-gram drafting for --spec: if the last n ids (n=3, then 2) already
 * appeared earlier in the sequence, propose the ids that followed them there. Costs
 * nothing when it misses: no draft means the step runs exactly as without --spec. The
 * drafts are PROPOSALS only; batched greedy verification accepts precisely the prefix
 * the model itself would have emitted, so the output stream is identical to serial
 * decode by construction, and the A/B gate checks it. */
/* Reads only the header, so the caller can size buffers before committing to a load. */
static int k3_state_peek(const char *path, K3StateHdr *hd)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    const size_t got = fread(hd, 1, sizeof *hd, f);
    fclose(f);
    if (got != sizeof *hd || memcmp(hd->magic, K3_STATE_MAGIC, 4) != 0) {
        fprintf(stderr, "%s is not a k3 state file\n", path);
        return -1;
    }
    if (hd->version != K3_STATE_VER) {
        fprintf(stderr, "%s is state version %d, this build writes %d\n",
                path, hd->version, K3_STATE_VER);
        return -1;
    }
    return 0;
}

/* Rows of KV written per MLA layer, and how wide each row is. A windowed latent cache
 * is a ring, so the whole slot array goes out; everything else writes only the occupied
 * positions and lets a resumed run size its cache differently. */
static int32_t k3_state_rows(const Weights *w)
{
    if (w->kv_mode != K3_KV_LATENT) return w->cached;
    return w->kv_window > 0 ? w->kv_slots
                            : (w->cached < w->kv_slots ? w->cached : w->kv_slots);
}

static int k3_state_load(const char *path, const K3Cfg *c, const K3StateHdr *hd,
                         int *seq, float *ks, Weights *w)
{
    int32_t fp[12];
    k3_state_fp(c, fp);
    if (memcmp(fp, hd->fp, sizeof fp) != 0) {
        fprintf(stderr, "REFUSING: %s was written by a different model architecture.\n"
                        "  Restoring it would produce fluent, wrong output.\n", path);
        return -1;
    }
    if (hd->n_bound != w->n_bound || hd->n_mla != w->n_mla) {
        fprintf(stderr, "REFUSING: %s holds %d bound layers and %d MLA layers, "
                        "this run has %d and %d\n",
                path, hd->n_bound, hd->n_mla, w->n_bound, w->n_mla);
        return -1;
    }
    /* The layout is not a formatting detail: the two caches hold different tensors, so
     * the same float count means different things. Refuse rather than reinterpret. */
    if (hd->kv_mode != w->kv_mode) {
        fprintf(stderr, "REFUSING: %s holds a %s KV cache and this run uses the %s one.\n"
                        "  Re-run with%s --mla-latent, or start a fresh session.\n",
                path, hd->kv_mode == K3_KV_LATENT ? "LATENT" : "expanded",
                w->kv_mode == K3_KV_LATENT ? "LATENT" : "expanded",
                hd->kv_mode == K3_KV_LATENT ? "" : "out");
        return -1;
    }
    if (hd->kv_window != w->kv_window || hd->kv_sinks != w->kv_sinks) {
        /* Slots are a pure function of absolute position GIVEN the geometry. Change the
         * geometry and every restored row lands under a different position. */
        fprintf(stderr, "REFUSING: %s was written with --kv-window %d --kv-sinks %d, "
                        "this run has %d and %d.\n"
                        "  The window decides which slot a position occupies, so the "
                        "rows would be misread.\n",
                path, hd->kv_window, hd->kv_sinks, w->kv_window, w->kv_sinks);
        return -1;
    }
    if (hd->cached > w->kv_cap) {
        fprintf(stderr, "REFUSING: %s holds %d positions, this run's KV cache is %d.\n"
                        "  Raise --gen or shorten the prompt.\n", path, hd->cached, w->kv_cap);
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    if (fseek(f, (long)sizeof *hd, SEEK_SET) != 0) { fclose(f); return -1; }

    int rc = 0;
    if (fread(seq, sizeof(int), (size_t)hd->nseq, f) != (size_t)hd->nseq) rc = -1;
    if (!rc && fread(ks, sizeof(float), (size_t)hd->kper * w->n_bound, f)
               != (size_t)hd->kper * (size_t)w->n_bound) rc = -1;

    if (w->kv_mode == K3_KV_LATENT) {
        const size_t kvw = kv_latent_width(c);
        if (hd->kv_row_floats != (int64_t)kvw || hd->kv_rows > w->kv_slots) {
            fprintf(stderr, "REFUSING: %s holds %d latent rows of %lld floats, this run "
                            "has %d rows of %zu\n",
                    path, hd->kv_rows, (long long)hd->kv_row_floats, w->kv_slots, kvw);
            fclose(f); return -1;
        }
        for (int mi = 0; !rc && mi < w->n_mla; mi++) {
            float *dst = w->latc + (size_t)mi * (size_t)w->kv_slots * kvw;
            const size_t n = (size_t)hd->kv_rows * kvw;
            if (fread(dst, sizeof(float), n, f) != n) rc = -1;
        }
    } else {
        /* Position-major inside each layer slice, so a differently-sized destination
         * cache is written slice by slice rather than as one block. */
        for (int mi = 0; !rc && mi < w->n_mla; mi++) {
            float *dst = w->kvc + (size_t)mi * w->kv_cap * hd->kvpp;
            const size_t n = (size_t)hd->kv_rows * hd->kvpp;
            if (fread(dst, sizeof(float), n, f) != n) rc = -1;
        }
        for (int mi = 0; !rc && mi < w->n_mla; mi++) {
            float *dst = w->ropec + (size_t)mi * w->kv_cap * hd->ropepp;
            const size_t n = (size_t)hd->kv_rows * hd->ropepp;
            if (fread(dst, sizeof(float), n, f) != n) rc = -1;
        }
    }
    fclose(f);
    if (rc) fprintf(stderr, "%s is truncated\n", path);
    return rc;
}

static int k3_state_save(const char *path, const K3Cfg *c, const int *seq, int nseq,
                         const float *ks, const Weights *w, int64_t kper)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    const int64_t kvpp   = (int64_t)c->n_heads * (c->qk_nope + c->v_head);
    const int64_t ropepp = (int64_t)c->qk_rope;
    const size_t  kvw    = kv_latent_width(c);
    const int32_t rows   = k3_state_rows(w);

    K3StateHdr hd;
    memset(&hd, 0, sizeof hd);
    memcpy(hd.magic, K3_STATE_MAGIC, 4);
    hd.version = K3_STATE_VER;
    k3_state_fp(c, hd.fp);
    hd.n_bound = w->n_bound; hd.n_mla = w->n_mla; hd.cached = w->cached; hd.nseq = nseq;
    hd.kper = kper; hd.kvpp = kvpp; hd.ropepp = ropepp;
    hd.kv_mode = w->kv_mode; hd.kv_window = w->kv_window; hd.kv_sinks = w->kv_sinks;
    hd.kv_rows = rows;
    hd.kv_row_floats = (w->kv_mode == K3_KV_LATENT) ? (int64_t)kvw : kvpp + ropepp;

    int rc = 0;
    if (fwrite(&hd, sizeof hd, 1, f) != 1) rc = -1;
    if (!rc && fwrite(seq, sizeof(int), (size_t)nseq, f) != (size_t)nseq) rc = -1;
    if (!rc && fwrite(ks, sizeof(float), (size_t)kper * w->n_bound, f)
               != (size_t)kper * (size_t)w->n_bound) rc = -1;
    if (w->kv_mode == K3_KV_LATENT) {
        for (int mi = 0; !rc && mi < w->n_mla; mi++) {
            const float *src = w->latc + (size_t)mi * (size_t)w->kv_slots * kvw;
            const size_t n = (size_t)rows * kvw;
            if (fwrite(src, sizeof(float), n, f) != n) rc = -1;
        }
    } else {
        for (int mi = 0; !rc && mi < w->n_mla; mi++) {
            const float *src = w->kvc + (size_t)mi * w->kv_cap * kvpp;
            const size_t n = (size_t)rows * kvpp;
            if (fwrite(src, sizeof(float), n, f) != n) rc = -1;
        }
        for (int mi = 0; !rc && mi < w->n_mla; mi++) {
            const float *src = w->ropec + (size_t)mi * w->kv_cap * ropepp;
            const size_t n = (size_t)rows * ropepp;
            if (fwrite(src, sizeof(float), n, f) != n) rc = -1;
        }
    }
    if (fclose(f) != 0) rc = -1;
    if (rc) fprintf(stderr, "failed writing %s\n", path);
    return rc;
}

/* Bytes a saved state occupies, for the line printed after writing it. */
static double k3_state_bytes(const K3Cfg *c, const Weights *w, int nseq, int64_t kper)
{
    const double rows = (double)k3_state_rows(w);
    const double per  = (w->kv_mode == K3_KV_LATENT)
        ? (double)kv_latent_width(c)
        : (double)c->n_heads * (c->qk_nope + c->v_head) + (double)c->qk_rope;
    return (double)sizeof(K3StateHdr) + (double)nseq * sizeof(int)
         + (double)kper * w->n_bound * sizeof(float)
         + rows * per * w->n_mla * sizeof(float);
}

static int spec_draft(const int *seq, int T, int cap, int *out)
{
    /* Evidence-gated: a draft only fires when the suffix n-gram's occurrences AGREE on
     * what follows. Measured on the released checkpoint, an eager most-recent-match
     * drafter went 0.91x on code: partial acceptances pay a replay sweep, so weak
     * drafts are worse than no drafts. Rules: match length 4 (then 3); if the n-gram
     * occurred more than once, every occurrence must propose the same next id, and the
     * draft stops at the first position where historical continuations diverge. */
    if (cap > K3_SPEC_MAX) cap = K3_SPEC_MAX;
    for (int n = 4; n >= 3; n--) {
        if (T < n + 1) continue;
        int m1 = -1, m2 = -1;                            /* two most recent matches */
        for (int j = T - n - 1; j >= 0; j--) {
            int hit = 1;
            for (int i = 0; i < n; i++)
                if (seq[j + i] != seq[T - n + i]) { hit = 0; break; }
            if (!hit) continue;
            if (m1 < 0) m1 = j;
            else { m2 = j; break; }
        }
        if (m1 < 0) continue;
        int nd = 0;
        for (int i = 0; nd < cap && m1 + n + i < T; i++) {
            const int cand = seq[m1 + n + i];
            if (m2 >= 0) {
                /* stop where the two histories stop agreeing */
                if (m2 + n + i >= m1 || seq[m2 + n + i] != cand) break;
            }
            out[nd++] = cand;
        }
        if (nd > 0) return nd;
    }
    return 0;
}

#ifndef K3_VERSION
#define K3_VERSION "1.0.0"
#endif

static void usage(FILE *f)
{
    fprintf(f,
"k3 " K3_VERSION ", Kimi K3 inference engine\n"
"\n"
"usage: k3 <model_dir> [options]\n"
"\n"
"prompt (exactly one):\n"
"  --prompt TEXT         tokenize TEXT and run it\n"
"  --prompt-file PATH    read the prompt from a file; use this for non-ASCII, since\n"
"                        argv is re-encoded by the shell\n"
"  --ids 1,2,3           raw token ids; the reproducible channel used by the tests\n"
"\n"
"memory:\n"
"  --preset NAME         auto | laptop | desktop | workstation | server | max\n"
"                        auto sizes both budgets from this machine's free RAM and from\n"
"                        the trunk manifest, trunk-first: it reads how big the trunk\n"
"                        actually is (a quantised one may fit entirely), computes the\n"
"                        fixed costs from the config, and gives the expert cache either\n"
"                        the floor or a whole working set, never a useless amount in\n"
"                        between. Also spelled --trunk-gb auto\n"
"  --list-presets        show each preset's split and expected speed\n"
"  --trunk DIR           packed trunk directory; enables streaming (see scripts/)\n"
"                        an MXFP4 trunk from tools/mxfp4_trunk.py is ~29 GB instead of\n"
"                        108.81 GB and is detected from its manifest. IT IS NOT THE\n"
"                        RELEASED MODEL: every bit-identity gate is void against it,\n"
"                        and --ppl is what replaces them\n"
"  --trunk-gb X          trunk ring / pinned-layer budget\n"
"  --trunk-ring N        streaming ring slots (default 2). One slot is the layer being\n"
"                        computed on, the rest are reads in flight. A third slot lets\n"
"                        the reader run a layer further ahead and costs one more slot\n"
"                        of RAM; the budget still wins if it does not fit\n"
"  --cache-gb X          routed-expert cache budget\n"
"\n"
"generation:\n"
"  --gen N               tokens to generate (default 8)\n"
"  --incremental         carry KV cache and recurrent state between tokens\n"
"  --mla-latent          cache the 576-float MLA latent instead of expanded per-head\n"
"                        k and v: 55.3 KB per position instead of 2.37 MB, a 42.9x\n"
"                        reduction, so 131k context costs 7.25 GB rather than 310 GB.\n"
"                        Weight absorption keeps the arithmetic equivalent but NOT\n"
"                        bit-identical; needs --incremental\n"
"  --kv-window N         cap the latent cache at N positions. This is LOCAL ATTENTION\n"
"                        beyond N and CHANGES THE OUTPUT; off by default. Needs\n"
"                        --mla-latent\n"
"  --kv-sinks S          of the window, keep the first S positions permanently\n"
"                        (default 4). Attention sinks; only meaningful with --kv-window\n"
"  --save-state PATH     write the carried state after the run, so the next turn of a\n"
"                        conversation resumes instead of re-reading the whole prompt\n"
"  --load-state PATH     resume from a saved state; the prompt given now is treated as\n"
"                        the CONTINUATION of the saved sequence. Needs --incremental\n"
"  --draft-trunk DIR     hybrid decode: a second packed trunk (typically a quantized\n"
"                        derivation of the real one, see tools/qdq_trunk.py) DRAFTS\n"
"                        tokens which the exact model verifies in batched sweeps.\n"
"                        Output remains exactly the exact model's greedy decode; the\n"
"                        draft only proposes. Needs --incremental; implies --spec 4\n"
"  --draft-trunk-gb X    trunk budget for the draft model (default 6)\n"
"  --spec N              speculative decode: draft up to N tokens by n-gram lookup and\n"
"                        verify them in ONE batched sweep. Output is identical to\n"
"                        serial decode by construction; needs --incremental. An extra\n"
"                        verified position costs ~22%% of a serial token when the trunk\n"
"                        streams, so repetitive text decodes up to several times faster\n"
"  --tok DIR             directory with tiktoken.model and tokenizer_config.json\n"
"\n"
"diagnostics:\n"
"  --ppl                 perplexity over the --ids sequence in one teacher-forced\n"
"                        sweep. THE gate for a quantised trunk, which cannot be\n"
"                        byte-compared against anything: run it on the bf16 trunk and\n"
"                        on the quantised one and compare. It is TEACHER-FORCED, so it\n"
"                        cannot see a model that loops in free generation; that needs\n"
"                        the second leg of benchmarks/eval/run_eval.py\n"
"  --ppl-file PATH       score a whole evaluation suite in ONE process: one document\n"
"                        per line, \"name<TAB>id,id,...\". Reports per document and in\n"
"                        total. Amortises the model load, and with a resident trunk\n"
"                        the trunk read too\n"
"  --calib-dump PATH     collect per-input-channel activation statistics over the\n"
"                        suite, for activation-aware quantisation. Implies --ppl;\n"
"                        feed the result to tools/awq_trunk.py. Run it against the\n"
"                        BF16 trunk: the statistics must come from the model being\n"
"                        approximated, not from an approximation of it\n"
"  --ppl-dump PATH       write one 20-byte record per scored position: target id,\n"
"                        top-1 id, both log-probabilities and the entropy. Aligned\n"
"                        position-for-position between two runs this gives the TOP-1\n"
"                        AGREEMENT RATE against a baseline model, which does not\n"
"                        average away the way a perplexity ratio does\n"
"  --config PATH         model config; defaults to <model_dir>/config.json\n"
"  --layers N            bind only the first N layers (partial shard sets)\n"
"  --dump-logits PATH    write float32 logits for the first step\n"
"  --dump-cache-trace D  write expert_hist.json and expert_trace.bin into D, for\n"
"                        offline analysis with tools/sim_cache.py\n"
"  --out FILE            JSON results (default k3_run.json)\n"
"  --version, --help\n"
"\n"
"Memory is a dial, not a floor: the same model runs in 8 GB and in 224 GB and produces\n"
"identical output. Give memory to the trunk before the expert cache, see\n"
"docs/TUNING.md for why, and scripts/k3-doctor.sh to size this machine.\n");
}

/* ------------------------------------------------------------------- presets ----
 * Named memory budgets, so a user does not have to discover the trunk/cache split
 * empirically.
 *
 * The split is not arbitrary and it is not symmetric. Per token the engine re-reads the
 * ENTIRE 108.81 GB trunk but only ~25.8 GB of routed experts, so a gigabyte given to the
 * trunk removes roughly 1.17 GB/token of guaranteed traffic (one pinned layer) while a
 * gigabyte given to the expert cache removes, below about 36 GB of arena, nothing
 * measurable, K3's router is trained for flat expert usage, which defeats an LRU.
 *
 * Measured consequence: at a fixed 128 GB budget, trunk-first runs 1.69x faster than
 * cache-first. So every preset fills the trunk before it feeds the cache.
 * docs/PERFORMANCE.md carries the data and the noise floor that bounds it. */
typedef struct { const char *name; double trunk_gb, cache_gb; const char *note; } K3Preset;

/* The trunk/cache figures are BUDGETS passed to the two allocators. The description
 * quotes measured peak RSS for the whole process, which is the number that decides
 * whether a machine can run the preset, it includes the safetensors index, the KV
 * cache and scratch, none of which appear in either budget. Measured on the reference
 * machine in docs/PERFORMANCE.md; expect a little variation elsewhere. */
static const K3Preset K3_PRESETS[] = {
    { "laptop",      3.0,   1.0,  "8.2 GB peak RSS. The floor. Runs, slowly." },
    { "desktop",    16.0,  10.0,  "31.9 GB peak RSS." },
    { "workstation", 60.0, 30.0,  "95.5 GB peak RSS; the expert cache starts to matter here." },
    { "server",     110.0, 13.0,  "~128 GB peak RSS; 90 of 93 trunk layers pinned. Fastest." },
    { "max",        110.0,109.0,  "~224 GB peak RSS; trunk pinned and a large expert cache." },
};
enum { K3_NPRESET = (int)(sizeof K3_PRESETS / sizeof K3_PRESETS[0]) };

static const K3Preset *k3_preset_find(const char *name)
{
    for (int i = 0; i < K3_NPRESET; i++)
        if (!strcmp(name, K3_PRESETS[i].name)) return &K3_PRESETS[i];
    return NULL;
}

static void k3_preset_list(FILE *f)
{
    fprintf(f, "presets (trunk / expert-cache, in GB):\n");
    for (int i = 0; i < K3_NPRESET; i++)
        fprintf(f, "  %-12s %6.1f / %-6.1f  %s\n", K3_PRESETS[i].name,
                K3_PRESETS[i].trunk_gb, K3_PRESETS[i].cache_gb, K3_PRESETS[i].note);
    fprintf(f, "  %-12s %6s / %-6s  %s\n", "auto", "fit", "fit",
            "sizes both from this machine's free RAM, trunk-first. Recommended.");
    fprintf(f, "\nAll presets stream the trunk, so they need --trunk <packed_dir>.\n"
               "Run scripts/k3-doctor.sh to see which one this machine fits.\n");
}

/* PEAK resident set, in bytes. ru_maxrss is kilobytes on Linux and BYTES on Darwin, so
 * the scale factor differs by platform; applying the Linux one on macOS would overstate
 * the peak by 1024x.
 *
 * This is the authoritative memory figure. The banner printed before allocation is a
 * PLAN and understates: it omits the safetensors index (~78 MB at full scale), reports
 * requested budgets rather than actual reservations, and cannot observe fragmentation.
 * Quote this value, not the plan. */
static double peak_rss_bytes(void)
{
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
#if defined(__APPLE__)
    return (double)ru.ru_maxrss;            /* already bytes */
#else
    return (double)ru.ru_maxrss * 1024.0;   /* kilobytes */
#endif
}

/* MemAvailable, which is what the kernel thinks can actually be handed out, not
 * MemFree. Returns 0 if it cannot be read. */
static double mem_available_bytes(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0.0;
    char line[256];
    double kb = 0.0;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "MemAvailable:", 13)) { kb = atof(line + 13); break; }
    fclose(f);
    return kb * 1024.0;
}

/* One full forward over T tokens, writing logits for the LAST position only. Every
 * step rebuilds state from scratch, matching the path the oracle validates.
 *
 * Returns 0 on success and -1 if the forward could not be completed. The caller MUST
 * check: on failure logits_last is left untouched, and argmaxing an untouched buffer
 * yields a token drawn from uninitialised memory, printed as though it were output. */
/* arg_all: when non-NULL, receives argmax(logits) for EVERY position 0..T-1, which is
 * what batched greedy verification consumes. logits_last still gets the final position's
 * full vector either way. The extra cost is one lm_head matmul per additional position,
 * pure RAM-resident compute; measured, an extra verified position costs ~22% of a serial
 * token at streamed-trunk budgets, which is the entire economics of --spec. */
/* ppl: when non-NULL, accumulate the negative log-likelihood the model assigns to the
 * id the sequence ACTUALLY continues with at each position, summed over positions
 * 0..T-2, and count them. That is perplexity's numerator, and it is the gate that
 * replaces bit-identity when the trunk is quantised: a quantised model cannot be
 * byte-compared against anything, but it can be asked how surprised it is.
 *
 * ppl->rec, when non-NULL, additionally records each position individually. The mean is
 * not enough to compare two models. Perplexity is an average over a heavy-tailed
 * quantity, so a change of a few points is consistent with "slightly worse everywhere"
 * and with "identical almost everywhere and catastrophic on the rare tokens", which are
 * different failures with different fixes. Per position, the two are trivially
 * distinguishable -- and the top-1 id makes an AGREEMENT rate against a baseline run
 * possible, which is a far sharper instrument than the ratio of two means. */
static int forward(Weights *w, const K3Cfg *c, K3Cache *cache, const int *ids, int T,
                   float *logits_last, float *scratch, float *h, float *br, float *kstate,
                   int *arg_all, K3Ppl *ppl)
{
    double *nll  = ppl ? &ppl->nll : NULL;
    long   *nll_n = ppl ? &ppl->n  : NULL;
    const int E = c->hidden;
    const int maxb = c->n_layers / c->attn_res_block + 2;
    const int P = c->kda_heads * c->kda_head_dim;
    const size_t kper = (size_t)P * c->kda_head_dim + (size_t)3 * P * (c->conv_k - 1);

    for (int t = 0; t < T; t++)
        k3_embed_row(h + (size_t)t * E, w->mb.embed, w->mb.wdt, ids[t], E);

    memset(br, 0, (size_t)T * maxb * E * sizeof(float));
    /* Incremental decode carries the KDA recurrent matrix and ShortConv history across
     * steps, so it must NOT be cleared here; the full-recompute path rebuilds from
     * scratch every step and must be. */
    if (!w->kv_on) memset(kstate, 0, kper * (size_t)w->n_bound * sizeof(float));
    int nb = 0;
    for (int L = 0; L < w->n_bound; L++) {
        /* Streaming: bring this layer in, and hint the next one so its read overlaps
         * this layer's arithmetic. The order is fixed 0..92 every token, so the hint is
         * never wrong. */
        if (w->trunk) {
            if (k3_trunk_bind(w->trunk, c, L, &w->lay[L]) != 0) {
                fprintf(stderr, "trunk bind failed at layer %d\n", L);
                return -1;
            }
            k3_trunk_prefetch(w->trunk, L + 1);
        }
        /* Point this layer's MoE at the cache before use. Doing it here rather than at
         * bind time keeps K3LayerBind independent of any particular cache. */
        if (w->lay[L].lay.moe) {
            w->lay[L].moe.src = &cache->src;
            w->lay[L].moe.layer = L;
            /* The draft routes only among resident experts, reading zero new expert bytes;
             * the exact model keeps true routing. This is what makes a draft step cheap. */
            w->lay[L].moe.cache_only = w->draft_mode;
        }
        if (w->kv_on && w->mla_slot[L] >= 0) {
            K3KvCache kv;
            kv_for_layer(w, c, w->mla_slot[L], &kv);
            k3_decoder_layer_kv(h, br, &nb, &w->lay[L].lay, c, L, T,
                                kstate + kper * (size_t)L, scratch, &kv);
        } else {
            k3_decoder_layer_kv(h, br, &nb, &w->lay[L].lay, c, L, T,
                                kstate + kper * (size_t)L, scratch, NULL);
        }
    }

    /* The model-level aggregator, beyond the two per layer. Exactly one pair exists in
     * the checkpoint; skipping it is silent. */
    if (w->mb.out_res_norm && w->mb.out_res_proj) {
        float *fold = scratch;
        float *src  = fold + E;
        for (int i = 0; i < E; i++) fold[i] = w->mb.out_res_norm[i] * w->mb.out_res_proj[i];
        for (int t = 0; t < T; t++) {
            for (int b = 0; b < nb; b++)
                memcpy(src + (size_t)b * E, br + ((size_t)t * maxb + b) * E,
                       (size_t)E * sizeof(float));
            memcpy(src + (size_t)nb * E, h + (size_t)t * E, (size_t)E * sizeof(float));
            k3_attn_res(h + (size_t)t * E, src, fold, nb + 1, E, c->rms_eps);
        }
    }

    float *nrm = scratch;
    if (arg_all || nll) {
        for (int t = 0; t < T; t++) {
            k3_rmsnorm(nrm, h + (size_t)t * E, w->mb.norm, E, c->rms_eps);
            k3_mmw(logits_last, nrm, w->mb.lm_head, w->mb.wdt, E, c->vocab);
            if (arg_all) arg_all[t] = argmax_(logits_last, c->vocab);
            if (nll && t + 1 < T) {
                /* log-softmax the stable way: subtract the max before exponentiating.
                 * A 163,840-wide vocabulary with logits in the tens overflows expf()
                 * without it, and the result would be inf rather than wrong-by-a-little. */
                int am = 0;
                float m = logits_last[0];
                for (int i = 1; i < c->vocab; i++)
                    if (logits_last[i] > m) { m = logits_last[i]; am = i; }
                /* One pass for both the partition function and sum(p*logit), so the
                 * entropy costs no extra sweep of a 163,840-wide vector. */
                double z = 0.0, zx = 0.0;
                for (int i = 0; i < c->vocab; i++) {
                    const double e = exp((double)logits_last[i] - m);
                    z += e; zx += e * ((double)logits_last[i] - m);
                }
                const int tgt = ids[t + 1];
                const double lz = log(z);
                const double tlp = ((double)logits_last[tgt] - m) - lz;
                *nll -= tlp;
                if (nll_n) (*nll_n)++;
                if (ppl->rec) {
                    K3PplRec *r = &ppl->rec[ppl->nrec++];
                    r->target   = tgt;
                    r->top1     = am;
                    r->target_lp = (float)tlp;
                    r->top1_lp   = (float)(-lz);   /* logit_am - m is 0 by construction */
                    /* H = log z - E_p[logit - m], in nats. A model that has collapsed
                     * onto a short loop shows this falling toward zero, which is the
                     * one teacher-forced signal that anticipates the degeneration the
                     * free-running leg measures directly. */
                    r->entropy   = (float)(lz - zx / z);
                }
            }
        }
        /* logits_last now holds the FINAL position's vector, same as the plain path. */
        return 0;
    }
    k3_rmsnorm(nrm, h + (size_t)(T - 1) * E, w->mb.norm, E, c->rms_eps);
    k3_mmw(logits_last, nrm, w->mb.lm_head, w->mb.wdt, E, c->vocab);
    return 0;
}

int main(int argc, char **argv)
{
    /* Informational flags are answered before anything else, because they must work
     * without a model directory, `k3 --help` on a machine with no checkpoint is the
     * first thing most people type. */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(stdout); return 0; }
        if (!strcmp(argv[i], "--version")) { printf("k3 %s\n", K3_VERSION); return 0; }
        if (!strcmp(argv[i], "--list-presets")) { k3_preset_list(stdout); return 0; }
    }
    if (argc < 2) { usage(stderr); return 2; }

    const char *dir = argv[1];
    if (dir[0] == '-') {
        fprintf(stderr, "the first argument must be the model directory, got '%s'\n\n", dir);
        usage(stderr);
        return 2;
    }
    const char *ids_s = NULL, *outp = "k3_run.json", *trunk_dir = NULL;
    /* Expert-cache diagnostics are opt-in. They are only meaningful for cache research,
     * and writing them unconditionally drops two undeclared files into whatever
     * directory the user happened to run from. */
    const char *trace_dir = NULL;
    const char *logits_path = NULL;
    const char *prompt_text = NULL, *prompt_file = NULL, *tok_dir = NULL;
    const char *cfg_path = NULL;
    int gen = 8, want_layers = -1;
    double cache_gb = 64.0, trunk_gb = 16.0;
    int budget_auto = 0;
    int spec_n = 0;
    int tf_check = 0, want_ppl = 0, gen_set = 0;
    const char *ppl_file = NULL, *ppl_dump = NULL, *calib_dump = NULL;
    const char *draft_dir = NULL;
    double draft_gb = 6.0;
    const char *load_state = NULL, *save_state = NULL;
    const char *preset_name = NULL;
    int incremental = 0;
    int mla_latent = 0, kv_window = 0, kv_sinks = 4;
    int trunk_ring = 0;   /* 0 selects k3_trunk_open's default of 2 */
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_s = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt_text = argv[++i];
        else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) prompt_file = argv[++i];
        else if (!strcmp(argv[i], "--tok") && i + 1 < argc) tok_dir = argv[++i];
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) { gen = atoi(argv[++i]); gen_set = 1; }
        else if (!strcmp(argv[i], "--cache-gb") && i + 1 < argc) cache_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) want_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
        else if (!strcmp(argv[i], "--trunk") && i + 1 < argc) trunk_dir = argv[++i];
        else if (!strcmp(argv[i], "--spec") && i + 1 < argc) spec_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tf-check")) tf_check = 1;
        else if (!strcmp(argv[i], "--ppl")) want_ppl = 1;
        else if (!strcmp(argv[i], "--ppl-file") && i + 1 < argc) { ppl_file = argv[++i]; want_ppl = 1; }
        else if (!strcmp(argv[i], "--ppl-dump") && i + 1 < argc) ppl_dump = argv[++i];
        else if (!strcmp(argv[i], "--calib-dump") && i + 1 < argc) { calib_dump = argv[++i]; want_ppl = 1; }
        else if (!strcmp(argv[i], "--load-state") && i + 1 < argc) load_state = argv[++i];
        else if (!strcmp(argv[i], "--save-state") && i + 1 < argc) save_state = argv[++i];
        else if (!strcmp(argv[i], "--draft-trunk") && i + 1 < argc) draft_dir = argv[++i];
        else if (!strcmp(argv[i], "--draft-trunk-gb") && i + 1 < argc) draft_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--trunk-gb") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "auto")) budget_auto = 1;
            else { trunk_gb = atof(v); budget_auto = 0; }
        }
        else if (!strcmp(argv[i], "--trunk-ring") && i + 1 < argc) trunk_ring = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--incremental")) incremental = 1;
        else if (!strcmp(argv[i], "--mla-latent")) mla_latent = 1;
        else if (!strcmp(argv[i], "--kv-window") && i + 1 < argc) kv_window = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv-sinks") && i + 1 < argc) kv_sinks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) logits_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-cache-trace") && i + 1 < argc) trace_dir = argv[++i];
        else if (!strcmp(argv[i], "--preset") && i + 1 < argc && !strcmp(argv[i + 1], "auto")) {
            /* Not in the table: the table is fixed budgets, auto is computed from this
             * machine's MemAvailable at startup, below, once parsing is complete. */
            i++;
            budget_auto = 1;
            preset_name = "auto";
        }
        else if (!strcmp(argv[i], "--preset") && i + 1 < argc) {
            const K3Preset *p = k3_preset_find(argv[++i]);
            if (!p) {
                fprintf(stderr, "unknown preset '%s'\n\n", argv[i]);
                k3_preset_list(stderr);
                return 2;
            }
            /* A preset sets the budget; an explicit --trunk-gb/--cache-gb after it still
             * wins, because the flags are applied in argv order. */
            trunk_gb = p->trunk_gb;
            cache_gb = p->cache_gb;
            preset_name = p->name;
        }
        /* --list-presets, --version and --help/-h are NOT handled here: the pre-scan
         * above already walks every element of argv for these three, in any position,
         * and returns before this loop starts. A branch for them here could never run. */
        else { fprintf(stderr, "unknown option %s\n\n", argv[i]); usage(stderr); return 2; }
    }
    {
        int nsrc = (ids_s != NULL) + (prompt_text != NULL) + (prompt_file != NULL);
        /* --ppl-file carries its own ids for every document, so it IS the prompt source.
         * Combining it with another one would leave the other silently unused. */
        if (ppl_file && nsrc > 0) {
            fprintf(stderr, "--ppl-file supplies its own ids; drop --ids/--prompt/--prompt-file\n");
            return 2;
        }
        if (nsrc == 0 && !ppl_file) {
            fprintf(stderr, "one of --ids, --prompt, --prompt-file or --ppl-file is required\n");
            return 2;
        }
        if (nsrc > 1) {
            /* Refuse rather than pick: silently preferring one source would make a
             * mistyped invocation run the WRONG prompt for tens of minutes. */
            fprintf(stderr, "--ids, --prompt and --prompt-file are mutually exclusive\n");
            return 2;
        }
    }


    /* fa is sized for the released 24 MLA layers with generous headroom; k3_cfg_load
     * refuses a config that would overrun it rather than truncating the layer map. */
    K3Cfg c; static int fa[128];
    if (!real_cfg(&c, fa, 128, dir, cfg_path)) {
        fprintf(stderr, "ABORTED: the model config could not be read with confidence.\n");
        return 2;
    }
    if (want_layers > 0 && want_layers < c.n_layers) {
        printf("NOTE: binding only the first %d of %d layers. Output is NOT the full "
               "model; it is a partial stack for testing the machinery.\n\n",
               want_layers, c.n_layers);
    }

    /* ---- prompt ----
     * Three entry points, one representation. --ids is the reproducible channel every
     * fixture and the oracle use, and stays the default for validation work. --prompt /
     * --prompt-file tokenize here in C, which is what makes the engine text-in/text-out
     * without a Python step. The tokenizer is loaded ONLY when actually needed, so the
     * id path keeps working on a box that has no tokenizer files at all. */
    /* Heap, not stack. This was `int prompt[4096]` and it was the reason the engine
     * refused prompts longer than 4096 ids -- a stack-array size, not a model or memory
     * limit. */
    int *prompt = (int *)malloc((size_t)K3_MAX_PROMPT * sizeof(int));
    if (!prompt) { fprintf(stderr, "OOM allocating prompt buffer\n"); return 2; }
    int np = 0;
    Tok tok; int have_tok = 0;

    /* The evaluation suite is loaded HERE, before anything is sized, because every
     * scratch buffer below is allocated from np and the suite's longest document is what
     * they have to hold. Loading it later would size the run for one document and then
     * run a longer one through the same buffers. */
    K3PplDoc *docs = NULL; int ndoc = 0, doc_max = 0; long doc_pos = 0;
    if (ppl_file) {
        docs = ppl_load_suite(ppl_file, &ndoc, &doc_max, c.vocab);
        if (!docs) return 2;
        for (int i = 0; i < ndoc; i++) doc_pos += docs[i].n - 1;
        if (doc_max > K3_MAX_PROMPT) {
            fprintf(stderr, "k3: --ppl-file document of %d ids exceeds the %d-id ceiling\n",
                    doc_max, K3_MAX_PROMPT);
            return 2;
        }
        /* Stand the longest document in as the prompt so every size below is derived
         * from the largest sweep that will actually run. */
        np = doc_max;
        for (int i = 0; i < ndoc; i++)
            if (docs[i].n == doc_max) { memcpy(prompt, docs[i].ids, (size_t)np * sizeof(int)); break; }
        printf("  eval suite: %d documents, %ld scored positions, longest %d ids (%s)\n",
               ndoc, doc_pos, doc_max, ppl_file);
    }

    if (ppl_file) {
        /* already populated */
    } else if (prompt_text || prompt_file) {
        if (!tok_dir) {
            fprintf(stderr, "--prompt/--prompt-file need --tok DIR (the directory with "
                            "tiktoken.model and tokenizer_config.json)\n");
            return 2;
        }
        k3_tok_load(&tok, tok_dir);
        have_tok = 1;

        char *ptext = NULL; long plen = 0;
        if (prompt_file) {
            ptext = tk_read_file(prompt_file, &plen);   /* exits if unreadable */
        } else {
            plen  = (long)strlen(prompt_text);
            ptext = (char *)malloc((size_t)plen + 1);
            if (!ptext) { fprintf(stderr, "OOM on prompt\n"); return 2; }
            memcpy(ptext, prompt_text, (size_t)plen + 1);
        }
        np = tok_encode(&tok, ptext, (int)plen, prompt, K3_MAX_PROMPT);
        free(ptext);
        printf("  tokenized: %ld bytes -> %d ids\n", plen, np);
    } else {
        for (const char *p = ids_s; *p && np < K3_MAX_PROMPT; ) {
            prompt[np++] = (int)strtol(p, (char **)&p, 10);
            while (*p == ',' || *p == ' ') p++;
        }
    }
    if (np == 0) { fprintf(stderr, "no prompt ids parsed\n"); return 2; }
    for (int i = 0; i < np; i++)
        if (prompt[i] < 0 || prompt[i] >= c.vocab) {
            fprintf(stderr, "token id %d is outside the vocabulary of %d\n", prompt[i], c.vocab);
            return 2;
        }

    /* Validate the request before allocating anything.
     *
     * Refuse rather than clamp: a caller who asks for more tokens than this build
     * supports should be told, not quietly handed fewer. The decode loop's own guard
     * (T >= Tmax) is a backstop, not a bounds check. */
    if (gen < 0 || gen > K3_MAX_GEN) {
        fprintf(stderr, "--gen %d is out of range: this build generates at most %d "
                        "tokens (outtok[%d])\n", gen, K3_MAX_GEN, K3_MAX_GEN);
        return 2;
    }
    if (np > K3_MAX_PROMPT) {
        fprintf(stderr, "prompt of %d ids exceeds the %d-id ceiling (seq[%d])\n",
                np, K3_MAX_PROMPT, K3_MAX_PROMPT + K3_MAX_GEN);
        return 2;
    }
    if (np + gen + 1 > K3_MAX_PROMPT + K3_MAX_GEN) {
        fprintf(stderr, "prompt %d + gen %d + 1 exceeds the %d-position ceiling\n",
                np, gen, K3_MAX_PROMPT + K3_MAX_GEN);
        return 2;
    }
    /* THE REAL CONTEXT LIMIT is the MLA KV cache, not any array size. Check it against
     * what the kernel says is actually available and refuse with both numbers, rather
     * than letting a long prompt get 40 minutes into a run and then be OOM-killed. Only
     * incremental decode allocates the KV cache; full recompute carries no cache. */
    /* ---- KV layout, and the flags that only make sense together ---- */
    if (mla_latent && !incremental) {
        fprintf(stderr, "--mla-latent needs --incremental: without a carried cache "
                        "there is no KV layout to choose\n");
        return 2;
    }
    if (kv_window > 0 && !mla_latent) {
        /* The window is a property of the latent cache. Applying it to the expanded one
         * would work arithmetically and save 42.9x less memory for the same loss of
         * context, which is not a trade worth offering. */
        fprintf(stderr, "--kv-window needs --mla-latent\n");
        return 2;
    }
    if (kv_window > 0 && (kv_sinks < 0 || kv_sinks >= kv_window)) {
        fprintf(stderr, "--kv-sinks %d must be in [0, --kv-window %d)\n",
                kv_sinks, kv_window);
        return 2;
    }
    if (kv_window > 0 && (spec_n > 0 || draft_dir)) {
        /* REFUSED, and this is a correctness limit rather than a policy.
         *
         * Speculation writes drafted positions into the cache and then, on a partial
         * acceptance, REWINDS and replays only the accepted prefix. In an unbounded
         * cache the rejected rows sit past `cached` and are simply never read. In a
         * windowed one the slots wrap: a rejected position p lands in the slot of
         * position p - R, and p - R is exactly the oldest position the replay still has
         * to attend over. The replay would then read a rejected draft's latent as
         * history -- not the approximation the window openly is, but a corruption on top
         * of it, and one that depends on how many drafts happened to be rejected. */
        fprintf(stderr,
            "--kv-window cannot be combined with --spec or --draft-trunk.\n"
            "  A rejected draft's rows wrap onto positions the replay still needs, so\n"
            "  the rewind would read them as history. Drop one of the two.\n");
        return 2;
    }
    if (incremental) {
        const int npos = np + gen + 1;
        const double per_pos = mla_latent ? K3_KV_LATENT_BYTES_PER_POS
                                          : K3_KV_BYTES_PER_POS;
        const int held = (kv_window > 0 && kv_window < npos) ? kv_window : npos;
        const double kv_need = (double)held * per_pos;
        const double avail   = mem_available_bytes();
        char kb[32], ab[32];
        human(kv_need, kb, sizeof kb);
        human(avail, ab, sizeof ab);
        printf("  KV cache : %s for %d positions (%.2f KB/position, %s layout)\n",
               kb, held, per_pos / 1e3, mla_latent ? "LATENT" : "expanded");
        if (mla_latent)
            printf("             absorbed: W_UK folded onto the query and W_UV applied to "
                   "the latent attention sum,\n"
                   "             so per-head k and v are never materialised. NOT "
                   "bit-identical to the expanded path.\n");
        if (kv_window > 0 && kv_window < npos)
            printf("             WINDOWED at %d positions (+%d sinks inside it): "
                   "attention beyond the window is\n"
                   "             not computed at all, so THE OUTPUT DIFFERS from an "
                   "unwindowed run. This is a memory\n"
                   "             ceiling, not a speed feature. See docs/TUNING.md.\n",
                   kv_window, kv_sinks);
        if (avail > 0.0 && kv_need > avail * 0.9) {
            fprintf(stderr,
                "\nREFUSING: the KV cache for %d positions needs %s but only %s is\n"
                "available. This is a MEMORY limit, not an engine ceiling: MLA caches\n"
                "%s across 24 layers, so context costs ~%.2f %s per\n"
                "position regardless of budget. %sShorten the request, or use full\n"
                "recompute (drop --incremental), which carries no KV cache at all.\n",
                held, kb, ab,
                mla_latent ? "the compressed latent in fp32"
                           : "expanded k and v in fp32",
                mla_latent ? per_pos / 1e3 : per_pos / 1e6,
                mla_latent ? "KB" : "MB",
                mla_latent ? "" : "Try --mla-latent, which is 42.9x smaller. ");
            return 2;
        }
    }

    char b1[32];
    printf("Kimi K3, pure C, released checkpoint\n");
    /* The directory, not a shard count: the index has not been built yet at this point.
     * The count is printed by the "indexed N tensors from M shards" line below, once
     * k3_st_open has actually counted them. */
    printf("  model    : %s\n", dir);
    printf("  prompt   : %d tokens, generating %d\n", np, gen);
    /* Echo the preset so a captured log is self-describing: a timing figure is
     * meaningless without the budget that produced it. */
    /* Named presets are fixed numbers and can be echoed here. `auto` is not decided
     * until the checkpoint and the trunk manifest have been read, several steps below,
     * so echoing it here printed the DEFAULTS and called them the preset. It prints its
     * own line, with the reasoning, once it actually knows. */
    if (preset_name && strcmp(preset_name, "auto") != 0)
        printf("  preset   : %s (trunk %.1f GB / expert cache %.1f GB)\n",
               preset_name, trunk_gb, cache_gb);
    printf("\n");

    K3St st;
    double t0 = now_s();
    if (k3_st_open(&st, dir) != 0) return 1;
    printf("indexed %d tensors from %d shards in %.2f s\n", st.nt, st.nshard, now_s() - t0);

    /* ---- auto budget ----------------------------------------------------------------
     * RAM-first, and now sized against the trunk actually being streamed rather than
     * against a constant. Per token the engine re-reads the ENTIRE trunk but only
     * ~25.8 GB of experts, so a gigabyte pinned in the trunk is worth roughly 70x a
     * gigabyte of expert cache at the margin. Auto therefore fills the trunk first, and
     * feeds the cache only with what is genuinely left over.
     *
     * Three things it now computes instead of assuming, each of which was a constant
     * that stopped being true:
     *
     *   THE TRUNK'S SIZE. This was hardcoded at 111 GB, the bf16 trunk plus headroom. A
     *   quantised trunk (tools/mxfp4_trunk.py) is 29.81 GB and fits entirely in RAM on a
     *   machine where auto would previously have concluded it could not, and given up on
     *   full residency -- which is the one configuration where per-token trunk I/O stops
     *   happening rather than merely getting faster. k3_trunk_probe reads the manifest.
     *
     *   THE FIXED COSTS. embed + lm_head was hardcoded at 4.70 GB and the recurrent state
     *   at 1.70 GB. Both follow from the config, and both are wrong for any model that is
     *   not the released one. The KV cache was not counted at all, which is fine at
     *   2.37 MB/position for a short run and is not fine for a long one.
     *
     *   WHERE THE EXPERT CACHE STARTS PAYING. Below one token's working set -- topk
     *   experts in each MoE layer, 25.8 GB on the released model -- the hit rate does not
     *   move with capacity at all: measured flat from 4 GB to 24 GB on the recorded trace
     *   (tools/sim_cache.py). So a cache of 8 GB buys exactly what a cache of 0.5 GB buys,
     *   and the difference belongs in the trunk. Auto now gives the cache either the
     *   minimum or at least a whole working set, and never a useless amount in between. */
    if (budget_auto) {
        const double avail = mem_available_bytes();
        if (avail <= 0.0) {
            fprintf(stderr, "--preset auto needs /proc/meminfo; pass explicit "
                            "--trunk-gb/--cache-gb on this platform\n");
            return 2;
        }
        const int64_t E64 = c.hidden;
        const int Tm = np + gen + 1;
        const int Pp = c.kda_heads * c.kda_head_dim;
        const int mb_ = c.n_layers / c.attn_res_block + 2;

        /* Everything that lives outside both budgets, from the config rather than from
         * memory of one checkpoint. */
        const double w_model = 2.0 * (double)c.vocab * E64 * 2 + 3.0 * E64 * 4;
        const double w_state = (double)((size_t)Pp * c.kda_head_dim
                             + (size_t)3 * Pp * (c.conv_k - 1)) * c.n_layers * 4;
        const double w_buf = ((double)Tm * E64 + (double)Tm * mb_ * E64
                            + (double)k3_layer_scratch(&c, Tm) + (double)c.vocab) * 4;
        int n_mla_ = 0, n_moe_ = 0;
        for (int L = 0; L < c.n_layers; L++) {
            if (k3_is_mla(&c, L)) n_mla_++;
            if (!k3_is_dense(&c, L)) n_moe_++;
        }
        const int kv_pos = (kv_window > 0 && kv_window < Tm) ? kv_window : Tm;
        const double w_kv = !incremental ? 0.0
            : mla_latent ? (double)kv_pos * n_mla_ * (c.kv_lora + c.qk_rope) * 4
                         : (double)Tm * n_mla_
                           * ((double)c.n_heads * (c.qk_nope + c.v_head) + c.qk_rope) * 4;

        /* 2 GB + 2% margin on top, so auto never invites the OOM killer. */
        const double reserve = (w_model + w_state + w_buf + w_kv) / 1e9
                             + 2.0 + 0.02 * (avail / 1e9);
        double usable = avail / 1e9 - reserve;
        const double cache_min = 0.5;   /* topk+1 expert slots is ~0.3 GB */

        /* What full residency would actually cost, from the manifest. */
        double trunk_full = 111.0;      /* no manifest: assume the released bf16 trunk */
        int quantised = 0;
        if (trunk_dir) {
            int64_t packed = 0; int nlay = 0;
            if (k3_trunk_probe(trunk_dir, &packed, &nlay, &quantised) == 0 && packed > 0)
                trunk_full = ((double)packed
                            + (double)nlay * (double)k3_bind_widen_bytes(&c)) / 1e9 + 0.5;
        }
        const double slot_min = 2.5;    /* one ring slot + headroom; refuse below */
        if (usable < slot_min + cache_min) {
            fprintf(stderr, "auto: only %.1f GB usable after the %.1f GB reserve; "
                            "below the %.1f GB floor. Pass explicit budgets.\n",
                    usable, reserve, slot_min + cache_min);
            return 2;
        }

        /* One token's expert working set, the threshold below which cache capacity does
         * not move the hit rate. Measured from the checkpoint, not assumed. */
        int64_t ebytes = 17547264;
        {
            K3ExpertRef pr;
            for (int L = 0; L < c.n_layers; L++) {
                if (k3_is_dense(&c, L)) continue;
                if (k3_expert_ref(&st, L, 0, &pr) == 0) { ebytes = pr.nbytes; break; }
            }
        }
        const double ws_gb = (double)c.topk * n_moe_ * (double)ebytes / 1e9;

        const char *why;
        if (usable - cache_min >= trunk_full) {
            /* Full residency: per-token trunk reads disappear entirely. This is the
             * configuration auto exists for, and a quantised trunk reaches it on
             * hardware a bf16 one never could. */
            trunk_gb = trunk_full;
            cache_gb = usable - trunk_full;
            why = "trunk fully resident: no per-token trunk read at all";
        } else {
            /* Partial pinning has WEAK returns and real hazards, both measured on the
             * released checkpoint: pinning 51 of 109 GB ran 14% SLOWER than pinning
             * nothing (48.2 vs 42.1 s/token) because peak RSS at ~90% of RAM put the
             * kernel into reclaim and the device served the remaining tail of the packed
             * trunk a third slower, while a moderate pin stayed neutral to mildly
             * positive (40.1 s/token at 25 GB). So below full residency, auto pins only
             * while the whole process stays comfortably clear of the RAM ceiling. */
            double memtotal = 0.0;
            FILE *mf = fopen("/proc/meminfo", "r");
            if (mf) {
                char ln[256];
                while (fgets(ln, sizeof ln, mf))
                    if (!strncmp(ln, "MemTotal:", 9)) { memtotal = atof(ln + 9) * 1024.0; break; }
                fclose(mf);
            }
            const double rss_ceiling = memtotal > 0.0 ? 0.55 * memtotal / 1e9 : usable;
            double cap = rss_ceiling - reserve - cache_min;
            if (cap < slot_min) cap = slot_min;

            /* Spend on the cache ONLY if a whole working set is affordable; otherwise
             * every gigabyte is worth more in the trunk, and the leftover the trunk
             * cannot use goes to the cache regardless because it has nowhere better. */
            if (usable - cap >= ws_gb) {
                trunk_gb = cap;
                cache_gb = usable - cap;
                why = "trunk at its safe pin ceiling, and the rest clears one expert "
                      "working set";
            } else {
                trunk_gb = usable - cache_min;
                if (trunk_gb > cap) trunk_gb = cap;
                cache_gb = cache_min;
                /* Whatever the RAM ceiling stopped the trunk taking is LEFT
                 * UNALLOCATED, deliberately. Handing it to the cache instead would
                 * spend it where the measurement says it buys nothing -- below one
                 * working set the hit rate does not move with capacity -- while
                 * pushing peak RSS back through the ceiling the cap exists to keep
                 * clear of. That ceiling is not a guess: pinning to ~90% of RAM ran
                 * 14% SLOWER than pinning nothing, because reclaim cost the device a
                 * third of its throughput. Idle RAM is the cheaper outcome. */
                why = "cache at its floor and the trunk at the RAM ceiling: below one "
                      "working set, cache capacity does not move the hit rate";
            }
        }
        printf("auto budget: %.1f GB available, %.1f GB reserved (model %.1f + state %.1f "
               "+ KV %.2f + buffers %.2f)\n"
               "             -> trunk %.1f GB / expert cache %.1f GB\n"
               "             %s\n",
               avail / 1e9, reserve, w_model / 1e9, w_state / 1e9, w_kv / 1e9,
               w_buf / 1e9, trunk_gb, cache_gb, why);
        printf("             trunk needs %.1f GB to be fully resident%s; one expert "
               "working set is %.1f GB\n",
               trunk_full, quantised ? " (QUANTISED trunk)" : "", ws_gb);
        {
            const double idle = usable - trunk_gb - cache_gb;
            if (idle > 0.5)
                printf("             %.1f GB deliberately left unallocated: peak RSS near "
                       "the RAM ceiling puts the\n"
                       "             kernel into reclaim, which measured 14%% SLOWER than "
                       "pinning less\n", idle);
        }
    }

    /* ---- how much will this take? Report BEFORE allocating, so a box that cannot
     * hold it fails with a number rather than an OOM kill. ---- */
    const int NL = (want_layers > 0 && want_layers < c.n_layers) ? want_layers : c.n_layers;
    int64_t total = 0; int missing = 0;
    for (int L = 0; L < NL; L++) {
        const int64_t n = k3_bind_layer_bytes(&st, &c, L);
        if (n < 0) { missing++; continue; }
        total += n;
    }
    if (missing) {
        fprintf(stderr, "\n%d of %d layers are missing tensors in this shard set. "
                        "A partial download cannot run the model.\n", missing, NL);
        return 1;
    }
    human((double)total, b1, sizeof b1);
    /* Report the mode actually in effect. The trunk is either resident or streamed and
     * the two have very different memory profiles, so the banner must reflect the real
     * choice rather than a default. */
    if (trunk_dir)
        printf("trunk on disk : %s total (STREAMED from %s, not held in RAM)\n",
               b1, trunk_dir);
    else
        printf("resident trunk: %s in RAM (large matrices kept in the checkpoint's bf16,\n"
               "  fp32 only for the norms and biases that kernels read elementwise)\n", b1);

    /* Add up EVERYTHING before allocating anything. Being OOM-killed halfway through
     * binding wastes the whole load and reports nothing useful; a refusal with the two
     * numbers side by side says exactly what box this needs. */
    {
        const int64_t E64 = c.hidden;
        const double w_trunk = trunk_dir ? trunk_gb * 1e9 : (double)total;
        const double w_model = 2.0 * (double)c.vocab * E64 * 2    /* embed + lm_head, bf16 */
                             + 3.0 * E64 * 4;                     /* norms, aggregator */
        const double w_cache = cache_gb * 1e9;
        const int Tm = np + gen + 1;
        const int mb = c.n_layers / c.attn_res_block + 2;
        const int Pp = c.kda_heads * c.kda_head_dim;
        const double w_state = (double)((size_t)Pp * c.kda_head_dim
                              + (size_t)3 * Pp * (c.conv_k - 1)) * NL * 4;
        const double w_buf = ((double)Tm * E64 + (double)Tm * mb * E64
                              + (double)k3_layer_scratch(&c, Tm) + (double)c.vocab) * 4;
        /* The KV cache MUST be in this total: it is the only term that grows with
         * context, so a guard that omits it is blind to the one thing it exists to
         * catch. k3_mla_cached stores expanded per-head k and v plus the shared rope
         * slot, in fp32, across all 24 MLA layers -- 2.37 MB per position, so a
         * 4096-token prompt alone is 9.7 GB. */
        int n_mla = 0;
        for (int L = 0; L < c.n_layers; L++) if (k3_is_mla(&c, L)) n_mla++;
        const int kv_pos = (kv_window > 0 && kv_window < Tm) ? kv_window : Tm;
        const double w_kv = !incremental ? 0.0
            : mla_latent
              ? (double)kv_pos * n_mla * (double)(c.kv_lora + c.qk_rope) * 4
              : (double)Tm * n_mla
                * ((double)c.n_heads * (c.qk_nope + c.v_head) + c.qk_rope) * 4;
        const double need_b = w_trunk + w_model + w_cache + w_state + w_buf + w_kv;
        const double have = mem_available_bytes();

        char b2[32], b3[32], b4[32], b5[32], b6[32], b7[32];
        human(w_kv, b7, sizeof b7);
        human(w_trunk, b1, sizeof b1); human(w_model, b2, sizeof b2);
        human(w_cache, b3, sizeof b3); human(w_state, b4, sizeof b4);
        human(w_buf, b5, sizeof b5);   human(need_b, b6, sizeof b6);
        printf("\nmemory plan\n");
        printf("  trunk %-10s %s\n  embed + lm_head  %s\n  expert cache     %s\n"
               "  recurrent state  %s\n  buffers          %s\n  KV cache         %s\n"
               "  TOTAL            %s\n",
               trunk_dir ? "(STREAMED)" : "(resident)", b1, b2, b3, b4, b5, b7, b6);
        if (have > 0.0) {
            human(have, b1, sizeof b1);
            printf("  available        %s\n", b1);
            /* A 5% headroom, because "available" is a snapshot and the run has to survive
             * the whole token, not just the allocation. Say which of the two reasons is
             * refusing, though: the message used to print need - have unconditionally,
             * which is NEGATIVE for a plan that fits and only fails the margin, and
             * "a shortfall of -788330496.00 B" tells the reader nothing they can act on. */
            const double margin = have * 0.95;
            if (need_b > margin) {
                const double over = need_b - margin;
                human(over, b2, sizeof b2);
                if (need_b > have) {
                    char b8[32]; human(need_b - have, b8, sizeof b8);
                    fprintf(stderr,
                            "\nREFUSING TO START: this needs %s and the machine has %s "
                            "available, a shortfall of %s.\n", b6, b1, b8);
                } else {
                    fprintf(stderr,
                            "\nREFUSING TO START: this needs %s and the machine has %s "
                            "available.\nIt fits, but only just: the guard keeps 5%% "
                            "headroom so a run does not get OOM-killed\npart way through, "
                            "and this is %s over that line.\n", b6, b1, b2);
                }
                /* Give the number to change, not just the name of the knob. */
                if (w_cache > over)
                    fprintf(stderr, "Lower --cache-gb by at least %.2f GB (to about "
                                    "%.2f GB) and it will start.\n",
                            over / 1e9, (w_cache - over) / 1e9);
                fprintf(stderr,
                        "Options: a larger box, a smaller --cache-gb, or fewer --layers.\n");
                return 1;
            }
        }
        printf("\n");
    }

    Weights w; memset(&w, 0, sizeof w);
    w.lay = (K3LayerBind *)calloc((size_t)NL, sizeof(K3LayerBind));
    if (!w.lay) return 1;

    static K3Trunk trunk;
    t0 = now_s();
    if (trunk_dir) {
        /* STREAMED. Nothing is bound up front: each layer is read from the packed trunk
         * on fast local storage as the forward pass reaches it. RAM stops being a floor
         * and becomes a dial, and unlike quantisation it costs no accuracy, which
         * matters because the K3 report (4.1.4) keeps exactly these tensors in higher
         * precision on purpose. */
        if (k3_trunk_open(&trunk, trunk_dir, &c, (int64_t)(trunk_gb * 1e9),
                          trunk_ring) != 0) return 1;
        if (trunk.n_layers < NL) {
            fprintf(stderr, "packed trunk has %d layers, need %d\n", trunk.n_layers, NL);
            return 1;
        }
        w.trunk = &trunk;
        w.n_bound = NL;
        printf("trunk streaming enabled from %s in %.1f s\n", trunk_dir, now_s() - t0);
    } else {
        for (int L = 0; L < NL; L++) {
            if (k3_bind_layer(&st, &c, L, &w.lay[L]) != 0) {
                fprintf(stderr, "bind failed at layer %d\n", L); return 1;
            }
            w.n_bound = L + 1;
            if ((L + 1) % 10 == 0 || L + 1 == NL) {
                printf("  bound %d/%d layers, %.1f s elapsed\n", L + 1, NL, now_s() - t0);
                fflush(stdout);
            }
        }
        const double t_bind = now_s() - t0;
        printf("trunk loaded in %.1f s (%.0f MB/s from disk)\n",
               t_bind, (double)total / 1e6 / t_bind);
    }

    t0 = now_s();
    if (k3_bind_model(&st, &c, 1, &w.mb) != 0) return 1;
    human((double)w.mb.nbytes, b1, sizeof b1);
    printf("embedding, final norm and lm_head: %s in %.1f s\n\n", b1, now_s() - t0);

    K3Cache cache;
    if (k3_cache_init(&cache, &st, &c, (int64_t)(cache_gb * 1e9)) != 0) return 1;
    {   /* The plan is a forecast. This is the outcome. */
        char rb[32];
        human(peak_rss_bytes(), rb, sizeof rb);
        printf("peak RSS after loading weights: %s  (the plan above is a forecast, "
               "this is measured)\n", rb);
    }
    printf("expert cache: %d slots x %.2f MB = %.2f GB (%.2f%% of the 1.45 TB expert pool)\n\n",
           cache.nslot, (double)cache.slot_bytes / 1e6,
           (double)cache.nslot * cache.slot_bytes / 1e9,
           100.0 * cache.nslot / (double)(92 * c.n_experts));

    /* ---- buffers ----
     * A resumed session must hold the saved history as well as the new tokens, so the
     * KV cache and every per-position buffer are sized for both. The header is read
     * here, before anything is allocated; the payload is restored after. */
    K3StateHdr shd;
    int prior = 0;
    if (load_state) {
        if (!incremental) {
            fprintf(stderr, "--load-state needs --incremental\n");
            return 2;
        }
        if (k3_state_peek(load_state, &shd) != 0) return 1;
        prior = shd.nseq;
        printf("resuming from %s: %d prior positions, %d new\n\n", load_state, prior, np);
    }
    const int Tmax = prior + np + gen + 1;
    const int E = c.hidden;
    const int maxb = c.n_layers / c.attn_res_block + 2;
    const int P = c.kda_heads * c.kda_head_dim;
    const size_t kper = (size_t)P * c.kda_head_dim + (size_t)3 * P * (c.conv_k - 1);

    /* The model-level aggregator lays out fold[E] followed by (nb+1) source rows of E
     * inside scratch, and nb reaches n_layers/attn_res_block = 8 at full depth. So
     * scratch must hold at least (maxb + 2) * hidden floats. k3_layer_scratch includes
     * exactly that term, but an off-by-one here would overwrite whatever follows
     * without any symptom until the logits came out subtly wrong, so it is checked
     * rather than assumed. */
    {
        const size_t need_scratch = (size_t)(maxb + 2) * E;
        const size_t have_scratch = k3_layer_scratch(&c, Tmax);
        if (have_scratch < need_scratch) {
            fprintf(stderr, "scratch is %zu floats, the attn-res aggregator needs %zu\n",
                    have_scratch, need_scratch);
            return 1;
        }
    }

    float *h  = (float *)malloc((size_t)Tmax * E * sizeof(float));
    float *br = (float *)malloc((size_t)Tmax * maxb * E * sizeof(float));
    float *ks = (float *)malloc(kper * (size_t)NL * sizeof(float));
    /* Size for the layout that will actually run. The latent path carries H absorbed
     * queries and a latent-space accumulator that the expanded path has no use for, so
     * sizing a latent run with the expanded figure under-allocates silently. */
    size_t sc_need = k3_layer_scratch_kv(&c, Tmax, Tmax,
                                         mla_latent ? K3_KV_LATENT : K3_KV_EXPANDED);
    {   /* the cached MLA path sizes its score buffer by cache capacity, not by T */
        const size_t ic = k3_mla_scratch_cached(&c, Tmax, Tmax, 1);
        if (ic > sc_need) sc_need = ic;
    }
    float *sc = (float *)malloc(sc_need * sizeof(float));
    float *lg = (float *)malloc((size_t)c.vocab * sizeof(float));
    if (!h || !br || !ks || !sc || !lg) { fprintf(stderr, "buffer allocation failed\n"); return 1; }
    human((double)(kper * NL) * 4, b1, sizeof b1);
    printf("recurrent state for %d layers: %s\n\n", NL, b1);

    /* ---- generate ----
     * Heap and sized from the ACTUAL request, not from the ceiling. These were
     * `int seq[K3_MAX_PROMPT + K3_MAX_GEN]` and `int outtok[K3_MAX_GEN]` on the stack,
     * which is why the ceiling had to stay small enough to be a stack array. */
    int *seq = (int *)malloc((size_t)(prior + np + gen + 8) * sizeof(int));
    int *outtok = (int *)malloc((size_t)(gen + 8) * sizeof(int));
    if (!seq || !outtok) { fprintf(stderr, "OOM allocating sequence buffers\n"); return 1; }
    /* On a resume the saved history occupies the front of the sequence and the prompt
     * given now is its continuation; the restore below fills seq[0..prior). */
    memcpy(seq + prior, prompt, (size_t)np * sizeof(int));
    int T = prior + np;
    int nout = 0;

    /* ---- optional incremental decode ----
     * Full recompute re-runs the whole prefix every step, so expert traffic grows with
     * context: measured 99.7 -> 126.0 GB across just three tokens. Incremental prefills
     * once and then feeds ONE token per step, carrying the KDA recurrent state (which
     * k3_kda_layer already updates in place) and an MLA KV cache. Validated by GATE 3
     * of the tiny-model oracle, which requires the SAME tokens as full recompute. */
    if (incremental) {
        w.mla_slot = (int *)malloc((size_t)NL * sizeof(int));
        if (!w.mla_slot) return 1;
        w.n_mla = 0;
        for (int L = 0; L < NL; L++)
            w.mla_slot[L] = k3_is_mla(&c, L) ? w.n_mla++ : -1;
        w.kv_cap = Tmax;
        w.kv_mode = mla_latent ? K3_KV_LATENT : K3_KV_EXPANDED;
        w.kv_window = kv_window;
        w.kv_sinks  = kv_window > 0 ? kv_sinks : 0;
        /* A window that is not smaller than the run is no window at all; dropping it
         * here keeps the fast, modulo-free slot path and stops the banner claiming a
         * restriction that is not in force. */
        if (w.kv_window >= Tmax) { w.kv_window = 0; w.kv_sinks = 0; }
        w.kv_slots = w.kv_window > 0 ? w.kv_window : w.kv_cap;

        if (w.kv_mode == K3_KV_LATENT) {
            const size_t kvw = kv_latent_width(&c);
            const size_t lper = (size_t)w.kv_slots * kvw;
            human((double)lper * w.n_mla * sizeof(float), b1, sizeof b1);
            printf("incremental decode: LATENT KV cache %s for %d MLA layers at %d "
                   "positions\n", b1, w.n_mla, w.kv_slots);
            {   /* the same run in the expanded layout, so the saving is on the record */
                const double exp_b = (double)w.kv_cap * w.n_mla
                    * ((double)c.n_heads * (c.qk_nope + c.v_head) + c.qk_rope)
                    * sizeof(float);
                char b8[32]; human(exp_b, b8, sizeof b8);
                printf("                    the expanded layout would be %s for the same "
                       "run (%.1fx)\n\n", b8,
                       exp_b / ((double)lper * w.n_mla * sizeof(float)));
            }
            w.latc = (float *)calloc(lper * (size_t)w.n_mla, sizeof(float));
            if (!w.latc) { fprintf(stderr, "KV cache allocation failed\n"); return 1; }
        } else {
            const size_t kvper = (size_t)w.kv_cap * c.n_heads * (c.qk_nope + c.v_head);
            const size_t rpper = (size_t)w.kv_cap * c.qk_rope;
            const double kvb = (double)(kvper + rpper) * w.n_mla * sizeof(float);
            human(kvb, b1, sizeof b1);
            printf("incremental decode: KV cache %s for %d MLA layers at %d positions\n\n",
                   b1, w.n_mla, w.kv_cap);
            w.kvc   = (float *)calloc(kvper * (size_t)w.n_mla, sizeof(float));
            w.ropec = (float *)calloc(rpper * (size_t)w.n_mla, sizeof(float));
            if (!w.kvc || !w.ropec) { fprintf(stderr, "KV cache allocation failed\n"); return 1; }
        }
        w.kv_on = 1;
        memset(ks, 0, kper * (size_t)NL * sizeof(float));
        w.cached = 0;

        if (load_state) {
            const double tl = now_s();
            if (k3_state_load(load_state, &c, &shd, seq, ks, &w) != 0)
                return 1;
            w.cached = shd.cached;
            printf("restored %d positions in %.2f s: decode continues without "
                   "re-reading the prior context\n\n", w.cached, now_s() - tl);
        }
    }

    /* --spec needs a snapshot of the carried KDA/ShortConv state to roll back a
     * partially-rejected draft batch: the recurrent state is updated in place and is
     * not positional, so the only sound recovery is restore-and-replay the accepted
     * prefix. The snapshot is one memcpy; the replay is one short batched sweep. */
    const size_t kperP  = (size_t)c.kda_heads * c.kda_head_dim;
    const size_t kper_f = kperP * c.kda_head_dim + 3 * kperP * (c.conv_k - 1);
    float *spec_snap = NULL;
    if (spec_n > 0) {
        if (!incremental) {
            fprintf(stderr, "--spec needs --incremental; ignoring --spec\n");
            spec_n = 0;
        } else {
            if (spec_n > K3_SPEC_MAX) spec_n = K3_SPEC_MAX;
            spec_snap = (float *)malloc(kper_f * (size_t)w.n_bound * sizeof(float));
            if (!spec_snap) { fprintf(stderr, "OOM for the --spec snapshot\n"); return 1; }
            printf("speculative decode: up to %d drafted tokens per sweep, n-gram lookup, "
                   "verified batched\n\n", spec_n);
        }
    }

    /* ---- hybrid decode: a second, typically quantized, trunk drafts ----
     * The draft model shares everything that is identical between the two models: the
     * embedding, the lm_head, the layer map, and the routed experts (the qdq derivation
     * touches only 2D trunk tensors). It differs ONLY in trunk weights, so it needs its
     * own trunk stream, its own layer bindings, and its own recurrent/KV state. Output
     * exactness is structural: drafts feed the SAME batched greedy verification as
     * --spec, so what gets emitted is precisely what the exact model would have chosen.
     * Measured teacher-forced agreement of an int8-derived draft on the released
     * checkpoint is 94.2 percent against a 96.2 percent measurement ceiling, which is
     * what makes the draft worth consulting at all. */
    static K3Trunk trunk_d;
    Weights dw; memset(&dw, 0, sizeof dw);
    float *dks = NULL, *dsnap = NULL;
    long hyb_rounds = 0, hyb_drafted = 0, hyb_accepted = 0;
    if (draft_dir) {
        if (!incremental || !trunk_dir) {
            fprintf(stderr, "--draft-trunk needs --incremental and --trunk; ignoring\n");
            draft_dir = NULL;
        } else {
            if (spec_n <= 0) spec_n = 4;
            if (spec_n > K3_SPEC_MAX) spec_n = K3_SPEC_MAX;
            if (!spec_snap) {
                spec_snap = (float *)malloc(kper_f * (size_t)w.n_bound * sizeof(float));
                if (!spec_snap) { fprintf(stderr, "OOM for the --spec snapshot\n"); return 1; }
            }
            if (k3_trunk_open(&trunk_d, draft_dir, &c, (int64_t)(draft_gb * 1e9),
                              trunk_ring) != 0)
                return 1;
            dw.lay = (K3LayerBind *)calloc((size_t)NL, sizeof(K3LayerBind));
            dks   = (float *)calloc(kper_f * (size_t)w.n_bound, sizeof(float));
            dsnap = (float *)malloc(kper_f * (size_t)w.n_bound * sizeof(float));
            /* The draft carries its own KV in the SAME layout as the exact model, so a
             * latent run drafts against a latent cache and the memory saving applies to
             * both halves of the hybrid rather than being eaten by the draft. */
            dw.kv_mode   = w.kv_mode;
            dw.kv_window = w.kv_window;
            dw.kv_sinks  = w.kv_sinks;
            dw.kv_slots  = w.kv_slots;
            dw.kv_cap    = w.kv_cap;
            if (w.kv_mode == K3_KV_LATENT) {
                const size_t lperd = (size_t)w.kv_slots * kv_latent_width(&c);
                dw.latc = (float *)calloc(lperd * (size_t)w.n_mla, sizeof(float));
                if (!dw.latc) { fprintf(stderr, "OOM for the draft model state\n"); return 1; }
            } else {
                const size_t kvperd = (size_t)w.kv_cap * c.n_heads * (c.qk_nope + c.v_head);
                const size_t rpperd = (size_t)w.kv_cap * c.qk_rope;
                dw.kvc   = (float *)calloc(kvperd * (size_t)w.n_mla, sizeof(float));
                dw.ropec = (float *)calloc(rpperd * (size_t)w.n_mla, sizeof(float));
                if (!dw.kvc || !dw.ropec) {
                    fprintf(stderr, "OOM for the draft model state\n"); return 1;
                }
            }
            if (!dw.lay || !dks || !dsnap) {
                fprintf(stderr, "OOM for the draft model state\n"); return 1;
            }
            dw.mb = w.mb;              /* embed + lm_head are the same tensors */
            dw.trunk = &trunk_d;
            dw.n_bound = w.n_bound;
            dw.mla_slot = w.mla_slot;  /* read-only map, safely shared */
            dw.n_mla = w.n_mla;
            dw.kv_on = 1;
            dw.cached = 0;
            dw.draft_mode = 1;   /* cache-only routing: draft tokens read no new experts */
            printf("hybrid decode: draft trunk %s (%.1f GB budget) proposes up to %d "
                   "tokens per sweep;\n               the exact model verifies every one "
                   "before it is emitted\n\n", draft_dir, draft_gb, spec_n);
        }
    }

    /* --ppl: perplexity over the whole --ids sequence in ONE teacher-forced sweep.
     *
     * THIS IS THE GATE THAT REPLACES BIT-IDENTITY. Everything else in this project is
     * validated by producing exactly the same bytes as a reference: the op fixtures, the
     * oracle, the memory-ladder claim that output is identical at 8 GB and at 224 GB. A
     * quantised trunk (tools/mxfp4_trunk.py, bound as K3_WMX4) breaks all of it by
     * construction, because it is not the released model's weights any more.
     *
     * What can still be measured is how surprised the model is by real text. Run this
     * against the bf16 trunk and against the quantised one on the SAME ids; the
     * difference is the whole of what quantisation cost. A single number, but it is a
     * number, which is more than "it still emits fluent text" ever was. */
    if (want_ppl) {
        /* ONE PROCESS, EVERY DOCUMENT. Each document is its own teacher-forced sweep,
         * but they share a model load -- and, when the trunk is resident, share the
         * trunk entirely. That is the difference between a suite that costs one warm-up
         * and one that pays it per document.
         *
         * Single-document --ppl is the same path with a suite of one, so there is one
         * implementation of the arithmetic rather than two that can drift apart. */
        K3PplDoc solo;
        if (!docs) {
            if (np < 2) { fprintf(stderr, "--ppl needs at least 2 ids\n"); return 2; }
            solo.name = (char *)"stdin"; solo.ids = seq; solo.n = np;
            docs = &solo; ndoc = 1; doc_pos = np - 1;
        }

        /* Collection spans the WHOLE suite: the scale vector wants statistics over as
         * much varied text as can be afforded, and per-document scales would be
         * meaningless -- a trunk has one set of weights. */
        if (calib_dump && k3_calib_begin(c.n_layers) != 0) {
            fprintf(stderr, "OOM starting calibration\n");
            return 1;
        }

        K3Ppl pp; memset(&pp, 0, sizeof pp);
        if (ppl_dump) {
            pp.rec = (K3PplRec *)malloc((size_t)doc_pos * sizeof(K3PplRec));
            if (!pp.rec) { fprintf(stderr, "OOM for --ppl-dump (%ld records)\n", doc_pos); return 1; }
        }

        const double t0p = now_s();
        double *dnll = (double *)calloc((size_t)ndoc, sizeof(double));
        long   *dpos = (long *)calloc((size_t)ndoc, sizeof(long));
        if (!dnll || !dpos) { fprintf(stderr, "OOM for the per-document totals\n"); return 1; }

        for (int di = 0; di < ndoc; di++) {
            const double n0 = pp.nll; const long p0 = pp.n;
            /* Each document is scored from a clean slate: no carried KV cache and no
             * carried recurrent state, so document i's score cannot depend on document
             * i-1 having been run first. forward() clears the recurrent state itself
             * whenever kv_on is false, which is the case here -- --ppl is a full
             * recompute by construction. */
            if (forward(&w, &c, &cache, docs[di].ids, docs[di].n, lg, sc, h, br, ks,
                        NULL, &pp) != 0) {
                fprintf(stderr, "forward failed on document %s\n", docs[di].name);
                return 1;
            }
            dnll[di] = pp.nll - n0;
            dpos[di] = pp.n - p0;
            if (ndoc > 1) {
                const double dm = dpos[di] ? dnll[di] / (double)dpos[di] : 0.0;
                printf("  %-28s %6ld pos   ppl %10.4f   NLL %7.4f\n",
                       docs[di].name, dpos[di], exp(dm), dm);
                fflush(stdout);
            }
        }

        const long npos = pp.n;
        const double mean = npos ? pp.nll / (double)npos : 0.0;
        printf("perplexity: %.4f over %ld positions in %d document(s) "
               "(mean NLL %.4f nats) in %.1f s\n",
               exp(mean), npos, ndoc, mean, now_s() - t0p);

        /* SAY WHEN THE NUMBER IS TOO SMALL TO USE. Per-token NLL varies by whole nats
         * between positions, so the standard error on a short sequence is large enough
         * to swallow the effect this figure exists to measure -- a few points of
         * perplexity between a bf16 trunk and a quantised one. A tool that prints four
         * decimal places on twenty samples and says nothing is inviting the reader to
         * believe all four. */
        if (npos < 512)
            printf("  CAUTION: %ld positions is too few to conclude anything. Per-token\n"
                   "  NLL varies by whole nats, so the error on this mean is comparable\n"
                   "  to the difference you are probably trying to measure. Use a few\n"
                   "  thousand ids of held-out text before quoting it.\n", npos);
        /* Only when it was actually ASKED for. gen defaults to 8, so testing gen > 0
         * printed this on every --ppl run and trained the reader to skip it. */
        if (gen_set && gen > 0)
            printf("  NOTE: --gen %d was ignored. --ppl is one teacher-forced sweep over\n"
                   "  the ids given; it generates nothing.\n", gen);
        printf("  weights  : %s\n",
               w.mb.wdt == K3_WBF16 ? "bf16 embed/lm_head"
                                    : "fp32 embed/lm_head");
        printf("  compare this against the SAME ids on a bf16 trunk. A quantised trunk\n"
               "  cannot be byte-compared against anything, so this difference is the\n"
               "  entire measurement. See docs/notes/compressed-trunk.md.\n");
        /* AND SAY WHAT PERPLEXITY CANNOT SEE, because this measurement has already been
         * quoted past its range once. It is TEACHER-FORCED: every position is scored
         * against the real text, so the model is never conditioned on its own output and
         * a collapse into a repeating loop -- the failure mode a quantised trunk actually
         * produced here -- leaves this number almost unchanged. Free-running generation
         * is a separate leg, and benchmarks/eval/run_eval.py runs both. */
        if (!ppl_dump)
            printf("  NOTE: this is TEACHER-FORCED and cannot see degeneration. A model\n"
                   "  that loops in free generation scores nearly the same here. Run\n"
                   "  benchmarks/eval/run_eval.py for both legs.\n");

        if (ppl_dump && pp.rec) {
            FILE *df = fopen(ppl_dump, "wb");
            if (!df) { fprintf(stderr, "k3: cannot write --ppl-dump %s\n", ppl_dump); return 1; }
            if (fwrite(pp.rec, sizeof(K3PplRec), (size_t)pp.nrec, df) != (size_t)pp.nrec) {
                fprintf(stderr, "k3: short write to %s\n", ppl_dump);
                fclose(df); return 1;
            }
            fclose(df);
            printf("  wrote %ld per-position records to %s (%zu bytes each)\n",
                   pp.nrec, ppl_dump, sizeof(K3PplRec));
        }

        if (calib_dump) {
            if (k3_calib_write(calib_dump) != 0) {
                fprintf(stderr, "k3: cannot write --calib-dump %s\n", calib_dump);
                k3_calib_end();
                return 1;
            }
            k3_calib_end();
            printf("  wrote activation calibration for %d layers to %s\n"
                   "  next: python3 tools/awq_trunk.py <bf16 trunk> <out trunk> --calib %s\n",
                   c.n_layers, calib_dump, calib_dump);
        }

        FILE *pf = fopen(outp, "w");
        if (pf) {
            fprintf(pf, "{\"ppl_positions\":%ld,\"mean_nll\":%.6f,\"perplexity\":%.6f,"
                        "\"documents\":[", npos, mean, exp(mean));
            long at = 0;
            for (int di = 0; di < ndoc; di++) {
                const double dm = dpos[di] ? dnll[di] / (double)dpos[di] : 0.0;
                /* rec_off lets a reader slice the dump per document without re-deriving
                 * the boundaries from document lengths it would have to trust. */
                fprintf(pf, "%s{\"name\":\"%s\",\"positions\":%ld,\"rec_off\":%ld,"
                            "\"mean_nll\":%.6f,\"perplexity\":%.6f}",
                        di ? "," : "", docs[di].name, dpos[di], at, dm, exp(dm));
                at += dpos[di];
            }
            fprintf(pf, "],\"teacher_forced\":true}\n");
            fclose(pf);
            printf("  wrote %s\n", outp);
        }
        free(dnll); free(dpos); free(pp.rec);
        return 0;
    }

    /* --tf-check: teacher-forced agreement over the whole --ids sequence in ONE sweep.
     * Prediction i is the argmax after positions 0..i; it is compared to the id the
     * sequence actually continues with. This is the acceptance rate a draft model
     * would see under batched greedy verification, measured directly, and it is the
     * one number a quantized-draft design stands on. Free-running comparisons cannot
     * measure it: a single early divergence changes every later context. */
    if (tf_check) {
        if (np < 2) { fprintf(stderr, "--tf-check needs at least 2 ids\n"); return 2; }
        int *arg = (int *)malloc((size_t)np * sizeof(int));
        if (!arg) { fprintf(stderr, "OOM for --tf-check\n"); return 1; }
        const double t0c = now_s();
        if (forward(&w, &c, &cache, seq, np, lg, sc, h, br, ks, arg, NULL) != 0) {
            fprintf(stderr, "forward failed in --tf-check\n");
            return 1;
        }
        int match = 0;
        for (int i = 0; i + 1 < np; i++) match += (arg[i] == seq[i + 1]);
        printf("teacher-forced agreement: %d/%d positions (%.1f%%) in %.1f s\n",
               match, np - 1, 100.0 * match / (np - 1), now_s() - t0c);
        printf("  per-position (p=predicted a=actual): ");
        for (int i = 0; i + 1 < np; i++)
            if (arg[i] != seq[i + 1])
                printf("[%d p=%d a=%d] ", i, arg[i], seq[i + 1]);
        printf("\n");
        FILE *tf = fopen(outp, "w");
        if (tf) {
            fprintf(tf, "{\"tf_positions\":%d,\"tf_matches\":%d,\"tf_agreement\":%.4f}\n",
                    np - 1, match, (double)match / (np - 1));
            fclose(tf);
        }
        free(arg);
        return 0;
    }

    printf("%-6s %-10s %-12s %-10s %-10s %s\n",
           "STEP", "TOKEN", "SECONDS", "CACHE HIT", "READ GB", "TOK/S");
    printf("--------------------------------------------------------------------\n");
    double t_total = 0.0;
    /* Per-step cache statistics are reset each iteration so the columns below describe
     * that step alone. The end-of-run summary needs whole-run totals, so accumulate the
     * expert side here; the trunk side is already cumulative. Comparing a cumulative
     * figure against a single step would misstate the I/O share. */
    double expert_s_total = 0.0, expert_gb_total = 0.0;
    uint64_t expert_reqs_total = 0, expert_evict_total = 0;
    for (int g = 0; nout < gen; g++) {
        k3_cache_reset_stats(&cache);
        const double ts = now_s();
        int frc;
        int emit[K3_SPEC_MAX + 1];
        int emitn = 0;
        if (incremental && g == 0) {
            /* Step 0 feeds everything not yet consumed: the whole prompt on a fresh
             * run, and on a resume the carried pending token PLUS the new prompt.
             * T - base covers both exactly; feeding np here instead dropped the last
             * new token from a resumed batch, and the first generated token then came
             * from a context one token short: fluent, plausible, and wrong. */
            const int base = w.cached;
            const int nT0 = T - base;
            frc = forward(&w, &c, &cache, seq + base, nT0, lg, sc, h, br, ks, NULL, NULL);
            if (frc == 0) { w.cached = base + nT0; emit[emitn++] = argmax_(lg, c.vocab); }
            /* The draft model must absorb the same context, or its first proposals
             * come from a shorter one; one draft sweep, paid once. Saved state does
             * not include the draft's, so a resumed run replays the WHOLE sequence
             * through the draft once; correctness never depends on this, only
             * acceptance does. */
            if (dw.trunk && frc == 0) {
                const int db = load_state ? 0 : base;
                if (forward(&dw, &c, &cache, seq + db, base + nT0 - db, lg, sc, h, br,
                            dks, NULL, NULL) == 0)
                    dw.cached = base + nT0;
                else frc = -1;
            }
        } else if (incremental) {
            const int base = w.cached;
            int d[K3_SPEC_MAX], nd = 0;
            if (spec_snap && T + spec_n + 1 < Tmax && base + spec_n + 1 <= w.kv_cap) {
                if (dw.trunk) {
                    /* The draft model proposes: k sequential one-token steps through
                     * the draft trunk, chaining its own argmax. Its state is
                     * snapshotted first so a partial acceptance can rewind it the
                     * same way the exact side rewinds. */
                    memcpy(dsnap, dks, kper_f * (size_t)w.n_bound * sizeof(float));
                    int prev = seq[base];
                    while (nd < spec_n) {
                        if (forward(&dw, &c, &cache, &prev, 1, lg, sc, h, br,
                                    dks, NULL, NULL) != 0) break;
                        dw.cached += 1;
                        prev = argmax_(lg, c.vocab);
                        d[nd++] = prev;
                    }
                    hyb_rounds  += 1;
                    hyb_drafted += nd;
                } else {
                    nd = spec_draft(seq, T, spec_n, d);
                }
            }
            if (nd > 0) {
                /* One sweep verifies the pending token plus nd drafts. arg[i] is the
                 * model's own next token after batch position i; the accepted prefix is
                 * exactly what serial decode would have emitted, and arg[m] after it is
                 * clean because its context contains only accepted tokens. */
                int arg[K3_SPEC_MAX + 1];
                memcpy(spec_snap, ks, kper_f * (size_t)w.n_bound * sizeof(float));
                for (int i = 0; i < nd; i++) seq[T + i] = d[i];
                frc = forward(&w, &c, &cache, seq + base, nd + 1, lg, sc, h, br, ks, arg, NULL);
                if (frc == 0) {
                    int m = 0;
                    while (m < nd && arg[m] == d[m]) m++;
                    if (m == nd) {
                        /* every fed position had true context; state is exact */
                        w.cached = base + nd + 1;
                    } else {
                        /* the recurrent state absorbed rejected tokens: restore, then
                         * replay only the accepted prefix. The replay also rewrites the
                         * KV rows those positions touched, so nothing stale survives. */
                        memcpy(ks, spec_snap, kper_f * (size_t)w.n_bound * sizeof(float));
                        w.cached = base;
                        frc = forward(&w, &c, &cache, seq + base, m + 1, lg, sc, h, br,
                                      ks, NULL, NULL);
                        if (frc == 0) w.cached = base + m + 1;
                    }
                    /* Resync the draft model to the ACCEPTED sequence. On full
                     * acceptance its state already contains every fed token except
                     * the last draft, so one step closes the gap; on partial
                     * acceptance it rewinds to its snapshot and replays only the
                     * accepted prefix, mirroring the exact side. */
                    if (dw.trunk && frc == 0) {
                        hyb_accepted += m;
                        if (m == nd) {
                            int last = d[nd - 1];
                            if (forward(&dw, &c, &cache, &last, 1, lg, sc, h, br,
                                        dks, NULL, NULL) == 0) dw.cached += 1;
                            else frc = -1;
                        } else {
                            memcpy(dks, dsnap, kper_f * (size_t)w.n_bound * sizeof(float));
                            dw.cached = base;
                            if (forward(&dw, &c, &cache, seq + base, m + 1, lg, sc,
                                        h, br, dks, NULL, NULL) == 0) dw.cached = base + m + 1;
                            else frc = -1;
                        }
                    }
                    if (frc == 0) {
                        for (int i = 0; i < m; i++) emit[emitn++] = d[i];
                        emit[emitn++] = arg[m];
                    }
                }
            } else {
                frc = forward(&w, &c, &cache, seq + base, 1, lg, sc, h, br, ks, NULL, NULL);
                if (frc == 0) { w.cached = base + 1; emit[emitn++] = argmax_(lg, c.vocab); }
                /* keep the draft in lockstep through non-drafted steps */
                if (dw.trunk && frc == 0) {
                    if (forward(&dw, &c, &cache, seq + base, 1, lg, sc, h, br,
                                dks, NULL, NULL) == 0) dw.cached = base + 1;
                    else frc = -1;
                }
            }
        } else {
            frc = forward(&w, &c, &cache, seq, T, lg, sc, h, br, ks, NULL, NULL);
            if (frc == 0) emit[emitn++] = argmax_(lg, c.vocab);
        }
        /* Abort the run rather than argmax a buffer the forward never wrote. */
        if (frc != 0 || emitn == 0) {
            fprintf(stderr, "forward pass failed at generation step %d; aborting.\n", g);
            return 1;
        }
        const int nxt = emit[emitn - 1];
        /* Dump the FIRST step's logits as raw float32 bits.
         * Comparing generated tokens against a reference only compares argmax, which
         * hides near-ties: two engines can agree on every token while disagreeing
         * substantially on the logit vector behind it. tools/ref_forward.py produces the
         * same vector from the same shards in torch, and tools/cmp_logits.py compares
         * them elementwise. That is the only check here that can see a small systematic
         * error in the final norm, the lm_head, or the model-level AttnRes. */
        if (logits_path && g == 0) {
            FILE *lf = fopen(logits_path, "wb");
            if (lf) {
                fwrite(lg, sizeof(float), (size_t)c.vocab, lf);
                fclose(lf);
                printf("wrote %s (%d float32 logits)\n", logits_path, c.vocab);
            } else {
                fprintf(stderr, "cannot open %s for the logits dump\n", logits_path);
            }
        }
        const double dt = now_s() - ts;
        t_total += dt;
        const uint64_t req = cache.hits + cache.misses;
        printf("%-6d %-10d %-12.2f %-10.1f %-10.2f %.3f\n", g, nxt, dt,
               req ? 100.0 * cache.hits / req : 0.0,
               (double)cache.bytes_read / 1e9, 1.0 / dt);
        fflush(stdout);
        /* Roll the per-step figures up before the next reset wipes them. */
        expert_s_total     += cache.load_seconds;
        expert_gb_total    += (double)cache.bytes_read / 1e9;
        expert_reqs_total  += cache.hits + cache.misses;
        expert_evict_total += cache.evictions;
        for (int i = 0; i < emitn && nout < gen && T < Tmax; i++) {
            seq[T++] = emit[i];
            outtok[nout++] = emit[i];
        }
        if (T >= Tmax) break;
    }
    if (save_state) {
        if (!incremental) {
            fprintf(stderr, "--save-state needs --incremental; nothing written\n");
        } else {
            const double tsv = now_s();
            if (k3_state_save(save_state, &c, seq, T, ks, &w, (int64_t)kper) == 0) {
                const double bytes = k3_state_bytes(&c, &w, T, (int64_t)kper);
                char sb[32]; human(bytes, sb, sizeof sb);
                printf("wrote %s (%s, %d positions) in %.2f s\n",
                       save_state, sb, w.cached, now_s() - tsv);
            }
        }
    }

    if (dw.trunk && hyb_rounds > 0) {
        printf("\nhybrid decode: %ld rounds, %ld drafted, %ld accepted (%.1f%%), "
               "mean accepted run %.2f\n",
               hyb_rounds, hyb_drafted, hyb_accepted,
               hyb_drafted ? 100.0 * hyb_accepted / hyb_drafted : 0.0,
               (double)hyb_accepted / hyb_rounds);
        k3_trunk_close(&trunk_d);
        free(dw.lay); free(dks); free(dsnap); free(dw.kvc); free(dw.ropec); free(dw.latc);
    }
    free(spec_snap);
    printf("--------------------------------------------------------------------\n");
    printf("%d tokens in %.1f s, %.2f s/token average\n", nout, t_total, t_total / nout);

    /* Decoded text, when a tokenizer is loaded. Printed as a distinct block rather than
     * streamed per token: a partially-decoded multi-byte sequence is not valid UTF-8, so
     * streaming would emit mojibake at every token boundary that splits a codepoint. */
    if (have_tok && nout > 0) {
        char *txt = (char *)malloc((size_t)nout * 8 + 1);
        if (txt) {
            int m = tok_decode(&tok, outtok, nout, txt, nout * 8);
            txt[m] = 0;
            printf("\n--- generated text ---\n%s\n----------------------\n\n", txt);
            free(txt);
        }
    }
    {
        char rb[32];
        human(peak_rss_bytes(), rb, sizeof rb);
        printf("PEAK RSS for the whole run: %s   <- quote this, not the plan\n\n", rb);
    }
    k3_cache_report(&cache, "final step");

    FILE *f = fopen(outp, "w");
    if (f) {
        fprintf(f, "{\"prompt_ids\":[");
        for (int i = 0; i < np; i++) fprintf(f, "%s%d", i ? "," : "", prompt[i]);
        fprintf(f, "],\"generated_ids\":[");
        for (int i = 0; i < nout; i++) fprintf(f, "%s%d", i ? "," : "", outtok[i]);
        fprintf(f, "],\"full_ids\":[");
        for (int i = 0; i < T; i++) fprintf(f, "%s%d", i ? "," : "", seq[i]);
        fprintf(f, "],\"layers\":%d,\"seconds_per_token\":%.4f}\n", NL, t_total / nout);
        fclose(f);
        printf("\nwrote %s\n", outp);
    }
    if (trace_dir) {
        char p[4096];
        snprintf(p, sizeof p, "%s/expert_hist.json", trace_dir);
        k3_cache_dump_hist(&cache, p);
        snprintf(p, sizeof p, "%s/expert_trace.bin", trace_dir);
        k3_cache_dump_trace(&cache, p);
    }

    free(w.kvc); free(w.ropec); free(w.latc); free(w.mla_slot);
    /* Report the compute-versus-I/O split rather than leaving it to be inferred.
     *
     * It cannot be inferred safely: a flat curve across a RAM sweep looks like evidence
     * of a compute-bound engine, but it is equally consistent with the trunk being
     * streamed in full at every point of the sweep, so that the bytes moved barely
     * change. Those two have opposite tuning implications, and only a direct measurement
     * separates them. */
    {
        const double trunk_s = w.trunk ? w.trunk->load_seconds : 0.0;
        /* Both terms MUST be whole-run totals over the same window. Mixing a cumulative
         * trunk time with a last-step expert time and dividing by the whole run
         * understates the expert share by roughly the token count. */
        const double io_s = trunk_s + expert_s_total;
        const double share = t_total > 0 ? 100.0 * io_s / t_total : 0.0;
        printf("I/O share of wall clock: %.1f%%  (trunk %.1f s + experts %.1f s of %.1f s)\n",
               share, trunk_s, expert_s_total, t_total);
        printf("  both figures are WHOLE-RUN totals over %d steps\n", nout);
        /* Above 100% is not a bug in the arithmetic: with more than one trunk ring slot
         * the reader thread does device work while the main thread computes, so the two
         * terms genuinely overlap and their sum can exceed wall clock. Say so, rather
         * than printing an impossible percentage with no explanation. */
        if (share > 100.0)
            printf("  over 100%% because trunk reads overlap compute on the reader thread;\n"
                   "  %.1f s of device time was hidden behind arithmetic\n", io_s - t_total);
        /* Report the DERIVED retention, not the raw hit count. `hits` counts an expert
         * the batch prefetch pulled off disk microseconds earlier, so it equals the
         * request count at every cache size and means nothing on its own. An expert that
         * had to be evicted is one that was not retained, so retained = requests -
         * evictions. The raw hit count is deliberately not printed beside this
         * percentage: "35328 of 35328 requests hit ... 2.09%% retained" reads as a
         * contradiction even though both numbers are correct. */
        const unsigned long long retained =
            (expert_reqs_total > expert_evict_total)
                ? (unsigned long long)(expert_reqs_total - expert_evict_total) : 0ULL;
        printf("  experts, whole run: %.2f GB read | %llu of %llu requests retained in RAM"
               " (%.2f%%) | %llu evictions\n"
               "    (retention = requests - evictions; the raw `hits` counter includes\n"
               "     experts the prefetcher had just read from disk, so it is not a\n"
               "     measure of avoided I/O)\n\n",
               expert_gb_total, retained,
               (unsigned long long)expert_reqs_total,
               expert_reqs_total ? 100.0 * (double)retained / (double)expert_reqs_total : 0.0,
               (unsigned long long)expert_evict_total);
    }
    if (w.trunk) { k3_trunk_report(w.trunk, "final"); k3_trunk_close(w.trunk); }
    k3_cache_free(&cache);
    for (int L = 0; L < w.n_bound; L++) k3_bind_free(&w.lay[L]);
    free(w.lay);
    k3_bind_model_free(&w.mb);
    k3_st_close(&st);
    free(h); free(br); free(ks); free(sc); free(lg);

    /* A dropped expert means some token was computed with part of its routed sum
     * missing. The run still produced token ids and they still look plausible, which is
     * exactly why this has to be an error rather than a note: silent numerical
     * corruption that exits 0 is indistinguishable from a good run. */
    if (k3_expert_drops) {
        fprintf(stderr,
                "\nRUN INVALID: %ld routed expert load(s) failed and were dropped from\n"
                "the MoE sum. The token ids above are CORRUPT. Re-run; if this repeats,\n"
                "the shard set or the storage is at fault.\n", k3_expert_drops);
        return 4;
    }
    return 0;
}
