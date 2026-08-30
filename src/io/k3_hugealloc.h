/* k3_hugealloc.h - one hugepage-backed allocation policy, shared by the trunk ring and
 * the expert cache arena.
 *
 * WHY THIS IS NOT COSMETIC. Every O_DIRECT read must pin its destination pages in the
 * kernel (get_user_pages) for the duration of the transfer. A multi-gigabyte buffer
 * backed by 4 KB pages is hundreds of thousands of pages pinned and released PER READ,
 * on both the trunk ring (k3_trunk.c) and the expert cache arena (k3_cache.c) -- the
 * same cost, for the same reason, on two unrelated buffers. Backing either with 2 MB
 * pages cuts the count by 512x, losslessly: the allocation is otherwise identical, so
 * this cannot change a single output bit.
 *
 * K3_NOHUGE=1 restores 4 KB alignment on BOTH so the two can be A/B compared on one
 * binary, which is the only way to attribute a timing difference to this decision
 * rather than to the compiler or the weather. One function means one place to keep
 * that env var, the rounding rule and the madvise call in sync -- they used to be two
 * independently maintained copies, kept aligned only by a comment pointing at the other
 * one.
 */
#ifndef K3_HUGEALLOC_H
#define K3_HUGEALLOC_H

#include <stddef.h>
#include <stdlib.h>
#include <sys/mman.h>

static inline int k3_alloc_hugepage(void **out, size_t bytes)
{
    const int huge = !getenv("K3_NOHUGE");
    const size_t align = huge ? (2u << 20) : 4096u;
    /* Round the LENGTH up too: madvise only covers whole pages, so a 2 MB-aligned start
     * with a ragged tail leaves the last stretch on 4 KB pages. */
    const size_t len = huge ? ((bytes + align - 1) & ~(align - 1)) : bytes;
    if (posix_memalign(out, align, len) != 0) return -1;
#if defined(MADV_HUGEPAGE)
    if (huge) madvise(*out, len, MADV_HUGEPAGE);   /* advisory: failure is not an error */
#endif
    return 0;
}

#endif /* K3_HUGEALLOC_H */
