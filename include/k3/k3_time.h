/* SPDX-License-Identifier: Apache-2.0 */
/* k3_time.h - a monotonic clock in seconds.
 *
 * Three translation units time themselves against a disk read or a bind: the expert
 * cache, the trunk streamer and the CLI driver. Each carried its own identical
 * `static double now_s(void)` before this header existed; one definition removes the
 * risk of the copies drifting (a rounding tweak applied to one and not the others,
 * say) without changing what any of them compute -- these values are reported, never
 * fed back into a kernel, so nothing here is part of the bit-identity contract.
 */
#ifndef K3_TIME_H
#define K3_TIME_H

#include <time.h>

static inline double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

#endif /* K3_TIME_H */
