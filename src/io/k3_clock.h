/* k3_clock.h - a monotonic wall clock in seconds, as a double.
 *
 * Every caller wants elapsed time for logging or accounting, never wall-clock date, so
 * CLOCK_MONOTONIC (immune to NTP/date jumps) is the only clock this returns. Header-only
 * so it can be pulled into the cache, trunk reader and CLI without adding a translation
 * unit for four lines of code.
 */
#ifndef K3_CLOCK_H
#define K3_CLOCK_H

#include <time.h>

static inline double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

#endif /* K3_CLOCK_H */
