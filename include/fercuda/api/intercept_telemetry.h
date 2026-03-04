#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fercuda_intercept_stats {
    uint64_t init_calls;
    uint64_t init_success;
    uint64_t init_fail;

    uint64_t alloc_calls_total;
    uint64_t free_calls_total;

    uint64_t alloc_calls_driver;
    uint64_t free_calls_driver;
    uint64_t alloc_calls_runtime;
    uint64_t free_calls_runtime;

    uint64_t alloc_calls_async;
    uint64_t free_calls_async;

    uint64_t alloc_bytes_requested;

    uint64_t tlsf_alloc_success;
    uint64_t tlsf_alloc_fail;
    uint64_t tlsf_free_success;
    uint64_t tlsf_free_miss;

    uint64_t fallback_alloc_calls;
    uint64_t fallback_free_calls;
} fercuda_intercept_stats_t;

// Returns 0 on success, non-zero on invalid argument.
int fercuda_intercept_telemetry_get(fercuda_intercept_stats_t* out_stats);

// Resets all counters to 0.
void fercuda_intercept_telemetry_reset(void);

// Returns 1 when interceptor is currently enabled, else 0.
int fercuda_intercept_telemetry_enabled(void);

#ifdef __cplusplus
}
#endif
