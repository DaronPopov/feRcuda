/*
 * PTX-OS Stable Runtime ABI
 *
 * Thin, versioned C ABI intended as the foundation contract for upper layers
 * (ferrite-llm, ferrite-training, Python/Rust adapters). This header should
 * remain backward-compatible across internal runtime refactors.
 */

#ifndef PTX_STABLE_RUNTIME_H
#define PTX_STABLE_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PTX_STABLE_ABI_VERSION 1u
#define PTX_STABLE_CONFIG_DEFAULT 0u
#define PTX_STABLE_INVALID_DEVICE (-1)

typedef struct PTXStableRuntime PTXStableRuntime;

typedef enum PTXStableStatus {
    PTX_STABLE_OK = 0,
    PTX_STABLE_ERR_INVALID_ARG = 1,
    PTX_STABLE_ERR_ABI_MISMATCH = 2,
    PTX_STABLE_ERR_INIT_FAILED = 3,
    PTX_STABLE_ERR_NOT_INITIALIZED = 4,
    PTX_STABLE_ERR_NOT_OWNED = 5,
    PTX_STABLE_ERR_ALLOC_FAILED = 6,
    PTX_STABLE_ERR_INTERNAL = 7,
} PTXStableStatus;

typedef struct PTXStableConfig {
    uint32_t struct_size;      /* must be sizeof(PTXStableConfig) */
    uint32_t abi_version;      /* must be PTX_STABLE_ABI_VERSION */
    uint32_t flags;            /* reserved for future behavior toggles */
    int32_t device_id;         /* PTX_STABLE_INVALID_DEVICE => default device 0 */
    float pool_fraction;       /* 0.0 => ignored */
    uint64_t fixed_pool_size;  /* bytes, 0 => ignored */
    uint64_t reserve_vram;     /* bytes, 0 => runtime default */
    uint32_t max_streams;      /* 0 => runtime default */
    uint8_t quiet_init;        /* 1 => suppress runtime banner output */
    uint8_t enable_leak_detection;
    uint8_t enable_pool_health;
    uint8_t _reserved0;
} PTXStableConfig;

typedef struct PTXStableStats {
    uint64_t vram_allocated;
    uint64_t vram_used;
    uint64_t vram_free;
    float gpu_utilization;

    uint64_t pool_total;
    uint64_t pool_used;
    uint64_t pool_free;
    uint64_t pool_peak;
    uint64_t pool_fallbacks;
    float pool_fragmentation;

    uint64_t total_ops;
    uint32_t active_streams;
    uint32_t refcount;
    uint8_t watchdog_tripped;
    uint8_t _reserved[3];
} PTXStableStats;

const char* ptx_stable_runtime_version(void);
uint32_t ptx_stable_runtime_abi_version(void);
const char* ptx_stable_strerror(PTXStableStatus status);

PTXStableStatus ptx_stable_init(const PTXStableConfig* config, PTXStableRuntime** out_runtime);
PTXStableStatus ptx_stable_retain(PTXStableRuntime** out_runtime);
PTXStableStatus ptx_stable_release(PTXStableRuntime* runtime);
PTXStableStatus ptx_stable_get(PTXStableRuntime** out_runtime);

PTXStableStatus ptx_stable_alloc(PTXStableRuntime* runtime, size_t size, void** out_ptr);
PTXStableStatus ptx_stable_free(PTXStableRuntime* runtime, void* ptr);
PTXStableStatus ptx_stable_owns_ptr(PTXStableRuntime* runtime, const void* ptr, bool* out_owned);

PTXStableStatus ptx_stable_get_context(PTXStableRuntime* runtime, void** out_cu_context);
PTXStableStatus ptx_stable_get_stats(PTXStableRuntime* runtime, PTXStableStats* out_stats);
PTXStableStatus ptx_stable_get_hot_runtime(PTXStableRuntime* runtime, void** out_gpu_hot_runtime);

#ifdef __cplusplus
}
#endif

#endif /* PTX_STABLE_RUNTIME_H */
