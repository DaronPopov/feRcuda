/**
 * PTX-OS CUDA Context Interception Hook
 *
 * Ensures the PTX runtime's primary context is the single enforced context
 * for all CUDA operations. Prevents external libraries (cudarc, Candle, etc.)
 * from creating independent CUDA contexts that bypass TLSF memory management.
 *
 * Three-phase bootstrap:
 *   Phase 0 (uninit):     Hook loaded, PTX runtime not yet initialized.
 *                          All context calls pass through to real CUDA driver.
 *   Phase 1 (capturing):  PTX runtime is initializing. Recursion guard ensures
 *                          its own cuDevicePrimaryCtxRetain passes through.
 *   Phase 2 (enforcing):  PTX runtime is fully initialized. All context calls
 *                          are intercepted and enforced.
 *
 * Compiled into libptx_hook.so alongside cuda_intercept.c.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include <dlfcn.h>

#include "ptx_context_hook.h"

/* ============================================================================
 * CUDA Driver API types (minimal, no cuda.h dependency)
 * ============================================================================ */

typedef int CUresult;
typedef int CUdevice;
/* CUcontext is already defined in ptx_context_hook.h as void* */

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_NOT_INITIALIZED 3
#define CUDA_ERROR_INVALID_CONTEXT 201

/* ============================================================================
 * Extern globals from cuda_intercept.c (non-static)
 * ============================================================================ */

typedef struct GPUHotRuntime GPUHotRuntime;

extern GPUHotRuntime* g_runtime;
extern volatile int   g_init_state;
extern bool           g_verbose;

/* ============================================================================
 * Per-device tracking
 * ============================================================================ */

#define PTX_CTX_MAX_DEVICES 16

typedef struct {
    CUcontext ptx_context;
    CUdevice  cu_device;
    int       refcount;
    bool      captured;
} PTXContextState;

static PTXContextState g_ctx_devices[PTX_CTX_MAX_DEVICES];
static pthread_mutex_t g_ctx_mutex = PTHREAD_MUTEX_INITIALIZER;
static volatile int    g_ctx_init_state = 0;  /* 0=uninit, 1=capturing, 2=enforcing, -1=disabled */
static __thread int    t_ctx_in_hook = 0;     /* Recursion guard */

static bool g_ctx_warn_only = false;

static PTXContextStats g_ctx_stats;

/* ============================================================================
 * Real CUDA driver function pointers
 * ============================================================================ */

typedef CUresult (*cuInit_fn)(unsigned int);
typedef CUresult (*cuCtxCreate_fn)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*cuCtxCreate_v2_fn)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*cuCtxCreate_v3_fn)(CUcontext*, void*, int, unsigned int, CUdevice);
typedef CUresult (*cuCtxDestroy_fn)(CUcontext);
typedef CUresult (*cuCtxDestroy_v2_fn)(CUcontext);
typedef CUresult (*cuCtxSetCurrent_fn)(CUcontext);
typedef CUresult (*cuCtxGetCurrent_fn)(CUcontext*);
typedef CUresult (*cuCtxPushCurrent_fn)(CUcontext);
typedef CUresult (*cuCtxPushCurrent_v2_fn)(CUcontext);
typedef CUresult (*cuCtxPopCurrent_fn)(CUcontext*);
typedef CUresult (*cuCtxPopCurrent_v2_fn)(CUcontext*);
typedef CUresult (*cuCtxSynchronize_fn)(void);
typedef CUresult (*cuDevicePrimaryCtxRetain_fn)(CUcontext*, CUdevice);
typedef CUresult (*cuDevicePrimaryCtxRelease_fn)(CUdevice);
typedef CUresult (*cuDevicePrimaryCtxRelease_v2_fn)(CUdevice);
typedef CUresult (*cuDevicePrimaryCtxReset_fn)(CUdevice);
typedef CUresult (*cuDevicePrimaryCtxReset_v2_fn)(CUdevice);
typedef CUresult (*cuDevicePrimaryCtxSetFlags_fn)(CUdevice, unsigned int);
typedef CUresult (*cuDevicePrimaryCtxSetFlags_v2_fn)(CUdevice, unsigned int);
typedef CUresult (*cuDeviceGet_fn)(CUdevice*, int);

static cuInit_fn                       real_cuInit = NULL;
static cuCtxCreate_v2_fn               real_cuCtxCreate_v2 = NULL;
static cuCtxCreate_v3_fn               real_cuCtxCreate_v3 = NULL;
static cuCtxDestroy_v2_fn              real_cuCtxDestroy_v2 = NULL;
static cuCtxSetCurrent_fn              real_cuCtxSetCurrent = NULL;
static cuCtxGetCurrent_fn              real_cuCtxGetCurrent = NULL;
static cuCtxPushCurrent_v2_fn          real_cuCtxPushCurrent_v2 = NULL;
static cuCtxPopCurrent_v2_fn           real_cuCtxPopCurrent_v2 = NULL;
static cuCtxSynchronize_fn             real_cuCtxSynchronize = NULL;
static cuDevicePrimaryCtxRetain_fn     real_cuDevicePrimaryCtxRetain = NULL;
static cuDevicePrimaryCtxRelease_v2_fn real_cuDevicePrimaryCtxRelease_v2 = NULL;
static cuDevicePrimaryCtxReset_v2_fn   real_cuDevicePrimaryCtxReset_v2 = NULL;
static cuDevicePrimaryCtxSetFlags_v2_fn real_cuDevicePrimaryCtxSetFlags_v2 = NULL;
static cuDeviceGet_fn                  real_cuDeviceGet = NULL;

static void resolve_real_ctx_functions(void) {
    if (real_cuCtxCreate_v2 && real_cuDevicePrimaryCtxRetain) return;
    pthread_mutex_lock(&g_ctx_mutex);

    if (!real_cuInit)
        real_cuInit = (cuInit_fn)dlsym(RTLD_NEXT, "cuInit");
    if (!real_cuCtxCreate_v2)
        real_cuCtxCreate_v2 = (cuCtxCreate_v2_fn)dlsym(RTLD_NEXT, "cuCtxCreate_v2");
    if (!real_cuCtxCreate_v3)
        real_cuCtxCreate_v3 = (cuCtxCreate_v3_fn)dlsym(RTLD_NEXT, "cuCtxCreate_v3");
    if (!real_cuCtxDestroy_v2)
        real_cuCtxDestroy_v2 = (cuCtxDestroy_v2_fn)dlsym(RTLD_NEXT, "cuCtxDestroy_v2");
    if (!real_cuCtxSetCurrent)
        real_cuCtxSetCurrent = (cuCtxSetCurrent_fn)dlsym(RTLD_NEXT, "cuCtxSetCurrent");
    if (!real_cuCtxGetCurrent)
        real_cuCtxGetCurrent = (cuCtxGetCurrent_fn)dlsym(RTLD_NEXT, "cuCtxGetCurrent");
    if (!real_cuCtxPushCurrent_v2)
        real_cuCtxPushCurrent_v2 = (cuCtxPushCurrent_v2_fn)dlsym(RTLD_NEXT, "cuCtxPushCurrent_v2");
    if (!real_cuCtxPopCurrent_v2)
        real_cuCtxPopCurrent_v2 = (cuCtxPopCurrent_v2_fn)dlsym(RTLD_NEXT, "cuCtxPopCurrent_v2");
    if (!real_cuCtxSynchronize)
        real_cuCtxSynchronize = (cuCtxSynchronize_fn)dlsym(RTLD_NEXT, "cuCtxSynchronize");
    if (!real_cuDevicePrimaryCtxRetain)
        real_cuDevicePrimaryCtxRetain = (cuDevicePrimaryCtxRetain_fn)dlsym(RTLD_NEXT, "cuDevicePrimaryCtxRetain");
    if (!real_cuDevicePrimaryCtxRelease_v2) {
        real_cuDevicePrimaryCtxRelease_v2 = (cuDevicePrimaryCtxRelease_v2_fn)dlsym(RTLD_NEXT, "cuDevicePrimaryCtxRelease_v2");
        if (!real_cuDevicePrimaryCtxRelease_v2)
            real_cuDevicePrimaryCtxRelease_v2 = (cuDevicePrimaryCtxRelease_v2_fn)dlsym(RTLD_NEXT, "cuDevicePrimaryCtxRelease");
    }
    if (!real_cuDevicePrimaryCtxReset_v2) {
        real_cuDevicePrimaryCtxReset_v2 = (cuDevicePrimaryCtxReset_v2_fn)dlsym(RTLD_NEXT, "cuDevicePrimaryCtxReset_v2");
        if (!real_cuDevicePrimaryCtxReset_v2)
            real_cuDevicePrimaryCtxReset_v2 = (cuDevicePrimaryCtxReset_v2_fn)dlsym(RTLD_NEXT, "cuDevicePrimaryCtxReset");
    }
    if (!real_cuDevicePrimaryCtxSetFlags_v2) {
        real_cuDevicePrimaryCtxSetFlags_v2 = (cuDevicePrimaryCtxSetFlags_v2_fn)dlsym(RTLD_NEXT, "cuDevicePrimaryCtxSetFlags_v2");
        if (!real_cuDevicePrimaryCtxSetFlags_v2)
            real_cuDevicePrimaryCtxSetFlags_v2 = (cuDevicePrimaryCtxSetFlags_v2_fn)dlsym(RTLD_NEXT, "cuDevicePrimaryCtxSetFlags");
    }
    if (!real_cuDeviceGet)
        real_cuDeviceGet = (cuDeviceGet_fn)dlsym(RTLD_NEXT, "cuDeviceGet");

    pthread_mutex_unlock(&g_ctx_mutex);
}

/* ============================================================================
 * Helper: check if we are in enforcement mode
 * ============================================================================ */

static inline bool ctx_enforcing(void) {
    return g_ctx_init_state == 2 && !t_ctx_in_hook;
}

static inline bool device_valid(int dev) {
    return dev >= 0 && dev < PTX_CTX_MAX_DEVICES;
}

/* ============================================================================
 * Public API
 * ============================================================================ */

void ptx_context_hook_init(void) {
    if (getenv("PTX_CONTEXT_HOOK_DISABLE")) {
        g_ctx_init_state = -1;
        if (g_verbose) {
            fprintf(stderr, "[PTX-CTX-HOOK] Disabled via PTX_CONTEXT_HOOK_DISABLE\n");
        }
        return;
    }
    g_ctx_warn_only = getenv("PTX_CONTEXT_HOOK_WARN_ONLY") != NULL;

    memset(g_ctx_devices, 0, sizeof(g_ctx_devices));
    memset(&g_ctx_stats, 0, sizeof(g_ctx_stats));

    resolve_real_ctx_functions();

    if (g_verbose) {
        fprintf(stderr, "[PTX-CTX-HOOK] Initialized (warn_only=%d)\n", g_ctx_warn_only);
    }
}

void ptx_context_hook_capture(int device_id, CUcontext ctx) {
    if (g_ctx_init_state == -1) return;
    if (!device_valid(device_id) || !ctx) return;

    pthread_mutex_lock(&g_ctx_mutex);
    g_ctx_devices[device_id].ptx_context = ctx;
    g_ctx_devices[device_id].cu_device = device_id;
    g_ctx_devices[device_id].refcount = 1;
    g_ctx_devices[device_id].captured = true;
    g_ctx_init_state = 2;
    pthread_mutex_unlock(&g_ctx_mutex);

    if (g_verbose) {
        fprintf(stderr, "[PTX-CTX-HOOK] Captured primary context %p for device %d\n",
                ctx, device_id);
    }
}

void ptx_context_hook_set_ptx_active(bool active) {
    if (g_ctx_init_state == -1) return;
    if (active) {
        g_ctx_init_state = 2;
    } else {
        /* Transition back to passthrough for clean shutdown */
        g_ctx_init_state = 0;
    }
}

bool ptx_context_hook_is_active(void) {
    return g_ctx_init_state == 2;
}

CUcontext ptx_context_hook_get_context(int device_id) {
    if (!device_valid(device_id)) return NULL;
    return g_ctx_devices[device_id].ptx_context;
}

void ptx_context_hook_get_stats(PTXContextStats* stats) {
    if (!stats) return;
    pthread_mutex_lock(&g_ctx_mutex);
    *stats = g_ctx_stats;
    pthread_mutex_unlock(&g_ctx_mutex);
}

void ptx_context_hook_reset_stats(void) {
    pthread_mutex_lock(&g_ctx_mutex);
    memset(&g_ctx_stats, 0, sizeof(g_ctx_stats));
    pthread_mutex_unlock(&g_ctx_mutex);
}

void ptx_context_hook_print_stats(void) {
    if (g_ctx_init_state == -1) return;

    PTXContextStats s;
    ptx_context_hook_get_stats(&s);

    fprintf(stderr, "[PTX-CTX-HOOK] Context Hook Statistics:\n");
    fprintf(stderr, "  cuCtxCreate blocked:           %lu\n", (unsigned long)s.ctx_create_blocked);
    fprintf(stderr, "  cuCtxCreate passthrough:       %lu\n", (unsigned long)s.ctx_create_passthrough);
    fprintf(stderr, "  cuCtxDestroy blocked:          %lu\n", (unsigned long)s.ctx_destroy_blocked);
    fprintf(stderr, "  Context switch prevented:      %lu\n", (unsigned long)s.ctx_switch_prevented);
    fprintf(stderr, "  Context switch substituted:    %lu\n", (unsigned long)s.ctx_switch_substituted);
    fprintf(stderr, "  PrimaryCtxRetain intercepted:  %lu\n", (unsigned long)s.primary_retain_intercepted);
    fprintf(stderr, "  PrimaryCtxRelease suppressed:  %lu\n", (unsigned long)s.primary_release_suppressed);
    fprintf(stderr, "  PrimaryCtxReset blocked:       %lu\n", (unsigned long)s.primary_reset_blocked);
    fprintf(stderr, "  PrimaryCtxSetFlags blocked:    %lu\n", (unsigned long)s.primary_setflags_blocked);
    fprintf(stderr, "  Push substituted:              %lu\n", (unsigned long)s.push_substituted);
    fprintf(stderr, "  Pop corrected:                 %lu\n", (unsigned long)s.pop_corrected);
}

void ptx_context_hook_capture_from_runtime(void) {
    if (g_ctx_init_state == -1) return;
    if (!g_runtime) return;

    /* gpu_hot_get_context is declared in gpu_hot_runtime.h */
    extern void* gpu_hot_get_context(GPUHotRuntime* runtime);
    CUcontext ctx = (CUcontext)gpu_hot_get_context(g_runtime);
    if (ctx) {
        /* Determine device from environment or default to 0 */
        int device_id = 0;
        const char* dev_env = getenv("PTX_HOOK_DEVICE");
        if (dev_env) device_id = atoi(dev_env);
        ptx_context_hook_capture(device_id, ctx);
    }
}

/* ============================================================================
 * Helper: find device index for a given context
 * ============================================================================ */

static int find_device_for_ctx(CUcontext ctx) {
    for (int i = 0; i < PTX_CTX_MAX_DEVICES; i++) {
        if (g_ctx_devices[i].captured && g_ctx_devices[i].ptx_context == ctx)
            return i;
    }
    return -1;
}

static bool is_ptx_context(CUcontext ctx) {
    return find_device_for_ctx(ctx) >= 0;
}

/* Default PTX context (device 0 or first captured) */
static CUcontext default_ptx_ctx(void) {
    if (g_ctx_devices[0].captured)
        return g_ctx_devices[0].ptx_context;
    for (int i = 0; i < PTX_CTX_MAX_DEVICES; i++) {
        if (g_ctx_devices[i].captured)
            return g_ctx_devices[i].ptx_context;
    }
    return NULL;
}

/* ============================================================================
 * Intercepted CUDA Driver API functions
 * ============================================================================ */

/* --- cuInit --- */
CUresult cuInit(unsigned int flags) {
    resolve_real_ctx_functions();
    if (real_cuInit) {
        return real_cuInit(flags);
    }
    return CUDA_ERROR_NOT_INITIALIZED;
}

/* --- cuCtxCreate_v2 --- */
CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    resolve_real_ctx_functions();

    if (!ctx_enforcing()) {
        /* Phase 0-1: passthrough */
        __sync_fetch_and_add(&g_ctx_stats.ctx_create_passthrough, 1);
        if (real_cuCtxCreate_v2)
            return real_cuCtxCreate_v2(pctx, flags, dev);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    /* Phase 2: Block and return PTX context instead */
    __sync_fetch_and_add(&g_ctx_stats.ctx_create_blocked, 1);

    if (g_verbose) {
        fprintf(stderr, "[PTX-CTX-HOOK] BLOCKED cuCtxCreate on device %d%s\n",
                dev, g_ctx_warn_only ? " (warn-only)" : "");
    }

    if (g_ctx_warn_only && real_cuCtxCreate_v2) {
        return real_cuCtxCreate_v2(pctx, flags, dev);
    }

    if (pctx && device_valid(dev) && g_ctx_devices[dev].captured) {
        *pctx = g_ctx_devices[dev].ptx_context;
    } else if (pctx) {
        CUcontext fallback = default_ptx_ctx();
        if (fallback) {
            *pctx = fallback;
        } else if (real_cuCtxCreate_v2) {
            /* No PTX context for this device yet, must passthrough */
            return real_cuCtxCreate_v2(pctx, flags, dev);
        }
    }
    return CUDA_SUCCESS;
}

/* Legacy cuCtxCreate -> forwards to _v2 */
CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    return cuCtxCreate_v2(pctx, flags, dev);
}

/* --- cuCtxCreate_v3 --- */
CUresult cuCtxCreate_v3(CUcontext* pctx, void* paramsArray, int numParams,
                         unsigned int flags, CUdevice dev) {
    resolve_real_ctx_functions();

    if (!ctx_enforcing()) {
        if (real_cuCtxCreate_v3)
            return real_cuCtxCreate_v3(pctx, paramsArray, numParams, flags, dev);
        /* Fall back to v2 if v3 not available */
        return cuCtxCreate_v2(pctx, flags, dev);
    }

    /* Phase 2: Block */
    __sync_fetch_and_add(&g_ctx_stats.ctx_create_blocked, 1);
    if (g_verbose) {
        fprintf(stderr, "[PTX-CTX-HOOK] BLOCKED cuCtxCreate_v3 on device %d\n", dev);
    }

    if (g_ctx_warn_only && real_cuCtxCreate_v3) {
        return real_cuCtxCreate_v3(pctx, paramsArray, numParams, flags, dev);
    }

    if (pctx && device_valid(dev) && g_ctx_devices[dev].captured) {
        *pctx = g_ctx_devices[dev].ptx_context;
    } else if (pctx) {
        CUcontext fallback = default_ptx_ctx();
        if (fallback) *pctx = fallback;
    }
    return CUDA_SUCCESS;
}

/* --- cuCtxDestroy_v2 --- */
CUresult cuCtxDestroy_v2(CUcontext ctx) {
    resolve_real_ctx_functions();

    if (!ctx_enforcing()) {
        if (real_cuCtxDestroy_v2)
            return real_cuCtxDestroy_v2(ctx);
        return CUDA_SUCCESS;
    }

    /* Block if it's a PTX context */
    if (is_ptx_context(ctx)) {
        __sync_fetch_and_add(&g_ctx_stats.ctx_destroy_blocked, 1);
        if (g_verbose) {
            fprintf(stderr, "[PTX-CTX-HOOK] BLOCKED cuCtxDestroy on PTX context %p\n", ctx);
        }
        if (g_ctx_warn_only && real_cuCtxDestroy_v2) {
            return real_cuCtxDestroy_v2(ctx);
        }
        return CUDA_SUCCESS;
    }

    /* Non-PTX context on another device: allow */
    if (real_cuCtxDestroy_v2)
        return real_cuCtxDestroy_v2(ctx);
    return CUDA_SUCCESS;
}

/* Legacy cuCtxDestroy -> forwards to _v2 */
CUresult cuCtxDestroy(CUcontext ctx) {
    return cuCtxDestroy_v2(ctx);
}

/* --- cuDevicePrimaryCtxRetain --- */
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    resolve_real_ctx_functions();

    if (!ctx_enforcing()) {
        /* Phase 0-1: passthrough, but observe and record */
        if (!real_cuDevicePrimaryCtxRetain)
            return CUDA_ERROR_NOT_INITIALIZED;
        CUresult res = real_cuDevicePrimaryCtxRetain(pctx, dev);
        if (res == CUDA_SUCCESS && pctx && device_valid(dev) && !g_ctx_devices[dev].captured) {
            /* Record observed context (not yet enforcing) */
            g_ctx_devices[dev].ptx_context = *pctx;
            g_ctx_devices[dev].cu_device = dev;
        }
        return res;
    }

    /* Phase 2: return captured PTX context */
    __sync_fetch_and_add(&g_ctx_stats.primary_retain_intercepted, 1);

    if (device_valid(dev) && g_ctx_devices[dev].captured) {
        if (pctx) *pctx = g_ctx_devices[dev].ptx_context;

        pthread_mutex_lock(&g_ctx_mutex);
        g_ctx_devices[dev].refcount++;
        pthread_mutex_unlock(&g_ctx_mutex);

        if (g_verbose) {
            fprintf(stderr, "[PTX-CTX-HOOK] cuDevicePrimaryCtxRetain(dev=%d) -> PTX context %p (refcount=%d)\n",
                    dev, g_ctx_devices[dev].ptx_context, g_ctx_devices[dev].refcount);
        }
        return CUDA_SUCCESS;
    }

    /* Device not captured by PTX: passthrough */
    if (real_cuDevicePrimaryCtxRetain)
        return real_cuDevicePrimaryCtxRetain(pctx, dev);
    return CUDA_ERROR_NOT_INITIALIZED;
}

/* --- cuDevicePrimaryCtxRelease_v2 --- */
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    resolve_real_ctx_functions();

    if (!ctx_enforcing()) {
        if (real_cuDevicePrimaryCtxRelease_v2)
            return real_cuDevicePrimaryCtxRelease_v2(dev);
        return CUDA_SUCCESS;
    }

    if (device_valid(dev) && g_ctx_devices[dev].captured) {
        __sync_fetch_and_add(&g_ctx_stats.primary_release_suppressed, 1);

        pthread_mutex_lock(&g_ctx_mutex);
        if (g_ctx_devices[dev].refcount > 0)
            g_ctx_devices[dev].refcount--;
        pthread_mutex_unlock(&g_ctx_mutex);

        if (g_verbose) {
            fprintf(stderr, "[PTX-CTX-HOOK] cuDevicePrimaryCtxRelease(dev=%d) suppressed (refcount=%d)\n",
                    dev, g_ctx_devices[dev].refcount);
        }

        if (g_ctx_warn_only && real_cuDevicePrimaryCtxRelease_v2) {
            return real_cuDevicePrimaryCtxRelease_v2(dev);
        }

        /* Never actually release while PTX is active */
        return CUDA_SUCCESS;
    }

    if (real_cuDevicePrimaryCtxRelease_v2)
        return real_cuDevicePrimaryCtxRelease_v2(dev);
    return CUDA_SUCCESS;
}

/* Legacy alias */
CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    return cuDevicePrimaryCtxRelease_v2(dev);
}

/* --- cuDevicePrimaryCtxReset_v2 --- */
CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
    resolve_real_ctx_functions();

    if (!ctx_enforcing()) {
        if (real_cuDevicePrimaryCtxReset_v2)
            return real_cuDevicePrimaryCtxReset_v2(dev);
        return CUDA_SUCCESS;
    }

    if (device_valid(dev) && g_ctx_devices[dev].captured) {
        __sync_fetch_and_add(&g_ctx_stats.primary_reset_blocked, 1);
        if (g_verbose) {
            fprintf(stderr, "[PTX-CTX-HOOK] BLOCKED cuDevicePrimaryCtxReset on device %d\n", dev);
        }
        if (g_ctx_warn_only && real_cuDevicePrimaryCtxReset_v2) {
            return real_cuDevicePrimaryCtxReset_v2(dev);
        }
        return CUDA_SUCCESS;
    }

    if (real_cuDevicePrimaryCtxReset_v2)
        return real_cuDevicePrimaryCtxReset_v2(dev);
    return CUDA_SUCCESS;
}

/* Legacy alias */
CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
    return cuDevicePrimaryCtxReset_v2(dev);
}

/* --- cuDevicePrimaryCtxSetFlags_v2 --- */
CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
    resolve_real_ctx_functions();

    if (!ctx_enforcing()) {
        if (real_cuDevicePrimaryCtxSetFlags_v2)
            return real_cuDevicePrimaryCtxSetFlags_v2(dev, flags);
        return CUDA_SUCCESS;
    }

    if (device_valid(dev) && g_ctx_devices[dev].captured) {
        __sync_fetch_and_add(&g_ctx_stats.primary_setflags_blocked, 1);
        if (g_verbose) {
            fprintf(stderr, "[PTX-CTX-HOOK] BLOCKED cuDevicePrimaryCtxSetFlags on device %d (flags=0x%x)\n",
                    dev, flags);
        }
        if (g_ctx_warn_only && real_cuDevicePrimaryCtxSetFlags_v2) {
            return real_cuDevicePrimaryCtxSetFlags_v2(dev, flags);
        }
        return CUDA_SUCCESS;
    }

    if (real_cuDevicePrimaryCtxSetFlags_v2)
        return real_cuDevicePrimaryCtxSetFlags_v2(dev, flags);
    return CUDA_SUCCESS;
}

/* Legacy alias */
CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    return cuDevicePrimaryCtxSetFlags_v2(dev, flags);
}

/* --- cuCtxSetCurrent --- */
CUresult cuCtxSetCurrent(CUcontext ctx) {
    resolve_real_ctx_functions();

    if (!ctx_enforcing()) {
        if (real_cuCtxSetCurrent)
            return real_cuCtxSetCurrent(ctx);
        return CUDA_SUCCESS;
    }

    /* Allow NULL (clears context) or PTX context */
    if (ctx == NULL || is_ptx_context(ctx)) {
        if (real_cuCtxSetCurrent)
            return real_cuCtxSetCurrent(ctx);
        return CUDA_SUCCESS;
    }

    /* Substitute PTX context for anything else */
    __sync_fetch_and_add(&g_ctx_stats.ctx_switch_substituted, 1);
    CUcontext ptx_ctx = default_ptx_ctx();
    if (g_verbose) {
        fprintf(stderr, "[PTX-CTX-HOOK] cuCtxSetCurrent: substituted %p -> PTX %p\n",
                ctx, ptx_ctx);
    }

    if (g_ctx_warn_only && real_cuCtxSetCurrent) {
        return real_cuCtxSetCurrent(ctx);
    }

    if (ptx_ctx && real_cuCtxSetCurrent) {
        return real_cuCtxSetCurrent(ptx_ctx);
    }
    return CUDA_SUCCESS;
}

/* --- cuCtxGetCurrent --- */
CUresult cuCtxGetCurrent(CUcontext* pctx) {
    resolve_real_ctx_functions();

    if (!ctx_enforcing()) {
        if (real_cuCtxGetCurrent)
            return real_cuCtxGetCurrent(pctx);
        return CUDA_SUCCESS;
    }

    /* Return PTX context if captured */
    CUcontext ptx_ctx = default_ptx_ctx();
    if (ptx_ctx && pctx) {
        /* First get real current, but ensure it's the PTX one */
        if (real_cuCtxGetCurrent) {
            CUresult res = real_cuCtxGetCurrent(pctx);
            if (res == CUDA_SUCCESS && *pctx != ptx_ctx && *pctx != NULL) {
                /* Someone switched context under us - report PTX context */
                *pctx = ptx_ctx;
            }
            return res;
        }
        *pctx = ptx_ctx;
        return CUDA_SUCCESS;
    }

    if (real_cuCtxGetCurrent)
        return real_cuCtxGetCurrent(pctx);
    return CUDA_SUCCESS;
}

/* --- cuCtxPushCurrent_v2 --- */
CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    resolve_real_ctx_functions();

    if (!ctx_enforcing()) {
        if (real_cuCtxPushCurrent_v2)
            return real_cuCtxPushCurrent_v2(ctx);
        return CUDA_SUCCESS;
    }

    /* Substitute non-PTX context */
    if (ctx != NULL && !is_ptx_context(ctx)) {
        __sync_fetch_and_add(&g_ctx_stats.push_substituted, 1);
        CUcontext ptx_ctx = default_ptx_ctx();
        if (g_verbose) {
            fprintf(stderr, "[PTX-CTX-HOOK] cuCtxPushCurrent: substituted %p -> PTX %p\n",
                    ctx, ptx_ctx);
        }
        if (g_ctx_warn_only && real_cuCtxPushCurrent_v2) {
            return real_cuCtxPushCurrent_v2(ctx);
        }
        if (ptx_ctx && real_cuCtxPushCurrent_v2) {
            return real_cuCtxPushCurrent_v2(ptx_ctx);
        }
    }

    if (real_cuCtxPushCurrent_v2)
        return real_cuCtxPushCurrent_v2(ctx);
    return CUDA_SUCCESS;
}

/* Legacy alias */
CUresult cuCtxPushCurrent(CUcontext ctx) {
    return cuCtxPushCurrent_v2(ctx);
}

/* --- cuCtxPopCurrent_v2 --- */
CUresult cuCtxPopCurrent_v2(CUcontext* pctx) {
    resolve_real_ctx_functions();

    if (!ctx_enforcing()) {
        if (real_cuCtxPopCurrent_v2)
            return real_cuCtxPopCurrent_v2(pctx);
        return CUDA_SUCCESS;
    }

    if (real_cuCtxPopCurrent_v2) {
        CUresult res = real_cuCtxPopCurrent_v2(pctx);
        if (res == CUDA_SUCCESS) {
            /* Ensure PTX context stays on top */
            CUcontext current = NULL;
            if (real_cuCtxGetCurrent) {
                real_cuCtxGetCurrent(&current);
            }
            CUcontext ptx_ctx = default_ptx_ctx();
            if (ptx_ctx && current != ptx_ctx && current == NULL) {
                /* Stack became empty, push PTX context back */
                __sync_fetch_and_add(&g_ctx_stats.pop_corrected, 1);
                if (real_cuCtxSetCurrent) {
                    real_cuCtxSetCurrent(ptx_ctx);
                }
                if (g_verbose) {
                    fprintf(stderr, "[PTX-CTX-HOOK] cuCtxPopCurrent: restored PTX context %p\n",
                            ptx_ctx);
                }
            }
        }
        return res;
    }
    return CUDA_SUCCESS;
}

/* Legacy alias */
CUresult cuCtxPopCurrent(CUcontext* pctx) {
    return cuCtxPopCurrent_v2(pctx);
}

/* --- cuCtxSynchronize --- */
CUresult cuCtxSynchronize(void) {
    resolve_real_ctx_functions();
    /* Always passthrough - no lifecycle effect */
    if (real_cuCtxSynchronize)
        return real_cuCtxSynchronize();
    return CUDA_SUCCESS;
}
