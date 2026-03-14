/**
 * PTX-OS CUDA Memory Intercept Hook
 *
 * Intercepts cudaMalloc/cudaFree calls via LD_PRELOAD to redirect
 * all GPU memory allocations through PTX-OS TLSF allocator.
 *
 * Usage:
 *   LD_PRELOAD=libptx_hook.so python train.py
 *   LD_PRELOAD=libptx_hook.so ./my_cuda_app
 *
 * Environment variables:
 *   PTX_HOOK_VERBOSE=1             - Print allocation/free info
 *   PTX_HOOK_DEVICE=0              - CUDA device ID (default: 0)
 *   PTX_HOOK_DISABLE=1             - Disable hook (allocations will fail)
 *   PTX_HOOK_MODE=tlsf|cuda|hybrid - Allocator mode (default: tlsf)
 *   PTX_HOOK_HYBRID_FALLBACK=1     - Allow TLSF alloc to fall back to CUDA (hybrid only)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <dlfcn.h>

#include "ptx_debug.h"
#include "ptx_context_hook.h"

// CUDA Runtime API types
typedef int cudaError_t;
typedef void* cudaStream_t;
#define cudaSuccess 0
#define cudaErrorMemoryAllocation 2
#define cudaErrorInvalidDevicePointer 17

// CUDA Driver API types
typedef int CUresult;
typedef void* CUdeviceptr;
typedef void* CUstream;
typedef void* CUmemoryPool;
typedef void* CUarray;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_INVALID_VALUE 1

// CUDA Memory Pool types (CUDA 11.2+)
typedef void* cudaMemPool_t;
typedef enum cudaMemPoolAttr { cudaMemPoolAttrPlaceholder = 0 } cudaMemPoolAttr;

// PTX-OS API
typedef struct GPUHotRuntime GPUHotRuntime;
extern GPUHotRuntime* gpu_hot_init(int device_id, const char* token);
extern void gpu_hot_shutdown(GPUHotRuntime* runtime);
extern void* gpu_hot_alloc(GPUHotRuntime* runtime, size_t size);
extern void gpu_hot_free(GPUHotRuntime* runtime, void* ptr);
extern void* gpu_hot_alloc_async(GPUHotRuntime* runtime, size_t size, cudaStream_t stream);
extern void gpu_hot_free_async(GPUHotRuntime* runtime, void* ptr, cudaStream_t stream);
extern bool gpu_hot_owns_ptr(GPUHotRuntime* runtime, void* ptr);

// Global state (non-static: shared with cuda_context_hook.c)
GPUHotRuntime* g_runtime = NULL;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_resolve_mutex = PTHREAD_MUTEX_INITIALIZER;
volatile int g_init_state = 0;  // 0=uninit, 1=initializing, 2=ready, -1=disabled
bool g_verbose = false;
static bool g_hybrid_fallback = false;

typedef enum {
    PTX_HOOK_MODE_TLSF = 0,
    PTX_HOOK_MODE_CUDA = 1,
    PTX_HOOK_MODE_HYBRID = 2,
} PTXHookMode;

static PTXHookMode g_mode = PTX_HOOK_MODE_TLSF;

// Thread-local override: 0=default, 1=TLSF, 2=CUDA
static __thread int t_thread_mode = 0;

// Real CUDA Runtime API symbols
typedef cudaError_t (*cudaMalloc_fn)(void**, size_t);
typedef cudaError_t (*cudaFree_fn)(void*);
typedef cudaError_t (*cudaMallocAsync_fn)(void**, size_t, cudaStream_t);
typedef cudaError_t (*cudaFreeAsync_fn)(void*, cudaStream_t);

static cudaMalloc_fn real_cudaMalloc = NULL;
static cudaFree_fn real_cudaFree = NULL;
static cudaMallocAsync_fn real_cudaMallocAsync = NULL;
static cudaFreeAsync_fn real_cudaFreeAsync = NULL;

// Real CUDA Driver API symbols
typedef CUresult (*cuMemAlloc_fn)(CUdeviceptr*, size_t);
typedef CUresult (*cuMemFree_fn)(CUdeviceptr);
typedef CUresult (*cuMemAllocAsync_fn)(CUdeviceptr*, size_t, CUstream);
typedef CUresult (*cuMemFreeAsync_fn)(CUdeviceptr, CUstream);
typedef CUresult (*cuMemAllocManaged_fn)(CUdeviceptr*, size_t, unsigned int);
typedef CUresult (*cuMemAllocPitch_fn)(CUdeviceptr*, size_t*, size_t, size_t, unsigned int);
typedef CUresult (*cuMemAllocFromPoolAsync_fn)(CUdeviceptr*, size_t, CUmemoryPool, CUstream);

static cuMemAlloc_fn real_cuMemAlloc_v2 = NULL;
static cuMemFree_fn real_cuMemFree_v2 = NULL;
static cuMemAllocAsync_fn real_cuMemAllocAsync = NULL;
static cuMemFreeAsync_fn real_cuMemFreeAsync = NULL;
static cuMemAllocManaged_fn real_cuMemAllocManaged = NULL;
static cuMemAllocPitch_fn real_cuMemAllocPitch_v2 = NULL;
static cuMemAllocFromPoolAsync_fn real_cuMemAllocFromPoolAsync = NULL;

// Real CUDA Runtime API pool symbols (CUDA 11.2+)
typedef cudaError_t (*cudaMallocFromPoolAsync_fn)(void**, size_t, cudaMemPool_t, cudaStream_t);
typedef cudaError_t (*cudaMallocManaged_fn)(void**, size_t, unsigned int);
typedef cudaError_t (*cudaMallocPitch_fn)(void**, size_t*, size_t, size_t);
typedef cudaError_t (*cudaMallocHost_fn)(void**, size_t);
typedef cudaError_t (*cudaFreeHost_fn)(void*);
typedef cudaError_t (*cudaHostAlloc_fn)(void**, size_t, unsigned int);

static cudaMallocFromPoolAsync_fn real_cudaMallocFromPoolAsync = NULL;
static cudaMallocManaged_fn real_cudaMallocManaged = NULL;
static cudaMallocPitch_fn real_cudaMallocPitch = NULL;
static cudaMallocHost_fn real_cudaMallocHost = NULL;
static cudaFreeHost_fn real_cudaFreeHost = NULL;
static cudaHostAlloc_fn real_cudaHostAlloc = NULL;
// Statistics
static uint64_t g_alloc_count = 0;
static uint64_t g_free_count = 0;
static uint64_t g_bytes_allocated = 0;

// Thread-local recursion guard
static __thread int t_in_hook = 0;

static void resolve_real_cuda(void) {
    if (real_cudaMalloc && real_cudaFree && real_cuMemAlloc_v2 && real_cuMemFree_v2) return;
    pthread_mutex_lock(&g_resolve_mutex);

    // Runtime API
    if (!real_cudaMalloc) {
        real_cudaMalloc = (cudaMalloc_fn)dlsym(RTLD_NEXT, "cudaMalloc");
    }
    if (!real_cudaFree) {
        real_cudaFree = (cudaFree_fn)dlsym(RTLD_NEXT, "cudaFree");
    }
    if (!real_cudaMallocAsync) {
        real_cudaMallocAsync = (cudaMallocAsync_fn)dlsym(RTLD_NEXT, "cudaMallocAsync");
    }
    if (!real_cudaFreeAsync) {
        real_cudaFreeAsync = (cudaFreeAsync_fn)dlsym(RTLD_NEXT, "cudaFreeAsync");
    }

    // Driver API - Basic
    // Resolve _v2 first: libcuda's unversioned cuMemAlloc is just a wrapper
    // that calls cuMemAlloc_v2.  If we resolve to the wrapper, passthrough
    // calls bounce back through our cuMemAlloc_v2 hook → infinite loop.
    // Pointing directly at the _v2 symbol in libcuda avoids this.
    if (!real_cuMemAlloc_v2) {
        real_cuMemAlloc_v2 = (cuMemAlloc_fn)dlsym(RTLD_NEXT, "cuMemAlloc_v2");
    }
    // Fall back to the unversioned symbol only if _v2 is unavailable.
    // Some driver builds may not export the versioned name.
    if (!real_cuMemAlloc_v2) {
        real_cuMemAlloc_v2 = (cuMemAlloc_fn)dlsym(RTLD_NEXT, "cuMemAlloc");
    }
    if (!real_cuMemFree_v2) {
        real_cuMemFree_v2 = (cuMemFree_fn)dlsym(RTLD_NEXT, "cuMemFree_v2");
    }
    if (!real_cuMemFree_v2) {
        real_cuMemFree_v2 = (cuMemFree_fn)dlsym(RTLD_NEXT, "cuMemFree");
    }
    if (!real_cuMemAllocAsync) {
        real_cuMemAllocAsync = (cuMemAllocAsync_fn)dlsym(RTLD_NEXT, "cuMemAllocAsync");
    }
    if (!real_cuMemFreeAsync) {
        real_cuMemFreeAsync = (cuMemFreeAsync_fn)dlsym(RTLD_NEXT, "cuMemFreeAsync");
    }

    // Driver API - Extended
    if (!real_cuMemAllocManaged) {
        real_cuMemAllocManaged = (cuMemAllocManaged_fn)dlsym(RTLD_NEXT, "cuMemAllocManaged");
    }
    if (!real_cuMemAllocPitch_v2) {
        real_cuMemAllocPitch_v2 = (cuMemAllocPitch_fn)dlsym(RTLD_NEXT, "cuMemAllocPitch_v2");
    }
    if (!real_cuMemAllocPitch_v2) {
        real_cuMemAllocPitch_v2 = (cuMemAllocPitch_fn)dlsym(RTLD_NEXT, "cuMemAllocPitch");
    }
    if (!real_cuMemAllocFromPoolAsync) {
        real_cuMemAllocFromPoolAsync = (cuMemAllocFromPoolAsync_fn)dlsym(RTLD_NEXT, "cuMemAllocFromPoolAsync");
    }

    // Runtime API - Extended
    if (!real_cudaMallocFromPoolAsync) {
        real_cudaMallocFromPoolAsync = (cudaMallocFromPoolAsync_fn)dlsym(RTLD_NEXT, "cudaMallocFromPoolAsync");
    }
    if (!real_cudaMallocManaged) {
        real_cudaMallocManaged = (cudaMallocManaged_fn)dlsym(RTLD_NEXT, "cudaMallocManaged");
    }
    if (!real_cudaMallocPitch) {
        real_cudaMallocPitch = (cudaMallocPitch_fn)dlsym(RTLD_NEXT, "cudaMallocPitch");
    }
    if (!real_cudaMallocHost) {
        real_cudaMallocHost = (cudaMallocHost_fn)dlsym(RTLD_NEXT, "cudaMallocHost");
    }
    if (!real_cudaFreeHost) {
        real_cudaFreeHost = (cudaFreeHost_fn)dlsym(RTLD_NEXT, "cudaFreeHost");
    }
    if (!real_cudaHostAlloc) {
        real_cudaHostAlloc = (cudaHostAlloc_fn)dlsym(RTLD_NEXT, "cudaHostAlloc");
    }

    pthread_mutex_unlock(&g_resolve_mutex);
}

static bool use_cuda_allocator(void) {
    if (g_mode == PTX_HOOK_MODE_CUDA) return true;
    if (g_mode == PTX_HOOK_MODE_HYBRID && t_thread_mode == 2) return true;
    return false;
}

static bool use_tlsf_allocator(void) {
    if (g_mode == PTX_HOOK_MODE_CUDA) return false;
    if (g_mode == PTX_HOOK_MODE_HYBRID && t_thread_mode == 2) return false;
    return true;
}

// Public API for thread-local allocator override
void ptx_hook_set_thread_mode(int mode) {
    t_thread_mode = mode;
}

int ptx_hook_get_thread_mode(void) {
    return t_thread_mode;
}

/**
 * Initialize the hook (called lazily on first qualifying allocation)
 */
static bool try_init_runtime(void) {
    if (g_init_state == 2) return true;   // Already initialized
    if (g_init_state == -1) return false; // Disabled

    pthread_mutex_lock(&g_mutex);

    // Double-check after acquiring lock
    if (g_init_state == 2) {
        pthread_mutex_unlock(&g_mutex);
        return true;
    }
    if (g_init_state == -1 || g_init_state == 1) {
        pthread_mutex_unlock(&g_mutex);
        return false;
    }

    // Mark as initializing to prevent recursion
    g_init_state = 1;

    // Check environment
    g_verbose = getenv("PTX_HOOK_VERBOSE") != NULL;

    if (getenv("PTX_HOOK_DISABLE")) {
        if (g_verbose) fprintf(stderr, "[PTX-HOOK] Disabled via environment\n");
        g_init_state = -1;
        pthread_mutex_unlock(&g_mutex);
        return false;
    }

    // Get device ID from environment
    int device_id = 0;
    const char* device_env = getenv("PTX_HOOK_DEVICE");
    if (device_env) {
        device_id = atoi(device_env);
    }

    // Check if a runtime pointer was passed via environment (from Rust)
    const char* runtime_ptr_env = getenv("PTX_RUNTIME_PTR");
    if (runtime_ptr_env) {
        // Reuse existing runtime from Rust
        unsigned long long ptr_val = strtoull(runtime_ptr_env, NULL, 16);
        g_runtime = (GPUHotRuntime*)(uintptr_t)ptr_val;

        if (g_verbose) {
            fprintf(stderr, "[PTX-HOOK] Reusing existing runtime from Rust (ptr=%p)\n", g_runtime);
        }
    } else {
        // Create new runtime (legacy path)
        if (g_verbose) {
            fprintf(stderr, "[PTX-HOOK] Initializing PTX-OS runtime on device %d...\n", device_id);
        }

        // Initialize PTX-OS runtime (this will use real CUDA functions due to recursion guard)
        t_in_hook = 1;
        g_runtime = gpu_hot_init(device_id, NULL);
        t_in_hook = 0;

        if (!g_runtime) {
            fprintf(stderr, "[PTX-HOOK] ERROR: Failed to initialize PTX-OS runtime\n");
            g_init_state = -1;
            pthread_mutex_unlock(&g_mutex);
            return false;
        }

        if (g_verbose) {
            fprintf(stderr, "[PTX-HOOK] Created new runtime (ptr=%p)\n", g_runtime);
        }
    }

    if (g_verbose) {
        const char* mode_label = (g_mode == PTX_HOOK_MODE_HYBRID) ? "hybrid" : "TLSF only";
        fprintf(stderr, "[PTX-HOOK] Ready! Intercepting allocations (%s)\n", mode_label);
    }

    g_init_state = 2;

    // Capture context from runtime for context hook enforcement
    ptx_context_hook_capture_from_runtime();

    pthread_mutex_unlock(&g_mutex);
    return true;
}

/**
 * Constructor - resolve CUDA functions early
 */
__attribute__((constructor))
static void init_hook(void) {
    // Check for verbose mode early
    g_verbose = getenv("PTX_HOOK_VERBOSE") != NULL;

    const char* mode_env = getenv("PTX_HOOK_MODE");
    if (mode_env) {
        if (strcasecmp(mode_env, "cuda") == 0) {
            g_mode = PTX_HOOK_MODE_CUDA;
        } else if (strcasecmp(mode_env, "hybrid") == 0) {
            g_mode = PTX_HOOK_MODE_HYBRID;
        } else {
            g_mode = PTX_HOOK_MODE_TLSF;
        }
    }
    g_hybrid_fallback = getenv("PTX_HOOK_HYBRID_FALLBACK") != NULL;

    if (g_mode != PTX_HOOK_MODE_TLSF) {
        resolve_real_cuda();
    }

    // Initialize context hook subsystem
    ptx_context_hook_init();

    if (g_verbose) {
        fprintf(stderr, "[PTX-HOOK] Loaded (lazy initialization enabled)\n");
    }
}

/**
 * Destructor - cleanup on library unload
 */
__attribute__((destructor))
static void fini_hook(void) {
    if (g_runtime) {
        if (g_verbose) {
            fprintf(stderr, "[PTX-HOOK] Statistics:\n");
            fprintf(stderr, "  Intercepted allocs: %lu\n", g_alloc_count);
            fprintf(stderr, "  Intercepted frees: %lu\n", g_free_count);
        }
        ptx_context_hook_print_stats();
        // Shutdown must use real CUDA frees for backing pool teardown.
        t_in_hook = 1;
        gpu_hot_shutdown(g_runtime);
        t_in_hook = 0;
        g_runtime = NULL;
    }
}

/**
 * Intercepted cudaMalloc
 */
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    // Debug: Always print to see if we're being called
    static int call_count = 0;
    if (g_verbose || ++call_count <= 5) {
        fprintf(stderr, "[PTX-HOOK] cudaMalloc called #%d: size=%zu\n", call_count, size);
    }

    if (!devPtr) {
        return cudaSuccess;
    }

    // Recursion guard - prefer real CUDA if available
    if (t_in_hook) {
        if (use_cuda_allocator()) {
            resolve_real_cuda();
            if (real_cudaMalloc) {
                return real_cudaMalloc(devPtr, size);
            }
        }
        return cudaErrorMemoryAllocation;
    }

    if (g_init_state == 1) {
        return cudaErrorMemoryAllocation;
    }

    if (use_cuda_allocator()) {
        resolve_real_cuda();
        if (!real_cudaMalloc) {
            return cudaErrorMemoryAllocation;
        }
        return real_cudaMalloc(devPtr, size);
    }

    // Try to initialize runtime if needed
    if (g_init_state == 0 && use_tlsf_allocator()) {
        if (!try_init_runtime()) {
            if (g_mode == PTX_HOOK_MODE_HYBRID && g_hybrid_fallback) {
                resolve_real_cuda();
                if (real_cudaMalloc) {
                    return real_cudaMalloc(devPtr, size);
                }
            }
            return cudaErrorMemoryAllocation;
        }
    }

    // If disabled or not ready, fail
    if (g_init_state != 2 || !g_runtime) {
        if (g_mode == PTX_HOOK_MODE_HYBRID && g_hybrid_fallback) {
            resolve_real_cuda();
            if (real_cudaMalloc) {
                return real_cudaMalloc(devPtr, size);
            }
        }
        return cudaErrorMemoryAllocation;
    }

    // Allocate via PTX-OS
    t_in_hook = 1;
    void* ptr = gpu_hot_alloc(g_runtime, size);
    t_in_hook = 0;

    if (!ptr) {
        if (g_verbose) {
            fprintf(stderr, "[PTX-HOOK] Allocation failed: %zu bytes\n", size);
        }
        if (g_mode == PTX_HOOK_MODE_HYBRID && g_hybrid_fallback) {
            resolve_real_cuda();
            if (real_cudaMalloc) {
                return real_cudaMalloc(devPtr, size);
            }
        }
        return cudaErrorMemoryAllocation;
    }

    *devPtr = ptr;
    __sync_fetch_and_add(&g_alloc_count, 1);
    __sync_fetch_and_add(&g_bytes_allocated, size);

    if (g_verbose) {
        fprintf(stderr, "[PTX-HOOK] cudaMalloc(%zu) = %p\n", size, ptr);
    }

    return cudaSuccess;
}

/**
 * Intercepted cudaFree
 */
cudaError_t cudaFree(void* devPtr) {
    if (!devPtr) {
        return cudaSuccess;
    }
    // Recursion guard
    if (t_in_hook) {
        if (use_cuda_allocator()) {
            resolve_real_cuda();
            if (real_cudaFree) {
                return real_cudaFree(devPtr);
            }
        }
        return cudaErrorInvalidDevicePointer;
    }

    // Prefer TLSF free when pointer is owned by PTX-OS
    if (g_init_state == 2 && g_runtime) {
        t_in_hook = 1;
        bool owns = gpu_hot_owns_ptr(g_runtime, devPtr);
        t_in_hook = 0;

        if (owns) {
            if (g_verbose) {
                fprintf(stderr, "[PTX-HOOK] cudaFree(%p)\n", devPtr);
            }
            t_in_hook = 1;
            gpu_hot_free(g_runtime, devPtr);
            t_in_hook = 0;
            __sync_fetch_and_add(&g_free_count, 1);
            return cudaSuccess;
        }
    }

    // Not a TLSF pointer (or runtime not ready)
    if (use_cuda_allocator() || g_mode == PTX_HOOK_MODE_HYBRID) {
        resolve_real_cuda();
        if (real_cudaFree) {
            return real_cudaFree(devPtr);
        }
    }

    ptx_strict_free_violation("PTX-HOOK", devPtr);
    return cudaErrorInvalidDevicePointer;
}

/**
 * Intercepted cudaMallocAsync
 */
cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream) {
    if (t_in_hook) {
        return cudaErrorMemoryAllocation;
    }

    if (!devPtr) {
        return cudaSuccess;
    }

    if (use_cuda_allocator()) {
        resolve_real_cuda();
        if (real_cudaMallocAsync) {
            return real_cudaMallocAsync(devPtr, size, stream);
        }
        if (real_cudaMalloc) {
            return real_cudaMalloc(devPtr, size);
        }
        return cudaErrorMemoryAllocation;
    }

    if (g_init_state == 0 && use_tlsf_allocator()) {
        if (!try_init_runtime()) {
            if (g_mode == PTX_HOOK_MODE_HYBRID && g_hybrid_fallback) {
                resolve_real_cuda();
                if (real_cudaMallocAsync) {
                    return real_cudaMallocAsync(devPtr, size, stream);
                }
                if (real_cudaMalloc) {
                    return real_cudaMalloc(devPtr, size);
                }
            }
            return cudaErrorMemoryAllocation;
        }
    }

    if (g_init_state != 2 || !g_runtime) {
        if (g_mode == PTX_HOOK_MODE_HYBRID && g_hybrid_fallback) {
            resolve_real_cuda();
            if (real_cudaMallocAsync) {
                return real_cudaMallocAsync(devPtr, size, stream);
            }
            if (real_cudaMalloc) {
                return real_cudaMalloc(devPtr, size);
            }
        }
        return cudaErrorMemoryAllocation;
    }

    t_in_hook = 1;
    void* ptr = gpu_hot_alloc_async(g_runtime, size, stream);
    t_in_hook = 0;

    if (!ptr) {
        if (g_mode == PTX_HOOK_MODE_HYBRID && g_hybrid_fallback) {
            resolve_real_cuda();
            if (real_cudaMallocAsync) {
                return real_cudaMallocAsync(devPtr, size, stream);
            }
            if (real_cudaMalloc) {
                return real_cudaMalloc(devPtr, size);
            }
        }
        return cudaErrorMemoryAllocation;
    }
    *devPtr = ptr;
    __sync_fetch_and_add(&g_alloc_count, 1);
    __sync_fetch_and_add(&g_bytes_allocated, size);
    return cudaSuccess;
}

/**
 * Intercepted cudaFreeAsync
 */
cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream) {
    if (t_in_hook) {
        return cudaErrorInvalidDevicePointer;
    }

    if (!devPtr) {
        return cudaSuccess;
    }

    // Prefer TLSF free when pointer is owned by PTX-OS
    if (g_init_state == 2 && g_runtime) {
        t_in_hook = 1;
        bool owns = gpu_hot_owns_ptr(g_runtime, devPtr);
        t_in_hook = 0;

        if (owns) {
            t_in_hook = 1;
            gpu_hot_free_async(g_runtime, devPtr, stream);
            t_in_hook = 0;
            __sync_fetch_and_add(&g_free_count, 1);
            return cudaSuccess;
        }
    }

    // Not a TLSF pointer (or runtime not ready)
    if (use_cuda_allocator() || g_mode == PTX_HOOK_MODE_HYBRID) {
        resolve_real_cuda();
        if (real_cudaFreeAsync) {
            return real_cudaFreeAsync(devPtr, stream);
        }
        if (real_cudaFree) {
            return real_cudaFree(devPtr);
        }
    }
    if (g_mode == PTX_HOOK_MODE_TLSF) {
        ptx_strict_free_violation("PTX-HOOK", devPtr);
    }
    return cudaErrorInvalidDevicePointer;
}

// ============================================================================
// CUDA Driver API Interceptions
// ============================================================================

/**
 * Intercepted cuMemAlloc_v2 (Driver API - canonical)
 * cuda.h maps cuMemAlloc -> cuMemAlloc_v2 for most callers.
 */
CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    // Debug: Always print to see if we're being called
    static int call_count = 0;
    if (g_verbose || ++call_count <= 5) {
        fprintf(stderr, "[PTX-HOOK] cuMemAlloc_v2 called #%d: size=%zu\n", call_count, bytesize);
    }

    if (!dptr) {
        return CUDA_SUCCESS;
    }

    // Recursion guard
    if (t_in_hook) {
        resolve_real_cuda();
        if (real_cuMemAlloc_v2) {
            return real_cuMemAlloc_v2(dptr, bytesize);
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    // Use real CUDA if disabled or in CUDA mode
    if (use_cuda_allocator()) {
        resolve_real_cuda();
        if (real_cuMemAlloc_v2) {
            return real_cuMemAlloc_v2(dptr, bytesize);
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    // Initialize runtime if needed
    if (!try_init_runtime()) {
        resolve_real_cuda();
        if (real_cuMemAlloc_v2) {
            return real_cuMemAlloc_v2(dptr, bytesize);
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    // Allocate through TLSF
    void* ptr = gpu_hot_alloc(g_runtime, bytesize);
    if (!ptr) {
        if (g_hybrid_fallback && g_mode == PTX_HOOK_MODE_HYBRID) {
            resolve_real_cuda();
            if (real_cuMemAlloc_v2) {
                return real_cuMemAlloc_v2(dptr, bytesize);
            }
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    *dptr = (CUdeviceptr)ptr;
    g_alloc_count++;
    g_bytes_allocated += bytesize;

    if (g_verbose) {
        fprintf(stderr, "[PTX-HOOK] cuMemAlloc_v2(%zu) = %p (TLSF)\n", bytesize, ptr);
    }

    return CUDA_SUCCESS;
}

/**
 * Intercepted cuMemAlloc (Driver API - compatibility alias)
 */
CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    return cuMemAlloc_v2(dptr, bytesize);
}

/**
 * Intercepted cuMemFree_v2 (Driver API - canonical)
 */
CUresult cuMemFree_v2(CUdeviceptr dptr) {
    void* ptr = (void*)dptr;

    if (!ptr) {
        return CUDA_SUCCESS;
    }

    // Recursion guard
    if (t_in_hook) {
        resolve_real_cuda();
        if (real_cuMemFree_v2) {
            return real_cuMemFree_v2(dptr);
        }
        return CUDA_SUCCESS;
    }

    // Check if this is a TLSF pointer
    if (g_runtime && gpu_hot_owns_ptr(g_runtime, ptr)) {
        gpu_hot_free(g_runtime, ptr);
        g_free_count++;

        if (g_verbose) {
            fprintf(stderr, "[PTX-HOOK] cuMemFree(%p) (TLSF)\n", ptr);
        }

        return CUDA_SUCCESS;
    }

    // Not a TLSF pointer
    if (use_cuda_allocator() || g_mode == PTX_HOOK_MODE_HYBRID) {
        resolve_real_cuda();
        if (real_cuMemFree_v2) {
            return real_cuMemFree_v2(dptr);
        }
    }
    if (g_mode == PTX_HOOK_MODE_TLSF) {
        ptx_strict_free_violation("PTX-HOOK", ptr);
    }
    return CUDA_SUCCESS;
}

/**
 * Intercepted cuMemFree (Driver API - compatibility alias)
 */
CUresult cuMemFree(CUdeviceptr dptr) {
    return cuMemFree_v2(dptr);
}

/**
 * Intercepted cuMemAllocAsync (Driver API)
 */
CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) {
    if (!dptr) {
        return CUDA_SUCCESS;
    }

    // For now, treat async as sync (TLSF allocations are fast enough)
    // TODO: Could use gpu_hot_alloc_async if needed
    return cuMemAlloc_v2(dptr, bytesize);
}

/**
 * Intercepted cuMemFreeAsync (Driver API)
 */
CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    // For now, treat async as sync
    // TODO: Could use gpu_hot_free_async if needed
    return cuMemFree_v2(dptr);
}

// ============================================================================
// Extended CUDA Memory API Interceptions (Pools, Pitched, Managed, Host)
// ============================================================================

/**
 * Intercepted cudaMallocFromPoolAsync (Runtime API - CUDA 11.2+)
 */
cudaError_t cudaMallocFromPoolAsync(void** devPtr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) {
    static int call_count = 0;
    if (g_verbose || ++call_count <= 5) {
        fprintf(stderr, "[PTX-HOOK] cudaMallocFromPoolAsync called #%d: size=%zu\n", call_count, size);
    }

    // Redirect pool allocations to TLSF (pools are incompatible with our allocator)
    if (!devPtr) return cudaSuccess;

    if (t_in_hook || use_cuda_allocator()) {
        resolve_real_cuda();
        if (real_cudaMallocFromPoolAsync) {
            return real_cudaMallocFromPoolAsync(devPtr, size, memPool, stream);
        }
        return cudaErrorMemoryAllocation;
    }

    if (!try_init_runtime()) {
        resolve_real_cuda();
        if (real_cudaMallocFromPoolAsync) {
            return real_cudaMallocFromPoolAsync(devPtr, size, memPool, stream);
        }
        return cudaErrorMemoryAllocation;
    }

    // Allocate through TLSF instead of pool
    void* ptr = gpu_hot_alloc(g_runtime, size);
    if (!ptr) {
        if (g_hybrid_fallback && g_mode == PTX_HOOK_MODE_HYBRID) {
            resolve_real_cuda();
            if (real_cudaMallocFromPoolAsync) {
                return real_cudaMallocFromPoolAsync(devPtr, size, memPool, stream);
            }
        }
        return cudaErrorMemoryAllocation;
    }

    *devPtr = ptr;
    g_alloc_count++;
    g_bytes_allocated += size;

    if (g_verbose) {
        fprintf(stderr, "[PTX-HOOK] cudaMallocFromPoolAsync(%zu) = %p (TLSF override)\n", size, ptr);
    }

    return cudaSuccess;
}

/**
 * Intercepted cudaMallocManaged (Runtime API)
 */
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {
    static int call_count = 0;
    if (g_verbose || ++call_count <= 5) {
        fprintf(stderr, "[PTX-HOOK] cudaMallocManaged called #%d: size=%zu\n", call_count, size);
    }

    // Managed memory not supported by TLSF - fall back to real CUDA
    if (g_verbose) {
        fprintf(stderr, "[PTX-HOOK] WARNING: Managed memory not supported by TLSF, using real cudaMallocManaged\n");
    }

    resolve_real_cuda();
    if (real_cudaMallocManaged) {
        return real_cudaMallocManaged(devPtr, size, flags);
    }
    return cudaErrorMemoryAllocation;
}

/**
 * Intercepted cudaMallocPitch (Runtime API)
 */
cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) {
    static int call_count = 0;
    size_t size = width * height;
    if (g_verbose || ++call_count <= 5) {
        fprintf(stderr, "[PTX-HOOK] cudaMallocPitch called #%d: %zux%zu\n", call_count, width, height);
    }

    if (!devPtr || !pitch) return cudaSuccess;

    if (t_in_hook || use_cuda_allocator()) {
        resolve_real_cuda();
        if (real_cudaMallocPitch) {
            return real_cudaMallocPitch(devPtr, pitch, width, height);
        }
        return cudaErrorMemoryAllocation;
    }

    if (!try_init_runtime()) {
        resolve_real_cuda();
        if (real_cudaMallocPitch) {
            return real_cudaMallocPitch(devPtr, pitch, width, height);
        }
        return cudaErrorMemoryAllocation;
    }

    // Allocate through TLSF with alignment
    *pitch = ((width + 255) / 256) * 256;  // 256-byte alignment
    size_t total_size = (*pitch) * height;

    void* ptr = gpu_hot_alloc(g_runtime, total_size);
    if (!ptr) {
        if (g_hybrid_fallback && g_mode == PTX_HOOK_MODE_HYBRID) {
            resolve_real_cuda();
            if (real_cudaMallocPitch) {
                return real_cudaMallocPitch(devPtr, pitch, width, height);
            }
        }
        return cudaErrorMemoryAllocation;
    }

    *devPtr = ptr;
    g_alloc_count++;
    g_bytes_allocated += total_size;

    if (g_verbose) {
        fprintf(stderr, "[PTX-HOOK] cudaMallocPitch(%zux%zu) = %p, pitch=%zu (TLSF)\n", 
                width, height, ptr, *pitch);
    }

    return cudaSuccess;
}

/**
 * Intercepted cudaMallocHost / cudaHostAlloc (Runtime API)
 * Note: Host memory is NOT managed by TLSF (it's CPU memory)
 */
cudaError_t cudaMallocHost(void** ptr, size_t size) {
    // Host memory - pass through to real CUDA
    resolve_real_cuda();
    if (real_cudaMallocHost) {
        return real_cudaMallocHost(ptr, size);
    }
    return cudaErrorMemoryAllocation;
}

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
    // Host memory - pass through to real CUDA
    resolve_real_cuda();
    if (real_cudaHostAlloc) {
        return real_cudaHostAlloc(pHost, size, flags);
    }
    return cudaErrorMemoryAllocation;
}

cudaError_t cudaFreeHost(void* ptr) {
    // Host memory - pass through to real CUDA
    resolve_real_cuda();
    if (real_cudaFreeHost) {
        return real_cudaFreeHost(ptr);
    }
    return cudaSuccess;
}

/**
 * Intercepted cuMemAllocManaged (Driver API)
 */
CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
    // Managed memory not supported - fall back
    if (g_verbose) {
        fprintf(stderr, "[PTX-HOOK] WARNING: Managed memory not supported, using real cuMemAllocManaged\n");
    }
    resolve_real_cuda();
    if (real_cuMemAllocManaged) {
        return real_cuMemAllocManaged(dptr, bytesize, flags);
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
}

/**
 * Intercepted cuMemAllocPitch_v2 (Driver API - canonical)
 */
CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes,
                            size_t Height, unsigned int ElementSizeBytes) {
    static int call_count = 0;
    if (g_verbose || ++call_count <= 5) {
        fprintf(stderr, "[PTX-HOOK] cuMemAllocPitch called #%d\n", call_count);
    }

    if (!dptr || !pPitch) return CUDA_SUCCESS;

    if (t_in_hook || use_cuda_allocator()) {
        resolve_real_cuda();
        if (real_cuMemAllocPitch_v2) {
            return real_cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    if (!try_init_runtime()) {
        resolve_real_cuda();
        if (real_cuMemAllocPitch_v2) {
            return real_cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    // Allocate through TLSF
    *pPitch = ((WidthInBytes + 255) / 256) * 256;
    size_t total_size = (*pPitch) * Height;

    void* ptr = gpu_hot_alloc(g_runtime, total_size);
    if (!ptr) {
        if (g_hybrid_fallback && g_mode == PTX_HOOK_MODE_HYBRID) {
            resolve_real_cuda();
            if (real_cuMemAllocPitch_v2) {
                return real_cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
            }
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    *dptr = (CUdeviceptr)ptr;
    g_alloc_count++;
    g_bytes_allocated += total_size;

    if (g_verbose) {
        fprintf(stderr, "[PTX-HOOK] cuMemAllocPitch = %p, pitch=%zu (TLSF)\n", ptr, *pPitch);
    }

    return CUDA_SUCCESS;
}

/**
 * Intercepted cuMemAllocPitch (Driver API - compatibility alias)
 */
CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes,
                         size_t Height, unsigned int ElementSizeBytes) {
    return cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

/**
 * Intercepted cuMemAllocFromPoolAsync (Driver API - CUDA 11.2+)
 */
CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize,
                                  CUmemoryPool pool, CUstream hStream) {
    static int call_count = 0;
    if (g_verbose || ++call_count <= 5) {
        fprintf(stderr, "[PTX-HOOK] cuMemAllocFromPoolAsync called #%d: size=%zu\n", call_count, bytesize);
    }

    // Redirect pool allocations to TLSF
    if (!dptr) return CUDA_SUCCESS;

    if (t_in_hook || use_cuda_allocator()) {
        resolve_real_cuda();
        if (real_cuMemAllocFromPoolAsync) {
            return real_cuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream);
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    if (!try_init_runtime()) {
        resolve_real_cuda();
        if (real_cuMemAllocFromPoolAsync) {
            return real_cuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream);
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    // Use TLSF instead of pool
    void* ptr = gpu_hot_alloc(g_runtime, bytesize);
    if (!ptr) {
        if (g_hybrid_fallback && g_mode == PTX_HOOK_MODE_HYBRID) {
            resolve_real_cuda();
            if (real_cuMemAllocFromPoolAsync) {
                return real_cuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream);
            }
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    *dptr = (CUdeviceptr)ptr;
    g_alloc_count++;
    g_bytes_allocated += bytesize;

    if (g_verbose) {
        fprintf(stderr, "[PTX-HOOK] cuMemAllocFromPoolAsync(%zu) = %p (TLSF override)\n", bytesize, ptr);
    }

    return CUDA_SUCCESS;
}
