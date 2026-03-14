/*
 * CUDA Allocation Hook - Routes ALL CUDA allocations through PTX-OS TLSF
 *
 * This intercepts cuMemAlloc/cudaMalloc at runtime using dlsym(RTLD_NEXT)
 * to find the original functions, then routes allocations through TLSF.
 *
 * The hooks are enabled when ptx_hook_init() is called with a valid runtime.
 */

#include "gpu/gpu_hot_runtime.h"
#include "ptx_debug.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <dlfcn.h>

// ============================================================================
// Global State
// ============================================================================

static GPUHotRuntime* g_ptx_runtime = NULL;
static bool g_hook_enabled = false;
static bool g_hook_verbose = false;


// ============================================================================
// Hook Management API
// ============================================================================

extern "C" void ptx_hook_init(GPUHotRuntime* runtime, bool verbose) {
    if (!runtime) return;

    g_ptx_runtime = runtime;
    g_hook_verbose = verbose;

    // No fallback to legacy CUDA allocators in strict mode

    g_hook_enabled = true;

    if (g_hook_verbose) {
        printf("[PTX-HOOK] CUDA allocation hooks initialized\n");
        printf("[PTX-HOOK]   All GPU allocations will use TLSF pool\n");
    }
}

extern "C" void ptx_hook_disable() {
    g_hook_enabled = false;
    g_ptx_runtime = NULL;
    if (g_hook_verbose) {
        printf("[PTX-HOOK] Hooks disabled\n");
    }
}

extern "C" bool ptx_hook_owns_ptr(void* ptr) {
    if (!g_ptx_runtime) return false;
    return gpu_hot_owns_ptr(g_ptx_runtime, ptr);
}

extern "C" bool ptx_hook_is_enabled() {
    return g_hook_enabled && g_ptx_runtime != NULL;
}

// ============================================================================
// Allocation Functions for External Use
// ============================================================================

// These are the functions that should be called by the application
// They route through TLSF when hooks are enabled

extern "C" void* ptx_cuda_alloc(size_t size) {
    if (g_hook_enabled && g_ptx_runtime) {
        void* ptr = gpu_hot_alloc(g_ptx_runtime, size);
        if (ptr) {
            if (g_hook_verbose) {
                printf("[PTX-HOOK] alloc(%zu) -> %p (TLSF)\n", size, ptr);
            }
            return ptr;
        }
    }

    if (g_hook_verbose) {
        printf("[PTX-HOOK] alloc(%zu) failed (TLSF only)\n", size);
    }
    return NULL;
}

extern "C" void ptx_cuda_free(void* ptr) {
    if (!ptr) return;

    if (g_hook_enabled && g_ptx_runtime) {
        if (gpu_hot_owns_ptr(g_ptx_runtime, ptr)) {
            if (g_hook_verbose) {
                printf("[PTX-HOOK] free(%p) -> TLSF\n", ptr);
            }
            gpu_hot_free(g_ptx_runtime, ptr);
            return;
        }
    }

    // Not a TLSF pointer
    if (g_hook_verbose) {
        printf("[PTX-HOOK] free(%p) rejected (non-PTX allocation)\n", ptr);
    }
    ptx_strict_free_violation("PTX-HOOK", ptr);
}

// ============================================================================
// Direct TLSF wrappers (bypass hook check for internal use)
// ============================================================================

extern "C" void* ptx_tlsf_cuda_alloc(size_t size) {
    if (!g_ptx_runtime) return NULL;
    return gpu_hot_alloc(g_ptx_runtime, size);
}

extern "C" void ptx_tlsf_cuda_free(void* ptr) {
    if (!g_ptx_runtime || !ptr) return;
    gpu_hot_free(g_ptx_runtime, ptr);
}

// ============================================================================
// Kernel Launch Acceleration
// ============================================================================

// Stats for kernel launches
static uint64_t g_total_launches = 0;
static int g_current_stream_idx = 0;

// Original kernel launch function
typedef CUresult (*cuLaunchKernel_fn)(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra);

static cuLaunchKernel_fn orig_cuLaunchKernel = NULL;

// Get next stream from PTX-OS pool (round-robin)
static CUstream ptx_get_next_stream() {
    if (!g_ptx_runtime) return NULL;

    int num_streams = 16;  // PTX-OS default
    CUstream stream = (CUstream)gpu_hot_get_stream(g_ptx_runtime, g_current_stream_idx);
    g_current_stream_idx = (g_current_stream_idx + 1) % num_streams;
    return stream;
}

// Hooked kernel launch - routes through PTX-OS streams
extern "C" CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra)
{
    // Resolve original function on first call
    if (!orig_cuLaunchKernel) {
        orig_cuLaunchKernel = (cuLaunchKernel_fn)dlsym(RTLD_NEXT, "cuLaunchKernel");
        if (!orig_cuLaunchKernel) {
            return CUDA_ERROR_NOT_FOUND;
        }
    }

    // If hooks enabled and stream is NULL/default, use our stream pool
    CUstream targetStream = hStream;
    if (g_hook_enabled && g_ptx_runtime && (hStream == NULL || hStream == 0)) {
        targetStream = ptx_get_next_stream();
    }

    g_total_launches++;

    if (g_hook_verbose) {
        printf("[PTX-HOOK] kernel launch #%lu -> stream %p\n",
               g_total_launches, (void*)targetStream);
    }

    return orig_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                blockDimX, blockDimY, blockDimZ,
                                sharedMemBytes, targetStream,
                                kernelParams, extra);
}

// Get launch stats
extern "C" uint64_t ptx_hook_get_launch_count() {
    return g_total_launches;
}

extern "C" void ptx_hook_reset_launch_count() {
    g_total_launches = 0;
}
