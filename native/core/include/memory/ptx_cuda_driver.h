#ifndef PTX_CUDA_DRIVER_H
#define PTX_CUDA_DRIVER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static inline int ptx_driver_warn_enabled(void) {
    const char* v = getenv("PTX_DRIVER_WARN");
    return v && v[0] && v[0] != '0';
}

static inline const char* ptx_driver_err_str(CUresult res) {
    const char* msg = NULL;
    if (cuGetErrorString(res, &msg) == CUDA_SUCCESS && msg) {
        return msg;
    }
    return "Unknown CUDA driver error";
}

static inline CUresult ptx_driver_ensure_context(void) {
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        return res;
    }

    CUcontext ctx = NULL;
    res = cuCtxGetCurrent(&ctx);
    if (res != CUDA_SUCCESS) {
        return res;
    }

    if (!ctx) {
        int dev = 0;
        cudaError_t cerr = cudaGetDevice(&dev);
        if (cerr != cudaSuccess) {
            return CUDA_ERROR_INVALID_DEVICE;
        }

        CUdevice cu_dev;
        res = cuDeviceGet(&cu_dev, dev);
        if (res != CUDA_SUCCESS) {
            return res;
        }

        res = cuDevicePrimaryCtxRetain(&ctx, cu_dev);
        if (res != CUDA_SUCCESS) {
            return res;
        }

        res = cuCtxSetCurrent(ctx);
        if (res != CUDA_SUCCESS) {
            return res;
        }
    }

    return CUDA_SUCCESS;
}

static inline void* ptx_driver_alloc(size_t size) {
    if (size == 0) return NULL;

    CUresult res = ptx_driver_ensure_context();
    if (res != CUDA_SUCCESS) {
        if (ptx_driver_warn_enabled()) {
            fprintf(stderr, "[PTX-DRIVER] Failed to init context: %s\n", ptx_driver_err_str(res));
        }
        return NULL;
    }

    CUdeviceptr dptr = 0;
    res = cuMemAlloc(&dptr, size);
    if (res != CUDA_SUCCESS) {
        if (ptx_driver_warn_enabled()) {
            fprintf(stderr, "[PTX-DRIVER] cuMemAlloc failed: %s\n", ptx_driver_err_str(res));
        }
        return NULL;
    }

    return (void*)(uintptr_t)dptr;
}

static inline void ptx_driver_free(void* ptr) {
    if (!ptr) return;

    CUresult init_res = ptx_driver_ensure_context();
    if (init_res != CUDA_SUCCESS) {
        // Common during process teardown when CUDA is already deinitialized.
        // Best effort free: skip noisy warnings in that path.
        return;
    }

    CUdeviceptr dptr = (CUdeviceptr)(uintptr_t)ptr;
    CUresult res = cuMemFree(dptr);
    if (res != CUDA_SUCCESS) {
        if (res == CUDA_ERROR_DEINITIALIZED || res == CUDA_ERROR_INVALID_CONTEXT) {
            return;
        }
        if (ptx_driver_warn_enabled()) {
            fprintf(stderr, "[PTX-DRIVER] cuMemFree failed (code=%d): %s\n",
                    (int)res, ptx_driver_err_str(res));
        }
    }
}

#endif // PTX_CUDA_DRIVER_H
