#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

#include "fercuda/api/intercept_telemetry.h"

namespace {

struct GPUHotRuntime;

struct GPUHotConfig {
    float pool_fraction;
    size_t fixed_pool_size;
    size_t min_pool_size;
    size_t max_pool_size;
    size_t reserve_vram;
    bool enable_leak_detection;
    bool enable_pool_health;
    float warning_threshold;
    bool force_daemon_mode;
    bool quiet_init;
};

using gpu_hot_init_fn = GPUHotRuntime* (*)(int device_id, const char* token);
using gpu_hot_init_with_config_fn = GPUHotRuntime* (*)(int device_id, const char* token, const GPUHotConfig* config);
using gpu_hot_default_config_fn = GPUHotConfig (*)();
using gpu_hot_shutdown_fn = void (*)(GPUHotRuntime* runtime);
using gpu_hot_alloc_fn = void* (*)(GPUHotRuntime* runtime, size_t size);
using gpu_hot_free_fn = void (*)(GPUHotRuntime* runtime, void* ptr);
using gpu_hot_runtime_validate_fn = bool (*)(GPUHotRuntime* runtime);

using cu_mem_alloc_v2_fn = CUresult (*)(CUdeviceptr* dptr, size_t bytesize);
using cu_mem_free_v2_fn = CUresult (*)(CUdeviceptr dptr);
using cu_mem_alloc_async_fn = CUresult (*)(CUdeviceptr* dptr, size_t bytesize, CUstream hStream);
using cu_mem_free_async_fn = CUresult (*)(CUdeviceptr dptr, CUstream hStream);

using cuda_malloc_fn = cudaError_t (*)(void** devPtr, size_t size);
using cuda_free_fn = cudaError_t (*)(void* devPtr);
using cuda_malloc_async_fn = cudaError_t (*)(void** devPtr, size_t size, cudaStream_t hStream);
using cuda_free_async_fn = cudaError_t (*)(void* devPtr, cudaStream_t hStream);
using cuda_get_device_fn = cudaError_t (*)(int* device);

enum class Owner : uint8_t {
    TLSF = 1,
};

struct State {
    std::once_flag init_once;
    std::mutex owner_mu;
    std::unordered_map<void*, Owner> owners;

    bool enabled = true;
    bool log = false;
    bool async_tlsf = false;
    bool initialized = false;

    void* gpu_hot_handle = nullptr;
    GPUHotRuntime* gpu_hot_runtime = nullptr;
    gpu_hot_init_fn gpu_hot_init = nullptr;
    gpu_hot_init_with_config_fn gpu_hot_init_with_config = nullptr;
    gpu_hot_default_config_fn gpu_hot_default_config = nullptr;
    gpu_hot_shutdown_fn gpu_hot_shutdown = nullptr;
    gpu_hot_alloc_fn gpu_hot_alloc = nullptr;
    gpu_hot_free_fn gpu_hot_free = nullptr;
    gpu_hot_runtime_validate_fn gpu_hot_runtime_validate = nullptr;

    cu_mem_alloc_v2_fn real_cu_mem_alloc_v2 = nullptr;
    cu_mem_free_v2_fn real_cu_mem_free_v2 = nullptr;
    cu_mem_alloc_async_fn real_cu_mem_alloc_async = nullptr;
    cu_mem_free_async_fn real_cu_mem_free_async = nullptr;

    cuda_malloc_fn real_cuda_malloc = nullptr;
    cuda_free_fn real_cuda_free = nullptr;
    cuda_malloc_async_fn real_cuda_malloc_async = nullptr;
    cuda_free_async_fn real_cuda_free_async = nullptr;
    cuda_get_device_fn real_cuda_get_device = nullptr;
};

State& state() {
    static State s{};
    return s;
}

struct Telemetry {
    std::atomic<uint64_t> init_calls{0};
    std::atomic<uint64_t> init_success{0};
    std::atomic<uint64_t> init_fail{0};

    std::atomic<uint64_t> alloc_calls_total{0};
    std::atomic<uint64_t> free_calls_total{0};

    std::atomic<uint64_t> alloc_calls_driver{0};
    std::atomic<uint64_t> free_calls_driver{0};
    std::atomic<uint64_t> alloc_calls_runtime{0};
    std::atomic<uint64_t> free_calls_runtime{0};

    std::atomic<uint64_t> alloc_calls_async{0};
    std::atomic<uint64_t> free_calls_async{0};

    std::atomic<uint64_t> alloc_bytes_requested{0};

    std::atomic<uint64_t> tlsf_alloc_success{0};
    std::atomic<uint64_t> tlsf_alloc_fail{0};
    std::atomic<uint64_t> tlsf_free_success{0};
    std::atomic<uint64_t> tlsf_free_miss{0};

    std::atomic<uint64_t> fallback_alloc_calls{0};
    std::atomic<uint64_t> fallback_free_calls{0};
};

Telemetry& telemetry() {
    static Telemetry t{};
    return t;
}

thread_local bool g_in_hook = false;

bool env_enabled(const char* key, bool def) {
    const char* v = std::getenv(key);
    if (!v) return def;
    if (v[0] == '0') return false;
    if (v[0] == 'n' || v[0] == 'N') return false;
    if (v[0] == 'f' || v[0] == 'F') return false;
    return true;
}

uint64_t env_u64(const char* key, uint64_t def) {
    const char* v = std::getenv(key);
    if (!v || v[0] == '\0') return def;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(v, &end, 10);
    if (!end || *end != '\0') return def;
    return static_cast<uint64_t>(parsed);
}

void* resolve_next(const char* name) {
    return dlsym(RTLD_NEXT, name);
}

void* resolve_default_or_handle(void* handle, const char* name) {
    void* p = dlsym(RTLD_DEFAULT, name);
    if (p) return p;
    if (handle) return dlsym(handle, name);
    return nullptr;
}

void resolve_real_cuda_symbols(State& s) {
    if (!s.real_cu_mem_alloc_v2) s.real_cu_mem_alloc_v2 = reinterpret_cast<cu_mem_alloc_v2_fn>(resolve_next("cuMemAlloc_v2"));
    if (!s.real_cu_mem_free_v2) s.real_cu_mem_free_v2 = reinterpret_cast<cu_mem_free_v2_fn>(resolve_next("cuMemFree_v2"));
    if (!s.real_cu_mem_alloc_async) s.real_cu_mem_alloc_async = reinterpret_cast<cu_mem_alloc_async_fn>(resolve_next("cuMemAllocAsync"));
    if (!s.real_cu_mem_free_async) s.real_cu_mem_free_async = reinterpret_cast<cu_mem_free_async_fn>(resolve_next("cuMemFreeAsync"));
    if (!s.real_cuda_malloc) s.real_cuda_malloc = reinterpret_cast<cuda_malloc_fn>(resolve_next("cudaMalloc"));
    if (!s.real_cuda_free) s.real_cuda_free = reinterpret_cast<cuda_free_fn>(resolve_next("cudaFree"));
    if (!s.real_cuda_malloc_async) s.real_cuda_malloc_async = reinterpret_cast<cuda_malloc_async_fn>(resolve_next("cudaMallocAsync"));
    if (!s.real_cuda_free_async) s.real_cuda_free_async = reinterpret_cast<cuda_free_async_fn>(resolve_next("cudaFreeAsync"));
    if (!s.real_cuda_get_device) s.real_cuda_get_device = reinterpret_cast<cuda_get_device_fn>(resolve_next("cudaGetDevice"));
}

void resolve_gpu_hot_symbols(State& s) {
    const char* so_path = std::getenv("FERCUDA_GPU_HOT_SO");
    static std::string fallback_so;
    if (so_path && so_path[0] != '\0') {
        s.gpu_hot_handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    }
    if (!s.gpu_hot_handle) {
        const char* home = std::getenv("HOME");
        if (home && home[0] != '\0') {
            fallback_so = std::string(home) + "/persistant_gpu_os/build/libptx_os_shared.so";
            s.gpu_hot_handle = dlopen(fallback_so.c_str(), RTLD_NOW | RTLD_LOCAL);
        }
    }

    s.gpu_hot_init = reinterpret_cast<gpu_hot_init_fn>(resolve_default_or_handle(s.gpu_hot_handle, "gpu_hot_init"));
    s.gpu_hot_init_with_config = reinterpret_cast<gpu_hot_init_with_config_fn>(
        resolve_default_or_handle(s.gpu_hot_handle, "gpu_hot_init_with_config"));
    s.gpu_hot_default_config = reinterpret_cast<gpu_hot_default_config_fn>(
        resolve_default_or_handle(s.gpu_hot_handle, "gpu_hot_default_config"));
    s.gpu_hot_shutdown = reinterpret_cast<gpu_hot_shutdown_fn>(resolve_default_or_handle(s.gpu_hot_handle, "gpu_hot_shutdown"));
    s.gpu_hot_alloc = reinterpret_cast<gpu_hot_alloc_fn>(resolve_default_or_handle(s.gpu_hot_handle, "gpu_hot_alloc"));
    s.gpu_hot_free = reinterpret_cast<gpu_hot_free_fn>(resolve_default_or_handle(s.gpu_hot_handle, "gpu_hot_free"));
    s.gpu_hot_runtime_validate = reinterpret_cast<gpu_hot_runtime_validate_fn>(
        resolve_default_or_handle(s.gpu_hot_handle, "gpu_hot_runtime_validate"));
}

void initialize() {
    if (g_in_hook) return;
    State& s = state();
    Telemetry& t = telemetry();
    std::call_once(s.init_once, [&]() {
        g_in_hook = true;
        t.init_calls.fetch_add(1, std::memory_order_relaxed);
        s.enabled = env_enabled("FERCUDA_INTERCEPT_ENABLE", true);
        s.log = env_enabled("FERCUDA_INTERCEPT_LOG", false);
        s.async_tlsf = env_enabled("FERCUDA_INTERCEPT_ASYNC_TLSF", false);
        resolve_real_cuda_symbols(s);
        resolve_gpu_hot_symbols(s);

        if (!s.enabled) {
            t.init_fail.fetch_add(1, std::memory_order_relaxed);
            s.initialized = true;
            g_in_hook = false;
            return;
        }
        if (!s.gpu_hot_init || !s.gpu_hot_alloc || !s.gpu_hot_free) {
            s.enabled = false;
            t.init_fail.fetch_add(1, std::memory_order_relaxed);
            s.initialized = true;
            g_in_hook = false;
            return;
        }

        int device = 0;
        if (s.real_cuda_get_device && s.real_cuda_get_device(&device) != cudaSuccess) device = 0;
        if (s.gpu_hot_init_with_config && s.gpu_hot_default_config) {
            GPUHotConfig cfg = s.gpu_hot_default_config();
            const uint64_t pool_mb = env_u64("FERCUDA_INTERCEPT_POOL_MB", 128);
            const uint64_t reserve_mb = env_u64("FERCUDA_INTERCEPT_RESERVE_MB", 64);
            const size_t pool_bytes = static_cast<size_t>(pool_mb) << 20;
            const size_t reserve_bytes = static_cast<size_t>(reserve_mb) << 20;
            cfg.pool_fraction = 0.0f;
            cfg.fixed_pool_size = pool_bytes;
            cfg.min_pool_size = pool_bytes;
            cfg.max_pool_size = pool_bytes;
            cfg.reserve_vram = reserve_bytes;
            cfg.quiet_init = true;
            s.gpu_hot_runtime = s.gpu_hot_init_with_config(device, nullptr, &cfg);
        } else {
            s.gpu_hot_runtime = s.gpu_hot_init(device, nullptr);
        }
        if (!s.gpu_hot_runtime) {
            s.enabled = false;
        } else {
            if (s.gpu_hot_runtime_validate && !s.gpu_hot_runtime_validate(s.gpu_hot_runtime)) {
                s.enabled = false;
            }
            // Health probe: ensure allocator path is functional before enabling interception.
            if (s.enabled) {
                void* probe = s.gpu_hot_alloc(s.gpu_hot_runtime, 4096);
                if (!probe) {
                    s.enabled = false;
                } else {
                    s.gpu_hot_free(s.gpu_hot_runtime, probe);
                }
            }
        }
        if (s.enabled) t.init_success.fetch_add(1, std::memory_order_relaxed);
        else t.init_fail.fetch_add(1, std::memory_order_relaxed);
        s.initialized = true;
        g_in_hook = false;
    });
}

void remember_tlsf_ptr(void* p) {
    if (!p) return;
    State& s = state();
    std::lock_guard<std::mutex> lk(s.owner_mu);
    s.owners[p] = Owner::TLSF;
}

bool erase_if_tlsf(void* p) {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.owner_mu);
    auto it = s.owners.find(p);
    if (it == s.owners.end()) return false;
    bool tlsf = (it->second == Owner::TLSF);
    s.owners.erase(it);
    return tlsf;
}

bool can_use_tlsf_for_sync_alloc() {
    State& s = state();
    if (!(s.enabled && s.gpu_hot_runtime && s.gpu_hot_alloc && s.gpu_hot_free)) {
        return false;
    }
    if (s.gpu_hot_runtime_validate && !s.gpu_hot_runtime_validate(s.gpu_hot_runtime)) {
        s.enabled = false;
        return false;
    }
    return true;
}

} // namespace

extern "C" CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.alloc_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_driver.fetch_add(1, std::memory_order_relaxed);
    t.alloc_bytes_requested.fetch_add(static_cast<uint64_t>(bytesize), std::memory_order_relaxed);
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    if (g_in_hook || !can_use_tlsf_for_sync_alloc()) {
        t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
        return s.real_cu_mem_alloc_v2 ? s.real_cu_mem_alloc_v2(dptr, bytesize) : CUDA_ERROR_NOT_INITIALIZED;
    }

    g_in_hook = true;
    void* p = s.gpu_hot_alloc(s.gpu_hot_runtime, bytesize);
    g_in_hook = false;
    if (p) {
        *dptr = reinterpret_cast<CUdeviceptr>(p);
        remember_tlsf_ptr(p);
        t.tlsf_alloc_success.fetch_add(1, std::memory_order_relaxed);
        return CUDA_SUCCESS;
    }
    t.tlsf_alloc_fail.fetch_add(1, std::memory_order_relaxed);
    t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    return s.real_cu_mem_alloc_v2 ? s.real_cu_mem_alloc_v2(dptr, bytesize) : CUDA_ERROR_OUT_OF_MEMORY;
}

extern "C" CUresult cuMemFree_v2(CUdeviceptr dptr) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.free_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.free_calls_driver.fetch_add(1, std::memory_order_relaxed);
    void* p = reinterpret_cast<void*>(dptr);
    if (g_in_hook) return s.real_cu_mem_free_v2 ? s.real_cu_mem_free_v2(dptr) : CUDA_ERROR_NOT_INITIALIZED;

    if (erase_if_tlsf(p) && can_use_tlsf_for_sync_alloc()) {
        g_in_hook = true;
        s.gpu_hot_free(s.gpu_hot_runtime, p);
        g_in_hook = false;
        t.tlsf_free_success.fetch_add(1, std::memory_order_relaxed);
        return CUDA_SUCCESS;
    }
    t.tlsf_free_miss.fetch_add(1, std::memory_order_relaxed);
    t.fallback_free_calls.fetch_add(1, std::memory_order_relaxed);
    return s.real_cu_mem_free_v2 ? s.real_cu_mem_free_v2(dptr) : CUDA_ERROR_INVALID_VALUE;
}

extern "C" CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.alloc_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_driver.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_async.fetch_add(1, std::memory_order_relaxed);
    t.alloc_bytes_requested.fetch_add(static_cast<uint64_t>(bytesize), std::memory_order_relaxed);
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    if (!s.async_tlsf || g_in_hook || !can_use_tlsf_for_sync_alloc()) {
        t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
        return s.real_cu_mem_alloc_async ? s.real_cu_mem_alloc_async(dptr, bytesize, hStream) : CUDA_ERROR_NOT_SUPPORTED;
    }
    g_in_hook = true;
    void* p = s.gpu_hot_alloc(s.gpu_hot_runtime, bytesize);
    g_in_hook = false;
    if (p) {
        *dptr = reinterpret_cast<CUdeviceptr>(p);
        remember_tlsf_ptr(p);
        t.tlsf_alloc_success.fetch_add(1, std::memory_order_relaxed);
        return CUDA_SUCCESS;
    }
    t.tlsf_alloc_fail.fetch_add(1, std::memory_order_relaxed);
    t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    return s.real_cu_mem_alloc_async ? s.real_cu_mem_alloc_async(dptr, bytesize, hStream) : CUDA_ERROR_OUT_OF_MEMORY;
}

extern "C" CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.free_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.free_calls_driver.fetch_add(1, std::memory_order_relaxed);
    t.free_calls_async.fetch_add(1, std::memory_order_relaxed);
    void* p = reinterpret_cast<void*>(dptr);
    if (!s.async_tlsf || g_in_hook) {
        t.fallback_free_calls.fetch_add(1, std::memory_order_relaxed);
        return s.real_cu_mem_free_async ? s.real_cu_mem_free_async(dptr, hStream) : CUDA_ERROR_NOT_SUPPORTED;
    }
    if (erase_if_tlsf(p) && can_use_tlsf_for_sync_alloc()) {
        g_in_hook = true;
        s.gpu_hot_free(s.gpu_hot_runtime, p);
        g_in_hook = false;
        t.tlsf_free_success.fetch_add(1, std::memory_order_relaxed);
        return CUDA_SUCCESS;
    }
    t.tlsf_free_miss.fetch_add(1, std::memory_order_relaxed);
    t.fallback_free_calls.fetch_add(1, std::memory_order_relaxed);
    return s.real_cu_mem_free_async ? s.real_cu_mem_free_async(dptr, hStream) : CUDA_ERROR_INVALID_VALUE;
}

extern "C" cudaError_t cudaMalloc(void** devPtr, size_t size) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.alloc_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_runtime.fetch_add(1, std::memory_order_relaxed);
    t.alloc_bytes_requested.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    if (!devPtr) return cudaErrorInvalidValue;
    if (g_in_hook || !can_use_tlsf_for_sync_alloc()) {
        t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
        return s.real_cuda_malloc ? s.real_cuda_malloc(devPtr, size) : cudaErrorInitializationError;
    }

    g_in_hook = true;
    void* p = s.gpu_hot_alloc(s.gpu_hot_runtime, size);
    g_in_hook = false;
    if (p) {
        *devPtr = p;
        remember_tlsf_ptr(p);
        t.tlsf_alloc_success.fetch_add(1, std::memory_order_relaxed);
        return cudaSuccess;
    }
    t.tlsf_alloc_fail.fetch_add(1, std::memory_order_relaxed);
    t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    return s.real_cuda_malloc ? s.real_cuda_malloc(devPtr, size) : cudaErrorMemoryAllocation;
}

extern "C" cudaError_t cudaFree(void* devPtr) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.free_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.free_calls_runtime.fetch_add(1, std::memory_order_relaxed);
    if (g_in_hook) return s.real_cuda_free ? s.real_cuda_free(devPtr) : cudaErrorInitializationError;

    if (erase_if_tlsf(devPtr) && can_use_tlsf_for_sync_alloc()) {
        g_in_hook = true;
        s.gpu_hot_free(s.gpu_hot_runtime, devPtr);
        g_in_hook = false;
        t.tlsf_free_success.fetch_add(1, std::memory_order_relaxed);
        return cudaSuccess;
    }
    t.tlsf_free_miss.fetch_add(1, std::memory_order_relaxed);
    t.fallback_free_calls.fetch_add(1, std::memory_order_relaxed);
    return s.real_cuda_free ? s.real_cuda_free(devPtr) : cudaErrorInvalidDevicePointer;
}

extern "C" cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.alloc_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_runtime.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_async.fetch_add(1, std::memory_order_relaxed);
    t.alloc_bytes_requested.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    if (!devPtr) return cudaErrorInvalidValue;
    if (!s.async_tlsf || g_in_hook || !can_use_tlsf_for_sync_alloc()) {
        t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
        return s.real_cuda_malloc_async ? s.real_cuda_malloc_async(devPtr, size, hStream) : cudaErrorNotSupported;
    }
    g_in_hook = true;
    void* p = s.gpu_hot_alloc(s.gpu_hot_runtime, size);
    g_in_hook = false;
    if (p) {
        *devPtr = p;
        remember_tlsf_ptr(p);
        t.tlsf_alloc_success.fetch_add(1, std::memory_order_relaxed);
        return cudaSuccess;
    }
    t.tlsf_alloc_fail.fetch_add(1, std::memory_order_relaxed);
    t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    return s.real_cuda_malloc_async ? s.real_cuda_malloc_async(devPtr, size, hStream) : cudaErrorMemoryAllocation;
}

extern "C" cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t hStream) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.free_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.free_calls_runtime.fetch_add(1, std::memory_order_relaxed);
    t.free_calls_async.fetch_add(1, std::memory_order_relaxed);
    if (!s.async_tlsf || g_in_hook) {
        t.fallback_free_calls.fetch_add(1, std::memory_order_relaxed);
        return s.real_cuda_free_async ? s.real_cuda_free_async(devPtr, hStream) : cudaErrorNotSupported;
    }
    if (erase_if_tlsf(devPtr) && can_use_tlsf_for_sync_alloc()) {
        g_in_hook = true;
        s.gpu_hot_free(s.gpu_hot_runtime, devPtr);
        g_in_hook = false;
        t.tlsf_free_success.fetch_add(1, std::memory_order_relaxed);
        return cudaSuccess;
    }
    t.tlsf_free_miss.fetch_add(1, std::memory_order_relaxed);
    t.fallback_free_calls.fetch_add(1, std::memory_order_relaxed);
    return s.real_cuda_free_async ? s.real_cuda_free_async(devPtr, hStream) : cudaErrorInvalidDevicePointer;
}

extern "C" int fercuda_intercept_telemetry_get(fercuda_intercept_stats_t* out_stats) {
    if (!out_stats) return 1;
    Telemetry& t = telemetry();
    out_stats->init_calls = t.init_calls.load(std::memory_order_relaxed);
    out_stats->init_success = t.init_success.load(std::memory_order_relaxed);
    out_stats->init_fail = t.init_fail.load(std::memory_order_relaxed);
    out_stats->alloc_calls_total = t.alloc_calls_total.load(std::memory_order_relaxed);
    out_stats->free_calls_total = t.free_calls_total.load(std::memory_order_relaxed);
    out_stats->alloc_calls_driver = t.alloc_calls_driver.load(std::memory_order_relaxed);
    out_stats->free_calls_driver = t.free_calls_driver.load(std::memory_order_relaxed);
    out_stats->alloc_calls_runtime = t.alloc_calls_runtime.load(std::memory_order_relaxed);
    out_stats->free_calls_runtime = t.free_calls_runtime.load(std::memory_order_relaxed);
    out_stats->alloc_calls_async = t.alloc_calls_async.load(std::memory_order_relaxed);
    out_stats->free_calls_async = t.free_calls_async.load(std::memory_order_relaxed);
    out_stats->alloc_bytes_requested = t.alloc_bytes_requested.load(std::memory_order_relaxed);
    out_stats->tlsf_alloc_success = t.tlsf_alloc_success.load(std::memory_order_relaxed);
    out_stats->tlsf_alloc_fail = t.tlsf_alloc_fail.load(std::memory_order_relaxed);
    out_stats->tlsf_free_success = t.tlsf_free_success.load(std::memory_order_relaxed);
    out_stats->tlsf_free_miss = t.tlsf_free_miss.load(std::memory_order_relaxed);
    out_stats->fallback_alloc_calls = t.fallback_alloc_calls.load(std::memory_order_relaxed);
    out_stats->fallback_free_calls = t.fallback_free_calls.load(std::memory_order_relaxed);
    return 0;
}

extern "C" void fercuda_intercept_telemetry_reset(void) {
    Telemetry& t = telemetry();
    t.init_calls.store(0, std::memory_order_relaxed);
    t.init_success.store(0, std::memory_order_relaxed);
    t.init_fail.store(0, std::memory_order_relaxed);
    t.alloc_calls_total.store(0, std::memory_order_relaxed);
    t.free_calls_total.store(0, std::memory_order_relaxed);
    t.alloc_calls_driver.store(0, std::memory_order_relaxed);
    t.free_calls_driver.store(0, std::memory_order_relaxed);
    t.alloc_calls_runtime.store(0, std::memory_order_relaxed);
    t.free_calls_runtime.store(0, std::memory_order_relaxed);
    t.alloc_calls_async.store(0, std::memory_order_relaxed);
    t.free_calls_async.store(0, std::memory_order_relaxed);
    t.alloc_bytes_requested.store(0, std::memory_order_relaxed);
    t.tlsf_alloc_success.store(0, std::memory_order_relaxed);
    t.tlsf_alloc_fail.store(0, std::memory_order_relaxed);
    t.tlsf_free_success.store(0, std::memory_order_relaxed);
    t.tlsf_free_miss.store(0, std::memory_order_relaxed);
    t.fallback_alloc_calls.store(0, std::memory_order_relaxed);
    t.fallback_free_calls.store(0, std::memory_order_relaxed);
}

extern "C" int fercuda_intercept_telemetry_enabled(void) {
    initialize();
    return state().enabled ? 1 : 0;
}

extern "C" __attribute__((destructor)) void fercuda_intercept_shutdown() {
    State& s = state();
    if (!s.initialized) return;
    if (s.gpu_hot_shutdown && s.gpu_hot_runtime) {
        s.gpu_hot_shutdown(s.gpu_hot_runtime);
        s.gpu_hot_runtime = nullptr;
    }
    if (s.gpu_hot_handle) {
        dlclose(s.gpu_hot_handle);
        s.gpu_hot_handle = nullptr;
    }
}
