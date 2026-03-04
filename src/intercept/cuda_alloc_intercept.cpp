#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

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
using cu_mem_alloc_fn = CUresult (*)(CUdeviceptr* dptr, size_t bytesize);
using cu_mem_free_fn = CUresult (*)(CUdeviceptr dptr);
using cu_mem_alloc_managed_fn = CUresult (*)(CUdeviceptr* dptr, size_t bytesize, unsigned int flags);
using cu_mem_alloc_async_fn = CUresult (*)(CUdeviceptr* dptr, size_t bytesize, CUstream hStream);
using cu_mem_free_async_fn = CUresult (*)(CUdeviceptr dptr, CUstream hStream);

using cuda_malloc_fn = cudaError_t (*)(void** devPtr, size_t size);
using cuda_free_fn = cudaError_t (*)(void* devPtr);
using cuda_malloc_managed_fn = cudaError_t (*)(void** devPtr, size_t size, unsigned int flags);
using cuda_host_alloc_fn = cudaError_t (*)(void** pHost, size_t size, unsigned int flags);
using cuda_free_host_fn = cudaError_t (*)(void* pHost);
using cuda_malloc_pitch_fn = cudaError_t (*)(void** devPtr, size_t* pitch, size_t width, size_t height);
using cuda_malloc_async_fn = cudaError_t (*)(void** devPtr, size_t size, cudaStream_t hStream);
using cuda_free_async_fn = cudaError_t (*)(void* devPtr, cudaStream_t hStream);
using cuda_get_device_fn = cudaError_t (*)(int* device);

enum class Owner : uint8_t {
    TLSF = 1,
    SizeClass = 2,
};

enum class RegimeKind : uint8_t {
    TLSF = 1,
    SizeClass = 2,
};

enum class ApiKind : uint8_t {
    Unknown = 0,
    DriverSync = 1,
    DriverAsync = 2,
    RuntimeSync = 3,
    RuntimeAsync = 4,
};

struct AllocationMeta {
    Owner owner = Owner::TLSF;
    ApiKind api = ApiKind::Unknown;
    uint64_t bytes = 0;
};

struct State {
    static constexpr size_t kSizeClassMin = 256;
    static constexpr size_t kSizeClassMax = 1u << 24; // 16 MiB
    static constexpr size_t kSizeClassCount = 17;     // 256B..16MiB

    std::once_flag init_once;
    std::mutex owner_mu;
    std::unordered_map<void*, AllocationMeta> owners;
    std::mutex sizeclass_mu;
    std::unordered_map<void*, size_t> sizeclass_alloc_sizes;
    std::array<std::vector<void*>, kSizeClassCount> sizeclass_free_lists;

    bool enabled = true;
    bool log = false;
    bool async_tlsf = false;
    RegimeKind regime = RegimeKind::TLSF;
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
    cu_mem_alloc_fn real_cu_mem_alloc = nullptr;
    cu_mem_free_fn real_cu_mem_free = nullptr;
    cu_mem_alloc_managed_fn real_cu_mem_alloc_managed = nullptr;
    cu_mem_alloc_async_fn real_cu_mem_alloc_async = nullptr;
    cu_mem_free_async_fn real_cu_mem_free_async = nullptr;

    cuda_malloc_fn real_cuda_malloc = nullptr;
    cuda_free_fn real_cuda_free = nullptr;
    cuda_malloc_managed_fn real_cuda_malloc_managed = nullptr;
    cuda_host_alloc_fn real_cuda_host_alloc = nullptr;
    cuda_free_host_fn real_cuda_free_host = nullptr;
    cuda_malloc_pitch_fn real_cuda_malloc_pitch = nullptr;
    cuda_malloc_async_fn real_cuda_malloc_async = nullptr;
    cuda_free_async_fn real_cuda_free_async = nullptr;
    cuda_get_device_fn real_cuda_get_device = nullptr;

    void* cuda_driver_handle = nullptr;
    void* cuda_runtime_handle = nullptr;
    bool strict_mode = false;
    std::atomic<bool> shutdown_done{false};
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
    std::atomic<uint64_t> alloc_calls_managed{0};
    std::atomic<uint64_t> alloc_calls_host{0};
    std::atomic<uint64_t> free_calls_host{0};
    std::atomic<uint64_t> alloc_calls_pitch{0};

    std::atomic<uint64_t> alloc_bytes_requested{0};

    std::atomic<uint64_t> tlsf_alloc_success{0};
    std::atomic<uint64_t> tlsf_alloc_fail{0};
    std::atomic<uint64_t> tlsf_free_success{0};
    std::atomic<uint64_t> tlsf_free_miss{0};
    std::atomic<uint64_t> sizeclass_alloc_success{0};
    std::atomic<uint64_t> sizeclass_alloc_fail{0};
    std::atomic<uint64_t> sizeclass_free_success{0};
    std::atomic<uint64_t> sizeclass_free_miss{0};

    std::atomic<uint64_t> fallback_alloc_calls{0};
    std::atomic<uint64_t> fallback_free_calls{0};
    std::atomic<uint64_t> fallback_async_to_sync_calls{0};
    std::atomic<uint64_t> strict_mode_reject_calls{0};
    std::atomic<uint64_t> late_symbol_resolve_attempts{0};
};

Telemetry& telemetry() {
    static Telemetry t{};
    return t;
}

bool can_use_sizeclass_for_sync_alloc();

thread_local bool g_in_hook = false;

bool env_enabled(const char* key, bool def) {
    const char* v = std::getenv(key);
    if (!v) return def;
    if (v[0] == '0') return false;
    if (v[0] == 'n' || v[0] == 'N') return false;
    if (v[0] == 'f' || v[0] == 'F') return false;
    return true;
}

bool env_is_strict_mode() {
    const char* v = std::getenv("FERCUDA_INTERCEPT_MODE");
    if (!v || v[0] == '\0') return false;
    return std::strcmp(v, "strict") == 0 || std::strcmp(v, "STRICT") == 0;
}

uint64_t env_u64(const char* key, uint64_t def) {
    const char* v = std::getenv(key);
    if (!v || v[0] == '\0') return def;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(v, &end, 10);
    if (!end || *end != '\0') return def;
    return static_cast<uint64_t>(parsed);
}

RegimeKind env_regime_kind() {
    const char* v = std::getenv("FERCUDA_INTERCEPT_REGIME");
    if (!v || v[0] == '\0') return RegimeKind::TLSF;
    if (std::strcmp(v, "sizeclass") == 0 || std::strcmp(v, "SIZECLASS") == 0 ||
        std::strcmp(v, "regime2") == 0 || std::strcmp(v, "REGIME2") == 0 ||
        std::strcmp(v, "segv2") == 0 || std::strcmp(v, "SEGV2") == 0) {
        return RegimeKind::SizeClass;
    }
    return RegimeKind::TLSF;
}

void log_line(const State& s, const std::string& msg) {
    if (!s.log) return;
    std::fprintf(stderr, "[fercuda_intercept] %s\n", msg.c_str());
}

void append_unique(std::vector<void*>& handles, void* h) {
    if (!h) return;
    for (void* e : handles) {
        if (e == h) return;
    }
    handles.push_back(h);
}

void open_cuda_lib_handles(State& s) {
    if (!s.cuda_driver_handle) {
        s.cuda_driver_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        if (!s.cuda_driver_handle) s.cuda_driver_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!s.cuda_driver_handle) s.cuda_driver_handle = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        if (!s.cuda_driver_handle) s.cuda_driver_handle = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!s.cuda_runtime_handle) {
        s.cuda_runtime_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        if (!s.cuda_runtime_handle) s.cuda_runtime_handle = dlopen("libcudart.so.12", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        if (!s.cuda_runtime_handle) s.cuda_runtime_handle = dlopen("libcudart.so.11.0", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        if (!s.cuda_runtime_handle) s.cuda_runtime_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    }
}

void* resolve_symbol_all(State& s, const char* name, void* prefer_handle = nullptr) {
    static const char* kSymVersions[] = {
        "libcuda.so.1",
        "libcuda.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "libcudart.so",
    };
    if (void* p = dlsym(RTLD_NEXT, name)) return p;
    if (void* p = dlsym(RTLD_DEFAULT, name)) return p;
#if defined(__GLIBC__)
    for (const char* ver : kSymVersions) {
        if (void* p = dlvsym(RTLD_DEFAULT, name, ver)) return p;
    }
#endif
    std::vector<void*> handles;
    append_unique(handles, prefer_handle);
    append_unique(handles, s.cuda_driver_handle);
    append_unique(handles, s.cuda_runtime_handle);
    append_unique(handles, s.gpu_hot_handle);
    for (void* h : handles) {
        if (void* p = dlsym(h, name)) return p;
#if defined(__GLIBC__)
        for (const char* ver : kSymVersions) {
            if (void* p = dlvsym(h, name, ver)) return p;
        }
#endif
    }
    return nullptr;
}

void* resolve_default_or_handle(void* handle, const char* name) {
    void* p = dlsym(RTLD_DEFAULT, name);
    if (p) return p;
    if (handle) return dlsym(handle, name);
    return nullptr;
}

void maybe_open_path_list(State& s, const char* path_list) {
    if (!path_list || path_list[0] == '\0' || s.gpu_hot_handle) return;
    std::string all(path_list);
    size_t pos = 0;
    while (pos <= all.size()) {
        size_t end = all.find(':', pos);
        std::string one = (end == std::string::npos) ? all.substr(pos) : all.substr(pos, end - pos);
        if (!one.empty()) {
            s.gpu_hot_handle = dlopen(one.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (s.gpu_hot_handle) {
                log_line(s, std::string("loaded GPU hot runtime from FERCUDA_GPU_HOT_SO_PATHS: ") + one);
                return;
            }
        }
        if (end == std::string::npos) break;
        pos = end + 1;
    }
}

void resolve_real_cuda_symbols(State& s) {
    open_cuda_lib_handles(s);
    if (!s.real_cu_mem_alloc_v2) s.real_cu_mem_alloc_v2 = reinterpret_cast<cu_mem_alloc_v2_fn>(resolve_symbol_all(s, "cuMemAlloc_v2", s.cuda_driver_handle));
    if (!s.real_cu_mem_free_v2) s.real_cu_mem_free_v2 = reinterpret_cast<cu_mem_free_v2_fn>(resolve_symbol_all(s, "cuMemFree_v2", s.cuda_driver_handle));
    if (!s.real_cu_mem_alloc) s.real_cu_mem_alloc = reinterpret_cast<cu_mem_alloc_fn>(resolve_symbol_all(s, "cuMemAlloc", s.cuda_driver_handle));
    if (!s.real_cu_mem_free) s.real_cu_mem_free = reinterpret_cast<cu_mem_free_fn>(resolve_symbol_all(s, "cuMemFree", s.cuda_driver_handle));
    if (!s.real_cu_mem_alloc_managed) s.real_cu_mem_alloc_managed = reinterpret_cast<cu_mem_alloc_managed_fn>(resolve_symbol_all(s, "cuMemAllocManaged", s.cuda_driver_handle));
    if (!s.real_cu_mem_alloc_async) s.real_cu_mem_alloc_async = reinterpret_cast<cu_mem_alloc_async_fn>(resolve_symbol_all(s, "cuMemAllocAsync", s.cuda_driver_handle));
    if (!s.real_cu_mem_free_async) s.real_cu_mem_free_async = reinterpret_cast<cu_mem_free_async_fn>(resolve_symbol_all(s, "cuMemFreeAsync", s.cuda_driver_handle));
    if (!s.real_cuda_malloc) s.real_cuda_malloc = reinterpret_cast<cuda_malloc_fn>(resolve_symbol_all(s, "cudaMalloc", s.cuda_runtime_handle));
    if (!s.real_cuda_free) s.real_cuda_free = reinterpret_cast<cuda_free_fn>(resolve_symbol_all(s, "cudaFree", s.cuda_runtime_handle));
    if (!s.real_cuda_malloc_managed) s.real_cuda_malloc_managed = reinterpret_cast<cuda_malloc_managed_fn>(resolve_symbol_all(s, "cudaMallocManaged", s.cuda_runtime_handle));
    if (!s.real_cuda_host_alloc) s.real_cuda_host_alloc = reinterpret_cast<cuda_host_alloc_fn>(resolve_symbol_all(s, "cudaHostAlloc", s.cuda_runtime_handle));
    if (!s.real_cuda_free_host) s.real_cuda_free_host = reinterpret_cast<cuda_free_host_fn>(resolve_symbol_all(s, "cudaFreeHost", s.cuda_runtime_handle));
    if (!s.real_cuda_malloc_pitch) s.real_cuda_malloc_pitch = reinterpret_cast<cuda_malloc_pitch_fn>(resolve_symbol_all(s, "cudaMallocPitch", s.cuda_runtime_handle));
    if (!s.real_cuda_malloc_async) s.real_cuda_malloc_async = reinterpret_cast<cuda_malloc_async_fn>(resolve_symbol_all(s, "cudaMallocAsync", s.cuda_runtime_handle));
    if (!s.real_cuda_free_async) s.real_cuda_free_async = reinterpret_cast<cuda_free_async_fn>(resolve_symbol_all(s, "cudaFreeAsync", s.cuda_runtime_handle));
    if (!s.real_cuda_get_device) s.real_cuda_get_device = reinterpret_cast<cuda_get_device_fn>(resolve_symbol_all(s, "cudaGetDevice", s.cuda_runtime_handle));
}

void resolve_gpu_hot_symbols(State& s) {
    const char* so_path = std::getenv("FERCUDA_GPU_HOT_SO");
    const char* so_paths = std::getenv("FERCUDA_GPU_HOT_SO_PATHS");
    static std::string fallback_so;
    if (so_path && so_path[0] != '\0') {
        s.gpu_hot_handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
        if (s.gpu_hot_handle) log_line(s, std::string("loaded GPU hot runtime from FERCUDA_GPU_HOT_SO: ") + so_path);
    }
    maybe_open_path_list(s, so_paths);
    if (!s.gpu_hot_handle) {
        const char* home = std::getenv("HOME");
        if (home && home[0] != '\0') {
            fallback_so = std::string(home) + "/persistant_gpu_os/build/libptx_os_shared.so";
            s.gpu_hot_handle = dlopen(fallback_so.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (s.gpu_hot_handle) log_line(s, std::string("loaded GPU hot runtime fallback: ") + fallback_so);
        }
    }
    if (!s.gpu_hot_handle) {
        s.gpu_hot_handle = dlopen("libptx_os_shared.so", RTLD_NOW | RTLD_LOCAL);
        if (s.gpu_hot_handle) log_line(s, "loaded GPU hot runtime by soname: libptx_os_shared.so");
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

    if (!s.gpu_hot_handle) log_line(s, "GPU hot runtime shared library not found; intercept will use CUDA fallback path");
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
        s.async_tlsf = env_enabled("FERCUDA_INTERCEPT_ASYNC_REGIME",
            env_enabled("FERCUDA_INTERCEPT_ASYNC_TLSF", false));
        s.regime = env_regime_kind();
        s.strict_mode = env_is_strict_mode();
        resolve_real_cuda_symbols(s);
        resolve_gpu_hot_symbols(s);

        if (!s.enabled) {
            t.init_fail.fetch_add(1, std::memory_order_relaxed);
            s.initialized = true;
            g_in_hook = false;
            return;
        }
        if (s.regime == RegimeKind::TLSF) {
            if (!s.gpu_hot_init || !s.gpu_hot_alloc || !s.gpu_hot_free) {
                if (!s.strict_mode) s.enabled = false;
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
                if (!s.strict_mode) s.enabled = false;
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
        } else if (!can_use_sizeclass_for_sync_alloc()) {
            if (!s.strict_mode) s.enabled = false;
        }
        if (s.enabled) t.init_success.fetch_add(1, std::memory_order_relaxed);
        else t.init_fail.fetch_add(1, std::memory_order_relaxed);
        s.initialized = true;
        g_in_hook = false;
    });
}

void try_recover_symbols() {
    State& s = state();
    Telemetry& t = telemetry();
    t.late_symbol_resolve_attempts.fetch_add(1, std::memory_order_relaxed);
    resolve_real_cuda_symbols(s);
}

void remember_owned_ptr(void* p, Owner owner, uint64_t bytes, ApiKind api) {
    if (!p) return;
    State& s = state();
    std::lock_guard<std::mutex> lk(s.owner_mu);
    s.owners[p] = AllocationMeta{owner, api, bytes};
}

bool erase_if_owned(void* p, Owner owner, AllocationMeta* out_meta = nullptr) {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.owner_mu);
    auto it = s.owners.find(p);
    if (it == s.owners.end()) return false;
    if (out_meta) *out_meta = it->second;
    bool owned = (it->second.owner == owner);
    s.owners.erase(it);
    return owned;
}

uint64_t tracked_alloc_count() {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.owner_mu);
    return static_cast<uint64_t>(s.owners.size());
}

size_t sizeclass_class_index_from_size(size_t bytes) {
    size_t clamped = bytes < State::kSizeClassMin ? State::kSizeClassMin : bytes;
    size_t bucket = State::kSizeClassMin;
    size_t idx = 0;
    while (bucket < clamped && idx + 1 < State::kSizeClassCount) {
        bucket <<= 1;
        ++idx;
    }
    return idx;
}

size_t sizeclass_class_size(size_t bytes) {
    size_t idx = sizeclass_class_index_from_size(bytes);
    return State::kSizeClassMin << idx;
}

bool sizeclass_is_large_alloc(size_t bytes) {
    return bytes > State::kSizeClassMax;
}

void* sizeclass_backend_alloc(State& s, size_t bytes) {
    if (!s.real_cuda_malloc && !s.real_cu_mem_alloc_v2 && !s.real_cu_mem_alloc && !s.real_cuda_malloc_async) {
        try_recover_symbols();
    }
    if (s.real_cuda_malloc) {
        void* ptr = nullptr;
        if (s.real_cuda_malloc(&ptr, bytes) == cudaSuccess) return ptr;
    }
    if (s.real_cu_mem_alloc_v2) {
        CUdeviceptr dptr = 0;
        if (s.real_cu_mem_alloc_v2(&dptr, bytes) == CUDA_SUCCESS) return reinterpret_cast<void*>(dptr);
    }
    if (s.real_cu_mem_alloc) {
        CUdeviceptr dptr = 0;
        if (s.real_cu_mem_alloc(&dptr, bytes) == CUDA_SUCCESS) return reinterpret_cast<void*>(dptr);
    }
    return nullptr;
}

void sizeclass_backend_free(State& s, void* ptr) {
    if (!s.real_cuda_free && !s.real_cu_mem_free_v2 && !s.real_cu_mem_free && !s.real_cuda_free_async) {
        try_recover_symbols();
    }
    if (s.real_cuda_free && s.real_cuda_free(ptr) == cudaSuccess) return;
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);
    if (s.real_cu_mem_free_v2 && s.real_cu_mem_free_v2(dptr) == CUDA_SUCCESS) return;
    if (s.real_cu_mem_free) (void)s.real_cu_mem_free(dptr);
}

void* sizeclass_alloc(State& s, size_t req_bytes) {
    const size_t rounded = sizeclass_is_large_alloc(req_bytes) ? req_bytes : sizeclass_class_size(req_bytes);
    if (!sizeclass_is_large_alloc(req_bytes)) {
        const size_t idx = sizeclass_class_index_from_size(req_bytes);
        std::lock_guard<std::mutex> lk(s.sizeclass_mu);
        auto& free_list = s.sizeclass_free_lists[idx];
        if (!free_list.empty()) {
            void* p = free_list.back();
            free_list.pop_back();
            s.sizeclass_alloc_sizes[p] = rounded;
            return p;
        }
    }

    void* p = sizeclass_backend_alloc(s, rounded);
    if (!p) return nullptr;
    std::lock_guard<std::mutex> lk(s.sizeclass_mu);
    s.sizeclass_alloc_sizes[p] = rounded;
    return p;
}

bool sizeclass_free(State& s, void* ptr) {
    size_t alloc_size = 0;
    {
        std::lock_guard<std::mutex> lk(s.sizeclass_mu);
        auto it = s.sizeclass_alloc_sizes.find(ptr);
        if (it == s.sizeclass_alloc_sizes.end()) return false;
        alloc_size = it->second;
        s.sizeclass_alloc_sizes.erase(it);
    }

    if (sizeclass_is_large_alloc(alloc_size)) {
        sizeclass_backend_free(s, ptr);
        return true;
    }

    const size_t idx = sizeclass_class_index_from_size(alloc_size);
    std::lock_guard<std::mutex> lk(s.sizeclass_mu);
    s.sizeclass_free_lists[idx].push_back(ptr);
    return true;
}

void sizeclass_shutdown(State& s) {
    std::lock_guard<std::mutex> lk(s.sizeclass_mu);
    s.sizeclass_alloc_sizes.clear();
    for (auto& list : s.sizeclass_free_lists) list.clear();
}

bool can_use_tlsf_for_sync_alloc() {
    State& s = state();
    if (!(s.enabled && s.gpu_hot_runtime && s.gpu_hot_alloc && s.gpu_hot_free)) {
        return false;
    }
    if (s.gpu_hot_runtime_validate && !s.gpu_hot_runtime_validate(s.gpu_hot_runtime)) {
        if (!s.strict_mode) s.enabled = false;
        return false;
    }
    return true;
}

bool can_use_sizeclass_for_sync_alloc() {
    State& s = state();
    if (!s.enabled) return false;
    if (!s.real_cuda_malloc && !s.real_cu_mem_alloc_v2 && !s.real_cu_mem_alloc) {
        try_recover_symbols();
    }
    return s.real_cuda_malloc || s.real_cu_mem_alloc_v2 || s.real_cu_mem_alloc;
}

bool can_use_active_regime_for_sync_alloc() {
    State& s = state();
    if (s.regime == RegimeKind::SizeClass) return can_use_sizeclass_for_sync_alloc();
    return can_use_tlsf_for_sync_alloc();
}

Owner active_owner(const State& s) {
    return s.regime == RegimeKind::SizeClass ? Owner::SizeClass : Owner::TLSF;
}

bool active_regime_alloc(State& s, void** out_ptr, size_t bytes) {
    if (!out_ptr) return false;
    if (s.regime == RegimeKind::SizeClass) {
        *out_ptr = sizeclass_alloc(s, bytes);
        return *out_ptr != nullptr;
    }
    *out_ptr = s.gpu_hot_alloc(s.gpu_hot_runtime, bytes);
    return *out_ptr != nullptr;
}

bool active_regime_free(State& s, void* ptr) {
    if (s.regime == RegimeKind::SizeClass) return sizeclass_free(s, ptr);
    s.gpu_hot_free(s.gpu_hot_runtime, ptr);
    return true;
}

void telemetry_alloc_success() {
    Telemetry& t = telemetry();
    State& s = state();
    if (s.regime == RegimeKind::SizeClass) t.sizeclass_alloc_success.fetch_add(1, std::memory_order_relaxed);
    else t.tlsf_alloc_success.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_alloc_fail() {
    Telemetry& t = telemetry();
    State& s = state();
    if (s.regime == RegimeKind::SizeClass) t.sizeclass_alloc_fail.fetch_add(1, std::memory_order_relaxed);
    else t.tlsf_alloc_fail.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_free_success() {
    Telemetry& t = telemetry();
    State& s = state();
    if (s.regime == RegimeKind::SizeClass) t.sizeclass_free_success.fetch_add(1, std::memory_order_relaxed);
    else t.tlsf_free_success.fetch_add(1, std::memory_order_relaxed);
}

void telemetry_free_miss() {
    Telemetry& t = telemetry();
    State& s = state();
    if (s.regime == RegimeKind::SizeClass) t.sizeclass_free_miss.fetch_add(1, std::memory_order_relaxed);
    else t.tlsf_free_miss.fetch_add(1, std::memory_order_relaxed);
}

bool should_strict_reject() {
    State& s = state();
    Telemetry& t = telemetry();
    if (!s.strict_mode) return false;
    if (can_use_active_regime_for_sync_alloc()) return false;
    t.strict_mode_reject_calls.fetch_add(1, std::memory_order_relaxed);
    return true;
}

CUresult driver_alloc_async_fallback(State& s, CUdeviceptr* dptr, size_t bytesize, CUstream hStream) {
    if (!s.real_cu_mem_alloc_async && !s.real_cu_mem_alloc_v2 && !s.real_cu_mem_alloc) try_recover_symbols();
    if (s.real_cu_mem_alloc_async) return s.real_cu_mem_alloc_async(dptr, bytesize, hStream);
    telemetry().fallback_async_to_sync_calls.fetch_add(1, std::memory_order_relaxed);
    if (s.real_cu_mem_alloc_v2) return s.real_cu_mem_alloc_v2(dptr, bytesize);
    if (s.real_cu_mem_alloc) return s.real_cu_mem_alloc(dptr, bytesize);
    return CUDA_ERROR_NOT_INITIALIZED;
}

CUresult driver_free_async_fallback(State& s, CUdeviceptr dptr, CUstream hStream) {
    if (!s.real_cu_mem_free_async && !s.real_cu_mem_free_v2 && !s.real_cu_mem_free) try_recover_symbols();
    if (s.real_cu_mem_free_async) return s.real_cu_mem_free_async(dptr, hStream);
    telemetry().fallback_async_to_sync_calls.fetch_add(1, std::memory_order_relaxed);
    if (s.real_cu_mem_free_v2) return s.real_cu_mem_free_v2(dptr);
    if (s.real_cu_mem_free) return s.real_cu_mem_free(dptr);
    return CUDA_ERROR_NOT_INITIALIZED;
}

cudaError_t runtime_alloc_async_fallback(State& s, void** devPtr, size_t size, cudaStream_t hStream) {
    if (!s.real_cuda_malloc_async && !s.real_cuda_malloc) try_recover_symbols();
    if (s.real_cuda_malloc_async) return s.real_cuda_malloc_async(devPtr, size, hStream);
    telemetry().fallback_async_to_sync_calls.fetch_add(1, std::memory_order_relaxed);
    if (s.real_cuda_malloc) return s.real_cuda_malloc(devPtr, size);
    return cudaErrorInitializationError;
}

cudaError_t runtime_free_async_fallback(State& s, void* devPtr, cudaStream_t hStream) {
    if (!s.real_cuda_free_async && !s.real_cuda_free) try_recover_symbols();
    if (s.real_cuda_free_async) return s.real_cuda_free_async(devPtr, hStream);
    telemetry().fallback_async_to_sync_calls.fetch_add(1, std::memory_order_relaxed);
    if (s.real_cuda_free) return s.real_cuda_free(devPtr);
    return cudaErrorInitializationError;
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
    if (should_strict_reject()) return CUDA_ERROR_NOT_SUPPORTED;
    if (g_in_hook || !can_use_active_regime_for_sync_alloc()) {
        t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
        if (!s.real_cu_mem_alloc_v2 && !s.real_cu_mem_alloc) try_recover_symbols();
        return s.real_cu_mem_alloc_v2 ? s.real_cu_mem_alloc_v2(dptr, bytesize) : CUDA_ERROR_NOT_INITIALIZED;
    }

    g_in_hook = true;
    void* p = nullptr;
    const bool ok = active_regime_alloc(s, &p, bytesize);
    g_in_hook = false;
    if (ok) {
        *dptr = reinterpret_cast<CUdeviceptr>(p);
        remember_owned_ptr(p, active_owner(s), static_cast<uint64_t>(bytesize), ApiKind::DriverSync);
        telemetry_alloc_success();
        return CUDA_SUCCESS;
    }
    telemetry_alloc_fail();
    t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    if (!s.real_cu_mem_alloc_v2 && !s.real_cu_mem_alloc) try_recover_symbols();
    if (s.real_cu_mem_alloc_v2) return s.real_cu_mem_alloc_v2(dptr, bytesize);
    if (s.real_cu_mem_alloc) return s.real_cu_mem_alloc(dptr, bytesize);
    return CUDA_ERROR_OUT_OF_MEMORY;
}

extern "C" CUresult cuMemFree_v2(CUdeviceptr dptr) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.free_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.free_calls_driver.fetch_add(1, std::memory_order_relaxed);
    void* p = reinterpret_cast<void*>(dptr);
    if (g_in_hook) return s.real_cu_mem_free_v2 ? s.real_cu_mem_free_v2(dptr) : CUDA_ERROR_NOT_INITIALIZED;

    if (should_strict_reject()) return CUDA_ERROR_NOT_SUPPORTED;
    if (erase_if_owned(p, active_owner(s)) && can_use_active_regime_for_sync_alloc()) {
        g_in_hook = true;
        (void)active_regime_free(s, p);
        g_in_hook = false;
        telemetry_free_success();
        return CUDA_SUCCESS;
    }
    telemetry_free_miss();
    t.fallback_free_calls.fetch_add(1, std::memory_order_relaxed);
    if (!s.real_cu_mem_free_v2 && !s.real_cu_mem_free) try_recover_symbols();
    if (s.real_cu_mem_free_v2) return s.real_cu_mem_free_v2(dptr);
    if (s.real_cu_mem_free) return s.real_cu_mem_free(dptr);
    return CUDA_ERROR_INVALID_VALUE;
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
    if (should_strict_reject()) return CUDA_ERROR_NOT_SUPPORTED;
    if (!s.async_tlsf || g_in_hook || !can_use_active_regime_for_sync_alloc()) {
        t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
        return driver_alloc_async_fallback(s, dptr, bytesize, hStream);
    }
    g_in_hook = true;
    void* p = nullptr;
    const bool ok = active_regime_alloc(s, &p, bytesize);
    g_in_hook = false;
    if (ok) {
        *dptr = reinterpret_cast<CUdeviceptr>(p);
        remember_owned_ptr(p, active_owner(s), static_cast<uint64_t>(bytesize), ApiKind::DriverAsync);
        telemetry_alloc_success();
        return CUDA_SUCCESS;
    }
    telemetry_alloc_fail();
    t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    return driver_alloc_async_fallback(s, dptr, bytesize, hStream);
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
        return driver_free_async_fallback(s, dptr, hStream);
    }
    if (should_strict_reject()) return CUDA_ERROR_NOT_SUPPORTED;
    if (erase_if_owned(p, active_owner(s)) && can_use_active_regime_for_sync_alloc()) {
        g_in_hook = true;
        (void)active_regime_free(s, p);
        g_in_hook = false;
        telemetry_free_success();
        return CUDA_SUCCESS;
    }
    telemetry_free_miss();
    t.fallback_free_calls.fetch_add(1, std::memory_order_relaxed);
    return driver_free_async_fallback(s, dptr, hStream);
}

extern "C" CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.alloc_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_driver.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_managed.fetch_add(1, std::memory_order_relaxed);
    t.alloc_bytes_requested.fetch_add(static_cast<uint64_t>(bytesize), std::memory_order_relaxed);
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    if (should_strict_reject()) return CUDA_ERROR_NOT_SUPPORTED;
    t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    if (!s.real_cu_mem_alloc_managed) try_recover_symbols();
    if (s.real_cu_mem_alloc_managed) return s.real_cu_mem_alloc_managed(dptr, bytesize, flags);
    if (s.real_cu_mem_alloc_v2) return s.real_cu_mem_alloc_v2(dptr, bytesize);
    if (s.real_cu_mem_alloc) return s.real_cu_mem_alloc(dptr, bytesize);
    return CUDA_ERROR_NOT_INITIALIZED;
}

extern "C" cudaError_t cudaMalloc(void** devPtr, size_t size) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.alloc_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_runtime.fetch_add(1, std::memory_order_relaxed);
    t.alloc_bytes_requested.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    if (!devPtr) return cudaErrorInvalidValue;
    if (should_strict_reject()) return cudaErrorNotSupported;
    if (g_in_hook || !can_use_active_regime_for_sync_alloc()) {
        t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
        if (!s.real_cuda_malloc) try_recover_symbols();
        return s.real_cuda_malloc ? s.real_cuda_malloc(devPtr, size) : cudaErrorInitializationError;
    }

    g_in_hook = true;
    void* p = nullptr;
    const bool ok = active_regime_alloc(s, &p, size);
    g_in_hook = false;
    if (ok) {
        *devPtr = p;
        remember_owned_ptr(p, active_owner(s), static_cast<uint64_t>(size), ApiKind::RuntimeSync);
        telemetry_alloc_success();
        return cudaSuccess;
    }
    telemetry_alloc_fail();
    t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    if (!s.real_cuda_malloc) try_recover_symbols();
    return s.real_cuda_malloc ? s.real_cuda_malloc(devPtr, size) : cudaErrorMemoryAllocation;
}

extern "C" cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.alloc_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_runtime.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_managed.fetch_add(1, std::memory_order_relaxed);
    t.alloc_bytes_requested.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    if (!devPtr) return cudaErrorInvalidValue;
    if (should_strict_reject()) return cudaErrorNotSupported;
    t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    if (!s.real_cuda_malloc_managed) try_recover_symbols();
    if (s.real_cuda_malloc_managed) return s.real_cuda_malloc_managed(devPtr, size, flags);
    if (s.real_cuda_malloc) return s.real_cuda_malloc(devPtr, size);
    return cudaErrorInitializationError;
}

extern "C" cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.alloc_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_runtime.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_pitch.fetch_add(1, std::memory_order_relaxed);
    t.alloc_bytes_requested.fetch_add(static_cast<uint64_t>(width * height), std::memory_order_relaxed);
    if (!devPtr || !pitch) return cudaErrorInvalidValue;
    if (should_strict_reject()) return cudaErrorNotSupported;
    t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    if (!s.real_cuda_malloc_pitch) try_recover_symbols();
    if (s.real_cuda_malloc_pitch) return s.real_cuda_malloc_pitch(devPtr, pitch, width, height);
    if (s.real_cuda_malloc) {
        cudaError_t rc = s.real_cuda_malloc(devPtr, width * height);
        if (rc == cudaSuccess) *pitch = width;
        return rc;
    }
    return cudaErrorInitializationError;
}

extern "C" cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.alloc_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_runtime.fetch_add(1, std::memory_order_relaxed);
    t.alloc_calls_host.fetch_add(1, std::memory_order_relaxed);
    t.alloc_bytes_requested.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    if (!pHost) return cudaErrorInvalidValue;
    t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    if (!s.real_cuda_host_alloc) try_recover_symbols();
    return s.real_cuda_host_alloc ? s.real_cuda_host_alloc(pHost, size, flags) : cudaErrorInitializationError;
}

extern "C" cudaError_t cudaFreeHost(void* pHost) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.free_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.free_calls_runtime.fetch_add(1, std::memory_order_relaxed);
    t.free_calls_host.fetch_add(1, std::memory_order_relaxed);
    t.fallback_free_calls.fetch_add(1, std::memory_order_relaxed);
    if (!s.real_cuda_free_host) try_recover_symbols();
    return s.real_cuda_free_host ? s.real_cuda_free_host(pHost) : cudaErrorInitializationError;
}

extern "C" cudaError_t cudaFree(void* devPtr) {
    initialize();
    State& s = state();
    Telemetry& t = telemetry();
    t.free_calls_total.fetch_add(1, std::memory_order_relaxed);
    t.free_calls_runtime.fetch_add(1, std::memory_order_relaxed);
    if (g_in_hook) return s.real_cuda_free ? s.real_cuda_free(devPtr) : cudaErrorInitializationError;

    if (should_strict_reject()) return cudaErrorNotSupported;
    if (erase_if_owned(devPtr, active_owner(s)) && can_use_active_regime_for_sync_alloc()) {
        g_in_hook = true;
        (void)active_regime_free(s, devPtr);
        g_in_hook = false;
        telemetry_free_success();
        return cudaSuccess;
    }
    telemetry_free_miss();
    t.fallback_free_calls.fetch_add(1, std::memory_order_relaxed);
    if (!s.real_cuda_free) try_recover_symbols();
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
    if (should_strict_reject()) return cudaErrorNotSupported;
    if (!s.async_tlsf || g_in_hook || !can_use_active_regime_for_sync_alloc()) {
        t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
        return runtime_alloc_async_fallback(s, devPtr, size, hStream);
    }
    g_in_hook = true;
    void* p = nullptr;
    const bool ok = active_regime_alloc(s, &p, size);
    g_in_hook = false;
    if (ok) {
        *devPtr = p;
        remember_owned_ptr(p, active_owner(s), static_cast<uint64_t>(size), ApiKind::RuntimeAsync);
        telemetry_alloc_success();
        return cudaSuccess;
    }
    telemetry_alloc_fail();
    t.fallback_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    return runtime_alloc_async_fallback(s, devPtr, size, hStream);
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
        return runtime_free_async_fallback(s, devPtr, hStream);
    }
    if (should_strict_reject()) return cudaErrorNotSupported;
    if (erase_if_owned(devPtr, active_owner(s)) && can_use_active_regime_for_sync_alloc()) {
        g_in_hook = true;
        (void)active_regime_free(s, devPtr);
        g_in_hook = false;
        telemetry_free_success();
        return cudaSuccess;
    }
    telemetry_free_miss();
    t.fallback_free_calls.fetch_add(1, std::memory_order_relaxed);
    return runtime_free_async_fallback(s, devPtr, hStream);
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
    out_stats->alloc_calls_managed = t.alloc_calls_managed.load(std::memory_order_relaxed);
    out_stats->alloc_calls_host = t.alloc_calls_host.load(std::memory_order_relaxed);
    out_stats->free_calls_host = t.free_calls_host.load(std::memory_order_relaxed);
    out_stats->alloc_calls_pitch = t.alloc_calls_pitch.load(std::memory_order_relaxed);
    out_stats->alloc_bytes_requested = t.alloc_bytes_requested.load(std::memory_order_relaxed);
    out_stats->tlsf_alloc_success = t.tlsf_alloc_success.load(std::memory_order_relaxed);
    out_stats->tlsf_alloc_fail = t.tlsf_alloc_fail.load(std::memory_order_relaxed);
    out_stats->tlsf_free_success = t.tlsf_free_success.load(std::memory_order_relaxed);
    out_stats->tlsf_free_miss = t.tlsf_free_miss.load(std::memory_order_relaxed);
    out_stats->sizeclass_alloc_success = t.sizeclass_alloc_success.load(std::memory_order_relaxed);
    out_stats->sizeclass_alloc_fail = t.sizeclass_alloc_fail.load(std::memory_order_relaxed);
    out_stats->sizeclass_free_success = t.sizeclass_free_success.load(std::memory_order_relaxed);
    out_stats->sizeclass_free_miss = t.sizeclass_free_miss.load(std::memory_order_relaxed);
    out_stats->fallback_alloc_calls = t.fallback_alloc_calls.load(std::memory_order_relaxed);
    out_stats->fallback_free_calls = t.fallback_free_calls.load(std::memory_order_relaxed);
    out_stats->fallback_async_to_sync_calls = t.fallback_async_to_sync_calls.load(std::memory_order_relaxed);
    out_stats->strict_mode_reject_calls = t.strict_mode_reject_calls.load(std::memory_order_relaxed);
    out_stats->late_symbol_resolve_attempts = t.late_symbol_resolve_attempts.load(std::memory_order_relaxed);
    out_stats->tracked_allocations_current = tracked_alloc_count();
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
    t.alloc_calls_managed.store(0, std::memory_order_relaxed);
    t.alloc_calls_host.store(0, std::memory_order_relaxed);
    t.free_calls_host.store(0, std::memory_order_relaxed);
    t.alloc_calls_pitch.store(0, std::memory_order_relaxed);
    t.alloc_bytes_requested.store(0, std::memory_order_relaxed);
    t.tlsf_alloc_success.store(0, std::memory_order_relaxed);
    t.tlsf_alloc_fail.store(0, std::memory_order_relaxed);
    t.tlsf_free_success.store(0, std::memory_order_relaxed);
    t.tlsf_free_miss.store(0, std::memory_order_relaxed);
    t.sizeclass_alloc_success.store(0, std::memory_order_relaxed);
    t.sizeclass_alloc_fail.store(0, std::memory_order_relaxed);
    t.sizeclass_free_success.store(0, std::memory_order_relaxed);
    t.sizeclass_free_miss.store(0, std::memory_order_relaxed);
    t.fallback_alloc_calls.store(0, std::memory_order_relaxed);
    t.fallback_free_calls.store(0, std::memory_order_relaxed);
    t.fallback_async_to_sync_calls.store(0, std::memory_order_relaxed);
    t.strict_mode_reject_calls.store(0, std::memory_order_relaxed);
    t.late_symbol_resolve_attempts.store(0, std::memory_order_relaxed);
}

extern "C" int fercuda_intercept_telemetry_enabled(void) {
    initialize();
    return state().enabled ? 1 : 0;
}

extern "C" __attribute__((destructor)) void fercuda_intercept_shutdown() {
    State& s = state();
    if (!s.initialized) return;
    bool expected = false;
    if (!s.shutdown_done.compare_exchange_strong(expected, true)) return;
    if (s.regime == RegimeKind::SizeClass) {
        sizeclass_shutdown(s);
    }
    if (s.gpu_hot_shutdown && s.gpu_hot_runtime) {
        s.gpu_hot_shutdown(s.gpu_hot_runtime);
        s.gpu_hot_runtime = nullptr;
    }
    const bool do_dlclose = env_enabled("FERCUDA_INTERCEPT_DLCLOSE", false);
    if (do_dlclose && s.gpu_hot_handle) {
        dlclose(s.gpu_hot_handle);
        s.gpu_hot_handle = nullptr;
    }
    if (do_dlclose && s.cuda_driver_handle) {
        dlclose(s.cuda_driver_handle);
        s.cuda_driver_handle = nullptr;
    }
    if (do_dlclose && s.cuda_runtime_handle) {
        dlclose(s.cuda_runtime_handle);
        s.cuda_runtime_handle = nullptr;
    }
}
