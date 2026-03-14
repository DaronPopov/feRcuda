// Weak references to context hook functions (defined in cuda_context_hook.c).
// These must be at file scope with extern "C" because the hook library is
// compiled as C by gcc — without C linkage, nvcc applies C++ name mangling
// and the weak references silently resolve to NULL at runtime.
extern "C" void ptx_context_hook_capture(int, CUcontext) __attribute__((weak));
extern "C" void ptx_context_hook_set_ptx_active(bool) __attribute__((weak));

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#endif

// Cross-platform atomic operations and timing for host-side shared memory
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <time.h>
#include <errno.h>

#define InterlockedIncrement(ptr) __sync_add_and_fetch(ptr, 1)
#define InterlockedDecrement(ptr) __sync_sub_and_fetch(ptr, 1)
#define InterlockedAdd64(ptr, val) __sync_add_and_fetch(ptr, val)
#define LONG int
#define LONG64 int64_t

// Linux implementation of GetTickCount64
static uint64_t GetTickCount64() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
#endif

// ============================================================================
// Async TLSF deferred free support
// ============================================================================

#ifdef _WIN32
typedef CRITICAL_SECTION PTXMutex;
static void ptx_mutex_init(PTXMutex* m) { InitializeCriticalSection(m); }
static void ptx_mutex_lock(PTXMutex* m) { EnterCriticalSection(m); }
static void ptx_mutex_unlock(PTXMutex* m) { LeaveCriticalSection(m); }
static void ptx_mutex_destroy(PTXMutex* m) { DeleteCriticalSection(m); }
#else
typedef pthread_mutex_t PTXMutex;
static void ptx_mutex_init(PTXMutex* m) { pthread_mutex_init(m, NULL); }
static void ptx_mutex_lock(PTXMutex* m) { pthread_mutex_lock(m); }
static void ptx_mutex_unlock(PTXMutex* m) { pthread_mutex_unlock(m); }
static void ptx_mutex_destroy(PTXMutex* m) { pthread_mutex_destroy(m); }
#endif

typedef struct DeferredFreeEntry {
    void* ptr;
    cudaEvent_t event;
    cudaStream_t stream;
    struct DeferredFreeEntry* next;
} DeferredFreeEntry;

static cudaEvent_t ptx_event_acquire(GPUHotRuntime* runtime);
static void ptx_event_release(GPUHotRuntime* runtime, cudaEvent_t ev);

static bool ptx_tlsf_debug_enabled() {
    static int initialized = 0;
    static bool enabled = false;
    if (!initialized) {
        const char* v = getenv("PTX_TLSF_DEBUG");
        enabled = (v && v[0] && v[0] != '0');
        initialized = 1;
    }
    return enabled;
}
static void gpu_hot_poll_deferred_internal(GPUHotRuntime* runtime, int max_drain);

// Runtime state — defined here (before NVML/CUPTI helpers that dereference it)
struct GPUHotRuntime {
    int device_id;
    CUcontext cu_context;  // Captured primary context for context hook enforcement
    cudaStream_t streams[GPU_HOT_MAX_STREAMS];
    int num_streams;

    // TLSF VRAM Allocator (Unified)
    void* vram_pool;
    size_t vram_pool_size;
    PTXTLSFAllocator* tlsf_allocator;
    int fallback_count;           // Number of cudaMalloc fallbacks

    // CUDA Graphs
    GPUGraphHandle graphs[GPU_HOT_MAX_GRAPHS];
    int num_graphs;
    int capturing_stream_id;  // -1 if not capturing

    // Registered kernels
    GPUKernelHandle kernels[GPU_HOT_MAX_KERNELS];
    int num_kernels;

    // Shared Memory Registry (The GPU "File System")
    GPURegistryEntry* global_registry;
    PTXSystemState* system_state; // New global system vitals
#ifdef _WIN32
    HANDLE shm_handle;
#else
    int shm_handle; // Not used on Linux, but keep for compatibility
#endif
    int shm_count;

    // Keepalive
    bool keepalive_running;
    cudaEvent_t keepalive_event;

    // Vitals & Guardrails
    uint64_t total_ops;
    int watchdog_timeout_ms;
    bool watchdog_tripped;
    long long last_launch_time;

    // Async deferred free queue
    PTXMutex async_lock;
    DeferredFreeEntry* deferred_head;
    DeferredFreeEntry* deferred_tail;
    int deferred_count;

    // Event pool for async frees
    cudaEvent_t* event_pool;
    int event_pool_count;
    int event_pool_capacity;

    // VMM reference for eviction bridge
    VMMState* vmm;
};

#ifndef _WIN32
// ============================================================================
// Optional NVML dynamic loading (Linux/NVIDIA)
// ============================================================================
typedef int nvmlReturn_t;
typedef struct nvmlDevice_st* nvmlDevice_t;
typedef struct nvmlUtilization_st {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

#define NVML_SUCCESS 0
#define NVML_CLOCK_SM 1
#define NVML_CLOCK_MEM 2
#define NVML_TEMPERATURE_GPU 0

typedef nvmlReturn_t (*nvmlInit_v2_fn)(void);
typedef nvmlReturn_t (*nvmlShutdown_fn)(void);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_v2_fn)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*nvmlDeviceGetUtilizationRates_fn)(nvmlDevice_t, nvmlUtilization_t*);
typedef nvmlReturn_t (*nvmlDeviceGetClockInfo_fn)(nvmlDevice_t, int, unsigned int*);
typedef nvmlReturn_t (*nvmlDeviceGetPowerUsage_fn)(nvmlDevice_t, unsigned int*);
typedef nvmlReturn_t (*nvmlDeviceGetTemperature_fn)(nvmlDevice_t, unsigned int, unsigned int*);

typedef struct NVMLDynApi {
    int initialized;
    int available;
    void* lib;
    nvmlDevice_t device;
    int device_id;
    nvmlInit_v2_fn init_v2;
    nvmlShutdown_fn shutdown;
    nvmlDeviceGetHandleByIndex_v2_fn get_handle;
    nvmlDeviceGetUtilizationRates_fn get_util;
    nvmlDeviceGetClockInfo_fn get_clock;
    nvmlDeviceGetPowerUsage_fn get_power;
    nvmlDeviceGetTemperature_fn get_temp;
} NVMLDynApi;

static NVMLDynApi g_nvml = {0};

// ============================================================================
// Optional CUPTI dynamic loading (Linux/NVIDIA)
// ============================================================================
typedef int CUptiResult;
typedef uint32_t CUpti_EventID;
typedef void* CUpti_EventGroup;
typedef struct CUpti_EventGroupSet {
    uint32_t numEventGroups;
    CUpti_EventGroup* eventGroups;
} CUpti_EventGroupSet;
typedef struct CUpti_EventGroupSets {
    uint32_t numSets;
    CUpti_EventGroupSet* sets;
} CUpti_EventGroupSets;

#define CUPTI_SUCCESS 0
#define CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS 1

typedef CUptiResult (*cuptiEventGetIdFromName_fn)(CUdevice, const char*, CUpti_EventID*);
typedef CUptiResult (*cuptiSetEventCollectionMode_fn)(CUcontext, int);
typedef CUptiResult (*cuptiEventGroupSetsCreate_fn)(CUcontext, size_t, CUpti_EventID*, CUpti_EventGroupSets**);
typedef CUptiResult (*cuptiEventGroupSetEnable_fn)(CUpti_EventGroupSet*);
typedef CUptiResult (*cuptiEventGroupReadEvent_fn)(CUpti_EventGroup, uint32_t, CUpti_EventID, size_t*, uint64_t*);

typedef struct CUPTIDynApi {
    int initialized;
    int available;
    void* lib;
    cuptiEventGetIdFromName_fn event_id_from_name;
    cuptiSetEventCollectionMode_fn set_collection_mode;
    cuptiEventGroupSetsCreate_fn group_sets_create;
    cuptiEventGroupSetEnable_fn group_set_enable;
    cuptiEventGroupReadEvent_fn read_event;
    CUpti_EventGroupSets* sets;
    CUpti_EventID event_ids[4];
    int event_count;
    uint64_t last_counts[4];
    uint64_t last_ms;
} CUPTIDynApi;

static CUPTIDynApi g_cupti = {0};

static void nvml_try_init(int device_id) {
    if (g_nvml.initialized) {
        return;
    }
    g_nvml.initialized = 1;
    g_nvml.device_id = device_id;

    const char* libs[] = {
        "libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib64/libnvidia-ml.so.1",
        NULL
    };
    for (int i = 0; libs[i] != NULL && !g_nvml.lib; ++i) {
        g_nvml.lib = dlopen(libs[i], RTLD_LAZY);
    }
    if (!g_nvml.lib) return;

    g_nvml.init_v2 = (nvmlInit_v2_fn)dlsym(g_nvml.lib, "nvmlInit_v2");
    g_nvml.shutdown = (nvmlShutdown_fn)dlsym(g_nvml.lib, "nvmlShutdown");
    g_nvml.get_handle =
        (nvmlDeviceGetHandleByIndex_v2_fn)dlsym(g_nvml.lib, "nvmlDeviceGetHandleByIndex_v2");
    g_nvml.get_util =
        (nvmlDeviceGetUtilizationRates_fn)dlsym(g_nvml.lib, "nvmlDeviceGetUtilizationRates");
    g_nvml.get_clock = (nvmlDeviceGetClockInfo_fn)dlsym(g_nvml.lib, "nvmlDeviceGetClockInfo");
    g_nvml.get_power = (nvmlDeviceGetPowerUsage_fn)dlsym(g_nvml.lib, "nvmlDeviceGetPowerUsage");
    g_nvml.get_temp = (nvmlDeviceGetTemperature_fn)dlsym(g_nvml.lib, "nvmlDeviceGetTemperature");

    // Only init/get_handle/get_util are required for "valid" polling.
    if (!g_nvml.init_v2 || !g_nvml.get_handle || !g_nvml.get_util) {
        dlclose(g_nvml.lib);
        memset(&g_nvml, 0, sizeof(g_nvml));
        g_nvml.initialized = 1;
        return;
    }

    if (g_nvml.init_v2() != NVML_SUCCESS) {
        dlclose(g_nvml.lib);
        memset(&g_nvml, 0, sizeof(g_nvml));
        g_nvml.initialized = 1;
        return;
    }

    if (g_nvml.get_handle((unsigned int)device_id, &g_nvml.device) != NVML_SUCCESS) {
        // Fallback: probe first few devices in case runtime device id is stale/mapped differently.
        for (unsigned int i = 0; i < 8; ++i) {
            if (g_nvml.get_handle(i, &g_nvml.device) == NVML_SUCCESS) {
                g_nvml.device_id = (int)i;
                break;
            }
        }
    }
    if (!g_nvml.device) {
        if (g_nvml.shutdown) g_nvml.shutdown();
        dlclose(g_nvml.lib);
        memset(&g_nvml, 0, sizeof(g_nvml));
        g_nvml.initialized = 1;
        return;
    }

    g_nvml.available = 1;
}

static void nvml_poll_stats(GPUHotRuntime* runtime, GPUHotStats* stats) {
    nvml_try_init(runtime->device_id);
    if (!g_nvml.available) {
        return;
    }

    nvmlUtilization_t util;
    unsigned int sm_clock = 0;
    unsigned int mem_clock = 0;
    unsigned int power_mw = 0;
    unsigned int temp_c = 0;

    if (g_nvml.get_util(g_nvml.device, &util) != NVML_SUCCESS) return;

    stats->gpu_utilization = (float)util.gpu;
    stats->mem_utilization = (float)util.memory;
    stats->nvml_valid = true;

    if (g_nvml.get_clock &&
        g_nvml.get_clock(g_nvml.device, NVML_CLOCK_SM, &sm_clock) == NVML_SUCCESS) {
        stats->sm_clock_mhz = sm_clock;
    }
    if (g_nvml.get_clock &&
        g_nvml.get_clock(g_nvml.device, NVML_CLOCK_MEM, &mem_clock) == NVML_SUCCESS) {
        stats->mem_clock_mhz = mem_clock;
    }
    if (g_nvml.get_power && g_nvml.get_power(g_nvml.device, &power_mw) == NVML_SUCCESS) {
        stats->power_w = (float)power_mw / 1000.0f;
    }
    if (g_nvml.get_temp &&
        g_nvml.get_temp(g_nvml.device, NVML_TEMPERATURE_GPU, &temp_c) == NVML_SUCCESS) {
        stats->temperature_c = (int32_t)temp_c;
    }
}

static void cupti_try_init(GPUHotRuntime* runtime) {
    if (g_cupti.initialized) return;
    g_cupti.initialized = 1;

    const char* cupti_env = getenv("CUPTI_LIB_DIR");
    char env_lib_0[512] = {0};
    char env_lib_1[512] = {0};
    if (cupti_env && cupti_env[0]) {
        snprintf(env_lib_0, sizeof(env_lib_0), "%s/libcupti.so", cupti_env);
        snprintf(env_lib_1, sizeof(env_lib_1), "%s/libcupti.so.12", cupti_env);
    }
    const char* libs[] = {
        env_lib_0,
        env_lib_1,
        "libcupti.so",
        "libcupti.so.12",
        "libcupti.so.11",
        "/usr/local/cuda/targets/x86_64-linux/lib/libcupti.so",
        "/usr/local/cuda/targets/aarch64-linux/lib/libcupti.so",
        "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so",
        NULL
    };
    for (int i = 0; libs[i] != NULL && !g_cupti.lib; ++i) {
        if (!libs[i][0]) continue;
        g_cupti.lib = dlopen(libs[i], RTLD_LAZY);
    }
    if (!g_cupti.lib) return;

    g_cupti.event_id_from_name =
        (cuptiEventGetIdFromName_fn)dlsym(g_cupti.lib, "cuptiEventGetIdFromName");
    g_cupti.set_collection_mode =
        (cuptiSetEventCollectionMode_fn)dlsym(g_cupti.lib, "cuptiSetEventCollectionMode");
    g_cupti.group_sets_create =
        (cuptiEventGroupSetsCreate_fn)dlsym(g_cupti.lib, "cuptiEventGroupSetsCreate");
    g_cupti.group_set_enable =
        (cuptiEventGroupSetEnable_fn)dlsym(g_cupti.lib, "cuptiEventGroupSetEnable");
    g_cupti.read_event =
        (cuptiEventGroupReadEvent_fn)dlsym(g_cupti.lib, "cuptiEventGroupReadEvent");

    if (!g_cupti.event_id_from_name || !g_cupti.set_collection_mode || !g_cupti.group_sets_create ||
        !g_cupti.group_set_enable || !g_cupti.read_event) {
        dlclose(g_cupti.lib);
        memset(&g_cupti, 0, sizeof(g_cupti));
        g_cupti.initialized = 1;
        return;
    }

    CUcontext ctx = runtime->cu_context;
    if (!ctx) {
        cuCtxGetCurrent(&ctx);
    }
    if (!ctx) {
        if (cuInit(0) == CUDA_SUCCESS) {
            CUdevice dev0 = 0;
            if (cuDeviceGet(&dev0, runtime->device_id) == CUDA_SUCCESS) {
                cuDevicePrimaryCtxRetain(&ctx, dev0);
                if (ctx) {
                    cuCtxSetCurrent(ctx);
                }
            }
        }
    }
    if (!ctx) return;

    CUdevice dev = 0;
    if (cuDeviceGet(&dev, runtime->device_id) != CUDA_SUCCESS) return;

    const char* names[] = {"flop_count_sp", "flop_count_hp", "flop_count_dp", "flop_count_tensor"};
    for (int i = 0; i < 4; ++i) {
        CUpti_EventID id = 0;
        if (g_cupti.event_id_from_name(dev, names[i], &id) == CUPTI_SUCCESS) {
            g_cupti.event_ids[g_cupti.event_count++] = id;
        }
    }
    if (g_cupti.event_count == 0) return;

    if (g_cupti.set_collection_mode(ctx, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS) != CUPTI_SUCCESS) {
        return;
    }
    if (g_cupti.group_sets_create(ctx, g_cupti.event_count * sizeof(CUpti_EventID), g_cupti.event_ids,
                                  &g_cupti.sets) != CUPTI_SUCCESS) {
        return;
    }
    if (!g_cupti.sets || g_cupti.sets->numSets == 0) return;
    if (g_cupti.group_set_enable(&g_cupti.sets->sets[0]) != CUPTI_SUCCESS) return;

    g_cupti.last_ms = GetTickCount64();
    g_cupti.available = 1;
}

static void cupti_poll_stats(GPUHotRuntime* runtime, GPUHotStats* stats) {
    cupti_try_init(runtime);
    if (!g_cupti.available || !g_cupti.sets || g_cupti.sets->numSets == 0) return;

    CUpti_EventGroupSet* set0 = &g_cupti.sets->sets[0];
    uint64_t accum_now[4] = {0, 0, 0, 0};
    for (uint32_t g = 0; g < set0->numEventGroups; ++g) {
        for (int e = 0; e < g_cupti.event_count; ++e) {
            size_t sz = sizeof(uint64_t);
            uint64_t v = 0;
            if (g_cupti.read_event(set0->eventGroups[g], 0, g_cupti.event_ids[e], &sz, &v) == CUPTI_SUCCESS) {
                accum_now[e] += v;
            }
        }
    }

    uint64_t now_ms = GetTickCount64();
    double dt = (double)(now_ms - g_cupti.last_ms) / 1000.0;
    if (dt <= 0.0) dt = 1e-3;

    double total_delta = 0.0;
    for (int e = 0; e < g_cupti.event_count; ++e) {
        uint64_t prev = g_cupti.last_counts[e];
        uint64_t curr = accum_now[e];
        if (curr >= prev) {
            total_delta += (double)(curr - prev);
        }
        g_cupti.last_counts[e] = curr;
    }
    g_cupti.last_ms = now_ms;

    stats->hw_ops_per_sec = (float)(total_delta / dt);
    stats->gflops_total = (float)(total_delta / dt / 1e9);
    stats->cupti_valid = true;
}
#endif


// Keepalive kernel - runs continuously to keep GPU hot
__global__ void keepalive_kernel(volatile int* flag) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Minimal work to keep GPU active
    if (tid == 0) {
        atomicAdd((int*)flag, 1);
    }
    
    // Touch some memory to keep caches warm
    __shared__ int shared_data[256];
    shared_data[threadIdx.x] = tid;
    __syncthreads();
    
    // Prevent optimization
    if (shared_data[threadIdx.x] < 0) {
        atomicAdd((int*)flag, shared_data[threadIdx.x]);
    }
}

// ============================================================================
// ============================================================================
// Configuration API
// ============================================================================

GPUHotConfig gpu_hot_default_config(void) {
    GPUHotConfig config;
    memset(&config, 0, sizeof(GPUHotConfig));

    // Default: Use all available VRAM minus headroom
    config.pool_fraction = 1.0f;
    config.fixed_pool_size = 0;  // Use fraction-based sizing

    // Size limits
    config.min_pool_size = 256ULL * 1024 * 1024;   // 256MB minimum
    config.max_pool_size = 0;                       // No maximum (0 = unlimited)

    // Safety margin for CUDA runtime (0 = auto)
    config.reserve_vram = 0;    // Auto-reserve headroom

    // Allocator features
    config.enable_leak_detection = true;
    config.enable_pool_health = true;
    config.warning_threshold = 0.9f;  // Warn at 90% utilization

    // Behavior
    config.force_daemon_mode = false;
    config.quiet_init = false;

    // Stream configuration
    config.max_streams = 16;    // Default to 16 streams

    // Single-pool strict mode (off by default)
    config.single_pool_strict = false;

    return config;
}

// ============================================================================
// Environment overrides
// ============================================================================

static bool parse_env_double(const char* key, double* out) {
    const char* val = getenv(key);
    if (!val || !out) return false;
    char* end = NULL;
    double v = strtod(val, &end);
    if (end == val) return false;
    *out = v;
    return true;
}

static bool parse_env_size(const char* key, size_t* out) {
    const char* val = getenv(key);
    if (!val || !out) return false;
    char* end = NULL;
    double v = strtod(val, &end);
    if (end == val) return false;
    while (*end == ' ' || *end == '\t') end++;
    double scale = 1.0;
    if (*end != '\0') {
        if (strcasecmp(end, "k") == 0 || strcasecmp(end, "kb") == 0) {
            scale = 1024.0;
        } else if (strcasecmp(end, "m") == 0 || strcasecmp(end, "mb") == 0) {
            scale = 1024.0 * 1024.0;
        } else if (strcasecmp(end, "g") == 0 || strcasecmp(end, "gb") == 0) {
            scale = 1024.0 * 1024.0 * 1024.0;
        } else {
            // Unknown suffix; treat as bytes
            scale = 1.0;
        }
    }
    if (v <= 0) return false;
    *out = (size_t)(v * scale);
    return true;
}

static void apply_env_overrides(GPUHotConfig* config) {
    if (!config) return;

    // Stream count override
    size_t max_streams = 0;
    if (parse_env_size("PTX_MAX_STREAMS", &max_streams) && max_streams > 0) {
        if (max_streams > GPU_HOT_MAX_STREAMS) {
            max_streams = GPU_HOT_MAX_STREAMS;
        }
        config->max_streams = (unsigned int)max_streams;
    }

    // Fixed pool size override
    size_t fixed = 0;
    if (parse_env_size("PTX_POOL_SIZE", &fixed) && fixed > 0) {
        config->fixed_pool_size = fixed;
        config->pool_fraction = 0.0f;
    }

    // Pool fraction override (0..1 or percent 1..100)
    double frac = 0.0;
    if (parse_env_double("PTX_POOL_FRACTION", &frac)) {
        if (frac > 1.0 && frac <= 100.0) {
            frac = frac / 100.0;
        }
        if (frac > 0.0) {
            config->pool_fraction = (float)frac;
            if (config->fixed_pool_size > 0) {
                config->fixed_pool_size = 0;
            }
        }
    }

    // Warning threshold override (0..1 or percent 1..100)
    double warn = 0.0;
    if (parse_env_double("PTX_POOL_WARN_THRESHOLD", &warn)) {
        if (warn > 1.0 && warn <= 100.0) {
            warn = warn / 100.0;
        }
        if (warn >= 0.0) {
            config->warning_threshold = (float)warn;
        }
    }
    if (getenv("PTX_POOL_WARN_DISABLE")) {
        config->warning_threshold = 0.0f;
    }

    // Single-pool strict mode override
    const char* strict_val = getenv("PTX_SINGLE_POOL_STRICT");
    if (strict_val && strict_val[0] == '1') {
        config->single_pool_strict = true;
    }
}
