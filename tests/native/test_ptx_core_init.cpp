#include <unistd.h>

#include "gpu/gpu_hot_runtime.h"
#include "ptx_stable_runtime.h"

#include <cstdio>
#include <cuda_runtime.h>

static int fail(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("test_ptx_core_init: SKIP (no CUDA device)\n");
        return 0;
    }

    // Test gpu_hot_init with a small pool
    GPUHotConfig cfg = gpu_hot_default_config();
    cfg.fixed_pool_size = 64ull * 1024ull * 1024ull;  // 64 MiB
    cfg.quiet_init = true;
    cfg.enable_leak_detection = true;

    GPUHotRuntime* rt = gpu_hot_init_with_config(0, "test_init", &cfg);
    if (!rt) return fail("gpu_hot_init_with_config returned null");

    // Validate stats are populated
    GPUHotStats stats{};
    gpu_hot_get_stats(rt, &stats);
    if (stats.vram_allocated == 0) return fail("vram_allocated is 0 after init");

    // TLSF pool should be healthy
    TLSFHealthReport health{};
    gpu_hot_validate_tlsf_pool(rt, &health);
    if (!health.is_valid) return fail("TLSF pool not valid after init");
    if (health.has_corrupted_blocks) return fail("TLSF has corrupted blocks");

    // Basic alloc/free cycle
    void* ptr = gpu_hot_alloc(rt, 4096);
    if (!ptr) return fail("gpu_hot_alloc(4096) returned null");
    if (!gpu_hot_owns_ptr(rt, ptr)) return fail("gpu_hot_owns_ptr returned false");
    gpu_hot_free(rt, ptr);

    // Stable runtime API
    PTXStableConfig scfg{};
    scfg.struct_size = sizeof(scfg);
    scfg.abi_version = PTX_STABLE_ABI_VERSION;
    scfg.device_id = 0;
    scfg.fixed_pool_size = 64ull * 1024ull * 1024ull;
    scfg.quiet_init = 1;
    PTXStableRuntime* srt = nullptr;
    PTXStableStatus ss = ptx_stable_init(&scfg, &srt);
    if (ss != PTX_STABLE_OK || !srt) return fail("ptx_stable_init failed");

    void* sptr = nullptr;
    ss = ptx_stable_alloc(srt, 8192, &sptr);
    if (ss != PTX_STABLE_OK || !sptr) return fail("ptx_stable_alloc failed");

    bool owned = false;
    ss = ptx_stable_owns_ptr(srt, sptr, &owned);
    if (ss != PTX_STABLE_OK || !owned) return fail("ptx_stable_owns_ptr failed");

    ss = ptx_stable_free(srt, sptr);
    if (ss != PTX_STABLE_OK) return fail("ptx_stable_free failed");

    ptx_stable_release(srt);
    gpu_hot_shutdown(rt);

    std::printf("PTX CORE INIT TEST PASSED\n");
    return 0;
}
