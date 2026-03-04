#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "fercuda/api/intercept_telemetry.h"

static bool has_symbol(void* h, const char* name) {
    return dlsym(h, name) != nullptr;
}

int main() {
    // Force deterministic init behavior for test environment.
    setenv("FERCUDA_INTERCEPT_ENABLE", "0", 1);
    setenv("FERCUDA_INTERCEPT_MODE", "permissive", 1);

    // Telemetry API smoke.
    fercuda_intercept_telemetry_reset();
    fercuda_intercept_stats_t st{};
    if (fercuda_intercept_telemetry_get(&st) != 0) {
        std::cerr << "telemetry_get failed\n";
        return 2;
    }
    (void)fercuda_intercept_telemetry_enabled();

    // ABI surface check: ensure major intercept wrappers are exported.
    void* so = dlopen("libfercuda_intercept.so", RTLD_NOW | RTLD_LOCAL);
    if (!so) {
        std::cerr << "dlopen(libfercuda_intercept.so) failed: " << dlerror() << "\n";
        return 3;
    }

    const char* required[] = {
        "cudaMalloc",
        "cudaFree",
        "cudaMallocAsync",
        "cudaFreeAsync",
        "cudaMallocManaged",
        "cudaHostAlloc",
        "cudaFreeHost",
        "cudaMallocPitch",
        "cuMemAlloc_v2",
        "cuMemFree_v2",
        "cuMemAllocAsync",
        "cuMemFreeAsync",
        "cuMemAllocManaged",
        "fercuda_intercept_telemetry_get",
        "fercuda_intercept_telemetry_reset",
        "fercuda_intercept_telemetry_enabled",
    };

    for (const char* sym : required) {
        if (!has_symbol(so, sym)) {
            std::cerr << "missing intercept symbol: " << sym << "\n";
            dlclose(so);
            return 4;
        }
    }

    dlclose(so);
    std::cout << "intercept compatibility test: PASS\n";
    return 0;
}
