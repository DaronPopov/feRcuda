#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

static bool has_symbol(void* h, const char* name) {
    return dlsym(h, name) != nullptr;
}

int main() {
    void* so = dlopen("libptx_hook.so", RTLD_NOW | RTLD_LOCAL);
    if (!so) {
        std::cerr << "dlopen(libptx_hook.so) failed: " << dlerror() << "\n";
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
