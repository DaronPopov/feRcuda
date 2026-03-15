/**
 * Benchmark: TLSF (libptx_hook) vs native cudaMalloc/cudaFree
 *
 * Run twice:
 *   1. Native:  ./bench_alloc_vs_cudamalloc
 *   2. TLSF:    LD_PRELOAD=libptx_hook.so ./bench_alloc_vs_cudamalloc
 *
 * Or use: bash scripts/bench_alloc_vs_cudamalloc.sh
 */
#include <cuda_runtime_api.h>

#include <chrono>
#ifdef __linux__
#include <dlfcn.h>
#endif
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct Config {
    int iters = 50000;
    size_t bytes = 65536;  // 64KB
    int warmup = 1000;
};

static Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            c.iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
            c.bytes = static_cast<size_t>(std::atoll(argv[++i]));
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            c.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("Usage: bench_alloc_vs_cudamalloc [--iters N] [--bytes N] [--warmup N]\n");
            std::printf("  --iters N   alloc+free iterations (default: 50000)\n");
            std::printf("  --bytes N   bytes per allocation (default: 65536)\n");
            std::printf("  --warmup N  warmup iterations (default: 1000)\n");
            std::printf("\nRun with LD_PRELOAD=libptx_hook.so for TLSF path.\n");
            std::exit(0);
        }
    }
    return c;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    // Detect if we're using TLSF (ptx_hook) by checking for the symbol
    bool using_tlsf = false;
#ifdef __linux__
    void* handle = dlopen(nullptr, RTLD_NOW);
    if (handle) {
        using_tlsf = (dlsym(handle, "gpu_hot_alloc") != nullptr);
        dlclose(handle);
    }
#endif

    cudaError_t e = cudaSetDevice(0);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice(0) failed: %s\n", cudaGetErrorString(e));
        return 1;
    }

    e = cudaFree(nullptr);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
        std::fprintf(stderr, "cuda init failed: %s\n", cudaGetErrorString(e));
        return 2;
    }

    // Warmup
    for (int i = 0; i < cfg.warmup; ++i) {
        void* p = nullptr;
        e = cudaMalloc(&p, cfg.bytes);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "warmup cudaMalloc failed: %s\n", cudaGetErrorString(e));
            return 3;
        }
        e = cudaFree(p);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "warmup cudaFree failed: %s\n", cudaGetErrorString(e));
            return 4;
        }
    }

    // Timed run
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.iters; ++i) {
        void* p = nullptr;
        e = cudaMalloc(&p, cfg.bytes);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc failed at iter %d: %s\n", i, cudaGetErrorString(e));
            return 5;
        }
        e = cudaFree(p);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "cudaFree failed at iter %d: %s\n", i, cudaGetErrorString(e));
            return 6;
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    double ns_per_op = ns / static_cast<double>(cfg.iters);
    double ops_per_s = 1e9 / ns_per_op;

    std::printf("mode=%s\n", using_tlsf ? "tlsf" : "cudaMalloc");
    std::printf("iters=%d\n", cfg.iters);
    std::printf("bytes=%zu\n", cfg.bytes);
    std::printf("ns_per_alloc_free=%.1f\n", ns_per_op);
    std::printf("alloc_free_per_sec=%.0f\n", ops_per_s);
    std::printf("total_sec=%.4f\n", ns / 1e9);

    return 0;
}
