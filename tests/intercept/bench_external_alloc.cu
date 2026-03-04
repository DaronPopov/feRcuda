#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <string>

#include "fercuda/api/intercept_telemetry.h"

struct Args {
    std::string label = "native";
    int iters = 20000;
    size_t bytes = 64 * 1024;
    bool async_api = false;
};

static int parse_int(const char* s, int def) {
    if (!s) return def;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0' || v <= 0) return def;
    return static_cast<int>(v);
}

static size_t parse_size(const char* s, size_t def) {
    if (!s) return def;
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (!end || *end != '\0' || v == 0) return def;
    return static_cast<size_t>(v);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--label") == 0 && i + 1 < argc) {
            a.label = argv[++i];
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            a.iters = parse_int(argv[++i], a.iters);
        } else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
            a.bytes = parse_size(argv[++i], a.bytes);
        } else if (std::strcmp(argv[i], "--async") == 0) {
            a.async_api = true;
        }
    }
    return a;
}

static void print_stats(const Args& a, double seconds) {
    const double ops = static_cast<double>(a.iters);
    const double bytes_total = static_cast<double>(a.bytes) * ops;
    const double ops_per_s = ops / seconds;
    const double gb_per_s = bytes_total / seconds / 1e9;

    std::cout << "bench.label=" << a.label << "\n";
    std::cout << "bench.iters=" << a.iters << "\n";
    std::cout << "bench.bytes=" << a.bytes << "\n";
    std::cout << "bench.async_api=" << (a.async_api ? 1 : 0) << "\n";
    std::cout << "bench.seconds=" << seconds << "\n";
    std::cout << "bench.alloc_ops_per_s=" << ops_per_s << "\n";
    std::cout << "bench.gb_per_s=" << gb_per_s << "\n";
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // Initialize runtime.
    cudaError_t e = cudaFree(nullptr);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
        std::cerr << "cuda_init_error=" << cudaGetErrorString(e) << "\n";
        return 2;
    }

    using reset_fn = void (*)();
    using get_fn = int (*)(fercuda_intercept_stats_t*);
    auto telemetry_reset = reinterpret_cast<reset_fn>(dlsym(RTLD_DEFAULT, "fercuda_intercept_telemetry_reset"));
    auto telemetry_get = reinterpret_cast<get_fn>(dlsym(RTLD_DEFAULT, "fercuda_intercept_telemetry_get"));
    const bool telemetry_present = (telemetry_reset != nullptr && telemetry_get != nullptr);
    std::cout << "intercept.telemetry_present=" << (telemetry_present ? 1 : 0) << "\n";
    if (telemetry_present) telemetry_reset();

    cudaStream_t stream = nullptr;
    if (args.async_api) {
        e = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (e != cudaSuccess) {
            std::cerr << "cuda_stream_create_error=" << cudaGetErrorString(e) << "\n";
            return 3;
        }
    }

    // Warmup.
    void* warm = nullptr;
    if (args.async_api) {
        if (cudaMallocAsync(&warm, args.bytes, stream) == cudaSuccess) {
            cudaFreeAsync(warm, stream);
            cudaStreamSynchronize(stream);
        }
    } else {
        if (cudaMalloc(&warm, args.bytes) == cudaSuccess) {
            cudaFree(warm);
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < args.iters; ++i) {
        void* p = nullptr;
        if (args.async_api) {
            e = cudaMallocAsync(&p, args.bytes, stream);
            if (e != cudaSuccess) {
                std::cerr << "cuda_malloc_async_error=" << cudaGetErrorString(e) << "\n";
                if (stream) cudaStreamDestroy(stream);
                return 4;
            }
            e = cudaFreeAsync(p, stream);
            if (e != cudaSuccess) {
                std::cerr << "cuda_free_async_error=" << cudaGetErrorString(e) << "\n";
                if (stream) cudaStreamDestroy(stream);
                return 5;
            }
        } else {
            e = cudaMalloc(&p, args.bytes);
            if (e != cudaSuccess) {
                std::cerr << "cuda_malloc_error=" << cudaGetErrorString(e) << "\n";
                if (stream) cudaStreamDestroy(stream);
                return 6;
            }
            e = cudaFree(p);
            if (e != cudaSuccess) {
                std::cerr << "cuda_free_error=" << cudaGetErrorString(e) << "\n";
                if (stream) cudaStreamDestroy(stream);
                return 7;
            }
        }
    }

    if (args.async_api) {
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) {
            std::cerr << "cuda_stream_sync_error=" << cudaGetErrorString(e) << "\n";
            cudaStreamDestroy(stream);
            return 8;
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    if (stream) cudaStreamDestroy(stream);

    double seconds = std::chrono::duration<double>(t1 - t0).count();
    if (seconds <= 0.0) seconds = 1e-9;
    print_stats(args, seconds);

    if (telemetry_present) {
        fercuda_intercept_stats_t st{};
        if (telemetry_get(&st) == 0) {
            std::cout << "intercept.init_success=" << st.init_success << "\n";
            std::cout << "intercept.alloc_calls_total=" << st.alloc_calls_total << "\n";
            std::cout << "intercept.fallback_alloc_calls=" << st.fallback_alloc_calls << "\n";
            std::cout << "intercept.tlsf_alloc_success=" << st.tlsf_alloc_success << "\n";
            std::cout << "intercept.tlsf_alloc_fail=" << st.tlsf_alloc_fail << "\n";
            std::cout << "intercept.sizeclass_alloc_success=" << st.sizeclass_alloc_success << "\n";
            std::cout << "intercept.sizeclass_alloc_fail=" << st.sizeclass_alloc_fail << "\n";
            std::cout << "intercept.alloc_bytes_requested=" << st.alloc_bytes_requested << "\n";
            std::cout << "intercept.fallback_async_to_sync_calls=" << st.fallback_async_to_sync_calls << "\n";
            std::cout << "intercept.strict_mode_reject_calls=" << st.strict_mode_reject_calls << "\n";
        }
    }

    return 0;
}
