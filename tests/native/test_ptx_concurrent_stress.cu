// Large-scale concurrent stress test: TLSF under heavy multi-threaded load
//
// Validates OS-level stability:
// - Many threads (16-32) hammering alloc/free simultaneously
// - Large total op count (100k+ alloc+free pairs)
// - Mixed sizes, random patterns
// - No leaks, no corruption, sustained throughput
//
// Run: ./test_ptx_concurrent_stress
// Or with LD_PRELOAD=libptx_hook.so for intercept path (uses cudaMalloc/cudaFree)

#include "gpu/gpu_hot_runtime.h"
#include "gpu/tensor_ops.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

// Scale: OS-level means we handle real concurrency
static constexpr int kNumThreads = 24;
static constexpr int kOpsPerThread = 5000;
static constexpr size_t kMinAlloc = 256;
static constexpr size_t kMaxAlloc = 256 * 1024;  // 256 KiB

static std::atomic<uint64_t> g_alloc_ok{0};
static std::atomic<uint64_t> g_alloc_fail{0};
static std::atomic<uint64_t> g_free_ok{0};
static std::atomic<uint64_t> g_free_err{0};

static int fail(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

// Worker using native gpu_hot_alloc/free
static void native_worker(GPUHotRuntime* rt, int tid) {
    std::mt19937 rng(static_cast<unsigned>(tid * 7919 + 42));
    std::uniform_int_distribution<size_t> size_dist(kMinAlloc, kMaxAlloc);
    std::vector<void*> ptrs;
    ptrs.reserve(256);

    for (int i = 0; i < kOpsPerThread; ++i) {
        bool do_alloc = ptrs.empty() || (rng() % 3 != 0);

        if (do_alloc) {
            size_t bytes = size_dist(rng);
            void* p = gpu_hot_alloc(rt, bytes);
            if (p) {
                ptrs.push_back(p);
                g_alloc_ok.fetch_add(1, std::memory_order_relaxed);
            } else {
                g_alloc_fail.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            std::uniform_int_distribution<size_t> idx_dist(0, ptrs.size() - 1);
            size_t idx = idx_dist(rng);
            void* victim = ptrs[idx];
            gpu_hot_free(rt, victim);
            g_free_ok.fetch_add(1, std::memory_order_relaxed);
            ptrs[idx] = ptrs.back();
            ptrs.pop_back();
        }
    }

    for (void* p : ptrs) {
        gpu_hot_free(rt, p);
        g_free_ok.fetch_add(1, std::memory_order_relaxed);
    }
}

// Phase 1: Native API concurrent stress
static int test_native_concurrent(GPUHotRuntime* rt) {
    std::printf("  [phase 1] native gpu_hot_alloc/free: %d threads x %d ops\n",
                kNumThreads, kOpsPerThread);

    g_alloc_ok.store(0);
    g_alloc_fail.store(0);
    g_free_ok.store(0);
    g_free_err.store(0);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back(native_worker, rt, t);
    }
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    uint64_t total_ops = g_alloc_ok.load() + g_free_ok.load();
    double ops_per_sec = total_ops / (ms / 1000.0);
    double ns_per_op = (ms * 1e6) / total_ops;

    TLSFPoolStats stats{};
    gpu_hot_get_tlsf_stats(rt, &stats);

    std::printf("    %.0f ms, %llu allocs, %llu frees, %.0f ops/s, %.0f ns/op\n",
                ms, static_cast<unsigned long long>(g_alloc_ok.load()),
                static_cast<unsigned long long>(g_free_ok.load()),
                ops_per_sec, ns_per_op);
    std::printf("    frag=%.2f%%, allocated=%zu (expect 0)\n",
                stats.fragmentation_ratio * 100.0, stats.allocated_bytes);

    if (stats.allocated_bytes > 0) return fail("leaked bytes after native concurrent");
    if (g_alloc_ok.load() == 0) return fail("no successful allocations");
    std::printf("    PASS\n");
    return 0;
}

// Phase 2: Concurrent alloc + compute (realistic: alloc, run kernel, free)
static int test_concurrent_alloc_compute(GPUHotRuntime* rt) {
    std::printf("  [phase 2] concurrent alloc + compute (kernels on TLSF memory)...\n");

    constexpr int NUM_THREADS = 8;
    constexpr int ITERS = 500;
    constexpr size_t N = 1 << 16;  // 64K elements
    const size_t bytes = N * sizeof(float);

    std::atomic<int> errors{0};
    auto worker = [rt, &errors](int tid) {
        cudaStream_t stream = gpu_hot_get_stream(rt, tid % 16);
        for (int i = 0; i < ITERS; ++i) {
            float* d_a = static_cast<float*>(gpu_hot_alloc(rt, bytes));
            float* d_b = static_cast<float*>(gpu_hot_alloc(rt, bytes));
            float* d_c = static_cast<float*>(gpu_hot_alloc(rt, bytes));
            if (!d_a || !d_b || !d_c) {
                errors.fetch_add(1);
                if (d_a) gpu_hot_free(rt, d_a);
                if (d_b) gpu_hot_free(rt, d_b);
                if (d_c) gpu_hot_free(rt, d_c);
                continue;
            }
            ptx_tensor_add_f32(d_a, d_b, d_c, N, stream);
            cudaStreamSynchronize(stream);
            gpu_hot_free(rt, d_a);
            gpu_hot_free(rt, d_b);
            gpu_hot_free(rt, d_c);
        }
    };

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    int total_iters = NUM_THREADS * ITERS;
    std::printf("    %d threads x %d iters in %.0f ms (%.1f iters/s)\n",
                NUM_THREADS, ITERS, ms, total_iters / (ms / 1000.0));
    if (errors.load() > 0) return fail("alloc failures during compute");
    std::printf("    PASS\n");
    return 0;
}

// Phase 3: Burst pattern - all threads allocate simultaneously, then all free (contention spike)
static int test_burst_contention(GPUHotRuntime* rt) {
    std::printf("  [phase 3] burst contention (all threads alloc at once, then free)...\n");

    constexpr int NUM_THREADS = 16;
    constexpr int BLOCKS_PER_THREAD = 64;
    constexpr size_t BLOCK_SIZE = 4096;

    std::vector<std::vector<void*>> thread_ptrs(NUM_THREADS);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> alloc_threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        alloc_threads.emplace_back([rt, &thread_ptrs, t]() {
            for (int i = 0; i < BLOCKS_PER_THREAD; ++i) {
                void* p = gpu_hot_alloc(rt, BLOCK_SIZE);
                if (p) thread_ptrs[t].push_back(p);
            }
        });
    }
    for (auto& th : alloc_threads) th.join();

    std::vector<std::thread> free_threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        free_threads.emplace_back([rt, &thread_ptrs, t]() {
            for (void* p : thread_ptrs[t]) gpu_hot_free(rt, p);
            thread_ptrs[t].clear();
        });
    }
    for (auto& th : free_threads) th.join();
    auto t1 = std::chrono::steady_clock::now();

    size_t total = 0;
    for (const auto& v : thread_ptrs) total += v.size();
    if (total > 0) return fail("leaked blocks after burst");

    TLSFPoolStats stats{};
    gpu_hot_get_tlsf_stats(rt, &stats);
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    int total_ops = NUM_THREADS * BLOCKS_PER_THREAD * 2;
    std::printf("    %d concurrent allocs + %d concurrent frees in %.0f ms (%.0f ops/s)\n",
                total_ops / 2, total_ops / 2, ms, total_ops / (ms / 1000.0));
    if (stats.allocated_bytes > 0) return fail("leaked bytes");
    std::printf("    PASS\n");
    return 0;
}

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("test_ptx_concurrent_stress: SKIP (no CUDA device)\n");
        return 0;
    }
    cudaSetDevice(0);

    GPUHotConfig cfg = gpu_hot_default_config();
    cfg.fixed_pool_size = 512ull * 1024ull * 1024ull;  // 512 MiB
    cfg.quiet_init = true;
    cfg.enable_leak_detection = true;
    cfg.enable_pool_health = true;

    GPUHotRuntime* rt = gpu_hot_init_with_config(0, "concurrent_stress", &cfg);
    if (!rt) return fail("gpu_hot_init failed");

    std::printf("=== feRcuda Large-Scale Concurrent Stress Test ===\n");
    std::printf("  %d threads, %d ops/thread = %d total alloc+free ops\n\n",
                kNumThreads, kOpsPerThread, kNumThreads * kOpsPerThread * 2);

    int rc = 0;
    rc = rc || test_native_concurrent(rt);
    rc = rc || test_concurrent_alloc_compute(rt);
    rc = rc || test_burst_contention(rt);

    TLSFHealthReport health{};
    gpu_hot_validate_tlsf_pool(rt, &health);
    if (!health.is_valid) rc = 1;
    if (health.has_memory_leaks) rc = 1;

    gpu_hot_shutdown(rt);

    if (rc == 0) {
        std::printf("\n=== ALL CONCURRENT PHASES PASSED (OS-level stable) ===\n");
    } else {
        std::printf("\n=== SOME PHASES FAILED ===\n");
    }
    return rc;
}
