#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

static constexpr int kNumThreads = 4;
static constexpr int kOpsPerThread = 64;
static constexpr size_t kMinAlloc = 256;
static constexpr size_t kMaxAlloc = 1 << 16;  // 64 KiB

static std::atomic<int> g_alloc_ok{0};
static std::atomic<int> g_alloc_fail{0};
static std::atomic<int> g_free_ok{0};
static std::atomic<int> g_free_errors{0};

static void stress_worker(int tid) {
    std::mt19937 rng(static_cast<unsigned>(tid * 7919 + 42));
    std::uniform_int_distribution<size_t> size_dist(kMinAlloc, kMaxAlloc);
    std::vector<void*> ptrs;
    ptrs.reserve(kOpsPerThread);

    for (int i = 0; i < kOpsPerThread; ++i) {
        const bool do_alloc = ptrs.empty() || (rng() % 3 != 0);

        if (do_alloc) {
            const size_t bytes = size_dist(rng);
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, bytes);
            if (err == cudaSuccess && ptr) {
                ptrs.push_back(ptr);
                g_alloc_ok.fetch_add(1, std::memory_order_relaxed);
            } else {
                g_alloc_fail.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            std::uniform_int_distribution<size_t> idx_dist(0, ptrs.size() - 1);
            const size_t idx = idx_dist(rng);
            cudaError_t err = cudaFree(ptrs[idx]);
            if (err == cudaSuccess) {
                g_free_ok.fetch_add(1, std::memory_order_relaxed);
            } else {
                g_free_errors.fetch_add(1, std::memory_order_relaxed);
            }
            ptrs[idx] = ptrs.back();
            ptrs.pop_back();
        }
    }

    for (void* p : ptrs) {
        cudaError_t err = cudaFree(p);
        if (err == cudaSuccess) {
            g_free_ok.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_free_errors.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

int main() {
    int device_count = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&device_count);
    if (dev_err != cudaSuccess || device_count == 0) {
        std::printf("intercept stress test: SKIP (no CUDA device)\n");
        return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        std::printf("intercept stress test: SKIP (cudaSetDevice failed)\n");
        return 0;
    }

    const auto t0 = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back(stress_worker, t);
    }
    for (auto& th : threads) th.join();

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

    const int allocs = g_alloc_ok.load();
    const int alloc_fails = g_alloc_fail.load();
    const int frees = g_free_ok.load();
    const int free_errors = g_free_errors.load();

    std::printf("intercept stress: %d threads, %d allocs, %d alloc_denied, %d frees, %d free_errors, %lld ms\n",
                kNumThreads, allocs, alloc_fails, frees, free_errors, static_cast<long long>(elapsed_ms));

    if (allocs == 0) {
        std::fprintf(stderr, "FAIL: no successful allocations\n");
        return 2;
    }

    const int total_frees = frees + free_errors;
    const double fail_rate = total_frees > 0
        ? static_cast<double>(free_errors) / total_frees : 0.0;
    if (fail_rate > 0.25) {
        std::fprintf(stderr, "FAIL: free error rate %.1f%% exceeds 25%% threshold\n",
                     fail_rate * 100.0);
        return 1;
    }

    std::printf("intercept stress test: PASS\n");
    return 0;
}
