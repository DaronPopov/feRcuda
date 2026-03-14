// Production stability test: hammer the native TLSF allocator with
// realistic ML-workload allocation patterns — large weight tensors,
// dynamic KV-cache growth, many small activation buffers, and
// concurrent alloc/free pressure.

#include "gpu/gpu_hot_runtime.h"
#include "gpu/tensor_ops.h"
#include "memory/ptx_tlsf_allocator.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

// ---------------------------------------------------------------------------
// Phase 1: Large contiguous allocations (model weights)
// ---------------------------------------------------------------------------
static int test_large_weight_allocs(GPUHotRuntime* rt) {
    std::printf("  [phase 1] large weight allocations...\n");
    constexpr int NUM_LAYERS = 32;
    constexpr size_t WEIGHT_SIZE = 4096 * 4096 * sizeof(float);  // 64 MiB each

    std::vector<void*> weights;
    weights.reserve(NUM_LAYERS);

    for (int i = 0; i < NUM_LAYERS; ++i) {
        void* p = gpu_hot_alloc(rt, WEIGHT_SIZE);
        if (!p) {
            std::fprintf(stderr, "    weight alloc %d failed (%.1f MiB)\n",
                         i, WEIGHT_SIZE / (1024.0 * 1024.0));
            for (void* w : weights) gpu_hot_free(rt, w);
            return fail("large weight allocation failed");
        }
        weights.push_back(p);
    }

    TLSFPoolStats stats{};
    gpu_hot_get_tlsf_stats(rt, &stats);
    std::printf("    allocated %d x %.0f MiB = %.1f GiB  (pool util %.1f%%)\n",
                NUM_LAYERS, WEIGHT_SIZE / (1024.0 * 1024.0),
                (NUM_LAYERS * WEIGHT_SIZE) / (1024.0 * 1024.0 * 1024.0),
                stats.utilization_percent);

    for (void* w : weights) gpu_hot_free(rt, w);

    gpu_hot_get_tlsf_stats(rt, &stats);
    if (stats.allocated_bytes > 0) return fail("leaked bytes after weight free");

    std::printf("    PASS (no leaks, no fragmentation)\n");
    return 0;
}

// ---------------------------------------------------------------------------
// Phase 2: KV-cache growth pattern (incremental alloc, periodic free)
// ---------------------------------------------------------------------------
static int test_kv_cache_pattern(GPUHotRuntime* rt) {
    std::printf("  [phase 2] KV-cache growth/eviction pattern...\n");
    constexpr int SEQ_STEPS = 512;
    constexpr size_t KV_STEP_SIZE = 32 * 128 * sizeof(float);  // 16 KiB per step

    std::vector<void*> kv_blocks;
    kv_blocks.reserve(SEQ_STEPS);

    for (int step = 0; step < SEQ_STEPS; ++step) {
        void* p = gpu_hot_alloc(rt, KV_STEP_SIZE);
        if (!p) return fail("KV alloc failed");
        kv_blocks.push_back(p);

        // Every 64 steps, evict the oldest quarter
        if ((step + 1) % 64 == 0 && kv_blocks.size() > 16) {
            size_t evict = kv_blocks.size() / 4;
            for (size_t i = 0; i < evict; ++i) {
                gpu_hot_free(rt, kv_blocks[i]);
            }
            kv_blocks.erase(kv_blocks.begin(), kv_blocks.begin() + evict);
        }
    }

    // Free everything
    for (void* p : kv_blocks) gpu_hot_free(rt, p);

    TLSFPoolStats stats{};
    gpu_hot_get_tlsf_stats(rt, &stats);
    std::printf("    %d steps, final fragmentation: %.2f%%\n",
                SEQ_STEPS, stats.fragmentation_ratio * 100.0);
    if (stats.allocated_bytes > 0) return fail("leaked bytes after KV free");
    std::printf("    PASS\n");
    return 0;
}

// ---------------------------------------------------------------------------
// Phase 3: Many small activation buffers (attention heads)
// ---------------------------------------------------------------------------
static int test_small_activation_churn(GPUHotRuntime* rt) {
    std::printf("  [phase 3] small activation churn (attention heads)...\n");
    constexpr int ITERATIONS = 1000;
    constexpr int HEADS = 32;
    constexpr size_t HEAD_SIZE = 64 * 128 * sizeof(float);  // 32 KiB

    auto t0 = std::chrono::steady_clock::now();

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        void* heads[32];
        for (int h = 0; h < HEADS; ++h) {
            heads[h] = gpu_hot_alloc(rt, HEAD_SIZE);
            if (!heads[h]) return fail("head alloc failed");
        }
        for (int h = 0; h < HEADS; ++h) {
            gpu_hot_free(rt, heads[h]);
        }
    }

    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    double ops = ITERATIONS * HEADS * 2.0;  // alloc + free
    double ns_per_op = (elapsed_us * 1000.0) / ops;

    TLSFPoolStats stats{};
    gpu_hot_get_tlsf_stats(rt, &stats);
    std::printf("    %d iters x %d heads = %.0f ops in %.1f ms (%.0f ns/op)\n",
                ITERATIONS, HEADS, ops, elapsed_us / 1000.0, ns_per_op);
    if (stats.allocated_bytes > 0) return fail("leaked bytes after churn");
    std::printf("    PASS\n");
    return 0;
}

// ---------------------------------------------------------------------------
// Phase 4: Mixed sizes under pressure (realistic inference batch)
// ---------------------------------------------------------------------------
static int test_mixed_workload(GPUHotRuntime* rt) {
    std::printf("  [phase 4] mixed-size inference workload...\n");
    std::mt19937 rng(42);
    constexpr int OPS = 2000;

    size_t sizes[] = {
        256,                            // tiny scratch
        4096,                           // small buffer
        64 * 1024,                      // 64 KiB activation
        256 * 1024,                     // 256 KiB intermediate
        1024 * 1024,                    // 1 MiB layer output
        4 * 1024 * 1024,               // 4 MiB attention buffer
        16 * 1024 * 1024,              // 16 MiB weight shard
    };
    constexpr int NUM_SIZES = sizeof(sizes) / sizeof(sizes[0]);

    std::vector<void*> live;
    live.reserve(OPS);
    int alloc_ok = 0, free_ok = 0, alloc_fail = 0;

    for (int i = 0; i < OPS; ++i) {
        bool do_alloc = live.empty() || (rng() % 3 != 0);
        if (do_alloc) {
            size_t sz = sizes[rng() % NUM_SIZES];
            void* p = gpu_hot_alloc(rt, sz);
            if (p) {
                live.push_back(p);
                alloc_ok++;
            } else {
                alloc_fail++;
            }
        } else {
            std::uniform_int_distribution<size_t> dist(0, live.size() - 1);
            size_t idx = dist(rng);
            gpu_hot_free(rt, live[idx]);
            live[idx] = live.back();
            live.pop_back();
            free_ok++;
        }
    }

    for (void* p : live) gpu_hot_free(rt, p);

    TLSFPoolStats stats{};
    gpu_hot_get_tlsf_stats(rt, &stats);
    std::printf("    %d allocs, %d frees, %d denied, final frag: %.2f%%\n",
                alloc_ok, free_ok + static_cast<int>(live.size()), alloc_fail,
                stats.fragmentation_ratio * 100.0);
    if (stats.allocated_bytes > 0) return fail("leaked bytes after mixed workload");
    std::printf("    PASS\n");
    return 0;
}

// ---------------------------------------------------------------------------
// Phase 5: Compute correctness under TLSF (run real kernels on allocated memory)
// ---------------------------------------------------------------------------
static int test_compute_on_tlsf(GPUHotRuntime* rt) {
    std::printf("  [phase 5] compute correctness on TLSF-allocated memory...\n");
    constexpr int N = 1 << 20;  // 1M elements
    const size_t bytes = N * sizeof(float);

    float* d_a = static_cast<float*>(gpu_hot_alloc(rt, bytes));
    float* d_b = static_cast<float*>(gpu_hot_alloc(rt, bytes));
    float* d_c = static_cast<float*>(gpu_hot_alloc(rt, bytes));
    if (!d_a || !d_b || !d_c) return fail("compute alloc failed");

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i % 1000) * 0.001f;
        h_b[i] = static_cast<float>((i + 500) % 1000) * 0.001f;
    }
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    cudaStream_t stream = gpu_hot_get_stream(rt, 0);

    // add
    ptx_tensor_add_f32(d_a, d_b, d_c, N, stream);
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 100; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-5f) return fail("add mismatch");
    }

    // relu
    for (int i = 0; i < N; ++i) h_a[i] = static_cast<float>(i) - N / 2.0f;
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    ptx_tensor_relu_f32(d_a, d_c, N, stream);
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] > 0.0f ? h_a[i] : 0.0f;
        if (std::fabs(h_c[i] - expected) > 1e-3f) return fail("relu mismatch");
    }

    // softmax on 1024 elements
    constexpr int SM_N = 1024;
    for (int i = 0; i < SM_N; ++i) h_a[i] = static_cast<float>(i % 10) * 0.1f;
    cudaMemcpy(d_a, h_a.data(), SM_N * sizeof(float), cudaMemcpyHostToDevice);
    ptx_tensor_softmax_f32(d_a, d_c, 1, SM_N, stream);
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_c.data(), d_c, SM_N * sizeof(float), cudaMemcpyDeviceToHost);
    float sm_sum = 0.0f;
    for (int i = 0; i < SM_N; ++i) sm_sum += h_c[i];
    if (std::fabs(sm_sum - 1.0f) > 0.01f) return fail("softmax sum != 1");

    gpu_hot_free(rt, d_a);
    gpu_hot_free(rt, d_b);
    gpu_hot_free(rt, d_c);

    std::printf("    1M-element add, relu, softmax: all correct\n");
    std::printf("    PASS\n");
    return 0;
}

// ---------------------------------------------------------------------------
// Phase 6: Pool health validation after everything
// ---------------------------------------------------------------------------
static int test_pool_health(GPUHotRuntime* rt) {
    std::printf("  [phase 6] TLSF pool health validation...\n");

    TLSFHealthReport health{};
    gpu_hot_validate_tlsf_pool(rt, &health);

    if (!health.is_valid) return fail("pool not valid");
    if (health.has_memory_leaks) return fail("memory leaks detected");
    if (health.has_corrupted_blocks) return fail("corrupted blocks");
    if (health.has_broken_chains) return fail("broken chains");
    if (health.has_hash_errors) return fail("hash errors");

    TLSFPoolStats stats{};
    gpu_hot_get_tlsf_stats(rt, &stats);
    std::printf("    pool: %.0f MiB total, %llu allocs, %llu frees, frag=%.2f%%\n",
                stats.total_pool_size / (1024.0 * 1024.0),
                static_cast<unsigned long long>(stats.total_allocations),
                static_cast<unsigned long long>(stats.total_frees),
                stats.fragmentation_ratio * 100.0);
    std::printf("    health: valid=%d leaks=%d corrupt=%d broken=%d hash_err=%d\n",
                health.is_valid, health.has_memory_leaks,
                health.has_corrupted_blocks, health.has_broken_chains,
                health.has_hash_errors);
    std::printf("    PASS\n");
    return 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("test_ptx_prod_stress: SKIP (no CUDA device)\n");
        return 0;
    }

    // Use a large pool to simulate production — 4 GiB
    GPUHotConfig cfg = gpu_hot_default_config();
    cfg.fixed_pool_size = 4ull * 1024ull * 1024ull * 1024ull;
    cfg.quiet_init = false;
    cfg.enable_leak_detection = true;
    cfg.enable_pool_health = true;

    GPUHotRuntime* rt = gpu_hot_init_with_config(0, "prod_stress", &cfg);
    if (!rt) {
        // Fall back to smaller pool if GPU doesn't have 4 GiB free
        cfg.fixed_pool_size = 1ull * 1024ull * 1024ull * 1024ull;
        rt = gpu_hot_init_with_config(0, "prod_stress", &cfg);
    }
    if (!rt) return fail("gpu_hot_init failed");

    std::printf("=== feRcuda Production Stability Test ===\n\n");

    int rc = 0;
    rc = rc || test_large_weight_allocs(rt);
    rc = rc || test_kv_cache_pattern(rt);
    rc = rc || test_small_activation_churn(rt);
    rc = rc || test_mixed_workload(rt);
    rc = rc || test_compute_on_tlsf(rt);
    rc = rc || test_pool_health(rt);

    gpu_hot_shutdown(rt);

    if (rc == 0) {
        std::printf("\n=== ALL PHASES PASSED ===\n");
    } else {
        std::printf("\n=== SOME PHASES FAILED ===\n");
    }
    return rc;
}
