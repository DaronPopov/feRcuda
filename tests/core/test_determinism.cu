/*
 * test_determinism.cu
 *
 * Runs the same ops 100 times and verifies every output is bitwise identical.
 * If any result differs: the rounding discipline broke somewhere.
 */

#include "fercuda/compute/types.cuh"
#include "fercuda/compute/ops.cuh"
#include "fercuda/alloc/memory.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

using namespace fer;

static void check(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); exit(1); }
}

static bool vectors_equal(const float* a, const float* b, size_t n) {
    return memcmp(a, b, n * sizeof(float)) == 0;
}

int main() {
    constexpr int N    = 1024;
    constexpr int RUNS = 100;
    constexpr int M    = 64, K = 64;

    ElasticPool pool(0, {.mutable_bytes = 64ULL<<20, .immutable_bytes = 64ULL<<20});

    // ── Setup input data ──────────────────────────────────────────────────────
    std::vector<float> h_a(N), h_b(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(i % 97) * 0.01234f - 0.6f;
        h_b[i] = (float)(i % 53) * 0.00791f + 0.3f;
    }

    auto d_a   = pool.alloc_mutable<F32, 1>(Shape<1>(N));
    auto d_b   = pool.alloc_mutable<F32, 1>(Shape<1>(N));
    auto d_out = pool.alloc_mutable<F32, 1>(Shape<1>(N));

    check(cudaMemcpy(d_a.data, h_a.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_b.data, h_b.data(), N*sizeof(float), cudaMemcpyHostToDevice));

    // ── Elementwise add: run once to get reference ────────────────────────────
    ops::add(d_a, d_b, d_out);
    check(cudaDeviceSynchronize());
    std::vector<float> ref(N), cur(N);
    check(cudaMemcpy(ref.data(), d_out.data, N*sizeof(float), cudaMemcpyDeviceToHost));

    printf("[add] Reference computed. Running %d repetitions...\n", RUNS);
    bool add_ok = true;
    for (int r = 0; r < RUNS; r++) {
        ops::add(d_a, d_b, d_out);
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(cur.data(), d_out.data, N*sizeof(float), cudaMemcpyDeviceToHost));
        if (!vectors_equal(ref.data(), cur.data(), N)) {
            printf("  FAIL at run %d\n", r);
            add_ok = false;
            break;
        }
    }
    printf("[add] %s\n", add_ok ? "PASS (bitwise identical x100)" : "FAIL");

    // ── GELU ─────────────────────────────────────────────────────────────────
    ops::gelu(d_a, d_out);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(ref.data(), d_out.data, N*sizeof(float), cudaMemcpyDeviceToHost));

    printf("[gelu] Running %d repetitions...\n", RUNS);
    bool gelu_ok = true;
    for (int r = 0; r < RUNS; r++) {
        ops::gelu(d_a, d_out);
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(cur.data(), d_out.data, N*sizeof(float), cudaMemcpyDeviceToHost));
        if (!vectors_equal(ref.data(), cur.data(), N)) {
            printf("  FAIL at run %d\n", r);
            gelu_ok = false;
            break;
        }
    }
    printf("[gelu] %s\n", gelu_ok ? "PASS (bitwise identical x100)" : "FAIL");

    // ── RMS Norm ──────────────────────────────────────────────────────────────
    ops::rms_norm(d_a, d_out);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(ref.data(), d_out.data, N*sizeof(float), cudaMemcpyDeviceToHost));

    printf("[rms_norm] Running %d repetitions...\n", RUNS);
    bool norm_ok = true;
    for (int r = 0; r < RUNS; r++) {
        ops::rms_norm(d_a, d_out);
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(cur.data(), d_out.data, N*sizeof(float), cudaMemcpyDeviceToHost));
        if (!vectors_equal(ref.data(), cur.data(), N)) {
            printf("  FAIL at run %d\n", r);
            norm_ok = false;
            break;
        }
    }
    printf("[rms_norm] %s\n", norm_ok ? "PASS (bitwise identical x100)" : "FAIL");

    // ── MatMul ────────────────────────────────────────────────────────────────
    std::vector<float> h_mat_a(M*K), h_mat_b(K*M);
    for (int i = 0; i < M*K; i++) h_mat_a[i] = (float)(i%17) * 0.01f - 0.08f;
    for (int i = 0; i < K*M; i++) h_mat_b[i] = (float)(i%13) * 0.02f - 0.13f;

    auto d_ma  = pool.alloc_mutable<F32, 2>(Shape<2>(M, K));
    auto d_mb  = pool.alloc_mutable<F32, 2>(Shape<2>(K, M));
    auto d_mc  = pool.alloc_mutable<F32, 2>(Shape<2>(M, M));
    check(cudaMemcpy(d_ma.data, h_mat_a.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_mb.data, h_mat_b.data(), K*M*sizeof(float), cudaMemcpyHostToDevice));

    ops::matmul(d_ma, d_mb, d_mc);
    check(cudaDeviceSynchronize());
    std::vector<float> ref_mm(M*M), cur_mm(M*M);
    check(cudaMemcpy(ref_mm.data(), d_mc.data, M*M*sizeof(float), cudaMemcpyDeviceToHost));

    printf("[matmul %dx%d] Running %d repetitions...\n", M, M, RUNS);
    bool mm_ok = true;
    for (int r = 0; r < RUNS; r++) {
        ops::matmul(d_ma, d_mb, d_mc);
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(cur_mm.data(), d_mc.data, M*M*sizeof(float), cudaMemcpyDeviceToHost));
        if (!vectors_equal(ref_mm.data(), cur_mm.data(), M*M)) {
            printf("  FAIL at run %d\n", r);
            mm_ok = false;
            break;
        }
    }
    printf("[matmul] %s\n", mm_ok ? "PASS (bitwise identical x100)" : "FAIL");

    bool all_pass = add_ok && gelu_ok && norm_ok && mm_ok;
    printf("\n%s\n", all_pass ? "ALL DETERMINISM TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
