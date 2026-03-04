/*
 * test_types.cu — Compile-time and runtime checks for DFloat and Tensor.
 */

#include "fercuda/compute/types.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace fer;

// ── Device kernel: verify DFloat ops produce expected IEEE results ────────────

__global__ void k_dfloat_checks(bool* results) {
    F32 a(1.0f), b(3.0f);

    // 1.0 + 3.0 = 4.0 exactly in IEEE 754
    results[0] = ((a + b).v == 4.0f);

    // 2.0 * 3.0 = 6.0 exactly
    F32 c(2.0f);
    results[1] = ((c * b).v == 6.0f);

    // FMA: 2*3+1 = 7.0 exactly
    results[2] = (F32::fma(c, b, a).v == 7.0f);

    // Different rounding modes on a non-exact value
    // 1.0f / 3.0f — check rn vs rz differ (or at least both finite)
    DFloat<Round::NEAREST> rn_result(1.0f / 3.0f);
    DFloat<Round::ZERO>    rz_result(1.0f / 3.0f);
    results[3] = (rn_result.v > 0.f && rz_result.v > 0.f);

    // Tensor shape numel
    Shape<2> sh(4, 8);
    results[4] = (sh.numel() == 32);
}

static void check(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); exit(1); }
}

int main() {
    // Static checks
    static_assert(sizeof(F32) == sizeof(float), "DFloat must be same size as float");
    static_assert(sizeof(Tensor<F32, 2>) > 0, "Tensor must be instantiable");

    bool* d_res;
    check(cudaMalloc(&d_res, 8 * sizeof(bool)));

    k_dfloat_checks<<<1, 1>>>(d_res);
    check(cudaDeviceSynchronize());

    bool h_res[8] = {};
    check(cudaMemcpy(h_res, d_res, 8*sizeof(bool), cudaMemcpyDeviceToHost));
    cudaFree(d_res);

    const char* names[] = {
        "F32 add (1+3=4)",
        "F32 mul (2*3=6)",
        "F32 fma (2*3+1=7)",
        "RoundMode variants finite",
        "Shape<2> numel",
    };

    bool all = true;
    for (int i = 0; i < 5; i++) {
        printf("[%s] %s\n", h_res[i] ? "PASS" : "FAIL", names[i]);
        if (!h_res[i]) all = false;
    }

    printf("\n%s\n", all ? "ALL TYPE TESTS PASSED" : "SOME TESTS FAILED");
    return all ? 0 : 1;
}
