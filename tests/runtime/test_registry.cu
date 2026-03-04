/*
 * test_registry.cu
 *
 * Verifies typed op attachments:
 *  - matmul validate/launch
 *  - layer_norm validate/launch
 */

#include "fercuda/algorithms/matmul_attachment.cuh"
#include "fercuda/alloc/memory.cuh"
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace fer;

static void check(cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

static bool nearly_equal(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    ElasticPool pool(0, {.mutable_bytes = 32ULL << 20, .immutable_bytes = 32ULL << 20});

    auto a = pool.alloc_mutable<F32, 2>(Shape<2>(2, 3));
    auto b = pool.alloc_mutable<F32, 2>(Shape<2>(3, 2));
    auto out = pool.alloc_mutable<F32, 2>(Shape<2>(2, 2));

    std::vector<float> h_a = {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f
    };
    std::vector<float> h_b = {
         7.f,  8.f,
         9.f, 10.f,
        11.f, 12.f
    };
    std::vector<float> h_out(4, 0.f);

    check(cudaMemcpy(a.data, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(b.data, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto registry = algorithms::make_default_registry();
    OpContext ctx{DeviceId(0), StreamHandle(0)};

    algorithms::MatmulSpec good{a, b, out};
    Status vgood = registry.validate(OpTag::MATMUL, good);

    auto bad_out = pool.alloc_mutable<F32, 2>(Shape<2>(3, 1));
    algorithms::MatmulSpec bad{a, b, bad_out};
    Status vbad = registry.validate(OpTag::MATMUL, bad);

    Status launch_status = registry.launch(OpTag::MATMUL, good, ctx);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(h_out.data(), out.data, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const float ref[4] = {58.f, 64.f, 139.f, 154.f};
    bool matmul_ok = true;
    for (int i = 0; i < 4; i++) {
        if (!nearly_equal(h_out[i], ref[i])) {
            matmul_ok = false;
            break;
        }
    }

    bool all = true;

    bool t1 = vgood.ok();
    printf("[%s] Registry validate accepts valid MatmulSpec\n", t1 ? "PASS" : "FAIL");
    all = all && t1;

    bool t2 = (!vbad.ok() && vbad.code == StatusCode::INVALID_ARGUMENT);
    printf("[%s] Registry validate rejects invalid output shape\n", t2 ? "PASS" : "FAIL");
    all = all && t2;

    bool t3 = launch_status.ok();
    printf("[%s] Registry launch returned OK\n", t3 ? "PASS" : "FAIL");
    all = all && t3;

    bool t4 = matmul_ok;
    printf("[%s] Registry-dispatched matmul output matches reference\n", t4 ? "PASS" : "FAIL");
    all = all && t4;

    auto lx = pool.alloc_mutable<F32, 1>(Shape<1>(4));
    auto lout = pool.alloc_mutable<F32, 1>(Shape<1>(4));
    auto lbad_out = pool.alloc_mutable<F32, 1>(Shape<1>(3));
    std::vector<float> h_lx = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> h_lout(4, 0.f);
    check(cudaMemcpy(lx.data, h_lx.data(), h_lx.size() * sizeof(float), cudaMemcpyHostToDevice));

    algorithms::LayerNormSpec lgood{lx, lout, 1e-6f};
    algorithms::LayerNormSpec lbad{lx, lbad_out, 1e-6f};

    Status lv_good = registry.validate(OpTag::LAYER_NORM, lgood);
    bool t5 = lv_good.ok();
    printf("[%s] Registry validate accepts valid LayerNormSpec\n", t5 ? "PASS" : "FAIL");
    all = all && t5;

    Status lv_bad = registry.validate(OpTag::LAYER_NORM, lbad);
    bool t6 = (!lv_bad.ok() && lv_bad.code == StatusCode::INVALID_ARGUMENT);
    printf("[%s] Registry validate rejects invalid LayerNormSpec\n", t6 ? "PASS" : "FAIL");
    all = all && t6;

    Status llaunch = registry.launch(OpTag::LAYER_NORM, lgood, ctx);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(h_lout.data(), lout.data, h_lout.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float mean = 0.f;
    for (float v : h_lx) mean += v;
    mean /= 4.f;
    float var = 0.f;
    for (float v : h_lx) {
        float d = v - mean;
        var += d * d;
    }
    var /= 4.f;
    float inv_std = 1.f / std::sqrt(var + 1e-6f);
    bool lnorm_ok = true;
    for (int i = 0; i < 4; i++) {
        float ref_v = (h_lx[i] - mean) * inv_std;
        if (!nearly_equal(h_lout[i], ref_v, 1e-5f)) {
            lnorm_ok = false;
            break;
        }
    }

    bool t7 = llaunch.ok();
    printf("[%s] Registry launch returned OK for LayerNorm\n", t7 ? "PASS" : "FAIL");
    all = all && t7;

    bool t8 = lnorm_ok;
    printf("[%s] Registry-dispatched layer_norm output matches reference\n", t8 ? "PASS" : "FAIL");
    all = all && t8;

    printf("\n%s\n", all ? "ALL REGISTRY TESTS PASSED" : "SOME REGISTRY TESTS FAILED");
    return all ? 0 : 1;
}
