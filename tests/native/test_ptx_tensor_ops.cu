#include "gpu/gpu_hot_runtime.h"
#include "gpu/tensor_ops.h"

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("test_ptx_tensor_ops: SKIP (no CUDA device)\n");
        return 0;
    }

    GPUHotConfig cfg = gpu_hot_default_config();
    cfg.fixed_pool_size = 64ull * 1024ull * 1024ull;
    cfg.quiet_init = true;

    GPUHotRuntime* rt = gpu_hot_init_with_config(0, "test_tensor", &cfg);
    if (!rt) return fail("gpu_hot_init failed");

    constexpr int N = 1024;
    const size_t bytes = N * sizeof(float);

    float* d_a = static_cast<float*>(gpu_hot_alloc(rt, bytes));
    float* d_b = static_cast<float*>(gpu_hot_alloc(rt, bytes));
    float* d_c = static_cast<float*>(gpu_hot_alloc(rt, bytes));
    if (!d_a || !d_b || !d_c) return fail("gpu_hot_alloc for tensors failed");

    std::vector<float> h_a(N), h_b(N), h_c(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    cudaStream_t stream = gpu_hot_get_stream(rt, 0);

    // Test elementwise add
    ptx_tensor_add_f32(d_a, d_b, d_c, N, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-5f)
            return fail("add result mismatch");
    }

    // Test ReLU activation
    std::vector<float> h_neg(N);
    for (int i = 0; i < N; ++i) h_neg[i] = static_cast<float>(i) - 512.0f;
    cudaMemcpy(d_a, h_neg.data(), bytes, cudaMemcpyHostToDevice);

    ptx_tensor_relu_f32(d_a, d_c, N, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        float expected = h_neg[i] > 0.0f ? h_neg[i] : 0.0f;
        if (std::fabs(h_c[i] - expected) > 1e-5f)
            return fail("relu result mismatch");
    }

    // Test reduce sum (outer=1, reduce=N, inner=1)
    for (int i = 0; i < N; ++i) h_a[i] = 1.0f;
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);

    ptx_tensor_reduce_sum_f32(d_a, d_c, 1, N, 1, stream);
    cudaStreamSynchronize(stream);

    float reduce_result = 0.0f;
    cudaMemcpy(&reduce_result, d_c, sizeof(float), cudaMemcpyDeviceToHost);
    if (std::fabs(reduce_result - static_cast<float>(N)) > 1.0f)
        return fail("reduce_sum result mismatch");

    // Test softmax
    for (int i = 0; i < N; ++i) h_a[i] = static_cast<float>(i % 10);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);

    ptx_tensor_softmax_f32(d_a, d_c, 1, N, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    float sm_sum = 0.0f;
    for (int i = 0; i < N; ++i) sm_sum += h_c[i];
    if (std::fabs(sm_sum - 1.0f) > 0.01f)
        return fail("softmax doesn't sum to 1");

    // Test affine: out = mul * in + add
    for (int i = 0; i < N; ++i) h_a[i] = static_cast<float>(i);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);

    ptx_tensor_affine_f32(d_a, d_c, N, 2.0f, 1.0f, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        float expected = 2.0f * static_cast<float>(i) + 1.0f;
        if (std::fabs(h_c[i] - expected) > 1e-4f)
            return fail("affine result mismatch");
    }

    gpu_hot_free(rt, d_a);
    gpu_hot_free(rt, d_b);
    gpu_hot_free(rt, d_c);
    gpu_hot_shutdown(rt);

    std::printf("PTX TENSOR OPS TEST PASSED\n");
    return 0;
}
