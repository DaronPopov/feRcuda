// Bridge: fer::ops → native ptx_tensor_* kernels + inline GEMM/norm

#include "fercuda/compute/ops.cuh"
#include "gpu/tensor_ops.h"

#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int TILE = 16;

__global__ void k_matmul_f32(const float* A, const float* B, float* C,
                             int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            acc = __fmaf_rn(As[threadIdx.y][k], Bs[k][threadIdx.x], acc);
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

__global__ void k_layer_norm_f32(const float* x, float* out, int n, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) sum += x[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / n;
    float var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float d = x[i] - mean;
        var_sum += d * d;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = (x[i] - mean) * inv_std;
}

} // namespace

namespace fer::ops {

static float* raw(F32* p) { return reinterpret_cast<float*>(p); }

void add(FTensor1D a, FTensor1D b, FTensor1D out, cudaStream_t s) {
    ptx_tensor_add_f32(raw(a.data), raw(b.data), raw(out.data), a.shape[0], s);
}

void sub(FTensor1D a, FTensor1D b, FTensor1D out, cudaStream_t s) {
    ptx_tensor_sub_f32(raw(a.data), raw(b.data), raw(out.data), a.shape[0], s);
}

void mul(FTensor1D a, FTensor1D b, FTensor1D out, cudaStream_t s) {
    ptx_tensor_mul_f32(raw(a.data), raw(b.data), raw(out.data), a.shape[0], s);
}

void add_scalar(FTensor1D a, float scalar, FTensor1D out, cudaStream_t s) {
    ptx_tensor_add_scalar_f32(raw(a.data), scalar, raw(out.data), a.shape[0], s);
}

void mul_scalar(FTensor1D a, float scalar, FTensor1D out, cudaStream_t s) {
    ptx_tensor_mul_scalar_f32(raw(a.data), scalar, raw(out.data), a.shape[0], s);
}

void matmul(FTensor2D a, FTensor2D b, FTensor2D out, cudaStream_t s) {
    int M = a.shape[0], K = a.shape[1], N = b.shape[1];
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    dim3 block(TILE, TILE);
    k_matmul_f32<<<grid, block, 0, s>>>(raw(a.data), raw(b.data), raw(out.data), M, K, N);
}

void rms_norm(FTensor1D x, FTensor1D out, float eps, cudaStream_t s) {
    int n = x.shape[0];
    int threads = (n < 256) ? n : 256;
    k_layer_norm_f32<<<1, threads, threads * sizeof(float), s>>>(raw(x.data), raw(out.data), n, eps);
}

void layer_norm(FTensor1D x, FTensor1D out, float eps, cudaStream_t s) {
    int n = x.shape[0];
    int threads = (n < 256) ? n : 256;
    k_layer_norm_f32<<<1, threads, threads * sizeof(float), s>>>(raw(x.data), raw(out.data), n, eps);
}

void relu(FTensor1D x, FTensor1D out, cudaStream_t s) {
    ptx_tensor_relu_f32(raw(x.data), raw(out.data), x.shape[0], s);
}

void gelu(FTensor1D x, FTensor1D out, cudaStream_t s) {
    ptx_tensor_gelu_f32(raw(x.data), raw(out.data), x.shape[0], s);
}

void silu(FTensor1D x, FTensor1D out, cudaStream_t s) {
    ptx_tensor_silu_f32(raw(x.data), raw(out.data), x.shape[0], s);
}

} // namespace fer::ops
