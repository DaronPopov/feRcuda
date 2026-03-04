#include "fercuda/compute/ops.cuh"
#include "fercuda/compute/types.cuh"
#include <cuda_runtime.h>

namespace fer::ops {

static constexpr int BLK = 256;

// ─── Kernels ──────────────────────────────────────────────────────────────────

__global__ void k_add(const F32* __restrict__ a, const F32* __restrict__ b,
                      F32* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];   // DFloat::operator+ → __fadd_rn
}

__global__ void k_sub(const F32* __restrict__ a, const F32* __restrict__ b,
                      F32* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

__global__ void k_mul(const F32* __restrict__ a, const F32* __restrict__ b,
                      F32* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];   // DFloat::operator* → __fmul_rn
}

__global__ void k_add_scalar(const F32* __restrict__ a, F32 scalar,
                              F32* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + scalar;
}

__global__ void k_mul_scalar(const F32* __restrict__ a, F32 scalar,
                              F32* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * scalar;
}

// ─── Launch Wrappers ──────────────────────────────────────────────────────────

void add(FTensor1D a, FTensor1D b, FTensor1D out, cudaStream_t s) {
    size_t n = a.numel();
    k_add<<<(n + BLK - 1) / BLK, BLK, 0, s>>>(a.data, b.data, out.data, n);
}

void sub(FTensor1D a, FTensor1D b, FTensor1D out, cudaStream_t s) {
    size_t n = a.numel();
    k_sub<<<(n + BLK - 1) / BLK, BLK, 0, s>>>(a.data, b.data, out.data, n);
}

void mul(FTensor1D a, FTensor1D b, FTensor1D out, cudaStream_t s) {
    size_t n = a.numel();
    k_mul<<<(n + BLK - 1) / BLK, BLK, 0, s>>>(a.data, b.data, out.data, n);
}

void add_scalar(FTensor1D a, float scalar, FTensor1D out, cudaStream_t s) {
    size_t n = a.numel();
    k_add_scalar<<<(n + BLK - 1) / BLK, BLK, 0, s>>>(
        a.data, F32(scalar), out.data, n);
}

void mul_scalar(FTensor1D a, float scalar, FTensor1D out, cudaStream_t s) {
    size_t n = a.numel();
    k_mul_scalar<<<(n + BLK - 1) / BLK, BLK, 0, s>>>(
        a.data, F32(scalar), out.data, n);
}

} // namespace fer::ops
