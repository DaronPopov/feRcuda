#include "fercuda/compute/ops.cuh"
#include "fercuda/compute/types.cuh"
#include <cuda_runtime.h>

namespace fer::ops {

// ─── Warp Reduction ───────────────────────────────────────────────────────────

__device__ float warp_sum(float v) {
    // Full warp shuffle reduction — deterministic (fixed lane order)
    for (int offset = 16; offset > 0; offset >>= 1)
        v = __fadd_rn(v, __shfl_down_sync(0xFFFFFFFF, v, offset));
    return v;
}

// ─── RMS Norm ─────────────────────────────────────────────────────────────────
//
// out[i] = x[i] / sqrt(mean(x^2) + eps)
//
// One CTA per norm call (N <= 1024 for now). Two passes to avoid
// non-deterministic parallel accumulation order.
//
__global__ void k_rms_norm(const F32* __restrict__ x, F32* __restrict__ out,
                           uint32_t n, float eps) {
    extern __shared__ float smem[];

    uint32_t tid = threadIdx.x;

    // Pass 1: accumulate x^2
    float sq = 0.f;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        float v = x[i].v;
        sq = __fadd_rn(sq, __fmul_rn(v, v));
    }
    smem[tid] = sq;
    __syncthreads();

    // Parallel reduction in shared memory (power-of-2 steps)
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = __fadd_rn(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    float mean_sq  = __fdiv_rn(smem[0], (float)n);
    float rms      = sqrtf(__fadd_rn(mean_sq, eps));
    float inv_rms  = __fdiv_rn(1.f, rms);

    // Pass 2: scale
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        out[i] = F32(__fmul_rn(x[i].v, inv_rms));
    }
}

// ─── Layer Norm ───────────────────────────────────────────────────────────────

__global__ void k_layer_norm(const F32* __restrict__ x, F32* __restrict__ out,
                             uint32_t n, float eps) {
    extern __shared__ float smem[];

    uint32_t tid = threadIdx.x;

    // Pass 1: mean
    float sum = 0.f;
    for (uint32_t i = tid; i < n; i += blockDim.x)
        sum = __fadd_rn(sum, x[i].v);
    smem[tid] = sum;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = __fadd_rn(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float mean = __fdiv_rn(smem[0], (float)n);
    __syncthreads();

    // Pass 2: variance
    float var = 0.f;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        float d = __fadd_rn(x[i].v, -mean);
        var = __fadd_rn(var, __fmul_rn(d, d));
    }
    smem[tid] = var;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = __fadd_rn(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float std_inv = __fdiv_rn(1.f, sqrtf(__fadd_rn(__fdiv_rn(smem[0], (float)n), eps)));

    // Pass 3: normalize
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        float d = __fadd_rn(x[i].v, -mean);
        out[i]  = F32(__fmul_rn(d, std_inv));
    }
}

// ─── Launch Wrappers ──────────────────────────────────────────────────────────

void rms_norm(FTensor1D x, FTensor1D out, float eps, cudaStream_t s) {
    uint32_t n    = (uint32_t)x.numel();
    int      blk  = (n < 256) ? 32 : 256;
    size_t   smem = blk * sizeof(float);
    k_rms_norm<<<1, blk, smem, s>>>(x.data, out.data, n, eps);
}

void layer_norm(FTensor1D x, FTensor1D out, float eps, cudaStream_t s) {
    uint32_t n    = (uint32_t)x.numel();
    int      blk  = (n < 256) ? 32 : 256;
    size_t   smem = blk * sizeof(float);
    k_layer_norm<<<1, blk, smem, s>>>(x.data, out.data, n, eps);
}

} // namespace fer::ops
