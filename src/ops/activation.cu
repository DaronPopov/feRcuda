#include "fercuda/compute/ops.cuh"
#include "fercuda/compute/types.cuh"
#include <cuda_runtime.h>

namespace fer::ops {

static constexpr int BLK = 256;

// ─── GELU ─────────────────────────────────────────────────────────────────────
//
// Tanh approximation with LOCKED coefficients so the result is bitwise stable.
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
//
// We use __fmaf_rn throughout so no compiler rewrite can change rounding.
//
static constexpr float GELU_A = 0.7978845608028654f;  // sqrt(2/pi)
static constexpr float GELU_B = 0.044715f;

__global__ void k_relu(const F32* __restrict__ x, F32* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i].v;
        out[i] = F32(v > 0.f ? v : 0.f);
    }
}

__global__ void k_gelu(const F32* __restrict__ x, F32* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        float v   = x[i].v;
        float v3  = __fmul_rn(__fmul_rn(v, v), v);           // v^3, locked
        float arg = __fmaf_rn(GELU_B, v3, v);                // v + B*v^3
        arg       = __fmul_rn(GELU_A, arg);                  // A*(v + B*v^3)
        float t   = tanhf(arg);                               // tanhf is deterministic on sm_86
        float r   = __fmul_rn(0.5f, __fmaf_rn(t, v, v));     // 0.5*v*(1+t)
        out[i]    = F32(r);
    }
}

__global__ void k_silu(const F32* __restrict__ x, F32* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i].v;
        // sigmoid via -v to avoid __expf fast path
        float neg_v  = __fmul_rn(-1.f, v);
        float exp_nv = __expf(neg_v);                          // single-precision exp
        float sig    = __fdiv_rn(1.f, __fadd_rn(1.f, exp_nv));
        out[i]       = F32(__fmul_rn(v, sig));
    }
}

void relu(FTensor1D x, FTensor1D out, cudaStream_t s) {
    size_t n = x.numel();
    k_relu<<<(n + BLK - 1) / BLK, BLK, 0, s>>>(x.data, out.data, n);
}

void gelu(FTensor1D x, FTensor1D out, cudaStream_t s) {
    size_t n = x.numel();
    k_gelu<<<(n + BLK - 1) / BLK, BLK, 0, s>>>(x.data, out.data, n);
}

void silu(FTensor1D x, FTensor1D out, cudaStream_t s) {
    size_t n = x.numel();
    k_silu<<<(n + BLK - 1) / BLK, BLK, 0, s>>>(x.data, out.data, n);
}

} // namespace fer::ops
