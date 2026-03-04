#pragma once
/*
 * feRcuda :: ops.cuh
 *
 * Public op launch interfaces. All ops operate on typed Tensors and use
 * DFloat<NEAREST> arithmetic — no implicit float promotion.
 *
 * Every function takes an explicit cudaStream_t so callers control ordering.
 */

#include <cuda_runtime.h>
#include "fercuda/compute/types.cuh"

namespace fer::ops {

// ─── Elementwise ──────────────────────────────────────────────────────────────

// out = a + b  (element-wise, deterministic)
void add(FTensor1D a, FTensor1D b, FTensor1D out, cudaStream_t s = 0);
void sub(FTensor1D a, FTensor1D b, FTensor1D out, cudaStream_t s = 0);
void mul(FTensor1D a, FTensor1D b, FTensor1D out, cudaStream_t s = 0);

// out = a + scalar
void add_scalar(FTensor1D a, float scalar, FTensor1D out, cudaStream_t s = 0);
void mul_scalar(FTensor1D a, float scalar, FTensor1D out, cudaStream_t s = 0);

// ─── Matrix Multiply ──────────────────────────────────────────────────────────

// out[M,N] = a[M,K] @ b[K,N]
// Uses tiled shared-memory GEMM with __fmaf_rn accumulators.
// Block tile = 16x16. M, N, K need not be multiples of 16 (padded internally).
void matmul(FTensor2D a, FTensor2D b, FTensor2D out, cudaStream_t s = 0);

// ─── Normalization ────────────────────────────────────────────────────────────

// RMS norm: out[i] = x[i] / sqrt(mean(x^2) + eps)
void rms_norm(FTensor1D x, FTensor1D out, float eps = 1e-6f, cudaStream_t s = 0);

// Layer norm: out[i] = (x[i] - mean) / std
void layer_norm(FTensor1D x, FTensor1D out, float eps = 1e-6f, cudaStream_t s = 0);

// ─── Activations ──────────────────────────────────────────────────────────────

// All use deterministic polynomial/intrinsic approximations (no fast-math tanhf)
void relu (FTensor1D x, FTensor1D out, cudaStream_t s = 0);
void gelu (FTensor1D x, FTensor1D out, cudaStream_t s = 0);  // tanh approx, locked coefficients
void silu (FTensor1D x, FTensor1D out, cudaStream_t s = 0);  // x * sigmoid(x), no __expf

} // namespace fer::ops
