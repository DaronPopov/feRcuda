#include "fercuda/compute/ops.cuh"
#include "fercuda/compute/types.cuh"
#include <cuda_runtime.h>

namespace fer::ops {

static constexpr int TILE = 16;

// ─── Tiled GEMM ───────────────────────────────────────────────────────────────
//
// out[M,N] = A[M,K] @ B[K,N]
//
// Accumulator uses __fmaf_rn (FMA with locked rounding) so every partial sum
// is bit-reproducible regardless of grid/block configuration.
//
__global__ void k_matmul(const F32* __restrict__ A,
                         const F32* __restrict__ B,
                         F32*       __restrict__ C,
                         uint32_t M, uint32_t K, uint32_t N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    uint32_t row = blockIdx.y * TILE + threadIdx.y;
    uint32_t col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.f;  // accumulator in registers

    for (uint32_t t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint32_t ak = t * TILE + threadIdx.x;
        uint32_t bk = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak].v : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col].v : 0.f;
        __syncthreads();

        // Deterministic accumulation: strict order, locked FMA
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc = __fmaf_rn(As[threadIdx.y][k], Bs[k][threadIdx.x], acc);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = F32(acc);
    }
}

void matmul(FTensor2D a, FTensor2D b, FTensor2D out, cudaStream_t s) {
    uint32_t M = a.shape[0], K = a.shape[1];
    uint32_t K2 = b.shape[0], N = b.shape[1];
    // K must match
    (void)K2;

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    k_matmul<<<grid, block, 0, s>>>(a.data, b.data, out.data, M, K, N);
}

} // namespace fer::ops
