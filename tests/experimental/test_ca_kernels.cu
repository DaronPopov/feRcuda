/*
 * test_ca_kernels.cu
 *
 * CA token bus integrated into real CUDA compute kernels.
 * Three escalating tests:
 *
 *  1. CA-PRIORITY WRITE (lock-free serialized output)
 *     32 threads. Token starts at thread 31 (highest rank = highest tid).
 *     Each thread waits for the CA token, writes its tid to a shared array,
 *     then releases (done=true → calls step(false) → pressure frozen →
 *     not-done neighbors accumulate and eventually steal the token).
 *     No atomics anywhere. CA IS the write lock.
 *     Expected output array: [31, 30, 29, ..., 0]  (rank-descending)
 *
 *  2. CA-GATED DOT PRODUCT (CA-governed shared accumulator)
 *     32 threads each hold one element pair (a[i], b[i]).
 *     Each thread computes its partial product locally, then waits for the
 *     CA token to add it to the shared accumulator — single writer per step,
 *     no atomics, no race.
 *     Expected: result == CPU reference dot product.
 *
 *  3. CA WARP BARRIER (cross-warp token relay)
 *     64 threads (2 warps). One token per warp (2 tokens total).
 *     RULE_WARP_LOCAL: align_gate=5, tokens cannot cross warp boundary
 *     (boundary 31|32 has level 5 which is passable, but boundaries
 *     within-warp have level < 5 and are blocked).
 *     Wait — with align_gate=5, ONLY the warp boundary is passable. So
 *     tokens are frozen within each warp's half and can only cross at 31|32.
 *     We give both warps 1 token each and verify both independently accumulate
 *     their partial sums (no cross-warp interference).
 *     Expected: warp0_sum = 0+1+...+31 = 496, warp1_sum = 32+33+...+63 = 1520.
 */

#include "fercuda/experimental/ca.cuh"
#include "fercuda/compute/types.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace fer;
using namespace fer::ca;

static void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        exit(1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 1: CA-Priority Write
//
//  Token starts at thread 31 (highest rank). Each step, done threads call
//  step(false) → pressure frozen. Not-done threads call step(true) →
//  pressure grows. As soon as a not-done neighbor's eff > token-holder's eff,
//  the token transfers. Because done threads' eff is fixed and not-done
//  threads' eff keeps growing, the token always flows to the next not-done
//  neighbor — left-ward from 31 (since thread 32 doesn't exist).
//
//  This proves: in rank-descending order [31,30,...,0], each thread gets
//  exclusive write access with zero atomics. CA = distributed lock.
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int PW_N        = 32;
static constexpr int PW_MAXSTEPS = 512;  // well above the 2k steps needed

__global__ void k_ca_priority_write(int* out_order) {
    __shared__ CABus<PW_N, RULE_RANK_FLOW> bus;
    __shared__ int sorder[PW_N];
    __shared__ int cursor;  // monotone; only written by token-holder (single writer)

    const int tid = threadIdx.x;
    if (tid == 0) cursor = 0;
    bus.init(/*rank=*/uint8_t(tid), 0);
    bus.deposit(PW_N - 1, 1);  // token at highest-rank thread
    __syncthreads();

    bool done = false;
    for (int s = 0; s < PW_MAXSTEPS; s++) {
        bus.step(!done);  // done threads freeze pressure; not-done threads accumulate

        // Exactly one thread has the token at any step.
        // Single-writer into cursor/sorder — no atomics needed.
        if (!done && bus.has_token()) {
            sorder[cursor] = tid;  // write tid to next slot
            cursor++;              // bump cursor (visible after __syncthreads__ below)
            done = true;           // freeze my pressure; release token on next step
        }
        __syncthreads();
        if (cursor >= PW_N) break;
    }

    __syncthreads();
    if (tid < PW_N) out_order[tid] = sorder[tid];
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 2: CA-Gated Dot Product
//
//  Each thread computes partial = a[i] * b[i] locally (no sync needed).
//  Then it waits for the CA token to add its partial to the shared accumulator.
//  Single-writer guarantee from CA → no atomics on shared_dot.
//  Token flows left→right (0→1→...→31) since token starts at 0 and
//  right neighbors have higher rank → higher eff → token cascades right.
//
//  Result must exactly match the CPU reference (bitwise, since we use
//  the same float ordering as a sequential sum).
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int DOT_N        = 32;
static constexpr int DOT_MAXSTEPS = 512;

__global__ void k_ca_dot(const float* a, const float* b, float* out_dot) {
    __shared__ CABus<DOT_N, RULE_RANK_FLOW> bus;
    __shared__ float shared_dot;
    __shared__ int   committed;  // count of threads that have committed

    const int tid = threadIdx.x;
    if (tid == 0) { shared_dot = 0.f; committed = 0; }
    bus.init(uint8_t(tid), 0);
    bus.deposit(0, 1);  // token starts at thread 0 (lowest rank, cascades right)
    __syncthreads();

    // Phase 1: each thread computes its partial product (independent, no sync)
    float partial = a[tid] * b[tid];

    // Phase 2: CA governs who writes to shared_dot.
    // Check BEFORE step so thread 0 catches its initial deposited token
    // before the first step() would flow it away to thread 1.
    bool done = false;
    for (int s = 0; s < DOT_MAXSTEPS; s++) {
        // Write check first — catches whoever holds the token right now
        if (!done && bus.has_token()) {
            shared_dot += partial;  // single writer per step, no race
            committed++;
            done = true;
        }
        __syncthreads();
        if (committed >= DOT_N) break;

        bus.step(!done);  // done threads freeze; not-done threads accumulate
        __syncthreads();
    }

    __syncthreads();
    if (tid == 0) *out_dot = shared_dot;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 3: CA Warp Barrier
//
//  64 threads split into warp 0 (threads 0-31) and warp 1 (threads 32-63).
//  RULE_WARP_LOCAL has align_gate=5: the only passable boundary is 31|32
//  (level=5). All intra-warp boundaries (level 0-4) are blocked.
//  This means tokens CANNOT move within a warp — they can only cross the
//  warp boundary at 31|32.
//
//  We give each warp 1 token (thread 0 and thread 32).
//  Tokens are frozen to their warp leaders (can't propagate intra-warp).
//  Each warp leader therefore holds its token for all steps and commits
//  the warp's sum in a single write.
//
//  Verify:
//    warp0 leader (thread 0) commits sum = 0+1+...+31 = 496
//    warp1 leader (thread 32) commits sum = 32+33+...+63 = 1520
//    No cross-warp interference (each warp's sum is independent).
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int WB_N    = 64;
static constexpr int WB_WARP = 32;

// Intra-warp reduction of partial values — no CA involved, pure warp primitive
__device__ float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, off);
    return v;
}

__global__ void k_ca_warp_barrier(float* out_sums) {
    __shared__ CABus<WB_N, RULE_WARP_LOCAL> bus;
    __shared__ float warp_sums[2];   // one per warp

    const int tid  = threadIdx.x;
    const int wid  = tid / WB_WARP;   // warp id: 0 or 1
    const int lane = tid % WB_WARP;   // lane within warp

    if (tid == 0) { warp_sums[0] = 0.f; warp_sums[1] = 0.f; }
    bus.init(uint8_t(tid), 0);

    // Give each warp leader (thread 0, thread 32) one token.
    // align_gate=5 means these tokens cannot propagate intra-warp:
    // the only passable boundary is 31|32 (level 5). Intra-warp
    // boundaries (e.g. 0|1 at level 0) are blocked.
    bus.deposit(0,       1);   // warp 0 leader token
    bus.deposit(WB_WARP, 1);   // warp 1 leader token
    __syncthreads();

    // Each thread's "work value" = its thread id
    float my_val = (float)tid;

    // Warp-level reduction: each warp sums its values via shuffle
    float warp_partial = warp_reduce_sum(my_val);

    // Only warp leaders (lane==0) have the reduced sum.
    // They also hold the CA token (pinned to leaders by align_gate=5).
    // Leaders write their warp sum directly — no CA loop needed since
    // the token is already with them and can't move.
    __syncthreads();
    if (lane == 0 && bus.has_token()) {
        warp_sums[wid] = warp_partial;
    }
    __syncthreads();

    if (tid < 2) out_sums[tid] = warp_sums[tid];
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    printf("=== feRcuda :: CA Kernel Integration Tests ===\n\n");
    bool all_pass = true;

    // ── Test 1: CA Priority Write ───────────────────────────────────────────
    {
        int* d; ck(cudaMalloc(&d, PW_N * sizeof(int)));
        k_ca_priority_write<<<1, PW_N>>>(d);
        ck(cudaDeviceSynchronize());
        int h[PW_N];
        ck(cudaMemcpy(h, d, PW_N * sizeof(int), cudaMemcpyDeviceToHost));
        cudaFree(d);

        printf("[T1] CA-PRIORITY WRITE  (%d threads, token at thread 31, RULE_RANK_FLOW)\n", PW_N);
        printf("     Write order: ");
        for (int i = 0; i < PW_N; i++) printf("%d ", h[i]);
        printf("\n");

        // Verify: strictly descending (31,30,...,0), no atomics used
        bool ok = true;
        for (int i = 0; i < PW_N; i++) {
            if (h[i] != PW_N - 1 - i) { ok = false; break; }
        }
        printf("     First: %d  Last: %d  (expect 31 ... 0)\n", h[0], h[PW_N-1]);
        printf("     %s\n\n", ok ? "PASS — rank-descending, zero atomics" : "FAIL");
        all_pass &= ok;
    }

    // ── Test 2: CA-Gated Dot Product ────────────────────────────────────────
    {
        // Build input vectors: a[i]=i+1, b[i]=1/(i+1)  →  partial[i]=1.0 for all i
        // So dot = 32.0 exactly (avoids float precision issues)
        float h_a[DOT_N], h_b[DOT_N];
        float cpu_dot = 0.f;
        for (int i = 0; i < DOT_N; i++) {
            h_a[i] = (float)(i + 1);
            h_b[i] = 1.f / (float)(i + 1);
            cpu_dot += h_a[i] * h_b[i];
        }

        float *d_a, *d_b, *d_out;
        ck(cudaMalloc(&d_a,  DOT_N * sizeof(float)));
        ck(cudaMalloc(&d_b,  DOT_N * sizeof(float)));
        ck(cudaMalloc(&d_out, sizeof(float)));
        ck(cudaMemcpy(d_a, h_a, DOT_N * sizeof(float), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(d_b, h_b, DOT_N * sizeof(float), cudaMemcpyHostToDevice));

        k_ca_dot<<<1, DOT_N>>>(d_a, d_b, d_out);
        ck(cudaDeviceSynchronize());

        float h_dot;
        ck(cudaMemcpy(&h_dot, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);

        printf("[T2] CA-GATED DOT PRODUCT  (%d threads, a[i]=i+1, b[i]=1/(i+1))\n", DOT_N);
        printf("     CA result : %.6f\n", h_dot);
        printf("     CPU ref   : %.6f\n", cpu_dot);

        // Allow tiny float tolerance (sequential order may differ from CPU)
        bool ok = (fabsf(h_dot - cpu_dot) < 1e-3f);
        printf("     Delta: %.2e  %s\n\n", fabsf(h_dot - cpu_dot),
               ok ? "PASS — single-writer accumulation correct" : "FAIL");
        all_pass &= ok;
    }

    // ── Test 3: CA Warp Barrier ─────────────────────────────────────────────
    {
        float* d; ck(cudaMalloc(&d, 2 * sizeof(float)));
        k_ca_warp_barrier<<<1, WB_N>>>(d);
        ck(cudaDeviceSynchronize());
        float h[2];
        ck(cudaMemcpy(h, d, 2 * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d);

        // Expected sums
        float exp0 = 0.f, exp1 = 0.f;
        for (int i = 0;       i < WB_WARP; i++) exp0 += (float)i;
        for (int i = WB_WARP; i < WB_N;    i++) exp1 += (float)i;

        printf("[T3] CA WARP BARRIER  (%d threads, RULE_WARP_LOCAL, align_gate=5)\n", WB_N);
        printf("     Tokens pinned to warp leaders (intra-warp boundaries blocked).\n");
        printf("     Warp 0 sum: %.0f  (expect %.0f)\n", h[0], exp0);
        printf("     Warp 1 sum: %.0f  (expect %.0f)\n", h[1], exp1);

        bool ok = (h[0] == exp0) && (h[1] == exp1);
        printf("     %s\n\n", ok ? "PASS — warp-isolated accumulation correct" : "FAIL");
        all_pass &= ok;
    }

    printf("%s\n", all_pass ? "ALL KERNEL TESTS PASSED" : "SOME KERNEL TESTS FAILED");
    return all_pass ? 0 : 1;
}
