/*
 * test_ca.cu — CA token bus: three scenarios demonstrating emergent routing.
 *
 * KEY INSIGHT: tokens flow toward neighbors with higher "effective pressure":
 *   eff(cell) = pressure + rank * rank_weight / 15
 *
 * For tokens to move, intermediate threads must also accumulate pressure
 * (needs_token=true for all). Rank creates a gradient so tokens flow
 * directionally toward higher-rank cells, not just toward the nearest hungry one.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * Scenario 1  RANK CONVERGENCE  (single token flows to highest-rank thread)
 *   32 threads, rank = threadIdx.x (thread 31 = highest priority).
 *   1 token starts at thread 0. All threads need a token.
 *   RULE_RANK_FLOW: thresh=0, rank_weight=15, gate=0, decay=0, max=1.
 *   After k steps: token at thread min(k, 31).  After 31 steps: token at 31.
 *   (Beyond step ~33, thread 30's pressure exceeds thread 31's rank bonus and
 *    the token drifts back.  Test stops at step 31 before backward drift.)
 *
 * Scenario 2  MULTI-TOKEN PACKING  (8 tokens converge to the high-rank end)
 *   64 threads, rank = threadIdx.x. 8 tokens at threads 0-7. All need tokens.
 *   RULE_RANK_FLOW. After 60 steps: all tokens in threads 48-63.
 *
 * Scenario 3  ALIGNMENT GATE  (gate blocks or allows flow depending on boundary)
 *   8 threads, rank = threadIdx.x. 1 token at thread 0. All need tokens.
 *   Run twice:
 *     A) RULE_RANK_FLOW (gate=0, free flow)   → token reaches thread 7 after 7 steps.
 *     B) RULE_STRIDE2_GATE (gate=1)           → boundary 0|1 level=0 < gate=1, blocked.
 *        Token is frozen at thread 0.
 */

#include "fercuda/experimental/ca.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

using namespace fer::ca;

static void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        exit(1);
    }
}

// ─── Scenario 1: Rank Convergence ─────────────────────────────────────────────

static constexpr int S1_THREADS = 32;
// 31 steps: token hops 1 right per step, arriving at thread 31 exactly.
// Beyond step ~33, thread 30's accumulated pressure overtakes thread 31's
// effective pressure (rank bonus = fixed 31; pressure grows unboundedly),
// causing backward drift.  Stop exactly when token reaches max-rank position.
static constexpr int S1_STEPS   = 31;

struct S1Result { uint8_t tokens[S1_THREADS]; };

__global__ void k_rank_convergence(S1Result* out) {
    __shared__ CABus<S1_THREADS, RULE_RANK_FLOW> bus;
    const int tid = threadIdx.x;

    // rank = tid  (thread 31 = highest priority)
    bus.init(/*rank=*/uint8_t(tid), /*initial_tokens=*/0);
    bus.deposit(0, 1);   // single token at thread 0
    __syncthreads();

    for (int s = 0; s < S1_STEPS; s++)
        bus.step(/*needs_token=*/true);

    __syncthreads();
    if (tid < S1_THREADS) out->tokens[tid] = bus.my_tokens();
}

// ─── Scenario 2: Multi-Token Packing ─────────────────────────────────────────

static constexpr int S2_THREADS = 64;
static constexpr int S2_STEPS   = 60;

struct S2Result { uint8_t tokens[S2_THREADS]; };

__global__ void k_multi_token_pack(S2Result* out) {
    __shared__ CABus<S2_THREADS, RULE_RANK_FLOW> bus;
    const int tid = threadIdx.x;

    uint8_t init_tok = (tid < 8) ? 1 : 0;
    bus.init(/*rank=*/uint8_t(tid), init_tok);
    __syncthreads();

    for (int s = 0; s < S2_STEPS; s++)
        bus.step(/*needs_token=*/true);

    __syncthreads();
    if (tid < S2_THREADS) out->tokens[tid] = bus.my_tokens();
}

// ─── Scenario 3: Alignment Gate ───────────────────────────────────────────────

static constexpr int S3_THREADS = 8;
static constexpr int S3_STEPS   = 7;

struct S3Result {
    uint8_t tokens_free[S3_THREADS];   // RULE_RANK_FLOW  (gate=0)
    uint8_t tokens_gated[S3_THREADS];  // RULE_STRIDE2_GATE (gate=1)
};

__global__ void k_align_gate_free(uint8_t* out) {
    __shared__ CABus<S3_THREADS, RULE_RANK_FLOW> bus;
    const int tid = threadIdx.x;

    bus.init(uint8_t(tid), 0);
    bus.deposit(0, 1);
    __syncthreads();

    for (int s = 0; s < S3_STEPS; s++)
        bus.step(true);

    __syncthreads();
    if (tid < S3_THREADS) out[tid] = bus.my_tokens();
}

__global__ void k_align_gate_stride2(uint8_t* out) {
    __shared__ CABus<S3_THREADS, RULE_STRIDE2_GATE> bus;
    const int tid = threadIdx.x;

    bus.init(uint8_t(tid), 0);
    bus.deposit(0, 1);
    __syncthreads();

    for (int s = 0; s < S3_STEPS; s++)
        bus.step(true);

    __syncthreads();
    if (tid < S3_THREADS) out[tid] = bus.my_tokens();
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    printf("=== feRcuda :: CA Token Bus Tests ===\n\n");
    bool all_pass = true;

    // ── Scenario 1 ─────────────────────────────────────────────────────────
    {
        S1Result* d; ck(cudaMalloc(&d, sizeof(S1Result)));
        k_rank_convergence<<<1, S1_THREADS>>>(d);
        ck(cudaDeviceSynchronize());
        S1Result h; ck(cudaMemcpy(&h, d, sizeof(S1Result), cudaMemcpyDeviceToHost));
        cudaFree(d);

        printf("[S1] RANK CONVERGENCE  (%d threads, %d steps, RULE_RANK_FLOW)\n",
               S1_THREADS, S1_STEPS);
        printf("     1 token starts at thread 0, rank=tid.  Expect: token at thread 31.\n");

        int tok_pos = -1;
        for (int i = S1_THREADS - 1; i >= 0; i--) {
            if (h.tokens[i] > 0) { tok_pos = i; break; }
        }
        printf("     Token final position: thread %d\n", tok_pos);
        bool ok = (tok_pos == 31);
        printf("     %s\n\n", ok ? "PASS" : "FAIL");
        all_pass &= ok;
    }

    // ── Scenario 2 ─────────────────────────────────────────────────────────
    {
        S2Result* d; ck(cudaMalloc(&d, sizeof(S2Result)));
        k_multi_token_pack<<<1, S2_THREADS>>>(d);
        ck(cudaDeviceSynchronize());
        S2Result h; ck(cudaMemcpy(&h, d, sizeof(S2Result), cudaMemcpyDeviceToHost));
        cudaFree(d);

        printf("[S2] MULTI-TOKEN PACKING  (%d threads, %d steps, RULE_RANK_FLOW)\n",
               S2_THREADS, S2_STEPS);
        printf("     8 tokens start at threads 0-7, rank=tid.  Expect: all tokens in 48-63.\n");

        int in_upper = 0, in_lower = 0, total = 0;
        for (int i = 0; i < S2_THREADS; i++) {
            total += h.tokens[i];
            if (i >= 48) in_upper += h.tokens[i];
            else         in_lower += h.tokens[i];
        }
        printf("     Total tokens: %d  |  threads 48-63: %d  |  threads 0-47: %d\n",
               total, in_upper, in_lower);
        bool ok = (in_upper == 8 && total == 8);
        printf("     %s\n\n", ok ? "PASS" : "FAIL");
        all_pass &= ok;
    }

    // ── Scenario 3 ─────────────────────────────────────────────────────────
    {
        uint8_t* d_free;   ck(cudaMalloc(&d_free,   S3_THREADS));
        uint8_t* d_gated;  ck(cudaMalloc(&d_gated,  S3_THREADS));

        k_align_gate_free  <<<1, S3_THREADS>>>(d_free);
        k_align_gate_stride2<<<1, S3_THREADS>>>(d_gated);
        ck(cudaDeviceSynchronize());

        uint8_t h_free[S3_THREADS], h_gated[S3_THREADS];
        ck(cudaMemcpy(h_free,  d_free,  S3_THREADS, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(h_gated, d_gated, S3_THREADS, cudaMemcpyDeviceToHost));
        cudaFree(d_free);  cudaFree(d_gated);

        printf("[S3] ALIGNMENT GATE  (%d threads, %d steps)\n", S3_THREADS, S3_STEPS);
        printf("     Token starts at thread 0.  gate=0 vs gate=1 (stride-2 boundary lock).\n");

        int free_pos = -1, gated_pos = -1;
        for (int i = 0; i < S3_THREADS; i++) {
            if (h_free[i]  > 0) free_pos  = i;
            if (h_gated[i] > 0) gated_pos = i;
        }
        printf("     gate=0 (free):    token at thread %d  (expect 7)\n", free_pos);
        printf("     gate=1 (stride2): token at thread %d  (expect 0, blocked)\n", gated_pos);

        // Free flow: token should have reached thread 7 (7 hops in 7 steps)
        // Gated: boundary 0|1 has alignment level 0 < gate 1 — token frozen at 0
        bool ok = (free_pos == 7) && (gated_pos == 0);
        printf("     %s\n\n", ok ? "PASS" : "FAIL");
        all_pass &= ok;
    }

    printf("%s\n", all_pass ? "ALL CA TESTS PASSED" : "SOME CA TESTS FAILED");
    return all_pass ? 0 : 1;
}
