/*
 * test_h2d_isolate.cu
 *
 * Isolates whether the H2D DMA hang is caused by:
 *   A) 128-thread __syncthreads kernel alone (no DMA submits)
 *   B) 1-thread kernel + submit() DMA traffic on control_stream
 *   C) Both together
 *
 * Each sub-test exits the process immediately after the result so that
 * persistent kernels don't interfere with each other (no cudaFree hang).
 * Run with: ./test_h2d_isolate A  or  B
 */
#include "fercuda/scheduler/scheduler.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
using namespace fer;

static void ck(cudaError_t e, const char* s) {
    if (e != cudaSuccess) { fprintf(stderr, "ERR %s: %s\n", s, cudaGetErrorString(e)); exit(1); }
}

// Try H2D copy on ops_stream.  Returns 1 on success, calls exit if timeout-like.
static int try_h2d(const char* tag) {
    cudaStream_t ops;
    ck(cudaStreamCreateWithFlags(&ops, cudaStreamNonBlocking), "ops");

    float *da;
    ck(cudaMalloc(&da, 512*sizeof(float)), "malloc");
    float *hp;
    ck(cudaMallocHost(&hp, 512*sizeof(float)), "mallocHost");
    for (int i = 0; i < 512; i++) hp[i] = (float)i;

    ck(cudaMemcpyAsync(da, hp, 512*sizeof(float), cudaMemcpyHostToDevice, ops), "h2d");
    printf("[%s] copy queued\n", tag); fflush(stdout);

    // Query first — if done already, no need to sync
    cudaError_t q = cudaStreamQuery(ops);
    printf("[%s] streamQuery = %s\n", tag,
           q == cudaSuccess       ? "cudaSuccess" :
           q == cudaErrorNotReady ? "cudaErrorNotReady" : cudaGetErrorString(q));
    fflush(stdout);

    ck(cudaStreamSynchronize(ops), "sync");
    printf("[%s] cudaStreamSynchronize returned: PASS\n", tag); fflush(stdout);
    return 1;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s A|B|C\n", argv[0]);
        return 1;
    }
    char mode = argv[1][0];

    printf("[start] mode=%c\n", mode); fflush(stdout);
    ck(cudaSetDevice(0), "setDevice");

    if (mode == 'A') {
        // ── Test A: 128-thread __syncthreads persistent kernel, NO submits ──
        printf("[A] launching scheduler with 128 threads, 0 submits\n"); fflush(stdout);
        auto* sched = Scheduler::create(0);
        sched->launch(128);
        printf("[A] scheduler launched\n"); fflush(stdout);

        try_h2d("A");
        // Don't cleanup — just exit cleanly
        printf("[A] done\n"); fflush(stdout);

    } else if (mode == 'B') {
        // ── Test B: 1-thread scheduler, 9 DMA submits, then H2D ─────────────
        printf("[B] launching scheduler with 1 thread, 9 submits\n"); fflush(stdout);
        auto* sched = Scheduler::create(0);
        sched->launch(1);
        printf("[B] scheduler launched\n"); fflush(stdout);

        uint64_t args[SCHED_MAX_ARGS] = {};
        for (int i = 0; i < 9; i++) sched->submit(OpCode::ELEMENTWISE, 1, args, 0);
        printf("[B] 9 items submitted\n"); fflush(stdout);

        try_h2d("B");
        printf("[B] done\n"); fflush(stdout);

    } else if (mode == 'C') {
        // ── Test C: 128-thread scheduler, 9 DMA submits, then H2D ───────────
        printf("[C] launching scheduler with 128 threads, 9 submits\n"); fflush(stdout);
        auto* sched = Scheduler::create(0);
        sched->launch(128);
        printf("[C] scheduler launched\n"); fflush(stdout);

        uint64_t args[SCHED_MAX_ARGS] = {};
        for (int i = 0; i < 9; i++) sched->submit(OpCode::ELEMENTWISE, 1, args, 0);
        printf("[C] 9 items submitted\n"); fflush(stdout);

        try_h2d("C");
        printf("[C] done\n"); fflush(stdout);

    } else if (mode == 'D') {
        // ── Test D: REPLICATES test_h2d_while_sched exactly ──────────────────
        // ops_stream created first, H2D BEFORE scheduler, then scheduler + submits,
        // then H2D AGAIN on same ops_stream.  If this hangs but C passes, the
        // pre-scheduler H2D on ops_stream is what causes the subsequent hang.
        printf("[D] creating ops_stream BEFORE scheduler\n"); fflush(stdout);
        cudaStream_t ops;
        ck(cudaStreamCreateWithFlags(&ops, cudaStreamNonBlocking), "ops_d");

        float *da;
        ck(cudaMalloc(&da, 512*sizeof(float)), "malloc_d");
        float *hp;
        ck(cudaMallocHost(&hp, 512*sizeof(float)), "mallocHost_d");
        for (int i = 0; i < 512; i++) hp[i] = (float)i;

        // First H2D copy BEFORE scheduler
        printf("[D] H2D copy #1 (before scheduler)...\n"); fflush(stdout);
        ck(cudaMemcpyAsync(da, hp, 512*sizeof(float), cudaMemcpyHostToDevice, ops), "h2d_d1");
        ck(cudaStreamSynchronize(ops), "sync_d1");
        printf("[D] H2D #1 PASS\n"); fflush(stdout);

        // Launch scheduler with 128 threads + 9 submits
        auto* sched = Scheduler::create(0);
        sched->launch(128);
        printf("[D] scheduler launched\n"); fflush(stdout);

        uint64_t args[SCHED_MAX_ARGS] = {};
        for (int i = 0; i < 9; i++) sched->submit(OpCode::ELEMENTWISE, 1, args, 0);
        printf("[D] 9 items submitted\n"); fflush(stdout);

        // Second H2D copy AFTER scheduler — on SAME ops_stream
        printf("[D] H2D copy #2 (after scheduler, same ops_stream)...\n"); fflush(stdout);
        ck(cudaMemcpyAsync(da, hp, 512*sizeof(float), cudaMemcpyHostToDevice, ops), "h2d_d2");
        printf("[D] copy #2 queued, streamQuery...\n"); fflush(stdout);
        cudaError_t q = cudaStreamQuery(ops);
        printf("[D] streamQuery = %s\n",
               q == cudaSuccess       ? "cudaSuccess" :
               q == cudaErrorNotReady ? "cudaErrorNotReady" : cudaGetErrorString(q));
        fflush(stdout);
        ck(cudaStreamSynchronize(ops), "sync_d2");
        printf("[D] H2D #2 PASS\n"); fflush(stdout);
        printf("[D] done\n"); fflush(stdout);
    }

    return 0;
}
