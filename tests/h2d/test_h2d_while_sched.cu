/*
 * test_h2d_while_sched.cu
 *
 * Diagnostic: can we H2D-copy pinned host→device on a non-blocking stream
 * while the persistent scheduler kernel is running on another non-blocking stream?
 *
 * If the sync after [F] hangs, the H2D copy engine is being blocked by the
 * persistent kernel — which would explain the demo_persistent hang.
 */
#include "fercuda/scheduler/scheduler.cuh"
#include <cstdio>
#include <cstring>
using namespace fer;

static void ck(cudaError_t e, const char* s) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA ERR at %s: %s\n", s, cudaGetErrorString(e));
        exit(1);
    }
}

int main() {
    printf("[A] start\n"); fflush(stdout);
    cudaSetDevice(0);

    cudaStream_t ops_stream;
    ck(cudaStreamCreateWithFlags(&ops_stream, cudaStreamNonBlocking), "create ops_stream");
    printf("[B] ops_stream created\n"); fflush(stdout);

    // Device buffer + pinned host buffer
    float* da;
    ck(cudaMalloc(&da, 512 * sizeof(float)), "cudaMalloc da");
    float* h_pinned;
    ck(cudaMallocHost(&h_pinned, 512 * sizeof(float)), "cudaMallocHost");
    for (int i = 0; i < 512; i++) h_pinned[i] = (float)i;
    printf("[C] buffers ready\n"); fflush(stdout);

    // ── H2D copy BEFORE scheduler ─────────────────────────────────────────────
    printf("[D] H2D copy (pinned→device) BEFORE scheduler launch...\n"); fflush(stdout);
    ck(cudaMemcpyAsync(da, h_pinned, 512*sizeof(float), cudaMemcpyHostToDevice, ops_stream),
       "h2d_before");
    ck(cudaStreamSynchronize(ops_stream), "sync_before");
    printf("[E] H2D BEFORE scheduler: PASS\n"); fflush(stdout);

    // ── Launch persistent scheduler ──────────────────────────────────────────
    auto* sched = Scheduler::create(0);
    sched->launch(128);
    printf("[F] scheduler launched\n"); fflush(stdout);

    uint64_t args[SCHED_MAX_ARGS] = {};
    for (int i = 0; i < 9; i++) sched->submit(OpCode::ELEMENTWISE, 1, args, 0);
    printf("[G] 9 items submitted\n"); fflush(stdout);

    // ── H2D copy AFTER scheduler is running ──────────────────────────────────
    printf("[H] H2D copy (pinned→device) WHILE scheduler running...\n"); fflush(stdout);
    ck(cudaMemcpyAsync(da, h_pinned, 512*sizeof(float), cudaMemcpyHostToDevice, ops_stream),
       "h2d_after_queue");
    printf("[I] cudaMemcpyAsync returned — calling cudaStreamSynchronize...\n"); fflush(stdout);
    ck(cudaStreamSynchronize(ops_stream), "sync_after");
    printf("[J] H2D WHILE scheduler running: PASS\n"); fflush(stdout);

    // ── D2H readback WHILE scheduler running ─────────────────────────────────
    float* h_res;
    ck(cudaMallocHost(&h_res, sizeof(float)), "cudaMallocHost h_res");
    printf("[K] D2H copy (device→pinned) WHILE scheduler running...\n"); fflush(stdout);
    ck(cudaMemcpyAsync(h_res, da, sizeof(float), cudaMemcpyDeviceToHost, ops_stream),
       "d2h_after_queue");
    ck(cudaStreamSynchronize(ops_stream), "sync_d2h");
    printf("[L] D2H WHILE scheduler running: PASS  da[0]=%.1f\n", *h_res); fflush(stdout);

    // ── Shutdown FIRST — cudaFreeHost/cudaFree trigger implicit device sync
    // in CUDA 12.x which blocks while a persistent kernel is still running.
    sched->shutdown();
    printf("[M] scheduler shutdown\n"); fflush(stdout);
    cudaFreeHost(h_res);
    cudaFreeHost(h_pinned);
    cudaFree(da);
    delete sched;
    cudaStreamDestroy(ops_stream);
    printf("[N] done\n"); fflush(stdout);
    return 0;
}
