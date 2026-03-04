/*
 * test_sched_dma_then_ops.cu
 *
 * Adds DMA-before-scheduler to the passing test_sched_pool_interop to isolate
 * whether H2D DMA on a destroyed dma_stream is what breaks subsequent compute.
 */
#include "fercuda/scheduler/scheduler.cuh"
#include "fercuda/alloc/memory.cuh"
#include "fercuda/compute/ops.cuh"
#include <cstdio>
#include <cstring>
using namespace fer;

static void ck(cudaError_t e, const char* s) {
    if (e != cudaSuccess) { fprintf(stderr, "ERR %s: %s\n", s, cudaGetErrorString(e)); exit(1); }
}

int main() {
    printf("[A] start\n"); fflush(stdout);
    cudaSetDevice(0);

    cudaStream_t ops_stream;
    ck(cudaStreamCreateWithFlags(&ops_stream, cudaStreamNonBlocking), "ops_stream");
    printf("[B] ops_stream created\n"); fflush(stdout);

    ElasticPool pool(0, {
        .mutable_bytes   = 128ULL << 20,
        .immutable_bytes = 128ULL << 20,
        .verbose         = false,
    });
    printf("[C] pool created\n"); fflush(stdout);

    constexpr int N = 512;
    auto a   = pool.alloc_mutable<F32, 1>(Shape<1>(N));
    auto b   = pool.alloc_mutable<F32, 1>(Shape<1>(N));
    auto out = pool.alloc_mutable<F32, 1>(Shape<1>(N));
    printf("[D] pool tensors allocated\n"); fflush(stdout);

    // ── NEW: H2D DMA via ephemeral dma_stream (the extra step demo_persistent has)
    float* h_stage;
    ck(cudaMallocHost(&h_stage, 2 * N * sizeof(float)), "mallocHost stage");
    memset(h_stage, 0, 2 * N * sizeof(float));
    h_stage[0] = 2.5f;
    h_stage[N] = 1.5f;
    {
        cudaStream_t dma_stream;
        ck(cudaStreamCreateWithFlags(&dma_stream, cudaStreamNonBlocking), "dma_stream");
        ck(cudaMemcpyAsync(a.data, h_stage,     N*sizeof(float), cudaMemcpyHostToDevice, dma_stream), "h2d_a");
        ck(cudaMemcpyAsync(b.data, h_stage + N, N*sizeof(float), cudaMemcpyHostToDevice, dma_stream), "h2d_b");
        ck(cudaStreamSynchronize(dma_stream), "sync_dma");
        ck(cudaStreamDestroy(dma_stream), "destroy_dma");
    }
    cudaFreeHost(h_stage);
    printf("[E] H2D via dma_stream done (destroyed)\n"); fflush(stdout);

    // ── Pre-scheduler compute warmup on ops_stream (same as test_sched_pool_interop)
    ck(cudaMemsetAsync(a.data, 0, N*sizeof(float), ops_stream), "memset a");
    ck(cudaMemsetAsync(b.data, 0, N*sizeof(float), ops_stream), "memset b");
    ops::add(a, b, out, ops_stream);
    ck(cudaStreamSynchronize(ops_stream), "sync_warmup");
    printf("[F] pre-scheduler compute warmup: PASS\n"); fflush(stdout);

    auto* sched = Scheduler::create(0);
    sched->launch(128);
    printf("[G] scheduler launched\n"); fflush(stdout);

    uint64_t args[SCHED_MAX_ARGS] = {};
    for (int i = 0; i < 9; i++) sched->submit(OpCode::ELEMENTWISE, 1, args, 0);
    printf("[H] 9 items submitted\n"); fflush(stdout);

    // ── Ops while scheduler running (same as test_sched_pool_interop pattern)
    ck(cudaMemsetAsync(a.data, 0, N*sizeof(float), ops_stream), "memset2");
    printf("[I] memsetAsync done\n"); fflush(stdout);
    ops::add(a, b, out, ops_stream);
    printf("[J] add kernel queued\n"); fflush(stdout);
    ck(cudaStreamSynchronize(ops_stream), "sync2");
    printf("[K] ops while scheduler running: PASS\n"); fflush(stdout);

    sched->shutdown();
    printf("[L] shutdown done\n"); fflush(stdout);
    delete sched;
    ck(cudaStreamDestroy(ops_stream), "ops stream destroy");
    printf("[M] done\n"); fflush(stdout);
    return 0;
}
