/*
 * test_sched_pool_interop.cu
 *
 * Isolate whether ElasticPool (256MB pre-alloc) interferes with
 * compute kernels while the persistent scheduler is running.
 * Mirrors test_sched_ops_interop exactly but adds ElasticPool.
 */
#include "fercuda/scheduler/scheduler.cuh"
#include "fercuda/alloc/memory.cuh"
#include "fercuda/compute/ops.cuh"
#include <cstdio>
using namespace fer;

static void ck(cudaError_t e, const char* s) {
    if (e != cudaSuccess) { fprintf(stderr, "ERR %s: %s\n", s, cudaGetErrorString(e)); exit(1); }
}

int main() {
    printf("[A] start\n"); fflush(stdout);
    cudaSetDevice(0);

    cudaStream_t ops_stream;
    ck(cudaStreamCreateWithFlags(&ops_stream, cudaStreamNonBlocking), "stream");
    printf("[B] ops_stream created\n"); fflush(stdout);

    // ── Add ElasticPool (256MB pre-alloc) — the key new variable
    ElasticPool pool(0, {
        .mutable_bytes   = 128ULL << 20,
        .immutable_bytes = 128ULL << 20,
        .verbose         = false,
    });
    printf("[C] pool created (256MB pre-alloc)\n"); fflush(stdout);

    constexpr int N = 512;
    auto a   = pool.alloc_mutable<F32, 1>(Shape<1>(N));
    auto b   = pool.alloc_mutable<F32, 1>(Shape<1>(N));
    auto out = pool.alloc_mutable<F32, 1>(Shape<1>(N));
    printf("[D] pool tensors allocated\n"); fflush(stdout);

    // Pre-scheduler compute ops (identical to test_sched_ops_interop [D])
    ck(cudaMemsetAsync(a.data, 0, N*sizeof(float), ops_stream), "memset a");
    ck(cudaMemsetAsync(b.data, 0, N*sizeof(float), ops_stream), "memset b");
    ops::add(a, b, out, ops_stream);
    ck(cudaStreamSynchronize(ops_stream), "sync1");
    printf("[E] ops before scheduler: PASS\n"); fflush(stdout);

    // Launch scheduler
    auto* sched = Scheduler::create(0);
    sched->launch(128);
    printf("[F] scheduler launched\n"); fflush(stdout);

    // Submit 9 items
    uint64_t args[SCHED_MAX_ARGS] = {};
    for (int i = 0; i < 9; i++) sched->submit(OpCode::ELEMENTWISE, 1, args, 0);
    printf("[G] 9 items submitted\n"); fflush(stdout);

    // Ops while scheduler running (identical to test_sched_ops_interop [H-J])
    ck(cudaMemsetAsync(a.data, 0, N*sizeof(float), ops_stream), "memset2");
    printf("[H] memsetAsync done\n"); fflush(stdout);
    ops::add(a, b, out, ops_stream);
    printf("[I] add kernel queued\n"); fflush(stdout);
    ck(cudaStreamSynchronize(ops_stream), "sync2");
    printf("[J] ops while scheduler running: PASS\n"); fflush(stdout);

    sched->shutdown();
    printf("[K] shutdown done\n"); fflush(stdout);
    delete sched;
    ck(cudaStreamDestroy(ops_stream), "stream destroy");
    printf("[L] done\n"); fflush(stdout);
    return 0;
}
