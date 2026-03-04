
#include "fercuda/scheduler/scheduler.cuh"
#include "fercuda/compute/ops.cuh"
#include "fercuda/compute/types.cuh"
#include <cstdio>
using namespace fer;

static void ck(cudaError_t e, const char* s) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA ERR at %s: %s\n", s, cudaGetErrorString(e)); exit(1); }
}

int main() {
    printf("[A] start\n"); fflush(stdout);
    cudaSetDevice(0);

    // Non-blocking ops stream
    cudaStream_t ops_stream;
    ck(cudaStreamCreateWithFlags(&ops_stream, cudaStreamNonBlocking), "stream");
    printf("[B] ops_stream created\n"); fflush(stdout);

    // Allocate device buffers
    float *da, *db, *dc;
    ck(cudaMalloc(&da, 512*sizeof(float)), "mallocA");
    ck(cudaMalloc(&db, 512*sizeof(float)), "mallocB");
    ck(cudaMalloc(&dc, 512*sizeof(float)), "mallocC");
    printf("[C] buffers allocated\n"); fflush(stdout);

    // Test ops BEFORE launching scheduler
    ck(cudaMemsetAsync(da, 0, 512*sizeof(float), ops_stream), "memset da");
    ck(cudaMemsetAsync(db, 0, 512*sizeof(float), ops_stream), "memset db");
    auto ta = FTensor1D(reinterpret_cast<F32*>(da), Shape<1>(512));
    auto tb = FTensor1D(reinterpret_cast<F32*>(db), Shape<1>(512));
    auto tc = FTensor1D(reinterpret_cast<F32*>(dc), Shape<1>(512));
    ops::add(ta, tb, tc, ops_stream);
    ck(cudaStreamSynchronize(ops_stream), "sync1");
    printf("[D] ops work before scheduler: PASS\n"); fflush(stdout);

    // Now launch the scheduler
    auto* sched = Scheduler::create(0);
    sched->launch(128);
    printf("[E] scheduler launched\n"); fflush(stdout);

    // Submit 9 items
    uint64_t args[SCHED_MAX_ARGS] = {};
    for (int i = 0; i < 9; i++) sched->submit(OpCode::ELEMENTWISE, 1, args, 0);
    printf("[F] 9 items submitted\n"); fflush(stdout);

    // Now test ops WHILE scheduler is running
    printf("[G] about to run ops on ops_stream while scheduler running...\n"); fflush(stdout);
    ck(cudaMemsetAsync(da, 0, 512*sizeof(float), ops_stream), "memset2");
    printf("[H] memsetAsync done\n"); fflush(stdout);
    ops::add(ta, tb, tc, ops_stream);
    printf("[I] add kernel queued\n"); fflush(stdout);
    ck(cudaStreamSynchronize(ops_stream), "sync2");
    printf("[J] ops while scheduler running: PASS\n"); fflush(stdout);

    sched->shutdown();
    printf("[K] shutdown done\n"); fflush(stdout);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    delete sched;
    printf("[L] done\n"); fflush(stdout);
    return 0;
}
