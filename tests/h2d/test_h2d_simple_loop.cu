/*
 * test_h2d_simple_loop.cu
 *
 * Minimal test: can we H2D-copy while a SIMPLE infinite loop runs on
 * another non-blocking stream?  If this passes but test_h2d_while_sched
 * fails, the issue is specific to the scheduler kernel (e.g. __syncthreads
 * or __threadfence).  If this also fails, ANY persistent kernel blocks
 * the PCIe copy engine.
 */
#include <cuda_runtime.h>
#include <cstdio>

// Minimal infinite-loop kernel (single thread, no __syncthreads, no __threadfence)
__global__ void k_spin_single() {
    volatile unsigned n = 0;
    while (true) { ++n; }
}

// Persistent kernel with __syncthreads (like the scheduler uses)
__global__ void k_spin_sync(int nthreads) {
    volatile unsigned n = 0;
    while (true) {
        __syncthreads();
        if (threadIdx.x == 0) ++n;
        __syncthreads();
    }
}

// Persistent kernel with __threadfence
__global__ void k_spin_fence() {
    volatile unsigned n = 0;
    while (true) {
        ++n;
        __threadfence();
    }
}

static void ck(cudaError_t e, const char* s) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA ERR at %s: %s\n", s, cudaGetErrorString(e));
        exit(1);
    }
}

static bool test_h2d(const char* label, cudaStream_t kern_stream,
                     const char* launch_desc) {
    printf("  [%s] H2D after launching: %s\n", label, launch_desc);
    fflush(stdout);

    cudaStream_t ops_stream;
    ck(cudaStreamCreateWithFlags(&ops_stream, cudaStreamNonBlocking), "ops_stream");

    float* da;
    ck(cudaMalloc(&da, 512 * sizeof(float)), "malloc da");
    float* h_pinned;
    ck(cudaMallocHost(&h_pinned, 512 * sizeof(float)), "mallocHost");
    for (int i = 0; i < 512; i++) h_pinned[i] = (float)i;

    ck(cudaMemcpyAsync(da, h_pinned, 512*sizeof(float),
                       cudaMemcpyHostToDevice, ops_stream), "h2d");
    printf("  [%s] copy enqueued, syncing...\n", label); fflush(stdout);

    // Use cudaStreamQuery to check if it completed without blocking
    cudaError_t q = cudaStreamQuery(ops_stream);
    printf("  [%s] streamQuery = %s\n", label,
           q == cudaSuccess       ? "cudaSuccess (done!)" :
           q == cudaErrorNotReady ? "cudaErrorNotReady (pending)" :
                                    cudaGetErrorString(q));
    fflush(stdout);

    // Now actually synchronize (will hang here if copy doesn't execute)
    ck(cudaStreamSynchronize(ops_stream), "sync");
    printf("  [%s] PASS\n", label); fflush(stdout);

    cudaFreeHost(h_pinned);
    cudaFree(da);
    cudaStreamDestroy(ops_stream);
    return true;
}

int main() {
    printf("[A] start\n"); fflush(stdout);
    ck(cudaSetDevice(0), "setDevice");

    cudaStream_t ks;

    // ── Test 1: single-thread spin loop, no sync, no fence ───────────────────
    ck(cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking), "ks1");
    k_spin_single<<<1, 1, 0, ks>>>();
    printf("[1] launched k_spin_single (1 thread, no sync/fence)\n"); fflush(stdout);
    test_h2d("1", ks, "k_spin_single");
    // Note: we never destroy ks since the kernel runs forever.
    // The process cleanup will kill it.
    printf("\n"); fflush(stdout);

    // ── Test 2: multi-thread with __syncthreads ───────────────────────────────
    ck(cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking), "ks2");
    k_spin_sync<<<1, 128, 0, ks>>>(128);
    printf("[2] launched k_spin_sync (128 threads, __syncthreads)\n"); fflush(stdout);
    test_h2d("2", ks, "k_spin_sync");
    printf("\n"); fflush(stdout);

    // ── Test 3: single-thread with __threadfence ──────────────────────────────
    ck(cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking), "ks3");
    k_spin_fence<<<1, 1, 0, ks>>>();
    printf("[3] launched k_spin_fence (1 thread, __threadfence)\n"); fflush(stdout);
    test_h2d("3", ks, "k_spin_fence");
    printf("\n"); fflush(stdout);

    printf("[Z] all tests done\n"); fflush(stdout);
    return 0;
}
