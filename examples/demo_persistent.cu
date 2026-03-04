/*
 * demo_persistent.cu
 *
 * Shows the persistent scheduler receiving typed work from the host,
 * dispatching it, and reporting completion.
 *
 * CE/SM inter-engine rules discovered in CUDA 12.x (RTX 3070, sm_86):
 *
 *  1. Ephemeral DMA stream: use a dedicated non-blocking stream for H2D DMA,
 *     then destroy it before launching the persistent kernel. Never reuse a
 *     stream that carried H2D work for subsequent SM kernels while a persistent
 *     kernel is running — CE-order tracking persists across a cudaStreamDestroy
 *     of a different stream and taints the ops stream.
 *
 *  2. SM context warm-up: before the persistent kernel launches, run at least
 *     one real SM kernel on the compute stream (cudaMemsetAsync alone is CE-
 *     based and is not sufficient). This registers the stream as SM-active with
 *     the CUDA driver.
 *
 *  3. No CE→SM interleaving while persistent kernel runs: after each D2H or
 *     H2D copy engine operation on the compute stream, launching an SM kernel
 *     causes cudaStreamSynchronize to hang. Keep all CE and SM work in separate
 *     phases. While the persistent kernel is running, use the compute stream
 *     for SM-only work (chain multiple kernels with one sync at the end). Do
 *     all D2H readbacks after sched->shutdown().
 *
 *  4. cudaFreeHost/cudaFree ordering: in CUDA 12.x these trigger an implicit
 *     device-wide sync. Always call sched->shutdown() first.
 */

#include "fercuda/scheduler/scheduler.cuh"
#include "fercuda/alloc/memory.cuh"
#include "fercuda/compute/ops.cuh"
#include <cstdio>
#include <cstdlib>

using namespace fer;

static void check(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); exit(1); }
}

int main() {
    printf("=== feRcuda :: Persistent Scheduler Demo ===\n\n");
    check(cudaSetDevice(0));

    // ── Elastic pool for tensors ──────────────────────────────────────────────
    ElasticPool pool(0, {
        .mutable_bytes   = 128ULL << 20,
        .immutable_bytes = 128ULL << 20,
        .verbose         = true,
    });
    pool.print_stats();
    printf("\n");

    constexpr int N = 512;
    auto a        = pool.alloc_mutable<F32, 1>(Shape<1>(N));
    auto b        = pool.alloc_mutable<F32, 1>(Shape<1>(N));
    auto add_out  = pool.alloc_mutable<F32, 1>(Shape<1>(N));  // result of a+b
    auto gelu_out = pool.alloc_mutable<F32, 1>(Shape<1>(N));  // result of gelu(a)

    // ── Upload via ephemeral DMA stream, BEFORE scheduler launch (rule 1) ─────
    float* h_stage;
    check(cudaMallocHost(&h_stage, 2 * N * sizeof(float)));
    memset(h_stage, 0, 2 * N * sizeof(float));
    h_stage[0] = 2.5f;   // a[0]
    h_stage[N] = 1.5f;   // b[0]
    {
        cudaStream_t dma;
        check(cudaStreamCreateWithFlags(&dma, cudaStreamNonBlocking));
        check(cudaMemcpyAsync(a.data, h_stage,     N*sizeof(float), cudaMemcpyHostToDevice, dma));
        check(cudaMemcpyAsync(b.data, h_stage + N, N*sizeof(float), cudaMemcpyHostToDevice, dma));
        check(cudaStreamSynchronize(dma));
        check(cudaStreamDestroy(dma));
    }
    cudaFreeHost(h_stage);
    printf("[host] Tensor data uploaded\n");

    // ── Compute stream: warm up with a real SM kernel before scheduler (rule 2)
    cudaStream_t ops_stream;
    check(cudaStreamCreateWithFlags(&ops_stream, cudaStreamNonBlocking));
    check(cudaMemsetAsync(add_out.data, 0, N * sizeof(float), ops_stream));
    ops::add(a, b, add_out, ops_stream);   // establishes SM context on ops_stream
    check(cudaStreamSynchronize(ops_stream));
    printf("[host] ops_stream SM context established (pre-scheduler warmup)\n");

    // Pinned result buffers — allocate before scheduler to avoid implicit sync
    float h_add_result = 0.f, h_gelu_result = 0.f;
    float* h_res;
    check(cudaMallocHost(&h_res, 2 * sizeof(float)));

    // ── Create and launch the persistent scheduler ────────────────────────────
    auto* sched = Scheduler::create(0);
    sched->launch(128);
    printf("[host] Persistent scheduler kernel launched (128 threads)\n");

    // ── Submit some work items ────────────────────────────────────────────────
    uint64_t args[SCHED_MAX_ARGS] = {};
    for (int i = 0; i < 8; i++) {
        args[0] = (uint64_t)i;
        uint32_t id = sched->submit(OpCode::ELEMENTWISE, /*priority=*/1, args, 1);
        printf("[host] Submitted ELEMENTWISE id=%u priority=1\n", id);
    }
    args[0] = 0xDEADBEEF;
    uint32_t hp_id = sched->submit(OpCode::MATMUL, /*priority=*/0, args, 1);
    printf("[host] Submitted MATMUL id=%u priority=0 (HIGH)\n", hp_id);

    // ── SM-only compute while scheduler runs (rule 3) ────────────────────────
    // Only ONE cudaStreamSynchronize is safe per "epoch" while the persistent
    // kernel is spinning.  A second sync on the same stream — even after the
    // first one returns — hangs in CUDA 12.x.
    // Workaround: chain all SM kernels in one batch (one prime memset, then all
    // user kernels back-to-back), and call cudaStreamSynchronize exactly once.
    // ── SM compute while scheduler dispatches (ops::add — SFU-free, safe) ─────
    // ops::gelu uses tanhf() which calls the SFU; SFU interaction with the
    // persistent kernel's BAR.SYNC appears to block while the persistent kernel
    // is running on the same device.  Non-SFU kernels (ops::add) work fine.
    // Run gelu after sched->shutdown() instead.
    check(cudaMemsetAsync(add_out.data, 0, N * sizeof(float), ops_stream));
    ops::add(a, b, add_out, ops_stream);
    check(cudaStreamSynchronize(ops_stream));
    printf("[host] ops::add complete (concurrent with scheduler)\n");

    // ── Shutdown scheduler BEFORE any cudaFree/cudaFreeHost (rule 4) ─────────
    sched->shutdown();
    printf("\n[host] Scheduler shut down. ops_completed=%llu\n",
           (unsigned long long)sched->ops_completed());

    // ── Post-shutdown: gelu + D2H (persistent kernel gone — no SFU conflict) ─
    ops::gelu(a, gelu_out, ops_stream);
    check(cudaStreamSynchronize(ops_stream));

    check(cudaMemcpyAsync(h_res,     add_out.data,  sizeof(float), cudaMemcpyDeviceToHost, ops_stream));
    check(cudaMemcpyAsync(h_res + 1, gelu_out.data, sizeof(float), cudaMemcpyDeviceToHost, ops_stream));
    check(cudaStreamSynchronize(ops_stream));
    h_add_result  = h_res[0];
    h_gelu_result = h_res[1];

    cudaFreeHost(h_res);
    printf("\n[ops] add(2.5, 1.5)[0]  = %.6f  (expect 4.0)\n",   h_add_result);
    printf("[ops] gelu(2.5)[0]      = %.6f  (expect ~2.496)\n", h_gelu_result);

    pool.print_stats();
    delete sched;
    check(cudaStreamDestroy(ops_stream));
    printf("\ndone.\n");
    return 0;
}
