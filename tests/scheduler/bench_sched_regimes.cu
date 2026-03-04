#include "fercuda/scheduler/scheduler.cuh"

#include <chrono>
#include <cstdlib>
#include <cstdio>

using namespace fer;

struct BenchResult {
    const char* mode;
    int items;
    double seconds;
    double ops_per_sec;
    uint64_t done;
    uint64_t warp_dispatches;
    uint32_t launch_blocks;
    float sample;
};

static BenchResult run_mode(ExecRegime mode, const char* label, int items, int work_iters) {
    auto* sched = Scheduler::create(0);
    sched->launch(128, mode);

    int out_elems = 1;
    while (out_elems < items) out_elems <<= 1;
    float* d_out = nullptr;
    cudaMalloc(&d_out, static_cast<size_t>(out_elems) * sizeof(float));
    cudaMemset(d_out, 0, static_cast<size_t>(out_elems) * sizeof(float));

    uint64_t args[SCHED_MAX_ARGS] = {};
    args[0] = static_cast<uint64_t>(work_iters);
    args[1] = reinterpret_cast<uint64_t>(d_out);
    args[2] = static_cast<uint64_t>(out_elems - 1);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < items; ++i) {
        (void)sched->submit(OpCode::ELEMENTWISE, 1, args, 3);
    }
    sched->sync();
    auto t1 = std::chrono::steady_clock::now();

    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const uint64_t done = sched->ops_completed();
    const uint64_t warp_dispatches = sched->warp_dispatches();
    const uint32_t launch_blocks = sched->launch_blocks();
    float sample = 0.0f;
    cudaMemcpy(&sample, d_out + ((items - 1) & (out_elems - 1)), sizeof(float), cudaMemcpyDeviceToHost);

    sched->shutdown();
    delete sched;
    cudaFree(d_out);

    BenchResult r{};
    r.mode = label;
    r.items = items;
    r.seconds = seconds > 0.0 ? seconds : 1e-9;
    r.ops_per_sec = static_cast<double>(items) / r.seconds;
    r.done = done;
    r.warp_dispatches = warp_dispatches;
    r.launch_blocks = launch_blocks;
    r.sample = sample;
    return r;
}

int main(int argc, char** argv) {
    int items = 20000;
    int work_iters = 1024;
    if (argc > 1) {
        const int v = std::atoi(argv[1]);
        if (v > 0) items = v;
    }
    if (argc > 2) {
        const int v = std::atoi(argv[2]);
        if (v > 0) work_iters = v;
    }

    cudaSetDevice(0);

    const BenchResult single = run_mode(ExecRegime::SINGLE_CTA, "single_cta", items, work_iters);
    const BenchResult smwarp = run_mode(ExecRegime::SMWARP, "smwarp", items, work_iters);

    std::printf("bench.items=%d\n", items);
    std::printf("bench.work_iters=%d\n", work_iters);
    std::printf("single.seconds=%.6f\n", single.seconds);
    std::printf("single.ops_per_sec=%.2f\n", single.ops_per_sec);
    std::printf("single.done=%llu\n", static_cast<unsigned long long>(single.done));
    std::printf("single.warp_dispatches=%llu\n", static_cast<unsigned long long>(single.warp_dispatches));
    std::printf("single.launch_blocks=%u\n", single.launch_blocks);
    std::printf("single.sample=%f\n", single.sample);

    std::printf("smwarp.seconds=%.6f\n", smwarp.seconds);
    std::printf("smwarp.ops_per_sec=%.2f\n", smwarp.ops_per_sec);
    std::printf("smwarp.done=%llu\n", static_cast<unsigned long long>(smwarp.done));
    std::printf("smwarp.warp_dispatches=%llu\n", static_cast<unsigned long long>(smwarp.warp_dispatches));
    std::printf("smwarp.launch_blocks=%u\n", smwarp.launch_blocks);
    std::printf("smwarp.sample=%f\n", smwarp.sample);

    const double speedup = smwarp.ops_per_sec / (single.ops_per_sec > 0.0 ? single.ops_per_sec : 1e-9);
    std::printf("smwarp.speedup_vs_single=%.4fx\n", speedup);

    if (single.done != static_cast<uint64_t>(items) || smwarp.done != static_cast<uint64_t>(items)) {
        std::fprintf(stderr, "done mismatch single=%llu smwarp=%llu expected=%d\n",
                     static_cast<unsigned long long>(single.done),
                     static_cast<unsigned long long>(smwarp.done),
                     items);
        return 1;
    }
    return 0;
}
