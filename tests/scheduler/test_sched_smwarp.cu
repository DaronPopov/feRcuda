#include "fercuda/scheduler/scheduler.cuh"

#include <cstdio>
#include <exception>

using namespace fer;

int main() {
    try {
        cudaSetDevice(0);

        auto* sched = Scheduler::create(0);
        sched->launch(128, ExecRegime::SMWARP);

        const int kItems = 4096;
        uint64_t args[SCHED_MAX_ARGS] = {};
        for (int i = 0; i < kItems; ++i) {
            (void)sched->submit(OpCode::ELEMENTWISE, 1, args, 0);
        }

        sched->sync();
        const uint64_t done = sched->ops_completed();
        const uint64_t warp_done = sched->warp_dispatches();
        const uint32_t launch_blocks = sched->launch_blocks();
        const uint32_t resident_blocks = sched->resident_blocks();

        printf("mode=smwarp submitted=%d done=%llu warp_dispatches=%llu launch_blocks=%u resident_blocks=%u\n",
               kItems,
               static_cast<unsigned long long>(done),
               static_cast<unsigned long long>(warp_done),
               launch_blocks,
               resident_blocks);

        sched->shutdown();
        delete sched;

        if (done != static_cast<uint64_t>(kItems)) {
            fprintf(stderr, "expected done=%d got=%llu\n",
                    kItems, static_cast<unsigned long long>(done));
            return 1;
        }
        if (warp_done == 0) {
            fprintf(stderr, "expected warp_dispatches>0\n");
            return 2;
        }
        if (launch_blocks == 0) {
            fprintf(stderr, "expected launch_blocks>0\n");
            return 3;
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "smwarp test runtime error: %s\n", e.what());
        return 99;
    }
}
