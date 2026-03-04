
#include "fercuda/scheduler/scheduler.cuh"
#include <cstdio>
using namespace fer;

int main() {
    printf("[1] start\n"); fflush(stdout);
    cudaSetDevice(0);

    auto* sched = Scheduler::create(0);
    sched->launch(128);
    printf("[2] launched\n"); fflush(stdout);

    // Submit 3 work items
    uint64_t args[SCHED_MAX_ARGS] = {};
    for (int i = 0; i < 3; i++) {
        uint32_t id = sched->submit(OpCode::ELEMENTWISE, 1, args, 0);
        printf("[3] submitted id=%u\n", id); fflush(stdout);
    }

    // sync: wait for all items to be processed
    printf("[4] calling sync()...\n"); fflush(stdout);
    sched->sync();
    printf("[5] sync() returned! ops_completed=%llu\n",
           (unsigned long long)sched->ops_completed()); fflush(stdout);

    printf("[6] calling shutdown...\n"); fflush(stdout);
    sched->shutdown();
    printf("[7] shutdown returned!\n"); fflush(stdout);

    delete sched;
    printf("[8] done\n"); fflush(stdout);
    return 0;
}
