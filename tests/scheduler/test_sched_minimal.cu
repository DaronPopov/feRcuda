
#include "fercuda/scheduler/scheduler.cuh"
#include <cstdio>
#include <cstdlib>
using namespace fer;

int main() {
    fflush(stdout);
    printf("[1] start\n"); fflush(stdout);
    cudaSetDevice(0);
    printf("[2] device set\n"); fflush(stdout);

    auto* sched = Scheduler::create(0);
    printf("[3] sched created\n"); fflush(stdout);

    sched->launch(128);
    printf("[4] kernel launched\n"); fflush(stdout);

    // shutdown immediately (no work submitted)
    printf("[5] calling shutdown...\n"); fflush(stdout);
    sched->shutdown();
    printf("[6] shutdown returned!\n"); fflush(stdout);

    delete sched;
    printf("[7] done\n"); fflush(stdout);
    return 0;
}
