#include "fercuda/scheduler/scheduler.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace fer {

static void cuda_check(cudaError_t e, const char* loc) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(loc) + ": " + cudaGetErrorString(e));
}
#define CK(x) cuda_check((x), #x)

// ─── Persistent Kernel ────────────────────────────────────────────────────────
//
// One CTA, N threads. Thread 0 is the scheduler; others are worker slots.
// Deterministic dispatch: lowest priority wins; FIFO on tie (monotonic id).
//
__global__ void fer_scheduler_kernel(SchedState* st) {
    const uint32_t tid = threadIdx.x;

    if (tid == 0) {
        st->running      = true;
        st->ops_completed = 0;
        st->cycles_idle   = 0;
    }
    __syncthreads();

    while (!st->shutdown_req) {
        __syncthreads();

        if (tid == 0) {
            uint32_t head = st->head;
            uint32_t tail = st->tail;

            if (head == tail) {
                // Queue empty — idle
                st->cycles_idle++;
            } else {
                // Pick best item: lowest priority, lowest id on tie
                int      best = -1;
                uint8_t  best_pri = 255;
                uint32_t best_id  = 0xFFFFFFFFu;

                for (uint32_t i = tail; i != head; i++) {
                    uint32_t idx = i & (SCHED_QUEUE_LEN - 1);
                    // Read via volatile pointer to bypass L1.
                    // WorkItem entries are written by the host via H2D DMA which
                    // lands in L2; the kernel's L1 may hold a stale cache line
                    // (e.g. from a neighboring item on the same 128-byte line).
                    // Without volatile, the kernel can misread op=NOP instead of
                    // SHUTDOWN and "dispatch" the item as a NOP, advancing tail
                    // past the SHUTDOWN entry and spinning in idle forever.
                    volatile WorkItem* vw = (volatile WorkItem*)&st->queue[idx];
                    if (vw->done) continue;
                    OpCode op = vw->op;
                    if (op == OpCode::SHUTDOWN) {
                        st->shutdown_req = true;
                        vw->done = true;
                        goto next_cycle;
                    }
                    uint8_t  pri = vw->priority;
                    uint32_t id  = vw->id;
                    if (pri < best_pri ||
                        (pri == best_pri && id < best_id)) {
                        best     = (int)idx;
                        best_pri = pri;
                        best_id  = id;
                    }
                }

                if (best >= 0) {
                    volatile WorkItem* vw = (volatile WorkItem*)&st->queue[best];
                    // Volatile write: bypasses L1, goes directly to L2 so the
                    // host DMA read of ops_completed is coherent with done flag.
                    vw->done = true;
                    __threadfence();
                    st->ops_completed++;
                    // Advance tail past consumed items (kernel wrote done, so
                    // L1 is valid here — volatile read for consistency).
                    while (st->tail != st->head &&
                           ((volatile WorkItem*)&st->queue[
                               st->tail & (SCHED_QUEUE_LEN-1)])->done) {
                        st->tail++;
                    }
                }
                next_cycle:;
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        st->running = false;
    }
}

// ─── Host Handle ──────────────────────────────────────────────────────────────

Scheduler::Scheduler(int device, cudaStream_t stream)
    : device_(device),
      kernel_stream_(stream),
      control_stream_(0),
      next_id_(0),
      launched_(false),
      owns_kernel_stream_(false),
      owns_control_stream_(false)
{
    CK(cudaSetDevice(device_));
    if (kernel_stream_ == 0) {
        CK(cudaStreamCreateWithFlags(&kernel_stream_, cudaStreamNonBlocking));
        owns_kernel_stream_ = true;
    }
    CK(cudaStreamCreateWithFlags(&control_stream_, cudaStreamNonBlocking));
    owns_control_stream_ = true;
    CK(cudaMalloc(&dstate_, sizeof(SchedState)));
    CK(cudaMallocHost(&hstate_, sizeof(SchedState)));
    CK(cudaMemset(dstate_, 0, sizeof(SchedState)));
}

Scheduler* Scheduler::create(int device, cudaStream_t stream) {
    return new Scheduler(device, stream);
}

Scheduler::~Scheduler() {
    if (launched_) shutdown();
    cudaFree(dstate_);
    cudaFreeHost(hstate_);
    if (owns_control_stream_ && control_stream_) cudaStreamDestroy(control_stream_);
    if (owns_kernel_stream_ && kernel_stream_) cudaStreamDestroy(kernel_stream_);
}

uint32_t Scheduler::submit(OpCode op, uint8_t priority,
                           uint64_t* args, int nargs)
{
    // Read current head from device (pinned mirror)
    uint32_t head;
    CK(cudaMemcpyAsync(&head,
                       const_cast<uint32_t*>(&dstate_->head),
                       sizeof(uint32_t),
                       cudaMemcpyDeviceToHost,
                       control_stream_));
    CK(cudaStreamSynchronize(control_stream_));

    uint32_t idx = head & (SCHED_QUEUE_LEN - 1);
    WorkItem item{};
    item.op       = op;
    item.priority = priority;
    item.id       = next_id_++;
    item.done     = false;
    for (int i = 0; i < nargs && i < SCHED_MAX_ARGS; i++)
        item.args[i] = args[i];

    // Write item then bump head
    CK(cudaMemcpyAsync(&dstate_->queue[idx],
                       &item,
                       sizeof(WorkItem),
                       cudaMemcpyHostToDevice,
                       control_stream_));
    uint32_t new_head = head + 1;
    CK(cudaMemcpyAsync(const_cast<uint32_t*>(&dstate_->head),
                       &new_head,
                       sizeof(uint32_t),
                       cudaMemcpyHostToDevice,
                       control_stream_));
    CK(cudaStreamSynchronize(control_stream_));

    return item.id;
}

void Scheduler::launch(int threads) {
    fer_scheduler_kernel<<<1, threads, 0, kernel_stream_>>>(dstate_);
    launched_ = true;
}

void Scheduler::sync() {
    // Poll ops_completed via DMA until the kernel has processed all submitted work.
    // Yield between polls to avoid burning the host CPU in a tight spin.
    uint64_t done = 0;
    do {
        std::this_thread::yield();
        CK(cudaMemcpyAsync(&done,
                           const_cast<uint64_t*>(
                               reinterpret_cast<volatile uint64_t*>(&dstate_->ops_completed)),
                           sizeof(uint64_t),
                           cudaMemcpyDeviceToHost,
                           control_stream_));
        CK(cudaStreamSynchronize(control_stream_));
    } while (done < next_id_);
}

void Scheduler::shutdown() {
    uint64_t args[SCHED_MAX_ARGS] = {};
    submit(OpCode::SHUTDOWN, 0, args, 0);
    CK(cudaStreamSynchronize(kernel_stream_));
    launched_ = false;
}

uint64_t Scheduler::ops_completed() const {
    uint64_t v;
    cudaMemcpyAsync(&v,
                    const_cast<uint64_t*>(
                        reinterpret_cast<volatile uint64_t*>(&dstate_->ops_completed)),
                    sizeof(uint64_t), cudaMemcpyDeviceToHost, control_stream_);
    cudaStreamSynchronize(control_stream_);
    return v;
}

} // namespace fer
