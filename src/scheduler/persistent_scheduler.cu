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
__device__ inline void execute_work_item(volatile WorkItem* vw) {
    const OpCode op = vw->op;
    if (op == OpCode::ELEMENTWISE) {
        uint32_t iters = static_cast<uint32_t>(vw->args[0]);
        if (iters == 0) iters = 128;
        float x = 0.001f * static_cast<float>((vw->id & 1023u) + 1u);
        for (uint32_t i = 0; i < iters; ++i) {
            x = x * 1.00010002f + 0.000099f;
        }
        float* out = reinterpret_cast<float*>(static_cast<uintptr_t>(vw->args[1]));
        uint32_t mask = static_cast<uint32_t>(vw->args[2]);
        if (out && mask) {
            out[vw->id & mask] = x;
        }
    }
}

__global__ void fer_scheduler_kernel(SchedState* st) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const bool smwarp = (st->regime == static_cast<uint32_t>(ExecRegime::SMWARP));

    if (tid == 0) {
        atomicAdd(reinterpret_cast<unsigned int*>(const_cast<uint32_t*>(&st->resident_blocks)), 1u);
        if (blockIdx.x == 0) {
            st->running = true;
            st->ops_completed = 0;
            st->cycles_idle = 0;
            st->warp_dispatches = 0;
        }
    }
    __syncthreads();

    while (!st->shutdown_req) {
        if (smwarp) {
            uint32_t claim = 0xFFFFFFFFu;
            if (lane == 0) {
                while (true) {
                    uint32_t tail = st->tail;
                    uint32_t head = st->head;
                    if (tail == head) break;
                    if (atomicCAS(reinterpret_cast<unsigned int*>(const_cast<uint32_t*>(&st->tail)),
                                  tail, tail + 1) == tail) {
                        claim = tail;
                        break;
                    }
                }
            }
            claim = __shfl_sync(0xFFFFFFFFu, claim, 0);
            if (claim == 0xFFFFFFFFu) {
                if (lane == 0 && warp == 0) {
                    atomicAdd(reinterpret_cast<unsigned long long*>(const_cast<uint64_t*>(&st->cycles_idle)), 1ULL);
                }
                __syncthreads();
                continue;
            }

            uint32_t idx = claim & (SCHED_QUEUE_LEN - 1);
            volatile WorkItem* vw = (volatile WorkItem*)&st->queue[idx];
            if (!vw->done && lane == 0) {
                OpCode op = vw->op;
                if (op == OpCode::SHUTDOWN) {
                    vw->done = true;
                    __threadfence();
                    st->shutdown_req = true;
                } else {
                    execute_work_item(vw);
                    vw->done = true;
                    __threadfence();
                    atomicAdd(reinterpret_cast<unsigned long long*>(const_cast<uint64_t*>(&st->ops_completed)), 1ULL);
                    atomicAdd(reinterpret_cast<unsigned long long*>(const_cast<uint64_t*>(&st->warp_dispatches)), 1ULL);
                }
            }
            __syncthreads();
            continue;
        }

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
                    execute_work_item(vw);
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
        uint32_t left = atomicSub(reinterpret_cast<unsigned int*>(const_cast<uint32_t*>(&st->resident_blocks)), 1u) - 1u;
        if (left == 0u) st->running = false;
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
    uint32_t head = 0;
    uint32_t tail = 0;
    while (true) {
        CK(cudaMemcpyAsync(&head,
                           const_cast<uint32_t*>(&dstate_->head),
                           sizeof(uint32_t),
                           cudaMemcpyDeviceToHost,
                           control_stream_));
        CK(cudaMemcpyAsync(&tail,
                           const_cast<uint32_t*>(&dstate_->tail),
                           sizeof(uint32_t),
                           cudaMemcpyDeviceToHost,
                           control_stream_));
        CK(cudaStreamSynchronize(control_stream_));
        if ((head - tail) < static_cast<uint32_t>(SCHED_QUEUE_LEN - 1)) break;
        std::this_thread::yield();
    }

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

void Scheduler::launch(int threads, ExecRegime regime) {
    int sms = 1;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_));
    const uint32_t blocks = (regime == ExecRegime::SMWARP) ? static_cast<uint32_t>(sms) : 1u;
    const uint32_t regime_raw = static_cast<uint32_t>(regime);
    const uint32_t zero_u32 = 0;
    CK(cudaMemcpyAsync(const_cast<uint32_t*>(&dstate_->regime), &regime_raw,
                       sizeof(uint32_t), cudaMemcpyHostToDevice, control_stream_));
    CK(cudaMemcpyAsync(const_cast<uint32_t*>(&dstate_->launch_blocks), &blocks,
                       sizeof(uint32_t), cudaMemcpyHostToDevice, control_stream_));
    CK(cudaMemcpyAsync(const_cast<uint32_t*>(&dstate_->resident_blocks), &zero_u32,
                       sizeof(uint32_t), cudaMemcpyHostToDevice, control_stream_));
    CK(cudaStreamSynchronize(control_stream_));
    fer_scheduler_kernel<<<blocks, threads, 0, kernel_stream_>>>(dstate_);
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

uint64_t Scheduler::warp_dispatches() const {
    uint64_t v = 0;
    cudaMemcpyAsync(&v,
                    const_cast<uint64_t*>(
                        reinterpret_cast<volatile uint64_t*>(&dstate_->warp_dispatches)),
                    sizeof(uint64_t), cudaMemcpyDeviceToHost, control_stream_);
    cudaStreamSynchronize(control_stream_);
    return v;
}

uint32_t Scheduler::resident_blocks() const {
    uint32_t v = 0;
    cudaMemcpyAsync(&v,
                    const_cast<uint32_t*>(
                        reinterpret_cast<volatile uint32_t*>(&dstate_->resident_blocks)),
                    sizeof(uint32_t), cudaMemcpyDeviceToHost, control_stream_);
    cudaStreamSynchronize(control_stream_);
    return v;
}

uint32_t Scheduler::launch_blocks() const {
    uint32_t v = 0;
    cudaMemcpyAsync(&v,
                    const_cast<uint32_t*>(
                        reinterpret_cast<volatile uint32_t*>(&dstate_->launch_blocks)),
                    sizeof(uint32_t), cudaMemcpyDeviceToHost, control_stream_);
    cudaStreamSynchronize(control_stream_);
    return v;
}

} // namespace fer
