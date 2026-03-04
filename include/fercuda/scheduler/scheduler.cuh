#pragma once
/*
 * feRcuda :: scheduler.cuh
 *
 * PersistentScheduler — GPU-resident work queue + dispatcher.
 *
 * Pattern inherited from persistant_gpu_os/os_kernel.cu, rewritten with:
 *   - Typed ops (OpCode enum, not raw int)
 *   - Deterministic dispatch order (priority + FIFO tie-break)
 *   - No printf in hot path (only in debug builds)
 *   - Clean shutdown via atomic flag
 *
 * Usage (host side):
 *   auto sched = Scheduler::create();
 *   sched->submit({ OpCode::MATMUL, ... });
 *   sched->launch();       // fires persistent kernel
 *   sched->shutdown();     // signals kernel to exit
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "fercuda/compute/types.cuh"

namespace fer {

// ─── Op Codes ─────────────────────────────────────────────────────────────────

enum class OpCode : uint8_t {
    NOP       = 0,
    ELEMENTWISE = 1,
    MATMUL    = 2,
    NORM      = 3,
    ACTIVATION= 4,
    COPY      = 5,
    SHUTDOWN  = 255,
};

// ─── Work Item ────────────────────────────────────────────────────────────────

static constexpr int SCHED_MAX_ARGS = 6;
static constexpr int SCHED_QUEUE_LEN = 256;  // must be power of 2

struct WorkItem {
    OpCode   op;
    uint8_t  priority;     // 0 = highest
    uint32_t id;           // monotonic, for FIFO tie-break
    bool     done;
    // Generic args: pointers or scalars packed as uint64_t
    uint64_t args[SCHED_MAX_ARGS];
};

// ─── Shared State (lives in device memory) ────────────────────────────────────

struct SchedState {
    // Ring buffer
    WorkItem queue[SCHED_QUEUE_LEN];
    volatile uint32_t head;   // host writes here
    volatile uint32_t tail;   // kernel reads and advances

    // Control
    volatile bool running;
    volatile bool shutdown_req;

    // Counters — volatile so kernel writes bypass L1 and are visible to host DMA
    volatile uint64_t ops_completed;
    volatile uint64_t cycles_idle;
};

// ─── Host-side Handle ─────────────────────────────────────────────────────────

class Scheduler {
public:
    static Scheduler* create(int device = 0, cudaStream_t stream = 0);
    ~Scheduler();

    // Submit a work item (host side). Returns assigned id.
    uint32_t submit(OpCode op, uint8_t priority,
                    uint64_t* args, int nargs);

    // Launch persistent kernel (call once)
    void launch(int threads = 128);

    // Block until all submitted work is done
    void sync();

    // Signal shutdown and wait for kernel to exit
    void shutdown();

    // Read counters
    uint64_t ops_completed() const;

    SchedState* device_state() const { return dstate_; }

private:
    Scheduler(int device, cudaStream_t stream);

    int            device_;
    cudaStream_t   kernel_stream_;
    cudaStream_t   control_stream_;
    SchedState*    dstate_;   // device ptr
    SchedState*    hstate_;   // pinned host mirror
    uint32_t       next_id_;
    bool           launched_;
    bool           owns_kernel_stream_;
    bool           owns_control_stream_;
};

// ─── Device-side Kernel Declaration ───────────────────────────────────────────

__global__ void fer_scheduler_kernel(SchedState* state);

} // namespace fer
