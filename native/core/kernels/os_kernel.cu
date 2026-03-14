#include <cuda_runtime.h>
#include <stdio.h>
#include "gpu/gpu_hot_runtime.h"

// ============================================================================
// Device-Global System State Pointer
// Set by the persistent kernel at boot so any device code (including child
// kernels launched via CDP) can find the scheduler's task queue.
// ============================================================================

__device__ PTXSystemState* d_ptx_system_state = nullptr;

// ============================================================================
// Device-Side Task Submission (CUDA Dynamic Parallelism)
//
// This is the missing link that closes the "kernel launching kernel" loop.
// Any GPU kernel can call this function to enqueue a new task into the
// scheduler's task queue. The persistent OS kernel picks it up and dispatches
// it — enabling fully autonomous GPU-side program scheduling.
//
// Uses CUDA atomics instead of GCC __sync builtins (which are host-only).
// ============================================================================

__device__ int ptx_device_submit_task(
    PTXSystemState* state,
    uint32_t opcode,
    uint32_t priority,
    void** args
) {
    if (!state) return -1;

    PTXTaskQueue* queue = &state->queue;

    // Spinlock acquire via atomicCAS (compare-and-swap 0 → 1)
    while (atomicCAS((unsigned int*)&queue->lock, 0u, 1u) != 0u) {
        // Spin — the lock is held briefly by the scheduler or another enqueuer
    }

    uint32_t head = queue->head;
    uint32_t next_head = (head + 1) % PTX_MAX_QUEUE_SIZE;
    if (next_head == queue->tail) {
        // Queue full — release lock and fail
        atomicExch((unsigned int*)&queue->lock, 0u);
        return -1;
    }

    PTXOSTask* task = &queue->tasks[head];
    task->task_id = head;
    task->opcode = opcode;
    task->priority = priority;
    task->active = true;
    task->completed = false;
    if (args) {
        for (int i = 0; i < PTX_MAX_TASK_ARGS; i++) {
            task->args[i] = args[i];
        }
    } else {
        for (int i = 0; i < PTX_MAX_TASK_ARGS; i++) {
            task->args[i] = nullptr;
        }
    }

    // Ensure task fields are visible before advancing head
    __threadfence();
    queue->head = next_head;

    // Ensure head update is visible before releasing the lock
    __threadfence();
    atomicExch((unsigned int*)&queue->lock, 0u);

    return (int)task->task_id;
}

// Convenience overload using the device-global system state
__device__ int ptx_device_submit_task(
    uint32_t opcode,
    uint32_t priority,
    void** args
) {
    return ptx_device_submit_task(d_ptx_system_state, opcode, priority, args);
}

// ============================================================================
// PTX-OS Persistent Kernel
// This kernel stays resident in the GPU, polling for tasks and managing
// system state without CPU intervention.
// ============================================================================

__global__ void ptx_os_kernel(PTXSystemState* state) {
    uint32_t tid = threadIdx.x; // We assume 1D block for OS management

    // Boot sequence
    if (tid == 0) {
        state->kernel_running = true;
        state->shutdown_requested = false;

        // Publish system state to device-global so CDP child kernels can find it
        d_ptx_system_state = state;

        printf("[GPU-OS] Kernel Life-Cycle Initialized. VRAM OS is now ACTIVE.\n");
        printf("[GPU-OS] Device-side task submission ENABLED (CDP loop active).\n");
    }
    
    __syncthreads();
    
    uint64_t iterations = 0;
    
    // The "infinite" loop of the OS
    while (!state->shutdown_requested) {
        // Standard block sync
        __syncthreads();
        
        // Thread 0: Primary Scheduler / Task Dispatcher
        if (tid == 0) {
            // Check Task Queue
            if (state->queue.head != state->queue.tail) {
                int best_idx = -1;
                int min_priority = 256;
                
                uint32_t current_tail = state->queue.tail;
                uint32_t current_head = state->queue.head;
                
                for (uint32_t i = current_tail; i != current_head; i++) {
                    PTXOSTask* candidate = &state->queue.tasks[i % PTX_MAX_QUEUE_SIZE];
                    if (candidate->active && !candidate->completed) {
                        if (candidate->priority < min_priority) {
                            min_priority = candidate->priority;
                            best_idx = i % PTX_MAX_QUEUE_SIZE;
                            if (min_priority == 0) break; // Realtime optimization
                        }
                    }
                }

                if (best_idx != -1) {
                    PTXOSTask* task = &state->queue.tasks[best_idx];
                    state->active_priority_level = task->priority;
                    
                    switch (task->opcode) {
                        case 0: // NOP
                            break;
                            
                        case 1: // COMPUTE - dispatch kernel via function pointer in args[0]
                            {
                                typedef void (*compute_fn_t)(void**, int);
                                compute_fn_t fn = (compute_fn_t)task->args[0];
                                if (fn) {
                                    fn<<<1, 256>>>(&task->args[1], task->priority);
                                }
                            }
                            break;
                            
                        case 3: // SHUTDOWN
                            state->shutdown_requested = true;
                            break;

                        case 4: // SWAP_IN (Virtual Memory Manager)
                            printf("[GPU-VMM] Swapping task_id %d back to Resident VRAM\n", task->task_id);
                            // Signal host to perform VMM swap via signal_mask bit 2
                            atomicOr((unsigned long long*)&state->signal_mask, 0x4ULL);
                            break;

                        case 5: // VFS_MOUNT (Tensor Filesystem)
                            state->fs_node_count++;
                            printf("[PTX-FS] Mounted Segment Index: %d | Nodes Active: %d\n", (int)(size_t)task->args[0], state->fs_node_count);
                            break;

                        case 6: // INTERRUPT (Simulated hardware interrupt)
                            printf("[PTX-INT] Software Interrupt Generated: %p\n", task->args[0]);
                            atomicOr((unsigned long long*)&state->signal_mask, (unsigned long long)task->args[0]);
                            break;

                        case 7: // LAUNCH_KERNEL (Recursive Task Dispatch / CDP)
                            {
                                 // Cast the task's argument buffer directly to a PTXKernelLaunch descriptor
                                 PTXKernelLaunch* launch = (PTXKernelLaunch*)&task->args[0];
                                 if (launch && launch->kernel_func) {
                                     printf("[PTX-CDP] Recursive Launch: Func=%p | Grid=(%d,%d) | Block=%d\n", 
                                            launch->kernel_func, launch->grid.x, launch->grid.y, launch->block.x);
                                     
                                     typedef void (*kernel_ptr_t)(void**);
                                     kernel_ptr_t func = (kernel_ptr_t)launch->kernel_func;
                                     
                                     // Pass the address of the inline argument array
                                     func<<<launch->grid, launch->block, launch->shared_mem, launch->stream>>>(launch->arg_values);
                                 }
                            }
                            break;
                            
                        default:
                            printf("[GPU-OS] Unknown Opcode: %d\n", task->opcode);
                            break;
                    }
                    
                    task->completed = true;
                    task->active = false;
                    state->total_ops++;
                    
                    // Advance tail ONLY if we processed the oldest task
                    if (best_idx == (current_tail % PTX_MAX_QUEUE_SIZE)) {
                        state->queue.tail++;
                    }
                }
            }
            
            // --- Feature 4: Async Signal/Interrupt Processing ---
            if (state->signal_mask != 0) {
                uint64_t signals = atomicExch((unsigned long long*)&state->signal_mask, 0);
                if (signals & 0x1) {
                    printf("[PTX-OS] Heartbeat Pulse Received from Host/App\n");
                }
                if (signals & 0x2) {
                    printf("[PTX-OS] EMERGENCY FLUSH REQUESTED. Clearing non-critical pools.\n");
                }
                state->interrupt_cnt++;
            }
        }

        // Feature: OS Health Debugger
        if (tid == 0 && iterations % 10000000 == 0) {
            printf("[GPU-OS-DEBUG] Head: %d | Tail: %d | Total Ops: %lld\n", 
                   state->queue.head, state->queue.tail, state->total_ops);
        }

        // Small delay to prevent saturation of the command bridge
        for (int i = 0; i < 500; i++) {
             __threadfence_system(); // Ensure visibility across GPU and CPU
        }
        iterations++;
    }
    
    if (tid == 0) {
        state->kernel_running = false;
        printf("[GPU-OS] Kernel Shutdown. Returning control to hardware.\n");
    }
}

// ============================================================================
// Internal Test Kernels for CDP
// ============================================================================

__global__ void k_test_cdp(void** args) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float* data = (float*)args[0];
        float value = *(float*)&args[1];
        printf("[PTX-APP] CDP Kernel Executing! Writing %f to %p\n", value, data);
        if (data) *data = value;
    }
}

// Device-side pointer to the test kernel (for CDP retrieval)
__device__ void (*d_test_kernel_ptr)(void**) = k_test_cdp;

extern "C" void* ptx_get_test_kernel_ptr_internal() {
    void* h_ptr;
    cudaMemcpyFromSymbol(&h_ptr, d_test_kernel_ptr, sizeof(void*));
    return h_ptr;
}

// ============================================================================
// Self-Scheduling CDP Kernel
//
// This kernel demonstrates the full "kernel launching kernel through the
// scheduler" loop.  When dispatched by the persistent OS kernel (opcode 1),
// it increments a counter, then — if more iterations remain — enqueues
// itself back into the task queue.  The scheduler picks it up and launches
// it again via CDP, creating a fully autonomous GPU-side execution loop:
//
//   Host submit → Scheduler → CDP child → device_submit → Scheduler → ...
//
// Args layout (as received from opcode 1 — &task->args[1]):
//   args[0] = PTXSystemState*  (system state for re-enqueue)
//   args[1] = int*             (device counter — atomicAdd target)
//   args[2] = int              (max iterations, cast to void*)
// ============================================================================

__global__ void k_cdp_self_schedule(void** args, int priority) {
    if (threadIdx.x != 0) return;

    PTXSystemState* state = (PTXSystemState*)args[0];
    int*  counter   = (int*)args[1];
    int   max_iters = (int)(size_t)args[2];

    int current = atomicAdd(counter, 1);
    printf("[PTX-CDP-LOOP] Iteration %d / %d  (priority=%d)\n",
           current + 1, max_iters, priority);

    if (current + 1 < max_iters) {
        // Re-enqueue self via opcode 1 (COMPUTE)
        // args layout for opcode 1:
        //   [0] = function pointer  (kernel to launch)
        //   [1] = args[0] inside kernel  → system state
        //   [2] = args[1] inside kernel  → counter
        //   [3] = args[2] inside kernel  → max_iters
        void* new_args[PTX_MAX_TASK_ARGS];
        for (int i = 0; i < PTX_MAX_TASK_ARGS; i++) new_args[i] = nullptr;

        // We need the device-side function pointer for k_cdp_self_schedule.
        // In CDP, we can take the address of a __global__ function directly.
        new_args[0] = (void*)k_cdp_self_schedule;
        new_args[1] = (void*)state;
        new_args[2] = (void*)counter;
        new_args[3] = (void*)(size_t)max_iters;

        int task_id = ptx_device_submit_task(state, 1, (uint32_t)priority, new_args);
        if (task_id >= 0) {
            printf("[PTX-CDP-LOOP] Enqueued next iteration (task_id=%d)\n", task_id);
        } else {
            printf("[PTX-CDP-LOOP] ERROR: Queue full, could not enqueue\n");
        }
    } else {
        printf("[PTX-CDP-LOOP] Complete! All %d iterations executed.\n", max_iters);
    }
}

// Device-side pointer for host retrieval
__device__ void (*d_cdp_self_schedule_ptr)(void**, int) = k_cdp_self_schedule;

// ============================================================================
// Host-Callable CDP Test
//
// Boots the persistent kernel (if not already running), allocates a device
// counter, submits the first self-scheduling task, then polls until all
// iterations complete.  Returns the number of iterations executed, or
// a negative error code.
// ============================================================================

// Forward declaration (defined in hot_runtime_shared_context.inl)
extern "C" PTXSystemState* gpu_hot_get_system_state(GPUHotRuntime* runtime);

extern "C" int ptx_cdp_test_recursive(GPUHotRuntime* runtime, int iterations) {
    if (!runtime || iterations <= 0) return -1;

    // Get device-accessible system state (zero-copy mapped pointer)
    PTXSystemState* d_state = gpu_hot_get_system_state(runtime);
    if (!d_state) {
        printf("[PTX-CDP] ERROR: Could not resolve device system state\n");
        return -2;
    }

    // Retrieve device function pointer for the self-scheduling kernel
    void* kernel_ptr = nullptr;
    cudaMemcpyFromSymbol(&kernel_ptr, d_cdp_self_schedule_ptr, sizeof(void*));
    if (!kernel_ptr) {
        printf("[PTX-CDP] ERROR: Could not retrieve device kernel pointer\n");
        return -3;
    }

    // Allocate device counter via TLSF
    void* counter_mem = gpu_hot_alloc(runtime, sizeof(int));
    if (!counter_mem) {
        printf("[PTX-CDP] ERROR: Could not allocate device counter\n");
        return -4;
    }
    int* d_counter = (int*)counter_mem;
    cudaMemset(d_counter, 0, sizeof(int));

    printf("[PTX-CDP] Starting recursive self-scheduling test: %d iterations\n", iterations);

    // Submit first task: opcode 1 (COMPUTE)
    //   args[0] = function pointer
    //   args[1..] = kernel arguments (become args[0..] inside the kernel)
    void* args[PTX_MAX_TASK_ARGS] = {0};
    args[0] = kernel_ptr;
    args[1] = (void*)d_state;
    args[2] = (void*)d_counter;
    args[3] = (void*)(size_t)iterations;

    int task_id = ptx_os_submit_task(runtime, 1, PTX_PRIORITY_NORMAL, args);
    if (task_id < 0) {
        printf("[PTX-CDP] ERROR: Failed to submit initial task\n");
        gpu_hot_free(runtime, counter_mem);
        return -5;
    }

    printf("[PTX-CDP] First task submitted (task_id=%d), waiting for completion...\n", task_id);

    // Poll until all iterations are done (with timeout)
    int result = 0;
    int timeout_ms = iterations * 2000 + 5000; // generous timeout
    for (int t = 0; t < timeout_ms; t++) {
        cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
        if (result >= iterations) break;

        // Sleep 1ms between polls
        struct timespec ts = {0, 1000000};
        nanosleep(&ts, nullptr);
    }

    printf("[PTX-CDP] Recursive test result: %d / %d iterations completed\n", result, iterations);

    gpu_hot_free(runtime, counter_mem);
    return result;
}

// ============================================================================
// Persistent Kernel Launcher
// ============================================================================

extern "C" void ptx_os_launch_persistent_kernel(GPUHotRuntime* runtime, PTXSystemState* d_state) {
    int blockSize = 256;
    int osBlocks = 1;

    printf("[PTX-OS] Launching Persistent Kernel...\n");
    printf("[PTX-OS] Configuration: %d Blocks x %d Threads/Block\n", osBlocks, blockSize);

    // Launch the Persistent Kernel
    ptx_os_kernel<<<osBlocks, blockSize, 0, 0>>>(d_state);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("[PTX-OS] [ERROR] Kernel Launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("[PTX-OS] OS Kernel is now resident in VRAM.\n");
}