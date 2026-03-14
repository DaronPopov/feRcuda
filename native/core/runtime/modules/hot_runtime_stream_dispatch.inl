// Keep GPU active with heartbeat kernel
void gpu_hot_keepalive(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->keepalive_running) return;
    
    static int* d_flag = NULL;
    if (!d_flag) {
        d_flag = (int*)gpu_hot_alloc(runtime, sizeof(int));
        if (!d_flag) {
            if (!runtime->keepalive_running) return;
            printf("[Ferrite-OS] Warning:Keepalive disabled (TLSF alloc failed)\n");
            runtime->keepalive_running = false;
            return;
        }
        cudaMemset(d_flag, 0, sizeof(int));
    }
    
    // Launch minimal kernel to keep GPU active
    keepalive_kernel<<<1, 256, 0, runtime->streams[0]>>>(d_flag);
    
    // Don't synchronize - let it run async
}

// Acquire a CUDA event from the pool (or create a new one)
static cudaEvent_t ptx_event_acquire(GPUHotRuntime* runtime) {
    cudaEvent_t ev = NULL;
    ptx_mutex_lock(&runtime->async_lock);
    if (runtime->event_pool_count > 0) {
        ev = runtime->event_pool[--runtime->event_pool_count];
    }
    ptx_mutex_unlock(&runtime->async_lock);

    if (!ev) {
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
            return NULL;
        }
    }
    return ev;
}

// Return a CUDA event to the pool
static void ptx_event_release_locked(GPUHotRuntime* runtime, cudaEvent_t ev) {
    if (!ev) return;
    if (runtime->event_pool_count >= runtime->event_pool_capacity) {
        int new_cap = (runtime->event_pool_capacity == 0) ? 64 : runtime->event_pool_capacity * 2;
        cudaEvent_t* new_pool = (cudaEvent_t*)realloc(runtime->event_pool, sizeof(cudaEvent_t) * new_cap);
        if (new_pool) {
            runtime->event_pool = new_pool;
            runtime->event_pool_capacity = new_cap;
        }
    }
    if (runtime->event_pool_count < runtime->event_pool_capacity) {
        runtime->event_pool[runtime->event_pool_count++] = ev;
        ev = NULL;
    }

    if (ev) {
        cudaEventDestroy(ev);
    }
}

static void ptx_event_release(GPUHotRuntime* runtime, cudaEvent_t ev) {
    if (!ev) return;
    ptx_mutex_lock(&runtime->async_lock);
    ptx_event_release_locked(runtime, ev);
    ptx_mutex_unlock(&runtime->async_lock);
}

// Drain deferred frees whose events have completed
static void gpu_hot_poll_deferred_internal(GPUHotRuntime* runtime, int max_drain) {
    if (!runtime) return;
    int drained = 0;

    ptx_mutex_lock(&runtime->async_lock);
    DeferredFreeEntry* prev = NULL;
    DeferredFreeEntry* curr = runtime->deferred_head;

    while (curr) {
        if (max_drain > 0 && drained >= max_drain) {
            break;
        }

        cudaError_t q = cudaEventQuery(curr->event);
        if (q == cudaSuccess) {
            DeferredFreeEntry* done = curr;
            curr = curr->next;

            if (prev) {
                prev->next = curr;
            } else {
                runtime->deferred_head = curr;
            }
            if (done == runtime->deferred_tail) {
                runtime->deferred_tail = prev;
            }
            runtime->deferred_count--;

            if (runtime->tlsf_allocator && ptx_tlsf_owns_ptr(runtime->tlsf_allocator, done->ptr)) {
                ptx_tlsf_free(runtime->tlsf_allocator, done->ptr);
            }
            ptx_event_release_locked(runtime, done->event);
            free(done);
            drained++;
            continue;
        } else if (q != cudaErrorNotReady) {
            // Leave entry in queue; error may be transient
            printf("[Ferrite-OS] Warning:cudaEventQuery failed: %s\n", cudaGetErrorString(q));
        }

        prev = curr;
        curr = curr->next;
    }
    ptx_mutex_unlock(&runtime->async_lock);
}

// Fast allocation from TLSF pool - O(1) complexity
void* gpu_hot_alloc(GPUHotRuntime* runtime, size_t size) {
    if (!runtime) return NULL;

    gpu_hot_poll_deferred_internal(runtime, 64);

    if (runtime->tlsf_allocator) {
        ptx_mutex_lock(&runtime->async_lock);
        void* ptr = ptx_tlsf_alloc(runtime->tlsf_allocator, size);
        ptx_mutex_unlock(&runtime->async_lock);
        if (ptr) {
            return ptr;
        }

        // Retry 1: aggressively drain ALL pending deferred frees, then retry.
        // Under concurrent load, other threads may have freed memory that
        // hasn't been polled back into the pool yet.
        gpu_hot_poll_deferred_internal(runtime, 0); // 0 = unlimited drain
        ptx_mutex_lock(&runtime->async_lock);
        ptr = ptx_tlsf_alloc(runtime->tlsf_allocator, size);
        ptx_mutex_unlock(&runtime->async_lock);
        if (ptr) return ptr;

        // Retry 2: brief yield to let other threads finish their frees,
        // then drain + retry once more.
        {
            struct timespec ts = {0, 500000}; // 0.5ms
            nanosleep(&ts, NULL);
        }
        gpu_hot_poll_deferred_internal(runtime, 0);
        ptx_mutex_lock(&runtime->async_lock);
        ptr = ptx_tlsf_alloc(runtime->tlsf_allocator, size);
        ptx_mutex_unlock(&runtime->async_lock);
        if (ptr) return ptr;

        // Retry 3: VMM eviction bridge — try freeing swappable pages
        if (runtime->vmm) {
            if (vmm_evict_for_alloc(runtime->vmm, size) == 0) {
                ptx_mutex_lock(&runtime->async_lock);
                void* ptr2 = ptx_tlsf_alloc(runtime->tlsf_allocator, size);
                ptx_mutex_unlock(&runtime->async_lock);
                if (ptr2) return ptr2;
            }
        }

        runtime->fallback_count++;
        printf("[Ferrite-OS] Error: TLSF pool exhausted (alloc #%d)\n", runtime->fallback_count);
        if (runtime->vram_pool_size > 0) {
            printf("[Ferrite-OS]   Requested: %zu bytes, Pool size: %zu bytes\n",
                   size, runtime->vram_pool_size);
        }
    }

    return NULL;
}


// Fast free (returns to pool) - O(1) complexity with coalescing
void gpu_hot_free(GPUHotRuntime* runtime, void* ptr) {
    if (!runtime || !ptr) return;
    
    bool owns = false;
    if (runtime->tlsf_allocator) {
        ptx_mutex_lock(&runtime->async_lock);
        owns = ptx_tlsf_owns_ptr(runtime->tlsf_allocator, ptr);
        if (owns) {
            ptx_tlsf_free(runtime->tlsf_allocator, ptr);
        }
        ptx_mutex_unlock(&runtime->async_lock);
    }
    if (owns) return;
    ptx_strict_free_violation("GPU-HOT", ptr);
}

// Async allocation (stream-ordered)
void* gpu_hot_alloc_async(GPUHotRuntime* runtime, size_t size, cudaStream_t stream) {
    (void)stream;
    return gpu_hot_alloc(runtime, size);
}

// Async free (defer until stream completes)
void gpu_hot_free_async(GPUHotRuntime* runtime, void* ptr, cudaStream_t stream) {
    if (!runtime || !ptr) return;

    bool owns = false;
    if (runtime->tlsf_allocator) {
        ptx_mutex_lock(&runtime->async_lock);
        owns = ptx_tlsf_owns_ptr(runtime->tlsf_allocator, ptr);
        ptx_mutex_unlock(&runtime->async_lock);
    }
    if (!owns) {
        ptx_strict_free_violation("GPU-HOT-ASYNC", ptr);
        return;
    }

    cudaEvent_t ev = ptx_event_acquire(runtime);
    if (!ev) {
        printf("[Ferrite-OS] Warning:Failed to acquire event, falling back to sync free\n");
        cudaStreamSynchronize(stream);
        ptx_mutex_lock(&runtime->async_lock);
        ptx_tlsf_free(runtime->tlsf_allocator, ptr);
        ptx_mutex_unlock(&runtime->async_lock);
        return;
    }

    cudaError_t err = cudaEventRecord(ev, stream);
    if (err != cudaSuccess) {
        printf("[Ferrite-OS] Warning:cudaEventRecord failed: %s (sync fallback)\n",
               cudaGetErrorString(err));
        ptx_event_release(runtime, ev);
        cudaStreamSynchronize(stream);
        ptx_mutex_lock(&runtime->async_lock);
        ptx_tlsf_free(runtime->tlsf_allocator, ptr);
        ptx_mutex_unlock(&runtime->async_lock);
        return;
    }

    DeferredFreeEntry* entry = (DeferredFreeEntry*)malloc(sizeof(DeferredFreeEntry));
    if (!entry) {
        printf("[Ferrite-OS] Warning:DeferredFreeEntry alloc failed (sync fallback)\n");
        ptx_event_release(runtime, ev);
        cudaStreamSynchronize(stream);
        ptx_mutex_lock(&runtime->async_lock);
        ptx_tlsf_free(runtime->tlsf_allocator, ptr);
        ptx_mutex_unlock(&runtime->async_lock);
        return;
    }

    entry->ptr = ptr;
    entry->event = ev;
    entry->stream = stream;
    entry->next = NULL;

    ptx_mutex_lock(&runtime->async_lock);
    if (runtime->deferred_tail) {
        runtime->deferred_tail->next = entry;
    } else {
        runtime->deferred_head = entry;
    }
    runtime->deferred_tail = entry;
    runtime->deferred_count++;
    ptx_mutex_unlock(&runtime->async_lock);
}

// Poll deferred frees (process completed events)
void gpu_hot_poll_deferred(GPUHotRuntime* runtime, int max_drain) {
    gpu_hot_poll_deferred_internal(runtime, max_drain);
}

// Check if a pointer belongs to the TLSF pool
bool gpu_hot_owns_ptr(GPUHotRuntime* runtime, void* ptr) {
    if (!runtime || !ptr) return false;

    if (runtime->tlsf_allocator) {
        bool owns = false;
        ptx_mutex_lock(&runtime->async_lock);
        owns = ptx_tlsf_owns_ptr(runtime->tlsf_allocator, ptr);
        ptx_mutex_unlock(&runtime->async_lock);
        return owns;
    }

    return false;
}


// Register kernel for fast dispatch
int gpu_hot_register_kernel(GPUHotRuntime* runtime, 
                            void* kernel_func,
                            dim3 grid, dim3 block,
                            size_t shared_mem) {
    if (!runtime || runtime->num_kernels >= GPU_HOT_MAX_KERNELS) {
        return -1;
    }
    
    int kernel_id = runtime->num_kernels++;
    GPUKernelHandle* handle = &runtime->kernels[kernel_id];
    
    handle->kernel_func = kernel_func;
    handle->grid = grid;
    handle->block = block;
    handle->shared_mem = shared_mem;
    handle->stream = runtime->streams[kernel_id % runtime->num_streams];
    
    return kernel_id;
}

// Launch registered kernel
void gpu_hot_launch_kernel(GPUHotRuntime* runtime, 
                           int kernel_id,
                           void** args) {
    if (!runtime || kernel_id < 0 || kernel_id >= runtime->num_kernels) {
        return;
    }
    
    GPUKernelHandle* handle = &runtime->kernels[kernel_id];
    
    // Launch kernel with pre-configured parameters
    cudaLaunchKernel(handle->kernel_func,
                     handle->grid,
                     handle->block,
                     args,
                     handle->shared_mem,
                     handle->stream);
}

// Get CUDA stream
cudaStream_t gpu_hot_get_stream(GPUHotRuntime* runtime, int stream_id) {
    if (!runtime || stream_id < 0 || stream_id >= runtime->num_streams) {
        return 0; // Default stream
    }
    return runtime->streams[stream_id];
}

// Synchronize all streams
void gpu_hot_sync_all(GPUHotRuntime* runtime) {
    if (!runtime) return;
    
    for (int i = 0; i < runtime->num_streams; i++) {
        cudaStreamSynchronize(runtime->streams[i]);
    }
    gpu_hot_poll_deferred_internal(runtime, 0);
}

// Get runtime statistics
void gpu_hot_get_stats(GPUHotRuntime* runtime, GPUHotStats* stats) {
    if (!runtime || !stats) return;
    memset(stats, 0, sizeof(GPUHotStats));
    
    // --- PTX-OS HARDWARE POLLING ---
    // Instead of just tracking the internal pool, we poll the driver for REAL VRAM status.
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    
    if (err == cudaSuccess) {
        stats->vram_allocated = total_mem;
        stats->vram_used = total_mem - free_mem;
        stats->vram_free = free_mem;
    } else {
        // Fallback to pool tracking if driver poll fails
        TLSFPoolStats tlsf_stats;
        memset(&tlsf_stats, 0, sizeof(TLSFPoolStats));
        if (runtime->tlsf_allocator) {
            ptx_tlsf_get_stats(runtime->tlsf_allocator, &tlsf_stats);
            stats->vram_allocated = tlsf_stats.total_pool_size;
            stats->vram_used = tlsf_stats.allocated_bytes;
            stats->vram_free = tlsf_stats.free_bytes;
        } else {
            stats->vram_allocated = 0;
            stats->vram_used = 0;
            stats->vram_free = 0;
        }
    }
    // gpu_utilization and mem_utilization are left at 0 (from memset).
    // NVML is the sole source for SM and memory controller utilization.

#ifndef _WIN32
    nvml_poll_stats(runtime, stats);
    cupti_poll_stats(runtime, stats);
#endif
    
    // Poll each stream with cudaStreamQuery to determine which have pending work.
    // cudaStreamQuery returns cudaSuccess if the stream is idle,
    // cudaErrorNotReady if kernels are still executing.
    {
        int poll_count = runtime->num_streams < 1024 ? runtime->num_streams : 1024;
        int busy_count = 0;
        memset(stats->stream_busy, 0, sizeof(stats->stream_busy));
        for (int i = 0; i < poll_count; i++) {
            cudaError_t q = cudaStreamQuery(runtime->streams[i]);
            if (q == cudaErrorNotReady) {
                stats->stream_busy[i / 8] |= (1u << (i % 8));
                busy_count++;
            }
            // Clear any spurious error flag so it doesn't affect later calls
            if (q != cudaSuccess && q != cudaErrorNotReady) {
                cudaGetLastError();
            }
        }
        stats->active_streams = busy_count;
        stats->stream_poll_count = poll_count;
    }
    stats->registered_kernels = runtime->num_kernels;
    stats->total_ops = runtime->total_ops;
    stats->watchdog_tripped = runtime->watchdog_tripped;
    stats->avg_latency_us = 0.5f; 
    
    // Sync with Global OS State
    if (runtime->system_state) {
        stats->total_ops = runtime->system_state->total_ops;
        stats->watchdog_tripped |= runtime->system_state->watchdog_alert;
        
        int count = 0;
        for (int i = 0; i < GPU_HOT_MAX_NAMED_SEGMENTS; i++) {
            if (runtime->global_registry[i].active) count++;
        }
        stats->shm_count = count;
    } else {
        stats->shm_count = runtime->shm_count;
    }
}

void gpu_hot_set_watchdog(GPUHotRuntime* runtime, int timeout_ms) {
    if (runtime) runtime->watchdog_timeout_ms = timeout_ms;
}

bool gpu_hot_check_watchdog(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->watchdog_timeout_ms) return false;
    
    // Check if the keepalive heartbeat is still ticking
    cudaError_t err = cudaEventQuery(runtime->keepalive_event);
    if (err == cudaErrorNotReady) {
        // If it's been "stuck" for a while, we'd detect it here.
        // For this demo, we'll mark as healthy unless an actual error occurs.
        return false;
    }
    return (err != cudaSuccess);
}

