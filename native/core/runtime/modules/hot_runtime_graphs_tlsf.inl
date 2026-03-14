// ============================================================================
// CUDA Graph API - Sub-Microsecond Kernel Launches
// ============================================================================

// Begin graph capture on a stream
int gpu_hot_begin_capture(GPUHotRuntime* runtime, int stream_id, const char* graph_name) {
    if (!runtime || stream_id < 0 || stream_id >= runtime->num_streams) {
        printf("[Ferrite-OS] Error: Invalid stream ID for graph capture\n");
        return -1;
    }
    
    if (runtime->capturing_stream_id >= 0) {
        printf("[Ferrite-OS] Error: Already capturing on stream %d\n", runtime->capturing_stream_id);
        return -1;
    }
    
    if (runtime->num_graphs >= GPU_HOT_MAX_GRAPHS) {
        printf("[Ferrite-OS] Error: Maximum number of graphs reached\n");
        return -1;
    }
    
    cudaStream_t stream = runtime->streams[stream_id];
    
    // Begin stream capture
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        printf("[Ferrite-OS] Error: Failed to begin graph capture: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    runtime->capturing_stream_id = stream_id;
    
    // Prepare graph handle
    int graph_id = runtime->num_graphs;
    GPUGraphHandle* handle = &runtime->graphs[graph_id];
    strncpy(handle->name, graph_name ? graph_name : "unnamed", sizeof(handle->name) - 1);
    handle->is_valid = false;
    
    printf("[Ferrite-OS] Graph capture started on stream %d: '%s'\n", stream_id, handle->name);
    
    return graph_id;
}

// End graph capture and return graph handle
int gpu_hot_end_capture(GPUHotRuntime* runtime, int stream_id) {
    if (!runtime || stream_id < 0 || stream_id >= runtime->num_streams) {
        printf("[Ferrite-OS] Error: Invalid stream ID\n");
        return -1;
    }
    
    if (runtime->capturing_stream_id != stream_id) {
        printf("[Ferrite-OS] Error: Not capturing on stream %d\n", stream_id);
        return -1;
    }
    
    cudaStream_t stream = runtime->streams[stream_id];
    
    int graph_id = runtime->num_graphs;
    GPUGraphHandle* handle = &runtime->graphs[graph_id];
    
    // End stream capture
    cudaError_t err = cudaStreamEndCapture(stream, &handle->graph);
    if (err != cudaSuccess) {
        printf("[Ferrite-OS] Error: Failed to end graph capture: %s\n", cudaGetErrorString(err));
        runtime->capturing_stream_id = -1;
        return -1;
    }
    
    // Instantiate graph for execution
    err = cudaGraphInstantiate(&handle->graph_exec, handle->graph, NULL, NULL, 0);
    if (err != cudaSuccess) {
        printf("[Ferrite-OS] Error: Failed to instantiate graph: %s\n", cudaGetErrorString(err));
        cudaGraphDestroy(handle->graph);
        runtime->capturing_stream_id = -1;
        return -1;
    }
    
    handle->is_valid = true;
    runtime->num_graphs++;
    runtime->capturing_stream_id = -1;
    
    printf("[Ferrite-OS] Graph '%s' captured (ID: %d)\n", handle->name, graph_id);
    
    return graph_id;
}

// Launch a captured graph (0.5us latency vs 5us for cudaLaunchKernel)
void gpu_hot_launch_graph(GPUHotRuntime* runtime, int graph_id, cudaStream_t stream) {
    if (!runtime || graph_id < 0 || graph_id >= runtime->num_graphs) return;
    
    GPUGraphHandle* handle = &runtime->graphs[graph_id];
    if (!handle->is_valid) return;
    
    cudaStream_t launch_stream = stream ? stream : runtime->streams[0];
    cudaError_t err = cudaGraphLaunch(handle->graph_exec, launch_stream);
    
    if (err == cudaSuccess) {
        runtime->total_ops++;
        if (runtime->system_state) {
            InterlockedAdd64((LONG64*)&runtime->system_state->total_ops, 1);
            InterlockedIncrement((LONG*)&runtime->system_state->active_tasks);
        }
        cudaEventRecord(runtime->keepalive_event, launch_stream);
    } else {
        printf("[Ferrite-OS] Error: Failed to launch graph: %s\n", cudaGetErrorString(err));
    }
}

// Destroy a graph
void gpu_hot_destroy_graph(GPUHotRuntime* runtime, int graph_id) {
    if (!runtime || graph_id < 0 || graph_id >= runtime->num_graphs) {
        return;
    }
    
    GPUGraphHandle* handle = &runtime->graphs[graph_id];
    
    if (handle->is_valid) {
        cudaGraphExecDestroy(handle->graph_exec);
        cudaGraphDestroy(handle->graph);
        handle->is_valid = false;
        printf("[Ferrite-OS] Graph '%s' destroyed\n", handle->name);
    }
}


// ============================================================================
// TLSF Pool Monitoring & Health Validation Implementation
// ============================================================================

// Get detailed TLSF pool statistics
void gpu_hot_get_tlsf_stats(GPUHotRuntime* runtime, TLSFPoolStats* stats) {
    if (!runtime || !stats) return;

    memset(stats, 0, sizeof(TLSFPoolStats));
    if (!runtime->tlsf_allocator) {
        stats->fallback_count = runtime->fallback_count;
        return;
    }

    ptx_tlsf_get_stats(runtime->tlsf_allocator, stats);
    stats->fallback_count += runtime->fallback_count;

    // Debug: Print stats being returned
    static int stats_call_count = 0;
    if (ptx_tlsf_debug_enabled() &&
        (++stats_call_count % 10 == 0 || stats_call_count <= 3)) {
        printf("[TLSF-DEBUG] get_stats call #%d: allocs=%llu, frees=%llu, allocated=%zu MB\n",
               stats_call_count,
               (unsigned long long)stats->total_allocations,
               (unsigned long long)stats->total_frees,
               stats->allocated_bytes / 1024 / 1024);
    }
}

// Validate TLSF pool integrity
void gpu_hot_validate_tlsf_pool(GPUHotRuntime* runtime, TLSFHealthReport* report) {
    if (!runtime || !report) return;
    
    memset(report, 0, sizeof(TLSFHealthReport));
    if (!runtime->tlsf_allocator) {
        report->is_valid = false;
        snprintf(report->error_messages[report->error_count++], 256,
                "TLSF allocator not initialized");
        return;
    }

    ptx_tlsf_validate(runtime->tlsf_allocator, report);
}

// Check if allocation will succeed without fallback
bool gpu_hot_can_allocate(GPUHotRuntime* runtime, size_t size) {
    if (!runtime) return false;

    if (runtime->tlsf_allocator) {
        return ptx_tlsf_can_allocate(runtime->tlsf_allocator, size);
    }
    return false;
}

// Get largest allocatable block size
size_t gpu_hot_get_max_allocatable(GPUHotRuntime* runtime) {
    if (!runtime) return 0;

    if (runtime->tlsf_allocator) {
        return ptx_tlsf_get_max_allocatable(runtime->tlsf_allocator);
    }
    return 0;
}

// Defragment pool (merge all adjacent free blocks)
void gpu_hot_defragment_pool(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->tlsf_allocator) return;
    ptx_tlsf_defragment(runtime->tlsf_allocator);
}

// Print detailed pool visualization
void gpu_hot_print_pool_map(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->tlsf_allocator) return;
    ptx_tlsf_print_pool_map(runtime->tlsf_allocator);
}

// Set warning threshold
void gpu_hot_set_warning_threshold(GPUHotRuntime* runtime, float threshold_percent) {
    if (!runtime) return;

    if (runtime->tlsf_allocator) {
        ptx_tlsf_set_warning_threshold(runtime->tlsf_allocator, threshold_percent / 100.0f);
        printf("[Ferrite-OS] Warning threshold set to %.1f%%\n", threshold_percent);
    }
}

// Enable/disable automatic defragmentation
void gpu_hot_set_auto_defrag(GPUHotRuntime* runtime, bool enable) {
    if (!runtime) return;

    if (runtime->tlsf_allocator) {
        ptx_tlsf_set_auto_defrag(runtime->tlsf_allocator, enable);
        printf("[Ferrite-OS] Auto-defragmentation %s\n", enable ? "enabled" : "disabled");
    }
}

