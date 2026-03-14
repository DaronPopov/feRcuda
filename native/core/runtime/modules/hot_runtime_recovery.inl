// Shutdown GPU Hot Runtime
void gpu_hot_shutdown(GPUHotRuntime* runtime) {
    if (!runtime) return;

    printf("[Ferrite-OS] Shutting down GPU Hot Runtime...\n");

    // Deactivate context hook so driver calls during shutdown pass through
    if (ptx_context_hook_set_ptx_active) {
        ptx_context_hook_set_ptx_active(false);
    }

    runtime->keepalive_running = false;

    // Ensure all stream work is complete before draining deferred frees
    cudaDeviceSynchronize();
    gpu_hot_poll_deferred_internal(runtime, 0);
    // Force free any remaining deferred entries
    ptx_mutex_lock(&runtime->async_lock);
    DeferredFreeEntry* entry = runtime->deferred_head;
    runtime->deferred_head = NULL;
    runtime->deferred_tail = NULL;
    runtime->deferred_count = 0;
    ptx_mutex_unlock(&runtime->async_lock);
    while (entry) {
        DeferredFreeEntry* next = entry->next;
        if (runtime->tlsf_allocator && ptx_tlsf_owns_ptr(runtime->tlsf_allocator, entry->ptr)) {
            ptx_tlsf_free(runtime->tlsf_allocator, entry->ptr);
        }
        if (entry->event) {
            cudaEventDestroy(entry->event);
        }
        free(entry);
        entry = next;
    }
    
    // Destroy streams
    for (int i = 0; i < runtime->num_streams; i++) {
        cudaStreamDestroy(runtime->streams[i]);
    }
    
    // Destroy Enhanced TLSF Allocator
    if (runtime->tlsf_allocator) {
        printf("[Ferrite-OS] Destroying enhanced TLSF allocator...\n");
        ptx_tlsf_destroy(runtime->tlsf_allocator);
        runtime->tlsf_allocator = NULL;
    }
    
    // Free VRAM pool (TLSF structures are inside the pool)
    if (runtime->vram_pool) {
        ptx_driver_free(runtime->vram_pool);
    }
    
    // Destroy CUDA Graphs
    for (int i = 0; i < runtime->num_graphs; i++) {
        if (runtime->graphs[i].is_valid) {
            cudaGraphExecDestroy(runtime->graphs[i].graph_exec);
            cudaGraphDestroy(runtime->graphs[i].graph);
        }
    }

    
    // Destroy event
    if (runtime->keepalive_event) {
        cudaEventDestroy(runtime->keepalive_event);
    }

    // Destroy pooled events
    ptx_mutex_lock(&runtime->async_lock);
    for (int i = 0; i < runtime->event_pool_count; i++) {
        cudaEventDestroy(runtime->event_pool[i]);
    }
    free(runtime->event_pool);
    runtime->event_pool = NULL;
    runtime->event_pool_count = 0;
    runtime->event_pool_capacity = 0;
    ptx_mutex_unlock(&runtime->async_lock);
    ptx_mutex_destroy(&runtime->async_lock);
    
    // Close global registry
    if (runtime->system_state) {
        InterlockedDecrement((LONG*)&runtime->system_state->active_processes);
    }
#ifdef _WIN32
    if (runtime->global_registry) UnmapViewOfFile(runtime->global_registry);
    if (runtime->shm_handle) CloseHandle(runtime->shm_handle);
#else
    if (runtime->global_registry) munmap(runtime->global_registry, sizeof(GPURegistryEntry) * GPU_HOT_MAX_NAMED_SEGMENTS + sizeof(PTXSystemState));
    // Linux: shm file descriptor already closed after mmap, no cleanup needed
#endif
    
    // Release primary context
    if (runtime->cu_context) {
        CUdevice cu_dev;
        if (cuDeviceGet(&cu_dev, runtime->device_id) == CUDA_SUCCESS) {
            cuDevicePrimaryCtxRelease(cu_dev);
        }
        runtime->cu_context = NULL;
    }

    free(runtime);
    printf("[Ferrite-OS] Shutdown complete\n");
}

