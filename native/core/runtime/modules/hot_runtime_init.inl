// Initialize with default config (backward compatible)
GPUHotRuntime* gpu_hot_init(int device_id, const char* token) {
    GPUHotConfig config = gpu_hot_default_config();
    return gpu_hot_init_with_config(device_id, token, &config);
}

// Initialize GPU Hot Runtime with configuration
GPUHotRuntime* gpu_hot_init_with_config(int device_id, const char* token, const GPUHotConfig* config) {
    GPUHotRuntime* runtime = (GPUHotRuntime*)malloc(sizeof(GPUHotRuntime));
    if (!runtime) return NULL;

    // Use default config if none provided
    GPUHotConfig effective_config;
    if (config) {
        effective_config = *config;
    } else {
        effective_config = gpu_hot_default_config();
    }
    apply_env_overrides(&effective_config);

    // Single-pool strict mode guard: deny competing pool init from daemon children
    if (effective_config.single_pool_strict) {
        const char* daemon_client = getenv("PTX_DAEMON_CLIENT");
        if (daemon_client && daemon_client[0] == '1') {
            printf("[Ferrite-OS] DENIED: single-pool strict mode active — external pool init refused\n");
            printf("[Ferrite-OS] This process attempted to create a competing TLSF pool.\n");
            printf("[Ferrite-OS] Heavy GPU workloads must run inside the daemon's allocator domain.\n");
            free(runtime);
            return NULL;
        }
    }

    memset(runtime, 0, sizeof(GPUHotRuntime));
    runtime->device_id = device_id;
    ptx_mutex_init(&runtime->async_lock);
    runtime->event_pool = NULL;
    runtime->event_pool_capacity = 0;
    runtime->event_pool_count = 0;
    runtime->deferred_head = NULL;
    runtime->deferred_tail = NULL;
    runtime->deferred_count = 0;
    // Set device and initialize runtime
    cudaError_t cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
        printf("[Ferrite-OS] Error: cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(cuda_err));
        free(runtime);
        return NULL;
    }
    cuda_err = cudaFree(0);
    if (cuda_err != cudaSuccess) {
        printf("[Ferrite-OS] Error: cudaFree(0) failed: %s\n", cudaGetErrorString(cuda_err));
        free(runtime);
        return NULL;
    }

    // Capture primary context via CUDA driver API
    {
        CUdevice cu_dev;
        CUresult cu_err = cuDeviceGet(&cu_dev, device_id);
        if (cu_err == CUDA_SUCCESS) {
            cu_err = cuDevicePrimaryCtxRetain(&runtime->cu_context, cu_dev);
            if (cu_err == CUDA_SUCCESS) {
                if (!effective_config.quiet_init) {
                    printf("[Ferrite-OS] Captured primary context %p for device %d\n",
                           (void*)runtime->cu_context, device_id);
                }
                // Notify context hook (weak symbol: no-op if hook not loaded)
                if (ptx_context_hook_capture) {
                    ptx_context_hook_capture(device_id, runtime->cu_context);
                }
            } else {
                printf("[Ferrite-OS] Warning: cuDevicePrimaryCtxRetain failed (%d)\n", cu_err);
                runtime->cu_context = NULL;
            }
        } else {
            printf("[Ferrite-OS] Warning: cuDeviceGet failed (%d)\n", cu_err);
            runtime->cu_context = NULL;
        }
    }

    if (!effective_config.quiet_init) {
        printf("[Ferrite-OS] Initializing GPU Hot Runtime on device %d\n", device_id);
    }
    
    // ========================================================================
    // PTX-OS Global Registry & System State (IPC Support)
    // ========================================================================
    size_t registry_size = sizeof(GPURegistryEntry) * GPU_HOT_MAX_NAMED_SEGMENTS;
    size_t system_state_size = sizeof(PTXSystemState);
    size_t total_shm_size = registry_size + system_state_size;

    // Construct per-UID IPC key to avoid collisions on multi-user systems
    char ipc_key[GPU_HOT_IPC_KEY_MAX_LEN];
    snprintf(ipc_key, sizeof(ipc_key), "%s%u%s",
             GPU_HOT_IPC_KEY_PREFIX, (unsigned)getuid(), GPU_HOT_IPC_KEY_SUFFIX);

#ifdef _WIN32
    runtime->shm_handle = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)total_shm_size, ipc_key);
    bool is_daemon = (GetLastError() != ERROR_ALREADY_EXISTS);
    void* shm_base = MapViewOfFile(runtime->shm_handle, FILE_MAP_ALL_ACCESS, 0, 0, total_shm_size);
#else
    void* shm_base = NULL;
    // Linux POSIX shared memory
    // Try O_EXCL to detect if we are the creator (daemon)
    int shm_fd = shm_open(ipc_key, O_CREAT | O_EXCL | O_RDWR, 0666);
    bool is_daemon = true;

    if (shm_fd == -1 && errno == EEXIST) {
        // Segment already exists, we are a client
        is_daemon = false;
        shm_fd = shm_open(ipc_key, O_RDWR, 0666);
    }

    if (shm_fd == -1) {
        printf("[Ferrite-OS] Error: Failed to create/open shared memory (errno: %d)\n", errno);
        shm_base = NULL;
    } else {
        // Set size (only needed for daemon)
        if (is_daemon && ftruncate(shm_fd, total_shm_size) == -1) {
            printf("[Ferrite-OS] Error: Failed to set shared memory size\n");
            close(shm_fd);
            shm_base = NULL;
        } else {
            shm_base = mmap(NULL, total_shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            close(shm_fd); // File descriptor no longer needed after mmap
            if (shm_base == MAP_FAILED) {
                printf("[Ferrite-OS] Error: Failed to map shared memory (errno: %d)\n", errno);
                shm_base = NULL;
            }
        }
    }
#endif
    if (!shm_base) {
        printf("[Ferrite-OS] Error: Failed to map Global OS Space\n");
    } else {
        runtime->global_registry = (GPURegistryEntry*)shm_base;
        runtime->system_state = (PTXSystemState*)((char*)shm_base + registry_size);
        
        if (is_daemon) {
            memset(shm_base, 0, total_shm_size);
            // Generate session token
            runtime->system_state->auth_token = 0xDEADC0DE; // In prod, this would be random
            printf("[Ferrite-OS] Global state initialized (daemon mode)\n");
        } else {
            printf("[Ferrite-OS] Attached to global state (client mode)\n");
            InterlockedIncrement((LONG*)&runtime->system_state->active_processes);
        }

        // ========================================================================
        // GPU Native Access: Register SHM for Zero-Copy GPU access
        // ========================================================================
        cudaError_t shm_err = cudaHostRegister(shm_base, total_shm_size, cudaHostRegisterMapped);
        if (shm_err == cudaErrorHostMemoryAlreadyRegistered) {
            // Benign when multiple clients attach to the same shared segment.
            printf("[Ferrite-OS] Global state already registered with GPU context\n");
        } else if (shm_err != cudaSuccess) {
            printf("[Ferrite-OS] Warning: Could not register SHM for GPU access: %s\n", cudaGetErrorString(shm_err));
        } else {
            printf("[Ferrite-OS] Global state mapped to GPU context\n");
        }
    }
    // ========================================================================
    
    // Get device properties
    cudaDeviceProp prop;
    cuda_err = cudaGetDeviceProperties(&prop, device_id);
    if (cuda_err != cudaSuccess) {
        printf("[Ferrite-OS] Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(cuda_err));
        free(runtime);
        return NULL;
    }
    printf("[Ferrite-OS] Device: %s\n", prop.name);
    printf("[Ferrite-OS] Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("[Ferrite-OS] Total VRAM: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Create CUDA streams with priority range sensing
    int least_priority, greatest_priority;
    cuda_err = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    if (cuda_err != cudaSuccess) {
        printf("[Ferrite-OS] Error: cudaDeviceGetStreamPriorityRange failed: %s\n", cudaGetErrorString(cuda_err));
        free(runtime);
        return NULL;
    }

    // Use configured stream count, bounded by max
    runtime->num_streams = effective_config.max_streams;
    if (runtime->num_streams > GPU_HOT_MAX_STREAMS) {
        runtime->num_streams = GPU_HOT_MAX_STREAMS;
    }
    if (runtime->num_streams < 1) {
        runtime->num_streams = 16; // Minimum default
    }

    for (int i = 0; i < runtime->num_streams; i++) {
        // First 2 streams are Real-Time (High Priority)
        int priority = (i < 2) ? greatest_priority : least_priority;
        cuda_err = cudaStreamCreateWithPriority(&runtime->streams[i], cudaStreamNonBlocking, priority);
        if (cuda_err != cudaSuccess) {
            printf("[Ferrite-OS] Error: cudaStreamCreateWithPriority failed: %s\n", cudaGetErrorString(cuda_err));
            free(runtime);
            return NULL;
        }
    }
    printf("[Ferrite-OS] Created %d async streams (Priority Range: %d to %d)\n",
           runtime->num_streams, least_priority, greatest_priority);
    
    // ========================================================================
    // Pool Sizing with Configuration
    // ========================================================================
    size_t free_mem, total_mem;
    cuda_err = cudaMemGetInfo(&free_mem, &total_mem);
    if (cuda_err != cudaSuccess) {
        printf("[Ferrite-OS] Error: cudaMemGetInfo failed: %s\n", cudaGetErrorString(cuda_err));
        free(runtime);
        return NULL;
    }

    // Calculate pool size based on config
    size_t reserve = effective_config.reserve_vram;
    if (reserve == 0) {
        // Auto headroom: 5% of total VRAM, clamped to [256MB, 1GB]
        size_t auto_reserve = total_mem / 20;
        size_t min_reserve = 256ULL * 1024 * 1024;
        size_t max_reserve = 1024ULL * 1024 * 1024;
        if (auto_reserve < min_reserve) auto_reserve = min_reserve;
        if (auto_reserve > max_reserve) auto_reserve = max_reserve;
        reserve = auto_reserve;
    }
    size_t available = (free_mem > reserve) ? (free_mem - reserve) : 0;

    if (effective_config.fixed_pool_size > 0) {
        // Use fixed pool size
        runtime->vram_pool_size = effective_config.fixed_pool_size;
    } else if (effective_config.pool_fraction > 0.0f) {
        // Use fraction of available VRAM
        runtime->vram_pool_size = (size_t)(available * effective_config.pool_fraction);
    } else {
        // Fallback to minimum
        runtime->vram_pool_size = effective_config.min_pool_size;
    }

    // Apply size constraints
    if (runtime->vram_pool_size < effective_config.min_pool_size) {
        runtime->vram_pool_size = effective_config.min_pool_size;
    }
    if (effective_config.max_pool_size > 0 && runtime->vram_pool_size > effective_config.max_pool_size) {
        runtime->vram_pool_size = effective_config.max_pool_size;
    }

    // Safety: Don't exceed available memory
    if (runtime->vram_pool_size > available) {
        runtime->vram_pool_size = available;
    }

    if (!effective_config.quiet_init) {
        printf("[Ferrite-OS] Pool Configuration:\n");
        if (effective_config.fixed_pool_size > 0) {
            printf("[Ferrite-OS]   Mode: Fixed size (%.2f GB)\n",
                   effective_config.fixed_pool_size / (1024.0 * 1024.0 * 1024.0));
        } else {
            printf("[Ferrite-OS]   Mode: %.0f%% of available VRAM\n", effective_config.pool_fraction * 100.0f);
        }
        printf("[Ferrite-OS]   Available VRAM: %.2f GB (%.2f GB total, %.2f GB reserved)\n",
               available / (1024.0 * 1024.0 * 1024.0),
               free_mem / (1024.0 * 1024.0 * 1024.0),
               reserve / (1024.0 * 1024.0 * 1024.0));
        printf("[Ferrite-OS]   Claiming: %.2f GB for Hot Pool\n",
               runtime->vram_pool_size / (1024.0 * 1024.0 * 1024.0));
    }
    
    runtime->vram_pool = ptx_driver_alloc(runtime->vram_pool_size);
    if (!runtime->vram_pool) {
        printf("[Ferrite-OS] Warning:Dynamic allocation failed, falling back to min pool size\n");
        runtime->vram_pool_size = effective_config.min_pool_size;
        runtime->vram_pool = ptx_driver_alloc(runtime->vram_pool_size);
    }

    if (runtime->vram_pool) {
        if (!effective_config.quiet_init) {
            printf("[Ferrite-OS] VRAM pool allocated: %.2f GB\n",
                   runtime->vram_pool_size / (1024.0 * 1024.0 * 1024.0));
        }

        // ========================================================================
        // Initialize Enhanced TLSF Allocator (Production Grade)
        // ========================================================================
        if (!effective_config.quiet_init) {
            printf("[Ferrite-OS] Initializing Enhanced TLSF Allocator...\n");
        }
        runtime->tlsf_allocator = ptx_tlsf_create_from_pool(runtime->vram_pool,
                                                              runtime->vram_pool_size,
                                                              effective_config.enable_leak_detection);

        if (runtime->tlsf_allocator) {
            ptx_tlsf_set_warning_threshold(runtime->tlsf_allocator, effective_config.warning_threshold);
            // Warning throttling overrides
            double warn_step = 0.0;
            if (parse_env_double("PTX_POOL_WARN_STEP", &warn_step)) {
                if (warn_step > 1.0 && warn_step <= 100.0) {
                    warn_step = warn_step / 100.0;
                }
                if (warn_step >= 0.0) {
                    ptx_tlsf_set_warning_step(runtime->tlsf_allocator, (float)warn_step);
                }
            }
            double warn_interval_ms = 0.0;
            if (parse_env_double("PTX_POOL_WARN_INTERVAL_MS", &warn_interval_ms)) {
                if (warn_interval_ms >= 0.0) {
                    ptx_tlsf_set_warning_interval_ms(runtime->tlsf_allocator, (uint64_t)warn_interval_ms);
                }
            }
            if (getenv("PTX_POOL_WARN_DISABLE")) {
                ptx_tlsf_set_warnings_enabled(runtime->tlsf_allocator, false);
            }
            ptx_tlsf_set_auto_defrag(runtime->tlsf_allocator, true);
            if (!effective_config.quiet_init) {
                printf("[Ferrite-OS] Enhanced TLSF allocator initialized\n");
                printf("[Ferrite-OS]      - O(1) allocation/deallocation\n");
                printf("[Ferrite-OS]      - O(1) block lookup via hash table\n");
                if (effective_config.enable_leak_detection) {
                    printf("[Ferrite-OS]      - Memory leak detection enabled\n");
                }
                if (effective_config.enable_pool_health) {
                    printf("[Ferrite-OS]      - Pool health monitoring active\n");
                }
            }
        } else {
            printf("[Ferrite-OS] Error: Failed to create enhanced TLSF allocator\n");
            printf("[Ferrite-OS] Error: TLSF allocator is required; aborting init\n");
            runtime->tlsf_allocator = NULL;
            gpu_hot_shutdown(runtime);
            return NULL;
        }
        
        runtime->fallback_count = 0;

        if (!effective_config.quiet_init) {
            printf("[Ferrite-OS] Memory allocator ready\n");
        }
    } else {
        printf("[Ferrite-OS] Warning: Could not pre-allocate VRAM pool\n");
    }

    // Initialize CUDA Graphs
    runtime->num_graphs = 0;
    runtime->capturing_stream_id = -1;
    memset(runtime->graphs, 0, sizeof(runtime->graphs));
    if (!effective_config.quiet_init) {
        printf("[Ferrite-OS] CUDA graphs initialized\n");
    }

    // Create keepalive event
    cudaEventCreate(&runtime->keepalive_event);

    // Launch initial keepalive kernel
    runtime->keepalive_running = true;

    if (!effective_config.quiet_init) {
        printf("[Ferrite-OS] Runtime initialized\n\n");
    }

    return runtime;
}

