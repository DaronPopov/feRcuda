// Cluster Management
GPUMultiCluster* gpu_multicluster_init(int* device_ids, int num_devices) {
    printf("[MULTI-GPU] Initializing GPU cluster with %d devices\n", num_devices);

    GPUMultiCluster* cluster = (GPUMultiCluster*)malloc(sizeof(GPUMultiCluster));
    if (!cluster) return NULL;

    memset(cluster, 0, sizeof(GPUMultiCluster));
    cluster->num_devices = num_devices;
    cluster->primary_device = device_ids ? device_ids[0] : 0;

    // Initialize devices
    for (int i = 0; i < num_devices; i++) {
        int device_id = device_ids ? device_ids[i] : i;
        cluster->devices[i].device_id = device_id;

        CUDA_CHECK_SUCCESS(cudaGetDeviceProperties(&cluster->devices[i].properties, device_id));
        cluster->devices[i].vram_total = cluster->devices[i].properties.totalGlobalMem;

        // Create runtime for each device
        cluster->runtimes[i] = gpu_hot_init(device_id, NULL);
        if (!cluster->runtimes[i]) {
            printf("[MULTI-GPU] Failed to initialize runtime for GPU %d\n", device_id);
            // Cleanup and return NULL
            for (int j = 0; j < i; j++) {
                gpu_hot_shutdown(cluster->runtimes[j]);
            }
            free(cluster);
            return NULL;
        }

        printf("[MULTI-GPU] GPU %d (%s): %.1f GB VRAM\n",
               device_id,
               cluster->devices[i].properties.name,
               cluster->devices[i].vram_total / (1024.0 * 1024.0 * 1024.0));
    }

    // Setup P2P communication
    gpu_setup_p2p_communication(cluster);

    // Calculate aggregate statistics
    cluster->total_vram = 0;
    cluster->aggregate_bandwidth = 0;

    for (int i = 0; i < num_devices; i++) {
        cluster->total_vram += cluster->devices[i].vram_total;
        cluster->aggregate_bandwidth += cluster->devices[i].bandwidth_to_host;
    }

    size_t total_vram_gb = (size_t)(cluster->total_vram / (1024ULL * 1024ULL * 1024ULL));
    printf("[MULTI-GPU] Cluster initialized: %zu GB total VRAM, %.1f GB/s aggregate bandwidth\n",
           total_vram_gb, cluster->aggregate_bandwidth);

    return cluster;
}

void gpu_multicluster_shutdown(GPUMultiCluster* cluster) {
    if (!cluster) return;

    printf("[MULTI-GPU] Shutting down GPU cluster\n");

    // Shutdown all runtimes
    for (int i = 0; i < cluster->num_devices; i++) {
        if (cluster->runtimes[i]) {
            gpu_hot_shutdown(cluster->runtimes[i]);
        }
    }

    // Disable P2P communication
    for (int i = 0; i < cluster->num_devices; i++) {
        for (int j = 0; j < cluster->num_devices; j++) {
            if (i != j && cluster->p2p_enabled[i][j]) {
                gpu_disable_p2p(i, j);
            }
        }
    }

    free(cluster);
}

// Distributed Tensor Operations
DistributedTensor* gpu_distributed_tensor_create(GPUMultiCluster* cluster,
                                                const char* name,
                                                int* shape, int dims,
                                                MultiTensorDtype dtype,
                                                int* device_distribution) {

    if (!cluster || dims < 1 || dims > 4) return NULL;

    DistributedTensor* tensor = (DistributedTensor*)malloc(sizeof(DistributedTensor));
    if (!tensor) return NULL;

    memset(tensor, 0, sizeof(DistributedTensor));
    tensor->cluster = cluster;
    strncpy(tensor->name, name, GPU_HOT_MAX_NAME_LEN - 1);
    tensor->dims = dims;
    tensor->dtype = dtype;

    // Calculate total elements
    size_t total_elements = 1;
    for (int i = 0; i < dims; i++) {
        tensor->shape[i] = shape[i];
        total_elements *= shape[i];
    }
    tensor->total_elements = total_elements;

    // Auto-distribution if not specified
    int devices_per_tensor[GPU_HOT_MAX_GPUS];
    if (!device_distribution) {
        // Simple round-robin distribution
        for (int i = 0; i < cluster->num_devices; i++) {
            devices_per_tensor[i] = 1;
        }
    } else {
        memcpy(devices_per_tensor, device_distribution, sizeof(int) * cluster->num_devices);
    }

    // Create shards on each device
    size_t elements_per_device = total_elements / cluster->num_devices;
    size_t remainder = total_elements % cluster->num_devices;

    size_t offset = 0;
    for (int i = 0; i < cluster->num_devices; i++) {
        if (!devices_per_tensor[i]) continue;

        size_t shard_elements = elements_per_device + (i < remainder ? 1 : 0);

        tensor->device_ids[tensor->num_devices] = i;
        tensor->shards[tensor->num_devices] = (TensorShard*)malloc(sizeof(TensorShard));
        if (!tensor->shards[tensor->num_devices]) {
            // Cleanup and return NULL
            for (int j = 0; j < tensor->num_devices; j++) {
                free(tensor->shards[j]);
            }
            free(tensor);
            return NULL;
        }

        TensorShard* shard = tensor->shards[tensor->num_devices];
        shard->device_id = i;
        shard->elements = shard_elements;
        shard->offset = offset;

        // Allocate memory on device via PTX-OS runtime
        CUDA_CHECK_SUCCESS(cudaSetDevice(i));
        if (!cluster->runtimes[i]) {
            printf("[MULTI-GPU] ERROR: No runtime for device %d\n", i);
            gpu_distributed_tensor_free(tensor);
            return NULL;
        }
        shard->data = gpu_hot_alloc(cluster->runtimes[i], shard_elements * sizeof(float));
        if (!shard->data) {
            printf("[MULTI-GPU] ERROR: TLSF allocation failed on device %d\n", i);
            gpu_distributed_tensor_free(tensor);
            return NULL;
        }

        // Create IPC handle for P2P access
        CUDA_CHECK_SUCCESS(cudaIpcGetMemHandle(&shard->ipc_handle, shard->data));

        tensor->num_devices++;
        offset += shard_elements;
    }

    printf("[MULTI-GPU] Created distributed tensor '%s': %zu elements across %d devices\n",
           name, total_elements, tensor->num_devices);

    return tensor;
}

void gpu_distributed_tensor_free(DistributedTensor* tensor) {
    if (!tensor) return;

    for (int i = 0; i < tensor->num_devices; i++) {
        if (tensor->shards[i]) {
            CUDA_CHECK_SUCCESS(cudaSetDevice(tensor->device_ids[i]));
            if (tensor->cluster && tensor->cluster->runtimes[tensor->device_ids[i]] &&
                gpu_hot_owns_ptr(tensor->cluster->runtimes[tensor->device_ids[i]], tensor->shards[i]->data)) {
                gpu_hot_free(tensor->cluster->runtimes[tensor->device_ids[i]], tensor->shards[i]->data);
            } else {
                ptx_strict_free_violation("MULTI-GPU", tensor->shards[i]->data);
            }
            free(tensor->shards[i]);
        }
    }

    free(tensor);
}

// Data Movement Operations
int gpu_tensor_migrate(DistributedTensor* tensor, int src_device, int dst_device,
                      size_t offset, size_t size) {

    if (!tensor || src_device >= tensor->num_devices || dst_device >= tensor->num_devices) {
        return -1;
    }

    // Find source and destination shards
    TensorShard* src_shard = NULL;
    TensorShard* dst_shard = NULL;

    for (int i = 0; i < tensor->num_devices; i++) {
        if (tensor->device_ids[i] == src_device) src_shard = tensor->shards[i];
        if (tensor->device_ids[i] == dst_device) dst_shard = tensor->shards[i];
    }

    if (!src_shard || !dst_shard) return -1;

    // Use P2P copy if available
    if (src_device != dst_device) {
        CUDA_CHECK(cudaMemcpyPeer(dst_shard->data, dst_device,
                                 src_shard->data, src_device, size));
    }

    printf("[MULTI-GPU] Migrated %zu bytes from GPU %d to GPU %d\n",
           size, src_device, dst_device);

    return 0;
}

int gpu_broadcast_tensor(DistributedTensor* tensor, int src_device) {
    if (!tensor) return -1;

    // Find source shard
    TensorShard* src_shard = NULL;
    for (int i = 0; i < tensor->num_devices; i++) {
        if (tensor->device_ids[i] == src_device) {
            src_shard = tensor->shards[i];
            break;
        }
    }
    if (!src_shard) return -1;

    // Broadcast to all other devices
    for (int i = 0; i < tensor->num_devices; i++) {
        int dst_device = tensor->device_ids[i];
        if (dst_device == src_device) continue;

        TensorShard* dst_shard = tensor->shards[i];
        CUDA_CHECK(cudaMemcpyPeer(dst_shard->data, dst_device,
                                 src_shard->data, src_device,
                                 src_shard->elements * sizeof(float)));
    }

    printf("[MULTI-GPU] Broadcast tensor '%s' from GPU %d to all devices\n",
           tensor->name, src_device);

    return 0;
}

