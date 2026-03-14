// ============================================================================
// REDUCTION KERNELS - GPU-accelerated reduction operations
// ============================================================================

// Sum reduction kernel - reduces array to single value per block
__global__ void reduce_sum_kernel(float* input, float* output, size_t n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // First level of reduction during load
    float val = 0.0f;
    if (i < n) val = input[i];
    if (i + blockDim.x < n) val += input[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Max reduction kernel
__global__ void reduce_max_kernel(float* input, float* output, size_t n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    float val = -INFINITY;
    if (i < n) val = input[i];
    if (i + blockDim.x < n) val = fmaxf(val, input[i + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Min reduction kernel
__global__ void reduce_min_kernel(float* input, float* output, size_t n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    float val = INFINITY;
    if (i < n) val = input[i];
    if (i + blockDim.x < n) val = fminf(val, input[i + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Element-wise addition kernel
__global__ void elementwise_add_kernel(float* a, float* b, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// Element-wise max kernel
__global__ void elementwise_max_kernel(float* a, float* b, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(a[idx], b[idx]);
    }
}

// Element-wise min kernel
__global__ void elementwise_min_kernel(float* a, float* b, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(a[idx], b[idx]);
    }
}

// Scale kernel (for averaging)
__global__ void scale_kernel(float* data, float scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

/**
 * gpu_reduce_tensor - Perform reduction across distributed tensor shards
 * 
 * Supports operations: "sum", "max", "min", "avg"
 * 
 * Algorithm:
 * 1. Each GPU reduces its local shard
 * 2. Results are gathered to root GPU via P2P
 * 3. Final reduction on root GPU
 * 4. Result is stored in result tensor
 */
int gpu_reduce_tensor(DistributedTensor* result, DistributedTensor* input,
                     int root_device, const char* operation) {
    
    if (!result || !input || !operation) {
        printf("[MULTI-GPU] Error: NULL arguments to gpu_reduce_tensor\n");
        return -1;
    }
    
    if (input->num_devices == 0) {
        printf("[MULTI-GPU] Error: Input tensor has no shards\n");
        return -1;
    }
    
    printf("[MULTI-GPU] Reducing tensor '%s' with operation '%s' to GPU %d\n",
           input->name, operation, root_device);
    
    // Determine reduction type
    enum { OP_SUM, OP_MAX, OP_MIN, OP_AVG } op_type;
    if (strcmp(operation, "sum") == 0) op_type = OP_SUM;
    else if (strcmp(operation, "max") == 0) op_type = OP_MAX;
    else if (strcmp(operation, "min") == 0) op_type = OP_MIN;
    else if (strcmp(operation, "avg") == 0 || strcmp(operation, "mean") == 0) op_type = OP_AVG;
    else {
        printf("[MULTI-GPU] Error: Unknown reduction operation '%s'\n", operation);
        return -1;
    }
    
    // Find root device shard in result tensor
    TensorShard* result_shard = NULL;
    for (int i = 0; i < result->num_devices; i++) {
        if (result->device_ids[i] == root_device) {
            result_shard = result->shards[i];
            break;
        }
    }
    
    if (!result_shard) {
        printf("[MULTI-GPU] Error: Root device %d not found in result tensor\n", root_device);
        return -1;
    }
    
    // Step 1: Reduce each local shard
    // For each shard, we reduce all elements to a single value (for scalar reduction)
    // OR we keep element-wise (for element-wise reduction)
    
    // We'll implement element-wise reduction (more useful for neural network gradients)
    // Each shard contributes to the final result element-wise
    
    size_t min_shard_size = SIZE_MAX;
    for (int i = 0; i < input->num_devices; i++) {
        if (input->shards[i]->elements < min_shard_size) {
            min_shard_size = input->shards[i]->elements;
        }
    }
    
    // Step 2: Initialize result with first shard's data
    int first_device = input->device_ids[0];
    TensorShard* first_shard = input->shards[0];
    
    cudaSetDevice(root_device);
    
    // Copy first shard to result
    if (first_device == root_device) {
        // Same device - use memcpy
        CUDA_CHECK(cudaMemcpy(result_shard->data, first_shard->data, 
                             first_shard->elements * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
    } else {
        // Different device - use P2P copy
        CUDA_CHECK(cudaMemcpyPeer(result_shard->data, root_device,
                                  first_shard->data, first_device,
                                  first_shard->elements * sizeof(float)));
    }
    
    // Step 3: Reduce remaining shards into result
    // Allocate temporary buffer on root device for P2P copies
    float* temp_buffer = NULL;
    if (!result->cluster || !result->cluster->runtimes[root_device]) {
        printf("[MULTI-GPU] ERROR: No runtime for root device %d\n", root_device);
        return -1;
    }
    temp_buffer = (float*)gpu_hot_alloc(result->cluster->runtimes[root_device],
                                        min_shard_size * sizeof(float));
    if (!temp_buffer) {
        printf("[MULTI-GPU] ERROR: TLSF allocation failed on root device %d\n", root_device);
        return -1;
    }
    
    const int block_size = 256;
    
    for (int i = 1; i < input->num_devices; i++) {
        int src_device = input->device_ids[i];
        TensorShard* src_shard = input->shards[i];
        size_t num_elements = src_shard->elements < min_shard_size ? 
                              src_shard->elements : min_shard_size;
        int num_blocks = (num_elements + block_size - 1) / block_size;
        
        // Copy shard data to root device
        if (src_device == root_device) {
            CUDA_CHECK(cudaMemcpy(temp_buffer, src_shard->data,
                                 num_elements * sizeof(float),
                                 cudaMemcpyDeviceToDevice));
        } else {
            CUDA_CHECK(cudaMemcpyPeer(temp_buffer, root_device,
                                      src_shard->data, src_device,
                                      num_elements * sizeof(float)));
        }
        
        // Apply reduction operation element-wise
        cudaSetDevice(root_device);
        
        switch (op_type) {
            case OP_SUM:
            case OP_AVG:
                elementwise_add_kernel<<<num_blocks, block_size>>>(
                    (float*)result_shard->data, temp_buffer, 
                    (float*)result_shard->data, num_elements);
                break;
            case OP_MAX:
                elementwise_max_kernel<<<num_blocks, block_size>>>(
                    (float*)result_shard->data, temp_buffer,
                    (float*)result_shard->data, num_elements);
                break;
            case OP_MIN:
                elementwise_min_kernel<<<num_blocks, block_size>>>(
                    (float*)result_shard->data, temp_buffer,
                    (float*)result_shard->data, num_elements);
                break;
        }
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Step 4: For average, divide by number of devices
    if (op_type == OP_AVG) {
        size_t num_elements = result_shard->elements;
        int num_blocks = (num_elements + block_size - 1) / block_size;
        float scale = 1.0f / (float)input->num_devices;
        
        scale_kernel<<<num_blocks, block_size>>>(
            (float*)result_shard->data, scale, num_elements);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Synchronize
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Cleanup
    gpu_hot_free(result->cluster->runtimes[root_device], temp_buffer);
    
    printf("[MULTI-GPU] Reduction complete: %d shards -> GPU %d\n",
           input->num_devices, root_device);
    
    return 0;
}

