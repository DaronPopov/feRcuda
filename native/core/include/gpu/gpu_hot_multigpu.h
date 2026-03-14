/*
 * PTX-OS Multi-GPU Support - Header
 * Distributed computing across multiple GPUs
 */

#ifndef GPU_HOT_MULTIGPU_H
#define GPU_HOT_MULTIGPU_H

#include "gpu_hot_runtime.h"
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Configuration
// ============================================================================

#define GPU_HOT_MAX_GPUS            16
#define GPU_HOT_MAX_TENSOR_DIMS     8

// ============================================================================
// Communication Types
// ============================================================================

typedef enum GPUCommType {
    GPU_COMM_PCIE,
    GPU_COMM_NVLINK,
    GPU_COMM_NVSWITCH
} GPUCommType;

// ============================================================================
// Multi-GPU Tensor Data Types
// ============================================================================

typedef enum MultiTensorDtype {
    DTYPE_FLOAT32,
    DTYPE_FLOAT16,
    DTYPE_BFLOAT16,
    DTYPE_INT32,
    DTYPE_INT16,
    DTYPE_INT8,
    DTYPE_UINT8
} MultiTensorDtype;

// ============================================================================
// Device Information
// ============================================================================

typedef struct GPUDeviceInfo {
    int device_id;
    cudaDeviceProp properties;
    size_t vram_total;
    size_t vram_free;
    int compute_capability;
    float bandwidth_to_host;
    std::vector<float> bandwidth_to_gpu;
    GPUCommType comm_type;
} GPUDeviceInfo;

// ============================================================================
// Tensor Shard (portion of tensor on a single GPU)
// ============================================================================

typedef struct TensorShard {
    int device_id;
    void* data;
    size_t elements;
    size_t offset;
    cudaIpcMemHandle_t ipc_handle;
} TensorShard;

// ============================================================================
// Distributed Tensor
// ============================================================================

struct GPUMultiCluster;
typedef struct GPUMultiCluster GPUMultiCluster;

typedef struct DistributedTensor {
    GPUMultiCluster* cluster;
    char name[GPU_HOT_MAX_NAME_LEN];
    int shape[GPU_HOT_MAX_TENSOR_DIMS];
    int dims;
    MultiTensorDtype dtype;
    size_t total_elements;

    int num_devices;
    int device_ids[GPU_HOT_MAX_GPUS];
    TensorShard* shards[GPU_HOT_MAX_GPUS];
} DistributedTensor;

// ============================================================================
// Multi-GPU Cluster
// ============================================================================

typedef struct GPUMultiCluster {
    int num_devices;
    int primary_device;
    GPUDeviceInfo devices[GPU_HOT_MAX_GPUS];
    GPUHotRuntime* runtimes[GPU_HOT_MAX_GPUS];

    // P2P communication matrix
    int p2p_enabled[GPU_HOT_MAX_GPUS][GPU_HOT_MAX_GPUS];
    float bandwidth_matrix[GPU_HOT_MAX_GPUS][GPU_HOT_MAX_GPUS];

    // Aggregate stats
    size_t total_vram;
    float aggregate_bandwidth;
} GPUMultiCluster;

// ============================================================================
// Multi-GPU Statistics
// ============================================================================

typedef struct GPUMultiStats {
    float utilization[GPU_HOT_MAX_GPUS];
    float memory_usage[GPU_HOT_MAX_GPUS];
    float p2p_bandwidth[GPU_HOT_MAX_GPUS][GPU_HOT_MAX_GPUS];
} GPUMultiStats;

// ============================================================================
// Device Enumeration API
// ============================================================================

int gpu_enumerate_devices(GPUDeviceInfo* devices, int max_devices);
int gpu_analyze_topology(GPUDeviceInfo* devices, int num_devices);
float gpu_measure_bandwidth(int src_device, int dst_device);

// ============================================================================
// P2P Communication API
// ============================================================================

int gpu_enable_p2p(int src_device, int dst_device);
int gpu_disable_p2p(int src_device, int dst_device);
int gpu_setup_p2p_communication(GPUMultiCluster* cluster);

// ============================================================================
// Cluster Management API
// ============================================================================

GPUMultiCluster* gpu_multicluster_init(int* device_ids, int num_devices);
void gpu_multicluster_shutdown(GPUMultiCluster* cluster);
void gpu_multicluster_get_stats(GPUMultiCluster* cluster, GPUMultiStats* stats);

// ============================================================================
// Distributed Tensor API
// ============================================================================

DistributedTensor* gpu_distributed_tensor_create(GPUMultiCluster* cluster,
                                                  const char* name,
                                                  int* shape, int dims,
                                                  MultiTensorDtype dtype,
                                                  int* device_distribution);
void gpu_distributed_tensor_free(DistributedTensor* tensor);

// ============================================================================
// Data Movement API
// ============================================================================

int gpu_tensor_migrate(DistributedTensor* tensor, int src_device, int dst_device,
                       size_t offset, size_t size);
int gpu_broadcast_tensor(DistributedTensor* tensor, int src_device);
int gpu_reduce_tensor(DistributedTensor* result, DistributedTensor* input,
                      int root_device, const char* operation);

// ============================================================================
// Distributed Kernel Launch
// ============================================================================

cudaError_t gpu_launch_distributed_kernel(GPUMultiCluster* cluster,
                                          const char* kernel_name,
                                          const void** args,
                                          dim3 grid, dim3 block,
                                          size_t shared_mem,
                                          cudaStream_t stream,
                                          int* device_distribution);

// ============================================================================
// Load Balancing
// ============================================================================

int gpu_balance_workload(GPUMultiCluster* cluster, int* workload_per_device,
                         int total_work, int* device_capacities);

#ifdef __cplusplus
}
#endif

#endif // GPU_HOT_MULTIGPU_H
