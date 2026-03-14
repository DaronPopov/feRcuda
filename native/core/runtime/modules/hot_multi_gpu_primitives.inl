#include <cmath>
#include <cfloat>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("[MULTI-GPU] CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#define CUDA_CHECK_VOID(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("[MULTI-GPU] CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUDA_CHECK_SUCCESS(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("[MULTI-GPU] CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

#define CUDA_CHECK_VOID(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("[MULTI-GPU] CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

// Device enumeration and analysis
int gpu_enumerate_devices(GPUDeviceInfo* devices, int max_devices) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    int actual_count = (device_count < max_devices) ? device_count : max_devices;

    for (int i = 0; i < actual_count; i++) {
        devices[i].device_id = i;
        CUDA_CHECK(cudaGetDeviceProperties(&devices[i].properties, i));

        devices[i].vram_total = devices[i].properties.totalGlobalMem;
        devices[i].vram_free = devices[i].vram_total; // Approximation
        devices[i].compute_capability = devices[i].properties.major * 10 + devices[i].properties.minor;

        // Measure bandwidth to host (simplified)
        devices[i].bandwidth_to_host = 15.0f; // GB/s - typical PCIe bandwidth
    }

    printf("[MULTI-GPU] Enumerated %d GPU devices\n", actual_count);
    return actual_count;
}

int gpu_analyze_topology(GPUDeviceInfo* devices, int num_devices) {
    printf("[MULTI-GPU] Analyzing GPU topology...\n");

    // Check P2P capabilities
    for (int i = 0; i < num_devices; i++) {
        for (int j = 0; j < num_devices; j++) {
            if (i == j) {
                devices[i].bandwidth_to_gpu.push_back(0.0f);
                continue;
            }

            int p2p_supported = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&p2p_supported, i, j));

            if (p2p_supported) {
                // Measure P2P bandwidth (simplified)
                float bandwidth = 50.0f; // GB/s for NVLink, less for PCIe
                devices[i].bandwidth_to_gpu.push_back(bandwidth);
                devices[i].comm_type = GPU_COMM_NVLINK;
                printf("[MULTI-GPU] NVLink detected: GPU %d <-> GPU %d (%.1f GB/s)\n", i, j, bandwidth);
            } else {
                devices[i].bandwidth_to_gpu.push_back(15.0f); // PCIe bandwidth
                devices[i].comm_type = GPU_COMM_PCIE;
                printf("[MULTI-GPU] PCIe communication: GPU %d <-> GPU %d\n", i, j);
            }
        }
    }

    return 0;
}

float gpu_measure_bandwidth(int src_device, int dst_device) {
    // Simplified bandwidth measurement
    // In a real implementation, this would do actual bandwidth tests
    int p2p_supported = 0;
    cudaDeviceCanAccessPeer(&p2p_supported, src_device, dst_device);

    return p2p_supported ? 50.0f : 15.0f; // GB/s
}

// P2P Communication Setup
int gpu_enable_p2p(int src_device, int dst_device) {
    printf("[MULTI-GPU] Enabling P2P between GPU %d and GPU %d\n", src_device, dst_device);

    CUDA_CHECK(cudaSetDevice(src_device));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(dst_device, 0));

    CUDA_CHECK(cudaSetDevice(dst_device));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(src_device, 0));

    return 0;
}

int gpu_disable_p2p(int src_device, int dst_device) {
    printf("[MULTI-GPU] Disabling P2P between GPU %d and GPU %d\n", src_device, dst_device);

    CUDA_CHECK(cudaSetDevice(src_device));
    CUDA_CHECK(cudaDeviceDisablePeerAccess(dst_device));

    CUDA_CHECK(cudaSetDevice(dst_device));
    CUDA_CHECK(cudaDeviceDisablePeerAccess(src_device));

    return 0;
}

int gpu_setup_p2p_communication(GPUMultiCluster* cluster) {
    printf("[MULTI-GPU] Setting up P2P communication matrix\n");

    for (int i = 0; i < cluster->num_devices; i++) {
        for (int j = 0; j < cluster->num_devices; j++) {
            if (i == j) {
                cluster->p2p_enabled[i][j] = 1; // Self-access always enabled
                continue;
            }

            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));

            if (can_access) {
                if (gpu_enable_p2p(i, j) == 0) {
                    cluster->p2p_enabled[i][j] = 1;
                    printf("[MULTI-GPU] P2P enabled: GPU %d <-> GPU %d\n", i, j);
                } else {
                    cluster->p2p_enabled[i][j] = 0;
                }
            } else {
                cluster->p2p_enabled[i][j] = 0;
                printf("[MULTI-GPU] P2P not supported: GPU %d <-> GPU %d\n", i, j);
            }
        }
    }

    return 0;
}

