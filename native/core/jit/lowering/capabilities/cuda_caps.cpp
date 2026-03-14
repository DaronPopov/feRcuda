#include "fercuda/jit/lowering/capabilities.h"

#include <cuda_runtime.h>

namespace fer::jit::lowering {

Status detect_device_capabilities(DeviceCapabilities* out_caps) {
    if (!out_caps) return Status::invalid_argument("out_caps is null");

    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return Status::internal_error("cudaGetDevice failed");

    cudaDeviceProp prop{};
    e = cudaGetDeviceProperties(&prop, dev);
    if (e != cudaSuccess) return Status::internal_error("cudaGetDeviceProperties failed");

    out_caps->sm_major = prop.major;
    out_caps->sm_minor = prop.minor;
    out_caps->max_threads_per_block = prop.maxThreadsPerBlock;
    out_caps->tensor_cores = (prop.major >= 7);
    out_caps->cooperative_groups = (prop.cooperativeLaunch != 0);
    return Status::ok_status();
}

} // namespace fer::jit::lowering
