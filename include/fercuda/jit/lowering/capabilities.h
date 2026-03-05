#pragma once

#include "fercuda/core/status.cuh"

namespace fer::jit::lowering {

struct DeviceCapabilities {
    int sm_major = 0;
    int sm_minor = 0;
    int max_threads_per_block = 1024;
    bool tensor_cores = false;
    bool cooperative_groups = false;
};

Status detect_device_capabilities(DeviceCapabilities* out_caps);

} // namespace fer::jit::lowering
