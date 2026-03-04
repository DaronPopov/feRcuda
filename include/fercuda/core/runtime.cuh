#pragma once

#include <cuda_runtime.h>

namespace fer {

struct DeviceId {
    int value = 0;
    explicit constexpr DeviceId(int v = 0) : value(v) {}
};

struct StreamHandle {
    cudaStream_t value = 0;
    explicit constexpr StreamHandle(cudaStream_t v = 0) : value(v) {}
};

struct OpContext {
    DeviceId device;
    StreamHandle stream;
};

} // namespace fer
