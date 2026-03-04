#pragma once

#include "fercuda/runtime/session.cuh"

namespace fer::integration {

using GPUHotAllocFn = void* (*)(void* runtime, size_t bytes);
using GPUHotFreeFn = void (*)(void* runtime, void* ptr);

struct GPUHotBinding {
    void* runtime = nullptr;
    GPUHotAllocFn alloc = nullptr;
    GPUHotFreeFn free = nullptr;
};

inline void* gpu_hot_alloc_bridge(size_t bytes, Tier, uint32_t, void* user_ctx) {
    auto* b = static_cast<GPUHotBinding*>(user_ctx);
    if (!b || !b->alloc) return nullptr;
    return b->alloc(b->runtime, bytes);
}

inline void gpu_hot_free_bridge(void* ptr, void* user_ctx) {
    auto* b = static_cast<GPUHotBinding*>(user_ctx);
    if (!b || !b->free) return;
    b->free(b->runtime, ptr);
}

inline runtime::CustomAllocator make_gpu_hot_allocator(GPUHotBinding* binding) {
    runtime::CustomAllocator a{};
    a.alloc = &gpu_hot_alloc_bridge;
    a.free = &gpu_hot_free_bridge;
    a.user_ctx = static_cast<void*>(binding);
    return a;
}

} // namespace fer::integration
