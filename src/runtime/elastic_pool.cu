#include "fercuda/alloc/memory.cuh"

#include "gpu/gpu_hot_runtime.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace fer {

static size_t align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

ElasticPool::ElasticPool(int device_id, PoolConfig cfg)
    : device_(device_id), cfg_(cfg) {
    cudaError_t e = cudaSetDevice(device_);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("[feRcuda] cudaSetDevice failed: ") + cudaGetErrorString(e));
}

ElasticPool::~ElasticPool() {
    std::lock_guard<std::mutex> lk(live_lock_);
    for (const auto& meta : live_) {
        if (!meta.ptr) continue;
        if (meta.regime == MemoryRegime::CUSTOM_POOL) {
            if (native_rt_) gpu_hot_free(native_rt_, meta.ptr);
        } else {
            cudaFree(meta.ptr);
        }
    }
    live_.clear();
}

void* ElasticPool::alloc_raw(size_t bytes, Tier tier, uint32_t tag, MemoryRegime regime) {
    if (bytes == 0)
        throw std::runtime_error("[feRcuda] zero-byte allocation is invalid");

    void* ptr = nullptr;
    if (regime == MemoryRegime::CUSTOM_POOL) {
        if (!native_rt_)
            throw std::runtime_error("[feRcuda] CUSTOM_POOL requires native runtime");
        ptr = gpu_hot_alloc(native_rt_, align_up(bytes, 256));
        if (!ptr)
            throw std::runtime_error("[feRcuda] gpu_hot_alloc failed");
    } else {
        {
            std::lock_guard<std::mutex> lk(live_lock_);
            const size_t used = (tier == Tier::MUTABLE) ? mutable_live_bytes_ : immutable_live_bytes_;
            const size_t cap = (tier == Tier::MUTABLE) ? cfg_.mutable_bytes : cfg_.immutable_bytes;
            if (cap > 0 && used + bytes > cap)
                throw std::runtime_error("[feRcuda] regime budget exhausted");
        }
        cudaError_t e = cudaSuccess;
        if (regime == MemoryRegime::CUDA_MALLOC) {
            e = cudaMalloc(&ptr, bytes);
        } else if (regime == MemoryRegime::CUDA_MANAGED) {
            e = cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal);
        } else {
            throw std::runtime_error("[feRcuda] unknown memory regime");
        }
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("[feRcuda] allocation failed: ") + cudaGetErrorString(e));
    }

    {
        std::lock_guard<std::mutex> lk(live_lock_);
        live_.push_back({ptr, bytes, tier, tag, regime});
        if (tier == Tier::MUTABLE) mutable_live_bytes_ += bytes;
        else immutable_live_bytes_ += bytes;
    }
    return ptr;
}

void ElasticPool::free(void* ptr) {
    if (!ptr) return;
    AllocMeta meta{};
    bool found = false;
    {
        std::lock_guard<std::mutex> lk(live_lock_);
        for (auto it = live_.begin(); it != live_.end(); ++it) {
            if (it->ptr != ptr) continue;
            meta = *it;
            if (meta.tier == Tier::MUTABLE) mutable_live_bytes_ -= meta.bytes;
            else immutable_live_bytes_ -= meta.bytes;
            live_.erase(it);
            found = true;
            break;
        }
    }
    if (!found) return;

    if (meta.regime == MemoryRegime::CUSTOM_POOL) {
        if (native_rt_) gpu_hot_free(native_rt_, ptr);
    } else {
        cudaFree(ptr);
    }
}

ElasticPool::Stats ElasticPool::stats() const {
    size_t free_vram = 0, total_vram = 0;
    cudaMemGetInfo(&free_vram, &total_vram);
    size_t mutable_used = 0, immutable_used = 0;
    int live_allocs = 0;
    {
        std::lock_guard<std::mutex> lk(live_lock_);
        mutable_used = mutable_live_bytes_;
        immutable_used = immutable_live_bytes_;
        live_allocs = static_cast<int>(live_.size());
    }
    return {
        .mutable_used = mutable_used,
        .mutable_free = (cfg_.mutable_bytes > mutable_used) ? (cfg_.mutable_bytes - mutable_used) : 0,
        .mutable_total = cfg_.mutable_bytes,
        .immutable_used = immutable_used,
        .immutable_total = cfg_.immutable_bytes,
        .vram_free = free_vram,
        .live_allocs = live_allocs,
    };
}

void ElasticPool::print_stats() const {
    auto s = stats();
    printf("[feRcuda::ElasticPool] mutable %zu/%zu MB  immutable %zu/%zu MB  vram_free=%zu MB  live=%d  native=%s\n",
           s.mutable_used >> 20, s.mutable_total >> 20,
           s.immutable_used >> 20, s.immutable_total >> 20,
           s.vram_free >> 20, s.live_allocs,
           native_rt_ ? "yes" : "no");
}

} // namespace fer
