#pragma once
/*
 * feRcuda :: memory.cuh
 *
 * ElasticPool — thin delegation layer to the native GPUHotRuntime TLSF.
 *
 * All CUSTOM_POOL allocations route through gpu_hot_alloc / gpu_hot_free.
 * Non-pool regimes (CUDA_MALLOC, CUDA_MANAGED) fall through to cudaMalloc.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <vector>
#include "fercuda/compute/types.cuh"

struct GPUHotRuntime;

namespace fer {

enum class MemoryRegime : uint8_t {
    CUSTOM_POOL = 0,
    CUDA_MALLOC = 1,
    CUDA_MANAGED = 2,
};

struct PoolConfig {
    size_t mutable_bytes   = 512ULL << 20;
    size_t immutable_bytes = 2ULL   << 30;
    size_t cuda_reserve    = 256ULL << 20;
    bool   verbose         = false;
    MemoryRegime regime    = MemoryRegime::CUSTOM_POOL;
};

enum class Tier : uint8_t { MUTABLE = 0, IMMUTABLE = 1 };

struct AllocMeta {
    void*    ptr;
    size_t   bytes;
    Tier     tier;
    uint32_t tag;
    MemoryRegime regime;
};

class ElasticPool {
public:
    explicit ElasticPool(int device_id = 0, PoolConfig cfg = {});
    ~ElasticPool();

    void* alloc_bytes(size_t bytes, Tier tier = Tier::MUTABLE, uint32_t tag = 0) {
        return alloc_raw(bytes, tier, tag);
    }
    void* alloc_bytes_regime(size_t bytes, Tier tier, uint32_t tag, MemoryRegime regime) {
        return alloc_raw(bytes, tier, tag, regime);
    }

    template<typename T, int N>
    Tensor<T, N> alloc_mutable(Shape<N> shape, uint32_t tag = 0) {
        void* ptr = alloc_raw(shape.numel() * sizeof(T), Tier::MUTABLE, tag);
        return Tensor<T, N>(static_cast<T*>(ptr), shape);
    }

    template<typename T, int N>
    Tensor<T, N> alloc_immutable(Shape<N> shape, uint32_t tag = 0) {
        void* ptr = alloc_raw(shape.numel() * sizeof(T), Tier::IMMUTABLE, tag);
        return Tensor<T, N>(static_cast<T*>(ptr), shape);
    }

    void free(void* ptr);

    template<typename T, int N>
    Tensor<T, N> upload(const T* host_src, Shape<N> shape, uint32_t tag = 0,
                        cudaStream_t stream = 0) {
        auto t = alloc_immutable<T, N>(shape, tag);
        cudaMemcpyAsync(t.data, host_src, t.nbytes(), cudaMemcpyHostToDevice, stream);
        return t;
    }

    void set_native_runtime(GPUHotRuntime* rt) { native_rt_ = rt; }
    GPUHotRuntime* native_runtime() const { return native_rt_; }

    struct Stats {
        size_t mutable_used, mutable_free, mutable_total;
        size_t immutable_used, immutable_total;
        size_t vram_free;
        int    live_allocs;
    };
    Stats stats() const;
    void  print_stats() const;

private:
    void* alloc_raw(size_t bytes, Tier tier, uint32_t tag, MemoryRegime regime);
    void* alloc_raw(size_t bytes, Tier tier, uint32_t tag) {
        return alloc_raw(bytes, tier, tag, cfg_.regime);
    }

    int             device_;
    PoolConfig      cfg_;
    GPUHotRuntime*  native_rt_ = nullptr;

    size_t mutable_live_bytes_ = 0;
    size_t immutable_live_bytes_ = 0;
    std::vector<AllocMeta> live_;
    mutable std::mutex     live_lock_;
};

} // namespace fer
