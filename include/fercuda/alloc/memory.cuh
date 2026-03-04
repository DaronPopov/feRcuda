#pragma once
/*
 * feRcuda :: memory.cuh
 *
 * ElasticPool — two-tier GPU memory manager (evolved from persistant_gpu_os).
 *
 *  Tier 1 (MUTABLE)   — TLSF O(1) alloc/free for activations, KV cache, scratch
 *  Tier 2 (IMMUTABLE) — append-only bump allocator for model weights
 *
 * All allocs return typed Tensor views (no raw void*).
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <array>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "fercuda/compute/types.cuh"

namespace fer {

enum class MemoryRegime : uint8_t {
    CUSTOM_POOL = 0,
    CUDA_MALLOC = 1,
    CUDA_MANAGED = 2,
};

// ─── Pool Configuration ───────────────────────────────────────────────────────

struct PoolConfig {
    size_t mutable_bytes   = 512ULL << 20;   // 512 MB TLSF pool
    size_t immutable_bytes = 2ULL   << 30;   // 2   GB weight region
    size_t cuda_reserve    = 256ULL << 20;   // 256 MB left for runtime
    bool   verbose         = false;
    MemoryRegime regime    = MemoryRegime::CUSTOM_POOL;
};

// ─── Allocation Tags ──────────────────────────────────────────────────────────

enum class Tier : uint8_t { MUTABLE = 0, IMMUTABLE = 1 };

struct AllocMeta {
    void*    ptr;
    size_t   bytes;
    Tier     tier;
    uint32_t tag;   // user-defined (layer_id, op_id, etc.)
    MemoryRegime regime;
};

// ─── ElasticPool ──────────────────────────────────────────────────────────────

class ElasticPool {
public:
    explicit ElasticPool(int device_id = 0, PoolConfig cfg = {});
    ~ElasticPool();

    // Dynamic raw allocation API for runtime-managed buffer tables.
    void* alloc_bytes(size_t bytes, Tier tier = Tier::MUTABLE, uint32_t tag = 0) {
        return alloc_raw(bytes, tier, tag);
    }
    void* alloc_bytes_regime(size_t bytes, Tier tier, uint32_t tag, MemoryRegime regime) {
        return alloc_raw(bytes, tier, tag, regime);
    }

    // Typed allocation — returns a Tensor view
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

    // Free a mutable allocation (immutable are freed on pool destruction)
    void free(void* ptr);

    // Copy host → device into an immutable tensor (convenience)
    template<typename T, int N>
    Tensor<T, N> upload(const T* host_src, Shape<N> shape, uint32_t tag = 0,
                        cudaStream_t stream = 0)
    {
        auto t = alloc_immutable<T, N>(shape, tag);
        cudaMemcpyAsync(t.data, host_src, t.nbytes(),
                        cudaMemcpyHostToDevice, stream);
        return t;
    }

    struct Stats {
        size_t mutable_used, mutable_free, mutable_total;
        size_t immutable_used, immutable_total;
        size_t vram_free;
        int    live_allocs;
    };
    Stats stats() const;
    void  print_stats() const;

private:
    // TLSF-style segregated free lists for mutable CUSTOM_POOL allocations.
    static constexpr size_t kMutableAlign = 256;
    static constexpr int kTlsfFli = 32;
    static constexpr int kTlsfSli = 8;

    struct MutableBlock {
        size_t offset = 0;
        size_t size = 0;
        int prev_phys = -1;
        int next_phys = -1;
        int prev_free = -1;
        int next_free = -1;
        bool is_free = false;
        bool active = false;
    };

    void* alloc_raw(size_t bytes, Tier tier, uint32_t tag, MemoryRegime regime);
    void* alloc_raw(size_t bytes, Tier tier, uint32_t tag) {
        return alloc_raw(bytes, tier, tag, cfg_.regime);
    }
    static size_t align_up(size_t x, size_t a);
    static void size_to_bin(size_t size, int* out_fli, int* out_sli);
    void mutable_tlsf_init();
    void mutable_insert_free(int block_idx);
    void mutable_remove_free(int block_idx);
    int mutable_find_suitable(size_t size) const;
    int mutable_split_block(int block_idx, size_t req_size);
    void* mutable_alloc(size_t bytes);
    void mutable_free(void* ptr);

    int         device_;
    PoolConfig  cfg_;

    // Slab-backed custom pool region.
    void*  mut_base_   = nullptr;
    mutable std::mutex mut_lock_;
    std::vector<MutableBlock> mut_blocks_;
    int mut_phys_head_ = -1;
    uint32_t mut_fli_bitmap_ = 0;
    std::array<uint32_t, kTlsfFli> mut_sli_bitmap_{};
    std::array<std::array<int, kTlsfSli>, kTlsfFli> mut_free_heads_{};
    std::unordered_map<void*, int> mut_ptr_to_block_;

    // Slab-backed custom pool immutable region.
    void*  imm_base_   = nullptr;
    size_t imm_offset_ = 0;
    mutable std::mutex imm_lock_;

    // Live usage counters for regime-aware stats/budget checks.
    size_t mutable_live_bytes_ = 0;
    size_t immutable_live_bytes_ = 0;

    // Tracking
    std::vector<AllocMeta> live_;
    mutable std::mutex     live_lock_;
};

} // namespace fer
