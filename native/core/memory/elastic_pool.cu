/*
 * Elastic GPU Memory Pool Implementation
 *
 * Manages two-tier memory: TLSF for mutable, append-only for immutable weights
 */

#include "memory/elastic_pool.cuh"
#include "memory/ptx_tlsf_allocator.h"
#include "memory/ptx_cuda_driver.h"
#include <cuda.h>
#include <chrono>
#include <algorithm>

// ============================================================================
// ElasticPool Implementation
// ============================================================================

ElasticPool::ElasticPool(int device_id, ElasticPoolConfig config)
    : device_id_(device_id)
    , config_(config)
    , tlsf_(nullptr)
    , tlsf_base_(nullptr)
    , tlsf_size_(0)
    , immutable_total_used_(0)
{
    cudaSetDevice(device_id);

    // Query total VRAM
    size_t vram_free, vram_total;
    cudaMemGetInfo(&vram_free, &vram_total);

    if (config_.verbose) {
        printf("[ElasticPool] Initializing on GPU %d\n", device_id);
        printf("[ElasticPool] VRAM: %.1f GB total, %.1f GB free\n",
               vram_total / 1e9, vram_free / 1e9);
    }

    // Reserve space for CUDA runtime
    size_t usable = vram_free - config_.cuda_reserve;

    // Allocate initial TLSF pool
    size_t tlsf_size = std::min(config_.tlsf_initial_size, usable / 2);

    tlsf_base_ = ptx_driver_alloc(tlsf_size);
    if (!tlsf_base_) {
        fprintf(stderr, "[ElasticPool] Failed to allocate TLSF pool\n");
        return;
    }

    tlsf_size_ = tlsf_size;

    // Initialize TLSF allocator (use from_pool since we already allocated)
    tlsf_ = ptx_tlsf_create_from_pool(tlsf_base_, tlsf_size, config_.verbose);
    if (!tlsf_) {
        fprintf(stderr, "[ElasticPool] Failed to initialize TLSF\n");
        ptx_driver_free(tlsf_base_);
        tlsf_base_ = nullptr;
        tlsf_size_ = 0;
        return;
    }

    if (config_.verbose) {
        printf("[ElasticPool] TLSF pool: %.1f MB at %p\n",
               tlsf_size_ / 1e6, tlsf_base_);
        printf("[ElasticPool] Remaining for weights: %.1f MB\n",
               (usable - tlsf_size) / 1e6);
    }
}

ElasticPool::~ElasticPool() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Free immutable chunks
    for (auto& chunk : immutable_chunks_) {
        if (chunk.base) {
            ptx_driver_free(chunk.base);
        }
    }
    immutable_chunks_.clear();
    immutable_blocks_.clear();

    // Free TLSF
    if (tlsf_) {
        ptx_tlsf_destroy(tlsf_);
        tlsf_ = nullptr;
    }
    if (tlsf_base_) {
        ptx_driver_free(tlsf_base_);
        tlsf_base_ = nullptr;
    }
}

// ============================================================================
// Mutable Allocations (TLSF)
// ============================================================================

void* ElasticPool::alloc_mutable(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!tlsf_) return nullptr;

    void* ptr = ptx_tlsf_alloc(tlsf_, size);

    if (!ptr && config_.verbose) {
        printf("[ElasticPool] TLSF allocation failed for %zu bytes\n", size);
    }

    return ptr;
}

void ElasticPool::free_mutable(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!tlsf_ || !ptr) return;

    ptx_tlsf_free(tlsf_, ptr);
}

bool ElasticPool::owns_mutable(void* ptr) const {
    if (!tlsf_base_ || !ptr) return false;

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t base = reinterpret_cast<uintptr_t>(tlsf_base_);

    return addr >= base && addr < base + tlsf_size_;
}

// ============================================================================
// Immutable Allocations (Append-only weights)
// ============================================================================

void* ElasticPool::alloc_immutable(size_t size, uint32_t layer_id,
                                    uint32_t tensor_id, bool is_quantized) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Align size to 256 bytes for GPU efficiency
    size = (size + 255) & ~255;

    // Try to find space in existing chunks
    void* ptr = find_immutable_space(size);

    // If no space, try to expand
    if (!ptr) {
        if (!expand_immutable_region(size)) {
            if (config_.verbose) {
                printf("[ElasticPool] Cannot expand immutable region for %zu bytes\n", size);
            }
            return nullptr;
        }
        ptr = find_immutable_space(size);
    }

    if (ptr) {
        // Record the block
        ImmutableBlock block = {
            .ptr = ptr,
            .size = size,
            .layer_id = layer_id,
            .tensor_id = tensor_id,
            .is_quantized = is_quantized,
        };
        immutable_blocks_.push_back(block);
        immutable_total_used_ += size;

        if (config_.verbose) {
            printf("[ElasticPool] Immutable alloc: L%u.T%u = %zu bytes at %p\n",
                   layer_id, tensor_id, size, ptr);
        }
    }

    return ptr;
}

void* ElasticPool::load_quantized_weight(const void* host_data, size_t size,
                                          uint32_t layer_id, uint32_t tensor_id) {
    auto start = std::chrono::high_resolution_clock::now();

    // Allocate space
    void* ptr = alloc_immutable(size, layer_id, tensor_id, true);
    if (!ptr) return nullptr;

    // Copy from host
    cudaError_t err = cudaMemcpy(ptr, host_data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ElasticPool] Weight copy failed: %s\n",
                cudaGetErrorString(err));
        // Note: We don't have a way to "free" immutable allocations
        // They're append-only by design
        return nullptr;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    total_loads_++;
    total_load_time_us_ += duration.count();

    return ptr;
}

void* ElasticPool::get_immutable(uint32_t layer_id, uint32_t tensor_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& block : immutable_blocks_) {
        if (block.layer_id == layer_id && block.tensor_id == tensor_id) {
            return block.ptr;
        }
    }
    return nullptr;
}

bool ElasticPool::can_grow_immutable(size_t additional_size) const {
    size_t available = query_available_vram();
    return available >= additional_size + config_.immutable_reserve;
}

// ============================================================================
// Memory Management
// ============================================================================

size_t ElasticPool::poll_and_expand() {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t available = query_available_vram();
    if (available <= config_.immutable_reserve) {
        return 0;
    }

    size_t can_allocate = available - config_.immutable_reserve;
    size_t chunk_size = std::min(can_allocate, config_.immutable_chunk_size);

    if (chunk_size < 64 * 1024 * 1024) {  // Don't bother with < 64MB
        return 0;
    }

    void* chunk_base = ptx_driver_alloc(chunk_size);
    if (!chunk_base) {
        return 0;
    }

    ImmutableChunk chunk = {
        .base = chunk_base,
        .size = chunk_size,
        .used = 0,
    };
    immutable_chunks_.push_back(chunk);

    if (config_.verbose) {
        printf("[ElasticPool] Expanded immutable region: +%.1f MB at %p\n",
               chunk_size / 1e6, chunk_base);
    }

    return chunk_size;
}

void ElasticPool::defragment_mutable() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (tlsf_) {
        ptx_tlsf_defragment(tlsf_);
    }
}

ElasticPoolStats ElasticPool::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    ElasticPoolStats stats = {};

    // TLSF stats
    if (tlsf_) {
        TLSFPoolStats tlsf_stats;
        ptx_tlsf_get_stats(tlsf_, &tlsf_stats);
        stats.tlsf_total = tlsf_stats.total_pool_size;
        stats.tlsf_used = tlsf_stats.allocated_bytes;
        stats.tlsf_free = tlsf_stats.free_bytes;
    }

    // Immutable stats
    size_t immutable_committed = 0;
    for (const auto& chunk : immutable_chunks_) {
        immutable_committed += chunk.size;
    }
    stats.immutable_total = immutable_committed;
    stats.immutable_used = immutable_total_used_;
    stats.immutable_committed = immutable_committed;
    stats.immutable_blocks = static_cast<uint32_t>(immutable_blocks_.size());

    // System stats
    size_t vram_free, vram_total;
    cudaMemGetInfo(&vram_free, &vram_total);
    stats.vram_total = vram_total;
    stats.vram_available = vram_free;
    stats.vram_reserved = config_.cuda_reserve;

    // Calculate compression ratio (quantized size vs f32)
    if (stats.immutable_blocks > 0) {
        // Estimate: count quantized blocks and assume ~4x compression
        size_t quantized_size = 0;
        for (const auto& block : immutable_blocks_) {
            if (block.is_quantized) {
                quantized_size += block.size;
            }
        }
        if (quantized_size > 0) {
            // Assume Q4 compression ratio
            stats.compression_ratio = 7.0f;  // ~7x for Q4_K
        }
    }

    // Performance
    uint64_t loads = total_loads_.load();
    uint64_t time = total_load_time_us_.load();
    stats.total_loads = loads;
    stats.avg_load_time_us = loads > 0 ? static_cast<float>(time) / loads : 0;

    return stats;
}

void ElasticPool::print_memory_map() const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto stats = get_stats();

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              Elastic Pool Memory Map                         ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ VRAM Total:     %8.1f MB                                  ║\n", stats.vram_total / 1e6);
    printf("║ VRAM Available: %8.1f MB                                  ║\n", stats.vram_available / 1e6);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ TLSF (Mutable):                                              ║\n");
    printf("║   Total:        %8.1f MB                                  ║\n", stats.tlsf_total / 1e6);
    printf("║   Used:         %8.1f MB                                  ║\n", stats.tlsf_used / 1e6);
    printf("║   Free:         %8.1f MB                                  ║\n", stats.tlsf_free / 1e6);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Immutable (Weights):                                         ║\n");
    printf("║   Committed:    %8.1f MB                                  ║\n", stats.immutable_committed / 1e6);
    printf("║   Used:         %8.1f MB                                  ║\n", stats.immutable_used / 1e6);
    printf("║   Blocks:       %8u                                      ║\n", stats.immutable_blocks);
    printf("║   Compression:  %8.1fx                                    ║\n", stats.compression_ratio);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Load Performance:                                            ║\n");
    printf("║   Total loads:  %8lu                                      ║\n", stats.total_loads);
    printf("║   Avg time:     %8.1f us                                  ║\n", stats.avg_load_time_us);
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

// ============================================================================
// Internal Helpers
// ============================================================================

bool ElasticPool::expand_immutable_region(size_t min_size) {
    size_t available = query_available_vram();

    if (available <= config_.immutable_reserve + min_size) {
        return false;
    }

    // Allocate at least min_size, up to chunk_size
    size_t chunk_size = std::max(min_size, config_.immutable_chunk_size);
    chunk_size = std::min(chunk_size, available - config_.immutable_reserve);

    // Align to 4MB for efficiency
    chunk_size = (chunk_size + (4 * 1024 * 1024 - 1)) & ~(4 * 1024 * 1024 - 1);

    void* chunk_base = ptx_driver_alloc(chunk_size);
    if (!chunk_base) {
        if (config_.verbose) {
            printf("[ElasticPool] Failed to expand immutable region\n");
        }
        return false;
    }

    ImmutableChunk chunk = {
        .base = chunk_base,
        .size = chunk_size,
        .used = 0,
    };
    immutable_chunks_.push_back(chunk);

    if (config_.verbose) {
        printf("[ElasticPool] Expanded immutable region: +%.1f MB at %p (total: %zu chunks)\n",
               chunk_size / 1e6, chunk_base, immutable_chunks_.size());
    }

    return true;
}

size_t ElasticPool::query_available_vram() const {
    size_t vram_free, vram_total;
    cudaMemGetInfo(&vram_free, &vram_total);
    return vram_free;
}

void* ElasticPool::find_immutable_space(size_t size) {
    // Find a chunk with enough space
    for (auto& chunk : immutable_chunks_) {
        if (chunk.size - chunk.used >= size) {
            void* ptr = static_cast<char*>(chunk.base) + chunk.used;
            chunk.used += size;
            return ptr;
        }
    }
    return nullptr;
}

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

ElasticPool* elastic_pool_create(int device_id, const ElasticPoolConfig* config) {
    ElasticPoolConfig cfg = config ? *config : ElasticPoolConfig::defaults();
    return new ElasticPool(device_id, cfg);
}

void elastic_pool_destroy(ElasticPool* pool) {
    delete pool;
}

void* elastic_pool_alloc_mutable(ElasticPool* pool, size_t size) {
    return pool ? pool->alloc_mutable(size) : nullptr;
}

void elastic_pool_free_mutable(ElasticPool* pool, void* ptr) {
    if (pool) pool->free_mutable(ptr);
}

void* elastic_pool_alloc_immutable(ElasticPool* pool, size_t size,
                                    uint32_t layer_id, uint32_t tensor_id, bool is_quantized) {
    return pool ? pool->alloc_immutable(size, layer_id, tensor_id, is_quantized) : nullptr;
}

void* elastic_pool_load_weight(ElasticPool* pool, const void* host_data, size_t size,
                                uint32_t layer_id, uint32_t tensor_id) {
    return pool ? pool->load_quantized_weight(host_data, size, layer_id, tensor_id) : nullptr;
}

void* elastic_pool_get_immutable(ElasticPool* pool, uint32_t layer_id, uint32_t tensor_id) {
    return pool ? pool->get_immutable(layer_id, tensor_id) : nullptr;
}

size_t elastic_pool_poll_expand(ElasticPool* pool) {
    return pool ? pool->poll_and_expand() : 0;
}

void elastic_pool_defragment(ElasticPool* pool) {
    if (pool) pool->defragment_mutable();
}

void elastic_pool_get_stats(ElasticPool* pool, ElasticPoolStats* stats) {
    if (pool && stats) {
        *stats = pool->get_stats();
    }
}

void elastic_pool_print_map(ElasticPool* pool) {
    if (pool) pool->print_memory_map();
}

bool elastic_pool_can_grow(ElasticPool* pool, size_t additional) {
    return pool ? pool->can_grow_immutable(additional) : false;
}

bool elastic_pool_owns_mutable(ElasticPool* pool, void* ptr) {
    return pool ? pool->owns_mutable(ptr) : false;
}

} // extern "C"
