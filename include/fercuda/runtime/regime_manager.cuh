#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "fercuda/core/status.cuh"
#include "fercuda/alloc/memory.cuh"

namespace fer::runtime {

static constexpr uint32_t MEMORY_REGIME_AUTO = 0xFFFFFFFFu;

struct CustomAllocator {
    void* (*alloc)(size_t bytes, Tier tier, uint32_t tag, void* user_ctx) = nullptr;
    void (*free)(void* ptr, void* user_ctx) = nullptr;
    void* user_ctx = nullptr;
};

struct ResolvedRegime {
    bool is_custom = false;
    MemoryRegime builtin = MemoryRegime::CUSTOM_POOL;
    uint32_t raw = static_cast<uint32_t>(MemoryRegime::CUSTOM_POOL);
};

class RegimeManager {
public:
    explicit RegimeManager(ElasticPool* pool, uint32_t default_regime_id);

    Status register_custom_allocator(uint32_t regime_id, const CustomAllocator& allocator);
    Status unregister_custom_allocator(uint32_t regime_id);
    Status set_default_memory_regime(uint32_t regime_id);
    Status get_default_memory_regime(uint32_t* out_regime_id) const;
    Status list_registered_regimes(uint32_t* out_regime_ids, size_t capacity, size_t* out_count) const;

    Status resolve(uint32_t requested_raw, ResolvedRegime* out) const;
    Status alloc_bytes(size_t bytes, Tier tier, uint32_t tag, const ResolvedRegime& regime, void** out_ptr);
    Status free_bytes(void* ptr, uint32_t regime_id);

private:
    bool is_builtin_regime(uint32_t raw) const;

    ElasticPool* pool_ = nullptr;
    uint32_t default_regime_id_ = static_cast<uint32_t>(MemoryRegime::CUSTOM_POOL);
    std::unordered_map<uint32_t, CustomAllocator> custom_allocators_;
};

} // namespace fer::runtime
