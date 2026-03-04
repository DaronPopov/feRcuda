#include "fercuda/runtime/regime_manager.cuh"

#include <algorithm>
#include <vector>

namespace fer::runtime {

RegimeManager::RegimeManager(ElasticPool* pool, uint32_t default_regime_id)
    : pool_(pool), default_regime_id_(default_regime_id) {}

bool RegimeManager::is_builtin_regime(uint32_t raw) const {
    return raw <= static_cast<uint32_t>(MemoryRegime::CUDA_MANAGED);
}

Status RegimeManager::register_custom_allocator(uint32_t regime_id, const CustomAllocator& allocator) {
    if (is_builtin_regime(regime_id) || regime_id == MEMORY_REGIME_AUTO) {
        return Status::invalid_argument("regime id is reserved for builtin/auto");
    }
    if (!allocator.alloc || !allocator.free) {
        return Status::invalid_argument("custom allocator callbacks must be non-null");
    }
    custom_allocators_[regime_id] = allocator;
    return Status::ok_status();
}

Status RegimeManager::unregister_custom_allocator(uint32_t regime_id) {
    if (is_builtin_regime(regime_id) || regime_id == MEMORY_REGIME_AUTO) {
        return Status::invalid_argument("cannot unregister builtin/auto regime");
    }
    auto it = custom_allocators_.find(regime_id);
    if (it == custom_allocators_.end()) return Status::not_found("custom allocator regime not registered");
    if (default_regime_id_ == regime_id) {
        default_regime_id_ = static_cast<uint32_t>(MemoryRegime::CUSTOM_POOL);
    }
    custom_allocators_.erase(it);
    return Status::ok_status();
}

Status RegimeManager::set_default_memory_regime(uint32_t regime_id) {
    ResolvedRegime rr{};
    Status st = resolve(regime_id, &rr);
    if (!st.ok()) return st;
    default_regime_id_ = rr.raw;
    return Status::ok_status();
}

Status RegimeManager::get_default_memory_regime(uint32_t* out_regime_id) const {
    if (!out_regime_id) return Status::invalid_argument("out_regime_id is null");
    *out_regime_id = default_regime_id_;
    return Status::ok_status();
}

Status RegimeManager::list_registered_regimes(uint32_t* out_regime_ids, size_t capacity, size_t* out_count) const {
    if (!out_count) return Status::invalid_argument("out_count is null");
    std::vector<uint32_t> ids;
    ids.reserve(3 + custom_allocators_.size());
    ids.push_back(static_cast<uint32_t>(MemoryRegime::CUSTOM_POOL));
    ids.push_back(static_cast<uint32_t>(MemoryRegime::CUDA_MALLOC));
    ids.push_back(static_cast<uint32_t>(MemoryRegime::CUDA_MANAGED));
    for (const auto& kv : custom_allocators_) ids.push_back(kv.first);
    std::sort(ids.begin(), ids.end());

    *out_count = ids.size();
    if (out_regime_ids && capacity > 0) {
        const size_t n = std::min(capacity, ids.size());
        for (size_t i = 0; i < n; i++) out_regime_ids[i] = ids[i];
    }
    return Status::ok_status();
}

Status RegimeManager::resolve(uint32_t requested_raw, ResolvedRegime* out) const {
    if (!out) return Status::invalid_argument("out regime is null");
    uint32_t raw = (requested_raw == MEMORY_REGIME_AUTO) ? default_regime_id_ : requested_raw;
    if (is_builtin_regime(raw)) {
        out->is_custom = false;
        out->builtin = static_cast<MemoryRegime>(raw);
        out->raw = raw;
        return Status::ok_status();
    }
    auto it = custom_allocators_.find(raw);
    if (it == custom_allocators_.end()) return Status::not_found("custom allocator regime not registered");
    out->is_custom = true;
    out->builtin = MemoryRegime::CUSTOM_POOL;
    out->raw = raw;
    return Status::ok_status();
}

Status RegimeManager::alloc_bytes(size_t bytes, Tier tier, uint32_t tag, const ResolvedRegime& regime, void** out_ptr) {
    if (!out_ptr) return Status::invalid_argument("out_ptr is null");
    if (!pool_) return Status::internal_error("regime manager pool is null");

    try {
        if (!regime.is_custom) {
            *out_ptr = pool_->alloc_bytes_regime(bytes, tier, tag, regime.builtin);
            return Status::ok_status();
        }
    } catch (...) {
        return Status::internal_error("builtin regime allocation failed");
    }

    auto it = custom_allocators_.find(regime.raw);
    if (it == custom_allocators_.end()) return Status::not_found("custom allocator regime not registered");
    void* p = it->second.alloc(bytes, tier, tag, it->second.user_ctx);
    if (!p) return Status::internal_error("custom allocator returned null");
    *out_ptr = p;
    return Status::ok_status();
}

Status RegimeManager::free_bytes(void* ptr, uint32_t regime_id) {
    if (!ptr) return Status::ok_status();
    if (!pool_) return Status::internal_error("regime manager pool is null");

    if (is_builtin_regime(regime_id)) {
        pool_->free(ptr);
        return Status::ok_status();
    }
    auto it = custom_allocators_.find(regime_id);
    if (it == custom_allocators_.end()) return Status::not_found("custom allocator regime not registered");
    it->second.free(ptr, it->second.user_ctx);
    return Status::ok_status();
}

} // namespace fer::runtime
