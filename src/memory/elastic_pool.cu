#include "fercuda/alloc/memory.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace fer {

static void cuda_check(cudaError_t e, const char* loc) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(loc) + ": " + cudaGetErrorString(e)) ; }
}
#define CK(x) cuda_check((x), #x)

size_t ElasticPool::align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1) ; }

void ElasticPool::size_to_bin(size_t size, int* out_fli, int* out_sli) {
    if (size == 0) {
        *out_fli = 0 ; *out_sli = 0 ; return ; }

    int fli = 0 ; size_t v = size ; while (v >>= 1) ++fli ; if (fli >= kTlsfFli) fli = kTlsfFli - 1 ; size_t base = size_t(1) << fli ; size_t step = std::max<size_t>(base / kTlsfSli, 1) ; int sli = static_cast<int>((size - base) / step) ; if (sli < 0) sli = 0 ; if (sli >= kTlsfSli) sli = kTlsfSli - 1 ; *out_fli = fli ; *out_sli = sli ; }

void ElasticPool::mutable_insert_free(int block_idx) {
    MutableBlock& b = mut_blocks_[block_idx];
    int fli = 0;
    int sli = 0;
    size_to_bin(b.size, &fli, &sli);
    b.is_free = true;
    b.prev_free = -1;
    b.next_free = mut_free_heads_[fli][sli];
    if (b.next_free >= 0) mut_blocks_[b.next_free].prev_free = block_idx;
    mut_free_heads_[fli][sli] = block_idx;
    mut_sli_bitmap_[fli] = mut_sli_bitmap_[fli] | (uint32_t(1) << sli);
    mut_fli_bitmap_ = mut_fli_bitmap_ | (uint32_t(1) << fli);
}

void ElasticPool::mutable_remove_free(int block_idx) {
    MutableBlock& b = mut_blocks_[block_idx] ; if (!b.is_free) return ; int fli = 0 ; int sli = 0 ; size_to_bin(b.size, &fli, &sli) ; if (b.prev_free >= 0) mut_blocks_[b.prev_free].next_free = b.next_free ; if (b.next_free >= 0) mut_blocks_[b.next_free].prev_free = b.prev_free ; if (mut_free_heads_[fli][sli] == block_idx) {
        mut_free_heads_[fli][sli] = b.next_free ; if (b.next_free < 0) {
            mut_sli_bitmap_[fli] &= ~(uint32_t(1) << sli) ; if (mut_sli_bitmap_[fli] == 0) {
                mut_fli_bitmap_ &= ~(uint32_t(1) << fli) ; }
        }
    }

    b.prev_free = -1 ; b.next_free = -1 ; b.is_free = false ; }

int ElasticPool::mutable_find_suitable(size_t size) const {
    int fli = 0 ; int sli = 0 ; size_to_bin(size, &fli, &sli) ; uint32_t fmask = mut_fli_bitmap_ & (~uint32_t(0) << fli) ; while (fmask) {
        int fi = __builtin_ctz(fmask) ; uint32_t smask = mut_sli_bitmap_[fi] ; if (fi == fli) smask &= (~uint32_t(0) << sli) ; if (smask) {
            int si = __builtin_ctz(smask) ; return mut_free_heads_[fi][si] ; }
        fmask &= (fmask - 1) ; }
    return -1 ; }

int ElasticPool::mutable_split_block(int block_idx, size_t req_size) {
    MutableBlock& b = mut_blocks_[block_idx] ; if (b.size < req_size) return block_idx ; size_t remain = b.size - req_size ; if (remain < kMutableAlign) return block_idx ; int new_idx = static_cast<int>(mut_blocks_.size()) ; MutableBlock nb ; nb.offset = b.offset + req_size ; nb.size = remain ; nb.prev_phys = block_idx ; nb.next_phys = b.next_phys ; nb.is_free = false ; nb.active = true ; mut_blocks_.push_back(nb) ; if (b.next_phys >= 0) mut_blocks_[b.next_phys].prev_phys = new_idx ; b.next_phys = new_idx ; b.size = req_size ; mutable_insert_free(new_idx) ; return block_idx ; }

void ElasticPool::mutable_tlsf_init() {
    for (auto& row : mut_free_heads_) row.fill(-1) ; mut_sli_bitmap_.fill(0) ; mut_fli_bitmap_ = 0 ; mut_ptr_to_block_.clear() ; mut_blocks_.clear() ; MutableBlock root ; root.offset = 0 ; root.size = cfg_.mutable_bytes ; root.prev_phys = -1 ; root.next_phys = -1 ; root.is_free = false ; root.active = true ; mut_blocks_.push_back(root) ; mut_phys_head_ = 0 ; mutable_insert_free(0) ; }

void* ElasticPool::mutable_alloc(size_t bytes) {
    size_t req = align_up(bytes, kMutableAlign) ; int idx = mutable_find_suitable(req) ; if (idx < 0) {
        throw std::runtime_error("[feRcuda] MUTABLE pool exhausted") ; }

    mutable_remove_free(idx) ; idx = mutable_split_block(idx, req) ; MutableBlock& b = mut_blocks_[idx] ; b.is_free = false ; void* ptr = static_cast<uint8_t*>(mut_base_) + b.offset ; mut_ptr_to_block_[ptr] = idx ; return ptr ; }

void ElasticPool::mutable_free(void* ptr) {
    auto it = mut_ptr_to_block_.find(ptr) ; if (it == mut_ptr_to_block_.end()) return ; int idx = it->second ; mut_ptr_to_block_.erase(it) ; MutableBlock* b = &mut_blocks_[idx] ; if (!b->active || b->is_free) return ; b->is_free = true ; if (b->prev_phys >= 0) {
        int prev_idx = b->prev_phys ; MutableBlock& p = mut_blocks_[prev_idx] ; if (p.active && p.is_free) {
            mutable_remove_free(prev_idx) ; p.size += b->size ; p.next_phys = b->next_phys ; if (b->next_phys >= 0) mut_blocks_[b->next_phys].prev_phys = prev_idx ; b->active = false ; b->is_free = false ; b = &p ; idx = prev_idx ; }
    }

    if (b->next_phys >= 0) {
        int next_idx = b->next_phys ; MutableBlock& n = mut_blocks_[next_idx] ; if (n.active && n.is_free) {
            mutable_remove_free(next_idx) ; b->size += n.size ; b->next_phys = n.next_phys ; if (n.next_phys >= 0) mut_blocks_[n.next_phys].prev_phys = idx ; n.active = false ; n.is_free = false ; }
    }

    mutable_insert_free(idx) ; }

ElasticPool::ElasticPool(int device_id, PoolConfig cfg)
    : device_(device_id), cfg_(cfg) {
    CK(cudaSetDevice(device_)) ; if (cfg_.regime == MemoryRegime::CUSTOM_POOL) {
        CK(cudaMalloc(&mut_base_, cfg_.mutable_bytes)) ; CK(cudaMalloc(&imm_base_, cfg_.immutable_bytes)) ; mutable_tlsf_init() ; }

    if (cfg_.verbose) {
        printf("[feRcuda::ElasticPool] regime=%u mutable=%.0fMB immutable=%.0fMB\n",
               (unsigned)cfg_.regime,
               cfg_.mutable_bytes / (1024.0 * 1024.0),
               cfg_.immutable_bytes / (1024.0 * 1024.0)) ; }
}

ElasticPool::~ElasticPool() {
    if (mut_base_) cudaFree(mut_base_) ; if (imm_base_) cudaFree(imm_base_) ; std::lock_guard<std::mutex> lk(live_lock_) ; for (const auto& meta : live_) {
        if (meta.ptr && meta.regime != MemoryRegime::CUSTOM_POOL) cudaFree(meta.ptr) ; }
    live_.clear() ; }

void* ElasticPool::alloc_raw(size_t bytes, Tier tier, uint32_t tag, MemoryRegime regime) {
    if (bytes == 0) throw std::runtime_error("[feRcuda] zero-byte allocation is invalid") ; void* ptr = nullptr ; if (regime == MemoryRegime::CUSTOM_POOL) {
        if (tier == Tier::MUTABLE) {
            std::lock_guard<std::mutex> lk(mut_lock_) ; ptr = mutable_alloc(bytes) ; } else {
            std::lock_guard<std::mutex> lk(imm_lock_) ; size_t aligned = align_up(imm_offset_, kMutableAlign) ; if (aligned + bytes > cfg_.immutable_bytes) {
                throw std::runtime_error("[feRcuda] IMMUTABLE pool exhausted") ; }
            ptr = static_cast<uint8_t*>(imm_base_) + aligned ; imm_offset_ = aligned + bytes ; }
    } else {
        {
            std::lock_guard<std::mutex> lk(live_lock_) ; const size_t used = (tier == Tier::MUTABLE) ? mutable_live_bytes_ : immutable_live_bytes_ ; const size_t cap = (tier == Tier::MUTABLE) ? cfg_.mutable_bytes : cfg_.immutable_bytes ; if (cap > 0 && used + bytes > cap) {
                throw std::runtime_error("[feRcuda] regime budget exhausted") ; }
        }

        cudaError_t e = cudaSuccess ; if (regime == MemoryRegime::CUDA_MALLOC) {
            e = cudaMalloc(&ptr, bytes) ; } else if (regime == MemoryRegime::CUDA_MANAGED) {
            e = cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal) ; } else {
            throw std::runtime_error("[feRcuda] unknown memory regime") ; }
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("[feRcuda] allocation failed: ") + cudaGetErrorString(e)) ; }
    }

    {
        std::lock_guard<std::mutex> lk(live_lock_) ; live_.push_back({ptr, bytes, tier, tag, regime}) ; if (tier == Tier::MUTABLE) mutable_live_bytes_ += bytes ; else immutable_live_bytes_ += bytes ; }

    return ptr ; }

void ElasticPool::free(void* ptr) {
    if (!ptr) return ; AllocMeta meta{} ; bool found = false ; {
        std::lock_guard<std::mutex> lk(live_lock_) ; for (auto it = live_.begin() ; it != live_.end() ; ++it) {
            if (it->ptr != ptr) continue ; meta = *it ; if (meta.tier == Tier::MUTABLE) mutable_live_bytes_ -= meta.bytes ; else immutable_live_bytes_ -= meta.bytes ; live_.erase(it) ; found = true ; break ; }
    }
    if (!found) return ; if (meta.regime == MemoryRegime::CUSTOM_POOL) {
        if (meta.tier == Tier::MUTABLE) {
            std::lock_guard<std::mutex> lk(mut_lock_) ; mutable_free(ptr) ; }
        return ; }

    cudaFree(ptr) ; }

ElasticPool::Stats ElasticPool::stats() const {
    size_t free_vram = 0 ; size_t total_vram = 0 ; cudaMemGetInfo(&free_vram, &total_vram) ; size_t mutable_used = 0 ; size_t immutable_used = 0 ; int live_allocs = 0 ; {
        std::lock_guard<std::mutex> lk(live_lock_) ; mutable_used = mutable_live_bytes_ ; immutable_used = (cfg_.regime == MemoryRegime::CUSTOM_POOL) ? imm_offset_ : immutable_live_bytes_ ; live_allocs = static_cast<int>(live_.size()) ; }

    return {
        .mutable_used = mutable_used,
        .mutable_free = (cfg_.mutable_bytes > mutable_used) ? (cfg_.mutable_bytes - mutable_used) : 0,
        .mutable_total = cfg_.mutable_bytes,
        .immutable_used = immutable_used,
        .immutable_total = cfg_.immutable_bytes,
        .vram_free = free_vram,
        .live_allocs = live_allocs,
    } ; }

void ElasticPool::print_stats() const {
    auto s = stats() ; printf("[feRcuda::ElasticPool] mutable %zu/%zu MB  immutable %zu/%zu MB  vram_free=%zu MB  live=%d\n",
           s.mutable_used >> 20,
           s.mutable_total >> 20,
           s.immutable_used >> 20,
           s.immutable_total >> 20,
           s.vram_free >> 20,
           s.live_allocs) ; }

} // namespace fer
