// Complete PyTorch TLSF Allocator Implementation
// Implements ALL required methods from c10::cuda::CUDACachingAllocator::CUDAAllocator
// Compatible with PyTorch 2.9.0+

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <memory>

// Rust FFI functions
extern "C" {
    void tlsf_init_ffi(int device_id, size_t block_size, double pool_fraction);
    void* tlsf_alloc_ffi(size_t size);
    void tlsf_free_ffi(void* ptr);
    void tlsf_print_stats_ffi();
}

namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {

// Bring Stat/StatType into scope from their 2.9.0 location (c10::CachingAllocator)
using c10::CachingAllocator::Stat;
using c10::CachingAllocator::StatType;
using c10::CachingAllocator::StatArray;

class TLSFAllocator : public CUDAAllocator {
private:
    int device_id_;
    bool initialized_;
    bool enabled_;
    std::mutex mutex_;
    std::unordered_map<void*, size_t> allocations_;
    size_t peak_allocated_;
    size_t current_allocated_;

public:
    TLSFAllocator() : device_id_(0), initialized_(false), enabled_(true),
                       peak_allocated_(0), current_allocated_(0) {}

    // REQUIRED: Basic allocation
    void* raw_alloc(size_t nbytes) override {
        void* ptr = tlsf_alloc_ffi(nbytes);
        if (ptr) {
            std::lock_guard<std::mutex> lock(mutex_);
            allocations_[ptr] = nbytes;
            current_allocated_ += nbytes;
            if (current_allocated_ > peak_allocated_) {
                peak_allocated_ = current_allocated_;
            }
        }
        return ptr;
    }

    // REQUIRED: Allocation with stream
    void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
        // TLSF doesn't need stream info, just allocate
        (void)stream;  // Unused
        return raw_alloc(nbytes);
    }

    // REQUIRED: Free
    void raw_delete(void* ptr) override {
        if (!ptr) return;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = allocations_.find(ptr);
            if (it != allocations_.end()) {
                current_allocated_ -= it->second;
                allocations_.erase(it);
            }
        }

        tlsf_free_ffi(ptr);
    }

    // REQUIRED: Initialize
    void init(int /*device_count*/) override {
        if (!initialized_) {
            device_id_ = 0;  // Use first device
            tlsf_init_ffi(device_id_, 2 * 1024 * 1024, 0.70);
            initialized_ = true;
            std::cout << "[aten-ptx] TLSF allocator initialized" << std::endl;
        }
    }

    // REQUIRED: Check if initialized
    bool initialized() override {
        return initialized_;
    }

    // REQUIRED: Get memory fraction
    double getMemoryFraction(c10::DeviceIndex device) override {
        (void)device;
        return 0.70;  // Report our configured pool fraction
    }

    // REQUIRED: Set memory fraction
    void setMemoryFraction(double fraction, c10::DeviceIndex device) override {
        // Could reinitialize with new fraction, but for now just note it
        (void)fraction;
        (void)device;
    }

    // REQUIRED: Enable/disable allocator
    void enable(bool value) override {
        enabled_ = value;
    }

    // REQUIRED: Check if enabled
    bool isEnabled() const override {
        return enabled_;
    }

    // REQUIRED: Empty cache (signature changed in 2.9.0 — now takes MempoolId_t)
    void emptyCache(MempoolId_t mempool_id = {0, 0}) override {
        (void)mempool_id;
        // TLSF doesn't cache — no-op
    }

    // REQUIRED: Cache info
    void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) override {
        (void)device;
        *largestBlock = 0;  // TLSF doesn't track this
    }

    // REQUIRED: Get base allocation
    void* getBaseAllocation(void* ptr, size_t* size) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            *size = it->second;
            return ptr;
        }
        *size = 0;
        return nullptr;
    }

    // REQUIRED: Record stream
    void recordStream(const DataPtr& ptr, CUDAStream stream) override {
        // TLSF manages memory directly, no stream tracking needed
        (void)ptr;
        (void)stream;
    }

    // REQUIRED: Get device stats
    c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override {
        (void)device;
        c10::CachingDeviceAllocator::DeviceStats stats;

        std::lock_guard<std::mutex> lock(mutex_);

        // Initialize stats properly - StatArray is std::array<Stat, 3>
        Stat allocated_stat;
        allocated_stat.current = current_allocated_;
        allocated_stat.peak = peak_allocated_;

        Stat zero_stat;
        zero_stat.current = 0;
        zero_stat.peak = 0;

        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)] = allocated_stat;
        stats.allocated_bytes[static_cast<size_t>(StatType::SMALL_POOL)] = zero_stat;
        stats.allocated_bytes[static_cast<size_t>(StatType::LARGE_POOL)] = zero_stat;

        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)] = allocated_stat;
        stats.reserved_bytes[static_cast<size_t>(StatType::SMALL_POOL)] = zero_stat;
        stats.reserved_bytes[static_cast<size_t>(StatType::LARGE_POOL)] = zero_stat;

        stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)] = allocated_stat;
        stats.active_bytes[static_cast<size_t>(StatType::SMALL_POOL)] = zero_stat;
        stats.active_bytes[static_cast<size_t>(StatType::LARGE_POOL)] = zero_stat;

        stats.inactive_split_bytes[static_cast<size_t>(StatType::AGGREGATE)] = zero_stat;
        stats.inactive_split_bytes[static_cast<size_t>(StatType::SMALL_POOL)] = zero_stat;
        stats.inactive_split_bytes[static_cast<size_t>(StatType::LARGE_POOL)] = zero_stat;

        stats.num_alloc_retries = 0;
        stats.num_ooms = 0;

        return stats;
    }

    // REQUIRED: Reset accumulated stats
    void resetAccumulatedStats(c10::DeviceIndex device) override {
        (void)device;
        // No-op
    }

    // REQUIRED: Reset peak stats
    void resetPeakStats(c10::DeviceIndex device) override {
        (void)device;
        std::lock_guard<std::mutex> lock(mutex_);
        peak_allocated_ = current_allocated_;
    }

    // REQUIRED: Snapshot (signature changed in 2.9.0 — now takes MempoolId_t)
    SnapshotInfo snapshot(MempoolId_t mempool_id = {0, 0}) override {
        (void)mempool_id;
        return SnapshotInfo();  // Return empty snapshot
    }

    // REQUIRED: Begin allocate to pool
    void beginAllocateToPool(
        c10::DeviceIndex device,
        MempoolId_t mempool_id,
        std::function<bool(cudaStream_t)> filter) override {
        (void)device;
        (void)mempool_id;
        (void)filter;
        // TLSF doesn't use pools
    }

    // REQUIRED: End allocate to pool
    void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) override {
        (void)device;
        (void)mempool_id;
    }

    // REQUIRED: Release pool
    void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) override {
        (void)device;
        (void)mempool_id;
    }

    // REQUIRED: Share IPC handle (new in 2.9.0)
    ShareableHandle shareIpcHandle(void* ptr) override {
        (void)ptr;
        return ShareableHandle{0, ""};  // IPC not supported
    }

    // REQUIRED: Get IPC device pointer
    std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
        (void)handle;
        return nullptr;  // IPC not supported
    }

    // REQUIRED: Record history (signature changed in 2.9.0 — added clearHistory param)
    void recordHistory(
        bool enabled,
        CreateContextFn context_recorder,
        size_t alloc_trace_max_entries,
        RecordContext when,
        bool clearHistory) override {
        (void)enabled;
        (void)context_recorder;
        (void)alloc_trace_max_entries;
        (void)when;
        (void)clearHistory;
        // History recording not implemented
    }

    // REQUIRED: Attach OOM observer
    void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
        (void)observer;
    }

    // REQUIRED: Attach trace tracker
    void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) override {
        (void)tracker;
    }

    // REQUIRED: Enable peer access
    void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) override {
        (void)dev;
        (void)dev_to_access;
    }

    // REQUIRED: Async memcpy
    cudaError_t memcpyAsync(
        void* dst,
        int dstDevice,
        const void* src,
        int srcDevice,
        size_t count,
        cudaStream_t stream,
        bool p2p_enabled) override {
        (void)dstDevice;
        (void)srcDevice;
        (void)p2p_enabled;
        return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
    }

    // REQUIRED: Get checkpoint state
    std::shared_ptr<AllocatorState> getCheckpointState(
        c10::DeviceIndex device,
        MempoolId_t id) override {
        (void)device;
        (void)id;
        return nullptr;
    }

    // REQUIRED: Set checkpoint pool state
    CheckpointDelta setCheckpointPoolState(
        c10::DeviceIndex device,
        std::shared_ptr<AllocatorState> pps) override {
        (void)device;
        (void)pps;
        return CheckpointDelta();
    }

    // REQUIRED: Allocator name
    std::string name() override {
        return "TLSF";
    }

    // REQUIRED: Return raw deleter function pointer
    DeleterFnPtr raw_deleter() const override {
        return &tlsf_free_ffi;
    }

    // REQUIRED: Base Allocator::allocate method
    DataPtr allocate(size_t nbytes) override {
        void* ptr = raw_alloc(nbytes);
        if (!ptr) {
            return DataPtr(nullptr, Device(DeviceType::CUDA, device_id_));
        }

        // Return DataPtr with raw deleter function pointer
        return DataPtr(
            ptr,
            ptr,
            raw_deleter(),
            Device(DeviceType::CUDA, device_id_)
        );
    }

    // REQUIRED: Base Allocator::copy_data method
    void copy_data(void* dest, const void* src, std::size_t count) const override {
        cudaError_t err = cudaMemcpy(dest, src, count, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[aten-ptx] cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        }
    }
};

// Global instance
static TLSFAllocator* g_tlsf_allocator = nullptr;

} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10

// C interface to initialize
extern "C" {
    void aten_tlsf_init(int /*device_id*/) {
        using namespace c10::cuda::CUDACachingAllocator;

        if (!g_tlsf_allocator) {
            g_tlsf_allocator = new TLSFAllocator();
            g_tlsf_allocator->init(1);

            // Set the global allocator pointer directly!
            allocator.store(g_tlsf_allocator, std::memory_order_release);

            std::cout << "[aten-ptx] TLSF allocator registered!" << std::endl;
            std::cout << "[aten-ptx] All PyTorch CUDA ops now use TLSF!" << std::endl;
        }
    }

    void aten_tlsf_stats() {
        tlsf_print_stats_ffi();
    }

    // Stream management: bind a raw cudaStream_t as PyTorch's current stream
    void aten_set_cuda_stream(void* raw_stream, int device_id) {
        auto stream = c10::cuda::getStreamFromExternal(
            static_cast<cudaStream_t>(raw_stream),
            static_cast<c10::DeviceIndex>(device_id)
        );
        c10::cuda::setCurrentCUDAStream(stream);
    }

    // Reset PyTorch to its default CUDA stream
    void aten_reset_default_stream(int device_id) {
        auto stream = c10::cuda::getDefaultCUDAStream(
            static_cast<c10::DeviceIndex>(device_id)
        );
        c10::cuda::setCurrentCUDAStream(stream);
    }
}
