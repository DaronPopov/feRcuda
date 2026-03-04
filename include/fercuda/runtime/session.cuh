#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "fercuda/algorithms/matmul_attachment.cuh"
#include "fercuda/core/runtime.cuh"
#include "fercuda/core/status.cuh"
#include "fercuda/alloc/memory.cuh"
#include "fercuda/runtime/api_types.cuh"
#include "fercuda/runtime/exec_planner.cuh"
#include "fercuda/runtime/job_manager.cuh"
#include "fercuda/runtime/regime_manager.cuh"

namespace fer::runtime {

using BufferId = uint64_t;

class RuntimeSession {
public:
    explicit RuntimeSession(int device = 0, PoolConfig cfg = {});
    ~RuntimeSession();

    Status register_custom_allocator(uint32_t regime_id, const CustomAllocator& allocator);
    Status unregister_custom_allocator(uint32_t regime_id);
    Status set_default_memory_regime(uint32_t regime_id);
    Status get_default_memory_regime(uint32_t* out_regime_id) const;
    Status list_registered_regimes(uint32_t* out_regime_ids, size_t capacity, size_t* out_count) const;

    Status alloc_buffer(const BufferDesc& desc, BufferId* out_id);
    Status alloc_buffer_with_regime(const BufferDesc& desc, uint32_t regime_id, BufferId* out_id);
    Status free_buffer(BufferId id);

    Status upload_bytes(BufferId id, const void* host, size_t bytes);
    Status download_bytes(BufferId id, void* host, size_t bytes) const;
    Status upload_f32(BufferId id, const float* host, size_t count);
    Status download_f32(BufferId id, float* host, size_t count) const;

    Status submit_matmul(const MatmulRequest& req, JobId* out_job);
    Status submit_layer_norm(const LayerNormRequest& req, JobId* out_job);

    Status job_status(JobId id, bool* done) const;
    Status job_wait(JobId id);

private:
    struct BufferRecord {
        void* ptr = nullptr;
        BufferDesc desc{};
        size_t numel = 0;
        uint32_t regime_id = static_cast<uint32_t>(MemoryRegime::CUSTOM_POOL);
    };

    Status lookup_f32_1d(BufferId id, FTensor1D* out) const;
    Status lookup_f32_2d(BufferId id, FTensor2D* out) const;
    static bool same_regime(uint32_t lhs, uint32_t rhs) { return lhs == rhs; }

    int device_ = 0;
    ElasticPool pool_;
    RegimeManager regime_mgr_;
    JobManager job_mgr_;
    ExecPlanner exec_planner_;
    KernelRegistry registry_;
    OpContext ctx_{DeviceId(0), StreamHandle(0)};
    BufferId next_buffer_id_ = 1;
    std::unordered_map<BufferId, BufferRecord> buffers_;
};

} // namespace fer::runtime
