#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "fercuda/core/status.cuh"
#include "fercuda/runtime/regime_manager.cuh"

namespace fer::runtime {

using JobId = uint64_t;

struct TempAlloc {
    void* ptr = nullptr;
    uint32_t regime_id = static_cast<uint32_t>(MemoryRegime::CUSTOM_POOL);
};

class JobManager {
public:
    explicit JobManager(RegimeManager* regime_mgr);
    ~JobManager();

    Status create_job(const std::vector<TempAlloc>& temps, cudaStream_t stream, JobId* out_job);
    Status status(JobId id, bool* done);
    Status wait(JobId id);
    Status shutdown();

private:
    struct JobRecord {
        cudaEvent_t done = nullptr;
        bool finalized = false;
        std::vector<TempAlloc> temps;
    };

    Status finalize_if_done(JobId id, bool* done_now = nullptr);
    Status free_temps(const std::vector<TempAlloc>& temps);

    RegimeManager* regime_mgr_ = nullptr;
    JobId next_job_id_ = 1;
    std::unordered_map<JobId, JobRecord> jobs_;
};

} // namespace fer::runtime
