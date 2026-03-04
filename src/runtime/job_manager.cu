#include "fercuda/runtime/job_manager.cuh"

#include <cuda_runtime.h>

namespace fer::runtime {

JobManager::JobManager(RegimeManager* regime_mgr)
    : regime_mgr_(regime_mgr) {}

JobManager::~JobManager() {
    shutdown();
}

Status JobManager::free_temps(const std::vector<TempAlloc>& temps) {
    if (!regime_mgr_) return Status::internal_error("job manager regime manager is null");
    for (const auto& t : temps) {
        Status st = regime_mgr_->free_bytes(t.ptr, t.regime_id);
        if (!st.ok()) return st;
    }
    return Status::ok_status();
}

Status JobManager::create_job(const std::vector<TempAlloc>& temps, cudaStream_t stream, JobId* out_job) {
    if (!out_job) return Status::invalid_argument("out_job is null");
    JobRecord rec{};
    rec.temps = temps;
    cudaError_t e = cudaEventCreateWithFlags(&rec.done, cudaEventDisableTiming);
    if (e != cudaSuccess) return Status::internal_error("cudaEventCreate failed");
    e = cudaEventRecord(rec.done, stream);
    if (e != cudaSuccess) {
        cudaEventDestroy(rec.done);
        return Status::internal_error("cudaEventRecord failed");
    }
    JobId id = next_job_id_++;
    jobs_[id] = rec;
    *out_job = id;
    return Status::ok_status();
}

Status JobManager::finalize_if_done(JobId id, bool* done_now) {
    auto it = jobs_.find(id);
    if (it == jobs_.end()) return Status::not_found("job id not found");
    JobRecord& jr = it->second;
    if (jr.finalized) {
        if (done_now) *done_now = true;
        return Status::ok_status();
    }

    cudaError_t e = cudaEventQuery(jr.done);
    if (e == cudaErrorNotReady) {
        if (done_now) *done_now = false;
        return Status::ok_status();
    }
    if (e != cudaSuccess) return Status::internal_error("cudaEventQuery failed");

    Status ft = free_temps(jr.temps);
    if (!ft.ok()) return ft;
    jr.temps.clear();
    jr.finalized = true;
    if (done_now) *done_now = true;
    return Status::ok_status();
}

Status JobManager::status(JobId id, bool* done) {
    if (!done) return Status::invalid_argument("done pointer is null");
    return finalize_if_done(id, done);
}

Status JobManager::wait(JobId id) {
    auto it = jobs_.find(id);
    if (it == jobs_.end()) return Status::not_found("job id not found");
    cudaError_t e = cudaEventSynchronize(it->second.done);
    if (e != cudaSuccess) return Status::internal_error("cudaEventSynchronize failed");
    bool done = false;
    return finalize_if_done(id, &done);
}

Status JobManager::shutdown() {
    for (auto& kv : jobs_) {
        if (!kv.second.done) continue;
        cudaEventSynchronize(kv.second.done);
        Status st = free_temps(kv.second.temps);
        if (!st.ok()) return st;
        kv.second.temps.clear();
        cudaEventDestroy(kv.second.done);
        kv.second.done = nullptr;
    }
    jobs_.clear();
    return Status::ok_status();
}

} // namespace fer::runtime
