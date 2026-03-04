#include "fercuda/runtime/session.cuh"

#include <cuda_runtime.h>

namespace fer::runtime {

namespace {

size_t numel_from_desc(const BufferDesc& desc) {
    size_t n = 1;
    for (uint8_t i = 0; i < desc.rank; i++) n *= desc.dims[i];
    return n;
}

size_t bytes_for_desc(const BufferDesc& desc) {
    auto bytes_per_elem = [](BufferDType dt) -> size_t {
        switch (dt) {
            case BufferDType::F32: return 4;
            case BufferDType::F16: return 2;
            case BufferDType::BF16: return 2;
            case BufferDType::I8: return 1;
            case BufferDType::U8: return 1;
            case BufferDType::I16: return 2;
            case BufferDType::U16: return 2;
            case BufferDType::I32: return 4;
            case BufferDType::U32: return 4;
            case BufferDType::I64: return 8;
            case BufferDType::U64: return 8;
            case BufferDType::F64: return 8;
            default: return 0;
        }
    };
    const size_t bpe = bytes_per_elem(desc.dtype);
    if (bpe == 0) return 0;
    return numel_from_desc(desc) * bpe;
}

FTensor1D tensor1d_from(void* ptr, const BufferDesc& desc) {
    return FTensor1D(static_cast<F32*>(ptr), Shape<1>(desc.dims[0]));
}

FTensor2D tensor2d_from(void* ptr, const BufferDesc& desc) {
    return FTensor2D(static_cast<F32*>(ptr), Shape<2>(desc.dims[0], desc.dims[1]));
}

bool is_compute_float_dtype(BufferDType dt) {
    return dt == BufferDType::F32 || dt == BufferDType::F16 || dt == BufferDType::BF16;
}

bool is_compute_quant_dtype(BufferDType dt) {
    return dt == BufferDType::I8 || dt == BufferDType::U8;
}

bool is_compute_supported_dtype(BufferDType dt) {
    return is_compute_float_dtype(dt) || is_compute_quant_dtype(dt);
}

} // namespace

RuntimeSession::RuntimeSession(int device, PoolConfig cfg)
    : device_(device),
      pool_(device, cfg),
      regime_mgr_(&pool_, static_cast<uint32_t>(cfg.regime)),
      job_mgr_(&regime_mgr_),
      registry_(algorithms::make_default_registry()),
      exec_planner_(&regime_mgr_, &registry_, &ctx_) {
    ctx_.device = DeviceId(device);
    ctx_.stream = StreamHandle(0);
}

RuntimeSession::~RuntimeSession() {
    job_mgr_.shutdown();
}

Status RuntimeSession::register_custom_allocator(uint32_t regime_id, const CustomAllocator& allocator) {
    return regime_mgr_.register_custom_allocator(regime_id, allocator);
}

Status RuntimeSession::unregister_custom_allocator(uint32_t regime_id) {
    return regime_mgr_.unregister_custom_allocator(regime_id);
}

Status RuntimeSession::set_default_memory_regime(uint32_t regime_id) {
    return regime_mgr_.set_default_memory_regime(regime_id);
}

Status RuntimeSession::get_default_memory_regime(uint32_t* out_regime_id) const {
    return regime_mgr_.get_default_memory_regime(out_regime_id);
}

Status RuntimeSession::list_registered_regimes(uint32_t* out_regime_ids, size_t capacity, size_t* out_count) const {
    return regime_mgr_.list_registered_regimes(out_regime_ids, capacity, out_count);
}

Status RuntimeSession::alloc_buffer(const BufferDesc& desc, BufferId* out_id) {
    return alloc_buffer_with_regime(desc, MEMORY_REGIME_AUTO, out_id);
}

Status RuntimeSession::alloc_buffer_with_regime(const BufferDesc& desc, uint32_t regime_id, BufferId* out_id) {
    if (!out_id) return Status::invalid_argument("out_id is null");
    if (desc.rank == 0 || desc.rank > 4) return Status::invalid_argument("rank must be in [1,4]");
    size_t bytes = bytes_for_desc(desc);
    if (bytes == 0) return Status::invalid_argument("unsupported dtype or empty buffer");

    ResolvedRegime rr{};
    Status rs = regime_mgr_.resolve(regime_id, &rr);
    if (!rs.ok()) return rs;

    Tier tier = desc.immutable ? Tier::IMMUTABLE : Tier::MUTABLE;
    void* ptr = nullptr;
    Status as = regime_mgr_.alloc_bytes(bytes, tier, desc.tag, rr, &ptr);
    if (!as.ok()) return as;

    BufferId id = next_buffer_id_++;
    buffers_[id] = BufferRecord{ptr, desc, numel_from_desc(desc), rr.raw};
    *out_id = id;
    return Status::ok_status();
}

Status RuntimeSession::free_buffer(BufferId id) {
    auto it = buffers_.find(id);
    if (it == buffers_.end()) return Status::not_found("buffer id not found");
    Status st = regime_mgr_.free_bytes(it->second.ptr, it->second.regime_id);
    if (!st.ok()) return st;
    buffers_.erase(it);
    return Status::ok_status();
}

Status RuntimeSession::upload_f32(BufferId id, const float* host, size_t count) {
    if (!host) return Status::invalid_argument("host pointer is null");
    auto it = buffers_.find(id);
    if (it == buffers_.end()) return Status::not_found("buffer id not found");
    const BufferRecord& rec = it->second;
    if (rec.desc.dtype != BufferDType::F32) return Status::invalid_argument("buffer is not f32");
    if (count != rec.numel) return Status::invalid_argument("upload count mismatch");
    return upload_bytes(id, host, count * sizeof(float));
}

Status RuntimeSession::upload_bytes(BufferId id, const void* host, size_t bytes) {
    if (!host) return Status::invalid_argument("host pointer is null");
    auto it = buffers_.find(id);
    if (it == buffers_.end()) return Status::not_found("buffer id not found");
    const BufferRecord& rec = it->second;
    const size_t want = bytes_for_desc(rec.desc);
    if (want == 0) return Status::invalid_argument("unsupported dtype");
    if (bytes != want) return Status::invalid_argument("upload byte size mismatch");
    cudaError_t e = cudaMemcpyAsync(rec.ptr, host, bytes, cudaMemcpyHostToDevice, ctx_.stream.value);
    if (e != cudaSuccess) return Status::internal_error("cudaMemcpyAsync H2D failed");
    return Status::ok_status();
}

Status RuntimeSession::download_f32(BufferId id, float* host, size_t count) const {
    if (!host) return Status::invalid_argument("host pointer is null");
    auto it = buffers_.find(id);
    if (it == buffers_.end()) return Status::not_found("buffer id not found");
    const BufferRecord& rec = it->second;
    if (rec.desc.dtype != BufferDType::F32) return Status::invalid_argument("buffer is not f32");
    if (count != rec.numel) return Status::invalid_argument("download count mismatch");
    return download_bytes(id, host, count * sizeof(float));
}

Status RuntimeSession::download_bytes(BufferId id, void* host, size_t bytes) const {
    if (!host) return Status::invalid_argument("host pointer is null");
    auto it = buffers_.find(id);
    if (it == buffers_.end()) return Status::not_found("buffer id not found");
    const BufferRecord& rec = it->second;
    const size_t want = bytes_for_desc(rec.desc);
    if (want == 0) return Status::invalid_argument("unsupported dtype");
    if (bytes != want) return Status::invalid_argument("download byte size mismatch");
    cudaError_t e = cudaMemcpy(host, rec.ptr, bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return Status::internal_error("cudaMemcpy D2H failed");
    return Status::ok_status();
}

Status RuntimeSession::lookup_f32_1d(BufferId id, FTensor1D* out) const {
    if (!out) return Status::invalid_argument("out tensor is null");
    auto it = buffers_.find(id);
    if (it == buffers_.end()) return Status::not_found("buffer id not found");
    const BufferRecord& rec = it->second;
    if (rec.desc.dtype != BufferDType::F32) return Status::invalid_argument("buffer is not f32");
    if (rec.desc.rank != 1) return Status::invalid_argument("buffer rank is not 1");
    *out = tensor1d_from(rec.ptr, rec.desc);
    return Status::ok_status();
}

Status RuntimeSession::lookup_f32_2d(BufferId id, FTensor2D* out) const {
    if (!out) return Status::invalid_argument("out tensor is null");
    auto it = buffers_.find(id);
    if (it == buffers_.end()) return Status::not_found("buffer id not found");
    const BufferRecord& rec = it->second;
    if (rec.desc.dtype != BufferDType::F32) return Status::invalid_argument("buffer is not f32");
    if (rec.desc.rank != 2) return Status::invalid_argument("buffer rank is not 2");
    *out = tensor2d_from(rec.ptr, rec.desc);
    return Status::ok_status();
}

Status RuntimeSession::submit_matmul(const MatmulRequest& req, JobId* out_job) {
    auto ita = buffers_.find(req.a);
    auto itb = buffers_.find(req.b);
    auto ito = buffers_.find(req.out);
    if (ita == buffers_.end() || itb == buffers_.end() || ito == buffers_.end()) {
        return Status::not_found("buffer id not found");
    }
    const BufferRecord& rec_a = ita->second;
    const BufferRecord& rec_b = itb->second;
    const BufferRecord& rec_out = ito->second;
    if (!is_compute_supported_dtype(rec_a.desc.dtype) ||
        !is_compute_supported_dtype(rec_b.desc.dtype) ||
        !is_compute_supported_dtype(rec_out.desc.dtype)) {
        return Status::invalid_argument("matmul requires supported buffers (f32/f16/bf16/i8/u8)");
    }
    if (rec_a.desc.dtype != rec_b.desc.dtype || rec_a.desc.dtype != rec_out.desc.dtype) {
        return Status::invalid_argument("matmul requires matching dtypes across A/B/out");
    }
    if (rec_a.desc.rank != 2 || rec_b.desc.rank != 2 || rec_out.desc.rank != 2) {
        return Status::invalid_argument("matmul requires rank-2 buffers");
    }

    std::vector<TempAlloc> temps;
    Status plan = exec_planner_.launch_matmul(
        ExecBufferView{rec_a.ptr, rec_a.desc, rec_a.regime_id},
        ExecBufferView{rec_b.ptr, rec_b.desc, rec_b.regime_id},
        ExecBufferView{rec_out.ptr, rec_out.desc, rec_out.regime_id},
        req.memory_regime,
        &temps);
    if (!plan.ok()) return plan;

    JobId jid = 0;
    Status js = job_mgr_.create_job(temps, ctx_.stream.value, &jid);
    if (!js.ok()) {
        for (const auto& t : temps) regime_mgr_.free_bytes(t.ptr, t.regime_id);
        return js;
    }
    *out_job = jid;
    return Status::ok_status();
}

Status RuntimeSession::submit_layer_norm(const LayerNormRequest& req, JobId* out_job) {
    auto itx = buffers_.find(req.x);
    auto ito = buffers_.find(req.out);
    if (itx == buffers_.end() || ito == buffers_.end()) return Status::not_found("buffer id not found");
    const BufferRecord& rec_x = itx->second;
    const BufferRecord& rec_out = ito->second;
    if (!is_compute_supported_dtype(rec_x.desc.dtype) || !is_compute_supported_dtype(rec_out.desc.dtype)) {
        return Status::invalid_argument("layer_norm requires supported buffers (f32/f16/bf16/i8/u8)");
    }
    if (rec_x.desc.dtype != rec_out.desc.dtype) {
        return Status::invalid_argument("layer_norm requires matching dtypes for x/out");
    }
    if (rec_x.desc.rank != 1 || rec_out.desc.rank != 1) {
        return Status::invalid_argument("layer_norm requires rank-1 buffers");
    }

    std::vector<TempAlloc> temps;
    Status plan = exec_planner_.launch_layer_norm(
        ExecBufferView{rec_x.ptr, rec_x.desc, rec_x.regime_id},
        ExecBufferView{rec_out.ptr, rec_out.desc, rec_out.regime_id},
        req.eps,
        req.memory_regime,
        &temps);
    if (!plan.ok()) return plan;

    JobId jid = 0;
    Status js = job_mgr_.create_job(temps, ctx_.stream.value, &jid);
    if (!js.ok()) {
        for (const auto& t : temps) regime_mgr_.free_bytes(t.ptr, t.regime_id);
        return js;
    }
    *out_job = jid;
    return Status::ok_status();
}

Status RuntimeSession::job_status(JobId id, bool* done) const {
    RuntimeSession* self = const_cast<RuntimeSession*>(this);
    return self->job_mgr_.status(id, done);
}

Status RuntimeSession::job_wait(JobId id) {
    return job_mgr_.wait(id);
}

} // namespace fer::runtime
