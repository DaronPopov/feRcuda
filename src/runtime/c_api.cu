#include "fercuda/api/c_api.h"
#include "fercuda/jit/api.h"
#include "fercuda/jit/manager.cuh"

#include "fercuda/runtime/session.cuh"

#include <memory>
#include <new>
#include <unordered_map>

using fer::PoolConfig;
using fer::MemoryRegime;
using fer::Status;
using fer::StatusCode;
using fer::Tier;
using fer::runtime::BufferDesc;
using fer::runtime::BufferDType;
using fer::runtime::CustomAllocator;
using fer::runtime::LayerNormRequest;
using fer::runtime::MatmulRequest;
using fer::runtime::RuntimeSession;
using fer::jit::JitManager;

struct CAllocatorBridge {
    fer_allocator_alloc_fn alloc = nullptr;
    fer_allocator_free_fn free = nullptr;
    void* user_ctx = nullptr;
};

struct CGPUHotBridge {
    void* runtime = nullptr;
    fer_gpu_hot_alloc_fn alloc = nullptr;
    fer_gpu_hot_free_fn free = nullptr;
};

struct fer_session {
    RuntimeSession* impl;
    std::unordered_map<uint32_t, CAllocatorBridge> allocators;
    std::unordered_map<uint32_t, CGPUHotBridge> gpu_hot_allocators;
    std::unique_ptr<JitManager> jit;
};

namespace {

fer_status_t make_status(Status s) {
    return fer_status_t{static_cast<int32_t>(s.code), s.message};
}

fer_status_t ok_status() {
    return make_status(Status::ok_status());
}

void* bridge_alloc(size_t bytes, Tier tier, uint32_t tag, void* user_ctx) {
    auto* b = static_cast<CAllocatorBridge*>(user_ctx);
    if (!b || !b->alloc) return nullptr;
    const uint32_t c_tier = (tier == Tier::IMMUTABLE) ? FER_ALLOC_TIER_IMMUTABLE : FER_ALLOC_TIER_MUTABLE;
    return b->alloc(static_cast<uint64_t>(bytes), c_tier, tag, b->user_ctx);
}

void bridge_free(void* ptr, void* user_ctx) {
    auto* b = static_cast<CAllocatorBridge*>(user_ctx);
    if (!b || !b->free) return;
    b->free(ptr, b->user_ctx);
}

void* bridge_gpu_hot_alloc(size_t bytes, Tier, uint32_t, void* user_ctx) {
    auto* b = static_cast<CGPUHotBridge*>(user_ctx);
    if (!b || !b->alloc) return nullptr;
    return b->alloc(b->runtime, static_cast<uint64_t>(bytes));
}

void bridge_gpu_hot_free(void* ptr, void* user_ctx) {
    auto* b = static_cast<CGPUHotBridge*>(user_ctx);
    if (!b || !b->free) return;
    b->free(b->runtime, ptr);
}

BufferDesc to_buffer_desc(const fer_buffer_desc_t& in) {
    BufferDesc out{};
    switch (in.dtype) {
        case FER_DTYPE_F32: out.dtype = BufferDType::F32; break;
        case FER_DTYPE_F16: out.dtype = BufferDType::F16; break;
        case FER_DTYPE_BF16: out.dtype = BufferDType::BF16; break;
        case FER_DTYPE_I8: out.dtype = BufferDType::I8; break;
        case FER_DTYPE_U8: out.dtype = BufferDType::U8; break;
        case FER_DTYPE_I16: out.dtype = BufferDType::I16; break;
        case FER_DTYPE_U16: out.dtype = BufferDType::U16; break;
        case FER_DTYPE_I32: out.dtype = BufferDType::I32; break;
        case FER_DTYPE_U32: out.dtype = BufferDType::U32; break;
        case FER_DTYPE_I64: out.dtype = BufferDType::I64; break;
        case FER_DTYPE_U64: out.dtype = BufferDType::U64; break;
        case FER_DTYPE_F64: out.dtype = BufferDType::F64; break;
        default: out.dtype = BufferDType::F32; break;
    }
    out.rank = static_cast<uint8_t>(in.rank);
    out.dims = {in.dims[0], in.dims[1], in.dims[2], in.dims[3]};
    out.immutable = (in.immutable != 0);
    out.tag = in.tag;
    return out;
}

PoolConfig to_pool_config(const fer_pool_config_t& in) {
    PoolConfig out{};
    out.mutable_bytes = static_cast<size_t>(in.mutable_bytes);
    out.immutable_bytes = static_cast<size_t>(in.immutable_bytes);
    out.cuda_reserve = static_cast<size_t>(in.cuda_reserve);
    out.verbose = (in.verbose != 0);
    switch (in.memory_regime) {
        case FER_MEMORY_CUSTOM_POOL:
            out.regime = MemoryRegime::CUSTOM_POOL;
            break;
        case FER_MEMORY_CUDA_MALLOC:
            out.regime = MemoryRegime::CUDA_MALLOC;
            break;
        case FER_MEMORY_CUDA_MANAGED:
            out.regime = MemoryRegime::CUDA_MANAGED;
            break;
        default:
            out.regime = MemoryRegime::CUSTOM_POOL;
            break;
    }
    return out;
}

fer_buffer_desc_t to_buffer_desc_c(const fer_tensor_spec_t& in) {
    fer_buffer_desc_t out{};
    out.dtype = in.dtype;
    out.rank = in.rank;
    out.dims[0] = in.dims[0];
    out.dims[1] = in.dims[1];
    out.dims[2] = in.dims[2];
    out.dims[3] = in.dims[3];
    out.immutable = in.immutable;
    out.tag = in.tag;
    return out;
}

} // namespace

extern "C" fer_status_t fer_session_create(int32_t device, const fer_pool_config_t* cfg, fer_session_t** out_session) {
    if (!out_session) return make_status(Status::invalid_argument("out_session is null"));
    try {
        PoolConfig pcfg{};
        if (cfg) pcfg = to_pool_config(*cfg);
        RuntimeSession* impl = new RuntimeSession(device, pcfg);
        fer_session_t* s = new fer_session{impl, {}, {}, std::make_unique<JitManager>()};
        *out_session = s;
        return ok_status();
    } catch (const std::bad_alloc&) {
        return make_status(Status::internal_error("allocation failure"));
    } catch (...) {
        return make_status(Status::internal_error("session creation failure"));
    }
}

extern "C" fer_status_t fer_session_destroy(fer_session_t* session) {
    if (!session) return ok_status();
    try {
        delete session->impl;
        session->impl = nullptr;
        delete session;
        return ok_status();
    } catch (...) {
        return make_status(Status::internal_error("session destroy failure"));
    }
}

extern "C" fer_status_t fer_alloc_buffer(fer_session_t* session, const fer_buffer_desc_t* desc, uint64_t* out_buffer_id) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    if (!desc) return make_status(Status::invalid_argument("desc is null"));
    return make_status(session->impl->alloc_buffer(to_buffer_desc(*desc), out_buffer_id));
}

extern "C" fer_status_t fer_register_custom_allocator(fer_session_t* session, uint32_t regime_id, const fer_custom_allocator_t* allocator) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    if (!allocator) return make_status(Status::invalid_argument("allocator is null"));
    if (!allocator->alloc || !allocator->free) {
        return make_status(Status::invalid_argument("allocator callbacks must be non-null"));
    }
    CAllocatorBridge bridge{};
    bridge.alloc = allocator->alloc;
    bridge.free = allocator->free;
    bridge.user_ctx = allocator->user_ctx;
    auto it = session->allocators.insert_or_assign(regime_id, bridge).first;

    CustomAllocator ca{};
    ca.alloc = &bridge_alloc;
    ca.free = &bridge_free;
    ca.user_ctx = static_cast<void*>(&it->second);
    return make_status(session->impl->register_custom_allocator(regime_id, ca));
}

extern "C" fer_status_t fer_register_gpu_hot_allocator(fer_session_t* session, uint32_t regime_id, const fer_gpu_hot_allocator_t* allocator) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    if (!allocator) return make_status(Status::invalid_argument("allocator is null"));
    if (!allocator->alloc || !allocator->free) {
        return make_status(Status::invalid_argument("allocator callbacks must be non-null"));
    }
    CGPUHotBridge bridge{};
    bridge.runtime = allocator->runtime;
    bridge.alloc = allocator->alloc;
    bridge.free = allocator->free;
    auto it = session->gpu_hot_allocators.insert_or_assign(regime_id, bridge).first;

    CustomAllocator ca{};
    ca.alloc = &bridge_gpu_hot_alloc;
    ca.free = &bridge_gpu_hot_free;
    ca.user_ctx = static_cast<void*>(&it->second);
    return make_status(session->impl->register_custom_allocator(regime_id, ca));
}

extern "C" fer_status_t fer_unregister_custom_allocator(fer_session_t* session, uint32_t regime_id) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    Status st = session->impl->unregister_custom_allocator(regime_id);
    if (st.ok()) {
        session->allocators.erase(regime_id);
        session->gpu_hot_allocators.erase(regime_id);
    }
    return make_status(st);
}

extern "C" fer_status_t fer_set_default_memory_regime(fer_session_t* session, uint32_t regime_id) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->impl->set_default_memory_regime(regime_id));
}

extern "C" fer_status_t fer_get_default_memory_regime(fer_session_t* session, uint32_t* out_regime_id) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->impl->get_default_memory_regime(out_regime_id));
}

extern "C" fer_status_t fer_list_registered_regimes(
    fer_session_t* session, uint32_t* out_regime_ids, size_t capacity, size_t* out_count) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->impl->list_registered_regimes(out_regime_ids, capacity, out_count));
}

extern "C" fer_status_t fer_alloc_buffer_in_regime(fer_session_t* session, const fer_buffer_desc_t* desc, uint32_t regime_id, uint64_t* out_buffer_id) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    if (!desc) return make_status(Status::invalid_argument("desc is null"));
    return make_status(session->impl->alloc_buffer_with_regime(to_buffer_desc(*desc), regime_id, out_buffer_id));
}

extern "C" fer_status_t fer_import_external_buffer(
    fer_session_t* session,
    const fer_buffer_desc_t* desc,
    void* device_ptr,
    uint32_t regime_id,
    uint64_t* out_buffer_id) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    if (!desc) return make_status(Status::invalid_argument("desc is null"));
    return make_status(session->impl->import_external_buffer(to_buffer_desc(*desc), device_ptr, regime_id, out_buffer_id));
}

extern "C" fer_status_t fer_import_external_buffer_with_deleter(
    fer_session_t* session,
    const fer_buffer_desc_t* desc,
    void* device_ptr,
    uint32_t regime_id,
    fer_external_buffer_deleter_fn deleter,
    void* deleter_user_ctx,
    uint64_t* out_buffer_id) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    if (!desc) return make_status(Status::invalid_argument("desc is null"));
    return make_status(session->impl->import_external_buffer_with_deleter(
        to_buffer_desc(*desc), device_ptr, regime_id, deleter, deleter_user_ctx, out_buffer_id));
}

extern "C" fer_status_t fer_export_buffer_device_ptr(
    fer_session_t* session,
    uint64_t buffer_id,
    void** out_device_ptr) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->impl->export_buffer_device_ptr(buffer_id, out_device_ptr));
}

extern "C" fer_status_t fer_free_buffer(fer_session_t* session, uint64_t buffer_id) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->impl->free_buffer(buffer_id));
}

extern "C" fer_status_t fer_upload_f32(fer_session_t* session, uint64_t buffer_id, const float* host, size_t count) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->impl->upload_f32(buffer_id, host, count));
}

extern "C" fer_status_t fer_download_f32(fer_session_t* session, uint64_t buffer_id, float* host, size_t count) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->impl->download_f32(buffer_id, host, count));
}

extern "C" fer_status_t fer_upload_bytes(fer_session_t* session, uint64_t buffer_id, const void* host, size_t bytes) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->impl->upload_bytes(buffer_id, host, bytes));
}

extern "C" fer_status_t fer_download_bytes(fer_session_t* session, uint64_t buffer_id, void* host, size_t bytes) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->impl->download_bytes(buffer_id, host, bytes));
}

extern "C" fer_status_t fer_submit_matmul(fer_session_t* session, const fer_matmul_request_t* req, uint64_t* out_job_id) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    if (!req) return make_status(Status::invalid_argument("request is null"));
    MatmulRequest m{req->a, req->b, req->out, req->memory_regime};
    return make_status(session->impl->submit_matmul(m, out_job_id));
}

extern "C" fer_status_t fer_submit_layer_norm(fer_session_t* session, const fer_layer_norm_request_t* req, uint64_t* out_job_id) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    if (!req) return make_status(Status::invalid_argument("request is null"));
    LayerNormRequest l{req->x, req->out, req->eps, req->memory_regime};
    return make_status(session->impl->submit_layer_norm(l, out_job_id));
}

extern "C" fer_status_t fer_set_session_stream(fer_session_t* session, uint64_t stream_handle) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_handle));
    return make_status(session->impl->set_stream(stream));
}

extern "C" fer_status_t fer_get_session_stream(fer_session_t* session, uint64_t* out_stream_handle) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    if (!out_stream_handle) return make_status(Status::invalid_argument("out_stream_handle is null"));
    cudaStream_t stream = nullptr;
    Status st = session->impl->get_stream(&stream);
    if (!st.ok()) return make_status(st);
    *out_stream_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
    return ok_status();
}

extern "C" fer_status_t fer_job_status(fer_session_t* session, uint64_t job_id, uint8_t* out_done) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    if (!out_done) return make_status(Status::invalid_argument("out_done is null"));
    bool done = false;
    Status st = session->impl->job_status(job_id, &done);
    *out_done = done ? 1u : 0u;
    return make_status(st);
}

extern "C" fer_status_t fer_job_wait(fer_session_t* session, uint64_t job_id) {
    if (!session || !session->impl) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->impl->job_wait(job_id));
}

extern "C" fer_status_t fer_stream_attach(fer_session_t* session, uint64_t stream_handle) {
    return fer_set_session_stream(session, stream_handle);
}

extern "C" fer_status_t fer_stream_current(fer_session_t* session, uint64_t* out_stream_handle) {
    return fer_get_session_stream(session, out_stream_handle);
}

extern "C" fer_status_t fer_tensor_create(
    fer_session_t* session,
    const fer_tensor_spec_t* spec,
    fer_tensor_handle_t* out_tensor) {
    if (!out_tensor) return make_status(Status::invalid_argument("out_tensor is null"));
    out_tensor->id = 0;
    if (!spec) return make_status(Status::invalid_argument("spec is null"));
    fer_buffer_desc_t desc = to_buffer_desc_c(*spec);
    uint64_t id = 0;
    fer_status_t st = fer_alloc_buffer_in_regime(
        session, &desc, spec->memory_regime, &id);
    if (st.code != FER_STATUS_OK) return st;
    out_tensor->id = id;
    return ok_status();
}

extern "C" fer_status_t fer_tensor_attach_external(
    fer_session_t* session,
    const fer_tensor_spec_t* spec,
    void* device_ptr,
    fer_external_buffer_deleter_fn deleter,
    void* deleter_user_ctx,
    fer_tensor_handle_t* out_tensor) {
    if (!out_tensor) return make_status(Status::invalid_argument("out_tensor is null"));
    out_tensor->id = 0;
    if (!spec) return make_status(Status::invalid_argument("spec is null"));
    fer_buffer_desc_t desc = to_buffer_desc_c(*spec);
    uint64_t id = 0;
    fer_status_t st = fer_import_external_buffer_with_deleter(
        session, &desc, device_ptr, spec->memory_regime, deleter, deleter_user_ctx, &id);
    if (st.code != FER_STATUS_OK) return st;
    out_tensor->id = id;
    return ok_status();
}

extern "C" fer_status_t fer_tensor_device_ptr(
    fer_session_t* session,
    fer_tensor_handle_t tensor,
    void** out_device_ptr) {
    return fer_export_buffer_device_ptr(session, tensor.id, out_device_ptr);
}

extern "C" fer_status_t fer_tensor_release(
    fer_session_t* session,
    fer_tensor_handle_t tensor) {
    return fer_free_buffer(session, tensor.id);
}

extern "C" fer_status_t fer_tensor_upload(
    fer_session_t* session,
    fer_tensor_handle_t tensor,
    const void* host,
    size_t bytes) {
    return fer_upload_bytes(session, tensor.id, host, bytes);
}

extern "C" fer_status_t fer_tensor_download(
    fer_session_t* session,
    fer_tensor_handle_t tensor,
    void* host,
    size_t bytes) {
    return fer_download_bytes(session, tensor.id, host, bytes);
}

extern "C" fer_status_t fer_tensor_run_affine_f32(
    fer_session_t* session,
    fer_tensor_handle_t input,
    fer_tensor_handle_t output,
    uint32_t n,
    float alpha,
    float beta,
    uint32_t fusion_mask,
    uint32_t caps_mask,
    uint32_t memory_regime,
    uint64_t* out_job_id) {
    fer_jit_intent_t intent{};
    intent.abi_version = FER_JIT_INTENT_ABI_VERSION;
    intent.op = FER_JIT_INTENT_OP_AFFINE_F32;
    intent.fusion_mask = fusion_mask;
    intent.caps_mask = caps_mask;
    intent.memory_regime = memory_regime;
    intent.n = n;
    intent.alpha = alpha;
    intent.beta = beta;
    fer_jit_intent_bindings_t bindings{};
    bindings.input = input.id;
    bindings.output = output.id;
    return fer_jit_run_intent(session, &intent, &bindings, out_job_id);
}

extern "C" fer_status_t fer_jit_compile(
    fer_session_t* session,
    const fer_jit_source_t* source,
    const fer_jit_options_t* options,
    fer_jit_program_t* out_program,
    fer_jit_compile_result_t* out_result) {
    if (!session || !session->jit) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->jit->compile(session->impl, source, options, out_program, out_result));
}

extern "C" fer_status_t fer_jit_release_program(
    fer_session_t* session,
    fer_jit_program_t program) {
    if (!session || !session->jit) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->jit->release_program(program));
}

extern "C" fer_status_t fer_jit_get_kernel(
    fer_session_t* session,
    fer_jit_program_t program,
    const char* kernel_name,
    const fer_jit_kernel_sig_t* signature,
    fer_jit_kernel_t* out_kernel) {
    if (!session || !session->jit) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->jit->get_kernel(program, kernel_name, signature, out_kernel));
}

extern "C" fer_status_t fer_jit_release_kernel(
    fer_session_t* session,
    fer_jit_kernel_t kernel) {
    if (!session || !session->jit) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->jit->release_kernel(kernel));
}

extern "C" fer_status_t fer_jit_launch(
    fer_session_t* session,
    fer_jit_kernel_t kernel,
    const fer_jit_launch_cfg_t* cfg,
    const fer_jit_arg_pack_t* args,
    uint64_t* out_job_id) {
    if (!session || !session->jit) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->jit->launch(session->impl, kernel, cfg, args, out_job_id));
}

extern "C" fer_status_t fer_jit_cache_clear(
    fer_session_t* session) {
    if (!session || !session->jit) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->jit->cache_clear());
}

extern "C" fer_status_t fer_jit_get_stats(
    fer_session_t* session,
    fer_jit_stats_t* out_stats) {
    if (!session || !session->jit) return make_status(Status::invalid_argument("session is null"));
    return make_status(session->jit->get_stats(out_stats));
}
