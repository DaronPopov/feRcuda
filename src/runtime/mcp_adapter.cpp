#include "fercuda/agent/mcp_adapter.h"

#include "fercuda/jit/api.h"
#include "fercuda/jit/intent/intent.h"

#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

fer_status_t ok_status() {
    return fer_status_t{FER_STATUS_OK, "ok"};
}

fer_status_t invalid(const char* msg) {
    return fer_status_t{FER_STATUS_INVALID_ARGUMENT, msg};
}

fer_status_t not_found(const char* msg) {
    return fer_status_t{FER_STATUS_NOT_FOUND, msg};
}

size_t dtype_size_bytes(uint32_t dtype) {
    switch (dtype) {
        case FER_DTYPE_F32: return sizeof(float);
        case FER_DTYPE_F16: return sizeof(uint16_t);
        case FER_DTYPE_BF16: return sizeof(uint16_t);
        case FER_DTYPE_I8: return sizeof(int8_t);
        case FER_DTYPE_U8: return sizeof(uint8_t);
        case FER_DTYPE_I16: return sizeof(int16_t);
        case FER_DTYPE_U16: return sizeof(uint16_t);
        case FER_DTYPE_I32: return sizeof(int32_t);
        case FER_DTYPE_U32: return sizeof(uint32_t);
        case FER_DTYPE_I64: return sizeof(int64_t);
        case FER_DTYPE_U64: return sizeof(uint64_t);
        case FER_DTYPE_F64: return sizeof(double);
        default: return 0;
    }
}

struct TensorRecord {
    uint64_t session_id = 0;
    fer_tensor_handle_t handle{};
};

struct JobRecord {
    uint64_t session_id = 0;
};

}  // namespace

struct ProgramRecord {
    uint64_t session_id = 0;
    fer_jit_program_t handle = nullptr;
};

struct fer_agent_adapter {
    std::mutex mu;
    std::atomic<uint64_t> next_session_id{1};
    std::atomic<uint64_t> next_tensor_id{1};
    std::atomic<uint64_t> next_blob_id{1};
    std::atomic<uint64_t> next_program_id{1};

    std::unordered_map<uint64_t, fer_session_t*> sessions;
    std::unordered_map<uint64_t, TensorRecord> tensors;
    std::unordered_map<uint64_t, JobRecord> jobs;
    std::unordered_map<uint64_t, ProgramRecord> programs;
    std::unordered_map<std::string, std::vector<uint8_t>> blobs;

    fer_agent_progress_callback_fn progress_cb = nullptr;
    void* progress_ctx = nullptr;
};

static uint64_t now_us() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

static void emit_progress(fer_agent_adapter* adapter, uint32_t kind,
                          uint64_t session_id, uint64_t entity_id,
                          float fraction, const char* message) {
    if (!adapter || !adapter->progress_cb) return;
    fer_agent_progress_event_t ev{};
    ev.kind = kind;
    ev.session_id = session_id;
    ev.entity_id = entity_id;
    ev.timestamp_us = now_us();
    ev.fraction = fraction;
    ev.message = message;
    adapter->progress_cb(&ev, adapter->progress_ctx);
}

extern "C" fer_status_t fer_agent_adapter_create(fer_agent_adapter_t** out_adapter) {
    if (!out_adapter) return invalid("out_adapter is null");
    *out_adapter = nullptr;
    try {
        *out_adapter = new fer_agent_adapter();
        return ok_status();
    } catch (...) {
        return fer_status_t{FER_STATUS_INTERNAL_ERROR, "agent adapter allocation failed"};
    }
}

extern "C" fer_status_t fer_agent_adapter_destroy(fer_agent_adapter_t* adapter) {
    if (!adapter) return ok_status();
    std::lock_guard<std::mutex> lock(adapter->mu);
    for (auto& kv : adapter->tensors) {
        const uint64_t session_id = kv.second.session_id;
        const auto sit = adapter->sessions.find(session_id);
        if (sit != adapter->sessions.end()) {
            (void)fer_tensor_release(sit->second, kv.second.handle);
        }
    }
    adapter->tensors.clear();
    for (auto& kv : adapter->programs) {
        const auto sit = adapter->sessions.find(kv.second.session_id);
        if (sit != adapter->sessions.end()) {
            (void)fer_jit_release_program(sit->second, kv.second.handle);
        }
    }
    adapter->programs.clear();
    adapter->jobs.clear();
    for (auto& kv : adapter->sessions) {
        (void)fer_session_destroy(kv.second);
    }
    adapter->sessions.clear();
    delete adapter;
    return ok_status();
}

extern "C" fer_status_t fer_agent_runtime_inspect(fer_agent_adapter_t* adapter, fer_agent_runtime_inspect_t* out_info) {
    if (!adapter) return invalid("adapter is null");
    if (!out_info) return invalid("out_info is null");
    out_info->agent_api_version = FER_AGENT_API_VERSION_V1ALPHA1;
    out_info->jit_intent_abi_version = FER_JIT_INTENT_ABI_VERSION;
    out_info->supports_jit_compile = 1u;
    out_info->supports_jit_intent = 1u;
    out_info->supports_external_ptr_import = 1u;
    out_info->supports_session_stream_handoff = 1u;
    return ok_status();
}

extern "C" fer_status_t fer_agent_session_create(
    fer_agent_adapter_t* adapter,
    const fer_agent_session_create_request_t* req,
    uint64_t* out_session_id) {
    if (!adapter) return invalid("adapter is null");
    if (!req) return invalid("req is null");
    if (!out_session_id) return invalid("out_session_id is null");

    fer_pool_config_t cfg{};
    cfg.mutable_bytes = req->mutable_bytes;
    cfg.immutable_bytes = req->immutable_bytes;
    cfg.cuda_reserve = req->cuda_reserve;
    cfg.verbose = req->verbose;
    cfg.memory_regime = req->memory_regime;

    fer_session_t* session = nullptr;
    fer_status_t st = fer_session_create(req->device, &cfg, &session);
    if (st.code != FER_STATUS_OK) return st;

    const uint64_t session_id = adapter->next_session_id.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        adapter->sessions.emplace(session_id, session);
    }
    *out_session_id = session_id;
    return ok_status();
}

extern "C" fer_status_t fer_agent_session_destroy(fer_agent_adapter_t* adapter, uint64_t session_id) {
    if (!adapter) return invalid("adapter is null");
    std::vector<uint64_t> tensor_ids;
    fer_session_t* session = nullptr;
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        auto it = adapter->sessions.find(session_id);
        if (it == adapter->sessions.end()) return not_found("session_id not found");
        session = it->second;
        for (const auto& t : adapter->tensors) {
            if (t.second.session_id == session_id) tensor_ids.push_back(t.first);
        }
        for (uint64_t tid : tensor_ids) {
            (void)fer_tensor_release(session, adapter->tensors[tid].handle);
            adapter->tensors.erase(tid);
        }
        for (auto itj = adapter->jobs.begin(); itj != adapter->jobs.end();) {
            if (itj->second.session_id == session_id) {
                itj = adapter->jobs.erase(itj);
            } else {
                ++itj;
            }
        }
        adapter->sessions.erase(it);
    }
    return fer_session_destroy(session);
}

extern "C" fer_status_t fer_agent_tensor_create(
    fer_agent_adapter_t* adapter,
    const fer_agent_tensor_create_request_t* req,
    uint64_t* out_tensor_id) {
    if (!adapter) return invalid("adapter is null");
    if (!req) return invalid("req is null");
    if (!out_tensor_id) return invalid("out_tensor_id is null");

    fer_session_t* session = nullptr;
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        auto sit = adapter->sessions.find(req->session_id);
        if (sit == adapter->sessions.end()) return not_found("session_id not found");
        session = sit->second;
    }

    fer_tensor_spec_t spec{};
    spec.dtype = req->dtype;
    spec.rank = req->rank;
    for (uint32_t i = 0; i < 4; ++i) spec.dims[i] = req->dims[i];
    spec.immutable = req->immutable;
    spec.tag = req->tag;
    spec.memory_regime = req->memory_regime;

    fer_tensor_handle_t handle{};
    fer_status_t st = fer_tensor_create(session, &spec, &handle);
    if (st.code != FER_STATUS_OK) return st;

    const uint64_t tensor_id = adapter->next_tensor_id.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        adapter->tensors.emplace(tensor_id, TensorRecord{req->session_id, handle});
    }
    *out_tensor_id = tensor_id;
    return ok_status();
}

extern "C" fer_status_t fer_agent_tensor_copy(
    fer_agent_adapter_t* adapter,
    const fer_agent_tensor_copy_request_t* req) {
    if (!adapter) return invalid("adapter is null");
    if (!req) return invalid("req is null");
    if (!req->host_ptr) return invalid("host_ptr is null");
    const size_t elem = dtype_size_bytes(req->dtype);
    if (elem == 0) return invalid("unsupported dtype");
    const size_t bytes = req->count * elem;

    fer_session_t* session = nullptr;
    TensorRecord record{};
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        auto sit = adapter->sessions.find(req->session_id);
        if (sit == adapter->sessions.end()) return not_found("session_id not found");
        auto tit = adapter->tensors.find(req->tensor_id);
        if (tit == adapter->tensors.end()) return not_found("tensor_id not found");
        if (tit->second.session_id != req->session_id) return invalid("tensor/session mismatch");
        session = sit->second;
        record = tit->second;
    }

    emit_progress(adapter, FER_AGENT_PROGRESS_TENSOR_COPY_START,
                  req->session_id, req->tensor_id, 0.0f, "tensor copy started");

    fer_status_t st;
    if (req->direction == FER_AGENT_COPY_HOST_TO_DEVICE) {
        st = fer_tensor_upload(session, record.handle, req->host_ptr, bytes);
    } else if (req->direction == FER_AGENT_COPY_DEVICE_TO_HOST) {
        st = fer_tensor_download(session, record.handle, req->host_ptr, bytes);
    } else {
        return invalid("invalid copy direction");
    }

    emit_progress(adapter, FER_AGENT_PROGRESS_TENSOR_COPY_DONE,
                  req->session_id, req->tensor_id, 1.0f,
                  st.code == FER_STATUS_OK ? "tensor copy done" : "tensor copy failed");
    return st;
}

extern "C" fer_status_t fer_agent_tensor_release(
    fer_agent_adapter_t* adapter,
    uint64_t session_id,
    uint64_t tensor_id) {
    if (!adapter) return invalid("adapter is null");
    fer_session_t* session = nullptr;
    TensorRecord record{};
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        auto sit = adapter->sessions.find(session_id);
        if (sit == adapter->sessions.end()) return not_found("session_id not found");
        auto tit = adapter->tensors.find(tensor_id);
        if (tit == adapter->tensors.end()) return not_found("tensor_id not found");
        if (tit->second.session_id != session_id) return invalid("tensor/session mismatch");
        session = sit->second;
        record = tit->second;
    }
    fer_status_t st = fer_tensor_release(session, record.handle);
    if (st.code != FER_STATUS_OK) return st;
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        adapter->tensors.erase(tensor_id);
    }
    return ok_status();
}

extern "C" fer_status_t fer_agent_tensor_list(
    fer_agent_adapter_t* adapter,
    uint64_t session_id,
    uint64_t* out_tensor_ids,
    size_t capacity,
    size_t* out_count) {
    if (!adapter) return invalid("adapter is null");
    if (!out_count) return invalid("out_count is null");
    std::lock_guard<std::mutex> lock(adapter->mu);
    if (adapter->sessions.find(session_id) == adapter->sessions.end()) {
        return not_found("session_id not found");
    }
    std::vector<uint64_t> ids;
    ids.reserve(adapter->tensors.size());
    for (const auto& kv : adapter->tensors) {
        if (kv.second.session_id == session_id) ids.push_back(kv.first);
    }
    std::sort(ids.begin(), ids.end());
    *out_count = ids.size();
    if (out_tensor_ids && capacity > 0) {
        const size_t n = std::min(capacity, ids.size());
        for (size_t i = 0; i < n; ++i) out_tensor_ids[i] = ids[i];
    }
    return ok_status();
}

extern "C" fer_status_t fer_agent_jit_intent_run_affine_f32(
    fer_agent_adapter_t* adapter,
    const fer_agent_intent_affine_f32_request_t* req,
    uint64_t* out_job_id) {
    if (!adapter) return invalid("adapter is null");
    if (!req) return invalid("req is null");
    if (!out_job_id) return invalid("out_job_id is null");

    fer_session_t* session = nullptr;
    TensorRecord in_record{};
    TensorRecord out_record{};
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        auto sit = adapter->sessions.find(req->session_id);
        if (sit == adapter->sessions.end()) return not_found("session_id not found");
        auto in_it = adapter->tensors.find(req->input_tensor_id);
        auto out_it = adapter->tensors.find(req->output_tensor_id);
        if (in_it == adapter->tensors.end()) return not_found("input_tensor_id not found");
        if (out_it == adapter->tensors.end()) return not_found("output_tensor_id not found");
        if (in_it->second.session_id != req->session_id || out_it->second.session_id != req->session_id) {
            return invalid("tensor/session mismatch");
        }
        session = sit->second;
        in_record = in_it->second;
        out_record = out_it->second;
    }

    fer_jit_intent_t intent{};
    intent.abi_version = FER_JIT_INTENT_ABI_VERSION;
    intent.op = FER_JIT_INTENT_OP_AFFINE_F32;
    intent.fusion_mask = req->fusion_mask;
    intent.caps_mask = req->caps_mask;
    intent.memory_regime = req->memory_regime;
    intent.n = req->n;
    intent.alpha = req->alpha;
    intent.beta = req->beta;

    fer_jit_intent_bindings_t bindings{};
    bindings.input = in_record.handle.id;
    bindings.output = out_record.handle.id;

    uint64_t job_id = 0;
    fer_status_t st = fer_jit_run_intent(session, &intent, &bindings, &job_id);
    if (st.code != FER_STATUS_OK) return st;
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        adapter->jobs[job_id] = JobRecord{req->session_id};
    }
    *out_job_id = job_id;
    return ok_status();
}

extern "C" fer_status_t fer_agent_job_wait(
    fer_agent_adapter_t* adapter,
    uint64_t session_id,
    uint64_t job_id) {
    if (!adapter) return invalid("adapter is null");
    fer_session_t* session = nullptr;
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        auto sit = adapter->sessions.find(session_id);
        if (sit == adapter->sessions.end()) return not_found("session_id not found");
        auto jit = adapter->jobs.find(job_id);
        if (jit == adapter->jobs.end()) return not_found("job_id not found");
        if (jit->second.session_id != session_id) return invalid("job/session mismatch");
        session = sit->second;
    }
    fer_status_t st = fer_job_wait(session, job_id);
    if (st.code != FER_STATUS_OK) return st;
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        adapter->jobs.erase(job_id);
    }
    emit_progress(adapter, FER_AGENT_PROGRESS_JOB_COMPLETE,
                  session_id, job_id, 1.0f, "job complete");
    return ok_status();
}

extern "C" fer_status_t fer_agent_job_cancel(
    fer_agent_adapter_t* adapter,
    uint64_t session_id,
    uint64_t job_id,
    uint8_t* out_cancelled) {
    if (!adapter) return invalid("adapter is null");
    if (!out_cancelled) return invalid("out_cancelled is null");
    *out_cancelled = 0u;
    fer_session_t* session = nullptr;
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        auto sit = adapter->sessions.find(session_id);
        if (sit == adapter->sessions.end()) return not_found("session_id not found");
        auto jit = adapter->jobs.find(job_id);
        if (jit == adapter->jobs.end()) return not_found("job_id not found");
        if (jit->second.session_id != session_id) return invalid("job/session mismatch");
        session = sit->second;
    }

    uint8_t done = 0u;
    fer_status_t st = fer_job_status(session, job_id, &done);
    if (st.code != FER_STATUS_OK) return st;
    if (!done) {
        return invalid("job cancel is unsupported for running jobs");
    }
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        adapter->jobs.erase(job_id);
    }
    *out_cancelled = 1u;
    return ok_status();
}

extern "C" fer_status_t fer_agent_session_list(
    fer_agent_adapter_t* adapter,
    uint64_t* out_session_ids,
    size_t capacity,
    size_t* out_count) {
    if (!adapter) return invalid("adapter is null");
    if (!out_count) return invalid("out_count is null");
    std::lock_guard<std::mutex> lock(adapter->mu);
    std::vector<uint64_t> ids;
    ids.reserve(adapter->sessions.size());
    for (const auto& kv : adapter->sessions) ids.push_back(kv.first);
    std::sort(ids.begin(), ids.end());
    *out_count = ids.size();
    if (out_session_ids && capacity > 0) {
        const size_t n = std::min(capacity, ids.size());
        for (size_t i = 0; i < n; ++i) out_session_ids[i] = ids[i];
    }
    return ok_status();
}

extern "C" fer_status_t fer_agent_job_list(
    fer_agent_adapter_t* adapter,
    uint64_t session_id,
    uint64_t* out_job_ids,
    size_t capacity,
    size_t* out_count) {
    if (!adapter) return invalid("adapter is null");
    if (!out_count) return invalid("out_count is null");
    std::lock_guard<std::mutex> lock(adapter->mu);
    if (adapter->sessions.find(session_id) == adapter->sessions.end()) {
        return not_found("session_id not found");
    }
    std::vector<uint64_t> ids;
    ids.reserve(adapter->jobs.size());
    for (const auto& kv : adapter->jobs) {
        if (kv.second.session_id == session_id) ids.push_back(kv.first);
    }
    std::sort(ids.begin(), ids.end());
    *out_count = ids.size();
    if (out_job_ids && capacity > 0) {
        const size_t n = std::min(capacity, ids.size());
        for (size_t i = 0; i < n; ++i) out_job_ids[i] = ids[i];
    }
    return ok_status();
}

extern "C" fer_status_t fer_agent_session_stats(
    fer_agent_adapter_t* adapter,
    uint64_t session_id,
    size_t* out_tensor_count,
    size_t* out_job_count) {
    if (!adapter) return invalid("adapter is null");
    if (!out_tensor_count || !out_job_count) return invalid("stats out pointers are null");
    std::lock_guard<std::mutex> lock(adapter->mu);
    if (adapter->sessions.find(session_id) == adapter->sessions.end()) {
        return not_found("session_id not found");
    }
    size_t tensor_count = 0;
    for (const auto& kv : adapter->tensors) {
        if (kv.second.session_id == session_id) ++tensor_count;
    }
    size_t job_count = 0;
    for (const auto& kv : adapter->jobs) {
        if (kv.second.session_id == session_id) ++job_count;
    }
    *out_tensor_count = tensor_count;
    *out_job_count = job_count;
    return ok_status();
}

// --- Generic intent runner ---------------------------------------------------

extern "C" fer_status_t fer_agent_jit_intent_run(
    fer_agent_adapter_t* adapter,
    const fer_agent_intent_generic_request_t* req,
    uint64_t* out_job_id) {
    if (!adapter) return invalid("adapter is null");
    if (!req) return invalid("req is null");
    if (!out_job_id) return invalid("out_job_id is null");

    fer_session_t* session = nullptr;
    TensorRecord in_rec{}, out_rec{};
    TensorRecord weights_rec{}, bias_rec{};
    bool has_weights = (req->weights_tensor_id != 0);
    bool has_bias = (req->bias_tensor_id != 0);
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        auto sit = adapter->sessions.find(req->session_id);
        if (sit == adapter->sessions.end()) return not_found("session_id not found");
        session = sit->second;
        auto in_it = adapter->tensors.find(req->input_tensor_id);
        auto out_it = adapter->tensors.find(req->output_tensor_id);
        if (in_it == adapter->tensors.end()) return not_found("input_tensor_id not found");
        if (out_it == adapter->tensors.end()) return not_found("output_tensor_id not found");
        if (in_it->second.session_id != req->session_id || out_it->second.session_id != req->session_id)
            return invalid("tensor/session mismatch");
        in_rec = in_it->second;
        out_rec = out_it->second;
        if (has_weights) {
            auto wit = adapter->tensors.find(req->weights_tensor_id);
            if (wit == adapter->tensors.end()) return not_found("weights_tensor_id not found");
            weights_rec = wit->second;
        }
        if (has_bias) {
            auto bit = adapter->tensors.find(req->bias_tensor_id);
            if (bit == adapter->tensors.end()) return not_found("bias_tensor_id not found");
            bias_rec = bit->second;
        }
    }

    fer_jit_intent_t intent{};
    intent.abi_version = FER_JIT_INTENT_ABI_VERSION;
    intent.op = req->op;
    intent.fusion_mask = req->fusion_mask;
    intent.caps_mask = req->caps_mask;
    intent.memory_regime = req->memory_regime;
    intent.n = req->n;
    intent.alpha = req->alpha;
    intent.beta = req->beta;
    intent.height = req->height;
    intent.width = req->width;
    intent.channels = req->channels;
    intent.kernel_h = req->kernel_h;
    intent.kernel_w = req->kernel_w;
    intent.pad_h = req->pad_h;
    intent.pad_w = req->pad_w;
    intent.stride_h = req->stride_h;
    intent.stride_w = req->stride_w;
    intent.num_filters = req->num_filters;

    fer_jit_intent_bindings_t bindings{};
    bindings.input = in_rec.handle.id;
    bindings.output = out_rec.handle.id;
    bindings.weights = has_weights ? weights_rec.handle.id : 0;
    bindings.bias = has_bias ? bias_rec.handle.id : 0;

    uint64_t job_id = 0;
    fer_status_t st = fer_jit_run_intent(session, &intent, &bindings, &job_id);
    if (st.code != FER_STATUS_OK) return st;
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        adapter->jobs[job_id] = JobRecord{req->session_id};
    }
    *out_job_id = job_id;
    return ok_status();
}

// --- Progress callback ------------------------------------------------------

extern "C" fer_status_t fer_agent_set_progress_callback(
    fer_agent_adapter_t* adapter,
    fer_agent_progress_callback_fn callback,
    void* user_ctx) {
    if (!adapter) return invalid("adapter is null");
    std::lock_guard<std::mutex> lock(adapter->mu);
    adapter->progress_cb = callback;
    adapter->progress_ctx = user_ctx;
    return ok_status();
}

// --- JIT kernel compile/launch ----------------------------------------------

extern "C" fer_status_t fer_agent_jit_compile(
    fer_agent_adapter_t* adapter,
    const fer_agent_jit_compile_request_t* req,
    fer_agent_jit_compile_result_t* out_result) {
    if (!adapter) return invalid("adapter is null");
    if (!req) return invalid("req is null");
    if (!out_result) return invalid("out_result is null");
    if (!req->source || req->source_len == 0) return invalid("source is empty");

    fer_session_t* session = nullptr;
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        auto sit = adapter->sessions.find(req->session_id);
        if (sit == adapter->sessions.end()) return not_found("session_id not found");
        session = sit->second;
    }

    emit_progress(adapter, FER_AGENT_PROGRESS_JIT_COMPILE_START,
                  req->session_id, 0, 0.0f, "compile started");

    fer_jit_source_t source{};
    source.kind = req->language;
    source.code = req->source;
    source.code_len = req->source_len;

    fer_jit_options_t options{};
    options.backend = FER_JIT_BACKEND_NVRTC;
    options.mode = req->strict ? FER_JIT_MODE_STRICT : FER_JIT_MODE_PERMISSIVE;
    options.enable_disk_cache = 1;
    options.cache_dir = "/tmp/fercuda_jit_cache_agent";

    fer_jit_program_t program = nullptr;
    fer_jit_compile_result_t cr{};
    fer_status_t st = fer_jit_compile(session, &source, &options, &program, &cr);
    if (st.code != FER_STATUS_OK || !program) {
        out_result->diagnostics_log = cr.log;
        emit_progress(adapter, FER_AGENT_PROGRESS_JIT_COMPILE_DONE,
                      req->session_id, 0, 1.0f, "compile failed");
        return st.code == FER_STATUS_OK
            ? fer_status_t{FER_STATUS_INTERNAL_ERROR, "compile returned null program"}
            : st;
    }

    const uint64_t program_id = adapter->next_program_id.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        adapter->programs[program_id] = ProgramRecord{req->session_id, program};
    }
    out_result->program_id = program_id;
    out_result->cache_hit = cr.cache_hit;
    out_result->diagnostics_log = cr.log;

    emit_progress(adapter, FER_AGENT_PROGRESS_JIT_COMPILE_DONE,
                  req->session_id, program_id, 1.0f,
                  cr.cache_hit ? "compile done (cache hit)" : "compile done");
    return ok_status();
}

extern "C" fer_status_t fer_agent_jit_launch(
    fer_agent_adapter_t* adapter,
    const fer_agent_jit_launch_request_t* req,
    uint64_t* out_job_id) {
    if (!adapter) return invalid("adapter is null");
    if (!req) return invalid("req is null");
    if (!out_job_id) return invalid("out_job_id is null");
    if (!req->kernel_name) return invalid("kernel_name is null");

    fer_session_t* session = nullptr;
    fer_jit_program_t program = nullptr;
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        auto sit = adapter->sessions.find(req->session_id);
        if (sit == adapter->sessions.end()) return not_found("session_id not found");
        session = sit->second;
        auto pit = adapter->programs.find(req->program_id);
        if (pit == adapter->programs.end()) return not_found("program_id not found");
        if (pit->second.session_id != req->session_id) return invalid("program/session mismatch");
        program = pit->second.handle;
    }

    emit_progress(adapter, FER_AGENT_PROGRESS_JIT_LAUNCH_START,
                  req->session_id, req->program_id, 0.0f, "launch started");

    // Build arg descriptors and values from the launch args
    std::vector<fer_jit_arg_desc_t> descs(req->arg_count);
    std::vector<fer_jit_arg_value_t> vals(req->arg_count);

    for (size_t i = 0; i < req->arg_count; ++i) {
        const fer_agent_jit_launch_arg_t& la = req->args[i];
        descs[i].kind = la.kind;
        descs[i].expected_dtype = FER_JIT_WILDCARD_U32;
        descs[i].expected_rank = FER_JIT_WILDCARD_U32;
        descs[i].expected_bytes = FER_JIT_WILDCARD_U64;
        for (int d = 0; d < 4; ++d) descs[i].expected_dims[d] = FER_JIT_WILDCARD_U32;

        vals[i].kind = la.kind;
        if (la.kind == FER_JIT_ARG_BUFFER) {
            descs[i].access = la.access;
            TensorRecord rec{};
            {
                std::lock_guard<std::mutex> lock(adapter->mu);
                auto tit = adapter->tensors.find(la.tensor_id);
                if (tit == adapter->tensors.end()) return not_found("tensor_id not found in launch args");
                if (tit->second.session_id != req->session_id) return invalid("tensor/session mismatch in launch args");
                rec = tit->second;
            }
            vals[i].as.buffer_id = rec.handle.id;
        } else {
            switch (la.kind) {
                case FER_JIT_ARG_SCALAR_F32: vals[i].as.f32 = la.scalar.f32; break;
                case FER_JIT_ARG_SCALAR_F64: vals[i].as.f64 = la.scalar.f64; break;
                case FER_JIT_ARG_SCALAR_I32: vals[i].as.i32 = la.scalar.i32; break;
                case FER_JIT_ARG_SCALAR_U32: vals[i].as.u32 = la.scalar.u32; break;
                case FER_JIT_ARG_SCALAR_I64: vals[i].as.i64 = la.scalar.i64; break;
                case FER_JIT_ARG_SCALAR_U64: vals[i].as.u64 = la.scalar.u64; break;
                default: return invalid("unsupported arg kind in launch args");
            }
        }
    }

    fer_jit_kernel_sig_t sig{};
    sig.args = descs.empty() ? nullptr : descs.data();
    sig.arg_count = descs.size();

    fer_jit_kernel_t kernel = nullptr;
    fer_status_t st = fer_jit_get_kernel(session, program, req->kernel_name, &sig, &kernel);
    if (st.code != FER_STATUS_OK || !kernel) {
        emit_progress(adapter, FER_AGENT_PROGRESS_JIT_LAUNCH_DONE,
                      req->session_id, req->program_id, 1.0f, "launch failed: kernel not found");
        return st.code == FER_STATUS_OK
            ? fer_status_t{FER_STATUS_INTERNAL_ERROR, "get_kernel returned null"}
            : st;
    }

    fer_jit_arg_pack_t pack{};
    pack.args = vals.empty() ? nullptr : vals.data();
    pack.arg_count = vals.size();

    fer_jit_launch_cfg_t cfg{};
    cfg.grid_x = req->grid[0]; cfg.grid_y = req->grid[1]; cfg.grid_z = req->grid[2];
    cfg.block_x = req->block[0]; cfg.block_y = req->block[1]; cfg.block_z = req->block[2];
    cfg.shared_mem_bytes = req->shared_mem_bytes;
    cfg.memory_regime = 0xFFFFFFFFu;  // allow any regime

    uint64_t job_id = 0;
    st = fer_jit_launch(session, kernel, &cfg, &pack, &job_id);
    fer_jit_release_kernel(session, kernel);

    if (st.code != FER_STATUS_OK) {
        emit_progress(adapter, FER_AGENT_PROGRESS_JIT_LAUNCH_DONE,
                      req->session_id, req->program_id, 1.0f, "launch failed");
        return st;
    }

    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        adapter->jobs[job_id] = JobRecord{req->session_id};
    }
    *out_job_id = job_id;

    emit_progress(adapter, FER_AGENT_PROGRESS_JIT_LAUNCH_DONE,
                  req->session_id, job_id, 1.0f, "launch submitted");
    return ok_status();
}

extern "C" fer_status_t fer_agent_jit_release_program(
    fer_agent_adapter_t* adapter,
    uint64_t session_id,
    uint64_t program_id) {
    if (!adapter) return invalid("adapter is null");
    fer_session_t* session = nullptr;
    fer_jit_program_t program = nullptr;
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        auto sit = adapter->sessions.find(session_id);
        if (sit == adapter->sessions.end()) return not_found("session_id not found");
        session = sit->second;
        auto pit = adapter->programs.find(program_id);
        if (pit == adapter->programs.end()) return not_found("program_id not found");
        if (pit->second.session_id != session_id) return invalid("program/session mismatch");
        program = pit->second.handle;
        adapter->programs.erase(pit);
    }
    return fer_jit_release_program(session, program);
}

// --- Blob store -------------------------------------------------------------

extern "C" fer_status_t fer_agent_blob_put(
    fer_agent_adapter_t* adapter,
    const char* blob_ref_or_null,
    const void* data,
    size_t bytes,
    char* out_blob_ref,
    size_t out_blob_ref_len) {
    if (!adapter) return invalid("adapter is null");
    if (!data && bytes != 0) return invalid("data is null");
    if (!out_blob_ref || out_blob_ref_len == 0) return invalid("out_blob_ref is null");

    std::string ref;
    if (blob_ref_or_null && blob_ref_or_null[0] != '\0') {
        ref = blob_ref_or_null;
    } else {
        const uint64_t id = adapter->next_blob_id.fetch_add(1);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "blob_%llu", static_cast<unsigned long long>(id));
        ref = buf;
    }
    std::vector<uint8_t> v(bytes);
    if (bytes) std::memcpy(v.data(), data, bytes);
    {
        std::lock_guard<std::mutex> lock(adapter->mu);
        adapter->blobs[ref] = std::move(v);
    }
    if (ref.size() + 1 > out_blob_ref_len) return invalid("out_blob_ref buffer too small");
    std::memcpy(out_blob_ref, ref.c_str(), ref.size() + 1);
    return ok_status();
}

extern "C" fer_status_t fer_agent_blob_get(
    fer_agent_adapter_t* adapter,
    const char* blob_ref,
    const void** out_data,
    size_t* out_bytes) {
    if (!adapter) return invalid("adapter is null");
    if (!blob_ref || blob_ref[0] == '\0') return invalid("blob_ref is empty");
    if (!out_data) return invalid("out_data is null");
    if (!out_bytes) return invalid("out_bytes is null");
    std::lock_guard<std::mutex> lock(adapter->mu);
    auto it = adapter->blobs.find(blob_ref);
    if (it == adapter->blobs.end()) return not_found("blob_ref not found");
    *out_data = it->second.empty() ? nullptr : it->second.data();
    *out_bytes = it->second.size();
    return ok_status();
}

extern "C" fer_status_t fer_agent_blob_resize_for_write(
    fer_agent_adapter_t* adapter,
    const char* blob_ref,
    size_t bytes,
    void** out_data) {
    if (!adapter) return invalid("adapter is null");
    if (!blob_ref || blob_ref[0] == '\0') return invalid("blob_ref is empty");
    if (!out_data) return invalid("out_data is null");
    std::lock_guard<std::mutex> lock(adapter->mu);
    auto& blob = adapter->blobs[blob_ref];
    blob.resize(bytes);
    *out_data = blob.empty() ? nullptr : blob.data();
    return ok_status();
}
