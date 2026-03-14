#pragma once

#include <stddef.h>
#include <stdint.h>

#include "fercuda/api/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FER_AGENT_API_VERSION_V1ALPHA1 1u

typedef struct fer_agent_adapter fer_agent_adapter_t;

typedef enum fer_agent_copy_direction {
    FER_AGENT_COPY_HOST_TO_DEVICE = 0,
    FER_AGENT_COPY_DEVICE_TO_HOST = 1
} fer_agent_copy_direction_t;

typedef struct fer_agent_session_create_request {
    int32_t device;
    uint64_t mutable_bytes;
    uint64_t immutable_bytes;
    uint64_t cuda_reserve;
    uint8_t verbose;
    uint32_t memory_regime;
} fer_agent_session_create_request_t;

typedef struct fer_agent_tensor_create_request {
    uint64_t session_id;
    uint32_t dtype;
    uint32_t rank;
    uint32_t dims[4];
    uint8_t immutable;
    uint32_t tag;
    uint32_t memory_regime;
} fer_agent_tensor_create_request_t;

typedef struct fer_agent_tensor_copy_request {
    uint64_t session_id;
    uint64_t tensor_id;
    uint32_t dtype;
    size_t count;
    void* host_ptr;
    uint32_t direction;
} fer_agent_tensor_copy_request_t;

typedef struct fer_agent_intent_affine_f32_request {
    uint64_t session_id;
    uint64_t input_tensor_id;
    uint64_t output_tensor_id;
    uint32_t n;
    float alpha;
    float beta;
    uint32_t fusion_mask;
    uint32_t caps_mask;
    uint32_t memory_regime;
} fer_agent_intent_affine_f32_request_t;

typedef struct fer_agent_runtime_inspect {
    uint32_t agent_api_version;
    uint32_t jit_intent_abi_version;
    uint8_t supports_jit_compile;
    uint8_t supports_jit_intent;
    uint8_t supports_external_ptr_import;
    uint8_t supports_session_stream_handoff;
} fer_agent_runtime_inspect_t;

fer_status_t fer_agent_adapter_create(fer_agent_adapter_t** out_adapter);
fer_status_t fer_agent_adapter_destroy(fer_agent_adapter_t* adapter);

fer_status_t fer_agent_runtime_inspect(fer_agent_adapter_t* adapter, fer_agent_runtime_inspect_t* out_info);

fer_status_t fer_agent_session_create(
    fer_agent_adapter_t* adapter,
    const fer_agent_session_create_request_t* req,
    uint64_t* out_session_id);
fer_status_t fer_agent_session_destroy(fer_agent_adapter_t* adapter, uint64_t session_id);

fer_status_t fer_agent_tensor_create(
    fer_agent_adapter_t* adapter,
    const fer_agent_tensor_create_request_t* req,
    uint64_t* out_tensor_id);
fer_status_t fer_agent_tensor_copy(
    fer_agent_adapter_t* adapter,
    const fer_agent_tensor_copy_request_t* req);
fer_status_t fer_agent_tensor_release(
    fer_agent_adapter_t* adapter,
    uint64_t session_id,
    uint64_t tensor_id);
fer_status_t fer_agent_tensor_list(
    fer_agent_adapter_t* adapter,
    uint64_t session_id,
    uint64_t* out_tensor_ids,
    size_t capacity,
    size_t* out_count);

fer_status_t fer_agent_jit_intent_run_affine_f32(
    fer_agent_adapter_t* adapter,
    const fer_agent_intent_affine_f32_request_t* req,
    uint64_t* out_job_id);

typedef struct fer_agent_intent_generic_request {
    uint64_t session_id;
    uint32_t op;
    uint64_t input_tensor_id;
    uint64_t output_tensor_id;
    uint64_t weights_tensor_id;
    uint64_t bias_tensor_id;
    uint32_t n;
    float alpha;
    float beta;
    uint32_t fusion_mask;
    uint32_t caps_mask;
    uint32_t memory_regime;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t num_filters;
} fer_agent_intent_generic_request_t;

fer_status_t fer_agent_jit_intent_run(
    fer_agent_adapter_t* adapter,
    const fer_agent_intent_generic_request_t* req,
    uint64_t* out_job_id);

// --- JIT kernel compile/launch (full compile path) --------------------------

typedef struct fer_agent_jit_compile_request {
    uint64_t session_id;
    const char* source;
    size_t source_len;
    uint32_t language;        // fer_jit_source_kind_t
    uint32_t optimization_level;
    uint8_t strict;
} fer_agent_jit_compile_request_t;

typedef struct fer_agent_jit_compile_result {
    uint64_t program_id;
    uint8_t cache_hit;
    const char* diagnostics_log;
} fer_agent_jit_compile_result_t;

typedef struct fer_agent_jit_launch_arg {
    uint32_t kind;            // "tensor", "f32", "u32", "i32", "u64", "f64"
    uint64_t tensor_id;       // valid when kind == FER_JIT_ARG_BUFFER
    uint32_t access;          // fer_jit_access_t, valid for tensor args
    union {
        float f32;
        double f64;
        int32_t i32;
        uint32_t u32;
        int64_t i64;
        uint64_t u64;
    } scalar;
} fer_agent_jit_launch_arg_t;

typedef struct fer_agent_jit_launch_request {
    uint64_t session_id;
    uint64_t program_id;
    const char* kernel_name;
    uint32_t grid[3];
    uint32_t block[3];
    uint32_t shared_mem_bytes;
    const fer_agent_jit_launch_arg_t* args;
    size_t arg_count;
} fer_agent_jit_launch_request_t;

fer_status_t fer_agent_jit_compile(
    fer_agent_adapter_t* adapter,
    const fer_agent_jit_compile_request_t* req,
    fer_agent_jit_compile_result_t* out_result);

fer_status_t fer_agent_jit_launch(
    fer_agent_adapter_t* adapter,
    const fer_agent_jit_launch_request_t* req,
    uint64_t* out_job_id);

fer_status_t fer_agent_jit_release_program(
    fer_agent_adapter_t* adapter,
    uint64_t session_id,
    uint64_t program_id);

// --- Progress channel -------------------------------------------------------

typedef enum fer_agent_progress_kind {
    FER_AGENT_PROGRESS_JIT_COMPILE_START   = 0,
    FER_AGENT_PROGRESS_JIT_COMPILE_DONE    = 1,
    FER_AGENT_PROGRESS_JIT_LAUNCH_START    = 2,
    FER_AGENT_PROGRESS_JIT_LAUNCH_DONE     = 3,
    FER_AGENT_PROGRESS_TENSOR_COPY_START   = 4,
    FER_AGENT_PROGRESS_TENSOR_COPY_DONE    = 5,
    FER_AGENT_PROGRESS_JOB_COMPLETE        = 6
} fer_agent_progress_kind_t;

typedef struct fer_agent_progress_event {
    uint32_t kind;
    uint64_t session_id;
    uint64_t entity_id;       // job_id, program_id, or tensor_id depending on kind
    uint64_t timestamp_us;
    float fraction;           // 0.0 .. 1.0 for partial progress, 1.0 on completion
    const char* message;
} fer_agent_progress_event_t;

typedef void (*fer_agent_progress_callback_fn)(
    const fer_agent_progress_event_t* event,
    void* user_ctx);

fer_status_t fer_agent_set_progress_callback(
    fer_agent_adapter_t* adapter,
    fer_agent_progress_callback_fn callback,
    void* user_ctx);

fer_status_t fer_agent_job_wait(
    fer_agent_adapter_t* adapter,
    uint64_t session_id,
    uint64_t job_id);
fer_status_t fer_agent_job_cancel(
    fer_agent_adapter_t* adapter,
    uint64_t session_id,
    uint64_t job_id,
    uint8_t* out_cancelled);
fer_status_t fer_agent_session_list(
    fer_agent_adapter_t* adapter,
    uint64_t* out_session_ids,
    size_t capacity,
    size_t* out_count);
fer_status_t fer_agent_job_list(
    fer_agent_adapter_t* adapter,
    uint64_t session_id,
    uint64_t* out_job_ids,
    size_t capacity,
    size_t* out_count);
fer_status_t fer_agent_session_stats(
    fer_agent_adapter_t* adapter,
    uint64_t session_id,
    size_t* out_tensor_count,
    size_t* out_job_count);

// Blob store APIs for out-of-process transport adapters.
// Data is copied into adapter-owned memory.
fer_status_t fer_agent_blob_put(
    fer_agent_adapter_t* adapter,
    const char* blob_ref_or_null,
    const void* data,
    size_t bytes,
    char* out_blob_ref,
    size_t out_blob_ref_len);

// Returns adapter-owned pointer valid until next write/destroy.
fer_status_t fer_agent_blob_get(
    fer_agent_adapter_t* adapter,
    const char* blob_ref,
    const void** out_data,
    size_t* out_bytes);

// Resizes/creates a mutable blob and returns writable pointer.
fer_status_t fer_agent_blob_resize_for_write(
    fer_agent_adapter_t* adapter,
    const char* blob_ref,
    size_t bytes,
    void** out_data);

#ifdef __cplusplus
}
#endif
