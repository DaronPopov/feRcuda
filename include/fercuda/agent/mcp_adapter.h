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
