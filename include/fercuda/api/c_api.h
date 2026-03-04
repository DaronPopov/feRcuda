#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fer_session fer_session_t;

typedef enum fer_status_code {
    FER_STATUS_OK = 0,
    FER_STATUS_INVALID_ARGUMENT = 1,
    FER_STATUS_NOT_FOUND = 2,
    FER_STATUS_INTERNAL_ERROR = 3
} fer_status_code_t;

typedef struct fer_status {
    int32_t code;
    const char* message;
} fer_status_t;

typedef struct fer_pool_config {
    uint64_t mutable_bytes;
    uint64_t immutable_bytes;
    uint64_t cuda_reserve;
    uint8_t verbose;
    uint32_t memory_regime;
} fer_pool_config_t;

typedef enum fer_memory_regime {
    FER_MEMORY_CUSTOM_POOL = 0,
    FER_MEMORY_CUDA_MALLOC = 1,
    FER_MEMORY_CUDA_MANAGED = 2,
    FER_MEMORY_AUTO = 0xFFFFFFFFu
} fer_memory_regime_t;

typedef enum fer_alloc_tier {
    FER_ALLOC_TIER_MUTABLE = 0,
    FER_ALLOC_TIER_IMMUTABLE = 1
} fer_alloc_tier_t;

typedef void* (*fer_allocator_alloc_fn)(uint64_t bytes, uint32_t tier, uint32_t tag, void* user_ctx);
typedef void (*fer_allocator_free_fn)(void* ptr, void* user_ctx);

typedef struct fer_custom_allocator {
    fer_allocator_alloc_fn alloc;
    fer_allocator_free_fn free;
    void* user_ctx;
} fer_custom_allocator_t;

typedef void* (*fer_gpu_hot_alloc_fn)(void* runtime, uint64_t bytes);
typedef void (*fer_gpu_hot_free_fn)(void* runtime, void* ptr);

typedef struct fer_gpu_hot_allocator {
    void* runtime;
    fer_gpu_hot_alloc_fn alloc;
    fer_gpu_hot_free_fn free;
} fer_gpu_hot_allocator_t;

typedef enum fer_buffer_dtype {
    FER_DTYPE_F32 = 0,
    FER_DTYPE_F16 = 1,
    FER_DTYPE_BF16 = 2,
    FER_DTYPE_I8 = 3,
    FER_DTYPE_U8 = 4,
    FER_DTYPE_I16 = 5,
    FER_DTYPE_U16 = 6,
    FER_DTYPE_I32 = 7,
    FER_DTYPE_U32 = 8,
    FER_DTYPE_I64 = 9,
    FER_DTYPE_U64 = 10,
    FER_DTYPE_F64 = 11
} fer_buffer_dtype_t;

typedef struct fer_buffer_desc {
    uint32_t dtype;
    uint32_t rank;
    uint32_t dims[4];
    uint8_t immutable;
    uint32_t tag;
} fer_buffer_desc_t;

typedef struct fer_matmul_request {
    uint64_t a;
    uint64_t b;
    uint64_t out;
    uint32_t memory_regime;
} fer_matmul_request_t;

typedef struct fer_layer_norm_request {
    uint64_t x;
    uint64_t out;
    float eps;
    uint32_t memory_regime;
} fer_layer_norm_request_t;

fer_status_t fer_session_create(int32_t device, const fer_pool_config_t* cfg, fer_session_t** out_session);
fer_status_t fer_session_destroy(fer_session_t* session);

fer_status_t fer_register_custom_allocator(fer_session_t* session, uint32_t regime_id, const fer_custom_allocator_t* allocator);
fer_status_t fer_register_gpu_hot_allocator(fer_session_t* session, uint32_t regime_id, const fer_gpu_hot_allocator_t* allocator);
fer_status_t fer_unregister_custom_allocator(fer_session_t* session, uint32_t regime_id);
fer_status_t fer_set_default_memory_regime(fer_session_t* session, uint32_t regime_id);
fer_status_t fer_get_default_memory_regime(fer_session_t* session, uint32_t* out_regime_id);
fer_status_t fer_list_registered_regimes(fer_session_t* session, uint32_t* out_regime_ids, size_t capacity, size_t* out_count);

fer_status_t fer_alloc_buffer(fer_session_t* session, const fer_buffer_desc_t* desc, uint64_t* out_buffer_id);
fer_status_t fer_alloc_buffer_in_regime(fer_session_t* session, const fer_buffer_desc_t* desc, uint32_t regime_id, uint64_t* out_buffer_id);
fer_status_t fer_free_buffer(fer_session_t* session, uint64_t buffer_id);

fer_status_t fer_upload_f32(fer_session_t* session, uint64_t buffer_id, const float* host, size_t count);
fer_status_t fer_download_f32(fer_session_t* session, uint64_t buffer_id, float* host, size_t count);
fer_status_t fer_upload_bytes(fer_session_t* session, uint64_t buffer_id, const void* host, size_t bytes);
fer_status_t fer_download_bytes(fer_session_t* session, uint64_t buffer_id, void* host, size_t bytes);

fer_status_t fer_submit_matmul(fer_session_t* session, const fer_matmul_request_t* req, uint64_t* out_job_id);
fer_status_t fer_submit_layer_norm(fer_session_t* session, const fer_layer_norm_request_t* req, uint64_t* out_job_id);
fer_status_t fer_job_status(fer_session_t* session, uint64_t job_id, uint8_t* out_done);
fer_status_t fer_job_wait(fer_session_t* session, uint64_t job_id);

#ifdef __cplusplus
}
#endif
