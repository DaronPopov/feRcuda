#pragma once

#include "fercuda/api/c_api.h"
#include "fercuda/jit/api.h"
#include "fercuda/jit/intent/intent.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fer_script_context fer_script_context_t;

typedef uint32_t (*fer_script_abi_version_fn)(void);

typedef fer_status_t (*fer_script_main_fn)(
    const fer_script_context_t* ctx,
    const char* args_json,
    char* err_buf,
    size_t err_buf_len);

typedef struct fer_script_host_api {
    uint32_t abi_version;

    fer_status_t (*session_create)(
        int32_t device,
        const fer_pool_config_t* cfg,
        fer_session_t** out_session);
    fer_status_t (*session_destroy)(fer_session_t* session);

    fer_status_t (*alloc_buffer)(
        fer_session_t* session,
        const fer_buffer_desc_t* desc,
        uint64_t* out_buffer_id);
    fer_status_t (*alloc_buffer_in_regime)(
        fer_session_t* session,
        const fer_buffer_desc_t* desc,
        uint32_t regime_id,
        uint64_t* out_buffer_id);
    fer_status_t (*import_external_buffer)(
        fer_session_t* session,
        const fer_buffer_desc_t* desc,
        void* device_ptr,
        uint32_t regime_id,
        uint64_t* out_buffer_id);
    fer_status_t (*import_external_buffer_with_deleter)(
        fer_session_t* session,
        const fer_buffer_desc_t* desc,
        void* device_ptr,
        uint32_t regime_id,
        fer_external_buffer_deleter_fn deleter,
        void* deleter_user_ctx,
        uint64_t* out_buffer_id);
    fer_status_t (*export_buffer_device_ptr)(
        fer_session_t* session,
        uint64_t buffer_id,
        void** out_device_ptr);
    fer_status_t (*free_buffer)(
        fer_session_t* session,
        uint64_t buffer_id);
    fer_status_t (*upload_bytes)(
        fer_session_t* session,
        uint64_t buffer_id,
        const void* host,
        size_t bytes);
    fer_status_t (*download_bytes)(
        fer_session_t* session,
        uint64_t buffer_id,
        void* host,
        size_t bytes);

    fer_status_t (*jit_compile)(
        fer_session_t* session,
        const fer_jit_source_t* source,
        const fer_jit_options_t* options,
        fer_jit_program_t* out_program,
        fer_jit_compile_result_t* out_result);
    fer_status_t (*jit_release_program)(
        fer_session_t* session,
        fer_jit_program_t program);
    fer_status_t (*jit_get_kernel)(
        fer_session_t* session,
        fer_jit_program_t program,
        const char* kernel_name,
        const fer_jit_kernel_sig_t* signature,
        fer_jit_kernel_t* out_kernel);
    fer_status_t (*jit_release_kernel)(
        fer_session_t* session,
        fer_jit_kernel_t kernel);
    fer_status_t (*jit_launch)(
        fer_session_t* session,
        fer_jit_kernel_t kernel,
        const fer_jit_launch_cfg_t* cfg,
        const fer_jit_arg_pack_t* args,
        uint64_t* out_job_id);
    fer_status_t (*jit_run_intent)(
        fer_session_t* session,
        const fer_jit_intent_t* intent,
        const fer_jit_intent_bindings_t* bindings,
        uint64_t* out_job_id);
    fer_status_t (*job_wait)(
        fer_session_t* session,
        uint64_t job_id);
    fer_status_t (*set_session_stream)(
        fer_session_t* session,
        uint64_t stream_handle);
    fer_status_t (*get_session_stream)(
        fer_session_t* session,
        uint64_t* out_stream_handle);
    fer_status_t (*tensor_create)(
        fer_session_t* session,
        const fer_tensor_spec_t* spec,
        fer_tensor_handle_t* out_tensor);
    fer_status_t (*tensor_attach_external)(
        fer_session_t* session,
        const fer_tensor_spec_t* spec,
        void* device_ptr,
        fer_external_buffer_deleter_fn deleter,
        void* deleter_user_ctx,
        fer_tensor_handle_t* out_tensor);
    fer_status_t (*tensor_device_ptr)(
        fer_session_t* session,
        fer_tensor_handle_t tensor,
        void** out_device_ptr);
    fer_status_t (*tensor_release)(
        fer_session_t* session,
        fer_tensor_handle_t tensor);
    fer_status_t (*tensor_upload)(
        fer_session_t* session,
        fer_tensor_handle_t tensor,
        const void* host,
        size_t bytes);
    fer_status_t (*tensor_download)(
        fer_session_t* session,
        fer_tensor_handle_t tensor,
        void* host,
        size_t bytes);
    fer_status_t (*tensor_run_affine_f32)(
        fer_session_t* session,
        fer_tensor_handle_t input,
        fer_tensor_handle_t output,
        uint32_t n,
        float alpha,
        float beta,
        uint32_t fusion_mask,
        uint32_t caps_mask,
        uint32_t memory_regime,
        uint64_t* out_job_id);
    uint8_t (*script_should_cancel)(
        const fer_script_context_t* ctx);
    uint64_t (*time_now_ms)(void);
} fer_script_host_api_t;

struct fer_script_context {
    uint32_t abi_version;
    fer_session_t* session;
    const fer_script_host_api_t* api;
    void* user_data;
};

#define FER_SCRIPT_HOST_ABI_VERSION 1u

static inline fer_status_t fer_script_tensor_create_upload(
    const fer_script_context_t* ctx,
    const fer_tensor_spec_t* spec,
    const void* host,
    size_t bytes,
    fer_tensor_handle_t* out_tensor) {
    if (!ctx || !ctx->api || !ctx->session || !ctx->api->tensor_create || !ctx->api->tensor_upload) {
        return fer_status_t{FER_STATUS_INVALID_ARGUMENT, "script ctx/api is invalid"};
    }
    fer_status_t st = ctx->api->tensor_create(ctx->session, spec, out_tensor);
    if (st.code != FER_STATUS_OK) return st;
    st = ctx->api->tensor_upload(ctx->session, *out_tensor, host, bytes);
    if (st.code != FER_STATUS_OK) {
        if (ctx->api->tensor_release) (void)ctx->api->tensor_release(ctx->session, *out_tensor);
        out_tensor->id = 0;
    }
    return st;
}

static inline fer_status_t fer_script_tensor_download_release(
    const fer_script_context_t* ctx,
    fer_tensor_handle_t tensor,
    void* host,
    size_t bytes) {
    if (!ctx || !ctx->api || !ctx->session || !ctx->api->tensor_download || !ctx->api->tensor_release) {
        return fer_status_t{FER_STATUS_INVALID_ARGUMENT, "script ctx/api is invalid"};
    }
    fer_status_t st = ctx->api->tensor_download(ctx->session, tensor, host, bytes);
    fer_status_t rel = ctx->api->tensor_release(ctx->session, tensor);
    return (st.code != FER_STATUS_OK) ? st : rel;
}

#define FER_SCRIPT_DEFINE_TYPED_IO(name, ctype)                                             \
static inline fer_status_t fer_script_tensor_create_upload_##name(                          \
    const fer_script_context_t* ctx,                                                         \
    const fer_tensor_spec_t* spec,                                                           \
    const ctype* host,                                                                       \
    size_t count,                                                                            \
    fer_tensor_handle_t* out_tensor) {                                                       \
    return fer_script_tensor_create_upload(ctx, spec, host, count * sizeof(ctype), out_tensor); \
}                                                                                            \
static inline fer_status_t fer_script_tensor_download_release_##name(                       \
    const fer_script_context_t* ctx,                                                         \
    fer_tensor_handle_t tensor,                                                              \
    ctype* host,                                                                             \
    size_t count) {                                                                          \
    return fer_script_tensor_download_release(ctx, tensor, host, count * sizeof(ctype));    \
}

FER_SCRIPT_DEFINE_TYPED_IO(f32, float)
FER_SCRIPT_DEFINE_TYPED_IO(f16, uint16_t)
FER_SCRIPT_DEFINE_TYPED_IO(bf16, uint16_t)
FER_SCRIPT_DEFINE_TYPED_IO(i8, int8_t)
FER_SCRIPT_DEFINE_TYPED_IO(u8, uint8_t)
FER_SCRIPT_DEFINE_TYPED_IO(i16, int16_t)
FER_SCRIPT_DEFINE_TYPED_IO(u16, uint16_t)
FER_SCRIPT_DEFINE_TYPED_IO(i32, int32_t)
FER_SCRIPT_DEFINE_TYPED_IO(u32, uint32_t)
FER_SCRIPT_DEFINE_TYPED_IO(i64, int64_t)
FER_SCRIPT_DEFINE_TYPED_IO(u64, uint64_t)
FER_SCRIPT_DEFINE_TYPED_IO(f64, double)

#undef FER_SCRIPT_DEFINE_TYPED_IO

#ifdef __cplusplus
}
#endif
