#include "fercuda/jit/script_host.h"

#include <cmath>
#include <cstring>
#include <vector>

extern "C" uint32_t fer_script_abi_version(void) {
    return FER_SCRIPT_HOST_ABI_VERSION;
}

static fer_status_t fail(char* err_buf, size_t err_buf_len, const char* msg) {
    if (err_buf && err_buf_len > 0) {
        const size_t n = std::strlen(msg);
        const size_t m = (n < (err_buf_len - 1)) ? n : (err_buf_len - 1);
        std::memcpy(err_buf, msg, m);
        err_buf[m] = '\0';
    }
    return fer_status_t{FER_STATUS_INTERNAL_ERROR, msg};
}

extern "C" fer_status_t fer_script_main(
    const fer_script_context_t* ctx,
    const char*,
    char* err_buf,
    size_t err_buf_len) {
    if (!ctx || !ctx->api || !ctx->session) return fail(err_buf, err_buf_len, "invalid ctx");

    constexpr uint32_t n = 128;
    std::vector<float> in(n, 5.0f);
    std::vector<float> out(n, 0.0f);

    const fer_tensor_spec_t spec =
        fer_tensor_spec_f32_1d(n, FER_MEMORY_CUSTOM_POOL);
    fer_tensor_handle_t tensor{};
    if (fer_script_tensor_create_upload_f32(ctx, &spec, in.data(), in.size(), &tensor).code != FER_STATUS_OK) {
        return fail(err_buf, err_buf_len, "fer_script_tensor_create_upload_f32 failed");
    }

    fer_jit_intent_t intent{};
    intent.abi_version = FER_JIT_INTENT_ABI_VERSION;
    intent.op = FER_JIT_INTENT_OP_AFFINE_F32;
    intent.fusion_mask = FER_JIT_INTENT_FUSION_NONE;
    intent.caps_mask = FER_JIT_INTENT_CAPS_NONE;
    intent.memory_regime = FER_MEMORY_CUSTOM_POOL;
    intent.n = n;
    intent.alpha = 1.0f;
    intent.beta = 1.25f;

    uint64_t job = 0;
    if (!ctx->api->tensor_run_affine_f32) return fail(err_buf, err_buf_len, "tensor_run_affine_f32 api missing");
    if (ctx->api->tensor_run_affine_f32(
            ctx->session,
            tensor,
            tensor,
            n,
            intent.alpha,
            intent.beta,
            intent.fusion_mask,
            intent.caps_mask,
            intent.memory_regime,
            &job).code != FER_STATUS_OK || job == 0) {
        return fail(err_buf, err_buf_len, "tensor_run_affine_f32 failed");
    }
    if (ctx->api->job_wait(ctx->session, job).code != FER_STATUS_OK) {
        return fail(err_buf, err_buf_len, "job_wait failed");
    }
    if (fer_script_tensor_download_release_f32(ctx, tensor, out.data(), out.size()).code != FER_STATUS_OK) {
        return fail(err_buf, err_buf_len, "fer_script_tensor_download_release_f32 failed");
    }

    for (uint32_t i = 0; i < n; ++i) {
        if (std::fabs(out[i] - 6.25f) > 1e-5f) {
            return fail(err_buf, err_buf_len, "result mismatch");
        }
    }

    return fer_status_t{FER_STATUS_OK, "ok"};
}
