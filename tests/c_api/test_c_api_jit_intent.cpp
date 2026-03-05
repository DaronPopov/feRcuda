#include "fercuda/api/c_api.h"
#include "fercuda/jit/intent/intent.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

int main() {
    fer_session_t* session = nullptr;
    fer_pool_config_t cfg{};
    cfg.mutable_bytes = 64ull * 1024ull * 1024ull;
    cfg.immutable_bytes = 64ull * 1024ull * 1024ull;
    cfg.memory_regime = FER_MEMORY_CUSTOM_POOL; // TLSF/custom-pool baseline
    if (fer_session_create(0, &cfg, &session).code != FER_STATUS_OK || !session) {
        return fail("fer_session_create");
    }

    constexpr uint32_t n = 512;
    std::vector<float> in(n);
    std::vector<float> out(n, 0.0f);
    for (uint32_t i = 0; i < n; ++i) in[i] = static_cast<float>(i) - 100.0f;

    fer_buffer_desc_t desc{};
    desc.dtype = FER_DTYPE_F32;
    desc.rank = 1;
    desc.dims[0] = n;

    uint64_t in_buf = 0;
    uint64_t out_buf = 0;
    if (fer_alloc_buffer(session, &desc, &in_buf).code != FER_STATUS_OK) return fail("alloc in");
    if (fer_alloc_buffer(session, &desc, &out_buf).code != FER_STATUS_OK) return fail("alloc out");
    if (fer_upload_bytes(session, in_buf, in.data(), in.size() * sizeof(float)).code != FER_STATUS_OK) {
        return fail("upload in");
    }

    fer_jit_intent_t intent{};
    intent.abi_version = FER_JIT_INTENT_ABI_VERSION;
    intent.op = FER_JIT_INTENT_OP_AFFINE_F32;
    intent.fusion_mask = FER_JIT_INTENT_FUSION_RELU;
    intent.caps_mask = FER_JIT_INTENT_CAPS_NONE;
    intent.memory_regime = FER_MEMORY_CUSTOM_POOL;
    intent.n = n;
    intent.alpha = 0.5f;
    intent.beta = 10.0f;

    fer_jit_intent_bindings_t binds{};
    binds.input = in_buf;
    binds.output = out_buf;

    uint64_t job = 0;
    fer_status_t st = fer_jit_run_intent(session, &intent, &binds, &job);
    if (st.code != FER_STATUS_OK || job == 0) return fail("fer_jit_run_intent");
    if (fer_job_wait(session, job).code != FER_STATUS_OK) return fail("fer_job_wait");
    if (fer_download_bytes(session, out_buf, out.data(), out.size() * sizeof(float)).code != FER_STATUS_OK) {
        return fail("download out");
    }

    for (uint32_t i = 0; i < n; ++i) {
        float v = 0.5f * in[i] + 10.0f;
        if (v < 0.0f) v = 0.0f;
        if (std::fabs(out[i] - v) > 1e-5f) {
            std::fprintf(stderr, "mismatch i=%u got=%f expect=%f\n", i, out[i], v);
            return fail("intent result mismatch");
        }
    }

    if (fer_free_buffer(session, in_buf).code != FER_STATUS_OK) return fail("free in");
    if (fer_free_buffer(session, out_buf).code != FER_STATUS_OK) return fail("free out");
    if (fer_session_destroy(session).code != FER_STATUS_OK) return fail("destroy");
    std::printf("JIT INTENT TEST PASSED\n");
    return 0;
}
