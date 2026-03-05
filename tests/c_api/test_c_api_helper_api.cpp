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
    cfg.mutable_bytes = 32ull * 1024ull * 1024ull;
    cfg.immutable_bytes = 32ull * 1024ull * 1024ull;
    cfg.memory_regime = FER_MEMORY_CUSTOM_POOL;
    if (fer_session_create(0, &cfg, &session).code != FER_STATUS_OK || !session) {
        return fail("fer_session_create");
    }

    constexpr uint32_t n = 64;
    std::vector<float> in(n, 3.0f);
    std::vector<float> out(n, 0.0f);

    const fer_tensor_spec_t spec =
        fer_tensor_spec_f32_1d(n, FER_MEMORY_CUSTOM_POOL);

    fer_tensor_handle_t x{};
    fer_tensor_handle_t y{};
    if (fer_tensor_create_upload_f32(session, &spec, in.data(), n, &x).code != FER_STATUS_OK) {
        return fail("fer_tensor_create_upload_f32 x");
    }
    if (fer_tensor_create(session, &spec, &y).code != FER_STATUS_OK) return fail("fer_tensor_create y");

    uint64_t job = 0;
    if (fer_tensor_run_affine_f32(
            session,
            x, y,
            n,
            2.0f, 1.0f,
            FER_JIT_INTENT_FUSION_RELU,
            FER_JIT_INTENT_CAPS_NONE,
            FER_MEMORY_CUSTOM_POOL,
            &job).code != FER_STATUS_OK || job == 0) {
        return fail("fer_tensor_run_affine_f32");
    }
    if (fer_job_wait(session, job).code != FER_STATUS_OK) return fail("fer_job_wait");
    if (fer_tensor_download_release_f32(session, y, out.data(), n).code != FER_STATUS_OK) {
        return fail("fer_tensor_download_release_f32 y");
    }

    for (uint32_t i = 0; i < n; ++i) {
        const float expect = 7.0f;
        if (std::fabs(out[i] - expect) > 1e-5f) return fail("result mismatch");
    }

    if (fer_tensor_release(session, x).code != FER_STATUS_OK) return fail("fer_tensor_release x");
    if (fer_session_destroy(session).code != FER_STATUS_OK) return fail("fer_session_destroy");
    std::printf("HELPER API TEST PASSED\n");
    return 0;
}
