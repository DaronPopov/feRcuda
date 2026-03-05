#include "fercuda/api/c_api.h"
#include "fercuda/jit/intent/intent.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

static void deleter_cb(void*, void* user_ctx) {
    if (!user_ctx) return;
    int* count = static_cast<int*>(user_ctx);
    (*count)++;
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

    constexpr uint32_t n = 256;
    std::vector<float> h_in(n), h_out(n, 0.0f);
    for (uint32_t i = 0; i < n; ++i) h_in[i] = static_cast<float>(i);

    float* d_in = nullptr;
    float* d_out = nullptr;
    if (cudaMalloc(&d_in, n * sizeof(float)) != cudaSuccess) return fail("cudaMalloc d_in");
    if (cudaMalloc(&d_out, n * sizeof(float)) != cudaSuccess) return fail("cudaMalloc d_out");
    if (cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return fail("cudaMemcpy H2D");
    }

    fer_buffer_desc_t desc{};
    desc.dtype = FER_DTYPE_F32;
    desc.rank = 1;
    desc.dims[0] = n;

    uint64_t in_id = 0;
    uint64_t out_id = 0;
    int deleter_calls = 0;
    if (fer_import_external_buffer(session, &desc, d_in, FER_MEMORY_CUSTOM_POOL, &in_id).code != FER_STATUS_OK) {
        return fail("fer_import_external_buffer in");
    }
    if (fer_import_external_buffer_with_deleter(
            session, &desc, d_out, FER_MEMORY_CUSTOM_POOL, &deleter_cb, &deleter_calls, &out_id).code != FER_STATUS_OK) {
        return fail("fer_import_external_buffer_with_deleter out");
    }

    void* exported = nullptr;
    if (fer_export_buffer_device_ptr(session, out_id, &exported).code != FER_STATUS_OK) {
        return fail("fer_export_buffer_device_ptr");
    }
    if (exported != d_out) return fail("exported pointer mismatch");

    fer_jit_intent_t intent{};
    intent.abi_version = FER_JIT_INTENT_ABI_VERSION;
    intent.op = FER_JIT_INTENT_OP_AFFINE_F32;
    intent.fusion_mask = FER_JIT_INTENT_FUSION_NONE;
    intent.caps_mask = FER_JIT_INTENT_CAPS_NONE;
    intent.memory_regime = FER_MEMORY_CUSTOM_POOL;
    intent.n = n;
    intent.alpha = 3.0f;
    intent.beta = 2.0f;

    fer_jit_intent_bindings_t binds{};
    binds.input = in_id;
    binds.output = out_id;

    uint64_t job = 0;
    if (fer_jit_run_intent(session, &intent, &binds, &job).code != FER_STATUS_OK || job == 0) {
        return fail("fer_jit_run_intent");
    }
    if (fer_job_wait(session, job).code != FER_STATUS_OK) return fail("fer_job_wait");

    if (cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail("cudaMemcpy D2H");
    }
    for (uint32_t i = 0; i < n; ++i) {
        const float expect = 3.0f * h_in[i] + 2.0f;
        if (std::fabs(h_out[i] - expect) > 1e-5f) {
            std::fprintf(stderr, "mismatch i=%u got=%f expect=%f\n", i, h_out[i], expect);
            return fail("interop result mismatch");
        }
    }

    if (fer_free_buffer(session, in_id).code != FER_STATUS_OK) return fail("fer_free_buffer in");
    if (fer_free_buffer(session, out_id).code != FER_STATUS_OK) return fail("fer_free_buffer out");
    if (deleter_calls != 1) return fail("deleter callback not invoked exactly once");
    if (cudaFree(d_in) != cudaSuccess) return fail("cudaFree d_in");
    if (cudaFree(d_out) != cudaSuccess) return fail("cudaFree d_out");
    if (fer_session_destroy(session).code != FER_STATUS_OK) return fail("fer_session_destroy");

    std::printf("EXTERNAL PTR INTEROP TEST PASSED\n");
    return 0;
}
