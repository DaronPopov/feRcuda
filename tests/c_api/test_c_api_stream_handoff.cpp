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

int main() {
    fer_session_t* session = nullptr;
    fer_pool_config_t cfg{};
    cfg.mutable_bytes = 32ull * 1024ull * 1024ull;
    cfg.immutable_bytes = 32ull * 1024ull * 1024ull;
    cfg.memory_regime = FER_MEMORY_CUSTOM_POOL;
    if (fer_session_create(0, &cfg, &session).code != FER_STATUS_OK || !session) {
        return fail("fer_session_create");
    }

    cudaStream_t ext_stream = nullptr;
    if (cudaStreamCreateWithFlags(&ext_stream, cudaStreamNonBlocking) != cudaSuccess) {
        return fail("cudaStreamCreateWithFlags");
    }
    const uint64_t ext_handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ext_stream));
    if (fer_set_session_stream(session, ext_handle).code != FER_STATUS_OK) return fail("fer_set_session_stream");
    uint64_t got = 0;
    if (fer_get_session_stream(session, &got).code != FER_STATUS_OK) return fail("fer_get_session_stream");
    if (got != ext_handle) return fail("session stream handle mismatch");

    constexpr uint32_t n = 128;
    std::vector<float> in(n, 2.0f), out(n, 0.0f);
    fer_buffer_desc_t desc{};
    desc.dtype = FER_DTYPE_F32;
    desc.rank = 1;
    desc.dims[0] = n;
    uint64_t in_id = 0, out_id = 0;
    if (fer_alloc_buffer(session, &desc, &in_id).code != FER_STATUS_OK) return fail("alloc in");
    if (fer_alloc_buffer(session, &desc, &out_id).code != FER_STATUS_OK) return fail("alloc out");
    if (fer_upload_bytes(session, in_id, in.data(), n * sizeof(float)).code != FER_STATUS_OK) return fail("upload");

    fer_jit_intent_t intent{};
    intent.abi_version = FER_JIT_INTENT_ABI_VERSION;
    intent.op = FER_JIT_INTENT_OP_AFFINE_F32;
    intent.fusion_mask = FER_JIT_INTENT_FUSION_NONE;
    intent.caps_mask = FER_JIT_INTENT_CAPS_NONE;
    intent.memory_regime = FER_MEMORY_CUSTOM_POOL;
    intent.n = n;
    intent.alpha = 4.0f;
    intent.beta = 1.0f;
    fer_jit_intent_bindings_t binds{};
    binds.input = in_id;
    binds.output = out_id;
    uint64_t job = 0;
    if (fer_jit_run_intent(session, &intent, &binds, &job).code != FER_STATUS_OK || job == 0) {
        return fail("fer_jit_run_intent");
    }
    if (fer_job_wait(session, job).code != FER_STATUS_OK) return fail("fer_job_wait");
    if (fer_download_bytes(session, out_id, out.data(), n * sizeof(float)).code != FER_STATUS_OK) return fail("download");

    for (uint32_t i = 0; i < n; ++i) {
        const float expect = 9.0f;
        if (std::fabs(out[i] - expect) > 1e-5f) return fail("result mismatch");
    }

    if (fer_free_buffer(session, in_id).code != FER_STATUS_OK) return fail("free in");
    if (fer_free_buffer(session, out_id).code != FER_STATUS_OK) return fail("free out");
    if (cudaStreamDestroy(ext_stream) != cudaSuccess) return fail("cudaStreamDestroy");
    if (fer_session_destroy(session).code != FER_STATUS_OK) return fail("destroy");
    std::printf("STREAM HANDOFF TEST PASSED\n");
    return 0;
}
