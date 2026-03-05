#include "fercuda/api/c_api.h"
#include "fercuda/jit/api.h"

#include <cstdio>
#include <cstring>

static int fail(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

int main() {
    fer_session_t* session = nullptr;
    fer_pool_config_t cfg{};
    cfg.mutable_bytes = 16ull * 1024ull * 1024ull;
    cfg.immutable_bytes = 16ull * 1024ull * 1024ull;
    cfg.memory_regime = FER_MEMORY_CUSTOM_POOL;
    if (fer_session_create(0, &cfg, &session).code != FER_STATUS_OK || !session) {
        return fail("fer_session_create");
    }

    // Contract: capability gate should reject impossible SM requirement.
    const char* bad_src =
        "// fercuda:require_sm>=99\n"
        "extern \"C\" __global__ void k(float* x){ if(threadIdx.x==0) x[0]+=1.0f; }\n";
    fer_jit_source_t bad{};
    bad.kind = FER_JIT_SOURCE_CUDA;
    bad.code = bad_src;
    bad.code_len = std::strlen(bad_src);
    fer_jit_options_t opts{};
    opts.backend = FER_JIT_BACKEND_NVRTC;
    opts.mode = FER_JIT_MODE_STRICT;

    fer_jit_program_t p_bad = nullptr;
    fer_jit_compile_result_t r_bad{};
    fer_status_t st = fer_jit_compile(session, &bad, &opts, &p_bad, &r_bad);
    if (st.code != FER_STATUS_INVALID_ARGUMENT) return fail("expected lowering capability rejection");

    // Contract: fusion marker should pass and emit lowering notes.
    const char* ok_src =
        "// fercuda:fuse=elementwise_chain\n"
        "// fercuda:ir_node=elementwise_add\n"
        "// fercuda:ir_node=elementwise_mul\n"
        "extern \"C\" __global__ void k(float* x){ if(threadIdx.x==0) x[0]+=1.0f; }\n";
    fer_jit_source_t ok{};
    ok.kind = FER_JIT_SOURCE_CUDA;
    ok.code = ok_src;
    ok.code_len = std::strlen(ok_src);
    fer_jit_program_t p_ok = nullptr;
    fer_jit_compile_result_t r_ok{};
    st = fer_jit_compile(session, &ok, &opts, &p_ok, &r_ok);
    if (st.code != FER_STATUS_OK || !p_ok) return fail("expected lowering compile success");
    if (!r_ok.log || std::strstr(r_ok.log, "fused") == nullptr) {
        return fail("expected fusion lowering notes in compile log");
    }
    if (fer_jit_release_program(session, p_ok).code != FER_STATUS_OK) return fail("release program");

    if (fer_session_destroy(session).code != FER_STATUS_OK) return fail("fer_session_destroy");
    std::printf("JIT LOWERING TEST PASSED\n");
    return 0;
}
