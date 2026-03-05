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
    cfg.mutable_bytes = 64ull * 1024ull * 1024ull;
    cfg.immutable_bytes = 64ull * 1024ull * 1024ull;
    cfg.memory_regime = FER_MEMORY_CUSTOM_POOL;

    fer_status_t st = fer_session_create(0, &cfg, &session);
    if (st.code != FER_STATUS_OK || !session) return fail("fer_session_create");

    const char* src = "extern \"C\" __global__ void k(float* x){ if(threadIdx.x==0) x[0]+=1.0f; }";
    fer_jit_source_t source{};
    source.kind = FER_JIT_SOURCE_CUDA;
    source.code = src;
    source.code_len = std::strlen(src);

    fer_jit_options_t opts{};
    opts.backend = FER_JIT_BACKEND_NVRTC;
    opts.mode = FER_JIT_MODE_STRICT;
    opts.enable_disk_cache = 1;
    opts.cache_dir = "/tmp/fercuda_jit_cache_test";

    fer_jit_program_t program = nullptr;
    fer_jit_compile_result_t result{};
    st = fer_jit_compile(session, &source, &opts, &program, &result);
    if (st.code != FER_STATUS_OK || !program) return fail("fer_jit_compile");

    fer_jit_program_t program2 = nullptr;
    fer_jit_compile_result_t result2{};
    st = fer_jit_compile(session, &source, &opts, &program2, &result2);
    if (st.code != FER_STATUS_OK || !program2) return fail("second fer_jit_compile");
    if (result2.cache_hit != 1) return fail("second compile should be cache hit");

    fer_jit_kernel_t k = nullptr;
    st = fer_jit_get_kernel(session, program, "k", nullptr, &k);
    if (st.code != FER_STATUS_INVALID_ARGUMENT) return fail("strict mode should require signature");

    fer_jit_stats_t stats{};
    st = fer_jit_get_stats(session, &stats);
    if (st.code != FER_STATUS_OK) return fail("fer_jit_get_stats");
    if ((stats.compile_count + stats.cache_hit_count) < 1) return fail("expected compile or cache hit");

    st = fer_jit_cache_clear(session);
    if (st.code != FER_STATUS_OK) return fail("fer_jit_cache_clear");

    fer_jit_program_t program3 = nullptr;
    fer_jit_compile_result_t result3{};
    st = fer_jit_compile(session, &source, &opts, &program3, &result3);
    if (st.code != FER_STATUS_OK || !program3) return fail("third fer_jit_compile");
    if (result3.cache_hit != 1) return fail("expected disk cache hit after memory cache clear");

    st = fer_jit_release_program(session, program2);
    if (st.code != FER_STATUS_OK) return fail("release program2");
    st = fer_jit_release_program(session, program);
    if (st.code != FER_STATUS_OK) return fail("release program");
    st = fer_jit_release_program(session, program3);
    if (st.code != FER_STATUS_OK) return fail("release program3");

    st = fer_session_destroy(session);
    if (st.code != FER_STATUS_OK) return fail("fer_session_destroy");

    std::printf("JIT STUB API TEST PASSED\n");
    return 0;
}
