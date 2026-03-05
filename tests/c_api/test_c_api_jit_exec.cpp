#include "fercuda/api/c_api.h"
#include "fercuda/jit/api.h"

#include <cmath>
#include <cstdio>
#include <cstring>
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
    cfg.memory_regime = FER_MEMORY_CUSTOM_POOL;
    if (fer_session_create(0, &cfg, &session).code != FER_STATUS_OK || !session) {
        return fail("fer_session_create");
    }

    constexpr uint32_t n = 256;
    std::vector<float> in(n, 1.5f);
    std::vector<float> out(n, 0.0f);

    fer_buffer_desc_t desc{};
    desc.dtype = FER_DTYPE_F32;
    desc.rank = 1;
    desc.dims[0] = n;
    uint64_t buf = 0;
    if (fer_alloc_buffer(session, &desc, &buf).code != FER_STATUS_OK) return fail("fer_alloc_buffer");
    if (fer_upload_bytes(session, buf, in.data(), in.size() * sizeof(float)).code != FER_STATUS_OK) {
        return fail("fer_upload_bytes");
    }

    const char* src =
        "extern \"C\" __global__ void add_scalar(float* x, float v, unsigned int n) {"
        "  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;"
        "  if (i < n) x[i] += v;"
        "}";
    fer_jit_source_t source{};
    source.kind = FER_JIT_SOURCE_CUDA;
    source.code = src;
    source.code_len = std::strlen(src);
    fer_jit_options_t opts{};
    opts.backend = FER_JIT_BACKEND_NVRTC;
    opts.mode = FER_JIT_MODE_STRICT;

    fer_jit_program_t program = nullptr;
    fer_jit_compile_result_t compile_result{};
    fer_status_t st = fer_jit_compile(session, &source, &opts, &program, &compile_result);
    if (st.code != FER_STATUS_OK || !program) {
        std::fprintf(stderr, "compile log: %s\n", compile_result.log ? compile_result.log : "(null)");
        return fail("fer_jit_compile");
    }

    fer_jit_arg_desc_t arg_descs[3]{};
    arg_descs[0].kind = FER_JIT_ARG_BUFFER;
    arg_descs[0].access = FER_JIT_ACCESS_READ_WRITE;
    arg_descs[0].expected_dtype = FER_DTYPE_F32;
    arg_descs[0].expected_rank = 1;
    arg_descs[0].expected_bytes = static_cast<uint64_t>(n) * sizeof(float);
    arg_descs[0].expected_dims[0] = n;
    arg_descs[0].expected_dims[1] = FER_JIT_WILDCARD_U32;
    arg_descs[0].expected_dims[2] = FER_JIT_WILDCARD_U32;
    arg_descs[0].expected_dims[3] = FER_JIT_WILDCARD_U32;
    arg_descs[1].kind = FER_JIT_ARG_SCALAR_F32;
    arg_descs[1].expected_dtype = FER_JIT_WILDCARD_U32;
    arg_descs[1].expected_rank = FER_JIT_WILDCARD_U32;
    arg_descs[1].expected_bytes = FER_JIT_WILDCARD_U64;
    arg_descs[1].expected_dims[0] = FER_JIT_WILDCARD_U32;
    arg_descs[1].expected_dims[1] = FER_JIT_WILDCARD_U32;
    arg_descs[1].expected_dims[2] = FER_JIT_WILDCARD_U32;
    arg_descs[1].expected_dims[3] = FER_JIT_WILDCARD_U32;
    arg_descs[2].kind = FER_JIT_ARG_SCALAR_U32;
    arg_descs[2].expected_dtype = FER_JIT_WILDCARD_U32;
    arg_descs[2].expected_rank = FER_JIT_WILDCARD_U32;
    arg_descs[2].expected_bytes = FER_JIT_WILDCARD_U64;
    arg_descs[2].expected_dims[0] = FER_JIT_WILDCARD_U32;
    arg_descs[2].expected_dims[1] = FER_JIT_WILDCARD_U32;
    arg_descs[2].expected_dims[2] = FER_JIT_WILDCARD_U32;
    arg_descs[2].expected_dims[3] = FER_JIT_WILDCARD_U32;
    fer_jit_kernel_sig_t sig{};
    sig.args = arg_descs;
    sig.arg_count = 3;

    fer_jit_kernel_t kernel = nullptr;
    if (fer_jit_get_kernel(session, program, "add_scalar", &sig, &kernel).code != FER_STATUS_OK || !kernel) {
        return fail("fer_jit_get_kernel");
    }

    fer_jit_arg_value_t values[3]{};
    values[0].kind = FER_JIT_ARG_BUFFER;
    values[0].as.buffer_id = buf;
    values[1].kind = FER_JIT_ARG_SCALAR_F32;
    values[1].as.f32 = 2.25f;
    values[2].kind = FER_JIT_ARG_SCALAR_U32;
    values[2].as.u32 = n;
    fer_jit_arg_pack_t pack{};
    pack.args = values;
    pack.arg_count = 3;

    fer_jit_launch_cfg_t launch{};
    launch.grid_x = (n + 127u) / 128u;
    launch.grid_y = 1;
    launch.grid_z = 1;
    launch.block_x = 128;
    launch.block_y = 1;
    launch.block_z = 1;

    uint64_t job = 0;
    if (fer_jit_launch(session, kernel, &launch, &pack, &job).code != FER_STATUS_OK || job == 0) {
        return fail("fer_jit_launch");
    }
    if (fer_job_wait(session, job).code != FER_STATUS_OK) return fail("fer_job_wait");
    if (fer_download_bytes(session, buf, out.data(), out.size() * sizeof(float)).code != FER_STATUS_OK) {
        return fail("fer_download_bytes");
    }

    for (uint32_t i = 0; i < n; ++i) {
        const float expect = 3.75f;
        if (std::fabs(out[i] - expect) > 1e-5f) {
            std::fprintf(stderr, "mismatch i=%u got=%f expect=%f\n", i, out[i], expect);
            return fail("result mismatch");
        }
    }

    fer_jit_stats_t stats{};
    if (fer_jit_get_stats(session, &stats).code != FER_STATUS_OK) return fail("fer_jit_get_stats");
    if (stats.compile_count < 1 || stats.launch_count < 1) return fail("jit stats not incremented");

    fer_jit_arg_desc_t bad_descs[3] = {arg_descs[0], arg_descs[1], arg_descs[2]};
    bad_descs[0].expected_rank = 2;
    fer_jit_kernel_sig_t bad_sig{};
    bad_sig.args = bad_descs;
    bad_sig.arg_count = 3;
    fer_jit_kernel_t bad_kernel = nullptr;
    if (fer_jit_get_kernel(session, program, "add_scalar", &bad_sig, &bad_kernel).code != FER_STATUS_OK || !bad_kernel) {
        return fail("fer_jit_get_kernel bad_sig");
    }
    values[0].as.buffer_id = buf;
    uint64_t bad_job = 0;
    st = fer_jit_launch(session, bad_kernel, &launch, &pack, &bad_job);
    if (st.code != FER_STATUS_INVALID_ARGUMENT) return fail("expected rank contract mismatch");
    if (fer_jit_release_kernel(session, bad_kernel).code != FER_STATUS_OK) return fail("release bad kernel");

    fer_buffer_desc_t imm_desc = desc;
    imm_desc.immutable = 1;
    uint64_t imm_buf = 0;
    if (fer_alloc_buffer(session, &imm_desc, &imm_buf).code != FER_STATUS_OK) return fail("alloc immutable buffer");
    if (fer_upload_bytes(session, imm_buf, in.data(), in.size() * sizeof(float)).code != FER_STATUS_OK) {
        return fail("upload immutable");
    }
    values[0].as.buffer_id = imm_buf;
    uint64_t imm_job = 0;
    st = fer_jit_launch(session, kernel, &launch, &pack, &imm_job);
    if (st.code != FER_STATUS_INVALID_ARGUMENT) return fail("expected immutable write rejection");

    if (fer_free_buffer(session, imm_buf).code != FER_STATUS_OK) return fail("free immutable buffer");

    if (fer_jit_release_kernel(session, kernel).code != FER_STATUS_OK) return fail("release kernel");
    if (fer_jit_release_program(session, program).code != FER_STATUS_OK) return fail("release program");
    if (fer_free_buffer(session, buf).code != FER_STATUS_OK) return fail("fer_free_buffer");
    if (fer_session_destroy(session).code != FER_STATUS_OK) return fail("fer_session_destroy");
    std::printf("JIT EXEC API TEST PASSED\n");
    return 0;
}
