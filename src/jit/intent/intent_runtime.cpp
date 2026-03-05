#include "fercuda/jit/intent/intent.h"

#include "fercuda/jit/api.h"

#include <sstream>
#include <string>

namespace {

fer_status_t invalid(const char* msg) {
    return fer_status_t{FER_STATUS_INVALID_ARGUMENT, msg};
}

fer_status_t internal(const char* msg) {
    return fer_status_t{FER_STATUS_INTERNAL_ERROR, msg};
}

std::string build_affine_kernel_source(const fer_jit_intent_t* intent) {
    std::ostringstream src;
    src << "// fercuda:fuse=elementwise_chain\n";
    src << "// fercuda:ir_node=elementwise_mul\n";
    src << "// fercuda:ir_node=elementwise_add\n";
    if (intent->fusion_mask & FER_JIT_INTENT_FUSION_RELU) {
        src << "// fercuda:ir_node=elementwise_relu\n";
        src << "#define FERCUDA_INTENT_FUSE_RELU 1\n";
    }
    if (intent->caps_mask & FER_JIT_INTENT_CAPS_REQUIRE_TENSOR_CORES) {
        src << "// fercuda:require=tensor_cores\n";
    }
    if (intent->caps_mask & FER_JIT_INTENT_CAPS_REQUIRE_COOP_GROUPS) {
        src << "// fercuda:require=cooperative_groups\n";
    }
    src << "extern \"C\" __global__ void fer_intent_affine_f32(const float* in, float* out, float alpha, float beta, unsigned int n) {\n";
    src << "  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n";
    src << "  if (i < n) {\n";
    src << "    float v = alpha * in[i] + beta;\n";
    src << "#ifdef FERCUDA_INTENT_FUSE_RELU\n";
    src << "    if (v < 0.0f) v = 0.0f;\n";
    src << "#endif\n";
    src << "    out[i] = v;\n";
    src << "  }\n";
    src << "}\n";
    return src.str();
}

} // namespace

extern "C" fer_status_t fer_jit_run_intent(
    fer_session_t* session,
    const fer_jit_intent_t* intent,
    const fer_jit_intent_bindings_t* bindings,
    uint64_t* out_job_id) {
    if (!session) return invalid("session is null");
    if (!intent) return invalid("intent is null");
    if (!bindings) return invalid("bindings are null");
    if (!out_job_id) return invalid("out_job_id is null");
    if (intent->abi_version != FER_JIT_INTENT_ABI_VERSION) return invalid("intent ABI mismatch");
    if (intent->op != FER_JIT_INTENT_OP_AFFINE_F32) return invalid("unsupported intent op");
    if (intent->n == 0) return invalid("intent n must be > 0");

    const std::string source_text = build_affine_kernel_source(intent);
    fer_jit_source_t source{};
    source.kind = FER_JIT_SOURCE_CUDA;
    source.code = source_text.c_str();
    source.code_len = source_text.size();

    fer_jit_options_t options{};
    options.backend = FER_JIT_BACKEND_NVRTC;
    options.mode = FER_JIT_MODE_STRICT;
    options.enable_disk_cache = 1;
    options.cache_dir = "/tmp/fercuda_jit_cache_intent";

    fer_jit_program_t program = nullptr;
    fer_jit_compile_result_t compile_result{};
    fer_status_t st = fer_jit_compile(session, &source, &options, &program, &compile_result);
    if (st.code != FER_STATUS_OK || !program) return st.code == FER_STATUS_OK ? internal("intent compile failed") : st;

    fer_jit_arg_desc_t descs[5]{};
    descs[0].kind = FER_JIT_ARG_BUFFER;
    descs[0].access = FER_JIT_ACCESS_READ;
    descs[0].expected_dtype = FER_DTYPE_F32;
    descs[0].expected_rank = 1;
    descs[0].expected_dims[0] = intent->n;
    descs[0].expected_dims[1] = FER_JIT_WILDCARD_U32;
    descs[0].expected_dims[2] = FER_JIT_WILDCARD_U32;
    descs[0].expected_dims[3] = FER_JIT_WILDCARD_U32;
    descs[0].expected_bytes = static_cast<uint64_t>(intent->n) * sizeof(float);

    descs[1] = descs[0];
    descs[1].access = FER_JIT_ACCESS_WRITE;

    descs[2].kind = FER_JIT_ARG_SCALAR_F32;
    descs[2].expected_dtype = FER_JIT_WILDCARD_U32;
    descs[2].expected_rank = FER_JIT_WILDCARD_U32;
    descs[2].expected_bytes = FER_JIT_WILDCARD_U64;
    descs[2].expected_dims[0] = FER_JIT_WILDCARD_U32;
    descs[2].expected_dims[1] = FER_JIT_WILDCARD_U32;
    descs[2].expected_dims[2] = FER_JIT_WILDCARD_U32;
    descs[2].expected_dims[3] = FER_JIT_WILDCARD_U32;
    descs[3] = descs[2];

    descs[4].kind = FER_JIT_ARG_SCALAR_U32;
    descs[4].expected_dtype = FER_JIT_WILDCARD_U32;
    descs[4].expected_rank = FER_JIT_WILDCARD_U32;
    descs[4].expected_bytes = FER_JIT_WILDCARD_U64;
    descs[4].expected_dims[0] = FER_JIT_WILDCARD_U32;
    descs[4].expected_dims[1] = FER_JIT_WILDCARD_U32;
    descs[4].expected_dims[2] = FER_JIT_WILDCARD_U32;
    descs[4].expected_dims[3] = FER_JIT_WILDCARD_U32;

    fer_jit_kernel_sig_t sig{};
    sig.args = descs;
    sig.arg_count = 5;

    fer_jit_kernel_t kernel = nullptr;
    st = fer_jit_get_kernel(session, program, "fer_intent_affine_f32", &sig, &kernel);
    if (st.code != FER_STATUS_OK || !kernel) {
        fer_jit_release_program(session, program);
        return st.code == FER_STATUS_OK ? internal("intent get kernel failed") : st;
    }

    fer_jit_arg_value_t args[5]{};
    args[0].kind = FER_JIT_ARG_BUFFER;
    args[0].as.buffer_id = bindings->input;
    args[1].kind = FER_JIT_ARG_BUFFER;
    args[1].as.buffer_id = bindings->output;
    args[2].kind = FER_JIT_ARG_SCALAR_F32;
    args[2].as.f32 = intent->alpha;
    args[3].kind = FER_JIT_ARG_SCALAR_F32;
    args[3].as.f32 = intent->beta;
    args[4].kind = FER_JIT_ARG_SCALAR_U32;
    args[4].as.u32 = intent->n;

    fer_jit_arg_pack_t pack{};
    pack.args = args;
    pack.arg_count = 5;

    fer_jit_launch_cfg_t launch{};
    launch.grid_x = (intent->n + 255u) / 256u;
    launch.grid_y = 1;
    launch.grid_z = 1;
    launch.block_x = 256;
    launch.block_y = 1;
    launch.block_z = 1;
    launch.shared_mem_bytes = 0;
    launch.memory_regime = (intent->memory_regime == FER_MEMORY_AUTO) ? FER_MEMORY_CUSTOM_POOL : intent->memory_regime;

    st = fer_jit_launch(session, kernel, &launch, &pack, out_job_id);

    fer_jit_release_kernel(session, kernel);
    fer_jit_release_program(session, program);
    return st;
}
