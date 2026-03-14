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

fer_jit_arg_desc_t make_wildcard_scalar(uint32_t kind) {
    fer_jit_arg_desc_t d{};
    d.kind = kind;
    d.expected_dtype = FER_JIT_WILDCARD_U32;
    d.expected_rank = FER_JIT_WILDCARD_U32;
    d.expected_bytes = FER_JIT_WILDCARD_U64;
    for (int i = 0; i < 4; ++i) d.expected_dims[i] = FER_JIT_WILDCARD_U32;
    return d;
}

fer_jit_arg_desc_t make_buffer_desc(uint32_t access, uint32_t dtype, uint32_t rank, uint64_t bytes) {
    fer_jit_arg_desc_t d{};
    d.kind = FER_JIT_ARG_BUFFER;
    d.access = access;
    d.expected_dtype = dtype;
    d.expected_rank = rank;
    d.expected_bytes = bytes;
    for (int i = 0; i < 4; ++i) d.expected_dims[i] = FER_JIT_WILDCARD_U32;
    return d;
}

// ---------------------------------------------------------------------------
// Kernel source generators
// ---------------------------------------------------------------------------

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
    src << "extern \"C\" __global__ void fer_intent_affine_f32("
           "const float* in, float* out, float alpha, float beta, unsigned int n) {\n"
           "  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) {\n"
           "    float v = alpha * in[i] + beta;\n"
           "#ifdef FERCUDA_INTENT_FUSE_RELU\n"
           "    if (v < 0.0f) v = 0.0f;\n"
           "#endif\n"
           "    out[i] = v;\n"
           "  }\n"
           "}\n";
    return src.str();
}

std::string build_softmax_kernel_source(const fer_jit_intent_t*) {
    // Two-pass softmax: find row-max, then exp and normalize.
    // Launched with one block per row; shared memory used for reductions.
    return R"(
extern "C" __global__ void fer_intent_softmax_f32(
        const float* __restrict__ in,
        float* __restrict__ out,
        unsigned int cols) {
    extern __shared__ float sdata[];
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const float* row_in = in + row * cols;
    float* row_out = out + row * cols;

    // Phase 1: row-max reduction
    float local_max = -1e30f;
    for (unsigned int j = tid; j < cols; j += blockDim.x)
        local_max = fmaxf(local_max, row_in[j]);
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    const float row_max = sdata[0];

    // Phase 2: exp and sum
    float local_sum = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        float e = expf(row_in[j] - row_max);
        row_out[j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float inv_sum = 1.0f / sdata[0];

    // Phase 3: normalize
    for (unsigned int j = tid; j < cols; j += blockDim.x)
        row_out[j] *= inv_sum;
}
)";
}

std::string build_reduce_sum_kernel_source(const fer_jit_intent_t*) {
    return R"(
extern "C" __global__ void fer_intent_reduce_sum_f32(
        const float* __restrict__ in,
        float* __restrict__ out,
        unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
    float v = 0.0f;
    if (i < n) v = in[i];
    if (i + blockDim.x < n) v += in[i + blockDim.x];
    sdata[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(&out[0], sdata[0]);
}
)";
}

std::string build_reduce_max_kernel_source(const fer_jit_intent_t*) {
    return R"(
extern "C" __global__ void fer_intent_reduce_max_f32(
        const float* __restrict__ in,
        float* __restrict__ out,
        unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
    float v = -1e30f;
    if (i < n) v = in[i];
    if (i + blockDim.x < n) v = fmaxf(v, in[i + blockDim.x]);
    sdata[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        // Atomic max for float via CAS
        float old = out[0];
        while (sdata[0] > old) {
            unsigned int assumed = __float_as_uint(old);
            unsigned int val = __float_as_uint(sdata[0]);
            unsigned int result = atomicCAS(
                reinterpret_cast<unsigned int*>(&out[0]), assumed, val);
            old = __uint_as_float(result);
        }
    }
}
)";
}

std::string build_conv2d_kernel_source(const fer_jit_intent_t* intent) {
    std::ostringstream src;
    if (intent->fusion_mask & FER_JIT_INTENT_FUSION_RELU) {
        src << "#define FERCUDA_INTENT_FUSE_RELU 1\n";
    }
    // Direct convolution: each thread computes one output pixel for one filter.
    src << R"(
extern "C" __global__ void fer_intent_conv2d_f32(
        const float* __restrict__ input,
        const float* __restrict__ weights,
        const float* __restrict__ bias,
        float* __restrict__ output,
        unsigned int H, unsigned int W, unsigned int C,
        unsigned int KH, unsigned int KW, unsigned int F,
        unsigned int pad_h, unsigned int pad_w,
        unsigned int stride_h, unsigned int stride_w) {
    const unsigned int out_h = (H + 2 * pad_h - KH) / stride_h + 1;
    const unsigned int out_w = (W + 2 * pad_w - KW) / stride_w + 1;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = out_h * out_w * F;
    if (idx >= total) return;
    const unsigned int f = idx % F;
    const unsigned int pw = (idx / F) % out_w;
    const unsigned int ph = (idx / F) / out_w;
    float acc = (bias != nullptr) ? bias[f] : 0.0f;
    for (unsigned int c = 0; c < C; ++c) {
        for (unsigned int kh = 0; kh < KH; ++kh) {
            for (unsigned int kw = 0; kw < KW; ++kw) {
                int ih = (int)(ph * stride_h + kh) - (int)pad_h;
                int iw = (int)(pw * stride_w + kw) - (int)pad_w;
                if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                    float iv = input[((unsigned int)ih * W + (unsigned int)iw) * C + c];
                    float wv = weights[((f * C + c) * KH + kh) * KW + kw];
                    acc += iv * wv;
                }
            }
        }
    }
#ifdef FERCUDA_INTENT_FUSE_RELU
    if (acc < 0.0f) acc = 0.0f;
#endif
    output[(ph * out_w + pw) * F + f] = acc;
}
)";
    return src.str();
}

// ---------------------------------------------------------------------------
// Shared launch helper
// ---------------------------------------------------------------------------

fer_status_t compile_and_launch(
    fer_session_t* session,
    const std::string& source_text,
    const char* kernel_name,
    const fer_jit_arg_desc_t* descs,
    size_t desc_count,
    const fer_jit_arg_value_t* args,
    size_t arg_count,
    const fer_jit_launch_cfg_t* launch_cfg,
    uint64_t* out_job_id) {

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
    if (st.code != FER_STATUS_OK || !program)
        return st.code == FER_STATUS_OK ? internal("intent compile failed") : st;

    fer_jit_kernel_sig_t sig{};
    sig.args = descs;
    sig.arg_count = desc_count;

    fer_jit_kernel_t kernel = nullptr;
    st = fer_jit_get_kernel(session, program, kernel_name, &sig, &kernel);
    if (st.code != FER_STATUS_OK || !kernel) {
        fer_jit_release_program(session, program);
        return st.code == FER_STATUS_OK ? internal("intent get kernel failed") : st;
    }

    fer_jit_arg_pack_t pack{};
    pack.args = args;
    pack.arg_count = arg_count;

    st = fer_jit_launch(session, kernel, launch_cfg, &pack, out_job_id);
    fer_jit_release_kernel(session, kernel);
    fer_jit_release_program(session, program);
    return st;
}

// ---------------------------------------------------------------------------
// Per-op dispatchers
// ---------------------------------------------------------------------------

fer_status_t run_affine_f32(
    fer_session_t* session,
    const fer_jit_intent_t* intent,
    const fer_jit_intent_bindings_t* bindings,
    uint64_t* out_job_id) {
    if (intent->n == 0) return invalid("intent n must be > 0");

    const std::string source_text = build_affine_kernel_source(intent);

    fer_jit_arg_desc_t descs[5]{};
    descs[0] = make_buffer_desc(FER_JIT_ACCESS_READ, FER_DTYPE_F32, 1,
                                static_cast<uint64_t>(intent->n) * sizeof(float));
    descs[0].expected_dims[0] = intent->n;
    descs[1] = descs[0];
    descs[1].access = FER_JIT_ACCESS_WRITE;
    descs[2] = make_wildcard_scalar(FER_JIT_ARG_SCALAR_F32);
    descs[3] = make_wildcard_scalar(FER_JIT_ARG_SCALAR_F32);
    descs[4] = make_wildcard_scalar(FER_JIT_ARG_SCALAR_U32);

    fer_jit_arg_value_t args[5]{};
    args[0].kind = FER_JIT_ARG_BUFFER;  args[0].as.buffer_id = bindings->input;
    args[1].kind = FER_JIT_ARG_BUFFER;  args[1].as.buffer_id = bindings->output;
    args[2].kind = FER_JIT_ARG_SCALAR_F32; args[2].as.f32 = intent->alpha;
    args[3].kind = FER_JIT_ARG_SCALAR_F32; args[3].as.f32 = intent->beta;
    args[4].kind = FER_JIT_ARG_SCALAR_U32; args[4].as.u32 = intent->n;

    fer_jit_launch_cfg_t launch{};
    launch.grid_x = (intent->n + 255u) / 256u;
    launch.grid_y = 1; launch.grid_z = 1;
    launch.block_x = 256; launch.block_y = 1; launch.block_z = 1;
    launch.memory_regime = (intent->memory_regime == FER_MEMORY_AUTO)
                               ? FER_MEMORY_CUSTOM_POOL : intent->memory_regime;

    return compile_and_launch(session, source_text, "fer_intent_affine_f32",
                              descs, 5, args, 5, &launch, out_job_id);
}

fer_status_t run_softmax_f32(
    fer_session_t* session,
    const fer_jit_intent_t* intent,
    const fer_jit_intent_bindings_t* bindings,
    uint64_t* out_job_id) {
    if (intent->n == 0) return invalid("intent n (cols) must be > 0");
    const uint32_t rows = (intent->height > 0) ? intent->height : 1;
    const uint32_t cols = intent->n;
    const uint64_t total_bytes = static_cast<uint64_t>(rows) * cols * sizeof(float);

    const std::string source_text = build_softmax_kernel_source(intent);

    fer_jit_arg_desc_t descs[3]{};
    descs[0] = make_buffer_desc(FER_JIT_ACCESS_READ, FER_DTYPE_F32, FER_JIT_WILDCARD_U32, total_bytes);
    descs[1] = make_buffer_desc(FER_JIT_ACCESS_WRITE, FER_DTYPE_F32, FER_JIT_WILDCARD_U32, total_bytes);
    descs[2] = make_wildcard_scalar(FER_JIT_ARG_SCALAR_U32);

    fer_jit_arg_value_t args[3]{};
    args[0].kind = FER_JIT_ARG_BUFFER;  args[0].as.buffer_id = bindings->input;
    args[1].kind = FER_JIT_ARG_BUFFER;  args[1].as.buffer_id = bindings->output;
    args[2].kind = FER_JIT_ARG_SCALAR_U32; args[2].as.u32 = cols;

    const uint32_t block_x = (cols < 256) ? cols : 256;
    fer_jit_launch_cfg_t launch{};
    launch.grid_x = rows; launch.grid_y = 1; launch.grid_z = 1;
    launch.block_x = block_x; launch.block_y = 1; launch.block_z = 1;
    launch.shared_mem_bytes = block_x * sizeof(float);
    launch.memory_regime = (intent->memory_regime == FER_MEMORY_AUTO)
                               ? FER_MEMORY_CUSTOM_POOL : intent->memory_regime;

    return compile_and_launch(session, source_text, "fer_intent_softmax_f32",
                              descs, 3, args, 3, &launch, out_job_id);
}

fer_status_t run_reduce_f32(
    fer_session_t* session,
    const fer_jit_intent_t* intent,
    const fer_jit_intent_bindings_t* bindings,
    uint64_t* out_job_id,
    bool is_sum) {
    if (intent->n == 0) return invalid("intent n must be > 0");

    const std::string source_text = is_sum
        ? build_reduce_sum_kernel_source(intent)
        : build_reduce_max_kernel_source(intent);
    const char* kernel_name = is_sum
        ? "fer_intent_reduce_sum_f32"
        : "fer_intent_reduce_max_f32";

    fer_jit_arg_desc_t descs[3]{};
    descs[0] = make_buffer_desc(FER_JIT_ACCESS_READ, FER_DTYPE_F32, 1,
                                static_cast<uint64_t>(intent->n) * sizeof(float));
    descs[1] = make_buffer_desc(FER_JIT_ACCESS_READ_WRITE, FER_DTYPE_F32, 1, sizeof(float));
    descs[2] = make_wildcard_scalar(FER_JIT_ARG_SCALAR_U32);

    fer_jit_arg_value_t args[3]{};
    args[0].kind = FER_JIT_ARG_BUFFER;  args[0].as.buffer_id = bindings->input;
    args[1].kind = FER_JIT_ARG_BUFFER;  args[1].as.buffer_id = bindings->output;
    args[2].kind = FER_JIT_ARG_SCALAR_U32; args[2].as.u32 = intent->n;

    const uint32_t block_x = 256;
    const uint32_t grid_x = (intent->n + block_x * 2 - 1) / (block_x * 2);
    fer_jit_launch_cfg_t launch{};
    launch.grid_x = grid_x; launch.grid_y = 1; launch.grid_z = 1;
    launch.block_x = block_x; launch.block_y = 1; launch.block_z = 1;
    launch.shared_mem_bytes = block_x * sizeof(float);
    launch.memory_regime = (intent->memory_regime == FER_MEMORY_AUTO)
                               ? FER_MEMORY_CUSTOM_POOL : intent->memory_regime;

    return compile_and_launch(session, source_text, kernel_name,
                              descs, 3, args, 3, &launch, out_job_id);
}

fer_status_t run_conv2d_f32(
    fer_session_t* session,
    const fer_jit_intent_t* intent,
    const fer_jit_intent_bindings_t* bindings,
    uint64_t* out_job_id) {
    if (intent->height == 0 || intent->width == 0 || intent->channels == 0)
        return invalid("conv2d requires height, width, channels > 0");
    if (intent->kernel_h == 0 || intent->kernel_w == 0 || intent->num_filters == 0)
        return invalid("conv2d requires kernel_h, kernel_w, num_filters > 0");

    const uint32_t H = intent->height, W = intent->width, C = intent->channels;
    const uint32_t KH = intent->kernel_h, KW = intent->kernel_w;
    const uint32_t F = intent->num_filters;
    const uint32_t sh = (intent->stride_h > 0) ? intent->stride_h : 1;
    const uint32_t sw = (intent->stride_w > 0) ? intent->stride_w : 1;
    const uint32_t out_h = (H + 2 * intent->pad_h - KH) / sh + 1;
    const uint32_t out_w = (W + 2 * intent->pad_w - KW) / sw + 1;
    const uint32_t total_out = out_h * out_w * F;
    const bool has_bias = (bindings->bias != 0);

    const std::string source_text = build_conv2d_kernel_source(intent);

    // input: [H*W*C], weights: [F*C*KH*KW], bias: [F], output: [out_h*out_w*F]
    fer_jit_arg_desc_t descs[14]{};
    descs[0] = make_buffer_desc(FER_JIT_ACCESS_READ, FER_DTYPE_F32, FER_JIT_WILDCARD_U32,
                                static_cast<uint64_t>(H) * W * C * sizeof(float));
    descs[1] = make_buffer_desc(FER_JIT_ACCESS_READ, FER_DTYPE_F32, FER_JIT_WILDCARD_U32,
                                static_cast<uint64_t>(F) * C * KH * KW * sizeof(float));
    descs[2] = has_bias
        ? make_buffer_desc(FER_JIT_ACCESS_READ, FER_DTYPE_F32, FER_JIT_WILDCARD_U32,
                           static_cast<uint64_t>(F) * sizeof(float))
        : make_buffer_desc(FER_JIT_ACCESS_READ, FER_DTYPE_F32, FER_JIT_WILDCARD_U32, FER_JIT_WILDCARD_U64);
    descs[3] = make_buffer_desc(FER_JIT_ACCESS_WRITE, FER_DTYPE_F32, FER_JIT_WILDCARD_U32,
                                static_cast<uint64_t>(out_h) * out_w * F * sizeof(float));
    for (int i = 4; i < 14; ++i) descs[i] = make_wildcard_scalar(FER_JIT_ARG_SCALAR_U32);

    fer_jit_arg_value_t args[14]{};
    args[0].kind = FER_JIT_ARG_BUFFER;  args[0].as.buffer_id = bindings->input;
    args[1].kind = FER_JIT_ARG_BUFFER;  args[1].as.buffer_id = bindings->weights;
    args[2].kind = FER_JIT_ARG_BUFFER;  args[2].as.buffer_id = has_bias ? bindings->bias : bindings->weights;
    args[3].kind = FER_JIT_ARG_BUFFER;  args[3].as.buffer_id = bindings->output;
    args[4].kind  = FER_JIT_ARG_SCALAR_U32; args[4].as.u32  = H;
    args[5].kind  = FER_JIT_ARG_SCALAR_U32; args[5].as.u32  = W;
    args[6].kind  = FER_JIT_ARG_SCALAR_U32; args[6].as.u32  = C;
    args[7].kind  = FER_JIT_ARG_SCALAR_U32; args[7].as.u32  = KH;
    args[8].kind  = FER_JIT_ARG_SCALAR_U32; args[8].as.u32  = KW;
    args[9].kind  = FER_JIT_ARG_SCALAR_U32; args[9].as.u32  = F;
    args[10].kind = FER_JIT_ARG_SCALAR_U32; args[10].as.u32 = intent->pad_h;
    args[11].kind = FER_JIT_ARG_SCALAR_U32; args[11].as.u32 = intent->pad_w;
    args[12].kind = FER_JIT_ARG_SCALAR_U32; args[12].as.u32 = sh;
    args[13].kind = FER_JIT_ARG_SCALAR_U32; args[13].as.u32 = sw;

    const uint32_t block_x = 256;
    fer_jit_launch_cfg_t launch{};
    launch.grid_x = (total_out + block_x - 1) / block_x;
    launch.grid_y = 1; launch.grid_z = 1;
    launch.block_x = block_x; launch.block_y = 1; launch.block_z = 1;
    launch.memory_regime = (intent->memory_regime == FER_MEMORY_AUTO)
                               ? FER_MEMORY_CUSTOM_POOL : intent->memory_regime;

    return compile_and_launch(session, source_text, "fer_intent_conv2d_f32",
                              descs, 14, args, 14, &launch, out_job_id);
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

    switch (intent->op) {
        case FER_JIT_INTENT_OP_AFFINE_F32:
            return run_affine_f32(session, intent, bindings, out_job_id);
        case FER_JIT_INTENT_OP_SOFTMAX_F32:
            return run_softmax_f32(session, intent, bindings, out_job_id);
        case FER_JIT_INTENT_OP_REDUCE_SUM_F32:
            return run_reduce_f32(session, intent, bindings, out_job_id, true);
        case FER_JIT_INTENT_OP_REDUCE_MAX_F32:
            return run_reduce_f32(session, intent, bindings, out_job_id, false);
        case FER_JIT_INTENT_OP_CONV2D_F32:
            return run_conv2d_f32(session, intent, bindings, out_job_id);
        default:
            return invalid("unsupported intent op");
    }
}
