#pragma once

#include <stdint.h>

#include "fercuda/api/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FER_JIT_INTENT_ABI_VERSION 1u

typedef enum fer_jit_intent_op {
    FER_JIT_INTENT_OP_AFFINE_F32 = 0,
    FER_JIT_INTENT_OP_SOFTMAX_F32 = 1,
    FER_JIT_INTENT_OP_REDUCE_SUM_F32 = 2,
    FER_JIT_INTENT_OP_REDUCE_MAX_F32 = 3,
    FER_JIT_INTENT_OP_CONV2D_F32 = 4
} fer_jit_intent_op_t;

typedef enum fer_jit_intent_fusion {
    FER_JIT_INTENT_FUSION_NONE = 0,
    FER_JIT_INTENT_FUSION_RELU = 1u << 0
} fer_jit_intent_fusion_t;

typedef enum fer_jit_intent_caps {
    FER_JIT_INTENT_CAPS_NONE = 0,
    FER_JIT_INTENT_CAPS_REQUIRE_TENSOR_CORES = 1u << 0,
    FER_JIT_INTENT_CAPS_REQUIRE_COOP_GROUPS = 1u << 1
} fer_jit_intent_caps_t;

typedef struct fer_jit_intent {
    uint32_t abi_version;
    uint32_t op;
    uint32_t fusion_mask;
    uint32_t caps_mask;
    uint32_t memory_regime;
    uint32_t n;
    float alpha;
    float beta;
    // Extended fields for conv2d / multi-dimensional ops
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t num_filters;
} fer_jit_intent_t;

typedef struct fer_jit_intent_bindings {
    uint64_t input;
    uint64_t output;
    uint64_t weights;    // used by conv2d
    uint64_t bias;       // used by conv2d (optional, 0 = unused)
} fer_jit_intent_bindings_t;

fer_status_t fer_jit_run_intent(
    fer_session_t* session,
    const fer_jit_intent_t* intent,
    const fer_jit_intent_bindings_t* bindings,
    uint64_t* out_job_id);

#ifdef __cplusplus
}
#endif
