#pragma once

#include <stdint.h>

#include "fercuda/api/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FER_JIT_INTENT_ABI_VERSION 1u

typedef enum fer_jit_intent_op {
    FER_JIT_INTENT_OP_AFFINE_F32 = 0
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
} fer_jit_intent_t;

typedef struct fer_jit_intent_bindings {
    uint64_t input;
    uint64_t output;
} fer_jit_intent_bindings_t;

fer_status_t fer_jit_run_intent(
    fer_session_t* session,
    const fer_jit_intent_t* intent,
    const fer_jit_intent_bindings_t* bindings,
    uint64_t* out_job_id);

#ifdef __cplusplus
}
#endif
