#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fer_session fer_session_t;

typedef struct fer_jit_program* fer_jit_program_t;
typedef struct fer_jit_kernel* fer_jit_kernel_t;

typedef enum fer_jit_backend {
    FER_JIT_BACKEND_NVRTC = 0,
    FER_JIT_BACKEND_AUTO = 0xFFFFFFFFu
} fer_jit_backend_t;

typedef enum fer_jit_mode {
    FER_JIT_MODE_PERMISSIVE = 0,
    FER_JIT_MODE_STRICT = 1
} fer_jit_mode_t;

typedef enum fer_jit_source_kind {
    FER_JIT_SOURCE_CUDA = 0,
    FER_JIT_SOURCE_PTX = 1
} fer_jit_source_kind_t;

typedef enum fer_jit_arg_kind {
    FER_JIT_ARG_BUFFER = 0,
    FER_JIT_ARG_SCALAR_I32 = 1,
    FER_JIT_ARG_SCALAR_U32 = 2,
    FER_JIT_ARG_SCALAR_I64 = 3,
    FER_JIT_ARG_SCALAR_U64 = 4,
    FER_JIT_ARG_SCALAR_F32 = 5,
    FER_JIT_ARG_SCALAR_F64 = 6
} fer_jit_arg_kind_t;

typedef enum fer_jit_access {
    FER_JIT_ACCESS_READ = 0,
    FER_JIT_ACCESS_WRITE = 1,
    FER_JIT_ACCESS_READ_WRITE = 2
} fer_jit_access_t;

#define FER_JIT_WILDCARD_U32 0xFFFFFFFFu
#define FER_JIT_WILDCARD_U64 0xFFFFFFFFFFFFFFFFull

typedef struct fer_jit_source {
    uint32_t kind;
    const char* code;
    size_t code_len;
} fer_jit_source_t;

typedef struct fer_jit_options {
    uint32_t backend;
    uint32_t mode;
    const char* arch;
    const char* extra_nvrtc_opts;
    const char* cache_dir;
    uint8_t enable_disk_cache;
} fer_jit_options_t;

typedef struct fer_jit_compile_result {
    uint8_t cache_hit;
    const char* backend_name;
    const char* log;
} fer_jit_compile_result_t;

typedef struct fer_jit_arg_desc {
    uint32_t kind;
    uint32_t access;
    const char* name;
    uint32_t expected_dtype;
    uint32_t expected_rank;
    uint64_t expected_bytes;
    uint32_t expected_dims[4];
} fer_jit_arg_desc_t;

typedef struct fer_jit_kernel_sig {
    const fer_jit_arg_desc_t* args;
    size_t arg_count;
} fer_jit_kernel_sig_t;

typedef struct fer_jit_launch_cfg {
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    uint32_t block_x;
    uint32_t block_y;
    uint32_t block_z;
    uint32_t shared_mem_bytes;
    uint32_t memory_regime;
} fer_jit_launch_cfg_t;

typedef struct fer_jit_arg_value {
    uint32_t kind;
    union {
        uint64_t buffer_id;
        int32_t i32;
        uint32_t u32;
        int64_t i64;
        uint64_t u64;
        float f32;
        double f64;
    } as;
} fer_jit_arg_value_t;

typedef struct fer_jit_arg_pack {
    const fer_jit_arg_value_t* args;
    size_t arg_count;
} fer_jit_arg_pack_t;

typedef struct fer_jit_stats {
    uint64_t compile_count;
    uint64_t cache_hit_count;
    uint64_t launch_count;
    uint64_t compile_time_us;
    uint64_t launch_time_us;
    uint64_t disk_cache_entries;
    uint64_t disk_cache_bytes;
    uint64_t memory_cache_entries;
} fer_jit_stats_t;

#ifdef __cplusplus
}
#endif
