#pragma once

#include "fercuda/api/c_api.h"
#include "fercuda/jit/intent/intent.h"
#include "fercuda/jit/types.h"

#ifdef __cplusplus
extern "C" {
#endif

fer_status_t fer_jit_compile(
    fer_session_t* session,
    const fer_jit_source_t* source,
    const fer_jit_options_t* options,
    fer_jit_program_t* out_program,
    fer_jit_compile_result_t* out_result);

fer_status_t fer_jit_release_program(
    fer_session_t* session,
    fer_jit_program_t program);

fer_status_t fer_jit_get_kernel(
    fer_session_t* session,
    fer_jit_program_t program,
    const char* kernel_name,
    const fer_jit_kernel_sig_t* signature,
    fer_jit_kernel_t* out_kernel);

fer_status_t fer_jit_release_kernel(
    fer_session_t* session,
    fer_jit_kernel_t kernel);

fer_status_t fer_jit_launch(
    fer_session_t* session,
    fer_jit_kernel_t kernel,
    const fer_jit_launch_cfg_t* cfg,
    const fer_jit_arg_pack_t* args,
    uint64_t* out_job_id);

fer_status_t fer_jit_cache_clear(
    fer_session_t* session);

fer_status_t fer_jit_get_stats(
    fer_session_t* session,
    fer_jit_stats_t* out_stats);

#ifdef __cplusplus
}
#endif
