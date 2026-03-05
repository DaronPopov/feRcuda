#pragma once

#include "fercuda/api/c_api.h"
#include "fercuda/jit/script_host.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fer_script_runtime fer_script_runtime_t;

typedef struct fer_script_run_options {
    uint64_t timeout_ms;
} fer_script_run_options_t;

fer_status_t fer_script_runtime_create(
    fer_session_t* session,
    fer_script_runtime_t** out_runtime);

fer_status_t fer_script_runtime_destroy(
    fer_script_runtime_t* runtime);

fer_status_t fer_script_load(
    fer_script_runtime_t* runtime,
    const char* module_path);

fer_status_t fer_script_unload(
    fer_script_runtime_t* runtime);

fer_status_t fer_script_is_loaded(
    fer_script_runtime_t* runtime,
    uint8_t* out_loaded);

fer_status_t fer_script_run(
    fer_script_runtime_t* runtime,
    const char* args_json,
    char* err_buf,
    size_t err_buf_len);

fer_status_t fer_script_run_with_options(
    fer_script_runtime_t* runtime,
    const fer_script_run_options_t* options,
    const char* args_json,
    char* err_buf,
    size_t err_buf_len);

fer_status_t fer_script_cancel(
    fer_script_runtime_t* runtime);

#ifdef __cplusplus
}
#endif
