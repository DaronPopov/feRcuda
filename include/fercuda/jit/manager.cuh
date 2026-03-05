#pragma once

#include "fercuda/core/status.cuh"
#include "fercuda/jit/types.h"

#include <memory>

namespace fer::runtime {
class RuntimeSession;
}

namespace fer::jit {

class JitManager {
public:
    JitManager();
    ~JitManager();

    Status compile(
        runtime::RuntimeSession* session,
        const fer_jit_source_t* source,
        const fer_jit_options_t* options,
        fer_jit_program_t* out_program,
        fer_jit_compile_result_t* out_result);

    Status release_program(fer_jit_program_t program);

    Status get_kernel(
        fer_jit_program_t program,
        const char* kernel_name,
        const fer_jit_kernel_sig_t* signature,
        fer_jit_kernel_t* out_kernel);

    Status release_kernel(fer_jit_kernel_t kernel);

    Status launch(
        runtime::RuntimeSession* session,
        fer_jit_kernel_t kernel,
        const fer_jit_launch_cfg_t* cfg,
        const fer_jit_arg_pack_t* args,
        uint64_t* out_job_id);

    Status cache_clear();

    Status get_stats(fer_jit_stats_t* out_stats) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fer::jit
