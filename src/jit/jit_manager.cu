#include "fercuda/jit/manager.cuh"

#include "fercuda/api/c_api.h"
#include "fercuda/jit/lowering/pipeline.h"
#include "fercuda/runtime/session.cuh"

#include <cuda.h>
#include <nvrtc.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

namespace {

const char* kBackendNvrtc = "nvrtc";
const char* kDiskCacheVersion = "v1";

const char* cu_err(CUresult r) {
    const char* out = nullptr;
    cuGetErrorString(r, &out);
    return out ? out : "unknown CUresult";
}

std::string nvrtc_log(nvrtcProgram prog) {
    size_t sz = 0;
    nvrtcGetProgramLogSize(prog, &sz);
    if (sz == 0) return {};
    std::string log(sz, '\0');
    nvrtcGetProgramLog(prog, &log[0]);
    return log;
}

std::vector<std::string> split_opts(const char* opts) {
    std::vector<std::string> out;
    if (!opts) return out;
    std::string s(opts);
    std::string cur;
    for (char c : s) {
        if (c == ' ' || c == '\t' || c == '\n') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
            continue;
        }
        cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::string compute_arch_opt(const fer_jit_options_t* options) {
    if (options && options->arch && options->arch[0] != '\0') {
        return std::string("--gpu-architecture=") + options->arch;
    }
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        std::ostringstream oss;
        oss << "--gpu-architecture=compute_" << prop.major << prop.minor;
        return oss.str();
    }
    return "--gpu-architecture=compute_52";
}

uint64_t hash_u64(const void* data, size_t len) {
    const auto* p = static_cast<const unsigned char*>(data);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ull;
    }
    return h;
}

std::string make_key(const fer_jit_source_t* source, const fer_jit_options_t* options) {
    std::ostringstream oss;
    oss << "k" << source->kind << ":h" << hash_u64(source->code, source->code_len);
    if (options) {
        oss << ":b" << options->backend << ":m" << options->mode;
        if (options->arch) oss << ":a" << options->arch;
        if (options->extra_nvrtc_opts) oss << ":o" << options->extra_nvrtc_opts;
    }
    return oss.str();
}

std::string join_notes(const std::vector<std::string>& notes) {
    if (notes.empty()) return {};
    std::ostringstream oss;
    for (size_t i = 0; i < notes.size(); ++i) {
        if (i) oss << "; ";
        oss << notes[i];
    }
    return oss.str();
}

std::string key_to_hex(const std::string& key) {
    std::ostringstream oss;
    oss << std::hex << hash_u64(key.data(), key.size());
    return oss.str();
}

bool ensure_dir(const char* dir) {
    if (!dir || dir[0] == '\0') return false;
    if (mkdir(dir, 0755) == 0) return true;
    return errno == EEXIST;
}

std::string cache_ptx_path(const fer_jit_options_t* options, const std::string& key) {
    if (!options || !options->enable_disk_cache || !options->cache_dir || options->cache_dir[0] == '\0') {
        return {};
    }
    return std::string(options->cache_dir) + "/" + kDiskCacheVersion + "_" + key_to_hex(key) + ".ptx";
}

bool read_file_string(const std::string& path, std::string* out) {
    if (!out) return false;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        return false;
    }
    long sz = std::ftell(f);
    if (sz < 0) {
        std::fclose(f);
        return false;
    }
    std::rewind(f);
    out->assign(static_cast<size_t>(sz), '\0');
    if (sz > 0) {
        const size_t n = std::fread(&(*out)[0], 1, static_cast<size_t>(sz), f);
        if (n != static_cast<size_t>(sz)) {
            std::fclose(f);
            return false;
        }
    }
    std::fclose(f);
    return true;
}

bool write_file_string(const std::string& path, const std::string& data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const size_t n = std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
    return n == data.size();
}

uint32_t to_fer_dtype(fer::runtime::BufferDType dt) {
    using fer::runtime::BufferDType;
    switch (dt) {
        case BufferDType::F32: return FER_DTYPE_F32;
        case BufferDType::F16: return FER_DTYPE_F16;
        case BufferDType::BF16: return FER_DTYPE_BF16;
        case BufferDType::I8: return FER_DTYPE_I8;
        case BufferDType::U8: return FER_DTYPE_U8;
        case BufferDType::I16: return FER_DTYPE_I16;
        case BufferDType::U16: return FER_DTYPE_U16;
        case BufferDType::I32: return FER_DTYPE_I32;
        case BufferDType::U32: return FER_DTYPE_U32;
        case BufferDType::I64: return FER_DTYPE_I64;
        case BufferDType::U64: return FER_DTYPE_U64;
        case BufferDType::F64: return FER_DTYPE_F64;
        default: return FER_JIT_WILDCARD_U32;
    }
}

} // namespace

struct fer_jit_program_impl {
    CUmodule module = nullptr;
    std::string key;
    bool strict_mode = false;
};

struct fer_jit_program {
    std::shared_ptr<fer_jit_program_impl> impl;
};

struct fer_jit_kernel {
    std::shared_ptr<fer_jit_program_impl> program;
    CUfunction fn = nullptr;
    std::vector<fer_jit_arg_desc_t> signature;
};

namespace fer::jit {

struct JitManager::Impl {
    std::mutex mu;
    std::unordered_map<std::string, std::shared_ptr<fer_jit_program_impl>> cache;
    fer_jit_stats_t stats{};
    mutable std::string last_log;
    bool cu_inited = false;
};

JitManager::JitManager() : impl_(std::make_unique<Impl>()) {}
JitManager::~JitManager() = default;

Status JitManager::compile(
    runtime::RuntimeSession*,
    const fer_jit_source_t* source,
    const fer_jit_options_t* options,
    fer_jit_program_t* out_program,
    fer_jit_compile_result_t* out_result) {
    if (!source) return Status::invalid_argument("source is null");
    if (!source->code || source->code_len == 0) return Status::invalid_argument("source code is empty");
    if (!out_program) return Status::invalid_argument("out_program is null");
    *out_program = nullptr;

    if (!impl_->cu_inited) {
        CUresult cr = cuInit(0);
        if (cr != CUDA_SUCCESS) return Status::internal_error("cuInit failed");
        impl_->cu_inited = true;
    }

    lowering::KernelModule lowered{};
    lowering::DeviceCapabilities caps{};
    Status lower_st = lowering::LoweringPipeline::run(source, options, &lowered, &caps);
    if (!lower_st.ok()) return lower_st;

    fer_jit_source_t lowered_source{};
    lowered_source.kind = source->kind;
    lowered_source.code = lowered.source.c_str();
    lowered_source.code_len = lowered.source.size();

    const std::string key = make_key(&lowered_source, options);
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        auto hit = impl_->cache.find(key);
        if (hit != impl_->cache.end()) {
            auto* handle = new fer_jit_program{};
            handle->impl = hit->second;
            *out_program = handle;
            if (out_result) {
                out_result->cache_hit = 1;
                out_result->backend_name = kBackendNvrtc;
                out_result->log = "";
            }
            impl_->stats.cache_hit_count++;
            return Status::ok_status();
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    CUmodule module = nullptr;
    std::string build_log;
    std::string ptx_for_disk_cache;
    const std::string disk_cache_path = cache_ptx_path(options, key);

    if (!disk_cache_path.empty()) {
        if (ensure_dir(options->cache_dir)) {
            std::string cached_ptx;
            if (read_file_string(disk_cache_path, &cached_ptx) && !cached_ptx.empty()) {
                CUresult lr = cuModuleLoadDataEx(&module, cached_ptx.c_str(), 0, nullptr, nullptr);
                if (lr == CUDA_SUCCESS) {
                    auto prog_impl = std::shared_ptr<fer_jit_program_impl>(
                        new fer_jit_program_impl{},
                        [](fer_jit_program_impl* p) {
                            if (p) {
                                if (p->module) cuModuleUnload(p->module);
                                delete p;
                            }
                        });
                    prog_impl->module = module;
                    prog_impl->key = key;
                    prog_impl->strict_mode = options && options->mode == FER_JIT_MODE_STRICT;
                    {
                        std::lock_guard<std::mutex> lock(impl_->mu);
                        impl_->cache[key] = prog_impl;
                        impl_->stats.cache_hit_count++;
                    }
                    auto* handle = new fer_jit_program{};
                    handle->impl = prog_impl;
                    *out_program = handle;
                    if (out_result) {
                        out_result->cache_hit = 1;
                        out_result->backend_name = kBackendNvrtc;
                        out_result->log = "";
                    }
                    return Status::ok_status();
                }
            }
        }
    }

    if (lowered_source.kind == FER_JIT_SOURCE_PTX) {
        CUresult lr = cuModuleLoadDataEx(&module, lowered_source.code, 0, nullptr, nullptr);
        if (lr != CUDA_SUCCESS) {
            {
                std::lock_guard<std::mutex> lock(impl_->mu);
                impl_->last_log = std::string("cuModuleLoadDataEx failed: ") + cu_err(lr);
            }
            if (out_result) {
                out_result->cache_hit = 0;
                out_result->backend_name = kBackendNvrtc;
                out_result->log = impl_->last_log.c_str();
            }
            return Status::internal_error("PTX module load failed");
        }
    } else {
        nvrtcProgram prog{};
        nvrtcResult nr = nvrtcCreateProgram(&prog, lowered_source.code, "fer_jit_kernel.cu", 0, nullptr, nullptr);
        if (nr != NVRTC_SUCCESS) return Status::internal_error("nvrtcCreateProgram failed");

        std::vector<std::string> opt_storage;
        opt_storage.emplace_back("--std=c++17");
        opt_storage.emplace_back(compute_arch_opt(options));
        for (const auto& opt : lowered.extra_nvrtc_opts) {
            if (!opt.empty()) opt_storage.push_back(opt);
        }
        auto extra = split_opts(options ? options->extra_nvrtc_opts : nullptr);
        opt_storage.insert(opt_storage.end(), extra.begin(), extra.end());
        std::vector<const char*> opt_ptrs;
        opt_ptrs.reserve(opt_storage.size());
        for (const auto& o : opt_storage) opt_ptrs.push_back(o.c_str());

        nr = nvrtcCompileProgram(prog, static_cast<int>(opt_ptrs.size()), opt_ptrs.data());
        build_log = nvrtc_log(prog);
        if (nr != NVRTC_SUCCESS) {
            {
                std::lock_guard<std::mutex> lock(impl_->mu);
                impl_->last_log = build_log;
            }
            if (out_result) {
                out_result->cache_hit = 0;
                out_result->backend_name = kBackendNvrtc;
                out_result->log = impl_->last_log.c_str();
            }
            nvrtcDestroyProgram(&prog);
            return Status::internal_error("NVRTC compile failed");
        }

        size_t ptx_sz = 0;
        nvrtcGetPTXSize(prog, &ptx_sz);
        std::string ptx(ptx_sz, '\0');
        nvrtcGetPTX(prog, &ptx[0]);
        ptx_for_disk_cache = ptx;
        nvrtcDestroyProgram(&prog);

        CUresult lr = cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr);
        if (lr != CUDA_SUCCESS) {
            {
                std::lock_guard<std::mutex> lock(impl_->mu);
                impl_->last_log = std::string("cuModuleLoadDataEx failed: ") + cu_err(lr);
            }
            if (out_result) {
                out_result->cache_hit = 0;
                out_result->backend_name = kBackendNvrtc;
                out_result->log = impl_->last_log.c_str();
            }
            return Status::internal_error("PTX module load failed");
        }
    }

    if (!disk_cache_path.empty() && !ptx_for_disk_cache.empty()) {
        (void)write_file_string(disk_cache_path, ptx_for_disk_cache);
    }

    auto prog_impl = std::shared_ptr<fer_jit_program_impl>(
        new fer_jit_program_impl{},
        [](fer_jit_program_impl* p) {
            if (p) {
                if (p->module) cuModuleUnload(p->module);
                delete p;
            }
        });
    prog_impl->module = module;
    prog_impl->key = key;
    prog_impl->strict_mode = options && options->mode == FER_JIT_MODE_STRICT;
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        impl_->cache[key] = prog_impl;
    }

    auto* handle = new fer_jit_program{};
    handle->impl = prog_impl;
    *out_program = handle;

    const auto dt_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        impl_->stats.compile_count++;
        impl_->stats.compile_time_us += static_cast<uint64_t>(dt_us);
        const std::string notes = join_notes(lowered.notes);
        if (notes.empty()) {
            impl_->last_log = build_log;
        } else if (build_log.empty()) {
            impl_->last_log = std::string("[lowering] ") + notes;
        } else {
            impl_->last_log = std::string("[lowering] ") + notes + "\n" + build_log;
        }
    }
    if (out_result) {
        out_result->cache_hit = 0;
        out_result->backend_name = kBackendNvrtc;
        out_result->log = impl_->last_log.c_str();
    }
    return Status::ok_status();
}

Status JitManager::release_program(fer_jit_program_t program) {
    if (!program) return Status::ok_status();
    delete program;
    return Status::ok_status();
}

Status JitManager::get_kernel(
    fer_jit_program_t program,
    const char* kernel_name,
    const fer_jit_kernel_sig_t* signature,
    fer_jit_kernel_t* out_kernel) {
    if (!program || !program->impl) return Status::invalid_argument("program is null");
    if (!kernel_name || kernel_name[0] == '\0') return Status::invalid_argument("kernel_name is empty");
    if (!out_kernel) return Status::invalid_argument("out_kernel is null");
    if (program->impl->strict_mode && (!signature || !signature->args || signature->arg_count == 0)) {
        return Status::invalid_argument("strict mode requires kernel signature");
    }
    *out_kernel = nullptr;

    CUfunction fn = nullptr;
    CUresult gr = cuModuleGetFunction(&fn, program->impl->module, kernel_name);
    if (gr != CUDA_SUCCESS) return Status::not_found("kernel symbol not found");

    auto* k = new fer_jit_kernel{};
    k->program = program->impl;
    k->fn = fn;
    if (signature && signature->args && signature->arg_count > 0) {
        k->signature.assign(signature->args, signature->args + signature->arg_count);
    }
    *out_kernel = k;
    return Status::ok_status();
}

Status JitManager::release_kernel(fer_jit_kernel_t kernel) {
    if (!kernel) return Status::ok_status();
    delete kernel;
    return Status::ok_status();
}

Status JitManager::launch(
    runtime::RuntimeSession* session,
    fer_jit_kernel_t kernel,
    const fer_jit_launch_cfg_t* cfg,
    const fer_jit_arg_pack_t* args,
    uint64_t* out_job_id) {
    if (!session) return Status::invalid_argument("session is null");
    if (!kernel) return Status::invalid_argument("kernel is null");
    if (!cfg) return Status::invalid_argument("cfg is null");
    if (!args) return Status::invalid_argument("args is null");
    if (!out_job_id) return Status::invalid_argument("out_job_id is null");
    if (args->arg_count > 0 && !args->args) return Status::invalid_argument("args array is null");
    if (cfg->grid_x == 0 || cfg->block_x == 0) return Status::invalid_argument("invalid launch dimensions");
    if (cfg->block_y == 0 || cfg->block_z == 0) return Status::invalid_argument("invalid block dimensions");
    const uint64_t threads =
        static_cast<uint64_t>(cfg->block_x) * cfg->block_y * cfg->block_z;
    if (threads == 0 || threads > 1024) return Status::invalid_argument("threads per block out of range");
    if (!kernel->signature.empty() && kernel->signature.size() != args->arg_count) {
        return Status::invalid_argument("arg_count mismatch");
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint64_t> arg_storage(args->arg_count, 0);
    std::vector<void*> param_ptrs(args->arg_count, nullptr);

    for (size_t i = 0; i < args->arg_count; ++i) {
        const fer_jit_arg_value_t& av = args->args[i];
        const fer_jit_arg_desc_t* desc = (i < kernel->signature.size()) ? &kernel->signature[i] : nullptr;
        if (desc && desc->kind != av.kind) return Status::invalid_argument("arg kind mismatch");

        switch (av.kind) {
            case FER_JIT_ARG_BUFFER: {
                void* ptr = nullptr;
                fer::runtime::BufferDesc bdesc{};
                uint32_t regime = 0;
                size_t actual_bytes = 0;
                Status rs = session->resolve_buffer(av.as.buffer_id, &ptr, &bdesc, &actual_bytes, &regime);
                if (!rs.ok()) return rs;
                if (desc) {
                    if ((desc->access == FER_JIT_ACCESS_WRITE || desc->access == FER_JIT_ACCESS_READ_WRITE) &&
                        bdesc.immutable) {
                        return Status::invalid_argument("immutable buffer cannot be used for write access");
                    }
                    const uint32_t actual_dtype = to_fer_dtype(bdesc.dtype);
                    if (desc->expected_dtype != FER_JIT_WILDCARD_U32 &&
                        actual_dtype != desc->expected_dtype) {
                        return Status::invalid_argument("buffer dtype contract mismatch");
                    }
                    if (desc->expected_rank != FER_JIT_WILDCARD_U32 &&
                        bdesc.rank != desc->expected_rank) {
                        return Status::invalid_argument("buffer rank contract mismatch");
                    }
                    if (desc->expected_rank != FER_JIT_WILDCARD_U32 && bdesc.rank <= 4) {
                        for (uint32_t d = 0; d < bdesc.rank && d < 4; ++d) {
                            if (desc->expected_dims[d] != FER_JIT_WILDCARD_U32 &&
                                bdesc.dims[d] != desc->expected_dims[d]) {
                                return Status::invalid_argument("buffer shape contract mismatch");
                            }
                        }
                    }
                    if (desc->expected_bytes != FER_JIT_WILDCARD_U64) {
                        if (actual_bytes != desc->expected_bytes) {
                            return Status::invalid_argument("buffer byte-size contract mismatch");
                        }
                    }
                }
                if (cfg->memory_regime != 0xFFFFFFFFu && regime != cfg->memory_regime) {
                    return Status::invalid_argument("buffer regime does not match launch regime");
                }
                arg_storage[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
                break;
            }
            case FER_JIT_ARG_SCALAR_I32: {
                int32_t v = av.as.i32;
                std::memcpy(&arg_storage[i], &v, sizeof(v));
                break;
            }
            case FER_JIT_ARG_SCALAR_U32: {
                uint32_t v = av.as.u32;
                std::memcpy(&arg_storage[i], &v, sizeof(v));
                break;
            }
            case FER_JIT_ARG_SCALAR_I64: {
                int64_t v = av.as.i64;
                std::memcpy(&arg_storage[i], &v, sizeof(v));
                break;
            }
            case FER_JIT_ARG_SCALAR_U64: {
                uint64_t v = av.as.u64;
                std::memcpy(&arg_storage[i], &v, sizeof(v));
                break;
            }
            case FER_JIT_ARG_SCALAR_F32: {
                float v = av.as.f32;
                std::memcpy(&arg_storage[i], &v, sizeof(v));
                break;
            }
            case FER_JIT_ARG_SCALAR_F64: {
                double v = av.as.f64;
                std::memcpy(&arg_storage[i], &v, sizeof(v));
                break;
            }
            default:
                return Status::invalid_argument("unsupported arg kind");
        }
        param_ptrs[i] = &arg_storage[i];
    }

    cudaStream_t stream = session->stream();
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);
    CUresult lr = cuLaunchKernel(
        kernel->fn,
        cfg->grid_x, cfg->grid_y, cfg->grid_z,
        cfg->block_x, cfg->block_y, cfg->block_z,
        cfg->shared_mem_bytes,
        cu_stream,
        param_ptrs.empty() ? nullptr : param_ptrs.data(),
        nullptr);
    if (lr != CUDA_SUCCESS) return Status::internal_error("cuLaunchKernel failed");

    Status js = session->create_external_job(stream, out_job_id);
    if (!js.ok()) return js;

    const auto dt_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        impl_->stats.launch_count++;
        impl_->stats.launch_time_us += static_cast<uint64_t>(dt_us);
    }
    return Status::ok_status();
}

Status JitManager::cache_clear() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->cache.clear();
    return Status::ok_status();
}

Status JitManager::get_stats(fer_jit_stats_t* out_stats) const {
    if (!out_stats) return Status::invalid_argument("out_stats is null");
    std::lock_guard<std::mutex> lock(impl_->mu);
    *out_stats = impl_->stats;
    return Status::ok_status();
}

} // namespace fer::jit
