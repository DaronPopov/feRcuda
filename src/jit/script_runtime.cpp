#include "fercuda/jit/script_runtime.h"

#include "fercuda/jit/api.h"

#include <dlfcn.h>

#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <string>
#include <condition_variable>
#include <atomic>
#include <chrono>

struct fer_script_runtime {
    std::mutex mu;
    std::condition_variable cv;
    fer_session_t* session = nullptr;
    void* module_handle = nullptr;
    fer_script_main_fn main_fn = nullptr;
    std::string module_path;
    std::string last_error;
    bool running = false;
    std::atomic<bool> cancel_requested{false};
    std::atomic<uint64_t> deadline_ms{0};
};

namespace {

fer_status_t status_ok() { return fer_status_t{FER_STATUS_OK, "ok"}; }
fer_status_t status_invalid(const char* msg) { return fer_status_t{FER_STATUS_INVALID_ARGUMENT, msg}; }
fer_status_t status_internal(const char* msg) { return fer_status_t{FER_STATUS_INTERNAL_ERROR, msg}; }

void copy_err(char* err_buf, size_t err_buf_len, const char* msg) {
    if (!err_buf || err_buf_len == 0) return;
    if (!msg) {
        err_buf[0] = '\0';
        return;
    }
    const size_t n = std::strlen(msg);
    const size_t m = (n < (err_buf_len - 1)) ? n : (err_buf_len - 1);
    std::memcpy(err_buf, msg, m);
    err_buf[m] = '\0';
}

uint64_t now_ms() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

uint8_t host_script_should_cancel(const fer_script_context_t* ctx) {
    if (!ctx) return 0u;
    auto* rt = static_cast<fer_script_runtime_t*>(ctx->user_data);
    if (!rt) return 0u;
    if (rt->cancel_requested.load(std::memory_order_relaxed)) return 1u;
    const uint64_t deadline = rt->deadline_ms.load(std::memory_order_relaxed);
    if (deadline == 0) return 0u;
    return (now_ms() >= deadline) ? 1u : 0u;
}

const fer_script_host_api_t kHostApi = {
    FER_SCRIPT_HOST_ABI_VERSION,
    &fer_session_create,
    &fer_session_destroy,
    &fer_alloc_buffer,
    &fer_alloc_buffer_in_regime,
    &fer_import_external_buffer,
    &fer_import_external_buffer_with_deleter,
    &fer_export_buffer_device_ptr,
    &fer_free_buffer,
    &fer_upload_bytes,
    &fer_download_bytes,
    &fer_jit_compile,
    &fer_jit_release_program,
    &fer_jit_get_kernel,
    &fer_jit_release_kernel,
    &fer_jit_launch,
    &fer_jit_run_intent,
    &fer_job_wait,
    &fer_set_session_stream,
    &fer_get_session_stream,
    &fer_tensor_create,
    &fer_tensor_attach_external,
    &fer_tensor_device_ptr,
    &fer_tensor_release,
    &fer_tensor_upload,
    &fer_tensor_download,
    &fer_tensor_run_affine_f32,
    &host_script_should_cancel,
    &now_ms,
};

} // namespace

extern "C" fer_status_t fer_script_runtime_create(
    fer_session_t* session,
    fer_script_runtime_t** out_runtime) {
    if (!session) return status_invalid("session is null");
    if (!out_runtime) return status_invalid("out_runtime is null");
    *out_runtime = nullptr;
    try {
        auto* rt = new fer_script_runtime{};
        rt->session = session;
        *out_runtime = rt;
        return status_ok();
    } catch (const std::bad_alloc&) {
        return status_internal("allocation failure");
    } catch (...) {
        return status_internal("script runtime create failure");
    }
}

extern "C" fer_status_t fer_script_runtime_destroy(
    fer_script_runtime_t* runtime) {
    if (!runtime) return status_ok();
    {
        std::unique_lock<std::mutex> lock(runtime->mu);
        while (runtime->running) {
            runtime->cv.wait(lock);
        }
        if (runtime->module_handle) {
            dlclose(runtime->module_handle);
            runtime->module_handle = nullptr;
            runtime->main_fn = nullptr;
        }
    }
    delete runtime;
    return status_ok();
}

extern "C" fer_status_t fer_script_load(
    fer_script_runtime_t* runtime,
    const char* module_path) {
    if (!runtime) return status_invalid("runtime is null");
    if (!module_path || module_path[0] == '\0') return status_invalid("module_path is empty");
    std::lock_guard<std::mutex> lock(runtime->mu);
    if (runtime->module_handle) return status_invalid("script already loaded");
    if (runtime->running) return status_invalid("script is running");

    dlerror();
    void* h = dlopen(module_path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        const char* err = dlerror();
        runtime->last_error = err ? err : "dlopen failed";
        return status_internal("script load failed");
    }

    dlerror();
    auto* main_fn = reinterpret_cast<fer_script_main_fn>(dlsym(h, "fer_script_main"));
    const char* sym_err = dlerror();
    if (sym_err != nullptr || !main_fn) {
        runtime->last_error = sym_err ? sym_err : "fer_script_main not found";
        dlclose(h);
        return status_invalid("fer_script_main symbol not found");
    }

    dlerror();
    auto* ver_fn = reinterpret_cast<fer_script_abi_version_fn>(dlsym(h, "fer_script_abi_version"));
    (void)dlerror();
    if (ver_fn) {
        const uint32_t script_abi = ver_fn();
        if (script_abi != FER_SCRIPT_HOST_ABI_VERSION) {
            runtime->last_error = "script ABI mismatch";
            dlclose(h);
            return status_invalid("script ABI mismatch");
        }
    }

    runtime->module_handle = h;
    runtime->main_fn = main_fn;
    runtime->module_path = module_path;
    runtime->last_error.clear();
    return status_ok();
}

extern "C" fer_status_t fer_script_unload(
    fer_script_runtime_t* runtime) {
    if (!runtime) return status_invalid("runtime is null");
    std::lock_guard<std::mutex> lock(runtime->mu);
    if (runtime->running) return status_invalid("script is running");
    if (!runtime->module_handle) return status_ok();
    dlclose(runtime->module_handle);
    runtime->module_handle = nullptr;
    runtime->main_fn = nullptr;
    runtime->module_path.clear();
    runtime->last_error.clear();
    return status_ok();
}

extern "C" fer_status_t fer_script_is_loaded(
    fer_script_runtime_t* runtime,
    uint8_t* out_loaded) {
    if (!runtime) return status_invalid("runtime is null");
    if (!out_loaded) return status_invalid("out_loaded is null");
    std::lock_guard<std::mutex> lock(runtime->mu);
    *out_loaded = runtime->main_fn ? 1u : 0u;
    return status_ok();
}

extern "C" fer_status_t fer_script_run(
    fer_script_runtime_t* runtime,
    const char* args_json,
    char* err_buf,
    size_t err_buf_len) {
    return fer_script_run_with_options(runtime, nullptr, args_json, err_buf, err_buf_len);
}

extern "C" fer_status_t fer_script_run_with_options(
    fer_script_runtime_t* runtime,
    const fer_script_run_options_t* options,
    const char* args_json,
    char* err_buf,
    size_t err_buf_len) {
    if (!runtime) return status_invalid("runtime is null");
    fer_script_main_fn fn = nullptr;
    fer_session_t* session = nullptr;
    uint64_t timeout_ms = options ? options->timeout_ms : 0;
    {
        std::lock_guard<std::mutex> lock(runtime->mu);
        if (!runtime->main_fn || !runtime->module_handle) return status_invalid("no script loaded");
        if (runtime->running) return status_invalid("script already running");
        runtime->running = true;
        fn = runtime->main_fn;
        session = runtime->session;
        runtime->cancel_requested.store(false, std::memory_order_relaxed);
        runtime->deadline_ms.store(timeout_ms ? (now_ms() + timeout_ms) : 0, std::memory_order_relaxed);
    }
    copy_err(err_buf, err_buf_len, "");

    fer_script_context_t ctx{};
    ctx.abi_version = FER_SCRIPT_HOST_ABI_VERSION;
    ctx.session = session;
    ctx.api = &kHostApi;
    ctx.user_data = runtime;
    const char* payload = args_json ? args_json : "{}";

    try {
        fer_status_t st = fn(&ctx, payload, err_buf, err_buf_len);
        {
            std::lock_guard<std::mutex> lock(runtime->mu);
            runtime->running = false;
            runtime->cancel_requested.store(false, std::memory_order_relaxed);
            runtime->deadline_ms.store(0, std::memory_order_relaxed);
            runtime->cv.notify_all();
        }
        if (st.code != FER_STATUS_OK && err_buf && err_buf_len > 1 && err_buf[0] == '\0' && st.message) {
            copy_err(err_buf, err_buf_len, st.message);
        }
        return st;
    } catch (const std::exception& e) {
        {
            std::lock_guard<std::mutex> lock(runtime->mu);
            runtime->running = false;
            runtime->cancel_requested.store(false, std::memory_order_relaxed);
            runtime->deadline_ms.store(0, std::memory_order_relaxed);
            runtime->cv.notify_all();
        }
        copy_err(err_buf, err_buf_len, e.what());
        return status_internal("script execution exception");
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(runtime->mu);
            runtime->running = false;
            runtime->cancel_requested.store(false, std::memory_order_relaxed);
            runtime->deadline_ms.store(0, std::memory_order_relaxed);
            runtime->cv.notify_all();
        }
        copy_err(err_buf, err_buf_len, "unknown exception");
        return status_internal("script execution exception");
    }
}

extern "C" fer_status_t fer_script_cancel(
    fer_script_runtime_t* runtime) {
    if (!runtime) return status_invalid("runtime is null");
    runtime->cancel_requested.store(true, std::memory_order_relaxed);
    return status_ok();
}
