#include "fercuda/api/c_api.h"
#include <sys/types.h>
#include "gpu/gpu_hot_runtime.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

static constexpr uint32_t kGPUHotRegime = 3001u;

bool is_ok(fer_status_t s) { return s.code == FER_STATUS_OK; }

bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

void* gpu_hot_alloc_bridge(void* runtime, uint64_t bytes) {
    return gpu_hot_alloc(static_cast<GPUHotRuntime*>(runtime), static_cast<size_t>(bytes));
}

void gpu_hot_free_bridge(void* runtime, void* ptr) {
    gpu_hot_free(static_cast<GPUHotRuntime*>(runtime), ptr);
}

} // namespace

int main() {
    bool all = true;

    GPUHotRuntime* hot = gpu_hot_init(0, nullptr);
    bool t0 = (hot != nullptr);
    std::printf("[%s] gpu_hot_init\n", t0 ? "PASS" : "FAIL");
    all = all && t0;
    if (!hot) return 1;

    fer_session_t* sess = nullptr;
    fer_pool_config_t cfg{};
    cfg.mutable_bytes = 128ULL << 20;
    cfg.immutable_bytes = 128ULL << 20;
    cfg.cuda_reserve = 0;
    cfg.verbose = 0;
    cfg.memory_regime = FER_MEMORY_CUDA_MALLOC;
    fer_status_t st = fer_session_create(0, &cfg, &sess);
    bool t1 = is_ok(st) && sess != nullptr;
    std::printf("[%s] fer_session_create\n", t1 ? "PASS" : "FAIL");
    all = all && t1;
    if (!sess) {
        gpu_hot_shutdown(hot);
        return 1;
    }

    fer_gpu_hot_allocator_t gha{};
    gha.runtime = hot;
    gha.alloc = &gpu_hot_alloc_bridge;
    gha.free = &gpu_hot_free_bridge;
    bool t2 = is_ok(fer_register_gpu_hot_allocator(sess, kGPUHotRegime, &gha));
    std::printf("[%s] fer_register_gpu_hot_allocator\n", t2 ? "PASS" : "FAIL");
    all = all && t2;

    bool t3 = is_ok(fer_set_default_memory_regime(sess, kGPUHotRegime));
    std::printf("[%s] fer_set_default_memory_regime(gpu_hot)\n", t3 ? "PASS" : "FAIL");
    all = all && t3;

    fer_buffer_desc_t d2{};
    d2.dtype = FER_DTYPE_F32;
    d2.rank = 2;
    d2.dims[0] = 2;
    d2.dims[1] = 2;

    uint64_t a = 0, b = 0, out = 0;
    all = all && is_ok(fer_alloc_buffer(sess, &d2, &a));
    all = all && is_ok(fer_alloc_buffer(sess, &d2, &b));
    all = all && is_ok(fer_alloc_buffer(sess, &d2, &out));

    std::vector<float> h_a = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> h_b = {5.f, 6.f, 7.f, 8.f};
    std::vector<float> h_out(4, 0.f);
    all = all && is_ok(fer_upload_f32(sess, a, h_a.data(), h_a.size()));
    all = all && is_ok(fer_upload_f32(sess, b, h_b.data(), h_b.size()));

    fer_matmul_request_t mm{};
    mm.a = a;
    mm.b = b;
    mm.out = out;
    mm.memory_regime = FER_MEMORY_AUTO;
    uint64_t jid = 0;
    all = all && is_ok(fer_submit_matmul(sess, &mm, &jid));
    all = all && is_ok(fer_job_wait(sess, jid));
    all = all && is_ok(fer_download_f32(sess, out, h_out.data(), h_out.size()));

    const float ref[4] = {19.f, 22.f, 43.f, 50.f};
    bool t4 = true;
    for (int i = 0; i < 4; i++) {
        if (!nearly_equal(h_out[i], ref[i])) {
            t4 = false;
            break;
        }
    }
    std::printf("[%s] gpu_hot regime matmul output\n", t4 ? "PASS" : "FAIL");
    all = all && t4;

    all = all && is_ok(fer_free_buffer(sess, a));
    all = all && is_ok(fer_free_buffer(sess, b));
    all = all && is_ok(fer_free_buffer(sess, out));

    bool t5 = is_ok(fer_session_destroy(sess));
    std::printf("[%s] fer_session_destroy\n", t5 ? "PASS" : "FAIL");
    all = all && t5;

    gpu_hot_shutdown(hot);
    std::printf("[%s] gpu_hot_shutdown\n", "PASS");

    std::printf("\n%s\n", all ? "ALL GPU HOT RUNTIME REGIME TESTS PASSED" : "SOME GPU HOT RUNTIME REGIME TESTS FAILED");
    return all ? 0 : 1;
}
