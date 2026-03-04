#include "fercuda/api/c_api.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct AllocStats {
    int allocs = 0;
    int frees = 0;
};

static bool is_ok(fer_status_t s) { return s.code == FER_STATUS_OK; }

void* cb_gpu_hot_alloc(void* runtime, uint64_t bytes) {
    auto* stats = static_cast<AllocStats*>(runtime);
    void* p = nullptr;
    if (cudaMalloc(&p, static_cast<size_t>(bytes)) != cudaSuccess) return nullptr;
    stats->allocs++;
    return p;
}

void cb_gpu_hot_free(void* runtime, void* ptr) {
    auto* stats = static_cast<AllocStats*>(runtime);
    cudaFree(ptr);
    stats->frees++;
}

bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

} // namespace

int main() {
    constexpr uint32_t kCustomRegime = 2001u;
    bool all = true;

    fer_session_t* sess = nullptr;
    fer_status_t st = fer_session_create(0, nullptr, &sess);
    bool t0 = is_ok(st) && sess != nullptr;
    std::printf("[%s] fer_session_create\n", t0 ? "PASS" : "FAIL");
    all = all && t0;

    AllocStats stats{};
    fer_gpu_hot_allocator_t a{};
    a.runtime = &stats;
    a.alloc = &cb_gpu_hot_alloc;
    a.free = &cb_gpu_hot_free;
    bool t1 = is_ok(fer_register_gpu_hot_allocator(sess, kCustomRegime, &a));
    std::printf("[%s] fer_register_gpu_hot_allocator\n", t1 ? "PASS" : "FAIL");
    all = all && t1;

    bool t2 = is_ok(fer_set_default_memory_regime(sess, kCustomRegime));
    std::printf("[%s] fer_set_default_memory_regime(custom)\n", t2 ? "PASS" : "FAIL");
    all = all && t2;

    uint32_t def_regime = 0;
    bool t2b = is_ok(fer_get_default_memory_regime(sess, &def_regime)) && def_regime == kCustomRegime;
    std::printf("[%s] fer_get_default_memory_regime(custom)\n", t2b ? "PASS" : "FAIL");
    all = all && t2b;

    uint32_t ids[16] = {};
    size_t count = 0;
    bool list_ok = is_ok(fer_list_registered_regimes(sess, ids, 16, &count));
    bool found = false;
    for (size_t i = 0; i < count && i < 16; i++) if (ids[i] == kCustomRegime) found = true;
    bool t2c = list_ok && found;
    std::printf("[%s] fer_list_registered_regimes includes custom regime\n", t2c ? "PASS" : "FAIL");
    all = all && t2c;

    fer_buffer_desc_t d2{};
    d2.dtype = FER_DTYPE_F32;
    d2.rank = 2;
    d2.dims[0] = 2;
    d2.dims[1] = 2;

    uint64_t a_id = 0, b_id = 0, out_id = 0;
    all = all && is_ok(fer_alloc_buffer(sess, &d2, &a_id));
    all = all && is_ok(fer_alloc_buffer(sess, &d2, &b_id));
    all = all && is_ok(fer_alloc_buffer(sess, &d2, &out_id));

    std::vector<float> h_a = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> h_b = {5.f, 6.f, 7.f, 8.f};
    std::vector<float> h_out(4, 0.f);
    all = all && is_ok(fer_upload_f32(sess, a_id, h_a.data(), h_a.size()));
    all = all && is_ok(fer_upload_f32(sess, b_id, h_b.data(), h_b.size()));

    fer_matmul_request_t mm{};
    mm.a = a_id;
    mm.b = b_id;
    mm.out = out_id;
    mm.memory_regime = FER_MEMORY_AUTO;
    uint64_t jid = 0;
    all = all && is_ok(fer_submit_matmul(sess, &mm, &jid));
    all = all && is_ok(fer_job_wait(sess, jid));
    all = all && is_ok(fer_download_f32(sess, out_id, h_out.data(), h_out.size()));

    const float ref[4] = {19.f, 22.f, 43.f, 50.f};
    bool t3 = true;
    for (int i = 0; i < 4; i++) {
        if (!nearly_equal(h_out[i], ref[i])) {
            t3 = false;
            break;
        }
    }
    std::printf("[%s] C API custom allocator matmul output\n", t3 ? "PASS" : "FAIL");
    all = all && t3;

    all = all && is_ok(fer_free_buffer(sess, a_id));
    all = all && is_ok(fer_free_buffer(sess, b_id));
    all = all && is_ok(fer_free_buffer(sess, out_id));

    bool t4 = (stats.allocs >= 3 && stats.frees >= 3);
    std::printf("[%s] C allocator callbacks invoked (allocs=%d frees=%d)\n",
                t4 ? "PASS" : "FAIL", stats.allocs, stats.frees);
    all = all && t4;

    bool t5 = is_ok(fer_session_destroy(sess));
    std::printf("[%s] fer_session_destroy\n", t5 ? "PASS" : "FAIL");
    all = all && t5;

    // Create a new session to validate unregister + fallback semantics.
    sess = nullptr;
    all = all && is_ok(fer_session_create(0, nullptr, &sess));
    all = all && is_ok(fer_register_gpu_hot_allocator(sess, kCustomRegime, &a));
    all = all && is_ok(fer_set_default_memory_regime(sess, kCustomRegime));
    bool t6 = is_ok(fer_unregister_custom_allocator(sess, kCustomRegime));
    std::printf("[%s] fer_unregister_custom_allocator\n", t6 ? "PASS" : "FAIL");
    all = all && t6;

    uint32_t fallback = 999u;
    bool t7 = is_ok(fer_get_default_memory_regime(sess, &fallback)) && fallback == FER_MEMORY_CUSTOM_POOL;
    std::printf("[%s] default regime falls back after unregister\n", t7 ? "PASS" : "FAIL");
    all = all && t7;
    all = all && is_ok(fer_session_destroy(sess));

    std::printf("\n%s\n", all ? "ALL C API CUSTOM ALLOCATOR TESTS PASSED" : "SOME C API CUSTOM ALLOCATOR TESTS FAILED");
    return all ? 0 : 1;
}
