#include "fercuda/runtime/session.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace fer;
using namespace fer::runtime;

namespace {

struct AllocStats {
    int allocs = 0;
    int frees = 0;
};

void* cb_alloc(size_t bytes, Tier, uint32_t, void* user_ctx) {
    auto* s = static_cast<AllocStats*>(user_ctx);
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    s->allocs++;
    return p;
}

void cb_free(void* ptr, void* user_ctx) {
    auto* s = static_cast<AllocStats*>(user_ctx);
    cudaFree(ptr);
    s->frees++;
}

bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

} // namespace

int main() {
    constexpr uint32_t kCustomRegime = 1001u;
    bool all = true;

    RuntimeSession sess(0, {});
    AllocStats stats{};
    CustomAllocator custom{};
    custom.alloc = &cb_alloc;
    custom.free = &cb_free;
    custom.user_ctx = &stats;

    Status r = sess.register_custom_allocator(kCustomRegime, custom);
    bool t1 = r.ok();
    std::printf("[%s] register_custom_allocator\n", t1 ? "PASS" : "FAIL");
    all = all && t1;

    Status d = sess.set_default_memory_regime(kCustomRegime);
    bool t2 = d.ok();
    std::printf("[%s] set_default_memory_regime(custom)\n", t2 ? "PASS" : "FAIL");
    all = all && t2;

    BufferDesc d2{};
    d2.dtype = BufferDType::F32;
    d2.rank = 2;
    d2.dims = {2, 2, 0, 0};

    BufferId a = 0, b = 0, out = 0;
    all = all && sess.alloc_buffer(d2, &a).ok();
    all = all && sess.alloc_buffer(d2, &b).ok();
    all = all && sess.alloc_buffer(d2, &out).ok();

    std::vector<float> h_a = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> h_b = {5.f, 6.f, 7.f, 8.f};
    std::vector<float> h_out(4, 0.f);
    all = all && sess.upload_f32(a, h_a.data(), h_a.size()).ok();
    all = all && sess.upload_f32(b, h_b.data(), h_b.size()).ok();

    MatmulRequest req{};
    req.a = a;
    req.b = b;
    req.out = out;
    req.memory_regime = MEMORY_REGIME_AUTO;

    JobId jid = 0;
    all = all && sess.submit_matmul(req, &jid).ok();
    all = all && sess.job_wait(jid).ok();
    all = all && sess.download_f32(out, h_out.data(), h_out.size()).ok();

    const float ref[4] = {19.f, 22.f, 43.f, 50.f};
    bool t3 = true;
    for (int i = 0; i < 4; i++) {
        if (!nearly_equal(h_out[i], ref[i])) {
            t3 = false;
            break;
        }
    }
    std::printf("[%s] custom allocator matmul output\n", t3 ? "PASS" : "FAIL");
    all = all && t3;

    all = all && sess.free_buffer(a).ok();
    all = all && sess.free_buffer(b).ok();
    all = all && sess.free_buffer(out).ok();

    bool t4 = (stats.allocs >= 3 && stats.frees >= 3);
    std::printf("[%s] custom allocator callbacks invoked (allocs=%d frees=%d)\n",
                t4 ? "PASS" : "FAIL", stats.allocs, stats.frees);
    all = all && t4;

    uint32_t default_regime = 0;
    bool t5 = sess.get_default_memory_regime(&default_regime).ok() && default_regime == kCustomRegime;
    std::printf("[%s] get_default_memory_regime returns custom regime\n", t5 ? "PASS" : "FAIL");
    all = all && t5;

    uint32_t ids[16] = {};
    size_t count = 0;
    bool list_ok = sess.list_registered_regimes(ids, 16, &count).ok();
    bool found = false;
    for (size_t i = 0; i < count && i < 16; i++) if (ids[i] == kCustomRegime) found = true;
    bool t6 = list_ok && found;
    std::printf("[%s] list_registered_regimes includes custom regime\n", t6 ? "PASS" : "FAIL");
    all = all && t6;

    bool t7 = sess.unregister_custom_allocator(kCustomRegime).ok();
    std::printf("[%s] unregister_custom_allocator\n", t7 ? "PASS" : "FAIL");
    all = all && t7;

    bool t8 = sess.get_default_memory_regime(&default_regime).ok() &&
              default_regime == static_cast<uint32_t>(MemoryRegime::CUSTOM_POOL);
    std::printf("[%s] default regime falls back to CUSTOM_POOL after unregister\n", t8 ? "PASS" : "FAIL");
    all = all && t8;

    std::printf("\n%s\n", all ? "ALL CUSTOM ALLOCATOR TESTS PASSED" : "SOME CUSTOM ALLOCATOR TESTS FAILED");
    return all ? 0 : 1;
}
