#include "fercuda/runtime/session.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace fer;
using namespace fer::runtime;

static bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

static bool run_matmul_case(MemoryRegime session_regime, uint32_t op_regime, const char* name) {
    PoolConfig cfg{};
    cfg.mutable_bytes = 64ULL << 20;
    cfg.immutable_bytes = 64ULL << 20;
    cfg.regime = session_regime;

    RuntimeSession sess(0, cfg);

    BufferDesc d2{};
    d2.dtype = BufferDType::F32;
    d2.rank = 2;
    d2.dims = {2, 2, 0, 0};

    BufferId a = 0, b = 0, out = 0;
    if (!sess.alloc_buffer(d2, &a).ok()) return false;
    if (!sess.alloc_buffer(d2, &b).ok()) return false;
    if (!sess.alloc_buffer(d2, &out).ok()) return false;

    std::vector<float> h_a = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> h_b = {5.f, 6.f, 7.f, 8.f};
    std::vector<float> h_out(4, 0.f);
    if (!sess.upload_f32(a, h_a.data(), h_a.size()).ok()) return false;
    if (!sess.upload_f32(b, h_b.data(), h_b.size()).ok()) return false;

    JobId jid = 0;
    MatmulRequest req{};
    req.a = a;
    req.b = b;
    req.out = out;
    req.memory_regime = op_regime;
    if (!sess.submit_matmul(req, &jid).ok()) return false;
    if (!sess.job_wait(jid).ok()) return false;
    if (!sess.download_f32(out, h_out.data(), h_out.size()).ok()) return false;

    const float ref[4] = {19.f, 22.f, 43.f, 50.f};
    bool ok = true;
    for (int i = 0; i < 4; i++) {
        if (!nearly_equal(h_out[i], ref[i])) {
            ok = false;
            break;
        }
    }

    std::printf("[%s] regime=%s matmul result\n", ok ? "PASS" : "FAIL", name);
    return ok;
}

int main() {
    bool ok_custom = run_matmul_case(
        MemoryRegime::CUSTOM_POOL, 0xFFFFFFFFu, "session=custom_pool op=auto");
    bool ok_malloc = run_matmul_case(
        MemoryRegime::CUDA_MALLOC, 0xFFFFFFFFu, "session=cuda_malloc op=auto");
    bool ok_override_malloc = run_matmul_case(
        MemoryRegime::CUSTOM_POOL, static_cast<uint32_t>(MemoryRegime::CUDA_MALLOC),
        "session=custom_pool op=cuda_malloc");
    bool ok_override_managed = run_matmul_case(
        MemoryRegime::CUSTOM_POOL, static_cast<uint32_t>(MemoryRegime::CUDA_MANAGED),
        "session=custom_pool op=cuda_managed");
    bool all = ok_custom && ok_malloc && ok_override_malloc && ok_override_managed;
    std::printf("\n%s\n", all ? "ALL MEMORY REGIME TESTS PASSED" : "SOME MEMORY REGIME TESTS FAILED");
    return all ? 0 : 1;
}
