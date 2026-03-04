#include "fercuda/api/c_api.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

static bool is_ok(fer_status_t s) {
    return s.code == FER_STATUS_OK;
}

static uint16_t float_to_f16_bits(float x) {
    uint32_t f = 0;
    std::memcpy(&f, &x, sizeof(f));
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = f & 0x007FFFFFu;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant = (mant | 0x00800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x00001000u) >> 13));
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x00001000u) >> 13));
}

static float f16_bits_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;

    uint32_t out = 0;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            int e = -1;
            uint32_t m = mant;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            out = sign | (static_cast<uint32_t>(127 - 15 + 1 + e) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static uint16_t float_to_bf16_bits(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

static float bf16_bits_to_float(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

int main() {
    fer_pool_config_t cfg{};
    cfg.mutable_bytes = 64ULL << 20;
    cfg.immutable_bytes = 64ULL << 20;
    cfg.cuda_reserve = 0;
    cfg.verbose = 0;
    cfg.memory_regime = FER_MEMORY_CUDA_MALLOC;

    fer_session_t* sess = nullptr;
    fer_status_t st = fer_session_create(0, &cfg, &sess);

    bool all = true;
    bool t0 = is_ok(st) && sess != nullptr;
    std::printf("[%s] fer_session_create\n", t0 ? "PASS" : "FAIL");
    all = all && t0;

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

    fer_matmul_request_t mm{a, b, out, FER_MEMORY_AUTO};
    uint64_t j1 = 0;
    st = fer_submit_matmul(sess, &mm, &j1);
    bool t1 = is_ok(st) && j1 != 0;
    std::printf("[%s] fer_submit_matmul\n", t1 ? "PASS" : "FAIL");
    all = all && t1;

    st = fer_job_wait(sess, j1);
    bool t2 = is_ok(st);
    std::printf("[%s] fer_job_wait(matmul)\n", t2 ? "PASS" : "FAIL");
    all = all && t2;

    all = all && is_ok(fer_download_f32(sess, out, h_out.data(), h_out.size()));
    const float mm_ref[4] = {19.f, 22.f, 43.f, 50.f};
    bool t3 = true;
    for (int i = 0; i < 4; i++) {
        if (!nearly_equal(h_out[i], mm_ref[i])) {
            t3 = false;
            break;
        }
    }
    std::printf("[%s] C-API matmul output matches\n", t3 ? "PASS" : "FAIL");
    all = all && t3;

    fer_buffer_desc_t d1{};
    d1.dtype = FER_DTYPE_F32;
    d1.rank = 1;
    d1.dims[0] = 4;
    uint64_t x = 0, y = 0;
    all = all && is_ok(fer_alloc_buffer(sess, &d1, &x));
    all = all && is_ok(fer_alloc_buffer(sess, &d1, &y));

    std::vector<float> h_x = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> h_y(4, 0.f);
    all = all && is_ok(fer_upload_f32(sess, x, h_x.data(), h_x.size()));

    fer_layer_norm_request_t ln{x, y, 1e-6f, FER_MEMORY_AUTO};
    uint64_t j2 = 0;
    st = fer_submit_layer_norm(sess, &ln, &j2);
    bool t4 = is_ok(st) && j2 != 0;
    std::printf("[%s] fer_submit_layer_norm\n", t4 ? "PASS" : "FAIL");
    all = all && t4;

    all = all && is_ok(fer_job_wait(sess, j2));
    all = all && is_ok(fer_download_f32(sess, y, h_y.data(), h_y.size()));

    float mean = 0.f;
    for (float v : h_x) mean += v;
    mean /= 4.f;
    float var = 0.f;
    for (float v : h_x) {
        float d = v - mean;
        var += d * d;
    }
    var /= 4.f;
    float inv_std = 1.f / std::sqrt(var + 1e-6f);
    bool t5 = true;
    for (int i = 0; i < 4; i++) {
        float ref = (h_x[i] - mean) * inv_std;
        if (!nearly_equal(h_y[i], ref)) {
            t5 = false;
            break;
        }
    }
    std::printf("[%s] C-API layer_norm output matches\n", t5 ? "PASS" : "FAIL");
    all = all && t5;

    // F16 matmul path (exec planner cast-in/cast-out)
    fer_buffer_desc_t d2_f16{};
    d2_f16.dtype = FER_DTYPE_F16;
    d2_f16.rank = 2;
    d2_f16.dims[0] = 2;
    d2_f16.dims[1] = 2;
    uint64_t a16 = 0, b16 = 0, out16 = 0;
    all = all && is_ok(fer_alloc_buffer(sess, &d2_f16, &a16));
    all = all && is_ok(fer_alloc_buffer(sess, &d2_f16, &b16));
    all = all && is_ok(fer_alloc_buffer(sess, &d2_f16, &out16));
    std::vector<uint16_t> h_a16 = {
        float_to_f16_bits(1.f), float_to_f16_bits(2.f), float_to_f16_bits(3.f), float_to_f16_bits(4.f)};
    std::vector<uint16_t> h_b16 = {
        float_to_f16_bits(5.f), float_to_f16_bits(6.f), float_to_f16_bits(7.f), float_to_f16_bits(8.f)};
    std::vector<uint16_t> h_out16(4, 0);
    all = all && is_ok(fer_upload_bytes(sess, a16, h_a16.data(), h_a16.size() * sizeof(uint16_t)));
    all = all && is_ok(fer_upload_bytes(sess, b16, h_b16.data(), h_b16.size() * sizeof(uint16_t)));
    fer_matmul_request_t mm16{a16, b16, out16, FER_MEMORY_AUTO};
    uint64_t j16 = 0;
    all = all && is_ok(fer_submit_matmul(sess, &mm16, &j16));
    all = all && is_ok(fer_job_wait(sess, j16));
    all = all && is_ok(fer_download_bytes(sess, out16, h_out16.data(), h_out16.size() * sizeof(uint16_t)));
    bool t7 = true;
    for (int i = 0; i < 4; ++i) {
        if (!nearly_equal(f16_bits_to_float(h_out16[i]), mm_ref[i], 5e-2f)) {
            t7 = false;
            break;
        }
    }
    std::printf("[%s] C-API f16 matmul output matches\n", t7 ? "PASS" : "FAIL");
    all = all && t7;

    // BF16 matmul path (exec planner cast-in/cast-out)
    fer_buffer_desc_t d2_bf16{};
    d2_bf16.dtype = FER_DTYPE_BF16;
    d2_bf16.rank = 2;
    d2_bf16.dims[0] = 2;
    d2_bf16.dims[1] = 2;
    uint64_t ab = 0, bb = 0, outb = 0;
    all = all && is_ok(fer_alloc_buffer(sess, &d2_bf16, &ab));
    all = all && is_ok(fer_alloc_buffer(sess, &d2_bf16, &bb));
    all = all && is_ok(fer_alloc_buffer(sess, &d2_bf16, &outb));
    std::vector<uint16_t> h_ab = {
        float_to_bf16_bits(1.f), float_to_bf16_bits(2.f), float_to_bf16_bits(3.f), float_to_bf16_bits(4.f)};
    std::vector<uint16_t> h_bb = {
        float_to_bf16_bits(5.f), float_to_bf16_bits(6.f), float_to_bf16_bits(7.f), float_to_bf16_bits(8.f)};
    std::vector<uint16_t> h_outb(4, 0);
    all = all && is_ok(fer_upload_bytes(sess, ab, h_ab.data(), h_ab.size() * sizeof(uint16_t)));
    all = all && is_ok(fer_upload_bytes(sess, bb, h_bb.data(), h_bb.size() * sizeof(uint16_t)));
    fer_matmul_request_t mmb{ab, bb, outb, FER_MEMORY_AUTO};
    uint64_t jb = 0;
    all = all && is_ok(fer_submit_matmul(sess, &mmb, &jb));
    all = all && is_ok(fer_job_wait(sess, jb));
    all = all && is_ok(fer_download_bytes(sess, outb, h_outb.data(), h_outb.size() * sizeof(uint16_t)));
    bool t8 = true;
    for (int i = 0; i < 4; ++i) {
        if (!nearly_equal(bf16_bits_to_float(h_outb[i]), mm_ref[i], 2e-1f)) {
            t8 = false;
            break;
        }
    }
    std::printf("[%s] C-API bf16 matmul output matches\n", t8 ? "PASS" : "FAIL");
    all = all && t8;

    // F16 layer_norm path (exec planner cast-in/cast-out)
    fer_buffer_desc_t d1_f16{};
    d1_f16.dtype = FER_DTYPE_F16;
    d1_f16.rank = 1;
    d1_f16.dims[0] = 4;
    uint64_t x16 = 0, y16 = 0;
    all = all && is_ok(fer_alloc_buffer(sess, &d1_f16, &x16));
    all = all && is_ok(fer_alloc_buffer(sess, &d1_f16, &y16));
    std::vector<uint16_t> h_x16 = {
        float_to_f16_bits(1.f), float_to_f16_bits(2.f), float_to_f16_bits(3.f), float_to_f16_bits(4.f)};
    std::vector<uint16_t> h_y16(4, 0);
    all = all && is_ok(fer_upload_bytes(sess, x16, h_x16.data(), h_x16.size() * sizeof(uint16_t)));
    fer_layer_norm_request_t ln16{x16, y16, 1e-6f, FER_MEMORY_AUTO};
    uint64_t jln16 = 0;
    all = all && is_ok(fer_submit_layer_norm(sess, &ln16, &jln16));
    all = all && is_ok(fer_job_wait(sess, jln16));
    all = all && is_ok(fer_download_bytes(sess, y16, h_y16.data(), h_y16.size() * sizeof(uint16_t)));
    bool t9 = true;
    for (int i = 0; i < 4; ++i) {
        float got = f16_bits_to_float(h_y16[i]);
        float ref = (h_x[i] - mean) * inv_std;
        if (!nearly_equal(got, ref, 7e-2f)) {
            t9 = false;
            break;
        }
    }
    std::printf("[%s] C-API f16 layer_norm output matches\n", t9 ? "PASS" : "FAIL");
    all = all && t9;

    // BF16 layer_norm path (exec planner cast-in/cast-out)
    fer_buffer_desc_t d1_bf16{};
    d1_bf16.dtype = FER_DTYPE_BF16;
    d1_bf16.rank = 1;
    d1_bf16.dims[0] = 4;
    uint64_t xb = 0, yb = 0;
    all = all && is_ok(fer_alloc_buffer(sess, &d1_bf16, &xb));
    all = all && is_ok(fer_alloc_buffer(sess, &d1_bf16, &yb));
    std::vector<uint16_t> h_xb = {
        float_to_bf16_bits(1.f), float_to_bf16_bits(2.f), float_to_bf16_bits(3.f), float_to_bf16_bits(4.f)};
    std::vector<uint16_t> h_yb(4, 0);
    all = all && is_ok(fer_upload_bytes(sess, xb, h_xb.data(), h_xb.size() * sizeof(uint16_t)));
    fer_layer_norm_request_t lnb{xb, yb, 1e-6f, FER_MEMORY_AUTO};
    uint64_t jlnb = 0;
    all = all && is_ok(fer_submit_layer_norm(sess, &lnb, &jlnb));
    all = all && is_ok(fer_job_wait(sess, jlnb));
    all = all && is_ok(fer_download_bytes(sess, yb, h_yb.data(), h_yb.size() * sizeof(uint16_t)));
    bool t10 = true;
    for (int i = 0; i < 4; ++i) {
        float got = bf16_bits_to_float(h_yb[i]);
        float ref = (h_x[i] - mean) * inv_std;
        if (!nearly_equal(got, ref, 2e-1f)) {
            t10 = false;
            break;
        }
    }
    std::printf("[%s] C-API bf16 layer_norm output matches\n", t10 ? "PASS" : "FAIL");
    all = all && t10;

    // Guardrail: dtype mismatch must fail
    fer_buffer_desc_t d2f32{};
    d2f32.dtype = FER_DTYPE_F32;
    d2f32.rank = 2;
    d2f32.dims[0] = 2;
    d2f32.dims[1] = 2;
    uint64_t mix_a = 0, mix_b = 0, mix_out = 0;
    all = all && is_ok(fer_alloc_buffer(sess, &d2_f16, &mix_a));
    all = all && is_ok(fer_alloc_buffer(sess, &d2f32, &mix_b));
    all = all && is_ok(fer_alloc_buffer(sess, &d2_f16, &mix_out));
    fer_matmul_request_t mm_mix{mix_a, mix_b, mix_out, FER_MEMORY_AUTO};
    uint64_t jmix = 0;
    fer_status_t st_mix = fer_submit_matmul(sess, &mm_mix, &jmix);
    bool t11 = !is_ok(st_mix) && st_mix.code == FER_STATUS_INVALID_ARGUMENT;
    std::printf("[%s] C-API dtype mismatch rejected\n", t11 ? "PASS" : "FAIL");
    all = all && t11;

    // I8 matmul compute path (cast-in/cast-out with saturation)
    fer_buffer_desc_t d2_i8{};
    d2_i8.dtype = FER_DTYPE_I8;
    d2_i8.rank = 2;
    d2_i8.dims[0] = 2;
    d2_i8.dims[1] = 2;
    uint64_t ai8 = 0, bi8 = 0, oi8 = 0;
    all = all && is_ok(fer_alloc_buffer(sess, &d2_i8, &ai8));
    all = all && is_ok(fer_alloc_buffer(sess, &d2_i8, &bi8));
    all = all && is_ok(fer_alloc_buffer(sess, &d2_i8, &oi8));
    std::vector<int8_t> h_ai8 = {1, 2, 3, 4};
    std::vector<int8_t> h_bi8 = {5, 6, 7, 8};
    std::vector<int8_t> h_oi8(4, 0);
    all = all && is_ok(fer_upload_bytes(sess, ai8, h_ai8.data(), h_ai8.size() * sizeof(int8_t)));
    all = all && is_ok(fer_upload_bytes(sess, bi8, h_bi8.data(), h_bi8.size() * sizeof(int8_t)));
    fer_matmul_request_t mmi8{ai8, bi8, oi8, FER_MEMORY_AUTO};
    uint64_t ji8 = 0;
    all = all && is_ok(fer_submit_matmul(sess, &mmi8, &ji8));
    all = all && is_ok(fer_job_wait(sess, ji8));
    all = all && is_ok(fer_download_bytes(sess, oi8, h_oi8.data(), h_oi8.size() * sizeof(int8_t)));
    bool t12 = (h_oi8[0] == 19 && h_oi8[1] == 22 && h_oi8[2] == 43 && h_oi8[3] == 50);
    std::printf("[%s] C-API i8 matmul output matches\n", t12 ? "PASS" : "FAIL");
    all = all && t12;

    // U8 matmul compute path
    fer_buffer_desc_t d2_u8{};
    d2_u8.dtype = FER_DTYPE_U8;
    d2_u8.rank = 2;
    d2_u8.dims[0] = 2;
    d2_u8.dims[1] = 2;
    uint64_t au8 = 0, bu8 = 0, ou8 = 0;
    all = all && is_ok(fer_alloc_buffer(sess, &d2_u8, &au8));
    all = all && is_ok(fer_alloc_buffer(sess, &d2_u8, &bu8));
    all = all && is_ok(fer_alloc_buffer(sess, &d2_u8, &ou8));
    std::vector<uint8_t> h_au8 = {1, 2, 3, 4};
    std::vector<uint8_t> h_bu8 = {5, 6, 7, 8};
    std::vector<uint8_t> h_ou8(4, 0);
    all = all && is_ok(fer_upload_bytes(sess, au8, h_au8.data(), h_au8.size() * sizeof(uint8_t)));
    all = all && is_ok(fer_upload_bytes(sess, bu8, h_bu8.data(), h_bu8.size() * sizeof(uint8_t)));
    fer_matmul_request_t mmu8{au8, bu8, ou8, FER_MEMORY_AUTO};
    uint64_t ju8 = 0;
    all = all && is_ok(fer_submit_matmul(sess, &mmu8, &ju8));
    all = all && is_ok(fer_job_wait(sess, ju8));
    all = all && is_ok(fer_download_bytes(sess, ou8, h_ou8.data(), h_ou8.size() * sizeof(uint8_t)));
    bool t13 = (h_ou8[0] == 19 && h_ou8[1] == 22 && h_ou8[2] == 43 && h_ou8[3] == 50);
    std::printf("[%s] C-API u8 matmul output matches\n", t13 ? "PASS" : "FAIL");
    all = all && t13;

    fer_status_t d = fer_session_destroy(sess);
    bool t6 = is_ok(d);
    std::printf("[%s] fer_session_destroy\n", t6 ? "PASS" : "FAIL");
    all = all && t6;

    std::printf("\n%s\n", all ? "ALL C API BRIDGE TESTS PASSED" : "SOME C API BRIDGE TESTS FAILED");
    return all ? 0 : 1;
}
