/*
 * test_runtime_session.cu
 *
 * Validates RuntimeSession as a control-plane surface:
 *  - typed buffer allocation and transfers
 *  - async op submission and job waiting
 *  - matmul + layer_norm request execution
 */

#include "fercuda/runtime/session.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace fer;
using namespace fer::runtime;

static bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    RuntimeSession sess(0, {.mutable_bytes = 64ULL << 20, .immutable_bytes = 64ULL << 20});

    BufferId a = 0, b = 0, out = 0;
    BufferDesc d2{};
    d2.dtype = BufferDType::F32;
    d2.rank = 2;
    d2.dims = {2, 2, 0, 0};

    bool all = true;

    all = all && sess.alloc_buffer(d2, &a).ok();
    all = all && sess.alloc_buffer(d2, &b).ok();
    all = all && sess.alloc_buffer(d2, &out).ok();

    std::vector<float> h_a = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> h_b = {5.f, 6.f, 7.f, 8.f};
    std::vector<float> h_out(4, 0.f);

    all = all && sess.upload_f32(a, h_a.data(), h_a.size()).ok();
    all = all && sess.upload_f32(b, h_b.data(), h_b.size()).ok();

    JobId j1 = 0;
    Status sm = sess.submit_matmul({a, b, out}, &j1);
    bool t1 = sm.ok();
    printf("[%s] submit_matmul accepted request\n", t1 ? "PASS" : "FAIL");
    all = all && t1;

    bool done = false;
    Status qs = sess.job_status(j1, &done);
    bool t2 = qs.ok();
    printf("[%s] job_status returned cleanly\n", t2 ? "PASS" : "FAIL");
    all = all && t2;

    Status ws = sess.job_wait(j1);
    bool t3 = ws.ok();
    printf("[%s] job_wait completed\n", t3 ? "PASS" : "FAIL");
    all = all && t3;

    all = all && sess.download_f32(out, h_out.data(), h_out.size()).ok();
    const float mm_ref[4] = {19.f, 22.f, 43.f, 50.f};
    bool t4 = true;
    for (int i = 0; i < 4; i++) {
        if (!nearly_equal(h_out[i], mm_ref[i])) {
            t4 = false;
            break;
        }
    }
    printf("[%s] matmul output matches reference\n", t4 ? "PASS" : "FAIL");
    all = all && t4;

    BufferId x = 0, y = 0;
    BufferDesc d1{};
    d1.dtype = BufferDType::F32;
    d1.rank = 1;
    d1.dims = {4, 0, 0, 0};
    all = all && sess.alloc_buffer(d1, &x).ok();
    all = all && sess.alloc_buffer(d1, &y).ok();

    std::vector<float> h_x = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> h_y(4, 0.f);
    all = all && sess.upload_f32(x, h_x.data(), h_x.size()).ok();

    JobId j2 = 0;
    Status sl = sess.submit_layer_norm({x, y, 1e-6f}, &j2);
    bool t5 = sl.ok();
    printf("[%s] submit_layer_norm accepted request\n", t5 ? "PASS" : "FAIL");
    all = all && t5;

    all = all && sess.job_wait(j2).ok();
    all = all && sess.download_f32(y, h_y.data(), h_y.size()).ok();

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
    bool t6 = true;
    for (int i = 0; i < 4; i++) {
        float ref = (h_x[i] - mean) * inv_std;
        if (!nearly_equal(h_y[i], ref)) {
            t6 = false;
            break;
        }
    }
    printf("[%s] layer_norm output matches reference\n", t6 ? "PASS" : "FAIL");
    all = all && t6;

    printf("\n%s\n", all ? "ALL RUNTIME SESSION TESTS PASSED" : "SOME RUNTIME SESSION TESTS FAILED");
    return all ? 0 : 1;
}
