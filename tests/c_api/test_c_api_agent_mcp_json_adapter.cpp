#include "fercuda/agent/mcp_json_adapter.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

static bool json_has(const char* s, const char* needle) {
    return s && needle && std::strstr(s, needle) != nullptr;
}

static uint64_t json_u64(const char* s, const char* key) {
    const std::string pat = std::string("\"") + key + "\":";
    const char* p = std::strstr(s, pat.c_str());
    if (!p) return 0;
    p += pat.size();
    return static_cast<uint64_t>(std::strtoull(p, nullptr, 10));
}

int main() {
    fer_agent_adapter_t* adapter = nullptr;
    if (fer_agent_adapter_create(&adapter).code != FER_STATUS_OK || !adapter) {
        return fail("fer_agent_adapter_create");
    }

    char out[8192];

    if (fer_agent_mcp_dispatch(adapter, "fer.runtime.inspect", "{}", out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch runtime.inspect");
    }
    if (!json_has(out, "\"ok\":true") || !json_has(out, "\"agent_api_version\":\"v1alpha1\"")) {
        return fail("runtime.inspect response");
    }

    const char* create_json =
        "{\"agent_api_version\":\"v1alpha1\",\"device\":0,"
        "\"mutable_bytes\":33554432,\"immutable_bytes\":33554432,"
        "\"memory_regime\":\"custom_pool\"}";
    if (fer_agent_mcp_dispatch(adapter, "fer.session.create", create_json, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch session.create");
    }
    uint64_t session_id = json_u64(out, "session_id");
    if (session_id == 0) return fail("session_id parse");

    if (fer_agent_mcp_dispatch(adapter, "fer.session.list", "{\"agent_api_version\":\"v1alpha1\"}", out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch session.list");
    }
    if (!json_has(out, "\"count\":1")) return fail("session.list count");

    char tensor_create[512];
    std::snprintf(
        tensor_create, sizeof(tensor_create),
        "{\"agent_api_version\":\"v1alpha1\",\"session_id\":%llu,"
        "\"dtype\":\"f32\",\"shape\":[128],\"memory_regime\":\"custom_pool\"}",
        static_cast<unsigned long long>(session_id));

    if (fer_agent_mcp_dispatch(adapter, "fer.tensor.create", tensor_create, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch tensor.create x");
    }
    uint64_t x = json_u64(out, "tensor_id");
    if (!x) return fail("tensor_id x parse");
    if (fer_agent_mcp_dispatch(adapter, "fer.tensor.create", tensor_create, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch tensor.create y");
    }
    uint64_t y = json_u64(out, "tensor_id");
    if (!y) return fail("tensor_id y parse");

    char tensor_list_json[256];
    std::snprintf(
        tensor_list_json, sizeof(tensor_list_json),
        "{\"agent_api_version\":\"v1alpha1\",\"session_id\":%llu}",
        static_cast<unsigned long long>(session_id));
    if (fer_agent_mcp_dispatch(adapter, "fer.tensor.list", tensor_list_json, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch tensor.list");
    }
    if (!json_has(out, "\"count\":2")) return fail("tensor.list count");

    char session_stats_json[256];
    std::snprintf(
        session_stats_json, sizeof(session_stats_json),
        "{\"agent_api_version\":\"v1alpha1\",\"session_id\":%llu}",
        static_cast<unsigned long long>(session_id));
    if (fer_agent_mcp_dispatch(adapter, "fer.session.stats", session_stats_json, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch session.stats before run");
    }
    if (!json_has(out, "\"tensor_count\":2")) return fail("session.stats tensor_count before run");

    constexpr size_t n = 128;
    std::vector<float> in(n, 4.0f);
    std::vector<float> out_h(n, 0.0f);
    const uint8_t* in_bytes = reinterpret_cast<const uint8_t*>(in.data());
    const size_t in_bytes_len = n * sizeof(float);
    std::string in_hex;
    in_hex.reserve(in_bytes_len * 2);
    static const char* kHex = "0123456789abcdef";
    for (size_t i = 0; i < in_bytes_len; ++i) {
        in_hex.push_back(kHex[(in_bytes[i] >> 4) & 0xF]);
        in_hex.push_back(kHex[in_bytes[i] & 0xF]);
    }

    std::string blob_put_json =
        std::string("{\"agent_api_version\":\"v1alpha1\",\"host_blob_ref\":\"blob_in\",\"blob_hex\":\"") +
        in_hex + "\"}";
    if (fer_agent_mcp_dispatch(adapter, "fer.blob.put", blob_put_json.c_str(), out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch blob.put");
    }
    if (!json_has(out, "\"host_blob_ref\":\"blob_in\"")) return fail("blob.put response");

    char copy_h2d[512];
    std::snprintf(
        copy_h2d, sizeof(copy_h2d),
        "{\"agent_api_version\":\"v1alpha1\",\"session_id\":%llu,"
        "\"tensor_id\":%llu,\"dtype\":\"f32\",\"count\":%zu,"
        "\"direction\":\"host_to_device\",\"host_blob_ref\":\"blob_in\"}",
        static_cast<unsigned long long>(session_id),
        static_cast<unsigned long long>(x),
        n);
    if (fer_agent_mcp_dispatch(adapter, "fer.tensor.copy", copy_h2d, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch tensor.copy h2d");
    }

    char run_intent[768];
    std::snprintf(
        run_intent, sizeof(run_intent),
        "{\"agent_api_version\":\"v1alpha1\",\"session_id\":%llu,"
        "\"op\":\"affine_f32\",\"n\":128,\"alpha\":2.0,\"beta\":1.0,"
        "\"fusion_mask\":[\"relu\"],\"caps_mask\":[],\"memory_regime\":\"custom_pool\","
        "\"input\":%llu,\"output\":%llu}",
        static_cast<unsigned long long>(session_id),
        static_cast<unsigned long long>(x),
        static_cast<unsigned long long>(y));
    if (fer_agent_mcp_dispatch(adapter, "fer.jit.intent.run", run_intent, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch jit.intent.run");
    }
    uint64_t job_id = json_u64(out, "job_id");
    if (!job_id) return fail("job_id parse");

    char job_list_json[256];
    std::snprintf(
        job_list_json, sizeof(job_list_json),
        "{\"agent_api_version\":\"v1alpha1\",\"session_id\":%llu}",
        static_cast<unsigned long long>(session_id));
    if (fer_agent_mcp_dispatch(adapter, "fer.job.list", job_list_json, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch job.list before wait");
    }
    if (!json_has(out, "\"count\":1")) return fail("job.list count before wait");

    if (fer_agent_mcp_dispatch(adapter, "fer.session.stats", session_stats_json, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch session.stats with in-flight job");
    }
    if (!json_has(out, "\"job_count\":1")) return fail("session.stats job_count in-flight");

    char wait_json[256];
    std::snprintf(
        wait_json, sizeof(wait_json),
        "{\"agent_api_version\":\"v1alpha1\",\"session_id\":%llu,\"job_id\":%llu}",
        static_cast<unsigned long long>(session_id),
        static_cast<unsigned long long>(job_id));
    if (fer_agent_mcp_dispatch(adapter, "fer.job.wait", wait_json, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch job.wait");
    }

    if (fer_agent_mcp_dispatch(adapter, "fer.job.list", job_list_json, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch job.list after wait");
    }
    if (!json_has(out, "\"count\":0")) return fail("job.list count after wait");

    if (fer_agent_mcp_dispatch(adapter, "fer.jit.intent.run", run_intent, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch jit.intent.run #2");
    }
    uint64_t job_id2 = json_u64(out, "job_id");
    if (!job_id2) return fail("job_id2 parse");
    char wait_json2[256];
    std::snprintf(
        wait_json2, sizeof(wait_json2),
        "{\"agent_api_version\":\"v1alpha1\",\"session_id\":%llu,\"job_id\":%llu}",
        static_cast<unsigned long long>(session_id),
        static_cast<unsigned long long>(job_id2));
    bool cancelled_ok = false;
    for (int i = 0; i < 100; ++i) {
        const fer_status_t st_cancel = fer_agent_mcp_dispatch(adapter, "fer.job.cancel", wait_json2, out, sizeof(out));
        if (st_cancel.code == FER_STATUS_OK && json_has(out, "\"cancelled\":true")) {
            cancelled_ok = true;
            break;
        }
        if (!json_has(out, "unsupported for running jobs")) {
            return fail("dispatch job.cancel unexpected error");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!cancelled_ok) return fail("dispatch job.cancel completed");

    if (fer_agent_mcp_dispatch(adapter, "fer.session.stats", session_stats_json, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch session.stats after wait");
    }
    if (!json_has(out, "\"job_count\":0")) return fail("session.stats job_count after wait");

    char copy_d2h[512];
    std::snprintf(
        copy_d2h, sizeof(copy_d2h),
        "{\"agent_api_version\":\"v1alpha1\",\"session_id\":%llu,"
        "\"tensor_id\":%llu,\"dtype\":\"f32\",\"count\":%zu,"
        "\"direction\":\"device_to_host\",\"host_blob_ref\":\"blob_out\"}",
        static_cast<unsigned long long>(session_id),
        static_cast<unsigned long long>(y),
        n);
    if (fer_agent_mcp_dispatch(adapter, "fer.tensor.copy", copy_d2h, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch tensor.copy d2h");
    }

    const char* blob_get_json = "{\"agent_api_version\":\"v1alpha1\",\"host_blob_ref\":\"blob_out\"}";
    if (fer_agent_mcp_dispatch(adapter, "fer.blob.get", blob_get_json, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch blob.get");
    }
    const char* hex_pos = std::strstr(out, "\"blob_hex\":\"");
    if (!hex_pos) return fail("blob.get missing blob_hex");
    hex_pos += std::strlen("\"blob_hex\":\"");
    std::string out_hex;
    while (*hex_pos && *hex_pos != '"') out_hex.push_back(*hex_pos++);
    if (out_hex.size() != in_bytes_len * 2) return fail("blob.get hex size mismatch");
    for (size_t i = 0; i < in_bytes_len; ++i) {
        auto nib = [](char c) -> uint8_t {
            if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
            if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
            if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(c - 'A' + 10);
            return 0xFF;
        };
        const uint8_t hi = nib(out_hex[2 * i]);
        const uint8_t lo = nib(out_hex[2 * i + 1]);
        if (hi == 0xFF || lo == 0xFF) return fail("blob.get invalid hex");
        reinterpret_cast<uint8_t*>(out_h.data())[i] = static_cast<uint8_t>((hi << 4) | lo);
    }

    for (size_t i = 0; i < n; ++i) {
        if (out_h[i] != 9.0f) return fail("result mismatch");
    }

    const char* bad_ver = "{\"agent_api_version\":\"v9\",\"session_id\":1}";
    if (fer_agent_mcp_dispatch(adapter, "fer.job.wait", bad_ver, out, sizeof(out)).code == FER_STATUS_OK) {
        return fail("expected version error");
    }
    if (!json_has(out, "\"ok\":false")) return fail("error envelope");

    if (setenv("FERCUDA_AGENT_MAX_TENSOR_BYTES", "128", 1) != 0) {
        return fail("setenv FERCUDA_AGENT_MAX_TENSOR_BYTES");
    }
    if (fer_agent_mcp_dispatch(adapter, "fer.tensor.create", tensor_create, out, sizeof(out)).code == FER_STATUS_OK) {
        return fail("expected policy denial for oversized tensor");
    }
    if (!json_has(out, "RESOURCE_EXHAUSTED")) return fail("policy denial error code text");
    unsetenv("FERCUDA_AGENT_MAX_TENSOR_BYTES");

    char release_x_json[256];
    std::snprintf(
        release_x_json, sizeof(release_x_json),
        "{\"agent_api_version\":\"v1alpha1\",\"session_id\":%llu,\"tensor_id\":%llu}",
        static_cast<unsigned long long>(session_id),
        static_cast<unsigned long long>(x));
    if (fer_agent_mcp_dispatch(adapter, "fer.tensor.release", release_x_json, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch tensor.release x");
    }
    char release_y_json[256];
    std::snprintf(
        release_y_json, sizeof(release_y_json),
        "{\"agent_api_version\":\"v1alpha1\",\"session_id\":%llu,\"tensor_id\":%llu}",
        static_cast<unsigned long long>(session_id),
        static_cast<unsigned long long>(y));
    if (fer_agent_mcp_dispatch(adapter, "fer.tensor.release", release_y_json, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch tensor.release y");
    }
    if (fer_agent_mcp_dispatch(adapter, "fer.tensor.list", tensor_list_json, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch tensor.list after release");
    }
    if (!json_has(out, "\"count\":0")) return fail("tensor.list count after release");

    char destroy_json[256];
    std::snprintf(
        destroy_json, sizeof(destroy_json),
        "{\"agent_api_version\":\"v1alpha1\",\"session_id\":%llu}",
        static_cast<unsigned long long>(session_id));
    if (fer_agent_mcp_dispatch(adapter, "fer.session.destroy", destroy_json, out, sizeof(out)).code != FER_STATUS_OK) {
        return fail("dispatch session.destroy");
    }

    if (fer_agent_adapter_destroy(adapter).code != FER_STATUS_OK) {
        return fail("fer_agent_adapter_destroy");
    }

    std::printf("AGENT MCP JSON ADAPTER TEST PASSED\n");
    return 0;
}
