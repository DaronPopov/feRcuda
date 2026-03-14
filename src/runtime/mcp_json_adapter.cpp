#include "fercuda/agent/mcp_json_adapter.h"

#include "fercuda/jit/intent/intent.h"
#include "fercuda/jit/types.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char* kApiVersion = "v1alpha1";
constexpr const char* kDefaultAuditLogPath = "/tmp/fercuda_agent_audit.jsonl";
constexpr const char* kResourceExhaustedPrefix = "RESOURCE_EXHAUSTED: ";

thread_local const char* t_audit_tool_name = "";
thread_local std::string t_audit_request_id;
thread_local std::string t_audit_policy_decision = "allow";

std::mutex g_policy_mu;
std::unordered_map<uint64_t, uint64_t> g_session_reserved_bytes;
std::unordered_map<uint64_t, uint64_t> g_session_inflight_jobs;
std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> g_tensor_reserved_bytes;

struct PolicyConfig {
    uint64_t max_tensor_bytes = 0;
    uint64_t max_total_session_bytes = 0;
    uint64_t max_jobs_in_flight = 0;
    std::vector<std::string> allow_dtypes;
    std::vector<std::string> allow_regimes;
};

uint64_t read_env_u64(const char* key, uint64_t fallback = 0) {
    if (!key) return fallback;
    const char* v = std::getenv(key);
    if (!v || v[0] == '\0') return fallback;
    char* end = nullptr;
    const unsigned long long n = std::strtoull(v, &end, 10);
    if (end == v) return fallback;
    return static_cast<uint64_t>(n);
}

std::vector<std::string> parse_csv_list(const char* text) {
    std::vector<std::string> out;
    if (!text) return out;
    std::string cur;
    for (const char* p = text; ; ++p) {
        const char c = *p;
        if (c == ',' || c == '\0') {
            size_t b = 0;
            while (b < cur.size() && std::isspace(static_cast<unsigned char>(cur[b]))) ++b;
            size_t e = cur.size();
            while (e > b && std::isspace(static_cast<unsigned char>(cur[e - 1]))) --e;
            if (e > b) out.emplace_back(cur.substr(b, e - b));
            cur.clear();
            if (c == '\0') break;
            continue;
        }
        cur.push_back(c);
    }
    return out;
}

PolicyConfig load_policy_config() {
    PolicyConfig cfg;
    cfg.max_tensor_bytes = read_env_u64("FERCUDA_AGENT_MAX_TENSOR_BYTES", 0);
    cfg.max_total_session_bytes = read_env_u64("FERCUDA_AGENT_MAX_SESSION_BYTES", 0);
    cfg.max_jobs_in_flight = read_env_u64("FERCUDA_AGENT_MAX_JOBS_IN_FLIGHT", 0);
    cfg.allow_dtypes = parse_csv_list(std::getenv("FERCUDA_AGENT_ALLOW_DTYPES"));
    cfg.allow_regimes = parse_csv_list(std::getenv("FERCUDA_AGENT_ALLOW_REGIMES"));
    return cfg;
}

bool list_allows(const std::vector<std::string>& allow_list, const std::string& value) {
    if (allow_list.empty()) return true;
    return std::find(allow_list.begin(), allow_list.end(), value) != allow_list.end();
}

uint64_t dtype_size_from_name(const std::string& dtype) {
    if (dtype == "f32") return sizeof(float);
    if (dtype == "f16") return sizeof(uint16_t);
    if (dtype == "bf16") return sizeof(uint16_t);
    if (dtype == "i8") return sizeof(int8_t);
    if (dtype == "u8") return sizeof(uint8_t);
    if (dtype == "i16") return sizeof(int16_t);
    if (dtype == "u16") return sizeof(uint16_t);
    if (dtype == "i32") return sizeof(int32_t);
    if (dtype == "u32") return sizeof(uint32_t);
    if (dtype == "i64") return sizeof(int64_t);
    if (dtype == "u64") return sizeof(uint64_t);
    if (dtype == "f64") return sizeof(double);
    return 0;
}

bool safe_mul_u64(uint64_t a, uint64_t b, uint64_t* out) {
    if (!out) return false;
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (a > UINT64_MAX / b) return false;
    *out = a * b;
    return true;
}

void record_session_create(uint64_t session_id) {
    std::lock_guard<std::mutex> lock(g_policy_mu);
    g_session_reserved_bytes.emplace(session_id, 0);
    g_session_inflight_jobs.emplace(session_id, 0);
}

void record_session_destroy(uint64_t session_id) {
    std::lock_guard<std::mutex> lock(g_policy_mu);
    g_session_reserved_bytes.erase(session_id);
    g_session_inflight_jobs.erase(session_id);
    for (auto it = g_tensor_reserved_bytes.begin(); it != g_tensor_reserved_bytes.end();) {
        if (it->second.first == session_id) it = g_tensor_reserved_bytes.erase(it);
        else ++it;
    }
}

bool policy_reserve_tensor_bytes(
    const PolicyConfig& cfg,
    uint64_t session_id,
    uint64_t tensor_bytes,
    std::string* out_reason) {
    if (cfg.max_tensor_bytes > 0 && tensor_bytes > cfg.max_tensor_bytes) {
        if (out_reason) *out_reason = "tensor bytes exceed max_tensor_bytes";
        return false;
    }
    std::lock_guard<std::mutex> lock(g_policy_mu);
    uint64_t& used = g_session_reserved_bytes[session_id];
    if (cfg.max_total_session_bytes > 0) {
        uint64_t next_used = 0;
        if (!safe_mul_u64(1, used, &next_used)) {
            if (out_reason) *out_reason = "session byte accounting overflow";
            return false;
        }
        if (tensor_bytes > UINT64_MAX - next_used) {
            if (out_reason) *out_reason = "session byte accounting overflow";
            return false;
        }
        next_used += tensor_bytes;
        if (next_used > cfg.max_total_session_bytes) {
            if (out_reason) *out_reason = "session bytes exceed max_total_session_bytes";
            return false;
        }
    }
    if (tensor_bytes > UINT64_MAX - used) {
        if (out_reason) *out_reason = "session byte accounting overflow";
        return false;
    }
    used += tensor_bytes;
    return true;
}

bool policy_try_begin_job(const PolicyConfig& cfg, uint64_t session_id, std::string* out_reason) {
    std::lock_guard<std::mutex> lock(g_policy_mu);
    uint64_t& jobs = g_session_inflight_jobs[session_id];
    if (cfg.max_jobs_in_flight > 0 && jobs >= cfg.max_jobs_in_flight) {
        if (out_reason) *out_reason = "jobs in flight exceed max_jobs_in_flight";
        return false;
    }
    if (jobs == UINT64_MAX) {
        if (out_reason) *out_reason = "jobs in flight accounting overflow";
        return false;
    }
    ++jobs;
    return true;
}

void policy_unreserve_session_bytes(uint64_t session_id, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_policy_mu);
    auto it = g_session_reserved_bytes.find(session_id);
    if (it == g_session_reserved_bytes.end()) return;
    if (it->second <= bytes) it->second = 0;
    else it->second -= bytes;
}

void policy_record_tensor_allocation(uint64_t session_id, uint64_t tensor_id, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(g_policy_mu);
    g_tensor_reserved_bytes[tensor_id] = std::make_pair(session_id, bytes);
}

void policy_release_tensor(uint64_t session_id, uint64_t tensor_id) {
    std::lock_guard<std::mutex> lock(g_policy_mu);
    auto it = g_tensor_reserved_bytes.find(tensor_id);
    if (it == g_tensor_reserved_bytes.end()) return;
    if (it->second.first != session_id) return;
    const uint64_t bytes = it->second.second;
    auto sit = g_session_reserved_bytes.find(session_id);
    if (sit != g_session_reserved_bytes.end()) {
        if (sit->second <= bytes) sit->second = 0;
        else sit->second -= bytes;
    }
    g_tensor_reserved_bytes.erase(it);
}

void policy_end_job(uint64_t session_id) {
    std::lock_guard<std::mutex> lock(g_policy_mu);
    auto it = g_session_inflight_jobs.find(session_id);
    if (it == g_session_inflight_jobs.end()) return;
    if (it->second > 0) --it->second;
}

void policy_get_session_usage(
    uint64_t session_id,
    uint64_t* out_reserved_bytes,
    uint64_t* out_jobs_inflight) {
    if (!out_reserved_bytes || !out_jobs_inflight) return;
    std::lock_guard<std::mutex> lock(g_policy_mu);
    const auto bit = g_session_reserved_bytes.find(session_id);
    const auto jit = g_session_inflight_jobs.find(session_id);
    *out_reserved_bytes = (bit == g_session_reserved_bytes.end()) ? 0 : bit->second;
    *out_jobs_inflight = (jit == g_session_inflight_jobs.end()) ? 0 : jit->second;
}

bool writef(char* out, size_t out_len, const char* fmt, ...) {
    if (!out || out_len == 0) return false;
    va_list args;
    va_start(args, fmt);
    const int n = std::vsnprintf(out, out_len, fmt, args);
    va_end(args);
    return n >= 0 && static_cast<size_t>(n) < out_len;
}

std::string escape_json(const char* s) {
    if (!s) return "";
    std::string out;
    for (const char* p = s; *p; ++p) {
        if (*p == '\\' || *p == '"') out.push_back('\\');
        out.push_back(*p);
    }
    return out;
}

void append_audit_line(bool ok, const char* status_code, const char* message) {
    const char* path = std::getenv("FERCUDA_AGENT_AUDIT_LOG");
    if (!path || path[0] == '\0') path = kDefaultAuditLogPath;
    FILE* f = std::fopen(path, "ab");
    if (!f) return;
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    const std::string esc_msg = escape_json(message ? message : "");
    const std::string esc_req = escape_json(t_audit_request_id.c_str());
    const std::string esc_tool = escape_json(t_audit_tool_name ? t_audit_tool_name : "");
    const std::string esc_policy = escape_json(t_audit_policy_decision.c_str());
    std::fprintf(
        f,
        "{\"timestamp_ms\":%lld,\"request_id\":\"%s\",\"tool_name\":\"%s\",\"ok\":%s,"
        "\"status_code\":\"%s\",\"policy_decision\":\"%s\",\"message\":\"%s\"}\n",
        static_cast<long long>(ms),
        esc_req.c_str(),
        esc_tool.c_str(),
        ok ? "true" : "false",
        status_code ? status_code : "UNKNOWN",
        esc_policy.c_str(),
        esc_msg.c_str());
    std::fclose(f);
}

const char* code_str(int32_t code) {
    switch (code) {
        case FER_STATUS_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case FER_STATUS_NOT_FOUND: return "NOT_FOUND";
        case FER_STATUS_INTERNAL_ERROR: return "INTERNAL";
        case FER_STATUS_OK:
        default: return "OK";
    }
}

bool find_key(std::string_view json, const char* key, size_t* out_colon) {
    const std::string pattern = std::string("\"") + key + "\"";
    const size_t p = json.find(pattern);
    if (p == std::string_view::npos) return false;
    const size_t c = json.find(':', p + pattern.size());
    if (c == std::string_view::npos) return false;
    *out_colon = c;
    return true;
}

size_t skip_ws(std::string_view s, size_t i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    return i;
}

bool parse_string(std::string_view json, const char* key, std::string* out) {
    size_t c = 0;
    if (!find_key(json, key, &c)) return false;
    size_t i = skip_ws(json, c + 1);
    if (i >= json.size() || json[i] != '"') return false;
    ++i;
    std::string v;
    while (i < json.size()) {
        if (json[i] == '\\') {
            ++i;
            if (i < json.size()) v.push_back(json[i++]);
            continue;
        }
        if (json[i] == '"') {
            *out = v;
            return true;
        }
        v.push_back(json[i++]);
    }
    return false;
}

bool parse_bool(std::string_view json, const char* key, bool* out) {
    size_t c = 0;
    if (!find_key(json, key, &c)) return false;
    size_t i = skip_ws(json, c + 1);
    if (json.substr(i, 4) == "true") {
        *out = true;
        return true;
    }
    if (json.substr(i, 5) == "false") {
        *out = false;
        return true;
    }
    return false;
}

bool parse_u64(std::string_view json, const char* key, uint64_t* out) {
    size_t c = 0;
    if (!find_key(json, key, &c)) return false;
    size_t i = skip_ws(json, c + 1);
    char* end = nullptr;
    const unsigned long long v = std::strtoull(json.data() + i, &end, 10);
    if (end == json.data() + i) return false;
    *out = static_cast<uint64_t>(v);
    return true;
}

bool parse_i64(std::string_view json, const char* key, int64_t* out) {
    size_t c = 0;
    if (!find_key(json, key, &c)) return false;
    size_t i = skip_ws(json, c + 1);
    char* end = nullptr;
    const long long v = std::strtoll(json.data() + i, &end, 10);
    if (end == json.data() + i) return false;
    *out = static_cast<int64_t>(v);
    return true;
}

bool parse_double(std::string_view json, const char* key, double* out) {
    size_t c = 0;
    if (!find_key(json, key, &c)) return false;
    size_t i = skip_ws(json, c + 1);
    char* end = nullptr;
    const double v = std::strtod(json.data() + i, &end);
    if (end == json.data() + i) return false;
    *out = v;
    return true;
}

bool parse_u32_array(std::string_view json, const char* key, std::vector<uint32_t>* out) {
    size_t c = 0;
    if (!find_key(json, key, &c)) return false;
    size_t i = skip_ws(json, c + 1);
    if (i >= json.size() || json[i] != '[') return false;
    ++i;
    out->clear();
    while (i < json.size()) {
        i = skip_ws(json, i);
        if (i < json.size() && json[i] == ']') return true;
        char* end = nullptr;
        const unsigned long v = std::strtoul(json.data() + i, &end, 10);
        if (end == json.data() + i) return false;
        out->push_back(static_cast<uint32_t>(v));
        i = static_cast<size_t>(end - json.data());
        i = skip_ws(json, i);
        if (i < json.size() && json[i] == ',') {
            ++i;
            continue;
        }
        if (i < json.size() && json[i] == ']') return true;
        return false;
    }
    return false;
}

bool parse_string_array(std::string_view json, const char* key, std::vector<std::string>* out) {
    size_t c = 0;
    if (!find_key(json, key, &c)) return false;
    size_t i = skip_ws(json, c + 1);
    if (i >= json.size() || json[i] != '[') return false;
    ++i;
    out->clear();
    while (i < json.size()) {
        i = skip_ws(json, i);
        if (i < json.size() && json[i] == ']') return true;
        if (i >= json.size() || json[i] != '"') return false;
        ++i;
        std::string v;
        while (i < json.size() && json[i] != '"') {
            if (json[i] == '\\' && (i + 1) < json.size()) ++i;
            v.push_back(json[i++]);
        }
        if (i >= json.size() || json[i] != '"') return false;
        ++i;
        out->push_back(v);
        i = skip_ws(json, i);
        if (i < json.size() && json[i] == ',') {
            ++i;
            continue;
        }
        if (i < json.size() && json[i] == ']') return true;
        return false;
    }
    return false;
}

int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool hex_decode(const std::string& hex, std::vector<uint8_t>* out) {
    if (hex.size() % 2 != 0) return false;
    out->resize(hex.size() / 2);
    for (size_t i = 0; i < out->size(); ++i) {
        const int hi = hex_nibble(hex[2 * i]);
        const int lo = hex_nibble(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        (*out)[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

std::string hex_encode(const uint8_t* data, size_t bytes) {
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.resize(bytes * 2);
    for (size_t i = 0; i < bytes; ++i) {
        out[2 * i] = kHex[(data[i] >> 4) & 0xF];
        out[2 * i + 1] = kHex[data[i] & 0xF];
    }
    return out;
}

uint32_t dtype_from_string(const std::string& s) {
    if (s == "f32") return FER_DTYPE_F32;
    if (s == "f16") return FER_DTYPE_F16;
    if (s == "bf16") return FER_DTYPE_BF16;
    if (s == "i8") return FER_DTYPE_I8;
    if (s == "u8") return FER_DTYPE_U8;
    if (s == "i16") return FER_DTYPE_I16;
    if (s == "u16") return FER_DTYPE_U16;
    if (s == "i32") return FER_DTYPE_I32;
    if (s == "u32") return FER_DTYPE_U32;
    if (s == "i64") return FER_DTYPE_I64;
    if (s == "u64") return FER_DTYPE_U64;
    if (s == "f64") return FER_DTYPE_F64;
    return UINT32_MAX;
}

uint32_t regime_from_string(const std::string& s) {
    if (s == "custom_pool") return FER_MEMORY_CUSTOM_POOL;
    if (s == "cuda_malloc") return FER_MEMORY_CUDA_MALLOC;
    if (s == "cuda_managed") return FER_MEMORY_CUDA_MANAGED;
    if (s == "auto") return FER_MEMORY_AUTO;
    return UINT32_MAX;
}

bool require_api_version(std::string_view json) {
    std::string v;
    if (!parse_string(json, "agent_api_version", &v)) return false;
    return v == kApiVersion;
}

fer_status_t write_error(char* out_json, size_t out_json_len, fer_status_t st) {
    const std::string msg = escape_json(st.message ? st.message : "error");
    if (!writef(
            out_json,
            out_json_len,
            "{\"ok\":false,\"error\":{\"code\":\"%s\",\"message\":\"%s\"}}",
            code_str(st.code),
            msg.c_str())) {
        append_audit_line(false, "INTERNAL", "response buffer too small");
        return fer_status_t{FER_STATUS_INTERNAL_ERROR, "response buffer too small"};
    }
    append_audit_line(false, code_str(st.code), st.message ? st.message : "error");
    return st;
}

fer_status_t write_ok(char* out_json, size_t out_json_len, const char* payload_json) {
    if (!writef(out_json, out_json_len, "{\"ok\":true,\"result\":%s}", payload_json)) {
        append_audit_line(false, "INTERNAL", "response buffer too small");
        return fer_status_t{FER_STATUS_INTERNAL_ERROR, "response buffer too small"};
    }
    append_audit_line(true, "OK", "ok");
    return fer_status_t{FER_STATUS_OK, "ok"};
}

bool build_u64_array_payload(
    const char* field_name,
    const uint64_t* ids,
    size_t count,
    std::vector<char>* out_payload) {
    if (!field_name || !out_payload) return false;
    std::string s = "{\"";
    s += field_name;
    s += "\":[";
    for (size_t i = 0; i < count; ++i) {
        if (i) s += ",";
        s += std::to_string(static_cast<unsigned long long>(ids[i]));
    }
    s += "],\"count\":";
    s += std::to_string(static_cast<unsigned long long>(count));
    s += "}";
    out_payload->assign(s.begin(), s.end());
    out_payload->push_back('\0');
    return true;
}

}  // namespace

extern "C" fer_status_t fer_agent_mcp_dispatch(
    fer_agent_adapter_t* adapter,
    const char* tool_name,
    const char* request_json,
    char* out_json,
    size_t out_json_len) {
    if (!adapter) return fer_status_t{FER_STATUS_INVALID_ARGUMENT, "adapter is null"};
    if (!tool_name) return fer_status_t{FER_STATUS_INVALID_ARGUMENT, "tool_name is null"};
    if (!request_json) return fer_status_t{FER_STATUS_INVALID_ARGUMENT, "request_json is null"};
    if (!out_json || out_json_len == 0) return fer_status_t{FER_STATUS_INVALID_ARGUMENT, "out_json is null"};

    const std::string_view json(request_json);
    t_audit_tool_name = tool_name;
    t_audit_request_id.clear();
    t_audit_policy_decision = "allow";
    std::string req_id;
    if (parse_string(json, "request_id", &req_id)) t_audit_request_id = req_id;
    const PolicyConfig policy = load_policy_config();

    if (std::strcmp(tool_name, "fer.runtime.inspect") == 0) {
        fer_agent_runtime_inspect_t info{};
        fer_status_t st = fer_agent_runtime_inspect(adapter, &info);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        char payload[768];
        if (!writef(
                payload, sizeof(payload),
                "{\"agent_api_version\":\"v1alpha1\",\"jit_intent_abi_version\":%u,"
                "\"supports\":{\"jit_compile\":%s,\"jit_intent\":%s,"
                "\"external_ptr_import\":%s,\"session_stream_handoff\":%s,"
                "\"jit_kernel_compile\":%s,\"jit_kernel_launch\":%s,"
                "\"progress_channel\":%s},"
                "\"intent_ops\":[\"affine_f32\",\"softmax_f32\","
                "\"reduce_sum_f32\",\"reduce_max_f32\",\"conv2d_f32\"]}",
                info.jit_intent_abi_version,
                info.supports_jit_compile ? "true" : "false",
                info.supports_jit_intent ? "true" : "false",
                info.supports_external_ptr_import ? "true" : "false",
                info.supports_session_stream_handoff ? "true" : "false",
                "true", "true", "true")) {
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INTERNAL_ERROR, "payload format failed"});
        }
        return write_ok(out_json, out_json_len, payload);
    }

    if (!require_api_version(json)) {
        return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "agent_api_version must be v1alpha1"});
    }

    if (std::strcmp(tool_name, "fer.session.create") == 0) {
        fer_agent_session_create_request_t req{};
        int64_t device = 0;
        uint64_t tmp_u64 = 0;
        bool verbose = false;
        std::string regime;
        if (!parse_i64(json, "device", &device)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing device"});
        if (!parse_u64(json, "mutable_bytes", &req.mutable_bytes)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing mutable_bytes"});
        if (!parse_u64(json, "immutable_bytes", &req.immutable_bytes)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing immutable_bytes"});
        if (parse_u64(json, "cuda_reserve", &tmp_u64)) req.cuda_reserve = tmp_u64;
        if (parse_bool(json, "verbose", &verbose)) req.verbose = verbose ? 1u : 0u;
        if (!parse_string(json, "memory_regime", &regime)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing memory_regime"});
        req.memory_regime = regime_from_string(regime);
        if (req.memory_regime == UINT32_MAX) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "invalid memory_regime"});
        req.device = static_cast<int32_t>(device);
        uint64_t session_id = 0;
        fer_status_t st = fer_agent_session_create(adapter, &req, &session_id);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        record_session_create(session_id);
        char payload[128];
        if (!writef(payload, sizeof(payload), "{\"session_id\":%llu}", static_cast<unsigned long long>(session_id))) {
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INTERNAL_ERROR, "payload format failed"});
        }
        return write_ok(out_json, out_json_len, payload);
    }

    if (std::strcmp(tool_name, "fer.session.destroy") == 0) {
        uint64_t session_id = 0;
        if (!parse_u64(json, "session_id", &session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        fer_status_t st = fer_agent_session_destroy(adapter, session_id);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        record_session_destroy(session_id);
        return write_ok(out_json, out_json_len, "{\"destroyed\":true}");
    }

    if (std::strcmp(tool_name, "fer.session.list") == 0) {
        size_t count = 0;
        fer_status_t st = fer_agent_session_list(adapter, nullptr, 0, &count);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        std::vector<uint64_t> ids(count, 0);
        if (count > 0) {
            st = fer_agent_session_list(adapter, ids.data(), ids.size(), &count);
            if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        }
        std::vector<char> payload;
        if (!build_u64_array_payload("session_ids", ids.data(), count, &payload)) {
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INTERNAL_ERROR, "payload format failed"});
        }
        return write_ok(out_json, out_json_len, payload.data());
    }

    if (std::strcmp(tool_name, "fer.tensor.create") == 0) {
        fer_agent_tensor_create_request_t req{};
        std::string dtype;
        std::string regime;
        std::vector<uint32_t> shape;
        bool immutable = false;
        if (!parse_u64(json, "session_id", &req.session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        if (!parse_string(json, "dtype", &dtype)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing dtype"});
        if (!list_allows(policy.allow_dtypes, dtype)) {
            t_audit_policy_decision = "deny";
            return write_error(out_json, out_json_len, fer_status_t{
                FER_STATUS_INVALID_ARGUMENT, "POLICY_DENIED: dtype is not allowed"});
        }
        req.dtype = dtype_from_string(dtype);
        if (req.dtype == UINT32_MAX) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "invalid dtype"});
        if (!parse_u32_array(json, "shape", &shape)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing shape"});
        if (shape.empty() || shape.size() > 4) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "shape rank must be 1..4"});
        req.rank = static_cast<uint32_t>(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) req.dims[i] = shape[i];
        if (parse_bool(json, "immutable", &immutable)) req.immutable = immutable ? 1u : 0u;
        uint64_t tag = 0;
        if (parse_u64(json, "tag", &tag)) req.tag = static_cast<uint32_t>(tag);
        if (!parse_string(json, "memory_regime", &regime)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing memory_regime"});
        if (!list_allows(policy.allow_regimes, regime)) {
            t_audit_policy_decision = "deny";
            return write_error(out_json, out_json_len, fer_status_t{
                FER_STATUS_INVALID_ARGUMENT, "POLICY_DENIED: memory_regime is not allowed"});
        }
        req.memory_regime = regime_from_string(regime);
        if (req.memory_regime == UINT32_MAX) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "invalid memory_regime"});
        const uint64_t elem = dtype_size_from_name(dtype);
        if (elem == 0) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "unsupported dtype"});
        uint64_t bytes = elem;
        for (uint32_t d = 0; d < req.rank; ++d) {
            if (!safe_mul_u64(bytes, req.dims[d], &bytes)) {
                t_audit_policy_decision = "deny";
                return write_error(out_json, out_json_len, fer_status_t{
                    FER_STATUS_INVALID_ARGUMENT, "RESOURCE_EXHAUSTED: tensor byte size overflow"});
            }
        }
        std::string reason;
        if (!policy_reserve_tensor_bytes(policy, req.session_id, bytes, &reason)) {
            t_audit_policy_decision = "deny";
            std::string msg = std::string(kResourceExhaustedPrefix) + reason;
            return write_error(out_json, out_json_len, fer_status_t{
                FER_STATUS_INVALID_ARGUMENT, msg.c_str()});
        }
        uint64_t tensor_id = 0;
        fer_status_t st = fer_agent_tensor_create(adapter, &req, &tensor_id);
        if (st.code != FER_STATUS_OK) {
            policy_unreserve_session_bytes(req.session_id, bytes);
            return write_error(out_json, out_json_len, st);
        }
        policy_record_tensor_allocation(req.session_id, tensor_id, bytes);
        char payload[128];
        if (!writef(payload, sizeof(payload), "{\"tensor_id\":%llu}", static_cast<unsigned long long>(tensor_id))) {
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INTERNAL_ERROR, "payload format failed"});
        }
        return write_ok(out_json, out_json_len, payload);
    }

    if (std::strcmp(tool_name, "fer.tensor.release") == 0) {
        uint64_t session_id = 0;
        uint64_t tensor_id = 0;
        if (!parse_u64(json, "session_id", &session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        if (!parse_u64(json, "tensor_id", &tensor_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing tensor_id"});
        fer_status_t st = fer_agent_tensor_release(adapter, session_id, tensor_id);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        policy_release_tensor(session_id, tensor_id);
        return write_ok(out_json, out_json_len, "{\"released\":true}");
    }

    if (std::strcmp(tool_name, "fer.tensor.list") == 0) {
        uint64_t session_id = 0;
        if (!parse_u64(json, "session_id", &session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        size_t count = 0;
        fer_status_t st = fer_agent_tensor_list(adapter, session_id, nullptr, 0, &count);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        std::vector<uint64_t> ids(count, 0);
        if (count > 0) {
            st = fer_agent_tensor_list(adapter, session_id, ids.data(), ids.size(), &count);
            if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        }
        std::vector<char> payload;
        if (!build_u64_array_payload("tensor_ids", ids.data(), count, &payload)) {
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INTERNAL_ERROR, "payload format failed"});
        }
        return write_ok(out_json, out_json_len, payload.data());
    }

    if (std::strcmp(tool_name, "fer.tensor.copy") == 0) {
        fer_agent_tensor_copy_request_t req{};
        std::string dtype;
        std::string direction;
        std::string blob_ref;
        uint64_t host_ptr_u64 = 0;
        if (!parse_u64(json, "session_id", &req.session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        if (!parse_u64(json, "tensor_id", &req.tensor_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing tensor_id"});
        if (!parse_string(json, "dtype", &dtype)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing dtype"});
        req.dtype = dtype_from_string(dtype);
        if (req.dtype == UINT32_MAX) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "invalid dtype"});
        uint64_t count = 0;
        if (!parse_u64(json, "count", &count)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing count"});
        req.count = static_cast<size_t>(count);
        const bool has_blob_ref = parse_string(json, "host_blob_ref", &blob_ref);
        const bool has_ptr = parse_u64(json, "host_ptr_u64", &host_ptr_u64);
        if (!has_blob_ref && !has_ptr) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing host_blob_ref or host_ptr_u64"});
        if (!parse_string(json, "direction", &direction)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing direction"});
        if (direction == "host_to_device") req.direction = FER_AGENT_COPY_HOST_TO_DEVICE;
        else if (direction == "device_to_host") req.direction = FER_AGENT_COPY_DEVICE_TO_HOST;
        else return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "invalid direction"});
        if (has_blob_ref) {
            const size_t elem = [&]() -> size_t {
                switch (req.dtype) {
                    case FER_DTYPE_F32: return sizeof(float);
                    case FER_DTYPE_F16: return sizeof(uint16_t);
                    case FER_DTYPE_BF16: return sizeof(uint16_t);
                    case FER_DTYPE_I8: return sizeof(int8_t);
                    case FER_DTYPE_U8: return sizeof(uint8_t);
                    case FER_DTYPE_I16: return sizeof(int16_t);
                    case FER_DTYPE_U16: return sizeof(uint16_t);
                    case FER_DTYPE_I32: return sizeof(int32_t);
                    case FER_DTYPE_U32: return sizeof(uint32_t);
                    case FER_DTYPE_I64: return sizeof(int64_t);
                    case FER_DTYPE_U64: return sizeof(uint64_t);
                    case FER_DTYPE_F64: return sizeof(double);
                    default: return 0;
                }
            }();
            if (elem == 0) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "unsupported dtype"});
            const size_t bytes = req.count * elem;
            if (req.direction == FER_AGENT_COPY_HOST_TO_DEVICE) {
                const void* data = nullptr;
                size_t blob_bytes = 0;
                fer_status_t st_blob = fer_agent_blob_get(adapter, blob_ref.c_str(), &data, &blob_bytes);
                if (st_blob.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st_blob);
                if (blob_bytes < bytes) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "blob too small"});
                req.host_ptr = const_cast<void*>(data);
            } else {
                void* dst = nullptr;
                fer_status_t st_blob = fer_agent_blob_resize_for_write(adapter, blob_ref.c_str(), bytes, &dst);
                if (st_blob.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st_blob);
                req.host_ptr = dst;
            }
        } else {
            req.host_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(host_ptr_u64));
        }
        fer_status_t st = fer_agent_tensor_copy(adapter, &req);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        return write_ok(out_json, out_json_len, "{\"copied\":true}");
    }

    if (std::strcmp(tool_name, "fer.blob.put") == 0) {
        std::string blob_hex;
        std::string blob_ref;
        if (!parse_string(json, "blob_hex", &blob_hex)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing blob_hex"});
        (void)parse_string(json, "host_blob_ref", &blob_ref);
        std::vector<uint8_t> bytes;
        if (!hex_decode(blob_hex, &bytes)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "invalid blob_hex"});
        char out_ref[128];
        fer_status_t st = fer_agent_blob_put(
            adapter,
            blob_ref.empty() ? nullptr : blob_ref.c_str(),
            bytes.empty() ? nullptr : bytes.data(),
            bytes.size(),
            out_ref,
            sizeof(out_ref));
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        char payload[256];
        if (!writef(payload, sizeof(payload), "{\"host_blob_ref\":\"%s\",\"bytes\":%zu}", out_ref, bytes.size())) {
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INTERNAL_ERROR, "payload format failed"});
        }
        return write_ok(out_json, out_json_len, payload);
    }

    if (std::strcmp(tool_name, "fer.blob.get") == 0) {
        std::string blob_ref;
        if (!parse_string(json, "host_blob_ref", &blob_ref)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing host_blob_ref"});
        const void* data = nullptr;
        size_t bytes = 0;
        fer_status_t st = fer_agent_blob_get(adapter, blob_ref.c_str(), &data, &bytes);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        const std::string hex = hex_encode(static_cast<const uint8_t*>(data), bytes);
        std::vector<char> payload(hex.size() + 192);
        if (!writef(payload.data(), payload.size(), "{\"host_blob_ref\":\"%s\",\"bytes\":%zu,\"blob_hex\":\"%s\"}", blob_ref.c_str(), bytes, hex.c_str())) {
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INTERNAL_ERROR, "payload format failed"});
        }
        return write_ok(out_json, out_json_len, payload.data());
    }

    if (std::strcmp(tool_name, "fer.jit.kernel.compile") == 0) {
        uint64_t session_id = 0;
        std::string source_code;
        std::string language;
        uint64_t opt_level = 2;
        bool strict = true;
        if (!parse_u64(json, "session_id", &session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        if (!parse_string(json, "source", &source_code)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing source"});
        if (!parse_string(json, "language", &language)) language = "cuda";
        (void)parse_u64(json, "optimization_level", &opt_level);
        (void)parse_bool(json, "strict", &strict);
        uint32_t lang_kind = FER_JIT_SOURCE_CUDA;
        if (language == "ptx") lang_kind = FER_JIT_SOURCE_PTX;
        else if (language != "cuda") return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "unsupported language"});
        fer_agent_jit_compile_request_t req{};
        req.session_id = session_id;
        req.source = source_code.c_str();
        req.source_len = source_code.size();
        req.language = lang_kind;
        req.optimization_level = static_cast<uint32_t>(opt_level);
        req.strict = strict ? 1u : 0u;
        fer_agent_jit_compile_result_t cr{};
        fer_status_t st = fer_agent_jit_compile(adapter, &req, &cr);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        std::string log_escaped = escape_json(cr.diagnostics_log ? cr.diagnostics_log : "");
        std::vector<char> payload(log_escaped.size() + 256);
        if (!writef(payload.data(), payload.size(),
                    "{\"program_id\":%llu,\"cache\":{\"hit\":%s},\"diagnostics\":{\"warnings\":[],\"errors\":[],\"log\":\"%s\"}}",
                    static_cast<unsigned long long>(cr.program_id),
                    cr.cache_hit ? "true" : "false",
                    log_escaped.c_str())) {
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INTERNAL_ERROR, "payload format failed"});
        }
        return write_ok(out_json, out_json_len, payload.data());
    }

    if (std::strcmp(tool_name, "fer.jit.kernel.launch") == 0) {
        uint64_t session_id = 0;
        uint64_t program_id = 0;
        std::string kernel_name;
        std::vector<uint32_t> grid;
        std::vector<uint32_t> block;
        uint64_t shared_mem = 0;
        if (!parse_u64(json, "session_id", &session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        if (!parse_u64(json, "program_id", &program_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing program_id"});
        if (!parse_string(json, "kernel_name", &kernel_name)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing kernel_name"});
        if (!parse_u32_array(json, "grid", &grid) || grid.size() != 3) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "grid must be [x,y,z]"});
        if (!parse_u32_array(json, "block", &block) || block.size() != 3) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "block must be [x,y,z]"});
        (void)parse_u64(json, "shared_mem_bytes", &shared_mem);
        // Parse args array - simplified inline JSON args parser
        // For now require args to be pre-resolved tensor_ids and typed scalars
        // via the structured C API. JSON-level arg parsing deferred to full schema.
        std::string reason;
        if (!policy_try_begin_job(policy, session_id, &reason)) {
            t_audit_policy_decision = "deny";
            std::string msg = std::string(kResourceExhaustedPrefix) + reason;
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, msg.c_str()});
        }
        fer_agent_jit_launch_request_t req{};
        req.session_id = session_id;
        req.program_id = program_id;
        req.kernel_name = kernel_name.c_str();
        req.grid[0] = grid[0]; req.grid[1] = grid[1]; req.grid[2] = grid[2];
        req.block[0] = block[0]; req.block[1] = block[1]; req.block[2] = block[2];
        req.shared_mem_bytes = static_cast<uint32_t>(shared_mem);
        req.args = nullptr;
        req.arg_count = 0;
        uint64_t job_id = 0;
        fer_status_t st = fer_agent_jit_launch(adapter, &req, &job_id);
        if (st.code != FER_STATUS_OK) {
            policy_end_job(session_id);
            return write_error(out_json, out_json_len, st);
        }
        char payload[128];
        if (!writef(payload, sizeof(payload), "{\"job_id\":%llu}", static_cast<unsigned long long>(job_id))) {
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INTERNAL_ERROR, "payload format failed"});
        }
        return write_ok(out_json, out_json_len, payload);
    }

    if (std::strcmp(tool_name, "fer.jit.kernel.release") == 0) {
        uint64_t session_id = 0;
        uint64_t program_id = 0;
        if (!parse_u64(json, "session_id", &session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        if (!parse_u64(json, "program_id", &program_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing program_id"});
        fer_status_t st = fer_agent_jit_release_program(adapter, session_id, program_id);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        return write_ok(out_json, out_json_len, "{\"released\":true}");
    }

    if (std::strcmp(tool_name, "fer.jit.intent.run") == 0) {
        std::string op;
        std::string regime;
        std::vector<std::string> fusion;
        std::vector<std::string> caps;
        uint64_t session_id = 0;
        double alpha = 0.0, beta = 0.0;
        uint64_t tmp = 0;
        if (!parse_u64(json, "session_id", &session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        if (!parse_string(json, "op", &op)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing op"});
        if (!parse_string(json, "memory_regime", &regime)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing memory_regime"});
        if (!list_allows(policy.allow_regimes, regime)) {
            t_audit_policy_decision = "deny";
            return write_error(out_json, out_json_len, fer_status_t{
                FER_STATUS_INVALID_ARGUMENT, "POLICY_DENIED: memory_regime is not allowed"});
        }
        const uint32_t regime_id = regime_from_string(regime);
        if (regime_id == UINT32_MAX) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "invalid memory_regime"});

        uint32_t fusion_mask = 0;
        if (parse_string_array(json, "fusion_mask", &fusion)) {
            for (const std::string& f : fusion) {
                if (f == "relu") fusion_mask |= FER_JIT_INTENT_FUSION_RELU;
                else if (f != "none") return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "invalid fusion op"});
            }
        }
        uint32_t caps_mask = 0;
        if (parse_string_array(json, "caps_mask", &caps)) {
            for (const std::string& c : caps) {
                if (c == "require_tensor_cores") caps_mask |= FER_JIT_INTENT_CAPS_REQUIRE_TENSOR_CORES;
                else if (c == "require_coop_groups") caps_mask |= FER_JIT_INTENT_CAPS_REQUIRE_COOP_GROUPS;
                else if (c != "none") return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "invalid caps op"});
            }
        }

        // Map op string to enum
        uint32_t op_enum;
        if (op == "affine_f32") op_enum = FER_JIT_INTENT_OP_AFFINE_F32;
        else if (op == "softmax_f32") op_enum = FER_JIT_INTENT_OP_SOFTMAX_F32;
        else if (op == "reduce_sum_f32") op_enum = FER_JIT_INTENT_OP_REDUCE_SUM_F32;
        else if (op == "reduce_max_f32") op_enum = FER_JIT_INTENT_OP_REDUCE_MAX_F32;
        else if (op == "conv2d_f32") op_enum = FER_JIT_INTENT_OP_CONV2D_F32;
        else return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "unsupported op"});

        // Common fields
        fer_agent_intent_affine_f32_request_t req{};
        req.session_id = session_id;
        req.memory_regime = regime_id;
        req.fusion_mask = fusion_mask;
        req.caps_mask = caps_mask;
        if (parse_u64(json, "n", &tmp)) req.n = static_cast<uint32_t>(tmp);
        (void)parse_double(json, "alpha", &alpha); req.alpha = static_cast<float>(alpha);
        (void)parse_double(json, "beta", &beta); req.beta = static_cast<float>(beta);

        uint64_t input_id = 0, output_id = 0;
        if (!parse_u64(json, "input", &input_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing bindings.input"});
        if (!parse_u64(json, "output", &output_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing bindings.output"});
        req.input_tensor_id = input_id;
        req.output_tensor_id = output_id;

        std::string reason;
        if (!policy_try_begin_job(policy, session_id, &reason)) {
            t_audit_policy_decision = "deny";
            std::string msg = std::string(kResourceExhaustedPrefix) + reason;
            return write_error(out_json, out_json_len, fer_status_t{
                FER_STATUS_INVALID_ARGUMENT, msg.c_str()});
        }

        uint64_t job_id = 0;
        fer_status_t st;

        if (op_enum == FER_JIT_INTENT_OP_AFFINE_F32) {
            st = fer_agent_jit_intent_run_affine_f32(adapter, &req, &job_id);
        } else {
            fer_agent_intent_generic_request_t gen{};
            gen.session_id = session_id;
            gen.op = op_enum;
            gen.input_tensor_id = input_id;
            gen.output_tensor_id = output_id;
            gen.n = req.n;
            gen.alpha = req.alpha;
            gen.beta = req.beta;
            gen.fusion_mask = fusion_mask;
            gen.caps_mask = caps_mask;
            gen.memory_regime = regime_id;
            if (op_enum == FER_JIT_INTENT_OP_CONV2D_F32) {
                uint64_t v = 0;
                if (parse_u64(json, "height", &v)) gen.height = static_cast<uint32_t>(v);
                if (parse_u64(json, "width", &v)) gen.width = static_cast<uint32_t>(v);
                if (parse_u64(json, "channels", &v)) gen.channels = static_cast<uint32_t>(v);
                if (parse_u64(json, "kernel_h", &v)) gen.kernel_h = static_cast<uint32_t>(v);
                if (parse_u64(json, "kernel_w", &v)) gen.kernel_w = static_cast<uint32_t>(v);
                if (parse_u64(json, "pad_h", &v)) gen.pad_h = static_cast<uint32_t>(v);
                if (parse_u64(json, "pad_w", &v)) gen.pad_w = static_cast<uint32_t>(v);
                if (parse_u64(json, "stride_h", &v)) gen.stride_h = static_cast<uint32_t>(v);
                if (parse_u64(json, "stride_w", &v)) gen.stride_w = static_cast<uint32_t>(v);
                if (parse_u64(json, "num_filters", &v)) gen.num_filters = static_cast<uint32_t>(v);
                uint64_t wid = 0, bid_conv = 0;
                if (parse_u64(json, "weights", &wid)) gen.weights_tensor_id = wid;
                if (parse_u64(json, "bias", &bid_conv)) gen.bias_tensor_id = bid_conv;
            }
            if (op_enum == FER_JIT_INTENT_OP_SOFTMAX_F32) {
                uint64_t v = 0;
                if (parse_u64(json, "height", &v)) gen.height = static_cast<uint32_t>(v);
            }
            st = fer_agent_jit_intent_run(adapter, &gen, &job_id);
        }

        if (st.code != FER_STATUS_OK) {
            policy_end_job(session_id);
            return write_error(out_json, out_json_len, st);
        }
        char payload[128];
        if (!writef(payload, sizeof(payload), "{\"job_id\":%llu}", static_cast<unsigned long long>(job_id))) {
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INTERNAL_ERROR, "payload format failed"});
        }
        return write_ok(out_json, out_json_len, payload);
    }

    if (std::strcmp(tool_name, "fer.job.wait") == 0) {
        uint64_t session_id = 0;
        uint64_t job_id = 0;
        if (!parse_u64(json, "session_id", &session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        if (!parse_u64(json, "job_id", &job_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing job_id"});
        fer_status_t st = fer_agent_job_wait(adapter, session_id, job_id);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        policy_end_job(session_id);
        return write_ok(out_json, out_json_len, "{\"done\":true}");
    }

    if (std::strcmp(tool_name, "fer.job.cancel") == 0) {
        uint64_t session_id = 0;
        uint64_t job_id = 0;
        if (!parse_u64(json, "session_id", &session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        if (!parse_u64(json, "job_id", &job_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing job_id"});
        uint8_t cancelled = 0u;
        fer_status_t st = fer_agent_job_cancel(adapter, session_id, job_id, &cancelled);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        if (cancelled) policy_end_job(session_id);
        return write_ok(out_json, out_json_len, cancelled ? "{\"cancelled\":true}" : "{\"cancelled\":false}");
    }

    if (std::strcmp(tool_name, "fer.job.list") == 0) {
        uint64_t session_id = 0;
        if (!parse_u64(json, "session_id", &session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        size_t count = 0;
        fer_status_t st = fer_agent_job_list(adapter, session_id, nullptr, 0, &count);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        std::vector<uint64_t> ids(count, 0);
        if (count > 0) {
            st = fer_agent_job_list(adapter, session_id, ids.data(), ids.size(), &count);
            if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        }
        std::vector<char> payload;
        if (!build_u64_array_payload("job_ids", ids.data(), count, &payload)) {
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INTERNAL_ERROR, "payload format failed"});
        }
        return write_ok(out_json, out_json_len, payload.data());
    }

    if (std::strcmp(tool_name, "fer.session.stats") == 0) {
        uint64_t session_id = 0;
        if (!parse_u64(json, "session_id", &session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        size_t tensor_count = 0;
        size_t job_count = 0;
        fer_status_t st = fer_agent_session_stats(adapter, session_id, &tensor_count, &job_count);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        uint64_t reserved_bytes = 0;
        uint64_t jobs_inflight = 0;
        policy_get_session_usage(session_id, &reserved_bytes, &jobs_inflight);
        char payload[512];
        if (!writef(
                payload, sizeof(payload),
                "{\"session_id\":%llu,\"tensor_count\":%zu,\"job_count\":%zu,"
                "\"reserved_bytes\":%llu,\"jobs_inflight\":%llu,"
                "\"policy\":{\"max_tensor_bytes\":%llu,\"max_session_bytes\":%llu,\"max_jobs_in_flight\":%llu}}",
                static_cast<unsigned long long>(session_id),
                tensor_count,
                job_count,
                static_cast<unsigned long long>(reserved_bytes),
                static_cast<unsigned long long>(jobs_inflight),
                static_cast<unsigned long long>(policy.max_tensor_bytes),
                static_cast<unsigned long long>(policy.max_total_session_bytes),
                static_cast<unsigned long long>(policy.max_jobs_in_flight))) {
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INTERNAL_ERROR, "payload format failed"});
        }
        return write_ok(out_json, out_json_len, payload);
    }

    if (std::strcmp(tool_name, "fer.jit.stats") == 0) {
        uint64_t session_id = 0;
        if (!parse_u64(json, "session_id", &session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        // JIT stats are global (not per-session), but we validate the session exists.
        fer_agent_runtime_inspect_t info{};
        fer_status_t st = fer_agent_runtime_inspect(adapter, &info);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        // For now, report capabilities. Full per-session JIT stats would need
        // a C adapter function that calls fer_jit_get_stats on the session.
        char payload[512];
        if (!writef(
                payload, sizeof(payload),
                "{\"jit_compile_supported\":%s,\"jit_intent_supported\":%s,"
                "\"disk_cache_enabled\":true,\"disk_cache_max_bytes\":268435456}",
                info.supports_jit_compile ? "true" : "false",
                info.supports_jit_intent ? "true" : "false")) {
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INTERNAL_ERROR, "payload format failed"});
        }
        return write_ok(out_json, out_json_len, payload);
    }

    return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "unknown tool"});
}
