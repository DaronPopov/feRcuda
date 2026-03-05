#include "fercuda/agent/mcp_json_adapter.h"

#include "fercuda/jit/intent/intent.h"

#include <cctype>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr const char* kApiVersion = "v1alpha1";

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
        return fer_status_t{FER_STATUS_INTERNAL_ERROR, "response buffer too small"};
    }
    return st;
}

fer_status_t write_ok(char* out_json, size_t out_json_len, const char* payload_json) {
    if (!writef(out_json, out_json_len, "{\"ok\":true,\"result\":%s}", payload_json)) {
        return fer_status_t{FER_STATUS_INTERNAL_ERROR, "response buffer too small"};
    }
    return fer_status_t{FER_STATUS_OK, "ok"};
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

    if (std::strcmp(tool_name, "fer.runtime.inspect") == 0) {
        fer_agent_runtime_inspect_t info{};
        fer_status_t st = fer_agent_runtime_inspect(adapter, &info);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        char payload[256];
        if (!writef(
                payload, sizeof(payload),
                "{\"agent_api_version\":\"v1alpha1\",\"jit_intent_abi_version\":%u,"
                "\"supports\":{\"jit_compile\":%s,\"jit_intent\":%s,"
                "\"external_ptr_import\":%s,\"session_stream_handoff\":%s}}",
                info.jit_intent_abi_version,
                info.supports_jit_compile ? "true" : "false",
                info.supports_jit_intent ? "true" : "false",
                info.supports_external_ptr_import ? "true" : "false",
                info.supports_session_stream_handoff ? "true" : "false")) {
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
        return write_ok(out_json, out_json_len, "{\"destroyed\":true}");
    }

    if (std::strcmp(tool_name, "fer.tensor.create") == 0) {
        fer_agent_tensor_create_request_t req{};
        std::string dtype;
        std::string regime;
        std::vector<uint32_t> shape;
        bool immutable = false;
        if (!parse_u64(json, "session_id", &req.session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        if (!parse_string(json, "dtype", &dtype)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing dtype"});
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
        req.memory_regime = regime_from_string(regime);
        if (req.memory_regime == UINT32_MAX) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "invalid memory_regime"});
        uint64_t tensor_id = 0;
        fer_status_t st = fer_agent_tensor_create(adapter, &req, &tensor_id);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
        char payload[128];
        if (!writef(payload, sizeof(payload), "{\"tensor_id\":%llu}", static_cast<unsigned long long>(tensor_id))) {
            return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INTERNAL_ERROR, "payload format failed"});
        }
        return write_ok(out_json, out_json_len, payload);
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

    if (std::strcmp(tool_name, "fer.jit.intent.run") == 0) {
        fer_agent_intent_affine_f32_request_t req{};
        std::string op;
        std::string regime;
        std::vector<std::string> fusion;
        std::vector<std::string> caps;
        double alpha = 0.0;
        double beta = 0.0;
        uint64_t tmp = 0;
        if (!parse_u64(json, "session_id", &req.session_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing session_id"});
        if (!parse_string(json, "op", &op)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing op"});
        if (op != "affine_f32") return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "only affine_f32 supported"});
        if (!parse_u64(json, "n", &tmp)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing n"});
        req.n = static_cast<uint32_t>(tmp);
        if (!parse_double(json, "alpha", &alpha)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing alpha"});
        if (!parse_double(json, "beta", &beta)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing beta"});
        req.alpha = static_cast<float>(alpha);
        req.beta = static_cast<float>(beta);
        if (!parse_string(json, "memory_regime", &regime)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing memory_regime"});
        req.memory_regime = regime_from_string(regime);
        if (req.memory_regime == UINT32_MAX) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "invalid memory_regime"});
        req.fusion_mask = 0;
        if (parse_string_array(json, "fusion_mask", &fusion)) {
            for (const std::string& f : fusion) {
                if (f == "relu") req.fusion_mask |= FER_JIT_INTENT_FUSION_RELU;
                else if (f == "none") {}
                else return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "invalid fusion op"});
            }
        }
        req.caps_mask = 0;
        if (parse_string_array(json, "caps_mask", &caps)) {
            for (const std::string& c : caps) {
                if (c == "require_tensor_cores") req.caps_mask |= FER_JIT_INTENT_CAPS_REQUIRE_TENSOR_CORES;
                else if (c == "require_coop_groups") req.caps_mask |= FER_JIT_INTENT_CAPS_REQUIRE_COOP_GROUPS;
                else if (c == "none") {}
                else return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "invalid caps op"});
            }
        }
        if (!parse_u64(json, "input", &req.input_tensor_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing bindings.input"});
        if (!parse_u64(json, "output", &req.output_tensor_id)) return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "missing bindings.output"});
        uint64_t job_id = 0;
        fer_status_t st = fer_agent_jit_intent_run_affine_f32(adapter, &req, &job_id);
        if (st.code != FER_STATUS_OK) return write_error(out_json, out_json_len, st);
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
        return write_ok(out_json, out_json_len, "{\"done\":true}");
    }

    return write_error(out_json, out_json_len, fer_status_t{FER_STATUS_INVALID_ARGUMENT, "unknown tool"});
}
