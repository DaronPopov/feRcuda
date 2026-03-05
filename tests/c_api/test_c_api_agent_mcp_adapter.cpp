#include "fercuda/agent/mcp_adapter.h"

#include "fercuda/api/c_api.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

int main() {
    fer_agent_adapter_t* adapter = nullptr;
    if (fer_agent_adapter_create(&adapter).code != FER_STATUS_OK || !adapter) {
        return fail("fer_agent_adapter_create");
    }

    fer_agent_runtime_inspect_t info{};
    if (fer_agent_runtime_inspect(adapter, &info).code != FER_STATUS_OK) {
        return fail("fer_agent_runtime_inspect");
    }
    if (info.agent_api_version != FER_AGENT_API_VERSION_V1ALPHA1) {
        return fail("agent_api_version mismatch");
    }
    if (!info.supports_jit_intent || !info.supports_external_ptr_import) {
        return fail("runtime inspect features");
    }

    fer_agent_session_create_request_t sreq{};
    sreq.device = 0;
    sreq.mutable_bytes = 32ull * 1024ull * 1024ull;
    sreq.immutable_bytes = 32ull * 1024ull * 1024ull;
    sreq.memory_regime = FER_MEMORY_CUSTOM_POOL;

    uint64_t session_id = 0;
    if (fer_agent_session_create(adapter, &sreq, &session_id).code != FER_STATUS_OK || session_id == 0) {
        return fail("fer_agent_session_create");
    }

    constexpr uint32_t n = 128;
    std::vector<float> in(n, 4.0f);
    std::vector<float> out(n, 0.0f);

    fer_agent_tensor_create_request_t treq{};
    treq.session_id = session_id;
    treq.dtype = FER_DTYPE_F32;
    treq.rank = 1;
    treq.dims[0] = n;
    treq.memory_regime = FER_MEMORY_CUSTOM_POOL;

    uint64_t x_id = 0;
    uint64_t y_id = 0;
    if (fer_agent_tensor_create(adapter, &treq, &x_id).code != FER_STATUS_OK || x_id == 0) {
        return fail("fer_agent_tensor_create x");
    }
    if (fer_agent_tensor_create(adapter, &treq, &y_id).code != FER_STATUS_OK || y_id == 0) {
        return fail("fer_agent_tensor_create y");
    }

    fer_agent_tensor_copy_request_t copy_in{};
    copy_in.session_id = session_id;
    copy_in.tensor_id = x_id;
    copy_in.dtype = FER_DTYPE_F32;
    copy_in.count = n;
    copy_in.host_ptr = in.data();
    copy_in.direction = FER_AGENT_COPY_HOST_TO_DEVICE;
    if (fer_agent_tensor_copy(adapter, &copy_in).code != FER_STATUS_OK) {
        return fail("fer_agent_tensor_copy H2D");
    }

    fer_agent_intent_affine_f32_request_t ireq{};
    ireq.session_id = session_id;
    ireq.input_tensor_id = x_id;
    ireq.output_tensor_id = y_id;
    ireq.n = n;
    ireq.alpha = 2.0f;
    ireq.beta = 1.0f;
    ireq.memory_regime = FER_MEMORY_CUSTOM_POOL;

    uint64_t job_id = 0;
    if (fer_agent_jit_intent_run_affine_f32(adapter, &ireq, &job_id).code != FER_STATUS_OK || job_id == 0) {
        return fail("fer_agent_jit_intent_run_affine_f32");
    }
    if (fer_agent_job_wait(adapter, session_id, job_id).code != FER_STATUS_OK) {
        return fail("fer_agent_job_wait");
    }

    fer_agent_tensor_copy_request_t copy_out{};
    copy_out.session_id = session_id;
    copy_out.tensor_id = y_id;
    copy_out.dtype = FER_DTYPE_F32;
    copy_out.count = n;
    copy_out.host_ptr = out.data();
    copy_out.direction = FER_AGENT_COPY_DEVICE_TO_HOST;
    if (fer_agent_tensor_copy(adapter, &copy_out).code != FER_STATUS_OK) {
        return fail("fer_agent_tensor_copy D2H");
    }

    for (uint32_t i = 0; i < n; ++i) {
        const float expect = 9.0f;
        if (std::fabs(out[i] - expect) > 1e-5f) {
            return fail("result mismatch");
        }
    }

    if (fer_agent_session_destroy(adapter, session_id).code != FER_STATUS_OK) {
        return fail("fer_agent_session_destroy");
    }
    if (fer_agent_adapter_destroy(adapter).code != FER_STATUS_OK) {
        return fail("fer_agent_adapter_destroy");
    }

    std::printf("AGENT MCP ADAPTER TEST PASSED\n");
    return 0;
}
