#include "fercuda/api/c_api.h"
#include "fercuda/jit/script_runtime.h"

#include <cstdio>
#include <cstring>

#ifndef TEST_SCRIPT_MODULE_PATH
#define TEST_SCRIPT_MODULE_PATH ""
#endif

static int fail(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

int main() {
    if (std::strlen(TEST_SCRIPT_MODULE_PATH) == 0) return fail("missing TEST_SCRIPT_MODULE_PATH");

    fer_session_t* session = nullptr;
    fer_pool_config_t cfg{};
    cfg.mutable_bytes = 64ull * 1024ull * 1024ull;
    cfg.immutable_bytes = 64ull * 1024ull * 1024ull;
    cfg.memory_regime = FER_MEMORY_CUSTOM_POOL;
    if (fer_session_create(0, &cfg, &session).code != FER_STATUS_OK || !session) {
        return fail("fer_session_create");
    }

    fer_script_runtime_t* rt = nullptr;
    if (fer_script_runtime_create(session, &rt).code != FER_STATUS_OK || !rt) {
        return fail("fer_script_runtime_create");
    }

    if (fer_script_load(rt, TEST_SCRIPT_MODULE_PATH).code != FER_STATUS_OK) {
        return fail("fer_script_load");
    }

    uint8_t loaded = 0;
    if (fer_script_is_loaded(rt, &loaded).code != FER_STATUS_OK || loaded != 1u) {
        return fail("fer_script_is_loaded");
    }

    char err[1024];
    err[0] = '\0';
    fer_status_t st = fer_script_run(rt, "{\"mode\":\"jit\"}", err, sizeof(err));
    if (st.code != FER_STATUS_OK) {
        std::fprintf(stderr, "script error: %s\n", err[0] ? err : (st.message ? st.message : "(none)"));
        return fail("fer_script_run");
    }

    if (fer_script_unload(rt).code != FER_STATUS_OK) return fail("fer_script_unload");
    if (fer_script_runtime_destroy(rt).code != FER_STATUS_OK) return fail("fer_script_runtime_destroy");
    if (fer_session_destroy(session).code != FER_STATUS_OK) return fail("fer_session_destroy");

    std::printf("SCRIPT RUNTIME TEST PASSED\n");
    return 0;
}
