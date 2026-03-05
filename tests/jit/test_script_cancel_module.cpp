#include "fercuda/jit/script_host.h"

#include <cstring>

extern "C" uint32_t fer_script_abi_version(void) {
    return FER_SCRIPT_HOST_ABI_VERSION;
}

static fer_status_t fail(char* err_buf, size_t err_buf_len, const char* msg) {
    if (err_buf && err_buf_len > 0) {
        const size_t n = std::strlen(msg);
        const size_t m = (n < (err_buf_len - 1)) ? n : (err_buf_len - 1);
        std::memcpy(err_buf, msg, m);
        err_buf[m] = '\0';
    }
    return fer_status_t{FER_STATUS_INTERNAL_ERROR, msg};
}

extern "C" fer_status_t fer_script_main(
    const fer_script_context_t* ctx,
    const char*,
    char* err_buf,
    size_t err_buf_len) {
    if (!ctx || !ctx->api) return fail(err_buf, err_buf_len, "invalid ctx");
    for (;;) {
        if (ctx->api->script_should_cancel && ctx->api->script_should_cancel(ctx)) {
            return fail(err_buf, err_buf_len, "cancelled");
        }
        volatile int spin = 0;
        for (int i = 0; i < 10000; ++i) spin += i;
        (void)spin;
    }
}
