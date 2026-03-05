#pragma once

#include <stddef.h>

#include "fercuda/agent/mcp_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Dispatch a single MCP-style tool call.
// Input:
// - tool_name: e.g. "fer.session.create"
// - request_json: strict JSON payload for the tool.
// Output:
// - out_json: {"ok":true,"result":...} or {"ok":false,"error":...}
fer_status_t fer_agent_mcp_dispatch(
    fer_agent_adapter_t* adapter,
    const char* tool_name,
    const char* request_json,
    char* out_json,
    size_t out_json_len);

#ifdef __cplusplus
}
#endif
