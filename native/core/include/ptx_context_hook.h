/**
 * PTX-OS CUDA Context Interception Hook - Public API
 *
 * Ensures the PTX runtime's primary context is the single enforced context
 * for all CUDA operations, preventing external libraries from creating
 * independent contexts that bypass TLSF memory management.
 *
 * Environment variables:
 *   PTX_HOOK_VERBOSE=1              - Verbose logging (shared with cuda_intercept.c)
 *   PTX_CONTEXT_HOOK_DISABLE=1      - Disable context interception entirely
 *   PTX_CONTEXT_HOOK_WARN_ONLY=1    - Log warnings but don't block (debug mode)
 */

#ifndef PTX_CONTEXT_HOOK_H
#define PTX_CONTEXT_HOOK_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CUcontext is an opaque pointer in the CUDA driver API */
typedef void* CUcontext;

/**
 * Statistics tracked by the context hook.
 */
typedef struct PTXContextStats {
    uint64_t ctx_create_blocked;
    uint64_t ctx_create_passthrough;
    uint64_t ctx_destroy_blocked;
    uint64_t ctx_switch_prevented;
    uint64_t ctx_switch_substituted;
    uint64_t primary_retain_intercepted;
    uint64_t primary_release_suppressed;
    uint64_t primary_reset_blocked;
    uint64_t primary_setflags_blocked;
    uint64_t push_substituted;
    uint64_t pop_corrected;
} PTXContextStats;

/**
 * Notify the hook that the PTX runtime has captured the primary context
 * for a given device. After this call, the hook transitions to enforcement.
 */
void ptx_context_hook_capture(int device_id, CUcontext ctx);

/**
 * Set whether the PTX runtime is actively using its context.
 * When active, context calls are enforced. When inactive (shutdown),
 * calls pass through to the real driver.
 */
void ptx_context_hook_set_ptx_active(bool active);

/**
 * Check if the context hook is in enforcement mode.
 */
bool ptx_context_hook_is_active(void);

/**
 * Get the captured PTX context for a given device.
 * Returns NULL if no context has been captured for that device.
 */
CUcontext ptx_context_hook_get_context(int device_id);

/**
 * Copy the current context hook statistics into the provided struct.
 */
void ptx_context_hook_get_stats(PTXContextStats* stats);

/**
 * Reset all context hook statistics to zero.
 */
void ptx_context_hook_reset_stats(void);

/**
 * Initialize context hook subsystem (called from cuda_intercept.c constructor).
 */
void ptx_context_hook_init(void);

/**
 * Print context hook statistics (called from cuda_intercept.c destructor).
 */
void ptx_context_hook_print_stats(void);

/**
 * Capture context from the already-initialized runtime.
 * Reads PTX_RUNTIME_PTR and calls gpu_hot_get_context to obtain the CUcontext.
 */
void ptx_context_hook_capture_from_runtime(void);

#ifdef __cplusplus
}
#endif

#endif /* PTX_CONTEXT_HOOK_H */
