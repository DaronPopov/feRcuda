/*
 * PTX-OS CUDA Hook Control API
 *
 * For LD_PRELOAD hook library (libptx_hook.so) hybrid coordination.
 */

#ifndef PTX_HOOK_H
#define PTX_HOOK_H

#ifdef __cplusplus
extern "C" {
#endif

// Thread-local allocator mode overrides (hybrid mode only)
#define PTX_HOOK_THREAD_DEFAULT 0
#define PTX_HOOK_THREAD_TLSF    1
#define PTX_HOOK_THREAD_CUDA    2

// Set/get thread-local allocator preference
void ptx_hook_set_thread_mode(int mode);
int ptx_hook_get_thread_mode(void);

#ifdef __cplusplus
}
#endif

#endif // PTX_HOOK_H
