#ifndef PTX_TORCH_ALLOCATOR_H
#define PTX_TORCH_ALLOCATOR_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * PTX-Torch Allocator Bridge
 *
 * This adapter allows PyTorch (libtorch) to use PTX-OS's custom TLSF
 * memory allocator instead of the standard CUDA allocator.
 *
 * Benefits:
 * - Zero fragmentation (TLSF allocator)
 * - Support for thousands of concurrent streams
 * - Unified memory management with PTX kernels
 * - No memory leaks over long-running inference
 */

// Initialize PyTorch to use PTX-OS allocator
// Must be called BEFORE any PyTorch operations
// device_id: CUDA device to use
// Returns: 0 on success, -1 on failure
int ptx_torch_init_allocator(int device_id);

// Cleanup and restore default PyTorch allocator
void ptx_torch_cleanup_allocator(void);

// Check if PTX allocator is active
bool ptx_torch_is_active(void);

// Get statistics from PTX allocator
typedef struct {
    size_t allocated_bytes;
    size_t reserved_bytes;
    size_t active_allocations;
    size_t peak_allocated_bytes;
} PTXTorchAllocatorStats;

void ptx_torch_get_stats(PTXTorchAllocatorStats* stats);

// Stream mapping
// Map PyTorch stream to PTX-OS stream ID
void ptx_torch_set_stream_mapping(void* torch_stream, int ptx_stream_id);

#ifdef __cplusplus
}
#endif

#endif // PTX_TORCH_ALLOCATOR_H
