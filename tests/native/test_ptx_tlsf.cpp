#include <unistd.h>

#include "memory/ptx_tlsf_allocator.h"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

static int fail(const char* msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("test_ptx_tlsf: SKIP (no CUDA device)\n");
        return 0;
    }
    cudaSetDevice(0);

    constexpr size_t pool_size = 32ull * 1024ull * 1024ull;  // 32 MiB
    PTXTLSFAllocator* alloc = ptx_tlsf_create(pool_size, true);
    if (!alloc) return fail("ptx_tlsf_create returned null");

    // Basic allocation
    void* p1 = ptx_tlsf_alloc(alloc, 4096);
    if (!p1) return fail("ptx_tlsf_alloc(4096) failed");
    if (!ptx_tlsf_owns_ptr(alloc, p1)) return fail("owns_ptr false for allocated block");

    // Allocate a series of blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 100; ++i) {
        void* p = ptx_tlsf_alloc(alloc, 1024 * (i + 1));
        if (!p) return fail("bulk allocation failed");
        ptrs.push_back(p);
    }

    // Check stats
    TLSFPoolStats stats{};
    ptx_tlsf_get_stats(alloc, &stats);
    if (stats.allocated_blocks < 101) return fail("allocated_blocks too low");
    if (stats.total_allocations < 101) return fail("total_allocations too low");

    // Free everything
    ptx_tlsf_free(alloc, p1);
    for (void* p : ptrs) ptx_tlsf_free(alloc, p);

    // Validate pool health after free
    TLSFHealthReport health{};
    ptx_tlsf_validate(alloc, &health);
    if (!health.is_valid) return fail("pool not valid after free cycle");
    if (health.has_memory_leaks) return fail("memory leaks detected after full free");

    // Per-owner allocation
    void* o1 = ptx_tlsf_alloc_owned(alloc, 8192, 42);
    void* o2 = ptx_tlsf_alloc_owned(alloc, 16384, 42);
    if (!o1 || !o2) return fail("owned allocation failed");

    TLSFOwnerStats ostats{};
    ptx_tlsf_get_owner_stats(alloc, &ostats);
    bool found_owner = false;
    for (uint32_t i = 0; i < ostats.num_owners; ++i) {
        if (ostats.owners[i].owner_id == 42) {
            found_owner = true;
            if (ostats.owners[i].block_count < 2)
                return fail("owner block count too low");
        }
    }
    if (!found_owner) return fail("owner 42 not found in stats");

    ptx_tlsf_free_owner(alloc, 42);

    // Defragment
    ptx_tlsf_defragment(alloc);

    ptx_tlsf_destroy(alloc);

    std::printf("PTX TLSF TEST PASSED\n");
    return 0;
}
