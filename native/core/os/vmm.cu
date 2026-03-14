/*
 * PTX-OS Virtual Memory Manager
 * GPU memory paging with host swap support
 */

#include "gpu/gpu_hot_runtime.h"
#include "ptx_debug.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

// ============================================================================
// VMM Internal Helpers
// ============================================================================

static inline uint64_t vmm_get_timestamp() {
#ifdef _WIN32
    LARGE_INTEGER counter, freq;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&freq);
    return (counter.QuadPart * 1000000) / freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
#endif
}

// Find page by original address (the address returned to user)
static VMMPageEntry* vmm_find_page(VMMState* vmm, void* addr) {
    for (uint32_t i = 0; i < vmm->num_pages; i++) {
        if (vmm->pages[i].original_addr == addr) {
            return &vmm->pages[i];
        }
    }
    return NULL;
}

// Find LRU page for eviction
static VMMPageEntry* vmm_find_lru_page(VMMState* vmm) {
    VMMPageEntry* lru = NULL;
    uint64_t oldest = UINT64_MAX;

    for (uint32_t i = 0; i < vmm->num_pages; i++) {
        VMMPageEntry* page = &vmm->pages[i];
        if (page->resident && !page->pinned && page->last_access < oldest) {
            oldest = page->last_access;
            lru = page;
        }
    }

    return lru;
}

static void* vmm_alloc_device(VMMState* vmm, size_t size) {
    if (vmm && vmm->runtime) {
        return gpu_hot_alloc(vmm->runtime, size);
    }
    return NULL;
}

static void vmm_free_device(VMMState* vmm, void* ptr) {
    if (!ptr) return;
    if (vmm && vmm->runtime && gpu_hot_owns_ptr(vmm->runtime, ptr)) {
        gpu_hot_free(vmm->runtime, ptr);
        return;
    }
    ptx_strict_free_violation("VMM", ptr);
}

// ============================================================================
// VMM Initialization
// ============================================================================

VMMState* vmm_init(GPUHotRuntime* runtime, size_t swap_size) {
    printf("[VMM] Initializing Virtual Memory Manager...\n");

    VMMState* vmm = (VMMState*)malloc(sizeof(VMMState));
    if (!vmm) {
        printf("[VMM] ERROR: Failed to allocate VMM state\n");
        return NULL;
    }

    memset(vmm, 0, sizeof(VMMState));
    vmm->runtime = runtime;

    // Pre-allocate swap region pool
    size_t regions_to_alloc = swap_size / VMM_PAGE_SIZE;
    if (regions_to_alloc > VMM_MAX_SWAP_REGIONS) {
        regions_to_alloc = VMM_MAX_SWAP_REGIONS;
    }

    vmm->total_swap_size = swap_size;
    vmm->num_swap_regions = 0;

    printf("[VMM] Swap space: %.2f MB (%zu regions)\n",
           swap_size / (1024.0 * 1024.0), regions_to_alloc);
    printf("[VMM] Page size: %d KB\n", VMM_PAGE_SIZE / 1024);
    printf("[VMM] Max pages: %d\n", VMM_MAX_PAGES);
    printf("[VMM] [OK] VMM initialized\n");

    return vmm;
}

void vmm_shutdown(VMMState* vmm) {
    if (!vmm) return;

    printf("[VMM] Shutting down...\n");

    // Free all swap regions
    for (uint32_t i = 0; i < VMM_MAX_SWAP_REGIONS; i++) {
        if (vmm->swap_regions[i]) {
            cudaFreeHost(vmm->swap_regions[i]);
            vmm->swap_regions[i] = NULL;
        }
    }

    // Free GPU pages
    for (uint32_t i = 0; i < vmm->num_pages; i++) {
        if (vmm->pages[i].gpu_addr && vmm->pages[i].resident) {
            vmm_free_device(vmm, vmm->pages[i].gpu_addr);
        }
    }

    free(vmm);
    printf("[VMM] [OK] Shutdown complete\n");
}

// ============================================================================
// Page Allocation
// ============================================================================

void* vmm_alloc_page(VMMState* vmm, uint32_t flags) {
    if (!vmm || vmm->num_pages >= VMM_MAX_PAGES) {
        return NULL;
    }

    // Allocate GPU memory
    void* gpu_addr = vmm_alloc_device(vmm, VMM_PAGE_SIZE);

    if (!gpu_addr) {
        // Try to evict a page and retry
        printf("[VMM] Allocation failed, attempting eviction...\n");

        VMMPageEntry* victim = vmm_find_lru_page(vmm);
        if (victim) {
            if (vmm_swap_out(vmm, victim->original_addr) == 0) {
                // Retry allocation
                gpu_addr = vmm_alloc_device(vmm, VMM_PAGE_SIZE);
            }
        }

        if (!gpu_addr) {
            printf("[VMM] ERROR: Failed to allocate page\n");
            vmm->page_faults++;
            return NULL;
        }
    }

    // Initialize page entry
    VMMPageEntry* page = &vmm->pages[vmm->num_pages++];
    page->gpu_addr = gpu_addr;
    page->original_addr = gpu_addr;  // Store original address for lookup
    page->host_addr = NULL;
    page->size = VMM_PAGE_SIZE;
    page->flags = flags;
    page->last_access = vmm_get_timestamp();
    page->access_count = 1;
    page->resident = true;
    page->dirty = false;
    page->pinned = (flags & VMM_FLAG_PINNED) != 0;

    vmm->resident_pages++;

    printf("[VMM] Allocated page %u at %p (flags: 0x%x)\n",
           vmm->num_pages - 1, gpu_addr, flags);

    return gpu_addr;
}

void vmm_free_page(VMMState* vmm, void* addr) {
    if (!vmm || !addr) return;

    VMMPageEntry* page = vmm_find_page(vmm, addr);
    if (!page) {
        printf("[VMM] WARNING: Attempted to free unknown page %p\n", addr);
        return;
    }

    // Free GPU memory if resident
    if (page->resident) {
        vmm_free_device(vmm, page->gpu_addr);
        vmm->resident_pages--;
    }

    // Free host swap memory if swapped
    if (page->host_addr) {
        cudaFreeHost(page->host_addr);
        vmm->swapped_pages--;
        vmm->used_swap_size -= page->size;
    }

    // Clear page entry (don't compact array for simplicity)
    memset(page, 0, sizeof(VMMPageEntry));

    printf("[VMM] Freed page at %p\n", addr);
}

// ============================================================================
// Page Swapping
// ============================================================================

int vmm_swap_out(VMMState* vmm, void* addr) {
    if (!vmm || !addr) return -1;

    VMMPageEntry* page = vmm_find_page(vmm, addr);
    if (!page) {
        printf("[VMM] ERROR: Page not found for swap-out: %p\n", addr);
        return -1;
    }

    if (!page->resident) {
        printf("[VMM] WARNING: Page already swapped: %p\n", addr);
        return 0;
    }

    if (page->pinned) {
        printf("[VMM] ERROR: Cannot swap pinned page: %p\n", addr);
        return -1;
    }

    printf("[VMM] Swapping out page %p to host...\n", addr);

    // Allocate host memory for swap
    if (!page->host_addr) {
        cudaError_t err = cudaMallocHost(&page->host_addr, page->size);
        if (err != cudaSuccess) {
            printf("[VMM] ERROR: Failed to allocate swap space: %s\n",
                   cudaGetErrorString(err));
            return -1;
        }
        vmm->used_swap_size += page->size;
    }

    // Copy GPU -> Host
    cudaError_t err = cudaMemcpy(page->host_addr, page->gpu_addr,
                                  page->size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[VMM] ERROR: Swap-out copy failed: %s\n",
               cudaGetErrorString(err));
        return -1;
    }

    // Free GPU memory (original_addr preserved for lookup)
    vmm_free_device(vmm, page->gpu_addr);
    page->gpu_addr = NULL;  // Will be reassigned on swap-in
    page->resident = false;
    page->dirty = false;

    vmm->resident_pages--;
    vmm->swapped_pages++;
    vmm->swap_outs++;

    printf("[VMM] [OK] Page swapped to host (%zu bytes)\n", page->size);

    return 0;
}

int vmm_swap_in(VMMState* vmm, void* addr) {
    if (!vmm || !addr) return -1;

    VMMPageEntry* page = vmm_find_page(vmm, addr);
    if (!page) {
        printf("[VMM] ERROR: Page not found for swap-in: %p\n", addr);
        vmm->page_faults++;
        return -1;
    }

    if (page->resident) {
        // Page already resident, just update access time
        page->last_access = vmm_get_timestamp();
        page->access_count++;
        return 0;
    }

    if (!page->host_addr) {
        printf("[VMM] ERROR: No swap data for page: %p\n", addr);
        vmm->page_faults++;
        return -1;
    }

    printf("[VMM] Swapping in page %p from host...\n", addr);

    // Allocate GPU memory
    void* new_gpu_addr = vmm_alloc_device(vmm, page->size);

    if (!new_gpu_addr) {
        // Need to evict another page
        printf("[VMM] No GPU memory, evicting LRU page...\n");

        VMMPageEntry* victim = vmm_find_lru_page(vmm);
        if (!victim || vmm_swap_out(vmm, victim->original_addr) != 0) {
            printf("[VMM] ERROR: Eviction failed\n");
            vmm->page_faults++;
            return -1;
        }

        vmm->evictions++;

        // Retry allocation
        new_gpu_addr = vmm_alloc_device(vmm, page->size);
        if (!new_gpu_addr) {
            printf("[VMM] ERROR: Allocation still failed after eviction\n");
            vmm->page_faults++;
            return -1;
        }
    }

    // Copy Host -> GPU
    cudaError_t err = cudaMemcpy(new_gpu_addr, page->host_addr,
                     page->size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("[VMM] ERROR: Swap-in copy failed: %s\n",
               cudaGetErrorString(err));
        vmm_free_device(vmm, new_gpu_addr);
        vmm->page_faults++;
        return -1;
    }

    // Update page entry
    page->gpu_addr = new_gpu_addr;
    page->resident = true;
    page->last_access = vmm_get_timestamp();
    page->access_count++;

    vmm->resident_pages++;
    vmm->swapped_pages--;
    vmm->swap_ins++;

    printf("[VMM] [OK] Page swapped in (%zu bytes)\n", page->size);

    return 0;
}

// ============================================================================
// Page Pinning
// ============================================================================

void vmm_pin_page(VMMState* vmm, void* addr) {
    if (!vmm || !addr) return;

    VMMPageEntry* page = vmm_find_page(vmm, addr);
    if (page) {
        page->pinned = true;
        page->flags |= VMM_FLAG_PINNED;
        printf("[VMM] Pinned page %p\n", addr);
    }
}

void vmm_unpin_page(VMMState* vmm, void* addr) {
    if (!vmm || !addr) return;

    VMMPageEntry* page = vmm_find_page(vmm, addr);
    if (page) {
        page->pinned = false;
        page->flags &= ~VMM_FLAG_PINNED;
        printf("[VMM] Unpinned page %p\n", addr);
    }
}

// ============================================================================
// VMM Statistics
// ============================================================================

void vmm_get_stats(VMMState* vmm, uint64_t* resident, uint64_t* swapped,
                   uint64_t* faults, uint64_t* evictions) {
    if (!vmm) return;

    if (resident) *resident = vmm->resident_pages;
    if (swapped) *swapped = vmm->swapped_pages;
    if (faults) *faults = vmm->page_faults;
    if (evictions) *evictions = vmm->evictions;
}

// ============================================================================
// VMM Debug
// ============================================================================

void vmm_print_stats(VMMState* vmm) {
    if (!vmm) return;

    printf("\n========== VMM Statistics ==========\n");
    printf("Total pages:    %u\n", vmm->num_pages);
    printf("Resident pages: %u\n", vmm->resident_pages);
    printf("Swapped pages:  %u\n", vmm->swapped_pages);
    printf("Page faults:    %lu\n", vmm->page_faults);
    printf("Swap ins:       %lu\n", vmm->swap_ins);
    printf("Swap outs:      %lu\n", vmm->swap_outs);
    printf("Evictions:      %lu\n", vmm->evictions);
    printf("Swap usage:     %.2f MB / %.2f MB\n",
           vmm->used_swap_size / (1024.0 * 1024.0),
           vmm->total_swap_size / (1024.0 * 1024.0));
    printf("====================================\n\n");
}

void vmm_print_page_table(VMMState* vmm) {
    if (!vmm) return;

    printf("\n========== VMM Page Table ==========\n");
    printf("%-4s %-16s %-16s %-8s %-8s %-8s\n",
           "ID", "GPU Addr", "Host Addr", "Size", "Resident", "Pinned");
    printf("----------------------------------------------------\n");

    for (uint32_t i = 0; i < vmm->num_pages; i++) {
        VMMPageEntry* page = &vmm->pages[i];
        if (page->gpu_addr || page->host_addr) {
            printf("%-4u %-16p %-16p %-8zu %-8s %-8s\n",
                   i, page->gpu_addr, page->host_addr, page->size,
                   page->resident ? "YES" : "NO",
                   page->pinned ? "YES" : "NO");
        }
    }
    printf("====================================\n\n");
}

// ============================================================================
// VMM Eviction for Allocation Pressure
// ============================================================================

int vmm_evict_for_alloc(VMMState* vmm, size_t needed_size) {
    if (!vmm) return -1;

    size_t freed = 0;
    while (freed < needed_size) {
        VMMPageEntry* lru = vmm_find_lru_page(vmm);
        if (!lru) {
            return (freed >= needed_size) ? 0 : -1;
        }

        size_t page_size = lru->size;
        if (vmm_swap_out(vmm, lru->original_addr) != 0) {
            return -1;
        }
        freed += page_size;
    }

    return 0;
}
