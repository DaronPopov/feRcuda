// ============================================================================
// Public API Implementation
// ============================================================================
// Note: Allocation tracking for leak detection now uses debug info stored
// directly in TLSFBlock (alloc_file, alloc_line, alloc_timestamp).
// No separate tracking array needed - O(1) lookup via hash table.

// Create allocator using an existing GPU memory pool (does not take ownership)
PTXTLSFAllocator* ptx_tlsf_create_from_pool(void* pool, size_t pool_size, bool debug_mode) {
    if (!pool || pool_size == 0) return NULL;

    PTXTLSFAllocator* allocator = (PTXTLSFAllocator*)malloc(sizeof(PTXTLSFAllocator));
    if (!allocator) return NULL;

    memset(allocator, 0, sizeof(PTXTLSFAllocator));

    // Use external pool
    allocator->vram_pool = pool;
    allocator->vram_pool_size = pool_size;
    allocator->owns_pool = false;  // External pool, don't free on destroy
    allocator->debug_mode = debug_mode;
    allocator->warning_threshold = 0.80f;
    allocator->warning_step = 0.01f;          // 1% utilization step
    allocator->warning_interval_us = 1000000; // 1s minimum interval
    allocator->warning_last_ts = 0;
    allocator->warning_last_utilization = 0.0f;
    allocator->warnings_enabled = true;
    allocator->auto_defrag_enabled = true;

    // Initialize first block
    TLSFBlock* first_block = (TLSFBlock*)malloc(sizeof(TLSFBlock));
    if (!first_block) {
        free(allocator);
        return NULL;
    }
    memset(first_block, 0, sizeof(TLSFBlock));

    first_block->device_ptr = pool;
    first_block->size = pool_size;
    first_block->magic = TLSF_BLOCK_MAGIC;
    first_block->is_free = true;
    first_block->is_last = true;

    allocator->first_block = first_block;
    allocator->last_block = first_block;

    // Add to hash table and free list
    hash_insert(allocator, first_block);
    insert_free_block(allocator, first_block);

    // Initialize stats
    allocator->stats.total_pool_size = pool_size;
    allocator->stats.free_bytes = pool_size;
    allocator->stats.total_blocks = 1;
    allocator->stats.is_healthy = true;

    printf("[TLSF] Allocator created: %.2f MB pool\n",
           pool_size / (1024.0 * 1024.0));

    return allocator;
}

// Create allocator with its own GPU memory pool (takes ownership)
PTXTLSFAllocator* ptx_tlsf_create(size_t pool_size, bool debug_mode) {
    // Align pool size
    pool_size = align_up(pool_size);

    // Allocate GPU memory pool
    void* pool = ptx_driver_alloc(pool_size);
    if (!pool) {
        printf("[TLSF] Failed to allocate GPU pool\n");
        return NULL;
    }

    PTXTLSFAllocator* allocator = ptx_tlsf_create_from_pool(pool, pool_size, debug_mode);
    if (!allocator) {
        ptx_driver_free(pool);
        return NULL;
    }

    allocator->owns_pool = true;  // We own this pool, free on destroy
    return allocator;
}

void ptx_tlsf_destroy(PTXTLSFAllocator* allocator) {
    if (!allocator) return;

    // Free GPU pool only if we own it
    if (allocator->owns_pool && allocator->vram_pool) {
        ptx_driver_free(allocator->vram_pool);
    }

    // Free all block headers
    TLSFBlock* block = allocator->first_block;
    while (block) {
        TLSFBlock* next = block->next_phys;
        free(block);
        block = next;
    }

    // No separate allocation tracker to free (debug info stored in blocks)

    free(allocator);
    printf("[TLSF] Allocator destroyed\n");
}

void* ptx_tlsf_alloc(PTXTLSFAllocator* allocator, size_t size) {
    return ptx_tlsf_alloc_debug(allocator, size, NULL, 0);
}

void* ptx_tlsf_alloc_debug(PTXTLSFAllocator* allocator, size_t size,
                           const char* file, int line) {
    if (!allocator || size == 0) return NULL;
    
    // Align size
    size = align_up(size);
    
    // Find suitable free block
    TLSFBlock* block = find_free_block(allocator, size);
    
    if (!block) {
        // Pool exhausted
        allocator->stats.fallback_count++;
        printf("[TLSF] Pool exhausted, allocation failed\n");
        return NULL;
    }
    
    // Remove from free list
    remove_free_block(allocator, block);
    
    // Split if block is too large
    split_block(allocator, block, size);
    
    // Update block info
    block->alloc_id = allocator->next_alloc_id;
    block->alloc_file = file;
    block->alloc_line = line;
    block->alloc_timestamp = get_timestamp();
    block->owner_id = 0;
    
    // Update stats
    allocator->stats.allocated_bytes += block->size;
    allocator->stats.free_bytes -= block->size;
    allocator->stats.allocated_blocks++;
    allocator->stats.total_allocations++;

    // Debug: Print every 100th allocation to track counters
    if (ptx_tlsf_debug_enabled() &&
        (allocator->stats.total_allocations % 100 == 0 || allocator->stats.total_allocations <= 5)) {
        printf("[TLSF-DEBUG] Alloc #%llu: %zu bytes, total allocated: %zu MB\n",
               (unsigned long long)allocator->stats.total_allocations,
               size,
               allocator->stats.allocated_bytes / 1024 / 1024);
    }

    if (allocator->stats.allocated_bytes > allocator->stats.peak_allocated) {
        allocator->stats.peak_allocated = allocator->stats.allocated_bytes;
    }
    
    // Debug info already stored in block (alloc_file, alloc_line, alloc_timestamp)
    // No separate tracking needed - we use hash_find for O(1) lookup

    // Check warning threshold (throttled)
    float utilization = (float)allocator->stats.allocated_bytes / allocator->vram_pool_size;
    if (allocator->warnings_enabled && allocator->warning_threshold > 0.0f) {
        if (utilization < allocator->warning_threshold) {
            allocator->warning_last_ts = 0;
            allocator->warning_last_utilization = utilization;
        } else {
            uint64_t now = get_timestamp();
            bool step_ok = (allocator->warning_step <= 0.0f) ||
                           (utilization >= allocator->warning_last_utilization + allocator->warning_step);
            bool time_ok = (allocator->warning_interval_us == 0) ||
                           (allocator->warning_last_ts == 0) ||
                           (now - allocator->warning_last_ts >= allocator->warning_interval_us);
            if (step_ok || time_ok) {
                allocator->warning_last_ts = now;
                allocator->warning_last_utilization = utilization;
                printf("[TLSF] WARNING: Pool utilization at %.1f%%\n", utilization * 100.0f);
            }
        }
    }
    
    // Push ALLOC event
    {
        uint32_t idx = allocator->event_ring.head % TLSF_EVENT_RING_SIZE;
        TLSFAllocEvent* ev = &allocator->event_ring.events[idx];
        ev->timestamp = block->alloc_timestamp;
        ev->size = block->size;
        ev->owner_id = block->owner_id;
        ev->alloc_id = block->alloc_id;
        ev->event_type = 0; // ALLOC
        allocator->event_ring.head++;
        allocator->event_ring.count++;
    }

    return block->device_ptr;
}

void ptx_tlsf_free(PTXTLSFAllocator* allocator, void* ptr) {
    if (!allocator || !ptr) {
        // Debug: trace null arguments
        if (!allocator && ptx_tlsf_debug_enabled()) {
            printf("[TLSF-DEBUG] free called with NULL allocator\n");
        }
        return;
    }

    // Check if pointer is in pool
    if (ptr < allocator->vram_pool ||
        ptr >= (char*)allocator->vram_pool + allocator->vram_pool_size) {
        ptx_strict_free_violation("TLSF", ptr);
        return;
    }

    // Find block using hash table (O(1))
    TLSFBlock* block = hash_find(allocator, ptr);
    if (!block) {
        ptx_strict_free_violation("TLSF", ptr);
        return;
    }

    // Validate block
    if (block->magic != TLSF_BLOCK_MAGIC) {
        printf("[TLSF] ERROR: Block corruption detected at %p\n", ptr);
        return;
    }

    if (block->is_free) {
        printf("[TLSF] ERROR: Double free detected at %p\n", ptr);
        return;
    }

    // Update stats
    allocator->stats.allocated_bytes -= block->size;
    allocator->stats.free_bytes += block->size;
    allocator->stats.allocated_blocks--;
    allocator->stats.total_frees++;

    // Debug: Print every 100th free to track counters
    if (ptx_tlsf_debug_enabled() &&
        (allocator->stats.total_frees % 100 == 0 || allocator->stats.total_frees <= 5)) {
        printf("[TLSF-DEBUG] Free #%llu: %zu bytes, total allocated: %zu MB\n",
               (unsigned long long)allocator->stats.total_frees,
               block->size,
               allocator->stats.allocated_bytes / 1024 / 1024);
    }

    // Reset warning state if utilization drops below threshold
    if (allocator->warnings_enabled && allocator->warning_threshold > 0.0f) {
        float utilization = (float)allocator->stats.allocated_bytes / allocator->vram_pool_size;
        if (utilization < allocator->warning_threshold) {
            allocator->warning_last_ts = 0;
            allocator->warning_last_utilization = utilization;
        }
    }

    // Update owner tracking before clearing
    if (block->owner_id != 0) {
        for (uint32_t i = 0; i < allocator->owner_stats.num_owners; i++) {
            if (allocator->owner_stats.owners[i].owner_id == block->owner_id) {
                allocator->owner_stats.owners[i].allocated_bytes -= block->size;
                allocator->owner_stats.owners[i].block_count--;
                break;
            }
        }
    }

    // Push FREE event
    {
        uint32_t idx = allocator->event_ring.head % TLSF_EVENT_RING_SIZE;
        TLSFAllocEvent* ev = &allocator->event_ring.events[idx];
        ev->timestamp = get_timestamp();
        ev->size = block->size;
        ev->owner_id = block->owner_id;
        ev->alloc_id = block->alloc_id;
        ev->event_type = 1; // FREE
        allocator->event_ring.head++;
        allocator->event_ring.count++;
    }

    // Clear debug info (block lookup via hash is O(1), no separate untracking needed)
    block->alloc_file = NULL;
    block->alloc_line = 0;
    block->owner_id = 0;

    // Coalesce with adjacent free blocks
    block = coalesce(allocator, block);

    // Add to free list
    insert_free_block(allocator, block);

    // Debug: uncomment to trace every free
    // printf("[TLSF-DEBUG] freed %zu bytes, now allocated=%zu\n",
    //        block->size, allocator->stats.allocated_bytes);
}

// ============================================================================
// Diagnostics API Implementation
// ============================================================================

void ptx_tlsf_get_stats(PTXTLSFAllocator* allocator, TLSFPoolStats* stats) {
    if (!allocator || !stats) return;
    
    // Copy current stats
    *stats = allocator->stats;
    
    // Calculate fragmentation metrics
    if (allocator->stats.free_blocks > 0) {
        // External fragmentation: ratio of largest free block to total free memory
        size_t largest_free = 0;
        size_t smallest_free = SIZE_MAX;
        
        for (int fl = 0; fl < TLSF_FL_INDEX_MAX; fl++) {
            for (int sl = 0; sl < TLSF_SL_INDEX_COUNT; sl++) {
                TLSFBlock* block = allocator->free_lists[fl][sl];
                while (block) {
                    if (block->size > largest_free) largest_free = block->size;
                    if (block->size < smallest_free) smallest_free = block->size;
                    block = block->next_free;
                }
            }
        }
        
        stats->largest_free_block = largest_free;
        stats->smallest_free_block = smallest_free;
        
        if (allocator->stats.free_bytes > 0) {
            stats->external_fragmentation = 1.0f - ((float)largest_free / allocator->stats.free_bytes);
        }

        // Fragmentation ratio: how much free memory is unusable due to
        // being split into small blocks.  Same as external_fragmentation:
        // 0.0 = one big free block (perfect), 1.0 = all free memory is in
        // tiny scattered fragments with no large contiguous region.
        stats->fragmentation_ratio = stats->external_fragmentation;
    }

    // Calculate hash table stats
    uint32_t total_chain_length = 0;
    for (int i = 0; i < TLSF_HASH_TABLE_SIZE; i++) {
        total_chain_length += allocator->hash_table[i].count;
    }
    stats->avg_chain_length = (float)total_chain_length / TLSF_HASH_TABLE_SIZE;

    // Utilization
    stats->utilization_percent = (float)allocator->stats.allocated_bytes /
                                 allocator->vram_pool_size * 100.0f;

    // Health = no failed allocations and pool isn't nearly full.
    // Fragmentation percentage is informational only — scattered free
    // blocks are normal under parallel workloads and don't indicate
    // a problem unless allocations actually start failing.
    stats->is_healthy = (stats->fallback_count == 0) &&
                       (stats->utilization_percent < 95.0f);

    // Defrag only worth doing if allocations have actually failed
    stats->needs_defrag = (stats->fallback_count > 0);
}

void ptx_tlsf_validate(PTXTLSFAllocator* allocator, TLSFHealthReport* report) {
    if (!allocator || !report) return;
    
    memset(report, 0, sizeof(TLSFHealthReport));
    report->is_valid = true;
    
    // Validate physical chain
    TLSFBlock* block = allocator->first_block;
    int block_count = 0;
    size_t total_size = 0;
    
    while (block) {
        block_count++;
        total_size += block->size;
        
        // Check magic number
        if (block->magic != TLSF_BLOCK_MAGIC) {
            snprintf(report->error_messages[report->error_count++], 256,
                    "Block %d: Corrupted magic number", block_count);
            report->has_corrupted_blocks = true;
            report->is_valid = false;
        }
        
        // Check chain integrity
        if (block->next_phys && block->next_phys->prev_phys != block) {
            snprintf(report->error_messages[report->error_count++], 256,
                    "Block %d: Broken physical chain", block_count);
            report->has_broken_chains = true;
            report->is_valid = false;
        }
        
        // Check device pointer alignment
        if (((uintptr_t)block->device_ptr) % TLSF_ALIGNMENT != 0) {
            snprintf(report->error_messages[report->error_count++], 256,
                    "Block %d: Misaligned device pointer", block_count);
            report->is_valid = false;
        }
        
        block = block->next_phys;
        
        if (report->error_count >= 16) break;
    }
    
    // Check total size
    if (total_size != allocator->vram_pool_size) {
        snprintf(report->error_messages[report->error_count++], 256,
                "Total block size mismatch: %zu vs %zu", total_size, allocator->vram_pool_size);
        report->is_valid = false;
    }
    
    // Validate hash table
    for (int i = 0; i < TLSF_HASH_TABLE_SIZE; i++) {
        TLSFBlock* hash_block = allocator->hash_table[i].head;
        int chain_len = 0;
        
        while (hash_block) {
            chain_len++;
            
            // Verify hash is correct
            uint32_t expected_hash = hash_ptr(hash_block->device_ptr);
            if (expected_hash != i) {
                snprintf(report->error_messages[report->error_count++], 256,
                        "Hash bucket %d: Block in wrong bucket", i);
                report->has_hash_errors = true;
                report->is_valid = false;
            }
            
            hash_block = hash_block->hash_next;
            
            if (chain_len > 100) {
                snprintf(report->error_messages[report->error_count++], 256,
                        "Hash bucket %d: Infinite loop detected", i);
                report->has_hash_errors = true;
                report->is_valid = false;
                break;
            }
        }
    }
    
    // Check for memory leaks by counting allocated blocks
    if (allocator->debug_mode && allocator->stats.allocated_blocks > 0) {
        report->has_memory_leaks = true;
        snprintf(report->error_messages[report->error_count++], 256,
                "Memory leak: %u allocations not freed", allocator->stats.allocated_blocks);
    }
}

bool ptx_tlsf_can_allocate(PTXTLSFAllocator* allocator, size_t size) {
    if (!allocator) return false;
    
    size = align_up(size);
    TLSFBlock* block = find_free_block(allocator, size);
    return (block != NULL);
}

size_t ptx_tlsf_get_max_allocatable(PTXTLSFAllocator* allocator) {
    if (!allocator) return 0;
    
    size_t max_size = 0;
    
    for (int fl = TLSF_FL_INDEX_MAX - 1; fl >= 0; fl--) {
        for (int sl = TLSF_SL_INDEX_COUNT - 1; sl >= 0; sl--) {
            TLSFBlock* block = allocator->free_lists[fl][sl];
            if (block && block->size > max_size) {
                max_size = block->size;
            }
        }
    }
    
    return max_size;
}

bool ptx_tlsf_owns_ptr(PTXTLSFAllocator* allocator, void* ptr) {
    if (!allocator || !ptr) return false;

    // Check if pointer is within pool bounds
    void* pool_start = allocator->vram_pool;
    void* pool_end = (char*)pool_start + allocator->vram_pool_size;

    return (ptr >= pool_start && ptr < pool_end);
}

void ptx_tlsf_print_pool_map(PTXTLSFAllocator* allocator) {
    if (!allocator) return;
    
    printf("\n========== TLSF Pool Memory Map ==========\n");
    printf("Pool: %p - %p (%.2f MB)\n", 
           allocator->vram_pool,
           (char*)allocator->vram_pool + allocator->vram_pool_size,
           allocator->vram_pool_size / (1024.0 * 1024.0));
    printf("==========================================\n");
    
    TLSFBlock* block = allocator->first_block;
    int index = 0;
    
    while (block) {
        printf("[%3d] %p: %8zu bytes %s%s%s\n",
               index++,
               block->device_ptr,
               block->size,
               block->is_free ? "[FREE]" : "[USED]",
               block->is_pinned ? " [PINNED]" : "",
               block->is_last ? " [LAST]" : "");
        
        if (allocator->debug_mode && !block->is_free && block->alloc_file) {
            printf("      Allocated at %s:%d (ID: %u)\n",
                   block->alloc_file, block->alloc_line, block->alloc_id);
        }
        
        block = block->next_phys;
    }
    
    printf("==========================================\n");
    printf("Total blocks: %d\n", allocator->stats.total_blocks);
    printf("Free blocks: %d\n", allocator->stats.free_blocks);
    printf("Allocated: %.2f MB / %.2f MB (%.1f%%)\n",
           allocator->stats.allocated_bytes / (1024.0 * 1024.0),
           allocator->vram_pool_size / (1024.0 * 1024.0),
           allocator->stats.allocated_bytes * 100.0 / allocator->vram_pool_size);
    printf("==========================================\n\n");
}

void ptx_tlsf_print_allocations(PTXTLSFAllocator* allocator) {
    if (!allocator || !allocator->debug_mode) {
        printf("[TLSF] Debug mode not enabled\n");
        return;
    }

    printf("\n========== Active Allocations ==========\n");
    printf("Total: %u allocations\n", allocator->stats.allocated_blocks);
    printf("========================================\n");

    // Walk physical block chain to find allocated blocks
    TLSFBlock* block = allocator->first_block;
    uint32_t count = 0;
    while (block) {
        if (!block->is_free) {
            printf("[%4u] %p: %8zu bytes at %s:%d (age: %lu us)\n",
                   block->alloc_id,
                   block->device_ptr,
                   block->size,
                   block->alloc_file ? block->alloc_file : "unknown",
                   block->alloc_line,
                   get_timestamp() - block->alloc_timestamp);
            count++;
        }
        block = block->next_phys;
    }

    printf("========================================\n\n");
}

// ============================================================================
// Maintenance API Implementation
// ============================================================================

void ptx_tlsf_defragment(PTXTLSFAllocator* allocator) {
    if (!allocator) return;
    
    printf("[TLSF] Starting defragmentation...\n");
    
    int merges = 0;
    TLSFBlock* block = allocator->first_block;
    
    while (block) {
        if (block->is_free && block->next_phys && block->next_phys->is_free) {
            TLSFBlock* next = block->next_phys;
            
            remove_free_block(allocator, block);
            remove_free_block(allocator, next);
            hash_remove(allocator, next);
            
            block->size += next->size;
            block->next_phys = next->next_phys;
            block->is_last = next->is_last;
            
            if (block->next_phys) {
                block->next_phys->prev_phys = block;
            } else {
                allocator->last_block = block;
            }
            
            free(next);
            allocator->stats.total_blocks--;
            merges++;
            
            insert_free_block(allocator, block);
            // Don't advance - check again with new next block
        } else {
            block = block->next_phys;
        }
    }
    
    printf("[TLSF] Defragmentation complete: %d merges\n", merges);
}

void ptx_tlsf_compact(PTXTLSFAllocator* allocator) {
    // DEFERRED: True compaction (moving live allocations) requires a pointer-
    // forwarding or handle-indirection layer so consumers get updated
    // addresses. defragment() already coalesces adjacent free blocks which
    // covers the common case. Full compaction may be revisited if heavy
    // fragmentation is observed under sustained workloads.
    (void)allocator;
}

void ptx_tlsf_set_warning_threshold(PTXTLSFAllocator* allocator, float threshold) {
    if (allocator) {
        allocator->warning_threshold = threshold;
    }
}

void ptx_tlsf_set_warning_step(PTXTLSFAllocator* allocator, float step) {
    if (allocator) {
        allocator->warning_step = step;
    }
}

void ptx_tlsf_set_warning_interval_ms(PTXTLSFAllocator* allocator, uint64_t interval_ms) {
    if (allocator) {
        allocator->warning_interval_us = interval_ms * 1000;
    }
}

void ptx_tlsf_set_warnings_enabled(PTXTLSFAllocator* allocator, bool enabled) {
    if (allocator) {
        allocator->warnings_enabled = enabled;
        if (!enabled) {
            allocator->warning_last_ts = 0;
            allocator->warning_last_utilization = 0.0f;
        }
    }
}

void ptx_tlsf_set_auto_defrag(PTXTLSFAllocator* allocator, bool enable) {
    if (allocator) {
        allocator->auto_defrag_enabled = enable;
    }
}

void ptx_tlsf_reset_stats(PTXTLSFAllocator* allocator) {
    if (!allocator) return;
    
    allocator->stats.total_allocations = 0;
    allocator->stats.total_frees = 0;
    allocator->stats.total_splits = 0;
    allocator->stats.total_merges = 0;
    allocator->stats.fallback_count = 0;
    allocator->stats.hash_collisions = 0;
    allocator->stats.peak_allocated = allocator->stats.allocated_bytes;
}

// ============================================================================
// Advanced API Implementation
// ============================================================================

void ptx_tlsf_pin_block(PTXTLSFAllocator* allocator, void* ptr) {
    if (!allocator || !ptr) return;
    
    TLSFBlock* block = hash_find(allocator, ptr);
    if (block) {
        block->is_pinned = true;
    }
}

void ptx_tlsf_unpin_block(PTXTLSFAllocator* allocator, void* ptr) {
    if (!allocator || !ptr) return;
    
    TLSFBlock* block = hash_find(allocator, ptr);
    if (block) {
        block->is_pinned = false;
    }
}

bool ptx_tlsf_get_block_info(PTXTLSFAllocator* allocator, void* ptr,
                             size_t* size_out, bool* is_free_out) {
    if (!allocator || !ptr) return false;
    
    TLSFBlock* block = hash_find(allocator, ptr);
    if (!block) return false;
    
    if (size_out) *size_out = block->size;
    if (is_free_out) *is_free_out = block->is_free;
    
    return true;
}

bool ptx_tlsf_expand_pool(PTXTLSFAllocator* allocator, size_t additional_size) {
    // DEFERRED: Pool expansion requires allocating a new CUDA memory region
    // and managing multiple disjoint pools within the TLSF bitmap structure.
    // Current mitigation: size the initial pool via GPUHotConfig::pool_fraction
    // or GPUHotConfig::fixed_pool_size at init time.
    (void)allocator;
    (void)additional_size;
    return false;
}

bool ptx_tlsf_shrink_pool(PTXTLSFAllocator* allocator) {
    // DEFERRED: Pool shrinking requires relocating live allocations to the
    // beginning of the pool then returning the tail region to CUDA. This
    // needs pointer-update cooperation from all consumers (tensors, buffers)
    // which is not yet feasible. Use defragment() to coalesce free blocks.
    (void)allocator;
    return false;
}

void* ptx_tlsf_realloc(PTXTLSFAllocator* allocator, void* ptr, size_t new_size) {
    if (!allocator) return NULL;
    
    if (!ptr) {
        return ptx_tlsf_alloc(allocator, new_size);
    }
    
    if (new_size == 0) {
        ptx_tlsf_free(allocator, ptr);
        return NULL;
    }
    
    // Find existing block
    TLSFBlock* block = hash_find(allocator, ptr);
    if (!block) {
        // Not in our pool, can't realloc
        return NULL;
    }
    
    new_size = align_up(new_size);
    
    // If new size fits in current block, just return
    if (new_size <= block->size) {
        return ptr;
    }
    
    // Try to expand in place by merging with next block
    if (block->next_phys && block->next_phys->is_free &&
        block->size + block->next_phys->size >= new_size) {
        
        TLSFBlock* next = block->next_phys;
        remove_free_block(allocator, next);
        hash_remove(allocator, next);
        
        block->size += next->size;
        block->next_phys = next->next_phys;
        block->is_last = next->is_last;
        
        if (block->next_phys) {
            block->next_phys->prev_phys = block;
        }
        
        free(next);
        allocator->stats.total_blocks--;
        
        // Split if too large
        split_block(allocator, block, new_size);
        
        return ptr;
    }
    
    // Can't expand in place, allocate new block and copy
    void* new_ptr = ptx_tlsf_alloc(allocator, new_size);
    if (!new_ptr) return NULL;
    
    // Copy data
    cudaMemcpy(new_ptr, ptr, block->size, cudaMemcpyDeviceToDevice);

    // Free old block
    ptx_tlsf_free(allocator, ptr);

    return new_ptr;
}

// ============================================================================
// Per-Owner Memory Tracking
// ============================================================================

static void owner_stats_add(PTXTLSFAllocator* allocator, uint32_t owner_id, size_t size) {
    TLSFOwnerStats* os = &allocator->owner_stats;
    for (uint32_t i = 0; i < os->num_owners; i++) {
        if (os->owners[i].owner_id == owner_id) {
            os->owners[i].allocated_bytes += size;
            os->owners[i].block_count++;
            return;
        }
    }
    // New owner
    if (os->num_owners < TLSF_MAX_OWNERS) {
        os->owners[os->num_owners].owner_id = owner_id;
        os->owners[os->num_owners].allocated_bytes = size;
        os->owners[os->num_owners].block_count = 1;
        os->num_owners++;
    }
}

void* ptx_tlsf_alloc_owned(PTXTLSFAllocator* allocator, size_t size, uint32_t owner_id) {
    void* ptr = ptx_tlsf_alloc_debug(allocator, size, NULL, 0);
    if (ptr) {
        TLSFBlock* block = hash_find(allocator, ptr);
        if (block) {
            block->owner_id = owner_id;
            owner_stats_add(allocator, owner_id, block->size);
        }
    }
    return ptr;
}

void ptx_tlsf_free_owner(PTXTLSFAllocator* allocator, uint32_t owner_id) {
    if (!allocator) return;

    // Collect pointers to free (can't free while iterating - coalesce modifies chain)
    void* to_free[4096];
    int count = 0;

    TLSFBlock* block = allocator->first_block;
    while (block && count < 4096) {
        if (!block->is_free && block->owner_id == owner_id) {
            to_free[count++] = block->device_ptr;
        }
        block = block->next_phys;
    }

    for (int i = 0; i < count; i++) {
        ptx_tlsf_free(allocator, to_free[i]);
    }
}

void ptx_tlsf_get_owner_stats(PTXTLSFAllocator* allocator, TLSFOwnerStats* stats) {
    if (!allocator || !stats) return;
    memcpy(stats, &allocator->owner_stats, sizeof(TLSFOwnerStats));
}

void ptx_tlsf_get_events(PTXTLSFAllocator* allocator, TLSFEventRing* ring_out) {
    if (!allocator || !ring_out) return;
    memcpy(ring_out, &allocator->event_ring, sizeof(TLSFEventRing));
}
