// ============================================================================
// Platform-specific utilities
// ============================================================================

#ifdef _WIN32
static inline int clz32(uint32_t x) {
    unsigned long index;
    return _BitScanReverse(&index, x) ? (31 - index) : 32;
}

static inline int clz64(uint64_t x) {
    unsigned long index;
    return _BitScanReverse64(&index, x) ? (63 - index) : 64;
}
#else
static inline int clz64(uint64_t x) {
    return x ? __builtin_clzll(x) : 64;
}
#endif

static inline uint64_t get_timestamp() {
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

static inline bool ptx_tlsf_debug_enabled() {
    static int initialized = 0;
    static bool enabled = false;
    if (!initialized) {
        const char* v = getenv("PTX_TLSF_DEBUG");
        enabled = (v && v[0] && v[0] != '0');
        initialized = 1;
    }
    return enabled;
}

// ============================================================================
// Hash function for O(1) block lookup
// ============================================================================

static inline uint32_t hash_ptr(void* ptr) {
    // FNV-1a hash
    uint64_t addr = (uint64_t)ptr;
    uint32_t hash = 2166136261u;
    
    for (int i = 0; i < 8; i++) {
        hash ^= (addr & 0xFF);
        hash *= 16777619u;
        addr >>= 8;
    }
    
    return hash % TLSF_HASH_TABLE_SIZE;
}

// ============================================================================
// TLSF mapping functions
// ============================================================================

static inline size_t align_up(size_t size) {
    return (size + TLSF_ALIGNMENT - 1) & ~(TLSF_ALIGNMENT - 1);
}

static inline void tlsf_mapping(size_t size, int* fl, int* sl) {
    if (size < TLSF_MIN_BLOCK_SIZE) {
        *fl = 0;
        *sl = 0;
    } else {
        // First level: log2(size)
        *fl = 63 - clz64((uint64_t)size);
        
        // Clamp FL to valid range
        if (*fl >= TLSF_FL_INDEX_MAX) {
            *fl = TLSF_FL_INDEX_MAX - 1;
        }
        
        // Second level subdivision
        if (*fl >= 4) {
            *sl = (int)((size >> (*fl - 4)) & (TLSF_SL_INDEX_COUNT - 1));
        } else {
            *sl = 0;
        }
    }
}

// ============================================================================
// Hash table operations
// ============================================================================

static void hash_insert(PTXTLSFAllocator* allocator, TLSFBlock* block) {
    uint32_t hash = hash_ptr(block->device_ptr);
    TLSFHashBucket* bucket = &allocator->hash_table[hash];
    
    block->hash_next = bucket->head;
    bucket->head = block;
    bucket->count++;
    
    // Track collisions
    if (bucket->count > 1) {
        allocator->stats.hash_collisions++;
    }
    
    // Update max chain length
    if (bucket->count > allocator->stats.max_chain_length) {
        allocator->stats.max_chain_length = bucket->count;
    }
}

static void hash_remove(PTXTLSFAllocator* allocator, TLSFBlock* block) {
    uint32_t hash = hash_ptr(block->device_ptr);
    TLSFHashBucket* bucket = &allocator->hash_table[hash];

    int guard = 0;
    int max_iter = bucket->count + 1;
    TLSFBlock** curr = &bucket->head;
    while (*curr) {
        if (guard++ > max_iter) {
            printf("[TLSF-CRITICAL] Hash chain corruption detected in hash_remove (bucket %u, count %u)\n",
                   hash, bucket->count);
            return;
        }
        if (*curr == block) {
            *curr = block->hash_next;
            bucket->count--;
            block->hash_next = NULL;
            return;
        }
        curr = &(*curr)->hash_next;
    }
}

static TLSFBlock* hash_find(PTXTLSFAllocator* allocator, void* device_ptr) {
    uint32_t hash = hash_ptr(device_ptr);
    TLSFHashBucket* bucket = &allocator->hash_table[hash];

    int guard = 0;
    int max_iter = bucket->count + 1;
    TLSFBlock* curr = bucket->head;
    while (curr) {
        if (guard++ > max_iter) {
            printf("[TLSF-CRITICAL] Hash chain corruption detected in hash_find (bucket %u, count %u)\n",
                   hash, bucket->count);
            return NULL;
        }
        if (curr->device_ptr == device_ptr) {
            return curr;
        }
        curr = curr->hash_next;
    }

    return NULL;
}

// ============================================================================
// Free list operations
// ============================================================================

static void insert_free_block(PTXTLSFAllocator* allocator, TLSFBlock* block) {
    int fl, sl;
    tlsf_mapping(block->size, &fl, &sl);
    
    // Bounds check
    if (fl < 0 || fl >= TLSF_FL_INDEX_MAX || sl < 0 || sl >= TLSF_SL_INDEX_COUNT) {
        printf("[TLSF-ERROR] Invalid indices: fl=%d, sl=%d\n", fl, sl);
        return;
    }
    
    block->next_free = allocator->free_lists[fl][sl];
    block->prev_free = NULL;
    
    if (allocator->free_lists[fl][sl]) {
        allocator->free_lists[fl][sl]->prev_free = block;
    }
    
    allocator->free_lists[fl][sl] = block;
    block->is_free = true;
    
    allocator->stats.free_blocks++;
    allocator->stats.free_list_counts[fl]++;
}

static void remove_free_block(PTXTLSFAllocator* allocator, TLSFBlock* block) {
    int fl, sl;
    tlsf_mapping(block->size, &fl, &sl);
    
    if (block->prev_free) {
        block->prev_free->next_free = block->next_free;
    } else {
        allocator->free_lists[fl][sl] = block->next_free;
    }
    
    if (block->next_free) {
        block->next_free->prev_free = block->prev_free;
    }
    
    block->is_free = false;
    allocator->stats.free_blocks--;
    allocator->stats.free_list_counts[fl]--;
}

static TLSFBlock* find_free_block(PTXTLSFAllocator* allocator, size_t size) {
    int fl, sl;
    tlsf_mapping(size, &fl, &sl);
    
    // Search in same second-level list first
    for (int s = sl; s < TLSF_SL_INDEX_COUNT; s++) {
        TLSFBlock* block = allocator->free_lists[fl][s];
        if (block && block->size >= size) {
            return block;
        }
    }
    
    // Search in higher first-level lists
    for (int f = fl + 1; f < TLSF_FL_INDEX_MAX; f++) {
        for (int s = 0; s < TLSF_SL_INDEX_COUNT; s++) {
            TLSFBlock* block = allocator->free_lists[f][s];
            if (block) {
                return block;
            }
        }
    }
    
    return NULL;
}

// ============================================================================
// Block operations
// ============================================================================

static void split_block(PTXTLSFAllocator* allocator, TLSFBlock* block, size_t size) {
    size_t remaining = block->size - size;
    
    if (remaining >= TLSF_MIN_BLOCK_SIZE) {
        // Create new block
        TLSFBlock* new_block = (TLSFBlock*)malloc(sizeof(TLSFBlock));
        memset(new_block, 0, sizeof(TLSFBlock));
        
        new_block->device_ptr = (char*)block->device_ptr + size;
        new_block->size = remaining;
        new_block->magic = TLSF_BLOCK_MAGIC;
        new_block->is_free = true;
        new_block->is_last = block->is_last;
        
        new_block->prev_phys = block;
        new_block->next_phys = block->next_phys;
        
        if (block->next_phys) {
            block->next_phys->prev_phys = new_block;
        } else {
            allocator->last_block = new_block;
        }
        
        block->size = size;
        block->next_phys = new_block;
        block->is_last = false;
        
        // Add to hash table and free list
        hash_insert(allocator, new_block);
        insert_free_block(allocator, new_block);
        
        allocator->stats.total_blocks++;
        allocator->stats.total_splits++;
    }
}

static TLSFBlock* coalesce(PTXTLSFAllocator* allocator, TLSFBlock* block) {
    // Coalesce with next block
    if (block->next_phys && block->next_phys->is_free && !block->next_phys->is_pinned) {
        TLSFBlock* next = block->next_phys;
        
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
        allocator->stats.total_merges++;
    }
    
    // Coalesce with previous block
    if (block->prev_phys && block->prev_phys->is_free && !block->prev_phys->is_pinned) {
        TLSFBlock* prev = block->prev_phys;
        
        remove_free_block(allocator, prev);
        hash_remove(allocator, block);
        
        prev->size += block->size;
        prev->next_phys = block->next_phys;
        prev->is_last = block->is_last;
        
        if (block->next_phys) {
            block->next_phys->prev_phys = prev;
        } else {
            allocator->last_block = prev;
        }
        
        free(block);
        allocator->stats.total_blocks--;
        allocator->stats.total_merges++;
        
        return prev;
    }
    
    return block;
}

