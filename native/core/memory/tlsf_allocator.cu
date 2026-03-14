/*
 * PTX-OS Enhanced TLSF Memory Allocator - Implementation
 *
 * This is a production-grade TLSF allocator with:
 * - O(1) allocation/deallocation via segregated free lists
 * - O(1) block lookup via hash table
 * - Comprehensive diagnostics and health monitoring
 * - Thread-safe operations
 * - Memory leak detection
 */

#include "memory/ptx_tlsf_allocator.h"
#include "memory/ptx_cuda_driver.h"
#include "ptx_debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <pthread.h>
#include <time.h>
#endif

#include "memory/modules/tlsf_allocator_internals.inl"
#include "memory/modules/tlsf_allocator_api.inl"
