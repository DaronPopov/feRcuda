/*
 * GPU Hot Runtime - Implementation
 * Persistent GPU context with pre-allocated VRAM pools
 */

#include "gpu/gpu_hot_runtime.h"
#include "memory/ptx_tlsf_allocator.h"
#include "memory/ptx_cuda_driver.h"
#include "ptx_debug.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <time.h>
#include <errno.h>
#endif

#include "runtime/modules/hot_runtime_prelude.inl"
#include "runtime/modules/hot_runtime_init.inl"
#include "runtime/modules/hot_runtime_recovery.inl"
#include "runtime/modules/hot_runtime_stream_dispatch.inl"
#include "runtime/modules/hot_runtime_graphs_tlsf.inl"
#include "runtime/modules/hot_runtime_shared_context.inl"
#include "runtime/modules/hot_runtime_tasks.inl"
