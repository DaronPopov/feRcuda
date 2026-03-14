/*
 * GPU Hot Runtime - Multi-GPU Implementation
 * Distributed computing across multiple GPUs
 */

#include "gpu/gpu_hot_multigpu.h"
#include "ptx_debug.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <cfloat>

#include "runtime/modules/hot_multi_gpu_primitives.inl"
#include "runtime/modules/hot_multi_gpu_policy_main.inl"
#include "runtime/modules/hot_multi_gpu_reduction.inl"
#include "runtime/modules/hot_multi_gpu_policy_tail.inl"
