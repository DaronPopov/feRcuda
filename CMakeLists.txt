cmake_minimum_required(VERSION 3.20)
project(feRcuda LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Allow caller-provided arch list. If omitted, use a portable default that
# includes datacenter + modern Jetson targets.
set(FERCUDA_DEFAULT_CUDA_ARCHS "75;80;86;87;89;90" CACHE STRING
    "Default CUDA architectures for feRcuda when CMAKE_CUDA_ARCHITECTURES is not provided")
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${FERCUDA_DEFAULT_CUDA_ARCHS}" CACHE STRING
        "CUDA architectures to build for" FORCE)
endif()
message(STATUS "feRcuda CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(FERCUDA_GPU_HOT_ROOT "/home/daron/persistant_gpu_os" CACHE PATH "Path to GPU Hot Runtime project root")

# Deterministic math: no fast-math, no implicit FMA fusion.
# -fmad=false prevents nvcc from fusing a*b+c into FFMA (changes rounding).
# We deliberately omit --use_fast_math (off by default).
set(CUDA_DETERMINISTIC_FLAGS
    -fmad=false
    --prec-div=true
    --prec-sqrt=true
    -Xptxas=-O3
    --generate-line-info
)

# ─── Library ───────────────────────────────────────────────────────────────────
add_library(fercuda STATIC
    src/memory/elastic_pool.cu
    src/scheduler/persistent_scheduler.cu
    src/runtime/regime_manager.cu
    src/runtime/job_manager.cu
    src/runtime/exec_planner.cu
    src/runtime/session.cu
    src/ops/elementwise.cu
    src/ops/matmul.cu
    src/ops/activation.cu
    src/ops/norm.cu
)

target_include_directories(fercuda PUBLIC include)

target_compile_options(fercuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DETERMINISTIC_FLAGS}>
)
target_compile_definitions(fercuda PRIVATE FERCUDA_FLIGHT_MODE=1)

set_target_properties(fercuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_library(fercuda_capi SHARED
    src/runtime/c_api.cu
)
target_include_directories(fercuda_capi PUBLIC include)
target_link_libraries(fercuda_capi PRIVATE fercuda CUDA::cudart)
set_target_properties(fercuda_capi PROPERTIES
    OUTPUT_NAME fercuda_capi
)

if(UNIX AND NOT APPLE)
    add_library(fercuda_intercept SHARED
        src/intercept/cuda_alloc_intercept.cpp
    )
    target_include_directories(fercuda_intercept PRIVATE include)
    target_link_libraries(fercuda_intercept PRIVATE CUDA::cudart CUDA::cuda_driver dl)
    target_compile_definitions(fercuda_intercept PRIVATE _GNU_SOURCE)
    set_target_properties(fercuda_intercept PROPERTIES
        OUTPUT_NAME fercuda_intercept
    )
endif()

# ─── Tests ─────────────────────────────────────────────────────────────────────
add_executable(test_types        tests/core/test_types.cu)
add_executable(test_determinism  tests/core/test_determinism.cu)
add_executable(test_ca           tests/experimental/test_ca.cu)
add_executable(test_ca_kernels   tests/experimental/test_ca_kernels.cu)
add_executable(test_registry     tests/runtime/test_registry.cu)
add_executable(test_runtime_session tests/runtime/test_runtime_session.cu)
add_executable(test_memory_regime tests/memory/test_memory_regime.cu)
add_executable(test_custom_allocator tests/memory/test_custom_allocator.cu)
add_executable(test_pool_reclaim tests/memory/test_pool_reclaim.cu)
add_executable(test_pool_fragmentation_ab tests/memory/test_pool_fragmentation_ab.cu)
add_executable(test_c_api_bridge tests/c_api/test_c_api_bridge.cpp)
add_executable(test_c_api_custom_allocator tests/c_api/test_c_api_custom_allocator.cpp)

foreach(t test_types test_determinism test_ca test_ca_kernels test_registry test_runtime_session test_memory_regime test_custom_allocator test_pool_reclaim test_pool_fragmentation_ab)
    target_include_directories(${t} PRIVATE include)
    target_link_libraries(${t} PRIVATE fercuda)
    target_compile_options(${t} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DETERMINISTIC_FLAGS}>
    )
    set_target_properties(${t} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

target_include_directories(test_c_api_bridge PRIVATE include)
target_link_libraries(test_c_api_bridge PRIVATE fercuda_capi)
target_include_directories(test_c_api_custom_allocator PRIVATE include)
target_link_libraries(test_c_api_custom_allocator PRIVATE fercuda_capi CUDA::cudart)

set(FERCUDA_GPU_HOT_INCLUDE_DIR "${FERCUDA_GPU_HOT_ROOT}/include")
set(FERCUDA_GPU_HOT_LIB "${FERCUDA_GPU_HOT_ROOT}/build/libptx_os_shared.so")
if(EXISTS "${FERCUDA_GPU_HOT_INCLUDE_DIR}/gpu/gpu_hot_runtime.h" AND EXISTS "${FERCUDA_GPU_HOT_LIB}")
    add_executable(test_gpu_hot_runtime_regime tests/integration/test_gpu_hot_runtime_regime.cpp)
    target_include_directories(test_gpu_hot_runtime_regime PRIVATE include "${FERCUDA_GPU_HOT_INCLUDE_DIR}")
    target_link_libraries(test_gpu_hot_runtime_regime PRIVATE fercuda_capi CUDA::cudart "${FERCUDA_GPU_HOT_LIB}")
    set_target_properties(test_gpu_hot_runtime_regime PROPERTIES
        BUILD_RPATH "${FERCUDA_GPU_HOT_ROOT}/build"
    )
endif()

# ─── Examples ──────────────────────────────────────────────────────────────────
add_executable(test_sched_minimal     tests/scheduler/test_sched_minimal.cu)
target_include_directories(test_sched_minimal PRIVATE include)
target_link_libraries(test_sched_minimal PRIVATE fercuda)
set_target_properties(test_sched_minimal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_sched_work        tests/scheduler/test_sched_work.cu)
target_include_directories(test_sched_work PRIVATE include)
target_link_libraries(test_sched_work PRIVATE fercuda)
set_target_properties(test_sched_work PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_sched_ops_interop tests/scheduler/test_sched_ops_interop.cu)
target_include_directories(test_sched_ops_interop PRIVATE include)
target_link_libraries(test_sched_ops_interop PRIVATE fercuda)
set_target_properties(test_sched_ops_interop PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_sched_dma_then_ops tests/scheduler/test_sched_dma_then_ops.cu)
target_include_directories(test_sched_dma_then_ops PRIVATE include)
target_link_libraries(test_sched_dma_then_ops PRIVATE fercuda)
set_target_properties(test_sched_dma_then_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_sched_pool_interop tests/scheduler/test_sched_pool_interop.cu)
target_include_directories(test_sched_pool_interop PRIVATE include)
target_link_libraries(test_sched_pool_interop PRIVATE fercuda)
set_target_properties(test_sched_pool_interop PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_h2d_while_sched   tests/h2d/test_h2d_while_sched.cu)

add_executable(test_h2d_simple_loop   tests/h2d/test_h2d_simple_loop.cu)
target_include_directories(test_h2d_simple_loop PRIVATE include)
target_link_libraries(test_h2d_simple_loop PRIVATE fercuda)
set_target_properties(test_h2d_simple_loop PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_h2d_isolate       tests/h2d/test_h2d_isolate.cu)
target_include_directories(test_h2d_isolate PRIVATE include)
target_link_libraries(test_h2d_isolate PRIVATE fercuda)
set_target_properties(test_h2d_isolate PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_h2d_while_sched PRIVATE include)
target_link_libraries(test_h2d_while_sched PRIVATE fercuda)
set_target_properties(test_h2d_while_sched PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(demo_persistent   examples/demo_persistent.cu)
target_include_directories(demo_persistent PRIVATE include)
target_link_libraries(demo_persistent PRIVATE fercuda)
target_compile_options(demo_persistent PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DETERMINISTIC_FLAGS}>
)
set_target_properties(demo_persistent PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

message(STATUS "feRcuda Regime Baseline: ON")

# ─── Install ───────────────────────────────────────────────────────────────────
install(TARGETS fercuda fercuda_capi
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

if(TARGET fercuda_intercept)
    install(TARGETS fercuda_intercept
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

install(DIRECTORY include/
    DESTINATION include
)
