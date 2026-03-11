cmake_minimum_required(VERSION 3.20)
project(feRcuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Prefer the local GPU's compute capability when available. This keeps fresh
# builds portable across developer workstations and ARM64 systems like DGX
# Spark, while still allowing scripts/install.sh --arch-list to override.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
    if(NVIDIA_SMI_EXECUTABLE)
        execute_process(
            COMMAND ${NVIDIA_SMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE FERCUDA_COMPUTE_CAPS_RAW
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(FERCUDA_COMPUTE_CAPS_RAW)
            string(REPLACE "\r" "" FERCUDA_COMPUTE_CAPS_RAW "${FERCUDA_COMPUTE_CAPS_RAW}")
            string(REPLACE "\n" ";" FERCUDA_COMPUTE_CAPS_LIST "${FERCUDA_COMPUTE_CAPS_RAW}")
            set(FERCUDA_CUDA_ARCH_LIST "")
            foreach(FERCUDA_CC IN LISTS FERCUDA_COMPUTE_CAPS_LIST)
                string(STRIP "${FERCUDA_CC}" FERCUDA_CC)
                if(FERCUDA_CC STREQUAL "")
                    continue()
                endif()
                string(REPLACE "." "" FERCUDA_ARCH "${FERCUDA_CC}")
                if(FERCUDA_ARCH MATCHES "^[0-9]+$")
                    list(APPEND FERCUDA_CUDA_ARCH_LIST "${FERCUDA_ARCH}")
                endif()
            endforeach()
            list(REMOVE_DUPLICATES FERCUDA_CUDA_ARCH_LIST)
            if(FERCUDA_CUDA_ARCH_LIST)
                set(CMAKE_CUDA_ARCHITECTURES "${FERCUDA_CUDA_ARCH_LIST}" CACHE STRING
                    "CUDA architectures to build for" FORCE)
                message(STATUS "Auto-detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
            endif()
        endif()
    endif()
endif()

set(FERCUDA_GPU_HOT_ROOT "/home/daron/persistant_gpu_os" CACHE PATH
    "Path to GPU Hot Runtime project root")

# Deterministic math: no fast-math, no implicit FMA fusion.
set(CUDA_DETERMINISTIC_FLAGS
    -fmad=false
    -Xptxas=-O3
    --generate-line-info
)

set(FERCUDA_LIBRARY_SOURCES
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

set(FERCUDA_CUDA_TESTS
    test_types tests/core/test_types.cu
    test_determinism tests/core/test_determinism.cu
    test_ca tests/experimental/test_ca.cu
    test_ca_kernels tests/experimental/test_ca_kernels.cu
    test_registry tests/runtime/test_registry.cu
    test_runtime_session tests/runtime/test_runtime_session.cu
    test_memory_regime tests/memory/test_memory_regime.cu
    test_custom_allocator tests/memory/test_custom_allocator.cu
    test_pool_fragmentation_ab tests/memory/test_pool_fragmentation_ab.cu
    test_pool_reclaim tests/memory/test_pool_reclaim.cu
    test_sched_dma_then_ops tests/scheduler/test_sched_dma_then_ops.cu
    test_sched_minimal tests/scheduler/test_sched_minimal.cu
    test_sched_ops_interop tests/scheduler/test_sched_ops_interop.cu
    test_sched_pool_interop tests/scheduler/test_sched_pool_interop.cu
    test_sched_smwarp tests/scheduler/test_sched_smwarp.cu
    test_sched_work tests/scheduler/test_sched_work.cu
    test_h2d_isolate tests/h2d/test_h2d_isolate.cu
    test_h2d_simple_loop tests/h2d/test_h2d_simple_loop.cu
    test_h2d_while_sched tests/h2d/test_h2d_while_sched.cu
)

function(fercuda_configure_cuda_target target)
    target_include_directories(${target} PRIVATE include)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DETERMINISTIC_FLAGS}>
    )
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

add_library(fercuda STATIC ${FERCUDA_LIBRARY_SOURCES})
target_include_directories(fercuda PUBLIC include)
target_compile_options(fercuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DETERMINISTIC_FLAGS}>
)
set_target_properties(fercuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_library(fercuda_capi SHARED
    src/runtime/c_api.cu
    src/runtime/mcp_adapter.cpp
    src/runtime/mcp_json_adapter.cpp
    src/jit/jit_manager.cu
    src/jit/script_runtime.cpp
    src/jit/intent/intent_runtime.cpp
    src/jit/lowering/capabilities/cuda_caps.cpp
    src/jit/lowering/capabilities/capability_lowering_pass.cpp
    src/jit/lowering/fusion/fusion_pass.cpp
    src/jit/lowering/pipeline/lowering_pipeline.cpp
    src/jit/lowering/transforms/pragma_parse_pass.cpp
    src/jit/lowering/transforms/canonicalize_pass.cpp
)
target_include_directories(fercuda_capi PUBLIC include)
target_link_libraries(fercuda_capi PRIVATE fercuda CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
if(UNIX AND NOT APPLE)
    target_link_libraries(fercuda_capi PRIVATE dl)
endif()
set_target_properties(fercuda_capi PROPERTIES
    OUTPUT_NAME fercuda_capi
)

if(UNIX AND NOT APPLE)
    add_library(fercuda_intercept SHARED src/intercept/cuda_alloc_intercept.cpp)
    target_include_directories(fercuda_intercept PRIVATE include)
    target_link_libraries(fercuda_intercept PRIVATE CUDA::cudart CUDA::cuda_driver dl)
    target_compile_definitions(fercuda_intercept PRIVATE _GNU_SOURCE)
    set_target_properties(fercuda_intercept PROPERTIES
        OUTPUT_NAME fercuda_intercept
        POSITION_INDEPENDENT_CODE ON
    )
endif()

list(LENGTH FERCUDA_CUDA_TESTS FERCUDA_CUDA_TEST_COUNT)
math(EXPR FERCUDA_CUDA_TEST_LAST "${FERCUDA_CUDA_TEST_COUNT} - 1")
foreach(idx RANGE 0 ${FERCUDA_CUDA_TEST_LAST} 2)
    math(EXPR source_idx "${idx} + 1")
    list(GET FERCUDA_CUDA_TESTS ${idx} target_name)
    list(GET FERCUDA_CUDA_TESTS ${source_idx} source_path)
    add_executable(${target_name} ${source_path})
    target_link_libraries(${target_name} PRIVATE fercuda)
    fercuda_configure_cuda_target(${target_name})
endforeach()

add_executable(test_c_api_bridge tests/c_api/test_c_api_bridge.cpp)
target_include_directories(test_c_api_bridge PRIVATE include)
target_link_libraries(test_c_api_bridge PRIVATE fercuda_capi)

add_executable(test_c_api_custom_allocator tests/c_api/test_c_api_custom_allocator.cpp)
target_include_directories(test_c_api_custom_allocator PRIVATE include)
target_link_libraries(test_c_api_custom_allocator PRIVATE fercuda_capi CUDA::cudart)

add_executable(test_c_api_jit_stub tests/c_api/test_c_api_jit_stub.cpp)
target_include_directories(test_c_api_jit_stub PRIVATE include)
target_link_libraries(test_c_api_jit_stub PRIVATE fercuda_capi)

add_executable(test_c_api_jit_exec tests/c_api/test_c_api_jit_exec.cpp)
target_include_directories(test_c_api_jit_exec PRIVATE include)
target_link_libraries(test_c_api_jit_exec PRIVATE fercuda_capi)

add_executable(test_c_api_jit_lowering tests/c_api/test_c_api_jit_lowering.cpp)
target_include_directories(test_c_api_jit_lowering PRIVATE include)
target_link_libraries(test_c_api_jit_lowering PRIVATE fercuda_capi)

add_executable(test_c_api_jit_intent tests/c_api/test_c_api_jit_intent.cpp)
target_include_directories(test_c_api_jit_intent PRIVATE include)
target_link_libraries(test_c_api_jit_intent PRIVATE fercuda_capi)

add_executable(test_c_api_external_ptr_interop tests/c_api/test_c_api_external_ptr_interop.cpp)
target_include_directories(test_c_api_external_ptr_interop PRIVATE include)
target_link_libraries(test_c_api_external_ptr_interop PRIVATE fercuda_capi CUDA::cudart)

add_executable(test_c_api_stream_handoff tests/c_api/test_c_api_stream_handoff.cpp)
target_include_directories(test_c_api_stream_handoff PRIVATE include)
target_link_libraries(test_c_api_stream_handoff PRIVATE fercuda_capi CUDA::cudart)

add_executable(test_c_api_helper_api tests/c_api/test_c_api_helper_api.cpp)
target_include_directories(test_c_api_helper_api PRIVATE include)
target_link_libraries(test_c_api_helper_api PRIVATE fercuda_capi)

add_executable(test_c_api_agent_mcp_adapter tests/c_api/test_c_api_agent_mcp_adapter.cpp)
target_include_directories(test_c_api_agent_mcp_adapter PRIVATE include)
target_link_libraries(test_c_api_agent_mcp_adapter PRIVATE fercuda_capi)

add_executable(test_c_api_agent_mcp_json_adapter tests/c_api/test_c_api_agent_mcp_json_adapter.cpp)
target_include_directories(test_c_api_agent_mcp_json_adapter PRIVATE include)
target_link_libraries(test_c_api_agent_mcp_json_adapter PRIVATE fercuda_capi)

add_library(test_script_module MODULE tests/jit/test_script_module.cpp)
set_target_properties(test_script_module PROPERTIES PREFIX "")
target_include_directories(test_script_module PRIVATE include)
target_link_libraries(test_script_module PRIVATE fercuda_capi)

add_library(test_script_cancel_module MODULE tests/jit/test_script_cancel_module.cpp)
set_target_properties(test_script_cancel_module PROPERTIES PREFIX "")
target_include_directories(test_script_cancel_module PRIVATE include)
target_link_libraries(test_script_cancel_module PRIVATE fercuda_capi)

add_executable(test_c_api_script_runtime tests/c_api/test_c_api_script_runtime.cpp)
target_include_directories(test_c_api_script_runtime PRIVATE include)
target_link_libraries(test_c_api_script_runtime PRIVATE fercuda_capi)
target_compile_definitions(test_c_api_script_runtime PRIVATE
    TEST_SCRIPT_MODULE_PATH="$<TARGET_FILE:test_script_module>")

add_executable(test_c_api_script_timeout tests/c_api/test_c_api_script_timeout.cpp)
target_include_directories(test_c_api_script_timeout PRIVATE include)
target_link_libraries(test_c_api_script_timeout PRIVATE fercuda_capi)
target_compile_definitions(test_c_api_script_timeout PRIVATE
    TEST_SCRIPT_CANCEL_MODULE_PATH="$<TARGET_FILE:test_script_cancel_module>")

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

if(TARGET fercuda_intercept)
    add_executable(test_intercept_compat tests/intercept/test_intercept_compat.cpp)
    target_include_directories(test_intercept_compat PRIVATE include)
    target_link_libraries(test_intercept_compat PRIVATE fercuda_intercept CUDA::cudart CUDA::cuda_driver)
    add_test(NAME test_intercept_compat COMMAND test_intercept_compat)

    add_executable(bench_external_alloc tests/intercept/bench_external_alloc.cu)
    target_include_directories(bench_external_alloc PRIVATE include)
    target_link_libraries(bench_external_alloc PRIVATE CUDA::cudart CUDA::cuda_driver dl)
    fercuda_configure_cuda_target(bench_external_alloc)
endif()

add_executable(demo_persistent examples/demo_persistent.cu)
target_link_libraries(demo_persistent PRIVATE fercuda)
fercuda_configure_cuda_target(demo_persistent)

enable_testing()

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
