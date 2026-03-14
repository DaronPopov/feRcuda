cmake_minimum_required(VERSION 3.20)
project(feRcuda LANGUAGES C CXX CUDA)

enable_testing()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

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

if(DEFINED ENV{FERCUDA_GPU_HOT_ROOT})
    set(FERCUDA_GPU_HOT_ROOT "$ENV{FERCUDA_GPU_HOT_ROOT}" CACHE PATH
        "Path to GPU Hot Runtime project root")
else()
    set(FERCUDA_GPU_HOT_ROOT "" CACHE PATH
        "Path to GPU Hot Runtime project root (set FERCUDA_GPU_HOT_ROOT env var or -DFERCUDA_GPU_HOT_ROOT=...)")
endif()

# Deterministic math: no fast-math, no implicit FMA fusion.
set(CUDA_DETERMINISTIC_FLAGS
    -fmad=false
    -Xptxas=-O3
    --generate-line-info
)

set(FERCUDA_LIBRARY_SOURCES
    src/runtime/elastic_pool.cu
    src/runtime/ops_bridge.cu
    src/runtime/regime_manager.cu
    src/runtime/job_manager.cu
    src/runtime/exec_planner.cu
    src/runtime/session.cu
)

set(FERCUDA_CUDA_TESTS
    test_types tests/core/test_types.cu
    test_ca tests/experimental/test_ca.cu
    test_ca_kernels tests/experimental/test_ca_kernels.cu
    test_registry tests/runtime/test_registry.cu
    test_runtime_session tests/runtime/test_runtime_session.cu
    test_memory_regime tests/memory/test_memory_regime.cu
    test_custom_allocator tests/memory/test_custom_allocator.cu
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

function(fercuda_add_smoke_test target)
    if(TARGET ${target})
        add_test(NAME ${target} COMMAND ${target})
    endif()
endfunction()

add_library(fercuda STATIC ${FERCUDA_LIBRARY_SOURCES})
target_include_directories(fercuda PUBLIC include)
target_link_libraries(fercuda PUBLIC ptx_core ptx_kernels)
target_compile_options(fercuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DETERMINISTIC_FLAGS}>
)
set_target_properties(fercuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# =============================================================================
# Native PTX-OS core layer (in-tree low-level GPU runtime)
# =============================================================================

set(PTX_CORE_SOURCES
    native/core/runtime/hot_runtime.cu
    native/core/runtime/ptx_stable_runtime.cu
    native/core/runtime/cuda_alloc_hook.cu
    native/core/runtime/hot_multi_gpu.cu
    native/core/memory/tlsf_allocator.cu
    native/core/memory/elastic_pool.cu
    native/core/os/vmm.cu
    native/core/os/vfs.cu
)

add_library(ptx_core SHARED ${PTX_CORE_SOURCES})
target_include_directories(ptx_core PUBLIC native/core/include)
target_include_directories(ptx_core PRIVATE native/core)
target_link_libraries(ptx_core PUBLIC CUDA::cudart CUDA::cuda_driver)
target_link_libraries(ptx_core PRIVATE ptx_kernels)
target_compile_options(ptx_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true ${CUDA_DETERMINISTIC_FLAGS}>
)
set_target_properties(ptx_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

set(PTX_KERNEL_SOURCES
    native/core/kernels/tensor_kernels.cu
    native/core/kernels/tensor_backward.cu
    native/core/kernels/graphics_kernels.cu
    native/core/kernels/os_kernel.cu
)

add_library(ptx_kernels SHARED ${PTX_KERNEL_SOURCES})
target_include_directories(ptx_kernels PUBLIC native/core/include)
target_include_directories(ptx_kernels PRIVATE native/core)
target_link_libraries(ptx_kernels PRIVATE CUDA::cudart)
target_compile_options(ptx_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true ${CUDA_DETERMINISTIC_FLAGS}>
)
set_target_properties(ptx_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

if(UNIX AND NOT APPLE)
    add_library(ptx_hook SHARED
        native/core/hooks/cuda_intercept.c
        native/core/hooks/cuda_context_hook.c
    )
    target_include_directories(ptx_hook PRIVATE native/core/include)
    target_link_libraries(ptx_hook PRIVATE ptx_core CUDA::cudart CUDA::cuda_driver dl)
    target_compile_definitions(ptx_hook PRIVATE _GNU_SOURCE)
    set_target_properties(ptx_hook PROPERTIES
        OUTPUT_NAME ptx_hook
        POSITION_INDEPENDENT_CODE ON
        LINKER_LANGUAGE C
    )
endif()

# =============================================================================
# feRcuda C API shared library
# =============================================================================

add_library(fercuda_capi SHARED
    src/runtime/c_api.cu
    src/runtime/mcp_adapter.cpp
    src/runtime/mcp_json_adapter.cpp
    native/core/jit/jit_manager.cu
    native/core/jit/script_runtime.cpp
    native/core/jit/intent/intent_runtime.cpp
    native/core/jit/lowering/capabilities/cuda_caps.cpp
    native/core/jit/lowering/capabilities/capability_lowering_pass.cpp
    native/core/jit/lowering/fusion/fusion_pass.cpp
    native/core/jit/lowering/pipeline/lowering_pipeline.cpp
    native/core/jit/lowering/transforms/pragma_parse_pass.cpp
    native/core/jit/lowering/transforms/canonicalize_pass.cpp
)
target_include_directories(fercuda_capi PUBLIC include)
target_link_libraries(fercuda_capi PRIVATE fercuda CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
if(UNIX AND NOT APPLE)
    target_link_libraries(fercuda_capi PRIVATE dl)
endif()
set_target_properties(fercuda_capi PROPERTIES
    OUTPUT_NAME fercuda_capi
)

# fercuda_intercept removed — use native ptx_hook instead

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
    fercuda_add_smoke_test(test_gpu_hot_runtime_regime)
endif()

# Intercept tests now use native ptx_hook
if(TARGET ptx_hook)
    add_executable(test_intercept_compat tests/intercept/test_intercept_compat.cpp)
    target_include_directories(test_intercept_compat PRIVATE include native/core/include)
    target_link_libraries(test_intercept_compat PRIVATE ptx_hook CUDA::cudart CUDA::cuda_driver)
    fercuda_add_smoke_test(test_intercept_compat)

    add_executable(test_intercept_stress tests/intercept/test_intercept_stress.cpp)
    target_include_directories(test_intercept_stress PRIVATE include native/core/include)
    target_link_libraries(test_intercept_stress PRIVATE ptx_hook ptx_core ptx_kernels CUDA::cudart Threads::Threads)
    add_test(NAME test_intercept_stress COMMAND test_intercept_stress)
    set_tests_properties(test_intercept_stress PROPERTIES TIMEOUT 300)
endif()

fercuda_add_smoke_test(test_c_api_bridge)
fercuda_add_smoke_test(test_c_api_custom_allocator)
fercuda_add_smoke_test(test_c_api_external_ptr_interop)
fercuda_add_smoke_test(test_c_api_helper_api)
fercuda_add_smoke_test(test_c_api_jit_exec)
fercuda_add_smoke_test(test_c_api_jit_intent)
fercuda_add_smoke_test(test_c_api_jit_lowering)
fercuda_add_smoke_test(test_c_api_jit_stub)
fercuda_add_smoke_test(test_c_api_script_runtime)
fercuda_add_smoke_test(test_c_api_script_timeout)
fercuda_add_smoke_test(test_c_api_stream_handoff)

install(TARGETS fercuda fercuda_capi
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)


install(DIRECTORY include/
    DESTINATION include
)

# =============================================================================
# Native PTX-OS targets install & tests
# =============================================================================

install(TARGETS ptx_core ptx_kernels
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

if(TARGET ptx_hook)
    install(TARGETS ptx_hook
        LIBRARY DESTINATION lib
    )
endif()

install(DIRECTORY native/core/include/
    DESTINATION include/ptx
)

# Native smoke tests
add_executable(test_ptx_core_init tests/native/test_ptx_core_init.cpp)
target_include_directories(test_ptx_core_init PRIVATE native/core/include)
target_link_libraries(test_ptx_core_init PRIVATE ptx_core ptx_kernels CUDA::cudart CUDA::cuda_driver)
fercuda_add_smoke_test(test_ptx_core_init)

add_executable(test_ptx_tlsf tests/native/test_ptx_tlsf.cpp)
target_include_directories(test_ptx_tlsf PRIVATE native/core/include)
target_link_libraries(test_ptx_tlsf PRIVATE ptx_core ptx_kernels CUDA::cudart CUDA::cuda_driver)
fercuda_add_smoke_test(test_ptx_tlsf)

add_executable(test_ptx_tensor_ops tests/native/test_ptx_tensor_ops.cu)
target_include_directories(test_ptx_tensor_ops PRIVATE native/core/include)
target_link_libraries(test_ptx_tensor_ops PRIVATE ptx_core ptx_kernels CUDA::cudart CUDA::cuda_driver)
target_compile_options(test_ptx_tensor_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
set_target_properties(test_ptx_tensor_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
fercuda_add_smoke_test(test_ptx_tensor_ops)

add_executable(test_ptx_prod_stress tests/native/test_ptx_prod_stress.cu)
target_include_directories(test_ptx_prod_stress PRIVATE native/core/include)
target_link_libraries(test_ptx_prod_stress PRIVATE ptx_core ptx_kernels CUDA::cudart CUDA::cuda_driver)
target_compile_options(test_ptx_prod_stress PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
set_target_properties(test_ptx_prod_stress PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
fercuda_add_smoke_test(test_ptx_prod_stress)
