#pragma once

#include <cstdint>
#include <vector>

#include "fercuda/algorithms/layernorm_attachment.cuh"
#include "fercuda/algorithms/matmul_attachment.cuh"
#include "fercuda/core/runtime.cuh"
#include "fercuda/core/status.cuh"
#include "fercuda/runtime/api_types.cuh"
#include "fercuda/runtime/job_manager.cuh"
#include "fercuda/runtime/regime_manager.cuh"

namespace fer::runtime {

struct ExecBufferView {
    void* ptr = nullptr;
    BufferDesc desc{};
    uint32_t regime_id = static_cast<uint32_t>(MemoryRegime::CUSTOM_POOL);
};

class ExecPlanner {
public:
    ExecPlanner(RegimeManager* regime_mgr, KernelRegistry* registry, const OpContext* ctx);

    Status launch_matmul(
        const ExecBufferView& a,
        const ExecBufferView& b,
        const ExecBufferView& out,
        uint32_t memory_regime_override,
        std::vector<TempAlloc>* out_temps);

    Status launch_layer_norm(
        const ExecBufferView& x,
        const ExecBufferView& out,
        float eps,
        uint32_t memory_regime_override,
        std::vector<TempAlloc>* out_temps);

private:
    static bool same_regime(uint32_t lhs, uint32_t rhs) { return lhs == rhs; }
    static size_t bytes_for_desc(const BufferDesc& desc);
    static FTensor1D tensor1d_from(void* ptr, const BufferDesc& desc);
    static FTensor2D tensor2d_from(void* ptr, const BufferDesc& desc);
    Status free_temps(const std::vector<TempAlloc>& temps);

    RegimeManager* regime_mgr_ = nullptr;
    KernelRegistry* registry_ = nullptr;
    const OpContext* ctx_ = nullptr;
};

} // namespace fer::runtime
