#pragma once

#include "fercuda/core/status.cuh"
#include "fercuda/jit/types.h"
#include "fercuda/jit/lowering/capabilities.h"
#include "fercuda/jit/lowering/ir.h"

namespace fer::jit::lowering {

class LoweringPipeline {
public:
    static Status run(
        const fer_jit_source_t* source,
        const fer_jit_options_t* options,
        KernelModule* out_module,
        DeviceCapabilities* out_caps);
};

} // namespace fer::jit::lowering
