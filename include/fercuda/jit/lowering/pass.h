#pragma once

#include "fercuda/core/status.cuh"
#include "fercuda/jit/lowering/capabilities.h"
#include "fercuda/jit/lowering/ir.h"

namespace fer::jit::lowering {

class LoweringPass {
public:
    virtual ~LoweringPass() = default;
    virtual const char* name() const = 0;
    virtual Status run(KernelModule* module, const DeviceCapabilities& caps) = 0;
};

} // namespace fer::jit::lowering
