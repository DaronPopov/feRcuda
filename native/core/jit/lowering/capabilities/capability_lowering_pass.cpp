#include "fercuda/jit/lowering/fusion.h"

#include <sstream>

namespace fer::jit::lowering {

namespace {

class CapabilityLoweringPass final : public LoweringPass {
public:
    const char* name() const override { return "capability_lowering"; }

    Status run(KernelModule* module, const DeviceCapabilities& caps) override {
        if (!module) return Status::invalid_argument("module is null");
        if (module->source.empty()) return Status::invalid_argument("module source is empty");

        const int sm = caps.sm_major * 10 + caps.sm_minor;
        if (module->min_sm > 0 && sm < module->min_sm) {
            return Status::invalid_argument("required SM not available");
        }
        if (module->wants_tensor_cores && !caps.tensor_cores) {
            return Status::invalid_argument("tensor cores required but unavailable");
        }
        if (module->wants_cooperative_groups && !caps.cooperative_groups) {
            return Status::invalid_argument("cooperative groups required but unavailable");
        }

        std::ostringstream oss;
        oss << "-DFERCUDA_TARGET_SM=" << sm;
        module->extra_nvrtc_opts.push_back(oss.str());
        if (module->wants_tensor_cores) {
            module->extra_nvrtc_opts.push_back("-DFERCUDA_ENABLE_TENSOR_CORES=1");
        }
        if (module->wants_cooperative_groups) {
            module->extra_nvrtc_opts.push_back("-DFERCUDA_ENABLE_COOP_GROUPS=1");
        }
        module->notes.emplace_back("capability contracts validated");
        return Status::ok_status();
    }
};

} // namespace

std::unique_ptr<LoweringPass> make_capability_lowering_pass() {
    return std::make_unique<CapabilityLoweringPass>();
}

} // namespace fer::jit::lowering
