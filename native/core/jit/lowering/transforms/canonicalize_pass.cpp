#include "fercuda/jit/lowering/fusion.h"

namespace fer::jit::lowering {

namespace {

class CanonicalizePass final : public LoweringPass {
public:
    const char* name() const override { return "canonicalize"; }

    Status run(KernelModule* module, const DeviceCapabilities&) override {
        if (!module) return Status::invalid_argument("module is null");
        if (module->source.empty()) return Status::invalid_argument("module source is empty");
        if (module->source.back() != '\n') {
            module->source.push_back('\n');
            module->modified = true;
        }
        module->notes.emplace_back("canonicalized source formatting");
        return Status::ok_status();
    }
};

} // namespace

std::unique_ptr<LoweringPass> make_canonicalize_pass() {
    return std::make_unique<CanonicalizePass>();
}

} // namespace fer::jit::lowering
