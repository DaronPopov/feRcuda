#include "fercuda/jit/lowering/fusion.h"

namespace fer::jit::lowering {

namespace {

class FusionHintPass final : public LoweringPass {
public:
    const char* name() const override { return "fusion_hints"; }

    Status run(KernelModule* module, const DeviceCapabilities& caps) override {
        if (!module) return Status::invalid_argument("module is null");
        if (module->source.empty()) return Status::invalid_argument("module source is empty");

        // MVP behavior: honor source-level hints for future graph fusion work.
        // This keeps contract continuity now without silently rewriting kernels.
        if (module->source.find("fercuda:fuse") != std::string::npos) {
            module->notes.emplace_back("fusion hint detected");
            if (caps.tensor_cores) {
                module->notes.emplace_back("tensor-core capable target");
            }
        }
        return Status::ok_status();
    }
};

class ElementwiseFusionRewritePass final : public LoweringPass {
public:
    const char* name() const override { return "fusion_rewrite_elementwise"; }

    static bool is_elementwise(IROpKind op) {
        return op == IROpKind::ElementwiseAdd ||
               op == IROpKind::ElementwiseMul ||
               op == IROpKind::ElementwiseRelu;
    }

    Status run(KernelModule* module, const DeviceCapabilities&) override {
        if (!module) return Status::invalid_argument("module is null");
        if (module->source.empty()) return Status::invalid_argument("module source is empty");
        if (!module->fuse_elementwise_chain) return Status::ok_status();

        int first = -1;
        int last = -1;
        int count = 0;
        for (size_t i = 0; i < module->ir.nodes.size(); ++i) {
            const IRNode& n = module->ir.nodes[i];
            if (n.removed || !is_elementwise(n.op)) continue;
            if (first < 0) first = static_cast<int>(i);
            last = static_cast<int>(i);
            count++;
        }

        if (count >= 2 && first >= 0 && last >= 0) {
            IRNode fused{};
            fused.id = static_cast<int>(module->ir.nodes.size());
            fused.op = IROpKind::FusedElementwiseChain;
            fused.inputs = module->ir.nodes[first].inputs;
            fused.output = module->ir.nodes[last].output;
            for (int i = first; i <= last; ++i) {
                if (module->ir.nodes[static_cast<size_t>(i)].removed) continue;
                if (is_elementwise(module->ir.nodes[static_cast<size_t>(i)].op)) {
                    module->ir.nodes[static_cast<size_t>(i)].removed = true;
                }
            }
            module->ir.nodes.push_back(fused);
            module->notes.emplace_back("elementwise IR chain fused");
        }

        bool has_fused = false;
        for (const auto& n : module->ir.nodes) {
            if (!n.removed && n.op == IROpKind::FusedElementwiseChain) {
                has_fused = true;
                break;
            }
        }
        if (has_fused && module->source.find("#define FERCUDA_FUSED_ELEMENTWISE 1") == std::string::npos) {
            module->source = std::string("#define FERCUDA_FUSED_ELEMENTWISE 1\n") + module->source;
            module->modified = true;
            module->notes.emplace_back("elementwise fusion rewrite materialized");
        }
        return Status::ok_status();
    }
};

} // namespace

std::unique_ptr<LoweringPass> make_fusion_hint_pass() {
    return std::make_unique<FusionHintPass>();
}

std::unique_ptr<LoweringPass> make_elementwise_fusion_rewrite_pass() {
    return std::make_unique<ElementwiseFusionRewritePass>();
}

} // namespace fer::jit::lowering
