#include "fercuda/jit/lowering/fusion.h"

#include <sstream>

namespace fer::jit::lowering {

namespace {

class FusionHintPass final : public LoweringPass {
public:
    const char* name() const override { return "fusion_hints"; }

    Status run(KernelModule* module, const DeviceCapabilities& caps) override {
        if (!module) return Status::invalid_argument("module is null");
        if (module->source.empty()) return Status::invalid_argument("module source is empty");

        if (module->source.find("fercuda:fuse") != std::string::npos) {
            module->notes.emplace_back("fusion hint detected");
            if (caps.tensor_cores) {
                module->notes.emplace_back("tensor-core capable target");
            }
        }
        return Status::ok_status();
    }
};

bool is_elementwise(IROpKind op) {
    return op == IROpKind::ElementwiseAdd ||
           op == IROpKind::ElementwiseMul ||
           op == IROpKind::ElementwiseRelu ||
           op == IROpKind::ElementwiseSigmoid ||
           op == IROpKind::ElementwiseTanh ||
           op == IROpKind::ElementwiseExp;
}

bool is_reduce(IROpKind op) {
    return op == IROpKind::ReduceSum ||
           op == IROpKind::ReduceMax;
}

const char* op_codegen_expr(IROpKind op) {
    switch (op) {
        case IROpKind::ElementwiseAdd:     return "(a + b)";
        case IROpKind::ElementwiseMul:     return "(a * b)";
        case IROpKind::ElementwiseRelu:    return "(a < 0.0f ? 0.0f : a)";
        case IROpKind::ElementwiseSigmoid: return "(1.0f / (1.0f + expf(-a)))";
        case IROpKind::ElementwiseTanh:    return "tanhf(a)";
        case IROpKind::ElementwiseExp:     return "expf(a)";
        default: return "a";
    }
}

class ElementwiseFusionRewritePass final : public LoweringPass {
public:
    const char* name() const override { return "fusion_rewrite_elementwise"; }

    Status run(KernelModule* module, const DeviceCapabilities&) override {
        if (!module) return Status::invalid_argument("module is null");
        if (module->source.empty()) return Status::invalid_argument("module source is empty");
        if (!module->fuse_elementwise_chain) return Status::ok_status();

        // Collect contiguous elementwise runs
        std::vector<size_t> chain_indices;
        for (size_t i = 0; i < module->ir.nodes.size(); ++i) {
            const IRNode& n = module->ir.nodes[i];
            if (n.removed || !is_elementwise(n.op)) continue;
            chain_indices.push_back(i);
        }

        if (chain_indices.size() < 2) return Status::ok_status();

        // Fuse the elementwise chain at IR level
        const size_t first = chain_indices.front();
        const size_t last = chain_indices.back();
        IRNode fused{};
        fused.id = static_cast<int>(module->ir.nodes.size());
        fused.op = IROpKind::FusedElementwiseChain;
        fused.inputs = module->ir.nodes[first].inputs;
        fused.output = module->ir.nodes[last].output;
        for (size_t idx : chain_indices) {
            module->ir.nodes[idx].removed = true;
        }
        module->ir.nodes.push_back(fused);
        module->notes.emplace_back("elementwise IR chain fused (" +
                                   std::to_string(chain_indices.size()) + " ops)");

        // Generate a fused kernel body if the chain is purely unary ops
        // (i.e. each op takes a single input value and produces one output).
        // This is the common case for activation chains.
        bool all_unary = true;
        for (size_t idx : chain_indices) {
            IROpKind op = module->ir.nodes[idx].op;
            if (op == IROpKind::ElementwiseAdd || op == IROpKind::ElementwiseMul) {
                all_unary = false;
                break;
            }
        }

        if (all_unary) {
            std::ostringstream body;
            body << "// Auto-generated fused elementwise kernel\n";
            body << "extern \"C\" __global__ void fer_fused_elementwise("
                    "const float* __restrict__ in, float* __restrict__ out, "
                    "unsigned int n) {\n";
            body << "  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n";
            body << "  if (i < n) {\n";
            body << "    float a = in[i];\n";
            for (size_t idx : chain_indices) {
                body << "    a = " << op_codegen_expr(module->ir.nodes[idx].op) << ";\n";
            }
            body << "    out[i] = a;\n";
            body << "  }\n}\n";
            module->fused_kernel_source = body.str();
            module->notes.emplace_back("fused unary elementwise kernel generated");
        }

        if (module->source.find("#define FERCUDA_FUSED_ELEMENTWISE 1") == std::string::npos) {
            module->source = std::string("#define FERCUDA_FUSED_ELEMENTWISE 1\n") + module->source;
            module->modified = true;
            module->notes.emplace_back("elementwise fusion rewrite materialized");
        }
        return Status::ok_status();
    }
};

class ReduceFusionPass final : public LoweringPass {
public:
    const char* name() const override { return "fusion_rewrite_reduce"; }

    Status run(KernelModule* module, const DeviceCapabilities&) override {
        if (!module) return Status::invalid_argument("module is null");
        if (module->source.empty()) return Status::invalid_argument("module source is empty");

        // Detect elementwise -> reduce patterns and fuse them.
        // e.g.  exp(x) -> sum  can become a single fused kernel.
        bool has_reduce = false;
        int reduce_idx = -1;
        for (size_t i = 0; i < module->ir.nodes.size(); ++i) {
            if (module->ir.nodes[i].removed) continue;
            if (is_reduce(module->ir.nodes[i].op)) {
                has_reduce = true;
                reduce_idx = static_cast<int>(i);
                break;
            }
        }
        if (!has_reduce || reduce_idx <= 0) return Status::ok_status();

        // Check if there's an elementwise op immediately before the reduce
        int pre_idx = -1;
        for (int i = reduce_idx - 1; i >= 0; --i) {
            if (module->ir.nodes[static_cast<size_t>(i)].removed) continue;
            if (is_elementwise(module->ir.nodes[static_cast<size_t>(i)].op)) {
                pre_idx = i;
            }
            break;
        }
        if (pre_idx < 0) return Status::ok_status();

        IRNode fused{};
        fused.id = static_cast<int>(module->ir.nodes.size());
        fused.op = IROpKind::FusedReduceChain;
        fused.inputs = module->ir.nodes[static_cast<size_t>(pre_idx)].inputs;
        fused.output = module->ir.nodes[static_cast<size_t>(reduce_idx)].output;
        module->ir.nodes[static_cast<size_t>(pre_idx)].removed = true;
        module->ir.nodes[static_cast<size_t>(reduce_idx)].removed = true;
        module->ir.nodes.push_back(fused);
        module->notes.emplace_back("elementwise+reduce fused");

        if (module->source.find("#define FERCUDA_FUSED_REDUCE 1") == std::string::npos) {
            module->source = std::string("#define FERCUDA_FUSED_REDUCE 1\n") + module->source;
            module->modified = true;
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

std::unique_ptr<LoweringPass> make_reduce_fusion_pass() {
    return std::make_unique<ReduceFusionPass>();
}

} // namespace fer::jit::lowering
