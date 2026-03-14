#include "fercuda/jit/lowering/fusion.h"

#include <string>

namespace fer::jit::lowering {

namespace {

bool parse_int_suffix(const std::string& s, const std::string& prefix, int* out) {
    if (!out) return false;
    if (s.rfind(prefix, 0) != 0) return false;
    const std::string n = s.substr(prefix.size());
    if (n.empty()) return false;
    try {
        *out = std::stoi(n);
        return true;
    } catch (...) {
        return false;
    }
}

class PragmaParsePass final : public LoweringPass {
public:
    const char* name() const override { return "pragma_parse"; }

    static IROpKind parse_op_kind(const std::string& op) {
        if (op == "elementwise_add") return IROpKind::ElementwiseAdd;
        if (op == "elementwise_mul") return IROpKind::ElementwiseMul;
        if (op == "elementwise_relu") return IROpKind::ElementwiseRelu;
        return IROpKind::Unknown;
    }

    static void append_ir_node(KernelModule* module, IROpKind op, const std::string& op_name) {
        IRValue in{};
        in.id = static_cast<int>(module->ir.values.size());
        in.name = "v" + std::to_string(in.id);
        module->ir.values.push_back(in);

        IRValue out{};
        out.id = static_cast<int>(module->ir.values.size());
        out.name = "v" + std::to_string(out.id);
        module->ir.values.push_back(out);

        IRNode node{};
        node.id = static_cast<int>(module->ir.nodes.size());
        node.op = op;
        node.inputs = {in.id};
        node.output = out.id;
        module->ir.nodes.push_back(node);
        module->notes.emplace_back("ir node parsed: " + op_name);
    }

    Status run(KernelModule* module, const DeviceCapabilities&) override {
        if (!module) return Status::invalid_argument("module is null");
        if (module->source.empty()) return Status::invalid_argument("module source is empty");

        size_t pos = 0;
        while (pos < module->source.size()) {
            size_t end = module->source.find('\n', pos);
            if (end == std::string::npos) end = module->source.size();
            const std::string line = module->source.substr(pos, end - pos);

            const size_t tag = line.find("fercuda:");
            if (tag != std::string::npos) {
                const std::string cmd = line.substr(tag + 8);
                if (cmd.find("require=tensor_cores") != std::string::npos) {
                    module->wants_tensor_cores = true;
                } else if (cmd.find("require=cooperative_groups") != std::string::npos) {
                    module->wants_cooperative_groups = true;
                } else if (cmd.find("fuse=elementwise_chain") != std::string::npos) {
                    module->fuse_elementwise_chain = true;
                } else if (cmd.find("ir_node=") == 0) {
                    const std::string op_name = cmd.substr(8);
                    const IROpKind op = parse_op_kind(op_name);
                    if (op != IROpKind::Unknown) append_ir_node(module, op, op_name);
                } else if (cmd.find("nvrtc_opt=") == 0) {
                    module->extra_nvrtc_opts.push_back(cmd.substr(10));
                } else {
                    int sm = 0;
                    if (parse_int_suffix(cmd, "require_sm>=", &sm)) {
                        module->min_sm = sm;
                    }
                }
            }

            pos = (end == module->source.size()) ? end : end + 1;
        }

        module->notes.emplace_back("pragmas parsed");
        return Status::ok_status();
    }
};

} // namespace

std::unique_ptr<LoweringPass> make_pragma_parse_pass() {
    return std::make_unique<PragmaParsePass>();
}

} // namespace fer::jit::lowering
