#include "fercuda/jit/lowering/pipeline.h"

#include "fercuda/jit/lowering/fusion.h"

#include <memory>
#include <vector>

namespace fer::jit::lowering {

Status LoweringPipeline::run(
    const fer_jit_source_t* source,
    const fer_jit_options_t*,
    KernelModule* out_module,
    DeviceCapabilities* out_caps) {
    if (!source) return Status::invalid_argument("source is null");
    if (!source->code || source->code_len == 0) return Status::invalid_argument("source code is empty");
    if (!out_module) return Status::invalid_argument("out_module is null");
    if (!out_caps) return Status::invalid_argument("out_caps is null");

    Status caps_st = detect_device_capabilities(out_caps);
    if (!caps_st.ok()) return caps_st;

    KernelModule module{};
    module.source.assign(source->code, source->code_len);
    module.entry = "";

    std::vector<std::unique_ptr<LoweringPass>> passes;
    passes.emplace_back(make_canonicalize_pass());
    passes.emplace_back(make_pragma_parse_pass());
    passes.emplace_back(make_capability_lowering_pass());
    passes.emplace_back(make_fusion_hint_pass());
    passes.emplace_back(make_elementwise_fusion_rewrite_pass());
    passes.emplace_back(make_reduce_fusion_pass());

    for (const auto& pass : passes) {
        Status st = pass->run(&module, *out_caps);
        if (!st.ok()) return st;
        module.notes.emplace_back(std::string("pass:") + pass->name());
    }

    *out_module = std::move(module);
    return Status::ok_status();
}

} // namespace fer::jit::lowering
