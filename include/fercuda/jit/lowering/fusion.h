#pragma once

#include "fercuda/jit/lowering/pass.h"

#include <memory>

namespace fer::jit::lowering {

std::unique_ptr<LoweringPass> make_fusion_hint_pass();
std::unique_ptr<LoweringPass> make_canonicalize_pass();
std::unique_ptr<LoweringPass> make_pragma_parse_pass();
std::unique_ptr<LoweringPass> make_capability_lowering_pass();
std::unique_ptr<LoweringPass> make_elementwise_fusion_rewrite_pass();
std::unique_ptr<LoweringPass> make_reduce_fusion_pass();

} // namespace fer::jit::lowering
