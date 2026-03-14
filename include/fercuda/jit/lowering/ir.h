#pragma once

#include <string>
#include <vector>

namespace fer::jit::lowering {

enum class IROpKind {
    Unknown = 0,
    ElementwiseAdd,
    ElementwiseMul,
    ElementwiseRelu,
    ElementwiseSigmoid,
    ElementwiseTanh,
    ElementwiseExp,
    ReduceSum,
    ReduceMax,
    Softmax,
    Conv2d,
    FusedElementwiseChain,
    FusedReduceChain,
};

struct IRValue {
    int id = -1;
    std::string name;
};

struct IRNode {
    int id = -1;
    IROpKind op = IROpKind::Unknown;
    std::vector<int> inputs;
    int output = -1;
    bool removed = false;
};

struct KernelIR {
    std::vector<IRValue> values;
    std::vector<IRNode> nodes;
};

struct KernelModule {
    std::string source;
    std::string entry;
    bool modified = false;
    bool wants_tensor_cores = false;
    bool wants_cooperative_groups = false;
    int min_sm = 0;
    bool fuse_elementwise_chain = false;
    KernelIR ir;
    std::vector<std::string> extra_nvrtc_opts;
    std::vector<std::string> notes;
    std::string fused_kernel_source;
};

} // namespace fer::jit::lowering
