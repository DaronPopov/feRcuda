#pragma once

#include "fercuda/algorithms/layernorm_attachment.cuh"
#include "fercuda/compute/ops.cuh"
#include "fercuda/compute/registry.cuh"

namespace fer::algorithms {

struct MatmulSpec {
    FTensor2D a;
    FTensor2D b;
    FTensor2D out;
};

struct CudaMatmulOp {
    static Status validate(const MatmulSpec& spec) {
        if (!spec.a.valid() || !spec.b.valid() || !spec.out.valid()) {
            return Status::invalid_argument("null tensor pointer in matmul spec");
        }
        if (spec.a.shape[1] != spec.b.shape[0]) {
            return Status::invalid_argument("matmul inner dimensions mismatch");
        }
        if (spec.out.shape[0] != spec.a.shape[0] || spec.out.shape[1] != spec.b.shape[1]) {
            return Status::invalid_argument("matmul output shape mismatch");
        }
        return Status::ok_status();
    }

    static Status launch(const MatmulSpec& spec, const OpContext& ctx) {
        Status st = validate(spec);
        if (!st.ok()) return st;
        (void)ctx.device;
        ops::matmul(spec.a, spec.b, spec.out, ctx.stream.value);
        return Status::ok_status();
    }
};

inline void register_matmul(KernelRegistry& registry) {
    registry.register_impl<MatmulSpec, CudaMatmulOp>(OpTag::MATMUL, "cuda.matmul.basic");
}

inline KernelRegistry make_default_registry() {
    KernelRegistry registry;
    register_matmul(registry);
    register_layer_norm(registry);
    return registry;
}

} // namespace fer::algorithms
