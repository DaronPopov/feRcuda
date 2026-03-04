#pragma once

#include "fercuda/compute/ops.cuh"
#include "fercuda/compute/registry.cuh"

namespace fer::algorithms {

struct LayerNormSpec {
    FTensor1D x;
    FTensor1D out;
    float eps = 1e-6f;
};

struct CudaLayerNormOp {
    static Status validate(const LayerNormSpec& spec) {
        if (!spec.x.valid() || !spec.out.valid()) {
            return Status::invalid_argument("null tensor pointer in layer_norm spec");
        }
        if (spec.x.shape[0] == 0) {
            return Status::invalid_argument("layer_norm input must be non-empty");
        }
        if (spec.x.shape[0] != spec.out.shape[0]) {
            return Status::invalid_argument("layer_norm output shape mismatch");
        }
        if (!(spec.eps > 0.0f)) {
            return Status::invalid_argument("layer_norm eps must be > 0");
        }
        return Status::ok_status();
    }

    static Status launch(const LayerNormSpec& spec, const OpContext& ctx) {
        Status st = validate(spec);
        if (!st.ok()) return st;
        (void)ctx.device;
        ops::layer_norm(spec.x, spec.out, spec.eps, ctx.stream.value);
        return Status::ok_status();
    }
};

inline void register_layer_norm(KernelRegistry& registry) {
    registry.register_impl<LayerNormSpec, CudaLayerNormOp>(
        OpTag::LAYER_NORM, "cuda.layer_norm.basic");
}

} // namespace fer::algorithms
