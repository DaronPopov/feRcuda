#include "fercuda/runtime/exec_planner.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fer::runtime {

namespace {

size_t numel_from_desc(const BufferDesc& desc) {
    size_t n = 1;
    for (uint8_t i = 0; i < desc.rank; i++) n *= desc.dims[i];
    return n;
}

size_t bytes_per_elem(BufferDType dt) {
    switch (dt) {
        case BufferDType::F32: return 4;
        case BufferDType::F16: return 2;
        case BufferDType::BF16: return 2;
        case BufferDType::I8: return 1;
        case BufferDType::U8: return 1;
        case BufferDType::I16: return 2;
        case BufferDType::U16: return 2;
        case BufferDType::I32: return 4;
        case BufferDType::U32: return 4;
        case BufferDType::I64: return 8;
        case BufferDType::U64: return 8;
        case BufferDType::F64: return 8;
        default: return 0;
    }
}

template <typename SrcT>
__global__ void k_cast_to_f32(const SrcT* __restrict__ in, float* __restrict__ out, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<float>(in[i]);
}

template <>
__global__ void k_cast_to_f32<__half>(const __half* __restrict__ in, float* __restrict__ out, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

template <>
__global__ void k_cast_to_f32<__nv_bfloat16>(const __nv_bfloat16* __restrict__ in, float* __restrict__ out, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

template <>
__global__ void k_cast_to_f32<int8_t>(const int8_t* __restrict__ in, float* __restrict__ out, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<float>(in[i]);
}

template <>
__global__ void k_cast_to_f32<uint8_t>(const uint8_t* __restrict__ in, float* __restrict__ out, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<float>(in[i]);
}

template <typename DstT>
__global__ void k_cast_from_f32(const float* __restrict__ in, DstT* __restrict__ out, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<DstT>(in[i]);
}

template <>
__global__ void k_cast_from_f32<__half>(const float* __restrict__ in, __half* __restrict__ out, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(in[i]);
}

template <>
__global__ void k_cast_from_f32<__nv_bfloat16>(const float* __restrict__ in, __nv_bfloat16* __restrict__ out, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

template <>
__global__ void k_cast_from_f32<int8_t>(const float* __restrict__ in, int8_t* __restrict__ out, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        out[i] = static_cast<int8_t>(nearbyintf(v));
    }
}

template <>
__global__ void k_cast_from_f32<uint8_t>(const float* __restrict__ in, uint8_t* __restrict__ out, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        if (v > 255.0f) v = 255.0f;
        if (v < 0.0f) v = 0.0f;
        out[i] = static_cast<uint8_t>(nearbyintf(v));
    }
}

Status cast_to_f32(BufferDType src_dt, const void* src, float* dst, size_t n, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    switch (src_dt) {
        case BufferDType::F32:
            if (cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                return Status::internal_error("cast_to_f32 memcpy failed");
            }
            return Status::ok_status();
        case BufferDType::F16:
            k_cast_to_f32<__half><<<grid, kBlock, 0, stream>>>(static_cast<const __half*>(src), dst, n);
            break;
        case BufferDType::BF16:
            k_cast_to_f32<__nv_bfloat16><<<grid, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(src), dst, n);
            break;
        case BufferDType::I8:
            k_cast_to_f32<int8_t><<<grid, kBlock, 0, stream>>>(static_cast<const int8_t*>(src), dst, n);
            break;
        case BufferDType::U8:
            k_cast_to_f32<uint8_t><<<grid, kBlock, 0, stream>>>(static_cast<const uint8_t*>(src), dst, n);
            break;
        default:
            return Status::invalid_argument("cast_to_f32 unsupported source dtype");
    }
    if (cudaGetLastError() != cudaSuccess) return Status::internal_error("cast_to_f32 kernel launch failed");
    return Status::ok_status();
}

Status cast_from_f32(BufferDType dst_dt, const float* src, void* dst, size_t n, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    switch (dst_dt) {
        case BufferDType::F32:
            if (cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                return Status::internal_error("cast_from_f32 memcpy failed");
            }
            return Status::ok_status();
        case BufferDType::F16:
            k_cast_from_f32<__half><<<grid, kBlock, 0, stream>>>(src, static_cast<__half*>(dst), n);
            break;
        case BufferDType::BF16:
            k_cast_from_f32<__nv_bfloat16><<<grid, kBlock, 0, stream>>>(src, static_cast<__nv_bfloat16*>(dst), n);
            break;
        case BufferDType::I8:
            k_cast_from_f32<int8_t><<<grid, kBlock, 0, stream>>>(src, static_cast<int8_t*>(dst), n);
            break;
        case BufferDType::U8:
            k_cast_from_f32<uint8_t><<<grid, kBlock, 0, stream>>>(src, static_cast<uint8_t*>(dst), n);
            break;
        default:
            return Status::invalid_argument("cast_from_f32 unsupported destination dtype");
    }
    if (cudaGetLastError() != cudaSuccess) return Status::internal_error("cast_from_f32 kernel launch failed");
    return Status::ok_status();
}

} // namespace

ExecPlanner::ExecPlanner(RegimeManager* regime_mgr, KernelRegistry* registry, const OpContext* ctx)
    : regime_mgr_(regime_mgr), registry_(registry), ctx_(ctx) {}

size_t ExecPlanner::bytes_for_desc(const BufferDesc& desc) {
    const size_t bpe = bytes_per_elem(desc.dtype);
    if (bpe == 0) return 0;
    return numel_from_desc(desc) * bpe;
}

FTensor1D ExecPlanner::tensor1d_from(void* ptr, const BufferDesc& desc) {
    return FTensor1D(static_cast<F32*>(ptr), Shape<1>(desc.dims[0]));
}

FTensor2D ExecPlanner::tensor2d_from(void* ptr, const BufferDesc& desc) {
    return FTensor2D(static_cast<F32*>(ptr), Shape<2>(desc.dims[0], desc.dims[1]));
}

Status ExecPlanner::free_temps(const std::vector<TempAlloc>& temps) {
    if (!regime_mgr_) return Status::internal_error("exec planner regime manager is null");
    for (const auto& t : temps) {
        Status st = regime_mgr_->free_bytes(t.ptr, t.regime_id);
        if (!st.ok()) return st;
    }
    return Status::ok_status();
}

Status ExecPlanner::launch_matmul(
    const ExecBufferView& a,
    const ExecBufferView& b,
    const ExecBufferView& out,
    uint32_t memory_regime_override,
    std::vector<TempAlloc>* out_temps) {
    if (!regime_mgr_ || !registry_ || !ctx_) return Status::internal_error("exec planner is not initialized");
    if (!out_temps) return Status::invalid_argument("out_temps is null");

    ResolvedRegime target{};
    Status rs = regime_mgr_->resolve(memory_regime_override, &target);
    if (!rs.ok()) return rs;

    std::vector<TempAlloc> temps;
    void* a_ptr = a.ptr;
    void* b_ptr = b.ptr;
    void* out_ptr = out.ptr;
    BufferDesc a_desc = a.desc;
    BufferDesc b_desc = b.desc;
    BufferDesc out_desc = out.desc;

    const bool needs_float_cast = a.desc.dtype != BufferDType::F32 ||
                                  b.desc.dtype != BufferDType::F32 ||
                                  out.desc.dtype != BufferDType::F32;

    if (needs_float_cast) {
        a_desc.dtype = BufferDType::F32;
        b_desc.dtype = BufferDType::F32;
        out_desc.dtype = BufferDType::F32;

        Status st = regime_mgr_->alloc_bytes(bytes_for_desc(a_desc), Tier::MUTABLE, a.desc.tag, target, &a_ptr);
        if (!st.ok()) return st;
        temps.push_back({a_ptr, target.raw});

        st = cast_to_f32(a.desc.dtype, a.ptr, static_cast<float*>(a_ptr), numel_from_desc(a.desc), ctx_->stream.value);
        if (!st.ok()) {
            free_temps(temps);
            return st;
        }

        st = regime_mgr_->alloc_bytes(bytes_for_desc(b_desc), Tier::MUTABLE, b.desc.tag, target, &b_ptr);
        if (!st.ok()) {
            free_temps(temps);
            return st;
        }
        temps.push_back({b_ptr, target.raw});

        st = cast_to_f32(b.desc.dtype, b.ptr, static_cast<float*>(b_ptr), numel_from_desc(b.desc), ctx_->stream.value);
        if (!st.ok()) {
            free_temps(temps);
            return st;
        }

        st = regime_mgr_->alloc_bytes(bytes_for_desc(out_desc), Tier::MUTABLE, out.desc.tag, target, &out_ptr);
        if (!st.ok()) {
            free_temps(temps);
            return st;
        }
        temps.push_back({out_ptr, target.raw});
    }

    if (!needs_float_cast && !same_regime(a.regime_id, target.raw)) {
        Status st = regime_mgr_->alloc_bytes(bytes_for_desc(a.desc), Tier::MUTABLE, a.desc.tag, target, &a_ptr);
        if (!st.ok()) return st;
        temps.push_back({a_ptr, target.raw});
        if (cudaMemcpyAsync(a_ptr, a.ptr, bytes_for_desc(a.desc), cudaMemcpyDeviceToDevice, ctx_->stream.value) != cudaSuccess) {
            free_temps(temps);
            return Status::internal_error("matmul A staging copy failed");
        }
    }
    if (!needs_float_cast && !same_regime(b.regime_id, target.raw)) {
        Status st = regime_mgr_->alloc_bytes(bytes_for_desc(b.desc), Tier::MUTABLE, b.desc.tag, target, &b_ptr);
        if (!st.ok()) {
            free_temps(temps);
            return st;
        }
        temps.push_back({b_ptr, target.raw});
        if (cudaMemcpyAsync(b_ptr, b.ptr, bytes_for_desc(b.desc), cudaMemcpyDeviceToDevice, ctx_->stream.value) != cudaSuccess) {
            free_temps(temps);
            return Status::internal_error("matmul B staging copy failed");
        }
    }
    if (!needs_float_cast && !same_regime(out.regime_id, target.raw)) {
        Status st = regime_mgr_->alloc_bytes(bytes_for_desc(out.desc), Tier::MUTABLE, out.desc.tag, target, &out_ptr);
        if (!st.ok()) {
            free_temps(temps);
            return st;
        }
        temps.push_back({out_ptr, target.raw});
    }

    algorithms::MatmulSpec spec{tensor2d_from(a_ptr, a_desc), tensor2d_from(b_ptr, b_desc), tensor2d_from(out_ptr, out_desc)};
    Status launch = registry_->launch(OpTag::MATMUL, spec, *ctx_);
    if (!launch.ok()) {
        free_temps(temps);
        return launch;
    }

    if (needs_float_cast) {
        Status st = cast_from_f32(out.desc.dtype, static_cast<const float*>(out_ptr), out.ptr, numel_from_desc(out.desc), ctx_->stream.value);
        if (!st.ok()) {
            free_temps(temps);
            return st;
        }
    } else if (!same_regime(out.regime_id, target.raw)) {
        if (cudaMemcpyAsync(out.ptr, out_ptr, bytes_for_desc(out.desc), cudaMemcpyDeviceToDevice, ctx_->stream.value) != cudaSuccess) {
            free_temps(temps);
            return Status::internal_error("matmul output copy-back failed");
        }
    }

    *out_temps = temps;
    return Status::ok_status();
}

Status ExecPlanner::launch_layer_norm(
    const ExecBufferView& x,
    const ExecBufferView& out,
    float eps,
    uint32_t memory_regime_override,
    std::vector<TempAlloc>* out_temps) {
    if (!regime_mgr_ || !registry_ || !ctx_) return Status::internal_error("exec planner is not initialized");
    if (!out_temps) return Status::invalid_argument("out_temps is null");

    ResolvedRegime target{};
    Status rs = regime_mgr_->resolve(memory_regime_override, &target);
    if (!rs.ok()) return rs;

    std::vector<TempAlloc> temps;
    void* x_ptr = x.ptr;
    void* out_ptr = out.ptr;
    BufferDesc x_desc = x.desc;
    BufferDesc out_desc = out.desc;

    const bool needs_float_cast = x.desc.dtype != BufferDType::F32 || out.desc.dtype != BufferDType::F32;

    if (needs_float_cast) {
        x_desc.dtype = BufferDType::F32;
        out_desc.dtype = BufferDType::F32;

        Status st = regime_mgr_->alloc_bytes(bytes_for_desc(x_desc), Tier::MUTABLE, x.desc.tag, target, &x_ptr);
        if (!st.ok()) return st;
        temps.push_back({x_ptr, target.raw});

        st = cast_to_f32(x.desc.dtype, x.ptr, static_cast<float*>(x_ptr), numel_from_desc(x.desc), ctx_->stream.value);
        if (!st.ok()) {
            free_temps(temps);
            return st;
        }

        st = regime_mgr_->alloc_bytes(bytes_for_desc(out_desc), Tier::MUTABLE, out.desc.tag, target, &out_ptr);
        if (!st.ok()) {
            free_temps(temps);
            return st;
        }
        temps.push_back({out_ptr, target.raw});
    }

    if (!needs_float_cast && !same_regime(x.regime_id, target.raw)) {
        Status st = regime_mgr_->alloc_bytes(bytes_for_desc(x.desc), Tier::MUTABLE, x.desc.tag, target, &x_ptr);
        if (!st.ok()) return st;
        temps.push_back({x_ptr, target.raw});
        if (cudaMemcpyAsync(x_ptr, x.ptr, bytes_for_desc(x.desc), cudaMemcpyDeviceToDevice, ctx_->stream.value) != cudaSuccess) {
            free_temps(temps);
            return Status::internal_error("layer_norm X staging copy failed");
        }
    }
    if (!needs_float_cast && !same_regime(out.regime_id, target.raw)) {
        Status st = regime_mgr_->alloc_bytes(bytes_for_desc(out.desc), Tier::MUTABLE, out.desc.tag, target, &out_ptr);
        if (!st.ok()) {
            free_temps(temps);
            return st;
        }
        temps.push_back({out_ptr, target.raw});
    }

    algorithms::LayerNormSpec spec{tensor1d_from(x_ptr, x_desc), tensor1d_from(out_ptr, out_desc), eps};
    Status launch = registry_->launch(OpTag::LAYER_NORM, spec, *ctx_);
    if (!launch.ok()) {
        free_temps(temps);
        return launch;
    }

    if (needs_float_cast) {
        Status st = cast_from_f32(out.desc.dtype, static_cast<const float*>(out_ptr), out.ptr, numel_from_desc(out.desc), ctx_->stream.value);
        if (!st.ok()) {
            free_temps(temps);
            return st;
        }
    } else if (!same_regime(out.regime_id, target.raw)) {
        if (cudaMemcpyAsync(out.ptr, out_ptr, bytes_for_desc(out.desc), cudaMemcpyDeviceToDevice, ctx_->stream.value) != cudaSuccess) {
            free_temps(temps);
            return Status::internal_error("layer_norm output copy-back failed");
        }
    }

    *out_temps = temps;
    return Status::ok_status();
}

} // namespace fer::runtime
