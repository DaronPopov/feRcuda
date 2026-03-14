// ============================================================================
// Arange / Linspace kernels
// ============================================================================

__global__ void k_arange_f32(float* __restrict__ out, size_t n, float start, float step) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = start + (float)idx * step;
    }
}

extern "C" void ptx_tensor_arange_f32(float* out, size_t n, float start, float step, cudaStream_t stream) {
    k_arange_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(out, n, start, step);
}

// ============================================================================
// Logical ops (on U8 boolean tensors)
// ============================================================================

__global__ void k_logical_and_u8(const unsigned char* __restrict__ a, const unsigned char* __restrict__ b,
                                  unsigned char* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] && b[idx]) ? 1 : 0; }
}

__global__ void k_logical_or_u8(const unsigned char* __restrict__ a, const unsigned char* __restrict__ b,
                                 unsigned char* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] || b[idx]) ? 1 : 0; }
}

__global__ void k_logical_not_u8(const unsigned char* __restrict__ in, unsigned char* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = in[idx] ? 0 : 1; }
}

__global__ void k_logical_xor_u8(const unsigned char* __restrict__ a, const unsigned char* __restrict__ b,
                                  unsigned char* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] != b[idx]) ? 1 : 0; }
}

extern "C" void ptx_tensor_logical_and_u8(unsigned char* a, unsigned char* b, unsigned char* out, size_t n, cudaStream_t stream) {
    k_logical_and_u8<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}
extern "C" void ptx_tensor_logical_or_u8(unsigned char* a, unsigned char* b, unsigned char* out, size_t n, cudaStream_t stream) {
    k_logical_or_u8<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}
extern "C" void ptx_tensor_logical_not_u8(unsigned char* in, unsigned char* out, size_t n, cudaStream_t stream) {
    k_logical_not_u8<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}
extern "C" void ptx_tensor_logical_xor_u8(unsigned char* a, unsigned char* b, unsigned char* out, size_t n, cudaStream_t stream) {
    k_logical_xor_u8<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

// ============================================================================
// Broadcast binary op kernel
// ============================================================================

// Computes flat index from nd-index using strides
__device__ size_t broadcast_offset(size_t flat_idx, const size_t* out_shape, const size_t* in_strides, int ndim) {
    size_t offset = 0;
    for (int d = ndim - 1; d >= 0; d--) {
        size_t coord = flat_idx % out_shape[d];
        flat_idx /= out_shape[d];
        offset += coord * in_strides[d];
    }
    return offset;
}

__global__ void k_broadcast_binary_f32(const float* __restrict__ a, const float* __restrict__ b,
                                        float* __restrict__ out, size_t n,
                                        const size_t* out_shape, const size_t* a_strides,
                                        const size_t* b_strides, int ndim, int op) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = a[broadcast_offset(idx, out_shape, a_strides, ndim)];
        float vb = b[broadcast_offset(idx, out_shape, b_strides, ndim)];
        float result;
        switch (op) {
            case 0: result = va + vb; break;
            case 1: result = va - vb; break;
            case 2: result = va * vb; break;
            case 3: result = va / vb; break;
            case 4: result = fmaxf(va, vb); break;
            case 5: result = fminf(va, vb); break;
            case 6: result = fmodf(va, vb); break;
            default: result = 0.0f;
        }
        out[idx] = result;
    }
}

extern "C" void ptx_tensor_broadcast_binary_f32(
    float* a, float* b, float* out, size_t n,
    size_t* out_shape, size_t* a_strides, size_t* b_strides,
    int ndim, int op, cudaStream_t stream
) {
    k_broadcast_binary_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(
        a, b, out, n, out_shape, a_strides, b_strides, ndim, op);
}

// ============================================================================
// Pad kernel (constant padding)
// ============================================================================

// 2D padding: pad [N, C, H, W] with constant value
__global__ void k_pad2d_f32(const float* __restrict__ in, float* __restrict__ out,
                             int N, int C, int H, int W,
                             int pad_top, int pad_bottom, int pad_left, int pad_right,
                             float pad_value) {
    int oH = H + pad_top + pad_bottom;
    int oW = W + pad_left + pad_right;
    size_t total = (size_t)N * (size_t)C * (size_t)oH * (size_t)oW;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int ow = idx % oW;
        int oh = (idx / oW) % oH;
        int c  = (idx / (oW * oH)) % C;
        int n  = idx / (oW * oH * C);

        int ih = oh - pad_top;
        int iw = ow - pad_left;

        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            out[idx] = in[((n * C + c) * H + ih) * W + iw];
        } else {
            out[idx] = pad_value;
        }
    }
}

extern "C" void ptx_tensor_pad2d_f32(
    float* in, float* out,
    int N, int C, int H, int W,
    int pad_top, int pad_bottom, int pad_left, int pad_right,
    float pad_value, cudaStream_t stream
) {
    int oH = H + pad_top + pad_bottom;
    int oW = W + pad_left + pad_right;
    size_t total = (size_t)N * C * oH * oW;
    k_pad2d_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(
        in, out, N, C, H, W, pad_top, pad_bottom, pad_left, pad_right, pad_value);
}

// ============================================================================
// Repeat / Tile kernel
// ============================================================================

__global__ void k_repeat_f32(const float* __restrict__ in, float* __restrict__ out,
                              size_t out_n, const size_t* in_shape, const size_t* out_shape,
                              int ndim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_n) {
        // Map output index to input index by taking modulo of each dim
        size_t in_idx = 0;
        size_t remaining = idx;
        size_t in_stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            size_t coord = remaining % out_shape[d];
            remaining /= out_shape[d];
            size_t in_coord = coord % in_shape[d];
            in_idx += in_coord * in_stride;
            in_stride *= in_shape[d];
        }
        out[idx] = in[in_idx];
    }
}

extern "C" void ptx_tensor_repeat_f32(
    float* in, float* out, size_t out_n,
    size_t* in_shape, size_t* out_shape, int ndim, cudaStream_t stream
) {
    k_repeat_f32<<<PTX_GRID_SIZE(out_n), PTX_BLOCK_SIZE, 0, stream>>>(
        in, out, out_n, in_shape, out_shape, ndim);
}

// ============================================================================
// Masked fill kernel
// ============================================================================

__global__ void k_masked_fill_f32(float* __restrict__ data, const unsigned char* __restrict__ mask,
                                   float value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (mask[idx]) {
            data[idx] = value;
        }
    }
}

extern "C" void ptx_tensor_masked_fill_f32(float* data, unsigned char* mask, float value, size_t n, cudaStream_t stream) {
    k_masked_fill_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(data, mask, value, n);
}

// ============================================================================
// Generic Dispatch Implementation
// ============================================================================

extern "C" cudaError_t ptx_tensor_dispatch(const PTXTensorOp* op) {
    if (!op) return cudaErrorInvalidValue;

    cudaStream_t stream = op->stream;

    switch (op->opcode) {
        // Binary ops
        case PTX_OP_ADD:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_add_f32((float*)op->input_a, (float*)op->input_b, (float*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_add_f64((double*)op->input_a, (double*)op->input_b, (double*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F16) {
                ptx_tensor_add_f16((__half*)op->input_a, (__half*)op->input_b, (__half*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_SUB:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_sub_f32((float*)op->input_a, (float*)op->input_b, (float*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_sub_f64((double*)op->input_a, (double*)op->input_b, (double*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F16) {
                ptx_tensor_sub_f16((__half*)op->input_a, (__half*)op->input_b, (__half*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_MUL:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_mul_f32((float*)op->input_a, (float*)op->input_b, (float*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_mul_f64((double*)op->input_a, (double*)op->input_b, (double*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F16) {
                ptx_tensor_mul_f16((__half*)op->input_a, (__half*)op->input_b, (__half*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_DIV:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_div_f32((float*)op->input_a, (float*)op->input_b, (float*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_div_f64((double*)op->input_a, (double*)op->input_b, (double*)op->output, op->elem_count, stream);
            } else if (op->dtype == PTX_DTYPE_F16) {
                ptx_tensor_div_f16((__half*)op->input_a, (__half*)op->input_b, (__half*)op->output, op->elem_count, stream);
            }
            break;

        // Unary ops
        case PTX_OP_NEG:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_neg_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_neg_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_EXP:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_exp_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_exp_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_LOG:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_log_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_log_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SQRT:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_sqrt_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_sqrt_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_TANH:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_tanh_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_tanh_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SIN:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_sin_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_COS:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_cos_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_ABS:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_abs_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SQR:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_sqr_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_RECIP:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_recip_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_CEIL:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_ceil_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_FLOOR:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_floor_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_ROUND:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_round_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SIGN:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_sign_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_LOG2:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_log2_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_LOG10:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_log10_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_TAN:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_tan_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SINH:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_sinh_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_COSH:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_cosh_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_ERF:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_erf_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_MOD:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_mod_f32((float*)op->input_a, (float*)op->input_b, (float*)op->output, op->elem_count, stream);
            break;

        // Activations
        case PTX_OP_RELU:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_relu_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_relu_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F16) ptx_tensor_relu_f16((__half*)op->input_a, (__half*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_GELU:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_gelu_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_gelu_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F16) ptx_tensor_gelu_f16((__half*)op->input_a, (__half*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SIGMOID:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_sigmoid_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F64) ptx_tensor_sigmoid_f64((double*)op->input_a, (double*)op->output, op->elem_count, stream);
            else if (op->dtype == PTX_DTYPE_F16) ptx_tensor_sigmoid_f16((__half*)op->input_a, (__half*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SILU:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_silu_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_ELU:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_elu_f32((float*)op->input_a, (float*)op->output, op->elem_count, op->scalar_a, stream);
            break;

        case PTX_OP_LEAKY_RELU:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_leaky_relu_f32((float*)op->input_a, (float*)op->output, op->elem_count, op->scalar_a, stream);
            break;

        case PTX_OP_SELU:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_selu_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_SOFTPLUS:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_softplus_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_MISH:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_mish_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_GELU_TANH:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_gelu_tanh_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_HARDSWISH:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_hardswish_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        case PTX_OP_HARDSIGMOID:
            if (op->dtype == PTX_DTYPE_F32) ptx_tensor_hardsigmoid_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            break;

        // Reductions
        case PTX_OP_REDUCE_SUM:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_reduce_sum_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_reduce_sum_f64((double*)op->input_a, (double*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        case PTX_OP_REDUCE_MEAN:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_reduce_mean_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        case PTX_OP_REDUCE_MAX:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_reduce_max_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_reduce_max_f64((double*)op->input_a, (double*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        case PTX_OP_REDUCE_MIN:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_reduce_min_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_reduce_min_f64((double*)op->input_a, (double*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        case PTX_OP_REDUCE_PROD:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_reduce_prod_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        case PTX_OP_REDUCE_ARGMAX:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_reduce_argmax_f32((float*)op->input_a, (int32_t*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        case PTX_OP_REDUCE_ARGMIN:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_reduce_argmin_f32((float*)op->input_a, (int32_t*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        // Softmax
        case PTX_OP_SOFTMAX:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_softmax_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_softmax_f64((double*)op->input_a, (double*)op->output, op->outer_size, op->reduce_size, stream);
            } else if (op->dtype == PTX_DTYPE_F16) {
                ptx_tensor_softmax_f16((__half*)op->input_a, (__half*)op->output, op->outer_size, op->reduce_size, stream);
            }
            break;

        case PTX_OP_LOG_SOFTMAX:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_log_softmax_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, stream);
            }
            break;

        // Affine/Transform
        case PTX_OP_AFFINE:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_affine_f32((float*)op->input_a, (float*)op->output, op->elem_count, op->scalar_a, op->scalar_b, stream);
            } else if (op->dtype == PTX_DTYPE_F64) {
                ptx_tensor_affine_f64((double*)op->input_a, (double*)op->output, op->elem_count, (double)op->scalar_a, (double)op->scalar_b, stream);
            }
            break;

        case PTX_OP_CLAMP:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_clamp_f32((float*)op->input_a, (float*)op->output, op->elem_count, op->scalar_a, op->scalar_b, stream);
            }
            break;

        case PTX_OP_POW:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_powf_f32((float*)op->input_a, (float*)op->output, op->elem_count, op->scalar_a, stream);
            }
            break;

        // Comparison
        case PTX_OP_CMP_EQ:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_cmp_eq_f32((float*)op->input_a, (float*)op->input_b, (uint8_t*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_CMP_LT:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_cmp_lt_f32((float*)op->input_a, (float*)op->input_b, (uint8_t*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_CMP_LE:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_cmp_le_f32((float*)op->input_a, (float*)op->input_b, (uint8_t*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_CMP_GT:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_cmp_gt_f32((float*)op->input_a, (float*)op->input_b, (uint8_t*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_CMP_GE:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_cmp_ge_f32((float*)op->input_a, (float*)op->input_b, (uint8_t*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_CMP_NE:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_cmp_ne_f32((float*)op->input_a, (float*)op->input_b, (uint8_t*)op->output, op->elem_count, stream);
            }
            break;

        // Where
        case PTX_OP_WHERE:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_where_f32((uint8_t*)op->input_a, (float*)op->input_b, (float*)op->output, (float*)op->output, op->elem_count, stream);
            }
            break;

        // Copy
        case PTX_OP_COPY:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_copy_f32((float*)op->input_a, (float*)op->output, op->elem_count, stream);
            }
            break;

        case PTX_OP_FILL:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_fill_f32((float*)op->output, op->elem_count, op->scalar_a, stream);
            }
            break;

        case PTX_OP_GATHER:
            // For gather dispatch: outer_size=outer, reduce_size=input_dim_size,
            // inner_size=idx_dim_size, elem_count=inner
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_gather_f32((float*)op->input_a, (int32_t*)op->input_b, (float*)op->output,
                                      op->outer_size, op->reduce_size, op->inner_size, op->elem_count, stream);
            }
            break;

        case PTX_OP_CUMSUM:
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_cumsum_f32((float*)op->input_a, (float*)op->output, op->outer_size, op->reduce_size, op->inner_size, stream);
            }
            break;

        case PTX_OP_TOPK:
            // For topk dispatch: outer_size=outer, reduce_size=dim_size, inner_size=inner,
            // elem_count=k, scalar_a=largest (1.0 or 0.0), input_b=indices output (or NULL)
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_topk_f32((float*)op->input_a, (float*)op->output, (int32_t*)op->input_b,
                                    op->outer_size, op->reduce_size, op->inner_size,
                                    op->elem_count, (int)(op->scalar_a > 0.5f), stream);
            }
            break;

        case PTX_OP_INDEX_SELECT:
            // outer_size=left, reduce_size=src_dim, inner_size=ids_dim, elem_count=right
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_index_select_f32((float*)op->input_a, (int32_t*)op->input_b, (float*)op->output,
                                            op->outer_size, op->reduce_size, op->inner_size, op->elem_count, stream);
            }
            break;

        case PTX_OP_SCATTER_ADD:
            // outer_size=left, reduce_size=src_dim, inner_size=dst_dim, elem_count=right
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_scatter_add_f32((int32_t*)op->input_b, (float*)op->input_a, (float*)op->output,
                                           op->outer_size, op->reduce_size, op->inner_size, op->elem_count, stream);
            }
            break;

        case PTX_OP_ARGSORT:
            // outer_size=nrows, reduce_size=ncols, scalar_a=ascending (1.0 or 0.0)
            if (op->dtype == PTX_DTYPE_F32) {
                ptx_tensor_argsort_f32((float*)op->input_a, (unsigned int*)op->output,
                                       op->outer_size, op->reduce_size,
                                       (int)(op->scalar_a > 0.5f), stream);
            }
            break;

        default:
            return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

extern "C" cudaError_t ptx_tensor_dispatch_sync(const PTXTensorOp* op) {
    cudaError_t err = ptx_tensor_dispatch(op);
    if (err != cudaSuccess) return err;

    if (op->stream) {
        return cudaStreamSynchronize(op->stream);
    } else {
        return cudaDeviceSynchronize();
    }
}

// ============================================================================
// CUDA Graph Integration
// ============================================================================

extern "C" cudaError_t ptx_tensor_graph_begin_capture(cudaStream_t stream) {
    return cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
}

extern "C" cudaError_t ptx_tensor_graph_end_capture(cudaStream_t stream, cudaGraph_t* graph_out) {
    return cudaStreamEndCapture(stream, graph_out);
}

extern "C" cudaError_t ptx_tensor_graph_instantiate(cudaGraph_t graph, cudaGraphExec_t* exec_out) {
    return cudaGraphInstantiate(exec_out, graph, NULL, NULL, 0);
}

extern "C" cudaError_t ptx_tensor_graph_launch(cudaGraphExec_t exec, cudaStream_t stream) {
    return cudaGraphLaunch(exec, stream);
}

// ============================================================================
// Strided Copy kernel (for making non-contiguous tensors contiguous)
// ============================================================================

__global__ void k_strided_copy_f32(const float* __restrict__ in, float* __restrict__ out,
                                    const size_t* __restrict__ shape,
                                    const size_t* __restrict__ in_strides,
                                    int ndim, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Decompose linear output index into nd-index, then compute source offset
    size_t src_offset = 0;
    size_t remaining = idx;
    for (int d = ndim - 1; d >= 0; d--) {
        size_t coord = remaining % shape[d];
        remaining /= shape[d];
        src_offset += coord * in_strides[d];
    }
    out[idx] = in[src_offset];
}

extern "C" void ptx_tensor_strided_copy_f32(
    const float* in, float* out,
    const size_t* shape, const size_t* in_strides,
    int ndim, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    k_strided_copy_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(
        in, out, shape, in_strides, ndim, n);
}

__global__ void k_strided_copy_f64(const double* __restrict__ in, double* __restrict__ out,
                                    const size_t* __restrict__ shape,
                                    const size_t* __restrict__ in_strides,
                                    int ndim, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    size_t src_offset = 0;
    size_t remaining = idx;
    for (int d = ndim - 1; d >= 0; d--) {
        size_t coord = remaining % shape[d];
        remaining /= shape[d];
        src_offset += coord * in_strides[d];
    }
    out[idx] = in[src_offset];
}

extern "C" void ptx_tensor_strided_copy_f64(
    const double* in, double* out,
    const size_t* shape, const size_t* in_strides,
    int ndim, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    k_strided_copy_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(
        in, out, shape, in_strides, ndim, n);
}

__global__ void k_strided_copy_u8(const unsigned char* __restrict__ in, unsigned char* __restrict__ out,
                                   const size_t* __restrict__ shape,
                                   const size_t* __restrict__ in_strides,
                                   int ndim, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    size_t src_offset = 0;
    size_t remaining = idx;
    for (int d = ndim - 1; d >= 0; d--) {
        size_t coord = remaining % shape[d];
        remaining /= shape[d];
        src_offset += coord * in_strides[d];
    }
    out[idx] = in[src_offset];
}

extern "C" void ptx_tensor_strided_copy_u8(
    const unsigned char* in, unsigned char* out,
    const size_t* shape, const size_t* in_strides,
    int ndim, size_t n, cudaStream_t stream
) {
    if (n == 0) return;
    k_strided_copy_u8<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(
        in, out, shape, in_strides, ndim, n);
}

