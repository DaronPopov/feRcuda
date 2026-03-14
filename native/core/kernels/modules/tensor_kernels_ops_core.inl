// ============================================================================
// Binary Operation Kernels (F32)
// ============================================================================

__global__ void k_add_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void k_sub_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void k_mul_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void k_div_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void k_max_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(a[idx], b[idx]);
    }
}

__global__ void k_min_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(a[idx], b[idx]);
    }
}

__global__ void k_mod_f32(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmodf(a[idx], b[idx]);
    }
}

// Scalar broadcast variants
__global__ void k_add_scalar_f32(const float* __restrict__ a, float scalar,
                                  float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void k_sub_scalar_f32(const float* __restrict__ a, float scalar,
                                  float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - scalar;
    }
}

__global__ void k_mul_scalar_f32(const float* __restrict__ a, float scalar,
                                  float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void k_div_scalar_f32(const float* __restrict__ a, float scalar,
                                  float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

// ============================================================================
// Unary Operation Kernels (F32)
// ============================================================================

__global__ void k_neg_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -in[idx];
    }
}

__global__ void k_abs_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fabsf(in[idx]);
    }
}

__global__ void k_exp_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = expf(in[idx]);
    }
}

__global__ void k_log_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = logf(in[idx]);
    }
}

__global__ void k_sqrt_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrtf(in[idx]);
    }
}

__global__ void k_rsqrt_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = rsqrtf(in[idx]);
    }
}

__global__ void k_sin_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sinf(in[idx]);
    }
}

__global__ void k_cos_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cosf(in[idx]);
    }
}

__global__ void k_tanh_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanhf(in[idx]);
    }
}

__global__ void k_ceil_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = ceilf(in[idx]);
    }
}

__global__ void k_floor_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = floorf(in[idx]);
    }
}

__global__ void k_round_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = roundf(in[idx]);
    }
}

__global__ void k_log2_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = log2f(in[idx]);
    }
}

__global__ void k_log10_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = log10f(in[idx]);
    }
}

__global__ void k_tan_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanf(in[idx]);
    }
}

__global__ void k_sinh_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sinhf(in[idx]);
    }
}

__global__ void k_cosh_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = coshf(in[idx]);
    }
}

__global__ void k_sign_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    }
}

__global__ void k_erf_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Abramowitz & Stegun approximation (max error ~1.5e-7)
        float x = in[idx];
        float sign = (x >= 0.0f) ? 1.0f : -1.0f;
        float a = fabsf(x);
        float t = 1.0f / (1.0f + 0.3275911f * a);
        float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * expf(-a * a);
        out[idx] = sign * y;
    }
}

__global__ void k_sqr_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = x * x;
    }
}

__global__ void k_recip_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / in[idx];
    }
}

// ============================================================================
// Activation Kernels (F32)
// ============================================================================

__global__ void k_relu_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(0.0f, in[idx]);
    }
}

__global__ void k_relu6_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(6.0f, fmaxf(0.0f, in[idx]));
    }
}

__global__ void k_leaky_relu_f32(const float* __restrict__ in, float* __restrict__ out,
                                  size_t n, float alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = x > 0.0f ? x : alpha * x;
    }
}

__global__ void k_elu_f32(const float* __restrict__ in, float* __restrict__ out,
                          size_t n, float alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
    }
}

__global__ void k_selu_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = PTX_SELU_LAMBDA * (x > 0.0f ? x : PTX_SELU_ALPHA * (expf(x) - 1.0f));
    }
}

__global__ void k_gelu_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = PTX_SQRT_2_OVER_PI * (x + PTX_GELU_COEF * x3);
        out[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void k_sigmoid_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}

__global__ void k_silu_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = x / (1.0f + expf(-x));  // x * sigmoid(x)
    }
}

__global__ void k_softplus_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        // Numerically stable: log(1 + exp(x)) = x + log(1 + exp(-x)) for x > 0
        if (x > 20.0f) {
            out[idx] = x;
        } else if (x < -20.0f) {
            out[idx] = expf(x);
        } else {
            out[idx] = log1pf(expf(x));
        }
    }
}

__global__ void k_mish_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        float sp = (x > 20.0f) ? x : log1pf(expf(x));  // softplus
        out[idx] = x * tanhf(sp);
    }
}

__global__ void k_gelu_tanh_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        float inner = PTX_SQRT_2_OVER_PI * (x + PTX_GELU_COEF * x * x * x);
        out[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void k_hardswish_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
    }
}

__global__ void k_hardsigmoid_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(fmaxf(in[idx] / 6.0f + 0.5f, 0.0f), 1.0f);
    }
}

// ============================================================================
// Affine/Transform Kernels (F32)
// ============================================================================

__global__ void k_affine_f32(const float* __restrict__ in, float* __restrict__ out,
                              size_t n, float mul, float add) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * mul + add;
    }
}

__global__ void k_clamp_f32(const float* __restrict__ in, float* __restrict__ out,
                             size_t n, float min_val, float max_val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(max_val, fmaxf(min_val, in[idx]));
    }
}

__global__ void k_powf_f32(const float* __restrict__ in, float* __restrict__ out,
                            size_t n, float exp) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = powf(in[idx], exp);
    }
}

// ============================================================================
// Comparison Kernels (F32)
// ============================================================================

__global__ void k_cmp_eq_f32(const float* __restrict__ a, const float* __restrict__ b,
                              uint8_t* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] == b[idx]) ? 1 : 0;
    }
}

__global__ void k_cmp_lt_f32(const float* __restrict__ a, const float* __restrict__ b,
                              uint8_t* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? 1 : 0;
    }
}

__global__ void k_cmp_le_f32(const float* __restrict__ a, const float* __restrict__ b,
                              uint8_t* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] <= b[idx]) ? 1 : 0;
    }
}

__global__ void k_cmp_gt_f32(const float* __restrict__ a, const float* __restrict__ b,
                              uint8_t* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? 1 : 0;
    }
}

__global__ void k_cmp_ge_f32(const float* __restrict__ a, const float* __restrict__ b,
                              uint8_t* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] >= b[idx]) ? 1 : 0;
    }
}

__global__ void k_cmp_ne_f32(const float* __restrict__ a, const float* __restrict__ b,
                              uint8_t* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] != b[idx]) ? 1 : 0;
    }
}

// ============================================================================
// Where/Select Kernel
// ============================================================================

__global__ void k_where_f32(const uint8_t* __restrict__ cond, const float* __restrict__ t,
                             const float* __restrict__ f, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cond[idx] ? t[idx] : f[idx];
    }
}

// ============================================================================
// Copy/Fill Kernels
// ============================================================================

__global__ void k_copy_f32(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void k_fill_f32(float* __restrict__ out, size_t n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

// ============================================================================
// Reduction Kernels (F32) - Using shared memory for efficiency
// ============================================================================

// Warp-level reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    for (int offset = PTX_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    for (int offset = PTX_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_min_f32(float val) {
    for (int offset = PTX_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Reduce along a dimension: [outer, reduce, inner] -> [outer, inner]
__global__ void k_reduce_sum_f32(const float* __restrict__ in, float* __restrict__ out,
                                  size_t outer, size_t reduce, size_t inner) {
    extern __shared__ float sdata[];

    size_t out_idx = blockIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    // Each thread sums multiple elements
    float sum = 0.0f;
    for (size_t r = threadIdx.x; r < reduce; r += blockDim.x) {
        size_t in_idx = outer_idx * reduce * inner + r * inner + inner_idx;
        sum += in[in_idx];
    }

    // Warp-level reduction
    sum = warp_reduce_sum_f32(sum);

    // Store to shared memory
    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) {
        sdata[wid] = sum;
    }
    __syncthreads();

    // Final reduction in first warp
    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    if (threadIdx.x < numWarps) {
        sum = sdata[threadIdx.x];
    } else {
        sum = 0.0f;
    }

    if (wid == 0) {
        sum = warp_reduce_sum_f32(sum);
        if (lane == 0) {
            out[out_idx] = sum;
        }
    }
}

__global__ void k_reduce_max_f32(const float* __restrict__ in, float* __restrict__ out,
                                  size_t outer, size_t reduce, size_t inner) {
    extern __shared__ float sdata[];

    size_t out_idx = blockIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    float max_val = -INFINITY;
    for (size_t r = threadIdx.x; r < reduce; r += blockDim.x) {
        size_t in_idx = outer_idx * reduce * inner + r * inner + inner_idx;
        max_val = fmaxf(max_val, in[in_idx]);
    }

    max_val = warp_reduce_max_f32(max_val);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) {
        sdata[wid] = max_val;
    }
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    if (threadIdx.x < numWarps) {
        max_val = sdata[threadIdx.x];
    } else {
        max_val = -INFINITY;
    }

    if (wid == 0) {
        max_val = warp_reduce_max_f32(max_val);
        if (lane == 0) {
            out[out_idx] = max_val;
        }
    }
}

__global__ void k_reduce_min_f32(const float* __restrict__ in, float* __restrict__ out,
                                  size_t outer, size_t reduce, size_t inner) {
    extern __shared__ float sdata[];

    size_t out_idx = blockIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    float min_val = INFINITY;
    for (size_t r = threadIdx.x; r < reduce; r += blockDim.x) {
        size_t in_idx = outer_idx * reduce * inner + r * inner + inner_idx;
        min_val = fminf(min_val, in[in_idx]);
    }

    min_val = warp_reduce_min_f32(min_val);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) {
        sdata[wid] = min_val;
    }
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    if (threadIdx.x < numWarps) {
        min_val = sdata[threadIdx.x];
    } else {
        min_val = INFINITY;
    }

    if (wid == 0) {
        min_val = warp_reduce_min_f32(min_val);
        if (lane == 0) {
            out[out_idx] = min_val;
        }
    }
}

__device__ __forceinline__ float warp_reduce_prod_f32(float val) {
    for (int offset = PTX_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void k_reduce_prod_f32(const float* __restrict__ in, float* __restrict__ out,
                                   size_t outer, size_t reduce, size_t inner) {
    extern __shared__ float sdata[];

    size_t out_idx = blockIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    float prod = 1.0f;
    for (size_t r = threadIdx.x; r < reduce; r += blockDim.x) {
        size_t in_idx = outer_idx * reduce * inner + r * inner + inner_idx;
        prod *= in[in_idx];
    }

    prod = warp_reduce_prod_f32(prod);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) {
        sdata[wid] = prod;
    }
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    if (threadIdx.x < numWarps) {
        prod = sdata[threadIdx.x];
    } else {
        prod = 1.0f;
    }

    if (wid == 0) {
        prod = warp_reduce_prod_f32(prod);
        if (lane == 0) {
            out[out_idx] = prod;
        }
    }
}

// Argmax/Argmin: one thread per (outer, inner) lane, sequential scan over reduce dim
__global__ void k_reduce_argmax_f32(const float* __restrict__ in, int32_t* __restrict__ out,
                                     size_t outer, size_t reduce, size_t inner) {
    size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = outer * inner;
    if (lane >= total) return;

    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t base = o * reduce * inner + i;

    float best = -INFINITY;
    int32_t best_idx = 0;
    for (size_t r = 0; r < reduce; r++) {
        float v = in[base + r * inner];
        if (v > best) {
            best = v;
            best_idx = (int32_t)r;
        }
    }
    out[lane] = best_idx;
}

__global__ void k_reduce_argmin_f32(const float* __restrict__ in, int32_t* __restrict__ out,
                                     size_t outer, size_t reduce, size_t inner) {
    size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = outer * inner;
    if (lane >= total) return;

    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t base = o * reduce * inner + i;

    float best = INFINITY;
    int32_t best_idx = 0;
    for (size_t r = 0; r < reduce; r++) {
        float v = in[base + r * inner];
        if (v < best) {
            best = v;
            best_idx = (int32_t)r;
        }
    }
    out[lane] = best_idx;
}

// ============================================================================
// Softmax Kernels (F32)
// ============================================================================

// Softmax along last dimension: [batch, dim] -> [batch, dim]
__global__ void k_softmax_f32(const float* __restrict__ in, float* __restrict__ out,
                               size_t batch, size_t dim) {
    extern __shared__ float sdata[];

    size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch) return;

    const float* in_row = in + batch_idx * dim;
    float* out_row = out + batch_idx * dim;

    // Step 1: Find max for numerical stability
    float max_val = -INFINITY;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    max_val = warp_reduce_max_f32(max_val);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) {
        sdata[wid] = max_val;
    }
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    if (threadIdx.x < numWarps) {
        max_val = sdata[threadIdx.x];
    } else {
        max_val = -INFINITY;
    }

    if (wid == 0) {
        max_val = warp_reduce_max_f32(max_val);
        if (lane == 0) {
            sdata[0] = max_val;  // Store final max
        }
    }
    __syncthreads();
    max_val = sdata[0];

    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float exp_val = expf(in_row[i] - max_val);
        out_row[i] = exp_val;  // Store temporarily
        sum += exp_val;
    }
    sum = warp_reduce_sum_f32(sum);

    if (lane == 0) {
        sdata[wid] = sum;
    }
    __syncthreads();

    if (threadIdx.x < numWarps) {
        sum = sdata[threadIdx.x];
    } else {
        sum = 0.0f;
    }

    if (wid == 0) {
        sum = warp_reduce_sum_f32(sum);
        if (lane == 0) {
            sdata[0] = sum;  // Store final sum
        }
    }
    __syncthreads();
    sum = sdata[0];

    // Step 3: Normalize
    float inv_sum = 1.0f / sum;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] *= inv_sum;
    }
}

__global__ void k_log_softmax_f32(const float* __restrict__ in, float* __restrict__ out,
                                   size_t batch, size_t dim) {
    extern __shared__ float sdata[];

    size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch) return;

    const float* in_row = in + batch_idx * dim;
    float* out_row = out + batch_idx * dim;

    // Step 1: Find max
    float max_val = -INFINITY;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    max_val = warp_reduce_max_f32(max_val);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) {
        sdata[wid] = max_val;
    }
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    if (threadIdx.x < numWarps) {
        max_val = sdata[threadIdx.x];
    } else {
        max_val = -INFINITY;
    }

    if (wid == 0) {
        max_val = warp_reduce_max_f32(max_val);
        if (lane == 0) {
            sdata[0] = max_val;
        }
    }
    __syncthreads();
    max_val = sdata[0];

    // Step 2: Compute sum(exp(x - max))
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += expf(in_row[i] - max_val);
    }
    sum = warp_reduce_sum_f32(sum);

    if (lane == 0) {
        sdata[wid] = sum;
    }
    __syncthreads();

    if (threadIdx.x < numWarps) {
        sum = sdata[threadIdx.x];
    } else {
        sum = 0.0f;
    }

    if (wid == 0) {
        sum = warp_reduce_sum_f32(sum);
        if (lane == 0) {
            sdata[0] = sum;
        }
    }
    __syncthreads();
    sum = sdata[0];

    // Step 3: Compute x - max - log(sum)
    float log_sum = logf(sum);
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] = in_row[i] - max_val - log_sum;
    }
}

// ============================================================================
// F64 Kernels
// ============================================================================

__global__ void k_add_f64(const double* __restrict__ a, const double* __restrict__ b,
                          double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void k_sub_f64(const double* __restrict__ a, const double* __restrict__ b,
                          double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] - b[idx];
}

__global__ void k_mul_f64(const double* __restrict__ a, const double* __restrict__ b,
                          double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

__global__ void k_div_f64(const double* __restrict__ a, const double* __restrict__ b,
                          double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] / b[idx];
}

__global__ void k_neg_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = -in[idx];
}

__global__ void k_exp_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = exp(in[idx]);
}

__global__ void k_log_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = log(in[idx]);
}

__global__ void k_sqrt_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = sqrt(in[idx]);
}

__global__ void k_tanh_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = tanh(in[idx]);
}

__global__ void k_relu_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fmax(0.0, in[idx]);
}

__global__ void k_gelu_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = in[idx];
        double x3 = x * x * x;
        double inner = 0.7978845608028654 * (x + 0.044715 * x3);
        out[idx] = 0.5 * x * (1.0 + tanh(inner));
    }
}

__global__ void k_sigmoid_f64(const double* __restrict__ in, double* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 1.0 / (1.0 + exp(-in[idx]));
}

__global__ void k_affine_f64(const double* __restrict__ in, double* __restrict__ out,
                              size_t n, double mul, double add) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * mul + add;
}

// F64 Reduction
__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    for (int offset = PTX_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ double warp_reduce_max_f64(double val) {
    for (int offset = PTX_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void k_reduce_sum_f64(const double* __restrict__ in, double* __restrict__ out,
                                  size_t outer, size_t reduce, size_t inner) {
    extern __shared__ double sdata_f64[];

    size_t out_idx = blockIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    double sum = 0.0;
    for (size_t r = threadIdx.x; r < reduce; r += blockDim.x) {
        size_t in_idx = outer_idx * reduce * inner + r * inner + inner_idx;
        sum += in[in_idx];
    }

    sum = warp_reduce_sum_f64(sum);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) sdata_f64[wid] = sum;
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    sum = (threadIdx.x < numWarps) ? sdata_f64[threadIdx.x] : 0.0;

    if (wid == 0) {
        sum = warp_reduce_sum_f64(sum);
        if (lane == 0) out[out_idx] = sum;
    }
}

__global__ void k_reduce_max_f64(const double* __restrict__ in, double* __restrict__ out,
                                  size_t outer, size_t reduce, size_t inner) {
    extern __shared__ double sdata_f64[];

    size_t out_idx = blockIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    double max_val = -INFINITY;
    for (size_t r = threadIdx.x; r < reduce; r += blockDim.x) {
        size_t in_idx = outer_idx * reduce * inner + r * inner + inner_idx;
        max_val = fmax(max_val, in[in_idx]);
    }

    max_val = warp_reduce_max_f64(max_val);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) sdata_f64[wid] = max_val;
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    max_val = (threadIdx.x < numWarps) ? sdata_f64[threadIdx.x] : -INFINITY;

    if (wid == 0) {
        max_val = warp_reduce_max_f64(max_val);
        if (lane == 0) out[out_idx] = max_val;
    }
}

__global__ void k_reduce_min_f64(const double* __restrict__ in, double* __restrict__ out,
                                  size_t outer, size_t reduce, size_t inner) {
    extern __shared__ double sdata_f64[];

    size_t out_idx = blockIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    double min_val = INFINITY;
    for (size_t r = threadIdx.x; r < reduce; r += blockDim.x) {
        size_t in_idx = outer_idx * reduce * inner + r * inner + inner_idx;
        min_val = fmin(min_val, in[in_idx]);
    }

    min_val = warp_reduce_max_f64(min_val);  // Note: using max for clarity in code

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) sdata_f64[wid] = min_val;
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    min_val = (threadIdx.x < numWarps) ? sdata_f64[threadIdx.x] : INFINITY;

    if (wid == 0) {
        for (int offset = PTX_WARP_SIZE / 2; offset > 0; offset /= 2) {
            min_val = fmin(min_val, __shfl_down_sync(0xffffffff, min_val, offset));
        }
        if (lane == 0) out[out_idx] = min_val;
    }
}

__global__ void k_softmax_f64(const double* __restrict__ in, double* __restrict__ out,
                               size_t batch, size_t dim) {
    extern __shared__ double sdata_f64[];

    size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch) return;

    const double* in_row = in + batch_idx * dim;
    double* out_row = out + batch_idx * dim;

    // Find max
    double max_val = -INFINITY;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        max_val = fmax(max_val, in_row[i]);
    }
    max_val = warp_reduce_max_f64(max_val);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) sdata_f64[wid] = max_val;
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    max_val = (threadIdx.x < numWarps) ? sdata_f64[threadIdx.x] : -INFINITY;

    if (wid == 0) {
        max_val = warp_reduce_max_f64(max_val);
        if (lane == 0) sdata_f64[0] = max_val;
    }
    __syncthreads();
    max_val = sdata_f64[0];

    // Compute exp and sum
    double sum = 0.0;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        double exp_val = exp(in_row[i] - max_val);
        out_row[i] = exp_val;
        sum += exp_val;
    }
    sum = warp_reduce_sum_f64(sum);

    if (lane == 0) sdata_f64[wid] = sum;
    __syncthreads();

    sum = (threadIdx.x < numWarps) ? sdata_f64[threadIdx.x] : 0.0;

    if (wid == 0) {
        sum = warp_reduce_sum_f64(sum);
        if (lane == 0) sdata_f64[0] = sum;
    }
    __syncthreads();
    sum = sdata_f64[0];

    // Normalize
    double inv_sum = 1.0 / sum;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] *= inv_sum;
    }
}

// ============================================================================
// F16 Kernels (using CUDA half)
// ============================================================================

__global__ void k_add_f16(const __half* __restrict__ a, const __half* __restrict__ b,
                          __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void k_sub_f16(const __half* __restrict__ a, const __half* __restrict__ b,
                          __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hsub(a[idx], b[idx]);
    }
}

__global__ void k_mul_f16(const __half* __restrict__ a, const __half* __restrict__ b,
                          __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hmul(a[idx], b[idx]);
    }
}

__global__ void k_div_f16(const __half* __restrict__ a, const __half* __restrict__ b,
                          __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hdiv(a[idx], b[idx]);
    }
}

__global__ void k_relu_f16(const __half* __restrict__ in, __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half zero = __float2half(0.0f);
        out[idx] = __hgt(in[idx], zero) ? in[idx] : zero;
    }
}

__global__ void k_gelu_f16(const __half* __restrict__ in, __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(in[idx]);
        float x3 = x * x * x;
        float inner = PTX_SQRT_2_OVER_PI * (x + PTX_GELU_COEF * x3);
        float result = 0.5f * x * (1.0f + tanhf(inner));
        out[idx] = __float2half(result);
    }
}

__global__ void k_sigmoid_f16(const __half* __restrict__ in, __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(in[idx]);
        out[idx] = __float2half(1.0f / (1.0f + expf(-x)));
    }
}

// F16 Softmax
__global__ void k_softmax_f16(const __half* __restrict__ in, __half* __restrict__ out,
                               size_t batch, size_t dim) {
    extern __shared__ float sdata[];

    size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch) return;

    const __half* in_row = in + batch_idx * dim;
    __half* out_row = out + batch_idx * dim;

    // Find max (in float for stability)
    float max_val = -INFINITY;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, __half2float(in_row[i]));
    }
    max_val = warp_reduce_max_f32(max_val);

    int lane = threadIdx.x % PTX_WARP_SIZE;
    int wid = threadIdx.x / PTX_WARP_SIZE;

    if (lane == 0) sdata[wid] = max_val;
    __syncthreads();

    int numWarps = (blockDim.x + PTX_WARP_SIZE - 1) / PTX_WARP_SIZE;
    max_val = (threadIdx.x < numWarps) ? sdata[threadIdx.x] : -INFINITY;

    if (wid == 0) {
        max_val = warp_reduce_max_f32(max_val);
        if (lane == 0) sdata[0] = max_val;
    }
    __syncthreads();
    max_val = sdata[0];

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float exp_val = expf(__half2float(in_row[i]) - max_val);
        out_row[i] = __float2half(exp_val);
        sum += exp_val;
    }
    sum = warp_reduce_sum_f32(sum);

    if (lane == 0) sdata[wid] = sum;
    __syncthreads();

    sum = (threadIdx.x < numWarps) ? sdata[threadIdx.x] : 0.0f;

    if (wid == 0) {
        sum = warp_reduce_sum_f32(sum);
        if (lane == 0) sdata[0] = sum;
    }
    __syncthreads();
    sum = sdata[0];

    // Normalize
    float inv_sum = 1.0f / sum;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(out_row[i]) * inv_sum;
        out_row[i] = __float2half(val);
    }
}

// Cast kernels
__global__ void k_cast_f32_to_f16(const float* __restrict__ in, __half* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

__global__ void k_cast_f16_to_f32(const __half* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

// ============================================================================
// External API Implementation
// ============================================================================

// F32 Binary
extern "C" void ptx_tensor_add_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream) {
    k_add_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_sub_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream) {
    k_sub_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_mul_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream) {
    k_mul_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_div_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream) {
    k_div_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_max_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream) {
    k_max_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_min_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream) {
    k_min_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_mod_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream) {
    k_mod_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

// F32 Scalar broadcast
extern "C" void ptx_tensor_add_scalar_f32(float* a, float scalar, float* out, size_t n, cudaStream_t stream) {
    k_add_scalar_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, scalar, out, n);
}

extern "C" void ptx_tensor_sub_scalar_f32(float* a, float scalar, float* out, size_t n, cudaStream_t stream) {
    k_sub_scalar_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, scalar, out, n);
}

extern "C" void ptx_tensor_mul_scalar_f32(float* a, float scalar, float* out, size_t n, cudaStream_t stream) {
    k_mul_scalar_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, scalar, out, n);
}

extern "C" void ptx_tensor_div_scalar_f32(float* a, float scalar, float* out, size_t n, cudaStream_t stream) {
    k_div_scalar_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, scalar, out, n);
}

// F32 Unary
extern "C" void ptx_tensor_neg_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_neg_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_abs_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_abs_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_exp_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_exp_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_log_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_log_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sqrt_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_sqrt_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_rsqrt_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_rsqrt_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sin_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_sin_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_cos_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_cos_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_tanh_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_tanh_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_ceil_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_ceil_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_floor_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_floor_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_round_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_round_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sqr_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_sqr_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_recip_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_recip_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_log2_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_log2_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_log10_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_log10_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_tan_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_tan_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sinh_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_sinh_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_cosh_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_cosh_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sign_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_sign_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_erf_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_erf_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

// F32 Activations
extern "C" void ptx_tensor_relu_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_relu_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_relu6_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_relu6_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_leaky_relu_f32(float* in, float* out, size_t n, float alpha, cudaStream_t stream) {
    k_leaky_relu_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n, alpha);
}

extern "C" void ptx_tensor_elu_f32(float* in, float* out, size_t n, float alpha, cudaStream_t stream) {
    k_elu_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n, alpha);
}

extern "C" void ptx_tensor_selu_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_selu_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_gelu_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_gelu_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sigmoid_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_sigmoid_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_silu_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_silu_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_softplus_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_softplus_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_mish_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_mish_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_gelu_tanh_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_gelu_tanh_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_hardswish_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_hardswish_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_hardsigmoid_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_hardsigmoid_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

// F32 Affine/Transform
extern "C" void ptx_tensor_affine_f32(float* in, float* out, size_t n, float mul, float add, cudaStream_t stream) {
    k_affine_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n, mul, add);
}

extern "C" void ptx_tensor_clamp_f32(float* in, float* out, size_t n, float min_val, float max_val, cudaStream_t stream) {
    k_clamp_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n, min_val, max_val);
}

extern "C" void ptx_tensor_powf_f32(float* in, float* out, size_t n, float exp, cudaStream_t stream) {
    k_powf_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n, exp);
}

// F32 Comparison
extern "C" void ptx_tensor_cmp_eq_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream) {
    k_cmp_eq_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_cmp_lt_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream) {
    k_cmp_lt_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_cmp_le_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream) {
    k_cmp_le_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_cmp_gt_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream) {
    k_cmp_gt_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_cmp_ge_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream) {
    k_cmp_ge_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_cmp_ne_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream) {
    k_cmp_ne_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

// F32 Where
extern "C" void ptx_tensor_where_f32(uint8_t* cond, float* t, float* f, float* out, size_t n, cudaStream_t stream) {
    k_where_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(cond, t, f, out, n);
}

// F32 Copy/Fill
extern "C" void ptx_tensor_copy_f32(float* in, float* out, size_t n, cudaStream_t stream) {
    k_copy_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_fill_f32(float* out, size_t n, float value, cudaStream_t stream) {
    k_fill_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(out, n, value);
}

// F32 Reductions
extern "C" void ptx_tensor_reduce_sum_f32(float* in, float* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t num_outputs = outer * inner;
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE) * sizeof(float);
    k_reduce_sum_f32<<<num_outputs, PTX_BLOCK_SIZE, smem, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_reduce_mean_f32(float* in, float* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    // Mean = sum / count, so we compute sum first then scale
    ptx_tensor_reduce_sum_f32(in, out, outer, reduce, inner, stream);
    float scale = 1.0f / (float)reduce;
    k_mul_scalar_f32<<<PTX_GRID_SIZE(outer * inner), PTX_BLOCK_SIZE, 0, stream>>>(out, scale, out, outer * inner);
}

extern "C" void ptx_tensor_reduce_max_f32(float* in, float* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t num_outputs = outer * inner;
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE) * sizeof(float);
    k_reduce_max_f32<<<num_outputs, PTX_BLOCK_SIZE, smem, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_reduce_min_f32(float* in, float* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t num_outputs = outer * inner;
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE) * sizeof(float);
    k_reduce_min_f32<<<num_outputs, PTX_BLOCK_SIZE, smem, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_reduce_prod_f32(float* in, float* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t num_outputs = outer * inner;
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE) * sizeof(float);
    k_reduce_prod_f32<<<num_outputs, PTX_BLOCK_SIZE, smem, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_reduce_argmax_f32(float* in, int32_t* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t total = outer * inner;
    if (total == 0) return;
    k_reduce_argmax_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_reduce_argmin_f32(float* in, int32_t* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t total = outer * inner;
    if (total == 0) return;
    k_reduce_argmin_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(in, out, outer, reduce, inner);
}

