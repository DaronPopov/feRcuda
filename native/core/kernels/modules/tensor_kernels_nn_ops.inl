// ============================================================================
// Im2col / Col2im (for Conv2d)
// ============================================================================

__global__ void k_im2col_f32(
    const float* __restrict__ data_im,
    float* __restrict__ data_col,
    int N, int C, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilationH, int dilationW,
    int H_out, int W_out
) {
    size_t total = (size_t)N * C * kH * kW * H_out * W_out;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        size_t w_out = idx % W_out;
        size_t tmp = idx / W_out;
        size_t h_out = tmp % H_out;
        tmp /= H_out;
        size_t kw = tmp % kW;
        tmp /= kW;
        size_t kh = tmp % kH;
        tmp /= kH;
        size_t c = tmp % C;
        size_t n = tmp / C;

        int h_in = (int)h_out * strideH - padH + (int)kh * dilationH;
        int w_in = (int)w_out * strideW - padW + (int)kw * dilationW;

        float val = 0.0f;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            val = data_im[((n * C + c) * H + h_in) * W + w_in];
        }

        // Output layout: (N, C*kH*kW, H_out*W_out)
        size_t col_c = c * kH * kW + kh * kW + kw;
        size_t col_hw = h_out * W_out + w_out;
        data_col[(n * (C * kH * kW) + col_c) * (H_out * W_out) + col_hw] = val;
    }
}

extern "C" void ptx_tensor_im2col_f32(
    const float* data_im, float* data_col,
    int N, int C, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilationH, int dilationW,
    int H_out, int W_out,
    cudaStream_t stream
) {
    size_t total = (size_t)N * C * kH * kW * H_out * W_out;
    if (total == 0) return;
    k_im2col_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(
        data_im, data_col, N, C, H, W, kH, kW,
        padH, padW, strideH, strideW, dilationH, dilationW, H_out, W_out);
}

__global__ void k_col2im_f32(
    const float* __restrict__ data_col,
    float* __restrict__ data_im,
    int N, int C, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilationH, int dilationW,
    int H_out, int W_out
) {
    size_t total = (size_t)N * C * H * W;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        size_t w = idx % W;
        size_t tmp = idx / W;
        size_t h = tmp % H;
        tmp /= H;
        size_t c = tmp % C;
        size_t n = tmp / C;

        float sum = 0.0f;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int h_padded = (int)h + padH - kh * dilationH;
                int w_padded = (int)w + padW - kw * dilationW;

                if (h_padded % strideH != 0 || w_padded % strideW != 0) continue;
                int ho = h_padded / strideH;
                int wo = w_padded / strideW;
                if (ho < 0 || ho >= H_out || wo < 0 || wo >= W_out) continue;

                size_t col_c = c * kH * kW + kh * kW + kw;
                size_t col_hw = ho * W_out + wo;
                sum += data_col[(n * (C * kH * kW) + col_c) * (H_out * W_out) + col_hw];
            }
        }
        data_im[idx] = sum;
    }
}

extern "C" void ptx_tensor_col2im_f32(
    const float* data_col, float* data_im,
    int N, int C, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int dilationH, int dilationW,
    int H_out, int W_out,
    cudaStream_t stream
) {
    size_t total = (size_t)N * C * H * W;
    if (total == 0) return;
    // Zero the output first
    cudaMemsetAsync(data_im, 0, total * sizeof(float), stream);
    k_col2im_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(
        data_col, data_im, N, C, H, W, kH, kW,
        padH, padW, strideH, strideW, dilationH, dilationW, H_out, W_out);
}

// ============================================================================
// Pooling kernels
// ============================================================================

__global__ void k_max_pool2d_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int kH, int kW,
    int strideH, int strideW,
    int padH, int padW,
    int H_out, int W_out
) {
    size_t total = (size_t)N * C * H_out * W_out;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        size_t wo = idx % W_out;
        size_t tmp = idx / W_out;
        size_t ho = tmp % H_out;
        tmp /= H_out;
        size_t c = tmp % C;
        size_t n = tmp / C;

        float max_val = -INFINITY;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int hi = (int)ho * strideH - padH + kh;
                int wi = (int)wo * strideW - padW + kw;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    float v = input[((n * C + c) * H + hi) * W + wi];
                    if (v > max_val) max_val = v;
                }
            }
        }
        output[idx] = max_val;
    }
}

extern "C" void ptx_tensor_max_pool2d_f32(
    const float* input, float* output,
    int N, int C, int H, int W,
    int kH, int kW,
    int strideH, int strideW,
    int padH, int padW,
    int H_out, int W_out,
    cudaStream_t stream
) {
    size_t total = (size_t)N * C * H_out * W_out;
    if (total == 0) return;
    k_max_pool2d_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(
        input, output, N, C, H, W, kH, kW, strideH, strideW, padH, padW, H_out, W_out);
}

__global__ void k_avg_pool2d_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int kH, int kW,
    int strideH, int strideW,
    int padH, int padW,
    int H_out, int W_out
) {
    size_t total = (size_t)N * C * H_out * W_out;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        size_t wo = idx % W_out;
        size_t tmp = idx / W_out;
        size_t ho = tmp % H_out;
        tmp /= H_out;
        size_t c = tmp % C;
        size_t n = tmp / C;

        float sum = 0.0f;
        int count = 0;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int hi = (int)ho * strideH - padH + kh;
                int wi = (int)wo * strideW - padW + kw;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    sum += input[((n * C + c) * H + hi) * W + wi];
                    count++;
                }
            }
        }
        output[idx] = (count > 0) ? sum / (float)count : 0.0f;
    }
}

extern "C" void ptx_tensor_avg_pool2d_f32(
    const float* input, float* output,
    int N, int C, int H, int W,
    int kH, int kW,
    int strideH, int strideW,
    int padH, int padW,
    int H_out, int W_out,
    cudaStream_t stream
) {
    size_t total = (size_t)N * C * H_out * W_out;
    if (total == 0) return;
    k_avg_pool2d_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(
        input, output, N, C, H, W, kH, kW, strideH, strideW, padH, padW, H_out, W_out);
}

// ============================================================================
// Optimizer kernels
// ============================================================================

__global__ void k_sgd_step_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ velocity,
    size_t n,
    float lr, float momentum, float weight_decay
) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        float g = grads[idx];
        if (weight_decay != 0.0f) {
            g += weight_decay * params[idx];
        }
        if (momentum != 0.0f && velocity != NULL) {
            velocity[idx] = momentum * velocity[idx] + g;
            g = velocity[idx];
        }
        params[idx] -= lr * g;
    }
}

extern "C" void ptx_tensor_sgd_step_f32(
    float* params, const float* grads, float* velocity,
    size_t n, float lr, float momentum, float weight_decay,
    cudaStream_t stream
) {
    if (n == 0) return;
    k_sgd_step_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(
        params, grads, velocity, n, lr, momentum, weight_decay);
}

__global__ void k_adam_step_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ m,
    float* __restrict__ v,
    size_t n,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2
) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        float g = grads[idx];
        if (weight_decay != 0.0f) {
            g += weight_decay * params[idx];
        }
        // Update biased first moment
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // Update biased second moment
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        // Bias-corrected estimates
        float m_hat = m[idx] / bc1;
        float v_hat = v[idx] / bc2;
        // Update parameters
        params[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

extern "C" void ptx_tensor_adam_step_f32(
    float* params, const float* grads, float* m, float* v,
    size_t n, float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2,
    cudaStream_t stream
) {
    if (n == 0) return;
    k_adam_step_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(
        params, grads, m, v, n, lr, beta1, beta2, eps, weight_decay, bc1, bc2);
}
