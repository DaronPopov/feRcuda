/*
 * PTX-OS Tensor Backward Operations
 * CUDA kernels for automatic differentiation
 */

#include "gpu/tensor_backward.h"
#include <math.h>

// Block size for kernels
#define BLOCK_SIZE 256

// ============================================================================
// Binary Operation Backward Kernels
// ============================================================================

// Add backward: both gradients are copies of grad_out
__global__ void add_backward_kernel_f32(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_a,
    float* __restrict__ grad_b,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad_out[idx];
        grad_a[idx] = g;
        grad_b[idx] = g;
    }
}

extern "C" void ptx_tensor_add_backward_f32(
    float* grad_out,
    float* grad_a,
    float* grad_b,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, grad_a, grad_b, n);
}

// Sub backward: grad_a = grad_out, grad_b = -grad_out
__global__ void sub_backward_kernel_f32(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_a,
    float* __restrict__ grad_b,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad_out[idx];
        grad_a[idx] = g;
        grad_b[idx] = -g;
    }
}

extern "C" void ptx_tensor_sub_backward_f32(
    float* grad_out,
    float* grad_a,
    float* grad_b,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sub_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, grad_a, grad_b, n);
}

// Mul backward: grad_a = grad_out * b, grad_b = grad_out * a
__global__ void mul_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ grad_a,
    float* __restrict__ grad_b,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad_out[idx];
        grad_a[idx] = g * b[idx];
        grad_b[idx] = g * a[idx];
    }
}

extern "C" void ptx_tensor_mul_backward_f32(
    float* grad_out,
    float* a,
    float* b,
    float* grad_a,
    float* grad_b,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mul_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, a, b, grad_a, grad_b, n);
}

// Div backward: grad_a = grad_out / b, grad_b = -grad_out * a / b^2
__global__ void div_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ grad_a,
    float* __restrict__ grad_b,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad_out[idx];
        float b_val = b[idx];
        grad_a[idx] = g / b_val;
        grad_b[idx] = -g * a[idx] / (b_val * b_val);
    }
}

extern "C" void ptx_tensor_div_backward_f32(
    float* grad_out,
    float* a,
    float* b,
    float* grad_a,
    float* grad_b,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    div_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, a, b, grad_a, grad_b, n);
}

// ============================================================================
// Unary Operation Backward Kernels
// ============================================================================

// Neg backward
__global__ void neg_backward_kernel_f32(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = -grad_out[idx];
    }
}

extern "C" void ptx_tensor_neg_backward_f32(
    float* grad_out,
    float* grad_in,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    neg_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, grad_in, n);
}

// Exp backward: grad_in = grad_out * output
__global__ void exp_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ output,
    float* __restrict__ grad_in,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = grad_out[idx] * output[idx];
    }
}

extern "C" void ptx_tensor_exp_backward_f32(
    float* grad_out,
    float* output,
    float* grad_in,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    exp_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, output, grad_in, n);
}

// Log backward: grad_in = grad_out / input
__global__ void log_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = grad_out[idx] / input[idx];
    }
}

extern "C" void ptx_tensor_log_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    log_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, input, grad_in, n);
}

// Sqrt backward: grad_in = grad_out * 0.5 / output
__global__ void sqrt_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ output,
    float* __restrict__ grad_in,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = grad_out[idx] * 0.5f / output[idx];
    }
}

extern "C" void ptx_tensor_sqrt_backward_f32(
    float* grad_out,
    float* output,
    float* grad_in,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sqrt_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, output, grad_in, n);
}

// Tanh backward: grad_in = grad_out * (1 - output^2)
__global__ void tanh_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ output,
    float* __restrict__ grad_in,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float o = output[idx];
        grad_in[idx] = grad_out[idx] * (1.0f - o * o);
    }
}

extern "C" void ptx_tensor_tanh_backward_f32(
    float* grad_out,
    float* output,
    float* grad_in,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    tanh_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, output, grad_in, n);
}

// Sin backward: grad_in = grad_out * cos(input)
__global__ void sin_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = grad_out[idx] * cosf(input[idx]);
    }
}

extern "C" void ptx_tensor_sin_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sin_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, input, grad_in, n);
}

// Cos backward: grad_in = -grad_out * sin(input)
__global__ void cos_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = -grad_out[idx] * sinf(input[idx]);
    }
}

extern "C" void ptx_tensor_cos_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cos_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, input, grad_in, n);
}

// Sqr backward: grad_in = grad_out * 2 * input
__global__ void sqr_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = grad_out[idx] * 2.0f * input[idx];
    }
}

extern "C" void ptx_tensor_sqr_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sqr_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, input, grad_in, n);
}

// Recip backward: grad_in = -grad_out / input^2
__global__ void recip_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        grad_in[idx] = -grad_out[idx] / (x * x);
    }
}

extern "C" void ptx_tensor_recip_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    recip_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, input, grad_in, n);
}

// ============================================================================
// Activation Backward Kernels
// ============================================================================

// ReLU backward: grad_in = grad_out * (input > 0)
__global__ void relu_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = (input[idx] > 0.0f) ? grad_out[idx] : 0.0f;
    }
}

extern "C" void ptx_tensor_relu_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, input, grad_in, n);
}

// GELU backward (tanh approximation)
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Derivative is complex, using approximation
__global__ void gelu_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        float pdf = 0.3989422804f * expf(-0.5f * x * x);
        grad_in[idx] = grad_out[idx] * (cdf + x * pdf);
    }
}

extern "C" void ptx_tensor_gelu_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gelu_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, input, grad_in, n);
}

// Sigmoid backward: grad_in = grad_out * output * (1 - output)
__global__ void sigmoid_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ output,
    float* __restrict__ grad_in,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float o = output[idx];
        grad_in[idx] = grad_out[idx] * o * (1.0f - o);
    }
}

extern "C" void ptx_tensor_sigmoid_backward_f32(
    float* grad_out,
    float* output,
    float* grad_in,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoid_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, output, grad_in, n);
}

// SiLU/Swish backward: grad_in = grad_out * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
__global__ void silu_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        grad_in[idx] = grad_out[idx] * (sig + x * sig * (1.0f - sig));
    }
}

extern "C" void ptx_tensor_silu_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    silu_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, input, grad_in, n);
}

// Softplus backward: grad_in = grad_out * sigmoid(input)
__global__ void softplus_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sig = 1.0f / (1.0f + expf(-input[idx]));
        grad_in[idx] = grad_out[idx] * sig;
    }
}

extern "C" void ptx_tensor_softplus_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    softplus_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, input, grad_in, n);
}

// Leaky ReLU backward
__global__ void leaky_relu_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_in,
    float alpha,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = (input[idx] > 0.0f) ? grad_out[idx] : alpha * grad_out[idx];
    }
}

extern "C" void ptx_tensor_leaky_relu_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    float alpha,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    leaky_relu_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, input, grad_in, alpha, n);
}

// ELU backward
__global__ void elu_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    const float* __restrict__ output,
    float* __restrict__ grad_in,
    float alpha,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        if (x > 0.0f) {
            grad_in[idx] = grad_out[idx];
        } else {
            // For x <= 0: output = alpha * (exp(x) - 1)
            // gradient = alpha * exp(x) = output + alpha
            grad_in[idx] = grad_out[idx] * (output[idx] + alpha);
        }
    }
}

extern "C" void ptx_tensor_elu_backward_f32(
    float* grad_out,
    float* input,
    float* output,
    float* grad_in,
    float alpha,
    size_t n,
    cudaStream_t stream
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    elu_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, input, output, grad_in, alpha, n);
}

// ============================================================================
// Reduction Backward Kernels
// ============================================================================

// Sum backward: broadcast grad_out to all positions
__global__ void sum_backward_kernel_f32(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    size_t outer,
    size_t reduce,
    size_t inner
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = outer * reduce * inner;

    if (idx < total) {
        size_t o = idx / (reduce * inner);
        size_t i = idx % inner;

        // Output position in grad_out
        size_t out_idx = o * inner + i;
        grad_in[idx] = grad_out[out_idx];
    }
}

extern "C" void ptx_tensor_sum_backward_f32(
    float* grad_out,
    float* grad_in,
    size_t outer,
    size_t reduce,
    size_t inner,
    cudaStream_t stream
) {
    size_t n = outer * reduce * inner;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sum_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, grad_in, outer, reduce, inner);
}

// Mean backward: broadcast grad_out / reduce_size
__global__ void mean_backward_kernel_f32(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    size_t outer,
    size_t reduce,
    size_t inner
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = outer * reduce * inner;

    if (idx < total) {
        size_t o = idx / (reduce * inner);
        size_t i = idx % inner;

        size_t out_idx = o * inner + i;
        grad_in[idx] = grad_out[out_idx] / (float)reduce;
    }
}

extern "C" void ptx_tensor_mean_backward_f32(
    float* grad_out,
    float* grad_in,
    size_t outer,
    size_t reduce,
    size_t inner,
    cudaStream_t stream
) {
    size_t n = outer * reduce * inner;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mean_backward_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(grad_out, grad_in, outer, reduce, inner);
}

// ============================================================================
// Softmax Backward Kernel
// ============================================================================

// Softmax backward: grad_in = output * (grad_out - sum(grad_out * output))
__global__ void softmax_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ output,
    float* __restrict__ grad_in,
    size_t batch,
    size_t dim
) {
    size_t b = blockIdx.x;
    if (b >= batch) return;

    // Shared memory for reduction
    extern __shared__ float sdata[];

    const float* g_row = grad_out + b * dim;
    const float* o_row = output + b * dim;
    float* gi_row = grad_in + b * dim;

    // Compute sum(grad_out * output) for this batch
    float local_sum = 0.0f;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += g_row[i] * o_row[i];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float sum = sdata[0];

    // Compute gradient
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        gi_row[i] = o_row[i] * (g_row[i] - sum);
    }
}

extern "C" void ptx_tensor_softmax_backward_f32(
    float* grad_out,
    float* output,
    float* grad_in,
    size_t batch,
    size_t dim,
    cudaStream_t stream
) {
    // Launch one block per batch
    int threads = (dim < BLOCK_SIZE) ? dim : BLOCK_SIZE;
    size_t shared_mem = threads * sizeof(float);
    softmax_backward_kernel_f32<<<batch, threads, shared_mem, stream>>>(
        grad_out, output, grad_in, batch, dim
    );
}

// Log-softmax backward: grad_in = grad_out - softmax * sum(grad_out)
__global__ void log_softmax_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ output,  // log_softmax output
    float* __restrict__ grad_in,
    size_t batch,
    size_t dim
) {
    size_t b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ float sdata[];

    const float* g_row = grad_out + b * dim;
    const float* o_row = output + b * dim;
    float* gi_row = grad_in + b * dim;

    // Compute sum(grad_out)
    float local_sum = 0.0f;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += g_row[i];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float sum = sdata[0];

    // grad_in = grad_out - exp(output) * sum = grad_out - softmax * sum
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float softmax_i = expf(o_row[i]);
        gi_row[i] = g_row[i] - softmax_i * sum;
    }
}

extern "C" void ptx_tensor_log_softmax_backward_f32(
    float* grad_out,
    float* output,
    float* grad_in,
    size_t batch,
    size_t dim,
    cudaStream_t stream
) {
    int threads = (dim < BLOCK_SIZE) ? dim : BLOCK_SIZE;
    size_t shared_mem = threads * sizeof(float);
    log_softmax_backward_kernel_f32<<<batch, threads, shared_mem, stream>>>(
        grad_out, output, grad_in, batch, dim
    );
}

// ============================================================================
// Matmul Backward (Placeholders - actual impl uses cuBLAS)
// ============================================================================

extern "C" void ptx_tensor_matmul_backward_a_f32(
    float* grad_out,
    float* b,
    float* grad_a,
    size_t m,
    size_t n,
    size_t k,
    cudaStream_t stream
) {
    // Use cuBLAS: grad_A = grad_C @ B^T
    // This is a placeholder - actual implementation needs cuBLAS handle
}

extern "C" void ptx_tensor_matmul_backward_b_f32(
    float* grad_out,
    float* a,
    float* grad_b,
    size_t m,
    size_t n,
    size_t k,
    cudaStream_t stream
) {
    // Use cuBLAS: grad_B = A^T @ grad_C
    // This is a placeholder - actual implementation needs cuBLAS handle
}
