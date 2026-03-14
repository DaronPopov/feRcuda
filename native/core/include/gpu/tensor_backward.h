/*
 * PTX-OS Tensor Backward Operations
 * Gradient kernels for automatic differentiation
 */

#ifndef PTX_TENSOR_BACKWARD_H
#define PTX_TENSOR_BACKWARD_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Binary Operation Backward Kernels
// ============================================================================

// Add backward: grad_a = grad_out, grad_b = grad_out (just copy)
void ptx_tensor_add_backward_f32(
    float* grad_out,
    float* grad_a,
    float* grad_b,
    size_t n,
    cudaStream_t stream
);

// Sub backward: grad_a = grad_out, grad_b = -grad_out
void ptx_tensor_sub_backward_f32(
    float* grad_out,
    float* grad_a,
    float* grad_b,
    size_t n,
    cudaStream_t stream
);

// Mul backward: grad_a = grad_out * b, grad_b = grad_out * a
void ptx_tensor_mul_backward_f32(
    float* grad_out,
    float* a,
    float* b,
    float* grad_a,
    float* grad_b,
    size_t n,
    cudaStream_t stream
);

// Div backward: grad_a = grad_out / b, grad_b = -grad_out * a / b^2
void ptx_tensor_div_backward_f32(
    float* grad_out,
    float* a,
    float* b,
    float* grad_a,
    float* grad_b,
    size_t n,
    cudaStream_t stream
);

// ============================================================================
// Unary Operation Backward Kernels
// ============================================================================

// Neg backward: grad_in = -grad_out
void ptx_tensor_neg_backward_f32(
    float* grad_out,
    float* grad_in,
    size_t n,
    cudaStream_t stream
);

// Exp backward: grad_in = grad_out * output (where output = exp(input))
void ptx_tensor_exp_backward_f32(
    float* grad_out,
    float* output,
    float* grad_in,
    size_t n,
    cudaStream_t stream
);

// Log backward: grad_in = grad_out / input
void ptx_tensor_log_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
);

// Sqrt backward: grad_in = grad_out * 0.5 / output (where output = sqrt(input))
void ptx_tensor_sqrt_backward_f32(
    float* grad_out,
    float* output,
    float* grad_in,
    size_t n,
    cudaStream_t stream
);

// Tanh backward: grad_in = grad_out * (1 - output^2) (where output = tanh(input))
void ptx_tensor_tanh_backward_f32(
    float* grad_out,
    float* output,
    float* grad_in,
    size_t n,
    cudaStream_t stream
);

// Sin backward: grad_in = grad_out * cos(input)
void ptx_tensor_sin_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
);

// Cos backward: grad_in = -grad_out * sin(input)
void ptx_tensor_cos_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
);

// Sqr backward: grad_in = grad_out * 2 * input
void ptx_tensor_sqr_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
);

// Recip backward: grad_in = -grad_out / input^2
void ptx_tensor_recip_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
);

// ============================================================================
// Activation Backward Kernels
// ============================================================================

// ReLU backward: grad_in = grad_out * (input > 0 ? 1 : 0)
void ptx_tensor_relu_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
);

// GELU backward (approximate)
void ptx_tensor_gelu_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
);

// Sigmoid backward: grad_in = grad_out * output * (1 - output)
void ptx_tensor_sigmoid_backward_f32(
    float* grad_out,
    float* output,
    float* grad_in,
    size_t n,
    cudaStream_t stream
);

// SiLU/Swish backward
void ptx_tensor_silu_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
);

// Softplus backward: grad_in = grad_out * sigmoid(input)
void ptx_tensor_softplus_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    size_t n,
    cudaStream_t stream
);

// Leaky ReLU backward
void ptx_tensor_leaky_relu_backward_f32(
    float* grad_out,
    float* input,
    float* grad_in,
    float alpha,
    size_t n,
    cudaStream_t stream
);

// ELU backward
void ptx_tensor_elu_backward_f32(
    float* grad_out,
    float* input,
    float* output,
    float* grad_in,
    float alpha,
    size_t n,
    cudaStream_t stream
);

// ============================================================================
// Reduction Backward Kernels
// ============================================================================

// Sum backward: broadcast grad_out to all positions in the reduced dimension
void ptx_tensor_sum_backward_f32(
    float* grad_out,
    float* grad_in,
    size_t outer,
    size_t reduce,
    size_t inner,
    cudaStream_t stream
);

// Mean backward: broadcast grad_out / reduce_size to all positions
void ptx_tensor_mean_backward_f32(
    float* grad_out,
    float* grad_in,
    size_t outer,
    size_t reduce,
    size_t inner,
    cudaStream_t stream
);

// ============================================================================
// Softmax Backward Kernel
// ============================================================================

// Softmax backward: grad_in = output * (grad_out - sum(grad_out * output))
void ptx_tensor_softmax_backward_f32(
    float* grad_out,
    float* output,
    float* grad_in,
    size_t batch,
    size_t dim,
    cudaStream_t stream
);

// Log-softmax backward
void ptx_tensor_log_softmax_backward_f32(
    float* grad_out,
    float* output,
    float* grad_in,
    size_t batch,
    size_t dim,
    cudaStream_t stream
);

// ============================================================================
// Matmul Backward Kernels (Note: These use cuBLAS)
// ============================================================================

// These are placeholders - actual implementation uses cuBLAS for efficiency

// Matmul backward for A: grad_A = grad_C @ B^T
void ptx_tensor_matmul_backward_a_f32(
    float* grad_out,    // (M, N)
    float* b,           // (K, N)
    float* grad_a,      // (M, K)
    size_t m,
    size_t n,
    size_t k,
    cudaStream_t stream
);

// Matmul backward for B: grad_B = A^T @ grad_C
void ptx_tensor_matmul_backward_b_f32(
    float* grad_out,    // (M, N)
    float* a,           // (M, K)
    float* grad_b,      // (K, N)
    size_t m,
    size_t n,
    size_t k,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // PTX_TENSOR_BACKWARD_H
