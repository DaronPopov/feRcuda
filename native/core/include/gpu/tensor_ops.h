/*
 * PTX-OS Tensor Operations - Zero-Copy GPU Compute
 * Native CUDA kernels for element-wise, reduction, and activation operations
 */

#ifndef PTX_TENSOR_OPS_H
#define PTX_TENSOR_OPS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Tensor Operation Opcodes (for task queue dispatch)
// ============================================================================

typedef enum PTXTensorOpcode {
    // Binary Operations (0x10 - 0x1F)
    PTX_OP_ADD          = 0x10,
    PTX_OP_SUB          = 0x11,
    PTX_OP_MUL          = 0x12,
    PTX_OP_DIV          = 0x13,
    PTX_OP_MAX          = 0x14,
    PTX_OP_MIN          = 0x15,
    PTX_OP_POW          = 0x16,
    PTX_OP_MOD          = 0x17,

    // Unary Operations (0x20 - 0x3F)
    PTX_OP_NEG          = 0x20,
    PTX_OP_ABS          = 0x21,
    PTX_OP_EXP          = 0x22,
    PTX_OP_LOG          = 0x23,
    PTX_OP_LOG2         = 0x24,
    PTX_OP_LOG10        = 0x25,
    PTX_OP_SQRT         = 0x26,
    PTX_OP_RSQRT        = 0x27,
    PTX_OP_SIN          = 0x28,
    PTX_OP_COS          = 0x29,
    PTX_OP_TAN          = 0x2A,
    PTX_OP_TANH         = 0x2B,
    PTX_OP_SINH         = 0x2C,
    PTX_OP_COSH         = 0x2D,
    PTX_OP_CEIL         = 0x2E,
    PTX_OP_FLOOR        = 0x2F,
    PTX_OP_ROUND        = 0x30,
    PTX_OP_SIGN         = 0x31,
    PTX_OP_RECIP        = 0x32,
    PTX_OP_SQR          = 0x33,
    PTX_OP_ERF          = 0x34,

    // Activation Functions (0x40 - 0x4F)
    PTX_OP_RELU         = 0x40,
    PTX_OP_RELU6        = 0x41,
    PTX_OP_LEAKY_RELU   = 0x42,
    PTX_OP_ELU          = 0x43,
    PTX_OP_SELU         = 0x44,
    PTX_OP_GELU         = 0x45,
    PTX_OP_GELU_TANH    = 0x46,
    PTX_OP_SIGMOID      = 0x47,
    PTX_OP_SILU         = 0x48,  // Swish
    PTX_OP_SOFTPLUS     = 0x49,
    PTX_OP_MISH         = 0x4A,
    PTX_OP_HARDSWISH    = 0x4B,
    PTX_OP_HARDSIGMOID  = 0x4C,

    // Reduction Operations (0x50 - 0x5F)
    PTX_OP_REDUCE_SUM   = 0x50,
    PTX_OP_REDUCE_MEAN  = 0x51,
    PTX_OP_REDUCE_MAX   = 0x52,
    PTX_OP_REDUCE_MIN   = 0x53,
    PTX_OP_REDUCE_PROD  = 0x54,
    PTX_OP_REDUCE_ARGMAX= 0x55,
    PTX_OP_REDUCE_ARGMIN= 0x56,

    // Softmax Operations (0x60 - 0x6F)
    PTX_OP_SOFTMAX      = 0x60,
    PTX_OP_LOG_SOFTMAX  = 0x61,

    // Comparison Operations (0x70 - 0x7F)
    PTX_OP_CMP_EQ       = 0x70,
    PTX_OP_CMP_NE       = 0x71,
    PTX_OP_CMP_LT       = 0x72,
    PTX_OP_CMP_LE       = 0x73,
    PTX_OP_CMP_GT       = 0x74,
    PTX_OP_CMP_GE       = 0x75,

    // Affine/Transform Operations (0x80 - 0x8F)
    PTX_OP_AFFINE       = 0x80,  // out = a * x + b
    PTX_OP_CLAMP        = 0x81,
    PTX_OP_WHERE        = 0x82,  // conditional select

    // Copy Operations (0x90 - 0x9F)
    PTX_OP_COPY         = 0x90,
    PTX_OP_CAST         = 0x91,
    PTX_OP_FILL         = 0x92,

    // Gather/Scatter Operations (0xB0 - 0xB5)
    PTX_OP_GATHER       = 0xB0,
    PTX_OP_SCATTER_ADD  = 0xB1,
    PTX_OP_INDEX_SELECT = 0xB2,
    PTX_OP_INDEX_ADD    = 0xB3,

    // Scan/Prefix Operations (0xC0 - 0xC5)
    PTX_OP_CUMSUM       = 0xC0,

    // Sort/Select Operations (0xE0 - 0xEF)
    PTX_OP_ARGSORT      = 0xE1,
    PTX_OP_TOPK         = 0xE2,

} PTXTensorOpcode;

// ============================================================================
// Data Types
// ============================================================================

typedef enum PTXDType {
    PTX_DTYPE_F32  = 0,
    PTX_DTYPE_F64  = 1,
    PTX_DTYPE_F16  = 2,
    PTX_DTYPE_BF16 = 3,
    PTX_DTYPE_I8   = 4,
    PTX_DTYPE_I16  = 5,
    PTX_DTYPE_I32  = 6,
    PTX_DTYPE_I64  = 7,
    PTX_DTYPE_U8   = 8,
    PTX_DTYPE_U32  = 9,
} PTXDType;

// ============================================================================
// Tensor Descriptor (GPU-resident)
// ============================================================================

typedef struct PTXTensorDesc {
    void* data;             // GPU memory pointer
    size_t elem_count;      // Total number of elements
    PTXDType dtype;         // Data type
    uint32_t ndim;          // Number of dimensions
    uint32_t shape[8];      // Shape (max 8 dims)
    uint32_t strides[8];    // Strides in elements
    bool is_contiguous;     // Whether memory is contiguous
} PTXTensorDesc;

// ============================================================================
// Tensor Operation Descriptor (for queue dispatch)
// ============================================================================

typedef struct PTXTensorOp {
    PTXTensorOpcode opcode;
    PTXDType dtype;

    // Input tensors
    void* input_a;
    void* input_b;          // For binary ops, NULL for unary
    void* output;

    size_t elem_count;

    // Scalar parameters (for affine, leaky_relu alpha, etc.)
    float scalar_a;
    float scalar_b;

    // Reduction parameters
    uint32_t reduce_dim;
    uint32_t reduce_size;   // Size of dimension being reduced
    uint32_t outer_size;    // Product of dims before reduce_dim
    uint32_t inner_size;    // Product of dims after reduce_dim

    // Stream for async execution
    cudaStream_t stream;
} PTXTensorOp;

// ============================================================================
// Direct Kernel Launch API (Synchronous)
// ============================================================================

// Binary Operations
void ptx_tensor_add_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_sub_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_mul_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_div_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_max_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_min_f32(float* a, float* b, float* out, size_t n, cudaStream_t stream);

// Unary Operations
void ptx_tensor_neg_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_abs_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_exp_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_log_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_sqrt_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_rsqrt_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_sin_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_cos_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_tanh_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_ceil_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_floor_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_round_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_sqr_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_recip_f32(float* in, float* out, size_t n, cudaStream_t stream);

// Activation Functions
void ptx_tensor_relu_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_relu6_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_leaky_relu_f32(float* in, float* out, size_t n, float alpha, cudaStream_t stream);
void ptx_tensor_elu_f32(float* in, float* out, size_t n, float alpha, cudaStream_t stream);
void ptx_tensor_selu_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_gelu_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_sigmoid_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_silu_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_softplus_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_mish_f32(float* in, float* out, size_t n, cudaStream_t stream);

// Reduction Operations
void ptx_tensor_reduce_sum_f32(float* in, float* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream);
void ptx_tensor_reduce_mean_f32(float* in, float* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream);
void ptx_tensor_reduce_max_f32(float* in, float* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream);
void ptx_tensor_reduce_min_f32(float* in, float* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream);

// Gather/Scatter Operations
void ptx_tensor_gather_f32(float* in, int32_t* indices, float* out, size_t outer, size_t input_dim_size, size_t idx_dim_size, size_t inner, cudaStream_t stream);
void ptx_tensor_index_select_f32(float* in, int32_t* ids, float* out, size_t left_size, size_t src_dim_size, size_t ids_dim_size, size_t right_size, cudaStream_t stream);
void ptx_tensor_scatter_add_f32(int32_t* ids, float* src, float* out, size_t left_size, size_t src_dim_size, size_t dst_dim_size, size_t right_size, cudaStream_t stream);

// Scan/Prefix Operations
void ptx_tensor_cumsum_f32(float* in, float* out, size_t outer, size_t dim_size, size_t inner, cudaStream_t stream);

// Argsort/Sort Operations
void ptx_tensor_argsort_f32(float* in, uint32_t* out, size_t nrows, size_t ncols, int ascending, cudaStream_t stream);

// TopK/Selection Operations
void ptx_tensor_topk_f32(float* in, float* values_out, int32_t* indices_out,
                          size_t outer, size_t dim_size, size_t inner,
                          size_t k, int largest, cudaStream_t stream);

// Softmax Operations
void ptx_tensor_softmax_f32(float* in, float* out, size_t batch, size_t dim, cudaStream_t stream);
void ptx_tensor_log_softmax_f32(float* in, float* out, size_t batch, size_t dim, cudaStream_t stream);

// Affine/Transform Operations
void ptx_tensor_affine_f32(float* in, float* out, size_t n, float mul, float add, cudaStream_t stream);
void ptx_tensor_clamp_f32(float* in, float* out, size_t n, float min_val, float max_val, cudaStream_t stream);
void ptx_tensor_powf_f32(float* in, float* out, size_t n, float exp, cudaStream_t stream);

// Comparison Operations
void ptx_tensor_cmp_eq_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream);
void ptx_tensor_cmp_lt_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream);
void ptx_tensor_cmp_le_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream);
void ptx_tensor_cmp_gt_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream);
void ptx_tensor_cmp_ge_f32(float* a, float* b, uint8_t* out, size_t n, cudaStream_t stream);

// Where/Select Operation
void ptx_tensor_where_f32(uint8_t* cond, float* t, float* f, float* out, size_t n, cudaStream_t stream);

// Copy/Cast Operations
void ptx_tensor_copy_f32(float* in, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_fill_f32(float* out, size_t n, float value, cudaStream_t stream);
void ptx_tensor_cast_f32_to_f16(float* in, __half* out, size_t n, cudaStream_t stream);
void ptx_tensor_cast_f16_to_f32(__half* in, float* out, size_t n, cudaStream_t stream);

// ============================================================================
// F64 Variants
// ============================================================================

void ptx_tensor_add_f64(double* a, double* b, double* out, size_t n, cudaStream_t stream);
void ptx_tensor_sub_f64(double* a, double* b, double* out, size_t n, cudaStream_t stream);
void ptx_tensor_mul_f64(double* a, double* b, double* out, size_t n, cudaStream_t stream);
void ptx_tensor_div_f64(double* a, double* b, double* out, size_t n, cudaStream_t stream);

void ptx_tensor_neg_f64(double* in, double* out, size_t n, cudaStream_t stream);
void ptx_tensor_exp_f64(double* in, double* out, size_t n, cudaStream_t stream);
void ptx_tensor_log_f64(double* in, double* out, size_t n, cudaStream_t stream);
void ptx_tensor_sqrt_f64(double* in, double* out, size_t n, cudaStream_t stream);
void ptx_tensor_tanh_f64(double* in, double* out, size_t n, cudaStream_t stream);

void ptx_tensor_relu_f64(double* in, double* out, size_t n, cudaStream_t stream);
void ptx_tensor_gelu_f64(double* in, double* out, size_t n, cudaStream_t stream);
void ptx_tensor_sigmoid_f64(double* in, double* out, size_t n, cudaStream_t stream);

void ptx_tensor_reduce_sum_f64(double* in, double* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream);
void ptx_tensor_reduce_max_f64(double* in, double* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream);
void ptx_tensor_reduce_min_f64(double* in, double* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream);

void ptx_tensor_softmax_f64(double* in, double* out, size_t batch, size_t dim, cudaStream_t stream);
void ptx_tensor_affine_f64(double* in, double* out, size_t n, double mul, double add, cudaStream_t stream);

// ============================================================================
// F16 Variants (using CUDA half)
// ============================================================================

void ptx_tensor_add_f16(__half* a, __half* b, __half* out, size_t n, cudaStream_t stream);
void ptx_tensor_sub_f16(__half* a, __half* b, __half* out, size_t n, cudaStream_t stream);
void ptx_tensor_mul_f16(__half* a, __half* b, __half* out, size_t n, cudaStream_t stream);
void ptx_tensor_div_f16(__half* a, __half* b, __half* out, size_t n, cudaStream_t stream);

void ptx_tensor_relu_f16(__half* in, __half* out, size_t n, cudaStream_t stream);
void ptx_tensor_gelu_f16(__half* in, __half* out, size_t n, cudaStream_t stream);
void ptx_tensor_sigmoid_f16(__half* in, __half* out, size_t n, cudaStream_t stream);

void ptx_tensor_softmax_f16(__half* in, __half* out, size_t batch, size_t dim, cudaStream_t stream);

// ============================================================================
// Generic Dispatch API (Dispatches based on opcode and dtype)
// ============================================================================

// Execute a tensor operation (non-blocking)
cudaError_t ptx_tensor_dispatch(const PTXTensorOp* op);

// Execute and synchronize
cudaError_t ptx_tensor_dispatch_sync(const PTXTensorOp* op);

// ============================================================================
// CUDA Graph Integration
// ============================================================================

// Capture a sequence of tensor operations into a graph
cudaError_t ptx_tensor_graph_begin_capture(cudaStream_t stream);
cudaError_t ptx_tensor_graph_end_capture(cudaStream_t stream, cudaGraph_t* graph_out);
cudaError_t ptx_tensor_graph_instantiate(cudaGraph_t graph, cudaGraphExec_t* exec_out);
cudaError_t ptx_tensor_graph_launch(cudaGraphExec_t exec, cudaStream_t stream);

// ============================================================================
// Broadcast Helpers
// ============================================================================

// Binary op with broadcasting (scalar on right)
void ptx_tensor_add_scalar_f32(float* a, float scalar, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_sub_scalar_f32(float* a, float scalar, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_mul_scalar_f32(float* a, float scalar, float* out, size_t n, cudaStream_t stream);
void ptx_tensor_div_scalar_f32(float* a, float scalar, float* out, size_t n, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // PTX_TENSOR_OPS_H
