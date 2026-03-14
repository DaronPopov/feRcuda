//! FFI bindings for PTX-OS native CUDA tensor kernels
//!
//! Zero-copy GPU compute - all operations execute directly on GPU memory

use std::ffi::c_void;
use std::os::raw::c_int;

/// CUDA stream handle (opaque pointer)
pub type CudaStream = *mut c_void;

/// Tensor operation opcodes (must match tensor_ops.h)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PTXTensorOpcode {
    // Binary Operations
    Add = 0x10,
    Sub = 0x11,
    Mul = 0x12,
    Div = 0x13,
    Max = 0x14,
    Min = 0x15,
    Pow = 0x16,

    // Unary Operations
    Neg = 0x20,
    Abs = 0x21,
    Exp = 0x22,
    Log = 0x23,
    Log2 = 0x24,
    Log10 = 0x25,
    Sqrt = 0x26,
    Rsqrt = 0x27,
    Sin = 0x28,
    Cos = 0x29,
    Tan = 0x2A,
    Tanh = 0x2B,
    Sinh = 0x2C,
    Cosh = 0x2D,
    Ceil = 0x2E,
    Floor = 0x2F,
    Round = 0x30,
    Sign = 0x31,
    Recip = 0x32,
    Sqr = 0x33,
    Erf = 0x34,

    // Activation Functions
    Relu = 0x40,
    Relu6 = 0x41,
    LeakyRelu = 0x42,
    Elu = 0x43,
    Selu = 0x44,
    Gelu = 0x45,
    GeluTanh = 0x46,
    Sigmoid = 0x47,
    Silu = 0x48,
    Softplus = 0x49,
    Mish = 0x4A,
    HardSwish = 0x4B,
    HardSigmoid = 0x4C,

    // Reduction Operations
    ReduceSum = 0x50,
    ReduceMean = 0x51,
    ReduceMax = 0x52,
    ReduceMin = 0x53,
    ReduceProd = 0x54,
    ReduceArgmax = 0x55,
    ReduceArgmin = 0x56,

    // Softmax Operations
    Softmax = 0x60,
    LogSoftmax = 0x61,

    // Comparison Operations
    CmpEq = 0x70,
    CmpNe = 0x71,
    CmpLt = 0x72,
    CmpLe = 0x73,
    CmpGt = 0x74,
    CmpGe = 0x75,

    // Affine/Transform Operations
    Affine = 0x80,
    Clamp = 0x81,
    Where = 0x82,

    // Copy Operations
    Copy = 0x90,
    Cast = 0x91,
    Fill = 0x92,
}

/// Data types (must match tensor_ops.h)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PTXDType {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
    I8 = 4,
    I16 = 5,
    I32 = 6,
    I64 = 7,
    U8 = 8,
    U32 = 9,
}

/// Tensor operation descriptor for dispatch
#[repr(C)]
#[derive(Debug, Clone)]
pub struct PTXTensorOp {
    pub opcode: PTXTensorOpcode,
    pub dtype: PTXDType,
    pub input_a: *mut c_void,
    pub input_b: *mut c_void,
    pub output: *mut c_void,
    pub elem_count: usize,
    pub scalar_a: f32,
    pub scalar_b: f32,
    pub reduce_dim: u32,
    pub reduce_size: u32,
    pub outer_size: u32,
    pub inner_size: u32,
    pub stream: CudaStream,
}

impl Default for PTXTensorOp {
    fn default() -> Self {
        Self {
            opcode: PTXTensorOpcode::Add,
            dtype: PTXDType::F32,
            input_a: std::ptr::null_mut(),
            input_b: std::ptr::null_mut(),
            output: std::ptr::null_mut(),
            elem_count: 0,
            scalar_a: 0.0,
            scalar_b: 0.0,
            reduce_dim: 0,
            reduce_size: 0,
            outer_size: 0,
            inner_size: 0,
            stream: std::ptr::null_mut(),
        }
    }
}

// Ensure PTXTensorOp is Send + Sync (pointers are to GPU memory)
unsafe impl Send for PTXTensorOp {}
unsafe impl Sync for PTXTensorOp {}

#[link(name = "ptx_core")]
extern "C" {
    // ========================================================================
    // Generic Dispatch
    // ========================================================================
    pub fn ptx_tensor_dispatch(op: *const PTXTensorOp) -> c_int;
    pub fn ptx_tensor_dispatch_sync(op: *const PTXTensorOp) -> c_int;

    // ========================================================================
    // F32 Binary Operations
    // ========================================================================
    pub fn ptx_tensor_add_f32(a: *mut f32, b: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_sub_f32(a: *mut f32, b: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_mul_f32(a: *mut f32, b: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_div_f32(a: *mut f32, b: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_max_f32(a: *mut f32, b: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_min_f32(a: *mut f32, b: *mut f32, out: *mut f32, n: usize, stream: CudaStream);

    // Scalar broadcast
    pub fn ptx_tensor_add_scalar_f32(a: *mut f32, scalar: f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_sub_scalar_f32(a: *mut f32, scalar: f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_mul_scalar_f32(a: *mut f32, scalar: f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_div_scalar_f32(a: *mut f32, scalar: f32, out: *mut f32, n: usize, stream: CudaStream);

    // ========================================================================
    // F32 Unary Operations
    // ========================================================================
    pub fn ptx_tensor_neg_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_abs_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_exp_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_log_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_sqrt_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_rsqrt_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_sin_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_cos_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_tanh_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_ceil_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_floor_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_round_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_sqr_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_recip_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);

    // ========================================================================
    // F32 Activations
    // ========================================================================
    pub fn ptx_tensor_relu_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_relu6_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_leaky_relu_f32(inp: *mut f32, out: *mut f32, n: usize, alpha: f32, stream: CudaStream);
    pub fn ptx_tensor_elu_f32(inp: *mut f32, out: *mut f32, n: usize, alpha: f32, stream: CudaStream);
    pub fn ptx_tensor_selu_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_gelu_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_sigmoid_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_silu_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_softplus_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_mish_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);

    // ========================================================================
    // F32 Reductions
    // ========================================================================
    pub fn ptx_tensor_reduce_sum_f32(inp: *mut f32, out: *mut f32, outer: usize, reduce: usize, inner: usize, stream: CudaStream);
    pub fn ptx_tensor_reduce_mean_f32(inp: *mut f32, out: *mut f32, outer: usize, reduce: usize, inner: usize, stream: CudaStream);
    pub fn ptx_tensor_reduce_max_f32(inp: *mut f32, out: *mut f32, outer: usize, reduce: usize, inner: usize, stream: CudaStream);
    pub fn ptx_tensor_reduce_min_f32(inp: *mut f32, out: *mut f32, outer: usize, reduce: usize, inner: usize, stream: CudaStream);

    // ========================================================================
    // F32 Softmax
    // ========================================================================
    pub fn ptx_tensor_softmax_f32(inp: *mut f32, out: *mut f32, batch: usize, dim: usize, stream: CudaStream);
    pub fn ptx_tensor_log_softmax_f32(inp: *mut f32, out: *mut f32, batch: usize, dim: usize, stream: CudaStream);

    // ========================================================================
    // F32 Affine/Transform
    // ========================================================================
    pub fn ptx_tensor_affine_f32(inp: *mut f32, out: *mut f32, n: usize, mul: f32, add: f32, stream: CudaStream);
    pub fn ptx_tensor_clamp_f32(inp: *mut f32, out: *mut f32, n: usize, min_val: f32, max_val: f32, stream: CudaStream);
    pub fn ptx_tensor_powf_f32(inp: *mut f32, out: *mut f32, n: usize, exp: f32, stream: CudaStream);

    // ========================================================================
    // F32 Comparison
    // ========================================================================
    pub fn ptx_tensor_cmp_eq_f32(a: *mut f32, b: *mut f32, out: *mut u8, n: usize, stream: CudaStream);
    pub fn ptx_tensor_cmp_lt_f32(a: *mut f32, b: *mut f32, out: *mut u8, n: usize, stream: CudaStream);
    pub fn ptx_tensor_cmp_le_f32(a: *mut f32, b: *mut f32, out: *mut u8, n: usize, stream: CudaStream);
    pub fn ptx_tensor_cmp_gt_f32(a: *mut f32, b: *mut f32, out: *mut u8, n: usize, stream: CudaStream);
    pub fn ptx_tensor_cmp_ge_f32(a: *mut f32, b: *mut f32, out: *mut u8, n: usize, stream: CudaStream);

    // ========================================================================
    // F32 Where/Select
    // ========================================================================
    pub fn ptx_tensor_where_f32(cond: *mut u8, t: *mut f32, f: *mut f32, out: *mut f32, n: usize, stream: CudaStream);

    // ========================================================================
    // F32 Copy/Fill
    // ========================================================================
    pub fn ptx_tensor_copy_f32(inp: *mut f32, out: *mut f32, n: usize, stream: CudaStream);
    pub fn ptx_tensor_fill_f32(out: *mut f32, n: usize, value: f32, stream: CudaStream);

    // ========================================================================
    // F64 Operations
    // ========================================================================
    pub fn ptx_tensor_add_f64(a: *mut f64, b: *mut f64, out: *mut f64, n: usize, stream: CudaStream);
    pub fn ptx_tensor_sub_f64(a: *mut f64, b: *mut f64, out: *mut f64, n: usize, stream: CudaStream);
    pub fn ptx_tensor_mul_f64(a: *mut f64, b: *mut f64, out: *mut f64, n: usize, stream: CudaStream);
    pub fn ptx_tensor_div_f64(a: *mut f64, b: *mut f64, out: *mut f64, n: usize, stream: CudaStream);

    pub fn ptx_tensor_neg_f64(inp: *mut f64, out: *mut f64, n: usize, stream: CudaStream);
    pub fn ptx_tensor_exp_f64(inp: *mut f64, out: *mut f64, n: usize, stream: CudaStream);
    pub fn ptx_tensor_log_f64(inp: *mut f64, out: *mut f64, n: usize, stream: CudaStream);
    pub fn ptx_tensor_sqrt_f64(inp: *mut f64, out: *mut f64, n: usize, stream: CudaStream);
    pub fn ptx_tensor_tanh_f64(inp: *mut f64, out: *mut f64, n: usize, stream: CudaStream);

    pub fn ptx_tensor_relu_f64(inp: *mut f64, out: *mut f64, n: usize, stream: CudaStream);
    pub fn ptx_tensor_gelu_f64(inp: *mut f64, out: *mut f64, n: usize, stream: CudaStream);
    pub fn ptx_tensor_sigmoid_f64(inp: *mut f64, out: *mut f64, n: usize, stream: CudaStream);

    pub fn ptx_tensor_reduce_sum_f64(inp: *mut f64, out: *mut f64, outer: usize, reduce: usize, inner: usize, stream: CudaStream);
    pub fn ptx_tensor_reduce_max_f64(inp: *mut f64, out: *mut f64, outer: usize, reduce: usize, inner: usize, stream: CudaStream);
    pub fn ptx_tensor_reduce_min_f64(inp: *mut f64, out: *mut f64, outer: usize, reduce: usize, inner: usize, stream: CudaStream);

    pub fn ptx_tensor_softmax_f64(inp: *mut f64, out: *mut f64, batch: usize, dim: usize, stream: CudaStream);
    pub fn ptx_tensor_affine_f64(inp: *mut f64, out: *mut f64, n: usize, mul: f64, add: f64, stream: CudaStream);

    // ========================================================================
    // F16 Operations (half precision)
    // ========================================================================
    pub fn ptx_tensor_add_f16(a: *mut c_void, b: *mut c_void, out: *mut c_void, n: usize, stream: CudaStream);
    pub fn ptx_tensor_sub_f16(a: *mut c_void, b: *mut c_void, out: *mut c_void, n: usize, stream: CudaStream);
    pub fn ptx_tensor_mul_f16(a: *mut c_void, b: *mut c_void, out: *mut c_void, n: usize, stream: CudaStream);
    pub fn ptx_tensor_div_f16(a: *mut c_void, b: *mut c_void, out: *mut c_void, n: usize, stream: CudaStream);

    pub fn ptx_tensor_relu_f16(inp: *mut c_void, out: *mut c_void, n: usize, stream: CudaStream);
    pub fn ptx_tensor_gelu_f16(inp: *mut c_void, out: *mut c_void, n: usize, stream: CudaStream);
    pub fn ptx_tensor_sigmoid_f16(inp: *mut c_void, out: *mut c_void, n: usize, stream: CudaStream);

    pub fn ptx_tensor_softmax_f16(inp: *mut c_void, out: *mut c_void, batch: usize, dim: usize, stream: CudaStream);

    pub fn ptx_tensor_cast_f32_to_f16(inp: *mut f32, out: *mut c_void, n: usize, stream: CudaStream);
    pub fn ptx_tensor_cast_f16_to_f32(inp: *mut c_void, out: *mut f32, n: usize, stream: CudaStream);

    // ========================================================================
    // CUDA Graph Integration
    // ========================================================================
    pub fn ptx_tensor_graph_begin_capture(stream: CudaStream) -> c_int;
    pub fn ptx_tensor_graph_end_capture(stream: CudaStream, graph_out: *mut *mut c_void) -> c_int;
    pub fn ptx_tensor_graph_instantiate(graph: *mut c_void, exec_out: *mut *mut c_void) -> c_int;
    pub fn ptx_tensor_graph_launch(exec: *mut c_void, stream: CudaStream) -> c_int;
}

/// Helper to convert candle DType to PTX DType
impl PTXDType {
    pub fn from_candle(dtype: candle_core::DType) -> Option<Self> {
        match dtype {
            candle_core::DType::F32 => Some(PTXDType::F32),
            candle_core::DType::F64 => Some(PTXDType::F64),
            candle_core::DType::F16 => Some(PTXDType::F16),
            candle_core::DType::BF16 => Some(PTXDType::BF16),
            candle_core::DType::I64 => Some(PTXDType::I64),
            candle_core::DType::U32 => Some(PTXDType::U32),
            candle_core::DType::U8 => Some(PTXDType::U8),
        }
    }
}
