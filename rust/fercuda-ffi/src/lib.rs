pub use fercuda_math as math;
pub use fercuda_ml_lite as ml;
pub use fercuda_vision_lite as vision;
pub mod adapter_backend;
pub mod ml_adapter;
#[cfg(feature = "candle")]
pub mod candle_adapter;
#[cfg(feature = "cudarc")]
pub mod cudarc_adapter;
#[cfg(feature = "candle")]
pub use candle_adapter::{
    download_tensor_f32, upload_tensor_f32, CandleAdapterError, CandleBuffer, CandleSessionAdapter,
    OpRunConfig, validate_layer_norm_dims, validate_matmul_dims,
};
pub use adapter_backend::{AdapterExecutionBackend, BACKEND_SLOT_IN_PATTERN};
#[cfg(feature = "cudarc")]
pub use cudarc_adapter::{
    CudarcAdapterError, CudarcOpRunConfig, CudarcSessionAdapter, CudarcTensor,
    validate_layer_norm_dims_cudarc, validate_matmul_dims_cudarc,
};
pub use ml_adapter::{
    decode_bf16_le_bytes, decode_f16_le_bytes, decode_f32_le_bytes,
    dequantize_q4_affine_packed, dequantize_q4_symmetric_packed, dequantize_q8_affine_u8,
    dequantize_q8_symmetric_i8, load_safetensors_f32_into_session, load_safetensors_into_session,
    upload_quantized_into_session, AdapterError, LoadedModel, LoadedTensor, QuantMode, QuantParams,
};

use std::ffi::CStr ; use std::fmt ; use std::os::raw::{c_char, c_int, c_uchar} ; #[repr(C)]
struct FerSessionOpaque {
    _private: [u8 ; 0],
}

pub type BufferId = u64 ; pub type JobId = u64 ; #[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum StatusCode {
    Ok = 0,
    InvalidArgument = 1,
    NotFound = 2,
    InternalError = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MemoryRegime {
    CustomPool = 0,
    CudaMalloc = 1,
    CudaManaged = 2,
    Auto = 0xFFFF_FFFF,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FerStatus {
    code: i32,
    message: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PoolConfig {
    pub mutable_bytes: u64,
    pub immutable_bytes: u64,
    pub cuda_reserve: u64,
    pub verbose: u8,
    pub memory_regime: u32,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            mutable_bytes: 512u64 << 20,
            immutable_bytes: 2u64 << 30,
            cuda_reserve: 256u64 << 20,
            verbose: 0,
            memory_regime: MemoryRegime::CustomPool as u32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BufferDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I8 = 3,
    U8 = 4,
    I16 = 5,
    U16 = 6,
    I32 = 7,
    U32 = 8,
    I64 = 9,
    U64 = 10,
    F64 = 11,
}

impl BufferDType {
    pub fn elem_size(self) -> usize {
        match self {
            BufferDType::F32 => 4,
            BufferDType::F16 => 2,
            BufferDType::BF16 => 2,
            BufferDType::I8 => 1,
            BufferDType::U8 => 1,
            BufferDType::I16 => 2,
            BufferDType::U16 => 2,
            BufferDType::I32 => 4,
            BufferDType::U32 => 4,
            BufferDType::I64 => 8,
            BufferDType::U64 => 8,
            BufferDType::F64 => 8,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FerBufferDesc {
    dtype: u32,
    rank: u32,
    dims: [u32 ; 4],
    immutable: u8,
    tag: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct BufferDesc {
    pub dtype: BufferDType,
    pub rank: u32,
    pub dims: [u32 ; 4],
    pub immutable: bool,
    pub tag: u32,
}

impl BufferDesc {
    pub fn new(dtype: BufferDType, rank: u32, dims: [u32; 4], immutable: bool, tag: u32) -> Self {
        Self {
            dtype,
            rank,
            dims,
            immutable,
            tag,
        }
    }

    pub fn f32_1d(n: u32) -> Self {
        Self { dtype: BufferDType::F32, rank: 1, dims: [n, 0, 0, 0], immutable: false, tag: 0 }
    }

    pub fn f32_2d(m: u32, n: u32) -> Self {
        Self { dtype: BufferDType::F32, rank: 2, dims: [m, n, 0, 0], immutable: false, tag: 0 }
    }

    fn to_c(self) -> FerBufferDesc {
        FerBufferDesc {
            dtype: self.dtype as u32,
            rank: self.rank,
            dims: self.dims,
            immutable: if self.immutable { 1 } else { 0 },
            tag: self.tag,
        }
    }

    pub fn elem_count(&self) -> usize {
        (0..self.rank as usize).map(|i| self.dims[i] as usize).product::<usize>()
    }

    pub fn byte_len(&self) -> usize {
        self.elem_count() * self.dtype.elem_size()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatmulRequest {
    pub a: BufferId,
    pub b: BufferId,
    pub out: BufferId,
    pub memory_regime: u32,
}

impl MatmulRequest {
    pub fn auto(a: BufferId, b: BufferId, out: BufferId) -> Self {
        Self { a, b, out, memory_regime: MemoryRegime::Auto as u32 }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FerMatmulRequest {
    a: u64,
    b: u64,
    out: u64,
    memory_regime: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct LayerNormRequest {
    pub x: BufferId,
    pub out: BufferId,
    pub eps: f32,
    pub memory_regime: u32,
}

impl LayerNormRequest {
    pub fn auto(x: BufferId, out: BufferId, eps: f32) -> Self {
        Self { x, out, eps, memory_regime: MemoryRegime::Auto as u32 }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FerLayerNormRequest {
    x: u64,
    out: u64,
    eps: f32,
    memory_regime: u32,
}

#[derive(Debug, Clone)]
pub struct Error {
    pub code: StatusCode,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

fn status_to_result(s: FerStatus) -> Result<(), Error> {
    if s.code == StatusCode::Ok as i32 {
        return Ok(()) ; }
    let code = match s.code {
        1 => StatusCode::InvalidArgument,
        2 => StatusCode::NotFound,
        3 => StatusCode::InternalError,
        _ => StatusCode::InternalError,
    } ; let message = if s.message.is_null() {
        "unknown error".to_string()
    } else {
        unsafe { CStr::from_ptr(s.message) }.to_string_lossy().into_owned()
    } ; Err(Error { code, message })
}

extern "C" {
    fn fer_session_create(device: c_int, cfg: *const PoolConfig, out_session: *mut *mut FerSessionOpaque) -> FerStatus ; fn fer_session_destroy(session: *mut FerSessionOpaque) -> FerStatus ; fn fer_alloc_buffer(session: *mut FerSessionOpaque, desc: *const FerBufferDesc, out_buffer_id: *mut u64) -> FerStatus ; fn fer_free_buffer(session: *mut FerSessionOpaque, buffer_id: u64) -> FerStatus ; fn fer_upload_f32(session: *mut FerSessionOpaque, buffer_id: u64, host: *const f32, count: usize) -> FerStatus ; fn fer_download_f32(session: *mut FerSessionOpaque, buffer_id: u64, host: *mut f32, count: usize) -> FerStatus ; fn fer_upload_bytes(session: *mut FerSessionOpaque, buffer_id: u64, host: *const std::ffi::c_void, bytes: usize) -> FerStatus ; fn fer_download_bytes(session: *mut FerSessionOpaque, buffer_id: u64, host: *mut std::ffi::c_void, bytes: usize) -> FerStatus ; fn fer_submit_matmul(session: *mut FerSessionOpaque, req: *const FerMatmulRequest, out_job_id: *mut u64) -> FerStatus ; fn fer_submit_layer_norm(session: *mut FerSessionOpaque, req: *const FerLayerNormRequest, out_job_id: *mut u64) -> FerStatus ; fn fer_job_status(session: *mut FerSessionOpaque, job_id: u64, out_done: *mut c_uchar) -> FerStatus ; fn fer_job_wait(session: *mut FerSessionOpaque, job_id: u64) -> FerStatus ; }

pub struct Session {
    raw: *mut FerSessionOpaque,
}

impl Session {
    pub fn new(device: i32, cfg: Option<PoolConfig>) -> Result<Self, Error> {
        let mut raw = std::ptr::null_mut() ; let status = unsafe {
            match cfg {
                Some(c) => fer_session_create(device as c_int, &c as *const PoolConfig, &mut raw),
                None => fer_session_create(device as c_int, std::ptr::null(), &mut raw),
            }
        } ; status_to_result(status)? ; Ok(Self { raw })
    }

    pub fn alloc_buffer(&self, desc: BufferDesc) -> Result<BufferId, Error> {
        let mut id = 0u64 ; let cdesc = desc.to_c() ; let status = unsafe { fer_alloc_buffer(self.raw, &cdesc as *const FerBufferDesc, &mut id) } ; status_to_result(status)? ; Ok(id)
    }

    pub fn free_buffer(&self, id: BufferId) -> Result<(), Error> {
        let status = unsafe { fer_free_buffer(self.raw, id) } ; status_to_result(status)
    }

    pub fn upload_f32(&self, id: BufferId, host: &[f32]) -> Result<(), Error> {
        let status = unsafe { fer_upload_f32(self.raw, id, host.as_ptr(), host.len()) } ; status_to_result(status)
    }

    pub fn download_f32(&self, id: BufferId, host: &mut [f32]) -> Result<(), Error> {
        let status = unsafe { fer_download_f32(self.raw, id, host.as_mut_ptr(), host.len()) } ; status_to_result(status)
    }

    pub fn upload_bytes(&self, id: BufferId, host: &[u8]) -> Result<(), Error> {
        let status = unsafe {
            fer_upload_bytes(
                self.raw,
                id,
                host.as_ptr() as *const std::ffi::c_void,
                host.len(),
            )
        } ;
        status_to_result(status)
    }

    pub fn download_bytes(&self, id: BufferId, host: &mut [u8]) -> Result<(), Error> {
        let status = unsafe {
            fer_download_bytes(
                self.raw,
                id,
                host.as_mut_ptr() as *mut std::ffi::c_void,
                host.len(),
            )
        } ;
        status_to_result(status)
    }

    pub fn submit_matmul(&self, req: MatmulRequest) -> Result<JobId, Error> {
        let creq = FerMatmulRequest { a: req.a, b: req.b, out: req.out, memory_regime: req.memory_regime } ; let mut id = 0u64 ; let status = unsafe { fer_submit_matmul(self.raw, &creq as *const FerMatmulRequest, &mut id) } ; status_to_result(status)? ; Ok(id)
    }

    pub fn submit_layer_norm(&self, req: LayerNormRequest) -> Result<JobId, Error> {
        let creq = FerLayerNormRequest { x: req.x, out: req.out, eps: req.eps, memory_regime: req.memory_regime } ; let mut id = 0u64 ; let status = unsafe { fer_submit_layer_norm(self.raw, &creq as *const FerLayerNormRequest, &mut id) } ; status_to_result(status)? ; Ok(id)
    }

    pub fn job_status(&self, job_id: JobId) -> Result<bool, Error> {
        let mut done: c_uchar = 0 ; let status = unsafe { fer_job_status(self.raw, job_id, &mut done as *mut c_uchar) } ; status_to_result(status)? ; Ok(done != 0)
    }

    pub fn job_wait(&self, job_id: JobId) -> Result<(), Error> {
        let status = unsafe { fer_job_wait(self.raw, job_id) } ; status_to_result(status)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { fer_session_destroy(self.raw) } ; self.raw = std::ptr::null_mut() ; }
    }
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}
