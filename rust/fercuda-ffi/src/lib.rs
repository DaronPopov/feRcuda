pub use fercuda_math as math;
pub use fercuda_ml_lite as ml;
pub use fercuda_vision_lite as vision;
pub mod adapter_backend;
#[cfg(feature = "candle")]
pub mod candle_adapter;
#[cfg(feature = "cudarc")]
pub mod cudarc_adapter;
pub mod ml_adapter;
pub use adapter_backend::{AdapterExecutionBackend, BACKEND_SLOT_IN_PATTERN};
#[cfg(feature = "candle")]
pub use candle_adapter::{
    download_tensor_f32, upload_tensor_f32, validate_layer_norm_dims, validate_matmul_dims,
    CandleAdapterError, CandleBuffer, CandleSessionAdapter, OpRunConfig,
};
#[cfg(feature = "cudarc")]
pub use cudarc_adapter::{
    validate_layer_norm_dims_cudarc, validate_matmul_dims_cudarc, CudarcAdapterError,
    CudarcOpRunConfig, CudarcSessionAdapter, CudarcTensor,
};
pub use ml_adapter::{
    decode_bf16_le_bytes, decode_f16_le_bytes, decode_f32_le_bytes, dequantize_q4_affine_packed,
    dequantize_q4_symmetric_packed, dequantize_q8_affine_u8, dequantize_q8_symmetric_i8,
    load_safetensors_f32_into_session, load_safetensors_into_session,
    upload_quantized_into_session, AdapterError, LoadedModel, LoadedTensor, QuantMode, QuantParams,
};

use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_int, c_uchar, c_void};
#[repr(C)]
struct FerSessionOpaque {
    _private: [u8; 0],
}

#[repr(C)]
pub struct FerJitProgramOpaque {
    _private: [u8; 0],
}

#[repr(C)]
pub struct FerJitKernelOpaque {
    _private: [u8; 0],
}

pub type BufferId = u64;
pub type JobId = u64;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

pub type JitProgram = *mut FerJitProgramOpaque;
pub type JitKernel = *mut FerJitKernelOpaque;

pub const JIT_WILDCARD_U32: u32 = 0xFFFF_FFFF;
pub const JIT_WILDCARD_U64: u64 = 0xFFFF_FFFF_FFFF_FFFF;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum JitBackend {
    Nvrtc = 0,
    Auto = 0xFFFF_FFFF,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum JitMode {
    Permissive = 0,
    Strict = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum JitSourceKind {
    Cuda = 0,
    Ptx = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum JitArgKind {
    Buffer = 0,
    ScalarI32 = 1,
    ScalarU32 = 2,
    ScalarI64 = 3,
    ScalarU64 = 4,
    ScalarF32 = 5,
    ScalarF64 = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum JitAccess {
    Read = 0,
    Write = 1,
    ReadWrite = 2,
}

#[derive(Debug, Clone, Copy)]
pub struct JitSource<'a> {
    pub kind: JitSourceKind,
    pub code: &'a str,
}

#[derive(Debug, Clone, Default)]
pub struct JitOptions {
    pub backend: Option<JitBackend>,
    pub mode: Option<JitMode>,
    pub arch: Option<String>,
    pub extra_nvrtc_opts: Option<String>,
    pub cache_dir: Option<String>,
    pub enable_disk_cache: bool,
}

#[derive(Debug, Clone)]
pub struct JitCompileResult {
    pub cache_hit: bool,
    pub backend_name: String,
    pub log: String,
}

#[derive(Debug, Clone)]
pub struct JitArgDesc {
    pub kind: JitArgKind,
    pub access: JitAccess,
    pub name: Option<String>,
    pub expected_dtype: u32,
    pub expected_rank: u32,
    pub expected_bytes: u64,
    pub expected_dims: [u32; 4],
}

#[derive(Debug, Clone, Copy)]
pub struct JitLaunchCfg {
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub block_x: u32,
    pub block_y: u32,
    pub block_z: u32,
    pub shared_mem_bytes: u32,
    pub memory_regime: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum JitArgValue {
    Buffer(BufferId),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
}

#[derive(Debug, Clone, Copy)]
pub struct JitStats {
    pub compile_count: u64,
    pub cache_hit_count: u64,
    pub launch_count: u64,
    pub compile_time_us: u64,
    pub launch_time_us: u64,
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
    dims: [u32; 4],
    immutable: u8,
    tag: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct BufferDesc {
    pub dtype: BufferDType,
    pub rank: u32,
    pub dims: [u32; 4],
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
        Self {
            dtype: BufferDType::F32,
            rank: 1,
            dims: [n, 0, 0, 0],
            immutable: false,
            tag: 0,
        }
    }

    pub fn f32_2d(m: u32, n: u32) -> Self {
        Self {
            dtype: BufferDType::F32,
            rank: 2,
            dims: [m, n, 0, 0],
            immutable: false,
            tag: 0,
        }
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
        (0..self.rank as usize)
            .map(|i| self.dims[i] as usize)
            .product::<usize>()
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
        Self {
            a,
            b,
            out,
            memory_regime: MemoryRegime::Auto as u32,
        }
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
        Self {
            x,
            out,
            eps,
            memory_regime: MemoryRegime::Auto as u32,
        }
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

#[derive(Debug, Clone, Copy)]
pub struct AffineF32Request {
    pub input: BufferId,
    pub output: BufferId,
    pub n: u32,
    pub alpha: f32,
    pub beta: f32,
    pub fusion_mask: u32,
    pub caps_mask: u32,
    pub memory_regime: u32,
}

#[repr(C)]
struct FerJitSource {
    kind: u32,
    code: *const c_char,
    code_len: usize,
}

#[repr(C)]
struct FerJitOptions {
    backend: u32,
    mode: u32,
    arch: *const c_char,
    extra_nvrtc_opts: *const c_char,
    cache_dir: *const c_char,
    enable_disk_cache: u8,
}

#[repr(C)]
struct FerJitCompileResult {
    cache_hit: u8,
    backend_name: *const c_char,
    log: *const c_char,
}

#[repr(C)]
struct FerJitArgDesc {
    kind: u32,
    access: u32,
    name: *const c_char,
    expected_dtype: u32,
    expected_rank: u32,
    expected_bytes: u64,
    expected_dims: [u32; 4],
}

#[repr(C)]
struct FerJitKernelSig {
    args: *const FerJitArgDesc,
    arg_count: usize,
}

#[repr(C)]
struct FerJitLaunchCfg {
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_mem_bytes: u32,
    memory_regime: u32,
}

#[repr(C)]
union FerJitArgPayload {
    buffer_id: u64,
    i32_: i32,
    u32_: u32,
    i64_: i64,
    u64_: u64,
    f32_: f32,
    f64_: f64,
}

#[repr(C)]
struct FerJitArgValue {
    kind: u32,
    as_: FerJitArgPayload,
}

#[repr(C)]
struct FerJitArgPack {
    args: *const FerJitArgValue,
    arg_count: usize,
}

#[repr(C)]
struct FerJitStats {
    compile_count: u64,
    cache_hit_count: u64,
    launch_count: u64,
    compile_time_us: u64,
    launch_time_us: u64,
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
        return Ok(());
    }
    let code = match s.code {
        1 => StatusCode::InvalidArgument,
        2 => StatusCode::NotFound,
        3 => StatusCode::InternalError,
        _ => StatusCode::InternalError,
    };
    let message = if s.message.is_null() {
        "unknown error".to_string()
    } else {
        unsafe { CStr::from_ptr(s.message) }
            .to_string_lossy()
            .into_owned()
    };
    Err(Error { code, message })
}

fn c_ptr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

extern "C" {
    fn fer_session_create(
        device: c_int,
        cfg: *const PoolConfig,
        out_session: *mut *mut FerSessionOpaque,
    ) -> FerStatus;
    fn fer_session_destroy(session: *mut FerSessionOpaque) -> FerStatus;
    fn fer_alloc_buffer(
        session: *mut FerSessionOpaque,
        desc: *const FerBufferDesc,
        out_buffer_id: *mut u64,
    ) -> FerStatus;
    fn fer_free_buffer(session: *mut FerSessionOpaque, buffer_id: u64) -> FerStatus;
    fn fer_upload_f32(
        session: *mut FerSessionOpaque,
        buffer_id: u64,
        host: *const f32,
        count: usize,
    ) -> FerStatus;
    fn fer_download_f32(
        session: *mut FerSessionOpaque,
        buffer_id: u64,
        host: *mut f32,
        count: usize,
    ) -> FerStatus;
    fn fer_upload_bytes(
        session: *mut FerSessionOpaque,
        buffer_id: u64,
        host: *const c_void,
        bytes: usize,
    ) -> FerStatus;
    fn fer_download_bytes(
        session: *mut FerSessionOpaque,
        buffer_id: u64,
        host: *mut c_void,
        bytes: usize,
    ) -> FerStatus;
    fn fer_submit_matmul(
        session: *mut FerSessionOpaque,
        req: *const FerMatmulRequest,
        out_job_id: *mut u64,
    ) -> FerStatus;
    fn fer_submit_layer_norm(
        session: *mut FerSessionOpaque,
        req: *const FerLayerNormRequest,
        out_job_id: *mut u64,
    ) -> FerStatus;
    fn fer_job_status(
        session: *mut FerSessionOpaque,
        job_id: u64,
        out_done: *mut c_uchar,
    ) -> FerStatus;
    fn fer_job_wait(session: *mut FerSessionOpaque, job_id: u64) -> FerStatus;
    fn fer_jit_compile(
        session: *mut FerSessionOpaque,
        source: *const FerJitSource,
        options: *const FerJitOptions,
        out_program: *mut JitProgram,
        out_result: *mut FerJitCompileResult,
    ) -> FerStatus;
    fn fer_jit_release_program(session: *mut FerSessionOpaque, program: JitProgram) -> FerStatus;
    fn fer_jit_get_kernel(
        session: *mut FerSessionOpaque,
        program: JitProgram,
        kernel_name: *const c_char,
        signature: *const FerJitKernelSig,
        out_kernel: *mut JitKernel,
    ) -> FerStatus;
    fn fer_jit_release_kernel(session: *mut FerSessionOpaque, kernel: JitKernel) -> FerStatus;
    fn fer_jit_launch(
        session: *mut FerSessionOpaque,
        kernel: JitKernel,
        cfg: *const FerJitLaunchCfg,
        args: *const FerJitArgPack,
        out_job_id: *mut u64,
    ) -> FerStatus;
    fn fer_jit_get_stats(session: *mut FerSessionOpaque, out_stats: *mut FerJitStats) -> FerStatus;
    fn fer_tensor_run_affine_f32(
        session: *mut FerSessionOpaque,
        input: u64,
        output: u64,
        n: u32,
        alpha: f32,
        beta: f32,
        fusion_mask: u32,
        caps_mask: u32,
        memory_regime: u32,
        out_job_id: *mut u64,
    ) -> FerStatus;
}

pub struct Session {
    raw: *mut FerSessionOpaque,
}

impl Session {
    pub fn new(device: i32, cfg: Option<PoolConfig>) -> Result<Self, Error> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            match cfg {
                Some(c) => fer_session_create(device as c_int, &c as *const PoolConfig, &mut raw),
                None => fer_session_create(device as c_int, std::ptr::null(), &mut raw),
            }
        };
        status_to_result(status)?;
        Ok(Self { raw })
    }

    pub fn alloc_buffer(&self, desc: BufferDesc) -> Result<BufferId, Error> {
        let mut id = 0u64;
        let cdesc = desc.to_c();
        let status = unsafe { fer_alloc_buffer(self.raw, &cdesc as *const FerBufferDesc, &mut id) };
        status_to_result(status)?;
        Ok(id)
    }

    pub fn free_buffer(&self, id: BufferId) -> Result<(), Error> {
        let status = unsafe { fer_free_buffer(self.raw, id) };
        status_to_result(status)
    }

    pub fn upload_f32(&self, id: BufferId, host: &[f32]) -> Result<(), Error> {
        let status = unsafe { fer_upload_f32(self.raw, id, host.as_ptr(), host.len()) };
        status_to_result(status)
    }

    pub fn download_f32(&self, id: BufferId, host: &mut [f32]) -> Result<(), Error> {
        let status = unsafe { fer_download_f32(self.raw, id, host.as_mut_ptr(), host.len()) };
        status_to_result(status)
    }

    pub fn upload_bytes(&self, id: BufferId, host: &[u8]) -> Result<(), Error> {
        let status = unsafe {
            fer_upload_bytes(
                self.raw,
                id,
                host.as_ptr() as *const std::ffi::c_void,
                host.len(),
            )
        };
        status_to_result(status)
    }

    pub fn download_bytes(&self, id: BufferId, host: &mut [u8]) -> Result<(), Error> {
        let status = unsafe {
            fer_download_bytes(self.raw, id, host.as_mut_ptr() as *mut c_void, host.len())
        };
        status_to_result(status)
    }

    pub fn jit_compile(
        &self,
        source: JitSource<'_>,
        options: &JitOptions,
    ) -> Result<(JitProgram, JitCompileResult), Error> {
        let arch = options
            .arch
            .as_deref()
            .map(CString::new)
            .transpose()
            .map_err(|e| Error {
                code: StatusCode::InvalidArgument,
                message: format!("invalid jit arch string: {e}"),
            })?;
        let extra_nvrtc_opts = options
            .extra_nvrtc_opts
            .as_deref()
            .map(CString::new)
            .transpose()
            .map_err(|e| Error {
                code: StatusCode::InvalidArgument,
                message: format!("invalid jit extra_nvrtc_opts string: {e}"),
            })?;
        let cache_dir = options
            .cache_dir
            .as_deref()
            .map(CString::new)
            .transpose()
            .map_err(|e| Error {
                code: StatusCode::InvalidArgument,
                message: format!("invalid jit cache_dir string: {e}"),
            })?;
        let csource = FerJitSource {
            kind: source.kind as u32,
            code: source.code.as_ptr() as *const c_char,
            code_len: source.code.len(),
        };
        let coptions = FerJitOptions {
            backend: options.backend.unwrap_or(JitBackend::Auto) as u32,
            mode: options.mode.unwrap_or(JitMode::Strict) as u32,
            arch: arch.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            extra_nvrtc_opts: extra_nvrtc_opts
                .as_ref()
                .map_or(std::ptr::null(), |s| s.as_ptr()),
            cache_dir: cache_dir.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            enable_disk_cache: u8::from(options.enable_disk_cache),
        };
        let mut program: JitProgram = std::ptr::null_mut();
        let mut result = FerJitCompileResult {
            cache_hit: 0,
            backend_name: std::ptr::null(),
            log: std::ptr::null(),
        };
        let status =
            unsafe { fer_jit_compile(self.raw, &csource, &coptions, &mut program, &mut result) };
        status_to_result(status)?;
        Ok((
            program,
            JitCompileResult {
                cache_hit: result.cache_hit != 0,
                backend_name: c_ptr_to_string(result.backend_name),
                log: c_ptr_to_string(result.log),
            },
        ))
    }

    pub fn jit_release_program(&self, program: JitProgram) -> Result<(), Error> {
        let status = unsafe { fer_jit_release_program(self.raw, program) };
        status_to_result(status)
    }

    pub fn jit_get_kernel(
        &self,
        program: JitProgram,
        kernel_name: &str,
        descs: &[JitArgDesc],
    ) -> Result<JitKernel, Error> {
        let kernel_name = CString::new(kernel_name).map_err(|e| Error {
            code: StatusCode::InvalidArgument,
            message: format!("invalid kernel_name: {e}"),
        })?;
        let names = descs
            .iter()
            .map(|desc| desc.name.as_deref().map(CString::new).transpose())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| Error {
                code: StatusCode::InvalidArgument,
                message: format!("invalid jit arg name: {e}"),
            })?;
        let cdescs = descs
            .iter()
            .zip(names.iter())
            .map(|(desc, name)| FerJitArgDesc {
                kind: desc.kind as u32,
                access: desc.access as u32,
                name: name.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                expected_dtype: desc.expected_dtype,
                expected_rank: desc.expected_rank,
                expected_bytes: desc.expected_bytes,
                expected_dims: desc.expected_dims,
            })
            .collect::<Vec<_>>();
        let sig = if cdescs.is_empty() {
            None
        } else {
            Some(FerJitKernelSig {
                args: cdescs.as_ptr(),
                arg_count: cdescs.len(),
            })
        };
        let mut kernel: JitKernel = std::ptr::null_mut();
        let status = unsafe {
            fer_jit_get_kernel(
                self.raw,
                program,
                kernel_name.as_ptr(),
                sig.as_ref()
                    .map_or(std::ptr::null(), |s| s as *const FerJitKernelSig),
                &mut kernel,
            )
        };
        status_to_result(status)?;
        Ok(kernel)
    }

    pub fn jit_release_kernel(&self, kernel: JitKernel) -> Result<(), Error> {
        let status = unsafe { fer_jit_release_kernel(self.raw, kernel) };
        status_to_result(status)
    }

    pub fn jit_launch(
        &self,
        kernel: JitKernel,
        cfg: JitLaunchCfg,
        args: &[JitArgValue],
    ) -> Result<JobId, Error> {
        let carg_values = args
            .iter()
            .map(|arg| match *arg {
                JitArgValue::Buffer(buffer_id) => FerJitArgValue {
                    kind: JitArgKind::Buffer as u32,
                    as_: FerJitArgPayload { buffer_id },
                },
                JitArgValue::I32(v) => FerJitArgValue {
                    kind: JitArgKind::ScalarI32 as u32,
                    as_: FerJitArgPayload { i32_: v },
                },
                JitArgValue::U32(v) => FerJitArgValue {
                    kind: JitArgKind::ScalarU32 as u32,
                    as_: FerJitArgPayload { u32_: v },
                },
                JitArgValue::I64(v) => FerJitArgValue {
                    kind: JitArgKind::ScalarI64 as u32,
                    as_: FerJitArgPayload { i64_: v },
                },
                JitArgValue::U64(v) => FerJitArgValue {
                    kind: JitArgKind::ScalarU64 as u32,
                    as_: FerJitArgPayload { u64_: v },
                },
                JitArgValue::F32(v) => FerJitArgValue {
                    kind: JitArgKind::ScalarF32 as u32,
                    as_: FerJitArgPayload { f32_: v },
                },
                JitArgValue::F64(v) => FerJitArgValue {
                    kind: JitArgKind::ScalarF64 as u32,
                    as_: FerJitArgPayload { f64_: v },
                },
            })
            .collect::<Vec<_>>();
        let cpack = FerJitArgPack {
            args: carg_values.as_ptr(),
            arg_count: carg_values.len(),
        };
        let ccfg = FerJitLaunchCfg {
            grid_x: cfg.grid_x,
            grid_y: cfg.grid_y,
            grid_z: cfg.grid_z,
            block_x: cfg.block_x,
            block_y: cfg.block_y,
            block_z: cfg.block_z,
            shared_mem_bytes: cfg.shared_mem_bytes,
            memory_regime: cfg.memory_regime,
        };
        let mut job_id = 0u64;
        let status = unsafe { fer_jit_launch(self.raw, kernel, &ccfg, &cpack, &mut job_id) };
        status_to_result(status)?;
        Ok(job_id)
    }

    pub fn jit_get_stats(&self) -> Result<JitStats, Error> {
        let mut stats = FerJitStats {
            compile_count: 0,
            cache_hit_count: 0,
            launch_count: 0,
            compile_time_us: 0,
            launch_time_us: 0,
        };
        let status = unsafe { fer_jit_get_stats(self.raw, &mut stats) };
        status_to_result(status)?;
        Ok(JitStats {
            compile_count: stats.compile_count,
            cache_hit_count: stats.cache_hit_count,
            launch_count: stats.launch_count,
            compile_time_us: stats.compile_time_us,
            launch_time_us: stats.launch_time_us,
        })
    }

    pub fn run_affine_f32(&self, req: AffineF32Request) -> Result<JobId, Error> {
        let mut job_id = 0u64;
        let status = unsafe {
            fer_tensor_run_affine_f32(
                self.raw,
                req.input,
                req.output,
                req.n,
                req.alpha,
                req.beta,
                req.fusion_mask,
                req.caps_mask,
                req.memory_regime,
                &mut job_id,
            )
        };
        status_to_result(status)?;
        Ok(job_id)
    }

    pub fn submit_matmul(&self, req: MatmulRequest) -> Result<JobId, Error> {
        let creq = FerMatmulRequest {
            a: req.a,
            b: req.b,
            out: req.out,
            memory_regime: req.memory_regime,
        };
        let mut id = 0u64;
        let status =
            unsafe { fer_submit_matmul(self.raw, &creq as *const FerMatmulRequest, &mut id) };
        status_to_result(status)?;
        Ok(id)
    }

    pub fn submit_layer_norm(&self, req: LayerNormRequest) -> Result<JobId, Error> {
        let creq = FerLayerNormRequest {
            x: req.x,
            out: req.out,
            eps: req.eps,
            memory_regime: req.memory_regime,
        };
        let mut id = 0u64;
        let status = unsafe {
            fer_submit_layer_norm(self.raw, &creq as *const FerLayerNormRequest, &mut id)
        };
        status_to_result(status)?;
        Ok(id)
    }

    pub fn job_status(&self, job_id: JobId) -> Result<bool, Error> {
        let mut done: c_uchar = 0;
        let status = unsafe { fer_job_status(self.raw, job_id, &mut done as *mut c_uchar) };
        status_to_result(status)?;
        Ok(done != 0)
    }

    pub fn job_wait(&self, job_id: JobId) -> Result<(), Error> {
        let status = unsafe { fer_job_wait(self.raw, job_id) };
        status_to_result(status)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { fer_session_destroy(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}
