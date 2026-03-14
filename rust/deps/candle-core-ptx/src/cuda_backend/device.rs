use crate::backend::BackendDevice;
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
pub use candle_kernels as kernels;
pub use cudarc;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    DeviceRepr, HostSlice, LaunchArgs, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};
use half::{bf16, f16};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

#[cfg(feature = "ptx-os")]
use cudarc::driver::{PtxContextExt, PtxStreamManager};

use super::{CudaError, CudaStorage, CudaStorageSlice, WrapErr};

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

struct CudaRng(cudarc::curand::CudaRng);
unsafe impl Send for CudaRng {}

pub(crate) trait KernelArgs<'a> {
    fn push(self, args: &mut LaunchArgs<'a>);
}

macro_rules! impl_kernel_args {
    ($($name:ident),+) => {
        impl<'a, $($name),+> KernelArgs<'a> for ($($name,)+)
        where
            $(LaunchArgs<'a>: PushKernelArg<$name>,)+
        {
            #[allow(non_snake_case)]
            fn push(self, args: &mut LaunchArgs<'a>) {
                let ($($name,)+) = self;
                $(args.arg($name);)+
            }
        }
    };
}

impl_kernel_args!(A);
impl_kernel_args!(A, B);
impl_kernel_args!(A, B, C);
impl_kernel_args!(A, B, C, D);
impl_kernel_args!(A, B, C, D, E);
impl_kernel_args!(A, B, C, D, E, F);
impl_kernel_args!(A, B, C, D, E, F, G);
impl_kernel_args!(A, B, C, D, E, F, G, H);
impl_kernel_args!(A, B, C, D, E, F, G, H, I);
impl_kernel_args!(A, B, C, D, E, F, G, H, I, J);
impl_kernel_args!(A, B, C, D, E, F, G, H, I, J, K);
impl_kernel_args!(A, B, C, D, E, F, G, H, I, J, K, L);

#[derive(Clone)]
pub struct CudaDevice {
    id: DeviceId,
    ctx: Arc<CudaContext>,
    default_stream: Arc<CudaStream>,
    #[cfg(feature = "ptx-os")]
    ptx_streams: Option<Arc<PtxStreamManager>>,
    modules: Arc<RwLock<HashMap<String, Arc<CudaModule>>>>,
    pub(crate) blas: Arc<cudarc::cublas::CudaBlas>,
    pub(crate) blas_stream: Arc<CudaStream>,
    curand: Arc<Mutex<CudaRng>>,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl CudaDevice {
    pub fn context(&self) -> Arc<CudaContext> {
        self.ctx.clone()
    }

    pub fn default_stream(&self) -> Arc<CudaStream> {
        self.default_stream.clone()
    }

    pub fn blas_stream(&self) -> Arc<CudaStream> {
        self.blas_stream.clone()
    }

    pub(crate) fn launch_stream(&self) -> Arc<CudaStream> {
        #[cfg(feature = "ptx-os")]
        if let Some(streams) = &self.ptx_streams {
            return streams.next_stream();
        }
        self.default_stream.clone()
    }

    pub(crate) fn launch_on_stream<'a, A>(
        &self,
        stream: &'a Arc<CudaStream>,
        func: &'a CudaFunction,
        cfg: LaunchConfig,
        args: A,
    ) -> std::result::Result<(), cudarc::driver::DriverError>
    where
        A: KernelArgs<'a>,
    {
        let mut builder = stream.launch_builder(func);
        args.push(&mut builder);
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn launch<A>(
        &self,
        func: &CudaFunction,
        cfg: LaunchConfig,
        args: A,
    ) -> std::result::Result<(), cudarc::driver::DriverError>
    where
        for<'a> A: KernelArgs<'a>,
    {
        let stream = self.launch_stream();
        self.launch_on_stream(&stream, func, cfg, args)
    }

    pub(crate) fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> std::result::Result<CudaSlice<T>, cudarc::driver::DriverError> {
        self.default_stream.alloc_zeros(len)
    }

    pub unsafe fn alloc<T: DeviceRepr>(
        &self,
        len: usize,
    ) -> std::result::Result<CudaSlice<T>, cudarc::driver::DriverError> {
        self.default_stream.alloc(len)
    }

    pub(crate) fn htod_copy<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
    ) -> std::result::Result<CudaSlice<T>, cudarc::driver::DriverError> {
        self.default_stream.clone_htod(src)
    }

    pub(crate) fn htod_sync_copy<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
    ) -> std::result::Result<CudaSlice<T>, cudarc::driver::DriverError> {
        let stream = self.default_stream.clone();
        let slice = stream.clone_htod(src)?;
        stream.synchronize()?;
        Ok(slice)
    }

    pub(crate) fn htod_sync_copy_into<T: DeviceRepr, Src: HostSlice<T> + ?Sized, Dst: DevicePtrMut<T>>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> std::result::Result<(), cudarc::driver::DriverError> {
        let stream = self.default_stream.clone();
        stream.memcpy_htod(src, dst)?;
        stream.synchronize()
    }

    pub(crate) fn dtoh_sync_copy<T: DeviceRepr, Src: DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> std::result::Result<Vec<T>, cudarc::driver::DriverError> {
        let stream = self.default_stream.clone();
        let host = stream.clone_dtoh(src)?;
        stream.synchronize()?;
        Ok(host)
    }

    pub(crate) fn dtod_copy<T: DeviceRepr, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> std::result::Result<(), cudarc::driver::DriverError> {
        let stream = self.default_stream.clone();
        stream.memcpy_dtod(src, dst)
    }

    pub(crate) fn load_ptx(
        &self,
        ptx: cudarc::nvrtc::Ptx,
        module_name: &str,
        _funcs: &[&str],
    ) -> std::result::Result<(), cudarc::driver::DriverError> {
        if self.modules.read().unwrap().contains_key(module_name) {
            return Ok(());
        }
        let module = self.ctx.load_module(ptx)?;
        let mut modules = self.modules.write().unwrap();
        modules.entry(module_name.to_string()).or_insert(module);
        Ok(())
    }

    pub(crate) fn has_func(&self, module_name: &str, _func_name: &str) -> bool {
        self.modules.read().unwrap().contains_key(module_name)
    }

    pub(crate) fn get_func(&self, module_name: &str, func_name: &str) -> Option<CudaFunction> {
        let module = self.modules.read().unwrap().get(module_name).cloned()?;
        module.load_function(func_name).ok()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: ug::lang::ssa::Kernel,
    ) -> Result<CudaFunction> {
        let mut buf = vec![];
        ug_cuda::code_gen::gen(&mut buf, func_name, &kernel)?;
        let cuda_code = String::from_utf8(buf)?;
        let opts = cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cuda_code, opts).w()?;
        let module = self.ctx.load_module(ptx).w()?;
        let func = module.load_function(func_name).w()?;
        Ok(func)
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    fn const_impl(&self, v: f64, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let stream = self.launch_stream();
        let slice = match dtype {
            DType::U8 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<u8>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_u8", kernels::FILL)?;
                let value = v as u8;
                let params = (&data, &value, &elem_count);
                self.launch_on_stream(&stream, &func, cfg, params).w()?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<u32>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_u32", kernels::FILL)?;
                let value = v as u32;
                let params = (&data, &value, &elem_count);
                self.launch_on_stream(&stream, &func, cfg, params).w()?;
                CudaStorageSlice::U32(data)
            }
            DType::I64 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<i64>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_i64", kernels::FILL)?;
                let value = v as i64;
                let params = (&data, &value, &elem_count);
                self.launch_on_stream(&stream, &func, cfg, params).w()?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<bf16>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_bf16", kernels::FILL)?;
                let value = bf16::from_f64(v);
                let params = (&data, &value, &elem_count);
                self.launch_on_stream(&stream, &func, cfg, params).w()?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<f16>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_f16", kernels::FILL)?;
                let value = f16::from_f64(v);
                let params = (&data, &value, &elem_count);
                self.launch_on_stream(&stream, &func, cfg, params).w()?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<f32>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_f32", kernels::FILL)?;
                let value = v as f32;
                let params = (&data, &value, &elem_count);
                self.launch_on_stream(&stream, &func, cfg, params).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<f64>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_f64", kernels::FILL)?;
                let value = v;
                let params = (&data, &value, &elem_count);
                self.launch_on_stream(&stream, &func, cfg, params).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    pub fn get_or_load_func(&self, module_name: &str, ptx: &'static str) -> Result<CudaFunction> {
        if !self.has_func(module_name, module_name) {
            // Leaking the string here is a bit sad but we need a &'static str and this is only
            // done once per kernel name.
            let static_module_name = Box::leak(module_name.to_string().into_boxed_str());
            self.load_ptx(ptx.into(), module_name, &[static_module_name])
                .map_err(|cuda| CudaError::Load {
                    cuda,
                    module_name: module_name.to_string(),
                })
                .w()?;
        }
        self.get_func(module_name, module_name)
            // Clippy recommends this `ok_or` rather than `ok_or_else` so hopefully the compiler is
            // able to only build the error value if needed.
            .ok_or(CudaError::MissingKernel {
                module_name: module_name.to_string(),
            })
            .w()
    }

    #[cfg(feature = "ptx-os")]
    fn build_device(
        ctx: Arc<CudaContext>,
        default_stream: Arc<CudaStream>,
        ptx_streams: Option<Arc<PtxStreamManager>>,
    ) -> Result<Self> {
        let modules = Arc::new(RwLock::new(HashMap::new()));
        let library_stream = {
            if let Some(streams) = &ptx_streams {
                streams.priority_stream()
            } else {
                default_stream.clone()
            }
        };
        let blas = cudarc::cublas::CudaBlas::new(library_stream.clone()).w()?;
        let curand = cudarc::curand::CudaRng::new(299792458, library_stream.clone()).w()?;
        Ok(Self {
            id: DeviceId::new(),
            ctx,
            default_stream,
            #[cfg(feature = "ptx-os")]
            ptx_streams,
            modules,
            blas: Arc::new(blas),
            blas_stream: library_stream,
            curand: Arc::new(Mutex::new(CudaRng(curand))),
        })
    }

    #[cfg(not(feature = "ptx-os"))]
    fn build_device(ctx: Arc<CudaContext>, default_stream: Arc<CudaStream>) -> Result<Self> {
        let modules = Arc::new(RwLock::new(HashMap::new()));
        let blas_stream = default_stream.clone();
        let blas = cudarc::cublas::CudaBlas::new(blas_stream.clone()).w()?;
        let curand = cudarc::curand::CudaRng::new(299792458, blas_stream.clone()).w()?;
        Ok(Self {
            id: DeviceId::new(),
            ctx,
            default_stream,
            modules,
            blas: Arc::new(blas),
            blas_stream,
            curand: Arc::new(Mutex::new(CudaRng(curand))),
        })
    }

    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).w()?;
        let stream = ctx.new_stream().w()?;
        #[cfg(feature = "ptx-os")]
        {
            Self::build_device(ctx, stream, None)
        }
        #[cfg(not(feature = "ptx-os"))]
        {
            Self::build_device(ctx, stream)
        }
    }

    #[cfg(feature = "ptx-os")]
    pub unsafe fn new_with_ptx_streams(
        ordinal: usize,
        runtime_ptr: *mut std::ffi::c_void,
        num_streams: usize,
    ) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).w()?;
        let streams = ctx.ptx_stream_manager(runtime_ptr, num_streams).w()?;
        let priority = streams.priority_stream();
        Self::build_device(ctx, priority, Some(Arc::new(streams)))
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).w()?;
        let stream = ctx.default_stream();
        #[cfg(feature = "ptx-os")]
        {
            Self::build_device(ctx, stream, None)
        }
        #[cfg(not(feature = "ptx-os"))]
        {
            Self::build_device(ctx, stream)
        }
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        // We do not call set_seed but instead create a new curand object. This ensures that the
        // state will be identical and the same random numbers will be generated.
        let mut curand = self.curand.lock().unwrap();
        curand.0 = cudarc::curand::CudaRng::new(seed, self.blas_stream.clone()).w()?;
        Ok(())
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cuda {
            gpu_id: self.ctx.ordinal(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count).w()?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count).w()?;
                CudaStorageSlice::U32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count).w()?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count).w()?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count).w()?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        let slice = match dtype {
            // TODO: Add support for F16 and BF16 though this is likely to require some upstream
            // cudarc changes.
            DType::U8 | DType::U32 | DType::I64 | DType::F16 | DType::BF16 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_uniform",
                })
                .w()?
            }
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count) }.w()?;
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count) }.w()?;
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        let slice = if lo == 0. && up == 1.0 {
            slice
        } else {
            use super::utils::Map1;
            let layout = Layout::contiguous(shape);
            super::Affine(up - lo, lo).map(&slice, self, &layout)?
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CudaStorage> {
        // TODO: Add support for F16 and BF16 though this is likely to require some upstream
        // cudarc changes.
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        // curand can only generate an odd number of values.
        // https://github.com/huggingface/candle/issues/734
        let elem_count_round = if elem_count % 2 == 1 {
            elem_count + 1
        } else {
            elem_count
        };
        let slice = match dtype {
            DType::U8 | DType::U32 | DType::I64 | DType::F16 | DType::BF16 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_normal",
                })
                .w()?
            }
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count_round) }.w()?;
                curand
                    .0
                    .fill_with_normal(&mut data, mean as f32, std as f32)
                    .w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count_round) }.w()?;
                curand.0.fill_with_normal(&mut data, mean, std).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        self.const_impl(1., shape, dtype)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc::<u8>(elem_count).w()?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc::<u32>(elem_count).w()?;
                CudaStorageSlice::U32(data)
            }
            DType::I64 => {
                let data = self.alloc::<i64>(elem_count).w()?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc::<bf16>(elem_count).w()?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc::<f16>(elem_count).w()?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc::<f32>(elem_count).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc::<f64>(elem_count).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                CudaStorageSlice::U8(data)
            }
            CpuStorageRef::U32(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                CudaStorageSlice::U32(data)
            }
            CpuStorageRef::I64(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                CudaStorageSlice::I64(data)
            }
            CpuStorageRef::BF16(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorageRef::F16(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                CudaStorageSlice::F16(data)
            }
            CpuStorageRef::F32(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                CudaStorageSlice::F32(data)
            }
            CpuStorageRef::F64(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.htod_sync_copy(storage).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.htod_copy(&storage).w()?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.htod_copy(&storage).w()?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.htod_copy(&storage).w()?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.htod_copy(&storage).w()?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.htod_copy(&storage).w()?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.htod_copy(&storage).w()?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.htod_copy(&storage).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "ptx-os")]
        if let Some(streams) = &self.ptx_streams {
            streams.sync_all().w()?;
            return Ok(());
        }
        self.default_stream.synchronize().w()?;
        Ok(())
    }
}
