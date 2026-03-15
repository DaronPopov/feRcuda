use std::ffi::CStr;
use std::os::raw::c_void;

extern "C" {
    fn aten_tlsf_init(device_id: i32);
    fn aten_set_cuda_stream(raw_stream: *mut c_void, device_id: i32);
    fn aten_reset_default_stream(device_id: i32);
    fn dlopen(filename: *const i8, flags: i32) -> *mut c_void;
    fn dlerror() -> *const i8;
}

pub const RTLD_NOW: i32 = 0x2;
pub const RTLD_GLOBAL: i32 = 0x100;
pub const RTLD_NOLOAD: i32 = 0x4;

/// Load a shared library by full path. Returns true if newly loaded, false if already loaded.
pub fn load_shared_lib_path(path: &std::path::Path) -> Result<bool, String> {
    let path_str = path
        .to_str()
        .ok_or_else(|| "Invalid path".to_string())?;
    let mut nul_terminated = String::with_capacity(path_str.len() + 1);
    nul_terminated.push_str(path_str);
    nul_terminated.push('\0');
    unsafe {
        let handle = dlopen(
            nul_terminated.as_ptr() as *const i8,
            RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL,
        );
        if !handle.is_null() {
            return Ok(false);
        }
        let loaded = dlopen(
            nul_terminated.as_ptr() as *const i8,
            RTLD_NOW | RTLD_GLOBAL,
        );
        if loaded.is_null() {
            let err = dlerror();
            let msg = if err.is_null() {
                "unknown error".to_string()
            } else {
                CStr::from_ptr(err).to_string_lossy().into_owned()
            };
            return Err(msg);
        }
    }
    Ok(true)
}

pub fn maybe_load_shared_lib(name: &str) -> Result<bool, String> {
    let mut nul_terminated = String::with_capacity(name.len() + 1);
    nul_terminated.push_str(name);
    nul_terminated.push('\0');
    unsafe {
        let handle = dlopen(
            nul_terminated.as_ptr() as *const i8,
            RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL,
        );
        if !handle.is_null() {
            return Ok(false);
        }

        let loaded = dlopen(
            nul_terminated.as_ptr() as *const i8,
            RTLD_NOW | RTLD_GLOBAL,
        );
        if loaded.is_null() {
            let err = dlerror();
            let msg = if err.is_null() {
                "unknown error".to_string()
            } else {
                CStr::from_ptr(err).to_string_lossy().into_owned()
            };
            return Err(msg);
        }
    }
    Ok(true)
}

pub fn init_torch_allocator(device_id: i32) {
    unsafe { aten_tlsf_init(device_id) };
}

pub fn set_torch_stream(raw_stream: *mut c_void, device_id: i32) {
    unsafe { aten_set_cuda_stream(raw_stream, device_id) };
}

pub fn reset_torch_stream(device_id: i32) {
    unsafe { aten_reset_default_stream(device_id) };
}

pub fn warmup_cuda_allocator() -> Result<(), String> {
    unsafe {
        let ptr = cudarc_ptx::driver::result::malloc_sync(256)
            .map_err(|e| format!("cudaMalloc warmup failed: {:?}", e))?;
        cudarc_ptx::driver::result::free_sync(ptr)
            .map_err(|e| format!("cudaFree warmup failed: {:?}", e))?;
    }
    Ok(())
}
