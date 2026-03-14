use std::ffi::c_void;

use anyhow::{anyhow, Result};
use candle_core::DType;
use candle_ptx_os::{PtxDevice, PtxTensor};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct InterceptStats {
    init_calls: u64,
    init_success: u64,
    init_fail: u64,
    alloc_calls_total: u64,
    free_calls_total: u64,
    alloc_calls_driver: u64,
    free_calls_driver: u64,
    alloc_calls_runtime: u64,
    free_calls_runtime: u64,
    alloc_calls_async: u64,
    free_calls_async: u64,
    alloc_bytes_requested: u64,
    tlsf_alloc_success: u64,
    tlsf_alloc_fail: u64,
    tlsf_free_success: u64,
    tlsf_free_miss: u64,
    fallback_alloc_calls: u64,
    fallback_free_calls: u64,
}

type GetFn = unsafe extern "C" fn(*mut InterceptStats) -> i32;
type ResetFn = unsafe extern "C" fn();
type EnabledFn = unsafe extern "C" fn() -> i32;
type ShutdownFn = unsafe extern "C" fn();

#[link(name = "dl")]
extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const i8) -> *mut c_void;
}

const RTLD_DEFAULT: *mut c_void = 0 as *mut c_void;

unsafe fn load_symbol_raw(name: &'static [u8]) -> Result<*mut c_void> {
    let ptr = dlsym(RTLD_DEFAULT, name.as_ptr() as *const i8);
    if ptr.is_null() {
        return Err(anyhow!(
            "symbol not found: {}",
            String::from_utf8_lossy(name)
        ));
    }
    Ok(ptr)
}

fn load_hooks() -> Result<(GetFn, ResetFn, EnabledFn, ShutdownFn)> {
    unsafe {
        let get = load_symbol_raw(b"fercuda_intercept_telemetry_get\0")?;
        let reset = load_symbol_raw(b"fercuda_intercept_telemetry_reset\0")?;
        let enabled = load_symbol_raw(b"fercuda_intercept_telemetry_enabled\0")?;
        let shutdown = load_symbol_raw(b"fercuda_intercept_shutdown\0")?;
        Ok((
            std::mem::transmute::<*mut c_void, GetFn>(get),
            std::mem::transmute::<*mut c_void, ResetFn>(reset),
            std::mem::transmute::<*mut c_void, EnabledFn>(enabled),
            std::mem::transmute::<*mut c_void, ShutdownFn>(shutdown),
        ))
    }
}

fn run_candle_ptx_os_workload() -> Result<()> {
    let device = PtxDevice::with_pool_size(0, 128 * 1024 * 1024)?;
    let a = PtxTensor::randn(&device, (1024, 1024), 0.0, 1.0)?.to_dtype(DType::F32)?;
    let b = PtxTensor::randn(&device, (1024, 1024), 0.0, 1.0)?.to_dtype(DType::F32)?;
    let c = a.matmul(&b)?;
    let _ = c.sum_all()?;
    Ok(())
}

fn main() -> Result<()> {
    let (get_stats, reset_stats, is_enabled, shutdown_hook) = load_hooks()?;
    unsafe { reset_stats() };

    run_candle_ptx_os_workload()?;

    let enabled = unsafe { is_enabled() };
    let mut st = InterceptStats::default();
    let rc = unsafe { get_stats(&mut st as *mut InterceptStats) };
    if rc != 0 {
        return Err(anyhow!("failed to read intercept telemetry"));
    }

    println!("fercuda intercept enabled = {}", enabled);
    println!("fercuda intercept stats (candle-ptx-os):");
    println!("  init_calls            = {}", st.init_calls);
    println!("  init_success          = {}", st.init_success);
    println!("  init_fail             = {}", st.init_fail);
    println!("  alloc_calls_total     = {}", st.alloc_calls_total);
    println!("  tlsf_alloc_success    = {}", st.tlsf_alloc_success);
    println!("  tlsf_alloc_fail       = {}", st.tlsf_alloc_fail);
    println!("  fallback_alloc_calls  = {}", st.fallback_alloc_calls);
    println!("  free_calls_total      = {}", st.free_calls_total);
    println!("  tlsf_free_success     = {}", st.tlsf_free_success);
    println!("  fallback_free_calls   = {}", st.fallback_free_calls);

    if enabled == 0 {
        return Err(anyhow!("interceptor is disabled"));
    }
    if st.alloc_calls_total == 0 {
        return Err(anyhow!("no intercepted allocation calls observed"));
    }
    if st.tlsf_alloc_success == 0 {
        return Err(anyhow!("no TLSF allocation successes observed"));
    }

    println!("pattern check: PASS (candle-ptx-os exercised fercuda malloc interception)");
    unsafe { shutdown_hook() };
    Ok(())
}
