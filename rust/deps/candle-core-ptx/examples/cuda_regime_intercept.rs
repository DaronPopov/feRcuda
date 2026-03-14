use anyhow::{anyhow, Result} ; use candle_core::{Device, DType, Tensor} ; use ptx_os::RegimeRuntimeCore ; fn run_candle_workload() -> Result<()> {
    let device = Device::new_cuda(0)? ; let a = Tensor::randn(0f32, 1.0, (2048, 2048), &device)?.to_dtype(DType::F32)? ; let b = Tensor::randn(0f32, 1.0, (2048, 2048), &device)?.to_dtype(DType::F32)? ; let c = a.matmul(&b)? ; let _ = c.sum_all()?.to_scalar::<f32>()? ; device.synchronize()? ; Ok(())
}

fn main() -> Result<()> {
    // Initializes runtime + installs global CUDA allocation hook for this process.
    let runtime = RegimeRuntimeCore::new(0)? ; let before = runtime.pool_stats() ; run_candle_workload()? ; let after = runtime.pool_stats() ; let alloc_delta = after.total_allocations.saturating_sub(before.total_allocations) ; let free_delta = after.total_frees.saturating_sub(before.total_frees) ; println!("ptx-os pool stats delta around Candle workload:") ; println!("  total_allocations Δ = {}", alloc_delta) ; println!("  total_frees Δ       = {}", free_delta) ; println!("  allocated bytes     = {}", after.allocated) ; println!("  free bytes          = {}", after.free) ; println!("  utilization %       = {:.2}", after.utilization_percent) ; println!("  fragmentation ratio = {:.6}", after.fragmentation_ratio) ; if alloc_delta == 0 {
        return Err(anyhow!(
            "no PTX-OS allocations observed during Candle workload"
        )) ; }

    println!("pattern check: PASS (Candle workload allocations routed through PTX-OS regime)") ; Ok(())
}
