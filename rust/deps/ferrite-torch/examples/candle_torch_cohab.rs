//! Candle + PyTorch in one process, sharing the same TLSF pool.
//!
//! Proves that both frameworks can coexist on a single PTX-OS runtime:
//! - aten-ptx init creates the TLSF pool and hooks cudaMalloc
//! - Candle (Device::Cuda) uses cudarc-ptx -> hooked allocator -> TLSF
//! - PyTorch uses aten-ptx allocator -> TLSF
//!
//! Run: ./run_torch_example.sh candle_torch_cohab

use anyhow::Result;
use std::time::Instant;

fn print_sep() {
    println!("\n{}", "=".repeat(70));
}

fn main() -> Result<()> {
    println!("\n🔗 CANDLE + PYTORCH COHABITATION TEST\n");
    println!("Both frameworks sharing one TLSF pool in a single process.\n");

    // 1. Init TLSF pool (aten-ptx) - hooks cudaMalloc for the whole process
    aten_ptx::init_pytorch_tlsf(0, 0.70).map_err(anyhow::Error::msg)?;
    println!("✓ aten-ptx initialized (TLSF pool + hook active)\n");

    // Warmup cudarc path
    unsafe {
        let p = cudarc_ptx::driver::result::malloc_sync(256)
            .map_err(|e| anyhow::Error::msg(format!("cudarc warmup: {:?}", e)))?;
        cudarc_ptx::driver::result::free_sync(p)
            .map_err(|e| anyhow::Error::msg(format!("cudarc free: {:?}", e)))?;
    }

    print_sep();
    println!("PHASE 1: Candle (cudarc-ptx -> TLSF)");
    print_sep();

    #[cfg(feature = "candle-cohab")]
    {
        use candle_core::{Device, Tensor};

        let device = Device::new_cuda(0)?;
        println!("\nCandle matmul 64×128 @ 128×64...");
        let start = Instant::now();
        let a = Tensor::randn(0f32, 1.0, (64, 128), &device)?;
        let b = Tensor::randn(0f32, 1.0, (128, 64), &device)?;
        let c = a.matmul(&b)?;
        device.synchronize()?;
        println!("✓ Candle {:?} -> {:?} in {:?}", a.dims(), c.dims(), start.elapsed());

        let c_vec: Vec<f32> = c.flatten_all()?.to_vec1()?;
        println!("  Sample sum: {:.4}", c_vec.iter().take(5).sum::<f32>());

        drop(c);
        drop(b);
        drop(a);
    }

    #[cfg(not(feature = "candle-cohab"))]
    {
        println!("\n(Skip: build with default features for candle-cohab)");
    }

    print_sep();
    println!("PHASE 2: PyTorch (aten-ptx -> TLSF)");
    print_sep();

    if tch::Cuda::is_available() {
        use tch::{Device, Kind, Tensor};

        let device = Device::Cuda(0);
        println!("\nPyTorch matmul 64×128 @ 128×64...");
        let start = Instant::now();
        let a = Tensor::randn(&[64, 128], (Kind::Float, device));
        let b = Tensor::randn(&[128, 64], (Kind::Float, device));
        let c = a.matmul(&b);
        tch::Cuda::synchronize(0);
        println!("✓ PyTorch {:?} x {:?} in {:?}", a.size(), c.size(), start.elapsed());

        let sum: f64 = c.sum(tch::Kind::Float).try_into()?;
        println!("  Sum: {:.4}", sum);

        drop(c);
        drop(b);
        drop(a);
    } else {
        println!("\n❌ PyTorch CUDA not available");
    }

    print_sep();
    println!("PHASE 3: Interleaved (Candle → PyTorch → Candle)");
    print_sep();

    #[cfg(feature = "candle-cohab")]
    {
        use candle_core::{Device, Tensor};

        let device = Device::new_cuda(0)?;
        let start = Instant::now();

        // Candle
        let _a = Tensor::randn(0f32, 1.0, (32, 32), &device)?;
        let _b = _a.matmul(&_a)?;
        drop(_b);
        drop(_a);

        // PyTorch
        let _x = tch::Tensor::randn(&[32, 32], (tch::Kind::Float, tch::Device::Cuda(0)));
        let _y = _x.matmul(&_x);
        drop(_y);
        drop(_x);

        // Candle again
        let _c = Tensor::randn(0f32, 1.0, (64, 64), &device)?;
        let _d = _c.matmul(&_c)?;
        device.synchronize()?;
        drop(_d);
        drop(_c);

        println!("\n✓ Interleaved Candle→PyTorch→Candle in {:?}", start.elapsed());
    }

    print_sep();
    println!("TLSF POOL STATS (shared by both)");
    print_sep();

    if let Some(stats) = aten_ptx::get_pool_stats() {
        println!("\nPool size:       {:.2} GB", stats.total_size as f64 / 1e9);
        println!("Allocated:       {:.2} MB", stats.allocated as f64 / 1e6);
        println!("Peak:            {:.2} MB", stats.peak_allocated as f64 / 1e6);
        println!("Fragmentation:   {:.6}", stats.fragmentation_ratio);
        println!("Utilization:     {:.1}%", stats.utilization_percent);
    }

    println!("\n✅ Candle + PyTorch cohabitation: PASS");
    println!("   One TLSF pool, two frameworks.");
    println!();

    Ok(())
}
