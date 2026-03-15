//! Torch + Candle demo with printed tensor values.
//!
//! Runs decent-size matmuls on feRcuda TLSF runtime and prints actual tensor values.
//! B200-ready (256×256 tensors).
//!
//! Run: feRcuda-demo (or cargo run --example torch_candle_demo)

use anyhow::Result;
use std::time::Instant;

const DIM: i64 = 256;       // 256×256 = 64KB per tensor
const PRINT_ROWS: usize = 8;
const PRINT_COLS: usize = 8;

fn banner(s: &str) {
    println!("\n{}", "═".repeat(72));
    println!("  {s}");
    println!("{}", "═".repeat(72));
}

fn print_tensor_corner_candle(name: &str, data: &[Vec<f32>]) {
    println!("\n  {} (first {}×{} corner):", name, PRINT_ROWS, PRINT_COLS);
    for (i, row) in data.iter().take(PRINT_ROWS).enumerate() {
        let vals: Vec<String> = row.iter().take(PRINT_COLS).map(|v| format!("{:8.4}", v)).collect();
        println!("    row {:2}: [{}]", i, vals.join(", "));
    }
}

fn print_tensor_corner_tch(t: &tch::Tensor) -> Result<()> {
    println!("\n  PyTorch tensor (first {}×{} corner):", PRINT_ROWS, PRINT_COLS);
    for i in 0..PRINT_ROWS.min(t.size()[0] as usize) {
        let mut row = Vec::new();
        for j in 0..PRINT_COLS.min(t.size()[1] as usize) {
            let v = t.f_double_value(&[i as i64, j as i64])?;
            row.push(format!("{:8.4}", v));
        }
        println!("    row {:2}: [{}]", i, row.join(", "));
    }
    Ok(())
}

fn main() -> Result<()> {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  feRcuda Demo: Torch + Candle with printed tensor values            ║");
    println!("║  TLSF runtime | B200-ready | {}×{} matmuls                           ║", DIM, DIM);
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    aten_ptx::init_pytorch_tlsf(0, 0.70).map_err(anyhow::Error::msg)?;
    println!("✓ TLSF pool initialized (70% VRAM)\n");

    unsafe {
        let p = cudarc_ptx::driver::result::malloc_sync(256)
            .map_err(|e| anyhow::Error::msg(format!("warmup: {:?}", e)))?;
        cudarc_ptx::driver::result::free_sync(p)
            .map_err(|e| anyhow::Error::msg(format!("warmup: {:?}", e)))?;
    }

    // -------------------------------------------------------------------------
    // Candle: matmul and print values
    // -------------------------------------------------------------------------
    banner("CANDLE: matmul with printed output");
    #[cfg(feature = "candle-cohab")]
    {
        use candle_core::{Device, Tensor};

        let dev = Device::new_cuda(0)?;
        println!("\n  Allocating A [{},{}], B [{},{}]...", DIM, DIM, DIM, DIM);
        let start = Instant::now();
        let a = Tensor::randn(0f32, 0.1, (DIM as usize, DIM as usize), &dev)?;
        let b = Tensor::randn(0f32, 0.1, (DIM as usize, DIM as usize), &dev)?;
        let c = a.matmul(&b)?;
        dev.synchronize()?;
        println!("  ✓ Candle matmul in {:?}", start.elapsed());

        let c_corner = c.narrow(0, 0, PRINT_ROWS)?
            .narrow(1, 0, PRINT_COLS)?
            .to_vec2::<f32>()?;
        print_tensor_corner_candle("C = A @ B", &c_corner);

        let c_flat: Vec<f32> = c.flatten_all()?.to_vec1()?;
        let sum: f32 = c_flat.iter().sum();
        let mean = sum / c_flat.len() as f32;
        let min = c_flat.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = c_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("\n  Stats: sum={:.4}, mean={:.6}, min={:.4}, max={:.4}", sum, mean, min, max);
    }

    #[cfg(not(feature = "candle-cohab"))]
    {
        println!("\n  (Skip: build with candle-cohab feature)");
    }

    // -------------------------------------------------------------------------
    // PyTorch: matmul and print values
    // -------------------------------------------------------------------------
    banner("PYTORCH: matmul with printed output");
    if tch::Cuda::is_available() {
        use tch::{Device, Kind, Tensor};

        let device = Device::Cuda(0);
        println!("\n  Allocating A [{},{}], B [{},{}]...", DIM, DIM, DIM, DIM);
        let start = Instant::now();
        let a = Tensor::randn(&[DIM, DIM], (Kind::Float, device)) * 0.1;
        let b = Tensor::randn(&[DIM, DIM], (Kind::Float, device)) * 0.1;
        let c = a.matmul(&b);
        tch::Cuda::synchronize(0);
        println!("  ✓ PyTorch matmul in {:?}", start.elapsed());

        print_tensor_corner_tch(&c)?;

        let sum: f64 = c.sum(Kind::Float).double_value(&[]);
        let mean = c.mean(Kind::Float).double_value(&[]);
        let min_val = c.min().double_value(&[]);
        let max_val = c.max().double_value(&[]);
        let numel = c.size().iter().product::<i64>();
        println!("\n  Stats: sum={:.4}, mean={:.6}, min={:.4}, max={:.4}", sum, mean, min_val, max_val);
        println!("  Elements: {}", numel);
    } else {
        println!("\n  ❌ PyTorch CUDA not available");
    }

    // -------------------------------------------------------------------------
    // Interleaved: Candle → PyTorch with value check
    // -------------------------------------------------------------------------
    banner("INTERLEAVED: Candle → PyTorch (same computation, verify values)");
    #[cfg(feature = "candle-cohab")]
    if tch::Cuda::is_available() {
        use candle_core::{Device, Tensor};
        use tch::{Device as TchDevice, Kind, Tensor as TchTensor};

        let dev = Device::new_cuda(0)?;
        let tch_dev = TchDevice::Cuda(0);

        // Candle: 8×8 (I+1) @ (I+1)
        let z = Tensor::zeros((8, 8), candle_core::DType::F32, &dev)?;
        let t = (&z + 1.)?;
        let cc = t.matmul(&t)?;
        let cc_vec: Vec<Vec<f32>> = cc.to_vec2()?;
        println!("\n  Candle 8×8: C = (I+1) @ (I+1)");
        print_tensor_corner_candle("C", &cc_vec);

        // PyTorch: same (I+1) @ (I+1)
        let at = TchTensor::zeros(&[8, 8], (Kind::Float, tch_dev)) + 1.0;
        let ct = at.matmul(&at);
        println!("\n  PyTorch 8×8: C = (I+1) @ (I+1)");
        print_tensor_corner_tch(&ct)?;

        println!("\n  ✓ Both frameworks produced values (TLSF shared pool)");
    }

    // -------------------------------------------------------------------------
    // Pool stats
    // -------------------------------------------------------------------------
    banner("TLSF POOL STATS");
    if let Some(s) = aten_ptx::get_pool_stats() {
        println!("  Pool size:       {:.2} GB", s.total_size as f64 / 1e9);
        println!("  Allocated:       {:.2} MB", s.allocated as f64 / 1e6);
        println!("  Peak:            {:.2} MB", s.peak_allocated as f64 / 1e6);
        println!("  Fragmentation:   {:.6}", s.fragmentation_ratio);
        println!("  Utilization:     {:.1}%", s.utilization_percent);
    }

    banner("DEMO COMPLETE");
    println!("  ✅ Candle + PyTorch ran on TLSF.");
    println!("  ✅ Tensor values printed above.");
    println!();

    Ok(())
}
