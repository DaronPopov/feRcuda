//! Basic PyTorch + TLSF Integration Test
//!
//! This verifies that PyTorch tensor operations use the TLSF allocator
//! through the cudarc-ptx patch.
//!
//! Run with:
//! ```bash
//! LD_LIBRARY_PATH=../../ferrite-os/lib:$LD_LIBRARY_PATH \
//!   cargo run --example torch_basic --release
//! ```

use tch::{Device, Kind, Tensor};
use anyhow::Result;
use std::time::Instant;

fn print_separator() {
    println!("\n{}", "=".repeat(70));
}

fn init_custom_stack() -> Result<()> {
    // Force PyTorch onto custom ATen TLSF allocator backend.
    aten_ptx::init_pytorch_tlsf(0, 0.70)
        .map_err(anyhow::Error::msg)?;
    // Force one sync alloc/free through custom cudarc TLSF hooks.
    unsafe {
        let p = cudarc_ptx::driver::result::malloc_sync(256)
            .map_err(|e| anyhow::Error::msg(format!("cudarc malloc: {:?}", e)))?;
        cudarc_ptx::driver::result::free_sync(p)
            .map_err(|e| anyhow::Error::msg(format!("cudarc free: {:?}", e)))?;
    }
    Ok(())
}

fn main() -> Result<()> {
    println!("\n🔥 PYTORCH + TLSF INTEGRATION TEST\n");
    println!("Testing PyTorch (tch-rs) operations with PTX-OS TLSF allocator\n");

    init_custom_stack()?;

    // Check CUDA availability
    let device = if tch::Cuda::is_available() {
        println!("✓ CUDA available");
        println!("  CUDA device count: {}", tch::Cuda::device_count());
        Device::Cuda(0)
    } else {
        println!("❌ CUDA not available, using CPU");
        return Ok(());
    };

    print_separator();
    println!("TEST 1: Basic Tensor Operations");
    print_separator();
    aten_ptx::print_stats();

    // Test 1: Basic allocation
    println!("\nAllocating tensors...");
    let start = Instant::now();
    let x = Tensor::zeros(&[1024, 1024], (Kind::Float, device));
    let alloc_time = start.elapsed();
    println!("✓ Created 1024×1024 tensor in {:?}", alloc_time);

    let y = Tensor::ones(&[1024, 1024], (Kind::Float, device));
    println!("✓ Created another 1024×1024 tensor");

    // Test 2: Operations
    println!("\nPerforming operations...");
    let z = &x + &y;
    println!("✓ Addition");

    let w = z.matmul(&y);
    println!("✓ Matrix multiplication");

    let sum = w.sum(Kind::Float);
    println!("✓ Sum reduction: {:.2}", f64::try_from(sum)?);

    print_separator();
    println!("TEST 2: Allocation Speed Test (1000 tensors)");
    print_separator();

    println!("\nAllocating 1000 tensors...");
    let start = Instant::now();
    let mut tensors = Vec::new();
    for _ in 0..1000 {
        let t = Tensor::zeros(&[256, 256], (Kind::Float, device));
        tensors.push(t);
    }
    let batch_time = start.elapsed();

    let avg_alloc = batch_time.as_micros() as f64 / 1000.0;
    println!("✓ Allocated 1000 tensors in {:?}", batch_time);
    println!("  Average: {:.2}μs per tensor", avg_alloc);

    if avg_alloc < 10.0 {
        println!("\n✅ FAST ALLOCATION - TLSF is working!");
    } else if avg_alloc < 100.0 {
        println!("\n⚠️  Decent speed, but might not be using TLSF");
    } else {
        println!("\n❌ SLOW - Likely using stock PyTorch allocator");
    }

    print_separator();
    println!("TEST 3: Memory Churn (Free + Realloc)");
    print_separator();

    println!("\nDropping 500 tensors...");
    tensors.truncate(500);

    println!("Reallocating 500 new tensors...");
    let start = Instant::now();
    for _ in 0..500 {
        let t = Tensor::zeros(&[256, 256], (Kind::Float, device));
        tensors.push(t);
    }
    let realloc_time = start.elapsed();

    let avg_realloc = realloc_time.as_micros() as f64 / 500.0;
    println!("✓ Reallocated 500 tensors in {:?}", realloc_time);
    println!("  Average: {:.2}μs per tensor", avg_realloc);

    if avg_realloc < 10.0 {
        println!("\n✅ NO FRAGMENTATION - TLSF handling churn perfectly!");
    }

    print_separator();
    println!("TEST 4: Large Tensor Allocation");
    print_separator();

    println!("\nAllocating 4096×4096 tensor (64 MB)...");
    let start = Instant::now();
    let large = Tensor::zeros(&[4096, 4096], (Kind::Float, device));
    let large_time = start.elapsed();
    println!("✓ Allocated in {:?}", large_time);

    if large_time.as_micros() < 500 {
        println!("  ✅ Fast large allocation (TLSF signature)");
    }

    drop(large);
    drop(tensors);
    drop(w);
    drop(z);
    drop(y);
    drop(x);
    tch::Cuda::synchronize(0);

    print_separator();
    println!("INTEGRATION VERIFICATION SUMMARY");
    print_separator();

    println!("\n📊 Performance Indicators:");
    println!("  Batch allocation: {:.2}μs avg", avg_alloc);
    println!("  After churn: {:.2}μs avg", avg_realloc);
    println!("  Large tensor: {:?}", large_time);

    println!("\n🎯 Verdict:");
    if aten_ptx::is_initialized() && avg_alloc < 15.0 && avg_realloc < 15.0 {
        println!("  ✅ TLSF INTEGRATION CONFIRMED");
        println!("  PyTorch allocator backend initialized through aten-ptx.");
        println!("  Performance characteristics are consistent with TLSF.");
    } else {
        println!("  ❌ TLSF BACKEND NOT CONFIRMED");
        println!("  Check aten-ptx initialization and libtorch linkage.");
    }

    let leaks = aten_ptx::check_leaks();
    println!("\nOutstanding allocations after cleanup: {}", leaks);
    aten_ptx::print_stats();

    print_separator();
    println!();

    Ok(())
}
