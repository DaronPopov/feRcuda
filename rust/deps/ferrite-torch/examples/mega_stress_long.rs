//! LONG BENCHMARK: feRcuda TLSF extended stress with numeric results
//!
//! Extended phases with detailed throughput and GFLOPS reporting.
//! Run: feRcuda-mega-test-long (or cargo run --example mega_stress_long)

use anyhow::Result;
use std::time::Instant;

// Larger-scale B200 run tuned to stay interactive while producing more useful numerics.
const CHURN_CYCLES: usize = 25_000;      // 512x512 matmuls (~1MB inputs each)
const CHURN_M: i64 = 512;
const CHURN_N: i64 = 512;
const CHURN_K: i64 = 512;
const NAS_SAMPLES: usize = 3_000;
const NAS_BATCH: i64 = 32;
const NAS_HIDDEN: i64 = 1024;
const LARGE_TENSORS: usize = 160;        // 4096x4096 = 64MB each (~10.2GB peak)
const LARGE_DIM: i64 = 4096;
const STREAMING_OPS: usize = 200_000;    // 512x512 = 1MB each
const STREAM_DIM: i64 = 512;
const GIANT_TENSORS: usize = 12;         // 8192x8192 = 256MB each (~3GB burst)
const GIANT_DIM: i64 = 8192;

fn banner(s: &str) {
    println!("\n{}", "═".repeat(72));
    println!("  {s}");
    println!("{}", "═".repeat(72));
}

fn print_numeric(label: &str, value: f64, unit: &str) {
    println!("  {label}: {value:.2} {unit}");
}

fn gib(bytes: f64) -> f64 {
    bytes / (1024.0 * 1024.0 * 1024.0)
}

fn main() -> Result<()> {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  LONG BENCHMARK: feRcuda TLSF extended stress (numeric results)       ║");
    println!("║  Candle + PyTorch | One pool | O(1) alloc | Zero fragmentation       ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    banner("RUN CONFIG");
    println!("  Phase 1 churn:  {CHURN_CYCLES} mixed Candle/PyTorch matmuls ({CHURN_M}x{CHURN_K} x {CHURN_K}x{CHURN_N})");
    println!("  Phase 2 NAS:    {NAS_SAMPLES} sampled architectures (batch {NAS_BATCH}, hidden {NAS_HIDDEN})");
    println!("  Phase 3 burst:  {LARGE_TENSORS} tensors @ {LARGE_DIM}x{LARGE_DIM} ({:.2} GiB live set)", gib(LARGE_TENSORS as f64 * LARGE_DIM as f64 * LARGE_DIM as f64 * 4.0));
    println!("  Phase 4 stream: {STREAMING_OPS} alloc-compute-free iterations ({STREAM_DIM}x{STREAM_DIM})");
    println!("  Phase 5 giant:  {GIANT_TENSORS} tensors @ {GIANT_DIM}x{GIANT_DIM} ({:.2} GiB burst)", gib(GIANT_TENSORS as f64 * GIANT_DIM as f64 * GIANT_DIM as f64 * 4.0));
    println!();

    aten_ptx::init_pytorch_tlsf(0, 0.70).map_err(anyhow::Error::msg)?;
    println!("✓ TLSF pool initialized (70% VRAM)\n");

    unsafe {
        let p = cudarc_ptx::driver::result::malloc_sync(256)
            .map_err(|e| anyhow::Error::msg(format!("warmup: {:?}", e)))?;
        cudarc_ptx::driver::result::free_sync(p)
            .map_err(|e| anyhow::Error::msg(format!("warmup: {:?}", e)))?;
    }

    let mut total_elapsed = std::time::Duration::ZERO;

    // -------------------------------------------------------------------------
    // Phase 1: Large matmul churn (512x512, ~1MB per tensor)
    // -------------------------------------------------------------------------
    banner(&format!("PHASE 1: Large matmul churn ({}x{} @ {} cycles)", CHURN_M, CHURN_N, CHURN_CYCLES));
    let start = Instant::now();

    #[cfg(feature = "candle-cohab")]
    {
        use candle_core::{Device, Tensor};

        let candle_dev = Device::new_cuda(0)?;
        for i in 0..CHURN_CYCLES {
            if i % 2 == 0 {
                let a = Tensor::randn(0f32, 1.0, (CHURN_M as usize, CHURN_K as usize), &candle_dev)?;
                let b = Tensor::randn(0f32, 1.0, (CHURN_K as usize, CHURN_N as usize), &candle_dev)?;
                let _c = a.matmul(&b)?;
            } else if tch::Cuda::is_available() {
                let a = tch::Tensor::randn(&[CHURN_M, CHURN_K], (tch::Kind::Float, tch::Device::Cuda(0)));
                let b = tch::Tensor::randn(&[CHURN_K, CHURN_N], (tch::Kind::Float, tch::Device::Cuda(0)));
                let _c = a.matmul(&b);
            }
            if (i + 1) % 5000 == 0 {
                let elapsed = start.elapsed();
                let ops_per_sec = (i + 1) as f64 / elapsed.as_secs_f64();
                let us_per_op = elapsed.as_micros() as f64 / (i + 1) as f64;
                println!("  {} cycles | {:.0} ops/sec | {:.1} µs/op", i + 1, ops_per_sec, us_per_op);
            }
        }
        candle_dev.synchronize()?;
        if tch::Cuda::is_available() {
            tch::Cuda::synchronize(0);
        }
    }

    #[cfg(not(feature = "candle-cohab"))]
    {
        if tch::Cuda::is_available() {
            for i in 0..CHURN_CYCLES {
                let a = tch::Tensor::randn(&[CHURN_M, CHURN_K], (tch::Kind::Float, tch::Device::Cuda(0)));
                let b = tch::Tensor::randn(&[CHURN_K, CHURN_N], (tch::Kind::Float, tch::Device::Cuda(0)));
                let _c = a.matmul(&b);
                if (i + 1) % 5000 == 0 {
                    let elapsed = start.elapsed();
                    let ops_per_sec = (i + 1) as f64 / elapsed.as_secs_f64();
                    let us_per_op = elapsed.as_micros() as f64 / (i + 1) as f64;
                    println!("  {} cycles | {:.0} ops/sec | {:.1} µs/op", i + 1, ops_per_sec, us_per_op);
                }
            }
            tch::Cuda::synchronize(0);
        }
    }

    let phase1 = start.elapsed();
    total_elapsed += phase1;
    let phase1_ops_per_sec = CHURN_CYCLES as f64 / phase1.as_secs_f64();
    let flops_per_matmul = 2.0 * CHURN_M as f64 * CHURN_N as f64 * CHURN_K as f64;
    let phase1_gflops = phase1_ops_per_sec * flops_per_matmul / 1e9;
    let phase1_tflops = phase1_gflops / 1e3;
    println!("✓ Phase 1 done in {:?}", phase1);
    print_numeric("Throughput", phase1_ops_per_sec, "ops/sec");
    print_numeric("Latency", phase1.as_micros() as f64 / CHURN_CYCLES as f64, "µs/op");
    print_numeric("Est. Throughput", phase1_gflops, "GFLOPS");
    print_numeric("Est. Throughput", phase1_tflops, "TFLOPS");

    // -------------------------------------------------------------------------
    // Phase 2: Dynamic NAS (large batch, 1024 hidden)
    // -------------------------------------------------------------------------
    banner(&format!("PHASE 2: Dynamic NAS ({}x{} batch, {} arch)", NAS_BATCH, NAS_HIDDEN, NAS_SAMPLES));
    let start = Instant::now();

    if tch::Cuda::is_available() {
        use tch::{nn, Device, Kind, Tensor};

        let device = Device::Cuda(0);
        let vs = nn::VarStore::new(device);
        let mut layers = Vec::new();
        for i in 0..100 {
            layers.push(nn::linear(&vs.root() / format!("l{i}"), NAS_HIDDEN, NAS_HIDDEN, Default::default()));
        }

        let mut total_depth = 0usize;
        for i in 0..NAS_SAMPLES {
            let depth = 5 + (i * 7) % 95;
            total_depth += depth;
            let x = Tensor::randn(&[NAS_BATCH, NAS_HIDDEN], (Kind::Float, device));
            let mut act = x;
            for j in 0..depth.min(100) {
                act = act.apply(&layers[j as usize]).relu();
            }
            if (i + 1) % 500 == 0 {
                let elapsed = start.elapsed();
                let arch_per_sec = (i + 1) as f64 / elapsed.as_secs_f64();
                println!("  {} architectures | {:.1} arch/sec", i + 1, arch_per_sec);
            }
        }
        tch::Cuda::synchronize(0);
        let avg_depth = total_depth as f64 / NAS_SAMPLES as f64;
        print_numeric("Avg sampled depth", avg_depth, "layers");
    }

    let phase2 = start.elapsed();
    total_elapsed += phase2;
    let phase2_arch_per_sec = NAS_SAMPLES as f64 / phase2.as_secs_f64();
    let phase2_avg_depth = (0..NAS_SAMPLES)
        .map(|i| 5 + (i * 7) % 95)
        .sum::<usize>() as f64
        / NAS_SAMPLES as f64;
    let phase2_layer_evals_per_sec = phase2_arch_per_sec * phase2_avg_depth;
    println!("✓ Phase 2 done in {:?}", phase2);
    print_numeric("Throughput", phase2_arch_per_sec, "architectures/sec");
    print_numeric("Effective work", phase2_layer_evals_per_sec, "layer-evals/sec");

    // -------------------------------------------------------------------------
    // Phase 3: Large-tensor burst (4096x4096 = 64MB each)
    // -------------------------------------------------------------------------
    banner(&format!("PHASE 3: Large-tensor burst ({}x{} x {} tensors)", LARGE_DIM, LARGE_DIM, LARGE_TENSORS));
    let start = Instant::now();

    #[cfg(feature = "candle-cohab")]
    {
        use candle_core::{Device, Tensor};

        let dev = Device::new_cuda(0)?;
        let mut tensors = Vec::new();
        for _ in 0..LARGE_TENSORS {
            tensors.push(Tensor::randn(0f32, 1.0, (LARGE_DIM as usize, LARGE_DIM as usize), &dev)?);
        }
        for _ in 0..LARGE_TENSORS / 2 {
            tensors.pop();
        }
        for _ in 0..LARGE_TENSORS / 2 {
            tensors.push(Tensor::randn(0f32, 1.0, (LARGE_DIM as usize, LARGE_DIM as usize), &dev)?);
        }
        dev.synchronize()?;
    }

    if tch::Cuda::is_available() {
        let mut tensors = Vec::new();
        for _ in 0..LARGE_TENSORS {
            tensors.push(tch::Tensor::randn(&[LARGE_DIM, LARGE_DIM], (tch::Kind::Float, tch::Device::Cuda(0))));
        }
        for _ in 0..LARGE_TENSORS / 2 {
            tensors.pop();
        }
        for _ in 0..LARGE_TENSORS / 2 {
            tensors.push(tch::Tensor::randn(&[LARGE_DIM, LARGE_DIM], (tch::Kind::Float, tch::Device::Cuda(0))));
        }
        tch::Cuda::synchronize(0);
    }

    let phase3 = start.elapsed();
    total_elapsed += phase3;
    let mb_per_tensor = LARGE_DIM as f64 * LARGE_DIM as f64 * 4.0 / 1e6; // f32 = 4 bytes
    let total_mb = LARGE_TENSORS as f64 * 1.5 * mb_per_tensor;
    let phase3_mb_per_sec = total_mb / phase3.as_secs_f64();
    let phase3_live_gib = gib(LARGE_TENSORS as f64 * LARGE_DIM as f64 * LARGE_DIM as f64 * 4.0);
    println!("✓ Phase 3 done in {:?}", phase3);
    print_numeric("Peak tensors", LARGE_TENSORS as f64, "");
    print_numeric("Live set", phase3_live_gib, "GiB");
    print_numeric("Memory throughput", phase3_mb_per_sec, "MB/sec");

    // -------------------------------------------------------------------------
    // Phase 4: Streaming alloc-compute-free (512x512 = 1MB per tensor)
    // -------------------------------------------------------------------------
    banner(&format!("PHASE 4: Streaming ({}k ops, {}x{} tensors)", STREAMING_OPS / 1000, STREAM_DIM, STREAM_DIM));
    let start = Instant::now();

    #[cfg(feature = "candle-cohab")]
    {
        use candle_core::{Device, Tensor};

        let dev = Device::new_cuda(0)?;
        for i in 0..STREAMING_OPS {
            let a = Tensor::randn(0f32, 1.0, (STREAM_DIM as usize, STREAM_DIM as usize), &dev)?;
            let b = a.matmul(&a)?;
            drop(b);
            drop(a);
            if (i + 1) % 25_000 == 0 {
                let elapsed = start.elapsed();
                let ops_per_sec = (i + 1) as f64 / elapsed.as_secs_f64();
                let ns_per_op = elapsed.as_nanos() as f64 / (i + 1) as f64;
                println!("  {} ops | {:.0} ops/sec | {:.0} ns/op", i + 1, ops_per_sec, ns_per_op);
            }
        }
        dev.synchronize()?;
    }

    #[cfg(not(feature = "candle-cohab"))]
    {
        if tch::Cuda::is_available() {
            for i in 0..STREAMING_OPS {
                let a = tch::Tensor::randn(&[STREAM_DIM, STREAM_DIM], (tch::Kind::Float, tch::Device::Cuda(0)));
                let b = a.matmul(&a);
                drop(b);
                drop(a);
                if (i + 1) % 25_000 == 0 {
                    let elapsed = start.elapsed();
                    let ops_per_sec = (i + 1) as f64 / elapsed.as_secs_f64();
                    let ns_per_op = elapsed.as_nanos() as f64 / (i + 1) as f64;
                    println!("  {} ops | {:.0} ops/sec | {:.0} ns/op", i + 1, ops_per_sec, ns_per_op);
                }
            }
            tch::Cuda::synchronize(0);
        }
    }

    let phase4 = start.elapsed();
    total_elapsed += phase4;
    let phase4_ops_per_sec = STREAMING_OPS as f64 / phase4.as_secs_f64();
    let phase4_ns_per_op = phase4.as_nanos() as f64 / STREAMING_OPS as f64;
    let stream_bytes_per_op = 3.0 * STREAM_DIM as f64 * STREAM_DIM as f64 * 4.0;
    let phase4_gib_per_sec = gib(phase4_ops_per_sec * stream_bytes_per_op);
    println!("✓ Phase 4 done in {:?}", phase4);
    print_numeric("Throughput", phase4_ops_per_sec, "ops/sec");
    print_numeric("Latency", phase4_ns_per_op, "ns/op");
    print_numeric("Tensor traffic", phase4_gib_per_sec, "GiB/sec");

    // -------------------------------------------------------------------------
    // Phase 5: Giant tensor burst (8192x8192 = 256MB each)
    // -------------------------------------------------------------------------
    banner(&format!("PHASE 5: Giant tensor burst ({}x{} x {} tensors = ~{}GB)", GIANT_DIM, GIANT_DIM, GIANT_TENSORS, (GIANT_TENSORS as f64 * GIANT_DIM as f64 * GIANT_DIM as f64 * 4.0 / 1e9) as i64));
    let start = Instant::now();

    #[cfg(feature = "candle-cohab")]
    {
        use candle_core::{Device, Tensor};

        let dev = Device::new_cuda(0)?;
        let mut tensors = Vec::new();
        for i in 0..GIANT_TENSORS {
            tensors.push(Tensor::randn(0f32, 1.0, (GIANT_DIM as usize, GIANT_DIM as usize), &dev)?);
            println!("  Allocated giant tensor {} ({}x{} = 256MB)", i + 1, GIANT_DIM, GIANT_DIM);
        }
        tensors.clear();
        dev.synchronize()?;
    }

    if tch::Cuda::is_available() {
        let mut tensors = Vec::new();
        for i in 0..GIANT_TENSORS {
            tensors.push(tch::Tensor::randn(&[GIANT_DIM, GIANT_DIM], (tch::Kind::Float, tch::Device::Cuda(0))));
            println!("  Allocated giant tensor {} ({}x{} = 256MB)", i + 1, GIANT_DIM, GIANT_DIM);
        }
        tensors.clear();
        tch::Cuda::synchronize(0);
    }

    let phase5 = start.elapsed();
    total_elapsed += phase5;
    let phase5_gib_allocated = gib(GIANT_TENSORS as f64 * GIANT_DIM as f64 * GIANT_DIM as f64 * 4.0);
    let phase5_allocs_per_sec = GIANT_TENSORS as f64 / phase5.as_secs_f64();
    println!("✓ Phase 5 done in {:?}", phase5);
    print_numeric("Peak allocation", phase5_gib_allocated, "GiB");
    print_numeric("Allocation rate", phase5_allocs_per_sec, "giant allocs/sec");

    // -------------------------------------------------------------------------
    // Final stats
    // -------------------------------------------------------------------------
    banner("TLSF POOL STATS");
    if let Some(s) = aten_ptx::get_pool_stats() {
        print_numeric("Pool size", s.total_size as f64 / 1e9, "GB");
        print_numeric("Allocated", s.allocated as f64 / 1e6, "MB");
        print_numeric("Peak", s.peak_allocated as f64 / 1e6, "MB");
        print_numeric("Fragmentation", s.fragmentation_ratio as f64, "");
        print_numeric("Utilization", s.utilization_percent as f64, "%");
    }

    banner("LONG BENCHMARK COMPLETE - NUMERIC SUMMARY");
    println!("  Total runtime:   {:?}", total_elapsed);
    println!("  Phase 1:         {:?} ({:.0} ops/sec, {:.2} TFLOPS est.)", phase1, phase1_ops_per_sec, phase1_tflops);
    println!("  Phase 2:         {:?} ({:.0} arch/sec, {:.0} layer-evals/sec)", phase2, phase2_arch_per_sec, phase2_layer_evals_per_sec);
    println!("  Phase 3:         {:?} ({:.0} MB/sec, {:.2} GiB live set)", phase3, phase3_mb_per_sec, phase3_live_gib);
    println!("  Phase 4:         {:?} ({:.0} ops/sec, {:.0} ns/op, {:.2} GiB/sec)", phase4, phase4_ops_per_sec, phase4_ns_per_op, phase4_gib_per_sec);
    println!("  Phase 5:         {:?} ({:.2} GiB burst, {:.1} giant allocs/sec)", phase5, phase5_gib_allocated, phase5_allocs_per_sec);
    println!("  Outstanding:     {} allocations", aten_ptx::check_leaks());
    println!();
    println!("  ✅ Extended TLSF stress complete.");
    println!("  ✅ Numeric results above.");
    println!();

    Ok(())
}
