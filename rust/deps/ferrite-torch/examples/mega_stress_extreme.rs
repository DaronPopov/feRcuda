//! EXTREME BENCHMARK: combined compute-heavy and allocation-heavy TLSF stress.
//!
//! Run: feRcuda-mega-test-extreme (or cargo run --example mega_stress_extreme)

use anyhow::Result;
use std::time::{Duration, Instant};

const PIPELINE_STAGES: usize = 6;
const PIPELINE_ITERS: usize = 120;
const PIPELINE_BATCH: i64 = 128;
const PIPELINE_DIM: i64 = 4096;

const FRAG_BUCKETS: usize = 10;
const FRAG_ROUNDS: usize = 80;
const FRAG_BASE_DIM: i64 = 768;
const FRAG_STEP_DIM: i64 = 320;

const STREAM_ITERS: usize = 60;
const STREAM_BATCH: i64 = 64;
const STREAM_DIM: i64 = 2048;

fn banner(s: &str) {
    println!("\n{}", "═".repeat(72));
    println!("  {s}");
    println!("{}", "═".repeat(72));
}

fn print_metric(label: &str, value: f64, unit: &str) {
    println!("  {label:<18} {value:>12.2} {unit}");
}

fn gib(bytes: f64) -> f64 {
    bytes / (1024.0 * 1024.0 * 1024.0)
}

fn sync_cuda() {
    if tch::Cuda::is_available() {
        tch::Cuda::synchronize(0);
    }
}

fn pipeline_phase() -> Result<(Duration, f64, f64)> {
    use tch::{Device, Kind, Tensor};

    let device = Device::Cuda(0);
    let start = Instant::now();
    let mut total_flops = 0.0;

    for iter in 0..PIPELINE_ITERS {
        let mut activations = Vec::with_capacity(PIPELINE_STAGES);
        let mut x = Tensor::randn(&[PIPELINE_BATCH, PIPELINE_DIM], (Kind::Float, device));

        for stage in 0..PIPELINE_STAGES {
            let w = Tensor::randn(&[PIPELINE_DIM, PIPELINE_DIM], (Kind::Float, device));
            let b = Tensor::randn(&[PIPELINE_DIM], (Kind::Float, device));
            x = x.matmul(&w) + b;
            x = x.relu();
            if stage % 2 == 0 {
                x = x.dropout(0.10, false);
            }
            activations.push(x.shallow_clone());
            total_flops += 2.0 * PIPELINE_BATCH as f64 * PIPELINE_DIM as f64 * PIPELINE_DIM as f64;
        }

        // Chain: start [B,D], then [B,D]@[D,B]=[B,B], then [B,B]@[B,D]=[B,D], alternating
        let mut reduce = activations.pop().expect("pipeline produces activations");
        let mut use_transpose = true; // first: reduce [B,D] @ next.T [D,B]
        while let Some(next) = activations.pop() {
            let next_arg = if use_transpose {
                next.transpose(0, 1)
            } else {
                next
            };
            reduce = reduce.matmul(&next_arg);
            total_flops += if use_transpose {
                2.0 * PIPELINE_BATCH as f64 * PIPELINE_DIM as f64 * PIPELINE_BATCH as f64
            } else {
                2.0 * PIPELINE_BATCH as f64 * PIPELINE_BATCH as f64 * PIPELINE_DIM as f64
            };
            use_transpose = !use_transpose;
        }
        drop(reduce);

        if (iter + 1) % 30 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "  {}/{} pipeline iters | {:.2} it/s | {:.2} TFLOPS est.",
                iter + 1,
                PIPELINE_ITERS,
                (iter + 1) as f64 / elapsed,
                total_flops / elapsed / 1e12
            );
        }
    }

    sync_cuda();
    let elapsed = start.elapsed();
    Ok((elapsed, PIPELINE_ITERS as f64 / elapsed.as_secs_f64(), total_flops / elapsed.as_secs_f64() / 1e12))
}

fn fragmentation_phase() -> Result<(Duration, f64, f64)> {
    use tch::{Device, Kind, Tensor};

    let device = Device::Cuda(0);
    let start = Instant::now();
    let mut peak_live_bytes: f64 = 0.0;

    for round in 0..FRAG_ROUNDS {
        let mut buckets: Vec<Vec<Tensor>> = Vec::with_capacity(FRAG_BUCKETS);
        let mut live_bytes: f64 = 0.0;

        for bucket in 0..FRAG_BUCKETS {
            let dim = FRAG_BASE_DIM + bucket as i64 * FRAG_STEP_DIM;
            let reps = 3 + (bucket % 4);
            let mut bucket_tensors = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t = Tensor::randn(&[dim, dim], (Kind::Float, device));
                live_bytes += dim as f64 * dim as f64 * 4.0;
                bucket_tensors.push(t);
            }
            buckets.push(bucket_tensors);
        }
        peak_live_bytes = peak_live_bytes.max(live_bytes);

        for bucket in (0..FRAG_BUCKETS).step_by(2) {
            buckets[bucket].clear();
        }
        for bucket in (1..FRAG_BUCKETS).step_by(2) {
            let dim = FRAG_BASE_DIM + bucket as i64 * FRAG_STEP_DIM + 128;
            for _ in 0..2 {
                buckets[bucket].push(Tensor::randn(&[dim, dim], (Kind::Float, device)));
            }
        }

        if (round + 1) % 20 == 0 {
            println!(
                "  {}/{} fragmentation rounds | live-set peak {:.2} GiB",
                round + 1,
                FRAG_ROUNDS,
                gib(peak_live_bytes)
            );
        }
    }

    sync_cuda();
    let elapsed = start.elapsed();
    Ok((elapsed, FRAG_ROUNDS as f64 / elapsed.as_secs_f64(), gib(peak_live_bytes)))
}

fn sustained_stream_phase() -> Result<(Duration, f64, f64)> {
    use tch::{Device, Kind, Tensor};

    let device = Device::Cuda(0);
    let start = Instant::now();
    let mut total_bytes = 0.0;

    for iter in 0..STREAM_ITERS {
        let q = Tensor::randn(&[STREAM_BATCH, STREAM_DIM], (Kind::Float, device));
        let k = Tensor::randn(&[STREAM_BATCH, STREAM_DIM], (Kind::Float, device));
        let v = Tensor::randn(&[STREAM_BATCH, STREAM_DIM], (Kind::Float, device));
        let scores = q.matmul(&k.transpose(0, 1));
        let probs = scores.softmax(-1, Kind::Float);
        let out = probs.matmul(&v);
        let norm = out.layer_norm(
            [STREAM_DIM],
            Option::<&Tensor>::None,
            Option::<&Tensor>::None,
            1e-5,
            false,
        );
        drop(norm);

        total_bytes += 6.0 * STREAM_BATCH as f64 * STREAM_DIM as f64 * 4.0;
        total_bytes += 2.0 * STREAM_BATCH as f64 * STREAM_BATCH as f64 * 4.0;

        if (iter + 1) % 15 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "  {}/{} stream iters | {:.2} it/s | {:.2} GiB/s tensor traffic",
                iter + 1,
                STREAM_ITERS,
                (iter + 1) as f64 / elapsed,
                gib(total_bytes / elapsed)
            );
        }
    }

    sync_cuda();
    let elapsed = start.elapsed();
    Ok((elapsed, STREAM_ITERS as f64 / elapsed.as_secs_f64(), gib(total_bytes / elapsed.as_secs_f64())))
}

fn main() -> Result<()> {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  EXTREME BENCHMARK: compute-heavy + allocation-heavy TLSF stress    ║");
    println!("║  Deep GEMM pipeline | fragmentation churn | attention-style stream  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    banner("RUN CONFIG");
    println!(
        "  Pipeline:       {PIPELINE_ITERS} iters, {PIPELINE_STAGES} GEMM stages, batch {PIPELINE_BATCH}, hidden {PIPELINE_DIM}"
    );
    println!(
        "  Fragmentation:  {FRAG_ROUNDS} rounds across {FRAG_BUCKETS} size buckets ({FRAG_BASE_DIM}..{})",
        FRAG_BASE_DIM + (FRAG_BUCKETS as i64 - 1) * FRAG_STEP_DIM
    );
    println!(
        "  Stream:         {STREAM_ITERS} attention-style iters, batch {STREAM_BATCH}, hidden {STREAM_DIM}"
    );
    println!();

    aten_ptx::init_pytorch_tlsf(0, 0.70).map_err(anyhow::Error::msg)?;
    println!("✓ TLSF pool initialized (70% VRAM)");

    let (pipeline_elapsed, pipeline_iter_per_sec, pipeline_tflops) = pipeline_phase()?;
    banner("PHASE 1: Deep compute pipeline");
    println!("✓ Phase 1 done in {:?}", pipeline_elapsed);
    print_metric("Throughput", pipeline_iter_per_sec, "iters/sec");
    print_metric("Compute", pipeline_tflops, "TFLOPS est.");

    let (frag_elapsed, frag_rounds_per_sec, frag_peak_gib) = fragmentation_phase()?;
    banner("PHASE 2: Fragmentation churn");
    println!("✓ Phase 2 done in {:?}", frag_elapsed);
    print_metric("Throughput", frag_rounds_per_sec, "rounds/sec");
    print_metric("Peak live set", frag_peak_gib, "GiB");

    let (stream_elapsed, stream_iters_per_sec, stream_gib_per_sec) = sustained_stream_phase()?;
    banner("PHASE 3: Attention-style stream");
    println!("✓ Phase 3 done in {:?}", stream_elapsed);
    print_metric("Throughput", stream_iters_per_sec, "iters/sec");
    print_metric("Traffic", stream_gib_per_sec, "GiB/sec");

    banner("TLSF POOL STATS");
    if let Some(s) = aten_ptx::get_pool_stats() {
        print_metric("Pool size", s.total_size as f64 / 1e9, "GB");
        print_metric("Allocated", s.allocated as f64 / 1e6, "MB");
        print_metric("Peak", s.peak_allocated as f64 / 1e6, "MB");
        print_metric("Fragmentation", s.fragmentation_ratio as f64, "");
        print_metric("Utilization", s.utilization_percent as f64, "%");
    }

    banner("EXTREME BENCHMARK COMPLETE");
    println!(
        "  Total runtime:   {:?}",
        pipeline_elapsed + frag_elapsed + stream_elapsed
    );
    println!("  Outstanding:     {} allocations", aten_ptx::check_leaks());
    println!();
    println!("  ✅ Heavy compute and heavy allocation stress completed.");
    println!("  ✅ Numeric results above.");
    println!();

    Ok(())
}
