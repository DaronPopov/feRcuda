//! Image Classification Pipeline on TLSF Allocator
//!
//! 3-layer CNN processing image batches through real cuDNN kernels:
//!   Conv2D -> BatchNorm -> ReLU -> MaxPool (x3) -> Flatten -> Linear -> Softmax
//!
//! Kernels exercised:
//!   - conv2d (cuDNN forward convolution)
//!   - batch_norm (running mean/var normalization)
//!   - relu (element-wise activation)
//!   - max_pool2d (spatial downsampling)
//!   - matmul (fully-connected classifier)
//!   - softmax (probability output)
//!
//! Every feature map, intermediate buffer, and gradient is managed by TLSF.

use aten_ptx::{init_pytorch_tlsf, print_stats, get_fragmentation, check_leaks};
use tch::{Device, Tensor, Kind};
use anyhow::Result;
use std::time::Instant;

/// Conv -> BatchNorm -> ReLU -> optional MaxPool
fn conv_block(
    input: &Tensor,
    conv_w: &Tensor, conv_b: &Tensor,
    bn_w: &Tensor, bn_b: &Tensor,
    bn_mean: &Tensor, bn_var: &Tensor,
    pool: bool,
) -> Tensor {
    let x = input.conv2d(conv_w, Some(conv_b), &[1, 1], &[1, 1], &[1, 1], 1);
    let x = x.batch_norm(
        Some(bn_w), Some(bn_b), Some(bn_mean), Some(bn_var),
        false, 0.1, 1e-5, true,
    );
    let x = x.relu();
    if pool {
        x.max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)
    } else {
        x
    }
}

/// Create batch-norm parameter set (weight, bias, running_mean, running_var)
fn bn_params(channels: i64, device: Device) -> (Tensor, Tensor, Tensor, Tensor) {
    (
        Tensor::ones(&[channels], (Kind::Float, device)),
        Tensor::zeros(&[channels], (Kind::Float, device)),
        Tensor::zeros(&[channels], (Kind::Float, device)),
        Tensor::ones(&[channels], (Kind::Float, device)),
    )
}

fn main() -> Result<()> {
    println!("\n=== Image Classification Pipeline on TLSF ===\n");
    println!("3-layer CNN: Conv2D -> BatchNorm -> ReLU -> MaxPool");
    println!("Input: 224x224 RGB -> Output: 10 classes\n");

    init_pytorch_tlsf(0, 0.70).map_err(|e| anyhow::anyhow!("{}", e))?;

    let device = Device::Cuda(0);
    let _guard = tch::no_grad_guard();

    // --- CNN weights ---
    // Architecture: 3 -> 64 -> 128 -> 256, with maxpool after each block
    // Spatial: 224 -> 112 -> 56 -> 28
    println!("Initializing CNN...");

    // Layer 1: 3 -> 64
    let c1_w = Tensor::randn(&[64, 3, 3, 3], (Kind::Float, device)) * 0.1;
    let c1_b = Tensor::zeros(&[64], (Kind::Float, device));
    let (bn1_w, bn1_b, bn1_m, bn1_v) = bn_params(64, device);

    // Layer 2: 64 -> 128
    let c2_w = Tensor::randn(&[128, 64, 3, 3], (Kind::Float, device)) * 0.1;
    let c2_b = Tensor::zeros(&[128], (Kind::Float, device));
    let (bn2_w, bn2_b, bn2_m, bn2_v) = bn_params(128, device);

    // Layer 3: 128 -> 256
    let c3_w = Tensor::randn(&[256, 128, 3, 3], (Kind::Float, device)) * 0.1;
    let c3_b = Tensor::zeros(&[256], (Kind::Float, device));
    let (bn3_w, bn3_b, bn3_m, bn3_v) = bn_params(256, device);

    // Classifier: 256*28*28 -> 10
    let fc_w = Tensor::randn(&[256 * 28 * 28, 10], (Kind::Float, device)) * 0.01;
    let fc_b = Tensor::zeros(&[10], (Kind::Float, device));

    println!("  Layer 1: Conv2d(3, 64, 3x3) + BN + ReLU + MaxPool");
    println!("  Layer 2: Conv2d(64, 128, 3x3) + BN + ReLU + MaxPool");
    println!("  Layer 3: Conv2d(128, 256, 3x3) + BN + ReLU + MaxPool");
    println!("  Classifier: Linear(200704, 10) + Softmax");
    print_stats();

    // --- Process batches with varying sizes ---
    let batch_sizes: Vec<i64> = vec![1, 4, 8, 16, 32, 16, 8, 4, 1];
    let total_batches = 200;
    let mut latencies: Vec<(i64, f64)> = Vec::new();
    let mut total_images: i64 = 0;

    println!("\nProcessing {} batches (variable batch sizes)...\n", total_batches);

    for i in 0..total_batches {
        let bs = batch_sizes[i % batch_sizes.len()];
        total_images += bs;

        let start = Instant::now();

        // Input: [batch, 3, 224, 224]
        let img = Tensor::randn(&[bs, 3, 224, 224], (Kind::Float, device));

        // Forward through CNN
        let x = conv_block(&img, &c1_w, &c1_b, &bn1_w, &bn1_b, &bn1_m, &bn1_v, true);
        let x = conv_block(&x, &c2_w, &c2_b, &bn2_w, &bn2_b, &bn2_m, &bn2_v, true);
        let x = conv_block(&x, &c3_w, &c3_b, &bn3_w, &bn3_b, &bn3_m, &bn3_v, true);

        // Classify
        let x = x.view([bs, -1]);
        let logits = x.matmul(&fc_w) + &fc_b;
        let probs = logits.softmax(-1, Kind::Float);

        // Extract prediction for first image in batch
        let first_probs = probs.get(0);
        let predicted_class = first_probs.argmax(0, false).int64_value(&[]);
        let confidence = first_probs.max().double_value(&[]);

        let lat = start.elapsed();
        latencies.push((bs, lat.as_micros() as f64));

        if i % 50 == 0 {
            println!("  batch {:>4} | bs={:>2} | {:.0}us | class={} conf={:.4} | frag={:.6}",
                     i, bs, lat.as_micros(), predicted_class, confidence,
                     get_fragmentation());
        }
    }

    // --- Throughput report ---
    let total_us: f64 = latencies.iter().map(|(_, l)| l).sum();
    let img_per_sec = total_images as f64 / (total_us / 1_000_000.0);

    println!("\n--- Throughput Report ---");
    println!("  Total images:   {}", total_images);
    println!("  Total time:     {:.1} ms", total_us / 1000.0);
    println!("  Throughput:     {:.0} images/sec", img_per_sec);
    println!("  Fragmentation:  {:.6}", get_fragmentation());

    // Per batch-size breakdown
    println!("\n--- Per Batch Size ---");
    for &bs in &[1i64, 4, 8, 16, 32] {
        let samples: Vec<f64> = latencies.iter()
            .filter(|(b, _)| *b == bs)
            .map(|(_, l)| *l)
            .collect();
        if !samples.is_empty() {
            let avg = samples.iter().sum::<f64>() / samples.len() as f64;
            println!("  bs={:>2}: {:.0}us avg, {:.0}us/image", bs, avg, avg / bs as f64);
        }
    }

    println!("\n--- Allocator State ---");
    print_stats();

    let active = check_leaks();
    println!("Active allocations: {} (CNN weights, expected)", active);

    println!("\nKernels exercised on TLSF:");
    println!("  - conv2d (cuDNN forward convolution, 3 layers)");
    println!("  - batch_norm (running mean/var, 3 layers)");
    println!("  - relu (element-wise, 3 layers)");
    println!("  - max_pool2d (2x2 stride 2, 3 layers)");
    println!("  - matmul (fully-connected classifier)");
    println!("  - softmax (class probabilities)");
    println!("  Every feature map allocated/freed through TLSF.\n");

    Ok(())
}
