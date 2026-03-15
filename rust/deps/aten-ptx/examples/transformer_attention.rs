//! Transformer Inference on TLSF Allocator
//!
//! Real multi-head self-attention + FFN running entirely on TLSF.
//! Simulates a production inference server processing variable-length requests.
//!
//! Kernels exercised:
//!   - Linear projections (matmul) for Q/K/V/O
//!   - Scaled dot-product attention (matmul + softmax)
//!   - LayerNorm with learned parameters
//!   - GELU activation
//!   - Residual connections
//!
//! Every intermediate buffer is allocated/freed through TLSF.

use aten_ptx::{init_pytorch_tlsf, print_stats, get_fragmentation, check_leaks};
use tch::{Device, Tensor, Kind};
use anyhow::Result;
use std::time::Instant;

const HIDDEN: i64 = 768;
const HEADS: i64 = 12;
const HEAD_DIM: i64 = HIDDEN / HEADS; // 64
const FFN: i64 = 3072; // 4x hidden

/// Multi-head self-attention (real Q/K/V/O projections + scaled dot-product)
fn attention(
    x: &Tensor,
    wq: &Tensor, wk: &Tensor, wv: &Tensor, wo: &Tensor,
    batch: i64, seq: i64,
) -> Tensor {
    let q = x.matmul(wq);
    let k = x.matmul(wk);
    let v = x.matmul(wv);

    // [batch, seq, hidden] -> [batch, heads, seq, head_dim]
    let q = q.view([batch, seq, HEADS, HEAD_DIM]).permute(&[0, 2, 1, 3]);
    let k = k.view([batch, seq, HEADS, HEAD_DIM]).permute(&[0, 2, 1, 3]);
    let v = v.view([batch, seq, HEADS, HEAD_DIM]).permute(&[0, 2, 1, 3]);

    // Scaled dot-product attention
    let scale = (HEAD_DIM as f64).sqrt();
    let scores = q.matmul(&k.transpose(-2, -1)) / scale;
    let weights = scores.softmax(-1, Kind::Float);
    let out = weights.matmul(&v);

    // [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    let out = out.permute(&[0, 2, 1, 3]).contiguous().view([batch, seq, HIDDEN]);

    // Output projection
    out.matmul(wo)
}

/// Feed-forward: linear -> GELU -> linear
fn ffn(x: &Tensor, w1: &Tensor, w2: &Tensor) -> Tensor {
    x.matmul(w1).gelu("none").matmul(w2)
}

/// Full transformer block: pre-norm attention + pre-norm FFN with residuals
fn transformer_block(
    x: &Tensor,
    wq: &Tensor, wk: &Tensor, wv: &Tensor, wo: &Tensor,
    ff1: &Tensor, ff2: &Tensor,
    ln1_w: &Tensor, ln1_b: &Tensor,
    ln2_w: &Tensor, ln2_b: &Tensor,
    batch: i64, seq: i64,
) -> Tensor {
    // Pre-norm attention with residual
    let normed = x.layer_norm(&[HIDDEN], Some(ln1_w), Some(ln1_b), 1e-5, true);
    let attn = attention(&normed, wq, wk, wv, wo, batch, seq);
    let x = x + attn;

    // Pre-norm FFN with residual
    let normed = x.layer_norm(&[HIDDEN], Some(ln2_w), Some(ln2_b), 1e-5, true);
    let ff = ffn(&normed, ff1, ff2);
    x + ff
}

fn main() -> Result<()> {
    println!("\n=== Transformer Inference on TLSF Allocator ===\n");
    println!("Multi-head attention + FFN block, variable sequence lengths");
    println!("Config: hidden={}, heads={}, ffn={}\n", HIDDEN, HEADS, FFN);

    init_pytorch_tlsf(0, 0.70).map_err(|e| anyhow::anyhow!("{}", e))?;

    let device = Device::Cuda(0);
    let _guard = tch::no_grad_guard();

    // Initialize transformer weights (random, but all kernels are real)
    println!("Loading transformer weights...");
    let wq = Tensor::randn(&[HIDDEN, HIDDEN], (Kind::Float, device)) * 0.02;
    let wk = Tensor::randn(&[HIDDEN, HIDDEN], (Kind::Float, device)) * 0.02;
    let wv = Tensor::randn(&[HIDDEN, HIDDEN], (Kind::Float, device)) * 0.02;
    let wo = Tensor::randn(&[HIDDEN, HIDDEN], (Kind::Float, device)) * 0.02;
    let ff1 = Tensor::randn(&[HIDDEN, FFN], (Kind::Float, device)) * 0.02;
    let ff2 = Tensor::randn(&[FFN, HIDDEN], (Kind::Float, device)) * 0.02;
    let ln1_w = Tensor::ones(&[HIDDEN], (Kind::Float, device));
    let ln1_b = Tensor::zeros(&[HIDDEN], (Kind::Float, device));
    let ln2_w = Tensor::ones(&[HIDDEN], (Kind::Float, device));
    let ln2_b = Tensor::zeros(&[HIDDEN], (Kind::Float, device));

    println!("  10 weight tensors loaded via TLSF");
    print_stats();

    // --- Variable-length inference (simulates real serving) ---
    let seq_lengths: Vec<i64> = vec![32, 128, 64, 256, 16, 512, 48, 192, 384, 96];
    let num_requests = 500;
    let mut latencies_us: Vec<f64> = Vec::with_capacity(num_requests);

    println!("\nProcessing {} requests with variable sequence lengths...\n", num_requests);

    for i in 0..num_requests {
        let seq = seq_lengths[i % seq_lengths.len()];
        let batch: i64 = 1;

        let start = Instant::now();

        let input = Tensor::randn(&[batch, seq, HIDDEN], (Kind::Float, device));

        let output = transformer_block(
            &input, &wq, &wk, &wv, &wo,
            &ff1, &ff2,
            &ln1_w, &ln1_b, &ln2_w, &ln2_b,
            batch, seq,
        );

        let out_mean = output.mean(Kind::Float).double_value(&[]);
        let out_std = output.std(true).double_value(&[]);
        let latency = start.elapsed();
        latencies_us.push(latency.as_micros() as f64);

        if i % 100 == 0 {
            println!("  req {:>4} | seq_len={:>3} | latency={:>8.0}us | out mean={:>8.5} std={:>7.5} | frag={:.6}",
                     i, seq, latency.as_micros(), out_mean, out_std, get_fragmentation());
        }
    }

    // --- Latency report ---
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies_us[latencies_us.len() / 2];
    let p90 = latencies_us[(latencies_us.len() as f64 * 0.90) as usize];
    let p99 = latencies_us[(latencies_us.len() as f64 * 0.99) as usize];
    let avg = latencies_us.iter().sum::<f64>() / latencies_us.len() as f64;

    println!("\n--- Latency Report ({} requests) ---", num_requests);
    println!("  avg:  {:>10.0} us", avg);
    println!("  p50:  {:>10.0} us", p50);
    println!("  p90:  {:>10.0} us", p90);
    println!("  p99:  {:>10.0} us", p99);
    println!("  frag: {:.6}", get_fragmentation());

    // --- Final state ---
    println!("\n--- Allocator State ---");
    print_stats();

    let active = check_leaks();
    println!("Active allocations: {} (model weights, expected)", active);

    println!("\nKernels exercised on TLSF:");
    println!("  - matmul (Q/K/V/O projections, attention scores, FFN)");
    println!("  - softmax (attention weights)");
    println!("  - layer_norm (pre-norm with learned params)");
    println!("  - gelu (FFN activation)");
    println!("  - add (residual connections)");
    println!("  - view/permute/contiguous (head reshape)");
    println!("  All intermediates allocated and freed via TLSF.\n");

    Ok(())
}
