//! Transformer Inference on TLSF — B200 demo
//!
//! Real multi-head self-attention + FFN running on feRcuda TLSF.
//! Variable-length requests, latency percentiles, fragmentation stats.
//!
//! Run: feRcuda-transformer (or cargo run --example transformer_demo)

use anyhow::Result;
use aten_ptx::{check_leaks, get_fragmentation, init_pytorch_tlsf, print_stats};
use std::time::Instant;
use tch::{Device, Kind, Tensor};

const HIDDEN: i64 = 384;
const HEADS: i64 = 6;
const HEAD_DIM: i64 = HIDDEN / HEADS;
const FFN: i64 = 1536;

fn attention(
    x: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    batch: i64,
    seq: i64,
) -> Tensor {
    let q = x.matmul(wq);
    let k = x.matmul(wk);
    let v = x.matmul(wv);

    let q = q.view([batch, seq, HEADS, HEAD_DIM]).permute(&[0, 2, 1, 3]);
    let k = k.view([batch, seq, HEADS, HEAD_DIM]).permute(&[0, 2, 1, 3]);
    let v = v.view([batch, seq, HEADS, HEAD_DIM]).permute(&[0, 2, 1, 3]);

    let scale = (HEAD_DIM as f64).sqrt();
    let scores = q.matmul(&k.transpose(-2, -1)) / scale;
    let weights = scores.softmax(-1, Kind::Float);
    let out = weights.matmul(&v);

    let out = out.permute(&[0, 2, 1, 3]).contiguous().view([batch, seq, HIDDEN]);
    out.matmul(wo)
}

fn ffn(x: &Tensor, w1: &Tensor, w2: &Tensor) -> Tensor {
    x.matmul(w1).gelu("none").matmul(w2)
}

fn transformer_block(
    x: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    ff1: &Tensor,
    ff2: &Tensor,
    ln1_w: &Tensor,
    ln1_b: &Tensor,
    ln2_w: &Tensor,
    ln2_b: &Tensor,
    batch: i64,
    seq: i64,
) -> Tensor {
    let normed = x.layer_norm(&[HIDDEN], Some(ln1_w), Some(ln1_b), 1e-5, true);
    let attn = attention(&normed, wq, wk, wv, wo, batch, seq);
    let x = x + attn;

    let normed = x.layer_norm(&[HIDDEN], Some(ln2_w), Some(ln2_b), 1e-5, true);
    let ff = ffn(&normed, ff1, ff2);
    x + ff
}

fn main() -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  feRcuda Transformer Demo: Multi-head attention + FFN on TLSF        ║");
    println!("║  Variable sequence lengths | Latency percentiles | B200-ready        ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!("\n  Config: hidden={}, heads={}, ffn={}\n", HIDDEN, HEADS, FFN);

    init_pytorch_tlsf(0, 0.70).map_err(anyhow::Error::msg)?;
    println!("✓ TLSF pool initialized (70% VRAM)\n");

    let device = Device::Cuda(0);
    let _guard = tch::no_grad_guard();

    println!("  Loading transformer weights...");
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
    println!("  ✓ 10 weight tensors loaded via TLSF\n");

    let seq_lengths: Vec<i64> = vec![32, 128, 64, 256, 16, 512, 48, 192];
    let num_requests = 300;
    let mut latencies_us: Vec<f64> = Vec::with_capacity(num_requests);

    println!("  Processing {} requests with variable sequence lengths...\n", num_requests);

    for i in 0..num_requests {
        let seq = seq_lengths[i % seq_lengths.len()];
        let batch: i64 = 1;

        let start = Instant::now();

        let input = Tensor::randn(&[batch, seq, HIDDEN], (Kind::Float, device));

        let output = transformer_block(
            &input,
            &wq,
            &wk,
            &wv,
            &wo,
            &ff1,
            &ff2,
            &ln1_w,
            &ln1_b,
            &ln2_w,
            &ln2_b,
            batch,
            seq,
        );

        let _out_mean = output.mean(Kind::Float).double_value(&[]);
        let latency = start.elapsed();
        latencies_us.push(latency.as_micros() as f64);

        if i % 75 == 0 {
            println!(
                "    req {:>4} | seq_len={:>3} | latency={:>8.0} µs | frag={:.6}",
                i,
                seq,
                latency.as_micros(),
                get_fragmentation()
            );
        }
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies_us[latencies_us.len() / 2];
    let p90 = latencies_us[(latencies_us.len() as f64 * 0.90) as usize];
    let p99 = latencies_us[(latencies_us.len() as f64 * 0.99) as usize];
    let avg = latencies_us.iter().sum::<f64>() / latencies_us.len() as f64;

    println!("\n  ═══════════════════════════════════════════════════════");
    println!("  LATENCY REPORT ({} requests)", num_requests);
    println!("  ═══════════════════════════════════════════════════════");
    println!("    avg:  {:>10.0} µs", avg);
    println!("    p50:  {:>10.0} µs", p50);
    println!("    p90:  {:>10.0} µs", p90);
    println!("    p99:  {:>10.0} µs", p99);
    println!("    frag: {:.6}", get_fragmentation());
    println!("  ═══════════════════════════════════════════════════════\n");

    if let Some(s) = aten_ptx::get_pool_stats() {
        println!("  TLSF pool: {:.2} GB | peak {:.2} MB | util {:.1}%",
            s.total_size as f64 / 1e9,
            s.peak_allocated as f64 / 1e6,
            s.utilization_percent);
    }
    println!("  Active allocations: {} (model weights)", check_leaks());
    println!("\n  ✅ Transformer inference on TLSF complete.");
    println!("  ✅ Kernels: matmul, softmax, layer_norm, gelu, add.\n");

    Ok(())
}
