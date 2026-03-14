//! Finance Math Benchmark on PTX-OS
//!
//! Tests real-world financial computations using Candle tensors with CUDA,
//! while demonstrating PTX-OS's O(1) TLSF allocator for memory management.
//!
//! Run: cargo run --example finance_bench --release

use candle_core::{Device, Result, Tensor};
use candle_ptx_os::PtxDevice;
use std::time::Instant;

/// Moving Average calculation (SMA)
fn simple_moving_average(prices: &Tensor, window: usize) -> Result<Tensor> {
    let n = prices.dim(0)?;
    if n < window {
        return Err(candle_core::Error::Msg("Not enough data for window".into()));
    }

    // Manual sliding window average
    let mut sma_values = Vec::new();
    for i in 0..=(n - window) {
        let window_slice = prices.narrow(0, i, window)?;
        let mean = window_slice.mean_all()?.to_scalar::<f32>()?;
        sma_values.push(mean);
    }

    Tensor::from_vec(sma_values, (n - window + 1,), prices.device())
}

/// Exponential Moving Average
fn exponential_moving_average(prices: &Tensor, alpha: f32) -> Result<Tensor> {
    let n = prices.dim(0)?;
    let prices_vec: Vec<f32> = prices.to_vec1()?;

    let mut ema = Vec::with_capacity(n);
    ema.push(prices_vec[0]);

    for i in 1..n {
        let new_ema = alpha * prices_vec[i] + (1.0 - alpha) * ema[i - 1];
        ema.push(new_ema);
    }

    Tensor::from_vec(ema, (n,), prices.device())
}

/// Calculate returns from prices
fn calculate_returns(prices: &Tensor) -> Result<Tensor> {
    let n = prices.dim(0)?;
    let prices_t0 = prices.narrow(0, 0, n - 1)?;
    let prices_t1 = prices.narrow(0, 1, n - 1)?;

    // Returns = (P1 - P0) / P0
    let diff = prices_t1.sub(&prices_t0)?;
    diff.div(&prices_t0)
}

/// Historical Volatility (annualized)
fn historical_volatility(returns: &Tensor, trading_days: f32) -> Result<f32> {
    let mean = returns.mean_all()?.to_scalar::<f32>()?;
    let centered = returns.affine(1.0, -mean as f64)?;
    let squared = centered.mul(&centered)?;
    let variance = squared.mean_all()?.to_scalar::<f32>()?;
    let daily_vol = variance.sqrt();
    Ok(daily_vol * (trading_days).sqrt())
}

/// Covariance matrix for portfolio analysis
fn covariance_matrix(returns_matrix: &Tensor) -> Result<Tensor> {
    // returns_matrix: [num_assets, num_periods]
    let (n_assets, n_periods) = returns_matrix.dims2()?;

    // Center the data
    let means = returns_matrix.mean(1)?;
    let means_expanded = means.reshape((n_assets, 1))?;
    let centered = returns_matrix.broadcast_sub(&means_expanded)?;

    // Covariance = (X * X^T) / (n - 1)
    let centered_t = centered.t()?;
    let cov = centered.matmul(&centered_t)?;
    cov.affine(1.0 / (n_periods as f64 - 1.0), 0.0)
}

/// Black-Scholes Option Pricing (simplified - call option)
fn black_scholes_call(
    spot: &Tensor,      // Current stock prices
    strike: &Tensor,    // Strike prices
    time: f32,          // Time to expiration (years)
    rate: f32,          // Risk-free rate
    volatility: f32,    // Volatility
) -> Result<Tensor> {
    // d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
    // d2 = d1 - σ√T
    // Call = S*N(d1) - K*e^(-rT)*N(d2)

    let sqrt_t = time.sqrt();
    let vol_sqrt_t = volatility * sqrt_t;

    // ln(S/K)
    let s_over_k = spot.div(strike)?;
    let ln_s_k = s_over_k.log()?;

    // (r + σ²/2)T
    let drift = (rate + 0.5 * volatility * volatility) * time;

    // d1 = (ln(S/K) + drift) / (σ√T)
    let d1 = ln_s_k.affine(1.0 / vol_sqrt_t as f64, drift as f64 / vol_sqrt_t as f64)?;

    // d2 = d1 - σ√T
    let d2 = d1.affine(1.0, -(vol_sqrt_t as f64))?;

    // Approximate N(x) using tanh for GPU efficiency
    // N(x) ≈ 0.5 * (1 + tanh(0.8 * x))
    let n_d1 = d1.affine(0.8, 0.0)?.tanh()?.affine(0.5, 0.5)?;
    let n_d2 = d2.affine(0.8, 0.0)?.tanh()?.affine(0.5, 0.5)?;

    // Call = S*N(d1) - K*e^(-rT)*N(d2)
    let discount = (-rate * time).exp();
    let term1 = spot.mul(&n_d1)?;
    let term2 = strike.affine(discount as f64, 0.0)?.mul(&n_d2)?;

    term1.sub(&term2)
}

/// Monte Carlo simulation for option pricing
fn monte_carlo_option(
    device: &Device,
    spot: f32,
    strike: f32,
    time: f32,
    rate: f32,
    volatility: f32,
    num_paths: usize,
    num_steps: usize,
) -> Result<f32> {
    let dt = time / num_steps as f32;
    let sqrt_dt = dt.sqrt();
    let drift = (rate - 0.5 * volatility * volatility) * dt;

    // Generate random numbers for all paths and steps
    let randoms = Tensor::randn(0.0f32, 1.0, (num_paths, num_steps), device)?;

    // Simulate paths - start with spot price
    let mut prices = Tensor::full(spot, (num_paths,), device)?;

    for step in 0..num_steps {
        let random_step = randoms.narrow(1, step, 1)?.squeeze(1)?;
        let shock = random_step.affine(volatility as f64 * sqrt_dt as f64, drift as f64)?;
        let multiplier = shock.exp()?;
        prices = prices.mul(&multiplier)?;
    }

    // Calculate payoffs
    let strike_tensor = Tensor::full(strike, (num_paths,), device)?;
    let payoffs = prices.sub(&strike_tensor)?.maximum(0.0)?;

    // Discount and average
    let discount = (-rate * time).exp();
    let mean_payoff = payoffs.mean_all()?.to_scalar::<f32>()?;

    Ok(mean_payoff * discount)
}

/// VaR (Value at Risk) calculation
fn value_at_risk(returns: &Tensor, confidence: f32) -> Result<f32> {
    // Move to CPU for sort (CUDA sort has limitations)
    let returns_cpu = returns.to_device(&Device::Cpu)?;
    let sorted = returns_cpu.sort_last_dim(true)?.0;
    let n = sorted.dim(0)?;
    let index = ((1.0 - confidence) * n as f32) as usize;
    sorted.narrow(0, index, 1)?.squeeze(0)?.to_scalar::<f32>()
}

fn format_duration(d: std::time::Duration) -> String {
    if d.as_micros() < 1000 {
        format!("{} μs", d.as_micros())
    } else if d.as_millis() < 1000 {
        format!("{:.2} ms", d.as_micros() as f64 / 1000.0)
    } else {
        format!("{:.2} s", d.as_secs_f64())
    }
}

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     PTX-OS Finance Math Benchmark                         ║");
    println!("║     Real CUDA Kernels on Hot GPU Runtime                  ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Initialize PTX-OS with small 128MB pool for O(1) allocator overhead
    // The rest of VRAM is left free for Candle's CUDA backend
    let start = Instant::now();
    let ptx_device = PtxDevice::with_pool_size(0, 128 * 1024 * 1024)?; // 128 MB
    let ptx_init_time = start.elapsed();
    println!("PTX-OS device initialized in {}", format_duration(ptx_init_time));

    let stats = ptx_device.pool_stats();
    println!("TLSF Pool: {:.2} MB (lightweight manager, rest free for compute)",
        stats.total_size as f64 / 1024.0 / 1024.0
    );

    // Initialize standard Candle CUDA device for tensor operations
    let start = Instant::now();
    let device = Device::cuda_if_available(0)?;
    println!("Candle device: {:?} (init: {})", device, format_duration(start.elapsed()));
    println!();

    // Generate synthetic market data - LARGE SCALE
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Generating LARGE SCALE synthetic market data...");

    let num_days = 252 * 20; // 20 years of daily data
    let num_assets = 500;    // 500 assets for portfolio

    // Generate random walk prices
    let start = Instant::now();
    let returns_data: Vec<f32> = (0..num_days)
        .map(|_| (rand::random::<f32>() - 0.5) * 0.04) // ±2% daily
        .collect();

    let _returns = Tensor::from_vec(returns_data.clone(), (num_days,), &device)?;
    let prices = {
        let mut p = vec![100.0f32];
        for r in &returns_data {
            p.push(p.last().unwrap() * (1.0 + r));
        }
        Tensor::from_vec(p, (num_days + 1,), &device)?
    };
    println!("Generated {} days of price data in {}", num_days, format_duration(start.elapsed()));

    // Generate multi-asset returns matrix
    let multi_asset_data: Vec<f32> = (0..num_assets * num_days)
        .map(|_| (rand::random::<f32>() - 0.5) * 0.04)
        .collect();
    let multi_returns = Tensor::from_vec(multi_asset_data, (num_assets, num_days), &device)?;
    println!("Generated {}x{} multi-asset matrix", num_assets, num_days);
    println!();

    // Benchmark 1: Moving Averages
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Moving Averages");

    let start = Instant::now();
    let _sma_20 = simple_moving_average(&prices, 20)?;
    let sma_time = start.elapsed();
    println!("  SMA(20) computed in {}", format_duration(sma_time));

    let start = Instant::now();
    let _ema = exponential_moving_average(&prices, 0.1)?;
    let ema_time = start.elapsed();
    println!("  EMA(alpha=0.1) computed in {}", format_duration(ema_time));
    println!();

    // Benchmark 2: Volatility
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Volatility Analysis");

    let start = Instant::now();
    let calc_returns = calculate_returns(&prices)?;
    let returns_time = start.elapsed();

    let start = Instant::now();
    let vol = historical_volatility(&calc_returns, 252.0)?;
    let vol_time = start.elapsed();
    println!("  Returns calculated in {}", format_duration(returns_time));
    println!("  Annualized volatility: {:.2}% (computed in {})", vol * 100.0, format_duration(vol_time));
    println!();

    // Benchmark 3: Covariance Matrix
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Covariance Matrix ({}x{} assets)", num_assets, num_assets);

    let start = Instant::now();
    let cov = covariance_matrix(&multi_returns)?;
    let cov_time = start.elapsed();

    let cov_dims = cov.dims();
    println!("  {}x{} covariance matrix computed in {}",
        cov_dims[0], cov_dims[1], format_duration(cov_time));

    // Get a sample value (squeeze to get scalar from 1x1 tensor)
    let sample = cov.narrow(0, 0, 1)?.narrow(1, 0, 1)?.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?;
    println!("  Sample variance [0,0]: {:.6}", sample);
    println!();

    // Benchmark 4: Black-Scholes - LARGE SCALE
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Black-Scholes Option Pricing (LARGE SCALE)");

    let num_options = 10_000_000; // 10 million options
    let spots: Vec<f32> = (0..num_options).map(|i| 90.0 + (i as f32 / num_options as f32) * 20.0).collect();
    let strikes: Vec<f32> = vec![100.0; num_options];

    let spots_tensor = Tensor::from_vec(spots, (num_options,), &device)?;
    let strikes_tensor = Tensor::from_vec(strikes, (num_options,), &device)?;

    let start = Instant::now();
    let call_prices = black_scholes_call(
        &spots_tensor,
        &strikes_tensor,
        0.25,   // 3 months
        0.05,   // 5% risk-free rate
        0.20,   // 20% volatility
    )?;
    let bs_time = start.elapsed();

    let sample_price = call_prices.narrow(0, num_options / 2, 1)?.squeeze(0)?.to_scalar::<f32>()?;
    println!("  {} options priced in {}", num_options, format_duration(bs_time));
    println!("  Throughput: {:.0} options/second", num_options as f64 / bs_time.as_secs_f64());
    println!("  Sample (ATM) call price: ${:.2}", sample_price);
    println!();

    // Benchmark 5: Monte Carlo - LARGE SCALE
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Monte Carlo Option Pricing (LARGE SCALE)");

    let num_paths = 1_000_000;  // 1 million paths
    let num_steps = 252;        // Daily steps for 1 year

    let start = Instant::now();
    let mc_price = monte_carlo_option(
        &device,
        100.0,  // spot
        100.0,  // strike
        1.0,    // 1 year
        0.05,   // 5% rate
        0.20,   // 20% vol
        num_paths,
        num_steps,
    )?;
    let mc_time = start.elapsed();

    println!("  {} paths x {} steps in {}", num_paths, num_steps, format_duration(mc_time));
    println!("  Total simulations: {}", num_paths * num_steps);
    println!("  Throughput: {:.0} sims/second", (num_paths * num_steps) as f64 / mc_time.as_secs_f64());
    println!("  Monte Carlo call price: ${:.2}", mc_price);
    println!();

    // Benchmark 6: VaR
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Value at Risk");

    let start = Instant::now();
    let var_95 = value_at_risk(&calc_returns, 0.95)?;
    let var_99 = value_at_risk(&calc_returns, 0.99)?;
    let var_time = start.elapsed();

    println!("  VaR(95%): {:.2}%", var_95 * 100.0);
    println!("  VaR(99%): {:.2}%", var_99 * 100.0);
    println!("  Computed in {}", format_duration(var_time));
    println!();

    // Final stats
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PTX-OS Pool Statistics (O(1) TLSF Allocator)");
    let final_stats = ptx_device.pool_stats();
    println!("  Allocated: {:.2} MB", final_stats.allocated as f64 / 1024.0 / 1024.0);
    println!("  Free: {:.2} MB", final_stats.free as f64 / 1024.0 / 1024.0);
    println!("  Fragmentation: {:.1}%", final_stats.fragmentation_ratio * 100.0);
    println!("  Health: {}", if final_stats.is_healthy { "OK" } else { "DEGRADED" });

    let exec_stats = ptx_device.executor_stats();
    println!("  Graph cache: {} hits, {} misses",
        exec_stats.graph_cache_hits, exec_stats.graph_cache_misses);
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("Benchmark Complete");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
