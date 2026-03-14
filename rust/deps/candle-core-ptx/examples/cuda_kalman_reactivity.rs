use std::env;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use ptx_os::{AllocMode, RegimeConfig, RegimeRuntimeCore};

unsafe extern "C" {
    fn dup(fd: i32) -> i32;
    fn dup2(oldfd: i32, newfd: i32) -> i32;
    fn close(fd: i32) -> i32;
    fn open(pathname: *const i8, flags: i32) -> i32;
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
}

const O_WRONLY: i32 = 1;

struct StdIoSilencer {
    saved_out: i32,
    saved_err: i32,
}

impl StdIoSilencer {
    fn new() -> Result<Self> {
        let devnull = b"/dev/null\0";
        // SAFETY: C FFI calls with valid args.
        let (saved_out, saved_err, null_fd) = unsafe {
            let so = dup(1);
            let se = dup(2);
            let nf = open(devnull.as_ptr() as *const i8, O_WRONLY);
            (so, se, nf)
        };
        if saved_out < 0 || saved_err < 0 || null_fd < 0 {
            return Err(anyhow!("failed to silence stdio"));
        }
        // SAFETY: Redirect stdout/stderr to /dev/null.
        unsafe {
            if dup2(null_fd, 1) < 0 || dup2(null_fd, 2) < 0 {
                close(null_fd);
                return Err(anyhow!("failed to redirect stdio"));
            }
            close(null_fd);
        }
        Ok(Self { saved_out, saved_err })
    }

    fn emit_line(&self, line: &str) {
        let mut s = line.as_bytes().to_vec();
        s.push(b'\n');
        // SAFETY: Writing bytes to a valid duplicated stdout fd.
        unsafe {
            let _ = write(self.saved_out, s.as_ptr(), s.len());
        }
    }
}

impl Drop for StdIoSilencer {
    fn drop(&mut self) {
        // SAFETY: Restoring previously duplicated fds.
        unsafe {
            let _ = dup2(self.saved_out, 1);
            let _ = dup2(self.saved_err, 2);
            let _ = close(self.saved_out);
            let _ = close(self.saved_err);
        }
    }
}

#[derive(Clone, Copy)]
struct SensorSample {
    accel_mps2: f32,
    alt_m: f32,
}

#[derive(Clone, Copy)]
struct TruthState {
    alt_m: f32,
    vel_mps: f32,
}

#[derive(Clone, Copy)]
struct Config {
    hz: f32,
    seconds: f32,
    realtime: bool,
}

fn parse_args() -> Result<Config> {
    let mut hz = 200.0f32;
    let mut seconds = 5.0f32;
    let mut realtime = false;

    let args: Vec<String> = env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--hz" => {
                i += 1;
                if i >= args.len() {
                    return Err(anyhow!("--hz requires a value"));
                }
                hz = args[i].parse::<f32>()?;
            }
            "--seconds" => {
                i += 1;
                if i >= args.len() {
                    return Err(anyhow!("--seconds requires a value"));
                }
                seconds = args[i].parse::<f32>()?;
            }
            "--realtime" => realtime = true,
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --example cuda_kalman_reactivity -- [--hz N] [--seconds N] [--realtime]"
                );
                std::process::exit(0);
            }
            other => return Err(anyhow!("unknown arg: {other}")),
        }
        i += 1;
    }

    if hz <= 0.0 {
        return Err(anyhow!("--hz must be > 0"));
    }
    if seconds <= 0.0 {
        return Err(anyhow!("--seconds must be > 0"));
    }
    Ok(Config {
        hz,
        seconds,
        realtime,
    })
}

fn percentile_sorted(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    if v.len() == 1 {
        return v[0];
    }
    let x = (v.len() - 1) as f64 * p;
    let lo = x.floor() as usize;
    let hi = x.ceil() as usize;
    if lo == hi {
        return v[lo];
    }
    let frac = x - lo as f64;
    v[lo] * (1.0 - frac) + v[hi] * frac
}

fn rand_lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let v = ((*seed >> 32) as u32) as f64 / u32::MAX as f64;
    v as f32
}

fn noise(seed: &mut u64, scale: f32) -> f32 {
    // Uniform approx [-scale, scale]
    (rand_lcg(seed) * 2.0 - 1.0) * scale
}

fn sensor_model(truth: &TruthState, t: f32, seed: &mut u64) -> SensorSample {
    let a_truth = 1.2 * (2.0 * std::f32::consts::PI * 0.35 * t).sin()
        + 0.25 * (2.0 * std::f32::consts::PI * 1.8 * t).sin();
    let accel_mps2 = a_truth + noise(seed, 0.05);
    let alt_m = truth.alt_m + noise(seed, 0.35);
    SensorSample { accel_mps2, alt_m }
}

fn t_from(device: &Device, data: &[f32], shape: (usize, usize)) -> Result<Tensor> {
    Ok(Tensor::from_vec(data.to_vec(), shape, device)?.to_dtype(DType::F32)?)
}

fn scalar_from(device: &Device, v: f32) -> Result<Tensor> {
    Ok(Tensor::from_vec(vec![v], (1, 1), device)?.to_dtype(DType::F32)?)
}

fn build_regime_config() -> RegimeConfig {
    RegimeConfig::new()
        .pool_fraction(0.85)
        .quiet()
        .alloc_mode(AllocMode::CpuLike)
}

fn pool_mode_label(regime: &RegimeConfig) -> String {
    let cfg = regime.runtime_config();
    if cfg.pool_fraction > 0.0 {
        format!("frac:{:.2}", cfg.pool_fraction)
    } else {
        format!("fixed:{}MB", cfg.fixed_pool_size / (1024 * 1024))
    }
}

fn main() -> Result<()> {
    let cfg = parse_args()?;
    let dt = 1.0f32 / cfg.hz;
    let steps = (cfg.seconds * cfg.hz).max(1.0) as usize;

    // Silence native runtime banners for this benchmark output.
    let silencer = StdIoSilencer::new()?;
    let regime = build_regime_config();
    let runtime = RegimeRuntimeCore::with_regime(0, regime.clone())?;
    let before = runtime.pool_stats();

    let device = Device::new_cuda(0)?;

    // Constant matrices for 2-state (alt, vel) 1D Kalman.
    let f = t_from(&device, &[1.0, dt, 0.0, 1.0], (2, 2))?;
    let ft = t_from(&device, &[1.0, 0.0, dt, 1.0], (2, 2))?;
    let b = t_from(&device, &[0.5 * dt * dt, dt], (2, 1))?;
    let h = t_from(&device, &[1.0, 0.0], (1, 2))?;
    let ht = t_from(&device, &[1.0, 0.0], (2, 1))?;
    let i2 = t_from(&device, &[1.0, 0.0, 0.0, 1.0], (2, 2))?;

    let q = t_from(&device, &[1e-4, 0.0, 0.0, 1e-3], (2, 2))?;
    let r = scalar_from(&device, 0.35f32 * 0.35f32)?;

    let mut x = t_from(&device, &[0.0, 0.0], (2, 1))?;
    let mut p = t_from(&device, &[1.0, 0.0, 0.0, 1.0], (2, 2))?;

    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut truth = TruthState { alt_m: 0.0, vel_mps: 0.0 };
    let mut t = 0.0f32;

    let mut loop_us: Vec<f64> = Vec::with_capacity(steps);
    let mut phase_us: Vec<f64> = Vec::with_capacity(steps);
    let mut abs_err_m: Vec<f64> = Vec::with_capacity(steps);
    let mut deadline_misses = 0usize;

    let wall_start = Instant::now();
    let mut next_tick = Instant::now();

    for _ in 0..steps {
        if cfg.realtime {
            let now = Instant::now();
            if now < next_tick {
                std::thread::sleep(next_tick - now);
            } else {
                deadline_misses += 1;
            }
        }

        let loop_start = Instant::now();
        let phase_err = loop_start.saturating_duration_since(next_tick).as_secs_f64() * 1e6;
        phase_us.push(phase_err);

        // Update truth dynamics.
        let a_truth = 1.2 * (2.0 * std::f32::consts::PI * 0.35 * t).sin()
            + 0.25 * (2.0 * std::f32::consts::PI * 1.8 * t).sin();
        truth.alt_m += truth.vel_mps * dt + 0.5 * a_truth * dt * dt;
        truth.vel_mps += a_truth * dt;

        // Poll sensor values (simulated).
        let s = sensor_model(&truth, t, &mut seed);

        // Kalman predict/update on GPU tensors.
        if regime.alloc_mode_value() == AllocMode::CpuLike {
            let p = runtime.alloc_raw(4096)?;
            // SAFETY: pointer came from this runtime in the same scope.
            unsafe { runtime.free_raw(p) };
        }

        let u = scalar_from(&device, s.accel_mps2)?;
        let z = scalar_from(&device, s.alt_m)?;

        let x_pred = f.matmul(&x)?.add(&b.matmul(&u)?)?;
        let p_pred = f.matmul(&p)?.matmul(&ft)?.add(&q)?;
        let y = z.sub(&h.matmul(&x_pred)?)?;
        let s_cov = h.matmul(&p_pred)?.matmul(&ht)?.add(&r)?;
        let s_scalar = s_cov.flatten_all()?.to_vec1::<f32>()?[0];
        let inv_s = scalar_from(&device, 1.0f32 / s_scalar.max(1e-9))?;
        let k = p_pred.matmul(&ht)?.matmul(&inv_s)?;
        x = x_pred.add(&k.matmul(&y)?)?;
        p = i2.sub(&k.matmul(&h)?)?.matmul(&p_pred)?;

        device.synchronize()?;

        let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
        abs_err_m.push((x_vec[0] - truth.alt_m).abs() as f64);

        loop_us.push(loop_start.elapsed().as_secs_f64() * 1e6);
        t += dt;
        next_tick += Duration::from_secs_f64(dt as f64);
    }

    let wall_s = wall_start.elapsed().as_secs_f64();
    let mut loop_sorted = loop_us.clone();
    let mut phase_sorted = phase_us.clone();
    let mut err_sorted = abs_err_m.clone();
    loop_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    phase_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    err_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let after = runtime.pool_stats();
    let alloc_delta = after.total_allocations.saturating_sub(before.total_allocations);
    let free_delta = after.total_frees.saturating_sub(before.total_frees);

    let achieved_hz = steps as f64 / wall_s.max(1e-9);
    let loop_p99_us = percentile_sorted(&loop_sorted, 0.99);
    let deadline_pct = 100.0 * deadline_misses as f64 / steps as f64;

    if alloc_delta == 0 {
        silencer.emit_line("kalman_reactivity: FAIL");
        silencer.emit_line("reason=no_ptx_allocations_observed");
        std::process::exit(1);
    }
    silencer.emit_line("kalman_reactivity: PASS");
    silencer.emit_line(&format!(
        "target_hz={:.1} achieved_hz={:.2} loop_p99_us={:.1} deadline_miss_pct={:.2} pool_mode={} alloc_mode={}",
        cfg.hz,
        achieved_hz,
        loop_p99_us,
        deadline_pct,
        pool_mode_label(&regime),
        match regime.alloc_mode_value() {
            AllocMode::CpuLike => "cpu_like",
            AllocMode::SessionBuffers => "session_buffers",
        }
    ));
    silencer.emit_line(&format!(
        "ptx_alloc_delta={} ptx_free_delta={}",
        alloc_delta, free_delta
    ));
    // Exit immediately with silencing still active to avoid teardown banner noise.
    std::process::exit(0);
}
