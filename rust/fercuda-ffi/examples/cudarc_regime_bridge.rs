#[cfg(feature = "cudarc")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fercuda_ffi::{
        CudarcOpRunConfig, CudarcSessionAdapter, MemoryRegime, PoolConfig, Session,
    };

    let cfg = PoolConfig {
        mutable_bytes: 64u64 << 20,
        immutable_bytes: 64u64 << 20,
        cuda_reserve: 0,
        verbose: 0,
        memory_regime: MemoryRegime::CudaMalloc as u32,
    };
    let session = Session::new(0, Some(cfg))?;

    let adapter = CudarcSessionAdapter::with_config(
        &session,
        0,
        CudarcOpRunConfig {
            tag: 0,
            memory_regime: MemoryRegime::Auto as u32,
        },
    )?;

    let a = adapter.upload_host_f32(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = adapter.upload_host_f32(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2])?;
    let out = adapter.matmul(&a, &b)?;
    let got = adapter.download_host_f32(&out)?;

    let expected = [19.0f32, 22.0, 43.0, 50.0];
    for (g, e) in got.iter().zip(expected.iter()) {
        if (g - e).abs() > 1e-4 {
            return Err(format!("mismatch: got {g}, expected {e}").into());
        }
    }

    println!("cudarc_regime_bridge: PASS");
    println!("result={got:?}");
    Ok(())
}

#[cfg(not(feature = "cudarc"))]
fn main() {
    println!("Enable feature `cudarc` to run this example.");
}
