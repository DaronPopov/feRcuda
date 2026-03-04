#[cfg(feature = "candle")]
fn main() -> anyhow::Result<()> {
    use candle_core::{Device, Tensor};
    use fercuda_ffi::{
        CandleSessionAdapter, MemoryRegime, OpRunConfig, PoolConfig, Session,
    };

    let cfg = PoolConfig {
        mutable_bytes: 64u64 << 20,
        immutable_bytes: 64u64 << 20,
        cuda_reserve: 0,
        verbose: 0,
        memory_regime: MemoryRegime::CudaMalloc as u32,
    };
    let session = Session::new(0, Some(cfg))?;

    let dev = Device::Cpu;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &dev)?;
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (2, 2), &dev)?;
    let adapter = CandleSessionAdapter::with_config(
        &session,
        OpRunConfig {
            immutable_inputs: false,
            tag: 0,
            memory_regime: MemoryRegime::Auto as u32,
        },
    );
    let out = adapter.matmul(&a, &b)?;
    let got = out.flatten_all()?.to_vec1::<f32>()?;
    let expected = [19.0f32, 22.0, 43.0, 50.0];
    for (g, e) in got.iter().zip(expected.iter()) {
        anyhow::ensure!((g - e).abs() <= 1e-4, "mismatch: got {g}, expected {e}");
    }

    println!("candle_regime_bridge: PASS");
    println!("result={got:?}");
    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() {
    println!("Enable feature `candle` to run this example.");
}
