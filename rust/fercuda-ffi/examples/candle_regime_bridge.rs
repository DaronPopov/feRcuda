#[cfg(feature = "candle")]
fn main() -> anyhow::Result<()> {
    use candle_core::{Device, Tensor};
    use fercuda_ffi::{
        download_tensor_f32, upload_tensor_f32, BufferDesc, BufferDType, MatmulRequest, MemoryRegime,
        PoolConfig, Session,
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

    let (a_id, _) = upload_tensor_f32(&session, &a, false, 0)?;
    let (b_id, _) = upload_tensor_f32(&session, &b, false, 0)?;
    let out_desc = BufferDesc::new(BufferDType::F32, 2, [2, 2, 0, 0], false, 0);
    let out_id = session.alloc_buffer(out_desc)?;

    let job = session.submit_matmul(MatmulRequest::auto(a_id, b_id, out_id))?;
    session.job_wait(job)?;

    let out = download_tensor_f32(&session, out_id, &[2, 2])?;
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
