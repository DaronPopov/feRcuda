use crate::{
    BufferDesc, BufferDType, BufferId, Error, LayerNormRequest, MatmulRequest, MemoryRegime, Session,
};
use crate::adapter_backend::AdapterExecutionBackend;
use candle_core::{DType, Device, Tensor};

#[derive(Debug, thiserror::Error)]
pub enum CandleAdapterError {
    #[error("ffi error: {0}")]
    Ffi(#[from] Error),
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("unsupported tensor rank: {0}, expected 1..=4")]
    Rank(usize),
    #[error("shape dimension too large at axis {axis}: {dim}")]
    DimTooLarge { axis: usize, dim: usize },
    #[error("matmul expects rank-2 tensors, got rank(a)={a_rank}, rank(b)={b_rank}")]
    MatmulRank { a_rank: usize, b_rank: usize },
    #[error("matmul shape mismatch: a={a_m}x{a_k}, b={b_k}x{b_n}")]
    MatmulShape {
        a_m: usize,
        a_k: usize,
        b_k: usize,
        b_n: usize,
    },
    #[error("layer_norm expects rank >= 1")]
    LayerNormRank,
}

#[derive(Debug, Clone, Copy)]
pub struct OpRunConfig {
    pub immutable_inputs: bool,
    pub tag: u32,
    pub memory_regime: u32,
}

impl Default for OpRunConfig {
    fn default() -> Self {
        Self {
            immutable_inputs: false,
            tag: 0,
            memory_regime: MemoryRegime::Auto as u32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CandleBuffer {
    pub id: BufferId,
    pub desc: BufferDesc,
    pub dims: Vec<usize>,
}

fn dims_to_desc(dims: &[usize], immutable: bool, tag: u32) -> Result<BufferDesc, CandleAdapterError> {
    if dims.is_empty() || dims.len() > 4 {
        return Err(CandleAdapterError::Rank(dims.len()));
    }
    let mut out_dims = [0u32; 4];
    for (i, d) in dims.iter().enumerate() {
        out_dims[i] = u32::try_from(*d).map_err(|_| CandleAdapterError::DimTooLarge {
            axis: i,
            dim: *d,
        })?;
    }
    Ok(BufferDesc::new(
        BufferDType::F32,
        dims.len() as u32,
        out_dims,
        immutable,
        tag,
    ))
}

fn tensor_to_f32_flat(tensor: &Tensor) -> Result<(Vec<usize>, Vec<f32>), CandleAdapterError> {
    let t = tensor.to_dtype(DType::F32)?;
    let dims = t.dims().to_vec();
    let flat = t.flatten_all()?.to_vec1::<f32>()?;
    Ok((dims, flat))
}

pub fn validate_matmul_dims(
    a_dims: &[usize],
    b_dims: &[usize],
) -> Result<(usize, usize, usize), CandleAdapterError> {
    if a_dims.len() != 2 || b_dims.len() != 2 {
        return Err(CandleAdapterError::MatmulRank {
            a_rank: a_dims.len(),
            b_rank: b_dims.len(),
        });
    }
    let (m, k_a) = (a_dims[0], a_dims[1]);
    let (k_b, n) = (b_dims[0], b_dims[1]);
    if k_a != k_b {
        return Err(CandleAdapterError::MatmulShape {
            a_m: m,
            a_k: k_a,
            b_k: k_b,
            b_n: n,
        });
    }
    Ok((m, k_a, n))
}

pub fn validate_layer_norm_dims(x_dims: &[usize]) -> Result<(), CandleAdapterError> {
    if x_dims.is_empty() {
        return Err(CandleAdapterError::LayerNormRank);
    }
    Ok(())
}

pub fn upload_tensor_f32(
    session: &Session,
    tensor: &Tensor,
    immutable: bool,
    tag: u32,
) -> Result<(BufferId, BufferDesc), CandleAdapterError> {
    let (dims, flat) = tensor_to_f32_flat(tensor)?;
    let desc = dims_to_desc(&dims, immutable, tag)?;
    let id = session.alloc_buffer(desc)?;
    if let Err(e) = session.upload_f32(id, &flat) {
        let _ = session.free_buffer(id);
        return Err(CandleAdapterError::Ffi(e));
    }
    Ok((id, desc))
}

pub fn download_tensor_f32(
    session: &Session,
    buffer_id: BufferId,
    dims: &[usize],
) -> Result<Tensor, CandleAdapterError> {
    let elem_count = dims.iter().product::<usize>();
    let mut data = vec![0.0f32; elem_count];
    session.download_f32(buffer_id, &mut data)?;
    Ok(Tensor::from_vec(data, dims.to_vec(), &Device::Cpu)?)
}

pub struct CandleSessionAdapter<'a> {
    session: &'a Session,
    cfg: OpRunConfig,
}

impl<'a> CandleSessionAdapter<'a> {
    pub fn new(session: &'a Session) -> Self {
        Self {
            session,
            cfg: OpRunConfig::default(),
        }
    }

    pub fn with_config(session: &'a Session, cfg: OpRunConfig) -> Self {
        Self { session, cfg }
    }

    pub fn upload(&self, tensor: &Tensor) -> Result<CandleBuffer, CandleAdapterError> {
        let (id, desc) = upload_tensor_f32(
            self.session,
            tensor,
            self.cfg.immutable_inputs,
            self.cfg.tag,
        )?;
        Ok(CandleBuffer {
            id,
            desc,
            dims: tensor.dims().to_vec(),
        })
    }

    pub fn free(&self, buffer: &CandleBuffer) -> Result<(), CandleAdapterError> {
        self.session.free_buffer(buffer.id)?;
        Ok(())
    }

    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, CandleAdapterError> {
        let (m, _k, n) = validate_matmul_dims(a.dims(), b.dims())?;

        let mut owned: Vec<BufferId> = Vec::new();
        let result = (|| {
            let a_buf = self.upload(a)?;
            owned.push(a_buf.id);
            let b_buf = self.upload(b)?;
            owned.push(b_buf.id);

            let out_desc = dims_to_desc(&[m, n], false, self.cfg.tag)?;
            let out_id = self.session.alloc_buffer(out_desc)?;
            owned.push(out_id);

            let job = self.session.submit_matmul(MatmulRequest {
                a: a_buf.id,
                b: b_buf.id,
                out: out_id,
                memory_regime: self.cfg.memory_regime,
            })?;
            self.session.job_wait(job)?;
            download_tensor_f32(self.session, out_id, &[m, n])
        })();

        for id in owned {
            let _ = self.session.free_buffer(id);
        }
        result
    }

    pub fn layer_norm(&self, x: &Tensor, eps: f32) -> Result<Tensor, CandleAdapterError> {
        validate_layer_norm_dims(x.dims())?;

        let dims = x.dims().to_vec();
        let mut owned: Vec<BufferId> = Vec::new();
        let result = (|| {
            let x_buf = self.upload(x)?;
            owned.push(x_buf.id);

            let out_desc = dims_to_desc(&dims, false, self.cfg.tag)?;
            let out_id = self.session.alloc_buffer(out_desc)?;
            owned.push(out_id);

            let job = self.session.submit_layer_norm(LayerNormRequest {
                x: x_buf.id,
                out: out_id,
                eps,
                memory_regime: self.cfg.memory_regime,
            })?;
            self.session.job_wait(job)?;
            download_tensor_f32(self.session, out_id, &dims)
        })();

        for id in owned {
            let _ = self.session.free_buffer(id);
        }
        result
    }
}

impl<'a> AdapterExecutionBackend for CandleSessionAdapter<'a> {
    type TensorHandle = Tensor;
    type BackendError = CandleAdapterError;

    fn upload_host_f32(
        &self,
        host: &[f32],
        dims: &[usize],
    ) -> Result<Self::TensorHandle, Self::BackendError> {
        Ok(Tensor::from_vec(host.to_vec(), dims.to_vec(), &Device::Cpu)?)
    }

    fn download_host_f32(
        &self,
        tensor: &Self::TensorHandle,
    ) -> Result<Vec<f32>, Self::BackendError> {
        Ok(tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
    }

    fn matmul(
        &self,
        a: &Self::TensorHandle,
        b: &Self::TensorHandle,
    ) -> Result<Self::TensorHandle, Self::BackendError> {
        CandleSessionAdapter::matmul(self, a, b)
    }

    fn layer_norm(
        &self,
        x: &Self::TensorHandle,
        eps: f32,
    ) -> Result<Self::TensorHandle, Self::BackendError> {
        CandleSessionAdapter::layer_norm(self, x, eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn dims_to_desc_checks_rank() {
        assert!(matches!(
            dims_to_desc(&[], false, 0),
            Err(CandleAdapterError::Rank(0))
        ));
        assert!(dims_to_desc(&[2, 3], false, 0).is_ok());
    }

    #[test]
    fn matmul_shape_validation_works() {
        let dev = Device::Cpu;
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &dev).expect("a");
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &dev).expect("b");

        let err = validate_matmul_dims(a.dims(), b.dims()).expect_err("shape mismatch");
        assert!(matches!(err, CandleAdapterError::MatmulShape { .. }));
    }

    #[test]
    fn tensor_flatten_roundtrip_cpu() {
        let dev = Device::Cpu;
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &dev).expect("tensor");
        let (dims, flat) = tensor_to_f32_flat(&t).expect("flatten");
        assert_eq!(dims, vec![2, 2]);
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn layer_norm_dim_validation_works() {
        assert!(validate_layer_norm_dims(&[16]).is_ok());
        let err = validate_layer_norm_dims(&[]).expect_err("rank 0 should fail");
        assert!(matches!(err, CandleAdapterError::LayerNormRank));
    }

    #[test]
    fn trait_impl_is_present() {
        fn assert_impl<T: AdapterExecutionBackend>() {}
        assert_impl::<CandleSessionAdapter<'static>>();
    }

}
