use crate::{
    BufferDesc, BufferDType, BufferId, Error, LayerNormRequest, MatmulRequest, MemoryRegime, Session,
};
use crate::adapter_backend::AdapterExecutionBackend;
use cudarc::driver::{CudaDevice, CudaSlice, DriverError};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum CudarcAdapterError {
    #[error("ffi error: {0}")]
    Ffi(#[from] Error),
    #[error("cudarc driver error: {0}")]
    Driver(String),
    #[error("shape error: expected element count {expected}, got {got}")]
    ElemCountMismatch { expected: usize, got: usize },
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
    #[error("unsupported rank {rank}, expected 1..=4")]
    RankUnsupported { rank: usize },
    #[error("dimension too large at axis {axis}: {dim}")]
    DimTooLarge { axis: usize, dim: usize },
}

fn map_driver_err(e: DriverError) -> CudarcAdapterError {
    CudarcAdapterError::Driver(format!("{e:?}"))
}

#[derive(Debug, Clone, Copy)]
pub struct CudarcOpRunConfig {
    pub tag: u32,
    pub memory_regime: u32,
}

impl Default for CudarcOpRunConfig {
    fn default() -> Self {
        Self {
            tag: 0,
            memory_regime: MemoryRegime::Auto as u32,
        }
    }
}

pub struct CudarcTensor {
    pub data: CudaSlice<f32>,
    pub dims: Vec<usize>,
}

pub fn validate_matmul_dims_cudarc(
    a_dims: &[usize],
    b_dims: &[usize],
) -> Result<(usize, usize, usize), CudarcAdapterError> {
    if a_dims.len() != 2 || b_dims.len() != 2 {
        return Err(CudarcAdapterError::MatmulRank {
            a_rank: a_dims.len(),
            b_rank: b_dims.len(),
        });
    }
    let (m, k_a) = (a_dims[0], a_dims[1]);
    let (k_b, n) = (b_dims[0], b_dims[1]);
    if k_a != k_b {
        return Err(CudarcAdapterError::MatmulShape {
            a_m: m,
            a_k: k_a,
            b_k: k_b,
            b_n: n,
        });
    }
    Ok((m, k_a, n))
}

pub fn validate_layer_norm_dims_cudarc(x_dims: &[usize]) -> Result<(), CudarcAdapterError> {
    if x_dims.is_empty() {
        return Err(CudarcAdapterError::LayerNormRank);
    }
    Ok(())
}

fn dims_to_desc(dims: &[usize], tag: u32) -> Result<BufferDesc, CudarcAdapterError> {
    if dims.is_empty() || dims.len() > 4 {
        return Err(CudarcAdapterError::RankUnsupported { rank: dims.len() });
    }
    let mut out_dims = [0u32; 4];
    for (i, d) in dims.iter().enumerate() {
        out_dims[i] = u32::try_from(*d).map_err(|_| CudarcAdapterError::DimTooLarge {
            axis: i,
            dim: *d,
        })?;
    }
    Ok(BufferDesc::new(
        BufferDType::F32,
        dims.len() as u32,
        out_dims,
        false,
        tag,
    ))
}

pub struct CudarcSessionAdapter<'a> {
    session: &'a Session,
    dev: Arc<CudaDevice>,
    cfg: CudarcOpRunConfig,
}

impl<'a> CudarcSessionAdapter<'a> {
    pub fn new(session: &'a Session, device_ordinal: usize) -> Result<Self, CudarcAdapterError> {
        Ok(Self {
            session,
            dev: CudaDevice::new(device_ordinal).map_err(map_driver_err)?,
            cfg: CudarcOpRunConfig::default(),
        })
    }

    pub fn with_config(
        session: &'a Session,
        device_ordinal: usize,
        cfg: CudarcOpRunConfig,
    ) -> Result<Self, CudarcAdapterError> {
        Ok(Self {
            session,
            dev: CudaDevice::new(device_ordinal).map_err(map_driver_err)?,
            cfg,
        })
    }

    pub fn upload_host_f32(
        &self,
        host: &[f32],
        dims: &[usize],
    ) -> Result<CudarcTensor, CudarcAdapterError> {
        let expected = dims.iter().product::<usize>();
        if host.len() != expected {
            return Err(CudarcAdapterError::ElemCountMismatch {
                expected,
                got: host.len(),
            });
        }
        let data = self.dev.htod_copy(host.to_vec()).map_err(map_driver_err)?;
        Ok(CudarcTensor {
            data,
            dims: dims.to_vec(),
        })
    }

    pub fn download_host_f32(&self, t: &CudarcTensor) -> Result<Vec<f32>, CudarcAdapterError> {
        self.dev.dtoh_sync_copy(&t.data).map_err(map_driver_err)
    }

    pub fn matmul(&self, a: &CudarcTensor, b: &CudarcTensor) -> Result<CudarcTensor, CudarcAdapterError> {
        let (m, _k, n) = validate_matmul_dims_cudarc(&a.dims, &b.dims)?;

        let a_host = self.dev.dtoh_sync_copy(&a.data).map_err(map_driver_err)?;
        let b_host = self.dev.dtoh_sync_copy(&b.data).map_err(map_driver_err)?;

        let mut owned: Vec<BufferId> = Vec::new();
        let out_host = (|| {
            let a_desc = dims_to_desc(&a.dims, self.cfg.tag)?;
            let b_desc = dims_to_desc(&b.dims, self.cfg.tag)?;
            let out_desc = dims_to_desc(&[m, n], self.cfg.tag)?;

            let a_id = self.session.alloc_buffer(a_desc)?;
            owned.push(a_id);
            let b_id = self.session.alloc_buffer(b_desc)?;
            owned.push(b_id);
            let out_id = self.session.alloc_buffer(out_desc)?;
            owned.push(out_id);

            self.session.upload_f32(a_id, &a_host)?;
            self.session.upload_f32(b_id, &b_host)?;

            let job = self.session.submit_matmul(MatmulRequest {
                a: a_id,
                b: b_id,
                out: out_id,
                memory_regime: self.cfg.memory_regime,
            })?;
            self.session.job_wait(job)?;

            let mut out = vec![0.0f32; m * n];
            self.session.download_f32(out_id, &mut out)?;
            Ok::<Vec<f32>, CudarcAdapterError>(out)
        })();

        for id in owned {
            let _ = self.session.free_buffer(id);
        }

        let out = out_host?;
        let out_dev = self.dev.htod_copy(out).map_err(map_driver_err)?;
        Ok(CudarcTensor {
            data: out_dev,
            dims: vec![m, n],
        })
    }

    pub fn layer_norm(&self, x: &CudarcTensor, eps: f32) -> Result<CudarcTensor, CudarcAdapterError> {
        validate_layer_norm_dims_cudarc(&x.dims)?;
        let x_host = self.dev.dtoh_sync_copy(&x.data).map_err(map_driver_err)?;

        let mut owned: Vec<BufferId> = Vec::new();
        let out_host = (|| {
            let x_desc = dims_to_desc(&x.dims, self.cfg.tag)?;
            let out_desc = dims_to_desc(&x.dims, self.cfg.tag)?;

            let x_id = self.session.alloc_buffer(x_desc)?;
            owned.push(x_id);
            let out_id = self.session.alloc_buffer(out_desc)?;
            owned.push(out_id);

            self.session.upload_f32(x_id, &x_host)?;

            let job = self.session.submit_layer_norm(LayerNormRequest {
                x: x_id,
                out: out_id,
                eps,
                memory_regime: self.cfg.memory_regime,
            })?;
            self.session.job_wait(job)?;

            let mut out = vec![0.0f32; x_host.len()];
            self.session.download_f32(out_id, &mut out)?;
            Ok::<Vec<f32>, CudarcAdapterError>(out)
        })();

        for id in owned {
            let _ = self.session.free_buffer(id);
        }

        let out = out_host?;
        let out_dev = self.dev.htod_copy(out).map_err(map_driver_err)?;
        Ok(CudarcTensor {
            data: out_dev,
            dims: x.dims.clone(),
        })
    }
}

impl<'a> AdapterExecutionBackend for CudarcSessionAdapter<'a> {
    type TensorHandle = CudarcTensor;
    type BackendError = CudarcAdapterError;

    fn upload_host_f32(
        &self,
        host: &[f32],
        dims: &[usize],
    ) -> Result<Self::TensorHandle, Self::BackendError> {
        CudarcSessionAdapter::upload_host_f32(self, host, dims)
    }

    fn download_host_f32(
        &self,
        tensor: &Self::TensorHandle,
    ) -> Result<Vec<f32>, Self::BackendError> {
        CudarcSessionAdapter::download_host_f32(self, tensor)
    }

    fn matmul(
        &self,
        a: &Self::TensorHandle,
        b: &Self::TensorHandle,
    ) -> Result<Self::TensorHandle, Self::BackendError> {
        CudarcSessionAdapter::matmul(self, a, b)
    }

    fn layer_norm(
        &self,
        x: &Self::TensorHandle,
        eps: f32,
    ) -> Result<Self::TensorHandle, Self::BackendError> {
        CudarcSessionAdapter::layer_norm(self, x, eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_dims_validation() {
        let err = validate_matmul_dims_cudarc(&[2, 2], &[3, 2]).expect_err("shape mismatch");
        assert!(matches!(err, CudarcAdapterError::MatmulShape { .. }));
    }

    #[test]
    fn layer_norm_dims_validation() {
        assert!(validate_layer_norm_dims_cudarc(&[1]).is_ok());
        assert!(matches!(
            validate_layer_norm_dims_cudarc(&[]),
            Err(CudarcAdapterError::LayerNormRank)
        ));
    }

    #[test]
    fn dims_to_desc_validation() {
        assert!(dims_to_desc(&[2, 2], 0).is_ok());
        assert!(matches!(
            dims_to_desc(&[], 0),
            Err(CudarcAdapterError::RankUnsupported { .. })
        ));
    }

    #[test]
    fn trait_impl_is_present() {
        fn assert_impl<T: AdapterExecutionBackend>() {}
        assert_impl::<CudarcSessionAdapter<'static>>();
    }
}
