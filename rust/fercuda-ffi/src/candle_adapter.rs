use crate::{BufferDesc, BufferDType, BufferId, Error, Session};
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

pub fn upload_tensor_f32(
    session: &Session,
    tensor: &Tensor,
    immutable: bool,
    tag: u32,
) -> Result<(BufferId, BufferDesc), CandleAdapterError> {
    let t = tensor.to_dtype(DType::F32)?;
    let dims = t.dims().to_vec();
    let desc = dims_to_desc(&dims, immutable, tag)?;
    let flat = t.flatten_all()?.to_vec1::<f32>()?;
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
