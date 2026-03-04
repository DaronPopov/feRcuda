use crate::{BufferDesc, BufferDType, BufferId, Error, Session};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] crate::ml::safetensors::SafeTensorError),
    #[error("ml-lite error: {0}")]
    MlLite(#[from] crate::ml::MlLiteError),
    #[error("ffi error: {0}")]
    Ffi(#[from] Error),
    #[error("unsupported dtype for tensor '{tensor}': {dtype}")]
    UnsupportedDType { tensor: String, dtype: String },
    #[error("rank too high for tensor '{tensor}': {rank} (max 4)")]
    RankTooHigh { tensor: String, rank: usize },
    #[error("dimension too large for tensor '{tensor}': {dim}")]
    DimTooLarge { tensor: String, dim: usize },
    #[error("invalid f32 byte length: {len}")]
    InvalidF32Bytes { len: usize },
    #[error("invalid f16 byte length: {len}")]
    InvalidF16Bytes { len: usize },
    #[error("invalid bf16 byte length: {len}")]
    InvalidBf16Bytes { len: usize },
    #[error("invalid q4 byte length: packed {packed_len} for elem_count {elem_count}")]
    InvalidQ4Bytes { packed_len: usize, elem_count: usize },
    #[error("byte size mismatch for tensor '{tensor}': expected {expected}, got {got}")]
    ByteSizeMismatch {
        tensor: String,
        expected: usize,
        got: usize,
    },
}

#[derive(Debug, Clone)]
pub struct LoadedTensor {
    pub name: String,
    pub buffer_id: BufferId,
    pub dtype: BufferDType,
    pub rank: u32,
    pub dims: [u32; 4],
    pub elem_count: usize,
    pub byte_len: usize,
}

#[derive(Debug, Clone, Default)]
pub struct LoadedModel {
    pub tensors: Vec<LoadedTensor>,
    pub by_name: HashMap<String, BufferId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// q in i8; x = q * scale
    Q8Symmetric,
    /// q in u8; x = (q - zero_point) * scale
    Q8Affine,
    /// 2x q4 packed per byte, signed nibble in [-8, 7]
    Q4SymmetricPacked,
    /// 2x q4 packed per byte, unsigned nibble in [0, 15]
    Q4AffinePacked,
}

#[derive(Debug, Clone, Copy)]
pub struct QuantParams {
    pub mode: QuantMode,
    pub scale: f32,
    pub zero_point: i32,
}

fn shape_to_rank_dims(name: &str, shape: &[usize]) -> Result<(u32, [u32; 4]), AdapterError> {
    let rank = shape.len();
    if rank > 4 {
        return Err(AdapterError::RankTooHigh {
            tensor: name.to_string(),
            rank,
        });
    }
    let mut dims = [0u32; 4];
    for (i, d) in shape.iter().enumerate() {
        dims[i] = u32::try_from(*d).map_err(|_| AdapterError::DimTooLarge {
            tensor: name.to_string(),
            dim: *d,
        })?;
    }
    Ok((rank as u32, dims))
}

pub fn decode_f32_le_bytes(data: &[u8]) -> Result<Vec<f32>, AdapterError> {
    if data.len() % 4 != 0 {
        return Err(AdapterError::InvalidF32Bytes { len: data.len() });
    }
    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn decode_f16_le_bytes(data: &[u8]) -> Result<Vec<f32>, AdapterError> {
    if data.len() % 2 != 0 {
        return Err(AdapterError::InvalidF16Bytes { len: data.len() });
    }
    Ok(data
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            crate::ml::half::f16::from_bits(bits).to_f32()
        })
        .collect())
}

pub fn decode_bf16_le_bytes(data: &[u8]) -> Result<Vec<f32>, AdapterError> {
    if data.len() % 2 != 0 {
        return Err(AdapterError::InvalidBf16Bytes { len: data.len() });
    }
    Ok(data
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            crate::ml::half::bf16::from_bits(bits).to_f32()
        })
        .collect())
}

pub fn dequantize_q8_symmetric_i8(data: &[u8], scale: f32) -> Vec<f32> {
    data.iter()
        .map(|b| {
            let q = i8::from_le_bytes([*b]);
            (q as f32) * scale
        })
        .collect()
}

pub fn dequantize_q8_affine_u8(data: &[u8], scale: f32, zero_point: i32) -> Vec<f32> {
    data.iter()
        .map(|q| ((*q as i32 - zero_point) as f32) * scale)
        .collect()
}

pub fn dequantize_q4_symmetric_packed(
    packed: &[u8],
    elem_count: usize,
    scale: f32,
) -> Result<Vec<f32>, AdapterError> {
    let needed = elem_count.div_ceil(2);
    if packed.len() < needed {
        return Err(AdapterError::InvalidQ4Bytes {
            packed_len: packed.len(),
            elem_count,
        });
    }
    let mut out = Vec::with_capacity(elem_count);
    for byte in packed.iter().take(needed) {
        let lo = byte & 0x0F;
        let hi = (byte >> 4) & 0x0F;

        let lo_q = if lo >= 8 { (lo as i8) - 16 } else { lo as i8 };
        out.push((lo_q as f32) * scale);
        if out.len() == elem_count {
            break;
        }

        let hi_q = if hi >= 8 { (hi as i8) - 16 } else { hi as i8 };
        out.push((hi_q as f32) * scale);
        if out.len() == elem_count {
            break;
        }
    }
    Ok(out)
}

pub fn dequantize_q4_affine_packed(
    packed: &[u8],
    elem_count: usize,
    scale: f32,
    zero_point: i32,
) -> Result<Vec<f32>, AdapterError> {
    let needed = elem_count.div_ceil(2);
    if packed.len() < needed {
        return Err(AdapterError::InvalidQ4Bytes {
            packed_len: packed.len(),
            elem_count,
        });
    }
    let mut out = Vec::with_capacity(elem_count);
    for byte in packed.iter().take(needed) {
        let lo = byte & 0x0F;
        let hi = (byte >> 4) & 0x0F;

        out.push(((lo as i32 - zero_point) as f32) * scale);
        if out.len() == elem_count {
            break;
        }
        out.push(((hi as i32 - zero_point) as f32) * scale);
        if out.len() == elem_count {
            break;
        }
    }
    Ok(out)
}

fn upload_f32_tensor(
    session: &Session,
    name: &str,
    shape: &[usize],
    values: Vec<f32>,
    raw_byte_len: usize,
    immutable: bool,
    tag: u32,
) -> Result<LoadedTensor, AdapterError> {
    let (rank, dims) = shape_to_rank_dims(name, shape)?;
    let desc = BufferDesc {
        dtype: BufferDType::F32,
        rank,
        dims,
        immutable,
        tag,
    };
    let buffer_id = session.alloc_buffer(desc)?;
    if let Err(e) = session.upload_f32(buffer_id, &values) {
        let _ = session.free_buffer(buffer_id);
        return Err(AdapterError::Ffi(e));
    }

    Ok(LoadedTensor {
        name: name.to_string(),
        buffer_id,
        dtype: BufferDType::F32,
        rank,
        dims,
        elem_count: values.len(),
        byte_len: raw_byte_len,
    })
}

fn upload_typed_bytes_tensor(
    session: &Session,
    name: &str,
    dtype: BufferDType,
    shape: &[usize],
    raw_bytes: &[u8],
    immutable: bool,
    tag: u32,
) -> Result<LoadedTensor, AdapterError> {
    let (rank, dims) = shape_to_rank_dims(name, shape)?;
    let desc = BufferDesc {
        dtype,
        rank,
        dims,
        immutable,
        tag,
    };
    if raw_bytes.len() != desc.byte_len() {
        return Err(AdapterError::ByteSizeMismatch {
            tensor: name.to_string(),
            expected: desc.byte_len(),
            got: raw_bytes.len(),
        });
    }
    let buffer_id = session.alloc_buffer(desc)?;
    if let Err(e) = session.upload_bytes(buffer_id, raw_bytes) {
        let _ = session.free_buffer(buffer_id);
        return Err(AdapterError::Ffi(e));
    }
    Ok(LoadedTensor {
        name: name.to_string(),
        buffer_id,
        dtype,
        rank,
        dims,
        elem_count: desc.elem_count(),
        byte_len: raw_bytes.len(),
    })
}

/// Load safetensors into a feRcuda session.
///
/// Supported input dtypes:
/// - F32, F16, BF16 (native upload)
/// - I8, U8 (normalized to F32 today)
pub fn load_safetensors_into_session<P: AsRef<Path>>(
    session: &Session,
    path: P,
    immutable: bool,
    tag: u32,
) -> Result<LoadedModel, AdapterError> {
    let file = File::open(path)?;
    // SAFETY: read-only mapping for safetensors deserialization.
    let mmap = unsafe { crate::ml::memmap2::MmapOptions::new().map(&file)? };
    let st = crate::ml::safetensors::SafeTensors::deserialize(&mmap)?;

    let mut model = LoadedModel::default();
    for name in st.names() {
        let view = st.tensor(name)?;
        let loaded = match view.dtype() {
            crate::ml::safetensors::Dtype::F32 => upload_typed_bytes_tensor(
                session,
                name,
                BufferDType::F32,
                view.shape(),
                view.data(),
                immutable,
                tag,
            )?,
            crate::ml::safetensors::Dtype::F16 => upload_typed_bytes_tensor(
                session,
                name,
                BufferDType::F16,
                view.shape(),
                view.data(),
                immutable,
                tag,
            )?,
            crate::ml::safetensors::Dtype::BF16 => upload_typed_bytes_tensor(
                session,
                name,
                BufferDType::BF16,
                view.shape(),
                view.data(),
                immutable,
                tag,
            )?,
            crate::ml::safetensors::Dtype::I8 => {
                let values = dequantize_q8_symmetric_i8(view.data(), 1.0);
                upload_f32_tensor(
                    session,
                    name,
                    view.shape(),
                    values,
                    view.data().len(),
                    immutable,
                    tag,
                )?
            }
            crate::ml::safetensors::Dtype::U8 => {
                let values = dequantize_q8_affine_u8(view.data(), 1.0, 0);
                upload_f32_tensor(
                    session,
                    name,
                    view.shape(),
                    values,
                    view.data().len(),
                    immutable,
                    tag,
                )?
            }
            other => {
                return Err(AdapterError::UnsupportedDType {
                    tensor: name.to_string(),
                    dtype: format!("{other:?}"),
                })
            }
        };
        model.by_name.insert(loaded.name.clone(), loaded.buffer_id);
        model.tensors.push(loaded);
    }
    Ok(model)
}

/// Backward-compatible name: now supports mixed dtypes and normalizes to F32.
pub fn load_safetensors_f32_into_session<P: AsRef<Path>>(
    session: &Session,
    path: P,
    immutable: bool,
    tag: u32,
) -> Result<LoadedModel, AdapterError> {
    load_safetensors_into_session(session, path, immutable, tag)
}

/// Upload already-quantized payload into session as F32 using explicit quant params.
///
/// This is the direct Q8/Q4 adapter path for formats that are not native
/// safetensors dtypes (for example packed Q4 blobs).
pub fn upload_quantized_into_session(
    session: &Session,
    name: &str,
    shape: &[usize],
    quantized: &[u8],
    params: QuantParams,
    immutable: bool,
    tag: u32,
) -> Result<LoadedTensor, AdapterError> {
    let elem_count = shape.iter().product::<usize>();
    let values = match params.mode {
        QuantMode::Q8Symmetric => dequantize_q8_symmetric_i8(quantized, params.scale),
        QuantMode::Q8Affine => dequantize_q8_affine_u8(quantized, params.scale, params.zero_point),
        QuantMode::Q4SymmetricPacked => {
            dequantize_q4_symmetric_packed(quantized, elem_count, params.scale)?
        }
        QuantMode::Q4AffinePacked => {
            dequantize_q4_affine_packed(quantized, elem_count, params.scale, params.zero_point)?
        }
    };
    upload_f32_tensor(session, name, shape, values, quantized.len(), immutable, tag)
}
