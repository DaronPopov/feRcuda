//! fercuda-ml-lite
//!
//! Lightweight ML helpers for feRcuda:
//! - safetensors metadata inspection
//! - simple JSON serde helpers
//! - deterministic gaussian sampling

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use safetensors::SafeTensors;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fs::File;
use std::path::Path;

pub use half;
pub use memmap2;
pub use rand;
pub use rand_distr;
pub use safetensors;
pub use serde;
pub use serde_json;

#[derive(Debug, thiserror::Error)]
pub enum MlLiteError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("serde json error: {0}")]
    SerdeJson(#[from] serde_json::Error),
    #[error("distribution error: {0}")]
    Distribution(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub nbytes: usize,
}

pub fn inspect_safetensors<P: AsRef<Path>>(path: P) -> Result<Vec<TensorMeta>, MlLiteError> {
    let file = File::open(path)?;
    // SAFETY: Read-only file mapping for immutable metadata inspection.
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let mut out = Vec::new();
    for name in st.names() {
        let view = st.tensor(name)?;
        out.push(TensorMeta {
            name: name.to_string(),
            dtype: format!("{:?}", view.dtype()),
            shape: view.shape().to_vec(),
            nbytes: view.data().len(),
        });
    }
    Ok(out)
}

pub fn to_json<T: Serialize>(value: &T) -> Result<String, MlLiteError> {
    Ok(serde_json::to_string(value)?)
}

pub fn from_json<T: DeserializeOwned>(s: &str) -> Result<T, MlLiteError> {
    Ok(serde_json::from_str(s)?)
}

pub fn gaussian_noise_f32(
    len: usize,
    mean: f32,
    stddev: f32,
    seed: u64,
) -> Result<Vec<f32>, MlLiteError> {
    let dist = Normal::<f32>::new(mean, stddev)
        .map_err(|e| MlLiteError::Distribution(e.to_string()))?;
    let mut rng = StdRng::seed_from_u64(seed);
    Ok((0..len).map(|_| dist.sample(&mut rng)).collect())
}
