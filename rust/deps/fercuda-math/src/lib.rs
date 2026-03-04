//! fercuda-math
//!
//! Canonical math facade for feRcuda:
//! - `nalgebra` as the default low/medium-dimensional linear algebra layer.
//! - Optional `faer` feature for heavier dense CPU linear algebra.
//! - Optional `candle` feature for tensor interop boundaries.

pub use nalgebra;

pub type Vec3f = nalgebra::SVector<f32, 3>;
pub type Vec6f = nalgebra::SVector<f32, 6>;
pub type Mat3f = nalgebra::SMatrix<f32, 3, 3>;
pub type Mat6f = nalgebra::SMatrix<f32, 6, 6>;
pub type Quatf = nalgebra::UnitQuaternion<f32>;

#[derive(Debug, thiserror::Error)]
pub enum MathError {
    #[cfg(feature = "candle")]
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("shape mismatch: expected {expected} elements, got {got}")]
    ShapeMismatch { expected: usize, got: usize },
}

/// One simple, deterministic Kalman predict/update helper for 2-state systems.
///
/// State:
/// x = [position, velocity]^T
pub fn kalman_step_2x1(
    x: nalgebra::SVector<f32, 2>,
    p: nalgebra::SMatrix<f32, 2, 2>,
    f: nalgebra::SMatrix<f32, 2, 2>,
    b: nalgebra::SVector<f32, 2>,
    u: f32,
    q: nalgebra::SMatrix<f32, 2, 2>,
    z: f32,
    r: f32,
) -> (nalgebra::SVector<f32, 2>, nalgebra::SMatrix<f32, 2, 2>) {
    let h = nalgebra::SMatrix::<f32, 1, 2>::new(1.0, 0.0);
    let ht = h.transpose();
    let i = nalgebra::SMatrix::<f32, 2, 2>::identity();

    let x_pred = f * x + b * u;
    let p_pred = f * p * f.transpose() + q;
    let y = z - (h * x_pred)[0];
    let s = (h * p_pred * ht)[0] + r;
    let k = (p_pred * ht) * (1.0 / s.max(1e-9));
    let x_next = x_pred + k.column(0) * y;
    let p_next = (i - k * h) * p_pred;
    (x_next, p_next)
}

#[cfg(feature = "faer")]
pub mod dense {
    pub use faer::Mat;
}

#[cfg(feature = "candle")]
pub mod candle_bridge {
    use super::MathError;
    use candle_core::{DType, Device, Tensor};
    use nalgebra::DMatrix;

    pub fn dmatrix_to_tensor_f32(m: &DMatrix<f32>, device: &Device) -> Result<Tensor, MathError> {
        let rows = m.nrows();
        let cols = m.ncols();
        let data: Vec<f32> = m.as_slice().to_vec();
        Ok(Tensor::from_vec(data, (rows, cols), device)?.to_dtype(DType::F32)?)
    }

    pub fn tensor_to_dmatrix_f32(t: &Tensor) -> Result<DMatrix<f32>, MathError> {
        let dims = t.dims();
        if dims.len() != 2 {
            return Err(MathError::ShapeMismatch {
                expected: 2,
                got: dims.len(),
            });
        }
        let rows = dims[0];
        let cols = dims[1];
        let data = t.flatten_all()?.to_vec1::<f32>()?;
        if data.len() != rows * cols {
            return Err(MathError::ShapeMismatch {
                expected: rows * cols,
                got: data.len(),
            });
        }
        Ok(DMatrix::from_row_slice(rows, cols, &data))
    }
}
