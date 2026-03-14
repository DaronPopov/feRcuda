//! Demonstration of PtxTensor - Seamless Candle Tensor API with PTX-OS allocation
//!
//! This example shows how PtxTensor provides a drop-in Tensor API that uses
//! PTX-OS's O(1) TLSF allocator under the hood.
//!
//! Run with: cargo run --example tensor_demo --release

use candle_core::DType;
use candle_ptx_os::{PtxDevice, PtxDeviceExt, PtxTensor, Result};

fn main() -> Result<()> {
    println!("PTX-OS Tensor API Demo");
    println!("======================\n");

    // Create device with 256MB TLSF pool
    let device = PtxDevice::with_pool_size(0, 256 * 1024 * 1024)?;
    println!("Created PtxDevice with 256MB TLSF pool\n");

    // ========================================================================
    // Tensor Creation - Familiar Candle-like API
    // ========================================================================
    println!("Tensor Creation:");
    println!("-----------------");

    // Method 1: Static constructors on PtxTensor
    let zeros = PtxTensor::zeros(&device, (3, 4), DType::F32)?;
    println!("zeros((3,4)): shape={:?}", zeros.dims());

    let ones = PtxTensor::ones(&device, (2, 2), DType::F32)?;
    println!("ones((2,2)): shape={:?}", ones.dims());

    let randn = PtxTensor::randn(&device, (64, 128), 0.0, 1.0)?;
    println!("randn((64,128)): shape={:?}, elem_count={}", randn.dims(), randn.elem_count());

    let rand = PtxTensor::rand(&device, (10, 10), DType::F32, -1.0, 1.0)?;
    println!("rand((10,10), -1..1): shape={:?}", rand.dims());

    // Method 2: Extension trait on PtxDevice
    let t1 = device.zeros((5, 5), DType::F32)?;
    let t2 = device.randn((5, 5), 0.0, 1.0)?;
    println!("device.zeros((5,5)): shape={:?}", t1.dims());
    println!("device.randn((5,5)): shape={:?}", t2.dims());

    // Method 3: From data
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let from_vec = PtxTensor::from_vec(&device, data, (2, 3))?;
    println!("from_vec((2,3)): shape={:?}", from_vec.dims());

    let identity = PtxTensor::eye(&device, 4, DType::F32)?;
    println!("eye(4): shape={:?}", identity.dims());

    let range = PtxTensor::arange(&device, 0.0, 10.0, DType::F32)?;
    println!("arange(0, 10): shape={:?}, len={}", range.dims(), range.elem_count());

    println!();

    // ========================================================================
    // Math Operations
    // ========================================================================
    println!("Math Operations:");
    println!("-----------------");

    let a = PtxTensor::randn(&device, (64, 128), 0.0, 1.0)?;
    let b = PtxTensor::randn(&device, (128, 64), 0.0, 1.0)?;

    let start = std::time::Instant::now();
    let c = a.matmul(&b)?;
    let elapsed = start.elapsed();
    println!("matmul({:?} x {:?}) = {:?} ({:.2?})", a.dims(), b.dims(), c.dims(), elapsed);

    let x = PtxTensor::randn(&device, (100,), 0.0, 1.0)?;
    let y = PtxTensor::randn(&device, (100,), 0.0, 1.0)?;
    let sum = x.add(&y)?;
    let _diff = x.sub(&y)?;
    let _prod = x.mul(&y)?;
    println!("add/sub/mul({:?}, {:?}): all result in {:?}", x.dims(), y.dims(), sum.dims());

    let scaled = x.mul_scalar(2.5)?;
    let _shifted = x.add_scalar(1.0)?;
    println!("mul_scalar, add_scalar: {:?}", scaled.dims());

    let sq = x.sqrt()?;
    let _ex = x.exp()?;
    println!("sqrt, exp: {:?}", sq.dims());

    println!();

    // ========================================================================
    // Reductions
    // ========================================================================
    println!("Reductions:");
    println!("-----------");

    let matrix = PtxTensor::randn(&device, (10, 20), 0.0, 1.0)?;
    let sum_all = matrix.sum_all()?;
    let mean_all = matrix.mean_all()?;
    println!("matrix({:?}) -> sum_all: {:?}, mean_all: {:?}",
             matrix.dims(), sum_all.dims(), mean_all.dims());

    let sum_dim0 = matrix.sum(0)?;
    let mean_dim1 = matrix.mean(1)?;
    println!("sum(dim=0): {:?}, mean(dim=1): {:?}", sum_dim0.dims(), mean_dim1.dims());

    println!();

    // ========================================================================
    // Activation Functions
    // ========================================================================
    println!("Activation Functions:");
    println!("---------------------");

    let input = PtxTensor::randn(&device, (32, 64), 0.0, 1.0)?;

    let relu = input.relu()?;
    let _gelu = input.gelu()?;
    let _sigmoid = input.sigmoid()?;
    let _tanh = input.tanh()?;
    let _silu = input.silu()?;
    let _softmax = input.softmax(1)?;

    println!("relu, gelu, sigmoid, tanh, silu, softmax - all {:?}", relu.dims());

    println!();

    // ========================================================================
    // Shape Operations
    // ========================================================================
    println!("Shape Operations:");
    println!("-----------------");

    let t = PtxTensor::randn(&device, (2, 3, 4), 0.0, 1.0)?;
    println!("original: {:?}", t.dims());

    let reshaped = t.reshape((6, 4))?;
    println!("reshape((6,4)): {:?}", reshaped.dims());

    let flattened = t.flatten_all()?;
    println!("flatten_all: {:?}, elem_count={}", flattened.dims(), flattened.elem_count());

    let with_dim = PtxTensor::randn(&device, (3, 1, 4), 0.0, 1.0)?;
    let squeezed = with_dim.squeeze(1)?;
    println!("squeeze(1): {:?} -> {:?}", with_dim.dims(), squeezed.dims());

    let unsqueezed = squeezed.unsqueeze(0)?;
    println!("unsqueeze(0): {:?} -> {:?}", squeezed.dims(), unsqueezed.dims());

    println!();

    // ========================================================================
    // Data Transfer
    // ========================================================================
    println!("Data Transfer:");
    println!("--------------");

    let small = PtxTensor::from_vec(&device, vec![1.0f32, 2.0, 3.0, 4.0], (2, 2))?;
    let data_back: Vec<f32> = small.to_vec()?;
    println!("to_vec(): {:?}", data_back);

    let data_2d: Vec<Vec<f32>> = small.to_vec2()?;
    println!("to_vec2(): {:?}", data_2d);

    // Convert to Candle tensor for advanced ops
    let candle_tensor = small.to_candle_cpu()?;
    println!("to_candle_cpu(): shape={:?}, dtype={:?}",
             candle_tensor.dims(), candle_tensor.dtype());

    // Convert back
    let back_to_ptx = PtxTensor::from_candle(&device, &candle_tensor)?;
    println!("from_candle(): {:?}", back_to_ptx.dims());

    println!();

    // ========================================================================
    // Memory Stats
    // ========================================================================
    println!("TLSF Pool Stats:");
    println!("----------------");
    let stats = device.pool_stats();
    println!("Allocated: {:.2} MB", stats.allocated as f64 / 1024.0 / 1024.0);
    println!("Free: {:.2} MB", stats.free as f64 / 1024.0 / 1024.0);
    println!("Fragmentation: {:.2}%", stats.fragmentation_ratio * 100.0);

    println!();
    println!("Demo complete! All tensor operations use O(1) TLSF allocation.");

    Ok(())
}
