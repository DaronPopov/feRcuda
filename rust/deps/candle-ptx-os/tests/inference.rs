//! Inference tests for candle-ptx-os

use candle_core::backend::{BackendDevice, BackendStorage};
use candle_core::{DType, Layout, Shape};
use candle_ptx_os::PtxDevice;

#[test]
#[ignore = "requires CUDA device"]
fn test_matmul_basic() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    // Create two matrices: A (2x3) and B (3x4)
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];

    let a_storage = device
        .storage_from_slice(&a_data)
        .expect("Failed to create A storage");
    let b_storage = device
        .storage_from_slice(&b_data)
        .expect("Failed to create B storage");

    let a_shape = Shape::from_dims(&[2, 3]);
    let b_shape = Shape::from_dims(&[3, 4]);
    let a_layout = Layout::contiguous(&a_shape);
    let b_layout = Layout::contiguous(&b_shape);

    // C = A @ B should be (2x4)
    // batch=1, m=2, n=4, k=3
    let c_storage = a_storage
        .matmul(&b_storage, (1, 2, 4, 3), &a_layout, &b_layout)
        .expect("Failed to matmul");

    let cpu = c_storage.to_cpu_storage().expect("Failed to copy to CPU");
    if let candle_core::CpuStorage::F32(data) = cpu {
        assert_eq!(data.len(), 8); // 2x4

        // Expected result:
        // [1,2,3] @ [[1,2,3,4], [5,6,7,8], [9,10,11,12]] = [38, 44, 50, 56]
        // [4,5,6] @ [[1,2,3,4], [5,6,7,8], [9,10,11,12]] = [83, 98, 113, 128]
        let expected = [38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0];
        for (i, (&actual, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i,
                actual,
                exp
            );
        }
    } else {
        panic!("Expected F32 storage");
    }
}

#[test]
#[ignore = "requires CUDA device"]
fn test_matmul_larger() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    let m = 64;
    let k = 128;
    let n = 256;

    // Create random matrices
    let a_shape = Shape::from_dims(&[m, k]);
    let b_shape = Shape::from_dims(&[k, n]);

    let a_storage = device
        .rand_uniform(&a_shape, DType::F32, -1.0, 1.0)
        .expect("Failed to create A");
    let b_storage = device
        .rand_uniform(&b_shape, DType::F32, -1.0, 1.0)
        .expect("Failed to create B");

    let a_layout = Layout::contiguous(&a_shape);
    let b_layout = Layout::contiguous(&b_shape);

    let c_storage = a_storage
        .matmul(&b_storage, (1, m, n, k), &a_layout, &b_layout)
        .expect("Failed to matmul");

    let cpu = c_storage.to_cpu_storage().expect("Failed to copy to CPU");
    if let candle_core::CpuStorage::F32(data) = cpu {
        assert_eq!(data.len(), m * n);

        // Just verify we got numbers (not NaN or Inf)
        for val in data {
            assert!(val.is_finite(), "Got non-finite value: {}", val);
        }
    } else {
        panic!("Expected F32 storage");
    }
}

#[test]
#[ignore = "requires CUDA device"]
fn test_batched_matmul() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    let batch = 4;
    let m = 32;
    let k = 64;
    let n = 32;

    let a_shape = Shape::from_dims(&[batch, m, k]);
    let b_shape = Shape::from_dims(&[batch, k, n]);

    let a_storage = device
        .rand_uniform(&a_shape, DType::F32, -1.0, 1.0)
        .expect("Failed to create A");
    let b_storage = device
        .rand_uniform(&b_shape, DType::F32, -1.0, 1.0)
        .expect("Failed to create B");

    let a_layout = Layout::contiguous(&a_shape);
    let b_layout = Layout::contiguous(&b_shape);

    let c_storage = a_storage
        .matmul(&b_storage, (batch, m, n, k), &a_layout, &b_layout)
        .expect("Failed to batched matmul");

    let cpu = c_storage.to_cpu_storage().expect("Failed to copy to CPU");
    if let candle_core::CpuStorage::F32(data) = cpu {
        assert_eq!(data.len(), batch * m * n);
    } else {
        panic!("Expected F32 storage");
    }
}

#[test]
#[ignore = "requires CUDA device"]
fn test_binary_ops() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    let shape = Shape::from_dims(&[100]);
    let _layout = Layout::contiguous(&shape);

    // Create two tensors with known values
    let a_data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..100).map(|i| (100 - i) as f32).collect();

    let a_storage = device
        .storage_from_slice(&a_data)
        .expect("Failed to create A");
    let b_storage = device
        .storage_from_slice(&b_data)
        .expect("Failed to create B");

    // Test binary add via direct FFI call
    let out_ptr = device.alloc_raw(100 * std::mem::size_of::<f32>()).expect("Failed to alloc");
    let stream = std::ptr::null_mut();

    unsafe {
        candle_ptx_os::ffi::ptx_tensor_add_f32(
            a_storage.as_ptr() as *mut f32,
            b_storage.as_ptr() as *mut f32,
            out_ptr as *mut f32,
            100,
            stream,
        );
    }

    // Copy result back using cuda_utils
    let mut result = vec![0.0f32; 100];
    candle_ptx_os::cuda_utils::copy_slice_from_device(out_ptr as *const f32, &mut result)
        .expect("Failed to copy from device");

    // Each element should be i + (100 - i) = 100
    for val in result {
        assert!((val - 100.0).abs() < 1e-6);
    }

    unsafe { device.free_raw(out_ptr); }
}

#[test]
#[ignore = "requires CUDA device"]
fn test_unary_ops() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    let shape = Shape::from_dims(&[10]);
    let _layout = Layout::contiguous(&shape);

    let data: Vec<f32> = vec![0.0, 1.0, 2.0, -1.0, -2.0, 0.5, -0.5, 3.0, -3.0, 0.1];
    let storage = device
        .storage_from_slice(&data)
        .expect("Failed to create storage");

    // Test unary exp via direct FFI call
    let out_ptr = device.alloc_raw(10 * std::mem::size_of::<f32>()).expect("Failed to alloc");
    let stream = std::ptr::null_mut();

    unsafe {
        candle_ptx_os::ffi::ptx_tensor_exp_f32(
            storage.as_ptr() as *mut f32,
            out_ptr as *mut f32,
            10,
            stream,
        );
    }

    // Copy result back using cuda_utils
    let mut result = vec![0.0f32; 10];
    candle_ptx_os::cuda_utils::copy_slice_from_device(out_ptr as *const f32, &mut result)
        .expect("Failed to copy from device");

    for (i, (&actual, &original)) in result.iter().zip(data.iter()).enumerate() {
        let expected = original.exp();
        assert!(
            (actual - expected).abs() < 1e-5,
            "Mismatch at {}: exp({}) = {} vs {}",
            i,
            original,
            expected,
            actual
        );
    }

    unsafe { device.free_raw(out_ptr); }
}

#[test]
#[ignore = "requires CUDA device"]
fn test_affine() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    let shape = Shape::from_dims(&[100]);
    let layout = Layout::contiguous(&shape);

    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let storage = device
        .storage_from_slice(&data)
        .expect("Failed to create storage");

    // out = 2 * x + 3
    let result = storage.affine(&layout, 2.0, 3.0).expect("Failed to affine");

    let cpu = result.to_cpu_storage().expect("Failed to copy to CPU");
    if let candle_core::CpuStorage::F32(result_data) = cpu {
        for (i, &val) in result_data.iter().enumerate() {
            let expected = 2.0 * (i as f32) + 3.0;
            assert!(
                (val - expected).abs() < 1e-5,
                "Mismatch at {}: {} vs {}",
                i,
                val,
                expected
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA device"]
fn test_simple_mlp() {
    // Test a simple MLP-like computation: relu(x @ W1 + b1) @ W2 + b2
    let device = PtxDevice::new(0).expect("Failed to create device");

    // Input: batch=8, features=32
    // Hidden: 64
    // Output: 10

    let batch = 8;
    let in_features = 32;
    let hidden = 64;
    let out_features = 10;

    // Create random weights and biases
    let x_shape = Shape::from_dims(&[batch, in_features]);
    let w1_shape = Shape::from_dims(&[in_features, hidden]);
    let w2_shape = Shape::from_dims(&[hidden, out_features]);

    let x = device
        .rand_uniform(&x_shape, DType::F32, -1.0, 1.0)
        .expect("Failed to create x");
    let w1 = device
        .rand_uniform(&w1_shape, DType::F32, -0.1, 0.1)
        .expect("Failed to create w1");
    let w2 = device
        .rand_uniform(&w2_shape, DType::F32, -0.1, 0.1)
        .expect("Failed to create w2");

    let x_layout = Layout::contiguous(&x_shape);
    let w1_layout = Layout::contiguous(&w1_shape);
    let w2_layout = Layout::contiguous(&w2_shape);

    // Forward pass: h = x @ w1
    let h = x
        .matmul(&w1, (1, batch, hidden, in_features), &x_layout, &w1_layout)
        .expect("Failed first matmul");

    let h_shape = Shape::from_dims(&[batch, hidden]);
    let h_layout = Layout::contiguous(&h_shape);

    // Apply ReLU via affine + clamp simulation (or unary if available)
    // For now just continue with the hidden activations

    // Output: y = h @ w2
    let y = h
        .matmul(&w2, (1, batch, out_features, hidden), &h_layout, &w2_layout)
        .expect("Failed second matmul");

    let cpu = y.to_cpu_storage().expect("Failed to copy to CPU");
    if let candle_core::CpuStorage::F32(data) = cpu {
        assert_eq!(data.len(), batch * out_features);
        // Verify we got finite values
        for val in data {
            assert!(val.is_finite());
        }
    }
}

#[test]
#[ignore = "requires CUDA device"]
fn test_allocation_performance() {
    use std::time::Instant;

    let device = PtxDevice::new(0).expect("Failed to create device");
    let iterations = 1000;
    let alloc_size = 4096; // 4KB - typical small allocation

    // Test 1: Raw TLSF allocation (no zeroing)
    let start = Instant::now();
    for _ in 0..iterations {
        let ptr = device.alloc_raw(alloc_size).expect("Failed to allocate");
        unsafe { device.free_raw(ptr); }
    }
    let raw_elapsed = start.elapsed();
    let raw_avg_ns = raw_elapsed.as_nanos() / iterations as u128;
    println!(
        "Raw TLSF alloc+free: {}ns ({} iterations)",
        raw_avg_ns, iterations
    );

    // Test 2: Zeroed allocation (includes cudaMemset overhead)
    let shape = Shape::from_dims(&[1024]);
    let start = Instant::now();
    for _ in 0..iterations {
        let _storage = device
            .zeros_impl(&shape, DType::F32)
            .expect("Failed to allocate");
        // Storage is dropped here, freeing memory
    }
    let zeros_elapsed = start.elapsed();
    let zeros_avg_ns = zeros_elapsed.as_nanos() / iterations as u128;
    println!(
        "Zeroed alloc+free: {}ns ({} iterations)",
        zeros_avg_ns, iterations
    );

    // Raw TLSF should be very fast (~130ns alloc + ~55ns free = ~185ns)
    // Allow generous headroom for test stability
    assert!(
        raw_avg_ns < 5_000,
        "Raw TLSF allocation too slow: {}ns (expected <5000ns)",
        raw_avg_ns
    );

    // Zeroed allocation includes cudaMemset, so allow more time
    // Should still be much faster than cudaMalloc (~2000ns) + memset
    assert!(
        zeros_avg_ns < 50_000,
        "Zeroed allocation too slow: {}ns (expected <50000ns)",
        zeros_avg_ns
    );
}
