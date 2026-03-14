//! Basic operations tests for candle-ptx-os

use candle_core::backend::{BackendDevice, BackendStorage};
use candle_core::{DType, Shape};
use candle_ptx_os::PtxDevice;

#[test]
#[ignore = "requires CUDA device"]
fn test_device_creation() {
    let device = PtxDevice::new(0).expect("Failed to create device");
    assert_eq!(device.device_id(), 0);
}

#[test]
#[ignore = "requires CUDA device"]
fn test_zeros() {
    let device = PtxDevice::new(0).expect("Failed to create device");
    let shape = Shape::from_dims(&[64, 128]);

    let storage = device
        .zeros_impl(&shape, DType::F32)
        .expect("Failed to create zeros");

    // Copy back to CPU and verify
    let cpu = storage.to_cpu_storage().expect("Failed to copy to CPU");
    if let candle_core::CpuStorage::F32(data) = cpu {
        assert_eq!(data.len(), 64 * 128);
        for val in data {
            assert_eq!(val, 0.0);
        }
    } else {
        panic!("Expected F32 storage");
    }
}

#[test]
#[ignore = "requires CUDA device"]
fn test_ones() {
    let device = PtxDevice::new(0).expect("Failed to create device");
    let shape = Shape::from_dims(&[32, 64]);

    let storage = device
        .ones_impl(&shape, DType::F32)
        .expect("Failed to create ones");

    let cpu = storage.to_cpu_storage().expect("Failed to copy to CPU");
    if let candle_core::CpuStorage::F32(data) = cpu {
        assert_eq!(data.len(), 32 * 64);
        for val in data {
            assert!((val - 1.0).abs() < 1e-6);
        }
    } else {
        panic!("Expected F32 storage");
    }
}

#[test]
#[ignore = "requires CUDA device"]
fn test_storage_from_slice() {
    let device = PtxDevice::new(0).expect("Failed to create device");
    let data: Vec<f32> = (0..256).map(|i| i as f32).collect();

    let storage = device
        .storage_from_slice(&data)
        .expect("Failed to create storage from slice");

    let cpu = storage.to_cpu_storage().expect("Failed to copy to CPU");
    if let candle_core::CpuStorage::F32(result) = cpu {
        assert_eq!(result.len(), 256);
        for (i, val) in result.iter().enumerate() {
            assert!((val - i as f32).abs() < 1e-6);
        }
    } else {
        panic!("Expected F32 storage");
    }
}

#[test]
#[ignore = "requires CUDA device"]
fn test_random_uniform() {
    let device = PtxDevice::new(0).expect("Failed to create device");
    let shape = Shape::from_dims(&[1000]);

    let storage = device
        .rand_uniform(&shape, DType::F32, 0.0, 1.0)
        .expect("Failed to create random uniform");

    let cpu = storage.to_cpu_storage().expect("Failed to copy to CPU");
    if let candle_core::CpuStorage::F32(data) = cpu {
        assert_eq!(data.len(), 1000);

        // Verify all values are in [0, 1)
        for val in &data {
            assert!(*val >= 0.0 && *val < 1.0);
        }

        // Verify there's some variance (not all the same)
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean > 0.4 && mean < 0.6, "Mean should be around 0.5");
    } else {
        panic!("Expected F32 storage");
    }
}

#[test]
#[ignore = "requires CUDA device"]
fn test_random_normal() {
    let device = PtxDevice::new(0).expect("Failed to create device");
    let shape = Shape::from_dims(&[10000]);

    let storage = device
        .rand_normal(&shape, DType::F32, 0.0, 1.0)
        .expect("Failed to create random normal");

    let cpu = storage.to_cpu_storage().expect("Failed to copy to CPU");
    if let candle_core::CpuStorage::F32(data) = cpu {
        assert_eq!(data.len(), 10000);

        // Verify statistics are approximately correct
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let variance: f32 =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;

        assert!(mean.abs() < 0.1, "Mean should be close to 0, got {}", mean);
        assert!(
            (variance - 1.0).abs() < 0.2,
            "Variance should be close to 1, got {}",
            variance
        );
    } else {
        panic!("Expected F32 storage");
    }
}

#[test]
#[ignore = "requires CUDA device"]
fn test_f16_support() {
    let device = PtxDevice::new(0).expect("Failed to create device");
    let shape = Shape::from_dims(&[64]);

    let storage = device
        .ones_impl(&shape, DType::F16)
        .expect("Failed to create F16 ones");

    let cpu = storage.to_cpu_storage().expect("Failed to copy to CPU");
    if let candle_core::CpuStorage::F16(data) = cpu {
        assert_eq!(data.len(), 64);
        for val in data {
            assert!((val.to_f32() - 1.0).abs() < 1e-3);
        }
    } else {
        panic!("Expected F16 storage");
    }
}

#[test]
#[ignore = "requires CUDA device"]
fn test_bf16_support() {
    let device = PtxDevice::new(0).expect("Failed to create device");
    let shape = Shape::from_dims(&[64]);

    let storage = device
        .ones_impl(&shape, DType::BF16)
        .expect("Failed to create BF16 ones");

    let cpu = storage.to_cpu_storage().expect("Failed to copy to CPU");
    if let candle_core::CpuStorage::BF16(data) = cpu {
        assert_eq!(data.len(), 64);
        for val in data {
            assert!((val.to_f32() - 1.0).abs() < 1e-2);
        }
    } else {
        panic!("Expected BF16 storage");
    }
}

#[test]
#[ignore = "requires CUDA device"]
fn test_pool_stats() {
    let device = PtxDevice::new(0).expect("Failed to create device");
    let shape = Shape::from_dims(&[1024, 1024]);

    // Allocate some memory
    let _storage = device
        .zeros_impl(&shape, DType::F32)
        .expect("Failed to create zeros");

    let stats = device.pool_stats();
    assert!(stats.total_size > 0);
    assert!(stats.allocated > 0);
}

#[test]
#[ignore = "requires CUDA device"]
fn test_seed_reproducibility() {
    let device = PtxDevice::new(0).expect("Failed to create device");
    let shape = Shape::from_dims(&[100]);

    // Set seed and generate
    device.set_seed(42).expect("Failed to set seed");
    let storage1 = device
        .rand_uniform(&shape, DType::F32, 0.0, 1.0)
        .expect("Failed to create random");

    // Reset seed and generate again
    device.set_seed(42).expect("Failed to set seed");
    let storage2 = device
        .rand_uniform(&shape, DType::F32, 0.0, 1.0)
        .expect("Failed to create random");

    // Should be identical
    let cpu1 = storage1.to_cpu_storage().expect("Failed to copy to CPU");
    let cpu2 = storage2.to_cpu_storage().expect("Failed to copy to CPU");

    if let (candle_core::CpuStorage::F32(data1), candle_core::CpuStorage::F32(data2)) =
        (cpu1, cpu2)
    {
        for (a, b) in data1.iter().zip(data2.iter()) {
            assert!((a - b).abs() < 1e-10, "Values should be identical");
        }
    }
}
