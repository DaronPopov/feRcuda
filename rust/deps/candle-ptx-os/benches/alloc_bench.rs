//! Allocation performance benchmark for candle-ptx-os

use candle_core::backend::BackendDevice;
use candle_core::{DType, Shape};
use candle_ptx_os::PtxDevice;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_allocation(c: &mut Criterion) {
    let device = match PtxDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping benchmarks - no CUDA device: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("allocation");

    // Test various tensor sizes
    let sizes = [
        (1024,),
        (4096,),
        (16384,),
        (65536,),
        (262144,),
        (1048576,),
    ];

    for size in sizes {
        let shape = Shape::from_dims(&[size.0]);

        group.bench_with_input(
            BenchmarkId::new("zeros_f32", size.0),
            &size,
            |b, _| {
                b.iter(|| {
                    let storage = device
                        .zeros_impl(&shape, DType::F32)
                        .expect("Failed to allocate");
                    black_box(storage);
                });
            },
        );
    }

    group.finish();
}

fn bench_allocation_cycle(c: &mut Criterion) {
    let device = match PtxDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping benchmarks - no CUDA device: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("alloc_free_cycle");

    // Test rapid alloc/free cycles
    let shape = Shape::from_dims(&[1024]);

    group.bench_function("1024_f32_cycle", |b| {
        b.iter(|| {
            // Allocate and immediately drop (free)
            let storage = device
                .zeros_impl(&shape, DType::F32)
                .expect("Failed to allocate");
            black_box(storage);
        });
    });

    // Test with retained allocations (memory pressure)
    group.bench_function("1024_f32_cycle_with_pressure", |b| {
        // Pre-allocate some memory to create fragmentation
        let mut retained = Vec::new();
        for _ in 0..100 {
            retained.push(
                device
                    .zeros_impl(&Shape::from_dims(&[4096]), DType::F32)
                    .expect("Failed"),
            );
        }

        b.iter(|| {
            let storage = device
                .zeros_impl(&shape, DType::F32)
                .expect("Failed to allocate");
            black_box(storage);
        });
    });

    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let device = match PtxDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping benchmarks - no CUDA device: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("matmul");

    let sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)];

    for (m, k, n) in sizes {
        let a_shape = Shape::from_dims(&[m, k]);
        let b_shape = Shape::from_dims(&[k, n]);

        let a = device
            .rand_uniform(&a_shape, DType::F32, -1.0, 1.0)
            .expect("Failed to create A");
        let b = device
            .rand_uniform(&b_shape, DType::F32, -1.0, 1.0)
            .expect("Failed to create B");

        let a_layout = candle_core::Layout::contiguous(&a_shape);
        let b_layout = candle_core::Layout::contiguous(&b_shape);

        group.bench_with_input(
            BenchmarkId::new("f32", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |bench, _| {
                use candle_core::backend::BackendStorage;
                bench.iter(|| {
                    let c = a
                        .matmul(&b, (1, m, n, k), &a_layout, &b_layout)
                        .expect("Failed to matmul");
                    black_box(c);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_allocation, bench_allocation_cycle, bench_matmul);
criterion_main!(benches);
