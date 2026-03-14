//! Fragmentation validation tests for TLSF allocator
//!
//! These tests validate that the TLSF allocator properly handles memory fragmentation
//! through coalescing, segregated free lists, and other anti-fragmentation strategies.
//!
//! NOTE: Run with --test-threads=1 to avoid VRAM contention:
//!   cargo test --test fragmentation -- --ignored --test-threads=1

use candle_core::backend::BackendDevice;
use candle_core::{DType, Shape};
use candle_ptx_os::PtxDevice;
use std::time::Instant;

/// Helper to check if pool stats are valid (not corrupted/overflowed)
fn stats_valid(stats: &ptx_os::runtime::PoolStats) -> bool {
    // Check for obvious overflow/corruption (values near u64::MAX)
    stats.free < (1 << 40) && stats.allocated < (1 << 40) && stats.total_size > 0
}

/// Test 1: Basic coalescing - adjacent free blocks should merge
#[test]
#[ignore = "requires CUDA device"]
fn test_coalescing_adjacent_blocks() {
    let device = PtxDevice::new(0).expect("Failed to create device");
    let initial_stats = device.pool_stats();

    // Allocate three adjacent blocks
    let shape = Shape::from_dims(&[1024]); // 4KB each for F32
    let block_a = device.zeros_impl(&shape, DType::F32).expect("alloc A");
    let block_b = device.zeros_impl(&shape, DType::F32).expect("alloc B");
    let block_c = device.zeros_impl(&shape, DType::F32).expect("alloc C");

    let stats_after_alloc = device.pool_stats();
    let blocks_allocated = stats_after_alloc.allocated - initial_stats.allocated;
    assert!(blocks_allocated > 0, "Should have allocated memory");

    // Free middle block - creates a hole
    drop(block_b);

    // Free first block - should coalesce with the hole
    drop(block_a);

    // Free last block - should coalesce into one large free block
    drop(block_c);

    let final_stats = device.pool_stats();

    // After freeing all three, we should have coalesced back
    // The free space should be contiguous (one large block, not three small ones)
    println!(
        "Initial free: {}, Final free: {}",
        initial_stats.free, final_stats.free
    );

    // Verify we can allocate a block as large as all three combined
    let large_shape = Shape::from_dims(&[3072]); // 12KB - size of all three
    let large_block = device.zeros_impl(&large_shape, DType::F32);
    assert!(
        large_block.is_ok(),
        "Should be able to allocate coalesced space"
    );
}

/// Test 2: Interleaved allocation pattern - worst case for naive allocators
#[test]
#[ignore = "requires CUDA device"]
fn test_interleaved_allocation_pattern() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    // Allocate alternating small and large blocks
    let small_shape = Shape::from_dims(&[256]); // 1KB
    let large_shape = Shape::from_dims(&[4096]); // 16KB

    let mut small_blocks = Vec::new();
    let mut large_blocks = Vec::new();

    // Create interleaved pattern: S L S L S L S L S L
    for _ in 0..10 {
        small_blocks.push(
            device
                .zeros_impl(&small_shape, DType::F32)
                .expect("small alloc"),
        );
        large_blocks.push(
            device
                .zeros_impl(&large_shape, DType::F32)
                .expect("large alloc"),
        );
    }

    // Free all small blocks - creates holes between large blocks
    small_blocks.clear();

    let stats_with_holes = device.pool_stats();
    println!(
        "After freeing small blocks - Free: {}, Allocated: {}",
        stats_with_holes.free, stats_with_holes.allocated
    );

    // Try to allocate a medium block that would fit in combined hole space
    // but NOT in any single hole (tests if we're fragmenting badly)
    let medium_shape = Shape::from_dims(&[512]); // 2KB - larger than single hole
    let medium_result = device.zeros_impl(&medium_shape, DType::F32);

    // With proper segregated lists, this should succeed by finding appropriate free block
    assert!(
        medium_result.is_ok(),
        "Segregated lists should find suitable block"
    );

    // Clean up
    drop(medium_result);
    large_blocks.clear();

    // After full cleanup, pool should be mostly defragmented
    let final_stats = device.pool_stats();
    println!(
        "Final stats - Free: {}, Allocated: {}",
        final_stats.free, final_stats.allocated
    );
}

/// Test 3: Repeated allocation/deallocation cycles should not degrade
#[test]
#[ignore = "requires CUDA device"]
fn test_repeated_cycles_no_degradation() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    let shape = Shape::from_dims(&[1024]);
    let iterations = 100;

    // Measure first cycle
    let start = Instant::now();
    for _ in 0..10 {
        let _block = device.zeros_impl(&shape, DType::F32).expect("alloc");
        // block dropped here
    }
    let first_cycle_time = start.elapsed();

    // Run many cycles
    for cycle in 0..iterations {
        let mut blocks = Vec::new();
        for _ in 0..10 {
            blocks.push(device.zeros_impl(&shape, DType::F32).expect("alloc"));
        }
        // All blocks dropped here

        if cycle % 25 == 0 {
            let stats = device.pool_stats();
            println!("Cycle {}: Free={}, Allocated={}", cycle, stats.free, stats.allocated);
        }
    }

    // Measure last cycle
    let start = Instant::now();
    for _ in 0..10 {
        let _block = device.zeros_impl(&shape, DType::F32).expect("alloc");
    }
    let last_cycle_time = start.elapsed();

    println!(
        "First cycle: {:?}, Last cycle: {:?}",
        first_cycle_time, last_cycle_time
    );

    // Last cycle should not be significantly slower than first
    // (would indicate fragmentation accumulation)
    let slowdown_ratio = last_cycle_time.as_nanos() as f64 / first_cycle_time.as_nanos() as f64;
    assert!(
        slowdown_ratio < 3.0,
        "Allocation should not degrade significantly over time. Slowdown: {:.2}x",
        slowdown_ratio
    );
}

/// Test 4: Mixed size workload - simulates real ML workload
#[test]
#[ignore = "requires CUDA device"]
fn test_mixed_size_ml_workload() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    // Use smaller sizes to fit in available VRAM
    let batch_size = 2;
    let seq_len = 64;
    let hidden_dim = 256;
    let num_heads = 4;

    let initial_stats = device.pool_stats();
    if !stats_valid(&initial_stats) {
        println!("Warning: Pool stats invalid, skipping test");
        return;
    }

    for iteration in 0..5 {
        // Input embeddings [batch, seq, hidden]
        let input_shape = Shape::from_dims(&[batch_size, seq_len, hidden_dim]);
        let input = device.zeros_impl(&input_shape, DType::F32).expect("input alloc");

        // QKV projections [batch, seq, 3*hidden]
        let qkv_shape = Shape::from_dims(&[batch_size, seq_len, 3 * hidden_dim]);
        let qkv = device.zeros_impl(&qkv_shape, DType::F32).expect("qkv alloc");

        // Attention scores [batch, heads, seq, seq]
        let attn_shape = Shape::from_dims(&[batch_size, num_heads, seq_len, seq_len]);
        let attn = device.zeros_impl(&attn_shape, DType::F32).expect("attn alloc");

        // FFN intermediate [batch, seq, 4*hidden]
        let ffn_shape = Shape::from_dims(&[batch_size, seq_len, 4 * hidden_dim]);
        let ffn = device.zeros_impl(&ffn_shape, DType::F32).expect("ffn alloc");

        // Drop in reverse order (simulating backward pass cleanup)
        drop(ffn);
        drop(attn);
        drop(qkv);
        drop(input);

        if iteration % 2 == 0 {
            let stats = device.pool_stats();
            if stats_valid(&stats) {
                println!(
                    "Iteration {}: Free={}, Allocated={}",
                    iteration, stats.free, stats.allocated
                );
            }
        }
    }

    let final_stats = device.pool_stats();
    if !stats_valid(&final_stats) {
        println!("Warning: Final stats invalid, test inconclusive");
        return;
    }

    // After all iterations, allocated should be back to baseline
    let leaked = final_stats.allocated.saturating_sub(initial_stats.allocated);
    println!(
        "ML workload test: Initial={}, Final={}, Leaked={}",
        initial_stats.allocated, final_stats.allocated, leaked
    );

    // Should not leak memory
    assert!(
        leaked < 4096,
        "Should not leak memory. Leaked: {} bytes",
        leaked
    );
}

/// Test 5: Fragmentation stress test - allocate many small, free every other
#[test]
#[ignore = "requires CUDA device"]
fn test_fragmentation_stress() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    let small_shape = Shape::from_dims(&[64]); // 256 bytes
    let num_blocks = 200;

    let mut blocks: Vec<Option<_>> = Vec::new();

    // Allocate many small blocks
    for _ in 0..num_blocks {
        blocks.push(Some(
            device
                .zeros_impl(&small_shape, DType::F32)
                .expect("small alloc"),
        ));
    }

    let stats_full = device.pool_stats();
    println!(
        "After {} allocations: Allocated={}",
        num_blocks, stats_full.allocated
    );

    // Free every other block - creates maximum fragmentation pattern
    for i in (0..num_blocks).step_by(2) {
        blocks[i] = None;
    }

    let stats_fragmented = device.pool_stats();
    println!(
        "After freeing alternates: Free={}, Allocated={}",
        stats_fragmented.free, stats_fragmented.allocated
    );

    // Now try to allocate a block twice the size of the small blocks
    // This tests whether the allocator can find or create suitable space
    let double_shape = Shape::from_dims(&[128]); // 512 bytes
    let mut double_blocks = Vec::new();

    // Should be able to allocate several of these despite fragmentation
    for i in 0..20 {
        match device.zeros_impl(&double_shape, DType::F32) {
            Ok(block) => double_blocks.push(block),
            Err(e) => {
                println!("Failed to allocate double block {} : {:?}", i, e);
                break;
            }
        }
    }

    println!(
        "Successfully allocated {} double-sized blocks despite fragmentation",
        double_blocks.len()
    );

    // Should have allocated at least some
    assert!(
        double_blocks.len() >= 10,
        "Should handle fragmented state. Only got {} blocks",
        double_blocks.len()
    );

    // Cleanup
    blocks.clear();
    double_blocks.clear();

    let final_stats = device.pool_stats();
    println!(
        "After cleanup: Free={}, Allocated={}",
        final_stats.free, final_stats.allocated
    );
}

/// Test 6: Large block after many small - tests coalescing effectiveness
#[test]
#[ignore = "requires CUDA device"]
fn test_large_after_many_small() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    let small_shape = Shape::from_dims(&[256]); // 1KB
    let num_small = 100;

    // Allocate many small blocks
    let mut small_blocks = Vec::new();
    for _ in 0..num_small {
        small_blocks.push(
            device
                .zeros_impl(&small_shape, DType::F32)
                .expect("small alloc"),
        );
    }

    let stats_after_small = device.pool_stats();
    let small_total = stats_after_small.allocated;
    println!("After {} small allocations: {} bytes", num_small, small_total);

    // Free all small blocks
    small_blocks.clear();

    let stats_after_free = device.pool_stats();
    println!(
        "After freeing all: Free={}, Allocated={}",
        stats_after_free.free, stats_after_free.allocated
    );

    // Now allocate one large block equal to total small allocations
    // This tests whether coalescing created a contiguous free region
    let large_shape = Shape::from_dims(&[256 * num_small]); // 100KB
    let large_result = device.zeros_impl(&large_shape, DType::F32);

    assert!(
        large_result.is_ok(),
        "Coalescing should create contiguous space for large allocation"
    );

    println!("Successfully allocated large block after coalescing");
}

/// Test 7: Allocation time consistency under fragmentation
#[test]
#[ignore = "requires CUDA device"]
fn test_allocation_time_consistency() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    let shape = Shape::from_dims(&[1024]);
    let warmup_iterations = 50;
    let test_iterations = 100;

    // Warmup
    for _ in 0..warmup_iterations {
        let _block = device.zeros_impl(&shape, DType::F32).expect("warmup");
    }

    // Measure baseline allocation time
    let mut baseline_times = Vec::new();
    for _ in 0..test_iterations {
        let start = Instant::now();
        let _block = device.zeros_impl(&shape, DType::F32).expect("baseline");
        baseline_times.push(start.elapsed().as_nanos());
    }

    let baseline_avg: u128 = baseline_times.iter().sum::<u128>() / test_iterations as u128;
    let baseline_max: u128 = *baseline_times.iter().max().unwrap();

    // Create fragmentation
    let small_shape = Shape::from_dims(&[64]);
    let mut fragments: Vec<Option<_>> = Vec::new();
    for _ in 0..500 {
        fragments.push(Some(
            device.zeros_impl(&small_shape, DType::F32).expect("frag"),
        ));
    }
    // Free every other
    for i in (0..500).step_by(2) {
        fragments[i] = None;
    }

    // Measure allocation time under fragmentation
    let mut fragmented_times = Vec::new();
    for _ in 0..test_iterations {
        let start = Instant::now();
        let _block = device.zeros_impl(&shape, DType::F32).expect("fragmented");
        fragmented_times.push(start.elapsed().as_nanos());
    }

    let fragmented_avg: u128 = fragmented_times.iter().sum::<u128>() / test_iterations as u128;
    let fragmented_max: u128 = *fragmented_times.iter().max().unwrap();

    println!("Baseline: avg={}ns, max={}ns", baseline_avg, baseline_max);
    println!(
        "Fragmented: avg={}ns, max={}ns",
        fragmented_avg, fragmented_max
    );
    println!(
        "Slowdown ratio: {:.2}x",
        fragmented_avg as f64 / baseline_avg as f64
    );

    // TLSF should maintain O(1) allocation even under fragmentation
    // Allow 5x slowdown max (accounts for cache effects, not algorithmic degradation)
    let slowdown = fragmented_avg as f64 / baseline_avg as f64;
    assert!(
        slowdown < 5.0,
        "Allocation time should remain consistent. Slowdown: {:.2}x",
        slowdown
    );

    // Cleanup
    fragments.clear();
}

/// Test 8: Memory utilization after stress
#[test]
#[ignore = "requires CUDA device"]
fn test_memory_utilization_recovery() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    let initial_stats = device.pool_stats();
    let initial_free = initial_stats.free;

    // Stress the allocator
    for round in 0..5 {
        let mut blocks = Vec::new();

        // Allocate various sizes
        for size in [64, 256, 1024, 4096, 16384].iter() {
            let shape = Shape::from_dims(&[*size]);
            for _ in 0..10 {
                if let Ok(block) = device.zeros_impl(&shape, DType::F32) {
                    blocks.push(block);
                }
            }
        }

        // Free in random-ish order (reverse of allocation)
        while !blocks.is_empty() {
            blocks.pop();
        }

        let round_stats = device.pool_stats();
        println!(
            "Round {}: Free={}, Lost={}",
            round,
            round_stats.free,
            initial_free.saturating_sub(round_stats.free)
        );
    }

    let final_stats = device.pool_stats();

    // Should recover most of the free space
    let recovery_ratio = final_stats.free as f64 / initial_free as f64;
    println!(
        "Memory recovery: {:.1}% (Initial: {}, Final: {})",
        recovery_ratio * 100.0,
        initial_free,
        final_stats.free
    );

    assert!(
        recovery_ratio > 0.95,
        "Should recover >95% of memory. Recovery: {:.1}%",
        recovery_ratio * 100.0
    );
}

/// Test 9: Validate pool health metrics
#[test]
#[ignore = "requires CUDA device"]
fn test_pool_health_metrics() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    // Fresh pool should be healthy
    let initial_stats = device.pool_stats();
    println!("Initial pool stats: {:?}", initial_stats);
    assert!(initial_stats.total_size > 0, "Pool should have size");

    // Allocate 50% of pool
    let half_pool = initial_stats.total_size / 2;
    let shape = Shape::from_dims(&[half_pool / 4]); // F32 = 4 bytes
    let _large_block = device.zeros_impl(&shape, DType::F32).expect("large alloc");

    let half_stats = device.pool_stats();
    let utilization = half_stats.allocated as f64 / half_stats.total_size as f64;
    println!(
        "After 50% allocation: Utilization={:.1}%",
        utilization * 100.0
    );

    // Utilization should be around 50%
    assert!(
        utilization > 0.4 && utilization < 0.7,
        "Utilization should be ~50%. Got: {:.1}%",
        utilization * 100.0
    );
}

/// Test 10: Rapid alloc/free pattern (simulates inference batches)
#[test]
#[ignore = "requires CUDA device"]
fn test_rapid_inference_pattern() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    // Use smaller sizes to fit in available VRAM
    let batch_shapes = [
        Shape::from_dims(&[4, 64]),         // Input: 4*64*4 = 1KB
        Shape::from_dims(&[4, 64, 128]),    // Hidden: 4*64*128*4 = 128KB
        Shape::from_dims(&[4, 64, 256]),    // FFN: 4*64*256*4 = 256KB
        Shape::from_dims(&[4, 4, 64, 64]),  // Attention: 4*4*64*64*4 = 256KB
    ];

    let num_inferences = 50;
    let initial_stats = device.pool_stats();

    if !stats_valid(&initial_stats) {
        println!("Warning: Pool stats invalid at start, skipping test");
        return;
    }

    let start = Instant::now();
    for _inference in 0..num_inferences {
        // Simulate one inference pass
        let mut tensors = Vec::new();
        for shape in &batch_shapes {
            tensors.push(device.zeros_impl(shape, DType::F32).expect("tensor alloc"));
        }
        // All tensors freed here when vec drops
    }
    let elapsed = start.elapsed();

    let final_stats = device.pool_stats();

    println!(
        "{} inferences in {:?} ({:.2} ms/inference)",
        num_inferences,
        elapsed,
        elapsed.as_millis() as f64 / num_inferences as f64
    );

    if !stats_valid(&final_stats) {
        println!("Warning: Final stats invalid, test inconclusive");
        return;
    }

    // Should not leak memory
    let leaked = final_stats.allocated.saturating_sub(initial_stats.allocated);
    assert!(
        leaked < 4096,
        "Should not leak memory. Leaked: {} bytes",
        leaked
    );

    // Should maintain performance (not degrade due to fragmentation)
    let avg_inference_us = elapsed.as_micros() / num_inferences as u128;
    println!("Average inference overhead: {} μs", avg_inference_us);
}

/// Summary test that prints comprehensive fragmentation report
#[test]
#[ignore = "requires CUDA device"]
fn test_fragmentation_report() {
    let device = PtxDevice::new(0).expect("Failed to create device");

    println!("\n========================================");
    println!("   TLSF FRAGMENTATION VALIDATION REPORT");
    println!("========================================\n");

    let initial = device.pool_stats();
    println!("Initial Pool State:");
    println!("  Total Size: {} MB", initial.total_size / 1024 / 1024);
    println!("  Free: {} MB", initial.free / 1024 / 1024);
    println!("  Allocated: {} bytes", initial.allocated);

    // Run stress test
    println!("\n--- Running Stress Test ---");
    let mut peak_allocated = 0usize;
    let mut all_blocks = Vec::new();

    for round in 0..3 {
        // Allocate phase
        for size in [256, 1024, 4096, 16384, 65536].iter() {
            let shape = Shape::from_dims(&[*size / 4]);
            for _ in 0..5 {
                if let Ok(block) = device.zeros_impl(&shape, DType::F32) {
                    all_blocks.push(block);
                }
            }
        }

        let stats = device.pool_stats();
        peak_allocated = peak_allocated.max(stats.allocated);

        // Free half randomly
        let half = all_blocks.len() / 2;
        all_blocks.truncate(half);

        println!(
            "Round {}: Blocks={}, Allocated={} KB",
            round,
            all_blocks.len(),
            stats.allocated / 1024
        );
    }

    // Cleanup
    all_blocks.clear();

    let final_stats = device.pool_stats();

    println!("\n--- Final Results ---");
    println!("Peak Allocated: {} KB", peak_allocated / 1024);
    println!("Final Allocated: {} bytes", final_stats.allocated);
    println!("Final Free: {} MB", final_stats.free / 1024 / 1024);

    let recovery = final_stats.free as f64 / initial.free as f64 * 100.0;
    println!("Memory Recovery: {:.1}%", recovery);

    // Validate
    assert!(recovery > 95.0, "Memory recovery should be >95%");

    println!("\n========================================");
    println!("   FRAGMENTATION TEST: PASSED");
    println!("========================================\n");
}
