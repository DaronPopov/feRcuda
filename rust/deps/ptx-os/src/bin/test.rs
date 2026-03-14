//! PTX-OS Integration Tests

use ptx_os::prelude::*;
use ptx_os::{VirtualFs, VirtualMemory};

fn test_runtime_init() -> Result<()> {
    println!("Test: Runtime Initialization...");
    let runtime = RegimeRuntimeCore::new(0)?;
    assert!(runtime.device_id() >= 0);
    println!("  PASS: Runtime initialized on device {}", runtime.device_id());
    Ok(())
}

fn test_config_api() -> Result<()> {
    println!("Test: Configuration API...");

    // Test with 90% of available VRAM
    let regime = RegimeConfig::new().pool_fraction(0.90).quiet();
    let runtime = RegimeRuntimeCore::with_regime(0, regime)?;
    let stats = runtime.pool_stats();
    println!("  PASS: 90% fraction config - pool size: {:.2} GB",
             stats.total_size as f64 / (1024.0 * 1024.0 * 1024.0));
    drop(runtime);

    // Test with fixed 1GB pool
    let regime = RegimeConfig::new().pool_fixed_bytes(1024 * 1024 * 1024).quiet();
    let runtime = RegimeRuntimeCore::with_regime(0, regime)?;
    let stats = runtime.pool_stats();
    assert!(stats.total_size >= 1024 * 1024 * 1024 - 1024 * 1024); // Allow some overhead
    println!("  PASS: Fixed 1GB config - pool size: {:.2} GB",
             stats.total_size as f64 / (1024.0 * 1024.0 * 1024.0));
    drop(runtime);

    // Test max_vram convenience constructor
    let runtime = RegimeRuntimeCore::with_regime(0, RegimeConfig::max_vram().quiet())?;
    let stats = runtime.pool_stats();
    println!("  PASS: max_vram() - pool size: {:.2} GB",
             stats.total_size as f64 / (1024.0 * 1024.0 * 1024.0));

    Ok(())
}

fn test_memory_allocation() -> Result<()> {
    println!("Test: Memory Allocation...");
    let runtime = RegimeRuntimeCore::new(0)?;

    // Test small allocation
    let small = Tensor::zeros(&runtime, &[100], ptx_os::tensor::DType::Float32)?;
    assert_eq!(small.numel(), 100);
    println!("  PASS: Small allocation (100 elements)");

    // Test large allocation
    let large = Tensor::zeros(&runtime, &[1024, 1024], ptx_os::tensor::DType::Float32)?;
    assert_eq!(large.numel(), 1024 * 1024);
    println!("  PASS: Large allocation (1M elements)");

    // Test can_allocate
    let can_alloc = runtime.can_allocate(1024);
    assert!(can_alloc);
    println!("  PASS: can_allocate check");

    // Test max_allocatable
    let max_alloc = runtime.max_allocatable();
    assert!(max_alloc > 0);
    println!("  PASS: max_allocatable = {} bytes", max_alloc);

    Ok(())
}

fn test_pool_health() -> Result<()> {
    println!("Test: Pool Health...");
    let runtime = RegimeRuntimeCore::new(0)?;

    let stats = runtime.pool_stats();
    assert!(stats.is_healthy);
    println!("  PASS: Pool is healthy");

    assert!(stats.utilization_percent < 100.0);
    println!("  PASS: Utilization = {:.1}%", stats.utilization_percent);

    Ok(())
}

fn test_vfs_operations() -> Result<()> {
    println!("Test: VFS Operations...");
    let runtime = RegimeRuntimeCore::new(0)?;
    let vfs = VirtualFs::new(&runtime)?;

    // Test mkdir
    vfs.mkdir("/test", 0o755)?;
    println!("  PASS: mkdir /test");

    // Test nested mkdir
    vfs.mkdir("/test/subdir", 0o755)?;
    println!("  PASS: mkdir /test/subdir");

    // Test tensor creation
    vfs.create_tensor("/test/tensor", &[64, 64], ptx_os::tensor::DType::Float32)?;
    println!("  PASS: create_tensor /test/tensor");

    // Test mmap
    let ptr = vfs.mmap_tensor("/test/tensor")?;
    assert!(!ptr.is_null());
    println!("  PASS: mmap_tensor -> {:?}", ptr);

    // Test sync
    vfs.sync_tensor("/test/tensor")?;
    println!("  PASS: sync_tensor");

    // Cleanup
    vfs.rmdir("/test/subdir")?;
    println!("  PASS: rmdir /test/subdir");

    Ok(())
}

fn test_vmm_operations() -> Result<()> {
    println!("Test: VMM Operations...");
    let runtime = RegimeRuntimeCore::new(0)?;
    let vmm = VirtualMemory::new(&runtime, 64 * 1024 * 1024)?; // 64MB swap

    // Test page allocation
    let mut page = vmm.alloc_page(ptx_os::vmm::PageFlags::readwrite())?;
    println!("  PASS: alloc_page");

    // Test stats
    let stats = vmm.stats();
    assert!(stats.resident_pages > 0);
    println!("  PASS: resident_pages = {}", stats.resident_pages);

    // Test swap out
    page.swap_out()?;
    println!("  PASS: swap_out");

    let stats = vmm.stats();
    assert!(stats.swapped_pages > 0);
    println!("  PASS: swapped_pages = {}", stats.swapped_pages);

    // Test swap in
    page.swap_in()?;
    println!("  PASS: swap_in");

    // Test pin/unpin
    page.pin();
    println!("  PASS: pin");
    page.unpin();
    println!("  PASS: unpin");

    Ok(())
}

fn test_shared_memory() -> Result<()> {
    println!("Test: Shared Memory...");
    let runtime = RegimeRuntimeCore::new(0)?;

    // Create shared segment
    let ptr = runtime.shm_alloc("test_segment", 1024)?;
    assert!(!ptr.is_null());
    println!("  PASS: shm_alloc -> {:?}", ptr);

    // Open existing segment
    let ptr2 = runtime.shm_open("test_segment")?;
    assert_eq!(ptr, ptr2);
    println!("  PASS: shm_open returns same pointer");

    // Cleanup
    runtime.shm_unlink("test_segment");
    println!("  PASS: shm_unlink");

    Ok(())
}

fn test_watchdog() -> Result<()> {
    println!("Test: Watchdog...");
    let runtime = RegimeRuntimeCore::new(0)?;

    runtime.set_watchdog(1000);
    println!("  PASS: set_watchdog(1000ms)");

    let tripped = runtime.check_watchdog();
    println!("  PASS: check_watchdog = {}", tripped);

    runtime.reset_watchdog();
    println!("  PASS: reset_watchdog");

    Ok(())
}

fn main() {
    println!("========================================");
    println!("  PTX-OS Integration Tests");
    println!("========================================\n");

    let tests: Vec<(&str, fn() -> Result<()>)> = vec![
        ("Runtime Init", test_runtime_init),
        ("Config API", test_config_api),
        ("Memory Allocation", test_memory_allocation),
        ("Pool Health", test_pool_health),
        ("VFS Operations", test_vfs_operations),
        ("VMM Operations", test_vmm_operations),
        ("Shared Memory", test_shared_memory),
        ("Watchdog", test_watchdog),
    ];

    let mut passed = 0;
    let mut failed = 0;

    for (name, test_fn) in tests {
        match test_fn() {
            Ok(()) => {
                println!("  [PASS] {}\n", name);
                passed += 1;
            }
            Err(e) => {
                println!("  [FAIL] {}: {:?}\n", name, e);
                failed += 1;
            }
        }
    }

    println!("========================================");
    println!("  Results: {} passed, {} failed", passed, failed);
    println!("========================================");

    if failed > 0 {
        std::process::exit(1);
    }
}
