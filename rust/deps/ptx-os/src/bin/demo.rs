//! PTX-OS Demo - Demonstrates the persistent GPU runtime

use ptx_os::prelude::*;
use ptx_os::{VirtualFs, VirtualMemory};

fn main() -> Result<()> {
    println!("========================================");
    println!("  PTX-OS: Persistent GPU Operating System");
    println!("========================================\n");

    // Initialize runtime on GPU 0
    println!("[1] Initializing feR-os runtime...");
    let runtime = RegimeRuntimeCore::new(0)?;
    println!("    Device ID: {}", runtime.device_id());

    // Print initial stats
    let stats = runtime.stats();
    println!("    VRAM: {:.2} MB used / {:.2} MB free",
             stats.vram_used as f64 / 1024.0 / 1024.0,
             stats.vram_free as f64 / 1024.0 / 1024.0);
    println!();

    // Test memory allocation
    println!("[2] Testing Memory Allocation...");
    let tensor = Tensor::zeros(&runtime, &[1024, 1024], ptx_os::tensor::DType::Float32)?;
    println!("    Allocated tensor: {:?}", tensor.shape());
    println!("    Size: {:.2} MB", tensor.size_bytes() as f64 / 1024.0 / 1024.0);

    let pool_stats = runtime.pool_stats();
    println!("    Pool utilization: {:.1}%", pool_stats.utilization_percent);
    println!("    Pool healthy: {}", pool_stats.is_healthy);
    println!();

    // Initialize VFS
    println!("[3] Initializing Virtual Filesystem...");
    let vfs = VirtualFs::new(&runtime)?;

    // Create directory structure
    vfs.mkdir("/models", 0o755)?;
    vfs.mkdir("/models/weights", 0o755)?;
    vfs.mkdir("/tensors", 0o755)?;
    println!("    Created: /models, /models/weights, /tensors");

    // Create a tensor in VFS
    vfs.create_tensor("/tensors/activations", &[256, 512], ptx_os::tensor::DType::Float32)?;
    println!("    Created tensor: /tensors/activations [256x512]");

    // Memory-map the tensor
    let tensor_ptr = vfs.mmap_tensor("/tensors/activations")?;
    println!("    Mapped tensor at: {:?}", tensor_ptr);
    println!();

    // Initialize VMM
    println!("[4] Initializing Virtual Memory Manager...");
    let swap_size = 256 * 1024 * 1024; // 256 MB swap
    let vmm = VirtualMemory::new(&runtime, swap_size)?;

    // Allocate some pages
    let mut page1 = vmm.alloc_page(ptx_os::vmm::PageFlags::readwrite())?;
    let _page2 = vmm.alloc_page(ptx_os::vmm::PageFlags::readwrite().pinned())?;
    println!("    Allocated 2 pages (1 pinned)");

    let vmm_stats = vmm.stats();
    println!("    Resident pages: {}", vmm_stats.resident_pages);
    println!();

    // Test swap operations
    println!("[5] Testing Swap Operations...");
    page1.swap_out()?;
    println!("    Page 1 swapped out to host");

    let vmm_stats = vmm.stats();
    println!("    Swapped pages: {}", vmm_stats.swapped_pages);

    page1.swap_in()?;
    println!("    Page 1 swapped back in");
    println!();

    // Boot persistent kernel (optional, requires CUDA device)
    println!("[6] System State...");
    let snapshot = runtime.system_snapshot();
    println!("    Total ops: {}", snapshot.total_ops);
    println!("    Active processes: {}", snapshot.active_processes);
    println!("    Kernel running: {}", snapshot.kernel_running);
    println!();

    // Final stats
    println!("[7] Final Statistics...");
    let final_stats = runtime.stats();
    println!("    Total operations: {}", final_stats.total_ops);
    println!("    VRAM used: {:.2} MB", final_stats.vram_used as f64 / 1024.0 / 1024.0);

    runtime.print_pool_map();

    println!("\n========================================");
    println!("  Demo Complete!");
    println!("========================================");

    Ok(())
}
