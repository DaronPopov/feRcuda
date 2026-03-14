//! Example: Using PTX-OS pre-warmed streams with cudarc-ptx
//!
//! This demonstrates how to use the PTX-OS stream pool for faster
//! kernel dispatch in candle operations.
//!
//! Run with: cargo run --example ptx_streams

use candle_ptx_os::prelude::*;
use candle_ptx_os::cudarc_bridge::PtxCudaBridge;

fn main() -> Result<()> {
    println!("PTX-OS Stream Integration Example");
    println!("==================================\n");

    // Initialize the bridge (creates KernelAccelerator + cudarc-ptx context)
    println!("Initializing PTX-OS stream pool...");
    let bridge = PtxCudaBridge::from_global(0)?;

    println!("  Context created on device {}", bridge.accelerator().device_id());
    println!("  {} pre-warmed streams ready", bridge.num_streams());
    println!();

    // Get streams - they're distributed round-robin
    println!("Getting streams (round-robin):");
    for i in 0..5 {
        let stream = bridge.next_stream();
        println!("  Stream {}: {:?} (external: {})",
            i,
            stream.cu_stream(),
            stream.is_external()
        );
    }
    println!();

    // Priority stream for latency-critical operations
    println!("Priority stream for latency-critical ops:");
    let priority = bridge.priority_stream();
    println!("  Priority stream: {:?}", priority.cu_stream());
    println!();

    // Show statistics
    let stats = bridge.stats();
    println!("Accelerator Statistics:");
    println!("  {}", stats);
    println!();

    // Sync all streams
    println!("Synchronizing all streams...");
    bridge.sync_all()?;
    println!("  Done!");
    println!();

    // Example: Using with kernel launches
    println!("Using with kernel launches:");
    println!("  bridge.launch_on_ptx(|stream| {{");
    println!("      stream.launch_builder(&kernel)");
    println!("          .arg(&data)");
    println!("          .launch(cfg)");
    println!("  }});");
    println!();

    // Global bridge for convenience
    println!("Global bridge access:");
    let global = global_bridge(0)?;
    println!("  global_bridge(0) -> {} streams", global.num_streams());

    println!("\nExample complete!");
    Ok(())
}
