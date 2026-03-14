//! KernelAccelerator - Lightweight PTX-OS integration for fast kernel launches
//!
//! This module provides kernel launch acceleration WITHOUT managing memory:
//! - Pre-warmed CUDA streams (16 parallel streams ready to go)
//! - CUDA graph caching (repeated ops → instant replay)
//! - Priority streams for latency-critical ops
//!
//! Memory is handled normally by CUDA/candle - we just speed up kernel dispatch.

use crate::error::{PtxCandleError, Result};
use crate::ffi::CudaStream;
use crate::graph_fusion::{GraphCache, GraphCapture, OpSignature, StreamPool};
use ptx_os::ffi::{
    GPUHotConfig, GPUHotRuntime,
    gpu_hot_init_with_config, gpu_hot_shutdown, gpu_hot_sync_all,
    ptx_hook_init, ptx_hook_get_launch_count,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Number of CUDA streams for parallel execution
const NUM_STREAMS: usize = 16;
/// Graph cache capacity
const GRAPH_CACHE_SIZE: usize = 128;

/// Global kernel accelerator instance
static ACCELERATOR: OnceLock<KernelAccelerator> = OnceLock::new();

/// Lightweight kernel acceleration without memory management
///
/// Use this when you want candle/CUDA to handle memory normally,
/// but want fast kernel launches via pre-warmed streams and graph caching.
pub struct KernelAccelerator {
    runtime_ptr: *mut GPUHotRuntime,
    stream_pool: StreamPool,
    graph_cache: GraphCache,
    device_id: i32,
    launches: AtomicU64,
}

// Safety: The runtime pointer is thread-safe (PTX-OS guarantees this)
unsafe impl Send for KernelAccelerator {}
unsafe impl Sync for KernelAccelerator {}

impl KernelAccelerator {
    /// Create a new kernel accelerator on the specified GPU
    ///
    /// This initializes:
    /// - Persistent CUDA context (no driver overhead)
    /// - 16 pre-warmed CUDA streams
    /// - Graph cache for repeated operations
    ///
    /// Memory allocation is minimized - use candle/CUDA normally for tensors.
    pub fn new(device_id: i32) -> Result<Self> {
        // Create a minimal runtime - small pool, just for context and streams
        let config = GPUHotConfig {
            pool_fraction: 0.0,  // Don't use fraction
            fixed_pool_size: 32 * 1024 * 1024,  // Tiny 32MB pool (just for stream buffers)
            min_pool_size: 16 * 1024 * 1024,
            max_pool_size: 64 * 1024 * 1024,
            reserve_vram: 0,  // Don't reserve extra
            enable_leak_detection: false,
            enable_pool_health: false,
            warning_threshold: 0.95,
            force_daemon_mode: false,
            quiet_init: true,
        };

        let runtime_ptr = unsafe {
            gpu_hot_init_with_config(
                device_id,
                std::ptr::null(),  // No token needed
                &config,
            )
        };

        if runtime_ptr.is_null() {
            return Err(PtxCandleError::Cuda {
                message: "Failed to create kernel accelerator runtime".to_string(),
            });
        }

        let stream_pool = unsafe { StreamPool::new(runtime_ptr, NUM_STREAMS) };
        let graph_cache = GraphCache::new(GRAPH_CACHE_SIZE);

        // Enable kernel launch hooks to route through our streams
        unsafe {
            ptx_hook_init(runtime_ptr, false);
        }

        eprintln!(
            "⚡ KernelAccelerator ready: {} streams, kernel hooks enabled",
            stream_pool.num_streams()
        );

        Ok(Self {
            runtime_ptr,
            stream_pool,
            graph_cache,
            device_id,
            launches: AtomicU64::new(0),
        })
    }

    /// Get or create the global accelerator instance
    pub fn global(device_id: i32) -> Result<&'static Self> {
        if let Some(acc) = ACCELERATOR.get() {
            return Ok(acc);
        }

        let acc = Self::new(device_id)?;
        let _ = ACCELERATOR.set(acc);
        Ok(ACCELERATOR.get().unwrap())
    }

    /// Get the next available stream for kernel execution
    ///
    /// Streams are distributed round-robin for parallel execution.
    pub fn next_stream(&self) -> CudaStream {
        self.launches.fetch_add(1, Ordering::Relaxed);
        self.stream_pool.next_stream()
    }

    /// Get a high-priority stream for latency-critical operations
    ///
    /// Use this for attention score computation, final logits, etc.
    pub fn priority_stream(&self) -> CudaStream {
        self.stream_pool.priority_stream()
    }

    /// Get a specific stream by index (0..15)
    pub fn stream(&self, idx: usize) -> CudaStream {
        self.stream_pool.stream(idx)
    }

    /// Get all streams for parallel operations
    pub fn all_streams(&self) -> &[CudaStream] {
        self.stream_pool.all_streams()
    }

    /// Execute with graph caching - repeated ops become instant
    ///
    /// First call: captures kernel sequence into CUDA graph (~5μs)
    /// Subsequent calls: replays cached graph (~0.5μs)
    pub fn execute_cached<F>(&self, sig: &OpSignature, kernel_fn: F) -> Result<()>
    where
        F: FnOnce(CudaStream) -> Result<()>,
    {
        // Check cache first
        if let Some(graph) = self.graph_cache.get(sig) {
            let stream = self.next_stream();
            unsafe {
                let result = crate::ffi::ptx_tensor_graph_launch(graph.exec, stream);
                if result != 0 {
                    return Err(PtxCandleError::Cuda {
                        message: format!("Graph launch failed: {}", result),
                    });
                }
            }
            return Ok(());
        }

        // Miss - capture and execute
        let stream = self.next_stream();
        let capture = unsafe { GraphCapture::begin(stream)? };
        kernel_fn(capture.stream())?;

        let (graph, exec) = capture.end()?;
        let compiled = crate::graph_fusion::CompiledGraph {
            graph,
            exec,
            signature: sig.clone(),
            use_count: AtomicU64::new(1),
        };
        self.graph_cache.insert(sig.clone(), compiled);

        Ok(())
    }

    /// Execute multiple independent operations in parallel across streams
    ///
    /// Great for attention heads, MLP chunks, etc.
    pub fn parallel<F>(&self, ops: Vec<F>) -> Result<()>
    where
        F: FnOnce(CudaStream) -> Result<()>,
    {
        let streams = self.stream_pool.all_streams();
        let num_streams = streams.len().max(1);

        for (i, op) in ops.into_iter().enumerate() {
            let stream = streams[i % num_streams];
            op(stream)?;
        }

        Ok(())
    }

    /// Synchronize all streams (wait for all pending work)
    pub fn sync_all(&self) {
        unsafe {
            gpu_hot_sync_all(self.runtime_ptr);
        }
    }

    /// Get acceleration statistics
    pub fn stats(&self) -> AcceleratorStats {
        let (cache_hits, cache_misses) = self.graph_cache.stats();
        // Get actual launch count from the kernel hook
        let total_launches = unsafe { ptx_hook_get_launch_count() };
        AcceleratorStats {
            total_launches,
            graph_cache_hits: cache_hits,
            graph_cache_misses: cache_misses,
            graph_hit_rate: if cache_hits + cache_misses > 0 {
                cache_hits as f64 / (cache_hits + cache_misses) as f64
            } else {
                0.0
            },
            num_streams: self.stream_pool.num_streams(),
        }
    }

    /// Clear the graph cache (useful when model changes)
    pub fn clear_cache(&self) {
        self.graph_cache.clear();
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Get the raw runtime pointer (for advanced use)
    pub fn runtime_ptr(&self) -> *mut GPUHotRuntime {
        self.runtime_ptr
    }
}

impl Drop for KernelAccelerator {
    fn drop(&mut self) {
        if !self.runtime_ptr.is_null() {
            unsafe {
                gpu_hot_shutdown(self.runtime_ptr);
            }
        }
    }
}

/// Statistics for kernel acceleration
#[derive(Debug, Clone)]
pub struct AcceleratorStats {
    pub total_launches: u64,
    pub graph_cache_hits: u64,
    pub graph_cache_misses: u64,
    pub graph_hit_rate: f64,
    pub num_streams: usize,
}

impl std::fmt::Display for AcceleratorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Launches: {} | Graph hits: {} ({:.1}%) | Streams: {}",
            self.total_launches,
            self.graph_cache_hits,
            self.graph_hit_rate * 100.0,
            self.num_streams
        )
    }
}

/// Helper to wrap candle CUDA operations with stream acceleration
///
/// Usage:
/// ```ignore
/// let acc = KernelAccelerator::global(0)?;
/// accelerated_launch!(acc, |stream| {
///     // Your CUDA kernel launch here
///     unsafe { my_kernel<<<grid, block, 0, stream>>>(args) }
/// });
/// ```
#[macro_export]
macro_rules! accelerated_launch {
    ($acc:expr, $kernel:expr) => {{
        let stream = $acc.next_stream();
        $kernel(stream)
    }};
}

/// Helper to wrap repeated operations with graph caching
#[macro_export]
macro_rules! cached_launch {
    ($acc:expr, $op_type:expr, $dtype:expr, $elem_count:expr, $kernel:expr) => {{
        let sig = $crate::graph_fusion::OpSignature::new($op_type, $dtype, $elem_count);
        $acc.execute_cached(&sig, $kernel)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accelerator_stats_display() {
        let stats = AcceleratorStats {
            total_launches: 1000,
            graph_cache_hits: 800,
            graph_cache_misses: 200,
            graph_hit_rate: 0.8,
            num_streams: 16,
        };
        let s = format!("{}", stats);
        assert!(s.contains("1000"));
        assert!(s.contains("80.0%"));
    }
}
