//! CUDA Graph Fusion - Sub-microsecond kernel launches via graph capture/replay
//!
//! This module provides automatic graph capture and fusion for sequences of tensor
//! operations, reducing kernel launch overhead from ~5μs to ~0.5μs per operation.
//!
//! # Architecture
//!
//! - **GraphCapture**: Records sequences of operations into CUDA graphs
//! - **GraphCache**: Caches compiled graphs by operation signature
//! - **StreamPool**: Manages parallel streams for concurrent execution
//! - **FusedExecutor**: Orchestrates parallel graph execution

use crate::error::{PtxCandleError, Result};
use crate::ffi::{self, CudaStream};
use parking_lot::RwLock;
use ptx_os::ffi::GPUHotRuntime;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// CUDA Graph handle
pub type CudaGraph = *mut c_void;
/// CUDA Graph Executable handle
pub type CudaGraphExec = *mut c_void;

/// Operation signature for graph caching
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct OpSignature {
    pub op_type: u32,
    pub dtype: u32,
    pub elem_count: usize,
    pub extra_dims: [usize; 4], // For reductions, softmax, etc.
}

impl OpSignature {
    pub fn new(op_type: u32, dtype: u32, elem_count: usize) -> Self {
        Self {
            op_type,
            dtype,
            elem_count,
            extra_dims: [0; 4],
        }
    }

    pub fn with_dims(mut self, dims: &[usize]) -> Self {
        for (i, &d) in dims.iter().take(4).enumerate() {
            self.extra_dims[i] = d;
        }
        self
    }
}

/// Compiled graph ready for execution
pub struct CompiledGraph {
    pub graph: CudaGraph,
    pub exec: CudaGraphExec,
    pub signature: OpSignature,
    pub use_count: AtomicU64,
}

impl Drop for CompiledGraph {
    fn drop(&mut self) {
        // Graphs are managed by CUDA runtime, but we should track cleanup
        // Note: In production, call cudaGraphExecDestroy and cudaGraphDestroy
    }
}

unsafe impl Send for CompiledGraph {}
unsafe impl Sync for CompiledGraph {}

/// Graph cache for compiled operation sequences
pub struct GraphCache {
    cache: RwLock<HashMap<OpSignature, Arc<CompiledGraph>>>,
    max_entries: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl GraphCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::with_capacity(max_entries)),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn get(&self, sig: &OpSignature) -> Option<Arc<CompiledGraph>> {
        let cache = self.cache.read();
        if let Some(graph) = cache.get(sig) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            graph.use_count.fetch_add(1, Ordering::Relaxed);
            Some(graph.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn insert(&self, sig: OpSignature, graph: CompiledGraph) -> Arc<CompiledGraph> {
        let arc = Arc::new(graph);
        let mut cache = self.cache.write();

        // Evict least-used entries if at capacity
        if cache.len() >= self.max_entries {
            // Find entry with lowest use count
            if let Some(key) = cache
                .iter()
                .min_by_key(|(_, v)| v.use_count.load(Ordering::Relaxed))
                .map(|(k, _)| k.clone())
            {
                cache.remove(&key);
            }
        }

        cache.insert(sig, arc.clone());
        arc
    }

    pub fn stats(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }

    pub fn clear(&self) {
        self.cache.write().clear();
    }
}

/// Stream pool for parallel execution
pub struct StreamPool {
    streams: Vec<CudaStream>,
    current: AtomicUsize,
    priority_streams: Vec<CudaStream>, // High-priority streams for critical ops
}

impl StreamPool {
    /// Create a new stream pool
    /// Streams are created by the PTX-OS runtime
    ///
    /// # Safety
    /// `runtime_ptr` must be a valid, non-null `GPUHotRuntime` pointer.
    pub unsafe fn new(runtime_ptr: *mut GPUHotRuntime, num_streams: usize) -> Self {
        let mut streams = Vec::with_capacity(num_streams);
        let mut priority_streams = Vec::with_capacity(2);

        unsafe {
            // Get regular streams
            for i in 0..num_streams {
                let stream = ptx_os::ffi::gpu_hot_get_stream(runtime_ptr, i as i32);
                if !stream.is_null() {
                    streams.push(stream);
                }
            }

            // Get priority streams (0 = realtime, 1 = high)
            for priority in 0..2 {
                let stream = ptx_os::ffi::gpu_hot_get_priority_stream(runtime_ptr, priority);
                if !stream.is_null() {
                    priority_streams.push(stream);
                }
            }
        }

        Self {
            streams,
            current: AtomicUsize::new(0),
            priority_streams,
        }
    }

    /// Get the next stream in round-robin fashion
    pub fn next_stream(&self) -> CudaStream {
        if self.streams.is_empty() {
            return std::ptr::null_mut();
        }
        let idx = self.current.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        self.streams[idx]
    }

    /// Get a high-priority stream for critical operations
    pub fn priority_stream(&self) -> CudaStream {
        self.priority_streams.first().copied().unwrap_or(std::ptr::null_mut())
    }

    /// Get a specific stream by index
    pub fn stream(&self, idx: usize) -> CudaStream {
        self.streams.get(idx).copied().unwrap_or(std::ptr::null_mut())
    }

    /// Get all streams for parallel execution
    pub fn all_streams(&self) -> &[CudaStream] {
        &self.streams
    }

    /// Number of available streams
    pub fn num_streams(&self) -> usize {
        self.streams.len()
    }
}

/// Graph capture context for recording operations
pub struct GraphCapture {
    stream: CudaStream,
    is_capturing: bool,
    ops_recorded: usize,
}

impl GraphCapture {
    /// Begin capturing operations on the given stream
    ///
    /// # Safety
    /// `stream` must be a valid CUDA stream.
    pub unsafe fn begin(stream: CudaStream) -> Result<Self> {
        let result = ffi::ptx_tensor_graph_begin_capture(stream);
        if result != 0 {
            return Err(PtxCandleError::Cuda {
                message: format!("Failed to begin graph capture: error {}", result),
            });
        }

        Ok(Self {
            stream,
            is_capturing: true,
            ops_recorded: 0,
        })
    }

    /// Check if currently capturing
    pub fn is_capturing(&self) -> bool {
        self.is_capturing
    }

    /// Get the capture stream
    pub fn stream(&self) -> CudaStream {
        self.stream
    }

    /// Record that an operation was captured
    pub fn record_op(&mut self) {
        self.ops_recorded += 1;
    }

    /// End capture and return the compiled graph
    pub fn end(mut self) -> Result<(CudaGraph, CudaGraphExec)> {
        if !self.is_capturing {
            return Err(PtxCandleError::InvalidArgument(
                "Not currently capturing".to_string(),
            ));
        }

        self.is_capturing = false;

        let mut graph: CudaGraph = std::ptr::null_mut();
        let mut exec: CudaGraphExec = std::ptr::null_mut();

        unsafe {
            let result = ffi::ptx_tensor_graph_end_capture(self.stream, &mut graph);
            if result != 0 {
                return Err(PtxCandleError::Cuda {
                    message: format!("Failed to end graph capture: error {}", result),
                });
            }

            let result = ffi::ptx_tensor_graph_instantiate(graph, &mut exec);
            if result != 0 {
                return Err(PtxCandleError::Cuda {
                    message: format!("Failed to instantiate graph: error {}", result),
                });
            }
        }

        Ok((graph, exec))
    }

    /// Number of operations recorded
    pub fn ops_count(&self) -> usize {
        self.ops_recorded
    }
}

impl Drop for GraphCapture {
    fn drop(&mut self) {
        if self.is_capturing {
            // Abandon capture - this shouldn't happen in normal use
            unsafe {
                let mut graph: CudaGraph = std::ptr::null_mut();
                let _ = ffi::ptx_tensor_graph_end_capture(self.stream, &mut graph);
            }
        }
    }
}

/// Fused executor for parallel graph execution
pub struct FusedExecutor {
    pub stream_pool: StreamPool,
    pub graph_cache: GraphCache,
    runtime_ptr: *mut GPUHotRuntime,
    parallel_threshold: usize, // Min ops to enable parallel execution
}

impl FusedExecutor {
    /// Create a new fused executor
    ///
    /// # Safety
    /// `runtime_ptr` must be a valid, non-null `GPUHotRuntime` pointer.
    pub unsafe fn new(runtime_ptr: *mut GPUHotRuntime, num_streams: usize, cache_size: usize) -> Self {
        Self {
            stream_pool: StreamPool::new(runtime_ptr, num_streams),
            graph_cache: GraphCache::new(cache_size),
            runtime_ptr,
            parallel_threshold: 2,
        }
    }

    /// Launch a cached graph or execute directly
    pub fn execute_cached(
        &self,
        sig: &OpSignature,
        capture_fn: impl FnOnce(CudaStream) -> Result<()>,
    ) -> Result<()> {
        // Check cache first
        if let Some(graph) = self.graph_cache.get(sig) {
            // Fast path: replay cached graph (~0.5μs)
            let stream = self.stream_pool.next_stream();
            unsafe {
                let result = ffi::ptx_tensor_graph_launch(graph.exec, stream);
                if result != 0 {
                    return Err(PtxCandleError::Cuda {
                        message: format!("Failed to launch graph: error {}", result),
                    });
                }
            }
            return Ok(());
        }

        // Slow path: capture and cache
        let stream = self.stream_pool.next_stream();
        let capture = unsafe { GraphCapture::begin(stream)? };

        // Execute the operation (this records to the graph)
        capture_fn(capture.stream())?;

        // End capture and cache
        let (graph, exec) = capture.end()?;
        let compiled = CompiledGraph {
            graph,
            exec,
            signature: sig.clone(),
            use_count: AtomicU64::new(1),
        };
        self.graph_cache.insert(sig.clone(), compiled);

        Ok(())
    }

    /// Execute operations in parallel across multiple streams
    pub fn execute_parallel<F>(&self, ops: Vec<F>) -> Result<()>
    where
        F: FnOnce(CudaStream) -> Result<()> + Send,
    {
        let num_ops = ops.len();
        if num_ops == 0 {
            return Ok(());
        }

        if num_ops < self.parallel_threshold || self.stream_pool.num_streams() < 2 {
            // Sequential execution for small batches
            let stream = self.stream_pool.next_stream();
            for op in ops {
                op(stream)?;
            }
            return Ok(());
        }

        // Parallel execution across streams
        let streams = self.stream_pool.all_streams();
        let num_streams = streams.len();

        for (i, op) in ops.into_iter().enumerate() {
            let stream = streams[i % num_streams];
            op(stream)?;
        }

        Ok(())
    }

    /// Synchronize all streams
    pub fn sync_all(&self) {
        unsafe {
            ptx_os::ffi::gpu_hot_sync_all(self.runtime_ptr);
        }
    }

    /// Get execution statistics
    pub fn stats(&self) -> FusedExecutorStats {
        let (hits, misses) = self.graph_cache.stats();
        FusedExecutorStats {
            graph_cache_hits: hits,
            graph_cache_misses: misses,
            num_streams: self.stream_pool.num_streams(),
        }
    }
}

unsafe impl Send for FusedExecutor {}
unsafe impl Sync for FusedExecutor {}

/// Statistics for the fused executor
#[derive(Debug, Clone)]
pub struct FusedExecutorStats {
    pub graph_cache_hits: u64,
    pub graph_cache_misses: u64,
    pub num_streams: usize,
}

/// Batch operation builder for graph fusion
pub struct BatchBuilder {
    ops: Vec<Box<dyn FnOnce(CudaStream) -> Result<()> + Send>>,
}

impl BatchBuilder {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add<F>(mut self, op: F) -> Self
    where
        F: FnOnce(CudaStream) -> Result<()> + Send + 'static,
    {
        self.ops.push(Box::new(op));
        self
    }

    pub fn execute(self, executor: &FusedExecutor) -> Result<()> {
        let num_ops = self.ops.len();
        if num_ops == 0 {
            return Ok(());
        }

        let streams = executor.stream_pool.all_streams();
        let num_streams = streams.len().max(1);

        for (i, op) in self.ops.into_iter().enumerate() {
            let stream = if streams.is_empty() {
                std::ptr::null_mut()
            } else {
                streams[i % num_streams]
            };
            op(stream)?;
        }

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl Default for BatchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel tensor operation executor
/// Automatically distributes independent operations across streams
pub struct ParallelOps {
    executor: Arc<FusedExecutor>,
}

impl ParallelOps {
    pub fn new(executor: Arc<FusedExecutor>) -> Self {
        Self { executor }
    }

    /// Execute multiple independent unary operations in parallel
    pub fn unary_parallel<F>(&self, inputs: &[(*mut c_void, *mut c_void, usize)], op: F) -> Result<()>
    where
        F: Fn(*mut c_void, *mut c_void, usize, CudaStream) + Send + Sync + Copy,
    {
        let streams = self.executor.stream_pool.all_streams();
        let num_streams = streams.len().max(1);

        for (i, &(inp, out, n)) in inputs.iter().enumerate() {
            let stream = if streams.is_empty() {
                std::ptr::null_mut()
            } else {
                streams[i % num_streams]
            };
            op(inp, out, n, stream);
        }

        Ok(())
    }

    /// Execute multiple independent binary operations in parallel
    pub fn binary_parallel<F>(
        &self,
        inputs: &[(*mut c_void, *mut c_void, *mut c_void, usize)],
        op: F,
    ) -> Result<()>
    where
        F: Fn(*mut c_void, *mut c_void, *mut c_void, usize, CudaStream) + Send + Sync + Copy,
    {
        let streams = self.executor.stream_pool.all_streams();
        let num_streams = streams.len().max(1);

        for (i, &(a, b, out, n)) in inputs.iter().enumerate() {
            let stream = if streams.is_empty() {
                std::ptr::null_mut()
            } else {
                streams[i % num_streams]
            };
            op(a, b, out, n, stream);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_signature() {
        let sig1 = OpSignature::new(0x10, 0, 1024);
        let sig2 = OpSignature::new(0x10, 0, 1024);
        assert_eq!(sig1, sig2);

        let sig3 = OpSignature::new(0x10, 0, 2048);
        assert_ne!(sig1, sig3);
    }

    #[test]
    fn test_batch_builder() {
        let builder = BatchBuilder::new();
        assert!(builder.is_empty());

        let builder = builder.add(|_stream| Ok(()));
        assert_eq!(builder.len(), 1);
    }
}
