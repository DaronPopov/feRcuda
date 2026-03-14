//! Safe abstractions over [crate::driver::result] provided by [CudaSlice], [CudaContext], [CudaStream], and more.

pub(crate) mod core;
pub(crate) mod external_memory;
pub(crate) mod graph;
pub(crate) mod launch;
pub(crate) mod profile;
pub(crate) mod stream_provider;
pub(crate) mod unified_memory;

#[cfg(feature = "ptx-os")]
pub(crate) mod ptx_integration;

pub use self::core::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
    DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, HostSlice, PinnedHostSlice, SyncOnDrop,
    ValidAsZeroBits,
};
pub use self::external_memory::{ExternalMemory, MappedBuffer};
pub use self::graph::CudaGraph;
pub use self::launch::{LaunchArgs, LaunchArgsTuple, LaunchAsync, LaunchConfig, PushKernelArg};
pub use self::profile::{profiler_start, profiler_stop, Profiler};
pub use self::stream_provider::{ExternalStreamPool, SingleStreamProvider, StreamProvider};
#[cfg(feature = "ptx-os")]
pub use self::stream_provider::PtxStreamPool;
#[cfg(feature = "ptx-os")]
pub use self::ptx_integration::{PtxStreamManager, PtxContextExt};
pub use self::unified_memory::{UnifiedSlice, UnifiedView, UnifiedViewMut};
pub use crate::driver::result::DriverError;
