//! Virtual Memory Manager for GPU memory paging

use crate::error::{PtxError, Result};
use crate::ffi;
use crate::runtime::RegimeRuntimeCore;
use std::ptr::NonNull;

/// Page flags for VMM
#[derive(Debug, Clone, Copy)]
pub struct PageFlags(u32);

impl PageFlags {
    pub const READ: Self = Self(ffi::VMM_FLAG_READ);
    pub const WRITE: Self = Self(ffi::VMM_FLAG_WRITE);
    pub const EXEC: Self = Self(ffi::VMM_FLAG_EXEC);
    pub const SHARED: Self = Self(ffi::VMM_FLAG_SHARED);
    pub const PINNED: Self = Self(ffi::VMM_FLAG_PINNED);

    pub fn new() -> Self {
        Self(0)
    }

    pub fn read(self) -> Self {
        Self(self.0 | ffi::VMM_FLAG_READ)
    }

    pub fn write(self) -> Self {
        Self(self.0 | ffi::VMM_FLAG_WRITE)
    }

    pub fn readwrite() -> Self {
        Self(ffi::VMM_FLAG_READ | ffi::VMM_FLAG_WRITE)
    }

    pub fn pinned(self) -> Self {
        Self(self.0 | ffi::VMM_FLAG_PINNED)
    }
}

impl Default for PageFlags {
    fn default() -> Self {
        Self::readwrite()
    }
}

/// VMM statistics
#[derive(Debug, Clone, Default)]
pub struct VmmStats {
    pub resident_pages: u64,
    pub swapped_pages: u64,
    pub page_faults: u64,
    pub evictions: u64,
}

/// A page of GPU memory managed by the VMM.
pub struct Page {
    ptr: NonNull<std::ffi::c_void>,
    vmm: VirtualMemory,
}

impl Page {
    /// Get raw pointer to page data.
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.ptr.as_ptr()
    }

    /// Get mutable raw pointer to page data.
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.ptr.as_ptr()
    }

    /// Swap this page out to host memory.
    pub fn swap_out(&mut self) -> Result<()> {
        let guard = self.vmm.inner.lock();
        let result = unsafe { ffi::vmm_swap_out(guard.ptr.as_ptr(), self.ptr.as_ptr()) };

        if result < 0 {
            Err(PtxError::VmmError {
                message: "Swap out failed".into(),
            })
        } else {
            Ok(())
        }
    }

    /// Swap this page back into GPU memory.
    pub fn swap_in(&mut self) -> Result<()> {
        let guard = self.vmm.inner.lock();
        let result = unsafe { ffi::vmm_swap_in(guard.ptr.as_ptr(), self.ptr.as_ptr()) };

        if result < 0 {
            Err(PtxError::VmmError {
                message: "Swap in failed".into(),
            })
        } else {
            Ok(())
        }
    }

    /// Pin this page (prevent eviction).
    pub fn pin(&mut self) {
        let guard = self.vmm.inner.lock();
        unsafe {
            ffi::vmm_pin_page(guard.ptr.as_ptr(), self.ptr.as_ptr());
        }
    }

    /// Unpin this page (allow eviction).
    pub fn unpin(&mut self) {
        let guard = self.vmm.inner.lock();
        unsafe {
            ffi::vmm_unpin_page(guard.ptr.as_ptr(), self.ptr.as_ptr());
        }
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        let guard = self.vmm.inner.lock();
        unsafe {
            ffi::vmm_free_page(guard.ptr.as_ptr(), self.ptr.as_ptr());
        }
    }
}

/// Inner VMM state
struct VmmInner {
    ptr: NonNull<ffi::VMMState>,
}

unsafe impl Send for VmmInner {}
unsafe impl Sync for VmmInner {}

impl Drop for VmmInner {
    fn drop(&mut self) {
        unsafe {
            ffi::vmm_shutdown(self.ptr.as_ptr());
        }
    }
}

/// Virtual Memory Manager for GPU memory paging
#[derive(Clone)]
pub struct VirtualMemory {
    inner: std::sync::Arc<parking_lot::Mutex<VmmInner>>,
    _runtime: RegimeRuntimeCore,
}

impl VirtualMemory {
    /// Initialize the virtual memory manager.
    ///
    /// # Arguments
    /// * `runtime` - The GPU runtime
    /// * `swap_size` - Size of host swap space in bytes
    pub fn new(runtime: &RegimeRuntimeCore, swap_size: usize) -> Result<Self> {
        let ptr = unsafe { ffi::vmm_init(runtime.as_ptr(), swap_size) };

        let ptr = NonNull::new(ptr).ok_or(PtxError::VmmError {
            message: "Failed to initialize VMM".into(),
        })?;

        Ok(Self {
            inner: std::sync::Arc::new(parking_lot::Mutex::new(VmmInner { ptr })),
            _runtime: runtime.clone(),
        })
    }

    /// Allocate a new page.
    pub fn alloc_page(&self, flags: PageFlags) -> Result<Page> {
        let guard = self.inner.lock();
        let ptr = unsafe { ffi::vmm_alloc_page(guard.ptr.as_ptr(), flags.0) };

        let ptr = NonNull::new(ptr).ok_or(PtxError::VmmError {
            message: "Page allocation failed".into(),
        })?;

        Ok(Page {
            ptr,
            vmm: self.clone(),
        })
    }

    /// Get VMM statistics.
    pub fn stats(&self) -> VmmStats {
        let guard = self.inner.lock();
        let mut resident = 0u64;
        let mut swapped = 0u64;
        let mut faults = 0u64;
        let mut evictions = 0u64;

        unsafe {
            ffi::vmm_get_stats(
                guard.ptr.as_ptr(),
                &mut resident,
                &mut swapped,
                &mut faults,
                &mut evictions,
            );
        }

        VmmStats {
            resident_pages: resident,
            swapped_pages: swapped,
            page_faults: faults,
            evictions,
        }
    }
}

impl std::fmt::Debug for VirtualMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("VirtualMemory")
            .field("resident_pages", &stats.resident_pages)
            .field("swapped_pages", &stats.swapped_pages)
            .finish()
    }
}
