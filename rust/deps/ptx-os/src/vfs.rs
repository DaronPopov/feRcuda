//! Virtual Filesystem for GPU-resident data

use crate::error::{PtxError, Result};
use crate::ffi;
use crate::runtime::RegimeRuntimeCore;
use crate::tensor::DType;
use std::ffi::CString;
use std::ptr::NonNull;

/// Open flags for VFS files
#[derive(Debug, Clone, Copy)]
pub struct OpenFlags(u32);

impl OpenFlags {
    pub const READ: Self = Self(ffi::VFS_O_RDONLY);
    pub const WRITE: Self = Self(ffi::VFS_O_WRONLY);
    pub const READWRITE: Self = Self(ffi::VFS_O_RDWR);
    pub const CREATE: Self = Self(ffi::VFS_O_CREAT);
    pub const TRUNCATE: Self = Self(ffi::VFS_O_TRUNC);
    pub const APPEND: Self = Self(ffi::VFS_O_APPEND);

    pub fn new() -> Self {
        Self(0)
    }

    pub fn read(self) -> Self {
        Self(self.0 | ffi::VFS_O_RDONLY)
    }

    pub fn write(self) -> Self {
        Self(self.0 | ffi::VFS_O_WRONLY)
    }

    pub fn create(self) -> Self {
        Self(self.0 | ffi::VFS_O_CREAT)
    }

    pub fn truncate(self) -> Self {
        Self(self.0 | ffi::VFS_O_TRUNC)
    }
}

impl Default for OpenFlags {
    fn default() -> Self {
        Self::new()
    }
}

/// File handle for VFS operations
pub struct File {
    fd: i32,
    vfs: VirtualFs,
}

impl File {
    /// Read from file into buffer.
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let guard = self.vfs.inner.lock();
        let bytes_read = unsafe {
            ffi::vfs_read(
                guard.ptr.as_ptr(),
                self.fd,
                buf.as_mut_ptr() as *mut _,
                buf.len(),
            )
        };

        if bytes_read < 0 {
            Err(PtxError::VfsError {
                message: "Read failed".into(),
            })
        } else {
            Ok(bytes_read as usize)
        }
    }

    /// Write buffer to file.
    pub fn write(&mut self, buf: &[u8]) -> Result<usize> {
        let guard = self.vfs.inner.lock();
        let bytes_written = unsafe {
            ffi::vfs_write(
                guard.ptr.as_ptr(),
                self.fd,
                buf.as_ptr() as *const _,
                buf.len(),
            )
        };

        if bytes_written < 0 {
            Err(PtxError::VfsError {
                message: "Write failed".into(),
            })
        } else {
            Ok(bytes_written as usize)
        }
    }

    /// Seek to position.
    pub fn seek(&mut self, offset: usize, whence: i32) -> Result<usize> {
        let guard = self.vfs.inner.lock();
        let pos = unsafe { ffi::vfs_seek(guard.ptr.as_ptr(), self.fd, offset, whence) };

        if pos < 0 {
            Err(PtxError::VfsError {
                message: "Seek failed".into(),
            })
        } else {
            Ok(pos as usize)
        }
    }
}

impl Drop for File {
    fn drop(&mut self) {
        let guard = self.vfs.inner.lock();
        unsafe {
            ffi::vfs_close(guard.ptr.as_ptr(), self.fd);
        }
    }
}

/// Inner VFS state
struct VfsInner {
    ptr: NonNull<ffi::VFSState>,
}

unsafe impl Send for VfsInner {}
unsafe impl Sync for VfsInner {}

impl Drop for VfsInner {
    fn drop(&mut self) {
        unsafe {
            ffi::vfs_shutdown(self.ptr.as_ptr());
        }
    }
}

/// Virtual Filesystem for GPU-resident data
#[derive(Clone)]
pub struct VirtualFs {
    inner: std::sync::Arc<parking_lot::Mutex<VfsInner>>,
    _runtime: RegimeRuntimeCore,
}

impl VirtualFs {
    /// Initialize the virtual filesystem.
    pub fn new(runtime: &RegimeRuntimeCore) -> Result<Self> {
        let ptr = unsafe { ffi::vfs_init(runtime.as_ptr()) };

        let ptr = NonNull::new(ptr).ok_or(PtxError::VfsError {
            message: "Failed to initialize VFS".into(),
        })?;

        Ok(Self {
            inner: std::sync::Arc::new(parking_lot::Mutex::new(VfsInner { ptr })),
            _runtime: runtime.clone(),
        })
    }

    /// Create a directory.
    pub fn mkdir(&self, path: &str, mode: u32) -> Result<()> {
        let guard = self.inner.lock();
        let c_path = CString::new(path).map_err(|_| PtxError::InvalidArgument {
            message: "Invalid path".into(),
        })?;

        let result = unsafe { ffi::vfs_mkdir(guard.ptr.as_ptr(), c_path.as_ptr(), mode) };

        if result < 0 {
            Err(PtxError::VfsError {
                message: format!("Failed to create directory: {}", path),
            })
        } else {
            Ok(())
        }
    }

    /// Remove a directory.
    pub fn rmdir(&self, path: &str) -> Result<()> {
        let guard = self.inner.lock();
        let c_path = CString::new(path).map_err(|_| PtxError::InvalidArgument {
            message: "Invalid path".into(),
        })?;

        let result = unsafe { ffi::vfs_rmdir(guard.ptr.as_ptr(), c_path.as_ptr()) };

        if result < 0 {
            Err(PtxError::VfsError {
                message: format!("Failed to remove directory: {}", path),
            })
        } else {
            Ok(())
        }
    }

    /// Open a file.
    pub fn open(&self, path: &str, flags: OpenFlags) -> Result<File> {
        let guard = self.inner.lock();
        let c_path = CString::new(path).map_err(|_| PtxError::InvalidArgument {
            message: "Invalid path".into(),
        })?;

        let fd = unsafe { ffi::vfs_open(guard.ptr.as_ptr(), c_path.as_ptr(), flags.0) };

        if fd < 0 {
            Err(PtxError::VfsError {
                message: format!("Failed to open file: {}", path),
            })
        } else {
            Ok(File {
                fd,
                vfs: self.clone(),
            })
        }
    }

    /// Create a GPU tensor at the given path.
    pub fn create_tensor(
        &self,
        path: &str,
        shape: &[i32],
        dtype: DType,
    ) -> Result<()> {
        let guard = self.inner.lock();
        let c_path = CString::new(path).map_err(|_| PtxError::InvalidArgument {
            message: "Invalid path".into(),
        })?;

        let result = unsafe {
            ffi::vfs_create_tensor(
                guard.ptr.as_ptr(),
                c_path.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
                dtype as i32,
            )
        };

        if result < 0 {
            Err(PtxError::VfsError {
                message: format!("Failed to create tensor: {}", path),
            })
        } else {
            Ok(())
        }
    }

    /// Memory-map a tensor (get raw GPU pointer).
    pub fn mmap_tensor(&self, path: &str) -> Result<*mut std::ffi::c_void> {
        let guard = self.inner.lock();
        let c_path = CString::new(path).map_err(|_| PtxError::InvalidArgument {
            message: "Invalid path".into(),
        })?;

        let ptr = unsafe { ffi::vfs_mmap_tensor(guard.ptr.as_ptr(), c_path.as_ptr()) };

        if ptr.is_null() {
            Err(PtxError::NotFound {
                name: path.into(),
            })
        } else {
            Ok(ptr)
        }
    }

    /// Sync tensor to ensure GPU writes are complete.
    pub fn sync_tensor(&self, path: &str) -> Result<()> {
        let guard = self.inner.lock();
        let c_path = CString::new(path).map_err(|_| PtxError::InvalidArgument {
            message: "Invalid path".into(),
        })?;

        let result = unsafe { ffi::vfs_sync_tensor(guard.ptr.as_ptr(), c_path.as_ptr()) };

        if result < 0 {
            Err(PtxError::VfsError {
                message: format!("Failed to sync tensor: {}", path),
            })
        } else {
            Ok(())
        }
    }
}

impl std::fmt::Debug for VirtualFs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtualFs").finish()
    }
}
