use std::env;
use std::path::{Path, PathBuf};

/// Resolve the torch/libtorch root path (directory containing lib/ and include/).
pub fn resolve_torch_path() -> Option<PathBuf> {
    // 1. LIBTORCH env (explicit)
    if let Ok(p) = env::var("LIBTORCH") {
        let path = PathBuf::from(&p);
        if path.join("lib").exists() && (path.join("lib/libc10_cuda.so").exists()
            || path.join("lib/libtorch_cuda.so").exists()
            || path.join("lib/libc10.so").exists())
        {
            return Some(path);
        }
    }

    // 2. Pip-installed torch: VIRTUAL_ENV or CONDA_PREFIX
    for base in [
        env::var("VIRTUAL_ENV").ok(),
        env::var("CONDA_PREFIX").ok(),
    ]
    .into_iter()
    .flatten()
    {
        let torch = PathBuf::from(&base).join("lib/python3.12/site-packages/torch");
        if torch.join("lib/libc10_cuda.so").exists() {
            return Some(torch);
        }
        // Try python3.11, 3.10, etc.
        for v in ["3.11", "3.10", "3.9"] {
            let t = PathBuf::from(&base).join(format!("lib/python{v}/site-packages/torch"));
            if t.join("lib/libc10_cuda.so").exists() {
                return Some(t);
            }
        }
    }

    // 3. ~/.local
    if let Some(home) = env::var("HOME").ok() {
        for v in ["3.12", "3.11", "3.10"] {
            let t = PathBuf::from(&home)
                .join(format!(".local/lib/python{v}/site-packages/torch"));
            if t.join("lib/libc10_cuda.so").exists() {
                return Some(t);
            }
        }
    }

    None
}

/// Preload nvidia libs (nvjitlink, etc.) from pip torch to fix CUDA version mismatch.
/// Pip torch bundles CUDA 12.8 libs; system may have 12.6. Preloading ensures correct symbols.
fn preload_nvidia_libs(torch_path: &Path) {
    let site_packages = match torch_path.parent() {
        Some(p) => p,
        None => return,
    };
    let nvidia = site_packages.join("nvidia");
    if !nvidia.exists() {
        return;
    }

    // Order matters: nvjitlink must load before cusparse (which depends on it)
    let preload_order = ["nvjitlink/lib/libnvJitLink.so.12", "cuda_runtime/lib/libcudart.so.12"];
    for rel in preload_order {
        let full = nvidia.join(rel);
        if full.exists() {
            match crate::adapter::load_shared_lib_path(&full) {
                Ok(true) => eprintln!("[aten-ptx] preloaded {}", full.display()),
                Ok(false) => {}
                Err(e) => eprintln!("[aten-ptx] preload {}: {} (non-fatal)", full.display(), e),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InitPolicy {
    pub device_id: i32,
    pub pool_fraction: f64,
    pub num_streams: u32,
}

impl InitPolicy {
    pub const DEFAULT_NUM_STREAMS: u32 = 8;

    pub fn validate(self) -> Result<Self, String> {
        if self.device_id < 0 {
            return Err("Invalid device_id (must be >= 0)".to_string());
        }
        if !(0.1..=0.9).contains(&self.pool_fraction) {
            return Err("Invalid pool_fraction (must be 0.1-0.9)".to_string());
        }
        if self.num_streams == 0 {
            return Err("num_streams must be > 0".to_string());
        }
        Ok(self)
    }
}

pub fn ensure_libtorch_cuda_loaded() {
    let torch_path = resolve_torch_path();

    // Preload nvidia libs first to fix CUDA version mismatch (pip torch vs system)
    if let Some(ref p) = torch_path {
        preload_nvidia_libs(p);
    }

    let libs = ["libtorch_cpu.so", "libtorch.so", "libtorch_cuda.so"];

    for lib in libs {
        // Prefer loading from resolved path (fixes LD_LIBRARY_PATH / search order)
        let loaded = if let Some(ref torch) = torch_path {
            let path = torch.join("lib").join(lib);
            if path.exists() {
                crate::adapter::load_shared_lib_path(&path)
            } else {
                crate::adapter::maybe_load_shared_lib(lib)
            }
        } else {
            crate::adapter::maybe_load_shared_lib(lib)
        };

        match loaded {
            Ok(true) => eprintln!("[aten-ptx] loaded {}", lib),
            Ok(false) => {}
            Err(msg) => eprintln!("[aten-ptx] warning: failed to dlopen {}: {}", lib, msg),
        }
    }
}
