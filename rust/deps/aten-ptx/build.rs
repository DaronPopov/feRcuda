// Build script to compile C++ PyTorch allocator
//
// Resolution order for libtorch:
//   1. LIBTORCH env var (set by scripts/install.sh / ferrite-run)
//   2. ../libtorch  (external/libtorch relative to external/aten-ptx)
//   3. ../../external/libtorch  (repo root)
//
// No Python fallback — libtorch is provisioned by scripts/install.sh as a standalone
// C++ distribution. This avoids version mismatches between pip-installed
// PyTorch and the headers we compile against.

use std::env;
use std::path::{Path, PathBuf};

fn looks_like_libtorch(root: &Path) -> bool {
    let has_include = root.join("include/torch/torch.h").exists()
        || root.join("include/torch/csrc/api/include/torch/torch.h").exists();
    let has_lib = root.join("lib/libc10_cuda.so").exists()
        || root.join("lib/libtorch_cuda.so").exists()
        || root.join("lib/libc10.so").exists();
    has_include && has_lib
}

fn resolve_libtorch(manifest_dir: &Path) -> PathBuf {
    // 1. LIBTORCH env var (highest priority — set by scripts/install.sh and ferrite-run)
    if let Ok(p) = env::var("LIBTORCH") {
        let path = PathBuf::from(&p);
        if looks_like_libtorch(&path) {
            return path;
        }
        eprintln!(
            "cargo:warning=LIBTORCH={} does not look like a valid libtorch directory, trying other paths",
            p
        );
    }

    // 2. Local paths relative to this crate
    let mut candidates = Vec::new();
    if let Some(deps_dir) = manifest_dir.parent() {
        candidates.push(deps_dir.join("libtorch"));
        if let Some(rust_dir) = deps_dir.parent() {
            candidates.push(rust_dir.join("external/libtorch"));
            if let Some(repo_root) = rust_dir.parent() {
                candidates.push(repo_root.join("external/libtorch"));
            }
        }
    }

    for c in &candidates {
        if looks_like_libtorch(c) {
            return c.clone();
        }
    }

    // 3. Pip-installed torch: run python to get path
    for python in ["python3", "python"] {
        if let Ok(out) = std::process::Command::new(python)
            .args(["-c", "import torch; print(torch.__path__[0])"])
            .output()
        {
            if out.status.success() {
                let path = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
                if looks_like_libtorch(&path) {
                    eprintln!("cargo:warning=aten-ptx: auto-detected libtorch from pip: {}", path.display());
                    return path;
                }
            }
        }
    }

    panic!(
        "Could not find libtorch.\n\
         Run ./scripts/install.sh to provision it, or set LIBTORCH=/path/to/libtorch.\n\
         Searched:\n  - LIBTORCH env var\n{}",
        candidates
            .iter()
            .map(|c| format!("  - {}", c.display()))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn main() {
    println!("cargo:rerun-if-changed=tlsf_allocator_full.cpp");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=CUDA_INCLUDE");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let libtorch = resolve_libtorch(&manifest_dir);
    let required_hdr = libtorch.join("include/c10/cuda/CUDACachingAllocator.h");
    assert!(
        required_hdr.exists(),
        "libtorch at {} is missing c10/cuda/CUDACachingAllocator.h (need CUDA-enabled libtorch)",
        libtorch.display()
    );

    eprintln!("cargo:warning=aten-ptx: using libtorch at {}", libtorch.display());

    // Find CUDA
    let cuda_include = env::var("CUDA_INCLUDE").unwrap_or_else(|_| "/usr/local/cuda/include".to_string());

    // Compile C++ allocator
    // Use -isystem for libtorch/CUDA headers to suppress upstream warnings
    cc::Build::new()
        .cpp(true)
        .file("tlsf_allocator_full.cpp")
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-Wno-unused-parameter")
        .flag(&format!("-isystem{}/include", libtorch.display()))
        .flag(&format!(
            "-isystem{}/include/torch/csrc/api/include",
            libtorch.display()
        ))
        .flag(&format!("-isystem{}", cuda_include))
        .compile("aten_tlsf_allocator");

    println!("cargo:rustc-link-lib=static=aten_tlsf_allocator");
    println!("cargo:rustc-link-lib=stdc++");

    // Link c10_cuda which contains the allocator symbol
    println!("cargo:rustc-link-search=native={}", libtorch.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=c10_cuda");
}
