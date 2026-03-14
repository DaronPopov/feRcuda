use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // Walk up to the feRcuda root: rust/deps/ptx-os -> rust/deps -> rust -> feRcuda
    let fercuda_root = PathBuf::from(&manifest_dir)
        .parent().unwrap()    // rust/deps
        .parent().unwrap()    // rust
        .parent().unwrap()    // feRcuda
        .to_path_buf();
    let build_dir = fercuda_root.join("build");

    // Prefer the in-tree CMake-built static libraries.
    // If FERCUDA_BUILD_DIR is set, use that instead.
    let lib_dir = env::var("FERCUDA_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or(build_dir);

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Link the native PTX-OS core and kernels (shared — device link handled by CMake)
    println!("cargo:rustc-link-lib=dylib=ptx_core");
    println!("cargo:rustc-link-lib=dylib=ptx_kernels");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // CUDA runtime + device runtime (needed for separable compilation / -rdc=true)
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/x86_64-linux/lib");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=static=cudadevrt");

    // C++ standard library (needed for the .cu files compiled as C++)
    println!("cargo:rustc-link-lib=stdc++");

    // pthread and rt for shared memory / IPC
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=rt");

    // Rerun if the native libraries change
    println!("cargo:rerun-if-changed={}/libptx_core.a", lib_dir.display());
    println!("cargo:rerun-if-changed={}/libptx_kernels.a", lib_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
}
