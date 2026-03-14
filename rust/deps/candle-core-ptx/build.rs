use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if env::var_os("CARGO_FEATURE_PTX_OS").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let project_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("candle-core-ptx should live under <root>/rust/")
        .to_path_buf();
    let build_dir = project_root.join("build");

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir.display());
    println!(
        "cargo:rerun-if-changed={}",
        build_dir.join("libptx_os_shared.so").display()
    );
}
