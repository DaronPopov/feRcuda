use std::env;
use std::path::PathBuf;

fn main() {
    let lib_dir = env::var("FERCUDA_LIB_DIR").unwrap_or_else(|_| {
        let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home)
            .join(".local")
            .join("lib")
            .display()
            .to_string()
    });
    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=dylib=fercuda_capi");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);
    println!("cargo:rerun-if-env-changed=FERCUDA_LIB_DIR");
}
