use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BINROOT");

    let bin_root = env::var("BINROOT");
    if let Ok(bin_root) = bin_root {
        let graphblas_dir = env::var("GRAPHBLAS_BINDIR").unwrap();
        let clang = env::var("CLANG").unwrap_or("0".to_string());

        println!("cargo:rustc-link-arg=-Wl,-rpath,{bin_root}/src");
        println!("cargo:rustc-link-arg={bin_root}/src/falkordb.so");
        println!("cargo:rustc-link-arg=-L{graphblas_dir}");
        println!("cargo:rustc-link-arg=-lgraphblas");
        if clang == "1" {
            println!("cargo:rustc-link-arg=-L/opt/homebrew/opt/libomp/lib");
            println!("cargo:rustc-link-arg=-L/usr/lib/llvm-17/lib");
            println!("cargo:rustc-link-arg=-L/usr/lib/llvm-18/lib/");
            println!("cargo:rustc-link-arg=-lomp");
        } else {
            println!("cargo:rustc-link-arg=-lgomp");
        }
    }
}
