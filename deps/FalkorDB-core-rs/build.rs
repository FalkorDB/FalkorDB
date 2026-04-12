use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BINROOT");

    let bin_root = env::var("BINROOT");
    if let Ok(bin_root) = bin_root {
        let clang = env::var("CLANG").unwrap_or("0".to_string());

        println!("cargo:rustc-link-arg=-Wl,-rpath,{bin_root}");
        println!("cargo:rustc-link-arg={bin_root}/falkordb.so");
    }
}
