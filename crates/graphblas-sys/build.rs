//! Link against the GraphBLAS archive that `native-deps` builds.
//!
//! `native-deps` owns *producing* GraphBLAS (the submodule pin, the GB_control
//! patch, the PreJIT kernels, the cache key). This script only turns the
//! resolved prefix into `cargo:rustc-link-*` directives. LAGraph and
//! RediSearch are linked by their own crates / by `graph`.

use std::path::Path;

use native_deps::{Dep, Request, Resolved};

fn main() {
    let mut request = Request::from_env().unwrap_or_else(|e| panic!("native-deps: {e}"));
    // Only what this crate links. `from_env` asks for every dependency, and
    // RediSearch is neither needed here nor quick to build.
    request.deps = vec![Dep::GraphBlas];

    for var in Request::env_inputs() {
        println!("cargo:rerun-if-env-changed={var}");
    }
    for path in native_deps::watch_paths(&request.root) {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let deps = native_deps::ensure(&request).unwrap_or_else(|e| panic!("native-deps: {e}"));
    let graphblas = deps.get(Dep::GraphBlas).expect("graphblas resolved");
    // A key change moves the prefix, so watching the stamp is enough to make
    // cargo re-run this script whenever GraphBLAS is rebuilt.
    println!(
        "cargo:rerun-if-changed={}",
        graphblas.stamp_path().display()
    );
    if let Some(source) = &graphblas.local_source {
        println!("cargo:rerun-if-changed={}", source.display());
    }

    link_openmp_search_paths();
    link_graphblas(graphblas);
}

/// Search paths for libomp and the LLVM runtime, emitted before any `-l` so the
/// linker can resolve whichever OpenMP flavour `native-deps` built against.
fn link_openmp_search_paths() {
    println!("cargo:rustc-link-search={}/lib", libomp_prefix());

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-search=/opt/homebrew/opt/llvm/lib");
        println!("cargo:rustc-link-search=/opt/homebrew/opt/llvm/lib/c++");
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-search=/usr/lib/llvm-23/lib");
        println!("cargo:rustc-link-search=/usr/lib/llvm-22/lib");
        println!("cargo:rustc-link-search=/usr/lib/llvm-21/lib");
        println!("cargo:rustc-link-search=/usr/lib/llvm-20/lib");
    }
}

fn libomp_prefix() -> String {
    std::env::var("LIBOMP_PREFIX").unwrap_or_else(|_| "/opt/libomp".to_string())
}

fn link_graphblas(graphblas: &Resolved) {
    println!(
        "cargo:rustc-link-search=native={}",
        graphblas.lib().display()
    );
    println!("cargo:rustc-link-lib=static=graphblas");

    // omp must come AFTER graphblas: GNU ld resolves static archives
    // left-to-right, and the archive references __kmpc_* symbols. Static when
    // `${LIBOMP_PREFIX:-/opt/libomp}/lib/libomp.a` exists (the CI/Docker path),
    // otherwise dynamic `-lomp` from apt's libomp-NN-dev or Homebrew's libomp.
    if Path::new(&libomp_prefix()).join("lib/libomp.a").exists() {
        println!("cargo:rustc-link-lib=static=omp");
    } else {
        println!("cargo:rustc-link-lib=omp");
    }
}
