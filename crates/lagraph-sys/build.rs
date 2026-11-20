//! Link against the LAGraph archives that `native-deps` builds.
//!
//! `native-deps` builds LAGraph against the GraphBLAS archive of the same
//! request, so the two are always one consistent pair. This script only
//! emits the link directives for `liblagraphx.a` (the experimental
//! algorithms) and `liblagraph.a`; `graphblas-sys`, which this crate depends
//! on, links GraphBLAS and OpenMP after them.

use native_deps::{Dep, Request};

fn main() {
    let mut request = Request::from_env().unwrap_or_else(|e| panic!("native-deps: {e}"));
    // Only what this crate links, and GraphBLAS, which LAGraph is built
    // against. `from_env` asks for RediSearch as well.
    request.deps = vec![Dep::GraphBlas, Dep::LaGraph];

    for var in Request::env_inputs() {
        println!("cargo:rerun-if-env-changed={var}");
    }
    for path in native_deps::watch_paths(&request.root) {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let deps = native_deps::ensure(&request).unwrap_or_else(|e| panic!("native-deps: {e}"));
    let lagraph = deps.get(Dep::LaGraph).expect("lagraph resolved");
    // A key change moves the prefix, so watching the stamp is enough to make
    // cargo re-run this script whenever LAGraph is rebuilt.
    println!("cargo:rerun-if-changed={}", lagraph.stamp_path().display());
    if let Some(source) = &lagraph.local_source {
        println!("cargo:rerun-if-changed={}", source.display());
    }

    println!("cargo:rustc-link-search=native={}", lagraph.lib().display());
    // GNU ld resolves static archives left to right, so a dependent precedes
    // what it depends on: lagraphx -> lagraph -> graphblas (from
    // `graphblas-sys`, which rustc places after this crate's libraries).
    println!("cargo:rustc-link-lib=static=lagraphx");
    println!("cargo:rustc-link-lib=static=lagraph");
}
