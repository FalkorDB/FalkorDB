use std::fs;

// Suppress too_many_lines: this build script handles multiple platform/configuration
// cases and splitting it would reduce clarity.
#[allow(clippy::too_many_lines)]
fn main() {
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-search=/opt/homebrew/opt/llvm/lib");
        println!("cargo:rustc-link-search=/opt/homebrew/opt/llvm/lib/c++");
    }

    #[cfg(target_os = "linux")]
    {
        // libomp.a is built from llvm-project source by build/libomp.sh and
        // installed under /opt/libomp. Linking statically here makes
        // libfalkordb.so self-contained for OpenMP — no libomp.so.5 /
        // libgomp.so.1 dynamic dependency in the runtime image.
        //
        // For local Linux dev without /opt/libomp (e.g. apt's libomp-22-dev
        // which ships only the .so), fall back to dynamic linking. CI/Docker
        // is the source of truth for the fully self-contained .so.
        println!("cargo:rustc-link-search=/opt/libomp/lib");
        println!("cargo:rustc-link-search=/usr/lib/llvm-22/lib");
        println!("cargo:rustc-link-search=/usr/lib/llvm-21/lib");
        println!("cargo:rustc-link-search=/usr/lib/llvm-20/lib");

        if std::path::Path::new("/opt/libomp/lib/libomp.a").exists() {
            println!("cargo:rustc-link-lib=static=omp");
        } else {
            println!("cargo:rustc-link-lib=omp");
        }
    }

    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=omp");

    println!("cargo:rustc-link-search=/usr/local/lib");
    println!("cargo:rustc-link-lib=static=graphblas");

    // LAGraph static libraries
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let lagraph_dir = std::path::Path::new(&manifest_dir).join("../lagraph_lib");
    println!("cargo:rustc-link-search=native={}", lagraph_dir.display());
    println!("cargo:rustc-link-search=native=/data/lagraph_lib");
    println!("cargo:rustc-link-lib=static=lagraph");
    println!("cargo:rustc-link-lib=static=lagraphx");

    // VecSim/RediSearch are built with a C++ toolchain.
    // - macOS uses libc++ / libc++abi
    // - Linux generally uses libstdc++ (and does not need explicit c++abi)
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=static=c++");
        println!("cargo:rustc-link-lib=static=c++abi");
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=stdc++");
    }

    #[cfg(target_os = "macos")]
    {
        println!(
            "cargo:rustc-link-search=native=redisearch/RediSearch/bin/macos-arm64v8-release/search-static"
        );
        println!(
            "cargo:rustc-link-search=native=redisearch/RediSearch/bin/macos-arm64v8-release/search-static/deps/VectorSimilarity/src/VecSim"
        );
        println!(
            "cargo:rustc-link-search=native=redisearch/RediSearch/bin/macos-arm64v8-release/search-static/deps/VectorSimilarity/src/VecSim/spaces"
        );
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        println!(
            "cargo:rustc-link-search=native=redisearch/RediSearch/bin/linux-x64-release/search-static"
        );
        println!(
            "cargo:rustc-link-search=native=redisearch/RediSearch/bin/linux-x64-release/search-static/deps/VectorSimilarity/src/VecSim"
        );
        println!(
            "cargo:rustc-link-search=native=redisearch/RediSearch/bin/linux-x64-release/search-static/deps/VectorSimilarity/src/VecSim/spaces"
        );
        println!(
            "cargo:rustc-link-search=native=/data/redisearch/bin/linux-x64-release/search-static"
        );
        println!(
            "cargo:rustc-link-search=native=/data/redisearch/bin/linux-x64-release/search-static/deps/VectorSimilarity/src/VecSim"
        );
        println!(
            "cargo:rustc-link-search=native=/data/redisearch/bin/linux-x64-release/search-static/deps/VectorSimilarity/src/VecSim/spaces"
        );
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        println!(
            "cargo:rustc-link-search=native=redisearch/RediSearch/bin/linux-arm64v8-release/search-static"
        );
        println!(
            "cargo:rustc-link-search=native=redisearch/RediSearch/bin/linux-arm64v8-release/search-static/deps/VectorSimilarity/src/VecSim"
        );
        println!(
            "cargo:rustc-link-search=native=redisearch/RediSearch/bin/linux-arm64v8-release/search-static/deps/VectorSimilarity/src/VecSim/spaces"
        );
        println!(
            "cargo:rustc-link-search=native=/data/redisearch/bin/linux-arm64v8-release/search-static"
        );
        println!(
            "cargo:rustc-link-search=native=/data/redisearch/bin/linux-arm64v8-release/search-static/deps/VectorSimilarity/src/VecSim"
        );
        println!(
            "cargo:rustc-link-search=native=/data/redisearch/bin/linux-arm64v8-release/search-static/deps/VectorSimilarity/src/VecSim/spaces"
        );
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    let paths = fs::read_dir("../redisearch/RediSearch/bin/linux-x64-release/search-static/deps/VectorSimilarity/src/VecSim/spaces").unwrap_or_else(|_| {
        fs::read_dir("/data/redisearch/bin/linux-x64-release/search-static/deps/VectorSimilarity/src/VecSim/spaces").unwrap()
    });

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    let paths = fs::read_dir("../redisearch/RediSearch/bin/linux-arm64v8-release/search-static/deps/VectorSimilarity/src/VecSim/spaces").unwrap_or_else(|_| {
        fs::read_dir("/data/redisearch/bin/linux-arm64v8-release/search-static/deps/VectorSimilarity/src/VecSim/spaces").unwrap()
    });

    #[cfg(target_os = "macos")]
    let paths = fs::read_dir("../redisearch/RediSearch/bin/macos-arm64v8-release/search-static/deps/VectorSimilarity/src/VecSim/spaces").unwrap();

    for path in paths.flatten() {
        let path = path.path();
        let name = &path.file_name().unwrap().to_str().unwrap();
        let name = &name[3..&name.len() - 2];
        let extention = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if extention.eq_ignore_ascii_case("a") {
            println!("cargo:rustc-link-lib=static={name}");
        }
    }
    println!("cargo:rustc-link-lib=static=VectorSimilarity");
    println!("cargo:rustc-link-lib=static=redisearch-static");
}
