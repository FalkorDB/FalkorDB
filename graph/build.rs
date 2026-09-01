//! Link FalkorDB against its native dependencies.
//!
//! Everything about *producing* GraphBLAS, LAGraph and RediSearch lives in the
//! `native-deps` crate (a path build-dependency, so this is a plain function
//! call -- no nested cargo, nothing to inherit `RUSTFLAGS` from). This script's
//! job is narrower: turn the resolved prefixes into `cargo:rustc-link-*`
//! directives, in the right order, plus two fix-ups that only make sense at
//! link time.

use std::fs;
use std::path::{Path, PathBuf};

use native_deps::{Dep, Request, Resolved};

fn main() {
    let request = Request::from_env().unwrap_or_else(|e| panic!("native-deps: {e}"));

    for var in Request::env_inputs() {
        println!("cargo:rerun-if-env-changed={var}");
    }
    for path in native_deps::watch_paths(&request.root) {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let deps = native_deps::ensure(&request).unwrap_or_else(|e| panic!("native-deps: {e}"));
    for (_, resolved) in deps.iter() {
        // A key change moves the prefix, so watching the stamp is enough to make
        // cargo re-run this script whenever a dependency is rebuilt.
        println!("cargo:rerun-if-changed={}", resolved.stamp_path().display());
    }

    let graphblas = deps.get(Dep::GraphBlas).expect("graphblas resolved");
    let lagraph = deps.get(Dep::LaGraph).expect("lagraph resolved");
    let redisearch = deps.get(Dep::RediSearch).expect("redisearch resolved");

    link_openmp_search_paths();
    link_graphblas(graphblas, lagraph);
    link_cxx_and_openssl();
    link_redisearch(redisearch);
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
        println!("cargo:rustc-link-search=/usr/lib/llvm-22/lib");
        println!("cargo:rustc-link-search=/usr/lib/llvm-21/lib");
        println!("cargo:rustc-link-search=/usr/lib/llvm-20/lib");
    }
}

fn libomp_prefix() -> String {
    std::env::var("LIBOMP_PREFIX").unwrap_or_else(|_| "/opt/libomp".to_string())
}

fn link_graphblas(
    graphblas: &Resolved,
    lagraph: &Resolved,
) {
    println!(
        "cargo:rustc-link-search=native={}",
        graphblas.lib().display()
    );
    println!("cargo:rustc-link-search=native={}", lagraph.lib().display());

    // Link order is load-bearing: GNU ld resolves static archives left-to-right,
    // so dependents must precede dependencies -- lagraphx -> lagraph -> graphblas.
    println!("cargo:rustc-link-lib=static=lagraphx");
    println!("cargo:rustc-link-lib=static=lagraph");
    println!("cargo:rustc-link-lib=static=graphblas");

    // omp must come AFTER graphblas/lagraph: GNU ld resolves static archives
    // left-to-right, and those archives reference __kmpc_* symbols. The wrong
    // order goes unnoticed for the cdylib (shared links tolerate undefined
    // symbols) but breaks test executables with static libomp.
    //
    // Static when `${LIBOMP_PREFIX:-/opt/libomp}/lib/libomp.a` exists -- the path
    // CI/Docker takes (build/libomp.sh installs there), which makes
    // libfalkordb.{so,dylib} self-contained for OpenMP. Otherwise dynamic
    // `-lomp`, resolved against apt's libomp-NN-dev or Homebrew's libomp.
    if Path::new(&libomp_prefix()).join("lib/libomp.a").exists() {
        println!("cargo:rustc-link-lib=static=omp");
    } else {
        println!("cargo:rustc-link-lib=omp");
    }
}

fn link_cxx_and_openssl() {
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

    // OpenSSL: RediSearch's coordinator and hiredis_ssl reference libssl/libcrypto.
    // (In 8.6 the coordinator objects are always compiled into the static archive.)
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-search=native=/opt/homebrew/opt/openssl@3/lib");
        println!("cargo:rustc-link-lib=static=ssl");
        println!("cargo:rustc-link-lib=static=crypto");
    }
    #[cfg(target_os = "linux")]
    {
        // Dynamic on Linux: the build image's libssl-dev provides the link-time
        // .so and the runtime image ships libssl3. Static would need
        // libssl.a/libcrypto.a, which Debian's libssl-dev doesn't package.
        println!("cargo:rustc-link-lib=ssl");
        println!("cargo:rustc-link-lib=crypto");
    }
}

/// RediSearch 8.6, embedded as a static library.
///
/// `native-deps` publishes it under stable names, so there is no variant-dir
/// scan and no flavour guessing here any more: the sanitizer flavour is a
/// cache-key input, which means the prefix we were handed is already the right
/// one.
fn link_redisearch(redisearch: &Resolved) {
    // Link the main archive FIRST, then the dependency archives. GNU ld resolves
    // static archives left-to-right and won't rescan earlier ones, so the main
    // `libredisearch.a` -- which references symbols defined in its VecSim / C
    // deps -- must precede them. (Redundant archives are otherwise harmless: the
    // linker only pulls members that resolve an undefined symbol.)
    let main = redisearch.lib().join("libredisearch.a");
    assert!(
        main.is_file(),
        "{} missing - run `native-deps ensure --force --dep redisearch`",
        main.display()
    );
    link_static(&main);
    for archive in find_archives(&redisearch.lib().join("deps")) {
        link_static(&archive);
    }

    // libredisearch_rs.a embeds its own copy of the `redis-module` crate (pulled
    // by rlookup / redis_mock). That crate unconditionally defines linkme
    // `#[distributed_slice]`s (COMMANDS_LIST, configs, info, ...). FalkorDB links
    // its own `redis-module`, so without intervention the final cdylib carries two
    // definitions of each slice and linkme aborts at module load with
    // `duplicate #[distributed_slice] with name "COMMANDS_LIST"`. In RediSearch's
    // library build those slices are empty and never iterated, so we strip the
    // linkme sections from a private copy of the archive and link that instead.
    let patched = strip_linkme_sections(&redisearch.redisearch_rs_archive());
    link_static(&patched);
}

/// RediSearch's Rust archive carries a second copy of the `redis-module` crate
/// whose linkme `#[distributed_slice]` markers collide with FalkorDB's at load
/// time. Produce a copy in `OUT_DIR` with every linkme section removed and return
/// its path. The slices are empty in RediSearch's library build, so dropping them
/// only removes dead bookkeeping.
fn strip_linkme_sections(archive: &Path) -> PathBuf {
    assert!(
        archive.is_file(),
        "{} missing - run `native-deps ensure --force --dep redisearch`",
        archive.display()
    );
    println!("cargo:rerun-if-changed={}", archive.display());

    let objdump = llvm_tool("LLVM_OBJDUMP", "llvm-objdump");
    let objcopy = llvm_tool("LLVM_OBJCOPY", "llvm-objcopy");

    // Enumerate linkme section names from the archive's section headers. The names
    // are content-hashed (e.g. `__linkm2JsytAsd7`), so discover them rather than
    // hard-coding; this also keeps working if linkme is bumped.
    let headers = std::process::Command::new(&objdump)
        .arg("--section-headers")
        .arg(archive)
        .output()
        .expect("failed to run llvm-objdump on libredisearch_rs.a");
    assert!(
        headers.status.success(),
        "llvm-objdump --section-headers failed on {}",
        archive.display()
    );
    let stdout = String::from_utf8_lossy(&headers.stdout);
    let mut sections: Vec<&str> = stdout
        .lines()
        .filter_map(|line| line.split_whitespace().find(|tok| tok.contains("linkm")))
        .collect();
    sections.sort_unstable();
    sections.dedup();

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let patched = Path::new(&out_dir).join("libredisearch_rs.a");
    fs::copy(archive, &patched).expect("failed to copy libredisearch_rs.a into OUT_DIR");

    if sections.is_empty() {
        return patched;
    }

    let mut cmd = std::process::Command::new(&objcopy);
    // A sanitizer/debug build of redisearch_rs carries DWARF whose `.debug_info`
    // holds relocations INTO the linkme sections, so objcopy refuses to remove
    // them ("section ... cannot be removed: ... has relocation against symbol").
    // Strip debug info first (ELF only). We don't need redisearch_rs's line
    // tables; VecSim's debug info lives in a separate archive and is preserved.
    #[cfg(not(target_os = "macos"))]
    cmd.arg("--strip-debug");
    for sect in &sections {
        // Mach-O removal wants `SEG,SECT` (the linkme sections live in __DATA);
        // ELF takes the bare section name.
        let spec = if cfg!(target_os = "macos") {
            format!("__DATA,{sect}")
        } else {
            (*sect).to_owned()
        };
        cmd.arg("--remove-section").arg(spec);
    }
    cmd.arg(&patched);
    let status = cmd.status().expect("failed to run llvm-objcopy");
    assert!(
        status.success(),
        "llvm-objcopy failed to strip linkme sections from libredisearch_rs.a"
    );

    patched
}

/// Resolve an LLVM binutil: honour `$ENV`, then the Homebrew LLVM location used
/// elsewhere in this build, then fall back to the bare name on `PATH`.
fn llvm_tool(
    env: &str,
    bin: &str,
) -> String {
    if let Ok(p) = std::env::var(env) {
        return p;
    }
    // macOS: Homebrew LLVM.
    let brew = format!("/opt/homebrew/opt/llvm/bin/{bin}");
    if Path::new(&brew).exists() {
        return brew;
    }
    // Linux toolchain images install versioned binaries via apt.llvm.org
    // (e.g. /usr/bin/llvm-objdump-22) and often lack the bare name. Prefer the
    // bare name if present, else the highest available versioned one.
    for cand in [
        bin.to_owned(),
        format!("{bin}-22"),
        format!("{bin}-21"),
        format!("{bin}-20"),
    ] {
        if Path::new(&format!("/usr/bin/{cand}")).exists() {
            return cand;
        }
    }
    bin.to_owned()
}

/// Emit link-search + link-lib for a static archive at `archive`, deriving the
/// `-l` name from its `lib<name>.a` filename.
fn link_static(archive: &Path) {
    let dir = archive.parent().unwrap();
    let stem = archive.file_stem().unwrap().to_str().unwrap();
    let name = stem.strip_prefix("lib").unwrap_or(stem);
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static={name}");
}

/// Recursively collect every `.a` archive under `dir`.
fn find_archives(dir: &Path) -> Vec<PathBuf> {
    let mut archives = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in fs::read_dir(&d).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("a"))
            {
                archives.push(path);
            }
        }
    }
    // `read_dir` yields entries in arbitrary order; sort so the static-library
    // link order is deterministic and the build is reproducible.
    archives.sort();
    archives
}
