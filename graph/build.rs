use std::fs;

// Suppress too_many_lines: this build script handles multiple platform/configuration
// cases and splitting it would reduce clarity.
#[allow(clippy::too_many_lines)]
fn main() {
    // libomp linking strategy (same shape on Linux + macOS):
    //   * if `${LIBOMP_PREFIX:-/opt/libomp}/lib/libomp.a` exists → link static.
    //     This makes libfalkordb.{so,dylib} self-contained for OpenMP and is
    //     the path CI/Docker takes (build/libomp.sh installs to /opt/libomp).
    //   * otherwise → fall back to dynamic `-lomp`, resolved against the
    //     system search path (apt's libomp-22-dev on Linux, homebrew's
    //     /opt/homebrew/opt/llvm/lib/libomp.dylib on macOS).
    // The LIBOMP_PREFIX env var lets local devs point at a non-root install
    // (e.g. PREFIX=$HOME/libomp ./build/libomp.sh) without needing sudo.
    let libomp_prefix =
        std::env::var("LIBOMP_PREFIX").unwrap_or_else(|_| "/opt/libomp".to_string());
    let libomp_static = std::path::Path::new(&libomp_prefix)
        .join("lib/libomp.a")
        .exists();
    println!("cargo:rustc-link-search={libomp_prefix}/lib");

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

    if libomp_static {
        println!("cargo:rustc-link-lib=static=omp");
    } else {
        println!("cargo:rustc-link-lib=omp");
    }

    // libgraphblas.a search path. Defaults to /usr/local/lib (what
    // graphblas.sh installs to). Set GRAPHBLAS_LIB_DIR to point at a
    // local out-of-tree build directory — used by gen_prejit.sh to
    // avoid needing sudo for the harvest's intermediate GraphBLAS build.
    let graphblas_lib_dir =
        std::env::var("GRAPHBLAS_LIB_DIR").unwrap_or_else(|_| "/usr/local/lib".to_string());
    println!("cargo:rerun-if-env-changed=GRAPHBLAS_LIB_DIR");
    println!("cargo:rustc-link-search={graphblas_lib_dir}");
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

    // ---- RediSearch 8.6, embedded as a static library ----
    // `redisearch.sh` builds the fork into
    // redisearch/RediSearch/bin/<variant>/search-community. The main archive is
    // `redisearch.so` (an ar archive despite the suffix); its C/C++ dependencies are
    // sibling .a files; the Rust crate is a separate libredisearch_rs.a one level up.
    // Local dev (and the asan Dockerfile, which runs redisearch.sh in-tree)
    // build into `<repo>/redisearch/RediSearch/bin`; the Docker toolchain image
    // (build/Dockerfile) builds into `/data/redisearch/RediSearch/bin`. Take the
    // first that exists so the same build.rs works in every environment.
    let rs_bin = [
        std::path::Path::new(&manifest_dir).join("../redisearch/RediSearch/bin"),
        std::path::PathBuf::from("/data/redisearch/RediSearch/bin"),
    ]
    .into_iter()
    .find_map(|p| p.canonicalize().ok().filter(|p| p.is_dir()))
    .expect("redisearch/RediSearch/bin missing - run ./redisearch.sh first");

    // The main archive is `redisearch.so` (macOS release) or `redisearch.a`
    // (Linux / sanitizer `debug-asan`), an ar archive in both cases. The
    // sanitizer build also emits two variant dirs (e.g. linux-aarch64-debug-asan
    // and linux-arm64v8-debug-asan), only one of which holds the archive, so
    // require its presence when picking the search dir.
    let main_names = ["libredisearch.a", "redisearch.a", "redisearch.so"];
    let search_dir = fs::read_dir(&rs_bin)
        .unwrap()
        .flatten()
        .map(|e| e.path().join("search-community"))
        .find(|p| p.is_dir() && main_names.iter().any(|n| p.join(n).exists()))
        .expect(
            "search-community dir with a redisearch archive not found - run ./redisearch.sh first",
        );

    // Expose the main archive under a lib*.a name so `-l static=redisearch` can
    // find it. This matters: unlike `cargo:rustc-link-arg`, link-lib/link-search
    // directives propagate from this (dependency) build script to the dependent
    // `falkordb` cdylib's final link, where RediSearch symbols are resolved.
    let main_a = search_dir.join("libredisearch.a");
    if !main_a.exists() {
        let src = ["redisearch.a", "redisearch.so"]
            .iter()
            .map(|n| search_dir.join(n))
            .find(|p| p.exists())
            .expect("RediSearch main archive (redisearch.a/.so) not found");
        std::os::unix::fs::symlink(src, &main_a).expect("failed to create libredisearch.a symlink");
    }

    // Link the main archive and every C/C++ dependency archive. Redundant archives
    // are harmless: the linker only pulls members that resolve an undefined symbol.
    for archive in find_archives(&search_dir) {
        link_static(&archive);
    }

    // RediSearch's Rust crate is a separate archive one level up. The profile
    // subdir is `release` for a normal build but `debug-asan` (etc.) under a
    // sanitizer build, so search for it by name rather than hard-coding the path.
    let rs_rs = find_archives(&rs_bin.join("redisearch_rs"))
        .into_iter()
        .find(|p| p.file_name().and_then(|n| n.to_str()) == Some("libredisearch_rs.a"))
        .expect("libredisearch_rs.a missing under redisearch_rs/ - run ./redisearch.sh first");

    // libredisearch_rs.a embeds its own copy of the `redis-module` crate (pulled
    // by rlookup / redis_mock). That crate unconditionally defines linkme
    // `#[distributed_slice]`s (COMMANDS_LIST, configs, info, ...). FalkorDB links
    // its own `redis-module`, so without intervention the final cdylib carries two
    // definitions of each slice and linkme aborts at module load with
    // `duplicate #[distributed_slice] with name "COMMANDS_LIST"`. In RediSearch's
    // library build those slices are empty and never iterated, so we strip the
    // linkme sections from a private copy of the archive and link that instead.
    let patched = strip_linkme_sections(&rs_rs);
    link_static(&patched);
}

/// RediSearch's Rust archive carries a second copy of the `redis-module` crate
/// whose linkme `#[distributed_slice]` markers collide with FalkorDB's at load
/// time. Produce a copy in `OUT_DIR` with every linkme section removed and return
/// its path. The slices are empty in RediSearch's library build, so dropping them
/// only removes dead bookkeeping.
fn strip_linkme_sections(archive: &std::path::Path) -> std::path::PathBuf {
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
    let patched = std::path::Path::new(&out_dir).join("libredisearch_rs.a");
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
    if std::path::Path::new(&brew).exists() {
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
        if std::path::Path::new(&format!("/usr/bin/{cand}")).exists() {
            return cand;
        }
    }
    bin.to_owned()
}

/// Emit link-search + link-lib for a static archive at `archive`, deriving the
/// `-l` name from its `lib<name>.a` filename.
fn link_static(archive: &std::path::Path) {
    let dir = archive.parent().unwrap();
    let stem = archive.file_stem().unwrap().to_str().unwrap();
    let name = stem.strip_prefix("lib").unwrap_or(stem);
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static={name}");
}

/// Recursively collect every `.a` archive under `dir`.
fn find_archives(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
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
    archives
}
