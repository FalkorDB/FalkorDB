//! SuiteSparse:GraphBLAS.
//!
//! Two things make this more than a stock cmake build:
//!
//! 1. `build/graphblas/GB_control.patch` disables the FP32/FP64/FC32/FC64
//!    FactoryKernel families that FalkorDB query plans never hit, mirroring the
//!    tweak in the C engine's vendored copy.
//! 2. `build/graphblas/PreJIT/GB_jit_*.c` are harvested kernels (see
//!    `native-deps prejit`). GraphBLAS's cmake globs `PreJIT/*.c` and bakes them into
//!    `libgraphblas.a`, giving factory-comparable speed for the operations we
//!    actually execute without any runtime JIT compilation.
//!
//! Both are cache-key inputs, so a re-harvest or a patch edit produces a new
//! key rather than a silently stale archive.
//!
//! Harvest mode (`FALKORDB_PREJIT_HARVEST=1`, set by `native-deps prejit`) skips
//! step 2 so every op falls through to the JIT engine and writes a fresh kernel
//! into the JIT cache, which [`harvest_prejit`] then copies back. Vendoring and
//! harvesting are the two directions of one flow, so both live here.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use crate::err;
use crate::error::Result;
use crate::hash::collect_files;
use crate::recipes::{CMake, Ctx, SourceGuard, cleanup_build_dir, prepare_entry};
use crate::util::{copy_file, env_opt, is_prejit_kernel, log, run_isolated};

/// The vendored kernels, relative to the FalkorDB checkout.
pub const PREJIT_DIR: &str = "build/graphblas/PreJIT";

/// The GraphBLAS version the submodule is at, e.g. `10.5.0`, read from its
/// cmake. v10.5.0 renamed `GraphBLAS_VERSION_*` to `GraphBLAS_VER_*`; accept
/// both so a bump across that rename is not a silent failure.
pub fn version(source: &Path) -> Result<String> {
    let file = source.join("cmake_modules/GraphBLAS_version.cmake");
    let text = fs::read_to_string(&file).map_err(|e| {
        err!(
            "cannot read {}: {e} - run `git submodule update --init deps/GraphBLAS`",
            file.display()
        )
    })?;
    let field = |part: &str| -> Result<String> {
        ["VER", "VERSION"]
            .iter()
            .find_map(|prefix| {
                let tag = format!("GraphBLAS_{prefix}_{part} ");
                text.lines()
                    .find(|l| l.trim_start().starts_with("set") && l.contains(&tag))
                    .and_then(|l| l.split(&tag).nth(1))
                    .and_then(|rest| rest.split_whitespace().next())
                    .map(str::to_owned)
            })
            .ok_or_else(|| err!("{}: no GraphBLAS_VER_{part}", file.display()))
    };
    Ok(format!(
        "{}.{}.{}",
        field("MAJOR")?,
        field("MINOR")?,
        field("SUB")?
    ))
}

/// GraphBLAS's runtime JIT cache for `version`: `~/.SuiteSparse/GrB<version>`.
pub fn jit_cache(version: &str) -> Result<PathBuf> {
    let home = env_opt("HOME").ok_or_else(|| err!("HOME is not set"))?;
    Ok(PathBuf::from(home).join(format!(".SuiteSparse/GrB{version}")))
}

/// Start a harvest from nothing: delete the vendored kernels and the JIT
/// cache's compiled output, so what [`harvest_prejit`] later finds is exactly
/// what the harvest run compiled.
pub fn clear_prejit(
    root: &Path,
    jit_cache: &Path,
) -> Result<()> {
    let vendor = root.join(PREJIT_DIR);
    fs::create_dir_all(&vendor)?;
    for entry in fs::read_dir(&vendor)?.flatten() {
        if is_prejit_kernel(&entry.path()) {
            fs::remove_file(entry.path())?;
        }
    }
    for sub in ["c", "lib", "tmp"] {
        // A leftover kernel would be harvested as if this run compiled it.
        match fs::remove_dir_all(jit_cache.join(sub)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(err!("cannot clear {}: {e}", jit_cache.join(sub).display()));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Copy every kernel the JIT compiled into the vendored set -- the reverse of
/// [`vendor_prejit`], with the same filter. Returns how many were copied.
pub fn harvest_prejit(
    root: &Path,
    jit_cache: &Path,
) -> Result<usize> {
    let compiled = jit_cache.join("c");
    let kernels: Vec<PathBuf> = collect_files(&compiled)
        .map_err(|e| err!("cannot list {}: {e}", compiled.display()))?
        .into_iter()
        .filter(|p| is_prejit_kernel(p))
        .collect();
    if kernels.is_empty() {
        return Err(err!(
            "no kernels under {} - did the harvest build run with \
             FALKORDB_PREJIT_HARVEST=1 and --features prejit_harvest?",
            compiled.display()
        ));
    }
    let vendor = root.join(PREJIT_DIR);
    for kernel in &kernels {
        copy_file(kernel, &vendor.join(kernel.file_name().unwrap_or_default()))?;
    }
    Ok(kernels.len())
}

/// Build GraphBLAS and install it into `entry`, which becomes a cmake prefix
/// (`include/suitesparse`, `lib/libgraphblas.a`, `lib/cmake/GraphBLAS`).
pub fn build(
    ctx: &Ctx<'_>,
    entry: &Path,
) -> Result<()> {
    let source = ctx.source("graphblas")?;
    let build_dir = prepare_entry(entry)?;

    let mut guard = SourceGuard::new(
        ctx.source_state("graphblas")?.join("journal"),
        &ctx.lock.get("graphblas")?.rev,
    )?;
    apply_patch(ctx, &source, &mut guard)?;
    vendor_prejit(ctx, &source, &mut guard)?;
    // GraphBLAS_PreJIT.cmake regenerates this tracked file in the source tree
    // from whatever PreJIT/*.c it finds, so it has to be restored too or the
    // submodule stays dirty after every build.
    guard.snapshot(&source.join("Config/GB_prejit.c"))?;

    let flags = ctx.toolchain.common_c_flags();
    CMake::new(&source, &build_dir)
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", entry.display()))
        .arg("-DSUITESPARSE_USE_FORTRAN=OFF")
        .arg("-DBUILD_STATIC_LIBS=ON")
        .arg("-DBUILD_SHARED_LIBS=OFF")
        // COMPACT=OFF keeps FactoryKernels enabled; disabling them is
        // incompatible with the PreJIT-baking strategy above.
        .arg("-DGRAPHBLAS_COMPACT=OFF")
        .arg("-DGRAPHBLAS_BUILD_STATIC_LIBS=ON")
        .arg("-DBUILD_TESTING=OFF")
        // Enables PreJIT compilation so the vendored kernels land in the
        // archive. The *runtime* JIT is separately pinned to GxB_JIT_PAUSE in
        // matrix.rs, which keeps us fork-safe and avoids dlopen at query time.
        .arg("-DGRAPHBLAS_USE_JIT=1")
        .arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")
        .arg(format!("-DCMAKE_C_FLAGS={flags}"))
        .arg(format!("-DCMAKE_CXX_FLAGS={flags}"))
        .args(ctx.toolchain.compiler_cmake_args())
        .args(ctx.toolchain.openmp.cmake_args())
        .pipeline()?;

    // Checked before the scratch tree goes: its cmake logs are what explain a
    // missing archive.
    let archive = entry.join("lib/libgraphblas.a");
    if !archive.is_file() {
        return Err(err!(
            "GraphBLAS build finished but {} is missing (build tree kept at {})",
            archive.display(),
            build_dir.display()
        ));
    }
    cleanup_build_dir(&build_dir);
    Ok(())
}

fn apply_patch(
    ctx: &Ctx<'_>,
    source: &Path,
    guard: &mut SourceGuard,
) -> Result<()> {
    let patch = ctx.root.join("build/graphblas/GB_control.patch");
    // The patch only touches Source/GB_control.h; snapshot it so the submodule
    // goes back to pristine when this build finishes.
    guard.snapshot(&source.join("Source/GB_control.h"))?;
    super::ensure_git_root(source, guard)?;
    log(&format!("applying {}", patch.display()));
    run_isolated(
        "git",
        &[OsStr::new("apply"), patch.as_os_str()],
        source,
        &BTreeMap::new(),
    )
}

fn vendor_prejit(
    ctx: &Ctx<'_>,
    source: &Path,
    guard: &mut SourceGuard,
) -> Result<()> {
    if ctx.prejit_harvest {
        log("FALKORDB_PREJIT_HARVEST=1: skipping PreJIT vendoring (harvest mode)");
        return Ok(());
    }

    let vendor_dir = ctx.root.join(PREJIT_DIR);
    let dest_dir = source.join("PreJIT");
    let mut kernels: Vec<_> = fs::read_dir(&vendor_dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| is_prejit_kernel(p))
        .collect();
    kernels.sort();

    if kernels.is_empty() {
        log(&format!(
            "no PreJIT kernels in {} - run `native-deps prejit` to populate",
            vendor_dir.display()
        ));
        return Ok(());
    }

    // Kernels MUST be harvested on Linux inside the Docker toolchain image: the
    // JIT `defn` strings baked into each kernel are captured after host header
    // macro expansion (Apple's fortify rewrites memcpy to __builtin___memcpy_chk,
    // for one), so a macOS-harvested kernel fails its `_query` hash check on
    // Linux and silently falls back to slow generic kernels.
    for kernel in &kernels {
        let dest = dest_dir.join(kernel.file_name().unwrap_or_default());
        guard.snapshot(&dest)?;
        copy_file(kernel, &dest)?;
    }
    log(&format!(
        "vendored {} PreJIT kernels from {}",
        kernels.len(),
        vendor_dir.display()
    ));
    Ok(())
}
