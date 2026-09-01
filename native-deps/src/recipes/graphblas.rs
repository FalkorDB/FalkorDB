//! SuiteSparse:GraphBLAS.
//!
//! Two things make this more than a stock cmake build:
//!
//! 1. `build/graphblas/GB_control.patch` disables the FP32/FP64/FC32/FC64
//!    FactoryKernel families that FalkorDB query plans never hit, mirroring the
//!    tweak in the C engine's vendored copy.
//! 2. `build/graphblas/PreJIT/GB_jit_*.c` are harvested kernels (see
//!    `gen_prejit.sh`). GraphBLAS's cmake globs `PreJIT/*.c` and bakes them into
//!    `libgraphblas.a`, giving factory-comparable speed for the operations we
//!    actually execute without any runtime JIT compilation.
//!
//! Both are cache-key inputs, so a re-harvest or a patch edit produces a new
//! key rather than a silently stale archive.
//!
//! Harvest mode (`FALKORDB_PREJIT_HARVEST=1`, set by `gen_prejit.sh`) skips step
//! 2 so every op falls through to the JIT engine and writes a fresh kernel into
//! `~/.SuiteSparse/GrBx.y.z/c/` for collection.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;

use crate::err;
use crate::error::Result;
use crate::recipes::{CMake, Ctx, SourceGuard, cleanup_build_dir, prepare_entry};
use crate::util::{copy_file, log, run};

/// Build GraphBLAS and install it into `entry`, which becomes a cmake prefix
/// (`include/suitesparse`, `lib/libgraphblas.a`, `lib/cmake/GraphBLAS`).
pub fn build(
    ctx: &Ctx<'_>,
    entry: &Path,
) -> Result<()> {
    let source = ctx.source("graphblas")?;
    let build_dir = prepare_entry(entry)?;

    let mut guard = SourceGuard::new();
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

    cleanup_build_dir(&build_dir);

    let archive = entry.join("lib/libgraphblas.a");
    if !archive.is_file() {
        return Err(err!(
            "GraphBLAS build finished but {} is missing",
            archive.display()
        ));
    }
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
    run(
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

    let vendor_dir = ctx.root.join("build/graphblas/PreJIT");
    let dest_dir = source.join("PreJIT");
    let mut kernels: Vec<_> = fs::read_dir(&vendor_dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "c")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("GB_jit_"))
        })
        .collect();
    kernels.sort();

    if kernels.is_empty() {
        log(&format!(
            "no PreJIT kernels in {} - run gen_prejit.sh to populate",
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
