//! LAGraph.
//!
//! Kept in lockstep with GraphBLAS: it statically links the `libgraphblas.a`
//! produced by *that* build, so the GraphBLAS cache key is part of LAGraph's key
//! and `graphblas_prefix` is threaded through rather than rediscovered.
//!
//! Upstream has no useful install target for our purposes, so the four artifacts
//! we consume are copied out by hand -- the same four `graphblas.sh` copied into
//! `lagraph_lib/`.

use std::path::Path;

use crate::err;
use crate::error::Result;
use crate::recipes::{CMake, Ctx, cleanup_build_dir, prepare_entry};
use crate::util::copy_file;

pub fn build(
    ctx: &Ctx<'_>,
    entry: &Path,
    graphblas_prefix: &Path,
) -> Result<()> {
    let source = ctx.source("lagraph")?;
    let build_dir = prepare_entry(entry)?;

    let flags = ctx.toolchain.common_c_flags();
    CMake::new(&source, &build_dir)
        .arg("-DBUILD_STATIC_LIBS=ON")
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DLIBRARY_ONLY=ON")
        .arg("-DBUILD_TESTING=OFF")
        .arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")
        .arg(format!(
            "-DGRAPHBLAS_INCLUDE_DIR={}/include/suitesparse",
            graphblas_prefix.display()
        ))
        .arg(format!(
            "-DGRAPHBLAS_LIBRARY={}/lib/libgraphblas.a",
            graphblas_prefix.display()
        ))
        .arg(format!(
            "-DGraphBLAS_DIR={}/lib/cmake/GraphBLAS",
            graphblas_prefix.display()
        ))
        .arg(format!(
            "-DCMAKE_PREFIX_PATH={}",
            graphblas_prefix.display()
        ))
        .arg("-DSUITESPARSE_USE_FORTRAN=OFF")
        .arg(format!("-DCMAKE_C_FLAGS={flags}"))
        .arg(format!("-DCMAKE_CXX_FLAGS={flags}"))
        .args(ctx.toolchain.compiler_cmake_args())
        .args(ctx.toolchain.openmp.cmake_args())
        .compile()?;

    for (from, to) in [
        ("build/src/liblagraph.a", "lib/liblagraph.a"),
        ("build/experimental/liblagraphx.a", "lib/liblagraphx.a"),
    ] {
        // `build/...` here is relative to the scratch tree, not the source.
        let src = build_dir.join(from.trim_start_matches("build/"));
        if !src.is_file() {
            return Err(err!("LAGraph build did not produce {}", src.display()));
        }
        copy_file(&src, &entry.join(to))?;
    }
    for header in ["LAGraph.h", "LAGraphX.h"] {
        copy_file(
            &source.join("include").join(header),
            &entry.join("include").join(header),
        )?;
    }

    cleanup_build_dir(&build_dir);
    Ok(())
}
