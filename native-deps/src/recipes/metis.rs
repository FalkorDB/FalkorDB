//! METIS and GKlib (KarypisLab), the nested-dissection ordering behind the CCH
//! index.
//!
//! The C engine builds both through their own cmake, then has to undo two
//! things that cmake forces on it: METIS's `CMakeLists.txt` turns on LTO (an
//! LLVM-bitcode `libmetis.a` that does not link into non-LTO builds and
//! miscompiled in the LTO one) and `conf/gkbuild.cmake` adds `-march=native`
//! (a SIGILL on any CPU older than the build host's). Neither build system
//! does anything else we need: both libraries are a flat directory of C files
//! with no generated sources. So this recipe compiles them directly, with the
//! shared [`Toolchain`](crate::toolchain::Toolchain) flags, and archives the
//! objects. Nothing in the source tree is touched.
//!
//! The defines are the ones GKlib's own cmake would pass on this platform,
//! used for both libraries so that every translation unit sees `GKlib.h` the
//! same way. No OpenMP: GKlib's `#pragma omp` loops are in its CSR and graph
//! utilities, which METIS never calls (it has its own graph type), and a
//! serial build keeps the library free of any thread-pool interaction.
//!
//! Two decisions are ours. The first is the random number generator: GKlib
//! is built with `USE_GKRAND` (its cmake's `GKRAND` option, off by default
//! and off in the C engine), so METIS draws from GKlib's own generator
//! instead of the C library's `srand`/`rand`. The C library's is one state
//! for the whole process, and anything else in it that calls `rand` (a
//! module's timer jitter, say) would shift the sequence a METIS call draws
//! and so its permutation. GKlib's own is only ever touched by METIS, whose
//! calls the index serialises.
//!
//! The second is the index width. `idx_t` is 64-bit, as in
//! the C engine, so the ordering never caps the graph below what the rest of
//! the index can hold; `real_t` is 32-bit (METIS's default), since it carries
//! only balance tolerances. Both widths are compiled into the archive and
//! written into the installed `metis.h`, so the header always describes the
//! library next to it.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::err;
use crate::error::{Error, Result};
use crate::recipes::{Ctx, cleanup_build_dir, prepare_entry};
use crate::util::{copy_file, jobs, run_isolated};

/// `idx_t` width, in bits. Mirrored by the Rust FFI declarations in
/// `graph/src/index/falkordb/cch/metis.rs`, which assert it.
pub const IDXTYPEWIDTH: u32 = 64;
/// `real_t` width, in bits.
pub const REALTYPEWIDTH: u32 = 32;

/// GKlib: `lib/libGKlib.a` and its headers.
pub fn build_gklib(
    ctx: &Ctx<'_>,
    entry: &Path,
) -> Result<()> {
    let source = ctx.source("gklib")?;
    let build_dir = prepare_entry(entry)?;

    let include = source.join("include");
    let objects = compile(
        ctx,
        &c_files(&source.join("src"))?,
        &[include.as_path()],
        &build_dir,
    )?;
    archive(&objects, &entry.join("lib/libGKlib.a"), &build_dir)?;
    for header in fs::read_dir(&include)? {
        let header = header?.path();
        if header.extension().is_some_and(|e| e == "h") {
            copy_file(
                &header,
                &entry
                    .join("include")
                    .join(header.file_name().expect("a file")),
            )?;
        }
    }

    cleanup_build_dir(&build_dir);
    Ok(())
}

/// METIS: `lib/libmetis.a` and `include/metis.h`, compiled against the GKlib
/// headers installed at `gklib_prefix`.
pub fn build_metis(
    ctx: &Ctx<'_>,
    entry: &Path,
    gklib_prefix: &Path,
) -> Result<()> {
    let source = ctx.source("metis")?;
    let build_dir = prepare_entry(entry)?;

    let libmetis = source.join("libmetis");
    let objects = compile(
        ctx,
        &c_files(&libmetis)?,
        &[
            libmetis.as_path(),
            &source.join("include"),
            &gklib_prefix.join("include"),
        ],
        &build_dir,
    )?;
    archive(&objects, &entry.join("lib/libmetis.a"), &build_dir)?;

    // What METIS's own `make config` writes into build/xinclude/metis.h: the
    // widths, then the upstream header, which leaves them to the includer.
    let header = fs::read_to_string(source.join("include/metis.h"))?;
    let out = entry.join("include/metis.h");
    fs::create_dir_all(out.parent().expect("include/"))?;
    fs::write(
        &out,
        format!(
            "#define IDXTYPEWIDTH {IDXTYPEWIDTH}\n#define REALTYPEWIDTH {REALTYPEWIDTH}\n{header}"
        ),
    )?;

    cleanup_build_dir(&build_dir);
    Ok(())
}

/// Every `*.c` directly in `dir`, sorted, so the archive's member order does
/// not depend on the filesystem.
fn c_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for file in fs::read_dir(dir).map_err(|e| err!("cannot list {}: {e}", dir.display()))? {
        let path = file?.path();
        if path.extension().is_some_and(|e| e == "c") {
            out.push(path);
        }
    }
    if out.is_empty() {
        return Err(err!("no C sources in {}", dir.display()));
    }
    out.sort();
    Ok(out)
}

/// The defines GKlib's cmake passes on this platform (its `CMakeLists.txt`
/// feature checks, all of which hold on the Linux and macOS targets we
/// build), plus `USE_GKRAND` (see the module docs) and the widths.
/// `_FILE_OFFSET_BITS` comes from METIS's `conf/gkbuild.cmake`.
fn defines() -> Vec<String> {
    let mut out: Vec<String> = [
        "NDEBUG",
        "NDEBUG2",
        "HAVE_REGEX_H",
        "HAVE_GETLINE",
        "_FILE_OFFSET_BITS=64",
        "USE_GKRAND",
    ]
    .iter()
    .map(|d| format!("-D{d}"))
    .collect();
    if cfg!(target_os = "linux") {
        out.push("-DLINUX".to_owned());
        out.push("-D_POSIX_C_SOURCE=200809L".to_owned());
    }
    out.push(format!("-DIDXTYPEWIDTH={IDXTYPEWIDTH}"));
    out.push(format!("-DREALTYPEWIDTH={REALTYPEWIDTH}"));
    out
}

/// Compile each of `sources` to an object in `build_dir`, `jobs()` at a time.
/// Objects are named after their source, so the two never collide within one
/// library.
fn compile(
    ctx: &Ctx<'_>,
    sources: &[PathBuf],
    includes: &[&Path],
    build_dir: &Path,
) -> Result<Vec<PathBuf>> {
    let cc = ctx.toolchain.cc.clone().unwrap_or_else(|| "cc".to_owned());
    let mut flags: Vec<OsString> = ctx
        .toolchain
        .common_c_flags()
        .split_whitespace()
        .map(OsString::from)
        .collect();
    flags.extend(defines().into_iter().map(OsString::from));
    for dir in includes {
        let mut flag = OsString::from("-I");
        flag.push(dir);
        flags.push(flag);
    }

    let objects: Vec<PathBuf> = sources
        .iter()
        .map(|src| {
            build_dir
                .join(src.file_name().expect("a file"))
                .with_extension("o")
        })
        .collect();
    let next = AtomicUsize::new(0);
    let failure: Mutex<Option<Error>> = Mutex::new(None);
    std::thread::scope(|s| {
        for _ in 0..jobs().min(sources.len()) {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= sources.len() || failure.lock().expect("lock").is_some() {
                        return;
                    }
                    let mut args: Vec<&OsStr> = flags.iter().map(OsString::as_os_str).collect();
                    args.extend([
                        OsStr::new("-c"),
                        sources[k].as_os_str(),
                        OsStr::new("-o"),
                        objects[k].as_os_str(),
                    ]);
                    if let Err(e) = run_isolated(&cc, &args, build_dir, &Default::default()) {
                        failure.lock().expect("lock").get_or_insert(e);
                        return;
                    }
                }
            });
        }
    });
    match failure.into_inner().expect("lock") {
        Some(e) => Err(e),
        None => Ok(objects),
    }
}

/// `ar` the objects into `lib`, with a symbol index.
fn archive(
    objects: &[PathBuf],
    lib: &Path,
    build_dir: &Path,
) -> Result<()> {
    fs::create_dir_all(lib.parent().expect("lib/"))?;
    let mut args: Vec<&OsStr> = vec![OsStr::new("crs"), lib.as_os_str()];
    args.extend(objects.iter().map(|o| o.as_os_str()));
    run_isolated("ar", &args, build_dir, &Default::default())
}
