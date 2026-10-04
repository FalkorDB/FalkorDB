//! RediSearch (the FalkorDB fork), embedded as a static library.
//!
//! Unlike GraphBLAS and LAGraph this one builds in place: RediSearch's own
//! `build.sh` insists on writing to `<source>/bin`, and letting it do so keeps
//! its cmake and cargo caches warm. The artifacts are then collected into the
//! cache entry under *stable* names, which is what lets `graph/build.rs` stop
//! spelunking through `bin/<variant>/search-community` and guessing flavors.
//!
//! Published layout:
//!
//! ```text
//! <entry>/lib/libredisearch.a      the main archive, renamed
//! <entry>/lib/deps/**.a            its C/C++ dependency archives
//! <entry>/rs/libredisearch_rs.a    the Rust archive (linkme-stripped by build.rs)
//! ```

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use crate::err;
use crate::error::Result;
use crate::recipes::{Ctx, SourceGuard, cleanup_build_dir, prepare_entry};
use crate::util::{capture_opt, copy_file, find_archives, log, run_isolated};

const MAIN_ARCHIVE_NAMES: [&str; 2] = ["redisearch.a", "redisearch.so"];
const VECSIM_CMAKELISTS: &str = "deps/VectorSimilarity/src/VecSim/CMakeLists.txt";

pub fn build(
    ctx: &Ctx<'_>,
    entry: &Path,
) -> Result<()> {
    let source = ctx.source("redisearch")?;
    // build.sh builds in place, under bin/, so the scratch dir stays empty.
    let scratch = prepare_entry(entry)?;

    let mut guard = SourceGuard::new(
        ctx.source_state("redisearch")?.join("journal"),
        &ctx.lock.get("redisearch")?.rev,
    )?;
    // `src/redisearch_rs/ffi/build.rs` resolves its include paths relative to
    // the nearest ancestor containing a `.git`, and panics if there is none.
    super::ensure_git_root(&source, &mut guard)?;
    relax_vecsim_warnings(&source, &mut guard)?;

    // Build RediSearch 8.6 as an embeddable STATIC library:
    //   REDISEARCH_BUILD_AS_LIBRARY=ON  omit src/module_main.c and its
    //                                   RedisModule_OnLoad; FalkorDB supplies
    //                                   its own.
    //   REDISEARCH_BUILD_SHARED=OFF     produce a static archive. A standalone
    //                                   build defaults to SHARED, so this is
    //                                   required, not merely tidy.
    //   CMAKE_POSITION_INDEPENDENT_CODE=ON  the archive lands in our cdylib.
    // On macOS build.sh forces clang/clang++ from PATH (Homebrew LLVM, needed
    // for OpenMP in VectorSimilarity); on Linux the compiler comes from
    // $CC/$CXX, since the toolchain image uses versioned clang-NN.
    let mut cmake_args = String::from(
        "-DREDISEARCH_BUILD_AS_LIBRARY=ON -DREDISEARCH_BUILD_SHARED=OFF \
         -DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    );
    if let Some(cc) = &ctx.toolchain.cc {
        cmake_args.push_str(&format!(" -DCMAKE_C_COMPILER={cc}"));
    }
    if let Some(cxx) = &ctx.toolchain.cxx {
        cmake_args.push_str(&format!(" -DCMAKE_CXX_COMPILER={cxx}"));
    }

    let mut env = BTreeMap::new();
    env.insert("CMAKE_ARGS".to_owned(), cmake_args);

    let mut args: Vec<&OsStr> = vec![OsStr::new("BUILD_SEARCH_UNIT_TESTS=OFF")];
    let san_arg;
    if let Some(san) = ctx.san {
        install_nightly_rust_src(&source)?;
        san_arg = format!("SAN={san}");
        args.push(OsStr::new(&san_arg));
    }

    run_isolated("./build.sh", &args, &source, &env)?;
    collect(&source, entry, ctx.san.is_some())?;
    cleanup_build_dir(&scratch);
    Ok(())
}

/// VecSim promotes some warnings to errors; our toolchain is newer than the one
/// it was tuned for.
fn relax_vecsim_warnings(
    source: &Path,
    guard: &mut SourceGuard,
) -> Result<()> {
    let path = source.join(VECSIM_CMAKELISTS);
    guard.snapshot(&path)?;
    let text =
        fs::read_to_string(&path).map_err(|e| err!("cannot read {}: {e}", path.display()))?;
    fs::write(&path, text.replace("-Werror", ""))
        .map_err(|e| err!("cannot write {}: {e}", path.display()))?;
    Ok(())
}

/// A sanitizer build compiles `redisearch_rs`'s std with `-Zbuild-std` using the
/// nightly pinned in `.rust-nightly`, which needs the `rust-src` component.
fn install_nightly_rust_src(source: &Path) -> Result<()> {
    let pin = source.join(".rust-nightly");
    let Ok(toolchain) = fs::read_to_string(&pin) else {
        return Ok(());
    };
    let toolchain = toolchain.trim();
    if toolchain.is_empty() {
        return Ok(());
    }
    // Best effort, exactly as redisearch.sh had it: an offline or already
    // provisioned machine must not fail the build here.
    if capture_opt("rustup", &["--version"]).is_none() {
        log("rustup not found; skipping rust-src provisioning for the sanitizer build");
        return Ok(());
    }
    let _ = run_isolated(
        "rustup",
        &[
            OsStr::new("toolchain"),
            OsStr::new("install"),
            OsStr::new(toolchain),
            OsStr::new("-c"),
            OsStr::new("rust-src"),
        ],
        source,
        &BTreeMap::new(),
    );
    Ok(())
}

fn collect(
    source: &Path,
    entry: &Path,
    want_asan: bool,
) -> Result<()> {
    let bin = source.join("bin");
    let search_dir = pick_variant(&bin, want_asan)?;
    log(&format!(
        "collecting archives from {}",
        search_dir.display()
    ));

    // The main archive is `redisearch.so` on macOS release builds and
    // `redisearch.a` on Linux / sanitizer builds -- an ar archive either way.
    // Republish it as `libredisearch.a` so `-l static=redisearch` resolves.
    let main = MAIN_ARCHIVE_NAMES
        .iter()
        .map(|n| search_dir.join(n))
        .find(|p| p.is_file())
        .ok_or_else(|| {
            err!(
                "no redisearch.a/.so under {} after build.sh",
                search_dir.display()
            )
        })?;
    copy_file(&main, &entry.join("lib/libredisearch.a"))?;

    let mut count = 0usize;
    for archive in find_archives(&search_dir)? {
        if archive == main {
            continue;
        }
        let rel = archive.strip_prefix(&search_dir).unwrap_or(&archive);
        copy_file(&archive, &entry.join("lib/deps").join(rel))?;
        count += 1;
    }

    // RediSearch's Rust crate is a separate archive one level up, under a
    // profile subdir named for the flavor (`release`, `debug-asan`, ...). Both
    // flavors' outputs coexist there, so match the profile, not just the name:
    // sorted, `debug-asan` comes first and a clean build would publish it.
    let rs_root = bin.join("redisearch_rs");
    let rs = find_archives(&rs_root)?
        .into_iter()
        .filter(|p| p.file_name().and_then(|n| n.to_str()) == Some("libredisearch_rs.a"))
        .find(|p| {
            p.strip_prefix(&rs_root)
                .ok()
                .and_then(|rel| rel.components().next())
                .is_some_and(|profile| is_asan(&profile.as_os_str().to_string_lossy()) == want_asan)
        })
        .ok_or_else(|| {
            err!(
                "libredisearch_rs.a missing under {}",
                bin.join("redisearch_rs").display()
            )
        })?;
    copy_file(&rs, &entry.join("rs/libredisearch_rs.a"))?;

    log(&format!(
        "collected libredisearch.a + {count} dependency archives + libredisearch_rs.a"
    ));
    Ok(())
}

/// Whether a build.sh variant or profile name is the sanitizer flavor.
fn is_asan(name: &str) -> bool {
    name.contains("asan")
}

/// Choose `bin/<variant>/search-community` for the requested flavor.
///
/// A sanitizer build emits two variant dirs (e.g. `linux-aarch64-debug-asan` and
/// `linux-arm64v8-debug-asan`), only one of which holds the archive, so require
/// its presence. There is deliberately no cross-flavor fallback: the flavor is a
/// cache-key input now, and quietly publishing the wrong one is precisely the
/// bug this design removes.
fn pick_variant(
    bin: &Path,
    want_asan: bool,
) -> Result<PathBuf> {
    let mut variants: Vec<PathBuf> = fs::read_dir(bin)
        .map_err(|e| err!("cannot list {}: {e}", bin.display()))?
        .flatten()
        .map(|e| e.path().join("search-community"))
        .collect();
    variants.sort();

    variants
        .iter()
        .find(|p| {
            p.is_dir()
                && MAIN_ARCHIVE_NAMES.iter().any(|n| p.join(n).is_file())
                // The variant dir's own name, not the whole path: a checkout
                // under a path containing "asan" must not flip the flavor.
                && p.parent()
                    .and_then(Path::file_name)
                    .is_some_and(|v| is_asan(&v.to_string_lossy()) == want_asan)
        })
        .cloned()
        .ok_or_else(|| {
            err!(
                "no {} search-community dir with a redisearch archive under {}; found: {}",
                if want_asan {
                    "sanitizer"
                } else {
                    "non-sanitizer"
                },
                bin.display(),
                variants
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::collect;
    use crate::testing::TempDir;

    fn put(
        path: &Path,
        contents: &str,
    ) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn each_flavor_collects_its_own_archives() {
        // Both flavors' outputs side by side, as build.sh leaves them, under a
        // checkout whose path itself contains "asan".
        let tmp = TempDir::new("asan-in-the-path");
        let source = tmp.0.join("deps/RediSearch");
        let bin = source.join("bin");
        put(
            &bin.join("linux-x64-release/search-community/redisearch.a"),
            "main-clean",
        );
        put(
            &bin.join("linux-x64-debug-asan/search-community/redisearch.a"),
            "main-asan",
        );
        put(
            &bin.join("redisearch_rs/release/libredisearch_rs.a"),
            "rs-clean",
        );
        put(
            &bin.join("redisearch_rs/debug-asan/libredisearch_rs.a"),
            "rs-asan",
        );

        for (want_asan, flavor) in [(false, "clean"), (true, "asan")] {
            let entry = tmp.0.join(format!("entry-{flavor}"));
            collect(&source, &entry, want_asan).unwrap();
            let read = |p: &str| fs::read_to_string(entry.join(p)).unwrap();
            assert_eq!(read("lib/libredisearch.a"), format!("main-{flavor}"));
            assert_eq!(read("rs/libredisearch_rs.a"), format!("rs-{flavor}"));
        }
    }
}
