//! `native-deps prejit`: regenerate `build/graphblas/PreJIT/`.
//!
//! GraphBLAS compiles a kernel the first time a query needs an operation shape,
//! and PreJIT bakes those kernels into the archive so production never compiles
//! at runtime. Which kernels exist depends on what FalkorDB executes, not on
//! GraphBLAS's inputs, so they are committed sources rather than a cache entry,
//! and regenerating them means running FalkorDB:
//!
//! 1. clear the vendored kernels and the JIT cache;
//! 2. build in harvest mode -- GraphBLAS without PreJIT, FalkorDB with the
//!    runtime JIT on (`--features prejit_harvest`);
//! 3. run the workload: every suite, plus the benchmark queries, whose shapes
//!    no functional suite issues;
//! 4. copy what the JIT compiled into `build/graphblas/PreJIT/`;
//! 5. build normally, which bakes them in.
//!
//! A failing workload step is reported, not fatal: the kernels of every step
//! that ran are still valid. Run it inside the Linux toolchain image
//! (`ghcr.io/falkordb/falkordb-build`): macOS headers expand into the kernels'
//! JIT definition strings, and a kernel harvested there fails its hash check
//! on Linux, silently falling back to slow generic kernels.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::Path;

use crate::bail;
use crate::error::Result;
use crate::lock::LockFile;
use crate::recipes::graphblas;
use crate::util::{capture_opt, copy_file, env_flag, env_opt, is_prejit_kernel, log, run};

pub fn regenerate(root: &Path) -> Result<()> {
    if env_flag("FALKORDB_PREJIT_HARVEST") {
        bail!(
            "FALKORDB_PREJIT_HARVEST is set; unset it -- prejit sets it for the harvest \
             build itself, and the final build must run without it"
        );
    }
    let source = LockFile::load(root)?.source_dir(root, "graphblas")?;
    let version = graphblas::version(&source)?;
    let jit_cache = graphblas::jit_cache(&version)?;
    log(&format!(
        "regenerating PreJIT kernels for GraphBLAS {version} (JIT cache {})",
        jit_cache.display()
    ));
    if cfg!(target_os = "macos") {
        log(
            "WARNING: harvesting on macOS -- these kernels will fail their hash check \
             on Linux; regenerate inside the Linux toolchain image for anything you commit",
        );
    }

    // Checked before anything is cleared: the benchmark queries carry shapes
    // no functional suite issues, and a harvest without them would quietly
    // drop those kernels -- the very regression PreJIT exists to prevent.
    if !root.join("bench/pyproject.toml").is_file() || capture_opt("uv", &["--version"]).is_none() {
        bail!(
            "the benchmark workload needs bench/pyproject.toml and `uv` on PATH \
             (`pip install uv`); without it benchmark-only kernels go missing"
        );
    }

    let env = workload_env(root);
    let mut harvest = env.clone();
    harvest.insert("FALKORDB_PREJIT_HARVEST".into(), "1".into());
    link_static_libomp_for_jit();

    // The vendored kernels are cleared first, so put them back if anything
    // before the harvest fails: a failed run must not leave the directory empty.
    let backup = Backup::take(root)?;
    let outcome = (|| -> Result<(usize, Vec<&'static str>)> {
        graphblas::clear_prejit(root, &jit_cache)?;
        cargo(
            root,
            &harvest,
            &["build", "--release", "--features", "prejit_harvest"],
        )?;
        let mut failures = Vec::new();
        for (label, program, args) in workload() {
            log(&format!("workload: {label}"));
            let args: Vec<&OsStr> = args.iter().map(OsStr::new).collect();
            if let Err(e) = run(program, &args, root, &harvest) {
                log(&format!("WARNING: {label} failed ({e}); continuing"));
                failures.push(label);
            }
        }
        Ok((graphblas::harvest_prejit(root, &jit_cache)?, failures))
    })();
    let (harvested, failures) = match outcome {
        Ok(done) => {
            backup.discard();
            done
        }
        Err(e) => {
            backup.restore()?;
            return Err(e);
        }
    };
    log(&format!(
        "harvested {harvested} kernel(s) into {}",
        graphblas::PREJIT_DIR
    ));
    cargo(root, &env, &["build", "--release"])?;

    if !failures.is_empty() {
        log(&format!(
            "these workload steps failed, so any kernels only they would have \
             compiled are missing: {}",
            failures.join(", ")
        ));
    }
    log(&format!(
        "done: {harvested} kernel(s) in {}; commit them",
        graphblas::PREJIT_DIR
    ));
    Ok(())
}

type Step = (&'static str, &'static str, Vec<&'static str>);

/// Everything the harvest should exercise, in order.
fn workload() -> Vec<Step> {
    let mut steps = vec![
        (
            "cargo test",
            "cargo",
            vec![
                "test",
                "-p",
                "graph",
                "--release",
                "--features",
                "prejit_harvest",
            ],
        ),
        (
            "pytest e2e + functions",
            "pytest",
            vec!["tests/test_e2e.py", "tests/test_functions.py", "-vv"],
        ),
        (
            "pytest mvcc + concurrency",
            "pytest",
            vec!["tests/test_mvcc.py", "tests/test_concurrency.py", "-vv"],
        ),
        ("TCK", "pytest", vec!["tests/tck/test_tck.py", "-s"]),
        // Includes test_harmonic_centrality.py, whose tiny and 200-node graphs
        // are the only source of LAGraph's HLL dot4 kernels.
        ("flow tests", "./flow.sh", vec![]),
    ];
    // The benchmark queries carry shapes no functional suite issues.
    steps.push((
        "bench queries",
        "uv",
        vec!["run", "--project", "bench", "bench", "measure", "--once"],
    ));
    steps
}

/// A copy of the vendored kernels, taken before a harvest clears them.
struct Backup {
    root: std::path::PathBuf,
    dir: std::path::PathBuf,
}

impl Backup {
    fn take(root: &Path) -> Result<Self> {
        let dir = std::env::temp_dir().join(format!("native-deps-prejit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;
        for kernel in kernels(&root.join(graphblas::PREJIT_DIR))? {
            copy_file(&kernel, &dir.join(kernel.file_name().unwrap_or_default()))?;
        }
        Ok(Self {
            root: root.to_path_buf(),
            dir,
        })
    }

    fn restore(&self) -> Result<()> {
        let vendor = self.root.join(graphblas::PREJIT_DIR);
        for stale in kernels(&vendor)? {
            std::fs::remove_file(stale)?;
        }
        for kernel in kernels(&self.dir)? {
            copy_file(
                &kernel,
                &vendor.join(kernel.file_name().unwrap_or_default()),
            )?;
        }
        log(&format!(
            "restored the vendored kernels in {}",
            vendor.display()
        ));
        self.discard();
        Ok(())
    }

    fn discard(&self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn kernels(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    Ok(std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_prejit_kernel(p))
        .collect())
}

/// The environment every child runs with.
fn workload_env(root: &Path) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    // tests/common.py loads target/release when RELEASE=1, which is the build
    // the JIT is enabled in.
    env.insert("RELEASE".into(), "1".into());
    env.insert("TCK_DONE".into(), "tck_done.txt".into());
    // JIT-compiled kernels link their own libomp; next to the module's static
    // one, OpenMP aborts on the duplicate runtime unless told to tolerate it.
    env.insert("KMP_DUPLICATE_LIB_OK".into(), "TRUE".into());

    let venv = root.join("venv");
    if venv.is_dir() && env_opt("VIRTUAL_ENV").is_none() {
        let path = env_opt("PATH").unwrap_or_default();
        env.insert(
            "PATH".into(),
            format!("{}:{path}", venv.join("bin").display()),
        );
        env.insert("VIRTUAL_ENV".into(), venv.display().to_string());
    }
    if cfg!(target_os = "linux") {
        // rustdoc ignores the `[target.'cfg(linux)']` rustflags, so the doctests
        // need the duplicate-symbol allowance passed this way.
        let flags = env_opt("RUSTDOCFLAGS")
            .map(|f| format!("{f} "))
            .unwrap_or_default();
        env.insert(
            "RUSTDOCFLAGS".into(),
            format!("{flags}-C link-arg=-Wl,--allow-multiple-definition"),
        );
    }
    if cfg!(target_os = "macos")
        && env_opt("CC").is_none()
        && let Some(llvm) = capture_opt("brew", &["--prefix", "llvm"])
    {
        let llvm = llvm.trim().to_owned();
        env.insert("CC".into(), format!("{llvm}/bin/clang"));
        env.insert("CXX".into(), format!("{llvm}/bin/clang++"));
    }
    env
}

/// The JIT's compile line ends in `-fopenmp=libomp`, whose implicit `-lomp`
/// cannot find the toolchain image's static libomp outside the default search
/// path -- every JIT compile then fails, and each algorithm aborts at its first
/// uncached kernel.
fn link_static_libomp_for_jit() {
    let (archive, link) = (
        Path::new("/opt/libomp/lib/libomp.a"),
        Path::new("/usr/lib/libomp.a"),
    );
    if !cfg!(target_os = "linux") || !archive.is_file() || link.exists() {
        return;
    }
    #[cfg(unix)]
    if let Err(e) = std::os::unix::fs::symlink(archive, link) {
        log(&format!(
            "WARNING: cannot link {} -> {} ({e}); JIT compiles may fail with \
             `cannot find -lomp`",
            link.display(),
            archive.display()
        ));
    }
}

fn cargo(
    root: &Path,
    env: &BTreeMap<String, String>,
    args: &[&str],
) -> Result<()> {
    let args: Vec<&OsStr> = args.iter().map(OsStr::new).collect();
    run("cargo", &args, root, env)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::recipes::graphblas::{PREJIT_DIR, clear_prejit, harvest_prejit, version};
    use crate::testing::TempDir;

    #[test]
    fn version_reads_either_field_spelling() {
        let tmp = TempDir::new("gbver");
        let cmake = tmp.0.join("cmake_modules");
        fs::create_dir_all(&cmake).unwrap();
        for prefix in ["VER", "VERSION"] {
            fs::write(
                cmake.join("GraphBLAS_version.cmake"),
                format!(
                    "set ( GraphBLAS_{prefix}_MAJOR 10 CACHE STRING \"\" FORCE )\n\
                     set ( GraphBLAS_{prefix}_MINOR 5 CACHE STRING \"\" FORCE )\n\
                     set ( GraphBLAS_{prefix}_SUB   0 CACHE STRING \"\" FORCE )\n"
                ),
            )
            .unwrap();
            assert_eq!(version(&tmp.0).unwrap(), "10.5.0", "{prefix}");
        }
    }

    #[test]
    fn version_of_the_pinned_submodule() {
        // Names the JIT cache directory, so it has to match GraphBLAS's own
        // spelling: `GrB10.5.0`, not `GrB10_5_0`.
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../deps/GraphBLAS");
        if source.join("cmake_modules").is_dir() {
            let v = version(&source).unwrap();
            assert_eq!(v.split('.').count(), 3, "{v}");
            assert!(v.split('.').all(|p| p.parse::<u32>().is_ok()), "{v}");
        }
    }

    #[test]
    fn a_failed_run_restores_the_vendored_kernels() {
        let tmp = TempDir::new("prejit-backup");
        let vendor = tmp.0.join(PREJIT_DIR);
        fs::create_dir_all(&vendor).unwrap();
        fs::write(vendor.join("GB_jit__committed.c"), "committed").unwrap();
        fs::write(vendor.join("README"), "not a kernel").unwrap();

        let backup = super::Backup::take(&tmp.0).unwrap();
        // What a run that fails part-way leaves: cleared, plus a partial harvest.
        fs::remove_file(vendor.join("GB_jit__committed.c")).unwrap();
        fs::write(vendor.join("GB_jit__partial.c"), "partial").unwrap();
        backup.restore().unwrap();

        assert_eq!(
            fs::read_to_string(vendor.join("GB_jit__committed.c")).unwrap(),
            "committed"
        );
        assert!(!vendor.join("GB_jit__partial.c").exists());
        assert!(vendor.join("README").exists());
        assert!(!backup.dir.exists(), "the backup is removed once restored");
    }

    #[test]
    fn clear_then_harvest_replaces_the_vendored_set() {
        let tmp = TempDir::new("harvest");
        let (root, jit) = (tmp.0.join("repo"), tmp.0.join("GrB10.5.0"));
        let vendor = root.join(PREJIT_DIR);
        fs::create_dir_all(&vendor).unwrap();
        fs::write(vendor.join("GB_jit__stale.c"), "old").unwrap();
        fs::write(vendor.join("README"), "kept").unwrap();
        fs::create_dir_all(jit.join("c/ab")).unwrap();
        fs::write(jit.join("c/ab/GB_jit__stale_compile.c"), "previous run").unwrap();

        clear_prejit(&root, &jit).unwrap();
        assert!(!vendor.join("GB_jit__stale.c").exists());
        assert!(vendor.join("README").exists(), "only kernels are cleared");
        assert!(
            !jit.join("c").exists(),
            "a previous run's output must not be harvested"
        );
        assert!(
            harvest_prejit(&root, &jit).is_err(),
            "nothing compiled is an error"
        );

        // What a harvest run's JIT leaves behind, in GraphBLAS's hashed subdirs.
        fs::create_dir_all(jit.join("c/12")).unwrap();
        fs::write(jit.join("c/12/GB_jit__AxB_dot2__new.c"), "kernel").unwrap();
        fs::write(jit.join("c/12/GB_jit__AxB_dot2__new.h"), "not a kernel").unwrap();
        assert_eq!(harvest_prejit(&root, &jit).unwrap(), 1);
        assert_eq!(
            fs::read_to_string(vendor.join("GB_jit__AxB_dot2__new.c")).unwrap(),
            "kernel"
        );
    }
}
