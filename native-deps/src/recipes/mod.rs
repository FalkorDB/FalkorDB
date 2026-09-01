//! Per-dependency build recipes -- the direct port of `graphblas.sh` and
//! `redisearch.sh`.

pub mod graphblas;
pub mod lagraph;
pub mod redisearch;

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};

use crate::cache::Cache;
use crate::err;
use crate::error::Result;
use crate::lock::LockFile;
use crate::toolchain::Toolchain;
use crate::util::{env_flag, jobs, log, run};

/// Everything a recipe needs to build one dep.
pub struct Ctx<'a> {
    pub root: &'a Path,
    pub lock: &'a LockFile,
    pub toolchain: &'a Toolchain,
    pub cache: &'a Cache,
    /// Sanitizer flavor, e.g. `Some("address")`.
    pub san: Option<&'a str>,
    pub prejit_harvest: bool,
}

impl Ctx<'_> {
    /// Absolute path to a dep's submodule worktree, with an actionable error
    /// when the submodule has not been checked out.
    pub fn source(
        &self,
        name: &str,
    ) -> Result<PathBuf> {
        let dir = self.lock.source_dir(self.root, name)?;
        // An uninitialised submodule is an empty directory, so test for content
        // rather than existence.
        let populated = fs::read_dir(&dir)
            .ok()
            .is_some_and(|mut d| d.next().is_some());
        if !populated {
            return Err(err!(
                "{} is empty - run `git submodule update --init --recursive`",
                dir.display()
            ));
        }
        Ok(dir)
    }
}

/// Restores source files that a build mutates in place.
///
/// GraphBLAS needs `GB_control.patch` applied and PreJIT kernels copied in;
/// RediSearch needs `-Werror` stripped from VectorSimilarity. Snapshotting the
/// original bytes and writing them back on drop makes those edits idempotent and
/// keeps `git status` clean -- and unlike `git checkout --`, it works in a
/// Docker context where the submodule has no `.git`.
#[derive(Default)]
pub struct SourceGuard {
    /// `None` means the file did not exist and should be deleted on restore.
    saved: Vec<(PathBuf, Option<Vec<u8>>)>,
    /// Directories we created and must remove again on restore.
    created_dirs: Vec<PathBuf>,
}

impl SourceGuard {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Remember `path`'s current state so it can be put back later. Recording
    /// the same path twice keeps the first (i.e. pristine) snapshot.
    pub fn snapshot(
        &mut self,
        path: &Path,
    ) -> Result<()> {
        if self.saved.iter().any(|(p, _)| p == path) {
            return Ok(());
        }
        let content = match fs::read(path) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(err!("cannot snapshot {}: {e}", path.display())),
        };
        self.saved.push((path.to_path_buf(), content));
        Ok(())
    }

    /// Create an empty directory at `path`, removing it again on restore. Does
    /// nothing if something is already there.
    pub fn create_dir(
        &mut self,
        path: &Path,
    ) -> Result<()> {
        if path.exists() {
            return Ok(());
        }
        fs::create_dir(path).map_err(|e| err!("cannot create {}: {e}", path.display()))?;
        self.created_dirs.push(path.to_path_buf());
        Ok(())
    }
}

impl Drop for SourceGuard {
    fn drop(&mut self) {
        for path in self.created_dirs.drain(..).rev() {
            if let Err(e) = fs::remove_dir(&path) {
                log(&format!(
                    "WARNING: could not remove {}: {e}",
                    path.display()
                ));
            }
        }
        for (path, content) in self.saved.drain(..).rev() {
            match content {
                Some(bytes) => {
                    if let Err(e) = fs::write(&path, bytes) {
                        log(&format!(
                            "WARNING: could not restore {}: {e}",
                            path.display()
                        ));
                    }
                }
                None => {
                    let _ = fs::remove_file(&path);
                }
            }
        }
    }
}

/// Give the source tree a `.git` that git — and build scripts that look for a
/// "git root" — can live with, for the duration of the build.
///
/// A submodule's `.git` is a *file* reading `gitdir: <path into the superproject>`.
/// A Docker build context carries neither the superproject's `.git` nor (for a
/// worktree-backed checkout) the path that file names, so the link dangles. A
/// dangling gitlink is strictly worse than no `.git` at all: git aborts with
/// "fatal: not a git repository: <target>" instead of walking past it, which is
/// what broke `git apply` for the GraphBLAS patch.
///
/// Deleting it outright isn't enough either. RediSearch's `ffi/build.rs` locates
/// its include paths by walking up until it finds a path where `.git` *exists*,
/// and panics with "Could not find git root" if it never does. `redisearch.sh`
/// happened to satisfy that by `git init`-ing its own clone; a submodule in a
/// `.git`-less context does not.
///
/// An empty `.git` **directory** satisfies both: the existence check passes, and
/// git treats an invalid gitdir as "not a repository here" and keeps walking,
/// rather than hard-failing. `SourceGuard` removes it again afterwards.
///
/// A healthy `.git` — a real clone, or a gitlink that resolves — is left alone.
pub fn ensure_git_root(
    source: &Path,
    guard: &mut SourceGuard,
) -> Result<()> {
    let link = source.join(".git");
    if link.is_dir() || gitlink_resolves(&link) {
        return Ok(());
    }
    if link.is_file() {
        guard.snapshot(&link)?;
        fs::remove_file(&link).map_err(|e| err!("cannot move {} aside: {e}", link.display()))?;
        log(&format!(
            "{} points outside the build context; replacing it with an empty marker",
            link.display()
        ));
    }
    guard.create_dir(&link)
}

/// True if `link` is a gitlink file whose `gitdir:` target actually exists.
fn gitlink_resolves(link: &Path) -> bool {
    let Ok(text) = fs::read_to_string(link) else {
        return false;
    };
    let Some(target) = text.trim().strip_prefix("gitdir:") else {
        return false;
    };
    let target = target.trim();
    if Path::new(target).is_absolute() {
        PathBuf::from(target).exists()
    } else {
        link.parent().is_some_and(|p| p.join(target).exists())
    }
}

/// Configure, build and optionally install a cmake project.
pub struct CMake {
    source: PathBuf,
    build: PathBuf,
    args: Vec<String>,
}

impl CMake {
    pub fn new(
        source: &Path,
        build: &Path,
    ) -> Self {
        Self {
            source: source.to_path_buf(),
            build: build.to_path_buf(),
            args: Vec::new(),
        }
    }

    #[must_use]
    pub fn arg(
        mut self,
        arg: impl Into<String>,
    ) -> Self {
        self.args.push(arg.into());
        self
    }

    #[must_use]
    pub fn args<I, S>(
        mut self,
        args: I,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    pub fn configure(&self) -> Result<()> {
        fs::create_dir_all(&self.build)?;
        let mut argv: Vec<OsString> = vec![
            "-S".into(),
            self.source.clone().into(),
            "-B".into(),
            self.build.clone().into(),
        ];
        argv.extend(self.args.iter().map(OsString::from));
        let refs: Vec<&OsStr> = argv.iter().map(AsRef::as_ref).collect();
        run("cmake", &refs, &self.source, &BTreeMap::new())
    }

    pub fn build(&self) -> Result<()> {
        let jobs = jobs().to_string();
        let argv: Vec<OsString> = vec![
            "--build".into(),
            self.build.clone().into(),
            "-j".into(),
            jobs.into(),
        ];
        let refs: Vec<&OsStr> = argv.iter().map(AsRef::as_ref).collect();
        run("cmake", &refs, &self.source, &BTreeMap::new())
    }

    pub fn install(&self) -> Result<()> {
        let argv: Vec<OsString> = vec!["--install".into(), self.build.clone().into()];
        let refs: Vec<&OsStr> = argv.iter().map(AsRef::as_ref).collect();
        run("cmake", &refs, &self.source, &BTreeMap::new())
    }

    /// Configure then build.
    pub fn compile(&self) -> Result<()> {
        self.configure()?;
        self.build()
    }

    /// Configure, build, install.
    pub fn pipeline(&self) -> Result<()> {
        self.compile()?;
        self.install()
    }
}

/// Prepare `<entry>` for a fresh build: wipe whatever a previous interrupted
/// attempt left behind (there is no `.stamp`, so nothing there is usable) and
/// return the scratch build directory to use.
pub fn prepare_entry(entry: &Path) -> Result<PathBuf> {
    if entry.exists() {
        fs::remove_dir_all(entry).map_err(|e| err!("cannot clear {}: {e}", entry.display()))?;
    }
    let build = entry.join(".build");
    fs::create_dir_all(&build)?;
    Ok(build)
}

/// Drop the scratch build tree once the artifacts are installed. GraphBLAS's is
/// roughly a gigabyte, and the cache never evicts.
pub fn cleanup_build_dir(build: &Path) {
    if env_flag("FALKORDB_NATIVE_DEPS_KEEP_BUILD") {
        log(&format!("keeping build tree at {}", build.display()));
        return;
    }
    let _ = fs::remove_dir_all(build);
}
