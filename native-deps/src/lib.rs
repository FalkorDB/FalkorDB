//! Builds and caches FalkorDB's native dependencies.
//!
//! This crate replaces the old `graphblas.sh` and `redisearch.sh`. It is both a
//! library and a binary over the same code, which is what lets one
//! implementation serve every consumer:
//!
//! * `graph/build.rs` takes it as a path build-dependency and calls [`ensure`]
//!   directly -- a plain function call, so nothing inherits `RUSTFLAGS` /
//!   `CARGO_ENCODED_RUSTFLAGS` / `CARGO_BUILD_TARGET` the way a nested
//!   `cargo run` would under the sanitizer build.
//! * The Docker dep stages run the `native-deps` binary, so each stage's inputs
//!   are exactly its own dependency.
//! * Developers run `native-deps ensure`, or nothing at all -- `cargo build`
//!   triggers it.
//!
//! See [`key`] for how artifacts are addressed and [`cache`] for where they live.

pub mod cache;
pub mod dep;
pub mod error;
/// Minimal vendored SHA-256 -- see the file header for why it is hand-rolled.
pub mod hash;
pub mod key;
pub mod local;
pub mod lock;
pub mod prejit;
pub mod prune;
pub mod recipes;
#[cfg(test)]
mod testing;
pub mod toolchain;
pub mod util;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub use crate::cache::{Cache, Stamp};
pub use crate::dep::Dep;
pub use crate::error::{Error, Result};
pub use crate::lock::LockFile;
pub use crate::toolchain::Toolchain;

use crate::cache::BuildLock;
use crate::recipes::{Ctx, SourceGuard};
use crate::util::{env_flag, env_opt, find_repo_root, log, now_secs};

/// What to resolve, and how.
#[derive(Debug, Clone)]
pub struct Request {
    /// FalkorDB checkout root (the directory holding `deps/native-deps.lock`).
    pub root: PathBuf,
    pub deps: Vec<Dep>,
    /// Sanitizer flavor for RediSearch, e.g. `Some("address")`.
    pub san: Option<String>,
    /// Build a GraphBLAS with no PreJIT kernels, for `native-deps prejit`.
    pub prejit_harvest: bool,
    /// Rebuild even on a cache hit.
    pub force: bool,
}

impl Request {
    /// All three deps, with flavor and mode taken from the environment.
    pub fn from_env() -> Result<Self> {
        let cwd = std::env::current_dir()?;
        Ok(Self {
            root: find_repo_root(&cwd)?,
            deps: Dep::ALL.to_vec(),
            san: env_opt("REDISEARCH_SAN"),
            prejit_harvest: env_flag("FALKORDB_PREJIT_HARVEST"),
            force: env_flag("FALKORDB_NATIVE_DEPS_FORCE"),
        })
    }

    /// The env vars that change what [`ensure`] returns. `graph/build.rs` emits
    /// a `cargo:rerun-if-env-changed` for each.
    #[must_use]
    pub const fn env_inputs() -> &'static [&'static str] {
        &[
            "CC",
            "CXX",
            "FALKORDB_DEPS_CACHE",
            "FALKORDB_NATIVE_DEPS_FORCE",
            "FALKORDB_NATIVE_DEPS_PREBUILT",
            "FALKORDB_PREJIT_HARVEST",
            "FALKORDB_REPO_ROOT",
            "GRAPHBLAS_PREFIX",
            "LAGRAPH_PREFIX",
            "LIBOMP_PREFIX",
            "REDISEARCH_PREFIX",
            "REDISEARCH_SAN",
        ]
    }
}

/// Where a dependency's artifacts ended up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    pub dep: Dep,
    pub key: String,
    /// Directory holding `include/` and `lib/`.
    pub prefix: PathBuf,
    /// Set when this was built from a checkout that differs from its pin: the
    /// source directory, so `graph/build.rs` can rebuild when it is edited.
    pub local_source: Option<PathBuf>,
}

impl Resolved {
    #[must_use]
    pub fn include(&self) -> PathBuf {
        self.prefix.join("include")
    }

    #[must_use]
    pub fn lib(&self) -> PathBuf {
        self.prefix.join("lib")
    }

    /// RediSearch only: the Rust archive, kept out of `lib/` because
    /// `graph/build.rs` links a linkme-stripped copy of it rather than the
    /// original.
    #[must_use]
    pub fn redisearch_rs_archive(&self) -> PathBuf {
        self.prefix.join("rs/libredisearch_rs.a")
    }

    #[must_use]
    pub fn stamp_path(&self) -> PathBuf {
        self.prefix.join(cache::STAMP_NAME)
    }
}

/// The outcome of [`ensure`].
#[derive(Debug, Clone, Default)]
pub struct Resolution(BTreeMap<Dep, Resolved>);

impl Resolution {
    pub fn get(
        &self,
        dep: Dep,
    ) -> Result<&Resolved> {
        self.0
            .get(&dep)
            .ok_or_else(|| err!("{dep} was not resolved"))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Dep, &Resolved)> {
        self.0.iter()
    }
}

/// Resolve every requested dep, building whatever is missing.
///
/// Order of preference, per dep:
///
/// 1. `GRAPHBLAS_PREFIX` / `LAGRAPH_PREFIX` / `REDISEARCH_PREFIX` -- used
///    verbatim, BYPASSING the key. This is an explicit escape hatch for
///    supplying a hand-built or distro-packaged artifact; because it skips the
///    key it cannot detect an ABI mismatch, so taking it logs a warning. It is
///    opt-in per dep and nothing in this repo sets these.
/// 2. A read-only prebuilt root from `FALKORDB_NATIVE_DEPS_PREBUILT` holding an
///    entry for this exact key (how the Docker images ship prebuilt deps).
/// 3. The writable cache.
/// 4. Build it.
pub fn ensure(req: &Request) -> Result<Resolution> {
    let lock = LockFile::load(&req.root)?;
    let toolchain = Toolchain::detect()?;
    let cache = Cache::discover()?;

    let ctx = Ctx {
        root: &req.root,
        lock: &lock,
        toolchain: &toolchain,
        san: req.san.as_deref(),
        prejit_harvest: req.prejit_harvest,
    };

    // LAGraph links the GraphBLAS archive, so it is always resolved alongside
    // it and after it.
    let mut wanted = req.deps.clone();
    if wanted.contains(&Dep::LaGraph) && !wanted.contains(&Dep::GraphBlas) {
        wanted.push(Dep::GraphBlas);
    }

    let mut out = Resolution::default();
    for dep in Dep::ALL {
        if !wanted.contains(&dep) {
            continue;
        }
        let resolved = resolve_one(req, &ctx, &cache, dep, &out)?;
        // Only our own entries: an override prefix is someone else's directory.
        if resolved.prefix.starts_with(&cache.root) {
            cache::mark_used(&resolved.prefix);
        }
        out.0.insert(dep, resolved);
    }
    Ok(out)
}

fn resolve_one(
    req: &Request,
    ctx: &Ctx<'_>,
    cache: &Cache,
    dep: Dep,
    resolved_so_far: &Resolution,
) -> Result<Resolved> {
    // LAGraph's key includes GraphBLAS's, and the loop in `ensure` always
    // resolves GraphBLAS first, so read it back rather than threading it
    // through as a second, redundant parameter.
    let graphblas_key = resolved_so_far
        .0
        .get(&Dep::GraphBlas)
        .map(|r| r.key.as_str());
    if let Some(prefix) = override_prefix(dep) {
        log(&format!(
            "{dep}: using override prefix {}",
            prefix.display()
        ));
        let key = Stamp::read(&prefix).map_or_else(|_| "override".to_owned(), |s| s.key);
        return Ok(Resolved {
            dep,
            key,
            prefix,
            local_source: None,
        });
    }

    let manifest = ctx.manifest(dep, graphblas_key)?;
    let key = manifest.key();

    // The common case takes no lock: a checkout on its pin, already published.
    if !req.force
        && ctx.local_changes(dep.name())?.is_none()
        && let Some(prefix) = cache.lookup(dep, &key)
    {
        log(&format!("{dep}: cache hit {}", prefix.display()));
        return Ok(Resolved {
            dep,
            key,
            prefix,
            local_source: None,
        });
    }

    // Past here we may build in place. The checkout may also be mid-way through
    // another process's build -- an IDE's `cargo check` with a different key,
    // say -- whose patches look exactly like a local edit, or carry the debris
    // of a build that was killed. So serialise with everyone using this
    // checkout, undo what a killed build left, and only then believe it.
    let source = ctx.lock.source_dir(&req.root, dep.name())?;
    let state = ctx.source_state(dep.name())?;
    let _source_lock = BuildLock::acquire(state.join("lock"))?;
    let undone = SourceGuard::recover(&state.join("journal"))?;
    if undone > 0 {
        log(&format!(
            "{dep}: undid {undone} change(s) an interrupted build left in {}",
            source.display()
        ));
    }
    if let Some(local) = ctx.local_changes(dep.name())? {
        return resolve_local(req, ctx, cache, dep, manifest, &local, resolved_so_far);
    }

    if !req.force
        && let Some(prefix) = cache.lookup(dep, &key)
    {
        log(&format!("{dep}: cache hit {}", prefix.display()));
        return Ok(Resolved {
            dep,
            key,
            prefix,
            local_source: None,
        });
    }

    let entry = cache.entry_dir(dep, &key);
    let _guard = BuildLock::acquire(cache.root.join(dep.name()).join(format!("{key}.lock")))?;

    // Double-check: whoever held the lock has very likely just published it.
    if !req.force
        && let Some(prefix) = cache.lookup(dep, &key)
    {
        log(&format!("{dep}: cache hit after wait {}", prefix.display()));
        return Ok(Resolved {
            dep,
            key,
            prefix,
            local_source: None,
        });
    }

    build_entry(ctx, dep, &entry, &key, &manifest, resolved_so_far)?;
    Ok(Resolved {
        dep,
        key,
        prefix: entry,
        local_source: None,
    })
}

/// Entry names of local builds start with this; keys never do, being hex.
pub const LOCAL_PREFIX: &str = "local-";

/// The key of a local build of `manifest`.
fn local_key(
    mut manifest: key::Manifest,
    local: &local::LocalChanges,
) -> String {
    manifest.set("local", &local.fingerprint);
    format!("{LOCAL_PREFIX}{}", manifest.key())
}

/// The one local-build slot per dep that belongs to the worktree at `root`.
fn local_entry_name(root: &Path) -> String {
    let worktree = &hash::sha256_hex(root.to_string_lossy().as_bytes())[..12];
    format!("{LOCAL_PREFIX}{worktree}")
}

/// The cache entry names this checkout resolves to right now, computed without
/// building anything: its pinned keys, and its own local-build slots. `prune`
/// never removes these.
pub fn current_entries(req: &Request) -> Result<BTreeSet<String>> {
    let lock = LockFile::load(&req.root)?;
    let toolchain = Toolchain::detect()?;
    let ctx = Ctx {
        root: &req.root,
        lock: &lock,
        toolchain: &toolchain,
        san: req.san.as_deref(),
        prejit_harvest: req.prejit_harvest,
    };
    let mut out = BTreeSet::from([local_entry_name(&req.root)]);
    // LAGraph's key embeds GraphBLAS's as resolved -- a local key when the
    // GraphBLAS checkout is off its pin -- exactly as in `ensure`.
    let pinned = ctx.manifest(Dep::GraphBlas, None)?;
    let graphblas = match ctx.local_changes(Dep::GraphBlas.name())? {
        Some(local) => local_key(pinned, &local),
        None => pinned.key(),
    };
    out.insert(ctx.manifest(Dep::LaGraph, Some(&graphblas))?.key());
    out.insert(ctx.manifest(Dep::RediSearch, None)?.key());
    out.insert(graphblas);
    Ok(out)
}

/// Resolve a dep whose checkout differs from its pin.
///
/// It is built into one slot per worktree, `<dep>/local-<worktree>`, which no
/// key lookup can ever find -- so it is never handed to another worktree, nor
/// to this one once its checkout is back on the pin. The slot is reused while
/// the local state is unchanged: `graph/build.rs` reruns whenever the stamp
/// moves, so rebuilding unconditionally would rebuild on every `cargo build`.
fn resolve_local(
    req: &Request,
    ctx: &Ctx<'_>,
    cache: &Cache,
    dep: Dep,
    mut manifest: key::Manifest,
    local: &local::LocalChanges,
    resolved_so_far: &Resolution,
) -> Result<Resolved> {
    let source = ctx.lock.source_dir(&req.root, dep.name())?;
    log(&format!(
        "WARNING: {} {}; using a build for this worktree only, outside the shared cache",
        source.display(),
        local.reason
    ));

    let key = local_key(manifest.clone(), local);
    manifest.set("local", &local.fingerprint);
    let entry = cache.entry_dir(dep, &local_entry_name(&req.root));

    if !req.force && Stamp::read(&entry).is_ok_and(|s| s.key == key) {
        log(&format!("{dep}: local build unchanged {}", entry.display()));
    } else {
        let _guard = BuildLock::acquire(entry.with_extension("lock"))?;
        build_entry(ctx, dep, &entry, &key, &manifest, resolved_so_far)?;
    }
    Ok(Resolved {
        dep,
        key,
        prefix: entry,
        local_source: Some(source),
    })
}

/// Build `dep` into `entry` and stamp it.
fn build_entry(
    ctx: &Ctx<'_>,
    dep: Dep,
    entry: &Path,
    key: &str,
    manifest: &key::Manifest,
    resolved_so_far: &Resolution,
) -> Result<()> {
    log(&format!("{dep}: building into {}", entry.display()));
    match dep {
        Dep::GraphBlas => recipes::graphblas::build(ctx, entry)?,
        Dep::LaGraph => {
            let gb = resolved_so_far.get(Dep::GraphBlas)?;
            recipes::lagraph::build(ctx, entry, &gb.prefix)?;
        }
        Dep::RediSearch => recipes::redisearch::build(ctx, entry)?,
    }

    // Written last: its presence is what marks the entry complete, so an
    // interrupted build is never adopted.
    Stamp {
        key: key.to_owned(),
        dep: dep.name().to_owned(),
        built_at: now_secs(),
        manifest: manifest.render(),
    }
    .write(entry)?;

    log(&format!("{dep}: built {key}"));
    Ok(())
}

fn override_prefix(dep: Dep) -> Option<PathBuf> {
    let var = match dep {
        Dep::GraphBlas => "GRAPHBLAS_PREFIX",
        Dep::LaGraph => "LAGRAPH_PREFIX",
        Dep::RediSearch => "REDISEARCH_PREFIX",
    };
    let prefix = env_opt(var).map(PathBuf::from)?;
    // Taking this path skips the cache key entirely, so nothing here can tell
    // whether these artifacts were built by the compiler that is about to link
    // them. Say so out loud: a silent stale-ABI reuse is precisely what the key
    // exists to prevent, and an escape hatch that is quiet is indistinguishable
    // from the bug.
    log(&format!(
        "WARNING: {var} is set, using {} verbatim -- the cache key is NOT \
         checked, so an ABI mismatch with $CC/$CXX will not be detected",
        prefix.display()
    ));
    Some(prefix)
}
/// Paths whose contents feed the cache key, for `cargo:rerun-if-changed`.
#[must_use]
pub fn watch_paths(root: &Path) -> Vec<PathBuf> {
    vec![
        root.join(lock::LOCK_RELPATH),
        root.join("build/graphblas/GB_control.patch"),
        root.join("build/graphblas/PreJIT"),
    ]
}
