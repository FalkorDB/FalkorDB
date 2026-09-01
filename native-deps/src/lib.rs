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
pub mod key;
pub mod lock;
pub mod recipes;
/// Minimal vendored SHA-256 -- see the file header for why it is hand-rolled.
pub mod sha256;
pub mod toolchain;
pub mod util;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub use crate::cache::{Cache, Stamp};
pub use crate::dep::Dep;
pub use crate::error::{Error, Result};
pub use crate::lock::LockFile;
pub use crate::toolchain::Toolchain;

use crate::cache::BuildLock;
use crate::key::KeyContext;
use crate::recipes::Ctx;
use crate::util::{env_flag, env_opt, find_repo_root, log, now_secs};

/// What to resolve, and how.
#[derive(Debug, Clone)]
pub struct Request {
    /// FalkorDB checkout root (the directory holding `deps/native-deps.lock`).
    pub root: PathBuf,
    pub deps: Vec<Dep>,
    /// Sanitizer flavor for RediSearch, e.g. `Some("address")`.
    pub san: Option<String>,
    /// Build a GraphBLAS with no PreJIT kernels, for `gen_prejit.sh`.
    pub prejit_harvest: bool,
    /// Rebuild even on a cache hit.
    pub force: bool,
    /// Turn a cache miss into an actionable error instead of a build. Set in
    /// the runtime Docker images, where a miss means the prebuilt artifacts
    /// don't match the sources and we want to hear about it loudly.
    pub offline: bool,
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
            offline: env_flag("FALKORDB_NATIVE_DEPS_OFFLINE"),
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
            "FALKORDB_NATIVE_DEPS_OFFLINE",
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
///    verbatim, no key check.
/// 2. A read-only prebuilt root from `FALKORDB_NATIVE_DEPS_PREBUILT` holding an
///    entry for this exact key (how the Docker images ship prebuilt deps).
/// 3. The writable cache.
/// 4. Build it -- unless `offline`, in which case the miss is reported with the
///    manifest so the mismatching input is obvious.
pub fn ensure(req: &Request) -> Result<Resolution> {
    let lock = LockFile::load(&req.root)?;
    let toolchain = Toolchain::detect()?;
    let cache = Cache::discover()?;

    let kctx = KeyContext {
        root: &req.root,
        lock: &lock,
        toolchain: &toolchain,
        san: req.san.as_deref(),
        prejit_harvest: req.prejit_harvest,
    };
    let ctx = Ctx {
        root: &req.root,
        lock: &lock,
        toolchain: &toolchain,
        cache: &cache,
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
        let graphblas_key = out.0.get(&Dep::GraphBlas).map(|r| r.key.clone());
        let resolved = resolve_one(
            req,
            &ctx,
            &kctx,
            &cache,
            dep,
            graphblas_key.as_deref(),
            &out,
        )?;
        out.0.insert(dep, resolved);
    }
    Ok(out)
}

/// Just the keys, without building anything. Used by `native-deps key` to feed
/// CI cache keys.
pub fn keys(req: &Request) -> Result<BTreeMap<Dep, String>> {
    let lock = LockFile::load(&req.root)?;
    let toolchain = Toolchain::detect()?;
    let kctx = KeyContext {
        root: &req.root,
        lock: &lock,
        toolchain: &toolchain,
        san: req.san.as_deref(),
        prejit_harvest: req.prejit_harvest,
    };

    let mut out = BTreeMap::new();
    let mut graphblas_key = None;
    for dep in Dep::ALL {
        let key = kctx.manifest(dep, graphblas_key.as_deref())?.key();
        if dep == Dep::GraphBlas {
            graphblas_key = Some(key.clone());
        }
        if req.deps.contains(&dep) {
            out.insert(dep, key);
        }
    }
    Ok(out)
}

fn resolve_one(
    req: &Request,
    ctx: &Ctx<'_>,
    kctx: &KeyContext<'_>,
    cache: &Cache,
    dep: Dep,
    graphblas_key: Option<&str>,
    resolved_so_far: &Resolution,
) -> Result<Resolved> {
    if let Some(prefix) = override_prefix(dep) {
        log(&format!(
            "{dep}: using override prefix {}",
            prefix.display()
        ));
        let key = Stamp::read(&prefix).map_or_else(|_| "override".to_owned(), |s| s.key);
        return Ok(Resolved { dep, key, prefix });
    }

    let manifest = kctx.manifest(dep, graphblas_key)?;
    let key = manifest.key();

    if !req.force
        && let Some(prefix) = cache.lookup(dep, &key)
    {
        log(&format!("{dep}: cache hit {}", prefix.display()));
        return Ok(Resolved { dep, key, prefix });
    }

    if req.offline {
        return Err(err!(
            "{dep} is not prebuilt for key {key} and FALKORDB_NATIVE_DEPS_OFFLINE is set.\n\
             manifest:\n{}\navailable entries:\n  {}\n\
             hint: run `native-deps build {dep}` to produce it",
            indent(&manifest.render()),
            if cache.available(dep).is_empty() {
                "(none)".to_owned()
            } else {
                cache.available(dep).join("\n  ")
            }
        ));
    }

    let entry = cache.entry_dir(dep, &key);
    let _guard = BuildLock::acquire(cache.root.join(dep.name()).join(format!("{key}.lock")))?;

    // Double-check: whoever held the lock has very likely just published it.
    if !req.force
        && let Some(prefix) = cache.lookup(dep, &key)
    {
        log(&format!("{dep}: cache hit after wait {}", prefix.display()));
        return Ok(Resolved { dep, key, prefix });
    }

    log(&format!("{dep}: building into {}", entry.display()));
    match dep {
        Dep::GraphBlas => recipes::graphblas::build(ctx, &entry)?,
        Dep::LaGraph => {
            let gb = resolved_so_far.get(Dep::GraphBlas)?;
            recipes::lagraph::build(ctx, &entry, &gb.prefix)?;
        }
        Dep::RediSearch => recipes::redisearch::build(ctx, &entry)?,
    }

    // Written last: its presence is what marks the entry complete, so an
    // interrupted build is never adopted.
    Stamp {
        key: key.clone(),
        dep: dep.name().to_owned(),
        built_at: now_secs(),
        manifest: manifest.render(),
    }
    .write(&entry)?;

    log(&format!("{dep}: built {key}"));
    Ok(Resolved {
        dep,
        key,
        prefix: entry,
    })
}

fn override_prefix(dep: Dep) -> Option<PathBuf> {
    let var = match dep {
        Dep::GraphBlas => "GRAPHBLAS_PREFIX",
        Dep::LaGraph => "LAGRAPH_PREFIX",
        Dep::RediSearch => "REDISEARCH_PREFIX",
    };
    env_opt(var).map(PathBuf::from)
}

fn indent(text: &str) -> String {
    text.lines()
        .map(|l| format!("  {l}"))
        .collect::<Vec<_>>()
        .join("\n")
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
