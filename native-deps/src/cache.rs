//! The content-addressed artifact cache.
//!
//! Layout:
//!
//! ```text
//! $HOME/.cache/falkordb/native-deps/       # override: $FALKORDB_DEPS_CACHE
//!   graphblas/4f9c2a71e0d83b56/
//!     .stamp                               # key + the manifest that produced it
//!     include/...  lib/libgraphblas.a
//!   redisearch/2d77ba95c1e6408f/
//!     ...
//! ```
//!
//! `.stamp` is written **last**, so a Ctrl-C'd or OOM-killed build leaves a
//! directory that is never mistaken for a complete one -- the next run simply
//! wipes it and starts over. An `O_EXCL` lock file keeps two worktrees from
//! building the same dep at the same time; the loser waits and then gets a hit.

use std::fs::{self, TryLockError};
use std::path::{Path, PathBuf};

use crate::dep::Dep;
use crate::error::Result;
use crate::util::{env_opt, log};
use crate::{bail, err};

pub const STAMP_NAME: &str = ".stamp";
/// Touched whenever an entry is resolved, so `prune` can tell what is in use.
/// Not the stamp itself: graph/build.rs watches the stamp, and touching it on
/// every resolve would rerun the build script -- and recompile `graph` -- on
/// every `cargo build`.
pub const USED_NAME: &str = ".used";
const MANIFEST_SEPARATOR: &str = "--- manifest ---";

/// Where artifacts are looked up and written.
#[derive(Debug, Clone)]
pub struct Cache {
    /// The single writable root.
    pub root: PathBuf,
    /// Read-only roots searched first, from `FALKORDB_NATIVE_DEPS_PREBUILT`
    /// (colon-separated). This is how a Docker image ships prebuilt deps: the
    /// key match is structural, so an image whose baked artifacts no longer
    /// match the sources simply misses instead of silently linking stale code.
    pub prebuilt: Vec<PathBuf>,
}

impl Cache {
    pub fn discover() -> Result<Self> {
        let root = if let Some(explicit) = env_opt("FALKORDB_DEPS_CACHE") {
            PathBuf::from(explicit)
        } else if let Some(xdg) = env_opt("XDG_CACHE_HOME") {
            PathBuf::from(xdg).join("falkordb/native-deps")
        } else if let Some(home) = env_opt("HOME") {
            PathBuf::from(home).join(".cache/falkordb/native-deps")
        } else {
            bail!(
                "cannot determine a cache directory: set FALKORDB_DEPS_CACHE, \
                 XDG_CACHE_HOME or HOME"
            );
        };

        let prebuilt = env_opt("FALKORDB_NATIVE_DEPS_PREBUILT")
            .map(|v| {
                v.split(':')
                    .filter(|s| !s.is_empty())
                    .map(PathBuf::from)
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self { root, prebuilt })
    }

    /// Where a build for `key` writes.
    #[must_use]
    pub fn entry_dir(
        &self,
        dep: Dep,
        key: &str,
    ) -> PathBuf {
        self.root.join(dep.name()).join(key)
    }

    /// First complete entry for `key` across the prebuilt roots and then the
    /// writable root.
    #[must_use]
    pub fn lookup(
        &self,
        dep: Dep,
        key: &str,
    ) -> Option<PathBuf> {
        self.prebuilt
            .iter()
            .map(|r| r.join(dep.name()).join(key))
            .chain(std::iter::once(self.entry_dir(dep, key)))
            .find(|d| d.join(STAMP_NAME).is_file())
    }
}

/// Record that `entry` was just resolved. Best effort: a read-only prebuilt root
/// simply cannot record it, and is never pruned anyway.
pub fn mark_used(entry: &Path) {
    let _ = fs::write(entry.join(USED_NAME), b"");
}

/// The completion marker for a cache entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stamp {
    pub key: String,
    pub dep: String,
    pub built_at: u64,
    /// The manifest that produced `key`, verbatim, so a surprising miss is
    /// diffable.
    pub manifest: String,
}

impl Stamp {
    #[must_use]
    pub fn render(&self) -> String {
        format!(
            "# native-deps stamp -- written last; its presence means this entry is complete.\n\
             key = {}\ndep = {}\nbuilt_at = {}\n\n{MANIFEST_SEPARATOR}\n{}",
            self.key, self.dep, self.built_at, self.manifest
        )
    }

    pub fn write(
        &self,
        dir: &Path,
    ) -> Result<()> {
        fs::create_dir_all(dir)?;
        fs::write(dir.join(STAMP_NAME), self.render())
            .map_err(|e| err!("cannot write {}/{STAMP_NAME}: {e}", dir.display()))?;
        Ok(())
    }

    pub fn read(dir: &Path) -> Result<Self> {
        let path = dir.join(STAMP_NAME);
        let text =
            fs::read_to_string(&path).map_err(|e| err!("cannot read {}: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| err!("{}: {e}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self> {
        let (head, manifest) = text
            .split_once(MANIFEST_SEPARATOR)
            .ok_or_else(|| err!("missing `{MANIFEST_SEPARATOR}` marker"))?;

        let mut key = String::new();
        let mut dep = String::new();
        let mut built_at = 0u64;
        for line in head.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            match k.trim() {
                "key" => key = v.trim().to_owned(),
                "dep" => dep = v.trim().to_owned(),
                "built_at" => built_at = v.trim().parse().unwrap_or(0),
                _ => {}
            }
        }
        if key.is_empty() || dep.is_empty() {
            bail!("stamp is missing `key` or `dep`");
        }
        Ok(Self {
            key,
            dep,
            built_at,
            manifest: manifest.trim_start_matches('\n').to_owned(),
        })
    }
}

/// An exclusive build lock: a kernel advisory lock (`flock`) on a lock file.
///
/// The kernel drops it when the holder exits, however it exits, so a killed
/// build can never leave a lock behind and nothing has to guess whether a
/// holder is still alive. The lock file itself is never deleted: unlinking a
/// path another process may be about to lock would let two processes hold
/// "the" lock at once.
#[derive(Debug)]
pub struct BuildLock {
    _file: fs::File,
}

impl BuildLock {
    /// Block until the lock is ours.
    ///
    /// Callers must re-check the cache afterwards: the process we waited on has
    /// most likely just published the very entry we wanted.
    pub fn acquire(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(|e| err!("cannot open lock {}: {e}", path.display()))?;
        match file.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                log(&format!(
                    "waiting for another build holding {}",
                    path.display()
                ));
                file.lock()
                    .map_err(|e| err!("cannot lock {}: {e}", path.display()))?;
            }
            Err(TryLockError::Error(e)) => bail!("cannot lock {}: {e}", path.display()),
        }
        Ok(Self { _file: file })
    }
}

/// True while some process holds the lock at `path`.
#[must_use]
pub fn is_locked(path: &Path) -> bool {
    fs::File::open(path).is_ok_and(|f| matches!(f.try_lock(), Err(TryLockError::WouldBlock)))
}

#[cfg(test)]
mod tests {
    use super::{BuildLock, Stamp, is_locked};
    use crate::testing::TempDir;

    #[test]
    fn a_lock_is_held_until_dropped() {
        let tmp = TempDir::new("lock");
        let path = tmp.0.join("x.lock");
        assert!(!is_locked(&path), "no file, no holder");
        let lock = BuildLock::acquire(path.clone()).unwrap();
        assert!(is_locked(&path));
        drop(lock);
        // Other tests in this process fork `git` concurrently, and a child
        // forked just before the drop shares the descriptor until its exec --
        // so release is prompt, not instant. Allow it a moment.
        let released = (0..100).any(|_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            !is_locked(&path)
        });
        assert!(released, "released on drop -- and on exit, by the kernel");
        assert!(path.exists(), "the file stays; only the lock goes");
    }

    #[test]
    fn stamp_round_trips() {
        let stamp = Stamp {
            key: "4f9c2a71e0d83b56".into(),
            dep: "graphblas".into(),
            built_at: 1_700_000_000,
            manifest: "cc=clang version 22\ndep=graphblas\nsource=abc\n".into(),
        };
        let parsed = Stamp::parse(&stamp.render()).unwrap();
        assert_eq!(parsed, stamp);
    }

    #[test]
    fn stamp_without_marker_is_rejected() {
        assert!(Stamp::parse("key = a\ndep = b\n").is_err());
    }

    #[test]
    fn stamp_without_key_is_rejected() {
        assert!(Stamp::parse("dep = b\n\n--- manifest ---\nx=1\n").is_err());
    }
}
