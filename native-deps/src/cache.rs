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

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::dep::Dep;
use crate::error::Result;
use crate::util::{env_opt, log, now_secs};
use crate::{bail, err};

pub const STAMP_NAME: &str = ".stamp";
const MANIFEST_SEPARATOR: &str = "--- manifest ---";
const DEFAULT_LOCK_TIMEOUT_SECS: u64 = 90 * 60;

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

    /// Every key present for `dep`, for diagnostics when a lookup misses.
    #[must_use]
    pub fn available(
        &self,
        dep: Dep,
    ) -> Vec<String> {
        let mut keys = Vec::new();
        for root in self.prebuilt.iter().chain(std::iter::once(&self.root)) {
            let dir = root.join(dep.name());
            for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
                if entry.path().join(STAMP_NAME).is_file()
                    && let Some(name) = entry.file_name().to_str()
                {
                    keys.push(format!("{}/{name}", dir.display()));
                }
            }
        }
        keys.sort();
        keys
    }
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

/// An `O_EXCL` build lock. Released on drop, including on unwind.
#[derive(Debug)]
pub struct BuildLock {
    path: PathBuf,
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
        let timeout = Duration::from_secs(
            env_opt("FALKORDB_NATIVE_DEPS_LOCK_TIMEOUT")
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_LOCK_TIMEOUT_SECS),
        );

        let mut announced = false;
        loop {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut f) => {
                    use std::io::Write as _;
                    let _ = writeln!(f, "pid {}\nsince {}", std::process::id(), now_secs());
                    return Ok(Self { path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if lock_age(&path).is_some_and(|age| age > timeout) {
                        log(&format!(
                            "stale build lock {} (older than {}s) - taking it over",
                            path.display(),
                            timeout.as_secs()
                        ));
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    if !announced {
                        log(&format!(
                            "waiting for another build holding {}",
                            path.display()
                        ));
                        announced = true;
                    }
                    std::thread::sleep(Duration::from_secs(2));
                }
                Err(e) => bail!("cannot create lock {}: {e}", path.display()),
            }
        }
    }
}

impl Drop for BuildLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn lock_age(path: &Path) -> Option<Duration> {
    fs::metadata(path).ok()?.modified().ok()?.elapsed().ok()
}

#[cfg(test)]
mod tests {
    use super::Stamp;

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
