//! `native-deps prune`: reclaim the cache, which otherwise never shrinks.
//!
//! Every pin bump, recipe edit or compiler change adds a full set of entries,
//! and nothing else ever removes one.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::cache::{Cache, STAMP_NAME, USED_NAME, lock_is_stale};
use crate::dep::Dep;
use crate::error::Result;
use crate::recipes::BUILD_DIR;

/// What to remove beyond the always-safe debris.
pub struct Options {
    /// Keep entries used within this long.
    pub keep_for: Duration,
    /// Keys this checkout resolves to right now; always kept.
    pub keep_keys: BTreeSet<String>,
    /// Also remove `deps/RediSearch/bin`, the in-tree RediSearch build output.
    pub worktree: Option<PathBuf>,
    pub dry_run: bool,
}

/// One thing prune removes, with why.
#[derive(Debug, PartialEq, Eq)]
pub struct Removal {
    pub path: PathBuf,
    pub bytes: u64,
    pub reason: &'static str,
}

/// Decide what to remove from `cache`, and remove it unless `dry_run`.
pub fn prune(
    cache: &Cache,
    opts: &Options,
) -> Result<Vec<Removal>> {
    let mut out = Vec::new();
    for dep in Dep::ALL {
        let dir = cache.root.join(dep.name());
        for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if is_being_built(&path) {
                // Its lock is live: a build is writing it, `.build` included.
                continue;
            }
            if let Some(reason) = verdict(&path, &name, opts) {
                out.push(Removal {
                    bytes: size_of(&path),
                    path,
                    reason,
                });
            } else if path.join(BUILD_DIR).is_dir() {
                // A finished entry keeps nothing in its scratch build tree.
                let scratch = path.join(BUILD_DIR);
                out.push(Removal {
                    bytes: size_of(&scratch),
                    path: scratch,
                    reason: "leftover build tree",
                });
            }
        }
    }
    if let Some(root) = &opts.worktree {
        let bin = root.join("deps/RediSearch/bin");
        if bin.is_dir() {
            out.push(Removal {
                bytes: size_of(&bin),
                path: bin,
                reason: "in-tree RediSearch build output (--worktree)",
            });
        }
    }

    if !opts.dry_run {
        for r in &out {
            let removed = if r.path.is_dir() {
                fs::remove_dir_all(&r.path)
            } else {
                fs::remove_file(&r.path)
            };
            if let Err(e) = removed {
                crate::util::log(&format!(
                    "WARNING: could not remove {}: {e}",
                    r.path.display()
                ));
            }
        }
    }
    Ok(out)
}

/// Why `path` should go, or `None` to keep it.
fn verdict(
    path: &Path,
    name: &str,
    opts: &Options,
) -> Option<&'static str> {
    if name.ends_with(".lock") {
        return lock_is_stale(path).then_some("stale lock");
    }
    if !path.is_dir() {
        return None;
    }
    if !path.join(STAMP_NAME).is_file() {
        return Some("incomplete build");
    }
    if opts.keep_keys.contains(name) {
        return None;
    }
    let idle = last_used(path).and_then(|t| SystemTime::now().duration_since(t).ok());
    match idle {
        Some(idle) if idle < opts.keep_for => None,
        _ if name.starts_with(crate::LOCAL_PREFIX) => Some("unused local build"),
        _ => Some("unused"),
    }
}

fn is_being_built(entry: &Path) -> bool {
    let lock = entry.with_extension("lock");
    entry.is_dir() && lock.exists() && !lock_is_stale(&lock)
}

/// When an entry was last resolved: `.used` if present, else the build time.
fn last_used(entry: &Path) -> Option<SystemTime> {
    [USED_NAME, STAMP_NAME]
        .iter()
        .find_map(|f| fs::metadata(entry.join(f)).ok()?.modified().ok())
}

fn size_of(path: &Path) -> u64 {
    if path.is_file() {
        return fs::metadata(path).map_or(0, |m| m.len());
    }
    crate::hash::collect_files(path)
        .unwrap_or_default()
        .iter()
        .filter_map(|f| fs::symlink_metadata(f).ok())
        .map(|m| m.len())
        .sum()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::Path;
    use std::time::{Duration, SystemTime};

    use super::{Options, prune};
    use crate::cache::{Cache, STAMP_NAME, USED_NAME};
    use crate::testing::TempDir;

    const DAY: Duration = Duration::from_secs(24 * 3600);

    fn entry(
        dir: &Path,
        stamp: bool,
        age: Duration,
    ) {
        fs::create_dir_all(dir.join("lib")).unwrap();
        fs::write(dir.join("lib/libx.a"), vec![0u8; 1000]).unwrap();
        if stamp {
            fs::write(dir.join(STAMP_NAME), "key = x\n").unwrap();
            let used = dir.join(USED_NAME);
            fs::write(&used, "").unwrap();
            fs::File::options()
                .write(true)
                .open(&used)
                .unwrap()
                .set_modified(SystemTime::now() - age)
                .unwrap();
        }
    }

    fn opts(keep: &[&str]) -> Options {
        Options {
            keep_for: 14 * DAY,
            keep_keys: keep
                .iter()
                .map(|s| (*s).to_owned())
                .collect::<BTreeSet<_>>(),
            worktree: None,
            dry_run: false,
        }
    }

    #[test]
    fn keeps_current_and_recent_and_removes_the_rest() {
        let tmp = TempDir::new("prune");
        let gb = tmp.0.join("graphblas");
        entry(&gb.join("current0000000000"), true, 90 * DAY);
        entry(&gb.join("recent0000000000"), true, DAY);
        entry(&gb.join("old0000000000000"), true, 30 * DAY);
        entry(&gb.join("halfbuilt0000000"), false, DAY);
        entry(&gb.join("local-0123456789ab"), true, 30 * DAY);
        fs::create_dir_all(gb.join("recent0000000000/.build")).unwrap();
        fs::write(
            gb.join("dead0000000000000.lock"),
            "pid 999999999\nsince 0\n",
        )
        .unwrap();
        let cache = Cache {
            root: tmp.0.clone(),
            prebuilt: Vec::new(),
        };

        let mut dry = opts(&["current0000000000"]);
        dry.dry_run = true;
        let planned = prune(&cache, &dry).unwrap();
        assert_eq!(planned.len(), 5, "{planned:#?}");
        assert!(
            gb.join("old0000000000000").exists(),
            "dry run removed something"
        );

        prune(&cache, &opts(&["current0000000000"])).unwrap();
        let left: BTreeSet<String> = fs::read_dir(&gb)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            left,
            ["current0000000000", "recent0000000000"]
                .map(String::from)
                .into(),
        );
        assert!(!gb.join("recent0000000000/.build").exists());
        assert!(gb.join("recent0000000000/lib/libx.a").exists());
    }

    #[test]
    fn an_entry_whose_build_is_running_is_left_alone() {
        let tmp = TempDir::new("prune-live");
        let gb = tmp.0.join("graphblas");
        entry(&gb.join("building00000000"), false, DAY);
        // Mid-build an entry has no stamp yet, but does have its scratch tree.
        fs::create_dir_all(gb.join("building00000000/.build")).unwrap();
        fs::write(
            gb.join("building00000000.lock"),
            format!("pid {}\nsince 0\n", std::process::id()),
        )
        .unwrap();
        let cache = Cache {
            root: tmp.0.clone(),
            prebuilt: Vec::new(),
        };
        assert!(prune(&cache, &opts(&[])).unwrap().is_empty());
    }
}
