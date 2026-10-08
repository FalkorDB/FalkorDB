//! Telling a pinned dep checkout from a locally changed one.

use std::fs;
use std::path::Path;

use crate::err;
use crate::error::Result;
use crate::hash::{hash_file, sha256_hex};
use crate::recipes::{Ctx, gitlink_resolves};
use crate::util::capture;

impl Ctx<'_> {
    /// How a dep's checkout differs from the commit the lock pins, or `None`
    /// when it matches -- or when there is no git repository to ask, as in a
    /// Docker context, where the lock is all there is.
    ///
    /// The cache key names the *pinned* commit, so a build from anything else
    /// must not be published under it: every worktree on that pin would link
    /// it. A local change is a real workflow (FalkorDB carries its own
    /// RediSearch branch), so the caller builds it into a per-worktree entry
    /// instead of refusing.
    pub fn local_changes(
        &self,
        name: &str,
    ) -> Result<Option<LocalChanges>> {
        let dir = self.lock.source_dir(self.root, name)?;
        let link = dir.join(".git");
        if !(link.is_dir() || gitlink_resolves(&link)) {
            return Ok(None);
        }
        // A `.git` that is not a repository of its own (an empty marker left
        // by an interrupted build) makes git walk up to the superproject, whose
        // HEAD would read as a mismatch. Only trust an answer about `dir` itself.
        // git can also fail outright -- an empty marker in a Docker context,
        // with no superproject to walk up to. That too means "cannot tell",
        // and must not stop `ensure` before it recovers the marker.
        let Ok(toplevel) = git(&dir, &["rev-parse", "--show-toplevel"]) else {
            return Ok(None);
        };
        if fs::canonicalize(toplevel.trim()).ok() != fs::canonicalize(&dir).ok() {
            return Ok(None);
        }

        let head = git(&dir, &["rev-parse", "HEAD"])?.trim().to_owned();
        // A binary edit shows only as "Binary files differ", but the diff's
        // `index <old>..<new>` line still carries the new blob hash. Untracked
        // files do not appear at all, and count too: GraphBLAS's cmake globs
        // PreJIT/*.c, tracked or not.
        let mut diff = git(&dir, &["diff", "HEAD", "--submodule=diff"])?;
        for file in git(&dir, &["ls-files", "--others", "--exclude-standard"])?.lines() {
            let contents = hash_file(&dir.join(file))
                .map_err(|e| err!("cannot read {}/{file}: {e}", dir.display()))?;
            diff.push_str(&format!("untracked {file} {contents}\n"));
        }
        let pinned = &self.lock.get(name)?.rev;
        let mut reasons = Vec::new();
        if &head != pinned {
            reasons.push(format!("checked out at {head}, the lock pins {pinned}"));
        }
        if !diff.is_empty() {
            reasons.push("has uncommitted changes".to_owned());
        }
        if reasons.is_empty() {
            return Ok(None);
        }
        Ok(Some(LocalChanges {
            reason: reasons.join(" and "),
            fingerprint: sha256_hex(format!("{head}\n{diff}").as_bytes()),
        }))
    }
}

/// A dep checkout that differs from its pin. See [`Ctx::local_changes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalChanges {
    /// Human-readable, for the warning.
    pub reason: String,
    /// Changes whenever the checked-out commit or the uncommitted diff does,
    /// so an unchanged local build is reused rather than redone.
    pub fingerprint: String,
}

fn git(
    dir: &Path,
    args: &[&str],
) -> Result<String> {
    capture("git", args, Some(dir))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use crate::lock::{Entry, LockFile};
    use crate::recipes::Ctx;
    use crate::testing::TempDir;
    use crate::toolchain::{OpenMp, Toolchain};

    fn git(
        dir: &Path,
        args: &[&str],
    ) -> String {
        let out = Command::new("git")
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .current_dir(dir)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_owned()
    }

    /// `<root>/deps/Dep` as a repository with one commit, and a lock pinning it.
    fn checkout(root: &Path) -> (PathBuf, LockFile) {
        let dir = root.join("deps/Dep");
        fs::create_dir_all(&dir).unwrap();
        git(&dir, &["init", "-q"]);
        fs::write(dir.join("a.c"), "int a;\n").unwrap();
        git(&dir, &["add", "a.c"]);
        git(&dir, &["commit", "-q", "-m", "one"]);
        let lock = LockFile {
            entries: vec![Entry {
                name: "dep".into(),
                path: "deps/Dep".into(),
                url: "u".into(),
                rev: git(&dir, &["rev-parse", "HEAD"]),
                pin: String::new(),
            }],
        };
        (dir, lock)
    }

    fn toolchain() -> Toolchain {
        Toolchain {
            cc: None,
            cxx: None,
            cc_version: "cc".into(),
            cxx_version: "cxx".into(),
            target: "t".into(),
            openmp: OpenMp::Auto,
        }
    }

    fn ctx<'a>(
        root: &'a Path,
        lock: &'a LockFile,
        toolchain: &'a Toolchain,
    ) -> Ctx<'a> {
        Ctx {
            root,
            lock,
            toolchain,
            san: None,
            prejit_harvest: false,
        }
    }

    #[test]
    fn pinned_checkout_is_not_local() {
        let tmp = TempDir::new("pinned");
        let (_, lock) = checkout(&tmp.0);
        let tc = toolchain();
        assert_eq!(ctx(&tmp.0, &lock, &tc).local_changes("dep").unwrap(), None);
    }

    #[test]
    fn uncommitted_edit_is_local_and_fingerprinted() {
        let tmp = TempDir::new("dirty");
        let (dir, lock) = checkout(&tmp.0);
        let tc = toolchain();
        let ctx = ctx(&tmp.0, &lock, &tc);

        fs::write(dir.join("a.c"), "int a = 1;\n").unwrap();
        let first = ctx.local_changes("dep").unwrap().expect("an edit is local");
        assert!(first.reason.contains("uncommitted"), "{}", first.reason);
        assert_eq!(
            ctx.local_changes("dep").unwrap().unwrap().fingerprint,
            first.fingerprint,
            "an unchanged edit must keep its fingerprint, or the local build is redone"
        );

        fs::write(dir.join("a.c"), "int a = 2;\n").unwrap();
        assert_ne!(
            ctx.local_changes("dep").unwrap().unwrap().fingerprint,
            first.fingerprint,
            "a different edit must change the fingerprint"
        );
    }

    #[test]
    fn untracked_file_is_local() {
        // GraphBLAS's cmake globs PreJIT/*.c, so an untracked kernel is built in.
        let tmp = TempDir::new("untracked");
        let (dir, lock) = checkout(&tmp.0);
        let tc = toolchain();
        fs::write(dir.join("extra.c"), "int extra;\n").unwrap();
        let local = ctx(&tmp.0, &lock, &tc)
            .local_changes("dep")
            .unwrap()
            .expect("an untracked file is local");
        assert!(local.reason.contains("uncommitted"), "{}", local.reason);
    }

    #[test]
    fn binary_edits_get_distinct_fingerprints() {
        let tmp = TempDir::new("binary");
        let (dir, lock) = checkout(&tmp.0);
        fs::write(dir.join("blob.bin"), [0u8, 1, 2, 3]).unwrap();
        git(&dir, &["add", "blob.bin"]);
        git(&dir, &["commit", "-q", "-m", "blob"]);
        let lock = LockFile {
            entries: vec![Entry {
                rev: git(&dir, &["rev-parse", "HEAD"]),
                ..lock.entries[0].clone()
            }],
        };
        let tc = toolchain();
        let ctx = ctx(&tmp.0, &lock, &tc);
        fs::write(dir.join("blob.bin"), [0u8, 9, 9, 9]).unwrap();
        let first = ctx.local_changes("dep").unwrap().unwrap().fingerprint;
        fs::write(dir.join("blob.bin"), [0u8, 7, 7, 7]).unwrap();
        let second = ctx.local_changes("dep").unwrap().unwrap().fingerprint;
        assert_ne!(
            first, second,
            "the diff's index line must carry the new blob hash"
        );
    }

    #[test]
    fn commit_off_the_pin_is_local() {
        let tmp = TempDir::new("moved");
        let (dir, lock) = checkout(&tmp.0);
        let tc = toolchain();

        fs::write(dir.join("a.c"), "int a = 1;\n").unwrap();
        git(&dir, &["commit", "-q", "-am", "two"]);
        let local = ctx(&tmp.0, &lock, &tc)
            .local_changes("dep")
            .unwrap()
            .expect("a commit off the pin is local");
        assert!(local.reason.contains("the lock pins"), "{}", local.reason);
        assert!(!local.reason.contains("uncommitted"), "{}", local.reason);
    }

    #[test]
    fn no_repository_means_trust_the_lock() {
        // A Docker context: the sources are there, git is not.
        let tmp = TempDir::new("nogit");
        let (dir, lock) = checkout(&tmp.0);
        fs::remove_dir_all(dir.join(".git")).unwrap();
        fs::write(dir.join("a.c"), "int a = 1;\n").unwrap();
        let tc = toolchain();
        assert_eq!(ctx(&tmp.0, &lock, &tc).local_changes("dep").unwrap(), None);
    }

    #[test]
    fn empty_git_marker_without_a_superproject_is_not_an_error() {
        // A Docker context after an interrupted build: the marker, and no
        // repository anywhere above it. git fails; that must read as "cannot
        // tell", or ensure stops before it recovers the marker.
        let tmp = TempDir::new("marker-alone");
        let (dir, lock) = checkout(&tmp.0);
        fs::remove_dir_all(dir.join(".git")).unwrap();
        fs::create_dir(dir.join(".git")).unwrap();
        let tc = toolchain();
        let result = ctx(&tmp.0, &lock, &tc).local_changes("dep");
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn empty_git_marker_does_not_borrow_the_superprojects_head() {
        // An interrupted build can leave the empty `.git` marker behind. git
        // then answers for the enclosing repository, whose HEAD is not ours.
        let tmp = TempDir::new("marker");
        let (dir, lock) = checkout(&tmp.0);
        fs::remove_dir_all(dir.join(".git")).unwrap();
        fs::create_dir(dir.join(".git")).unwrap();
        git(&tmp.0, &["init", "-q"]);
        git(&tmp.0, &["commit", "-q", "--allow-empty", "-m", "super"]);
        let tc = toolchain();
        assert_eq!(ctx(&tmp.0, &lock, &tc).local_changes("dep").unwrap(), None);
    }
}
