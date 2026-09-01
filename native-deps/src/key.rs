//! Cache keys.
//!
//! Each dep's artifacts live at a path derived from *everything that can change
//! their bytes*. Identical inputs give an identical path and an instant reuse;
//! different inputs give a different path, so flavors (a sanitizer RediSearch, a
//! PreJIT-harvest GraphBLAS) coexist instead of overwriting each other.
//!
//! The manifest is canonical -- sorted, newline-delimited `key=value` -- so it
//! never depends on environment iteration order, and it is stored verbatim in
//! the `.stamp` so `diff`ing two stamps says exactly which input moved.

use std::collections::BTreeMap;

use crate::dep::Dep;
use crate::error::Result;
use crate::recipes::Ctx;
use crate::sha256::sha256_hex;
use crate::util::{hash_file, hash_tree, is_prejit_kernel};

/// Hex characters of the SHA-256 digest kept as the key. 64 bits is ample for a
/// per-developer artifact cache and keeps paths readable.
const KEY_LEN: usize = 16;

/// Hash of this crate's own sources, embedded at compile time by build.rs.
///
/// The cmake flags live in Rust now, so flipping `-O3` to `-O2` has to produce a
/// new key. Computing it at compile time (rather than by hashing `native-deps/
/// src/**` at runtime) means the library works even when it is consumed as a
/// build-dependency from a directory that no longer exists.
pub const RECIPE_HASH: &str = env!("NATIVE_DEPS_RECIPE_HASH");

/// A canonical `key=value` manifest.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Manifest {
    entries: BTreeMap<String, String>,
}

impl Manifest {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(
        &mut self,
        key: &str,
        value: impl Into<String>,
    ) -> &mut Self {
        self.entries.insert(key.to_owned(), value.into());
        self
    }

    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        for (k, v) in &self.entries {
            out.push_str(k);
            out.push('=');
            out.push_str(v);
            out.push('\n');
        }
        out
    }

    /// The cache key: a truncated SHA-256 of the rendered manifest.
    #[must_use]
    pub fn key(&self) -> String {
        let mut digest = sha256_hex(self.render().as_bytes());
        digest.truncate(KEY_LEN);
        digest
    }
}

/// Everything outside the manifest that the key computation needs.
impl Ctx<'_> {
    /// Inputs shared by every dep.
    fn common(
        &self,
        dep: Dep,
    ) -> Result<Manifest> {
        let mut m = Manifest::new();
        m.set("dep", dep.name())
            .set("source", &self.lock.get(dep.name())?.rev)
            .set("recipe", RECIPE_HASH)
            .set("target", &self.toolchain.target)
            .set("cc", &self.toolchain.cc_version)
            .set("cxx", &self.toolchain.cxx_version)
            .set("openmp", self.toolchain.openmp.tag());
        Ok(m)
    }

    /// The manifest for `dep`. `graphblas_key` must be supplied when `dep` is
    /// [`Dep::LaGraph`], because LAGraph statically links the GraphBLAS archive
    /// produced by that exact build.
    pub fn manifest(
        &self,
        dep: Dep,
        graphblas_key: Option<&str>,
    ) -> Result<Manifest> {
        let mut m = self.common(dep)?;
        match dep {
            Dep::GraphBlas => {
                let patch = self.root.join("build/graphblas/GB_control.patch");
                m.set("patch", hash_file(&patch)?);
                m.set(
                    "prejit_harvest",
                    if self.prejit_harvest { "1" } else { "0" },
                );
                // In harvest mode the vendored kernels are deliberately not
                // copied in, so their contents cannot affect the artifacts.
                let prejit = if self.prejit_harvest {
                    "skipped".to_owned()
                } else {
                    hash_tree(&self.root.join("build/graphblas/PreJIT"), &|p| {
                        is_prejit_kernel(p)
                    })?
                };
                m.set("prejit", prejit);
            }
            Dep::LaGraph => {
                m.set(
                    "graphblas",
                    graphblas_key.expect("lagraph key requires the graphblas key"),
                );
            }
            Dep::RediSearch => {
                m.set("san", self.san.unwrap_or("none"));
            }
        }
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::Manifest;

    #[test]
    fn render_is_sorted_and_stable() {
        let mut a = Manifest::new();
        a.set("zebra", "1").set("alpha", "2").set("mid", "3");
        assert_eq!(a.render(), "alpha=2\nmid=3\nzebra=1\n");

        let mut b = Manifest::new();
        b.set("mid", "3").set("zebra", "1").set("alpha", "2");
        assert_eq!(a.key(), b.key(), "insertion order must not affect the key");
    }

    #[test]
    fn key_changes_with_any_input() {
        let mut base = Manifest::new();
        base.set("dep", "graphblas").set("source", "aaaa");
        let mut bumped = base.clone();
        bumped.set("source", "bbbb");
        assert_ne!(base.key(), bumped.key());

        let mut extra = base.clone();
        extra.set("san", "address");
        assert_ne!(base.key(), extra.key());
    }

    #[test]
    fn key_is_hex_and_fixed_width() {
        let mut m = Manifest::new();
        m.set("dep", "lagraph");
        let key = m.key();
        assert_eq!(key.len(), super::KEY_LEN);
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
