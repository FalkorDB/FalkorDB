//! Cache keys: a hash of everything that can change a dep's artifacts.
//!
//! The manifest behind a key is sorted `key=value` lines, stored verbatim in
//! the `.stamp`, so diffing two stamps names the input that moved.

use std::collections::BTreeMap;

use crate::dep::Dep;
use crate::err;
use crate::error::Result;
use crate::hash::{hash_file, hash_tree, sha256_hex};
use crate::recipes::Ctx;
use crate::util::is_prejit_kernel;

/// Hex characters of the SHA-256 kept: 64 bits, readable paths.
const KEY_LEN: usize = 16;

/// Hash of the recipe sources, embedded by build.rs, so a flag change in Rust
/// is a key change.
pub const RECIPE_HASH: &str = env!("NATIVE_DEPS_RECIPE_HASH");

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Manifest {
    entries: BTreeMap<String, String>,
}

impl Manifest {
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

    #[must_use]
    pub fn key(&self) -> String {
        let mut digest = sha256_hex(self.render().as_bytes());
        digest.truncate(KEY_LEN);
        digest
    }
}

impl Ctx<'_> {
    fn common(
        &self,
        dep: Dep,
    ) -> Result<Manifest> {
        let mut m = Manifest::default();
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
                m.set(
                    "patch",
                    hash_file(&patch).map_err(|e| err!("cannot read {}: {e}", patch.display()))?,
                );
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
                    })
                    .map_err(|e| err!("cannot hash build/graphblas/PreJIT: {e}"))?
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
        let mut a = Manifest::default();
        a.set("zebra", "1").set("alpha", "2").set("mid", "3");
        assert_eq!(a.render(), "alpha=2\nmid=3\nzebra=1\n");

        let mut b = Manifest::default();
        b.set("mid", "3").set("zebra", "1").set("alpha", "2");
        assert_eq!(a.key(), b.key(), "insertion order must not affect the key");
    }

    #[test]
    fn key_changes_with_any_input() {
        let mut base = Manifest::default();
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
        let mut m = Manifest::default();
        m.set("dep", "lagraph");
        let key = m.key();
        assert_eq!(key.len(), super::KEY_LEN);
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
