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
use crate::recipes::graphblas::PREJIT_DIR;
use crate::util::is_prejit_kernel;

/// Hash of the recipe sources, embedded by build.rs, so a flag change in Rust
/// is a key change.
pub const RECIPE_HASH: &str = env!("NATIVE_DEPS_RECIPE_HASH");

/// The inputs a key is computed from.
#[derive(Debug, Clone, Default)]
pub struct Manifest(BTreeMap<&'static str, String>);

impl Manifest {
    pub fn set(
        &mut self,
        name: &'static str,
        value: impl Into<String>,
    ) {
        self.0.insert(name, value.into());
    }

    #[must_use]
    pub fn render(&self) -> String {
        self.0.iter().map(|(k, v)| format!("{k}={v}\n")).collect()
    }

    /// The first 16 hex characters (64 bits) of the manifest's SHA-256.
    #[must_use]
    pub fn key(&self) -> String {
        sha256_hex(self.render().as_bytes())[..16].to_owned()
    }
}

impl Ctx<'_> {
    /// The manifest for `dep`. LAGraph needs `graphblas_key`: it links the
    /// archive of that exact GraphBLAS build.
    pub fn manifest(
        &self,
        dep: Dep,
        graphblas_key: Option<&str>,
    ) -> Result<Manifest> {
        let mut m = Manifest::default();
        m.set("dep", dep.name());
        m.set("source", &self.lock.get(dep.name())?.rev);
        m.set("recipe", RECIPE_HASH);
        m.set("target", &self.toolchain.target);
        m.set("cc", &self.toolchain.cc_version);
        m.set("cxx", &self.toolchain.cxx_version);
        m.set("openmp", self.toolchain.openmp.tag());
        match dep {
            Dep::GraphBlas => {
                let patch = self.root.join("build/graphblas/GB_control.patch");
                let patch_hash =
                    hash_file(&patch).map_err(|e| err!("cannot read {}: {e}", patch.display()))?;
                let kernels = hash_tree(&self.root.join(PREJIT_DIR), &is_prejit_kernel)
                    .map_err(|e| err!("cannot hash {PREJIT_DIR}: {e}"))?;
                m.set("patch", patch_hash);
                m.set("prejit", kernels);
                m.set(
                    "prejit_harvest",
                    if self.prejit_harvest { "1" } else { "0" },
                );
            }
            Dep::LaGraph => m.set(
                "graphblas",
                graphblas_key.expect("lagraph key requires the graphblas key"),
            ),
            Dep::RediSearch => {
                m.set("san", self.san.unwrap_or("none"));
                m.set("rustc", &self.toolchain.rustc_version);
            }
        }
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::Manifest;

    #[test]
    fn key_ignores_insertion_order_and_follows_every_input() {
        let mut a = Manifest::default();
        a.set("source", "aaaa");
        a.set("dep", "graphblas");
        let mut b = Manifest::default();
        b.set("dep", "graphblas");
        b.set("source", "aaaa");
        assert_eq!(a.render(), "dep=graphblas\nsource=aaaa\n");
        assert_eq!(a.key(), b.key());

        b.set("source", "bbbb");
        assert_ne!(a.key(), b.key());
        a.set("san", "address");
        assert_ne!(a.key(), Manifest::default().key());
        assert_eq!(a.key().len(), 16);
        assert!(a.key().chars().all(|c| c.is_ascii_hexdigit()));
    }
}
