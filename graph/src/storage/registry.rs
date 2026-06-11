//! Runtime injection for the shared storage seam.
//!
//! The core exposes [`register_backend`] / [`register_residency`] and **defaults**
//! to the built-in impls (a store falls back to its own default backend;
//! residency defaults to [`AllHot`]). A statically-linked add-on crate may call
//! these once at module init to swap in its own impls. A Cargo feature only
//! decides whether a *consumer* is compiled in; it does not name the add-on
//! impls. Everything stays statically linked into the single `.so`.
//!
//! Backends are per-store **instances of one shared [`Backend`] trait**, so the
//! registry holds them directly as `Arc<dyn Backend>` — no type erasure. The
//! index registers its backend under [`StoreKind::Index`]; the attribute store
//! will register its own under [`StoreKind::Attrs`]. The residency pool is
//! **shared**: a single `Arc<dyn Residency>` for the whole process.

#![allow(dead_code)] // Consumed by index/falkordb (feature-gated) and, later, the attribute store.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use parking_lot::RwLock;

use super::StoreKind;
use super::backend::Backend;
use super::residency::{AllHot, Residency};

/// The injected impls: per-store backend instances + one shared residency.
struct Registry {
    /// Per-store backend instances, keyed by [`StoreKind`].
    backends: HashMap<StoreKind, Arc<dyn Backend>>,
    /// The one process-wide residency pool, shared by every store.
    residency: Arc<dyn Residency>,
}

impl Default for Registry {
    fn default() -> Self {
        Self {
            backends: HashMap::new(),
            residency: Arc::new(AllHot),
        }
    }
}

fn registry() -> &'static RwLock<Registry> {
    static REGISTRY: OnceLock<RwLock<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Registry::default()))
}

/// Register `store`'s durable backend instance. Call once at startup, before any
/// of that store's structures are created.
pub fn register_backend(
    store: StoreKind,
    backend: Arc<dyn Backend>,
) {
    registry().write().backends.insert(store, backend);
}

/// The backend registered for `store`, or `None` if nothing was registered (the
/// store then uses its own default — e.g. the index falls back to `NullBackend`).
#[must_use]
pub fn backend(store: StoreKind) -> Option<Arc<dyn Backend>> {
    registry().read().backends.get(&store).cloned()
}

/// Install the one shared residency controller. Call once at startup. By default
/// this is never called and [`AllHot`] stands; an add-on crate calls it with its
/// budgeted impl.
pub fn register_residency(residency: Arc<dyn Residency>) {
    registry().write().residency = residency;
}

/// The shared residency controller (an `Arc` clone). Defaults to [`AllHot`]
/// until [`register_residency`] runs.
#[must_use]
pub fn residency() -> Arc<dyn Residency> {
    registry().read().residency.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dummy;
    impl Backend for Dummy {}

    #[test]
    fn backend_registers_and_reads_back_per_store() {
        // Nothing registered under Attrs by default.
        assert!(backend(StoreKind::Attrs).is_none());

        register_backend(StoreKind::Attrs, Arc::new(Dummy));
        assert!(backend(StoreKind::Attrs).is_some());
    }

    #[test]
    fn residency_defaults_to_all_hot() {
        // The default pool is unbounded (AllHot reports 0 resident bytes).
        assert_eq!(residency().resident_bytes(), 0);
    }
}
