//! Runtime injection for the storage-backend seam.
//!
//! Two pluggable backends — one for the index, one for the attribute store —
//! each a concrete instance of its own trait ([`IndexBackend`] / [`AttrBackend`]),
//! held directly as `Arc<dyn _>` (no type erasure, no enum key). A consumer
//! registers its backend exactly once at startup: the default build registers an
//! in-memory backend; a statically-linked add-on may register an alternate
//! backend instead. A Cargo feature only decides which *consumer* is compiled in;
//! it does not name the add-on impls. Everything stays in the single `.so`.
//!
//! Each slot is a set-once [`OnceLock`]: registered once before first use, read
//! everywhere after — no `Option` to thread, and a missing registration is a
//! startup bug that panics, not a runtime branch.
//!
//! Residency is **not** here: keeping data resident under a storage budget is a
//! private internal of a backend that does so (the in-memory backend keeps
//! everything resident, and nothing outside a backend ever coordinates it), so it
//! never appears in this registry.

#![allow(dead_code)] // Consumed by index/falkordb (feature-gated) and, later, the attribute store.

use std::sync::{Arc, OnceLock};

use super::backend::{AttrBackend, IndexBackend};

static INDEX_BACKEND: OnceLock<Arc<dyn IndexBackend>> = OnceLock::new();
static ATTR_BACKEND: OnceLock<Arc<dyn AttrBackend>> = OnceLock::new();

/// Register the index's storage backend. Call once at startup, before any index
/// structure is created; panics if one was already registered.
pub fn register_index_backend(backend: Arc<dyn IndexBackend>) {
    assert!(
        INDEX_BACKEND.set(backend).is_ok(),
        "index backend already registered"
    );
}

/// The registered index backend.
///
/// Panics if none was registered — a required backend that was never wired is a
/// startup bug, not a runtime condition to thread through every call site.
#[must_use]
pub fn index_backend() -> Arc<dyn IndexBackend> {
    INDEX_BACKEND
        .get()
        .expect("index backend not registered")
        .clone()
}

/// Register the attribute store's backend. Call once at startup; panics if one
/// was already registered.
pub fn register_attr_backend(backend: Arc<dyn AttrBackend>) {
    assert!(
        ATTR_BACKEND.set(backend).is_ok(),
        "attr backend already registered"
    );
}

/// The registered attribute-store backend. Panics if none was registered (see
/// [`index_backend`]).
#[must_use]
pub fn attr_backend() -> Arc<dyn AttrBackend> {
    ATTR_BACKEND
        .get()
        .expect("attr backend not registered")
        .clone()
}
