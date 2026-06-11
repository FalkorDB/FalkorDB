//! Process-wide storage seams shared across the engine's pluggable stores.
//!
//! This module is the **general home** for the storage seam that both the native
//! index subsystem ([`crate::index::falkordb`]) and the attribute store will
//! share. It owns three things:
//!
//! - [`StoreKind`] — the identity of a pluggable store (index, attribute store).
//! - [`Backend`] — the single shared durable-backend trait; each store registers
//!   its own backend instance.
//! - [`Residency`] — the **one process-wide resident pool** under a single
//!   storage budget, shared by every store so they compete for one budget rather
//!   than each carrying its own. Default [`AllHot`].
//! - the registry — runtime injection: per-store **backends** plus the one shared
//!   residency, defaulting to the built-in impls, swapped in by a statically
//!   linked add-on crate at module init.
//!
//! This keeps store-specific record types out of the general namespace (each
//! store serializes its own records to bytes before calling [`Backend`]) while
//! still giving both stores one registry and one storage budget.
//!
//! Unlike [`crate::index::falkordb`], this module is **not** gated behind a Cargo
//! feature: it is core infrastructure the (ungated) attribute store will also
//! consume. Until both consumers are wired, some items are unused in the default
//! build — see the `allow(dead_code)` on the registry's internals.

pub mod backend;
pub mod error;
mod registry;
pub mod residency;

pub use backend::Backend;
pub use error::{Result, StorageError};
pub use registry::{backend, register_backend, register_residency, residency};
pub use residency::{AllHot, Residency, ResidentId, ShardGuard};

/// Identity of a pluggable store that participates in the shared storage seam.
/// Used as the registry key and as the store component of a [`ResidentId`], so
/// two stores never collide in the one shared resident pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoreKind {
    /// The native MVCC index subsystem ([`crate::index::falkordb`]).
    Index,
    /// The attribute (property) store. **Stub** — the attribute-store backend
    /// and its registration land when the attribute store migrates onto this
    /// seam; today nothing registers or reads under this key. It exists so the
    /// shared registry and the shared resident pool already have a slot for it.
    Attrs,
}
