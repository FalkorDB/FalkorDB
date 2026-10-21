//! Storage seams shared across the engine's pluggable stores.
//!
//! This module is the **general home** for the backend seam that both the native
//! index subsystem ([`crate::index::falkordb`]) and the attribute store will
//! share. It owns two things:
//!
//! - [`IndexBackend`] / [`AttrBackend`] — one storage-backend trait per store.
//!   Each is implemented by the default in-memory backend and, optionally, an
//!   alternate backend installed by a statically-linked add-on; a store talks to
//!   whichever is registered, unaware of which.
//! - the registry — runtime injection: a store registers its backend instance at
//!   module init ([`register_index_backend`] / [`register_attr_backend`]).
//!
//! This keeps each store's record types out of the general namespace — a store
//! serializes its own records to bytes before calling its backend — while giving
//! both stores one place to plug in.
//!
//! **Residency is deliberately not here.** Keeping data resident under a storage
//! budget is a private internal of a backend that does so: the in-memory backend
//! keeps everything resident, and nothing outside a backend ever coordinates
//! residency (a reader is unaware of backend internals; any I/O it triggers
//! happens inside backend code). So this module exposes no residency seam.
//!
//! This module is **not** gated behind a Cargo feature: it is core infrastructure
//! the (ungated) attribute store will also consume. Until both consumers are
//! wired, its items are unused in the default build — see the `allow(dead_code)`
//! on the registry.

pub mod backend;
pub mod error;
mod registry;

pub use backend::{AttrBackend, IndexBackend};
pub use error::{Result, StorageError};
pub use registry::{attr_backend, index_backend, register_attr_backend, register_index_backend};
