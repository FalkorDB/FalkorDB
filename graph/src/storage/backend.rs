//! The storage-backend seams — one trait per pluggable store.

/// Storage backend for the native index subsystem ([`crate::index::falkordb`]).
///
/// Implemented by the default in-memory backend and, optionally, an alternate
/// backend installed by a statically-linked add-on; the index talks to whichever
/// is registered ([`super::registry`]) and is unaware of which. Where the bytes
/// live is entirely the implementation's concern — an alternate backend may keep
/// data under a storage budget and manage its own residency internally; the
/// in-memory backend keeps everything resident and has none.
///
/// **Minimal and expandable by design.** It starts empty and grows its real
/// surface — a durable byte-store (append-only log, blob, key/value) — as the
/// index wires storage. The index keeps its own record types and serializes them
/// to bytes before calling these methods.
pub trait IndexBackend: Send + Sync {}

/// Storage backend for the attribute (property) store.
///
/// **Stub** until the attribute store migrates onto this seam — defined now so
/// the registry already has its slot. Same minimal-and-evolving contract as
/// [`IndexBackend`].
pub trait AttrBackend: Send + Sync {}
