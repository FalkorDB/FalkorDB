//! The shared storage-backend seam.

/// A durable backend for one store.
///
/// This is the **single shared** trait every store's backend implements; a store
/// registers its backend instance in the [registry](super::registry) under its
/// [`StoreKind`](super::StoreKind), and the registry holds it as
/// `Arc<dyn Backend>`.
///
/// It is deliberately a **single, minimal, expandable** trait rather than a
/// per-store trait — that is what lets the registry store `Arc<dyn Backend>`
/// directly (no `Any` erasure, no per-store downcast). It starts empty and grows
/// the durable byte-store surface (append-only log, blob, and key/value
/// operations) as stores wire durability; each store keeps its own record types
/// and serializes them to bytes before calling these methods.
pub trait Backend: Send + Sync {}
