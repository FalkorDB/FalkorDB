//! The one thing this crate needs from its host's locks.
//!
//! Everything else — which graph is locked, when the locks are taken and released,
//! what they even are — is the host's business. Under Redis they are the module GIL
//! and a per-graph `RwLock`; nothing here knows that.
//!
//! ## Ordering rule
//!
//! **global lock → per-graph lock → indexer lock.** One direction, always. A
//! background task that took the indexer lock and then reached for the host's global
//! lock deadlocked the server for six hours (issue #726). That inversion is now
//! unrepresentable here: this crate cannot take the global lock at all, so code that
//! holds the indexer lock has nothing to invert against. Host FFI that *does* need
//! the global lock (the RediSearch index spec lifecycle) documents it as a
//! precondition and relies on the caller having escalated.
//!
//! ## Two-phase locking (mirrors C's `QueryCtx_AcquireWriteLock`)
//!
//! A write query cannot know up front that it will write, so it starts as a
//! *reader* — per-graph read lock only, concurrent with other readers through its
//! match phase — and **escalates** on its first mutation: release read, take the
//! global lock, take the per-graph write lock, in that order. Taking the global lock
//! while still holding the read lock is the #726 AB-BA against a main-thread command
//! holding the GIL and waiting for the write lock.
//!
//! Escalation is **idempotent** and **sticky**: once a writer, a query stays one
//! until it ends, so everything after the first mutation — including reads of the
//! shared, non-MVCC index — sees an exclusively locked graph.

/// The one thing a query plan can ask its host to do: become a writer.
///
/// Implemented by whatever value the host uses to hold a query's locks, and handed
/// to [`Runtime::new`](crate::runtime::runtime::Runtime::new); operators reach it
/// through [`Runtime::write_escalation`](crate::runtime::runtime::Runtime::write_escalation).
/// No `Send`/`Sync`: a query's locks belong to one thread and are never shared.
pub trait WriteEscalation {
    /// Escalate this query to writer mode, acquiring the global lock and the
    /// per-graph write lock in the host's required order.
    ///
    /// Must be **idempotent**: a no-op when already a writer. `Err` means the
    /// escalation could not be honoured (e.g. the graph was deleted while no
    /// per-graph lock was held), and the caller must abort instead of mutating.
    fn upgrade_to_write(&self) -> Result<(), String>;
}
