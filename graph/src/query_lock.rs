//! Host-agnostic seam for **escalating a query to writer mode**.
//!
//! Mutating shared, non-MVCC state (the RediSearch index) and publishing a new
//! graph version must happen in *writer* mode. What "writer mode" means is a
//! property of the **host**, not of this crate: under Redis it is
//! `RedisModule_ThreadSafeContextLock` (the GIL) plus the per-graph write lock,
//! acquired in that order. This crate must not know any of that — hence the
//! trait: each [`Runtime`](crate::runtime::runtime::Runtime) is handed the host's
//! implementation, and the query runtime merely says "I am about to write".
//!
//! Everything else about the lock — which graph it targets, when it is taken and
//! released, how reader mode is entered — is host bookkeeping and deliberately
//! *not* part of this seam.
//!
//! ## Two-phase locking (mirrors FalkorDB C)
//!
//! A write query starts in *reader* mode (holding only the per-graph read lock)
//! so it runs concurrently with other readers during its match phase, and
//! escalates on its first mutation. The escalation itself must
//!
//! 1. release the read lock,
//! 2. acquire the host lock (GIL),
//! 3. acquire the per-graph write lock,
//!
//! in that order — never GIL-while-holding-read, which is the AB-BA deadlock of
//! issue #726 (a worker holding the read lock and waiting for the GIL vs. the
//! main thread holding the GIL and waiting for the write lock). This is exactly
//! C's `QueryCtx_AcquireWriteLock` flow (`src/query_ctx.c`).
//!
//! Escalation is **idempotent** and **sticky**: once a query is a writer it
//! stays a writer until it finishes, so everything after the first mutation —
//! including reads of the shared index — observes a consistent, exclusively
//! locked graph.

/// Host-provided escalation hook for the query currently running on this thread.
///
/// One implementation is shared by every query thread, so per-query state must
/// live per-thread inside the impl (the analogue of C's thread-local `QueryCtx`).
pub trait QueryLock: Send + Sync {
    /// Escalate the calling thread's query to writer mode, acquiring the host
    /// lock and the per-graph write lock in the host's required order.
    ///
    /// Must be **idempotent**: a no-op when this thread is already a writer.
    /// Returns `Err` if the escalation cannot be honoured (e.g. the graph key
    /// disappeared while the read lock was released), in which case the caller
    /// must abort the query rather than mutate.
    fn upgrade_to_write(&self) -> Result<(), String>;
}
