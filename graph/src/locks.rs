//! The two things this crate needs from its host's locks.
//!
//! Everything else about locking — which graph is locked, when the locks are
//! taken and released, what they even are — is the host's business and stays in
//! the host crate. Under Redis they are the module GIL and a per-graph
//! `RwLock`; nothing here knows that.
//!
//! ## Ordering rule
//!
//! **global lock → per-graph lock → indexer lock.** One direction, always. A
//! background task that took the indexer lock and then reached for the global
//! lock deadlocked the server for six hours (issue #726).
//!
//! ## Two-phase locking (mirrors FalkorDB C's `QueryCtx`)
//!
//! A write query cannot know up front that it will write, so it starts in
//! *reader* mode — holding only the per-graph read lock, running concurrently
//! with other readers through its match phase — and **escalates** on its first
//! mutation. Escalation must
//!
//! 1. release the read lock,
//! 2. acquire the global lock,
//! 3. acquire the per-graph write lock,
//!
//! in that order; taking the global lock while holding the read lock is the #726
//! AB-BA (a worker holding the read lock waiting for the GIL, against the main
//! thread holding the GIL waiting for the write lock). This is C's
//! `QueryCtx_AcquireWriteLock` flow (`src/query_ctx.c`).
//!
//! Escalation is **idempotent** and **sticky**: once a query is a writer it stays
//! one until it ends, so everything after the first mutation — including reads of
//! the shared, non-MVCC index — sees an exclusively locked graph.

use std::{marker::PhantomData, sync::OnceLock};

/// The one thing a query plan can ask its host to do: become a writer.
///
/// Implemented by whatever value the host uses to hold a query's locks, and
/// handed to [`Runtime::new`](crate::runtime::runtime::Runtime::new); operators
/// reach it through [`Runtime::write_escalation`](crate::runtime::runtime::Runtime::write_escalation).
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

/// The host's process-wide lock — under Redis, the module GIL.
///
/// Needed because some host FFI mutates global host state: the RediSearch index
/// spec lifecycle registers and stops garbage-collection timers in the Redis
/// event loop, which must be serialised against the host's own thread. Such code
/// runs both inside queries and far from any query (background index
/// maintenance, teardown), which is why this is a process-wide registration
/// rather than something threaded through a query.
///
/// Implementations must be **re-entrancy tolerant**: `lock` is a no-op when the
/// calling thread already holds the lock (the module GIL is not recursive), and
/// `unlock` releases only at the outermost level.
///
/// Use through [`GlobalLockGuard`]; never call these directly.
pub trait GlobalLock: Send + Sync {
    /// Acquire the lock, or note one more level of nesting if already held.
    fn lock(&self);
    /// Release one level; releases the lock itself only at the outermost level.
    fn unlock(&self);
}

static GLOBAL_LOCK: OnceLock<Box<dyn GlobalLock>> = OnceLock::new();

/// Register the host's implementation, once, from its module-init callback —
/// before any query can run.
pub fn set_global_lock(lock: Box<dyn GlobalLock>) {
    let already_set = GLOBAL_LOCK.set(lock).is_err();
    debug_assert!(!already_set, "the global lock was registered twice");
}

/// RAII guard for the global lock. Take it **before** any lock of your own:
/// `let _global = GlobalLockGuard::acquire();`
///
/// Not `Send`: the host releases the lock on the thread that took it.
#[must_use]
pub struct GlobalLockGuard {
    lock: Option<&'static dyn GlobalLock>,
    _not_send: PhantomData<*const ()>,
}

impl GlobalLockGuard {
    /// Acquire the global lock for the guard's lifetime.
    ///
    /// A no-op when no host has registered an implementation — the case for this
    /// crate's own unit tests, where there is no host state to serialise against.
    pub fn acquire() -> Self {
        let lock = GLOBAL_LOCK.get().map(|l| &**l);
        if let Some(lock) = lock {
            lock.lock();
        }
        Self {
            lock,
            _not_send: PhantomData,
        }
    }
}

impl Drop for GlobalLockGuard {
    fn drop(&mut self) {
        if let Some(lock) = self.lock {
            lock.unlock();
        }
    }
}
