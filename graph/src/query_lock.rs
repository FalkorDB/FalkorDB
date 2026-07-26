//! Host-agnostic abstraction over the **per-query graph lock**.
//!
//! Mutating shared, non-MVCC state (the RediSearch index) and publishing a new
//! graph version must happen in *writer* mode. What "writer mode" means is a
//! property of the **host**, not of this crate: under Redis it is
//! `RedisModule_ThreadSafeContextLock` (the GIL) plus the per-graph write lock,
//! acquired in that order. This crate must not know any of that — hence the
//! trait: the host registers an implementation once at startup, and the query
//! runtime merely says "I am about to write".
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

use std::sync::{Arc, OnceLock};

/// Lock mode held for the current query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// No per-graph lock held.
    Unlocked,
    /// Per-graph read lock only (the match phase).
    Read,
    /// Host global lock + per-graph write lock (from the first mutation on).
    Write,
}

/// Host-provided lock protocol for the current query.
///
/// Implementations are registered once via [`set_query_lock`] and are shared
/// across all query threads, so state must be per-thread inside the impl (the
/// analogue of C's thread-local `QueryCtx`).
pub trait QueryLock: Send + Sync {
    /// Enter **reader** mode for the current query: acquire the host's per-graph
    /// read lock (and nothing else — no host global lock).
    ///
    /// Which graph is a host concern: the host resolves the target before the
    /// query starts and hands it to its own implementation. This call is what
    /// *acquires* the lock, so the whole read → write → release lifecycle is
    /// expressed here rather than half here and half in host code.
    fn acquire_read(&self) -> Result<(), String>;

    /// Release every lock held for the current query (write lock and host lock if
    /// escalated, otherwise the read lock). Idempotent.
    fn release(&self);

    /// The lock mode currently held by the calling thread.
    fn mode(&self) -> AccessMode;

    /// Escalate the calling thread's current query to writer mode, acquiring
    /// the host lock and the per-graph write lock in the host's required order.
    ///
    /// Must be **idempotent**: a no-op when this thread is already a writer.
    /// Returns `Err` if the escalation cannot be honoured (e.g. the graph key
    /// disappeared while the read lock was released), in which case the caller
    /// must abort the query rather than mutate.
    fn upgrade_to_write(&self) -> Result<(), String>;

    /// Acquire **only** the host global lock — no per-graph lock — for
    /// background maintenance that is not part of a query (index population,
    /// index teardown).
    ///
    /// Such tasks take their own internal locks (e.g. the indexer's
    /// serialization lock) and may call host FFI that requires the global lock.
    /// They must therefore acquire the host lock *first*, matching the order a
    /// query uses after escalating (host lock → … → indexer lock); acquiring it
    /// afterwards is an AB-BA deadlock against a query doing index DDL.
    ///
    /// Must be a no-op when the calling thread already holds the host lock.
    fn lock_host(&self);

    /// Release a host lock taken by [`QueryLock::lock_host`]; a no-op if that
    /// call was itself a no-op.
    fn unlock_host(&self);

    /// True if the calling thread holds the host global lock by any route:
    /// implicitly (host's main thread), via escalation, or via [`Self::lock_host`].
    fn holds_host_lock(&self) -> bool;
}

/// RAII guard for [`QueryLock::lock_host`].
///
/// Bind one *before* taking any lock of your own:
/// `let _host = HostLock::acquire();`
#[must_use]
pub struct HostLock(bool);

impl HostLock {
    /// Acquire the host global lock for the duration of the guard. No-op (and
    /// harmless) when no host implementation is registered.
    pub fn acquire() -> Self {
        match QUERY_LOCK.get() {
            Some(lock) => {
                lock.lock_host();
                Self(true)
            }
            None => Self(false),
        }
    }
}

impl Drop for HostLock {
    fn drop(&mut self) {
        if self.0
            && let Some(lock) = QUERY_LOCK.get()
        {
            lock.unlock_host();
        }
    }
}

/// True if the calling thread holds the host global lock. `true` when no host
/// implementation is registered (nothing to hold).
#[must_use]
pub fn holds_host_lock() -> bool {
    QUERY_LOCK.get().is_none_or(|lock| lock.holds_host_lock())
}

/// Debug-only guard for taking the host global lock: the current query must not
/// be holding the per-graph **read** lock.
///
/// That is precisely the issue #726 AB-BA inversion — a worker holding the read
/// lock and waiting for the host lock, versus the host's main thread holding the
/// host lock and waiting for the write lock. Escalation avoids it by releasing
/// the read lock first, so this catches any *other* path that reaches for the
/// host lock mid-query.
#[inline]
pub fn assert_safe_to_take_host_lock() {
    debug_assert!(
        mode() != AccessMode::Read,
        "taking the host global lock while holding the per-graph read lock is the \
         #726 deadlock; release the read lock first (upgrade_to_write does this)"
    );
}

/// Debug-only assertion that the caller holds the host global lock.
///
/// Guards host FFI that mutates global host state (for Redis: the RediSearch
/// spec lifecycle, which registers/stops GC timers in the Redis event loop).
#[inline]
pub fn assert_holds_host_lock(what: &str) {
    debug_assert!(
        holds_host_lock(),
        "{what} requires the host global lock; acquire it before your own locks \
         (HostLock::acquire / upgrade_to_write) — issue #726"
    );
}

static QUERY_LOCK: OnceLock<Arc<dyn QueryLock>> = OnceLock::new();

/// Register the host's commit-lock implementation. Idempotent; the first call
/// wins. Intended to be called once from the host's module-init callback.
pub fn set_query_lock(lock: Arc<dyn QueryLock>) {
    let _ = QUERY_LOCK.set(lock);
}

/// Escalate the current query to writer mode; see
/// [`QueryLock::upgrade_to_write`].
///
/// When no host implementation is registered (unit tests, embedders that drive
/// the graph directly with no concurrent readers) this is a no-op success: with
/// no host lock there is nothing to order and nothing to exclude.
pub fn upgrade_to_write() -> Result<(), String> {
    match QUERY_LOCK.get() {
        Some(lock) => lock.upgrade_to_write(),
        None => Ok(()),
    }
}

/// Enter reader mode for the current query; see [`QueryLock::acquire_read`].
pub fn acquire_read() -> Result<(), String> {
    match QUERY_LOCK.get() {
        Some(lock) => lock.acquire_read(),
        None => Ok(()),
    }
}

/// Release all locks held for the current query; see [`QueryLock::release`].
pub fn release() {
    if let Some(lock) = QUERY_LOCK.get() {
        lock.release();
    }
}

/// The lock mode held by the calling thread.
#[must_use]
pub fn mode() -> AccessMode {
    QUERY_LOCK
        .get()
        .map_or(AccessMode::Unlocked, |lock| lock.mode())
}

/// True if the calling thread holds writer mode. `false` when no host
/// implementation is registered.
#[must_use]
pub fn holds_write() -> bool {
    mode() == AccessMode::Write
}

/// Debug-only assertion that the caller is in writer mode.
///
/// Guards the operations that mutate shared non-MVCC state (index documents,
/// RediSearch spec lifecycle) or publish a graph version. No-op in release
/// builds and when no host implementation is registered.
#[inline]
pub fn assert_holds_write(what: &str) {
    debug_assert!(
        QUERY_LOCK
            .get()
            .is_none_or(|lock| lock.mode() == AccessMode::Write),
        "{what} requires writer mode (commit lock held); \
         call query_lock::upgrade_to_write() first (issue #726)"
    );
}
