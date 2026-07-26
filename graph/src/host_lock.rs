//! The host's **global** lock.
//!
//! Distinct from [`crate::query_lock`], and deliberately shaped differently:
//!
//! * A query lock is **per query** — a query acquires it, escalates it, releases
//!   it — so it is *injected* into the runtime that owns the query.
//! * The host lock is **per process**. Under Redis it is the module GIL, of which
//!   there is exactly one. A global is therefore an accurate model of it, not a
//!   shortcut.
//!
//! It exists because some host FFI mutates global host state — for Redis, the
//! RediSearch index spec lifecycle registers and stops garbage-collection timers
//! in the Redis event loop — and that must be serialised against the host's own
//! thread. Such code can sit far from any query (background index maintenance,
//! teardown), which is the other reason it cannot be an injected per-query value.
//!
//! ## Ordering
//!
//! Acquire this lock **before** any lock of your own. A query that has escalated
//! to writer mode holds it and then takes the indexer lock; a background task that
//! took the indexer lock first and reached for this one deadlocked the server for
//! six hours (issue #726). One direction only: host lock → everything else.

use std::sync::{Arc, OnceLock};

/// Host-provided access to the process-wide host lock.
///
/// Implementations must be **re-entrancy tolerant**: `lock` is expected to be a
/// no-op when the calling thread already holds the lock, because the underlying
/// primitive (Redis's module GIL) is not recursive and re-acquiring it would
/// self-deadlock.
pub trait HostLock: Send + Sync {
    /// Acquire the host lock, or note one more level of nesting if this thread
    /// already holds it.
    fn lock(&self);

    /// Release one level; releases the lock itself only at the outermost level.
    fn unlock(&self);

    /// True if the calling thread holds the host lock by any route.
    fn holds(&self) -> bool;
}

static HOST_LOCK: OnceLock<Arc<dyn HostLock>> = OnceLock::new();

/// Register the host's implementation. Idempotent; the first call wins. Intended
/// to be called once from the host's module-init callback.
pub fn set_host_lock(lock: Arc<dyn HostLock>) {
    let _ = HOST_LOCK.set(lock);
}

/// True if the calling thread holds the host lock — or if no host registered one,
/// in which case there is nothing to hold and every such precondition is met.
#[must_use]
pub fn holds_host_lock() -> bool {
    HOST_LOCK.get().is_none_or(|lock| lock.holds())
}

/// RAII guard for the host lock.
///
/// Bind one *before* taking any lock of your own:
/// `let _host = HostLockGuard::acquire();`
#[must_use]
pub struct HostLockGuard(bool);

impl HostLockGuard {
    /// Acquire the host lock for the guard's lifetime. A no-op — and harmless —
    /// when no host has registered an implementation.
    pub fn acquire() -> Self {
        match HOST_LOCK.get() {
            Some(lock) => {
                lock.lock();
                Self(true)
            }
            None => Self(false),
        }
    }
}

impl Drop for HostLockGuard {
    fn drop(&mut self) {
        if self.0
            && let Some(lock) = HOST_LOCK.get()
        {
            lock.unlock();
        }
    }
}

/// Debug-only assertion that the caller holds the host lock.
///
/// Guards host FFI that mutates global host state.
#[inline]
pub fn assert_holds_host_lock(what: &str) {
    debug_assert!(
        holds_host_lock(),
        "{what} requires the host lock; acquire it before your own locks \
         (HostLockGuard::acquire) — issue #726"
    );
}
