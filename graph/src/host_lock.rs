//! The host's **global** lock.
//!
//! Distinct from [`crate::query_lock`], and deliberately shaped differently:
//!
//! * A query lock is **per query** — a query escalates it and releases it — so it
//!   is *injected* into the runtime that owns the query.
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

use std::{marker::PhantomData, sync::OnceLock};

/// Host-provided access to the process-wide host lock.
///
/// Implementations must be **re-entrancy tolerant**: `lock` is expected to be a
/// no-op when the calling thread already holds the lock, because the underlying
/// primitive (Redis's module GIL) is not recursive and re-acquiring it would
/// self-deadlock. `unlock` must likewise release only at the outermost level.
///
/// Use through [`HostLockGuard`] rather than calling these directly.
pub trait HostLock: Send + Sync {
    /// Acquire the host lock, or note one more level of nesting if this thread
    /// already holds it.
    fn lock(&self);

    /// Release one level; releases the lock itself only at the outermost level.
    fn unlock(&self);
}

static HOST_LOCK: OnceLock<Box<dyn HostLock>> = OnceLock::new();

/// Register the host's implementation. Intended to be called once from the host's
/// module-init callback, before any query can run.
pub fn set_host_lock(lock: Box<dyn HostLock>) {
    let already_set = HOST_LOCK.set(lock).is_err();
    debug_assert!(!already_set, "the host lock was registered twice");
}

/// RAII guard for the host lock.
///
/// Bind one *before* taking any lock of your own:
/// `let _host = HostLockGuard::acquire();`
///
/// Not `Send`: the lock is released by `Drop`, and the host releases it on the
/// thread that took it.
#[must_use]
pub struct HostLockGuard {
    lock: Option<&'static dyn HostLock>,
    _not_send: PhantomData<*const ()>,
}

impl HostLockGuard {
    /// Acquire the host lock for the guard's lifetime. A no-op — and harmless —
    /// when no host has registered an implementation (unit tests of the `graph`
    /// crate, where there is no host state to serialise against).
    pub fn acquire() -> Self {
        let lock = HOST_LOCK.get().map(|l| &**l);
        if let Some(lock) = lock {
            lock.lock();
        }
        Self {
            lock,
            _not_send: PhantomData,
        }
    }
}

impl Drop for HostLockGuard {
    fn drop(&mut self) {
        if let Some(lock) = self.lock {
            lock.unlock();
        }
    }
}
