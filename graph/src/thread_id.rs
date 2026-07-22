//! Redis main-thread identity, used to decide whether GIL acquisition is
//! needed before calling Redis-module FFI from a `Drop`.
//!
//! `RedisModule_ThreadSafeContextLock` is a non-recursive mutex that the Redis
//! main thread holds implicitly during command execution; calling it again
//! from the main thread would deadlock. Off-thread callers (worker threads,
//! lazyfree) must acquire it explicitly.
//!
//! The host crate calls [`set_main_thread`] once from the module-init
//! callback (which Redis runs on the main thread). After that,
//! [`is_main_thread`] tells `Drop` impls which path to take.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, ThreadId};

static MAIN_THREAD_ID: OnceLock<ThreadId> = OnceLock::new();
static PROCESS_IS_CHILD: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// True while *this* thread already holds the Redis module GIL explicitly
    /// (e.g. the write loop's commit/DDL window took it via
    /// `RedisModule_ThreadSafeContextLock`). Because that lock is
    /// non-recursive, RediSearch spec-lifecycle FFI reached under such a window
    /// must NOT re-acquire it — [`gil_held`] lets the `GilGuard` no-op instead
    /// of self-deadlocking. Mirrors the implicit-GIL role of [`is_main_thread`].
    static GIL_HELD: Cell<bool> = const { Cell::new(false) };
}

/// Mark whether the calling (worker) thread currently holds the module GIL.
/// Pair `set_gil_held(true)` immediately after `ThreadSafeContextLock` with
/// `set_gil_held(false)` immediately before `ThreadSafeContextUnlock`.
pub fn set_gil_held(v: bool) {
    GIL_HELD.with(|c| c.set(v));
}

/// Returns true if the calling thread already holds the module GIL (set via
/// [`set_gil_held`]). Used to make a nested `GilGuard` a no-op rather than a
/// recursive (deadlocking) lock.
#[must_use]
pub fn gil_held() -> bool {
    GIL_HELD.with(Cell::get)
}

thread_local! {
    /// Count of outer per-graph locks (L1) held by this thread. Tracked on the
    /// write-queue and bulk-insert paths (the only ones that acquire the module
    /// GIL explicitly) so [`assert_gil_lock_order`] can catch the #726 AB-BA
    /// inversion — acquiring the GIL while still holding L1.
    static L1_HELD: Cell<u32> = const { Cell::new(0) };
}

/// RAII marker that the current thread holds an outer graph lock (L1) for its
/// lifetime. Bind one (`let _l1 = L1HeldScope::new();`) next to each L1 guard on
/// a path that can reach an explicit GIL acquisition, so a future re-ordering
/// that holds L1 across the GIL trips [`assert_gil_lock_order`] deterministically.
#[must_use]
pub struct L1HeldScope(());

impl L1HeldScope {
    pub fn new() -> Self {
        L1_HELD.with(|c| c.set(c.get() + 1));
        Self(())
    }
}

impl Default for L1HeldScope {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for L1HeldScope {
    fn drop(&mut self) {
        L1_HELD.with(|c| c.set(c.get().saturating_sub(1)));
    }
}

/// Debug-only lock-order guard: panics if the calling thread tries to acquire
/// the module GIL while holding an outer graph lock (L1). That is the #726
/// AB-BA deadlock — a background writer holding L1 and waiting for the GIL vs.
/// a main-thread command holding the GIL and waiting for L1. The correct order
/// is GIL → L1 (see `process_write_queued_query`'s two-phase structure). No-op
/// in release builds; catches a re-introduction deterministically, without the
/// deadlock having to actually manifest.
#[inline]
pub fn assert_gil_lock_order() {
    debug_assert_eq!(
        L1_HELD.with(Cell::get),
        0,
        "lock-order violation (#726): acquiring the module GIL while holding the \
         per-graph L1 lock is the AB-BA deadlock; acquire the GIL first, then L1"
    );
}

/// Record the calling thread as the Redis main thread. Idempotent; only the
/// first call wins. Intended to be called once from the module-init callback.
pub fn set_main_thread() {
    let _ = MAIN_THREAD_ID.set(thread::current().id());
}

/// Returns true if the calling thread is the recorded main thread.
///
/// If `set_main_thread` has not been called yet (e.g. unit tests that build
/// `graph` without the module shell), this returns `false` so `Drop` impls
/// take the GIL-acquiring branch — which itself no-ops when the FFI symbols
/// are unresolved.
pub fn is_main_thread() -> bool {
    MAIN_THREAD_ID
        .get()
        .is_some_and(|id| *id == thread::current().id())
}

/// Mark the current process as a fork child. Set from the pthread_atfork
/// CHILD handler so downstream code can detect whether it is running in
/// the original parent or in a forked descendant.
pub fn set_process_is_child(v: bool) {
    PROCESS_IS_CHILD.store(v, Ordering::Relaxed);
}

/// Returns true if this process is a fork child (BGSAVE/ForkGC/etc.).
/// Mirrors the C port's `Globals_Get_ProcessIsChild`.
#[must_use]
pub fn process_is_child() -> bool {
    PROCESS_IS_CHILD.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gil_lock_order_ok_when_no_l1_held() {
        // No L1 guard live → acquiring the GIL is legal (GIL→L1 order).
        assert_gil_lock_order();
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "lock-order violation (#726)")]
    fn gil_lock_order_panics_while_holding_l1() {
        // Simulate the AB-BA inversion: hold L1, then try to take the GIL.
        let _l1 = L1HeldScope::new();
        assert_gil_lock_order();
    }

    #[test]
    fn l1_scope_balances() {
        assert_eq!(L1_HELD.with(Cell::get), 0);
        {
            let _a = L1HeldScope::new();
            let _b = L1HeldScope::new();
            assert_eq!(L1_HELD.with(Cell::get), 2);
        }
        assert_eq!(L1_HELD.with(Cell::get), 0);
    }
}
