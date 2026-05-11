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

use std::sync::OnceLock;
use std::thread::{self, ThreadId};

static MAIN_THREAD_ID: OnceLock<ThreadId> = OnceLock::new();

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
