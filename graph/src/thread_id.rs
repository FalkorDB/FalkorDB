//! Thread identity helpers: which thread is the host's main thread, and whether this
//! process is a fork child.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, ThreadId};

static MAIN_THREAD_ID: OnceLock<ThreadId> = OnceLock::new();
static PROCESS_IS_CHILD: AtomicBool = AtomicBool::new(false);

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
