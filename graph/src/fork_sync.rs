//! Cooperative barrier between graph writers and `fork()`.
//!
//! Redis's BGSAVE forks the server process. The Rust port runs writes on a
//! dedicated thread that mutates GraphBLAS matrices via `Cow<Matrix>` — which
//! transiently swaps the `Arc<GrB_Matrix>` field of a `Matrix` value. If
//! `fork()` runs an atfork prepare handler that calls `Matrix::wait` (to
//! flush pending updates so the snapshot in the child is consistent) while
//! a writer is mid-swap, the wait reads a NULL `GrB_Matrix` and panics.
//!
//! This module provides a flag + counter that lets the fork-prepare handler
//! drain in-flight writers before reading any matrix:
//!
//! - **Writer side**: wrap the matrix-mutation phase in [`GraphOpGuard::new`].
//!   The guard increments [`IN_FLIGHT_GRAPH_OPS`]; if a fork is pending
//!   ([`FORK_PENDING`] = true), the writer waits until the fork clears.
//!
//! - **Fork side**: from `pre_fork_prepare`, call
//!   [`wait_for_fork_drain`]. It sets [`FORK_PENDING`] and blocks until
//!   the counter reaches zero. After fork completes, `after_fork_parent`
//!   must call [`clear_fork_pending`] to release blocked writers.
//!
//! ## Lock-order rule (must not be violated)
//!
//! Writers must drop the `GraphOpGuard` **before** acquiring Redis's
//! thread-safe context ("GIL"). Otherwise the writer holds the counter
//! while waiting for the main thread to drain its event loop, and the
//! main thread is in `wait_for_fork_drain` waiting for the counter — a
//! cycle the kernel will never break.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use parking_lot::{Condvar, Mutex};

static FORK_PENDING: AtomicBool = AtomicBool::new(false);
static IN_FLIGHT_GRAPH_OPS: AtomicUsize = AtomicUsize::new(0);

/// Held while waiting for the fork to clear (writer side).
static FORK_LOCK: Mutex<()> = Mutex::new(());
static FORK_CV: Condvar = Condvar::new();

/// Held while waiting for in-flight writers to drain (fork side).
static DRAIN_LOCK: Mutex<()> = Mutex::new(());
static DRAIN_CV: Condvar = Condvar::new();

fn enter_graph_op() {
    loop {
        // Fast path: optimistically increment, then re-check the flag.
        // If the flag flipped between our load and our increment we back
        // off so the drainer sees us at zero.
        if !FORK_PENDING.load(Ordering::Acquire) {
            IN_FLIGHT_GRAPH_OPS.fetch_add(1, Ordering::AcqRel);
            if !FORK_PENDING.load(Ordering::Acquire) {
                return;
            }
            // Race: fork started after our flag-check. Back out and wait.
            let prev = IN_FLIGHT_GRAPH_OPS.fetch_sub(1, Ordering::AcqRel);
            if prev == 1 {
                let _l = DRAIN_LOCK.lock();
                DRAIN_CV.notify_all();
            }
        }
        let mut g = FORK_LOCK.lock();
        while FORK_PENDING.load(Ordering::Acquire) {
            FORK_CV.wait(&mut g);
        }
    }
}

fn leave_graph_op() {
    let prev = IN_FLIGHT_GRAPH_OPS.fetch_sub(1, Ordering::AcqRel);
    debug_assert!(prev >= 1, "leave without matching enter");
    if prev == 1 && FORK_PENDING.load(Ordering::Acquire) {
        let _l = DRAIN_LOCK.lock();
        DRAIN_CV.notify_all();
    }
}

/// Block until all in-flight graph writers have left their critical
/// sections, then return with `FORK_PENDING` set. New writers will wait
/// for [`clear_fork_pending`] before proceeding.
pub fn wait_for_fork_drain() {
    FORK_PENDING.store(true, Ordering::Release);
    let mut g = DRAIN_LOCK.lock();
    while IN_FLIGHT_GRAPH_OPS.load(Ordering::Acquire) > 0 {
        DRAIN_CV.wait(&mut g);
    }
}

/// Release writers blocked in [`enter_graph_op`].
pub fn clear_fork_pending() {
    FORK_PENDING.store(false, Ordering::Release);
    let _l = FORK_LOCK.lock();
    FORK_CV.notify_all();
}

/// RAII gate around any block of code that may mutate GraphBLAS matrices
/// on the writer thread. Drops [`leave_graph_op`] when out of scope.
///
/// **Important**: drop this guard *before* acquiring Redis's thread-safe
/// context, or you will deadlock with `pre_fork_prepare`. See module docs.
#[must_use = "GraphOpGuard releases on drop; bind it to a name"]
pub struct GraphOpGuard;

impl GraphOpGuard {
    pub fn new() -> Self {
        enter_graph_op();
        Self
    }
}

impl Default for GraphOpGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for GraphOpGuard {
    fn drop(&mut self) {
        leave_graph_op();
    }
}
