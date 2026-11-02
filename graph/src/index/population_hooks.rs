//! Deterministic control over background index population, for tests only.
//!
//! The race `PendingSlots::written` closes depends on how a population batch
//! interleaves with a writer's two steps: publishing its index documents
//! (during the query) and swapping in its graph version (at MVCC commit). A
//! test cannot reach that window from outside, so under the
//! `index-population-hooks` feature this module lets it
//!
//! * shrink the batch size, so a small graph takes many batches;
//! * pause population and release it one batch at a time;
//! * hold a writer between publishing its documents and its commit.
//!
//! Without the feature every function here is an empty `const`-foldable stub:
//! the shipped build has no gate, no counter and no `GRAPH.DEBUG` subcommand
//! (a writer parked in the window would otherwise let a client stall writes).

#[cfg(feature = "index-population-hooks")]
mod imp {
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::time::Duration;

    static BATCH_SIZE: AtomicUsize = AtomicUsize::new(0);
    static PAUSED: AtomicBool = AtomicBool::new(false);
    /// Batches the gate may still let through while paused.
    static TOKENS: AtomicU64 = AtomicU64::new(0);
    /// Batches that ran to completion.
    static BATCHES_DONE: AtomicU64 = AtomicU64::new(0);
    /// Batches the next writer that publishes documents waits for.
    static WINDOW_BATCHES: AtomicU64 = AtomicU64::new(0);
    /// Windows in which every awaited batch ran.
    static WINDOWS_FIRED: AtomicU64 = AtomicU64::new(0);
    /// Windows that gave up waiting (no batch left to run, or population stuck).
    static WINDOWS_TIMED_OUT: AtomicU64 = AtomicU64::new(0);

    pub fn batch_size(default: usize) -> usize {
        match BATCH_SIZE.load(Ordering::Relaxed) {
            0 => default,
            n => n,
        }
    }

    pub fn set_batch_size(n: usize) {
        BATCH_SIZE.store(n, Ordering::Relaxed);
    }

    /// Pauses (`true`) or resumes (`false`) population. Pausing discards
    /// unused tokens.
    pub fn set_paused(paused: bool) {
        TOKENS.store(0, Ordering::SeqCst);
        PAUSED.store(paused, Ordering::SeqCst);
    }

    /// Lets `n` more batches run while paused.
    pub fn step(n: u64) {
        TOKENS.fetch_add(n, Ordering::SeqCst);
    }

    pub fn batches_done() -> u64 {
        BATCHES_DONE.load(Ordering::SeqCst)
    }

    /// `(fired, timed_out)`: windows that saw their batches run, and windows
    /// that gave up. A test asserts the second is zero, so it cannot pass
    /// without having exercised the interleaving.
    pub fn windows() -> (u64, u64) {
        (
            WINDOWS_FIRED.load(Ordering::SeqCst),
            WINDOWS_TIMED_OUT.load(Ordering::SeqCst),
        )
    }

    /// Arms the window: the next writer to publish index documents lets `n`
    /// population batches run, and waits for them, before it returns to its
    /// commit. `0` disarms. The batches read the committed graph, which does
    /// not yet hold the writer's changes.
    pub fn set_window_batches(n: u64) {
        WINDOW_BATCHES.store(n, Ordering::SeqCst);
    }

    /// Called by a population batch before it takes the indexer lock.
    pub fn before_batch() {
        while PAUSED.load(Ordering::SeqCst) {
            let t = TOKENS.load(Ordering::SeqCst);
            if t > 0
                && TOKENS
                    .compare_exchange(t, t - 1, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            {
                return;
            }
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    /// Called by a population batch once it has committed its documents.
    pub fn after_batch() {
        BATCHES_DONE.fetch_add(1, Ordering::SeqCst);
    }

    /// Called by a writer that published index documents, before its graph
    /// version is committed (`published` is false for a commit with no index
    /// documents, which leaves an armed window for the next writer). Runs on
    /// the Redis main thread while the query holds the write lock, so it must
    /// not wait for a command.
    pub fn after_publish(published: bool) {
        if !published {
            return;
        }
        let n = WINDOW_BATCHES.swap(0, Ordering::SeqCst);
        if n == 0 {
            return;
        }
        let target = BATCHES_DONE.load(Ordering::SeqCst) + n;
        step(n);
        // Bounded: population may be finished or paused-and-reset, and a test
        // must fail rather than hang.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while BATCHES_DONE.load(Ordering::SeqCst) < target && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_micros(200));
        }
        if BATCHES_DONE.load(Ordering::SeqCst) >= target {
            WINDOWS_FIRED.fetch_add(1, Ordering::SeqCst);
        } else {
            WINDOWS_TIMED_OUT.fetch_add(1, Ordering::SeqCst);
        }
    }
}

#[cfg(feature = "index-population-hooks")]
pub use imp::*;

#[cfg(not(feature = "index-population-hooks"))]
mod stubs {
    #[inline(always)]
    #[must_use]
    pub const fn batch_size(default: usize) -> usize {
        default
    }

    #[inline(always)]
    pub const fn before_batch() {}

    #[inline(always)]
    pub const fn after_batch() {}

    #[inline(always)]
    pub const fn after_publish(_published: bool) {}
}

#[cfg(not(feature = "index-population-hooks"))]
pub use stubs::*;
