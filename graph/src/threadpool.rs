//! Thread pool for parallel query execution.
//!
//! This module provides a global thread pool used to execute queries off
//! the Redis main thread. This prevents long-running queries from blocking
//! Redis command processing.
//!
//! ## Architecture
//!
//! ```text
//! Redis Main Thread                Thread Pool
//!       |                              |
//!   GRAPH.QUERY ───spawn()───>  [shared MPMC queue]
//!       |                       /     |        \
//!       |                  [Worker 1] [Worker 2] [Worker N]
//!   BlockedClient <──────────── result
//! ```
//!
//! ## Scheduling
//!
//! All workers consume from a single shared MPMC channel. Whichever worker
//! is idle picks up the next job, so a long-running job on one worker does
//! not block jobs queued behind it. This avoids the head-of-line blocking
//! that arose with per-worker SPSC queues + round-robin dispatch (e.g. a
//! write-queue drain held its assigned worker, blocking unrelated jobs that
//! happened to be dispatched to the same worker).
//!
//! ## Initialization
//!
//! The pool is stored in a global `OnceCell` and must be initialized once
//! via [`init_thread_pool`] before any calls to [`spawn`].

use std::{
    sync::atomic::{AtomicUsize, Ordering},
    thread::{self, JoinHandle},
};

use crossfire::{MRx, MTx, mpmc::List};
use once_cell::sync::OnceCell;
use parking_lot::Mutex;

/// A closure dispatched to a worker thread. `None` is a sentinel that asks
/// the receiving worker to exit its recv loop so it can be joined.
type Job = Option<Box<dyn FnOnce() + Send + 'static>>;

/// A pool of worker threads for executing jobs.
struct ThreadPool {
    /// Interior mutability only: the pool lives in a `&'static OnceCell`,
    /// so reaching the workers via `&self` from `shutdown` needs a lock
    /// to call `drain(..).join()`. Never touched by `spawn` — the hot
    /// path is lock-free.
    workers: Mutex<Vec<JoinHandle<()>>>,
    sender: MTx<List<Job>>,
    size: usize,
}

unsafe impl Sync for ThreadPool {}

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        let mut workers = Vec::with_capacity(size);
        // Unlimited channel, bounded by admission: `spawn` runs on the Redis
        // main thread, which holds the GIL, so a send that could wait on a full
        // channel deadlocks against every running worker waiting for that same
        // GIL to escalate its write. Queries are counted instead by a
        // [`QueueSlot`], taken before dispatch, which is what bounds this queue.
        let (sender, receiver): (MTx<List<Job>>, MRx<List<Job>>) =
            crossfire::mpmc::unbounded_blocking();
        for _ in 0..size {
            let rx = receiver.clone();
            let worker = thread::spawn(move || {
                while let Ok(job) = rx.recv() {
                    match job {
                        Some(j) => {
                            // Isolate panics: a single panicking job must not
                            // kill the worker thread. A dead worker would shrink
                            // the pool and, once all workers die, make `spawn`
                            // fail its dispatch on the Redis main thread —
                            // unwinding across the extern "C" boundary and
                            // aborting the whole server.
                            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(j)).is_err() {
                                eprintln!("thread pool: job panicked and was contained");
                            }
                        }
                        None => break,
                    }
                }
            });
            workers.push(worker);
        }
        Self {
            workers: Mutex::new(workers),
            sender,
            size,
        }
    }

    pub fn spawn<F>(
        &self,
        job: F,
        _idx: Option<usize>,
    ) where
        F: FnOnce() + Send + 'static,
    {
        if self.sender.send(Some(Box::new(job))).is_err() {
            // The channel only disconnects once every worker has exited, i.e.
            // during pool shutdown. Drop the job and log rather than panicking:
            // on the Redis command path this runs on the main server thread,
            // where a panic would unwind across the extern "C" boundary and
            // abort the process.
            eprintln!("thread pool: dropping job, no workers available to dispatch to");
        }
    }
    /// Drain queued work, signal every worker to exit, and join them.
    ///
    /// Joining matters under sanitizers: pthread TLS destructors only fire
    /// on a clean thread exit (`pthread_exit`/return from the start
    /// routine), not when the process is torn down via `exit(2)`. Without
    /// the join, LSan reports every per-thread allocation as leaked.
    pub fn shutdown(&self) {
        for _ in 0..self.size {
            let _ = self.sender.send(None);
        }
        let mut workers = self.workers.lock();
        for worker in workers.drain(..) {
            let _ = worker.join();
        }
    }
}

static GLOBAL_THREAD_POOL: OnceCell<ThreadPool> = OnceCell::new();

pub fn spawn<F>(
    job: F,
    idx: Option<usize>,
) where
    F: FnOnce() + Send + 'static,
{
    GLOBAL_THREAD_POOL
        .get()
        .expect("Thread pool not initialized")
        .spawn(job, idx);
}

/// Initialize the global thread pool with a specific size.
/// Must be called before any `spawn` calls. Returns `Ok(())` if the pool
/// was successfully initialized, or `Err(())` if it was already initialized.
#[allow(clippy::result_unit_err)]
pub fn init_thread_pool(size: usize) -> Result<(), ()> {
    GLOBAL_THREAD_POOL
        .set(ThreadPool::new(size))
        .map_err(|_| ())
}

/// Drain pending work, signal every worker to exit, and join them.
/// No-op if the pool was never initialized. Intended for module shutdown
/// under sanitizers so per-worker TLS destructors fire.
pub fn shutdown() {
    if let Some(pool) = GLOBAL_THREAD_POOL.get() {
        pool.shutdown();
    }
}

/// Queries admitted but not yet picked up for execution — waiting in the pool
/// queue, or (for a write) anywhere until the graph's write drainer dequeues it.
static QUEUED_QUERIES: AtomicUsize = AtomicUsize::new(0);

/// One admitted, still-queued query's share of [`QUEUED_QUERIES`].
///
/// Taken on the Redis main thread before the query is dispatched, carried with
/// it (inside the pool job, then inside the graph's write message), and dropped
/// once a worker or the write drainer starts executing it. Dropping is the only
/// release, so every path that discards a queued query — a failed send, a pool
/// shutting down, a graph freed with writes still queued, a panicking job —
/// gives its slot back exactly once.
#[must_use]
pub struct QueueSlot {
    counter: &'static AtomicUsize,
}

impl QueueSlot {
    /// Admit one more query unless `max` are already queued. Never waits.
    pub fn try_acquire(max: usize) -> Option<Self> {
        Self::try_acquire_in(&QUEUED_QUERIES, max)
    }

    fn try_acquire_in(
        counter: &'static AtomicUsize,
        max: usize,
    ) -> Option<Self> {
        let mut n = counter.load(Ordering::Acquire);
        while n < max {
            match counter.compare_exchange_weak(n, n + 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(Self { counter }),
                Err(seen) => n = seen,
            }
        }
        None
    }
}

impl Drop for QueueSlot {
    fn drop(&mut self) {
        let prev = self.counter.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(prev > 0, "queued-query counter released below zero");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_slot_bounds_and_releases() {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let slots: Vec<_> = (0..3)
            .map(|_| QueueSlot::try_acquire_in(&COUNTER, 3).expect("under the limit"))
            .collect();
        assert_eq!(COUNTER.load(Ordering::Acquire), 3);
        assert!(QueueSlot::try_acquire_in(&COUNTER, 3).is_none());
        // A rejected admission takes nothing.
        assert_eq!(COUNTER.load(Ordering::Acquire), 3);
        // Lowering the limit below the queued count rejects; raising admits.
        assert!(QueueSlot::try_acquire_in(&COUNTER, 1).is_none());
        let extra = QueueSlot::try_acquire_in(&COUNTER, 4).expect("raised limit");
        drop(extra);
        drop(slots);
        assert_eq!(COUNTER.load(Ordering::Acquire), 0);
        assert!(QueueSlot::try_acquire_in(&COUNTER, 1).is_some());
        assert_eq!(COUNTER.load(Ordering::Acquire), 0);
    }
}
