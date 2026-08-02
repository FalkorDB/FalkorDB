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

use std::thread::{self, JoinHandle};

use crossfire::{MRx, MTx, mpmc::Array};
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
    sender: MTx<Array<Job>>,
    size: usize,
}

unsafe impl Sync for ThreadPool {}

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        let mut workers = Vec::with_capacity(size);
        let (sender, receiver): (MTx<Array<Job>>, MRx<Array<Job>>) =
            crossfire::mpmc::bounded_blocking(1024);
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
    pub fn pending_count(&self) -> usize {
        crossfire::BlockingTxTrait::len(&self.sender)
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

/// Get the total number of pending jobs across all worker channels.
pub fn pending_count() -> usize {
    GLOBAL_THREAD_POOL
        .get()
        .expect("Thread pool not initialized")
        .pending_count()
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
