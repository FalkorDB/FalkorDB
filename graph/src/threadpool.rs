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

/// A closure that can be sent to a worker thread.
type Job = Box<dyn FnOnce() + Send + 'static>;

/// A pool of worker threads for executing jobs.
struct ThreadPool {
    _workers: Vec<JoinHandle<()>>,
    sender: MTx<Array<Job>>,
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
                    job();
                }
            });
            workers.push(worker);
        }
        Self {
            _workers: workers,
            sender,
        }
    }

    pub fn spawn<F>(
        &self,
        job: F,
        _idx: Option<usize>,
    ) where
        F: FnOnce() + Send + 'static,
    {
        self.sender
            .send(Box::new(job))
            .expect("thread pool worker died: cannot dispatch job");
    }
    pub fn pending_count(&self) -> usize {
        crossfire::BlockingTxTrait::len(&self.sender)
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
