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
//!   GRAPH.QUERY ───spawn()───>  [Worker 1] -> execute query
//!       |                       [Worker 2]
//!   (continues)                 [Worker N]
//!       |                              |
//!   BlockedClient <────────────── result
//! ```
//!
//! ## Scheduling
//!
//! Each worker has its own bounded SPSC (single-producer, single-consumer)
//! channel. When a job is dispatched without a specific worker index, the
//! pool picks the worker with the shortest queue (or the first empty one),
//! spreading load across threads. When an explicit index is provided, the
//! job is pinned to that worker (modulo worker count) for thread affinity.
//!
//! ## Initialization
//!
//! The pool is stored in a global `OnceCell` and must be initialized once
//! via [`init_thread_pool`] before any calls to [`spawn`].

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};

use crossfire::{Tx, spsc::Array};
use once_cell::sync::OnceCell;

/// A closure that can be sent to a worker thread.
type Job = Box<dyn FnOnce() + Send + 'static>;

/// A pool of worker threads for executing jobs.
struct ThreadPool {
    workers: Vec<JoinHandle<()>>,
    sender: Vec<Tx<Array<Job>>>,
    /// Round-robin counter for `spawn` calls without explicit affinity.
    next_worker: AtomicUsize,
}

unsafe impl Sync for ThreadPool {}

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        let mut workers = Vec::with_capacity(size);
        let mut sender = Vec::with_capacity(size);
        for _ in 0..size {
            let (tx, rx) = crossfire::spsc::bounded_blocking::<Job>(1024);
            sender.push(tx);
            let worker = thread::spawn(move || {
                while let Ok(job) = rx.recv() {
                    job();
                }
            });
            workers.push(worker);
        }
        Self {
            workers,
            sender,
            next_worker: AtomicUsize::new(0),
        }
    }

    pub fn spawn<F>(
        &self,
        job: F,
        idx: Option<usize>,
    ) where
        F: FnOnce() + Send + 'static,
    {
        let n = self.workers.len();
        let target = idx.map_or_else(
            || self.next_worker.fetch_add(1, Ordering::Relaxed) % n,
            |i| i % n,
        );
        self.sender[target]
            .send(Box::new(job))
            .expect("thread pool worker died: cannot dispatch job");
    }
    pub fn pending_count(&self) -> usize {
        self.sender
            .iter()
            .map(crossfire::BlockingTxTrait::len)
            .sum()
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
