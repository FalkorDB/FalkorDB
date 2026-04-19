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

use std::thread::{self, JoinHandle};

use crossfire::{BlockingTxTrait, Tx, spsc::Array};
use once_cell::sync::OnceCell;

/// A closure that can be sent to a worker thread.
type Job = Box<dyn FnOnce() + Send + 'static>;

/// A pool of worker threads for executing jobs.
struct ThreadPool {
    workers: Vec<JoinHandle<()>>,
    sender: Vec<Tx<Array<Job>>>,
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
        Self { workers, sender }
    }

    pub fn spawn<F>(
        &self,
        job: F,
        idx: Option<usize>,
    ) where
        F: FnOnce() + Send + 'static,
    {
        let sender = if let Some(i) = idx {
            &self.sender[i % self.workers.len()]
        } else {
            let mut min_tx = &self.sender[0];
            let mut min_len = usize::MAX;
            for tx in &self.sender {
                if tx.is_empty() {
                    return tx
                        .send(Box::new(job))
                        .expect("thread pool worker died: cannot dispatch job");
                }
                let len = tx.len();
                if len < min_len {
                    min_len = len;
                    min_tx = tx;
                }
            }
            min_tx
        };
        sender
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
