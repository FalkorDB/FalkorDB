//! Per-query object pool for `Vec<T>` buffers.
//!
//! [`Pool`] recycles `Vec<T>` buffers to amortize allocation cost across
//! millions of `Env` clones that occur during scan and traversal operators.
//!
//! ```text
//!  Pool lifecycle during a query
//!
//!  acquire()        ┌──────────┐
//!  ─────────────►   │ Pooled   │   (RAII wrapper)
//!                   │ Vec<T>   │
//!                   └────┬─────┘
//!                        │ Drop
//!  release()             ▼
//!  ◄─────────────   free_vecs.push(vec)
//!
//!  Next acquire() pops from free_vecs instead of allocating.
//! ```
//!
//! The pool uses `UnsafeCell` for interior mutability. Each query runs on a
//! single thread and `Pool` is `!Sync`, so there are no concurrent borrows;
//! the `RefCell` runtime flag would just be dead overhead on a per-row hot
//! path.
//!
//! ## Cross-query recycling
//!
//! A fresh [`Pool`] is created per query, but the underlying `Vec` buffers are
//! recycled across queries via a **thread-local** bin (see [`PoolItem`]).
//! Without this, short queries — which never get a chance to amortise within a
//! single run — would pay a full allocate-then-free for every `Env` value
//! buffer on every query. On `Drop`, a pool deposits its idle buffers into the
//! thread-local bin; the next query on the same worker thread drains them on
//! first `acquire`, so steady-state query execution allocates almost no `Env`
//! value buffers at all.

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};

use crate::runtime::value::Value;

/// Maximum number of buffers retained in the thread-local recycle bin.
///
/// Bounds idle memory held between queries so a single large result set cannot
/// pin an unbounded number of buffers on the worker thread.
const RECYCLE_CAP: usize = 1024;

/// Types poolable with cross-query, thread-local buffer recycling.
///
/// The default implementations are no-ops, so an arbitrary `Pool<T>` behaves
/// as a plain per-query pool. [`Value`] overrides them to recycle `Env` value
/// buffers across queries on the same worker thread.
pub trait PoolItem: Sized {
    /// Move all recycled buffers from the thread-local bin into `dst`.
    #[inline]
    fn drain_recycled(_dst: &mut Vec<Vec<Self>>) {}

    /// Deposit idle buffers into the thread-local bin (capped at [`RECYCLE_CAP`]).
    #[inline]
    fn deposit_recycled(_bufs: Vec<Vec<Self>>) {}
}

thread_local! {
    /// Per-worker-thread bin of idle `Vec<Value>` buffers, shared across queries.
    static VALUE_RECYCLE: UnsafeCell<Vec<Vec<Value>>> = const { UnsafeCell::new(Vec::new()) };
}

impl PoolItem for Value {
    #[inline]
    fn drain_recycled(dst: &mut Vec<Vec<Self>>) {
        VALUE_RECYCLE.with(|bin| {
            // SAFETY: thread-local, single-threaded access; `drain_recycled`
            // and `deposit_recycled` never re-enter each other.
            let bin = unsafe { &mut *bin.get() };
            dst.append(bin);
        });
    }

    #[inline]
    fn deposit_recycled(mut bufs: Vec<Vec<Self>>) {
        VALUE_RECYCLE.with(|bin| {
            // SAFETY: see `drain_recycled`.
            let bin = unsafe { &mut *bin.get() };
            let room = RECYCLE_CAP.saturating_sub(bin.len());
            if bufs.len() > room {
                bufs.truncate(room);
            }
            bin.append(&mut bufs);
        });
    }
}

/// Per-query object pool for `Vec<T>` buffers.
///
/// Reuses previously allocated `Vec`s to amortise allocation cost
/// across millions of clones in scan/traversal loops.
pub struct Pool<T: PoolItem> {
    // SAFETY invariant: only one mutable reference exists at a time. Enforced
    // by Pool being !Sync (UnsafeCell), each query owning its own Pool, and
    // all access being scoped inside `acquire_raw` / `release` — neither
    // re-enters the other.
    free_vecs: UnsafeCell<Vec<Vec<T>>>,
}

impl<T: PoolItem> Default for Pool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PoolItem> Pool<T> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            free_vecs: UnsafeCell::new(Vec::new()),
        }
    }

    /// Acquire a raw `Vec<T>` from the pool (or allocate if empty).
    /// Prefer [`acquire`](Self::acquire) which returns a [`Pooled`] handle.
    #[inline]
    pub(crate) fn acquire_raw(
        &self,
        capacity: usize,
    ) -> Vec<T> {
        // SAFETY: see struct-level invariant.
        let free = unsafe { &mut *self.free_vecs.get() };
        if free.is_empty() {
            // Replenish from the thread-local bin populated by prior queries.
            T::drain_recycled(free);
        }
        free.pop().map_or_else(
            || Vec::with_capacity(capacity),
            |mut v| {
                v.clear();
                if v.capacity() < capacity {
                    v.reserve(capacity - v.capacity());
                }
                v
            },
        )
    }

    /// Acquire a [`Pooled`] handle that automatically returns the buffer on drop.
    #[inline]
    pub fn acquire(
        &self,
        capacity: usize,
    ) -> Pooled<'_, T> {
        Pooled {
            value: self.acquire_raw(capacity),
            pool: self,
        }
    }

    /// Return a `Vec<T>` to the pool for later reuse.
    #[inline]
    pub fn release(
        &self,
        mut v: Vec<T>,
    ) {
        // Drop the contained values now so the recycle bin only ever holds
        // empty buffers (no stale values pinned across queries).
        v.clear();
        // SAFETY: see struct-level invariant.
        let free = unsafe { &mut *self.free_vecs.get() };
        free.push(v);
    }
}

impl<T: PoolItem> Drop for Pool<T> {
    fn drop(&mut self) {
        // SAFETY: see struct-level invariant; exclusive access during drop.
        let free = std::mem::take(unsafe { &mut *self.free_vecs.get() });
        if !free.is_empty() {
            T::deposit_recycled(free);
        }
    }
}

/// RAII wrapper around a pooled `Vec<T>`.
///
/// When dropped, the buffer is automatically returned to the originating
/// [`Pool`].
pub struct Pooled<'a, T: PoolItem> {
    value: Vec<T>,
    pool: &'a Pool<T>,
}

impl<T: PoolItem> Drop for Pooled<'_, T> {
    #[inline]
    fn drop(&mut self) {
        let v = std::mem::take(&mut self.value);
        self.pool.release(v);
    }
}

impl<T: PoolItem> Deref for Pooled<'_, T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T: PoolItem> DerefMut for Pooled<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_buffer_within_query() {
        let pool: Pool<Value> = Pool::new();
        let ptr = {
            let mut v = pool.acquire(8);
            v.push(Value::Int(1));
            v.as_ptr()
        };
        // Released on drop; next acquire should hand back the same allocation.
        let v2 = pool.acquire(8);
        assert!(v2.is_empty());
        assert_eq!(v2.as_ptr(), ptr);
    }

    #[test]
    fn recycles_buffers_across_pools_on_same_thread() {
        // Empty the thread-local recycle bin so other tests sharing this
        // worker thread can't leave stray buffers that perturb the assertion.
        let mut drained = Vec::new();
        Value::drain_recycled(&mut drained);
        drop(drained);

        let ptr = {
            let pool: Pool<Value> = Pool::new();
            let v = pool.acquire(16);
            let p = v.as_ptr();
            drop(v);
            // Pool drop deposits the idle buffer into the thread-local bin.
            p
        };

        // A brand-new pool (as created per query) drains the recycled buffer
        // instead of allocating afresh.
        let pool2: Pool<Value> = Pool::new();
        let v = pool2.acquire(16);
        assert!(v.is_empty());
        assert_eq!(v.as_ptr(), ptr);
    }

    #[test]
    fn release_clears_values_before_recycling() {
        let pool: Pool<Value> = Pool::new();
        {
            let mut v = pool.acquire(4);
            v.push(Value::Int(42));
            v.push(Value::Int(7));
        }
        // Reacquired buffer must be logically empty (no stale values pinned).
        let v = pool.acquire(4);
        assert!(v.is_empty());
    }
}
