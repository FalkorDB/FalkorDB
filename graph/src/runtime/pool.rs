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

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};

/// Per-query object pool for `Vec<T>` buffers.
///
/// Reuses previously allocated `Vec`s to amortise allocation cost
/// across millions of clones in scan/traversal loops.
pub struct Pool<T> {
    // SAFETY invariant: only one mutable reference exists at a time. Enforced
    // by Pool being !Sync (UnsafeCell), each query owning its own Pool, and
    // all access being scoped inside `acquire_raw` / `release` — neither
    // re-enters the other.
    free_vecs: UnsafeCell<Vec<Vec<T>>>,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Pool<T> {
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
        match free.pop() {
            Some(mut v) => {
                v.clear();
                if v.capacity() < capacity {
                    v.reserve(capacity - v.capacity());
                }
                v
            }
            None => Vec::with_capacity(capacity),
        }
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
        v: Vec<T>,
    ) {
        // SAFETY: see struct-level invariant.
        let free = unsafe { &mut *self.free_vecs.get() };
        free.push(v);
    }
}

/// RAII wrapper around a pooled `Vec<T>`.
///
/// When dropped, the buffer is automatically returned to the originating
/// [`Pool`].
pub struct Pooled<'a, T> {
    value: Vec<T>,
    pool: &'a Pool<T>,
}

impl<T> Drop for Pooled<'_, T> {
    #[inline]
    fn drop(&mut self) {
        let v = std::mem::take(&mut self.value);
        self.pool.release(v);
    }
}

impl<T> Deref for Pooled<'_, T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for Pooled<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}
