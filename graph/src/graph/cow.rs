//! Copy-on-Write wrapper for efficient MVCC versioning.
//!
//! This module provides [`Cow`] (Copy-on-Write), a wrapper that defers
//! duplication of data until mutation is needed. This enables efficient
//! MVCC by sharing immutable data between the committed graph snapshot
//! and in-flight write transactions.
//!
//! ## How It Works
//!
//! When `MvccGraph::write()` creates a new graph version, every matrix is
//! wrapped in a `Cow` with `dup = true`. The underlying GraphBLAS handle
//! is shared (cheap clone of the handle pointer). Only when a write
//! operation calls `deref_mut()` does the matrix get physically duplicated
//! via the `Dup` trait.
//!
//! ```text
//!  Graph v1 (committed)          Graph v2 (write tx)
//!  ┌──────────────────┐          ┌──────────────────┐
//!  │ Cow { dup: false }│          │ Cow { dup: true  }│
//!  │ inner: Matrix A   │─ clone ─▶│ inner: Matrix A   │  (same handle)
//!  └──────────────────┘          └──────────────────┘
//!
//!  After v2 calls deref_mut():
//!  ┌──────────────────┐          ┌──────────────────┐
//!  │ Cow { dup: false }│          │ Cow { dup: false }│
//!  │ inner: Matrix A   │          │ inner: Matrix A'  │  (deep copy)
//!  └──────────────────┘          └──────────────────┘
//! ```
//!
//! Read-only transactions never call `deref_mut()`, so they incur zero
//! copying overhead. Only the first mutation in a write transaction
//! triggers a deep copy of the affected matrix.

use std::ops::{Deref, DerefMut};
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicBool, Ordering};

use crate::graph::graphblas::matrix::Dup;

/// Copy-on-Write wrapper that defers duplication until mutation.
///
/// Wraps any type implementing `Dup` and `Clone`. The inner value is
/// shared until `deref_mut` is called, at which point it's duplicated.
///
/// # Contract: isolation is one-directional
///
/// [`Cow::new_version`] marks only the *new* version copy-on-write. The
/// source keeps writing in place into the inner value it now shares, so a
/// write through the source after `new_version` leaks into the new version.
/// Snapshot isolation therefore holds only while a version that has been
/// branched from is never written again, which is the rule `MvccGraph`
/// follows: the committed graph is read-only and every write goes through the
/// version `write()` hands out. Debug builds assert the rule in `deref_mut`;
/// release builds do not check it.
pub struct Cow<T: Dup<T> + Clone> {
    inner: T,
    /// If true, the next mutable access will duplicate the inner value
    dup: bool,
    /// Set once [`Cow::new_version`] has shared `inner` with a new version;
    /// a later in-place write through `self` would leak into it.
    #[cfg(debug_assertions)]
    branched: AtomicBool,
}

impl<T: Dup<T> + Clone> Clone for Cow<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            dup: self.dup,
            #[cfg(debug_assertions)]
            branched: AtomicBool::new(self.branched.load(Ordering::Relaxed)),
        }
    }
}

impl<T: Dup<T> + Clone> Cow<T> {
    pub const fn new(inner: T) -> Self {
        Self {
            inner,
            dup: false,
            #[cfg(debug_assertions)]
            branched: AtomicBool::new(false),
        }
    }

    /// A new version sharing `inner`, copied on its first mutable access.
    /// `self` must not be written afterwards (see the type-level contract).
    #[must_use]
    pub fn new_version(&self) -> Self {
        #[cfg(debug_assertions)]
        self.branched.store(true, Ordering::Relaxed);
        Self {
            inner: self.inner.clone(),
            dup: true,
            #[cfg(debug_assertions)]
            branched: AtomicBool::new(false),
        }
    }

    /// Replace the inner value outright, skipping the copy-on-write dup.
    /// For when the new content is computed from scratch (e.g. a delta fold
    /// building the merged base into a fresh matrix), where deep-copying the
    /// shared inner first would be pure waste.
    pub fn replace(
        &mut self,
        inner: T,
    ) {
        self.inner = inner;
        self.dup = false;
        #[cfg(debug_assertions)]
        self.branched.store(false, Ordering::Relaxed);
    }
}

impl<T: Dup<T> + Clone> Deref for Cow<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: Dup<T> + Clone> DerefMut for Cow<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.dup {
            self.inner = self.inner.dup();
            self.dup = false;
            #[cfg(debug_assertions)]
            self.branched.store(false, Ordering::Relaxed);
        }
        #[cfg(debug_assertions)]
        debug_assert!(
            !self.branched.load(Ordering::Relaxed),
            "Cow: write through a version after new_version() would leak into the new version"
        );
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::*;

    /// A stand-in for `Matrix`: `Clone` shares the handle, `Dup` deep-copies,
    /// and writes take `&mut self` so they go through `DerefMut`.
    #[derive(Clone)]
    struct Handle(Rc<Cell<i32>>);

    impl Handle {
        fn get(&self) -> i32 {
            self.0.get()
        }

        fn set(
            &mut self,
            v: i32,
        ) {
            self.0.set(v);
        }
    }

    impl Dup<Self> for Handle {
        fn dup(&self) -> Self {
            Self(Rc::new(Cell::new(self.0.get())))
        }
    }

    #[test]
    fn new_version_write_is_isolated() {
        let mut v1 = Cow::new(Handle(Rc::new(Cell::new(1))));
        let mut v2 = v1.new_version();
        v2.set(2);
        assert_eq!(v1.get(), 1);
        assert_eq!(v2.get(), 2);
        // `replace` gives the source a fresh inner it shares with nobody, so
        // writing it again is allowed and still isolated.
        v1.replace(Handle(Rc::new(Cell::new(3))));
        v1.set(4);
        assert_eq!(v2.get(), 2);
    }

    /// Isolation is one-directional: writing the source after `new_version`
    /// would leak into the new version, so debug builds refuse it.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "would leak into the new version")]
    fn write_through_branched_source_is_caught() {
        let mut v1 = Cow::new(Handle(Rc::new(Cell::new(1))));
        let v2 = v1.new_version();
        v1.set(99);
        assert_eq!(v2.get(), 1);
    }
}
