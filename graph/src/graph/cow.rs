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

use crate::graph::graphblas::matrix::Dup;

/// Copy-on-Write wrapper that defers duplication until mutation.
///
/// Wraps any type implementing `Dup` and `Clone`. The inner value is
/// shared until `deref_mut` is called, at which point it's duplicated.
#[derive(Clone)]
pub struct Cow<T: Dup<T> + Clone> {
    inner: T,
    /// If true, the next mutable access will duplicate the inner value
    dup: bool,
}

impl<T: Dup<T> + Clone> Cow<T> {
    pub const fn new(inner: T) -> Self {
        Self { inner, dup: false }
    }

    #[must_use]
    pub fn new_version(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            dup: true,
        }
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
        }
        &mut self.inner
    }
}
