//! A static created on first use.
//!
//! The same shape as [`std::sync::LazyLock`], except that a creation that
//! panics leaves the static uninitialised rather than poisoned: a built-in
//! or user-defined operator read before [`init`](crate::init) panics with
//! its name, and works once `init` has run.

use std::{fmt, ops::Deref, sync::OnceLock};

/// The type of every built-in and every static declared with
/// [`user_ops!`](crate::user_ops). Dereferences to the handle, creating it on
/// the first use.
pub struct Lazy<T> {
    cell: OnceLock<T>,
    create: fn() -> T,
}

impl<T> Lazy<T> {
    #[must_use]
    pub const fn new(create: fn() -> T) -> Self {
        Self {
            cell: OnceLock::new(),
            create,
        }
    }

    /// The value, created now if it was not yet.
    ///
    /// # Panics
    ///
    /// When the creation does; a later call tries again.
    pub fn force(this: &Self) -> &T {
        this.cell.get_or_init(this.create)
    }
}

impl<T> Deref for Lazy<T> {
    type Target = T;

    fn deref(&self) -> &T {
        Self::force(self)
    }
}

impl<T: fmt::Debug> fmt::Debug for Lazy<T> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self.cell.get() {
            Some(value) => value.fmt(f),
            None => f.write_str("<not yet created>"),
        }
    }
}
